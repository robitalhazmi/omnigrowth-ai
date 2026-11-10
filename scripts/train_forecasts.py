#!/usr/bin/env python3
"""Offline model training pipeline for Module 3 (Time-Series & S-Curve Growth).

Trains Meta Prophet, Nixtla StatsForecast AutoARIMA, and Logistic / Bass Diffusion
S-Curve models across all regional cohorts on the local dev machine, exporting
Zstandard-compressed Parquet artifacts to ``data/processed/``:
  1. ``data/processed/forecasts_pretrained.parquet``
  2. ``data/processed/scurve_params.parquet``
  3. ``data/processed/model_backtest_metrics.parquet``
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402
from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager  # noqa: E402
from src.forecasting.s_curve import fit_bass_diffusion, fit_logistic_s_curve  # noqa: E402
from src.forecasting.time_series import (  # noqa: E402
    BacktestMetrics,
    fit_predict_auto_arima,
    fit_predict_prophet,
    rolling_origin_backtest,
)

logger = logging.getLogger("train_forecasts")

REGIONAL_PANEL_SQL = """
WITH state_merchants AS (
    SELECT
        order_date,
        macro_region,
        region_code,
        SUM(daily_orders) AS state_orders,
        SUM(gross_revenue) AS state_revenue,
        SUM(gross_revenue * promo_discount_pct) AS weighted_promo_num,
        MAX(cumulative_active_merchants) AS state_merchants
    FROM v_regional_demand
    GROUP BY order_date, macro_region, region_code
),
macro_daily AS (
    SELECT
        macro_region AS region,
        order_date,
        SUM(state_orders) AS daily_orders,
        ROUND(SUM(state_revenue), 2) AS gross_revenue,
        ROUND(
            COALESCE(SUM(weighted_promo_num) / NULLIF(SUM(state_revenue), 0), 0.0), 4
        ) AS promo_discount_pct,
        SUM(state_merchants) AS cumulative_active_merchants
    FROM state_merchants
    GROUP BY macro_region, order_date
),
national_daily AS (
    SELECT
        'ALL' AS region,
        order_date,
        SUM(daily_orders) AS daily_orders,
        ROUND(SUM(gross_revenue), 2) AS gross_revenue,
        ROUND(
            COALESCE(SUM(gross_revenue * promo_discount_pct) / NULLIF(SUM(gross_revenue), 0), 0.0), 4
        ) AS promo_discount_pct,
        SUM(cumulative_active_merchants) AS cumulative_active_merchants
    FROM macro_daily
    GROUP BY order_date
)
SELECT * FROM national_daily
UNION ALL
SELECT * FROM macro_daily
ORDER BY region ASC, order_date ASC
"""


def fetch_regional_series(manager: DuckDBManager | None = None) -> pd.DataFrame:
    """Loads daily demand and cumulative active merchants per macro-region and national total."""
    db = manager or get_duckdb_manager()
    return db.query_df(REGIONAL_PANEL_SQL)


def train_all_cohorts(
    *,
    horizon_days: int | None = None,
    cv_windows: int = 2,
    holiday_country: str = "ID",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fits Prophet, AutoARIMA, and Bass/Logistic S-curves across all regional cohorts."""
    cfg = get_config()
    h_days = horizon_days or cfg.forecast.horizon_days
    panel_df = fetch_regional_series()

    regions = sorted(panel_df["region"].unique().tolist(), key=lambda r: (r != "ALL", r))
    forecast_frames: list[pd.DataFrame] = []
    scurve_rows: list[dict[str, object]] = []
    backtest_rows: list[BacktestMetrics] = []

    for region in regions:
        t0 = time.perf_counter()
        sub = panel_df[panel_df["region"] == region].sort_values("order_date").reset_index(drop=True)

        # 1. Prophet & AutoARIMA full-history fit + 90-day forecast
        prophet_df = fit_predict_prophet(
            sub, region=region, horizon_days=h_days, holiday_country=holiday_country
        )
        arima_df = fit_predict_auto_arima(
            sub, region=region, horizon_days=h_days, season_length=cfg.forecast.arima.season_length
        )
        forecast_frames.extend([prophet_df, arima_df])

        # 2. Rolling-origin cross-validation metrics
        bt_metrics = rolling_origin_backtest(
            sub,
            region=region,
            horizon_days=min(30, h_days),
            cv_windows=cv_windows,
            step_days=cfg.forecast.cv_step_days,
        )
        backtest_rows.extend(bt_metrics)

        # 3. Parametric S-Curve fitting on cumulative active merchant network adoption
        t_steps = np.arange(1, len(sub) + 1, dtype=float)
        merchants = sub["cumulative_active_merchants"].to_numpy(dtype=float)
        bass_res = fit_bass_diffusion(t_steps, merchants)
        log_res = fit_logistic_s_curve(t_steps, merchants)

        scurve_rows.append(
            {
                "region": region,
                "current_active_merchants": float(merchants[-1]),
                "bass_p": bass_res.p_innovation,
                "bass_q": bass_res.q_imitation,
                "bass_m": bass_res.carrying_capacity_m,
                "bass_t_star": bass_res.peak_adoption_t_star,
                "bass_peak_rate": bass_res.peak_adoption_rate,
                "bass_inflection_level": bass_res.inflection_adoption_level,
                "bass_r2": bass_res.r_squared,
                "bass_rmse": bass_res.rmse,
                "logistic_m": log_res.carrying_capacity_m,
                "logistic_k": log_res.growth_rate_k,
                "logistic_t0": log_res.inflection_t0,
                "logistic_peak_rate": log_res.peak_adoption_rate,
                "logistic_r2": log_res.r_squared,
                "logistic_rmse": log_res.rmse,
                "start_date": str(sub["order_date"].iloc[0]),
                "n_days_observed": int(len(sub)),
            }
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "Cohort %-12s trained in %.2fs | Bass(p=%.4f, q=%.4f, M=%.0f, R2=%.4f)",
            region,
            elapsed,
            bass_res.p_innovation,
            bass_res.q_imitation,
            bass_res.carrying_capacity_m,
            bass_res.r_squared,
        )

    forecasts_all = pd.concat(forecast_frames, ignore_index=True)
    scurve_df = pd.DataFrame(scurve_rows)
    backtest_df = pd.DataFrame([m.to_dict() for m in backtest_rows])

    paths = cfg.paths
    paths.ensure_directories()
    compression = cfg.pipeline.parquet_compression

    forecasts_all.to_parquet(paths.forecasts_pretrained, compression=compression, index=False)
    scurve_df.to_parquet(paths.scurve_params, compression=compression, index=False)
    backtest_df.to_parquet(paths.model_backtest_metrics, compression=compression, index=False)

    logger.info("Saved %s (%d rows)", paths.forecasts_pretrained.name, len(forecasts_all))
    logger.info("Saved %s (%d rows)", paths.scurve_params.name, len(scurve_df))
    logger.info("Saved %s (%d rows)", paths.model_backtest_metrics.name, len(backtest_df))
    return forecasts_all, scurve_df, backtest_df


def main() -> int:
    """CLI entrypoint for offline forecast and S-curve training."""
    parser = argparse.ArgumentParser(description="Train offline Prophet, AutoARIMA, and Bass S-Curve models.")
    parser.add_argument("--horizon", type=int, default=90, help="Forecast horizon in days (default: 90).")
    parser.add_argument("--cv-windows", type=int, default=2, help="Rolling-origin CV windows (default: 2).")
    parser.add_argument("--holiday-country", type=str, default="ID", help="Holiday calendar country code.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    started = time.perf_counter()
    train_all_cohorts(
        horizon_days=args.horizon,
        cv_windows=args.cv_windows,
        holiday_country=args.holiday_country,
    )
    logger.info("All regional forecasts & S-curve models trained in %.2fs.", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
