r"""Time-series forecasting engine (Meta Prophet & Nixtla AutoARIMA) and online scenario planner.

Follows the offline-training / online-serving architectural split in ``AGENTS.md``:
- Heavy model fitting and rolling-origin cross-validation run offline via ``scripts/train_forecasts.py``.
- Online scenario simulation (:func:`apply_scenario_shock`) applies analytical impulse-response
  adjustments and Bass S-curve parameter shifts in $< 50\text{ ms}$.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager, quote_path
from src.forecasting.s_curve import BassFitResult, shift_bass_scenario

logger = logging.getLogger(__name__)

_Z80_OVER_Z95 = 1.2815515655446004 / 1.959963984540054


@dataclass(frozen=True)
class BacktestMetrics:
    """Rolling-origin cross-validation accuracy and interval calibration metrics."""

    region: str
    model_name: str
    cv_windows: int
    horizon_days: int
    mape: float
    rmse: float
    smape: float
    coverage_80: float
    coverage_95: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes the dataclass to a flat dictionary."""
        return asdict(self)


def enforce_interval_ordering(df: pd.DataFrame) -> pd.DataFrame:
    r"""Enforces non-negativity and strict quantile ordering on forecast bounds."""
    out = df.copy()
    yhat = np.maximum(out["yhat"].to_numpy(dtype=float), 0.0)
    lo80 = np.clip(out["yhat_lower_80"].to_numpy(dtype=float), 0.0, yhat)
    lo95 = np.clip(out["yhat_lower_95"].to_numpy(dtype=float), 0.0, lo80)
    hi80 = np.maximum(out["yhat_upper_80"].to_numpy(dtype=float), yhat)
    hi95 = np.maximum(out["yhat_upper_95"].to_numpy(dtype=float), hi80)

    out["yhat"] = np.round(yhat, 2)
    out["yhat_lower_80"] = np.round(lo80, 2)
    out["yhat_lower_95"] = np.round(lo95, 2)
    out["yhat_upper_80"] = np.round(hi80, 2)
    out["yhat_upper_95"] = np.round(hi95, 2)
    return out


def _assemble_forecast_df(
    region: str,
    model_name: str,
    dates: Any,
    n_hist: int,
    horizon_days: int,
    hist_rev: np.ndarray,
    hist_orders: np.ndarray,
    promo: np.ndarray,
    yhat: np.ndarray,
    lo80: np.ndarray,
    hi80: np.ndarray,
    lo95: np.ndarray,
    hi95: np.ndarray,
) -> pd.DataFrame:
    """Builds a standardized, interval-validated forecast DataFrame with residuals."""
    out = pd.DataFrame(
        {
            "region": region,
            "model_name": model_name,
            "order_date": dates,
            "is_forecast": np.concatenate([np.zeros(n_hist, dtype=bool), np.ones(horizon_days, dtype=bool)]),
            "actual_revenue": np.concatenate([hist_rev, np.full(horizon_days, np.nan)]),
            "actual_orders": np.concatenate([hist_orders, np.full(horizon_days, np.nan)]),
            "promo_discount_pct": np.round(promo, 3),
            "yhat": yhat,
            "yhat_lower_80": lo80,
            "yhat_upper_80": hi80,
            "yhat_lower_95": lo95,
            "yhat_upper_95": hi95,
        }
    )
    out = enforce_interval_ordering(out)
    out["residual"] = np.where(~out["is_forecast"], np.round(out["actual_revenue"] - out["yhat"], 2), np.nan)
    return out


def compute_backtest_metrics(
    region: str,
    model_name: str,
    y_true: np.ndarray,
    yhat: np.ndarray,
    lo80: np.ndarray,
    hi80: np.ndarray,
    lo95: np.ndarray,
    hi95: np.ndarray,
    *,
    cv_windows: int = 3,
    horizon_days: int = 30,
) -> BacktestMetrics:
    """Computes out-of-sample MAPE, RMSE, sMAPE, and 80%/95% empirical coverage."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(yhat, dtype=float)
    err = np.abs(yt - yp)
    denom_mape = np.where(yt > 1.0, yt, np.maximum(np.mean(yt), 1.0))
    mape = float(100.0 * np.mean(err / denom_mape))
    rmse = math.sqrt(float(np.mean((yt - yp) ** 2)))
    smape = float(100.0 * np.mean(2.0 * err / np.maximum(np.abs(yt) + np.abs(yp), 1.0)))
    cov80 = float(100.0 * np.mean((yt >= lo80) & (yt <= hi80)))
    cov95 = float(100.0 * np.mean((yt >= lo95) & (yt <= hi95)))
    return BacktestMetrics(
        region, model_name, cv_windows, horizon_days,
        round(mape, 2), round(rmse, 2), round(smape, 2), round(cov80, 2), round(cov95, 2),
    )


def fit_predict_prophet(
    series_df: pd.DataFrame, *, region: str, horizon_days: int = 90, holiday_country: str = "ID"
) -> pd.DataFrame:
    """Fits Meta Prophet with seasonality, national holidays, and promo discount regressor."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    hist = series_df.sort_values("order_date").reset_index(drop=True)
    train_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(hist["order_date"]),
            "y": hist["gross_revenue"].astype(float),
            "promo_discount_pct": hist["promo_discount_pct"].astype(float),
        }
    )
    model = Prophet(
        interval_width=0.95, weekly_seasonality=True, yearly_seasonality=True,
        daily_seasonality=False, changepoint_prior_scale=0.05, seasonality_mode="additive", uncertainty_samples=250,
    )
    model.add_country_holidays(country_name=holiday_country)
    model.add_regressor("promo_discount_pct")
    model.fit(train_df)

    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    recent_promo = float(train_df["promo_discount_pct"].tail(30).mean())
    future["promo_discount_pct"] = (
        train_df["promo_discount_pct"].reindex(future.index, fill_value=recent_promo).to_numpy()
    )
    pred = model.predict(future)
    yhat = pred["yhat"].to_numpy(dtype=float)
    half_lo95 = np.maximum(yhat - pred["yhat_lower"].to_numpy(dtype=float), 0.0)
    half_hi95 = np.maximum(pred["yhat_upper"].to_numpy(dtype=float) - yhat, 0.0)

    return _assemble_forecast_df(
        region, "prophet", pd.to_datetime(pred["ds"]).dt.date, len(train_df), horizon_days,
        train_df["y"].to_numpy(), hist["daily_orders"].to_numpy(dtype=float),
        future["promo_discount_pct"].to_numpy(dtype=float), yhat,
        yhat - _Z80_OVER_Z95 * half_lo95, yhat + _Z80_OVER_Z95 * half_hi95, yhat - half_lo95, yhat + half_hi95,
    )


def fit_predict_auto_arima(
    series_df: pd.DataFrame, *, region: str, horizon_days: int = 90, season_length: int = 7
) -> pd.DataFrame:
    """Fits Nixtla StatsForecast AutoARIMA and returns in-sample + future interval forecasts."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    hist = series_df.sort_values("order_date").reset_index(drop=True)
    sf_input = pd.DataFrame(
        {"unique_id": region, "ds": pd.to_datetime(hist["order_date"]), "y": hist["gross_revenue"].astype(float)}
    )
    arima = AutoARIMA(season_length=season_length, max_p=3, max_q=3, max_P=1, max_Q=1, approximation=True)
    sf = StatsForecast(models=[arima], freq="D", n_jobs=1)
    fc = sf.forecast(df=sf_input, h=horizon_days, level=[80, 95], fitted=True)
    fitted = sf.forecast_fitted_values()

    in_yhat = fitted["AutoARIMA"].bfill().ffill().to_numpy(dtype=float)
    resid_std = float(np.nanstd(sf_input["y"].to_numpy() - in_yhat))
    in_lo80 = fitted.get("AutoARIMA-lo-80", pd.Series(in_yhat - 1.2816 * resid_std)).bfill().to_numpy()
    in_hi80 = fitted.get("AutoARIMA-hi-80", pd.Series(in_yhat + 1.2816 * resid_std)).bfill().to_numpy()
    in_lo95 = fitted.get("AutoARIMA-lo-95", pd.Series(in_yhat - 1.9600 * resid_std)).bfill().to_numpy()
    in_hi95 = fitted.get("AutoARIMA-hi-95", pd.Series(in_yhat + 1.9600 * resid_std)).bfill().to_numpy()

    recent_promo = float(hist["promo_discount_pct"].tail(30).mean())
    all_dates = pd.concat([sf_input["ds"], pd.to_datetime(fc["ds"])], ignore_index=True).dt.date
    promo_all = np.concatenate([hist["promo_discount_pct"].to_numpy(dtype=float), np.full(horizon_days, recent_promo)])

    return _assemble_forecast_df(
        region, "auto_arima", all_dates, len(hist), horizon_days,
        sf_input["y"].to_numpy(), hist["daily_orders"].to_numpy(dtype=float), promo_all,
        np.concatenate([in_yhat, fc["AutoARIMA"].to_numpy(dtype=float)]),
        np.concatenate([in_lo80, fc["AutoARIMA-lo-80"].to_numpy(dtype=float)]),
        np.concatenate([in_hi80, fc["AutoARIMA-hi-80"].to_numpy(dtype=float)]),
        np.concatenate([in_lo95, fc["AutoARIMA-lo-95"].to_numpy(dtype=float)]),
        np.concatenate([in_hi95, fc["AutoARIMA-hi-95"].to_numpy(dtype=float)]),
    )


def rolling_origin_backtest(
    series_df: pd.DataFrame, *, region: str, horizon_days: int = 30, cv_windows: int = 2, step_days: int = 30
) -> list[BacktestMetrics]:
    """Runs rolling-origin cross-validation for Prophet and AutoARIMA."""
    hist = series_df.sort_values("order_date").reset_index(drop=True)
    n_total = len(hist)
    results: list[BacktestMetrics] = []

    for model_name, fn in (("prophet", fit_predict_prophet), ("auto_arima", fit_predict_auto_arima)):
        yt_all, yp_all, lo80_all, hi80_all, lo95_all, hi95_all = [], [], [], [], [], []
        for w in range(cv_windows):
            cutoff_idx = n_total - horizon_days - w * step_days
            if cutoff_idx < 120:
                continue
            train_slice = hist.iloc[:cutoff_idx]
            test_slice = hist.iloc[cutoff_idx : cutoff_idx + horizon_days]
            fc_only = fn(train_slice, region=region, horizon_days=horizon_days).query("is_forecast")

            yt_all.append(test_slice["gross_revenue"].to_numpy(dtype=float))
            yp_all.append(fc_only["yhat"].to_numpy(dtype=float))
            lo80_all.append(fc_only["yhat_lower_80"].to_numpy(dtype=float))
            hi80_all.append(fc_only["yhat_upper_80"].to_numpy(dtype=float))
            lo95_all.append(fc_only["yhat_lower_95"].to_numpy(dtype=float))
            hi95_all.append(fc_only["yhat_upper_95"].to_numpy(dtype=float))

        results.append(
            compute_backtest_metrics(
                region, model_name, np.concatenate(yt_all), np.concatenate(yp_all),
                np.concatenate(lo80_all), np.concatenate(hi80_all),
                np.concatenate(lo95_all), np.concatenate(hi95_all),
                cv_windows=len(yt_all), horizon_days=horizon_days,
            )
        )
    return results


def apply_scenario_shock(
    base_forecast_df: pd.DataFrame,
    spend_change_pct: float = 0.0,
    promo_discount_pct: float = 0.0,
    conversion_lift_pct: float = 0.0,
    *,
    bass_fit: BassFitResult | None = None,
    spend_elasticity: float = 0.22,
    promo_elasticity: float = 0.018,
) -> pd.DataFrame:
    """Applies analytical marketing spend, promo discount, and conversion shocks in $< 50\\text{ ms}$."""
    out = base_forecast_df.copy()
    fc_mask = out["is_forecast"].to_numpy(dtype=bool) if "is_forecast" in out.columns else np.ones(len(out), dtype=bool)

    spend_mult = 1.0 + spend_elasticity * math.log(max(1.0 + spend_change_pct / 100.0, 0.05))
    has_promo = "promo_discount_pct" in out.columns and bool(np.any(fc_mask))
    base_promo = float(out.loc[fc_mask, "promo_discount_pct"].mean()) if has_promo else 0.0
    promo_delta = promo_discount_pct - base_promo if promo_discount_pct > 0 else 0.0
    promo_mult = max(1.0 + promo_elasticity * promo_delta, 0.20)
    conv_mult = max(1.0 + conversion_lift_pct / 100.0, 0.10)

    total_mult = max(spend_mult * promo_mult * conv_mult, 0.05)
    for col in ("yhat", "yhat_lower_80", "yhat_upper_80", "yhat_lower_95", "yhat_upper_95"):
        if col in out.columns:
            vals = out[col].to_numpy(dtype=float)
            out[col] = np.where(fc_mask, vals * total_mult, vals)

    out = enforce_interval_ordering(out)
    p_base = bass_fit.p_innovation if bass_fit is not None else 0.015
    out.attrs["demand_multiplier"] = round(total_mult, 6)
    out.attrs["bass_p_shifted"] = round(max(p_base * (1.0 + spend_change_pct / 100.0), 1e-7), 6)
    if bass_fit is not None:
        shifted_fit, _ = shift_bass_scenario(
            bass_fit, np.arange(10), spend_change_pct=spend_change_pct, viral_boost_pct=0.5 * conversion_lift_pct
        )
        out.attrs["shifted_bass_fit"] = shifted_fit
    return out


def load_pretrained_forecasts(
    region: str = "ALL", model_name: str | None = None, *, manager: DuckDBManager | None = None
) -> pd.DataFrame:
    """Queries pre-computed forecasts directly from ``forecasts_pretrained.parquet`` via DuckDB."""
    db = manager or get_duckdb_manager()
    src = f"read_parquet({quote_path(get_config().paths.forecasts_pretrained)})"
    if model_name:
        sql = f"SELECT * FROM {src} WHERE region = $region AND model_name = $model ORDER BY order_date"
        return db.query_df(sql, {"region": region, "model": model_name})
    sql = f"SELECT * FROM {src} WHERE region = $region ORDER BY model_name, order_date"
    return db.query_df(sql, {"region": region})
