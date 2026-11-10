"""Unit and integration tests for Module 3 (Time-Series Forecasting & S-Curve Growth)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.config import get_config
from src.forecasting.s_curve import (
    bass_adoption_rate,
    bass_diffusion_curve,
    compare_s_curve_models,
    fit_bass_diffusion,
    fit_logistic_s_curve,
    logistic_curve,
    shift_bass_scenario,
)
from src.forecasting.time_series import (
    apply_scenario_shock,
    compute_backtest_metrics,
    enforce_interval_ordering,
    load_pretrained_forecasts,
)


def test_bass_diffusion_parameter_recovery_within_5pct() -> None:
    """Verifies Bass Diffusion recovers (p=0.03, q=0.38, M=10000) within 5% tolerance."""
    true_p = 0.03
    true_q = 0.38
    true_m = 10_000.0
    t = np.linspace(0.0, 25.0, 51)
    y = bass_diffusion_curve(t, true_p, true_q, true_m)

    fit = fit_bass_diffusion(t, y)
    assert abs(fit.p_innovation - true_p) / true_p < 0.05
    assert abs(fit.q_imitation - true_q) / true_q < 0.05
    assert abs(fit.carrying_capacity_m - true_m) / true_m < 0.05
    assert fit.r_squared > 0.999

    expected_t_star = np.log(true_q / true_p) / (true_p + true_q)
    assert abs(fit.peak_adoption_t_star - expected_t_star) / expected_t_star < 0.05

    rates = bass_adoption_rate(t, true_p, true_q, true_m)
    assert np.all(rates >= 0.0)


def test_logistic_s_curve_parameter_recovery_and_comparison() -> None:
    """Verifies Logistic S-curve recovery and model comparison trajectory generation."""
    true_m, true_k, true_t0 = 8_000.0, 0.25, 16.0
    t = np.arange(0, 35, dtype=float)
    y = logistic_curve(t, true_m, true_k, true_t0)

    fit = fit_logistic_s_curve(t, y)
    assert abs(fit.carrying_capacity_m - true_m) / true_m < 0.05
    assert abs(fit.growth_rate_k - true_k) / true_k < 0.05
    assert abs(fit.inflection_t0 - true_t0) / true_t0 < 0.05
    assert fit.r_squared > 0.999

    log_fit, bass_fit, df = compare_s_curve_models(t, y, horizon_steps=10)
    assert len(df) == len(t) + 10
    assert df["is_forecast"].sum() == 10
    assert log_fit.r_squared > 0.99
    assert bass_fit.r_squared > 0.98


def test_pretrained_forecast_artifacts_exist_and_satisfy_interval_bounds() -> None:
    """Verifies pretrained Parquet artifacts and confidence bound ordering."""
    paths = get_config().paths
    assert paths.forecasts_pretrained.exists()
    assert paths.scurve_params.exists()
    assert paths.model_backtest_metrics.exists()

    fc_df = pd.read_parquet(paths.forecasts_pretrained)
    sc_df = pd.read_parquet(paths.scurve_params)
    bt_df = pd.read_parquet(paths.model_backtest_metrics)

    assert not fc_df.empty
    assert set(fc_df["model_name"].unique()) == {"prophet", "auto_arima"}
    assert {"ALL", "Southeast", "South", "Northeast", "Center-West", "North"} <= set(fc_df["region"].unique())

    # Strict confidence interval ordering: yhat_lower_95 <= yhat_lower_80 <= yhat <= yhat_upper_80 <= yhat_upper_95
    assert (fc_df["yhat_lower_95"] <= fc_df["yhat_lower_80"] + 1e-6).all()
    assert (fc_df["yhat_lower_80"] <= fc_df["yhat"] + 1e-6).all()
    assert (fc_df["yhat"] <= fc_df["yhat_upper_80"] + 1e-6).all()
    assert (fc_df["yhat_upper_80"] <= fc_df["yhat_upper_95"] + 1e-6).all()

    assert len(sc_df) >= 6
    assert (sc_df["bass_m"] > 0).all()
    assert (sc_df["bass_p"] > 0).all()
    assert (sc_df["bass_q"] > 0).all()
    assert (sc_df["bass_r2"] > 0.85).all()

    assert len(bt_df) >= 12
    assert (bt_df["mape"] >= 0).all()
    assert (bt_df["rmse"] >= 0).all()
    assert (bt_df["smape"] >= 0).all()
    assert bt_df["coverage_80"].between(0.0, 100.0).all()
    assert bt_df["coverage_95"].between(0.0, 100.0).all()


def test_scenario_shock_executes_under_50ms_and_shifts_bass_p() -> None:
    """Verifies online scenario simulation runs in < 50 ms and shifts p -> p(1 + delta_spend)."""
    base_df = load_pretrained_forecasts(region="Southeast", model_name="prophet")
    assert not base_df.empty

    t = np.linspace(0, 25, 26)
    bass_fit = fit_bass_diffusion(t, bass_diffusion_curve(t, 0.03, 0.38, 10_000.0))

    started = time.perf_counter()
    shocked_df = apply_scenario_shock(
        base_df,
        spend_change_pct=25.0,
        promo_discount_pct=8.0,
        conversion_lift_pct=10.0,
        bass_fit=bass_fit,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < 50.0, f"Scenario shock took {elapsed_ms:.2f} ms (limit < 50 ms)"

    fc_base = base_df[base_df["is_forecast"]]["yhat"].to_numpy()
    fc_shock = shocked_df[shocked_df["is_forecast"]]["yhat"].to_numpy()
    assert np.all(fc_shock > fc_base)
    assert pytest.approx(shocked_df.attrs["bass_p_shifted"], rel=1e-3) == bass_fit.p_innovation * 1.25

    shifted_fit, shifted_curve = shift_bass_scenario(
        bass_fit, t, spend_change_pct=25.0, viral_boost_pct=10.0, tam_multiplier=1.10
    )
    assert pytest.approx(shifted_fit.p_innovation, rel=1e-3) == bass_fit.p_innovation * 1.25
    assert pytest.approx(shifted_fit.carrying_capacity_m, rel=1e-3) == bass_fit.carrying_capacity_m * 1.10
    assert shifted_curve[-1] > bass_diffusion_curve(t, 0.03, 0.38, 10_000.0)[-1]


def test_enforce_interval_ordering_and_validation_errors() -> None:
    """Verifies interval clamping and input validation on malformed S-curve arrays."""
    raw = pd.DataFrame(
        {
            "yhat": [100.0, -5.0],
            "yhat_lower_80": [110.0, -10.0],
            "yhat_upper_80": [90.0, 5.0],
            "yhat_lower_95": [105.0, -20.0],
            "yhat_upper_95": [95.0, 2.0],
        }
    )
    ordered = enforce_interval_ordering(raw)
    assert (ordered["yhat_lower_95"] <= ordered["yhat_lower_80"]).all()
    assert (ordered["yhat_lower_80"] <= ordered["yhat"]).all()
    assert (ordered["yhat"] <= ordered["yhat_upper_80"]).all()
    assert (ordered["yhat_upper_80"] <= ordered["yhat_upper_95"]).all()

    metrics = compute_backtest_metrics(
        "ALL",
        "prophet",
        np.array([100.0, 120.0, 140.0]),
        np.array([102.0, 118.0, 138.0]),
        np.array([90.0, 110.0, 130.0]),
        np.array([110.0, 130.0, 150.0]),
        np.array([80.0, 100.0, 120.0]),
        np.array([120.0, 140.0, 160.0]),
    )
    assert metrics.coverage_80 == 100.0
    assert metrics.coverage_95 == 100.0

    with pytest.raises(ValueError):
        fit_bass_diffusion([1.0, 2.0], [10.0, 20.0])
    with pytest.raises(ValueError):
        fit_logistic_s_curve([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0])
