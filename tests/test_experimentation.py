"""Unit tests for Phase 3: Frequentist (CUPED, Sequential) & Bayesian A/B Testing Engine."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from src.experimentation.bayesian_ab import (
    run_bayesian_conversion_test,
    run_bayesian_revenue_test,
)
from src.experimentation.frequentist_ab import (
    analyze_experiment_from_duckdb,
    run_conversion_ztest,
    run_cuped_analysis,
    run_welch_ttest,
)
from src.experimentation.power_analysis import (
    calculate_continuous_sample_size,
    calculate_proportion_sample_size,
    compute_achieved_power_proportions,
    generate_power_curves,
)
from src.experimentation.sequential_testing import (
    evaluate_sequential_experiment,
    obrien_fleming_alpha_spending,
    obrien_fleming_z_boundaries,
    run_sequential_from_daily_logs,
)


# ------------------------------------------------------------------ Power Analysis
def test_proportion_sample_size_and_achieved_power_roundtrip() -> None:
    res = calculate_proportion_sample_size(baseline_rate=0.12, mde=0.01, alpha=0.05, power=0.80)
    assert res.n_per_arm > 10_000
    assert res.total_sample_size == 2 * res.n_per_arm

    achieved = compute_achieved_power_proportions(
        baseline_rate=0.12, mde=0.01, n_per_arm=res.n_per_arm, alpha=0.05
    )
    assert achieved >= 0.80
    assert achieved < 0.805

    higher_power = calculate_proportion_sample_size(baseline_rate=0.12, mde=0.01, alpha=0.05, power=0.90)
    stricter_alpha = calculate_proportion_sample_size(baseline_rate=0.12, mde=0.01, alpha=0.01, power=0.80)
    assert higher_power.n_per_arm > res.n_per_arm
    assert stricter_alpha.n_per_arm > res.n_per_arm


def test_continuous_sample_size_shrinks_by_cuped_r_squared() -> None:
    unadj = calculate_continuous_sample_size(baseline_mean=50.0, baseline_std=25.0, mde=2.0)
    cuped = calculate_continuous_sample_size(
        baseline_mean=50.0, baseline_std=25.0, mde=2.0, cuped_correlation=0.60
    )
    assert cuped.variance_reduction_pct == 36.0
    assert math.isclose(cuped.n_per_arm / unadj.n_per_arm, 0.64, rel_tol=1e-2)


def test_generate_power_curves_monotonic_with_sample_size() -> None:
    df = generate_power_curves(baseline_rate=0.10, mde_values=[0.005, 0.01], sample_sizes=[2000, 10000, 30000])
    assert len(df) == 6
    for _, grp in df.groupby("absolute_mde"):
        assert grp["power"].is_monotonic_increasing


# ------------------------------------------------------- Frequentist & CUPED Engine
def test_conversion_ztest_detects_known_lift() -> None:
    null_res = run_conversion_ztest(1200, 10_000, 1210, 10_000, alpha=0.05)
    assert not null_res.is_significant
    assert null_res.ci_lower < 0.0 < null_res.ci_upper

    sig_res = run_conversion_ztest(1200, 10_000, 1400, 10_000, alpha=0.05)
    assert sig_res.is_significant
    assert sig_res.p_value < 0.001
    assert sig_res.ci_lower > 0.0


def test_cuped_strictly_reduces_standard_error_when_corr_above_0_3() -> None:
    rng = np.random.default_rng(42)
    n = 15_000
    xc = rng.normal(40.0, 12.0, size=n)
    xt = rng.normal(40.0, 12.0, size=n)
    # Construct outcome Y with Corr(X, Y) ~ 0.75 and true treatment lift = +1.50
    yc = 10.0 + 0.85 * xc + rng.normal(0.0, 9.0, size=n)
    yt = 11.5 + 0.85 * xt + rng.normal(0.0, 9.0, size=n)

    res = run_cuped_analysis(yc, yt, xc, xt, alpha=0.05)
    assert res.correlation_xy > 0.30
    assert res.cuped.standard_error < res.unadjusted.standard_error
    assert res.se_reduction_pct > 25.0

    expected_var_red = 100.0 * (res.correlation_xy**2)
    assert math.isclose(res.variance_reduction_pct, expected_var_red, abs_tol=1.5)
    assert math.isclose(res.cuped.absolute_lift, 1.5, abs_tol=0.25)

    cuped_width = res.cuped.ci_upper - res.cuped.ci_lower
    raw_width = res.unadjusted.ci_upper - res.unadjusted.ci_lower
    assert cuped_width < raw_width


def test_welch_ttest_and_duckdb_ab_logs_integration() -> None:
    rng = np.random.default_rng(7)
    res_t = run_welch_ttest(rng.normal(10.0, 2.0, 500), rng.normal(11.0, 2.0, 500))
    assert res_t.is_significant and res_t.absolute_lift > 0.5

    conv_res, cuped_res = analyze_experiment_from_duckdb()
    assert conv_res.n_control + conv_res.n_treatment == 290_584
    assert cuped_res.correlation_xy > 0.30
    assert cuped_res.cuped.standard_error < cuped_res.unadjusted.standard_error
    assert cuped_res.cuped.is_significant


# ---------------------------------------------- O'Brien-Fleming Sequential Testing
def test_obrien_fleming_boundaries_converge_to_z_critical_at_final_look() -> None:
    alpha = 0.05
    t_frac = np.linspace(0.1, 1.0, 10)
    alpha_spent = obrien_fleming_alpha_spending(t_frac, alpha=alpha)
    z_bounds = obrien_fleming_z_boundaries(t_frac, alpha=alpha)

    assert np.all(np.diff(alpha_spent) > 0)
    assert np.all(np.diff(z_bounds) < 0)
    assert math.isclose(float(alpha_spent[-1]), alpha, rel_tol=1e-6)

    expected_z_final = float(stats.norm.ppf(1.0 - alpha / 2.0))
    assert math.isclose(float(z_bounds[-1]), expected_z_final, rel_tol=1e-6)


def test_sequential_evaluation_flags_efficacy_and_futility() -> None:
    efficacy_run = evaluate_sequential_experiment([0.8, 1.5, 3.6, 3.8, 4.0], alpha=0.05)
    assert efficacy_run.stopped_early
    assert efficacy_run.decision == "stop_efficacy"
    assert efficacy_run.stop_look == 3

    futility_run = evaluate_sequential_experiment([0.2, 0.15, 0.05, 0.04, 0.08], alpha=0.05)
    assert futility_run.stopped_early
    assert futility_run.decision == "stop_futility"

    seq_db = run_sequential_from_daily_logs(metric="post_exp_revenue", alpha=0.05)
    assert seq_db.k_looks > 10
    assert math.isclose(seq_db.z_final_critical, float(stats.norm.ppf(0.975)), rel_tol=1e-4)


# ---------------------------------------------------- Analytical Conjugate Bayesian
def test_bayesian_conversion_posterior_probabilities_and_expected_loss() -> None:
    sym = run_bayesian_conversion_test(1200, 10_000, 1200, 10_000)
    assert math.isclose(sym.prob_treatment_beats_control, 0.50, abs_tol=1e-3)
    assert math.isclose(sym.expected_loss_treatment, sym.expected_loss_control, abs_tol=1e-5)
    assert sym.expected_loss_treatment >= 0.0

    win = run_bayesian_conversion_test(1200, 10_000, 1380, 10_000, rope=(-0.002, 0.002))
    assert 0.0 <= win.prob_treatment_beats_control <= 1.0
    assert win.prob_treatment_beats_control > 0.99
    assert win.expected_loss_treatment >= 0.0
    assert win.expected_loss_control > win.expected_loss_treatment
    assert 0.0 <= win.prob_in_rope <= 1.0

    # Fundamental linear-loss identity: E[max(p_B - p_A, 0)] - E[max(p_A - p_B, 0)] == E[p_B - p_A]
    loss_diff = win.expected_loss_control - win.expected_loss_treatment
    assert math.isclose(loss_diff, win.expected_lift, abs_tol=1e-4)
    assert win.control_hdi_95[0] < win.control_posterior_mean < win.control_hdi_95[1]
    assert win.treatment_hdi_95[0] < win.treatment_posterior_mean < win.treatment_hdi_95[1]


def test_bayesian_revenue_normal_inverse_gamma_posteriors() -> None:
    rng = np.random.default_rng(99)
    yc = rng.lognormal(mean=3.2, sigma=0.5, size=4_000)
    yt = yc * 1.08

    res = run_bayesian_revenue_test(yc, yt, rope=(-0.20, 0.20))
    assert 0.0 <= res.prob_treatment_beats_control <= 1.0
    assert res.prob_treatment_beats_control > 0.95
    assert res.expected_loss_treatment >= 0.0
    assert res.expected_loss_control > res.expected_loss_treatment
    assert math.isclose(
        res.expected_loss_control - res.expected_loss_treatment,
        res.expected_lift,
        abs_tol=1e-4,
    )


def test_validation_errors_on_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        calculate_proportion_sample_size(0.10, 0.0, alpha=0.05)
    with pytest.raises(ValueError):
        run_conversion_ztest(10, 0, 10, 100)
    with pytest.raises(ValueError):
        obrien_fleming_alpha_spending([0.0, 0.5])
    with pytest.raises(ValueError):
        run_bayesian_conversion_test(10, 100, 12, 100, rope=(0.01, -0.01))
