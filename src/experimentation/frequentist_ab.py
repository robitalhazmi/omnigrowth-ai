r"""Frequentist A/B testing and CUPED variance reduction estimator.

Implements:
1. Two-sample $Z$-test for binary conversion rates.
2. Welch's two-sample $t$-test for continuous revenue metrics.
3. CUPED (Controlled-experiment Using Pre-Experiment Data) covariate adjustment:
   $$\hat{Y}_i^{\text{CUPED}} = Y_i - \theta (X_i - \bar{X}), \quad \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager


@dataclass(frozen=True)
class FrequentistTestResult:
    """Summary of a Frequentist two-sample hypothesis test."""

    test_name: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float
    standard_error: float
    test_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    alpha: float
    is_significant: bool


@dataclass(frozen=True)
class CUPEDComparisonResult:
    r"""Side-by-side comparison of unadjusted vs. CUPED-adjusted continuous metric inference."""

    unadjusted: FrequentistTestResult
    cuped: FrequentistTestResult
    theta: float
    covariate_mean: float
    correlation_xy: float
    variance_reduction_pct: float
    se_reduction_pct: float


def run_conversion_ztest(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    *,
    alpha: float = 0.05,
) -> FrequentistTestResult:
    """Runs a two-sided two-proportion $Z$-test with unpooled Wald confidence intervals.

    Args:
        control_conversions: Number of conversions in control ($s_C$).
        control_n: Sample size in control ($n_C$).
        treatment_conversions: Number of conversions in treatment ($s_T$).
        treatment_n: Sample size in treatment ($n_T$).
        alpha: Two-sided significance level $\\alpha$.

    Returns:
        :class:`FrequentistTestResult` for the conversion rate difference.

    Raises:
        ValueError: If counts are invalid or sample sizes are non-positive.
    """
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("Sample sizes must be positive.")
    if not (0 <= control_conversions <= control_n and 0 <= treatment_conversions <= treatment_n):
        raise ValueError("Conversions must lie between 0 and n.")

    p_c = control_conversions / control_n
    p_t = treatment_conversions / treatment_n
    lift = p_t - p_c
    rel_lift = 100.0 * lift / p_c if p_c > 0 else 0.0

    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se_pooled = math.sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / control_n + 1.0 / treatment_n), 1e-18))
    se_unpooled = math.sqrt(
        max(p_c * (1.0 - p_c) / control_n + p_t * (1.0 - p_t) / treatment_n, 1e-18)
    )

    z_stat = lift / se_pooled
    p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_low = lift - z_crit * se_unpooled
    ci_high = lift + z_crit * se_unpooled

    return FrequentistTestResult(
        test_name="two_proportion_z",
        n_control=int(control_n),
        n_treatment=int(treatment_n),
        control_mean=round(p_c, 6),
        treatment_mean=round(p_t, 6),
        absolute_lift=round(lift, 6),
        relative_lift_pct=round(rel_lift, 4),
        standard_error=round(se_unpooled, 6),
        test_statistic=round(z_stat, 4),
        p_value=round(p_val, 6),
        ci_lower=round(ci_low, 6),
        ci_upper=round(ci_high, 6),
        alpha=alpha,
        is_significant=bool(p_val < alpha),
    )


def run_welch_ttest(
    control_y: ArrayLike,
    treatment_y: ArrayLike,
    *,
    alpha: float = 0.05,
    test_name: str = "welch_t",
    baseline_mean_override: float | None = None,
) -> FrequentistTestResult:
    """Runs Welch's unequal-variances two-sample $t$-test on continuous arrays.

    Args:
        control_y: 1D array-like of control outcomes.
        treatment_y: 1D array-like of treatment outcomes.
        alpha: Two-sided significance level $\\alpha$.
        test_name: Label stored in the result.
        baseline_mean_override: Optional baseline denominator for ``relative_lift_pct``.

    Returns:
        :class:`FrequentistTestResult`.
    """
    yc = np.asarray(control_y, dtype=float)
    yt = np.asarray(treatment_y, dtype=float)
    nc, nt = yc.size, yt.size
    if nc < 2 or nt < 2:
        raise ValueError("Both control and treatment require at least 2 observations.")

    mean_c = float(np.mean(yc))
    mean_t = float(np.mean(yt))
    var_c = float(np.var(yc, ddof=1))
    var_t = float(np.var(yt, ddof=1))

    lift = mean_t - mean_c
    denom_mean = baseline_mean_override if baseline_mean_override is not None else mean_c
    rel_lift = 100.0 * lift / denom_mean if denom_mean != 0 else 0.0

    se = math.sqrt(max(var_c / nc + var_t / nt, 1e-18))
    t_stat = lift / se

    # Welch-Satterthwaite degrees of freedom
    num_df = (var_c / nc + var_t / nt) ** 2
    den_df = ((var_c / nc) ** 2) / (nc - 1) + ((var_t / nt) ** 2) / (nt - 1)
    dof = num_df / max(den_df, 1e-18)

    p_val = float(2.0 * stats.t.sf(abs(t_stat), df=dof))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=dof))
    ci_low = lift - t_crit * se
    ci_high = lift + t_crit * se

    return FrequentistTestResult(
        test_name=test_name,
        n_control=nc,
        n_treatment=nt,
        control_mean=round(mean_c, 6),
        treatment_mean=round(mean_t, 6),
        absolute_lift=round(lift, 6),
        relative_lift_pct=round(rel_lift, 4),
        standard_error=round(se, 6),
        test_statistic=round(t_stat, 4),
        p_value=round(p_val, 6),
        ci_lower=round(ci_low, 6),
        ci_upper=round(ci_high, 6),
        alpha=alpha,
        is_significant=bool(p_val < alpha),
    )


def run_cuped_analysis(
    control_y: ArrayLike,
    treatment_y: ArrayLike,
    control_x: ArrayLike,
    treatment_x: ArrayLike,
    *,
    alpha: float = 0.05,
) -> CUPEDComparisonResult:
    """Applies CUPED variance reduction using pre-experiment covariate $X$.

    Computes pooled $\\theta = \\text{Cov}(Y, X)/\\text{Var}(X)$ and adjusts every observation:
    $$\\hat{Y}_i^{\\text{CUPED}} = Y_i - \\theta (X_i - \\bar{X})$$

    Args:
        control_y: Control post-experiment outcome $Y_C$.
        treatment_y: Treatment post-experiment outcome $Y_T$.
        control_x: Control pre-experiment covariate $X_C$.
        treatment_x: Treatment pre-experiment covariate $X_T$.
        alpha: Two-sided significance level $\\alpha$.

    Returns:
        :class:`CUPEDComparisonResult` with unadjusted and CUPED-adjusted estimates.

    Raises:
        ValueError: If array lengths do not match or covariate variance is zero.
    """
    yc, yt = np.asarray(control_y, dtype=float), np.asarray(treatment_y, dtype=float)
    xc, xt = np.asarray(control_x, dtype=float), np.asarray(treatment_x, dtype=float)
    if yc.size != xc.size or yt.size != xt.size:
        raise ValueError("Outcome Y and covariate X arrays must have identical lengths per arm.")

    y_all = np.concatenate([yc, yt])
    x_all = np.concatenate([xc, xt])
    x_bar = float(np.mean(x_all))
    var_x = float(np.var(x_all, ddof=1))
    if var_x <= 1e-12:
        raise ValueError("Pre-experiment covariate X has zero variance.")

    cov_xy = float(np.cov(y_all, x_all, ddof=1)[0, 1])
    theta = cov_xy / var_x
    corr_xy = float(np.corrcoef(y_all, x_all)[0, 1])

    yc_cuped = yc - theta * (xc - x_bar)
    yt_cuped = yt - theta * (xt - x_bar)

    unadjusted = run_welch_ttest(yc, yt, alpha=alpha, test_name="welch_t_unadjusted")
    cuped = run_welch_ttest(
        yc_cuped,
        yt_cuped,
        alpha=alpha,
        test_name="cuped_t_adjusted",
        baseline_mean_override=unadjusted.control_mean,
    )

    se_raw, se_adj = unadjusted.standard_error, cuped.standard_error
    var_red_pct = 100.0 * (1.0 - (se_adj**2) / max(se_raw**2, 1e-18))
    se_red_pct = 100.0 * (1.0 - se_adj / max(se_raw, 1e-18))

    return CUPEDComparisonResult(
        unadjusted=unadjusted,
        cuped=cuped,
        theta=round(theta, 6),
        covariate_mean=round(x_bar, 6),
        correlation_xy=round(corr_xy, 6),
        variance_reduction_pct=round(var_red_pct, 2),
        se_reduction_pct=round(se_red_pct, 2),
    )


def analyze_experiment_from_duckdb(
    *,
    alpha: float = 0.05,
    manager: DuckDBManager | None = None,
) -> tuple[FrequentistTestResult, CUPEDComparisonResult]:
    """Runs conversion $Z$-test and revenue CUPED analysis directly on ``v_ab_logs``."""
    db = manager or get_duckdb_manager()
    df = db.query_df(
        "SELECT experiment_group, converted, pre_exp_spend, post_exp_revenue FROM v_ab_logs"
    )
    ctrl = df[df["experiment_group"] == "control"]
    treat = df[df["experiment_group"] == "treatment"]

    conv_res = run_conversion_ztest(
        int(ctrl["converted"].sum()), len(ctrl), int(treat["converted"].sum()), len(treat), alpha=alpha
    )
    cuped_res = run_cuped_analysis(
        ctrl["post_exp_revenue"].to_numpy(),
        treat["post_exp_revenue"].to_numpy(),
        ctrl["pre_exp_spend"].to_numpy(),
        treat["pre_exp_spend"].to_numpy(),
        alpha=alpha,
    )
    return conv_res, cuped_res
