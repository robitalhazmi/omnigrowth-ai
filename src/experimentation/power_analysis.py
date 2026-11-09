"""Statistical power analysis and minimum sample size calculators for A/B experiments.

Implements two-proportion ($Z$-test) and two-sample continuous ($t$/$Z$-test with optional
CUPED variance reduction) sample size calculators and power curve generators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SampleSizeResult:
    """Result of a minimum sample size and statistical power calculation.

    Attributes:
        metric_type: ``'proportion'`` or ``'continuous'``.
        baseline_value: Control baseline rate $p_1$ or mean $\\mu_1$.
        treatment_value: Expected treatment rate $p_2$ or mean $\\mu_2$.
        absolute_mde: Absolute Minimum Detectable Effect ($p_2 - p_1$ or $\\mu_2 - \\mu_1$).
        relative_mde_pct: Relative MDE as a percentage of baseline.
        alpha: Two-sided Type I error rate $\\alpha$.
        target_power: Target statistical power $1 - \\beta$.
        n_per_arm: Minimum required sample size per arm.
        total_sample_size: Total required users across both arms ($2n$).
        variance_reduction_pct: Percentage variance reduction applied (0 if unadjusted).
    """

    metric_type: str
    baseline_value: float
    treatment_value: float
    absolute_mde: float
    relative_mde_pct: float
    alpha: float
    target_power: float
    n_per_arm: int
    total_sample_size: int
    variance_reduction_pct: float = 0.0


def _validate_alpha_and_power(alpha: float, power: float) -> None:
    """Validates significance level alpha and statistical power."""
    if not 0.001 <= alpha <= 0.20:
        raise ValueError(f"alpha must be within [0.001, 0.20], got {alpha}")
    if not 0.50 <= power < 1.0:
        raise ValueError(f"power must be within [0.50, 1.0), got {power}")


def calculate_proportion_sample_size(
    baseline_rate: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    is_relative_mde: bool = False,
) -> SampleSizeResult:
    r"""Computes minimum sample size per arm for a two-proportion conversion A/B test.

    Uses the exact pooled/unpooled normal approximation formula:
    $$n = \frac{(Z_{1-\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + Z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)})^2}{(p_2-p_1)^2}$$

    Args:
        baseline_rate: Control conversion rate $p_1 \\in (0, 1)$.
        mde: Minimum Detectable Effect ($p_2 - p_1$ if absolute, or $(p_2 - p_1)/p_1$ if relative).
        alpha: Two-sided significance level $\\alpha$ (typically in $[0.01, 0.10]$).
        power: Statistical power $1 - \\beta$ (typically in $[0.70, 0.95]$).
        is_relative_mde: If ``True``, interprets ``mde`` as a relative fraction of ``baseline_rate``.

    Returns:
        :class:`SampleSizeResult` containing ``n_per_arm`` and ``total_sample_size``.

    Raises:
        ValueError: If rates, MDE, ``alpha``, or ``power`` are out of bounds.
    """
    _validate_alpha_and_power(alpha, power)
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    abs_mde = baseline_rate * mde if is_relative_mde else mde
    if abs(abs_mde) < 1e-7:
        raise ValueError("mde must be non-zero.")
    p1 = baseline_rate
    p2 = p1 + abs_mde
    if not 0.0 < p2 < 1.0:
        raise ValueError(f"treatment rate p2 = {p2:.4f} must lie strictly inside (0, 1).")

    p_bar = 0.5 * (p1 + p2)
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(stats.norm.ppf(power))

    term_null = z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
    term_alt = z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    n_exact = ((term_null + term_alt) / abs_mde) ** 2
    n_per_arm = max(2, int(math.ceil(n_exact)))

    return SampleSizeResult(
        metric_type="proportion",
        baseline_value=round(p1, 6),
        treatment_value=round(p2, 6),
        absolute_mde=round(abs_mde, 6),
        relative_mde_pct=round(100.0 * abs_mde / p1, 4),
        alpha=alpha,
        target_power=power,
        n_per_arm=n_per_arm,
        total_sample_size=2 * n_per_arm,
        variance_reduction_pct=0.0,
    )


def calculate_continuous_sample_size(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    is_relative_mde: bool = False,
    cuped_correlation: float = 0.0,
) -> SampleSizeResult:
    """Computes minimum sample size per arm for a continuous revenue/metric A/B test.

    When a pre-experiment covariate has correlation $\\rho$ with the outcome, CUPED reduces
    the effective variance by $1 - \\rho^2$, shrinking the required sample size proportionally.

    Args:
        baseline_mean: Baseline control mean $\\mu_1$.
        baseline_std: Baseline standard deviation $\\sigma > 0$.
        mde: Minimum Detectable Effect ($\\mu_2 - \\mu_1$, or relative fraction if ``is_relative_mde``).
        alpha: Two-sided significance level $\\alpha$.
        power: Target statistical power $1 - \\beta$.
        is_relative_mde: If ``True``, interprets ``mde`` as a fraction of ``baseline_mean``.
        cuped_correlation: Pre/post covariate correlation $\\rho \\in [0, 1)$.

    Returns:
        :class:`SampleSizeResult` with required sample sizes.
    """
    _validate_alpha_and_power(alpha, power)
    if baseline_std <= 0.0:
        raise ValueError(f"baseline_std must be > 0, got {baseline_std}")
    if not 0.0 <= abs(cuped_correlation) < 1.0:
        raise ValueError(f"cuped_correlation must be in (-1, 1), got {cuped_correlation}")

    abs_mde = baseline_mean * mde if is_relative_mde else mde
    if abs(abs_mde) < 1e-7:
        raise ValueError("mde must be non-zero.")

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(stats.norm.ppf(power))
    var_factor = 1.0 - (cuped_correlation**2)
    effective_var = (baseline_std**2) * var_factor

    n_exact = 2.0 * effective_var * ((z_alpha + z_beta) ** 2) / (abs_mde**2)
    n_per_arm = max(2, int(math.ceil(n_exact)))
    rel_pct = 100.0 * abs_mde / baseline_mean if baseline_mean != 0 else 0.0

    return SampleSizeResult(
        metric_type="continuous",
        baseline_value=round(baseline_mean, 4),
        treatment_value=round(baseline_mean + abs_mde, 4),
        absolute_mde=round(abs_mde, 4),
        relative_mde_pct=round(rel_pct, 4),
        alpha=alpha,
        target_power=power,
        n_per_arm=n_per_arm,
        total_sample_size=2 * n_per_arm,
        variance_reduction_pct=round(100.0 * (cuped_correlation**2), 2),
    )


def compute_achieved_power_proportions(
    baseline_rate: float,
    mde: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Computes achieved statistical power $1 - \\beta$ for a given sample size per arm.

    Args:
        baseline_rate: Control conversion rate $p_1 \\in (0, 1)$.
        mde: Absolute Minimum Detectable Effect $p_2 - p_1$.
        n_per_arm: Sample size per arm $n \\ge 2$.
        alpha: Two-sided significance level $\\alpha$.

    Returns:
        Achieved power in $[0, 1]$.
    """
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be >= 2.")
    p1 = baseline_rate
    p2 = np.clip(p1 + mde, 1e-6, 1.0 - 1e-6)
    p_bar = 0.5 * (p1 + p2)
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))

    se_null = math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n_per_arm)
    se_alt = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n_per_arm)
    z_score = (abs(p2 - p1) - z_alpha * se_null) / se_alt
    return float(np.clip(stats.norm.cdf(z_score), 0.0, 1.0))


def generate_power_curves(
    baseline_rate: float,
    mde_values: list[float] | tuple[float, ...],
    sample_sizes: list[int] | tuple[int, ...] | None = None,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Generates a tidy DataFrame of statistical power curves across sample sizes and MDEs.

    Args:
        baseline_rate: Control conversion rate $p_1$.
        mde_values: Sequence of absolute MDE thresholds (e.g. ``(0.005, 0.01, 0.015)``).
        sample_sizes: Optional sample sizes per arm; defaults to a 25-point grid up to 100k.
        alpha: Two-sided significance level $\\alpha$.

    Returns:
        DataFrame with columns ``n_per_arm``, ``total_sample_size``, ``absolute_mde``,
        ``relative_mde_pct``, and ``power``.
    """
    if sample_sizes is None:
        sample_sizes = [int(x) for x in np.linspace(1_000, 100_000, 25)]
    rows: list[dict[str, float | int]] = []
    for mde in mde_values:
        rel_pct = round(100.0 * mde / baseline_rate, 2)
        for n in sample_sizes:
            pwr = compute_achieved_power_proportions(baseline_rate, mde, int(n), alpha=alpha)
            rows.append(
                {
                    "n_per_arm": int(n),
                    "total_sample_size": 2 * int(n),
                    "absolute_mde": round(float(mde), 5),
                    "relative_mde_pct": rel_pct,
                    "power": round(pwr, 4),
                }
            )
    return pd.DataFrame(rows)
