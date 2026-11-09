"""Analytical conjugate Bayesian A/B testing engine (zero MCMC memory overhead).

Implements:
1. Beta-Binomial conjugate posteriors ($\\text{Beta}(\\alpha_0 + s, \\beta_0 + n - s)$) for binary
   conversion rates with exact incomplete-beta Expected Loss, $P(p_B > p_A)$, $ROPE$, and $95\\%$ HDI.
2. Normal-Inverse-Gamma conjugate posteriors for continuous revenue with analytical linear-loss
   integrals, $ROPE$, and $95\\%$ HDI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats


@dataclass(frozen=True)
class BayesianABResult:
    r"""Analytical posterior summary for a Bayesian A/B test.

    Attributes:
        model_type: ``'beta_binomial'`` or ``'normal_inverse_gamma'``.
        control_posterior_mean: Posterior mean of Control ($\mathbb{E}[p_A]$ or $\mathbb{E}[\mu_A]$).
        treatment_posterior_mean: Posterior mean of Treatment ($\mathbb{E}[p_B]$ or $\mathbb{E}[\mu_B]$).
        expected_lift: Posterior mean difference $\mathbb{E}[\theta_B - \theta_A]$.
        relative_lift_pct: Relative posterior lift percentage.
        prob_treatment_beats_control: Exact posterior probability $P(\theta_B > \theta_A) \in [0, 1]$.
        prob_control_beats_treatment: $P(\theta_A > \theta_B) = 1 - P(\theta_B > \theta_A)$.
        expected_loss_treatment: Risk of choosing Treatment $\mathbb{E}[\max(\theta_A - \theta_B, 0)] \ge 0$.
        expected_loss_control: Risk of choosing Control $\mathbb{E}[\max(\theta_B - \theta_A, 0)] \ge 0$.
        rope_bounds: Region of Practical Equivalence $(\delta_L, \delta_U)$ for $\theta_B - \theta_A$.
        prob_in_rope: Posterior probability $P(\theta_B - \theta_A \in [\delta_L, \delta_U])$.
        control_hdi_95: $95\%$ Highest Density Credible Interval for Control.
        treatment_hdi_95: $95\%$ Highest Density Credible Interval for Treatment.
        lift_hdi_95: $95\%$ Highest Density Credible Interval for the lift $\theta_B - \theta_A$.
        posterior_params: Dictionary of conjugate posterior parameters for plotting PDFs.
    """

    model_type: str
    control_posterior_mean: float
    treatment_posterior_mean: float
    expected_lift: float
    relative_lift_pct: float
    prob_treatment_beats_control: float
    prob_control_beats_treatment: float
    expected_loss_treatment: float
    expected_loss_control: float
    rope_bounds: tuple[float, float]
    prob_in_rope: float
    control_hdi_95: tuple[float, float]
    treatment_hdi_95: tuple[float, float]
    lift_hdi_95: tuple[float, float]
    posterior_params: dict[str, float]


def _beta_hdi(alpha_post: float, beta_post: float, cred_mass: float = 0.95) -> tuple[float, float]:
    """Computes the shortest credible interval (HDI) for a Beta(alpha, beta) distribution."""
    if alpha_post <= 1.0 or beta_post <= 1.0:
        tail = 0.5 * (1.0 - cred_mass)
        return (
            float(stats.beta.ppf(tail, alpha_post, beta_post)),
            float(stats.beta.ppf(1.0 - tail, alpha_post, beta_post)),
        )
    tail_grid = np.linspace(1e-5, 1.0 - cred_mass - 1e-5, 401)
    lows = stats.beta.ppf(tail_grid, alpha_post, beta_post)
    highs = stats.beta.ppf(tail_grid + cred_mass, alpha_post, beta_post)
    best = int(np.argmin(highs - lows))
    return float(lows[best]), float(highs[best])


def run_bayesian_conversion_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    rope: tuple[float, float] = (-0.002, 0.002),
    cred_mass: float = 0.95,
) -> BayesianABResult:
    """Runs analytical Beta-Binomial conjugate Bayesian inference for conversion rates.

    Uses exact incomplete-beta reduction for Expected Loss and 1D numerical quadrature over
    the high-density support:
    $$\\mathbb{E}[\\max(p_A - p_B, 0)] = \\int_0^1 \\left[\\mu_A(1 - I_p(\\alpha_A+1, \\beta_A))
      - p(1 - I_p(\\alpha_A, \\beta_A))\\right] f_B(p) \\, dp$$

    Args:
        control_conversions: Observed conversions in Control ($s_A$).
        control_n: Sample size in Control ($n_A$).
        treatment_conversions: Observed conversions in Treatment ($s_B$).
        treatment_n: Sample size in Treatment ($n_B$).
        prior_alpha: Beta prior parameter $\\alpha_0 > 0$ (default ``1.0``).
        prior_beta: Beta prior parameter $\\beta_0 > 0$ (default ``1.0``).
        rope: Region of Practical Equivalence $(\\delta_L, \\delta_U)$ on $p_B - p_A$,
            default ``(-0.002, 0.002)`` ($[-0.2\\%, +0.2\\%]$).
        cred_mass: Credible interval mass (default ``0.95``).

    Returns:
        :class:`BayesianABResult`.

    Raises:
        ValueError: If counts, priors, or ROPE bounds are invalid.
    """
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("Sample sizes must be positive.")
    if not (0 <= control_conversions <= control_n and 0 <= treatment_conversions <= treatment_n):
        raise ValueError("Conversions must lie between 0 and n.")
    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("Beta prior parameters must be positive.")
    if rope[0] > rope[1]:
        raise ValueError(f"ROPE lower bound must be <= upper bound, got {rope}")

    a_c = prior_alpha + float(control_conversions)
    b_c = prior_beta + float(control_n - control_conversions)
    a_t = prior_alpha + float(treatment_conversions)
    b_t = prior_beta + float(treatment_n - treatment_conversions)

    mean_c = a_c / (a_c + b_c)
    mean_t = a_t / (a_t + b_t)
    var_c = (a_c * b_c) / (((a_c + b_c) ** 2) * (a_c + b_c + 1.0))
    var_t = (a_t * b_t) / (((a_t + b_t) ** 2) * (a_t + b_t + 1.0))
    sd_c, sd_t = math.sqrt(var_c), math.sqrt(var_t)

    # Adaptive integration support covering >= 9 standard deviations of both posteriors
    low = max(1e-9, min(mean_c - 9.0 * sd_c, mean_t - 9.0 * sd_t))
    high = min(1.0 - 1e-9, max(mean_c + 9.0 * sd_c, mean_t + 9.0 * sd_t))
    grid = np.linspace(low, high, 2501)

    pdf_c = stats.beta.pdf(grid, a_c, b_c)
    pdf_t = stats.beta.pdf(grid, a_t, b_t)
    cdf_c = stats.beta.cdf(grid, a_c, b_c)

    prob_t_beats_c = float(np.clip(np.trapezoid(pdf_t * cdf_c, grid), 0.0, 1.0))

    # Exact inner integral E_A[max(p_A - p, 0)] integrated against pdf_t(p)
    inner_loss_t = mean_c * (1.0 - special.betainc(a_c + 1.0, b_c, grid)) - grid * (
        1.0 - special.betainc(a_c, b_c, grid)
    )
    loss_t = float(max(0.0, np.trapezoid(np.maximum(inner_loss_t, 0.0) * pdf_t, grid)))

    inner_loss_c = mean_t * (1.0 - special.betainc(a_t + 1.0, b_t, grid)) - grid * (
        1.0 - special.betainc(a_t, b_t, grid)
    )
    loss_c = float(max(0.0, np.trapezoid(np.maximum(inner_loss_c, 0.0) * pdf_c, grid)))

    # ROPE probability P(p_B - p_A in [rope_low, rope_high])
    cdf_t_upper = stats.beta.cdf(np.clip(grid + rope[1], 0.0, 1.0), a_t, b_t)
    cdf_t_lower = stats.beta.cdf(np.clip(grid + rope[0], 0.0, 1.0), a_t, b_t)
    prob_rope = float(np.clip(np.trapezoid(pdf_c * np.maximum(cdf_t_upper - cdf_t_lower, 0.0), grid), 0.0, 1.0))

    hdi_c = _beta_hdi(a_c, b_c, cred_mass=cred_mass)
    hdi_t = _beta_hdi(a_t, b_t, cred_mass=cred_mass)

    lift_mean = mean_t - mean_c
    lift_sd = math.sqrt(var_c + var_t)
    z_crit = float(stats.norm.ppf(0.5 + 0.5 * cred_mass))
    hdi_lift = (round(lift_mean - z_crit * lift_sd, 6), round(lift_mean + z_crit * lift_sd, 6))

    return BayesianABResult(
        model_type="beta_binomial",
        control_posterior_mean=round(mean_c, 6),
        treatment_posterior_mean=round(mean_t, 6),
        expected_lift=round(lift_mean, 6),
        relative_lift_pct=round(100.0 * lift_mean / mean_c if mean_c > 0 else 0.0, 4),
        prob_treatment_beats_control=round(prob_t_beats_c, 6),
        prob_control_beats_treatment=round(1.0 - prob_t_beats_c, 6),
        expected_loss_treatment=round(loss_t, 6),
        expected_loss_control=round(loss_c, 6),
        rope_bounds=(float(rope[0]), float(rope[1])),
        prob_in_rope=round(prob_rope, 6),
        control_hdi_95=(round(hdi_c[0], 6), round(hdi_c[1], 6)),
        treatment_hdi_95=(round(hdi_t[0], 6), round(hdi_t[1], 6)),
        lift_hdi_95=hdi_lift,
        posterior_params={"alpha_c": a_c, "beta_c": b_c, "alpha_t": a_t, "beta_t": b_t},
    )


def run_bayesian_revenue_test(
    control_y: ArrayLike,
    treatment_y: ArrayLike,
    *,
    rope: tuple[float, float] = (-0.25, 0.25),
    cred_mass: float = 0.95,
) -> BayesianABResult:
    """Runs analytical Normal-Inverse-Gamma conjugate Bayesian inference on continuous revenue.

    Updates weak conjugate Normal-Inverse-Gamma priors with sample sufficient statistics,
    yielding marginal Student-$t$ posteriors for $\\mu_A$ and $\\mu_B$.

    Args:
        control_y: 1D array-like of Control revenue observations.
        treatment_y: 1D array-like of Treatment revenue observations.
        rope: Region of Practical Equivalence $(\\delta_L, \\delta_U)$ in currency units.
        cred_mass: Credible interval mass (default ``0.95``).

    Returns:
        :class:`BayesianABResult`.
    """
    yc = np.asarray(control_y, dtype=float)
    yt = np.asarray(treatment_y, dtype=float)
    nc, nt = yc.size, yt.size
    if nc < 2 or nt < 2:
        raise ValueError("Both arms require at least 2 observations.")
    if rope[0] > rope[1]:
        raise ValueError(f"ROPE lower bound must be <= upper bound, got {rope}")

    # Weakly informative Normal-Inverse-Gamma prior (mu0=0, kappa0=1e-6, alpha0=0.5, beta0=1e-6)
    mu0, kappa0, alpha0, beta0 = 0.0, 1e-6, 0.5, 1e-6

    def _nig_update(y: np.ndarray) -> tuple[float, float, float]:
        n = float(y.size)
        y_bar = float(np.mean(y))
        ssd = float(np.sum((y - y_bar) ** 2))
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * y_bar) / kappa_n
        alpha_n = alpha0 + 0.5 * n
        beta_n = beta0 + 0.5 * ssd + 0.5 * (kappa0 * n * ((y_bar - mu0) ** 2)) / kappa_n
        scale = math.sqrt(max(beta_n / (alpha_n * kappa_n), 1e-18))
        dof = 2.0 * alpha_n
        return mu_n, scale, dof

    mu_c, se_c, df_c = _nig_update(yc)
    mu_t, se_t, df_t = _nig_update(yt)

    delta = mu_t - mu_c
    se_diff = math.sqrt(se_c**2 + se_t**2)
    z_ratio = delta / se_diff

    prob_t_beats_c = float(np.clip(stats.norm.cdf(z_ratio), 0.0, 1.0))
    # Exact analytical normal linear-loss formula: E[max(X, 0)] = sigma * phi(mu/sigma) + mu * Phi(mu/sigma)
    loss_t = float(max(0.0, se_diff * stats.norm.pdf(-z_ratio) - delta * stats.norm.cdf(-z_ratio)))
    loss_c = float(max(0.0, se_diff * stats.norm.pdf(z_ratio) + delta * stats.norm.cdf(z_ratio)))

    prob_rope = float(
        np.clip(
            stats.norm.cdf((rope[1] - delta) / se_diff) - stats.norm.cdf((rope[0] - delta) / se_diff),
            0.0,
            1.0,
        )
    )

    q_half = 0.5 + 0.5 * cred_mass
    t_crit_c = float(stats.t.ppf(q_half, df=df_c))
    t_crit_t = float(stats.t.ppf(q_half, df=df_t))
    z_crit = float(stats.norm.ppf(q_half))

    return BayesianABResult(
        model_type="normal_inverse_gamma",
        control_posterior_mean=round(mu_c, 6),
        treatment_posterior_mean=round(mu_t, 6),
        expected_lift=round(delta, 6),
        relative_lift_pct=round(100.0 * delta / mu_c if mu_c != 0 else 0.0, 4),
        prob_treatment_beats_control=round(prob_t_beats_c, 6),
        prob_control_beats_treatment=round(1.0 - prob_t_beats_c, 6),
        expected_loss_treatment=round(loss_t, 6),
        expected_loss_control=round(loss_c, 6),
        rope_bounds=(float(rope[0]), float(rope[1])),
        prob_in_rope=round(prob_rope, 6),
        control_hdi_95=(round(mu_c - t_crit_c * se_c, 6), round(mu_c + t_crit_c * se_c, 6)),
        treatment_hdi_95=(round(mu_t - t_crit_t * se_t, 6), round(mu_t + t_crit_t * se_t, 6)),
        lift_hdi_95=(round(delta - z_crit * se_diff, 6), round(delta + z_crit * se_diff, 6)),
        posterior_params={
            "mu_c": mu_c,
            "scale_c": se_c,
            "df_c": df_c,
            "mu_t": mu_t,
            "scale_t": se_t,
            "df_t": df_t,
        },
    )
