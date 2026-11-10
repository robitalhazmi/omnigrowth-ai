r"""Parametric S-Curve market adoption and network saturation modeling.

Implements non-linear least squares curve fitting via :func:`scipy.optimize.curve_fit`
for two canonical technology and product diffusion models:

1. **Logistic Growth Model:** $N(t) = \frac{M}{1 + \exp(-k(t - t_0))}$
2. **Bass Diffusion Model:**
   $$N(t) = M \cdot \frac{1 - \exp(-(p+q)t)}{1 + (q/p)\exp(-(p+q)t)}, \quad t^* = \frac{\ln(q/p)}{p+q}$$
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class LogisticFitResult:
    """Fitted parameters and diagnostics for the Logistic S-Curve growth model."""

    carrying_capacity_m: float
    growth_rate_k: float
    inflection_t0: float
    peak_adoption_t_star: float
    peak_adoption_rate: float
    r_squared: float
    rmse: float
    model_type: str = "logistic"


@dataclass(frozen=True)
class BassFitResult:
    r"""Fitted parameters and diagnostics for the Bass Diffusion S-Curve model ($p, q, M$)."""

    p_innovation: float
    q_imitation: float
    carrying_capacity_m: float
    peak_adoption_t_star: float
    peak_adoption_rate: float
    inflection_adoption_level: float
    r_squared: float
    rmse: float
    model_type: str = "bass"


def logistic_curve(t: ArrayLike, m: float, k: float, t0: float) -> np.ndarray:
    """Evaluates cumulative Logistic adoption $N(t) = M / (1 + \\exp(-k(t - t_0)))$."""
    t_arr = np.asarray(t, dtype=float)
    z = np.clip(-k * (t_arr - t0), -500.0, 500.0)
    return m / (1.0 + np.exp(z))


def logistic_adoption_rate(t: ArrayLike, m: float, k: float, t0: float) -> np.ndarray:
    """Evaluates instantaneous Logistic adoption velocity $dN(t)/dt$."""
    n_t = logistic_curve(t, m, k, t0)
    return k * n_t * (1.0 - n_t / max(m, 1e-12))


def bass_diffusion_curve(t: ArrayLike, p: float, q: float, m: float) -> np.ndarray:
    """Evaluates cumulative Bass Diffusion adoption $N(t) = M \\frac{1 - e^{-(p+q)t}}{1 + (q/p)e^{-(p+q)t}}$."""
    t_arr = np.maximum(np.asarray(t, dtype=float), 0.0)
    p_safe = max(float(p), 1e-9)
    q_safe = max(float(q), 0.0)
    exp_term = np.exp(np.clip(-(p_safe + q_safe) * t_arr, -500.0, 0.0))
    return float(m) * ((1.0 - exp_term) / np.maximum(1.0 + (q_safe / p_safe) * exp_term, 1e-15))


def bass_adoption_rate(t: ArrayLike, p: float, q: float, m: float) -> np.ndarray:
    """Evaluates instantaneous Bass adoption velocity $dN(t)/dt = (p + q N(t)/M)(M - N(t))$."""
    n_t = bass_diffusion_curve(t, p, q, m)
    m_safe = max(float(m), 1e-9)
    return (float(p) + float(q) * (n_t / m_safe)) * np.maximum(m_safe - n_t, 0.0)


def _compute_fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Returns $(R^2, \\text{RMSE})$ between observed and fitted trajectories."""
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12), math.sqrt(float(np.mean(residuals**2)))


def _build_bass_result(p: float, q: float, m: float, r2: float, rmse: float) -> BassFitResult:
    """Constructs a :class:`BassFitResult` with inflection timing $t^*$ and peak rate."""
    if q > p:
        t_star = math.log(q / p) / (p + q)
        peak_rate = m * ((p + q) ** 2) / (4.0 * q)
        inflection_level = float(bass_diffusion_curve(t_star, p, q, m))
    else:
        t_star, peak_rate, inflection_level = 0.0, m * p, 0.0
    return BassFitResult(
        p_innovation=round(p, 6),
        q_imitation=round(q, 6),
        carrying_capacity_m=round(m, 4),
        peak_adoption_t_star=round(t_star, 4),
        peak_adoption_rate=round(peak_rate, 4),
        inflection_adoption_level=round(inflection_level, 4),
        r_squared=round(r2, 6),
        rmse=round(rmse, 4),
    )


def fit_logistic_s_curve(t: ArrayLike, y: ArrayLike) -> LogisticFitResult:
    """Fits a bounded Logistic Growth S-Curve via non-linear least squares."""
    t_arr, y_arr = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t_arr.size < 4 or t_arr.size != y_arr.size:
        raise ValueError("Logistic S-curve fitting requires at least 4 paired (t, y) points.")
    y_max = float(np.max(y_arr))
    if y_max <= 0:
        raise ValueError("Cumulative adoption y must contain positive values.")

    t_min, t_max = float(np.min(t_arr)), float(np.max(t_arr))
    t_span = max(t_max - t_min, 1.0)
    t_mid = float(t_arr[int(np.argmin(np.abs(y_arr - 0.5 * y_max)))])
    bounds_low = [0.50 * y_max, 1e-5 / t_span, t_min - 0.5 * t_span]
    bounds_high = [10.0 * y_max, 50.0 / t_span, t_max + 2.0 * t_span]

    candidates = [
        (1.15 * y_max, 4.0 / t_span, t_mid),
        (1.02 * y_max, 6.0 / t_span, t_mid),
        (1.60 * y_max, 2.5 / t_span, t_min + 0.65 * t_span),
    ]
    best_popt: np.ndarray | None = None
    best_sse = float("inf")
    for cand in candidates:
        p0 = np.clip(cand, bounds_low, bounds_high)
        try:
            popt, _ = curve_fit(
                logistic_curve, t_arr, y_arr, p0=p0, bounds=(bounds_low, bounds_high), method="trf", maxfev=15000
            )
            sse = float(np.sum((y_arr - logistic_curve(t_arr, *popt)) ** 2))
            if sse < best_sse:
                best_sse, best_popt = sse, popt
        except RuntimeError:
            continue

    if best_popt is None:
        best_popt = np.asarray(candidates[0], dtype=float)
    m_hat, k_hat, t0_hat = float(best_popt[0]), float(best_popt[1]), float(best_popt[2])
    r2, rmse = _compute_fit_metrics(y_arr, logistic_curve(t_arr, m_hat, k_hat, t0_hat))

    return LogisticFitResult(
        carrying_capacity_m=round(m_hat, 4),
        growth_rate_k=round(k_hat, 6),
        inflection_t0=round(t0_hat, 4),
        peak_adoption_t_star=round(t0_hat, 4),
        peak_adoption_rate=round(0.25 * m_hat * k_hat, 4),
        r_squared=round(r2, 6),
        rmse=round(rmse, 4),
    )


def fit_bass_diffusion(t: ArrayLike, y: ArrayLike) -> BassFitResult:
    r"""Fits a bounded Bass Diffusion S-Curve model ($p, q, M$) via ``scipy.optimize.curve_fit``."""
    t_arr, y_arr = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t_arr.size < 4 or t_arr.size != y_arr.size:
        raise ValueError("Bass Diffusion fitting requires at least 4 paired (t, y) points.")
    y_max = float(np.max(y_arr))
    if y_max <= 0:
        raise ValueError("Cumulative adoption y must contain positive values.")

    t_span = max(float(np.max(t_arr) - np.min(t_arr)), 1.0)
    bounds_low = [1e-6, 1e-5, 0.75 * y_max]
    bounds_high = [0.95, 5.0, 12.0 * y_max]

    grid_guesses = [
        (0.5 / t_span, 6.0 / t_span, 1.05 * y_max),
        (0.9 / t_span, 11.0 / t_span, 1.02 * y_max),
        (0.3 / t_span, 4.0 / t_span, 1.35 * y_max),
        (0.03, 0.38, 1.05 * y_max),
        (0.01, 0.15, 1.20 * y_max),
    ]
    best_popt: np.ndarray | None = None
    best_sse = float("inf")
    for p0_raw in grid_guesses:
        p0 = np.clip(p0_raw, bounds_low, bounds_high)
        try:
            popt, _ = curve_fit(
                bass_diffusion_curve,
                t_arr,
                y_arr,
                p0=p0,
                bounds=(bounds_low, bounds_high),
                method="trf",
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
                maxfev=25000,
            )
            sse = float(np.sum((y_arr - bass_diffusion_curve(t_arr, *popt)) ** 2))
            if sse < best_sse:
                best_sse, best_popt = sse, popt
        except RuntimeError:
            continue

    if best_popt is None:
        best_popt = np.clip(grid_guesses[0], bounds_low, bounds_high)
    p_hat, q_hat, m_hat = float(best_popt[0]), float(best_popt[1]), float(best_popt[2])
    r2, rmse = _compute_fit_metrics(y_arr, bass_diffusion_curve(t_arr, p_hat, q_hat, m_hat))
    return _build_bass_result(p_hat, q_hat, m_hat, r2, rmse)


def shift_bass_scenario(
    fit: BassFitResult,
    t: ArrayLike,
    *,
    spend_change_pct: float = 0.0,
    viral_boost_pct: float = 0.0,
    tam_multiplier: float = 1.0,
) -> tuple[BassFitResult, np.ndarray]:
    """Applies marketing spend and viral network shocks to a fitted Bass S-Curve in $< 5\\text{ ms}$."""
    p_new = max(fit.p_innovation * (1.0 + spend_change_pct / 100.0), 1e-7)
    q_new = max(fit.q_imitation * (1.0 + viral_boost_pct / 100.0), 1e-7)
    m_new = max(fit.carrying_capacity_m * tam_multiplier, 1.0)
    shifted = _build_bass_result(p_new, q_new, m_new, fit.r_squared, fit.rmse)
    return shifted, bass_diffusion_curve(t, p_new, q_new, m_new)


def compare_s_curve_models(
    t: ArrayLike,
    y: ArrayLike,
    *,
    horizon_steps: int = 90,
) -> tuple[LogisticFitResult, BassFitResult, pd.DataFrame]:
    """Fits both Logistic and Bass Diffusion models and returns historical + future trajectories."""
    t_arr, y_arr = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    log_fit, bass_fit = fit_logistic_s_curve(t_arr, y_arr), fit_bass_diffusion(t_arr, y_arr)

    dt = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else 1.0
    future_t = (
        np.arange(1, horizon_steps + 1, dtype=float) * dt + float(t_arr[-1])
        if horizon_steps > 0
        else np.array([], dtype=float)
    )
    full_t = np.concatenate([t_arr, future_t])
    df = pd.DataFrame(
        {
            "t": full_t,
            "is_forecast": np.concatenate([np.zeros(t_arr.size, dtype=bool), np.ones(future_t.size, dtype=bool)]),
            "actual_cumulative": np.concatenate([y_arr, np.full(future_t.size, np.nan)]),
            "logistic_cumulative": logistic_curve(
                full_t, log_fit.carrying_capacity_m, log_fit.growth_rate_k, log_fit.inflection_t0
            ),
            "logistic_instantaneous": logistic_adoption_rate(
                full_t, log_fit.carrying_capacity_m, log_fit.growth_rate_k, log_fit.inflection_t0
            ),
            "bass_cumulative": bass_diffusion_curve(
                full_t, bass_fit.p_innovation, bass_fit.q_imitation, bass_fit.carrying_capacity_m
            ),
            "bass_instantaneous": bass_adoption_rate(
                full_t, bass_fit.p_innovation, bass_fit.q_imitation, bass_fit.carrying_capacity_m
            ),
        }
    )
    return log_fit, bass_fit, df
