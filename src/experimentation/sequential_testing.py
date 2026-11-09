"""O'Brien-Fleming sequential testing and alpha-spending boundary evaluation.

Allows continuous/daily interim monitoring of A/B experiments across $K$ looks
($t_k = k / K \\in (0, 1]$) without inflating the overall Type I error rate $\\alpha$:
$$\\alpha(t_k) = 2 - 2\\Phi\\left(\\frac{Z_{1 - \\alpha/2}}{\\sqrt{t_k}}\\right), \\quad
z_{\\text{boundary}}(t_k) = \\frac{Z_{1 - \\alpha/2}}{\\sqrt{t_k}}$$
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager


@dataclass(frozen=True)
class SequentialTestResult:
    """Result of an O'Brien-Fleming group-sequential evaluation.

    Attributes:
        k_looks: Total number of planned interim looks $K$.
        alpha: Overall two-sided Type I error rate $\\alpha$.
        z_final_critical: Terminal critical value $Z_{1 - \\alpha/2}$ at $t_K = 1.0$.
        stopped_early: ``True`` if an interim look crossed the efficacy or futility boundary.
        stop_look: 1-based look index where stopping occurred (or $K$ if completed).
        information_fraction_at_stop: $t_k = k / K$ at stopping look.
        decision: ``'stop_efficacy'``, ``'stop_futility'``, or ``'continue_to_end'``.
        final_z_stat: Observed cumulative $Z$-statistic at the terminal look $K$.
        schedule: DataFrame with per-look information fraction, $\\alpha(t_k)$, boundaries,
            observed $Z(t_k)$, and interim status.
    """

    k_looks: int
    alpha: float
    z_final_critical: float
    stopped_early: bool
    stop_look: int
    information_fraction_at_stop: float
    decision: str
    final_z_stat: float
    schedule: pd.DataFrame


def obrien_fleming_alpha_spending(
    information_fractions: ArrayLike,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Computes the Lan-DeMets O'Brien-Fleming cumulative $\\alpha$-spending function $\\alpha(t_k)$.

    $$\\alpha(t_k) = 2 - 2\\Phi\\left(\\frac{Z_{1 - \\alpha/2}}{\\sqrt{t_k}}\\right)$$

    Args:
        information_fractions: 1D array-like of information fractions $t_k \\in (0, 1]$.
        alpha: Overall two-sided significance level $\\alpha \\in (0, 1)$.

    Returns:
        1D NumPy array of cumulative $\\alpha(t_k)$ values.

    Raises:
        ValueError: If any $t_k \\notin (0, 1]$ or ``alpha`` is invalid.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    t = np.asarray(information_fractions, dtype=float)
    if t.size == 0 or np.any(t <= 0.0) or np.any(t > 1.0 + 1e-12):
        raise ValueError("information_fractions must lie in (0, 1].")
    t = np.clip(t, 1e-12, 1.0)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return 2.0 - 2.0 * stats.norm.cdf(z_crit / np.sqrt(t))


def obrien_fleming_z_boundaries(
    information_fractions: ArrayLike,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Computes the two-sided O'Brien-Fleming critical $Z$-boundaries $Z_{1-\\alpha/2}/\\sqrt{t_k}$.

    At $t_K = 1.0$, the boundary converges exactly to $Z_{1 - \\alpha/2}$.

    Args:
        information_fractions: 1D array-like of information fractions $t_k \\in (0, 1]$.
        alpha: Overall two-sided significance level $\\alpha$.

    Returns:
        1D NumPy array of positive critical $Z$ thresholds.
    """
    alpha_t = obrien_fleming_alpha_spending(information_fractions, alpha=alpha)
    return np.asarray(stats.norm.ppf(1.0 - alpha_t / 2.0), dtype=float)


def evaluate_sequential_experiment(
    cumulative_z_stats: ArrayLike,
    *,
    alpha: float = 0.05,
    information_fractions: ArrayLike | None = None,
    labels: list[str] | None = None,
    futility_min_fraction: float = 0.50,
    futility_z_floor: float = 0.35,
) -> SequentialTestResult:
    """Evaluates cumulative $Z$-statistics across $K$ interim looks against O'Brien-Fleming bounds.

    Args:
        cumulative_z_stats: Sequence of cumulative two-sample $Z$-statistics across looks $1..K$.
        alpha: Overall two-sided significance level $\\alpha$.
        information_fractions: Optional $t_k \\in (0, 1]$ array; defaults to $k / K$ for $k=1..K$.
        labels: Optional look labels (e.g. dates ``'2017-01-02'``).
        futility_min_fraction: Minimum information fraction $t_k$ before futility stopping is active.
        futility_z_floor: Baseline $|Z|$ floor scaled by $t_k^2$ for early futility detection.

    Returns:
        :class:`SequentialTestResult` containing the stopping decision and boundary schedule.

    Raises:
        ValueError: If ``cumulative_z_stats`` is empty or array lengths mismatch.
    """
    z_obs = np.asarray(cumulative_z_stats, dtype=float)
    k_looks = z_obs.size
    if k_looks < 1:
        raise ValueError("cumulative_z_stats must contain at least 1 look.")

    if information_fractions is None:
        t_frac = np.arange(1, k_looks + 1, dtype=float) / float(k_looks)
    else:
        t_frac = np.asarray(information_fractions, dtype=float)
        if t_frac.size != k_looks:
            raise ValueError("information_fractions must match cumulative_z_stats length.")

    alpha_spent = obrien_fleming_alpha_spending(t_frac, alpha=alpha)
    z_eff = obrien_fleming_z_boundaries(t_frac, alpha=alpha)
    z_fut = np.where(t_frac >= futility_min_fraction, futility_z_floor * (t_frac**2), 0.0)
    z_final_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    stopped_early = False
    stop_look = k_looks
    decision = "continue_to_end"
    statuses: list[str] = []

    for idx in range(k_looks):
        look_num = idx + 1
        abs_z = abs(float(z_obs[idx]))
        if abs_z >= float(z_eff[idx]):
            status = "efficacy_crossed"
            if not stopped_early and look_num < k_looks:
                stopped_early = True
                stop_look = look_num
                decision = "stop_efficacy"
            elif not stopped_early and look_num == k_looks:
                decision = "stop_efficacy"
        elif t_frac[idx] >= futility_min_fraction and abs_z < float(z_fut[idx]) and look_num < k_looks:
            status = "futility_zone"
            if not stopped_early:
                stopped_early = True
                stop_look = look_num
                decision = "stop_futility"
        else:
            status = "continue"
        statuses.append(status)

    look_labels = labels if labels is not None else [f"Look {i + 1}" for i in range(k_looks)]
    schedule = pd.DataFrame(
        {
            "look_index": np.arange(1, k_looks + 1, dtype=int),
            "look_label": look_labels,
            "information_fraction": np.round(t_frac, 4),
            "alpha_spent": np.round(alpha_spent, 6),
            "efficacy_boundary_z": np.round(z_eff, 4),
            "neg_efficacy_boundary_z": np.round(-z_eff, 4),
            "futility_boundary_z": np.round(z_fut, 4),
            "observed_z": np.round(z_obs, 4),
            "status": statuses,
        }
    )

    return SequentialTestResult(
        k_looks=k_looks,
        alpha=alpha,
        z_final_critical=round(z_final_crit, 5),
        stopped_early=stopped_early,
        stop_look=stop_look,
        information_fraction_at_stop=round(float(t_frac[stop_look - 1]), 4),
        decision=decision,
        final_z_stat=round(float(z_obs[-1]), 4),
        schedule=schedule,
    )


def run_sequential_from_daily_logs(
    *,
    metric: str = "post_exp_revenue",
    alpha: float = 0.05,
    manager: DuckDBManager | None = None,
) -> SequentialTestResult:
    """Computes daily cumulative $Z$-statistics from ``v_ab_logs`` and evaluates O'Brien-Fleming bounds.

    Args:
        metric: Column in ``v_ab_logs`` to monitor (``'converted'``, ``'post_exp_revenue'``,
            ``'retention_d7'``, or ``'retention_d30'``).
        alpha: Overall two-sided significance level $\\alpha$.
        manager: Optional :class:`DuckDBManager` override.

    Returns:
        :class:`SequentialTestResult` over the experiment calendar days.

    Raises:
        ValueError: If ``metric`` is not an allowed outcome column.
    """
    allowed = {"converted", "post_exp_revenue", "retention_d7", "retention_d30"}
    if metric not in allowed:
        raise ValueError(f"Unsupported sequential metric {metric!r}; allowed: {sorted(allowed)}")

    db = manager or get_duckdb_manager()
    daily = db.query_df(
        f"""
        SELECT
            assigned_date,
            COUNT(*) FILTER (WHERE experiment_group = 'control') AS n_c,
            SUM({metric}) FILTER (WHERE experiment_group = 'control') AS sum_c,
            SUM(POWER({metric}, 2)) FILTER (WHERE experiment_group = 'control') AS sum_sq_c,
            COUNT(*) FILTER (WHERE experiment_group = 'treatment') AS n_t,
            SUM({metric}) FILTER (WHERE experiment_group = 'treatment') AS sum_t,
            SUM(POWER({metric}, 2)) FILTER (WHERE experiment_group = 'treatment') AS sum_sq_t
        FROM v_ab_logs
        GROUP BY assigned_date
        ORDER BY assigned_date ASC
        """
    )
    cum_nc = daily["n_c"].cumsum().to_numpy(dtype=float)
    cum_nt = daily["n_t"].cumsum().to_numpy(dtype=float)
    cum_sc = daily["sum_c"].cumsum().to_numpy(dtype=float)
    cum_st = daily["sum_t"].cumsum().to_numpy(dtype=float)
    cum_sqc = daily["sum_sq_c"].cumsum().to_numpy(dtype=float)
    cum_sqt = daily["sum_sq_t"].cumsum().to_numpy(dtype=float)

    mean_c = cum_sc / cum_nc
    mean_t = cum_st / cum_nt
    var_c = np.maximum((cum_sqc - cum_nc * (mean_c**2)) / np.maximum(cum_nc - 1.0, 1.0), 1e-12)
    var_t = np.maximum((cum_sqt - cum_nt * (mean_t**2)) / np.maximum(cum_nt - 1.0, 1.0), 1e-12)

    se = np.sqrt(var_c / cum_nc + var_t / cum_nt)
    z_series = [float((mean_t[i] - mean_c[i]) / max(float(se[i]), 1e-12)) for i in range(len(daily))]
    t_frac = (cum_nc + cum_nt) / float(cum_nc[-1] + cum_nt[-1])
    labels = [str(d) for d in daily["assigned_date"]]

    return evaluate_sequential_experiment(
        z_series,
        alpha=alpha,
        information_fractions=t_frac,
        labels=labels,
    )
