r"""Prescriptive supply chain and marketing budget allocation solver (Module 4).

Solves the multi-region constrained resource allocation problem via
:func:`scipy.optimize.minimize` (``method='SLSQP'``):

$$\max_{b_r, I_r} \sum_{r=1}^{R} \left[
    m_r \hat{D}_r(b_r) - b_r
    - c_{\text{hold}} \max\left(0, I_r - \hat{D}_r(b_r)\right)
    - c_{\text{short}} \max\left(0, \hat{D}_{r, 0.95}(b_r) - I_r\right)
\right]$$
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import get_config
from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager, quote_path


@dataclass(frozen=True)
class RegionalOptimizationProfile:
    """Economic, demand-forecast, and elasticity parameters for a single region $r$."""

    region: str
    base_demand_units: float
    demand_q80_ratio: float
    demand_q95_ratio: float
    unit_price: float
    gross_margin_pct: float
    elasticity_gamma: float
    spend_scale_b0: float
    holding_cost_per_unit: float
    stockout_penalty_per_unit: float

    @property
    def unit_margin(self) -> float:
        """Net contribution margin $m_r$ per unit sold before marketing & logistics penalties."""
        return self.unit_price * self.gross_margin_pct


@dataclass(frozen=True)
class PrescriptiveOptimizationResult:
    """Summary of baseline vs. SLSQP-optimal budget and inventory allocation."""

    success: bool
    status_message: str
    iterations: int
    allocation_df: pd.DataFrame
    total_budget_limit: float
    total_budget_allocated: float
    total_warehouse_capacity: float
    total_inventory_allocated: float
    baseline_total_revenue: float
    optimal_total_revenue: float
    expected_incremental_revenue: float
    baseline_total_net_margin: float
    optimal_total_net_margin: float
    expected_incremental_margin: float
    stockout_reduction_pct: float
    baseline_roi_pct: float
    optimal_roi_pct: float
    roi_lift_pct: float


DEFAULT_REGIONAL_PROFILES: tuple[RegionalOptimizationProfile, ...] = (
    RegionalOptimizationProfile("Southeast", 6800.0, 1.12, 1.24, 132.0, 0.46, 0.28, 18000.0, 4.20, 26.0),
    RegionalOptimizationProfile("South", 2450.0, 1.14, 1.27, 138.0, 0.44, 0.24, 8500.0, 4.50, 25.0),
    RegionalOptimizationProfile("Northeast", 1750.0, 1.16, 1.31, 155.0, 0.41, 0.31, 6500.0, 5.10, 28.0),
    RegionalOptimizationProfile("Center-West", 1120.0, 1.15, 1.29, 148.0, 0.43, 0.26, 4800.0, 4.90, 26.5),
    RegionalOptimizationProfile("North", 520.0, 1.18, 1.35, 168.0, 0.38, 0.19, 3200.0, 6.20, 24.0),
)


def _demand_at_spend(
    base_d: np.ndarray, gamma: np.ndarray, b0: np.ndarray, spend: np.ndarray
) -> np.ndarray:
    """Computes expected regional demand with diminishing log returns."""
    ratio = np.maximum(spend, 0.0) / np.maximum(b0, 1e-6)
    return base_d * (1.0 + gamma * np.log1p(ratio))


def _demand_grad_spend(
    base_d: np.ndarray, gamma: np.ndarray, b0: np.ndarray, spend: np.ndarray
) -> np.ndarray:
    """Derivative $\\partial \\hat{D}_r(b_r) / \\partial b_r$."""
    return (base_d * gamma) / np.maximum(b0 + np.maximum(spend, 0.0), 1e-6)


def _evaluate_net_margin(
    spend: np.ndarray, inv: np.ndarray, base_d: np.ndarray, q80: np.ndarray, q95: np.ndarray,
    margin_unit: np.ndarray, gamma: np.ndarray, b0: np.ndarray, c_hold: np.ndarray, c_short: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates regional expected demand, 80%/95% quantiles, stockout units, and net margin."""
    d_exp = _demand_at_spend(base_d, gamma, b0, spend)
    d_80, d_95 = q80 * d_exp, q95 * d_exp
    overstock = np.maximum(0.0, inv - d_exp)
    stockout = np.maximum(0.0, d_95 - inv)
    net_margin = margin_unit * d_exp - spend - c_hold * overstock - c_short * stockout
    return d_exp, d_80, d_95, stockout, net_margin


def load_profiles_from_forecasts(
    *, manager: DuckDBManager | None = None
) -> tuple[RegionalOptimizationProfile, ...]:
    """Builds regional optimization profiles calibrated from ``forecasts_pretrained.parquet`` if present."""
    paths = get_config().paths
    if not paths.forecasts_pretrained.exists():
        return DEFAULT_REGIONAL_PROFILES

    db = manager or get_duckdb_manager()
    df = db.query_df(
        f"SELECT region, SUM(yhat) FILTER (WHERE is_forecast) AS fc_rev, "
        f"SUM(yhat_upper_80) FILTER (WHERE is_forecast) AS fc_hi80, "
        f"SUM(yhat_upper_95) FILTER (WHERE is_forecast) AS fc_hi95, "
        f"AVG(actual_revenue / NULLIF(actual_orders, 0)) FILTER (WHERE NOT is_forecast) AS aov "
        f"FROM read_parquet({quote_path(paths.forecasts_pretrained)}) "
        f"WHERE model_name = 'prophet' AND region != 'ALL' GROUP BY region"
    )
    if df.empty:
        return DEFAULT_REGIONAL_PROFILES

    by_reg = {row["region"]: row for _, row in df.iterrows()}
    calibrated: list[RegionalOptimizationProfile] = []
    for prof in DEFAULT_REGIONAL_PROFILES:
        row = by_reg.get(prof.region)
        if row is None or not row["fc_rev"] or row["fc_rev"] <= 0:
            calibrated.append(prof)
            continue
        price = float(row["aov"]) if row["aov"] and row["aov"] > 10 else prof.unit_price
        base_units = max(float(row["fc_rev"]) / price, 50.0)
        q80 = float(np.clip(float(row["fc_hi80"]) / float(row["fc_rev"]), 1.03, 1.45))
        q95 = float(np.clip(float(row["fc_hi95"]) / float(row["fc_rev"]), q80 + 0.03, 1.75))
        calibrated.append(
            RegionalOptimizationProfile(
                prof.region, round(base_units, 2), round(q80, 4), round(q95, 4), round(price, 2),
                prof.gross_margin_pct, prof.elasticity_gamma, prof.spend_scale_b0,
                prof.holding_cost_per_unit, prof.stockout_penalty_per_unit,
            )
        )
    return tuple(calibrated)


def solve_prescriptive_allocation(
    total_budget: float = 100_000.0,
    total_warehouse_inventory: float = 22_000.0,
    *,
    profiles: Sequence[RegionalOptimizationProfile] | None = None,
) -> PrescriptiveOptimizationResult:
    """Solves the constrained regional marketing spend and inventory allocation problem via SLSQP."""
    if total_budget <= 0 or total_warehouse_inventory <= 0:
        raise ValueError("total_budget and total_warehouse_inventory must be positive.")

    reg_profiles = tuple(profiles) if profiles is not None else DEFAULT_REGIONAL_PROFILES
    n_reg = len(reg_profiles)
    if n_reg == 0:
        raise ValueError("At least one RegionalOptimizationProfile is required.")

    base_d = np.array([p.base_demand_units for p in reg_profiles], dtype=float)
    q80 = np.array([p.demand_q80_ratio for p in reg_profiles], dtype=float)
    q95 = np.array([p.demand_q95_ratio for p in reg_profiles], dtype=float)
    price = np.array([p.unit_price for p in reg_profiles], dtype=float)
    margin_unit = np.array([p.unit_margin for p in reg_profiles], dtype=float)
    gamma = np.array([p.elasticity_gamma for p in reg_profiles], dtype=float)
    b0 = np.array([p.spend_scale_b0 for p in reg_profiles], dtype=float)
    c_hold = np.array([p.holding_cost_per_unit for p in reg_profiles], dtype=float)
    c_short = np.array([p.stockout_penalty_per_unit for p in reg_profiles], dtype=float)

    min_floor_zero = float(np.sum(q80 * base_d))
    effective_warehouse = max(float(total_warehouse_inventory), min_floor_zero * 1.05)

    b_base = np.full(n_reg, total_budget / n_reg, dtype=float)
    d80_base = q80 * _demand_at_spend(base_d, gamma, b0, b_base)
    if float(np.sum(d80_base)) > effective_warehouse:
        b_base = b_base * 0.25
        d80_base = q80 * _demand_at_spend(base_d, gamma, b0, b_base)
    slack_inv = max(effective_warehouse - float(np.sum(d80_base)), 0.0)
    i_base = d80_base + slack_inv / n_reg

    d95_init = q95 * _demand_at_spend(base_d, gamma, b0, b_base)
    x0 = np.concatenate([b_base / total_budget, np.minimum(i_base, d95_init) / effective_warehouse])

    def neg_objective(x: np.ndarray) -> float:
        b, inv = x[:n_reg] * total_budget, x[n_reg:] * effective_warehouse
        _, _, _, _, net_m = _evaluate_net_margin(
            b, inv, base_d, q80, q95, margin_unit, gamma, b0, c_hold, c_short
        )
        return -float(np.sum(net_m)) / total_budget

    def neg_objective_jac(x: np.ndarray) -> np.ndarray:
        b, inv = x[:n_reg] * total_budget, x[n_reg:] * effective_warehouse
        d_exp = _demand_at_spend(base_d, gamma, b0, b)
        d_prime = _demand_grad_spend(base_d, gamma, b0, b)
        ind_over, ind_short = (inv >= d_exp).astype(float), (q95 * d_exp >= inv).astype(float)
        grad_b = (margin_unit + c_hold * ind_over - c_short * q95 * ind_short) * d_prime - 1.0
        grad_i = -c_hold * ind_over + c_short * ind_short
        return np.concatenate([-grad_b, -grad_i * (effective_warehouse / total_budget)])

    constraints = [
        {"type": "ineq", "fun": lambda x: 1.0 - float(np.sum(x[:n_reg]))},
        {"type": "ineq", "fun": lambda x: 1.0 - float(np.sum(x[n_reg:]))},
        {
            "type": "ineq",
            "fun": lambda x: (
                x[n_reg:] * effective_warehouse - q80 * _demand_at_spend(base_d, gamma, b0, x[:n_reg] * total_budget)
            )
            / effective_warehouse,
        },
        {
            "type": "ineq",
            "fun": lambda x: (
                q95 * _demand_at_spend(base_d, gamma, b0, x[:n_reg] * total_budget) - x[n_reg:] * effective_warehouse
            )
            / effective_warehouse,
        },
    ]
    res = minimize(
        neg_objective,
        x0,
        method="SLSQP",
        jac=neg_objective_jac,
        bounds=[(0.0, 1.0)] * (2 * n_reg),
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-10},
    )

    b_opt = np.maximum(res.x[:n_reg] * total_budget, 0.0)
    if float(np.sum(b_opt)) > total_budget:
        b_opt *= total_budget / float(np.sum(b_opt))
    d80_opt = q80 * _demand_at_spend(base_d, gamma, b0, b_opt)
    i_opt = np.maximum(res.x[n_reg:] * effective_warehouse, d80_opt)

    d_base_u, _, _, short_base, nm_base = _evaluate_net_margin(
        b_base, i_base, base_d, q80, q95, margin_unit, gamma, b0, c_hold, c_short
    )
    d_opt_u, d80_opt_u, d95_opt_u, short_opt, nm_opt = _evaluate_net_margin(
        b_opt, i_opt, base_d, q80, q95, margin_unit, gamma, b0, c_hold, c_short
    )

    rev_base, rev_opt = float(np.sum(price * d_base_u)), float(np.sum(price * d_opt_u))
    tot_nm_base, tot_nm_opt = float(np.sum(nm_base)), float(np.sum(nm_opt))
    sum_short_base = float(np.sum(short_base))
    short_red = (
        100.0 * (1.0 - float(np.sum(short_opt)) / max(sum_short_base, 1e-6)) if sum_short_base > 0 else 0.0
    )
    roi_base = 100.0 * tot_nm_base / max(float(np.sum(b_base)), 1.0)
    roi_opt = 100.0 * tot_nm_opt / max(float(np.sum(b_opt)), 1.0)

    alloc_df = pd.DataFrame(
        {
            "region": [p.region for p in reg_profiles],
            "unit_price": price,
            "margin_rate": [p.gross_margin_pct for p in reg_profiles],
            "elasticity_gamma": gamma,
            "baseline_spend": np.round(b_base, 2),
            "optimal_spend": np.round(b_opt, 2),
            "spend_delta_pct": np.round(100.0 * (b_opt - b_base) / np.maximum(b_base, 1.0), 2),
            "baseline_demand": np.round(d_base_u, 2),
            "optimal_demand": np.round(d_opt_u, 2),
            "safety_stock_80": np.round(d80_opt_u, 2),
            "demand_upper_95": np.round(d95_opt_u, 2),
            "baseline_inventory": np.round(i_base, 2),
            "optimal_inventory": np.round(i_opt, 2),
            "inventory_delta_pct": np.round(100.0 * (i_opt - i_base) / np.maximum(i_base, 1.0), 2),
            "baseline_stockout_units": np.round(short_base, 2),
            "optimal_stockout_units": np.round(short_opt, 2),
            "baseline_net_margin": np.round(nm_base, 2),
            "optimal_net_margin": np.round(nm_opt, 2),
            "incremental_margin": np.round(nm_opt - nm_base, 2),
        }
    )

    return PrescriptiveOptimizationResult(
        success=bool(res.success),
        status_message=str(res.message),
        iterations=int(getattr(res, "nit", 0)),
        allocation_df=alloc_df,
        total_budget_limit=round(total_budget, 2),
        total_budget_allocated=round(float(np.sum(b_opt)), 2),
        total_warehouse_capacity=round(effective_warehouse, 2),
        total_inventory_allocated=round(float(np.sum(i_opt)), 2),
        baseline_total_revenue=round(rev_base, 2),
        optimal_total_revenue=round(rev_opt, 2),
        expected_incremental_revenue=round(rev_opt - rev_base, 2),
        baseline_total_net_margin=round(tot_nm_base, 2),
        optimal_total_net_margin=round(tot_nm_opt, 2),
        expected_incremental_margin=round(tot_nm_opt - tot_nm_base, 2),
        stockout_reduction_pct=round(short_red, 2),
        baseline_roi_pct=round(roi_base, 2),
        optimal_roi_pct=round(roi_opt, 2),
        roi_lift_pct=round(roi_opt - roi_base, 2),
    )
