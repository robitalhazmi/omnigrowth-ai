"""Unit and integration tests for Module 4 (Prescriptive Supply Chain & Marketing Optimizer)."""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.prescriptive_solver import (
    DEFAULT_REGIONAL_PROFILES,
    load_profiles_from_forecasts,
    solve_prescriptive_allocation,
)


def test_slsqp_solver_converges_and_respects_all_constraints() -> None:
    """Verifies SLSQP convergence, budget cap, warehouse cap, safety stock floor, and margin lift."""
    total_budget = 100_000.0
    total_inventory = 22_000.0
    res = solve_prescriptive_allocation(
        total_budget=total_budget,
        total_warehouse_inventory=total_inventory,
        profiles=DEFAULT_REGIONAL_PROFILES,
    )

    assert res.success is True
    assert res.total_budget_allocated <= total_budget + 1e-4
    assert res.total_inventory_allocated <= res.total_warehouse_capacity + 1e-4

    df = res.allocation_df
    assert len(df) == len(DEFAULT_REGIONAL_PROFILES)
    assert np.all(df["optimal_spend"].to_numpy() >= -1e-6)
    assert np.all(df["optimal_inventory"].to_numpy() + 1e-2 >= df["safety_stock_80"].to_numpy())

    # Optimal allocation must outperform or equal the equal-weight baseline allocation
    assert res.optimal_total_net_margin >= res.baseline_total_net_margin
    assert res.expected_incremental_margin >= 0.0
    assert res.optimal_roi_pct >= res.baseline_roi_pct


def test_solver_reallocates_toward_high_elasticity_and_margin_regions() -> None:
    """Verifies high-return regions receive higher optimal marketing spend than low-return regions."""
    res = solve_prescriptive_allocation(total_budget=120_000.0, total_warehouse_inventory=25_000.0)
    df = res.allocation_df.set_index("region")

    # Southeast has the largest demand base and high margin; North has the smallest base and lowest margin
    assert df.loc["Southeast", "optimal_spend"] > df.loc["North", "optimal_spend"]
    assert df.loc["Southeast", "optimal_inventory"] > df.loc["North", "optimal_inventory"]


def test_solver_with_forecast_calibrated_profiles() -> None:
    """Verifies profiles loaded from forecasts_pretrained.parquet solve cleanly."""
    profiles = load_profiles_from_forecasts()
    assert len(profiles) == 5
    for p in profiles:
        assert p.base_demand_units > 0
        assert 1.0 < p.demand_q80_ratio < p.demand_q95_ratio

    total_min_stock = sum(p.base_demand_units * p.demand_q80_ratio for p in profiles)
    res = solve_prescriptive_allocation(
        total_budget=150_000.0,
        total_warehouse_inventory=total_min_stock * 1.35,
        profiles=profiles,
    )
    assert res.success is True
    assert res.total_budget_allocated <= 150_000.0 + 1e-4
    assert res.optimal_total_net_margin >= res.baseline_total_net_margin


def test_solver_handles_tight_warehouse_capacity_gracefully() -> None:
    """Verifies solver remains feasible when user specifies a tight warehouse capacity."""
    res = solve_prescriptive_allocation(total_budget=80_000.0, total_warehouse_inventory=5_000.0)
    assert res.success is True
    assert res.total_budget_allocated <= 80_000.0 + 1e-4
    df = res.allocation_df
    assert np.all(df["optimal_inventory"].to_numpy() + 1e-2 >= df["safety_stock_80"].to_numpy())


def test_solver_rejects_nonpositive_inputs() -> None:
    """Verifies ValueError on non-positive budget, inventory, or empty profile list."""
    with pytest.raises(ValueError):
        solve_prescriptive_allocation(total_budget=0.0, total_warehouse_inventory=10_000.0)
    with pytest.raises(ValueError):
        solve_prescriptive_allocation(total_budget=50_000.0, total_warehouse_inventory=-100.0)
    with pytest.raises(ValueError):
        solve_prescriptive_allocation(total_budget=50_000.0, total_warehouse_inventory=10_000.0, profiles=[])
