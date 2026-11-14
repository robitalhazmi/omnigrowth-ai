"""Tab 4: Prescriptive SLSQP Budget & Inventory Optimization Solver."""

from __future__ import annotations

import time
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.optimization.prescriptive_solver import (
    DEFAULT_REGIONAL_PROFILES,
    PrescriptiveOptimizationResult,
    load_profiles_from_forecasts,
    solve_prescriptive_allocation,
)


def render_optimization_tab() -> dict[str, Any]:
    """Renders Tab 4 and returns the optimization result for executive reporting."""
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    total_budget = c1.slider("Total Marketing Budget ($)", 40_000.0, 250_000.0, 100_000.0, 5_000.0, key="tab4_budget")
    warehouse_cap = c2.slider(
        "Warehouse Inventory Pool (Units)", 10_000.0, 50_000.0, 22_000.0, 1_000.0, key="tab4_warehouse"
    )
    use_pretrained = c3.checkbox("Calibrate with 90-Day Prophet Demand", value=False, key="tab4_pretrained")

    t0 = time.perf_counter()
    profiles = load_profiles_from_forecasts() if use_pretrained else DEFAULT_REGIONAL_PROFILES
    res: PrescriptiveOptimizationResult = solve_prescriptive_allocation(
        total_budget=total_budget,
        total_warehouse_inventory=warehouse_cap,
        profiles=profiles,
    )
    solve_ms = (time.perf_counter() - t0) * 1000.0
    df = res.allocation_df

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Optimal Net Margin", f"${res.optimal_total_net_margin:,.0f}")
    inc_margin = res.expected_incremental_margin
    m2.metric("Incremental Margin Gain", f"{'+' if inc_margin >= 0 else '-'}${abs(inc_margin):,.0f}")
    m3.metric("Marketing ROI Lift", f"{res.optimal_roi_pct:.1f}%", f"{res.roi_lift_pct:+.1f} pp")
    m4.metric("95th-Pctl Stockout Reduction", f"{res.stockout_reduction_pct:.1f}%")
    m5.metric("SLSQP Solve Latency", f"{solve_ms:.1f} ms", f"{res.iterations} iters")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Regional Marketing Budget Reallocation ($)")
        fig_b = go.Figure()
        fig_b.add_trace(
            go.Bar(x=df["region"], y=df["baseline_spend"], name="Equal Baseline ($)", marker_color="#94A3B8")
        )
        fig_b.add_trace(
            go.Bar(x=df["region"], y=df["optimal_spend"], name="SLSQP Optimal ($)", marker_color="#1B365D")
        )
        fig_b.update_layout(barmode="group", height=340, yaxis_title="Spend ($)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_b, width="stretch")

    with col_right:
        st.subheader("Regional Warehouse Stock vs. 80% Safety Floor")
        fig_i = go.Figure()
        fig_i.add_trace(
            go.Bar(x=df["region"], y=df["baseline_inventory"], name="Equal Baseline Units", marker_color="#94A3B8")
        )
        fig_i.add_trace(
            go.Bar(x=df["region"], y=df["optimal_inventory"], name="SLSQP Optimal Units", marker_color="#0D9488")
        )
        fig_i.add_trace(
            go.Scatter(
                x=df["region"],
                y=df["safety_stock_80"],
                name="80% Safety Floor",
                mode="lines+markers",
                line={"color": "#DC2626", "dash": "dash"},
            )
        )
        fig_i.update_layout(
            barmode="group", height=340, yaxis_title="Inventory Units", margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_i, width="stretch")

    st.subheader("Net Margin Contribution Lift by Region ($)")
    fig_m = go.Figure(
        go.Bar(
            x=df["region"],
            y=df["incremental_margin"],
            text=[f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}" for v in df["incremental_margin"]],
            textposition="outside",
            marker_color=["#059669" if v >= 0 else "#DC2626" for v in df["incremental_margin"]],
        )
    )
    fig_m.update_layout(height=300, yaxis_title="Incremental Margin ($)", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_m, width="stretch")

    st.subheader("Regional Prescriptive Allocation Ledger")
    st.dataframe(df, width="stretch", hide_index=True)

    top_row = df.sort_values("optimal_spend", ascending=False).iloc[0]
    return {
        "result": res,
        "top_growth_region": str(top_row["region"]),
    }
