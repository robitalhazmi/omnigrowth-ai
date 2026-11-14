"""Tab 3: Pretrained Prophet vs. AutoARIMA Forecasting & Bass S-Curve Simulator."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_config
from src.database.duckdb_manager import get_duckdb_manager, quote_path
from src.forecasting.s_curve import (
    BassFitResult,
    bass_adoption_rate,
    bass_diffusion_curve,
    logistic_curve,
    shift_bass_scenario,
)
from src.forecasting.time_series import apply_scenario_shock, load_pretrained_forecasts


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def _cached_forecasts(region: str, model_name: str) -> pd.DataFrame:
    return load_pretrained_forecasts(region=region, model_name=model_name)


@st.cache_data(ttl=300, max_entries=2, show_spinner=False)
def _cached_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, db = get_config(), get_duckdb_manager()
    sc = db.query_df(f"SELECT * FROM read_parquet({quote_path(cfg.paths.scurve_params)})")
    bt = db.query_df(f"SELECT * FROM read_parquet({quote_path(cfg.paths.model_backtest_metrics)})")
    return sc, bt


def _row_to_bass_fit(row: dict[str, Any]) -> BassFitResult:
    p, q, m = float(row["bass_p"]), float(row["bass_q"]), float(row["bass_m"])
    t_star = float(row["bass_t_star"])
    peak_rate = float(row["bass_peak_rate"])
    return BassFitResult(
        p_innovation=p,
        q_imitation=q,
        carrying_capacity_m=m,
        peak_adoption_t_star=t_star,
        peak_adoption_rate=peak_rate,
        inflection_adoption_level=float(row.get("bass_inflection_level", m * 0.5)),
        r_squared=float(row.get("bass_r2", 0.99)),
        rmse=float(row.get("bass_rmse", 0.0)),
    )


def render_forecasting_tab(default_region: str = "ALL") -> dict[str, Any]:
    """Renders Tab 3 (< 50 ms what-if simulation over pretrained parquets)."""
    scurve_all, bt_all = _cached_artifacts()
    regions = scurve_all["region"].tolist()
    reg_idx = regions.index(default_region) if default_region in regions else 0

    c1, c2 = st.columns([1.2, 1.0])
    region = c1.selectbox("Macro-Region Cohort", options=regions, index=reg_idx, key="tab3_region")
    model_name = c2.radio("Pretrained Engine", ["prophet", "auto_arima"], horizontal=True, key="tab3_model")

    sc_row = scurve_all[scurve_all["region"] == region].iloc[0].to_dict()
    bass_fit = _row_to_bass_fit(sc_row)
    base_df = _cached_forecasts(region, model_name)

    st.subheader("Real-Time Analytical Scenario Simulator (< 50 ms)")
    s1, s2, s3, s4, s5 = st.columns(5)
    spend_pct = s1.slider("Marketing Spend Delta (%)", -40.0, 80.0, 0.0, 5.0, key="tab3_spend")
    promo_pct = s2.slider("Promo Discount (%)", 0.0, 25.0, 0.0, 1.0, key="tab3_promo")
    cvr_pct = s3.slider("Conversion Lift (%)", -20.0, 40.0, 0.0, 2.5, key="tab3_cvr")
    viral_pct = s4.slider("Viral Boost q (%)", -20.0, 50.0, 0.0, 5.0, key="tab3_viral")
    tam_mult = s5.slider("TAM Expansion (x)", 0.8, 1.5, 1.0, 0.05, key="tab3_tam")

    t0 = time.perf_counter()
    shock_df = apply_scenario_shock(
        base_df,
        spend_change_pct=spend_pct,
        promo_discount_pct=promo_pct,
        conversion_lift_pct=cvr_pct,
        bass_fit=bass_fit,
    )
    t_horizon = np.arange(1, int(sc_row.get("n_days_observed", 600)) + 120, dtype=float)
    shifted_bass, shifted_curve = shift_bass_scenario(
        bass_fit, t_horizon, spend_change_pct=spend_pct, viral_boost_pct=viral_pct, tam_multiplier=tam_mult
    )
    sim_ms = (time.perf_counter() - t0) * 1000.0

    fc_only_base = base_df[base_df["is_forecast"]]
    fc_only_shock = shock_df[shock_df["is_forecast"]]
    base_90d = float(fc_only_base["yhat"].sum())
    shock_90d = float(fc_only_shock["yhat"].sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("90-Day Baseline Forecast", f"${base_90d:,.0f}")
    delta_fc_pct = (shock_90d - base_90d) / max(base_90d, 1) * 100.0
    m2.metric("90-Day Scenario Forecast", f"${shock_90d:,.0f}", f"{delta_fc_pct:+.1f}%")
    m3.metric("Bass TAM Ceiling (M)", f"{shifted_bass.carrying_capacity_m:,.0f} merchants")
    m4.metric("Peak Adoption t*", f"Day {shifted_bass.peak_adoption_t_star:.1f}")
    m5.metric("Simulation Latency", f"{sim_ms:.2f} ms")

    hist_tail = base_df[~base_df["is_forecast"]].tail(180)
    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=pd.concat([fc_only_base["order_date"], fc_only_base["order_date"].iloc[::-1]]),
            y=pd.concat([fc_only_base["yhat_upper_95"], fc_only_base["yhat_lower_95"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(37, 92, 153, 0.12)",
            line={"color": "rgba(255,255,255,0)"},
            name="95% Confidence Band",
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=pd.concat([fc_only_base["order_date"], fc_only_base["order_date"].iloc[::-1]]),
            y=pd.concat([fc_only_base["yhat_upper_80"], fc_only_base["yhat_lower_80"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(37, 92, 153, 0.25)",
            line={"color": "rgba(255,255,255,0)"},
            name="80% Confidence Band",
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=hist_tail["order_date"], y=hist_tail["actual_revenue"],
            name="Actual Revenue", line={"color": "#1F2937"},
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=fc_only_base["order_date"], y=fc_only_base["yhat"],
            name=f"{model_name} Baseline", line={"color": "#255C99", "width": 2.5},
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=fc_only_shock["order_date"], y=fc_only_shock["yhat"],
            name="What-If Scenario", line={"color": "#059669", "width": 2.5, "dash": "dash"},
        )
    )
    fig_fc.update_layout(
        height=360, xaxis_title="Order Date", yaxis_title="Daily Revenue ($)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_fc, width="stretch")

    col_sc, col_bt = st.columns([1.25, 1.0])
    with col_sc:
        st.subheader("Bass Diffusion vs. Logistic S-Curve Saturation")
        base_bass = bass_diffusion_curve(
            t_horizon, bass_fit.p_innovation, bass_fit.q_imitation, bass_fit.carrying_capacity_m
        )
        base_log = logistic_curve(
            t_horizon, float(sc_row["logistic_m"]), float(sc_row["logistic_k"]), float(sc_row["logistic_t0"])
        )
        vel = bass_adoption_rate(
            t_horizon, shifted_bass.p_innovation, shifted_bass.q_imitation, shifted_bass.carrying_capacity_m
        )

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=t_horizon, y=base_bass, name="Fitted Bass N(t)"))
        fig_sc.add_trace(go.Scatter(x=t_horizon, y=base_log, name="Fitted Logistic N(t)", line={"dash": "dot"}))
        fig_sc.add_trace(
            go.Scatter(
                x=t_horizon, y=shifted_curve,
                name="Scenario Bass N(t)", line={"dash": "dash", "color": "#059669"},
            )
        )
        fig_sc.add_trace(
            go.Scatter(
                x=t_horizon, y=vel * 30.0,
                name="Adoption Velocity (x30)", line={"color": "#D97706", "width": 1.5},
            )
        )
        fig_sc.add_vline(
            x=shifted_bass.peak_adoption_t_star, line_dash="dot", line_color="#DC2626", annotation_text="t*"
        )
        fig_sc.add_hline(
            y=shifted_bass.carrying_capacity_m, line_dash="dot", line_color="#1B365D", annotation_text="M (TAM)"
        )
        fig_sc.update_layout(height=330, xaxis_title="Market Days Since Launch (t)", yaxis_title="Active Merchants")
        st.plotly_chart(fig_sc, width="stretch")

    with col_bt:
        st.subheader("Rolling-Origin Cross-Validation Accuracy")
        bt_sub = bt_all[bt_all["region"] == region].copy()
        st.dataframe(bt_sub, width="stretch", hide_index=True)
        st.caption(
            f"Bass Parameters: `p={shifted_bass.p_innovation:.5f}`, `q={shifted_bass.q_imitation:.5f}`, "
            f"`R^2={sc_row.get('bass_r2', 0.0):.4f}` vs Logistic `R^2={sc_row.get('logistic_r2', 0.0):.4f}`."
        )

    bt_row = bt_sub[bt_sub["model_name"] == model_name]
    mape_val = float(bt_row["mape"].iloc[0]) if not bt_row.empty else 15.0
    return {"forecast_df": shock_df, "scurve_params": sc_row, "forecast_mape": mape_val}
