"""Tab 1: DuckDB SQL Analytics, GA4 Conversion Funnel & Looker Studio BI."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.analytics.ga4_funnel import (
    ALLOWED_SEGMENT_DIMENSIONS,
    execute_readonly_sql_lab,
    get_cohort_retention_matrix,
    get_dropoff_attribution_by_segment,
    get_funnel_conversion_summary,
)
from src.analytics.sql_queries import SQL_CATALOG
from src.config import get_config


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def _cached_funnel(region: str | None, device: str | None, start_str: str | None, end_str: str | None) -> pd.DataFrame:
    return get_funnel_conversion_summary(region=region, device=device, start_date=start_str, end_date=end_str)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def _cached_segment(dimension: str) -> pd.DataFrame:
    return get_dropoff_attribution_by_segment(dimension=dimension)


@st.cache_data(ttl=300, max_entries=2, show_spinner=False)
def _cached_cohort() -> pd.DataFrame:
    return get_cohort_retention_matrix()


def render_sql_ga4_tab(
    ga4_region: str | None,
    ga4_device: str | None,
    date_bounds: tuple[date, date] | None,
) -> dict[str, Any]:
    """Renders Tab 1 and returns key funnel metrics for executive reporting."""
    start_str = date_bounds[0].isoformat() if date_bounds else None
    end_str = date_bounds[1].isoformat() if date_bounds else None

    funnel_df = _cached_funnel(ga4_region, ga4_device, start_str, end_str)
    kpis = funnel_df.attrs.get("kpis", {})
    elapsed_ms = funnel_df.attrs.get("elapsed_ms", 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Funnel Sessions", f"{kpis.get('total_sessions', 0):,}")
    c2.metric("Completed Purchases", f"{kpis.get('total_purchases', 0):,}")
    c3.metric("Session -> Purchase CVR", f"{kpis.get('overall_conversion_pct', 0.0):.2f}%")
    c4.metric("Purchase Revenue", f"${kpis.get('total_revenue_usd', 0.0):,.0f}")
    c5.metric("DuckDB Query Latency", f"{elapsed_ms:.1f} ms")

    col_left, col_right = st.columns([1.15, 1.0])
    with col_left:
        st.subheader("Chronological GA4 Conversion Funnel")
        fig_funnel = go.Figure(
            go.Funnel(
                y=funnel_df["stage_name"],
                x=funnel_df["sessions_count"],
                textposition="inside",
                textinfo="value+percent initial+percent previous",
                marker={"color": ["#1B365D", "#255C99", "#3B82F6", "#0D9488", "#059669"]},
            )
        )
        fig_funnel.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=370)
        st.plotly_chart(fig_funnel, width="stretch")

    with col_right:
        st.subheader("Segment Drop-Off Attribution")
        seg_dim = st.selectbox(
            "Breakdown Dimension",
            options=list(ALLOWED_SEGMENT_DIMENSIONS),
            index=0,
            key="tab1_seg_dim",
        )
        seg_df = _cached_segment(seg_dim)
        if not seg_df.empty and "overall_cvr_pct" in seg_df.columns:
            fig_seg = px.bar(
                seg_df.head(10),
                x=seg_dim,
                y="overall_cvr_pct",
                color="purchase_revenue_usd",
                color_continuous_scale="Blues",
                labels={"overall_cvr_pct": "Overall CVR (%)", "purchase_revenue_usd": "Revenue ($)"},
            )
            fig_seg.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=310)
            st.plotly_chart(fig_seg, width="stretch")
        st.dataframe(seg_df.head(8), width="stretch", hide_index=True)

    st.subheader("Weekly User Retention Cohort Matrix (%)")
    cohort_df = _cached_cohort()
    if not cohort_df.empty:
        pivot = cohort_df.pivot(index="cohort_week", columns="weeks_since_first", values="retention_pct")
        fig_heat = px.imshow(
            pivot,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            labels={"x": "Weeks Since First Session", "y": "Cohort Week", "color": "Retention (%)"},
        )
        fig_heat.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=340)
        st.plotly_chart(fig_heat, width="stretch")

    with st.expander("SQL Lab & Pre-Built DuckDB Query Inspector", expanded=False):
        q_keys = list(SQL_CATALOG.keys())
        sel_key = st.selectbox(
            "Pre-Built Window & CTE Query Catalog",
            options=q_keys,
            format_func=lambda k: f"{SQL_CATALOG[k].title} ({', '.join(SQL_CATALOG[k].constructs)})",
            key="tab1_sql_key",
        )
        spec = SQL_CATALOG[sel_key]
        st.caption(spec.description)
        sql_input = st.text_area("Read-Only DuckDB SQL", value=spec.sql.strip(), height=190, key="tab1_sql_text")
        if st.button("Execute SQL Query", key="tab1_run_sql"):
            try:
                res_df = execute_readonly_sql_lab(sql_input, max_rows=500)
                st.success(f"Returned {len(res_df)} rows (zero-copy Parquet execution).")
                st.dataframe(res_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"SQL execution blocked or failed: {exc}")

    with st.expander("Google Looker Studio Interactive Embed & Semantic LookML", expanded=False):
        embed_url = get_config().looker_embed_url
        show_embed = st.checkbox("Render Looker Studio Live Iframe", value=False, key="tab1_looker_toggle")
        if show_embed and embed_url:
            components.iframe(embed_url, height=580, scrolling=True)
        elif show_embed:
            st.info(
                "Set `LOOKER_STUDIO_EMBED_URL` in `.env` to render your published Google Looker Studio dashboard here. "
                "Local LookML definitions are stored in `looker/ga4_funnel.view.lkml`, and the exported BI mart "
                "is refreshed at `data/processed/looker_executive_mart.csv`."
            )
            st.caption(f"Configured CSV mart: `{get_config().paths.looker_mart_csv}`")

    return {"funnel_df": funnel_df, "kpis": kpis}
