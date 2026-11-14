"""OmniGrowth-AI Unified 5-Tab Prescriptive Analytics & Experimentation Application."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

for _env_k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import streamlit as st  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components import (  # noqa: E402
    render_ab_testing_tab,
    render_forecasting_tab,
    render_genai_excel_tab,
    render_optimization_tab,
    render_sql_ga4_tab,
)
from src.config import get_config  # noqa: E402
from src.database.duckdb_manager import get_duckdb_manager  # noqa: E402

st.set_page_config(
    page_title="OmniGrowth-AI | Prescriptive Growth & Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _read_rss_mb() -> float:
    """Reads current process resident set size (RSS) in MB via Linux /proc/self/statm."""
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return 0.0
    pages = int(statm.read_text(encoding="utf-8").split()[1])
    return (pages * os.sysconf("SC_PAGE_SIZE")) / (1024.0 * 1024.0)


@st.cache_data(ttl=600, max_entries=1, show_spinner=False)
def _load_filter_metadata() -> dict[str, object]:
    db = get_duckdb_manager()
    regs = db.query_df(
        "SELECT DISTINCT geo_region FROM v_ga4_events WHERE geo_region IS NOT NULL ORDER BY 1"
    )["geo_region"].tolist()
    devs = db.query_df(
        "SELECT DISTINCT device_category FROM v_ga4_events WHERE device_category IS NOT NULL ORDER BY 1"
    )["device_category"].tolist()
    bounds = db.query_df("SELECT MIN(event_date) AS min_d, MAX(event_date) AS max_d FROM v_ga4_events").iloc[0]
    cfg = get_config()
    lineage: dict[str, object] = {}
    if cfg.paths.lineage_json.exists():
        lineage = json.loads(cfg.paths.lineage_json.read_text(encoding="utf-8"))
    return {
        "ga4_regions": regs,
        "ga4_devices": devs,
        "min_date": bounds["min_d"].date() if hasattr(bounds["min_d"], "date") else date(2020, 11, 1),
        "max_date": bounds["max_d"].date() if hasattr(bounds["max_d"], "date") else date(2021, 1, 31),
        "lineage": lineage,
    }


def _render_sidebar(meta: dict[str, object]) -> tuple[str | None, str | None, tuple[date, date], str]:
    with st.sidebar:
        st.title("OmniGrowth-AI")
        st.caption("Omnichannel Product Growth, Experimentation & Prescriptive Solver")

        st.subheader("Global Filters")
        sel_reg = st.selectbox("GA4 Geo Region", options=["All Regions", *meta["ga4_regions"]], index=0)
        sel_dev = st.selectbox("GA4 Device Category", options=["All Devices", *meta["ga4_devices"]], index=0)
        min_d, max_d = meta["min_date"], meta["max_date"]
        d_range = st.date_input("GA4 Date Window", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        date_bounds = (d_range[0], d_range[1]) if isinstance(d_range, tuple) and len(d_range) == 2 else (min_d, max_d)

        demand_reg = st.selectbox(
            "Default Demand Cohort (Tab 3)",
            options=["ALL", "Southeast", "South", "Northeast", "Center-West", "North"],
            index=0,
        )

        st.divider()
        st.subheader("VPS Architecture & Memory")
        rss_mb = _read_rss_mb()
        cap_mb = 300.0
        st.metric("Live Process RSS Memory", f"{rss_mb:.1f} MB", f"Limit {cap_mb:.0f} MB")
        st.progress(min(rss_mb / cap_mb, 1.0))

        settings = get_duckdb_manager().current_settings()
        st.caption(
            f"DuckDB: `memory_limit={settings.get('memory_limit', '192MB')}`, "
            f"`threads={settings.get('threads', '1')}` | Zero-copy Parquet"
        )
        with st.expander("Data Lineage & Sources", expanded=False):
            lineage = meta.get("lineage", {})
            st.json(
                {
                    "ga4_events_source": "BigQuery GA4 Ecommerce Sample",
                    "regional_demand_source": "Brazilian E-Commerce Olist + Synth",
                    "offline_pretrain_models": ["Prophet", "StatsForecast AutoARIMA", "Bass/Logistic S-Curve"],
                    "processed_datasets": list(lineage.get("datasets", {}).keys())
                    if isinstance(lineage, dict)
                    else [],
                }
            )

    ga4_region = None if sel_reg == "All Regions" else sel_reg
    ga4_device = None if sel_dev == "All Devices" else sel_dev
    return ga4_region, ga4_device, date_bounds, demand_reg


def main() -> None:
    """Entrypoint for the OmniGrowth-AI Streamlit application."""
    meta = _load_filter_metadata()
    ga4_region, ga4_device, date_bounds, demand_reg = _render_sidebar(meta)

    st.markdown(
        "<h2 style='margin-bottom:0.2rem;color:#1B365D;'>"
        "OmniGrowth-AI — Prescriptive Product & Growth Intelligence</h2>"
        "<p style='color:#475569;margin-top:0;'>"
        "DuckDB Zero-Copy Analytics · CUPED & Bayesian Experimentation · "
        "Prophet/AutoARIMA & Bass S-Curve · SLSQP Optimization · GenAI & Excel"
        "</p>",
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "1. SQL Funnel & Looker BI",
            "2. A/B Testing Lab",
            "3. Forecasting & S-Curves",
            "4. Prescriptive Optimizer",
            "5. GenAI Copilot & Excel",
        ]
    )

    with t1:
        t1_out = render_sql_ga4_tab(ga4_region, ga4_device, date_bounds)
    with t2:
        t2_out = render_ab_testing_tab()
    with t3:
        t3_out = render_forecasting_tab(default_region=demand_reg)
    with t4:
        t4_out = render_optimization_tab()
    with t5:
        render_genai_excel_tab(
            funnel_df=t1_out["funnel_df"],
            ab_metrics=t2_out,
            forecast_df=t3_out["forecast_df"],
            scurve_params=t3_out["scurve_params"],
            opt_payload=t4_out,
            funnel_kpis=t1_out["kpis"],
            forecast_mape=t3_out["forecast_mape"],
        )


if __name__ == "__main__":
    main()
