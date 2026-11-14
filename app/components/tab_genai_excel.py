"""Tab 5: GenAI SQL Copilot, C-Suite Memo Synthesizer & Live 5-Tab Excel Export."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.genai.assistant import GenAIAssistant
from src.reporting.excel_generator import generate_executive_workbook


def render_genai_excel_tab(
    *,
    funnel_df: pd.DataFrame,
    ab_metrics: dict[str, Any],
    forecast_df: pd.DataFrame,
    scurve_params: dict[str, Any],
    opt_payload: dict[str, Any],
    funnel_kpis: dict[str, Any],
    forecast_mape: float,
) -> None:
    """Renders Tab 5 for NL-to-SQL exploration, C-Suite Board Memos, and .xlsx download."""
    assistant = GenAIAssistant()
    opt_res = opt_payload["result"]
    alloc_df = opt_res.allocation_df

    st.subheader("Natural-Language to Read-Only DuckDB SQL Copilot")
    curated = assistant.list_curated_prompts()
    if curated:
        labels = [item["question"] for item in curated]
        preset = st.selectbox(
            "Quick Curated Executive Questions", options=["-- Custom Question --", *labels], key="t5_q"
        )
    else:
        preset = "-- Custom Question --"

    fallback_q = curated[0]["question"] if curated else "Show funnel drop-off by stage"
    default_q = preset if preset != "-- Custom Question --" else fallback_q
    question = st.text_input(
        "Ask OmniGrowth Copilot (Groq -> Gemini -> Offline Cache)", value=default_q, key="t5_input"
    )

    if st.button("Run NL-to-DuckDB Copilot", type="primary", key="t5_run_sql"):
        with st.spinner("Validating SQL & querying Parquet views..."):
            resp = assistant.ask_analytics_sql(question, max_rows=200)
        badge = "Offline Cache" if resp.from_cache else f"Live LLM ({resp.provider})"
        st.caption(f"Provider: `{badge}` | Latency: `{resp.latency_ms:.1f} ms` | AST Validator: `READ-ONLY PASSED`")
        st.info(resp.interpretation)
        with st.expander("Generated & Sanitized DuckDB SQL", expanded=True):
            st.code(resp.sql, language="sql")
        st.dataframe(resp.data, width="stretch", hide_index=True)

    st.divider()
    col_memo, col_excel = st.columns([1.3, 1.0])

    with col_memo:
        st.subheader("One-Click C-Suite Board Memo Generator")
        sorted_stages = funnel_df.sort_values("step_order")
        drop_stage = "checkout"
        drop_pct = 0.0
        if len(sorted_stages) > 1 and "step_dropoff_pct" in sorted_stages.columns:
            worst = sorted_stages.iloc[1:].sort_values("step_dropoff_pct", ascending=False).iloc[0]
            drop_stage = str(worst["stage_name"])
            drop_pct = float(worst["step_dropoff_pct"])

        kpi_payload = {
            "total_sessions": funnel_kpis.get("total_sessions", 0),
            "total_revenue_usd": funnel_kpis.get("total_revenue_usd", 0.0),
            "overall_conversion_pct": funnel_kpis.get("overall_conversion_pct", 0.0),
            "expected_incremental_margin": opt_res.expected_incremental_margin,
            "roi_lift_pct": opt_res.roi_lift_pct,
            "biggest_dropoff_stage": drop_stage,
            "biggest_dropoff_pct": drop_pct,
            "ab_revenue_lift": ab_metrics.get("ab_revenue_lift", 0.0),
            "cuped_var_reduction_pct": ab_metrics.get("cuped_var_reduction_pct", 0.0),
            "bayesian_prob_win": ab_metrics.get("bayesian_prob_win", 0.0),
            "bayesian_expected_loss": ab_metrics.get("bayesian_expected_loss", 0.0),
            "bass_tam_m": scurve_params.get("bass_m", 0.0),
            "bass_p": scurve_params.get("bass_p", 0.0),
            "bass_q": scurve_params.get("bass_q", 0.0),
            "total_budget_allocated": opt_res.total_budget_allocated,
            "total_inventory_allocated": opt_res.total_inventory_allocated,
            "top_growth_region": opt_payload.get("top_growth_region", "Southeast"),
            "stockout_reduction_pct": opt_res.stockout_reduction_pct,
            "forecast_mape": forecast_mape,
        }
        if st.button("Generate Executive Board Memo", key="t5_gen_memo"):
            with st.spinner("Synthesizing live KPIs..."):
                st.session_state["t5_memo_result"] = assistant.generate_executive_memo(kpi_payload)

        if "t5_memo_result" in st.session_state:
            memo_res = st.session_state["t5_memo_result"]
            src = "Offline Template" if memo_res.from_cache else memo_res.provider
            st.caption(f"Synthesized via `{src}` in `{memo_res.latency_ms:.1f} ms`")
            st.markdown(memo_res.markdown)

    with col_excel:
        st.subheader("Live 5-Tab Financial Model (.xlsx)")
        st.markdown(
            "Generates a formula-driven OpenPyXL workbook containing:\n"
            "- `Executive_Summary`: Cross-sheet live KPI scorecard\n"
            "- `GA4_Funnel_Economics`: Drop-off & recovery formulas\n"
            "- `AB_Test_Model`: Live `NORM.S.DIST` p-value & CUPED lift\n"
            "- `Forecast_And_SCurve`: Bass Diffusion formula + `LineChart`\n"
            "- `Prescriptive_Allocation`: `SUM` & variance formulas"
        )
        workbook_bytes = generate_executive_workbook(
            funnel_df=funnel_df,
            ab_metrics=ab_metrics,
            forecast_df=forecast_df,
            scurve_params=scurve_params,
            allocation_df=alloc_df,
        )
        st.download_button(
            label="Download Executive Financial Workbook (.xlsx)",
            data=workbook_bytes.getvalue(),
            file_name="omnigrowth_executive_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            key="t5_dl_xlsx",
        )
