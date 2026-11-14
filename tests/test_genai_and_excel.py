"""Unit and integration tests for Module 5 (GenAI Resilience & Live Excel Engine)."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
from openpyxl import load_workbook

from src.analytics.ga4_funnel import get_funnel_conversion_summary
from src.config import GenAISettings, get_config
from src.forecasting.time_series import load_pretrained_forecasts
from src.genai.assistant import GenAIAssistant
from src.genai.sql_validator import SQLValidationError, inspect_sql, validate_readonly_sql
from src.optimization.prescriptive_solver import solve_prescriptive_allocation
from src.reporting.excel_generator import generate_executive_workbook


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "",
        "DROP TABLE v_ga4_events;",
        "SELECT * FROM v_ga4_events; COPY v_ga4_events TO '/tmp/leak.csv';",
        "ALTER TABLE v_ga4_events ADD COLUMN x INT",
        "INSERT INTO v_ab_logs VALUES (1)",
        "DELETE FROM v_regional_demand WHERE 1=1",
        "INSTALL httpfs; SELECT 1",
        "PRAGMA table_info('v_ga4_events')",
        "ATTACH '/tmp/evil.db' AS evil",
        "/* comment */ UPDATE v_ab_logs SET converted = 1",
    ],
)
def test_sql_validator_blocks_destructive_and_multi_statement_sql(malicious_sql: str) -> None:
    """Verifies sql_validator rejects DDL/DML, extensions, and multiple statements."""
    with pytest.raises(SQLValidationError):
        validate_readonly_sql(malicious_sql)
    assert inspect_sql(malicious_sql).is_valid is False


def test_sql_validator_allows_multi_cte_and_markdown_wrapped_select() -> None:
    """Verifies multi-CTE WITH queries and fenced SELECT statements pass sanitization."""
    cte_sql = """
    ```sql
    -- Leading comment
    WITH regional AS (
        SELECT macro_region, SUM(gross_revenue) AS rev
        FROM v_regional_demand
        GROUP BY macro_region
    )
    SELECT macro_region, rev FROM regional ORDER BY rev DESC;
    ```
    """
    normalized = validate_readonly_sql(cte_sql)
    assert normalized.startswith("WITH regional AS")
    assert not normalized.endswith(";")
    assert inspect_sql(cte_sql).statement_type == "WITH"


def test_genai_assistant_works_offline_with_zero_api_keys() -> None:
    """Verifies zero-key offline cache returns valid SQL, DataFrames, and Executive Memo."""
    offline_settings = GenAISettings(groq_api_key=None, gemini_api_key=None, timeout_s=10.0)
    assistant = GenAIAssistant(settings=offline_settings)

    prompts = assistant.list_curated_prompts()
    assert len(prompts) == 10

    # Verify every curated prompt executes cleanly against live DuckDB views
    for item in prompts:
        res = assistant.ask_analytics_sql(item["question"])
        assert res.from_cache is True
        assert res.provider == "offline_cache"
        assert not res.data.empty
        assert res.interpretation

    memo = assistant.generate_executive_memo(
        {
            "total_sessions": 90000,
            "total_revenue_usd": 420000.0,
            "overall_conversion_pct": 3.45,
            "expected_incremental_margin": 48500.0,
            "roi_lift_pct": 12.4,
            "biggest_dropoff_stage": "begin_checkout",
            "biggest_dropoff_pct": 35.0,
            "ab_revenue_lift": 0.72,
            "cuped_var_reduction_pct": 38.0,
            "bayesian_prob_win": 0.994,
            "bayesian_expected_loss": 0.0002,
            "bass_tam_m": 21500,
            "bass_p": 0.015,
            "bass_q": 0.025,
            "total_budget_allocated": 100000,
            "total_inventory_allocated": 22000,
            "stockout_reduction_pct": 64.0,
            "forecast_mape": 18.5,
        }
    )
    assert memo.from_cache is True
    assert "## Executive Summary" in memo.markdown
    assert "## Prescriptive Action Plan" in memo.markdown
    assert "## Risk & Confidence Intervals" in memo.markdown


def test_excel_generator_creates_5_tabs_with_live_formulas_and_chart() -> None:
    """Verifies OpenPyXL workbook contains all 5 sheets, live '=' formulas, and LineChart."""
    funnel_df = get_funnel_conversion_summary()
    fc_df = load_pretrained_forecasts(region="ALL", model_name="prophet")
    scurve_df = pd.read_parquet(get_config().paths.scurve_params)
    scurve_row = scurve_df[scurve_df["region"] == "ALL"].iloc[0].to_dict()
    opt_res = solve_prescriptive_allocation(total_budget=100_000.0, total_warehouse_inventory=22_000.0)

    stream = generate_executive_workbook(
        funnel_df=funnel_df,
        ab_metrics={
            "n_control": 145000,
            "n_treatment": 145000,
            "conv_control": 17400,
            "conv_treatment": 18125,
            "cuped_var_reduction_pct": 36.2,
            "ctrl_rev_user": 18.4,
            "treat_rev_user": 19.2,
            "daily_traffic": 9600,
        },
        forecast_df=fc_df,
        scurve_params=scurve_row,
        allocation_df=opt_res.allocation_df,
    )

    wb = load_workbook(stream, data_only=False)
    expected_tabs = [
        "Executive_Summary",
        "GA4_Funnel_Economics",
        "AB_Test_Model",
        "Forecast_And_SCurve",
        "Prescriptive_Allocation",
    ]
    assert wb.sheetnames == expected_tabs

    # Verify cross-sheet links on Tab 1
    ws_sum = wb["Executive_Summary"]
    assert str(ws_sum["C4"].value).startswith("=GA4_Funnel_Economics!")
    assert str(ws_sum["C8"].value).startswith("=AB_Test_Model!")
    assert str(ws_sum["C11"].value).startswith("=Forecast_And_SCurve!")
    assert str(ws_sum["C13"].value).startswith("=Prescriptive_Allocation!")

    # Verify Tab 2, Tab 3, Tab 4, Tab 5 live formulas & LineChart
    assert str(wb["GA4_Funnel_Economics"]["D6"].value) == "=C6/C5"
    assert str(wb["GA4_Funnel_Economics"]["E6"].value) == "=1-D6"
    assert str(wb["AB_Test_Model"]["D9"].value) == "=2*(1-NORM.S.DIST(ABS(D8),TRUE))"
    assert "EXP(" in str(wb["Forecast_And_SCurve"]["F9"].value)
    assert len(wb["Forecast_And_SCurve"]._charts) == 1
    assert str(wb["Prescriptive_Allocation"]["D4"].value).startswith("=(")
    assert str(wb["Prescriptive_Allocation"]["C9"].value).startswith("=SUM(")


def test_genai_http_failover_groq_to_gemini_to_cache() -> None:
    """Verifies exponential backoff failover: Groq 503 -> Gemini 200 -> offline cache."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if "groq.com" in request.url.host:
            return httpx.Response(503, json={"error": "rate_limit"})
        body = request.content.decode("utf-8")
        if "Write a concise" in body:
            reply = "Southeast leads total gross revenue across macro regions."
        else:
            reply = "SELECT macro_region, ROUND(SUM(gross_revenue), 2) AS rev FROM v_regional_demand GROUP BY 1"
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": reply}]}}]},
        )

    settings = GenAISettings(
        groq_api_key="dummy-groq",
        gemini_api_key="dummy-gemini",
        timeout_s=2.0,
        max_retries=2,
        backoff_base_s=0.0,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assistant = GenAIAssistant(settings=settings, client=client)
        res = assistant.ask_analytics_sql("Show revenue by macro region")

    assert calls.count("api.groq.com") == 4
    assert "generativelanguage.googleapis.com" in calls
    assert res.from_cache is False
    assert res.provider.startswith("gemini:")
    assert not res.data.empty

    # Dual failure falls back cleanly to offline deterministic cache
    def always_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    with httpx.Client(transport=httpx.MockTransport(always_fail)) as fail_client:
        fallback_assistant = GenAIAssistant(settings=settings, client=fail_client)
        res_fb = fallback_assistant.ask_analytics_sql("Show funnel drop-off by stage")
    assert res_fb.from_cache is True
    assert res_fb.provider == "offline_cache"
