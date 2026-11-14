"""Resilient multi-provider GenAI assistant with deterministic offline cache fallback.

Supports a 3-tier execution hierarchy mandated by ``AGENTS.md``:
1. **Groq Cloud API** (``llama-3.3-70b-versatile``) with a 10s timeout and exponential backoff.
2. **Google Gemini API** (``gemini-2.5-flash``) automatic failover.
3. **Deterministic Offline Cache** (``data/fallback/genai_cache.json``) when API keys are
   unset, rate-limited, or unreachable.

All generated SQL passes :func:`src.genai.sql_validator.validate_readonly_sql` before execution.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.config import GenAISettings, get_config
from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager
from src.genai.sql_validator import SQLValidationError, validate_readonly_sql

logger = logging.getLogger(__name__)

_SCHEMA_CONTEXT = """
Available read-only DuckDB views (do NOT modify data; write a single SELECT or WITH query):
1. v_ga4_events(event_date DATE, event_timestamp TIMESTAMP, event_name VARCHAR, user_pseudo_id VARCHAR,
   ga_session_id BIGINT, device_category VARCHAR, traffic_source VARCHAR, traffic_medium VARCHAR,
   geo_region VARCHAR, item_category VARCHAR, event_value_in_usd DOUBLE)
   -- event_name in ('session_start','view_item','add_to_cart','begin_checkout','purchase')
2. v_regional_demand(order_date DATE, region_code VARCHAR, region_name VARCHAR, macro_region VARCHAR,
   product_segment VARCHAR, daily_orders INTEGER, units_sold INTEGER, unique_customers INTEGER,
   gross_revenue DOUBLE, freight_cost DOUBLE, promo_discount_pct DOUBLE, cumulative_active_merchants INTEGER)
3. v_ab_logs(user_id BIGINT, experiment_group VARCHAR, assigned_date DATE, device_os VARCHAR,
   region_tier VARCHAR, pre_exp_sessions SMALLINT, pre_exp_spend DOUBLE, converted TINYINT,
   post_exp_revenue DOUBLE, retention_d7 TINYINT, retention_d30 TINYINT)
4. v_mql_funnel(mql_id VARCHAR, first_contact_date DATE, lead_origin VARCHAR, won_timestamp TIMESTAMP,
   business_segment VARCHAR, lead_type VARCHAR, business_type VARCHAR, is_won BOOLEAN, days_to_close BIGINT)
""".strip()


@dataclass(frozen=True)
class SQLAgentResponse:
    """Result of the Natural-Language-to-DuckDB SQL workflow."""

    question: str
    sql: str
    data: pd.DataFrame
    interpretation: str
    provider: str
    latency_ms: float
    from_cache: bool


@dataclass(frozen=True)
class MemoResponse:
    """Result of the C-Suite Board Memo generation workflow."""

    markdown: str
    provider: str
    latency_ms: float
    from_cache: bool


class GenAIAssistant:
    """Fault-tolerant LLM client for NL-to-SQL analytics and executive memo synthesis."""

    def __init__(
        self,
        *,
        settings: GenAISettings | None = None,
        cache_path: Path | None = None,
        manager: DuckDBManager | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        cfg = get_config()
        self._settings = settings or cfg.genai
        self._cache_path = cache_path or cfg.paths.genai_cache
        self._db = manager or get_duckdb_manager()
        self._client = client
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        if not self._cache_path.exists():
            return {"curated_prompts": [], "memo_fallback_sections": {}}
        with self._cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_curated_prompts(self) -> list[dict[str, Any]]:
        """Returns the curated prompt catalog stored in ``genai_cache.json``."""
        return list(self._cache.get("curated_prompts", []))

    def _call_with_backoff(self, fn: Any, *args: Any) -> str:
        """Executes an HTTP provider call with exponential backoff retries."""
        last_exc: Exception | None = None
        for attempt in range(self._settings.max_retries):
            try:
                return str(fn(*args))
            except (httpx.HTTPError, KeyError, ValueError, RuntimeError) as exc:
                last_exc = exc
                if attempt + 1 < self._settings.max_retries:
                    delay = self._settings.backoff_base_s * (2**attempt)
                    if delay > 0:
                        time.sleep(delay)
        raise RuntimeError(f"Provider call failed after {self._settings.max_retries} attempts: {last_exc}")

    def _post_groq(self, system_prompt: str, user_prompt: str) -> str:
        if not self._settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        payload = {
            "model": self._settings.groq_model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self._settings.groq_api_key}", "Content-Type": "application/json"}
        url = "https://api.groq.com/openai/v1/chat/completions"
        if self._client is not None:
            resp = self._client.post(url, json=payload, headers=headers, timeout=self._settings.timeout_s)
        else:
            with httpx.Client(timeout=self._settings.timeout_s) as client:
                resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _post_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if not self._settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._settings.gemini_model}:generateContent?key={self._settings.gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"temperature": 0.1},
        }
        if self._client is not None:
            resp = self._client.post(url, json=payload, timeout=self._settings.timeout_s)
        else:
            with httpx.Client(timeout=self._settings.timeout_s) as client:
                resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _complete_with_failover(self, system_prompt: str, user_prompt: str) -> tuple[str | None, str]:
        """Attempts Groq -> Gemini -> returns ``(None, 'offline_cache')``."""
        if self._settings.groq_api_key:
            try:
                text = self._call_with_backoff(self._post_groq, system_prompt, user_prompt)
                return text, f"groq:{self._settings.groq_model}"
            except RuntimeError as exc:
                logger.warning("Groq failed, failing over to Gemini: %s", exc)
        if self._settings.gemini_api_key:
            try:
                text = self._call_with_backoff(self._post_gemini, system_prompt, user_prompt)
                return text, f"gemini:{self._settings.gemini_model}"
            except RuntimeError as exc:
                logger.warning("Gemini failed, falling back to offline cache: %s", exc)
        return None, "offline_cache"

    def _match_cached_prompt(self, user_question: str) -> dict[str, Any]:
        prompts = self.list_curated_prompts()
        if not prompts:
            return {
                "sql": "SELECT macro_region, ROUND(SUM(gross_revenue), 2) AS total_revenue "
                "FROM v_regional_demand GROUP BY macro_region ORDER BY total_revenue DESC",
                "interpretation": "Default regional revenue summary executed against v_regional_demand.",
            }
        q_tokens = set(re.findall(r"[a-z0-9_]+", (user_question or "").lower()))
        best_item, best_score = prompts[0], -1.0
        for item in prompts:
            if item["question"].strip().lower() == (user_question or "").strip().lower():
                return item
            kw_tokens = set(item.get("keywords", [])) | set(re.findall(r"[a-z0-9_]+", item["question"].lower()))
            overlap = len(q_tokens & kw_tokens)
            bonus = 2.0 if any(k in (user_question or "").lower() for k in item.get("keywords", [])) else 0.0
            score = overlap + bonus
            if score > best_score:
                best_score, best_item = score, item
        return best_item

    @staticmethod
    def _summarize_df_offline(base_note: str, df: pd.DataFrame) -> str:
        if df.empty:
            return f"{base_note} Query returned 0 rows."
        preview_cols = list(df.columns[:4])
        top = ", ".join(f"`{col}={df.iloc[0][col]}`" for col in preview_cols)
        return f"{base_note} Returned **{len(df)} rows**; leading record: {top}."

    def ask_analytics_sql(self, user_question: str, *, max_rows: int = 200) -> SQLAgentResponse:
        """Generates, validates, executes read-only DuckDB SQL and explains the result."""
        started = time.perf_counter()
        sql_sys = (
            "You are a principal analytics engineer writing DuckDB SQL.\n"
            f"{_SCHEMA_CONTEXT}\nReturn ONLY valid single-statement DuckDB SELECT or WITH SQL."
        )
        raw_sql, provider = self._complete_with_failover(sql_sys, user_question)
        from_cache = raw_sql is None

        if raw_sql is not None:
            try:
                safe_sql = validate_readonly_sql(raw_sql)
            except SQLValidationError as exc:
                logger.warning("LLM SQL rejected (%s); falling back to cached query.", exc)
                cached = self._match_cached_prompt(user_question)
                safe_sql = validate_readonly_sql(cached["sql"])
                provider, from_cache = "offline_cache", True
        else:
            cached = self._match_cached_prompt(user_question)
            safe_sql = validate_readonly_sql(cached["sql"])

        df = self._db.query_df(f"SELECT * FROM ({safe_sql}) AS _ai_sql_sub LIMIT {int(max_rows)}")

        if not from_cache:
            interp_prompt = (
                f"Question: {user_question}\nSQL:\n{safe_sql}\n"
                f"Top rows CSV:\n{df.head(8).to_csv(index=False)}\n"
                "Write a concise 2-3 sentence executive interpretation referencing only observed numbers."
            )
            interp, _ = self._complete_with_failover("You are a concise Chief Analytics Officer.", interp_prompt)
            interpretation = (interp or "").strip() or self._summarize_df_offline("Query completed.", df)
        else:
            cached_note = self._match_cached_prompt(user_question).get("interpretation", "Analytical query executed.")
            interpretation = self._summarize_df_offline(cached_note, df)

        elapsed = round((time.perf_counter() - started) * 1000.0, 2)
        return SQLAgentResponse(
            question=user_question,
            sql=safe_sql,
            data=df,
            interpretation=interpretation,
            provider=provider,
            latency_ms=elapsed,
            from_cache=from_cache,
        )

    def generate_executive_memo(self, kpi_payload: Mapping[str, Any]) -> MemoResponse:
        """Synthesizes live metrics from Modules 1-4 into a C-Suite Board Memo in Markdown."""
        started = time.perf_counter()
        sys_prompt = (
            "You are a VP of Product Growth & Data Science preparing a concise C-Suite Board Memo. "
            "Use exact sections: ## Executive Summary, ## Key Analytical Findings, "
            "## Prescriptive Action Plan, and ## Risk & Confidence Intervals. Use only the provided metrics."
        )
        user_prompt = (
            "Synthesize these live OmniGrowth-AI model metrics into a board memo:\n"
            f"{json.dumps(kpi_payload, default=str, indent=2)}"
        )
        raw_md, provider = self._complete_with_failover(sys_prompt, user_prompt)
        from_cache = raw_md is None
        markdown = (raw_md or "").strip() if raw_md else self._format_offline_memo(kpi_payload)
        elapsed = round((time.perf_counter() - started) * 1000.0, 2)
        return MemoResponse(markdown=markdown, provider=provider, latency_ms=elapsed, from_cache=from_cache)

    def _format_offline_memo(self, p: Mapping[str, Any]) -> str:
        note = self._cache.get("memo_fallback_sections", {}).get("disclaimer", "Deterministic offline synthesis.")
        sessions = p.get("total_sessions", 0)
        rev = p.get("total_revenue_usd", 0)
        cvr = p.get("overall_conversion_pct", 0)
        margin = p.get("expected_incremental_margin", 0)
        roi_lift = p.get("roi_lift_pct", 0)
        return (
            "## Executive Summary\n"
            f"Across **{sessions:,} GA4 sessions** generating **\\${rev:,.2f}** "
            f"(overall CVR **{cvr:.2f}%**), unified experimentation and prescriptive "
            f"resource allocation unlock **\\${margin:,.2f}** in incremental net margin "
            f"(ROI lift **+{roi_lift:.2f} pp**).\n\n"
            "## Key Analytical Findings\n"
            f"- **Funnel Bottleneck:** Largest drop-off occurs at `{p.get('biggest_dropoff_stage', 'checkout')}` "
            f"(**{p.get('biggest_dropoff_pct', 0):.2f}%** step loss).\n"
            f"- **CUPED & Bayesian A/B Test:** Revenue lift of **\\${p.get('ab_revenue_lift', 0):,.3f}/user** with "
            f"CUPED variance reduction of **{p.get('cuped_var_reduction_pct', 0):.1f}%** and "
            f"**P(Treatment > Control) = {p.get('bayesian_prob_win', 0) * 100:.2f}%**.\n"
            f"- **Market Diffusion:** Bass S-curve TAM ceiling estimated at **{p.get('bass_tam_m', 0):,.0f}** "
            f"merchants (`p={p.get('bass_p', 0):.4f}`, `q={p.get('bass_q', 0):.4f}`).\n\n"
            "## Prescriptive Action Plan\n"
            f"- Reallocate **\\${p.get('total_budget_allocated', 0):,.0f}** promotional budget and "
            f"**{p.get('total_inventory_allocated', 0):,.0f}** inventory units via SLSQP, targeting "
            f"**{p.get('top_growth_region', 'Southeast')}** for highest marginal return.\n"
            "- Ship winning A/B experience globally and prioritize checkout friction remediation.\n\n"
            "## Risk & Confidence Intervals\n"
            f"- **Stockout Mitigation:** Optimal inventory reduces 95th-percentile stockout risk by "
            f"**{p.get('stockout_reduction_pct', 0):.2f}%** while honoring all regional 80% safety-stock floors.\n"
            f"- **Bayesian Expected Loss:** Risk of shipping Treatment if inferior is bounded at "
            f"**{p.get('bayesian_expected_loss', 0):.5f}**; Prophet/AutoARIMA backtest MAPE is "
            f"**{p.get('forecast_mape', 0):.2f}%**.\n\n"
            f"*{note}*"
        )
