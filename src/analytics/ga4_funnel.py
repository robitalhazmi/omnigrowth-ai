"""GA4 funnel conversion, segment drop-off attribution, cohort retention & read-only SQL Lab.

All queries execute against zero-copy DuckDB Parquet views via :class:`DuckDBManager`
while enforcing the ``192MB`` memory limit and ``1`` worker thread guardrail.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from datetime import date
from typing import Final

import pandas as pd

from src.analytics.sql_queries import CLOSED_FUNNEL_PARAMETERIZED_SQL, SQL_CATALOG, SQLQuerySpec
from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager

ALLOWED_SEGMENT_DIMENSIONS: Final[tuple[str, ...]] = (
    "device_category",
    "geo_region",
    "traffic_source",
    "traffic_medium",
    "item_category",
)

_FORBIDDEN_SQL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(DROP|ALTER|TRUNCATE|INSERT|UPDATE|DELETE|CREATE|REPLACE|COPY|ATTACH|"
    r"DETACH|INSTALL|LOAD|PRAGMA|CALL|EXPORT)\b",
    re.IGNORECASE,
)
_LEADING_COMMENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(\s*(--[^\n]*\n|/\*[\s\S]*?\*/))+", re.MULTILINE
)


@dataclass(frozen=True)
class FunnelKPIs:
    """Top-level funnel conversion summary KPIs."""

    total_sessions: int
    total_purchases: int
    overall_conversion_pct: float
    biggest_dropoff_stage: str
    biggest_dropoff_pct: float
    total_revenue_usd: float
    average_order_value_usd: float


def _strip_sql_comments(sql: str) -> str:
    """Strips leading SQL comments and surrounding whitespace."""
    cleaned = _LEADING_COMMENT_PATTERN.sub("", sql or "").strip()
    return cleaned[:-1].strip() if cleaned.endswith(";") else cleaned


def validate_readonly_sql(query_str: str) -> str:
    """Validates that a SQL query is strictly read-only (``SELECT`` or ``WITH``).

    Args:
        query_str: Raw SQL string from the SQL Lab or natural-language agent.

    Returns:
        Normalized SQL statement with trailing semicolon removed.

    Raises:
        ValueError: If the query is empty, contains multiple statements, does not start
            with ``SELECT`` / ``WITH``, or contains any DDL/DML/administrative keyword.
    """
    cleaned = _strip_sql_comments(query_str)
    if not cleaned:
        raise ValueError("SQL query must not be empty.")
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed in read-only mode.")
    first_token = cleaned.split(None, 1)[0].upper()
    if first_token not in {"SELECT", "WITH"}:
        raise ValueError(f"Only SELECT or WITH queries are permitted; got {first_token!r}.")
    if match := _FORBIDDEN_SQL_PATTERN.search(cleaned):
        raise ValueError(f"Forbidden SQL keyword detected: {match.group(1).upper()!r}.")
    return cleaned


def extract_funnel_kpis(funnel_df: pd.DataFrame) -> FunnelKPIs:
    """Extracts executive summary KPIs from a closed-funnel conversion DataFrame.

    Args:
        funnel_df: Output of :func:`get_funnel_conversion_summary`.

    Returns:
        Populated :class:`FunnelKPIs` dataclass.
    """
    if funnel_df.empty:
        return FunnelKPIs(0, 0, 0.0, "N/A", 0.0, 0.0, 0.0)
    sessions = int(funnel_df.loc[funnel_df["step_order"] == 1, "sessions_count"].iloc[0])
    purchase_row = funnel_df.loc[funnel_df["step_order"] == 5].iloc[0]
    purchases = int(purchase_row["sessions_count"])
    revenue = float(purchase_row["stage_revenue_usd"])
    overall_cvr = float(purchase_row["overall_conversion_pct"])
    aov = round(revenue / purchases, 2) if purchases > 0 else 0.0

    non_root = funnel_df.loc[funnel_df["step_order"] > 1]
    worst_idx = non_root["step_dropoff_pct"].idxmax()
    worst_stage = str(non_root.loc[worst_idx, "stage_name"])
    worst_drop = float(non_root.loc[worst_idx, "step_dropoff_pct"])
    return FunnelKPIs(
        total_sessions=sessions,
        total_purchases=purchases,
        overall_conversion_pct=overall_cvr,
        biggest_dropoff_stage=worst_stage,
        biggest_dropoff_pct=worst_drop,
        total_revenue_usd=round(revenue, 2),
        average_order_value_usd=aov,
    )


def get_funnel_conversion_summary(
    region: str | None = None,
    device: str | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    *,
    manager: DuckDBManager | None = None,
) -> pd.DataFrame:
    """Computes strict chronological closed-funnel conversion metrics across the 5 GA4 stages.

    Args:
        region: Optional ``geo_region`` filter (or ``'ALL'`` / ``None`` for all regions).
        device: Optional ``device_category`` filter (or ``'ALL'`` / ``None`` for all devices).
        start_date: Optional inclusive start date (``YYYY-MM-DD`` or :class:`datetime.date`).
        end_date: Optional inclusive end date (``YYYY-MM-DD`` or :class:`datetime.date`).
        manager: Optional :class:`DuckDBManager` override.

    Returns:
        DataFrame with columns ``step_order``, ``stage_name``, ``sessions_count``,
        ``drop_off_sessions``, ``overall_conversion_pct``, ``step_conversion_pct``,
        ``step_dropoff_pct``, and ``stage_revenue_usd``. Summary KPIs are attached in
        ``df.attrs["kpis"]``.
    """
    db = manager or get_duckdb_manager()
    params = {
        "region": region or "ALL",
        "device": device or "ALL",
        "start_date": str(start_date) if start_date is not None else None,
        "end_date": str(end_date) if end_date is not None else None,
    }
    started = time.perf_counter()
    df = db.query_df(CLOSED_FUNNEL_PARAMETERIZED_SQL, params)
    df.attrs["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    df.attrs["kpis"] = asdict(extract_funnel_kpis(df))
    return df


def get_dropoff_attribution_by_segment(
    dimension: str = "device_category",
    *,
    manager: DuckDBManager | None = None,
) -> pd.DataFrame:
    """Computes stage-by-stage funnel conversion and drop-off rates broken down by a dimension.

    Args:
        dimension: Column name to slice by; must be in :data:`ALLOWED_SEGMENT_DIMENSIONS`.
        manager: Optional :class:`DuckDBManager` override.

    Returns:
        DataFrame with per-segment funnel counts, step drop-off rates, conversion rate,
        and revenue at risk.

    Raises:
        ValueError: If ``dimension`` is not in :data:`ALLOWED_SEGMENT_DIMENSIONS`.
    """
    if dimension not in ALLOWED_SEGMENT_DIMENSIONS:
        raise ValueError(
            f"Unsupported segment dimension {dimension!r}; allowed: {ALLOWED_SEGMENT_DIMENSIONS}"
        )
    db = manager or get_duckdb_manager()
    sql = f"""
    WITH session_flags AS (
        SELECT
            user_pseudo_id,
            ga_session_id,
            COALESCE(ANY_VALUE({dimension}), '(not set)') AS segment_value,
            MIN(event_timestamp) FILTER (WHERE event_name = 'session_start') AS t1,
            MIN(event_timestamp) FILTER (WHERE event_name = 'view_item') AS t2,
            MIN(event_timestamp) FILTER (WHERE event_name = 'add_to_cart') AS t3,
            MIN(event_timestamp) FILTER (WHERE event_name = 'begin_checkout') AS t4,
            MIN(event_timestamp) FILTER (WHERE event_name = 'purchase') AS t5,
            SUM(
                CASE WHEN event_name = 'purchase' THEN COALESCE(event_value_in_usd, 0.0) ELSE 0.0 END
            ) AS purchase_revenue
        FROM v_ga4_events
        GROUP BY user_pseudo_id, ga_session_id
    ),
    strict_flags AS (
        SELECT
            segment_value,
            (t1 IS NOT NULL) AS s1,
            (t1 IS NOT NULL AND t2 >= t1) AS s2,
            (t1 IS NOT NULL AND t2 >= t1 AND t3 >= t2) AS s3,
            (t1 IS NOT NULL AND t2 >= t1 AND t3 >= t2 AND t4 >= t3) AS s4,
            (t1 IS NOT NULL AND t2 >= t1 AND t3 >= t2 AND t4 >= t3 AND t5 >= t4) AS s5,
            purchase_revenue
        FROM session_flags
        WHERE t1 IS NOT NULL
    )
    SELECT
        segment_value,
        COUNT(*) FILTER (WHERE s1) AS sessions,
        COUNT(*) FILTER (WHERE s2) AS viewed_item,
        COUNT(*) FILTER (WHERE s3) AS added_to_cart,
        COUNT(*) FILTER (WHERE s4) AS began_checkout,
        COUNT(*) FILTER (WHERE s5) AS purchased,
        ROUND(100.0 * (1.0 - COUNT(*) FILTER (WHERE s2) / NULLIF(COUNT(*) FILTER (WHERE s1), 0)), 2)
            AS browse_dropoff_pct,
        ROUND(100.0 * (1.0 - COUNT(*) FILTER (WHERE s3) / NULLIF(COUNT(*) FILTER (WHERE s2), 0)), 2)
            AS cart_dropoff_pct,
        ROUND(100.0 * (1.0 - COUNT(*) FILTER (WHERE s4) / NULLIF(COUNT(*) FILTER (WHERE s3), 0)), 2)
            AS checkout_dropoff_pct,
        ROUND(100.0 * (1.0 - COUNT(*) FILTER (WHERE s5) / NULLIF(COUNT(*) FILTER (WHERE s4), 0)), 2)
            AS payment_dropoff_pct,
        ROUND(100.0 * COUNT(*) FILTER (WHERE s5) / NULLIF(COUNT(*) FILTER (WHERE s1), 0), 2)
            AS purchase_conversion_pct,
        ROUND(SUM(purchase_revenue) FILTER (WHERE s5), 2) AS revenue_usd
    FROM strict_flags
    GROUP BY segment_value
    ORDER BY sessions DESC
    """
    return db.query_df(sql)


def get_cohort_retention_matrix(*, manager: DuckDBManager | None = None) -> pd.DataFrame:
    """Computes weekly user acquisition cohort retention rates across weeks since first visit."""
    db = manager or get_duckdb_manager()
    sql = """
    WITH user_weeks AS (
        SELECT DISTINCT user_pseudo_id, CAST(DATE_TRUNC('week', event_date) AS DATE) AS active_week
        FROM v_ga4_events
    ),
    first_week AS (
        SELECT user_pseudo_id, MIN(active_week) AS cohort_week
        FROM user_weeks GROUP BY user_pseudo_id
    ),
    cohort_activity AS (
        SELECT f.cohort_week, date_diff('week', f.cohort_week, u.active_week) AS weeks_since_first,
               COUNT(DISTINCT u.user_pseudo_id) AS retained_users
        FROM user_weeks u INNER JOIN first_week f USING (user_pseudo_id)
        GROUP BY f.cohort_week, weeks_since_first
    ),
    with_base AS (
        SELECT cohort_week, weeks_since_first,
               FIRST_VALUE(retained_users) OVER (
                   PARTITION BY cohort_week ORDER BY weeks_since_first ASC
               ) AS cohort_users,
               retained_users
        FROM cohort_activity
    )
    SELECT cohort_week, weeks_since_first, cohort_users, retained_users,
           ROUND(100.0 * retained_users / NULLIF(cohort_users, 0), 2) AS retention_pct
    FROM with_base
    ORDER BY cohort_week ASC, weeks_since_first ASC
    """
    return db.query_df(sql)


def execute_readonly_sql_lab(
    query_str: str,
    *,
    max_rows: int = 1000,
    manager: DuckDBManager | None = None,
) -> pd.DataFrame:
    """Validates and executes a read-only SQL query against the DuckDB analytical views.

    Args:
        query_str: SQL statement starting with ``SELECT`` or ``WITH``.
        max_rows: Maximum number of rows returned to protect memory.
        manager: Optional :class:`DuckDBManager` override.

    Returns:
        Query result DataFrame with ``elapsed_ms`` stored in ``df.attrs["elapsed_ms"]``.

    Raises:
        ValueError: If the query fails read-only validation or ``max_rows <= 0``.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    cleaned = validate_readonly_sql(query_str)
    db = manager or get_duckdb_manager()
    wrapped = f"SELECT * FROM ({cleaned}) AS _sql_lab_subquery LIMIT {int(max_rows)}"
    started = time.perf_counter()
    df = db.query_df(wrapped)
    df.attrs["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    return df


def list_sql_catalog() -> dict[str, SQLQuerySpec]:
    """Returns the immutable dictionary of showcase SQL queries."""
    return dict(SQL_CATALOG)
