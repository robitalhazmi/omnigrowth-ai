"""Unit tests for Phase 2: Advanced DuckDB SQL Catalog & GA4 Funnel Analytics."""

from __future__ import annotations

import time

import pytest

from src.analytics.ga4_funnel import (
    ALLOWED_SEGMENT_DIMENSIONS,
    execute_readonly_sql_lab,
    extract_funnel_kpis,
    get_cohort_retention_matrix,
    get_dropoff_attribution_by_segment,
    get_funnel_conversion_summary,
    list_sql_catalog,
)
from src.analytics.sql_queries import (
    CLOSED_FUNNEL_SQL,
    PURCHASE_VELOCITY_SQL,
    RFM_ROLLING_REVENUE_SQL,
    SQL_CATALOG,
)
from src.database.duckdb_manager import get_duckdb_manager


def test_duckdb_memory_and_thread_guardrails_respected() -> None:
    manager = get_duckdb_manager()
    settings = manager.current_settings()
    assert settings["memory_limit"] == "183.1 MiB"
    assert settings["threads"] == "1"


def test_sql_catalog_contains_required_advanced_constructs() -> None:
    catalog = list_sql_catalog()
    assert set(catalog) == {"closed_funnel", "purchase_velocity", "rfm_rolling_revenue"}

    assert "WITH " in CLOSED_FUNNEL_SQL.upper()
    assert "ROW_NUMBER() OVER" in CLOSED_FUNNEL_SQL.upper()
    assert "LAG(" in PURCHASE_VELOCITY_SQL.upper()
    assert "LEAD(" in PURCHASE_VELOCITY_SQL.upper()
    assert "NTILE(4) OVER" in RFM_ROLLING_REVENUE_SQL.upper()
    assert "ROWS BETWEEN 6 PRECEDING AND CURRENT ROW" in RFM_ROLLING_REVENUE_SQL.upper()


@pytest.mark.parametrize("query_key", list(SQL_CATALOG.keys()))
def test_catalog_queries_execute_under_200ms_and_return_rows(query_key: str) -> None:
    manager = get_duckdb_manager()
    spec = SQL_CATALOG[query_key]
    # Warm-up metadata scan so OS page cache is hot before timing.
    _ = manager.query_df(spec.sql)

    started = time.perf_counter()
    df = manager.query_df(spec.sql)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert not df.empty, f"Catalog query {query_key!r} returned an empty DataFrame"
    assert elapsed_ms < 200.0, f"Catalog query {query_key!r} took {elapsed_ms:.1f} ms (>= 200 ms)"


def test_closed_funnel_enforces_monotonic_session_counts() -> None:
    df = get_funnel_conversion_summary()
    assert list(df["stage_name"]) == [
        "session_start",
        "view_item",
        "add_to_cart",
        "begin_checkout",
        "purchase",
    ]
    counts = df["sessions_count"].tolist()
    assert counts[0] > 0
    assert counts == sorted(counts, reverse=True), f"Non-monotonic funnel counts: {counts}"
    assert (df["step_conversion_pct"].between(0.0, 100.0)).all()
    assert (df["overall_conversion_pct"].between(0.0, 100.0)).all()

    kpis = extract_funnel_kpis(df)
    assert kpis.total_sessions == counts[0]
    assert kpis.total_purchases == counts[-1]
    assert kpis.total_revenue_usd > 0.0
    assert kpis.average_order_value_usd > 0.0
    assert df.attrs["kpis"]["total_sessions"] == counts[0]


def test_closed_funnel_supports_region_device_and_date_filters() -> None:
    df = get_funnel_conversion_summary(
        region="California",
        device="desktop",
        start_date="2020-11-15",
        end_date="2020-12-15",
    )
    counts = df["sessions_count"].tolist()
    assert len(counts) == 5
    assert counts[0] > 0
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("dimension", ["device_category", "geo_region", "traffic_source"])
def test_dropoff_attribution_by_segment(dimension: str) -> None:
    df = get_dropoff_attribution_by_segment(dimension)
    assert not df.empty
    for _, row in df.iterrows():
        stage_counts = [
            row["sessions"],
            row["viewed_item"],
            row["added_to_cart"],
            row["began_checkout"],
            row["purchased"],
        ]
        assert stage_counts == sorted(stage_counts, reverse=True)


def test_dropoff_attribution_rejects_invalid_dimension() -> None:
    assert "device_category" in ALLOWED_SEGMENT_DIMENSIONS
    with pytest.raises(ValueError, match="Unsupported segment dimension"):
        get_dropoff_attribution_by_segment("user_pseudo_id; DROP TABLE v_ga4_events")


def test_cohort_retention_matrix_structure_and_week0_baseline() -> None:
    df = get_cohort_retention_matrix()
    assert not df.empty
    assert {"cohort_week", "weeks_since_first", "cohort_users", "retained_users", "retention_pct"} <= set(
        df.columns
    )
    week_zero = df[df["weeks_since_first"] == 0]
    assert not week_zero.empty
    assert (week_zero["retention_pct"] == 100.0).all()
    assert (df["retention_pct"].between(0.0, 100.0)).all()


def test_execute_readonly_sql_lab_allows_select_and_with() -> None:
    df = execute_readonly_sql_lab(
        "WITH t AS (SELECT device_category, COUNT(*) AS n FROM v_ga4_events GROUP BY 1) "
        "SELECT * FROM t ORDER BY n DESC;"
    )
    assert len(df) == 3
    assert "elapsed_ms" in df.attrs


@pytest.mark.parametrize(
    "bad_query",
    [
        "",
        "DROP VIEW v_ga4_events",
        "SELECT 1; DROP VIEW v_ga4_events",
        "WITH x AS (SELECT 1) COPY x TO '/tmp/out.csv'",
        "ALTER TABLE v_ga4_events RENAME TO x",
        "INSTALL httpfs",
    ],
)
def test_execute_readonly_sql_lab_blocks_unsafe_sql(bad_query: str) -> None:
    with pytest.raises(ValueError):
        execute_readonly_sql_lab(bad_query)
