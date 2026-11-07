"""Tests for the low-memory DuckDB connection manager."""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb
import pytest

from src.config import DuckDBSettings, get_config
from src.database.duckdb_manager import (
    DuckDBManager,
    create_connection,
    default_view_map,
    quote_path,
)

EXPECTED_MEMORY = "183.1 MiB"  # DuckDB reports 192MB (decimal) in binary units


@pytest.fixture()
def parquet_dir(tmp_path: Path) -> Path:
    conn = duckdb.connect()
    conn.execute(
        f"COPY (SELECT range AS id, range % 3 AS grp FROM range(1000)) "
        f"TO {quote_path(tmp_path / 'demo.parquet')} (FORMAT PARQUET)"
    )
    conn.close()
    return tmp_path


def _settings(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    rows = conn.execute(
        "SELECT name, value FROM duckdb_settings() WHERE name IN ('memory_limit', 'threads')"
    ).fetchall()
    return dict(rows)


def test_create_connection_applies_guardrails() -> None:
    conn = create_connection()
    try:
        assert _settings(conn) == {"memory_limit": EXPECTED_MEMORY, "threads": "1"}
    finally:
        conn.close()


def test_lower_custom_limit_is_applied() -> None:
    conn = create_connection(settings=DuckDBSettings(memory_limit_mb=64))
    try:
        assert _settings(conn)["memory_limit"] == "61.0 MiB"
    finally:
        conn.close()


def test_views_are_zero_copy_and_missing_files_reported(parquet_dir: Path) -> None:
    manager = DuckDBManager(
        view_map={"v_demo": parquet_dir / "demo.parquet", "v_absent": parquet_dir / "nope.parquet"}
    )
    try:
        assert manager.available_views == ["v_demo"]
        assert "v_absent" in manager.missing_views
        assert manager.query_scalar("SELECT COUNT(*) FROM v_demo") == 1000
        kind = manager.query_scalar(
            "SELECT sql FROM duckdb_views() WHERE view_name = 'v_demo'"
        )
        assert "read_parquet" in kind  # a view, not a materialized table
        assert manager.query_scalar("SELECT COUNT(*) FROM duckdb_tables()") == 0
        df = manager.query_df("SELECT grp, COUNT(*) AS n FROM v_demo WHERE grp = ? GROUP BY grp", [1])
        assert df["n"].iloc[0] == 333
        assert manager.current_settings() == {"memory_limit": EXPECTED_MEMORY, "threads": "1"}
    finally:
        manager.close()


def test_thread_local_cursors_are_safe(parquet_dir: Path) -> None:
    manager = DuckDBManager(view_map={"v_demo": parquet_dir / "demo.parquet"})
    results: list[int] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(20):
                results.append(int(manager.query_scalar("SELECT SUM(id) FROM v_demo")))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    manager.close()
    assert not errors
    assert results == [sum(range(1000))] * 120


def test_unsafe_view_name_rejected(parquet_dir: Path) -> None:
    manager = DuckDBManager(view_map={})
    try:
        with pytest.raises(ValueError):
            manager.register_views({"x; DROP TABLE y": parquet_dir / "demo.parquet"})
    finally:
        manager.close()


def test_closed_manager_raises() -> None:
    manager = DuckDBManager(view_map={})
    manager.close()
    with pytest.raises(RuntimeError):
        manager.cursor()


def test_quote_path_escapes_quotes() -> None:
    assert quote_path(Path("/tmp/o'brien.parquet")) == "'/tmp/o''brien.parquet'"


def test_singleton_lifecycle() -> None:
    DuckDBManager.reset_instance()
    first = DuckDBManager.get_instance()
    assert DuckDBManager.get_instance() is first
    DuckDBManager.reset_instance()
    second = DuckDBManager.get_instance()
    assert second is not first
    DuckDBManager.reset_instance()


def test_default_view_map_points_at_processed_marts() -> None:
    views = default_view_map()
    assert set(views) == {"v_ga4_events", "v_regional_demand", "v_ab_logs", "v_mql_funnel"}
    assert all(p.parent == get_config().paths.processed_dir for p in views.values())
