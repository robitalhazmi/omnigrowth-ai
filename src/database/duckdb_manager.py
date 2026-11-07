"""Thread-safe, low-memory DuckDB connection manager.

Every connection created here is initialized with the mandatory guardrails from
``AGENTS.md``::

    SET memory_limit = '192MB';
    SET threads = 1;

The manager owns a single in-memory DuckDB database whose catalog contains only
zero-copy *views* over the compressed Parquet marts in ``data/processed/``. Nothing
is materialized into Pandas until a caller explicitly asks for a (small) result.
Each thread receives its own cursor (DuckDB cursors share the parent database,
its settings and its views, but are safe to use concurrently).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import duckdb
import pandas as pd

from src.config import DuckDBSettings, ProjectPaths, get_config

logger = logging.getLogger(__name__)


def create_connection(
    database: str = ":memory:",
    *,
    read_only: bool = False,
    settings: DuckDBSettings | None = None,
) -> duckdb.DuckDBPyConnection:
    """Opens a DuckDB connection with the mandatory memory guardrails applied.

    Args:
        database: Database path or ``":memory:"``.
        read_only: Whether to open a file-backed database read-only.
        settings: Optional guardrail settings; defaults to the global config.

    Returns:
        A configured DuckDB connection (``memory_limit`` and ``threads`` set).
    """
    settings = settings or get_config().duckdb
    conn = duckdb.connect(database=database, read_only=read_only)
    for statement in settings.init_statements():
        conn.execute(statement)
    return conn


def quote_path(path: Path) -> str:
    """Returns a safely single-quoted SQL string literal for a filesystem path."""
    return "'" + Path(path).as_posix().replace("'", "''") + "'"


def default_view_map(paths: ProjectPaths | None = None) -> dict[str, Path]:
    """Maps canonical view names to their backing Parquet marts.

    Args:
        paths: Optional project paths; defaults to the global config.

    Returns:
        Ordered mapping of ``view_name -> parquet_path``.
    """
    paths = paths or get_config().paths
    return {
        "v_ga4_events": paths.ga4_events,
        "v_regional_demand": paths.regional_demand,
        "v_ab_logs": paths.ab_experiment_logs,
        "v_mql_funnel": paths.mql_funnel,
    }


class DuckDBManager:
    """Process-wide singleton exposing guarded DuckDB access over Parquet views.

    Use :meth:`get_instance` instead of the constructor in application code.
    """

    _instance: ClassVar[DuckDBManager | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        view_map: Mapping[str, Path] | None = None,
        settings: DuckDBSettings | None = None,
    ) -> None:
        """Creates the manager and registers zero-copy Parquet views.

        Args:
            view_map: Optional ``view_name -> parquet_path`` mapping; defaults to
                :func:`default_view_map`.
            settings: Optional guardrail settings; defaults to the global config.
        """
        self._settings = settings or get_config().duckdb
        self._conn = create_connection(settings=self._settings)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._closed = False
        self._views: dict[str, Path] = {}
        self._missing: dict[str, Path] = {}
        self.register_views(view_map if view_map is not None else default_view_map())

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def get_instance(cls) -> DuckDBManager:
        """Returns (lazily creating) the process-wide manager singleton."""
        if cls._instance is None or cls._instance._closed:
            with cls._instance_lock:
                if cls._instance is None or cls._instance._closed:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Closes and discards the singleton (used by tests and after ETL rebuilds)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None

    def close(self) -> None:
        """Closes the underlying DuckDB connection."""
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    # ---------------------------------------------------------------------- views
    def register_views(self, view_map: Mapping[str, Path]) -> None:
        """Creates (or replaces) zero-copy views over Parquet files.

        Missing files are skipped and reported via :attr:`missing_views` so the app
        can degrade gracefully before the data pipeline has been run.

        Args:
            view_map: Mapping of ``view_name -> parquet_path``.

        Raises:
            ValueError: If a view name is not a safe SQL identifier.
        """
        with self._lock:
            for name, path in view_map.items():
                if not name.isidentifier():
                    raise ValueError(f"Unsafe view name: {name!r}")
                if not Path(path).exists():
                    self._missing[name] = Path(path)
                    logger.warning("Parquet mart missing for %s: %s", name, path)
                    continue
                self._conn.execute(
                    f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM read_parquet({quote_path(Path(path))})"
                )
                self._views[name] = Path(path)
                self._missing.pop(name, None)

    @property
    def available_views(self) -> list[str]:
        """Names of successfully registered views."""
        return list(self._views)

    @property
    def missing_views(self) -> dict[str, Path]:
        """Views whose backing Parquet file was not found."""
        return dict(self._missing)

    @property
    def settings(self) -> DuckDBSettings:
        """Guardrail settings applied to this manager's connection."""
        return self._settings

    # -------------------------------------------------------------------- queries
    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Returns this thread's dedicated cursor on the shared database.

        Raises:
            RuntimeError: If the manager has been closed.
        """
        if self._closed:
            raise RuntimeError("DuckDBManager is closed")
        cur = getattr(self._local, "cursor", None)
        if cur is None:
            with self._lock:
                cur = self._conn.cursor()
            self._local.cursor = cur
        return cur

    def query_df(self, sql: str, params: Sequence[Any] | Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Executes a query and returns the (small, aggregated) result as Pandas.

        Args:
            sql: SQL text, optionally with ``?`` or ``$name`` placeholders.
            params: Optional positional or named parameters.

        Returns:
            Query result as a DataFrame.
        """
        cur = self.cursor()
        return cur.execute(sql, params).df() if params is not None else cur.execute(sql).df()

    def query_scalar(self, sql: str, params: Sequence[Any] | Mapping[str, Any] | None = None) -> Any:
        """Executes a query and returns the first column of the first row.

        Args:
            sql: SQL text.
            params: Optional positional or named parameters.

        Returns:
            The scalar value, or ``None`` for an empty result.
        """
        cur = self.cursor()
        row = (cur.execute(sql, params) if params is not None else cur.execute(sql)).fetchone()
        return None if row is None else row[0]

    def current_settings(self) -> dict[str, str]:
        """Reports the live ``memory_limit`` and ``threads`` DuckDB settings."""
        rows = self.cursor().execute(
            "SELECT name, value FROM duckdb_settings() WHERE name IN ('memory_limit', 'threads')"
        ).fetchall()
        return {name: value for name, value in rows}


def get_duckdb_manager() -> DuckDBManager:
    """Convenience accessor for the process-wide :class:`DuckDBManager`."""
    return DuckDBManager.get_instance()
