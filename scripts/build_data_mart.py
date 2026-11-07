#!/usr/bin/env python3
"""DuckDB ETL: raw public datasets -> compressed Parquet serving marts.

Outputs (``data/processed/``, ZSTD Parquet, < 25 MB total):

* ``ga4_events.parquet``          — flattened GA4 ecommerce funnel events.
* ``regional_demand.parquet``     — dense daily region x segment Olist demand panel.
* ``ab_experiment_logs.parquet``  — cleaned A/B test with CUPED covariates.
* ``mql_funnel.parquet``          — Olist B2B merchant onboarding funnel.
* ``looker_executive_mart.csv``   — flat Google Looker Studio mart.
* ``data_lineage.json``           — provenance manifest for the UI lineage badge.

Every DuckDB connection uses the mandatory ``192MB`` / ``1`` thread guardrails.

Usage:
    python scripts/build_data_mart.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402
from src.database.duckdb_manager import create_connection, quote_path  # noqa: E402
from src.pipeline.ab_enrichment import ABEnrichmentConfig, enrich_ab_logs  # noqa: E402
from src.pipeline.looker_mart_sql import looker_mart_sql  # noqa: E402
from src.pipeline.mart_sql import (  # noqa: E402
    ab_clean_sql,
    ga4_events_sql,
    mql_funnel_sql,
    regional_demand_sql,
)
from src.pipeline.sources import GA4_RAW_FILENAME, RAW_MANIFEST_FILENAME, REMOTE_SOURCES  # noqa: E402

logger = logging.getLogger("build_data_mart")


def _raw_paths(raw_dir: Path) -> dict[str, Path]:
    """Resolves and validates every raw input file.

    Raises:
        FileNotFoundError: If any input is missing (run download_datasets.py first).
    """
    paths = {s.key: raw_dir / s.filename for s in REMOTE_SOURCES}
    paths["ga4_events"] = raw_dir / GA4_RAW_FILENAME
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw inputs (run scripts/download_datasets.py): {missing}")
    return paths


def _profile(conn: Any, path: Path) -> dict[str, Any]:
    """Returns row count, column count and size for a processed artifact."""
    reader = "read_parquet" if path.suffix == ".parquet" else "read_csv"
    rows, cols = conn.execute(
        f"SELECT COUNT(*), (SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM {reader}({quote_path(path)})))"
        f" FROM {reader}({quote_path(path)})"
    ).fetchone()
    return {"file": path.name, "rows": int(rows), "columns": int(cols), "size_bytes": path.stat().st_size}


def build_ab_logs(conn: Any, raw_csv: Path, out_parquet: Path, cfg: ABEnrichmentConfig) -> None:
    """Cleans the raw A/B CSV in DuckDB, enriches covariates and writes Parquet."""
    clean = conn.execute(ab_clean_sql(raw_csv)).df()
    enriched = enrich_ab_logs(clean, cfg)
    conn.register("enriched", enriched)
    conn.execute(
        f"""
        COPY (
          SELECT user_id, experiment_group, assigned_date::DATE AS assigned_date, device_os,
                 region_tier, pre_exp_sessions::SMALLINT AS pre_exp_sessions, pre_exp_spend,
                 converted::TINYINT AS converted, post_exp_revenue,
                 retention_d7::TINYINT AS retention_d7, retention_d30::TINYINT AS retention_d30
          FROM enriched ORDER BY assigned_date, user_id
        ) TO {quote_path(out_parquet)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
        """
    )
    conn.unregister("enriched")


def write_lineage(conn: Any, raw_dir: Path, outputs: list[Path], ab_cfg: ABEnrichmentConfig) -> dict:
    """Writes ``data_lineage.json`` combining raw provenance and processed profiles."""
    cfg = get_config()
    manifest_path = raw_dir / RAW_MANIFEST_FILENAME
    raw_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    lineage = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duckdb_guardrails": {"memory_limit": cfg.duckdb.memory_limit, "threads": cfg.duckdb.threads},
        "raw_sources": raw_manifest.get("files", []),
        "processed_artifacts": [_profile(conn, p) for p in outputs],
        "processed_total_mb": round(sum(p.stat().st_size for p in outputs) / 1e6, 3),
        "methodology_notes": {
            "ga4_events": "Flattened GA4 funnel events (BigQuery export, or schema-faithful "
                          "calibrated replica of the Google Merchandise Store sample when no "
                          "GCP credentials are available — see raw_sources[].mode).",
            "regional_demand": "Real Olist orders aggregated to a dense daily state x segment "
                               "panel; promo_discount_pct = real voucher share of order value; "
                               "cumulative_active_merchants = distinct sellers serving the region.",
            "ab_experiment_logs": "Real randomized assignment and conversion (ab_data.csv, "
                                  "misaligned rows removed, first exposure kept). CUPED covariates, "
                                  "revenue and retention are seeded imputations.",
            "ab_enrichment_config": ab_cfg.as_dict(),
            "mql_funnel": "Real Olist marketing-qualified leads joined to closed merchant deals.",
        },
    }
    cfg.paths.lineage_json.write_text(json.dumps(lineage, indent=2, default=str))
    return lineage


def main() -> int:
    """CLI entrypoint.

    Returns:
        Process exit code (0 on success, 1 if the 25 MB budget is exceeded).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    cfg = get_config()
    paths, pipe = cfg.paths, cfg.pipeline
    paths.ensure_directories()
    raw = _raw_paths(paths.raw_dir)
    ab_cfg = ABEnrichmentConfig(seed=cfg.random_seed)

    conn = create_connection(settings=cfg.duckdb)
    conn.execute("SET TimeZone = 'UTC'")
    conn.execute(f"SET temp_directory = {quote_path(paths.data_dir / '.duckdb_tmp')}")
    logger.info("DuckDB guardrails: %s", dict(conn.execute(
        "SELECT name, value FROM duckdb_settings() WHERE name IN ('memory_limit', 'threads')"
    ).fetchall()))

    steps: list[tuple[str, Path, Any]] = [
        ("ga4_events", paths.ga4_events,
         lambda: conn.execute(ga4_events_sql(raw["ga4_events"], paths.ga4_events))),
        ("regional_demand", paths.regional_demand,
         lambda: conn.execute(regional_demand_sql(
             raw, paths.regional_demand, pipe.demand_start_date, pipe.demand_end_date))),
        ("ab_experiment_logs", paths.ab_experiment_logs,
         lambda: build_ab_logs(conn, raw["ab_test"], paths.ab_experiment_logs, ab_cfg)),
        ("mql_funnel", paths.mql_funnel,
         lambda: conn.execute(mql_funnel_sql(raw, paths.mql_funnel))),
        ("looker_executive_mart", paths.looker_mart_csv,
         lambda: conn.execute(looker_mart_sql(paths.ga4_events, paths.regional_demand,
                                              paths.looker_mart_csv))),
    ]
    outputs: list[Path] = []
    for name, out, action in steps:
        started = time.perf_counter()
        action()
        outputs.append(out)
        logger.info("built %-22s %8.1f KB in %.2fs", name, out.stat().st_size / 1e3,
                    time.perf_counter() - started)

    lineage = write_lineage(conn, paths.raw_dir, outputs, ab_cfg)
    conn.close()
    total_mb = sum(p.stat().st_size for p in paths.processed_dir.iterdir() if p.is_file()) / 1e6
    for artifact in lineage["processed_artifacts"]:
        logger.info("  %-30s rows=%-8d cols=%-3d %8.1f KB", artifact["file"], artifact["rows"],
                    artifact["columns"], artifact["size_bytes"] / 1e3)
    logger.info("data/processed total: %.2f MB (budget %.0f MB)", total_mb, pipe.processed_budget_mb)
    if total_mb >= pipe.processed_budget_mb:
        logger.error("Processed artifacts exceed the %.0f MB budget!", pipe.processed_budget_mb)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
