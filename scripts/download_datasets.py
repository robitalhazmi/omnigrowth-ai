#!/usr/bin/env python3
"""Fetches the three public benchmark datasets into ``data/raw/``.

1. **Olist Brazilian E-Commerce + Marketing Funnel** (real CSVs from GitHub mirrors).
2. **E-Commerce landing-page A/B test** (real ``ab_data.csv``).
3. **GA4 Google Merchandise Store export** — exported from BigQuery when
   ``google-cloud-bigquery`` and credentials are available, otherwise produced by
   the deterministic calibrated replica in :mod:`src.pipeline.ga4_replica`.

If a download fails on every mirror (or ``--offline`` is passed) a seeded,
schema-identical fallback is written instead. Provenance for every file is
recorded in ``data/raw/_manifest.json``.

Usage:
    python scripts/download_datasets.py [--offline] [--force] [--ga4-source auto|bigquery|replica]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402
from src.pipeline.downloader import (  # noqa: E402
    DownloadError,
    count_csv_rows,
    download_source,
    file_sha256,
    validate_header,
)
from src.pipeline.ga4_replica import FUNNEL_EVENTS, GA4ReplicaConfig, generate_ga4_events  # noqa: E402
from src.pipeline.offline_fallback import write_ab_fallback, write_olist_fallback  # noqa: E402
from src.pipeline.sources import (  # noqa: E402
    GA4_DATASET,
    GA4_RAW_FILENAME,
    RAW_MANIFEST_FILENAME,
    REMOTE_SOURCES,
    RemoteSource,
)

logger = logging.getLogger("download_datasets")

GA4_BIGQUERY_SQL = """
SELECT
  PARSE_DATE('%Y%m%d', event_date) AS event_date,
  TIMESTAMP_MICROS(event_timestamp) AS event_timestamp,
  event_name,
  user_pseudo_id,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS ga_session_id,
  device.category AS device_category,
  traffic_source.source AS traffic_source,
  traffic_source.medium AS traffic_medium,
  geo.region AS geo_region,
  (SELECT i.item_category FROM UNNEST(items) AS i LIMIT 1) AS item_category,
  event_value_in_usd
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  AND event_name IN UNNEST(@events)
  AND MOD(ABS(FARM_FINGERPRINT(user_pseudo_id)), 100) < @sample_pct
"""


def _record(source: RemoteSource, path: Path, mode: str, url: str | None) -> dict[str, Any]:
    """Builds one manifest entry for a CSV file."""
    return {
        "key": source.key,
        "file": path.name,
        "dataset": source.dataset,
        "license": source.license,
        "mode": mode,
        "url": url,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "rows": count_csv_rows(path),
    }


def fetch_csv_sources(raw_dir: Path, *, offline: bool, force: bool) -> list[dict[str, Any]]:
    """Downloads every CSV source, falling back to seeded replicas per dataset family.

    Args:
        raw_dir: Destination directory.
        offline: Skip the network entirely and write fallbacks.
        force: Re-download even if a valid cached file exists.

    Returns:
        Manifest entries for all CSV files.
    """
    cfg = get_config()
    entries: dict[str, dict[str, Any]] = {}
    failed_families: set[str] = set()
    for source in REMOTE_SOURCES:
        family = "ab" if source.key == "ab_test" else "olist"
        dest = raw_dir / source.filename
        if offline:
            failed_families.add(family)
            continue
        if dest.exists() and not force:
            try:
                validate_header(dest, source.required_columns)
                entries[source.key] = _record(source, dest, "cached_download", source.urls[0])
                logger.info("cached     %-28s %8.1f KB", source.key, dest.stat().st_size / 1e3)
                continue
            except ValueError:
                logger.warning("cached file %s is invalid; re-downloading", dest.name)
        try:
            result = download_source(
                source, raw_dir,
                timeout_s=cfg.pipeline.http_timeout_s, max_retries=cfg.pipeline.http_max_retries,
            )
            entries[source.key] = _record(source, result.path, "downloaded", result.url)
            logger.info("downloaded %-28s %8.1f KB", source.key, result.size_bytes / 1e3)
        except DownloadError as exc:
            logger.error("%s", exc)
            failed_families.add(family)

    if "olist" in failed_families:
        logger.warning("Writing deterministic Olist offline fallback (schema-identical)")
        write_olist_fallback(raw_dir, seed=cfg.random_seed)
    if "ab" in failed_families:
        logger.warning("Writing deterministic A/B offline fallback (schema-identical)")
        write_ab_fallback(raw_dir, seed=cfg.random_seed)
    for source in REMOTE_SOURCES:
        family = "ab" if source.key == "ab_test" else "olist"
        if family in failed_families:
            entries[source.key] = _record(
                source, raw_dir / source.filename, "synthetic_offline_fallback", None
            )
    return [entries[s.key] for s in REMOTE_SOURCES]


def _export_ga4_bigquery(dest: Path, sample_pct: int) -> int:
    """Exports a user-sampled, flattened GA4 extract from BigQuery to Parquet.

    Args:
        dest: Destination Parquet path.
        sample_pct: Percentage of users (hash-sampled) to export.

    Returns:
        Number of exported rows.
    """
    from google.cloud import bigquery  # type: ignore[import-not-found]  # optional dependency

    client = bigquery.Client()
    job = client.query(
        GA4_BIGQUERY_SQL,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("events", "STRING", list(FUNNEL_EVENTS)),
                bigquery.ScalarQueryParameter("sample_pct", "INT64", sample_pct),
            ]
        ),
    )
    frame = job.result().to_dataframe()
    frame.to_parquet(dest, index=False, compression="zstd")
    return len(frame)


def fetch_ga4(raw_dir: Path, *, source_mode: str, n_users: int, sample_pct: int) -> dict[str, Any]:
    """Produces ``ga4_events_raw.parquet`` from BigQuery or the calibrated replica.

    Args:
        raw_dir: Destination directory.
        source_mode: ``auto`` (BigQuery if credentials exist), ``bigquery`` or ``replica``.
        n_users: Number of users for the replica.
        sample_pct: User sampling percentage for the BigQuery export.

    Returns:
        Manifest entry for the GA4 file.
    """
    cfg = get_config()
    dest = raw_dir / GA4_RAW_FILENAME
    has_creds = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT"))
    if source_mode == "bigquery" or (source_mode == "auto" and has_creds):
        try:
            rows = _export_ga4_bigquery(dest, sample_pct)
            logger.info("exported   ga4 from BigQuery (%d rows)", rows)
            return {"key": "ga4_events", "file": dest.name, "dataset": GA4_DATASET,
                    "license": "Google public dataset terms", "mode": "bigquery_export",
                    "url": "bigquery-public-data.ga4_obfuscated_sample_ecommerce",
                    "sha256": file_sha256(dest), "size_bytes": dest.stat().st_size, "rows": rows,
                    "sample_pct_users": sample_pct}
        except Exception as exc:  # noqa: BLE001 - any BigQuery failure -> deterministic replica
            logger.warning("BigQuery export unavailable (%s); using calibrated replica", exc)

    replica_cfg = GA4ReplicaConfig(n_users=n_users, seed=cfg.random_seed)
    frame = generate_ga4_events(replica_cfg)
    frame.to_parquet(dest, index=False, compression="zstd")
    logger.info("generated  ga4 calibrated replica (%d events, %d users)", len(frame), n_users)
    return {"key": "ga4_events", "file": dest.name, "dataset": GA4_DATASET,
            "license": "Schema-faithful calibrated replica (no raw GA4 rows redistributed)",
            "mode": "calibrated_replica", "url": None, "sha256": file_sha256(dest),
            "size_bytes": dest.stat().st_size, "rows": len(frame),
            "generator": {"n_users": n_users, "seed": cfg.random_seed,
                          "start_date": str(replica_cfg.start_date), "end_date": str(replica_cfg.end_date)}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="skip network; write seeded fallbacks")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--ga4-source", choices=("auto", "bigquery", "replica"), default="auto")
    parser.add_argument("--ga4-users", type=int, default=cfg.pipeline.ga4_n_users)
    parser.add_argument("--ga4-sample-pct", type=int, default=10, help="BigQuery user sample %%")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    args = parse_args(argv)
    cfg = get_config()
    cfg.paths.ensure_directories()
    raw_dir = cfg.paths.raw_dir

    entries = fetch_csv_sources(raw_dir, offline=args.offline, force=args.force)
    entries.append(fetch_ga4(raw_dir, source_mode="replica" if args.offline else args.ga4_source,
                             n_users=args.ga4_users, sample_pct=args.ga4_sample_pct))
    manifest = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "files": entries}
    (raw_dir / RAW_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    total_mb = sum(e["size_bytes"] for e in entries) / 1e6
    logger.info("raw datasets ready: %d files, %.1f MB -> %s", len(entries), total_mb, raw_dir)
    for entry in entries:
        logger.info("  %-28s %-28s rows=%d", entry["key"], entry["mode"], entry["rows"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
