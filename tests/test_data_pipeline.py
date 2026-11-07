"""Tests for the offline data pipeline: sources, generators, enrichment and ETL SQL."""

from __future__ import annotations

from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd
import pytest

from src.config import get_config
from src.database.duckdb_manager import create_connection, quote_path
from src.pipeline.ab_enrichment import AB_COLUMNS, enrich_ab_logs
from src.pipeline.downloader import DownloadError, download_source, validate_header
from src.pipeline.ga4_replica import GA4_COLUMNS, GA4ReplicaConfig, generate_ga4_events
from src.pipeline.looker_mart_sql import LOOKER_COLUMNS, looker_mart_sql
from src.pipeline.mart_sql import ab_clean_sql, ga4_events_sql, mql_funnel_sql, regional_demand_sql
from src.pipeline.offline_fallback import write_ab_fallback, write_olist_fallback
from src.pipeline.sources import REMOTE_SOURCES, RemoteSource

FUNNEL = ["session_start", "view_item", "add_to_cart", "begin_checkout", "add_payment_info", "purchase"]


@pytest.fixture(scope="module")
def ga4_small() -> pd.DataFrame:
    return generate_ga4_events(GA4ReplicaConfig(n_users=4_000, seed=7))


# --------------------------------------------------------------------------- GA4
def test_ga4_replica_schema_and_determinism(ga4_small: pd.DataFrame) -> None:
    assert tuple(ga4_small.columns) == GA4_COLUMNS
    again = generate_ga4_events(GA4ReplicaConfig(n_users=4_000, seed=7))
    pd.testing.assert_frame_equal(ga4_small, again)
    assert set(ga4_small["event_name"]) == set(FUNNEL)
    assert ga4_small["event_value_in_usd"].notna().sum() == (ga4_small["event_name"] == "purchase").sum()


def test_ga4_replica_funnel_is_monotonic_and_calibrated(ga4_small: pd.DataFrame) -> None:
    sessions = ga4_small.groupby("event_name")[["user_pseudo_id", "ga_session_id"]].apply(
        lambda g: len(g.drop_duplicates())
    )
    counts = [sessions[name] for name in FUNNEL]
    assert counts == sorted(counts, reverse=True)
    purchase_rate = counts[-1] / counts[0]
    assert 0.008 < purchase_rate < 0.035  # GA4 merch-store sample ~1.5-2% session conversion


def test_ga4_events_follow_session_chronology(ga4_small: pd.DataFrame) -> None:
    starts = ga4_small[ga4_small["event_name"] == "session_start"]
    session_epoch = pd.to_datetime(starts["ga_session_id"], unit="s")
    assert (starts["event_timestamp"].dt.floor("s") == session_epoch).mean() > 0.99
    first = ga4_small.groupby(["user_pseudo_id", "ga_session_id"])["event_timestamp"].min()
    start_ts = starts.set_index(["user_pseudo_id", "ga_session_id"])["event_timestamp"]
    assert (first.loc[start_ts.index] == start_ts).all()


def test_lookml_semantic_models_exist_and_define_required_fields() -> None:
    looker_dir = get_config().paths.looker_dir
    model_path = looker_dir / "omni_growth.model.lkml"
    view_path = looker_dir / "ga4_funnel.view.lkml"
    assert model_path.exists() and view_path.exists()

    model_text = model_path.read_text(encoding="utf-8")
    view_text = view_path.read_text(encoding="utf-8")
    assert 'include: "/looker/*.view.lkml"' in model_text
    assert "explore: ga4_funnel" in model_text
    for symbol in (
        "view: ga4_funnel",
        "dimension_group: event",
        "dimension: geo_region",
        "dimension: device_category",
        "dimension: traffic_source",
        "measure: session_starts",
        "measure: cart_additions",
        "measure: checkout_rate",
        "measure: purchase_conversion_rate",
        "measure: average_order_value",
    ):
        assert symbol in view_text, f"Missing LookML symbol: {symbol}"


# ---------------------------------------------------------------------------- A/B
def _clean_ab(n: int = 20_000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "user_id": rng.choice(np.arange(600_000, 1_000_000), n, replace=False),
        "timestamp": pd.Timestamp("2017-01-05") + pd.to_timedelta(rng.integers(0, 10**6, n), unit="s"),
        "group": np.where(rng.random(n) < 0.5, "control", "treatment"),
        "converted": (rng.random(n) < 0.12).astype(int),
    })


def test_ab_enrichment_schema_and_cuped_correlation() -> None:
    clean = _clean_ab()
    out = enrich_ab_logs(clean)
    assert tuple(out.columns) == AB_COLUMNS
    assert len(out) == len(clean)
    merged = out.merge(clean, on="user_id")
    assert (merged["converted_x"] == merged["converted_y"]).all()  # real outcome preserved
    assert np.corrcoef(out["pre_exp_spend"], out["post_exp_revenue"])[0, 1] > 0.3
    assert (out["retention_d30"] <= out["retention_d7"]).all()
    assert (out["pre_exp_spend"] >= 0).all() and (out["post_exp_revenue"] >= 0).all()
    pd.testing.assert_frame_equal(out, enrich_ab_logs(clean.sample(frac=1, random_state=3)))


def test_ab_enrichment_rejects_duplicates() -> None:
    clean = _clean_ab(100)
    with pytest.raises(ValueError):
        enrich_ab_logs(pd.concat([clean, clean.head(1)]))
    with pytest.raises(ValueError):
        enrich_ab_logs(clean.drop(columns="group"))


def test_ab_clean_sql_removes_misaligned_and_duplicates(tmp_path: Path) -> None:
    csv = tmp_path / "ab.csv"
    pd.DataFrame({
        "user_id": [1, 1, 2, 3, 4],
        "timestamp": ["2017-01-02 10:00:00", "2017-01-03 10:00:00", "2017-01-02 11:00:00",
                      "2017-01-02 12:00:00", "2017-01-02 13:00:00"],
        "group": ["control", "control", "treatment", "treatment", "control"],
        "landing_page": ["old_page", "old_page", "new_page", "old_page", "old_page"],
        "converted": [0, 1, 1, 0, 0],
    }).to_csv(csv, index=False)
    conn = create_connection()
    out = conn.execute(ab_clean_sql(csv)).df()
    conn.close()
    assert out["user_id"].tolist() == [1, 2, 4]
    assert out.loc[out["user_id"] == 1, "converted"].item() == 0  # first exposure kept


# ---------------------------------------------------------------------- downloads
def _source(urls: tuple[str, ...]) -> RemoteSource:
    return RemoteSource("demo", "demo.csv", urls, ("a", "b"), "demo", "demo")


def test_download_falls_back_to_mirror_and_validates(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "primary" in str(request.url):
            return httpx.Response(503)
        return httpx.Response(200, content=b"a,b\n1,2\n3,4\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download_source(
        _source(("https://primary/x.csv", "https://mirror/x.csv")), tmp_path,
        max_retries=2, backoff_base_s=0.0, client=client,
    )
    assert result.url == "https://mirror/x.csv" and result.rows == 2
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_wrong_schema(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x,y\n1,2\n")))
    with pytest.raises(DownloadError):
        download_source(_source(("https://only/x.csv",)), tmp_path, max_retries=1, client=client)
    assert not (tmp_path / "demo.csv").exists()


def test_source_catalog_is_complete() -> None:
    keys = {s.key for s in REMOTE_SOURCES}
    assert {"olist_orders", "olist_mql", "olist_closed_deals", "ab_test"} <= keys
    assert all(s.urls and s.required_columns for s in REMOTE_SOURCES)


# ------------------------------------------------------ end-to-end offline ETL SQL
def test_offline_fallback_feeds_the_full_etl(tmp_path: Path, ga4_small: pd.DataFrame) -> None:
    raw, out = tmp_path / "raw", tmp_path / "processed"
    out.mkdir()
    write_olist_fallback(raw, seed=3, n_orders=3_000)
    write_ab_fallback(raw, seed=3, n_users=2_000)
    for source in REMOTE_SOURCES:
        validate_header(raw / source.filename, source.required_columns)
    ga4_small.to_parquet(raw / "ga4_raw.parquet", index=False)
    paths = {s.key: raw / s.filename for s in REMOTE_SOURCES}

    conn = create_connection()
    conn.execute("SET TimeZone = 'UTC'")
    conn.execute(ga4_events_sql(raw / "ga4_raw.parquet", out / "ga4.parquet"))
    conn.execute(regional_demand_sql(paths, out / "demand.parquet", "2017-01-01", "2017-03-31"))
    conn.execute(mql_funnel_sql(paths, out / "mql.parquet"))
    conn.execute(looker_mart_sql(out / "ga4.parquet", out / "demand.parquet", out / "looker.csv"))
    clean = conn.execute(ab_clean_sql(raw / "ab_data.csv")).df()

    demand = conn.execute(f"SELECT * FROM read_parquet({quote_path(out / 'demand.parquet')})").df()
    looker = conn.execute(f"SELECT * FROM read_csv({quote_path(out / 'looker.csv')})").df()
    conn.close()

    assert demand["order_date"].nunique() == 90 and demand["region_code"].nunique() == 27
    assert (demand.groupby(["region_code", "product_segment"])["cumulative_active_merchants"]
            .apply(lambda s: s.is_monotonic_increasing)).all()
    assert demand["promo_discount_pct"].between(0, 100).all()
    assert tuple(looker.columns) == LOOKER_COLUMNS
    assert set(looker["metric_domain"]) == {"ga4_funnel", "regional_revenue", "cohort_retention"}
    assert len(clean) == 2_000


# ------------------------------------------------------------ processed artifacts
PROCESSED = get_config().paths


@pytest.mark.skipif(not PROCESSED.ga4_events.exists(), reason="run scripts/build_data_mart.py first")
def test_processed_marts_exist_within_budget() -> None:
    required = [PROCESSED.ga4_events, PROCESSED.regional_demand, PROCESSED.ab_experiment_logs,
                PROCESSED.looker_mart_csv]
    assert all(p.exists() for p in required)
    total_mb = sum(p.stat().st_size for p in PROCESSED.processed_dir.iterdir() if p.is_file()) / 1e6
    assert total_mb < get_config().pipeline.processed_budget_mb

    conn = duckdb.connect()
    ga4_cols = [r[0] for r in conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote_path(PROCESSED.ga4_events)})").fetchall()]
    ab_cols = [r[0] for r in conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote_path(PROCESSED.ab_experiment_logs)})").fetchall()]
    meta = conn.execute(
        f"SELECT DISTINCT compression FROM parquet_metadata({quote_path(PROCESSED.ga4_events)})"
    ).fetchall()
    conn.close()
    assert tuple(ga4_cols) == GA4_COLUMNS
    assert tuple(ab_cols) == AB_COLUMNS
    assert {m[0].upper() for m in meta} == {"ZSTD"}
