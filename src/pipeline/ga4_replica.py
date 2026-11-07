"""Calibrated GA4 ecommerce event replica (deterministic, seeded).

The authoritative source for the GA4 mart is the BigQuery public dataset
``bigquery-public-data.ga4_obfuscated_sample_ecommerce`` (Google Merchandise Store,
2020-11-01 .. 2021-01-31). Exporting it requires Google Cloud credentials, so when
BigQuery is unavailable this module produces a deterministic replica that follows
the *same flattened schema* and is calibrated to the published characteristics of
that export: device mix, traffic-source mix (including GA4's ``(data deleted)`` /
``<Other>`` obfuscation tokens), merchandise categories, Black Friday / Cyber Monday
and holiday-season seasonality, GA4-style ``ga_session_id`` (session start epoch
seconds) and a ~1.5-2% session purchase rate. A small share of out-of-order
"quick add" sessions is injected so strict closed-funnel SQL is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

GA4_COLUMNS: tuple[str, ...] = (
    "event_date", "event_timestamp", "event_name", "user_pseudo_id", "ga_session_id",
    "device_category", "traffic_source", "traffic_medium", "geo_region", "item_category",
    "event_value_in_usd",
)
FUNNEL_EVENTS: tuple[str, ...] = (
    "session_start", "view_item", "add_to_cart", "begin_checkout", "add_payment_info", "purchase",
)

_DEVICES = (("desktop", 0.57, 1.15), ("mobile", 0.41, 0.80), ("tablet", 0.02, 0.90))
_TRAFFIC = (  # source, medium, share, conversion multiplier
    ("google", "organic", 0.30, 1.00),
    ("(direct)", "(none)", 0.22, 1.10),
    ("(data deleted)", "(data deleted)", 0.18, 1.00),
    ("<Other>", "referral", 0.14, 0.85),
    ("shop.googlemerchandisestore.com", "referral", 0.10, 1.30),
    ("google", "cpc", 0.04, 1.15),
    ("newsletter", "email", 0.02, 1.40),
)
_REGIONS = (
    ("California", 0.17), ("New York", 0.08), ("Texas", 0.06), ("Illinois", 0.05),
    ("Washington", 0.04), ("Ontario", 0.05), ("England", 0.06), ("Karnataka", 0.05),
    ("Maharashtra", 0.04), ("Other Regions", 0.40),
)
_CATEGORIES = (  # category, share, average-order-value multiplier
    ("Apparel", 0.36, 1.00), ("New", 0.10, 0.95), ("Bags", 0.07, 1.35),
    ("Drinkware", 0.08, 0.80), ("Office", 0.08, 0.60), ("Lifestyle", 0.08, 0.85),
    ("Campus Collection", 0.06, 0.90), ("Shop by Brand", 0.07, 1.05),
    ("Electronics Accessories", 0.05, 1.20), ("Stationery", 0.05, 0.50),
)


@dataclass(frozen=True)
class GA4ReplicaConfig:
    """Generation parameters for the GA4 replica."""

    start_date: date = date(2020, 11, 1)
    end_date: date = date(2021, 1, 31)
    n_users: int = 90_000
    seed: int = 42
    mean_extra_sessions: float = 0.55
    mean_return_gap_days: float = 9.0


def _day_weights(days: list[date]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (traffic weight, purchase-propensity multiplier) per calendar day."""
    traffic, propensity = [], []
    for d in days:
        t = 1.15 if d.weekday() < 5 else 0.80
        p = 1.0
        if date(2020, 11, 23) <= d <= date(2020, 12, 2):  # Black Friday / Cyber week
            t, p = t * 1.5, 1.3
        if d in (date(2020, 11, 27), date(2020, 11, 30)):  # Black Friday, Cyber Monday
            t, p = t * 1.35, 1.45
        elif date(2020, 12, 3) <= d <= date(2020, 12, 18):  # gifting before ship cut-off
            t, p = t * 1.3, 1.2
        elif date(2020, 12, 19) <= d <= date(2021, 1, 1):  # holiday lull
            t, p = t * 0.65, 0.75
        elif d.month == 1:
            t, p = t * 0.85, 0.9
        traffic.append(t)
        propensity.append(p)
    w = np.asarray(traffic)
    return w / w.sum(), np.asarray(propensity)


def _unique_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draws ``n`` unique 10-digit integers (GA4 client-id prefix)."""
    ids = rng.integers(1_000_000_000, 2_147_483_647, size=n)
    while True:
        _, first_idx = np.unique(ids, return_index=True)
        dup = np.setdiff1d(np.arange(n), first_idx)
        if dup.size == 0:
            return ids
        ids[dup] = rng.integers(1_000_000_000, 2_147_483_647, size=dup.size)


def _pick(rng: np.random.Generator, table: tuple, n: int, share_col: int = 1) -> np.ndarray:
    """Samples row indices of a lookup table proportionally to its share column."""
    shares = np.asarray([row[share_col] for row in table], dtype=float)
    return rng.choice(len(table), size=n, p=shares / shares.sum())


def generate_ga4_events(cfg: GA4ReplicaConfig | None = None) -> pd.DataFrame:
    """Generates the deterministic GA4 replica event table.

    Args:
        cfg: Generation parameters; defaults to :class:`GA4ReplicaConfig`.

    Returns:
        Event-level DataFrame with the columns in :data:`GA4_COLUMNS`.
    """
    cfg = cfg or GA4ReplicaConfig()
    rng = np.random.default_rng(cfg.seed)
    n_days = (cfg.end_date - cfg.start_date).days + 1
    days = [cfg.start_date + timedelta(days=i) for i in range(n_days)]
    traffic_w, propensity = _day_weights(days)
    epoch0 = pd.Timestamp(cfg.start_date).timestamp()
    end_ts = epoch0 + n_days * 86_400

    # ---- users ---------------------------------------------------------------
    nu = cfg.n_users
    u_device, u_traffic = _pick(rng, _DEVICES, nu), _pick(rng, _TRAFFIC, nu, 2)
    u_region = _pick(rng, _REGIONS, nu)
    first_day = rng.choice(n_days, size=nu, p=traffic_w)
    first_ts = epoch0 + first_day * 86_400 + rng.beta(2.4, 1.9, size=nu) * 86_400
    engagement = rng.gamma(0.6, cfg.mean_extra_sessions / 0.6, size=nu)
    n_sess = 1 + np.minimum(rng.poisson(engagement), 12)
    pseudo_ids = np.char.add(
        np.char.add(_unique_ids(rng, nu).astype(str), "."), first_ts.astype(np.int64).astype(str)
    )

    # ---- sessions ------------------------------------------------------------
    s_user = np.repeat(np.arange(nu), n_sess)
    starts = np.repeat(np.cumsum(n_sess) - n_sess, n_sess)
    pos = np.arange(s_user.size) - starts
    gaps = np.where(pos == 0, 0.0, rng.exponential(cfg.mean_return_gap_days, s_user.size) * 86_400 + 1_800)
    cum = np.cumsum(gaps)
    s_ts = first_ts[s_user] + cum - cum[starts]
    keep = s_ts < end_ts - 600
    s_user, pos, s_ts = s_user[keep], pos[keep], s_ts[keep]
    ns = s_user.size
    s_day = ((s_ts - epoch0) // 86_400).astype(int)
    s_cat = _pick(rng, _CATEGORIES, ns)

    dev_mult = np.asarray([d[2] for d in _DEVICES])[u_device[s_user]]
    trf_mult = np.asarray([t[3] for t in _TRAFFIC])[u_traffic[s_user]]
    mult = dev_mult * trf_mult * np.where(pos > 0, 1.35, 1.0) * propensity[s_day]

    viewed = rng.random(ns) < np.clip(0.36 * np.where(pos > 0, 1.05, 1.0), 0, 0.95)
    quick_add = ~viewed & (rng.random(ns) < 0.025)
    carted = (viewed & (rng.random(ns) < np.clip(0.15 * mult, 0, 0.95))) | quick_add
    checkout = carted & (rng.random(ns) < 0.52)
    payment = checkout & (rng.random(ns) < 0.62)
    purchased = payment & (rng.random(ns) < np.clip(0.70 * np.sqrt(mult), 0, 0.97))

    # ---- events --------------------------------------------------------------
    frames: list[tuple[np.ndarray, np.ndarray, str]] = [(np.arange(ns), s_ts, "session_start")]
    v_sess = np.flatnonzero(viewed)
    n_views = 1 + rng.poisson(1.3, v_sess.size)
    v_idx = np.repeat(v_sess, n_views)
    v_gap = rng.exponential(40.0, v_idx.size) + 5.0
    v_cum = np.cumsum(v_gap)
    v_first = np.repeat(np.cumsum(n_views) - n_views, n_views)
    v_ts = s_ts[v_idx] + v_cum - v_cum[v_first] + v_gap[v_first]
    frames.append((v_idx, v_ts, "view_item"))

    step_ts = s_ts.copy()
    if v_sess.size:  # views are chronological within a session -> last one is the max
        step_ts[v_sess] = v_ts[np.cumsum(n_views) - 1]
    for mask, name, mean_s in (
        (carted, "add_to_cart", 60.0), (checkout, "begin_checkout", 90.0),
        (payment, "add_payment_info", 60.0), (purchased, "purchase", 45.0),
    ):
        step_ts = step_ts + rng.exponential(mean_s, ns) + 5.0
        idx = np.flatnonzero(mask)
        frames.append((idx, step_ts[idx], name))

    sess_idx = np.concatenate([f[0] for f in frames])
    ts = np.concatenate([f[1] for f in frames])
    names = np.concatenate([np.full(f[0].size, f[2]) for f in frames])
    user = s_user[sess_idx]

    aov_mult = np.asarray([c[2] for c in _CATEGORIES])[s_cat[sess_idx]]
    is_purchase = names == "purchase"
    value = np.where(
        is_purchase,
        np.round(rng.lognormal(np.log(55.0), 0.75, ts.size) * aov_mult * np.sqrt(propensity[s_day[sess_idx]]), 2),
        np.nan,
    )
    category = np.asarray([c[0] for c in _CATEGORIES], dtype=object)[s_cat[sess_idx]]
    category[names == "session_start"] = None

    event_ts = pd.to_datetime(np.round(ts * 1e6).astype(np.int64), unit="us")
    df = pd.DataFrame(
        {
            "event_date": event_ts.normalize().date,
            "event_timestamp": event_ts,
            "event_name": names,
            "user_pseudo_id": pseudo_ids[user],
            "ga_session_id": s_ts[sess_idx].astype(np.int64),
            "device_category": np.asarray([d[0] for d in _DEVICES])[u_device[user]],
            "traffic_source": np.asarray([t[0] for t in _TRAFFIC])[u_traffic[user]],
            "traffic_medium": np.asarray([t[1] for t in _TRAFFIC])[u_traffic[user]],
            "geo_region": np.asarray([r[0] for r in _REGIONS])[u_region[user]],
            "item_category": category,
            "event_value_in_usd": value,
        }
    )
    df = df[df["event_timestamp"] < pd.Timestamp(cfg.end_date + timedelta(days=1))]
    return df.sort_values(["event_timestamp", "user_pseudo_id"], kind="stable").reset_index(drop=True)
