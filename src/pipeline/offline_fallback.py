"""Deterministic offline fallbacks for the public CSV datasets.

Used only when every download mirror is unreachable (air-gapped laptop, CI
without network). Each generator writes a CSV with the *same filename and the
columns consumed by* ``scripts/build_data_mart.py``, so the ETL has a single code
path. Outputs are seeded and flagged as ``synthetic_offline_fallback`` in the
raw manifest and the processed lineage file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_STATES = (
    ("SP", 0.420), ("RJ", 0.130), ("MG", 0.117), ("RS", 0.055), ("PR", 0.051), ("SC", 0.036),
    ("BA", 0.034), ("DF", 0.021), ("ES", 0.020), ("GO", 0.020), ("PE", 0.017), ("CE", 0.013),
    ("PA", 0.010), ("MT", 0.009), ("MA", 0.007), ("MS", 0.007), ("PB", 0.005), ("PI", 0.005),
    ("RN", 0.005), ("AL", 0.004), ("SE", 0.003), ("TO", 0.003), ("RO", 0.0025), ("AM", 0.0015),
    ("AC", 0.0008), ("AP", 0.0007), ("RR", 0.0005),
)
_CATEGORIES = (
    ("cama_mesa_banho", "bed_bath_table"), ("beleza_saude", "health_beauty"),
    ("esporte_lazer", "sports_leisure"), ("moveis_decoracao", "furniture_decor"),
    ("informatica_acessorios", "computers_accessories"), ("utilidades_domesticas", "housewares"),
    ("relogios_presentes", "watches_gifts"), ("telefonia", "telephony"),
    ("ferramentas_jardim", "garden_tools"), ("automotivo", "auto"), ("brinquedos", "toys"),
    ("cool_stuff", "cool_stuff"), ("perfumaria", "perfumery"), ("bebes", "baby"),
    ("eletronicos", "electronics"), ("papelaria", "stationery"),
    ("fashion_bolsas_e_acessorios", "fashion_bags_accessories"), ("pet_shop", "pet_shop"),
)
_MQL_ORIGINS = (
    ("organic_search", 0.29), ("paid_search", 0.20), ("social", 0.17), ("unknown", 0.14),
    ("direct_traffic", 0.06), ("email", 0.06), ("referral", 0.03), ("other", 0.05),
)
_SEGMENTS = (
    "home_decor", "health_beauty", "car_accessories", "household_utilities",
    "construction_tools_house_garden", "audio_video_electronics", "computers", "pet", "food_supplement",
)


def _hex_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """Returns ``n`` unique 32-character hex identifiers (Olist style)."""
    raw = rng.integers(0, 2**63 - 1, size=(n, 2), dtype=np.int64)
    ids = np.char.add(
        np.char.zfill(np.vectorize(lambda v: format(v, "x"))(raw[:, 0]), 16),
        np.char.zfill(np.vectorize(lambda v: format(v, "x"))(raw[:, 1]), 16),
    )
    return ids


def _logistic(t: np.ndarray, mid: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (t - mid)))


def write_olist_fallback(raw_dir: Path, seed: int = 42, n_orders: int = 60_000) -> list[Path]:
    """Writes seeded Olist-schema CSVs (orders, items, customers, payments, products, MQL).

    Args:
        raw_dir: Destination directory.
        seed: RNG seed.
        n_orders: Number of orders to generate.

    Returns:
        Paths of the written files.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(date(2016, 10, 1), date(2018, 8, 31), freq="D")
    t = np.arange(days.size, dtype=float)
    weights = _logistic(t, days.size * 0.45, 0.012) * np.where(days.dayofweek < 5, 1.1, 0.85)
    weights[days == pd.Timestamp("2017-11-24")] *= 4.0
    order_day = rng.choice(days.size, n_orders, p=weights / weights.sum())
    ts = days[order_day] + pd.to_timedelta(rng.integers(0, 86_400, n_orders), unit="s")

    order_ids, customer_ids = _hex_ids(rng, n_orders), _hex_ids(rng, n_orders)
    unique_pool = _hex_ids(rng, int(n_orders * 0.97))
    states = rng.choice(len(_STATES), n_orders, p=np.asarray([w for _, w in _STATES]) / sum(w for _, w in _STATES))
    status = rng.choice(["delivered", "shipped", "canceled", "unavailable"], n_orders, p=[0.97, 0.015, 0.01, 0.005])

    n_products, n_sellers = 3_000, 1_500
    product_ids, seller_ids = _hex_ids(rng, n_products), _hex_ids(rng, n_sellers)
    product_cat = rng.choice(len(_CATEGORIES), n_products)
    active_sellers = np.maximum(20, (n_sellers * _logistic(order_day.astype(float), days.size * 0.5, 0.01)).astype(int))

    n_items = 1 + rng.binomial(2, 0.06, n_orders)
    item_order = np.repeat(np.arange(n_orders), n_items)
    item_seq = np.arange(item_order.size) - np.repeat(np.cumsum(n_items) - n_items, n_items) + 1
    price = np.round(rng.lognormal(np.log(80), 0.8, item_order.size), 2)
    freight = np.round(np.clip(rng.normal(20, 8, item_order.size), 5, None), 2)
    seller_idx = (rng.random(item_order.size) * active_sellers[item_order]).astype(int)

    order_total = np.bincount(item_order, weights=price + freight, minlength=n_orders)
    voucher_share = np.where(rng.random(n_orders) < 0.055, rng.uniform(0.1, 0.4, n_orders), 0.0)
    pay_type = rng.choice(["credit_card", "boleto", "debit_card"], n_orders, p=[0.77, 0.21, 0.02])

    files = {
        "olist_orders_dataset.csv": pd.DataFrame({
            "order_id": order_ids, "customer_id": customer_ids, "order_status": status,
            "order_purchase_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        }),
        "olist_customers_dataset.csv": pd.DataFrame({
            "customer_id": customer_ids,
            "customer_unique_id": unique_pool[rng.integers(0, unique_pool.size, n_orders)],
            "customer_state": np.asarray([s for s, _ in _STATES])[states],
        }),
        "olist_order_items_dataset.csv": pd.DataFrame({
            "order_id": order_ids[item_order], "order_item_id": item_seq,
            "product_id": product_ids[rng.integers(0, n_products, item_order.size)],
            "seller_id": seller_ids[seller_idx], "price": price, "freight_value": freight,
        }),
        "olist_order_payments_dataset.csv": pd.concat([
            pd.DataFrame({"order_id": order_ids, "payment_sequential": 1, "payment_type": pay_type,
                          "payment_value": np.round(order_total * (1 - voucher_share), 2)}),
            pd.DataFrame({"order_id": order_ids, "payment_sequential": 2, "payment_type": "voucher",
                          "payment_value": np.round(order_total * voucher_share, 2)})[voucher_share > 0],
        ]),
        "olist_products_dataset.csv": pd.DataFrame({
            "product_id": product_ids,
            "product_category_name": np.asarray([c for c, _ in _CATEGORIES])[product_cat],
        }),
        "product_category_name_translation.csv": pd.DataFrame(
            _CATEGORIES, columns=["product_category_name", "product_category_name_english"]
        ),
    }
    files.update(_mql_frames(rng, seller_ids))
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in files.items():
        frame.to_csv(raw_dir / name, index=False)
        written.append(raw_dir / name)
    return written


def _mql_frames(rng: np.random.Generator, seller_ids: np.ndarray) -> dict[str, pd.DataFrame]:
    """Builds the marketing-qualified-lead and closed-deal tables."""
    n_mql = 8_000
    mql_ids = _hex_ids(rng, n_mql)
    contact_days = pd.date_range(date(2017, 6, 1), date(2018, 5, 31), freq="D")
    w = np.linspace(0.3, 1.0, contact_days.size)
    contact = contact_days[rng.choice(contact_days.size, n_mql, p=w / w.sum())]
    origins = rng.choice([o for o, _ in _MQL_ORIGINS], n_mql, p=[p for _, p in _MQL_ORIGINS])
    won = rng.random(n_mql) < 0.105
    won_idx = np.flatnonzero(won)
    won_ts = contact[won_idx] + pd.to_timedelta(rng.gamma(1.2, 40, won_idx.size) * 86_400, unit="s")
    mql = pd.DataFrame({
        "mql_id": mql_ids, "first_contact_date": contact.strftime("%Y-%m-%d"),
        "landing_page_id": _hex_ids(rng, 50)[rng.integers(0, 50, n_mql)], "origin": origins,
    })
    deals = pd.DataFrame({
        "mql_id": mql_ids[won_idx],
        "seller_id": seller_ids[rng.integers(0, seller_ids.size, won_idx.size)],
        "won_date": won_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "business_segment": rng.choice(_SEGMENTS, won_idx.size),
        "lead_type": rng.choice(["online_medium", "online_big", "industry", "offline", "online_small"], won_idx.size),
        "business_type": rng.choice(["reseller", "manufacturer", "other"], won_idx.size, p=[0.69, 0.29, 0.02]),
    })
    return {
        "olist_marketing_qualified_leads_dataset.csv": mql,
        "olist_closed_deals_dataset.csv": deals,
    }


def write_ab_fallback(raw_dir: Path, seed: int = 42, n_users: int = 120_000) -> Path:
    """Writes a seeded ``ab_data.csv``-schema file (null-effect landing page test).

    Args:
        raw_dir: Destination directory.
        seed: RNG seed.
        n_users: Number of randomized users.

    Returns:
        Path of the written CSV.
    """
    rng = np.random.default_rng(seed)
    user_ids = rng.choice(np.arange(630_000, 946_000), n_users, replace=False)
    group = np.where(rng.random(n_users) < 0.5, "control", "treatment")
    start = pd.Timestamp("2017-01-02")
    ts = start + pd.to_timedelta(rng.uniform(0, 22 * 86_400, n_users), unit="s")
    p_conv = np.where(group == "control", 0.1204, 0.1188)
    frame = pd.DataFrame({
        "user_id": user_ids, "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f"), "group": group,
        "landing_page": np.where(group == "control", "old_page", "new_page"),
        "converted": (rng.random(n_users) < p_conv).astype(int),
    })
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "ab_data.csv"
    frame.to_csv(path, index=False)
    return path
