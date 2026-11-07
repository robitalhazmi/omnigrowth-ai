"""Catalog of the real public benchmark datasets used by OmniGrowth-AI.

Each :class:`RemoteSource` lists a primary URL plus community mirrors and the
header columns that must be present for the file to be accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

_OLIST = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets"
_OLIST_MKT_A = "https://raw.githubusercontent.com/dujiaying/olist/master/data"
_OLIST_MKT_B = "https://raw.githubusercontent.com/Ganesh7699/Brazilian-E-Commerce-OList/main"
_AB_A = "https://raw.githubusercontent.com/aaubs/data_science_master/main/data"


@dataclass(frozen=True)
class RemoteSource:
    """A downloadable CSV file with mirror fallbacks.

    Attributes:
        key: Logical dataset key.
        filename: Destination filename inside ``data/raw/``.
        urls: Primary URL followed by mirrors, tried in order.
        required_columns: Columns that must appear in the CSV header.
        dataset: Human-readable dataset family (for the lineage manifest).
        license: License / terms of the original publication.
    """

    key: str
    filename: str
    urls: tuple[str, ...]
    required_columns: tuple[str, ...]
    dataset: str
    license: str


OLIST_DATASET = "Olist Brazilian E-Commerce Public Dataset (Kaggle, 100k orders 2016-2018)"
OLIST_MKT_DATASET = "Olist Marketing Funnel Dataset (Kaggle, 8k MQLs 2017-2018)"
AB_DATASET = "E-Commerce Landing Page A/B Test (Udacity / Kaggle ab_data.csv, 294k users)"
GA4_DATASET = "Google Merchandise Store GA4 export (bigquery-public-data.ga4_obfuscated_sample_ecommerce)"
CC_BY_NC_SA = "CC BY-NC-SA 4.0"

REMOTE_SOURCES: tuple[RemoteSource, ...] = (
    RemoteSource(
        "olist_orders", "olist_orders_dataset.csv",
        (f"{_OLIST}/olist_orders_dataset.csv",),
        ("order_id", "customer_id", "order_status", "order_purchase_timestamp"),
        OLIST_DATASET, CC_BY_NC_SA,
    ),
    RemoteSource(
        "olist_order_items", "olist_order_items_dataset.csv",
        (f"{_OLIST}/olist_order_items_dataset.csv",),
        ("order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"),
        OLIST_DATASET, CC_BY_NC_SA,
    ),
    RemoteSource(
        "olist_customers", "olist_customers_dataset.csv",
        (f"{_OLIST}/olist_customers_dataset.csv",),
        ("customer_id", "customer_unique_id", "customer_state"),
        OLIST_DATASET, CC_BY_NC_SA,
    ),
    RemoteSource(
        "olist_payments", "olist_order_payments_dataset.csv",
        (f"{_OLIST}/olist_order_payments_dataset.csv",),
        ("order_id", "payment_sequential", "payment_type", "payment_value"),
        OLIST_DATASET, CC_BY_NC_SA,
    ),
    RemoteSource(
        "olist_products", "olist_products_dataset.csv",
        (f"{_OLIST}/olist_products_dataset.csv",),
        ("product_id", "product_category_name"),
        OLIST_DATASET, CC_BY_NC_SA,
    ),
    RemoteSource(
        "olist_category_translation", "product_category_name_translation.csv",
        (f"{_OLIST}/product_category_name_translation.csv",),
        ("product_category_name", "product_category_name_english"),
        OLIST_DATASET, CC_BY_NC_SA,
    ),
    RemoteSource(
        "olist_mql", "olist_marketing_qualified_leads_dataset.csv",
        (
            f"{_OLIST_MKT_A}/olist_marketing_qualified_leads_dataset.csv",
            f"{_OLIST_MKT_B}/olist_marketing_qualified_leads_dataset.csv",
        ),
        ("mql_id", "first_contact_date", "landing_page_id", "origin"),
        OLIST_MKT_DATASET, CC_BY_NC_SA,
    ),
    RemoteSource(
        "olist_closed_deals", "olist_closed_deals_dataset.csv",
        (
            f"{_OLIST_MKT_A}/olist_closed_deals_dataset.csv",
            f"{_OLIST_MKT_B}/olist_closed_deals_dataset.csv",
        ),
        ("mql_id", "seller_id", "won_date", "business_segment", "lead_type", "business_type"),
        OLIST_MKT_DATASET, CC_BY_NC_SA,
    ),
    RemoteSource(
        "ab_test", "ab_data.csv",
        (f"{_AB_A}/ab_data.csv",),
        ("user_id", "timestamp", "group", "landing_page", "converted"),
        AB_DATASET, "Public educational dataset (Udacity DAND / Kaggle)",
    ),
)

#: Raw GA4 events filename (exported from BigQuery or produced by the calibrated replica).
GA4_RAW_FILENAME = "ga4_events_raw.parquet"

#: Name of the provenance manifest written next to the raw files.
RAW_MANIFEST_FILENAME = "_manifest.json"


def source_by_key(key: str) -> RemoteSource:
    """Looks up a remote source by its logical key.

    Args:
        key: Logical dataset key, e.g. ``"olist_orders"``.

    Returns:
        The matching :class:`RemoteSource`.

    Raises:
        KeyError: If no source has that key.
    """
    for source in REMOTE_SOURCES:
        if source.key == key:
            return source
    raise KeyError(key)
