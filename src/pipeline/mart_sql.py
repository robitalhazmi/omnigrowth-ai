"""DuckDB SQL used by the offline ETL to build the serving marts.

All statements read raw CSV / Parquet directly (no Pandas round-trip) and are
rendered with safely quoted file paths via :func:`quote_path`.
"""

from __future__ import annotations

from pathlib import Path

from src.database.duckdb_manager import quote_path

#: Brazilian states -> (state name, macro region) used to label Olist regions.
BR_STATES: dict[str, tuple[str, str]] = {
    "AC": ("Acre", "North"), "AL": ("Alagoas", "Northeast"), "AM": ("Amazonas", "North"),
    "AP": ("Amapa", "North"), "BA": ("Bahia", "Northeast"), "CE": ("Ceara", "Northeast"),
    "DF": ("Distrito Federal", "Center-West"), "ES": ("Espirito Santo", "Southeast"),
    "GO": ("Goias", "Center-West"), "MA": ("Maranhao", "Northeast"),
    "MG": ("Minas Gerais", "Southeast"), "MS": ("Mato Grosso do Sul", "Center-West"),
    "MT": ("Mato Grosso", "Center-West"), "PA": ("Para", "North"), "PB": ("Paraiba", "Northeast"),
    "PE": ("Pernambuco", "Northeast"), "PI": ("Piaui", "Northeast"), "PR": ("Parana", "South"),
    "RJ": ("Rio de Janeiro", "Southeast"), "RN": ("Rio Grande do Norte", "Northeast"),
    "RO": ("Rondonia", "North"), "RR": ("Roraima", "North"), "RS": ("Rio Grande do Sul", "South"),
    "SC": ("Santa Catarina", "South"), "SE": ("Sergipe", "Northeast"), "SP": ("Sao Paulo", "Southeast"),
    "TO": ("Tocantins", "North"),
}

#: English Olist category -> executive product segment (first matching pattern wins).
SEGMENT_CASE = (
    "CASE\n"
    "  WHEN category SIMILAR TO '(health_beauty|perfumery|diapers_and_hygiene|baby)'\n"
    "       THEN 'Health & Beauty'\n"
    "  WHEN category SIMILAR TO '(computers.*|telephony|fixed_telephony|electronics|"
    "consoles_games|audio|tablets_printing_image|pc_gamer)'\n"
    "       THEN 'Electronics & Tech'\n"
    "  WHEN category SIMILAR TO '(bed_bath_table|furniture.*|housewares|garden_tools|home_.*|"
    "kitchen.*|small_appliances.*|construction_tools.*|air_conditioning|la_cuisine|flowers|office_furniture)'\n"
    "       THEN 'Home & Living'\n"
    "  WHEN category SIMILAR TO '(watches_gifts|fashion.*|luggage_accessories|cool_stuff|"
    "christmas_supplies|party_supplies)'\n"
    "       THEN 'Fashion & Gifts'\n"
    "  WHEN category SIMILAR TO '(sports_leisure|toys|auto|pet_shop|musical_instruments|books.*|"
    "stationery|art|arts_and_craftmanship|cine_photo|dvds_blu_ray|music|cds_dvds_musicals)'\n"
    "       THEN 'Leisure & Lifestyle'\n"
    "  ELSE 'Food, Industry & Other'\n"
    "END"
)


def _states_values() -> str:
    """Renders the state lookup as a SQL ``VALUES`` list."""
    rows = [
        f"('{code}', '{name}', '{macro}')" for code, (name, macro) in sorted(BR_STATES.items())
    ]
    return ",\n    ".join(rows)


def ga4_events_sql(raw_parquet: Path, out_parquet: Path) -> str:
    """Flattens / types raw GA4 events and writes the ZSTD Parquet mart."""
    return f"""
COPY (
  SELECT
    CAST(event_date AS DATE)                    AS event_date,
    CAST(event_timestamp AS TIMESTAMP)          AS event_timestamp,
    CAST(event_name AS VARCHAR)                 AS event_name,
    CAST(user_pseudo_id AS VARCHAR)             AS user_pseudo_id,
    CAST(ga_session_id AS BIGINT)               AS ga_session_id,
    CAST(device_category AS VARCHAR)            AS device_category,
    COALESCE(traffic_source, '(not set)')       AS traffic_source,
    COALESCE(traffic_medium, '(not set)')       AS traffic_medium,
    COALESCE(NULLIF(geo_region, ''), '(not set)') AS geo_region,
    CAST(item_category AS VARCHAR)              AS item_category,
    ROUND(CAST(event_value_in_usd AS DOUBLE), 2) AS event_value_in_usd
  FROM read_parquet({quote_path(raw_parquet)})
  WHERE ga_session_id IS NOT NULL
  ORDER BY event_date, user_pseudo_id, event_timestamp
) TO {quote_path(out_parquet)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
"""


def regional_demand_sql(raw: dict[str, Path], out_parquet: Path, start: str, end: str) -> str:
    """Builds the dense daily region x segment demand panel from Olist orders.

    ``promo_discount_pct`` is the real share of order value settled with Olist
    vouchers; ``cumulative_active_merchants`` counts distinct sellers that have
    served the region up to each day (a real network-adoption S-curve).
    """
    return f"""
COPY (
WITH states(region_code, region_name, macro_region) AS (
  VALUES
    {_states_values()}
),
orders AS (
  SELECT o.order_id, CAST(o.order_purchase_timestamp AS TIMESTAMP)::DATE AS order_date,
         c.customer_state AS region_code, c.customer_unique_id
  FROM read_csv({quote_path(raw["olist_orders"])}, header = true, auto_detect = true) o
  JOIN read_csv({quote_path(raw["olist_customers"])}, header = true, auto_detect = true) c
    USING (customer_id)
  WHERE o.order_status NOT IN ('canceled', 'unavailable')
),
payments AS (
  SELECT order_id,
         SUM(payment_value) AS paid_total,
         COALESCE(SUM(payment_value) FILTER (WHERE payment_type = 'voucher'), 0) AS voucher_total
  FROM read_csv({quote_path(raw["olist_payments"])}, header = true, auto_detect = true)
  GROUP BY order_id
),
items AS (
  SELECT i.order_id, i.seller_id, i.price, i.freight_value,
         COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category
  FROM read_csv({quote_path(raw["olist_order_items"])}, header = true, auto_detect = true) i
  LEFT JOIN read_csv({quote_path(raw["olist_products"])}, header = true, auto_detect = true) p
    USING (product_id)
  LEFT JOIN read_csv({quote_path(raw["olist_category_translation"])}, header = true, auto_detect = true) t
    USING (product_category_name)
),
item_facts AS (
  SELECT o.order_date, o.region_code, o.order_id, o.customer_unique_id, i.seller_id,
         {SEGMENT_CASE} AS product_segment, i.price, i.freight_value,
         COALESCE(pay.voucher_total / NULLIF(pay.paid_total, 0), 0) AS voucher_share
  FROM items i JOIN orders o USING (order_id) LEFT JOIN payments pay USING (order_id)
),
merchant_first AS (
  SELECT region_code, seller_id, MIN(order_date) AS first_date FROM item_facts GROUP BY ALL
),
calendar AS (
  SELECT CAST(d AS DATE) AS order_date
  FROM generate_series(DATE '{start}', DATE '{end}', INTERVAL 1 DAY) g(d)
),
merchants_daily AS (
  SELECT s.region_code, c.order_date,
         SUM(COUNT(m.seller_id)) OVER (PARTITION BY s.region_code ORDER BY c.order_date
                                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
           + COALESCE(ANY_VALUE(pre.n), 0) AS cumulative_active_merchants
  FROM states s CROSS JOIN calendar c
  LEFT JOIN merchant_first m ON m.region_code = s.region_code AND m.first_date = c.order_date
  LEFT JOIN (SELECT region_code, COUNT(*) AS n FROM merchant_first
             WHERE first_date < DATE '{start}' GROUP BY ALL) pre ON pre.region_code = s.region_code
  GROUP BY s.region_code, c.order_date
),
agg AS (
  SELECT order_date, region_code, product_segment,
         COUNT(DISTINCT order_id) AS daily_orders, COUNT(*) AS units_sold,
         COUNT(DISTINCT customer_unique_id) AS unique_customers,
         SUM(price) AS gross_revenue, SUM(freight_value) AS freight_cost,
         SUM(price * voucher_share) AS voucher_value
  FROM item_facts WHERE order_date BETWEEN DATE '{start}' AND DATE '{end}' GROUP BY ALL
),
segments AS (SELECT DISTINCT product_segment FROM item_facts)
SELECT c.order_date, s.region_code, s.region_name, s.macro_region, g.product_segment,
       COALESCE(a.daily_orders, 0)::INTEGER                   AS daily_orders,
       COALESCE(a.units_sold, 0)::INTEGER                     AS units_sold,
       COALESCE(a.unique_customers, 0)::INTEGER               AS unique_customers,
       ROUND(COALESCE(a.gross_revenue, 0), 2)                 AS gross_revenue,
       ROUND(COALESCE(a.freight_cost, 0), 2)                  AS freight_cost,
       ROUND(COALESCE(100 * a.voucher_value / NULLIF(a.gross_revenue, 0), 0), 3) AS promo_discount_pct,
       COALESCE(md.cumulative_active_merchants, 0)::INTEGER   AS cumulative_active_merchants
FROM calendar c CROSS JOIN states s CROSS JOIN segments g
LEFT JOIN agg a ON a.order_date = c.order_date AND a.region_code = s.region_code
               AND a.product_segment = g.product_segment
LEFT JOIN merchants_daily md ON md.region_code = s.region_code AND md.order_date = c.order_date
ORDER BY s.region_code, g.product_segment, c.order_date
) TO {quote_path(out_parquet)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
"""


def mql_funnel_sql(raw: dict[str, Path], out_parquet: Path) -> str:
    """Joins Olist marketing-qualified leads with closed merchant deals."""
    return f"""
COPY (
  SELECT m.mql_id,
         CAST(m.first_contact_date AS DATE)          AS first_contact_date,
         COALESCE(m.origin, 'unknown')               AS lead_origin,
         CAST(d.won_date AS TIMESTAMP)               AS won_timestamp,
         d.business_segment, d.lead_type, d.business_type,
         (d.mql_id IS NOT NULL)                      AS is_won,
         date_diff('day', CAST(m.first_contact_date AS DATE), CAST(d.won_date AS DATE)) AS days_to_close
  FROM read_csv({quote_path(raw["olist_mql"])}, header = true, auto_detect = true) m
  LEFT JOIN read_csv({quote_path(raw["olist_closed_deals"])}, header = true, auto_detect = true) d
    USING (mql_id)
  ORDER BY first_contact_date, mql_id
) TO {quote_path(out_parquet)} (FORMAT PARQUET, COMPRESSION ZSTD)
"""


def ab_clean_sql(raw_csv: Path) -> str:
    """Canonical cleaning of ``ab_data.csv``: drop misaligned rows, keep first exposure."""
    return f"""
WITH raw AS (
  SELECT user_id, CAST("timestamp" AS TIMESTAMP) AS ts, "group", landing_page, converted
  FROM read_csv({quote_path(raw_csv)}, header = true, auto_detect = true)
),
aligned AS (
  SELECT * FROM raw WHERE ("group" = 'treatment') = (landing_page = 'new_page')
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts) AS exposure_rank FROM aligned
)
SELECT user_id, ts AS "timestamp", "group", converted
FROM ranked WHERE exposure_rank = 1
ORDER BY user_id
"""
