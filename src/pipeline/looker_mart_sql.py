"""SQL for the flat Google Looker Studio executive mart.

Looker Studio's free *File Upload* connector accepts one flat table, so three
grains are unioned into a single long CSV distinguished by ``metric_domain``:

* ``ga4_funnel``       — daily open-funnel session counts by region/device/channel.
* ``regional_revenue`` — daily Olist orders, revenue, freight and voucher share by state.
* ``cohort_retention`` — weekly GA4 acquisition cohorts x weeks since first visit.

Unused columns for a given domain are ``NULL`` so Looker Studio filters stay simple.
"""

from __future__ import annotations

from pathlib import Path

from src.database.duckdb_manager import quote_path

LOOKER_COLUMNS: tuple[str, ...] = (
    "metric_domain", "report_date", "region", "device_category", "traffic_channel",
    "sessions", "product_views", "cart_additions", "checkouts", "purchases", "revenue_usd",
    "view_rate", "cart_rate", "checkout_rate", "purchase_conversion_rate",
    "orders", "gross_revenue_brl", "freight_cost_brl", "promo_discount_pct",
    "cohort_week", "weeks_since_first", "cohort_users", "retained_users", "retention_rate",
)


def looker_mart_sql(ga4_parquet: Path, demand_parquet: Path, out_csv: Path) -> str:
    """Renders the ``COPY`` statement producing ``looker_executive_mart.csv``.

    Args:
        ga4_parquet: Processed GA4 events mart.
        demand_parquet: Processed regional demand mart.
        out_csv: Destination CSV path.

    Returns:
        SQL text.
    """
    ga4, demand = quote_path(ga4_parquet), quote_path(demand_parquet)
    return f"""
COPY (
WITH sessions AS (
  SELECT user_pseudo_id, ga_session_id,
         MIN(event_date) AS session_date,
         ANY_VALUE(geo_region) AS region, ANY_VALUE(device_category) AS device_category,
         ANY_VALUE(traffic_medium) AS traffic_channel,
         MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END)      AS has_view,
         MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END)    AS has_cart,
         MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS has_checkout,
         MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END)       AS has_purchase,
         SUM(CASE WHEN event_name = 'purchase' THEN event_value_in_usd ELSE 0 END) AS revenue
  FROM read_parquet({ga4})
  GROUP BY user_pseudo_id, ga_session_id
),
funnel AS (
  SELECT 'ga4_funnel' AS metric_domain, session_date AS report_date, region, device_category,
         traffic_channel,
         COUNT(*) AS sessions, SUM(has_view) AS product_views, SUM(has_cart) AS cart_additions,
         SUM(has_checkout) AS checkouts, SUM(has_purchase) AS purchases,
         ROUND(SUM(revenue), 2) AS revenue_usd
  FROM sessions GROUP BY ALL
),
user_weeks AS (
  SELECT DISTINCT user_pseudo_id, DATE_TRUNC('week', session_date) AS active_week FROM sessions
),
user_cohort AS (
  SELECT user_pseudo_id, MIN(active_week) AS cohort_week FROM user_weeks GROUP BY user_pseudo_id
),
cohorts AS (
  SELECT c.cohort_week, date_diff('week', c.cohort_week, w.active_week) AS weeks_since_first,
         COUNT(DISTINCT w.user_pseudo_id) AS retained_users
  FROM user_weeks w JOIN user_cohort c USING (user_pseudo_id) GROUP BY ALL
),
cohort_rates AS (
  SELECT c.*, FIRST_VALUE(retained_users) OVER (
           PARTITION BY cohort_week ORDER BY weeks_since_first) AS cohort_users
  FROM cohorts c
),
revenue AS (
  SELECT order_date AS report_date, region_code AS region,
         SUM(daily_orders) AS orders, ROUND(SUM(gross_revenue), 2) AS gross_revenue_brl,
         ROUND(SUM(freight_cost), 2) AS freight_cost_brl,
         ROUND(100 * SUM(gross_revenue * promo_discount_pct / 100)
               / NULLIF(SUM(gross_revenue), 0), 3) AS promo_discount_pct
  FROM read_parquet({demand}) GROUP BY ALL
)
SELECT metric_domain, report_date, region, device_category, traffic_channel,
       sessions, product_views, cart_additions, checkouts, purchases, revenue_usd,
       ROUND(product_views / sessions, 4)                    AS view_rate,
       ROUND(cart_additions / NULLIF(product_views, 0), 4)   AS cart_rate,
       ROUND(checkouts / NULLIF(cart_additions, 0), 4)       AS checkout_rate,
       ROUND(purchases / sessions, 4)                        AS purchase_conversion_rate,
       NULL::BIGINT AS orders, NULL::DOUBLE AS gross_revenue_brl, NULL::DOUBLE AS freight_cost_brl,
       NULL::DOUBLE AS promo_discount_pct, NULL::DATE AS cohort_week,
       NULL::BIGINT AS weeks_since_first, NULL::BIGINT AS cohort_users,
       NULL::BIGINT AS retained_users, NULL::DOUBLE AS retention_rate
FROM funnel
UNION ALL BY NAME
SELECT 'regional_revenue' AS metric_domain, report_date, region, orders, gross_revenue_brl,
       freight_cost_brl, promo_discount_pct
FROM revenue
UNION ALL BY NAME
SELECT 'cohort_retention' AS metric_domain,
       CAST(cohort_week + INTERVAL (weeks_since_first * 7) DAY AS DATE) AS report_date,
       CAST(cohort_week AS DATE) AS cohort_week, weeks_since_first, cohort_users, retained_users,
       ROUND(retained_users / cohort_users, 4) AS retention_rate
FROM cohort_rates
ORDER BY metric_domain, report_date, region NULLS FIRST, device_category NULLS FIRST,
         traffic_channel NULLS FIRST, cohort_week NULLS FIRST, weeks_since_first NULLS FIRST
) TO {quote_path(out_csv)} (HEADER, DELIMITER ',')
"""
