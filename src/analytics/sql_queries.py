"""Catalog of parameterized DuckDB OLAP SQL queries for Module 1 (GA4 & Retail Analytics).

Showcases the core analytical SQL patterns used in production product growth analytics:
1. Strict chronological closed-funnel conversion via multi-stage CTEs and ``ROW_NUMBER()``.
2. User purchase/session velocity and inter-arrival times via ``LAG()`` and ``LEAD()``.
3. Customer RFM quartile segmentation via ``NTILE(4)`` and 7-day rolling revenue windows
   via ``AVG(...) OVER (... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SQLQuerySpec:
    """Metadata and SQL text for an inspectable query in the SQL Lab catalog.

    Attributes:
        key: Unique catalog identifier.
        title: Human-readable title shown in the UI selector.
        description: Explanation of the business question and SQL mechanics.
        sql: Executable DuckDB SQL statement.
        constructs: Tuple of SQL features demonstrated by the query.
    """

    key: str
    title: str
    description: str
    sql: str
    constructs: tuple[str, ...]


#: Parameterized strict chronological closed-funnel query across the 5 GA4 funnel stages.
CLOSED_FUNNEL_PARAMETERIZED_SQL: Final[str] = """
WITH filtered_events AS (
    SELECT user_pseudo_id, ga_session_id, event_date, event_timestamp, event_name,
           geo_region, device_category, COALESCE(event_value_in_usd, 0.0) AS event_value_in_usd
    FROM v_ga4_events
    WHERE ($region IS NULL OR $region = 'ALL' OR geo_region = $region)
      AND ($device IS NULL OR $device = 'ALL' OR device_category = $device)
      AND ($start_date IS NULL OR event_date >= CAST($start_date AS DATE))
      AND ($end_date IS NULL OR event_date <= CAST($end_date AS DATE))
      AND event_name IN ('session_start', 'view_item', 'add_to_cart', 'begin_checkout', 'purchase')
),
deep_stage_events AS (
    SELECT user_pseudo_id, ga_session_id, event_name, event_timestamp, event_value_in_usd,
           ROW_NUMBER() OVER (
               PARTITION BY user_pseudo_id, ga_session_id, event_name ORDER BY event_timestamp ASC
           ) AS stage_rn
    FROM filtered_events
    WHERE event_name IN ('add_to_cart', 'begin_checkout', 'purchase')
),
cart_first AS (
    SELECT user_pseudo_id, ga_session_id, event_timestamp AS ts_s3
    FROM deep_stage_events WHERE event_name = 'add_to_cart' AND stage_rn = 1
),
s1 AS (SELECT COUNT(*) AS sessions_count FROM filtered_events WHERE event_name = 'session_start'),
s2 AS (SELECT COUNT(DISTINCT ga_session_id) AS sessions_count FROM filtered_events WHERE event_name = 'view_item'),
s3 AS (
    SELECT DISTINCT c.user_pseudo_id, c.ga_session_id, c.ts_s3
    FROM cart_first c
    INNER JOIN filtered_events v
        ON v.user_pseudo_id = c.user_pseudo_id AND v.ga_session_id = c.ga_session_id
       AND v.event_name = 'view_item' AND v.event_timestamp <= c.ts_s3
),
s4 AS (
    SELECT e.user_pseudo_id, e.ga_session_id, e.event_timestamp AS ts_s4
    FROM deep_stage_events e INNER JOIN s3 USING (user_pseudo_id, ga_session_id)
    WHERE e.event_name = 'begin_checkout' AND e.stage_rn = 1 AND e.event_timestamp >= s3.ts_s3
),
s5 AS (
    SELECT e.user_pseudo_id, e.ga_session_id, e.event_timestamp AS ts_s5,
           e.event_value_in_usd AS purchase_revenue_usd
    FROM deep_stage_events e INNER JOIN s4 USING (user_pseudo_id, ga_session_id)
    WHERE e.event_name = 'purchase' AND e.stage_rn = 1 AND e.event_timestamp >= s4.ts_s4
),
stage_union AS (
    SELECT 1 AS step_order, 'session_start' AS stage_name, sessions_count, 0.0 AS stage_revenue_usd FROM s1
    UNION ALL
    SELECT 2 AS step_order, 'view_item' AS stage_name, sessions_count, 0.0 AS stage_revenue_usd FROM s2
    UNION ALL
    SELECT 3 AS step_order, 'add_to_cart' AS stage_name, COUNT(*) AS sessions_count, 0.0 AS stage_revenue_usd FROM s3
    UNION ALL
    SELECT 4 AS step_order, 'begin_checkout' AS stage_name, COUNT(*) AS sessions_count, 0.0 AS stage_revenue_usd FROM s4
    UNION ALL
    SELECT 5 AS step_order, 'purchase' AS stage_name, COUNT(*) AS sessions_count,
           COALESCE(SUM(purchase_revenue_usd), 0.0) AS stage_revenue_usd FROM s5
),
windowed AS (
    SELECT step_order, stage_name, sessions_count,
           FIRST_VALUE(sessions_count) OVER (ORDER BY step_order ASC) AS top_of_funnel_sessions,
           COALESCE(LAG(sessions_count) OVER (ORDER BY step_order ASC), sessions_count) AS prev_step_sessions,
           stage_revenue_usd
    FROM stage_union
)
SELECT step_order, stage_name, sessions_count,
       prev_step_sessions - sessions_count AS drop_off_sessions,
       ROUND(100.0 * sessions_count / NULLIF(top_of_funnel_sessions, 0), 2) AS overall_conversion_pct,
       ROUND(100.0 * sessions_count / NULLIF(prev_step_sessions, 0), 2) AS step_conversion_pct,
       ROUND(100.0 - (100.0 * sessions_count / NULLIF(prev_step_sessions, 0)), 2) AS step_dropoff_pct,
       ROUND(stage_revenue_usd, 2) AS stage_revenue_usd
FROM windowed
ORDER BY step_order ASC
""".strip()

#: Standalone inspectable version of Query 1 with default ('ALL' / NULL) parameters bound.
CLOSED_FUNNEL_SQL: Final[str] = (
    CLOSED_FUNNEL_PARAMETERIZED_SQL.replace("$region", "'ALL'")
    .replace("$device", "'ALL'")
    .replace("$start_date", "NULL")
    .replace("$end_date", "NULL")
)

#: Query 2: User repeat purchase/checkout velocity and inter-arrival lag/lead analysis.
PURCHASE_VELOCITY_SQL: Final[str] = """
WITH checkout_and_purchase_events AS (
    SELECT
        user_pseudo_id,
        ga_session_id,
        event_date,
        event_timestamp,
        event_name,
        device_category,
        COALESCE(event_value_in_usd, 0.0) AS event_value_in_usd
    FROM v_ga4_events
    WHERE event_name IN ('begin_checkout', 'purchase')
),
sequenced AS (
    SELECT
        user_pseudo_id,
        ga_session_id,
        event_date,
        device_category,
        event_value_in_usd,
        ROW_NUMBER() OVER (
            PARTITION BY user_pseudo_id ORDER BY event_timestamp ASC
        ) AS activity_seq,
        LAG(event_date) OVER (
            PARTITION BY user_pseudo_id ORDER BY event_timestamp ASC
        ) AS prev_event_date,
        LEAD(event_date) OVER (
            PARTITION BY user_pseudo_id ORDER BY event_timestamp ASC
        ) AS next_event_date,
        FIRST_VALUE(DATE_TRUNC('week', event_date)) OVER (
            PARTITION BY user_pseudo_id ORDER BY event_timestamp ASC
        ) AS cohort_week
    FROM checkout_and_purchase_events
),
intervals AS (
    SELECT
        cohort_week,
        device_category,
        user_pseudo_id,
        activity_seq,
        event_value_in_usd,
        date_diff('day', prev_event_date, event_date) AS days_since_prev_event,
        date_diff('day', event_date, next_event_date) AS days_until_next_event
    FROM sequenced
)
SELECT
    CAST(cohort_week AS DATE) AS cohort_week,
    device_category,
    COUNT(DISTINCT user_pseudo_id) AS cohort_users,
    COUNT(*) FILTER (WHERE activity_seq > 1) AS repeat_high_intent_events,
    ROUND(
        AVG(days_since_prev_event) FILTER (WHERE days_since_prev_event IS NOT NULL), 2
    ) AS avg_inter_arrival_days,
    ROUND(
        MEDIAN(days_since_prev_event) FILTER (WHERE days_since_prev_event IS NOT NULL), 2
    ) AS median_inter_arrival_days,
    ROUND(
        AVG(days_until_next_event) FILTER (WHERE days_until_next_event IS NOT NULL), 2
    ) AS avg_days_to_next_activity,
    ROUND(
        100.0 * COUNT(DISTINCT user_pseudo_id) FILTER (WHERE days_until_next_event IS NULL AND activity_seq = 1)
        / NULLIF(COUNT(DISTINCT user_pseudo_id), 0),
        2
    ) AS single_event_drop_pct,
    ROUND(SUM(event_value_in_usd), 2) AS cohort_revenue_usd
FROM intervals
GROUP BY cohort_week, device_category
ORDER BY cohort_week ASC, device_category ASC
""".strip()

#: Query 3: RFM customer quartile segmentation (NTILE(4)) & 7-day rolling regional revenue.
RFM_ROLLING_REVENUE_SQL: Final[str] = """
WITH customer_rfm_base AS (
    SELECT
        user_pseudo_id,
        ANY_VALUE(geo_region) AS geo_region,
        MAX(event_date) AS last_order_date,
        COUNT(DISTINCT ga_session_id) AS frequency_orders,
        SUM(COALESCE(event_value_in_usd, 0.0)) AS total_spend
    FROM v_ga4_events
    WHERE event_name = 'purchase'
    GROUP BY user_pseudo_id
),
rfm_quartiles AS (
    SELECT
        user_pseudo_id,
        geo_region,
        last_order_date,
        frequency_orders,
        total_spend,
        NTILE(4) OVER (ORDER BY total_spend DESC, frequency_orders DESC) AS monetary_quartile,
        NTILE(4) OVER (ORDER BY last_order_date DESC, frequency_orders DESC) AS recency_quartile
    FROM customer_rfm_base
),
regional_vip AS (
    SELECT
        geo_region,
        COUNT(*) AS total_buyers,
        COUNT(*) FILTER (WHERE monetary_quartile = 1) AS top_quartile_users,
        ROUND(AVG(total_spend) FILTER (WHERE monetary_quartile = 1), 2) AS top_quartile_avg_spend
    FROM rfm_quartiles
    GROUP BY geo_region
),
daily_regional AS (
    SELECT
        event_date,
        geo_region,
        COUNT(*) FILTER (WHERE event_name = 'session_start') AS daily_sessions,
        COUNT(*) FILTER (WHERE event_name = 'purchase') AS daily_orders,
        SUM(COALESCE(event_value_in_usd, 0.0)) FILTER (WHERE event_name = 'purchase') AS daily_revenue
    FROM v_ga4_events
    WHERE event_name IN ('session_start', 'purchase')
    GROUP BY event_date, geo_region
)
SELECT
    d.event_date,
    d.geo_region,
    d.daily_sessions,
    d.daily_orders,
    ROUND(COALESCE(d.daily_revenue, 0.0), 2) AS daily_revenue,
    ROUND(
        AVG(COALESCE(d.daily_revenue, 0.0)) OVER (
            PARTITION BY d.geo_region
            ORDER BY d.event_date ASC
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS rolling_7d_avg_revenue,
    ROUND(
        SUM(COALESCE(d.daily_revenue, 0.0)) OVER (
            PARTITION BY d.geo_region
            ORDER BY d.event_date ASC
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS rolling_7d_sum_revenue,
    COALESCE(v.top_quartile_users, 0) AS top_quartile_users,
    COALESCE(v.top_quartile_avg_spend, 0.0) AS top_quartile_avg_spend
FROM daily_regional d
LEFT JOIN regional_vip v USING (geo_region)
ORDER BY d.geo_region ASC, d.event_date ASC
""".strip()

SQL_CATALOG: Final[dict[str, SQLQuerySpec]] = {
    "closed_funnel": SQLQuerySpec(
        key="closed_funnel",
        title="1. Strict Chronological Closed-Funnel Conversion",
        description=(
            "Chains 5 stage CTEs (s1..s5) using ROW_NUMBER() OVER (PARTITION BY user_pseudo_id, "
            "ga_session_id, event_name ORDER BY event_timestamp) to enforce monotonic progression."
        ),
        sql=CLOSED_FUNNEL_SQL,
        constructs=("WITH (Multi-Stage CTEs)", "ROW_NUMBER() OVER (...)", "LAG()", "FIRST_VALUE()"),
    ),
    "purchase_velocity": SQLQuerySpec(
        key="purchase_velocity",
        title="2. User Purchase Velocity & Inter-Arrival Lag",
        description=(
            "Uses LAG(event_date) and LEAD(event_date) partitioned by user_pseudo_id to compute "
            "repeat high-intent activity intervals and drop-off rates across weekly cohorts."
        ),
        sql=PURCHASE_VELOCITY_SQL,
        constructs=("LAG() OVER (...)", "LEAD() OVER (...)", "ROW_NUMBER()", "FIRST_VALUE()"),
    ),
    "rfm_rolling_revenue": SQLQuerySpec(
        key="rfm_rolling_revenue",
        title="3. RFM Quartile Scoring & 7-Day Rolling Regional Revenue",
        description=(
            "Segments customers into spend quartiles via NTILE(4) OVER (ORDER BY total_spend DESC) "
            "and computes 7-day rolling revenue averages per region using window frame clauses."
        ),
        sql=RFM_ROLLING_REVENUE_SQL,
        constructs=("NTILE(4) OVER (...)", "AVG(...) OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)"),
    ),
}
