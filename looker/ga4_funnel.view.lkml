# =============================================================================
# OmniGrowth-AI — GA4 funnel semantic layer (LookML)
#
# Backing table: data/processed/ga4_events.parquet loaded into the warehouse as
# `omnigrowth.ga4_events` (BigQuery dialect shown; the same SQL runs on DuckDB /
# MotherDuck with STRING -> VARCHAR). Grain: one row per GA4 funnel event.
# =============================================================================

view: ga4_funnel {
  sql_table_name: `omnigrowth.ga4_events` ;;
  view_label: "GA4 Funnel Events"

  # ---------------------------------------------------------------- keys
  dimension: event_pk {
    primary_key: yes
    hidden: yes
    type: string
    sql: CONCAT(${TABLE}.user_pseudo_id, '|', CAST(${TABLE}.ga_session_id AS STRING), '|',
                ${TABLE}.event_name, '|', CAST(${TABLE}.event_timestamp AS STRING)) ;;
  }

  dimension: session_key {
    hidden: yes
    type: string
    sql: CONCAT(${TABLE}.user_pseudo_id, '|', CAST(${TABLE}.ga_session_id AS STRING)) ;;
  }

  # ---------------------------------------------------------- dimensions
  dimension_group: event {
    type: time
    timeframes: [raw, date, week, month, day_of_week, hour_of_day]
    datatype: date
    convert_tz: no
    sql: ${TABLE}.event_date ;;
    description: "Calendar date of the GA4 event (UTC)."
  }

  dimension_group: event_timestamp {
    type: time
    timeframes: [raw, time, minute15, hour]
    sql: ${TABLE}.event_timestamp ;;
  }

  dimension: event_name {
    type: string
    sql: ${TABLE}.event_name ;;
    description: "session_start, view_item, add_to_cart, begin_checkout, add_payment_info, purchase"
  }

  dimension: funnel_step_order {
    type: number
    sql: CASE ${event_name}
           WHEN 'session_start' THEN 1 WHEN 'view_item' THEN 2 WHEN 'add_to_cart' THEN 3
           WHEN 'begin_checkout' THEN 4 WHEN 'add_payment_info' THEN 5 WHEN 'purchase' THEN 6
         END ;;
  }

  dimension: user_pseudo_id {
    type: string
    sql: ${TABLE}.user_pseudo_id ;;
  }

  dimension: ga_session_id {
    type: number
    value_format_name: id
    sql: ${TABLE}.ga_session_id ;;
  }

  dimension: geo_region {
    type: string
    map_layer_name: us_states
    sql: ${TABLE}.geo_region ;;
  }

  dimension: device_category {
    type: string
    sql: ${TABLE}.device_category ;;
  }

  dimension: traffic_source {
    type: string
    sql: ${TABLE}.traffic_source ;;
  }

  dimension: traffic_medium {
    type: string
    sql: ${TABLE}.traffic_medium ;;
  }

  dimension: traffic_channel_group {
    type: string
    sql: CASE
           WHEN ${traffic_medium} = 'organic' THEN 'Organic Search'
           WHEN ${traffic_medium} = 'cpc' THEN 'Paid Search'
           WHEN ${traffic_medium} = 'referral' THEN 'Referral'
           WHEN ${traffic_medium} = 'email' THEN 'Email'
           WHEN ${traffic_medium} = '(none)' THEN 'Direct'
           ELSE 'Unattributed'
         END ;;
  }

  dimension: item_category {
    type: string
    sql: ${TABLE}.item_category ;;
  }

  dimension: event_value_in_usd {
    hidden: yes
    type: number
    sql: ${TABLE}.event_value_in_usd ;;
  }

  # ------------------------------------------------------------ measures
  measure: total_events {
    type: count
    drill_fields: [funnel_detail*]
  }

  measure: users {
    type: count_distinct
    sql: ${user_pseudo_id} ;;
  }

  measure: session_starts {
    label: "Session Starts"
    type: count_distinct
    sql: ${session_key} ;;
    filters: [event_name: "session_start"]
  }

  measure: product_view_sessions {
    type: count_distinct
    sql: ${session_key} ;;
    filters: [event_name: "view_item"]
  }

  measure: cart_additions {
    label: "Cart Addition Sessions"
    type: count_distinct
    sql: ${session_key} ;;
    filters: [event_name: "add_to_cart"]
  }

  measure: checkout_sessions {
    type: count_distinct
    sql: ${session_key} ;;
    filters: [event_name: "begin_checkout"]
  }

  measure: purchase_sessions {
    type: count_distinct
    sql: ${session_key} ;;
    filters: [event_name: "purchase"]
  }

  measure: purchase_events {
    hidden: yes
    type: count
    filters: [event_name: "purchase"]
  }

  measure: total_revenue_usd {
    type: sum
    value_format_name: usd
    sql: ${event_value_in_usd} ;;
    filters: [event_name: "purchase"]
  }

  measure: cart_rate {
    description: "Share of product-viewing sessions that added to cart."
    type: number
    value_format_name: percent_2
    sql: 1.0 * ${cart_additions} / NULLIF(${product_view_sessions}, 0) ;;
  }

  measure: checkout_rate {
    description: "Share of cart sessions that began checkout."
    type: number
    value_format_name: percent_2
    sql: 1.0 * ${checkout_sessions} / NULLIF(${cart_additions}, 0) ;;
  }

  measure: purchase_conversion_rate {
    description: "Session-level purchase conversion (purchase sessions / session starts)."
    type: number
    value_format_name: percent_2
    sql: 1.0 * ${purchase_sessions} / NULLIF(${session_starts}, 0) ;;
  }

  measure: average_order_value {
    type: number
    value_format_name: usd
    sql: ${total_revenue_usd} / NULLIF(${purchase_events}, 0) ;;
  }

  measure: revenue_per_session {
    type: number
    value_format_name: usd
    sql: ${total_revenue_usd} / NULLIF(${session_starts}, 0) ;;
  }

  set: funnel_detail {
    fields: [event_date, geo_region, device_category, traffic_source, event_name, total_events]
  }
}

# -----------------------------------------------------------------------------
# Session-grain derived table enforcing the *strict chronological* closed funnel:
# a step only counts if it happens after the previous step within the session.
# -----------------------------------------------------------------------------
view: ga4_session_funnel {
  view_label: "GA4 Strict Session Funnel"
  derived_table: {
    datagroup_trigger: omni_growth_daily
    sql:
      WITH firsts AS (
        SELECT user_pseudo_id, ga_session_id,
               MIN(event_date) AS session_date,
               ANY_VALUE(device_category) AS device_category,
               ANY_VALUE(geo_region) AS geo_region,
               ANY_VALUE(traffic_source) AS traffic_source,
               MIN(IF(event_name = 'session_start', event_timestamp, NULL)) AS t_start,
               MIN(IF(event_name = 'view_item', event_timestamp, NULL)) AS t_view,
               MIN(IF(event_name = 'add_to_cart', event_timestamp, NULL)) AS t_cart,
               MIN(IF(event_name = 'begin_checkout', event_timestamp, NULL)) AS t_checkout,
               MIN(IF(event_name = 'purchase', event_timestamp, NULL)) AS t_purchase
        FROM `omnigrowth.ga4_events`
        GROUP BY 1, 2
      )
      SELECT *,
             t_view IS NOT NULL AND t_view >= t_start AS reached_view,
             t_cart IS NOT NULL AND t_view IS NOT NULL AND t_cart >= t_view AS reached_cart,
             t_checkout IS NOT NULL AND t_cart IS NOT NULL AND t_view IS NOT NULL
               AND t_cart >= t_view AND t_checkout >= t_cart AS reached_checkout,
             t_purchase IS NOT NULL AND t_checkout IS NOT NULL AND t_cart IS NOT NULL
               AND t_view IS NOT NULL AND t_cart >= t_view AND t_checkout >= t_cart
               AND t_purchase >= t_checkout AS reached_purchase
      FROM firsts ;;
  }

  dimension: session_key {
    primary_key: yes
    hidden: yes
    sql: CONCAT(${TABLE}.user_pseudo_id, '|', CAST(${TABLE}.ga_session_id AS STRING)) ;;
  }

  dimension_group: session {
    type: time
    timeframes: [date, week, month]
    datatype: date
    convert_tz: no
    sql: ${TABLE}.session_date ;;
  }

  dimension: device_category { sql: ${TABLE}.device_category ;; }
  dimension: geo_region { sql: ${TABLE}.geo_region ;; }
  dimension: traffic_source { sql: ${TABLE}.traffic_source ;; }

  measure: sessions { type: count }
  measure: strict_view_sessions { type: count filters: [reached_view: "yes"] }
  measure: strict_cart_sessions { type: count filters: [reached_cart: "yes"] }
  measure: strict_checkout_sessions { type: count filters: [reached_checkout: "yes"] }
  measure: strict_purchase_sessions { type: count filters: [reached_purchase: "yes"] }

  dimension: reached_view { type: yesno sql: ${TABLE}.reached_view ;; }
  dimension: reached_cart { type: yesno sql: ${TABLE}.reached_cart ;; }
  dimension: reached_checkout { type: yesno sql: ${TABLE}.reached_checkout ;; }
  dimension: reached_purchase { type: yesno sql: ${TABLE}.reached_purchase ;; }

  measure: strict_purchase_conversion_rate {
    type: number
    value_format_name: percent_2
    sql: 1.0 * ${strict_purchase_sessions} / NULLIF(${sessions}, 0) ;;
  }
}
