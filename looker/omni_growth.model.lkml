# =============================================================================
# OmniGrowth-AI — LookML model
#
# Exposes the GA4 product-funnel semantic layer to Looker Explores. The warehouse
# connection is expected to contain the tables produced by
# `scripts/build_data_mart.py` (data/processed/*.parquet) under the `omnigrowth`
# dataset/schema.
# =============================================================================

connection: "omnigrowth_warehouse"

include: "/looker/*.view.lkml"

label: "OmniGrowth-AI — Product Growth"

datagroup: omni_growth_daily {
  sql_trigger: SELECT MAX(event_date) FROM `omnigrowth.ga4_events` ;;
  max_cache_age: "24 hours"
  description: "Rebuilds caches / PDTs when a new day of GA4 events lands."
}

persist_with: omni_growth_daily

week_start_day: monday

# -----------------------------------------------------------------------------
# Event-grain explore: funnel KPIs sliced by date, region, device, traffic source.
# -----------------------------------------------------------------------------
explore: ga4_funnel {
  label: "GA4 Funnel Events"
  description: "Open-funnel GA4 KPIs: session starts, cart additions, checkout rate, purchase conversion and AOV."

  always_filter: {
    filters: [ga4_funnel.event_date: "90 days"]
  }

  join: ga4_session_funnel {
    type: left_outer
    relationship: many_to_one
    sql_on: ${ga4_funnel.session_key} = ${ga4_session_funnel.session_key} ;;
  }
}

# -----------------------------------------------------------------------------
# Session-grain explore: strict chronological closed funnel.
# -----------------------------------------------------------------------------
explore: ga4_session_funnel {
  label: "GA4 Strict Session Funnel"
  description: "Session-level closed funnel (view -> cart -> checkout -> purchase in order)."
}
