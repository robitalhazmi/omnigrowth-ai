"""Streamlit UI tab components for OmniGrowth-AI."""

from app.components.tab_ab_testing import render_ab_testing_tab
from app.components.tab_forecasting import render_forecasting_tab
from app.components.tab_genai_excel import render_genai_excel_tab
from app.components.tab_optimization import render_optimization_tab
from app.components.tab_sql_ga4 import render_sql_ga4_tab

__all__ = [
    "render_ab_testing_tab",
    "render_forecasting_tab",
    "render_genai_excel_tab",
    "render_optimization_tab",
    "render_sql_ga4_tab",
]
