"""Automated 5-tab OpenPyXL executive financial model & formula workbook generator.

Builds an in-memory :class:`io.BytesIO` ``.xlsx`` workbook populated with live Excel
formulas across five linked worksheets:
1. ``Executive_Summary`` (cross-sheet KPI scorecard)
2. ``GA4_Funnel_Economics`` (step conversion, drop-off & sensitivity formulas)
3. ``AB_Test_Model`` (``NORM.S.DIST`` two-tailed p-value & annualized revenue lift)
4. ``Forecast_And_SCurve`` (live Bass Diffusion curve formula + embedded ``LineChart``)
5. ``Prescriptive_Allocation`` (``SUM`` totals & ``% Variance`` formulas)
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

_NAVY_FILL = PatternFill("solid", fgColor="1B365D")
_SUB_FILL = PatternFill("solid", fgColor="E8EEF5")
_ALT_FILL = PatternFill("solid", fgColor="F7F9FC")
_TOTAL_FILL = PatternFill("solid", fgColor="DCE6F1")
_TITLE_FONT = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
_HDR_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
_BOLD_FONT = Font(name="Calibri", size=11, bold=True, color="1B365D")
_THIN = Side(style="thin", color="CBD5E1")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _style_banner(ws: Worksheet, title: str, max_col: int) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
    cell = ws.cell(row=1, column=1, value=title)
    cell.fill = _NAVY_FILL
    cell.font = _TITLE_FONT
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[1].height = 28
    ws.views.sheetView[0].showGridLines = True


def _style_headers(ws: Worksheet, row: int, headers: list[str]) -> None:
    for col, text in enumerate(headers, start=1):
        c = ws.cell(row=row, column=col, value=text)
        c.fill = _NAVY_FILL
        c.font = _HDR_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = _BORDER
    ws.row_dimensions[row].height = 24


def _auto_fit(ws: Worksheet, min_width: int = 14) -> None:
    for col_cells in ws.columns:
        col_idx = col_cells[0].column
        lengths = [len(str(c.value or "")) for c in col_cells if not str(c.value or "").startswith("=")]
        ws.column_dimensions[get_column_letter(col_idx)].width = max(min_width, min(max(lengths, default=10) + 5, 38))


def _build_funnel_sheet(wb: Workbook, funnel_df: pd.DataFrame) -> None:
    ws = wb.create_sheet("GA4_Funnel_Economics")
    _style_banner(ws, "GA4 Chronological Funnel Economics & Revenue Sensitivity", 8)
    headers = [
        "Step", "Stage Name", "Sessions", "Step CVR",
        "Drop-Off %", "Drop-Off Sessions", "Revenue ($)", "Recoverable ($)",
    ]
    _style_headers(ws, 4, headers)

    for idx, (_, row) in enumerate(funnel_df.head(5).iterrows()):
        r = 5 + idx
        ws.cell(r, 1, int(row["step_order"]))
        ws.cell(r, 2, str(row["stage_name"]))
        ws.cell(r, 3, int(row["sessions_count"])).number_format = "#,##0"
        ws.cell(r, 4, 1.0 if r == 5 else f"=C{r}/C{r-1}").number_format = "0.00%"
        ws.cell(r, 5, f"=1-D{r}").number_format = "0.00%"
        ws.cell(r, 6, 0 if r == 5 else f"=C{r-1}-C{r}").number_format = "#,##0"
        ws.cell(r, 7, float(row.get("stage_revenue_usd", 0.0))).number_format = "$#,##0.00"
        ws.cell(r, 8, 0.0 if r == 5 else f"=F{r}*$C$12*$C$13").number_format = "$#,##0.00"
        for c in range(1, 9):
            ws.cell(r, c).border = _BORDER
            if idx % 2 == 1:
                ws.cell(r, c).fill = _ALT_FILL

    assumptions = [
        (12, "Average Order Value ($)", "=IFERROR(G9/C9,0)", "$#,##0.00"),
        (13, "Checkout Drop-Off Recovery Assumption", 0.05, "0.00%"),
        (14, "Overall Session-to-Purchase CVR", "=C9/C5", "0.00%"),
        (15, "Total Recoverable Revenue Sensitivity ($)", "=SUM(H6:H9)", "$#,##0.00"),
    ]
    ws.cell(11, 2, "Sensitivity Drivers & KPIs").font = _BOLD_FONT
    for r, label, formula, num_fmt in assumptions:
        ws.cell(r, 2, label).font = _BOLD_FONT
        val = ws.cell(r, 3, formula)
        val.number_format = num_fmt
        val.fill = _SUB_FILL
        ws.cell(r, 2).border = _BORDER
        val.border = _BORDER
    _auto_fit(ws)


def _build_ab_sheet(wb: Workbook, ab_metrics: dict[str, Any]) -> None:
    ws = wb.create_sheet("AB_Test_Model")
    _style_banner(ws, "Frequentist Z-Test, CUPED Variance Reduction & Annualized Lift", 4)
    _style_headers(ws, 3, ["Metric / Parameter", "Control (A)", "Treatment (B)", "Model Formula / Result"])

    n_c, n_t = int(ab_metrics.get("n_control", 145000)), int(ab_metrics.get("n_treatment", 145000))
    c_c, c_t = int(ab_metrics.get("conv_control", 17400)), int(ab_metrics.get("conv_treatment", 18125))
    vr_pct = float(ab_metrics.get("cuped_var_reduction_pct", 36.0)) / 100.0
    r_c, r_t = float(ab_metrics.get("ctrl_rev_user", 18.5)), float(ab_metrics.get("treat_rev_user", 19.3))
    rows = [
        (4, "Sample Size (Users)", n_c, n_t, "=B4+C4", "#,##0"),
        (5, "Conversions (Users)", c_c, c_t, "=C5-B5", "#,##0"),
        (6, "Conversion Rate (%)", "=B5/B4", "=C5/C4", "=C6-B6", "0.00%"),
        (7, "Pooled SE", "", "", "=SQRT(((B5+C5)/(B4+C4))*(1-(B5+C5)/(B4+C4))*(1/B4+1/C4))", "0.00000"),
        (8, "Z-Statistic (Treatment - Control)", "", "", "=(C6-B6)/D7", "0.0000"),
        (9, "Two-Tailed p-Value (NORM.S.DIST)", "", "", "=2*(1-NORM.S.DIST(ABS(D8),TRUE))", "0.00000"),
        (10, "Hypothesis Decision (alpha=0.05)", "", "", '=IF(D9<0.05,"SHIP TREATMENT","HOLD / ITERATE")', "@"),
        (11, "CUPED Variance Reduction (%)", "", "", vr_pct, "0.00%"),
        (12, "Post-Exp Mean Revenue / User ($)", r_c, r_t, "=C12-B12", "$#,##0.000"),
        (13, "Daily Exposed Traffic (Users/Day)", "", "", float(ab_metrics.get("daily_traffic", 9500.0)), "#,##0"),
        (14, "Annualized Incremental Revenue ($)", "", "", "=D12*D13*365", "$#,##0.00"),
    ]
    for r, label, ctrl_v, treat_v, res_v, fmt in rows:
        ws.cell(r, 1, label).font = _BOLD_FONT
        for col, v in ((2, ctrl_v), (3, treat_v), (4, res_v)):
            cell = ws.cell(r, col, v if v != "" else None)
            if v != "":
                cell.number_format = fmt
            cell.border = _BORDER
        ws.cell(r, 1).border = _BORDER
        if r % 2 == 0:
            ws.cell(r, 4).fill = _ALT_FILL
    _auto_fit(ws)


def _build_forecast_sheet(wb: Workbook, fc_df: pd.DataFrame, scurve: dict[str, Any]) -> None:
    ws = wb.create_sheet("Forecast_And_SCurve")
    _style_banner(ws, "Regional Time-Series Forecast & Bass Diffusion S-Curve Model", 6)

    params = [
        (3, "Bass Innovation Coeff (p)", float(scurve.get("bass_p", 0.015)), "0.000000"),
        (4, "Bass Imitation Coeff (q)", float(scurve.get("bass_q", 0.025)), "0.000000"),
        (5, "TAM Carrying Capacity (M)", float(scurve.get("bass_m", 25000.0)), "#,##0"),
        (6, "Peak Adoption Timing t* (Days)", "=LN(B4/B3)/(B3+B4)", "0.0"),
    ]
    for r, lbl, val, fmt in params:
        ws.cell(r, 1, lbl).font = _BOLD_FONT
        c = ws.cell(r, 2, val)
        c.number_format = fmt
        c.fill = _SUB_FILL
        ws.cell(r, 1).border = _BORDER
        c.border = _BORDER

    fc_headers = ["Day (t)", "Order Date", "Forecast ($)", "Lower 95% CI ($)", "Upper 95% CI ($)", "Bass S-Curve N(t)"]
    _style_headers(ws, 8, fc_headers)
    future_rows = fc_df[fc_df["is_forecast"]].head(30).reset_index(drop=True)
    for idx, row in future_rows.iterrows():
        r = 9 + idx
        t_step = idx + 1
        ws.cell(r, 1, t_step).number_format = "#,##0"
        ws.cell(r, 2, str(row["order_date"]))
        ws.cell(r, 3, float(row["yhat"])).number_format = "$#,##0.00"
        ws.cell(r, 4, float(row["yhat_lower_95"])).number_format = "$#,##0.00"
        ws.cell(r, 5, float(row["yhat_upper_95"])).number_format = "$#,##0.00"
        bass_f = f"=($B$5*(1-EXP(-($B$3+$B$4)*A{r}))/(1+($B$4/$B$3)*EXP(-($B$3+$B$4)*A{r})))"
        ws.cell(r, 6, bass_f).number_format = "#,##0.0"
        for c in range(1, 7):
            ws.cell(r, c).border = _BORDER

    last_r = 8 + len(future_rows)
    tot_r = last_r + 1
    ws.cell(tot_r, 2, "Total 30-Day Horizon").font = _BOLD_FONT
    for col in (3, 4, 5):
        ltr = get_column_letter(col)
        c = ws.cell(tot_r, col, f"=SUM({ltr}9:{ltr}{last_r})")
        c.number_format = "$#,##0.00"
        c.font = _BOLD_FONT
        c.fill = _TOTAL_FILL
        c.border = _BORDER

    if len(future_rows) >= 2:
        chart = LineChart()
        chart.title = "30-Day Pretrained Forecast Trajectory ($)"
        chart.y_axis.title = "Daily Revenue ($)"
        chart.x_axis.title = "Forecast Date"
        data_ref = Reference(ws, min_col=3, min_row=8, max_col=5, max_row=last_r)
        cats_ref = Reference(ws, min_col=2, min_row=9, max_row=last_r)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.width, chart.height = 18, 10
        ws.add_chart(chart, "H8")
    _auto_fit(ws)


def _build_optimization_sheet(wb: Workbook, alloc_df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Prescriptive_Allocation")
    _style_banner(ws, "SLSQP Prescriptive Marketing Budget & Inventory Reallocation", 10)
    headers = [
        "Region", "Baseline Spend ($)", "Optimal Spend ($)", "Spend Variance %",
        "Baseline Inventory", "Optimal Inventory", "Inventory Variance %",
        "Baseline Margin ($)", "Optimal Margin ($)", "Incremental Margin ($)",
    ]
    _style_headers(ws, 3, headers)

    for idx, (_, row) in enumerate(alloc_df.iterrows()):
        r = 4 + idx
        ws.cell(r, 1, str(row["region"]))
        ws.cell(r, 2, float(row["baseline_spend"])).number_format = "$#,##0.00"
        ws.cell(r, 3, float(row["optimal_spend"])).number_format = "$#,##0.00"
        ws.cell(r, 4, f"=(C{r}-B{r})/B{r}").number_format = "0.00%"
        ws.cell(r, 5, float(row["baseline_inventory"])).number_format = "#,##0"
        ws.cell(r, 6, float(row["optimal_inventory"])).number_format = "#,##0"
        ws.cell(r, 7, f"=(F{r}-E{r})/E{r}").number_format = "0.00%"
        ws.cell(r, 8, float(row["baseline_net_margin"])).number_format = "$#,##0.00"
        ws.cell(r, 9, float(row["optimal_net_margin"])).number_format = "$#,##0.00"
        ws.cell(r, 10, f"=I{r}-H{r}").number_format = "$#,##0.00"
        for c in range(1, 11):
            ws.cell(r, c).border = _BORDER

    tot_r = 4 + len(alloc_df)
    ws.cell(tot_r, 1, "TOTAL").font = _BOLD_FONT
    tot_cols = (
        (2, "$#,##0.00"), (3, "$#,##0.00"), (5, "#,##0"), (6, "#,##0"),
        (8, "$#,##0.00"), (9, "$#,##0.00"), (10, "$#,##0.00"),
    )
    for col, fmt in tot_cols:
        ltr = get_column_letter(col)
        ws.cell(tot_r, col, f"=SUM({ltr}4:{ltr}{tot_r-1})").number_format = fmt
    ws.cell(tot_r, 4, f"=(C{tot_r}-B{tot_r})/B{tot_r}").number_format = "0.00%"
    ws.cell(tot_r, 7, f"=(F{tot_r}-E{tot_r})/E{tot_r}").number_format = "0.00%"
    for c in range(1, 11):
        ws.cell(tot_r, c).font = _BOLD_FONT
        ws.cell(tot_r, c).fill = _TOTAL_FILL
        ws.cell(tot_r, c).border = _BORDER
    _auto_fit(ws)


def _build_summary_sheet(wb: Workbook, alloc_rows: int) -> None:
    ws = wb.active
    ws.title = "Executive_Summary"
    _style_banner(ws, "OmniGrowth-AI | Executive Decision Scorecard (Live Cross-Sheet Links)", 5)
    _style_headers(ws, 3, ["Module", "Executive KPI", "Live Formula Value", "Source Tab", "Notes"])

    opt_tot = 4 + alloc_rows
    ga, ab, fc, pa = "GA4_Funnel_Economics", "AB_Test_Model", "Forecast_And_SCurve", "Prescriptive_Allocation"
    kpis = [
        ("1. GA4 Funnel", "Total Funnel Sessions", f"={ga}!C5", "#,##0", "Chronological session_start"),
        ("1. GA4 Funnel", "Total Purchase Revenue ($)", f"={ga}!G9", "$#,##0.00", "Completed purchases"),
        ("1. GA4 Funnel", "Overall Conversion Rate (%)", f"={ga}!C14", "0.00%", "Purchase / Session"),
        ("1. GA4 Funnel", "Recoverable Revenue ($)", f"={ga}!C15", "$#,##0.00", "Linked to C13 recovery %"),
        ("2. A/B Testing", "Experiment Z-Stat", f"={ab}!D8", "0.0000", "Conversion lift Z-score"),
        ("2. A/B Testing", "Decision Recommendation", f"={ab}!D10", "@", "alpha = 0.05 threshold"),
        ("2. A/B Testing", "Annualized Revenue Lift ($)", f"={ab}!D14", "$#,##0.00", "365-day projection"),
        ("3. S-Curve Forecast", "Bass TAM Ceiling (M)", f"={fc}!B5", "#,##0", "Fitted merchant saturation"),
        ("4. Optimization", "Total Optimal Budget ($)", f"={pa}!C{opt_tot}", "$#,##0.00", "SLSQP budget"),
        ("4. Optimization", "Incremental Margin ($)", f"={pa}!J{opt_tot}", "$#,##0.00", "Vs equal baseline"),
    ]
    for idx, (mod, kpi, formula, fmt, note) in enumerate(kpis):
        r = 4 + idx
        ws.cell(r, 1, mod).font = _BOLD_FONT
        ws.cell(r, 2, kpi)
        val = ws.cell(r, 3, formula)
        val.number_format = fmt
        val.font = _BOLD_FONT
        ws.cell(r, 4, formula.split("!")[0].lstrip("="))
        ws.cell(r, 5, note)
        for c in range(1, 6):
            ws.cell(r, c).border = _BORDER
            if idx % 2 == 1:
                ws.cell(r, c).fill = _ALT_FILL
    _auto_fit(ws)


def generate_executive_workbook(
    funnel_df: pd.DataFrame,
    ab_metrics: dict[str, Any],
    forecast_df: pd.DataFrame,
    scurve_params: dict[str, Any],
    allocation_df: pd.DataFrame,
) -> BytesIO:
    """Builds the formatted 5-tab ``.xlsx`` financial workbook with live Excel formulas."""
    wb = Workbook()
    _build_summary_sheet(wb, len(allocation_df))
    _build_funnel_sheet(wb, funnel_df)
    _build_ab_sheet(wb, ab_metrics)
    _build_forecast_sheet(wb, forecast_df, scurve_params)
    _build_optimization_sheet(wb, allocation_df)

    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)
    return stream
