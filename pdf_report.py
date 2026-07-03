"""
pdf_report.py
-------------
Drop-in replacement for build_pdf_report() in app.py.

USAGE in app.py:
    1. Remove (or keep) the old build_pdf_report function.
    2. Add this import near the top:
           from pdf_report import build_pdf_report
    3. The function signature is identical — no other changes needed.

REQUIRES:
    pip install reportlab
"""

from __future__ import annotations

from io import BytesIO
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF


# ── Brand colours ─────────────────────────────────────────────────────────────
C_DARK       = colors.HexColor("#0d1117")
C_NAVY       = colors.HexColor("#0f172a")
C_ACCENT     = colors.HexColor("#4FC3F7")   # sky blue
C_ACCENT2    = colors.HexColor("#1d4ed8")   # deep blue
C_SUCCESS    = colors.HexColor("#22c55e")
C_WARNING    = colors.HexColor("#f59e0b")
C_DANGER     = colors.HexColor("#ef4444")
C_TEXT       = colors.HexColor("#e5e7eb")   # light text — use only on dark card/table backgrounds
C_TEXT_DARK  = colors.HexColor("#1f2937")   # dark text — use on the white page background
C_MUTED      = colors.HexColor("#6b7280")
C_BORDER     = colors.HexColor("#2d3748")
C_CARD       = colors.HexColor("#1c1f26")
C_WHITE      = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ── Styles ────────────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", fontSize=22, textColor=C_WHITE,
                             fontName="Helvetica-Bold", spaceAfter=4),
        "h2": ParagraphStyle("h2", fontSize=13, textColor=C_ACCENT,
                             fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontSize=10, textColor=C_TEXT_DARK,
                             fontName="Helvetica-Bold", spaceAfter=3),
        "body": ParagraphStyle("body", fontSize=9, textColor=C_TEXT_DARK,
                               fontName="Helvetica", leading=14),
        "muted": ParagraphStyle("muted", fontSize=8, textColor=C_MUTED,
                                fontName="Helvetica", leading=12),
        "center": ParagraphStyle("center", fontSize=9, textColor=C_TEXT_DARK,
                                 fontName="Helvetica", alignment=TA_CENTER),
        "tag_safe":    ParagraphStyle("tag_safe", fontSize=8, textColor=C_SUCCESS,
                                      fontName="Helvetica-Bold", alignment=TA_CENTER),
        "tag_warn":    ParagraphStyle("tag_warn", fontSize=8, textColor=C_WARNING,
                                      fontName="Helvetica-Bold", alignment=TA_CENTER),
        "tag_danger":  ParagraphStyle("tag_danger", fontSize=8, textColor=C_DANGER,
                                      fontName="Helvetica-Bold", alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", fontSize=10, textColor=C_MUTED,
                                   fontName="Helvetica", spaceAfter=2),
    }


# ── Header band (drawn on every page via onFirstPage / onLaterPages) ──────────
def _header_footer(canvas, doc):
    canvas.saveState()
    # Top bar
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, fill=1, stroke=0)
    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, PAGE_H - 14 * mm, 2 * mm, 14 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.setFillColor(C_WHITE)
    canvas.drawString(MARGIN, PAGE_H - 9 * mm, "FORESIGHT  |  Inventory Intelligence Report")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_MUTED)
    ts = datetime.now().strftime("%d %b %Y  %H:%M")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 9 * mm, ts)

    # Bottom bar
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, 0, PAGE_W, 10 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(MARGIN, 3.5 * mm, "Confidential — For internal use only")
    canvas.drawRightString(PAGE_W - MARGIN, 3.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ── KPI card row ──────────────────────────────────────────────────────────────
def _kpi_row(kpis: list[tuple[str, str, str]]) -> Table:
    """
    kpis: list of (label, value, status)  where status in 'ok'|'warn'|'danger'|'neutral'
    """
    S = _styles()
    status_color = {"ok": C_SUCCESS, "warn": C_WARNING, "danger": C_DANGER, "neutral": C_ACCENT}
    n = len(kpis)
    col_w = (PAGE_W - 2 * MARGIN) / n

    data = [[
        Table(
            [[Paragraph(v, ParagraphStyle("kv", fontSize=18, textColor=status_color.get(s, C_ACCENT),
                                          fontName="Helvetica-Bold", alignment=TA_CENTER))],
             [Paragraph(l, ParagraphStyle("kl", fontSize=7, textColor=C_MUTED,
                                          fontName="Helvetica", alignment=TA_CENTER))]],
            colWidths=[col_w - 6],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_CARD),
                ("ROUNDEDCORNERS", [6]),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ])
        )
        for l, v, s in kpis
    ]]
    t = Table(data, colWidths=[col_w] * n)
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# ── Status badge ──────────────────────────────────────────────────────────────
def _badge(text: str, kind: str) -> Table:
    color_map = {"ok": C_SUCCESS, "warn": C_WARNING, "danger": C_DANGER}
    bg_map    = {
        "ok":     colors.HexColor("#052e16"),
        "warn":   colors.HexColor("#451a03"),
        "danger": colors.HexColor("#450a0a"),
    }
    S = _styles()
    style_map = {"ok": S["tag_safe"], "warn": S["tag_warn"], "danger": S["tag_danger"]}
    t = Table([[Paragraph(text, style_map.get(kind, S["center"]))]],
              colWidths=[55 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_map.get(kind, C_CARD)),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0.5, color_map.get(kind, C_BORDER)),
    ]))
    return t


# ── Forecast chart → PNG bytes ─────────────────────────────────────────────────
def _chart_to_image(fig: plt.Figure, width_mm: float, height_mm: float) -> RLImage:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    return RLImage(buf, width=width_mm * mm, height=height_mm * mm)


# ── Forecast data table ───────────────────────────────────────────────────────
def _forecast_table(export_df: pd.DataFrame) -> Table:
    preview = export_df.tail(14)[
        ["date", "forecast_demand", "forecast_lower", "forecast_upper", "reorder_required"]
    ].copy()
    preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
    preview["forecast_demand"] = preview["forecast_demand"].round(1)
    preview["forecast_lower"]  = preview["forecast_lower"].round(1)
    preview["forecast_upper"]  = preview["forecast_upper"].round(1)

    headers = ["Date", "Predicted Demand", "Low Estimate", "High Estimate", "Reorder?"]
    col_w = [(PAGE_W - 2 * MARGIN) / 5] * 5

    S = _styles()
    h_style  = ParagraphStyle("th", fontSize=8, textColor=C_ACCENT,
                               fontName="Helvetica-Bold", alignment=TA_CENTER)
    td_style = ParagraphStyle("td", fontSize=8, textColor=C_TEXT,
                               fontName="Helvetica", alignment=TA_CENTER)
    flag_ok  = ParagraphStyle("fok",  fontSize=8, textColor=C_SUCCESS,
                               fontName="Helvetica-Bold", alignment=TA_CENTER)
    flag_bad = ParagraphStyle("fbad", fontSize=8, textColor=C_DANGER,
                               fontName="Helvetica-Bold", alignment=TA_CENTER)

    rows = [[Paragraph(h, h_style) for h in headers]]
    for _, row in preview.iterrows():
        reorder = bool(row["reorder_required"])
        rows.append([
            Paragraph(str(row["date"]),             td_style),
            Paragraph(str(row["forecast_demand"]),  td_style),
            Paragraph(str(row["forecast_lower"]),   td_style),
            Paragraph(str(row["forecast_upper"]),   td_style),
            Paragraph("Yes" if reorder else "No",   flag_bad if reorder else flag_ok),
        ])

    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("BACKGROUND",    (0, 1), (-1, -1), C_DARK),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_DARK, C_CARD]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ── PUBLIC FUNCTION ───────────────────────────────────────────────────────────
def build_pdf_report(
    chart: plt.Figure,
    export_df: pd.DataFrame,
    selected_store: str,
    selected_item: str,
    insights,
    metrics: dict,
) -> bytes:
    buffer = BytesIO()
    S = _styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=18 * mm,
        bottomMargin=14 * mm,
        title=f"Foresight Report — {selected_store} / {selected_item}",
    )

    story = []

    # ── PAGE 1: Cover + KPIs ─────────────────────────────────────────────────

    story.append(Spacer(1, 6 * mm))

    # Title block
    story.append(Paragraph("Inventory Forecast Report", S["h1"]))
    story.append(Paragraph(
        f"Store: <b>{selected_store}</b>  &nbsp;|&nbsp;  SKU: <b>{selected_item}</b>  "
        f"&nbsp;|&nbsp;  Generated: {datetime.now().strftime('%d %b %Y')}",
        S["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=C_ACCENT, spaceAfter=8))

    # Plain-language summary — the headline takeaway, no jargon
    if insights.stockout_alert:
        summary_text = (
            f"<b>Bottom line:</b> Tomorrow's expected demand ({insights.next_day_demand} units) "
            f"is higher than your current stock ({insights.current_stock} units). "
            f"You may run out soon unless you reorder now."
        )
    elif insights.reorder_required:
        summary_text = (
            f"<b>Bottom line:</b> Your stock ({insights.current_stock} units) has dropped below "
            f"the reorder point ({insights.reorder_point} units). It's time to place a new order."
        )
    else:
        summary_text = (
            f"<b>Bottom line:</b> Stock levels look healthy right now. No action needed based on "
            f"current sales predictions."
        )
    story.append(Paragraph(summary_text, S["body"]))
    story.append(Spacer(1, 5 * mm))
    stockout_status = "danger" if insights.stockout_alert else "ok"
    reorder_status  = "warn"   if insights.reorder_required else "ok"
    story.append(Paragraph("Inventory Status", S["h2"]))
    story.append(_kpi_row([
        ("Next-Day Demand",   str(insights.next_day_demand),  "neutral"),
        ("Current Stock",     str(insights.current_stock),    "neutral"),
        ("Lead-Time Demand",  str(insights.future_demand),    "neutral"),
        ("Reorder Point",     str(insights.reorder_point),    reorder_status),
    ]))
    story.append(Spacer(1, 4 * mm))

    # KPI row 2 — forecast reliability, explained in plain language
    mae_val  = f"{metrics['mae']} units"  if metrics.get("mae")  is not None else "N/A"
    rmse_val = f"{metrics['rmse']} units" if metrics.get("rmse") is not None else "N/A"
    story.append(Paragraph("Forecast Reliability", S["h2"]))
    story.append(_kpi_row([
        ("Typical Forecast Error",     mae_val,                              "neutral"),
        ("Size of Occasional Misses",  rmse_val,                             "neutral"),
        ("Avg Daily Demand", f"{insights.average_daily_demand}", "neutral"),
        ("Seasonal Peak Month",
         str(insights.seasonal_peak_month) if insights.seasonal_peak_month else "N/A",
         "neutral"),
    ]))
    story.append(Paragraph(
        "\"Typical Forecast Error\" is how far off predictions usually are, in units per day — "
        "smaller is better. \"Size of Occasional Misses\" shows how big the forecast's rare "
        "bigger mistakes tend to be; it's naturally a bit higher than the typical error.",
        S["muted"],
    ))
    story.append(Spacer(1, 4 * mm))

    # Alert badges
    story.append(Paragraph("Alerts", S["h2"]))
    badge_row = Table(
        [[
            _badge("✓ No Stockout Risk" if not insights.stockout_alert
                   else f"⚠ Stockout Risk — Demand {insights.next_day_demand} > Stock {insights.current_stock}",
                   "ok" if not insights.stockout_alert else "danger"),
            Spacer(6 * mm, 1),
            _badge("✓ Stock Level Healthy" if not insights.reorder_required
                   else f"⚠ Reorder Needed — Below reorder point ({insights.reorder_point})",
                   "ok" if not insights.reorder_required else "warn"),
        ]],
        colWidths=[55 * mm, 6 * mm, 90 * mm],
    )
    badge_row.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(badge_row)
    story.append(Spacer(1, 5 * mm))

    # Forecast chart
    story.append(Paragraph("Demand Forecast", S["h2"]))
    story.append(_chart_to_image(chart, width_mm=170, height_mm=70))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Teal line = your actual past sales &nbsp;|&nbsp; Blue line = predicted future demand "
        "&nbsp;|&nbsp; Light blue shaded area = the range demand will most likely fall within "
        "(actual results should land inside this band about 19 times out of 20)",
        S["muted"],
    ))

    story.append(PageBreak())

    # ── PAGE 2: Forecast Table + Notes ───────────────────────────────────────

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("14-Day Forecast Detail", S["h2"]))
    story.append(Paragraph(
        "This shows the next 14 days of predicted demand. If \"Reorder?\" says \"Yes\", your "
        "stock is expected to fall below the reorder point on that day — it's time to place a "
        "new order before that happens.",
        S["muted"],
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(_forecast_table(export_df))
    story.append(Spacer(1, 6 * mm))

    # Reorder point explanation — plain language, no formula notation
    formula_data = [[
        Paragraph("What Is the Reorder Point?", ParagraphStyle(
            "fh", fontSize=9, textColor=C_ACCENT, fontName="Helvetica-Bold")),
        Paragraph(
            f"It's the stock level that should trigger a new order. It's built from three things: "
            f"your average daily sales ({insights.average_daily_demand:.1f} units/day), how long "
            f"it takes to receive a new order, and an extra safety cushion in case demand spikes "
            f"or a delivery runs late. Putting those together, your reorder point is "
            f"<b>{insights.reorder_point} units</b> — once stock drops to this level, it's time "
            f"to reorder.",
            ParagraphStyle("fb", fontSize=9, textColor=C_TEXT,
                           fontName="Helvetica", leading=14)),
    ]]
    formula_table = Table(formula_data, colWidths=[52 * mm, 118 * mm])
    formula_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_CARD),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_ACCENT),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LINEAFTER",     (0, 0), (0, -1),  0.5, C_BORDER),
    ]))
    story.append(formula_table)
    story.append(Spacer(1, 6 * mm))

    # Footer note
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_BORDER, spaceAfter=4))
    story.append(Paragraph(
        "This report was generated automatically by the Foresight Inventory Intelligence System. "
        "Forecasts are based on historical sales data and statistical modelling — "
        "actual demand may vary. Review with your procurement team before placing orders.",
        S["muted"],
    ))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buffer.seek(0)
    return buffer.getvalue()