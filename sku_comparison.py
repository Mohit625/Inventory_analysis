"""
sku_comparison.py  —  Multi-SKU Comparison for Foresight
---------------------------------------------------------
See app_patch.py for the three changes needed in app.py.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from main import DataProcessor, ForecastModel, BusinessLogic

SKU_COLORS = ["#4FC3F7", "#FF7043", "#66BB6A", "#FFA726", "#AB47BC", "#26C6DA"]


def _sku_label(store_id: str, item_id: str) -> str:
    return f"{store_id} / {item_id}"


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Per-SKU forecast — cached at the individual SKU level ────────────────────
# Keyed by (store_id, item_id, forecast_days, lead_time_days, safety_stock)
# so changing sidebar params only re-runs SKUs whose params changed.

@st.cache_data(show_spinner=False)
def _forecast_one_sku(
    _df_series: pd.DataFrame,
    store_id: str,
    item_id: str,
    forecast_days: int,
    lead_time_days: int,
    safety_stock: int,
) -> dict:
    processor = DataProcessor()
    logic = BusinessLogic()
    prophet_df = processor.prepare_for_prophet(_df_series)
    try:
        artifacts = ForecastModel(forecast_days=forecast_days).fit_predict(prophet_df)
    except Exception:
        return {}
    insights = logic.calculate_inventory_insights(
        df_series=_df_series,
        forecast=artifacts.forecast,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock,
    )
    export = logic.build_export_frame(
        store_id=store_id,
        item_id=item_id,
        forecast=artifacts.forecast,
        insights=insights,
    )
    return {"export": export, "insights": insights, "series": _df_series}


# ── Build forecasts for ALL SKUs — called once from app.py's @st.cache_data ──

from joblib import Parallel, delayed
import streamlit as st


def _process_single_sku(args):
    dataset, store, item, processor, forecast_days, lead_time_days, safety_stock = args
    
    series = processor.filter_series(dataset, store, item)
    
    if len(series) < 30:
        return None
    
    key = f"{store}_{item}"
    
    data = _forecast_one_sku(
        series, store, item, forecast_days, lead_time_days, safety_stock
    )
    
    if data:
        return key, data
    
    return None


def _build_all_forecasts(
    dataset: pd.DataFrame,
    processor: DataProcessor,
    logic: BusinessLogic,
    forecast_days: int,
    lead_time_days: int,
    safety_stock: int,
) -> dict[str, dict]:

    results = {}

    skus = dataset[["store_id", "item_id"]].drop_duplicates().values.tolist()

    progress = st.progress(0, text="⚡ Running parallel forecasts...")

    # Prepare arguments
    args_list = [
        (dataset, store, item, processor, forecast_days, lead_time_days, safety_stock)
        for store, item in skus
    ]

    # 🔥 PARALLEL EXECUTION
    output = Parallel(n_jobs=-1)(
        delayed(_process_single_sku)(args)
        for args in args_list
    )

    # Collect results + update progress
    valid_results = [res for res in output if res is not None]

    for idx, res in enumerate(valid_results):
        key, data = res
        results[key] = data
        progress.progress((idx + 1) / len(valid_results))

    progress.empty()

    return results


# ── Charts ────────────────────────────────────────────────────────────────────

def _plot_forecast_overlay(all_data: dict, chosen: list, normalize: bool) -> go.Figure:
    fig = go.Figure()
    for i, key in enumerate(chosen):
        color = SKU_COLORS[i % len(SKU_COLORS)]
        export = all_data[key]["export"].copy()
        export = export[export["date"] >= export["date"].max() - pd.Timedelta(days=120)]
        y, ylo, yhi = export["forecast_demand"], export["forecast_lower"], export["forecast_upper"]
        if normalize:
            mx = y.max()
            if mx > 0:
                y, ylo, yhi = y / mx * 100, ylo / mx * 100, yhi / mx * 100
        fig.add_trace(go.Scatter(
            x=pd.concat([export["date"], export["date"][::-1]]),
            y=pd.concat([yhi, ylo[::-1]]),
            fill="toself", fillcolor=_hex_to_rgba(color, 0.10),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=export["date"], y=y, mode="lines", name=key,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{key}</b><br>%{{x|%Y-%m-%d}}<br>{'%' if normalize else 'units'}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Demand Forecast Overlay",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=12),
        xaxis=dict(gridcolor="#21262d", title="Date"),
        yaxis=dict(gridcolor="#21262d", title="% of peak" if normalize else "Units"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _build_kpi_table(all_data: dict, chosen: list) -> pd.DataFrame:
    rows = []
    for key in chosen:
        ins = all_data[key]["insights"]
        avg = ins.average_daily_demand
        days_out = int(ins.current_stock / avg) if avg > 0 else 999
        if ins.current_stock < ins.reorder_point * 0.5:
            status = "🔴 Critical"
        elif ins.reorder_required:
            status = "🟡 Reorder"
        else:
            status = "🟢 Safe"
        rows.append({
            "SKU": key,
            "Avg Daily Demand": f"{avg:.1f}",
            "Current Stock": ins.current_stock,
            "Reorder Point": ins.reorder_point,
            "Days to Stockout": days_out if days_out < 999 else "N/A",
            "Status": status,
        })
    return pd.DataFrame(rows)


def _plot_demand_bar(all_data: dict, chosen: list) -> go.Figure:
    pairs = [
        (all_data[k]["insights"].average_daily_demand, k, SKU_COLORS[i % len(SKU_COLORS)])
        for i, k in enumerate(chosen)
    ]
    pairs.sort(reverse=True)
    vals, labels, colors = zip(*pairs) if pairs else ([], [], [])
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(labels), orientation="h",
        marker_color=list(colors),
        text=[f"{v:.1f}" for v in vals], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg: %{x:.1f} units/day<extra></extra>",
    ))
    fig.update_layout(
        title="Average Daily Demand Comparison",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=12),
        xaxis=dict(gridcolor="#21262d", title="Units / Day"),
        yaxis=dict(gridcolor="#21262d"),
        margin=dict(l=20, r=60, t=40, b=20),
        height=max(250, len(labels) * 55),
    )
    return fig


def _plot_stock_gauges(all_data: dict, chosen: list) -> go.Figure:
    n = len(chosen)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
        subplot_titles=chosen,
    )
    for idx, key in enumerate(chosen):
        r, c = idx // cols + 1, idx % cols + 1
        ins = all_data[key]["insights"]
        color = SKU_COLORS[idx % len(SKU_COLORS)]
        axis_max = max(ins.reorder_point * 1.5, ins.current_stock * 1.2, 1)
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=ins.current_stock,
            gauge=dict(
                axis=dict(range=[0, axis_max], tickcolor="#e6edf3"),
                bar=dict(color=color), bgcolor="#161b22", bordercolor="#30363d",
                threshold=dict(line=dict(color="#FF7043", width=3), thickness=0.85, value=ins.reorder_point),
                steps=[
                    dict(range=[0, ins.reorder_point], color="#1c1c2e"),
                    dict(range=[ins.reorder_point, axis_max], color="#162032"),
                ],
            ),
            title=dict(text="Stock", font=dict(color="#8b949e", size=11)),
            number=dict(font=dict(color="#e6edf3")),
        ), row=r, col=c)
    fig.update_layout(
        paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
        margin=dict(l=20, r=20, t=50, b=20), height=260 * rows,
        title="Stock vs Reorder Point  (orange line = reorder threshold)",
    )
    return fig


def _plot_seasonality(all_data: dict, chosen: list) -> go.Figure:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure()
    for i, key in enumerate(chosen):
        series = all_data[key]["series"].copy()
        if "sales" not in series.columns:
            continue
        series["dow"] = pd.to_datetime(series["date"]).dt.dayofweek
        weekly = series.groupby("dow")["sales"].mean().reindex(range(7), fill_value=0)
        color = SKU_COLORS[i % len(SKU_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=weekly.values.tolist() + [weekly.values[0]],
            theta=days + [days[0]],
            fill="toself", name=key,
            line=dict(color=color),
            fillcolor=_hex_to_rgba(color, 0.15),
        ))
    fig.update_layout(
        title="Weekly Demand Pattern (Radar)",
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(gridcolor="#30363d", color="#8b949e"),
            angularaxis=dict(gridcolor="#30363d", color="#e6edf3"),
        ),
        paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ── MAIN RENDER FUNCTION ──────────────────────────────────────────────────────

def render_sku_comparison_from_cache(all_data: dict, dataset: pd.DataFrame):
    """
    Receives pre-built forecast data from app.py's @st.cache_data call.
    No re-processing happens here — only UI rendering.
    """
    st.markdown("Compare demand forecasts, stock health, and weekly patterns across all SKUs.")

    sku_labels = list(all_data.keys())

    if not sku_labels:
        st.warning("No SKUs with enough data (≥30 days) to forecast.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        chosen = st.multiselect(
            "Select SKUs to compare (up to 6)",
            options=sku_labels,
            default=sku_labels[:min(3, len(sku_labels))],
            max_selections=6,
        )
    with col2:
        normalize = st.toggle(
            "Normalize to %", value=False,
            help="Scale each SKU to 0–100% to compare patterns regardless of volume",
        )

    if not chosen:
        st.info("Select at least one SKU above to begin comparison.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast Overlay",
        "📊 Demand Ranking",
        "🟡 Stock Gauges",
        "🌀 Seasonality",
    ])

    with tab1:
        st.plotly_chart(_plot_forecast_overlay(all_data, chosen, normalize), use_container_width=True)
        st.markdown("#### KPI Summary")
        st.dataframe(_build_kpi_table(all_data, chosen), use_container_width=True, hide_index=True)
    with tab2:
        st.plotly_chart(_plot_demand_bar(all_data, chosen), use_container_width=True)
    with tab3:
        st.plotly_chart(_plot_stock_gauges(all_data, chosen), use_container_width=True)
        st.caption("Orange line = reorder point. Bar below it = reorder needed.")
    with tab4:
        st.plotly_chart(_plot_seasonality(all_data, chosen), use_container_width=True)

    st.markdown("---")
    st.download_button(
        label="⬇️ Download Comparison Report (CSV)",
        data=_build_kpi_table(all_data, chosen).to_csv(index=False).encode("utf-8"),
        file_name="sku_comparison_report.csv",
        mime="text/csv",
    )