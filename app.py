from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import plotly.graph_objects as go
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from models import predict_linear, train_linear_model
from sklearn.metrics import mean_squared_error

from main import (
    CANONICAL_COLUMNS,
    DEFAULT_DATASET_PATH,
    BusinessLogic,
    DataProcessor,
    ForecastModel,
)
from sku_comparison import render_sku_comparison_from_cache
from pdf_report import build_pdf_report

# ── Dark theme for all matplotlib charts (forecast, decomposition, model
# comparison) so they match the rest of the dark UI instead of rendering
# with a white background. Applied globally so every chart picks it up
# without needing to be styled individually.
MPL_BG = "#0d1117"
MPL_GRID = "#21262d"
MPL_TEXT = "#e6edf3"
plt.rcParams.update({
    "figure.facecolor": MPL_BG,
    "axes.facecolor": MPL_BG,
    "savefig.facecolor": MPL_BG,
    "axes.edgecolor": MPL_GRID,
    "axes.labelcolor": MPL_TEXT,
    "text.color": MPL_TEXT,
    "xtick.color": MPL_TEXT,
    "ytick.color": MPL_TEXT,
    "grid.color": MPL_GRID,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": MPL_GRID,
    "legend.labelcolor": MPL_TEXT,
})

# In your main app, wherever you want the section:
#render_sku_comparison(df, forecast_df)
st.markdown("""
<style>
/* Background */
body {
    background-color: #0e1117;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #2d3139;
    text-align: center;
}

/* Section Titles */
h2, h3 {
    color: #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Inventory Demand Forecasting System", layout="wide")

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "foresight_logo_transparent.svg"

processor = DataProcessor()
logic = BusinessLogic()

@st.cache_data(show_spinner=False)
def build_all_sku_forecasts_cached(
    _dataset: pd.DataFrame,       # leading _ = not hashed (DataFrame hashing is slow/unreliable)
    dataset_hash: str,            # we hash manually — see STEP 2
    selected_pairs: tuple[tuple[str, str], ...],
    forecast_days: int,
    lead_time_days: int,
    safety_stock: int,
) -> dict:
    """
    Runs Prophet only for the selected (store, item) pairs.
    Cached by (dataset_hash, selected_pairs, forecast_days, lead_time_days, safety_stock).
    Survives sidebar store/item changes and page interactions.
    Only re-runs if the dataset, the selected SKUs, or the three parameters change.
    """
    from sku_comparison import _build_all_forecasts
    processor = DataProcessor()
    logic = BusinessLogic()
    return _build_all_forecasts(
        _dataset, processor, logic, forecast_days, lead_time_days, safety_stock,
        skus=selected_pairs,
    )

@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return processor.load_data(DEFAULT_DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_uploaded_raw_data(file_bytes: bytes) -> pd.DataFrame:
    return processor.read_raw_data(StringIO(file_bytes.decode("utf-8")))


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes, mapping_items: tuple[tuple[str, str | None], ...]) -> pd.DataFrame:
    return processor.load_data(
        StringIO(file_bytes.decode("utf-8")),
        column_mapping=dict(mapping_items),
    )


def generate_ai_insights(components, residuals):
    insights = {}

    # 📈 TREND
    trend_start = components['trend'].iloc[0]
    trend_end = components['trend'].iloc[-1]

    if trend_end > trend_start * 1.05:
        insights["trend"] = "📈 Demand shows a clear upward trend — product popularity is increasing."
    elif trend_end < trend_start * 0.95:
        insights["trend"] = "📉 Demand is declining — possible drop in interest or market shift."
    else:
        insights["trend"] = "➡️ Demand is relatively stable with no major long-term trend."

    # 📅 WEEKLY
    if 'weekly' in components:
        weekly_avg = components.groupby(components['ds'].dt.dayofweek)['weekly'].mean()
        peak_day = weekly_avg.idxmax()
        low_day = weekly_avg.idxmin()

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        insights["weekly"] = f"📅 Highest demand on {days[peak_day]}, lowest on {days[low_day]}."

    # 🌍 YEARLY
    if 'yearly' in components:
        monthly_avg = components.groupby(components['ds'].dt.month)['yearly'].mean()
        peak_month = monthly_avg.idxmax()

        insights["yearly"] = f"🌍 Peak seasonal demand occurs around month {peak_month}."

    # ⚠️ RESIDUAL
    volatility = residuals.std()

    if volatility > 10:
        insights["residual"] = "⚠️ Demand is highly volatile — frequent unexpected spikes."
    elif volatility > 5:
        insights["residual"] = "🔄 Moderate variability in demand."
    else:
        insights["residual"] = "✅ Demand is stable and predictable."

    return insights

def build_forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    selected_store: str,
    selected_item: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history["date"], history["sales"], label="Actual Sales", linewidth=2, color="#2dd4bf")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", linewidth=2, color="#60a5fa")
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="#93c5fd",
        alpha=0.35,
        label="95% Confidence Interval",
    )

    ax.set_title(f"Demand Forecast for {selected_store} / {selected_item}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig



st.image(str(LOGO_PATH), width=780)

#with col1:

#with col2:
    # st.markdown("""
    # # DemandIQ
    # #### AI-Powered Demand Intelligence Platform
    # """)


uploaded_file = st.file_uploader("", type=["csv"])
mapping_items: tuple[tuple[str, str | None], ...] = tuple()

try:
    if uploaded_file is not None:
        raw_uploaded_df = load_uploaded_raw_data(uploaded_file.getvalue())
        inferred_mapping = processor.infer_column_mapping(raw_uploaded_df)

        with st.expander("Uploaded file column mapping", expanded=True):
            st.caption(
                "Review how this file maps to the fields used by the forecasting engine. "
                "You only need to change these if the automatic guess is wrong."
            )
            selected_mapping: dict[str, str | None] = {}
            available_columns = [None, *raw_uploaded_df.columns.tolist()]
            display_columns = ["<Not set>", *raw_uploaded_df.columns.tolist()]

            for canonical in CANONICAL_COLUMNS:
                inferred_value = inferred_mapping.get(canonical)
                default_index = 0
                if inferred_value in raw_uploaded_df.columns:
                    default_index = raw_uploaded_df.columns.tolist().index(inferred_value) + 1

                selected_value = st.selectbox(
                    f"{canonical} column",
                    options=available_columns,
                    index=default_index,
                    format_func=lambda value: "<Not set>" if value is None else str(value),
                    key=f"mapping_{canonical}",
                )
                selected_mapping[canonical] = selected_value

            st.dataframe(raw_uploaded_df.head(10), use_container_width=True, hide_index=True)

        mapping_items = tuple(sorted(selected_mapping.items()))
        dataset = load_uploaded_data(uploaded_file.getvalue(), mapping_items)
        data_source_label = f"Uploaded file: {uploaded_file.name}"
    else:
        dataset = load_default_data()
        data_source_label = f"Default dataset: {DEFAULT_DATASET_PATH}"
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

dataset_hash = str(len(dataset)) + "_" + str(dataset["date"].max()) + "_" + str(dataset["store_id"].nunique())

st.markdown("## 📊 Executive Summary")

st.sidebar.image(str(LOGO_PATH), width=350)
st.sidebar.markdown("### ⚙️ Control Panel")
store_options = sorted(dataset["store_id"].unique(), key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)
selected_store = st.sidebar.selectbox("Store", store_options)

item_options = sorted(dataset.loc[dataset["store_id"] == selected_store, "item_id"].unique(), key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)
selected_item = st.sidebar.selectbox("Item / SKU", item_options)

store_item_data = processor.filter_series(dataset, selected_store, selected_item)
if store_item_data.empty:
    st.warning("No data available for the selected store and item.")
    st.stop()

min_date = store_item_data["date"].min().date()
max_date = store_item_data["date"].max().date()
selected_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date = min_date
    end_date = max_date

filtered_series = processor.filter_series(
    dataset,
    selected_store,
    selected_item,
    start_date=pd.Timestamp(start_date),
    end_date=pd.Timestamp(end_date),
)

if len(filtered_series) < 30:
    st.warning("Select a wider date range. At least 30 rows are recommended for forecasting.")
    st.stop()

forecast_days = st.sidebar.slider("Forecast horizon (days)", min_value=30, max_value=180, value=90, step=15)
lead_time_days = st.sidebar.slider("Lead time (days)", min_value=1, max_value=30, value=4)
safety_stock = st.sidebar.number_input("Safety stock", min_value=0, max_value=5000, value=30, step=5)

estimated_stock = int(round(filtered_series["sales"].tail(14).mean() * 2))
use_estimated_stock = st.sidebar.checkbox("Use estimated current stock", value=True)
if use_estimated_stock:
    current_stock = estimated_stock
    st.sidebar.caption(f"Estimated stock based on recent sales: {estimated_stock}")
else:
    current_stock = st.sidebar.number_input(
        "Current stock",
        min_value=0,
        max_value=100000,
        value=estimated_stock,
        step=10,
    )

st.sidebar.subheader("🔮 What-if Simulation")

future_price = st.sidebar.slider("Future Price", 10, 100, 20)
future_promo = st.sidebar.selectbox("Future Promotion", [0, 1])

sku_summary = processor.summarize_skus(dataset)
top_skus = sku_summary.head(10)



with st.spinner("🚀 Running AI Forecast Engine..."):
    prophet_data = processor.prepare_for_prophet(filtered_series)
    artifacts = ForecastModel(forecast_days=forecast_days).fit_predict(prophet_data)
    artifacts.forecast['yhat'] = artifacts.forecast['yhat'] * (
    1 + 0.05 * (future_promo) - 0.01 * (future_price / 10)
)
    insights = logic.calculate_inventory_insights(
        df_series=filtered_series,
        forecast=artifacts.forecast,
        lead_time_days=lead_time_days,
        safety_stock=int(safety_stock),
        current_stock=int(current_stock),
    )
    export_df = logic.build_export_frame(
        store_id=selected_store,
        item_id=selected_item,
        forecast=artifacts.forecast,
        insights=insights,
    )
    linear_model = train_linear_model(prophet_data)
    linear_preds = predict_linear(linear_model, prophet_data, forecast_days)
# -------------------------------
# MODEL COMPARISON
# -------------------------------


col1, col2, col3, col4 = st.columns(4)

col1.metric("📈 Next-Day Demand", insights.next_day_demand)
col2.metric("📦 Current Stock", insights.current_stock)
col3.metric("⚠️ Reorder Point", insights.reorder_point)
col4.metric("📊 Lead-Time Demand", insights.future_demand)

eval_col_1, eval_col_2, eval_col_3 = st.columns(3)
eval_col_1.metric("MAE", artifacts.metrics["mae"] if artifacts.metrics["mae"] is not None else "N/A")
eval_col_2.metric("RMSE", artifacts.metrics["rmse"] if artifacts.metrics["rmse"] is not None else "N/A")
eval_col_3.metric(
    "Holdout Days",
    artifacts.metrics["holdout_days"] if artifacts.metrics["holdout_days"] is not None else "N/A",
)

left_col, right_col = st.columns([1, 1])

with left_col:

    if insights.stockout_alert:
        st.error(f"⚠️ Stockout Risk! Demand ({insights.next_day_demand}) > Stock ({insights.current_stock})")
    else:
        st.success("✅ No Stockout Risk")

with right_col:

    if insights.reorder_required:
        st.warning(f"📦 Reorder Needed! Stock below reorder point ({insights.reorder_point})")
    else:
        st.success("✅ Stock Level Healthy")
if insights.is_high_demand_sku:
    st.info("This SKU is currently behaving like a high-demand product based on recent sales.")

forecast_chart = build_forecast_chart(
    history=filtered_series,
    forecast=artifacts.forecast,
    selected_store=selected_store,
    selected_item=selected_item,
)
st.markdown("## 📊 Forecast & Analysis")
left_col, right_col = st.columns([2.5, 1])  # give chart more space

# LEFT → Chart
with left_col:

    st.pyplot(forecast_chart, use_container_width=True)
# RIGHT → Metrics Panel
with right_col:
   
    # Card 1
    with st.container(border=True):
        st.metric("Average Daily Demand", insights.average_daily_demand)
    # Card 2
    with st.container(border=True):
        st.metric("High-Demand Threshold", insights.high_demand_threshold)
    # Card 3
    with st.container(border=True):
        st.metric(
            "Seasonal Peak Month",
            insights.seasonal_peak_month if insights.seasonal_peak_month else "N/A",
        )
residuals = prophet_data['y'] - artifacts.forecast['yhat'][:len(prophet_data)]


#st.markdown("### 📊 Demand Decomposition")

components = artifacts.forecast
ai_insights = generate_ai_insights(components, residuals)

st.markdown("### 📊 Demand Decomposition")

fig, axes = plt.subplots(2, 2, figsize=(10, 5))

# Trend
axes[0, 0].plot(components['ds'], components['trend'])
axes[0, 0].set_title("Trend")

# Weekly
if 'weekly' in components:
    axes[0, 1].plot(components['ds'], components['weekly'])
    axes[0, 1].set_title("Weekly")

# Yearly
if 'yearly' in components:
    axes[1, 0].plot(components['ds'], components['yearly'])
    axes[1, 0].set_title("Yearly")

# Residual
axes[1, 1].plot(prophet_data['ds'], residuals, color='red')
axes[1, 1].set_title("Residual")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
st.markdown("### 🧠 AI Insights")

col1, col2 = st.columns(2)

with col1:
    st.info(ai_insights.get("trend", ""))
    st.info(ai_insights.get("yearly", ""))

with col2:
    st.info(ai_insights.get("weekly", ""))
    st.info(ai_insights.get("residual", ""))
st.markdown("---")

actual = prophet_data['y'].tail(30).values
prophet_pred = artifacts.forecast['yhat'][:len(prophet_data)].tail(30).values
linear_pred = linear_preds[:30]

rmse_prophet = np.sqrt(mean_squared_error(actual, prophet_pred))
rmse_linear = np.sqrt(mean_squared_error(actual, linear_pred))

best_model = "Prophet" if rmse_prophet < rmse_linear else "Linear Regression"


st.markdown("## 🤖 Model Performance")

left_col, right_col = st.columns([2.5, 1])

with left_col:

    fig2, ax2 = plt.subplots(figsize=(6,2))

# Actual
    ax2.plot(prophet_data['ds'], prophet_data['y'], label='Actual', color='#e5e7eb')

# Prophet
    ax2.plot(artifacts.forecast['ds'], artifacts.forecast['yhat'], label='Prophet', color='#60a5fa')

# Linear
    linear_dates = artifacts.forecast['ds'].iloc[len(prophet_data):len(prophet_data)+len(linear_preds)]
    ax2.plot(linear_dates, linear_preds, label='Linear Regression', color='#fb923c')

    ax2.legend()
    ax2.set_title("Model Comparison")

    st.pyplot(fig2)


with right_col:

    
    with st.container(border=True):
        st.metric("Prophet RMSE", round(rmse_prophet, 2))
    with st.container(border=True):
        st.metric("Linear RMSE", round(rmse_linear, 2))
    with st.container(border=True):
        st.metric("🏆 Best Model", best_model)



if rmse_prophet < rmse_linear:
    st.success("✅ Prophet performing better")
else:
    st.warning("⚠️ Linear model performing better")



st.markdown("---")

left_col, right_col = st.columns(2)

# LEFT → Residuals
with left_col:
    st.markdown("### 📉 Residual Analysis")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=residuals,
        mode='lines',
        name='Residuals'
    ))

    fig1.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        height=350
    )

    st.plotly_chart(fig1, use_container_width=True)


# RIGHT → Anomalies
with right_col:
    st.markdown("### 🚨 Anomaly Detection")

    threshold = residuals.std() * 2
    anomalies = prophet_data[abs(residuals) > threshold]

    fig2 = go.Figure()

    # main line
    fig2.add_trace(go.Scatter(
        x=prophet_data['ds'],
        y=prophet_data['y'],
        mode='lines',
        name='Actual'
    ))

    # anomalies
    fig2.add_trace(go.Scatter(
        x=anomalies['ds'],
        y=anomalies['y'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=6)
    ))

    fig2.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        height=350
    )

    st.plotly_chart(fig2, use_container_width=True)

st.write("Anomalies detected:", len(anomalies))
st.markdown("---")

# Multi-SKU comparison fits Prophet twice per SKU (evaluation + forecast), so
# forecasting every SKU in a large dataset (thousands of store/item pairs) can
# take hours. Instead of computing all of them, the user picks a bounded subset
# up front — ranked by total sales — and only that subset is ever forecast.
# It's gated behind a button so it never blocks the rest of the page from
# rendering, and the result is kept in session_state so it doesn't need to be
# recomputed on every rerun.
MAX_SKUS_FOR_COMPARISON = 15

st.markdown("## 🔀 Multi-SKU Comparison")

sku_label_to_pair = {
    f"{store} / {item}": (store, item)
    for store, item in zip(sku_summary["store_id"], sku_summary["item_id"])
}
all_sku_labels = list(sku_label_to_pair.keys())

if len(all_sku_labels) > MAX_SKUS_FOR_COMPARISON:
    st.warning(
        f"This dataset has {len(all_sku_labels)} SKUs. Forecasting all of them would need "
        f"{len(all_sku_labels) * 2} model fits and could take a very long time, so pick up to "
        f"{MAX_SKUS_FOR_COMPARISON} to compare (options are ranked by total sales)."
    )
else:
    st.info(
        "Forecasts the selected SKUs for side-by-side comparison. "
        "This can take a while, so it runs on demand."
    )

default_sku_labels = all_sku_labels[: min(6, len(all_sku_labels))]
selected_sku_labels = st.multiselect(
    "SKUs to forecast (ranked by total sales)",
    options=all_sku_labels,
    default=default_sku_labels,
    max_selections=MAX_SKUS_FOR_COMPARISON,
)
selected_pairs = tuple(sorted(sku_label_to_pair[label] for label in selected_sku_labels))

sku_comparison_key = (selected_pairs, forecast_days, lead_time_days, int(safety_stock))

if "sku_comparison_key" not in st.session_state:
    st.session_state.sku_comparison_key = None
    st.session_state.sku_comparison_data = None

comparison_ready = st.session_state.sku_comparison_key == sku_comparison_key

run_comparison = st.button(
    "🔄 Re-run Multi-SKU Comparison" if comparison_ready else "⚡ Run Multi-SKU Comparison",
    disabled=not selected_pairs,
)

if run_comparison:
    st.session_state.sku_comparison_data = build_all_sku_forecasts_cached(
        dataset,
        dataset_hash,
        selected_pairs,
        forecast_days=forecast_days,
        lead_time_days=lead_time_days,
        safety_stock=int(safety_stock),
    )
    st.session_state.sku_comparison_key = sku_comparison_key
    comparison_ready = True

if comparison_ready:
    render_sku_comparison_from_cache(st.session_state.sku_comparison_data, dataset)

st.markdown("---")

st.subheader("🧠 Insights")

st.write(f"Best performing model is **{best_model}** based on RMSE comparison.")

if future_promo == 1:
    st.info("Promotion applied → Demand is expected to increase.")
else:
    st.info("No promotion → Normal demand expected.")

if future_price > prophet_data['price'].mean():
    st.warning("Higher price may reduce demand.")

pdf_report = build_pdf_report(
    chart=forecast_chart,
    export_df=export_df,
    selected_store=selected_store,
    selected_item=selected_item,
    insights=insights,
    metrics=artifacts.metrics,
)

tab1, tab2, tab3 = st.tabs(["Forecast Report", "Raw Data", "SKU Summary"])

with tab1:
    st.subheader("Download Forecast and Alert Report")
    st.dataframe(
        export_df.tail(forecast_days),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download forecast report as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_report_{selected_store}_{selected_item}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download forecast summary as PDF",
        data=pdf_report,
        file_name=f"forecast_summary_{selected_store}_{selected_item}.pdf",
        mime="application/pdf",
    )

with tab2:
    st.subheader("Cleaned Input Data")
    st.dataframe(filtered_series.tail(50), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Top SKUs by Total Sales")
    st.dataframe(top_skus, use_container_width=True, hide_index=True)

    top_skuss = dataset.groupby("item_id")["sales"].sum().nlargest(10)
    st.bar_chart(top_skuss)