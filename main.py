from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
from models import predict_linear, train_linear_model

try:
    from prophet import Prophet
    PROPHET_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    Prophet = None
    PROPHET_IMPORT_ERROR = exc


DEFAULT_DATASET_PATH = Path("/Users/DELL/Downloads/retail_sales.csv")
REQUIRED_COLUMNS = {"date", "sales"}
OPTIONAL_DEFAULTS = {
    "store_id": "store_1",
    "item_id": "item_1",
    "price": 0.0,
    "promo": 0,
}
CANONICAL_COLUMNS = ["date", "sales", "store_id", "item_id", "price", "promo"]
COLUMN_ALIASES = {
    "date": [
        "date",
        "transaction_date",
        "order_date",
        "invoice_date",
        "purchase_date",
        "sales_date",
        "day",
        "timestamp",
    ],
    "sales": [
        "sales",
        "quantity",
        "qty",
        "units_sold",
        "units",
        "demand",
        "final_amount",
        "total_amount",
        "revenue",
    ],
    "store_id": [
        "store_id",
        "store",
        "store_name",
        "branch",
        "location",
        "shop",
        "outlet",
    ],
    "item_id": [
        "item_id",
        "item",
        "sku",
        "sku_id",
        "product_name",
        "product",
        "product_id",
        "item_name",
    ],
    "price": [
        "price",
        "unit_price",
        "selling_price",
        "mrp",
        "list_price",
        "cost",
    ],
    "promo": [
        "promo",
        "promotion",
        "is_promo",
        "discount_flag",
        "discount_amount",
        "offer",
        "coupon_used",
    ],
}


@dataclass
class ForecastArtifacts:
    cleaned_data: pd.DataFrame
    modeling_data: pd.DataFrame
    forecast: pd.DataFrame
    metrics: dict[str, float | None]
    future_regressors: dict[str, float]
    model: object


@dataclass
class InventoryInsights:
    current_stock: int
    next_day_demand: int
    future_demand: int
    reorder_point: int
    stockout_alert: bool
    reorder_required: bool
    average_daily_demand: float
    high_demand_threshold: float
    is_high_demand_sku: bool
    seasonal_peak_month: int | None


class DataProcessor:
    def load_data(
        self,
        source: str | Path | BinaryIO | None = None,
        column_mapping: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        resolved_source = DEFAULT_DATASET_PATH if source is None else source
        df = pd.read_csv(resolved_source)
        return self.clean_data(df, column_mapping=column_mapping)

    def read_raw_data(self, source: str | Path | BinaryIO | None = None) -> pd.DataFrame:
        resolved_source = DEFAULT_DATASET_PATH if source is None else source
        return pd.read_csv(resolved_source)

    def clean_data(
        self,
        df: pd.DataFrame,
        column_mapping: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        standardized = self.standardize_columns(df, column_mapping=column_mapping)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing and not REQUIRED_COLUMNS.issubset(standardized.columns):
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(sorted(missing))
            )

        cleaned = standardized.copy()
        cleaned = cleaned.drop_duplicates()
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned["sales"] = pd.to_numeric(cleaned["sales"], errors="coerce")
        cleaned = cleaned.dropna(subset=["date", "sales"])

        for column, default in OPTIONAL_DEFAULTS.items():
            if column not in cleaned.columns:
                cleaned[column] = default

        cleaned["store_id"] = cleaned["store_id"].astype(str)
        cleaned["item_id"] = cleaned["item_id"].astype(str)
        cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce").fillna(0.0)
        cleaned["promo"] = pd.to_numeric(cleaned["promo"], errors="coerce").fillna(0)
        cleaned["promo"] = (cleaned["promo"] > 0).astype(int)
        cleaned["sales"] = cleaned["sales"].clip(lower=0)
        cleaned = (
            cleaned.groupby(["store_id", "item_id", "date"], as_index=False)
            .agg(
                sales=("sales", "sum"),
                price=("price", "mean"),
                promo=("promo", "max"),
            )
        )
        cleaned = self.expand_to_daily_series(cleaned)
        cleaned = cleaned.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        return cleaned

    def expand_to_daily_series(self, df: pd.DataFrame) -> pd.DataFrame:
        expanded_groups: list[pd.DataFrame] = []

        for (store_id, item_id), group in df.groupby(["store_id", "item_id"], sort=False):
            series = group.sort_values("date").set_index("date")
            full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
            daily = series.reindex(full_index)
            daily.index.name = "date"

            daily["store_id"] = store_id
            daily["item_id"] = item_id
            daily["sales"] = daily["sales"].fillna(0.0)
            daily["price"] = daily["price"].ffill().bfill().fillna(0.0)
            daily["promo"] = daily["promo"].fillna(0).astype(int)

            expanded_groups.append(daily.reset_index())

        if not expanded_groups:
            return df

        return pd.concat(expanded_groups, ignore_index=True)

    def infer_column_mapping(self, df: pd.DataFrame) -> dict[str, str | None]:
        normalized_lookup = {
            self._normalize_column_name(column): column for column in df.columns
        }
        mapping: dict[str, str | None] = {}

        for canonical in CANONICAL_COLUMNS:
            chosen = None
            for alias in COLUMN_ALIASES.get(canonical, []):
                if alias in normalized_lookup:
                    chosen = normalized_lookup[alias]
                    break
            mapping[canonical] = chosen

        return mapping

    def standardize_columns(
        self,
        df: pd.DataFrame,
        column_mapping: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        mapping = self.infer_column_mapping(df)
        if column_mapping:
            mapping.update(column_mapping)

        standardized = pd.DataFrame()
        for canonical in ["date", "sales"]:
            source_column = mapping.get(canonical)
            if not source_column or source_column not in df.columns:
                raise ValueError(
                    f"Could not map the required '{canonical}' field. "
                    f"Available columns: {', '.join(df.columns)}"
                )
            standardized[canonical] = df[source_column]

        for canonical, default in OPTIONAL_DEFAULTS.items():
            source_column = mapping.get(canonical)
            if source_column and source_column in df.columns:
                standardized[canonical] = df[source_column]
            else:
                standardized[canonical] = default

        if mapping.get("promo") == "discount_amount":
            standardized["promo"] = pd.to_numeric(standardized["promo"], errors="coerce").fillna(0)

        return standardized

    def _normalize_column_name(self, value: str) -> str:
        return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")

    def filter_series(
        self,
        df: pd.DataFrame,
        store_id: str,
        item_id: str,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        filtered = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].copy()
        if start_date is not None:
            filtered = filtered[filtered["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            filtered = filtered[filtered["date"] <= pd.Timestamp(end_date)]
        return filtered.sort_values("date").reset_index(drop=True)

    def prepare_for_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df[["date", "sales", "price", "promo"]].rename(
            columns={"date": "ds", "sales": "y"}
        )
        prepared["price"] = prepared["price"].astype(float)
        prepared["promo"] = prepared["promo"].astype(int)
        return prepared

    def summarize_skus(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            df.groupby(["store_id", "item_id"], as_index=False)
            .agg(
                total_sales=("sales", "sum"),
                avg_daily_sales=("sales", "mean"),
                latest_date=("date", "max"),
                promo_days=("promo", "sum"),
            )
            .sort_values("total_sales", ascending=False)
        )
        summary["avg_daily_sales"] = summary["avg_daily_sales"].round(2)
        return summary


class ForecastModel:
    def __init__(self, forecast_days: int = 90) -> None:
        self.forecast_days = forecast_days

    def ensure_prophet_available(self) -> None:
        if Prophet is None:
            raise ModuleNotFoundError(
                "The 'prophet' package is not available in the Python environment running this app. "
                "Install it with 'python3 -m pip install prophet' and restart Streamlit."
            ) from PROPHET_IMPORT_ERROR

    def build_model(self) -> Prophet:
        self.ensure_prophet_available()
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.2,
            seasonality_prior_scale=10.0,
            interval_width=0.95,
        )
        model.add_regressor("price")
        model.add_regressor("promo")
        return model

    def fit_predict(self, df_prophet: pd.DataFrame) -> ForecastArtifacts:
        if len(df_prophet) < 30:
            raise ValueError("At least 30 daily records are recommended for forecasting.")

        metrics = self.evaluate(df_prophet)
        future_regressors = self._future_regressors(df_prophet)

        full_model = self.build_model()
        full_model.fit(df_prophet)

        future = full_model.make_future_dataframe(periods=self.forecast_days)
        future["price"] = future_regressors["price"]
        future["promo"] = future_regressors["promo"]
        history_len = len(df_prophet)
        future.loc[: history_len - 1, "price"] = df_prophet["price"].to_numpy()
        future.loc[: history_len - 1, "promo"] = df_prophet["promo"].to_numpy()

        forecast = full_model.predict(future)
        return ForecastArtifacts(
            cleaned_data=df_prophet.rename(columns={"ds": "date", "y": "sales"}),
            modeling_data=df_prophet,
            forecast=forecast,
            metrics=metrics,
            future_regressors=future_regressors,
            model=full_model, 
        )

    def evaluate(self, df_prophet: pd.DataFrame) -> dict[str, float | None]:
        holdout_days = min(30, max(14, len(df_prophet) // 5))
        if len(df_prophet) <= holdout_days + 30:
            return {"mae": None, "rmse": None, "holdout_days": None}

        train = df_prophet.iloc[:-holdout_days].copy()
        test = df_prophet.iloc[-holdout_days:].copy()

        model = self.build_model()
        model.fit(train)

        future = model.make_future_dataframe(periods=holdout_days)
        future["price"] = float(train["price"].tail(14).mean())
        future["promo"] = int(round(train["promo"].tail(14).mean()))
        future.loc[: len(train) - 1, "price"] = train["price"].to_numpy()
        future.loc[: len(train) - 1, "promo"] = train["promo"].to_numpy()

        predictions = model.predict(future).tail(holdout_days)
        actual = test["y"].to_numpy()
        predicted = predictions["yhat"].to_numpy()

        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "holdout_days": holdout_days}

    def _future_regressors(self, df_prophet: pd.DataFrame) -> dict[str, float]:
        return {
            "price": float(df_prophet["price"].tail(14).mean()),
            "promo": int(round(df_prophet["promo"].tail(14).mean())),
        }


class BusinessLogic:
    def calculate_inventory_insights(
        self,
        df_series: pd.DataFrame,
        forecast: pd.DataFrame,
        lead_time_days: int = 4,
        safety_stock: int = 30,
        current_stock: int | None = None,
    ) -> InventoryInsights:
        if df_series.empty:
            raise ValueError("Cannot generate inventory insights for an empty series.")

        estimated_stock = int(round(df_series["sales"].tail(14).mean() * 2))
        stock_level = estimated_stock if current_stock is None else int(current_stock)

        future_slice = forecast.tail(len(forecast) - len(df_series))
        next_day_demand = max(int(round(float(future_slice["yhat"].iloc[0]))), 0)
        future_demand = max(
            int(round(float(future_slice["yhat"].head(lead_time_days).sum()))),
            0,
        )
        avg_daily_demand = float(df_series["sales"].mean())
        demand_std = df_series["sales"].std()
        z = 1.65  # ~95% service level

        reorder_point = int(round(
            (avg_daily_demand * lead_time_days) +
            (z * demand_std * np.sqrt(lead_time_days)) +
            safety_stock
        ))

        high_demand_threshold = float(df_series["sales"].quantile(0.9))
        is_high_demand = bool(df_series["sales"].tail(30).mean() >= high_demand_threshold)

        monthly = df_series.assign(month=df_series["date"].dt.month).groupby("month")["sales"].mean()
        seasonal_peak_month = int(monthly.idxmax()) if not monthly.empty else None

        return InventoryInsights(
            current_stock=stock_level,
            next_day_demand=next_day_demand,
            future_demand=future_demand,
            reorder_point=reorder_point,
            stockout_alert=next_day_demand > stock_level,
            reorder_required=stock_level < reorder_point,
            average_daily_demand=round(avg_daily_demand, 2),
            high_demand_threshold=round(high_demand_threshold, 2),
            is_high_demand_sku=is_high_demand,
            seasonal_peak_month=seasonal_peak_month,
        )

    def build_export_frame(
        self,
        store_id: str,
        item_id: str,
        forecast: pd.DataFrame,
        insights: InventoryInsights,
    ) -> pd.DataFrame:
        export = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        export["store_id"] = store_id
        export["item_id"] = item_id
        export["current_stock"] = insights.current_stock
        export["reorder_point"] = insights.reorder_point
        export["stockout_alert"] = insights.stockout_alert
        export["reorder_required"] = insights.reorder_required
        export = export.rename(
            columns={
                "ds": "date",
                "yhat": "forecast_demand",
                "yhat_lower": "forecast_lower",
                "yhat_upper": "forecast_upper",
            }
        )
        return export


from sklearn.metrics import mean_squared_error

def run_cli_demo() -> None:
    processor = DataProcessor()
    df = processor.load_data()

    # -------------------------------
    # SELECT FIRST STORE + ITEM
    # -------------------------------
    first_row = df[["store_id", "item_id"]].drop_duplicates().iloc[0]
    series = processor.filter_series(df, first_row["store_id"], first_row["item_id"])
    prophet_df = processor.prepare_for_prophet(series)

    # -------------------------------
    # PROPHET MODEL
    # -------------------------------
    artifacts = ForecastModel(forecast_days=90).fit_predict(prophet_df)
    forecast = artifacts.forecast

    # -------------------------------
    # LINEAR MODEL
    # -------------------------------
    linear_model = train_linear_model(prophet_df)
    linear_preds = predict_linear(linear_model, prophet_df, future_days=90)

    # -------------------------------
    # MODEL COMPARISON (LAST 30 DAYS)
    # -------------------------------
    actual = prophet_df['y'].tail(30).values

    prophet_pred = forecast['yhat'][:len(prophet_df)].tail(30).values
    linear_pred = linear_preds[:30]

    rmse_prophet = np.sqrt(mean_squared_error(actual, prophet_pred))
    rmse_linear = np.sqrt(mean_squared_error(actual, linear_pred))

    print("\n📊 MODEL PERFORMANCE")
    print(f"Prophet RMSE: {round(rmse_prophet,2)}")
    print(f"Linear RMSE: {round(rmse_linear,2)}")

    # -------------------------------
    # SELECT BEST MODEL
    # -------------------------------
    if rmse_prophet < rmse_linear:
        best_model = "Prophet"
        best_forecast = forecast['yhat']
    else:
        best_model = "Linear Regression"
        best_forecast = linear_preds

    print(f"\n🏆 Best Model: {best_model}")

    # -------------------------------
    # BUSINESS LOGIC (USE PROPHET)
    # -------------------------------
    # (Important: business logic still uses Prophet for consistency)
    insights = BusinessLogic().calculate_inventory_insights(
        df_series=series,
        forecast=forecast,
    )

    # -------------------------------
    # OUTPUT
    # -------------------------------
    print("\n📦 INVENTORY INSIGHTS")
    print(f"Store: {first_row['store_id']}")
    print(f"Item: {first_row['item_id']}")
    print(f"Next Day Demand: {insights.next_day_demand}")
    print(f"Current Stock: {insights.current_stock}")
    print(f"Forecasted Demand (Lead Time): {insights.future_demand}")
    print(f"Reorder Point: {insights.reorder_point}")
    print(f"Stockout Alert: {insights.stockout_alert}")
    print(f"Reorder Required: {insights.reorder_required}")
if __name__ == "__main__":
    run_cli_demo()
