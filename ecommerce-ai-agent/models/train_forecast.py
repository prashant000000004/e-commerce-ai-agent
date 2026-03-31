"""
models/train_forecast.py
Trains a Facebook Prophet time-series model on historical sales data.
Prophet is 100% local — it runs on your CPU, no internet needed.
Logs all metrics and parameters to MLflow for experiment tracking.

RUN: python models/train_forecast.py
"""
import os
import sys
import pickle
import warnings
import pandas as pd
import mlflow
from prophet import Prophet
from dotenv import load_dotenv
from loguru import logger

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MODEL_DIR = os.getenv("MODEL_DIR", "./models/saved")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_sales_data(product_id: str) -> pd.DataFrame:
    """
    Load sales records for a product from the database.
    Prophet requires columns named exactly 'ds' (date) and 'y' (target value).
    """
    from database.db import SessionLocal
    from database.models import SalesRecord

    db = SessionLocal()
    try:
        records = db.query(SalesRecord).filter(
            SalesRecord.product_id == product_id
        ).all()

        df = pd.DataFrame([{
            "ds": pd.to_datetime(r.date),
            "y": float(r.units_sold)
        } for r in records])

        df = df.sort_values("ds").reset_index(drop=True)
        logger.info(f"Loaded {len(df)} records for {product_id}")
        return df
    finally:
        db.close()


def train_prophet(df: pd.DataFrame, product_id: str):
    """
    Train Prophet with seasonality and log everything to MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("inventory_forecasting")

    with mlflow.start_run(run_name=f"prophet_{product_id}"):
        # Prophet config — weekly and yearly seasonality works well for e-commerce
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )

        mlflow.log_params({
            "product_id": product_id,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "training_rows": len(df)
        })

        logger.info(f"Training Prophet for {product_id}...")
        model.fit(df)

        # Forecast 30 days into the future
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Calculate MAPE on the last 30 training days
        last_30 = df.tail(30).copy()
        forecast_last30 = forecast[forecast["ds"].isin(last_30["ds"])]
        if len(forecast_last30) > 0:
            merged = last_30.merge(forecast_last30[["ds", "yhat"]], on="ds")
            # Avoid division by zero
            non_zero = merged[merged["y"] > 0]
            if len(non_zero) > 0:
                mape = (abs(non_zero["y"] - non_zero["yhat"]) / non_zero["y"]).mean() * 100
                mlflow.log_metric("mape_last_30_days", round(mape, 2))
                logger.info(f"MAPE: {mape:.2f}%")

        # Save model to disk
        model_path = os.path.join(MODEL_DIR, f"prophet_{product_id}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)
        logger.success(f"Saved: {model_path}")

    return model, forecast


def main():
    product_ids = ["P001", "P002", "P003", "P004", "P005"]
    for pid in product_ids:
        df = load_sales_data(pid)
        if len(df) >= 10:
            train_prophet(df, pid)
        else:
            logger.warning(f"Not enough data for {pid}. Skipping.")

    print("\nAll models trained! View results: mlflow ui")


if __name__ == "__main__":
    main()
