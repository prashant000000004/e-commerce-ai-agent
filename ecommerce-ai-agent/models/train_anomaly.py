"""
models/train_anomaly.py
Trains an Isolation Forest to detect unusual sales patterns.
Isolation Forest is a local ML model — no internet, no GPU needed.

RUN: python models/train_anomaly.py
"""
import os
import sys
import pickle
import pandas as pd
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models/saved")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_engineer_features() -> pd.DataFrame:
    """
    Load all sales data and create ML features.
    Feature engineering is critical — better features = better anomaly detection.
    """
    from database.db import SessionLocal
    from database.models import SalesRecord

    db = SessionLocal()
    try:
        records = db.query(SalesRecord).all()
        df = pd.DataFrame([{
            "product_id": r.product_id,
            "date": pd.to_datetime(r.date),
            "units_sold": float(r.units_sold),
            "revenue": float(r.revenue)
        } for r in records])

        df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

        # Rolling stats — captures "is this unusual compared to recent history?"
        grp = df.groupby("product_id")["units_sold"]
        df["rolling_mean_7d"] = grp.transform(lambda x: x.rolling(7, min_periods=1).mean())
        df["rolling_std_7d"]  = grp.transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))

        # Calendar features — day of week and month matter for retail
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        # Deviation from rolling mean — the most powerful anomaly signal
        df["deviation_from_mean"] = df["units_sold"] - df["rolling_mean_7d"]

        logger.info(f"Feature engineering complete. {len(df)} records.")
        return df
    finally:
        db.close()


def train_isolation_forest(df: pd.DataFrame):
    """
    Train Isolation Forest. contamination=0.05 means we expect
    ~5% of data points to be anomalies (matches our synthetic data).
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("anomaly_detection")

    feature_cols = [
        "units_sold", "rolling_mean_7d", "rolling_std_7d",
        "day_of_week", "month", "deviation_from_mean"
    ]
    X = df[feature_cols].fillna(0)

    with mlflow.start_run(run_name="isolation_forest_v1"):
        # Scale features to zero mean and unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1        # Use all CPU cores
        )
        model.fit(X_scaled)

        # Log results
        preds = model.predict(X_scaled)
        n_anomalies = (preds == -1).sum()
        anomaly_pct = n_anomalies / len(preds) * 100

        mlflow.log_params({
            "n_estimators": 100,
            "contamination": 0.05,
            "training_samples": len(df),
            "features": str(feature_cols)
        })
        mlflow.log_metrics({
            "detected_anomalies": int(n_anomalies),
            "anomaly_pct": round(anomaly_pct, 2)
        })
        logger.info(f"Detected {n_anomalies} anomalies ({anomaly_pct:.1f}%)")

        # Save model, scaler, and feature list to disk
        for name, obj in [
            ("isolation_forest.pkl", model),
            ("scaler.pkl", scaler),
            ("feature_cols.pkl", feature_cols)
        ]:
            path = os.path.join(MODEL_DIR, name)
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            mlflow.log_artifact(path)

        logger.success("Anomaly model saved.")
    return model, scaler


if __name__ == "__main__":
    df = load_and_engineer_features()
    train_isolation_forest(df)
    print("\nAnomaly model trained! View: mlflow ui")
