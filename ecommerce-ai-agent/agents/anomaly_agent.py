"""
agents/anomaly_agent.py
Loads the trained Isolation Forest model and scans recent sales data for anomalies.
Any detected anomaly (spike or drop) is saved as an Alert in the database.
Run models/train_anomaly.py before using this agent.
"""
import os
import sys
import pickle
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models/saved")


class AnomalyDetectionAgent:
    """
    Detects unusual sales events using a pre-trained Isolation Forest model.
    Features used: units_sold, rolling 7-day stats, day_of_week, month, deviation.
    Feature engineering must exactly match what was used during training.
    """

    def __init__(self):
        # Three artifacts are saved during training; we need all three
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self._load_models()

    def _load_models(self):
        """Load the Isolation Forest, StandardScaler, and feature column list."""
        artifacts = [
            ("isolation_forest.pkl", "model"),
            ("scaler.pkl", "scaler"),
            ("feature_cols.pkl", "feature_cols"),
        ]
        for filename, attr in artifacts:
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    setattr(self, attr, pickle.load(f))
                logger.info(f"Loaded: {filename}")
            else:
                logger.error(f"Missing: {filename}. Run: python models/train_anomaly.py")

    def detect(self, days_back: int = 30) -> list:
        """
        Scan the last `days_back` days of sales data for anomalies.
        Returns a list of dicts describing each detected anomaly.
        Also writes Alert records to the database.
        """
        if self.model is None:
            logger.error("No model loaded. Returning empty results.")
            return []

        from database.db import SessionLocal
        from database.models import SalesRecord, Alert

        db = SessionLocal()
        try:
            # Load ALL historical sales (needed for rolling stats)
            records = db.query(SalesRecord).all()
            df = pd.DataFrame([{
                "product_id": r.product_id,
                "date": pd.to_datetime(r.date),
                "units_sold": float(r.units_sold),
                "revenue": float(r.revenue),
            } for r in records])

            df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

            # ── Feature engineering (must match train_anomaly.py exactly) ─────
            grp = df.groupby("product_id")["units_sold"]
            df["rolling_mean_7d"] = grp.transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df["rolling_std_7d"] = grp.transform(
                lambda x: x.rolling(7, min_periods=1).std().fillna(0)
            )
            df["day_of_week"] = df["date"].dt.dayofweek
            df["month"] = df["date"].dt.month
            df["deviation_from_mean"] = df["units_sold"] - df["rolling_mean_7d"]

            # Filter to recent window only for detection output
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
            recent = df[df["date"] >= cutoff].copy()

            if recent.empty:
                logger.warning("No recent sales data found.")
                return []

            # ── Run the model ──────────────────────────────────────────────────
            X = recent[self.feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)
            recent = recent.copy()
            recent["prediction"] = self.model.predict(X_scaled)     # 1=normal, -1=anomaly
            recent["score"] = self.model.score_samples(X_scaled)    # lower = more anomalous

            anomalies = recent[recent["prediction"] == -1]
            logger.info(f"Detected {len(anomalies)} anomalies in last {days_back} days.")

            results = []
            for _, row in anomalies.iterrows():
                # Classify whether the anomaly is a sales spike or a drop
                direction = "spike" if row["deviation_from_mean"] > 0 else "drop"
                # Score < -0.2 is a stronger anomaly signal → high severity
                severity = "high" if row["score"] < -0.2 else "medium"

                msg = (
                    f"ANOMALY ({direction.upper()}): {row['product_id']} — "
                    f"{row['units_sold']:.0f} units sold "
                    f"(7-day avg: {row['rolling_mean_7d']:.0f}) "
                    f"on {row['date'].strftime('%d %b %Y')}. "
                    f"Anomaly score: {row['score']:.3f}"
                )

                # Save to alerts table for the dashboard
                db.add(Alert(
                    alert_type="anomaly",
                    severity=severity,
                    product_id=row["product_id"],
                    message=msg,
                ))

                results.append({
                    "product_id": row["product_id"],
                    "date": row["date"],
                    "units_sold": row["units_sold"],
                    "expected": row["rolling_mean_7d"],
                    "direction": direction,
                    "severity": severity,
                    "score": row["score"],
                })

            db.commit()
            return results

        finally:
            db.close()


if __name__ == "__main__":
    agent = AnomalyDetectionAgent()
    anomalies = agent.detect(days_back=60)
    print(f"\nFound {len(anomalies)} anomalies in the last 60 days:")
    print(f"{'─'*70}")
    for a in anomalies[:10]:
        print(
            f"  [{a['severity'].upper():<6}] {a['product_id']}  "
            f"{a['date'].date()}  "
            f"{a['units_sold']:>5.0f} units  "
            f"(expected {a['expected']:>4.0f})  "
            f"{a['direction'].upper()}"
        )
