"""
agents/inventory_agent.py
Loads the trained Prophet models and forecasts inventory demand for all products.
Creates stockout alerts in the database when forecasted demand exceeds current stock.
"""
import os
import sys
import pickle
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models/saved")


class InventoryForecastAgent:
    """
    Forecasts 30-day product demand using Facebook Prophet.
    Compares forecast against current stock and fires alerts for at-risk products.
    Run models/train_forecast.py before using this agent.
    """

    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load all trained Prophet .pkl files from the models/saved directory."""
        for pid in ["P001", "P002", "P003", "P004", "P005"]:
            path = os.path.join(MODEL_DIR, f"prophet_{pid}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.models[pid] = pickle.load(f)
                logger.info(f"Loaded forecast model: {pid}")
            else:
                logger.warning(f"No model for {pid}. Run: python models/train_forecast.py")

    def forecast(self, product_id: str, periods: int = 30) -> pd.DataFrame:
        """
        Generate a demand forecast for a single product.
        Returns only future rows (not historical fitted values).
        All negative predictions are clipped to 0 (can't sell negative units).
        """
        if product_id not in self.models:
            return pd.DataFrame()

        # Ask Prophet to extend the timeline by `periods` days
        future = self.models[product_id].make_future_dataframe(periods=periods)
        forecast = self.models[product_id].predict(future)

        # Keep only future dates
        future_fc = forecast[forecast["ds"] >= datetime.now()][
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()

        # Clip and round — units sold must be non-negative integers
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            future_fc[col] = future_fc[col].clip(lower=0).round(0).astype(int)

        return future_fc

    def check_all_products(self) -> list:
        """
        Run forecasts for all 5 products. Compare predicted 30-day demand
        against current stock. Create Alert records for any stockout risks.
        Returns a list of result dicts — one per product.
        """
        from database.db import SessionLocal
        from database.models import Product, Alert

        results = []
        db = SessionLocal()
        try:
            for product in db.query(Product).all():
                fc = self.forecast(product.product_id, periods=30)
                if fc.empty:
                    continue

                total_demand = int(fc["yhat"].sum())
                stock = product.stock_quantity

                # Walk forward through the forecast to estimate days until stockout
                days_to_stockout = None
                cumulative = 0
                for _, row in fc.iterrows():
                    cumulative += row["yhat"]
                    if cumulative >= stock:
                        days_to_stockout = max(0, (row["ds"] - datetime.now()).days)
                        break

                will_stockout = total_demand > stock

                if will_stockout:
                    # High severity if stockout is within 7 days
                    severity = (
                        "high"
                        if days_to_stockout is not None and days_to_stockout <= 7
                        else "medium"
                    )
                    msg = (
                        f"STOCKOUT RISK: {product.name} — "
                        f"30-day forecast ({total_demand} units) exceeds "
                        f"current stock ({stock} units). "
                        f"Estimated stockout in {days_to_stockout} day(s)."
                    )
                    db.add(Alert(
                        alert_type="restock",
                        severity=severity,
                        product_id=product.product_id,
                        message=msg
                    ))
                    logger.warning(f"Stockout risk: {product.name} ({product.product_id})")

                results.append({
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "current_stock": stock,
                    "forecasted_demand_30d": total_demand,
                    "will_stockout": will_stockout,
                    "days_until_stockout": days_to_stockout,
                    "forecast_df": fc,
                })

            db.commit()
        finally:
            db.close()

        return results


if __name__ == "__main__":
    agent = InventoryForecastAgent()
    results = agent.check_all_products()
    print(f"\nInventory Forecast Results:")
    print(f"{'─'*65}")
    for r in results:
        status = "⚠  STOCKOUT RISK" if r["will_stockout"] else "✓  OK"
        print(
            f"[{status}] {r['product_name']:<22} "
            f"Stock: {r['current_stock']:>4}  |  "
            f"30d Forecast: {r['forecasted_demand_30d']:>5}"
        )
