"""
scripts/generate_sample_data.py
Generates realistic fake e-commerce data so the project works
without downloading any external dataset.

RUN THIS FIRST: python scripts/generate_sample_data.py
"""
import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_db, SessionLocal
from database.models import Order, Product, SalesRecord

# ── Product catalog ────────────────────────────────────────────────────────────
PRODUCTS = [
    ("P001", "Wireless Headphones", "Electronics", 1299.0),
    ("P002", "Running Shoes",        "Footwear",    2499.0),
    ("P003", "Yoga Mat",             "Sports",       799.0),
    ("P004", "Coffee Maker",         "Appliances",  3499.0),
    ("P005", "Backpack",             "Accessories", 1499.0),
]

STATUSES = ["pending", "shipped", "delivered", "cancelled"]
NAMES = [
    "Rahul Sharma", "Priya Singh", "Amit Kumar", "Sneha Patel",
    "Vikram Rao", "Anjali Gupta", "Rohit Verma", "Neha Joshi",
    "Karan Mehta", "Divya Nair"
]


def generate_products():
    """Create 5 products with varying stock levels."""
    return [
        Product(
            product_id=pid,
            name=name,
            category=cat,
            price=price,
            stock_quantity=random.randint(10, 500),
            restock_threshold=50
        )
        for pid, name, cat, price in PRODUCTS
    ]


def generate_orders(n=200):
    """Create n random orders spread over the last 90 days."""
    orders = []
    for i in range(n):
        product = random.choice(PRODUCTS)
        qty = random.randint(1, 5)
        orders.append(Order(
            order_id=f"ORD-{10000 + i}",
            customer_name=random.choice(NAMES),
            customer_email=f"user{i}@example.com",
            product_name=product[1],
            quantity=qty,
            price=product[3] * qty,
            status=random.choice(STATUSES),
            created_at=datetime.now() - timedelta(days=random.randint(0, 90))
        ))
    return orders


def generate_sales_records(days=365):
    """
    Generate 1 year of daily sales per product.
    Includes:
    - Weekend boost (40% more sales Sat/Sun)
    - December holiday season (2x sales)
    - Summer boost (June-July, 20% more)
    - Random anomalies (spikes and drops) for the anomaly agent to find
    """
    records = []
    base_date = datetime.now() - timedelta(days=days)

    for pid, name, cat, price in PRODUCTS:
        for day in range(days):
            date = base_date + timedelta(days=day)
            base = 20

            # Seasonal patterns
            if date.weekday() >= 5:      base = int(base * 1.4)   # Weekend
            if date.month == 12:         base = int(base * 2.0)   # Christmas
            if date.month in [6, 7]:     base = int(base * 1.2)   # Summer

            units = max(1, int(np.random.poisson(base)))

            # Inject artificial anomalies (~2% of days have spikes, ~1% have drops)
            if random.random() < 0.02:   units = units * random.randint(5, 10)  # spike
            if random.random() < 0.01:   units = 0                               # dropout

            records.append(SalesRecord(
                product_id=pid,
                date=date,
                units_sold=units,
                revenue=units * price
            ))

    return records


def main():
    print("Step 1: Initializing database...")
    init_db()

    db = SessionLocal()
    try:
        # Clear old data
        db.query(SalesRecord).delete()
        db.query(Order).delete()
        db.query(Product).delete()
        db.commit()

        print("Step 2: Creating products...")
        db.add_all(generate_products())
        db.commit()

        print("Step 3: Creating 200 orders...")
        db.add_all(generate_orders(200))
        db.commit()

        print("Step 4: Creating 1 year of sales data (this takes ~10 seconds)...")
        records = generate_sales_records(365)
        db.bulk_save_objects(records)
        db.commit()

        print(f"\nDone! Created:")
        print(f"  - 5 products")
        print(f"  - 200 orders")
        print(f"  - {len(records)} daily sales records")
        print(f"\nNext step: python models/train_forecast.py")

    finally:
        db.close()


if __name__ == "__main__":
    main()
