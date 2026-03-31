"""
database/models.py
SQLAlchemy ORM models for all database tables.
Each class maps to one table in the SQLite database.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.sql import func
from database.db import Base


class Order(Base):
    """
    Stores e-commerce orders.
    The customer agent queries this table to look up order status.
    """
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, unique=True, index=True)
    customer_name = Column(String)
    customer_email = Column(String)
    product_name = Column(String)
    quantity = Column(Integer)
    price = Column(Float)
    status = Column(String)   # pending / shipped / delivered / cancelled
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Product(Base):
    """
    Stores the product catalog with current stock levels.
    Used by the inventory forecasting agent.
    """
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, unique=True, index=True)
    name = Column(String)
    category = Column(String)
    price = Column(Float)
    stock_quantity = Column(Integer)
    restock_threshold = Column(Integer, default=50)
    created_at = Column(DateTime, default=func.now())


class SalesRecord(Base):
    """
    Daily sales records per product.
    This is the training data for Prophet (forecasting) and
    Isolation Forest (anomaly detection) models.
    """
    __tablename__ = "sales_records"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, index=True)
    date = Column(DateTime, index=True)
    units_sold = Column(Integer)
    revenue = Column(Float)
    created_at = Column(DateTime, default=func.now())


class Alert(Base):
    """
    Stores all alerts from both inventory and anomaly agents.
    Displayed on the Streamlit dashboard in real-time.
    """
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String)       # "anomaly" or "restock"
    severity = Column(String)         # "low", "medium", "high"
    product_id = Column(String)
    message = Column(Text)
    is_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())


class CustomerQuery(Base):
    """
    Logs every customer query and the agent's response.
    Used for the feedback loop and conversation history on the dashboard.
    """
    __tablename__ = "customer_queries"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text)
    intent = Column(String)
    response = Column(Text)
    confidence = Column(Float)
    escalated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
