"""
database/db.py
Sets up the SQLAlchemy database engine, session factory, and Base class.
All other database files import from here.
Uses SQLite — no database server needed, perfect for local development.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ecommerce.db")

# Create SQLite engine
# check_same_thread=False is required for SQLite when used across threads (Streamlit)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Session factory — always use this to get a database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class — all ORM models must inherit from this
Base = declarative_base()


def get_db():
    """
    Generator function to get a database session.
    Automatically closes the session after use (even if an error occurs).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Creates all database tables defined in models.py.
    Call this once at startup to initialize the database.
    """
    from database.models import Order, Product, SalesRecord, Alert, CustomerQuery
    Base.metadata.create_all(bind=engine)
    print("[DB] All tables created successfully.")
