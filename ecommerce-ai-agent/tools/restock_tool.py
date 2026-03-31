"""
tools/restock_tool.py
LangChain tool that checks inventory levels and creates restock alerts.
Called by the inventory agent and the customer agent when stock queries arise.
"""
from langchain.tools import tool
from database.db import SessionLocal
from database.models import Product, Alert
from loguru import logger


@tool
def check_restock_needed(product_id: str) -> str:
    """
    Check if a specific product needs to be restocked based on current stock
    level versus the restock threshold. Creates an alert if restock is needed.
    Input should be a product ID string like P001, P002, etc.
    """
    db = SessionLocal()
    try:
        product = db.query(Product).filter(
            Product.product_id == product_id.strip().upper()
        ).first()

        if not product:
            return f"Product '{product_id}' not found in catalog."

        needs_restock = product.stock_quantity <= product.restock_threshold

        if needs_restock:
            # Determine severity: high if nearly out, medium if just below threshold
            severity = "high" if product.stock_quantity <= 10 else "medium"
            message = (
                f"RESTOCK NEEDED: {product.name} (ID: {product.product_id}) "
                f"— Current stock: {product.stock_quantity} units "
                f"(below threshold of {product.restock_threshold}). "
                f"Severity: {severity.upper()}"
            )
            # Save alert to database so dashboard shows it
            db.add(Alert(
                alert_type="restock",
                severity=severity,
                product_id=product.product_id,
                message=message
            ))
            db.commit()
            logger.warning(f"Restock alert created: {product.product_id}")
            return message

        return (
            f"{product.name}: Stock OK — {product.stock_quantity} units available "
            f"(threshold: {product.restock_threshold})."
        )
    finally:
        db.close()
