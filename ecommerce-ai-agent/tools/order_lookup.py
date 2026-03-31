"""
tools/order_lookup.py
LangChain tool that looks up an order in the database by order ID.
The customer agent calls this automatically when a customer mentions an order.
"""
from langchain.tools import tool
from database.db import SessionLocal
from database.models import Order


@tool
def order_lookup(order_id: str) -> str:
    """
    Look up the status and details of a customer order by order ID.
    Use this when the customer asks about their order, delivery, or shipping.
    The order ID format is ORD-XXXXX (example: ORD-10005).
    Input should be just the order ID string.
    """
    db = SessionLocal()
    try:
        # Normalize: handle lowercase and spaces
        clean_id = order_id.strip().upper()
        order = db.query(Order).filter(Order.order_id == clean_id).first()

        if not order:
            return (
                f"No order found with ID '{clean_id}'. "
                f"Please check the order ID. It should look like ORD-10005."
            )

        return (
            f"Order Found:\n"
            f"  Order ID   : {order.order_id}\n"
            f"  Product    : {order.product_name}\n"
            f"  Quantity   : {order.quantity} unit(s)\n"
            f"  Status     : {order.status.upper()}\n"
            f"  Amount     : Rs {order.price:,.0f}\n"
            f"  Order Date : {order.created_at.strftime('%d %B %Y')}"
        )
    finally:
        db.close()
