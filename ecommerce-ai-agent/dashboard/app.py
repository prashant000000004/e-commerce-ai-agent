"""
dashboard/app.py
Streamlit real-time dashboard for the ShopAI e-commerce intelligence system.

4 Tabs:
  💬 Customer Support  — AI chat interface powered by the customer agent
  📦 Inventory         — Stock levels, sales trends, 30-day forecast status
  🚨 Alerts            — Live anomaly and restock alerts from both ML agents
  📊 Metrics           — Business KPIs: orders, revenue, order status breakdown

Run:  streamlit run dashboard/app.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ShopAI — E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide",
)


# ── Cached data loaders (auto-refresh every 60 seconds) ───────────────────────

@st.cache_data(ttl=60)
def load_orders() -> pd.DataFrame:
    """Load all orders from the database."""
    from database.db import SessionLocal
    from database.models import Order
    db = SessionLocal()
    try:
        rows = db.query(Order).all()
        return pd.DataFrame([{
            "order_id": o.order_id,
            "customer": o.customer_name,
            "product": o.product_name,
            "status": o.status,
            "price": o.price,
            "date": o.created_at,
        } for o in rows])
    finally:
        db.close()


@st.cache_data(ttl=60)
def load_alerts() -> pd.DataFrame:
    """Load the 100 most recent alerts."""
    from database.db import SessionLocal
    from database.models import Alert
    db = SessionLocal()
    try:
        rows = db.query(Alert).order_by(Alert.created_at.desc()).limit(100).all()
        return pd.DataFrame([{
            "type": a.alert_type,
            "severity": a.severity,
            "product": a.product_id,
            "message": a.message,
            "resolved": a.is_resolved,
            "time": a.created_at,
        } for a in rows])
    finally:
        db.close()


@st.cache_data(ttl=60)
def load_sales() -> pd.DataFrame:
    """Load all daily sales records."""
    from database.db import SessionLocal
    from database.models import SalesRecord
    db = SessionLocal()
    try:
        rows = db.query(SalesRecord).all()
        return pd.DataFrame([{
            "product_id": r.product_id,
            "date": r.date,
            "units_sold": r.units_sold,
            "revenue": r.revenue,
        } for r in rows])
    finally:
        db.close()


@st.cache_data(ttl=60)
def load_products() -> pd.DataFrame:
    """Load product catalog with current stock levels."""
    from database.db import SessionLocal
    from database.models import Product
    db = SessionLocal()
    try:
        rows = db.query(Product).all()
        return pd.DataFrame([{
            "product_id": p.product_id,
            "name": p.name,
            "category": p.category,
            "stock": p.stock_quantity,
            "threshold": p.restock_threshold,
            "price": p.price,
        } for p in rows])
    finally:
        db.close()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 ShopAI")
    st.caption("E-Commerce AI Intelligence Dashboard")
    st.divider()

    # Show HuggingFace connection status
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token and hf_token != "your_huggingface_token_here":
        st.success("🤖 HuggingFace: Connected")
        st.caption(f"Model: `{os.getenv('HF_MODEL', 'zephyr-7b-beta')}`")
    else:
        st.warning("⚠️ HuggingFace: Not configured\nRule-based mode is active.")
        st.caption("Add HF_TOKEN to .env to enable AI mode.")

    st.divider()
    st.subheader("Run Agents")

    # Inventory check button
    if st.button("📦 Run Inventory Check", use_container_width=True):
        with st.spinner("Forecasting demand..."):
            from orchestrator.graph import run_agent
            result = run_agent("inventory_check")
            msg = result.get("agent_response", "Done.")
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                st.success(msg)
            st.cache_data.clear()

    # Anomaly scan button
    if st.button("🔍 Run Anomaly Scan", use_container_width=True):
        with st.spinner("Scanning for anomalies..."):
            from orchestrator.graph import run_agent
            result = run_agent("anomaly_scan")
            msg = result.get("agent_response", "Done.")
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                st.success(msg)
            st.cache_data.clear()

    st.divider()
    st.caption(f"🕐 Page loaded: {datetime.now().strftime('%H:%M:%S')}")


# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Customer Support",
    "📦 Inventory",
    "🚨 Alerts",
    "📊 Metrics",
])


# ── Tab 1: Customer Support Chat ───────────────────────────────────────────────
with tab1:
    st.header("Customer Support Agent")

    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token and hf_token != "your_huggingface_token_here":
        st.info("🤖 Mode: HuggingFace AI (Zephyr-7B) — natural language understanding active")
    else:
        st.warning(
            "📋 Mode: Rule-based (offline). "
            "Add `HF_TOKEN` to your `.env` file for full AI mode."
        )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "👋 Hi! I'm your ShopAI support agent. How can I help you today?\n\n"
                "You can ask me about:\n"
                "- **Order status** — e.g. *'Where is my order ORD-10005?'*\n"
                "- **Returns & refunds** — e.g. *'How do I return an item?'*\n"
                "- **Shipping** — e.g. *'How long does delivery take?'*\n"
                "- **Payments** — e.g. *'Can I pay with UPI?'*"
            ),
        }]

    # Render all previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    if prompt := st.chat_input("Type your question here..."):
        # Show the user's message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the agent and show response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from orchestrator.graph import run_agent
                result = run_agent("customer_query", prompt)
                response = result.get("agent_response") or "Sorry, I couldn't process that."

            st.markdown(response)

            # Show metadata below the response
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.caption(f"Intent: `{result.get('intent', 'unknown')}`")
            meta_col2.caption(f"Mode: `{result.get('mode', 'unknown')}`")
            if result.get("escalated"):
                meta_col3.warning("⚠️ Escalated to human support")

        st.session_state.messages.append({"role": "assistant", "content": response})


# ── Tab 2: Inventory ───────────────────────────────────────────────────────────
with tab2:
    st.header("📦 Inventory & Demand Forecast")

    products_df = load_products()
    sales_df = load_sales()

    if products_df.empty:
        st.warning("No product data found. Run: `python scripts/generate_sample_data.py`")
        st.stop()

    # KPI metrics row
    low_stock = products_df[products_df["stock"] <= products_df["threshold"]]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", len(products_df))
    col2.metric("Low Stock", len(low_stock), delta=f"-{len(low_stock)} need restock" if len(low_stock) else None, delta_color="inverse")
    col3.metric("Total Stock Units", f"{products_df['stock'].sum():,}")
    col4.metric("Avg Stock / Product", f"{int(products_df['stock'].mean()):,}")

    st.divider()

    # Stock level bar chart
    fig = px.bar(
        products_df,
        x="name",
        y="stock",
        color="stock",
        color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
        title="Current Stock Levels by Product",
        labels={"name": "Product", "stock": "Units in Stock"},
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="red",
        annotation_text="Restock threshold (50 units)",
        annotation_position="top right",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Sales trend selector
    if not sales_df.empty:
        st.subheader("Sales Trend (Last 90 Days)")
        selected_pid = st.selectbox(
            "Select product:",
            options=products_df["product_id"].tolist(),
            format_func=lambda pid: (
                products_df[products_df["product_id"] == pid]["name"].values[0]
            ),
        )
        product_sales = sales_df[sales_df["product_id"] == selected_pid].copy()
        product_sales["date"] = pd.to_datetime(product_sales["date"])
        product_sales = product_sales[
            product_sales["date"] >= datetime.now() - timedelta(days=90)
        ].sort_values("date")

        if not product_sales.empty:
            fig2 = px.line(
                product_sales,
                x="date",
                y="units_sold",
                title=f"Daily Units Sold — Last 90 Days",
                labels={"date": "Date", "units_sold": "Units Sold"},
            )
            fig2.update_traces(line_color="#3498db")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No sales data in the last 90 days for this product.")


# ── Tab 3: Alerts ──────────────────────────────────────────────────────────────
with tab3:
    st.header("🚨 Anomaly & Restock Alerts")

    alerts_df = load_alerts()

    if alerts_df.empty:
        st.info(
            "No alerts yet. Use the sidebar buttons to run the Inventory Check "
            "or Anomaly Scan to generate alerts."
        )
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Alerts", len(alerts_df))
        col2.metric("High Severity", len(alerts_df[alerts_df["severity"] == "high"]))
        col3.metric("Medium Severity", len(alerts_df[alerts_df["severity"] == "medium"]))
        col4.metric("Unresolved", len(alerts_df[~alerts_df["resolved"]]))

        st.divider()

        # Filter controls
        fcol1, fcol2 = st.columns(2)
        type_filter = fcol1.selectbox("Filter by type:", ["All", "anomaly", "restock"])
        sev_filter = fcol2.selectbox("Filter by severity:", ["All", "high", "medium", "low"])

        filtered = alerts_df.copy()
        if type_filter != "All":
            filtered = filtered[filtered["type"] == type_filter]
        if sev_filter != "All":
            filtered = filtered[filtered["severity"] == sev_filter]

        st.caption(f"Showing {len(filtered)} of {len(alerts_df)} alerts")

        # Render each alert as an expander
        severity_icons = {"high": "🔴", "medium": "🟠", "low": "🟢"}
        for _, row in filtered.iterrows():
            icon = severity_icons.get(row["severity"], "⚪")
            label = (
                f"{icon} [{row['type'].upper()}] "
                f"Product {row['product']} — "
                f"{row['time'].strftime('%d %b %Y %H:%M') if row['time'] else 'N/A'}"
            )
            with st.expander(label):
                st.write(row["message"])
                if row["resolved"]:
                    st.success("✅ Resolved")
                else:
                    st.warning("⏳ Pending resolution")


# ── Tab 4: Business Metrics ────────────────────────────────────────────────────
with tab4:
    st.header("📊 Business Metrics")

    orders_df = load_orders()
    sales_df = load_sales()

    if orders_df.empty:
        st.warning("No order data found. Run: `python scripts/generate_sample_data.py`")
        st.stop()

    # Top-line KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{len(orders_df):,}")
    col2.metric("Total Revenue", f"₹{orders_df['price'].sum():,.0f}")
    col3.metric("Delivered", len(orders_df[orders_df["status"] == "delivered"]))
    col4.metric("Pending", len(orders_df[orders_df["status"] == "pending"]))

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    # Order status pie chart
    with chart_col1:
        status_counts = orders_df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_pie = px.pie(
            status_counts,
            values="count",
            names="status",
            title="Order Status Breakdown",
            color_discrete_map={
                "delivered": "#27ae60",
                "shipped": "#3498db",
                "pending": "#f39c12",
                "cancelled": "#e74c3c",
            },
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Daily revenue line chart
    with chart_col2:
        if not sales_df.empty:
            sales_df["date"] = pd.to_datetime(sales_df["date"])
            daily_rev = (
                sales_df.groupby("date")["revenue"]
                .sum()
                .reset_index()
            )
            daily_rev = daily_rev[
                daily_rev["date"] >= datetime.now() - timedelta(days=30)
            ]
            fig_rev = px.line(
                daily_rev,
                x="date",
                y="revenue",
                title="Daily Revenue — Last 30 Days",
                labels={"date": "Date", "revenue": "Revenue (₹)"},
            )
            fig_rev.update_traces(line_color="#9b59b6", fill="tozeroy", fillcolor="rgba(155,89,182,0.1)")
            st.plotly_chart(fig_rev, use_container_width=True)

    # Revenue by product bar chart
    st.subheader("Revenue by Product (Last 30 Days)")
    if not sales_df.empty:
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        last30 = sales_df[sales_df["date"] >= datetime.now() - timedelta(days=30)]
        rev_by_product = last30.groupby("product_id")["revenue"].sum().reset_index()
        # Merge with product names
        rev_by_product = rev_by_product.merge(
            products_df[["product_id", "name"]], on="product_id", how="left"
        )
        fig_prod = px.bar(
            rev_by_product,
            x="name",
            y="revenue",
            title="Revenue per Product (Last 30 Days)",
            labels={"name": "Product", "revenue": "Revenue (₹)"},
            color="revenue",
            color_continuous_scale="Blues",
        )
        fig_prod.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_prod, use_container_width=True)
