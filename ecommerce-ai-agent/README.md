# 🛒 Autonomous AI Agent for E-Commerce Operations
### 100% FREE — No OpenAI Key Required

A production-ready multi-agent AI system for e-commerce operations, powered by **free HuggingFace models**.

---

## 🏗️ Architecture

```
Customer Query ──→ [LangGraph Orchestrator]
                          │
             ┌────────────┼────────────┐
             ↓            ↓            ↓
    [Customer Agent] [Inventory]  [Anomaly]
    HuggingFace LLM   Prophet     Isolation
    + Rule-based      Forecast    Forest
    fallback          (Prophet)   (sklearn)
             │            │            │
             └────────────┴────────────┘
                          │
              [SQLite DB + Streamlit UI]
                          │
                    [MLflow Tracking]
```

**Agents:**
- **Customer Agent** — Answers support queries using Zephyr-7B (free HuggingFace API) with a rule-based offline fallback
- **Inventory Agent** — Forecasts 30-day demand using Facebook Prophet; fires stockout alerts
- **Anomaly Agent** — Detects unusual sales events using Isolation Forest

---

## 🚀 Quick Start (5 Steps)

### Step 1: Get a Free HuggingFace Token
1. Go to [huggingface.co](https://huggingface.co) → create a free account
2. Settings → Access Tokens → **New Token** → select **Read** permission
3. Copy the token (starts with `hf_`)

> **No token? No problem.** The project works without it — the customer agent automatically switches to rule-based mode.

### Step 2: Install & Configure
```bash
# Clone or extract the project
cd ecommerce-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# If Prophet fails to install:
# pip install prophet --no-build-isolation

# Set up environment variables
cp .env.example .env
# Open .env and set your token:  HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

### Step 3: Generate Sample Data
```bash
python scripts/generate_sample_data.py
```
Creates: 5 products, 200 orders, 1 year of daily sales (1,825 records)

### Step 4: Train the ML Models
```bash
python models/train_forecast.py    # ~2 minutes — trains 5 Prophet models
python models/train_anomaly.py     # ~30 seconds — trains Isolation Forest
```
Track all experiments at: `mlflow ui` → http://localhost:5000

### Step 5: Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Open: **http://localhost:8501**

---

## 🐳 Docker (Optional)

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN

docker-compose up --build
```
Dashboard available at http://localhost:8501

---

## 🧪 Test Individual Agents
```bash
python agents/customer_agent.py    # Test customer support
python agents/inventory_agent.py   # Test inventory forecast
python agents/anomaly_agent.py     # Test anomaly detection
python orchestrator/graph.py       # Test full orchestration
```

---

## 📁 Project Structure

```
ecommerce-ai-agent/
├── agents/
│   ├── customer_agent.py      # HuggingFace LLM + rule-based fallback
│   ├── inventory_agent.py     # Prophet-based demand forecasting
│   └── anomaly_agent.py       # Isolation Forest anomaly detection
├── orchestrator/
│   └── graph.py               # LangGraph state machine
├── models/
│   ├── train_forecast.py      # Train Prophet models
│   └── train_anomaly.py       # Train Isolation Forest
├── tools/
│   ├── order_lookup.py        # LangChain tool: DB order lookup
│   ├── faq_tool.py            # LangChain tool: FAQ matching
│   └── restock_tool.py        # LangChain tool: restock alerts
├── database/
│   ├── db.py                  # SQLAlchemy engine + session
│   └── models.py              # ORM models (Order, Product, etc.)
├── dashboard/
│   └── app.py                 # Streamlit UI (4 tabs)
├── scripts/
│   └── generate_sample_data.py
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🔑 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace API token (free) | — (uses rule-based fallback) |
| `HF_MODEL` | Model to use | `HuggingFaceH4/zephyr-7b-beta` |
| `DATABASE_URL` | SQLite path | `sqlite:///./ecommerce.db` |
| `MLFLOW_TRACKING_URI` | MLflow runs folder | `./mlruns` |
| `MODEL_DIR` | Saved model files | `./models/saved` |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `langchain_community` import error | `pip install langchain-community --upgrade` |
| `langchain_huggingface` not found | `pip install langchain-huggingface` |
| Prophet install fails | `pip install prophet --no-build-isolation` |
| LangGraph version conflict | `pip install langgraph==0.1.0 --force-reinstall` |
| HuggingFace API slow | Their servers are busy — wait 30s and retry |
| No data in dashboard | Run `python scripts/generate_sample_data.py` |
| Models not found | Run `train_forecast.py` and `train_anomaly.py` |

---

## 💡 How the AI Works Without an API Key

The customer agent has two modes:

1. **LLM Mode** (with `HF_TOKEN`): Uses Zephyr-7B on HuggingFace's free inference servers for natural language understanding
2. **Rule-based Mode** (no token): Uses keyword intent matching + direct database lookups + pre-written FAQ answers

Both modes use the same tools and database — the only difference is response quality for open-ended questions.
