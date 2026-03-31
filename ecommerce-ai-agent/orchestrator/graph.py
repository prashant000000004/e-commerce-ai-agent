"""
orchestrator/graph.py
LangGraph state machine — the central brain of the multi-agent system.

Architecture:
  [router_node]
       │
       ├─ customer_query  → [customer_node]  → END
       ├─ inventory_check → [inventory_node] → END
       └─ anomaly_scan    → [anomaly_node]   → END

The router reads the `event_type` field in the shared state and dispatches
to the correct specialist agent. All nodes receive and return the full state.
"""
import sys
import os
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AgentState(TypedDict):
    """
    Shared state object that travels through every node in the graph.
    All fields are Optional so nodes only need to populate what they produce.
    """
    event_type: str                        # What triggered this run
    user_query: Optional[str]             # Raw text from the customer
    agent_response: Optional[str]         # Final text response
    intent: Optional[str]                 # Detected intent (customer queries)
    escalated: bool                       # True if needs human review
    inventory_results: Optional[list]     # Output from inventory agent
    anomaly_results: Optional[list]       # Output from anomaly agent
    error: Optional[str]                  # Error message if something went wrong


# ── Lazy agent loader ──────────────────────────────────────────────────────────
# Agents are expensive to initialize (load ML models), so we cache them.
# Lazy loading also prevents circular import issues.
_agents: dict = {}


def _get_agent(name: str):
    """Return a cached agent instance, creating it on first call."""
    if name not in _agents:
        if name == "customer":
            from agents.customer_agent import CustomerSupportAgent
            _agents[name] = CustomerSupportAgent()
        elif name == "inventory":
            from agents.inventory_agent import InventoryForecastAgent
            _agents[name] = InventoryForecastAgent()
        elif name == "anomaly":
            from agents.anomaly_agent import AnomalyDetectionAgent
            _agents[name] = AnomalyDetectionAgent()
    return _agents[name]


# ── Graph Node Functions ───────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """
    Entry point node. Just logs the incoming event and passes state through.
    The actual routing happens in the conditional edge function below.
    """
    logger.info(f"[Router] Received event_type='{state['event_type']}'")
    return state


def customer_node(state: AgentState) -> AgentState:
    """Process a customer support query and return the agent's response."""
    query = state.get("user_query", "")
    if not query:
        logger.warning("[Customer Node] No query provided.")
        return {**state, "agent_response": "No query was provided.", "escalated": True}

    try:
        result = _get_agent("customer").run(query)
        return {
            **state,
            "agent_response": result["response"],
            "intent": result["intent"],
            "escalated": result["escalated"],
            "error": None,
        }
    except Exception as e:
        logger.error(f"[Customer Node] Error: {e}")
        return {
            **state,
            "error": str(e),
            "escalated": True,
            "agent_response": "Technical error occurred. Please contact support@shopai.in",
        }


def inventory_node(state: AgentState) -> AgentState:
    """Run inventory forecasting across all products and return a summary."""
    try:
        results = _get_agent("inventory").check_all_products()
        at_risk = sum(1 for r in results if r["will_stockout"])
        summary = f"Checked {len(results)} products. {at_risk} at stockout risk."
        logger.info(f"[Inventory Node] {summary}")
        return {**state, "inventory_results": results, "agent_response": summary, "error": None}
    except Exception as e:
        logger.error(f"[Inventory Node] Error: {e}")
        return {**state, "error": str(e), "agent_response": f"Inventory check failed: {e}"}


def anomaly_node(state: AgentState) -> AgentState:
    """Scan recent sales for anomalies and return a summary."""
    try:
        anomalies = _get_agent("anomaly").detect(days_back=30)
        summary = f"Anomaly scan complete. Found {len(anomalies)} anomaly events."
        logger.info(f"[Anomaly Node] {summary}")
        return {**state, "anomaly_results": anomalies, "agent_response": summary, "error": None}
    except Exception as e:
        logger.error(f"[Anomaly Node] Error: {e}")
        return {**state, "error": str(e), "agent_response": f"Anomaly scan failed: {e}"}


# ── Routing Logic ──────────────────────────────────────────────────────────────

def decide_route(state: AgentState) -> Literal["customer", "inventory", "anomaly", "end"]:
    """
    Conditional edge: inspect event_type and choose which node to call next.
    Unknown event types route to END safely.
    """
    route_map = {
        "customer_query": "customer",
        "inventory_check": "inventory",
        "anomaly_scan": "anomaly",
    }
    destination = route_map.get(state.get("event_type", ""), "end")
    logger.info(f"[Router] Routing to: {destination}")
    return destination


# ── Graph Assembly ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph state machine.
    Called fresh each time to avoid stale compiled graphs.
    """
    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("router", router_node)
    g.add_node("customer", customer_node)
    g.add_node("inventory", inventory_node)
    g.add_node("anomaly", anomaly_node)

    # Set the entry point
    g.set_entry_point("router")

    # The router dispatches to one of three agents (or END)
    g.add_conditional_edges(
        "router",
        decide_route,
        {
            "customer": "customer",
            "inventory": "inventory",
            "anomaly": "anomaly",
            "end": END,
        },
    )

    # All agent nodes terminate after completing their task
    g.add_edge("customer", END)
    g.add_edge("inventory", END)
    g.add_edge("anomaly", END)

    return g.compile()


def run_agent(event_type: str, user_query: str = None) -> AgentState:
    """
    Public API for the entire multi-agent system.
    Called by the Streamlit dashboard and the test suite.

    Args:
        event_type:  "customer_query" | "inventory_check" | "anomaly_scan"
        user_query:  The customer's message (only required for customer_query)

    Returns:
        Final AgentState dict with all populated fields.
    """
    initial_state: AgentState = {
        "event_type": event_type,
        "user_query": user_query,
        "agent_response": None,
        "intent": None,
        "escalated": False,
        "inventory_results": None,
        "anomaly_results": None,
        "error": None,
    }
    return build_graph().invoke(initial_state)


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Customer Query ──────────────────────────────────────")
    r = run_agent("customer_query", "What is your return policy?")
    print(f"Response : {r['agent_response']}")
    print(f"Intent   : {r['intent']}")

    print("\n── Inventory Check ─────────────────────────────────────")
    r = run_agent("inventory_check")
    print(f"Response : {r['agent_response']}")

    print("\n── Anomaly Scan ────────────────────────────────────────")
    r = run_agent("anomaly_scan")
    print(f"Response : {r['agent_response']}")
