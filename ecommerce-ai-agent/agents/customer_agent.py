"""
agents/customer_agent.py

Customer support agent using HuggingFace Inference API — 100% FREE.
No OpenAI key needed. Uses the Zephyr-7B model via HuggingFace's free API.

How it works:
1. Customer sends a query
2. Intent is detected (order_status / return_refund / shipping_info / general)
3. For order queries: directly calls the order_lookup tool
4. For FAQ queries: directly calls the faq_lookup tool
5. For complex queries: uses HuggingFace LLM via LangChain
6. Response is logged to the database

The agent has a smart fallback: if the HuggingFace API is down or slow,
it falls back to a fast rule-based system that needs zero internet.
"""
import os
import sys
import re
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# ── HuggingFace setup ──────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

# Friendly display names for supported models
MODEL_OPTIONS = {
    "HuggingFaceH4/zephyr-7b-beta": "Zephyr-7B (recommended)",
    "google/flan-t5-large": "Flan-T5 Large (faster, lighter)",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral-7B",
    "tiiuae/falcon-7b-instruct": "Falcon-7B",
}


def get_hf_llm():
    """
    Creates a HuggingFace LLM instance using the free Inference API.
    Returns None if the token is missing or invalid — triggers fallback mode.
    """
    if not HF_TOKEN or HF_TOKEN == "your_huggingface_token_here":
        logger.warning("No HF_TOKEN found. Will use rule-based fallback.")
        return None

    try:
        from langchain_community.llms import HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=HF_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,       # Lower = more focused, less creative
            max_new_tokens=300,    # Keep responses concise to save quota
            timeout=30,            # Wait max 30 seconds before giving up
        )
        logger.info(f"HuggingFace LLM loaded: {HF_MODEL}")
        return llm

    except Exception as e:
        logger.warning(f"Could not load HuggingFace LLM: {e}. Using fallback.")
        return None


class CustomerSupportAgent:
    """
    Customer support agent with two operating modes:

    1. LLM Mode  — Uses HuggingFace Zephyr-7B for natural language understanding.
                   Requires HF_TOKEN in .env. Needs internet.
    2. Fallback  — Rule-based intent matching + direct tool calls. Works 100% offline.

    Both modes use the same tools (order_lookup, faq_lookup) and log to the same DB.
    """

    # Keywords that map to each intent category (used in both modes)
    INTENT_KEYWORDS = {
        "order_status": [
            "order", "ord-", "where is", "delivery", "shipped", "dispatch",
            "track", "tracking", "when will", "status"
        ],
        "return_refund": [
            "return", "refund", "money back", "exchange", "replace", "damaged",
            "broken", "wrong item", "defective"
        ],
        "cancellation": [
            "cancel", "cancellation", "stop order", "dont want"
        ],
        "shipping_info": [
            "shipping", "delivery time", "how long", "days", "express",
            "free shipping", "charges"
        ],
        "payment": [
            "payment", "pay", "upi", "gpay", "card", "cod", "cash",
            "transaction", "failed payment"
        ],
        "warranty": [
            "warranty", "guarantee", "repair", "service center"
        ],
    }

    def __init__(self):
        # Try to load the LLM — falls back to None if unavailable
        self.llm = get_hf_llm()
        self.use_llm = self.llm is not None

        # Load tools once so they can be reused across queries
        from tools.order_lookup import order_lookup
        from tools.faq_tool import faq_lookup
        self.order_tool = order_lookup
        self.faq_tool = faq_lookup

        mode = "LLM mode (HuggingFace)" if self.use_llm else "Fallback mode (rule-based)"
        logger.info(f"Customer agent initialized: {mode}")

    def detect_intent(self, query: str) -> str:
        """
        Classify the customer query into one of our intent categories.
        Higher-specificity intents are checked first to avoid false matches
        (e.g. 'cancel my order' should be cancellation, not order_status).
        """
        q = query.lower()
        # Check high-specificity intents first to avoid collisions
        PRIORITY = [
            "cancellation", "return_refund", "warranty",
            "payment", "shipping_info", "order_status",
        ]
        for intent in PRIORITY:
            keywords = self.INTENT_KEYWORDS.get(intent, [])
            if any(kw in q for kw in keywords):
                return intent
        return "general"

    def extract_order_id(self, query: str) -> str | None:
        """
        Extract an order ID from the query text using regex.
        Handles formats like: ORD-10005, ord10005, ORD 10005
        """
        match = re.search(r"ORD[-\s]?\d{4,6}", query.upper())
        if match:
            # Normalize to ORD-XXXXX format
            raw = match.group(0).replace(" ", "-")
            if "-" not in raw:
                raw = raw[:3] + "-" + raw[3:]
            return raw
        return None

    def _rule_based_response(self, query: str, intent: str) -> str:
        """
        Fast, offline fallback response.
        Uses direct tool calls rather than the LLM for speed and reliability.
        """
        # If an order ID is present, always do a direct DB lookup
        order_id = self.extract_order_id(query)
        if order_id:
            return self.order_tool.invoke(order_id)

        # Order status intent without a visible order ID
        if intent == "order_status":
            return (
                "I'd be happy to check your order status! "
                "Please share your Order ID (format: ORD-XXXXX). "
                "You can find it in your order confirmation email."
            )

        # All other intents: route through the FAQ tool
        return self.faq_tool.invoke(query)

    def _llm_response(self, query: str, intent: str) -> str:
        """
        Generate a polished response using the HuggingFace LLM.
        Strategy: fetch tool data first, then ask LLM to present it nicely.
        Falls back to rule-based if the LLM call fails.
        """
        # Step 1: Collect relevant data from tools
        tool_output = ""
        order_id = self.extract_order_id(query)

        if order_id:
            tool_output = self.order_tool.invoke(order_id)
        elif intent in ["return_refund", "shipping_info", "payment",
                        "cancellation", "warranty", "general"]:
            tool_output = self.faq_tool.invoke(query)

        # Step 2: Build the Zephyr-style chat prompt
        system_msg = (
            "You are a helpful, friendly customer support agent for ShopAI, "
            "an Indian e-commerce platform. Be concise, polite, and helpful. "
            "Answer in 2-3 sentences maximum. Do not repeat the system info verbatim."
        )

        if tool_output:
            prompt = (
                f"<|system|>{system_msg}</s>"
                f"<|user|>Customer question: {query}\n\n"
                f"Data from our system:\n{tool_output}\n\n"
                f"Please write a friendly response for the customer.</s>"
                f"<|assistant|>"
            )
        else:
            prompt = (
                f"<|system|>{system_msg}</s>"
                f"<|user|>{query}</s>"
                f"<|assistant|>"
            )

        try:
            response = self.llm.invoke(prompt)
            response = response.strip()
            # Strip any leaked chat tokens from the model output
            if "<|" in response:
                response = response.split("<|")[0].strip()
            # If the model returned nothing useful, fall back to tool output
            return response if response else (tool_output or self._rule_based_response(query, intent))
        except Exception as e:
            logger.error(f"LLM call failed: {e}. Using fallback.")
            return tool_output or self._rule_based_response(query, intent)

    def run(self, query: str) -> dict:
        """
        Main entry point. Process a customer query and return a structured result.
        Always logs the interaction to the CustomerQuery table.
        """
        intent = self.detect_intent(query)
        logger.info(f"Query: '{query[:60]}' | Intent: {intent} | LLM: {self.use_llm}")

        try:
            response = (
                self._llm_response(query, intent)
                if self.use_llm
                else self._rule_based_response(query, intent)
            )

            # Escalate if the response suggests the agent couldn't handle it
            escalated = any(phrase in response.lower() for phrase in [
                "cannot help", "unable to", "please contact support",
                "human agent", "escalate", "call us"
            ])

            # Persist the interaction to the database
            from database.db import SessionLocal
            from database.models import CustomerQuery
            db = SessionLocal()
            try:
                db.add(CustomerQuery(
                    query=query,
                    intent=intent,
                    response=response,
                    confidence=0.70 if self.use_llm else 0.85,
                    escalated=escalated
                ))
                db.commit()
            finally:
                db.close()

            return {
                "query": query,
                "intent": intent,
                "response": response,
                "escalated": escalated,
                "mode": "llm" if self.use_llm else "rule-based",
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "query": query,
                "intent": intent,
                "response": "I'm experiencing technical difficulties. Please contact support@shopai.in",
                "escalated": True,
                "mode": "error",
            }


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = CustomerSupportAgent()
    mode_label = "LLM (HuggingFace)" if agent.use_llm else "Rule-based (offline)"
    print(f"Agent mode: {mode_label}\n{'─'*50}")

    test_queries = [
        "Where is my order ORD-10005?",
        "What is your return policy?",
        "How long does standard delivery take?",
        "Can I pay with UPI?",
        "My product arrived damaged, what do I do?",
    ]

    for q in test_queries:
        result = agent.run(q)
        print(f"\nQ: {q}")
        print(f"A: {result['response']}")
        print(f"   Intent: {result['intent']} | Mode: {result['mode']}")
