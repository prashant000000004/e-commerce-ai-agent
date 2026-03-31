"""
tests/test_customer_agent.py
Unit tests for the CustomerSupportAgent.
These tests use the rule-based fallback (no HuggingFace token needed).
Run: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agents.customer_agent import CustomerSupportAgent


@pytest.fixture(scope="module")
def agent():
    """Create one agent instance shared across all tests in this module."""
    return CustomerSupportAgent()


class TestIntentDetection:
    """Test that the intent classifier correctly categorises queries."""

    def test_order_status_intent(self, agent):
        assert agent.detect_intent("Where is my order ORD-10005?") == "order_status"

    def test_return_refund_intent(self, agent):
        assert agent.detect_intent("How do I return a damaged product?") == "return_refund"

    def test_cancellation_intent(self, agent):
        assert agent.detect_intent("I want to cancel my order") == "cancellation"

    def test_shipping_intent(self, agent):
        assert agent.detect_intent("How long does shipping take?") == "shipping_info"

    def test_payment_intent(self, agent):
        assert agent.detect_intent("Can I pay with UPI?") == "payment"

    def test_warranty_intent(self, agent):
        assert agent.detect_intent("Is there a warranty on electronics?") == "warranty"

    def test_general_intent_fallback(self, agent):
        assert agent.detect_intent("Hello there!") == "general"


class TestOrderIdExtraction:
    """Test the regex-based order ID extractor."""

    def test_standard_format(self, agent):
        assert agent.extract_order_id("My order ORD-10005 is late") == "ORD-10005"

    def test_no_dash_format(self, agent):
        result = agent.extract_order_id("Order ORD10005 status?")
        assert result is not None
        assert "10005" in result

    def test_lowercase(self, agent):
        result = agent.extract_order_id("where is ord-10005")
        assert result is not None

    def test_no_order_id(self, agent):
        assert agent.extract_order_id("What is your return policy?") is None


class TestAgentRun:
    """Integration tests for the full agent.run() method."""

    def test_run_returns_dict(self, agent):
        result = agent.run("What is your return policy?")
        assert isinstance(result, dict)

    def test_run_has_required_keys(self, agent):
        result = agent.run("How do I track my order?")
        for key in ["query", "intent", "response", "escalated", "mode"]:
            assert key in result, f"Missing key: {key}"

    def test_response_is_not_empty(self, agent):
        result = agent.run("What payment methods do you accept?")
        assert result["response"] and len(result["response"]) > 10

    def test_faq_response_content(self, agent):
        result = agent.run("What is your refund policy?")
        # Should mention refund-related content
        assert any(word in result["response"].lower() for word in ["refund", "business", "days"])

    def test_mode_is_valid(self, agent):
        result = agent.run("Test query")
        assert result["mode"] in ["llm", "rule-based", "error"]
