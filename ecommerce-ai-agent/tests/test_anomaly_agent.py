"""
tests/test_anomaly_agent.py
Tests for the AnomalyDetectionAgent.
Requires that models/train_anomaly.py has been run first.
Run: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agents.anomaly_agent import AnomalyDetectionAgent


@pytest.fixture(scope="module")
def agent():
    return AnomalyDetectionAgent()


class TestAnomalyAgent:

    def test_agent_initializes(self, agent):
        """Agent should initialize without raising exceptions."""
        assert agent is not None

    def test_detect_returns_list(self, agent):
        """detect() should always return a list."""
        result = agent.detect(days_back=30)
        assert isinstance(result, list)

    def test_detect_result_structure(self, agent):
        """Each detected anomaly must have the expected fields."""
        results = agent.detect(days_back=60)
        required_keys = {
            "product_id", "date", "units_sold",
            "expected", "direction", "severity", "score"
        }
        for r in results:
            assert required_keys.issubset(r.keys()), f"Missing keys: {r}"

    def test_direction_is_valid(self, agent):
        """Direction must be 'spike' or 'drop'."""
        results = agent.detect(days_back=90)
        for r in results:
            assert r["direction"] in ("spike", "drop"), \
                f"Invalid direction: {r['direction']}"

    def test_severity_is_valid(self, agent):
        """Severity must be one of the defined levels."""
        results = agent.detect(days_back=90)
        valid_severities = {"low", "medium", "high"}
        for r in results:
            assert r["severity"] in valid_severities, \
                f"Invalid severity: {r['severity']}"

    def test_product_ids_are_valid(self, agent):
        """All product IDs in results must be from the known catalog."""
        valid_ids = {"P001", "P002", "P003", "P004", "P005"}
        results = agent.detect(days_back=60)
        for r in results:
            assert r["product_id"] in valid_ids, \
                f"Unknown product ID: {r['product_id']}"

    def test_score_is_negative(self, agent):
        """
        Isolation Forest anomaly scores should be negative
        (more negative = more anomalous).
        """
        results = agent.detect(days_back=60)
        for r in results:
            assert r["score"] < 0, f"Anomaly score should be negative, got {r['score']}"

    def test_no_model_returns_empty(self):
        """If model files are missing, detect() should return an empty list."""
        agent_no_model = AnomalyDetectionAgent.__new__(AnomalyDetectionAgent)
        agent_no_model.model = None
        agent_no_model.scaler = None
        agent_no_model.feature_cols = None
        result = agent_no_model.detect(days_back=30)
        assert result == []
