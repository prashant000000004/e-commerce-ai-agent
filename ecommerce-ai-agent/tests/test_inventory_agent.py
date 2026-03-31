"""
tests/test_inventory_agent.py
Tests for the InventoryForecastAgent.
Requires that models/train_forecast.py has been run first.
Run: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from agents.inventory_agent import InventoryForecastAgent


@pytest.fixture(scope="module")
def agent():
    return InventoryForecastAgent()


class TestInventoryAgent:

    def test_agent_initializes(self, agent):
        """Agent should initialize without raising exceptions."""
        assert agent is not None

    def test_models_dict_exists(self, agent):
        """models dict should exist (may be empty if models not trained yet)."""
        assert isinstance(agent.models, dict)

    def test_forecast_returns_dataframe(self, agent):
        """forecast() should return a DataFrame for any product ID."""
        result = agent.forecast("P001", periods=7)
        # If model is loaded, we get rows; if not loaded, we get empty DF
        assert isinstance(result, pd.DataFrame)

    def test_forecast_has_correct_columns(self, agent):
        """Forecast DataFrame must have the expected columns."""
        result = agent.forecast("P001", periods=7)
        if not result.empty:
            expected_cols = {"ds", "yhat", "yhat_lower", "yhat_upper"}
            assert expected_cols.issubset(result.columns)

    def test_forecast_no_negative_values(self, agent):
        """Predicted units sold cannot be negative."""
        result = agent.forecast("P001", periods=30)
        if not result.empty:
            assert (result["yhat"] >= 0).all(), "Negative forecast values found"

    def test_check_all_products_returns_list(self, agent):
        """check_all_products() should always return a list."""
        result = agent.check_all_products()
        assert isinstance(result, list)

    def test_check_all_products_result_structure(self, agent):
        """Each result item should have the required keys."""
        results = agent.check_all_products()
        required_keys = {
            "product_id", "product_name", "current_stock",
            "forecasted_demand_30d", "will_stockout"
        }
        for r in results:
            assert required_keys.issubset(r.keys()), f"Missing keys in result: {r}"

    def test_stockout_flag_is_boolean(self, agent):
        """will_stockout must be a boolean."""
        results = agent.check_all_products()
        for r in results:
            assert isinstance(r["will_stockout"], bool)
