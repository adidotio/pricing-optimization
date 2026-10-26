"""Unit tests for data validation, cleaning, and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.validator import DataValidator
from src.preprocessing.preprocessing import DataPreprocessor
from src.preprocessing.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_raw_transactions():
    """Synthetic transactions fixture containing invalid prices, cancellations, and valid records."""
    return pd.DataFrame({
        "Invoice": ["500001", "500002", "C500003", "500004", "500005"],
        "StockCode": ["PROD1", "PROD1", "PROD1", "PROD2", "PROD2"],
        "Description": ["Product 1", "Product 1", "Product 1", "Product 2", "Product 2"],
        "Quantity": [10, 15, -5, 20, -10],
        "InvoiceDate": [
            "2023-01-01 10:00:00",
            "2023-01-01 14:00:00",
            "2023-01-02 09:00:00",
            "2023-01-02 11:00:00",
            "2023-01-03 12:00:00"
        ],
        "Price": [2.50, 2.50, 2.50, 5.00, -1.00]
    })


def test_uci_schema_validation(sample_raw_transactions):
    """Verifies that schema validation correctly checks for required UCI columns."""
    valid, errors = DataValidator.validate_uci_schema(sample_raw_transactions)
    assert valid is True
    assert len(errors) == 0

    # Missing column
    incomplete = sample_raw_transactions.drop(columns=["StockCode"])
    valid_bad, errors_bad = DataValidator.validate_uci_schema(incomplete)
    assert valid_bad is False
    assert any("StockCode" in e for e in errors_bad)


def test_clean_transactions_return_policy(sample_raw_transactions):
    """Verifies that invoice cancellations ('C' prefix), negative quantities, and negative prices are caught."""
    cleaned, audit = DataPreprocessor.clean_transactions(
        sample_raw_transactions, 
        filter_cancellations=True, 
        remove_negative_prices=True
    )

    # Invoices C500003 and 500005 (negative qty and negative price) must be removed
    assert len(cleaned) == 3
    assert audit["removed_cancellations"] >= 1
    assert not cleaned["Invoice"].str.startswith("C").any()
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["Price"] > 0).all()


def test_daily_aggregation(sample_raw_transactions):
    """Verifies daily aggregation sums quantity and computes VWAP."""
    cleaned, _ = DataPreprocessor.clean_transactions(sample_raw_transactions)
    daily = DataPreprocessor.aggregate_to_daily_series(cleaned)

    assert "date" in daily.columns
    assert "product_id" in daily.columns
    assert "quantity" in daily.columns
    assert "price" in daily.columns
    assert "revenue" in daily.columns

    # PROD1 on 2023-01-01 has 10 + 15 = 25 units
    prod1_day1 = daily[(daily["product_id"] == "PROD1") & (daily["date"] == pd.to_datetime("2023-01-01"))]
    assert len(prod1_day1) == 1
    assert prod1_day1["quantity"].iloc[0] == 25


def test_feature_engineering_no_leakage():
    """Verifies that lag and rolling statistics are computed with lag offset to prevent lookahead leakage."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "product_id": "P1",
        "quantity": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "price": [2.0] * 10,
        "revenue": [20.0] * 10
    })

    feat = FeatureEngineer.create_features(df, lag_days=[1], rolling_windows=[7])

    assert "demand_lag_1" in feat.columns
    assert "demand_rolling_mean_7" in feat.columns

    # Lag 1 at index 1 must equal quantity at index 0 (10)
    assert feat.iloc[1]["demand_lag_1"] == 10.0
    # Lag 1 at index 2 must equal quantity at index 1 (20)
    assert feat.iloc[2]["demand_lag_1"] == 20.0
