"""Unit tests for econometric price elasticity estimation and diagnostics."""
import pytest
import numpy as np
import pandas as pd

from src.models.elasticity import PriceElasticityEstimator


def test_elasticity_recovery_analytical():
    """Verifies that the log-log estimator recovers known elasticity parameter beta_1 = -1.5."""
    np.random.seed(42)
    n = 200
    prices = np.random.uniform(5.0, 15.0, n)
    true_elasticity = -1.50
    # ln(Q) = 4.0 - 1.5 * ln(P) + small noise
    log_q = 4.0 + true_elasticity * np.log(prices) + np.random.normal(0, 0.05, n)
    quantities = np.exp(log_q)

    df = pd.DataFrame({
        "product_id": "P1",
        "price": prices,
        "quantity": quantities
    })

    result = PriceElasticityEstimator.estimate_elasticity(df, "P1")
    assert result["is_reliable"] is True
    assert result["is_assumed"] is False
    assert pytest.approx(result["elasticity"], abs=0.1) == true_elasticity
    assert result["is_economically_plausible"] is True
    assert result["p_value"] < 0.001
    assert result["ci_lower"] < result["elasticity"] < result["ci_upper"]


def test_insufficient_variation_detection():
    """Verifies that products with near-zero price variation are flagged as unreliable."""
    n = 50
    # Prices vary by less than 1% (CV < 0.04)
    prices = np.full(n, 10.0) + np.random.uniform(0.0, 0.05, n)
    quantities = np.random.uniform(20.0, 40.0, n)

    df = pd.DataFrame({
        "product_id": "P_STATIC",
        "price": prices,
        "quantity": quantities
    })

    result = PriceElasticityEstimator.estimate_elasticity(df, "P_STATIC")
    assert result["is_reliable"] is False
    assert "Insufficient historical price variation" in result["warning"]
    assert result["elasticity"] is None


def test_elasticity_assumption_fallback():
    """Verifies that manual elasticity assumptions are properly labeled and structured."""
    fallback = PriceElasticityEstimator.get_assumed_elasticity_fallback("PROD_NEW", assumed_value=-1.8)
    assert fallback["is_assumed"] is True
    assert fallback["elasticity"] == -1.8
    assert "ASSUMPTION-BASED" in fallback["warning"]
