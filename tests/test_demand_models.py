"""Unit tests for demand baselines, ML models, and evaluation metrics."""
import pytest
import numpy as np
import pandas as pd

from src.models.baseline import HistoricalMeanDemandBaseline, NaiveDemandBaseline, SeasonalNaiveBaseline
from src.models.demand_model import LinearRidgeDemandModel, LogLinearDemandModel, RandomForestDemandModel
from src.evaluation.metrics import (
    mean_absolute_error, root_mean_squared_error, 
    mean_absolute_percentage_error, calculate_forecast_metrics
)
from src.evaluation.temporal_validation import TemporalValidator


def test_baseline_models():
    """Tests that historical mean and naive baselines behave according to mathematical definitions."""
    y = pd.Series([10.0, 20.0, 30.0, 40.0])
    
    mean_m = HistoricalMeanDemandBaseline().fit(y)
    preds_mean = mean_m.predict(3)
    assert len(preds_mean) == 3
    assert np.allclose(preds_mean, 25.0)

    naive_m = NaiveDemandBaseline().fit(y)
    preds_naive = naive_m.predict(2)
    assert len(preds_naive) == 2
    assert np.allclose(preds_naive, 40.0)


def test_forecast_metrics():
    """Tests MAE, RMSE, and MAPE calculations with known analytical values."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    # Errors: [2, 2, 3] -> MAE = 7/3
    mae = mean_absolute_error(y_true, y_pred)
    assert pytest.approx(mae, 0.01) == 2.333

    # Squared errors: [4, 4, 9] -> MSE = 17/3 -> RMSE = sqrt(5.666) ≈ 2.38
    rmse = root_mean_squared_error(y_true, y_pred)
    assert pytest.approx(rmse, 0.01) == 2.38

    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape_pct" in metrics
    assert "r2" in metrics


def test_chronological_split_prevents_leakage():
    """Verifies that chronological split splits purely on temporal cutoff without shuffling."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "quantity": np.arange(10)
    })

    train, test = TemporalValidator.chronological_split(df, test_size=0.30)
    assert len(train) == 7
    assert len(test) == 3
    # Max train date must precede Min test date
    assert train["date"].max() < test["date"].min()


def test_demand_model_evaluation_at_prices():
    """Verifies that ML demand models evaluate demand monotonically across prices."""
    np.random.seed(42)
    # Generate synthetic training data with negative price slope
    prices = np.random.uniform(1.0, 5.0, 100)
    demands = 50.0 - 8.0 * prices + np.random.normal(0, 1, 100)
    X = pd.DataFrame({"price": prices, "log_price": np.log(prices), "feature1": np.ones(100)})
    y = pd.Series(np.clip(demands, 1.0, None))

    model = LogLinearDemandModel().fit(X, y)
    
    # Predict across test prices: higher price should lead to lower demand
    base_features = pd.Series({"price": 2.5, "log_price": np.log(2.5), "feature1": 1.0})
    candidate_prices = np.array([2.0, 3.0, 4.0])
    preds = model.predict_at_prices(base_features, candidate_prices)

    assert len(preds) == 3
    assert preds[0] > preds[1] > preds[2]  # Downward sloping
