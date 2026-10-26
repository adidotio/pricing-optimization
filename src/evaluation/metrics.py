"""Evaluation metrics for demand prediction and pricing decisions."""
from typing import Dict, Any, Union
import numpy as np
import pandas as pd


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-4) -> float:
    """Calculates MAPE with zero and near-zero actual demand protection."""
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-4) -> float:
    """Calculates symmetric MAPE (sMAPE) bounded between 0% and 200%."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def r_squared_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates coefficient of determination R^2."""
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard demand forecast diagnostic metrics."""
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 3),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 3),
        "mape_pct": round(mean_absolute_percentage_error(y_true, y_pred), 2),
        "smape_pct": round(symmetric_mape(y_true, y_pred), 2),
        "r2": round(r_squared_score(y_true, y_pred), 3)
    }
