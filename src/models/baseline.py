"""Baseline demand prediction models for benchmarking."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class DemandBaselineModel:
    """Base interface for demand baselines."""
    def fit(self, y: pd.Series) -> "DemandBaselineModel":
        raise NotImplementedError

    def predict(self, n_periods: int = 1) -> np.ndarray:
        raise NotImplementedError


class HistoricalMeanDemandBaseline(DemandBaselineModel):
    """Predicts historical mean demand."""
    def __init__(self):
        self.mean_demand: float = 0.0

    def fit(self, y: pd.Series) -> "HistoricalMeanDemandBaseline":
        self.mean_demand = float(y.mean()) if len(y) > 0 else 0.0
        return self

    def predict(self, n_periods: int = 1) -> np.ndarray:
        return np.full(n_periods, self.mean_demand)


class NaiveDemandBaseline(DemandBaselineModel):
    """Predicts the most recently observed demand (random walk baseline)."""
    def __init__(self):
        self.last_demand: float = 0.0

    def fit(self, y: pd.Series) -> "NaiveDemandBaseline":
        self.last_demand = float(y.iloc[-1]) if len(y) > 0 else 0.0
        return self

    def predict(self, n_periods: int = 1) -> np.ndarray:
        return np.full(n_periods, self.last_demand)


class SeasonalNaiveBaseline(DemandBaselineModel):
    """Predicts demand based on weekly seasonality (7 periods ago)."""
    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.history: np.ndarray = np.array([])

    def fit(self, y: pd.Series) -> "SeasonalNaiveBaseline":
        self.history = y.to_numpy()
        return self

    def predict(self, n_periods: int = 1) -> np.ndarray:
        if len(self.history) == 0:
            return np.zeros(n_periods)
        preds = []
        for i in range(n_periods):
            idx = len(self.history) - self.season_length + (i % self.season_length)
            val = self.history[idx] if idx >= 0 and idx < len(self.history) else self.history.mean()
            preds.append(val)
        return np.array(preds)
