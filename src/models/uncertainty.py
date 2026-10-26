"""Uncertainty quantification for demand and elasticity parameters."""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.utils.logger import logger


class UncertaintyEstimator:
    """Calculates residual variance, empirical prediction intervals, and parametric uncertainty."""

    def __init__(self, residuals: Optional[np.ndarray] = None):
        if residuals is not None and len(residuals) > 0:
            self.residuals = residuals
            self.residual_std = float(np.std(residuals))
        else:
            self.residuals = np.array([0.0])
            self.residual_std = 1.0

    @classmethod
    def fit_from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "UncertaintyEstimator":
        """Calculates prediction residuals from test or validation predictions."""
        residuals = y_true - y_pred
        return cls(residuals=residuals)

    def get_demand_prediction_intervals(
        self, 
        point_prediction: float, 
        confidence_level: float = 0.90
    ) -> Tuple[float, float]:
        """Returns [lower, upper] prediction intervals for a given expected demand point."""
        alpha = 1.0 - confidence_level
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        margin = z_crit * self.residual_std
        lower = max(0.0, point_prediction - margin)
        upper = max(0.0, point_prediction + margin)
        return float(lower), float(upper)

    def sample_demand_shock(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Samples additive demand noise from the residual distribution (or parametric normal)."""
        rng = np.random.default_rng(seed)
        if len(self.residuals) > 10:
            # Empirical bootstrap sampling
            return rng.choice(self.residuals, size=size, replace=True)
        else:
            # Parametric Gaussian sampling
            return rng.normal(0, self.residual_std, size=size)

    @staticmethod
    def sample_elasticity(
        central_elasticity: float, 
        std_error: float = 0.2, 
        size: int = 1000, 
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Draws samples from the parameter uncertainty distribution of price elasticity."""
        rng = np.random.default_rng(seed)
        samples = rng.normal(central_elasticity, std_error, size=size)
        # Demand elasticity is typically strictly negative
        return np.clip(samples, -10.0, -0.05)
