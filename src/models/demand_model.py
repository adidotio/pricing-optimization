"""Supervised machine learning models for product-level demand forecasting."""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from src.utils.logger import logger


class BaseDemandModel:
    """Abstract interface for demand forecasting models."""
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseDemandModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_at_prices(self, base_features: pd.Series, candidate_prices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearRidgeDemandModel(BaseDemandModel):
    """Linear regression with L2 regularization."""
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRidgeDemandModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_names])
        return np.clip(preds, 0.0, None)

    def predict_at_prices(self, base_features: pd.Series, candidate_prices: np.ndarray) -> np.ndarray:
        X_eval = pd.DataFrame([base_features.to_dict()] * len(candidate_prices))
        X_eval["price"] = candidate_prices
        if "log_price" in self.feature_names:
            X_eval["log_price"] = np.log(np.clip(candidate_prices, 0.01, None))
        if "price_ratio_comp" in self.feature_names and "competitor_price" in X_eval.columns:
            X_eval["price_ratio_comp"] = candidate_prices / np.clip(X_eval["competitor_price"], 0.01, None)
        return self.predict(X_eval)


class LogLinearDemandModel(BaseDemandModel):
    """Log-linear specification: ln(Q) = beta_0 + beta_1 * ln(P) + gamma * X."""
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogLinearDemandModel":
        self.feature_names = list(X.columns)
        y_log = np.log(np.clip(y, 0.01, None))
        self.model.fit(X, y_log)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_preds = self.model.predict(X[self.feature_names])
        preds = np.exp(log_preds)
        return np.clip(preds, 0.0, None)

    def predict_at_prices(self, base_features: pd.Series, candidate_prices: np.ndarray) -> np.ndarray:
        X_eval = pd.DataFrame([base_features.to_dict()] * len(candidate_prices))
        X_eval["price"] = candidate_prices
        if "log_price" in self.feature_names:
            X_eval["log_price"] = np.log(np.clip(candidate_prices, 0.01, None))
        if "price_ratio_comp" in self.feature_names and "competitor_price" in X_eval.columns:
            X_eval["price_ratio_comp"] = candidate_prices / np.clip(X_eval["competitor_price"], 0.01, None)
        return self.predict(X_eval)


class RandomForestDemandModel(BaseDemandModel):
    """Random Forest regressor for nonlinear demand patterns."""
    def __init__(self, n_estimators: int = 50, max_depth: int = 8, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestDemandModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_names])
        return np.clip(preds, 0.0, None)

    def predict_at_prices(self, base_features: pd.Series, candidate_prices: np.ndarray) -> np.ndarray:
        X_eval = pd.DataFrame([base_features.to_dict()] * len(candidate_prices))
        X_eval["price"] = candidate_prices
        if "log_price" in self.feature_names:
            X_eval["log_price"] = np.log(np.clip(candidate_prices, 0.01, None))
        if "price_ratio_comp" in self.feature_names and "competitor_price" in X_eval.columns:
            X_eval["price_ratio_comp"] = candidate_prices / np.clip(X_eval["competitor_price"], 0.01, None)
        return self.predict(X_eval)


class GradientBoostingDemandModel(BaseDemandModel):
    """Histogram-based Gradient Boosting for fast and robust tree-based forecasting."""
    def __init__(self, max_iter: int = 60, max_depth: int = 6, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, 
            max_depth=max_depth, 
            random_state=random_state
        )
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingDemandModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_names])
        return np.clip(preds, 0.0, None)

    def predict_at_prices(self, base_features: pd.Series, candidate_prices: np.ndarray) -> np.ndarray:
        X_eval = pd.DataFrame([base_features.to_dict()] * len(candidate_prices))
        X_eval["price"] = candidate_prices
        if "log_price" in self.feature_names:
            X_eval["log_price"] = np.log(np.clip(candidate_prices, 0.01, None))
        if "price_ratio_comp" in self.feature_names and "competitor_price" in X_eval.columns:
            X_eval["price_ratio_comp"] = candidate_prices / np.clip(X_eval["competitor_price"], 0.01, None)
        return self.predict(X_eval)


def get_demand_model(model_name: str = "log_linear") -> BaseDemandModel:
    """Factory method for demand forecasting models."""
    models = {
        "linear_ridge": LinearRidgeDemandModel,
        "log_linear": LogLinearDemandModel,
        "random_forest": RandomForestDemandModel,
        "gradient_boosting": GradientBoostingDemandModel
    }
    if model_name not in models:
        raise ValueError(f"Unknown demand model '{model_name}'. Choose from: {list(models.keys())}")
    return models[model_name]()
