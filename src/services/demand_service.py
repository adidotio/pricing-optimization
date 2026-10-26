"""Demand modeling service orchestrating data ingestion, feature generation, and demand predictions."""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.data.repository import dataset_repo
from src.preprocessing.preprocessing import DataPreprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.models.demand_model import get_demand_model, BaseDemandModel
from src.utils.logger import logger


class DemandService:
    """Service providing product-level daily time series, feature engineering, and demand forecasting."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DemandService, cls).__new__(cls)
            cls._instance.daily_cache: Optional[pd.DataFrame] = None
            cls._instance.features_cache: Optional[pd.DataFrame] = None
            cls._instance.fitted_models: Dict[str, BaseDemandModel] = {}
        return cls._instance

    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns cleaned daily product series and engineered feature dataframe with caching."""
        if self.daily_cache is None or self.features_cache is None:
            raw_df, _, _ = dataset_repo.get_active_dataset()
            cleaned_df, _ = DataPreprocessor.clean_transactions(raw_df)
            daily_df = DataPreprocessor.aggregate_to_daily_series(cleaned_df)
            feat_df = FeatureEngineer.create_features(daily_df)
            self.daily_cache = daily_df
            self.features_cache = feat_df
        return self.daily_cache, self.features_cache

    def invalidate_cache(self) -> None:
        """Clears cached derived datasets when the active dataset changes."""
        self.daily_cache = None
        self.features_cache = None
        self.fitted_models.clear()
        logger.info("Demand service caches invalidated.")

    def get_available_products(self) -> List[Dict[str, Any]]:
        """Returns catalogue of unique products with transaction counts and latest price."""
        daily_df, _ = self.get_processed_data()
        grouped = daily_df.groupby("product_id")
        
        products = []
        for pid, group in grouped:
            desc = group["description"].iloc[0] if "description" in group.columns else f"Product {pid}"
            avg_p = float(group["price"].mean())
            latest_p = float(group["price"].iloc[-1])
            tot_qty = float(group["quantity"].sum())
            n_days = len(group)

            products.append({
                "product_id": str(pid),
                "description": desc,
                "latest_price": round(latest_p, 2),
                "avg_price": round(avg_p, 2),
                "total_units_sold": round(tot_qty, 1),
                "active_days": n_days,
                "unit_cost": round(float(group["unit_cost"].iloc[-1]), 2) if "unit_cost" in group.columns and not pd.isna(group["unit_cost"].iloc[-1]) else round(latest_p * 0.55, 2),
                "has_competitor": "competitor_price" in group.columns and group["competitor_price"].notnull().any(),
                "has_inventory": "inventory_level" in group.columns and group["inventory_level"].notnull().any(),
            })

        products.sort(key=lambda x: x["total_units_sold"], reverse=True)
        return products

    def train_or_get_model(
        self, 
        product_id: str, 
        model_name: str = "log_linear"
    ) -> Tuple[BaseDemandModel, pd.Series]:
        """Fits or retrieves a demand model for the target product, along with its most recent feature state."""
        cache_key = f"{product_id}_{model_name}"
        _, feat_df = self.get_processed_data()
        
        prod_data = feat_df[feat_df["product_id"] == str(product_id)].sort_values("date")
        if len(prod_data) < 5:
            raise ValueError(f"Product {product_id} has insufficient historical observations ({len(prod_data)} rows).")

        feature_cols = FeatureEngineer.get_feature_columns(prod_data)
        X = prod_data[feature_cols]
        y = prod_data["quantity"]

        if cache_key not in self.fitted_models:
            model = get_demand_model(model_name)
            model.fit(X, y)
            self.fitted_models[cache_key] = model
            logger.info(f"Fitted model {model_name} for product {product_id} on {len(X)} records.")

        latest_features = prod_data[feature_cols].iloc[-1]
        return self.fitted_models[cache_key], latest_features

    def get_demand_curve_evaluator(
        self, 
        product_id: str, 
        model_name: str = "log_linear"
    ):
        """Returns a callable f(candidate_prices) -> np.ndarray predicting demand for candidate prices."""
        model, latest_features = self.train_or_get_model(product_id, model_name)
        
        def evaluate_prices(prices: np.ndarray) -> np.ndarray:
            return model.predict_at_prices(latest_features, np.array(prices))

        return evaluate_prices


demand_service = DemandService()
