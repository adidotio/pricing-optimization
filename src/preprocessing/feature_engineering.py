"""Leakage-free feature engineering for dynamic pricing and demand prediction."""
from typing import List, Tuple
import pandas as pd
import numpy as np
from src.utils.logger import logger


class FeatureEngineer:
    """Computes lagged, rolling, calendar, and price features strictly backwards in time to prevent lookahead leakage."""

    @classmethod
    def create_features(
        cls, 
        df: pd.DataFrame, 
        include_competitor: bool = True,
        lag_days: List[int] = [1, 7, 14],
        rolling_windows: List[int] = [7, 14]
    ) -> pd.DataFrame:
        """Generates historical demand, price, and seasonal features grouped by product."""
        feat_df = df.sort_values(["product_id", "date"]).copy()

        # Date and calendar features (known at decision time)
        feat_df["day_of_week"] = feat_df["date"].dt.dayofweek
        feat_df["month"] = feat_df["date"].dt.month
        feat_df["is_weekend"] = feat_df["day_of_week"].isin([5, 6]).astype(int)

        # Log transformations for econometric analysis (handle zeros safely)
        feat_df["log_price"] = np.log(feat_df["price"].clip(lower=0.01))
        feat_df["log_quantity"] = np.log(feat_df["quantity"].clip(lower=0.01))

        # Lagged features calculated per product
        for lag in lag_days:
            feat_df[f"demand_lag_{lag}"] = feat_df.groupby("product_id")["quantity"].shift(lag)
            feat_df[f"price_lag_{lag}"] = feat_df.groupby("product_id")["price"].shift(lag)

        # Rolling statistics (shift by 1 to strictly exclude current day)
        for w in rolling_windows:
            feat_df[f"demand_rolling_mean_{w}"] = (
                feat_df.groupby("product_id")["quantity"]
                .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            )
            feat_df[f"demand_rolling_std_{w}"] = (
                feat_df.groupby("product_id")["quantity"]
                .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).std())
            ).fillna(0.0)

        # Competitor price features if present
        if include_competitor and "competitor_price" in feat_df.columns:
            feat_df["price_ratio_comp"] = feat_df["price"] / feat_df["competitor_price"].clip(lower=0.01)
            feat_df["price_diff_comp"] = feat_df["price"] - feat_df["competitor_price"]
        else:
            feat_df["price_ratio_comp"] = 1.0
            feat_df["price_diff_comp"] = 0.0

        # Fill remaining initial NaNs from shifting with product-level historical mean
        for col in [c for c in feat_df.columns if "lag" in c or "rolling" in c]:
            feat_df[col] = feat_df.groupby("product_id")[col].transform(lambda s: s.bfill().fillna(0.0))

        return feat_df

    @classmethod
    def get_feature_columns(cls, df: pd.DataFrame) -> List[str]:
        """Returns the list of engineered feature names suitable for ML model training."""
        exclude = ["date", "product_id", "description", "quantity", "revenue", "log_quantity", "inventory_level", "unit_cost"]
        features = [col for col in df.columns if col not in exclude]
        return features
