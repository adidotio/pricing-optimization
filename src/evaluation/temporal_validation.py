"""Temporal validation strategies to prevent lookahead leakage in time-series forecasting."""
from typing import Tuple, List, Generator, Dict, Any
import pandas as pd
import numpy as np


class TemporalValidator:
    """Provides chronologically ordered train/test splits and rolling-window folds."""

    @staticmethod
    def chronological_split(
        df: pd.DataFrame, 
        test_size: float = 0.20,
        date_col: str = "date"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits time series data strictly chronologically at an exact cutoff date."""
        df_sorted = df.sort_values(date_col).copy()
        unique_dates = df_sorted[date_col].drop_duplicates().sort_values()
        
        cutoff_idx = int(len(unique_dates) * (1.0 - test_size))
        cutoff_date = unique_dates.iloc[cutoff_idx]

        train_df = df_sorted[df_sorted[date_col] < cutoff_date].copy()
        test_df = df_sorted[df_sorted[date_col] >= cutoff_date].copy()

        return train_df, test_df

    @staticmethod
    def rolling_window_split(
        df: pd.DataFrame,
        n_splits: int = 3,
        date_col: str = "date",
        test_days: int = 14
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generates rolling/expanding window train-test splits."""
        df_sorted = df.sort_values(date_col).copy()
        unique_dates = df_sorted[date_col].drop_duplicates().sort_values().to_list()
        
        folds = []
        total_dates = len(unique_dates)
        
        for k in range(n_splits, 0, -1):
            end_idx = total_dates - (k - 1) * test_days
            start_test_idx = max(0, end_idx - test_days)
            if start_test_idx <= 0:
                continue

            split_date = unique_dates[start_test_idx]
            end_date = unique_dates[min(end_idx - 1, total_dates - 1)]

            train = df_sorted[df_sorted[date_col] < split_date].copy()
            test = df_sorted[(df_sorted[date_col] >= split_date) & (df_sorted[date_col] <= end_date)].copy()

            if len(train) > 0 and len(test) > 0:
                folds.append((train, test))

        return folds
