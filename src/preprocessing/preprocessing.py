"""Data cleaning, return handling, outlier management, and time-series aggregation."""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from src.utils.logger import logger


class DataPreprocessor:
    """Cleans raw transactional or scenario data, handles returns, and aggregates to daily product time series."""

    @classmethod
    def clean_transactions(
        cls, 
        df: pd.DataFrame, 
        filter_cancellations: bool = True,
        remove_negative_prices: bool = True,
        outlier_iqr_threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cleans raw transactional data and produces an audit report of discarded/adjusted rows."""
        audit: Dict[str, Any] = {
            "initial_rows": len(df),
            "removed_cancellations": 0,
            "removed_invalid_prices": 0,
            "removed_nulls": 0,
            "outliers_adjusted": 0,
            "final_rows": 0
        }
        cleaned = df.copy()

        # Standardize column names map
        col_map = {col.strip().lower(): col for col in cleaned.columns}

        # Date parsing
        date_col = col_map.get("invoicedate") or col_map.get("date") or col_map.get("invoice_date")
        if date_col:
            cleaned["parsed_date"] = pd.to_datetime(cleaned[date_col], errors="coerce")
            null_dates = cleaned["parsed_date"].isnull().sum()
            if null_dates > 0:
                cleaned = cleaned.dropna(subset=["parsed_date"])
                audit["removed_nulls"] += int(null_dates)

        # Invoice cancellation check (e.g. 'C' prefix in UCI Retail)
        inv_col = col_map.get("invoice") or col_map.get("invoice_number")
        qty_col = col_map.get("quantity") or col_map.get("units sold") or col_map.get("units_sold")
        price_col = col_map.get("price") or col_map.get("unit price") or col_map.get("unit_price")

        # Handle returns & cancellations
        if qty_col:
            if filter_cancellations:
                # Discard invoices with 'C' or negative quantities
                is_cancelled = pd.Series(False, index=cleaned.index)
                if inv_col:
                    is_cancelled = is_cancelled | cleaned[inv_col].astype(str).str.startswith("C", na=False)
                is_cancelled = is_cancelled | (cleaned[qty_col] <= 0)
                
                cancelled_count = int(is_cancelled.sum())
                audit["removed_cancellations"] = cancelled_count
                cleaned = cleaned[~is_cancelled]

        # Handle negative or zero prices
        if price_col and remove_negative_prices:
            invalid_prices = cleaned[price_col] <= 0
            count_invalid_p = int(invalid_prices.sum())
            audit["removed_invalid_prices"] = count_invalid_p
            cleaned = cleaned[~invalid_prices]

        # Handle extreme price outliers via IQR capping if specified
        if price_col and outlier_iqr_threshold > 0 and len(cleaned) > 20:
            q25 = cleaned[price_col].quantile(0.25)
            q75 = cleaned[price_col].quantile(0.75)
            iqr = q75 - q25
            upper_bound = q75 + outlier_iqr_threshold * iqr
            extreme_prices = cleaned[price_col] > upper_bound
            audit["outliers_adjusted"] = int(extreme_prices.sum())
            # Cap extreme prices rather than silently deleting to preserve transaction records
            cleaned.loc[extreme_prices, price_col] = upper_bound

        audit["final_rows"] = len(cleaned)
        logger.info(f"Cleaned dataset: {audit['initial_rows']} -> {audit['final_rows']} rows. Removed {audit['removed_cancellations']} cancellations.")
        return cleaned, audit

    @classmethod
    def aggregate_to_daily_series(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates transaction-level data to daily product-level sales and revenue."""
        cols = {col.strip().lower(): col for col in df.columns}
        date_col = cols.get("parsed_date") or cols.get("date") or cols.get("invoicedate")
        prod_col = cols.get("stockcode") or cols.get("product id") or cols.get("product_id")
        desc_col = cols.get("description")
        qty_col = cols.get("quantity") or cols.get("units sold") or cols.get("units_sold")
        price_col = cols.get("price") or cols.get("unit price") or cols.get("unit_price")
        comp_col = cols.get("competitor pricing") or cols.get("competitor_price")
        inv_col = cols.get("inventory level") or cols.get("inventory_level")
        cost_col = cols.get("unit cost") or cols.get("unit_cost")

        temp = df.copy()
        temp["date"] = pd.to_datetime(temp[date_col]).dt.date
        temp["product_id"] = temp[prod_col].astype(str)
        temp["quantity"] = pd.to_numeric(temp[qty_col], errors="coerce").fillna(0)
        temp["price"] = pd.to_numeric(temp[price_col], errors="coerce").fillna(0)
        temp["revenue"] = temp["quantity"] * temp["price"]

        # Aggregation dictionary
        agg_dict = {
            "quantity": "sum",
            "revenue": "sum",
            "price": "mean"  # Average transaction price for the day
        }

        if desc_col and desc_col in temp.columns:
            agg_dict["description"] = "first"
            temp["description"] = temp[desc_col].astype(str)

        if comp_col and comp_col in temp.columns:
            agg_dict["competitor_price"] = "mean"
            temp["competitor_price"] = pd.to_numeric(temp[comp_col], errors="coerce")

        if inv_col and inv_col in temp.columns:
            agg_dict["inventory_level"] = "last"
            temp["inventory_level"] = pd.to_numeric(temp[inv_col], errors="coerce")

        if cost_col and cost_col in temp.columns:
            agg_dict["unit_cost"] = "mean"
            temp["unit_cost"] = pd.to_numeric(temp[cost_col], errors="coerce")

        daily = temp.groupby(["date", "product_id"]).agg(agg_dict).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values(["product_id", "date"]).reset_index(drop=True)

        # Compute volume-weighted average price (VWAP) where quantity > 0
        has_qty = daily["quantity"] > 0
        daily.loc[has_qty, "price"] = daily.loc[has_qty, "revenue"] / daily.loc[has_qty, "quantity"]

        return daily
