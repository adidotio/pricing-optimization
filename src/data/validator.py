"""Data validation rules, schema definitions, and quality reporting."""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from src.utils.logger import logger


class DataValidator:
    """Validates schemas, missing values, duplicates, and physical bounds for pricing datasets."""

    UCI_RETAIL_REQUIRED_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price"]
    SYNTHETIC_REQUIRED_COLUMNS = ["Date", "Product ID", "Units Sold", "Price"]

    @classmethod
    def validate_uci_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validates that a DataFrame adheres to the UCI Online Retail II schema."""
        errors = []
        # Case-insensitive column matching
        col_map = {col.strip().lower(): col for col in df.columns}
        for req in cls.UCI_RETAIL_REQUIRED_COLUMNS:
            if req.lower() not in col_map:
                errors.append(f"Missing required UCI column: '{req}'")
        return len(errors) == 0, errors

    @classmethod
    def validate_synthetic_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validates that a DataFrame adheres to the synthetic scenario schema."""
        errors = []
        col_map = {col.strip().lower(): col for col in df.columns}
        for req in cls.SYNTHETIC_REQUIRED_COLUMNS:
            if req.lower() not in col_map:
                errors.append(f"Missing required synthetic scenario column: '{req}'")
        return len(errors) == 0, errors

    @classmethod
    def run_quality_checks(cls, df: pd.DataFrame, dataset_type: str = "uci_retail") -> Dict[str, Any]:
        """Runs comprehensive data quality checks and returns a structured audit report."""
        report: Dict[str, Any] = {
            "dataset_type": dataset_type,
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "warnings": [],
            "stats": {}
        }

        # Check for price and quantity columns (case-insensitive)
        cols_lower = {col.lower(): col for col in df.columns}

        # Price check
        price_col = cols_lower.get("price") or cols_lower.get("unit price") or cols_lower.get("unit_price")
        if price_col:
            non_positive_prices = int((df[price_col] <= 0).sum())
            null_prices = int(df[price_col].isnull().sum())
            report["stats"]["non_positive_prices"] = non_positive_prices
            report["stats"]["null_prices"] = null_prices
            if non_positive_prices > 0:
                report["warnings"].append(f"Found {non_positive_prices} non-positive prices in '{price_col}'.")
            if null_prices > 0:
                report["warnings"].append(f"Found {null_prices} missing prices in '{price_col}'.")

        # Quantity / Units sold check
        qty_col = cols_lower.get("quantity") or cols_lower.get("units sold") or cols_lower.get("units_sold")
        if qty_col:
            negative_qty = int((df[qty_col] < 0).sum())
            zero_qty = int((df[qty_col] == 0).sum())
            report["stats"]["negative_quantities"] = negative_qty
            report["stats"]["zero_quantities"] = zero_qty
            if negative_qty > 0:
                report["warnings"].append(f"Found {negative_qty} negative quantities (returns/cancellations) in '{qty_col}'.")

        # Competitor price check if available
        comp_col = cols_lower.get("competitor pricing") or cols_lower.get("competitor_price")
        if comp_col:
            comp_missing = int(df[comp_col].isnull().sum())
            report["stats"]["competitor_price_missing"] = comp_missing
            report["stats"]["has_competitor_data"] = True
        else:
            report["stats"]["has_competitor_data"] = False

        # Inventory level check if available
        inv_col = cols_lower.get("inventory level") or cols_lower.get("inventory_level")
        if inv_col:
            report["stats"]["has_inventory_data"] = True
        else:
            report["stats"]["has_inventory_data"] = False

        report["is_valid"] = len(report["warnings"]) == 0 or len(df) > 0
        return report
