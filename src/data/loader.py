"""Dataset loading, sample generation, and provenance tracking."""
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.config import RAW_DATA_DIR, SAMPLES_DIR, METADATA_DIR
from src.utils.logger import logger
from src.data.validator import DataValidator


class DatasetProvenance:
    """Provenance and license metadata for application datasets."""

    UCI_METADATA = {
        "name": "Online Retail II Data Set",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "doi": "10.24432/C5CG6D",
        "authors": "Daqing Chen, Sai Liang Sain, and Kun Guo",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "period": "01/12/2009 - 09/12/2011",
        "is_synthetic": False,
        "description": (
            "Transactional data containing 1,067,371 records from a UK-based online retailer. "
            "NOTE: This real-world dataset does NOT contain unit costs, competitor prices, "
            "or inventory levels. Unit costs and operational constraints must be explicitly "
            "provided by the user or defined as scenario assumptions."
        )
    }

    KAGGLE_SYNTHETIC_METADATA = {
        "name": "Retail Store Inventory and Demand Forecasting",
        "source": "Kaggle",
        "url": "https://www.kaggle.com/dsv/11895299",
        "license": "Apache 2.0",
        "is_synthetic": True,
        "description": (
            "Synthetic retail dataset featuring inventory levels, competitor pricing, weather conditions, "
            "and promotional discounts. Strictly intended for scenario testing and supply-constrained "
            "demonstrations. Not observed real-world customer transactions."
        )
    }


def generate_deterministic_uci_sample(output_path: Optional[Path] = None, n_records: int = 5000) -> pd.DataFrame:
    """Generates a deterministic sample mimicking the UCI Online Retail II schema and dynamics."""
    np.random.seed(42)
    start_date = datetime(2010, 1, 1, 8, 0, 0)

    # 15 distinct products with realistic prices
    catalog = [
        {"StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "base_price": 2.55, "elasticity": -1.35},
        {"StockCode": "71053", "Description": "WHITE METAL LANTERN", "base_price": 3.39, "elasticity": -0.85},
        {"StockCode": "84406B", "Description": "CREAM CUPID HEARTS COAT HANGER", "base_price": 2.75, "elasticity": -1.10},
        {"StockCode": "84029G", "Description": "KNITTED UNION FLAG HOT WATER BOTTLE", "base_price": 3.75, "elasticity": -1.45},
        {"StockCode": "84029E", "Description": "RED WOOLLY HOTTIE WHITE HEART.", "base_price": 3.75, "elasticity": -1.20},
        {"StockCode": "22752", "Description": "SET 7 BABUSHKA NESTING BOXES", "base_price": 7.65, "elasticity": -1.80},
        {"StockCode": "21730", "Description": "GLASS STAR FROSTED T-LIGHT HOLDER", "base_price": 4.25, "elasticity": -0.95},
        {"StockCode": "22632", "Description": "HAND WARMER RED RETROSPOT", "base_price": 1.85, "elasticity": -2.10},
        {"StockCode": "22633", "Description": "HAND WARMER UNION JACK", "base_price": 1.85, "elasticity": -1.95},
        {"StockCode": "22865", "Description": "HAND WARMER OWL DESIGN", "base_price": 2.10, "elasticity": -1.75},
        {"StockCode": "22866", "Description": "HAND WARMER SCOTTY DOG DESIGN", "base_price": 2.10, "elasticity": -1.65},
        {"StockCode": "22867", "Description": "HAND WARMER BIRD DESIGN", "base_price": 2.10, "elasticity": -1.55},
        {"StockCode": "85099B", "Description": "JUMBO BAG RED RETROSPOT", "base_price": 1.95, "elasticity": -1.15},
        {"StockCode": "22386", "Description": "JUMBO BAG PINK POLKADOT", "base_price": 1.95, "elasticity": -1.05},
        {"StockCode": "20725", "Description": "LUNCH BAG RED RETROSPOT", "base_price": 1.65, "elasticity": -1.30},
    ]

    countries = ["United Kingdom", "United Kingdom", "United Kingdom", "Germany", "France", "EIRE", "Spain"]
    
    records = []
    invoice_num = 500000
    current_time = start_date

    for i in range(n_records):
        prod = catalog[np.random.choice(len(catalog))]
        current_time += timedelta(minutes=int(np.random.exponential(15)))
        
        # Realistic price fluctuations (+/- 15% promotional or dynamic variations)
        price_multiplier = np.random.choice([0.85, 0.90, 1.0, 1.0, 1.05, 1.15], p=[0.1, 0.2, 0.4, 0.15, 0.1, 0.05])
        price = round(prod["base_price"] * price_multiplier, 2)
        
        # Demand responds to price via log-log: ln(Q) = base - elasticity * ln(price/base) + noise
        log_price_ratio = np.log(price / prod["base_price"])
        expected_log_q = 2.2 + prod["elasticity"] * log_price_ratio + np.random.normal(0, 0.35)
        quantity = max(1, int(np.round(np.exp(expected_log_q))))
        
        # 3% chance of cancellation/return
        is_cancellation = np.random.rand() < 0.03
        inv_str = f"C{invoice_num}" if is_cancellation else str(invoice_num)
        if is_cancellation:
            quantity = -max(1, int(quantity * np.random.uniform(0.5, 1.0)))

        records.append({
            "Invoice": inv_str,
            "StockCode": prod["StockCode"],
            "Description": prod["Description"],
            "Quantity": quantity,
            "InvoiceDate": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Price": price,
            "Customer ID": int(np.random.choice(range(12000, 18000))),
            "Country": np.random.choice(countries)
        })

        if np.random.rand() < 0.3:
            invoice_num += 1

    df = pd.DataFrame(records)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Generated deterministic UCI sample with {len(df)} records at {output_path}")

    return df


def generate_deterministic_synthetic_sample(output_path: Optional[Path] = None, n_days: int = 180) -> pd.DataFrame:
    """Generates a deterministic synthetic scenario dataset with inventory and competitor pricing."""
    np.random.seed(101)
    start_date = datetime(2023, 1, 1)

    products = [
        {"Product ID": "PROD_A", "Category": "Electronics", "base_price": 99.99, "base_cost": 55.0, "elasticity": -1.8},
        {"Product ID": "PROD_B", "Category": "Home & Kitchen", "base_price": 34.50, "base_cost": 18.0, "elasticity": -1.2},
        {"Product ID": "PROD_C", "Category": "Apparel", "base_price": 24.99, "base_cost": 10.0, "elasticity": -2.2},
        {"Product ID": "PROD_D", "Category": "Beauty", "base_price": 14.99, "base_cost": 5.0, "elasticity": -0.9},
        {"Product ID": "PROD_E", "Category": "Sports", "base_price": 49.99, "base_cost": 28.0, "elasticity": -1.5},
    ]

    records = []
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()
        weekend_boost = 1.25 if day_of_week in [5, 6] else 1.0

        for p in products:
            discount = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.5, 0.2, 0.15, 0.1, 0.05])
            price = round(p["base_price"] * (1 - discount), 2)
            
            # Competitor price fluctuates around base
            comp_ratio = np.random.normal(1.02, 0.08)
            competitor_price = round(p["base_price"] * comp_ratio, 2)
            
            # Relative price effect
            rel_price_effect = -1.2 * (price - competitor_price) / p["base_price"]
            
            # Inventory level
            inventory_level = int(np.random.uniform(50, 400))
            
            # Demand calculation
            base_demand = 40.0 * weekend_boost
            demand = base_demand * ((price / p["base_price"]) ** p["elasticity"]) * (1 + rel_price_effect) + np.random.normal(0, 3)
            demand = max(0.0, demand)
            
            # Constrained units sold
            units_sold = int(min(demand, inventory_level))
            
            records.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Store ID": "STORE_01",
                "Product ID": p["Product ID"],
                "Category": p["Category"],
                "Region": "North",
                "Inventory Level": inventory_level,
                "Units Sold": units_sold,
                "Units Ordered": int(np.random.uniform(30, 100)),
                "Price": price,
                "Unit Cost": p["base_cost"],
                "Discount": discount,
                "Weather Conditions": np.random.choice(["Sunny", "Rainy", "Cloudy"]),
                "Promotions": int(discount > 0),
                "Competitor Pricing": competitor_price,
                "Seasonality": round(1.0 + 0.2 * np.sin(2 * np.pi * day / 365.0), 3)
            })

    df = pd.DataFrame(records)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Generated deterministic synthetic scenario sample with {len(df)} records at {output_path}")

    return df


class DataLoader:
    """Manages loading, parsing, and caching of real and synthetic datasets."""

    def __init__(self):
        self.sample_uci_path = SAMPLES_DIR / "uci_sample.csv"
        self.sample_synthetic_path = SAMPLES_DIR / "kaggle_synthetic_sample.csv"
        self._ensure_samples_exist()

    def _ensure_samples_exist(self) -> None:
        """Guarantees default samples exist so application runs out of the box."""
        if not self.sample_uci_path.exists():
            generate_deterministic_uci_sample(self.sample_uci_path)
        if not self.sample_synthetic_path.exists():
            generate_deterministic_synthetic_sample(self.sample_synthetic_path)

    def load_dataset(self, source_type: str = "uci_retail", custom_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads dataset and returns (dataframe, provenance_metadata)."""
        if source_type == "uci_retail":
            raw_uci = RAW_DATA_DIR / "online_retail_II.csv"
            raw_uci_xlsx = RAW_DATA_DIR / "online_retail_II.xlsx"
            
            if custom_path and Path(custom_path).exists():
                path = Path(custom_path)
            elif raw_uci.exists():
                path = raw_uci
            elif raw_uci_xlsx.exists():
                path = raw_uci_xlsx
            else:
                path = self.sample_uci_path
                
            logger.info(f"Loading UCI Retail dataset from {path}")
            if str(path).endswith(".xlsx") or str(path).endswith(".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            
            meta = DatasetProvenance.UCI_METADATA.copy()
            meta["loaded_file"] = str(path)
            meta["is_sample"] = (path == self.sample_uci_path)
            return df, meta

        elif source_type == "synthetic_scenario":
            raw_synth = RAW_DATA_DIR / "retail_inventory_demand.csv"
            if custom_path and Path(custom_path).exists():
                path = Path(custom_path)
            elif raw_synth.exists():
                path = raw_synth
            else:
                path = self.sample_synthetic_path

            logger.info(f"Loading synthetic scenario dataset from {path}")
            df = pd.read_csv(path)
            meta = DatasetProvenance.KAGGLE_SYNTHETIC_METADATA.copy()
            meta["loaded_file"] = str(path)
            meta["is_sample"] = (path == self.sample_synthetic_path)
            return df, meta

        elif source_type == "uploaded":
            if not custom_path or not Path(custom_path).exists():
                raise FileNotFoundError(f"Custom uploaded dataset not found at {custom_path}")
            path = Path(custom_path)
            if str(path).endswith(".xlsx") or str(path).endswith(".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)

            meta = {
                "name": f"User Upload: {path.name}",
                "source": "User Upload",
                "loaded_file": str(path),
                "is_synthetic": False,
                "is_sample": False,
                "attribution": "User-supplied dataset",
                "description": "User uploaded transactional or pricing scenario dataset."
            }
            return df, meta

        else:
            raise ValueError(f"Unknown source_type '{source_type}'. Choose 'uci_retail', 'synthetic_scenario', or 'uploaded'.")
