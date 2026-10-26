"""Dataset repository managing active datasets, provenance, and data caching."""
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from src.data.loader import DataLoader, DatasetProvenance
from src.data.validator import DataValidator
from src.utils.logger import logger


class DatasetRepository:
    """Singleton-style repository managing active dataset state, metadata, and quality audits."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatasetRepository, cls).__new__(cls)
            cls._instance.loader = DataLoader()
            cls._instance.active_dataset_type = "uci_retail"
            cls._instance.active_raw_df: Optional[pd.DataFrame] = None
            cls._instance.active_metadata: Dict[str, Any] = {}
            cls._instance.active_audit_report: Dict[str, Any] = {}
            cls._instance.load_active_dataset("uci_retail")
        return cls._instance

    def load_active_dataset(self, source_type: str = "uci_retail", custom_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads and activates a dataset with full quality validation."""
        df, meta = self.loader.load_dataset(source_type=source_type, custom_path=custom_path)
        audit = DataValidator.run_quality_checks(df, dataset_type=source_type)

        self.active_dataset_type = source_type
        self.active_raw_df = df
        self.active_metadata = meta
        self.active_audit_report = audit

        logger.info(f"Activated dataset: {source_type} ({len(df)} rows, {len(df.columns)} columns)")
        return df, meta

    def get_active_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Returns the currently active (dataframe, metadata, quality_report)."""
        if self.active_raw_df is None:
            self.load_active_dataset(self.active_dataset_type)
        return self.active_raw_df, self.active_metadata, self.active_audit_report


dataset_repo = DatasetRepository()
