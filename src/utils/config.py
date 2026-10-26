"""Configuration settings for the Pricing Optimization System."""
import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
METADATA_DIR = DATA_DIR / "metadata"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure essential directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DIR, METADATA_DIR, ARTIFACTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """Application settings with environment variable support."""
    app_name: str = "Pricing Optimization System"
    app_version: str = "1.0.0"
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Database
    db_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", 
            f"sqlite:///{DATA_DIR / 'pricing_system.db'}"
        )
    )
    
    # Security & Auth
    secret_key: str = Field(
        default_factory=lambda: os.getenv(
            "SECRET_KEY", 
            "pricing-optimization-insecure-secret-key-change-in-prod-2026"
        )
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 120
    
    # Default Admin Credentials
    default_admin_user: str = Field(default_factory=lambda: os.getenv("DEFAULT_ADMIN_USER", "admin"))
    default_admin_password: str = Field(default_factory=lambda: os.getenv("DEFAULT_ADMIN_PASSWORD", "AdminPass123!"))
    
    # Default Analyst Credentials
    default_analyst_user: str = Field(default_factory=lambda: os.getenv("DEFAULT_ANALYST_USER", "analyst"))
    default_analyst_password: str = Field(default_factory=lambda: os.getenv("DEFAULT_ANALYST_PASSWORD", "AnalystPass123!"))
    
    # Server configuration
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    
    # Dataset sources
    uci_retail_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    uci_retail_csv_sample_url: str = "https://raw.githubusercontent.com/datasets/online-retail/master/data/online-retail.csv"


settings = Settings()
