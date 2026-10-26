"""Database models using SQLAlchemy."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    """User account model for authentication and access control."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="analyst", nullable=False)  # admin, analyst, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatasetMeta(Base):
    """Metadata tracking loaded datasets and their provenance."""
    __tablename__ = "dataset_metadata"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    source_type = Column(String(64), nullable=False)  # uci_retail, synthetic_scenario, uploaded
    file_path = Column(String(512), nullable=False)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    date_range_start = Column(String(64), nullable=True)
    date_range_end = Column(String(64), nullable=True)
    is_synthetic = Column(Boolean, default=False)
    attribution = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    """Tracking of trained demand models and evaluation metrics."""
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(64), nullable=False, index=True)
    model_name = Column(String(64), nullable=False)
    parameters_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)  # MAE, RMSE, MAPE
    created_at = Column(DateTime, default=datetime.utcnow)


class OptimizationRecord(Base):
    """History of dynamic pricing optimization decisions."""
    __tablename__ = "optimization_records"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(64), nullable=False, index=True)
    current_price = Column(Float, nullable=False)
    recommended_price = Column(Float, nullable=False)
    unit_cost = Column(Float, nullable=False)
    expected_demand_current = Column(Float, nullable=True)
    expected_demand_opt = Column(Float, nullable=True)
    expected_revenue_opt = Column(Float, nullable=True)
    expected_profit_opt = Column(Float, nullable=True)
    expected_margin_opt = Column(Float, nullable=True)
    objective = Column(String(32), default="profit")
    constraints_json = Column(Text, nullable=True)
    binding_constraints = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ScenarioRecord(Base):
    """Persisted user scenario simulations."""
    __tablename__ = "scenario_records"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    product_id = Column(String(64), nullable=False)
    scenario_type = Column(String(64), nullable=False)
    assumptions_json = Column(Text, nullable=False)
    results_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
