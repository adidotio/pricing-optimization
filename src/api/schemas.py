"""Pydantic schemas for FastAPI endpoints."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# Auth schemas
class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool


# Demand prediction schemas
class DemandPredictRequest(BaseModel):
    product_id: str
    candidate_prices: List[float] = Field(..., description="List of prices to evaluate")
    model_name: str = Field(default="log_linear", description="Model: linear_ridge, log_linear, random_forest, gradient_boosting")


class DemandPredictResponse(BaseModel):
    product_id: str
    model_used: str
    evaluations: List[Dict[str, float]]


# Elasticity schemas
class ElasticityRequest(BaseModel):
    product_id: str
    assumed_elasticity: Optional[float] = Field(default=None, description="Optional manual elasticity assumption")


class ElasticityResponse(BaseModel):
    product_id: str
    is_reliable: bool
    is_assumed: bool
    elasticity: Optional[float]
    std_error: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    sample_size: int
    price_cv: Optional[float] = None
    warning: Optional[str] = None
    observational_disclaimer: Optional[str] = None


# Price optimization schemas
class PriceOptimizeRequest(BaseModel):
    product_id: str
    current_price: float = Field(..., gt=0)
    unit_cost: float = Field(..., ge=0)
    min_price: float = Field(..., gt=0)
    max_price: float = Field(..., gt=0)
    max_price_change_pct: Optional[float] = Field(default=0.25, gt=0)
    min_gross_margin: Optional[float] = Field(default=0.15, lt=1.0)
    inventory_capacity: Optional[float] = Field(default=None, ge=0)
    competitor_price: Optional[float] = Field(default=None, gt=0)
    objective_type: str = Field(default="profit", description="'profit' or 'revenue'")
    demand_model_name: str = Field(default="log_linear")
    assumed_elasticity: Optional[float] = None


class PriceOptimizeResponse(BaseModel):
    status: str
    is_feasible: bool
    message: Optional[str] = None
    product_id: Optional[str] = None
    current_price: Optional[float] = None
    recommended_price: Optional[float] = None
    price_delta: Optional[float] = None
    price_delta_pct: Optional[float] = None
    unit_cost: Optional[float] = None
    current_demand: Optional[float] = None
    expected_demand: Optional[float] = None
    feasible_sales: Optional[float] = None
    current_revenue: Optional[float] = None
    expected_revenue: Optional[float] = None
    current_profit: Optional[float] = None
    expected_profit: Optional[float] = None
    profit_delta: Optional[float] = None
    profit_delta_pct: Optional[float] = None
    current_margin: Optional[float] = None
    expected_margin: Optional[float] = None
    binding_constraints: List[str] = []
    is_capacity_constrained: bool = False
    explanation: Optional[str] = None


# Simulation schemas
class SimulateRequest(BaseModel):
    product_id: str
    current_price: float
    recommended_price: float
    unit_cost: float
    elasticity: float = -1.5
    elasticity_se: float = 0.20
    inventory_capacity: Optional[float] = None
    profit_target: Optional[float] = None
    n_simulations: int = Field(default=500, le=5000)
    demand_model_name: str = "log_linear"
    random_seed: int = 42


class SimulateResponse(BaseModel):
    n_simulations: int
    random_seed: int
    current_price: float
    recommended_price: float
    current_summary: Dict[str, Any]
    recommended_summary: Dict[str, Any]
    opt_win_rate_pct: float
    profit_delta_mean: float
    profit_delta_p5: float


# Sensitivity schemas
class SensitivityRequest(BaseModel):
    product_id: str
    current_price: float
    unit_cost: float
    min_price: float
    max_price: float
    demand_model_name: str = "log_linear"
