"""FastAPI routes for dynamic pricing optimization, elasticity estimation, and simulations."""
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import numpy as np

from src.database.connection import get_db
from src.database.models import User
from src.auth.authentication import verify_password, create_access_token
from src.auth.authorization import get_current_user, require_role
from src.services.demand_service import demand_service
from src.services.pricing_service import PricingService
from src.services.evaluation_service import EvaluationService
from src.data.repository import dataset_repo
from src.api.schemas import (
    UserLogin, TokenResponse, UserResponse,
    DemandPredictRequest, DemandPredictResponse,
    ElasticityRequest, ElasticityResponse,
    PriceOptimizeRequest, PriceOptimizeResponse,
    SimulateRequest, SimulateResponse,
    SensitivityRequest
)

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint displaying system status and active dataset."""
    _, meta, audit = dataset_repo.get_active_dataset()
    return {
        "status": "healthy",
        "service": "Pricing Optimization System",
        "active_dataset": meta.get("name"),
        "is_synthetic": meta.get("is_synthetic", False),
        "total_records": len(audit.get("columns", [])) and audit.get("total_rows", 0)
    }


@router.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticates user and returns an HMAC-SHA256 JWT access token."""
    user = db.query(User).filter(User.username == login_data.username).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "username": user.username
    }


@router.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
def get_user_profile(current_user: User = Depends(get_current_user)):
    """Returns profile for currently authenticated user."""
    return current_user


@router.get("/products", tags=["Catalog"])
def list_products() -> List[Dict[str, Any]]:
    """Returns available products from the active dataset."""
    return demand_service.get_available_products()


@router.post("/demand/predict", response_model=DemandPredictResponse, tags=["Demand Modeling"])
def predict_demand(req: DemandPredictRequest):
    """Predicts expected demand across a list of candidate price points."""
    try:
        demand_fn = demand_service.get_demand_curve_evaluator(req.product_id, req.model_name)
        prices = np.array(req.candidate_prices)
        demands = demand_fn(prices)
        
        evaluations = [
            {"price": float(p), "predicted_demand": round(float(d), 2)}
            for p, d in zip(prices, demands)
        ]
        return {
            "product_id": req.product_id,
            "model_used": req.model_name,
            "evaluations": evaluations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/elasticity/estimate", response_model=ElasticityResponse, tags=["Price Elasticity"])
def estimate_elasticity(req: ElasticityRequest):
    """Estimates price elasticity using log-log regression or user assumption fallback."""
    try:
        res = PricingService.get_elasticity_analysis(req.product_id, req.assumed_elasticity)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pricing/optimize", response_model=PriceOptimizeResponse, tags=["Optimization"])
def optimize_price(req: PriceOptimizeRequest):
    """Executes constrained price optimization maximizing revenue or profit."""
    try:
        res = PricingService.optimize_price(
            product_id=req.product_id,
            current_price=req.current_price,
            unit_cost=req.unit_cost,
            min_price=req.min_price,
            max_price=req.max_price,
            max_price_change_pct=req.max_price_change_pct,
            min_gross_margin=req.min_gross_margin,
            inventory_capacity=req.inventory_capacity,
            competitor_price=req.competitor_price,
            objective_type=req.objective_type,
            demand_model_name=req.demand_model_name,
            assumed_elasticity=req.assumed_elasticity,
            persist=True
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pricing/simulate", response_model=SimulateResponse, tags=["Simulation"])
def simulate_pricing(req: SimulateRequest):
    """Runs Monte Carlo simulation comparing current vs recommended prices."""
    try:
        res = PricingService.simulate_monte_carlo(
            product_id=req.product_id,
            current_price=req.current_price,
            recommended_price=req.recommended_price,
            unit_cost=req.unit_cost,
            elasticity=req.elasticity,
            elasticity_se=req.elasticity_se,
            inventory_capacity=req.inventory_capacity,
            profit_target=req.profit_target,
            n_simulations=req.n_simulations,
            demand_model_name=req.demand_model_name,
            random_seed=req.random_seed
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pricing/sensitivity", tags=["Sensitivity Analysis"])
def analyze_sensitivity(req: SensitivityRequest):
    """Calculates 1D curves and 2D parameter sensitivity matrix."""
    try:
        return PricingService.run_sensitivity(
            product_id=req.product_id,
            current_price=req.current_price,
            unit_cost=req.unit_cost,
            min_price=req.min_price,
            max_price=req.max_price,
            demand_model_name=req.demand_model_name
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/evaluation", tags=["Evaluation"])
def evaluate_models(product_id: str):
    """Executes chronological benchmarking across baseline and ML demand models."""
    try:
        return EvaluationService.get_model_benchmark(product_id=product_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
