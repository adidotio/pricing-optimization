"""Pricing service orchestrating elasticity estimation, optimization, simulation, and sensitivity."""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.services.demand_service import demand_service
from src.models.elasticity import PriceElasticityEstimator
from src.optimization.constraints import PricingConstraints
from src.optimization.optimizer import PriceOptimizer
from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.scenarios import ScenarioManager
from src.simulation.sensitivity import SensitivityAnalyzer
from src.database.connection import SessionLocal
from src.database.models import OptimizationRecord
from src.utils.logger import logger
import json


class PricingService:
    """End-to-end pricing recommendation and analytical simulation service."""

    @classmethod
    def get_elasticity_analysis(
        cls, 
        product_id: str, 
        assumed_elasticity: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates empirical elasticity or applies user-assumed fallback."""
        daily_df, _ = demand_service.get_processed_data()
        
        if assumed_elasticity is not None:
            return PriceElasticityEstimator.get_assumed_elasticity_fallback(
                product_id=product_id,
                assumed_value=assumed_elasticity,
                rationale="User-specified manual elasticity assumption"
            )

        # Empirical estimation
        result = PriceElasticityEstimator.estimate_elasticity(daily_df, product_id=product_id)
        if not result["is_reliable"]:
            # Provide sensible fallback while preserving warning
            fallback = PriceElasticityEstimator.get_assumed_elasticity_fallback(
                product_id=product_id,
                assumed_value=-1.45,
                rationale="Fallback assumption due to insufficient variation in observational data"
            )
            fallback["warning"] = result.get("warning")
            fallback["price_cv"] = result.get("price_cv")
            return fallback

        return result

    @classmethod
    def optimize_price(
        cls,
        product_id: str,
        current_price: float,
        unit_cost: float,
        min_price: float,
        max_price: float,
        max_price_change_pct: Optional[float] = 0.25,
        min_gross_margin: Optional[float] = 0.15,
        inventory_capacity: Optional[float] = None,
        competitor_price: Optional[float] = None,
        objective_type: str = "profit",
        demand_model_name: str = "log_linear",
        assumed_elasticity: Optional[float] = None,
        persist: bool = True
    ) -> Dict[str, Any]:
        """Solves optimal price and records recommendation in the database."""
        # 1. Obtain demand function
        demand_fn = demand_service.get_demand_curve_evaluator(product_id, demand_model_name)

        # 2. Build constraints
        constraints = PricingConstraints(
            current_price=current_price,
            unit_cost=unit_cost,
            min_price=min_price,
            max_price=max_price,
            max_price_change_pct=max_price_change_pct,
            min_gross_margin=min_gross_margin,
            inventory_capacity=inventory_capacity,
            competitor_price=competitor_price
        )

        # 3. Solve optimization
        result = PriceOptimizer.optimize(
            demand_fn=demand_fn,
            constraints=constraints,
            objective_type=objective_type
        )
        result["product_id"] = product_id
        result["demand_model_used"] = demand_model_name

        # 4. Elasticity diagnostic
        elasticity_info = cls.get_elasticity_analysis(product_id, assumed_elasticity)
        result["elasticity_info"] = elasticity_info

        # 5. Database persistence
        if persist and result.get("is_feasible"):
            try:
                db = SessionLocal()
                rec = OptimizationRecord(
                    product_id=product_id,
                    current_price=current_price,
                    recommended_price=result["recommended_price"],
                    unit_cost=unit_cost,
                    expected_demand_current=result.get("current_demand"),
                    expected_demand_opt=result.get("expected_demand"),
                    expected_revenue_opt=result.get("expected_revenue"),
                    expected_profit_opt=result.get("expected_profit"),
                    expected_margin_opt=result.get("expected_margin"),
                    objective=objective_type,
                    constraints_json=json.dumps({
                        "min_price": min_price,
                        "max_price": max_price,
                        "max_price_change_pct": max_price_change_pct,
                        "min_gross_margin": min_gross_margin,
                        "inventory_capacity": inventory_capacity
                    }),
                    binding_constraints=", ".join(result.get("binding_constraints", [])),
                    explanation=result.get("explanation")
                )
                db.add(rec)
                db.commit()
                db.close()
            except Exception as e:
                logger.error(f"Failed to persist optimization record: {e}")

        return result

    @classmethod
    def simulate_monte_carlo(
        cls,
        product_id: str,
        current_price: float,
        recommended_price: float,
        unit_cost: float,
        elasticity: float = -1.5,
        elasticity_se: float = 0.20,
        inventory_capacity: Optional[float] = None,
        profit_target: Optional[float] = None,
        n_simulations: int = 1000,
        demand_model_name: str = "log_linear",
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """Runs stochastic Monte Carlo comparison between current and recommended prices."""
        demand_fn = demand_service.get_demand_curve_evaluator(product_id, demand_model_name)
        base_d_curr = float(demand_fn(np.array([current_price]))[0])
        base_d_opt = float(demand_fn(np.array([recommended_price]))[0])

        sim_res = MonteCarloSimulator.simulate_comparison(
            current_price=current_price,
            recommended_price=recommended_price,
            base_demand_current=base_d_curr,
            base_demand_recommended=base_d_opt,
            unit_cost=unit_cost,
            elasticity=elasticity,
            elasticity_se=elasticity_se,
            demand_residual_std=max(2.0, base_d_curr * 0.15),
            capacity=inventory_capacity,
            profit_target=profit_target,
            n_simulations=n_simulations,
            random_seed=random_seed
        )
        return sim_res

    @classmethod
    def run_scenarios(
        cls,
        product_id: str,
        current_price: float,
        recommended_price: float,
        unit_cost: float,
        min_price: float,
        max_price: float,
        inventory_capacity: Optional[float] = None,
        demand_model_name: str = "log_linear"
    ) -> List[Dict[str, Any]]:
        """Evaluates stress-test scenarios A through F."""
        demand_fn = demand_service.get_demand_curve_evaluator(product_id, demand_model_name)
        constraints = PricingConstraints(
            current_price=current_price,
            unit_cost=unit_cost,
            min_price=min_price,
            max_price=max_price,
            inventory_capacity=inventory_capacity
        )
        current_opt = {"recommended_price": recommended_price}
        return ScenarioManager.evaluate_scenario_suite(demand_fn, constraints, current_opt)

    @classmethod
    def run_sensitivity(
        cls,
        product_id: str,
        current_price: float,
        unit_cost: float,
        min_price: float,
        max_price: float,
        demand_model_name: str = "log_linear"
    ) -> Dict[str, Any]:
        """Generates 1D curves and 2D sensitivity matrix."""
        demand_fn = demand_service.get_demand_curve_evaluator(product_id, demand_model_name)
        constraints = PricingConstraints(
            current_price=current_price,
            unit_cost=unit_cost,
            min_price=min_price,
            max_price=max_price
        )
        current_demand = float(demand_fn(np.array([current_price]))[0])

        price_profit_curve = SensitivityAnalyzer.sweep_price_vs_profit(demand_fn, constraints, n_points=40)
        cost_sweep = SensitivityAnalyzer.sweep_cost_sensitivity(demand_fn, constraints)
        elasticity_sweep = SensitivityAnalyzer.sweep_elasticity_sensitivity(current_price, current_demand, unit_cost)
        matrix_2d = SensitivityAnalyzer.compute_2d_sensitivity_matrix(current_price, current_demand, unit_cost)

        return {
            "price_profit_curve": price_profit_curve,
            "cost_sensitivity": cost_sweep,
            "elasticity_sensitivity": elasticity_sweep,
            "matrix_2d": matrix_2d
        }
