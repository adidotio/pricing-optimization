"""Sensitivity and robustness analysis for dynamic pricing policies."""
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from src.optimization.objective import PricingObjective
from src.optimization.constraints import PricingConstraints
from src.optimization.optimizer import PriceOptimizer


class SensitivityAnalyzer:
    """Computes single-parameter and multi-parameter sensitivity sweeps and robustness diagnostics."""

    @classmethod
    def sweep_price_vs_profit(
        cls,
        demand_fn: Callable[[np.ndarray], np.ndarray],
        constraints: PricingConstraints,
        n_points: int = 50
    ) -> List[Dict[str, Any]]:
        """Generates the full price-versus-profit and price-versus-revenue curve across candidate prices."""
        p_min = max(0.01, constraints.current_price * 0.6)
        p_max = constraints.current_price * 1.5
        prices = np.linspace(p_min, p_max, n_points)
        demands = demand_fn(prices)

        curve = []
        for p, d in zip(prices, demands):
            metrics = PricingObjective.calculate_metrics(
                price=float(p),
                expected_demand=float(d),
                unit_cost=constraints.unit_cost,
                capacity=constraints.inventory_capacity
            )
            feasible, violations = constraints.check_feasibility(p)
            metrics["is_feasible"] = feasible
            metrics["violations"] = violations
            curve.append(metrics)

        return curve

    @classmethod
    def sweep_cost_sensitivity(
        cls,
        demand_fn: Callable[[np.ndarray], np.ndarray],
        base_constraints: PricingConstraints,
        cost_multipliers: List[float] = [0.7, 0.85, 1.0, 1.15, 1.30, 1.50]
    ) -> List[Dict[str, Any]]:
        """Evaluates how optimal price and profit shift as unit cost fluctuates."""
        results = []
        for mult in cost_multipliers:
            c = round(base_constraints.unit_cost * mult, 2)
            c_constraints = PricingConstraints(
                current_price=base_constraints.current_price,
                unit_cost=c,
                min_price=base_constraints.min_price,
                max_price=base_constraints.max_price,
                max_price_change_pct=base_constraints.max_price_change_pct,
                min_gross_margin=base_constraints.min_gross_margin,
                inventory_capacity=base_constraints.inventory_capacity
            )
            opt = PriceOptimizer.optimize(demand_fn, c_constraints, objective_type="profit")
            results.append({
                "cost_multiplier": mult,
                "unit_cost": c,
                "recommended_price": opt.get("recommended_price"),
                "expected_profit": opt.get("expected_profit"),
                "expected_margin": opt.get("expected_margin"),
                "is_feasible": opt.get("is_feasible", False)
            })
        return results

    @classmethod
    def sweep_elasticity_sensitivity(
        cls,
        current_price: float,
        current_demand: float,
        unit_cost: float,
        elasticities: List[float] = [-0.8, -1.0, -1.2, -1.5, -1.8, -2.2, -2.8]
    ) -> List[Dict[str, Any]]:
        """Evaluates optimal pricing under varying consumer elasticity assumptions using iso-elastic demand."""
        results = []
        for eps in elasticities:
            # Theoretical unconstrained monopoly price for constant elasticity: P* = c * (eps / (1 + eps)) if eps < -1
            def iso_elastic_demand(p_arr):
                return current_demand * ((p_arr / current_price) ** eps)

            constraints = PricingConstraints(
                current_price=current_price,
                unit_cost=unit_cost,
                min_price=round(current_price * 0.5, 2),
                max_price=round(current_price * 2.0, 2),
                max_price_change_pct=0.40,
                min_gross_margin=0.10
            )
            opt = PriceOptimizer.optimize(iso_elastic_demand, constraints, objective_type="profit")
            results.append({
                "elasticity": eps,
                "recommended_price": opt.get("recommended_price"),
                "expected_demand": opt.get("expected_demand"),
                "expected_profit": opt.get("expected_profit"),
                "expected_margin": opt.get("expected_margin"),
                "price_delta_pct": opt.get("price_delta_pct")
            })
        return results

    @classmethod
    def compute_2d_sensitivity_matrix(
        cls,
        current_price: float,
        current_demand: float,
        base_unit_cost: float,
        costs: Optional[List[float]] = None,
        elasticities: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Generates a 2D matrix of optimal prices and profits across (Cost x Elasticity) grid."""
        if costs is None:
            costs = [round(base_unit_cost * m, 2) for m in [0.8, 1.0, 1.2, 1.4]]
        if elasticities is None:
            elasticities = [-1.0, -1.4, -1.8, -2.4]

        price_matrix = []
        profit_matrix = []

        for c in costs:
            p_row = []
            prof_row = []
            for eps in elasticities:
                def d_fn(p_arr):
                    return current_demand * ((p_arr / current_price) ** eps)

                constraints = PricingConstraints(
                    current_price=current_price,
                    unit_cost=c,
                    min_price=round(current_price * 0.5, 2),
                    max_price=round(current_price * 2.0, 2),
                    max_price_change_pct=0.50,
                    min_gross_margin=0.10
                )
                opt = PriceOptimizer.optimize(d_fn, constraints)
                p_row.append(opt.get("recommended_price"))
                prof_row.append(opt.get("expected_profit"))
            price_matrix.append(p_row)
            profit_matrix.append(prof_row)

        return {
            "costs": costs,
            "elasticities": elasticities,
            "optimal_price_matrix": price_matrix,
            "expected_profit_matrix": profit_matrix
        }
