"""Constrained price optimization engine and baseline comparison."""
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.optimization.objective import PricingObjective
from src.optimization.constraints import PricingConstraints
from src.utils.logger import logger


class PriceOptimizer:
    """Solves bounded, constrained price optimization using candidate grid search and numerical refinement."""

    @classmethod
    def optimize(
        cls,
        demand_fn: Callable[[np.ndarray], np.ndarray],
        constraints: PricingConstraints,
        objective_type: str = "profit",
        n_grid_points: int = 500
    ) -> Dict[str, Any]:
        """Finds the optimal price maximizing profit or revenue under all operational constraints."""
        # 1. Validate inputs
        is_valid, input_errors = constraints.validate_inputs()
        if not is_valid:
            return {
                "status": "error",
                "is_feasible": False,
                "message": f"Invalid constraint parameters: {'; '.join(input_errors)}",
                "recommended_price": None
            }

        # 2. Get effective bounds
        eff_min, eff_max, conflict_notes = constraints.get_effective_price_bounds()
        if conflict_notes or eff_min > eff_max:
            return {
                "status": "infeasible",
                "is_feasible": False,
                "message": (
                    f"Infeasible constraint configuration: Lower bound ({eff_min:.2f}) exceeds upper bound ({eff_max:.2f}). "
                    f"Details: {'; '.join(conflict_notes)}"
                ),
                "conflicts": conflict_notes,
                "effective_min": eff_min,
                "effective_max": eff_max,
                "recommended_price": None
            }

        # 3. Dense candidate evaluation grid across feasible domain
        candidate_prices = np.linspace(eff_min, eff_max, n_grid_points)
        expected_demands = demand_fn(candidate_prices)

        grid_results = []
        best_objective_val = -float("inf")
        best_candidate: Optional[Dict[str, Any]] = None

        for p, d in zip(candidate_prices, expected_demands):
            feasible, violations = constraints.check_feasibility(p)
            metrics = PricingObjective.calculate_metrics(
                price=float(p),
                expected_demand=float(d),
                unit_cost=constraints.unit_cost,
                capacity=constraints.inventory_capacity
            )
            metrics["is_feasible"] = feasible
            metrics["violations"] = violations
            obj_val = PricingObjective.evaluate_objective(metrics, objective_type)
            metrics["objective_value"] = obj_val
            grid_results.append(metrics)

            if feasible and obj_val > best_objective_val:
                best_objective_val = obj_val
                best_candidate = metrics

        if best_candidate is None:
            return {
                "status": "infeasible",
                "is_feasible": False,
                "message": "No feasible price candidate found satisfying all constraints.",
                "effective_min": eff_min,
                "effective_max": eff_max,
                "recommended_price": None
            }

        recommended_price = round(best_candidate["price"], 2)

        # 4. Binding constraints detection
        binding_constraints = []
        tol = 0.02
        if abs(recommended_price - constraints.min_price) <= tol:
            binding_constraints.append("Minimum price bound")
        if abs(recommended_price - constraints.max_price) <= tol:
            binding_constraints.append("Maximum price bound")

        if constraints.min_gross_margin is not None:
            margin_bound = constraints.unit_cost / (1.0 - constraints.min_gross_margin)
            if abs(recommended_price - margin_bound) <= tol:
                binding_constraints.append(f"Minimum gross margin ({constraints.min_gross_margin:.1%}) floor")

        if constraints.max_price_change_pct is not None:
            chg_floor = constraints.current_price * (1.0 - constraints.max_price_change_pct)
            chg_ceil = constraints.current_price * (1.0 + constraints.max_price_change_pct)
            if abs(recommended_price - chg_floor) <= tol:
                binding_constraints.append(f"Maximum price change reduction limit (-{constraints.max_price_change_pct:.1%})")
            if abs(recommended_price - chg_ceil) <= tol:
                binding_constraints.append(f"Maximum price change increase limit (+{constraints.max_price_change_pct:.1%})")

        if best_candidate["is_capacity_constrained"]:
            binding_constraints.append(f"Inventory capacity ceiling ({constraints.inventory_capacity:.0f} units)")

        # 5. Baseline evaluation at current price
        curr_demand = float(demand_fn(np.array([constraints.current_price]))[0])
        curr_metrics = PricingObjective.calculate_metrics(
            price=constraints.current_price,
            expected_demand=curr_demand,
            unit_cost=constraints.unit_cost,
            capacity=constraints.inventory_capacity
        )

        # 6. Generate explainable recommendation rationale
        explanation = cls._generate_explanation(
            current_metrics=curr_metrics,
            opt_metrics=best_candidate,
            constraints=constraints,
            binding=binding_constraints,
            objective_type=objective_type
        )

        return {
            "status": "success",
            "is_feasible": True,
            "objective_type": objective_type,
            "current_price": constraints.current_price,
            "recommended_price": recommended_price,
            "price_delta": round(recommended_price - constraints.current_price, 2),
            "price_delta_pct": round((recommended_price - constraints.current_price) / constraints.current_price * 100.0, 2),
            "unit_cost": constraints.unit_cost,
            "current_demand": round(curr_metrics["expected_demand"], 1),
            "expected_demand": round(best_candidate["expected_demand"], 1),
            "feasible_sales": round(best_candidate["feasible_sales"], 1),
            "current_revenue": round(curr_metrics["revenue"], 2),
            "expected_revenue": round(best_candidate["revenue"], 2),
            "current_profit": round(curr_metrics["profit"], 2),
            "expected_profit": round(best_candidate["profit"], 2),
            "profit_delta": round(best_candidate["profit"] - curr_metrics["profit"], 2),
            "profit_delta_pct": round(
                ((best_candidate["profit"] - curr_metrics["profit"]) / curr_metrics["profit"] * 100.0) 
                if curr_metrics["profit"] > 0 else 0.0, 2
            ),
            "current_margin": round(curr_metrics["margin"] * 100.0, 2),
            "expected_margin": round(best_candidate["margin"] * 100.0, 2),
            "binding_constraints": binding_constraints,
            "is_capacity_constrained": best_candidate["is_capacity_constrained"],
            "explanation": explanation,
            "grid_results": grid_results
        }

    @classmethod
    def _generate_explanation(
        cls,
        current_metrics: Dict[str, float],
        opt_metrics: Dict[str, float],
        constraints: PricingConstraints,
        binding: List[str],
        objective_type: str
    ) -> str:
        """Generates plain-language explanation of why the recommended price was selected."""
        p_opt = opt_metrics["price"]
        p_curr = current_metrics["price"]
        diff = p_opt - p_curr
        dir_text = "increased" if diff > 0 else ("decreased" if diff < 0 else "maintained")

        profit_gain = opt_metrics["profit"] - current_metrics["profit"]
        rev_gain = opt_metrics["revenue"] - current_metrics["revenue"]

        expl = (
            f"The recommended price is £{p_opt:.2f}, {dir_text} by {abs(diff):.2f} "
            f"({abs(diff) / p_curr:.1%}) relative to the current price £{p_curr:.2f}. "
        )

        if objective_type == "profit":
            expl += (
                f"This maximizes expected gross profit, achieving £{opt_metrics['profit']:.2f} "
                f"(+£{profit_gain:.2f} or +{profit_gain / max(1e-4, current_metrics['profit']):.1%}) "
                f"with an expected unit margin of {opt_metrics['margin']:.1%}. "
            )
        else:
            expl += (
                f"This maximizes total expected revenue, achieving £{opt_metrics['revenue']:.2f} "
                f"(+£{rev_gain:.2f}) with expected sales of {opt_metrics['feasible_sales']:.1f} units. "
            )

        if binding:
            expl += f"Active binding constraints restricting further price adjustments: {', '.join(binding)}. "
        else:
            expl += "The recommendation is interior to all bounds and unconstrained by operational limits. "

        if opt_metrics["is_capacity_constrained"]:
            expl += (
                f"Note: Sales are strictly capped at inventory capacity ({constraints.inventory_capacity:.0f} units). "
                "Any unserved excess demand represents potential lost sales."
            )

        return expl
