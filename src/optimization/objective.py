"""Pricing objective functions for revenue and profit maximization under capacity limits."""
from typing import Dict, Any, Optional
import numpy as np


class PricingObjective:
    """Calculates revenue, profit, margin, and feasible sales for given prices and demands."""

    @staticmethod
    def calculate_metrics(
        price: float,
        expected_demand: float,
        unit_cost: float,
        capacity: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculates commercial performance metrics for a specific price point."""
        feasible_sales = min(expected_demand, capacity) if capacity is not None else expected_demand
        feasible_sales = max(0.0, feasible_sales)

        revenue = price * feasible_sales
        profit = (price - unit_cost) * feasible_sales
        margin = (price - unit_cost) / price if price > 0 else 0.0

        return {
            "price": float(price),
            "expected_demand": float(expected_demand),
            "feasible_sales": float(feasible_sales),
            "revenue": float(revenue),
            "profit": float(profit),
            "margin": float(margin),
            "is_capacity_constrained": bool(capacity is not None and expected_demand > capacity)
        }

    @staticmethod
    def evaluate_objective(
        metrics: Dict[str, float], 
        objective_type: str = "profit"
    ) -> float:
        """Returns the scalar value to maximize based on selected objective."""
        if objective_type == "revenue":
            return metrics["revenue"]
        elif objective_type == "profit":
            return metrics["profit"]
        else:
            raise ValueError(f"Unknown objective_type '{objective_type}'. Must be 'profit' or 'revenue'.")
