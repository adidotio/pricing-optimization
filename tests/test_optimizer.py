"""Unit tests for the constrained price optimization engine with analytical verification."""
import pytest
import numpy as np

from src.optimization.objective import PricingObjective
from src.optimization.constraints import PricingConstraints
from src.optimization.optimizer import PriceOptimizer


def test_analytical_monopoly_price_recovery():
    """Analytically verifies that the optimizer finds the exact monopoly price:
    For iso-elastic demand Q(P) = 100 * P^(-2.0) and cost c = 10.0,
    the unconstrained profit-maximizing price is:
    P* = c * (epsilon / (1 + epsilon)) = 10 * (-2 / -1) = 20.0.
    """
    cost = 10.0
    elasticity = -2.0

    def demand_fn(prices):
        return 1000.0 * (prices ** elasticity)

    constraints = PricingConstraints(
        current_price=15.0,
        unit_cost=cost,
        min_price=5.0,
        max_price=35.0,
        max_price_change_pct=1.0,  # Unconstraining
        min_gross_margin=0.10
    )

    opt = PriceOptimizer.optimize(demand_fn, constraints, objective_type="profit", n_grid_points=1000)

    assert opt["is_feasible"] is True
    # Recommended price must equal 20.0 within numerical grid tolerance
    assert pytest.approx(opt["recommended_price"], abs=0.15) == 20.0
    assert opt["profit_delta"] > 0


def test_maximum_price_change_constraint_binding():
    """Verifies that the price change constraint restricts recommendation when monopoly price is far."""
    # Monopoly price is 20.0, current price is 15.0, but max change is 10% (max allowable = 16.50)
    def demand_fn(prices):
        return 1000.0 * (prices ** -2.0)

    constraints = PricingConstraints(
        current_price=15.0,
        unit_cost=10.0,
        min_price=5.0,
        max_price=35.0,
        max_price_change_pct=0.10,  # Max allowable price is 15 * 1.1 = 16.50
        min_gross_margin=0.10
    )

    opt = PriceOptimizer.optimize(demand_fn, constraints, objective_type="profit")
    assert opt["is_feasible"] is True
    assert pytest.approx(opt["recommended_price"], abs=0.1) == 16.50
    assert any("Maximum price change" in c for c in opt["binding_constraints"])


def test_min_margin_constraint_binding():
    """Verifies that minimum gross margin constraint forces price upwards when cost is high."""
    cost = 20.0
    min_margin = 0.50  # Requires P >= 20 / (1 - 0.5) = 40.0

    def demand_fn(prices):
        return 100.0 * (prices ** -1.2)  # Elastic demand pulling price down

    constraints = PricingConstraints(
        current_price=30.0,
        unit_cost=cost,
        min_price=10.0,
        max_price=60.0,
        max_price_change_pct=1.0,
        min_gross_margin=min_margin
    )

    opt = PriceOptimizer.optimize(demand_fn, constraints, objective_type="profit")
    assert opt["is_feasible"] is True
    assert opt["recommended_price"] >= 40.0 - 0.05
    assert opt["expected_margin"] >= 49.9


def test_infeasible_conflicting_constraints():
    """Verifies that mathematically conflicting constraints are detected and reported as infeasible."""
    # Min margin requires price >= 50.0 (cost=25, margin=0.50), but max_price is 40.0
    constraints = PricingConstraints(
        current_price=30.0,
        unit_cost=25.0,
        min_price=10.0,
        max_price=40.0,  # Conflict: 40 < 50
        min_gross_margin=0.50
    )

    def dummy_demand(p):
        return np.ones_like(p) * 10

    opt = PriceOptimizer.optimize(dummy_demand, constraints)
    assert opt["is_feasible"] is False
    assert opt["status"] == "infeasible"
    assert opt["recommended_price"] is None
    assert "Infeasible constraint configuration" in opt["message"]


def test_inventory_capacity_capping():
    """Verifies that feasible sales cannot exceed supply capacity."""
    metrics = PricingObjective.calculate_metrics(
        price=10.0,
        expected_demand=150.0,
        unit_cost=4.0,
        capacity=80.0
    )
    assert metrics["expected_demand"] == 150.0
    assert metrics["feasible_sales"] == 80.0
    assert metrics["revenue"] == 800.0  # 10 * 80
    assert metrics["profit"] == 480.0   # (10 - 4) * 80
    assert metrics["is_capacity_constrained"] is True
