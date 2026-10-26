"""Unit tests for Monte Carlo simulation and stress-testing scenarios."""
import pytest
import numpy as np

from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.scenarios import ScenarioManager
from src.optimization.constraints import PricingConstraints


def test_monte_carlo_reproducibility():
    """Verifies that fixed random seed yields perfectly reproducible stochastic simulation distributions."""
    sim1 = MonteCarloSimulator.simulate_comparison(
        current_price=10.0,
        recommended_price=12.0,
        base_demand_current=50.0,
        base_demand_recommended=45.0,
        unit_cost=5.0,
        n_simulations=100,
        random_seed=123
    )

    sim2 = MonteCarloSimulator.simulate_comparison(
        current_price=10.0,
        recommended_price=12.0,
        base_demand_current=50.0,
        base_demand_recommended=45.0,
        unit_cost=5.0,
        n_simulations=100,
        random_seed=123
    )

    assert sim1["recommended_summary"]["mean"] == sim2["recommended_summary"]["mean"]
    assert sim1["recommended_summary"]["p5"] == sim2["recommended_summary"]["p5"]
    assert sim1["opt_win_rate_pct"] == sim2["opt_win_rate_pct"]


def test_monte_carlo_downside_metrics():
    """Verifies that 5% VaR and CVaR are logically ordered (CVaR <= VaR)."""
    sim = MonteCarloSimulator.simulate_comparison(
        current_price=15.0,
        recommended_price=17.0,
        base_demand_current=40.0,
        base_demand_recommended=36.0,
        unit_cost=8.0,
        n_simulations=200,
        random_seed=42
    )

    rec_sum = sim["recommended_summary"]
    # 5th percentile is lower than median
    assert rec_sum["p5"] < rec_sum["median"]
    # CVaR (expected shortfall) is less than or equal to VaR
    assert rec_sum["cvar_5"] <= rec_sum["p5"]


def test_scenario_suite_execution():
    """Verifies that Scenarios A through F execute and produce structured comparisons."""
    def dummy_demand_fn(prices):
        return 50.0 * (prices / 10.0) ** -1.5

    constraints = PricingConstraints(
        current_price=10.0,
        unit_cost=5.0,
        min_price=6.0,
        max_price=15.0,
        min_gross_margin=0.15
    )

    scenarios = ScenarioManager.evaluate_scenario_suite(
        base_demand_fn=dummy_demand_fn,
        constraints=constraints,
        current_opt_result={"recommended_price": 11.5}
    )

    assert len(scenarios) >= 6
    scenario_ids = [s["scenario_id"] for s in scenarios]
    assert any("scenario_a" in sid for sid in scenario_ids)
    assert any("scenario_c" in sid for sid in scenario_ids)
    assert any("scenario_f" in sid for sid in scenario_ids)
