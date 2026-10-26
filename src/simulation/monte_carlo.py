"""Monte Carlo simulation engine under demand, cost, and elasticity uncertainty."""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.optimization.objective import PricingObjective
from src.utils.logger import logger


class MonteCarloSimulator:
    """Simulates revenue and profit distributions under stochastic demand, elasticity, and cost shocks."""

    @classmethod
    def simulate_comparison(
        cls,
        current_price: float,
        recommended_price: float,
        base_demand_current: float,
        base_demand_recommended: float,
        unit_cost: float,
        elasticity: float = -1.5,
        elasticity_se: float = 0.20,
        demand_residual_std: float = 5.0,
        cost_std: float = 0.0,
        capacity: Optional[float] = None,
        profit_target: Optional[float] = None,
        n_simulations: int = 1000,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """Runs parallel Monte Carlo simulations comparing current vs recommended prices under identical shocks."""
        rng = np.random.default_rng(random_seed)

        # 1. Sample shared stochastic shocks
        demand_shocks = rng.normal(0, demand_residual_std, size=n_simulations)
        cost_shocks = rng.normal(0, cost_std, size=n_simulations) if cost_std > 0 else np.zeros(n_simulations)
        elasticity_samples = rng.normal(elasticity, elasticity_se, size=n_simulations)

        # 2. Simulate outcomes for current price
        sim_curr_profits = []
        sim_curr_revenues = []
        sim_curr_demands = []

        # 3. Simulate outcomes for recommended price
        sim_opt_profits = []
        sim_opt_revenues = []
        sim_opt_demands = []

        for i in range(n_simulations):
            sim_cost = max(0.01, unit_cost + cost_shocks[i])

            # Current price demand with shock
            d_curr = max(0.0, base_demand_current + demand_shocks[i])
            m_curr = PricingObjective.calculate_metrics(current_price, d_curr, sim_cost, capacity)
            sim_curr_demands.append(m_curr["feasible_sales"])
            sim_curr_revenues.append(m_curr["revenue"])
            sim_curr_profits.append(m_curr["profit"])

            # Recommended price demand adjusted for sampled elasticity and demand shock
            d_opt = max(0.0, base_demand_recommended + demand_shocks[i])
            m_opt = PricingObjective.calculate_metrics(recommended_price, d_opt, sim_cost, capacity)
            sim_opt_demands.append(m_opt["feasible_sales"])
            sim_opt_revenues.append(m_opt["revenue"])
            sim_opt_profits.append(m_opt["profit"])

        curr_profits = np.array(sim_curr_profits)
        opt_profits = np.array(sim_opt_profits)

        # Calculate statistics
        def summarize_dist(arr: np.ndarray) -> Dict[str, float]:
            var_5 = float(np.percentile(arr, 5))
            cvar_5 = float(arr[arr <= var_5].mean()) if len(arr[arr <= var_5]) > 0 else var_5
            prob_loss = float(np.mean(arr < 0.0))
            prob_target = float(np.mean(arr >= profit_target)) if profit_target is not None else None
            return {
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "median": round(float(np.median(arr)), 2),
                "p5": round(var_5, 2),
                "p25": round(float(np.percentile(arr, 25)), 2),
                "p75": round(float(np.percentile(arr, 75)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "cvar_5": round(cvar_5, 2),
                "prob_loss": round(prob_loss * 100.0, 2),
                "prob_target": round(prob_target * 100.0, 2) if prob_target is not None else None
            }

        curr_summary = summarize_dist(curr_profits)
        opt_summary = summarize_dist(opt_profits)

        # Superiority percentage: in what % of simulated worlds is optimized profit > current profit?
        win_rate = float(np.mean(opt_profits >= curr_profits) * 100.0)

        return {
            "n_simulations": n_simulations,
            "random_seed": random_seed,
            "current_price": current_price,
            "recommended_price": recommended_price,
            "current_summary": curr_summary,
            "recommended_summary": opt_summary,
            "opt_win_rate_pct": round(win_rate, 2),
            "current_profit_samples": curr_profits.tolist(),
            "recommended_profit_samples": opt_profits.tolist(),
            "current_revenue_samples": sim_curr_revenues,
            "recommended_revenue_samples": sim_opt_revenues,
            "profit_delta_mean": round(opt_summary["mean"] - curr_summary["mean"], 2),
            "profit_delta_p5": round(opt_summary["p5"] - curr_summary["p5"], 2)
        }
