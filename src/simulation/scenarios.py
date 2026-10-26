"""Scenario analysis suite for stress-testing pricing decisions under macro and competitive shocks."""
from typing import Dict, Any, List, Optional
from src.optimization.objective import PricingObjective
from src.optimization.constraints import PricingConstraints
from src.optimization.optimizer import PriceOptimizer
import numpy as np


class ScenarioManager:
    """Evaluates pricing recommendations across predefined and custom scenario stress tests."""

    @classmethod
    def evaluate_scenario_suite(
        cls,
        base_demand_fn,
        constraints: PricingConstraints,
        current_opt_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluates Scenarios A through F and compares baseline vs optimized outcomes."""
        scenarios = []

        # Current baseline and optimized prices
        p_curr = constraints.current_price
        p_opt = current_opt_result.get("recommended_price") or p_curr
        base_cost = constraints.unit_cost

        # --- SCENARIO A: Demand Shock (+/- 20%) ---
        for mult, name in [(1.20, "Demand Surge (+20%)"), (0.80, "Demand Contraction (-20%)")]:
            def shocked_fn(p_arr):
                return base_demand_fn(p_arr) * mult

            m_curr = PricingObjective.calculate_metrics(p_curr, float(shocked_fn(np.array([p_curr]))[0]), base_cost, constraints.inventory_capacity)
            m_opt = PricingObjective.calculate_metrics(p_opt, float(shocked_fn(np.array([p_opt]))[0]), base_cost, constraints.inventory_capacity)
            scenarios.append({
                "scenario_id": f"scenario_a_{int(mult*100)}",
                "name": f"Scenario A: {name}",
                "description": f"Macro demand changes by {mult-1.0:+.0%}",
                "current_profit": round(m_curr["profit"], 2),
                "opt_profit": round(m_opt["profit"], 2),
                "profit_delta": round(m_opt["profit"] - m_curr["profit"], 2),
                "current_revenue": round(m_curr["revenue"], 2),
                "opt_revenue": round(m_opt["revenue"], 2),
                "feasible": True,
                "notes": "Capacity constraints may bind under surge."
            })

        # --- SCENARIO B: Competitor Price Change (+/- 15%) ---
        comp_price = constraints.competitor_price or (constraints.current_price * 1.05)
        for comp_mult, name in [(0.85, "Competitor Price Cut (-15%)"), (1.15, "Competitor Price Hike (+15%)")]:
            # Cross-price effect on demand
            cross_elasticity = 0.5
            demand_shift = 1.0 + cross_elasticity * (comp_mult - 1.0)
            def comp_fn(p_arr):
                return base_demand_fn(p_arr) * demand_shift

            m_curr = PricingObjective.calculate_metrics(p_curr, float(comp_fn(np.array([p_curr]))[0]), base_cost, constraints.inventory_capacity)
            m_opt = PricingObjective.calculate_metrics(p_opt, float(comp_fn(np.array([p_opt]))[0]), base_cost, constraints.inventory_capacity)
            scenarios.append({
                "scenario_id": f"scenario_b_{int(comp_mult*100)}",
                "name": f"Scenario B: {name}",
                "description": f"Competitor shifts price to £{comp_price * comp_mult:.2f}",
                "current_profit": round(m_curr["profit"], 2),
                "opt_profit": round(m_opt["profit"], 2),
                "profit_delta": round(m_opt["profit"] - m_curr["profit"], 2),
                "current_revenue": round(m_curr["revenue"], 2),
                "opt_revenue": round(m_opt["revenue"], 2),
                "feasible": True,
                "notes": "Assumes cross-price elasticity of +0.5."
            })

        # --- SCENARIO C: Cost Increase (+25%) ---
        shocked_cost = round(base_cost * 1.25, 2)
        m_curr_c = PricingObjective.calculate_metrics(p_curr, float(base_demand_fn(np.array([p_curr]))[0]), shocked_cost, constraints.inventory_capacity)
        m_opt_c = PricingObjective.calculate_metrics(p_opt, float(base_demand_fn(np.array([p_opt]))[0]), shocked_cost, constraints.inventory_capacity)
        margin_opt_c = (p_opt - shocked_cost) / p_opt if p_opt > 0 else 0.0
        feasible_c = margin_opt_c >= (constraints.min_gross_margin or 0.0)
        scenarios.append({
            "scenario_id": "scenario_c_cost_surge",
            "name": "Scenario C: Cost Surge (+25%)",
            "description": f"Unit cost increases from £{base_cost:.2f} to £{shocked_cost:.2f}",
            "current_profit": round(m_curr_c["profit"], 2),
            "opt_profit": round(m_opt_c["profit"], 2),
            "profit_delta": round(m_opt_c["profit"] - m_curr_c["profit"], 2),
            "current_revenue": round(m_curr_c["revenue"], 2),
            "opt_revenue": round(m_opt_c["revenue"], 2),
            "feasible": feasible_c,
            "notes": "Margin constraint may be threatened if price cannot be adjusted."
        })

        # --- SCENARIO D: Inventory Bottleneck (50% of expected sales) ---
        expected_d = float(base_demand_fn(np.array([p_opt]))[0])
        tight_cap = round(max(5.0, expected_d * 0.5), 0)
        m_curr_d = PricingObjective.calculate_metrics(p_curr, float(base_demand_fn(np.array([p_curr]))[0]), base_cost, tight_cap)
        m_opt_d = PricingObjective.calculate_metrics(p_opt, expected_d, base_cost, tight_cap)
        scenarios.append({
            "scenario_id": "scenario_d_inventory_bottleneck",
            "name": "Scenario D: Inventory Bottleneck (-50% supply cap)",
            "description": f"Available inventory restricted to {tight_cap:.0f} units",
            "current_profit": round(m_curr_d["profit"], 2),
            "opt_profit": round(m_opt_d["profit"], 2),
            "profit_delta": round(m_opt_d["profit"] - m_curr_d["profit"], 2),
            "current_revenue": round(m_curr_d["revenue"], 2),
            "opt_revenue": round(m_opt_d["revenue"], 2),
            "feasible": True,
            "notes": "Capacity binds severely; unserved demand is lost."
        })

        # --- SCENARIO E: Elasticity Uncertainty ---
        # Highly elastic market (consumers twice as sensitive)
        def elastic_fn(p_arr):
            # Q(p) = Q(p0) * (p / p0)^(-2.5)
            q0 = float(base_demand_fn(np.array([p_curr]))[0])
            return q0 * ((p_arr / p_curr) ** -2.5)

        m_curr_e = PricingObjective.calculate_metrics(p_curr, float(elastic_fn(np.array([p_curr]))[0]), base_cost, constraints.inventory_capacity)
        m_opt_e = PricingObjective.calculate_metrics(p_opt, float(elastic_fn(np.array([p_opt]))[0]), base_cost, constraints.inventory_capacity)
        scenarios.append({
            "scenario_id": "scenario_e_high_elasticity",
            "name": "Scenario E: Elasticity Uncertainty (High Elasticity -2.5)",
            "description": "Consumers exhibit much higher price sensitivity than estimated",
            "current_profit": round(m_curr_e["profit"], 2),
            "opt_profit": round(m_opt_e["profit"], 2),
            "profit_delta": round(m_opt_e["profit"] - m_curr_e["profit"], 2),
            "current_revenue": round(m_curr_e["revenue"], 2),
            "opt_revenue": round(m_opt_e["revenue"], 2),
            "feasible": True,
            "notes": "Price hikes cause steeper volume drops than expected."
        })

        # --- SCENARIO F: Combined Adverse Scenario ---
        # Cost +20%, Demand -20%, Competitor -15%
        adverse_cost = round(base_cost * 1.20, 2)
        def adverse_fn(p_arr):
            return base_demand_fn(p_arr) * 0.80 * 0.925

        m_curr_f = PricingObjective.calculate_metrics(p_curr, float(adverse_fn(np.array([p_curr]))[0]), adverse_cost, constraints.inventory_capacity)
        m_opt_f = PricingObjective.calculate_metrics(p_opt, float(adverse_fn(np.array([p_opt]))[0]), adverse_cost, constraints.inventory_capacity)
        margin_f = (p_opt - adverse_cost) / p_opt if p_opt > 0 else 0.0
        feasible_f = margin_f >= (constraints.min_gross_margin or 0.0)

        scenarios.append({
            "scenario_id": "scenario_f_combined_adverse",
            "name": "Scenario F: Combined Adverse Shock",
            "description": "Simultaneous Cost +20%, Demand -20%, and Competitor discount -15%",
            "current_profit": round(m_curr_f["profit"], 2),
            "opt_profit": round(m_opt_f["profit"], 2),
            "profit_delta": round(m_opt_f["profit"] - m_curr_f["profit"], 2),
            "current_revenue": round(m_curr_f["revenue"], 2),
            "opt_revenue": round(m_opt_f["revenue"], 2),
            "feasible": feasible_f,
            "notes": "Stress test for profit survival and margin breach."
        })

        return scenarios
