"""Page 6 - Sensitivity & Robustness Analysis: 1D Sweeps and 2D Decision Surfaces."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.services.demand_service import demand_service
from src.services.pricing_service import PricingService

st.set_page_config(page_title="Sensitivity Analysis", page_icon="🔬", layout="wide")

st.title("🔬 Sensitivity & Robustness Analysis")
st.markdown("Examine policy fragility, optimal price trajectories, and margin resilience across shifting cost and elasticity regimes.")

products = demand_service.get_available_products()
if not products:
    st.error("No product transactions found in active dataset.")
    st.stop()

prod_options = {p["product_id"]: f"{p['product_id']} — {p['description']}" for p in products}
selected_pid = st.selectbox("Target Product", options=list(prod_options.keys()), format_func=lambda k: prod_options[k])
selected_meta = next(p for p in products if p["product_id"] == selected_pid)

curr_p = float(selected_meta["latest_price"])
unit_c = float(selected_meta["unit_cost"])

with st.spinner("Computing sensitivity sweeps and 2D parameter surface..."):
    sens_res = PricingService.run_sensitivity(
        product_id=selected_pid,
        current_price=curr_p,
        unit_cost=unit_c,
        min_price=round(curr_p * 0.5, 2),
        max_price=round(curr_p * 2.0, 2)
    )

tab_1d, tab_2d, tab_robust = st.tabs(["1D Parameter Sweeps", "2D Decision Surface (Cost × Elasticity)", "Robustness Bounds"])

with tab_1d:
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Optimal Price vs Unit Cost")
        cost_df = pd.DataFrame(sens_res["cost_sensitivity"])
        fig_cost = px.line(
            cost_df, 
            x="unit_cost", 
            y="recommended_price", 
            markers=True,
            title="Recommended Price Response to Rising Unit Cost",
            color_discrete_sequence=["#2563EB"]
        )
        fig_cost.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Unit Cost (£)", yaxis_title="Optimal Price (£)")
        st.plotly_chart(fig_cost, use_container_width=True)
        st.dataframe(cost_df[["cost_multiplier", "unit_cost", "recommended_price", "expected_profit", "expected_margin"]].round(2), hide_index=True)

    with col_s2:
        st.subheader("Optimal Price vs Elasticity")
        el_df = pd.DataFrame(sens_res["elasticity_sensitivity"])
        fig_el = px.line(
            el_df, 
            x="elasticity", 
            y="recommended_price", 
            markers=True,
            title="Optimal Price across Consumer Elasticity Regimes",
            color_discrete_sequence=["#10B981"]
        )
        fig_el.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Price Elasticity", yaxis_title="Optimal Price (£)")
        st.plotly_chart(fig_el, use_container_width=True)
        st.dataframe(el_df[["elasticity", "recommended_price", "expected_profit", "expected_margin", "price_delta_pct"]].round(2), hide_index=True)

with tab_2d:
    st.subheader("2D Parameter Grid: Optimal Price & Expected Profit")
    m2d = sens_res["matrix_2d"]
    costs = [f"£{c:.2f}" for c in m2d["costs"]]
    elasticities = [f"{e:.1f}" for e in m2d["elasticities"]]

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_h1 = px.imshow(
            m2d["optimal_price_matrix"],
            labels=dict(x="Elasticity", y="Unit Cost", color="Optimal Price (£)"),
            x=elasticities,
            y=costs,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            title="Optimal Price Surface £(Cost, Elasticity)"
        )
        fig_h1.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_h1, use_container_width=True)

    with col_h2:
        fig_h2 = px.imshow(
            m2d["expected_profit_matrix"],
            labels=dict(x="Elasticity", y="Unit Cost", color="Expected Profit (£)"),
            x=elasticities,
            y=costs,
            text_auto=".1f",
            color_continuous_scale="Blues",
            title="Expected Gross Profit Surface £(Cost, Elasticity)"
        )
        fig_h2.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_h2, use_container_width=True)

with tab_robust:
    st.subheader("Policy Robustness Diagnostics")
    
    # Calculate break-even and stability bounds
    break_even_cost = curr_p * (1.0 - 0.15)  # Cost where 15% margin binds
    st.markdown(f"""
    - **Current Price:** £{curr_p:.2f}
    - **Current Unit Cost:** £{unit_c:.2f} (Margin: {(curr_p - unit_c)/curr_p:.1%})
    - **Break-Even Cost Threshold for 15% Minimum Margin:** £{break_even_cost:.2f}
    """)

    if unit_c > break_even_cost:
        st.error(f"🚨 **Margin Infeasibility Warning:** Current cost (£{unit_c:.2f}) exceeds break-even threshold (£{break_even_cost:.2f}). Price increase is strictly required.")
    else:
        cost_cushion = break_even_cost - unit_c
        st.success(f"✅ **Margin Cushion:** Unit cost can increase by up to £{cost_cushion:.2f} (+{cost_cushion / unit_c:.1%}) before breaching minimum margin constraints.")

    st.info(
        "**Recommendation Stability Principle:** When demand is highly elastic (ε < -2.0), optimal pricing stays close to marginal cost. "
        "When demand is inelastic (ε > -1.0), unconstrained optimization pushes prices up until boundary constraints (such as max price change or minimum markup) bind."
    )
