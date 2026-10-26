"""Page 5 - Scenario Simulator: Monte Carlo Uncertainty & Stress Testing."""
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.services.demand_service import demand_service
from src.services.pricing_service import PricingService

st.set_page_config(page_title="Scenario Simulator", page_icon="🎲", layout="wide")

st.title("🎲 Stochastic Simulation & Scenario Stress-Testing")
st.markdown("Quantify uncertainty in demand, costs, and price elasticity via Monte Carlo simulation and stress-test Scenarios A through F.")

products = demand_service.get_available_products()
if not products:
    st.error("No product transactions found in active dataset.")
    st.stop()

# Product selection
prod_options = {p["product_id"]: f"{p['product_id']} — {p['description']}" for p in products}
selected_pid = st.selectbox("Select Target Product", options=list(prod_options.keys()), format_func=lambda k: prod_options[k])
selected_meta = next(p for p in products if p["product_id"] == selected_pid)

tab_mc, tab_scenarios = st.tabs(["Monte Carlo Simulation", "Scenario Suite (A through F)"])

with tab_mc:
    st.subheader("Monte Carlo Uncertainty Simulation")
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        curr_p = st.number_input("Current Price (£)", min_value=0.1, value=float(selected_meta["latest_price"]), step=0.1)
        rec_p = st.number_input("Recommended Price (£)", min_value=0.1, value=round(selected_meta["latest_price"] * 1.10, 2), step=0.1)
    with col_c2:
        unit_c = st.number_input("Unit Cost (£)", min_value=0.0, value=float(selected_meta["unit_cost"]), step=0.1)
        target_profit = st.number_input("Target Profit Threshold (£)", min_value=0.0, value=round(selected_meta["latest_price"] * 15.0, 0), step=10.0)
    with col_c3:
        n_sims = st.selectbox("Simulation Runs", [200, 500, 1000, 2000], index=2)
        seed = st.number_input("Random Seed (Reproducibility)", value=42, step=1)
    with col_c4:
        el_val = st.number_input("Central Elasticity", value=-1.45, step=0.05)
        el_se = st.number_input("Elasticity Std Error", value=0.20, step=0.02)

    if st.button("▶ Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {n_sims} stochastic simulation paths..."):
            sim_res = PricingService.simulate_monte_carlo(
                product_id=selected_pid,
                current_price=curr_p,
                recommended_price=rec_p,
                unit_cost=unit_c,
                elasticity=el_val,
                elasticity_se=el_se,
                profit_target=target_profit,
                n_simulations=n_sims,
                random_seed=seed
            )

        st.markdown("---")
        st.subheader("Stochastic Profit Distributions")

        s_cols = st.columns(4)
        c_sum = sim_res["current_summary"]
        r_sum = sim_res["recommended_summary"]

        s_cols[0].metric(
            "Expected Profit (Mean)",
            f"£{r_sum['mean']:.2f}",
            delta=f"£{sim_res['profit_delta_mean']:+.2f} vs Current"
        )
        s_cols[1].metric(
            "Downside Risk (VaR 5%)",
            f"£{r_sum['p5']:.2f}",
            delta=f"£{sim_res['profit_delta_p5']:+.2f} vs Current"
        )
        s_cols[2].metric(
            "Win Rate (Opt > Baseline)",
            f"{sim_res['opt_win_rate_pct']:.1f}%",
            help="Fraction of simulated market paths where recommended price generates higher profit than baseline."
        )
        s_cols[3].metric(
            "Target Profit Probability",
            f"{r_sum['prob_target']:.1f}%",
            delta=f"{r_sum['prob_target'] - c_sum['prob_target']:+.1f}% pts"
        )

        # Plotly Histogram overlay
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=sim_res["current_profit_samples"],
            name="Current Price Policy",
            opacity=0.6,
            marker_color="#94A3B8",
            nbinsx=40
        ))
        fig_dist.add_trace(go.Histogram(
            x=sim_res["recommended_profit_samples"],
            name="Recommended Price Policy",
            opacity=0.6,
            marker_color="#2563EB",
            nbinsx=40
        ))

        fig_dist.add_vline(x=r_sum["p5"], line_dash="dash", line_color="#DC2626", annotation_text="Opt 5% VaR")
        fig_dist.add_vline(x=c_sum["p5"], line_dash="dash", line_color="#64748B", annotation_text="Curr 5% VaR")

        fig_dist.update_layout(
            barmode="overlay",
            height=400,
            xaxis_title="Simulated Daily Gross Profit (£)",
            yaxis_title="Frequency",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Summary statistics table
        st.subheader("Distributional Quantile Metrics")
        comp_df = pd.DataFrame([
            {"Metric": "Mean (£)", "Current Baseline": c_sum["mean"], "Recommended Policy": r_sum["mean"], "Delta": round(r_sum["mean"] - c_sum["mean"], 2)},
            {"Metric": "Median (£)", "Current Baseline": c_sum["median"], "Recommended Policy": r_sum["median"], "Delta": round(r_sum["median"] - c_sum["median"], 2)},
            {"Metric": "Std Dev (£)", "Current Baseline": c_sum["std"], "Recommended Policy": r_sum["std"], "Delta": round(r_sum["std"] - c_sum["std"], 2)},
            {"Metric": "5th Percentile VaR (£)", "Current Baseline": c_sum["p5"], "Recommended Policy": r_sum["p5"], "Delta": round(r_sum["p5"] - c_sum["p5"], 2)},
            {"Metric": "Conditional VaR 5% (£)", "Current Baseline": c_sum["cvar_5"], "Recommended Policy": r_sum["cvar_5"], "Delta": round(r_sum["cvar_5"] - c_sum["cvar_5"], 2)},
            {"Metric": "Probability of Loss (%)", "Current Baseline": c_sum["prob_loss"], "Recommended Policy": r_sum["prob_loss"], "Delta": round(r_sum["prob_loss"] - c_sum["prob_loss"], 2)},
            {"Metric": f"Prob ≥ £{target_profit:.0f} Target (%)", "Current Baseline": c_sum["prob_target"], "Recommended Policy": r_sum["prob_target"], "Delta": round(r_sum["prob_target"] - c_sum["prob_target"], 2)},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

with tab_scenarios:
    st.subheader("Deterministic Stress Scenarios (A through F)")
    st.markdown("Evaluate how the pricing recommendation holds up under adverse operational shocks:")

    curr_p = float(selected_meta["latest_price"])
    rec_p = round(curr_p * 1.12, 2)
    unit_c = float(selected_meta["unit_cost"])

    scenarios = PricingService.run_scenarios(
        product_id=selected_pid,
        current_price=curr_p,
        recommended_price=rec_p,
        unit_cost=unit_c,
        min_price=round(curr_p * 0.6, 2),
        max_price=round(curr_p * 1.6, 2)
    )

    sc_df = pd.DataFrame(scenarios)
    st.dataframe(
        sc_df[["name", "description", "current_profit", "opt_profit", "profit_delta", "feasible", "notes"]],
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Visual Comparison Across Stress Scenarios")
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=sc_df["name"],
        y=sc_df["current_profit"],
        name="Current Baseline Profit (£)",
        marker_color="#94A3B8"
    ))
    fig_sc.add_trace(go.Bar(
        x=sc_df["name"],
        y=sc_df["opt_profit"],
        name="Recommended Dynamic Profit (£)",
        marker_color="#2563EB"
    ))
    fig_sc.update_layout(
        barmode="group",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_tickangle=-25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_sc, use_container_width=True)
