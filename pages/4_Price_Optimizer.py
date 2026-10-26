"""Page 4 - Price Optimizer: Constrained Revenue & Profit Optimization."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.services.demand_service import demand_service
from src.services.pricing_service import PricingService

st.set_page_config(page_title="Price Optimizer", page_icon="🎯", layout="wide")

st.title("🎯 Constrained Price Optimization Engine")
st.markdown("Solve mathematical optimization for profit or revenue under strict operational and business constraints.")

products = demand_service.get_available_products()
if not products:
    st.error("No product transactions found in active dataset.")
    st.stop()

# Form & Inputs
with st.sidebar:
    st.header("⚙️ Optimization Parameters")
    
    prod_options = {p["product_id"]: f"{p['product_id']} — {p['description']}" for p in products}
    selected_pid = st.selectbox("Product", options=list(prod_options.keys()), format_func=lambda k: prod_options[k])
    selected_meta = next(p for p in products if p["product_id"] == selected_pid)

    base_p = selected_meta["latest_price"]
    base_c = selected_meta["unit_cost"]

    st.subheader("Price & Cost Structure")
    current_price = st.number_input("Current Selling Price (£)", min_value=0.1, value=float(base_p), step=0.1)
    
    st.caption("⚠️ **Notice:** Never infer unit cost from selling price. Must be entered explicitly.")
    unit_cost = st.number_input("Unit Cost of Goods (£)", min_value=0.0, value=float(base_c), step=0.1)

    st.subheader("Business Constraints")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        min_price = st.number_input("Min Price (£)", min_value=0.1, value=round(current_price * 0.6, 2), step=0.1)
    with c_p2:
        max_price = st.number_input("Max Price (£)", min_value=0.1, value=round(current_price * 1.6, 2), step=0.1)

    max_change_pct = st.slider("Max Price Change Limit (±%)", min_value=5, max_value=60, value=25, step=5) / 100.0
    min_margin = st.slider("Min Gross Margin (%)", min_value=0, max_value=60, value=15, step=5) / 100.0

    st.subheader("Supply & Competitor")
    use_capacity = st.checkbox("Enable Inventory Capacity Limit", value=selected_meta["has_inventory"])
    capacity = st.number_input("Max Inventory / Supply Capacity (Units)", min_value=1.0, value=150.0, step=10.0) if use_capacity else None

    use_comp = st.checkbox("Enable Competitor Price Ceiling", value=selected_meta["has_competitor"])
    comp_price = st.number_input("Competitor Price (£)", min_value=0.1, value=round(current_price * 1.05, 2), step=0.1) if use_comp else None

    st.subheader("Model & Objective")
    objective_type = st.radio("Optimization Objective", ["profit", "revenue"], format_func=lambda x: "Maximize Gross Profit" if x == "profit" else "Maximize Revenue")
    model_name = st.selectbox(
        "Demand Model",
        ["log_linear", "linear_ridge", "random_forest", "gradient_boosting"],
        format_func=lambda x: {
            "log_linear": "Log-Linear Econometric Model",
            "linear_ridge": "Linear Ridge Regression",
            "random_forest": "Random Forest Regressor",
            "gradient_boosting": "Gradient Boosting Regressor"
        }[x]
    )

# Execute Optimization
opt_res = PricingService.optimize_price(
    product_id=selected_pid,
    current_price=current_price,
    unit_cost=unit_cost,
    min_price=min_price,
    max_price=max_price,
    max_price_change_pct=max_change_pct,
    min_gross_margin=min_margin,
    inventory_capacity=capacity,
    competitor_price=comp_price,
    objective_type=objective_type,
    demand_model_name=model_name,
    persist=True
)

if not opt_res.get("is_feasible"):
    st.error(f"❌ **Infeasible Problem Formulation:**\n{opt_res.get('message')}")
    st.stop()

# Primary Recommendation Card
rec_price = opt_res["recommended_price"]
price_delta = opt_res["price_delta"]
price_delta_pct = opt_res["price_delta_pct"]

st.markdown(f"### Optimal Recommendation: **£{rec_price:.2f}** ({price_delta_pct:+.1f}%)")

r_cols = st.columns(4)
r_cols[0].metric("Current Price", f"£{current_price:.2f}")
r_cols[1].metric("Recommended Price", f"£{rec_price:.2f}", delta=f"£{price_delta:+.2f} ({price_delta_pct:+.1f}%)")
r_cols[2].metric("Expected Profit", f"£{opt_res['expected_profit']:.2f}", delta=f"£{opt_res['profit_delta']:+.2f} ({opt_res['profit_delta_pct']:+.1f}%)")
r_cols[3].metric("Unit Margin", f"{opt_res['expected_margin']:.1f}%", delta=f"{opt_res['expected_margin'] - opt_res['current_margin']:+.1f}% pts")

st.markdown("---")
# Explanation and Active Constraints
col_e1, col_e2 = st.columns([3, 1])

with col_e1:
    st.subheader("Decision Explanation")
    st.info(opt_res["explanation"])

with col_e2:
    st.subheader("Binding Constraints")
    if opt_res["binding_constraints"]:
        for bc in opt_res["binding_constraints"]:
            st.warning(f"🔒 {bc}")
    else:
        st.success("✅ Free interior optimum")

# Price-Objective Curve Visualization
st.markdown("---")
st.subheader("Objective Curve Across Candidate Prices")

grid_df = pd.DataFrame(opt_res["grid_results"])
fig_curve = go.Figure()

# Plot Profit
fig_curve.add_trace(go.Scatter(
    x=grid_df["price"],
    y=grid_df["profit"],
    name="Expected Gross Profit (£)",
    mode="lines",
    line=dict(color="#2563EB", width=2.5)
))

# Plot Revenue
fig_curve.add_trace(go.Scatter(
    x=grid_df["price"],
    y=grid_df["revenue"],
    name="Expected Revenue (£)",
    mode="lines",
    line=dict(color="#10B981", width=2, dash="dash")
))

# Highlight Recommended Price
fig_curve.add_trace(go.Scatter(
    x=[rec_price],
    y=[opt_res["expected_profit"] if objective_type == "profit" else opt_res["expected_revenue"]],
    mode="markers",
    name=f"Optimal Price (£{rec_price:.2f})",
    marker=dict(color="#DC2626", size=12, symbol="star")
))

# Highlight Current Price
fig_curve.add_trace(go.Scatter(
    x=[current_price],
    y=[opt_res["current_profit"] if objective_type == "profit" else opt_res["current_revenue"]],
    mode="markers",
    name=f"Current Price (£{current_price:.2f})",
    marker=dict(color="#64748B", size=10, symbol="circle")
))

fig_curve.update_layout(
    height=420,
    xaxis_title="Candidate Price (£)",
    yaxis_title="Expected Commercial Value (£)",
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_curve, use_container_width=True)

# Grid Candidate Table
with st.expander("Inspect Candidate Price Grid Evaluation (Sample Rows)"):
    st.dataframe(grid_df[["price", "expected_demand", "feasible_sales", "revenue", "profit", "margin", "is_feasible"]].round(2).iloc[::20])
