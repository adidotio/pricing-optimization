"""Page 1 - Executive Dashboard: Pricing Recommendations & KPI Impacts."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.services.demand_service import demand_service
from src.services.pricing_service import PricingService
from src.data.repository import dataset_repo

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")

st.title("📊 Executive Pricing Dashboard")
st.markdown("High-level decision-support summary of pricing recommendations, profit impacts, and binding business constraints.")

# Get available catalog
products = demand_service.get_available_products()
if not products:
    st.error("No product transactions found in active dataset.")
    st.stop()

# Product selection
col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
with col_p1:
    prod_options = {p["product_id"]: f"{p['product_id']} — {p['description']} ({p['active_days']} days)" for p in products}
    selected_pid = st.selectbox("Select Target Product", options=list(prod_options.keys()), format_func=lambda k: prod_options[k])

selected_meta = next(p for p in products if p["product_id"] == selected_pid)

with col_p2:
    st.markdown("**Current Observed Price:**")
    curr_price = selected_meta["latest_price"]
    st.markdown(f"<h3 style='margin-top:-10px; color:#1E3A8A;'>£{curr_price:.2f} <span style='font-size:0.8rem;' class='badge-observed'>OBSERVED</span></h3>", unsafe_allow_html=True)

with col_p3:
    st.markdown("**Assumed Unit Cost:**")
    unit_cost = selected_meta["unit_cost"]
    st.markdown(f"<h3 style='margin-top:-10px; color:#475569;'>£{unit_cost:.2f} <span style='font-size:0.8rem;' class='badge-assumed'>ASSUMED</span></h3>", unsafe_allow_html=True)

# Run optimization with default standard business constraints
with st.spinner("Computing constrained pricing optimization..."):
    opt_res = PricingService.optimize_price(
        product_id=selected_pid,
        current_price=curr_price,
        unit_cost=unit_cost,
        min_price=round(curr_price * 0.70, 2),
        max_price=round(curr_price * 1.40, 2),
        max_price_change_pct=0.25,
        min_gross_margin=0.15,
        objective_type="profit",
        demand_model_name="log_linear",
        persist=False
    )

if not opt_res.get("is_feasible"):
    st.error(f"Optimization Infeasible: {opt_res.get('message')}")
    st.stop()

# Executive KPI Cards
st.markdown("---")
kpi_cols = st.columns(5)

rec_price = opt_res["recommended_price"]
price_delta_pct = opt_res["price_delta_pct"]
curr_profit = opt_res["current_profit"]
opt_profit = opt_res["expected_profit"]
profit_delta = opt_res["profit_delta"]
profit_delta_pct = opt_res["profit_delta_pct"]
curr_margin = opt_res["current_margin"]
opt_margin = opt_res["expected_margin"]

with kpi_cols[0]:
    st.metric(
        label="Recommended Price",
        value=f"£{rec_price:.2f}",
        delta=f"{price_delta_pct:+.1f}% vs Current"
    )
    st.caption("Constrained Optimum")

with kpi_cols[1]:
    st.metric(
        label="Expected Daily Profit",
        value=f"£{opt_profit:.2f}",
        delta=f"+£{profit_delta:.2f} ({profit_delta_pct:+.1f}%)"
    )
    st.caption("Predicted Objective")

with kpi_cols[2]:
    st.metric(
        label="Expected Daily Revenue",
        value=f"£{opt_res['expected_revenue']:.2f}",
        delta=f"£{opt_res['expected_revenue'] - opt_res['current_revenue']:+.2f}"
    )
    st.caption("Revenue Impact")

with kpi_cols[3]:
    st.metric(
        label="Unit Gross Margin",
        value=f"{opt_margin:.1f}%",
        delta=f"{opt_margin - curr_margin:+.1f}% pts"
    )
    st.caption(f"Min required: 15.0%")

with kpi_cols[4]:
    st.metric(
        label="Expected Daily Sales",
        value=f"{opt_res['expected_demand']:.1f} units",
        delta=f"{opt_res['expected_demand'] - opt_res['current_demand']:+.1f} units"
    )
    st.caption("Predicted Demand")

# Decision Rationale Banner
st.markdown("---")
col_exp1, col_exp2 = st.columns([3, 1])

with col_exp1:
    st.subheader("💡 Recommendation Explanation & Rationale")
    st.info(opt_res["explanation"])

with col_exp2:
    st.subheader("Active Constraints")
    if opt_res["binding_constraints"]:
        for bc in opt_res["binding_constraints"]:
            st.warning(f"🔒 **Binding:** {bc}")
    else:
        st.success("✅ **Interior Solution:** No boundary constraints active.")

# Visual Comparison Chart
st.markdown("---")
st.subheader("Decision Impact: Current Policy vs Optimized Policy")

fig = go.Figure()
metrics = ["Daily Profit (£)", "Daily Revenue (£)", "Expected Units Sold"]
current_vals = [curr_profit, opt_res["current_revenue"], opt_res["current_demand"]]
opt_vals = [opt_profit, opt_res["expected_revenue"], opt_res["expected_demand"]]

fig.add_trace(go.Bar(
    name="Current Baseline Price",
    x=metrics,
    y=current_vals,
    marker_color="#94A3B8"
))
fig.add_trace(go.Bar(
    name="Recommended Dynamic Price",
    x=metrics,
    y=opt_vals,
    marker_color="#2563EB"
))

fig.update_layout(
    barmode="group",
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Methodology Distinction Note
st.markdown("""
<div style='background-color:#F1F5F9; padding:12px; border-radius:6px; font-size:0.85rem; color:#475569;'>
<b>Evidence-Based Decision Support Notice:</b> The metrics above contrast historical observed pricing against model-based counterfactual forecasts. 
Realized commercial gains depend on actual market response and competitive stability. Stress-test this recommendation using the 
<b>Scenario Simulator</b> and <b>Sensitivity Analysis</b> modules before deployment.
</div>
""", unsafe_allow_html=True)
