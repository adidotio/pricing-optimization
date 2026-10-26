"""Page 3 - Price Elasticity Estimation & Econometric Diagnostics."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.services.demand_service import demand_service
from src.services.pricing_service import PricingService
from src.models.elasticity import PriceElasticityEstimator

st.set_page_config(page_title="Price Elasticity", page_icon="📉", layout="wide")

st.title("📉 Price Elasticity of Demand")
st.markdown(
    "Econometric estimation of price sensitivity using a log-log model: "
    r"$$\ln(Q) = \beta_0 + \beta_1 \ln(P) + \gamma X + \varepsilon$$"
)

daily_df, _ = demand_service.get_processed_data()
products = demand_service.get_available_products()

if not products:
    st.error("No product transactions found in active dataset.")
    st.stop()

# Product selection
col_sel, col_mode = st.columns([2, 1])
with col_sel:
    prod_options = {p["product_id"]: f"{p['product_id']} — {p['description']}" for p in products}
    selected_pid = st.selectbox("Select Target Product", options=list(prod_options.keys()), format_func=lambda k: prod_options[k])

with col_mode:
    elasticity_mode = st.radio(
        "Estimation Mode",
        ["Empirical Econometric Regression", "Manual Assumption Fallback"],
        help="Use empirical data if sufficient variation exists, or set a manual assumption."
    )

assumed_val = None
if elasticity_mode == "Manual Assumption Fallback":
    assumed_val = st.slider(
        "Assumed Price Elasticity",
        min_value=-5.0,
        max_value=-0.1,
        value=-1.45,
        step=0.05,
        help="Negative value indicates downward-sloping demand (e.g. -1.5 means 1% price hike leads to 1.5% drop in quantity)."
    )

# Run elasticity analysis
el_res = PricingService.get_elasticity_analysis(selected_pid, assumed_elasticity=assumed_val)

# Status and badges
st.markdown("---")
badge_class = "badge-assumed" if el_res["is_assumed"] else "badge-observed"
badge_text = "USER-ASSUMED ELASTICITY" if el_res["is_assumed"] else "EMPIRICAL ESTIMATE"

st.markdown(f"### Elasticity Result: <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)

if el_res.get("warning"):
    st.warning(f"⚠️ {el_res['warning']}")

# Econometric Metric Cards
m_cols = st.columns(5)
with m_cols[0]:
    st.metric("Price Elasticity (β₁)", f"{el_res['elasticity']:.3f}" if el_res['elasticity'] is not None else "N/A")
    st.caption("Demand % Δ / Price % Δ")

with m_cols[1]:
    st.metric("Standard Error", f"±{el_res['std_error']:.3f}" if el_res.get('std_error') is not None else "N/A")
    st.caption("Coefficient Uncertainty")

with m_cols[2]:
    ci_str = f"[{el_res['ci_lower']:.2f}, {el_res['ci_upper']:.2f}]" if el_res.get('ci_lower') is not None else "N/A"
    st.metric("95% Confidence Interval", ci_str)
    st.caption("Two-sided Student's t")

with m_cols[3]:
    st.metric("p-value (Significance)", f"{el_res.get('p_value'):.4f}" if el_res.get('p_value') is not None else "N/A")
    st.caption("H₀: β₁ = 0")

with m_cols[4]:
    st.metric("R² Fit", f"{el_res.get('r_squared'):.3f}" if el_res.get('r_squared') is not None else "N/A")
    st.caption("Variance Explained")

# Scatter and Fitted Demand Curve
prod_data = daily_df[daily_df["product_id"] == str(selected_pid)].copy()
prod_data = prod_data[(prod_data["quantity"] > 0) & (prod_data["price"] > 0)]

st.markdown("---")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Price vs Quantity (Log-Log Plane)")
    if len(prod_data) > 0 and el_res["elasticity"] is not None:
        p_vals = prod_data["price"].to_numpy()
        q_vals = prod_data["quantity"].to_numpy()
        log_p = np.log(p_vals)
        log_q = np.log(q_vals)

        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(
            x=log_p, y=log_q, mode="markers",
            name="Observed Transactions",
            marker=dict(color="#3B82F6", size=8, opacity=0.7)
        ))

        # Fitted regression line
        x_line = np.linspace(log_p.min(), log_p.max(), 50)
        intercept = np.mean(log_q) - el_res["elasticity"] * np.mean(log_p)
        y_line = intercept + el_res["elasticity"] * x_line

        fig_log.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            name=f"Log-Log Fit (Slope = {el_res['elasticity']:.3f})",
            line=dict(color="#DC2626", width=2.5)
        ))

        fig_log.update_layout(
            height=380,
            xaxis_title="ln(Price)",
            yaxis_title="ln(Quantity)",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_log, use_container_width=True)
    else:
        st.info("Insufficient scatter points to render log-log plane.")

with col_g2:
    st.subheader("Simulated Demand Curve Q(P)")
    if len(prod_data) > 0 and el_res["elasticity"] is not None:
        p_mean = prod_data["price"].mean()
        q_mean = prod_data["quantity"].mean()
        p_range = np.linspace(max(0.5, p_mean * 0.5), p_mean * 2.0, 50)
        # Q(P) = Q0 * (P / P0)^elasticity
        q_curve = q_mean * ((p_range / p_mean) ** el_res["elasticity"])

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=p_range, y=q_curve, mode="lines",
            name="Iso-Elastic Demand",
            line=dict(color="#10B981", width=3)
        ))
        fig_curve.add_trace(go.Scatter(
            x=[p_mean], y=[q_mean], mode="markers",
            name="Mean Observed Anchor",
            marker=dict(color="#0F172A", size=10, symbol="diamond")
        ))

        fig_curve.update_layout(
            height=380,
            xaxis_title="Price (£)",
            yaxis_title="Predicted Quantity Demanded",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

# Stability Diagnostic & Disclaimer
st.markdown("---")
st.subheader("Econometric Stability & Causal Identification Limits")

if el_res.get("stability_check"):
    sc = el_res["stability_check"]
    c_s1, c_s2, c_s3 = st.columns(3)
    c_s1.metric("First Half Period Elasticity", f"{sc['split_1_elasticity']:.3f}")
    c_s2.metric("Second Half Period Elasticity", f"{sc['split_2_elasticity']:.3f}")
    c_s3.metric("Parameter Stability Gap", f"{sc['stability_gap']:.3f}")

st.info(
    "**Observational Identification Warning:**\n"
    "Price elasticity estimated from observational retail data reflects historical correlations. "
    "Simultaneous shifts in consumer demand, seasonal stock clearance, and concurrent promotional marketing "
    "can induce omitted-variable endogeneity. Do not treat observational regression coefficients as causal without a randomized experiment."
)
