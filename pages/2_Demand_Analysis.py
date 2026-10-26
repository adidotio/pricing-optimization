"""Page 2 - Demand Analysis: Exploratory Data Analysis & Elasticity Eligibility."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.services.demand_service import demand_service
from src.models.elasticity import PriceElasticityEstimator

st.set_page_config(page_title="Demand Analysis", page_icon="📈", layout="wide")

st.title("📈 Demand & Price Exploratory Analysis")
st.markdown("Explore historical sales dynamics, price variation, and empirical feasibility for econometric elasticity estimation.")

daily_df, _ = demand_service.get_processed_data()
products = demand_service.get_available_products()

if not products:
    st.error("No product transactions found in active dataset.")
    st.stop()

# Product selection
prod_options = {p["product_id"]: f"{p['product_id']} — {p['description']}" for p in products}
selected_pid = st.selectbox("Select Target Product", options=list(prod_options.keys()), format_func=lambda k: prod_options[k])

prod_df = daily_df[daily_df["product_id"] == selected_pid].sort_values("date").reset_index(drop=True)

# Eligibility & Variation Diagnostics
st.subheader("Price Variation & Elasticity Estimation Eligibility")

prices = prod_df["price"].to_numpy()
price_mean = float(np.mean(prices))
price_std = float(np.std(prices))
price_cv = (price_std / price_mean) if price_mean > 0 else 0.0
n_obs = len(prod_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Historical Observations", f"{n_obs} days")
c2.metric("Mean Selling Price", f"£{price_mean:.2f}")
c3.metric("Price Std Dev", f"£{price_std:.2f}")
c4.metric("Price Coeff of Variation (CV)", f"{price_cv:.2%}")

is_eligible = (n_obs >= PriceElasticityEstimator.MIN_RECORDS_REQUIRED) and (price_cv >= PriceElasticityEstimator.MIN_PRICE_CV)

if is_eligible:
    st.success(
        f"✅ **Eligible for Empirical Elasticity Estimation**: The product possesses adequate observations ({n_obs} >= {PriceElasticityEstimator.MIN_RECORDS_REQUIRED}) "
        f"and historical price variation (CV = {price_cv:.2%} >= {PriceElasticityEstimator.MIN_PRICE_CV:.2%})."
    )
else:
    st.warning(
        f"⚠️ **Insufficient Price Variation for Reliable Regression**: Coeff of Variation is {price_cv:.2%} (minimum required: {PriceElasticityEstimator.MIN_PRICE_CV:.2%}). "
        "Observational price changes are clustered at a single point. Empirical elasticity cannot be identified without user assumption."
    )

st.markdown("---")
# Time-Series Trends: Price and Demand
st.subheader("Historical Demand & Price Evolution")

fig_ts = go.Figure()

fig_ts.add_trace(go.Scatter(
    x=prod_df["date"],
    y=prod_df["quantity"],
    name="Units Sold (Daily)",
    mode="lines+markers",
    line=dict(color="#2563EB", width=2),
    yaxis="y1"
))

fig_ts.add_trace(go.Scatter(
    x=prod_df["date"],
    y=prod_df["price"],
    name="Average Selling Price (£)",
    mode="lines",
    line=dict(color="#DC2626", width=2, dash="dash"),
    yaxis="y2"
))

if "competitor_price" in prod_df.columns and prod_df["competitor_price"].notnull().any():
    fig_ts.add_trace(go.Scatter(
        x=prod_df["date"],
        y=prod_df["competitor_price"],
        name="Competitor Price (£)",
        mode="lines",
        line=dict(color="#10B981", width=1.5, dash="dot"),
        yaxis="y2"
    ))

fig_ts.update_layout(
    height=450,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(title="Date"),
    yaxis=dict(title="Quantity Demanded (Units)", title_font=dict(color="#2563EB")),
    yaxis2=dict(
        title="Price (£)",
        title_font=dict(color="#DC2626"),
        overlaying="y",
        side="right"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_ts, use_container_width=True)

# Distributions & Price Gaps
st.markdown("---")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("Price Distribution")
    fig_p = px.histogram(
        prod_df, 
        x="price", 
        nbins=20, 
        title="Observed Selling Price Frequencies",
        color_discrete_sequence=["#3B82F6"]
    )
    fig_p.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_p, use_container_width=True)

with col_d2:
    st.subheader("Demand Distribution")
    fig_q = px.histogram(
        prod_df, 
        x="quantity", 
        nbins=20, 
        title="Observed Quantity Demanded Frequencies",
        color_discrete_sequence=["#10B981"]
    )
    fig_q.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_q, use_container_width=True)

# Competitor & Inventory Section if available
if "competitor_price" in prod_df.columns and prod_df["competitor_price"].notnull().any():
    st.markdown("---")
    st.subheader("Competitor Price Gap Analysis")
    prod_df["price_gap"] = prod_df["price"] - prod_df["competitor_price"]
    fig_gap = px.line(
        prod_df, 
        x="date", 
        y="price_gap", 
        title="Price Gap vs Competitor (Our Price - Competitor Price)",
        color_discrete_sequence=["#8B5CF6"]
    )
    fig_gap.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_gap.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gap, use_container_width=True)
