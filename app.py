"""Pricing Optimization System - Main Streamlit Application."""
import streamlit as st
import pandas as pd
from src.data.repository import dataset_repo
from src.services.demand_service import demand_service
from src.utils.config import settings

st.set_page_config(
    page_title="Pricing Optimization System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .kpi-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .kpi-title { font-size: 0.85rem; color: #64748B; font-weight: 600; text-transform: uppercase; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #0F172A; }
    .kpi-delta-pos { color: #16A34A; font-weight: 600; }
    .kpi-delta-neg { color: #DC2626; font-weight: 600; }
    .badge-observed { background: #DBEAFE; color: #1E40AF; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-predicted { background: #FEF3C7; color: #92400E; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-simulated { background: #E0E7FF; color: #3730A3; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-assumed { background: #FEE2E2; color: #991B1B; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Sidebar Dataset Selection
st.sidebar.title("⚖️ Pricing System")
st.sidebar.markdown("**Decision-Support Navigation**")

dataset_options = {
    "uci_retail": "Real: UCI Online Retail II (UK Trans.)",
    "synthetic_scenario": "Synthetic: Kaggle Inventory & Competitor",
    "uploaded": "User Uploaded Dataset"
}

current_type = dataset_repo.active_dataset_type
selected_type = st.sidebar.selectbox(
    "Active Data Source",
    options=list(dataset_options.keys()),
    format_func=lambda k: dataset_options[k],
    index=list(dataset_options.keys()).index(current_type)
)

if selected_type != current_type:
    with st.spinner("Switching active dataset and invalidating caches..."):
        try:
            dataset_repo.load_active_dataset(selected_type)
            demand_service.invalidate_cache()
            st.sidebar.success(f"Switched to {dataset_options[selected_type]}")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading dataset: {e}")

# Retrieve active metadata and quality audit
raw_df, meta, audit = dataset_repo.get_active_dataset()

# Provenance Badge & Details in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Provenance")
if meta.get("is_synthetic"):
    st.sidebar.warning("⚠️ **SYNTHETIC DATASET**\nFor supply & competitor scenario testing only.")
else:
    st.sidebar.info("📊 **AUTHENTIC HISTORICAL DATA**\nReal transactions (UK Online Retail).")

st.sidebar.markdown(f"**Records:** {len(raw_df):,}")
st.sidebar.markdown(f"**License:** {meta.get('license', 'N/A')}")
if meta.get("url"):
    st.sidebar.markdown(f"[Source Repository]({meta['url']})")

# Main Page Overview
st.markdown("<div class='main-header'>Dynamic Pricing & Profit Optimization System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>AI-powered decision-support system for pricing recommendations under demand uncertainty, "
    "empirical elasticity estimation, and operational constraints.</div>",
    unsafe_allow_html=True
)

# Central Research Question Banner
st.info("""
**Central Research Question:**
*How does uncertainty in demand and price elasticity affect the optimal pricing decision, and how robust is the recommended price under changes in cost, competitor pricing, demand, and inventory constraints?*
""")

# Metadata Provenance Card
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Active Data Source Metadata")
    st.markdown(f"**Dataset Name:** {meta.get('name')}")
    st.markdown(f"**Provenance:** {meta.get('source')} | [Direct Link]({meta.get('url', '#')})")
    st.markdown(f"**License & Attribution:** {meta.get('license')} — {meta.get('authors', 'Public Domain')}")
    st.markdown(f"**Observation Scope:** {meta.get('period', 'N/A')}")
    st.markdown(f"**Critical Limitation Notice:** {meta.get('description')}")

with col2:
    st.subheader("Data Quality Status")
    st.metric("Total Records Ingested", f"{len(raw_df):,}")
    st.metric("Detected Columns", f"{len(raw_df.columns)}")
    if audit.get("warnings"):
        st.warning(f"{len(audit['warnings'])} Data-Quality Warnings Detected")
        with st.expander("Inspect Warnings"):
            for w in audit["warnings"]:
                st.write(f"- {w}")
    else:
        st.success("Clean Schema Validation Passed")

st.markdown("---")
st.subheader("System Architecture & Workflow")

st.markdown("""
```
Sales & Pricing Data 
  → Cleaning & Return Policy Audit 
  → Time-Series Feature Engineering (No Lookahead Leakage)
  → Chronological Demand Modeling (Baselines, Ridge, Log-Linear, RF, GBM)
  → Econometric Elasticity Estimation with Uncertainty & Diagnostic Tests
  → Constrained Mathematical Optimization (Grid Search & Numerical Refinement)
  → Monte Carlo & Scenario Stress-Testing (Downside Risk & Robustness)
  → Explainable Recommendation Rationale
```
""")

st.markdown("### Decision-Support Modules")
cols = st.columns(3)
with cols[0]:
    st.markdown("#### 1. Executive Dashboard\nHigh-level recommendations, KPI impacts, and binding constraint alerts.")
    st.markdown("#### 4. Price Optimizer\nFull interactive constrained optimization with margin and change controls.")
    st.markdown("#### 7. Model Evaluation\nChronological walk-forward validation and component ablation experiments.")
with cols[1]:
    st.markdown("#### 2. Demand Analysis\nTime-series trends, price variation eligibility, and competitor gaps.")
    st.markdown("#### 5. Scenario Simulator\nStochastic Monte Carlo simulation and stress tests across Scenarios A-F.")
    st.markdown("#### 8. Data Management\nCSV/Excel uploads, cleaning audits, and provenance records.")
with cols[2]:
    st.markdown("#### 3. Price Elasticity\nLog-log econometric estimation, standard errors, and stability diagnostics.")
    st.markdown("#### 6. Sensitivity Analysis\n1D and 2D parameter sweeps, profit curves, and robustness bounds.")
    st.markdown("#### 9. Administration\nRole authorization, model persistence, and database inspection.")

st.markdown("---")
st.caption("Pricing Optimization System v1.0.0 | Stanford MS&E Quantitative Decision Support Architecture")
