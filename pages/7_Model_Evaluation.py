"""Page 7 - Model Evaluation & Ablation Experiments."""
import streamlit as st
import plotly.express as px
import pandas as pd

from src.services.demand_service import demand_service
from src.services.evaluation_service import EvaluationService

st.set_page_config(page_title="Model Evaluation", page_icon="🧪", layout="wide")

st.title("🧪 Model Evaluation & Ablation Experiments")
st.markdown("Chronological validation of demand forecasting algorithms and component ablation studies measuring pricing decision value.")

products = demand_service.get_available_products()
if not products:
    st.error("No product transactions found in active dataset.")
    st.stop()

prod_options = {p["product_id"]: f"{p['product_id']} — {p['description']}" for p in products}
selected_pid = st.selectbox("Select Target Product", options=list(prod_options.keys()), format_func=lambda k: prod_options[k])
selected_meta = next(p for p in products if p["product_id"] == selected_pid)

tab_bench, tab_ablation, tab_limits = st.tabs(["Chronological Demand Benchmark", "Component Ablation Study", "Limitations & Failure Modes"])

with tab_bench:
    st.subheader("Chronological Train/Test Benchmark")
    st.caption("Evaluated strictly out-of-time (80% train / 20% test split) without lookahead leakage.")

    with st.spinner("Executing walk-forward model evaluation across baselines and ML regressors..."):
        bench_res = EvaluationService.get_model_benchmark(selected_pid)

    if "error" in bench_res:
        st.warning(bench_res["error"])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Observations", f"{bench_res['train_obs']} days")
        c2.metric("Testing Out-of-Time", f"{bench_res['test_obs']} days")
        c3.metric("Top Performing Algorithm", bench_res["best_model"])

        b_df = pd.DataFrame(bench_res["benchmark_table"])
        st.dataframe(
            b_df[["model_name", "model_type", "mae", "rmse", "mape_pct", "smape_pct", "r2"]],
            hide_index=True,
            use_container_width=True
        )

        fig_mae = px.bar(
            b_df, 
            x="model_name", 
            y="mae", 
            color="model_type",
            title="Mean Absolute Error (MAE) Comparison across Models (Lower is Better)",
            color_discrete_map={"baseline": "#94A3B8", "machine_learning": "#2563EB"}
        )
        fig_mae.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), xaxis_tickangle=-20)
        st.plotly_chart(fig_mae, use_container_width=True)

with tab_ablation:
    st.subheader("Ablation Study: Quantifying the Decision Value of ML Modeling")
    st.markdown(
        "Does a sophisticated ML demand model generate superior pricing decisions compared to simpler heuristics? "
        "We optimize prices under four different demand formulations and evaluate all policies in the same commercial environment:"
    )

    with st.spinner("Running component ablation analysis..."):
        ablation_list = EvaluationService.get_ablation_study(
            product_id=selected_pid,
            current_price=selected_meta["latest_price"],
            unit_cost=selected_meta["unit_cost"]
        )

    ab_df = pd.DataFrame(ablation_list)
    st.dataframe(
        ab_df[["component", "description", "recommended_price", "price_delta_pct", "expected_demand", "expected_profit", "expected_margin", "decision_benefit"]],
        hide_index=True,
        use_container_width=True
    )

    fig_ab = px.bar(
        ab_df,
        x="component",
        y="expected_profit",
        color="component",
        title="Realized Daily Profit across Decision Formulations (£)",
        color_discrete_sequence=["#2563EB", "#3B82F6", "#60A5FA", "#94A3B8"]
    )
    fig_ab.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig_ab, use_container_width=True)

with tab_limits:
    st.subheader("Documented Limitations & Failure Modes")
    st.markdown("""
    1. **Endogeneity & Promotional Confounders:**
       - Observational sales records reflect historical pricing actions that were often synchronized with marketing events, clearance sales, or seasonal holidays.
       - A fitted regression may correlate low prices with high volumes without establishing that lowering the price in a regular period will reproduce the same demand surge.
    
    2. **Sparse Price Variation (Support Deficiency):**
       - If a product only traded at £2.50, £2.55, and £2.60 historically, evaluating candidate prices at £4.00 or £1.20 requires deep extrapolation outside the empirical support domain.
       - The optimizer enforces strict candidate price bounds (`max_price_change_pct`) specifically to prevent catastrophic out-of-domain extrapolation.
    
    3. **Stockout Truncation (Uncensored Demand Bias):**
       - Realized sales data measures *sales*, not *unconstrained demand*. If a product was out of stock for 10 days, zero sales is recorded, artificially biasing estimated demand downwards.
    
    4. **Competitor Retaliation Ignored in Static Optimization:**
       - The static optimizer models optimal price assuming competitor prices remain fixed. In duopolistic markets, aggressive price cuts may spark price wars, eroding industry margins.
    """)
