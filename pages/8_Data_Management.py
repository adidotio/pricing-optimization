"""Page 8 - Data Management: File Ingestion, Validation, and Provenance."""
import streamlit as st
import pandas as pd
from pathlib import Path

from src.data.repository import dataset_repo
from src.data.validator import DataValidator
from src.services.demand_service import demand_service
from src.utils.config import RAW_DATA_DIR

st.set_page_config(page_title="Data Management", page_icon="🗄️", layout="wide")

st.title("🗄️ Data Management & Preprocessing Audits")
st.markdown("Upload custom CSV/Excel datasets, inspect validation audits, and manage data provenance.")

tab_active, tab_upload, tab_provenance = st.tabs(["Active Dataset & Audit", "Upload New Dataset", "Provenance & Licensing"])

with tab_active:
    raw_df, meta, audit = dataset_repo.get_active_dataset()
    
    st.subheader(f"Active Data Source: {meta.get('name')}")
    st.markdown(f"**Loaded File:** `{meta.get('loaded_file')}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(raw_df):,}")
    c2.metric("Columns", f"{len(raw_df.columns)}")
    c3.metric("Duplicate Rows", f"{audit.get('duplicate_rows', 0)}")
    c4.metric("Non-Positive Prices", f"{audit.get('stats', {}).get('non_positive_prices', 0)}")

    st.markdown("### Missing Value Summary")
    missing_s = pd.Series(audit.get("missing_values", {}))
    if missing_s.sum() > 0:
        st.dataframe(missing_s[missing_s > 0].to_frame("Missing Count"), use_container_width=True)
    else:
        st.success("No missing values detected across active dataset columns.")

    if audit.get("warnings"):
        st.warning(f"**Data Quality Warnings ({len(audit['warnings'])}):**")
        for w in audit["warnings"]:
            st.write(f"- {w}")

    st.markdown("### Raw Data Sample Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    if st.button("🔄 Invalidate Cache & Refresh Pipeline"):
        demand_service.invalidate_cache()
        st.success("Cache cleared and pipeline refreshed.")
        st.rerun()

with tab_upload:
    st.subheader("Upload Custom Sales or Pricing Data")
    st.markdown("Upload a CSV or Excel (`.xlsx`) file containing either transaction-level sales or daily scenario observations.")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                upload_df = pd.read_csv(uploaded_file)
            else:
                upload_df = pd.read_excel(uploaded_file)

            st.write(f"**File preview ({len(upload_df)} rows, {len(upload_df.columns)} columns):**")
            st.dataframe(upload_df.head(5))

            # Validate against known schemas
            uci_valid, uci_errs = DataValidator.validate_uci_schema(upload_df)
            syn_valid, syn_errs = DataValidator.validate_synthetic_schema(upload_df)

            if uci_valid:
                st.success("✅ Adheres to UCI Online Retail transaction schema.")
            elif syn_valid:
                st.success("✅ Adheres to Synthetic Scenario schema.")
            else:
                st.warning(f"Custom schema detected. Please ensure Date, Product ID, Price, and Quantity exist.\nMissing UCI: {uci_errs}\nMissing Synthetic: {syn_errs}")

            if st.button("📥 Save & Activate Uploaded Dataset", type="primary"):
                save_path = RAW_DATA_DIR / f"uploaded_{uploaded_file.name}"
                if uploaded_file.name.endswith(".csv"):
                    upload_df.to_csv(save_path, index=False)
                else:
                    upload_df.to_excel(save_path, index=False)

                dataset_repo.load_active_dataset(source_type="uploaded", custom_path=str(save_path))
                demand_service.invalidate_cache()
                st.success(f"Successfully activated dataset: {uploaded_file.name}")
                st.rerun()

        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

with tab_provenance:
    st.subheader("Dataset Provenance & Legal Licensing")
    st.markdown("""
    ### 1. Primary Real-World Source: UCI Online Retail II
    - **Repository:** UCI Machine Learning Repository
    - **URL:** [archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
    - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
    - **Citation:** Chen, D., Sain, S.L., and Guo, K. (2012). Data mining for the online retail industry.
    - **Observed Period:** 01/12/2009 – 09/12/2011.
    - **Limitation Notice:** The raw data contains actual selling prices and quantities, but **does not** contain unit costs, wholesale procurement fees, inventory stockouts, or randomized pricing experiments.
    
    ### 2. Supplemental Scenario Source: Kaggle Inventory & Demand
    - **Repository:** Kaggle Datasets
    - **License:** Apache 2.0
    - **Notice:** This dataset is explicitly synthetic and used exclusively for demonstrating supply constraints, weather, and competitor-price bounds.
    """)
