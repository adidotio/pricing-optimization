"""Page 9 - Administration: User Management, Optimization Audit Logs, and System Health."""
import streamlit as st
import pandas as pd
from pathlib import Path

from src.database.connection import SessionLocal
from src.database.models import User, OptimizationRecord
from src.auth.authentication import hash_password
from src.utils.config import settings

st.set_page_config(page_title="Administration", page_icon="⚙️", layout="wide")

st.title("⚙️ System Administration & Governance")
st.markdown("User authorization management, optimization run audit logs, and persistence diagnostics.")

tab_users, tab_audit, tab_env = st.tabs(["User Management", "Optimization Run History", "Environment & Logs"])

with tab_users:
    st.subheader("Authorized System Users")
    db = SessionLocal()
    try:
        users = db.query(User).all()
        user_data = [
            {"ID": u.id, "Username": u.username, "Email": u.email, "Role": u.role, "Active": u.is_active, "Created At": u.created_at}
            for u in users
        ]
        st.dataframe(pd.DataFrame(user_data), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Create New User")
        with st.form("new_user_form"):
            new_u = st.text_input("Username")
            new_e = st.text_input("Email")
            new_p = st.text_input("Password", type="password")
            new_r = st.selectbox("Role", ["analyst", "viewer", "admin"])
            submitted = st.form_submit_button("Register User")

            if submitted:
                if not new_u or not new_p or not new_e:
                    st.error("Please fill in all fields.")
                else:
                    existing = db.query(User).filter(User.username == new_u).first()
                    if existing:
                        st.error(f"Username '{new_u}' already exists.")
                    else:
                        user_obj = User(
                            username=new_u,
                            email=new_e,
                            hashed_password=hash_password(new_p),
                            role=new_r,
                            is_active=True
                        )
                        db.add(user_obj)
                        db.commit()
                        st.success(f"User '{new_u}' created with role '{new_r}'.")
                        st.rerun()
    finally:
        db.close()

with tab_audit:
    st.subheader("Historical Optimization Audit Records")
    db = SessionLocal()
    try:
        records = db.query(OptimizationRecord).order_by(OptimizationRecord.created_at.desc()).limit(50).all()
        if records:
            rec_data = [
                {
                    "ID": r.id,
                    "Product ID": r.product_id,
                    "Current Price": f"£{r.current_price:.2f}",
                    "Recommended Price": f"£{r.recommended_price:.2f}",
                    "Unit Cost": f"£{r.unit_cost:.2f}",
                    "Expected Profit": f"£{r.expected_profit_opt:.2f}" if r.expected_profit_opt else "N/A",
                    "Binding Constraints": r.binding_constraints,
                    "Timestamp": r.created_at
                }
                for r in records
            ]
            st.dataframe(pd.DataFrame(rec_data), hide_index=True, use_container_width=True)
        else:
            st.info("No optimization runs recorded yet. Use Page 4 (Price Optimizer) to generate recommendations.")
    finally:
        db.close()

with tab_env:
    st.subheader("System Configuration & Infrastructure")
    st.markdown(f"**Application Name:** `{settings.app_name}` (v{settings.app_version})")
    st.markdown(f"**Database Connection:** `{settings.db_url}`")
    st.markdown(f"**JWT Token Validity:** `{settings.access_token_expire_minutes}` minutes")
    st.markdown(f"**API Host & Port:** `{settings.api_host}:{settings.api_port}`")

    log_path = Path(__file__).resolve().parent.parent / "artifacts" / "pricing_system.log"
    st.markdown(f"**System Log File:** `{log_path}`")
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with st.expander("View Recent System Log Entries (Last 30 lines)"):
            st.text("".join(lines[-30:]))
