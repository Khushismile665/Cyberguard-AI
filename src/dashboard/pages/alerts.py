"""
CyberGuard AI Dashboard - Alerts & Threat Triage Page (Enterprise SOC)
"""

import streamlit as st
import pandas as pd
from src.dashboard.utils import render_soc_header, render_kpi_card, style_soc_dataframe, render_workflow_footer, render_incident_drawer

def render_alerts_page(df: pd.DataFrame):
    render_soc_header("🚨 Alerts & Threat Triage Center", "Security Incident Investigation & Remediation Queue")

    if "is_anomaly" not in df.columns and "risk_level" not in df.columns:
        st.info("No alert annotations found in current dataset.")
        return

    # Global search filter support
    search_filter = st.session_state.get("search_filter_term", "").strip().lower()
    alerts_df = df[(df.get("is_anomaly", 0) == 1) | (df.get("risk_level", "").isin(["High", "Critical"]))].copy()

    if search_filter:
        match_mask = alerts_df.astype(str).apply(lambda row: row.str.lower().str.contains(search_filter).any(), axis=1)
        alerts_df = alerts_df[match_mask]
        st.info(f"🔍 Filtered by Global Search Term: '{search_filter}' ({len(alerts_df)} matching incidents)")

    if alerts_df.empty:
        st.success("🟢 Excellent! Zero Critical Alerts Found Pending Triage.")
        return

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        render_kpi_card("Total Incident Alerts", f"{len(alerts_df):,}", "Pending Triage", "neutral")
    with col_a2:
        high_risk_cnt = int((alerts_df.get("risk_level", "") == "High").sum())
        render_kpi_card("High Severity Alerts", f"{high_risk_cnt:,}", "SOC Tier-1 Queue", "negative")
    with col_a3:
        crit_risk_cnt = int((alerts_df.get("risk_level", "") == "Critical").sum())
        render_kpi_card("Critical Severity Alerts", f"{crit_risk_cnt:,}", "Immediate Action", "negative")

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    st.subheader("🔍 Active Incident Triage Queue")

    # Add interactive triage status column if not present
    if "triage_status" not in alerts_df.columns:
        alerts_df["triage_status"] = "Open"

    display_cols = [c for c in ["Timestamp", "User ID", "Source IP", "Country", "attack_type", "risk_score", "risk_level", "natural_language_explanation", "triage_status"] if c in alerts_df.columns]
    
    st.dataframe(style_soc_dataframe(alerts_df[display_cols].head(200)), width="stretch")

    # 4. Interactive Incident Detail Drawer
    st.markdown("---")
    st.subheader("📋 Incident Detail Inspector Drawer")
    
    user_options = alerts_df["User ID"].unique().tolist() if "User ID" in alerts_df.columns else []
    if user_options:
        selected_inc_user = st.selectbox("Select Incident Record by Target User ID to Open Inspector Panel", options=user_options)
        inc_row = alerts_df[alerts_df["User ID"] == selected_inc_user].iloc[0]
        render_incident_drawer(inc_row)

    st.subheader("📥 Export Triage Incident Report")
    csv_data = alerts_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Incident Triage CSV Report",
        data=csv_data,
        file_name="cyberguard_incident_triage_report.csv",
        mime="text/csv"
    )

    render_workflow_footer(
        "👤 User Behaviour",
        "👤 Analyze User Risk Profiles →",
        "Investigate a user: Profile high-risk entities & anomalous login footprints."
    )
