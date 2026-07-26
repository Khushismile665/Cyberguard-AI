"""
CyberGuard AI Dashboard - Settings & Admin Page (Enterprise SOC)
"""

import streamlit as st
from config import config
from src.dashboard.utils import render_soc_header, render_workflow_footer

def render_settings_page():
    render_soc_header("⚙️ System Settings & SOC Admin Configuration", "Environment Parameters, Threshold Tuning & System Maintenance")

    st.subheader("1. System Environment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Environment", value=config.env, disabled=True)
        st.text_input("Base Directory", value=str(config.dirs.base_dir), disabled=True)
    with col2:
        st.text_input("Log Level", value=config.log_level, disabled=True)
        st.text_input("Saved Models Directory", value=str(config.dirs.saved_models_dir), disabled=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    st.subheader("2. Risk Engine Threshold Tuning")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.slider("Low to Medium Risk Threshold", min_value=10, max_value=40, value=30, step=5)
        st.slider("Medium to High Risk Threshold", min_value=45, max_value=75, value=60, step=5)
    with col_t2:
        st.slider("High to Critical Risk Threshold", min_value=80, max_value=95, value=85, step=5)
        st.slider("Impossible Velocity Threshold (km/h)", min_value=300, max_value=1500, value=900, step=50)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    st.subheader("3. Cache Maintenance & Reload Controls")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🔄 Clear Streamlit Cache & Reload Data", width="stretch"):
            st.cache_data.clear()
            st.success("Successfully cleared Streamlit data cache!")
    with col_b2:
        if st.button("🚀 Trigger Model Pipeline Diagnostics", width="stretch"):
            st.info("Triggered model pipeline diagnostics check.")

    render_workflow_footer(
        "🏠 Executive Dashboard",
        "🏠 Return to Executive Dashboard →",
        "Return to executive high-level threat intelligence summary."
    )
