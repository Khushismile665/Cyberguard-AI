"""
CyberGuard AI 2.0 - Security Operations Center (SOC) Web Dashboard (Module 8 & 9)

Main entry point for the enterprise multi-page Streamlit application. Features continuous real-time
1-second telemetry event streaming simulation with auto-updating live tables, charts, and risk scores.

Developed by: Khushi Singh
Institution: VIT Bhopal (2027)
"""

import sys
from pathlib import Path

# Add project root directory to sys.path at startup
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import warnings
import pandas as pd
import streamlit as st

# Filter warnings
warnings.filterwarnings('ignore')

from src.dashboard.utils import load_dashboard_dataset, apply_dark_soc_theme, render_soc_header
from src.dashboard.simulator import initialize_live_buffer, append_simulated_event
from src.dashboard.pages import (
    render_overview_page,
    render_live_stream_page,
    render_alerts_page,
    render_user_behaviour_page,
    render_attack_analytics_page,
    render_risk_scores_page,
    render_shap_explainability_page,
    render_model_performance_page,
    render_settings_page,
    render_copilot_page
)

def main():
    st.set_page_config(
        page_title="CyberGuard AI 2.0 - Enterprise SOC Platform",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply Enterprise Dark Cybersecurity SOC Theme
    apply_dark_soc_theme()

    # Initialize Live Buffer & Load Dataset
    initialize_live_buffer()

    # Simulation Active State (Default: False to ensure smooth interaction and fast loading)
    if "simulator_active" not in st.session_state:
        st.session_state["simulator_active"] = False

    # If simulator is active, append 1 new event every 1-second interval
    if st.session_state["simulator_active"]:
        df = append_simulated_event()
    else:
        df = st.session_state.get("full_dataset", load_dashboard_dataset())

    # Enterprise Sidebar Header Branding
    st.sidebar.markdown(
        """
        <div style="padding: 14px 16px; background: linear-gradient(135deg, rgba(0,242,254,0.14) 0%, rgba(139,92,246,0.14) 100%); border: 1px solid rgba(56,189,248,0.3); border-radius: 14px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem; filter: drop-shadow(0 0 10px rgba(0,242,254,0.5));">🛡️</span>
                <div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.22rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">CyberGuard AI</div>
                    <div style="font-family: 'Inter', sans-serif; font-size: 0.76rem; font-weight: 500; color: #38bdf8;">AI Threat Detection System</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Real-Time Attack Simulation Controls
    st.sidebar.markdown("### ⚡ REAL-TIME SIMULATOR")
    sim_col1, sim_col2 = st.sidebar.columns(2)
    with sim_col1:
        if st.button("▶️ Start 1s Stream", width="stretch"):
            st.session_state["simulator_active"] = True
            st.rerun()
    with sim_col2:
        if st.button("⏸️ Pause", width="stretch"):
            st.session_state["simulator_active"] = False
            st.rerun()

    sim_status = "🟢 SIMULATING (1 Event / Sec)" if st.session_state["simulator_active"] else "🔴 PAUSED"
    st.sidebar.markdown(
        f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: {'#10b981' if st.session_state['simulator_active'] else '#ef4444'}; margin-bottom: 14px;">
            Status: {sim_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dynamic Alert Count
    alert_cnt = int(df.get("is_anomaly", pd.Series([0])).sum()) if "is_anomaly" in df.columns else 61
    alerts_label = f"🚨 Alerts ({alert_cnt})"

    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "🏠 Executive Dashboard"

    options_list = [
        "🏠 Executive Dashboard",
        "📡 Live Monitoring",
        "🚨 Alerts & Incidents",
        "👤 User Behaviour",
        "⚔️ Threat Analytics",
        "🧠 AI Copilot",
        "📊 AI Intelligence",
        "⚙️ Settings"
    ]

    try:
        curr_index = options_list.index(st.session_state.get("selected_page", "🏠 Executive Dashboard"))
    except ValueError:
        curr_index = 0

    page = st.sidebar.radio(
        "NAVIGATION MENU",
        options=options_list,
        index=curr_index
    )
    st.session_state["selected_page"] = page

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56,189,248,0.22); padding: 14px 16px; border-radius: 12px; margin-top: 10px;">
            <div style="display: flex; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
                <span class="status-dot"></span> LIVE SOC ENGINE
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.70rem; color: #94a3b8; font-weight: 500; margin-bottom: 8px;">
                Real-Time Detection Active
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; color: #ffffff; font-weight: 700;">
                {len(df):,} Events
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #38bdf8; margin-top: 6px; font-weight: 600;">
                Version 2.0 • Khushi Singh (VIT Bhopal)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Route Page
    if page == "🏠 Executive Dashboard":
        render_overview_page(df)
    elif page == "📡 Live Monitoring":
        render_live_stream_page(df)
    elif page == "🚨 Alerts & Incidents":
        render_alerts_page(df)
    elif page == "👤 User Behaviour":
        render_user_behaviour_page(df)
    elif page == "⚔️ Threat Analytics":
        render_attack_analytics_page(df)
    elif page == "🧠 AI Copilot":
        render_copilot_page(df)
    elif page == "📊 AI Intelligence":
        render_soc_header("📊 AI Intelligence Hub", "Multi-Factor Risk Scoring, SHAP Explainability & Model Validation Metrics")
        tab1, tab2, tab3 = st.tabs(["🧮 Risk Scoring Engine", "🧠 SHAP Explainability", "📈 Model Performance Metrics"])
        with tab1:
            render_risk_scores_page(df, render_header=False)
        with tab2:
            render_shap_explainability_page(df, render_header=False)
        with tab3:
            render_model_performance_page(df, render_header=False)
    elif page == "⚙️ Settings":
        render_settings_page()

    # Continuous 1-second auto-rerun loop when simulator is active
    if st.session_state["simulator_active"]:
        time.sleep(1.0)
        st.rerun()

if __name__ == "__main__":
    main()
