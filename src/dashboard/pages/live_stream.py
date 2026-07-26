"""
CyberGuard AI Dashboard - Live Login Stream Page (Enterprise SOC)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.utils import render_soc_header, render_kpi_card, get_dark_plotly_layout, style_soc_dataframe, render_workflow_footer

def render_live_stream_page(df: pd.DataFrame):
    render_soc_header("📡 Live Authentication Stream Console", "Real-Time Telemetry Event Streaming & Attack Simulation Feed")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        risk_filter = st.multiselect("Filter by Risk Level", options=["Low", "Medium", "High", "Critical"], default=["Low", "Medium", "High", "Critical"])
    with col_f2:
        attack_options = list(df["attack_type"].unique()) if "attack_type" in df.columns else []
        attack_filter = st.multiselect("Filter by Attack Category", options=attack_options, default=attack_options)
    with col_f3:
        records_to_show = st.slider("Stream Window Size", min_value=10, max_value=200, value=50, step=10)

    filtered_df = df.copy()
    if "risk_level" in filtered_df.columns and risk_filter:
        filtered_df = filtered_df[filtered_df["risk_level"].isin(risk_filter)]
    if "attack_type" in filtered_df.columns and attack_filter:
        filtered_df = filtered_df[filtered_df["attack_type"].isin(attack_filter)]

    st.subheader(f"⚡ Live Telemetry Feed (Displaying Latest {min(records_to_show, len(filtered_df)):,} Logins)")
    
    display_cols = [c for c in ["Timestamp", "User ID", "Device ID", "Source IP", "Country", "City", "Authentication Method", "Resource Accessed", "risk_score", "risk_level", "attack_type", "natural_language_explanation"] if c in filtered_df.columns]
    
    # Reverse so latest simulated events appear at the top!
    latest_events = filtered_df.tail(records_to_show).iloc[::-1]
    st.dataframe(style_soc_dataframe(latest_events[display_cols]), width="stretch")

    if "risk_score" in filtered_df.columns and "Timestamp" in filtered_df.columns:
        st.subheader("📊 Real-Time Stream Risk Score Trajectory")
        sample_stream = filtered_df.tail(records_to_show).copy()
        fig = px.scatter(
            sample_stream, x="Timestamp", y="risk_score", color="risk_level",
            size="risk_score",
            color_discrete_map={
                "Low": "#10b981",
                "Medium": "#f59e0b",
                "High": "#f97316",
                "Critical": "#ef4444"
            },
            title="Risk Score Trajectory Across Latest Streamed Authentication Events"
        )
        fig.update_layout(**get_dark_plotly_layout())
        st.plotly_chart(fig, width="stretch")

    render_workflow_footer(
        "🚨 Alerts & Incidents",
        "🚨 Investigate Triggered Alerts →",
        "Which events need attention? Triage flagged threat anomalies & risk levels."
    )
