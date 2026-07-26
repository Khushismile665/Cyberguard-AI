"""
CyberGuard AI Dashboard - Risk Scores Engine Page (Enterprise SOC)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.utils import render_soc_header, render_kpi_card, get_dark_plotly_layout, style_soc_dataframe

def render_risk_scores_page(df: pd.DataFrame, render_header: bool = True):
    if render_header:
        render_soc_header("🧮 Explainable Risk Scoring Engine", "Multi-Factor Risk Assessment (0-100 Continuous Scale)")

    if "risk_score" not in df.columns:
        st.warning("Risk scores not calculated in current dataset.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Mean System Risk Score", f"{df['risk_score'].mean():.2f}", "Continuous Scale", "neutral")
    with col2:
        render_kpi_card("Median Risk Score", f"{df['risk_score'].median():.2f}", "System Median", "neutral")
    with col3:
        render_kpi_card("Max Observed Risk", f"{df['risk_score'].max():.1f}", "Peak Risk Event", "negative")
    with col4:
        render_kpi_card("High Risk Threshold", "60.0 Score", "Trigger Boundary", "neutral")

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("##### 📉 Frequency Distribution of Calculated Risk Scores")
        fig_hist = px.histogram(
            df, x="risk_score", nbins=50,
            color="risk_level" if "risk_level" in df.columns else None,
            color_discrete_map={
                "Low": "#10b981",
                "Medium": "#f59e0b",
                "High": "#f97316",
                "Critical": "#ef4444"
            }
        )
        fig_hist.update_layout(**get_dark_plotly_layout(title_text="Risk Score Frequency Spectrum", height=450))
        st.plotly_chart(fig_hist, width="stretch")

    with col_right:
        st.markdown("##### 👑 High Risk User Leaderboard")
        if "User ID" in df.columns:
            user_risk_df = df.groupby("User ID")["risk_score"].agg(["max", "mean", "count"]).reset_index()
            user_risk_df.columns = ["User ID", "Max Risk", "Mean Risk", "Total Events"]
            user_risk_df = user_risk_df.sort_values(by="Max Risk", ascending=False).head(10)
            st.dataframe(style_soc_dataframe(user_risk_df), width="stretch")
