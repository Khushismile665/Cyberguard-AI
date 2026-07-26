"""
CyberGuard AI Dashboard - Attack Analytics Page (Enterprise SOC)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.utils import render_soc_header, render_kpi_card, get_dark_plotly_layout, style_soc_dataframe, render_workflow_footer

def render_attack_section_divider(title: str):
    st.markdown(
        f"""
        <div style="margin: 24px 0 16px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.22); position: relative; padding-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.1em; text-transform: uppercase; background: rgba(15, 23, 42, 0.95); padding: 4px 14px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.35);">
                {title}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_attack_analytics_page(df: pd.DataFrame):
    render_soc_header("⚔️ Cyber Attack Vector Analytics", "Multi-Class Threat Classification & Vector Distribution Analysis")

    if "attack_type" not in df.columns:
        st.warning("Attack classification labels not present in current dataset.")
        return

    # Filter anomaly records
    attacks_only = df[df["attack_type"] != "Normal"].copy()

    render_attack_section_divider("📊 ATTACK INCIDENT SUMMARY & TOP VECTOR METRICS")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card("Total Attack Incidents", f"{len(attacks_only):,}", "Injected Scenarios", "negative", icon="⚔️")
    with col2:
        top_attack = attacks_only["attack_type"].mode()[0] if not attacks_only.empty else "None"
        render_kpi_card("Top Attack Vector", top_attack, "Highest Frequency", "neutral", icon="🎯")
    with col3:
        avg_attack_risk = attacks_only["risk_score"].mean() if "risk_score" in attacks_only.columns else 0.0
        render_kpi_card("Average Attack Risk", f"{avg_attack_risk:.1f} / 100", "Risk Score Scale", "negative" if avg_attack_risk > 50 else "neutral", icon="🚨")

    render_attack_section_divider("📈 VECTOR DISTRIBUTION & TARGETED ENDPOINTS")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("##### 📊 Attack Category Breakdown")
        atk_counts = attacks_only["attack_type"].value_counts().reset_index()
        atk_counts.columns = ["Attack Vector", "Incident Count"]

        fig_bar = px.bar(
            atk_counts, x="Attack Vector", y="Incident Count",
            color="Attack Vector",
            color_discrete_map={
                "Brute Force": "#ef4444",
                "Credential Stuffing": "#f97316",
                "Impossible Travel": "#f59e0b",
                "Device Spoofing": "#38bdf8",
                "Lateral Movement": "#8b5cf6",
                "Insider Threat": "#ec4899"
            }
        )
        fig_bar.update_layout(**get_dark_plotly_layout(title_text="Attack Vector Distribution", height=500))
        st.plotly_chart(fig_bar, width="stretch")

    with col_right:
        st.markdown("##### 🎯 Target Endpoint Exposure")
        if "Resource Accessed" in attacks_only.columns:
            res_counts = attacks_only.groupby(["Resource Accessed", "attack_type"]).size().reset_index(name="Count")
            fig_heat = px.bar(
                res_counts, x="Resource Accessed", y="Count", color="attack_type"
            )
            fig_heat.update_layout(**get_dark_plotly_layout(title_text="Target Endpoint Exposure", height=500))
            st.plotly_chart(fig_heat, width="stretch")

    render_attack_section_divider(f"📜 INJECTED ATTACK INCIDENT QUEUE ({len(attacks_only)} INCIDENTS)")
    display_cols = [c for c in ["Timestamp", "User ID", "Source IP", "Country", "attack_type", "risk_score", "risk_level", "natural_language_explanation"] if c in attacks_only.columns]
    st.dataframe(style_soc_dataframe(attacks_only[display_cols]), width="stretch")

    render_workflow_footer(
        "🧠 AI Copilot",
        "🧠 Consult AI Copilot →",
        "Ask questions & get recommendations: Query the AI SOC Assistant for playbooks."
    )
