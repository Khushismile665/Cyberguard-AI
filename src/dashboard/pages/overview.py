"""
CyberGuard AI Dashboard - Executive Overview Page (Enterprise SOC)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.utils import (
    render_soc_header, render_kpi_card, get_dark_plotly_layout, style_soc_dataframe,
    render_workflow_footer, render_executive_summary_card, render_soc_health_widget, render_threat_intel_widget
)

def render_overview_section_divider(title: str):
    st.markdown(
        f"""
        <div style="margin: 26px 0 18px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.22); position: relative; padding-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.1em; text-transform: uppercase; background: rgba(15, 23, 42, 0.95); padding: 4px 14px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.35);">
                {title}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_overview_page(df: pd.DataFrame):
    render_soc_header("🛡️ Executive Security Overview", "Real-Time Cyber Threat Telemetry & Risk Intelligence Summary")

    total_events = len(df)
    total_anomalies = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    high_critical_count = int((df["risk_level"].isin(["High", "Critical"])).sum()) if "risk_level" in df.columns else 0
    unique_attacks = int(df["attack_type"].nunique() - (1 if "Normal" in df["attack_type"].values else 0)) if "attack_type" in df.columns else 0

    # 1. Executive KPI Cards (Positioned at the Top immediately below Header)
    render_overview_section_divider("📊 EXECUTIVE KEY PERFORMANCE INDICATORS")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        live_cnt = st.session_state.get('live_events_count', 0)
        render_kpi_card("Telemetry Events", f"{total_events:,}", f"↗️ +{live_cnt:,} Streamed", "positive" if live_cnt > 0 else "neutral", icon="📊")
    with col2:
        anom_ratio = (total_anomalies / total_events * 100) if total_events > 0 else 0
        render_kpi_card("Flagged Anomalies", f"{total_anomalies:,}", f"↘️ {anom_ratio:.1f}% Ratio", "negative", icon="⚠️")
    with col3:
        render_kpi_card("High/Critical Risks", f"{high_critical_count:,}", "🚨 SOC Action Required", "negative", icon="🚨")
    with col4:
        render_kpi_card("Attack Vectors", f"{unique_attacks}", "ACTIVE VECTORS", "positive", icon="⚔️")

    # 2. SOC Health & Threat Intelligence Widgets Side-by-Side
    render_overview_section_divider("⚡ SOC ENGINE HEALTH & GLOBAL THREAT INTELLIGENCE")
    hw_col1, hw_col2 = st.columns(2)
    with hw_col1:
        render_soc_health_widget()
    with hw_col2:
        render_threat_intel_widget(df)

    # 2. Analytics Charts (Positioned Directly Below KPIs)
    render_overview_section_divider("📈 THREAT VELOCITY TIMELINE & RISK LEVEL DISTRIBUTION")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("##### 📈 Daily Threat Velocity Activity (Normal vs Anomalies)")
        if "Timestamp" in df.columns:
            df_chart = df.copy()
            df_chart["dt"] = pd.to_datetime(df_chart["Timestamp"], errors="coerce")
            df_chart = df_chart.dropna(subset=["dt"])

            # Group daily timeline
            df_daily = df_chart.groupby([pd.Grouper(key="dt", freq="D"), "is_anomaly"]).size().unstack(fill_value=0).reset_index()

            # Ensure both 0 (Normal) and 1 (Anomaly) columns exist
            if 0 not in df_daily.columns:
                df_daily[0] = 0
            if 1 not in df_daily.columns:
                df_daily[1] = 0

            df_daily = df_daily.rename(columns={0: "Normal Logins", 1: "Anomalous Attacks"})

            fig = px.area(
                df_daily, x="dt", y=["Normal Logins", "Anomalous Attacks"],
                labels={"value": "Authentication Events", "dt": "Date", "variable": "Event Type"},
                color_discrete_map={"Normal Logins": "#10b981", "Anomalous Attacks": "#ef4444"}
            )
            fig.update_layout(**get_dark_plotly_layout(title_text="Daily Authentication Activity Timeline", height=500))
            st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown("##### 🍩 Risk Level Distribution Breakdown")
        if "risk_level" in df.columns:
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]

            fig_donut = px.pie(
                risk_counts, values="Count", names="Risk Level", hole=0.6,
                color="Risk Level",
                color_discrete_map={
                    "Low": "#10b981",
                    "Medium": "#f59e0b",
                    "High": "#f97316",
                    "Critical": "#ef4444"
                }
            )
            fig_donut.update_layout(**get_dark_plotly_layout(title_text="Risk Tier Proportion", height=500))
            st.plotly_chart(fig_donut, width="stretch")

    # 3. Risky User Leaderboard
    render_overview_section_divider("👤 TOP RISKY ENTITY LEADERBOARD (CRITICAL USER PROFILES)")
    if "User ID" in df.columns and "risk_score" in df.columns:
        user_leaderboard = df.groupby("User ID").agg(
            Max_Risk=("risk_score", "max"),
            Avg_Risk=("risk_score", "mean"),
            Anomalies=("is_anomaly", "sum") if "is_anomaly" in df.columns else ("risk_score", "count"),
            Primary_Attack=("attack_type", lambda x: x.mode()[0] if not x.empty else "Normal")
        ).reset_index().sort_values(by="Max_Risk", ascending=False).head(5)
        
        user_leaderboard = user_leaderboard.rename(columns={
            "Max_Risk": "Max Risk Score",
            "Avg_Risk": "Avg Risk Score",
            "Anomalies": "Total Anomalies",
            "Primary_Attack": "Primary Attack Vector"
        })
        st.dataframe(style_soc_dataframe(user_leaderboard), width="stretch")

    # 4. Live Alert Incident Feed Table
    render_overview_section_divider("🚨 RECENT SECURITY ALERT FEED (LIVE STREAMING QUEUE)")
    if "is_anomaly" in df.columns:
        anom_df = df[df["is_anomaly"] == 1].tail(15)[
            [c for c in ["Timestamp", "User ID", "Source IP", "Country", "attack_type", "risk_score", "risk_level", "natural_language_explanation"] if c in df.columns]
        ]
        st.dataframe(style_soc_dataframe(anom_df), width="stretch")

    render_workflow_footer(
        "📡 Live Monitoring",
        "📡 Launch Live Monitoring →",
        "What's happening right now? Inspect real-time 1-second login stream & active buffer."
    )
