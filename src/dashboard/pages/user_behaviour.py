"""
CyberGuard AI Dashboard - User Behaviour Analytics (Enterprise SOC)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.utils import render_soc_header, render_kpi_card, get_dark_plotly_layout, style_soc_dataframe, render_workflow_footer

CITY_COORDS = {
    "New York": (40.7128, -74.0060), "Los Angeles": (34.0522, -118.2437), "Chicago": (41.8781, -87.6298),
    "San Francisco": (37.7749, -122.4194), "Austin": (30.2672, -97.7431), "Seattle": (47.6062, -122.3321),
    "London": (51.5074, -0.1278), "Manchester": (53.4808, -2.2426), "Edinburgh": (55.9533, -3.1883),
    "Berlin": (52.5200, 13.4050), "Frankfurt": (50.1109, 8.6821), "Munich": (48.1351, 11.5820),
    "Bangalore": (12.9716, 77.5946), "Mumbai": (19.0760, 72.8777), "Delhi": (28.6139, 77.2090),
    "Tokyo": (35.6762, 139.6503), "Osaka": (34.6937, 135.5023), "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631), "Toronto": (43.6532, -79.3832), "Moscow": (55.7558, 37.6173),
    "Beijing": (39.9042, 116.4074), "Sao Paulo": (-23.5505, -46.6333), "Pyongyang": (39.0392, 125.7625)
}

def render_section_divider(title: str):
    st.markdown(
        f"""
        <div style="margin: 22px 0 16px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.22); position: relative; padding-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.1em; text-transform: uppercase; background: rgba(15, 23, 42, 0.95); padding: 4px 14px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.35);">
                {title}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_user_behaviour_page(df: pd.DataFrame):
    render_soc_header("👤 User Behaviour Analytics (UBA)", "Individual Entity Profiling & Historical Trajectory Investigation")

    if "User ID" not in df.columns:
        st.warning("User ID column not present in dataset.")
        return

    users = sorted(df["User ID"].unique())
    
    # Global search index selection
    search_user = st.session_state.get("selected_user_search", "")
    default_idx = 0
    if search_user and search_user in users:
        default_idx = users.index(search_user)

    selected_user = st.selectbox("🎯 Select Target User Profile to Inspect", options=users, index=default_idx)

    user_df = df[df["User ID"] == selected_user].copy()

    # Section 1: Summary Metrics
    render_section_divider(f"📊 SUMMARY METRICS & BASELINE PROFILE FOR {selected_user}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Auth Events", f"{len(user_df):,}", "30-Day Window", "neutral")
    with col2:
        user_anom_cnt = int(user_df.get("is_anomaly", pd.Series([0])).sum())
        render_kpi_card("Anomalies Flagged", f"{user_anom_cnt:,}", f"{(user_anom_cnt/len(user_df)*100):.1f}% Ratio", "negative" if user_anom_cnt > 0 else "positive")
    with col3:
        user_max_risk = float(user_df.get("risk_score", pd.Series([0.0])).max())
        render_kpi_card("Peak Risk Score", f"{user_max_risk:.1f} / 100", "Risk Scale", "negative" if user_max_risk > 60 else "positive")
    with col4:
        primary_country = user_df["Country"].mode()[0] if "Country" in user_df.columns and not user_df["Country"].empty else "N/A"
        render_kpi_card("Primary Home Region", primary_country, "Location Baseline", "neutral")

    # Section 2: Behavioral Timeline & Device Breakdown
    render_section_divider("⏱️ BEHAVIORAL TIMELINE & DEVICE INTENT ANALYTICS")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("##### 📈 Activity & Risk Trajectory Timeline")
        if "Timestamp" in user_df.columns:
            user_df["dt"] = pd.to_datetime(user_df["Timestamp"])
            fig = px.scatter(
                user_df, x="dt", y="Session Duration" if "Session Duration" in user_df.columns else "risk_score",
                color="attack_type" if "attack_type" in user_df.columns else "Login Success",
                title=f"Authentication History ({selected_user})",
                color_discrete_map={
                    "Normal": "#10b981",
                    "Brute Force": "#ef4444",
                    "Credential Stuffing": "#f97316",
                    "Impossible Travel": "#f59e0b",
                    "Device Spoofing": "#38bdf8",
                    "Lateral Movement": "#8b5cf6",
                    "Insider Threat": "#ec4899"
                }
            )
            fig.update_layout(**get_dark_plotly_layout())
            st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown("##### 📱 Registered Devices & Access Frequency")
        if "Device ID" in user_df.columns:
            dev_counts = user_df["Device ID"].value_counts().reset_index()
            dev_counts.columns = ["Device ID", "Event Count"]
            fig_dev = px.bar(dev_counts, x="Device ID", y="Event Count", color="Device ID", title=f"Device Fingerprints ({selected_user})")
            fig_dev.update_layout(**get_dark_plotly_layout())
            st.plotly_chart(fig_dev, width="stretch")

    # Section 3: Geographic Threat Map
    render_section_divider("🌍 GEOGRAPHIC THREAT MAP & LOG TRAJECTORY")

    if "City" in user_df.columns:
        user_df["lat"] = user_df["City"].map(lambda c: CITY_COORDS.get(c, (0, 0))[0])
        user_df["lon"] = user_df["City"].map(lambda c: CITY_COORDS.get(c, (0, 0))[1])
        
        map_df = user_df[user_df["lat"] != 0].copy()
        if not map_df.empty:
            fig_map = px.scatter_geo(
                map_df, lat="lat", lon="lon", hover_name="City",
                hover_data={c: True for c in ["Source IP", "attack_type", "risk_score", "Country"] if c in map_df.columns},
                size="risk_score" if "risk_score" in map_df.columns else None,
                color="risk_level" if "risk_level" in map_df.columns else "Country",
                projection="natural earth",
                title=f"Geographic Login Activity & Geo-Bubble Risk Heatmap ({selected_user})",
                color_discrete_map={
                    "Low": "#10b981",
                    "Medium": "#f59e0b",
                    "High": "#f97316",
                    "Critical": "#ef4444"
                }
            )
            fig_map.update_layout(**get_dark_plotly_layout())
            st.plotly_chart(fig_map, width="stretch")

    # Section 4: Detailed Event Table
    render_section_divider(f"📜 DETAILED TELEMETRY LOGS ({len(user_df)} EVENTS)")
    display_cols = [c for c in ["Timestamp", "Device ID", "Source IP", "Country", "City", "attack_type", "risk_score", "risk_level", "natural_language_explanation"] if c in user_df.columns]
    st.dataframe(style_soc_dataframe(user_df[display_cols]), width="stretch")

    render_workflow_footer(
        "⚔️ Threat Analytics",
        "⚔️ Analyze Attack Patterns →",
        "Analyze attack patterns: Inspect multi-class cyber attack vector distributions."
    )
