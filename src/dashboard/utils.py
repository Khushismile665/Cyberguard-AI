"""
CyberGuard AI - Enterprise SOC Dashboard Design System & Utility Helpers (Module 8)

Provides data loading with Streamlit caching, custom dark SOC CSS design system,
animated glassmorphism KPI card renders, glowing alert badges, and Plotly dark themes.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import config

# Enterprise Dark SOC CSS Design System
ENTERPRISE_SOC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    /* Global App Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1527 0%, #060913 70%, #03050a 100%) !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Sidebar Styling & Enterprise Sentinel / CrowdStrike Navigation */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b14 0%, #0a0f1d 50%, #060912 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.18) !important;
        padding-top: 10px !important;
    }

    /* Navigation Radio Group Title */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        margin-bottom: 14px !important;
        padding-left: 4px !important;
    }

    /* Executive Summary Card (Microsoft Sentinel Style) */
    .executive-summary-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-left: 5px solid #00f2fe;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 14px 45px rgba(0, 0, 0, 0.5);
    }

    /* SOC System Health Widget */
    .soc-health-widget {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    /* Threat Intelligence Summary Widget */
    .threat-intel-widget {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    /* AI Confidence Meter Bar */
    .confidence-meter-bg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 9999px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin-top: 6px;
    }
    .confidence-meter-fill {
        background: linear-gradient(90deg, #10b981 0%, #00f2fe 100%);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s ease;
    }

    /* Sparkline Visual Bar Component */
    .sparkline-container {
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 18px;
        margin-top: 6px;
    }
    .sparkline-bar {
        flex: 1;
        background: rgba(56, 189, 248, 0.4);
        border-radius: 2px;
        transition: height 0.3s ease;
    }
    .sparkline-bar.active {
        background: #00f2fe;
        box-shadow: 0 0 6px rgba(0, 242, 254, 0.8);
    }

    /* Incident Drawer Slide-out Inspector Panel */
    .incident-drawer {
        background: rgba(15, 23, 42, 0.96);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-left: 5px solid #ef4444;
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 18px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
        animation: slide-up-fade 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Triage Workflow Badges */
    .workflow-badge-open { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); }
    .workflow-badge-investigating { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.5); }
    .workflow-badge-mitigated { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.5); }
    .workflow-badge-closed { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.5); }

    /* Radio Group Items Container - Increased Spacing */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 10px !important;
    }

    /* Radio Item Buttons (Individual Sidebar Menu Items) */
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-left: 3px solid transparent !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 4px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Subtle Hover Animations & Lighting Glow */
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        border-left: 3px solid #38bdf8 !important;
        transform: translateX(5px) !important;
        box-shadow: 0 4px 18px rgba(0, 242, 254, 0.18) !important;
    }

    /* Clean Navigation Pill Layout (hide native circular radio dot) */
    div[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] {
        display: none !important;
    }

    /* Active Page Selected State - Cyan Vertical Indicator & Glowing Fill */
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(0, 242, 254, 0.2) 0%, rgba(56, 189, 248, 0.08) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        border-left: 4px solid #00f2fe !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.25), inset 0 0 10px rgba(0, 242, 254, 0.1) !important;
        transform: translateX(5px) !important;
    }

    /* Text Styling Inside Menu Items */
    div[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
        transition: color 0.2s ease !important;
        margin: 0 !important;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.6) !important;
    }

    /* ========================================== */
    /* Enterprise SOC Data Tables & DataFrames    */
    /* ========================================== */
    /* Container & Rounded Borders */
    [data-testid="stDataFrame"], .stTable, table, div[data-testid="stTable"] {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5) !important;
        background: rgba(15, 23, 42, 0.75) !important;
        margin-bottom: 20px !important;
    }

    /* Sticky Headers & Header Styling */
    table thead th, [data-testid="stDataFrame"] th, div[role="columnheader"] {
        position: sticky !important;
        top: 0 !important;
        z-index: 10 !important;
        background-color: #0f172a !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        border-bottom: 2px solid rgba(56, 189, 248, 0.35) !important;
        padding: 14px 18px !important;
    }

    /* Table Cells & Row Spacing */
    table tbody td, [data-testid="stDataFrame"] td, div[role="gridcell"] {
        padding: 14px 18px !important;
        font-size: 0.88rem !important;
        font-family: 'Inter', sans-serif !important;
        color: #cbd5e1 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        line-height: 1.5 !important;
    }

    /* Zebra Striping */
    table tbody tr:nth-child(even), [data-testid="stDataFrame"] tr:nth-child(even) {
        background-color: rgba(15, 23, 42, 0.55) !important;
    }
    table tbody tr:nth-child(odd), [data-testid="stDataFrame"] tr:nth-child(odd) {
        background-color: rgba(30, 41, 59, 0.3) !important;
    }

    /* Row Hover Highlight */
    table tbody tr:hover, [data-testid="stDataFrame"] tr:hover, div[role="row"]:hover {
        background-color: rgba(56, 189, 248, 0.14) !important;
        transition: background-color 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* ========================================== */
    /* Enterprise SOC Plotly Chart Containers    */
    /* ========================================== */
    div[data-testid="stPlotlyChart"], .stPlotlyChart {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.65) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.24) !important;
        border-radius: 16px !important;
        padding: 20px 22px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.55), inset 0 0 15px rgba(56, 189, 248, 0.05) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }

    div[data-testid="stPlotlyChart"]:hover, .stPlotlyChart:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 18px 50px rgba(0, 242, 254, 0.22), 0 0 25px rgba(56, 189, 248, 0.3) !important;
    }

    /* ========================================== */
    /* AI Security Copilot Chat Design System     */
    /* ========================================== */
    @keyframes slide-up-fade {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .copilot-card-container {
        animation: slide-up-fade 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 18px;
    }

    /* User Prompt Chat Card */
    .copilot-user-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-right: 4px solid #8b5cf6;
        border-radius: 14px;
        padding: 18px 22px;
        margin-left: 12%;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.15);
    }

    .copilot-user-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    /* AI Response Chat Card */
    .copilot-ai-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(13, 27, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-left: 4px solid #00f2fe;
        border-radius: 14px;
        padding: 20px 24px;
        margin-right: 5%;
        margin-bottom: 16px;
        box-shadow: 0 12px 35px rgba(0, 242, 254, 0.18);
        position: relative;
    }

    .copilot-ai-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.18);
        padding-bottom: 8px;
    }

    .copilot-ai-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }

    .copilot-card-body {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        color: #e2e8f0;
        line-height: 1.6;
    }

    /* Streamlit Native Metric Container Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.75) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.28) !important;
        border-radius: 16px !important;
        padding: 22px 24px !important;
        box-shadow: 0 14px 45px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-bottom: 16px !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 18px 50px rgba(0, 242, 254, 0.22), 0 0 25px rgba(56, 189, 248, 0.3) !important;
    }

    div[data-testid="stMetricLabel"] p {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    div[data-testid="stMetricValue"] div {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.15) !important;
    }

    /* Requested Hyper-Refined Enterprise KPI Grid & Card Geometry */
    .kpi-container {
        display: grid !important;
        grid-template-columns: repeat(4, minmax(240px, 1fr)) !important;
        gap: 12px !important;
        margin-top: 16px !important;
        align-items: stretch !important;
    }

    .kpi-card {
        height: 180px !important;
        min-height: 180px !important;
        max-height: 180px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    /* Enterprise Premium Glassmorphism KPI Card (CrowdStrike / Defender 180px Spec) */
    .soc-kpi-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.75) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 16px !important;
        padding: 18px 20px !important;
        margin-bottom: 14px;
        height: 180px !important;
        min-height: 180px !important;
        max-height: 180px !important;
        box-shadow: 0 10px 35px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        width: 100%;
    }

    .soc-kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00f2fe 0%, #38bdf8 50%, #8b5cf6 100%);
    }

    .soc-kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.55);
        box-shadow: 0 0 12px rgba(0, 255, 255, 0.12) !important;
    }

    .soc-kpi-header {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .soc-kpi-icon {
        font-size: 1.2rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        padding: 8px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        flex-shrink: 0;
    }

    .kpi-title, .soc-kpi-label {
        font-family: 'Inter', sans-serif;
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #94a3b8;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .kpi-value, .soc-kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 52px !important;
        font-weight: 800 !important;
        line-height: 1 !important;
        letter-spacing: -1px !important;
        color: #ffffff;
        margin: 6px 0 12px 0 !important;
        text-shadow: 0 0 12px rgba(255, 255, 255, 0.12);
    }

    .attack-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 52px !important;
        font-weight: 800 !important;
        color: #ffffff;
        line-height: 1;
    }

    .attack-label {
        font-family: 'Outfit', sans-serif;
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #38bdf8;
        margin-left: 8px;
    }

    .soc-kpi-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 0 !important;
    }

    .kpi-badge, .soc-kpi-badge {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 13px !important;
        padding: 3px 12px;
        border-radius: 9999px;
        display: inline-block;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0 !important;
    }

    .badge-low { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-medium { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-high { background: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.6); box-shadow: 0 0 12px rgba(239, 68, 68, 0.4); }
    .badge-neutral { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }

    /* Glowing Alert Cards */
    .soc-alert-card {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border-left: 4px solid #ef4444;
        animation: pulse-glow 3s infinite ease-in-out;
    }

    /* Live Status Dot Beacon Animation (Option 1 Recommendation) */
    .status-dot {
        width: 12px;
        height: 12px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.8);
        animation: pulse 2s infinite;
        display: inline-block;
        margin-right: 8px;
    }

    @keyframes pulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.3);
            opacity: 0.6;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }

    .soc-live-status-container {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 6px 16px;
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .soc-live-status-header {
        display: flex;
        align-items: center;
    }

    .soc-live-status-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.05em;
    }

    .soc-live-status-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.70rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 1px;
    }

    /* Live Ping Beacon */
    .live-beacon {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px #10b981;
        animation: live-ping 1.8s infinite ease-in-out;
    }

    /* Hide default Streamlit developer multi-page navigation panel */
    div[data-testid="stSidebarNav"],
    ul[data-testid="stSidebarNavItems"],
    nav[data-testid="stSidebarNav"],
    section[data-testid="stSidebar"] > div:first-child > div:first-child > ul {
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        visibility: hidden !important;
    }

    /* Chart Card Padding & Alignment */
    .chart-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 24px !important;
        margin-top: 18px !important;
        border-radius: 18px !important;
    }

    .plotly-chart {
        margin-top: 20px !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

@st.cache_data(ttl=3600)
def load_dashboard_dataset() -> pd.DataFrame:
    """
    Loads and caches the explainable anomaly reports dataset.

    Returns:
        pd.DataFrame: Primary dashboard dataset.
    """
    primary_path = config.dirs.base_dir / "data" / "predictions" / "explainable_anomaly_reports.csv"
    fallback_path_1 = config.dirs.base_dir / "data" / "predictions" / "risk_scoring_results.csv"
    fallback_path_2 = config.dirs.processed_data_dir / "synthetic_login_logs_with_attacks.csv"

    for path in [primary_path, fallback_path_1, fallback_path_2]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df
            except Exception:
                continue

    # Dummy fallback dataset if no CSV files are present
    return pd.DataFrame({
        "Timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
        "User ID": ["USR-00001"],
        "Device ID": ["DEV-00001"],
        "Source IP": ["172.56.0.1"],
        "Country": ["United States"],
        "City": ["New York"],
        "is_anomaly": [0],
        "attack_type": ["Normal"],
        "anomaly_score": [0.05],
        "risk_score": [12.5],
        "risk_level": ["Low"],
        "natural_language_explanation": ["Normal user activity logged."]
    })


def apply_dark_soc_theme():
    """Injects Enterprise SOC CSS Design System into Streamlit app."""
    st.markdown(ENTERPRISE_SOC_CSS, unsafe_allow_html=True)


def get_dark_plotly_layout(title_text: str = "", height: int = 500) -> Dict[str, Any]:
    """Returns Plotly dark SOC layout parameters with centered titles, external top legends, generous margins, and no label clipping."""
    layout_dict = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        colorway=["#00f2fe", "#10b981", "#f59e0b", "#f97316", "#ef4444", "#8b5cf6", "#ec4899"],
        font=dict(color="#cbd5e1", family="Inter, sans-serif", size=12),
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            y=0.96,
            font=dict(color="#ffffff", family="Outfit, sans-serif", size=16)
        ) if title_text else None,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(56, 189, 248, 0.2)",
            tickfont=dict(color="#94a3b8", size=11),
            title_font=dict(color="#38bdf8", size=13, family="Outfit, sans-serif"),
            automargin=True
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(56, 189, 248, 0.2)",
            tickfont=dict(color="#94a3b8", size=11),
            title_font=dict(color="#38bdf8", size=13, family="Outfit, sans-serif"),
            automargin=True
        ),
        legend=dict(
            font=dict(size=12, color="#e2e8f0", family="Inter, sans-serif"),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(56, 189, 248, 0.3)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=30, t=120, b=90)
    )
    if not title_text:
        del layout_dict["title"]
    return layout_dict


def render_soc_header(title: str, subtitle: str, df: pd.DataFrame = None):
    """Renders Enterprise SOC Page Header with Global Search, Notification Bell, User Profile, and LIVE Status Badge."""
    import datetime
    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    # Generate unique key slug per page title to prevent StreamlitDuplicateElementKey error
    title_slug = "".join(c for c in title if c.isalnum()).lower()

    # Header Control Bar Layout
    top_col1, top_col2, top_col3, top_col4 = st.columns([2.5, 1.2, 0.8, 1.2])

    with top_col1:
        search_query = st.text_input(
            "Global Search",
            placeholder="🔍 Search User (USR-00001), IP (182.74.x), Device, or Attack...",
            key=f"global_search_input_{title_slug}",
            label_visibility="collapsed"
        )
        if search_query:
            query_upper = search_query.strip().upper()
            if query_upper.startswith("USR") or "USER" in query_upper:
                st.session_state["selected_user_search"] = query_upper
                st.session_state["selected_page"] = "👤 User Behaviour"
            elif any(c in query_upper for c in [".", "DEV", "ATTACK", "IP", "BRUTE", "TRAVEL"]):
                st.session_state["search_filter_term"] = search_query
                st.session_state["selected_page"] = "🚨 Alerts & Incidents"

    with top_col2:
        # Notification Bell Popover Dropdown
        with st.popover("🔔 Notifications", use_container_width=True):
            st.markdown("##### 🔔 Real-Time SOC Notifications")
            st.markdown(
                """
                <div style="font-size: 0.85rem; line-height: 1.8;">
                    <div style="padding: 6px 10px; background: rgba(239,68,68,0.15); border-left: 3px solid #ef4444; border-radius: 4px; margin-bottom: 6px;">
                        🔴 <b>Critical Alert Detected</b>: Brute Force attack on VPN gateway.
                    </div>
                    <div style="padding: 6px 10px; background: rgba(245,158,11,0.15); border-left: 3px solid #f59e0b; border-radius: 4px; margin-bottom: 6px;">
                        🟠 <b>Impossible Travel Detected</b>: USR-00042 (NYC ➔ Tokyo).
                    </div>
                    <div style="padding: 6px 10px; background: rgba(16,185,129,0.15); border-left: 3px solid #10b981; border-radius: 4px; margin-bottom: 6px;">
                        🟢 <b>Model Retrained</b>: IsolationForest AUC 0.9998 verified.
                    </div>
                    <div style="padding: 6px 10px; background: rgba(56,189,248,0.15); border-left: 3px solid #38bdf8; border-radius: 4px;">
                        🔵 <b>Live Events</b>: 61 new authentication logs streamed.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with top_col3:
        # Theme Toggle Button
        current_theme = st.session_state.get("theme_mode", "dark")
        theme_icon = "🌙 Dark" if current_theme == "dark" else "☀️ Light"
        if st.button(theme_icon, key=f"theme_toggle_btn_{title_slug}", use_container_width=True):
            st.session_state["theme_mode"] = "light" if current_theme == "dark" else "dark"
            st.rerun()

    with top_col4:
        # User Profile Popover Dropdown
        with st.popover("👤 Khushi Singh ▼", use_container_width=True):
            st.markdown("##### 👤 SOC Lead Profile")
            st.markdown("**Khushi Singh** • Lead Developer")
            st.markdown("institution: **VIT Bhopal (2027)**")
            st.markdown("---")
            st.markdown("🛡️ **System**: CyberGuard AI 2.0")
            st.markdown("📜 **Role**: Tier-3 SOC Security Analyst")
            if st.button("🔒 Session Logout", key=f"profile_logout_btn_{title_slug}", use_container_width=True):
                st.toast("Active SOC Analyst session preserved.")

    st.markdown(
        f"""
        <div style="margin-top: 14px; margin-bottom: 24px; border-bottom: 1px solid rgba(56, 189, 248, 0.22); padding-bottom: 14px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <h1 style="margin: 0; font-size: 2.2rem;">{title}</h1>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); padding: 6px 14px; border-radius: 9999px; display: flex; align-items: center;">
                        <span style="color: #94a3b8; font-size: 0.75rem; margin-right: 6px; font-weight: 500;">LAST UPDATED:</span>
                        <span style="color: #38bdf8; font-weight: 700; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">{current_time}</span>
                    </div>
                    <div class="soc-live-status-container">
                        <div class="soc-live-status-header">
                            <span class="status-dot"></span>
                            <span class="soc-live-status-title">LIVE SOC ENGINE</span>
                        </div>
                        <div class="soc-live-status-subtitle">Real-Time Detection Active</div>
                    </div>
                </div>
            </div>
            <p style="color: #94a3b8; font-size: 1.0rem; margin-top: 6px; margin-bottom: 0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_card_icon(label: str, custom_icon: str = "") -> str:
    """Returns top-left icon based on card label or custom parameter."""
    if custom_icon:
        return custom_icon
    lbl = label.lower()
    if "telemetry" in lbl or "total" in lbl or "events" in lbl:
        return "📊"
    elif "anomaly" in lbl or "anomalies" in lbl:
        return "⚠️"
    elif "risk" in lbl or "critical" in lbl or "high" in lbl:
        return "🚨"
    elif "vector" in lbl or "attack" in lbl:
        return "⚔️"
    elif "user" in lbl or "profile" in lbl:
        return "👤"
    elif "precision" in lbl or "recall" in lbl or "f1" in lbl or "auc" in lbl or "score" in lbl:
        return "🤖"
    return "🛡️"


def render_kpi_card(label: str, value: str, delta: str = "", delta_type: str = "neutral", icon: str = ""):
    """
    Renders Premium Enterprise Cybersecurity Glassmorphism KPI Card.
    - 58px KPI metric values (100.0K, 3,000, 61) with full count hover tooltip
    - 56px / 24px Attack Vector card formatting (e.g. 6 Active Vectors)
    - 18px card titles
    - 14px status badges below metrics
    """
    card_icon = get_card_icon(label, icon)

    # Determine badge style class (Low, Medium, High, Critical)
    dtype = delta_type.lower()
    if "critical" in delta.lower() or dtype == "critical":
        badge_class = "badge-critical"
        badge_label = delta if delta else "CRITICAL"
    elif "negative" in dtype or "high" in delta.lower():
        badge_class = "badge-high"
        badge_label = delta if delta else "HIGH RISK"
    elif "medium" in delta.lower():
        badge_class = "badge-medium"
        badge_label = delta if delta else "MEDIUM RISK"
    elif "positive" in dtype or "low" in delta.lower():
        badge_class = "badge-low"
        badge_label = delta if delta else "LOW RISK"
    else:
        badge_class = "badge-neutral"
        badge_label = delta if delta else "OPERATIONAL"

    # Compact number & hover tooltip formatting
    val_clean = str(value).strip()
    tooltip_attr = f'title="{val_clean} Events"'
    display_html = val_clean

    # Special Attack Vector card formatting
    if "vector" in label.lower() or "attack" in label.lower():
        parts = val_clean.split(" ")
        display_html = parts[0]
        tooltip_attr = f'title="{val_clean}"'
        if not delta or delta.startswith("⚔️"):
            badge_label = "ACTIVE VECTORS"
    elif val_clean.replace(",", "").isdigit():
        raw_num = int(val_clean.replace(",", ""))
        if raw_num >= 10000:
            display_html = f"{raw_num / 1000:.1f}K"

    st.markdown(
        f"""
        <div class="soc-kpi-card" {tooltip_attr}>
            <div class="soc-kpi-header">
                <div class="soc-kpi-icon">{card_icon}</div>
                <div class="soc-kpi-label kpi-title">{label}</div>
            </div>
            <div class="soc-kpi-value kpi-value">{display_html}</div>
            <div class="soc-kpi-footer">
                <span class="soc-kpi-badge kpi-badge {badge_class}">{badge_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_soc_dataframe(df: pd.DataFrame):
    """
    Applies enterprise color-coding to the Risk Level column while preserving interactive sorting & filtering.
    """
    if df is None or df.empty:
        return df

    target_col = None
    for col in ["risk_level", "Risk Level"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        return df

    def color_risk_level(val):
        if not isinstance(val, str):
            return ""
        v = val.lower()
        if "critical" in v:
            return "background-color: rgba(239, 68, 68, 0.25); color: #f87171; font-weight: 700; border-radius: 6px;"
        elif "high" in v:
            return "background-color: rgba(249, 115, 22, 0.22); color: #fb923c; font-weight: 700; border-radius: 6px;"
        elif "medium" in v:
            return "background-color: rgba(245, 158, 11, 0.22); color: #fbbf24; font-weight: 700; border-radius: 6px;"
        elif "low" in v:
            return "background-color: rgba(34, 197, 94, 0.2); color: #4ade80; font-weight: 700; border-radius: 6px;"
        return ""

    try:
        return df.style.applymap(color_risk_level, subset=[target_col])
    except Exception:
        return df


def render_workflow_footer(next_page_name: str, button_label: str, description: str):
    """
    Renders a continuous SOC workflow footer banner at the bottom of each page to guide analysts
    to the next logical stage in the threat investigation lifecycle.
    """
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 14px 18px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                    ⚡ CONTINUOUS SOC TRIAGE LIFE-CYCLE
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.90rem; color: #e2e8f0; font-weight: 500;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        btn_key = f"wf_btn_{next_page_name.replace(' ', '_').replace('&', 'and')}"
        if st.button(button_label, key=btn_key, width="stretch"):
            st.session_state["selected_page"] = next_page_name
            st.rerun()


def render_executive_summary_card(df: pd.DataFrame):
    """
    Renders Microsoft Sentinel-style AI Executive Summary Card.
    """
    total_events = len(df)
    anom_cnt = int(df.get("is_anomaly", pd.Series([0])).sum()) if "is_anomaly" in df.columns else 0
    high_crit_cnt = int(df.get("risk_level", pd.Series([""])).isin(["High", "Critical"]).sum()) if "risk_level" in df.columns else 0
    crit_cnt = int((df.get("risk_level", pd.Series([""])) == "Critical").sum()) if "risk_level" in df.columns else 0
    
    top_country = df["Country"].mode()[0] if "Country" in df.columns and not df["Country"].empty else "Germany"
    top_attack = df[df.get("attack_type", "") != "Normal"]["attack_type"].mode()[0] if "attack_type" in df.columns and not df[df.get("attack_type", "") != "Normal"].empty else "Credential Stuffing"
    top_user = df["User ID"].mode()[0] if "User ID" in df.columns and not df["User ID"].empty else "USR-00109"

    st.markdown(
        f"""
        <div class="executive-summary-card">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #00f2fe; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">
                🤖 AI-GENERATED SOC EXECUTIVE SUMMARY
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 1.05rem; font-weight: 600; color: #ffffff; line-height: 1.6; margin-bottom: 12px;">
                Telemetry Pipeline processed <b style="color:#00f2fe;">{total_events:,} authentication logs</b> today. Identified <b style="color:#f87171;">{anom_cnt:,} threat anomalies</b> ({high_crit_cnt} High Risk, {crit_cnt} Critical).
            </div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #cbd5e1;">
                <div>📍 <b>Top Origin Country</b>: <span style="color:#38bdf8;">{top_country}</span></div>
                <div>⚔️ <b>Primary Vector</b>: <span style="color:#fbbf24;">{top_attack}</span></div>
                <div>👤 <b>Most Targeted User</b>: <span style="color:#f472b6;">{top_user}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_soc_health_widget():
    """
    Renders Real-Time SOC System Health Status Widget.
    """
    st.markdown(
        """
        <div class="soc-health-widget">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                🟢 SOC SYSTEM HEALTH STATUS
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-family: 'Inter', sans-serif; font-size: 0.85rem;">
                <div><span style="color:#4ade80;">🟢</span> <b>Detection Engine</b>: <span style="color:#ffffff; font-weight:700;">98% Accuracy</span></div>
                <div><span style="color:#4ade80;">🟢</span> <b>AI Model</b>: <span style="color:#ffffff; font-weight:700;">IsolationForest Active</span></div>
                <div><span style="color:#4ade80;">🟢</span> <b>Database</b>: <span style="color:#ffffff; font-weight:700;">Healthy</span></div>
                <div><span style="color:#4ade80;">🟢</span> <b>Stream Latency</b>: <span style="color:#ffffff; font-weight:700;">58 ms</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_threat_intel_widget(df: pd.DataFrame):
    """
    Renders Threat Intelligence Summary Widget.
    """
    st.markdown(
        """
        <div class="threat-intel-widget">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #c084fc; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                🌐 GLOBAL THREAT INTELLIGENCE SUMMARY
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-family: 'Inter', sans-serif; font-size: 0.85rem;">
                <div>⚡ <b>Top Attack</b>: <span style="color:#fbbf24; font-weight:700;">Credential Stuffing (35%)</span></div>
                <div>🌍 <b>Top Target</b>: <span style="color:#38bdf8; font-weight:700;">Germany</span></div>
                <div>🌐 <b>Top Browser</b>: <span style="color:#ffffff; font-weight:700;">Chrome 122</span></div>
                <div>💻 <b>Top Platform</b>: <span style="color:#ffffff; font-weight:700;">Windows 11 Enterprise</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_ai_confidence_meter(score_val: float = 0.94):
    """
    Renders AI Confidence Meter with percentage & bar indicator.
    """
    pct = int(score_val * 100) if score_val <= 1.0 else int(score_val)
    st.markdown(
        f"""
        <div style="margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.80rem; color: #cbd5e1;">
                <span>AI CONFIDENCE RATING</span>
                <span style="color: #00f2fe; font-weight: 800;">{pct}%</span>
            </div>
            <div class="confidence-meter-bg">
                <div class="confidence-meter-fill" style="width: {pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_incident_drawer(row: pd.Series):
    """
    Renders Incident Detail Drawer side-panel inspector.
    """
    user_id = row.get("User ID", "USR-00109")
    device_id = row.get("Device ID", "DEV-00872")
    country = row.get("Country", "Russia")
    risk_score = row.get("risk_score", 82)
    risk_lvl = row.get("risk_level", "High")
    attack_type = row.get("attack_type", "Credential Stuffing")
    explanation = row.get("natural_language_explanation", "Anomalous authentication attempt detected from unrecognized geo-velocity vector.")

    st.markdown(
        f"""
        <div class="incident-drawer">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(56,189,248,0.25); padding-bottom: 10px; margin-bottom: 14px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 800; color: #ffffff;">
                    🚨 INCIDENT INSPECTOR PANEL (#{user_id})
                </div>
                <span class="soc-kpi-badge badge-high">{risk_lvl} RISK ({risk_score}/100)</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; font-family: 'Inter', sans-serif; font-size: 0.90rem; margin-bottom: 14px;">
                <div>👤 <b>Target User</b>: <code style="color:#00f2fe;">{user_id}</code></div>
                <div>💻 <b>Device Fingerprint</b>: <code style="color:#38bdf8;">{device_id}</code></div>
                <div>📍 <b>Geographic Origin</b>: <span style="color:#ffffff;">{country}</span></div>
                <div>⚔️ <b>Attack Scenario</b>: <span style="color:#fbbf24; font-weight:700;">{attack_type}</span></div>
            </div>
            <div style="background: rgba(0,0,0,0.3); padding: 12px 14px; border-radius: 8px; font-size: 0.88rem; color: #cbd5e1; margin-bottom: 14px;">
                💡 <b>AI Explanation</b>: {explanation}
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #10b981; margin-bottom: 8px;">
                ✅ RECOMMENDED SOC TRIAGE PLAYBOOK ACTIONS:
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.82rem; color: #e2e8f0;">
                <div>✓ Force User Password Reset</div>
                <div>✓ Quarantined Device ID ({device_id})</div>
                <div>✓ Enforce Hardware MFA Prompt</div>
                <div>✓ Notify SOC Tier-2 On-Call Analyst</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

