"""
CyberGuard AI Dashboard - SHAP Explainability Page (Enterprise SOC)
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from config import config
from src.dashboard.utils import render_soc_header, style_soc_dataframe

def render_shap_section_divider(title: str):
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

def render_shap_explainability_page(df: pd.DataFrame, render_header: bool = True):
    if render_header:
        render_soc_header("🧠 Explainable AI & SHAP Attribution", "Transparent Machine Learning Predictions & Feature Influence")

    if "natural_language_explanation" not in df.columns:
        st.info("SHAP natural language explanations not present. Run scripts/explain_threats.py to populate.")

    # Filter Anomaly / Threat Records
    anom_df = df[df.get("is_anomaly", 0) == 1].copy()

    if not anom_df.empty:
        render_shap_section_divider("🔎 INDIVIDUAL INCIDENT EXPLAINABILITY INSPECTOR")
        
        sample_idx = st.selectbox(
            "🎯 Select Threat Incident Event ID to Inspect",
            options=anom_df.index,
            format_func=lambda i: f"Event #{i} | User: {anom_df.loc[i, 'User ID']} | Attack: {anom_df.loc[i].get('attack_type', 'Anomaly')} | Risk: {anom_df.loc[i].get('risk_score', 0):.1f}"
        )

        selected_row = anom_df.loc[sample_idx]

        # Natural Language Threat Explanation Premium Card
        explanation_text = selected_row.get("natural_language_explanation", "Anomalous behavior detected.")
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(13, 27, 42, 0.9) 100%); border: 1px solid rgba(56, 189, 248, 0.35); border-left: 4px solid #00f2fe; border-radius: 14px; padding: 22px 24px; margin: 16px 0 20px 0; box-shadow: 0 12px 35px rgba(0, 242, 254, 0.18);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                    <span>💬 NATURAL LANGUAGE THREAT EXPLANATION</span>
                    <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.35); font-size: 0.72rem; padding: 2px 10px; border-radius: 9999px; font-weight: 600;">GROUNDED XAI</span>
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.96rem; color: #f1f5f9; line-height: 1.6; font-weight: 500;">
                    💡 <b>SOC Analyst Summary</b>: {explanation_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Collapsible Deep-Dive Inspector
        with st.expander("🔍 Deep-Dive Feature Attribution & Parameter Breakdown", expanded=True):
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.markdown("##### 🧬 Top SHAP Feature Attributions:")
                top_feats = selected_row.get("top_contributing_features", "N/A")
                st.code(top_feats, language="text")
            with col_exp2:
                st.markdown("##### ⚙️ Incident Risk Parameters:")
                risk_lvl = selected_row.get('risk_level', 'N/A')
                risk_scr = selected_row.get('risk_score', 0)
                st.markdown(f"- **Risk Level**: `<span class='badge-pill badge-{risk_lvl.lower()}'>{risk_lvl}</span>`", unsafe_allow_html=True)
                st.markdown(f"- **Risk Score**: <b style='font-family: \"JetBrains Mono\", monospace; font-size: 1.2rem; color: #38bdf8;'>{risk_scr:.1f} / 100</b>", unsafe_allow_html=True)
                st.markdown(f"- **User ID**: `<span style='font-family: \"JetBrains Mono\", monospace;'>{selected_row.get('User ID', 'N/A')}</span>`", unsafe_allow_html=True)
                st.markdown(f"- **Timestamp**: `<span style='font-family: \"JetBrains Mono\", monospace;'>{selected_row.get('Timestamp', 'N/A')}</span>`", unsafe_allow_html=True)

        render_shap_section_divider(f"📜 EXPLAINABLE THREAT INCIDENT REPORTS ({len(anom_df)} ANOMALIES)")
        display_cols = [c for c in ["Timestamp", "User ID", "attack_type", "risk_score", "risk_level", "natural_language_explanation"] if c in anom_df.columns]
        st.dataframe(style_soc_dataframe(anom_df[display_cols]), width="stretch")

    # Diagnostic SHAP Visualizations Section
    render_shap_section_divider("🖼️ SAVED DIAGNOSTIC SHAP FEATURE INFLUENCE PLOTS")

    figures_dir = config.dirs.base_dir / "reports" / "figures"
    col_p1, col_p2 = st.columns(2)

    sum_plot_path = figures_dir / "shap_summary_plot.png"
    bar_plot_path = figures_dir / "shap_feature_importance.png"
    force_plot_path = figures_dir / "shap_force_plot.png"

    with col_p1:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%); border: 1px solid rgba(56, 189, 248, 0.28); border-radius: 16px; padding: 20px 22px; box-shadow: 0 14px 40px rgba(0,0,0,0.5); backdrop-filter: blur(16px); margin-bottom: 24px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;">📊 SHAP Summary Beeswarm Plot</div>
            """,
            unsafe_allow_html=True
        )
        if sum_plot_path.exists():
            st.image(str(sum_plot_path), caption="Global SHAP Value Distribution Across Features", width="stretch")
        else:
            st.caption("SHAP Summary plot not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_p2:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%); border: 1px solid rgba(56, 189, 248, 0.28); border-radius: 16px; padding: 20px 22px; box-shadow: 0 14px 40px rgba(0,0,0,0.5); backdrop-filter: blur(16px); margin-bottom: 24px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;">📈 Mean Absolute SHAP Feature Importance</div>
            """,
            unsafe_allow_html=True
        )
        if bar_plot_path.exists():
            st.image(str(bar_plot_path), caption="Feature Ranking by Average Impact Size", width="stretch")
        else:
            st.caption("SHAP Feature Importance plot not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    if force_plot_path.exists():
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%); border: 1px solid rgba(56, 189, 248, 0.28); border-radius: 16px; padding: 20px 22px; box-shadow: 0 14px 40px rgba(0,0,0,0.5); backdrop-filter: blur(16px); margin-bottom: 24px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;">⚡ SHAP Decision Trajectory Plot</div>
            """,
            unsafe_allow_html=True
        )
        st.image(str(force_plot_path), caption="SHAP Decision Plot - Anomaly Trajectories", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
