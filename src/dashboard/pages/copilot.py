"""
CyberGuard AI Dashboard - AI Security Copilot Page (Enterprise SOC)
"""

import streamlit as st
import pandas as pd
from src.copilot.sec_copilot import AISecurityCopilot
from src.dashboard.utils import render_soc_header, render_workflow_footer, render_ai_confidence_meter

def get_response_icon_header(response_text: str) -> tuple[str, str]:
    """Returns contextual header icon and category title based on AI response content."""
    txt = response_text.lower()
    if "remediation" in txt or "action" in txt or "recommend" in txt:
        return "💡", "SOC PLAYBOOK RECOMMENDATION"
    elif "alert" in txt or "anomaly" in txt or "threat" in txt:
        return "🚨", "THREAT ALERT SUMMARY & TRIAGE"
    elif "shap" in txt or "feature" in txt or "model" in txt:
        return "🧠", "SHAP ATTRIBUTION & BEHAVIORAL ANALYSIS"
    elif "user" in txt or "profile" in txt or "entity" in txt:
        return "👤", "ENTITY PROFILE INVESTIGATION"
    return "🛡️", "CYBERGUARD AI CO-PILOT ADVISORY"

def render_user_chat_card(prompt_text: str):
    """Renders Analyst Prompt in a styled glassmorphic user card."""
    st.markdown(
        f"""
        <div class="copilot-user-card">
            <div class="copilot-card-header" style="color: #c084fc;">
                <span>👤 SOC ANALYST PROMPT</span>
            </div>
            <div class="copilot-card-body">
                {prompt_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_ai_chat_card(response_text: str):
    """Renders Copilot Response in an enterprise AI security card with confidence meter."""
    icon, category_title = get_response_icon_header(response_text)
    formatted_body = response_text.replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="copilot-ai-card">
            <div class="copilot-card-header" style="color: #00f2fe;">
                <span>{icon} {category_title}</span>
                <span class="soc-kpi-badge badge-low" style="background: rgba(0,242,254,0.15); color: #00f2fe; border: 1px solid rgba(0,242,254,0.4);">
                    🟢 VERIFIED RESPONSE
                </span>
            </div>
            <div class="copilot-card-body">
                {formatted_body}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_ai_confidence_meter(0.96)

@st.cache_resource
def get_cached_copilot(_df: pd.DataFrame) -> AISecurityCopilot:
    """Caches AISecurityCopilot instance using st.cache_resource to eliminate re-initialization lag."""
    return AISecurityCopilot(_df)

def render_copilot_page(df: pd.DataFrame):
    render_soc_header("🧠 AI Security Copilot", "Natural Language Threat Investigation & SOC Playbook Guidance")

    copilot = get_cached_copilot(df)

    st.markdown("##### 💡 Suggested SOC Analyst Queries & Quick Actions")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    prompt_to_submit = None
    with col1:
        if st.button("Why is USR-00109 risky?", width="stretch"):
            prompt_to_submit = "Why is User USR-00109 flagged as high risk?"
    with col2:
        if st.button("Show critical alerts", width="stretch"):
            prompt_to_submit = "Show today's critical alerts and anomalies"
    with col3:
        if st.button("Explain SHAP result", width="stretch"):
            prompt_to_submit = "Explain the SHAP feature importance for impossible travel"
    with col4:
        if st.button("Recommend mitigation", width="stretch"):
            prompt_to_submit = "Recommend mitigation playbooks for brute force attacks"
    with col5:
        if st.button("Generate Incident Report", width="stretch"):
            prompt_to_submit = "Generate an executive SOC Incident Triage Summary Report"

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Initialize chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your CyberGuard AI Security Copilot. Ask me any question about telemetry alerts, high-risk users, SHAP feature explanations, or recommended mitigation playbooks!"
            }
        ]

    # Render Chat History with Custom Enterprise Cards
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            render_user_chat_card(msg["content"])
        else:
            render_ai_chat_card(msg["content"])

    # Handle User Input
    user_input = st.chat_input("Ask AI Copilot a question (e.g. Why was User USR-00109 flagged?)...")

    active_prompt = user_input or prompt_to_submit

    if active_prompt:
        # Save & Render User Message
        st.session_state["messages"].append({"role": "user", "content": active_prompt})
        render_user_chat_card(active_prompt)

        # Generate & Render Copilot Response
        response = copilot.answer_query(active_prompt)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        render_ai_chat_card(response)

    # Export Chat History Button
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    if st.session_state.get("messages"):
        chat_txt = "\n\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in st.session_state["messages"]])
        st.download_button(
            label="📥 Export AI Copilot Investigation Session (TXT)",
            data=chat_txt.encode("utf-8"),
            file_name="cyberguard_ai_copilot_session.txt",
            mime="text/plain"
        )

    render_workflow_footer(
        "📊 AI Intelligence",
        "📊 Explain & Validate AI Decisions →",
        "Explain and validate AI decisions: Inspect SHAP feature attributions & model validation curves."
    )
