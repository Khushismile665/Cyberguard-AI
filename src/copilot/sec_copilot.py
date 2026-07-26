"""
CyberGuard AI - AI Security Copilot Engine (Module 9)

Ground-truth grounded AI assistant providing natural-language threat explanations,
incident investigations, risk leaderboards, and SOC mitigation playbooks strictly
from telemetry prediction results and SHAP attributions.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("AISecurityCopilot")

RECOMMENDED_ACTIONS = {
    "Impossible Travel": [
        "🚨 **Immediate Revocation**: Force immediate session revocation and terminate active OAuth/SAML refresh tokens.",
        "🔐 **Step-up Authentication**: Require strict FIDO2 WebAuthn or hardware security key MFA challenge.",
        "🌐 **IP Perimeter Rule**: Temporarily block the remote source IP at the cloud API gateway / WAF.",
        "📊 **Geographic Audit**: Review all recent IP geolocation trajectories for this user account over the past 7 days."
    ],
    "Brute Force": [
        "🔒 **Account Lockout**: Temporarily suspend user account for 30 minutes to stop ongoing automated attempt.",
        "🔑 **Password Reset**: Enforce mandatory password reset upon next successful authentication.",
        "🛡️ **WAF Rate-Limiting**: Block origin IP / subnet at perimeter firewall and enable CAPTCHA challenge.",
        "🔍 **Credential Exposure**: Check if the targeted username/email appears in recent external data breaches."
    ],
    "Credential Stuffing": [
        "🔒 **Account Lockout & Force Reset**: Immediately reset credentials and invalidate active sessions.",
        "🛡️ **Bot Protection**: Enable automated bot detection and CAPTCHA challenge on the login endpoint.",
        "🚫 **IP Reputation Block**: Add offending IP addresses to firewall threat intelligence blocklist.",
        "📣 **User Notification**: Send automated security alert notification to the affected account holder."
    ],
    "Device Spoofing": [
        "📱 **Device Revocation**: Revoke untrusted device fingerprint and invalidate registered device certificate.",
        "💻 **MDM/EDR Inspection**: Trigger automated compliance and health check scan via Enterprise EDR agent.",
        "🔑 **MFA Re-enrollment**: Require user to re-register authentication factor from a trusted corporate device."
    ],
    "Lateral Movement": [
        "🖥️ **Host Isolation**: Isolate compromised source host via EDR agent to prevent subnet propagation.",
        "🔑 **Privileged Credential Reset**: Revoke domain admin/privileged Kerberos tickets and service account keys.",
        "📜 **Audit Log Inspection**: Review all internal SSH, RDP, and API access logs for the past 24 hours.",
        "🚨 **Escalate to Tier-2**: Escalate incident to Lead SOC Incident Response Handler immediately."
    ],
    "Insider Threat": [
        "👁️ **Insider Risk Alert**: Notify Insider Risk Management and HR Security compliance teams.",
        "🚫 **Resource Access Restriction**: Temporarily revoke access permissions to sensitive repositories (`/admin/settings`, `/dev/git-repository`).",
        "🎥 **Full Session Audit**: Enable high-fidelity audit logging and full session recording.",
        "📂 **DLP Inspection**: Audit recent Data Loss Prevention (DLP) file download and transfer activity."
    ],
    "Default": [
        "🔍 **Investigate Event**: Review raw login event parameters and verify user identity via out-of-band communication.",
        "🔐 **Enforce MFA**: Prompt user with step-up MFA challenge upon next login.",
        "📜 **Audit Logs**: Inspect adjacent user activity logs over the past 24 hours."
    ]
}


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Helper to convert DataFrame to clean markdown table without requiring tabulate package."""
    if df.empty:
        return ""
    headers = list(df.columns)
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in headers]
        md += "| " + " | ".join(vals) + " |\n"
    return md

class AISecurityCopilot:
    """
    AI Security Copilot engine grounded strictly on prediction results and SHAP attributions.
    """

    def __init__(self, df_predictions: pd.DataFrame):
        """
        Initializes AISecurityCopilot with prediction results dataframe.

        Args:
            df_predictions (pd.DataFrame): Dataframe containing predictions, risk scores, and SHAP explanations.
        """
        self.df = df_predictions.copy()
        logger.info(f"Initialized AISecurityCopilot with dataset shape: {self.df.shape}")

    def answer_query(self, query: str) -> str:
        """
        Parses natural language query intent and generates grounded markdown response.

        Args:
            query (str): Natural language question from SOC analyst.

        Returns:
            str: Markdown response formatted strictly from dataset facts.
        """
        if not query or not isinstance(query, str):
            return "Please provide a valid question regarding security alerts or user threats."

        q_clean = query.strip()
        q_lower = q_clean.lower()

        logger.info(f"Processing Copilot Query: '{q_clean}'")

        # 1. User Specific Inspection Intent (e.g., "Why was User USR-00923 flagged?", "USR-00923", "explain user USR-00102")
        user_match = re.search(r"usr-\d+", q_lower)
        if user_match or "user" in q_lower:
            user_id = user_match.group(0).upper() if user_match else None
            if not user_id:
                # Search for user ID in query
                for u in self.df.get("User ID", pd.Series([])).unique():
                    if str(u).lower() in q_lower:
                        user_id = u
                        break
            if user_id:
                return self.explain_user(user_id)

        # 2. Attack Category Intent (e.g., "Show all impossible travel attacks", "brute force attacks")
        attack_types = ["impossible travel", "brute force", "credential stuffing", "device spoofing", "lateral movement", "insider threat"]
        for atk in attack_types:
            if atk in q_lower:
                return self.filter_attacks(atk)

        # 3. Highest Risk Users Intent (e.g., "Which users have the highest risk?", "top risk users")
        if any(k in q_lower for k in ["highest risk", "top risk", "most risky", "leaderboard", "high risk users"]):
            return self.get_top_risk_users()

        # 4. Action Recommendation Intent (e.g., "Recommend actions for this alert", "remediation", "action plan")
        if any(k in q_lower for k in ["recommend", "action", "remediation", "mitigate", "what should i do"]):
            # Check if attack type is specified in query
            target_atk = None
            for atk in attack_types:
                if atk in q_lower:
                    target_atk = atk
                    break
            return self.recommend_actions(attack_type=target_atk)

        # 5. Explain Anomaly / Specific Event Index Intent (e.g., "Explain this anomaly", "explain event 42")
        event_match = re.search(r"event\s*#?(\d+)", q_lower)
        if event_match:
            event_idx = int(event_match.group(1))
            return self.explain_anomaly(event_idx=event_idx)
        elif "anomaly" in q_lower or "explain" in q_lower or "alert" in q_lower:
            return self.explain_anomaly()

        # 6. Fallback Zero-Hallucination Disclaimer
        return self._grounded_fallback_response(q_clean)

    def explain_user(self, user_id: str) -> str:
        """Explains why a specific user was flagged based strictly on empirical dataset records."""
        if "User ID" not in self.df.columns:
            return f"❌ User ID column not present in active telemetry dataset."

        user_records = self.df[self.df["User ID"] == user_id]
        if user_records.empty:
            return f"❓ **User ID `{user_id}`** was not found in the telemetry dataset."

        total_events = len(user_records)
        anom_records = user_records[user_records.get("is_anomaly", 0) == 1]
        anom_cnt = len(anom_records)
        max_risk = user_records.get("risk_score", pd.Series([0])).max()
        highest_risk_record = user_records.loc[user_records.get("risk_score", pd.Series([0])).idxmax()]

        risk_lvl = highest_risk_record.get("risk_level", "Low")
        atk_type = highest_risk_record.get("attack_type", "Normal")
        explanation = highest_risk_record.get("natural_language_explanation", "Normal baseline activity.")
        top_feats = highest_risk_record.get("top_contributing_features", "N/A")
        ip_addr = highest_risk_record.get("Source IP", "N/A")
        country = highest_risk_record.get("Country", "N/A")

        res = f"### 🛡️ Threat Analysis for User `{user_id}`\n\n"
        res += f"- **Risk Level**: `{risk_lvl}` (Peak Risk Score: **{max_risk:.1f} / 100**)\n"
        res += f"- **Total Authentication Events**: **{total_events:,}**\n"
        res += f"- **Flagged Anomaly Events**: **{anom_cnt:,}** ({(anom_cnt/total_events*100):.1f}% ratio)\n"
        res += f"- **Primary Attack Classification**: `{atk_type}`\n"
        res += f"- **Source IP / Country**: `{ip_addr}` ({country})\n\n"
        res += f"#### 💬 SHAP Plain-English Explanation:\n> {explanation}\n\n"
        res += f"#### 📊 Top SHAP Feature Attributions:\n```text\n{top_feats}\n```\n\n"

        # Attach recommended action
        actions = RECOMMENDED_ACTIONS.get(atk_type, RECOMMENDED_ACTIONS["Default"])
        res += "#### 🛠️ Recommended SOC Mitigation Actions:\n"
        for act in actions:
            res += f"- {act}\n"

        return res

    def filter_attacks(self, attack_type_str: str) -> str:
        """Filters dataset by attack category and provides summary table."""
        if "attack_type" not in self.df.columns:
            return "❌ Attack classification labels not present in dataset."

        # Case-insensitive matching
        matched_rows = self.df[self.df["attack_type"].str.lower() == attack_type_str.lower()]

        if matched_rows.empty:
            return f"ℹ️ Zero records matched attack category: **{attack_type_str.title()}**."

        count = len(matched_rows)
        unique_users = matched_rows.get("User ID", pd.Series([])).nunique()
        avg_risk = matched_rows.get("risk_score", pd.Series([0])).mean()
        max_risk = matched_rows.get("risk_score", pd.Series([0])).max()

        res = f"### ⚔️ Attack Vector Investigation: `{attack_type_str.title()}`\n\n"
        res += f"- **Total Identified Events**: **{count:,}**\n"
        res += f"- **Distinct Affected Users**: **{unique_users:,}**\n"
        res += f"- **Average Risk Score**: **{avg_risk:.1f} / 100** (Peak Risk: **{max_risk:.1f}**)\n\n"
        res += "#### 📋 Top Flagged Incidents Summary:\n\n"

        display_cols = [c for c in ["Timestamp", "User ID", "Source IP", "Country", "risk_score", "risk_level", "natural_language_explanation"] if c in matched_rows.columns]
        sample_df = matched_rows.head(5)[display_cols]
        res += _df_to_markdown(sample_df)
        res += "\n\n"

        # Action Recommendations
        norm_atk = [k for k in RECOMMENDED_ACTIONS.keys() if k.lower() == attack_type_str.lower()]
        key_atk = norm_atk[0] if norm_atk else "Default"
        res += f"#### 🛠️ Recommended Response Playbook for {attack_type_str.title()}:\n"
        for act in RECOMMENDED_ACTIONS[key_atk]:
            res += f"- {act}\n"

        return res

    def get_top_risk_users(self, top_n: int = 5) -> str:
        """Generates high-risk user leaderboard based on max risk score."""
        if "User ID" not in self.df.columns or "risk_score" not in self.df.columns:
            return "❌ User ID or risk score data not available."

        user_grp = self.df.groupby("User ID").agg(
            max_risk=("risk_score", "max"),
            mean_risk=("risk_score", "mean"),
            anom_count=("is_anomaly", "sum") if "is_anomaly" in self.df.columns else ("risk_score", "count"),
            primary_attack=("attack_type", lambda s: s[s != "Normal"].mode()[0] if not s[s != "Normal"].empty else "Normal") if "attack_type" in self.df.columns else ("risk_score", lambda x: "N/A")
        ).reset_index()

        user_grp = user_grp.sort_values(by="max_risk", ascending=False).head(top_n)

        res = f"### 👑 Top {top_n} Highest Risk Users Leaderboard\n\n"
        res += "| Rank | User ID | Peak Risk Score | Mean Risk | Flagged Anomalies | Primary Threat Vector |\n"
        res += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"

        for idx, (_, row) in enumerate(user_grp.iterrows(), 1):
            res += f"| **#{idx}** | `{row['User ID']}` | **{row['max_risk']:.1f} / 100** | {row['mean_risk']:.1f} | {int(row['anom_count'])} events | `{row['primary_attack']}` |\n"

        res += "\n💡 *Recommendation: Prioritize accounts with Peak Risk Score > 60.0 for immediate password reset and session revocation.*"
        return res

    def explain_anomaly(self, event_idx: Optional[int] = None) -> str:
        """Explains an individual anomaly event using SHAP attributions."""
        if self.df.empty:
            return "❌ Telemetry dataset is empty."

        if event_idx is not None and event_idx in self.df.index:
            row = self.df.loc[event_idx]
        else:
            # Pick highest risk anomaly event
            anom_df = self.df[self.df.get("is_anomaly", 0) == 1]
            if anom_df.empty:
                anom_df = self.df
            row = anom_df.loc[anom_df.get("risk_score", pd.Series([0])).idxmax()]
            event_idx = row.name

        user_id = row.get("User ID", "N/A")
        risk_score = row.get("risk_score", 0.0)
        risk_lvl = row.get("risk_level", "Low")
        atk_type = row.get("attack_type", "Anomaly")
        explanation = row.get("natural_language_explanation", "Anomalous behavior flagged.")
        top_feats = row.get("top_contributing_features", "N/A")

        res = f"### 🔎 Anomaly Investigation Report for Event `#{event_idx}`\n\n"
        res += f"- **Target User**: `{user_id}`\n"
        res += f"- **Classification**: `{atk_type}`\n"
        res += f"- **Calculated Risk Score**: **{risk_score:.1f} / 100** (`{risk_lvl}`)\n\n"
        res += f"#### 💬 SHAP Natural Language Explanation:\n> {explanation}\n\n"
        res += f"#### 📊 Top Feature Attributions:\n```text\n{top_feats}\n```\n\n"

        # Action Recommendations
        actions = RECOMMENDED_ACTIONS.get(atk_type, RECOMMENDED_ACTIONS["Default"])
        res += "#### 🛠️ Recommended Actions:\n"
        for act in actions:
            res += f"- {act}\n"

        return res

    def recommend_actions(self, attack_type: Optional[str] = None) -> str:
        """Provides SOC mitigation playbooks."""
        if attack_type:
            norm_atk = [k for k in RECOMMENDED_ACTIONS.keys() if k.lower() in attack_type.lower()]
            key = norm_atk[0] if norm_atk else "Default"
            acts = RECOMMENDED_ACTIONS[key]
            res = f"### 🛠️ Incident Response Playbook for `{key}`\n\n"
            for a in acts:
                res += f"- {a}\n"
            return res

        # Generic Overview Playbook
        res = "### 🛠️ Standard SOC Incident Response Guidelines\n\n"
        for atk, acts in RECOMMENDED_ACTIONS.items():
            if atk == "Default":
                continue
            res += f"#### 📌 {atk}\n"
            for a in acts:
                res += f"- {a}\n"
            res += "\n"
        return res

    def _grounded_fallback_response(self, query: str) -> str:
        """Grounded fallback for out-of-bounds queries."""
        res = f"🤖 **CyberGuard AI Copilot Assistance**\n\n"
        res += f"I analyzed your query: *\"{query}\"*\n\n"
        res += "As a strict, fact-grounded security assistant, I answer questions strictly based on active dataset predictions, SHAP attributions, and risk scores.\n\n"
        res += "💡 **Here are some queries you can ask me:**\n"
        res += "1. `Why was User USR-00923 flagged?`\n"
        res += "2. `Show all impossible travel attacks`\n"
        res += "3. `Which users have the highest risk?`\n"
        res += "4. `Explain this anomaly`\n"
        res += "5. `Recommend actions for Brute Force attacks`\n"
        return res
