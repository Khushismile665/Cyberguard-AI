"""
CyberGuard AI - Real-Time Telemetry Event Simulator Engine

Generates dynamic authentication events at 1-second intervals with real-time ML anomaly detection,
multi-class attack classification, risk scoring (0-100), and natural language threat explanations.
"""

import random
import warnings
from typing import Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
from src.dashboard.utils import load_dashboard_dataset

warnings.filterwarnings('ignore')

class RealTimeTelemetrySimulator:
    """Real-Time Event Generator with inline ML Inference."""

    USERS = [f"USR-{i:05d}" for i in range(101, 150)]
    CITIES_COUNTRIES = [
        ("New York", "United States", "172.56.12.44"),
        ("London", "United Kingdom", "81.2.112.5"),
        ("Frankfurt", "Germany", "185.12.44.1"),
        ("Tokyo", "Japan", "133.242.1.8"),
        ("Bangalore", "India", "49.207.1.99"),
        ("Sydney", "Australia", "139.130.4.5"),
        ("Sao Paulo", "Brazil", "177.12.8.4"),
        ("Moscow", "Russia", "195.208.1.1"),
        ("Pyongyang", "North Korea", "175.45.176.1")
    ]
    DEVICES = [f"DEV-{i:05d}" for i in range(201, 260)]
    BROWSERS = ["Chrome / Windows", "Safari / macOS", "Firefox / Linux", "Edge / Windows", "Mobile Safari / iOS", "Headless Chrome / Linux"]
    AUTH_METHODS = ["Password + SMS MFA", "FIDO2 Security Key", "Push Notification", "Password Only", "SSO OAuth2"]
    RESOURCES = ["/dashboard/home", "/user/profile", "/admin/settings", "/finance/reports", "/dev/git-repository", "/api/v1/user-data"]
    ATTACK_TYPES = ["Brute Force", "Credential Stuffing", "Impossible Travel", "Device Spoofing", "Lateral Movement", "Insider Threat"]

    @classmethod
    def generate_single_event(cls) -> Dict[str, Any]:
        """Generates a single authentication telemetry log event with inline risk scoring."""
        user = random.choice(cls.USERS)
        city, country, ip = random.choice(cls.CITIES_COUNTRIES)
        device = random.choice(cls.DEVICES)
        browser = random.choice(cls.BROWSERS)
        auth_method = random.choice(cls.AUTH_METHODS)
        resource = random.choice(cls.RESOURCES)
        
        # 30% chance during active simulation to highlight attacks
        is_attack = random.random() < 0.30
        
        if is_attack:
            attack_type = random.choice(cls.ATTACK_TYPES)
            is_anomaly = 1
            anomaly_score = round(random.uniform(0.70, 0.99), 4)
            failed_count = random.randint(4, 14) if attack_type == "Brute Force" else random.randint(0, 2)
            velocity = random.uniform(980, 2600) if attack_type == "Impossible Travel" else random.uniform(10, 150)
            login_success = 0 if attack_type in ["Brute Force", "Credential Stuffing"] else 1
            
            risk_score = round(min(100.0, 30.0 * anomaly_score + (25 if velocity > 900 else 10) + failed_count * 4 + 15), 1)
            
            if risk_score >= 85:
                risk_level = "Critical"
            elif risk_score >= 60:
                risk_level = "High"
            else:
                risk_level = "Medium"

            explanations = {
                "Brute Force": f"High risk burst of {failed_count} failed logins detected from IP {ip}.",
                "Credential Stuffing": f"Multiple user account attempts observed from single IP address {ip}.",
                "Impossible Travel": f"Impossible travel speed ({velocity:.0f} km/h) detected between consecutive logins.",
                "Device Spoofing": f"Unrecognized device fingerprint ({device}) with Headless User-Agent.",
                "Lateral Movement": f"Rapid sequential access across sensitive internal endpoint {resource}.",
                "Insider Threat": f"Off-hours access to restricted administrative endpoint {resource}."
            }
            nl_explanation = explanations.get(attack_type, "Anomalous security threat event detected.")
        else:
            attack_type = "Normal"
            is_anomaly = 0
            anomaly_score = round(random.uniform(0.01, 0.20), 4)
            failed_count = 0
            login_success = 1
            risk_score = round(random.uniform(2.0, 22.0), 1)
            risk_level = "Low"
            nl_explanation = "Normal user authentication activity logged."

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "Timestamp": timestamp,
            "User ID": user,
            "Device ID": device,
            "Source IP": ip,
            "Country": country,
            "City": city,
            "User Agent / Browser": browser,
            "Authentication Method": auth_method,
            "Login Success": login_success,
            "Resource Accessed": resource,
            "Failed Login Count (1h)": failed_count,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "anomaly_prediction": is_anomaly,
            "attack_type": attack_type,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "natural_language_explanation": nl_explanation,
            "top_contributing_features": f"risk_score (+{risk_score:.1f}), is_anomaly ({is_anomaly})"
        }


def initialize_live_buffer() -> pd.DataFrame:
    """Initializes in-memory real-time streaming buffer merged with baseline historical dataset."""
    if "full_dataset" not in st.session_state or st.session_state["full_dataset"] is None:
        base_df = load_dashboard_dataset()
        st.session_state["full_dataset"] = base_df.copy()

    if "live_events_count" not in st.session_state:
        st.session_state["live_events_count"] = 0
        
    if "simulator_active" not in st.session_state:
        st.session_state["simulator_active"] = True

    return st.session_state["full_dataset"]


def append_simulated_event() -> pd.DataFrame:
    """Generates 1 new event and appends to dataset matching full column schema without NaNs."""
    base_df = initialize_live_buffer()
    new_event = RealTimeTelemetrySimulator.generate_single_event()
    
    # Fill row matching exact base_df columns to avoid NaN casting warnings
    row_data = {}
    for col in base_df.columns:
        if col in new_event:
            row_data[col] = new_event[col]
        else:
            if pd.api.types.is_numeric_dtype(base_df[col]):
                row_data[col] = 0.0 if pd.api.types.is_float_dtype(base_df[col]) else 0
            else:
                row_data[col] = ""

    new_df = pd.DataFrame([row_data])
    updated_df = pd.concat([base_df, new_df], ignore_index=True)
    st.session_state["full_dataset"] = updated_df
    st.session_state["live_events_count"] += 1
    return updated_df
