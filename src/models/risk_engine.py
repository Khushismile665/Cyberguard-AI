"""
CyberGuard AI - Risk Scoring Engine (Module 6)

Computes an explainable, multi-factor risk score (0 to 100) and risk severity level
(Low, Medium, High, Critical) based on failed logins, device novelty, location novelty,
impossible travel velocity, sensitive resource access, off-hours timing, and anomaly scores.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from config import config
from src.utils.logger import setup_logger

logger = setup_logger("RiskScoringEngine")

RESTRICTED_ENDPOINTS = [
    "/admin/settings",
    "/dev/git-repository",
    "/finance/reports"
]


class RiskScoringEngine:
    """
    Explainable risk scoring engine for cybersecurity authentication events.
    """

    def __init__(self):
        """Initializes RiskScoringEngine."""
        logger.info("Initialized RiskScoringEngine.")

    def calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates sub-scores, sums total risk score (0-100), and assigns risk level tier.

        Args:
            df (pd.DataFrame): Dataset with features, anomaly scores, and predictions.

        Returns:
            pd.DataFrame: Dataframe with 'risk_score' and 'risk_level' columns added.
        """
        logger.info("Starting multi-factor risk score calculation...")
        df_out = df.copy()

        # 1. Isolation Forest Anomaly Score Contribution (0 - 30 pts)
        if "anomaly_score" in df_out.columns:
            sub_anomaly = df_out["anomaly_score"] * 30.0
        else:
            sub_anomaly = np.zeros(len(df_out))

        # 2. Impossible Travel Velocity Contribution (0 - 25 pts)
        # Check raw or scaled velocity column
        sub_velocity = np.zeros(len(df_out))
        if "login_velocity_kmh" in df_out.columns:
            vel = df_out["login_velocity_kmh"].values
        elif "login_velocity_kmh_scaled" in df_out.columns:
            # Unscale approximation if scaled feature is present
            vel = df_out["login_velocity_kmh_scaled"].values * 300.0
        else:
            vel = np.zeros(len(df_out))

        sub_velocity = np.where(vel > 900.0, 25.0, np.where(vel > 400.0, 15.0, np.where(vel > 150.0, 8.0, 0.0)))

        # 3. Failed Login Bursts Contribution (0 - 20 pts)
        if "failed_login_count_1h" in df_out.columns:
            failed_counts = df_out["failed_login_count_1h"].values
        elif "failed_login_count_1h_scaled" in df_out.columns:
            failed_counts = np.maximum(0, df_out["failed_login_count_1h_scaled"].values * 3.0)
        else:
            failed_counts = np.zeros(len(df_out))

        sub_failed = np.minimum(20.0, failed_counts * 5.0)

        # 4. Sensitive Resource Access Contribution (0 - 15 pts)
        sub_resource = np.zeros(len(df_out))
        if "Resource Accessed" in df_out.columns:
            sub_resource = np.where(df_out["Resource Accessed"].isin(RESTRICTED_ENDPOINTS), 15.0, 0.0)
        else:
            # Check One-Hot encoded resource columns
            res_mask = np.zeros(len(df_out), dtype=bool)
            for res_col in df_out.columns:
                if "Resource Accessed_" in res_col:
                    for restricted in RESTRICTED_ENDPOINTS:
                        if restricted in res_col:
                            res_mask |= (df_out[res_col] == 1.0)
            sub_resource = np.where(res_mask, 15.0, 0.0)

        # 5. Device Novelty Contribution (0 - 5 pts)
        sub_device = np.where(df_out.get("is_new_device", 0) == 1, 5.0, 0.0)

        # 6. Location Novelty Contribution (0 - 5 pts)
        sub_location = np.where(df_out.get("is_new_location", 0) == 1, 5.0, 0.0)

        # 7. Off-Hours Time Anomaly Contribution (0 - 5 pts)
        sub_time = np.zeros(len(df_out))
        if "login_hour" in df_out.columns:
            hours = df_out["login_hour"].values
            sub_time = np.where((hours >= 0) & (hours <= 5), 5.0, 0.0)

        # Sum total risk score
        raw_risk_score = (
            sub_anomaly +
            sub_velocity +
            sub_failed +
            sub_resource +
            sub_device +
            sub_location +
            sub_time
        )

        # Clamp strictly between 0 and 100
        total_risk_score = np.clip(np.round(raw_risk_score, 1), 0.0, 100.0)

        # Categorize into Risk Level Tiers
        risk_levels = []
        for score in total_risk_score:
            if score >= 85.0:
                risk_levels.append("Critical")
            elif score >= 60.0:
                risk_levels.append("High")
            elif score >= 30.0:
                risk_levels.append("Medium")
            else:
                risk_levels.append("Low")

        df_out["risk_score"] = total_risk_score
        df_out["risk_level"] = risk_levels

        logger.info("Risk score calculation complete.")
        logger.info("Risk Level Distribution:")
        for level, count in pd.Series(risk_levels).value_counts().items():
            logger.info(f"  - {level:10s}: {count:6,d} ({(count / len(df_out) * 100):.2f}%)")

        return df_out

    def save_results(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Saves results dataframe to CSV."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving risk scoring results to: {output_path}")
            df.to_csv(output_path, index=False, encoding="utf-8")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Successfully saved {len(df):,} risk-scored records ({file_size_mb:.2f} MB) "
                f"to {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to save risk scoring results to {output_path}: {e}", exc_info=True)
            raise
