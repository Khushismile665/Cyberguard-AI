"""
CyberGuard AI - Explainable AI Engine (Module 7)

Utilizes SHAP (SHapley Additive exPlanations) to compute local feature attribution
scores for detected security anomalies. Generates plain-English natural language
threat explanations for SOC analysts and exports diagnostic SHAP plots.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import shap

from config import config
from src.utils.logger import setup_logger

logger = setup_logger("ThreatExplainer")

NON_FEATURE_COLUMNS = [
    "Timestamp",
    "User ID",
    "Device ID",
    "Source IP",
    "Country",
    "City",
    "Login Success",
    "is_anomaly",
    "attack_type",
    "anomaly_score",
    "anomaly_prediction",
    "predicted_attack_type",
    "classification_confidence",
    "risk_score",
    "risk_level"
]


class ThreatExplainer:
    """
    SHAP-based Explainable AI engine for security threats.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initializes ThreatExplainer.

        Args:
            model_path (Optional[Path]): File path to trained tree-based model artifact.
        """
        self.model_path = model_path or (config.dirs.saved_models_dir / "isolation_forest.joblib")
        self.model: Any = None
        self.feature_columns: List[str] = []
        self.explainer: Optional[shap.TreeExplainer] = None

        if self.model_path.exists():
            self.load_model(self.model_path)

    def load_model(self, path: Path) -> None:
        """Loads trained model and initializes TreeExplainer."""
        logger.info(f"Loading trained model artifact for SHAP explainer from: {path}")
        data = joblib.load(path)

        if isinstance(data, dict):
            self.model = data.get("model", data)
            self.feature_columns = data.get("feature_columns", [])
        else:
            self.model = data
            self.feature_columns = []

        logger.info("Initializing SHAP TreeExplainer...")
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.warning(f"TreeExplainer initialization warning: {e}. Falling back to shap.Explainer.")
            self.explainer = shap.Explainer(self.model)

    def compute_shap_values(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Computes raw Shapley attribution matrix for feature rows.

        Args:
            df_features (pd.DataFrame): Dataframe of features.

        Returns:
            np.ndarray: Shapley values matrix (rows x features).
        """
        if not self.feature_columns:
            self.feature_columns = [c for c in df_features.columns if c not in NON_FEATURE_COLUMNS]

        X = df_features[self.feature_columns]

        logger.info(f"Computing SHAP values for {len(X):,} rows across {len(self.feature_columns)} features...")
        shap_vals = self.explainer.shap_values(X)

        # Handle multi-class output (3D array: samples x features x classes)
        if isinstance(shap_vals, list):
            shap_vals = np.mean(np.array(shap_vals), axis=0)
        elif len(shap_vals.shape) == 3:
            shap_vals = np.mean(shap_vals, axis=-1)

        logger.info("SHAP computation completed.")
        return shap_vals

    def generate_natural_language_explanation(
        self,
        row: pd.Series,
        feature_names: List[str],
        shap_row: np.ndarray,
        top_k: int = 3
    ) -> Tuple[str, str]:
        """
        Generates natural language explanation bullets based on top SHAP attribution values.

        Args:
            row: Dataframe row containing feature values.
            feature_names: List of feature names.
            shap_row: Array of SHAP attribution values for this row.
            top_k: Number of top features to include in report.

        Returns:
            Tuple[str, str]: (Top contributing features string, Natural language explanation string).
        """
        # Sort features by absolute SHAP contribution value
        abs_shap = np.abs(shap_row)
        top_indices = np.argsort(abs_shap)[::-1][:top_k]

        top_feature_pairs = []
        explanation_bullets = []

        for idx in top_indices:
            feat_name = feature_names[idx]
            shap_val = shap_row[idx]
            top_feature_pairs.append(f"{feat_name} ({shap_val:+.3f})")

            # Domain Rule Mappings based on Feature & Values
            if "is_new_location" in feat_name and row.get("is_new_location", 0) == 1:
                explanation_bullets.append("User logged in from a new or unfamiliar country.")
            elif "unique_countries" in feat_name and row.get("unique_countries_24h_scaled", 0) > 0.5:
                explanation_bullets.append("User logged in from multiple distinct countries within 24 hours.")
            elif "is_new_device" in feat_name and row.get("is_new_device", 0) == 1:
                explanation_bullets.append("Unknown or un-fingerprinted device detected.")
            elif "Browser_HeadlessChrome" in feat_name and row.get(feat_name, 0) == 1:
                explanation_bullets.append("Automated headless browser detected during authentication.")
            elif "Operating System_Unknown OS" in feat_name and row.get(feat_name, 0) == 1:
                explanation_bullets.append("Unregistered or conflicting Operating System detected.")
            elif "login_velocity" in feat_name and row.get("login_velocity_kmh_scaled", 0) > 0.5:
                explanation_bullets.append("Risk increased due to impossible travel velocity.")
            elif "geo_distance" in feat_name and row.get("geo_distance_km_scaled", 0) > 0.5:
                explanation_bullets.append("Abnormally large geographic distance between consecutive logins.")
            elif "failed_login" in feat_name and row.get("failed_login_count_1h_scaled", 0) > 0.5:
                explanation_bullets.append("Multiple failed login attempts detected within a short window.")
            elif "Resource Accessed_" in feat_name and row.get(feat_name, 0) == 1 and ("admin" in feat_name or "git" in feat_name or "finance" in feat_name):
                explanation_bullets.append("Unauthorized access attempt to a restricted corporate asset.")
            elif "login_hour" in feat_name or "hour_" in feat_name:
                hour = row.get("login_hour", 12)
                if hour >= 0 and hour <= 5:
                    explanation_bullets.append("Login occurred outside normal working hours.")
            elif "session_duration" in feat_name and row.get("session_duration_scaled", 0) < -0.5:
                explanation_bullets.append("Abnormally brief session duration detected.")

        # Fallback default explanations if rule triggers are empty
        if not explanation_bullets:
            explanation_bullets.append("Anomalous behavioral feature combination detected.")
            explanation_bullets.append(f"Primary threat factor: {feature_names[top_indices[0]]}.")

        # Remove duplicates while preserving order
        unique_bullets = list(dict.fromkeys(explanation_bullets))

        top_features_str = " | ".join(top_feature_pairs)
        explanation_str = " ".join(unique_bullets)

        return top_features_str, explanation_str

    def explain_dataset(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Computes SHAP values and generates natural language explanations for dataset rows.

        Args:
            df (pd.DataFrame): Dataset dataframe.
            sample_size (Optional[int]): Optional sample limit for fast processing.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Dataframe with explanation columns & SHAP values matrix.
        """
        logger.info("Starting dataset threat explanation pipeline...")
        df_out = df.copy()

        if sample_size and sample_size < len(df_out):
            logger.info(f"Sampling {sample_size:,} records for SHAP calculation...")
            df_out = df_out.sample(n=sample_size, random_state=42).sort_index().reset_index(drop=True)

        if not self.feature_columns:
            self.feature_columns = [c for c in df_out.columns if c not in NON_FEATURE_COLUMNS]

        shap_matrix = self.compute_shap_values(df_out)

        top_features_list = []
        explanation_list = []

        logger.info("Generating natural language explanations for dataset rows...")
        for i in range(len(df_out)):
            row = df_out.iloc[i]
            shap_row = shap_matrix[i]
            top_feats, exp_str = self.generate_natural_language_explanation(
                row, self.feature_columns, shap_row, top_k=3
            )
            top_features_list.append(top_feats)
            explanation_list.append(exp_str)

        df_out["top_contributing_features"] = top_features_list
        df_out["natural_language_explanation"] = explanation_list

        logger.info(f"Generated explanations for {len(df_out):,} records.")
        return df_out, shap_matrix

    def generate_shap_plots(
        self,
        df_features: pd.DataFrame,
        shap_matrix: np.ndarray,
        output_dir: Path
    ) -> List[Path]:
        """
        Generates SHAP summary, feature importance, and decision force plots.

        Args:
            df_features: Feature dataframe.
            shap_matrix: Calculated SHAP values matrix.
            output_dir: Folder to save PNG plots.

        Returns:
            List[Path]: Paths of saved plot images.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = []

        X = df_features[self.feature_columns]

        # 1. SHAP Summary Beeswarm Plot
        sum_path = output_dir / "shap_summary_plot.png"
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_matrix, X, show=False, max_display=12)
        plt.title("SHAP Summary Plot - Feature Impact on Threat Score", fontsize=12)
        plt.tight_layout()
        plt.savefig(sum_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_plots.append(sum_path)

        # 2. SHAP Feature Importance Bar Plot
        bar_path = output_dir / "shap_feature_importance.png"
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_matrix, X, plot_type="bar", show=False, max_display=12)
        plt.title("SHAP Mean Absolute Feature Importance", fontsize=12)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_plots.append(bar_path)

        # 3. SHAP Decision / Force Plot for Top Anomalies
        force_path = output_dir / "shap_force_plot.png"
        plt.figure(figsize=(10, 6))

        # Select top 50 anomalies for clear decision plot
        if "is_anomaly" in df_features.columns:
            anomaly_indices = np.where(df_features["is_anomaly"] == 1)[0][:50]
        else:
            anomaly_indices = np.arange(min(50, len(df_features)))

        if len(anomaly_indices) > 0:
            sample_shap = shap_matrix[anomaly_indices]
            sample_X = X.iloc[anomaly_indices]
            expected_val = (
                self.explainer.expected_value
                if not isinstance(self.explainer.expected_value, np.ndarray)
                else self.explainer.expected_value[0]
            )
            shap.decision_plot(expected_val, sample_shap, sample_X, show=False)
            plt.title("SHAP Decision Plot - Top Threat Anomaly Trajectories", fontsize=12)
            plt.tight_layout()
            plt.savefig(force_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_plots.append(force_path)

        logger.info(f"Generated {len(saved_plots)} diagnostic SHAP plots in: {output_dir}")
        return saved_plots

    def save_explanation_reports(self, df_reports: pd.DataFrame, output_path: Path) -> Path:
        """Saves explainable anomaly reports dataset to CSV."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving explainable anomaly reports to: {output_path}")
            df_reports.to_csv(output_path, index=False, encoding="utf-8")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Successfully saved {len(df_reports):,} report rows ({file_size_mb:.2f} MB) "
                f"to {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to save explanation reports to {output_path}: {e}", exc_info=True)
            raise
