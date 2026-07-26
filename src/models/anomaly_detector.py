"""
CyberGuard AI - Behavioral Anomaly Detection Model (Module 4)

Trains an unsupervised Isolation Forest model on normal user baseline records (is_anomaly == 0),
predicts anomaly scores & binary anomaly labels, evaluates model performance with Precision, Recall,
F1, ROC AUC, and Confusion Matrix, and exports diagnostic visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from config import config
from src.utils.logger import setup_logger

logger = setup_logger("BehavioralAnomalyDetector")

# Identifier columns to exclude from training features
NON_FEATURE_COLUMNS = [
    "Timestamp",
    "User ID",
    "Device ID",
    "Source IP",
    "Country",
    "City",
    "Login Success",
    "is_anomaly",
    "attack_type"
]


class BehavioralAnomalyDetector:
    """
    Unsupervised Isolation Forest model for cybersecurity threat & anomaly detection.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.03,
        random_state: int = 42,
        model_path: Optional[Path] = None
    ):
        """
        Initializes the Isolation Forest model wrapper.

        Args:
            n_estimators (int): Number of trees in isolation forest.
            contamination (float): Expected ratio of anomalies in telemetry.
            random_state (int): Random seed for reproducibility.
            model_path (Optional[Path]): File path to save/load trained model.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model_path = model_path or (config.dirs.saved_models_dir / "isolation_forest.joblib")

        self.model: IsolationForest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_columns: List[str] = []
        self.is_trained: bool = False

    def train(self, df: pd.DataFrame) -> IsolationForest:
        """
        Trains the Isolation Forest strictly on normal records (is_anomaly == 0).

        Args:
            df (pd.DataFrame): DataFrame containing features and 'is_anomaly' column.

        Returns:
            IsolationForest: Fitted model instance.
        """
        logger.info("Preparing training dataset...")
        
        # Determine feature columns
        self.feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
        logger.info(f"Identified {len(self.feature_columns)} feature columns for training.")

        # Filter ONLY normal records for training if is_anomaly exists
        if "is_anomaly" in df.columns:
            train_df = df[df["is_anomaly"] == 0][self.feature_columns]
            logger.info(
                f"Training ONLY on normal records (is_anomaly == 0): {len(train_df):,} rows "
                f"(out of {len(df):,} total)."
            )
        else:
            train_df = df[self.feature_columns]
            logger.info(f"Training on full dataset: {len(train_df):,} rows.")

        logger.info(
            f"Fitting IsolationForest (n_estimators={self.n_estimators}, "
            f"contamination={self.contamination}, seed={self.random_state})..."
        )
        self.model.fit(train_df)
        self.is_trained = True
        logger.info("IsolationForest training completed successfully.")

        # Save Model Artifact
        self.save_model()
        return self.model

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Saves trained model artifact to disk."""
        target_path = path or self.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns}, target_path)
        logger.info(f"Saved trained IsolationForest artifact to: {target_path}")
        return target_path

    def load_model(self, path: Optional[Path] = None) -> None:
        """Loads trained model artifact from disk."""
        target_path = path or self.model_path
        if not target_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {target_path}")

        logger.info(f"Loading trained IsolationForest artifact from: {target_path}")
        data = joblib.load(target_path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.is_trained = True
        logger.info(f"Loaded model successfully with {len(self.feature_columns)} feature columns.")

    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generates continuous anomaly scores and binary anomaly predictions.

        Args:
            df (pd.DataFrame): Dataset containing features.

        Returns:
            Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
                - DataFrame with anomaly_score & anomaly_prediction added.
                - Raw anomaly risk scores (numpy array).
                - Binary anomaly predictions (numpy array: 0 = Normal, 1 = Anomaly).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded before making predictions.")

        if not self.feature_columns:
            self.feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

        X = df[self.feature_columns]

        # Raw decision_function output: higher = normal, lower = anomaly
        raw_scores = self.model.decision_function(X)

        # Invert and normalize to [0, 1] range: 0.0 = completely normal, 1.0 = highly anomalous
        inverted_scores = -raw_scores
        score_min, score_max = inverted_scores.min(), inverted_scores.max()
        if score_max > score_min:
            normalized_scores = (inverted_scores - score_min) / (score_max - score_min)
        else:
            normalized_scores = np.zeros_like(inverted_scores)

        # Binary Predictions: Sklearn predict returns -1 for anomaly, 1 for normal
        raw_preds = self.model.predict(X)
        binary_predictions = (raw_preds == -1).astype(int)

        df_out = df.copy()
        df_out["anomaly_score"] = np.round(normalized_scores, 4)
        df_out["anomaly_prediction"] = binary_predictions

        logger.info(
            f"Predictions generated for {len(df_out):,} rows. "
            f"Flagged Anomalies: {binary_predictions.sum():,} ({(binary_predictions.mean()*100):.2f}%)."
        )
        return df_out, normalized_scores, binary_predictions

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates predictions against ground truth labels.

        Args:
            y_true (np.ndarray): True ground truth labels (0/1).
            y_pred (np.ndarray): Predicted binary labels (0/1).
            y_scores (np.ndarray): Continuous anomaly risk scores.

        Returns:
            Dict[str, Any]: Evaluation metrics dictionary.
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_scores)
        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }

        logger.info("==================================================")
        logger.info("Model Evaluation Metrics against Ground Truth:")
        logger.info(f"  - Precision  : {precision * 100:.2f}%")
        logger.info(f"  - Recall     : {recall * 100:.2f}%")
        logger.info(f"  - F1 Score   : {f1 * 100:.2f}%")
        logger.info(f"  - ROC AUC    : {roc_auc:.4f}")
        logger.info(f"  - Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
        logger.info("==================================================")

        return metrics

    def generate_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        output_dir: Path
    ) -> List[Path]:
        """
        Generates and saves the 4 required diagnostic visualization plots.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_scores: Continuous anomaly risk scores.
            output_dir: Folder path to save PNG images.

        Returns:
            List[Path]: List of saved plot file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = []

        # 1. ROC Curve
        roc_path = output_dir / "roc_curve.png"
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="#0055ff", lw=2, label=f"AUC = {auc_val:.4f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - CyberGuard AI Anomaly Detector")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.close()
        saved_plots.append(roc_path)

        # 2. Precision-Recall Curve
        pr_path = output_dir / "precision_recall_curve.png"
        precisions, recalls, _ = precision_recall_curve(y_true, y_scores)

        plt.figure(figsize=(7, 5))
        plt.plot(recalls, precisions, color="#00aa55", lw=2, label="Isolation Forest PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve - CyberGuard AI")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_path, dpi=300)
        plt.close()
        saved_plots.append(pr_path)

        # 3. Confusion Matrix Heatmap
        cm_path = output_dir / "confusion_matrix.png"
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Anomaly Detection")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Normal (0)", "Anomaly (1)"])
        plt.yticks(tick_marks, ["Normal (0)", "Anomaly (1)"])

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, f"{cm[i, j]:,}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold"
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        saved_plots.append(cm_path)

        # 4. Anomaly Score Distribution Histogram
        dist_path = output_dir / "anomaly_score_distribution.png"

        plt.figure(figsize=(8, 5))
        plt.hist(
            y_scores[y_true == 0], bins=50, alpha=0.6, color="#0066cc",
            label="Normal Telemetry (0)", density=True
        )
        plt.hist(
            y_scores[y_true == 1], bins=50, alpha=0.7, color="#cc0000",
            label="Attacks & Anomalies (1)", density=True
        )
        plt.xlabel("Normalized Anomaly Risk Score")
        plt.ylabel("Density")
        plt.title("Anomaly Score Distribution (Normal vs. Attack Telemetry)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(dist_path, dpi=300)
        plt.close()
        saved_plots.append(dist_path)

        logger.info(f"Generated {len(saved_plots)} diagnostic visualization charts in: {output_dir}")
        return saved_plots

    def save_predictions(self, df_predictions: pd.DataFrame, output_path: Path) -> Path:
        """Saves predictions dataframe to CSV."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving anomaly predictions to: {output_path}")
            df_predictions.to_csv(output_path, index=False, encoding="utf-8")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Successfully saved {len(df_predictions):,} prediction rows ({file_size_mb:.2f} MB) "
                f"to {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to save predictions to {output_path}: {e}", exc_info=True)
            raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluates Isolation Forest or classifier model against test set.

    Args:
        model: Trained scikit-learn model.
        X_test: Test feature matrix.
        y_test: Ground truth binary labels.

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, f1, and roc_auc metrics.
    """
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Invert score if IsolationForest (lower = more anomalous)
        scores = -scores
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.predict(X_test)

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, scores)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    }
