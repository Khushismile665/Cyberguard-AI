"""
CyberGuard AI - Attack Classifier (Module 5)

Classifies detected anomaly records into specific cyber attack categories:
- Brute Force
- Credential Stuffing
- Impossible Travel
- Device Spoofing
- Lateral Movement
- Insider Threat

Dynamically utilizes XGBoost Classifier if available, falling back seamlessly
to RandomForestClassifier. Computes multi-class evaluation metrics, feature importances,
and outputs diagnostic visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Attempt XGBoost Import with Fallback to RandomForest
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

from config import config
from src.utils.logger import setup_logger

logger = setup_logger("AttackClassifier")

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
    "anomaly_prediction"
]

ATTACK_CLASSES = [
    "Brute Force",
    "Credential Stuffing",
    "Device Spoofing",
    "Impossible Travel",
    "Insider Threat",
    "Lateral Movement"
]


class AttackClassifier:
    """
    Multi-class Machine Learning model for classifying cyber attack vectors.
    """

    def __init__(
        self,
        random_state: int = 42,
        model_path: Optional[Path] = None
    ):
        """
        Initializes AttackClassifier.

        Args:
            random_state (int): Random seed for reproducibility.
            model_path (Optional[Path]): Destination path for model serialization.
        """
        self.random_state = random_state
        self.model_path = model_path or (config.dirs.saved_models_dir / "attack_classifier.joblib")

        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.is_trained: bool = False
        self.use_xgboost = XGBOOST_AVAILABLE

        if self.use_xgboost:
            logger.info("XGBoost library detected. Initializing XGBClassifier.")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric="mlogloss",
                n_jobs=-1
            )
        else:
            logger.info("XGBoost not detected. Falling back to RandomForestClassifier.")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1
            )

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Trains the classifier strictly on anomaly records (is_anomaly == 1 / attack_type != 'Normal').

        Args:
            df (pd.DataFrame): Dataset containing features and ground truth 'attack_type'.
            test_size (float): Stratified test split fraction for evaluation.

        Returns:
            Tuple[Any, Dict[str, Any]]: Fitted model and test evaluation metrics.
        """
        logger.info("Filtering dataset to isolate anomaly records (attack_type != 'Normal')...")
        
        # Filter ONLY attack/anomaly records
        if "is_anomaly" in df.columns:
            attack_df = df[df["is_anomaly"] == 1].copy()
        elif "attack_type" in df.columns:
            attack_df = df[df["attack_type"] != "Normal"].copy()
        else:
            attack_df = df.copy()

        logger.info(f"Isolated {len(attack_df):,} anomaly records for multi-class training.")

        if len(attack_df) == 0:
            raise ValueError("No anomaly records found in dataset to train AttackClassifier.")

        self.feature_columns = [c for c in attack_df.columns if c not in NON_FEATURE_COLUMNS]
        logger.info(f"Identified {len(self.feature_columns)} feature columns for classification.")

        X = attack_df[self.feature_columns]
        y_labels = attack_df["attack_type"].values

        # Encode String Target Labels into Integers (0..N-1)
        y_encoded = self.label_encoder.fit_transform(y_labels)
        logger.info(f"Label Encoder mapped classes: {list(self.label_encoder.classes_)}")

        # Perform 80/20 Stratified Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )

        logger.info(
            f"Fitting {self.model.__class__.__name__} on {len(X_train):,} training anomaly samples..."
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on Test Split
        y_test_pred = self.model.predict(X_test)
        y_test_prob = self.model.predict_proba(X_test)
        metrics = self.evaluate(y_test, y_test_pred, y_test_prob)

        # Save Model Artifact
        self.save_model()
        return self.model, metrics

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Saves trained classifier and LabelEncoder to disk."""
        target_path = path or self.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "feature_columns": self.feature_columns,
                "use_xgboost": self.use_xgboost
            },
            target_path
        )
        logger.info(f"Saved trained AttackClassifier artifact to: {target_path}")
        return target_path

    def load_model(self, path: Optional[Path] = None) -> None:
        """Loads trained classifier and LabelEncoder from disk."""
        target_path = path or self.model_path
        if not target_path.exists():
            raise FileNotFoundError(f"Classifier artifact not found at: {target_path}")

        logger.info(f"Loading trained AttackClassifier artifact from: {target_path}")
        data = joblib.load(target_path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.feature_columns = data["feature_columns"]
        self.use_xgboost = data.get("use_xgboost", False)
        self.is_trained = True
        logger.info(f"Loaded classifier successfully with {len(self.feature_columns)} feature columns.")

    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Predicts attack category and classification confidence for dataset rows.

        Args:
            df (pd.DataFrame): Dataset with features.

        Returns:
            Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
                - DataFrame with predicted_attack_type and classification_confidence added.
                - Array of predicted string attack types.
                - Array of float confidence scores.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded before making predictions.")

        if not self.feature_columns:
            self.feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

        X = df[self.feature_columns]

        # Predict Class Probabilities
        probs = self.model.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        # Decode Class Names
        predicted_classes = self.label_encoder.inverse_transform(pred_indices)

        # For normal non-anomaly rows (if predicted as normal in baseline), maintain 'Normal'
        df_out = df.copy()
        df_out["predicted_attack_type"] = predicted_classes
        df_out["classification_confidence"] = np.round(confidences, 4)

        if "is_anomaly" in df_out.columns:
            # If is_anomaly == 0, override prediction to 'Normal'
            normal_mask = df_out["is_anomaly"] == 0
            df_out.loc[normal_mask, "predicted_attack_type"] = "Normal"
            df_out.loc[normal_mask, "classification_confidence"] = 1.0

        logger.info(f"Attack classification complete for {len(df_out):,} rows.")
        return df_out, predicted_classes, confidences

    def evaluate(
        self,
        y_true_encoded: np.ndarray,
        y_pred_encoded: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates multi-class classification performance.

        Args:
            y_true_encoded: Ground truth encoded labels.
            y_pred_encoded: Predicted encoded labels.
            y_prob: Predicted probability array.

        Returns:
            Dict[str, Any]: Classification metrics dictionary.
        """
        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        prec_macro = precision_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0)
        rec_macro = recall_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0)
        f1_macro = f1_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0)

        prec_weighted = precision_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0)

        cm = confusion_matrix(y_true_encoded, y_pred_encoded)

        metrics = {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(prec_weighted),
            "recall_weighted": float(rec_weighted),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix": cm.tolist()
        }

        logger.info("==================================================")
        logger.info("Attack Classifier Multi-Class Evaluation Metrics:")
        logger.info(f"  - Accuracy          : {acc * 100:.2f}%")
        logger.info(f"  - Macro Precision   : {prec_macro * 100:.2f}%")
        logger.info(f"  - Macro Recall      : {rec_macro * 100:.2f}%")
        logger.info(f"  - Macro F1 Score    : {f1_macro * 100:.2f}%")
        logger.info(f"  - Weighted F1 Score : {f1_weighted * 100:.2f}%")
        logger.info("==================================================")

        return metrics

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extracts feature importances from the trained model.

        Args:
            top_n (int): Number of top features to return.

        Returns:
            pd.DataFrame: Dataframe of top features and importance scores.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before extracting feature importances.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.zeros(len(self.feature_columns))

        df_imp = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp.head(top_n)

    def generate_plots(
        self,
        y_true_encoded: np.ndarray,
        y_pred_encoded: np.ndarray,
        output_dir: Path
    ) -> List[Path]:
        """
        Generates feature importance and multi-class confusion matrix plots.

        Args:
            y_true_encoded: Ground truth encoded labels.
            y_pred_encoded: Predicted encoded labels.
            output_dir: Output folder path.

        Returns:
            List[Path]: Paths of saved plot images.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = []

        class_names = list(self.label_encoder.classes_)

        # 1. Multi-Class Confusion Matrix
        cm_path = output_dir / "attack_classifier_confusion_matrix.png"
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Attack Classification Confusion Matrix ({self.model.__class__.__name__})")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, f"{cm[i, j]:,}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10, fontweight="bold"
                )

        plt.ylabel("True Attack Category")
        plt.xlabel("Predicted Attack Category")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        saved_plots.append(cm_path)

        # 2. Top 15 Feature Importances Bar Chart
        imp_path = output_dir / "feature_importance_classifier.png"
        df_top = self.get_feature_importances(top_n=15)

        plt.figure(figsize=(10, 6))
        plt.barh(df_top["feature"][::-1], df_top["importance"][::-1], color="#0066cc")
        plt.xlabel("Relative Feature Importance Score")
        plt.ylabel("Engineered ML Feature")
        plt.title(f"Top 15 Feature Importances - Attack Classifier ({self.model.__class__.__name__})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(imp_path, dpi=300)
        plt.close()
        saved_plots.append(imp_path)

        logger.info(f"Generated {len(saved_plots)} classification diagnostic plots in: {output_dir}")
        return saved_plots

    def save_predictions(self, df_predictions: pd.DataFrame, output_path: Path) -> Path:
        """Saves predictions dataframe to CSV."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving attack classification predictions to: {output_path}")
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
