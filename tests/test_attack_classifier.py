"""
CyberGuard AI - Unit Tests for Attack Classifier (Module 5)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.log_generator import SyntheticLogGenerator
from src.data.attack_injector import AttackInjector
from src.data.feature_engineering import FeatureEngineer
from src.models.attack_classifier import AttackClassifier

class TestAttackClassifier(unittest.TestCase):
    """Unit tests validating multi-class attack classification, metrics, feature importances, and plots."""

    @classmethod
    def setUpClass(cls):
        """Generate test feature dataset and fit attack classifier."""
        cls.raw_temp = Path("data/raw/test_base_cls_temp.csv")
        cls.attack_temp = Path("data/processed/test_attack_cls_temp.csv")
        cls.feat_temp = Path("data/processed/test_feat_cls_temp.csv")
        cls.model_temp = Path("saved_models/test_cls_model_temp.joblib")
        cls.pred_temp = Path("data/predictions/test_cls_preds_temp.csv")
        cls.reports_temp = Path("reports/test_cls_figures_temp")

        # 1. Generate normal logs
        gen = SyntheticLogGenerator(num_users=20, num_devices=40, days=7, random_seed=42)
        df_base = gen.generate_normal_logs(num_records=600)
        gen.save_to_csv(df_base, cls.raw_temp)

        # 2. Inject attacks (10% attack ratio for rich test sample)
        injector = AttackInjector(random_seed=42)
        df_loaded = injector.load_baseline_data(cls.raw_temp)
        df_attacks = injector.inject_attacks(df_loaded, attack_ratio=0.10)
        injector.save_dataset(df_attacks, cls.attack_temp)

        # 3. Engineer features
        scaler_temp = Path("saved_models/test_scaler_cls_temp.joblib")
        engineer = FeatureEngineer(scaler_path=scaler_temp)
        cls.df_features, _ = engineer.fit_transform(df_attacks)
        engineer.save_engineered_dataset(cls.df_features, cls.feat_temp)
        if scaler_temp.exists():
            scaler_temp.unlink()

        # 4. Fit Attack Classifier
        cls.classifier = AttackClassifier(random_state=42, model_path=cls.model_temp)
        cls.model, cls.metrics = cls.classifier.train(cls.df_features, test_size=0.25)
        cls.df_preds, cls.pred_classes, cls.confidences = cls.classifier.predict(cls.df_features)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        for path in [
            cls.raw_temp, cls.attack_temp, cls.feat_temp,
            cls.model_temp, cls.pred_temp
        ]:
            if path.exists():
                path.unlink()

        if cls.reports_temp.exists():
            for f in cls.reports_temp.glob("*.png"):
                f.unlink()
            cls.reports_temp.rmdir()

    def test_model_training_flag(self):
        """Verify classifier is_trained flag is set to True upon completion."""
        self.assertTrue(self.classifier.is_trained)

    def test_model_serialization(self):
        """Verify classifier and LabelEncoder are saved and reloaded cleanly."""
        self.assertTrue(self.model_temp.exists())
        new_classifier = AttackClassifier(model_path=self.model_temp)
        new_classifier.load_model()
        self.assertTrue(new_classifier.is_trained)
        self.assertEqual(len(new_classifier.label_encoder.classes_), 6)

    def test_classification_accuracy(self):
        """Verify multi-class classification accuracy exceeds 0.85 threshold."""
        self.assertGreater(self.metrics["accuracy"], 0.85)
        self.assertGreater(self.metrics["f1_macro"], 0.85)

    def test_prediction_dataframe_columns(self):
        """Verify prediction output dataframe contains required columns."""
        self.assertIn("predicted_attack_type", self.df_preds.columns)
        self.assertIn("classification_confidence", self.df_preds.columns)
        self.assertEqual(len(self.df_preds), len(self.df_features))

    def test_feature_importances(self):
        """Verify feature importances dataframe extraction."""
        df_imp = self.classifier.get_feature_importances(top_n=10)
        self.assertEqual(len(df_imp), 10)
        self.assertIn("feature", df_imp.columns)
        self.assertIn("importance", df_imp.columns)
        self.assertTrue((df_imp["importance"] >= 0.0).all())

    def test_generate_plots(self):
        """Verify generation of confusion matrix and feature importance PNG charts."""
        attack_mask = self.df_features["is_anomaly"] == 1
        df_attacks = self.df_features[attack_mask]
        y_true_enc = self.classifier.label_encoder.transform(df_attacks["attack_type"].values)
        y_pred_enc = self.classifier.label_encoder.transform(
            self.df_preds.loc[attack_mask, "predicted_attack_type"].values
        )

        saved_plots = self.classifier.generate_plots(y_true_enc, y_pred_enc, self.reports_temp)
        self.assertEqual(len(saved_plots), 2)
        for p in saved_plots:
            self.assertTrue(p.exists())

    def test_save_predictions(self):
        """Verify saving prediction CSV writes readable file matching dataset rows."""
        saved_path = self.classifier.save_predictions(self.df_preds, self.pred_temp)
        self.assertTrue(saved_path.exists())
        loaded_df = pd.read_csv(saved_path)
        self.assertEqual(len(loaded_df), len(self.df_features))
        self.assertIn("predicted_attack_type", loaded_df.columns)

if __name__ == "__main__":
    unittest.main()
