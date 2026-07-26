"""
CyberGuard AI - Unit Tests for Behavioral Anomaly Detection Model (Module 4)
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
from src.models.anomaly_detector import BehavioralAnomalyDetector

class TestBehavioralAnomalyDetector(unittest.TestCase):
    """Unit tests validating Isolation Forest training, scoring, evaluation, and plot rendering."""

    @classmethod
    def setUpClass(cls):
        """Generate test feature dataset and fit anomaly detector."""
        cls.raw_temp = Path("data/raw/test_base_ad_temp.csv")
        cls.attack_temp = Path("data/processed/test_attack_ad_temp.csv")
        cls.feat_temp = Path("data/processed/test_feat_ad_temp.csv")
        cls.model_temp = Path("saved_models/test_iforest_temp.joblib")
        cls.pred_temp = Path("data/predictions/test_preds_temp.csv")
        cls.reports_temp = Path("reports/test_figures_temp")

        # 1. Generate normal logs
        gen = SyntheticLogGenerator(num_users=15, num_devices=30, days=5, random_seed=42)
        df_base = gen.generate_normal_logs(num_records=400)
        gen.save_to_csv(df_base, cls.raw_temp)

        # 2. Inject attacks
        injector = AttackInjector(random_seed=42)
        df_loaded = injector.load_baseline_data(cls.raw_temp)
        df_attacks = injector.inject_attacks(df_loaded, attack_ratio=0.08)
        injector.save_dataset(df_attacks, cls.attack_temp)

        # 3. Engineer features
        scaler_temp = Path("saved_models/test_scaler_ad_temp.joblib")
        engineer = FeatureEngineer(scaler_path=scaler_temp)
        cls.df_features, _ = engineer.fit_transform(df_attacks)
        engineer.save_engineered_dataset(cls.df_features, cls.feat_temp)
        if scaler_temp.exists():
            scaler_temp.unlink()

        # 4. Fit Anomaly Detector
        cls.detector = BehavioralAnomalyDetector(
            n_estimators=50,
            contamination=0.08,
            random_state=42,
            model_path=cls.model_temp
        )
        cls.detector.train(cls.df_features)
        cls.df_preds, cls.scores, cls.preds = cls.detector.predict(cls.df_features)

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
        """Verify model sets is_trained to True upon completion."""
        self.assertTrue(self.detector.is_trained)

    def test_model_serialization(self):
        """Verify model is saved to disk and reloaded cleanly."""
        self.assertTrue(self.model_temp.exists())
        new_detector = BehavioralAnomalyDetector(model_path=self.model_temp)
        new_detector.load_model()
        self.assertTrue(new_detector.is_trained)
        self.assertEqual(len(new_detector.feature_columns), len(self.detector.feature_columns))

    def test_prediction_dataframe_columns(self):
        """Verify prediction dataframe contains anomaly_score and anomaly_prediction columns."""
        self.assertIn("anomaly_score", self.df_preds.columns)
        self.assertIn("anomaly_prediction", self.df_preds.columns)
        self.assertEqual(len(self.df_preds), len(self.df_features))

    def test_anomaly_score_range(self):
        """Verify normalized anomaly_score falls strictly within [0.0, 1.0]."""
        self.assertTrue((self.df_preds["anomaly_score"] >= 0.0).all())
        self.assertTrue((self.df_preds["anomaly_score"] <= 1.0).all())

    def test_evaluation_metrics(self):
        """Verify evaluation metrics calculation (Precision, Recall, F1, ROC AUC)."""
        y_true = self.df_features["is_anomaly"].values
        metrics = self.detector.evaluate(y_true, self.preds, self.scores)

        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertGreater(metrics["roc_auc"], 0.70)

    def test_generate_plots(self):
        """Verify all 4 required diagnostic PNG plots are generated."""
        y_true = self.df_features["is_anomaly"].values
        saved_plots = self.detector.generate_plots(y_true, self.preds, self.scores, self.reports_temp)

        self.assertEqual(len(saved_plots), 4)
        for p in saved_plots:
            self.assertTrue(p.exists())

    def test_save_predictions(self):
        """Verify saving predictions to CSV produces valid readable file."""
        saved_path = self.detector.save_predictions(self.df_preds, self.pred_temp)
        self.assertTrue(saved_path.exists())
        loaded_df = pd.read_csv(saved_path)
        self.assertEqual(len(loaded_df), len(self.df_features))
        self.assertIn("anomaly_score", loaded_df.columns)

if __name__ == "__main__":
    unittest.main()
