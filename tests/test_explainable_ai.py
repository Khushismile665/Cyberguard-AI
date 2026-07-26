"""
CyberGuard AI - Unit Tests for Explainable AI Engine (Module 7)
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
from src.models.explainable_ai import ThreatExplainer

class TestThreatExplainer(unittest.TestCase):
    """Unit tests validating SHAP calculations, natural language explanations, and plot generation."""

    @classmethod
    def setUpClass(cls):
        """Generate sample predictions dataset and fit SHAP explainer."""
        cls.raw_temp = Path("data/raw/test_base_xai_temp.csv")
        cls.attack_temp = Path("data/processed/test_attack_xai_temp.csv")
        cls.feat_temp = Path("data/processed/test_feat_xai_temp.csv")
        cls.model_temp = Path("saved_models/test_iforest_xai_temp.joblib")
        cls.report_output_temp = Path("data/predictions/test_xai_report_temp.csv")
        cls.figures_temp = Path("reports/test_xai_figures_temp")

        # 1. Generate normal logs
        gen = SyntheticLogGenerator(num_users=15, num_devices=30, days=5, random_seed=42)
        df_base = gen.generate_normal_logs(num_records=300)
        gen.save_to_csv(df_base, cls.raw_temp)

        # 2. Inject attacks
        injector = AttackInjector(random_seed=42)
        df_loaded = injector.load_baseline_data(cls.raw_temp)
        df_attacks = injector.inject_attacks(df_loaded, attack_ratio=0.10)
        injector.save_dataset(df_attacks, cls.attack_temp)

        # 3. Engineer features
        scaler_temp = Path("saved_models/test_scaler_xai_temp.joblib")
        engineer = FeatureEngineer(scaler_path=scaler_temp)
        cls.df_features, _ = engineer.fit_transform(df_attacks)
        engineer.save_engineered_dataset(cls.df_features, cls.feat_temp)
        if scaler_temp.exists():
            scaler_temp.unlink()

        # 4. Fit Anomaly Detector Model
        detector = BehavioralAnomalyDetector(n_estimators=30, contamination=0.10, random_state=42, model_path=cls.model_temp)
        detector.train(cls.df_features)

        # 5. Execute Threat Explainer
        cls.explainer = ThreatExplainer(model_path=cls.model_temp)
        cls.df_explained, cls.shap_matrix = cls.explainer.explain_dataset(cls.df_features)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        for path in [
            cls.raw_temp, cls.attack_temp, cls.feat_temp, cls.model_temp, cls.report_output_temp
        ]:
            if path.exists():
                path.unlink()

        if cls.figures_temp.exists():
            for f in cls.figures_temp.glob("*.png"):
                f.unlink()
            cls.figures_temp.rmdir()

    def test_shap_matrix_shape(self):
        """Verify SHAP matrix dimensions match (rows x feature_columns)."""
        self.assertEqual(self.shap_matrix.shape[0], len(self.df_features))
        self.assertEqual(self.shap_matrix.shape[1], len(self.explainer.feature_columns))

    def test_explanation_columns_presence(self):
        """Verify output dataframe contains top_contributing_features and natural_language_explanation."""
        self.assertIn("top_contributing_features", self.df_explained.columns)
        self.assertIn("natural_language_explanation", self.df_explained.columns)

    def test_natural_language_explanation_quality(self):
        """Verify natural language explanations produce valid non-empty text strings."""
        sample_exp = self.df_explained["natural_language_explanation"].iloc[0]
        self.assertIsInstance(sample_exp, str)
        self.assertGreater(len(sample_exp), 10)

    def test_generate_shap_plots(self):
        """Verify SHAP summary, bar, and decision force plots are generated."""
        saved_plots = self.explainer.generate_shap_plots(self.df_features, self.shap_matrix, self.figures_temp)
        self.assertGreaterEqual(len(saved_plots), 2)
        for p in saved_plots:
            self.assertTrue(p.exists())

    def test_save_explanation_reports(self):
        """Verify saving explanation report CSV writes valid readable file."""
        saved_path = self.explainer.save_explanation_reports(self.df_explained, self.report_output_temp)
        self.assertTrue(saved_path.exists())
        loaded_df = pd.read_csv(saved_path)
        self.assertEqual(len(loaded_df), len(self.df_features))
        self.assertIn("natural_language_explanation", loaded_df.columns)

if __name__ == "__main__":
    unittest.main()
