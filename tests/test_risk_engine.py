"""
CyberGuard AI - Unit Tests for Risk Scoring Engine (Module 6)
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
from src.models.risk_engine import RiskScoringEngine

class TestRiskScoringEngine(unittest.TestCase):
    """Unit tests validating risk score bounds, level tier mapping, and CSV persistence."""

    @classmethod
    def setUpClass(cls):
        """Generate test dataset with features and anomaly scores."""
        cls.raw_temp = Path("data/raw/test_base_risk_temp.csv")
        cls.attack_temp = Path("data/processed/test_attack_risk_temp.csv")
        cls.feat_temp = Path("data/processed/test_feat_risk_temp.csv")
        cls.risk_output_temp = Path("data/predictions/test_risk_results_temp.csv")

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
        scaler_temp = Path("saved_models/test_scaler_risk_temp.joblib")
        engineer = FeatureEngineer(scaler_path=scaler_temp)
        df_features, _ = engineer.fit_transform(df_attacks)
        engineer.save_engineered_dataset(df_features, cls.feat_temp)
        if scaler_temp.exists():
            scaler_temp.unlink()

        # 4. Predict Anomaly Scores
        detector = BehavioralAnomalyDetector(n_estimators=30, contamination=0.10, random_state=42)
        detector.train(df_features)
        cls.df_preds, _, _ = detector.predict(df_features)

        # 5. Execute Risk Engine
        cls.engine = RiskScoringEngine()
        cls.df_risk = cls.engine.calculate_risk_scores(cls.df_preds)
        cls.engine.save_results(cls.df_risk, cls.risk_output_temp)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        for path in [
            cls.raw_temp, cls.attack_temp, cls.feat_temp, cls.risk_output_temp
        ]:
            if path.exists():
                path.unlink()

    def test_risk_score_bounds(self):
        """Verify risk_score is strictly bounded between 0.0 and 100.0."""
        self.assertTrue((self.df_risk["risk_score"] >= 0.0).all())
        self.assertTrue((self.df_risk["risk_score"] <= 100.0).all())

    def test_risk_level_values(self):
        """Verify risk_level contains only valid tier categories."""
        valid_tiers = {"Low", "Medium", "High", "Critical"}
        present_tiers = set(self.df_risk["risk_level"].unique())
        self.assertTrue(present_tiers.issubset(valid_tiers))

    def test_no_missing_values(self):
        """Verify zero null or NaN values in risk output columns."""
        self.assertEqual(self.df_risk["risk_score"].isnull().sum(), 0)
        self.assertEqual(self.df_risk["risk_level"].isnull().sum(), 0)

    def test_higher_risk_for_attacks(self):
        """Verify attack records receive significantly higher risk scores than normal records."""
        normal_mean_risk = self.df_risk[self.df_risk["is_anomaly"] == 0]["risk_score"].mean()
        attack_mean_risk = self.df_risk[self.df_risk["is_anomaly"] == 1]["risk_score"].mean()
        self.assertGreater(attack_mean_risk, normal_mean_risk)

    def test_save_results(self):
        """Verify risk results CSV persistence and readback."""
        self.assertTrue(self.risk_output_temp.exists())
        loaded_df = pd.read_csv(self.risk_output_temp)
        self.assertEqual(len(loaded_df), len(self.df_risk))
        self.assertIn("risk_score", loaded_df.columns)
        self.assertIn("risk_level", loaded_df.columns)

if __name__ == "__main__":
    unittest.main()
