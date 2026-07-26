"""
CyberGuard AI - Unit Tests for Feature Engineering Pipeline (Module 3)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import joblib

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.log_generator import SyntheticLogGenerator
from src.data.attack_injector import AttackInjector
from src.data.feature_engineering import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    """Unit tests validating feature engineering calculations, scaling, and persistence."""

    @classmethod
    def setUpClass(cls):
        """Set up temporary sample dataset and run feature engineer once."""
        cls.raw_path = Path("data/raw/test_base_feat_temp.csv")
        cls.attacks_path = Path("data/processed/test_attacks_feat_temp.csv")
        cls.output_feat_path = Path("data/processed/test_engineered_feat_temp.csv")
        cls.scaler_temp_path = Path("saved_models/test_scaler_temp.joblib")

        # Generate sample baseline & inject attacks
        gen = SyntheticLogGenerator(num_users=15, num_devices=30, days=5, random_seed=42)
        df_base = gen.generate_normal_logs(num_records=400)
        gen.save_to_csv(df_base, cls.raw_path)

        injector = AttackInjector(random_seed=42)
        df_loaded = injector.load_baseline_data(cls.raw_path)
        cls.df_attacks = injector.inject_attacks(df_loaded, attack_ratio=0.05)
        injector.save_dataset(cls.df_attacks, cls.attacks_path)

        # Execute Feature Engineering
        cls.engineer = FeatureEngineer(scaler_path=cls.scaler_temp_path)
        cls.df_feat, cls.feature_cols = cls.engineer.fit_transform(cls.df_attacks)
        cls.engineer.save_engineered_dataset(cls.df_feat, cls.output_feat_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        for path in [cls.raw_path, cls.attacks_path, cls.output_feat_path, cls.scaler_temp_path]:
            if path.exists():
                path.unlink()

    def test_required_scaled_features_present(self):
        """Verify all 12 requested feature groups are extracted and scaled."""
        expected_scaled_features = [
            "login_hour_scaled",
            "day_of_week_scaled",
            "failed_login_count_1h_scaled",
            "unique_devices_24h_scaled",
            "unique_countries_24h_scaled",
            "session_duration_scaled",
            "resource_access_frequency_scaled",
            "time_since_prev_login_sec_scaled",
            "geo_distance_km_scaled",
            "login_velocity_kmh_scaled"
        ]

        for feat in expected_scaled_features:
            self.assertIn(feat, self.df_feat.columns)

    def test_novelty_binary_features_present(self):
        """Verify binary novelty flags (is_new_device, is_new_location) are present."""
        self.assertIn("is_new_device", self.df_feat.columns)
        self.assertIn("is_new_location", self.df_feat.columns)

    def test_target_columns_preserved(self):
        """Verify target labels (is_anomaly and attack_type) are preserved."""
        self.assertIn("is_anomaly", self.df_feat.columns)
        self.assertIn("attack_type", self.df_feat.columns)

    def test_no_missing_values(self):
        """Verify zero null or missing values exist in the engineered feature dataset."""
        self.assertEqual(self.df_feat.isnull().sum().sum(), 0)

    def test_scaler_persistence_and_loading(self):
        """Verify StandardScaler is serialized to disk and can be reloaded."""
        self.assertTrue(self.scaler_temp_path.exists())
        loaded_scaler = joblib.load(self.scaler_temp_path)
        self.assertIsNotNone(loaded_scaler)
        self.assertEqual(loaded_scaler.n_features_in_, 15)

    def test_output_csv_creation(self):
        """Verify output feature dataset CSV is written and readable."""
        self.assertTrue(self.output_feat_path.exists())
        df_loaded = pd.read_csv(self.output_feat_path)
        self.assertEqual(len(df_loaded), len(self.df_attacks))

if __name__ == "__main__":
    unittest.main()
