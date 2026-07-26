"""
CyberGuard AI - Unit Tests for Attack Injector (Module 2)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.log_generator import SyntheticLogGenerator, SCHEMA_COLUMNS
from src.data.attack_injector import AttackInjector, ATTACK_TYPES

class TestAttackInjector(unittest.TestCase):
    """Unit tests validating attack injection schema, ratio, attack type distribution, and sorting."""

    @classmethod
    def setUpClass(cls):
        """Generate a baseline dataset and inject attacks once for test assertions."""
        cls.raw_test_path = Path("data/raw/test_baseline_temp.csv")
        cls.processed_test_path = Path("data/processed/test_attacks_temp.csv")

        # Generate 1,000 record sample baseline
        gen = SyntheticLogGenerator(num_users=20, num_devices=40, days=7, random_seed=42)
        df_base = gen.generate_normal_logs(num_records=1000)
        gen.save_to_csv(df_base, cls.raw_test_path)

        # Inject attacks
        cls.injector = AttackInjector(random_seed=42)
        df_loaded = cls.injector.load_baseline_data(cls.raw_test_path)
        cls.df_attacks = cls.injector.inject_attacks(df_loaded, attack_ratio=0.06)  # 6% for small test pool
        cls.injector.save_dataset(cls.df_attacks, cls.processed_test_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test files."""
        if cls.raw_test_path.exists():
            cls.raw_test_path.unlink()
        if cls.processed_test_path.exists():
            cls.processed_test_path.unlink()

    def test_schema_columns(self):
        """Verify output contains 14 columns (12 original + is_anomaly + attack_type)."""
        expected_columns = SCHEMA_COLUMNS + ["is_anomaly", "attack_type"]
        self.assertListEqual(list(self.df_attacks.columns), expected_columns)

    def test_record_count_preservation(self):
        """Verify total record count is preserved (1,000 records)."""
        self.assertEqual(len(self.df_attacks), 1000)

    def test_no_missing_values(self):
        """Verify zero null or missing values across all 14 columns."""
        self.assertEqual(self.df_attacks.isnull().sum().sum(), 0)

    def test_all_attack_types_present(self):
        """Verify all 6 attack types exist in the attack_type column."""
        present_types = set(self.df_attacks["attack_type"].unique())
        for attack_name in ATTACK_TYPES:
            self.assertIn(attack_name, present_types)
        self.assertIn("Normal", present_types)

    def test_anomaly_flag_consistency(self):
        """Verify is_anomaly is 1 for attack rows and 0 for Normal rows."""
        normal_mask = self.df_attacks["attack_type"] == "Normal"
        attack_mask = self.df_attacks["attack_type"] != "Normal"

        self.assertTrue((self.df_attacks.loc[normal_mask, "is_anomaly"] == 0).all())
        self.assertTrue((self.df_attacks.loc[attack_mask, "is_anomaly"] == 1).all())

    def test_chronological_ordering(self):
        """Verify timestamps are sorted strictly in ascending chronological order."""
        timestamps = pd.to_datetime(self.df_attacks["Timestamp"])
        self.assertTrue(timestamps.is_monotonic_increasing)

    def test_processed_csv_saved(self):
        """Verify processed dataset CSV file was successfully saved to disk."""
        self.assertTrue(self.processed_test_path.exists())
        df_disk = pd.read_csv(self.processed_test_path)
        self.assertEqual(len(df_disk), 1000)
        self.assertEqual(len(df_disk.columns), 14)

if __name__ == "__main__":
    unittest.main()
