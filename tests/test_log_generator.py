"""
CyberGuard AI - Unit Tests for Synthetic Log Generator (Module 1)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.log_generator import SyntheticLogGenerator, SCHEMA_COLUMNS

class TestSyntheticLogGenerator(unittest.TestCase):
    """Unit tests validating log generator schema, profile behavior, and CSV persistence."""

    def setUp(self):
        """Set up test generator instance."""
        self.num_records = 500
        self.generator = SyntheticLogGenerator(
            num_users=20,
            num_devices=40,
            days=7,
            random_seed=123
        )
        self.test_df = self.generator.generate_normal_logs(num_records=self.num_records)

    def test_schema_columns(self):
        """Verify generated dataframe contains exact 12 columns with expected names."""
        self.assertListEqual(list(self.test_df.columns), SCHEMA_COLUMNS)

    def test_record_count(self):
        """Verify generated dataframe row count matches requested count."""
        self.assertEqual(len(self.test_df), self.num_records)

    def test_no_missing_values(self):
        """Verify no null or NaN values exist in generated records."""
        self.assertEqual(self.test_df.isnull().sum().sum(), 0)

    def test_chronological_ordering(self):
        """Verify timestamps are sorted in strict ascending chronological order."""
        timestamps = pd.to_datetime(self.test_df["Timestamp"])
        self.assertTrue(timestamps.is_monotonic_increasing)

    def test_device_fingerprint_persistence(self):
        """Verify each Device ID maintains a persistent OS and Browser signature."""
        device_os = self.test_df.groupby("Device ID")["Operating System"].nunique()
        device_browser = self.test_df.groupby("Device ID")["Browser"].nunique()
        
        # Each device must map to exactly 1 OS and 1 Browser
        self.assertTrue((device_os == 1).all())
        self.assertTrue((device_browser == 1).all())

    def test_user_profile_persistence(self):
        """Verify each User ID maintains consistent primary location (Country/City)."""
        user_country = self.test_df.groupby("User ID")["Country"].nunique()
        user_city = self.test_df.groupby("User ID")["City"].nunique()

        self.assertTrue((user_country == 1).all())
        self.assertTrue((user_city == 1).all())

    def test_csv_export(self):
        """Verify CSV export writes valid CSV file matching dataframe content."""
        temp_csv = Path("data/raw/test_logs_temp.csv")
        try:
            saved_path = self.generator.save_to_csv(self.test_df, temp_csv)
            self.assertTrue(saved_path.exists())
            loaded_df = pd.read_csv(saved_path)
            self.assertEqual(len(loaded_df), self.num_records)
            self.assertListEqual(list(loaded_df.columns), SCHEMA_COLUMNS)
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

if __name__ == "__main__":
    unittest.main()
