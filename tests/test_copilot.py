"""
CyberGuard AI - Unit Tests for AI Security Copilot (Module 9)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.copilot.sec_copilot import AISecurityCopilot

class TestAISecurityCopilot(unittest.TestCase):
    """Unit tests validating Copilot query intent handling, grounding, and response generation."""

    @classmethod
    def setUpClass(cls):
        """Create sample mock predictions dataset with SHAP explanations."""
        cls.df_mock = pd.DataFrame({
            "Timestamp": ["2026-07-25 10:00:00", "2026-07-25 10:05:00", "2026-07-25 10:10:00"],
            "User ID": ["USR-00102", "USR-00923", "USR-00102"],
            "Device ID": ["DEV-001", "DEV-002", "DEV-001"],
            "Source IP": ["192.168.1.1", "10.0.0.5", "192.168.1.1"],
            "Country": ["United States", "Germany", "United States"],
            "City": ["New York", "Berlin", "New York"],
            "is_anomaly": [0, 1, 1],
            "attack_type": ["Normal", "Impossible Travel", "Brute Force"],
            "anomaly_score": [0.05, 0.85, 0.72],
            "risk_score": [12.0, 88.5, 65.0],
            "risk_level": ["Low", "Critical", "High"],
            "top_contributing_features": [
                "login_hour (+0.010)",
                "login_velocity_kmh_scaled (+0.312) | is_new_location (+0.245)",
                "failed_login_count_1h_scaled (+0.280)"
            ],
            "natural_language_explanation": [
                "Normal baseline login activity.",
                "Risk increased due to impossible travel velocity. User logged in from a new country.",
                "Multiple failed login attempts detected within a short window."
            ]
        })
        cls.copilot = AISecurityCopilot(cls.df_mock)

    def test_explain_user_query(self):
        """Verify 'Why was User USR-00923 flagged?' returns grounded user report."""
        res = self.copilot.answer_query("Why was User USR-00923 flagged?")
        self.assertIn("USR-00923", res)
        self.assertIn("Impossible Travel", res)
        self.assertIn("88.5", res)
        self.assertIn("Recommended SOC Mitigation Actions", res)

    def test_filter_attacks_query(self):
        """Verify 'Show all impossible travel attacks' filters dataset correctly."""
        res = self.copilot.answer_query("Show all impossible travel attacks")
        self.assertIn("Impossible Travel", res)
        self.assertIn("USR-00923", res)
        self.assertIn("Recommended Response Playbook", res)

    def test_explain_anomaly_query(self):
        """Verify 'Explain this anomaly' returns SHAP explanation text."""
        res = self.copilot.answer_query("Explain this anomaly")
        self.assertIn("Anomaly Investigation Report", res)
        self.assertIn("SHAP Natural Language Explanation", res)

    def test_top_risk_users_query(self):
        """Verify 'Which users have the highest risk?' returns user leaderboard table."""
        res = self.copilot.answer_query("Which users have the highest risk?")
        self.assertIn("Highest Risk Users Leaderboard", res)
        self.assertIn("USR-00923", res)

    def test_recommend_actions_query(self):
        """Verify 'Recommend actions for this alert' returns mitigation playbooks."""
        res = self.copilot.answer_query("Recommend actions for Brute Force attacks")
        self.assertIn("Brute Force", res)
        self.assertIn("Account Lockout", res)

    def test_grounded_fallback_query(self):
        """Verify out-of-bounds questions trigger grounded fallback response."""
        res = self.copilot.answer_query("What is the weather in Paris?")
        self.assertIn("fact-grounded security assistant", res)

if __name__ == "__main__":
    unittest.main()
