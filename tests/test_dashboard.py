"""
CyberGuard AI - Unit Tests for Streamlit SOC Dashboard (Module 8)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.utils import load_dashboard_dataset, get_dark_plotly_layout
from src.dashboard.pages import (
    overview, live_stream, alerts, user_behaviour,
    attack_analytics, risk_scores, shap_explainability,
    model_performance, settings
)

class TestDashboardModules(unittest.TestCase):
    """Unit tests validating dashboard utilities and page module imports."""

    def test_load_dashboard_dataset(self):
        """Verify dashboard data loader returns a valid non-empty dataframe."""
        df = load_dashboard_dataset()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_get_dark_plotly_layout(self):
        """Verify Plotly dark SOC layout dictionary structure."""
        layout = get_dark_plotly_layout()
        self.assertIsInstance(layout, dict)
        self.assertIn("paper_bgcolor", layout)
        self.assertIn("plot_bgcolor", layout)

    def test_page_modules_imports(self):
        """Verify all 9 modular page components import cleanly."""
        self.assertTrue(callable(getattr(overview, "render_overview_page", None)))
        self.assertTrue(callable(getattr(live_stream, "render_live_stream_page", None)))
        self.assertTrue(callable(getattr(alerts, "render_alerts_page", None)))
        self.assertTrue(callable(getattr(user_behaviour, "render_user_behaviour_page", None)))
        self.assertTrue(callable(getattr(attack_analytics, "render_attack_analytics_page", None)))
        self.assertTrue(callable(getattr(risk_scores, "render_risk_scores_page", None)))
        self.assertTrue(callable(getattr(shap_explainability, "render_shap_explainability_page", None)))
        self.assertTrue(callable(getattr(model_performance, "render_model_performance_page", None)))
        self.assertTrue(callable(getattr(settings, "render_settings_page", None)))

if __name__ == "__main__":
    unittest.main()
