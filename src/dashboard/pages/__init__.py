"""
CyberGuard AI Dashboard Pages Package
"""

from src.dashboard.pages.overview import render_overview_page
from src.dashboard.pages.live_stream import render_live_stream_page
from src.dashboard.pages.alerts import render_alerts_page
from src.dashboard.pages.user_behaviour import render_user_behaviour_page
from src.dashboard.pages.attack_analytics import render_attack_analytics_page
from src.dashboard.pages.risk_scores import render_risk_scores_page
from src.dashboard.pages.shap_explainability import render_shap_explainability_page
from src.dashboard.pages.model_performance import render_model_performance_page
from src.dashboard.pages.settings import render_settings_page
from src.dashboard.pages.copilot import render_copilot_page

__all__ = [
    "render_overview_page",
    "render_live_stream_page",
    "render_alerts_page",
    "render_user_behaviour_page",
    "render_attack_analytics_page",
    "render_risk_scores_page",
    "render_shap_explainability_page",
    "render_model_performance_page",
    "render_settings_page",
    "render_copilot_page",
]
