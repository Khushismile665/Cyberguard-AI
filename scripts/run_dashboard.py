"""
CyberGuard AI - Run Dashboard CLI Launcher Script (Module 8)

Invokes Streamlit to launch the CyberGuard AI SOC Web Dashboard.

Usage:
    python scripts/run_dashboard.py
"""

import sys
import subprocess
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger("RunDashboardCLI")

def main():
    dashboard_app = Path(__file__).resolve().parent.parent / "src" / "dashboard" / "app.py"

    if not dashboard_app.exists():
        logger.error(f"Dashboard app entry point not found at: {dashboard_app}")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("Launching CyberGuard AI 2.0 SOC Web Dashboard")
    logger.info(f"Target App: {dashboard_app}")
    logger.info("==================================================")

    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_app)]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Dashboard server stopped by user.")
    except Exception as e:
        logger.error(f"Error executing Streamlit dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
