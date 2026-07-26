"""
CyberGuard AI 2.0 - Root Deployment Launcher

Forwards execution to src/dashboard/app.py for seamless deployment on 
Streamlit Cloud, Render, Railway, AWS, and GCP.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dashboard.app import main

if __name__ == "__main__":
    main()
