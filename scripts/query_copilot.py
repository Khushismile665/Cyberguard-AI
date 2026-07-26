"""
CyberGuard AI - Query AI Security Copilot CLI Script (Module 9)

Command-line entry point to query the AI Security Copilot assistant.

Usage:
    python scripts/query_copilot.py --query "Why was User USR-00923 flagged?"
    python scripts/query_copilot.py --query "Show all impossible travel attacks"
    python scripts/query_copilot.py --query "Which users have the highest risk?"
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.copilot.sec_copilot import AISecurityCopilot
from src.utils.logger import setup_logger

logger = setup_logger("QueryCopilotCLI")

def main():
    parser = argparse.ArgumentParser(
        description="CyberGuard AI - AI Security Copilot CLI (Module 9)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Why was User USR-00923 flagged?",
        help="Natural language question for the AI Security Copilot assistant"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.dirs.base_dir / "data" / "predictions" / "explainable_anomaly_reports.csv"),
        help="Input predictions dataset CSV containing SHAP explanations and risk scores"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Starting CyberGuard AI Security Copilot CLI")
    logger.info("==================================================")
    logger.info(f"Query Question: {args.query}")
    logger.info(f"Input Dataset : {args.input}")

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            input_path = config.dirs.base_dir / "data" / "predictions" / "risk_scoring_results.csv"
            logger.info(f"Explanation reports CSV not found. Falling back to: {input_path}")

        logger.info(f"Loading input dataset from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset shape: {df.shape}")

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")

        copilot = AISecurityCopilot(df)
        response_md = copilot.answer_query(args.query)

        print("\n" + "=" * 60)
        print(f"[*] COPILOT RESPONSE FOR: '{args.query}'")
        print("=" * 60 + "\n")
        print(response_md)
        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Fatal error during Copilot CLI query execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
