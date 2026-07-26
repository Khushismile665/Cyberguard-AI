"""
CyberGuard AI - Risk Scoring CLI Script (Module 6)

Command-line entry point to calculate explainable risk scores (0-100) and severity levels.

Usage:
    python scripts/calculate_risk_scores.py --input data/predictions/attack_classification_predictions.csv \
                                             --output data/predictions/risk_scoring_results.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.models.risk_engine import RiskScoringEngine
from src.utils.logger import setup_logger

logger = setup_logger("CalculateRiskScoresCLI")

def main():
    parser = argparse.ArgumentParser(
        description="CyberGuard AI - Risk Scoring Engine CLI (Module 6)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.dirs.base_dir / "data" / "predictions" / "attack_classification_predictions.csv"),
        help="Input predictions dataset CSV containing anomaly scores and attack classifications"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.dirs.base_dir / "data" / "predictions" / "risk_scoring_results.csv"),
        help="Destination path for risk scoring results CSV output"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Starting CyberGuard AI Risk Scoring Engine")
    logger.info("==================================================")
    logger.info(f"Input Dataset: {args.input}")
    logger.info(f"Output Dataset: {args.output}")

    try:
        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            input_path = config.dirs.base_dir / "data" / "predictions" / "anomaly_predictions.csv"
            logger.info(f"Classification predictions file not found. Falling back to: {input_path}")

        logger.info(f"Loading input dataset from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset shape: {df.shape}")

        engine = RiskScoringEngine()
        df_risk = engine.calculate_risk_scores(df)
        saved_path = engine.save_results(df_risk, output_path)

        logger.info("==================================================")
        logger.info("Risk Scoring Completed Successfully!")
        logger.info(f"File Saved: {saved_path}")
        logger.info(f"Total Rows: {len(df_risk):,}")
        logger.info("Risk Level Summary:")
        for level, count in df_risk["risk_level"].value_counts().items():
            logger.info(f"  - {level:10s}: {count:6,d} ({(count / len(df_risk) * 100):.2f}%)")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Fatal error during risk scoring script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
