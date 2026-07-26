"""
CyberGuard AI - Feature Engineering CLI Script (Module 3)

Command-line entry point to transform dataset into normalized machine learning features.

Usage:
    python scripts/engineer_features.py --input data/processed/synthetic_login_logs_with_attacks.csv \
                                         --output data/processed/engineered_features.csv \
                                         --scaler-path saved_models/feature_scaler.joblib
"""

import sys
import argparse
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.data.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger("EngineerFeaturesCLI")

def main():
    parser = argparse.ArgumentParser(
        description="CyberGuard AI - Feature Engineering Pipeline CLI (Module 3)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.dirs.processed_data_dir / "synthetic_login_logs_with_attacks.csv"),
        help="Input processed dataset CSV containing attack labels"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.dirs.processed_data_dir / "engineered_features.csv"),
        help="Destination path for output engineered feature CSV dataset"
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=str(config.dirs.saved_models_dir / "feature_scaler.joblib"),
        help="Destination path to serialize fitted StandardScaler object"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Starting CyberGuard AI Feature Engineering Pipeline")
    logger.info("==================================================")
    logger.info(f"Input Dataset: {args.input}")
    logger.info(f"Output Feature CSV: {args.output}")
    logger.info(f"Scaler Artifact Path: {args.scaler_path}")

    try:
        input_path = Path(args.input)
        output_path = Path(args.output)
        scaler_path = Path(args.scaler_path)

        if not input_path.exists():
            logger.error(f"Input dataset not found at: {input_path}")
            sys.exit(1)

        import pandas as pd
        df_input = pd.read_csv(input_path)

        engineer = FeatureEngineer(scaler_path=scaler_path)
        df_features, feature_cols = engineer.fit_transform(df_input)
        saved_path = engineer.save_engineered_dataset(df_features, output_path)

        logger.info("==================================================")
        logger.info("Feature Engineering Completed Successfully!")
        logger.info(f"File Saved: {saved_path}")
        logger.info(f"Total Rows: {len(df_features):,}")
        logger.info(f"Engineered ML Feature Columns: {len(feature_cols)}")
        logger.info(f"Scaler Artifact Saved: {scaler_path}")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Fatal error during feature engineering script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
