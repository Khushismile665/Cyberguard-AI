"""
CyberGuard AI - Anomaly Detection Training CLI Script (Module 4)

Command-line entry point to train Isolation Forest on normal baseline, evaluate model performance,
export predictions CSV, and render visualization plots.

Usage:
    python scripts/train_anomaly_detector.py --input data/processed/engineered_features.csv \
                                              --model-output saved_models/isolation_forest.joblib \
                                              --predictions-output data/predictions/anomaly_predictions.csv \
                                              --reports-dir reports/figures
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.models.anomaly_detector import BehavioralAnomalyDetector
from src.utils.logger import setup_logger

logger = setup_logger("TrainAnomalyDetectorCLI")

def main():
    parser = argparse.ArgumentParser(
        description="CyberGuard AI - Behavioral Anomaly Detection Training CLI (Module 4)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.dirs.processed_data_dir / "engineered_features.csv"),
        help="Input engineered feature CSV dataset"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(config.dirs.saved_models_dir / "isolation_forest.joblib"),
        help="Destination path for serialized Isolation Forest model"
    )
    parser.add_argument(
        "--predictions-output",
        type=str,
        default=str(BASE_DIR_PRED if 'BASE_DIR_PRED' in locals() else config.dirs.base_dir / "data" / "predictions" / "anomaly_predictions.csv"),
        help="Destination path for predictions CSV output"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(config.dirs.base_dir / "reports" / "figures"),
        help="Destination directory for diagnostic visualization plots"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Starting CyberGuard AI Behavioral Anomaly Detection Model Training")
    logger.info("==================================================")
    logger.info(f"Input Feature CSV: {args.input}")
    logger.info(f"Model Artifact Output: {args.model_output}")
    logger.info(f"Predictions CSV Output: {args.predictions_output}")
    logger.info(f"Reports Directory: {args.reports_dir}")

    try:
        input_path = Path(args.input)
        model_output = Path(args.model_output)
        pred_output = Path(args.predictions_output)
        reports_dir = Path(args.reports_dir)

        if not input_path.exists():
            logger.error(f"Input feature dataset not found at: {input_path}")
            sys.exit(1)

        logger.info(f"Loading engineered feature dataset from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded feature matrix shape: {df.shape}")

        detector = BehavioralAnomalyDetector(
            n_estimators=150,
            contamination=0.03,
            random_state=42,
            model_path=model_output
        )

        # Train strictly on normal records (is_anomaly == 0)
        detector.train(df)

        # Predict anomaly scores and predictions on full dataset
        df_preds, scores, binary_preds = detector.predict(df)

        # Evaluate model against ground truth if is_anomaly exists
        if "is_anomaly" in df.columns:
            y_true = df["is_anomaly"].values
            feature_cols = detector.feature_columns
            X = df[feature_cols].values
            from src.models.anomaly_detector import evaluate_model
            metrics = evaluate_model(detector.model, X, y_true)
            detector.evaluate(y_true, binary_preds, scores)
            detector.generate_plots(y_true, binary_preds, scores, reports_dir)

        # Save predictions CSV
        detector.save_predictions(df_preds, pred_output)

        logger.info("==================================================")
        logger.info("Anomaly Detection Model Training Completed Successfully!")
        logger.info(f"Trained Model Saved : {model_output}")
        logger.info(f"Predictions Saved   : {pred_output}")
        logger.info(f"Diagnostic Figures  : {reports_dir}")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Fatal error during model training script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
