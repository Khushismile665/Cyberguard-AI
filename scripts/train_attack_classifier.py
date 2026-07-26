"""
CyberGuard AI - Attack Classification CLI Script (Module 5)

Command-line entry point to train multi-class attack classifier on anomaly records, evaluate accuracy/F1,
compute feature importances, export predictions CSV, and generate diagnostic plots.

Usage:
    python scripts/train_attack_classifier.py --input data/predictions/anomaly_predictions.csv \
                                               --model-output saved_models/attack_classifier.joblib \
                                               --predictions-output data/predictions/attack_classification_predictions.csv \
                                               --reports-dir reports/figures
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.models.attack_classifier import AttackClassifier
from src.utils.logger import setup_logger

logger = setup_logger("TrainAttackClassifierCLI")

def main():
    parser = argparse.ArgumentParser(
        description="CyberGuard AI - Attack Classification Training CLI (Module 5)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.dirs.base_dir / "data" / "predictions" / "anomaly_predictions.csv"),
        help="Input predictions dataset CSV containing anomaly flags and engineered features"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(config.dirs.saved_models_dir / "attack_classifier.joblib"),
        help="Destination path for serialized attack classifier model"
    )
    parser.add_argument(
        "--predictions-output",
        type=str,
        default=str(config.dirs.base_dir / "data" / "predictions" / "attack_classification_predictions.csv"),
        help="Destination path for classification predictions CSV output"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(config.dirs.base_dir / "reports" / "figures"),
        help="Destination directory for diagnostic plots"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Starting CyberGuard AI Attack Classification Model Training")
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
            # Fall back to engineered features if anomaly predictions file isn't ready
            input_path = config.dirs.processed_data_dir / "engineered_features.csv"
            logger.info(f"Anomaly predictions file not found. Falling back to: {input_path}")

        logger.info(f"Loading input dataset from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset shape: {df.shape}")

        classifier = AttackClassifier(random_state=42, model_path=model_output)

        # Train model strictly on anomaly records (is_anomaly == 1)
        model, metrics = classifier.train(df, test_size=0.2)

        # Generate predictions across dataset
        df_preds, predicted_classes, confidences = classifier.predict(df)

        # Generate plots on anomaly test split
        attack_mask = df["is_anomaly"] == 1 if "is_anomaly" in df.columns else df["attack_type"] != "Normal"
        df_attacks = df[attack_mask].copy()
        y_true_enc = classifier.label_encoder.transform(df_attacks["attack_type"].values)
        y_pred_enc = classifier.label_encoder.transform(df_preds.loc[attack_mask, "predicted_attack_type"].values)

        classifier.generate_plots(y_true_enc, y_pred_enc, reports_dir)

        # Save predictions CSV
        classifier.save_predictions(df_preds, pred_output)

        # Display Top Feature Importances
        df_imp = classifier.get_feature_importances(top_n=10)
        logger.info("Top 10 Most Important Features:")
        for idx, row in df_imp.iterrows():
            logger.info(f"  {idx+1:2d}. {row['feature']:35s}: {row['importance']:.4f}")

        logger.info("==================================================")
        logger.info("Attack Classifier Training Completed Successfully!")
        logger.info(f"Trained Model Saved : {model_output}")
        logger.info(f"Predictions Saved   : {pred_output}")
        logger.info(f"Diagnostic Figures  : {reports_dir}")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Fatal error during classifier training script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
