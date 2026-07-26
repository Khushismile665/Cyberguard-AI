"""
CyberGuard AI - Explainable AI CLI Script (Module 7)

Command-line entry point to generate SHAP feature attributions, natural language threat explanations,
and SHAP diagnostic plots.

Usage:
    python scripts/explain_threats.py --input data/predictions/risk_scoring_results.csv \
                                       --model saved_models/isolation_forest.joblib \
                                       --output data/predictions/explainable_anomaly_reports.csv \
                                       --reports-dir reports/figures
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.models.explainable_ai import ThreatExplainer
from src.utils.logger import setup_logger

logger = setup_logger("ExplainThreatsCLI")

def main():
    parser = argparse.ArgumentParser(
        description="CyberGuard AI - Explainable AI Engine CLI (Module 7)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.dirs.base_dir / "data" / "predictions" / "risk_scoring_results.csv"),
        help="Input predictions dataset CSV containing features and risk scores"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.dirs.saved_models_dir / "isolation_forest.joblib"),
        help="Path to trained model artifact for SHAP tree explainer"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.dirs.base_dir / "data" / "predictions" / "explainable_anomaly_reports.csv"),
        help="Destination path for output explainable anomaly reports CSV"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(config.dirs.base_dir / "reports" / "figures"),
        help="Destination directory for SHAP visualization plots"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Starting CyberGuard AI Explainable AI Pipeline")
    logger.info("==================================================")
    logger.info(f"Input Dataset: {args.input}")
    logger.info(f"Model Artifact: {args.model}")
    logger.info(f"Output Dataset: {args.output}")
    logger.info(f"Reports Directory: {args.reports_dir}")

    try:
        input_path = Path(args.input)
        model_path = Path(args.model)
        output_path = Path(args.output)
        reports_dir = Path(args.reports_dir)

        if not input_path.exists():
            input_path = config.dirs.base_dir / "data" / "predictions" / "attack_classification_predictions.csv"
            logger.info(f"Risk results CSV not found. Falling back to: {input_path}")

        logger.info(f"Loading input dataset from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset shape: {df.shape}")

        explainer = ThreatExplainer(model_path=model_path)

        # Generate SHAP explanations & natural language strings
        df_explained, shap_matrix = explainer.explain_dataset(df)

        # Export SHAP plots
        explainer.generate_shap_plots(df, shap_matrix, reports_dir)

        # Save explanation reports CSV
        saved_path = explainer.save_explanation_reports(df_explained, output_path)

        logger.info("==================================================")
        logger.info("Explainable AI Pipeline Completed Successfully!")
        logger.info(f"File Saved      : {saved_path}")
        logger.info(f"Total Rows      : {len(df_explained):,}")
        logger.info(f"SHAP Plots Saved: {reports_dir}")
        logger.info("Sample Threat Explanation:")
        anom_sample = df_explained[df_explained.get("is_anomaly", 0) == 1].head(1)
        if not anom_sample.empty:
            attack_val = anom_sample['attack_type'].values[0] if 'attack_type' in anom_sample.columns else 'Anomaly'
            logger.info(f"  - User ID     : {anom_sample['User ID'].values[0]}")
            logger.info(f"  - Attack Type : {attack_val}")
            logger.info(f"  - Explanation : {anom_sample['natural_language_explanation'].values[0]}")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Fatal error during explainable AI script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
