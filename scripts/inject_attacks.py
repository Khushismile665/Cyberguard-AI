"""
CyberGuard AI - Attack Injection CLI Script (Module 2)

Command-line entry point to inject cyber attack scenarios into baseline dataset.

Usage:
    python scripts/inject_attacks.py --input data/raw/synthetic_login_logs.csv \
                                      --output data/processed/synthetic_login_logs_with_attacks.csv \
                                      --attack-ratio 0.03 --seed 42
"""

import sys
import argparse
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.data.attack_injector import AttackInjector
from src.utils.logger import setup_logger

logger = setup_logger("InjectAttacksCLI")

def main():
    parser = argparse.ArgumentParser(
        description="CyberGuard AI - Attack Injector CLI (Module 2)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.dirs.raw_data_dir / "synthetic_login_logs.csv"),
        help="Input raw baseline CSV dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.dirs.processed_data_dir / "synthetic_login_logs_with_attacks.csv"),
        help="Destination path for output dataset with attack labels"
    )
    parser.add_argument(
        "--attack-ratio",
        type=float,
        default=0.03,
        help="Fraction of dataset to inject as attacks (default: 0.03 / 3%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Starting CyberGuard AI Attack Injection")
    logger.info("==================================================")
    logger.info(f"Input Dataset: {args.input}")
    logger.info(f"Output Dataset: {args.output}")
    logger.info(f"Target Attack Ratio: {args.attack_ratio * 100:.1f}%")
    logger.info(f"Random Seed: {args.seed}")

    try:
        input_path = Path(args.input)
        output_path = Path(args.output)

        injector = AttackInjector(random_seed=args.seed)
        df_baseline = injector.load_baseline_data(input_path)
        df_attacks = injector.inject_attacks(df_baseline, attack_ratio=args.attack_ratio)
        saved_path = injector.save_dataset(df_attacks, output_path)

        logger.info("==================================================")
        logger.info("Attack Injection Completed Successfully!")
        logger.info(f"File Saved: {saved_path}")
        logger.info(f"Total Rows: {len(df_attacks):,}")
        logger.info(f"Normal Records: {(df_attacks['is_anomaly'] == 0).sum():,} ({(df_attacks['is_anomaly'] == 0).mean() * 100:.2f}%)")
        logger.info(f"Attack Records: {(df_attacks['is_anomaly'] == 1).sum():,} ({(df_attacks['is_anomaly'] == 1).mean() * 100:.2f}%)")
        logger.info("Breakdown by Attack Type:")
        for attack_name, count in df_attacks["attack_type"].value_counts().items():
            logger.info(f"  - {attack_name:20s}: {count:6,d}")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Fatal error during attack injection script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
