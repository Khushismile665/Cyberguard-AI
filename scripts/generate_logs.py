"""
CyberGuard AI - Synthetic Log Generation CLI Script

Command-line entry point to generate baseline normal cybersecurity authentication logs.

Usage:
    python scripts/generate_logs.py --records 100000 --users 1000 --devices 2500 --seed 42
"""

import sys
import argparse
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.data.log_generator import SyntheticLogGenerator
from src.utils.logger import setup_logger

logger = setup_logger("GenerateLogsCLI")

def main():
    parser = argparse.ArgumentParser(
        description="CyberGuard AI - Synthetic Cybersecurity Log Generator (Module 1)"
    )
    parser.add_argument(
        "--records",
        type=int,
        default=100000,
        help="Number of login records to generate (default: 100,000)"
    )
    parser.add_argument(
        "--users",
        type=int,
        default=1000,
        help="Number of unique user profiles (default: 1,000)"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=2500,
        help="Number of unique device fingerprints (default: 2,500)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Timeframe window in days (default: 30)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.dirs.raw_data_dir / "synthetic_login_logs.csv"),
        help="Destination path for CSV output"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Starting CyberGuard AI Synthetic Log Generation")
    logger.info("==================================================")
    logger.info(f"Target Records: {args.records:,}")
    logger.info(f"User Pool Size: {args.users:,}")
    logger.info(f"Device Pool Size: {args.devices:,}")
    logger.info(f"Simulated Days: {args.days}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Output File: {args.output}")

    try:
        generator = SyntheticLogGenerator(
            num_users=args.users,
            num_devices=args.devices,
            days=args.days,
            random_seed=args.seed
        )

        df = generator.generate_normal_logs(num_records=args.records)
        output_path = Path(args.output)
        saved_path = generator.save_to_csv(df, output_path)

        logger.info("==================================================")
        logger.info("Log Generation Completed Successfully!")
        logger.info(f"File Saved: {saved_path}")
        logger.info(f"Total Rows: {len(df):,}")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Fatal error during log generation script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
