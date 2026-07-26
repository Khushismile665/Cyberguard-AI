"""
CyberGuard AI - Centralized Logging Utility

Provides standardized, formatted loggers for application components, CLI scripts,
and background processes.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "CyberGuardAI",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Creates and configures a standardized logger instance.

    Args:
        name (str): Name of the logger instance.
        log_file (Optional[Path]): File path to save log outputs.
        level (int): Logging severity level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream Handler (Console Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Primary Default Logger
logger = setup_logger()
