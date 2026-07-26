"""
CyberGuard AI - Centralized Configuration Module

Provides production-ready configuration management for paths, model parameters,
logging settings, threat detection thresholds, and dashboard preferences.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent

@dataclass
class DirectoryConfig:
    """Directory paths for the project lifecycle."""
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    external_data_dir: Path = BASE_DIR / "data" / "external"
    saved_models_dir: Path = BASE_DIR / "saved_models"
    logs_dir: Path = BASE_DIR / "logs"
    notebooks_dir: Path = BASE_DIR / "notebooks"
    reports_dir: Path = BASE_DIR / "reports"

    def ensure_directories_exist(self) -> None:
        """Create all required directories if they do not exist."""
        for path in [
            self.data_dir, self.raw_data_dir, self.processed_data_dir,
            self.external_data_dir, self.saved_models_dir, self.logs_dir,
            self.notebooks_dir, self.reports_dir
        ]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Hyperparameters and threshold configurations for anomaly detection models."""
    random_state: int = 42
    contamination_rate: float = 0.05  # Expected ratio of anomalies in telemetry data
    
    # Isolation Forest Settings
    isolation_forest: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 150,
        "max_samples": "auto",
        "contamination": 0.05,
        "random_state": 42,
        "n_jobs": -1
    })

    # TensorFlow Deep Autoencoder Settings
    autoencoder: Dict[str, Any] = field(default_factory=lambda: {
        "encoding_dim": 16,
        "hidden_dim_1": 32,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "early_stopping_patience": 5
    })

@dataclass
class SecurityConfig:
    """Security alert criteria and risk score classification standards."""
    risk_threshold_low: float = 0.30
    risk_threshold_medium: float = 0.60
    risk_threshold_high: float = 0.85
    max_login_attempts_per_min: int = 10
    suspicious_payload_patterns: list = field(default_factory=lambda: [
        "SELECT * FROM", "<script>", "UNION SELECT", "../..", "/etc/passwd", "cmd.exe"
    ])

@dataclass
class AppConfig:
    """Master Application Configuration aggregating all sub-configs."""
    env: str = os.getenv("CYBERGUARD_ENV", "development")
    debug: bool = os.getenv("CYBERGUARD_DEBUG", "True").lower() in ("true", "1", "t")
    log_level: str = os.getenv("CYBERGUARD_LOG_LEVEL", "INFO")
    
    dirs: DirectoryConfig = field(default_factory=DirectoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self):
        """Ensure runtime directories are created upon initialization."""
        self.dirs.ensure_directories_exist()

# Global Singleton Configuration Instance
config = AppConfig()

if __name__ == "__main__":
    print(f"CyberGuard AI Initialized. Environment: {config.env}")
    print(f"Base Directory: {config.dirs.base_dir}")
