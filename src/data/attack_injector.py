"""
CyberGuard AI - Attack Injector (Module 2)

Injects realistic cyber attack scenarios (Brute Force, Credential Stuffing,
Impossible Travel, Device Spoofing, Lateral Movement, and Insider Threat)
into baseline authentication telemetry while keeping 97% of the dataset normal.
Appends 'is_anomaly' (0/1) and 'attack_type' labels to the 12 baseline columns.
"""

import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from faker import Faker

from config import config
from src.utils.logger import setup_logger

logger = setup_logger("AttackInjector")

ATTACK_TYPES = [
    "Brute Force",
    "Credential Stuffing",
    "Impossible Travel",
    "Device Spoofing",
    "Lateral Movement",
    "Insider Threat"
]

RESTRICTED_RESOURCES = ["/admin/settings", "/dev/git-repository", "/finance/reports"]

DISTANT_LOCATIONS: Dict[str, Tuple[List[str], str]] = {
    "Russia": (["Moscow", "Saint Petersburg"], "185.220."),
    "China": (["Beijing", "Shanghai"], "220.181."),
    "Brazil": (["Sao Paulo", "Rio de Janeiro"], "177.126."),
    "North Korea": (["Pyongyang"], "175.45.")
}


class AttackInjector:
    """
    Injects realistic cyber attacks into baseline cybersecurity login data.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initializes the AttackInjector with seed reproducibility.

        Args:
            random_seed (int): Seed for random number generation.
        """
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.faker = Faker()
        Faker.seed(self.random_seed)

        logger.info(f"Initialized AttackInjector (Seed: {self.random_seed})")

    def load_baseline_data(self, input_path: Path) -> pd.DataFrame:
        """
        Loads the baseline raw CSV dataset.

        Args:
            input_path (Path): Path to raw CSV file.

        Returns:
            pd.DataFrame: Baseline dataframe.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Baseline raw dataset not found at: {input_path}")

        logger.info(f"Loading baseline dataset from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded baseline dataset with {len(df):,} records and {len(df.columns)} columns.")
        return df

    def inject_attacks(self, df: pd.DataFrame, attack_ratio: float = 0.03) -> pd.DataFrame:
        """
        Injects attack scenarios into a specified ratio (default: 3%) of records.

        Args:
            df (pd.DataFrame): Input baseline dataframe.
            attack_ratio (float): Fraction of total records to represent as attacks.

        Returns:
            pd.DataFrame: Dataframe with injected attacks, sorted chronologically, with attack labels.
        """
        logger.info(f"Starting attack injection target: {attack_ratio * 100:.1f}% of total dataset.")
        df_copy = df.copy()

        # Initialize anomaly tracking columns
        df_copy["is_anomaly"] = 0
        df_copy["attack_type"] = "Normal"

        total_records = len(df_copy)
        target_attack_count = int(total_records * attack_ratio)
        per_attack_count = target_attack_count // len(ATTACK_TYPES)

        logger.info(
            f"Target Attack Records: {target_attack_count:,} (~{per_attack_count:,} per attack type across "
            f"{len(ATTACK_TYPES)} attack scenarios)"
        )

        # Parse Timestamps for temporal calculations
        df_copy["Timestamp_dt"] = pd.to_datetime(df_copy["Timestamp"])

        attack_records: List[Dict] = []
        df_normal = df_copy.copy()

        # 1. Inject Brute Force Attacks
        logger.info("Injecting Brute Force attacks...")
        brute_force_events = self._generate_brute_force(df_normal, count=per_attack_count)
        attack_records.extend(brute_force_events)

        # 2. Inject Credential Stuffing Attacks
        logger.info("Injecting Credential Stuffing attacks...")
        cred_stuffing_events = self._generate_credential_stuffing(df_normal, count=per_attack_count)
        attack_records.extend(cred_stuffing_events)

        # 3. Inject Impossible Travel Attacks
        logger.info("Injecting Impossible Travel attacks...")
        imp_travel_events = self._generate_impossible_travel(df_normal, count=per_attack_count)
        attack_records.extend(imp_travel_events)

        # 4. Inject Device Spoofing Attacks
        logger.info("Injecting Device Spoofing attacks...")
        device_spoof_events = self._generate_device_spoofing(df_normal, count=per_attack_count)
        attack_records.extend(device_spoof_events)

        # 5. Inject Lateral Movement Attacks
        logger.info("Injecting Lateral Movement attacks...")
        lateral_move_events = self._generate_lateral_movement(df_normal, count=per_attack_count)
        attack_records.extend(lateral_move_events)

        # 6. Inject Insider Threat Attacks
        logger.info("Injecting Insider Threat attacks...")
        insider_threat_events = self._generate_insider_threat(df_normal, count=per_attack_count)
        attack_records.extend(insider_threat_events)

        # Convert injected attack records to DataFrame
        df_attacks = pd.DataFrame(attack_records)
        logger.info(f"Total Injected Attack Records Created: {len(df_attacks):,}")

        # Remove an equal number of normal records to preserve exact 100,000 total dataset count
        replace_indices = np.random.choice(df_normal.index, size=len(df_attacks), replace=False)
        df_normal_pruned = df_normal.drop(index=replace_indices)

        # Combine normal and attack records
        df_combined = pd.concat([df_normal_pruned, df_attacks], ignore_index=True)

        # Clean up temporary datetime helper column
        if "Timestamp_dt" in df_combined.columns:
            df_combined.drop(columns=["Timestamp_dt"], inplace=True)

        # Sort strictly chronologically
        df_combined["Timestamp_dt_sort"] = pd.to_datetime(df_combined["Timestamp"])
        df_combined = df_combined.sort_values(by="Timestamp_dt_sort").reset_index(drop=True)
        df_combined.drop(columns=["Timestamp_dt_sort"], inplace=True)

        actual_attack_pct = (df_combined["is_anomaly"].sum() / len(df_combined)) * 100
        logger.info(
            f"Attack Injection Complete. Total Records: {len(df_combined):,}. "
            f"Anomalies: {df_combined['is_anomaly'].sum():,} ({actual_attack_pct:.2f}%)."
        )

        return df_combined

    def _generate_brute_force(self, df: pd.DataFrame, count: int) -> List[Dict]:
        """Generates Brute Force attack events (bursts of failed logins from 1 IP)."""
        events = []
        users = df["User ID"].unique()
        num_bursts = max(1, count // 10)

        for _ in range(num_bursts):
            target_user = random.choice(users)
            attacker_ip = f"198.51.100.{random.randint(1, 254)}"  # External attack IP
            base_time = random.choice(df["Timestamp_dt"])

            for j in range(count // num_bursts):
                time_offset = timedelta(seconds=j * random.randint(2, 10))
                events.append({
                    "Timestamp": (base_time + time_offset).strftime("%Y-%m-%d %H:%M:%S"),
                    "User ID": target_user,
                    "Device ID": f"DEV-ATTACK-{random.randint(100, 999)}",
                    "Source IP": attacker_ip,
                    "Country": "Unknown",
                    "City": "Unknown",
                    "Authentication Method": "Password",
                    "Operating System": "Linux",
                    "Browser": "Python-urllib/3.9",
                    "Resource Accessed": "/dashboard",
                    "Login Success": False,
                    "Session Duration": 0,
                    "is_anomaly": 1,
                    "attack_type": "Brute Force"
                })

        return events[:count]

    def _generate_credential_stuffing(self, df: pd.DataFrame, count: int) -> List[Dict]:
        """Generates Credential Stuffing events (1 IP targeting many distinct users)."""
        events = []
        users = list(df["User ID"].unique())
        attacker_ip = f"203.0.113.{random.randint(1, 254)}"
        base_time = random.choice(df["Timestamp_dt"])

        for i in range(count):
            target_user = random.choice(users)
            time_offset = timedelta(seconds=i * random.randint(1, 5))
            events.append({
                "Timestamp": (base_time + time_offset).strftime("%Y-%m-%d %H:%M:%S"),
                "User ID": target_user,
                "Device ID": f"DEV-STUFF-{random.randint(100, 999)}",
                "Source IP": attacker_ip,
                "Country": "External",
                "City": "External",
                "Authentication Method": "Password",
                "Operating System": "Windows 10",
                "Browser": "Chrome",
                "Resource Accessed": "/mail/inbox",
                "Login Success": random.random() < 0.05,  # 5% success rate in credential stuffing
                "Session Duration": 0,
                "is_anomaly": 1,
                "attack_type": "Credential Stuffing"
            })

        return events

    def _generate_impossible_travel(self, df: pd.DataFrame, count: int) -> List[Dict]:
        """Generates Impossible Travel events (rapid login from distant geo location)."""
        events = []
        sample_rows = df.sample(n=count, replace=True, random_state=self.random_seed)

        for _, row in sample_rows.iterrows():
            orig_time = row["Timestamp_dt"]
            # Impossible travel within 5 to 20 minutes
            impossible_time = orig_time + timedelta(minutes=random.randint(5, 20))
            
            foreign_country = random.choice(list(DISTANT_LOCATIONS.keys()))
            foreign_cities, ip_prefix = DISTANT_LOCATIONS[foreign_country]
            foreign_city = random.choice(foreign_cities)
            foreign_ip = f"{ip_prefix}{random.randint(1, 254)}.{random.randint(1, 254)}"

            events.append({
                "Timestamp": impossible_time.strftime("%Y-%m-%d %H:%M:%S"),
                "User ID": row["User ID"],
                "Device ID": f"DEV-FOREIGN-{random.randint(100, 999)}",
                "Source IP": foreign_ip,
                "Country": foreign_country,
                "City": foreign_city,
                "Authentication Method": "Password",
                "Operating System": "Linux Ubuntu",
                "Browser": "Firefox",
                "Resource Accessed": "/cloud/storage",
                "Login Success": True,
                "Session Duration": random.randint(120, 1800),
                "is_anomaly": 1,
                "attack_type": "Impossible Travel"
            })

        return events

    def _generate_device_spoofing(self, df: pd.DataFrame, count: int) -> List[Dict]:
        """Generates Device Spoofing events (login using unknown/conflicting device)."""
        events = []
        sample_rows = df.sample(n=count, replace=True, random_state=self.random_seed + 1)

        for _, row in sample_rows.iterrows():
            events.append({
                "Timestamp": row["Timestamp_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                "User ID": row["User ID"],
                "Device ID": f"DEV-SPOOFED-{random.randint(1000, 9999)}",
                "Source IP": row["Source IP"],
                "Country": row["Country"],
                "City": row["City"],
                "Authentication Method": "Password",
                "Operating System": "Unknown OS",
                "Browser": "HeadlessChrome/114.0",
                "Resource Accessed": row["Resource Accessed"],
                "Login Success": True,
                "Session Duration": random.randint(60, 600),
                "is_anomaly": 1,
                "attack_type": "Device Spoofing"
            })

        return events

    def _generate_lateral_movement(self, df: pd.DataFrame, count: int) -> List[Dict]:
        """Generates Lateral Movement events (accessing unauthorized sensitive endpoints)."""
        events = []
        sample_rows = df.sample(n=count, replace=True, random_state=self.random_seed + 2)

        for _, row in sample_rows.iterrows():
            restricted_res = random.choice(RESTRICTED_RESOURCES)
            events.append({
                "Timestamp": row["Timestamp_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                "User ID": row["User ID"],
                "Device ID": row["Device ID"],
                "Source IP": row["Source IP"],
                "Country": row["Country"],
                "City": row["City"],
                "Authentication Method": row["Authentication Method"],
                "Operating System": row["Operating System"],
                "Browser": row["Browser"],
                "Resource Accessed": restricted_res,
                "Login Success": True,
                "Session Duration": random.randint(300, 3600),
                "is_anomaly": 1,
                "attack_type": "Lateral Movement"
            })

        return events

    def _generate_insider_threat(self, df: pd.DataFrame, count: int) -> List[Dict]:
        """Generates Insider Threat events (escalating frequency of privileged resource access)."""
        events = []
        sample_rows = df.sample(n=count, replace=True, random_state=self.random_seed + 3)

        for _, row in sample_rows.iterrows():
            events.append({
                "Timestamp": row["Timestamp_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                "User ID": row["User ID"],
                "Device ID": row["Device ID"],
                "Source IP": row["Source IP"],
                "Country": row["Country"],
                "City": row["City"],
                "Authentication Method": "MFA_TOTP",
                "Operating System": row["Operating System"],
                "Browser": row["Browser"],
                "Resource Accessed": "/admin/settings",
                "Login Success": True,
                "Session Duration": random.randint(600, 7200),
                "is_anomaly": 1,
                "attack_type": "Insider Threat"
            })

        return events

    def save_dataset(self, df: pd.DataFrame, output_path: Path) -> Path:
        """
        Saves the processed dataset with attack labels to CSV.

        Args:
            df (pd.DataFrame): Dataset containing original columns + attack labels.
            output_path (Path): Path to output CSV file.

        Returns:
            Path: Path of saved file.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving dataset with attack labels to: {output_path}")
            df.to_csv(output_path, index=False, encoding="utf-8")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Successfully saved dataset with {len(df):,} records ({file_size_mb:.2f} MB) "
                f"to {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to save processed dataset to {output_path}: {e}", exc_info=True)
            raise
