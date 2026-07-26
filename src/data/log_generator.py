"""
CyberGuard AI - Synthetic Cybersecurity Log Generator (Module 1)

Generates high-fidelity baseline authentication logs simulating normal user behavior
across 30 days. Incorporates persistent user behavioral profiles, device fingerprints,
realistic location mappings, circadian shift distributions, and chronological ordering.
"""

import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from faker import Faker

from config import config
from src.utils.logger import setup_logger

logger = setup_logger("SyntheticLogGenerator")

# Standardized 12-Column Schema Definition
SCHEMA_COLUMNS = [
    "Timestamp",
    "User ID",
    "Device ID",
    "Source IP",
    "Country",
    "City",
    "Authentication Method",
    "Operating System",
    "Browser",
    "Resource Accessed",
    "Login Success",
    "Session Duration"
]

# Realistic Geographic Location Mappings (Country -> Cities & IP Prefixes)
GEO_LOCATION_MAP: Dict[str, Tuple[List[str], str]] = {
    "United States": (["New York", "Los Angeles", "Chicago", "San Francisco", "Austin", "Seattle"], "172.56."),
    "United Kingdom": (["London", "Manchester", "Birmingham", "Edinburgh"], "185.125."),
    "Germany": (["Berlin", "Frankfurt", "Munich", "Hamburg"], "91.198."),
    "India": (["Bangalore", "Mumbai", "Delhi", "Hyderabad"], "103.21."),
    "Japan": (["Tokyo", "Osaka", "Kyoto"], "133.242."),
    "Australia": (["Sydney", "Melbourne", "Brisbane"], "139.130."),
    "Canada": (["Toronto", "Vancouver", "Montreal"], "142.204.")
}

AUTHENTICATION_METHODS = ["SSO_SAML", "MFA_TOTP", "Password", "OAuth2", "Biometric"]
RESOURCE_ENDPOINTS = [
    "/dashboard",
    "/mail/inbox",
    "/cloud/storage",
    "/hr/portal",
    "/finance/reports",
    "/api/v1/user/profile",
    "/admin/settings",
    "/dev/git-repository"
]

DEVICE_OS_BROWSER_PAIRS = [
    ("Windows 11", "Chrome"),
    ("Windows 11", "Edge"),
    ("Windows 10", "Firefox"),
    ("macOS Sonoma", "Safari"),
    ("macOS Sonoma", "Chrome"),
    ("Linux Ubuntu", "Firefox"),
    ("iOS 17", "Safari"),
    ("Android 14", "Chrome")
]


@dataclass
class DeviceFingerprint:
    """Represents a persistent hardware device fingerprint."""
    device_id: str
    operating_system: str
    browser: str
    ip_subnet: str


@dataclass
class UserProfile:
    """Represents a persistent behavioral profile for a single user."""
    user_id: str
    home_country: str
    home_city: str
    primary_ip: str
    devices: List[DeviceFingerprint]
    work_start_hour: int
    work_end_hour: int
    is_night_shift: bool
    preferred_auth_method: str
    accessed_resources: List[str]


class SyntheticLogGenerator:
    """
    Generates synthetic, highly realistic normal user authentication logs for CyberGuard AI.
    """

    def __init__(
        self,
        num_users: int = 1000,
        num_devices: int = 2500,
        days: int = 30,
        random_seed: int = 42
    ):
        """
        Initializes the generator with configurable user and device profile pools.

        Args:
            num_users (int): Total number of unique user profiles.
            num_devices (int): Total number of unique device fingerprints.
            days (int): Timeframe window in days for log simulation.
            random_seed (int): Random seed for reproducible generation.
        """
        self.num_users = num_users
        self.num_devices = num_devices
        self.days = days
        self.random_seed = random_seed

        # Set seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.faker = Faker()
        Faker.seed(self.random_seed)

        logger.info(
            f"Initializing SyntheticLogGenerator (Users: {num_users}, Devices: {num_devices}, "
            f"Days: {days}, Seed: {random_seed})"
        )

        self.device_pool: List[DeviceFingerprint] = []
        self.user_pool: List[UserProfile] = []

        self._initialize_device_pool()
        self._initialize_user_pool()

    def _initialize_device_pool(self) -> None:
        """Constructs a deterministic pool of persistent device fingerprints."""
        try:
            for i in range(1, self.num_devices + 1):
                device_id = f"DEV-{i:05d}"
                os_name, browser = random.choice(DEVICE_OS_BROWSER_PAIRS)
                # Assign a subnet identifier
                country = random.choice(list(GEO_LOCATION_MAP.keys()))
                subnet = GEO_LOCATION_MAP[country][1]

                self.device_pool.append(
                    DeviceFingerprint(
                        device_id=device_id,
                        operating_system=os_name,
                        browser=browser,
                        ip_subnet=subnet
                    )
                )
            logger.info(f"Successfully created {len(self.device_pool)} persistent device fingerprints.")
        except Exception as e:
            logger.error(f"Error initializing device pool: {e}", exc_info=True)
            raise

    def _initialize_user_pool(self) -> None:
        """Constructs a deterministic pool of persistent user behavioral profiles."""
        try:
            countries = list(GEO_LOCATION_MAP.keys())
            for i in range(1, self.num_users + 1):
                user_id = f"USR-{i:05d}"
                country = random.choice(countries)
                cities, ip_prefix = GEO_LOCATION_MAP[country]
                city = random.choice(cities)
                primary_ip = f"{ip_prefix}{random.randint(1, 254)}.{random.randint(1, 254)}"

                # Assign 1 to 3 primary devices per user
                user_devices = random.sample(self.device_pool, k=random.randint(1, 3))

                # Define work shift (80% day shift 08:00-17:00, 20% night shift 22:00-06:00)
                is_night_shift = random.random() < 0.15
                if is_night_shift:
                    work_start = 22
                    work_end = 6
                else:
                    work_start = random.choice([8, 9])
                    work_end = work_start + 9

                # Assign resource affinities (2 to 5 endpoints typical for user role)
                user_resources = random.sample(RESOURCE_ENDPOINTS, k=random.randint(2, 5))
                pref_auth = random.choice(AUTHENTICATION_METHODS)

                self.user_pool.append(
                    UserProfile(
                        user_id=user_id,
                        home_country=country,
                        home_city=city,
                        primary_ip=primary_ip,
                        devices=user_devices,
                        work_start_hour=work_start,
                        work_end_hour=work_end,
                        is_night_shift=is_night_shift,
                        preferred_auth_method=pref_auth,
                        accessed_resources=user_resources
                    )
                )
            logger.info(f"Successfully created {len(self.user_pool)} user behavioral profiles.")
        except Exception as e:
            logger.error(f"Error initializing user pool: {e}", exc_info=True)
            raise

    def _generate_timestamp(self, user: UserProfile, end_date: datetime) -> datetime:
        """
        Generates a timestamp consistent with the user's preferred login hours over the timeframe.
        """
        start_date = end_date - timedelta(days=self.days)
        random_days = random.uniform(0, self.days)
        event_date = start_date + timedelta(days=random_days)

        # Diurnal pattern generation: 92% of logins fall around work shift
        if random.random() < 0.92:
            if user.is_night_shift:
                # Hour sampled around 22-06
                hour = (int(np.random.normal(loc=23, scale=2)) % 24)
            else:
                # Hour sampled around 09:00 - 17:00
                hour = int(np.clip(np.random.normal(loc=13, scale=3), 7, 19))
        else:
            # Off-hour login (benign variation)
            hour = random.randint(0, 23)

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        return event_date.replace(hour=hour, minute=minute, second=second, microsecond=0)

    def generate_normal_logs(self, num_records: int = 100000) -> pd.DataFrame:
        """
        Generates normal baseline login logs.

        Args:
            num_records (int): Total number of log records to generate.

        Returns:
            pd.DataFrame: Pandas DataFrame sorted chronologically containing 100,000 log rows.
        """
        logger.info(f"Generating {num_records:,} normal login records...")
        end_date = datetime.now()
        records = []

        try:
            for i in range(num_records):
                user: UserProfile = random.choice(self.user_pool)
                device: DeviceFingerprint = random.choice(user.devices)
                timestamp = self._generate_timestamp(user, end_date)

                # Benign failure probability (~3% password/MFA typos)
                login_success = random.random() > 0.03
                
                # Session Duration: 0 if failed login; log-normal distribution (30s to 8 hrs) if success
                if login_success:
                    session_duration = int(np.random.lognormal(mean=7.0, sigma=1.2))
                    session_duration = max(30, min(session_duration, 28800))  # Clamp between 30s & 8h
                else:
                    session_duration = 0

                resource = random.choice(user.accessed_resources)

                record = {
                    "Timestamp": timestamp,
                    "User ID": user.user_id,
                    "Device ID": device.device_id,
                    "Source IP": user.primary_ip,
                    "Country": user.home_country,
                    "City": user.home_city,
                    "Authentication Method": user.preferred_auth_method,
                    "Operating System": device.operating_system,
                    "Browser": device.browser,
                    "Resource Accessed": resource,
                    "Login Success": login_success,
                    "Session Duration": session_duration
                }
                records.append(record)

                if (i + 1) % 25000 == 0 or (i + 1) == num_records:
                    logger.info(f"Generated {i + 1:,} / {num_records:,} records...")

            df = pd.DataFrame(records)

            # Enforce strict ascending chronological sorting by Timestamp
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.sort_values(by="Timestamp").reset_index(drop=True)
            # Format timestamp to ISO 8601 string format
            df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"Log generation complete. Final DataFrame shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error during log generation trajectory: {e}", exc_info=True)
            raise

    def save_to_csv(self, df: pd.DataFrame, output_path: Path) -> Path:
        """
        Saves the generated DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): Log dataframe.
            output_path (Path): Path to output CSV file.

        Returns:
            Path: Absolute path of saved file.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Exporting logs to CSV at: {output_path}")
            df.to_csv(output_path, index=False, encoding="utf-8")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Successfully saved {len(df):,} records ({file_size_mb:.2f} MB) to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save CSV to {output_path}: {e}", exc_info=True)
            raise
