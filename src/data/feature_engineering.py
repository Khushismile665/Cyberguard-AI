"""
CyberGuard AI - Feature Engineering Pipeline (Module 3)

Extracts temporal, behavioral, geographic, velocity, and categorical features from login logs.
Normalizes numerical features using StandardScaler, encodes categorical variables,
saves scaler artifacts for model inference, and outputs the engineered dataset.
"""

import math
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from config import config
from src.utils.logger import setup_logger

logger = setup_logger("FeatureEngineer")

# Geographic Coordinate Mappings (Lat, Lon) for Distance Calculations
CITY_COORDINATES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("United States", "New York"): (40.7128, -74.0060),
    ("United States", "Los Angeles"): (34.0522, -118.2437),
    ("United States", "Chicago"): (41.8781, -87.6298),
    ("United States", "San Francisco"): (37.7749, -122.4194),
    ("United States", "Austin"): (30.2672, -97.7431),
    ("United States", "Seattle"): (47.6062, -122.3321),
    ("United Kingdom", "London"): (51.5074, -0.1278),
    ("United Kingdom", "Manchester"): (53.4808, -2.2426),
    ("United Kingdom", "Edinburgh"): (55.9533, -3.1883),
    ("Germany", "Berlin"): (52.5200, 13.4050),
    ("Germany", "Frankfurt"): (50.1109, 8.6821),
    ("Germany", "Munich"): (48.1351, 11.5820),
    ("India", "Bangalore"): (12.9716, 77.5946),
    ("India", "Mumbai"): (19.0760, 72.8777),
    ("India", "Delhi"): (28.6139, 77.2090),
    ("Japan", "Tokyo"): (35.6762, 139.6503),
    ("Japan", "Osaka"): (34.6937, 135.5023),
    ("Australia", "Sydney"): (-33.8688, 151.2093),
    ("Australia", "Melbourne"): (-37.8136, 144.9631),
    ("Canada", "Toronto"): (43.6532, -79.3832),
    ("Russia", "Moscow"): (55.7558, 37.6173),
    ("China", "Beijing"): (39.9042, 116.4074),
    ("Brazil", "Sao Paulo"): (-23.5505, -46.6333),
    ("North Korea", "Pyongyang"): (39.0392, 125.7625),
}

DEFAULT_LAT_LON = (0.0, 0.0)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the Great-Circle Haversine distance in kilometers between two geographic points.

    Args:
        lat1, lon1: Latitude and Longitude of first point.
        lat2, lon2: Latitude and Longitude of second point.

    Returns:
        float: Distance in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class FeatureEngineer:
    """
    Transforms raw/injected cybersecurity log data into machine-learning feature vectors.
    """

    def __init__(self, scaler_path: Optional[Path] = None):
        """
        Initializes FeatureEngineer.

        Args:
            scaler_path (Optional[Path]): Destination to save/load fitted StandardScaler.
        """
        self.scaler_path = scaler_path or (config.dirs.saved_models_dir / "feature_scaler.joblib")
        self.scaler: Optional[StandardScaler] = None
        self.feature_column_names: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extracts all features, fits StandardScaler on numerical features, and returns feature matrix.

        Args:
            df (pd.DataFrame): Input dataframe with log rows.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Engineered feature dataframe and list of feature column names.
        """
        logger.info("Starting feature engineering extraction pipeline...")
        df_feat = df.copy()

        # Parse Timestamps
        df_feat["dt"] = pd.to_datetime(df_feat["Timestamp"])
        df_feat = df_feat.sort_values(by="dt").reset_index(drop=True)

        # 1. Temporal Features
        logger.info("Extracting temporal features (login_hour, day_of_week, cyclical sin/cos)...")
        df_feat["login_hour"] = df_feat["dt"].dt.hour
        df_feat["day_of_week"] = df_feat["dt"].dt.dayofweek
        df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["login_hour"] / 24.0)
        df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["login_hour"] / 24.0)
        df_feat["day_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7.0)
        df_feat["day_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7.0)

        # 2. Behavioral Window Features (Failed Logins, Unique Devices, Unique Countries)
        logger.info("Extracting rolling behavioral features...")
        df_feat = self._extract_rolling_behavioral(df_feat)

        # 3. Novelty / Anomaly Features (Is New Device, Is New Location)
        logger.info("Extracting novelty features (is_new_device, is_new_location)...")
        df_feat = self._extract_novelty_features(df_feat)

        # 4. Resource Access Frequency
        logger.info("Extracting resource access frequency...")
        res_freq = df_feat.groupby(["User ID", "Resource Accessed"]).cumcount() + 1
        user_total = df_feat.groupby("User ID").cumcount() + 1
        df_feat["resource_access_frequency"] = res_freq / user_total

        # 5. Geographic Distance & Login Velocity Features
        logger.info("Extracting geographic distance and login velocity features...")
        df_feat = self._extract_geo_velocity(df_feat)

        # 6. Session Duration Normalization
        df_feat["session_duration"] = df_feat["Session Duration"].astype(float)
        df_feat["log_session_duration"] = np.log1p(df_feat["session_duration"])

        # 7. Categorical One-Hot Encoding
        logger.info("Encoding categorical variables...")
        cat_cols = ["Authentication Method", "Operating System", "Browser", "Resource Accessed"]
        df_encoded = pd.get_dummies(df_feat[cat_cols], prefix=cat_cols, drop_first=False, dtype=float)

        # Numerical Features to Scale
        numerical_cols = [
            "login_hour",
            "day_of_week",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "failed_login_count_1h",
            "unique_devices_24h",
            "unique_countries_24h",
            "session_duration",
            "log_session_duration",
            "resource_access_frequency",
            "time_since_prev_login_sec",
            "geo_distance_km",
            "login_velocity_kmh"
        ]

        binary_cols = ["is_new_device", "is_new_location"]

        # Scale Numerical Features
        logger.info("Fitting and applying StandardScaler to numerical features...")
        self.scaler = StandardScaler()
        scaled_array = self.scaler.fit_transform(df_feat[numerical_cols])
        df_scaled = pd.DataFrame(scaled_array, columns=[f"{col}_scaled" for col in numerical_cols])

        # Save Scaler Artifact
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Saved fitted StandardScaler artifact to: {self.scaler_path}")

        # Construct Final Feature Matrix
        feature_df = pd.concat([df_scaled, df_feat[binary_cols], df_encoded], axis=1)
        self.feature_column_names = list(feature_df.columns)

        # Append Original Identifiers & Target Labels
        final_df = df_feat[[
            "Timestamp", "User ID", "Device ID", "Source IP", "Country", "City",
            "Login Success"
        ]].copy()

        if "is_anomaly" in df_feat.columns:
            final_df["is_anomaly"] = df_feat["is_anomaly"]
        if "attack_type" in df_feat.columns:
            final_df["attack_type"] = df_feat["attack_type"]

        final_df = pd.concat([final_df, feature_df], axis=1)

        # Clean temporary columns
        if "dt" in final_df.columns:
            final_df.drop(columns=["dt"], inplace=True)

        logger.info(
            f"Feature Engineering Complete. Matrix Shape: {final_df.shape} "
            f"({len(self.feature_column_names)} engineered ML features)."
        )
        return final_df, self.feature_column_names

    def _extract_rolling_behavioral(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling window behavioral features per user."""
        df = df.copy()

        # Failed Login Count in rolling window (1h)
        df["failed_flag"] = (~df["Login Success"]).astype(int)
        failed_counts = []

        for user_id, group in df.groupby("User ID"):
            group_sorted = group.sort_values(by="dt")
            times = group_sorted["dt"].values
            failed = group_sorted["failed_flag"].values
            n = len(group_sorted)
            counts = np.zeros(n)

            for i in range(n):
                t_curr = times[i]
                t_window_start = t_curr - np.timedelta64(1, 'h')
                mask = (times <= t_curr) & (times >= t_window_start)
                counts[i] = failed[mask].sum()

            failed_counts.extend(zip(group_sorted.index, counts))

        failed_df = pd.DataFrame(failed_counts, columns=["index", "failed_login_count_1h"]).set_index("index")
        df["failed_login_count_1h"] = failed_df["failed_login_count_1h"]

        # Unique Devices in rolling window (24h)
        # Approximate using sliding window per user
        unique_devs = []
        unique_cntrs = []

        for user_id, group in df.groupby("User ID"):
            group_sorted = group.sort_values(by="dt")
            times = group_sorted["dt"].values
            devices = group_sorted["Device ID"].values
            countries = group_sorted["Country"].values

            n = len(group_sorted)
            dev_counts = np.zeros(n)
            cntr_counts = np.zeros(n)

            for i in range(n):
                t_curr = times[i]
                t_window_start = t_curr - np.timedelta64(24, 'h')
                mask = (times <= t_curr) & (times >= t_window_start)
                dev_counts[i] = len(set(devices[mask]))
                cntr_counts[i] = len(set(countries[mask]))

            group_sorted["unique_devices_24h"] = dev_counts
            group_sorted["unique_countries_24h"] = cntr_counts
            unique_devs.extend(zip(group_sorted.index, dev_counts))
            unique_cntrs.extend(zip(group_sorted.index, cntr_counts))

        dev_df = pd.DataFrame(unique_devs, columns=["index", "unique_devices_24h"]).set_index("index")
        cntr_df = pd.DataFrame(unique_cntrs, columns=["index", "unique_countries_24h"]).set_index("index")

        df["unique_devices_24h"] = dev_df["unique_devices_24h"]
        df["unique_countries_24h"] = cntr_df["unique_countries_24h"]

        df.drop(columns=["failed_flag"], inplace=True)
        return df

    def _extract_novelty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes is_new_device and is_new_location binary novelty flags."""
        df = df.copy()

        # Cumulative seen devices per user
        is_new_device = []
        is_new_location = []

        seen_devices_per_user: Dict[str, set] = {}
        seen_locations_per_user: Dict[str, set] = {}

        for _, row in df.iterrows():
            u = row["User ID"]
            d = row["Device ID"]
            loc = (row["Country"], row["City"])

            if u not in seen_devices_per_user:
                seen_devices_per_user[u] = set()
                seen_locations_per_user[u] = set()

            is_new_d = 1 if d not in seen_devices_per_user[u] else 0
            is_new_loc = 1 if loc not in seen_locations_per_user[u] else 0

            seen_devices_per_user[u].add(d)
            seen_locations_per_user[u].add(loc)

            is_new_device.append(is_new_d)
            is_new_location.append(is_new_loc)

        df["is_new_device"] = is_new_device
        df["is_new_location"] = is_new_location
        return df

    def _extract_geo_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes time delta, Haversine distance, and physical velocity (km/h)."""
        df = df.copy()

        # Calculate time delta since previous login per user
        df["prev_dt"] = df.groupby("User ID")["dt"].shift(1)
        df["time_since_prev_login_sec"] = (df["dt"] - df["prev_dt"]).dt.total_seconds().fillna(3600.0)

        # Coordinate mappings
        lats = []
        lons = []
        for _, row in df.iterrows():
            key = (row["Country"], row["City"])
            lat, lon = CITY_COORDINATES.get(key, DEFAULT_LAT_LON)
            lats.append(lat)
            lons.append(lon)

        df["lat"] = lats
        df["lon"] = lons

        df["prev_lat"] = df.groupby("User ID")["lat"].shift(1).fillna(df["lat"])
        df["prev_lon"] = df.groupby("User ID")["lon"].shift(1).fillna(df["lon"])

        distances = []
        for _, row in df.iterrows():
            d = haversine_distance_km(row["lat"], row["lon"], row["prev_lat"], row["prev_lon"])
            distances.append(d)

        df["geo_distance_km"] = distances

        # Login Velocity in km/h = distance / (time in hours)
        hours = df["time_since_prev_login_sec"] / 3600.0
        # Replace 0 hours with small epsilon to avoid divide by zero
        hours = np.where(hours <= 0, 0.0001, hours)
        df["login_velocity_kmh"] = df["geo_distance_km"] / hours

        # Clean temporary lat/lon/prev columns
        df.drop(columns=["prev_dt", "lat", "lon", "prev_lat", "prev_lon"], inplace=True)
        return df

    def save_engineered_dataset(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Saves engineered feature dataframe to CSV."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving engineered feature dataset to: {output_path}")
            df.to_csv(output_path, index=False, encoding="utf-8")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Successfully saved feature dataset with {len(df):,} records ({file_size_mb:.2f} MB) "
                f"to {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to save feature dataset to {output_path}: {e}", exc_info=True)
            raise
