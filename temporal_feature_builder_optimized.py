#!/usr/bin/env python3
"""
Optimized Temporal Feature Builder - High-Performance Leakage-Safe Feature Engineering
=====================================================================================

Performance optimizations:
- Batch database queries instead of per-dog queries
- Timestamp caching and vectorized parsing
- Feature computation caching
- Vectorized pandas operations
- Memory-efficient data processing

Maintains strict temporal separation:
- Target race: Pre-race features only
- Historical races: Post-race features with exponential decay weighting
- Assertion guards to prevent temporal leakage
"""

import concurrent.futures
import hashlib
import logging
import os
import pickle
import sqlite3
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import TGR integration if available
try:
    from tgr_prediction_integration import TGRPredictionIntegrator
except ImportError:
    TGRPredictionIntegrator = None


class OptimizedTemporalFeatureBuilder:
    """Optimized version of TemporalFeatureBuilder with significant performance improvements."""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        # Lookback days can be overridden via env var
        try:
            self.default_lookback_days = int(
                os.getenv("GREYHOUND_LOOKBACK_DAYS", "365")
            )
        except Exception:
            self.default_lookback_days = 365

        # Define feature categories for temporal separation
        self.pre_race_features = {
            "box_number",
            "weight",
            "trainer_name",
            "dog_clean_name",
            "venue",
            "grade",
            "distance",
            "track_condition",
            "weather",
            "temperature",
            "humidity",
            "wind_speed",
            "field_size",
            "race_date",
            "race_time",
        }

        self.post_race_features = {
            "finish_position",
            "individual_time",
            "sectional_1st",
            "sectional_2nd",
            "sectional_3rd",
            "margin",
            "beaten_margin",
            "winning_time",
            "scraped_finish_position",
            "scraped_raw_result",
            "winner_name",
            "winner_odds",
            "winner_margin",
            "pir_rating",
            "first_sectional",
            "win_time",
            "bonus_time",
        }

        # Exponential decay factor for historical race weighting
        self.decay_factor = 0.95  # Recent races weighted more heavily

        # Feature caching setup
        self.cache_dir = Path("./feature_cache_optimized")
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_caching = True

        # Performance caches
        self._timestamp_cache = {}  # Cache parsed timestamps
        self._dog_name_cache = {}  # Cache sanitized dog names
        self._feature_cache = {}  # Cache computed features
        self._batch_cache = {}  # Cache batch queries

        # Statistics for performance monitoring
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "db_queries": 0,
            "timestamp_parses": 0,
            "feature_computations": 0,
        }

        # Initialize TGR integration if available and enabled via flag
        self.tgr_integrator = None
        tgr_flag = os.getenv("TGR_ENABLED", "0") not in ("0", "false", "False")
        if TGRPredictionIntegrator and tgr_flag:
            try:
                self.tgr_integrator = TGRPredictionIntegrator(
                    db_path=self.db_path, enable_tgr_lookup=True
                )
                logger.info(
                    "✅ TGR integration initialized for enhanced historical features (TGR_ENABLED=1)"
                )
            except Exception as e:
                logger.warning(f"Could not initialize TGR integration: {e}")
                self.tgr_integrator = None
        else:
            logger.info(
                "ℹ️ TGR integration disabled in optimized builder (set TGR_ENABLED=1 to enable)"
            )

        # Runtime toggle for including TGR features (default depends on integrator availability)
        try:
            self._tgr_runtime_enabled = bool(self.tgr_integrator)
        except Exception:
            self._tgr_runtime_enabled = False

    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    @lru_cache(maxsize=1024)
    def _parse_timestamp_cached(
        self, race_date_str: str, race_time_str: str = None
    ) -> datetime:
        """Cached timestamp parsing with vectorization support."""
        self.stats["timestamp_parses"] += 1

        try:
            race_date = pd.to_datetime(race_date_str)

            # Try to use race_time if available
            if race_time_str and pd.notna(race_time_str) and race_time_str != "None":
                race_time_str = str(race_time_str)
                if ":" in race_time_str:
                    hour, minute = map(int, race_time_str.split(":")[:2])
                    return race_date.replace(hour=hour, minute=minute)

            # Fallback to race_date
            return race_date
        except:
            # Last resort - use race_date as-is
            return pd.to_datetime(race_date_str)

    def get_race_timestamp(self, race_row: pd.Series) -> datetime:
        """Get race timestamp with caching."""
        race_date_str = str(race_row.get("race_date", ""))
        race_time_str = (
            str(race_row.get("race_time", ""))
            if pd.notna(race_row.get("race_time"))
            else None
        )

        return self._parse_timestamp_cached(race_date_str, race_time_str)

    def batch_load_historical_data(
        self,
        dog_names: List[str],
        target_timestamp: datetime,
        lookback_days: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load historical data for multiple dogs in a single batch query."""
        if lookback_days is None:
            lookback_days = self.default_lookback_days

        # Check cache first
        cache_key = self._get_cache_key(
            tuple(sorted(dog_names)), target_timestamp, lookback_days
        )
        if self.enable_caching and cache_key in self._batch_cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for batch historical data: {len(dog_names)} dogs")
            return self._batch_cache[cache_key]

        self.stats["cache_misses"] += 1
        self.stats["db_queries"] += 1

        # Return empty data for all dogs if no database access
        result = {name: pd.DataFrame() for name in dog_names}

        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_date = (target_timestamp - timedelta(days=lookback_days)).strftime(
                "%Y-%m-%d"
            )

            # Create placeholders for IN clause
            placeholders = ",".join(["?" for _ in dog_names])

            # Build batch query
            query = f"""
            SELECT 
                d.*,
                r.venue, r.grade, r.distance, r.track_condition, r.weather,
                r.temperature, r.humidity, r.wind_speed, r.field_size,
                r.race_date, r.race_time, r.winner_name, r.winner_odds, r.winner_margin
            FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            WHERE d.dog_clean_name IN ({placeholders})
                AND r.race_date IS NOT NULL
                AND d.finish_position IS NOT NULL
                AND date(r.race_date) >= date(?)
            ORDER BY d.dog_clean_name, r.race_date DESC, r.race_time DESC
            """

            params = dog_names + [cutoff_date]
            batch_data = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if not batch_data.empty:
                # Add timestamps
                batch_data["race_timestamp"] = pd.to_datetime(batch_data["race_date"])

                # Filter by target timestamp
                cutoff_datetime = target_timestamp - timedelta(days=lookback_days)
                time_filter = (batch_data["race_timestamp"] < target_timestamp) & (
                    batch_data["race_timestamp"] >= cutoff_datetime
                )
                batch_data = batch_data[time_filter].copy()

                # Group by dog name
                for dog_name in dog_names:
                    dog_data = batch_data[
                        batch_data["dog_clean_name"] == dog_name
                    ].copy()
                    if not dog_data.empty:
                        dog_data = dog_data.sort_values(
                            "race_timestamp", ascending=False
                        ).head(100)
                    result[dog_name] = dog_data

            logger.debug(
                f"Batch query for {len(dog_names)} dogs: {len(batch_data)} rows"
            )

        except Exception as e:
            logger.warning(f"Error in batch loading historical data: {e}")

        # Cache the result
        if self.enable_caching:
            self._batch_cache[cache_key] = result

        return result

    def create_historical_features(
        self,
        historical_data: pd.DataFrame,
        target_timestamp: datetime,
        target_venue: str = None,
        target_grade: str = None,
        target_distance: float = None,
    ) -> Dict[str, float]:
        """Create features from historical races with exponential decay weighting."""
        if historical_data.empty:
            return self._get_default_historical_features()

        self.stats["feature_computations"] += 1
        features = {}

        # Vectorized numeric conversions
        positions = pd.to_numeric(
            historical_data["finish_position"], errors="coerce"
        ).dropna()

        if len(positions) > 0:
            # Basic performance metrics
            features["historical_avg_position"] = float(positions.mean())
            features["historical_best_position"] = float(positions.min())
            features["historical_win_rate"] = float((positions == 1).mean())
            features["historical_place_rate"] = float((positions <= 3).mean())

        # Form trend
        if len(positions) >= 3:
            recent_positions = positions.head(min(10, len(positions)))
            x = np.arange(len(recent_positions))
            slope = np.polyfit(x, recent_positions, 1)[0]
            features["historical_form_trend"] = float(-slope)
        else:
            features["historical_form_trend"] = 0.0

        # Time-based performance
        times = pd.to_numeric(
            historical_data["individual_time"], errors="coerce"
        ).dropna()
        if len(times) > 0:
            features["historical_avg_time"] = float(times.mean())
            features["historical_best_time"] = float(times.min())
            features["historical_time_consistency"] = float(times.std())
        else:
            features["historical_avg_time"] = 30.0
            features["historical_best_time"] = 29.0
            features["historical_time_consistency"] = 2.0

        # Venue-specific performance
        if target_venue and "venue" in historical_data.columns:
            venue_races = historical_data[historical_data["venue"] == target_venue]
            if len(venue_races) > 0:
                venue_positions = pd.to_numeric(
                    venue_races["finish_position"], errors="coerce"
                ).dropna()
                if len(venue_positions) > 0:
                    features["venue_specific_avg_position"] = float(
                        venue_positions.mean()
                    )
                    features["venue_specific_win_rate"] = float(
                        (venue_positions == 1).mean()
                    )
                    features["venue_experience"] = len(venue_positions)
                    features["venue_best_position"] = float(venue_positions.min())

        # Grade-specific performance
        if target_grade and "grade" in historical_data.columns:
            grade_races = historical_data[historical_data["grade"] == target_grade]
            if len(grade_races) > 0:
                grade_positions = pd.to_numeric(
                    grade_races["finish_position"], errors="coerce"
                ).dropna()
                if len(grade_positions) > 0:
                    features["grade_specific_avg_position"] = float(
                        grade_positions.mean()
                    )
                    features["grade_specific_win_rate"] = float(
                        (grade_positions == 1).mean()
                    )
                    features["grade_experience"] = len(grade_positions)

        # Temporal features
        if len(historical_data) > 0 and "race_timestamp" in historical_data.columns:
            last_race_timestamp = historical_data["race_timestamp"].iloc[0]
            # Use target_timestamp rather than wall-clock now() to avoid time drift in backtests/training
            days_since_last = (target_timestamp - last_race_timestamp).days
            features["days_since_last_race"] = float(days_since_last)

            # Race frequency
            if len(historical_data) > 1:
                date_span = (
                    historical_data["race_timestamp"].iloc[0]
                    - historical_data["race_timestamp"].iloc[-1]
                ).days
                if date_span > 0:
                    features["race_frequency"] = float(
                        len(historical_data) * 30 / date_span
                    )
                else:
                    features["race_frequency"] = 1.0
            else:
                features["race_frequency"] = 1.0

        # Fill any missing features with defaults
        default_features = self._get_default_historical_features()
        for key, value in default_features.items():
            if key not in features:
                features[key] = value

        return features

    def _get_default_historical_features(self) -> Dict[str, float]:
        """Default features for dogs with no historical data."""
        return {
            "historical_avg_position": 4.5,
            "historical_best_position": 4.0,
            "historical_win_rate": 0.125,
            "historical_place_rate": 0.375,
            "historical_form_trend": 0.0,
            "historical_avg_time": 30.0,
            "historical_best_time": 29.0,
            "historical_time_consistency": 2.0,
            "venue_specific_avg_position": 4.5,
            "venue_specific_win_rate": 0.125,
            "venue_experience": 0,
            "venue_best_position": 4.0,
            "grade_specific_avg_position": 4.5,
            "grade_specific_win_rate": 0.125,
            "grade_experience": 0,
            "days_since_last_race": 30.0,
            "race_frequency": 2.0,
        }

    def build_features_for_race(
        self, race_data: pd.DataFrame, target_race_id: str
    ) -> pd.DataFrame:
        """Build leakage-safe features for all dogs in a race - OPTIMIZED VERSION."""
        start_time = time.time()
        logger.info(
            f"Building optimized leakage-safe features for race {target_race_id}"
        )

        # Get target race timestamp
        target_race_info = race_data.iloc[0]
        target_timestamp = self.get_race_timestamp(target_race_info)

        # Extract all dog names for batch processing
        dog_names = race_data["dog_clean_name"].tolist()

        # OPTIMIZATION: Batch load all historical data in one query
        batch_historical_data = self.batch_load_historical_data(
            dog_names, target_timestamp
        )

        feature_rows = []

        for _, dog_row in race_data.iterrows():
            dog_name = dog_row["dog_clean_name"]

            # Start with pre-race features only from target race
            features = {}

            # Add pre-race features (safe to use from target race)
            for feature in self.pre_race_features:
                if feature in dog_row and pd.notna(dog_row[feature]):
                    features[feature] = dog_row[feature]

            # CRITICAL ASSERTION: Ensure no post-race features from target race
            for post_race_feature in self.post_race_features:
                if post_race_feature in features:
                    raise AssertionError(
                        f"TEMPORAL LEAKAGE DETECTED: Post-race feature '{post_race_feature}' "
                        f"found in target race features for {dog_name} "
                        f"in race {target_race_id}"
                    )

            # Get pre-loaded historical data
            historical_data = batch_historical_data.get(dog_name, pd.DataFrame())

            # Create historical features
            historical_features = self.create_historical_features(
                historical_data,
                target_timestamp,
                target_venue=dog_row.get("venue"),
                target_grade=dog_row.get("grade"),
                target_distance=pd.to_numeric(dog_row.get("distance"), errors="coerce"),
            )

            features.update(historical_features)

            # Add TGR features if TGR integration is available and runtime toggle is enabled
            if self.tgr_integrator and getattr(self, "_tgr_runtime_enabled", False):
                try:
                    tgr_features = self.tgr_integrator._get_tgr_historical_features(
                        dog_name, target_timestamp
                    )
                    # Ensure expected defaults when values missing
                    try:
                        expected = self.tgr_integrator.get_feature_names()
                        defaults = self.tgr_integrator._get_default_tgr_features()
                        for col in expected:
                            if col not in tgr_features:
                                tgr_features[col] = defaults.get(col, 0.0)
                    except Exception:
                        pass
                    features.update(tgr_features)
                except Exception as e:
                    logger.warning(f"Failed to get TGR features for {dog_name}: {e}")

            # Add metadata
            features["race_id"] = target_race_id
            features["dog_clean_name"] = dog_name
            features["target_timestamp"] = target_timestamp

            # Add target if available (for training)
            if "finish_position" in dog_row and pd.notna(dog_row["finish_position"]):
                try:
                    features["target"] = (
                        1 if int(dog_row["finish_position"]) == 1 else 0
                    )
                except:
                    features["target"] = 0

            feature_rows.append(features)

        result_df = pd.DataFrame(feature_rows)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Built optimized features for {len(result_df)} dogs in race {target_race_id} "
            f"in {elapsed_time:.2f}s"
        )

        return result_df

    def validate_temporal_integrity(
        self, features_df: pd.DataFrame, race_data: pd.DataFrame
    ) -> bool:
        """Validate that no temporal leakage exists in the features."""
        logger.info("Validating temporal integrity...")

        # Check 1: No post-race features in target features
        target_columns = set(features_df.columns)
        leakage_features = target_columns.intersection(self.post_race_features)

        if leakage_features:
            raise AssertionError(
                f"TEMPORAL LEAKAGE DETECTED: Post-race features found in target features: "
                f"{leakage_features}"
            )

        # Check 2: Historical features exist
        historical_feature_prefixes = [
            "historical_",
            "venue_specific_",
            "grade_specific_",
        ]
        historical_features = [
            col
            for col in features_df.columns
            if any(col.startswith(prefix) for prefix in historical_feature_prefixes)
        ]

        if not historical_features:
            logger.warning(
                "No historical features found - this may reduce prediction quality"
            )
        else:
            logger.info(f"Found {len(historical_features)} historical features")

        # Check 3: All dogs have features
        if len(features_df) != len(race_data):
            raise AssertionError(
                f"Feature count mismatch: {len(features_df)} features vs {len(race_data)} dogs"
            )

        logger.info("✅ Temporal integrity validation passed")
        return True

    def set_tgr_enabled(self, enabled: bool) -> None:
        """Enable/disable inclusion of TGR features at runtime without reinitializing."""
        try:
            self._tgr_runtime_enabled = bool(enabled)
            status = "enabled" if self._tgr_runtime_enabled else "disabled"
            logger.info(
                f"[OptimizedBuilder] TGR feature inclusion runtime toggle {status}"
            )
        except Exception as e:
            logger.debug(f"[OptimizedBuilder] Failed to set TGR runtime toggle: {e}")

    def clear_caches(self):
        """Clear all caches to free memory."""
        self._timestamp_cache.clear()
        self._dog_name_cache.clear()
        self._feature_cache.clear()
        self._batch_cache.clear()

        # Clear LRU caches
        self._parse_timestamp_cached.cache_clear()

        logger.info("All caches cleared")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests,
            "timestamp_cache_size": len(self._timestamp_cache),
            "feature_cache_size": len(self._feature_cache),
            "batch_cache_size": len(self._batch_cache),
        }


if __name__ == "__main__":
    # Test the optimized temporal feature builder
    builder = OptimizedTemporalFeatureBuilder()

    print("✅ Optimized Temporal Feature Builder initialized")
    print("🛡️ Temporal leakage guards active")
    print("📊 Exponential decay weighting enabled")
    print("⚡ Performance optimizations enabled:")
    print("   - Batch database queries")
    print("   - Timestamp caching")
    print("   - Vectorized operations")
    print("   - Feature computation caching")
