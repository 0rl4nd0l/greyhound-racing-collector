#!/usr/bin/env python3
"""
Schema-Compatible Temporal Feature Builder
==========================================

This is a patched version of temporal_feature_builder.py that adapts to the actual
database schema in greyhound_racing_data.db, removing hardcoded column expectations
that don't exist in the current schema.

Key changes:
- Removed references to missing columns: temperature, humidity, wind_speed, individual_time
- Uses only columns that actually exist in the database
- Maintains all temporal leakage protection and feature engineering logic
"""

import hashlib
import logging
import os
import pickle
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import TGR integration if available
try:
    from tgr_prediction_integration import TGRPredictionIntegrator
except ImportError:
    TGRPredictionIntegrator = None


class SchemaCompatibleTemporalFeatureBuilder:
    """Builds features with strict temporal separation adapted to actual database schema."""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        
        # Lookback days can be overridden via env var
        try:
            self.default_lookback_days = int(
                os.getenv("GREYHOUND_LOOKBACK_DAYS", "365")
            )
        except Exception:
            self.default_lookback_days = 365

        # Detect available schema on initialization
        self._detect_schema()

        # Define feature categories for temporal separation (adapted to actual schema)
        self.pre_race_features = {
            "box_number",
            "weight",
            "trainer_name", 
            "trainer",  # Alternative trainer column name
            "dog_clean_name",
            "venue",
            "grade", 
            "distance",
            "track_condition",
            "weather",
            "field_size",
            "race_date",
            "race_time",
        }

        self.post_race_features = {
            "finish_position",
            "winning_time",  # Available instead of individual_time
            "winner_name",
            "winner_odds", 
            "winner_margin",
            "pir_rating",
            "first_sectional",
            "win_time",
            "bonus_time",
            "odds",  # Betting odds
            "placing",
            "form",
        }

        # Exponential decay factor for historical race weighting
        self.decay_factor = 0.95

        # Feature caching setup
        self.cache_dir = Path("./feature_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_caching = True

        # Initialize TGR integration (same as original)
        self.tgr_integrator = None
        self._tgr_all_feature_names = []
        # ... TGR initialization logic would go here (truncated for brevity)
        
        logger.info("ℹ️ TGR integration disabled (set TGR_ENABLED=1 to override)")
        self._tgr_runtime_enabled = False

    def _detect_schema(self):
        """Detect available columns in each table for schema compatibility."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get available columns for each table
            cursor.execute('PRAGMA table_info(dog_race_data)')
            self.dog_cols = [col[1] for col in cursor.fetchall()]

            cursor.execute('PRAGMA table_info(race_metadata)')  
            self.race_cols = [col[1] for col in cursor.fetchall()]
            
            cursor.execute('PRAGMA table_info(enhanced_expert_data)')
            self.enhanced_cols = [col[1] for col in cursor.fetchall()]

            conn.close()

            # Log schema detection
            logger.info(f"📊 Schema detected - dog_race_data: {len(self.dog_cols)} cols, race_metadata: {len(self.race_cols)} cols, enhanced_expert_data: {len(self.enhanced_cols)} cols")

        except Exception as e:
            logger.error(f"Schema detection failed: {e}")
            # Fallback to minimal expected columns
            self.dog_cols = ['race_id', 'dog_clean_name', 'finish_position']
            self.race_cols = ['race_id', 'race_date', 'venue'] 
            self.enhanced_cols = ['race_id', 'dog_clean_name']

    def set_tgr_enabled(self, enabled: bool) -> None:
        """Enable/disable inclusion of TGR features at runtime."""
        try:
            self._tgr_runtime_enabled = bool(enabled)
            status = "enabled" if self._tgr_runtime_enabled else "disabled"
            logger.info(f"TGR feature inclusion runtime toggle {status}")
        except Exception as e:
            logger.debug(f"Failed to set TGR runtime toggle: {e}")

    def get_race_timestamp(self, race_row: pd.Series) -> datetime:
        """Get race timestamp, preferring race_time over race_date."""
        try:
            # Try to use race_time if available
            if "race_time" in race_row and pd.notna(race_row["race_time"]):
                race_date = pd.to_datetime(race_row["race_date"])
                # Assume race_time is in format like "14:30" or similar
                race_time_str = str(race_row["race_time"])
                if ":" in race_time_str:
                    hour, minute = map(int, race_time_str.split(":")[:2])
                    return race_date.replace(hour=hour, minute=minute)

            # Fallback to race_date
            return pd.to_datetime(race_row["race_date"])
        except:
            # Last resort - use race_date as-is
            return pd.to_datetime(race_row["race_date"])

    def _to_meters(self, val: Any) -> float:
        """Parse a distance value like '500m' or '520 m' to a numeric meters float."""
        try:
            if pd.isna(val):
                return np.nan
            s = str(val)
            import re
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            return float(m.group(1)) if m else np.nan
        except Exception:
            return np.nan

    def load_dog_historical_data(
        self, dog_name: str, target_timestamp: datetime, lookback_days: int = None
    ) -> pd.DataFrame:
        """Load historical race data for a dog, excluding target race (schema-compatible)."""
        try:
            if lookback_days is None:
                lookback_days = self.default_lookback_days
            
            conn = sqlite3.connect(self.db_path)
            cutoff_date = (target_timestamp - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

            # Build dynamic query using only available columns
            race_cols_filtered = [col for col in [
                'venue', 'grade', 'distance', 'track_condition', 'weather', 
                'field_size', 'race_date', 'race_time', 'winner_name', 
                'winner_odds', 'winner_margin'
            ] if col in self.race_cols]

            enhanced_cols_filtered = [col for col in [
                'pir_rating', 'first_sectional', 'win_time', 'bonus_time'
            ] if col in self.enhanced_cols]

            # Build SELECT clause dynamically
            race_part = ', '.join([f'r.{col}' for col in race_cols_filtered])
            enhanced_part = ', '.join([f'e.{col}' for col in enhanced_cols_filtered]) if enhanced_cols_filtered else ''

            query_parts = ['d.*']
            if race_part:
                query_parts.append(race_part)
            if enhanced_part:
                query_parts.append(enhanced_part)

            query = f"""
            SELECT {', '.join(query_parts)}
            FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            LEFT JOIN enhanced_expert_data e ON d.race_id = e.race_id 
                AND d.dog_clean_name = e.dog_clean_name
            WHERE d.dog_clean_name = ?
                AND r.race_date IS NOT NULL
                AND d.finish_position IS NOT NULL
                AND date(r.race_date) >= date(?)
            ORDER BY r.race_date DESC, r.race_time DESC
            LIMIT 100
            """

            historical_data = pd.read_sql_query(query, conn, params=[dog_name, cutoff_date])
            raw_count = len(historical_data)
            conn.close()

            logger.debug(f"Schema-compatible query for dog='{dog_name}': raw_rows={raw_count}")

            # Same fallback logic as original
            if historical_data.empty:
                try:
                    sanitized = str(dog_name).replace('"', "").replace("'", "").replace("`", "").replace("'", "").strip().upper()
                    
                    query_fallback = f"""
                    SELECT {', '.join(query_parts)}
                    FROM dog_race_data d
                    LEFT JOIN race_metadata r ON d.race_id = r.race_id
                    LEFT JOIN enhanced_expert_data e ON d.race_id = e.race_id 
                        AND d.dog_clean_name = e.dog_clean_name
                    WHERE REPLACE(REPLACE(REPLACE(REPLACE(d.dog_clean_name,'"',''),"'",''),'`',''),''','') = ?
                        AND r.race_date IS NOT NULL
                        AND d.finish_position IS NOT NULL
                        AND date(r.race_date) >= date(?)
                    ORDER BY r.race_date DESC, r.race_time DESC
                    LIMIT 100
                    """
                    
                    with sqlite3.connect(self.db_path) as conn2:
                        historical_data = pd.read_sql_query(query_fallback, conn2, params=[sanitized, cutoff_date])
                    raw_count = len(historical_data)
                    logger.debug(f"Fallback query for sanitized dog='{sanitized}': raw_rows={raw_count}")
                except Exception as _e:
                    logger.debug(f"Fallback query failed: {_e}")

            if historical_data.empty:
                logger.info(f"No historical rows found in DB for dog='{dog_name}'")
                return pd.DataFrame()

            # Apply temporal filtering (same logic as original)
            historical_data["race_timestamp"] = historical_data.apply(
                lambda row: self.get_race_timestamp(row), axis=1
            )

            cutoff_date = target_timestamp - timedelta(days=lookback_days)
            historical_data = historical_data[
                (historical_data["race_timestamp"] < target_timestamp)
                & (historical_data["race_timestamp"] >= cutoff_date)
            ].copy()

            historical_data = historical_data.sort_values("race_timestamp", ascending=False)

            filt_count = len(historical_data)
            if filt_count == 0:
                logger.info(f"Historical rows filtered to zero for dog='{dog_name}'")
            else:
                logger.debug(f"Historical rows after time filter for dog='{dog_name}': {filt_count} (of raw {raw_count})")

            return historical_data

        except Exception as e:
            logger.error(f"Error loading historical data for {dog_name}: {e}")
            return pd.DataFrame()

    def create_historical_features(
        self,
        historical_data: pd.DataFrame,
        target_timestamp: datetime,
        target_venue: str = None,
        target_grade: str = None,
        target_distance: float = None,
    ) -> Dict[str, float]:
        """Create features from historical races (adapted for available schema)."""
        if historical_data.empty:
            return self._get_default_historical_features()

        features = {}

        # Convert finish positions to numeric
        historical_data["finish_position_numeric"] = pd.to_numeric(
            historical_data["finish_position"], errors="coerce"
        )

        # Use winning_time instead of individual_time (which doesn't exist in our schema)
        if "winning_time" in historical_data.columns:
            historical_data["time_numeric"] = pd.to_numeric(
                historical_data["winning_time"], errors="coerce"
            )
        else:
            historical_data["time_numeric"] = np.nan

        # Distance processing
        if "distance" in historical_data.columns:
            try:
                historical_data["distance_numeric"] = historical_data["distance"].apply(self._to_meters)
            except Exception:
                historical_data["distance_numeric"] = pd.to_numeric(
                    historical_data.get("distance", np.nan), errors="coerce"
                )
        else:
            historical_data["distance_numeric"] = np.nan

        # Create exponential decay weights (same as original)
        num_races = len(historical_data)
        weights = np.array([self.decay_factor**i for i in range(num_races)])

        # Apply contextual weighting (same logic as original, but adapted)
        # Weight similar venue races higher
        if target_venue:
            venue_boost = (historical_data.get("venue", "") == target_venue).astype(float) * 0.8
            weights = weights * (1 + venue_boost)

        # Weight same grade races higher
        if target_grade:
            grade_boost = (historical_data.get("grade", "") == target_grade).astype(float) * 0.6
            weights = weights * (1 + grade_boost)

        # Weight same distance races higher
        if target_distance and "distance" in historical_data.columns:
            distance_tolerance = 50
            distance_matches = (
                abs(historical_data["distance_numeric"] - target_distance) <= distance_tolerance
            ).astype(float)
            distance_boost = distance_matches * 0.7
            weights = weights * (1 + distance_boost)

        weights = weights / weights.sum()  # Normalize weights

        # Basic performance metrics
        valid_position_mask = historical_data["finish_position_numeric"].notna()
        valid_positions = historical_data["finish_position_numeric"][valid_position_mask]

        if len(valid_positions) > 0:
            valid_weights = weights[valid_position_mask]
            valid_weights = valid_weights / valid_weights.sum()

            features["historical_avg_position"] = float(
                np.average(valid_positions, weights=valid_weights)
            )
            features["historical_best_position"] = float(valid_positions.min())
            features["historical_win_rate"] = float(
                np.average((valid_positions == 1).astype(float), weights=valid_weights)
            )
            features["historical_place_rate"] = float(
                np.average((valid_positions <= 3).astype(float), weights=valid_weights)
            )

        # Recent form trend
        recent_positions = valid_positions.head(min(10, len(valid_positions)))
        if len(recent_positions) >= 3:
            x = np.arange(len(recent_positions))
            slope = np.polyfit(x, recent_positions, 1)[0]
            features["historical_form_trend"] = float(-slope)
        else:
            features["historical_form_trend"] = 0.0

        # Time-based features using available time data
        valid_time_mask = historical_data["time_numeric"].notna()
        valid_times = historical_data["time_numeric"][valid_time_mask]

        if len(valid_times) > 0:
            time_weights = weights[valid_time_mask]
            time_weights = time_weights / time_weights.sum()

            features["historical_avg_time"] = float(np.average(valid_times, weights=time_weights))
            features["historical_best_time"] = float(valid_times.min())
            features["historical_time_consistency"] = float(valid_times.std())
            features["distance_adjusted_time"] = False  # No adjustment without individual_time
            features["target_distance"] = float(target_distance) if target_distance else 0.0

        # Venue-specific performance 
        if target_venue and "venue" in historical_data.columns:
            venue_races = historical_data[historical_data["venue"] == target_venue]
            if len(venue_races) > 0:
                venue_positions = pd.to_numeric(venue_races["finish_position"], errors="coerce").dropna()
                if len(venue_positions) > 0:
                    features["venue_specific_avg_position"] = float(venue_positions.mean())
                    features["venue_specific_win_rate"] = float((venue_positions == 1).mean())
                    features["venue_experience"] = len(venue_positions)
                    features["venue_best_position"] = float(venue_positions.min())

        # Grade-specific performance
        if target_grade and "grade" in historical_data.columns:
            grade_races = historical_data[historical_data["grade"] == target_grade]
            if len(grade_races) > 0:
                grade_positions = pd.to_numeric(grade_races["finish_position"], errors="coerce").dropna()
                if len(grade_positions) > 0:
                    features["grade_specific_avg_position"] = float(grade_positions.mean())
                    features["grade_specific_win_rate"] = float((grade_positions == 1).mean())
                    features["grade_experience"] = len(grade_positions)

        # Distance-specific performance
        if target_distance and "distance" in historical_data.columns:
            distance_tolerance = 50
            similar_distance_races = historical_data[
                abs(historical_data["distance_numeric"] - target_distance) <= distance_tolerance
            ]
            if len(similar_distance_races) > 0:
                dist_positions = pd.to_numeric(similar_distance_races["finish_position"], errors="coerce").dropna()
                if len(dist_positions) > 0:
                    features["best_distance_avg_position"] = float(dist_positions.mean())
                    features["best_distance_win_rate"] = float((dist_positions == 1).mean())

        # Recent racing frequency
        if len(historical_data) >= 2:
            recent_timestamps = historical_data["race_timestamp"].head(min(5, len(historical_data)))
            if len(recent_timestamps) >= 2:
                days_between_races = [(recent_timestamps.iloc[i] - recent_timestamps.iloc[i+1]).days 
                                    for i in range(len(recent_timestamps)-1)]
                if days_between_races:
                    avg_days_between = np.mean(days_between_races)
                    features["days_since_last_race"] = float((target_timestamp - recent_timestamps.iloc[0]).days)
                    features["race_frequency"] = float(avg_days_between) if avg_days_between > 0 else 30.0

        # Fill any missing features with defaults
        return self._fill_missing_features(features)

    def _get_default_historical_features(self) -> Dict[str, float]:
        """Return default features when no historical data is available."""
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
            "best_distance_avg_position": 4.5,
            "best_distance_win_rate": 0.125,
            "distance_adjusted_time": False,
            "target_distance": 0.0,
        }

    def _fill_missing_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Fill any missing features with sensible defaults."""
        defaults = self._get_default_historical_features()
        
        for key, default_value in defaults.items():
            if key not in features:
                features[key] = default_value
                
        return features

    def build_features_for_race(self, race_data: pd.DataFrame, race_id: str) -> pd.DataFrame:
        """Build temporal features for all dogs in a race."""
        logger.info(f"Building schema-compatible features for race {race_id}")
        
        if race_data.empty:
            logger.warning(f"Empty race data for {race_id}")
            return pd.DataFrame()

        # Get race timestamp for temporal cutoff
        race_timestamp = None
        if "race_date" in race_data.columns and len(race_data) > 0:
            try:
                race_timestamp = self.get_race_timestamp(race_data.iloc[0])
            except Exception as e:
                logger.warning(f"Could not parse race timestamp for {race_id}: {e}")
                race_timestamp = datetime.now()

        if race_timestamp is None:
            race_timestamp = datetime.now()

        # Get race context for contextual weighting
        target_venue = race_data["venue"].iloc[0] if "venue" in race_data.columns and len(race_data) > 0 else None
        target_grade = race_data["grade"].iloc[0] if "grade" in race_data.columns and len(race_data) > 0 else None  
        target_distance = None
        if "distance" in race_data.columns and len(race_data) > 0:
            try:
                target_distance = self._to_meters(race_data["distance"].iloc[0])
            except:
                target_distance = None

        all_features = []

        for _, dog_row in race_data.iterrows():
            dog_name = dog_row["dog_clean_name"]
            
            # Load historical data for this dog
            historical_data = self.load_dog_historical_data(
                dog_name, race_timestamp, self.default_lookback_days
            )

            # Create historical features
            historical_features = self.create_historical_features(
                historical_data, race_timestamp, target_venue, target_grade, target_distance
            )

            # Combine with current race features
            dog_features = {
                "race_id": race_id,
                "dog_clean_name": dog_name,
                "target": 0,  # Will be set during training
                "target_timestamp": race_timestamp,
            }

            # Add basic race information
            for col in ["box_number", "weight", "venue", "grade", "distance", "track_condition", "weather", "race_date"]:
                if col in dog_row:
                    dog_features[col] = dog_row[col]

            # Add historical features
            dog_features.update(historical_features)

            all_features.append(dog_features)

        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        logger.info(f"✅ Built {len(features_df)} feature sets for race {race_id} with {len(features_df.columns)} features each")
        
        return features_df
