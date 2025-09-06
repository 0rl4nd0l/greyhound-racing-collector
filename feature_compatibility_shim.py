#!/usr/bin/env python3
"""
Feature Compatibility Shim
===========================

Temporary adapter to bridge the schema mismatch between the trained model's
expected features and the current feature builder output.

This shim:
1. Adds missing expected columns with default/imputed values
2. Orders columns to match the model's expected order
3. Logs warnings to prompt a proper pipeline fix

Environment Variables:
- ENABLE_FEATURE_COMPAT_SHIM=1: Enable the compatibility shim
- FEATURE_SHIM_LOG_LEVEL=INFO: Set logging level for shim warnings

Author: AI Assistant
Date: August 30, 2025
"""

import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureCompatibilityShim:
    """Compatibility adapter for bridging feature schema mismatches."""

    def __init__(self, expected_features: List[str]):
        self.expected_features = expected_features
        self.enabled = os.getenv("ENABLE_FEATURE_COMPAT_SHIM", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.log_level = os.getenv("FEATURE_SHIM_LOG_LEVEL", "WARNING").upper()

        # Set up logging level
        if hasattr(logging, self.log_level):
            self.logger_level = getattr(logging, self.log_level)
        else:
            self.logger_level = logging.WARNING

        if self.enabled:
            logger.log(
                self.logger_level,
                f"🔧 Feature Compatibility Shim ENABLED - expected {len(expected_features)} features",
            )

        # Define default values for missing features by category
        self.default_values = {
            # TGR-derived features (18 missing)
            "tgr_avg_finish_position": 4.5,  # Average position for 8-dog field
            "tgr_best_finish_position": 3.0,  # Reasonable best position
            "tgr_consistency": 0.5,  # Middle consistency score
            "tgr_days_since_last_race": 14.0,  # 2 weeks since last race
            "tgr_form_trend": 0.0,  # Neutral form trend
            "tgr_has_comments": 0.0,  # No comments flag
            "tgr_last_race_position": 4.5,  # Average last race position
            "tgr_place_rate": 0.375,  # 3/8 place rate for typical dog
            "tgr_preferred_distance": 500.0,  # Common greyhound distance
            "tgr_preferred_distance_avg": 4.5,  # Average position at preferred distance
            "tgr_preferred_distance_races": 10.0,  # Number of races at preferred distance
            "tgr_recent_avg_position": 4.5,  # Recent average position
            "tgr_recent_best_position": 2.0,  # Recent best position
            "tgr_recent_races": 5.0,  # Number of recent races
            "tgr_sentiment_score": 0.0,  # Neutral sentiment
            "tgr_total_races": 25.0,  # Reasonable career race count
            "tgr_venues_raced": 3.0,  # Number of venues raced at
            "tgr_win_rate": 0.125,  # 1/8 win rate for typical dog
            # Race condition features (2 missing)
            "track_condition": "Good",  # Most common track condition
            "weather": "Clear",  # Most common weather
            # Dog/trainer features (2 missing)
            "trainer_name": "Unknown",  # Unknown trainer
            "weight": 32.5,  # Average greyhound weight in kg
            # Time-related features (1 missing)
            "race_time": "19:00",  # Typical evening race time
        }

    def apply_compatibility_fixes(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply compatibility fixes to align features with model expectations."""

        if not self.enabled:
            return features_df

        if features_df.empty:
            logger.warning("Empty features DataFrame passed to compatibility shim")
            return features_df

        # Make a copy to avoid modifying the original
        fixed_df = features_df.copy()

        # Identify missing and extra features
        current_features = set(fixed_df.columns)
        expected_features_set = set(self.expected_features)

        missing_features = expected_features_set - current_features
        extra_features = current_features - expected_features_set

        if missing_features or extra_features:
            logger.log(self.logger_level, f"🔧 FEATURE COMPATIBILITY SHIM ACTIVE:")
            logger.log(
                self.logger_level, f"   Missing features: {len(missing_features)}"
            )
            logger.log(self.logger_level, f"   Extra features: {len(extra_features)}")
            logger.log(
                self.logger_level,
                f"   ⚠️  This is a temporary fix - restore proper feature pipeline!",
            )

        # Add missing features with default values
        for feature in missing_features:
            default_value = self.default_values.get(feature, 0.0)
            fixed_df[feature] = default_value

            logger.debug(
                f"   Added missing feature '{feature}' with default value: {default_value}"
            )

        # Drop extra features (keep metadata columns)
        metadata_cols = {"race_id", "dog_clean_name", "target", "target_timestamp"}
        features_to_drop = [col for col in extra_features if col not in metadata_cols]

        if features_to_drop:
            fixed_df = fixed_df.drop(columns=features_to_drop)
            logger.debug(f"   Dropped {len(features_to_drop)} unexpected features")

        # Reorder columns to match expected order (keeping metadata at end)
        feature_cols = [
            col for col in self.expected_features if col in fixed_df.columns
        ]
        metadata_cols_present = [
            col for col in metadata_cols if col in fixed_df.columns
        ]

        final_column_order = feature_cols + metadata_cols_present
        fixed_df = fixed_df[final_column_order]

        # Data type consistency fixes
        fixed_df = self._fix_data_types(fixed_df)

        logger.log(
            self.logger_level,
            f"✅ Feature compatibility fixes applied: {len(fixed_df.columns)} columns, {len(fixed_df)} rows",
        )

        return fixed_df

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix common data type issues."""

        # Categorical features that should be strings
        categorical_features = [
            "venue",
            "grade",
            "track_condition",
            "weather",
            "trainer_name",
        ]

        for feature in categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(str)

        # Numeric features that should be float
        numeric_features = [
            col
            for col in df.columns
            if col.startswith(("tgr_", "historical_", "venue_", "grade_"))
            or col
            in [
                "box_number",
                "weight",
                "distance",
                "temperature",
                "humidity",
                "wind_speed",
                "field_size",
            ]
        ]

        for feature in numeric_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)

        return df

    def validate_output_schema(self, df: pd.DataFrame) -> bool:
        """Validate that the output DataFrame matches expected schema."""

        if not self.enabled:
            return True

        # Check that all expected features are present
        missing = set(self.expected_features) - set(df.columns)
        if missing:
            logger.error(f"Validation failed: missing features after shim: {missing}")
            return False

        # Check for any NaN values in critical features
        critical_features = ["box_number", "distance", "venue", "grade"]
        for feature in critical_features:
            if feature in df.columns and df[feature].isna().any():
                logger.warning(f"NaN values found in critical feature '{feature}'")

        logger.debug(f"✅ Output schema validation passed: {len(df.columns)} columns")
        return True


def apply_feature_compatibility_shim(
    features_df: pd.DataFrame, expected_features: List[str]
) -> pd.DataFrame:
    """Convenience function to apply compatibility shim."""

    shim = FeatureCompatibilityShim(expected_features)
    return shim.apply_compatibility_fixes(features_df)


# Test function to demonstrate usage
def test_compatibility_shim():
    """Test the compatibility shim with sample data."""

    # Sample current features (subset)
    current_data = {
        "box_number": [1, 2, 3],
        "venue": ["GEE", "GEE", "GEE"],
        "grade": ["5", "5", "5"],
        "distance": [500, 500, 500],
        "historical_win_rate": [0.1, 0.2, 0.15],
        "dog_clean_name": ["DOG_A", "DOG_B", "DOG_C"],
        "race_id": ["test_race"] * 3,
    }

    current_df = pd.DataFrame(current_data)

    # Expected features (from trained model)
    expected_features = [
        "box_number",
        "venue",
        "grade",
        "distance",
        "historical_win_rate",
        "tgr_win_rate",
        "tgr_avg_finish_position",
        "track_condition",
        "weather",
        "trainer_name",
        "weight",
    ]

    # Enable shim for testing
    os.environ["ENABLE_FEATURE_COMPAT_SHIM"] = "1"

    print("=== TESTING FEATURE COMPATIBILITY SHIM ===")
    print(f"Input features: {list(current_df.columns)}")
    print(f"Expected features: {expected_features}")

    # Apply shim
    fixed_df = apply_feature_compatibility_shim(current_df, expected_features)

    print(f"Output features: {list(fixed_df.columns)}")
    print(f"Rows: {len(fixed_df)}")
    print("\nSample output:")
    print(fixed_df.head())

    return fixed_df


if __name__ == "__main__":
    # Run test
    test_compatibility_shim()
