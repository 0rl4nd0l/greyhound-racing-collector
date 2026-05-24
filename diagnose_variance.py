#!/usr/bin/env python3
"""
Variance Diagnostic Script
=========================

Diagnose why prediction variance is too low and suggest fixes.
"""

import numpy as np
import pandas as pd

from ml_system_v4 import MLSystemV4
from prediction_pipeline_v4 import PredictionPipelineV4


def diagnose_variance():
    """Diagnose variance issues in predictions"""
    print("🔍 VARIANCE DIAGNOSTIC ANALYSIS")
    print("=" * 60)

    # Test race file
    race_file = "/Users/orlandolee/greyhound_racing_collector/archive/corrupt_or_legacy_race_files/20250730162231_Race 1 - RICH - 04 July 2025.csv"

    try:
        # Initialize ML system directly to access raw probabilities
        ml_system = MLSystemV4()

        # Load race data
        race_data = pd.read_csv(race_file)
        print(f"📊 Analyzing race with {len(race_data)} dogs")

        # Preprocess data
        race_id = "diagnostic_race"
        race_data["race_id"] = race_id
        race_data["race_date"] = "2025-07-04"
        race_data["race_time"] = "14:30"

        # Clean dog names
        race_data["dog_clean_name"] = race_data["Dog Name"].str.replace(
            r"^\d+\.\s*", "", regex=True
        )
        race_data["box_number"] = race_data["BOX"]
        race_data["weight"] = race_data["WGT"]
        race_data["venue"] = "RICH"
        race_data["grade"] = race_data["G"]
        race_data["distance"] = race_data["DIST"]

        # Add required fields
        required_fields = {
            "finish_position": None,
            "individual_time": None,
            "field_size": len(race_data),
            "track_condition": "Good",
            "weather": "Fine",
            "temperature": 20.0,
            "humidity": 50.0,
            "wind_speed": 0.0,
            "trainer_name": "Unknown",
        }

        for field, default_value in required_fields.items():
            if field not in race_data.columns:
                race_data[field] = default_value

        # Build features
        race_features = ml_system.temporal_builder.build_features_for_race(
            race_data, race_id
        )

        # Prepare features for prediction
        X_pred = race_features.drop(
            ["race_id", "dog_clean_name", "target", "target_timestamp"],
            axis=1,
            errors="ignore",
        )

        # Ensure proper feature alignment
        X_pred = X_pred.reindex(columns=ml_system.feature_columns, fill_value=0)

        # Handle categorical/numerical features
        categorical_features = [
            "venue",
            "grade",
            "track_condition",
            "weather",
            "trainer_name",
        ]
        for cat_col in categorical_features:
            if cat_col in X_pred.columns:
                default_values = {
                    "venue": "UNKNOWN",
                    "grade": "5",
                    "track_condition": "Good",
                    "weather": "Fine",
                    "trainer_name": "Unknown",
                }
                X_pred[cat_col] = X_pred[cat_col].apply(
                    lambda x: (
                        default_values.get(cat_col, "Unknown")
                        if (pd.isna(x) or x == 0 or x == "0")
                        else str(x)
                    )
                )

        # Convert numerical columns
        numerical_features = [
            col for col in X_pred.columns if col not in categorical_features
        ]
        for col in numerical_features:
            if col in X_pred.columns:
                X_pred[col] = pd.to_numeric(X_pred[col], errors="coerce").fillna(0.0)

        # Get raw probabilities
        raw_probs = ml_system.calibrated_pipeline.predict_proba(X_pred)[:, 1]

        print(f"\n📊 RAW PROBABILITIES ANALYSIS:")
        print(f"   Raw probabilities: {raw_probs}")
        print(f"   Min: {raw_probs.min():.6f}")
        print(f"   Max: {raw_probs.max():.6f}")
        print(f"   Range: {raw_probs.max() - raw_probs.min():.6f}")
        print(f"   Std Dev: {raw_probs.std():.6f}")
        print(f"   Mean: {raw_probs.mean():.6f}")

        # Apply current normalization
        current_normalized = ml_system._group_normalize_probabilities(raw_probs)

        print(f"\n📊 CURRENT NORMALIZATION (Softmax):")
        print(f"   Normalized: {current_normalized}")
        print(f"   Min: {current_normalized.min():.6f}")
        print(f"   Max: {current_normalized.max():.6f}")
        print(f"   Range: {current_normalized.max() - current_normalized.min():.6f}")
        print(f"   Std Dev: {current_normalized.std():.6f}")

        # Try alternative normalization methods
        print(f"\n🔧 ALTERNATIVE NORMALIZATION METHODS:")

        # Method 1: Simple normalization (just divide by sum)
        simple_normalized = raw_probs / raw_probs.sum()
        print(f"\n1. Simple Normalization:")
        print(f"   Range: {simple_normalized.max() - simple_normalized.min():.6f}")
        print(f"   Std Dev: {simple_normalized.std():.6f}")

        # Method 2: Enhanced softmax with temperature scaling
        temperatures = [0.5, 1.0, 2.0, 5.0]
        for temp in temperatures:
            temp_probs = np.exp((raw_probs - np.max(raw_probs)) / temp)
            temp_normalized = temp_probs / temp_probs.sum()
            print(f"\n2. Temperature Softmax (T={temp}):")
            print(f"   Range: {temp_normalized.max() - temp_normalized.min():.6f}")
            print(f"   Std Dev: {temp_normalized.std():.6f}")

        # Method 3: Rank-based normalization
        ranks = np.argsort(np.argsort(-raw_probs)) + 1  # Higher rank = better
        rank_weights = 1.0 / ranks  # Inverse rank weighting
        rank_normalized = rank_weights / rank_weights.sum()
        print(f"\n3. Rank-based Normalization:")
        print(f"   Range: {rank_normalized.max() - rank_normalized.min():.6f}")
        print(f"   Std Dev: {rank_normalized.std():.6f}")

        # Method 4: Power transformation
        powers = [2.0, 3.0, 0.5]
        for power in powers:
            power_probs = np.power(raw_probs, power)
            power_normalized = power_probs / power_probs.sum()
            print(f"\n4. Power Transformation (p={power}):")
            print(f"   Range: {power_normalized.max() - power_normalized.min():.6f}")
            print(f"   Std Dev: {power_normalized.std():.6f}")

        # Feature analysis
        print(f"\n🔍 FEATURE ANALYSIS:")
        print(f"   Feature shape: {X_pred.shape}")
        print(f"   Non-zero features per dog:")
        for i, dog_name in enumerate(race_features["dog_clean_name"]):
            non_zero = np.sum(X_pred.iloc[i] != 0)
            print(
                f"     {dog_name}: {non_zero}/{len(X_pred.columns)} ({non_zero/len(X_pred.columns)*100:.1f}%)"
            )

        # Feature variance analysis
        feature_vars = X_pred.var()
        print(f"\n   Top 10 most variant features:")
        top_variant = feature_vars.nlargest(10)
        for feature, variance in top_variant.items():
            print(f"     {feature}: {variance:.6f}")

        print(f"\n   Features with zero variance: {sum(feature_vars == 0)}")

        return True

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = diagnose_variance()
    exit(0 if success else 1)
