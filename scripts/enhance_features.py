#!/usr/bin/env python3
"""
Advanced Feature Enhancement Script
===================================

Adds high-value engineered features to boost prediction accuracy:
- Interaction features between key variables
- Time-based rolling statistics
- Class transition metrics
- Pace analysis features
- Seasonal patterns
- Competitive advantage metrics
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")


def enhance_temporal_features(df):
    """Add advanced temporal features."""
    enhanced = df.copy()

    # Rolling performance windows
    for window in [3, 5, 10]:
        enhanced[f"win_rate_last_{window}"] = (
            enhanced.groupby("dog_clean_name")["target"]
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(enhanced["win_rate"])
        )

        enhanced[f"avg_position_last_{window}"] = (
            enhanced.groupby("dog_clean_name")["finish_position"]
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(enhanced["avg_position"])
        )

    # Performance momentum
    enhanced["performance_momentum"] = (
        enhanced["win_rate_last_3"] - enhanced["win_rate_last_10"]
    )

    # Days since best performance
    enhanced["days_since_best_time"] = (
        enhanced.groupby("dog_clean_name")
        .apply(
            lambda x: (
                pd.to_datetime("today")
                - pd.to_datetime(
                    x[x["individual_time"] == x["individual_time"].min()][
                        "race_date"
                    ].iloc[0]
                    if len(x[x["individual_time"] == x["individual_time"].min()]) > 0
                    else x["race_date"].iloc[0]
                )
            ).days
        )
        .fillna(30)
    )

    return enhanced


def enhance_interaction_features(df):
    """Create interaction features between key variables."""
    enhanced = df.copy()

    # Weight-Distance interactions
    enhanced["weight_distance_ratio"] = (
        enhanced["current_weight"] / enhanced["distance"]
    )

    # Box-Venue interactions
    enhanced["box_venue_advantage"] = (
        enhanced["box_win_rate"] * enhanced["venue_experience"]
    )

    # Form-Class interactions
    enhanced["form_grade_fit"] = (
        enhanced["recent_form_avg"] * enhanced["grade_experience"]
    )

    # Weather-Performance interactions
    enhanced["weather_performance_adj"] = (
        enhanced["weather_performance"] * enhanced["weather_adjustment_factor"]
    )

    # Track condition adaptation
    enhanced["track_adaptation_score"] = (
        enhanced["traditional_track_condition_score"] * enhanced["venue_experience"]
    )

    return enhanced


def enhance_competitive_features(df):
    """Add competitive advantage features."""
    enhanced = df.copy()

    # Field strength indicators
    race_stats = (
        enhanced.groupby("race_id")
        .agg(
            {
                "win_rate": ["mean", "std", "max"],
                "avg_position": ["mean", "std", "min"],
                "traditional_overall_score": ["mean", "std", "max"],
            }
        )
        .round(4)
    )

    # Flatten column names
    race_stats.columns = ["_".join(col).strip() for col in race_stats.columns]

    # Merge back
    enhanced = enhanced.merge(
        race_stats, left_on="race_id", right_index=True, how="left"
    )

    # Competitive advantage metrics
    enhanced["win_rate_vs_field"] = enhanced["win_rate"] - enhanced["win_rate_mean"]
    enhanced["position_vs_field"] = (
        enhanced["avg_position"] - enhanced["avg_position_mean"]
    )
    enhanced["score_vs_field"] = (
        enhanced["traditional_overall_score"]
        - enhanced["traditional_overall_score_mean"]
    )

    # Field strength quartile
    enhanced["field_strength_quartile"] = pd.qcut(
        enhanced["win_rate_mean"],
        q=4,
        labels=["weak", "below_avg", "above_avg", "strong"],
    ).astype(str)

    return enhanced


def enhance_pace_features(df):
    """Add pace and sectional analysis features."""
    enhanced = df.copy()

    # Pace categories based on best times
    distance_pace = enhanced.groupby("distance")["best_time"].quantile([0.33, 0.67])

    def categorize_pace(row):
        if pd.isna(row["best_time"]) or pd.isna(row["distance"]):
            return "unknown"

        distance = row["distance"]
        time = row["best_time"]

        try:
            q33 = distance_pace[distance][0.33]
            q67 = distance_pace[distance][0.67]

            if time <= q33:
                return "fast"
            elif time <= q67:
                return "moderate"
            else:
                return "slow"
        except (KeyError, IndexError):
            return "unknown"

    enhanced["pace_category"] = enhanced.apply(categorize_pace, axis=1)

    # Early speed indicators
    enhanced["early_pace_score"] = np.where(
        enhanced["pace_category"] == "fast",
        enhanced["traditional_overall_score"] * 1.1,
        enhanced["traditional_overall_score"] * 0.95,
    )

    return enhanced


def enhance_class_transition_features(df):
    """Add class/grade transition analysis."""
    enhanced = df.copy()

    # Grade movement tracking
    enhanced = enhanced.sort_values(["dog_clean_name", "race_date"])
    enhanced["previous_grade"] = enhanced.groupby("dog_clean_name")["grade"].shift(1)
    enhanced["grade_change"] = pd.to_numeric(
        enhanced["grade"], errors="coerce"
    ) - pd.to_numeric(enhanced["previous_grade"], errors="coerce")

    # Class transition performance
    enhanced["moving_up_class"] = (enhanced["grade_change"] > 0).astype(int)
    enhanced["moving_down_class"] = (enhanced["grade_change"] < 0).astype(int)
    enhanced["staying_same_class"] = (enhanced["grade_change"] == 0).astype(int)

    # Success rate in class transitions
    class_success = enhanced.groupby(["dog_clean_name", "moving_up_class"])[
        "target"
    ].mean()
    enhanced["class_transition_success_rate"] = enhanced.apply(
        lambda row: class_success.get(
            (row["dog_clean_name"], row["moving_up_class"]), 0.1
        ),
        axis=1,
    )

    return enhanced


def enhance_seasonal_features(df):
    """Add seasonal and temporal pattern features."""
    enhanced = df.copy()

    # Convert race_date to datetime if needed
    enhanced["race_date"] = pd.to_datetime(enhanced["race_date"])

    # Seasonal features
    enhanced["month"] = enhanced["race_date"].dt.month
    enhanced["quarter"] = enhanced["race_date"].dt.quarter
    enhanced["day_of_week"] = enhanced["race_date"].dt.dayofweek

    # Performance by time periods
    monthly_performance = enhanced.groupby(["dog_clean_name", "month"])["target"].mean()
    enhanced["monthly_performance"] = enhanced.apply(
        lambda row: monthly_performance.get((row["dog_clean_name"], row["month"]), 0.1),
        axis=1,
    )

    # Weekend vs weekday performance
    enhanced["is_weekend"] = (enhanced["day_of_week"] >= 5).astype(int)
    weekend_performance = enhanced.groupby(["dog_clean_name", "is_weekend"])[
        "target"
    ].mean()
    enhanced["weekend_performance"] = enhanced.apply(
        lambda row: weekend_performance.get(
            (row["dog_clean_name"], row["is_weekend"]), 0.1
        ),
        axis=1,
    )

    return enhanced


def enhance_trainer_features(df):
    """Add advanced trainer performance features."""
    enhanced = df.copy()

    if "trainer_name" in enhanced.columns:
        # Trainer success metrics
        trainer_stats = (
            enhanced.groupby("trainer_name")
            .agg(
                {
                    "target": ["mean", "count"],
                    "traditional_overall_score": "mean",
                    "avg_position": "mean",
                }
            )
            .round(4)
        )

        trainer_stats.columns = [
            "trainer_win_rate",
            "trainer_race_count",
            "trainer_avg_score",
            "trainer_avg_position",
        ]

        enhanced = enhanced.merge(
            trainer_stats, left_on="trainer_name", right_index=True, how="left"
        )

        # Trainer specialization scores
        trainer_distance_stats = enhanced.groupby(["trainer_name", "distance"])[
            "target"
        ].mean()
        enhanced["trainer_distance_specialization"] = enhanced.apply(
            lambda row: trainer_distance_stats.get(
                (row["trainer_name"], row["distance"]), 0.1
            ),
            axis=1,
        )

    return enhanced


def create_enhanced_feature_dataset(input_data):
    """Apply all feature enhancements to the dataset."""
    print("🚀 Starting advanced feature enhancement...")

    enhanced = input_data.copy()
    original_features = len(enhanced.columns)

    print("  ⏱️ Adding temporal features...")
    enhanced = enhance_temporal_features(enhanced)

    print("  🔗 Adding interaction features...")
    enhanced = enhance_interaction_features(enhanced)

    print("  🏆 Adding competitive features...")
    enhanced = enhance_competitive_features(enhanced)

    print("  🏃 Adding pace features...")
    enhanced = enhance_pace_features(enhanced)

    print("  📈 Adding class transition features...")
    enhanced = enhance_class_transition_features(enhanced)

    print("  📅 Adding seasonal features...")
    enhanced = enhance_seasonal_features(enhanced)

    print("  👨‍🏫 Adding trainer features...")
    enhanced = enhance_trainer_features(enhanced)

    new_features = len(enhanced.columns) - original_features
    print(f"✅ Enhancement complete: Added {new_features} new features")
    print(f"📊 Total features: {len(enhanced.columns)} (was {original_features})")

    # Feature summary
    print("\n📋 NEW FEATURES ADDED:")
    print("  • Temporal: Rolling windows (3,5,10), momentum, days since best")
    print("  • Interactions: Weight-distance, box-venue, form-grade combinations")
    print("  • Competitive: Field strength, relative performance metrics")
    print("  • Pace: Speed categories, early pace scores")
    print("  • Class: Grade transitions, success rates")
    print("  • Seasonal: Month, quarter, weekend patterns")
    print("  • Trainer: Specialization scores, distance preferences")

    return enhanced


def main():
    """Main enhancement pipeline."""
    print("🔧 ADVANCED FEATURE ENHANCEMENT")
    print("=" * 50)

    # This would typically load from your data pipeline
    # For now, create a demonstration with the structure

    print("📊 Feature enhancement recommendations:")
    print()
    print("✅ IMMEDIATELY IMPLEMENTABLE (High Impact):")
    print("  1. Rolling Performance Windows (3, 5, 10 races)")
    print("  2. Weight-Distance Ratio interactions")
    print("  3. Field Strength relative metrics")
    print("  4. Class transition success tracking")
    print("  5. Pace category analysis")
    print()
    print("🎯 MODERATE IMPLEMENTATION (Medium Impact):")
    print("  6. Seasonal performance patterns")
    print("  7. Trainer specialization metrics")
    print("  8. Weather-performance interactions")
    print("  9. Box-venue advantage combinations")
    print("  10. Competition strength quartiles")
    print()
    print("🔬 ADVANCED FEATURES (Requires More Data):")
    print("  11. Sectional time analysis")
    print("  12. Race pace clustering")
    print("  13. Breeding/bloodline features")
    print("  14. Track bias adjustments")
    print("  15. Market efficiency indicators")

    print("\n💡 IMPLEMENTATION PRIORITY:")
    print("  Phase 1: Focus on rolling windows + interactions (#1, #2, #3)")
    print("  Phase 2: Add competitive metrics (#4, #5, #10)")
    print("  Phase 3: Temporal patterns (#6, #7, #8)")
    print("  Phase 4: Advanced analysis (#11-15)")


if __name__ == "__main__":
    main()
