#!/usr/bin/env python3


# Wrapper class for unified predictor compatibility
class AdvancedFeatureEngineer:
    """Advanced feature engineering v2 - wrapper for compatibility"""

    def __init__(self):
        pass

    def engineer_features(self, data):
        # Placeholder implementation
        return data


"""
Enhanced Feature Engineering for Greyhound Racing Predictions
============================================================

This module provides advanced feature engineering capabilities to improve
prediction accuracy through better data transformation and feature creation.

Author: AI Assistant
Date: July 23, 2025
"""

import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from features import (V3BoxPositionFeatures, V3CompetitionFeatures,
                      V3DistanceStatsFeatures, V3RecentFormFeatures,
                      V3TrainerFeatures, V3VenueAnalysisFeatures,
                      V3WeatherTrackFeatures)


class EnhancedFeatureEngineer:
    """Advanced feature engineering for greyhound racing predictions using versioned modules"""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path

    def create_advanced_features(self, dog_stats, race_context=None):
        """Create advanced features from basic dog statistics using versioned modules"""
        if not dog_stats:
            return self.get_default_features()

        features = {}

        # Use versioned feature classes
        distance_features = V3DistanceStatsFeatures().create_features(dog_stats)
        recent_form_features = V3RecentFormFeatures().create_features(dog_stats)
        venue_analysis_features = V3VenueAnalysisFeatures().create_features(dog_stats)
        box_position_features = V3BoxPositionFeatures().create_features(dog_stats)
        competition_features = V3CompetitionFeatures().create_features(dog_stats)
        weather_track_features = V3WeatherTrackFeatures().create_features(dog_stats)
        trainer_features = V3TrainerFeatures().create_features(dog_stats)

        # Aggregate all features
        features.update(distance_features)
        features.update(recent_form_features)
        features.update(venue_analysis_features)
        features.update(box_position_features)
        features.update(competition_features)
        features.update(weather_track_features)
        features.update(trainer_features)

        # Composite features
        features.update(self._create_composite_features(features, dog_stats))

        return features

    def _create_performance_features(self, dog_stats):
        """Create enhanced performance-based features"""
        features = {}

        # Win rate features with decay
        recent_form = dog_stats.get("recent_form", [])
        if recent_form:
            # Weighted recent performance (more recent races weighted higher)
            weights = np.exp(-0.1 * np.arange(len(recent_form)))  # Exponential decay
            
            # Ensure recent_form is a 1D array and weights match in length
            recent_form = np.asarray(recent_form).flatten()
            if len(weights) != len(recent_form):
                weights = weights[:len(recent_form)]  # Truncate weights if needed
            
            try:
                weighted_avg_position = np.average(recent_form, weights=weights, axis=0)
            except (ValueError, ZeroDivisionError):
                # Fallback to simple average if weighted average fails
                weighted_avg_position = np.mean(recent_form) if len(recent_form) > 0 else 4.0
            features["weighted_recent_position"] = weighted_avg_position

            # Performance trend (linear regression on recent form)
            if len(recent_form) >= 3:
                x = np.arange(len(recent_form))
                slope = np.polyfit(x, recent_form, 1)[0]
                features["form_trend_slope"] = -slope  # Negative slope = improving
            else:
                features["form_trend_slope"] = 0
        else:
            features["weighted_recent_position"] = 4.0
            features["form_trend_slope"] = 0

        # Strike rate features
        features["win_strike_rate"] = dog_stats.get("win_rate", 0)
        features["place_strike_rate"] = dog_stats.get("place_rate", 0)
        features["top_half_strike_rate"] = dog_stats.get("top_half_rate", 0)

        # Position quality score
        avg_pos = dog_stats.get("avg_position", 4.0)
        features["position_quality_score"] = max(0, (8 - avg_pos) / 8)

        # Finishing ability (how often finishes races vs DNF)
        races_count = dog_stats.get("races_count", 1)
        features["completion_rate"] = min(
            1.0, races_count / 10
        )  # Proxy for reliability

        return features

    def _create_temporal_features(self, dog_stats):
        """Create time-based features"""
        features = {}

        # Recent activity features
        activity = dog_stats.get("recent_activity", {})
        if isinstance(activity, dict):
            days_since_last = activity.get("days_since_last_race", 30)
            recent_frequency = activity.get("recent_frequency", 1)

            # Optimal racing frequency (peak performance around 2-3 weeks)
            optimal_gap = 21  # days
            gap_penalty = abs(days_since_last - optimal_gap) / optimal_gap
            features["optimal_gap_score"] = max(0, 1 - gap_penalty)

            # Racing rhythm (consistent vs irregular racing)
            features["racing_frequency"] = min(2.0, recent_frequency)
            features["activity_consistency"] = activity.get("activity_score", 0.5)
        else:
            features["optimal_gap_score"] = 0.5
            features["racing_frequency"] = 1.0
            features["activity_consistency"] = 0.5

        # Experience level
        races_count = dog_stats.get("races_count", 0)
        features["experience_level"] = min(1.0, races_count / 20)

        return features

    def _create_consistency_features(self, dog_stats):
        """Create consistency and reliability features"""
        features = {}

        # Time consistency
        features["time_consistency"] = dog_stats.get("time_consistency", 0.5)

        # Position consistency
        features["position_consistency"] = dog_stats.get("position_consistency", 0.5)

        # Performance variance (lower = more consistent)
        recent_form = dog_stats.get("recent_form", [])
        if len(recent_form) > 1:
            features["performance_variance"] = 1 / (np.std(recent_form) + 1)
        else:
            features["performance_variance"] = 0.5

        # Reliability score (combination of consistency metrics)
        features["reliability_score"] = np.mean(
            [
                features["time_consistency"],
                features["position_consistency"],
                features["performance_variance"],
            ]
        )

        return features

    def _create_speed_class_features(self, dog_stats):
        """Create speed and class-based features"""
        features = {}

        # Speed ratings
        features["speed_index"] = dog_stats.get("speed_index", 50) / 100
        features["avg_speed_rating"] = dog_stats.get("avg_speed_rating", 50) / 100
        features["best_time_rating"] = (
            min(1.0, (40 - dog_stats.get("best_time", 35)) / 10)
            if dog_stats.get("best_time", 0) > 0
            else 0.5
        )

        # Class assessment
        features["class_rating"] = dog_stats.get("avg_class_rating", 50) / 100
        features["class_assessment"] = dog_stats.get("class_assessment", 50) / 100

        # Performance ratings
        features["performance_rating"] = (
            dog_stats.get("avg_performance_rating", 50) / 100
        )

        # Speed-class combination
        features["speed_class_composite"] = (
            features["speed_index"] + features["class_rating"]
        ) / 2

        return features

    def _create_environmental_features(self, dog_stats):
        """Create environmental adaptation features"""
        features = {}

        # Track versatility
        features["track_versatility"] = min(
            1.0, dog_stats.get("track_versatility", 1) / 5
        )

        # Venue diversity
        features["venue_experience"] = min(1.0, dog_stats.get("venue_diversity", 1) / 5)

        # Track condition performance
        track_performance = dog_stats.get("track_condition_performance", {})
        if track_performance:
            # Average performance across different conditions
            avg_positions = [
                data.get("avg_position", 4)
                for data in track_performance.values()
                if isinstance(data, dict)
            ]
            if avg_positions:
                features["track_adaptability"] = max(
                    0, (8 - np.mean(avg_positions)) / 8
                )
            else:
                features["track_adaptability"] = 0.5
        else:
            features["track_adaptability"] = 0.5

        # Distance suitability
        features["distance_specialization"] = (
            0.5  # Default, could be enhanced with distance analysis
        )

        return features

    def _create_contextual_features(self, dog_stats, race_context):
        """Create race context-specific features"""
        features = {}

        # Field size adjustment
        field_size = race_context.get("field_size", 8)
        features["field_size_normalized"] = (
            field_size / 12
        )  # Normalize to typical range

        # Distance context
        distance = int(race_context.get("distance", "500").replace("m", ""))
        features["race_distance"] = distance / 1000  # Normalize to km

        # Track condition match
        current_condition = race_context.get("track_condition", "Good")
        track_performance = dog_stats.get("track_condition_performance", {})

        if current_condition in track_performance:
            condition_data = track_performance[current_condition]
            if isinstance(condition_data, dict) and condition_data.get("races", 0) >= 2:
                features["condition_match_score"] = max(
                    0, (8 - condition_data.get("avg_position", 4)) / 8
                )
            else:
                features["condition_match_score"] = 0.4  # Slight penalty for no data
        else:
            features["condition_match_score"] = 0.3  # Penalty for unknown condition

        # Weather factors (if available)
        if "temperature" in race_context:
            temp = race_context["temperature"]
            # Optimal temperature range (15-25°C)
            temp_score = 1 - abs(temp - 20) / 20 if temp else 0.5
            features["temperature_suitability"] = max(0, min(1, temp_score))
        else:
            features["temperature_suitability"] = 0.5

        return features

    def _create_composite_features(self, features, dog_stats):
        """Create composite features combining multiple factors"""
        composite_features = {}

        # Overall fitness score
        composite_features["fitness_score"] = np.mean(
            [
                features.get("optimal_gap_score", 0.5),
                features.get("activity_consistency", 0.5),
                features.get("reliability_score", 0.5),
            ]
        )

        # Competitive ability
        composite_features["competitive_ability"] = np.mean(
            [
                features.get("win_strike_rate", 0),
                features.get("position_quality_score", 0.5),
                features.get("speed_class_composite", 0.5),
            ]
        )

        # Adaptability score
        composite_features["adaptability"] = np.mean(
            [
                features.get("track_adaptability", 0.5),
                features.get("venue_experience", 0.5),
                features.get("track_versatility", 0.5),
            ]
        )

        # Form momentum
        composite_features["form_momentum"] = np.mean(
            [
                features.get("form_trend_slope", 0) + 0.5,  # Center around 0.5
                1
                - features.get("weighted_recent_position", 4)
                / 8,  # Convert position to score
                features.get("performance_rating", 0.5),
            ]
        )

        # Contextual advantage (how well suited for this specific race)
        composite_features["contextual_advantage"] = np.mean(
            [
                features.get("condition_match_score", 0.5),
                features.get("temperature_suitability", 0.5),
                features.get("distance_specialization", 0.5),
            ]
        )

        return composite_features

    def get_default_features(self):
        """Return default features for dogs with no historical data"""
        return {
            "weighted_recent_position": 4.0,
            "form_trend_slope": 0,
            "win_strike_rate": 0.1,
            "place_strike_rate": 0.3,
            "top_half_strike_rate": 0.5,
            "position_quality_score": 0.3,
            "completion_rate": 0.5,
            "optimal_gap_score": 0.5,
            "racing_frequency": 1.0,
            "activity_consistency": 0.5,
            "experience_level": 0.1,
            "time_consistency": 0.5,
            "position_consistency": 0.5,
            "performance_variance": 0.5,
            "reliability_score": 0.5,
            "speed_index": 0.5,
            "avg_speed_rating": 0.5,
            "best_time_rating": 0.5,
            "class_rating": 0.5,
            "class_assessment": 0.5,
            "performance_rating": 0.5,
            "speed_class_composite": 0.5,
            "track_versatility": 0.2,
            "venue_experience": 0.2,
            "track_adaptability": 0.5,
            "distance_specialization": 0.5,
            "field_size_normalized": 0.67,
            "race_distance": 0.5,
            "condition_match_score": 0.3,
            "temperature_suitability": 0.5,
            "fitness_score": 0.5,
            "competitive_ability": 0.3,
            "adaptability": 0.4,
            "form_momentum": 0.4,
            "contextual_advantage": 0.4,
        }

    def get_feature_importance_ranking(self):
        """Return feature importance ranking for model interpretation"""
        return {
            "high_importance": [
                "competitive_ability",
                "form_momentum",
                "win_strike_rate",
                "position_quality_score",
                "weighted_recent_position",
            ],
            "medium_importance": [
                "fitness_score",
                "contextual_advantage",
                "place_strike_rate",
                "reliability_score",
                "speed_class_composite",
            ],
            "low_importance": [
                "adaptability",
                "field_size_normalized",
                "race_distance",
                "temperature_suitability",
                "venue_experience",
            ],
        }

    def validate_features(self, features):
        """Validate and clean feature values"""
        validated = {}

        for key, value in features.items():
            # Ensure all values are numeric and within reasonable bounds
            if (
                isinstance(value, (int, float))
                and not np.isnan(value)
                and not np.isinf(value)
            ):
                # Clip values to reasonable ranges
                validated[key] = max(0, min(2, float(value)))
            else:
                # Use default value for invalid entries
                validated[key] = 0.5

        return validated
