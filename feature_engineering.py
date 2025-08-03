#!/usr/bin/env python3
"""
Feature Engineering for Greyhound Racing Predictions
===================================================

This module provides comprehensive feature engineering to match the expected
66 features from the trained comprehensive models.

Author: AI Assistant
Date: July 31, 2025
"""

import logging
import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Comprehensive feature engineering for greyhound racing predictions"""

    # Expected feature list (66 features total)
    EXPECTED_FEATURES = [
        "avg_position",
        "recent_form_avg",
        "market_confidence",
        "current_odds_log",
        "venue_experience",
        "place_rate",
        "current_weight",
        "time_consistency",
        "traditional_overall_score",
        "traditional_performance_score",
        "traditional_form_score",
        "traditional_consistency_score",
        "traditional_confidence_level",
        "win_rate",
        "long_term_form_trend",
        "position_consistency",
        "avg_time",
        "best_time",
        "time_improvement_trend",
        "avg_weight",
        "weight_consistency",
        "weight_vs_avg",
        "distance_specialization",
        "grade_experience",
        "fitness_score",
        "traditional_class_score",
        "traditional_fitness_score",
        "traditional_experience_score",
        "traditional_trainer_score",
        "traditional_track_condition_score",
        "traditional_distance_score",
        "traditional_key_factors_count",
        "traditional_risk_factors_count",
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "weather_adjustment_factor",
        "weather_clear",
        "weather_cloudy",
        "weather_rain",
        "weather_fog",
        "temp_cold",
        "temp_cool",
        "temp_optimal",
        "temp_warm",
        "temp_hot",
        "wind_calm",
        "wind_light",
        "wind_moderate",
        "wind_strong",
        "humidity_low",
        "humidity_normal",
        "humidity_high",
        "weather_experience_count",
        "weather_performance",
        "days_since_last",
        "competition_strength",
        "box_win_rate",
        "current_box",
        "field_size",
        "historical_races_count",
        "venue_encoded",
        "track_condition_encoded",
        "grade_encoded",
        "distance_numeric",
    ]

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        logger.info(
            "FeatureEngineer initialized with comprehensive feature set (66 features)"
        )

    def create_features_for_dogs(self, dog_data_list, race_context=None):
        """
        Create features for multiple dogs in a race with proper 4-12 runner constraint handling

        Args:
            dog_data_list: List of dog data dictionaries
            race_context: Race context information

        Returns:
            Padded/truncated feature array of shape (n_dogs, 66) where 4 <= n_dogs <= 12
        """
        if not dog_data_list:
            logger.warning("Empty dog data list provided")
            return self._get_default_race_features(8)  # Default to 8 runners

        # Ensure we have between 4 and 12 runners
        n_dogs = len(dog_data_list)
        if n_dogs < 4:
            logger.warning(f"Only {n_dogs} dogs provided, padding to 4 runners minimum")
            # Pad with duplicate of last dog
            while len(dog_data_list) < 4:
                dog_data_list.append(dog_data_list[-1].copy())
        elif n_dogs > 12:
            logger.warning(f"{n_dogs} dogs provided, truncating to 12 runners maximum")
            dog_data_list = dog_data_list[:12]

        # Generate features for each dog
        features_list = []
        for i, dog_data in enumerate(dog_data_list):
            try:
                features = self.create_features_for_dog(
                    dog_data, race_context, position_in_field=i + 1
                )
                features_list.append(features)
            except Exception as e:
                logger.error(
                    f"Error creating features for dog {dog_data.get('name', 'unknown')}: {e}"
                )
                features_list.append(self._get_default_features())

        # Convert to DataFrame and ensure all expected features are present
        features_df = pd.DataFrame(features_list)
        features_df = self._ensure_all_features(features_df)

        logger.info(
            f"Created features for {len(features_df)} dogs with {len(features_df.columns)} features each"
        )
        return features_df

    def create_features_for_dog(self, dog_data, race_context=None, position_in_field=1):
        """Create comprehensive features for a single dog"""
        features = {}

        # Basic performance features
        features.update(self._create_performance_features(dog_data))

        # Traditional analysis features
        features.update(self._create_traditional_features(dog_data))

        # Weather and environmental features
        features.update(self._create_weather_features(dog_data, race_context))

        # Race context features
        features.update(
            self._create_race_context_features(
                dog_data, race_context, position_in_field
            )
        )

        # Competition features
        features.update(self._create_competition_features(dog_data, race_context))

        # Form and fitness features
        features.update(self._create_form_fitness_features(dog_data))

        # Ensure all expected features are present with defaults
        for expected_feature in self.EXPECTED_FEATURES:
            if expected_feature not in features:
                features[expected_feature] = self._get_default_value(expected_feature)

        return features

    def _create_performance_features(self, dog_data):
        """Create core performance-based features with exponential decay weighting"""
        features = {}

        # Basic performance metrics
        features["avg_position"] = float(dog_data.get("avg_position", 4.0))
        features["win_rate"] = float(dog_data.get("win_rate", 0.1))
        features["place_rate"] = float(dog_data.get("place_rate", 0.3))

# Recent form with exponential decay weighting
        recent_form = dog_data.get("recent_form", [4, 4, 4])
        if isinstance(recent_form, list) and len(recent_form) > 0:
            # Apply exponential decay (λ = 0.95) where most recent races have highest weight
            decay_weights = np.exp(-0.05 * np.arange(len(recent_form)))
            features["weighted_recent_form"] = np.average(recent_form, weights=decay_weights)
            features["speed_trend"] = self._calculate_speed_trend(recent_form)
            features["competitiveness_score"] = self._calculate_competitiveness(dog_data)
            features["distance_win_rate"] = self._calculate_distance_win_rate(dog_data)
            features["box_position_win_rate"] = self._calculate_box_position_win_rate(dog_data)
            decay_weights = np.power(0.95, np.arange(len(recent_form)))
            features["recent_form_avg"] = float(np.average(recent_form, weights=decay_weights))
        else:
            features["recent_form_avg"] = 4.0

        # Market confidence
        starting_price = dog_data.get("starting_price", 10.0)
        if isinstance(starting_price, str):
            try:
                starting_price = float(starting_price.replace("$", "").replace(",", ""))
            except:
                starting_price = 10.0
        features["market_confidence"] = max(
            0.1, min(1.0, 20.0 / max(1.0, starting_price))
        )
        features["current_odds_log"] = float(np.log(max(1.1, starting_price)))

        # Time-based performance with decay-weighted averages
        time_history = dog_data.get("time_history", [30.0])
        if isinstance(time_history, list) and len(time_history) > 0:
            # Apply exponential decay to time history
            time_decay_weights = np.power(0.95, np.arange(len(time_history)))
            features["avg_time"] = float(np.average(time_history, weights=time_decay_weights))
            features["best_time"] = float(min(time_history))
        else:
            features["avg_time"] = float(dog_data.get("avg_time", 30.0))
            features["best_time"] = float(dog_data.get("best_time", 29.0))
        
        features["time_consistency"] = float(dog_data.get("time_consistency", 0.5))
        features["time_improvement_trend"] = float(
            dog_data.get("time_improvement_trend", 0.0)
        )

        # Weight features
        current_weight = dog_data.get("weight", 30.0)
        if isinstance(current_weight, str):
            try:
                current_weight = float(current_weight.replace("kg", ""))
            except:
                current_weight = 30.0
        features["current_weight"] = float(current_weight)
        features["avg_weight"] = float(dog_data.get("avg_weight", current_weight))
        features["weight_consistency"] = float(dog_data.get("weight_consistency", 0.8))
        features["weight_vs_avg"] = float(current_weight - features["avg_weight"])

        # Position consistency
        features["position_consistency"] = float(
            dog_data.get("position_consistency", 0.5)
        )
        features["long_term_form_trend"] = float(dog_data.get("form_trend", 0.0))

        return features

    def _calculate_speed_trend(self, recent_form):
        """Calculate speed trend from recent form data"""
        if len(recent_form) < 3:
            return 0.0
        
        # Linear regression on recent form to detect improvement/decline
        x = np.arange(len(recent_form))
        y = np.array(recent_form)
        
        # Fit line and return negative slope (improvement = negative slope in positions)
        try:
            slope = np.polyfit(x, y, 1)[0]
            return -slope  # Negative slope means improving positions
        except:
            return 0.0
    
    def _calculate_competitiveness(self, dog_data):
        """Calculate competitive level from historical data"""
        # Base competitiveness on win rate and place rate
        win_rate = dog_data.get('win_rate', 0.1)
        place_rate = dog_data.get('place_rate', 0.3)
        avg_position = dog_data.get('avg_position', 4.0)
        
        # Ensure realistic historical data
        win_rate = max(0, min(win_rate, 1))
        place_rate = max(0, min(place_rate, 1))
        avg_position = max(1, min(avg_position, 8))

        # Normalize position to 0-1 scale (1 = always first, 0 = always last)
        position_score = max(0, (8 - avg_position) / 7)
        
        # Combine metrics
        competitiveness = (win_rate * 0.4 + place_rate * 0.3 + position_score * 0.3)
        return min(1.0, competitiveness)

    def _calculate_speed_consistency(self, dog_data):
        """Calculate consistency of speed"""
        speeds = dog_data.get('time_history', [])
        if len(speeds) < 2:
            return 0.5  # default consistency if no data
        return 1 / (np.std(speeds) + 0.1)

    def _calculate_venue_avg_position(self, dog_data, venue):
        """Calculate average position at the given venue"""
        venue_history = [race for race in dog_data.get('history', []) if race.get('venue') == venue]
        if len(venue_history) == 0:
            return 4.0  # default average position
        positions = [race.get('position', 4) for race in venue_history]
        return np.mean(positions)

    def _calculate_recent_momentum(self, dog_data):
        """Calculate recent racing momentum"""
        recent_positions = dog_data.get('recent_form', [])
        if len(recent_positions) < 2:
            return 0.0
        return np.mean(recent_positions[:3]) - np.mean(recent_positions[-3:])
    
    def _calculate_distance_win_rate(self, dog_data):
        """Calculate win rate at specific distances"""
        # Ensure realistic historical data
        return max(0, min(dog_data.get('distance_win_rate', dog_data.get('win_rate', 0.1)), 1))
    
    def _calculate_box_position_win_rate(self, dog_data):
        """Calculate win rate from specific box positions"""
        # Ensure realistic historical data
        return max(0, min(dog_data.get('box_position_win_rate', dog_data.get('win_rate', 0.1)), 1))

    def _create_traditional_features(self, dog_data):
        """Create traditional analysis features"""
        features = {}

        # Core traditional scores
        features["traditional_overall_score"] = float(
            dog_data.get("traditional_overall_score", 0.35)
        )
        features["traditional_performance_score"] = float(
            dog_data.get("traditional_performance_score", 0.3)
        )
        features["traditional_form_score"] = float(
            dog_data.get("traditional_form_score", 0.3)
        )
        features["traditional_consistency_score"] = float(
            dog_data.get("traditional_consistency_score", 0.3)
        )
        features["traditional_confidence_level"] = float(
            dog_data.get("traditional_confidence_level", 0.2)
        )

        # Specialized traditional scores
        features["traditional_class_score"] = float(
            dog_data.get("traditional_class_score", 0.5)
        )
        features["traditional_fitness_score"] = float(
            dog_data.get("traditional_fitness_score", 0.5)
        )
        features["traditional_experience_score"] = float(
            dog_data.get("traditional_experience_score", 0.2)
        )
        features["traditional_trainer_score"] = float(
            dog_data.get("traditional_trainer_score", 0.5)
        )
        features["traditional_track_condition_score"] = float(
            dog_data.get("traditional_track_condition_score", 0.5)
        )
        features["traditional_distance_score"] = float(
            dog_data.get("traditional_distance_score", 0.5)
        )

        # Traditional factor counts
        features["traditional_key_factors_count"] = int(
            dog_data.get("traditional_key_factors_count", 0)
        )
        features["traditional_risk_factors_count"] = int(
            dog_data.get("traditional_risk_factors_count", 2)
        )

        return features

    def _create_weather_features(self, dog_data, race_context):
        """Create weather and environmental features"""
        features = {}

        if not race_context:
            race_context = {}

        # Raw weather values
        features["temperature"] = float(race_context.get("temperature", 20.0))
        features["humidity"] = float(race_context.get("humidity", 60.0))
        features["wind_speed"] = float(race_context.get("wind_speed", 10.0))
        features["pressure"] = float(race_context.get("pressure", 1013.0))

        # Weather adjustment factor
        temp = features["temperature"]
        humidity = features["humidity"]
        wind = features["wind_speed"]

        # Calculate weather impact (optimal conditions around 20°C, 50% humidity, light wind)
        temp_impact = 1.0 - abs(temp - 20) / 20
        humidity_impact = 1.0 - abs(humidity - 50) / 50
        wind_impact = 1.0 - min(wind / 30, 1.0)  # Higher wind = worse conditions

        features["weather_adjustment_factor"] = float(
            np.mean([temp_impact, humidity_impact, wind_impact])
        )

        # Weather condition dummies
        weather = race_context.get("weather", "clear").lower()
        features["weather_clear"] = (
            1.0 if "clear" in weather or "fine" in weather else 0.0
        )
        features["weather_cloudy"] = (
            1.0 if "cloud" in weather or "overcast" in weather else 0.0
        )
        features["weather_rain"] = (
            1.0 if "rain" in weather or "shower" in weather else 0.0
        )
        features["weather_fog"] = 1.0 if "fog" in weather or "mist" in weather else 0.0

        # Temperature categories
        features["temp_cold"] = 1.0 if temp < 10 else 0.0
        features["temp_cool"] = 1.0 if 10 <= temp < 18 else 0.0
        features["temp_optimal"] = 1.0 if 18 <= temp < 25 else 0.0
        features["temp_warm"] = 1.0 if 25 <= temp < 30 else 0.0
        features["temp_hot"] = 1.0 if temp >= 30 else 0.0

        # Wind categories
        features["wind_calm"] = 1.0 if wind < 5 else 0.0
        features["wind_light"] = 1.0 if 5 <= wind < 15 else 0.0
        features["wind_moderate"] = 1.0 if 15 <= wind < 25 else 0.0
        features["wind_strong"] = 1.0 if wind >= 25 else 0.0

        # Humidity categories
        features["humidity_low"] = 1.0 if humidity < 40 else 0.0
        features["humidity_normal"] = 1.0 if 40 <= humidity < 70 else 0.0
        features["humidity_high"] = 1.0 if humidity >= 70 else 0.0

        # Weather experience
        features["weather_experience_count"] = int(
            dog_data.get("weather_experience_count", 5)
        )
        features["weather_performance"] = float(
            dog_data.get("weather_performance", 0.5)
        )

        return features

    def _create_race_context_features(self, dog_data, race_context, position_in_field):
        """Create race context and positional features"""
        features = {}

        if not race_context:
            race_context = {}

        # Race timing
        features["days_since_last"] = int(dog_data.get("days_since_last_race", 21))

        # Box position
        box_number = dog_data.get("box_number", position_in_field)
        if isinstance(box_number, str):
            try:
                box_number = int(box_number)
            except:
                box_number = position_in_field
        features["current_box"] = int(box_number)
        features["box_win_rate"] = float(
            dog_data.get("box_win_rate", 0.125)
        )  # 1/8 default

        # Field characteristics
        features["field_size"] = int(race_context.get("field_size", 8))
        features["historical_races_count"] = int(dog_data.get("races_count", 10))

        # Encoded categorical features
        venue = race_context.get("venue", "Unknown")
        features["venue_encoded"] = float(
            hash(venue) % 100 / 100.0
        )  # Simple hash encoding

        track_condition = race_context.get("track_condition", "Good")
        features["track_condition_encoded"] = float(hash(track_condition) % 50 / 50.0)

        grade = race_context.get("grade", "Grade 5")
        grade_mapping = {
            "Group 1": 10,
            "Group 2": 9,
            "Group 3": 8,
            "Grade 1": 7,
            "Grade 2": 6,
            "Grade 3": 5,
            "Grade 4": 4,
            "Grade 5": 3,
            "Maiden": 2,
            "Novice": 1,
        }
        features["grade_encoded"] = float(grade_mapping.get(grade, 3))

        # Distance
        distance = race_context.get("distance", "500m")
        if isinstance(distance, str):
            try:
                distance_numeric = float(
                    distance.replace("m", "").replace("metres", "")
                )
            except:
                distance_numeric = 500.0
        else:
            distance_numeric = float(distance)
        features["distance_numeric"] = distance_numeric

        return features

    def _create_competition_features(self, dog_data, race_context):
        """Create competition strength features"""
        features = {}

        # Competition strength based on field quality
        field_ratings = race_context.get("field_ratings", []) if race_context else []
        if field_ratings:
            features["competition_strength"] = float(np.mean(field_ratings))
        else:
            # Estimate based on grade
            grade = race_context.get("grade", "Grade 5") if race_context else "Grade 5"
            grade_strength_map = {
                "Group 1": 0.9,
                "Group 2": 0.8,
                "Group 3": 0.7,
                "Grade 1": 0.6,
                "Grade 2": 0.5,
                "Grade 3": 0.4,
                "Grade 4": 0.3,
                "Grade 5": 0.2,
                "Maiden": 0.1,
                "Novice": 0.05,
            }
            features["competition_strength"] = float(grade_strength_map.get(grade, 0.2))

        return features

    def _create_form_fitness_features(self, dog_data):
        """Create form and fitness related features"""
        features = {}

        # Venue experience
        features["venue_experience"] = float(dog_data.get("venue_experience", 0.5))

        # Distance specialization
        features["distance_specialization"] = float(
            dog_data.get("distance_specialization", 0.5)
        )

        # Grade experience
        features["grade_experience"] = float(dog_data.get("grade_experience", 0.3))

        # Overall fitness score
        fitness_factors = [
            dog_data.get("time_consistency", 0.5),
            dog_data.get("weight_consistency", 0.8),
            dog_data.get("recent_activity_score", 0.5),
        ]
        features["fitness_score"] = float(
            np.mean([f for f in fitness_factors if f is not None])
        )

        return features

    def _get_default_value(self, feature_name):
        """Get default value for a specific feature"""
        defaults = {
            # Performance features
            "avg_position": 4.0,
            "recent_form_avg": 4.0,
            "market_confidence": 0.1,
            "current_odds_log": 2.3,
            "venue_experience": 0.5,
            "place_rate": 0.3,
            "current_weight": 30.0,
            "time_consistency": 0.5,
            "win_rate": 0.1,
            "long_term_form_trend": 0.0,
            "position_consistency": 0.5,
            "avg_time": 30.0,
            "best_time": 29.0,
            "time_improvement_trend": 0.0,
            "avg_weight": 30.0,
            "weight_consistency": 0.8,
            "weight_vs_avg": 0.0,
            "distance_specialization": 0.5,
            "grade_experience": 0.3,
            "fitness_score": 0.5,
            # Traditional features
            "traditional_overall_score": 0.35,
            "traditional_performance_score": 0.3,
            "traditional_form_score": 0.3,
            "traditional_consistency_score": 0.3,
            "traditional_confidence_level": 0.2,
            "traditional_class_score": 0.5,
            "traditional_fitness_score": 0.5,
            "traditional_experience_score": 0.2,
            "traditional_trainer_score": 0.5,
            "traditional_track_condition_score": 0.5,
            "traditional_distance_score": 0.5,
            "traditional_key_factors_count": 0,
            "traditional_risk_factors_count": 2,
            # Weather features
            "temperature": 20.0,
            "humidity": 60.0,
            "wind_speed": 10.0,
            "pressure": 1013.0,
            "weather_adjustment_factor": 0.7,
            "weather_clear": 1.0,
            "weather_cloudy": 0.0,
            "weather_rain": 0.0,
            "weather_fog": 0.0,
            "temp_cold": 0.0,
            "temp_cool": 0.0,
            "temp_optimal": 1.0,
            "temp_warm": 0.0,
            "temp_hot": 0.0,
            "wind_calm": 0.0,
            "wind_light": 1.0,
            "wind_moderate": 0.0,
            "wind_strong": 0.0,
            "humidity_low": 0.0,
            "humidity_normal": 1.0,
            "humidity_high": 0.0,
            "weather_experience_count": 5,
            "weather_performance": 0.5,
            # Context features
            "days_since_last": 21,
            "competition_strength": 0.2,
            "box_win_rate": 0.125,
            "current_box": 4,
            "field_size": 8,
            "historical_races_count": 10,
            "venue_encoded": 0.5,
            "track_condition_encoded": 0.5,
            "grade_encoded": 3.0,
            "distance_numeric": 500.0,
        }
        return defaults.get(feature_name, 0.5)

    def _get_default_features(self):
        """Get complete default feature set"""
        return {
            feature: self._get_default_value(feature)
            for feature in self.EXPECTED_FEATURES
        }

    def _get_default_race_features(self, n_dogs):
        """Get default features for an entire race"""
        race_features = []
        for i in range(n_dogs):
            features = self._get_default_features()
            features["current_box"] = i + 1
            race_features.append(features)
        return pd.DataFrame(race_features)

    def _ensure_all_features(self, features_df):
        """Ensure DataFrame has all expected features in correct order"""
        # Add missing features with defaults
        for feature in self.EXPECTED_FEATURES:
            if feature not in features_df.columns:
                features_df[feature] = self._get_default_value(feature)
                logger.debug(f"Added missing feature '{feature}' with default value")

        # Ensure correct order
        features_df = features_df.reindex(
            columns=self.EXPECTED_FEATURES, fill_value=0.5
        )

        # Validate data types and ranges
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(
                self._get_default_value(col)
            )
            # Clip extreme values
            if (
                not col.endswith("_count")
                and not col.endswith("_encoded")
                and not col.endswith("_numeric")
            ):
                features_df[col] = features_df[col].clip(0, 10)

        return features_df

    def validate_features(self, features_df):
        """Validate feature DataFrame for model compatibility"""
        validation_results = {"valid": True, "warnings": [], "errors": []}

        # Check shape
        if features_df.shape[1] != 66:
            validation_results["errors"].append(
                f"Expected 66 features, got {features_df.shape[1]}"
            )
            validation_results["valid"] = False

        # Check runner count constraint
        n_runners = len(features_df)
        if n_runners < 4 or n_runners > 12:
            validation_results["warnings"].append(
                f"Runner count {n_runners} outside optimal range (4-12)"
            )

        # Check for missing or infinite values
        if features_df.isnull().any().any():
            validation_results["errors"].append(
                "Missing values detected in feature set"
            )
            validation_results["valid"] = False

        if np.isinf(features_df.values).any():
            validation_results["errors"].append(
                "Infinite values detected in feature set"
            )
            validation_results["valid"] = False

        # Check feature names match exactly
        missing_features = set(self.EXPECTED_FEATURES) - set(features_df.columns)
        if missing_features:
            validation_results["errors"].append(
                f"Missing expected features: {missing_features}"
            )
            validation_results["valid"] = False

        extra_features = set(features_df.columns) - set(self.EXPECTED_FEATURES)
        if extra_features:
            validation_results["warnings"].append(
                f"Unexpected extra features: {extra_features}"
            )

        return validation_results


# Legacy compatibility wrapper
class AdvancedFeatureEngineer(FeatureEngineer):
    """Legacy wrapper for backward compatibility"""

    def __init__(self, db_path="greyhound_racing_data.db"):
        super().__init__(db_path)
        logger.info("Legacy AdvancedFeatureEngineer wrapper initialized")

    def engineer_features(self, data):
        """Legacy method for backward compatibility"""
        if isinstance(data, list):
            return self.create_features_for_dogs(data)
        else:
            return self.create_features_for_dog(data)


# For backward compatibility
EnhancedFeatureEngineer = FeatureEngineer

if __name__ == "__main__":
    # Test the feature engineer
    engineer = FeatureEngineer()

    # Test with sample dog data
    sample_dog = {
        "name": "Test Dog",
        "box_number": 1,
        "weight": 32.5,
        "starting_price": 5.0,
        "avg_position": 3.2,
        "win_rate": 0.15,
        "recent_form": [2, 3, 1, 4, 3],
    }

    sample_race_context = {
        "venue": "Sandown",
        "distance": "500m",
        "grade": "Grade 5",
        "track_condition": "Good",
        "weather": "Clear",
        "temperature": 22.0,
        "humidity": 55.0,
        "wind_speed": 8.0,
        "field_size": 8,
    }

    # Test single dog features
    features = engineer.create_features_for_dog(sample_dog, sample_race_context)
    print(f"Generated {len(features)} features for single dog")
    print(f"Feature validation: {len(features) == 66}")

    # Test race features with constraint handling
    dogs = [sample_dog.copy() for _ in range(6)]  # 6 dogs
    race_features = engineer.create_features_for_dogs(dogs, sample_race_context)
    validation = engineer.validate_features(race_features)

    print(f"Race features shape: {race_features.shape}")
    print(f"Validation result: {validation}")
    print("✅ Feature engineering system ready!")
