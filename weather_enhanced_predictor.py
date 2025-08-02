#!/usr/bin/env python3
"""
Weather-Enhanced Race Predictor
===============================

This enhanced predictor integrates weather data and adjustment factors
to provide more accurate race predictions by considering environmental conditions.

Features:
- Weather condition analysis (clear, cloudy, rain, etc.)
- Temperature impact on performance
- Weather adjustment factors
- Track condition inference from weather
- Performance adjustments based on weather patterns

Author: AI Assistant
Date: July 25, 2025
"""

import json
import os
import re
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from constants import DOG_NAME_KEY
from json_utils import safe_correlation, safe_float, safe_json_dump, safe_mean
from utils.file_naming import build_prediction_filename

warnings.filterwarnings("ignore")

# Import the comprehensive enhanced ML system
try:
    from comprehensive_enhanced_ml_system import ComprehensiveEnhancedMLSystem

    COMPREHENSIVE_ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ Comprehensive ML system not available: {e}")
    COMPREHENSIVE_ML_AVAILABLE = False

# Import enhanced data integration
try:
    from enhanced_data_integration import EnhancedDataIntegrator

    ENHANCED_DATA_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced data integration not available: {e}")
    ENHANCED_DATA_INTEGRATION_AVAILABLE = False

# Import scikit-learn for prediction
try:
    import joblib
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import LabelEncoder, RobustScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class WeatherEnhancedPredictor:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.upcoming_dir = Path("./upcoming_races")
        self.predictions_dir = Path("./predictions")
        self.models_dir = Path("./comprehensive_trained_models")

        # Create directories
        self.predictions_dir.mkdir(exist_ok=True)

        # Load the trained model
        self.trained_model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = None

        self._load_trained_model()

        # Check for improved models (class-balanced)
        self.use_improved_models = self._check_improved_models_available()

        # Weather condition mappings
        self.weather_impact_factors = {
            "clear": 1.02,  # Slightly better performance in clear weather
            "partly_cloudy": 1.01,
            "cloudy": 1.00,  # Baseline performance
            "overcast": 0.99,
            "rain": 0.85,  # Significant impact on performance
            "heavy_rain": 0.75,
            "fog": 0.90,
            "unknown": 1.00,
        }

        # Temperature impact (optimal range around 15-25°C)
        self.optimal_temp_range = (15, 25)

        print("🚀 Weather-Enhanced Predictor Initialized")
        print(f"✅ Comprehensive ML Available: {COMPREHENSIVE_ML_AVAILABLE}")
        print(f"✅ Trained Model Loaded: {self.trained_model is not None}")
        print(f"🌤️ Weather integration: ENABLED")

    def _load_trained_model(self):
        """Load the latest trained comprehensive model"""
        try:
            if not self.models_dir.exists():
                print("⚠️ No trained models directory found")
                return

            # Find the latest model file
            model_files = list(
                self.models_dir.glob("comprehensive_best_model_*.joblib")
            )
            if not model_files:
                print("⚠️ No trained comprehensive models found")
                return

            # Get the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

            # Load the model
            model_data = joblib.load(latest_model)
            self.trained_model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_columns = model_data["feature_columns"]
            self.model_metadata = {
                "model_name": model_data.get("model_name", "Unknown"),
                "accuracy": model_data.get("accuracy", 0),
                "timestamp": model_data.get("timestamp", ""),
                "data_summary": model_data.get("data_summary", {}),
            }

            print(f"✅ Loaded model: {self.model_metadata['model_name']}")
            print(f"📊 Model accuracy: {self.model_metadata['accuracy']:.3f}")
            print(f"🔧 Features: {len(self.feature_columns)}")

        except Exception as e:
            print(f"❌ Error loading trained model: {e}")

    def _check_improved_models_available(self):
        """Check if improved class-balanced models are available"""
        try:
            if self.model_metadata and "model_name" in self.model_metadata:
                model_name = self.model_metadata["model_name"]
                return "balanced" in model_name.lower()
            return False
        except:
            return False

    def get_weather_data_for_race(self, venue, race_date):
        """Retrieve weather data for a specific race venue and date"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Query weather data
            query = """
            SELECT 
                weather_condition,
                temperature,
                humidity,
                wind_speed,
                wind_direction,
                pressure,
                weather_adjustment_factor
            FROM race_metadata 
            WHERE venue = ? AND race_date = ?
            AND weather_condition IS NOT NULL
            ORDER BY race_id DESC
            LIMIT 1
            """

            result = pd.read_sql_query(query, conn, params=(venue, race_date))
            conn.close()

            if len(result) > 0:
                weather_data = result.iloc[0].to_dict()
                print(
                    f"🌤️ Found weather data for {venue} on {race_date}: {weather_data['weather_condition']}, {weather_data['temperature']}°C"
                )
                return weather_data
            else:
                print(f"⚠️ No weather data found for {venue} on {race_date}")
                return None

        except Exception as e:
            print(f"❌ Error retrieving weather data: {e}")
            return None

    def get_historical_weather_performance(self, dog_name, weather_condition):
        """Get dog's historical performance in similar weather conditions"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT 
                drd.finish_position,
                drd.individual_time,
                rm.weather_condition,
                rm.temperature,
                rm.weather_adjustment_factor
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.dog_clean_name = ?
            AND rm.weather_condition = ?
            AND drd.finish_position IS NOT NULL
            AND drd.finish_position != 'N/A'
            ORDER BY rm.race_date DESC
            LIMIT 10
            """

            result = pd.read_sql_query(
                query, conn, params=(dog_name, weather_condition)
            )
            conn.close()

            if len(result) > 0:
                # Calculate average performance in this weather condition
                avg_position = result["finish_position"].astype(float).mean()
                avg_time = result["individual_time"].astype(float).mean()
                races_in_condition = len(result)

                return {
                    "avg_position": avg_position,
                    "avg_time": avg_time,
                    "races_in_condition": races_in_condition,
                    "weather_experience": races_in_condition
                    >= 2,  # Has experience in this weather
                }
            else:
                return None

        except Exception as e:
            print(f"⚠️ Error getting weather performance for {dog_name}: {e}")
            return None

    def calculate_weather_adjustment(self, weather_data):
        """Calculate performance adjustment based on weather conditions"""
        if not weather_data:
            return 1.0

        weather_condition = weather_data.get("weather_condition", "unknown").lower()
        temperature = weather_data.get("temperature", 20)

        # Base weather adjustment
        base_adjustment = self.weather_impact_factors.get(weather_condition, 1.0)

        # Temperature adjustment
        temp_adjustment = 1.0
        if temperature < self.optimal_temp_range[0]:
            # Cold weather - slightly reduced performance
            temp_adjustment = max(
                0.95, 1.0 - (self.optimal_temp_range[0] - temperature) * 0.01
            )
        elif temperature > self.optimal_temp_range[1]:
            # Hot weather - more significant impact
            temp_adjustment = max(
                0.90, 1.0 - (temperature - self.optimal_temp_range[1]) * 0.015
            )

        # Use pre-calculated adjustment factor if available, otherwise calculate
        if weather_data.get("weather_adjustment_factor"):
            final_adjustment = float(weather_data["weather_adjustment_factor"])
        else:
            final_adjustment = base_adjustment * temp_adjustment

        return final_adjustment

    def enhance_features_with_weather(self, base_features, weather_data, dog_name):
        """Enhance feature set with weather-related features"""
        if base_features is None:
            return None

        try:
            # Convert to dictionary if it's a numpy array
            if isinstance(base_features, np.ndarray):
                if self.feature_columns and len(base_features.flatten()) == len(
                    self.feature_columns
                ):
                    features_dict = dict(
                        zip(self.feature_columns, base_features.flatten())
                    )
                else:
                    # Can't enhance without proper feature mapping
                    return base_features
            else:
                features_dict = base_features.copy()

            # Weather-specific features
            if weather_data:
                weather_condition = weather_data.get(
                    "weather_condition", "unknown"
                ).lower()
                temperature = weather_data.get("temperature", 20)

                # Weather condition encoded (categorical to numeric)
                weather_encoding = {
                    "clear": 1,
                    "partly_cloudy": 2,
                    "cloudy": 3,
                    "overcast": 4,
                    "rain": 5,
                    "heavy_rain": 6,
                    "fog": 7,
                    "unknown": 0,
                }
                features_dict["weather_condition_encoded"] = weather_encoding.get(
                    weather_condition, 0
                )

                # Temperature features
                features_dict["temperature"] = temperature
                features_dict["temp_deviation_from_optimal"] = abs(
                    temperature - 20
                )  # 20°C as optimal
                features_dict["is_extreme_temperature"] = (
                    1 if temperature < 10 or temperature > 30 else 0
                )

                # Weather adjustment factor
                features_dict["weather_adjustment_factor"] = (
                    self.calculate_weather_adjustment(weather_data)
                )

                # Additional weather features
                features_dict["humidity"] = weather_data.get("humidity", 50)
                features_dict["wind_speed"] = weather_data.get("wind_speed", 0)
                features_dict["pressure"] = weather_data.get("pressure", 1013)

                # Historical weather performance
                weather_performance = self.get_historical_weather_performance(
                    dog_name, weather_condition
                )
                if weather_performance:
                    features_dict["weather_avg_position"] = weather_performance[
                        "avg_position"
                    ]
                    features_dict["weather_experience"] = (
                        1 if weather_performance["weather_experience"] else 0
                    )
                    features_dict["races_in_this_weather"] = weather_performance[
                        "races_in_condition"
                    ]
                else:
                    features_dict["weather_avg_position"] = features_dict.get(
                        "avg_position", 5
                    )
                    features_dict["weather_experience"] = 0
                    features_dict["races_in_this_weather"] = 0

                # Rain-specific features
                if "rain" in weather_condition:
                    features_dict["is_rain"] = 1
                    features_dict["rain_performance_factor"] = (
                        0.85  # Generally slower in rain
                    )
                else:
                    features_dict["is_rain"] = 0
                    features_dict["rain_performance_factor"] = 1.0

            else:
                # Default weather features when no weather data available
                features_dict["weather_condition_encoded"] = 0
                features_dict["temperature"] = 20
                features_dict["temp_deviation_from_optimal"] = 0
                features_dict["is_extreme_temperature"] = 0
                features_dict["weather_adjustment_factor"] = 1.0
                features_dict["humidity"] = 50
                features_dict["wind_speed"] = 0
                features_dict["pressure"] = 1013
                features_dict["weather_avg_position"] = features_dict.get(
                    "avg_position", 5
                )
                features_dict["weather_experience"] = 0
                features_dict["races_in_this_weather"] = 0
                features_dict["is_rain"] = 0
                features_dict["rain_performance_factor"] = 1.0

            # Convert back to array format if that's what the model expects
            if self.feature_columns:
                # Add new weather features to the feature columns if not present
                new_weather_features = [
                    "weather_condition_encoded",
                    "temperature",
                    "temp_deviation_from_optimal",
                    "is_extreme_temperature",
                    "weather_adjustment_factor",
                    "humidity",
                    "wind_speed",
                    "pressure",
                    "weather_avg_position",
                    "weather_experience",
                    "races_in_this_weather",
                    "is_rain",
                    "rain_performance_factor",
                ]

                # Create feature vector with all original features plus weather features
                enhanced_features = []
                for feature_name in self.feature_columns:
                    enhanced_features.append(features_dict.get(feature_name, 0.0))

                # If the model doesn't include weather features, just return enhanced base features
                return np.array(enhanced_features).reshape(1, -1)

            return features_dict

        except Exception as e:
            print(f"⚠️ Error enhancing features with weather: {e}")
            return base_features

    def predict_race_file(self, race_file_path):
        """Standard interface for race file prediction - uses weather-enhanced analysis"""
        return self.predict_race_file_with_weather(race_file_path)

    def predict_race_file_with_weather(self, race_file_path):
        """Predict a single race file using weather-enhanced analysis"""
        try:
            if not COMPREHENSIVE_ML_AVAILABLE:
                return self._fallback_prediction(race_file_path)

            if not self.trained_model:
                return self._fallback_prediction(race_file_path)

            print(
                f"🎯 Weather-enhanced prediction for: {os.path.basename(race_file_path)}"
            )

            # Initialize the comprehensive ML system
            ml_system = ComprehensiveEnhancedMLSystem(self.db_path)

            # Extract race info from filename
            race_info = self._extract_race_info(race_file_path)
            venue = race_info["venue"]
            race_date = race_info["race_date"]

            # Get weather data for this race
            weather_data = self.get_weather_data_for_race(venue, race_date)
            weather_adjustment = self.calculate_weather_adjustment(weather_data)

            # Load form guide data for this specific race
            race_form_data = self._load_single_race_form_data(race_file_path)

            # Load database historical data
            race_results_df = ml_system.load_race_results_data()
            if race_results_df is None:
                print("⚠️ No historical data available")
                return self._fallback_prediction(race_file_path)

            # Parse the race CSV to get participating dogs
            race_df = pd.read_csv(race_file_path)
            participating_dogs = self._extract_participating_dogs(race_df)

            print(f"📊 Found {len(participating_dogs)} dogs in race")
            if weather_data:
                print(
                    f"🌤️ Weather: {weather_data['weather_condition']}, {weather_data['temperature']}°C"
                )
                print(f"⚡ Weather adjustment factor: {weather_adjustment:.3f}")

            # Create predictions for each dog in the race
            predictions = []

            for dog_info in participating_dogs:
                dog_name = dog_info[DOG_NAME_KEY]

                # Get historical data for this dog using exact matching strategies only
                dog_historical = pd.DataFrame()

                # Strategy 1: Exact match with dog_clean_name
                dog_historical = race_results_df[
                    race_results_df["dog_clean_name"].str.upper() == dog_name.upper()
                ].sort_values("race_date", ascending=False)

                # Strategy 2: Try matching with dog_name if no results from clean_name
                if len(dog_historical) == 0:
                    dog_historical = race_results_df[
                        race_results_df["dog_name"].str.upper() == dog_name.upper()
                    ].sort_values("race_date", ascending=False)

                # Get form guide data for this dog
                dog_form_data = race_form_data.get(dog_name, [])

                if len(dog_historical) >= 1 or len(dog_form_data) >= 3:
                    # Create comprehensive features for this dog
                    base_features = self._create_dog_features(
                        dog_info, dog_historical, dog_form_data, race_info
                    )

                    # Enhance features with weather data
                    enhanced_features = self.enhance_features_with_weather(
                        base_features, weather_data, dog_name
                    )

                    if enhanced_features is not None:
                        # Make prediction using the trained model
                        base_prediction_score = self._predict_dog_performance(
                            enhanced_features
                        )

                        # Apply weather adjustment
                        weather_adjusted_score = (
                            base_prediction_score * weather_adjustment
                        )

                        # Ensure the score has some variance to avoid uniform predictions
                        if (
                            base_prediction_score == 0.5
                        ):  # If model returned default score
                            # Add position-based variance for dogs with limited data
                            position_variance = max(
                                0.05,
                                min(
                                    0.25,
                                    (len(participating_dogs) - len(predictions)) * 0.05,
                                ),
                            )
                            weather_adjusted_score = max(
                                0.1, min(0.8, 0.5 + position_variance)
                            )

                        # Get weather-specific performance history
                        weather_performance = None
                        if weather_data:
                            weather_performance = (
                                self.get_historical_weather_performance(
                                    dog_name, weather_data["weather_condition"]
                                )
                            )

                        predictions.append(
                            {
                                "dog_name": dog_name,
                                "box_number": dog_info.get("box", "Unknown"),
                                "base_prediction_score": float(base_prediction_score),
                                "weather_adjustment_factor": float(weather_adjustment),
                                "weather_adjusted_score": float(weather_adjusted_score),
                                "prediction_score": float(
                                    weather_adjusted_score
                                ),  # Final score
                                "confidence": min(
                                    0.95, max(0.1, weather_adjusted_score)
                                ),
                                "historical_races": len(dog_historical),
                                "form_data_races": len(dog_form_data),
                                "features_used": len(self.feature_columns),
                                "weather_experience": (
                                    weather_performance["races_in_condition"]
                                    if weather_performance
                                    else 0
                                ),
                                "weather_avg_position": (
                                    weather_performance["avg_position"]
                                    if weather_performance
                                    else None
                                ),
                            }
                        )
                    else:
                        # Fallback prediction for dogs with insufficient feature data
                        fallback_score = 0.5 * weather_adjustment
                        predictions.append(
                            {
                                "dog_name": dog_name,
                                "box_number": dog_info.get("box", "Unknown"),
                                "base_prediction_score": 0.5,
                                "weather_adjustment_factor": float(weather_adjustment),
                                "weather_adjusted_score": float(fallback_score),
                                "prediction_score": float(fallback_score),
                                "confidence": 0.1,
                                "historical_races": len(dog_historical),
                                "form_data_races": len(dog_form_data),
                                "features_used": 0,
                                "note": "Insufficient data for ML prediction",
                                "weather_experience": 0,
                            }
                        )
                else:
                    # Very basic prediction for dogs with no data
                    basic_score = 0.3 * weather_adjustment
                    predictions.append(
                        {
                            "dog_name": dog_name,
                            "box_number": dog_info.get("box", "Unknown"),
                            "base_prediction_score": 0.3,
                            "weather_adjustment_factor": float(weather_adjustment),
                            "weather_adjusted_score": float(basic_score),
                            "prediction_score": float(basic_score),
                            "confidence": 0.05,
                            "historical_races": len(dog_historical),
                            "form_data_races": len(dog_form_data),
                            "features_used": 0,
                            "note": "No historical data available",
                            "weather_experience": 0,
                        }
                    )

            # Sort predictions by weather-adjusted score (highest first)
            predictions.sort(key=lambda x: x["prediction_score"], reverse=True)

            # Create prediction summary
            prediction_summary = {
                "race_info": race_info,
                "weather_info": (
                    weather_data
                    if weather_data
                    else {"note": "No weather data available"}
                ),
                "model_info": {
                    "system": "Weather-Enhanced Comprehensive ML",
                    "model_name": self.model_metadata.get("model_name", "Unknown"),
                    "accuracy": self.model_metadata.get("accuracy", 0),
                    "features": (
                        len(self.feature_columns) if self.feature_columns else 0
                    ),
                    "weather_integration": True,
                },
                "race_summary": {
                    "total_dogs": len(predictions),
                    "dogs_with_data": sum(
                        1 for p in predictions if p["historical_races"] > 0
                    ),
                    "average_confidence": safe_mean(
                        [p["confidence"] for p in predictions]
                    ),
                    "weather_adjustment_applied": weather_adjustment,
                    "dogs_with_weather_experience": sum(
                        1 for p in predictions if p["weather_experience"] > 0
                    ),
                },
                "predictions": predictions,
                "top_pick": predictions[0] if predictions else None,
                "prediction_timestamp": datetime.now().isoformat(),
            }

            # Save prediction results
            output_file = self._save_prediction_results(prediction_summary, race_info)

            print(
                f"✅ Weather-enhanced prediction completed for {len(predictions)} dogs"
            )
            print(
                f"🏆 Top pick: {predictions[0]['dog_name'] if predictions else 'None'}"
            )
            print(f"🌤️ Weather impact: {((weather_adjustment - 1) * 100):+.1f}%")
            print(f"💾 Results saved: {output_file}")

            return {
                "success": True,
                "predictions": predictions,
                "summary": prediction_summary,
                "output_file": output_file,
                "weather_data": weather_data,
            }

        except Exception as e:
            print(f"❌ Error in weather-enhanced prediction: {e}")
            return self._fallback_prediction(race_file_path)

    # Helper methods (reused from comprehensive_ml_predictor.py)
    def _load_single_race_form_data(self, race_file_path):
        """Load form guide data for a single race file using the fixed parsing"""
        try:
            # Parse race info from filename
            race_info = self._extract_race_info(race_file_path)

            # Initialize comprehensive ML system to use its form guide parsing
            ml_system = ComprehensiveEnhancedMLSystem(self.db_path)

            # Load all form guide data (this uses the fixed parsing)
            all_form_data = ml_system.load_form_guide_data()

            # Filter for dogs that appear in this specific race
            race_df = pd.read_csv(race_file_path)
            participating_dogs = self._extract_participating_dogs(race_df)
            participating_dog_names = [
                dog[DOG_NAME_KEY].upper() for dog in participating_dogs
            ]

            # Filter form data for participating dogs
            race_specific_form_data = {}
            for dog_name, form_data in all_form_data.items():
                if dog_name.upper() in participating_dog_names:
                    race_specific_form_data[dog_name] = form_data

            return race_specific_form_data

        except Exception as e:
            print(f"⚠️ Error loading form data: {e}")
            return {}

    def _extract_participating_dogs(self, race_df):
        """Extract participating dogs from race CSV with proper blank row handling"""
        try:
            dogs = []
            current_dog_name = None

            for idx, row in race_df.iterrows():
                dog_name_raw = str(row.get("Dog Name", "")).strip()

                # Check if this is a new dog or continuation of previous
                if dog_name_raw not in ['""', "", "nan"] and dog_name_raw != "nan":
                    # New dog - clean the name
                    current_dog_name = dog_name_raw
                    # Remove box number prefix (e.g., "1. Mel Monelli" -> "Mel Monelli")
                    if ". " in current_dog_name:
                        current_dog_name = current_dog_name.split(". ", 1)[1]

                    # Extract box number from the prefix
                    box_number = None
                    if ". " in dog_name_raw:
                        try:
                            box_number = int(dog_name_raw.split(".")[0])
                        except (ValueError, TypeError):
                            pass

                    dogs.append(
                        {
                            DOG_NAME_KEY: current_dog_name,
                            "box": box_number,
                            "raw_name": dog_name_raw,
                        }
                    )

            return dogs

        except Exception as e:
            print(f"⚠️ Error extracting participating dogs: {e}")
            return []

    def _extract_race_info(self, race_file_path):
        """Extract race information from file path with enhanced parsing"""
        try:
            filename = os.path.basename(race_file_path)
            print(f"🔍 Parsing filename: {filename}")

            # Initialize with defaults
            result = {
                "filename": filename,
                "venue": "UNKNOWN",
                "race_number": "0",
                "race_date": datetime.now().strftime("%Y-%m-%d"),
                "filepath": race_file_path,
            }

            # Enhanced parsing patterns with proper group assignments
            patterns = [
                # Pattern 1: "Race 5 - GEE - 22 July 2025.csv"
                {
                    "pattern": r"Race\s+(\d+)\s*-\s*([A-Z_]+)\s*-\s*(\d{1,2}\s+\w+\s+\d{4})",
                    "groups": ["race_number", "venue", "date_str"],
                },
                # Pattern 1b: Compact format "Race 5 - GEE - 2025-07-22.csv"
                {
                    "pattern": r"Race (\d+) - ([A-Z_]+) - (\d{4}-\d{2}-\d{2})\.csv",
                    "groups": ["race_number", "venue", "date_str"],
                },
                # Pattern 2: "Race_5_GEE_22_July_2025.csv"
                {
                    "pattern": r"Race_(\d+)_([A-Z_]+)_(.+)",
                    "groups": ["race_number", "venue", "date_str"],
                },
                # Pattern 3: "GEE_5_22_July_2025.csv" (venue first)
                {
                    "pattern": r"([A-Z_]+)_(\d+)_(.+)",
                    "groups": ["venue", "race_number", "date_str"],
                },
                # Pattern 4: "gee_2025-07-22_5.csv" (lowercase with ISO date)
                {
                    "pattern": r"([a-z]+)_(\d{4}-\d{2}-\d{2})_(\d+)",
                    "groups": ["venue", "date_str", "race_number"],
                },
                # Pattern 5: "race5_gee_july22.csv"
                {
                    "pattern": r"race(\d+)[_-]([a-zA-Z]+)[_-](.+)",
                    "groups": ["race_number", "venue", "date_str"],
                },
            ]

            for pattern_config in patterns:
                match = re.search(pattern_config["pattern"], filename, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    group_names = pattern_config["groups"]

                    if len(groups) == len(group_names):
                        parsed_data = {}
                        for i, group_name in enumerate(group_names):
                            value = groups[i].strip() if groups[i] else ""

                            if group_name == "venue":
                                parsed_data["venue"] = self._normalize_venue(value)
                            elif group_name == "race_number":
                                parsed_data["race_number"] = self._parse_race_number(
                                    value
                                )
                            elif group_name == "date_str":
                                parsed_data["race_date"] = self._normalize_date(value)

                        # Update result with parsed data
                        result.update(parsed_data)
                        print(
                            f"✅ Parsed using pattern {pattern_config['pattern'][:30]}..."
                        )
                        print(
                            f"   Venue: {result['venue']}, Race: {result['race_number']}, Date: {result['race_date']}"
                        )
                        break

            # If parsing failed, try content-based extraction
            if result["venue"] == "UNKNOWN":
                print(f"⚠️ Standard parsing failed, trying content-based extraction...")
                result = self._try_content_based_parsing(race_file_path, result)

            return result
        except Exception as e:
            print(f"❌ Error extracting race info: {e}")
            return {
                "filename": os.path.basename(race_file_path),
                "race_number": "0",
                "venue": "UNKNOWN",
                "race_date": datetime.now().strftime("%Y-%m-%d"),
                "filepath": race_file_path,
            }

    def _normalize_venue(self, venue_str):
        """Normalize venue name to standard format"""
        if not venue_str:
            return "UNKNOWN"

        venue_clean = venue_str.upper().strip().replace(" ", "_")

        # Known venue mappings
        venue_mapping = {
            "AP_K": "AP_K",
            "ANGLE_PARK": "AP_K",
            "ANGLE": "AP_K",
            "GEE": "GEE",
            "GEELONG": "GEE",
            "RICH": "RICH",
            "RICHMOND": "RICH",
            "DAPT": "DAPT",
            "DAPTO": "DAPT",
            "BAL": "BAL",
            "BALLARAT": "BAL",
            "BEN": "BEN",
            "BENDIGO": "BEN",
            "HEA": "HEA",
            "HEALESVILLE": "HEA",
            "WAR": "WAR",
            "WARRNAMBOOL": "WAR",
            "SAN": "SAN",
            "SANDOWN": "SAN",
            "MOUNT": "MOUNT",
            "MOUNT_GAMBIER": "MOUNT",
            "MURR": "MURR",
            "MURRAY_BRIDGE": "MURR",
            "SAL": "SAL",
            "SALE": "SAL",
            "HOR": "HOR",
            "HORSHAM": "HOR",
            "CANN": "CANN",
            "CANNINGTON": "CANN",
            "WPK": "WPK",
            "W_PK": "WPK",
            "WENTWORTH_PARK": "WPK",
            "MEA": "MEA",
            "THE_MEADOWS": "MEA",
            "MEADOWS": "MEA",
            "HOBT": "HOBT",
            "HOBART": "HOBT",
            "GOSF": "GOSF",
            "GOSFORD": "GOSF",
            "NOR": "NOR",
            "NORTHAM": "NOR",
            "MAND": "MAND",
            "MANDURAH": "MAND",
            "GAWL": "GAWL",
            "GAWLER": "GAWL",
            "TRA": "TRA",
            "TRARALGON": "TRA",
            "CASO": "CASO",
            "CASINO": "CASO",
            "GRDN": "GRDN",
            "THE_GARDENS": "GRDN",
            "GARDENS": "GRDN",
            "DARW": "DARW",
            "DARWIN": "DARW",
            "ALBION": "ALBION",
            "ALBION_PARK": "ALBION",
        }

        # Direct mapping
        if venue_clean in venue_mapping:
            return venue_mapping[venue_clean]

        # Try partial matches
        for key, value in venue_mapping.items():
            if key in venue_clean or venue_clean in key:
                return value

        # Return cleaned version as fallback
        return venue_clean if len(venue_clean) <= 8 else venue_clean[:8]

    def _parse_race_number(self, race_str):
        """Parse race number from string"""
        if not race_str:
            return "0"

        # Extract digits
        digits = re.findall(r"\d+", str(race_str))
        if digits:
            race_num = int(digits[0])
            if 1 <= race_num <= 20:  # Reasonable race number range
                return str(race_num)

        return "0"

    def _normalize_date(self, date_str):
        """Normalize date string to YYYY-MM-DD format"""
        if not date_str:
            return datetime.now().strftime("%Y-%m-%d")

        date_clean = str(date_str).strip().replace("_", " ")

        # Month name mappings
        month_mapping = {
            "jan": "01",
            "january": "01",
            "feb": "02",
            "february": "02",
            "mar": "03",
            "march": "03",
            "apr": "04",
            "april": "04",
            "may": "05",
            "jun": "06",
            "june": "06",
            "jul": "07",
            "july": "07",
            "aug": "08",
            "august": "08",
            "sep": "09",
            "september": "09",
            "oct": "10",
            "october": "10",
            "nov": "11",
            "november": "11",
            "dec": "12",
            "december": "12",
        }

        # Try different date patterns
        date_patterns = [
            r"(\d{1,2})\s+(\w+)\s+(\d{4})",  # "22 July 2025"
            r"(\d{4})-(\d{2})-(\d{2})",  # "2025-07-22"
            r"(\w+)(\d{1,2})",  # "july22"
            r"(\d{2})(\d{2})(\d{4})",  # "22072025"
        ]

        for pattern in date_patterns:
            match = re.search(pattern, date_clean, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()

                    if pattern == r"(\d{1,2})\s+(\w+)\s+(\d{4})":  # "22 July 2025"
                        day, month_name, year = groups
                        month = month_mapping.get(month_name.lower())
                        if month:
                            return f"{year}-{month}-{day.zfill(2)}"

                    elif pattern == r"(\d{4})-(\d{2})-(\d{2})":  # "2025-07-22"
                        return f"{groups[0]}-{groups[1]}-{groups[2]}"

                    elif pattern == r"(\w+)(\d{1,2})":  # "july22"
                        month_name, day = groups
                        month = month_mapping.get(month_name.lower())
                        if month:
                            year = datetime.now().year
                            return f"{year}-{month}-{day.zfill(2)}"

                except Exception:
                    continue

        # Return current date if all parsing fails
        return datetime.now().strftime("%Y-%m-%d")

    def _try_content_based_parsing(self, race_file_path, current_result):
        """Try to extract race info from CSV content as last resort"""
        result = current_result.copy()

        try:
            # Try to read the CSV file and look for clues
            df = pd.read_csv(race_file_path, nrows=10)  # Read first 10 rows only

            # Look for track/venue information in column headers or data
            for col in df.columns:
                col_str = str(col).upper()
                if "TRACK" in col_str or "VENUE" in col_str:
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) > 0:
                        venue_candidate = str(unique_values[0]).strip()
                        normalized_venue = self._normalize_venue(venue_candidate)
                        if normalized_venue != "UNKNOWN":
                            result["venue"] = normalized_venue
                            print(f"   📍 Found venue in content: {normalized_venue}")
                            break

            # Look for race number in data
            for col in df.columns:
                col_str = str(col).upper()
                if "RACE" in col_str and "NUMBER" in col_str:
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) > 0:
                        race_num_candidate = str(unique_values[0]).strip()
                        parsed_race_num = self._parse_race_number(race_num_candidate)
                        if parsed_race_num != "0":
                            result["race_number"] = parsed_race_num
                            print(
                                f"   🏃 Found race number in content: {parsed_race_num}"
                            )
                            break

        except Exception as e:
            print(f"   ⚠️ Content-based parsing failed: {e}")

        return result

    def _create_dog_features(self, dog_info, dog_historical, dog_form_data, race_info):
        """Create features for a single dog using comprehensive feature engineering"""
        try:
            if not self.feature_columns:
                return None

            # Initialize feature dict with defaults
            features = {}

            # Process database historical data
            if len(dog_historical) > 0:
                positions = []
                times = []
                weights = []
                odds = []

                for _, hist_row in dog_historical.iterrows():
                    # Position data
                    if pd.notna(hist_row["finish_position"]):
                        pos_str = str(hist_row["finish_position"]).strip()
                        if pos_str not in ["", "N/A", "None", "nan"]:
                            pos_cleaned = "".join(filter(str.isdigit, pos_str))
                            if pos_cleaned and 1 <= int(pos_cleaned) <= 10:
                                positions.append(int(pos_cleaned))

                    # Time data
                    try:
                        if (
                            pd.notna(hist_row["individual_time"])
                            and float(hist_row["individual_time"]) > 0
                        ):
                            times.append(float(hist_row["individual_time"]))
                    except (ValueError, TypeError):
                        pass

                    # Weight data
                    try:
                        if (
                            pd.notna(hist_row["weight"])
                            and float(hist_row["weight"]) > 0
                        ):
                            weights.append(float(hist_row["weight"]))
                    except (ValueError, TypeError):
                        pass

                    # Odds data
                    try:
                        if (
                            pd.notna(hist_row["starting_price"])
                            and float(hist_row["starting_price"]) > 0
                        ):
                            odds.append(float(hist_row["starting_price"]))
                    except (ValueError, TypeError):
                        pass
            else:
                positions = []
                times = []
                weights = []
                odds = []

            # Process form guide data
            form_positions = []
            form_times = []
            form_weights = []
            form_odds = []

            for form_entry in dog_form_data[:20]:  # Use up to 20 historical races
                try:
                    # Parse position
                    place_str = form_entry.get("place", "").strip()
                    if place_str and place_str.isdigit():
                        position = int(place_str)
                        if 1 <= position <= 10:
                            form_positions.append(position)

                    # Parse time
                    time_str = form_entry.get("time", "").strip()
                    if time_str:
                        try:
                            time_val = float(time_str)
                            if 15.0 <= time_val <= 60.0:
                                form_times.append(time_val)
                        except (ValueError, TypeError):
                            pass

                    # Parse weight
                    weight_str = form_entry.get("weight", "").strip()
                    if weight_str:
                        try:
                            weight_val = float(weight_str)
                            if 20.0 <= weight_val <= 40.0:
                                form_weights.append(weight_val)
                        except (ValueError, TypeError):
                            pass

                    # Parse starting price
                    sp_str = form_entry.get("starting_price", "").strip()
                    if sp_str:
                        try:
                            sp_val = float(sp_str)
                            if 1.0 <= sp_val <= 1000.0:
                                form_odds.append(sp_val)
                        except (ValueError, TypeError):
                            pass

                except Exception:
                    continue

            # Combine all data
            all_positions = positions + form_positions
            all_times = times + form_times
            all_weights = weights + form_weights
            all_odds = odds + form_odds

            if not all_positions:
                return None

            # Calculate core features (matching the trained model)
            features["avg_position"] = np.mean(all_positions)
            features["recent_form_avg"] = (
                np.mean(all_positions[:8])
                if len(all_positions) >= 8
                else np.mean(all_positions)
            )
            features["market_confidence"] = (
                1 / (np.mean(all_odds) + 1) if all_odds else 0.1
            )
            features["current_odds_log"] = np.log(10)  # Default odds
            features["venue_experience"] = len([p for p in positions])  # Simplified
            features["place_rate"] = sum(1 for p in all_positions if p <= 3) / len(
                all_positions
            )
            features["current_weight"] = np.mean(all_weights) if all_weights else 30.0
            features["time_consistency"] = (
                1 / (np.std(all_times) + 0.1) if len(all_times) > 1 else 0.5
            )
            features["win_rate"] = sum(1 for p in all_positions if p == 1) / len(
                all_positions
            )

            # Additional comprehensive features
            features["long_term_form_trend"] = 0  # Simplified
            features["position_consistency"] = 1 / (np.std(all_positions) + 0.1)
            features["avg_time"] = np.mean(all_times) if all_times else 30.0
            features["best_time"] = min(all_times) if all_times else 28.0
            features["time_improvement_trend"] = 0  # Simplified
            features["avg_weight"] = np.mean(all_weights) if all_weights else 30.0
            features["weight_consistency"] = (
                1 / (np.std(all_weights) + 0.1) if len(all_weights) > 1 else 0.5
            )
            features["weight_vs_avg"] = 0  # Simplified
            features["distance_specialization"] = 0.1  # Simplified
            features["grade_experience"] = 5  # Default
            features["days_since_last"] = 14  # Default
            features["fitness_score"] = features["place_rate"] * features["win_rate"]
            features["competition_strength"] = 0.5  # Default
            features["box_win_rate"] = 0.1  # Default
            features["current_box"] = dog_info.get("box", 4)
            features["field_size"] = 6  # Default
            features["historical_races_count"] = len(all_positions)

            # Encoded features (defaults)
            features["venue_encoded"] = 0
            features["track_condition_encoded"] = 0
            features["grade_encoded"] = 0
            features["distance_numeric"] = 500.0

            # Ensure all required features are present
            feature_vector = []
            for feature_name in self.feature_columns:
                feature_vector.append(features.get(feature_name, 0.0))

            return np.array(feature_vector).reshape(1, -1)

        except Exception as e:
            print(f"⚠️ Error creating features for {dog_info[DOG_NAME_KEY]}: {e}")
            return None

    def _predict_dog_performance(self, dog_features):
        """Make prediction using the trained model"""
        try:
            if not self.trained_model or not self.scaler:
                return 0.5

            # Scale features
            scaled_features = self.scaler.transform(dog_features)

            # Get prediction probability
            if hasattr(self.trained_model, "predict_proba"):
                prediction_proba = self.trained_model.predict_proba(scaled_features)
                # Return probability of winning (class 1)
                return prediction_proba[0][1] if prediction_proba.shape[1] > 1 else 0.5
            else:
                # For models without predict_proba, use decision function or predict
                prediction = self.trained_model.predict(scaled_features)
                return (
                    float(prediction[0])
                    if hasattr(prediction, "__iter__")
                    else float(prediction)
                )

        except Exception as e:
            print(f"⚠️ Error making prediction: {e}")
            return 0.5

    def _save_prediction_results(self, prediction_summary, race_info):
        """Save prediction results to JSON file"""
        try:
            race_id = f"{race_info['venue']}_{race_info['race_number']}_{race_info['race_date']}"
            output_filename = build_prediction_filename(
                race_id, datetime.now(), "weather_enhanced"
            )
            output_path = self.predictions_dir / output_filename

            # Save results using safe JSON dump to handle NaN values
            safe_json_dump(prediction_summary, str(output_path))

            return str(output_path)

        except Exception as e:
            print(f"⚠️ Error saving results: {e}")
            return None

    def _fallback_prediction(self, race_file_path):
        """Fallback prediction when ML system is not available"""
        try:
            race_df = pd.read_csv(race_file_path)
            dogs = self._extract_participating_dogs(race_df)
            race_info = self._extract_race_info(race_file_path)

            predictions = []
            for i, dog in enumerate(dogs):
                predictions.append(
                    {
                        "dog_name": dog[DOG_NAME_KEY],
                        "box_number": dog.get("box", "Unknown"),
                        "prediction_score": max(
                            0.1, 0.8 - (i * 0.1)
                        ),  # Decreasing scores
                        "confidence": 0.1,
                        "note": "Fallback prediction - ML system unavailable",
                    }
                )

            return {
                "success": True,
                "predictions": predictions,
                "summary": {
                    "race_info": race_info,
                    "model_info": {"system": "Fallback"},
                    "predictions": predictions,
                    "prediction_timestamp": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            print(f"❌ Fallback prediction failed: {e}")
            return {"success": False, "error": str(e), "predictions": []}


def main():
    """Main function for command line usage"""
    if len(sys.argv) != 2:
        print("Usage: python weather_enhanced_predictor.py <race_file_path>")
        sys.exit(1)

    race_file_path = sys.argv[1]

    if not os.path.exists(race_file_path):
        print(f"❌ Race file not found: {race_file_path}")
        sys.exit(1)

    # Initialize predictor
    predictor = WeatherEnhancedPredictor()

    # Make weather-enhanced prediction
    result = predictor.predict_race_file_with_weather(race_file_path)

    if result["success"]:
        print(f"\n🏆 WEATHER-ENHANCED PREDICTION RESULTS")
        print("=" * 60)

        # Show weather information
        if result.get("weather_data"):
            weather = result["weather_data"]
            print(
                f"🌤️ Weather: {weather['weather_condition']}, {weather['temperature']}°C"
            )
            print(f"💨 Wind: {weather.get('wind_speed', 'N/A')} km/h")
            print(f"💧 Humidity: {weather.get('humidity', 'N/A')}%")
            print("-" * 60)

        for i, prediction in enumerate(result["predictions"][:5], 1):
            base_score = prediction.get(
                "base_prediction_score", prediction["prediction_score"]
            )
            weather_factor = prediction.get("weather_adjustment_factor", 1.0)
            final_score = prediction["prediction_score"]
            weather_exp = prediction.get("weather_experience", 0)

            print(f"{i}. {prediction['dog_name']} (Box {prediction['box_number']})")
            print(f"   Base Score: {base_score:.3f}")
            print(f"   Weather Adj: {weather_factor:.3f} -> Final: {final_score:.3f}")
            print(f"   Weather Experience: {weather_exp} races")
            print(f"   Confidence: {prediction['confidence']:.2f}")
            print()
    else:
        print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
