#!/usr/bin/env python3
"""
Feature Mapper - Bridge Temporal Builder Output to Model Expectations
=====================================================================

Maps features from TemporalFeatureBuilder output to the format expected
by the migrated et_balanced model and other V4 models.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FeatureMapper:
    """Maps temporal builder features to model-expected feature format."""
    
    def __init__(self):
        # Define the mapping from temporal builder features to model expected features
        self.feature_mapping = {
            # Basic mappings
            'historical_avg_position': 'avg_position',
            'historical_avg_position': 'recent_form_avg',  # Approximate
            'weight': 'current_weight',
            'days_since_last_race': 'days_since_last',
            'field_size': 'field_size',
            'box_number': 'current_box',
            'temperature': 'temperature',
            'humidity': 'humidity', 
            'wind_speed': 'wind_speed',
            'historical_win_rate': 'win_rate',
            'historical_place_rate': 'place_rate',
            'historical_avg_time': 'avg_time',
            'historical_best_time': 'best_time',
            'weight': 'avg_weight',  # Approximate current weight as average
            'historical_time_consistency': 'time_consistency',
            'venue_experience': 'venue_experience',
            'grade_experience': 'grade_experience',
            'venue_specific_win_rate': 'box_win_rate',  # Approximate
            'distance': 'distance_numeric',
            'race_frequency': 'historical_races_count',
        }
        
        # Categorical encoding mappings
        self.categorical_encodings = {
            'venue': self._encode_venue,
            'track_condition': self._encode_track_condition, 
            'grade': self._encode_grade,
        }
    
    def _encode_venue(self, venue: str) -> float:
        """Encode venue as float using hash."""
        if pd.isna(venue) or venue == '0':
            venue = 'Unknown'
        return float(hash(str(venue)) % 100 / 100.0)
    
    def _encode_track_condition(self, condition: str) -> float:
        """Encode track condition as float using hash."""
        if pd.isna(condition) or condition == '0':
            condition = 'Good'
        return float(hash(str(condition)) % 50 / 50.0)
    
    def _encode_grade(self, grade: str) -> float:
        """Encode grade as numeric value."""
        if pd.isna(grade) or grade == '0':
            grade = 'Grade 5'
        
        grade_mapping = {
            'Group 1': 10, 'Group 2': 9, 'Group 3': 8,
            'Grade 1': 7, 'Grade 2': 6, 'Grade 3': 5,
            'Grade 4': 4, 'Grade 5': 3, 'Maiden': 2, 'Novice': 1
        }
        
        # Handle various formats
        grade_str = str(grade).strip()
        for key, value in grade_mapping.items():
            if key.lower() in grade_str.lower():
                return float(value)
        
        # Extract number from grade if possible
        import re
        match = re.search(r'(\d+)', grade_str)
        if match:
            return float(match.group(1))
        
        return 3.0  # Default to Grade 5 equivalent
    
    def transform_features(self, temporal_features: pd.DataFrame) -> pd.DataFrame:
        """Transform temporal builder features to model expected format."""
        
        logger.info(f"Transforming {len(temporal_features)} rows from temporal builder format to model format")
        
        # Start with copy of input
        transformed = temporal_features.copy()
        
        # Apply basic mappings
        for temporal_name, model_name in self.feature_mapping.items():
            if temporal_name in transformed.columns:
                if model_name not in transformed.columns:
                    transformed[model_name] = transformed[temporal_name]
        
        # Apply categorical encodings
        if 'venue' in transformed.columns:
            transformed['venue_encoded'] = transformed['venue'].apply(self._encode_venue)
        
        if 'track_condition' in transformed.columns:
            transformed['track_condition_encoded'] = transformed['track_condition'].apply(self._encode_track_condition)
        
        if 'grade' in transformed.columns:
            transformed['grade_encoded'] = transformed['grade'].apply(self._encode_grade)
        
        # Add missing features with reasonable defaults
        self._add_missing_features(transformed)
        
        # Add weather categorization features
        self._add_weather_features(transformed)
        
        # Add traditional features (approximations)
        self._add_traditional_features(transformed)
        
        logger.info(f"Feature transformation complete: {len(transformed.columns)} features")
        return transformed
    
    def _add_missing_features(self, df: pd.DataFrame):
        """Add missing features with sensible defaults."""
        
        defaults = {
            'market_confidence': 0.1,
            'current_odds_log': 2.3,  # log(10) approximate default odds
            'long_term_form_trend': 0.0,
            'position_consistency': 0.5,
            'time_improvement_trend': 0.0,
            'weight_consistency': 0.8,
            'weight_vs_avg': 0.0,
            'distance_specialization': 0.5,
            'fitness_score': 0.5,
            'pressure': 1013.0,
            'weather_adjustment_factor': 0.7,
            'weather_experience_count': 5,
            'weather_performance': 0.5,
            'competition_strength': 0.2,
        }
        
        for feature, default_value in defaults.items():
            if feature not in df.columns:
                df[feature] = default_value
    
    def _add_weather_features(self, df: pd.DataFrame):
        """Add weather categorization features."""
        
        # Weather condition dummies
        if 'weather' in df.columns:
            df['weather_clear'] = df['weather'].apply(
                lambda x: 1.0 if pd.notna(x) and ('clear' in str(x).lower() or 'fine' in str(x).lower()) else 0.0
            )
            df['weather_cloudy'] = df['weather'].apply(
                lambda x: 1.0 if pd.notna(x) and ('cloud' in str(x).lower() or 'overcast' in str(x).lower()) else 0.0
            )
            df['weather_rain'] = df['weather'].apply(
                lambda x: 1.0 if pd.notna(x) and ('rain' in str(x).lower() or 'shower' in str(x).lower()) else 0.0
            )
            df['weather_fog'] = df['weather'].apply(
                lambda x: 1.0 if pd.notna(x) and ('fog' in str(x).lower() or 'mist' in str(x).lower()) else 0.0
            )
        else:
            df['weather_clear'] = 1.0
            df['weather_cloudy'] = 0.0
            df['weather_rain'] = 0.0
            df['weather_fog'] = 0.0
        
        # Temperature categories
        if 'temperature' in df.columns:
            df['temp_cold'] = (df['temperature'] < 10).astype(float)
            df['temp_cool'] = ((df['temperature'] >= 10) & (df['temperature'] < 18)).astype(float)
            df['temp_optimal'] = ((df['temperature'] >= 18) & (df['temperature'] < 25)).astype(float)
            df['temp_warm'] = ((df['temperature'] >= 25) & (df['temperature'] < 30)).astype(float)
            df['temp_hot'] = (df['temperature'] >= 30).astype(float)
        else:
            df['temp_cold'] = 0.0
            df['temp_cool'] = 0.0
            df['temp_optimal'] = 1.0
            df['temp_warm'] = 0.0
            df['temp_hot'] = 0.0
        
        # Wind categories
        if 'wind_speed' in df.columns:
            df['wind_calm'] = (df['wind_speed'] < 5).astype(float)
            df['wind_light'] = ((df['wind_speed'] >= 5) & (df['wind_speed'] < 15)).astype(float)
            df['wind_moderate'] = ((df['wind_speed'] >= 15) & (df['wind_speed'] < 25)).astype(float)
            df['wind_strong'] = (df['wind_speed'] >= 25).astype(float)
        else:
            df['wind_calm'] = 0.0
            df['wind_light'] = 1.0
            df['wind_moderate'] = 0.0
            df['wind_strong'] = 0.0
        
        # Humidity categories
        if 'humidity' in df.columns:
            df['humidity_low'] = (df['humidity'] < 40).astype(float)
            df['humidity_normal'] = ((df['humidity'] >= 40) & (df['humidity'] < 70)).astype(float)
            df['humidity_high'] = (df['humidity'] >= 70).astype(float)
        else:
            df['humidity_low'] = 0.0
            df['humidity_normal'] = 1.0
            df['humidity_high'] = 0.0
    
    def _add_traditional_features(self, df: pd.DataFrame):
        """Add traditional analysis features as approximations."""
        
        # Use existing features to approximate traditional ones
        if 'historical_win_rate' in df.columns and 'historical_place_rate' in df.columns:
            df['traditional_overall_score'] = (df['historical_win_rate'] + df['historical_place_rate']) / 2
            df['traditional_performance_score'] = df['historical_win_rate'] * 1.2
            df['traditional_form_score'] = df['historical_place_rate'] * 1.1
        else:
            df['traditional_overall_score'] = 0.35
            df['traditional_performance_score'] = 0.3
            df['traditional_form_score'] = 0.3
        
        if 'historical_time_consistency' in df.columns:
            df['traditional_consistency_score'] = df['historical_time_consistency'] / 10.0  # Normalize
        else:
            df['traditional_consistency_score'] = 0.3
        
        # Traditional scores based on experience
        if 'venue_experience' in df.columns:
            df['traditional_experience_score'] = np.minimum(df['venue_experience'] / 10.0, 1.0)
        else:
            df['traditional_experience_score'] = 0.2
        
        if 'grade_experience' in df.columns:
            df['traditional_class_score'] = np.minimum(df['grade_experience'] / 5.0, 1.0)
        else:
            df['traditional_class_score'] = 0.5
        
        # Default traditional features
        defaults = {
            'traditional_confidence_level': 0.2,
            'traditional_fitness_score': 0.5,
            'traditional_trainer_score': 0.5,
            'traditional_track_condition_score': 0.5,
            'traditional_distance_score': 0.5,
            'traditional_key_factors_count': 0,
            'traditional_risk_factors_count': 2,
        }
        
        for feature, default_value in defaults.items():
            if feature not in df.columns:
                df[feature] = default_value


def apply_feature_mapping_to_migrated_model(temporal_features: pd.DataFrame) -> pd.DataFrame:
    """Apply feature mapping for migrated model compatibility."""
    mapper = FeatureMapper()
    return mapper.transform_features(temporal_features)
