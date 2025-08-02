#!/usr/bin/env python3
"""
Temporal Feature Builder - Leakage-Safe Feature Engineering
===========================================================

Implements strict temporal separation:
- Target race: Pre-race features only
- Historical races: Post-race features with exponential decay weighting
- Assertion guards to prevent temporal leakage
"""

import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings

logger = logging.getLogger(__name__)

class TemporalFeatureBuilder:
    """Builds features with strict temporal separation to prevent data leakage."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        
        # Define feature categories for temporal separation
        self.pre_race_features = {
            'box_number', 'weight', 'trainer_name', 'dog_clean_name', 'venue', 'grade', 'distance',
            'track_condition', 'weather', 'temperature', 'humidity', 'wind_speed',
            'field_size', 'race_date', 'race_time'
        }
        
        self.post_race_features = {
            'finish_position', 'individual_time', 'sectional_1st', 'sectional_2nd',
            'sectional_3rd', 'margin', 'beaten_margin', 'winning_time',
            'scraped_finish_position', 'scraped_raw_result', 'winner_name',
            'winner_odds', 'winner_margin', 'pir_rating', 'first_sectional',
            'win_time', 'bonus_time'
        }
        
        # Exponential decay factor for historical race weighting
        self.decay_factor = 0.95  # Recent races weighted more heavily
        
    def get_race_timestamp(self, race_row: pd.Series) -> datetime:
        """Get race timestamp, preferring race_time over race_date."""
        try:
            # Try to use race_time if available
            if 'race_time' in race_row and pd.notna(race_row['race_time']):
                race_date = pd.to_datetime(race_row['race_date'])
                # Assume race_time is in format like "14:30" or similar
                race_time_str = str(race_row['race_time'])
                if ':' in race_time_str:
                    hour, minute = map(int, race_time_str.split(':')[:2])
                    return race_date.replace(hour=hour, minute=minute)
            
            # Fallback to race_date
            return pd.to_datetime(race_row['race_date'])
        except:
            # Last resort - use race_date as-is
            return pd.to_datetime(race_row['race_date'])
    
    def load_dog_historical_data(self, dog_name: str, target_timestamp: datetime, 
                                lookback_days: int = 365) -> pd.DataFrame:
        """Load historical race data for a dog, excluding target race."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                d.*,
                r.venue, r.grade, r.distance, r.track_condition, r.weather,
                r.temperature, r.humidity, r.wind_speed, r.field_size,
                r.race_date, r.race_time, r.winner_name, r.winner_odds, r.winner_margin,
                e.pir_rating, e.first_sectional, e.win_time, e.bonus_time
            FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            LEFT JOIN enhanced_expert_data e ON d.race_id = e.race_id 
                AND d.dog_clean_name = e.dog_clean_name
            WHERE d.dog_clean_name = ?
                AND r.race_date IS NOT NULL
                AND d.finish_position IS NOT NULL
            ORDER BY r.race_date DESC, r.race_time DESC
            """
            
            historical_data = pd.read_sql_query(query, conn, params=[dog_name])
            conn.close()
            
            if historical_data.empty:
                return pd.DataFrame()
            
            # Filter to only races before target timestamp
            historical_data['race_timestamp'] = historical_data.apply(
                lambda row: self.get_race_timestamp(row), axis=1
            )
            
            # Exclude races at or after target timestamp
            cutoff_date = target_timestamp - timedelta(days=lookback_days)
            historical_data = historical_data[
                (historical_data['race_timestamp'] < target_timestamp) &
                (historical_data['race_timestamp'] >= cutoff_date)
            ].copy()
            
            # Sort by timestamp (most recent first)
            historical_data = historical_data.sort_values('race_timestamp', ascending=False)
            
            logger.debug(f"Loaded {len(historical_data)} historical races for {dog_name}")
            return historical_data
        
        except Exception as e:
            logger.error(f"Error loading historical data for {dog_name}: {e}")
            return pd.DataFrame()
    
    def create_historical_features(self, historical_data: pd.DataFrame, 
                                 target_venue: str = None, target_grade: str = None) -> Dict[str, float]:
        """Create features from historical races with exponential decay weighting."""
        if historical_data.empty:
            return self._get_default_historical_features()
        
        features = {}
        
        # Convert finish positions to numeric
        historical_data['finish_position_numeric'] = pd.to_numeric(
            historical_data['finish_position'], errors='coerce'
        )
        historical_data['individual_time_numeric'] = pd.to_numeric(
            historical_data['individual_time'], errors='coerce'
        )
        
        # Create exponential decay weights (most recent races weighted more)
        num_races = len(historical_data)
        weights = np.array([self.decay_factor ** i for i in range(num_races)])
        weights = weights / weights.sum()  # Normalize weights
        
        # Basic performance metrics with decay weighting
        valid_positions = historical_data['finish_position_numeric'].dropna()
        if len(valid_positions) > 0:
            valid_weights = weights[:len(valid_positions)]
            valid_weights = valid_weights / valid_weights.sum()
            
            features['historical_avg_position'] = float(
                np.average(valid_positions, weights=valid_weights)
            )
            features['historical_best_position'] = float(valid_positions.min())
            features['historical_win_rate'] = float(
                np.average((valid_positions == 1).astype(float), weights=valid_weights)
            )
            features['historical_place_rate'] = float(
                np.average((valid_positions <= 3).astype(float), weights=valid_weights)
            )
        
        # Recent form trend (improvement/decline over last 5-10 races)
        recent_positions = valid_positions.head(min(10, len(valid_positions)))
        if len(recent_positions) >= 3:
            # Linear regression slope (negative = improving)
            x = np.arange(len(recent_positions))
            slope = np.polyfit(x, recent_positions, 1)[0]
            features['historical_form_trend'] = float(-slope)  # Negative slope = improving
        else:
            features['historical_form_trend'] = 0.0
        
        # Time-based performance
        valid_times = historical_data['individual_time_numeric'].dropna()
        if len(valid_times) > 0:
            time_weights = weights[:len(valid_times)]
            time_weights = time_weights / time_weights.sum()
            
            features['historical_avg_time'] = float(
                np.average(valid_times, weights=time_weights)
            )
            features['historical_best_time'] = float(valid_times.min())
            features['historical_time_consistency'] = float(valid_times.std())
        
        # Venue-specific performance
        if target_venue:
            venue_races = historical_data[historical_data['venue'] == target_venue]
            if len(venue_races) > 0:
                venue_positions = pd.to_numeric(venue_races['finish_position'], errors='coerce').dropna()
                if len(venue_positions) > 0:
                    features['venue_specific_avg_position'] = float(venue_positions.mean())
                    features['venue_specific_win_rate'] = float((venue_positions == 1).mean())
                    features['venue_experience'] = len(venue_positions)
                    features['venue_best_position'] = float(venue_positions.min())
        
        # Grade-specific performance
        if target_grade:
            grade_races = historical_data[historical_data['grade'] == target_grade]
            if len(grade_races) > 0:
                grade_positions = pd.to_numeric(grade_races['finish_position'], errors='coerce').dropna()
                if len(grade_positions) > 0:
                    features['grade_specific_avg_position'] = float(grade_positions.mean())
                    features['grade_specific_win_rate'] = float((grade_positions == 1).mean())
                    features['grade_experience'] = len(grade_positions)
        
        # Temporal features
        if len(historical_data) > 0:
            # Days since last race
            last_race_timestamp = historical_data['race_timestamp'].iloc[0]
            days_since_last = (datetime.now() - last_race_timestamp).days
            features['days_since_last_race'] = float(days_since_last)
            
            # Race frequency (races per month)
            date_span = (historical_data['race_timestamp'].iloc[0] - 
                        historical_data['race_timestamp'].iloc[-1]).days
            if date_span > 0:
                features['race_frequency'] = float(len(historical_data) * 30 / date_span)
            else:
                features['race_frequency'] = 1.0
        
        # Distance-specific performance (if distance info available)
        if 'distance' in historical_data.columns:
            distance_performance = {}
            for distance in historical_data['distance'].unique():
                if pd.notna(distance):
                    dist_races = historical_data[historical_data['distance'] == distance]
                    if len(dist_races) > 0:
                        dist_positions = pd.to_numeric(dist_races['finish_position'], errors='coerce').dropna()
                        if len(dist_positions) > 0:
                            distance_performance[distance] = {
                                'avg_position': float(dist_positions.mean()),
                                'win_rate': float((dist_positions == 1).mean()),
                                'races': len(dist_positions)
                            }
            
            # Add best distance performance
            if distance_performance:
                best_distance = min(distance_performance.items(), 
                                  key=lambda x: x[1]['avg_position'])
                features['best_distance_avg_position'] = best_distance[1]['avg_position']
                features['best_distance_win_rate'] = best_distance[1]['win_rate']
        
        return features
    
    def _get_default_historical_features(self) -> Dict[str, float]:
        """Default features for dogs with no historical data."""
        return {
            'historical_avg_position': 4.5,
            'historical_best_position': 4.0,
            'historical_win_rate': 0.125,
            'historical_place_rate': 0.375,
            'historical_form_trend': 0.0,
            'historical_avg_time': 30.0,
            'historical_best_time': 29.0,
            'historical_time_consistency': 2.0,
            'venue_specific_avg_position': 4.5,
            'venue_specific_win_rate': 0.125,
            'venue_experience': 0,
            'venue_best_position': 4.0,
            'grade_specific_avg_position': 4.5,
            'grade_specific_win_rate': 0.125,
            'grade_experience': 0,
            'days_since_last_race': 30.0,
            'race_frequency': 2.0,
            'best_distance_avg_position': 4.5,
            'best_distance_win_rate': 0.125
        }
    
    def build_features_for_race(self, race_data: pd.DataFrame, target_race_id: str) -> pd.DataFrame:
        """Build leakage-safe features for all dogs in a race."""
        logger.info(f"Building leakage-safe features for race {target_race_id}")
        
        # Get target race timestamp
        target_race_info = race_data.iloc[0]
        target_timestamp = self.get_race_timestamp(target_race_info)
        
        feature_rows = []
        
        for _, dog_row in race_data.iterrows():
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
                        f"found in target race features for {dog_row['dog_clean_name']} "
                        f"in race {target_race_id}"
                    )
            
            # Load historical data and create historical features
            historical_data = self.load_dog_historical_data(
                dog_row['dog_clean_name'], 
                target_timestamp
            )
            
            historical_features = self.create_historical_features(
                historical_data,
                target_venue=dog_row.get('venue'),
                target_grade=dog_row.get('grade')
            )
            
            features.update(historical_features)
            
            # Add metadata
            features['race_id'] = target_race_id
            features['dog_clean_name'] = dog_row['dog_clean_name']
            features['target_timestamp'] = target_timestamp
            
            # Add target if available (for training)
            if 'finish_position' in dog_row and pd.notna(dog_row['finish_position']):
                try:
                    features['target'] = 1 if int(dog_row['finish_position']) == 1 else 0
                except:
                    features['target'] = 0
            
            feature_rows.append(features)
        
        result_df = pd.DataFrame(feature_rows)
        logger.info(f"Built features for {len(result_df)} dogs in race {target_race_id}")
        
        return result_df
    
    def validate_temporal_integrity(self, features_df: pd.DataFrame, 
                                  race_data: pd.DataFrame) -> bool:
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
        historical_feature_prefixes = ['historical_', 'venue_specific_', 'grade_specific_']
        historical_features = [
            col for col in features_df.columns 
            if any(col.startswith(prefix) for prefix in historical_feature_prefixes)
        ]
        
        if not historical_features:
            logger.warning("No historical features found - this may reduce prediction quality")
        else:
            logger.info(f"Found {len(historical_features)} historical features")
        
        # Check 3: All dogs have features
        if len(features_df) != len(race_data):
            raise AssertionError(
                f"Feature count mismatch: {len(features_df)} features vs {len(race_data)} dogs"
            )
        
        logger.info("✅ Temporal integrity validation passed")
        return True


def create_temporal_assertion_hook():
    """Create assertion hook to be used in prediction pipeline."""
    def assert_no_target_leakage(features: Dict[str, Any], race_id: str, dog_name: str):
        """Assert that no post-race features from target race are present."""
        post_race_features = {
            'finish_position', 'individual_time', 'sectional_1st', 'sectional_2nd',
            'sectional_3rd', 'margin', 'beaten_margin', 'winning_time',
            'scraped_finish_position', 'scraped_raw_result', 'winner_name',
            'winner_odds', 'winner_margin', 'pir_rating', 'first_sectional',
            'win_time', 'bonus_time'
        }
        
        # Disabled features (odds-related)
        disabled_features = {
            'odds', 'SP', 'starting_price', 'odds_decimal', 'market_odds',
            'implied_probability', 'odds_rank'
        }
        
        # Check for post-race leakage
        leakage_features = set(features.keys()).intersection(post_race_features)
        if leakage_features:
            raise AssertionError(
                f"TEMPORAL LEAKAGE DETECTED at predict time: "
                f"Dog {dog_name} in race {race_id} has post-race features: {leakage_features}"
            )
        
        # Check for disabled features
        disabled_found = set(features.keys()).intersection(disabled_features)
        if disabled_found:
            raise AssertionError(
                f"TEMPORAL LEAKAGE DETECTED at predict time: "
                f"Dog {dog_name} in race {race_id} has disabled features: {disabled_found}"
            )
    
    return assert_no_target_leakage


if __name__ == "__main__":
    # Test the temporal feature builder
    builder = TemporalFeatureBuilder()
    
    # This would normally be called with actual race data
    print("✅ Temporal Feature Builder initialized")
    print("🛡️ Temporal leakage guards active")
    print("📊 Exponential decay weighting enabled")
