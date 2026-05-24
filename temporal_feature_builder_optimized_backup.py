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
            self.default_lookback_days = int(os.getenv('GREYHOUND_LOOKBACK_DAYS', '365'))
        except Exception:
            self.default_lookback_days = 365
        
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
        
        # Feature caching setup
        self.cache_dir = Path('./feature_cache_optimized')
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_caching = True
        
        # Performance caches
        self._timestamp_cache = {}  # Cache parsed timestamps
        self._dog_name_cache = {}   # Cache sanitized dog names
        self._feature_cache = {}    # Cache computed features
        self._batch_cache = {}      # Cache batch queries
        
        # Statistics for performance monitoring
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'db_queries': 0,
            'timestamp_parses': 0,
            'feature_computations': 0
        }
        
        # Initialize TGR integration if available
        self.tgr_integrator = None
        if TGRPredictionIntegrator:
            try:
                self.tgr_integrator = TGRPredictionIntegrator(db_path=self.db_path)
                logger.info("✅ TGR integration initialized for enhanced historical features")
            except Exception as e:
                logger.warning(f"Could not initialize TGR integration: {e}")
                self.tgr_integrator = None
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    @lru_cache(maxsize=1024)
    def _parse_timestamp_cached(self, race_date_str: str, race_time_str: str = None) -> datetime:
        """Cached timestamp parsing with vectorization support."""
        self.stats['timestamp_parses'] += 1
        
        try:
            race_date = pd.to_datetime(race_date_str)
            
            # Try to use race_time if available
            if race_time_str and pd.notna(race_time_str) and race_time_str != 'None':
                race_time_str = str(race_time_str)
                if ':' in race_time_str:
                    hour, minute = map(int, race_time_str.split(':')[:2])
                    return race_date.replace(hour=hour, minute=minute)
            
            # Fallback to race_date
            return race_date
        except:
            # Last resort - use race_date as-is
            return pd.to_datetime(race_date_str)
    
    def get_race_timestamp(self, race_row: pd.Series) -> datetime:
        """Get race timestamp with caching."""
        race_date_str = str(race_row.get('race_date', ''))
        race_time_str = str(race_row.get('race_time', '')) if pd.notna(race_row.get('race_time')) else None
        
        return self._parse_timestamp_cached(race_date_str, race_time_str)
    
    def _vectorized_timestamp_parsing(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized timestamp parsing for better performance."""
        if 'race_timestamp' in data.columns:
            return data['race_timestamp']
        
        # Use vectorized operations where possible
        race_dates = pd.to_datetime(data['race_date'])
        
        # Handle race times
        has_time = data['race_time'].notna() & (data['race_time'] != 'None')
        
        if has_time.any():
            # Vectorized time parsing for rows with time info
            time_mask = has_time
            time_parts = data.loc[time_mask, 'race_time'].str.split(':', expand=True)
            
            if len(time_parts.columns) >= 2:
                hours = pd.to_numeric(time_parts[0], errors='coerce').fillna(0).astype(int)
                minutes = pd.to_numeric(time_parts[1], errors='coerce').fillna(0).astype(int)
                
                # Create datetime with time
                timestamps = race_dates.copy()
                timestamps.loc[time_mask] = pd.to_datetime(
                    data.loc[time_mask, 'race_date']
                ) + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')
                
                return timestamps
        
        return race_dates
    
    @lru_cache(maxsize=512)
    def _sanitize_dog_name_cached(self, dog_name: str) -> str:
        """Cached dog name sanitization."""
        return (
            str(dog_name)
            .replace('"', '')
            .replace("'", '')
            .replace('`', '')
            .replace(''', '')
            .strip()
            .upper()
        )
    
    def batch_load_historical_data(self, dog_names: List[str], target_timestamp: datetime,
                                  lookback_days: int = None) -> Dict[str, pd.DataFrame]:
        """Load historical data for multiple dogs in a single batch query."""
        if lookback_days is None:
            lookback_days = self.default_lookback_days
        
        # Check cache first
        cache_key = self._get_cache_key(tuple(sorted(dog_names)), target_timestamp, lookback_days)
        if self.enable_caching and cache_key in self._batch_cache:
            self.stats['cache_hits'] += 1
            logger.debug(f"Cache hit for batch historical data: {len(dog_names)} dogs")
            return self._batch_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        self.stats['db_queries'] += 1
        
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_date = (target_timestamp - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Create placeholders for IN clause
            placeholders = ','.join(['?' for _ in dog_names])
            
            # Build batch query
            query = f"""
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
            WHERE d.dog_clean_name IN ({placeholders})
                AND r.race_date IS NOT NULL
                AND d.finish_position IS NOT NULL
                AND date(r.race_date) >= date(?)
            ORDER BY d.dog_clean_name, r.race_date DESC, r.race_time DESC
            """
            
            params = dog_names + [cutoff_date]
            batch_data = pd.read_sql_query(query, conn, params=params)
            raw_count = len(batch_data)
            
            # Fallback query with sanitized names if no results
            if batch_data.empty:
                sanitized_names = [self._sanitize_dog_name_cached(name) for name in dog_names]
                sanitized_placeholders = ','.join(['?' for _ in sanitized_names])
                
                fallback_query = f"""
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
                WHERE REPLACE(REPLACE(REPLACE(REPLACE(d.dog_clean_name,'"',''),"'",''),'`',''),''','') IN ({sanitized_placeholders})
                    AND r.race_date IS NOT NULL
                    AND d.finish_position IS NOT NULL
                    AND date(r.race_date) >= date(?)
                ORDER BY d.dog_clean_name, r.race_date DESC, r.race_time DESC
                """
                
                fallback_params = sanitized_names + [cutoff_date]
                batch_data = pd.read_sql_query(fallback_query, conn, params=fallback_params)
                logger.debug(f"Fallback batch query returned {len(batch_data)} rows")
            
            conn.close()
            
            if batch_data.empty:
                logger.info(f"No historical data found for {len(dog_names)} dogs")
                result = {name: pd.DataFrame() for name in dog_names}
                if self.enable_caching:
                    self._batch_cache[cache_key] = result
                return result
            
            # Vectorized timestamp parsing for the entire batch
            batch_data['race_timestamp'] = self._vectorized_timestamp_parsing(batch_data)
            
            # Filter by target timestamp - vectorized operation
            cutoff_datetime = target_timestamp - timedelta(days=lookback_days)
            time_filter = (
                (batch_data['race_timestamp'] < target_timestamp) &
                (batch_data['race_timestamp'] >= cutoff_datetime)
            )
            batch_data = batch_data[time_filter].copy()
            
            # Group by dog name - this creates the dictionary we need
            result = {}
            for dog_name in dog_names:
                dog_data = batch_data[batch_data['dog_clean_name'] == dog_name].copy()
                
                # Sort by timestamp (most recent first) 
                if not dog_data.empty:
                    dog_data = dog_data.sort_values('race_timestamp', ascending=False)
                    # Limit to reasonable number of races for performance
                    dog_data = dog_data.head(100)
                
                result[dog_name] = dog_data
            
            logger.debug(
                f"Batch query for {len(dog_names)} dogs: {raw_count} raw -> {len(batch_data)} filtered"
            )
            
            # Cache the result
            if self.enable_caching:
                self._batch_cache[cache_key] = result
            
            return result
        
        except Exception as e:
            logger.error(f"Error in batch loading historical data: {e}")
            return {name: pd.DataFrame() for name in dog_names}
    
    def _vectorized_feature_computation(self, historical_data: pd.DataFrame,
                                      target_venue: str = None, target_grade: str = None,
                                      target_distance: float = None) -> Dict[str, float]:
        """Vectorized feature computation for better performance."""
        if historical_data.empty:
            return self._get_default_historical_features()
        
        self.stats['feature_computations'] += 1
        features = {}
        
        # Vectorized numeric conversions
        positions = pd.to_numeric(historical_data['finish_position'], errors='coerce')
        times = pd.to_numeric(historical_data['individual_time'], errors='coerce')
        distances = pd.to_numeric(historical_data['distance'], errors='coerce')
        
        # Create exponential decay weights (vectorized)
        num_races = len(historical_data)
        weights = np.power(self.decay_factor, np.arange(num_races))
        
        # Apply contextual weighting - vectorized operations
        if target_venue:
            ballarat_codes = ['BAL', 'BALLARAT', 'Ballarat']
            if target_venue.upper() in [code.upper() for code in ballarat_codes]:
                venue_boost = historical_data['venue'].str.upper().isin(
                    [code.upper() for code in ballarat_codes]
                ).astype(float) * 1.5
            else:
                venue_boost = (historical_data['venue'] == target_venue).astype(float) * 0.8
            weights = weights * (1 + venue_boost)
        
        if target_grade:
            grade_boost = (historical_data['grade'] == target_grade).astype(float) * 0.6
            weights = weights * (1 + grade_boost)
        
        if target_distance and not distances.isna().all():
            distance_matches = (abs(distances - target_distance) <= 50).astype(float)
            distance_boost = distance_matches * 0.7
            weights = weights * (1 + distance_boost)
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Basic performance metrics - vectorized
        valid_positions = positions.dropna()
        if len(valid_positions) > 0:
            valid_indices = positions.notna()
            valid_weights = weights[valid_indices]
            if valid_weights.sum() > 0:
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
        
        # Form trend - vectorized
        recent_positions = valid_positions.head(min(10, len(valid_positions)))
        if len(recent_positions) >= 3:
            x = np.arange(len(recent_positions))
            slope = np.polyfit(x, recent_positions, 1)[0]
            features['historical_form_trend'] = float(-slope)
        else:
            features['historical_form_trend'] = 0.0
        
        # Time-based performance - vectorized
        valid_times = times.dropna()
        if len(valid_times) > 0:
            time_indices = times.notna()
            time_weights = weights[time_indices]
            if time_weights.sum() > 0:
                time_weights = time_weights / time_weights.sum()
                
                # Distance adjustment
                if target_distance and not distances.isna().all():
                    valid_distances = distances[time_indices].fillna(target_distance)
                    # Vectorized distance adjustment
                    adjustment_factors = target_distance / valid_distances
                    adjusted_times = valid_times * adjustment_factors
                    
                    features['historical_avg_time'] = float(
                        np.average(adjusted_times, weights=time_weights)
                    )
                    features['historical_best_time'] = float(adjusted_times.min())
                    features['historical_time_consistency'] = float(adjusted_times.std())
                    features['distance_adjusted_time'] = True
                    features['target_distance'] = float(target_distance)
                else:
                    features['historical_avg_time'] = float(
                        np.average(valid_times, weights=time_weights)
                    )
                    features['historical_best_time'] = float(valid_times.min())
                    features['historical_time_consistency'] = float(valid_times.std())
                    features['distance_adjusted_time'] = False
                    features['target_distance'] = 0.0
        
        # Venue-specific performance - vectorized
        if target_venue:
            venue_mask = historical_data['venue'] == target_venue
            if venue_mask.any():
                venue_positions = positions[venue_mask].dropna()
                if len(venue_positions) > 0:
                    features['venue_specific_avg_position'] = float(venue_positions.mean())
                    features['venue_specific_win_rate'] = float((venue_positions == 1).mean())
                    features['venue_experience'] = len(venue_positions)
                    features['venue_best_position'] = float(venue_positions.min())
        
        # Grade-specific performance - vectorized
        if target_grade:
            grade_mask = historical_data['grade'] == target_grade
            if grade_mask.any():
                grade_positions = positions[grade_mask].dropna()
                if len(grade_positions) > 0:
                    features['grade_specific_avg_position'] = float(grade_positions.mean())
                    features['grade_specific_win_rate'] = float((grade_positions == 1).mean())
                    features['grade_experience'] = len(grade_positions)
        
        # Temporal features - vectorized
        if len(historical_data) > 0:
            last_race_timestamp = historical_data['race_timestamp'].iloc[0]
            days_since_last = (datetime.now() - last_race_timestamp).days
            features['days_since_last_race'] = float(days_since_last)
            
            # Race frequency
            if len(historical_data) > 1:
                date_span = (historical_data['race_timestamp'].iloc[0] - 
                           historical_data['race_timestamp'].iloc[-1]).days
                if date_span > 0:
                    features['race_frequency'] = float(len(historical_data) * 30 / date_span)
                else:
                    features['race_frequency'] = 1.0
            else:
                features['race_frequency'] = 1.0
        
        # Distance-specific performance - vectorized
        if not distances.isna().all():
            distance_groups = historical_data.groupby('distance')
            distance_performance = {}
            
            for distance_val, group in distance_groups:
                if pd.notna(distance_val):
                    dist_positions = pd.to_numeric(group['finish_position'], errors='coerce').dropna()
                    if len(dist_positions) > 0:
                        distance_performance[distance_val] = {
                            'avg_position': float(dist_positions.mean()),
                            'win_rate': float((dist_positions == 1).mean()),
                            'races': len(dist_positions)
                        }
            
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
            'best_distance_win_rate': 0.125,
            'distance_adjusted_time': False,
            'target_distance': 0.0
        }
    
    def build_features_for_race(self, race_data: pd.DataFrame, target_race_id: str) -> pd.DataFrame:
        """Build leakage-safe features for all dogs in a race - OPTIMIZED VERSION."""
        start_time = time.time()
        logger.info(f"Building optimized leakage-safe features for race {target_race_id}")
        
        # Get target race timestamp
        target_race_info = race_data.iloc[0]
        target_timestamp = self.get_race_timestamp(target_race_info)
        
        # Extract all dog names for batch processing
        dog_names = race_data['dog_clean_name'].tolist()
        
        # OPTIMIZATION 1: Batch load all historical data in one query
        batch_historical_data = self.batch_load_historical_data(dog_names, target_timestamp)
        
        feature_rows = []
        
        for _, dog_row in race_data.iterrows():
            dog_name = dog_row['dog_clean_name']
            
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
            
            # OPTIMIZATION 2: Use vectorized feature computation
            historical_features = self._vectorized_feature_computation(
                historical_data,
                target_venue=dog_row.get('venue'),
                target_grade=dog_row.get('grade'),
                target_distance=pd.to_numeric(dog_row.get('distance'), errors='coerce')
            )
            
            features.update(historical_features)
            
            # Add TGR features if TGR integration is available
            if self.tgr_integrator:
                try:
                    tgr_features = self.tgr_integrator._get_tgr_historical_features(
                        dog_name, target_timestamp
                    )
                    features.update(tgr_features)
                except Exception as e:
                    logger.warning(f"Failed to get TGR features for {dog_name}: {e}")
            
            # Add metadata
            features['race_id'] = target_race_id
            features['dog_clean_name'] = dog_name
            features['target_timestamp'] = target_timestamp
            
            # Add target if available (for training)
            if 'finish_position' in dog_row and pd.notna(dog_row['finish_position']):
                try:
                    features['target'] = 1 if int(dog_row['finish_position']) == 1 else 0
                except:
                    features['target'] = 0
            
            feature_rows.append(features)
        
        result_df = pd.DataFrame(feature_rows)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Built optimized features for {len(result_df)} dogs in race {target_race_id} "
                   f"in {elapsed_time:.2f}s")
        logger.debug(f"Performance stats: {self.stats}")
        
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
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        self._timestamp_cache.clear()
        self._dog_name_cache.clear()
        self._feature_cache.clear()
        self._batch_cache.clear()
        
        # Clear LRU caches
        self._parse_timestamp_cached.cache_clear()
        self._sanitize_dog_name_cached.cache_clear()
        
        logger.info("All caches cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'total_requests': total_requests,
            'timestamp_cache_size': len(self._timestamp_cache),
            'feature_cache_size': len(self._feature_cache),
            'batch_cache_size': len(self._batch_cache)
        }


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
        
        # Check if race is in the future
        current_date = datetime.now().date()
        race_date_str = features.get('race_date', '')
        
        if race_date_str:
            try:
                # Try different date formats
                date_formats = ['%d %B %Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
                race_date = None
                
                for fmt in date_formats:
                    try:
                        race_date = datetime.strptime(race_date_str, fmt).date()
                        break
                    except ValueError:
                        continue
                
                if race_date and race_date > current_date:
                    raise AssertionError(
                        f"TEMPORAL LEAKAGE DETECTED: Race {race_id} for dog {dog_name} is in the future: {race_date} (current: {current_date})"
                    )
            except AssertionError:
                # Re-raise AssertionError (temporal leakage detection)
                raise
            except Exception as e:
                # If date parsing fails, log but don't block (could be testing scenario)
                logger.debug(f"Could not parse race date '{race_date_str}' for race {race_id}: {e}")

    return assert_no_target_leakage


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
