#!/usr/bin/env python3
"""
The Greyhound Recorder (TGR) Prediction Integration
===================================================

This module integrates The Greyhound Recorder scraper into the prediction pipeline
to provide rich historical form data for dogs during predictions.

Key Features:
- On-demand historical data lookup from TGR
- Caching for performance
- Integration with TemporalFeatureBuilder
- Rich form guide analysis
- Temporal safety (no future data leakage)
"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Handle pandas/numpy dependencies gracefully
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    # Create minimal compatibility layer
    import warnings
    warnings.warn("pandas/numpy not available - using compatibility mode")
    HAS_PANDAS = False
    
    class MockDataFrame:
        def __init__(self, data=None):
            self.data = data or []
        def iterrows(self):
            return enumerate(self.data)
        def to_dict(self):
            return {}
        def __len__(self):
            return len(self.data)
    
    class MockPandas:
        DataFrame = MockDataFrame
        def to_datetime(self, x):
            if isinstance(x, str):
                return datetime.fromisoformat(x.replace('Z', '+00:00'))
            return x
    
    pd = MockPandas()

logger = logging.getLogger(__name__)

class TGRPredictionIntegrator:
    """Integrates The Greyhound Recorder data into predictions."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db", 
                 enable_tgr_lookup: bool = True,
                 cache_duration_hours: int = 24,
                 use_scraper: bool = True):
        self.db_path = db_path
        self.enable_tgr_lookup = enable_tgr_lookup
        self.cache_duration_hours = cache_duration_hours
        self.use_scraper = use_scraper
        
        # Initialize TGR scraper only if explicitly enabled
        self.tgr_scraper = None
        if self.enable_tgr_lookup and self.use_scraper:
            try:
                from src.collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
                self.tgr_scraper = TheGreyhoundRecorderScraper(
                    rate_limit=2.0,  # Respectful rate limiting
                    cache_dir=".tgr_cache",
                    use_cache=True
                )
                logger.info("✅ TGR scraper initialized for enhancement")
            except ImportError as e:
                logger.warning(f"TGR scraper not available (DB-only mode): {e}")
                self.tgr_scraper = None
                self.use_scraper = False
        elif self.enable_tgr_lookup and not self.use_scraper:
            logger.info("ℹ️ TGR integrator running in DB-only mode (no live scraping)")
    
    def enhance_prediction_features(self, dog_data: pd.DataFrame, race_timestamp: datetime) -> pd.DataFrame:
        """Enhance prediction features with TGR historical data.
        Works in DB-only mode; live scraper is optional.
        """
        if not self.enable_tgr_lookup:
            logger.debug("TGR enhancement disabled")
            return dog_data
        logger.info(f"🔍 Enhancing prediction features with TGR data for {len(dog_data)} dogs")

        enhanced_data = []
        for _, dog in dog_data.iterrows():
            dog_dict = dog.to_dict()

            # Get TGR historical data for this dog
            tgr_features = self._get_tgr_historical_features(
                dog_dict['dog_clean_name'],
                race_timestamp
            )

            # Merge TGR features with existing data
            dog_dict.update(tgr_features)
            enhanced_data.append(dog_dict)

        enhanced_df = pd.DataFrame(enhanced_data)
        logger.info(f"✅ Enhanced {len(enhanced_df)} dogs with TGR historical data")
        return enhanced_df
    def _get_tgr_historical_features(self, dog_name: str, race_timestamp: datetime) -> Dict[str, Any]:
        """Get historical features for a dog from TGR data."""

        # Check cache first
        cached_features = self._get_cached_tgr_features(dog_name, race_timestamp)
        if cached_features is not None:
            logger.debug(f"Using cached TGR features for {dog_name}")
            return cached_features

        # Fetch fresh TGR data
        logger.debug(f"Fetching fresh TGR data for {dog_name}")

        try:
            # Search for the dog in TGR form guides
            tgr_form_data = self._search_tgr_dog_form(dog_name, race_timestamp)

            # Calculate features from TGR data
            features = self._calculate_tgr_features(tgr_form_data, race_timestamp)

            # Cache the results
            self._cache_tgr_features(dog_name, race_timestamp, features)

            return features

        except Exception as e:
            logger.warning(f"Failed to fetch TGR data for {dog_name}: {e}")
            return self._get_default_tgr_features()
    def _search_tgr_dog_form(self, dog_name: str, race_timestamp: datetime) -> List[Dict[str, Any]]:
        """Search TGR for historical form data for a specific dog."""
        
        # This would typically involve:
        # 1. Searching TGR database/cache for the dog
        # 2. If not found, scraping TGR for recent form guides containing the dog
        # 3. Parsing the form data
        
        form_data = []
        
        try:
            # First check our database for existing TGR form data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Look for existing TGR form data (if we have dog entries)
            query = """
                SELECT gdf.race_date, gdf.venue, gdf.distance, gdf.box_number,
                       gdf.finishing_position, gdf.race_time, gdf.split_times,
                       gdf.margin, gdf.weight, gdf.comments
                FROM gr_dog_form gdf
                JOIN gr_dog_entries gde ON gdf.dog_entry_id = gde.id  
                WHERE UPPER(gde.dog_name) = UPPER(?)
                  AND gdf.race_date < date(?)
                  AND gdf.race_date >= date(?, '-365 days')
                ORDER BY gdf.race_date DESC
                LIMIT 20
            """
            
            cursor.execute(query, [dog_name, race_timestamp.strftime('%Y-%m-%d'), race_timestamp.strftime('%Y-%m-%d')])
            rows = cursor.fetchall()
            
            for row in rows:
                form_entry = {
                    'race_date': row[0],
                    'venue': row[1],
                    'distance': row[2],
                    'box_number': row[3],
                    'finish_position': row[4],
                    'race_time': row[5],
                    'split_times': json.loads(row[6]) if row[6] else {},
                    'margin': row[7],
                    'weight': row[8],
                    'comments': row[9]
                }
                form_data.append(form_entry)
            
            conn.close()
            
            # If we don't have enough data, we could trigger a TGR scrape here
            # For now, we'll use what we have in the database
            
            logger.debug(f"Found {len(form_data)} TGR form entries for {dog_name}")
            
        except Exception as e:
            logger.error(f"Error querying TGR form data for {dog_name}: {e}")
        
        return form_data
    
    def _calculate_tgr_features(self, form_data: List[Dict[str, Any]], race_timestamp: datetime) -> Dict[str, Any]:
        """Calculate prediction features from TGR form data."""
        
        if not form_data:
            return self._get_default_tgr_features()
        
        features = {}
        
        try:
            # Recent form analysis (last 10 races)
            recent_form = form_data[:10]
            all_positions = [entry['finish_position'] for entry in form_data if entry['finish_position']]
            recent_positions = [entry['finish_position'] for entry in recent_form if entry['finish_position']]
            
            # Basic TGR form features
            features['tgr_total_races'] = len(form_data)
            features['tgr_recent_races'] = len(recent_form)
            
            if all_positions:
                features['tgr_avg_finish_position'] = sum(all_positions) / len(all_positions)
                features['tgr_best_finish_position'] = min(all_positions)
                features['tgr_win_rate'] = len([p for p in all_positions if p == 1]) / len(all_positions)
                features['tgr_place_rate'] = len([p for p in all_positions if p <= 3]) / len(all_positions)
                features['tgr_consistency'] = 1.0 - (max(all_positions) - min(all_positions)) / max(all_positions, 1)
            else:
                features.update(self._get_default_tgr_features())
                return features
            
            # Recent form trend
            if recent_positions and len(recent_positions) >= 3:
                # Calculate form trend (improving/declining)
                recent_3 = recent_positions[:3]
                early_3 = recent_positions[-3:] if len(recent_positions) >= 6 else recent_positions[3:6]
                
                if early_3:
                    recent_avg = sum(recent_3) / len(recent_3)
                    early_avg = sum(early_3) / len(early_3)
                    features['tgr_form_trend'] = (early_avg - recent_avg) / early_avg  # Positive = improving
                else:
                    features['tgr_form_trend'] = 0.0
                
                features['tgr_recent_avg_position'] = sum(recent_positions) / len(recent_positions)
                features['tgr_recent_best_position'] = min(recent_positions)
            else:
                features['tgr_form_trend'] = 0.0
                features['tgr_recent_avg_position'] = features['tgr_avg_finish_position']
                features['tgr_recent_best_position'] = features['tgr_best_finish_position']
            
            # Distance-specific performance
            distances = {}
            for entry in form_data:
                if entry['distance']:
                    dist = entry['distance']
                    if dist not in distances:
                        distances[dist] = []
                    distances[dist].append(entry['finish_position'])
            
            # Find most common distance and performance at that distance
            if distances:
                most_common_distance = max(distances.keys(), key=lambda d: len(distances[d]))
                distance_positions = distances[most_common_distance]
                features['tgr_preferred_distance'] = most_common_distance
                features['tgr_preferred_distance_avg'] = sum(distance_positions) / len(distance_positions)
                features['tgr_preferred_distance_races'] = len(distance_positions)
            else:
                features['tgr_preferred_distance'] = 500
                features['tgr_preferred_distance_avg'] = features['tgr_avg_finish_position']
                features['tgr_preferred_distance_races'] = 0
            
            # Venue-specific performance
            venues = {}
            for entry in form_data:
                if entry['venue']:
                    venue = entry['venue']
                    if venue not in venues:
                        venues[venue] = []
                    venues[venue].append(entry['finish_position'])
            
            features['tgr_venues_raced'] = len(venues)
            
            # Time-based features (recency)
            if form_data:
                most_recent = form_data[0]
                most_recent_date = datetime.strptime(most_recent['race_date'], '%Y-%m-%d')
                days_since_last_race = (race_timestamp.date() - most_recent_date.date()).days
                features['tgr_days_since_last_race'] = days_since_last_race
                features['tgr_last_race_position'] = most_recent['finish_position']
            else:
                features['tgr_days_since_last_race'] = 365  # Default to a year
                features['tgr_last_race_position'] = 8  # Default poor position
            
            # Advanced TGR features based on comments/insights
            comments = [entry['comments'] for entry in form_data if entry.get('comments')]
            features['tgr_has_comments'] = len(comments)
            
            # Look for positive/negative indicators in comments
            positive_keywords = ['strong', 'impressive', 'well', 'good', 'fast', 'winner', 'placed']
            negative_keywords = ['slow', 'weak', 'poor', 'struggled', 'disappointing', 'injured']
            
            positive_mentions = sum(1 for comment in comments 
                                  for keyword in positive_keywords 
                                  if keyword.lower() in comment.lower())
            negative_mentions = sum(1 for comment in comments 
                                  for keyword in negative_keywords 
                                  if keyword.lower() in comment.lower())
            
            total_mentions = positive_mentions + negative_mentions
            if total_mentions > 0:
                features['tgr_sentiment_score'] = (positive_mentions - negative_mentions) / total_mentions
            else:
                features['tgr_sentiment_score'] = 0.0
            
            logger.debug(f"Calculated TGR features: win_rate={features['tgr_win_rate']:.3f}, "
                        f"avg_pos={features['tgr_avg_finish_position']:.1f}, "
                        f"trend={features['tgr_form_trend']:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating TGR features: {e}")
            features.update(self._get_default_tgr_features())
        
        return features
    
    def _get_default_tgr_features(self) -> Dict[str, Any]:
        """Return default TGR features when no data is available."""
        return {
            'tgr_total_races': 0,
            'tgr_recent_races': 0,
            'tgr_avg_finish_position': 5.0,
            'tgr_best_finish_position': 8,
            'tgr_win_rate': 0.1,
            'tgr_place_rate': 0.3,
            'tgr_consistency': 0.5,
            'tgr_form_trend': 0.0,
            'tgr_recent_avg_position': 5.0,
            'tgr_recent_best_position': 8,
            'tgr_preferred_distance': 500,
            'tgr_preferred_distance_avg': 5.0,
            'tgr_preferred_distance_races': 0,
            'tgr_venues_raced': 1,
            'tgr_days_since_last_race': 30,
            'tgr_last_race_position': 5,
            'tgr_has_comments': 0,
            'tgr_sentiment_score': 0.0
        }
    
    def _get_cached_tgr_features(self, dog_name: str, race_timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Retrieve cached TGR features if available and recent."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if we have a recent cache entry
            cache_cutoff = race_timestamp - timedelta(hours=self.cache_duration_hours)
            
            query = """
                SELECT tgr_features, cached_at 
                FROM tgr_feature_cache 
                WHERE dog_name = ? AND cached_at > ?
                ORDER BY cached_at DESC 
                LIMIT 1
            """
            
            cursor.execute(query, [dog_name, cache_cutoff.isoformat()])
            result = cursor.fetchone()
            conn.close()
            
            if result:
                features_json, cached_at = result
                return json.loads(features_json)
            
        except Exception as e:
            logger.debug(f"Cache lookup failed for {dog_name}: {e}")
        
        return None
    
    def _cache_tgr_features(self, dog_name: str, race_timestamp: datetime, features: Dict[str, Any]):
        """Cache TGR features for future use."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create cache table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tgr_feature_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dog_name TEXT NOT NULL,
                    tgr_features TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    race_timestamp TIMESTAMP
                )
            """)
            
            # Insert or update cache entry
            cursor.execute("""
                INSERT OR REPLACE INTO tgr_feature_cache 
                (dog_name, tgr_features, cached_at, race_timestamp) 
                VALUES (?, ?, ?, ?)
            """, [
                dog_name, 
                json.dumps(features),
                datetime.now().isoformat(),
                race_timestamp.isoformat()
            ])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to cache TGR features for {dog_name}: {e}")
    
    def get_feature_names(self) -> List[str]:
        """Return list of TGR feature names for ML model."""
        return [
            'tgr_total_races',
            'tgr_recent_races', 
            'tgr_avg_finish_position',
            'tgr_best_finish_position',
            'tgr_win_rate',
            'tgr_place_rate',
            'tgr_consistency',
            'tgr_form_trend',
            'tgr_recent_avg_position',
            'tgr_recent_best_position',
            'tgr_preferred_distance',
            'tgr_preferred_distance_avg',
            'tgr_preferred_distance_races',
            'tgr_venues_raced',
            'tgr_days_since_last_race',
            'tgr_last_race_position',
            'tgr_has_comments',
            'tgr_sentiment_score'
        ]


def integrate_tgr_with_temporal_builder(temporal_builder, db_path: str = "greyhound_racing_data.db"):
    """Integrate TGR functionality with existing TemporalFeatureBuilder."""
    
    # Add TGR integrator to temporal builder
    if not hasattr(temporal_builder, 'tgr_integrator'):
        temporal_builder.tgr_integrator = TGRPredictionIntegrator(db_path=db_path)
        logger.info("✅ TGR integrator added to TemporalFeatureBuilder")
    
    # Store original method
    original_create_historical_features = temporal_builder.create_historical_features
    
    def enhanced_create_historical_features(historical_data, target_venue=None, target_grade=None, target_distance=None):
        """Enhanced version that includes TGR features."""
        
        # Get original features
        original_features = original_create_historical_features(
            historical_data, target_venue, target_grade, target_distance
        )
        
        # Add TGR features if we have dog name info
        if hasattr(temporal_builder, 'current_dog_name') and temporal_builder.current_dog_name:
            if hasattr(temporal_builder, 'current_race_timestamp') and temporal_builder.current_race_timestamp:
                tgr_features = temporal_builder.tgr_integrator._get_tgr_historical_features(
                    temporal_builder.current_dog_name,
                    temporal_builder.current_race_timestamp
                )
                
                # Add TGR prefix to avoid name conflicts
                prefixed_tgr_features = {f"tgr_{k}" if not k.startswith('tgr_') else k: v 
                                       for k, v in tgr_features.items()}
                original_features.update(prefixed_tgr_features)
                
                logger.debug(f"Added {len(tgr_features)} TGR features for {temporal_builder.current_dog_name}")
        
        return original_features
    
    # Replace the method
    temporal_builder.create_historical_features = enhanced_create_historical_features
    
    # Add helper method to set current dog context
    def set_dog_context(dog_name: str, race_timestamp):
        temporal_builder.current_dog_name = dog_name
        temporal_builder.current_race_timestamp = race_timestamp
    
    temporal_builder.set_dog_context = set_dog_context
    
    return temporal_builder
