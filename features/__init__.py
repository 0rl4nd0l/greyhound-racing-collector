"""
Features Module - Consolidated Feature Engineering for Greyhound Racing
====================================================================

This module consolidates all feature engineering functionality into versioned feature groups
for better maintainability, reusability and drift detection.

Feature Groups:
- v3_distance_stats: Distance-based performance statistics and trends
- v3_recent_form: Recent performance and form analysis
- v3_venue_analysis: Venue-specific performance patterns
- v3_box_position: Box position and starting advantages
- v3_competition: Competition level and field analysis
- v3_weather_track: Weather and track condition effects
- v3_trainer: Trainer and ownership effects
"""

from .v3_distance_stats import V3DistanceStatsFeatures
from .v3_recent_form import V3RecentFormFeatures
from .v3_venue_analysis import V3VenueAnalysisFeatures
from .v3_box_position import V3BoxPositionFeatures
from .v3_competition import V3CompetitionFeatures
from .v3_weather_track import V3WeatherTrackFeatures
from .v3_trainer import V3TrainerFeatures
from .feature_store import FeatureStore

__all__ = [
    'V3DistanceStatsFeatures',
    'V3RecentFormFeatures', 
    'V3VenueAnalysisFeatures',
    'V3BoxPositionFeatures',
    'V3CompetitionFeatures',
    'V3WeatherTrackFeatures',
    'V3TrainerFeatures',
    'FeatureStore'
]

__version__ = "3.0.0"
