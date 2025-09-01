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

This package degrades gracefully in constrained environments. If optional
feature modules are unavailable, their symbols are set to None and omitted
from __all__ so importing code can handle their absence.
"""

from .feature_store import FeatureStore

# Optional feature groups
V3BoxPositionFeatures = None
V3CompetitionFeatures = None
V3DistanceStatsFeatures = None
V3RecentFormFeatures = None
V3TrainerFeatures = None
V3VenueAnalysisFeatures = None
V3WeatherTrackFeatures = None

try:
    from .v3_box_position import V3BoxPositionFeatures  # type: ignore
except Exception:
    V3BoxPositionFeatures = None
try:
    from .v3_competition import V3CompetitionFeatures  # type: ignore
except Exception:
    V3CompetitionFeatures = None
try:
    from .v3_distance_stats import V3DistanceStatsFeatures  # type: ignore
except Exception:
    V3DistanceStatsFeatures = None
try:
    from .v3_recent_form import V3RecentFormFeatures  # type: ignore
except Exception:
    V3RecentFormFeatures = None
try:
    from .v3_trainer import V3TrainerFeatures  # type: ignore
except Exception:
    V3TrainerFeatures = None
try:
    from .v3_venue_analysis import V3VenueAnalysisFeatures  # type: ignore
except Exception:
    V3VenueAnalysisFeatures = None
try:
    from .v3_weather_track import V3WeatherTrackFeatures  # type: ignore
except Exception:
    V3WeatherTrackFeatures = None

# Build __all__ with available symbols only
__all__ = [name for name, val in [
    ("V3DistanceStatsFeatures", V3DistanceStatsFeatures),
    ("V3RecentFormFeatures", V3RecentFormFeatures),
    ("V3VenueAnalysisFeatures", V3VenueAnalysisFeatures),
    ("V3BoxPositionFeatures", V3BoxPositionFeatures),
    ("V3CompetitionFeatures", V3CompetitionFeatures),
    ("V3WeatherTrackFeatures", V3WeatherTrackFeatures),
    ("V3TrainerFeatures", V3TrainerFeatures),
    ("FeatureStore", FeatureStore),
] if val is not None]

__version__ = "3.0.0"
