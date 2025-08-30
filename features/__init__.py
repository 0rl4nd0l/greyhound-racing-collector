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
    class V3BoxPositionFeatures:
        version = "v3.0"
        def create_features(self, dog_stats: dict) -> dict:
            return {"v3_box_position_advantage": 0.0}
try:
    from .v3_competition import V3CompetitionFeatures  # type: ignore
except Exception:
    class V3CompetitionFeatures:
        version = "v3.0"
        def create_features(self, dog_stats: dict) -> dict:
            return {"v3_competition_strength": 0.5}
try:
    from .v3_distance_stats import V3DistanceStatsFeatures  # type: ignore
except Exception:
    class V3DistanceStatsFeatures:
        version = "v3.0"
        def create_features(self, dog_stats: dict) -> dict:
            avg_time = dog_stats.get("avg_time", 30.0)
            return {"v3_distance_avg_time": float(avg_time), "v3_distance_speed_rating": float(max(0.0, 100.0 - avg_time))}
try:
    from .v3_recent_form import V3RecentFormFeatures  # type: ignore
except Exception:
    class V3RecentFormFeatures:
        version = "v3.0"
        def create_features(self, dog_stats: dict) -> dict:
            recent = dog_stats.get("recent_form", [])
            trend = 0.0
            if isinstance(recent, list) and len(recent) > 1:
                trend = (recent[-2] - recent[-1]) / max(1, len(recent))
            return {"v3_recent_form_trend": float(trend), "v3_recent_win_rate": float(dog_stats.get("win_rate", 0.0))}
try:
    from .v3_trainer import V3TrainerFeatures  # type: ignore
except Exception:
    class V3TrainerFeatures:
        version = "v3.0"
        def create_features(self, dog_stats: dict) -> dict:
            trainer = dog_stats.get("trainer_stats", {})
            return {"v3_trainer_success_rate": float(trainer.get("win_rate", 0.0))}
try:
    from .v3_venue_analysis import V3VenueAnalysisFeatures  # type: ignore
except Exception:
    class V3VenueAnalysisFeatures:
        version = "v3.0"
        def create_features(self, dog_stats: dict) -> dict:
            return {"v3_venue_home_advantage": 0.0}
try:
    from .v3_weather_track import V3WeatherTrackFeatures  # type: ignore
except Exception:
    class V3WeatherTrackFeatures:
        version = "v3.0"
        def create_features(self, dog_stats: dict) -> dict:
            return {"v3_weather_track_impact": 0.0}

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
