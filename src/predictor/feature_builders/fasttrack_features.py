"""
Feature engineering functions for FastTrack data.

This module contains disabled placeholders for FastTrack feature engineering.
All functions raise RuntimeError in production to avoid fabricated features.
"""

import logging

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Placeholder for database models. In a real implementation, these would be imported.
class DogPerformanceFtExtra:
    pass


class RaceFtExtra:
    pass


def get_last_n_races(session: Session, dog_id: int, n: int = 10):
    """Disabled placeholder. Implement DB-backed query before use."""
    logger.error(
        "FastTrack feature builder is not available in production (placeholder removed)"
    )
    raise RuntimeError("FastTrack features are not available in production")


def calculate_sectional_features(performances: list) -> dict:
    """Disabled placeholder to avoid fabricated features."""
    logger.error("calculate_sectional_features is not available (placeholder removed)")
    raise RuntimeError("FastTrack features are not available in production")


def calculate_performance_metrics(performances: list) -> dict:
    """Disabled placeholder to avoid fabricated features."""
    logger.error("calculate_performance_metrics is not available (placeholder removed)")
    raise RuntimeError("FastTrack features are not available in production")


def calculate_normalized_time_features(session: Session, performances: list) -> dict:
    """Disabled placeholder to avoid fabricated features."""
    logger.error(
        "calculate_normalized_time_features is not available (placeholder removed)"
    )
    raise RuntimeError("FastTrack features are not available in production")


def build_fasttrack_features(session: Session, dog_id: int) -> dict:
    """Disabled placeholder to avoid fabricated features."""
    logger.error("build_fasttrack_features is not available (placeholder removed)")
    raise RuntimeError("FastTrack features are not available in production")
