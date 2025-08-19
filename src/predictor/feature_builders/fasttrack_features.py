"""
Feature engineering functions for FastTrack data.

This module provides functions to build advanced features from the enriched
FastTrack dataset. These features are designed to provide a more granular and
accurate view of a greyhound's performance characteristics.
"""

import pandas as pd
from sqlalchemy.orm import Session


# Placeholder for database models. In a real implementation, these would be imported.
class DogPerformanceFtExtra:
    pass


class RaceFtExtra:
    pass


def get_last_n_races(session: Session, dog_id: int, n: int = 10):
    """
    Retrieves the last N performance records for a given dog.

    Args:
        session: The database session.
        dog_id: The ID of the dog.
        n: The number of recent races to retrieve.

    Returns:
        A list of DogPerformanceFtExtra objects.
    """
    # Placeholder: In a real implementation, this would query the database.
    # q = session.query(DogPerformanceFtExtra).filter_by(dog_id=dog_id).order_by(desc(Race.race_date)).limit(n)
    # return q.all()
    print(f"INFO: Retrieving last {n} races for dog_id {dog_id} (placeholder).")
    return []


def calculate_sectional_features(performances: list) -> dict:
    """
    Calculates features based on sectional times.

    Args:
        performances: A list of a dog's recent performance records.

    Returns:
        A dictionary of sectional time features.
    """
    if not performances:
        return {
            "avg_split_1_time": None,
            "avg_run_home_time": None,
            "sectional_consistency": None,
        }

    # Placeholder for actual calculation
    # split_1_times = [p.split_1_time for p in performances if p.split_1_time]
    # ... more calculations

    return {
        "avg_split_1_time": 0.0,  # Placeholder
        "avg_run_home_time": 0.0,  # Placeholder
        "sectional_consistency": 0.0,  # Placeholder
    }


def calculate_performance_metrics(performances: list) -> dict:
    """
    Calculates advanced performance metrics.

    Args:
        performances: A list of a dog's recent performance records.

    Returns:
        A dictionary of advanced performance metrics.
    """
    if not performances:
        return {
            "avg_beaten_margin": None,
            "avg_prize_money": None,
        }

    # Placeholder for actual calculation
    # beaten_margins = [p.beaten_margin for p in performances if p.beaten_margin is not None]
    # ... more calculations

    return {
        "avg_beaten_margin": 0.0,  # Placeholder
        "avg_prize_money": 0.0,  # Placeholder
    }


def calculate_normalized_time_features(session: Session, performances: list) -> dict:
    """
    Calculates weather and track-adjusted time features.

    Args:
        session: The database session.
        performances: A list of a dog's recent performance records.

    Returns:
        A dictionary of normalized time features.
    """
    if not performances:
        return {"avg_normalized_time": None}

    # Placeholder for actual calculation
    # for p in performances:
    #   race_extra = session.query(RaceFtExtra).filter_by(race_id=p.race_id).first()
    #   track_rating = race_extra.track_rating
    #   ... apply adjustment to p.race_time

    return {"avg_normalized_time": 0.0}  # Placeholder


def build_fasttrack_features(session: Session, dog_id: int) -> dict:
    """
    Builds all FastTrack features for a given dog.

    Args:
        session: The database session.
        dog_id: The ID of the dog.

    Returns:
        A dictionary containing all the new features.
    """
    last_10_races = get_last_n_races(session, dog_id, n=10)

    sectional_features = calculate_sectional_features(last_10_races)
    performance_metrics = calculate_performance_metrics(last_10_races)
    normalized_time_features = calculate_normalized_time_features(
        session, last_10_races
    )

    features = {
        **sectional_features,
        **performance_metrics,
        **normalized_time_features,
        "fasttrack_data_available": True if last_10_races else False,
    }

    return features
