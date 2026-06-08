import pandas as pd

from temporal_feature_builder import TemporalFeatureBuilder
from temporal_feature_builder_optimized import OptimizedTemporalFeatureBuilder


class EmptyHistoryTemporalBuilder(TemporalFeatureBuilder):
    def load_dog_historical_data(self, dog_name, target_timestamp, lookback_days=None):
        return pd.DataFrame()


class EmptyHistoryOptimizedBuilder(OptimizedTemporalFeatureBuilder):
    def batch_load_historical_data(self, dog_names, target_timestamp, lookback_days=None):
        return {dog_name: pd.DataFrame() for dog_name in dog_names}


def _race_row(distance_source):
    return pd.DataFrame(
        [
            {
                "dog_clean_name": "ALPHA",
                "box_number": 1,
                "weight": 30.0,
                "trainer_name": "Trainer",
                "venue": "TEST",
                "grade": "G5",
                "distance": "525m",
                "distance_source": distance_source,
                "track_condition": "Good",
                "weather": "Fine",
                "temperature": 20.0,
                "humidity": 60.0,
                "wind_speed": 10.0,
                "field_size": 1,
                "race_date": "2026-06-03",
                "race_time": "12:30",
            }
        ]
    )


def test_safe_target_distance_survives_when_db_history_is_empty():
    builder = EmptyHistoryTemporalBuilder(":memory:")

    features = builder.build_features_for_race(
        _race_row("canonical_pre_race_page"), "Race 1 - TEST - 2026-06-03"
    )

    assert features.loc[0, "target_distance"] == 525.0


def test_default_target_distance_does_not_get_promoted_to_model_feature():
    builder = EmptyHistoryTemporalBuilder(":memory:")

    features = builder.build_features_for_race(
        _race_row("default_missing_target"), "Race 1 - TEST - 2026-06-03"
    )

    assert "target_distance" not in features.columns


def test_optimized_builder_preserves_safe_target_distance_when_history_is_empty():
    builder = EmptyHistoryOptimizedBuilder(":memory:")

    features = builder.build_features_for_race(
        _race_row("canonical_pre_race_page"), "Race 1 - TEST - 2026-06-03"
    )

    assert features.loc[0, "target_distance"] == 525.0


def test_optimized_builder_rejects_default_target_distance_when_history_is_empty():
    builder = EmptyHistoryOptimizedBuilder(":memory:")

    features = builder.build_features_for_race(
        _race_row("default_missing_target"), "Race 1 - TEST - 2026-06-03"
    )

    assert "target_distance" not in features.columns
