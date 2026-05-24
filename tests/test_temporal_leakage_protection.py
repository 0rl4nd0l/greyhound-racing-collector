#!/usr/bin/env python3
"""
Test Suite for Temporal Leakage Protection
==========================================

Comprehensive tests to validate:
1. Temporal leakage protection
2. Feature integrity
3. Time-ordered splits
4. Encoding and calibration
5. Group normalization
6. EV calculations
"""

import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_system_v4 import MLSystemV4
from temporal_feature_builder import (
    TemporalFeatureBuilder,
    create_temporal_assertion_hook,
)


class TestTemporalLeakageProtection:
    """Test temporal leakage protection mechanisms."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database with test data."""
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(temp_db.name)

        # Create test tables
        conn.execute(
            """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            grade TEXT,
            distance TEXT,
            track_condition TEXT,
            weather TEXT,
            temperature REAL,
            humidity REAL,
            wind_speed REAL,
            field_size INTEGER,
            race_date TEXT,
            race_time TEXT
        )
        """
        )

        conn.execute(
            """
        CREATE TABLE dog_race_data (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            weight REAL,
            starting_price REAL,
            trainer_name TEXT,
            finish_position INTEGER,
            individual_time REAL,
            sectional_1st REAL,
            margin REAL,
            FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
        )
        """
        )

        # Insert test data
        test_races = [
            (
                "race_001",
                "TRACK_A",
                "G5",
                "500m",
                "Good",
                "Fine",
                20.0,
                60.0,
                10.0,
                8,
                "2024-01-01",
                "14:30",
            ),
            (
                "race_002",
                "TRACK_A",
                "G5",
                "500m",
                "Good",
                "Fine",
                21.0,
                65.0,
                12.0,
                8,
                "2024-01-02",
                "15:00",
            ),
            (
                "race_003",
                "TRACK_B",
                "G4",
                "520m",
                "Slow",
                "Cloudy",
                18.0,
                70.0,
                15.0,
                8,
                "2024-01-03",
                "16:30",
            ),
        ]

        conn.executemany(
            """
        INSERT INTO race_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            test_races,
        )

        test_dogs = [
            # Race 1 (past race)
            (1, "race_001", "DOG_A", 1, 30.5, 2.50, "TRAINER_1", 1, 29.50, 5.1, 0.0),
            (2, "race_001", "DOG_B", 2, 31.0, 3.20, "TRAINER_2", 2, 29.75, 5.2, 0.25),
            (3, "race_001", "DOG_C", 3, 29.8, 4.10, "TRAINER_3", 3, 30.10, 5.3, 0.60),
            # Race 2 (past race)
            (4, "race_002", "DOG_A", 2, 30.3, 2.80, "TRAINER_1", 2, 29.65, 5.15, 0.15),
            (5, "race_002", "DOG_D", 1, 30.8, 2.10, "TRAINER_4", 1, 29.45, 5.05, 0.0),
            (6, "race_002", "DOG_B", 3, 31.2, 3.50, "TRAINER_2", 3, 30.05, 5.25, 0.60),
            # Race 3 (target race - for prediction)
            (
                7,
                "race_003",
                "DOG_A",
                1,
                30.4,
                2.60,
                "TRAINER_1",
                None,
                None,
                None,
                None,
            ),
            (
                8,
                "race_003",
                "DOG_E",
                2,
                29.9,
                3.80,
                "TRAINER_5",
                None,
                None,
                None,
                None,
            ),
        ]

        conn.executemany(
            """
        INSERT INTO dog_race_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            test_dogs,
        )

        conn.commit()
        conn.close()

        yield temp_db.name

        # Cleanup
        os.unlink(temp_db.name)

    def test_temporal_feature_builder_initialization(self, temp_db):
        """Test that TemporalFeatureBuilder initializes correctly."""
        builder = TemporalFeatureBuilder(temp_db)

        assert builder.db_path == temp_db
        assert len(builder.pre_race_features) > 0
        assert len(builder.post_race_features) > 0
        assert "finish_position" in builder.post_race_features
        assert "box_number" in builder.pre_race_features

    def test_no_post_race_features_in_target_race(self, temp_db):
        """UNIT TEST: Ensure target rows contain no post-race fields."""
        builder = TemporalFeatureBuilder(temp_db)

        # Get target race data (race_003)
        conn = sqlite3.connect(temp_db)
        target_race_data = pd.read_sql_query(
            """
        SELECT d.*, r.venue, r.grade, r.track_condition, r.weather, 
               r.temperature, r.humidity, r.wind_speed, r.field_size,
               r.race_date, r.race_time
        FROM dog_race_data d
        JOIN race_metadata r ON d.race_id = r.race_id
        WHERE d.race_id = 'race_003'
        """,
            conn,
        )
        conn.close()

        # Build features for target race
        features = builder.build_features_for_race(target_race_data, "race_003")

        # Assert no post-race features in target features
        feature_columns = set(features.columns)
        leakage_features = feature_columns.intersection(builder.post_race_features)

        assert (
            len(leakage_features) == 0
        ), f"Found post-race features in target: {leakage_features}"

    def test_historical_features_present(self, temp_db):
        """UNIT TEST: Historical aggregates should be present."""
        builder = TemporalFeatureBuilder(temp_db)

        # Get target race data
        conn = sqlite3.connect(temp_db)
        target_race_data = pd.read_sql_query(
            """
        SELECT d.*, r.venue, r.grade, r.track_condition, r.weather, 
               r.temperature, r.humidity, r.wind_speed, r.field_size,
               r.race_date, r.race_time
        FROM dog_race_data d
        JOIN race_metadata r ON d.race_id = r.race_id
        WHERE d.race_id = 'race_003'
        """,
            conn,
        )
        conn.close()

        # Build features
        features = builder.build_features_for_race(target_race_data, "race_003")

        # Check for historical features
        historical_features = [
            col for col in features.columns if col.startswith("historical_")
        ]
        assert len(historical_features) > 0, "No historical features found"

        # Check that DOG_A has historical data (from races 001 and 002)
        dog_a_features = features[features["dog_clean_name"] == "DOG_A"].iloc[0]
        assert (
            dog_a_features["historical_win_rate"] > 0
        ), "DOG_A should have historical win rate"

    def test_temporal_order_validation(self, temp_db):
        """UNIT TEST: All historical rows used for target have timestamps strictly earlier."""
        builder = TemporalFeatureBuilder(temp_db)

        # Get target race timestamp
        conn = sqlite3.connect(temp_db)
        target_race = pd.read_sql_query(
            """
        SELECT * FROM race_metadata WHERE race_id = 'race_003'
        """,
            conn,
        )
        target_timestamp = builder.get_race_timestamp(target_race.iloc[0])

        # Get historical data for DOG_A
        historical_data = builder.load_dog_historical_data("DOG_A", target_timestamp)
        conn.close()

        # All historical timestamps should be before target
        for _, row in historical_data.iterrows():
            hist_timestamp = builder.get_race_timestamp(row)
            assert (
                hist_timestamp < target_timestamp
            ), f"Historical race {row['race_id']} timestamp {hist_timestamp} not before target {target_timestamp}"

    def test_temporal_assertion_hook(self):
        """Test that temporal assertion hook detects leakage."""
        assert_no_leakage = create_temporal_assertion_hook()

        # Test with safe features
        safe_features = {"box_number": 1, "weight": 30.5, "historical_win_rate": 0.5}

        # Should not raise exception
        assert_no_leakage(safe_features, "test_race", "test_dog")

        # Test with leakage features
        leakage_features = {
            "box_number": 1,
            "finish_position": 1,  # This is leakage!
            "historical_win_rate": 0.5,
        }

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="TEMPORAL LEAKAGE DETECTED"):
            assert_no_leakage(leakage_features, "test_race", "test_dog")


class TestMLSystemV4:
    """Test ML System V4 functionality."""

    @pytest.fixture
    def temp_db_with_more_data(self):
        """Create temporary database with more comprehensive test data."""
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(temp_db.name)

        # Create tables (same as above but with more data)
        conn.execute(
            """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            grade TEXT,
            distance TEXT,
            track_condition TEXT,
            weather TEXT,
            temperature REAL,
            humidity REAL,
            wind_speed REAL,
            field_size INTEGER,
            race_date TEXT,
            race_time TEXT
        )
        """
        )

        conn.execute(
            """
        CREATE TABLE dog_race_data (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            weight REAL,
            starting_price REAL,
            trainer_name TEXT,
            finish_position INTEGER,
            individual_time REAL,
            sectional_1st REAL,
            margin REAL,
            FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
        )
        """
        )

        # Create enhanced_expert_data table
        conn.execute(
            """
        CREATE TABLE enhanced_expert_data (
            race_id TEXT,
            dog_clean_name TEXT,
            pir_rating INTEGER,
            first_sectional REAL,
            win_time REAL,
            bonus_time REAL,
            PRIMARY KEY (race_id, dog_clean_name)
        )
        """
        )

        # Generate test data with time progression
        base_date = datetime(2024, 1, 1)
        races = []
        dogs = []
        expert_data = []

        for i in range(10):  # 10 races
            race_date = base_date + timedelta(days=i)
            race_id = f"race_{i+1:03d}"

            races.append(
                (
                    race_id,
                    "TRACK_A",
                    "G5",
                    "500m",
                    "Good",
                    "Fine",
                    20.0 + i * 0.5,
                    60.0 + i,
                    10.0 + i * 0.2,
                    8,
                    race_date.strftime("%Y-%m-%d"),
                    "14:30",
                )
            )

            # Create 8 dogs per race
            for j in range(8):
                dog_id = i * 8 + j + 1
                dog_name = f"DOG_{chr(65 + j)}"  # DOG_A, DOG_B, etc.
                box_number = j + 1
                weight = 29.0 + j * 0.3 + np.random.normal(0, 0.2)
                starting_price = 2.0 + j * 0.5 + np.random.normal(0, 0.3)
                trainer = f"TRAINER_{(j % 4) + 1}"

                # Simulate finish positions (some randomness)
                if j == 0:  # DOG_A wins sometimes
                    finish_pos = 1 if np.random.random() > 0.7 else j + 1
                else:
                    finish_pos = j + 1

                individual_time = 29.0 + j * 0.1 + np.random.normal(0, 0.5)
                sectional = 5.0 + j * 0.05 + np.random.normal(0, 0.1)
                margin = 0.0 if finish_pos == 1 else (finish_pos - 1) * 0.2

                dogs.append(
                    (
                        dog_id,
                        race_id,
                        dog_name,
                        box_number,
                        weight,
                        starting_price,
                        trainer,
                        finish_pos,
                        individual_time,
                        sectional,
                        margin,
                    )
                )

                expert_data.append(
                    (race_id, dog_name, 75 + j * 2, sectional, individual_time, 0.5)
                )

        conn.executemany(
            """
        INSERT INTO race_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            races,
        )

        conn.executemany(
            """
        INSERT INTO dog_race_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            dogs,
        )

        conn.executemany(
            """
        INSERT INTO enhanced_expert_data VALUES (?, ?, ?, ?, ?, ?)
        """,
            expert_data,
        )

        conn.commit()
        conn.close()

        yield temp_db.name

        # Cleanup
        os.unlink(temp_db.name)

    def test_time_ordered_split_no_overlap(self, temp_db_with_more_data):
        """INTEGRATION TEST: No race_id appears in both train and test."""
        system = MLSystemV4(temp_db_with_more_data)

        train_data, test_data = system.prepare_time_ordered_data()

        assert not train_data.empty, "Training data should not be empty"
        assert not test_data.empty, "Test data should not be empty"

        train_race_ids = set(train_data["race_id"].unique())
        test_race_ids = set(test_data["race_id"].unique())

        overlap = train_race_ids.intersection(test_race_ids)
        assert len(overlap) == 0, f"Race ID overlap detected: {overlap}"

        # Test temporal order
        train_max_date = train_data["race_timestamp"].max()
        test_min_date = test_data["race_timestamp"].min()

        assert (
            train_max_date <= test_min_date
        ), "Test races should occur after training races"

    def test_encoding_handles_unknown_categoricals(self, temp_db_with_more_data):
        """INTEGRATION TEST: Pipeline handles unseen categoricals without errors."""
        system = MLSystemV4(temp_db_with_more_data)

        # First train the model
        success = system.train_model()
        assert success, "Model training should succeed"

        # Create test data with unknown categorical value
        test_race_data = pd.DataFrame(
            [
                {
                    "race_id": "test_race",
                    "dog_clean_name": "TEST_DOG",
                    "box_number": 1,
                    "weight": 30.0,
                    "starting_price": 3.0,
                    "trainer_name": "UNKNOWN_TRAINER",  # This is unseen
                    "venue": "UNKNOWN_VENUE",  # This is also unseen
                    "grade": "RICH",  # Unseen grade
                    "track_condition": "Good",
                    "weather": "Fine",
                    "temperature": 20.0,
                    "humidity": 60.0,
                    "wind_speed": 10.0,
                    "field_size": 8,
                    "race_date": "2024-02-01",
                    "race_time": "15:00",
                }
            ]
        )

        # Should not raise encoding errors
        result = system.predict_race(test_race_data, "test_race")
        assert result["success"], f"Prediction failed: {result.get('error')}"

    def test_group_normalization_sums_to_one(self, temp_db_with_more_data):
        """INTEGRATION TEST: Per race, sum(win_prob_norm) ≈ 1.0."""
        system = MLSystemV4(temp_db_with_more_data)

        # Train model
        success = system.train_model()
        assert success, "Model training should succeed"

        # Create test race with multiple dogs
        test_race_data = pd.DataFrame(
            [
                {
                    "race_id": "norm_test_race",
                    "dog_clean_name": f"TEST_DOG_{i}",
                    "box_number": i + 1,
                    "weight": 30.0 + i * 0.2,
                    "starting_price": 2.0 + i * 0.5,
                    "trainer_name": f"TRAINER_{i % 3 + 1}",
                    "venue": "TRACK_A",
                    "grade": "G5",
                    "track_condition": "Good",
                    "weather": "Fine",
                    "temperature": 20.0,
                    "humidity": 60.0,
                    "wind_speed": 10.0,
                    "field_size": 8,
                    "race_date": "2024-02-01",
                    "race_time": "15:00",
                }
                for i in range(8)
            ]
        )

        result = system.predict_race(test_race_data, "norm_test_race")
        assert result["success"], f"Prediction failed: {result.get('error')}"

        # Check normalization
        win_probs = [pred["win_prob_norm"] for pred in result["predictions"]]
        prob_sum = sum(win_probs)

        assert (
            0.95 <= prob_sum <= 1.05
        ), f"Win probabilities sum to {prob_sum}, not ≈1.0"

    def test_ev_calculation(self, temp_db_with_more_data):
        """Test Expected Value calculation."""
        system = MLSystemV4(temp_db_with_more_data)

        # Train model
        success = system.train_model()
        assert success, "Model training should succeed"

        # Create test race
        test_race_data = pd.DataFrame(
            [
                {
                    "race_id": "ev_test_race",
                    "dog_clean_name": "EV_DOG_1",
                    "box_number": 1,
                    "weight": 30.0,
                    "starting_price": 2.0,
                    "trainer_name": "TRAINER_1",
                    "venue": "TRACK_A",
                    "grade": "G5",
                    "track_condition": "Good",
                    "weather": "Fine",
                    "temperature": 20.0,
                    "humidity": 60.0,
                    "wind_speed": 10.0,
                    "field_size": 8,
                    "race_date": "2024-02-01",
                    "race_time": "15:00",
                }
            ]
        )

        # Market odds
        market_odds = {"EV_DOG_1": 3.5}

        result = system.predict_race(test_race_data, "ev_test_race", market_odds)
        assert result["success"], f"Prediction failed: {result.get('error')}"

        # Check EV calculation
        prediction = result["predictions"][0]
        assert "ev_win" in prediction, "EV calculation missing"
        assert "odds" in prediction, "Market odds missing"
        assert isinstance(prediction["ev_win"], float), "EV should be float"

    def test_calibration_metadata(self, temp_db_with_more_data):
        """Test that calibration metadata is included."""
        system = MLSystemV4(temp_db_with_more_data)

        # Train model
        success = system.train_model()
        assert success, "Model training should succeed"

        # Create test race
        test_race_data = pd.DataFrame(
            [
                {
                    "race_id": "calib_test_race",
                    "dog_clean_name": "CALIB_DOG",
                    "box_number": 1,
                    "weight": 30.0,
                    "starting_price": 2.0,
                    "trainer_name": "TRAINER_1",
                    "venue": "TRACK_A",
                    "grade": "G5",
                    "track_condition": "Good",
                    "weather": "Fine",
                    "temperature": 20.0,
                    "humidity": 60.0,
                    "wind_speed": 10.0,
                    "field_size": 8,
                    "race_date": "2024-02-01",
                    "race_time": "15:00",
                }
            ]
        )

        result = system.predict_race(test_race_data, "calib_test_race")
        assert result["success"], f"Prediction failed: {result.get('error')}"

        # Check calibration metadata
        assert "calibration_meta" in result, "Calibration metadata missing"
        assert result["calibration_meta"]["applied"], "Calibration should be applied"
        assert (
            result["calibration_meta"]["method"] == "isotonic"
        ), "Should use isotonic calibration"

        # Check prediction has calibration info
        prediction = result["predictions"][0]
        assert prediction[
            "calibration_applied"
        ], "Calibration should be applied to prediction"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
