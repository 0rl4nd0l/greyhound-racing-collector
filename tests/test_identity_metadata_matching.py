import importlib.util
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from temporal_feature_builder import (
    TemporalFeatureBuilder,
    normalize_dog_identity_key,
)
from utils.leakage_guard import assert_no_target_leakage_columns
from utils.runner_completeness import analyze_csv_text_runner_completeness


ROOT = Path(__file__).resolve().parents[1]
_PP4_SPEC = importlib.util.spec_from_file_location(
    "prediction_pipeline_v4_real", ROOT / "prediction_pipeline_v4.py"
)
_PP4 = importlib.util.module_from_spec(_PP4_SPEC)
assert _PP4_SPEC.loader is not None
_PP4_SPEC.loader.exec_module(_PP4)
PredictionPipelineV4 = _PP4.PredictionPipelineV4


def _init_history_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            grade TEXT,
            distance TEXT,
            track_condition TEXT,
            weather TEXT,
            field_size INTEGER,
            race_date TEXT,
            race_time TEXT,
            winner_name TEXT,
            winner_odds REAL,
            winner_margin REAL
        );
        CREATE TABLE dog_race_data (
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            individual_time TEXT
        );
        CREATE TABLE enhanced_expert_data (
            race_id TEXT,
            dog_clean_name TEXT,
            pir_rating REAL,
            first_sectional REAL,
            win_time REAL,
            bonus_time REAL
        );
        """
    )
    return conn


def _insert_history(conn, race_id, dog_name, race_date="2026-05-01"):
    conn.execute(
        """
        INSERT INTO race_metadata (
            race_id, venue, grade, distance, track_condition, weather, field_size,
            race_date, race_time
        ) VALUES (?, 'MEA', 'G5', '525', 'Good', 'Fine', 8, ?, '19:20')
        """,
        (race_id, race_date),
    )
    conn.execute(
        """
        INSERT INTO dog_race_data (
            race_id, dog_name, dog_clean_name, box_number, finish_position,
            individual_time
        ) VALUES (?, ?, ?, 1, 2, '30.12')
        """,
        (race_id, dog_name, dog_name),
    )


def test_dog_identity_key_matches_punctuation_without_substring_false_positive(tmp_path):
    assert normalize_dog_identity_key("Nanny's Boy") == "NANNYSBOY"
    assert normalize_dog_identity_key("Nannys Boy") == "NANNYSBOY"

    db_path = tmp_path / "history.db"
    conn = _init_history_db(db_path)
    _insert_history(conn, "hist_nanny", "Nanny's Boy")
    _insert_history(conn, "hist_black_longhorn", "Black Longhorn")
    conn.commit()
    conn.close()

    builder = TemporalFeatureBuilder(str(db_path))
    target_ts = datetime(2026, 5, 21, 12, 0)

    matched = builder.load_dog_historical_data("Nannys Boy", target_ts)
    assert len(matched) == 1
    assert matched.iloc[0]["dog_clean_name"] == "Nanny's Boy"

    not_matched = builder.load_dog_historical_data("Longhorn", target_ts)
    assert not_matched.empty


def test_missing_db_history_has_concrete_reason(tmp_path):
    db_path = tmp_path / "history.db"
    conn = _init_history_db(db_path)
    _insert_history(conn, "target_only", "Target Only", race_date="2026-05-21")
    conn.commit()
    conn.close()

    builder = TemporalFeatureBuilder(str(db_path))
    result = builder.load_dog_historical_data(
        "Unknown Runner", datetime(2026, 5, 21, 12, 0)
    )

    assert result.empty
    assert (
        builder._missing_db_history_reason(
            "Unknown Runner", datetime(2026, 5, 21, 12, 0), "2025-05-21"
        )
        == "no_matching_dog_identity_in_db"
    )


def test_form_guide_csv_forward_fill_and_target_metadata_are_leakage_safe():
    race_data = pd.DataFrame(
        [
            {
                "Dog Name": "1. Nanny's Boy",
                "PLC": "2",
                "BOX": "1",
                "WGT": "31.2",
                "DIST": "520",
                "DATE": "2026-05-10",
                "TRACK": "MEA",
                "G": "5",
                "TIME": "30.12",
                "SP": "4.0",
                "Race Distance": "525m",
                "Race Grade": "Grade 5",
            },
            {
                "Dog Name": '""',
                "PLC": "1",
                "BOX": "2",
                "WGT": "31.0",
                "DIST": "515",
                "DATE": "2026-05-01",
                "TRACK": "MEA",
                "G": "5",
                "TIME": "29.99",
                "SP": "2.5",
                "Race Distance": "525m",
                "Race Grade": "Grade 5",
            },
        ]
    )
    pipeline = object.__new__(PredictionPipelineV4)

    mapped = pipeline._map_csv_to_v4_format(
        race_data, "Race 7 - MEA - 2026-05-21.csv"
    )

    assert mapped.loc[0, "dog_clean_name"] == "Nannys Boy"
    assert mapped.loc[0, "csv_historical_races"] == 2
    assert mapped.loc[0, "csv_blank_history_rows"] == 1
    assert mapped.loc[0, "distance"] == 525.0
    assert mapped.loc[0, "distance_source"] == "target_column:Race Distance"
    assert mapped.loc[0, "grade"] == "GRADE 5"
    assert mapped.loc[0, "grade_source"] == "target_column:Race Grade"
    assert mapped.loc[0, "finish_position"] is None
    assert mapped.loc[0, "individual_time"] is None
    assert mapped.loc[0, "starting_price_source"] == "default_missing_target"
    assert_no_target_leakage_columns(
        ["box_number", "dog_clean_name", "distance", "grade", "csv_historical_races"]
    )


def test_incomplete_runner_sets_are_rejected():
    report = analyze_csv_text_runner_completeness(
        "Dog Name,BOX\n1. Alpha,1\n1. Beta,1\n2. Gamma,2\n3. Delta,3\n",
        source="inline.csv",
    )

    assert report.status == "INCOMPLETE"
    assert "duplicate_box_numbers:1" in report.reasons
