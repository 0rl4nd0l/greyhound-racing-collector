import importlib.util
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from enhanced_accuracy_optimizer import AccuracyOptimizer
from temporal_feature_builder import (
    TemporalFeatureBuilder,
    classify_dog_history_status,
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


def test_history_status_reports_matched_identity_rows_missing_finish_position(tmp_path):
    db_path = tmp_path / "history.db"
    conn = _init_history_db(db_path)
    conn.execute(
        """
        INSERT INTO race_metadata (
            race_id, venue, grade, distance, track_condition, weather, field_size,
            race_date, race_time
        ) VALUES ('missing_finish', 'MEA', 'G5', '525', 'Good', 'Fine', 8,
            '2026-05-01', '19:20')
        """
    )
    conn.execute(
        """
        INSERT INTO dog_race_data (
            race_id, dog_name, dog_clean_name, box_number, finish_position,
            individual_time
        ) VALUES ('missing_finish', 'Missing Finish', 'Missing Finish', 1, NULL, NULL)
        """
    )
    conn.commit()
    conn.close()

    report = classify_dog_history_status(
        str(db_path), "Missing Finish", datetime(2026, 5, 21, 12, 0)
    )

    assert report["db_result_history_count"] == 0
    assert report["db_history_match_status"] == "matched_identity_rows_missing_finish_position"


def test_history_status_does_not_treat_target_null_finish_as_history(tmp_path):
    db_path = tmp_path / "history.db"
    conn = _init_history_db(db_path)
    conn.execute(
        """
        INSERT INTO race_metadata (
            race_id, venue, grade, distance, track_condition, weather, field_size,
            race_date, race_time
        ) VALUES ('target_unresulted', 'MEA', 'G5', '525', 'Good', 'Fine', 8,
            '2026-05-21', '19:20')
        """
    )
    conn.execute(
        """
        INSERT INTO dog_race_data (
            race_id, dog_name, dog_clean_name, box_number, finish_position,
            individual_time
        ) VALUES ('target_unresulted', 'Target Null', 'Target Null', 1, NULL, NULL)
        """
    )
    conn.commit()
    conn.close()

    report = classify_dog_history_status(
        str(db_path), "Target Null", datetime(2026, 5, 21, 12, 0)
    )

    assert report["db_result_history_count"] == 0
    assert report["db_history_match_status"] == "matched_identity_no_result_rows"


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
    assert mapped.loc[0, "embedded_history_race_count"] == 2
    assert mapped.loc[0, "embedded_history_recent_count"] == 2
    assert mapped.loc[0, "embedded_history_avg_finish"] == 1.5
    assert mapped.loc[0, "embedded_history_best_finish"] == 1
    assert mapped.loc[0, "embedded_history_same_track_count"] == 2
    assert mapped.loc[0, "embedded_history_same_distance_band_count"] == 2
    assert mapped.loc[0, "embedded_history_recency_days_min"] == 11
    assert mapped.loc[0, "embedded_history_recency_days_max"] == 20
    assert mapped.loc[0, "distance"] == 525.0
    assert mapped.loc[0, "distance_source"] == "target_column:Race Distance"
    assert mapped.loc[0, "grade"] == "GRADE 5"
    assert mapped.loc[0, "grade_source"] == "target_column:Race Grade"
    assert bool(mapped.loc[0, "metadata_is_leakage_safe"]) is True
    assert mapped.loc[0, "metadata_source_detail"] == {
        "distance": "source_csv_target_column:Race Distance",
        "grade": "source_csv_target_column:Race Grade",
    }
    assert mapped.loc[0, "finish_position"] is None
    assert mapped.loc[0, "individual_time"] is None
    assert mapped.loc[0, "starting_price_source"] == "default_missing_target"
    assert_no_target_leakage_columns(
        ["box_number", "dog_clean_name", "distance", "grade", "csv_historical_races"]
    )


def test_embedded_form_defaults_target_metadata_and_excludes_future_rows():
    race_data = pd.DataFrame(
        [
            {
                "Dog Name": "1. Alpha Runner",
                "PLC": "2",
                "BOX": "1",
                "DIST": "520",
                "DATE": "2026-05-10",
                "TRACK": "MEA",
                "G": "5",
                "TIME": "30.12",
            },
            {
                "Dog Name": '""',
                "PLC": "1",
                "BOX": "1",
                "DIST": "520",
                "DATE": "2026-05-21",
                "TRACK": "MEA",
                "G": "5",
                "TIME": "29.90",
            },
        ]
    )
    pipeline = object.__new__(PredictionPipelineV4)

    mapped = pipeline._map_csv_to_v4_format(
        race_data, "Race 7 - MEA - 2026-05-21.csv"
    )

    assert mapped.loc[0, "distance_source"] == "default_missing_target"
    assert mapped.loc[0, "grade_source"] == "default_missing_target"
    assert bool(mapped.loc[0, "metadata_is_leakage_safe"]) is False
    rejected = set(mapped.loc[0, "rejected_metadata_sources"])
    assert {
        "embedded_form_history:DIST",
        "embedded_form_history:G",
        "post_result_field:PLC",
        "post_result_field:TIME",
    }.issubset(rejected)
    assert mapped.loc[0, "csv_historical_races"] == 1
    assert mapped.loc[0, "csv_history_rows_dropped_post_target"] == 1
    assert mapped.loc[0, "embedded_history_race_count"] == 1
    assert mapped.loc[0, "embedded_history_recent_count"] == 1
    assert mapped.loc[0, "embedded_history_recency_days_min"] == 11
    assert mapped.loc[0, "finish_position"] is None


def test_sidecar_target_metadata_is_accepted_for_embedded_form_history(tmp_path):
    race_data = pd.DataFrame(
        [
            {
                "Dog Name": "1. Alpha Runner",
                "PLC": "2",
                "BOX": "1",
                "DIST": "520",
                "DATE": "2026-05-10",
                "TRACK": "MEA",
                "G": "5",
                "TIME": "30.12",
            },
            {
                "Dog Name": '""',
                "PLC": "1",
                "BOX": "1",
                "DIST": "525",
                "DATE": "2026-05-01",
                "TRACK": "MEA",
                "G": "5",
                "TIME": "29.90",
            },
        ]
    )
    csv_path = tmp_path / "Race 7 - MEA - 2026-05-21.csv"
    csv_path.with_suffix(csv_path.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "target_distance": "525m",
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": "Grade 5",
                "target_grade_source": "canonical_pre_race_page",
                "metadata_is_leakage_safe": True,
                "metadata_source_url": "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
            }
        ),
        encoding="utf-8",
    )
    pipeline = object.__new__(PredictionPipelineV4)

    mapped = pipeline._map_csv_to_v4_format(race_data, str(csv_path))

    assert mapped.loc[0, "distance"] == 525.0
    assert mapped.loc[0, "distance_source"] == "canonical_pre_race_page"
    assert mapped.loc[0, "grade"] == "GRADE 5"
    assert mapped.loc[0, "grade_source"] == "canonical_pre_race_page"
    assert bool(mapped.loc[0, "metadata_is_leakage_safe"]) is True
    assert mapped.loc[0, "embedded_history_same_distance_band_count"] == 2
    rejected = set(mapped.loc[0, "rejected_metadata_sources"])
    assert "embedded_form_history:DIST" in rejected
    assert "embedded_form_history:G" in rejected


def test_generic_post_result_metadata_is_rejected_for_embedded_form_history():
    race_data = pd.DataFrame(
        [
            {
                "Dog Name": "1. Alpha Runner",
                "PLC": "2",
                "BOX": "1",
                "DIST": "520",
                "Distance": "520m",
                "Grade": "Grade 5",
                "DATE": "2026-05-10",
                "TRACK": "MEA",
                "G": "5",
                "TIME": "30.12",
            }
        ]
    )
    pipeline = object.__new__(PredictionPipelineV4)

    mapped = pipeline._map_csv_to_v4_format(
        race_data, "Race 7 - MEA - 2026-05-21.csv"
    )

    assert mapped.loc[0, "distance_source"] == "default_missing_target"
    assert mapped.loc[0, "grade_source"] == "default_missing_target"
    assert bool(mapped.loc[0, "metadata_is_leakage_safe"]) is False
    rejected = set(mapped.loc[0, "rejected_metadata_sources"])
    assert "post_result_field:PLC" in rejected
    assert "post_result_field:TIME" in rejected


def test_filename_venue_token_does_not_fill_target_grade():
    race_data = pd.DataFrame(
        [
            {
                "Dog Name": "1. Alpha Runner",
                "PLC": "2",
                "BOX": "1",
                "DIST": "520",
                "DATE": "2026-05-10",
                "TRACK": "AP_K",
                "G": "5",
                "TIME": "30.12",
            }
        ]
    )
    pipeline = object.__new__(PredictionPipelineV4)

    mapped = pipeline._map_csv_to_v4_format(
        race_data, "Race 1 - AP_K - 2026-05-21.csv"
    )

    assert mapped.loc[0, "distance_source"] == "default_missing_target"
    assert mapped.loc[0, "grade_source"] == "default_missing_target"
    assert bool(mapped.loc[0, "metadata_is_leakage_safe"]) is False


def test_explicit_filename_target_metadata_is_accepted_when_labelled():
    race_data = pd.DataFrame(
        [
            {
                "Dog Name": "1. Alpha Runner",
                "BOX": "1",
                "DATE": "2026-05-10",
                "TRACK": "AP_K",
            }
        ]
    )
    pipeline = object.__new__(PredictionPipelineV4)

    mapped = pipeline._map_csv_to_v4_format(
        race_data,
        "Race 1 - AP_K - 2026-05-21 - distance-520m - grade-G5.csv",
    )

    assert mapped.loc[0, "distance"] == 520.0
    assert mapped.loc[0, "distance_source"] == "filename:distance"
    assert mapped.loc[0, "grade"] == "G5"
    assert mapped.loc[0, "grade_source"] == "filename:grade"
    assert bool(mapped.loc[0, "metadata_is_leakage_safe"]) is True


def test_history_provenance_distinguishes_db_embedded_and_no_history(tmp_path):
    db_path = tmp_path / "history.db"
    conn = _init_history_db(db_path)
    _insert_history(conn, "hist_alpha", "Alpha Runner", race_date="2026-05-01")
    conn.commit()
    conn.close()

    pipeline = object.__new__(PredictionPipelineV4)
    pipeline.db_path = str(db_path)
    pipeline.ml_system_v4 = SimpleNamespace(
        temporal_builder=TemporalFeatureBuilder(str(db_path))
    )
    race_data = pd.DataFrame(
        [
            {
                "dog_clean_name": "Alpha Runner",
                "race_date": "2026-05-21",
                "race_time": "12:00",
                "csv_historical_races": 2,
            },
            {
                "dog_clean_name": "Embedded Only",
                "race_date": "2026-05-21",
                "race_time": "12:00",
                "csv_historical_races": 3,
            },
            {
                "dog_clean_name": "No History",
                "race_date": "2026-05-21",
                "race_time": "12:00",
                "csv_historical_races": 0,
            },
        ]
    )

    annotated = pipeline._annotate_history_provenance(race_data)

    assert annotated.loc[0, "history_source"] == "db_and_embedded_csv_history"
    assert annotated.loc[0, "history_match_status"] == "matched_identity_with_pre_target_results"
    assert annotated.loc[1, "history_source"] == "embedded_csv_form_history"
    assert annotated.loc[1, "history_match_status"] == "embedded_history_only"
    assert "embedded_csv_history_only" in annotated.loc[1, "provenance_quality_flags"]
    assert annotated.loc[2, "history_source"] == "no_usable_history"
    assert annotated.loc[2, "history_match_status"] == "no_matching_identity"


def test_incomplete_runner_sets_are_rejected():
    report = analyze_csv_text_runner_completeness(
        "Dog Name,BOX\n1. Alpha,1\n1. Beta,1\n2. Gamma,2\n3. Delta,3\n",
        source="inline.csv",
    )

    assert report.status == "INCOMPLETE"
    assert "duplicate_box_numbers:1" in report.reasons


def test_optimizer_retains_low_confidence_runner_for_alignment(monkeypatch):
    monkeypatch.delenv("V4_OPTIMIZER_DROP_LOW_QUALITY", raising=False)
    optimizer = object.__new__(AccuracyOptimizer)
    optimizer.config = {"min_confidence_threshold": 0.3}

    predictions = [
        {
            "dog_clean_name": "Maximum Nana",
            "box_number": 1,
            "win_probability": 0.3845,
            "confidence": 0.2643,
            "ensemble_models": 1,
            "model_agreement": None,
        },
        {
            "dog_clean_name": "Fresh Eyes",
            "box_number": 2,
            "win_probability": 0.1641,
            "confidence": 0.3994,
            "ensemble_models": 1,
            "model_agreement": None,
        },
    ]

    filtered = optimizer._apply_quality_filters(predictions)

    assert [prediction["box_number"] for prediction in filtered] == [1, 2]
    retained = filtered[0]
    assert retained["quality_filter_status"] == "retained_for_runner_alignment"
    assert "optimizer_low_confidence" in retained["quality_flags"]
    assert (
        "optimizer_retained_low_quality_for_runner_alignment"
        in retained["quality_flags"]
    )
