import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from scripts import run_shadow_non_tgr_rf_evaluation as shadow_eval
from scripts.run_feature_recovery_execution_v1 import load_db_history
from scripts.run_shadow_non_tgr_rf_evaluation import (
    ALLOWED_OUTPUT_PREFIXES,
    DEFAULT_SCHEMA,
    FORBIDDEN_APPROVAL_ENV_VARS,
    POWER_GAMMA,
    STAGE2_FORWARD_SHADOW_COLLECTING,
    active_features_for_loaded_model,
    apply_power_gamma_by_race,
    assert_shadow_output_dir_safe,
    dataset_with_all_missing_train_policy,
    inactive_feature_policy_report,
    main,
    parse_live_runner_identity,
    probability_sum_report,
    ranking_preservation_report,
    same_distance_same_grade_history_provenance_report,
    stage2_shadow_prediction_rows,
    train_eval_feature_parity_report,
    validate_schema_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema(features):
    return {
        "schema_version": "test_schema",
        "feature_columns": features,
        "categorical_features": [],
        "numeric_or_boolean_features": features,
    }


def test_output_manifest_binds_declared_implementation_file_hashes(
    tmp_path, monkeypatch
):
    implementation_files = shadow_eval.IMPLEMENTATION_FILES
    for index, relative in enumerate(implementation_files, start=1):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"implementation-{index}\n", encoding="utf-8")
    output_dir = tmp_path / "artifacts/shadow_score_live"
    output_dir.mkdir(parents=True)
    (output_dir / "shadow_feature_rows.json").write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(shadow_eval, "ROOT", tmp_path)
    monkeypatch.setattr(shadow_eval, "git_output", lambda _: "test-ref")

    manifest = shadow_eval.output_file_manifest(output_dir)

    assert manifest["implementation_files"] == implementation_files
    assert manifest["implementation_file_hashes"] == {
        relative: _sha256(tmp_path / relative) for relative in implementation_files
    }


def test_pytest_database_environment_does_not_target_repo_root_db():
    protected = Path("greyhound_racing_data.db").resolve()
    for key in (
        "DATABASE_PATH",
        "GREYHOUND_DB_PATH",
        "STAGING_DB_PATH",
        "ANALYTICS_DB_PATH",
    ):
        assert Path(os.environ[key]).resolve() != protected


def test_pytest_redirects_legacy_writable_connect_away_from_repo_root_db():
    protected = Path("greyhound_racing_data.db").resolve()
    safe_test_db = Path(os.environ["DATABASE_PATH"]).resolve()
    before_hash = _sha256(protected) if protected.exists() else None

    connection = sqlite3.connect(str(protected))
    try:
        database_path = Path(
            connection.execute("PRAGMA database_list").fetchone()[2]
        ).resolve()
    finally:
        connection.close()

    assert safe_test_db != protected
    assert database_path == safe_test_db
    if before_hash is not None:
        assert _sha256(protected) == before_hash


def test_parse_live_runner_identity_prefers_target_box_prefix():
    assert parse_live_runner_identity("4. Paw Kiplin", "6") == ("Paw Kiplin", 4)
    assert parse_live_runner_identity("Plain Runner", "8") == ("Plain Runner", 8)
    assert parse_live_runner_identity("", "1") == ("", None)


def test_score_live_input_files_ignore_refresh_auxiliary_dirs(tmp_path):
    accepted = tmp_path / "Race 1 - TEST - 2026-06-08.csv"
    accepted.write_text("Dog Name|BOX\n1. Alpha Runner|1\n", encoding="utf-8")
    raw_export = tmp_path / "raw_exports" / "Race 1 - TEST - 2026-06-08.csv"
    raw_export.parent.mkdir()
    raw_export.write_text("Dog Name,BOX\nRaw Runner,1\n", encoding="utf-8")
    quarantine = tmp_path / "quarantine" / "Race 2 - TEST - 2026-06-08.csv"
    quarantine.parent.mkdir()
    quarantine.write_text("Dog Name,BOX\nBad Runner,2\n", encoding="utf-8")

    assert shadow_eval.input_files_from_path(tmp_path) == [accepted]
    assert shadow_eval.input_files_from_path(raw_export) == []
    assert shadow_eval.input_files_from_path(quarantine) == []


def test_live_feature_rows_ignore_embedded_history_rows_and_use_target_boxes(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 1 - TEST - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX|DATE|TRACK|DIST|G|TIME\n"
        "1. Alpha Runner|8|2026-06-01|TEST|300|Grade 5|17.10\n"
        "|3|2026-05-20|TEST|300|Grade 5|17.30\n"
        "2. Bravo Runner|7|2026-06-01|TEST|300|Grade 5|17.20\n"
        "|4|2026-05-20|TEST|300|Grade 5|17.40\n",
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["field_size", "box_number"]},
        db_path=Path("unused.db"),
    )

    assert len(rows) == 2
    assert [(row["dog_name"], row["box_number"], row["field_size"]) for row in rows] == [
        ("Alpha Runner", 1, 2),
        ("Bravo Runner", 2, 2),
    ]
    assert {row["race_date"] for row in rows} == {"2026-06-08"}
    assert {row["venue"] for row in rows} == {"TEST"}
    assert {row["race_number"] for row in rows} == {1}
    assert all(row["target_distance_safe"] is None for row in rows)
    assert all(row["target_grade_safe"] is None for row in rows)
    assert all(row["target_metadata_from_sidecar"] is False for row in rows)


def test_live_feature_rows_use_leakage_safe_sidecar_target_metadata_for_history_features(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 4 - TRA - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX\n"
        "1. Alpha Runner|8\n",
        encoding="utf-8",
    )
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_source_url": "https://www.thedogs.com.au/racing/traralgon/2026-06-08/4/test?trial=false",
                "target_distance": "350m",
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": "Grade 5",
                "target_grade_source": "canonical_pre_race_page",
                "race_info": {
                    "date": "2026-06-08",
                    "venue": "TRA",
                    "race_number": "4",
                    "race_time": "11:15 AM",
                    "url": "https://www.thedogs.com.au/racing/traralgon/2026-06-08/4/test?trial=false",
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(
        shadow_eval,
        "load_db_history",
        lambda _connection: {
            "alpha runner": [
                {
                    "race_date": "2026-06-01",
                    "venue": "TRA",
                    "distance_num": 350,
                    "grade_normalized": "Grade 5",
                    "time_num": 18.12,
                    "finish_num": 1,
                },
                {
                    "race_date": "2026-06-08",
                    "venue": "TRA",
                    "distance_num": 350,
                    "grade_normalized": "Grade 5",
                    "time_num": 17.99,
                    "finish_num": 1,
                },
            ]
        },
    )

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={
            "feature_columns": [
                "field_size",
                "box_number",
                "target_distance_safe",
                "target_grade_safe",
                "race_time_minutes_since_midnight",
                "same_distance_same_grade_start_count",
                "same_distance_same_grade_best_time",
                "same_distance_same_grade_avg_time",
            ]
        },
        db_path=Path("unused.db"),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["race_date"] == "2026-06-08"
    assert row["venue"] == "TRA"
    assert row["race_number"] == 4
    assert row["target_distance_safe"] == 350.0
    assert row["target_grade_safe"] == "Grade 5"
    assert row["race_time_minutes_since_midnight"] == 675
    assert row["target_distance_source"] == "canonical_pre_race_page"
    assert row["target_grade_source"] == "canonical_pre_race_page"
    assert row["target_metadata_from_sidecar"] is True
    assert row["same_distance_same_grade_start_count"] == 1
    assert row["same_distance_same_grade_best_time"] == 18.12
    assert row["same_distance_same_grade_avg_time"] == 18.12
    assert row["same_distance_same_grade_history_status"] == "PASS"
    assert row["same_distance_same_grade_history_source"] == "prior_dog_history"
    assert row["same_distance_same_grade_history_cutoff"] == "strictly_before_target_race"
    assert row["same_distance_same_grade_history_cutoff_basis"] == (
        "race_date_less_than_target_race_date"
    )
    assert row["same_distance_same_grade_prior_history_rows_used"] == 1
    assert row["same_distance_same_grade_target_race_rows_used"] == 0
    assert row["same_distance_same_grade_post_outcome_rows_used"] == 0

    report = same_distance_same_grade_history_provenance_report(rows)
    assert report["status"] == "PASS"
    assert report["target_race_rows_allowed"] == 0
    best = report["by_feature"]["same_distance_same_grade_best_time"]
    assert best["status"] == "PASS"
    assert best["source"] == "prior_dog_history"
    assert best["history_cutoff"] == "strictly_before_target_race"
    assert best["prior_history_rows_used"] == 1
    assert best["target_race_rows_used"] == 0
    avg = report["by_feature"]["same_distance_same_grade_avg_time"]
    assert avg["status"] == "PASS"


def test_live_feature_rows_use_leakage_safe_sidecar_weather_track_metadata(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 4 - TRA - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX\n"
        "1. Alpha Runner|8\n",
        encoding="utf-8",
    )
    source_url = "https://www.thedogs.com.au/racing/traralgon/2026-06-08/4/test?trial=false"
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": "2026-06-08T00:30:00Z",
                "metadata_source_url": source_url,
                "race_url": source_url,
                "weather": "Overcast",
                "track_condition": "Soft",
                "weather_track_metadata_source": "canonical_pre_race_page",
                "weather_track_metadata_is_leakage_safe": True,
                "race_info": {
                    "date": "2026-06-08",
                    "venue": "TRA",
                    "race_number": "4",
                    "race_time": "11:15 AM",
                    "url": source_url,
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["track_condition", "weather"]},
        db_path=Path("unused.db"),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["track_condition"] == "Soft"
    assert row["weather"] == "Overcast"
    assert row["track_condition_source_backed"] is True
    assert row["weather_source_backed"] is True
    assert row["metadata_is_leakage_safe"] is True
    assert row["source_url"] == source_url
    assert row["collection_timestamp"] == "2026-06-08T00:30:00Z"
    assert row["race_time"] == "2026-06-08T11:15:00+10:00"
    assert row["weather_track_metadata_from_sidecar"] is True


def test_live_feature_rows_reject_unsafe_weather_track_sources_and_placeholders(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 4 - TRA - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX|Track Condition|Weather\n"
        "1. Alpha Runner|8|Soft|Overcast\n",
        encoding="utf-8",
    )
    result_url = "https://www.thedogs.com.au/racing/traralgon/2026-06-08/4/results"
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": "2026-06-08T12:30:00+10:00",
                "metadata_source_url": result_url,
                "race_url": result_url,
                "weather": "Fine",
                "track_condition": "Good",
                "weather_track_metadata_source": "canonical_pre_race_page",
                "weather_track_metadata_is_leakage_safe": True,
                "race_info": {
                    "date": "2026-06-08",
                    "venue": "TRA",
                    "race_number": "4",
                    "race_time": "11:15 AM",
                    "url": result_url,
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["track_condition", "weather"]},
        db_path=Path("unused.db"),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["track_condition"] is None
    assert row["weather"] is None
    assert row["track_condition_source_backed"] is False
    assert row["weather_source_backed"] is False
    assert row["weather_track_metadata_from_sidecar"] is False
    assert "source_url_looks_post_result" in row["weather_track_metadata_rejected_sources"]


def test_live_feature_rows_accept_open_meteo_weather_sidecar(tmp_path, monkeypatch):
    race_file = tmp_path / "Race 4 - TRA - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX\n"
        "1. Alpha Runner|8\n",
        encoding="utf-8",
    )
    race_url = "https://www.thedogs.com.au/racing/traralgon/2026-06-08/4/test?trial=false"
    weather_url = "https://api.open-meteo.com/v1/forecast?latitude=-38.18&longitude=146.53"
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": "2026-06-08T00:30:00Z",
                "metadata_source_url": race_url,
                "race_url": race_url,
                "weather": "Overcast",
                "weather_condition": "Overcast",
                "weather_track_metadata_source": "open_meteo_forecast_api",
                "weather_track_metadata_source_url": weather_url,
                "weather_track_metadata_is_leakage_safe": True,
                "race_info": {
                    "date": "2026-06-08",
                    "venue": "TRA",
                    "race_number": "4",
                    "race_time": "11:15 AM",
                    "url": race_url,
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["track_condition", "weather"]},
        db_path=Path("unused.db"),
    )

    row = rows[0]
    assert row["track_condition"] is None
    assert row["weather"] == "Overcast"
    assert row["weather_source_backed"] is True
    assert row["track_condition_source_backed"] is False
    assert row["weather_track_metadata_from_sidecar"] is True
    assert row["source_url"] == weather_url


def test_live_feature_rows_accept_combined_sportsbet_track_and_weather_sidecar(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 9 - SAL - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX\n"
        "1. Alpha Runner|8\n",
        encoding="utf-8",
    )
    race_url = "https://www.thedogs.com.au/racing/sale/2026-06-08/9/test?trial=false"
    sportsbet_url = (
        "https://www.sportsbet.com.au/apigw/sportsbook-racing/"
        "Sportsbook/Racing/NextEvents?racingFilters=GH_DOMESTIC"
    )
    weather_url = "https://api.open-meteo.com/v1/forecast?latitude=-38.10&longitude=147.07"
    source_urls = {
        "sportsbet_pre_race_page": sportsbet_url,
        "open_meteo_forecast_api": weather_url,
    }
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": "2026-06-08T00:30:00Z",
                "metadata_source_url": race_url,
                "race_url": race_url,
                "weather": "Overcast",
                "weather_condition": "Overcast",
                "track_condition": "Good",
                "weather_track_metadata_source": (
                    "sportsbet_pre_race_page+open_meteo_forecast_api"
                ),
                "weather_track_metadata_source_url": source_urls,
                "weather_track_metadata_is_leakage_safe": True,
                "race_info": {
                    "date": "2026-06-08",
                    "venue": "SAL",
                    "race_number": "9",
                    "race_time": "11:15 AM",
                    "url": race_url,
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["track_condition", "weather"]},
        db_path=Path("unused.db"),
    )

    row = rows[0]
    assert row["track_condition"] == "Good"
    assert row["weather"] == "Overcast"
    assert row["track_condition_source_backed"] is True
    assert row["weather_source_backed"] is True
    assert row["weather_track_metadata_from_sidecar"] is True
    assert row["source_url"] == source_urls


def test_live_feature_rows_reject_post_jump_weather_track_capture_time(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 4 - TRA - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX\n"
        "1. Alpha Runner|8\n",
        encoding="utf-8",
    )
    source_url = "https://www.thedogs.com.au/racing/traralgon/2026-06-08/4/test?trial=false"
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": "2026-06-08T12:30:00+10:00",
                "metadata_source_url": source_url,
                "race_url": source_url,
                "weather": "Overcast",
                "track_condition": "Soft",
                "weather_track_metadata_source": "canonical_pre_race_page",
                "weather_track_metadata_is_leakage_safe": True,
                "race_info": {
                    "date": "2026-06-08",
                    "venue": "TRA",
                    "race_number": "4",
                    "race_time": "11:15 AM",
                    "url": source_url,
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["track_condition", "weather"]},
        db_path=Path("unused.db"),
    )

    row = rows[0]
    assert row["track_condition"] is None
    assert row["weather"] is None
    assert "metadata_captured_at_not_before_jump" in row[
        "weather_track_metadata_rejected_sources"
    ]


def test_live_feature_rows_require_explicit_weather_track_safe_flag(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 4 - TRA - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX\n"
        "1. Alpha Runner|8\n",
        encoding="utf-8",
    )
    source_url = "https://www.thedogs.com.au/racing/traralgon/2026-06-08/4/test?trial=false"
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": "2026-06-08T00:30:00Z",
                "metadata_source_url": source_url,
                "race_url": source_url,
                "weather": "Overcast",
                "track_condition": "Soft",
                "race_info": {
                    "date": "2026-06-08",
                    "venue": "TRA",
                    "race_number": "4",
                    "race_time": "11:15 AM",
                    "url": source_url,
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["track_condition", "weather"]},
        db_path=Path("unused.db"),
    )

    row = rows[0]
    assert row["track_condition"] is None
    assert row["weather"] is None
    assert "weather_track_metadata_is_leakage_safe_not_true" in row[
        "weather_track_metadata_rejected_sources"
    ]


def test_live_feature_rows_populate_same_distance_same_grade_from_form_history(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 4 - TRA - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX|DIST|DATE|TRACK|G|TIME|PLC|WGT|1 SEC|MGN\n"
        "1. Alpha Runner|8|400|2026-06-01|TRA|Grade 5|22.10|1|31.2|6.10|0.5\n"
        "|7|400|2026-05-24|TRA|Grade 5|22.40|2|31.0|6.20|1.0\n"
        "|6|330|2026-05-10|TRA|Grade 5|19.90|3|30.9|6.00|2.0\n",
        encoding="utf-8",
    )
    race_url = "https://www.thedogs.com.au/racing/traralgon/2026-06-08/4/test?trial=false"
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": "2026-06-08T00:30:00Z",
                "metadata_source_url": race_url,
                "race_url": race_url,
                "target_distance": "400m",
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": "Grade 5",
                "target_grade_source": "canonical_pre_race_page",
                "race_info": {
                    "date": "2026-06-08",
                    "venue": "TRA",
                    "race_number": "4",
                    "race_time": "11:15 AM",
                    "url": race_url,
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={
            "feature_columns": [
                "same_distance_same_grade_best_time",
                "same_distance_same_grade_avg_time",
                "same_distance_same_grade_start_count",
            ]
        },
        db_path=Path("unused.db"),
    )

    row = rows[0]
    assert row["same_distance_same_grade_start_count"] == 2
    assert row["same_distance_same_grade_best_time"] == 22.10
    assert row["same_distance_same_grade_avg_time"] == pytest.approx(22.25)
    assert row["same_distance_same_grade_history_status"] == "PASS"
    assert row["same_distance_same_grade_prior_history_rows_used"] == 2
    assert row["same_distance_same_grade_target_race_rows_used"] == 0


def test_live_feature_rows_ignore_unsafe_sidecar_target_metadata(tmp_path, monkeypatch):
    race_file = tmp_path / "Race 4 - TRA - 2026-06-08.csv"
    race_file.write_text(
        "Dog Name|BOX\n"
        "1. Alpha Runner|1\n",
        encoding="utf-8",
    )
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "target_distance": "350m",
                "target_distance_source": "result_page",
                "target_grade": "Grade 5",
                "target_grade_source": "embedded_form_history:G",
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["target_distance_safe", "target_grade_safe"]},
        db_path=Path("unused.db"),
    )

    assert len(rows) == 1
    assert rows[0]["target_distance_safe"] is None
    assert rows[0]["target_grade_safe"] is None
    assert rows[0]["target_metadata_from_sidecar"] is False
    assert "unsafe_sidecar_target_distance:result_page" in rows[0][
        "target_metadata_rejected_sources"
    ]
    assert "unsafe_sidecar_target_grade:embedded_form_history:G" in rows[0][
        "target_metadata_rejected_sources"
    ]


def test_same_distance_history_provenance_report_keeps_unpopulated_features_blocked():
    report = same_distance_same_grade_history_provenance_report(
        [
            {
                "same_distance_same_grade_best_time": None,
                "same_distance_same_grade_avg_time": "",
                "same_distance_same_grade_history_cutoff": "strictly_before_target_race",
                "same_distance_same_grade_target_race_rows_used": 0,
                "same_distance_same_grade_post_outcome_rows_used": 0,
            }
        ]
    )

    assert report["status"] == "NOT_POPULATED"
    assert report["by_feature"]["same_distance_same_grade_best_time"]["status"] == (
        "NOT_POPULATED"
    )
    assert report["by_feature"]["same_distance_same_grade_avg_time"]["fail_reasons"] == [
        "feature_not_populated"
    ]


def _parity_dataset():
    features = [
        "learned_num",
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
        "missing_both",
    ]
    return {
        "features": features,
        "categorical_features": [],
        "train_rows": [
            {
                "race_id": "race-train",
                "learned_num": 1.0,
                "same_distance_same_grade_best_time": None,
                "same_distance_same_grade_avg_time": "",
                "missing_both": None,
            },
            {
                "race_id": "race-train",
                "learned_num": 2.0,
                "same_distance_same_grade_best_time": "",
                "same_distance_same_grade_avg_time": None,
                "missing_both": "",
            },
        ],
        "holdout_rows": [
            {
                "race_id": "race-holdout",
                "learned_num": 3.0,
                "same_distance_same_grade_best_time": 29.91,
                "same_distance_same_grade_avg_time": 30.12,
                "missing_both": None,
            }
        ],
    }


def test_default_repaired_schema_has_78_features_and_no_tgr():
    schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))

    audit = validate_schema_contract(schema)

    assert audit["status"] == "PASS"
    assert audit["feature_count"] == 78
    assert audit["tgr_columns"] == []


def test_history_loader_tolerates_minimal_main_db_dog_schema():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            grade TEXT,
            distance TEXT,
            track_condition TEXT,
            weather TEXT,
            race_time TEXT,
            start_datetime TEXT
        );
        CREATE TABLE dog_race_data (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            dog_name TEXT,
            finish_position TEXT,
            dog_clean_name TEXT,
            box_number INTEGER
        );
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, grade, distance, race_time)
        VALUES
            ('race-1', 'TEST', 1, '2026-01-01', 'Grade 5', '450', '12:00');
        INSERT INTO dog_race_data
            (race_id, dog_name, finish_position, dog_clean_name, box_number)
        VALUES
            ('race-1', 'Schema Dog', '2', 'Schema Dog', 1);
        """
    )

    history = load_db_history(connection)

    assert list(history) == ["schema dog"]
    row = history["schema dog"][0]
    assert row["finish_num"] == 2
    assert row["time_num"] is None
    assert row["margin_num"] is None


def test_schema_rejects_tgr_identity_and_post_outcome_features():
    features = [f"feature_{index}" for index in range(75)]
    features.extend(["tgr_speed", "race_id", "finish_position"])

    audit = validate_schema_contract(_schema(features))

    assert audit["status"] == "FAIL"
    assert audit["tgr_columns"] == ["tgr_speed"]
    assert audit["identity_columns_present_as_features"] == ["race_id"]
    assert audit["post_outcome_columns_present_as_features"] == ["finish_position"]


def test_power_gamma_2p4_normalizes_and_preserves_ranking():
    rows = [
        {
            "shadow_race_group_id": "race-a",
            "race_id": "Race A",
            "dog_name": "Alpha",
            "box_number": 1,
            "shadow_rf_uncalibrated_probability": 0.60,
        },
        {
            "shadow_race_group_id": "race-a",
            "race_id": "Race A",
            "dog_name": "Bravo",
            "box_number": 2,
            "shadow_rf_uncalibrated_probability": 0.30,
        },
        {
            "shadow_race_group_id": "race-a",
            "race_id": "Race A",
            "dog_name": "Charlie",
            "box_number": 3,
            "shadow_rf_uncalibrated_probability": 0.10,
        },
        {
            "shadow_race_group_id": "race-b",
            "race_id": "Race B",
            "dog_name": "Delta",
            "box_number": 1,
            "shadow_rf_uncalibrated_probability": 0.55,
        },
        {
            "shadow_race_group_id": "race-b",
            "race_id": "Race B",
            "dog_name": "Echo",
            "box_number": 2,
            "shadow_rf_uncalibrated_probability": 0.45,
        },
    ]

    calibrated = apply_power_gamma_by_race(
        rows,
        gamma=POWER_GAMMA,
        input_key="shadow_rf_uncalibrated_probability",
        output_key="shadow_rf_calibrated_probability",
        output_rank_key="shadow_rf_calibrated_rank",
    )

    sums = probability_sum_report(calibrated, "shadow_rf_calibrated_probability")
    ranking = ranking_preservation_report(
        rows,
        calibrated,
        before_key="shadow_rf_uncalibrated_probability",
        after_key="shadow_rf_calibrated_probability",
    )
    assert sums["status"] == "PASS"
    assert sums["max_abs_error"] == pytest.approx(0.0)
    assert ranking["status"] == "PASS"
    assert [row["shadow_rf_calibrated_rank"] for row in calibrated[:3]] == [1, 2, 3]


def test_stage2_prediction_rows_are_shadow_only_and_do_not_use_odds_or_ev():
    rows = stage2_shadow_prediction_rows(
        [
            {
                "race_id": "Race 1 - TEST - 2026-06-10",
                "dog_name": "Alpha Runner",
                "box": 1,
                "shadow_rf_calibrated_probability": 0.62,
                "predicted_rank": 1,
            }
        ],
        stage2_status=STAGE2_FORWARD_SHADOW_COLLECTING,
    )

    assert rows == [
        {
            "schema_version": "stage2_shadow_prediction_v1",
            "race_id": "Race 1 - TEST - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box": 1,
            "shadow_rf_calibrated_probability": 0.62,
            "predicted_rank": 1,
            "stage2_forward_shadow_status": STAGE2_FORWARD_SHADOW_COLLECTING,
            "stage2_challenger_family": "RandomForest",
            "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            "odds_used_for_shadow_scoring": False,
            "ev_output": False,
            "betting_action": False,
            "production_prediction_write": False,
            "registry_mutation": False,
            "production_pointer_update": False,
        }
    ]


def test_train_eval_parity_detects_all_missing_train_present_holdout_features():
    report = train_eval_feature_parity_report(_parity_dataset(), policy="report_only")

    assert report["policy_status"] == "WARN"
    assert report["all_missing_train_present_holdout_features"] == [
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
    ]
    best = report["watched_features"]["same_distance_same_grade_best_time"]
    assert best["train_present_rows"] == 0
    assert best["holdout_present_rows"] == 1
    assert best["parity_status"] == "ALL_MISSING_IN_TRAIN_PRESENT_IN_HOLDOUT"


def test_report_only_policy_retains_features_and_records_warning():
    dataset = _parity_dataset()
    report = train_eval_feature_parity_report(dataset, policy="report_only")
    policy = inactive_feature_policy_report(report)
    active = dataset_with_all_missing_train_policy(dataset, report)

    assert policy["policy_action"] == "report_warning_keep_features_active"
    assert policy["inactive_features_due_to_train_all_missing"] == []
    assert active["features"] == dataset["features"]
    assert active["schema_features"] == dataset["features"]


def test_quarantine_policy_removes_train_all_missing_features_for_run_only():
    dataset = _parity_dataset()
    report = train_eval_feature_parity_report(dataset, policy="quarantine_feature")
    active = dataset_with_all_missing_train_policy(dataset, report)

    assert report["inactive_features_due_to_train_all_missing"] == [
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
        "missing_both",
    ]
    assert active["features"] == ["learned_num"]
    assert active["schema_features"] == dataset["features"]
    assert dataset["features"] == [
        "learned_num",
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
        "missing_both",
    ]


def test_loaded_model_feature_policy_recovers_quarantined_active_features(tmp_path):
    model_path = tmp_path / "shadow_randomforest_model.joblib"
    model_path.write_bytes(b"model-placeholder")
    (tmp_path / "shadow_training_report.json").write_text(
        json.dumps(
            {
                "active_feature_count": 2,
                "schema_feature_count": 4,
                "inactive_features_due_to_train_all_missing": [
                    "same_distance_same_grade_best_time",
                    "same_distance_same_grade_avg_time",
                ],
                "all_missing_train_policy": "quarantine_feature",
            }
        ),
        encoding="utf-8",
    )
    schema = _schema(
        [
            "learned_num",
            "same_distance_same_grade_best_time",
            "same_distance_same_grade_avg_time",
            "box_number",
        ]
    )

    active, policy = active_features_for_loaded_model(model_path=model_path, schema=schema)

    assert active == ["learned_num", "box_number"]
    assert policy["active_feature_count"] == 2
    assert policy["all_missing_train_policy"] == "quarantine_feature"


def test_fail_policy_aborts_before_training_or_scoring():
    dataset = _parity_dataset()
    report = train_eval_feature_parity_report(dataset, policy="fail")

    with pytest.raises(RuntimeError, match="all_missing_train_policy_failed"):
        dataset_with_all_missing_train_policy(dataset, report)


def test_probability_outputs_still_normalize_with_parity_warning():
    report = train_eval_feature_parity_report(_parity_dataset(), policy="report_only")
    assert report["policy_status"] == "WARN"

    rows = [
        {
            "shadow_race_group_id": "race-a",
            "shadow_rf_uncalibrated_probability": 0.6,
        },
        {
            "shadow_race_group_id": "race-a",
            "shadow_rf_uncalibrated_probability": 0.4,
        },
    ]
    calibrated = apply_power_gamma_by_race(
        rows,
        input_key="shadow_rf_uncalibrated_probability",
        output_key="shadow_rf_calibrated_probability",
        output_rank_key="shadow_rf_calibrated_rank",
    )
    assert probability_sum_report(calibrated, "shadow_rf_calibrated_probability")["status"] == "PASS"


def test_shadow_output_path_rejects_production_paths(tmp_path):
    assert_shadow_output_dir_safe(
        tmp_path / "artifacts" / "shadow_evaluation" / "run",
        root=tmp_path,
    )
    assert_shadow_output_dir_safe(
        tmp_path
        / "artifacts"
        / "full_evidence_orchestration_20260525"
        / "shadow_evaluation_implementation_test",
        root=tmp_path,
    )
    assert_shadow_output_dir_safe(
        tmp_path
        / "artifacts"
        / "full_evidence_orchestration_20260525"
        / "shadow_reliability_population_hardening_v1_test"
        / "phase_2_population_parity_guard"
        / "shadow_run",
        root=tmp_path,
    )
    assert_shadow_output_dir_safe(
        tmp_path
        / "artifacts"
        / "full_evidence_orchestration_20260525"
        / "shadow_reliability_resume_after_db_recovery_test"
        / "phase_5_shadow_rerun"
        / "shadow_run",
        root=tmp_path,
    )
    retained_root = (
        tmp_path.parent
        / f"{tmp_path.name}_retained"
        / "artifacts"
        / "full_evidence_orchestration_20260525"
    )
    assert_shadow_output_dir_safe(
        retained_root / "daily_race_ingest_shadow_test" / "shadow_score_live",
        root=tmp_path,
        evidence_root=retained_root,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_shadow_artifact"):
        assert_shadow_output_dir_safe(tmp_path / "predictions" / "shadow", root=tmp_path)
    with pytest.raises(ValueError, match="output_dir_must_be_shadow_artifact"):
        assert_shadow_output_dir_safe(tmp_path / "model_registry" / "shadow", root=tmp_path)
    with pytest.raises(ValueError, match="output_dir_must_be_shadow_artifact"):
        assert_shadow_output_dir_safe(
            retained_root / "not_daily_shadow" / "shadow_score_live",
            root=tmp_path,
            evidence_root=retained_root,
        )


def test_cli_stop_after_definition_writes_only_shadow_candidate(tmp_path, monkeypatch):
    for env_name in FORBIDDEN_APPROVAL_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)
    schema = _schema([f"feature_{index}" for index in range(78)])
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    output_dir = Path("artifacts/shadow_evaluation") / f"pytest_candidate_{tmp_path.name}"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        rc = main(
            [
                "run",
                "--schema",
                str(schema_path),
                "--output-dir",
                str(output_dir),
                "--stop-after-definition",
            ]
        )

        assert rc == 0
        assert (output_dir / "shadow_candidate_definition.json").exists()
        assert not (output_dir / "shadow_predictions.csv").exists()
        assert not any(
            "bet" in path.name.lower() or "ev" == path.stem.lower()
            for path in output_dir.rglob("*")
        )
        definition = json.loads((output_dir / "shadow_candidate_definition.json").read_text())
        assert definition["registry_mutation"] is False
        assert definition["promotion_allowed"] is False
        assert definition["tgr_enabled"] is False
        assert definition["output_mode"] == "shadow_only"
        assert definition["calibration"]["method_key"] == "power_gamma_2.4"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_allowed_output_prefixes_are_shadow_only():
    assert ALLOWED_OUTPUT_PREFIXES == (
        "artifacts/shadow_evaluation",
        "artifacts/full_evidence_orchestration_20260525/shadow_evaluation_",
        "artifacts/full_evidence_orchestration_20260525/shadow_reliability_population_hardening_v1_",
        "artifacts/full_evidence_orchestration_20260525/shadow_reliability_resume_after_db_recovery_",
        "artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_",
    )
