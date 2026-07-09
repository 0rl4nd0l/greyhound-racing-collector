import json
import sqlite3

from scripts.run_feature_recovery_execution_v1 import (
    can_reuse_packet_feature,
    resolve_target_metadata,
    safe_sidecar_metadata,
    target_metadata_recovery_audit_report,
    target_metadata_recovery_audit_row,
)


def _race_metadata_db(rows):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE race_metadata (
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            grade TEXT,
            distance TEXT,
            field_size INTEGER,
            race_time TEXT,
            track_condition TEXT,
            weather TEXT,
            weather_condition TEXT,
            url TEXT,
            data_source TEXT,
            start_datetime TEXT,
            results_status TEXT,
            winner_source TEXT,
            winner_name TEXT
        )
        """
    )
    for row in rows:
        payload = {
            "race_id": None,
            "venue": "SHEP",
            "race_number": 4,
            "race_date": "2026-06-15",
            "grade": None,
            "distance": None,
            "field_size": None,
            "race_time": None,
            "track_condition": None,
            "weather": None,
            "weather_condition": None,
            "url": None,
            "data_source": None,
            "start_datetime": None,
            "results_status": "pending",
            "winner_source": None,
            "winner_name": None,
        }
        payload.update(row)
        connection.execute(
            """
            INSERT INTO race_metadata (
                race_id, venue, race_number, race_date, grade, distance,
                field_size, race_time, track_condition, weather,
                weather_condition, url, data_source, start_datetime,
                results_status, winner_source, winner_name
            )
            VALUES (
                :race_id, :venue, :race_number, :race_date, :grade, :distance,
                :field_size, :race_time, :track_condition, :weather,
                :weather_condition, :url, :data_source, :start_datetime,
                :results_status, :winner_source, :winner_name
            )
            """,
            payload,
        )
    return connection


def test_safe_sidecar_metadata_accepts_top_level_canonical_target_fields(tmp_path):
    csv_path = tmp_path / "Race 4 - SHEP - 2026-06-15.csv"
    csv_path.write_text("Dog Name,BOX\n1. Runner,1\n", encoding="utf-8")
    sidecar = csv_path.with_name(csv_path.name + ".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_source_url": "https://www.thedogs.com.au/racing/shepparton/2026-06-15/4/test?trial=false",
                "target_distance": "450m",
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": "Grade 5",
                "target_grade_source": "canonical_pre_race_page",
                "normalization_verification": {
                    "target_metadata_status": "verified",
                    "race_time_mapping_status": "exact_url_match",
                    "race_time_source": "canonical_race_url",
                },
                "race_info": {"race_time": "7:24 PM"},
            }
        ),
        encoding="utf-8",
    )

    metadata = safe_sidecar_metadata({"source_file_path": str(csv_path)})

    assert metadata["distance"] == 450
    assert metadata["grade"] == "Grade 5"
    assert metadata["source"] == "safe_sidecar_metadata"


def test_resolve_target_metadata_accepts_safe_exact_db_identity():
    connection = _race_metadata_db(
        [
            {
                "race_id": "Race 4 - SHEP - 2026-06-15",
                "distance": "450",
                "grade": "Grade 5",
                "data_source": "canonical_pre_race_page",
                "url": "https://www.thedogs.com.au/racing/shepparton/2026-06-15/4/test?trial=false",
            }
        ]
    )

    target = resolve_target_metadata(
        {
            "race_id": "Race 4 - SHEP - 2026-06-15",
            "race_date": "2026-06-15",
            "venue": "SHEP",
        },
        {},
        {"race_number": 4},
        connection,
    )

    assert target["status"] == "SAFE"
    assert target["distance"] == 450
    assert target["grade"] == "Grade 5"
    assert target["source"] == "canonical_race_metadata_exact_identity"


def test_resolve_target_metadata_rejects_embedded_form_history_db_metadata():
    connection = _race_metadata_db(
        [
            {
                "race_id": "SHEP_2026-06-15_450m_GRADE_5",
                "race_number": None,
                "distance": "450",
                "grade": "Grade 5",
                "data_source": "embedded_form_guide",
            }
        ]
    )

    target = resolve_target_metadata(
        {
            "race_id": "Race 4 - SHEP - 2026-06-15",
            "race_date": "2026-06-15",
            "venue": "SHEP",
        },
        {},
        {"race_number": 4},
        connection,
    )

    assert target["status"] == "MISSING"
    assert target["distance"] is None
    assert target["grade"] is None
    assert target["reason"] == "no_safe_target_metadata"


def test_resolve_target_metadata_fails_closed_on_ambiguous_safe_db_metadata():
    connection = _race_metadata_db(
        [
            {
                "race_id": "safe-1",
                "race_number": None,
                "distance": "390",
                "grade": "Maiden",
                "data_source": "canonical_pre_race_page",
            },
            {
                "race_id": "safe-2",
                "race_number": None,
                "distance": "450",
                "grade": "Grade 5",
                "data_source": "canonical_pre_race_page",
            },
        ]
    )

    target = resolve_target_metadata(
        {
            "race_id": "Race 4 - SHEP - 2026-06-15",
            "race_date": "2026-06-15",
            "venue": "SHEP",
        },
        {},
        {},
        connection,
    )

    assert target["status"] == "AMBIGUOUS"
    assert target["distance"] is None
    assert target["grade"] is None
    assert target["reason"] == "ambiguous_safe_metadata_candidates:2"


def test_resolve_target_metadata_does_not_reuse_unsafe_packet_race_metadata_grade():
    connection = _race_metadata_db([])

    target = resolve_target_metadata(
        {
            "race_id": "Race 4 - SHEP - 2026-06-15",
            "race_date": "2026-06-15",
            "venue": "SHEP",
        },
        {
            "target_grade_safe": "Grade 5",
            "target_grade_source": "race_metadata.grade",
        },
        {},
        connection,
    )

    assert target["status"] == "MISSING"
    assert target["distance"] is None
    assert target["grade"] is None


def test_packet_reuse_blocks_target_metadata_dependent_features_without_safe_target():
    missing_target = {"status": "MISSING"}
    safe_target = {"status": "SAFE"}

    assert can_reuse_packet_feature("target_grade_safe", missing_target) is False
    assert can_reuse_packet_feature("safe_grade_rank", missing_target) is False
    assert can_reuse_packet_feature("same_distance_same_grade_best_time", missing_target) is False
    assert can_reuse_packet_feature("prior_start_count", missing_target) is True
    assert can_reuse_packet_feature("target_grade_safe", safe_target) is True


def test_target_metadata_recovery_audit_classifies_exact_db_row_without_target_metadata():
    connection = _race_metadata_db(
        [
            {
                "race_id": "Race 4 - SHEP - 2026-06-15",
                "distance": None,
                "grade": None,
                "data_source": "frozen_snapshot",
                "results_status": "resulted",
                "winner_source": "thedogs_official",
                "winner_name": "Known Winner",
            }
        ]
    )
    clean_row = {
        "race_id": "Race 4 - SHEP - 2026-06-15",
        "snapshot_instance_id": "snapshot-1",
        "race_date": "2026-06-15",
        "venue": "SHEP",
        "dog_name": "Runner",
        "box_number": 1,
    }
    target = resolve_target_metadata(clean_row, {}, {"race_number": 4}, connection)

    audit = target_metadata_recovery_audit_row(
        clean_row=clean_row,
        packet_row={},
        snapshot={"race_number": 4},
        target=target,
        connection=connection,
    )

    assert audit["target_metadata_blocker_reason"] == (
        "DATA_MISSING:canonical_exact_db_row_has_no_distance_grade"
    )
    assert audit["db_exact_row_count"] == 1
    assert audit["db_exact_metadata_row_count"] == 0
    assert audit["target_distance_safe_present"] is False
    assert audit["target_grade_safe_present"] is False


def test_target_metadata_recovery_audit_report_counts_safe_and_missing_rows():
    rows = [
        {
            "race_id": "Race 1 - SHEP - 2026-06-15",
            "target_metadata_status_v2": "SAFE",
            "target_metadata_source_v2": "safe_sidecar_metadata",
            "target_metadata_blocker_reason": "SAFE",
        },
        {
            "race_id": "Race 2 - SHEP - 2026-06-15",
            "target_metadata_status_v2": "MISSING",
            "target_metadata_source_v2": "missing",
            "target_metadata_blocker_reason": "DATA_MISSING:canonical_exact_db_row_has_no_distance_grade",
        },
    ]

    report = target_metadata_recovery_audit_report(rows)

    assert report["schema_version"] == "target_metadata_recovery_audit_v1"
    assert report["safe_rows"] == 1
    assert report["data_missing_rows"] == 1
    assert report["data_missing_races"] == 1
    assert report["blocker_counts"][
        "DATA_MISSING:canonical_exact_db_row_has_no_distance_grade"
    ] == 1
