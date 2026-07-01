import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

from scripts import daily_race_ingest_shadow_orchestrator as orchestrator
from scripts.daily_race_ingest_shadow_orchestrator import (
    ROOT,
    assert_daily_output_dir_safe,
    box_distribution_report_from_predictions,
    build_score_live_command,
    classify_candidate_csvs,
    is_thedogs_source_url,
    looks_post_result_source_url,
    parse_jump_datetime,
    prejump_metadata_report_from_classification,
    probability_sum_report_from_predictions,
    score_live_subprocess_env,
    stage_eligible_inputs,
    write_common_reports,
    validate_prejump_sidecar_metadata,
)


def _write_race_csv(path: Path) -> None:
    path.write_text(
        "Dog Name|BOX\n"
        "Alpha Runner|1\n"
        "Bravo Runner|2\n",
        encoding="utf-8",
    )


def _write_complete_sidecar(
    csv_path: Path,
    *,
    race_number: int = 1,
    venue: str = "TEST",
    race_date: str = "2026-06-07",
    distance: str = "350m",
    grade: str = "Grade 5",
    race_time: str = "11:15 AM",
) -> None:
    csv_path.with_name(csv_path.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": f"{race_date}T10:00:00+10:00",
                "metadata_source_url": (
                    "https://www.thedogs.com.au/racing/test/"
                    f"{race_date}/{race_number}/test?trial=false"
                ),
                "race_url": (
                    "https://www.thedogs.com.au/racing/test/"
                    f"{race_date}/{race_number}/test?trial=false"
                ),
                "target_distance": distance,
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": grade,
                "target_grade_source": "canonical_pre_race_page",
                "race_info": {
                    "date": race_date,
                    "venue": venue,
                    "race_number": str(race_number),
                    "race_time": race_time,
                    "url": (
                        "https://www.thedogs.com.au/racing/test/"
                        f"{race_date}/{race_number}/test?trial=false"
                    ),
                },
                "runner_completeness": {
                    "status": "COMPLETE",
                    "participants": [
                        {"box_number": 1, "dog_name": "Alpha Runner"},
                        {"box_number": 2, "dog_name": "Bravo Runner"},
                    ],
                },
                "runner_completeness_after_canonical_alignment": {
                    "status": "COMPLETE",
                    "runner_count": 2,
                    "boxes": [1, 2],
                    "participants": [
                        {"box_number": 1, "dog_name": "Alpha Runner"},
                        {"box_number": 2, "dog_name": "Bravo Runner"},
                    ],
                },
                "canonical_runner_alignment": {
                    "schema_version": "canonical_runner_alignment_v1",
                    "status": "aligned",
                    "reason": None,
                    "canonical_runner_set_status": "available",
                    "canonical_source_url": (
                        "https://www.thedogs.com.au/racing/test/"
                        f"{race_date}/{race_number}/test?trial=false"
                    ),
                    "canonical_runner_count": 2,
                    "prediction_runner_count": 2,
                    "remapped_participants": [],
                    "dropped_participants": [],
                },
            }
        ),
        encoding="utf-8",
    )


def test_classifies_current_future_stale_and_malformed_inputs(tmp_path):
    current = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    future = tmp_path / "Race 2 - TEST - 2026-06-08.csv"
    _write_race_csv(current)
    _write_complete_sidecar(current, race_number=1, race_date="2026-06-07")
    _write_race_csv(future)
    _write_complete_sidecar(future, race_number=2, race_date="2026-06-08")
    _write_race_csv(tmp_path / "Race 3 - TEST - 2026-06-06.csv")
    _write_race_csv(tmp_path / "not-a-race.csv")
    (tmp_path / "Race 4 - TEST - 2026-06-07.csv").write_text(
        "Dog Name,BOX\n"
        "Charlie Runner,3\n",
        encoding="utf-8",
    )

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["eligible_count"] == 2
    assert report["stale_count"] == 1
    assert report["malformed_count"] == 2
    assert {Path(row["path"]).name for row in report["eligible"]} == {
        "Race 1 - TEST - 2026-06-07.csv",
        "Race 2 - TEST - 2026-06-08.csv",
    }
    assert report["stale"][0]["reason"] == "stale_before_current_date"
    assert {
        row["reason"]
        for row in report["malformed"]
    } == {"race_date_not_found", "malformed_current_or_future_csv"}
    assert report["eligible"][0]["sidecar_metadata_report"]["status"] == "PASS"


def test_prejump_sidecar_metadata_report_requires_safe_target_fields_and_runner_list(
    tmp_path,
):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)

    report = validate_prejump_sidecar_metadata(source)

    assert report["status"] == "PASS"
    assert report["target_distance"] == "350m"
    assert report["target_grade"] == "Grade 5"
    assert report["race_date"] == "2026-06-07"
    assert report["venue"] == "TEST"
    assert report["race_number"] == 1
    assert report["metadata_captured_at"] == "2026-06-07T10:00:00+10:00"
    assert report["metadata_capture_timing_status"] == "PRE_JUMP"
    assert report["runner_count"] == 2
    assert report["canonical_runner_alignment_status"] == "aligned"
    assert report["canonical_runner_set_status"] == "available"
    assert report["canonical_runner_alignment_verified"] is True
    assert report["csv_target_runner_count"] == 2
    assert report["csv_sidecar_runner_identity_status"] == "PASS"


def test_prejump_sidecar_metadata_fails_when_sidecar_runner_name_does_not_match_csv(
    tmp_path,
):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    source.write_text(
        "Dog Name|BOX\n"
        "Alpha Runner|1\n"
        "Wrong Runner|2\n",
        encoding="utf-8",
    )
    _write_complete_sidecar(source)

    report = validate_prejump_sidecar_metadata(source)

    assert report["status"] == "FAIL"
    assert report["csv_sidecar_runner_identity_status"] == "FAIL"
    assert "runner_box_name_list_does_not_match_csv_target_rows" in report["fail_reasons"]
    assert "runner_box_name_list_name_mismatch" in report["fail_reasons"]
    assert report["csv_sidecar_runner_identity_mismatches"]["name_mismatches"] == [
        {
            "box_number": 2,
            "csv_dog_name": "Wrong Runner",
            "sidecar_dog_name": "Bravo Runner",
            "csv_identity": "WRONGRUNNER",
            "sidecar_identity": "BRAVORUNNER",
        }
    ]


def test_classification_quarantines_sidecar_runner_list_that_does_not_match_csv(
    tmp_path,
):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    source.write_text(
        "Dog Name|BOX\n"
        "Alpha Runner|1\n"
        "Bravo Runner|3\n",
        encoding="utf-8",
    )
    _write_complete_sidecar(source)

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["eligible_count"] == 0
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert "runner_box_name_list_does_not_match_csv_target_rows" in report["malformed"][0][
        "errors"
    ]
    assert "runner_box_name_list_missing_csv_boxes:2" in report["malformed"][0]["errors"]
    assert "runner_box_name_list_extra_csv_boxes:3" in report["malformed"][0]["errors"]


def test_classification_quarantines_sidecar_runner_list_with_duplicate_identity(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    sidecar = source.with_name(source.name + ".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["runner_completeness_after_canonical_alignment"]["participants"] = [
        {"box_number": 1, "dog_name": "Alpha Runner"},
        {"box_number": 1, "dog_name": "Alpha Runner"},
    ]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["eligible_count"] == 0
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert "runner_box_name_list_duplicate_boxes:1" in report["malformed"][0]["errors"]
    assert "runner_box_name_list_duplicate_dog_names:ALPHARUNNER" in report["malformed"][0][
        "errors"
    ]


def test_prejump_sidecar_metadata_accepts_flat_shadow_metadata_block(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    source.with_name(source.name + ".metadata.json").write_text(
        json.dumps(
            {
                "prejump_shadow_metadata": {
                    "schema_version": "prejump_shadow_metadata_v1",
                    "status": "PASS",
                    "fail_reasons": [],
                    "metadata_is_leakage_safe": True,
                    "race_date": "2026-06-07",
                    "venue": "TEST",
                    "race_number": 1,
                    "jump_time": "11:15 AM",
                    "metadata_captured_at": "2026-06-07T10:00:00+10:00",
                    "distance": "350m",
                    "grade": "Grade 5",
                    "target_distance_safe": "350m",
                    "target_distance_source": "canonical_pre_race_page",
                    "target_grade_safe": "Grade 5",
                    "target_grade_source": "canonical_pre_race_page",
                    "source_url": "https://www.thedogs.com.au/racing/test/2026-06-07/1/test?trial=false",
                    "runner_box_name_list": [
                        {"box_number": 1, "dog_name": "Alpha Runner"},
                        {"box_number": 2, "dog_name": "Bravo Runner"},
                    ],
                    "canonical_final_runner_alignment": {
                        "status": "aligned",
                        "canonical_runner_set_status": "available",
                        "canonical_runner_count": 2,
                        "prediction_runner_count": 2,
                        "source_url": "https://www.thedogs.com.au/racing/test/2026-06-07/1/test?trial=false",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    report = validate_prejump_sidecar_metadata(source)

    assert report["status"] == "PASS"
    assert report["metadata_is_leakage_safe"] is True
    assert report["target_distance"] == "350m"
    assert report["target_grade"] == "Grade 5"
    assert report["race_date"] == "2026-06-07"
    assert report["runner_count"] == 2
    assert report["canonical_runner_alignment_verified"] is True
    assert (
        report["canonical_runner_source_url"]
        == "https://www.thedogs.com.au/racing/test/2026-06-07/1/test?trial=false"
    )


def test_race_date_from_sidecar_reads_nested_prejump_shadow_metadata(tmp_path):
    source = tmp_path / "Race 1 - TEST.csv"
    _write_race_csv(source)
    source.with_name(source.name + ".metadata.json").write_text(
        json.dumps(
            {
                "prejump_shadow_metadata": {
                    "schema_version": "prejump_shadow_metadata_v1",
                    "race_date": "2026-06-07",
                }
            }
        ),
        encoding="utf-8",
    )

    assert orchestrator.race_date_from_sidecar(source) == date(2026, 6, 7)


def test_race_date_from_sidecar_reads_nested_race_info(tmp_path):
    source = tmp_path / "Race 1 - TEST.csv"
    _write_race_csv(source)
    source.with_name(source.name + ".metadata.json").write_text(
        json.dumps({"race_info": {"date": "2026-06-08"}}),
        encoding="utf-8",
    )

    assert orchestrator.race_date_from_sidecar(source) == date(2026, 6, 8)


def test_classification_quarantines_sidecar_without_canonical_runner_alignment(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    sidecar = source.with_name(source.name + ".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.pop("canonical_runner_alignment")
    payload.pop("runner_completeness_after_canonical_alignment")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["eligible_count"] == 0
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert "canonical_runner_alignment_missing" in report["malformed"][0]["errors"]
    assert "canonical_runner_set_not_available" in report["malformed"][0]["errors"]


def test_classification_quarantines_sidecar_without_metadata_capture_timestamp(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    sidecar = source.with_name(source.name + ".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.pop("metadata_captured_at")
    payload.pop("created_at", None)
    payload.get("prejump_shadow_metadata", {}).pop("metadata_captured_at", None)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_prejump_sidecar_metadata(source)
    report = classify_candidate_csvs(
        [tmp_path],
        date(2026, 6, 7),
        current_time=datetime.fromisoformat("2026-06-07T10:30:00+10:00"),
    )

    assert validation["status"] == "FAIL"
    assert "metadata_captured_at_missing" in validation["fail_reasons"]
    assert report["eligible_count"] == 0
    assert "metadata_captured_at_missing" in report["malformed"][0]["errors"]


def test_classification_quarantines_sidecar_captured_after_jump(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source, race_time="11:15 AM")
    sidecar = source.with_name(source.name + ".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["metadata_captured_at"] = "2026-06-07T11:16:00+10:00"
    payload.get("prejump_shadow_metadata", {})[
        "metadata_captured_at"
    ] = "2026-06-07T11:16:00+10:00"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_prejump_sidecar_metadata(source)
    report = classify_candidate_csvs(
        [tmp_path],
        date(2026, 6, 7),
        current_time=datetime.fromisoformat("2026-06-07T10:30:00+10:00"),
    )

    assert validation["status"] == "FAIL"
    assert validation["metadata_capture_timing_status"] == "AFTER_OR_AT_JUMP"
    assert "metadata_captured_at_not_before_jump" in validation["fail_reasons"]
    assert report["eligible_count"] == 0
    assert "metadata_captured_at_not_before_jump" in report["malformed"][0]["errors"]


def test_classification_quarantines_canonical_runner_post_result_source_url(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    sidecar = source.with_name(source.name + ".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["canonical_runner_alignment"][
        "canonical_source_url"
    ] = "https://www.thedogs.com.au/racing/test/2026-06-07/1/results"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_prejump_sidecar_metadata(source)
    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert validation["status"] == "FAIL"
    assert "canonical_runner_source_url_looks_post_result" in validation["fail_reasons"]
    assert validation["canonical_runner_alignment_verified"] is False
    assert report["eligible_count"] == 0
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert "canonical_runner_source_url_looks_post_result" in report["malformed"][0]["errors"]


def test_classification_quarantines_missing_canonical_runner_source_url(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    sidecar = source.with_name(source.name + ".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["canonical_runner_alignment"].pop("canonical_source_url")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_prejump_sidecar_metadata(source)

    assert validation["status"] == "FAIL"
    assert "canonical_runner_source_url_missing" in validation["fail_reasons"]
    assert validation["canonical_runner_alignment_verified"] is False


def test_classification_quarantines_current_future_csv_with_missing_sidecar(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["eligible_count"] == 0
    assert report["malformed_count"] == 1
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert "sidecar_metadata_missing" in report["malformed"][0]["errors"]


def test_classification_ignores_refresh_raw_exports_and_quarantine_dirs(tmp_path):
    accepted = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(accepted)
    _write_complete_sidecar(accepted)
    raw_export = tmp_path / "raw_exports" / "Race 2 - TEST - 2026-06-07.csv"
    raw_export.parent.mkdir()
    raw_export.write_text("Dog Name,BOX\nRaw Runner,1\n", encoding="utf-8")
    quarantine = tmp_path / "quarantine" / "Race 3 - TEST - 2026-06-07.csv"
    quarantine.parent.mkdir()
    quarantine.write_text("Dog Name,BOX\nBad Runner,1\n", encoding="utf-8")

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["scanned_csv_count"] == 1
    assert report["eligible_count"] == 1
    assert report["malformed_count"] == 0
    assert report["eligible"][0]["basename"] == accepted.name


def test_classification_quarantines_unsafe_sidecar_target_sources(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    payload = json.loads(source.with_name(source.name + ".metadata.json").read_text())
    payload["target_distance_source"] = "result_page"
    payload["target_grade_source"] = "embedded_form_history:G"
    source.with_name(source.name + ".metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["eligible_count"] == 0
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert "target_distance_missing_or_unsafe" in report["malformed"][0]["errors"]
    assert "target_grade_missing_or_unsafe" in report["malformed"][0]["errors"]


def test_classification_quarantines_non_thedogs_source_url(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    payload = json.loads(source.with_name(source.name + ".metadata.json").read_text())
    payload["metadata_source_url"] = "https://example.com/racing/test/2026-06-07/1"
    payload["race_url"] = "https://example.com/racing/test/2026-06-07/1"
    payload["race_info"]["url"] = "https://example.com/racing/test/2026-06-07/1"
    source.with_name(source.name + ".metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["eligible_count"] == 0
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert "source_url_not_thedogs" in report["malformed"][0]["errors"]


def test_classification_quarantines_thedogs_post_result_source_url(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    payload = json.loads(source.with_name(source.name + ".metadata.json").read_text())
    result_url = "https://www.thedogs.com.au/racing/test/2026-06-07/1/results?trial=false"
    payload["metadata_source_url"] = result_url
    payload["race_url"] = result_url
    payload["race_info"]["url"] = result_url
    payload["prejump_shadow_metadata"] = {
        "schema_version": "prejump_shadow_metadata_v1",
        "status": "PASS",
        "metadata_is_leakage_safe": True,
        "source_url": result_url,
        "race_date": "2026-06-07",
        "venue": "TEST",
        "race_number": 1,
        "jump_time": "11:15 AM",
        "distance": "350m",
        "grade": "Grade 5",
        "target_distance_safe": "350m",
        "target_distance_source": "canonical_pre_race_page",
        "target_grade_safe": "Grade 5",
        "target_grade_source": "canonical_pre_race_page",
        "runner_box_name_list": [
            {"box_number": 1, "dog_name": "Alpha Runner"},
            {"box_number": 2, "dog_name": "Bravo Runner"},
        ],
        "canonical_final_runner_alignment": {
            "status": "aligned",
            "canonical_runner_set_status": "available",
            "canonical_runner_count": 2,
            "prediction_runner_count": 2,
            "source_url": result_url,
        },
    }
    source.with_name(source.name + ".metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    report = classify_candidate_csvs([tmp_path], date(2026, 6, 7))

    assert report["eligible_count"] == 0
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert "source_url_looks_post_result" in report["malformed"][0]["errors"]


def test_classification_excludes_same_day_files_after_jump_time(tmp_path):
    jumped = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    future = tmp_path / "Race 2 - TEST - 2026-06-07.csv"
    _write_race_csv(jumped)
    _write_complete_sidecar(jumped, race_number=1, race_time="11:15 AM")
    _write_race_csv(future)
    _write_complete_sidecar(future, race_number=2, race_time="12:45 PM")

    report = classify_candidate_csvs(
        [tmp_path],
        date(2026, 6, 7),
        current_time=datetime.fromisoformat("2026-06-07T12:00:00+10:00"),
    )

    assert report["eligible_count"] == 1
    assert report["stale_count"] == 1
    assert Path(report["eligible"][0]["path"]).name == "Race 2 - TEST - 2026-06-07.csv"
    assert report["stale"][0]["reason"] == "stale_after_jump_time"
    assert report["stale"][0]["jump_datetime"] == "2026-06-07T11:15:00+10:00"


def test_classification_quarantines_unparseable_same_day_jump_time(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source, race_time="later")

    validation = validate_prejump_sidecar_metadata(source)
    report = classify_candidate_csvs(
        [tmp_path],
        date(2026, 6, 7),
        current_time=datetime.fromisoformat("2026-06-07T12:00:00+10:00"),
    )

    assert validation["status"] == "FAIL"
    assert validation["metadata_capture_timing_status"] == (
        "UNVERIFIED:jump_time_unparseable"
    )
    assert (
        "metadata_capture_timing_unverified:jump_time_unparseable"
        in validation["fail_reasons"]
    )
    assert report["eligible_count"] == 0
    assert report["malformed_count"] == 1
    assert report["malformed"][0]["reason"] == "prejump_sidecar_metadata_failed"
    assert (
        "metadata_capture_timing_unverified:jump_time_unparseable"
        in report["malformed"][0]["errors"]
    )


def test_parse_jump_datetime_accepts_sidecar_local_time():
    parsed, error = parse_jump_datetime(
        race_date=date(2026, 6, 7),
        jump_time="2:07 PM",
        current_time=datetime.fromisoformat("2026-06-07T12:00:00+10:00"),
    )

    assert error is None
    assert parsed.isoformat() == "2026-06-07T14:07:00+10:00"


def test_thedogs_source_url_validator_accepts_only_thedogs_hosts():
    assert is_thedogs_source_url("https://www.thedogs.com.au/racing/test/2026-06-07/1")
    assert is_thedogs_source_url("https://form.thedogs.com.au/racing/test/2026-06-07/1")
    assert not is_thedogs_source_url("https://example.com/racing/test/2026-06-07/1")
    assert not is_thedogs_source_url("file:///tmp/Race-1.csv")


def test_post_result_source_url_detector_flags_result_routes():
    assert looks_post_result_source_url(
        "https://www.thedogs.com.au/racing/test/2026-06-07/1/results?trial=false"
    )
    assert looks_post_result_source_url(
        "https://www.thedogs.com.au/racing/test/2026-06-07/1/dividends"
    )
    assert not looks_post_result_source_url(
        "https://www.thedogs.com.au/racing/test/2026-06-07/1/test?trial=false"
    )


def test_prejump_metadata_report_summarizes_required_eligible_fields(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)

    classification = classify_candidate_csvs([tmp_path], date(2026, 6, 7))
    report = prejump_metadata_report_from_classification(classification)

    assert report["status"] == "PASS"
    assert report["eligible_count"] == 1
    assert report["eligible_with_verified_prejump_metadata"] == 1
    assert report["field_coverage"]["race_date"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["venue"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["race_number"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["jump_time"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["metadata_captured_at"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["target_distance"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["target_grade"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["source_url"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["runner_box_name_list"]["eligible_present_rows"] == 1
    assert report["field_coverage"]["csv_sidecar_runner_identity"][
        "eligible_present_rows"
    ] == 1
    assert report["field_coverage"]["canonical_final_runner_alignment"][
        "eligible_present_rows"
    ] == 1
    assert report["field_coverage"]["canonical_runner_source_url"][
        "eligible_present_rows"
    ] == 1
    assert report["target_metadata_readiness"]["status"] == (
        "TARGET_METADATA_READY_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
    )
    assert report["target_metadata_readiness"]["target_metadata_capture_status"] == "READY"
    assert report["target_metadata_readiness"]["all_current_future_inputs_verified"] is True
    assert report["target_metadata_readiness"]["blocker_counts"] == {}
    assert report["target_metadata_readiness"]["future_train_row_target_metadata_status"] == (
        "PRE_RACE_SIDECAR_SAFE"
    )


def test_prejump_metadata_report_fails_when_sidecar_metadata_is_quarantined(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)

    classification = classify_candidate_csvs([tmp_path], date(2026, 6, 7))
    report = prejump_metadata_report_from_classification(classification)

    assert report["status"] == "FAIL"
    assert report["eligible_count"] == 0
    assert report["malformed_prejump_metadata_count"] == 1
    assert report["unsafe_or_incomplete_metadata"][0]["fail_reasons"] == [
        "sidecar_metadata_missing"
    ]
    readiness = report["target_metadata_readiness"]
    assert readiness["status"] == "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS"
    assert readiness["target_metadata_capture_status"] == "BLOCKED"
    assert readiness["blocker_counts"] == {"sidecar_metadata_missing": 1}
    assert readiness["missing_required_field_counts"] == {
        field: 1 for field in orchestrator.REQUIRED_PREJUMP_METADATA_FIELDS
    }


def test_prejump_metadata_report_blocks_missing_canonical_runner_source_url(tmp_path):
    source = tmp_path / "Race 1 - TEST - 2026-06-07.csv"
    _write_race_csv(source)
    _write_complete_sidecar(source)
    sidecar_path = source.with_name(source.name + ".metadata.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["canonical_runner_alignment"].pop("canonical_source_url")
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    classification = classify_candidate_csvs([tmp_path], date(2026, 6, 7))
    report = prejump_metadata_report_from_classification(classification)

    assert report["status"] == "FAIL"
    assert report["field_coverage"]["canonical_runner_source_url"][
        "eligible_present_rows"
    ] == 0
    readiness = report["target_metadata_readiness"]
    assert readiness["status"] == "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS"
    assert readiness["missing_required_field_counts"]["canonical_runner_source_url"] == 1
    assert readiness["blocker_counts"]["canonical_runner_source_url_missing"] == 1


def test_prejump_metadata_report_waits_when_no_current_future_inputs():
    classification = {
        "eligible": [],
        "malformed": [],
        "stale": [],
    }

    report = prejump_metadata_report_from_classification(classification)

    assert report["status"] == "PASS"
    assert report["target_metadata_readiness"]["status"] == (
        "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
    )
    assert report["target_metadata_readiness"]["target_metadata_capture_status"] == "WAITING"
    assert report["target_metadata_readiness"]["blocker_counts"] == {}


def test_common_reports_write_waiting_same_distance_history_provenance(tmp_path):
    output_dir = tmp_path / "daily"
    output_dir.mkdir()

    write_common_reports(
        output_dir=output_dir,
        final_status=orchestrator.FINAL_STATUS_WAITING,
        classification={
            "schema_version": "daily_shadow_input_classification_v1",
            "scanned_csv_count": 0,
            "eligible_count": 0,
            "stale_count": 0,
            "malformed_count": 0,
            "eligible": [],
            "stale": [],
            "malformed": [],
        },
        db_report={
            "status": "PASS",
            "quick_check": "ok",
            "official_races": 214,
            "official_dog_rows": 1493,
        },
        protected={"protected_paths_unchanged": True},
        predictions=[],
        score_output_dir=None,
        generated_at=datetime.fromisoformat("2026-06-09T00:12:00+10:00"),
        mode="full-dry-run",
        all_missing_train_policy="quarantine_feature",
        shadow_model=None,
    )

    report = json.loads(
        (output_dir / "same_distance_same_grade_history_provenance.json").read_text()
    )
    assert report["status"] == "NOT_POPULATED"
    assert report["live_input_status"] == "NO_ELIGIBLE_PREJUMP_RACES"
    assert report["target_race_rows_allowed"] == 0
    assert report["post_outcome_rows_allowed"] == 0
    assert report["by_feature"]["same_distance_same_grade_best_time"]["status"] == "NOT_POPULATED"


def test_waiting_matrix_reports_mark_data_missing_without_raising(tmp_path):
    output_dir = tmp_path / "daily"
    output_dir.mkdir()

    orchestrator.write_matrix_reports_for_waiting(
        output_dir=output_dir,
        clean_dataset=tmp_path / "missing_clean.jsonl",
        repaired_packet=tmp_path / "missing_repaired.csv",
        schema_path=tmp_path / "missing_schema.json",
        db_path=tmp_path / "missing.db",
        all_missing_train_policy="quarantine_feature",
    )

    for name in (
        "feature_population_report.json",
        "shadow_feature_matrix_audit.json",
        "train_eval_feature_parity_report.json",
        "inactive_feature_policy_report.json",
    ):
        report = json.loads((output_dir / name).read_text(encoding="utf-8"))
        assert report["status"] == "DATA_MISSING"
        assert report["reason"] == "waiting_feature_matrix_inputs_missing"
        assert report["live_input_status"] == "NO_ELIGIBLE_PREJUMP_RACES"
        assert report["shadow_scoring_allowed"] is False
        assert {row["name"] for row in report["missing_inputs"]} == {
            "clean_dataset",
            "repaired_packet",
            "schema",
            "db",
        }


def test_main_waits_when_no_eligible_inputs_and_matrix_inputs_missing(
    tmp_path,
    monkeypatch,
):
    repo_root = tmp_path / "repo"
    input_dir = tmp_path / "empty_refreshed_upcoming"
    output_dir = (
        repo_root
        / "artifacts/full_evidence_orchestration_20260525"
        / "daily_race_ingest_shadow_empty_input"
    )
    input_dir.mkdir(parents=True)

    monkeypatch.setattr(orchestrator, "ROOT", repo_root)
    monkeypatch.setattr(
        orchestrator,
        "verify_db_state",
        lambda _db: {
            "status": "PASS",
            "quick_check": "ok",
            "official_races": 214,
            "official_dog_rows": 1493,
        },
    )
    monkeypatch.setattr(orchestrator, "protected_path_snapshot", lambda: {})
    monkeypatch.setattr(
        orchestrator,
        "protected_path_verification",
        lambda _before: {"protected_paths_unchanged": True},
    )
    monkeypatch.setattr(
        orchestrator,
        "output_file_manifest",
        lambda output_path: {
            "schema_version": "test_manifest_v1",
            "output_dir": str(output_path),
            "artifact_files": {},
        },
    )

    status = orchestrator.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-20T13:47:11+10:00",
            "--db",
            str(tmp_path / "missing.db"),
            "--clean-dataset",
            str(tmp_path / "missing_clean.jsonl"),
            "--repaired-packet",
            str(tmp_path / "missing_repaired.csv"),
            "--schema",
            str(tmp_path / "missing_schema.json"),
        ]
    )

    assert status == 0
    assert (output_dir / "final_status.txt").read_text(encoding="utf-8").strip() == (
        orchestrator.FINAL_STATUS_WAITING
    )
    assert not (output_dir / "daily_shadow_runtime_error.json").exists()
    manifest = json.loads((output_dir / "shadow_manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_status"] == orchestrator.FINAL_STATUS_WAITING
    assert manifest["input_summary"]["eligible_count"] == 0
    assert manifest["prediction_rows"] == 0
    matrix_status = json.loads(
        (output_dir / "shadow_feature_matrix_audit.json").read_text(encoding="utf-8")
    )
    assert matrix_status["status"] == "DATA_MISSING"


def test_common_reports_copy_score_live_same_distance_history_provenance(tmp_path):
    output_dir = tmp_path / "daily"
    score_output_dir = tmp_path / "score"
    output_dir.mkdir()
    score_output_dir.mkdir()
    source_report = {
        "schema_version": "same_distance_same_grade_history_provenance_v1",
        "status": "PASS",
        "by_feature": {
            "same_distance_same_grade_best_time": {"status": "PASS"},
            "same_distance_same_grade_avg_time": {"status": "PASS"},
        },
    }
    (score_output_dir / "same_distance_same_grade_history_provenance.json").write_text(
        json.dumps(source_report),
        encoding="utf-8",
    )

    write_common_reports(
        output_dir=output_dir,
        final_status=orchestrator.FINAL_STATUS_FORWARD_COMPLETE,
        classification={
            "schema_version": "daily_shadow_input_classification_v1",
            "scanned_csv_count": 1,
            "eligible_count": 1,
            "stale_count": 0,
            "malformed_count": 0,
            "eligible": [],
            "stale": [],
            "malformed": [],
        },
        db_report={
            "status": "PASS",
            "quick_check": "ok",
            "official_races": 214,
            "official_dog_rows": 1493,
        },
        protected={"protected_paths_unchanged": True},
        predictions=[],
        score_output_dir=score_output_dir,
        generated_at=datetime.fromisoformat("2026-06-09T00:12:00+10:00"),
        mode="full-dry-run",
        all_missing_train_policy="quarantine_feature",
        shadow_model=Path("model.joblib"),
    )

    copied = json.loads(
        (output_dir / "same_distance_same_grade_history_provenance.json").read_text()
    )
    assert copied == source_report


def test_probability_sum_report_requires_normalized_race_probabilities():
    predictions = [
        {
            "race_id": "race-a",
            "shadow_rf_calibrated_probability": 0.25,
            "box": 1,
            "predicted_rank": 2,
        },
        {
            "race_id": "race-a",
            "shadow_rf_calibrated_probability": 0.75,
            "box": 2,
            "predicted_rank": 1,
        },
        {
            "race_id": "race-b",
            "shadow_rf_calibrated_probability": 1.0,
            "box": 1,
            "predicted_rank": 1,
        },
    ]

    report = probability_sum_report_from_predictions(predictions)

    assert report["status"] == "PASS"
    assert report["race_count"] == 2
    assert report["max_abs_error"] == 0.0


def test_box_distribution_report_tracks_top_pick_box_share():
    predictions = [
        {"race_id": "race-a", "box": 1, "predicted_rank": 1},
        {"race_id": "race-a", "box": 2, "predicted_rank": 2},
        {"race_id": "race-b", "box": 8, "predicted_rank": 1},
    ]

    report = box_distribution_report_from_predictions(predictions)

    assert report["top_pick_count"] == 2
    assert report["box1_top_pick_share"] == 0.5
    assert report["top_pick_box_counts"] == {"1": 1, "8": 1}


def test_score_live_command_is_shadow_training_with_quarantine_policy(tmp_path):
    command = build_score_live_command(
        input_dir=tmp_path / "eligible_inputs",
        output_dir=tmp_path / "shadow_score_live",
        db_path=Path("greyhound_racing_data.db"),
        schema_path=Path("schema.json"),
        clean_dataset=Path("clean.jsonl"),
        repaired_packet=Path("packet.csv"),
        all_missing_train_policy="quarantine_feature",
    )

    assert "score-live" in command
    assert "--train-if-missing" in command
    assert command[command.index("--all-missing-train-policy") + 1] == "quarantine_feature"
    assert command[command.index("--db") + 1] == "greyhound_racing_data.db"


def test_score_live_command_reuses_shadow_model_without_training(tmp_path):
    command = build_score_live_command(
        input_dir=tmp_path / "eligible_inputs",
        output_dir=tmp_path / "shadow_score_live",
        db_path=Path("greyhound_racing_data.db"),
        schema_path=Path("schema.json"),
        clean_dataset=Path("clean.jsonl"),
        repaired_packet=Path("packet.csv"),
        all_missing_train_policy="quarantine_feature",
        shadow_model=Path("artifacts/shadow/model.joblib"),
    )

    assert "--model" in command
    assert command[command.index("--model") + 1] == "artifacts/shadow/model.joblib"
    assert "--train-if-missing" not in command


def test_score_live_command_passes_retained_evidence_root(tmp_path):
    evidence_root = (
        tmp_path.parent
        / f"{tmp_path.name}_retained"
        / "artifacts/full_evidence_orchestration_20260525"
    )

    command = build_score_live_command(
        input_dir=evidence_root / "daily_race_ingest_shadow_x" / "eligible_inputs",
        output_dir=evidence_root / "daily_race_ingest_shadow_x" / "shadow_score_live",
        db_path=Path("greyhound_racing_data.db"),
        schema_path=Path("schema.json"),
        clean_dataset=Path("clean.jsonl"),
        repaired_packet=Path("packet.csv"),
        all_missing_train_policy="quarantine_feature",
        evidence_root=evidence_root,
    )

    assert "--evidence-root" in command
    assert command[command.index("--evidence-root") + 1] == str(evidence_root)


def test_score_live_command_auto_falls_back_to_uv_when_current_python_lacks_ml_deps(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(orchestrator, "shadow_ml_dependencies_available", lambda: False)
    monkeypatch.setattr(orchestrator.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    command = build_score_live_command(
        input_dir=tmp_path / "eligible_inputs",
        output_dir=tmp_path / "shadow_score_live",
        db_path=Path("greyhound_racing_data.db"),
        schema_path=Path("schema.json"),
        clean_dataset=Path("clean.jsonl"),
        repaired_packet=Path("packet.csv"),
        all_missing_train_policy="quarantine_feature",
    )

    assert command[:2] == ["/usr/bin/uv", "run"]
    assert "--with" in command
    assert "joblib" in command
    assert "scikit-learn" not in command
    assert f"scikit-learn=={orchestrator.SHADOW_MODEL_SKLEARN_VERSION}" in command
    assert "numpy" in command
    assert "requests" in command
    assert "python" in command
    assert command[command.index("python") + 1].endswith(
        "scripts/run_shadow_non_tgr_rf_evaluation.py"
    )


def test_score_live_subprocess_env_removes_repo_root_pythonpath():
    env = score_live_subprocess_env(
        {
            "PYTHONPATH": f"{ROOT}:/keep/me",
            "OTHER": "1",
        }
    )

    assert env["PYTHONPATH"] == "/keep/me"
    assert env["OTHER"] == "1"


def test_daily_manifest_records_generated_and_score_live_timing(tmp_path):
    score_dir = tmp_path / "shadow_score_live"
    output_dir = tmp_path / "daily_shadow"
    score_dir.mkdir()
    output_dir.mkdir()
    (score_dir / "shadow_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "shadow_live_scoring_manifest_v1",
                "generated_at": "2026-06-09T00:10:00+10:00",
                "prediction_timestamp": "2026-06-09T00:10:00+10:00",
                "feature_freeze_timestamp": "2026-06-09T00:05:00+10:00",
                "stage2_forward_shadow_status": "STAGE2_FORWARD_SHADOW_COLLECTING",
            }
        ),
        encoding="utf-8",
    )
    (score_dir / "shadow_feature_rows.json").write_text(
        json.dumps(
            [
                {
                    "race_id": "Race 1 - TEST - 2026-06-09",
                    "source_csv": "upcoming/Race 1 - TEST - 2026-06-09.csv",
                }
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "stage2_shadow_predictions.jsonl").write_text(
        json.dumps({"race_id": "Race 1 - TEST - 2026-06-09", "box": 1}) + "\n",
        encoding="utf-8",
    )

    orchestrator.write_manifest(
        output_dir=output_dir,
        generated_at=datetime.fromisoformat("2026-06-09T00:12:00+10:00"),
        mode="full-dry-run",
        db_report={
            "status": "PASS",
            "quick_check": "ok",
            "official_races": 214,
            "official_dog_rows": 1493,
        },
        classification={"scanned_csv_count": 1, "eligible_count": 1, "stale_count": 0, "malformed_count": 0},
        protected={"protected_paths_unchanged": True},
        predictions=[{"race_id": "Race 1 - TEST - 2026-06-09"}],
        score_output_dir=score_dir,
        final_status=orchestrator.FINAL_STATUS_FORWARD_COMPLETE,
        all_missing_train_policy="quarantine_feature",
        shadow_model=Path("artifacts/shadow/model.joblib"),
    )

    manifest = json.loads((tmp_path / "daily_shadow" / "shadow_manifest.json").read_text())
    assert manifest["generated_at"] == "2026-06-09T00:12:00+10:00"
    assert manifest["prediction_timestamp"] == "2026-06-09T00:10:00+10:00"
    assert manifest["feature_freeze_timestamp"] == "2026-06-09T00:05:00+10:00"
    assert manifest["score_live_manifest"]["feature_freeze_timestamp"] == (
        "2026-06-09T00:05:00+10:00"
    )
    assert manifest["stage2_shadow_predictions_jsonl"].endswith(
        "daily_shadow/stage2_shadow_predictions.jsonl"
    )
    assert manifest["stage2_prediction_rows"] == 1
    assert manifest["stage2_forward_shadow_status"] == "STAGE2_FORWARD_SHADOW_COLLECTING"
    assert manifest["shadow_feature_rows_json"].endswith(
        "daily_shadow/shadow_feature_rows.json"
    )
    assert (output_dir / "shadow_feature_rows.json").exists()


def test_staging_uses_explicit_source_path_and_copies_sidecar(tmp_path):
    source = tmp_path / "input" / "Race 1 - TEST - 2026-06-07.csv"
    source.parent.mkdir()
    _write_race_csv(source)
    sidecar = source.with_name(source.name + ".metadata.json")
    sidecar.write_text('{"race_date": "2026-06-07"}', encoding="utf-8")

    staged = stage_eligible_inputs(
        {
            "eligible": [
                {
                    "path": "does/not/exist.csv",
                    "source_path": str(source),
                    "basename": source.name,
                }
            ]
        },
        tmp_path / "stage",
    )

    assert len(staged) == 1
    assert staged[0].read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert staged[0].with_name(staged[0].name + ".metadata.json").read_text(
        encoding="utf-8"
    ) == sidecar.read_text(encoding="utf-8")


def test_daily_output_guard_accepts_only_shadow_artifact_prefix():
    output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/"
        "daily_race_ingest_shadow_20260607T220000+1000"
    )

    assert assert_daily_output_dir_safe(output_dir).name == (
        "daily_race_ingest_shadow_20260607T220000+1000"
    )


def test_daily_output_guard_accepts_configured_external_output_parent(tmp_path):
    evidence_root = tmp_path / "runtime_artifacts" / "full_evidence_orchestration_20260525"
    output_dir = evidence_root / "daily_race_ingest_shadow_external"

    assert (
        assert_daily_output_dir_safe(output_dir, output_parent=evidence_root)
        == output_dir.absolute()
    )

    with pytest.raises(ValueError, match="output_dir_must_be_daily_shadow_artifact"):
        assert_daily_output_dir_safe(
            evidence_root / "not_a_daily_shadow_artifact",
            output_parent=evidence_root,
        )


def test_daily_output_guard_rejects_production_paths():
    with pytest.raises(ValueError, match="output_dir_must_be_daily_shadow_artifact"):
        assert_daily_output_dir_safe(Path("model_registry/daily_race_ingest_shadow_bad"))


def test_daily_output_guard_rejects_parent_traversal():
    with pytest.raises(ValueError, match="output_dir_must_not_contain_parent_traversal"):
        assert_daily_output_dir_safe(
            Path(
                "artifacts/full_evidence_orchestration_20260525/"
                "daily_race_ingest_shadow_bad/../../prediction_snapshots"
            )
        )


def test_refresh_script_help_prefers_repo_utils_package_when_pythonpath_set():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/refresh_prejump_upcoming.py"), "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Refresh current pre-jump TheDogs form guides" in result.stdout
