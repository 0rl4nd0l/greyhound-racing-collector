"""Semantic field undercoverage is local only with independently replayable evidence."""

import csv
import io
from hashlib import sha256

import pytest
from scripts.refresh_prejump_upcoming import has_unisolated_refresh_failure
from tests.test_race_local_quarantine_refresh import qualified_mixed_report
from tests.test_empty_eligible_refresh import publish
from utils.csv_metadata import THEDOGS_EXPERT_FORM_COLUMNS
from utils.runner_completeness import analyze_csv_text_runner_completeness


def undercovered_report(root, count=2):
    report = qualified_mixed_report(root)
    failed = report["downloads"][1]["result"]
    candidate = report["selected_races"][1]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(THEDOGS_EXPERT_FORM_COLUMNS)
    for box in range(1, count + 1):
        row = [""] * len(THEDOGS_EXPERT_FORM_COLUMNS)
        row[0] = f"{box}. Dog {box}"
        row[THEDOGS_EXPERT_FORM_COLUMNS.index("DATE")] = "2026-07-18"
        writer.writerow(row)
    raw = output.getvalue().encode()
    for key in ("raw_export_path", "quarantine_path"):
        from pathlib import Path

        Path(failed[key]).write_bytes(raw)
    failed["runner_completeness"] = analyze_csv_text_runner_completeness(
        raw.decode(), source="download:" + candidate["race_url"]
    ).as_dict()
    failed["error"] = "Incomplete runner set in downloaded CSV"
    normal = failed["normalization"]
    normal.update(
        raw_content_length=len(raw),
        raw_content_sha256=sha256(raw).hexdigest(),
        normalization_failure_reason="runner_set_not_complete:INCOMPLETE;final_runner_set_not_aligned:canonical_participant_missing_from_source_csv",
        accepted_csv_path=str(root / "Race 6 - GUNN - 2026-07-19.csv"),
        original_delimiter=",",
        delimiter_status="verified",
        normalization_verification={
            "schema_status": "verified",
            "schema_failure_reasons": [],
            "runner_set_status": "INCOMPLETE",
            "target_metadata_status": "verified",
            "target_metadata_failure_reason": None,
            "race_time_mapping_status": "exact_url_match",
            "canonical_url_race_number": 6,
            "capture_race_number": 6,
        },
    )
    normal["canonical_runner_alignment"].update(
        source_runner_count=count,
        canonical_runner_count=4,
        missing_canonical_participants=[
            {"box_number": box, "dog_name": f"Dog {box}"} for box in range(count + 1, 5)
        ],
    )
    return report


@pytest.mark.parametrize("count", [1, 2, 3])
def test_replayable_under_minimum_field_stays_quarantined_without_stopping_eligible_race(
    tmp_path, count
):
    report = undercovered_report(tmp_path, count)
    assert not has_unisolated_refresh_failure(report)
    result = publish(tmp_path, tmp_path / "runtime/odds.json", report, "local")
    assert result["status"] == "PUBLISHED" and result["race_count"] == 1
    assert report["downloads"][1]["success"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "target_identity",
        "parse",
        "fake_count",
        "duplicate_boxes",
        "transport",
        "unknown_reason",
    ],
)
def test_undercoverage_does_not_hide_unverified_or_shared_failure(tmp_path, defect):
    report = undercovered_report(tmp_path)
    failed = report["downloads"][1]["result"]
    normal = failed["normalization"]
    if defect == "schema":
        normal["normalization_verification"]["schema_status"] = "rejected"
    elif defect == "target_identity":
        normal["normalization_verification"]["capture_race_number"] = 7
    elif defect == "fake_count":
        failed["runner_completeness"]["runner_count"] = 3
    elif defect == "duplicate_boxes":
        failed["runner_completeness"]["duplicate_boxes"] = [1]
    elif defect == "transport":
        failed["source_failure_category"] = "ReadTimeout"
    elif defect == "unknown_reason":
        failed["runner_completeness"]["reasons"] = ["unrecognized"]
    else:
        from pathlib import Path

        raw = Path(failed["raw_export_path"]).read_bytes()[:-8]
        for key in ("raw_export_path", "quarantine_path"):
            Path(failed[key]).write_bytes(raw)
        normal.update(raw_content_length=len(raw), raw_content_sha256=sha256(raw).hexdigest())
    assert has_unisolated_refresh_failure(report)
    assert publish(tmp_path, tmp_path / "runtime/odds.json", report, "stop")["status"] == "REJECTED"
