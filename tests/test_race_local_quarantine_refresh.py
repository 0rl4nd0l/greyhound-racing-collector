"""An authenticated race-local export quarantine must not stop unrelated races."""

from hashlib import sha256

import pytest

from tests.test_empty_eligible_refresh import report_fixture, publish
from scripts.refresh_prejump_upcoming import current_index_metadata_selection


def mixed_report(root):
    report = report_fixture(root)
    report["upcoming_dir"] = str(root)
    report["quarantine_count"] = 1
    race = {
        **report["selected_races"][0],
        "race_url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/6",
        "race_id": "Race 6 - GUNN - 2026-07-19",
        "race_id_aliases": ["Race 6 - GUNN - 2026-07-19"],
        "race_number": 6,
    }
    content = b"Box|Dog Name\n1|Alpha\n2|Beta\n"
    raw = root / "raw.csv"
    raw.write_bytes(content)
    quarantine = root / "quarantine.csv"
    quarantine.write_bytes(content)
    normalization = {
        "normalization_status": "rejected",
        "normalization_failure_reason": "final_runner_set_not_aligned:canonical_participant_missing_from_source_csv",
        "raw_content_length": len(content),
        "raw_content_sha256": sha256(content).hexdigest(),
        "raw_export_path": str(raw),
        "canonical_runner_alignment": {
            "status": "not_aligned",
            "reason": "canonical_participant_missing_from_source_csv",
            "canonical_runner_set_status": "available",
            "canonical_source_url": race["race_url"],
            "canonical_runner_count": 3,
            "source_runner_count": 2,
            "prediction_runner_count": 0,
            "missing_canonical_participants": [{"box_number": 3, "dog_name": "Gamma"}],
            "duplicate_source_runner_names": [],
            "remapped_participants": [],
            "dropped_participants": [],
            "native_identity_status": "available",
            "source_native_race_id": "16000",
            "native_identity_reasons": [],
        },
    }
    result = {
        "success": False,
        "error": "Downloaded CSV failed canonical final runner-set alignment gate",
        "normalization": normalization,
        "raw_export_path": str(raw),
        "quarantine_path": str(quarantine),
        "runner_completeness": {
            "status": "COMPLETE",
            "reasons": [],
            "runner_count": 2,
            "duplicate_boxes": [],
            "duplicate_dog_names": [],
            "invalid_runner_rows": [],
            "source": "download:" + race["race_url"],
        },
    }
    report["selected_races"].append(race)
    report["selected_count"] = 2
    report["downloads"].append({"race_url": race["race_url"], "success": False, "result": result})
    report["sidecar_metadata_coverage"]["races"].append(
        {
            "race_url": race["race_url"],
            "race_id": race["race_id"],
            "csv_path": None,
            "sidecar_path": None,
            "weather_track_rejected_reasons": ["accepted_csv_missing"],
        }
    )
    reselect(report)
    return report


def reselect(report):
    report["current_index_races"], report["current_index_metadata_selection"] = (
        current_index_metadata_selection(
            report["selected_races"],
            report["sidecar_metadata_coverage"],
            source_generated_at=report["generated_at"],
        )
    )
    report["current_index_race_count"] = len(report["current_index_races"])


def test_complete_attempt_with_local_export_quarantine_publishes_only_empty_eligible_set(tmp_path):
    report = mixed_report(tmp_path)
    result = publish(tmp_path, tmp_path / "runtime/odds.json", report, "empty")
    assert result["status"] == "PUBLISHED" and result["race_count"] == 0
    assert report["downloads"][1]["success"] is False
    assert report["current_index_metadata_selection"]["excluded_race_count"] == 2


def test_all_quarantines_require_validated_shared_snapshot(tmp_path):
    report = mixed_report(tmp_path)
    report["selected_races"] = report["selected_races"][1:]
    report["downloads"] = report["downloads"][1:]
    report["sidecar_metadata_coverage"]["races"] = report["sidecar_metadata_coverage"]["races"][1:]
    report.update(selected_count=1, accepted_csv_count=0, sidecar_count=0)
    reselect(report)
    assert (
        publish(tmp_path, tmp_path / "runtime/odds.json", report, "unproven")["status"]
        == "REJECTED"
    )
    report["shared_sportsbet_snapshot"] = {"status": "VALIDATED", "payload_sha256": "a" * 64}
    result = publish(tmp_path, tmp_path / "runtime/odds.json", report, "proved")
    assert result["status"] == "PUBLISHED" and result["race_count"] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_canonical_race",
        "unknown_error",
        "source_denial",
        "missing_raw",
        "changed_quarantine",
        "duplicate_names",
        "incomplete_download",
        "native_shared_failure",
        "shared_snapshot_failed",
        "discovery_failed",
        "cap_exhausted",
    ],
)
def test_local_quarantine_never_masks_shared_or_unproved_failure(tmp_path, defect):
    report = mixed_report(tmp_path)
    result = report["downloads"][1]["result"]
    normal = result["normalization"]
    alignment = normal["canonical_runner_alignment"]
    if defect == "wrong_canonical_race":
        alignment["canonical_source_url"] = (
            "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/7"
        )
    elif defect == "unknown_error":
        normal["normalization_failure_reason"] = "unrecognized"
    elif defect == "source_denial":
        result["source_http_status"] = 429
    elif defect == "missing_raw":
        result["raw_export_path"] = str(tmp_path / "missing.csv")
    elif defect == "changed_quarantine":
        (tmp_path / "quarantine.csv").write_bytes(b"changed")
    elif defect == "duplicate_names":
        alignment["duplicate_source_runner_names"] = ["Alpha"]
    elif defect == "incomplete_download":
        result["runner_completeness"]["status"] = "INCOMPLETE"
    elif defect == "native_shared_failure":
        alignment.update(
            native_identity_status="unavailable",
            source_native_race_id=None,
            native_identity_reasons=["HTTP_429"],
        )
    elif defect == "shared_snapshot_failed":
        report["shared_sportsbet_snapshot"] = {"status": "UNAVAILABLE"}
    elif defect == "discovery_failed":
        report["discovery_failures"] = [{"error_type": "ReadTimeout"}]
    elif defect == "cap_exhausted":
        report["status"] = "DISCOVERY_FAILED"
        report["reason"] = "REQUEST_CAP_EXHAUSTED"
    assert (
        publish(tmp_path, tmp_path / "runtime/odds.json", report, "rejected")["status"]
        == "REJECTED"
    )


def qualified_mixed_report(root):
    report = mixed_report(root)
    report["sidecar_metadata_coverage"]["races"][0].update(
        safe_weather_present=True,
        safe_track_condition_present=True,
        safe_all_weather_track_expert_form_present=True,
        weather_track_rejected_reasons=[],
    )
    reselect(report)
    report["status"] = "SUCCESS"
    return report


def test_qualified_race_continues_while_exact_local_quarantine_stays_excluded(tmp_path):
    report = qualified_mixed_report(tmp_path)
    result = publish(tmp_path, tmp_path / "runtime/odds.json", report, "qualified")
    assert result["status"] == "PUBLISHED" and result["race_count"] == 1
    assert [race["race_url"] for race in report["current_index_races"]] == [
        report["selected_races"][0]["race_url"]
    ]
    assert report["downloads"][1]["success"] is False


@pytest.mark.parametrize("defect", ["denial", "cap", "unknown", "shared", "discovery"])
def test_qualified_subset_never_publishes_after_unisolated_failure(tmp_path, defect):
    report = qualified_mixed_report(tmp_path)
    failed = report["downloads"][1]["result"]
    if defect == "denial":
        failed["source_http_status"] = 429
    elif defect == "cap":
        failed["source_failure_category"] = "REQUEST_CAP_EXHAUSTED"
    elif defect == "unknown":
        failed["error"] = "unexpected failure"
    elif defect == "shared":
        report["shared_sportsbet_snapshot"] = {"status": "UNAVAILABLE"}
    else:
        report["discovery_failures"] = [{"error_type": "ReadTimeout"}]
    assert (
        publish(tmp_path, tmp_path / "runtime/odds.json", report, "stopped")["status"] == "REJECTED"
    )


@pytest.mark.parametrize("tamper", ["csv", "sidecar", "eligible", "race_id", "race_url"])
def test_local_quarantine_cannot_be_reintroduced_as_eligible(tmp_path, tamper):
    from scripts.refresh_prejump_upcoming import has_unisolated_refresh_failure

    report = qualified_mixed_report(tmp_path)
    if tamper == "eligible":
        report["current_index_races"].append(report["selected_races"][1])
    elif tamper in {"race_id", "race_url"}:
        report["sidecar_metadata_coverage"]["races"][1][tamper] = "wrong race"
    else:
        report["sidecar_metadata_coverage"]["races"][1][tamper + "_path"] = str(
            tmp_path / "invented"
        )
    assert has_unisolated_refresh_failure(report)
    assert (
        publish(tmp_path, tmp_path / "runtime/odds.json", report, "forged")["status"] == "REJECTED"
    )
