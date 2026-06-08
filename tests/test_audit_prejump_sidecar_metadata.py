import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import audit_prejump_sidecar_metadata as audit


def _write_csv(path: Path) -> None:
    path.write_text(
        "Dog Name|BOX\n"
        "Alpha Runner|1\n"
        "Bravo Runner|2\n",
        encoding="utf-8",
    )


def _write_sidecar(
    csv_path: Path,
    *,
    status: str = "PASS",
    distance: str = "350m",
    race_date: str = "2026-06-08",
    jump_time: str = "8:40 PM",
) -> None:
    csv_path.with_name(csv_path.name + ".metadata.json").write_text(
        json.dumps(
            {
                "prejump_shadow_metadata": {
                    "schema_version": "prejump_shadow_metadata_v1",
                    "status": status,
                    "fail_reasons": [] if status == "PASS" else ["target_distance_missing_or_unsafe"],
                    "metadata_is_leakage_safe": True,
                    "race_date": race_date,
                    "venue": "TEST",
                    "race_number": 1,
                    "jump_time": jump_time,
                    "metadata_captured_at": f"{race_date}T12:00:00+10:00",
                    "distance": distance,
                    "grade": "Grade 5",
                    "target_distance_safe": distance,
                    "target_distance_source": "canonical_pre_race_page",
                    "target_grade_safe": "Grade 5",
                    "target_grade_source": "canonical_pre_race_page",
                    "source_url": "https://www.thedogs.com.au/racing/test/2026-06-08/1/test?trial=false",
                    "runner_box_name_list": [
                        {"box_number": 1, "dog_name": "Alpha Runner"},
                        {"box_number": 2, "dog_name": "Bravo Runner"},
                    ],
                    "canonical_final_runner_alignment": {
                        "status": "aligned",
                        "canonical_runner_set_status": "available",
                        "canonical_runner_count": 2,
                        "prediction_runner_count": 2,
                        "source_url": "https://www.thedogs.com.au/racing/test/2026-06-08/1/test?trial=false",
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_audit_prejump_sidecar_metadata_passes_complete_nested_contract(tmp_path):
    csv_path = tmp_path / "Race 1 - TEST - 2026-06-08.csv"
    _write_csv(csv_path)
    _write_sidecar(csv_path)

    report = audit.audit_sidecars(
        tmp_path,
        generated_at=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        current_time=datetime.fromisoformat("2026-06-08T12:00:00+10:00"),
    )

    assert report["final_status"] == audit.FINAL_PASS
    assert report["collection_status"] == "PREJUMP_INPUTS_READY"
    assert report["csv_count"] == 1
    assert report["pass_count"] == 1
    assert report["fail_count"] == 0
    assert report["current_or_future_prejump_count"] == 1
    assert report["current_or_future_prejump_pass_count"] == 1
    assert report["stale_count"] == 0
    required = report["records"][0]["required_fields"]
    assert required == {
        "race_date": "2026-06-08",
        "venue": "TEST",
        "race_number": 1,
        "jump_time": "8:40 PM",
        "metadata_captured_at": "2026-06-08T12:00:00+10:00",
        "distance": "350m",
        "grade": "Grade 5",
        "source_url": "https://www.thedogs.com.au/racing/test/2026-06-08/1/test?trial=false",
        "canonical_runner_source_url": "https://www.thedogs.com.au/racing/test/2026-06-08/1/test?trial=false",
        "runner_count": 2,
    }
    assert report["records"][0]["contract_field_presence"] == {
        "race_date": True,
        "venue": True,
        "race_number": True,
        "jump_time": True,
        "metadata_captured_at": True,
        "target_distance": True,
        "target_grade": True,
        "source_url": True,
        "runner_box_name_list": True,
        "csv_sidecar_runner_identity": True,
        "canonical_final_runner_alignment": True,
        "canonical_runner_source_url": True,
    }
    assert report["records"][0]["missing_contract_fields"] == []
    assert report["records"][0]["source_url_is_thedogs"] is True
    assert report["records"][0]["runner_box_name_list"] == [
        {"box_number": 1, "dog_name": "Alpha Runner"},
        {"box_number": 2, "dog_name": "Bravo Runner"},
    ]
    assert report["records"][0]["csv_sidecar_runner_identity_status"] == "PASS"
    assert report["records"][0]["canonical_runner_alignment_verified"] is True
    assert report["records"][0]["freshness"]["status"] == "current_prejump"
    assert report["records"][0]["freshness"]["is_current_or_future_prejump"] is True
    assert report["target_metadata_readiness"]["status"] == (
        "TARGET_METADATA_READY_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
    )
    assert report["target_metadata_readiness"]["target_metadata_capture_status"] == "READY"
    assert report["target_metadata_readiness"]["blocker_counts"] == {}
    assert report["no_write_guarantees"]["db_write"] is False


def test_audit_prejump_sidecar_metadata_fails_missing_sidecar(tmp_path):
    csv_path = tmp_path / "Race 1 - TEST - 2026-06-08.csv"
    _write_csv(csv_path)

    report = audit.audit_sidecars(
        tmp_path,
        current_time=datetime.fromisoformat("2026-06-08T12:00:00+10:00"),
    )

    assert report["final_status"] == audit.FINAL_FAIL
    assert report["collection_status"] == "PREJUMP_INPUTS_BLOCKED_BY_METADATA"
    assert report["csv_count"] == 1
    assert report["pass_count"] == 0
    assert report["fail_count"] == 1
    assert report["current_or_future_prejump_count"] == 1
    assert report["current_or_future_input_count"] == 1
    assert report["current_or_future_prejump_pass_count"] == 0
    assert "sidecar_metadata_missing" in report["records"][0]["errors"]
    assert report["records"][0]["missing_contract_fields"] == [
        "race_date",
        "venue",
        "race_number",
        "jump_time",
        "metadata_captured_at",
        "target_distance",
        "target_grade",
        "source_url",
        "runner_box_name_list",
        "csv_sidecar_runner_identity",
        "canonical_final_runner_alignment",
        "canonical_runner_source_url",
    ]
    assert report["records"][0]["freshness"]["status"] == "current_date_jump_time_missing"
    assert report["records"][0]["freshness"]["race_date"] == "2026-06-08"
    assert report["records"][0]["freshness"]["race_date_source"] == "filename"
    assert report["records"][0]["freshness"]["is_current_or_future_input"] is True
    assert report["target_metadata_readiness"]["status"] == (
        "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS"
    )
    assert report["target_metadata_readiness"]["target_metadata_capture_status"] == "BLOCKED"
    assert report["target_metadata_readiness"]["blocker_counts"][
        "sidecar_metadata_missing"
    ] == 1


def test_audit_prejump_sidecar_metadata_separates_contract_pass_from_stale(tmp_path):
    csv_path = tmp_path / "Race 1 - TEST - 2026-06-08.csv"
    _write_csv(csv_path)
    _write_sidecar(csv_path, jump_time="8:40 PM")

    report = audit.audit_sidecars(
        tmp_path,
        current_time=datetime.fromisoformat("2026-06-08T23:45:00+10:00"),
    )

    assert report["final_status"] == audit.FINAL_PASS
    assert report["collection_status"] == "NO_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
    assert report["pass_count"] == 1
    assert report["current_or_future_prejump_count"] == 0
    assert report["current_or_future_prejump_pass_count"] == 0
    assert report["stale_after_jump_time_count"] == 1
    assert report["records"][0]["freshness"]["status"] == "stale_after_jump_time"
    assert report["records"][0]["freshness"]["is_current_or_future_prejump"] is False


def test_audit_prejump_sidecar_metadata_reports_partial_contract_failure(tmp_path):
    csv_path = tmp_path / "Race 1 - TEST - 2026-06-08.csv"
    _write_csv(csv_path)
    _write_sidecar(csv_path)
    sidecar_path = csv_path.with_name(csv_path.name + ".metadata.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    shadow_metadata = payload["prejump_shadow_metadata"]
    shadow_metadata["source_url"] = "https://example.com/racing/test/2026-06-08/1/test"
    shadow_metadata["canonical_final_runner_alignment"]["status"] = "missing"
    shadow_metadata["canonical_final_runner_alignment"]["canonical_runner_set_status"] = "missing"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit.audit_sidecars(
        tmp_path,
        current_time=datetime.fromisoformat("2026-06-08T12:00:00+10:00"),
    )

    record = report["records"][0]
    assert report["final_status"] == audit.FINAL_FAIL
    assert report["collection_status"] == "PREJUMP_INPUTS_BLOCKED_BY_METADATA"
    assert "source_url_not_thedogs" in record["errors"]
    assert "canonical_runner_alignment_not_aligned" in record["errors"]
    assert "canonical_runner_set_not_available" in record["errors"]
    assert record["contract_field_presence"]["source_url"] is False
    assert record["contract_field_presence"]["canonical_final_runner_alignment"] is False
    assert record["missing_contract_fields"] == [
        "target_distance",
        "target_grade",
        "source_url",
        "canonical_final_runner_alignment",
    ]
    assert record["source_url_is_thedogs"] is False
    assert report["target_metadata_readiness"]["status"] == (
        "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS"
    )
    assert report["target_metadata_readiness"]["blocker_counts"][
        "canonical_runner_alignment_not_aligned"
    ] == 1
    assert report["target_metadata_readiness"]["missing_required_field_counts"] == {
        "canonical_final_runner_alignment": 1,
        "source_url": 1,
        "target_distance": 1,
        "target_grade": 1,
    }


def test_audit_writes_report_only_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    csv_path = tmp_path / "inputs/Race 1 - TEST - 2026-06-08.csv"
    csv_path.parent.mkdir()
    _write_csv(csv_path)
    _write_sidecar(csv_path)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "prejump_sidecar_metadata_audit_20260608T120000+1000"
    )

    result = audit.run_audit(
        input_dir=csv_path.parent,
        output_dir=output_dir,
        current_time=datetime.fromisoformat("2026-06-08T12:00:00+10:00"),
    )

    assert result["final_status"] == audit.FINAL_PASS
    assert result["collection_status"] == "PREJUMP_INPUTS_READY"
    assert result["current_or_future_prejump_pass_count"] == 1
    assert (output_dir / "prejump_sidecar_metadata_audit.json").exists()
    report = json.loads((output_dir / "prejump_sidecar_metadata_audit.json").read_text())
    assert report["target_metadata_readiness"]["target_metadata_capture_status"] == "READY"
    assert "Target metadata readiness" in (output_dir / "SUMMARY.md").read_text()
    assert (output_dir / "final_status.txt").read_text(encoding="utf-8").strip() == audit.FINAL_PASS
