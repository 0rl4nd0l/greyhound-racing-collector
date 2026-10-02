"""Native local rejection components compose; shared failures still veto refreshes."""

from copy import deepcopy
import pytest
from scripts.refresh_prejump_upcoming import has_unisolated_refresh_failure
from tests.test_semantic_runner_quarantine import undercovered_report
from tests.test_final_field_shortfall import aligned_shortfall_report
from tests.test_empty_eligible_refresh import publish


def with_missing_metadata(report, reasons):
    normal = report["downloads"][1]["result"]["normalization"]
    verification = normal["normalization_verification"]
    verification.update(
        target_metadata_status="missing",
        target_metadata_failure_reason=";".join(reasons),
        race_time_source="canonical_race_url",
    )
    normal["normalization_failure_reason"] += (
        (";" if normal["normalization_failure_reason"] else "")
        + "target_metadata_not_verified:"
        + ";".join(reasons)
    )
    return report


def composed_report(root, monkeypatch, stage="raw", reasons=("missing_target_grade",)):
    if stage in {"final", "metadata_only"}:
        report = aligned_shortfall_report(
            root, monkeypatch, final_count=4 if stage == "metadata_only" else 3
        )
        if stage == "metadata_only":
            failed = report["downloads"][1]["result"]
            normal = failed["normalization"]
            failed["error"] = "Downloaded CSV failed canonical TheDogs normalization gate"
            normal["normalization_failure_reason"] = ""
            normal["normalization_verification"]["runner_set_status"] = "COMPLETE"
    else:
        report = undercovered_report(root, 2 if stage == "raw" else 4)
        if stage == "complete":
            failed = report["downloads"][1]["result"]
            normal = failed["normalization"]
            failed["error"] = "Downloaded CSV failed canonical final runner-set alignment gate"
            normal["normalization_failure_reason"] = (
                "final_runner_set_not_aligned:canonical_participant_missing_from_source_csv"
            )
            normal["normalization_verification"]["runner_set_status"] = "COMPLETE"
            normal["canonical_runner_alignment"].update(
                canonical_runner_count=5,
                missing_canonical_participants=[{"box_number": 5, "dog_name": "Dog 5"}],
            )
    return with_missing_metadata(report, reasons)


@pytest.mark.parametrize("stage", ["raw", "complete", "final", "metadata_only"])
@pytest.mark.parametrize(
    "reasons",
    [
        ("missing_target_grade",),
        ("missing_target_distance",),
        ("missing_target_distance", "missing_target_grade"),
    ],
)
def test_local_field_and_nonidentity_metadata_rejections_compose(
    tmp_path, monkeypatch, stage, reasons
):
    report = composed_report(tmp_path, monkeypatch, stage, reasons)
    before = deepcopy(report)
    assert not has_unisolated_refresh_failure(report)
    result = publish(tmp_path, tmp_path / "runtime/index.json", report, "composed")
    assert result["status"] == "PUBLISHED" and result["race_count"] == 1
    assert report == before and report["downloads"][1]["success"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "unknown_metadata",
        "unsafe_metadata",
        "wrong_race",
        "missing_race",
        "time_mapping",
        "time_source",
        "unknown_component",
        "duplicated_component",
        "transport",
        "cap",
        "discovery",
        "quarantine_missing",
    ],
)
def test_composition_does_not_mask_shared_identity_or_unproved_failure(
    tmp_path, monkeypatch, defect
):
    report = composed_report(tmp_path, monkeypatch, "complete")
    failed = report["downloads"][1]["result"]
    normal = failed["normalization"]
    v = normal["normalization_verification"]
    if defect == "schema":
        v["schema_status"] = "rejected"
    elif defect == "unknown_metadata":
        v["target_metadata_failure_reason"] = "missing_canonical_race_url"
    elif defect == "unsafe_metadata":
        v["target_metadata_status"] = "unsafe"
    elif defect == "wrong_race":
        v["capture_race_number"] = 7
    elif defect == "missing_race":
        v.pop("canonical_url_race_number")
    elif defect == "time_mapping":
        v["race_time_mapping_status"] = "inferred"
    elif defect == "time_source":
        v["race_time_source"] = "fallback"
    elif defect == "unknown_component":
        normal["normalization_failure_reason"] += ";unknown"
    elif defect == "duplicated_component":
        normal[
            "normalization_failure_reason"
        ] += ";target_metadata_not_verified:missing_target_grade"
    elif defect == "transport":
        failed["source_http_status"] = 429
    elif defect == "cap":
        failed["error"] = "request_cap_exhausted"
    elif defect == "discovery":
        report["discovery_failures"] = [{"reason": "denied"}]
    else:
        from pathlib import Path

        Path(failed["quarantine_path"]).unlink()
    assert has_unisolated_refresh_failure(report)
    assert (
        publish(tmp_path, tmp_path / "runtime/index.json", report, "rejected")["status"]
        == "REJECTED"
    )


@pytest.mark.parametrize(
    "field,value",
    [("capture_race_number", 7), ("schema_status", "rejected"), ("race_time_source", "fallback")],
)
def test_verified_metadata_summary_never_hides_independent_integrity_failure(
    tmp_path, monkeypatch, field, value
):
    report = composed_report(tmp_path, monkeypatch, "complete")
    normal = report["downloads"][1]["result"]["normalization"]
    normal["normalization_failure_reason"] = (
        "final_runner_set_not_aligned:canonical_participant_missing_from_source_csv"
    )
    normal["normalization_verification"].update(
        target_metadata_status="verified", target_metadata_failure_reason=None
    )
    normal["normalization_verification"][field] = value
    assert has_unisolated_refresh_failure(report)
    assert (
        publish(tmp_path, tmp_path / "runtime/index.json", report, "rejected")["status"]
        == "REJECTED"
    )
