"""Both native completeness stages stay strict while isolated short fields stay excluded."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import capture_thedogs_market_history as identity_source
from scripts.refresh_prejump_upcoming import has_unisolated_refresh_failure
from tests.test_semantic_runner_quarantine import undercovered_report
from tests.test_empty_eligible_refresh import publish
from utils import runner_completeness as runners


def aligned_shortfall_report(root, monkeypatch, *, raw_count=4, final_count=3):
    report = undercovered_report(root, raw_count)
    failed = report["downloads"][1]["result"]
    normal = failed["normalization"]
    candidate = report["selected_races"][1]
    candidate["race_number"] = str(candidate["race_number"])
    canonical = {
        "canonical_runner_set_status": "available",
        "final_runner_source_url": candidate["race_url"],
        "source_native_race_id": "16000",
        "native_identity_status": "available",
        "native_identity_reasons": [],
        "final_runner_participants": [
            {"box_number": i, "dog_name": f"Dog {i}", "source_native_runner_id": str(100 + i)}
            for i in range(1, final_count + 1)
        ],
    }
    raw = Path(failed["raw_export_path"]).read_text()
    aligned, alignment = runners.align_csv_text_to_canonical_final_runner_set(
        raw, canonical, source=normal["accepted_csv_path"]
    )
    after = runners.analyze_csv_text_runner_completeness(
        aligned, source=normal["accepted_csv_path"]
    ).as_dict()
    for row in after["participants"]:
        row.update(scratch_state="ACTIVE", source_native_runner_id=str(100 + row["box_number"]))
    normal.update(
        canonical_runner_alignment=alignment,
        runner_completeness_after_canonical_alignment=after,
        normalization_failure_reason="runner_set_not_complete:INCOMPLETE",
        native_identity_evidence={"fixture": True, "race_page_http": "fixture"},
        source_native_race_id="16000",
        normalization_timestamp="2026-07-19T12:55:00Z",
    )
    normal["normalization_verification"].update(
        canonical_runner_set_status="available",
        canonical_runner_alignment_status="aligned",
        canonical_runner_alignment_reason=None,
        canonical_runner_count=final_count,
        canonical_prediction_runner_count=final_count,
    )

    def verify(evidence, **kwargs):
        assert kwargs["expected_race_url"] == candidate["race_url"]
        assert kwargs["expected_native_race_id"] == "16000"
        return evidence == {"fixture": True, "race_page_http": "fixture"}, None

    monkeypatch.setattr(identity_source, "validate_primary_native_identity_evidence", verify)
    monkeypatch.setattr(
        identity_source,
        "_stored_response",
        lambda *a, **k: SimpleNamespace(
            body=b"fixture", request_end_utc=datetime(2026, 7, 19, 12, 54, tzinfo=timezone.utc)
        ),
    )

    def extract(*args, **kwargs):
        assert type(kwargs["expected_race_number"]) is int
        return deepcopy(canonical)

    monkeypatch.setattr(runners, "extract_canonical_runner_set_from_html", extract)
    return report


@pytest.mark.parametrize("raw_count,final_count", [(4, 3), (4, 2), (3, 2)])
def test_raw_and_post_alignment_shortfalls_remain_excluded_without_aborting_other_races(
    tmp_path, monkeypatch, raw_count, final_count
):
    report = aligned_shortfall_report(
        tmp_path, monkeypatch, raw_count=raw_count, final_count=final_count
    )
    assert not has_unisolated_refresh_failure(report)
    result = publish(tmp_path, tmp_path / "runtime/odds.json", report, "local")
    assert result["status"] == "PUBLISHED" and result["race_count"] == 1
    assert report["downloads"][1]["success"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "native_identity",
        "after_count",
        "after_name",
        "after_id",
        "drop_witness",
        "alignment_count",
        "raw_count",
        "transport",
        "minimum",
    ],
)
def test_post_alignment_shortfall_never_weakens_identity_or_completeness(
    tmp_path, monkeypatch, defect
):
    report = aligned_shortfall_report(tmp_path, monkeypatch)
    failed = report["downloads"][1]["result"]
    normal = failed["normalization"]
    after = normal["runner_completeness_after_canonical_alignment"]
    if defect == "native_identity":
        normal["native_identity_evidence"] = {"fixture": False}
    elif defect == "after_count":
        after["runner_count"] = 2
    elif defect == "after_name":
        after["participants"][0]["dog_name"] = "Wrong Runner"
    elif defect == "after_id":
        after["participants"][0]["source_native_runner_id"] = "bad"
    elif defect == "drop_witness":
        normal["canonical_runner_alignment"]["dropped_participants"] = []
    elif defect == "alignment_count":
        normal["canonical_runner_alignment"]["prediction_runner_count"] = 2
    elif defect == "raw_count":
        failed["runner_completeness"]["runner_count"] = 3
    elif defect == "minimum":
        after["min_complete_runners"] = 3
    else:
        failed["source_http_status"] = 429
    assert has_unisolated_refresh_failure(report)
    assert (
        publish(tmp_path, tmp_path / "runtime/odds.json", report, "rejected")["status"]
        == "REJECTED"
    )
