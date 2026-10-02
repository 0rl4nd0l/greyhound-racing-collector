"""Completed discovery with every selected race excluded must remain truthful."""

from datetime import datetime, timedelta

import pytest

from race_collection import synchronous_manual_capture as capture
from scripts.refresh_prejump_upcoming import (
    current_index_metadata_selection,
    stable_race_id_variants,
)
from src.predictor.on_demand import canonical_bytes
from tests.race_collection.test_synchronous_manual_capture import (
    _runner_coverage,
    _write_publication_evidence,
)


def report_fixture(evidence, *, generated=None, eligible=False):
    generated = generated or datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    race = dict(
        date="2026-07-19",
        jump_datetime="2026-07-19T13:05:00+10:00",
        race_id="Race 5 - GUNN - 2026-07-19",
        race_id_aliases=["Race 5 - GUNN - 2026-07-19"],
        race_number=5,
        race_time="13:05",
        race_url=url,
        venue="GUNN",
    )
    race["race_id_aliases"] = sorted(stable_race_id_variants(race))
    coverage = _runner_coverage(evidence, url, generated)
    coverage["races"][0].update(
        race_id=race["race_id"],
        safe_weather_present=True,
        safe_track_condition_present=eligible,
        safe_expert_form_present=True,
        safe_all_weather_track_expert_form_present=eligible,
        runner_source_observed_at=generated.isoformat(),
    )
    races, selection = current_index_metadata_selection(
        [race], coverage, source_generated_at=generated
    )
    return dict(
        status="SUCCESS" if eligible else "NO_QUALIFIED_RACES",
        generated_at=generated.isoformat(),
        dry_run=False,
        selected_count=1,
        selected_races=[race],
        accepted_csv_count=1,
        sidecar_count=1,
        downloads=[dict(race_url=url, success=True, result={"success": True})],
        current_index_race_count=len(races),
        current_index_races=races,
        current_index_metadata_selection=selection,
        sidecar_metadata_coverage=coverage,
    )


def publish(evidence, state, value, name):
    source = evidence / name / "refresh.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(canonical_bytes(value))
    return capture.publish_current_race_index(
        state_path=state,
        evidence_root=evidence,
        source_refresh_report_path=source,
        run_id=name,
        enforce_monotonic=True,
    )


def test_qualified_then_explicit_empty_then_qualified_index_without_stale_races(tmp_path):
    evidence = tmp_path / "evidence"
    state = evidence / "runtime/odds.json"
    initial = report_fixture(evidence, eligible=True)
    before = publish(evidence, state, initial, "initial")
    assert before["status"] == "PUBLISHED" and before["race_count"] == 1
    _write_publication_evidence(evidence, state, before)
    empty = report_fixture(
        evidence, generated=datetime.fromisoformat(initial["generated_at"]) + timedelta(seconds=1)
    )
    result = publish(evidence, state, empty, "empty")
    assert result["status"] == "PUBLISHED"
    assert result["race_count"] == 0
    _write_publication_evidence(evidence, state, result)
    now = datetime.fromisoformat(empty["generated_at"]) + timedelta(seconds=1)
    view = capture.bounded_current_race_index(
        current_time=now,
        timeout_seconds=1,
        index_path=capture.current_race_index_path(state),
        evidence_root=evidence,
        max_age_seconds=300,
        return_verified_view=True,
    )
    assert not view.races and view.source_generated_at == empty["generated_at"]
    assert empty["current_index_metadata_selection"]["excluded_race_count"] == 1
    qualified = report_fixture(evidence, generated=now, eligible=True)
    after = publish(evidence, state, qualified, "qualified")
    assert after["status"] == "PUBLISHED" and after["race_count"] == 1
    _write_publication_evidence(evidence, state, after)
    view = capture.bounded_current_race_index(
        current_time=now + timedelta(seconds=1),
        timeout_seconds=1,
        index_path=capture.current_race_index_path(state),
        evidence_root=evidence,
        max_age_seconds=300,
        return_verified_view=True,
    )
    assert len(view.races) == 1


@pytest.mark.parametrize(
    "change",
    [
        "cap",
        "discovery",
        "budget",
        "denial",
        "failed_download",
        "missing_sidecar",
        "missing_csv",
        "missing_download",
        "dry_run",
        "unproved_empty",
        "tampered_exclusions",
        "hidden_eligible",
        "duplicate_download",
        "source_metadata_error",
        "missing_expert",
    ],
)
def test_empty_status_cannot_hide_partial_acquisition_or_change_eligibility(tmp_path, change):
    evidence = tmp_path / "evidence"
    state = evidence / "runtime/odds.json"
    value = report_fixture(evidence)
    if change == "cap":
        value["status"] = "DISCOVERY_FAILED"
        value["reason"] = "REQUEST_CAP_EXHAUSTED"
    elif change == "discovery":
        value["discovery_failures"] = [{"error_type": "ReadTimeout"}]
    elif change == "budget":
        value["status"] = "REFRESH_BUDGET_EXCEEDED"
    elif change == "denial":
        value["downloads"][0]["result"]["source_http_status"] = 429
    elif change == "failed_download":
        value["downloads"][0]["result"]["success"] = False
    elif change == "missing_sidecar":
        value["sidecar_count"] = 0
    elif change == "missing_csv":
        value["accepted_csv_count"] = 0
    elif change == "missing_download":
        value["downloads"] = []
    elif change == "dry_run":
        value["dry_run"] = True
    elif change == "unproved_empty":
        value["selected_count"] = 0
        value["selected_races"] = []
    elif change == "tampered_exclusions":
        value["current_index_metadata_selection"]["exclusions"] = []
    elif change == "hidden_eligible":
        value["sidecar_metadata_coverage"]["races"][0].update(
            safe_track_condition_present=True, safe_all_weather_track_expert_form_present=True
        )
    elif change == "duplicate_download":
        value["downloads"] *= 2
    elif change == "source_metadata_error":
        value["sidecar_metadata_coverage"]["races"][0]["weather_track_rejected_reasons"] = [
            "sportsbet_source_request_failed:HTTPError"
        ]
    elif change == "missing_expert":
        value["sidecar_metadata_coverage"]["races"][0]["safe_expert_form_present"] = False
    result = publish(evidence, state, value, "rejected")
    assert result["status"] == "REJECTED"
    assert not capture.current_race_index_path(state).exists()


def test_producer_and_cli_keep_explicit_empty_status_without_claiming_eligible_inputs(
    tmp_path, monkeypatch
):
    from argparse import Namespace
    from types import SimpleNamespace
    import sys
    from scripts import refresh_prejump_upcoming as refresh

    report = report_fixture(tmp_path / "fixture")
    now = datetime.fromisoformat(report["generated_at"])
    candidate = report["selected_races"][0]

    class Browser:
        def get_upcoming_races(self, days_ahead):
            return [{**candidate, "url": candidate["race_url"]}]

        def download_race_csv(self, url, *, race_info_hint=None):
            return {
                "success": True,
                "filepath": report["sidecar_metadata_coverage"]["races"][0]["csv_path"],
            }

    monkeypatch.setitem(
        sys.modules, "upcoming_race_browser", SimpleNamespace(UpcomingRaceBrowser=Browser)
    )
    monkeypatch.setattr(
        refresh,
        "_artifact_counts",
        lambda path: {
            "accepted_csv_count": 1,
            "sidecar_count": 1,
            "raw_export_count": 1,
            "quarantine_count": 0,
        },
    )
    monkeypatch.setattr(
        refresh, "sidecar_metadata_coverage", lambda *args: report["sidecar_metadata_coverage"]
    )
    args = Namespace(
        upcoming_dir=str(tmp_path / "upcoming"),
        days_ahead=0,
        min_minutes=5,
        max_minutes=60,
        limit=4,
        exclude_race_id=[],
        exclude_race_ids_file=None,
        dry_run=False,
        require_safe_metadata=True,
        current_time=now.isoformat(),
    )
    value = refresh.refresh_prejump_upcoming(args)
    assert value["status"] == "NO_QUALIFIED_RACES"
    assert value["current_index_races"] == []
    assert (
        value["current_index_metadata_selection"]["exclusions"]
        == report["current_index_metadata_selection"]["exclusions"]
    )
    monkeypatch.setattr(refresh, "refresh_prejump_upcoming", lambda args: value)
    assert refresh.main(["--upcoming-dir", str(tmp_path / "cli")]) == 0
    value["status"] = "DISCOVERY_FAILED"
    assert refresh.main(["--upcoming-dir", str(tmp_path / "cli")]) == 2


@pytest.mark.parametrize(
    "defect", ["metadata_alignment", "runner_source_timing", "native_identity_conflict"]
)
def test_empty_refresh_cannot_hide_integrity_or_timing_failure(tmp_path, defect):
    value = report_fixture(tmp_path)
    row = value["sidecar_metadata_coverage"]["races"][0]
    if defect == "metadata_alignment":
        row["race_url"] = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/6"
    elif defect == "runner_source_timing":
        row["runner_source_observed_at"] = "2026-07-19T10:00:00+10:00"
    else:
        row["source_native_race_id"] = "invalid"
        row["weather_track_rejected_reasons"] = ["sportsbet_matching_pre_race_event_not_found"]
    _, value["current_index_metadata_selection"] = current_index_metadata_selection(
        value["selected_races"],
        value["sidecar_metadata_coverage"],
        source_generated_at=value["generated_at"],
    )
    assert (
        publish(tmp_path, tmp_path / "runtime/odds.json", value, "rejected")["status"] == "REJECTED"
    )


def test_unmatched_event_with_absent_native_identity_stays_explicitly_excluded(tmp_path):
    value = report_fixture(tmp_path)
    row = value["sidecar_metadata_coverage"]["races"][0]
    row.update(
        source_native_race_id=None,
        source_native_runner_ids=[None, None],
        weather_track_rejected_reasons=["sportsbet_matching_pre_race_event_not_found"],
    )
    _, value["current_index_metadata_selection"] = current_index_metadata_selection(
        value["selected_races"],
        value["sidecar_metadata_coverage"],
        source_generated_at=value["generated_at"],
    )
    assert value["current_index_metadata_selection"]["exclusions"][0]["missing_safe_metadata"] == [
        "track_condition",
        "native_source_identity",
    ]
    result = publish(tmp_path, tmp_path / "runtime/odds.json", value, "empty")
    assert result["status"] == "PUBLISHED" and result["race_count"] == 0
