from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import race_collection.synchronous_manual_capture as capture
from scripts import shadow_autopilot_daemon as daemon
from scripts.refresh_prejump_upcoming import race_window_record
from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
)
from race_collection.synchronous_manual_capture import (
    bounded_current_race_index,
    CaptureCancelled,
    CaptureOneDependencies,
    CaptureOneRejected,
    CollectorBusy,
    current_race_index_path,
    LatencyBudget,
    invoke_capture_one,
    publish_current_race_index,
    publish_scheduled_capture_receipts,
    run_capture_one,
)
from src.predictor.on_demand import canonical_bytes, runner_set_sha256, sha256_bytes


def _runner_coverage(
    evidence_root: Path,
    race_url: str,
    observed_at: datetime | None = None,
    *,
    source_race_url: str | None = None,
) -> dict:
    observed_at = observed_at or NOW
    source_race_url = source_race_url or race_url
    csv_path = evidence_root / "upcoming/Race 5 - GUNN - 2026-07-19.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(b"box|dog_name\n1|Alpha\n2|Beta\n")
    sidecar = csv_path.with_name(csv_path.name + ".metadata.json")
    sidecar.write_bytes(canonical_bytes({
        "runner_completeness_after_canonical_alignment": {
            "status": "COMPLETE", "runner_count": 2,
            "participants": [
                {"box_number": 1, "dog_name": "Alpha", "scratch_state": "ACTIVE"},
                {"box_number": 2, "dog_name": "Beta", "scratch_state": "ACTIVE"},
            ],
        },
        "prejump_shadow_metadata": {
            "status": "PASS", "metadata_is_leakage_safe": True,
            "race_date": "2026-07-19", "venue": "GUNN", "race_number": 5,
            "source_url": source_race_url, "metadata_captured_at": observed_at.isoformat(),
            "runner_box_name_list": [
                {"box_number": 1, "dog_name": "Alpha"},
                {"box_number": 2, "dog_name": "Beta"},
            ],
            "canonical_final_runner_alignment": {
                "status": "aligned", "canonical_runner_set_status": "available"
            },
        }
    }))
    return {"schema_version": "prejump_sidecar_metadata_coverage_v1", "races": [{
        "race_url": source_race_url, "csv_path": str(csv_path), "sidecar_path": str(sidecar)
    }]}


def _write_publication_evidence(
    evidence_root: Path, state: Path, published: Mapping[str, Any]
) -> None:
    output_locator = "daemon_publication"
    output_dir = evidence_root / output_locator
    output_dir.mkdir(parents=True, exist_ok=True)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(canonical_bytes({
        "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
        "updated_at": published["source_generated_at"],
        "run_id": published["run_id"], "output_dir": output_locator,
        "autopilot_output_dir": output_locator,
        "final_status": "ODDS_CAPTURE_ONLY_READY", "status": "READY",
    }))
    (output_dir / capture.ODDS_CAPTURE_ONLY_REPORT_FILENAME).write_bytes(
        canonical_bytes({
            "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
            "generated_at": published["source_generated_at"],
            "run_id": published["run_id"],
            "output_dir": output_locator,
            "autopilot_output_dir": output_locator,
            "final_status": "ODDS_CAPTURE_ONLY_READY", "status": "READY",
            "current_race_index_publish": dict(published),
        })
    )

NOW = datetime.fromisoformat("2026-07-30T16:55:00+10:00")
JUMP = NOW + timedelta(minutes=20)
RACE_ID = "Race 1 - WARRNAMBOOL - 2026-07-30"
RACE_URL = "https://www.thedogs.com.au/racing/warrnambool/2026-07-30/1"
RUNNERS = [
    {"box_number": 1, "dog_name": "Alpha", "identity": "ALPHA"},
    {"box_number": 2, "dog_name": "Beta", "identity": "BETA"},
]


def publish_request(protocol: ManualPredictionCollectorProtocol) -> Mapping[str, Any]:
    return protocol.publish_request(
        race={
            "race_id": RACE_ID,
            "url": RACE_URL,
            "venue": "WARRNAMBOOL",
            "race_number": 1,
            "race_date": "2026-07-30",
            "jump_timestamp": JUMP.isoformat(),
        },
        expected_runners=RUNNERS,
        created_at=NOW,
        expires_at=JUMP,
    )


def validation() -> dict[str, Any]:
    rows = [
        {
            "dog_name": row["dog_name"],
            "dog_clean_name": row["dog_name"],
            "box_number": row["box_number"],
            "identity": row["identity"],
            "odds_decimal": 2.5 + row["box_number"],
            "sportsbet_box_source": "explicit_dom",
        }
        for row in RUNNERS
    ]
    return {
        "schema_version": "autonomous_live_odds_capture_validation_v1",
        "status": "PASS",
        "source_url": "https://www.sportsbet.com.au/greyhounds/warrnambool/race-1",
        "accepted_rows": rows,
        "accepted_place_rows": rows,
        "reasons": [],
    }


def plan_item(form: Path) -> dict[str, Any]:
    return {
        "schema_version": "autonomous_live_odds_capture_plan_item_v1",
        "status": "READY_TO_CAPTURE",
        "csv_path": str(form),
        "sidecar_path": str(form) + ".metadata.json",
        "race_id": RACE_ID,
        "venue": "WARRNAMBOOL",
        "race_number": 1,
        "race_date": "2026-07-30",
        "race_time": "17:15",
        "jump_datetime": JUMP.isoformat(),
        "minutes_to_jump": 20.0,
        "capture_window_minutes": 30,
        "window_status": "due_now_or_passed_pre_jump",
        "thedogs_source_url": RACE_URL,
        "runner_set_validation": {
            "status": "PASS",
            "expected_runners": RUNNERS,
        },
        "expected_runners": RUNNERS,
        "blockers": [],
    }


def successful_report() -> dict[str, Any]:
    return {
        "schema_version": "autonomous_live_odds_capture_report_v1",
        "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
        "appended_attempt_count": 1,
        "attempts": [
            {
                "schema_version": "autonomous_live_odds_capture_attempt_v1",
                "race_id": RACE_ID,
                "status": "APPENDED",
                "capture_window_minutes": 30,
                "fetch_time": NOW.isoformat(),
                "append_time": (NOW + timedelta(seconds=2)).isoformat(),
                "reasons": [],
                "validation": validation(),
                "append_report": {
                    "status": "SUCCESS",
                    "race_id": RACE_ID,
                    "inserted_rows": 4,
                    "append_only": True,
                    "capture_timestamp": (NOW + timedelta(seconds=2)).isoformat(),
                },
            }
        ],
        "inserted_live_odds_rows": 4,
    }


def dependencies(
    tmp_path: Path,
    *,
    report: Mapping[str, Any] | None = None,
    acquire: Any = None,
    phase_hook: Any = None,
) -> tuple[CaptureOneDependencies, dict[str, int]]:
    calls = {"refresh": 0, "execute": 0, "acquire": 0, "release": 0}

    def refresh(
        race: Mapping[str, Any], output_dir: Path, current_time: datetime
    ) -> tuple[Path, Path]:
        assert race["race_id"] == RACE_ID
        assert current_time == NOW
        calls["refresh"] += 1
        form = output_dir / "Race 1 - WARRNAMBOOL - 2026-07-30.csv"
        form.parent.mkdir(parents=True, exist_ok=True)
        form.write_text("dog_name,box_number\nAlpha,1\nBeta,2\n", encoding="utf-8")
        sidecar = form.with_name(form.name + ".metadata.json")
        sidecar.write_bytes(canonical_bytes({"race_id": RACE_ID}))
        return form, sidecar

    def build(form: Path, current_time: datetime) -> Mapping[str, Any]:
        assert current_time == NOW
        return plan_item(form)

    def execute(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        del args
        assert kwargs["execute"] is True
        assert kwargs["allow_auto_scrape_odds"] is True
        calls["execute"] += 1
        return dict(report or successful_report())

    def default_acquire(**kwargs: Any) -> object:
        del kwargs
        return object()

    def release(handle: object) -> None:
        assert handle is not None
        calls["release"] += 1

    def counted_acquire(**kwargs: Any) -> object:
        calls["acquire"] += 1
        if acquire is not None:
            return acquire(**kwargs)
        return default_acquire(**kwargs)

    return (
        CaptureOneDependencies(
            now=lambda: NOW,
            refresh_exact=refresh,
            build_plan_item=build,
            execute_capture_plan=execute,
            acquire_lock=counted_acquire,
            release_lock=release,
            phase_hook=phase_hook or (lambda phase: None),
        ),
        calls,
    )


def run(
    tmp_path: Path,
    deps: CaptureOneDependencies,
    *,
    minimum_margin_seconds: float = 120,
) -> tuple[ManualPredictionCollectorProtocol, Mapping[str, Any]]:
    protocol = ManualPredictionCollectorProtocol(tmp_path / "protocol")
    request = publish_request(protocol)
    result = run_capture_one(
        protocol_root=protocol.root,
        evidence_root=tmp_path / "evidence",
        request_id=str(request["request_id"]),
        db_path=tmp_path / "canonical.db",
        lock_path=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/"
            "shadow_autopilot_daemon_runtime/shadow_autopilot.lock"
        ),
        output_dir=tmp_path / "evidence" / "capture-one",
        minimum_margin_seconds=minimum_margin_seconds,
        minimum_post_lock_margin_seconds=max(
            1, minimum_margin_seconds - 1
        ),
        minimum_fetch_margin_seconds=max(
            1, minimum_margin_seconds - 1
        ),
        fetch_timeout_seconds=45,
        dependencies=deps,
    )
    return protocol, result


def test_latency_budget_declares_and_computes_enforced_margin():
    budget = LatencyBudget.from_config(
        {
            "discovery_seconds": 12,
            "lock_seconds": 1,
            "capture_seconds": 45,
            "validation_seconds": 8,
            "scoring_seconds": 30,
            "safety_seconds": 15,
        }
    )

    assert budget.capture_margin_seconds == 99
    assert budget.total_margin_seconds == 111
    assert budget.reuse_margin_seconds == 53
    assert budget.post_lock_margin_seconds == 98
    assert budget.pre_fetch_margin_seconds(45) == 98


def test_current_race_index_publication_is_atomic_bounded_and_source_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    evidence_root = tmp_path / "evidence"
    state = evidence_root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    source = evidence_root / "shadow_autopilot_v1_fixture/odds_capture_refresh_report.json"
    source.parent.mkdir(parents=True)
    state.parent.mkdir(parents=True)
    state.write_bytes(
        canonical_bytes(
            {
                "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
                "run_id": "prior-run",
                "status": "READY",
            }
        )
    )
    index_now = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    canonical_race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    race_url = f"{canonical_race_url}?trial=false"
    source.write_bytes(
        canonical_bytes(
            {
                "status": "SUCCESS",
                "generated_at": index_now.isoformat(),
                "sidecar_metadata_coverage": _runner_coverage(
                    evidence_root,
                    race_url,
                    index_now,
                    source_race_url=canonical_race_url,
                ),
                "selected_count": 1,
                "selected_races": [
                    {
                        "date": "2026-07-19",
                        "jump_datetime": "2026-07-19T13:00:00+10:00",
                        "race_id": "Race 5 - GUNN - 2026-07-19",
                        "race_id_aliases": [
                            "Race 5 - GUNN - 2026-07-19",
                            "Race 5 - GUNNEDAH - 2026-07-19",
                        ],
                        "race_number": 5,
                        "race_time": "13:00",
                        "race_url": (
                            race_url
                        ),
                        "venue": "GUNN",
                    }
                ],
            }
        )
    )

    published = publish_current_race_index(
        state_path=state,
        evidence_root=evidence_root,
        source_refresh_report_path=source,
        run_id="fixture",
    )
    index_path = current_race_index_path(state)
    original = index_path.read_bytes()
    lifecycle = capture.bind_current_race_index_publication_lifecycle(
        state_path=state,
        evidence_root=evidence_root,
        output_dir=source.parent,
        publication=published,
    )
    assert lifecycle["status"] == "BOUND"
    early_state = json.loads(state.read_bytes())
    assert early_state["run_id"] == "prior-run"
    assert early_state["current_race_index_state"]["run_id"] == "fixture"
    assert bounded_current_race_index(
        current_time=index_now + timedelta(seconds=1),
        timeout_seconds=1,
        index_path=index_path,
        evidence_root=evidence_root,
        max_age_seconds=300,
    )[0]["race_id"] == "Race 5 - GUNN - 2026-07-19"
    _write_publication_evidence(evidence_root, state, published)
    producer_root = evidence_root.parent
    monkeypatch.setattr(capture, "ROOT", producer_root)
    monkeypatch.setattr(daemon, "ROOT", producer_root)
    producer_output_dir = evidence_root / "daemon_publication"
    producer_output_locator = daemon.relpath(producer_output_dir)
    daemon.write_json(state, {
        "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
        "updated_at": published["source_generated_at"],
        "run_id": published["run_id"], "output_dir": producer_output_locator,
        "autopilot_output_dir": producer_output_locator,
        "final_status": "ODDS_CAPTURE_ONLY_READY", "status": "READY",
    })
    daemon.write_json(
        producer_output_dir / capture.ODDS_CAPTURE_ONLY_REPORT_FILENAME,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
            "generated_at": published["source_generated_at"],
            "run_id": published["run_id"],
            "output_dir": producer_output_locator,
            "autopilot_output_dir": producer_output_locator,
            "final_status": "ODDS_CAPTURE_ONLY_READY", "status": "READY",
            "current_race_index_publish": dict(published),
        },
    )

    assert published["status"] == "PUBLISHED"
    assert json.loads(original)["races"][0]["runner_set_sha256"] == (
        runner_set_sha256(RUNNERS)
    )
    assert json.loads(original)["source_refresh_report_sha256"] == sha256_bytes(
        source.read_bytes()
    )
    assert bounded_current_race_index(
        current_time=index_now,
        timeout_seconds=1,
        index_path=index_path,
        evidence_root=evidence_root,
        max_age_seconds=900,
    )[0]["race_id"] == "Race 5 - GUNN - 2026-07-19"
    verified_view = bounded_current_race_index(
        current_time=index_now, timeout_seconds=1, index_path=index_path,
        evidence_root=evidence_root, max_age_seconds=900,
        return_verified_view=True,
    )
    assert isinstance(verified_view, capture.VerifiedCurrentRaceIndex)
    assert verified_view.packet_bytes == original
    assert verified_view.packet_sha256 == sha256_bytes(original)
    assert verified_view.races[0]["race_id"] == "Race 5 - GUNN - 2026-07-19"
    assert verified_view.races[0]["runners"][0]["source_native_runner_id"] is None

    boundary = index_now + timedelta(seconds=1200)
    assert bounded_current_race_index(
        current_time=boundary, timeout_seconds=1, index_path=index_path,
        evidence_root=evidence_root, max_age_seconds=1200,
    )[0]["race_id"] == "Race 5 - GUNN - 2026-07-19"
    over_boundary = boundary + timedelta(microseconds=1)
    with pytest.raises(CaptureOneRejected) as stale:
        bounded_current_race_index(
            current_time=over_boundary, timeout_seconds=1,
            index_path=index_path, evidence_root=evidence_root,
            max_age_seconds=1200,
        )
    assert stale.value.code == "CURRENT_INDEX_STALE"
    retained_stale = bounded_current_race_index(
        current_time=over_boundary, timeout_seconds=1,
        index_path=index_path, evidence_root=evidence_root,
        max_age_seconds=1200, return_verified_view=True,
    )
    assert isinstance(retained_stale, capture.VerifiedCurrentRaceIndex)
    assert retained_stale.packet_bytes == original
    assert retained_stale.packet_sha256 == sha256_bytes(original)

    real_validate = capture._RetainedSafeFiles.validate

    def replace_early_packet(snapshot: Any) -> None:
        replacement = index_path.with_name("replacement.json")
        replacement.write_bytes(index_path.read_bytes())
        os.replace(replacement, index_path)
        real_validate(snapshot)

    with monkeypatch.context() as attack:
        attack.setattr(capture._RetainedSafeFiles, "validate", replace_early_packet)
        with pytest.raises(CaptureOneRejected) as replaced:
            bounded_current_race_index(
                current_time=index_now, timeout_seconds=1,
                index_path=index_path, evidence_root=evidence_root,
                max_age_seconds=900,
            )
    assert replaced.value.code == "CURRENT_INDEX_PATH_UNSAFE"
    assert replaced.value.details["reason"] == "path_replaced"

    state_bytes = state.read_bytes()
    state_payload = json.loads(state_bytes)
    report_path = (
        producer_root / state_payload["output_dir"]
        / capture.ODDS_CAPTURE_ONLY_REPORT_FILENAME
    )
    report_bytes = report_path.read_bytes()
    report_payload = json.loads(report_bytes)

    report_path.unlink()
    with pytest.raises(CaptureOneRejected) as missing_report:
        bounded_current_race_index(
            current_time=index_now, timeout_seconds=1, index_path=index_path,
            evidence_root=evidence_root, max_age_seconds=900,
        )
    assert missing_report.value.code == "CURRENT_INDEX_REPORT_MISSING"
    report_path.write_bytes(report_bytes)

    for target, field, value in (
        (state, "updated_at", (index_now + timedelta(seconds=1)).isoformat()),
        (state, "final_status", "SKIPPED_LOCK_HELD"),
        (report_path, "generated_at", (index_now - timedelta(seconds=901)).isoformat()),
        (report_path, "generated_at", (index_now + timedelta(seconds=1)).isoformat()),
        (report_path, "final_status", "ODDS_CAPTURE_ONLY_FAILED"),
        (report_path, "status", "SKIPPED"),
    ):
        original_target = target.read_bytes()
        changed = json.loads(original_target)
        changed[field] = value
        target.write_bytes(canonical_bytes(changed))
        with pytest.raises(CaptureOneRejected) as invalid_lifecycle:
            bounded_current_race_index(
                current_time=index_now, timeout_seconds=1,
                index_path=index_path, evidence_root=evidence_root,
                max_age_seconds=900,
            )
        assert invalid_lifecycle.value.code == "CURRENT_INDEX_REPORT_INVALID"
        target.write_bytes(original_target)

    malformed_report = dict(report_payload, schema_version="wrong_schema")
    report_path.write_bytes(canonical_bytes(malformed_report))
    with pytest.raises(CaptureOneRejected) as malformed:
        bounded_current_race_index(
            current_time=index_now, timeout_seconds=1, index_path=index_path,
            evidence_root=evidence_root, max_age_seconds=900,
        )
    assert malformed.value.code == "CURRENT_INDEX_REPORT_INVALID"
    report_path.write_bytes(report_bytes)

    for unsafe_output in ("../outside", str(evidence_root / "outside")):
        unsafe_state = dict(state_payload, output_dir=unsafe_output)
        state.write_bytes(canonical_bytes(unsafe_state))
        with pytest.raises(CaptureOneRejected) as unsafe:
            bounded_current_race_index(
                current_time=index_now, timeout_seconds=1, index_path=index_path,
                evidence_root=evidence_root, max_age_seconds=900,
            )
        assert unsafe.value.code == "CURRENT_INDEX_REPORT_INVALID"
    state.write_bytes(state_bytes)

    stale_state = dict(state_payload, run_id="stale-run")
    state.write_bytes(canonical_bytes(stale_state))
    with pytest.raises(CaptureOneRejected) as stale_run:
        bounded_current_race_index(
            current_time=index_now, timeout_seconds=1, index_path=index_path,
            evidence_root=evidence_root, max_age_seconds=900,
        )
    assert stale_run.value.code == "CURRENT_INDEX_REPORT_INVALID"
    state.write_bytes(state_bytes)

    for field, value in (
        ("run_id", "wrong-run"),
        ("publish_status", "SKIPPED"),
        ("publish_status", "REJECTED"),
        ("packet_sha256", "0" * 64),
    ):
        changed_report = json.loads(report_bytes)
        if field == "publish_status":
            changed_report["current_race_index_publish"]["status"] = value
        elif field == "packet_sha256":
            changed_report["current_race_index_publish"][field] = value
        else:
            changed_report[field] = value
        report_path.write_bytes(canonical_bytes(changed_report))
        with pytest.raises(CaptureOneRejected) as divergent:
            bounded_current_race_index(
                current_time=index_now, timeout_seconds=1, index_path=index_path,
                evidence_root=evidence_root, max_age_seconds=900,
            )
        assert divergent.value.code == "CURRENT_INDEX_REPORT_INVALID"
    report_path.write_bytes(report_bytes)

    source.write_bytes(
        canonical_bytes(
            {
                "generated_at": NOW.isoformat(),
                "selected_count": 33,
                "selected_races": [{}] * 33,
            }
        )
    )
    rejected = publish_current_race_index(
        state_path=state,
        evidence_root=evidence_root,
        source_refresh_report_path=source,
        run_id="invalid",
    )

    assert rejected["status"] == "REJECTED"
    assert index_path.read_bytes() == original


def test_current_race_index_survives_a_later_rejected_publication(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    state = evidence_root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    source = evidence_root / "published/odds_capture_refresh_report.json"
    source.parent.mkdir(parents=True)
    index_now = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    source.write_bytes(canonical_bytes({
        "status": "SUCCESS",
        "generated_at": index_now.isoformat(),
        "sidecar_metadata_coverage": _runner_coverage(
            evidence_root, race_url, index_now
        ),
        "selected_count": 1,
        "selected_races": [{
            "date": "2026-07-19",
            "jump_datetime": "2026-07-19T13:00:00+10:00",
            "race_id": "Race 5 - GUNN - 2026-07-19",
            "race_id_aliases": [
                "Race 5 - GUNN - 2026-07-19",
                "Race 5 - GUNNEDAH - 2026-07-19",
            ],
            "race_number": 5,
            "race_time": "13:00",
            "race_url": race_url,
            "venue": "GUNN",
        }],
    }))
    published = publish_current_race_index(
        state_path=state,
        evidence_root=evidence_root,
        source_refresh_report_path=source,
        run_id="published-run",
    )
    _write_publication_evidence(evidence_root, state, published)
    published_state = json.loads(state.read_bytes())
    published_pointer = {
        "schema_version": "collector_current_race_index_state_v1",
        "updated_at": published_state["updated_at"],
        "run_id": published_state["run_id"],
        "output_dir": published_state["output_dir"],
        "autopilot_output_dir": published_state["autopilot_output_dir"],
        "final_status": published_state["final_status"],
        "status": published_state["status"],
    }

    rejected_output = evidence_root / "rejected-daemon-run"
    rejected_output.mkdir()
    rejected_time = index_now + timedelta(minutes=1)
    rejected_publish = {
        "schema_version": "collector_current_race_index_publish_v2",
        "status": "REJECTED",
        "reason": "CURRENT_INDEX_SOURCE_INVALID",
        "run_id": "rejected-run",
    }
    state.write_bytes(canonical_bytes({
        "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
        "updated_at": rejected_time.isoformat(),
        "run_id": "rejected-run",
        "output_dir": "rejected-daemon-run",
        "autopilot_output_dir": "rejected-daemon-run",
        "final_status": "ODDS_CAPTURE_ONLY_READY",
        "status": "READY_WITH_BLOCKED_ATTEMPTS",
        "current_race_index_state": published_pointer,
    }))
    (rejected_output / capture.ODDS_CAPTURE_ONLY_REPORT_FILENAME).write_bytes(
        canonical_bytes({
            "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
            "generated_at": rejected_time.isoformat(),
            "run_id": "rejected-run",
            "output_dir": "rejected-daemon-run",
            "autopilot_output_dir": "rejected-daemon-run",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "status": "READY_WITH_BLOCKED_ATTEMPTS",
            "current_race_index_publish": rejected_publish,
        })
    )

    verified = bounded_current_race_index(
        current_time=rejected_time,
        timeout_seconds=1,
        index_path=current_race_index_path(state),
        evidence_root=evidence_root,
        max_age_seconds=900,
        return_verified_view=True,
    )

    assert verified.run_id == "published-run"
    assert verified.races[0]["race_id"] == "Race 5 - GUNN - 2026-07-19"


def test_current_race_index_rejects_stale_or_changed_source(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    state = evidence_root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    source = evidence_root / "shadow_autopilot_v1_fixture/odds_capture_refresh_report.json"
    source.parent.mkdir(parents=True)
    index_now = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    source.write_bytes(
        canonical_bytes(
            {
                "status": "SUCCESS",
                "generated_at": (index_now - timedelta(minutes=20)).isoformat(),
                "sidecar_metadata_coverage": _runner_coverage(
                    evidence_root, race_url, index_now - timedelta(minutes=20)
                ),
                "selected_count": 1,
                "selected_races": [
                    {
                        "date": "2026-07-19",
                        "jump_datetime": "2026-07-19T13:00:00+10:00",
                        "race_id": "Race 5 - GUNN - 2026-07-19",
                        "race_id_aliases": [
                            "Race 5 - GUNN - 2026-07-19",
                            "Race 5 - GUNNEDAH - 2026-07-19",
                        ],
                        "race_number": 5,
                        "race_time": "13:00",
                        "race_url": (
                            race_url
                        ),
                        "venue": "GUNN",
                    }
                ],
            }
        )
    )
    published = publish_current_race_index(
        state_path=state,
        evidence_root=evidence_root,
        source_refresh_report_path=source,
        run_id="fixture",
    )
    _write_publication_evidence(evidence_root, state, published)
    index_path = current_race_index_path(state)

    with pytest.raises(CaptureOneRejected) as stale:
        bounded_current_race_index(
            current_time=index_now,
            timeout_seconds=1,
            index_path=index_path,
            evidence_root=evidence_root,
            max_age_seconds=900,
        )
    assert stale.value.code == "CURRENT_INDEX_STALE"

    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(CaptureOneRejected) as changed:
        bounded_current_race_index(
            current_time=index_now - timedelta(minutes=20),
            timeout_seconds=1,
            index_path=index_path,
            evidence_root=evidence_root,
            max_age_seconds=900,
        )
    assert changed.value.code == "CURRENT_INDEX_SOURCE_CHANGED"


@pytest.mark.parametrize("existing_pair", [False, True])
def test_final_validation_rejection_never_publishes_new_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_pair: bool,
):
    evidence_root = tmp_path / "evidence"
    state = evidence_root / "runtime/odds_capture_state.json"
    source = evidence_root / "refresh/report.json"
    source.parent.mkdir(parents=True)
    generated = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    source.write_bytes(canonical_bytes({
        "status": "SUCCESS", "generated_at": generated.isoformat(),
        "sidecar_metadata_coverage": _runner_coverage(evidence_root, race_url, generated),
        "selected_count": 1, "selected_races": [{
            "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
            "race_id": "Race 5 - GUNN - 2026-07-19",
            "race_id_aliases": [
                "Race 5 - GUNN - 2026-07-19",
                "Race 5 - GUNNEDAH - 2026-07-19",
            ],
            "race_number": 5, "race_time": "13:00", "race_url": race_url,
            "venue": "GUNN",
        }],
    }))
    index_path = current_race_index_path(state)
    publication_path = index_path.parent / capture.CURRENT_RACE_INDEX_PUBLICATION_FILENAME
    prior_index = prior_publication = None
    if existing_pair:
        prior = publish_current_race_index(
            state_path=state, evidence_root=evidence_root,
            source_refresh_report_path=source, run_id="prior",
        )
        assert prior["status"] == "PUBLISHED"
        prior_index = index_path.read_bytes()
        prior_publication = publication_path.read_bytes()
    real_validate = capture._RetainedSafeFiles.validate

    def replace_refresh_at_final_boundary(retained: Any) -> None:
        assert index_path.exists()
        assert publication_path.exists() is existing_pair
        if prior_publication is not None:
            assert publication_path.read_bytes() == prior_publication
        replacement = source.with_name("replacement.json")
        replacement.write_bytes(source.read_bytes())
        os.replace(replacement, source)
        real_validate(retained)

    monkeypatch.setattr(
        capture._RetainedSafeFiles, "validate", replace_refresh_at_final_boundary
    )
    monkeypatch.setattr(
        Path, "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup denied")),
    )
    rejected = publish_current_race_index(
        state_path=state, evidence_root=evidence_root,
        source_refresh_report_path=source, run_id="adversarial",
    )

    assert rejected["status"] == "REJECTED"
    assert rejected["reason"] == "CURRENT_INDEX_PATH_UNSAFE"
    assert index_path.exists()
    if existing_pair:
        assert prior_index is not None and index_path.read_bytes() != prior_index
        assert prior_publication is not None
        assert publication_path.read_bytes() == prior_publication
        assert json.loads(prior_publication)["packet_sha256"] != sha256_bytes(
            index_path.read_bytes()
        )
    else:
        assert not publication_path.exists()


@pytest.mark.parametrize("case", ["duplicate", "conflict"])
def test_v2_runner_seal_rejects_native_id_collisions(
    tmp_path: Path, case: str,
):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)
    sidecar_path = Path(coverage["races"][0]["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_bytes())
    detailed = sidecar["runner_completeness_after_canonical_alignment"]["participants"]
    shadow = sidecar["prejump_shadow_metadata"]["runner_box_name_list"]
    if case == "duplicate":
        detailed[0]["source_native_runner_id"] = "native-1"
        detailed[1]["source_native_runner_id"] = "native-1"
    else:
        detailed[0]["source_native_runner_id"] = "native-detailed"
        shadow[0]["source_native_runner_id"] = "native-shadow"
    sidecar_path.write_bytes(canonical_bytes(sidecar))

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._v2_runner_rows(
            {
                "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
                "race_number": 5, "race_url": race_url, "venue": "GUNN",
            },
            {"generated_at": observed.isoformat(), "sidecar_metadata_coverage": coverage},
            evidence_root=evidence_root,
        )

    expected = "runner_id_duplicate" if case == "duplicate" else "runner_id_conflict"
    assert rejected.value.details["reason"] == expected


def test_retained_inputs_survive_expected_atomic_publication(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    source = evidence_root / "refresh.json"
    source.write_bytes(b"retained input")

    with capture._RetainedSafeFiles(evidence_root) as retained:
        assert retained.read(source, missing_code="CURRENT_INDEX_SOURCE_MISSING")
        capture._atomic_replace_canonical(
            evidence_root / "current.json", {"kind": "index"},
            evidence_root=evidence_root,
        )
        capture._atomic_replace_canonical(
            evidence_root / "publication.json", {"kind": "publication"},
            evidence_root=evidence_root,
        )
        retained.validate()


def test_v2_runner_identity_uses_canonical_punctuation_protocol(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    generated = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    coverage = _runner_coverage(evidence_root, race_url, generated)
    record = coverage["races"][0]
    csv_path = Path(record["csv_path"])
    sidecar_path = Path(record["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_bytes())
    for collection in (
        sidecar["runner_completeness_after_canonical_alignment"]["participants"],
        sidecar["prejump_shadow_metadata"]["runner_box_name_list"],
    ):
        collection[0]["dog_name"] = "O'MALLEY"
    sidecar_path.write_bytes(canonical_bytes(sidecar))
    csv_path.write_bytes(b"box|dog_name\n1|OMALLEY\n2|Beta\n")
    race = {
        "race_url": race_url, "date": "2026-07-19", "venue": "GUNN",
        "race_number": 5, "jump_datetime": "2026-07-19T13:00:00+10:00",
    }
    source = {"generated_at": generated.isoformat(), "sidecar_metadata_coverage": coverage}

    rows, _, _ = capture._v2_runner_rows(race, source, evidence_root=evidence_root)

    assert rows[0]["display_name"] == "O'MALLEY"
    assert rows[0]["identity"] == "OMALLEY"


def test_v2_runner_punctuation_variants_cannot_evade_duplicate_validation(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    generated = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    coverage = _runner_coverage(evidence_root, race_url, generated)
    record = coverage["races"][0]
    sidecar_path = Path(record["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_bytes())
    names = ("O'MALLEY", "OMALLEY")
    for key in (
        ("runner_completeness_after_canonical_alignment", "participants"),
        ("prejump_shadow_metadata", "runner_box_name_list"),
    ):
        for item, name in zip(sidecar[key[0]][key[1]], names):
            item["dog_name"] = name
    sidecar_path.write_bytes(canonical_bytes(sidecar))
    race = {
        "race_url": race_url, "date": "2026-07-19", "venue": "GUNN",
        "race_number": 5, "jump_datetime": "2026-07-19T13:00:00+10:00",
    }
    source = {"generated_at": generated.isoformat(), "sidecar_metadata_coverage": coverage}

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._v2_runner_rows(race, source, evidence_root=evidence_root)

    assert rejected.value.details["reason"] == "runner_duplicate_or_invalid"


@pytest.mark.parametrize(
    ("first_aliases", "second_aliases", "second_id"),
    [
        (["ALIAS", "ALIAS"], [], "Race 6 - GUNN - 2026-07-19"),
        (["ALIAS"], ["ALIAS"], "Race 6 - GUNN - 2026-07-19"),
        (["Race 6 - GUNN - 2026-07-19"], [], "Race 6 - GUNN - 2026-07-19"),
    ],
)
def test_current_index_rejects_alias_collisions(
    first_aliases: list[str], second_aliases: list[str], second_id: str
):
    rows = [
        {
            "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
            "race_id": "Race 5 - GUNN - 2026-07-19", "race_id_aliases": first_aliases,
            "race_number": 5, "race_time": "1:00 PM",
            "race_url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5",
            "venue": "GUNN",
        },
        {
            "date": "2026-07-19", "jump_datetime": "2026-07-19T13:30:00+10:00",
            "race_id": second_id, "race_id_aliases": second_aliases,
            "race_number": 6, "race_time": "13:30",
            "race_url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/6",
            "venue": "GUNN",
        },
    ]
    with pytest.raises(CaptureOneRejected) as rejected:
        capture._normalize_current_index_rows(
            {"selected_count": 2, "selected_races": rows}, max_races=32
        )
    assert rejected.value.code == "CURRENT_INDEX_INVALID"


def test_current_index_requires_time_and_normalizes_producer_time():
    row = race_window_record(
        {
            "date": "2026-07-19",
            "race_number": 5,
            "race_time": "1:00 PM",
            "url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5",
            "venue": "GUNN",
        },
        now=datetime.fromisoformat("2026-07-19T12:00:00+10:00"),
    )
    normalized = capture._normalize_current_index_rows(
        {"selected_count": 1, "selected_races": [row]}, max_races=32
    )[0]
    assert row["race_id"] in row["race_id_aliases"]
    assert normalized["race_id"] == "Race 5 - GUNN - 2026-07-19"
    assert normalized["jump_datetime"] == "2026-07-19T13:00:00+10:00"
    assert normalized["race_time"] == "13:00"
    row["race_time"] = ""
    with pytest.raises(CaptureOneRejected):
        capture._normalize_current_index_rows(
            {"selected_count": 1, "selected_races": [row]}, max_races=32
        )


@pytest.mark.parametrize("suffix", ["?trial=true", "?foo=bar", "#fragment"])
def test_current_index_rejects_unsafe_race_url_aliases(suffix: str):
    row = race_window_record(
        {
            "date": "2026-07-19",
            "race_number": 5,
            "race_time": "1:00 PM",
            "url": f"https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5{suffix}",
            "venue": "GUNN",
        },
        now=datetime.fromisoformat("2026-07-19T12:00:00+10:00"),
    )

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._normalize_current_index_rows(
            {"selected_count": 1, "selected_races": [row]}, max_races=32
        )

    assert rejected.value.code == "CURRENT_INDEX_INVALID"


def test_current_index_accepts_utc_z_runner_observation_timestamp():
    observed = capture._parse_current_index_datetime("2026-08-11T09:34:06Z")

    assert observed.isoformat() == "2026-08-11T09:34:06+00:00"


@pytest.mark.parametrize("mutation", ["arbitrary", "extra", "substitution", "omission"])
def test_current_index_rejects_noncanonical_aliases(mutation: str):
    row = race_window_record(
        {
            "date": "2026-07-19",
            "race_number": 5,
            "race_time": "1:00 PM",
            "url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5",
            "venue": "GUNN",
        },
        now=datetime.fromisoformat("2026-07-19T12:00:00+10:00"),
    )
    canonical = row["race_id_aliases"]
    row["race_id_aliases"] = {
        "arbitrary": ["ARBITRARY"],
        "extra": [*canonical, "ARBITRARY"],
        "substitution": ["ARBITRARY", *canonical[1:]],
        "omission": canonical[:-1],
    }[mutation]

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._normalize_current_index_rows(
            {"selected_count": 1, "selected_races": [row]}, max_races=32
        )

    assert rejected.value.code == "CURRENT_INDEX_INVALID"


def test_safe_file_bytes_reads_one_stable_descriptor(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    source = evidence_root / "source.json"
    source.parent.mkdir()
    source.write_bytes(b'{"stable":true}\n')

    assert capture._safe_file_bytes(
        source,
        evidence_root=evidence_root,
        missing_code="CURRENT_INDEX_SOURCE_MISSING",
    ) == b'{"stable":true}\n'


@pytest.mark.parametrize("kind", ["outside", "symlink", "oversize", "nonregular"])
def test_safe_file_bytes_rejects_unsafe_types_and_sizes(
    tmp_path: Path,
    kind: str,
):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    source = evidence_root / "source"
    if kind == "outside":
        source = tmp_path / "outside"
        source.write_bytes(b"outside")
        expected = "CURRENT_INDEX_PATH_UNSAFE"
    elif kind == "symlink":
        target = evidence_root / "target"
        target.write_bytes(b"target")
        source.symlink_to(target)
        expected = "CURRENT_INDEX_SOURCE_MISSING"
    elif kind == "oversize":
        source.write_bytes(b"x" * (capture.MAX_CURRENT_INDEX_BYTES + 1))
        expected = "CURRENT_INDEX_SIZE_INVALID"
    else:
        source.mkdir()
        expected = "CURRENT_INDEX_SOURCE_MISSING"

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._safe_file_bytes(
            source,
            evidence_root=evidence_root,
            missing_code="CURRENT_INDEX_SOURCE_MISSING",
        )

    assert rejected.value.code == expected


def test_safe_file_bytes_rejects_replacement_between_validation_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    source = evidence_root / "source"
    source.write_bytes(b"original")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    real_open = os.open

    def replace_then_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if str(path) == source.name:
            source.unlink()
            source.symlink_to(outside)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(capture.os, "open", replace_then_open)

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._safe_file_bytes(
            source,
            evidence_root=evidence_root,
            missing_code="CURRENT_INDEX_SOURCE_MISSING",
        )

    assert rejected.value.code == "CURRENT_INDEX_SOURCE_MISSING"


def test_safe_file_bytes_rejects_replacement_after_open_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    source = evidence_root / "source"
    source.write_bytes(b"original")
    replacement = evidence_root / "replacement"
    replacement.write_bytes(b"changed")
    real_read = os.read
    replaced = False

    def replace_then_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, source)
        return real_read(descriptor, size)

    monkeypatch.setattr(capture.os, "read", replace_then_read)

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._safe_file_bytes(
            source,
            evidence_root=evidence_root,
            missing_code="CURRENT_INDEX_SOURCE_MISSING",
        )

    assert rejected.value.code == "CURRENT_INDEX_PATH_UNSAFE"
    assert rejected.value.details["reason"] == "path_replaced"


def test_safe_file_bytes_rejects_same_inode_same_size_mutate_read_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    source = evidence_root / "source"
    source.write_bytes(b"original")
    real_read = os.read
    attacked = False

    def mutate_read_restore(descriptor: int, size: int) -> bytes:
        nonlocal attacked
        raw = real_read(descriptor, size)
        if not attacked and raw:
            attacked = True
            writable = os.open(source, os.O_RDWR)
            try:
                os.pwrite(writable, b"X", 0)
                os.pwrite(writable, raw[:1], 0)
            finally:
                os.close(writable)
            before = source.stat()
            os.utime(
                source,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1),
            )
        return raw

    monkeypatch.setattr(capture.os, "read", mutate_read_restore)
    with pytest.raises(CaptureOneRejected) as rejected:
        capture._safe_file_bytes(
            source, evidence_root=evidence_root,
            missing_code="CURRENT_INDEX_SOURCE_MISSING",
        )

    assert rejected.value.code == "CURRENT_INDEX_PATH_UNSAFE"
    assert rejected.value.details["reason"] == "file_mutated"


def test_atomic_publish_rejects_replaced_temporary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "evidence"
    root.mkdir()
    target = root / "runtime/index.json"
    real_replace = os.replace

    def replace_temp(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        source_path = root / "runtime" / str(source)
        raw = source_path.read_bytes()
        source_path.unlink()
        source_path.write_bytes(raw)
        real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(capture.os, "replace", replace_temp)
    with pytest.raises(CaptureOneRejected) as rejected:
        capture._atomic_replace_canonical(target, {"ok": True}, evidence_root=root)
    assert rejected.value.details["reason"] == "publish_final_replaced"


def test_atomic_publish_rejects_publication_root_swap_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "evidence"
    root.mkdir()
    target = root / "runtime/index.json"
    real_replace = os.replace
    swapped = False

    def swap_root(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            parked = tmp_path / "parked"
            real_replace(root, parked)
            real_replace(parked, root)
        real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(capture.os, "replace", swap_root)
    with pytest.raises(CaptureOneRejected) as rejected:
        capture._atomic_replace_canonical(target, {"ok": True}, evidence_root=root)
    assert rejected.value.details["reason"] == "publish_root_parent_mutated"


@pytest.mark.parametrize("attack_at", ["after_rename", "after_final_fsync"])
def test_atomic_publish_rejects_mutation_after_rename_and_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack_at: str,
):
    root = tmp_path / "evidence"
    root.mkdir()
    target = root / "runtime/index.json"
    real_fsync = os.fsync
    calls = 0

    def mutate_restore(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        real_fsync(descriptor)
        selected = calls == (2 if attack_at == "after_rename" else 3)
        if selected:
            published_fd = os.open(
                target, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                original = os.pread(published_fd, 1, 0)
                os.pwrite(published_fd, b"X", 0)
                os.pwrite(published_fd, original, 0)
                published = target.stat()
                os.utime(
                    target,
                    ns=(published.st_atime_ns, published.st_mtime_ns + 1),
                )
            finally:
                os.close(published_fd)

    monkeypatch.setattr(capture.os, "fsync", mutate_restore)
    with pytest.raises(CaptureOneRejected) as rejected:
        capture._atomic_replace_canonical(target, {"ok": True}, evidence_root=root)
    assert rejected.value.details["reason"] == "publish_final_mutated"


@pytest.mark.parametrize(
    "case",
    [
        "csv_sidecar_mismatch",
        "csv_malformed_encoding",
        "accepted_status_fail",
        "active_status_missing",
        "observation_beyond_bound",
        "post_jump_observation",
        "post_jump_generation",
    ],
)
def test_v2_runner_seal_rejects_untrusted_runner_sources(
    tmp_path: Path, case: str
):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)
    record = coverage["races"][0]
    csv_path = Path(record["csv_path"])
    sidecar_path = Path(record["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_bytes())
    generated = observed
    if case == "csv_sidecar_mismatch":
        csv_path.write_bytes(b"box|dog_name\n1|Alpha\n2|Gamma\n")
    elif case == "csv_malformed_encoding":
        csv_path.write_bytes(b"box|dog_name\n1|\xff\n")
    elif case == "accepted_status_fail":
        sidecar["prejump_shadow_metadata"]["status"] = "FAIL"
    elif case == "active_status_missing":
        sidecar["runner_completeness_after_canonical_alignment"].pop("status")
    elif case == "observation_beyond_bound":
        generated = observed - timedelta(seconds=1201)
    elif case == "post_jump_observation":
        sidecar["prejump_shadow_metadata"]["metadata_captured_at"] = (
            datetime.fromisoformat("2026-07-19T13:00:00+10:00")
        ).isoformat()
    elif case == "post_jump_generation":
        generated = datetime.fromisoformat("2026-07-19T13:00:00+10:00")
    sidecar_path.write_bytes(canonical_bytes(sidecar))
    race = {
        "date": "2026-07-19",
        "jump_datetime": "2026-07-19T13:00:00+10:00",
        "race_number": 5,
        "race_url": race_url,
        "venue": "GUNN",
    }
    source = {
        "generated_at": generated.isoformat(),
        "sidecar_metadata_coverage": coverage,
    }

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._v2_runner_rows(race, source, evidence_root=evidence_root)

    assert rejected.value.code == "CURRENT_INDEX_SOURCE_INVALID"


@pytest.mark.parametrize(
    "state",
    [pytest.param(None, id="missing"), pytest.param("NULL", id="null"),
     "active", "UNKNOWN", "PARTIAL", "RESERVE", "SCRATCHED", "INACTIVE"],
)
def test_v2_runner_seal_requires_explicit_exact_active_state(
    tmp_path: Path, state: str | None
):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)
    sidecar_path = Path(coverage["races"][0]["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_bytes())
    participant = sidecar["runner_completeness_after_canonical_alignment"]["participants"][0]
    if state is None:
        participant.pop("scratch_state")
    elif state == "NULL":
        participant["scratch_state"] = None
    else:
        participant["scratch_state"] = state
    sidecar_path.write_bytes(canonical_bytes(sidecar))

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._v2_runner_rows(
            {
                "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
                "race_number": 5, "race_url": race_url, "venue": "GUNN",
            },
            {"generated_at": observed.isoformat(), "sidecar_metadata_coverage": coverage},
            evidence_root=evidence_root,
        )

    assert rejected.value.code == "CURRENT_INDEX_SOURCE_INVALID"


def test_v2_runner_identity_is_stable_across_source_observations(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)
    race = {
        "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
        "race_number": 5, "race_url": race_url, "venue": "GUNN",
    }
    first = capture._v2_runner_rows(
        race, {"generated_at": observed.isoformat(), "sidecar_metadata_coverage": coverage},
        evidence_root=evidence_root,
    )
    changed_at = observed + timedelta(seconds=1)
    second = capture._v2_runner_rows(
        race,
        {"generated_at": changed_at.isoformat(), "sidecar_metadata_coverage": coverage},
        evidence_root=evidence_root,
    )

    assert first[1]["source_generated_at"] == observed.isoformat()
    assert second[1]["source_generated_at"] == changed_at.isoformat()
    assert first[2] == second[2] == runner_set_sha256(RUNNERS)


def test_v2_runner_seal_accepts_later_prejump_observation_and_normalized_venue(
    tmp_path: Path,
):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/townsville/2026-07-19/5"
    generated = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    observed = generated + timedelta(seconds=5)
    coverage = _runner_coverage(evidence_root, race_url, observed)
    sidecar_path = Path(coverage["races"][0]["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["prejump_shadow_metadata"]["venue"] = "TOWNSVILLE"
    sidecar["prejump_shadow_metadata"]["runner_box_name_list"][0]["dog_name"] = "Alpha Display"
    sidecar["runner_completeness_after_canonical_alignment"]["participants"][0]["dog_name"] = "Alpha Display"
    Path(coverage["races"][0]["csv_path"]).write_bytes(
        b"box|dog_name\n1|Alpha Display\n2|Beta\n"
    )
    sidecar_path.write_bytes(canonical_bytes(sidecar))

    rows, _, _ = capture._v2_runner_rows(
        {
            "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
            "race_number": 5, "race_url": race_url, "venue": "TWN",
        },
        {"generated_at": generated.isoformat(), "sidecar_metadata_coverage": coverage},
        evidence_root=evidence_root,
    )

    assert rows[0]["display_name"] == "Alpha Display"


def test_v2_runner_seal_rejects_unrelated_venue(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/townsville/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._v2_runner_rows(
            {
                "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
                "race_number": 5, "race_url": race_url, "venue": "TWN",
            },
            {"generated_at": observed.isoformat(), "sidecar_metadata_coverage": coverage},
            evidence_root=evidence_root,
        )

    assert rejected.value.details["reason"] == "runner_race_identity_mismatch"


def test_v2_runner_seal_accepts_expert_history_current_runner_prefixes(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)
    Path(coverage["races"][0]["csv_path"]).write_bytes(
        b"Dog Name|Sex|PLC|BOX|WGT|DIST|DATE|TRACK\n"
        b"1. Alpha|D|1|7|30|400|2026-07-01|GUNN\n"
        b"|D|2|4|30|400|2026-06-20|GUNN\n"
        b"2. Beta|B|3|1|29|400|2026-07-01|GUNN\n"
        b"|B|1|8|29|400|2026-06-20|GUNN\n"
    )

    rows, _, _ = capture._v2_runner_rows(
        {
            "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
            "race_number": 5, "race_url": race_url, "venue": "GUNN",
        },
        {"generated_at": observed.isoformat(), "sidecar_metadata_coverage": coverage},
        evidence_root=evidence_root,
    )

    assert [(row["box"], row["identity"]) for row in rows] == [(1, "ALPHA"), (2, "BETA")]


@pytest.mark.parametrize("first_name", ["Alpha", "1 Alpha", "2. Alpha", "1. Gamma"])
def test_v2_runner_seal_rejects_invalid_expert_history_current_prefix(
    tmp_path: Path, first_name: str,
):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)
    Path(coverage["races"][0]["csv_path"]).write_text(
        "Dog Name|PLC|BOX|DATE|TRACK\n"
        f"{first_name}|1|7|2026-07-01|GUNN\n"
        "2. Beta|2|1|2026-07-01|GUNN\n",
        encoding="utf-8",
    )

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._v2_runner_rows(
            {
                "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
                "race_number": 5, "race_url": race_url, "venue": "GUNN",
            },
            {"generated_at": observed.isoformat(), "sidecar_metadata_coverage": coverage},
            evidence_root=evidence_root,
        )

    assert rejected.value.details["reason"] in {
        "csv_runner_rows_invalid", "csv_sidecar_runner_mismatch"
    }


def test_v2_runner_seal_accepts_matching_name_prefix_with_explicit_box(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)
    csv_path = Path(coverage["races"][0]["csv_path"])
    csv_path.write_bytes(b"box|dog_name\n1|1. Alpha\n2|2. Beta\n")

    rows, _, _ = capture._v2_runner_rows(
        {
            "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
            "race_number": 5, "race_url": race_url, "venue": "GUNN",
        },
        {"generated_at": observed.isoformat(), "sidecar_metadata_coverage": coverage},
        evidence_root=evidence_root,
    )

    assert [(row["box"], row["identity"]) for row in rows] == [(1, "ALPHA"), (2, "BETA")]


@pytest.mark.parametrize(
    "name", ["2. Alpha", "1 Alpha", "1: Alpha", "123. Alpha", "1. 2. Alpha"]
)
def test_v2_runner_seal_rejects_invalid_name_prefix_with_explicit_box(
    tmp_path: Path, name: str
):
    evidence_root = tmp_path / "evidence"
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    observed = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    coverage = _runner_coverage(evidence_root, race_url, observed)
    csv_path = Path(coverage["races"][0]["csv_path"])
    csv_path.write_text(f"box|dog_name\n1|{name}\n2|Beta\n", encoding="utf-8")

    with pytest.raises(CaptureOneRejected) as rejected:
        capture._v2_runner_rows(
            {
                "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
                "race_number": 5, "race_url": race_url, "venue": "GUNN",
            },
            {"generated_at": observed.isoformat(), "sidecar_metadata_coverage": coverage},
            evidence_root=evidence_root,
        )

    assert rejected.value.details["reason"] == "csv_runner_rows_invalid"


def test_v2_requires_matching_successful_retained_publication(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    state = evidence_root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    source = evidence_root / "run/odds_capture_refresh_report.json"
    source.parent.mkdir(parents=True)
    race_url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    now = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    source.write_bytes(canonical_bytes({
        "status": "SUCCESS", "generated_at": now.isoformat(),
        "sidecar_metadata_coverage": _runner_coverage(evidence_root, race_url, now),
        "selected_count": 1,
        "selected_races": [{
            "date": "2026-07-19", "jump_datetime": "2026-07-19T13:00:00+10:00",
            "race_id": "Race 5 - GUNN - 2026-07-19",
            "race_id_aliases": [
                "Race 5 - GUNN - 2026-07-19",
                "Race 5 - GUNNEDAH - 2026-07-19",
            ],
            "race_number": 5, "race_time": "13:00", "race_url": race_url,
            "venue": "GUNN",
        }],
    }))
    published = publish_current_race_index(
        state_path=state, evidence_root=evidence_root,
        source_refresh_report_path=source, run_id="fixture",
    )
    publication_path = state.parent / capture.CURRENT_RACE_INDEX_PUBLICATION_FILENAME
    publication_bytes = publication_path.read_bytes()
    publication_path.unlink()
    with pytest.raises(CaptureOneRejected) as absent:
        bounded_current_race_index(
            current_time=now, timeout_seconds=1,
            index_path=current_race_index_path(state), evidence_root=evidence_root,
            max_age_seconds=900,
        )
    assert absent.value.code == "CURRENT_INDEX_PUBLICATION_MISSING"

    publication_path.write_bytes(publication_bytes)
    report = json.loads(publication_bytes)
    report["packet_sha256"] = "0" * 64
    publication_path.write_bytes(canonical_bytes(report))
    with pytest.raises(CaptureOneRejected) as mismatched:
        bounded_current_race_index(
            current_time=now, timeout_seconds=1,
            index_path=current_race_index_path(state), evidence_root=evidence_root,
            max_age_seconds=900,
        )
    assert mismatched.value.code == "CURRENT_INDEX_PUBLICATION_INVALID"

    index_path = current_race_index_path(state)
    legacy = json.loads(index_path.read_bytes())
    legacy["schema_version"] = capture.CURRENT_RACE_INDEX_V1_SCHEMA
    legacy["source_refresh_report_path"] = str(source)
    legacy["races"] = [
        {key: value for key, value in row.items() if key not in {
            "runners", "runner_set_sha256", "runner_source"
        }}
        for row in legacy["races"]
    ]
    index_path.write_bytes(canonical_bytes(legacy))
    assert bounded_current_race_index(
        current_time=now, timeout_seconds=1, index_path=index_path,
        evidence_root=evidence_root, max_age_seconds=900,
    )[0]["race_id"] == "Race 5 - GUNN - 2026-07-19"


def test_capture_one_success_is_one_capture_one_receipt_and_one_consumption(
    tmp_path: Path,
):
    deps, calls = dependencies(tmp_path)
    protocol, result = run(tmp_path, deps)

    assert result["status"] == "RECEIPT_READY"
    assert result["appended_attempt_count"] == 1
    assert calls == {"refresh": 1, "execute": 1, "acquire": 1, "release": 1}
    consumed = protocol.consume_response(result["request_id"], now=NOW + timedelta(seconds=3))
    assert consumed["consume"]["consume_once"] is True
    with pytest.raises(ProtocolRejected, match="RESPONSE_ALREADY_CONSUMED"):
        protocol.consume_response(result["request_id"], now=NOW + timedelta(seconds=4))
    assert not protocol.outstanding_request_ids()


def test_scheduled_capture_publishes_bounded_alias_receipts_for_reuse(
    tmp_path: Path,
):
    source_race_id = "Race 2 - LADBROKES-Q1-LAKESIDE - 2026-07-30"
    alias_race_id = "Race 2 - QOT - 2026-07-30"
    race_url = "https://www.thedogs.com.au/racing/q-straight/2026-07-30/2"
    evidence_root = tmp_path / "evidence"
    protocol = ManualPredictionCollectorProtocol(
        evidence_root / "manual_prediction_collector_requests_v1"
    )
    output_dir = evidence_root / "scheduled-capture"
    output_dir.mkdir(parents=True)
    form = output_dir / "Race 1 - WARRNAMBOOL - 2026-07-30.csv"
    form.write_text("dog_name,box_number\nAlpha,1\nBeta,2\n", encoding="utf-8")
    sidecar = form.with_name(form.name + ".metadata.json")
    sidecar.write_bytes(canonical_bytes({"race_id": RACE_ID}))
    item = {
        **plan_item(form),
        "race_id": source_race_id,
        "race_id_aliases": [
            alias_race_id,
            "Race 2 - Q STRAIGHT - 2026-07-30",
        ],
        "venue": "LADBROKES-Q1-LAKESIDE",
        "race_number": 2,
        "thedogs_source_url": race_url,
    }
    attempt = {
        **successful_report()["attempts"][0],
        "race_id": source_race_id,
    }
    attempt["append_report"] = {
        **attempt["append_report"],
        "race_id": source_race_id,
    }

    result = publish_scheduled_capture_receipts(
        protocol=protocol,
        evidence_root=evidence_root,
        collector_run_id="20260730T165500+1000_odds_capture",
        plan_item=item,
        attempt=attempt,
        output_dir=output_dir,
        emitted_at=NOW + timedelta(seconds=3),
    )

    assert result["status"] == "PUBLISHED"
    assert result["receipt_count"] >= 2
    reused = protocol.discover_collector_exact_handoff(
        race_id=alias_race_id,
        current_time=NOW + timedelta(seconds=4),
        max_age_seconds=900,
    )
    assert reused is not None
    assert reused["race_id"] == alias_race_id
    assert reused["race"]["url"] == race_url
    assert reused["append_timestamp"] == (NOW + timedelta(seconds=2)).isoformat()
    assert reused["_form_bytes"] == form.read_bytes()
    assert not protocol.outstanding_request_ids()
    receipt_path = protocol.collector_exact_receipt_path(
        alias_race_id,
        reused["capture_attempt_sha256"],
    )
    receipt_raw = receipt_path.read_bytes()
    receipt_value = json.loads(receipt_raw)
    receipt_value["sealed_handoff"]["append_report_sha256"] = "f" * 64
    receipt_path.write_bytes(canonical_bytes(receipt_value))
    with pytest.raises(ProtocolRejected, match="HASH_DRIFT"):
        protocol.discover_collector_exact_handoff(
            race_id=alias_race_id,
            current_time=NOW + timedelta(seconds=5),
            max_age_seconds=900,
        )
    receipt_path.write_bytes(receipt_raw)
    form.write_bytes(form.read_bytes() + b"tampered")
    with pytest.raises(ProtocolRejected, match="HASH_DRIFT"):
        protocol.discover_collector_exact_handoff(
            race_id=alias_race_id,
            current_time=NOW + timedelta(seconds=5),
            max_age_seconds=900,
        )


def test_capture_one_returns_immediate_busy_with_owner_and_phase(tmp_path: Path):
    def busy(**kwargs: Any) -> object:
        del kwargs
        raise CollectorBusy(
            {
                "lock_owner_run_id": "scheduled-collector",
                "lock_owner_pid": 1234,
                "lock_owner_phase": "sportsbet_fetch",
            }
        )

    deps, calls = dependencies(tmp_path, acquire=busy)
    protocol, result = run(tmp_path, deps)

    assert result["status"] == "BUSY"
    assert result["busy"] == {
        "lock_owner_run_id": "scheduled-collector",
        "lock_owner_pid": 1234,
        "lock_owner_phase": "sportsbet_fetch",
    }
    assert calls == {"refresh": 0, "execute": 0, "acquire": 1, "release": 0}
    assert protocol.read_response(result["request_id"])["status"] == "CAPTURE_FAILED"
    assert not protocol.outstanding_request_ids()


def test_capture_one_cli_emits_machine_only_busy_json(tmp_path: Path):
    now = datetime.now().astimezone()
    jump = now + timedelta(minutes=10)
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_bytes(b"")
    lock_path = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_daemon_runtime/shadow_autopilot.lock"
    )
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(
        canonical_bytes(
            {
                "schema_version": "shadow_autopilot_daemon_lock_v1",
                "run_id": "scheduled_fixture",
                "pid": 123,
                "hostname": "fixture",
                "started_at": now.isoformat(),
                "output_dir": str(tmp_path / "scheduled"),
                "phase": "odds_capture",
            }
        )
    )
    evidence_root = tmp_path / "evidence"
    protocol = ManualPredictionCollectorProtocol(
        evidence_root / "manual_prediction_collector_requests_v1"
    )
    request = protocol.publish_request(
        race={
            "race_id": f"Race 1 - QOT - {now.date().isoformat()}",
            "url": (
                "https://www.thedogs.com.au/racing/ladbrokes-q1-lakeside/"
                f"{now.date().isoformat()}/1/fixture?trial=false"
            ),
            "venue": "QOT",
            "race_number": 1,
            "race_date": now.date().isoformat(),
            "jump_timestamp": jump.isoformat(),
        },
        expected_runners=[],
        created_at=now,
        expires_at=jump,
    )

    result = invoke_capture_one(
        command=[
            sys.executable,
            str(Path("scripts/shadow_autopilot_daemon.py").resolve()),
            "capture-one",
            "--evidence-root",
            str(evidence_root),
            "--protocol-root",
            str(protocol.root),
            "--request-id",
            str(request["request_id"]),
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--output-dir",
            str(evidence_root / "capture-one"),
            "--minimum-margin-seconds",
            "114",
            "--minimum-post-lock-margin-seconds",
            "113",
            "--minimum-fetch-margin-seconds",
            "98",
            "--fetch-timeout-seconds",
            "45",
        ],
        timeout_seconds=10,
    )

    assert result["status"] == "BUSY"
    assert result["busy"]["lock_owner_run_id"] == "scheduled_fixture"
    assert result["busy"]["lock_owner_phase"] == "odds_capture"
    assert not protocol.outstanding_request_ids()


def test_capture_failure_terminalizes_request_and_releases_lock(tmp_path: Path):
    failed = successful_report()
    failed["final_status"] = "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    failed["appended_attempt_count"] = 0
    failed["attempts"][0]["status"] = "BLOCKED_FETCH_EXCEPTION"
    failed["attempts"][0]["reasons"] = ["fetch_exception:RuntimeError"]
    deps, calls = dependencies(tmp_path, report=failed)
    protocol, result = run(tmp_path, deps)

    assert result["status"] == "CAPTURE_FAILED"
    assert calls["release"] == 1
    assert not protocol.outstanding_request_ids()


def test_cancellation_before_sealing_terminalizes_and_releases(tmp_path: Path):
    def cancel(phase: str) -> None:
        if phase == "before_seal":
            raise CaptureCancelled("test cancellation")

    deps, calls = dependencies(tmp_path, phase_hook=cancel)
    protocol, result = run(tmp_path, deps)

    assert result["status"] == "CANCELLED"
    assert calls["release"] == 1
    assert not protocol.outstanding_request_ids()
    assert protocol.discover_exact_handoff(
        race_id=RACE_ID,
        current_time=NOW + timedelta(seconds=3),
        max_age_seconds=900,
    ) is None


def test_cancellation_after_sealing_leaves_reusable_exact_receipt(tmp_path: Path):
    def cancel(phase: str) -> None:
        if phase == "after_seal":
            raise CaptureCancelled("test cancellation")

    deps, calls = dependencies(tmp_path, phase_hook=cancel)
    protocol, result = run(tmp_path, deps)

    assert result["status"] == "RECEIPT_READY"
    assert calls["release"] == 1
    reused = protocol.discover_exact_handoff(
        race_id=RACE_ID,
        current_time=NOW + timedelta(seconds=3),
        max_age_seconds=900,
    )
    assert reused is not None
    assert reused["race_id"] == RACE_ID
    assert json.loads(reused["_report_bytes"])["appended_attempt_count"] == 1


def test_capture_one_rejects_insufficient_margin_before_lock_or_browser(tmp_path: Path):
    deps, calls = dependencies(tmp_path)
    protocol, result = run(tmp_path, deps, minimum_margin_seconds=1201)

    assert result["status"] == "INSUFFICIENT_PREJUMP_MARGIN"
    assert calls == {"refresh": 0, "execute": 0, "acquire": 0, "release": 0}
    assert not protocol.outstanding_request_ids()


def test_capture_one_rechecks_margin_after_lock_before_exact_refresh(tmp_path: Path):
    deps, calls = dependencies(tmp_path)
    clock = {"calls": 0}

    def now() -> datetime:
        clock["calls"] += 1
        return NOW if clock["calls"] == 1 else JUMP - timedelta(seconds=100)

    deps.now = now
    protocol, result = run(tmp_path, deps)

    assert result["status"] == "INSUFFICIENT_PREJUMP_MARGIN"
    assert calls == {"refresh": 0, "execute": 0, "acquire": 1, "release": 1}
    assert not protocol.outstanding_request_ids()


def test_capture_one_rechecks_margin_before_sportsbet_fetch(tmp_path: Path):
    deps, calls = dependencies(tmp_path)
    clock = {"now": NOW}
    original_build = deps.build_plan_item

    def build_then_advance(
        form_path: Path,
        current_time: datetime,
    ) -> Mapping[str, Any]:
        result = original_build(form_path, current_time)
        clock["now"] = JUMP - timedelta(seconds=100)
        return result

    deps.now = lambda: clock["now"]
    deps.build_plan_item = build_then_advance
    protocol, result = run(tmp_path, deps)

    assert result["status"] == "INSUFFICIENT_PREJUMP_MARGIN"
    assert calls == {"refresh": 1, "execute": 0, "acquire": 1, "release": 1}
    assert not protocol.outstanding_request_ids()


def test_sigterm_cancellation_reaps_collector_browser_process_group(tmp_path: Path):
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(60)"
    )

    readiness_errors: list[str] = []

    def signal_after_child_pid_is_published() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                int(child_pid_path.read_text())
            except (FileNotFoundError, ValueError):
                time.sleep(0.01)
                continue
            os.kill(os.getpid(), signal.SIGTERM)
            return
        readiness_errors.append("child.pid was not published before the readiness deadline")

    signal_thread = threading.Thread(target=signal_after_child_pid_is_published)
    signal_thread.start()
    try:
        with pytest.raises(CaptureOneRejected, match="CANCELLED"):
            invoke_capture_one(
                command=[sys.executable, "-c", code, str(child_pid_path)],
                timeout_seconds=10,
            )
    finally:
        signal_thread.join()

    assert not readiness_errors
    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{child_pid}").exists()
