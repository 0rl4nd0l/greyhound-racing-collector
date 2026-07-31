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
    run_capture_one,
)
from src.predictor.on_demand import canonical_bytes, sha256_bytes

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
    tmp_path: Path,
):
    evidence_root = tmp_path / "evidence"
    state = evidence_root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    source = evidence_root / "shadow_autopilot_v1_fixture/odds_capture_refresh_report.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        canonical_bytes(
            {
                "generated_at": NOW.isoformat(),
                "selected_count": 1,
                "selected_races": [
                    {
                        "date": "2026-07-19",
                        "jump_datetime": "2026-07-19T13:00:00+10:00",
                        "race_id": "Race 5 - GUNN - 2026-07-19",
                        "race_id_aliases": ["Race 5 - GUNN - 2026-07-19"],
                        "race_number": 5,
                        "race_time": "13:00",
                        "race_url": (
                            "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
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

    assert published["status"] == "PUBLISHED"
    assert json.loads(original)["source_refresh_report_sha256"] == sha256_bytes(
        source.read_bytes()
    )
    assert bounded_current_race_index(
        current_time=NOW,
        timeout_seconds=1,
        index_path=index_path,
        evidence_root=evidence_root,
        max_age_seconds=900,
    )[0]["race_id"] == "Race 5 - GUNN - 2026-07-19"

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


def test_current_race_index_rejects_stale_or_changed_source(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    state = evidence_root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    source = evidence_root / "shadow_autopilot_v1_fixture/odds_capture_refresh_report.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        canonical_bytes(
            {
                "generated_at": (NOW - timedelta(minutes=20)).isoformat(),
                "selected_count": 1,
                "selected_races": [
                    {
                        "date": "2026-07-19",
                        "jump_datetime": "2026-07-19T13:00:00+10:00",
                        "race_id": "Race 5 - GUNN - 2026-07-19",
                        "race_id_aliases": [],
                        "race_number": 5,
                        "race_time": "13:00",
                        "race_url": (
                            "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
                        ),
                        "venue": "GUNN",
                    }
                ],
            }
        )
    )
    publish_current_race_index(
        state_path=state,
        evidence_root=evidence_root,
        source_refresh_report_path=source,
        run_id="fixture",
    )
    index_path = current_race_index_path(state)

    with pytest.raises(CaptureOneRejected) as stale:
        bounded_current_race_index(
            current_time=NOW,
            timeout_seconds=1,
            index_path=index_path,
            evidence_root=evidence_root,
            max_age_seconds=900,
        )
    assert stale.value.code == "CURRENT_INDEX_STALE"

    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(CaptureOneRejected) as changed:
        bounded_current_race_index(
            current_time=NOW - timedelta(minutes=20),
            timeout_seconds=1,
            index_path=index_path,
            evidence_root=evidence_root,
            max_age_seconds=900,
        )
    assert changed.value.code == "CURRENT_INDEX_SOURCE_CHANGED"


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

    timer = threading.Timer(0.25, os.kill, args=(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        with pytest.raises(CaptureOneRejected, match="CANCELLED"):
            invoke_capture_one(
                command=[sys.executable, "-c", code, str(child_pid_path)],
                timeout_seconds=10,
            )
    finally:
        timer.cancel()

    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{child_pid}").exists()
