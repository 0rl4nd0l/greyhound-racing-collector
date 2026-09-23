import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from scripts import shadow_autopilot_daemon as daemon


@pytest.mark.parametrize("odds_only", [False, True])
@pytest.mark.parametrize("pinned_time", [False, True])
def test_new_source_observation_starts_after_lock_wait(
    tmp_path, monkeypatch, odds_only, pinned_time
):
    observed = datetime.fromisoformat("2026-07-19T12:00:00+10:00")
    acquired = observed + timedelta(seconds=600)
    clock = [observed]
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "protected_hashes", lambda: {})
    monkeypatch.setattr(daemon, "copy_if_exists", lambda *args: None)
    monkeypatch.setattr(daemon, "write_service_files", lambda **kwargs: {})
    monkeypatch.setattr(daemon, "probe_duplicate_lock", lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr(daemon, "probe_stale_lock_cleanup", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(daemon, "simulate_timeout_recovery", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(daemon, "release_lock", lambda *args, **kwargs: {"status": "RELEASED"})
    monkeypatch.setattr(daemon, "wall_clock_now", lambda: clock[0])

    def acquire(**kwargs):
        clock[0] = acquired
        return {"run_id": kwargs["run_id"]}

    acquisition = (
        "acquire_lock_with_t2_due_retry" if odds_only else "acquire_lock_with_odds_capture_retry"
    )
    monkeypatch.setattr(daemon, acquisition, acquire)

    class BoundaryReached(BaseException):
        pass

    forwarded = []

    def execute(**kwargs):
        command = kwargs["command"]
        forwarded.append(command[command.index("--current-time") + 1])
        raise BoundaryReached

    monkeypatch.setattr(daemon, "run_command", execute)
    mode = "run-odds-capture-once" if odds_only else "run-once"
    extra = ["--current-time", observed.isoformat()] if pinned_time else []
    args = daemon.parse_args(
        [
            mode,
            "--evidence-root",
            str(tmp_path / "artifacts/full_evidence_orchestration_20260525"),
            "--run-id",
            "clock",
            "--lock-path",
            str(tmp_path / "runtime/collector.lock"),
            "--state-path",
            str(tmp_path / "runtime/state.json"),
            *extra,
        ]
    )
    run = daemon.run_odds_capture_once if odds_only else daemon.run_once
    with pytest.raises(BoundaryReached):
        run(args)
    assert forwarded == [(observed if pinned_time else acquired).isoformat()]


def test_future_odds_window_does_not_cache_source_beyond_index_budget(tmp_path):
    state = tmp_path / "state.json"
    observed = datetime.fromisoformat("2026-07-19T12:00:00+10:00")
    daemon.write_json(
        state,
        {
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "odds_capture_refresh_status": "SUCCESS",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "window_state_source_updated_at": observed.isoformat(),
            "next_window_opens_at": (observed + timedelta(minutes=30)).isoformat(),
        },
    )
    assert (
        daemon.odds_capture_preflight_wait(state_path=state, now=observed + timedelta(minutes=5))
        is not None
    )
    assert (
        daemon.odds_capture_preflight_wait(state_path=state, now=observed + timedelta(seconds=601))
        is None
    )


@pytest.mark.parametrize("pinned_time", [False, True])
@pytest.mark.parametrize(
    "primary_seconds, expected_source_age", [(240, 555), (840, 1155), (1200, 1515)]
)
def test_elapsed_lock_wait_and_primary_work_use_completion_clock(
    tmp_path, monkeypatch, pinned_time, primary_seconds, expected_source_age
):
    started = datetime.fromisoformat("2026-07-19T12:00:00+10:00")
    clock = [started]
    evidence = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    lock = tmp_path / "runtime/collector.lock"
    odds_state = tmp_path / "runtime/odds.json"
    autopilot = evidence / "shadow_autopilot_v1_elapsed_daemon"
    output = evidence / "shadow_autopilot_daemonization_v1_elapsed"
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "wall_clock_now", lambda: clock[0])
    monkeypatch.setattr(daemon, "protected_hashes", lambda: {})
    monkeypatch.setattr(daemon, "copy_if_exists", lambda *args: None)
    monkeypatch.setattr(daemon, "write_service_files", lambda **kwargs: {})
    monkeypatch.setattr(daemon, "simulate_timeout_recovery", lambda *args: {"status": "PASS"})
    daemon.acquire_lock(
        lock_path=lock,
        run_id="owner_odds_capture",
        stale_after_seconds=3600,
        output_dir=evidence / "owner",
    )
    overlaps = []
    blocked_during_primary = []

    def odds_args(run_id):
        return daemon.parse_args(
            [
                "run-odds-capture-once",
                "--run-id",
                run_id,
                "--evidence-root",
                str(evidence),
                "--lock-path",
                str(lock),
                "--state-path",
                str(odds_state),
                "--current-time",
                clock[0].isoformat(),
            ]
        )

    def sleep(seconds):
        assert daemon.read_active_full_daemon_lock_wait_marker(lock) is not None
        clock[0] += timedelta(seconds=seconds)
        elapsed = (clock[0] - started).total_seconds()
        if elapsed % 135 == 0:
            overlap = daemon.run_odds_capture_once(
                odds_args(f"overlap_{int(elapsed)}_odds_capture")
            )
            overlaps.append(overlap["final_status"])
        if clock[0] >= started + timedelta(seconds=600):
            daemon.release_lock(lock, "owner_odds_capture")

    monkeypatch.setattr(
        daemon,
        "time",
        SimpleNamespace(
            sleep=sleep,
            monotonic=daemon.time.monotonic,
            time=daemon.time.time,
        ),
    )

    class DownstreamReached(BaseException):
        pass

    def downstream(**kwargs):
        raise DownstreamReached

    monkeypatch.setattr(daemon, "rejoin_pending_shadow_runs", downstream)

    def execute(**kwargs):
        if kwargs["name"] == "odds_capture_autopilot_cycle":
            clock[0] += timedelta(seconds=180)
            odds_output = evidence / "shadow_autopilot_v1_handoff_odds_capture_autopilot"
            daemon.write_json(
                odds_output / "odds_capture_refresh_report.json", {"status": "SUCCESS"}
            )
            daemon.write_json(
                odds_output / "autonomous_live_odds_capture_status.json",
                {
                    "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
                    "inserted_live_odds_rows": 0,
                },
            )
            daemon.write_json(
                kwargs["output_dir"] / "logs/odds_capture_autopilot_cycle.stdout.txt",
                {
                    "output_dir": str(odds_output),
                    "final_verdict": "AUTOPILOT_READY",
                },
            )
            return {"name": kwargs["name"], "status": "PASS", "returncode": 0}
        assert kwargs["name"] == "autopilot_cycle"
        assert clock[0] == started + timedelta(seconds=600)
        selected_at = kwargs["command"][kwargs["command"].index("--current-time") + 1]
        assert selected_at == (started if pinned_time else clock[0]).isoformat()
        daemon.write_json(
            autopilot / "odds_capture_refresh_report.json",
            {
                "status": "SUCCESS",
                "generated_at": selected_at,
                "next_preferred_window": {
                    "status": "WAITING_FOR_FUTURE_WINDOW",
                    "next_window_opens_at": (
                        started + timedelta(seconds=600 + primary_seconds - 240)
                    ).isoformat(),
                    "next_race": {
                        "race_id": "Race 1 - GUNN - 2026-07-19",
                        "jump_datetime": (
                            started + timedelta(seconds=600 + primary_seconds + 3360)
                        ).isoformat(),
                    },
                },
            },
        )
        daemon.write_json(
            autopilot / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            },
        )
        daemon.write_json(
            output / "logs/autopilot_cycle.stdout.txt", {"output_dir": str(autopilot)}
        )
        remaining = primary_seconds
        while remaining >= 135:
            clock[0] += timedelta(seconds=135)
            remaining -= 135
            held = daemon.run_odds_capture_once(odds_args(f"primary_{remaining}_odds_capture"))
            blocked_during_primary.append(held["final_status"])
        clock[0] += timedelta(seconds=remaining)
        return {"name": kwargs["name"], "status": "PASS", "returncode": 0}

    monkeypatch.setattr(daemon, "run_command", execute)
    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "elapsed",
            "--evidence-root",
            str(evidence),
            "--lock-path",
            str(lock),
            "--state-path",
            str(tmp_path / "runtime/full.json"),
            "--odds-capture-state-path",
            str(odds_state),
            "--enable-autonomous-odds-capture",
            *(["--current-time", started.isoformat()] if pinned_time else []),
        ]
    )
    if pinned_time:
        with pytest.raises(DownstreamReached):
            daemon.run_once(args)
    else:
        result = daemon.run_once(args)
        assert result["runtime_action"] == "RELEASE_FULL_DAEMON_FOR_ODDS_CAPTURE"
    decision = json.loads((output / "post_primary_odds_capture_release_decision.json").read_text())
    assert decision["should_release"] is not pinned_time
    assert decision["current_time"] == (started if pinned_time else clock[0]).isoformat()
    state = json.loads(odds_state.read_text())
    assert (
        state["window_state_source_updated_at"]
        == (started if pinned_time else started + timedelta(seconds=600)).isoformat()
    )
    assert not lock.exists()
    assert not daemon.full_daemon_lock_wait_marker_path(lock).exists()
    assert overlaps == ["SKIPPED_FULL_DAEMON_LOCK_HANDOFF"] * 4
    assert blocked_during_primary == ["SKIPPED_LOCK_HELD"] * (primary_seconds // 135)
    if not pinned_time:
        clock[0] += timedelta(seconds=135)
        odds = daemon.run_odds_capture_once(odds_args("handoff_odds_capture"))
        assert odds["final_status"] == "ODDS_CAPTURE_ONLY_READY"
        assert odds["lock_release"]["released"] is True
        assert not lock.exists()
        assert (
            clock[0] - (started + timedelta(seconds=600))
        ).total_seconds() == expected_source_age


def test_odds_failed_subprocess_does_not_claim_publication_was_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(
        daemon,
        "run_command",
        lambda **kwargs: {
            "name": kwargs["name"],
            "status": "FAIL",
            "returncode": 124,
            "timed_out": True,
        },
    )
    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "timeout_odds_capture",
            "--evidence-root",
            str(tmp_path / "artifacts/full_evidence_orchestration_20260525"),
            "--lock-path",
            str(tmp_path / "runtime/collector.lock"),
            "--state-path",
            str(tmp_path / "runtime/odds.json"),
            "--current-time",
            "2026-07-19T12:00:00+10:00",
        ]
    )
    report = daemon.run_odds_capture_once(args)
    assert report["current_race_index_publish"] == {
        "schema_version": "collector_current_race_index_publish_v2",
        "status": "UNAVAILABLE",
        "reason": "autopilot_output_unavailable",
    }
    assert report["lock_release"]["released"] is True
