import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts import live_collection_cycle as cycle
from scripts import shadow_autopilot_daemon as daemon
from scripts import autonomous_live_odds_capture as capture


@pytest.fixture(autouse=True)
def operational_reader_fixture(monkeypatch):
    from tests.operator_ui import test_live_adapters as helpers
    payloads, adapter = helpers.actual_payloads, helpers.make_live
    monkeypatch.setattr(helpers, "actual_payloads", lambda *args, **kwargs: payloads(*args, **{**kwargs, "include_models": False}))
    monkeypatch.setattr(helpers, "make_live", lambda *args, **kwargs: adapter(*args, **{**kwargs, "include_models": False}))


@pytest.mark.parametrize("odds_only", [False, True])
def test_native_cli_includes_interpreter_startup(monkeypatch, odds_only):
    started = []

    def run(args):
        started.append(args.live_started_monotonic)
        return {"runtime_action": "LIVE_COLLECTION_COMPLETE", "final_verdict": "DAEMON_READY"}

    before = daemon.time.monotonic()
    monkeypatch.setattr(daemon, "run_odds_capture_once" if odds_only else "run_once", run)
    assert (
        daemon.main(["run-odds-capture-once" if odds_only else "run-once", "--live-freshness"]) == 0
    )
    assert 0 < started[0] <= before


@pytest.mark.parametrize("odds_only", [False, True])
@pytest.mark.parametrize(
    "startup,verification,finalization,release_delay,expected",
    [
        (2, 0.25, 1, 0, "LIVE_COLLECTION_COMPLETE"),
        (2, 0.25, 5, 0, "LIVE_COLLECTION_COMPLETE"),
        (11, 0, 0, 0, "LIVE_TIMING_BUDGET_EXCEEDED"),
        (2, 0.25, 9, 0, "LIVE_TIMING_BUDGET_EXCEEDED"),
        (2, 0.25, 50, 0, "LIVE_TIMING_BUDGET_EXCEEDED"),
        (2, 0.25, 1, 12, "LIVE_TIMING_BUDGET_EXCEEDED"),
    ],
)
def test_native_entrypoint_accounts_for_nonzero_overhead(
    tmp_path, monkeypatch, odds_only, startup, verification, finalization, release_delay, expected
):
    from pathlib import Path

    evidence = tmp_path / "evidence"
    state = evidence / "runtime/state.json"
    lock = evidence / "runtime/collector.lock"
    clock = [datetime(2026, 7, 19, 12, tzinfo=timezone.utc)]
    if finalization == 50:
        clock[0] -= timedelta(seconds=47.5)
    elapsed = [0.0]
    observed = [clock[0]]
    writes = []

    def advance(seconds):
        elapsed[0] += seconds
        clock[0] += timedelta(seconds=seconds)

    class FixtureDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz)

    monkeypatch.setattr(daemon, "datetime", FixtureDateTime)
    monkeypatch.setattr(daemon, "wall_clock_now", lambda: clock[0])
    monkeypatch.setattr(cycle, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    release = daemon.release_lock

    def delayed_release(*args, **kwargs):
        advance(release_delay)
        return release(*args, **kwargs)

    monkeypatch.setattr(daemon, "release_lock", delayed_release)
    check_output = cycle.subprocess.check_output
    startup_pending = [True]

    def startup_delay(*args, **kwargs):
        if startup_pending[0]:
            startup_pending[0] = False
            advance(startup)
        return check_output(*args, **kwargs)

    monkeypatch.setattr(cycle.subprocess, "check_output", startup_delay)

    def view(**kwargs):
        advance(verification)
        return SimpleNamespace(
            source_generated_at=observed[0].isoformat(),
            source_refresh_report_path="empty-refresh.json",
            races=[],
        )

    monkeypatch.setattr(cycle, "bounded_current_race_index", view)
    monkeypatch.setattr(capture, "build_capture_plan", lambda *args, **kwargs: {"races": []})
    atomic = cycle.atomic_json
    finalization_pending = [True]

    def timed_write(path, payload):
        if path == state:
            assert lock.exists()
            if payload["runtime_action"] == "LIVE_TIMING_VALIDATION_PENDING":
                assert payload["status"] == "RUNNING"
                assert payload["final_status" if odds_only else "final_verdict"] == (
                    "ODDS_CAPTURE_ONLY_RUNNING" if odds_only else "DAEMON_RUNNING"
                )
            if finalization_pending[0]:
                finalization_pending[0] = False
                advance(finalization)
            writes.append(dict(payload))
        return atomic(path, payload)

    monkeypatch.setattr(cycle, "atomic_json", timed_write)

    def command(*, name, command, output_dir, **kwargs):
        assert lock.exists()
        report_name = (
            "odds_capture_only_daemon_report.json" if odds_only else "daemon_run_report.json"
        )
        running = json.loads((output_dir.parent / report_name).read_text())
        assert running["status"] == "RUNNING"
        assert running["final_status" if odds_only else "final_verdict"] == (
            "ODDS_CAPTURE_ONLY_RUNNING" if odds_only else "DAEMON_RUNNING"
        )
        from tests.operator_ui.test_live_adapters import actual_payloads, make_live

        values = actual_payloads(clock[0])
        lane = "odds" if odds_only else "full"
        values[f"{lane}_report"] = running
        adapter = make_live(
            tmp_path / "running-consumer",
            values,
            now=clock[0],
            **{f"{lane}_status": ("active", "running", running["lock"]["pid"])},
        )
        assert (
            adapter.collector(clock[0]).data["lanes"][1 if odds_only else 0]["status"] == "ACTIVE"
        )
        inactive = make_live(
            tmp_path / "inactive-consumer",
            values,
            now=clock[0],
            **{f"{lane}_status": ("inactive", "dead", 0)},
        )
        assert (
            inactive.collector(clock[0]).data["lanes"][1 if odds_only else 0]["status"]
            == "DIVERGENT"
        )
        observed[0] = clock[0]
        advance(55)
        logs = output_dir / "logs"
        logs.mkdir(parents=True)
        atomic(output_dir / "odds_capture_refresh_report.json", {"status": "SUCCESS"})
        atomic(
            logs / f"{name}.stdout.txt",
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "current_race_index_publish": {"status": "PUBLISHED"},
            },
        )
        return {"name": name, "returncode": 0}

    monkeypatch.setattr(daemon, "run_command", command)
    args = daemon.parse_args(
        [
            "run-odds-capture-once" if odds_only else "run-once",
            "--live-freshness",
            "--evidence-root",
            str(evidence),
            "--state-path",
            str(state),
            "--lock-path",
            str(lock),
            "--db",
            str(tmp_path / "never-opened.db"),
        ]
    )
    result = (daemon.run_odds_capture_once if odds_only else daemon.run_once)(args)
    if finalization == 50:
        source_age = (clock[0] - observed[0]).total_seconds()
        minute_one_trigger = datetime(2026, 7, 19, 12, 1, tzinfo=timezone.utc)
        next_latest_trigger = datetime(2026, 7, 19, 12, 3, 15, tzinfo=timezone.utc)
        assert clock[0] == minute_one_trigger
        timer_gap = (next_latest_trigger - clock[0]).total_seconds()
        assert timer_gap == 135, (
            "Conditional calendar ordering: 12:01 trigger precedes completion, "
            "minute 02 is omitted, and the next trigger is delayed to 12:03:15"
        )
        predecessor_age = source_age + timer_gap + 65
        assert predecessor_age > 300, (
            f"Conditional on 12:01 firing before completion: source age {source_age:.3f}s "
            f"+ timer gap {timer_gap:.3f}s + refresh 65s = {predecessor_age:.3f}s; "
            "expected >300s (calendar ordering assumed, not executed)"
        )
    assert result["runtime_action"] == expected
    assert not lock.exists()
    assert not (tmp_path / "never-opened.db").exists()
    timing = result["timing"]
    assert timing["startup_seconds"] == startup
    assert timing["finalization_seconds"] >= finalization
    assert timing["lock_held_seconds"] >= 55 + finalization
    assert timing["lock_wait_seconds"] == 0
    assert timing["release_seconds"] >= release_delay
    assert "process import" in timing["measurement_boundary"]
    assert json.loads(state.read_text())["runtime_action"] == expected
    if expected != "LIVE_COLLECTION_COMPLETE" and not release_delay:
        assert all(row["runtime_action"] != "LIVE_COLLECTION_COMPLETE" for row in writes)
    completed = json.loads(
        (state.parent / f"{'odds' if odds_only else 'full'}.live-phase-checkpoint.json").read_text()
    )
    result_path = Path(completed["phases"][0]["result_path"])
    assert json.loads(result_path.read_text())["status"] == "PASS"
    terminal = json.loads((Path(result["output_dir"]) / "terminal-timing.json").read_text())
    assert terminal["runtime_action"] == expected
    assert terminal["timing"]["lock_held_seconds"] == timing["lock_held_seconds"]
    if expected == "LIVE_COLLECTION_COMPLETE":
        assert completed["status"] == "LIVE_WORK_COMPLETE_TIMING_PENDING"


@pytest.mark.parametrize(
    "odds_only,refresh_seconds,resume",
    [
        (odds_only, refresh_seconds, resume)
        for odds_only in (False, True)
        for refresh_seconds in (60, 66)
        for resume in (False, "before_capture", "after_capture")
    ]
    + [(True, 60, "handoff")],
)
def test_native_entrypoint_phases_release_lock_and_fail_closed(
    tmp_path, monkeypatch, odds_only, refresh_seconds, resume
):
    evidence = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    state = evidence / "runtime/state.json"
    lock = evidence / "runtime/collector.lock"
    clock = [datetime(2026, 7, 19, 12, tzinfo=timezone.utc)]
    observed = [clock[0]]
    elapsed = [0]
    commands = []
    overlapping = []
    source = evidence / "fixture-refresh.json"
    directory = evidence / "workers/0"
    directory.mkdir(parents=True)
    (directory / "fixture.csv").write_text("fixture")
    source.write_text(
        json.dumps(
            {
                "sidecar_metadata_coverage": {
                    "races": [{"race_url": "fixture", "csv_path": str(directory / "fixture.csv")}]
                }
            }
        )
    )
    monkeypatch.setattr(daemon, "wall_clock_now", lambda: clock[0])
    monkeypatch.setattr(cycle, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))

    def view(**kwargs):
        return SimpleNamespace(
            source_generated_at=observed[0].isoformat(),
            source_refresh_report_path="fixture-refresh.json",
            packet_sha256=observed[0].isoformat(),
            races=[
                {
                    "race_id": "fixture",
                    "race_url": "fixture",
                    "source_native_race_id": "native-fixture",
                    "runner_set_sha256": "fixture-runners",
                    "jump_datetime": "2026-07-19T23:00:00+00:00",
                }
            ],
        )

    monkeypatch.setattr(cycle, "bounded_current_race_index", view)
    monkeypatch.setattr(
        capture,
        "build_capture_plan",
        lambda *args, **kwargs: {
            "races": [{"status": "READY_TO_CAPTURE", "thedogs_source_url": "fixture"}]
        },
    )
    monkeypatch.setattr(
        capture, "existing_capture_runner_status", lambda *args, **kwargs: {"status": "NONE"}
    )

    def odds_args(run_id):
        return daemon.parse_args(
            [
                "run-odds-capture-once",
                "--live-freshness",
                "--run-id",
                run_id,
                "--evidence-root",
                str(evidence),
                "--state-path",
                str(evidence / "runtime/odds.json"),
                "--lock-path",
                str(lock),
                "--db",
                str(tmp_path / "never-opened.db"),
            ]
        )

    def command(*, name, command, output_dir, **kwargs):
        assert lock.exists()
        with pytest.raises(daemon.LockBusy):
            daemon.acquire_lock(
                lock_path=lock, run_id="overlap", stale_after_seconds=9999, output_dir=output_dir
            )
        phase = command[command.index("--collection-phase") + 1]
        if not odds_only and not commands:
            overlapping.append(daemon.run_odds_capture_once(odds_args("overlap_odds_capture")))
        commands.append(phase)
        assert "--skip-shadow-run" in command
        assert "--input-retention-config" not in command
        duration = refresh_seconds if phase == "refresh" else 40
        if phase == "refresh":
            observed[0] = clock[0]
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "odds_capture_refresh_report.json").write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "generated_at": observed[0].isoformat(),
                        "selected_count": 0,
                        "selected_races": [],
                    }
                )
            )
        clock[0] += timedelta(seconds=duration)
        elapsed[0] += duration
        logs = output_dir / "logs"
        logs.mkdir(parents=True)
        (logs / f"{name}.stdout.txt").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "output_dir": str(output_dir),
                    "current_race_index_publish": {"status": "PUBLISHED"},
                    "autonomous_live_odds_capture_status": (
                        {"status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED", "inserted_rows": 2}
                        if phase == "capture"
                        else {}
                    ),
                }
            )
        )
        return {"name": name, "returncode": 0}

    monkeypatch.setattr(daemon, "run_command", command)
    args = daemon.parse_args(
        [
            "run-odds-capture-once" if odds_only else "run-once",
            "--live-freshness",
            "--run-id",
            "fixture_odds_capture" if odds_only else "fixture_full",
            "--evidence-root",
            str(evidence),
            "--state-path",
            str(state),
            "--lock-path",
            str(lock),
            "--db",
            str(tmp_path / "never-opened.db"),
        ]
        + (
            []
            if odds_only
            else [
                "--enable-autonomous-odds-capture",
                "--execute-autonomous-odds-capture",
                "--allow-auto-scrape-odds",
            ]
        )
    )
    run = daemon.run_odds_capture_once if odds_only else daemon.run_once
    if resume == "handoff":
        from tests.operator_ui.test_live_adapters import actual_payloads, make_live

        prior = actual_payloads(clock[0])["odds_state"]
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps(prior))
        before = state.read_bytes()
        monkeypatch.setattr(
            daemon,
            "read_active_full_daemon_lock_wait_marker",
            lambda *args: commands == ["refresh"],
        )
        result = run(args)
        assert result["final_status"] == "SKIPPED_FULL_DAEMON_LOCK_HANDOFF"
        assert not result.get("odds_capture_refresh_report")
        assert state.read_bytes() == before
        values = actual_payloads(clock[0])
        values["odds_report"] = result
        values["odds_state"] = prior
        native = make_live(tmp_path, values, now=clock[0]).collector(clock[0])
        assert native.data["lanes"][1]["status"] == "CAPTURE_WINDOW_CLOSED"
        assert not lock.exists()
        return
    if resume and refresh_seconds <= 65:
        acquire = daemon.acquire_lock_with_odds_capture_retry

        class Interrupted(BaseException):
            pass

        def interrupt_completed_boundary(**kwargs):
            boundary = ["refresh"] if resume == "before_capture" else ["refresh", "capture"]
            if kwargs["run_id"] == args.run_id and commands == boundary:
                raise Interrupted
            return acquire(**kwargs)

        monkeypatch.setattr(
            daemon, "acquire_lock_with_odds_capture_retry", interrupt_completed_boundary
        )
        with pytest.raises(Interrupted):
            run(args)
        assert not lock.exists()
        monkeypatch.setattr(daemon, "acquire_lock_with_odds_capture_retry", acquire)
    result = run(args)
    assert not lock.exists()
    assert not (tmp_path / "never-opened.db").exists()
    if refresh_seconds > 65:
        assert result["runtime_action"] == "LIVE_PHASE_BUDGET_EXCEEDED"
        assert commands == ["refresh"]
    else:
        assert result["runtime_action"] == "LIVE_COLLECTION_COMPLETE"
        assert commands == (["refresh"] if resume == "before_capture" else []) + [
            "refresh",
            "capture",
            "refresh",
        ]
        assert result["autonomous_live_odds_capture_status"]["inserted_rows"] == 2
        saved = json.loads(state.read_text())
        assert saved["live_capture_cursor"] == "fixture"
        from tests.operator_ui.test_live_adapters import actual_payloads, make_live

        values = actual_payloads(clock[0])
        lane = "odds" if odds_only else "full"
        values[f"{lane}_report"] = result
        values[f"{lane}_state"] = saved
        if odds_only:
            values["odds_refresh"] = result["odds_capture_refresh_report"]
        refresh_path = None
        if odds_only:
            from pathlib import Path

            refresh_path = Path(result["autopilot_output_dir"]) / "odds_capture_refresh_report.json"
        native = make_live(
            tmp_path, values, now=clock[0], odds_refresh_path=refresh_path
        ).collector(clock[0])
        assert native.data["lanes"][1 if odds_only else 0]["status"] == "RECEIPT_READY"
    checkpoint = json.loads(
        (
            state.parent
            / (
                "odds.live-phase-checkpoint.json"
                if odds_only
                else "full.live-phase-checkpoint.json"
            )
        ).read_text()
    )
    assert all(phase["status"] == "COMPLETE" for phase in checkpoint["phases"])
    assert checkpoint["maintenance"] == "DEFERRED_LIVE_FRESHNESS_PRIORITY"
    if resume == "before_capture" and refresh_seconds <= 65:
        assert len(checkpoint["rebindings"]) == 1
    if not odds_only:
        assert overlapping[0]["runtime_action"] == "DEFERRED_LOCK_HELD"
        assert overlapping[0]["final_status"] == "SKIPPED_LOCK_HELD"
        assert overlapping[0]["status"] == "SKIPPED_LOCK_HELD"
        if refresh_seconds <= 65:
            followup = daemon.run_odds_capture_once(odds_args("followup_odds_capture"))
            assert followup["runtime_action"] == "LIVE_COLLECTION_COMPLETE"
            assert (
                commands
                == (["refresh"] if resume == "before_capture" else [])
                + ["refresh", "capture", "refresh"] * 2
            )


@pytest.mark.parametrize(
    "odds_only,flags,expect_capture",
    [
        (False, (), False),
        (False, ("--execute-autonomous-odds-capture", "--allow-auto-scrape-odds"), False),
        (False, ("--enable-autonomous-odds-capture", "--allow-auto-scrape-odds"), False),
        (False, ("--enable-autonomous-odds-capture", "--execute-autonomous-odds-capture"), False),
        (
            False,
            (
                "--enable-autonomous-odds-capture",
                "--execute-autonomous-odds-capture",
                "--allow-auto-scrape-odds",
            ),
            True,
        ),
        (True, (), True),
    ],
    ids=[
        "no-flags",
        "missing-enable",
        "missing-execute",
        "missing-allow",
        "full-authorized",
        "odds-authorized",
    ],
)
def test_native_capture_requires_lane_authority(
    tmp_path, monkeypatch, odds_only, flags, expect_capture
):
    evidence = tmp_path / "evidence"
    directory = evidence / "workers/0"
    directory.mkdir(parents=True)
    (directory / "fixture.csv").write_text("fixture")
    cycle.atomic_json(
        evidence / "refresh.json",
        {
            "sidecar_metadata_coverage": {
                "races": [{"race_url": "fixture", "csv_path": str(directory / "fixture.csv")}]
            }
        },
    )
    now = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
    lock = evidence / "collector.lock"
    commands = []
    monkeypatch.setattr(daemon, "wall_clock_now", lambda: now)
    monkeypatch.setattr(cycle, "time", SimpleNamespace(monotonic=lambda: 0.0))
    monkeypatch.setattr(
        cycle,
        "bounded_current_race_index",
        lambda **kwargs: SimpleNamespace(
            source_generated_at=now.isoformat(),
            source_refresh_report_path="refresh.json",
            packet_sha256="fixture-packet",
            races=[
                {
                    "race_id": "fixture",
                    "race_url": "fixture",
                    "jump_datetime": (now + timedelta(minutes=30)).isoformat(),
                    "source_native_race_id": "fixture-native",
                    "runner_set_sha256": "fixture-runners",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        capture,
        "build_capture_plan",
        lambda *args, **kwargs: {
            "races": [{"status": "READY_TO_CAPTURE", "thedogs_source_url": "fixture"}]
        },
    )
    monkeypatch.setattr(
        capture, "existing_capture_runner_status", lambda *args, **kwargs: {"status": "NONE"}
    )

    def command(*, name, command, output_dir, **kwargs):
        assert lock.exists()
        phase = command[command.index("--collection-phase") + 1]
        commands.append(phase)
        if phase == "capture":
            assert {
                "--enable-autonomous-odds-capture",
                "--execute-autonomous-odds-capture",
                "--allow-auto-scrape-odds",
            }.issubset(command)
        cycle.atomic_json(
            output_dir / "odds_capture_refresh_report.json",
            {
                "status": "SUCCESS",
                "generated_at": now.isoformat(),
            },
        )
        cycle.atomic_json(
            output_dir / "logs" / f"{name}.stdout.txt",
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "current_race_index_publish": {"status": "PUBLISHED"},
            },
        )
        return {"name": name, "returncode": 0}

    monkeypatch.setattr(daemon, "run_command", command)
    args = daemon.parse_args(
        [
            "run-odds-capture-once" if odds_only else "run-once",
            "--live-freshness",
            "--evidence-root",
            str(evidence),
            "--state-path",
            str(evidence / "state.json"),
            "--lock-path",
            str(lock),
            "--db",
            str(tmp_path / "never-opened.db"),
            *flags,
        ]
    )
    result = (daemon.run_odds_capture_once if odds_only else daemon.run_once)(args)
    assert result["runtime_action"] == "LIVE_COLLECTION_COMPLETE"
    assert commands == (["refresh", "capture"] if expect_capture else ["refresh"])
    assert result["current_race_index_publish"]["status"] == "PUBLISHED"
    assert not lock.exists()
    assert not (tmp_path / "never-opened.db").exists()


def test_live_profile_is_opt_in_and_preserves_full_timer(tmp_path):
    from pathlib import Path

    assert not daemon.parse_args(["run-once"]).live_freshness
    for generator in (daemon.service_file_text, daemon.odds_capture_service_file_text):
        original = generator(repo_path=Path("/fixture"), timeout_seconds=300)
        candidate = generator(repo_path=Path("/fixture"), timeout_seconds=300, live_freshness=True)
        assert candidate.replace("--live-freshness ", "") == original
    for writer in (daemon.write_service_files, daemon.write_odds_capture_service_files):
        original = writer(service_dir=tmp_path / "original", repo_path=Path("/fixture"))
        candidate = writer(
            service_dir=tmp_path / "candidate", repo_path=Path("/fixture"), live_freshness=True
        )
        if writer is daemon.write_service_files:
            assert (
                Path(candidate["timer_path"]).read_bytes()
                == Path(original["timer_path"]).read_bytes()
            )
        else:
            assert "OnCalendar=*:*" in Path(candidate["timer_path"]).read_text()


@pytest.mark.parametrize("flag", ["--refresh-dry-run", "--skip-refresh"])
def test_conflicting_acquisition_flags_fail_before_any_work(tmp_path, flag):
    args = daemon.parse_args(
        ["run-once", "--live-freshness", flag, "--evidence-root", str(tmp_path / "must-not-exist")]
    )
    with pytest.raises(ValueError, match="conflicts_with_disabled_acquisition"):
        daemon.run_once(args)
    assert list(tmp_path.iterdir()) == []


def test_phase_timeout_reaps_descendants_before_releasing_control(tmp_path):
    import sys

    worker = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {worker!r}]); "
        "time.sleep(30)"
    )
    result = daemon.run_command(
        name="phase_timeout",
        command=[sys.executable, "-c", parent],
        output_dir=tmp_path,
        timeout_seconds=1,
        wait_for_descendants=True,
    )
    running = json.loads((tmp_path / "logs/phase_timeout.running.json").read_text())
    assert result["timed_out"] is True
    assert not daemon.process_group_has_running_members(running["pid"])
