import json
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from race_collection.live_phase_budget import LiveBudget
from scripts import autonomous_live_odds_capture as capture
from scripts import live_collection_cycle as cycle
from scripts import shadow_autopilot_daemon as daemon
from scripts.refresh_prejump_upcoming import bounded_discovery_days_ahead


@pytest.mark.parametrize(
    "refresh_seconds,capture_seconds,expected_verdict,proposed_budget,full_offset",
    [
        (55, 40, "LIVE_COLLECTION_COMPLETE", False, 45),
        (64.75, 40, "LIVE_COLLECTION_COMPLETE", False, 45),
        (64.75, 50, "LIVE_COLLECTION_COMPLETE", False, 45),
        (65, 40, "LIVE_PHASE_BUDGET_EXCEEDED", False, 45),
        (79.75, 50, "LIVE_COLLECTION_COMPLETE", True, 45),
        (80, 50, "LIVE_PHASE_BUDGET_EXCEEDED", True, 45),
        (79.75, 50, "LIVE_COLLECTION_COMPLETE", True, 0),
        (79.75, 50, "LIVE_COLLECTION_COMPLETE", True, 15),
        (79.75, 50, "LIVE_COLLECTION_COMPLETE", True, 94.75),
        (79.75, 50, "LIVE_COLLECTION_COMPLETE", True, 900),
        (79.75, 50, "LIVE_COLLECTION_COMPLETE", True, 930),
    ],
)
def test_all_minute_calendar_native_lanes_publish_after_overlap(
    tmp_path,
    monkeypatch,
    refresh_seconds,
    capture_seconds,
    expected_verdict,
    proposed_budget,
    full_offset,
):
    """Virtual source/consumer timing, real entrypoints, checkpoints and file locks."""
    # Review-only schedule allocation; production defaults and old failures stay intact.
    budget = (
        LiveBudget(refresh_seconds=80, completion_age_seconds=90)
        if proposed_budget
        else LiveBudget()
    )
    if not proposed_budget:
        monkeypatch.setattr(cycle, "LiveBudget", lambda: budget)
    evidence = tmp_path / "evidence"
    directory = evidence / "workers/0"
    directory.mkdir(parents=True)
    (directory / "fixture.csv").write_text("fixture")
    (evidence / "fixture-refresh.json").write_text(
        json.dumps(
            {
                "sidecar_metadata_coverage": {
                    "races": [{"race_url": "fixture", "csv_path": str(directory / "fixture.csv")}]
                }
            }
        )
    )
    lock = evidence / "runtime/collector.lock"
    origin = datetime(2026, 9, 22, 12, tzinfo=ZoneInfo("Australia/Melbourne"))
    contract_path = tmp_path / "contract.json"
    if proposed_budget:
        from race_collection.live_freshness_contract import (
            FreshnessContract,
            AttemptAllowance,
            digest,
        )

        reconciliation = {
            "schema_version": "freshness_attempt_reconciliation_v1",
            "complete": True,
            "consumed": [],
            "sources": [{"sha256": "a" * 64}],
        }
        contract = {
            "schema_version": "freshness_rehearsal_contract_v1",
            "profile": "bounded80-v1",
            "rehearsal_id": "calendar-synthetic",
            "starts_at": origin.isoformat(),
            "ends_at": (origin + timedelta(minutes=90)).isoformat(),
            "source_date": origin.date().isoformat(),
            "lock_path": str(lock),
            "evidence_root": str(evidence),
            "db_path": str(tmp_path / "never-opened.db"),
            "cleanup_seconds": 1200,
            "max_capture_attempts": 1,
            "max_logical_requests": 24000,
            "reconciliation_sha256": digest(reconciliation),
        }
        contract_path.write_text(json.dumps(contract))
        AttemptAllowance(FreshnessContract(contract)).initialize(reconciliation)
    condition = threading.Condition()
    local = threading.local()
    elapsed = [0.0]
    turn = [None]
    stopped = [False]
    active = {}
    deadlines = {}
    threads = []
    errors = []
    results = []
    refresh_elapsed = []
    capture_elapsed = []
    publications = []
    captures = []
    ignored = []
    activations = []
    full_due = [float(full_offset)]
    observed = [None]
    peaks = []

    def pause(seconds):
        lane = local.lane
        with condition:
            deadlines[lane] = elapsed[0] + seconds
            turn[0] = None
            condition.notify_all()
            condition.wait_for(lambda: stopped[0] or turn[0] == lane)
            if stopped[0]:
                raise RuntimeError("calendar_test_stopped")

    def wall_now():
        return origin + timedelta(seconds=elapsed[0])

    monkeypatch.setattr(daemon, "wall_clock_now", wall_now)
    monkeypatch.setattr(cycle, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    monkeypatch.setattr(
        daemon,
        "time",
        SimpleNamespace(
            monotonic=lambda: elapsed[0], time=lambda: wall_now().timestamp(), sleep=pause
        ),
    )

    def view(**kwargs):
        pause(0.25)
        assert observed[0] is not None
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
                    # The scoped path now consumes an exact runner-bound T10
                    # reservation. Keep its synthetic race naturally eligible
                    # during the first refresh, without bypassing window checks.
                    "jump_datetime": (
                        origin + timedelta(minutes=10 if proposed_budget else 240)
                    ).isoformat(),
                    "runners": [
                        {"box": 1, "display_name": "Fixture One", "identity": "FIXTURE ONE"},
                        {"box": 2, "display_name": "Fixture Two", "identity": "FIXTURE TWO"},
                    ],
                }
            ],
        )

    monkeypatch.setattr(cycle, "bounded_current_race_index", view)
    monkeypatch.setattr(
        capture,
        "build_capture_plan",
        lambda *args, **kwargs: {
            "races": [
                {
                    "status": "READY_TO_CAPTURE",
                    "thedogs_source_url": "fixture",
                    "race_id": "fixture",
                    "capture_window_minutes": 10,
                }
            ]
        },
    )
    monkeypatch.setattr(
        capture, "existing_capture_runner_status", lambda *args, **kwargs: {"status": "NONE"}
    )
    atomic = cycle.atomic_json

    def write(path, payload):
        if path.name in {"full.json", "odds.json"}:
            pause(0.25)
        return atomic(path, payload)

    monkeypatch.setattr(cycle, "atomic_json", write)

    def command(*, name, command, output_dir, **kwargs):
        owner = json.loads(lock.read_text())
        assert ("odds_capture" in owner["run_id"]) == (local.lane == "odds")
        phase = command[command.index("--collection-phase") + 1]
        assert "--skip-shadow-run" in command
        assert not {
            "--forward-corpus-root",
            "--enable-autonomous-result-capture",
            "--input-retention-config",
        }.intersection(command)
        source_time = wall_now()
        pause(refresh_seconds if phase == "refresh" else capture_seconds)
        assert json.loads(lock.read_text())["run_id"] == owner["run_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        if phase == "refresh":
            if observed[0] is not None:
                assert source_time >= observed[0]
                peaks.append((wall_now() - observed[0]).total_seconds())
            observed[0] = source_time
            publications.append((local.lane, elapsed[0], source_time))
            atomic(
                output_dir / "odds_capture_refresh_report.json",
                {
                    "status": "SUCCESS",
                    "generated_at": source_time.isoformat(),
                    "selected_count": 1,
                },
            )
        else:
            assert phase == "capture"
            captures.append((local.lane, elapsed[0]))
        atomic(
            output_dir / "logs" / f"{name}.stdout.txt",
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "current_race_index_publish": {"status": "PUBLISHED"},
                "autonomous_live_odds_capture_status": (
                    {"status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED", "inserted_rows": 2}
                    if phase == "capture"
                    else {}
                ),
            },
        )
        return {"name": name, "returncode": 0}

    monkeypatch.setattr(daemon, "run_command", command)

    def run(lane, run_id):
        local.lane, local.run_id = lane, run_id
        try:
            pause(0)
            arguments = [
                "run-once" if lane == "full" else "run-odds-capture-once",
                "--live-freshness",
                "--run-id",
                run_id,
                "--evidence-root",
                str(evidence),
                "--state-path",
                str(evidence / f"runtime/{lane}.json"),
                "--lock-path",
                str(lock),
                "--db",
                str(tmp_path / "never-opened.db"),
            ]
            if proposed_budget:
                arguments += [
                    "--live-freshness-profile",
                    "bounded80-v1",
                    "--live-freshness-contract",
                    str(contract_path),
                ]
            if lane == "full":
                arguments += [
                    "--odds-capture-state-path",
                    str(evidence / "runtime/odds.json"),
                    "--enable-autonomous-odds-capture",
                    "--execute-autonomous-odds-capture",
                    "--allow-auto-scrape-odds",
                ]
            args = daemon.parse_args(arguments)
            if proposed_budget:
                from race_collection.live_freshness_contract import FreshnessContract

                scope = FreshnessContract.load(contract_path)
                if (scope.session / "STOP.json").exists():
                    return
            result = (daemon.run_once if lane == "full" else daemon.run_odds_capture_once)(args)
            results.append((lane, elapsed[0], result))
            archived = Path(result["output_dir"]) / "phase-checkpoint.json"
            if archived.exists():
                refresh_elapsed.extend(
                    phase["elapsed_seconds"]
                    for phase in json.loads(archived.read_text())["phases"]
                    if phase["kind"] == "refresh"
                )
                capture_elapsed.extend(
                    phase["elapsed_seconds"]
                    for phase in json.loads(archived.read_text())["phases"]
                    if phase["kind"] == "capture"
                )
        except BaseException as error:
            errors.append(error)
        finally:
            with condition:
                if lane == "full":
                    full_due[0] = elapsed[0] + 900 + 30
                active.pop(lane, None)
                deadlines.pop(lane, None)
                turn[0] = None
                condition.notify_all()

    odds_events = [minute * 60 + 15 for minute in range(51)]
    on_active = [930] if proposed_budget and full_offset < 900 else []
    ticks = 0
    try:
        with condition:
            while odds_events or active:
                ticks += 1
                assert ticks < 5000, (elapsed, deadlines, full_due, odds_events, results)
                next_timer = (
                    min([*odds_events, *on_active, full_due[0]]) if odds_events else float("inf")
                )
                next_wake = min(deadlines.values(), default=float("inf"))
                elapsed[0] = min(next_timer, next_wake)
                if observed[0] is not None:
                    peaks.append((wall_now() - observed[0]).total_seconds())
                due = []
                if odds_events and odds_events[0] == elapsed[0]:
                    odds_events.pop(0)
                    due.append("odds")
                if full_due[0] == elapsed[0]:
                    full_due[0] = float("inf")
                    due.append("full")
                if on_active and on_active[0] == elapsed[0]:
                    on_active.pop(0)
                    if "full" not in due:
                        due.append("full")
                for lane in due:
                    if lane in active:
                        ignored.append((lane, elapsed[0]))
                        continue
                    run_id = f"calendar_{len(activations)}_{'odds_capture' if lane == 'odds' else 'full'}"
                    activations.append((lane, elapsed[0]))
                    active[lane] = run_id
                    thread = threading.Thread(target=run, args=(lane, run_id), daemon=True)
                    threads.append(thread)
                    thread.start()
                    assert condition.wait_for(lambda: lane in deadlines or errors, timeout=10)
                ready = sorted(
                    lane for lane, deadline in deadlines.items() if deadline <= elapsed[0]
                )
                for lane in ready:
                    turn[0] = lane
                    condition.notify_all()
                    assert condition.wait_for(lambda: turn[0] is None, timeout=10)
                assert not errors, repr(errors)
    finally:
        with condition:
            stopped[0] = True
            condition.notify_all()
        for thread in threads:
            thread.join(timeout=10)
        assert all(not thread.is_alive() for thread in threads)

    completed = [
        (lane, result)
        for lane, _, result in results
        if result["runtime_action"] == "LIVE_COLLECTION_COMPLETE"
    ]
    summary = {
        "calendar": "every_minute_plus_15_seconds_accuracy",
        "budget_next_trigger_seconds": LiveBudget().next_trigger_seconds,
        "refresh_seconds": refresh_seconds,
        "capture_seconds": capture_seconds,
        "named_bounded80_profile": proposed_budget,
        "initial_full_timer_offset_seconds": full_offset,
        "refresh_allocation_seconds": budget.refresh_seconds,
        "expected_verdict": expected_verdict,
        "maximum_native_refresh_phase_seconds": max(refresh_elapsed),
        "maximum_native_capture_phase_seconds": max(capture_elapsed, default=None),
        "verdict_counts": {
            lane: dict(
                Counter(result["runtime_action"] for owner, _, result in results if owner == lane)
            )
            for lane in ("full", "odds")
        },
        "results": [(lane, ended, result["runtime_action"]) for lane, ended, result in results],
        "peak_source_age": max(peaks),
        "completed_full_cycles": sum(lane == "full" for lane, _ in completed),
        "completed_odds_cycles": sum(lane == "odds" for lane, _ in completed),
        "full_publications": sum(lane == "full" for lane, _, _ in publications),
        "odds_publications": sum(lane == "odds" for lane, _, _ in publications),
        "ignored_active_service_triggers": len(ignored),
    }
    print(
        "CALENDAR_EVIDENCE "
        + json.dumps({key: value for key, value in summary.items() if key != "results"})
    )
    assert LiveBudget().next_trigger_seconds == 75, summary
    assert LiveBudget().refresh_seconds == 65
    assert max(refresh_elapsed) == refresh_seconds + 0.25, summary
    assert not lock.exists()
    assert not (tmp_path / "never-opened.db").exists()
    for counts in summary["verdict_counts"].values():
        if proposed_budget and expected_verdict == "LIVE_PHASE_BUDGET_EXCEEDED":
            continue
        assert counts.get(expected_verdict, 0) > 0, summary
        assert all(
            verdict == expected_verdict or verdict.startswith("DEFERRED_") for verdict in counts
        ), summary
    if expected_verdict == "LIVE_PHASE_BUDGET_EXCEEDED":
        assert not completed, summary
        assert any(result["runtime_action"] == expected_verdict for _, _, result in results)
        assert not captures
        assert max(refresh_elapsed) == budget.refresh_seconds + 0.25, summary
        return
    assert max(refresh_elapsed) <= budget.refresh_seconds, summary
    assert max(capture_elapsed) == capture_seconds, summary
    assert max(capture_elapsed) <= LiveBudget().work_seconds == 50, summary
    assert sum(lane == "full" for lane, _ in completed) >= 3, summary
    assert sum(lane == "odds" for lane, _ in completed) >= 6, summary
    assert {lane for lane, _, _ in publications} == {"full", "odds"}
    if proposed_budget:
        assert len(captures) == 1
    else:
        assert {lane for lane, _ in captures} == {"full", "odds"}
    assert any(result["timing"]["lock_wait_seconds"] > 0 for _, result in completed)
    assert any(result["runtime_action"].startswith("DEFERRED_") for _, _, result in results)
    assert ignored
    assert all(result["timing"]["overhead_seconds"] > 0 for _, result in completed)
    assert max(peaks) <= 270, {"peak": max(peaks), "publications": publications}


def test_legacy_135_second_gap_cannot_support_65_second_refresh_yield():
    legacy = LiveBudget(next_trigger_seconds=135)
    observed = datetime(2026, 9, 22, 12, tzinfo=ZoneInfo("Australia/Melbourne"))
    assert legacy.safe_to_yield(observed, observed + timedelta(seconds=60))
    assert not legacy.safe_to_yield(observed, observed + timedelta(seconds=65))
    successor_publication_age = 65 + 135 + 65 + 10
    assert successor_publication_age == 275
    assert successor_publication_age > legacy.publication_age


def test_review_allocation_accounts_for_yield_work_and_handoff():
    budget = LiveBudget(refresh_seconds=80, completion_age_seconds=90)
    observed = datetime.fromisoformat("2026-09-22T12:00:00+10:00")
    # Reserve 15s below 270 after both owners' overhead and a timer opportunity.
    assert 2 * budget.refresh_seconds + 75 + 2 * 10 + 15 == 270
    assert budget.safe_to_yield(observed, observed + timedelta(seconds=90))
    assert not budget.safe_to_yield(observed, observed + timedelta(seconds=90.001))
    assert 90 + 75 + 80 + 10 == 255
    # Work can consume the reserve, but must fund a successor and lock handoff.
    assert budget.admit_work(observed, observed + timedelta(seconds=115))
    assert not budget.admit_work(observed, observed + timedelta(seconds=115.001))
    assert 115 + 50 + 80 + 10 + 10 + 5 == 270
    # A maximal refresh/capture pair cannot simply yield or admit another capture.
    after_capture = observed + timedelta(seconds=80 + 50)
    assert not budget.safe_to_yield(observed, after_capture)
    assert not budget.admit_work(observed, after_capture)
    legacy = LiveBudget(refresh_seconds=80, completion_age_seconds=90, next_trigger_seconds=135)
    assert not legacy.safe_to_yield(observed, observed + timedelta(seconds=80))
    assert LiveBudget().refresh_seconds == 65  # Proposal does not change runtime policy.


@pytest.mark.parametrize(
    "minutes,before,boundary", [(160, "21:19:59", "21:20:00"), (60, "22:59:59", "23:00:00")]
)
def test_daytime_scope_ends_before_discovery_reaches_tomorrow(minutes, before, boundary):
    zone = ZoneInfo("Australia/Melbourne")
    for time_text, expected in ((before, 0), (boundary, 1)):
        now = datetime.fromisoformat(f"2026-09-22T{time_text}").replace(tzinfo=zone)
        assert (
            bounded_discovery_days_ahead(
                now=now, wall_now=now, max_minutes=minutes, requested_days_ahead=1
            )
            == expected
        )
        assert (
            bounded_discovery_days_ahead(
                now=now,
                wall_now=now.astimezone(ZoneInfo("UTC")),
                max_minutes=minutes,
                requested_days_ahead=1,
            )
            == 1
        )
