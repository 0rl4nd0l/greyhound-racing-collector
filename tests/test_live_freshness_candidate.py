import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from race_collection.live_phase_budget import LiveBudget
from scripts import shadow_autopilot_daemon as daemon


def test_named_profile_preserves_previous_budget():
    assert LiveBudget().refresh_seconds == 65
    budget = LiveBudget.for_profile("bounded80-v1")
    now = datetime(2026, 9, 23, 1, tzinfo=timezone.utc)
    assert budget.refresh_seconds == 80
    assert budget.safe_to_yield(now, now + timedelta(seconds=90))
    assert not budget.safe_to_yield(now, now + timedelta(seconds=91))
    assert budget.admit_work(now, now + timedelta(seconds=115))
    assert not budget.admit_work(now, now + timedelta(seconds=116))


@pytest.mark.parametrize("odds_only", [False, True])
def test_generated_profile_reaches_native_entrypoint(tmp_path, odds_only):
    writer = daemon.write_odds_capture_service_files if odds_only else daemon.write_service_files
    result = writer(
        service_dir=tmp_path,
        repo_path=Path("/candidate"),
        live_freshness=True,
        live_freshness_profile="bounded80-v1",
        live_freshness_contract=tmp_path / "contract.json",
    )
    unit = Path(result["service_path"]).read_text()
    command = next(
        line.removeprefix("ExecStart=")
        for line in unit.splitlines()
        if line.startswith("ExecStart=")
    )
    args = daemon.parse_args(shlex.split(command)[2:])
    assert args.live_freshness_profile == "bounded80-v1"
    assert args.live_freshness_contract == tmp_path / "contract.json"
    assert "--enable-autonomous-result-capture" not in command
    assert "--forward-corpus-root" not in command
    assert "--live-freshness-profile" not in daemon.service_file_text(
        repo_path=Path("/candidate"), timeout_seconds=600
    )


def contract_value(tmp_path):
    return dict(
        schema_version="freshness_rehearsal_contract_v1",
        profile="bounded80-v1",
        rehearsal_id="synthetic",
        starts_at="2026-09-23T01:00:00+00:00",
        ends_at="2026-09-23T02:30:00+00:00",
        source_date="2026-09-23",
        lock_path=str(tmp_path / "collector.lock"),
        evidence_root=str(tmp_path / "evidence"),
        db_path=str(tmp_path / "synthetic.sqlite"),
        cleanup_seconds=1200,
        max_capture_attempts=1,
        max_logical_requests=24000,
        reconciliation_sha256="a" * 64,
    )


def test_scope_requires_margin_and_one_date(tmp_path):
    from race_collection.live_freshness_contract import FreshnessContract

    value = contract_value(tmp_path)
    scope = FreshnessContract(value)
    scope.admit(datetime.fromisoformat(value["starts_at"]), seconds=90)
    with pytest.raises(ValueError, match="scope"):
        scope.admit(datetime.fromisoformat(value["ends_at"]), seconds=1)
    value["starts_at"] = "2026-09-23T10:00:00+00:00"
    value["ends_at"] = "2026-09-23T11:30:00+00:00"
    with pytest.raises(ValueError, match="one_date"):
        FreshnessContract(value)


def test_consumed_allowance_survives_failed_no_row_attempt_and_restart(tmp_path):
    from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract
    from race_collection.live_freshness_contract import digest

    reconciliation = {
        "schema_version": "freshness_attempt_reconciliation_v1",
        "complete": True,
        "consumed": [],
        "sources": [{"sha256": "b" * 64}],
    }
    value = contract_value(tmp_path)
    value["reconciliation_sha256"] = digest(reconciliation)
    scope = FreshnessContract(value)
    allowance = AttemptAllowance(scope)
    allowance.initialize(reconciliation)
    item = {
        "race_id": "synthetic-race",
        "capture_window_minutes": 10,
        "packet_sha256": "c" * 64,
        "race_identity": {"jump_datetime": "2026-09-23T01:10:00+00:00"},
    }
    claim = allowance.reserve(item, now=datetime.fromisoformat(scope.value["starts_at"]))
    allowance.start_fetch(claim, item, now=datetime.fromisoformat(scope.value["starts_at"]))
    allowance.finish(claim, {"status": "FAIL", "inserted_rows": 0})
    restarted = AttemptAllowance(scope)
    assert restarted.available() is False
    with pytest.raises(ValueError, match="consumed"):
        restarted.reserve(
            {**item, "race_id": "substitute"}, now=datetime.fromisoformat(scope.value["starts_at"])
        )
    with pytest.raises((ValueError, FileExistsError)):
        restarted.start_fetch(claim, item, now=datetime.fromisoformat(scope.value["starts_at"]))


def test_r3_rejects_old_unit_binding_after_calendar_change(tmp_path):
    from dataclasses import replace
    from tests.operator_ui.test_live_adapters import actual_payloads, make_live

    now = datetime(2026, 9, 23, 1, tzinfo=timezone.utc)
    adapter = make_live(
        tmp_path, actual_payloads(now, include_models=False), now=now, include_models=False
    )
    assert adapter.system(now).data["components"][0]["status"] == "HEALTHY"
    adapter._units = replace(
        adapter._units, odds_timer=daemon.odds_capture_timer_file_text(live_freshness=True).encode()
    )
    assert adapter.system(now).data["components"][0]["status"] == "DIVERGENT"


def test_older_publication_cannot_replace_newer_generation(tmp_path):
    import json
    from race_collection import synchronous_manual_capture as publisher
    from tests.race_collection.test_synchronous_manual_capture import _runner_coverage
    from scripts import shadow_autopilot_v1 as autopilot

    root = tmp_path / "evidence"
    state = root / "runtime/odds.json"
    now = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"

    def publish(name, stamp):
        output = root / name
        output.mkdir(parents=True)
        source = output / "refresh.json"
        coverage = _runner_coverage(output, url, stamp)
        source.write_text(
            json.dumps(
                {
                    "status": "SUCCESS",
                    "generated_at": stamp.isoformat(),
                    "sidecar_metadata_coverage": coverage,
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
                            "source_native_race_id": "15900",
                            "race_time": "13:00",
                            "race_url": url,
                            "venue": "GUNN",
                        }
                    ],
                }
            )
        )
        return autopilot.publish_current_race_index_after_refresh(
            state_path=state,
            evidence_root=root,
            output_dir=output,
            run_id=name,
            source_refresh_report_path=source,
            enforce_monotonic=True,
        )

    assert publish("newer", now)["status"] == "PUBLISHED"
    before = publisher.current_race_index_path(state).read_bytes()
    assert publish("late_old", now - timedelta(seconds=1))["status"] == "REJECTED"
    assert publisher.current_race_index_path(state).read_bytes() == before
    view = publisher.bounded_current_race_index(
        current_time=now + timedelta(seconds=255),
        timeout_seconds=2,
        index_path=publisher.current_race_index_path(state),
        evidence_root=root,
        max_age_seconds=300,
        return_verified_view=True,
    )
    assert view.source_generated_at == now.isoformat()


@pytest.mark.parametrize(
    "age,owner,expected",
    [
        (80, "full-1", "WAITING_FOR_PEER"),
        (271, "full-1", "CAPTURE_WINDOW_CLOSED"),
        (80, "other", "CAPTURE_WINDOW_CLOSED"),
    ],
)
def test_native_r3_peer_handoff_requires_fresh_index_and_matching_owner(
    tmp_path, monkeypatch, age, owner, expected
):
    from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
    from src.operator_ui import live_adapters
    from tests.operator_ui.test_live_adapters import actual_payloads, make_live

    now = datetime(2026, 9, 23, 1, tzinfo=timezone.utc)
    values = actual_payloads(now, include_models=False)
    values["odds_report"].update(
        status="SKIPPED_LOCK_HELD",
        final_status="SKIPPED_LOCK_HELD",
        runtime_action="DEFERRED_LOCK_HELD",
        live_freshness_profile="bounded80-v1",
        deferred_lock_owner={"run_id": owner, "pid": 123},
        autopilot_output_dir=None,
    )
    values["odds_report"].pop("odds_capture_refresh_report")
    view = VerifiedCurrentRaceIndex(
        "collector_current_race_index_v2",
        "full-1",
        (now - timedelta(seconds=age)).isoformat(),
        "1" * 64,
        b"packet",
        (),
        "refresh.json",
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )
    monkeypatch.setattr(live_adapters, "bounded_current_race_index", lambda **kwargs: view)
    kwargs = dict(
        repo_path=Path("/srv/app"),
        timeout_seconds=600,
        live_freshness=True,
        live_freshness_profile="bounded80-v1",
        live_freshness_contract=tmp_path / "contract.json",
    )
    adapter = make_live(
        tmp_path,
        values,
        now=now,
        include_models=False,
        upcoming_races=live_adapters.UpcomingRaceSource(tmp_path / "index.json", tmp_path),
        unit_overrides={
            "full_service": daemon.service_file_text(**kwargs).encode(),
            "odds_service": daemon.odds_capture_service_file_text(**kwargs).encode(),
            "odds_timer": daemon.odds_capture_timer_file_text(live_freshness=True).encode(),
        },
    )
    result = adapter.collector(now)
    assert result.data["lanes"][1]["status"] == expected
    assert (result.evidence.status == "AVAILABLE/FRESH") == (expected == "WAITING_FOR_PEER")


@pytest.mark.parametrize("evidence_state", ["complete", "empty", "odds_report_only"])
def test_raw_phase_report_serialization_matches_native_r3(tmp_path, evidence_state):
    import hashlib
    from race_collection.live_phase_checkpoint import atomic_json
    from race_collection.freshness_rehearsal import native_observation
    from tests.operator_ui.test_live_adapters import actual_payloads
    from src.operator_ui.live_adapters import InstalledUnits

    now = datetime(2026, 9, 23, 1, tzinfo=timezone.utc)
    values = actual_payloads(now, include_models=False)
    for lane in ("full", "odds"):
        values[lane + "_state"]["last_output_dir" if lane == "full" else "output_dir"] = str(
            tmp_path / lane
        )
        values[lane + "_report"]["output_dir"] = str(tmp_path / lane)
    for key in ("odds_report", "odds_state"):
        values[key]["autopilot_output_dir"] = str(tmp_path / "autopilot")
    paths = {}
    for key in ("full_report", "full_state", "odds_report", "odds_state", "odds_refresh"):
        path = (
            tmp_path / "autopilot/odds_capture_refresh_report.json"
            if key == "odds_refresh"
            else tmp_path / (key + ".json")
        )
        if evidence_state == "complete" or (
            evidence_state == "odds_report_only" and key == "odds_report"
        ):
            atomic_json(path, values[key])
        paths[key] = path
    raw = dict(
        full_service=daemon.service_file_text(repo_path=tmp_path, timeout_seconds=600).encode(),
        odds_service=daemon.odds_capture_service_file_text(
            repo_path=tmp_path, timeout_seconds=600
        ).encode(),
        full_timer=daemon.timer_file_text().encode(),
        odds_timer=daemon.odds_capture_timer_file_text(live_freshness=True).encode(),
    )
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
    units = InstalledUnits(
        **raw,
        **{key + "_sha256": val for key, val in hashes.items()},
        observed_at=now,
        working_directory=str(tmp_path),
        full_unit_name="shadow-autopilot.service",
        odds_unit_name="shadow-autopilot-odds-capture.service",
        full_active_state="inactive",
        full_sub_state="dead",
        full_exec_main_pid=0,
        odds_active_state="inactive",
        odds_sub_state="dead",
        odds_exec_main_pid=0,
    )
    result = native_observation(
        now=now,
        paths=paths,
        units=units,
        evidence_root=tmp_path,
        index_path=tmp_path / "absent-index.json",
        output=tmp_path / "measurement",
        authority={
            "commit": "b" * 40,
            "tree": "c" * 40,
            "source_root": str(tmp_path),
            "unit_sha256": hashes,
        },
    )
    assert result["authority_status"] == "AVAILABLE/FRESH"
    assert result["collector_status"] == (
        "AVAILABLE/FRESH" if evidence_state == "complete" else "UNAVAILABLE/DATA_MISSING"
    )
    assert result["index_status"] == "UNAVAILABLE/DATA_MISSING"


def test_missing_publication_or_clock_gap_is_not_omitted():
    from race_collection.freshness_rehearsal import assess_interval

    previous = {
        "monotonic_start": 0,
        "read_start": "2026-09-23T01:00:00+00:00",
        "source_at": "2026-09-23T00:55:32+00:00",
        "source_age_seconds": 268,
        "index_status": "AVAILABLE/FRESH",
        "packet_sha256": "old",
    }
    current = {
        "monotonic_end": 2,
        "read_end": "2026-09-23T01:00:02+00:00",
        "index_status": "AVAILABLE/FRESH",
        "packet_sha256": "new",
    }
    event = {"source_generated_at": "2026-09-23T00:59:00+00:00", "packet_sha256": "new"}
    assert assess_interval(previous, current, [event]) == 270
    with pytest.raises(ValueError, match="chain"):
        assess_interval(previous, current, [])
    with pytest.raises(ValueError, match="gap"):
        assess_interval(previous, {**current, "monotonic_end": 6}, [event])


def test_reconciliation_preserves_no_row_and_ambiguous_started_windows(tmp_path):
    import json
    import sqlite3
    from race_collection.freshness_attempt_reconciliation import reconcile, DOMAINS

    root = tmp_path / "operational"
    root.mkdir()
    for name in ("claims", "attempts", "requests"):
        (root / name).mkdir()
    race_id = "Race 1 - SYNTHETIC - 2026-09-23"
    (root / "autonomous_live_odds_capture_attempts.progress.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "autonomous_live_odds_capture_attempt_v1",
                "race_id": race_id,
                "capture_window_minutes": 10,
                "status": "FETCH_IN_PROGRESS",
                "inserted_rows": 0,
            }
        )
        + "\n"
    )
    (root / "phase-checkpoint.json").write_text(
        json.dumps({"phases": [{"kind": "capture", "inputs": {"race_id": race_id}}]})
    )
    lock = tmp_path / "lock"
    lock.write_text(json.dumps({"run_id": "owner", "pid": 1}))
    db = tmp_path / "operational.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE live_odds (race_id TEXT, capture_mode TEXT)")
    roots = {name: [str(root)] for name in DOMAINS if name != "live_odds"}
    result = reconcile(
        roots=roots, db_path=db, source_date="2026-09-23", lock_path=lock, owner_run_id="owner"
    )
    assert result["complete"] is True
    assert {row["capture_window_minutes"] for row in result["consumed"]} == {60, 30, 10, 2}
    (root / "autonomous_live_odds_capture_attempts.progress.jsonl").write_text("{")
    with pytest.raises(ValueError):
        reconcile(
            roots=roots, db_path=db, source_date="2026-09-23", lock_path=lock, owner_run_id="owner"
        )


@pytest.mark.parametrize(
    "partial_pause_failure,reserve_before_failure", [(False, False), (True, False), (False, True)]
)
def test_rehearsal_failure_restores_exact_pair_and_does_not_reset_consumption(
    tmp_path, monkeypatch, partial_pause_failure, reserve_before_failure
):
    import json
    import hashlib
    from scripts import run_freshness_rehearsal as run
    from race_collection import freshness_attempt_reconciliation as reconciliation
    from race_collection.live_freshness_contract import create_once, digest
    from scripts.prepare_freshness_rehearsal import UNITS

    installed = tmp_path / "installed"
    installed.mkdir()
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "units").mkdir()
    original = {}
    for name in (*UNITS, "greyhound-operator-ui-r3.service"):
        original[name] = ("original " + name).encode()
        (installed / name).write_bytes(original[name])
        if name in UNITS:
            (output / "units" / name).write_bytes(("candidate " + name).encode())
    stamp = datetime(2026, 9, 23, 1, tzinfo=timezone.utc)
    clock = [stamp - timedelta(seconds=2)]
    monkeypatch.setattr(run, "now", lambda: clock[0])
    monkeypatch.setattr(
        run.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + timedelta(seconds=seconds)),
    )
    restore_original = run.restore
    monkeypatch.setattr(
        run,
        "restore",
        lambda output, plan, control: restore_original(
            output,
            plan,
            control,
            clock=lambda: (clock[0] - stamp).total_seconds(),
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + timedelta(seconds=seconds)),
        ),
    )

    class Control:
        active = {name: True for name in run.TIMERS}
        fail_pause = partial_pause_failure

        def command(self, operation, *names):
            if self.fail_pause and operation == "stop" and names == (run.TIMERS[1],):
                self.fail_pause = False
                raise RuntimeError("injected_partial_pause_failure")
            if operation == "is-enabled":
                return "enabled\n"
            if operation in {"start", "stop"}:
                for name in names:
                    self.active[name] = operation == "start"
            return ""

        def show(self, name):
            return {
                "DropInPaths": "",
                "MainPID": "123" if name == "greyhound-operator-ui-r3.service" else "0",
                "ActiveState": "active" if self.active.get(name) else "inactive",
            }

        def idle(self):
            return True

    control = Control()
    monkeypatch.setattr(run, "SystemdControl", lambda: control)
    monkeypatch.setattr(run, "verify_source_package", lambda *args: {})
    monkeypatch.setattr(run, "verify_runtime", lambda *args: {"prefix": __import__("sys").prefix})
    accounting = {
        "schema_version": "freshness_attempt_reconciliation_v1",
        "complete": True,
        "consumed": [],
        "sources": [{"sha256": "b" * 64}],
        "source_date": "2026-09-23",
    }
    monkeypatch.setattr(reconciliation, "reconcile", lambda **kwargs: accounting)

    def fail(*args):
        if reserve_before_failure:
            from race_collection.live_freshness_contract import AttemptAllowance

            AttemptAllowance(args[-1]).reserve(
                {
                    "race_id": "synthetic",
                    "capture_window_minutes": 10,
                    "race_identity": {
                        "jump_datetime": (clock[0] + timedelta(minutes=10)).isoformat()
                    },
                },
                now=clock[0],
            )
        raise RuntimeError("injected_observation_failure")

    monkeypatch.setattr(run, "observe", fail)
    python = Path("/usr/bin/python3")
    value = contract_value(tmp_path)
    plan = {
        **value,
        "installed_dir": str(installed),
        "source_root": str(output),
        "source_identity_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "python": str(python),
        "python_sha256": hashlib.sha256(python.resolve().read_bytes()).hexdigest(),
        "admission_starts_at": (stamp - timedelta(minutes=30)).isoformat(),
        "unit_sha256": {
            name: hashlib.sha256((output / "units" / name).read_bytes()).hexdigest()
            for name in UNITS
        },
        "baseline_unit_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in original.items()
        },
        "reconciliation_roots": {},
    }
    create_once(output / "plan.json", plan)
    with pytest.raises(RuntimeError, match="injected"):
        run.execute(output / "plan.json", digest(plan), "offline-test")
    assert all((installed / name).read_bytes() == raw for name, raw in original.items())
    assert all(control.active.values())
    assert (output / "restored.json").exists()
    assert (output / "started.json").exists()
    if reserve_before_failure:
        assert clock[0] > stamp + timedelta(minutes=8)
    with pytest.raises(FileExistsError):
        run.execute(output / "plan.json", digest(plan), "offline-retry")


def test_shared_request_cap_stops_before_next_source_call(tmp_path, monkeypatch):
    import requests
    from race_collection.live_freshness_contract import FreshnessContract, install_request_guard

    scope = FreshnessContract({**contract_value(tmp_path), "max_logical_requests": 2})
    monkeypatch.setattr(scope, "admit", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(
        requests.Session, "request", lambda self, *args, **kwargs: calls.append(args)
    )
    restore = install_request_guard(scope)
    try:
        requests.Session().get("https://www.thedogs.com.au/fabricated/one")
        requests.Session().get("https://www.thedogs.com.au/fabricated/two")
        with pytest.raises(ValueError, match="cap"):
            requests.Session().get("https://www.thedogs.com.au/fabricated/three")
    finally:
        restore()
    assert len(calls) == 2
    assert (scope.session / "STOP.json").exists()


def test_native_fetch_rejects_changed_reserved_window_before_network(tmp_path, monkeypatch):
    from race_collection.live_freshness_contract import (
        FreshnessContract,
        AttemptAllowance,
        digest,
        create_once,
    )
    from scripts import autonomous_live_odds_capture as capture

    reconciliation = {
        "schema_version": "freshness_attempt_reconciliation_v1",
        "complete": True,
        "consumed": [],
        "sources": [{"sha256": "b" * 64}],
    }
    value = {**contract_value(tmp_path), "reconciliation_sha256": digest(reconciliation)}
    scope = FreshnessContract(value)
    allowance = AttemptAllowance(scope)
    allowance.initialize(reconciliation)
    stamp = datetime.fromisoformat(value["starts_at"])
    item = {
        "race_id": "synthetic",
        "capture_window_minutes": 10,
        "status": "READY_TO_CAPTURE",
        "jump_datetime": "2026-09-23T01:10:00+00:00",
        "race_identity": {"jump_datetime": "2026-09-23T01:10:00+00:00"},
    }
    claim = allowance.reserve(item, now=stamp)
    contract = tmp_path / "contract.json"
    create_once(contract, value)
    monkeypatch.setattr(
        capture,
        "refresh_plan_item_for_time",
        lambda item, now: {**item, "capture_window_minutes": 2},
    )
    monkeypatch.setattr(
        capture, "existing_capture_runner_status", lambda *a, **k: {"status": "NONE"}
    )
    monkeypatch.setattr(
        capture,
        "fetch_odds_for_target_race_with_timeout",
        lambda *a, **k: pytest.fail("network must not start"),
    )
    with pytest.raises(ValueError, match="identity_changed"):
        capture.execute_capture_plan(
            {"races": [item]},
            db_path=tmp_path / "synthetic.sqlite",
            current_time=stamp,
            execute=True,
            allow_auto_scrape_odds=True,
            current_time_provider=lambda: stamp,
            live_freshness_contract=contract,
            live_capture_reservation=claim,
        )
    assert not allowance.available()


def test_profile_without_opt_in_rejected_before_work():
    with pytest.raises(ValueError, match="opt_in"):
        daemon.main(
            [
                "run-once",
                "--live-freshness-profile",
                "bounded80-v1",
                "--live-freshness-contract",
                "/absent.json",
            ]
        )


def test_external_exit_time_is_part_of_overhead_allocation():
    from race_collection.freshness_rehearsal import completed_service_overhead

    status = {
        "ActiveState": "inactive",
        "ExecMainStartTimestampMonotonic": "100000000",
        "ExecMainExitTimestampMonotonic": "190000000",
    }
    report = {
        "timing": {"process_started_monotonic": 100.1, "phase_seconds": 80, "lock_wait_seconds": 0}
    }
    assert completed_service_overhead(status, report) == 10
    status["ExecMainExitTimestampMonotonic"] = "190001000"
    with pytest.raises(ValueError, match="overhead"):
        completed_service_overhead(status, report)


def test_partial_reservation_remains_consumed(tmp_path):
    from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest

    reconciliation = {
        "schema_version": "freshness_attempt_reconciliation_v1",
        "complete": True,
        "consumed": [],
        "sources": [{"sha256": "b" * 64}],
    }
    scope = FreshnessContract(
        {**contract_value(tmp_path), "reconciliation_sha256": digest(reconciliation)}
    )
    allowance = AttemptAllowance(scope)
    allowance.initialize(reconciliation)
    allowance.claim.write_bytes(b"{")
    assert not AttemptAllowance(scope).available()


def test_pinned_source_rejects_changed_file(tmp_path):
    import hashlib
    from race_collection.live_freshness_contract import create_once, digest, verify_source_package

    source = tmp_path / "worker.py"
    source.write_text("original")
    identity = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "files": {"worker.py": hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    create_once(tmp_path / "SOURCE_IDENTITY.json", identity)
    assert verify_source_package(tmp_path, digest(identity)) == identity
    source.write_text("changed")
    with pytest.raises(ValueError, match="file_changed"):
        verify_source_package(tmp_path, digest(identity))


def test_window_denominators_separate_missed_excluded_and_attempted(tmp_path):
    from scripts.run_freshness_rehearsal import window_accounting
    from race_collection.live_freshness_contract import create_once

    rows = [
        {
            "race_id": race,
            "capture_window_minutes": 10,
            "jump_datetime": "2026-09-23T01:10:00+00:00",
            "status": "READY_TO_CAPTURE",
        }
        for race in ("attempted", "excluded", "missed")
    ]
    claim = tmp_path / "claim.json"
    create_once(claim, {"item": rows[0]})
    result = window_accounting(
        rows, [rows[1]], claim, datetime(2026, 9, 23, 2, tzinfo=timezone.utc)
    )
    assert len(result["eligible_observed_windows"]) == 3
    assert result["attempted_windows"] == [{"race_id": "attempted", "capture_window_minutes": 10}]
    assert [row["race_id"] for row in result["missed_observed_windows"]] == ["missed"]
    assert result["outside_observed_refresh_coverage"] == "UNASSESSED"


def test_capture_navigation_count_uses_native_fetch_without_response_storage(tmp_path, monkeypatch):
    import sys
    import json
    from types import SimpleNamespace
    import odds_auto_integrator as integration

    calls = []

    class Driver:
        def get(self, url):
            calls.append(url)

        def find_elements(self, *args):
            return []

    class Integrator:
        def __init__(self, *args, **kwargs):
            self.driver = Driver()
            self.greyhound_url = "https://www.thedogs.com.au/fabricated/"

        def setup_driver(self):
            return True

        def close_driver(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "sportsbet_odds_integrator",
        SimpleNamespace(SportsbetOddsIntegrator=Integrator),
    )
    monkeypatch.setattr(integration, "_resolve_target_race_from_meeting", lambda *args: None)
    monkeypatch.setattr(integration.time, "sleep", lambda seconds: None)
    path = tmp_path / "counts.json"
    integration.fetch_odds_for_target_race(
        str(tmp_path / "unused.sqlite"),
        "synthetic",
        1,
        "2026-09-23",
        True,
        request_metrics_path=path,
    )
    assert len(calls) == 1
    assert json.loads(path.read_bytes()) == {
        "browser_navigation_attempts": 1,
        "subresource_requests": "UNMEASURED",
    }


def test_reserved_native_plan_reaches_one_validated_append(tmp_path, monkeypatch):
    import hashlib
    from tests.test_autonomous_live_odds_capture import _write_capture_input, _place_odds_rows
    from scripts import autonomous_live_odds_capture as capture
    from race_collection.live_freshness_contract import (
        AttemptAllowance,
        FreshnessContract,
        create_once,
        digest,
    )

    stamp = datetime.fromisoformat("2026-06-10T14:40:00+10:00")
    directory = tmp_path / "input"
    csv = _write_capture_input(directory)
    import json
    from race_collection.synchronous_manual_capture import runner_set_sha256

    sidecar = capture.sidecar_path_for(csv)
    metadata = json.loads(sidecar.read_bytes())
    metadata["prejump_shadow_metadata"]["source_native_race_id"] = "synthetic-1"
    sidecar.write_text(json.dumps(metadata))
    plan = capture.build_capture_plan([directory], current_time=stamp)
    item = plan["races"][0]
    accounting = {
        "schema_version": "freshness_attempt_reconciliation_v1",
        "complete": True,
        "consumed": [],
        "sources": [{"sha256": "b" * 64}],
    }
    value = {
        **contract_value(tmp_path),
        "starts_at": stamp.isoformat(),
        "ends_at": (stamp + timedelta(minutes=90)).isoformat(),
        "source_date": "2026-06-10",
        "reconciliation_sha256": digest(accounting),
    }
    scope = FreshnessContract(value)
    allowance = AttemptAllowance(scope)
    allowance.initialize(accounting)
    inputs = {
        **item,
        "race_identity": {
            "race_id": item["race_id"],
            "jump_datetime": item["jump_datetime"],
            "race_url": item["thedogs_source_url"],
            "source_native_race_id": "synthetic-1",
        },
        "capture_runner_set_sha256": runner_set_sha256(item["expected_runners"]),
        "input_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (csv, capture.sidecar_path_for(csv))
        },
    }
    claim = allowance.reserve(inputs, now=stamp)
    contract = tmp_path / "contract.json"
    create_once(contract, value)
    calls = []

    def fetch(*args, **kwargs):
        calls.append(kwargs)
        win = [
            {
                "dog_name": name,
                "box_number": box,
                "odds_decimal": price,
                "sportsbet_box_source": "runner_text",
            }
            for name, box, price in [("Alpha", 1, 2.4), ("Bravo", 2, 3.5)]
        ]
        return {
            "success": True,
            "win_count": 2,
            "place_count": 2,
            "odds_data": win,
            "race_info": {
                "venue_url": "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/wentworth-park/race-1",
                "race_number": 1,
                "odds_data_place": _place_odds_rows(win),
            },
        }

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fetch)
    monkeypatch.setattr(
        capture,
        "append_validated_capture",
        lambda **kw: {"status": "SUCCESS", "inserted_rows": 4, "warnings": []},
    )
    result = capture.execute_capture_plan(
        plan,
        db_path=tmp_path / "synthetic.sqlite",
        current_time=stamp,
        current_time_provider=lambda: stamp,
        execute=True,
        allow_auto_scrape_odds=True,
        live_freshness_contract=contract,
        live_capture_reservation=claim,
    )
    assert result["attempts"][0]["status"] == "APPENDED"
    assert len(calls) == 1
    assert calls[0]["request_metrics_path"] == scope.session / "capture-requests.json"
    assert claim.with_suffix(".fetch.json").exists()
    assert not allowance.available()


def test_profile_lock_does_not_erase_dead_or_unreadable_owner(tmp_path):
    path = tmp_path / "collector.lock"
    for raw in (b'{"pid":-1,"run_id":"old"}', b"{"):
        path.write_bytes(raw)
        with pytest.raises(daemon.LockBusy):
            daemon.acquire_lock_with_odds_capture_retry(
                lock_path=path,
                run_id="candidate",
                output_dir=tmp_path,
                stale_after_seconds=1,
                retry_seconds=0,
                allow_stale_cleanup=False,
            )
        assert path.read_bytes() == raw


def test_timer_accounting_includes_dispatch_and_distinguishes_skipped_ticks():
    from race_collection.freshness_rehearsal import TimerAccounting

    start = datetime(2026, 9, 23, 1, tzinfo=timezone.utc)
    accounting = TimerAccounting(start)

    def sample(seconds, trigger, active, overhead):
        empty = {"ActiveState": "inactive", "ExecMainStartTimestampMonotonic": "0"}
        return {
            "read_start": (start + timedelta(seconds=seconds)).isoformat(),
            "monotonic_start": 100 + seconds,
            "timer_status": {
                "full": {},
                "odds": {"LastTriggerUSecMonotonic": str(trigger * 1000000)},
            },
            "unit_status": {
                "full": empty,
                "odds": {
                    "ActiveState": active,
                    "ExecMainStartTimestampMonotonic": "102000000",
                    "ExecMainExitTimestampMonotonic": "190000000",
                },
            },
            "external_service_overhead_seconds": {"odds": overhead},
        }

    accounting.observe(sample(3, 100, "active", None))
    accounting.observe(sample(90, 160, "inactive", 8))
    summary = accounting.summary(start + timedelta(minutes=2))
    assert [row["status"] for row in summary["odds_calendar_slots"]] == [
        "ACTIVATION_OBSERVED",
        "TRIGGER_WHILE_ACTIVE_NO_NEW_START",
    ]
    assert summary["activations"]["odds"][0]["complete_overhead_seconds"] == 10
    with pytest.raises(ValueError, match="dispatch_plus_process"):
        accounting.observe(sample(91, 160, "inactive", 8.01))
    unknown = TimerAccounting(start)
    unknown.observe(sample(3, 100, "active", None))
    next_activation = sample(125, 220, "active", None)
    next_activation["unit_status"]["odds"]["ExecMainStartTimestampMonotonic"] = "222000000"
    unknown.observe(next_activation)
    assert (
        unknown.summary(start + timedelta(minutes=3))["odds_calendar_slots"][1]["status"]
        == "NO_TRIGGER_OBSERVED"
    )


def test_timer_monotonic_property_accepts_actual_systemctl_timespan_format():
    from race_collection.freshness_rehearsal import timer_monotonic_seconds

    assert timer_monotonic_seconds("1d 4h 52min 52.401544s") == pytest.approx(103972.401544)
    assert timer_monotonic_seconds("104133669160") == pytest.approx(104133.669160)
    with pytest.raises(ValueError):
        timer_monotonic_seconds("infinity")


@pytest.mark.parametrize("claim_after_drain", [False, True])
def test_restoration_keeps_triggers_paused_until_consumed_window_closes(
    tmp_path, monkeypatch, claim_after_drain
):
    from scripts import run_freshness_rehearsal as run
    from race_collection.live_freshness_contract import (
        FreshnessContract,
        AttemptAllowance,
        create_once,
        digest,
    )

    accounting = {
        "schema_version": "freshness_attempt_reconciliation_v1",
        "complete": True,
        "consumed": [],
        "sources": [{"sha256": "b" * 64}],
    }
    value = {**contract_value(tmp_path), "reconciliation_sha256": digest(accounting)}
    scope = FreshnessContract(value)
    allowance = AttemptAllowance(scope)
    allowance.initialize(accounting)
    start = datetime.fromisoformat(value["starts_at"])
    item = {
        "race_id": "synthetic",
        "capture_window_minutes": 60,
        "race_identity": {"jump_datetime": (start + timedelta(hours=1)).isoformat()},
    }
    if not claim_after_drain:
        allowance.reserve(item, now=start)
    output = tmp_path / "output"
    output.mkdir()
    create_once(output / "restoration.json", {})
    elapsed = [0.0]
    commands = []
    monkeypatch.setattr(run, "now", lambda: start + timedelta(seconds=elapsed[0]))

    class Control:
        first = True

        def command(self, *args):
            commands.append(args)

        def idle(self):
            if self.first and claim_after_drain:
                self.first = False
                create_once(allowance.claim, {"item": item})
                return False
            return True

    with pytest.raises(RuntimeError, match="restoration_pending"):
        run.restore(
            output,
            {**value, "cleanup_seconds": 2},
            Control(),
            clock=lambda: elapsed[0],
            sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        )
    assert (output / "RESTORATION_PENDING.json").exists()
    assert commands == [("stop", timer) for timer in run.TIMERS]
    assert not allowance.available()
