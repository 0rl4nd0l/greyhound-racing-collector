"""Phase scheduling for the existing collector entrypoints; not a standalone worker."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from race_collection.live_phase_budget import LiveBudget
from race_collection.live_phase_checkpoint import PhaseCheckpoint, atomic_json, native_publication_lock
from race_collection.synchronous_manual_capture import (
    bounded_current_race_index,
    current_race_index_path,
)


def run_live_collection_cycle(args, *, odds_only: bool):
    cycle_started = getattr(args, "live_started_monotonic", time.monotonic())
    from scripts import shadow_autopilot_daemon as daemon

    if args.current_time is not None or args.input_retention_config or args.forward_baseline_config:
        raise ValueError("live_freshness_requires_live_time_and_disabled_experiments")
    if not args.require_safe_refresh_metadata:
        raise ValueError("live_freshness_requires_safe_metadata")
    if getattr(args, "refresh_dry_run", False) or getattr(args, "skip_refresh", False):
        raise ValueError("live_freshness_conflicts_with_disabled_acquisition")
    if getattr(args, "enable_forward_official_result_observer", False):
        raise ValueError("live_freshness_forbids_result_access")
    profile = getattr(args, "live_freshness_profile", None)
    budget = LiveBudget.for_profile(profile) if profile else LiveBudget()
    evidence = args.evidence_root.resolve()
    state_path = args.state_path
    index_state = state_path if odds_only else args.odds_capture_state_path
    lock_path = (args.lock_path or daemon.DEFAULT_LOCK_PATH).resolve()
    scope = allowance = None
    refresh_deferred = False
    if profile:
        from race_collection.live_freshness_contract import FreshnessContract, AttemptAllowance

        scope = FreshnessContract.load(args.live_freshness_contract)
        scope.check_paths(lock_path=lock_path, evidence_root=evidence, db_path=args.db)
        if args.forward_corpus_root or getattr(
            args, "enable_forward_official_result_observer", False
        ):
            raise ValueError("live_profile_forbids_result_access")
        now = daemon.wall_clock_now()
        if scope.start <= now <= scope.end and (scope.end - now).total_seconds() < 90:
            # Closing admission is not a successful collection. Retain evidence
            # that the real service child deliberately made no source request.
            scope.admit(now, seconds=0)
            invocation = os.environ.get("GREYHOUND_SERVICE_INVOCATION", "")
            if len(invocation) == 32 and all(c in "0123456789abcdef" for c in invocation):
                from race_collection.live_freshness_contract import create_once, digest
                create_once(scope.session / "admission-closures" / (invocation + ".json"), {
                    "schema_version": "live_admission_closed_v1",
                    "runtime_action": "OPERATING_SCOPE_CLOSED",
                    "lane": "odds" if odds_only else "full",
                    "rehearsal_id": scope.value["rehearsal_id"],
                    "contract_sha256": digest(scope.value),
                    "ends_at": scope.value["ends_at"],
                    "service_invocation_id": invocation,
                    "process_pid": os.getpid(),
                    "observed_at": now.isoformat(),
                    "observed_monotonic": time.monotonic(),
                    "required_seconds": 90,
                })
            return {"runtime_action": "OPERATING_SCOPE_CLOSED", "status": "SKIPPED"}
        scope.admit(now, seconds=90)
        allowance = AttemptAllowance(scope)
        allowance.available()  # Missing reconciliation blocks even refresh startup.
    if scope and os.environ.get("GREYHOUND_SERVICE_INVOCATION"):

        def interrupted(signum, frame):
            scope.stop("SERVICE_INTERRUPTED")
            raise InterruptedError("live_service_interrupted")

        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
    lane = "odds" if odds_only else "full"
    run_id = args.run_id or daemon.now_id(daemon.wall_clock_now()) + (
        "_odds_capture" if odds_only else ""
    )
    checkpoint_path = state_path.with_name(f"{lane}.live-phase-checkpoint.json")
    resumable = daemon.load_json(checkpoint_path)
    if resumable and resumable.get("status") == "RUNNING":
        owner_pid = resumable.get("owner_pid")
        if owner_pid and owner_pid != os.getpid():
            try:
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                pass
            else:
                return {"status": "SKIPPED_LOCK_HELD", "runtime_action": "LIVE_CYCLE_OWNER_ACTIVE"}
        run_id = resumable["cycle_id"]
    output = daemon.unique_dir(
        daemon.assert_output_dir_safe(
            args.output_dir or evidence / f"shadow_autopilot_daemonization_v1_{run_id}",
            evidence_root=evidence,
        )
    )
    if resumable and resumable.get("status") == "RUNNING":
        output = daemon.assert_output_dir_safe(
            Path(resumable["output_dir"]), evidence_root=evidence
        )
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
    configuration = {
        key: str(value)
        for key, value in vars(args).items()
        if key not in {"run_id", "output_dir", "live_started_monotonic"}
    }
    source_files = [
        Path(__file__),
        Path(daemon.__file__),
        Path(daemon.autopilot.__file__),
        daemon.ROOT / "scripts/refresh_prejump_upcoming.py",
        daemon.ROOT / "scripts/autonomous_live_odds_capture.py",
        Path(sys.modules[LiveBudget.__module__].__file__),
        Path(sys.modules[PhaseCheckpoint.__module__].__file__),
    ]
    package_identity = None
    if scope and scope.value.get("source_identity_sha256"):
        from race_collection.live_freshness_contract import verify_source_package

        package_identity = verify_source_package(daemon.ROOT, scope.value["source_identity_sha256"])
    identity = hashlib.sha256(
        json.dumps(
            {
                "configuration": configuration,
                "revision": (
                    package_identity["commit"]
                    if package_identity
                    else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=daemon.ROOT)
                    .decode()
                    .strip()
                ),
                "tracked_python_diff": (
                    "PACKAGED_SOURCE"
                    if package_identity
                    else hashlib.sha256(
                        subprocess.check_output(
                            ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--", "*.py"],
                            cwd=daemon.ROOT,
                        )
                    ).hexdigest()
                ),
                "source": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in source_files
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    checkpoint = None
    refresh_result = None
    capture_result = None
    completed_report = None
    outcome = "LIVE_COLLECTION_COMPLETE"
    previous_state = daemon.load_json(state_path) or {}
    timing = {
        "service_invocation_id": os.environ.get("GREYHOUND_SERVICE_INVOCATION"),
        "process_pid": os.getpid(),
        "process_started_monotonic": cycle_started,
        "startup_seconds": time.monotonic() - cycle_started,
        "lock_wait_seconds": 0.0,
        "lock_held_seconds": 0.0,
        "phase_seconds": 0.0,
        "verification_seconds": 0.0,
        "finalization_seconds": 0.0,
        "release_seconds": 0.0,
        "measurement_boundary": (
            "Linux process start including process import through terminal assessment; "
            "stdout and process exit require external measurement"
            if hasattr(args, "live_started_monotonic")
            else "cycle entry through terminal assessment; process import and interpreter startup require external measurement"
        ),
    }
    active_lock_started = None
    last_observed = None
    deferred_owner = None

    def timing_snapshot():
        total = time.monotonic() - cycle_started
        return {
            **timing,
            "lock_held_seconds": timing["lock_held_seconds"]
            + (time.monotonic() - active_lock_started if active_lock_started is not None else 0),
            "total_seconds": total,
            "overhead_seconds": max(
                0, total - timing["phase_seconds"] - timing["lock_wait_seconds"]
            ),
            "source_observed_at": last_observed.isoformat() if last_observed else None,
            "source_age_seconds": budget.age(last_observed, daemon.wall_clock_now()),
        }

    def timing_failed():
        return budget.overhead_exceeded(timing_snapshot()["overhead_seconds"])

    def terminal_timing_failed():
        return timing_failed() or not budget.safe_to_yield(last_observed, daemon.wall_clock_now())

    def report_payload():
        success = outcome == "LIVE_COLLECTION_COMPLETE"
        verdict = "DAEMON_READY" if success else "NEEDS_MORE_AUTOMATION"
        result = {
            "schema_version": (
                "shadow_autopilot_odds_capture_only_daemon_report_v1"
                if odds_only
                else "shadow_autopilot_daemon_run_v1"
            ),
            "generated_at": daemon.wall_clock_now().isoformat(),
            "run_id": run_id,
            "status": ("READY" if success else "FAILED") if odds_only else verdict,
            "final_verdict": verdict,
            "final_status": "ODDS_CAPTURE_ONLY_READY" if success else "ODDS_CAPTURE_ONLY_FAILED",
            "runtime_action": outcome,
            "output_dir": str(output),
            "autopilot_output_dir": (refresh_result or {}).get("output_dir"),
            "current_race_index_publish": (refresh_result or {}).get(
                "current_race_index_publish", {"status": "UNAVAILABLE"}
            ),
            "steps": [(refresh_result or {})["step"]] if (refresh_result or {}).get("step") else [],
            "maintenance_status": "DEFERRED_LIVE_FRESHNESS_PRIORITY",
            "collection_only": True,
            "timing": timing_snapshot(),
            "autonomous_live_odds_capture_status": (capture_result or {}).get(
                "autonomous_live_odds_capture_status", {"status": "NO_ELIGIBLE_WINDOWS"}
            ),
            "phase_checkpoint": str(checkpoint_path),
            **(
                {"live_freshness_profile": profile, "deferred_lock_owner": deferred_owner}
                if profile
                else {}
            ),
            "forward_official_result_observer": previous_state.get(
                "forward_official_result_observer", {"status": "DEFERRED"}
            ),
        }
        if odds_only and refresh_result:
            directory = daemon.rooted_path(refresh_result["output_dir"])
            result["odds_capture_refresh_report"] = (
                daemon.load_json(directory / "odds_capture_refresh_report.json") or {}
            )
        if outcome == "LIVE_TIMING_VALIDATION_PENDING":
            result.update(
                status="RUNNING",
                final_verdict="DAEMON_RUNNING",
                final_status="ODDS_CAPTURE_ONLY_RUNNING",
            )
        if outcome in {"DEFERRED_LOCK_HELD", "DEFERRED_FULL_LOCK_HANDOFF"}:
            skipped = (
                "SKIPPED_FULL_DAEMON_LOCK_HANDOFF"
                if outcome == "DEFERRED_FULL_LOCK_HANDOFF"
                else "SKIPPED_LOCK_HELD"
            )
            result["status"] = skipped
            result["final_status"] = skipped
            result["final_verdict"] = "PARTIAL_DAEMONIZATION"
            result.pop("odds_capture_refresh_report", None)
            result["autopilot_output_dir"] = None
        return result

    def publish_native_report(report):
        name = 'odds_capture_only_daemon_report.json' if odds_only else 'daemon_run_report.json'
        with native_publication_lock(evidence, exclusive=True):
            atomic_json(output / name, report)

    @native_publication_lock(evidence, exclusive=True)
    def finish_checkpoint():
        nonlocal completed_report, outcome
        finalization_started = time.monotonic()
        intended_outcome = outcome
        outcome = "LIVE_TIMING_VALIDATION_PENDING"
        completed_report = report_payload()
        report_name = (
            "odds_capture_only_daemon_report.json" if odds_only else "daemon_run_report.json"
        )
        atomic_json(output / report_name, completed_report)
        for key in (
            "status",
            "runtime_action",
            "readiness_decision",
            "next_meaningful_action",
            "next_meaningful_action_at",
            "inserted_live_odds_rows",
            "ready_count",
            "status_counts",
            "blocked_attempt_count",
        ):
            previous_state.pop(key, None)
        previous_state.update(completed_report)
        previous_state.update(
            {
                "schema_version": (
                    "shadow_autopilot_odds_capture_only_state_v1"
                    if odds_only
                    else "shadow_autopilot_daemon_state_v1"
                ),
                "last_run_id": run_id,
                "last_output_dir": str(output),
                "last_verdict": completed_report["final_verdict"],
                "updated_at": completed_report["generated_at"],
            }
        )
        if odds_only:
            previous_state["odds_capture_refresh_status"] = completed_report.get(
                "odds_capture_refresh_report", {}
            ).get("status")
        atomic_json(state_path, previous_state)
        timing["finalization_seconds"] += time.monotonic() - finalization_started
        outcome = intended_outcome
        if outcome == "LIVE_COLLECTION_COMPLETE" and (
            timing_failed() or not budget.safe_to_yield(last_observed, daemon.wall_clock_now())
        ):
            outcome = "LIVE_TIMING_BUDGET_EXCEEDED"
        finalization_started = time.monotonic()
        checkpoint.value["terminal_timing_path"] = str(output / "terminal-timing.json")
        checkpoint.finish(
            "LIVE_WORK_COMPLETE_TIMING_PENDING"
            if outcome == "LIVE_COLLECTION_COMPLETE"
            else outcome
        )
        timing["finalization_seconds"] += time.monotonic() - finalization_started
        if outcome == "LIVE_COLLECTION_COMPLETE" and terminal_timing_failed():
            outcome = "LIVE_TIMING_BUDGET_EXCEEDED"
        completed_report = report_payload()
        state_schema = previous_state["schema_version"]
        previous_state.update(completed_report)
        previous_state["schema_version"] = state_schema
        previous_state["last_verdict"] = completed_report["final_verdict"]
        previous_state["updated_at"] = completed_report["generated_at"]
        finalization_started = time.monotonic()
        atomic_json(output / report_name, completed_report)
        atomic_json(state_path, previous_state)
        timing["finalization_seconds"] += time.monotonic() - finalization_started
        if outcome == "LIVE_COLLECTION_COMPLETE" and terminal_timing_failed():
            outcome = "LIVE_TIMING_BUDGET_EXCEEDED"
            completed_report = report_payload()
            previous_state.update(completed_report)
            previous_state["schema_version"] = state_schema
            previous_state["last_verdict"] = completed_report["final_verdict"]
            atomic_json(output / report_name, completed_report)
            atomic_json(state_path, previous_state)

    def capture_tasks(view):
        from race_collection.manual_prediction_collector_request import runner_set_sha256

        refresh = daemon.load_json(evidence / view.source_refresh_report_path) or {}
        coverage = (refresh.get("sidecar_metadata_coverage") or {}).get("races") or []
        tasks = []
        for row in view.races:
            match = next(
                (entry for entry in coverage if entry.get("race_url") == row["race_url"]), None
            )
            if match and match.get("csv_path"):
                directory = Path(match["csv_path"]).parent.resolve()
                directory.relative_to(evidence)
                if len(list(directory.glob("*.csv"))) != 1:
                    raise ValueError("capture_phase_requires_single_race_directory")
                tasks.append(
                    {
                        "kind": "capture",
                        "race_id": row["race_id"],
                        "race_id_aliases": row.get("race_id_aliases", [row["race_id"]]),
                        "input_dir": str(directory),
                        "packet_sha256": view.packet_sha256,
                        "capture_runner_set_sha256": (
                            runner_set_sha256(
                                [
                                    {
                                        "box_number": runner["box"],
                                        "dog_name": runner["display_name"],
                                        "identity": runner["identity"],
                                    }
                                    for runner in row["runners"]
                                ]
                            )
                            if scope
                            else None
                        ),
                        "input_files": {
                            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in directory.iterdir()
                            if path.is_file()
                            and (path.suffix == ".csv" or path.name.endswith(".csv.metadata.json"))
                        },
                        "race_identity": {
                            key: row[key]
                            for key in (
                                "race_id",
                                "race_url",
                                "jump_datetime",
                                "source_native_race_id",
                                "runner_set_sha256",
                            )
                        },
                    }
                )
        return tasks

    def queue_work(view):
        from scripts import autonomous_live_odds_capture as capture

        pending = []
        if odds_only or (
            args.enable_autonomous_odds_capture
            and args.execute_autonomous_odds_capture
            and args.allow_auto_scrape_odds
        ):
            tasks = capture_tasks(view)
            plan = capture.build_capture_plan(
                [Path(task["input_dir"]) for task in tasks], current_time=daemon.wall_clock_now()
            )
            ready = sorted(
                [row for row in plan["races"] if row.get("status") == "READY_TO_CAPTURE"],
                key=lambda row: (
                    str(row.get("jump_datetime")),
                    str(row.get("race_id")),
                    row.get("capture_window_minutes") or 0,
                ),
            )
            by_url = {task["race_identity"]["race_url"]: task for task in tasks}
            if scope:
                checkpoint.value.setdefault("window_observations", []).extend(
                    {
                        **{
                            key: row.get(key)
                            for key in (
                                "race_id",
                                "capture_window_minutes",
                                "jump_datetime",
                                "status",
                                "blockers",
                            )
                        },
                        "race_id": by_url.get(row.get("thedogs_source_url"), {}).get(
                            "race_id", row.get("race_id")
                        ),
                        "planner_race_id": row.get("race_id"),
                        "observed_at": daemon.wall_clock_now().isoformat(),
                    }
                    for row in plan["races"]
                )
                if not allowance.available():
                    checkpoint.value.setdefault("exclusions", []).extend(
                        {
                            "race_id": by_url.get(row.get("thedogs_source_url"), {}).get(
                                "race_id", row.get("race_id")
                            ),
                            "planner_race_id": row.get("race_id"),
                            "capture_window_minutes": row.get("capture_window_minutes"),
                            "reason": "shared_capture_allowance_consumed",
                        }
                        for row in ready
                    )
                    return []
            for row in ready:
                if row.get("thedogs_source_url") not in by_url:
                    continue
                existing = capture.existing_capture_runner_status(
                    args.db,
                    race_id=str(row.get("race_id")),
                    capture_mode=f"autonomous_prejump_t{row.get('capture_window_minutes')}m",
                    expected_runners=row.get("expected_runners") or [],
                    jump_datetime=capture.parse_iso_datetime(row.get("jump_datetime")),
                    capture_window_minutes=capture.parse_int_value(
                        row.get("capture_window_minutes")
                    ),
                )
                exclusion = {
                    "race_id": row.get("race_id"),
                    "capture_window_minutes": row.get("capture_window_minutes"),
                }
                if capture.block_or_skip_existing_capture_attempt(exclusion, existing):
                    checkpoint.value.setdefault("exclusions", []).append(exclusion)
                    continue
                task = dict(by_url[row["thedogs_source_url"]])
                if scope:
                    task["capture_window_minutes"] = row["capture_window_minutes"]
                    if allowance.consumed(task):
                        checkpoint.value.setdefault("exclusions", []).append(
                            {
                                "race_id": task["race_id"],
                                "capture_window_minutes": task["capture_window_minutes"],
                                "reason": "previously_consumed",
                            }
                        )
                        continue
                    if scope.value.get("operational_predictions") and task["capture_window_minutes"] != 10:
                        checkpoint.value.setdefault("exclusions", []).append({
                            "race_id": task["race_id"], "capture_window_minutes": task["capture_window_minutes"],
                            "reason": "operational_single_t10_window"})
                        continue
                    if scope.value.get("operational_predictions") and (
                        datetime.fromisoformat(task["race_identity"]["jump_datetime"]) - daemon.wall_clock_now()
                    ).total_seconds() < 300:
                        checkpoint.value.setdefault("exclusions", []).append({
                            "race_id": task["race_id"], "capture_window_minutes": task["capture_window_minutes"],
                            "reason": "insufficient_capture_retention_prediction_margin"})
                        continue
                    try:
                        allowance.check_window(task, now=daemon.wall_clock_now(), required_seconds=50)
                    except ValueError as error:
                        if str(error) not in {"capture_reservation_expired", "capture_reservation_not_open", "capture_window_insufficient_time"}:
                            raise
                        checkpoint.value.setdefault("exclusions", []).append({
                            "race_id": task["race_id"],
                            "capture_window_minutes": task["capture_window_minutes"],
                            "reason": str(error),
                            "observed_at": daemon.wall_clock_now().isoformat(),
                        })
                        continue
                pending.append(task)
                break
        return pending

    def view_now():
        nonlocal last_observed
        verification_started = time.monotonic()
        try:
            view = bounded_current_race_index(
                current_time=daemon.wall_clock_now(),
                timeout_seconds=5,
                index_path=current_race_index_path(index_state),
                evidence_root=evidence,
                max_age_seconds=300 if profile else 1200,
                return_verified_view=True,
            )
        finally:
            timing["verification_seconds"] += time.monotonic() - verification_started
        last_observed = datetime.fromisoformat(view.source_generated_at)
        return view

    def source_observed():
        try:
            return datetime.fromisoformat(view_now().source_generated_at)
        except Exception:
            return None

    def defer_failed_refresh():
        if not scope or not scope.value.get("operational_predictions") or timing_failed():
            return False
        from race_collection.live_freshness_contract import classify_refresh_outage, create_once
        from utils.sportsbet_access import SportsbetAccess
        try:
            scope.admit(daemon.wall_clock_now(), seconds=0)
            access = SportsbetAccess().read()
            if access["phase"] != "OPEN" or access["access_basis"]["status"] != "permitted":
                return False
            classified = classify_refresh_outage(evidence, run_id)
            if classified is None:
                return False
            # Verify the previous publication while still owning the collector
            # lock. This runs no provider/systemd work and holds no writer mutex.
            view_now()
            if not 0 <= budget.age(last_observed, daemon.wall_clock_now()) < 270 or timing_failed():
                return False
            directory = Path(args.live_freshness_contract).resolve().parent / "refresh-deferrals"
            with native_publication_lock(evidence, exclusive=True):
                retained = directory / (run_id + ".json")
                records = list(directory.glob("*.json"))
                if retained.exists():
                    value = json.loads(retained.read_bytes())
                    return all(value.get(key) == item for key, item in classified.items())
                if len(records) >= 2:
                    return False
                create_once(retained, {**classified,
                    "observed_at": daemon.wall_clock_now().isoformat(),
                    "failed_cycle_count": len(records) + 1})
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def command_for(kind, phase_id, inputs):
        command = [
            sys.executable,
            str(daemon.ROOT / "scripts/shadow_autopilot_v1.py"),
            "--run-id",
            phase_id,
            "--evidence-root",
            str(evidence),
            "--collector-lock-path",
            str(lock_path),
            "--current-race-index-state-path",
            str(index_state),
            "--db",
            str(args.db),
            "--skip-shadow-run",
            "--require-safe-refresh-metadata",
            "--refresh-command-mode",
            "python",
            "--current-time",
            daemon.wall_clock_now().isoformat(),
            "--collection-phase",
            kind,
            "--step-timeout-seconds",
            str(args.timeout_seconds),
        ]
        if profile:
            command += [
                "--live-freshness-profile",
                profile,
                "--live-freshness-contract",
                str(args.live_freshness_contract),
            ]
        if kind == "refresh":
            command += [
                "--days-ahead",
                str(args.days_ahead),
                "--refresh-limit",
                str(min(args.refresh_limit, 4) if scope and scope.value.get("operational_predictions") else args.refresh_limit),
            ]
            if odds_only:
                command += [
                    "--skip-primary-refresh",
                    "--enable-autonomous-odds-capture",
                    "--odds-capture-refresh-limit",
                    str(min(args.odds_capture_refresh_limit, 4) if scope and scope.value.get("operational_predictions") else args.odds_capture_refresh_limit),
                    "--odds-capture-min-minutes",
                    str((budget.refresh_seconds + 300) / 60 if scope and scope.value.get("operational_predictions") else args.odds_capture_min_minutes),
                    "--odds-capture-max-minutes",
                    str(args.odds_capture_max_minutes),
                ]
            else:
                command += [
                    "--min-minutes",
                    str((budget.refresh_seconds + 300) / 60 if scope and scope.value.get("operational_predictions") else args.min_minutes),
                    "--max-minutes",
                    str(args.max_minutes),
                ]
        else:
            command += [
                "--skip-refresh",
                "--skip-primary-refresh",
                "--input-dir",
                inputs["input_dir"],
                "--autonomous-odds-capture-limit",
                "1",
                "--enable-autonomous-odds-capture",
                "--execute-autonomous-odds-capture",
                "--allow-auto-scrape-odds",
            ]
            if profile:
                command += ["--live-capture-reservation", inputs["reservation_path"]]
            if args.forward_corpus_root:
                command += ["--forward-corpus-root", str(args.forward_corpus_root)]
        return command

    def phase(kind, inputs=None):
        nonlocal checkpoint, refresh_result, capture_result, outcome, output, run_id, active_lock_started, deferred_owner, refresh_deferred
        inputs = inputs or {}
        if scope:
            scope.admit(daemon.wall_clock_now(), seconds=90 if kind == "refresh" else 155)
        marker = daemon.read_active_full_daemon_lock_wait_marker(lock_path) if odds_only else None
        if marker:
            deferred_owner = (
                {"run_id": marker.get("run_id"), "pid": marker.get("pid")}
                if isinstance(marker, dict)
                else None
            )
            outcome = "DEFERRED_FULL_LOCK_HANDOFF"
            return False
        if profile:
            waiting = report_payload()
            waiting.update(
                status="RUNNING",
                final_verdict="DAEMON_RUNNING",
                final_status="ODDS_CAPTURE_ONLY_RUNNING",
            )
            publish_native_report(waiting)
        try:
            owner = daemon.acquire_lock_with_odds_capture_retry(
                lock_path=lock_path,
                run_id=run_id,
                stale_after_seconds=args.lock_stale_seconds,
                output_dir=output,
                retry_seconds=(
                    0 if odds_only else int(budget.refresh_seconds + budget.overhead_seconds + 5)
                ),
                **({"allow_stale_cleanup": False} if scope else {}),
            )
        except daemon.LockBusy as error:
            owner_fields = daemon.lock_owner_report_fields(error.payload)
            deferred_owner = {
                "run_id": owner_fields.get("lock_owner_run_id"),
                "pid": owner_fields.get("lock_owner_pid"),
            }
            timing["lock_wait_seconds"] += error.payload.get("lock_retry", {}).get(
                "waited_seconds", 0.0
            )
            outcome = "DEFERRED_LOCK_HELD"
            return False
        acquired = time.monotonic()
        timing["lock_wait_seconds"] += owner.get("lock_retry", {}).get("waited_seconds", 0.0)
        active_lock_started = acquired
        try:
            if scope:
                scope.admit(daemon.wall_clock_now(), seconds=90 if kind == "refresh" else 155)
                if scope.value.get("operational_predictions"):
                    directory = Path(args.live_freshness_contract).resolve().parent / "refresh-deferrals"
                    # Collector ownership serializes this check with the prior
                    # cycle's failure classification, before any next request.
                    with native_publication_lock(evidence, exclusive=True):
                        exhausted = len(list(directory.glob("*.json"))) >= 2
                    if exhausted:
                        scope.stop("REFRESH_OUTAGE_LIMIT_REACHED")
                        raise ValueError("refresh_outage_limit_reached")
            running = report_payload()
            running.update(
                status="RUNNING",
                final_verdict="DAEMON_RUNNING",
                runtime_action="FULL_DAEMON_IN_PROGRESS",
                lock=owner,
                lock_path=str(lock_path),
            )
            if odds_only:
                running.update(
                    final_status="ODDS_CAPTURE_ONLY_RUNNING",
                    **daemon.odds_capture_only_operator_fields("ODDS_CAPTURE_ONLY_RUNNING"),
                )
            publish_native_report(running)
            if checkpoint is None:
                checkpoint = PhaseCheckpoint(
                    checkpoint_path, identity=identity, cycle_id=run_id, output_dir=output
                )
                checkpoint.value["owner_pid"] = os.getpid()
                previous_state.update(checkpoint.value.get("progress", {}))
                for retained in reversed(checkpoint.value["phases"]):
                    if retained["kind"] == "capture" and retained["status"] == "COMPLETE":
                        capture_result = daemon.load_json(Path(retained["result_path"]))
                        break
                if checkpoint.value["cycle_id"] != run_id:
                    outcome = "INTERRUPTED_CYCLE_REQUIRES_RECONCILIATION"
                    return False
            if kind != "refresh" and not budget.admit_work(
                source_observed(), daemon.wall_clock_now()
            ):
                return "refresh"
            if kind == "capture":
                if scope and (not allowance.available() or allowance.consumed(inputs)):
                    checkpoint.value["pending"].pop(0)
                    checkpoint.value.setdefault("exclusions", []).append(
                        {
                            "race_id": inputs["race_id"],
                            "capture_window_minutes": inputs["capture_window_minutes"],
                            "reason": "shared_capture_allowance_consumed",
                        }
                    )
                    atomic_json(checkpoint.path, checkpoint.value)
                    return "excluded"
                view = view_now()
                selected = next(
                    (row for row in view.races if row["race_id"] == inputs["race_id"]), None
                )
                if (
                    selected is None
                    or datetime.fromisoformat(selected["jump_datetime"]) <= daemon.wall_clock_now()
                ):
                    checkpoint.value["pending"].pop(0)
                    checkpoint.value.setdefault("exclusions", []).append(
                        {
                            "race_id": inputs["race_id"],
                            "capture_window_minutes": inputs.get("capture_window_minutes"),
                            "reason": "race_missing_or_no_longer_prejump",
                        }
                    )
                    atomic_json(checkpoint.path, checkpoint.value)
                    if not checkpoint.value["pending"] and budget.safe_to_yield(
                        source_observed(), daemon.wall_clock_now()
                    ):
                        finish_checkpoint()
                    return "excluded"
                if view.packet_sha256 != inputs["packet_sha256"]:
                    replacement = next(
                        (
                            task
                            for task in capture_tasks(view)
                            if task["race_identity"] == inputs["race_identity"]
                        ),
                        None,
                    )
                    if replacement is None:
                        outcome = "DEFERRED_RACE_IDENTITY_CHANGED"
                        finish_checkpoint()
                        return False
                    checkpoint.value.setdefault("rebindings", []).append(
                        {
                            "previous": inputs,
                            "replacement": replacement,
                            "reason": "unstarted_same_race_and_runner_set_fresh_observation",
                        }
                    )
                    if scope:
                        replacement["capture_window_minutes"] = inputs["capture_window_minutes"]
                    inputs = replacement
                    checkpoint.value["pending"][0] = replacement
            if scope and kind == "capture":
                # Reservation is consumed before subprocess launch, even if launch fails.
                try:
                    # Fund the entire bounded capture, including child startup, before
                    # consuming. A stale queued item never becomes a later window.
                    if scope.value.get("operational_predictions") and (
                        datetime.fromisoformat(inputs["race_identity"]["jump_datetime"]) - daemon.wall_clock_now()
                    ).total_seconds() < 300:
                        raise ValueError("insufficient_capture_retention_prediction_margin")
                    allowance.check_window(inputs, now=daemon.wall_clock_now(), required_seconds=50)
                    inputs["reservation_path"] = str(
                        allowance.reserve(inputs, now=daemon.wall_clock_now())
                    )
                except ValueError as error:
                    if str(error) not in {
                        "capture_reservation_expired", "capture_reservation_not_open",
                        "capture_window_insufficient_time", "insufficient_capture_retention_prediction_margin",
                    }:
                        raise
                    checkpoint.value["pending"].pop(0)
                    checkpoint.value.setdefault("exclusions", []).append({
                        "race_id": inputs["race_id"],
                        "capture_window_minutes": inputs["capture_window_minutes"],
                        "reason": str(error),
                        "observed_at": daemon.wall_clock_now().isoformat(),
                    })
                    atomic_json(checkpoint.path, checkpoint.value)
                    return "excluded"
            started = time.monotonic()
            record = checkpoint.begin(kind, inputs, daemon.wall_clock_now().isoformat())
            phase_id = f"{run_id}_phase_{record['number']}"
            step = daemon.run_command(
                name="autopilot_cycle" if kind == "refresh" else "odds_capture_autopilot_cycle",
                command=command_for(kind, phase_id, inputs),
                output_dir=output / f"phase-{record['number']}",
                timeout_seconds=args.timeout_seconds * 2,
                wait_for_descendants=True,
            )
            stdout = output / f"phase-{record['number']}" / "logs" / f"{step['name']}.stdout.txt"
            result = daemon.load_json(stdout) or {
                "status": "FAIL",
                "reason": "phase_output_missing",
            }
            result["step"] = step
            if step.get("returncode") != 0:
                result["status"] = "FAIL"
            operational_unready = None
            if scope and kind == "capture":
                if scope.value.get("operational_predictions"):
                    from race_collection.operational_prediction import classify_unready_capture
                    operational_unready = classify_unready_capture(
                        inputs["reservation_path"], result, args.evidence_root, Path(__file__).resolve().parents[1])
                    if operational_unready:
                        result["operational_capture_outcome"] = operational_unready
                        checkpoint.value.setdefault("exclusions", []).append(operational_unready)
                allowance.finish(inputs["reservation_path"], result)
                if (
                    result.get("autonomous_live_odds_capture_status", {}).get("status")
                    != "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
                ):
                    result["status"] = "FAIL"
            elapsed = time.monotonic() - started
            timing["phase_seconds"] += elapsed
            overrun = budget.overrun(kind, elapsed)
            if kind != "refresh" and checkpoint.value["pending"]:
                checkpoint.value["pending"].pop(0)
            if kind == "capture":
                capture_result = result
                previous_state["live_capture_cursor"] = inputs["race_id"]
            checkpoint.value["progress"] = {
                key: previous_state[key]
                for key in (
                    "live_capture_cursor",
                    "live_result_cursor",
                    "forward_official_result_observer",
                )
                if key in previous_state
            }
            if (
                kind == "refresh"
                and record["number"] == 0
                and not overrun
                and result.get("status") == "PASS"
            ):
                checkpoint.value["pending"] = queue_work(view_now())
            elapsed = time.monotonic() - started
            overrun = budget.overrun(kind, elapsed)
            checkpoint.complete(result, elapsed=elapsed, overrun=overrun)
            if kind == "refresh":
                refresh_result = result
            if overrun or (result.get("status") != "PASS" and not operational_unready) or timing_failed():
                outcome = (
                    "LIVE_PHASE_BUDGET_EXCEEDED"
                    if overrun
                    else (
                        "LIVE_PHASE_FAILED"
                        if result.get("status") != "PASS"
                        else "LIVE_TIMING_BUDGET_EXCEEDED"
                    )
                )
                finish_checkpoint()
                if kind == "refresh" and outcome == "LIVE_PHASE_FAILED":
                    refresh_deferred = defer_failed_refresh()
                if scope and not refresh_deferred:
                    scope.stop(outcome)
                return False
            if not checkpoint.value["pending"] and budget.safe_to_yield(
                source_observed(), daemon.wall_clock_now()
            ):
                finish_checkpoint()
            return True
        finally:
            release_started = time.monotonic()
            release = daemon.release_lock(lock_path, owner["run_id"])
            timing["release_seconds"] += time.monotonic() - release_started
            timing["lock_held_seconds"] += time.monotonic() - acquired
            active_lock_started = None
            atomic_json(
                output / f"lock-release-{uuid.uuid4().hex}.json",
                {
                    "kind": kind,
                    "owner": owner,
                    "release": release,
                    "released_at": daemon.wall_clock_now().isoformat(),
                    "lock_held_seconds": time.monotonic() - acquired,
                },
            )

    try:
        if phase("refresh") is True:
            while checkpoint.value["pending"]:
                task = checkpoint.value["pending"][0]
                decision = phase(task["kind"], task)
                if decision == "refresh":
                    if phase("refresh") is not True:
                        break
                    continue
                if decision is False:
                    break
            if (
                completed_report is None
                and outcome == "LIVE_COLLECTION_COMPLETE"
                and not budget.safe_to_yield(source_observed(), daemon.wall_clock_now())
            ):
                phase("refresh")
            if (
                completed_report is None
                and outcome == "LIVE_COLLECTION_COMPLETE"
                and not budget.safe_to_yield(source_observed(), daemon.wall_clock_now())
            ):
                outcome = "LIVE_PHASE_BUDGET_EXCEEDED"
    except Exception as error:
        if scope:
            scope.stop(type(error).__name__)
        outcome = "LIVE_COLLECTION_BLOCKED"
        atomic_json(
            output / "phase_error.json", {"type": type(error).__name__, "reason": str(error)}
        )
    if outcome == "LIVE_COLLECTION_COMPLETE" and terminal_timing_failed():
        outcome = "LIVE_TIMING_BUDGET_EXCEEDED"
    if scope and outcome not in {
        "LIVE_COLLECTION_COMPLETE",
        "DEFERRED_LOCK_HELD",
        "DEFERRED_FULL_LOCK_HANDOFF",
    } and not (refresh_deferred and outcome == "LIVE_PHASE_FAILED" and not timing_failed()):
        scope.stop(outcome)
    with native_publication_lock(evidence, exclusive=True):
        terminal_report = report_payload()
        atomic_json(output / "terminal-timing.json", terminal_report)
    if outcome == "LIVE_COLLECTION_COMPLETE" and terminal_timing_failed():
        outcome = "LIVE_TIMING_BUDGET_EXCEEDED"
        with native_publication_lock(evidence, exclusive=True):
            terminal_report = report_payload()
            atomic_json(output / "terminal-timing.json", terminal_report)
    if completed_report is not None and completed_report["runtime_action"] == outcome:
        return terminal_report
    result = report_payload()
    report_name = "odds_capture_only_daemon_report.json" if odds_only else "daemon_run_report.json"
    publish_native_report(result)
    if completed_report is not None and outcome == "LIVE_TIMING_BUDGET_EXCEEDED":
        from race_collection.synchronous_manual_capture import (
            CollectorBusy,
            acquire_collector_lock_no_steal,
            release_owned_collector_lock,
        )

        try:
            correction_lock = acquire_collector_lock_no_steal(
                lock_path, run_id=run_id, output_dir=output, phase="timing_failure_report"
            )
        except CollectorBusy:
            result["state_correction"] = "DEFERRED_LOCK_HELD"
        else:
            try:
                current_state = daemon.load_json(state_path) or {}
                if current_state.get("last_run_id") == run_id:
                    state_schema = current_state["schema_version"]
                    current_state.update(result)
                    current_state["schema_version"] = state_schema
                    current_state["last_verdict"] = result["final_verdict"]
                    with native_publication_lock(evidence, exclusive=True):
                        atomic_json(state_path, current_state)
            finally:
                release_owned_collector_lock(correction_lock)
        atomic_json(output / "terminal-timing-failure.json", result)
    return result
