#!/usr/bin/env python3
"""One explicitly approved scheduled rehearsal, or restoration only.

Preparation never calls this executor. A plan digest and approval identity are
required; started.json is consumed even if admission fails. No retry/resume mode.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from race_collection.live_freshness_contract import (
    AttemptAllowance,
    FreshnessContract,
    create_once,
    digest,
    verify_source_package,
)
from race_collection.live_phase_checkpoint import atomic_json
from scripts.prepare_freshness_rehearsal import UNITS
from scripts.check_freshness_runtime import verify_runtime

TIMERS = ("shadow-autopilot.timer", "shadow-autopilot-odds-capture.timer")
SERVICES = ("shadow-autopilot.service", "shadow-autopilot-odds-capture.service")


def now():
    return datetime.now(timezone.utc)


class SystemdControl:
    def command(self, *args):
        try:
            return subprocess.check_output(["systemctl", "--user", *args], text=True, timeout=3)
        except subprocess.CalledProcessError as error:
            if args[0] == "is-enabled" and error.returncode == 1 and error.output.strip() == "disabled":
                return error.output
            raise

    def show(self, unit):
        raw = self.command(
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "ControlGroup",
            "-p",
            "ExecMainPID",
            "-p",
            "InvocationID",
            "-p",
            "ExecMainStartTimestampMonotonic",
            "-p",
            "ExecMainExitTimestampMonotonic",
            "-p",
            "DropInPaths",
            "-p",
            "WorkingDirectory",
        )
        return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)

    def idle(self):
        for unit in SERVICES:
            value = self.show(unit)
            if value["ActiveState"] not in {"inactive", "failed"} or int(value["MainPID"]) != 0:
                return False
            group = value.get("ControlGroup")
            if group:
                directory = Path("/sys/fs/cgroup") / group.lstrip("/")
                if any(p.read_text().strip() for p in directory.glob("**/cgroup.procs")):
                    return False
        return True


def restore(output, plan, control, *, clock=time.monotonic, sleep=time.sleep):
    backup = json.loads((output / "restoration.json").read_bytes())
    for timer in TIMERS:
        control.command("stop", timer)
    deadline = clock() + plan["cleanup_seconds"]
    claim = (
        Path(plan["lock_path"]).parent
        / "live-freshness-attempts-v1"
        / ("rehearsal-" + hashlib.sha256(plan["rehearsal_id"].encode()).hexdigest())
        / "capture-reservation.json"
    )
    try:
        create_once(claim.parent / "STOP.json", {"reason": "RESTORATION_REQUESTED"})
    except FileExistsError:
        pass
    # An admitted child may reserve just after STOP; classify only after drain.
    while not control.idle() or Path(plan["lock_path"]).exists():
        if clock() >= deadline:
            atomic_json(
                output / "RESTORATION_PENDING.json",
                {"reason": "natural_drain_deadline", "no_process_killed": True},
            )
            raise RuntimeError("restoration_pending_natural_drain")
        sleep(1)
    try:
        closes = None
        claims = list(claim.parent.glob("captures/*/capture-reservation.json")) if plan.get("campaign_root") else ([claim] if claim.exists() else [])
        items = [json.loads(path.read_bytes())["item"] for path in claims]
        if plan.get("campaign_root"):
            from race_collection.freshness_campaign import Campaign
            with Campaign(plan["campaign_root"]).ledger() as ledger:
                items.extend(row["item"] for row in ledger["attempts"]
                             if Path(row["claim"]).is_relative_to(claim.parent))
        for item in items:
            from scripts.autonomous_live_odds_capture import capture_window_bounds

            _, window = AttemptAllowance.key(item)
            _, boundary = capture_window_bounds(
                jump_datetime=datetime.fromisoformat(item["race_identity"]["jump_datetime"]),
                capture_window_minutes=window,
            )
            closes = max(closes, boundary) if closes else boundary
            if closes.utcoffset() is None:
                raise ValueError("ambiguous_reserved_window")
    except (ValueError, KeyError, TypeError) as error:
        atomic_json(
            output / "RESTORATION_PENDING.json", {"reason": "reserved_window_unclassifiable"}
        )
        raise RuntimeError("restoration_pending_reserved_window") from error
    while (
        not control.idle()
        or Path(plan["lock_path"]).exists()
        or (closes is not None and now() <= closes)
    ):
        if clock() >= deadline:
            atomic_json(
                output / "RESTORATION_PENDING.json",
                {"reason": "natural_drain_or_reserved_window_deadline", "no_process_killed": True},
            )
            raise RuntimeError("restoration_pending_natural_drain")
        sleep(1)
    for name in UNITS:
        raw = (output / "backup" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != backup["hashes"][name]:
            raise ValueError("restoration_backup_changed")
        target = Path(plan["installed_dir"]) / name
        temporary = target.with_name(
            target.name + ".freshness-restore-" + uuid.uuid4().hex + ".tmp"
        )
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, backup["modes"][name])
        os.replace(temporary, target)
    control.command("daemon-reload")
    source_hold = False
    if plan.get("sportsbet_access_state"):
        from utils.sportsbet_access import SportsbetAccess
        source_hold = SportsbetAccess(plan["sportsbet_access_state"]).blocks_restoration()
    legacy_unverified = bool(plan.get("sportsbet_access_state")) and not plan.get("baseline_source_coordination_verified", False)
    triggers_held = source_hold or legacy_unverified
    for timer in TIMERS:
        if triggers_held:
            control.command("disable", timer)
        else:
            if backup["enabled"][timer] in {"enabled", "disabled"} and control.command("is-enabled", timer).strip() != backup["enabled"][timer]:
                control.command("enable" if backup["enabled"][timer] == "enabled" else "disable", timer)
            if backup["active"][timer]:
                control.command("start", timer)
    for name, expected in backup["hashes"].items():
        if (
            hashlib.sha256((Path(plan["installed_dir"]) / name).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError("restored_unit_hash_mismatch")
    for timer in TIMERS:
        if (control.show(timer)["ActiveState"] == "active") != (backup["active"][timer] and not triggers_held):
            raise ValueError("restored_timer_activity_mismatch")
        expected_enabled = "disabled" if triggers_held else backup["enabled"][timer]
        if control.command("is-enabled", timer).strip() != expected_enabled:
            raise ValueError("restored_timer_enablement_mismatch")
    if control.show("greyhound-operator-ui-r3.service")["MainPID"] != backup["r3_pid"]:
        raise ValueError("r3_process_changed")
    atomic_json(
        output / "restored.json",
        {"status": "RESTORED_COLLECTOR_TRIGGERS_HELD" if triggers_held else "RESTORED",
         "at": now().isoformat(), "hashes": backup["hashes"],
         "sportsbet_hold": source_hold, "baseline_source_coordination_unverified": legacy_unverified,
         "paused_timers": list(TIMERS) if triggers_held else []},
    )


def snapshot(output, plan, control):
    backup = output / "backup"
    backup.mkdir(exist_ok=False)
    names = (*UNITS, "greyhound-operator-ui-r3.service")
    hashes, modes = {}, {}
    for name in names:
        if control.show(name).get("DropInPaths"):
            raise ValueError("unreviewed_unit_dropins")
        path = Path(plan["installed_dir"]) / name
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if hashes[name] != plan["baseline_unit_sha256"][name]:
            raise ValueError("installed_baseline_changed")
        with (backup / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        modes[name] = path.stat().st_mode & 0o777
    result = {
        "hashes": hashes,
        "modes": modes,
        "active": {timer: control.show(timer)["ActiveState"] == "active" for timer in TIMERS},
        "enabled": {timer: control.command("is-enabled", timer).strip() for timer in TIMERS},
        "r3_pid": control.show("greyhound-operator-ui-r3.service")["MainPID"],
    }
    directory = os.open(backup, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    create_once(output / "restoration.json", result)
    return result


def sample(plan, output, control):
    first = _sample_once(plan, output, control)
    if not _clock_boundary_rejection(first):
        return first
    # Retain the original failed read before a single full local resample.
    # No source requests occur here and no native validator is relaxed.
    retained = output / "clock-boundary-samples" / (uuid.uuid4().hex + ".json")
    create_once(retained, first)
    second = _sample_once(plan, output, control)
    second.update(read_start=first['read_start'], monotonic_start=first['monotonic_start'],
                  clock_boundary_resample={'initial_observation_path':str(retained),
                    'initial_observation_sha256':digest(first), 'attempts':2})
    return second


def _clock_boundary_rejection(value):
    if (value.get('collector_status') != 'INVALID/INTEGRITY_FAILED'
            or value.get('index_status') != 'AVAILABLE/FRESH'
            or value.get('authority_status') != 'AVAILABLE/FRESH'):
        return False
    failed = [lane for lane in value.get('lanes', []) if lane.get('status') in {'INTEGRITY_FAILED', 'DIVERGENT'}]
    if not failed:
        return False
    try:
        start, end = (datetime.fromisoformat(value[key]) for key in ('read_start', 'read_end'))
        elapsed = value['monotonic_end'] - value['monotonic_start']
        if (start.tzinfo is None or end.tzinfo is None or not 0 <= elapsed <= 1
                or abs((end-start).total_seconds()-elapsed) > .25):
            return False
        for lane in failed:
            identity = lane['component_identity']
            if (lane['status'] != 'INTEGRITY_FAILED'
                    or identity.get('rejection') != 'producer_timestamp_after_observation'
                    or not lane.get('run_id') or lane['run_id'] == 'unavailable'
                    or any(not re.fullmatch('[0-9a-f]{64}', lane['reference_hashes'].get(key, ''))
                           for key in ('report', 'state'))):
                return False
            stamps = [datetime.fromisoformat(identity[key]) for key in
                      ('report_generated_at', 'state_updated_at') if key in identity]
            if (not stamps or any(stamp.tzinfo is None or stamp > end for stamp in stamps)
                    or not any(stamp > start for stamp in stamps)):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _sample_once(plan, output, control):
    from race_collection.live_phase_checkpoint import native_publication_lock
    wall_started, started = now(), time.monotonic()
    status_started = time.monotonic()
    status_snapshot = _sample_status(control)
    status_elapsed = time.monotonic() - status_started
    wait_started = time.monotonic()
    with native_publication_lock(Path(plan['evidence_root']), exclusive=False, timeout_seconds=1.0):
        waited = time.monotonic() - wait_started
        result = _sample_locked(plan, output, status_snapshot)
    result['status_probe_seconds'] = status_elapsed
    result['publication_lock_wait_seconds'] = waited
    result['monotonic_start'] = started
    # Retain the complete read/wait interval; native validation uses the clock
    # sampled after acquisition, never an artificially advanced timestamp.
    result['read_start'] = wall_started.isoformat()
    return result


def _sample_status(control):
    # Service queries can each take seconds. Never hold a publication lock while
    # waiting for systemd; native file validation samples its own clock later.
    r3 = control.show("greyhound-operator-ui-r3.service")
    status = {lane: control.show(name) for lane, name in zip(("full", "odds"), SERVICES)}
    timers = {}
    for lane, name in zip(("full", "odds"), TIMERS):
        raw_timer = control.command(
            "show", name, "-p", "LastTriggerUSecMonotonic", "-p",
            "NextElapseUSecMonotonic", "-p", "NextElapseUSecRealtime", "-p", "ActiveState",
        )
        timers[lane] = dict(line.split("=", 1) for line in raw_timer.splitlines() if "=" in line)
    return r3, status, timers


def _sample_locked(plan, output, status_snapshot):
    from race_collection.freshness_rehearsal import native_observation, completed_service_overhead
    from race_collection.synchronous_manual_capture import current_race_index_path
    from src.operator_ui.live_adapters import InstalledUnits

    evidence = Path(plan["evidence_root"])
    runtime = evidence / "shadow_autopilot_daemon_runtime"
    read_start, mono_start = now(), time.monotonic()
    backup = json.loads((output / "restoration.json").read_bytes())
    r3 = "greyhound-operator-ui-r3.service"
    r3_status, status, timers = status_snapshot
    if (
        r3_status["MainPID"] != backup["r3_pid"]
        or hashlib.sha256((Path(plan["installed_dir"]) / r3).read_bytes()).hexdigest()
        != backup["hashes"][r3]
    ):
        raise ValueError("installed_r3_changed")
    paths = {
        "full_state": runtime / "state.json",
        "odds_state": runtime / "odds_capture_state.json",
    }
    for lane, filename in (
        ("full", "daemon_run_report.json"),
        ("odds", "odds_capture_only_daemon_report.json"),
    ):
        candidates = list(evidence.glob("shadow_autopilot_daemonization_v1_*/" + filename))
        paths[lane + "_report"] = (
            max(candidates, key=lambda path: path.stat().st_mtime_ns)
            if candidates
            else runtime / ("pending-" + filename)
        )
    state = json.loads(paths["odds_state"].read_bytes()) if paths["odds_state"].exists() else {}
    paths["odds_refresh"] = (
        Path(state.get("autopilot_output_dir") or str(runtime / "pending"))
        / "odds_capture_refresh_report.json"
    )
    unit_map = dict(zip(("full_service", "full_timer", "odds_service", "odds_timer"), UNITS))
    raw = {key: (Path(plan["installed_dir"]) / name).read_bytes() for key, name in unit_map.items()}
    if any(
        value.get("WorkingDirectory") != plan["source_root"] or value.get("DropInPaths")
        for value in status.values()
    ):
        raise ValueError("loaded_candidate_unit_changed")
    values = {}
    for lane, observed in status.items():
        values.update(
            {
                lane + "_unit_name": SERVICES[0 if lane == "full" else 1],
                lane + "_active_state": observed["ActiveState"],
                lane + "_sub_state": observed["SubState"],
                lane + "_exec_main_pid": int(observed["MainPID"]),
            }
        )
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
    units = InstalledUnits(
        **raw,
        **values,
        **{key + "_sha256": value for key, value in hashes.items()},
        observed_at=read_start,
        working_directory=plan["source_root"],
    )
    authority = {
        **plan,
        "unit_sha256": {key: plan["unit_sha256"][name] for key, name in unit_map.items()},
    }
    result = native_observation(
        now=read_start,
        paths=paths,
        units=units,
        evidence_root=evidence,
        index_path=current_race_index_path(runtime / "odds_capture_state.json"),
        authority=authority,
        output=output / "observations",
    )
    result.update(
        read_start=read_start.isoformat(),
        read_end=now().isoformat(),
        monotonic_start=mono_start,
        monotonic_end=time.monotonic(),
        unit_status=status,
        timer_status=timers,
    )
    result["external_service_overhead_seconds"] = {}
    for lane in ("full", "odds"):
        terminal = paths[lane + "_report"].parent / "terminal-timing.json"
        invocation = status[lane].get("InvocationID", "")
        lifecycle = (
            (runtime / "service-lifecycles" / (invocation + ".json"))
            if re.fullmatch(r"[0-9a-f]{32}", invocation)
            else None
        )
        if terminal.exists():
            result["external_service_overhead_seconds"][lane] = completed_service_overhead(
                status[lane],
                json.loads(terminal.read_bytes()),
                json.loads(lifecycle.read_bytes()) if lifecycle and lifecycle.exists() else None,
            )
    lock = Path(plan["lock_path"])
    result["lock"] = json.loads(lock.read_bytes()) if lock.exists() else None
    if result["lock"] and not str(result["lock"].get("output_dir", "")).startswith(
        str(evidence) + "/"
    ):
        raise ValueError("unapproved_lock_owner")
    return result


def window_accounting(rows, exclusions, claim, end):
    """Observed eligibility only; refresh caps leave other windows unassessed."""
    eligible = {}
    for row in rows:
        if row.get("status") == "READY_TO_CAPTURE":
            eligible[(row["race_id"], row["capture_window_minutes"])] = row
    attempted = set()
    for path in ([claim] if isinstance(claim, Path) and claim.exists() else (claim if isinstance(claim, list) else [])):
        item = json.loads(path.read_bytes())["item"]
        attempted.add((item["race_id"], item["capture_window_minutes"]))
    excluded = {(row["race_id"], row.get("capture_window_minutes")) for row in exclusions}
    missed = []
    pending = []
    for key, row in eligible.items():
        if key in attempted or key in excluded:
            continue
        from scripts.autonomous_live_odds_capture import capture_window_bounds

        # Native window boundaries, including the next-window cutoff.
        jump = datetime.fromisoformat(row["jump_datetime"])
        _, closes = capture_window_bounds(jump_datetime=jump, capture_window_minutes=key[1])
        (missed if closes <= end else pending).append(row)
    return {
        "observation_ended_at": end.isoformat(),
        "eligible_observed_windows": list(eligible.values()),
        "attempted_windows": [
            {"race_id": race, "capture_window_minutes": window}
            for race, window in sorted(attempted)
        ],
        "excluded_observations": exclusions,
        "missed_observed_windows": missed,
        "pending_at_end": pending,
        "outside_observed_refresh_coverage": "UNASSESSED",
    }


def verify_claim_receipt(claim, handoff, evidence, source_root):
    """A race-level discovery result is not proof of this reservation's append."""
    reserved = json.loads(claim.read_bytes())
    terminal = json.loads(claim.with_suffix(".terminal.json").read_bytes())["result"]
    directory = Path(terminal["autonomous_live_odds_capture_status"]["output_dir"])
    directory = directory if directory.is_absolute() else source_root / directory
    if not directory.resolve().is_relative_to(evidence.resolve()):
        raise ValueError("capture_report_outside_campaign_evidence")
    report = json.loads((directory / "autonomous_live_odds_capture_report.json").read_bytes())
    attempts = report.get("attempts", [])
    source = json.loads(handoff["_report_bytes"])
    sealed = source["source_attempt"]
    if (len(attempts) != 1 or attempts[0].get("collector_exact_receipt_publish", {}).get("status") != "PUBLISHED"
            or {key: attempts[0].get(key) for key in sealed} != sealed
            or sealed.get("race_id") != reserved["item"]["race_id"]
            or sealed.get("capture_window_minutes") != reserved["item"]["capture_window_minutes"]
            or source["source_plan_item"].get("capture_window_minutes") != reserved["item"]["capture_window_minutes"]
            or datetime.fromisoformat(sealed["fetch_time"]) < datetime.fromisoformat(reserved["reserved_at"])):
        raise ValueError("capture_receipt_reservation_mismatch")
    AttemptAllowance.check_window(reserved["item"], now=datetime.fromisoformat(sealed["append_time"]))


def observe(output, plan, control, scope, predictions=None):
    from race_collection.freshness_rehearsal import assess_interval, TimerAccounting

    evidence = Path(plan["evidence_root"])
    end = datetime.fromisoformat(plan["ends_at"])
    start = datetime.fromisoformat(plan["starts_at"])
    timer_accounting = TimerAccounting(start)
    previous = None
    event_count = 0
    maximum = 0
    samples = 0
    unavailable = 0
    completed = {"full": set(), "odds": set()}
    waits = []
    window_rows = []
    exclusions = []
    seen = set()
    external_overheads = {"full": [], "odds": []}
    while now() < end:
        tick = time.monotonic()
        if predictions is not None:
            predictions.tick()
        if (scope.session / "STOP.json").exists():
            raise ValueError("candidate_scope_stopped")
        allowance = AttemptAllowance(scope)
        for claim in allowance.claims():
            verification = (output / "capture-verifications" / (claim.parent.name + ".json")
                            if scope.campaign else output / "capture-receipt-verification.json")
            rejection = output / "capture-rejections" / (claim.parent.name + ".json")
            if rejection.exists():
                continue
            if not claim.with_suffix(".terminal.json").exists() or verification.exists():
                continue
            if plan.get("operational_predictions"):
                from race_collection.operational_prediction import classify_unready_capture
                terminal = json.loads(claim.with_suffix(".terminal.json").read_bytes())["result"]
                unready = classify_unready_capture(claim, terminal, evidence, Path(plan["source_root"]))
                if unready:
                    create_once(rejection, unready)
                    continue
            from race_collection.manual_prediction_collector_request import (
                ManualPredictionCollectorProtocol,
            )

            item = json.loads(claim.read_bytes())["item"]
            handoff = ManualPredictionCollectorProtocol(
                evidence / "manual_prediction_collector_requests_v1"
            ).discover_collector_exact_handoff(
                race_id=item["race_id"], current_time=now(), max_age_seconds=300
            )
            if handoff is None:
                raise ValueError("native_capture_receipt_unavailable")
            verify_claim_receipt(claim, handoff, evidence, Path(plan["source_root"]))
            create_once(
                verification,
                {
                    "status": "NATIVE_HANDOFF_VERIFIED",
                    "race_id": item["race_id"],
                    "capture_window_minutes": item["capture_window_minutes"],
                    "capture_attempt_sha256": handoff["capture_attempt_sha256"],
                    "observed_at": now().isoformat(),
                    "prediction_started": False,
                },
            )
        for checkpoint in evidence.glob(
            "shadow_autopilot_daemonization_v1_*/phase-checkpoint.json"
        ):
            if str(checkpoint) in seen:
                continue
            terminal = checkpoint.parent / "terminal-timing.json"
            if not terminal.exists():
                continue
            value = json.loads(checkpoint.read_bytes())
            report = json.loads(terminal.read_bytes())
            window_rows.extend(value.get("window_observations", []))
            exclusions.extend(value.get("exclusions", []))
            if report["runtime_action"] not in {
                "LIVE_COLLECTION_COMPLETE",
                "DEFERRED_LOCK_HELD",
                "DEFERRED_FULL_LOCK_HANDOFF",
            }:
                raise ValueError("candidate_terminal_failure")
            seen.add(str(checkpoint))
            lane = "odds" if "odds_capture" in value["cycle_id"] else "full"
            if report["runtime_action"] == "LIVE_COLLECTION_COMPLETE":
                completed[lane].add(value["cycle_id"])
            waits.append(report["timing"]["lock_wait_seconds"])
        age_from_start = (now() - start).total_seconds()
        try:
            current = sample(plan, output, control)
        except (FileNotFoundError, KeyError):
            if age_from_start >= plan["first_index_deadline_seconds"]:
                raise
            unavailable += 1
            atomic_json(
                output / "samples" / f"{samples:06d}.json",
                {"status": "WARMUP_MISSING", "at": now().isoformat()},
            )
            samples += 1
            time.sleep(max(0, 2 - (time.monotonic() - tick)))
            continue
        request_path = scope.session / "request-count.json"
        current["logical_requests"] = (
            json.loads(request_path.read_bytes())["started"] if request_path.exists() else 0
        )
        network_path = scope.session / "network-count.json"
        current["python_network"] = (
            json.loads(network_path.read_bytes())
            if network_path.exists()
            else {"provider_started": 0, "auxiliary_started": 0, "unexpected_blocked": 0}
        )
        capture_requests = scope.session / "capture-requests.json"
        current["capture_requests"] = (
            json.loads(capture_requests.read_bytes())
            if capture_requests.exists()
            else {"browser_navigation_attempts": 0, "subresource_requests": "UNMEASURED"}
        )
        if scope.campaign:
            with scope.campaign.ledger() as ledger:
                current["campaign_logical_requests"] = ledger["logical_requests"]
            current["capture_requests"] = [json.loads(path.read_bytes()) for path in
                scope.session.glob("captures/*/capture-reservation.requests.json")]
        # Preserve the failing observation before assessing it.
        atomic_json(output / "samples" / f"{samples:06d}.json", current)
        samples += 1
        timer_accounting.observe(current)
        for lane, overhead in current["external_service_overhead_seconds"].items():
            if overhead is not None:
                external_overheads[lane].append(overhead)
        publications = []
        events = sorted(
            (evidence / "shadow_autopilot_daemon_runtime/live-publication-events").glob("*.json")
        )
        preceding = None
        matched_event_count = None
        for number, path in enumerate(events):
            event = json.loads(path.read_bytes())
            if path.name != f"{number:06d}.json" or event["previous_event_sha256"] != (
                digest(preceding) if preceding else None
            ):
                raise ValueError("publication_chain_invalid")
            if number >= event_count:
                publications.append(event)
            if event["packet_sha256"] == current["packet_sha256"]:
                matched_event_count = number + 1
            preceding = event
        if current["index_status"] == "AVAILABLE/FRESH":
            if matched_event_count is None or matched_event_count < event_count:
                raise ValueError("observed_publication_not_in_chain")
            # A writer can publish again after the native reader's snapshot.
            # Retain those later events for the next observation interval.
            publications = publications[: matched_event_count - event_count]
            if previous:
                maximum = max(maximum, assess_interval(previous, current, publications))
            previous = current
            event_count = matched_event_count
        elif age_from_start >= plan["first_index_deadline_seconds"]:
            raise ValueError("native_index_not_available")
        if any(
            current[key] in {"INVALID/INTEGRITY_FAILED", "DIVERGENT"}
            for key in ("index_status", "collector_status", "authority_status")
        ):
            raise ValueError("native_integrity_or_authority_failed")
        if (
            current["collector_status"] != "AVAILABLE/FRESH"
            or current["authority_status"] != "AVAILABLE/FRESH"
        ):
            unavailable += 1
            if age_from_start >= plan["readiness_warmup_seconds"]:
                raise ValueError("native_readiness_failed")
        atomic_json(
            output / "progress.json",
            {
                "completed_cycles": {key: len(value) for key, value in completed.items()},
                "lock_wait_seconds": waits,
                "logical_requests": current["logical_requests"],
                "python_network": current["python_network"],
                "timer_accounting": timer_accounting.summary(now()),
                "windows": window_accounting(
                    window_rows, exclusions, AttemptAllowance(scope).claims(), now()
                ),
            },
        )
        time.sleep(max(0, plan["sample_period_seconds"] - (time.monotonic() - tick)))
    if len(completed["full"]) < 3 or len(completed["odds"]) < 6 or not waits or max(waits) <= 0:
        raise ValueError("lane_progress_or_handoff_unproven")
    if not all(external_overheads.values()):
        raise ValueError("external_completion_timing_unmeasured")
    allowance = AttemptAllowance(scope)
    captures = [path for path in allowance.claims()
                if path.with_suffix(".terminal.json").exists()
                and json.loads(path.with_suffix(".terminal.json").read_bytes())["result"]
                .get("autonomous_live_odds_capture_status", {}).get("status")
                == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"]
    verified = (list((output / "capture-verifications").glob("*.json")) if scope.campaign
                else list(output.glob("capture-receipt-verification.json")))
    if len(captures) < (3 if scope.campaign else 1) or len(verified) != len(captures):
        raise ValueError("required_distinct_captures_not_demonstrated")
    atomic_json(
        output / "measurement.json",
        {
            "status": "REHEARSAL_MEASURED_NOT_RELEASED",
            "sample_count": samples,
            "unavailable_samples_including_warmup": unavailable,
            "maximum_conservative_source_age": maximum,
            "completed_cycles": {key: len(value) for key, value in completed.items()},
            "max_lock_wait_seconds": max(waits),
            "maximum_external_overhead_seconds": {
                key: max(value) for key, value in external_overheads.items()
            },
            "timer_accounting": timer_accounting.summary(end),
            "windows": window_accounting(window_rows, exclusions, allowance.claims(), end),
            "logical_requests": current["logical_requests"],
            "python_network": current["python_network"],
            "capture_requests": current["capture_requests"],
            "capture_count": len(captures),
            "throughput_validated": False,
        },
    )


def execute(plan_path, expected_digest, approval_id):
    from race_collection.freshness_attempt_reconciliation import reconcile
    from race_collection.synchronous_manual_capture import (
        acquire_collector_lock_no_steal,
        release_owned_collector_lock,
    )

    output = plan_path.parent
    plan = json.loads(plan_path.read_bytes())
    if digest(plan) != expected_digest or not approval_id:
        raise ValueError("exact_plan_approval_required")
    create_once(
        output / "started.json",
        {"approval_id": approval_id, "plan_sha256": expected_digest, "at": now().isoformat()},
    )
    start = datetime.fromisoformat(plan["starts_at"])
    if not datetime.fromisoformat(plan["admission_starts_at"]) <= now() < start:
        raise ValueError("fixed_admission_window_closed")
    verify_source_package(plan["source_root"], plan["source_identity_sha256"])
    if (
        hashlib.sha256(Path(plan["python"]).resolve().read_bytes()).hexdigest()
        != plan["python_sha256"]
    ):
        raise ValueError("python_identity_changed")
    runtime = verify_runtime(plan)
    if runtime["prefix"] != sys.prefix:
        raise ValueError("executor_must_use_pinned_python_environment")
    campaign = None
    campaign_owner = None
    if plan.get("campaign_root"):
        import fcntl
        from race_collection.freshness_campaign import Campaign
        campaign = Campaign(plan["campaign_root"])
        if digest(campaign.value) != plan["campaign_authorization_sha256"]:
            raise ValueError("campaign_authorization_changed")
        # A single owner across packages and launches, held through restoration.
        campaign_owner = (campaign.root / "owner.lock").open("a")
        fcntl.flock(campaign_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control = SystemdControl()
    snapshot(output, plan, control)
    paused = False
    scope = None
    owned = None

    def interrupted(signum, frame):
        raise InterruptedError("rehearsal_interrupted")

    previous_handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        # Pause triggers first; existing owners drain naturally.
        paused = True
        for timer in TIMERS:
            control.command("stop", timer)
        while now() < start:
            if control.idle() and not Path(plan["lock_path"]).exists():
                paused = True
                for timer in TIMERS:
                    control.command("stop", timer)
                if control.idle() and not Path(plan["lock_path"]).exists():
                    owned = acquire_collector_lock_no_steal(
                        Path(plan["lock_path"]),
                        run_id=plan["rehearsal_id"],
                        output_dir=output,
                        phase="rehearsal_admission",
                    )
                    if control.idle():
                        break
                    release_owned_collector_lock(owned)
                    owned = None
            time.sleep(1)
        if owned is None:
            raise ValueError("natural_quiescence_not_reached")
        accounting = reconcile(
            roots=plan["reconciliation_roots"],
            db_path=Path(plan.get("operational_predictions", {}).get("history_db_path", plan["db_path"])),
            source_date=start.astimezone(__import__("zoneinfo").ZoneInfo("Australia/Melbourne"))
            .date()
            .isoformat(),
            lock_path=Path(plan["lock_path"]),
            owner_run_id=plan["rehearsal_id"],
        )
        contract = {
            key: plan[key]
            for key in (
                "profile",
                "rehearsal_id",
                "starts_at",
                "ends_at",
                "cleanup_seconds",
                "lock_path",
                "db_path",
                "evidence_root",
                "max_capture_attempts",
                "max_logical_requests",
                "source_identity_sha256",
                "runtime_sha256",
            )
        }
        if plan.get("campaign_root"):
            contract.update({key: plan[key] for key in ("campaign_root", "campaign_authorization_sha256")})
        contract.update(
            schema_version="freshness_rehearsal_contract_v1",
            source_date=accounting["source_date"],
            reconciliation_sha256=digest(accounting),
        )
        if plan.get("operational_predictions"):
            contract["operational_predictions"] = plan["operational_predictions"]
        scope = FreshnessContract(contract)
        AttemptAllowance(scope).initialize(accounting)
        create_once(output / "contract.json", contract)
        for name in UNITS:
            raw = (output / "units" / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != plan["unit_sha256"][name]:
                raise ValueError("candidate_unit_changed")
            target = Path(plan["installed_dir"]) / name
            temporary = target.with_name(target.name + ".freshness-candidate.tmp")
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        control.command("daemon-reload")
        release_owned_collector_lock(owned)
        owned = None
        while now() < start:
            time.sleep(min(1, (start - now()).total_seconds()))
        if (now() - start).total_seconds() > 5:
            raise ValueError("start_dispatch_late")
        if campaign:
            campaign.begin(plan["rehearsal_id"], now=now(),
                           deadline=datetime.fromisoformat(plan["ends_at"]) + timedelta(seconds=plan["cleanup_seconds"]))
        for timer in TIMERS:
            control.command("start", timer)
        from race_collection.operational_prediction import Supervisor
        predictions = Supervisor(output, plan, scope)
        try:
            observe(output, plan, control, scope, predictions=predictions)
        finally:
            predictions.drain()
    except BaseException as error:
        atomic_json(
            output / "failure.json",
            {"type": type(error).__name__, "reason": str(error), "at": now().isoformat()},
        )
        raise
    finally:
        observation_ended_at = min(now(), datetime.fromisoformat(plan["ends_at"]))
        # A second termination signal must not interrupt exact restoration.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if scope:
                scope.stop("REHEARSAL_ENDED")
        finally:
            if owned:
                release_owned_collector_lock(owned)
            # Restore even on acquisition failure; never restart the rehearsal.
            if paused:
                restore(output, plan, control)
            if campaign:
                with campaign.ledger() as ledger:
                    begun = plan["rehearsal_id"] in ledger["launches"]
                if begun:
                    from race_collection.operational_prediction import require_completed_lifetimes
                    require_completed_lifetimes(output, campaign)
                    campaign.close(plan["rehearsal_id"], now=now())
                campaign_owner.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        if scope:
            rows, excluded = [], []
            evidence = Path(plan["evidence_root"])
            checkpoints = list(evidence.glob("shadow_autopilot_daemonization_v1_*/phase-checkpoint.json"))
            checkpoints += list((evidence / "shadow_autopilot_daemon_runtime").glob("*.live-phase-checkpoint.json"))
            cycles = {}
            for path in checkpoints:
                checkpoint = json.loads(path.read_bytes())
                cycles[checkpoint.get("cycle_id", str(path))] = checkpoint
            for checkpoint in cycles.values():
                rows.extend(checkpoint.get("window_observations", []))
                excluded.extend(checkpoint.get("exclusions", []))
            atomic_json(output / "final-window-accounting.json", window_accounting(
                rows, excluded, AttemptAllowance(scope).claims(), observation_ended_at))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    if args.restore_only:
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.plan_sha256:
            raise ValueError("plan_identity_changed")
        if plan.get("campaign_root"):
            import fcntl
            from race_collection.freshness_campaign import Campaign
            campaign = Campaign(plan["campaign_root"])
            with (campaign.root / "owner.lock").open("a") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                restore(args.plan.parent, plan, SystemdControl())
                with campaign.ledger() as ledger:
                    begun = plan["rehearsal_id"] in ledger["launches"]
                if begun:
                    from race_collection.operational_prediction import require_completed_lifetimes
                    require_completed_lifetimes(args.plan.parent, campaign)
                    campaign.close(plan["rehearsal_id"], now=now())
        else:
            restore(args.plan.parent, plan, SystemdControl())
    else:
        plan = json.loads(args.plan.read_bytes())
        if ROOT.resolve() != Path(plan["source_root"]).resolve():
            raise ValueError("executor_must_run_from_pinned_package")
        execute(args.plan, args.plan_sha256, args.approval_id)


if __name__ == "__main__":
    main()
