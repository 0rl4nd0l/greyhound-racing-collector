"""Operational-only rehearsal measurement using the native R3 readers.

No model, corpus, outcome or prediction endpoints are constructed or read.
"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from race_collection.live_phase_checkpoint import atomic_json
from src.operator_ui.foundation import (
    JsonSerializationPolicy,
    JsonSource,
    OperatorEvidenceReader,
    SourceConfig,
    TimestampSyntax,
)
from src.operator_ui.live_adapters import InstalledUnits, LiveEvidenceAdapters, UpcomingRaceSource


def native_observation(*, now, paths, units, evidence_root, index_path, authority, output):
    """Authority is the pinned candidate and actual installed units, never old R3 binding."""
    if any(
        hashlib.sha256(getattr(units, name)).hexdigest() != authority["unit_sha256"][name]
        for name in ("full_timer", "full_service", "odds_timer", "odds_service")
    ):
        raise ValueError("rehearsal_unit_authority_changed")
    if units.working_directory != authority["source_root"]:
        raise ValueError("rehearsal_source_authority_changed")
    deployment = {
        "schema_version": "operator_ui_deployment_manifest_v1",
        "generated_at": now.isoformat(),
        "source_commit": authority["commit"],
        "deployed_commit": authority["commit"],
        "source_tree": authority["tree"],
        "deployed_tree": authority["tree"],
        "working_directory": authority["source_root"],
        "installed_unit_sha256": authority["unit_sha256"],
    }
    manifest = output / "candidate-deployment-observation.json"
    atomic_json(manifest, deployment)
    paths = {**paths, "deployment_manifest": manifest}
    sources = {}
    for key, path in paths.items():
        path = Path(path)
        missing = False
        try:
            value = json.loads(path.read_bytes())
        except FileNotFoundError:
            value = {}
            missing = True
        schema = value.get("schema_version")
        policy = (
            "P-DEPLOY-60"
            if key == "deployment_manifest"
            else (
                "P-COLLECTOR-ODDS-DYNAMIC" if key.startswith("odds") else "P-COLLECTOR-FULL-DYNAMIC"
            )
        )
        # Let the native reader report absent adapter evidence as missing. An
        # empty schema with a declared timestamp fails configuration validation.
        time_field = (
            None
            if missing or key == "full_state"
            else ("updated_at" if key == "odds_state" else "generated_at")
        )
        sources[key] = SourceConfig(
            locator=path,
            allowlisted_root=evidence_root if path.is_relative_to(evidence_root) else output,
            source_kind="producer_report",
            source_identity=schema or "shadow_autopilot_refresh_report",
            source_locator=f"operator_ui.{key}",
            policy=policy,
            supported_claim="Native operational rehearsal evidence only.",
            json=JsonSource(
                "schema_version" if schema else None,
                schema,
                tuple(value),
                time_field,
                identity_fields=("schema_version",) if schema else (),
                max_items=100000,
                serialization_policy=JsonSerializationPolicy.PRODUCER_PRETTY_SORTED,
                timestamp_syntax=TimestampSyntax.AWARE_ISO8601,
            ),
            max_bytes=512 * 1024,
        )
    adapter = LiveEvidenceAdapters(
        OperatorEvidenceReader(sources, clock=lambda: now),
        units=units,
        upcoming_races=UpcomingRaceSource(index_path, evidence_root, timeout_seconds=1),
    )
    upcoming, collector, system = adapter.upcoming(now), adapter.collector(now), adapter.system(now)
    return {
        "observed_at": now.isoformat(),
        "index_status": upcoming.evidence.status,
        "source_at": upcoming.evidence.source_at,
        "source_age_seconds": upcoming.evidence.age_seconds,
        "packet_sha256": upcoming.evidence.content_sha256,
        "index_reference_hashes": dict(upcoming.evidence.reference_hashes),
        "race_count": len(upcoming.data.get("races", [])),
        "collector_status": collector.evidence.status,
        "lanes": collector.data.get("lanes", []),
        "authority_status": system.evidence.status,
    }


def assess_interval(previous, current, publications):
    elapsed = current["monotonic_end"] - previous["monotonic_start"]
    if elapsed < 0 or elapsed > 5:
        raise ValueError("sample_gap_unverified")
    wall_elapsed = (
        datetime.fromisoformat(current["read_end"]) - datetime.fromisoformat(previous["read_start"])
    ).total_seconds()
    if abs(wall_elapsed - elapsed) > 0.25:
        raise ValueError("clock_discontinuity")
    if (
        previous["index_status"] != "AVAILABLE/FRESH"
        or current["index_status"] != "AVAILABLE/FRESH"
    ):
        raise ValueError("native_index_unavailable")
    times = [datetime.fromisoformat(previous["source_at"])]
    hashes = [previous["packet_sha256"]]
    for publication in publications:
        times.append(datetime.fromisoformat(publication["source_generated_at"]))
        hashes.append(publication["packet_sha256"])
    if any(later < earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("publication_source_regression")
    if hashes[-1] != current["packet_sha256"]:
        raise ValueError("publication_chain_incomplete")
    upper = previous["source_age_seconds"] + elapsed
    if upper > 270:
        raise ValueError("freshness_target_exceeded")
    return upper


def completed_service_overhead(status, report, lifecycle=None):
    """Include systemd dispatch, interpreter startup, final stdout and process exit."""
    started = int(status.get("ExecMainStartTimestampMonotonic") or 0) / 1e6
    ended = int(status.get("ExecMainExitTimestampMonotonic") or 0) / 1e6
    if status["ActiveState"] not in {"inactive", "failed"} or ended < started or not started:
        return None
    timing = report["timing"]
    if timing.get("service_invocation_id"):
        if lifecycle is None:
            return None
        if (
            lifecycle.get("invocation_id") != timing["service_invocation_id"]
            or status.get("InvocationID") != lifecycle["invocation_id"]
            or int(status.get("ExecMainPID") or 0) != lifecycle.get("wrapper_pid")
            or timing.get("process_pid") != lifecycle.get("child_pid")
        ):
            raise ValueError("service_lifecycle_identity_mismatch")
        if (
            lifecycle.get("children_reaped") is not True
            or not lifecycle["process_start_lower_bound"]
            <= timing["process_started_monotonic"]
            <= lifecycle["completed_monotonic"]
            <= ended
        ):
            raise ValueError("service_lifecycle_incomplete")
        started = min(started, lifecycle["process_start_lower_bound"])
    elif not started <= timing["process_started_monotonic"] <= ended:
        return None  # An old report cannot authenticate a new activation.
    overhead = ended - started - timing["phase_seconds"] - timing["lock_wait_seconds"]
    if overhead < 0 or overhead > 10:
        raise ValueError("external_service_overhead_exceeded")
    return overhead


def timer_monotonic_seconds(value):
    """systemctl renders Timer's timespan property, unlike service timestamps."""
    text = str(value or "0").strip()
    if text.isdigit():
        return int(text) / 1e6
    units = {
        "w": 604800,
        "d": 86400,
        "h": 3600,
        "min": 60,
        "s": 1,
        "ms": 0.001,
        "us": 0.000001,
        "µs": 0.000001,
    }
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)(min|ms|us|µs|w|d|h|s)", text))
    if "".join(match.group(0) for match in matches) != "".join(text.split()):
        raise ValueError("timer_monotonic_timestamp_unreadable")
    return sum(float(match.group(1)) * units[match.group(2)] for match in matches)


class TimerAccounting:
    """Reconcile actual trigger/start observations without assuming queued ticks."""

    def __init__(self, starts_at):
        self.start = starts_at
        self.began = None
        self.triggers = {"full": set(), "odds": set()}
        self.activations = {"full": {}, "odds": {}}

    def observe(self, sample):
        if self.began is None:
            self.began = (
                sample["monotonic_start"]
                - (datetime.fromisoformat(sample["read_start"]) - self.start).total_seconds()
            )
        for lane in self.triggers:
            timer = sample["timer_status"][lane]
            triggered = timer_monotonic_seconds(timer.get("LastTriggerUSecMonotonic"))
            if triggered >= self.began:
                self.triggers[lane].add(triggered)
            state = sample["unit_status"][lane]
            started = int(state.get("ExecMainStartTimestampMonotonic") or 0) / 1e6
            if started < self.began:
                continue
            candidates = [value for value in self.triggers[lane] if value <= started]
            if not candidates:
                raise ValueError("timer_dispatch_unattributed")
            record = self.activations[lane].setdefault(
                started,
                {
                    "started_monotonic": started,
                    "triggered_monotonic": max(candidates),
                    "dispatch_seconds": started - max(candidates),
                },
            )
            if lane == "odds":
                triggered_at = self.start + timedelta(
                    seconds=record["triggered_monotonic"] - self.began
                )
                record["calendar_delay_seconds"] = (
                    triggered_at.second + triggered_at.microsecond / 1e6
                )
            if record["dispatch_seconds"] > 10:
                raise ValueError("timer_dispatch_budget_exceeded")
            ended = int(state.get("ExecMainExitTimestampMonotonic") or 0) / 1e6
            if ended >= started and state["ActiveState"] in {"inactive", "failed"}:
                record["ended_monotonic"] = ended
            elif state["ActiveState"] in {"active", "activating"}:
                record["observed_active_through"] = sample["monotonic_start"]
            overhead = sample["external_service_overhead_seconds"].get(lane)
            if overhead is not None:
                record["complete_overhead_seconds"] = (
                    overhead
                    + record["dispatch_seconds"]
                    + max(0, record.get("calendar_delay_seconds", 0) - 15)
                )
                if record["complete_overhead_seconds"] > 10:
                    raise ValueError("dispatch_plus_process_overhead_exceeded")

    def summary(self, at):
        slots = []
        tick = self.start.replace(second=0, microsecond=0)
        if tick < self.start:
            tick += timedelta(minutes=1)
        while tick < at:
            monotonic_tick = self.began + (tick - self.start).total_seconds()
            starts = [
                row
                for row in self.activations["odds"].values()
                if monotonic_tick <= row["triggered_monotonic"] < monotonic_tick + 60
            ]
            active = any(
                row["started_monotonic"] <= monotonic_tick
                and row.get(
                    "ended_monotonic", row.get("observed_active_through", row["started_monotonic"])
                )
                > monotonic_tick + 15
                for row in self.activations["odds"].values()
            )
            triggers = [
                value
                for value in self.triggers["odds"]
                if monotonic_tick <= value < monotonic_tick + 60
            ]
            ignored = any(
                row["started_monotonic"]
                <= trigger
                < row.get(
                    "ended_monotonic", row.get("observed_active_through", row["started_monotonic"])
                )
                for row in self.activations["odds"].values()
                for trigger in triggers
            )
            status = (
                "ACTIVATION_OBSERVED"
                if starts
                else (
                    "TRIGGER_WHILE_ACTIVE_NO_NEW_START"
                    if ignored
                    else (
                        "ACTIVE_ACROSS_NOMINAL_WINDOW"
                        if active
                        else "TRIGGER_WITHOUT_OBSERVED_START" if triggers else "NO_TRIGGER_OBSERVED"
                    )
                )
            )
            slots.append({"nominal_at": tick.isoformat(), "status": status})
            tick += timedelta(minutes=1)
        return {
            "activations": {lane: list(rows.values()) for lane, rows in self.activations.items()},
            "trigger_monotonic_seconds": {
                lane: sorted(values) for lane, values in self.triggers.items()
            },
            "odds_calendar_slots": slots,
            "full_timer_semantics": "Both OnActiveSec and OnUnitInactiveSec; exact next-due fields retained in samples",
            "short_unobserved_activations": "Cannot infer absence from sampling; NO_TRIGGER_OBSERVED remains unverified",
        }
