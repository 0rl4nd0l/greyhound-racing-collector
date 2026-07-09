#!/usr/bin/env python3
"""Forecast label-gate status after proposed official DB repairs.

This report-only helper consumes the rolling repair triage packet and the
missing-runner insert policy packet. It answers what would be true after the
proposed row updates/inserts/metadata source repairs, without touching the DB:

- whether the official runner set would be complete for each race,
- whether terminal-status policy would still block label expansion,
- whether direct official-reverify label preflight would still be blocked by
  the existing-result-row contract.

It writes artifacts only and performs no DB writes, label writes, dataset
regeneration, model training, registry mutation, TGR enablement, betting, or EV
actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "post_repair_label_gate_forecast_packet_v1"
TRIAGE_SCHEMA_VERSION = "rolling_failure_repair_triage_packet_v1"
POLICY_SCHEMA_VERSION = "missing_runner_insert_policy_packet_v1"
STATUS_OK = "REPORT_ONLY_POST_REPAIR_LABEL_GATE_FORECAST_PACKET"
STATUS_FAILURES = "REPORT_ONLY_POST_REPAIR_LABEL_GATE_FORECAST_PACKET_WITH_FAILURES"

WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "dataset_regeneration": False,
    "model_training": False,
    "model_persistence": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}

FORBIDDEN_APPROVAL_ENV_VARS = (
    "APPROVE_RESULT_LABEL_WRITE",
    "APPROVE_GREYHOUND_DB_WRITE",
    "GREYHOUND_ALLOW_DB_WRITE",
    "GREYHOUND_ALLOW_TGR",
)

CSV_FIELDS = [
    "race_id",
    "review_lane",
    "forecast_gate",
    "terminal_status_count",
    "field_scope",
    "official_runner_count",
    "current_db_runner_count",
    "missing_runner_insert_candidate_count",
    "post_repair_runner_count",
    "runner_set_complete_after_proposed_repair",
    "changed_dog_update_candidate_count",
    "metadata_update_candidate_count",
    "direct_label_preflight_ready_forecast",
    "remaining_blockers",
    "recommended_next_action",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root_path = (root or ROOT).expanduser().resolve(strict=False)
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root_path / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{resolved}") from exc
    return resolved, relative


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    return _repo_output_path(path, root)[1]


def _assert_output_dir_safe(output_dir: Path, root: Path | None = None) -> Path:
    resolved, relative = _repo_output_path(output_dir, root)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _validate_source_packet(
    *,
    packet: Mapping[str, Any],
    expected_schema: str,
    packet_name: str,
    failures: list[str],
) -> None:
    if packet.get("schema_version") != expected_schema:
        failures.append(f"{packet_name}_schema_mismatch")
    writes = _mapping(packet.get("writes_performed"))
    for key, value in writes.items():
        if value is not False:
            failures.append(f"{packet_name}_write_flag_true:{key}")
    if packet.get("report_only") is not True:
        failures.append(f"{packet_name}_not_report_only")


def _insert_counts_by_race(policy_packet: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in _list(policy_packet.get("candidate_rows")):
        race_id = str(_mapping(row).get("race_id") or "")
        if race_id:
            counts[race_id] += 1
    return counts


def _metadata_policy_by_race(policy_packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in _list(policy_packet.get("metadata_policy_rows")):
        row_map = _mapping(row)
        race_id = str(row_map.get("race_id") or "")
        if race_id:
            result[race_id] = row_map
    return result


def _forecast_for_row(
    *,
    triage_row: Mapping[str, Any],
    insert_count: int,
    metadata_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    race_id = str(triage_row.get("race_id") or "")
    official_count = _safe_int(triage_row.get("official_runner_count"))
    db_count = _safe_int(triage_row.get("db_runner_count"))
    terminal_count = _safe_int(triage_row.get("terminal_status_count"))
    changed_updates = _safe_int(triage_row.get("changed_dog_update_candidate_count"))
    metadata_updates = _safe_int(triage_row.get("metadata_update_candidate_count"))
    post_repair_count = db_count + insert_count
    runner_complete = official_count > 0 and post_repair_count == official_count
    remaining_blockers = [
        "explicit_operator_approval_required",
        "db_backup_required_before_apply",
        "exact_candidate_allowlist_required",
        "post_apply_gap_review_required",
        "post_apply_label_preflight_required",
        "direct_label_preflight_still_blocks_existing_result_rows",
    ]
    if terminal_count > 0:
        remaining_blockers.append("terminal_status_policy_required")
    if not runner_complete:
        remaining_blockers.append("runner_set_still_incomplete_after_proposed_repair")
    if metadata_policy and metadata_policy.get("deferred_policy_candidates"):
        remaining_blockers.append("field_size_metadata_policy_required")

    if runner_complete and terminal_count == 0:
        gate = "POST_REPAIR_RUNNER_SET_COMPLETE_TERMINAL_FREE_RECHECK_REQUIRED"
        next_action = "after_approved_db_repair_rerun_gap_review_then_label_preflight"
    elif runner_complete:
        gate = "POST_REPAIR_RUNNER_SET_COMPLETE_TERMINAL_POLICY_REQUIRED"
        next_action = "resolve_terminal_status_policy_before_label_expansion"
    else:
        gate = "POST_REPAIR_FORECAST_STILL_INCOMPLETE"
        next_action = "revise_repair_allowlist_before_any_apply"

    return {
        "race_id": race_id,
        "review_lane": triage_row.get("review_lane"),
        "forecast_gate": gate,
        "terminal_status_count": terminal_count,
        "field_scope": triage_row.get("field_scope"),
        "official_runner_count": official_count,
        "current_db_runner_count": db_count,
        "missing_runner_insert_candidate_count": insert_count,
        "post_repair_runner_count": post_repair_count,
        "runner_set_complete_after_proposed_repair": runner_complete,
        "changed_dog_update_candidate_count": changed_updates,
        "metadata_update_candidate_count": metadata_updates,
        "direct_label_preflight_ready_forecast": False,
        "remaining_blockers": sorted(set(remaining_blockers)),
        "recommended_next_action": next_action,
    }


def build_forecast_packet(
    *,
    repair_triage_packet_path: Path,
    insert_policy_packet_path: Path,
) -> dict[str, Any]:
    triage_resolved = repair_triage_packet_path.expanduser().resolve()
    policy_resolved = insert_policy_packet_path.expanduser().resolve()
    triage_packet = _load_json(triage_resolved)
    policy_packet = _load_json(policy_resolved)
    failures: list[str] = []
    _validate_source_packet(
        packet=triage_packet,
        expected_schema=TRIAGE_SCHEMA_VERSION,
        packet_name="repair_triage_packet",
        failures=failures,
    )
    _validate_source_packet(
        packet=policy_packet,
        expected_schema=POLICY_SCHEMA_VERSION,
        packet_name="insert_policy_packet",
        failures=failures,
    )
    insert_counts = _insert_counts_by_race(policy_packet)
    metadata_by_race = _metadata_policy_by_race(policy_packet)
    forecast_rows = [
        _forecast_for_row(
            triage_row=_mapping(row),
            insert_count=insert_counts.get(str(_mapping(row).get("race_id") or ""), 0),
            metadata_policy=metadata_by_race.get(str(_mapping(row).get("race_id") or "")),
        )
        for row in _list(triage_packet.get("triage_rows"))
    ]

    forecast_gate_counts = Counter(str(row.get("forecast_gate")) for row in forecast_rows)
    review_lane_counts = Counter(str(row.get("review_lane")) for row in forecast_rows)
    blocker_counts: Counter[str] = Counter()
    for row in forecast_rows:
        for blocker in _list(row.get("remaining_blockers")):
            blocker_counts[str(blocker)] += 1
    status = STATUS_OK if not failures else STATUS_FAILURES
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "failures": failures,
        "report_only": True,
        "safe_to_write_now": False,
        "label_write_ready": False,
        "direct_label_preflight_ready_forecast": False,
        "source_evidence": {
            "repair_triage_packet": str(triage_resolved),
            "insert_policy_packet": str(policy_resolved),
        },
        "summary": {
            "races_considered": len(forecast_rows),
            "forecast_gate_counts": dict(sorted(forecast_gate_counts.items())),
            "review_lane_counts": dict(sorted(review_lane_counts.items())),
            "runner_set_complete_after_proposed_repair_count": sum(
                1 for row in forecast_rows if row.get("runner_set_complete_after_proposed_repair") is True
            ),
            "terminal_policy_required_count": sum(
                1 for row in forecast_rows if _safe_int(row.get("terminal_status_count")) > 0
            ),
            "direct_label_preflight_ready_forecast_count": 0,
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "recommended_next_action": (
                "after_explicit_db_repair_approval_apply_allowlist_then_rerun_gap_review_and_label_preflight"
            ),
        },
        "forecast_rows": forecast_rows,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "exact_candidate_allowlist_required": True,
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": [
            "db_write",
            "label_write",
            "metadata_write",
            "dog_row_insert",
            "field_size_update",
            "dataset_regeneration",
            "model_training_or_promotion",
            "registry_update",
            "enable_tgr",
            "betting_or_ev_action",
        ],
    }


def write_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "post_repair_label_gate_forecast_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "post_repair_label_gate_forecast.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in packet.get("forecast_rows") or []:
            output_row = dict(_mapping(row))
            output_row["remaining_blockers"] = "|".join(_list(output_row.get("remaining_blockers")))
            writer.writerow(output_row)
    _write_report(output_dir / "SUMMARY.md", packet)


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Post-Repair Label Gate Forecast",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Forecast gate counts: `{summary.get('forecast_gate_counts')}`",
        f"- Runner-set complete after proposed repair count: `{summary.get('runner_set_complete_after_proposed_repair_count')}`",
        f"- Terminal policy required count: `{summary.get('terminal_policy_required_count')}`",
        f"- Direct label-preflight ready forecast count: `{summary.get('direct_label_preflight_ready_forecast_count')}`",
        f"- Blocker counts: `{summary.get('blocker_counts')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Interpretation",
        "",
        "The proposed repair allowlist is forecast to complete the official runner set only for the count shown above; the remaining terminal/exclusion-scope races still need a revised allowlist or separate terminal policy. No race is direct-label-write-ready from this forecast. The existing preflight contract still requires an approved DB repair followed by fresh gap review and label preflight.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-triage-packet", required=True)
    parser.add_argument("--insert-policy-packet", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit("refusing report-only forecast while approval flags are set:" + ",".join(active))
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_forecast_packet(
        repair_triage_packet_path=Path(args.repair_triage_packet),
        insert_policy_packet_path=Path(args.insert_policy_packet),
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
