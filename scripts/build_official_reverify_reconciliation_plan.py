#!/usr/bin/env python3
"""Build a report-only reconciliation plan for recovered official labels.

The preflight proves direct label writes are unsafe. This planner classifies
blocked candidates into lanes so the next work can be bounded and evidence-led.
It performs no writes and does not approve any label mutation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "official_reverify_reconciliation_plan_v1"
PREFLIGHT_SCHEMA_VERSION = "official_reverify_label_preflight_v1"

WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "promotion": False,
    "betting_decision": False,
}

LANE_ORDER = [
    "existing_rows_exact_box_set_update_rehearsal_candidate",
    "existing_rows_exact_box_set_metadata_complete_review",
    "existing_rows_box_set_mismatch",
    "metadata_and_dog_rows_missing",
    "direct_preflight_ready_but_still_requires_approval",
    "other_blocked",
]

LANE_ACTIONS = {
    "existing_rows_exact_box_set_update_rehearsal_candidate": (
        "build_no_write_update_rehearsal_for_exact_box_set_existing_rows"
    ),
    "existing_rows_exact_box_set_metadata_complete_review": (
        "compare_existing_winner_and_positions_before_noop_or_correction_policy"
    ),
    "existing_rows_box_set_mismatch": (
        "manual_or_source-backed_runner_reconciliation_required"
    ),
    "metadata_and_dog_rows_missing": (
        "metadata_and_runner_row_creation_policy_required_before_labels"
    ),
    "direct_preflight_ready_but_still_requires_approval": (
        "hold_for_explicit_approval_after_human_sample"
    ),
    "other_blocked": "manual_review_required",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _classify_lane(candidate: Mapping[str, Any]) -> str:
    blockers = {str(item) for item in _list(candidate.get("blockers"))}
    alignment = _mapping(candidate.get("row_alignment"))
    box_set_matches = alignment.get("box_set_matches_official") is True

    if candidate.get("preflight_status") == "PREFLIGHT_READY":
        return "direct_preflight_ready_but_still_requires_approval"
    if {"race_metadata_missing", "db_dog_rows_missing"}.issubset(blockers):
        return "metadata_and_dog_rows_missing"
    if "db_has_existing_result_rows" in blockers and not box_set_matches:
        return "existing_rows_box_set_mismatch"
    if (
        "db_has_existing_result_rows" in blockers
        and box_set_matches
        and (
            "race_metadata_not_pending" in blockers
            or "race_metadata_winner_present" in blockers
            or "race_metadata_winner_source_present" in blockers
        )
    ):
        return "existing_rows_exact_box_set_metadata_complete_review"
    if "db_has_existing_result_rows" in blockers and box_set_matches:
        return "existing_rows_exact_box_set_update_rehearsal_candidate"
    return "other_blocked"


def _validate_preflight(packet: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        failures.append("preflight_schema_mismatch")
    if packet.get("status") not in {"PREFLIGHT_READY_WITH_BLOCKERS", "PREFLIGHT_READY"}:
        failures.append("preflight_status_unexpected")
    writes = _mapping(packet.get("writes_performed"))
    forbidden = [key for key, value in writes.items() if value is not False]
    if forbidden:
        failures.append("preflight_has_write_flags:" + ",".join(sorted(forbidden)))
    return failures


def _candidate_digest(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "legacy_race_id": candidate.get("legacy_race_id"),
        "resolved_db_race_id": candidate.get("resolved_db_race_id"),
        "lookup_key": candidate.get("lookup_key"),
        "blockers": candidate.get("blockers") or [],
        "row_alignment": candidate.get("row_alignment") or {},
        "dog_race_data_state": candidate.get("dog_race_data_state") or {},
    }


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Official Reverify Reconciliation Plan",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot mutations, manifest mutations, model training, registry mutations, promotions, betting decisions, or EV claims were performed.",
        "",
        "## Summary",
        "",
        f"- Safe to write now: `{summary.get('safe_to_write_now_count')}`",
        f"- First executable lane: `{summary.get('first_executable_lane')}`",
        f"- Lane counts: `{summary.get('lane_counts')}`",
        "",
        "## Lanes",
        "",
    ]
    for lane_name in LANE_ORDER:
        lane = _mapping(_mapping(packet.get("lanes")).get(lane_name))
        if not lane:
            continue
        lines.append(
            f"- `{lane_name}`: `{lane.get('count')}`; action `{lane.get('recommended_action')}`"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Do not write labels. Start with the exact-box-set update rehearsal lane, then separately design row/metadata reconstruction for the larger blocked lanes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_reconciliation_plan(
    *,
    preflight_packet_path: Path,
    sample_limit: int = 10,
) -> dict[str, Any]:
    preflight_resolved = preflight_packet_path.expanduser().resolve()
    preflight = _load_json(preflight_resolved)
    failures = _validate_preflight(preflight)

    lane_candidates: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANE_ORDER}
    lane_counts: Counter[str] = Counter()
    for candidate in _list(preflight.get("candidate_preflight")):
        if not isinstance(candidate, Mapping):
            continue
        lane = _classify_lane(candidate)
        lane_counts[lane] += 1
        if len(lane_candidates[lane]) < sample_limit:
            lane_candidates[lane].append(_candidate_digest(candidate))

    lanes = {
        lane: {
            "count": lane_counts.get(lane, 0),
            "recommended_action": LANE_ACTIONS[lane],
            "sample_candidates": lane_candidates[lane],
        }
        for lane in LANE_ORDER
        if lane_counts.get(lane, 0) > 0
    }
    first_executable = next(
        (lane for lane in LANE_ORDER if lane_counts.get(lane, 0) > 0),
        None,
    )
    direct_ready = lane_counts.get("direct_preflight_ready_but_still_requires_approval", 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "NOT_READY" if failures else "REPORT_ONLY_RECONCILIATION_REQUIRED",
        "failures": failures,
        "source_evidence": {"preflight_packet": str(preflight_resolved)},
        "summary": {
            "safe_to_write_now_count": 0,
            "direct_preflight_ready_requires_approval_count": direct_ready,
            "first_executable_lane": first_executable,
            "lane_counts": dict(sorted(lane_counts.items())),
        },
        "lanes": lanes,
        "approval_note": {
            "operator_may_approve_future_write": True,
            "approval_not_used_here": True,
            "reason": "no lane is safe for immediate label write from this plan",
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_new_rehearsal_packet": [
            "label_write",
            "metadata_write",
            "dog_row_insert",
            "model_training_or_promotion",
            "betting_decision",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-packet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sample-limit", type=int, default=10)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    if str(os.environ.get("APPROVE_RESULT_LABEL_WRITE") or "").strip():
        raise SystemExit("refusing reconciliation plan while APPROVE_RESULT_LABEL_WRITE is set")
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_reconciliation_plan(
        preflight_packet_path=Path(args.preflight_packet),
        sample_limit=max(0, int(args.sample_limit)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report, packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
