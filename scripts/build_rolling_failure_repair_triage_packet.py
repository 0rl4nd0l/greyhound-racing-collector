#!/usr/bin/env python3
"""Triage rolling failure official-review races into no-write repair lanes.

This helper scales the single-race official gap review across the rolling
failure queue. It reads existing lookup packets, prediction rows,
winner-only rows, and the current DB in read-only mode through the single-race
gap helper. It emits queue-level repair blocker counts only; it does not write
DB rows, labels, snapshots, manifests, datasets, models, registries, betting
decisions, or EV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_single_race_official_gap_review_packet import build_gap_review_packet
from scripts.build_single_race_official_repair_plan import (
    _dog_update_candidates,
    _metadata_update_candidate,
    _missing_runner_insert_candidates,
)


ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "rolling_failure_repair_triage_packet_v1"
STATUS_OK = "REPORT_ONLY_ROLLING_FAILURE_REPAIR_TRIAGE_PACKET"
STATUS_FAILURES = "REPORT_ONLY_ROLLING_FAILURE_REPAIR_TRIAGE_PACKET_WITH_FAILURES"

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
    "priority",
    "review_lane",
    "repair_lane",
    "race_id",
    "top1_hit",
    "top3_hit",
    "winner_rank",
    "field_scope",
    "lookup_status",
    "terminal_status_count",
    "official_runner_count",
    "db_runner_count",
    "missing_db_runner_count",
    "db_box_drift_count",
    "db_finish_drift_count",
    "changed_dog_update_candidate_count",
    "missing_runner_insert_candidate_count",
    "metadata_update_candidate_count",
    "field_size_policy_decision_required",
    "safe_to_write_now",
    "source_url",
    "recommended_next_action",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root_path = (root or ROOT).expanduser().resolve(strict=False)
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root_path / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{logical}") from exc
    return resolved, relative


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    return _repo_output_path(path, root)[1]


def _assert_output_dir_safe(output_dir: Path, root: Path | None = None) -> Path:
    resolved, relative = _repo_output_path(output_dir, root)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    return str(value or "")


def _source_url(row: Mapping[str, Any], gap_packet: Mapping[str, Any]) -> str | None:
    if row.get("source_url"):
        return str(row.get("source_url"))
    failure = _mapping(gap_packet.get("failure_review_row"))
    if failure.get("source_url"):
        return str(failure.get("source_url"))
    return None


def _changed_dog_updates(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [candidate for candidate in candidates if _list(candidate.get("changed_fields"))]


def _repair_lane(
    *,
    gap_packet: Mapping[str, Any],
    changed_dog_update_count: int,
    missing_insert_count: int,
    metadata_update_count: int,
) -> str:
    if _list(gap_packet.get("failures")):
        return "gap_review_failures_block_repair_planning"
    if missing_insert_count:
        return "missing_runner_insert_policy_required"
    if changed_dog_update_count:
        return "existing_runner_update_policy_required"
    if metadata_update_count:
        return "metadata_source_backfill_policy_required"
    return "no_row_repair_candidate_identified"


def _next_action(repair_lane: str) -> str:
    if repair_lane == "missing_runner_insert_policy_required":
        return "define_missing_runner_insert_policy_then_operator_review_exact_db_repair"
    if repair_lane == "existing_runner_update_policy_required":
        return "operator_review_existing_runner_updates_before_any_db_repair"
    if repair_lane == "metadata_source_backfill_policy_required":
        return "operator_review_metadata_source_backfill_before_any_label_preflight"
    if repair_lane == "gap_review_failures_block_repair_planning":
        return "fix_gap_review_failures_before_repair_planning"
    return "no_repair_action_from_current_evidence"


def _triage_row(
    *,
    queue_row: Mapping[str, Any],
    gap_packet: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(gap_packet.get("summary"))
    race_id = str(summary.get("race_id") or queue_row.get("race_id") or "")
    official_rows = [dict(_mapping(row)) for row in _list(gap_packet.get("official_rows"))]
    db_rows = [dict(_mapping(row)) for row in _list(gap_packet.get("db_rows"))]
    runner_rows = [dict(_mapping(row)) for row in _list(gap_packet.get("runner_review_rows"))]
    metadata = _metadata_update_candidate(
        packet=gap_packet,
        race_id=race_id,
        official_rows=official_rows,
    )
    dog_updates = _dog_update_candidates(
        race_id=race_id,
        official_rows=official_rows,
        db_rows=db_rows,
        runner_rows=runner_rows,
    )
    changed_updates = _changed_dog_updates(dog_updates)
    missing_inserts = _missing_runner_insert_candidates(
        race_id=race_id,
        official_rows=official_rows,
        db_rows=db_rows,
    )
    metadata_update_count = 1 if metadata.get("after_patch") else 0
    field_size_policy = bool(metadata.get("deferred_policy_candidates"))
    lane = _repair_lane(
        gap_packet=gap_packet,
        changed_dog_update_count=len(changed_updates),
        missing_insert_count=len(missing_inserts),
        metadata_update_count=metadata_update_count,
    )
    return {
        "priority": _safe_int(queue_row.get("priority")),
        "review_lane": queue_row.get("review_lane"),
        "repair_lane": lane,
        "race_id": race_id,
        "top1_hit": _bool_text(queue_row.get("top1_hit")),
        "top3_hit": _bool_text(queue_row.get("top3_hit")),
        "winner_rank": _safe_int(queue_row.get("winner_rank")),
        "field_scope": queue_row.get("field_scope"),
        "lookup_status": summary.get("lookup_status"),
        "terminal_status_count": _safe_int(queue_row.get("terminal_status_count")),
        "official_runner_count": summary.get("official_runner_count"),
        "db_runner_count": summary.get("db_runner_count"),
        "missing_db_runner_count": summary.get("missing_db_runner_count"),
        "db_box_drift_count": summary.get("db_box_drift_count"),
        "db_finish_drift_count": summary.get("db_finish_drift_count"),
        "changed_dog_update_candidate_count": len(changed_updates),
        "missing_runner_insert_candidate_count": len(missing_inserts),
        "metadata_update_candidate_count": metadata_update_count,
        "field_size_policy_decision_required": field_size_policy,
        "safe_to_write_now": False,
        "source_url": _source_url(queue_row, gap_packet),
        "recommended_next_action": _next_action(lane),
        "gap_review_failures": _list(gap_packet.get("failures")),
    }


def build_repair_triage_packet(
    *,
    failure_review_csv_path: Path,
    lookup_packet_paths: Sequence[Path],
    db_path: Path,
    prediction_rows_path: Path,
    winner_only_rows_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    queue_path = failure_review_csv_path.expanduser().resolve()
    rows = _load_csv_rows(queue_path)
    rows.sort(
        key=lambda row: (
            _safe_int(row.get("priority")) if _safe_int(row.get("priority")) is not None else 999,
            str(row.get("race_date") or ""),
            str(row.get("race_id") or ""),
        )
    )
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    triage_rows = []
    for row in rows:
        race_id = str(row.get("race_id") or "")
        if not race_id:
            failures.append("queue_row_missing_race_id")
            continue
        gap_packet = build_gap_review_packet(
            race_id=race_id,
            lookup_packet_paths=lookup_packet_paths,
            db_path=db_path,
            prediction_rows_path=prediction_rows_path,
            winner_only_rows_path=winner_only_rows_path,
            failure_review_csv_path=failure_review_csv_path,
        )
        if gap_packet.get("failures"):
            failures.append(f"gap_review_failures:{race_id}")
        triage_rows.append(_triage_row(queue_row=row, gap_packet=gap_packet))

    lane_counts = Counter(str(row.get("repair_lane") or "DATA_MISSING") for row in triage_rows)
    review_lane_counts = Counter(str(row.get("review_lane") or "DATA_MISSING") for row in triage_rows)
    total_changed_updates = sum(int(row["changed_dog_update_candidate_count"]) for row in triage_rows)
    total_inserts = sum(int(row["missing_runner_insert_candidate_count"]) for row in triage_rows)
    total_metadata = sum(int(row["metadata_update_candidate_count"]) for row in triage_rows)
    status = STATUS_OK if not failures else STATUS_FAILURES
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "failures": failures,
        "report_only": True,
        "safe_to_write_now": False,
        "label_write_ready": False,
        "approval_required_before_db_write": True,
        "approval_required_before_label_write": True,
        "source_evidence": {
            "failure_review_csv": str(queue_path),
            "lookup_packets": [str(path.expanduser().resolve()) for path in lookup_packet_paths],
            "db": str(db_path.expanduser().resolve()),
            "prediction_rows": str(prediction_rows_path.expanduser().resolve()),
            "winner_only_rows": str(winner_only_rows_path.expanduser().resolve()),
        },
        "summary": {
            "races_considered": len(triage_rows),
            "repair_lane_counts": dict(sorted(lane_counts.items())),
            "review_lane_counts": dict(sorted(review_lane_counts.items())),
            "total_changed_dog_update_candidates": total_changed_updates,
            "total_missing_runner_insert_candidates": total_inserts,
            "total_metadata_update_candidates": total_metadata,
            "field_size_policy_decision_required_count": sum(
                1 for row in triage_rows if row.get("field_size_policy_decision_required") is True
            ),
            "safe_to_write_now_count": 0,
            "direct_label_write_ready_count": 0,
            "recommended_next_action": (
                "resolve_missing_runner_insert_policy_before_label_expansion_or_retraining"
                if total_inserts
                else "review_existing_runner_or_metadata_repairs_before_label_preflight"
            ),
        },
        "triage_rows": triage_rows,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "schema_default_policy_required_for_inserts": total_inserts > 0,
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


def write_outputs(
    output_dir: Path,
    packet: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> None:
    output_dir = _assert_output_dir_safe(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rolling_failure_repair_triage_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "rolling_failure_repair_triage.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(packet.get("triage_rows") or [])
    _write_report(output_dir / "SUMMARY.md", packet)


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Rolling Failure Repair Triage",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Repair lane counts: `{summary.get('repair_lane_counts')}`",
        f"- Review lane counts: `{summary.get('review_lane_counts')}`",
        f"- Changed dog update candidates: `{summary.get('total_changed_dog_update_candidates')}`",
        f"- Missing runner insert candidates: `{summary.get('total_missing_runner_insert_candidates')}`",
        f"- Metadata update candidates: `{summary.get('total_metadata_update_candidates')}`",
        f"- Field-size policy decisions required: `{summary.get('field_size_policy_decision_required_count')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Recommendation",
        "",
        f"`{summary.get('recommended_next_action')}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-review-csv", required=True)
    parser.add_argument("--lookup-packet", action="append", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--winner-only-rows-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Iterable[str] | None = None, *, root: Path | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit("refusing report-only repair triage while approval flags are set:" + ",".join(active))
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_repair_triage_packet(
        failure_review_csv_path=Path(args.failure_review_csv),
        lookup_packet_paths=[Path(path) for path in args.lookup_packet],
        db_path=Path(args.db),
        prediction_rows_path=Path(args.predictions_jsonl),
        winner_only_rows_path=Path(args.winner_only_rows_jsonl),
        limit=args.limit,
    )
    write_outputs(Path(args.output_dir), packet, root=root)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
