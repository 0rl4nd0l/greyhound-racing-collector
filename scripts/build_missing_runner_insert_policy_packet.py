#!/usr/bin/env python3
"""Build a no-write missing-runner insert policy packet.

The rolling repair triage showed that reviewed official races are blocked by
missing DB runner rows. This helper turns that blocker into a concrete, reviewable
policy and exact candidate list. It reads the DB in read-only mode and writes
report artifacts only; it does not apply inserts, update metadata, write labels,
regenerate datasets, train models, update registries, enable TGR, or emit
betting/EV actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
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
    _metadata_update_candidate,
    _missing_runner_insert_candidates,
)


ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "missing_runner_insert_policy_packet_v1"
STATUS_OK = "REPORT_ONLY_MISSING_RUNNER_INSERT_POLICY_PACKET"
STATUS_FAILURES = "REPORT_ONLY_MISSING_RUNNER_INSERT_POLICY_PACKET_WITH_FAILURES"
OFFICIAL_SOURCE = "thedogs_official"

PROPOSED_INSERT_COLUMNS = [
    "race_id",
    "dog_name",
    "dog_clean_name",
    "box_number",
    "finish_position",
    "placing",
    "scraped_finish_position",
    "extraction_timestamp",
    "data_source",
]

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

CANDIDATE_FIELDS = [
    "candidate_id",
    "race_id",
    "review_lane",
    "field_scope",
    "source_url",
    "official_dog_name",
    "name_key",
    "box_number",
    "finish_position",
    "insert_columns",
    "insert_values_json",
    "duplicate_guard_sql",
    "status",
    "blockers",
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
        raise ValueError(f"output_dir_must_be_inside_repo:{resolved}") from exc
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


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _schema_info(db_path: Path) -> dict[str, Any]:
    db_resolved = db_path.expanduser().resolve()
    with _connect_read_only(db_resolved) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        dog_columns = [dict(row) for row in conn.execute("PRAGMA table_info(dog_race_data)")]
        metadata_columns = [dict(row) for row in conn.execute("PRAGMA table_info(race_metadata)")]
    return {
        "db_path": str(db_resolved),
        "quick_check": quick_check[0] if quick_check else None,
        "read_only": True,
        "query_only": True,
        "dog_race_data_columns": dog_columns,
        "race_metadata_columns": metadata_columns,
    }


def _required_insert_columns(schema: Mapping[str, Any]) -> list[str]:
    required = []
    for column in _list(schema.get("dog_race_data_columns")):
        col = _mapping(column)
        if col.get("pk"):
            continue
        if _safe_int(col.get("notnull")) == 1:
            required.append(str(col.get("name")))
    return required


def _source_url(queue_row: Mapping[str, Any], gap_packet: Mapping[str, Any]) -> str | None:
    if queue_row.get("source_url"):
        return str(queue_row.get("source_url"))
    failure = _mapping(gap_packet.get("failure_review_row"))
    if failure.get("source_url"):
        return str(failure.get("source_url"))
    return None


def _candidate_rows(
    *,
    queue_row: Mapping[str, Any],
    gap_packet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summary = _mapping(gap_packet.get("summary"))
    race_id = str(summary.get("race_id") or queue_row.get("race_id") or "")
    official_rows = [dict(_mapping(row)) for row in _list(gap_packet.get("official_rows"))]
    db_rows = [dict(_mapping(row)) for row in _list(gap_packet.get("db_rows"))]
    inserts = _missing_runner_insert_candidates(
        race_id=race_id,
        official_rows=official_rows,
        db_rows=db_rows,
    )
    source_url = _source_url(queue_row, gap_packet)
    rows = []
    for candidate in inserts:
        after = dict(_mapping(candidate.get("after")))
        values = {column: after.get(column) for column in PROPOSED_INSERT_COLUMNS}
        rows.append(
            {
                "candidate_id": f"{race_id}::{candidate.get('name_key')}",
                "race_id": race_id,
                "review_lane": queue_row.get("review_lane"),
                "field_scope": queue_row.get("field_scope"),
                "source_url": source_url,
                "official_dog_name": candidate.get("official_dog_name"),
                "name_key": candidate.get("name_key"),
                "box_number": _safe_int(after.get("box_number")),
                "finish_position": _safe_int(after.get("finish_position")),
                "insert_columns": list(PROPOSED_INSERT_COLUMNS),
                "insert_values": values,
                "duplicate_guard_sql": (
                    "SELECT 1 FROM dog_race_data WHERE race_id=? "
                    "AND (dog_name=? OR dog_clean_name=? OR CAST(box_number AS INTEGER)=?) LIMIT 1"
                ),
                "status": "BLOCKED_REQUIRES_OPERATOR_APPROVAL_AND_BACKUP",
                "blockers": [
                    "explicit_operator_approval_required",
                    "db_backup_required_before_apply",
                    "exact_candidate_allowlist_required",
                    "duplicate_guard_required_before_each_insert",
                    "post_apply_gap_review_and_label_preflight_required",
                ],
            }
        )
    return rows


def build_policy_packet(
    *,
    failure_review_csv_path: Path,
    lookup_packet_paths: Sequence[Path],
    db_path: Path,
    prediction_rows_path: Path,
    winner_only_rows_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    schema = _schema_info(db_path)
    if schema.get("quick_check") != "ok":
        failures.append("db_quick_check_failed")
    required_columns = _required_insert_columns(schema)
    missing_required = [column for column in required_columns if column not in PROPOSED_INSERT_COLUMNS]
    if missing_required:
        failures.append("proposed_insert_columns_missing_required:" + ",".join(missing_required))

    rows = _load_csv_rows(failure_review_csv_path.expanduser().resolve())
    rows.sort(
        key=lambda row: (
            _safe_int(row.get("priority")) if _safe_int(row.get("priority")) is not None else 999,
            str(row.get("race_date") or ""),
            str(row.get("race_id") or ""),
        )
    )
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    candidate_rows: list[dict[str, Any]] = []
    metadata_policy_rows: list[dict[str, Any]] = []
    race_counts: Counter[str] = Counter()
    review_lane_counts: Counter[str] = Counter()
    field_scope_counts: Counter[str] = Counter()
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
        rows_for_race = _candidate_rows(queue_row=row, gap_packet=gap_packet)
        candidate_rows.extend(rows_for_race)
        if rows_for_race:
            race_counts[race_id] += len(rows_for_race)
            review_lane_counts[str(row.get("review_lane") or "DATA_MISSING")] += len(rows_for_race)
            field_scope_counts[str(row.get("field_scope") or "DATA_MISSING")] += len(rows_for_race)
        metadata = _metadata_update_candidate(
            packet=gap_packet,
            race_id=race_id,
            official_rows=[dict(_mapping(item)) for item in _list(gap_packet.get("official_rows"))],
        )
        if metadata.get("after_patch") or metadata.get("deferred_policy_candidates"):
            metadata_policy_rows.append(
                {
                    "race_id": race_id,
                    "after_patch": metadata.get("after_patch"),
                    "deferred_policy_candidates": metadata.get("deferred_policy_candidates"),
                    "status": "REQUIRES_SEPARATE_METADATA_POLICY_AND_APPROVAL",
                }
            )

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
            "failure_review_csv": str(failure_review_csv_path.expanduser().resolve()),
            "lookup_packets": [str(path.expanduser().resolve()) for path in lookup_packet_paths],
            "db": str(db_path.expanduser().resolve()),
            "prediction_rows": str(prediction_rows_path.expanduser().resolve()),
            "winner_only_rows": str(winner_only_rows_path.expanduser().resolve()),
        },
        "db_schema": schema,
        "insert_policy": {
            "policy_status": "PROPOSED_REQUIRES_OPERATOR_APPROVAL",
            "source": OFFICIAL_SOURCE,
            "insert_only_official_finishers_missing_from_db": True,
            "do_not_synthesize_history_form_odds_times_or_ids": True,
            "proposed_insert_columns": list(PROPOSED_INSERT_COLUMNS),
            "required_notnull_columns": required_columns,
            "missing_required_columns": missing_required,
            "nullable_columns_intentionally_unset": [
                column["name"]
                for column in _list(schema.get("dog_race_data_columns"))
                if column.get("name") not in PROPOSED_INSERT_COLUMNS and not column.get("pk")
            ],
            "duplicate_guard": (
                "race_id plus official name/clean name/box must be absent immediately before insert"
            ),
            "post_apply_required_checks": [
                "rerun_single_race_gap_review_or_queue_triage",
                "rerun_official_label_preflight",
                "only_then_consider_label_expansion_or_retraining",
            ],
        },
        "summary": {
            "races_considered": len(rows),
            "races_with_missing_runner_insert_candidates": len(race_counts),
            "missing_runner_insert_candidate_count": len(candidate_rows),
            "review_lane_candidate_counts": dict(sorted(review_lane_counts.items())),
            "field_scope_candidate_counts": dict(sorted(field_scope_counts.items())),
            "metadata_policy_race_count": len(metadata_policy_rows),
            "safe_to_write_now_count": 0,
            "recommended_next_action": (
                "operator_review_missing_runner_insert_policy_and_candidate_allowlist_before_any_db_repair"
            ),
        },
        "candidate_rows": candidate_rows,
        "metadata_policy_rows": metadata_policy_rows,
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
    (output_dir / "missing_runner_insert_policy_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "missing_runner_insert_candidates.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in packet.get("candidate_rows") or []:
            item = dict(_mapping(row))
            item["insert_columns"] = "|".join(_list(item.get("insert_columns")))
            item["insert_values_json"] = _json_cell(item.pop("insert_values", {}))
            item["blockers"] = "|".join(_list(item.get("blockers")))
            writer.writerow(item)
    _write_report(output_dir / "SUMMARY.md", packet)


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    policy = _mapping(packet.get("insert_policy"))
    lines = [
        "# Missing Runner Insert Policy Packet",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Races with missing-runner candidates: `{summary.get('races_with_missing_runner_insert_candidates')}`",
        f"- Missing-runner insert candidates: `{summary.get('missing_runner_insert_candidate_count')}`",
        f"- Candidate counts by review lane: `{summary.get('review_lane_candidate_counts')}`",
        f"- Candidate counts by field scope: `{summary.get('field_scope_candidate_counts')}`",
        f"- Metadata policy races: `{summary.get('metadata_policy_race_count')}`",
        f"- Proposed insert columns: `{policy.get('proposed_insert_columns')}`",
        f"- Intentionally unset nullable columns: `{policy.get('nullable_columns_intentionally_unset')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Gate",
        "",
        "This is an approval packet only. Any apply step still requires explicit operator approval, a DB backup, exact candidate allowlist, duplicate guard immediately before each insert, and post-apply gap review plus label preflight before label expansion or retraining.",
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


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit("refusing report-only insert policy while approval flags are set:" + ",".join(active))
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_policy_packet(
        failure_review_csv_path=Path(args.failure_review_csv),
        lookup_packet_paths=[Path(path) for path in args.lookup_packet],
        db_path=Path(args.db),
        prediction_rows_path=Path(args.predictions_jsonl),
        winner_only_rows_path=Path(args.winner_only_rows_jsonl),
        limit=args.limit,
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
