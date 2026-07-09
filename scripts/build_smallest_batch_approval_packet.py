#!/usr/bin/env python3
"""Build a report-only approval packet for the smallest repair batch.

The post-update label-gate forecast identifies a smallest candidate race where
existing-row updates should clear a duplicate guard for one deferred insert.
This helper freezes that exact batch into an operator-review packet with fresh
read-only DB guards. It never writes DB rows, labels, snapshots, manifests,
datasets, models, registries, TGR settings, betting decisions, EV actions, or
official fetches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_non_terminal_repair_apply_manifest_forecast import (  # noqa: E402
    WRITES_PERFORMED,
    _assert_output_dir_safe,
    _connect_read_only,
    _list,
    _load_json,
    _mapping,
    _safe_int,
    utc_now,
)
from scripts.build_non_terminal_update_policy_manifest import (  # noqa: E402
    SCHEMA_VERSION as UPDATE_POLICY_SCHEMA_VERSION,
)
from scripts.build_post_update_label_gate_forecast import (  # noqa: E402
    GATE_BATCH_CANDIDATE,
    SCHEMA_VERSION as POST_UPDATE_FORECAST_SCHEMA_VERSION,
    _append_simulated_insert,
    _apply_updates,
    _duplicate_hits,
    _insert_box,
    _insert_finish,
    _insert_values,
    _path_from_source,
    _race_rows,
    _validate_packet,
)
from scripts.build_terminal_scope_reconciliation_packet import (  # noqa: E402
    FORBIDDEN_APPROVAL_ENV_VARS,
    _name_key,
)


SCHEMA_VERSION = "smallest_batch_approval_packet_v1"
STATUS_OK = "REPORT_ONLY_SMALLEST_BATCH_APPROVAL_PACKET"
STATUS_FAILURES = "REPORT_ONLY_SMALLEST_BATCH_APPROVAL_PACKET_WITH_FAILURES"

UPDATE_GUARD_MATCH = "CURRENT_ROW_MATCHES_UPDATE_GUARD"
UPDATE_GUARD_MISMATCH = "CURRENT_ROW_GUARD_MISMATCH"
INSERT_GUARD_CLEAR = "DUPLICATE_GUARD_CLEAR"
INSERT_GUARD_BLOCKED = "DUPLICATE_GUARD_BLOCKED"

UPDATE_CSV_FIELDS = [
    "update_id",
    "race_id",
    "db_rowid",
    "dog_name_key",
    "db_dog_name",
    "official_dog_name",
    "current_box_number",
    "proposed_box_number",
    "current_finish_position",
    "proposed_finish_position",
    "current_placing",
    "proposed_placing",
    "current_scraped_finish_position",
    "proposed_scraped_finish_position",
    "db_row_exists",
    "dog_name_key_matches",
    "current_box_matches",
    "current_finish_matches",
    "guard_status",
    "safe_to_apply_now",
]

INSERT_CSV_FIELDS = [
    "candidate_id",
    "race_id",
    "official_dog_name",
    "name_key",
    "box_number",
    "finish_position",
    "current_duplicate_guard_status",
    "current_duplicate_guard_hit_count",
    "current_duplicate_guard_hit_rowids",
    "current_duplicate_guard_hit_names",
    "simulated_duplicate_guard_status",
    "simulated_duplicate_guard_hit_count",
    "simulated_duplicate_guard_hit_rowids",
    "insert_values_json",
    "blocking_update_ids",
    "safe_to_apply_now",
]


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pipe(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values if value not in (None, ""))


def _forecast_rows_by_race(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in _list(packet.get("forecast_rows")):
        row_map = _mapping(row)
        race_id = str(row_map.get("race_id") or "")
        if race_id:
            rows[race_id] = row_map
    return rows


def _updates_for_race(packet: Mapping[str, Any], race_id: str) -> list[Mapping[str, Any]]:
    return [
        _mapping(row)
        for row in _list(packet.get("proposed_update_rows"))
        if str(_mapping(row).get("race_id") or "") == race_id
    ]


def _inserts_for_race(packet: Mapping[str, Any], race_id: str) -> list[Mapping[str, Any]]:
    return [
        _mapping(row)
        for row in _list(packet.get("deferred_insert_rows"))
        if str(_mapping(row).get("race_id") or "") == race_id
    ]


def _select_target_race(
    *,
    forecast_packet: Mapping[str, Any],
    requested_race_id: str | None,
    failures: list[str],
) -> str:
    if requested_race_id:
        return requested_race_id
    summary_race = _mapping(forecast_packet.get("summary")).get("smallest_batch_race_id")
    if summary_race:
        return str(summary_race)
    ranked = [
        _mapping(row)
        for row in _list(forecast_packet.get("forecast_rows"))
        if _mapping(row).get("batch_candidate_rank")
    ]
    ranked.sort(key=lambda row: _safe_int(row.get("batch_candidate_rank")) or 999999)
    if ranked:
        return str(ranked[0].get("race_id") or "")
    failures.append("target_race_id_missing")
    return "DATA_MISSING"


def _rowid_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("db_rowid")): row for row in rows if row.get("db_rowid") not in (None, "")}


def _update_guard_row(
    *,
    update: Mapping[str, Any],
    current_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = _rowid_map(current_rows).get(str(update.get("db_rowid")))
    row_map = _mapping(row)
    db_name_key = _name_key(row_map.get("dog_clean_name") or row_map.get("dog_name"))
    expected_name_key = str(update.get("dog_name_key") or "")
    row_exists = row is not None
    name_matches = row_exists and db_name_key == expected_name_key
    box_matches = row_exists and _safe_int(row_map.get("box_number")) == _safe_int(
        update.get("current_box_number")
    )
    finish_matches = row_exists and _safe_int(row_map.get("finish_position")) == _safe_int(
        update.get("current_finish_position")
    )
    if row_exists and name_matches and box_matches and finish_matches:
        status = UPDATE_GUARD_MATCH
    else:
        status = UPDATE_GUARD_MISMATCH
    return {
        "update_id": update.get("update_id"),
        "race_id": update.get("race_id"),
        "db_rowid": update.get("db_rowid"),
        "dog_name_key": expected_name_key,
        "db_dog_name": update.get("db_dog_name") or row_map.get("dog_name"),
        "official_dog_name": update.get("official_dog_name"),
        "current_box_number": update.get("current_box_number"),
        "proposed_box_number": update.get("proposed_box_number"),
        "current_finish_position": update.get("current_finish_position"),
        "proposed_finish_position": update.get("proposed_finish_position"),
        "current_placing": update.get("current_placing"),
        "proposed_placing": update.get("proposed_placing"),
        "current_scraped_finish_position": update.get("current_scraped_finish_position"),
        "proposed_scraped_finish_position": update.get("proposed_scraped_finish_position"),
        "db_row_exists": row_exists,
        "dog_name_key_matches": name_matches,
        "current_box_matches": box_matches,
        "current_finish_matches": finish_matches,
        "guard_status": status,
        "safe_to_apply_now": False,
        "db_current": dict(row_map),
        "source_update_row": dict(update),
    }


def _hit_rowids(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    return [row.get("db_rowid") for row in rows if row.get("db_rowid") not in (None, "")]


def _hit_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("dog_name") or row.get("dog_clean_name") or "") for row in rows]


def _insert_guard_rows(
    *,
    inserts: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    simulated_rows = [dict(row) for row in current_rows]
    _apply_updates(simulated_rows, updates)
    rows: list[dict[str, Any]] = []
    for insert in inserts:
        current_hits = _duplicate_hits(current_rows, insert)
        simulated_hits = _duplicate_hits(simulated_rows, insert)
        if not simulated_hits:
            _append_simulated_insert(simulated_rows, insert)
        rows.append(
            {
                "candidate_id": insert.get("candidate_id"),
                "race_id": insert.get("race_id"),
                "official_dog_name": insert.get("official_dog_name"),
                "name_key": insert.get("name_key"),
                "box_number": _insert_box(insert),
                "finish_position": _insert_finish(insert),
                "insert_values": dict(_insert_values(insert)),
                "blocking_update_ids": list(_list(insert.get("blocking_update_ids"))),
                "current_duplicate_guard_status": (
                    INSERT_GUARD_BLOCKED if current_hits else INSERT_GUARD_CLEAR
                ),
                "current_duplicate_guard_hit_count": len(current_hits),
                "current_duplicate_guard_hit_rows": [dict(row) for row in current_hits],
                "current_duplicate_guard_hit_rowids": _hit_rowids(current_hits),
                "current_duplicate_guard_hit_names": _hit_names(current_hits),
                "simulated_duplicate_guard_status": (
                    INSERT_GUARD_BLOCKED if simulated_hits else INSERT_GUARD_CLEAR
                ),
                "simulated_duplicate_guard_hit_count": len(simulated_hits),
                "simulated_duplicate_guard_hit_rows": [dict(row) for row in simulated_hits],
                "simulated_duplicate_guard_hit_rowids": _hit_rowids(simulated_hits),
                "safe_to_apply_now": False,
                "source_insert_row": dict(insert),
            }
        )
    return rows


def _update_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(_mapping(row)) for row in _list(packet.get("update_guard_rows"))]


def _insert_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _list(packet.get("insert_guard_rows")):
        row_map = dict(_mapping(row))
        row_map["insert_values_json"] = _json_cell(row_map.pop("insert_values", {}))
        row_map["blocking_update_ids"] = _pipe(_list(row_map.get("blocking_update_ids")))
        row_map["current_duplicate_guard_hit_rowids"] = _pipe(
            _list(row_map.get("current_duplicate_guard_hit_rowids"))
        )
        row_map["current_duplicate_guard_hit_names"] = _pipe(
            _list(row_map.get("current_duplicate_guard_hit_names"))
        )
        row_map["simulated_duplicate_guard_hit_rowids"] = _pipe(
            _list(row_map.get("simulated_duplicate_guard_hit_rowids"))
        )
        rows.append(row_map)
    return rows


def build_smallest_batch_approval_packet(
    *,
    post_update_forecast_packet_path: Path,
    update_policy_packet_path: Path | None = None,
    race_id: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    post_update_resolved = post_update_forecast_packet_path.expanduser().resolve()
    post_update_forecast = _load_json(post_update_resolved)
    failures: list[str] = []
    _validate_packet(
        packet=post_update_forecast,
        expected_schema=POST_UPDATE_FORECAST_SCHEMA_VERSION,
        packet_name="post_update_forecast_packet",
        failures=failures,
    )

    update_policy_path = update_policy_packet_path or _path_from_source(
        post_update_forecast,
        "update_policy_packet",
    )
    if update_policy_path is None:
        failures.append("update_policy_packet_path_missing")
        update_policy_path = Path("DATA_MISSING")
    update_policy_resolved = update_policy_path.expanduser().resolve(strict=False)
    update_policy: Mapping[str, Any] = {}
    if update_policy_resolved.name != "DATA_MISSING":
        update_policy = _load_json(update_policy_resolved)
        _validate_packet(
            packet=update_policy,
            expected_schema=UPDATE_POLICY_SCHEMA_VERSION,
            packet_name="update_policy_packet",
            failures=failures,
        )

    target_race_id = _select_target_race(
        forecast_packet=post_update_forecast,
        requested_race_id=race_id,
        failures=failures,
    )
    forecast_by_race = _forecast_rows_by_race(post_update_forecast)
    forecast_row = _mapping(forecast_by_race.get(target_race_id))
    if not forecast_row:
        failures.append(f"forecast_row_missing:{target_race_id}")
    elif forecast_row.get("post_update_gate") != GATE_BATCH_CANDIDATE:
        failures.append(f"target_race_not_batch_candidate:{target_race_id}")

    updates = _updates_for_race(update_policy, target_race_id)
    inserts = _inserts_for_race(update_policy, target_race_id)
    if not updates:
        failures.append(f"target_race_updates_missing:{target_race_id}")
    if not inserts:
        failures.append(f"target_race_deferred_inserts_missing:{target_race_id}")

    resolved_db = db_path or _path_from_source(post_update_forecast, "db") or _path_from_source(
        update_policy,
        "db",
    )
    if resolved_db is None:
        failures.append("db_path_missing")
        resolved_db = Path("DATA_MISSING")
    db_resolved = resolved_db.expanduser().resolve(strict=False)

    db_state: dict[str, Any] = {
        "db_path": str(db_resolved),
        "quick_check": None,
        "read_only": True,
        "query_only": True,
    }
    update_guard_rows: list[dict[str, Any]] = []
    insert_guard_rows: list[dict[str, Any]] = []
    current_db_rows: list[dict[str, Any]] = []
    if db_resolved.name != "DATA_MISSING":
        with _connect_read_only(db_resolved) as conn:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            db_state["quick_check"] = quick_check[0] if quick_check else None
            if db_state["quick_check"] != "ok":
                failures.append("db_quick_check_failed")
            current_db_rows = _race_rows(conn, target_race_id)
            update_guard_rows = [
                _update_guard_row(update=update, current_rows=current_db_rows)
                for update in updates
            ]
            insert_guard_rows = _insert_guard_rows(
                inserts=inserts,
                updates=updates,
                current_rows=current_db_rows,
            )

    update_guard_match_count = sum(
        1 for row in update_guard_rows if row.get("guard_status") == UPDATE_GUARD_MATCH
    )
    update_guard_mismatch_count = sum(
        1 for row in update_guard_rows if row.get("guard_status") != UPDATE_GUARD_MATCH
    )
    current_duplicate_hit_count = sum(
        _safe_int(row.get("current_duplicate_guard_hit_count")) or 0 for row in insert_guard_rows
    )
    simulated_duplicate_hit_count = sum(
        _safe_int(row.get("simulated_duplicate_guard_hit_count")) or 0 for row in insert_guard_rows
    )
    row_level_review_ready = (
        bool(update_guard_rows)
        and bool(insert_guard_rows)
        and update_guard_mismatch_count == 0
        and simulated_duplicate_hit_count == 0
        and forecast_row.get("post_update_gate") == GATE_BATCH_CANDIDATE
    )

    if update_guard_mismatch_count:
        failures.append(f"update_guard_mismatch_count:{update_guard_mismatch_count}")
    if simulated_duplicate_hit_count:
        failures.append(f"simulated_duplicate_guard_hit_count:{simulated_duplicate_hit_count}")

    exact_batch_review_ready = row_level_review_ready and not failures
    status = STATUS_OK if not failures else STATUS_FAILURES
    metadata_action_count = _safe_int(forecast_row.get("metadata_policy_action_count")) or 0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "failures": failures,
        "report_only": True,
        "safe_to_write_now": False,
        "label_write_ready": False,
        "target_race_id": target_race_id,
        "source_evidence": {
            "post_update_forecast_packet": str(post_update_resolved),
            "update_policy_packet": str(update_policy_resolved),
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "forecast_row": dict(forecast_row),
        "current_db_rows": current_db_rows,
        "summary": {
            "target_race_id": target_race_id,
            "post_update_gate": forecast_row.get("post_update_gate"),
            "update_operation_count": len(updates),
            "deferred_insert_operation_count": len(inserts),
            "metadata_action_count": metadata_action_count,
            "total_repair_operation_count": len(updates) + len(inserts) + metadata_action_count,
            "update_guard_match_count": update_guard_match_count,
            "update_guard_mismatch_count": update_guard_mismatch_count,
            "current_duplicate_guard_hit_count": current_duplicate_hit_count,
            "simulated_duplicate_guard_hit_count": simulated_duplicate_hit_count,
            "current_duplicate_guard_candidate_blocked_count": sum(
                1
                for row in insert_guard_rows
                if row.get("current_duplicate_guard_status") == INSERT_GUARD_BLOCKED
            ),
            "simulated_duplicate_guard_candidate_blocked_count": sum(
                1
                for row in insert_guard_rows
                if row.get("simulated_duplicate_guard_status") == INSERT_GUARD_BLOCKED
            ),
            "exact_batch_review_ready_after_backup_and_explicit_approval": (
                exact_batch_review_ready
            ),
            "safe_to_write_now_count": 0,
            "recommended_next_action": (
                "operator_review_exact_smallest_batch_then_explicitly_approve_backup_apply_or_keep_report_only"
            ),
        },
        "update_guard_rows": update_guard_rows,
        "insert_guard_rows": insert_guard_rows,
        "approval_manifest": {
            "approved_here": False,
            "target_race_id": target_race_id,
            "exact_update_ids": [str(row.get("update_id")) for row in update_guard_rows],
            "exact_insert_candidate_ids": [str(row.get("candidate_id")) for row in insert_guard_rows],
            "metadata_action_count": metadata_action_count,
            "required_before_any_apply": [
                "explicit_operator_approval_required",
                "current_db_backup_required_before_apply",
                "apply_exact_update_allowlist_only",
                "rerun_duplicate_guard_after_updates_before_insert",
                "apply_exact_insert_allowlist_only_if_guard_clear",
                "post_apply_gap_review_required",
                "post_apply_label_preflight_required",
            ],
        },
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "exact_update_allowlist_required": True,
            "exact_insert_allowlist_required_after_update_recheck": True,
            "metadata_policy_approval_required": metadata_action_count > 0,
            "rerun_duplicate_guard_after_updates_before_insert": True,
            "post_apply_gap_review_required": True,
            "post_apply_label_preflight_required": True,
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": [
            "db_write",
            "label_write",
            "metadata_write",
            "dog_row_insert",
            "dog_row_update",
            "dog_row_delete",
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
    (output_dir / "smallest_batch_approval_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "smallest_batch_update_guards.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=UPDATE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_update_csv_rows(packet))
    with (output_dir / "smallest_batch_insert_guards.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=INSERT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_insert_csv_rows(packet))
    _write_report(output_dir / "SUMMARY.md", packet)


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Smallest Batch Approval Packet",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Target race: `{summary.get('target_race_id')}`",
        f"- Update operations: `{summary.get('update_operation_count')}`",
        f"- Deferred insert operations: `{summary.get('deferred_insert_operation_count')}`",
        f"- Metadata actions: `{summary.get('metadata_action_count')}`",
        f"- Total repair operations: `{summary.get('total_repair_operation_count')}`",
        f"- Update guards matched: `{summary.get('update_guard_match_count')}`",
        f"- Update guard mismatches: `{summary.get('update_guard_mismatch_count')}`",
        f"- Current duplicate-guard hits: `{summary.get('current_duplicate_guard_hit_count')}`",
        f"- Simulated duplicate-guard hits after updates: `{summary.get('simulated_duplicate_guard_hit_count')}`",
        "- Exact batch review ready after backup and explicit approval: "
        f"`{summary.get('exact_batch_review_ready_after_backup_and_explicit_approval')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Gate",
        "",
        "This packet is not approval and does not make the batch safe to write now. "
        "It preserves the exact update and insert allowlists for operator review only. "
        "Any apply step still requires explicit DB repair approval, a current DB backup, "
        "immediate duplicate-guard rechecks, post-apply gap review, and official label "
        "preflight before label expansion or retraining.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-update-forecast-packet", required=True)
    parser.add_argument("--update-policy-packet", default=None)
    parser.add_argument("--race-id", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit(
            "refusing report-only smallest batch approval packet while approval flags are set:"
            + ",".join(active)
        )
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_smallest_batch_approval_packet(
        post_update_forecast_packet_path=Path(args.post_update_forecast_packet),
        update_policy_packet_path=Path(args.update_policy_packet)
        if args.update_policy_packet
        else None,
        race_id=args.race_id,
        db_path=Path(args.db) if args.db else None,
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
