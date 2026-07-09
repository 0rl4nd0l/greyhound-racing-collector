#!/usr/bin/env python3
"""Build a no-write update policy manifest for non-terminal repair races.

The duplicate-guard reconciliation packet identifies terminal-free races where
official missing-runner inserts collide with existing DB rows. This helper turns
that evidence into an exact report-only policy manifest: proposed current-row
box/finish updates are separated from missing-runner inserts that must remain
deferred until those update policies are approved. It never writes DB rows,
labels, snapshots, manifests, datasets, models, registries, TGR settings,
betting decisions, or EV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_non_terminal_duplicate_guard_update_reconciliation import (  # noqa: E402
    SCHEMA_VERSION as RECONCILIATION_SCHEMA_VERSION,
)
from scripts.build_non_terminal_repair_apply_manifest_forecast import (  # noqa: E402
    SCHEMA_VERSION as MANIFEST_SCHEMA_VERSION,
    WRITES_PERFORMED,
    _assert_output_dir_safe,
    _connect_read_only,
    _list,
    _load_json,
    _mapping,
    _safe_int,
    utc_now,
)
from scripts.build_terminal_scope_reconciliation_packet import (  # noqa: E402
    FORBIDDEN_APPROVAL_ENV_VARS,
    _name_key,
)


SCHEMA_VERSION = "non_terminal_update_policy_manifest_v1"
STATUS_OK = "REPORT_ONLY_NON_TERMINAL_UPDATE_POLICY_MANIFEST"
STATUS_FAILURES = "REPORT_ONLY_NON_TERMINAL_UPDATE_POLICY_MANIFEST_WITH_FAILURES"

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
    "row_match_guard",
    "policy_status",
    "safe_to_apply_now",
]

DEFERRED_INSERT_CSV_FIELDS = [
    "candidate_id",
    "race_id",
    "defer_reason",
    "official_dog_name",
    "name_key",
    "box_number",
    "finish_position",
    "insert_values_json",
    "blocking_update_ids",
    "policy_status",
    "safe_to_apply_now",
]

RACE_CSV_FIELDS = [
    "race_id",
    "review_lane",
    "proposed_update_count",
    "deferred_insert_count",
    "blocking_duplicate_candidate_count",
    "race_policy_status",
    "recommended_next_action",
]


def _validate_packet(
    *,
    packet: Mapping[str, Any],
    expected_schema: str,
    packet_name: str,
    failures: list[str],
) -> None:
    if packet.get("schema_version") != expected_schema:
        failures.append(f"{packet_name}_schema_mismatch")
    if packet.get("report_only") is not True:
        failures.append(f"{packet_name}_not_report_only")
    if packet.get("safe_to_write_now") is not False:
        failures.append(f"{packet_name}_safe_to_write_not_false")
    for key, value in _mapping(packet.get("writes_performed")).items():
        if value is not False:
            failures.append(f"{packet_name}_write_flag_true:{key}")


def _manifest_path_from_reconciliation(packet: Mapping[str, Any]) -> Path | None:
    raw = _mapping(packet.get("source_evidence")).get("manifest_packet")
    return Path(str(raw)) if raw else None


def _db_path_from_reconciliation(packet: Mapping[str, Any]) -> Path | None:
    raw = _mapping(packet.get("source_evidence")).get("db")
    return Path(str(raw)) if raw else None


def _candidate_by_id(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in _list(packet.get("candidate_manifest_rows")):
        row_map = _mapping(row)
        candidate_id = str(row_map.get("candidate_id") or "")
        if candidate_id:
            result[candidate_id] = row_map
    return result


def _race_review_lane(race: Mapping[str, Any]) -> str | None:
    lane = race.get("review_lane")
    return str(lane) if lane else None


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pipe(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values if value not in (None, ""))


def _race_db_rows_with_rowid(conn: sqlite3.Connection, race_id: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT rowid AS db_rowid, dog_name, dog_clean_name, box_number,
                   finish_position, placing, scraped_finish_position, data_source
            FROM dog_race_data
            WHERE race_id = ?
            ORDER BY box_number, dog_name
            """,
            (race_id,),
        ).fetchall()
    ]


def _find_db_row(
    *,
    rows: Sequence[Mapping[str, Any]],
    name_key: str,
    box_number: int | None,
    finish_position: int | None,
) -> Mapping[str, Any] | None:
    key_matches = [
        row
        for row in rows
        if _name_key(row.get("dog_clean_name") or row.get("dog_name")) == name_key
    ]
    if not key_matches:
        return None
    for row in key_matches:
        if _safe_int(row.get("box_number")) == box_number and _safe_int(row.get("finish_position")) == finish_position:
            return row
    return key_matches[0]


def _update_row(
    *,
    item: Mapping[str, Any],
    db_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    race_id = str(item.get("race_id") or "")
    name_key = str(item.get("name_key") or "")
    current_box = _safe_int(item.get("db_box_number"))
    current_finish = _safe_int(item.get("db_finish_position"))
    proposed_box = _safe_int(item.get("official_box_number"))
    proposed_finish = _safe_int(item.get("official_finish_position"))
    update_id = f"{race_id}::{name_key}::update_box_finish"
    return {
        "update_id": update_id,
        "race_id": race_id,
        "db_rowid": _mapping(db_row).get("db_rowid"),
        "dog_name_key": name_key,
        "db_dog_name": item.get("db_dog_name"),
        "official_dog_name": item.get("official_dog_name"),
        "current_box_number": current_box,
        "proposed_box_number": proposed_box,
        "current_finish_position": current_finish,
        "proposed_finish_position": proposed_finish,
        "current_placing": _safe_int(_mapping(db_row).get("placing")),
        "proposed_placing": proposed_finish,
        "current_scraped_finish_position": _mapping(db_row).get("scraped_finish_position"),
        "proposed_scraped_finish_position": str(proposed_finish) if proposed_finish is not None else None,
        "row_match_guard": (
            "race_id + db_rowid + dog_name_key + current_box_number + current_finish_position"
        ),
        "source_item_type": item.get("item_type"),
        "policy_status": "PROPOSED_REQUIRES_OPERATOR_APPROVAL_AND_BACKUP",
        "safe_to_apply_now": False,
        "required_before_apply": [
            "explicit_operator_approval_required",
            "db_backup_required_before_apply",
            "exact_update_allowlist_required",
            "rerun_duplicate_guard_after_updates_before_insert",
            "post_apply_gap_review_required",
            "post_apply_label_preflight_required",
        ],
    }


def _deferred_insert_row(
    *,
    item: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    blocking_update_ids: Sequence[str],
) -> dict[str, Any]:
    candidate_map = _mapping(candidate)
    insert_values = dict(_mapping(candidate_map.get("insert_values")))
    finish = _safe_int(item.get("official_finish_position") or candidate_map.get("finish_position") or insert_values.get("finish_position"))
    box = _safe_int(item.get("official_box_number") or candidate_map.get("box_number") or insert_values.get("box_number"))
    candidate_id = item.get("candidate_id") or candidate_map.get("candidate_id")
    item_type = str(item.get("item_type") or "")
    defer_reason = (
        "same_race_duplicate_guard_update_policy_required"
        if item_type == "candidate_deferred_until_same_race_duplicate_conflicts_resolved"
        else "candidate_box_occupied_until_update_policy_resolved"
    )
    return {
        "candidate_id": candidate_id,
        "race_id": item.get("race_id") or candidate_map.get("race_id") or insert_values.get("race_id"),
        "defer_reason": defer_reason,
        "official_dog_name": item.get("official_dog_name") or candidate_map.get("official_dog_name"),
        "name_key": item.get("name_key") or candidate_map.get("name_key"),
        "box_number": box,
        "finish_position": finish,
        "insert_columns": _list(candidate_map.get("insert_columns")),
        "insert_values": insert_values,
        "blocking_update_ids": list(blocking_update_ids),
        "policy_status": "DEFERRED_UNTIL_UPDATE_POLICY_APPROVED_AND_RECHECKED",
        "safe_to_apply_now": False,
        "required_before_apply": [
            "explicit_operator_approval_required",
            "db_backup_required_before_apply",
            "apply_or_reject_update_policy_first",
            "rerun_duplicate_guard_immediately_before_insert",
            "post_apply_gap_review_required",
            "post_apply_label_preflight_required",
        ],
    }


def _build_rows(
    *,
    reconciliation_packet: Mapping[str, Any],
    manifest_packet: Mapping[str, Any],
    conn: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = _candidate_by_id(manifest_packet)
    update_rows: list[dict[str, Any]] = []
    deferred_rows: list[dict[str, Any]] = []
    race_rows: list[dict[str, Any]] = []
    updates_by_race: dict[str, list[str]] = defaultdict(list)
    pending_deferred_by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    blocking_candidates_by_race: Counter[str] = Counter()
    review_lanes_by_race: dict[str, str | None] = {}

    for race in _list(reconciliation_packet.get("race_diagnostics")):
        race_map = _mapping(race)
        race_id = str(race_map.get("race_id") or "")
        if not race_id:
            continue
        review_lanes_by_race[race_id] = _race_review_lane(race_map)
        db_rows = _race_db_rows_with_rowid(conn, race_id)
        for item in _list(race_map.get("items")):
            item_map = _mapping(item)
            item_type = str(item_map.get("item_type") or "")
            if item_type == "matched_official_finisher_db_update_policy_required":
                name_key = str(item_map.get("name_key") or "")
                db_row = _find_db_row(
                    rows=db_rows,
                    name_key=name_key,
                    box_number=_safe_int(item_map.get("db_box_number")),
                    finish_position=_safe_int(item_map.get("db_finish_position")),
                )
                update = _update_row(item=item_map, db_row=db_row)
                update_rows.append(update)
                updates_by_race[race_id].append(str(update.get("update_id")))
            elif item_type == "duplicate_guard_box_or_name_conflict_policy_required":
                blocking_candidates_by_race[race_id] += 1
                pending_deferred_by_race[race_id].append(item_map)
            elif item_type == "candidate_deferred_until_same_race_duplicate_conflicts_resolved":
                pending_deferred_by_race[race_id].append(item_map)

    for race_id, items in sorted(pending_deferred_by_race.items()):
        blocking = updates_by_race.get(race_id, [])
        for item in items:
            candidate_id = str(_mapping(item).get("candidate_id") or "")
            deferred_rows.append(
                _deferred_insert_row(
                    item=_mapping(item),
                    candidate=candidates.get(candidate_id),
                    blocking_update_ids=blocking,
                )
            )

    race_ids = sorted(
        {str(row.get("race_id")) for row in update_rows if row.get("race_id")}
        | {str(row.get("race_id")) for row in deferred_rows if row.get("race_id")}
    )
    deferred_count_by_race = Counter(str(row.get("race_id") or "") for row in deferred_rows)
    update_count_by_race = Counter(str(row.get("race_id") or "") for row in update_rows)
    for race_id in race_ids:
        update_count = update_count_by_race.get(race_id, 0)
        deferred_count = deferred_count_by_race.get(race_id, 0)
        race_rows.append(
            {
                "race_id": race_id,
                "review_lane": review_lanes_by_race.get(race_id),
                "proposed_update_count": update_count,
                "deferred_insert_count": deferred_count,
                "blocking_duplicate_candidate_count": blocking_candidates_by_race.get(race_id, 0),
                "race_policy_status": "PROPOSED_REQUIRES_OPERATOR_APPROVAL_AND_BACKUP",
                "recommended_next_action": (
                    "operator_review_update_policy_then_rerun_duplicate_guard_and_gap_review"
                ),
            }
        )
    return update_rows, deferred_rows, race_rows


def build_update_policy_packet(
    *,
    reconciliation_packet_path: Path,
    manifest_packet_path: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    reconciliation_resolved = reconciliation_packet_path.expanduser().resolve()
    reconciliation_packet = _load_json(reconciliation_resolved)
    failures: list[str] = []
    _validate_packet(
        packet=reconciliation_packet,
        expected_schema=RECONCILIATION_SCHEMA_VERSION,
        packet_name="reconciliation_packet",
        failures=failures,
    )

    manifest_path = manifest_packet_path or _manifest_path_from_reconciliation(reconciliation_packet)
    if manifest_path is None:
        failures.append("manifest_path_missing")
        manifest_path = Path("DATA_MISSING")
    manifest_resolved = manifest_path.expanduser().resolve(strict=False)
    manifest_packet: Mapping[str, Any] = {}
    if manifest_resolved.name != "DATA_MISSING":
        manifest_packet = _load_json(manifest_resolved)
        _validate_packet(
            packet=manifest_packet,
            expected_schema=MANIFEST_SCHEMA_VERSION,
            packet_name="manifest_packet",
            failures=failures,
        )

    resolved_db = db_path or _db_path_from_reconciliation(reconciliation_packet)
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
    update_rows: list[dict[str, Any]] = []
    deferred_insert_rows: list[dict[str, Any]] = []
    race_policy_rows: list[dict[str, Any]] = []
    if db_resolved.name != "DATA_MISSING":
        with _connect_read_only(db_resolved) as conn:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            db_state["quick_check"] = quick_check[0] if quick_check else None
            if db_state["quick_check"] != "ok":
                failures.append("db_quick_check_failed")
            update_rows, deferred_insert_rows, race_policy_rows = _build_rows(
                reconciliation_packet=reconciliation_packet,
                manifest_packet=manifest_packet,
                conn=conn,
            )

    missing_rowids = sum(1 for row in update_rows if row.get("db_rowid") in (None, ""))
    if missing_rowids:
        failures.append(f"update_rows_missing_db_rowid:{missing_rowids}")

    status = STATUS_OK if not failures else STATUS_FAILURES
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "failures": failures,
        "report_only": True,
        "safe_to_write_now": False,
        "label_write_ready": False,
        "source_evidence": {
            "reconciliation_packet": str(reconciliation_resolved),
            "manifest_packet": str(manifest_resolved),
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "summary": {
            "races_considered": len(race_policy_rows),
            "proposed_update_count": len(update_rows),
            "deferred_insert_count": len(deferred_insert_rows),
            "blocking_duplicate_candidate_count": sum(
                _safe_int(row.get("blocking_duplicate_candidate_count")) or 0 for row in race_policy_rows
            ),
            "updates_missing_db_rowid_count": missing_rowids,
            "safe_to_write_now_count": 0,
            "recommended_next_action": (
                "operator_review_update_policy_then_explicitly_approve_backup_and_apply_or_keep_report_only"
            ),
        },
        "race_policy_rows": race_policy_rows,
        "proposed_update_rows": update_rows,
        "deferred_insert_rows": deferred_insert_rows,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "exact_update_allowlist_required": True,
            "exact_insert_allowlist_required_after_update_recheck": True,
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


def _update_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(_mapping(row)) for row in _list(packet.get("proposed_update_rows"))]


def _deferred_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _list(packet.get("deferred_insert_rows")):
        row_map = dict(_mapping(row))
        row_map["insert_values_json"] = _json_cell(row_map.pop("insert_values", {}))
        row_map["blocking_update_ids"] = _pipe(_list(row_map.get("blocking_update_ids")))
        rows.append(row_map)
    return rows


def _race_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(_mapping(row)) for row in _list(packet.get("race_policy_rows"))]


def write_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "non_terminal_update_policy_manifest_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "non_terminal_update_policy_updates.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=UPDATE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_update_csv_rows(packet))
    with (output_dir / "non_terminal_update_policy_deferred_inserts.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFERRED_INSERT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_deferred_csv_rows(packet))
    with (output_dir / "non_terminal_update_policy_races.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RACE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_race_csv_rows(packet))
    _write_report(output_dir / "SUMMARY.md", packet)


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Non-Terminal Update Policy Manifest",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Proposed update rows: `{summary.get('proposed_update_count')}`",
        f"- Deferred insert rows: `{summary.get('deferred_insert_count')}`",
        f"- Blocking duplicate candidates: `{summary.get('blocking_duplicate_candidate_count')}`",
        f"- Updates missing DB rowid: `{summary.get('updates_missing_db_rowid_count')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Gate",
        "",
        "This is an update policy manifest only. Any apply step still requires explicit operator approval, a current DB backup, exact update and insert allowlists, duplicate-guard rechecks, post-apply gap review, and label preflight before label expansion or retraining.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation-packet", required=True)
    parser.add_argument("--manifest-packet", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit(
            "refusing report-only update policy manifest while approval flags are set:"
            + ",".join(active)
        )
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_update_policy_packet(
        reconciliation_packet_path=Path(args.reconciliation_packet),
        manifest_packet_path=Path(args.manifest_packet) if args.manifest_packet else None,
        db_path=Path(args.db) if args.db else None,
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
