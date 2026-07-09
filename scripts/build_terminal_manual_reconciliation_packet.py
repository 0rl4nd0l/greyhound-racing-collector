#!/usr/bin/env python3
"""Build a no-write manual reconciliation packet for terminal-scope races.

Terminal/exclusion-scope races cannot safely move through the missing-runner
insert-only path. This helper rebuilds official, current-DB, terminal-status,
and candidate-insert evidence for those races and emits row-level manual policy
decisions. It never writes DB rows, labels, snapshots, manifests, datasets,
models, registries, TGR settings, betting decisions, or EV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_terminal_scope_reconciliation_packet import (
    FORBIDDEN_APPROVAL_ENV_VARS,
    WRITES_PERFORMED,
    _assert_output_dir_safe,
    _box,
    _connect_read_only,
    _fetch_db_rows,
    _list,
    _load_json,
    _lookup_index,
    _mapping,
    _name_key,
    _official_rows,
    _safe_int,
    _terminal_rows,
    utc_now,
)


SCHEMA_VERSION = "terminal_manual_reconciliation_packet_v1"
TERMINAL_SCOPE_SCHEMA_VERSION = "terminal_scope_reconciliation_packet_v1"
STATUS_OK = "REPORT_ONLY_TERMINAL_MANUAL_RECONCILIATION_PACKET"
STATUS_FAILURES = "REPORT_ONLY_TERMINAL_MANUAL_RECONCILIATION_PACKET_WITH_FAILURES"

CSV_FIELDS = [
    "race_id",
    "item_type",
    "name_key",
    "official_dog_name",
    "db_dog_name",
    "official_box_number",
    "db_box_number",
    "official_finish_position",
    "db_finish_position",
    "terminal_status",
    "conflict_box_number",
    "conflicting_db_name_keys",
    "apply_batch_status",
    "recommended_policy",
    "safe_to_apply_now",
]


def _validate_terminal_scope_packet(packet: Mapping[str, Any], failures: list[str]) -> None:
    if packet.get("schema_version") != TERMINAL_SCOPE_SCHEMA_VERSION:
        failures.append("terminal_scope_schema_mismatch")
    if packet.get("report_only") is not True:
        failures.append("terminal_scope_not_report_only")
    if packet.get("safe_to_write_now") is not False:
        failures.append("terminal_scope_safe_to_write_not_false")
    for key, value in _mapping(packet.get("writes_performed")).items():
        if value is not False:
            failures.append(f"terminal_scope_write_flag_true:{key}")


def _lookup_paths_from_packet(packet: Mapping[str, Any]) -> list[Path]:
    source_evidence = _mapping(packet.get("source_evidence"))
    return [Path(str(path)) for path in _list(source_evidence.get("lookup_packets"))]


def _db_path_from_packet(packet: Mapping[str, Any]) -> Path | None:
    raw = _mapping(packet.get("source_evidence")).get("db")
    return Path(str(raw)) if raw else None


def _target_race_ids(packet: Mapping[str, Any]) -> list[str]:
    race_ids = []
    for row in _list(packet.get("race_diagnostics")):
        row_map = _mapping(row)
        if row_map.get("reconciliation_lane") != "TERMINAL_SCOPE_MANUAL_RECONCILIATION_REQUIRED":
            continue
        race_id = str(row_map.get("race_id") or "")
        if race_id:
            race_ids.append(race_id)
    return sorted(dict.fromkeys(race_ids))


def _db_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dog_name": row.get("dog_name"),
        "dog_clean_name": row.get("dog_clean_name"),
        "name_key": _name_key(row.get("dog_clean_name") or row.get("dog_name")),
        "box_number": _box(row.get("box_number")),
        "finish_position": _safe_int(row.get("finish_position")),
        "placing": _safe_int(row.get("placing")),
        "scraped_finish_position": row.get("scraped_finish_position"),
        "data_source": row.get("data_source"),
    }


def _candidate_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    insert_values = _mapping(candidate.get("insert_values"))
    return {
        "candidate_id": candidate.get("candidate_id"),
        "official_dog_name": candidate.get("official_dog_name"),
        "name_key": _name_key(candidate.get("name_key") or candidate.get("official_dog_name")),
        "box_number": _box(candidate.get("box_number") or insert_values.get("box_number")),
        "finish_position": _safe_int(candidate.get("finish_position") or insert_values.get("finish_position")),
        "source_url": candidate.get("source_url"),
        "status": candidate.get("status"),
    }


def _candidate_by_name(diag: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = {}
    for candidate in _list(diag.get("naive_insert_candidates")):
        payload = _candidate_payload(_mapping(candidate))
        key = str(payload.get("name_key") or "")
        if key:
            candidates[key] = payload
    return candidates


def _db_rows_by_key(db_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in db_rows:
        payload = _db_row_payload(row)
        key = str(payload.get("name_key") or "")
        if key:
            grouped.setdefault(key, []).append(payload)
    return grouped


def _db_rows_by_box(db_rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in db_rows:
        payload = _db_row_payload(row)
        box = payload.get("box_number")
        if isinstance(box, int):
            grouped.setdefault(box, []).append(payload)
    return grouped


def _csv_text(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values if value not in (None, ""))


def _manual_item(
    *,
    race_id: str,
    item_type: str,
    name_key: str = "",
    official_dog_name: Any = None,
    db_dog_name: Any = None,
    official_box_number: Any = None,
    db_box_number: Any = None,
    official_finish_position: Any = None,
    db_finish_position: Any = None,
    terminal_status: Any = None,
    conflict_box_number: Any = None,
    conflicting_db_rows: Sequence[Mapping[str, Any]] = (),
    apply_batch_status: str,
    recommended_policy: str,
) -> dict[str, Any]:
    return {
        "race_id": race_id,
        "item_type": item_type,
        "name_key": name_key,
        "official_dog_name": official_dog_name,
        "db_dog_name": db_dog_name,
        "official_box_number": official_box_number,
        "db_box_number": db_box_number,
        "official_finish_position": official_finish_position,
        "db_finish_position": db_finish_position,
        "terminal_status": terminal_status,
        "conflict_box_number": conflict_box_number,
        "conflicting_db_rows": [dict(row) for row in conflicting_db_rows],
        "conflicting_db_name_keys": [row.get("name_key") for row in conflicting_db_rows],
        "apply_batch_status": apply_batch_status,
        "recommended_policy": recommended_policy,
        "safe_to_apply_now": False,
    }


def _official_name(row: Mapping[str, Any]) -> Any:
    return row.get("dog_name") or row.get("official_dog_name")


def _diagnose_manual_race(
    *,
    race_id: str,
    terminal_diag: Mapping[str, Any],
    lookup: Mapping[str, Any],
    db_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    official_rows = _official_rows(lookup)
    terminal_rows = _terminal_rows(lookup)
    db_by_key = _db_rows_by_key(db_rows)
    db_by_box = _db_rows_by_box(db_rows)
    candidates_by_key = _candidate_by_name(terminal_diag)

    official_keys = {str(row.get("name_key") or "") for row in official_rows if row.get("name_key")}
    terminal_boxes = {row.get("box_number") for row in terminal_rows if isinstance(row.get("box_number"), int)}
    items: list[dict[str, Any]] = []

    for terminal in terminal_rows:
        terminal_box = terminal.get("box_number")
        terminal_db_rows = db_by_box.get(terminal_box, []) if isinstance(terminal_box, int) else []
        if terminal_db_rows:
            for db_row in terminal_db_rows:
                items.append(
                    _manual_item(
                        race_id=race_id,
                        item_type="terminal_status_db_row_policy_required",
                        name_key=str(db_row.get("name_key") or ""),
                        db_dog_name=db_row.get("dog_clean_name") or db_row.get("dog_name"),
                        db_box_number=db_row.get("box_number"),
                        db_finish_position=db_row.get("finish_position"),
                        terminal_status=terminal.get("status"),
                        conflict_box_number=terminal_box,
                        conflicting_db_rows=[db_row],
                        apply_batch_status="exclude_from_insert_only_apply",
                        recommended_policy=(
                            "manual_decision_required_for_current_db_row_on_terminal_status_box"
                        ),
                    )
                )
        else:
            items.append(
                _manual_item(
                    race_id=race_id,
                    item_type="terminal_status_without_current_db_row_record_only",
                    db_box_number=terminal_box,
                    terminal_status=terminal.get("status"),
                    apply_batch_status="record_terminal_exclusion_only",
                    recommended_policy="no_insert_or_delete_in_report_only_packet",
                )
            )

    for official in official_rows:
        key = str(official.get("name_key") or "")
        if not key:
            continue
        official_box = official.get("box_number")
        official_finish = official.get("finish_position")
        db_matches = db_by_key.get(key, [])
        if db_matches:
            for db_row in db_matches:
                db_box = db_row.get("box_number")
                db_finish = db_row.get("finish_position")
                if db_box != official_box or db_finish != official_finish:
                    items.append(
                        _manual_item(
                            race_id=race_id,
                            item_type="matched_official_finisher_db_value_review_required",
                            name_key=key,
                            official_dog_name=_official_name(official),
                            db_dog_name=db_row.get("dog_clean_name") or db_row.get("dog_name"),
                            official_box_number=official_box,
                            db_box_number=db_box,
                            official_finish_position=official_finish,
                            db_finish_position=db_finish,
                            apply_batch_status="defer_until_manual_update_policy",
                            recommended_policy=(
                                "manual_decision_required_before_any_finish_or_box_update"
                            ),
                        )
                    )
            continue

        candidate = candidates_by_key.get(key)
        conflict_rows = db_by_box.get(official_box, []) if isinstance(official_box, int) else []
        if conflict_rows:
            items.append(
                _manual_item(
                    race_id=race_id,
                    item_type="missing_official_finisher_box_conflict_policy_required",
                    name_key=key,
                    official_dog_name=_official_name(official),
                    official_box_number=official_box,
                    official_finish_position=official_finish,
                    conflict_box_number=official_box,
                    conflicting_db_rows=conflict_rows,
                    apply_batch_status="exclude_from_insert_only_apply",
                    recommended_policy=(
                        "resolve_existing_box_occupant_before_any_missing_finisher_insert"
                    ),
                )
            )
        elif candidate is not None:
            items.append(
                _manual_item(
                    race_id=race_id,
                    item_type="missing_official_finisher_insert_deferred_until_terminal_policy",
                    name_key=key,
                    official_dog_name=_official_name(official),
                    official_box_number=official_box,
                    official_finish_position=official_finish,
                    apply_batch_status="defer_until_terminal_policy_resolved",
                    recommended_policy=(
                        "candidate_has_no_box_conflict_but_race_remains_terminal_scope"
                    ),
                )
            )
        else:
            items.append(
                _manual_item(
                    race_id=race_id,
                    item_type="missing_official_finisher_without_insert_candidate",
                    name_key=key,
                    official_dog_name=_official_name(official),
                    official_box_number=official_box,
                    official_finish_position=official_finish,
                    apply_batch_status="exclude_from_apply_until_candidate_exists",
                    recommended_policy="build_exact_source_bound_candidate_before_any_apply",
                )
            )

    for db_key, rows in db_by_key.items():
        if db_key in official_keys:
            continue
        for db_row in rows:
            item_type = (
                "extra_db_row_on_terminal_status_box_policy_required"
                if db_row.get("box_number") in terminal_boxes
                else "extra_db_row_not_in_official_finishers_policy_required"
            )
            items.append(
                _manual_item(
                    race_id=race_id,
                    item_type=item_type,
                    name_key=db_key,
                    db_dog_name=db_row.get("dog_clean_name") or db_row.get("dog_name"),
                    db_box_number=db_row.get("box_number"),
                    db_finish_position=db_row.get("finish_position"),
                    terminal_status=(
                        next(
                            (
                                row.get("status")
                                for row in terminal_rows
                                if row.get("box_number") == db_row.get("box_number")
                            ),
                            None,
                        )
                    ),
                    apply_batch_status="exclude_from_insert_only_apply",
                    recommended_policy=(
                        "manual_decision_required_for_extra_current_db_row_before_label_expansion"
                    ),
                )
            )

    item_counts = Counter(item["item_type"] for item in items)
    return {
        "race_id": race_id,
        "review_lane": terminal_diag.get("review_lane"),
        "forecast_gate": terminal_diag.get("forecast_gate"),
        "manual_policy_lane": "TERMINAL_SCOPE_MANUAL_RECONCILIATION_REQUIRED",
        "official_finisher_count": len(official_rows),
        "terminal_status_count": len(terminal_rows),
        "current_db_runner_count": len(db_rows),
        "manual_item_count": len(items),
        "item_type_counts": dict(sorted(item_counts.items())),
        "safe_for_insert_only": False,
        "safe_to_write_now": False,
        "recommended_next_action": (
            "operator_author_terminal_scope_policy_then_rebuild_apply_manifest"
        ),
        "items": items,
    }


def build_manual_reconciliation_packet(
    *,
    terminal_scope_packet_path: Path,
    db_path: Path | None = None,
    lookup_packet_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    terminal_resolved = terminal_scope_packet_path.expanduser().resolve()
    terminal_packet = _load_json(terminal_resolved)
    failures: list[str] = []
    _validate_terminal_scope_packet(terminal_packet, failures)

    resolved_db = (db_path or _db_path_from_packet(terminal_packet))
    if resolved_db is None:
        failures.append("db_path_missing")
        resolved_db = Path("DATA_MISSING")
    db_resolved = resolved_db.expanduser().resolve(strict=False)

    lookup_paths = list(lookup_packet_paths) if lookup_packet_paths is not None else _lookup_paths_from_packet(terminal_packet)
    if not lookup_paths:
        failures.append("lookup_packets_missing")
    lookup_by_race = _lookup_index(lookup_paths, failures)

    terminal_by_race = {
        str(_mapping(row).get("race_id") or ""): _mapping(row)
        for row in _list(terminal_packet.get("race_diagnostics"))
    }
    target_race_ids = _target_race_ids(terminal_packet)

    db_state: dict[str, Any] = {
        "db_path": str(db_resolved),
        "quick_check": None,
        "read_only": True,
        "query_only": True,
    }
    race_diagnostics = []
    if db_resolved.name != "DATA_MISSING":
        with _connect_read_only(db_resolved) as conn:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            db_state["quick_check"] = quick_check[0] if quick_check else None
            if db_state["quick_check"] != "ok":
                failures.append("db_quick_check_failed")
            for race_id in target_race_ids:
                lookup = lookup_by_race.get(race_id)
                if lookup is None:
                    failures.append(f"lookup_missing:{race_id}")
                    continue
                race_diagnostics.append(
                    _diagnose_manual_race(
                        race_id=race_id,
                        terminal_diag=terminal_by_race.get(race_id, {}),
                        lookup=lookup,
                        db_rows=_fetch_db_rows(conn, race_id),
                    )
                )

    all_items = [item for row in race_diagnostics for item in _list(row.get("items"))]
    item_counts = Counter(str(item.get("item_type")) for item in all_items)
    apply_counts = Counter(str(item.get("apply_batch_status")) for item in all_items)
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
            "terminal_scope_packet": str(terminal_resolved),
            "lookup_packets": [str(path.expanduser().resolve(strict=False)) for path in lookup_paths],
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "summary": {
            "races_considered": len(race_diagnostics),
            "manual_item_count": len(all_items),
            "item_type_counts": dict(sorted(item_counts.items())),
            "apply_batch_status_counts": dict(sorted(apply_counts.items())),
            "races_excluded_from_insert_only_apply_count": sum(
                1 for row in race_diagnostics if row.get("safe_for_insert_only") is False
            ),
            "missing_official_finisher_manual_count": sum(
                1 for item in all_items if str(item.get("item_type")).startswith("missing_official_finisher")
            ),
            "extra_db_row_manual_count": sum(
                1 for item in all_items if str(item.get("item_type")).startswith("extra_db_row")
            ),
            "terminal_status_manual_count": sum(
                1 for item in all_items if str(item.get("item_type")).startswith("terminal_status")
            ),
            "matched_official_update_review_count": sum(
                1
                for item in all_items
                if item.get("item_type") == "matched_official_finisher_db_value_review_required"
            ),
            "recommended_next_action": (
                "operator_review_exact_terminal_scope_manual_policy_before_any_db_or_label_apply"
            ),
        },
        "race_diagnostics": race_diagnostics,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "terminal_policy_required": True,
            "manual_row_decision_required": True,
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


def _csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for race in _list(packet.get("race_diagnostics")):
        for item in _list(_mapping(race).get("items")):
            item_map = _mapping(item)
            rows.append(
                {
                    "race_id": item_map.get("race_id"),
                    "item_type": item_map.get("item_type"),
                    "name_key": item_map.get("name_key"),
                    "official_dog_name": item_map.get("official_dog_name"),
                    "db_dog_name": item_map.get("db_dog_name"),
                    "official_box_number": item_map.get("official_box_number"),
                    "db_box_number": item_map.get("db_box_number"),
                    "official_finish_position": item_map.get("official_finish_position"),
                    "db_finish_position": item_map.get("db_finish_position"),
                    "terminal_status": item_map.get("terminal_status"),
                    "conflict_box_number": item_map.get("conflict_box_number"),
                    "conflicting_db_name_keys": _csv_text(_list(item_map.get("conflicting_db_name_keys"))),
                    "apply_batch_status": item_map.get("apply_batch_status"),
                    "recommended_policy": item_map.get("recommended_policy"),
                    "safe_to_apply_now": item_map.get("safe_to_apply_now"),
                }
            )
    return rows


def write_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "terminal_manual_reconciliation_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "terminal_manual_reconciliation_items.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_rows(packet))
    _write_report(output_dir / "SUMMARY.md", packet)


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Terminal Manual Reconciliation",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Manual reconciliation items: `{summary.get('manual_item_count')}`",
        f"- Item type counts: `{summary.get('item_type_counts')}`",
        f"- Apply batch status counts: `{summary.get('apply_batch_status_counts')}`",
        f"- Races excluded from insert-only apply: `{summary.get('races_excluded_from_insert_only_apply_count')}`",
        f"- Missing official finisher manual items: `{summary.get('missing_official_finisher_manual_count')}`",
        f"- Extra DB row manual items: `{summary.get('extra_db_row_manual_count')}`",
        f"- Terminal status manual items: `{summary.get('terminal_status_manual_count')}`",
        f"- Matched official update reviews: `{summary.get('matched_official_update_review_count')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Recommendation",
        "",
        "Keep these races out of any insert-only apply batch. Operator policy is still required for terminal-status DB rows, extra current DB rows, box conflicts, and deferred missing-finisher inserts before label expansion or retraining.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-scope-packet", required=True)
    parser.add_argument("--lookup-packet", action="append", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit(
            "refusing report-only terminal manual reconciliation while approval flags are set:"
            + ",".join(active)
        )
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_manual_reconciliation_packet(
        terminal_scope_packet_path=Path(args.terminal_scope_packet),
        db_path=Path(args.db) if args.db else None,
        lookup_packet_paths=[Path(path) for path in args.lookup_packet] if args.lookup_packet else None,
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
