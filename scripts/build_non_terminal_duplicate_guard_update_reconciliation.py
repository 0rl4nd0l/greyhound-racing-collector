#!/usr/bin/env python3
"""Build a no-write duplicate-guard/update reconciliation packet.

The non-terminal apply-manifest forecast separates terminal-scope races from
terminal-free races, but many terminal-free races still fail because a proposed
official missing-runner insert collides with an existing DB row in the same box.
This helper adds official lookup context to those collisions and emits row-level
manual update policy requirements. It never writes DB rows, labels, snapshots,
manifests, datasets, models, registries, TGR settings, betting decisions, or EV
artifacts.
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

from scripts.build_non_terminal_repair_apply_manifest_forecast import (  # noqa: E402
    DUPLICATE_EXCLUDED_STATUS,
    RACE_BLOCKED_STATUS,
    SCHEMA_VERSION as MANIFEST_SCHEMA_VERSION,
    WRITES_PERFORMED,
    _assert_output_dir_safe,
    _candidate_box,
    _candidate_insert_values,
    _connect_read_only,
    _duplicate_guard_hits,
    _list,
    _load_json,
    _mapping,
    _safe_int,
    utc_now,
)
from scripts.build_terminal_scope_reconciliation_packet import (  # noqa: E402
    FORBIDDEN_APPROVAL_ENV_VARS,
    _fetch_db_rows,
    _lookup_index,
    _name_key,
    _official_rows,
)


SCHEMA_VERSION = "non_terminal_duplicate_guard_update_reconciliation_packet_v1"
POLICY_SCHEMA_VERSION = "missing_runner_insert_policy_packet_v1"
STATUS_OK = "REPORT_ONLY_NON_TERMINAL_DUPLICATE_GUARD_UPDATE_RECONCILIATION"
STATUS_FAILURES = "REPORT_ONLY_NON_TERMINAL_DUPLICATE_GUARD_UPDATE_RECONCILIATION_WITH_FAILURES"

ITEM_CSV_FIELDS = [
    "race_id",
    "item_type",
    "candidate_id",
    "official_dog_name",
    "name_key",
    "official_box_number",
    "official_finish_position",
    "db_dog_name",
    "db_name_key",
    "db_box_number",
    "db_finish_position",
    "conflict_type",
    "conflict_db_name_in_official_finishers",
    "conflict_official_box_number",
    "conflict_official_finish_position",
    "recommended_policy",
    "safe_to_apply_now",
]

RACE_CSV_FIELDS = [
    "race_id",
    "review_lane",
    "candidate_count",
    "duplicate_guard_candidate_count",
    "race_deferred_candidate_count",
    "duplicate_guard_hit_count",
    "matched_official_update_review_count",
    "extra_db_conflict_count",
    "same_name_duplicate_conflict_count",
    "race_reconciliation_status",
    "recommended_next_action",
]


def _validate_report_packet(
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


def _db_path_from_manifest(packet: Mapping[str, Any]) -> Path | None:
    raw = _mapping(packet.get("source_evidence")).get("db")
    return Path(str(raw)) if raw else None


def _policy_path_from_manifest(packet: Mapping[str, Any]) -> Path | None:
    raw = _mapping(packet.get("source_evidence")).get("insert_policy_packet")
    return Path(str(raw)) if raw else None


def _lookup_paths_from_policy(packet: Mapping[str, Any]) -> list[Path]:
    return [Path(str(path)) for path in _list(_mapping(packet.get("source_evidence")).get("lookup_packets"))]


def _candidate_rows_by_race(packet: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _list(packet.get("candidate_manifest_rows")):
        row_map = _mapping(row)
        race_id = str(row_map.get("race_id") or "")
        if race_id:
            rows[race_id].append(row_map)
    return rows


def _target_race_ids(packet: Mapping[str, Any]) -> list[str]:
    race_ids = set()
    for row in _list(packet.get("candidate_manifest_rows")):
        row_map = _mapping(row)
        if row_map.get("apply_manifest_status") == DUPLICATE_EXCLUDED_STATUS:
            race_id = str(row_map.get("race_id") or "")
            if race_id:
                race_ids.add(race_id)
    return sorted(race_ids)


def _db_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dog_name": row.get("dog_name"),
        "dog_clean_name": row.get("dog_clean_name"),
        "name_key": _name_key(row.get("dog_clean_name") or row.get("dog_name")),
        "box_number": _safe_int(row.get("box_number")),
        "finish_position": _safe_int(row.get("finish_position")),
        "placing": _safe_int(row.get("placing")),
        "scraped_finish_position": row.get("scraped_finish_position"),
        "data_source": row.get("data_source"),
    }


def _official_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = str(row.get("name_key") or "")
        if key:
            result[key] = row
    return result


def _db_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        payload = _db_payload(row)
        key = str(payload.get("name_key") or "")
        if key:
            result[key].append(payload)
    return result


def _candidate_name_key(candidate: Mapping[str, Any]) -> str:
    return _name_key(candidate.get("name_key") or candidate.get("official_dog_name"))


def _item(
    *,
    race_id: str,
    item_type: str,
    candidate: Mapping[str, Any] | None = None,
    official_row: Mapping[str, Any] | None = None,
    db_row: Mapping[str, Any] | None = None,
    conflict_type: str | None = None,
    conflict_official_row: Mapping[str, Any] | None = None,
    recommended_policy: str,
) -> dict[str, Any]:
    candidate_map = _mapping(candidate)
    official = _mapping(official_row)
    db = _mapping(db_row)
    conflict_official = _mapping(conflict_official_row)
    db_name = db.get("dog_clean_name") or db.get("dog_name")
    db_key = str(db.get("name_key") or "")
    return {
        "race_id": race_id,
        "item_type": item_type,
        "candidate_id": candidate_map.get("candidate_id"),
        "official_dog_name": official.get("dog_name") or candidate_map.get("official_dog_name"),
        "name_key": official.get("name_key") or _candidate_name_key(candidate_map),
        "official_box_number": official.get("box_number") or _candidate_box(candidate_map),
        "official_finish_position": official.get("finish_position")
        or _safe_int(candidate_map.get("finish_position") or _candidate_insert_values(candidate_map).get("finish_position")),
        "db_dog_name": db_name,
        "db_name_key": db_key,
        "db_box_number": db.get("box_number"),
        "db_finish_position": db.get("finish_position"),
        "conflict_type": conflict_type,
        "conflict_db_name_in_official_finishers": bool(conflict_official),
        "conflict_official_box_number": conflict_official.get("box_number"),
        "conflict_official_finish_position": conflict_official.get("finish_position"),
        "conflict_official_dog_name": conflict_official.get("dog_name"),
        "recommended_policy": recommended_policy,
        "safe_to_apply_now": False,
    }


def _conflict_type(
    *,
    candidate_key: str,
    db_row: Mapping[str, Any],
    official_by_key: Mapping[str, Mapping[str, Any]],
) -> str:
    db_key = str(db_row.get("name_key") or "")
    if db_key == candidate_key:
        return "candidate_name_already_exists_in_db"
    if db_key in official_by_key:
        return "occupied_box_matches_other_official_finisher"
    return "occupied_box_extra_db_row_not_official"


def _conflict_policy(conflict_type: str) -> str:
    if conflict_type == "candidate_name_already_exists_in_db":
        return "reconcile_existing_candidate_identity_before_any_insert"
    if conflict_type == "occupied_box_matches_other_official_finisher":
        return "manual_update_policy_required_for_existing_official_finisher_before_any_insert"
    return "manual_extra_db_row_policy_required_before_any_insert"


def _diagnose_race(
    *,
    race_id: str,
    candidates: Sequence[Mapping[str, Any]],
    lookup: Mapping[str, Any],
    db_rows: Sequence[Mapping[str, Any]],
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    official_rows = _official_rows(lookup)
    official_by_name = _official_by_key(official_rows)
    db_by_name = _db_by_key(db_rows)
    items: list[dict[str, Any]] = []
    duplicate_candidates = [
        _mapping(row) for row in candidates if _mapping(row).get("apply_manifest_status") == DUPLICATE_EXCLUDED_STATUS
    ]
    deferred_candidates = [
        _mapping(row) for row in candidates if _mapping(row).get("apply_manifest_status") == RACE_BLOCKED_STATUS
    ]

    duplicate_hit_count = 0
    for candidate in duplicate_candidates:
        candidate_key = _candidate_name_key(candidate)
        official_row = official_by_name.get(candidate_key)
        duplicate_hits = [_db_payload(row) for row in _duplicate_guard_hits(conn, candidate)]
        duplicate_hit_count += len(duplicate_hits)
        if not duplicate_hits:
            items.append(
                _item(
                    race_id=race_id,
                    item_type="duplicate_guard_status_drifted_no_current_hit",
                    candidate=candidate,
                    official_row=official_row,
                    recommended_policy="rerun_manifest_forecast_before_any_apply",
                )
            )
            continue
        for hit in duplicate_hits:
            conflict_type = _conflict_type(
                candidate_key=candidate_key,
                db_row=hit,
                official_by_key=official_by_name,
            )
            items.append(
                _item(
                    race_id=race_id,
                    item_type="duplicate_guard_box_or_name_conflict_policy_required",
                    candidate=candidate,
                    official_row=official_row,
                    db_row=hit,
                    conflict_type=conflict_type,
                    conflict_official_row=official_by_name.get(str(hit.get("name_key") or "")),
                    recommended_policy=_conflict_policy(conflict_type),
                )
            )

    for key, db_matches in db_by_name.items():
        official_row = official_by_name.get(key)
        if not official_row:
            continue
        for db_row in db_matches:
            if db_row.get("box_number") != official_row.get("box_number") or db_row.get("finish_position") != official_row.get("finish_position"):
                items.append(
                    _item(
                        race_id=race_id,
                        item_type="matched_official_finisher_db_update_policy_required",
                        official_row=official_row,
                        db_row=db_row,
                        conflict_type="official_finisher_db_box_or_finish_drift",
                        recommended_policy="manual_update_policy_required_before_label_expansion",
                    )
                )

    for candidate in deferred_candidates:
        candidate_key = _candidate_name_key(candidate)
        items.append(
            _item(
                race_id=race_id,
                item_type="candidate_deferred_until_same_race_duplicate_conflicts_resolved",
                candidate=candidate,
                official_row=official_by_name.get(candidate_key),
                recommended_policy="keep_deferred_until_all_same_race_duplicate_guard_conflicts_have_update_policy",
            )
        )

    item_counts = Counter(str(item.get("item_type")) for item in items)
    conflict_counts = Counter(str(item.get("conflict_type")) for item in items if item.get("conflict_type"))
    race_status = (
        "RACE_REQUIRES_DUPLICATE_GUARD_UPDATE_POLICY"
        if duplicate_candidates
        else "RACE_HAS_NO_DUPLICATE_GUARD_CONFLICTS"
    )
    return {
        "race_id": race_id,
        "review_lane": next((row.get("review_lane") for row in candidates if row.get("review_lane")), None),
        "candidate_count": len(candidates),
        "duplicate_guard_candidate_count": len(duplicate_candidates),
        "race_deferred_candidate_count": len(deferred_candidates),
        "duplicate_guard_hit_count": duplicate_hit_count,
        "matched_official_update_review_count": item_counts.get(
            "matched_official_finisher_db_update_policy_required", 0
        ),
        "extra_db_conflict_count": conflict_counts.get("occupied_box_extra_db_row_not_official", 0),
        "same_name_duplicate_conflict_count": conflict_counts.get("candidate_name_already_exists_in_db", 0),
        "item_type_counts": dict(sorted(item_counts.items())),
        "conflict_type_counts": dict(sorted(conflict_counts.items())),
        "race_reconciliation_status": race_status,
        "recommended_next_action": "author_update_policy_or_keep_race_excluded_from_apply",
        "items": items,
    }


def build_reconciliation_packet(
    *,
    manifest_packet_path: Path,
    db_path: Path | None = None,
    lookup_packet_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    manifest_resolved = manifest_packet_path.expanduser().resolve()
    manifest_packet = _load_json(manifest_resolved)
    failures: list[str] = []
    _validate_report_packet(
        packet=manifest_packet,
        expected_schema=MANIFEST_SCHEMA_VERSION,
        packet_name="manifest_packet",
        failures=failures,
    )

    policy_path = _policy_path_from_manifest(manifest_packet)
    policy_packet: Mapping[str, Any] = {}
    if policy_path is None:
        failures.append("policy_path_missing_from_manifest")
    else:
        policy_packet = _load_json(policy_path.expanduser().resolve())
        _validate_report_packet(
            packet=policy_packet,
            expected_schema=POLICY_SCHEMA_VERSION,
            packet_name="insert_policy_packet",
            failures=failures,
        )

    resolved_db = db_path or _db_path_from_manifest(manifest_packet)
    if resolved_db is None:
        failures.append("db_path_missing")
        resolved_db = Path("DATA_MISSING")
    db_resolved = resolved_db.expanduser().resolve(strict=False)

    lookup_paths = list(lookup_packet_paths) if lookup_packet_paths is not None else _lookup_paths_from_policy(policy_packet)
    if not lookup_paths:
        failures.append("lookup_packets_missing")
    lookup_by_race = _lookup_index(lookup_paths, failures)
    candidates_by_race = _candidate_rows_by_race(manifest_packet)
    target_race_ids = _target_race_ids(manifest_packet)

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
                    _diagnose_race(
                        race_id=race_id,
                        candidates=candidates_by_race.get(race_id, []),
                        lookup=lookup,
                        db_rows=_fetch_db_rows(conn, race_id),
                        conn=conn,
                    )
                )

    all_items = [item for race in race_diagnostics for item in _list(race.get("items"))]
    item_counts = Counter(str(item.get("item_type")) for item in all_items)
    conflict_counts = Counter(str(item.get("conflict_type")) for item in all_items if item.get("conflict_type"))
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
            "manifest_packet": str(manifest_resolved),
            "insert_policy_packet": str(policy_path.expanduser().resolve(strict=False)) if policy_path else None,
            "lookup_packets": [str(path.expanduser().resolve(strict=False)) for path in lookup_paths],
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "summary": {
            "races_considered": len(race_diagnostics),
            "duplicate_guard_candidate_count": sum(
                _safe_int(row.get("duplicate_guard_candidate_count")) or 0 for row in race_diagnostics
            ),
            "race_deferred_candidate_count": sum(
                _safe_int(row.get("race_deferred_candidate_count")) or 0 for row in race_diagnostics
            ),
            "duplicate_guard_hit_count": sum(
                _safe_int(row.get("duplicate_guard_hit_count")) or 0 for row in race_diagnostics
            ),
            "matched_official_update_review_count": item_counts.get(
                "matched_official_finisher_db_update_policy_required", 0
            ),
            "duplicate_guard_conflict_item_count": item_counts.get(
                "duplicate_guard_box_or_name_conflict_policy_required", 0
            ),
            "extra_db_conflict_count": conflict_counts.get("occupied_box_extra_db_row_not_official", 0),
            "other_official_finisher_box_conflict_count": conflict_counts.get(
                "occupied_box_matches_other_official_finisher", 0
            ),
            "same_name_duplicate_conflict_count": conflict_counts.get("candidate_name_already_exists_in_db", 0),
            "item_type_counts": dict(sorted(item_counts.items())),
            "conflict_type_counts": dict(sorted(conflict_counts.items())),
            "safe_to_write_now_count": 0,
            "recommended_next_action": (
                "author_exact_update_policy_or_keep_duplicate_guard_races_excluded_from_apply"
            ),
        },
        "race_diagnostics": race_diagnostics,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "exact_candidate_allowlist_required": True,
            "update_policy_required_for_duplicate_guard_races": True,
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


def _item_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for race in _list(packet.get("race_diagnostics")):
        for item in _list(_mapping(race).get("items")):
            rows.append(dict(_mapping(item)))
    return rows


def _race_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(_mapping(row)) for row in _list(packet.get("race_diagnostics"))]


def write_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "non_terminal_duplicate_guard_update_reconciliation_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "non_terminal_duplicate_guard_update_reconciliation_items.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ITEM_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_item_csv_rows(packet))
    with (output_dir / "non_terminal_duplicate_guard_update_reconciliation_races.csv").open(
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
        "# Non-Terminal Duplicate Guard / Update Reconciliation",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Duplicate-guard candidates: `{summary.get('duplicate_guard_candidate_count')}`",
        f"- Same-race deferred candidates: `{summary.get('race_deferred_candidate_count')}`",
        f"- Duplicate-guard hits: `{summary.get('duplicate_guard_hit_count')}`",
        f"- Duplicate-guard conflict items: `{summary.get('duplicate_guard_conflict_item_count')}`",
        f"- Matched official update reviews: `{summary.get('matched_official_update_review_count')}`",
        f"- Other official finisher box conflicts: `{summary.get('other_official_finisher_box_conflict_count')}`",
        f"- Extra DB row conflicts: `{summary.get('extra_db_conflict_count')}`",
        f"- Same-name duplicate conflicts: `{summary.get('same_name_duplicate_conflict_count')}`",
        f"- Conflict type counts: `{summary.get('conflict_type_counts')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Gate",
        "",
        "These races still need exact update/delete/insert policy before any apply step. The packet only identifies conflicts and keeps all DB, label, model, and betting paths closed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-packet", required=True)
    parser.add_argument("--lookup-packet", action="append", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit(
            "refusing report-only duplicate-guard reconciliation while approval flags are set:"
            + ",".join(active)
        )
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_reconciliation_packet(
        manifest_packet_path=Path(args.manifest_packet),
        db_path=Path(args.db) if args.db else None,
        lookup_packet_paths=[Path(path) for path in args.lookup_packet] if args.lookup_packet else None,
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
