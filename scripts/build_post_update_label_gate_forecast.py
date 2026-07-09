#!/usr/bin/env python3
"""Forecast label-gate status after proposed non-terminal update policy.

This report-only helper simulates the current non-terminal update policy in
memory: proposed DB row updates are applied to a read-only snapshot, then
deferred missing-runner inserts are checked against the simulated rows. It also
keeps clean insert-only races in view and identifies the smallest fully
specified approval batch that could be worth running through a real post-apply
duplicate-guard check, gap review, and label preflight. It never writes DB rows,
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

from scripts.build_non_terminal_repair_apply_manifest_forecast import (  # noqa: E402
    REVIEWABLE_STATUS,
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
from scripts.build_non_terminal_update_policy_manifest import (  # noqa: E402
    SCHEMA_VERSION as UPDATE_POLICY_SCHEMA_VERSION,
)
from scripts.build_post_repair_label_gate_forecast_packet import (  # noqa: E402
    SCHEMA_VERSION as POST_REPAIR_FORECAST_SCHEMA_VERSION,
)
from scripts.build_missing_runner_insert_policy_packet import (  # noqa: E402
    FORBIDDEN_APPROVAL_ENV_VARS,
    SCHEMA_VERSION as INSERT_POLICY_SCHEMA_VERSION,
)


SCHEMA_VERSION = "post_update_label_gate_forecast_v1"
STATUS_OK = "REPORT_ONLY_POST_UPDATE_LABEL_GATE_FORECAST"
STATUS_FAILURES = "REPORT_ONLY_POST_UPDATE_LABEL_GATE_FORECAST_WITH_FAILURES"

GATE_BATCH_CANDIDATE = "POST_UPDATE_DUPLICATE_GUARD_CLEAR_LABEL_PREFLIGHT_REQUIRED"
GATE_INSERT_ONLY_UNMODELED_UPDATES = "INSERT_ONLY_DUPLICATE_GUARD_CLEAR_BUT_UPDATE_POLICY_MISSING"
GATE_DUPLICATE_STILL_BLOCKED = "POST_UPDATE_DUPLICATE_GUARD_STILL_BLOCKED"

CSV_FIELDS = [
    "race_id",
    "review_lane",
    "post_update_gate",
    "proposed_update_count",
    "deferred_insert_count",
    "insert_only_candidate_count",
    "metadata_policy_action_count",
    "total_repair_operation_count",
    "simulated_duplicate_guard_hit_count",
    "runner_set_complete_after_repair_forecast",
    "label_preflight_required",
    "batch_candidate_rank",
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


def _path_from_source(packet: Mapping[str, Any], key: str) -> Path | None:
    raw = _mapping(packet.get("source_evidence")).get(key)
    return Path(str(raw)) if raw else None


def _db_path_from_update_policy(packet: Mapping[str, Any]) -> Path | None:
    return _path_from_source(packet, "db")


def _manifest_path_from_update_policy(packet: Mapping[str, Any]) -> Path | None:
    return _path_from_source(packet, "manifest_packet")


def _policy_path_from_manifest(packet: Mapping[str, Any]) -> Path | None:
    return _path_from_source(packet, "insert_policy_packet")


def _post_repair_forecast_path_from_manifest(packet: Mapping[str, Any]) -> Path | None:
    return _path_from_source(packet, "post_repair_forecast_packet")


def _metadata_by_race(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in _list(packet.get("metadata_policy_rows")):
        row_map = _mapping(row)
        race_id = str(row_map.get("race_id") or "")
        if race_id:
            rows[race_id] = row_map
    return rows


def _forecast_by_race(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in _list(packet.get("forecast_rows")):
        row_map = _mapping(row)
        race_id = str(row_map.get("race_id") or "")
        if race_id:
            rows[race_id] = row_map
    return rows


def _manifest_race_by_id(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in _list(packet.get("race_manifest_rows")):
        row_map = _mapping(row)
        race_id = str(row_map.get("race_id") or "")
        if race_id:
            rows[race_id] = row_map
    return rows


def _reviewable_insert_candidates(packet: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in _list(packet.get("candidate_manifest_rows")):
        candidate_map = _mapping(candidate)
        if candidate_map.get("apply_manifest_status") != REVIEWABLE_STATUS:
            continue
        race_id = str(candidate_map.get("race_id") or "")
        if race_id:
            rows[race_id].append(candidate_map)
    return rows


def _updates_by_race(packet: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for update in _list(packet.get("proposed_update_rows")):
        update_map = _mapping(update)
        race_id = str(update_map.get("race_id") or "")
        if race_id:
            rows[race_id].append(update_map)
    return rows


def _deferred_inserts_by_race(packet: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for insert in _list(packet.get("deferred_insert_rows")):
        insert_map = _mapping(insert)
        race_id = str(insert_map.get("race_id") or "")
        if race_id:
            rows[race_id].append(insert_map)
    return rows


def _race_rows(conn: sqlite3.Connection, race_id: str) -> list[dict[str, Any]]:
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


def _apply_updates(rows: list[dict[str, Any]], updates: Sequence[Mapping[str, Any]]) -> None:
    by_rowid = {str(row.get("db_rowid")): row for row in rows}
    for update in updates:
        target = by_rowid.get(str(_mapping(update).get("db_rowid")))
        if not target:
            continue
        target["box_number"] = _safe_int(_mapping(update).get("proposed_box_number"))
        target["finish_position"] = _safe_int(_mapping(update).get("proposed_finish_position"))
        target["placing"] = _safe_int(_mapping(update).get("proposed_placing"))
        target["scraped_finish_position"] = _mapping(update).get("proposed_scraped_finish_position")


def _insert_values(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(candidate.get("insert_values"))


def _insert_name(candidate: Mapping[str, Any]) -> str:
    values = _insert_values(candidate)
    return str(values.get("dog_name") or values.get("dog_clean_name") or candidate.get("official_dog_name") or "")


def _insert_clean_name(candidate: Mapping[str, Any]) -> str:
    values = _insert_values(candidate)
    return str(values.get("dog_clean_name") or values.get("dog_name") or candidate.get("official_dog_name") or "")


def _insert_box(candidate: Mapping[str, Any]) -> int | None:
    values = _insert_values(candidate)
    return _safe_int(values.get("box_number") or candidate.get("box_number"))


def _insert_finish(candidate: Mapping[str, Any]) -> int | None:
    values = _insert_values(candidate)
    return _safe_int(values.get("finish_position") or candidate.get("finish_position"))


def _duplicate_hits(rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    dog_name = _insert_name(candidate)
    dog_clean_name = _insert_clean_name(candidate)
    box = _insert_box(candidate)
    return [
        row
        for row in rows
        if row.get("dog_name") == dog_name
        or row.get("dog_clean_name") == dog_clean_name
        or _safe_int(row.get("box_number")) == box
    ]


def _append_simulated_insert(rows: list[dict[str, Any]], candidate: Mapping[str, Any]) -> None:
    values = dict(_insert_values(candidate))
    rows.append(
        {
            "db_rowid": f"SIMULATED::{candidate.get('candidate_id')}",
            "dog_name": values.get("dog_name") or candidate.get("official_dog_name"),
            "dog_clean_name": values.get("dog_clean_name") or candidate.get("official_dog_name"),
            "box_number": _insert_box(candidate),
            "finish_position": _insert_finish(candidate),
            "placing": _safe_int(values.get("placing") or _insert_finish(candidate)),
            "scraped_finish_position": values.get("scraped_finish_position"),
            "data_source": values.get("data_source"),
        }
    )


def _simulate_duplicate_guard(
    *,
    conn: sqlite3.Connection,
    race_id: str,
    updates: Sequence[Mapping[str, Any]],
    inserts: Sequence[Mapping[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    simulated_rows = _race_rows(conn, race_id)
    _apply_updates(simulated_rows, updates)
    failures = []
    for candidate in inserts:
        hits = _duplicate_hits(simulated_rows, candidate)
        if hits:
            failures.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "hit_count": len(hits),
                    "hit_rows": [dict(row) for row in hits],
                }
            )
            continue
        _append_simulated_insert(simulated_rows, candidate)
    return sum(_safe_int(row.get("hit_count")) or 0 for row in failures), failures


def _metadata_action_count(row: Mapping[str, Any] | None) -> int:
    if not row:
        return 0
    row_map = _mapping(row)
    return 1 if row_map.get("after_patch") or row_map.get("deferred_policy_candidates") else 0


def _build_forecast_rows(
    *,
    conn: sqlite3.Connection,
    update_policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    insert_policy: Mapping[str, Any],
    post_repair_forecast: Mapping[str, Any],
) -> list[dict[str, Any]]:
    updates_by_race = _updates_by_race(update_policy)
    deferred_by_race = _deferred_inserts_by_race(update_policy)
    clean_candidates_by_race = _reviewable_insert_candidates(manifest)
    manifest_races = _manifest_race_by_id(manifest)
    metadata_rows = _metadata_by_race(insert_policy)
    post_repair_rows = _forecast_by_race(post_repair_forecast)

    race_ids = sorted(set(updates_by_race) | set(deferred_by_race) | set(clean_candidates_by_race))
    rows = []
    for race_id in race_ids:
        updates = updates_by_race.get(race_id, [])
        deferred = deferred_by_race.get(race_id, [])
        clean = clean_candidates_by_race.get(race_id, [])
        inserts = list(deferred) if deferred else list(clean)
        duplicate_hit_count, duplicate_failures = _simulate_duplicate_guard(
            conn=conn,
            race_id=race_id,
            updates=updates,
            inserts=inserts,
        )
        manifest_race = _mapping(manifest_races.get(race_id))
        post_repair = _mapping(post_repair_rows.get(race_id))
        metadata_count = _metadata_action_count(metadata_rows.get(race_id))
        changed_update_count = _safe_int(manifest_race.get("changed_dog_update_candidate_count")) or 0
        update_count = len(updates)
        deferred_count = len(deferred)
        clean_count = len(clean) if not deferred else 0
        operation_count = update_count + deferred_count + clean_count + metadata_count
        runner_complete = bool(post_repair.get("runner_set_complete_after_proposed_repair"))

        if duplicate_hit_count:
            gate = GATE_DUPLICATE_STILL_BLOCKED
            next_action = "resolve_simulated_duplicate_guard_failures_before_any_apply"
        elif update_count and deferred_count:
            gate = GATE_BATCH_CANDIDATE
            next_action = "candidate_for_small_approval_batch_then_rerun_duplicate_guard_gap_review_label_preflight"
        else:
            gate = GATE_INSERT_ONLY_UNMODELED_UPDATES
            next_action = "build_exact_update_policy_for_existing_runner_drift_before_label_expansion"

        rows.append(
            {
                "race_id": race_id,
                "review_lane": manifest_race.get("review_lane") or post_repair.get("review_lane"),
                "post_update_gate": gate,
                "proposed_update_count": update_count,
                "deferred_insert_count": deferred_count,
                "insert_only_candidate_count": clean_count,
                "metadata_policy_action_count": metadata_count,
                "changed_update_count_from_manifest": changed_update_count,
                "total_repair_operation_count": operation_count,
                "simulated_duplicate_guard_hit_count": duplicate_hit_count,
                "simulated_duplicate_guard_failures": duplicate_failures,
                "runner_set_complete_after_repair_forecast": runner_complete,
                "label_preflight_required": True,
                "direct_label_preflight_ready_now": False,
                "batch_candidate_rank": None,
                "recommended_next_action": next_action,
            }
        )
    batch_rows = [
        row
        for row in rows
        if row.get("post_update_gate") == GATE_BATCH_CANDIDATE
        and row.get("runner_set_complete_after_repair_forecast") is True
    ]
    batch_rows.sort(
        key=lambda row: (
            _safe_int(row.get("total_repair_operation_count")) or 999999,
            str(row.get("race_id") or ""),
        )
    )
    for index, row in enumerate(batch_rows, start=1):
        row["batch_candidate_rank"] = index
    return rows


def build_post_update_forecast_packet(
    *,
    update_policy_packet_path: Path,
    manifest_packet_path: Path | None = None,
    insert_policy_packet_path: Path | None = None,
    post_repair_forecast_packet_path: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    update_resolved = update_policy_packet_path.expanduser().resolve()
    update_policy = _load_json(update_resolved)
    failures: list[str] = []
    _validate_packet(
        packet=update_policy,
        expected_schema=UPDATE_POLICY_SCHEMA_VERSION,
        packet_name="update_policy_packet",
        failures=failures,
    )

    manifest_path = manifest_packet_path or _manifest_path_from_update_policy(update_policy)
    if manifest_path is None:
        failures.append("manifest_path_missing")
        manifest_path = Path("DATA_MISSING")
    manifest_resolved = manifest_path.expanduser().resolve(strict=False)
    manifest: Mapping[str, Any] = {}
    if manifest_resolved.name != "DATA_MISSING":
        manifest = _load_json(manifest_resolved)
        _validate_packet(
            packet=manifest,
            expected_schema=MANIFEST_SCHEMA_VERSION,
            packet_name="manifest_packet",
            failures=failures,
        )

    insert_policy_path = insert_policy_packet_path or _policy_path_from_manifest(manifest)
    if insert_policy_path is None:
        failures.append("insert_policy_path_missing")
        insert_policy_path = Path("DATA_MISSING")
    insert_policy_resolved = insert_policy_path.expanduser().resolve(strict=False)
    insert_policy: Mapping[str, Any] = {}
    if insert_policy_resolved.name != "DATA_MISSING":
        insert_policy = _load_json(insert_policy_resolved)
        _validate_packet(
            packet=insert_policy,
            expected_schema=INSERT_POLICY_SCHEMA_VERSION,
            packet_name="insert_policy_packet",
            failures=failures,
        )

    forecast_path = post_repair_forecast_packet_path or _post_repair_forecast_path_from_manifest(manifest)
    if forecast_path is None:
        failures.append("post_repair_forecast_path_missing")
        forecast_path = Path("DATA_MISSING")
    forecast_resolved = forecast_path.expanduser().resolve(strict=False)
    post_repair_forecast: Mapping[str, Any] = {}
    if forecast_resolved.name != "DATA_MISSING":
        post_repair_forecast = _load_json(forecast_resolved)
        _validate_packet(
            packet=post_repair_forecast,
            expected_schema=POST_REPAIR_FORECAST_SCHEMA_VERSION,
            packet_name="post_repair_forecast_packet",
            failures=failures,
        )

    resolved_db = db_path or _path_from_source(update_policy, "db")
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
    forecast_rows: list[dict[str, Any]] = []
    if db_resolved.name != "DATA_MISSING":
        with _connect_read_only(db_resolved) as conn:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            db_state["quick_check"] = quick_check[0] if quick_check else None
            if db_state["quick_check"] != "ok":
                failures.append("db_quick_check_failed")
            forecast_rows = _build_forecast_rows(
                conn=conn,
                update_policy=update_policy,
                manifest=manifest,
                insert_policy=insert_policy,
                post_repair_forecast=post_repair_forecast,
            )

    gate_counts = Counter(str(row.get("post_update_gate")) for row in forecast_rows)
    ranked = [row for row in forecast_rows if row.get("batch_candidate_rank")]
    ranked.sort(key=lambda row: _safe_int(row.get("batch_candidate_rank")) or 999999)
    smallest = ranked[0] if ranked else None
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
            "update_policy_packet": str(update_resolved),
            "manifest_packet": str(manifest_resolved),
            "insert_policy_packet": str(insert_policy_resolved),
            "post_repair_forecast_packet": str(forecast_resolved),
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "summary": {
            "races_considered": len(forecast_rows),
            "post_update_gate_counts": dict(sorted(gate_counts.items())),
            "batch_candidate_count": gate_counts.get(GATE_BATCH_CANDIDATE, 0),
            "insert_only_unmodeled_update_count": gate_counts.get(GATE_INSERT_ONLY_UNMODELED_UPDATES, 0),
            "simulated_duplicate_still_blocked_count": gate_counts.get(GATE_DUPLICATE_STILL_BLOCKED, 0),
            "runner_set_complete_after_repair_forecast_count": sum(
                1 for row in forecast_rows if row.get("runner_set_complete_after_repair_forecast") is True
            ),
            "direct_label_preflight_ready_now_count": 0,
            "label_preflight_required_count": len(forecast_rows),
            "smallest_batch_race_id": _mapping(smallest).get("race_id"),
            "smallest_batch_total_repair_operation_count": _mapping(smallest).get(
                "total_repair_operation_count"
            ),
            "smallest_batch_update_count": _mapping(smallest).get("proposed_update_count"),
            "smallest_batch_insert_count": _mapping(smallest).get("deferred_insert_count"),
            "smallest_batch_metadata_action_count": _mapping(smallest).get("metadata_policy_action_count"),
            "safe_to_write_now_count": 0,
            "recommended_next_action": (
                "review_smallest_batch_then_require_explicit_db_backup_apply_approval_or_keep_report_only"
            ),
        },
        "forecast_rows": forecast_rows,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "exact_update_allowlist_required": True,
            "exact_insert_allowlist_required": True,
            "metadata_policy_approval_required": True,
            "rerun_duplicate_guard_after_apply_required": True,
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


def _csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _list(packet.get("forecast_rows")):
        row_map = dict(_mapping(row))
        rows.append(row_map)
    return rows


def write_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "post_update_label_gate_forecast_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "post_update_label_gate_forecast.csv").open(
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
        "# Post-Update Label Gate Forecast",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Post-update gate counts: `{summary.get('post_update_gate_counts')}`",
        f"- Batch candidates: `{summary.get('batch_candidate_count')}`",
        f"- Insert-only races still missing update policy: `{summary.get('insert_only_unmodeled_update_count')}`",
        f"- Simulated duplicate still blocked: `{summary.get('simulated_duplicate_still_blocked_count')}`",
        f"- Runner-set complete after repair forecast: `{summary.get('runner_set_complete_after_repair_forecast_count')}`",
        f"- Direct label-preflight ready now: `{summary.get('direct_label_preflight_ready_now_count')}`",
        f"- Smallest batch race: `{summary.get('smallest_batch_race_id')}`",
        f"- Smallest batch operations: `{summary.get('smallest_batch_total_repair_operation_count')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Gate",
        "",
        "This is a forecast only. Even the smallest batch still requires explicit DB repair approval, a fresh backup, real duplicate-guard rechecks, post-apply gap review, and official label preflight before any label expansion or retraining.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-policy-packet", required=True)
    parser.add_argument("--manifest-packet", default=None)
    parser.add_argument("--insert-policy-packet", default=None)
    parser.add_argument("--post-repair-forecast-packet", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit(
            "refusing report-only post-update label gate forecast while approval flags are set:"
            + ",".join(active)
        )
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_post_update_forecast_packet(
        update_policy_packet_path=Path(args.update_policy_packet),
        manifest_packet_path=Path(args.manifest_packet) if args.manifest_packet else None,
        insert_policy_packet_path=Path(args.insert_policy_packet) if args.insert_policy_packet else None,
        post_repair_forecast_packet_path=Path(args.post_repair_forecast_packet)
        if args.post_repair_forecast_packet
        else None,
        db_path=Path(args.db) if args.db else None,
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
