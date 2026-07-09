#!/usr/bin/env python3
"""Build a no-write non-terminal repair apply-manifest forecast.

This helper narrows the missing-runner insert policy to terminal-free races
whose post-repair runner set is forecast complete. It rechecks each candidate's
duplicate guard against the current DB in read-only mode, excludes terminal
manual-reconciliation races, and emits an approval-gated candidate manifest. It
does not write DB rows, labels, snapshots, manifests, datasets, models,
registries, TGR settings, betting decisions, or EV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_missing_runner_insert_policy_packet import (  # noqa: E402
    FORBIDDEN_APPROVAL_ENV_VARS,
    WRITES_PERFORMED,
    _assert_output_dir_safe,
)


SCHEMA_VERSION = "non_terminal_repair_apply_manifest_forecast_v1"
POLICY_SCHEMA_VERSION = "missing_runner_insert_policy_packet_v1"
FORECAST_SCHEMA_VERSION = "post_repair_label_gate_forecast_packet_v1"
TERMINAL_MANUAL_SCHEMA_VERSION = "terminal_manual_reconciliation_packet_v1"
STATUS_OK = "REPORT_ONLY_NON_TERMINAL_REPAIR_APPLY_MANIFEST_FORECAST"
STATUS_FAILURES = "REPORT_ONLY_NON_TERMINAL_REPAIR_APPLY_MANIFEST_FORECAST_WITH_FAILURES"
TERMINAL_FREE_GATE = "POST_REPAIR_RUNNER_SET_COMPLETE_TERMINAL_FREE_RECHECK_REQUIRED"

REVIEWABLE_STATUS = "CANDIDATE_FOR_OPERATOR_REVIEW_AFTER_BACKUP_AND_EXPLICIT_APPROVAL"
TERMINAL_EXCLUDED_STATUS = "EXCLUDED_TERMINAL_SCOPE_MANUAL_POLICY_REQUIRED"
DUPLICATE_EXCLUDED_STATUS = "EXCLUDED_DUPLICATE_GUARD_HIT"
RACE_BLOCKED_STATUS = "DEFERRED_RACE_HAS_OTHER_DUPLICATE_GUARD_HIT"
FORECAST_EXCLUDED_STATUS = "EXCLUDED_FORECAST_GATE_NOT_TERMINAL_FREE_COMPLETE"
MISSING_FORECAST_STATUS = "EXCLUDED_FORECAST_ROW_MISSING"

CANDIDATE_CSV_FIELDS = [
    "candidate_id",
    "race_id",
    "review_lane",
    "field_scope",
    "official_dog_name",
    "name_key",
    "box_number",
    "finish_position",
    "forecast_gate",
    "duplicate_guard_hit_count",
    "apply_manifest_status",
    "safe_to_apply_now",
    "required_before_apply",
    "source_url",
]

RACE_CSV_FIELDS = [
    "race_id",
    "review_lane",
    "forecast_gate",
    "candidate_count",
    "reviewable_candidate_count",
    "excluded_candidate_count",
    "duplicate_guard_hit_count",
    "terminal_manual_excluded",
    "race_manifest_status",
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


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pipe(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values if value not in (None, ""))


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


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


def _db_path_from_policy(packet: Mapping[str, Any]) -> Path | None:
    raw = _mapping(packet.get("source_evidence")).get("db")
    return Path(str(raw)) if raw else None


def _forecast_by_race(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in _list(packet.get("forecast_rows")):
        row_map = _mapping(row)
        race_id = str(row_map.get("race_id") or "")
        if race_id:
            rows[race_id] = row_map
    return rows


def _terminal_race_ids(packet: Mapping[str, Any]) -> set[str]:
    race_ids = set()
    for row in _list(packet.get("race_diagnostics")):
        row_map = _mapping(row)
        race_id = str(row_map.get("race_id") or "")
        if race_id and row_map.get("manual_policy_lane") == "TERMINAL_SCOPE_MANUAL_RECONCILIATION_REQUIRED":
            race_ids.add(race_id)
    return race_ids


def _candidate_insert_values(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(candidate.get("insert_values"))


def _candidate_box(candidate: Mapping[str, Any]) -> int | None:
    values = _candidate_insert_values(candidate)
    return _safe_int(candidate.get("box_number") or values.get("box_number"))


def _duplicate_guard_hits(conn: sqlite3.Connection, candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = _candidate_insert_values(candidate)
    race_id = str(candidate.get("race_id") or values.get("race_id") or "")
    dog_name = str(values.get("dog_name") or candidate.get("official_dog_name") or "")
    dog_clean_name = str(values.get("dog_clean_name") or candidate.get("official_dog_name") or "")
    box_number = _candidate_box(candidate)
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT dog_name, dog_clean_name, box_number, finish_position,
                   placing, scraped_finish_position, data_source
            FROM dog_race_data
            WHERE race_id = ?
              AND (dog_name = ? OR dog_clean_name = ? OR CAST(box_number AS INTEGER) = ?)
            ORDER BY box_number, dog_name
            """,
            (race_id, dog_name, dog_clean_name, box_number),
        ).fetchall()
    ]


def _required_before_apply(forecast_row: Mapping[str, Any]) -> list[str]:
    required = [
        "explicit_operator_approval_required",
        "db_backup_required_before_apply",
        "exact_candidate_allowlist_required",
        "rerun_duplicate_guard_immediately_before_apply",
        "post_apply_gap_review_required",
        "post_apply_label_preflight_required",
        "no_label_expansion_until_post_apply_preflight_passes",
    ]
    if "field_size_metadata_policy_required" in _list(forecast_row.get("remaining_blockers")):
        required.append("field_size_metadata_policy_required")
    return required


def _candidate_status(
    *,
    candidate: Mapping[str, Any],
    forecast_row: Mapping[str, Any] | None,
    terminal_race_ids: set[str],
    duplicate_hits: Sequence[Mapping[str, Any]],
) -> str:
    race_id = str(candidate.get("race_id") or "")
    if forecast_row is None:
        return MISSING_FORECAST_STATUS
    if race_id in terminal_race_ids or _safe_int(forecast_row.get("terminal_status_count")):
        return TERMINAL_EXCLUDED_STATUS
    if forecast_row.get("forecast_gate") != TERMINAL_FREE_GATE:
        return FORECAST_EXCLUDED_STATUS
    if duplicate_hits:
        return DUPLICATE_EXCLUDED_STATUS
    return REVIEWABLE_STATUS


def _manifest_candidate(
    *,
    candidate: Mapping[str, Any],
    forecast_row: Mapping[str, Any] | None,
    terminal_race_ids: set[str],
    duplicate_hits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = _candidate_insert_values(candidate)
    forecast = forecast_row or {}
    status = _candidate_status(
        candidate=candidate,
        forecast_row=forecast_row,
        terminal_race_ids=terminal_race_ids,
        duplicate_hits=duplicate_hits,
    )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "race_id": candidate.get("race_id") or values.get("race_id"),
        "review_lane": candidate.get("review_lane") or forecast.get("review_lane"),
        "field_scope": candidate.get("field_scope") or forecast.get("field_scope"),
        "source_url": candidate.get("source_url"),
        "official_dog_name": candidate.get("official_dog_name"),
        "name_key": candidate.get("name_key"),
        "box_number": _candidate_box(candidate),
        "finish_position": _safe_int(candidate.get("finish_position") or values.get("finish_position")),
        "insert_columns": _list(candidate.get("insert_columns")),
        "insert_values": dict(values),
        "forecast_gate": forecast.get("forecast_gate"),
        "runner_set_complete_after_proposed_repair": forecast.get("runner_set_complete_after_proposed_repair"),
        "terminal_status_count": _safe_int(forecast.get("terminal_status_count")) or 0,
        "changed_dog_update_candidate_count": _safe_int(forecast.get("changed_dog_update_candidate_count")) or 0,
        "metadata_update_candidate_count": _safe_int(forecast.get("metadata_update_candidate_count")) or 0,
        "duplicate_guard_hit_count": len(duplicate_hits),
        "duplicate_guard_hits": [dict(row) for row in duplicate_hits],
        "apply_manifest_status": status,
        "safe_to_apply_now": False,
        "required_before_apply": _required_before_apply(forecast),
    }


def _race_rows(
    *,
    forecast_rows: Mapping[str, Mapping[str, Any]],
    terminal_race_ids: set[str],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_race[str(candidate.get("race_id") or "")].append(candidate)
    rows = []
    for race_id, forecast in sorted(forecast_rows.items()):
        race_candidates = by_race.get(race_id, [])
        reviewable = sum(1 for row in race_candidates if row.get("apply_manifest_status") == REVIEWABLE_STATUS)
        excluded = len(race_candidates) - reviewable
        duplicate_hits = sum(_safe_int(row.get("duplicate_guard_hit_count")) or 0 for row in race_candidates)
        terminal = race_id in terminal_race_ids
        if terminal:
            status = "RACE_EXCLUDED_TERMINAL_SCOPE_MANUAL_POLICY_REQUIRED"
        elif forecast.get("forecast_gate") != TERMINAL_FREE_GATE:
            status = "RACE_EXCLUDED_FORECAST_GATE_NOT_TERMINAL_FREE_COMPLETE"
        elif duplicate_hits:
            status = "RACE_EXCLUDED_DUPLICATE_GUARD_HIT"
        elif race_candidates and reviewable == len(race_candidates):
            status = "RACE_CANDIDATE_FOR_OPERATOR_REVIEW_AFTER_BACKUP_AND_EXPLICIT_APPROVAL"
        else:
            status = "RACE_HAS_NO_INSERT_CANDIDATES_IN_MANIFEST"
        rows.append(
            {
                "race_id": race_id,
                "review_lane": forecast.get("review_lane"),
                "forecast_gate": forecast.get("forecast_gate"),
                "candidate_count": len(race_candidates),
                "reviewable_candidate_count": reviewable,
                "excluded_candidate_count": excluded,
                "duplicate_guard_hit_count": duplicate_hits,
                "terminal_manual_excluded": terminal,
                "race_manifest_status": status,
                "changed_dog_update_candidate_count": _safe_int(
                    forecast.get("changed_dog_update_candidate_count")
                )
                or 0,
                "metadata_update_candidate_count": _safe_int(forecast.get("metadata_update_candidate_count")) or 0,
                "remaining_blockers": _list(forecast.get("remaining_blockers")),
            }
        )
    return rows


def _defer_mixed_race_candidates(candidates: list[dict[str, Any]]) -> None:
    blocked_races = {
        str(candidate.get("race_id") or "")
        for candidate in candidates
        if candidate.get("apply_manifest_status") == DUPLICATE_EXCLUDED_STATUS
    }
    for candidate in candidates:
        if (
            candidate.get("apply_manifest_status") == REVIEWABLE_STATUS
            and str(candidate.get("race_id") or "") in blocked_races
        ):
            candidate["apply_manifest_status"] = RACE_BLOCKED_STATUS
            candidate["required_before_apply"] = sorted(
                set(_list(candidate.get("required_before_apply")) + ["resolve_other_race_duplicate_guard_hits"])
            )


def build_manifest_forecast_packet(
    *,
    insert_policy_packet_path: Path,
    post_repair_forecast_packet_path: Path,
    terminal_manual_packet_path: Path,
    db_path: Path | None = None,
) -> dict[str, Any]:
    policy_resolved = insert_policy_packet_path.expanduser().resolve()
    forecast_resolved = post_repair_forecast_packet_path.expanduser().resolve()
    terminal_resolved = terminal_manual_packet_path.expanduser().resolve()
    policy_packet = _load_json(policy_resolved)
    forecast_packet = _load_json(forecast_resolved)
    terminal_packet = _load_json(terminal_resolved)
    failures: list[str] = []
    _validate_packet(
        packet=policy_packet,
        expected_schema=POLICY_SCHEMA_VERSION,
        packet_name="insert_policy_packet",
        failures=failures,
    )
    _validate_packet(
        packet=forecast_packet,
        expected_schema=FORECAST_SCHEMA_VERSION,
        packet_name="post_repair_forecast_packet",
        failures=failures,
    )
    _validate_packet(
        packet=terminal_packet,
        expected_schema=TERMINAL_MANUAL_SCHEMA_VERSION,
        packet_name="terminal_manual_packet",
        failures=failures,
    )

    resolved_db = db_path or _db_path_from_policy(policy_packet)
    if resolved_db is None:
        failures.append("db_path_missing")
        resolved_db = Path("DATA_MISSING")
    db_resolved = resolved_db.expanduser().resolve(strict=False)

    forecast_rows = _forecast_by_race(forecast_packet)
    terminal_race_ids = _terminal_race_ids(terminal_packet)
    db_state: dict[str, Any] = {
        "db_path": str(db_resolved),
        "quick_check": None,
        "read_only": True,
        "query_only": True,
    }

    manifest_candidates = []
    if db_resolved.name != "DATA_MISSING":
        with _connect_read_only(db_resolved) as conn:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            db_state["quick_check"] = quick_check[0] if quick_check else None
            if db_state["quick_check"] != "ok":
                failures.append("db_quick_check_failed")
            for candidate in _list(policy_packet.get("candidate_rows")):
                candidate_map = _mapping(candidate)
                race_id = str(candidate_map.get("race_id") or "")
                manifest_candidates.append(
                    _manifest_candidate(
                        candidate=candidate_map,
                        forecast_row=forecast_rows.get(race_id),
                        terminal_race_ids=terminal_race_ids,
                        duplicate_hits=_duplicate_guard_hits(conn, candidate_map),
                    )
                )
    _defer_mixed_race_candidates(manifest_candidates)

    race_manifest_rows = _race_rows(
        forecast_rows=forecast_rows,
        terminal_race_ids=terminal_race_ids,
        candidates=manifest_candidates,
    )
    candidate_status_counts = Counter(str(row.get("apply_manifest_status")) for row in manifest_candidates)
    race_status_counts = Counter(str(row.get("race_manifest_status")) for row in race_manifest_rows)
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
            "insert_policy_packet": str(policy_resolved),
            "post_repair_forecast_packet": str(forecast_resolved),
            "terminal_manual_packet": str(terminal_resolved),
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "summary": {
            "races_considered": len(forecast_rows),
            "candidate_count": len(manifest_candidates),
            "reviewable_after_backup_and_explicit_approval_candidate_count": candidate_status_counts.get(
                REVIEWABLE_STATUS, 0
            ),
            "excluded_terminal_candidate_count": candidate_status_counts.get(TERMINAL_EXCLUDED_STATUS, 0),
            "excluded_duplicate_guard_hit_candidate_count": candidate_status_counts.get(
                DUPLICATE_EXCLUDED_STATUS, 0
            ),
            "deferred_race_blocked_candidate_count": candidate_status_counts.get(RACE_BLOCKED_STATUS, 0),
            "excluded_forecast_gate_candidate_count": candidate_status_counts.get(FORECAST_EXCLUDED_STATUS, 0),
            "excluded_missing_forecast_candidate_count": candidate_status_counts.get(MISSING_FORECAST_STATUS, 0),
            "candidate_status_counts": dict(sorted(candidate_status_counts.items())),
            "race_status_counts": dict(sorted(race_status_counts.items())),
            "terminal_manual_race_exclusion_count": len(terminal_race_ids),
            "non_terminal_reviewable_race_count": race_status_counts.get(
                "RACE_CANDIDATE_FOR_OPERATOR_REVIEW_AFTER_BACKUP_AND_EXPLICIT_APPROVAL", 0
            ),
            "duplicate_guard_hit_count": sum(
                _safe_int(row.get("duplicate_guard_hit_count")) or 0 for row in manifest_candidates
            ),
            "safe_to_write_now_count": 0,
            "recommended_next_action": (
                "operator_review_non_terminal_manifest_then_explicitly_approve_backup_and_apply_or_keep_report_only"
            ),
        },
        "race_manifest_rows": race_manifest_rows,
        "candidate_manifest_rows": manifest_candidates,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "exact_candidate_allowlist_required": True,
            "duplicate_guard_required_immediately_before_each_insert": True,
            "terminal_scope_excluded": True,
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


def _candidate_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _list(packet.get("candidate_manifest_rows")):
        row_map = dict(_mapping(row))
        row_map["required_before_apply"] = _pipe(_list(row_map.get("required_before_apply")))
        rows.append(row_map)
    return rows


def _race_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(_mapping(row)) for row in _list(packet.get("race_manifest_rows"))]


def write_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "non_terminal_repair_apply_manifest_forecast_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "non_terminal_repair_apply_manifest_candidates.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_candidate_csv_rows(packet))
    with (output_dir / "non_terminal_repair_apply_manifest_races.csv").open(
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
        "# Non-Terminal Repair Apply Manifest Forecast",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Candidate rows: `{summary.get('candidate_count')}`",
        f"- Reviewable after backup and explicit approval: `{summary.get('reviewable_after_backup_and_explicit_approval_candidate_count')}`",
        f"- Excluded terminal-scope candidates: `{summary.get('excluded_terminal_candidate_count')}`",
        f"- Excluded duplicate-guard candidates: `{summary.get('excluded_duplicate_guard_hit_candidate_count')}`",
        f"- Deferred by other race duplicate-guard hits: `{summary.get('deferred_race_blocked_candidate_count')}`",
        f"- Terminal manual race exclusions: `{summary.get('terminal_manual_race_exclusion_count')}`",
        f"- Non-terminal reviewable races: `{summary.get('non_terminal_reviewable_race_count')}`",
        f"- Duplicate guard hits: `{summary.get('duplicate_guard_hit_count')}`",
        f"- Candidate status counts: `{summary.get('candidate_status_counts')}`",
        f"- Race status counts: `{summary.get('race_status_counts')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Gate",
        "",
        "This is still a forecast, not an apply plan. Any apply step requires explicit operator approval, a current DB backup, an exact candidate allowlist, duplicate-guard rechecks immediately before each insert, post-apply gap review, and label preflight before label expansion or retraining.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insert-policy-packet", required=True)
    parser.add_argument("--post-repair-forecast-packet", required=True)
    parser.add_argument("--terminal-manual-packet", required=True)
    parser.add_argument("--db", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit(
            "refusing report-only non-terminal apply-manifest forecast while approval flags are set:"
            + ",".join(active)
        )
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_manifest_forecast_packet(
        insert_policy_packet_path=Path(args.insert_policy_packet),
        post_repair_forecast_packet_path=Path(args.post_repair_forecast_packet),
        terminal_manual_packet_path=Path(args.terminal_manual_packet),
        db_path=Path(args.db) if args.db else None,
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
