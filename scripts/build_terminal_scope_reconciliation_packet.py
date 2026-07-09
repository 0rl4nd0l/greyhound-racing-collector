#!/usr/bin/env python3
"""Build a no-write terminal-scope reconciliation packet.

The post-repair forecast can identify terminal/exclusion-scope races where a
plain missing-runner insert allowlist would overrun the official finisher count.
This helper inspects those races against existing official lookup packets and
current DB rows in read-only mode. It emits manual reconciliation requirements
only; it never writes DB rows, labels, snapshots, manifests, datasets, models,
registries, TGR settings, betting decisions, or EV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "terminal_scope_reconciliation_packet_v1"
FORECAST_SCHEMA_VERSION = "post_repair_label_gate_forecast_packet_v1"
POLICY_SCHEMA_VERSION = "missing_runner_insert_policy_packet_v1"
STATUS_OK = "REPORT_ONLY_TERMINAL_SCOPE_RECONCILIATION_PACKET"
STATUS_FAILURES = "REPORT_ONLY_TERMINAL_SCOPE_RECONCILIATION_PACKET_WITH_FAILURES"

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
    "reconciliation_lane",
    "official_finisher_count",
    "terminal_status_count",
    "current_db_runner_count",
    "missing_official_finisher_count",
    "extra_db_name_count",
    "naive_insert_candidate_count",
    "insert_candidate_box_conflict_count",
    "insert_candidate_no_box_conflict_count",
    "terminal_box_db_row_count",
    "naive_post_repair_runner_count",
    "forecast_gate",
    "safe_for_insert_only",
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


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _name_key(value: Any) -> str:
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(value or "").strip())
    text = text.replace('"', "").replace("'", "").replace("`", "")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


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


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _fetch_db_rows(conn: sqlite3.Connection, race_id: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT dog_name, dog_clean_name, box_number, finish_position,
                   placing, scraped_finish_position, data_source
            FROM dog_race_data
            WHERE race_id = ?
            ORDER BY box_number, dog_name
            """,
            (race_id,),
        ).fetchall()
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
    for key, value in _mapping(packet.get("writes_performed")).items():
        if value is not False:
            failures.append(f"{packet_name}_write_flag_true:{key}")


def _validate_lookup_packet(path: Path, packet: Mapping[str, Any], failures: list[str]) -> None:
    if packet.get("status") != "REPORT_ONLY":
        failures.append(f"lookup_packet_status_not_report_only:{path}")
    for key, value in _mapping(packet.get("writes_performed")).items():
        if key == "official_fetch":
            continue
        if value is not False:
            failures.append(f"lookup_packet_write_flag_true:{path}:{key}")


def _lookup_index(paths: Sequence[Path], failures: list[str]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        packet = _load_json(resolved)
        _validate_lookup_packet(resolved, packet, failures)
        for row in _list(packet.get("results")):
            row_map = _mapping(row)
            race_id = str(row_map.get("legacy_race_id") or "")
            if race_id:
                index.setdefault(race_id, row_map)
    return index


def _candidate_rows_by_race(policy_packet: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in _list(policy_packet.get("candidate_rows")):
        candidate_map = _mapping(candidate)
        race_id = str(candidate_map.get("race_id") or "")
        if race_id:
            rows[race_id].append(candidate_map)
    return rows


def _target_forecast_rows(forecast_packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = []
    for row in _list(forecast_packet.get("forecast_rows")):
        row_map = _mapping(row)
        if (
            row_map.get("forecast_gate") == "POST_REPAIR_FORECAST_STILL_INCOMPLETE"
            or (_safe_int(row_map.get("terminal_status_count")) or 0) > 0
        ):
            rows.append(row_map)
    return rows


def _box(value: Any) -> int | None:
    parsed = _safe_int(value)
    return parsed if parsed and parsed > 0 else None


def _official_rows(lookup: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _list(lookup.get("official_runner_rows")) or _list(lookup.get("positions"))
    return [
        {
            "dog_name": _mapping(row).get("dog_name"),
            "name_key": _name_key(_mapping(row).get("dog_name")),
            "box_number": _box(_mapping(row).get("box_number")),
            "finish_position": _safe_int(_mapping(row).get("finish_position")),
        }
        for row in rows
    ]


def _terminal_rows(lookup: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "box_number": _box(_mapping(row).get("box_number")),
            "status": _mapping(row).get("status"),
            "dog_name": _mapping(row).get("dog_name"),
            "name_key": _name_key(_mapping(row).get("dog_name")),
        }
        for row in _list(lookup.get("terminal_statuses"))
    ]


def _diagnose_race(
    *,
    forecast_row: Mapping[str, Any],
    lookup: Mapping[str, Any],
    db_rows: Sequence[Mapping[str, Any]],
    insert_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    race_id = str(forecast_row.get("race_id") or "")
    official = _official_rows(lookup)
    terminal = _terminal_rows(lookup)
    official_keys = {row["name_key"] for row in official if row.get("name_key")}
    db_keys = {
        _name_key(row.get("dog_clean_name") or row.get("dog_name"))
        for row in db_rows
        if _name_key(row.get("dog_clean_name") or row.get("dog_name"))
    }
    existing_boxes = {
        box
        for box in (_box(row.get("box_number")) for row in db_rows)
        if box is not None
    }
    terminal_boxes = {
        box
        for box in (_box(row.get("box_number")) for row in terminal)
        if box is not None
    }
    missing_official = sorted(official_keys - db_keys)
    extra_db = sorted(db_keys - official_keys)
    terminal_box_db_rows = [
        dict(row)
        for row in db_rows
        if _box(row.get("box_number")) in terminal_boxes
    ]
    conflict_candidates = []
    no_conflict_candidates = []
    for candidate in insert_candidates:
        candidate_box = _box(candidate.get("box_number") or _mapping(candidate.get("insert_values")).get("box_number"))
        payload = {
            "candidate_id": candidate.get("candidate_id"),
            "official_dog_name": candidate.get("official_dog_name"),
            "name_key": candidate.get("name_key"),
            "box_number": candidate_box,
            "finish_position": _safe_int(
                candidate.get("finish_position")
                or _mapping(candidate.get("insert_values")).get("finish_position")
            ),
        }
        if candidate_box in existing_boxes:
            conflict_candidates.append(payload)
        else:
            no_conflict_candidates.append(payload)

    official_count = len(official)
    db_count = len(db_rows)
    naive_insert_count = len(insert_candidates)
    naive_post_count = db_count + naive_insert_count
    safe_insert_only = (
        not terminal
        and not extra_db
        and not conflict_candidates
        and naive_post_count == official_count
    )
    if terminal:
        lane = "TERMINAL_SCOPE_MANUAL_RECONCILIATION_REQUIRED"
        next_action = "exclude_terminal_scope_from_insert_only_apply_and_create_exact_manual_reconciliation"
    elif not safe_insert_only:
        lane = "INSERT_ONLY_ALLOWLIST_UNSAFE"
        next_action = "revise_insert_allowlist_before_any_apply"
    else:
        lane = "INSERT_ONLY_RECHECK_REQUIRED"
        next_action = "rerun_gap_review_before_any_apply"

    return {
        "race_id": race_id,
        "review_lane": forecast_row.get("review_lane"),
        "forecast_gate": forecast_row.get("forecast_gate"),
        "reconciliation_lane": lane,
        "official_finisher_count": official_count,
        "terminal_status_count": len(terminal),
        "current_db_runner_count": db_count,
        "missing_official_finisher_name_keys": missing_official,
        "extra_db_name_keys": extra_db,
        "naive_insert_candidates": list(insert_candidates),
        "insert_candidates_with_box_conflict": conflict_candidates,
        "insert_candidates_without_box_conflict": no_conflict_candidates,
        "terminal_status_rows": terminal,
        "terminal_box_db_rows": terminal_box_db_rows,
        "naive_post_repair_runner_count": naive_post_count,
        "safe_for_insert_only": safe_insert_only,
        "recommended_next_action": next_action,
    }


def build_terminal_scope_packet(
    *,
    forecast_packet_path: Path,
    insert_policy_packet_path: Path,
    lookup_packet_paths: Sequence[Path],
    db_path: Path,
) -> dict[str, Any]:
    forecast_resolved = forecast_packet_path.expanduser().resolve()
    policy_resolved = insert_policy_packet_path.expanduser().resolve()
    db_resolved = db_path.expanduser().resolve()
    forecast_packet = _load_json(forecast_resolved)
    policy_packet = _load_json(policy_resolved)
    failures: list[str] = []
    _validate_report_packet(
        packet=forecast_packet,
        expected_schema=FORECAST_SCHEMA_VERSION,
        packet_name="forecast_packet",
        failures=failures,
    )
    _validate_report_packet(
        packet=policy_packet,
        expected_schema=POLICY_SCHEMA_VERSION,
        packet_name="insert_policy_packet",
        failures=failures,
    )
    lookup_by_race = _lookup_index(lookup_packet_paths, failures)
    candidates_by_race = _candidate_rows_by_race(policy_packet)
    forecast_rows = _target_forecast_rows(forecast_packet)

    db_state: dict[str, Any] = {
        "db_path": str(db_resolved),
        "quick_check": None,
        "read_only": True,
        "query_only": True,
    }
    diagnostics = []
    with _connect_read_only(db_resolved) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        db_state["quick_check"] = quick_check[0] if quick_check else None
        if db_state["quick_check"] != "ok":
            failures.append("db_quick_check_failed")
        for row in forecast_rows:
            race_id = str(row.get("race_id") or "")
            lookup = lookup_by_race.get(race_id)
            if lookup is None:
                failures.append(f"lookup_missing:{race_id}")
                continue
            diagnostics.append(
                _diagnose_race(
                    forecast_row=row,
                    lookup=lookup,
                    db_rows=_fetch_db_rows(conn, race_id),
                    insert_candidates=candidates_by_race.get(race_id, []),
                )
            )

    lane_counts = Counter(str(row.get("reconciliation_lane")) for row in diagnostics)
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
            "forecast_packet": str(forecast_resolved),
            "insert_policy_packet": str(policy_resolved),
            "lookup_packets": [str(path.expanduser().resolve()) for path in lookup_packet_paths],
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "summary": {
            "races_considered": len(diagnostics),
            "reconciliation_lane_counts": dict(sorted(lane_counts.items())),
            "safe_for_insert_only_count": sum(1 for row in diagnostics if row.get("safe_for_insert_only") is True),
            "terminal_status_race_count": sum(1 for row in diagnostics if row.get("terminal_status_count", 0) > 0),
            "insert_candidate_box_conflict_count": sum(
                len(_list(row.get("insert_candidates_with_box_conflict"))) for row in diagnostics
            ),
            "extra_db_name_count": sum(len(_list(row.get("extra_db_name_keys"))) for row in diagnostics),
            "missing_official_finisher_count": sum(
                len(_list(row.get("missing_official_finisher_name_keys"))) for row in diagnostics
            ),
            "recommended_next_action": (
                "exclude_terminal_scope_from_insert_only_apply_and_prepare_manual_reconciliation_packet"
            ),
        },
        "race_diagnostics": diagnostics,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "terminal_policy_required": True,
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": [
            "db_write",
            "label_write",
            "metadata_write",
            "dog_row_insert",
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
    for row in _list(packet.get("race_diagnostics")):
        row_map = _mapping(row)
        rows.append(
            {
                "race_id": row_map.get("race_id"),
                "review_lane": row_map.get("review_lane"),
                "reconciliation_lane": row_map.get("reconciliation_lane"),
                "official_finisher_count": row_map.get("official_finisher_count"),
                "terminal_status_count": row_map.get("terminal_status_count"),
                "current_db_runner_count": row_map.get("current_db_runner_count"),
                "missing_official_finisher_count": len(_list(row_map.get("missing_official_finisher_name_keys"))),
                "extra_db_name_count": len(_list(row_map.get("extra_db_name_keys"))),
                "naive_insert_candidate_count": len(_list(row_map.get("naive_insert_candidates"))),
                "insert_candidate_box_conflict_count": len(_list(row_map.get("insert_candidates_with_box_conflict"))),
                "insert_candidate_no_box_conflict_count": len(_list(row_map.get("insert_candidates_without_box_conflict"))),
                "terminal_box_db_row_count": len(_list(row_map.get("terminal_box_db_rows"))),
                "naive_post_repair_runner_count": row_map.get("naive_post_repair_runner_count"),
                "forecast_gate": row_map.get("forecast_gate"),
                "safe_for_insert_only": row_map.get("safe_for_insert_only"),
                "recommended_next_action": row_map.get("recommended_next_action"),
            }
        )
    return rows


def write_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "terminal_scope_reconciliation_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "terminal_scope_reconciliation.csv").open(
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
        "# Terminal Scope Reconciliation",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Races considered: `{summary.get('races_considered')}`",
        f"- Reconciliation lane counts: `{summary.get('reconciliation_lane_counts')}`",
        f"- Safe for insert-only count: `{summary.get('safe_for_insert_only_count')}`",
        f"- Terminal-status race count: `{summary.get('terminal_status_race_count')}`",
        f"- Insert candidate box conflicts: `{summary.get('insert_candidate_box_conflict_count')}`",
        f"- Extra DB name count: `{summary.get('extra_db_name_count')}`",
        f"- Missing official finisher count: `{summary.get('missing_official_finisher_count')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Recommendation",
        "",
        "Exclude these terminal/exclusion-scope races from any insert-only apply batch. They need a separate manual reconciliation policy for extra DB rows, scratched/nonstarter rows, and box/name drift before label expansion or retraining.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-packet", required=True)
    parser.add_argument("--insert-policy-packet", required=True)
    parser.add_argument("--lookup-packet", action="append", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit("refusing report-only terminal reconciliation while approval flags are set:" + ",".join(active))
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_terminal_scope_packet(
        forecast_packet_path=Path(args.forecast_packet),
        insert_policy_packet_path=Path(args.insert_policy_packet),
        lookup_packet_paths=[Path(path) for path in args.lookup_packet],
        db_path=Path(args.db),
    )
    write_outputs(Path(args.output_dir), packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
