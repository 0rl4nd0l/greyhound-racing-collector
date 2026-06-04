#!/usr/bin/env python3
"""Build a no-write preflight for official reverify label candidates.

Consumes the report-only official reverify lookup packet and checks strict-ready
candidate races against the active DB in read-only mode. This script does not
write labels or create metadata; it only classifies what would need explicit
approval and what remains blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "official_reverify_label_preflight_v1"
LOOKUP_SCHEMA_VERSION = "official_reverify_lookup_dry_run_v1"
OFFICIAL_SOURCE = "thedogs_official"
READY_STATUS = "PREFLIGHT_READY"
BLOCKED_STATUS = "BLOCKED"

WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "promotion": False,
    "betting_decision": False,
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


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _venue_key(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def _race_id_variants(*, venue: str, race_date: str, race_number: int) -> list[str]:
    venue_key = _venue_key(venue)
    variants = {
        f"{venue_key}_{race_date}_{race_number}",
        f"Race {race_number} - {venue_key} - {race_date}",
    }
    return sorted(variants)


def _positions_valid_for_write(candidate: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected_rows = _safe_int(candidate.get("legacy_runner_rows"))
    positions = _list(candidate.get("positions"))
    terminal_statuses = _list(candidate.get("terminal_statuses"))
    if expected_rows <= 0:
        reasons.append("legacy_runner_count_missing")
    if len(positions) != expected_rows:
        reasons.append("official_positions_incomplete_for_legacy_runner_count")

    finish_positions: list[int] = []
    box_numbers: list[int] = []
    for item in positions:
        row = _mapping(item)
        finish_positions.append(_safe_int(row.get("finish_position")))
        box_numbers.append(_safe_int(row.get("box_number")))
    if 1 not in finish_positions:
        reasons.append("official_first_place_missing")
    if len(finish_positions) != len(set(finish_positions)):
        reasons.append("official_duplicate_finish_positions")
    if sorted(finish_positions) != list(range(1, len(finish_positions) + 1)):
        reasons.append("official_finish_positions_not_contiguous")
    if len(box_numbers) != len(set(box_numbers)):
        reasons.append("official_duplicate_box_numbers")
    if terminal_statuses:
        reasons.append("official_terminal_statuses_present")
    return sorted(set(reasons))


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _db_identity_rows(
    conn: sqlite3.Connection,
    *,
    venue: str,
    race_date: str,
    race_number: int,
) -> list[sqlite3.Row]:
    variants = _race_id_variants(
        venue=venue,
        race_date=race_date,
        race_number=race_number,
    )
    rows = conn.execute(
        """
        SELECT race_id, venue, race_number, race_date, results_status, winner_name, winner_source
        FROM race_metadata
        WHERE race_id IN (?, ?)
           OR (
                race_date = ?
            AND race_number = ?
            AND UPPER(REPLACE(COALESCE(venue, ''), ' ', '_')) = ?
           )
        ORDER BY race_id
        """,
        (variants[0], variants[1], race_date, race_number, _venue_key(venue)),
    ).fetchall()
    return list(rows)


def _dog_state(conn: sqlite3.Connection, race_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN data_source = ? THEN 1 ELSE 0 END) AS official_rows,
            SUM(
                CASE
                    WHEN finish_position IS NOT NULL
                      OR placing IS NOT NULL
                      OR scraped_finish_position IS NOT NULL
                    THEN 1 ELSE 0
                END
            ) AS labelled_rows
        FROM dog_race_data
        WHERE race_id = ?
        """,
        (OFFICIAL_SOURCE, race_id),
    ).fetchone()
    box_rows = conn.execute(
        """
        SELECT box_number
        FROM dog_race_data
        WHERE race_id = ?
          AND box_number IS NOT NULL
        ORDER BY box_number
        """,
        (race_id,),
    ).fetchall()
    return {
        "total_rows": int(row["total_rows"] or 0),
        "official_rows": int(row["official_rows"] or 0),
        "labelled_rows": int(row["labelled_rows"] or 0),
        "box_numbers": [int(item["box_number"]) for item in box_rows],
    }


def _preflight_candidate(
    conn: sqlite3.Connection,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    lookup_key = _mapping(candidate.get("lookup_key"))
    venue = str(lookup_key.get("venue") or "").strip()
    race_date = str(lookup_key.get("race_date") or "").strip()
    race_number = _safe_int(lookup_key.get("race_number"))
    blockers = _positions_valid_for_write(candidate)
    metadata_rows = _db_identity_rows(
        conn,
        venue=venue,
        race_date=race_date,
        race_number=race_number,
    )
    resolved_race_id = None
    metadata_payload = []
    for row in metadata_rows:
        payload = dict(row)
        metadata_payload.append(payload)
    if not metadata_rows:
        blockers.append("race_metadata_missing")
    elif len(metadata_rows) > 1:
        blockers.append("race_metadata_ambiguous")
    else:
        metadata = dict(metadata_rows[0])
        resolved_race_id = str(metadata["race_id"])
        if metadata.get("results_status") != "pending":
            blockers.append("race_metadata_not_pending")
        if metadata.get("winner_name") not in (None, ""):
            blockers.append("race_metadata_winner_present")
        if metadata.get("winner_source") not in (None, ""):
            blockers.append("race_metadata_winner_source_present")

    dog_state = (
        _dog_state(conn, resolved_race_id)
        if resolved_race_id
        else {"total_rows": 0, "official_rows": 0, "labelled_rows": 0, "box_numbers": []}
    )
    official_boxes = sorted(
        _safe_int(_mapping(item).get("box_number"))
        for item in _list(candidate.get("positions"))
    )
    existing_boxes = sorted(int(box) for box in dog_state.get("box_numbers") or [])
    missing_existing_boxes = sorted(set(official_boxes) - set(existing_boxes))
    extra_existing_boxes = sorted(set(existing_boxes) - set(official_boxes))
    row_alignment = {
        "official_box_numbers": official_boxes,
        "existing_box_numbers": existing_boxes,
        "box_set_matches_official": existing_boxes == official_boxes,
        "missing_existing_boxes": missing_existing_boxes,
        "extra_existing_boxes": extra_existing_boxes,
    }
    if dog_state["total_rows"] <= 0:
        blockers.append("db_dog_rows_missing")
    if dog_state["total_rows"] > 0 or dog_state["labelled_rows"] > 0:
        blockers.append("db_has_existing_result_rows")
    if dog_state["official_rows"] > 0:
        blockers.append("db_has_existing_official_rows")

    blockers = sorted(set(blockers))
    return {
        "legacy_race_id": candidate.get("legacy_race_id"),
        "lookup_key": candidate.get("lookup_key"),
        "source_url": candidate.get("source_url"),
        "legacy_runner_rows": candidate.get("legacy_runner_rows"),
        "resolved_db_race_id": resolved_race_id,
        "race_id_variants": _race_id_variants(
            venue=venue,
            race_date=race_date,
            race_number=race_number,
        ),
        "positions": candidate.get("positions") or [],
        "metadata_matches": metadata_payload,
        "dog_race_data_state": dog_state,
        "row_alignment": row_alignment,
        "blockers": blockers,
        "preflight_status": READY_STATUS if not blockers else BLOCKED_STATUS,
    }


def _validate_lookup_packet(packet: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("schema_version") != LOOKUP_SCHEMA_VERSION:
        failures.append("lookup_packet_schema_mismatch")
    if packet.get("status") != "REPORT_ONLY":
        failures.append("lookup_packet_status_not_report_only")
    writes = _mapping(packet.get("writes_performed"))
    forbidden_true = [
        key
        for key, value in writes.items()
        if key != "official_fetch" and value is not False
    ]
    if forbidden_true:
        failures.append("lookup_packet_has_forbidden_write_flags:" + ",".join(forbidden_true))
    return failures


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    ready_count = _safe_int(summary.get("preflight_ready_count"))
    next_step = (
        "Human-review a small sample of preflight-ready rows, then require explicit label-write approval before any DB mutation."
        if ready_count > 0
        else "Do not write labels. Reconcile existing DB rows and missing metadata first; direct label-write preflight has zero ready candidates."
    )
    lines = [
        "# Official Reverify Label Preflight",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, official fetches, snapshot mutations, manifest mutations, model training, registry mutations, promotions, betting decisions, or EV claims were performed.",
        "",
        "## Summary",
        "",
        f"- Lookup label-write-ready rows: `{summary.get('lookup_label_write_ready_count')}`",
        f"- Preflight-ready rows: `{summary.get('preflight_ready_count')}`",
        f"- Blocked rows: `{summary.get('blocked_count')}`",
        f"- Skipped not lookup-ready rows: `{summary.get('skipped_not_lookup_ready_count')}`",
        f"- Blocker counts: `{summary.get('blocker_counts')}`",
        "",
        "## Next Step",
        "",
        next_step,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_preflight_packet(*, lookup_packet_path: Path, db_path: Path) -> dict[str, Any]:
    lookup_resolved = lookup_packet_path.expanduser().resolve()
    db_resolved = db_path.expanduser().resolve()
    lookup_packet = _load_json(lookup_resolved)
    failures = _validate_lookup_packet(lookup_packet)
    db_state: dict[str, Any] = {
        "db_path": str(db_resolved),
        "quick_check": None,
        "read_only": True,
        "query_only": True,
    }
    candidate_preflight: list[dict[str, Any]] = []
    skipped_not_lookup_ready = 0

    with _connect_read_only(db_resolved) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        db_state["quick_check"] = quick_check[0] if quick_check else None
        if db_state["quick_check"] != "ok":
            failures.append("db_quick_check_failed")
        for table in ("race_metadata", "dog_race_data"):
            if not _table_exists(conn, table):
                failures.append(f"db_table_missing:{table}")
        if not failures:
            for candidate in _list(lookup_packet.get("results")):
                if not isinstance(candidate, Mapping):
                    continue
                if candidate.get("label_write_ready") is not True:
                    skipped_not_lookup_ready += 1
                    continue
                candidate_preflight.append(_preflight_candidate(conn, candidate))

    blocker_counts: Counter[str] = Counter()
    for row in candidate_preflight:
        for blocker in row.get("blockers") or []:
            blocker_counts[str(blocker)] += 1
    ready_count = sum(
        1 for row in candidate_preflight if row.get("preflight_status") == READY_STATUS
    )
    blocked_count = len(candidate_preflight) - ready_count
    status = (
        "NOT_READY"
        if failures
        else "PREFLIGHT_READY"
        if blocked_count == 0 and ready_count > 0
        else "PREFLIGHT_READY_WITH_BLOCKERS"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "failures": failures,
        "summary": {
            "lookup_label_write_ready_count": len(candidate_preflight),
            "preflight_ready_count": ready_count,
            "blocked_count": blocked_count,
            "skipped_not_lookup_ready_count": skipped_not_lookup_ready,
            "blocker_counts": dict(sorted(blocker_counts.items())),
        },
        "source_evidence": {
            "lookup_packet": str(lookup_resolved),
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "candidate_preflight": candidate_preflight,
        "approval_gate": {
            "required": True,
            "approved": False,
            "required_env_var": "APPROVE_RESULT_LABEL_WRITE",
            "required_cli_flag": "--write-labels-approved",
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": [
            "label_write",
            "metadata_create",
            "model_training_or_promotion",
            "betting_decision",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup-packet", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    if str(os.environ.get("APPROVE_RESULT_LABEL_WRITE") or "").strip():
        raise SystemExit("refusing preflight while APPROVE_RESULT_LABEL_WRITE is set")
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_preflight_packet(
        lookup_packet_path=Path(args.lookup_packet),
        db_path=Path(args.db),
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
