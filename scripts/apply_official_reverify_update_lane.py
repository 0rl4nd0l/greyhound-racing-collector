#!/usr/bin/env python3
"""Apply the approved exact-box-set official reverify update lane.

This is intentionally narrow. It only consumes a clean
official_reverify_update_rehearsal packet, creates a SQLite backup, verifies the
live DB still matches every rehearsed before-value, and then updates exactly the
rehearsed dog/result metadata rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "official_reverify_update_apply_v1"
REHEARSAL_SCHEMA_VERSION = "official_reverify_update_rehearsal_v1"
REHEARSAL_READY_STATUS = "READY_FOR_EXPLICIT_APPROVED_UPDATE_WRITE"
OFFICIAL_SOURCE = "thedogs_official"

NO_WRITE_FLAGS = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "promotion": False,
    "betting_decision": False,
}

APPLIED_WRITE_FLAGS = {
    **NO_WRITE_FLAGS,
    "db_write": True,
    "label_write": True,
    "metadata_write": True,
}
APPLIED_STATUSES = {"APPLIED", "APPLIED_WITH_POST_QUICK_CHECK_FAILURE"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _connect_read_write(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _quick_check(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA quick_check").fetchone()
    return str(row[0]) if row else "missing_quick_check_row"


def _create_backup(db_path: Path, output_dir: Path) -> dict[str, Any]:
    backup_dir = output_dir / "db_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{db_path.expanduser().resolve().stem}_{timestamp}.sqlite"
    with _connect_read_only(db_path) as source:
        with sqlite3.connect(backup_path) as dest:
            source.backup(dest)
            backup_quick_check = _quick_check(dest)
    return {
        "path": str(backup_path.resolve()),
        "method": "sqlite_backup_api",
        "quick_check": backup_quick_check,
    }


def _validate_rehearsal(rehearsal: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if rehearsal.get("schema_version") != REHEARSAL_SCHEMA_VERSION:
        failures.append("rehearsal_schema_mismatch")
    if rehearsal.get("status") != REHEARSAL_READY_STATUS:
        failures.append("rehearsal_not_ready")
    if _list(rehearsal.get("failures")):
        failures.append("rehearsal_has_failures")
    writes = _mapping(rehearsal.get("writes_performed"))
    if any(value is not False for value in writes.values()):
        failures.append("rehearsal_has_write_flags")

    candidates = _list(rehearsal.get("candidates"))
    if not candidates:
        failures.append("rehearsal_has_no_candidates")
    for index, candidate in enumerate(candidates):
        row = _mapping(candidate)
        prefix = f"candidate:{index}"
        if row.get("status") != "READY":
            failures.append(f"{prefix}:not_ready")
        if _list(row.get("blockers")):
            failures.append(f"{prefix}:has_blockers")
        if not row.get("race_id"):
            failures.append(f"{prefix}:race_id_missing")
        if not _list(row.get("dog_updates")):
            failures.append(f"{prefix}:dog_updates_missing")
        if not _mapping(row.get("metadata_update")).get("after"):
            failures.append(f"{prefix}:metadata_after_missing")
    return failures


def _fetch_dog_rows(conn: sqlite3.Connection, race_id: str) -> dict[int, sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT dog_name, box_number, finish_position, placing,
               scraped_finish_position, data_source
        FROM dog_race_data
        WHERE race_id = ?
        ORDER BY box_number
        """,
        (race_id,),
    ).fetchall()
    return {int(row["box_number"]): row for row in rows}


def _fetch_metadata(conn: sqlite3.Connection, race_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT results_status, winner_name, winner_source
        FROM race_metadata
        WHERE race_id = ?
        """,
        (race_id,),
    ).fetchone()


def _row_value(row: sqlite3.Row, key: str) -> Any:
    return row[key]


def _preimage_failures_for_candidate(
    conn: sqlite3.Connection,
    candidate: Mapping[str, Any],
) -> list[str]:
    race_id = str(candidate.get("race_id") or "")
    failures: list[str] = []
    dog_updates = [_mapping(item) for item in _list(candidate.get("dog_updates"))]
    expected_boxes = sorted(int(update.get("box_number")) for update in dog_updates)
    current_rows = _fetch_dog_rows(conn, race_id)
    current_boxes = sorted(current_rows)
    if current_boxes != expected_boxes:
        failures.append(
            f"{race_id}:box_set_mismatch:expected={expected_boxes}:actual={current_boxes}"
        )

    for update in dog_updates:
        box_number = int(update.get("box_number"))
        current = current_rows.get(box_number)
        if current is None:
            failures.append(f"{race_id}:box:{box_number}:missing")
            continue
        if _row_value(current, "dog_name") != update.get("dog_name"):
            failures.append(f"{race_id}:box:{box_number}:dog_name_changed")
        before = _mapping(update.get("before"))
        for key in (
            "finish_position",
            "placing",
            "scraped_finish_position",
            "data_source",
        ):
            if _row_value(current, key) != before.get(key):
                failures.append(f"{race_id}:box:{box_number}:{key}_changed")

    metadata = _fetch_metadata(conn, race_id)
    if metadata is None:
        failures.append(f"{race_id}:metadata_missing")
    else:
        before = _mapping(_mapping(candidate.get("metadata_update")).get("before"))
        for key in ("results_status", "winner_name", "winner_source"):
            if _row_value(metadata, key) != before.get(key):
                failures.append(f"{race_id}:metadata:{key}_changed")
    return failures


def _apply_candidate(conn: sqlite3.Connection, candidate: Mapping[str, Any]) -> dict[str, int]:
    race_id = str(candidate.get("race_id") or "")
    dog_rows_updated = 0
    for update in [_mapping(item) for item in _list(candidate.get("dog_updates"))]:
        after = _mapping(update.get("after"))
        result = conn.execute(
            """
            UPDATE dog_race_data
               SET finish_position = ?,
                   placing = ?,
                   scraped_finish_position = ?,
                   data_source = ?
             WHERE race_id = ?
               AND box_number = ?
            """,
            (
                after.get("finish_position"),
                after.get("placing"),
                after.get("scraped_finish_position"),
                after.get("data_source"),
                race_id,
                update.get("box_number"),
            ),
        )
        if result.rowcount != 1:
            raise RuntimeError(f"{race_id}:box:{update.get('box_number')}:update_rowcount:{result.rowcount}")
        dog_rows_updated += result.rowcount

    metadata_after = _mapping(_mapping(candidate.get("metadata_update")).get("after"))
    result = conn.execute(
        """
        UPDATE race_metadata
           SET results_status = ?,
               winner_name = ?,
               winner_source = ?
         WHERE race_id = ?
        """,
        (
            metadata_after.get("results_status"),
            metadata_after.get("winner_name"),
            metadata_after.get("winner_source"),
            race_id,
        ),
    )
    if result.rowcount != 1:
        raise RuntimeError(f"{race_id}:metadata:update_rowcount:{result.rowcount}")
    return {"dog_rows_updated": dog_rows_updated, "metadata_rows_updated": result.rowcount}


def _empty_summary(rehearsal: Mapping[str, Any]) -> dict[str, int]:
    summary = _mapping(rehearsal.get("summary"))
    return {
        "candidate_count": int(summary.get("candidate_count") or 0),
        "races_updated": 0,
        "dog_rows_updated": 0,
        "metadata_rows_updated": 0,
        "skipped_non_exact_lane_count": int(summary.get("skipped_non_exact_lane_count") or 0),
    }


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    backup = _mapping(packet.get("backup"))
    lines = [
        "# Official Reverify Update Apply",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "## Summary",
        "",
        f"- Candidate count: `{summary.get('candidate_count')}`",
        f"- Races updated: `{summary.get('races_updated')}`",
        f"- Dog rows updated: `{summary.get('dog_rows_updated')}`",
        f"- Metadata rows updated: `{summary.get('metadata_rows_updated')}`",
        f"- Skipped non-exact lane count: `{summary.get('skipped_non_exact_lane_count')}`",
        "",
        "## Scope",
        "",
        "No official fetches, snapshot mutations, manifest mutations, model training, registry mutations, promotions, betting decisions, or EV claims were performed.",
        "",
    ]
    if backup:
        lines.extend(
            [
                "## Backup",
                "",
                f"- Path: `{backup.get('path')}`",
                f"- Quick check: `{backup.get('quick_check')}`",
                "",
            ]
        )
    failures = _list(packet.get("failures"))
    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in failures)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _persist_packet(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_reverify_update_apply.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", packet)


def _packet(
    *,
    status: str,
    rehearsal_packet_path: Path,
    db_path: Path,
    output_dir: Path,
    rehearsal: Mapping[str, Any],
    failures: list[str] | None = None,
    backup: Mapping[str, Any] | None = None,
    summary: Mapping[str, int] | None = None,
    source_quick_check_before: str | None = None,
    source_quick_check_after: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "failures": failures or [],
        "source_evidence": {
            "rehearsal_packet": str(rehearsal_packet_path.expanduser().resolve()),
            "db": str(db_path.expanduser().resolve()),
        },
        "backup": dict(backup or {}),
        "summary": dict(summary or _empty_summary(rehearsal)),
        "approval_gate": {
            "required": True,
            "approved": status in APPLIED_STATUSES,
            "approved_by": "cli_flag" if status in APPLIED_STATUSES else None,
            "required_cli_flag": "--write-labels-approved",
            "env_APPROVE_RESULT_LABEL_WRITE_present": bool(
                str(os.environ.get("APPROVE_RESULT_LABEL_WRITE") or "").strip()
            ),
        },
        "source_quick_check_before": source_quick_check_before,
        "source_quick_check_after": source_quick_check_after,
        "candidate_race_ids": [
            str(_mapping(candidate).get("race_id") or "")
            for candidate in _list(rehearsal.get("candidates"))
        ],
        "writes_performed": dict(APPLIED_WRITE_FLAGS if status in APPLIED_STATUSES else NO_WRITE_FLAGS),
        "output_dir": str(output_dir.expanduser().resolve()),
    }


def apply_update_lane(
    *,
    rehearsal_packet_path: Path,
    db_path: Path,
    output_dir: Path,
    write_labels_approved: bool,
) -> dict[str, Any]:
    rehearsal_resolved = rehearsal_packet_path.expanduser().resolve()
    db_resolved = db_path.expanduser().resolve()
    output_resolved = output_dir.expanduser().resolve()
    rehearsal = _load_json(rehearsal_resolved)

    if not write_labels_approved:
        packet = _packet(
            status="NOT_APPROVED",
            rehearsal_packet_path=rehearsal_resolved,
            db_path=db_resolved,
            output_dir=output_resolved,
            rehearsal=rehearsal,
            failures=["write_labels_approved_flag_missing"],
        )
        _persist_packet(output_resolved, packet)
        return packet

    validation_failures = _validate_rehearsal(rehearsal)
    if validation_failures:
        packet = _packet(
            status="REHEARSAL_NOT_READY",
            rehearsal_packet_path=rehearsal_resolved,
            db_path=db_resolved,
            output_dir=output_resolved,
            rehearsal=rehearsal,
            failures=validation_failures,
        )
        _persist_packet(output_resolved, packet)
        return packet

    candidates = [_mapping(candidate) for candidate in _list(rehearsal.get("candidates"))]
    output_resolved.mkdir(parents=True, exist_ok=True)
    backup = _create_backup(db_resolved, output_resolved)
    if backup.get("quick_check") != "ok":
        packet = _packet(
            status="BACKUP_FAILED_QUICK_CHECK",
            rehearsal_packet_path=rehearsal_resolved,
            db_path=db_resolved,
            output_dir=output_resolved,
            rehearsal=rehearsal,
            failures=["backup_quick_check_failed"],
            backup=backup,
        )
        _persist_packet(output_resolved, packet)
        return packet

    with _connect_read_write(db_resolved) as conn:
        source_quick_check_before = _quick_check(conn)
        if source_quick_check_before != "ok":
            packet = _packet(
                status="SOURCE_FAILED_QUICK_CHECK",
                rehearsal_packet_path=rehearsal_resolved,
                db_path=db_resolved,
                output_dir=output_resolved,
                rehearsal=rehearsal,
                failures=["source_quick_check_failed_before_apply"],
                backup=backup,
                source_quick_check_before=source_quick_check_before,
            )
            _persist_packet(output_resolved, packet)
            return packet

        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            preimage_failures: list[str] = []
            for candidate in candidates:
                preimage_failures.extend(_preimage_failures_for_candidate(conn, candidate))
            if preimage_failures:
                conn.execute("ROLLBACK")
                packet = _packet(
                    status="PREIMAGE_MISMATCH",
                    rehearsal_packet_path=rehearsal_resolved,
                    db_path=db_resolved,
                    output_dir=output_resolved,
                    rehearsal=rehearsal,
                    failures=sorted(set(preimage_failures)),
                    backup=backup,
                    source_quick_check_before=source_quick_check_before,
                )
                _persist_packet(output_resolved, packet)
                return packet

            dog_rows_updated = 0
            metadata_rows_updated = 0
            for candidate in candidates:
                counts = _apply_candidate(conn, candidate)
                dog_rows_updated += counts["dog_rows_updated"]
                metadata_rows_updated += counts["metadata_rows_updated"]
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        source_quick_check_after = _quick_check(conn)

    summary = _mapping(rehearsal.get("summary"))
    failures = (
        ["source_quick_check_failed_after_apply"]
        if source_quick_check_after != "ok"
        else []
    )
    status = "APPLIED" if not failures else "APPLIED_WITH_POST_QUICK_CHECK_FAILURE"
    packet = _packet(
        status=status,
        rehearsal_packet_path=rehearsal_resolved,
        db_path=db_resolved,
        output_dir=output_resolved,
        rehearsal=rehearsal,
        failures=failures,
        backup=backup,
        summary={
            "candidate_count": len(candidates),
            "races_updated": len(candidates),
            "dog_rows_updated": dog_rows_updated,
            "metadata_rows_updated": metadata_rows_updated,
            "skipped_non_exact_lane_count": int(summary.get("skipped_non_exact_lane_count") or 0),
        },
        source_quick_check_before=source_quick_check_before,
        source_quick_check_after=source_quick_check_after,
    )
    _persist_packet(output_resolved, packet)
    return packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal-packet", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--write-labels-approved",
        action="store_true",
        help="Required explicit approval gate for this exact write lane.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = apply_update_lane(
        rehearsal_packet_path=Path(args.rehearsal_packet),
        db_path=Path(args.db),
        output_dir=Path(args.output_dir),
        write_labels_approved=bool(args.write_labels_approved),
    )
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2))
    return 0 if packet["status"] == "APPLIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
