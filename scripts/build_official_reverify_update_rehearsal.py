#!/usr/bin/env python3
"""Build a no-write update rehearsal for exact-box-set reverify candidates."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "official_reverify_update_rehearsal_v1"
PREFLIGHT_SCHEMA_VERSION = "official_reverify_label_preflight_v1"
OFFICIAL_SOURCE = "thedogs_official"

WRITES_PERFORMED = {
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


def _clean_dog_name(value: Any) -> str:
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(value or "").strip())
    text = text.replace('"', "").replace("'", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip().title()


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _is_exact_update_candidate(candidate: Mapping[str, Any]) -> bool:
    blockers = {str(item) for item in _list(candidate.get("blockers"))}
    alignment = _mapping(candidate.get("row_alignment"))
    return blockers == {"db_has_existing_result_rows"} and (
        alignment.get("box_set_matches_official") is True
    )


def _positions_by_box(candidate: Mapping[str, Any]) -> dict[int, int]:
    return {
        _safe_int(_mapping(item).get("box_number")): _safe_int(
            _mapping(item).get("finish_position")
        )
        for item in _list(candidate.get("positions"))
    }


def _fetch_existing_rows(conn: sqlite3.Connection, race_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT dog_name, box_number, finish_position, placing,
                   scraped_finish_position, data_source
            FROM dog_race_data
            WHERE race_id = ?
            ORDER BY box_number
            """,
            (race_id,),
        ).fetchall()
    )


def _fetch_metadata(conn: sqlite3.Connection, race_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT results_status, winner_name, winner_source
        FROM race_metadata
        WHERE race_id = ?
        """,
        (race_id,),
    ).fetchone()


def _candidate_rehearsal(
    conn: sqlite3.Connection,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    race_id = str(candidate.get("resolved_db_race_id") or "")
    positions = _positions_by_box(candidate)
    rows = _fetch_existing_rows(conn, race_id)
    metadata = _fetch_metadata(conn, race_id)
    blockers: list[str] = []
    if not rows:
        blockers.append("db_dog_rows_missing")
    if metadata is None:
        blockers.append("race_metadata_missing")
    if sorted(positions) != sorted(int(row["box_number"]) for row in rows):
        blockers.append("box_set_changed_since_preflight")

    dog_updates = []
    winner = None
    for row in rows:
        box_number = int(row["box_number"])
        official_position = positions.get(box_number)
        if not official_position:
            blockers.append(f"official_position_missing_for_box:{box_number}")
            continue
        dog_name = row["dog_name"]
        if official_position == 1:
            winner = {"box_number": box_number, "dog_name": _clean_dog_name(dog_name)}
        dog_updates.append(
            {
                "box_number": box_number,
                "dog_name": dog_name,
                "before": {
                    "finish_position": row["finish_position"],
                    "placing": row["placing"],
                    "scraped_finish_position": row["scraped_finish_position"],
                    "data_source": row["data_source"],
                },
                "after": {
                    "finish_position": official_position,
                    "placing": official_position,
                    "scraped_finish_position": str(official_position),
                    "data_source": OFFICIAL_SOURCE,
                },
            }
        )

    if winner is None:
        blockers.append("winner_not_resolved")
    metadata_before = (
        {
            "results_status": metadata["results_status"],
            "winner_name": metadata["winner_name"],
            "winner_source": metadata["winner_source"],
        }
        if metadata is not None
        else None
    )
    if metadata_before is not None:
        if metadata_before["results_status"] != "pending":
            blockers.append("race_metadata_not_pending")
        if metadata_before["winner_name"] not in (None, ""):
            blockers.append("race_metadata_winner_present")
        if metadata_before["winner_source"] not in (None, ""):
            blockers.append("race_metadata_winner_source_present")
    metadata_after = (
        {
            "results_status": "resulted",
            "winner_name": winner["dog_name"] if winner else None,
            "winner_source": OFFICIAL_SOURCE,
        }
        if winner
        else None
    )
    blockers = sorted(set(blockers))
    return {
        "legacy_race_id": candidate.get("legacy_race_id"),
        "race_id": race_id,
        "lookup_key": candidate.get("lookup_key"),
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": blockers,
        "winner": winner,
        "dog_updates": dog_updates,
        "metadata_update": {
            "before": metadata_before,
            "after": metadata_after,
        },
        "write_sql_shape": {
            "dog_race_data": (
                "UPDATE dog_race_data SET finish_position=?, placing=?, "
                "scraped_finish_position=?, data_source=? WHERE race_id=? AND box_number=?"
            ),
            "race_metadata": (
                "UPDATE race_metadata SET results_status=?, winner_name=?, "
                "winner_source=? WHERE race_id=?"
            ),
        },
    }


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Official Reverify Update Rehearsal",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot mutations, manifest mutations, model training, registry mutations, promotions, betting decisions, or EV claims were performed.",
        "",
        "## Summary",
        "",
        f"- Candidate count: `{summary.get('candidate_count')}`",
        f"- Ready count: `{summary.get('ready_count')}`",
        f"- Blocked count: `{summary.get('blocked_count')}`",
        f"- Skipped non-exact lane count: `{summary.get('skipped_non_exact_lane_count')}`",
        "",
        "## Recommendation",
        "",
        "Only apply this lane after creating a DB backup and running an approved update writer over exactly these rehearsed rows.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_update_rehearsal(
    *,
    preflight_packet_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    preflight_resolved = preflight_packet_path.expanduser().resolve()
    db_resolved = db_path.expanduser().resolve()
    preflight = _load_json(preflight_resolved)
    failures: list[str] = []
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        failures.append("preflight_schema_mismatch")
    writes = _mapping(preflight.get("writes_performed"))
    if any(value is not False for value in writes.values()):
        failures.append("preflight_has_write_flags")

    candidates = []
    skipped = 0
    with _connect_read_only(db_resolved) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        if not quick_check or quick_check[0] != "ok":
            failures.append("db_quick_check_failed")
        for candidate in _list(preflight.get("candidate_preflight")):
            if not isinstance(candidate, Mapping):
                continue
            if not _is_exact_update_candidate(candidate):
                skipped += 1
                continue
            candidates.append(_candidate_rehearsal(conn, candidate))

    ready_count = sum(1 for item in candidates if item.get("status") == "READY")
    blocked_count = len(candidates) - ready_count
    status = (
        "NOT_READY"
        if failures
        else "READY_FOR_EXPLICIT_APPROVED_UPDATE_WRITE"
        if ready_count and not blocked_count
        else "REHEARSAL_HAS_BLOCKERS"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "failures": failures,
        "summary": {
            "candidate_count": len(candidates),
            "ready_count": ready_count,
            "blocked_count": blocked_count,
            "skipped_non_exact_lane_count": skipped,
        },
        "source_evidence": {
            "preflight_packet": str(preflight_resolved),
            "db": str(db_resolved),
        },
        "candidates": candidates,
        "approval_gate": {
            "required": True,
            "approved": False,
            "required_env_var": "APPROVE_RESULT_LABEL_WRITE",
            "required_cli_flag": "--write-labels-approved",
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "backup_required_before_apply": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-packet", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    if str(os.environ.get("APPROVE_RESULT_LABEL_WRITE") or "").strip():
        raise SystemExit("refusing rehearsal while APPROVE_RESULT_LABEL_WRITE is set")
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_update_rehearsal(
        preflight_packet_path=Path(args.preflight_packet),
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
