#!/usr/bin/env python3
"""Diagnose official reverify box-set mismatches without writing labels.

This helper consumes an official reverify label preflight packet. For blocked
rows where official result boxes differ from existing DB boxes, it checks
whether runner names still match exactly. If they do, it emits a no-write review
packet showing the row-level correction shape required before any approved
label update can be considered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "official_reverify_box_mismatch_diagnosis_v1"
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


def _assert_output_path_safe(path: Path, root: Path | None = None) -> Path:
    resolved, relative = _repo_output_path(path, root)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


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


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _fetch_existing_rows(conn: sqlite3.Connection, race_id: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT dog_name, box_number, finish_position, placing,
                   scraped_finish_position, data_source
            FROM dog_race_data
            WHERE race_id = ?
            ORDER BY box_number, dog_name
            """,
            (race_id,),
        ).fetchall()
    ]


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(key for key, count in counts.items() if key and count > 1)


def _is_box_mismatch_candidate(candidate: Mapping[str, Any]) -> bool:
    alignment = _mapping(candidate.get("row_alignment"))
    blockers = {str(item) for item in _list(candidate.get("blockers"))}
    return (
        candidate.get("preflight_status") == "BLOCKED"
        and alignment.get("box_set_matches_official") is False
        and "db_has_existing_result_rows" in blockers
        and bool(candidate.get("resolved_db_race_id"))
        and bool(_list(candidate.get("positions")))
    )


def _diagnose_candidate(conn: sqlite3.Connection, candidate: Mapping[str, Any]) -> dict[str, Any]:
    race_id = str(candidate.get("resolved_db_race_id") or "")
    official_rows = [
        {
            "dog_name": _mapping(row).get("dog_name"),
            "name_key": _name_key(_mapping(row).get("dog_name")),
            "box_number": _safe_int(_mapping(row).get("box_number")),
            "finish_position": _safe_int(_mapping(row).get("finish_position")),
        }
        for row in _list(candidate.get("positions"))
    ]
    db_rows = [
        {
            **row,
            "name_key": _name_key(row.get("dog_name")),
            "box_number": _safe_int(row.get("box_number")),
            "finish_position": _safe_int(row.get("finish_position")),
            "placing": _safe_int(row.get("placing")),
        }
        for row in _fetch_existing_rows(conn, race_id)
    ]
    official_keys = [str(row.get("name_key") or "") for row in official_rows]
    db_keys = [str(row.get("name_key") or "") for row in db_rows]
    duplicate_official = _duplicates(official_keys)
    duplicate_db = _duplicates(db_keys)
    official_key_set = {key for key in official_keys if key}
    db_key_set = {key for key in db_keys if key}
    missing_db_names = sorted(official_key_set - db_key_set)
    extra_db_names = sorted(db_key_set - official_key_set)
    exact_name_set_match = (
        bool(official_key_set)
        and official_key_set == db_key_set
        and not duplicate_official
        and not duplicate_db
    )
    official_by_name = {str(row["name_key"]): row for row in official_rows if row.get("name_key")}
    proposed_updates = []
    unchanged_rows = []
    for db_row in db_rows:
        key = str(db_row.get("name_key") or "")
        official = official_by_name.get(key)
        if not official:
            continue
        before = {
            "box_number": db_row.get("box_number"),
            "finish_position": db_row.get("finish_position"),
            "placing": db_row.get("placing"),
            "scraped_finish_position": db_row.get("scraped_finish_position"),
            "data_source": db_row.get("data_source"),
        }
        after = {
            "box_number": official.get("box_number"),
            "finish_position": official.get("finish_position"),
            "placing": official.get("finish_position"),
            "scraped_finish_position": str(official.get("finish_position")),
            "data_source": OFFICIAL_SOURCE,
        }
        row_payload = {
            "dog_name": db_row.get("dog_name"),
            "name_key": key,
            "before": before,
            "after": after,
        }
        if before != after:
            proposed_updates.append(row_payload)
        else:
            unchanged_rows.append(row_payload)

    blockers = []
    if not exact_name_set_match:
        blockers.append("runner_name_set_not_exact")
    if missing_db_names:
        blockers.append("missing_db_names")
    if extra_db_names:
        blockers.append("extra_db_names")
    if duplicate_official:
        blockers.append("duplicate_official_names")
    if duplicate_db:
        blockers.append("duplicate_db_names")
    status = (
        "NAME_MATCHED_BOX_MISMATCH_REVIEW_READY"
        if exact_name_set_match and proposed_updates
        else "BOX_MISMATCH_MANUAL_REVIEW_REQUIRED"
    )
    return {
        "legacy_race_id": candidate.get("legacy_race_id"),
        "resolved_db_race_id": race_id,
        "lookup_key": candidate.get("lookup_key"),
        "source_url": candidate.get("source_url"),
        "status": status,
        "blockers": blockers,
        "exact_name_set_match": exact_name_set_match,
        "official_runner_count": len(official_rows),
        "db_runner_count": len(db_rows),
        "official_box_numbers": sorted(
            box for box in (row.get("box_number") for row in official_rows) if box is not None
        ),
        "db_box_numbers": sorted(
            box for box in (row.get("box_number") for row in db_rows) if box is not None
        ),
        "missing_db_name_keys": missing_db_names,
        "extra_db_name_keys": extra_db_names,
        "duplicate_official_name_keys": duplicate_official,
        "duplicate_db_name_keys": duplicate_db,
        "proposed_no_write_updates": proposed_updates,
        "unchanged_name_matched_rows": unchanged_rows,
        "write_sql_shape_if_later_approved": {
            "dog_race_data": (
                "UPDATE dog_race_data SET box_number=?, finish_position=?, placing=?, "
                "scraped_finish_position=?, data_source=? WHERE race_id=? AND dog_name=?"
            )
        },
    }


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Official Reverify Box Mismatch Diagnosis",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot mutations, manifest mutations, model training, registry mutations, promotions, betting decisions, or EV claims were performed.",
        "",
        "## Summary",
        "",
        f"- Candidates considered: `{summary.get('candidate_count')}`",
        f"- Name-matched review-ready candidates: `{summary.get('name_matched_review_ready_count')}`",
        f"- Manual-review candidates: `{summary.get('manual_review_count')}`",
        f"- Proposed row updates: `{summary.get('proposed_update_row_count')}`",
        f"- Safe to write now: `False`",
        "",
        "## Recommendation",
        "",
        "Review name-matched box mismatches manually. Any apply step still requires a DB backup, exact-row allowlist, and explicit approval.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_box_mismatch_diagnosis(
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
    skipped_non_mismatch = 0
    db_state = {
        "db_path": str(db_resolved),
        "quick_check": None,
        "read_only": True,
        "query_only": True,
    }
    with _connect_read_only(db_resolved) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        db_state["quick_check"] = quick_check[0] if quick_check else None
        if db_state["quick_check"] != "ok":
            failures.append("db_quick_check_failed")
        for candidate in _list(preflight.get("candidate_preflight")):
            if not isinstance(candidate, Mapping):
                continue
            if not _is_box_mismatch_candidate(candidate):
                skipped_non_mismatch += 1
                continue
            candidates.append(_diagnose_candidate(conn, candidate))

    name_ready = sum(
        1 for item in candidates if item.get("status") == "NAME_MATCHED_BOX_MISMATCH_REVIEW_READY"
    )
    manual = len(candidates) - name_ready
    proposed_update_rows = sum(
        len(_list(item.get("proposed_no_write_updates"))) for item in candidates
    )
    status = (
        "NOT_READY"
        if failures
        else "REPORT_ONLY_NAME_MATCHED_BOX_MISMATCH_REVIEW_READY"
        if name_ready and not manual
        else "REPORT_ONLY_BOX_MISMATCH_MANUAL_REVIEW_REQUIRED"
        if candidates
        else "REPORT_ONLY_NO_BOX_MISMATCH_CANDIDATES"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "failures": failures,
        "report_only": True,
        "source_evidence": {
            "preflight_packet": str(preflight_resolved),
            "db": str(db_resolved),
        },
        "db_state": db_state,
        "summary": {
            "candidate_count": len(candidates),
            "name_matched_review_ready_count": name_ready,
            "manual_review_count": manual,
            "proposed_update_row_count": proposed_update_rows,
            "skipped_non_box_mismatch_count": skipped_non_mismatch,
            "safe_to_write_now_count": 0,
        },
        "candidates": candidates,
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": [
            "label_write",
            "metadata_write",
            "box_number_update",
            "finish_position_update",
            "model_training_or_promotion",
            "betting_decision",
        ],
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
        raise SystemExit("refusing diagnosis while APPROVE_RESULT_LABEL_WRITE is set")
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_box_mismatch_diagnosis(
        preflight_packet_path=Path(args.preflight_packet),
        db_path=Path(args.db),
    )
    output = _assert_output_path_safe(Path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _assert_output_path_safe(Path(args.report))
    report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report, packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
