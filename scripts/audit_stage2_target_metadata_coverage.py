#!/usr/bin/env python3
"""Audit Stage 2 target metadata coverage without mutating prediction state."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525"
    / "stage2_feature_upgrade_packet_20260615T_final_remediation_local"
    / "phase_1_dataset_hardening/repaired_dataset_v2.csv"
)
DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525"
    / "stage2_target_metadata_coverage_audit_local"
)
SAFE_OUTPUT_PARENT = "artifacts/full_evidence_orchestration_20260525"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_output_dir(path: Path) -> Path:
    logical = path if path.is_absolute() else ROOT / path
    logical = logical.absolute()
    try:
        relative = logical.relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    text = relative.as_posix()
    if not text.startswith(SAFE_OUTPUT_PARENT + "/"):
        raise ValueError(f"output_dir_must_be_under:{SAFE_OUTPUT_PARENT}")
    if text.startswith("artifacts/prediction_snapshots/"):
        raise ValueError("output_dir_must_not_be_prediction_snapshots")
    return logical


def safe_int(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def parse_race_number(row: Mapping[str, Any]) -> int | None:
    for value in (row.get("race_number"), row.get("race_id"), row.get("snapshot_instance_id")):
        parsed = safe_int(value)
        if parsed is not None:
            return parsed
    return None


def current_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidates = [Path(text)]
    marker = "artifacts/"
    if marker in text:
        candidates.append(ROOT / text[text.index(marker) :])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def snapshot_source_csv(row: Mapping[str, Any]) -> Path | None:
    snapshot_path = current_path(row.get("snapshot_path"))
    if snapshot_path is None:
        return None
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return current_path(payload.get("source_file_path"))


def sidecar_status(csv_path: Path | None) -> str:
    if csv_path is None:
        return "source_csv_missing"
    sidecar = Path(str(csv_path) + ".metadata.json")
    if not sidecar.exists():
        return "sidecar_missing"
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return "sidecar_unreadable"
    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), dict) else {}
    has_distance = bool(payload.get("target_distance") or race_info.get("distance") or race_info.get("target_distance"))
    has_grade = bool(payload.get("target_grade") or race_info.get("grade") or race_info.get("target_grade"))
    if payload.get("metadata_is_leakage_safe") is True and has_distance and has_grade:
        return "verified_distance_grade"
    if payload.get("metadata_is_leakage_safe") is True:
        return "verified_missing_distance_or_grade"
    return "unsafe_or_unverified"


def sqlite_ro(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def db_candidate_status(
    connection: sqlite3.Connection,
    *,
    race_date: str,
    venue: str,
    race_number: int | None,
) -> dict[str, Any]:
    if not race_date or not venue or race_number is None:
        return {"status": "missing_identity", "candidate_rows": 0, "full_context_rows": 0}
    candidates = [
        dict(row)
        for row in connection.execute(
            """
            SELECT race_id, grade, distance, data_source, url
            FROM race_metadata
            WHERE race_date = ? AND upper(venue) = upper(?) AND race_number = ?
            """,
            (race_date, venue, race_number),
        )
    ]
    full = [
        row
        for row in candidates
        if str(row.get("grade") or "").strip() and str(row.get("distance") or "").strip()
    ]
    if len(full) == 1:
        status = "safe_exact_race_number_context_available"
    elif len(candidates) == 1:
        status = "exact_race_number_row_missing_distance_or_grade"
    elif len(candidates) > 1:
        status = "ambiguous_exact_race_number_rows"
    else:
        status = "no_exact_race_number_row"
    return {
        "status": status,
        "candidate_rows": len(candidates),
        "full_context_rows": len(full),
        "data_sources": sorted({str(row.get("data_source") or "MISSING") for row in candidates}),
    }


def embedded_context_status(connection: sqlite3.Connection, *, race_date: str, venue: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT race_id, grade, distance
            FROM race_metadata
            WHERE race_date = ?
              AND upper(venue) = upper(?)
              AND race_number IS NULL
              AND data_source = 'embedded_form_guide'
              AND grade IS NOT NULL
              AND trim(grade) != ''
              AND distance IS NOT NULL
              AND trim(distance) != ''
            """,
            (race_date, venue),
        )
    ]
    pairs = sorted({(str(row.get("distance")), str(row.get("grade"))) for row in rows})
    return {
        "embedded_context_rows": len(rows),
        "unique_distance_grade_pairs": len(pairs),
        "status": "unsafe_unmapped_embedded_form_context_present" if rows else "none",
    }


def audit_rows(rows: Sequence[Mapping[str, Any]], connection: sqlite3.Connection) -> dict[str, Any]:
    missing = [
        row
        for row in rows
        if not row.get("target_distance_safe") or not row.get("target_grade_safe")
    ]
    db_status_counts: Counter[str] = Counter()
    sidecar_status_counts: Counter[str] = Counter()
    embedded_status_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    safe_recoverable = 0
    for row in missing:
        race_date = str(row.get("race_date") or "")
        venue = str(row.get("venue") or "")
        race_number = parse_race_number(row)
        db_status = db_candidate_status(
            connection,
            race_date=race_date,
            venue=venue,
            race_number=race_number,
        )
        embedded_status = embedded_context_status(connection, race_date=race_date, venue=venue)
        sidecar = sidecar_status(snapshot_source_csv(row))
        db_status_counts[db_status["status"]] += 1
        sidecar_status_counts[sidecar] += 1
        embedded_status_counts[embedded_status["status"]] += 1
        if db_status["status"] == "safe_exact_race_number_context_available" or sidecar == "verified_distance_grade":
            safe_recoverable += 1
        if len(examples) < 20:
            examples.append(
                {
                    "race_id": row.get("race_id"),
                    "race_date": race_date,
                    "venue": venue,
                    "race_number": race_number,
                    "db_status": db_status,
                    "sidecar_status": sidecar,
                    "embedded_context_status": embedded_status,
                }
            )
    return {
        "schema_version": "stage2_target_metadata_coverage_audit_v1",
        "report_only": True,
        "rows": len(rows),
        "missing_target_metadata_rows": len(missing),
        "safe_recoverable_rows_from_existing_sources": safe_recoverable,
        "db_exact_race_number_status_counts": dict(db_status_counts),
        "sidecar_status_counts": dict(sidecar_status_counts),
        "embedded_context_status_counts": dict(embedded_status_counts),
        "verdict": "DATA_MISSING" if safe_recoverable == 0 else "SAFE_REPAIR_AVAILABLE",
        "notes": [
            "Embedded form-guide DIST/G rows are historical form fields and are not accepted as target race metadata without a strict current-race mapping.",
            "This audit reads only existing packet, sidecar, and read-only DB rows and performs no DB, label, snapshot, manifest, odds, EV, registry, or model writes.",
        ],
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    output_dir = safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.packet)
    connection = sqlite_ro(args.db)
    try:
        report = audit_rows(rows, connection)
    finally:
        connection.close()
    write_json(output_dir / "target_metadata_coverage_audit_v1.json", report)
    return 0 if report["verdict"] == "SAFE_REPAIR_AVAILABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
