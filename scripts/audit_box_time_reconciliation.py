#!/usr/bin/env python3
"""Audit whether timing-only rows can be safely reconciled with box rows.

This is a report-only diagnostic. It opens SQLite in read-only/query-only mode
and only writes an optional JSON report path requested by the operator.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "greyhound_racing_data.db"


@dataclass(frozen=True)
class AuditRow:
    table: str
    row_id: str
    race_id: str | None
    race_date: str | None
    venue: str | None
    race_number: int | None
    dog_name: str | None
    dog_clean_name: str | None
    box_number: int | None
    time_value: float | None
    dog_source: str | None
    race_source: str | None

    @property
    def source_key(self) -> str:
        dog_source = self.dog_source or "DATA_MISSING"
        race_source = self.race_source or "DATA_MISSING"
        return f"{self.table}:{dog_source}|race_metadata:{race_source}"

    @property
    def normalized_dog(self) -> str | None:
        return normalize_dog(self.dog_clean_name or self.dog_name)

    @property
    def race_id_dog_key(self) -> tuple[str, str] | None:
        dog = self.normalized_dog
        race_id = normalized_text(self.race_id)
        if not dog or not race_id:
            return None
        return (race_id, dog)

    @property
    def canonical_dog_key(self) -> tuple[str, str, int, str] | None:
        dog = self.normalized_dog
        race_date = normalized_text(self.race_date)
        venue = normalized_text(self.venue)
        if not dog or not race_date or not venue or self.race_number is None:
            return None
        return (race_date, venue, self.race_number, dog)


def normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", text).upper()


def normalize_dog(value: Any) -> str | None:
    text = normalized_text(value)
    if not text:
        return None
    text = re.sub(r"^\s*\d+\s*[\.)-]?\s*", "", text)
    text = text.strip(" '\"\t\r\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def safe_time(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        parsed = float(match.group(0))
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def box_band(value: int | None) -> str | None:
    if value in (1, 2):
        return "inside"
    if value in (3, 4, 5, 6):
        return "middle"
    if value is not None and value >= 7:
        return "outside"
    return None


def sqlite_ro(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def load_production_rows(connection: sqlite3.Connection) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for row in connection.execute(
        """
        SELECT
          d.id AS row_id,
          d.race_id AS race_id,
          rm.race_date AS race_date,
          rm.venue AS venue,
          rm.race_number AS race_number,
          d.dog_name AS dog_name,
          d.dog_clean_name AS dog_clean_name,
          d.box_number AS box_number,
          d.individual_time AS individual_time,
          d.winning_time AS winning_time,
          d.best_time AS best_time,
          d.data_source AS dog_source,
          rm.data_source AS race_source
        FROM dog_race_data d
        LEFT JOIN race_metadata rm ON rm.race_id = d.race_id
        """
    ):
        time_value = (
            safe_time(row["individual_time"])
            or safe_time(row["winning_time"])
            or safe_time(row["best_time"])
        )
        rows.append(
            AuditRow(
                table="dog_race_data",
                row_id=str(row["row_id"]),
                race_id=row["race_id"],
                race_date=row["race_date"],
                venue=row["venue"],
                race_number=safe_int(row["race_number"]),
                dog_name=row["dog_name"],
                dog_clean_name=row["dog_clean_name"],
                box_number=safe_int(row["box_number"]),
                time_value=time_value,
                dog_source=row["dog_source"],
                race_source=row["race_source"],
            )
        )
    return rows


def load_staging_rows(connection: sqlite3.Connection) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for row in connection.execute(
        """
        SELECT
          id AS row_id,
          race_id,
          race_date,
          venue,
          race_number,
          dog_name,
          dog_clean_name,
          box_number,
          individual_time,
          data_source
        FROM csv_dog_history_staging
        """
    ):
        rows.append(
            AuditRow(
                table="csv_dog_history_staging",
                row_id=str(row["row_id"]),
                race_id=row["race_id"],
                race_date=row["race_date"],
                venue=row["venue"],
                race_number=safe_int(row["race_number"]),
                dog_name=row["dog_name"],
                dog_clean_name=row["dog_clean_name"],
                box_number=safe_int(row["box_number"]),
                time_value=safe_time(row["individual_time"]),
                dog_source=row["data_source"],
                race_source=row["data_source"],
            )
        )
    return rows


def table_source_summary(rows: Sequence[AuditRow]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for source, grouped in group_rows(rows, lambda row: row.source_key).items():
        summary[source] = {
            "rows": len(grouped),
            "box_rows": sum(1 for row in grouped if row.box_number is not None),
            "timing_rows": sum(1 for row in grouped if row.time_value is not None),
            "both_rows": sum(
                1 for row in grouped if row.box_number is not None and row.time_value is not None
            ),
            "canonical_identity_rows": sum(1 for row in grouped if row.canonical_dog_key),
            "race_id_identity_rows": sum(1 for row in grouped if row.race_id_dog_key),
        }
    return dict(sorted(summary.items()))


def group_rows(
    rows: Iterable[AuditRow],
    key_func: Any,
) -> dict[Any, list[AuditRow]]:
    grouped: dict[Any, list[AuditRow]] = defaultdict(list)
    for row in rows:
        key = key_func(row)
        if key is not None:
            grouped[key].append(row)
    return grouped


def join_audit(
    *,
    name: str,
    timing_rows: Sequence[AuditRow],
    box_rows: Sequence[AuditRow],
    key_name: str,
) -> dict[str, Any]:
    if key_name == "race_id_dog":
        key_func = lambda row: row.race_id_dog_key
    elif key_name == "canonical_date_venue_race_number_dog":
        key_func = lambda row: row.canonical_dog_key
    else:
        raise ValueError(f"unknown key_name: {key_name}")

    timing_with_key = [row for row in timing_rows if key_func(row) is not None]
    box_index = group_rows(box_rows, key_func)
    matched_timing_ids: set[str] = set()
    safe_rows: list[tuple[AuditRow, AuditRow]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    pair_count = 0

    for timing in timing_with_key:
        matches = box_index.get(key_func(timing), [])
        if not matches:
            continue
        pair_count += len(matches)
        matched_timing_ids.add(timing.row_id)
        distinct_boxes = {match.box_number for match in matches if match.box_number is not None}
        if len(matches) == 1 and len(distinct_boxes) == 1:
            safe_rows.append((timing, matches[0]))
        else:
            ambiguous_rows.append(
                {
                    "timing_row_id": timing.row_id,
                    "key": key_func(timing),
                    "match_count": len(matches),
                    "distinct_boxes": sorted(box for box in distinct_boxes if box is not None),
                }
            )

    safe_box_bands = Counter(box_band(match.box_number) or "DATA_MISSING" for _, match in safe_rows)
    timing_sources = Counter(timing.source_key for timing, _ in safe_rows)
    box_sources = Counter(match.source_key for _, match in safe_rows)
    examples = [
        {
            "timing_table": timing.table,
            "timing_row_id": timing.row_id,
            "timing_source": timing.source_key,
            "box_table": match.table,
            "box_row_id": match.row_id,
            "box_source": match.source_key,
            "race_id": timing.race_id,
            "race_date": timing.race_date,
            "venue": timing.venue,
            "race_number": timing.race_number,
            "dog": timing.normalized_dog,
            "time_value": timing.time_value,
            "box_number": match.box_number,
            "box_band": box_band(match.box_number),
        }
        for timing, match in safe_rows[:10]
    ]

    return {
        "name": name,
        "key": key_name,
        "timing_rows": len(timing_rows),
        "timing_rows_with_key": len(timing_with_key),
        "box_rows": len(box_rows),
        "box_keys": len(box_index),
        "matched_pairs": pair_count,
        "matched_timing_rows": len(matched_timing_ids),
        "safe_recoverable_timing_rows": len(safe_rows),
        "ambiguous_timing_rows": len(ambiguous_rows),
        "safe_box_band_counts": dict(sorted(safe_box_bands.items())),
        "safe_timing_source_counts": dict(sorted(timing_sources.items())),
        "safe_box_source_counts": dict(sorted(box_sources.items())),
        "examples": examples,
        "ambiguous_examples": ambiguous_rows[:10],
    }


def build_report(db_path: Path) -> dict[str, Any]:
    with sqlite_ro(db_path) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        production_rows = load_production_rows(connection)
        staging_rows = load_staging_rows(connection)

    production_timing = [row for row in production_rows if row.time_value is not None]
    production_box = [row for row in production_rows if row.box_number is not None]
    staging_timing = [row for row in staging_rows if row.time_value is not None]
    staging_box = [row for row in staging_rows if row.box_number is not None]

    joins = [
        join_audit(
            name="production_timing_to_production_box_by_race_id_dog",
            timing_rows=production_timing,
            box_rows=production_box,
            key_name="race_id_dog",
        ),
        join_audit(
            name="production_timing_to_production_box_by_canonical_identity",
            timing_rows=production_timing,
            box_rows=production_box,
            key_name="canonical_date_venue_race_number_dog",
        ),
        join_audit(
            name="production_timing_to_staging_box_by_race_id_dog",
            timing_rows=production_timing,
            box_rows=staging_box,
            key_name="race_id_dog",
        ),
        join_audit(
            name="production_timing_to_staging_box_by_canonical_identity",
            timing_rows=production_timing,
            box_rows=staging_box,
            key_name="canonical_date_venue_race_number_dog",
        ),
        join_audit(
            name="staging_timing_to_staging_box_by_race_id_dog",
            timing_rows=staging_timing,
            box_rows=staging_box,
            key_name="race_id_dog",
        ),
        join_audit(
            name="staging_timing_to_staging_box_by_canonical_identity",
            timing_rows=staging_timing,
            box_rows=staging_box,
            key_name="canonical_date_venue_race_number_dog",
        ),
    ]

    externally_recoverable = [
        join
        for join in joins
        if join["name"].startswith("production_timing")
        and join["safe_recoverable_timing_rows"] > 0
    ]
    staging_self_contained = [
        join
        for join in joins
        if join["name"].startswith("staging_timing")
        and join["safe_recoverable_timing_rows"] > 0
    ]

    if externally_recoverable:
        verdict = "SAFE_RECOVERY_CANDIDATE_REQUIRES_FEATURE_PIPELINE_REVIEW"
        blocked_reason = None
    else:
        verdict = "DATA_MISSING"
        blocked_reason = (
            "No strict join recovered boxes for production timing rows by race_id+dog or "
            "date+venue+race_number+dog. Production timing rows and box rows remain disjoint "
            "for the identity fields available to Phase 1."
        )

    return {
        "schema_version": "box_time_reconciliation_audit_v1",
        "report_only": True,
        "db_path": str(db_path),
        "sqlite_quick_check": quick_check,
        "verdict": verdict,
        "blocked_reason": blocked_reason,
        "production_source_summary": table_source_summary(production_rows),
        "staging_source_summary": table_source_summary(staging_rows),
        "join_audits": joins,
        "safe_recovery_count": sum(
            join["safe_recoverable_timing_rows"] for join in externally_recoverable
        ),
        "staging_self_contained_safe_rows": max(
            [join["safe_recoverable_timing_rows"] for join in staging_self_contained] or [0]
        ),
        "interpretation": {
            "production_timing_rows_can_be_reconciled": bool(externally_recoverable),
            "staging_has_self_contained_box_time_rows": bool(staging_self_contained),
            "staging_self_contained_note": (
                "Staging rows with both box and time do not by themselves repair production "
                "Phase 1 rows unless they strictly overlap the target/history identity space."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.db)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["sqlite_quick_check"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
