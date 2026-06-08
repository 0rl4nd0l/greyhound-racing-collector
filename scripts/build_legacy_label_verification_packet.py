#!/usr/bin/env python3
"""Build a read-only packet for legacy greyhound label verification.

The verifier inspects legacy labelled rows and optionally compares result-like
sources against an official-reference SQLite DB. It does not scrape, write
labels, promote labels, mutate snapshots, train, or update registries.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "legacy_label_verification_packet_v1"

WRITES_PERFORMED = {
    "db_write": False,
    "label_promotion": False,
    "snapshot_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "scrape_or_fetch": False,
}

RESULT_LIKE_SOURCES = {
    "enhanced_processor_with_results",
    "navigator_results",
    "completed_race_update",
}

PARTIAL_SOURCES = {
    "sportsbet_results_top4",
    "partial_sportsbet_results",
}

HISTORY_ONLY_SOURCES = {
    "embedded_form_guide",
}


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _select_existing(columns: set[str], wanted: Iterable[str]) -> list[str]:
    return [column for column in wanted if column in columns]


def _norm_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _position_from_row(row: Mapping[str, Any]) -> int | None:
    for key in ("finish_position", "placing", "scraped_finish_position"):
        position = _safe_int(row.get(key))
        if position is not None:
            return position
    return None


def _source_name(value: Any) -> str:
    return str(value) if value not in (None, "") else "NULL"


def _source_where_clause(source: str) -> tuple[str, tuple[Any, ...]]:
    if source == "NULL":
        return "(data_source IS NULL OR data_source = '')", ()
    return "data_source = ?", (source,)


def _label_expr(columns: set[str]) -> str | None:
    clauses = [
        f"{column} IS NOT NULL"
        for column in ("finish_position", "placing", "scraped_finish_position")
        if column in columns
    ]
    if not clauses:
        return None
    return " OR ".join(clauses)


def _race_metadata(conn: sqlite3.Connection, race_id: str) -> dict[str, Any]:
    columns = _table_columns(conn, "race_metadata")
    wanted = _select_existing(
        columns,
        ["race_id", "race_date", "results_status", "winner_name", "winner_source"],
    )
    if not wanted:
        return {}
    select_clause = ", ".join(wanted)
    row = conn.execute(
        f"SELECT {select_clause} FROM race_metadata WHERE race_id = ?",
        (race_id,),
    ).fetchone()
    return dict(row) if row else {}


def _labelled_race_sources(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "dog_race_data")
    expr = _label_expr(columns)
    if not expr:
        return []
    if "data_source" not in columns:
        source_expr = "'NULL'"
    else:
        source_expr = "COALESCE(NULLIF(data_source, ''), 'NULL')"
    rows = conn.execute(
        f"""
        SELECT
            race_id,
            {source_expr} AS data_source,
            COUNT(*) AS runner_rows
        FROM dog_race_data
        WHERE {expr}
        GROUP BY race_id, {source_expr}
        ORDER BY race_id, data_source
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _runner_labels_for_source(
    conn: sqlite3.Connection,
    race_id: str,
    source: str | None = None,
    *,
    official_only: bool = False,
) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "dog_race_data")
    wanted = _select_existing(
        columns,
        [
            "race_id",
            "dog_name",
            "dog_clean_name",
            "box_number",
            "finish_position",
            "placing",
            "scraped_finish_position",
            "data_source",
        ],
    )
    expr = _label_expr(columns)
    if not wanted or not expr:
        return []

    clauses = ["race_id = ?", f"({expr})"]
    params: list[Any] = [race_id]
    if official_only:
        if "data_source" not in columns:
            return []
        clauses.append("data_source = 'thedogs_official'")
    elif source is not None and "data_source" in columns:
        source_clause, source_params = _source_where_clause(source)
        clauses.append(source_clause)
        params.extend(source_params)

    select_clause = ", ".join(wanted)
    rows = conn.execute(
        f"SELECT {select_clause} FROM dog_race_data WHERE {' AND '.join(clauses)}",
        tuple(params),
    ).fetchall()

    labels: list[dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        position = _position_from_row(data)
        if position is None:
            continue
        labels.append(
            {
                "dog_name": data.get("dog_name") or data.get("dog_clean_name"),
                "dog_key": _norm_name(data.get("dog_clean_name") or data.get("dog_name")),
                "box_number": data.get("box_number"),
                "finish_position": position,
                "data_source": _source_name(data.get("data_source")),
            }
        )
    return labels


def _label_key(label: Mapping[str, Any]) -> str:
    box_number = label.get("box_number")
    if box_number is not None:
        return f"box:{box_number}"
    return f"dog:{label.get('dog_key') or _norm_name(label.get('dog_name'))}"


def _compare_to_official(
    legacy_labels: list[Mapping[str, Any]],
    official_labels: list[Mapping[str, Any]] | None,
    *,
    official_db_available: bool,
) -> dict[str, Any]:
    if not official_db_available:
        return {
            "status": "NO_OFFICIAL_REFERENCE_DB",
            "legacy_rows": len(legacy_labels),
            "official_reference_rows": 0,
            "mismatches": [],
        }
    if not official_labels:
        return {
            "status": "OFFICIAL_REFERENCE_MISSING",
            "legacy_rows": len(legacy_labels),
            "official_reference_rows": 0,
            "mismatches": [],
        }

    legacy_by_key = {_label_key(label): label for label in legacy_labels}
    official_by_key = {_label_key(label): label for label in official_labels}
    mismatches: list[dict[str, Any]] = []

    for key in sorted(set(legacy_by_key) | set(official_by_key)):
        legacy = legacy_by_key.get(key)
        official = official_by_key.get(key)
        if legacy is None:
            mismatches.append(
                {
                    "key": key,
                    "reason": "missing_legacy_runner",
                    "official_finish_position": official.get("finish_position") if official else None,
                }
            )
            continue
        if official is None:
            mismatches.append(
                {
                    "key": key,
                    "reason": "missing_official_runner",
                    "legacy_finish_position": legacy.get("finish_position"),
                }
            )
            continue
        if legacy.get("finish_position") != official.get("finish_position"):
            mismatches.append(
                {
                    "key": key,
                    "reason": "finish_position_mismatch",
                    "legacy_finish_position": legacy.get("finish_position"),
                    "official_finish_position": official.get("finish_position"),
                }
            )

    legacy_winners = sum(1 for label in legacy_labels if label.get("finish_position") == 1)
    official_winners = sum(1 for label in official_labels if label.get("finish_position") == 1)
    if legacy_winners != 1 or official_winners != 1:
        mismatches.append(
            {
                "reason": "winner_count_not_one",
                "legacy_winner_count": legacy_winners,
                "official_winner_count": official_winners,
            }
        )

    return {
        "status": "MATCH" if not mismatches else "MISMATCH",
        "legacy_rows": len(legacy_labels),
        "official_reference_rows": len(official_labels),
        "mismatches": mismatches[:25],
    }


def _classify_source(
    source: str,
    metadata: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> tuple[str, str]:
    results_status = str(metadata.get("results_status") or "").lower()
    winner_source = str(metadata.get("winner_source") or "").lower()
    lowered_source = source.lower()

    if source == "thedogs_official":
        return "clean_official_already", "already_thedogs_official"
    if source in PARTIAL_SOURCES or "sportsbet" in lowered_source or "partial_sportsbet" in results_status or "sportsbet" in winner_source:
        return "partial_or_winner_only", "partial_or_winner_only_source"
    if source in HISTORY_ONLY_SOURCES or "reference_training_data.parquet" in source:
        return "embedded_history_only", "embedded_form_guide_not_result_label"
    if source == "NULL":
        return "legacy_unknown_provenance", "legacy_null_source_requires_reverification"
    if source in RESULT_LIKE_SOURCES:
        status = verification.get("status")
        if status == "MATCH":
            return "verified_official_candidate", "result_like_source_matches_official_reference"
        if status == "MISMATCH":
            return "official_mismatch", "result_like_source_mismatches_official_reference"
        return "result_like_reverify_candidate", "result_like_source_needs_official_reference"
    if verification.get("status") == "MATCH":
        return "verified_official_candidate", "other_source_matches_official_reference"
    if verification.get("status") == "MISMATCH":
        return "official_mismatch", "other_source_mismatches_official_reference"
    return "other_reverify_candidate", "source_requires_reverification"


def _database_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if _table_exists(conn, "race_metadata"):
        summary["race_metadata_count"] = conn.execute(
            "SELECT COUNT(*) FROM race_metadata"
        ).fetchone()[0]
    if _table_exists(conn, "dog_race_data"):
        summary["dog_race_data_count"] = conn.execute(
            "SELECT COUNT(*) FROM dog_race_data"
        ).fetchone()[0]
        columns = _table_columns(conn, "dog_race_data")
        expr = _label_expr(columns)
        if expr:
            summary["labelled_distinct_races"] = conn.execute(
                f"SELECT COUNT(DISTINCT race_id) FROM dog_race_data WHERE {expr}"
            ).fetchone()[0]
            summary["labelled_runner_rows"] = conn.execute(
                f"SELECT COUNT(*) FROM dog_race_data WHERE {expr}"
            ).fetchone()[0]
    return summary


def _append_example(
    examples: dict[str, list[dict[str, Any]]],
    classification: str,
    item: dict[str, Any],
    max_examples_per_class: int | None,
) -> None:
    if max_examples_per_class is None:
        examples[classification].append(item)
        return
    if len(examples[classification]) < max_examples_per_class:
        examples[classification].append(item)


def build_packet(
    *,
    legacy_db_paths: list[Path],
    official_db_path: Path | None = None,
    max_examples_per_class: int | None = None,
) -> dict[str, Any]:
    official_conn: sqlite3.Connection | None = None
    official_db_available = official_db_path is not None
    failures: list[str] = []
    warnings: list[str] = []

    if official_db_path is not None:
        try:
            official_conn = _open_readonly(official_db_path)
        except Exception as exc:  # noqa: BLE001 - report records exact blocker.
            official_db_available = False
            official_conn = None
            failures.append(f"official_db_unreadable:{type(exc).__name__}")

    classification_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    db_summaries: list[dict[str, Any]] = []
    race_classifications: list[dict[str, Any]] = []
    classification_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_runner_rows = 0

    for legacy_db_path in legacy_db_paths:
        resolved = legacy_db_path.expanduser().resolve()
        db_summary: dict[str, Any] = {
            "path": str(resolved),
            "exists": resolved.exists(),
            "read_only": True,
        }
        if not resolved.exists():
            failures.append(f"legacy_db_missing:{resolved}")
            db_summaries.append(db_summary)
            continue
        try:
            conn = _open_readonly(resolved)
        except Exception as exc:  # noqa: BLE001 - report records exact blocker.
            failures.append(f"legacy_db_unreadable:{resolved}:{type(exc).__name__}")
            db_summaries.append(db_summary)
            continue

        try:
            db_summary.update(_database_summary(conn))
            race_sources = _labelled_race_sources(conn)
            db_summary["labelled_race_source_groups"] = len(race_sources)
            for race_source in race_sources:
                race_id = str(race_source.get("race_id") or "")
                source = _source_name(race_source.get("data_source"))
                if not race_id:
                    continue
                source_counts[source] += 1
                metadata = _race_metadata(conn, race_id)
                legacy_labels = _runner_labels_for_source(conn, race_id, source)
                official_labels = (
                    _runner_labels_for_source(
                        official_conn,
                        race_id,
                        official_only=True,
                    )
                    if official_conn is not None
                    else None
                )
                verification = _compare_to_official(
                    legacy_labels,
                    official_labels,
                    official_db_available=official_db_available,
                )
                classification, reason = _classify_source(source, metadata, verification)
                classification_counts[classification] += 1
                if classification == "verified_official_candidate":
                    verified_runner_rows += len(legacy_labels)
                item = {
                    "race_id": race_id,
                    "legacy_db_path": str(resolved),
                    "source": source,
                    "classification": classification,
                    "reason": reason,
                    "legacy_runner_rows": len(legacy_labels),
                    "metadata": metadata,
                    "verification": verification,
                }
                race_classifications.append(item)
                _append_example(
                    classification_examples,
                    classification,
                    item,
                    max_examples_per_class,
                )
        finally:
            conn.close()
        db_summaries.append(db_summary)

    if official_conn is not None:
        official_conn.close()

    races_scanned = sum(classification_counts.values())
    status = "REPORT_ONLY"
    if not legacy_db_paths:
        status = "DATA_MISSING"
        failures.append("no_legacy_db_paths_supplied")

    if max_examples_per_class is not None:
        displayed_races = [
            item
            for items in classification_examples.values()
            for item in items
        ]
    else:
        displayed_races = race_classifications

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "read_only_safety": {
            "sqlite_mode": "mode=ro + PRAGMA query_only=ON",
            "scrape_or_fetch_performed": False,
            "output_writes_only": True,
        },
        "source_evidence": {
            "legacy_db_paths": [str(Path(path).expanduser().resolve()) for path in legacy_db_paths],
            "official_db_path": (
                str(official_db_path.expanduser().resolve())
                if official_db_path is not None
                else None
            ),
            "official_db_available": official_db_available,
        },
        "db_summaries": db_summaries,
        "summary": {
            "races_scanned": races_scanned,
            "classification_counts": dict(sorted(classification_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "verified_official_candidates": {
                "races": classification_counts.get("verified_official_candidate", 0),
                "runner_rows": verified_runner_rows,
            },
            "not_clean_now_races": races_scanned
            - classification_counts.get("verified_official_candidate", 0)
            - classification_counts.get("clean_official_already", 0),
        },
        "race_classifications": displayed_races,
        "classification_examples": dict(classification_examples),
        "writes_performed": dict(WRITES_PERFORMED),
        "recommended_next_actions": [
            "Do not promote legacy labels unless they are verified against official full-position results.",
            "Use verified_official_candidate rows only as candidates for a separate approved label-upgrade plan.",
            "Keep embedded_form_guide rows for history feature reconstruction, not clean result labels.",
            "Treat result_like_reverify_candidate rows as official-parser backfill targets before training.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-db",
        action="append",
        required=True,
        help="Legacy SQLite DB to inspect read-only. Repeatable.",
    )
    parser.add_argument(
        "--official-db",
        help="Optional official-reference SQLite DB. Rows must be tagged thedogs_official.",
    )
    parser.add_argument("--output", required=True, help="JSON packet output path")
    parser.add_argument(
        "--max-examples-per-class",
        type=int,
        default=50,
        help="Limit detailed race examples per class in CLI output. Use 0 for all.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_examples: int | None = args.max_examples_per_class
    if max_examples == 0:
        max_examples = None
    packet = build_packet(
        legacy_db_paths=[Path(path) for path in args.legacy_db],
        official_db_path=Path(args.official_db) if args.official_db else None,
        max_examples_per_class=max_examples,
    )
    text = json.dumps(packet, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if packet.get("status") == "REPORT_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
