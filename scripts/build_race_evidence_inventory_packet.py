#!/usr/bin/env python3
"""Build a report-only inventory of race evidence across artifacts and DB tables.

This packet answers a narrow operational question: which races do we already
have in shadow predictions, official-result artifacts, append-only
official-result evidence, and strict pre-jump odds?

It only writes report artifacts. It does not append evidence, write labels,
capture odds/results, train, promote, update registries, emit EV, or control the
daemon.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import append_official_result_evidence_backlog as backlog  # noqa: E402
from scripts.build_unified_evidence_dataset import validate_odds_row  # noqa: E402


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/race_evidence_inventory_"
REPORT_FILE = "race_evidence_inventory_report.json"
CSV_FILE = "race_evidence_inventory.csv"
JSONL_FILE = "race_evidence_inventory.jsonl"
SCORECARD_CSV_FILE = "race_evidence_scorecard.csv"
SCORECARD_JSONL_FILE = "race_evidence_scorecard.jsonl"
SUMMARY_FILE = "SUMMARY.md"
NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "official_result_write": False,
    "daemon_control": False,
    "betting_or_ev_action": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        logical = path if path.is_absolute() else ROOT / path
        return logical.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError:
        return str(path)


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_race_evidence_inventory_artifact:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "race_evidence_inventory_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(str(value)))
        except Exception:
            return None


def race_record(records: dict[str, dict[str, Any]], race_id: Any) -> dict[str, Any] | None:
    race_id_text = str(race_id or "").strip()
    if not race_id_text:
        return None
    return records.setdefault(
        race_id_text,
        {
            "race_id": race_id_text,
            "race_date": None,
            "venue": None,
            "race_number": None,
            "shadow_prediction_rows": 0,
            "shadow_prediction_paths": set(),
            "shadow_boxes": set(),
            "latest_shadow_prediction_by_box": {},
            "official_artifact_race_rows": 0,
            "official_artifact_runner_rows": 0,
            "official_artifact_paths": set(),
            "official_artifact_boxes": set(),
            "official_result_db_race_rows": 0,
            "official_result_db_runner_rows": 0,
            "official_result_db_boxes": set(),
            "official_result_by_box": {},
            "official_result_conflict_count": 0,
            "live_odds_rows": 0,
            "strict_live_odds_rows": 0,
            "live_odds_boxes": set(),
            "strict_live_odds_boxes": set(),
            "strict_odds_by_box": {},
        },
    )


def set_first(record: dict[str, Any], key: str, value: Any) -> None:
    if record.get(key) in (None, "") and value not in (None, ""):
        record[key] = value


def add_box(record: dict[str, Any], key: str, value: Any) -> None:
    parsed = parse_int(value)
    if parsed is not None:
        record[key].add(parsed)


def add_path(record: dict[str, Any], key: str, path: Path) -> None:
    record[key].add(relpath(path) or str(path))


def sorted_prediction_paths(artifact_roots: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in artifact_roots:
        logical = root if root.is_absolute() else ROOT / root
        if not logical.exists():
            continue
        for name in ("stage2_shadow_predictions.jsonl", "shadow_predictions.jsonl"):
            paths.update(path for path in logical.rglob(name) if path.is_file())
    return sorted(paths, key=lambda path: path.as_posix())


def scan_shadow_predictions(
    *,
    artifact_roots: Sequence[Path],
    records: dict[str, dict[str, Any]],
    max_prediction_files: int | None,
) -> dict[str, Any]:
    paths = sorted_prediction_paths(artifact_roots)
    truncated = False
    if max_prediction_files is not None and max_prediction_files >= 0:
        truncated = len(paths) > max_prediction_files
        paths = paths[:max_prediction_files]
    row_count = 0
    race_ids: set[str] = set()
    path_kind_counts = Counter()
    for path in paths:
        path_kind_counts[path.name] += 1
        source_mtime = path.stat().st_mtime
        for row in load_jsonl(path):
            record = race_record(records, row.get("race_id"))
            if record is None:
                continue
            row_count += 1
            race_ids.add(record["race_id"])
            record["shadow_prediction_rows"] += 1
            add_path(record, "shadow_prediction_paths", path)
            box_number = parse_int(
                row.get("box") if row.get("box") not in (None, "") else row.get("box_number")
            )
            add_box(record, "shadow_boxes", box_number)
            if box_number is not None:
                latest_by_box = record["latest_shadow_prediction_by_box"]
                existing = latest_by_box.get(box_number)
                candidate = {
                    "row": dict(row),
                    "source_path": relpath(path),
                    "source_mtime": source_mtime,
                }
                if existing is None or source_mtime >= float(existing.get("source_mtime") or 0):
                    latest_by_box[box_number] = candidate
            set_first(record, "race_date", row.get("race_date"))
            set_first(record, "venue", row.get("venue"))
            set_first(record, "race_number", row.get("race_number"))
    return {
        "prediction_file_count": len(paths),
        "prediction_file_limit": max_prediction_files,
        "prediction_file_scan_truncated": truncated,
        "prediction_file_kind_counts": dict(sorted(path_kind_counts.items())),
        "shadow_prediction_rows": row_count,
        "shadow_prediction_race_count": len(race_ids),
    }


def scan_official_artifacts(
    *,
    artifact_roots: Sequence[Path],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    artifact_dirs, discovery = backlog.discover_official_result_artifact_dirs(artifact_roots)
    race_row_count = 0
    runner_row_count = 0
    race_ids: set[str] = set()
    for artifact_dir in artifact_dirs:
        paths = backlog.artifact_paths(artifact_dir)
        for row in load_jsonl(paths.get("race_rows")):
            record = race_record(records, row.get("race_id"))
            if record is None:
                continue
            race_row_count += 1
            race_ids.add(record["race_id"])
            record["official_artifact_race_rows"] += 1
            add_path(record, "official_artifact_paths", artifact_dir)
            set_first(record, "race_date", row.get("race_date"))
            set_first(record, "venue", row.get("venue"))
            set_first(record, "race_number", row.get("race_number"))
        for row in load_jsonl(paths.get("runner_rows")):
            record = race_record(records, row.get("race_id"))
            if record is None:
                continue
            runner_row_count += 1
            race_ids.add(record["race_id"])
            record["official_artifact_runner_rows"] += 1
            add_path(record, "official_artifact_paths", artifact_dir)
            add_box(record, "official_artifact_boxes", row.get("box_number"))
            set_first(record, "race_date", row.get("race_date"))
            set_first(record, "venue", row.get("venue"))
            set_first(record, "race_number", row.get("race_number"))
    return {
        "input_artifact_root_count": len(artifact_roots),
        "artifact_discovery": discovery,
        "official_result_artifact_dir_count": len(artifact_dirs),
        "official_result_artifact_race_rows": race_row_count,
        "official_result_artifact_runner_rows": runner_row_count,
        "official_result_artifact_race_count": len(race_ids),
    }


def sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def db_status(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"status": "DATA_MISSING", "reason": "db_path_missing", "db_path": str(db_path)}
    if db_path.stat().st_size == 0:
        return {"status": "DATA_MISSING", "reason": "db_zero_byte", "db_path": str(db_path)}
    return {"status": "AVAILABLE", "db_path": str(db_path), "bytes": db_path.stat().st_size}


def scan_db(
    *,
    db_path: Path,
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    status = db_status(db_path)
    if status.get("status") != "AVAILABLE":
        return {"db_status": status, "table_status": {}, "counts": {}}

    table_status: dict[str, Any] = {}
    counts: dict[str, int] = {}
    db_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        for table in (
            "autonomous_official_result_evidence_races",
            "autonomous_official_result_evidence_runners",
            "live_odds",
        ):
            table_status[table] = {"present": sqlite_table_exists(conn, table)}

        if table_status["autonomous_official_result_evidence_races"]["present"]:
            rows = conn.execute(
                """
                SELECT race_id, race_date, venue, race_number, COUNT(*) AS row_count
                FROM autonomous_official_result_evidence_races
                WHERE race_id IS NOT NULL AND TRIM(race_id) != ''
                GROUP BY race_id, race_date, venue, race_number
                """
            ).fetchall()
            counts["official_result_evidence_race_rows"] = sum(int(row["row_count"]) for row in rows)
            counts["official_result_evidence_race_count"] = len({str(row["race_id"]) for row in rows})
            for row in rows:
                record = race_record(records, row["race_id"])
                if record is None:
                    continue
                record["official_result_db_race_rows"] += int(row["row_count"])
                set_first(record, "race_date", row["race_date"])
                set_first(record, "venue", row["venue"])
                set_first(record, "race_number", row["race_number"])
        else:
            counts["official_result_evidence_race_rows"] = 0
            counts["official_result_evidence_race_count"] = 0

        if table_status["autonomous_official_result_evidence_runners"]["present"]:
            rows = conn.execute(
                """
                SELECT
                    race_id,
                    race_date,
                    venue,
                    race_number,
                    box_number,
                    dog_name,
                    finish_position,
                    is_winner
                FROM autonomous_official_result_evidence_runners
                WHERE race_id IS NOT NULL AND TRIM(race_id) != ''
                """
            ).fetchall()
            counts["official_result_evidence_runner_rows"] = len(rows)
            counts["official_result_evidence_runner_race_count"] = len({str(row["race_id"]) for row in rows})
            for row in rows:
                record = race_record(records, row["race_id"])
                if record is None:
                    continue
                record["official_result_db_runner_rows"] += 1
                box_number = parse_int(row["box_number"])
                add_box(record, "official_result_db_boxes", box_number)
                if box_number is not None:
                    result = {
                        "box_number": box_number,
                        "dog_name": row["dog_name"],
                        "finish_position": parse_int(row["finish_position"]),
                        "is_winner": parse_int(row["is_winner"]),
                    }
                    existing = record["official_result_by_box"].get(box_number)
                    if existing is None:
                        record["official_result_by_box"][box_number] = result
                    elif (
                        existing.get("finish_position") != result.get("finish_position")
                        or existing.get("is_winner") != result.get("is_winner")
                    ):
                        record["official_result_conflict_count"] += 1
                set_first(record, "race_date", row["race_date"])
                set_first(record, "venue", row["venue"])
                set_first(record, "race_number", row["race_number"])
        else:
            counts["official_result_evidence_runner_rows"] = 0
            counts["official_result_evidence_runner_race_count"] = 0

        if table_status["live_odds"]["present"]:
            rows = conn.execute(
                """
                SELECT
                    race_id,
                    venue,
                    race_number,
                    race_date,
                    race_time,
                    dog_name,
                    dog_clean_name,
                    box_number,
                    odds_decimal,
                    odds_fractional,
                    market_type,
                    source,
                    timestamp,
                    is_current,
                    topN,
                    source_url,
                    capture_timestamp,
                    capture_mode,
                    odds_level,
                    sportsbet_box_source,
                    sportsbet_list_position,
                    sportsbet_raw_runner_text
                FROM live_odds
                WHERE race_id IS NOT NULL AND TRIM(race_id) != ''
                """
            ).fetchall()
            live_races: set[str] = set()
            strict_races: set[str] = set()
            strict_rows = 0
            for sqlite_row in rows:
                row = dict(sqlite_row)
                record = race_record(records, row.get("race_id"))
                if record is None:
                    continue
                live_races.add(record["race_id"])
                record["live_odds_rows"] += 1
                add_box(record, "live_odds_boxes", row.get("box_number"))
                set_first(record, "race_date", row.get("race_date"))
                set_first(record, "venue", row.get("venue"))
                set_first(record, "race_number", row.get("race_number"))
                if not validate_odds_row(row):
                    strict_rows += 1
                    strict_races.add(record["race_id"])
                    record["strict_live_odds_rows"] += 1
                    box_number = parse_int(row.get("box_number"))
                    add_box(record, "strict_live_odds_boxes", box_number)
                    if box_number is not None:
                        candidate = {
                            "box_number": box_number,
                            "dog_name": row.get("dog_name") or row.get("dog_clean_name"),
                            "odds_decimal": row.get("odds_decimal"),
                            "capture_timestamp": str(row.get("capture_timestamp") or ""),
                            "capture_mode": row.get("capture_mode"),
                        }
                        existing = record["strict_odds_by_box"].get(box_number)
                        if existing is None or str(candidate["capture_timestamp"]) >= str(
                            existing.get("capture_timestamp") or ""
                        ):
                            record["strict_odds_by_box"][box_number] = candidate
            counts["live_odds_rows"] = len(rows)
            counts["live_odds_race_count"] = len(live_races)
            counts["strict_live_odds_rows"] = strict_rows
            counts["strict_live_odds_race_count"] = len(strict_races)
        else:
            counts["live_odds_rows"] = 0
            counts["live_odds_race_count"] = 0
            counts["strict_live_odds_rows"] = 0
            counts["strict_live_odds_race_count"] = 0

    return {"db_status": status, "table_status": table_status, "counts": counts}


def limited(items: Iterable[Any], limit: int = 20) -> list[Any]:
    return list(sorted(items))[:limit]


def race_action(row: Mapping[str, Any]) -> str:
    if not row.get("has_shadow_predictions"):
        return "not_shadow_scored"
    if row.get("has_official_result_artifact") and not row.get("has_official_result_evidence_db"):
        return "append_official_result_evidence_backlog"
    if not row.get("has_official_result_evidence_db"):
        return "capture_official_result"
    if not row.get("has_complete_official_result_evidence_db_for_shadow"):
        return "repair_official_result_runner_set_or_identity_join"
    if not row.get("has_strict_prejump_odds"):
        return "collect_future_strict_prejump_odds"
    if not row.get("has_complete_strict_prejump_odds_for_shadow"):
        return "repair_strict_prejump_odds_runner_set"
    return "ready_for_unified_evidence_evaluation"


def build_race_rows(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for race_id, record in sorted(records.items()):
        shadow_boxes = set(record.get("shadow_boxes") or set())
        official_db_boxes = set(record.get("official_result_db_boxes") or set())
        strict_odds_boxes = set(record.get("strict_live_odds_boxes") or set())
        has_shadow = int(record.get("shadow_prediction_rows") or 0) > 0
        row = {
            "race_id": race_id,
            "race_date": record.get("race_date"),
            "venue": record.get("venue"),
            "race_number": record.get("race_number"),
            "shadow_prediction_rows": int(record.get("shadow_prediction_rows") or 0),
            "shadow_prediction_path_count": len(record.get("shadow_prediction_paths") or []),
            "shadow_boxes": json.dumps(sorted(shadow_boxes)),
            "shadow_box_count": len(shadow_boxes),
            "official_artifact_race_rows": int(record.get("official_artifact_race_rows") or 0),
            "official_artifact_runner_rows": int(record.get("official_artifact_runner_rows") or 0),
            "official_artifact_path_count": len(record.get("official_artifact_paths") or []),
            "official_artifact_boxes": json.dumps(sorted(record.get("official_artifact_boxes") or [])),
            "official_artifact_box_count": len(record.get("official_artifact_boxes") or []),
            "official_result_db_race_rows": int(record.get("official_result_db_race_rows") or 0),
            "official_result_db_runner_rows": int(record.get("official_result_db_runner_rows") or 0),
            "official_result_db_boxes": json.dumps(sorted(official_db_boxes)),
            "official_result_db_box_count": len(official_db_boxes),
            "official_result_conflict_count": int(record.get("official_result_conflict_count") or 0),
            "live_odds_rows": int(record.get("live_odds_rows") or 0),
            "live_odds_box_count": len(record.get("live_odds_boxes") or []),
            "strict_live_odds_rows": int(record.get("strict_live_odds_rows") or 0),
            "strict_live_odds_boxes": json.dumps(sorted(strict_odds_boxes)),
            "strict_live_odds_box_count": len(strict_odds_boxes),
            "has_shadow_predictions": has_shadow,
            "has_official_result_artifact": int(record.get("official_artifact_runner_rows") or 0) > 0,
            "has_official_result_evidence_db": int(record.get("official_result_db_runner_rows") or 0) > 0,
            "has_live_odds": int(record.get("live_odds_rows") or 0) > 0,
            "has_strict_prejump_odds": int(record.get("strict_live_odds_rows") or 0) > 0,
            "has_complete_official_result_evidence_db_for_shadow": (
                has_shadow and bool(shadow_boxes) and shadow_boxes.issubset(official_db_boxes)
            ),
            "has_complete_strict_prejump_odds_for_shadow": (
                has_shadow and bool(shadow_boxes) and shadow_boxes.issubset(strict_odds_boxes)
            ),
            "sample_shadow_prediction_paths": json.dumps(limited(record.get("shadow_prediction_paths") or [])),
            "sample_official_artifact_paths": json.dumps(limited(record.get("official_artifact_paths") or [])),
        }
        row["has_complete_shadow_official_and_strict_odds"] = bool(
            row["has_shadow_predictions"]
            and row["has_complete_official_result_evidence_db_for_shadow"]
            and row["has_complete_strict_prejump_odds_for_shadow"]
        )
        row["recommended_next_action"] = race_action(row)
        rows.append(row)
    return rows


def latest_backlog_append_report(artifact_roots: Sequence[Path]) -> dict[str, Any]:
    candidates: list[Path] = []
    for root in artifact_roots:
        logical = root if root.is_absolute() else ROOT / root
        if not logical.exists():
            continue
        candidates.extend(
            path
            for path in logical.glob("official_result_evidence_append_backlog_*/official_result_evidence_append_backlog_report.json")
            if path.is_file()
        )
    if not candidates:
        return {"status": "DATA_MISSING", "reason": "no_backlog_append_report_found"}
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    report = load_json(latest)
    return {
        "status": "FOUND",
        "path": relpath(latest) or str(latest),
        "final_status": report.get("final_status"),
        "artifact_count": report.get("artifact_count"),
        "processed_count": report.get("processed_count"),
        "status_counts": report.get("status_counts"),
        "inserted_race_rows": report.get("inserted_race_rows"),
        "inserted_runner_rows": report.get("inserted_runner_rows"),
        "db_write_performed": report.get("db_write_performed"),
        "shared_lock_status": report.get("shared_lock_status"),
        "shared_lock_release": report.get("shared_lock_release"),
    }


def build_summary_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(str(row.get("recommended_next_action") or "UNKNOWN") for row in rows)
    shadow_rows = [row for row in rows if row.get("has_shadow_predictions")]
    return {
        "race_union_count": len(rows),
        "shadow_prediction_race_count": sum(1 for row in rows if row.get("has_shadow_predictions")),
        "official_result_artifact_race_count": sum(1 for row in rows if row.get("has_official_result_artifact")),
        "official_result_evidence_db_race_count": sum(1 for row in rows if row.get("has_official_result_evidence_db")),
        "live_odds_race_count": sum(1 for row in rows if row.get("has_live_odds")),
        "strict_prejump_odds_race_count": sum(1 for row in rows if row.get("has_strict_prejump_odds")),
        "shadow_races_with_official_result_evidence_db": sum(
            1 for row in shadow_rows if row.get("has_official_result_evidence_db")
        ),
        "shadow_races_with_strict_prejump_odds": sum(
            1 for row in shadow_rows if row.get("has_strict_prejump_odds")
        ),
        "shadow_races_complete_official_and_strict_odds": sum(
            1 for row in shadow_rows if row.get("has_complete_shadow_official_and_strict_odds")
        ),
        "action_counts": dict(sorted(action_counts.items())),
    }


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def positive_probability(value: Any) -> float | None:
    parsed = parse_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def prediction_probability(row: Mapping[str, Any]) -> float | None:
    for key in (
        "shadow_rf_calibrated_probability",
        "stage2_probability",
        "predicted_probability",
        "win_probability",
        "probability",
    ):
        parsed = positive_probability(row.get(key))
        if parsed is not None:
            return parsed
    return None


def ranked_shadow_boxes(predictions_by_box: Mapping[int, Mapping[str, Any]]) -> list[int]:
    rankable: list[tuple[float, int]] = []
    for box, payload in predictions_by_box.items():
        row = payload.get("row") if isinstance(payload.get("row"), Mapping) else {}
        rank = parse_float(row.get("predicted_rank"))
        probability = prediction_probability(row)
        if rank is not None:
            key = rank
        elif probability is not None:
            key = -probability
        else:
            key = float(box)
        rankable.append((key, int(box)))
    return [box for _, box in sorted(rankable)]


def normalized_probability_by_box(raw: Mapping[int, float | None]) -> dict[int, float]:
    positives = {
        int(box): float(value)
        for box, value in raw.items()
        if value is not None and float(value) > 0
    }
    total = sum(positives.values())
    if total <= 0:
        return {}
    return {box: value / total for box, value in positives.items()}


def safe_logloss(probability: float | None) -> float | None:
    if probability is None:
        return None
    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -math.log(clipped)


def mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def build_scorecard(records: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped_reasons = Counter()
    for race_id, record in sorted(records.items()):
        shadow_boxes = set(record.get("shadow_boxes") or set())
        official_boxes = set(record.get("official_result_db_boxes") or set())
        strict_boxes = set(record.get("strict_live_odds_boxes") or set())
        if not shadow_boxes:
            skipped_reasons["shadow_predictions_missing"] += 1
            continue
        if not shadow_boxes.issubset(official_boxes):
            skipped_reasons["official_result_incomplete_for_shadow_boxes"] += 1
            continue
        if not shadow_boxes.issubset(strict_boxes):
            skipped_reasons["strict_odds_incomplete_for_shadow_boxes"] += 1
            continue
        if int(record.get("official_result_conflict_count") or 0) > 0:
            skipped_reasons["official_result_conflicts"] += 1
            continue

        results_by_box = record.get("official_result_by_box") or {}
        winner_boxes = [
            int(box)
            for box, result in results_by_box.items()
            if parse_int(result.get("finish_position")) == 1
            or parse_int(result.get("is_winner")) == 1
        ]
        if len(winner_boxes) != 1:
            skipped_reasons["winner_box_not_unique"] += 1
            continue
        winner_box = winner_boxes[0]

        predictions_by_box = record.get("latest_shadow_prediction_by_box") or {}
        if not set(predictions_by_box).issuperset(shadow_boxes):
            skipped_reasons["latest_shadow_prediction_rows_incomplete"] += 1
            continue
        model_order = ranked_shadow_boxes(predictions_by_box)
        if winner_box not in model_order:
            skipped_reasons["winner_box_missing_from_shadow_predictions"] += 1
            continue
        model_winner_rank = model_order.index(winner_box) + 1
        raw_model_probs = {
            int(box): prediction_probability(
                payload.get("row") if isinstance(payload.get("row"), Mapping) else {}
            )
            for box, payload in predictions_by_box.items()
            if int(box) in shadow_boxes
        }
        model_probs = normalized_probability_by_box(raw_model_probs)

        odds_by_box = record.get("strict_odds_by_box") or {}
        market_candidates: list[tuple[float, int]] = []
        raw_market_probs: dict[int, float | None] = {}
        for box in shadow_boxes:
            odds = odds_by_box.get(box) or {}
            decimal = parse_float(odds.get("odds_decimal"))
            if decimal is None or decimal <= 1:
                continue
            market_candidates.append((decimal, int(box)))
            raw_market_probs[int(box)] = 1.0 / decimal
        if len(market_candidates) < len(shadow_boxes):
            skipped_reasons["market_odds_missing_for_shadow_boxes"] += 1
            continue
        market_order = [box for _, box in sorted(market_candidates)]
        market_winner_rank = market_order.index(winner_box) + 1
        market_probs = normalized_probability_by_box(raw_market_probs)

        winner_payload = predictions_by_box.get(winner_box) or {}
        winner_prediction = (
            winner_payload.get("row") if isinstance(winner_payload.get("row"), Mapping) else {}
        )
        rows.append(
            {
                "race_id": race_id,
                "race_date": record.get("race_date"),
                "venue": record.get("venue"),
                "race_number": record.get("race_number"),
                "runner_count": len(shadow_boxes),
                "winner_box": winner_box,
                "winner_dog_name": (results_by_box.get(winner_box) or {}).get("dog_name"),
                "model_winner_rank": model_winner_rank,
                "model_top1_correct": model_winner_rank == 1,
                "model_top3_correct": model_winner_rank <= 3,
                "model_winner_probability": model_probs.get(winner_box),
                "model_logloss": safe_logloss(model_probs.get(winner_box)),
                "market_winner_rank": market_winner_rank,
                "market_top1_correct": market_winner_rank == 1,
                "market_top3_correct": market_winner_rank <= 3,
                "market_winner_probability": market_probs.get(winner_box),
                "market_logloss": safe_logloss(market_probs.get(winner_box)),
                "model_top_box": model_order[0] if model_order else None,
                "market_top_box": market_order[0] if market_order else None,
                "winner_prediction_source_path": winner_payload.get("source_path"),
                "winner_prediction_raw_probability": prediction_probability(winner_prediction),
            }
        )

    model_logloss_values = [
        float(row["model_logloss"]) for row in rows if row.get("model_logloss") is not None
    ]
    market_logloss_values = [
        float(row["market_logloss"]) for row in rows if row.get("market_logloss") is not None
    ]
    metrics = {
        "schema_version": "race_evidence_scorecard_metrics_v1",
        "evaluation_race_count": len(rows),
        "model_top1_accuracy": (
            sum(1 for row in rows if row.get("model_top1_correct")) / len(rows)
            if rows
            else None
        ),
        "model_top3_accuracy": (
            sum(1 for row in rows if row.get("model_top3_correct")) / len(rows)
            if rows
            else None
        ),
        "model_mean_winner_rank": mean([float(row["model_winner_rank"]) for row in rows]),
        "model_logloss": mean(model_logloss_values),
        "market_top1_accuracy": (
            sum(1 for row in rows if row.get("market_top1_correct")) / len(rows)
            if rows
            else None
        ),
        "market_top3_accuracy": (
            sum(1 for row in rows if row.get("market_top3_correct")) / len(rows)
            if rows
            else None
        ),
        "market_mean_winner_rank": mean([float(row["market_winner_rank"]) for row in rows]),
        "market_logloss": mean(market_logloss_values),
        "skipped_race_reason_counts": dict(sorted(skipped_reasons.items())),
        "metric_notes": [
            "report_only_latest_shadow_prediction_per_race_box",
            "official_results_from_append_only_evidence_db",
            "market_baseline_from_latest_strict_sportsbet_odds_per_box",
        ],
    }
    return rows, metrics


def top_gaps(rows: Sequence[Mapping[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    gap_rows = [
        row
        for row in rows
        if row.get("has_shadow_predictions")
        and row.get("recommended_next_action") != "ready_for_unified_evidence_evaluation"
    ]
    return [
        {
            "race_id": row.get("race_id"),
            "race_date": row.get("race_date"),
            "venue": row.get("venue"),
            "race_number": row.get("race_number"),
            "recommended_next_action": row.get("recommended_next_action"),
            "shadow_box_count": row.get("shadow_box_count"),
            "official_result_db_box_count": row.get("official_result_db_box_count"),
            "strict_live_odds_box_count": row.get("strict_live_odds_box_count"),
        }
        for row in sorted(
            gap_rows,
            key=lambda item: (
                int(item.get("shadow_prediction_rows") or 0),
                str(item.get("race_id") or ""),
            ),
            reverse=True,
        )[:limit]
    ]


def recommended_decision(summary_counts: Mapping[str, Any]) -> str:
    action_counts = summary_counts.get("action_counts") or {}
    if int(summary_counts.get("shadow_races_complete_official_and_strict_odds") or 0) > 0:
        return "RUN_POST_BACKLOG_UNIFIED_EVALUATION"
    if int(action_counts.get("append_official_result_evidence_backlog") or 0) > 0:
        return "RUN_BACKLOG_APPEND"
    if int(action_counts.get("collect_future_strict_prejump_odds") or 0) > 0:
        return "STRICT_PREJUMP_ODDS_COLLECTION_NEXT"
    if int(action_counts.get("capture_official_result") or 0) > 0:
        return "OFFICIAL_RESULT_CAPTURE_NEXT"
    return "KEEP_COLLECTING_OR_DATA_MISSING"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "race_id",
        "race_date",
        "venue",
        "race_number",
        "shadow_prediction_rows",
        "shadow_prediction_path_count",
        "shadow_box_count",
        "official_artifact_race_rows",
        "official_artifact_runner_rows",
        "official_artifact_path_count",
        "official_artifact_box_count",
        "official_result_db_race_rows",
        "official_result_db_runner_rows",
        "official_result_db_box_count",
        "official_result_conflict_count",
        "live_odds_rows",
        "live_odds_box_count",
        "strict_live_odds_rows",
        "strict_live_odds_box_count",
        "has_shadow_predictions",
        "has_official_result_artifact",
        "has_official_result_evidence_db",
        "has_live_odds",
        "has_strict_prejump_odds",
        "has_complete_official_result_evidence_db_for_shadow",
        "has_complete_strict_prejump_odds_for_shadow",
        "has_complete_shadow_official_and_strict_odds",
        "recommended_next_action",
        "shadow_boxes",
        "official_result_db_boxes",
        "strict_live_odds_boxes",
        "sample_shadow_prediction_paths",
        "sample_official_artifact_paths",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_scorecard_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "race_id",
        "race_date",
        "venue",
        "race_number",
        "runner_count",
        "winner_box",
        "winner_dog_name",
        "model_winner_rank",
        "model_top1_correct",
        "model_top3_correct",
        "model_winner_probability",
        "model_logloss",
        "market_winner_rank",
        "market_top1_correct",
        "market_top3_correct",
        "market_winner_probability",
        "market_logloss",
        "model_top_box",
        "market_top_box",
        "winner_prediction_source_path",
        "winner_prediction_raw_probability",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_markdown(report: Mapping[str, Any]) -> str:
    counts = report.get("summary_counts") if isinstance(report.get("summary_counts"), Mapping) else {}
    metrics = report.get("scorecard_metrics") if isinstance(report.get("scorecard_metrics"), Mapping) else {}
    return "\n".join(
        [
            "# Race Evidence Inventory",
            "",
            f"Final status: `{report.get('final_status')}`",
            f"Recommended decision: `{report.get('recommended_decision')}`",
            "",
            f"- Race union count: `{counts.get('race_union_count')}`",
            f"- Shadow prediction races: `{counts.get('shadow_prediction_race_count')}`",
            f"- Official-result artifact races: `{counts.get('official_result_artifact_race_count')}`",
            f"- Official-result evidence DB races: `{counts.get('official_result_evidence_db_race_count')}`",
            f"- Live odds races: `{counts.get('live_odds_race_count')}`",
            f"- Strict pre-jump odds races: `{counts.get('strict_prejump_odds_race_count')}`",
            f"- Shadow races with official-result evidence DB: `{counts.get('shadow_races_with_official_result_evidence_db')}`",
            f"- Shadow races with strict pre-jump odds: `{counts.get('shadow_races_with_strict_prejump_odds')}`",
            f"- Shadow races complete for official result and strict odds: `{counts.get('shadow_races_complete_official_and_strict_odds')}`",
            f"- Scorecard evaluation races: `{metrics.get('evaluation_race_count')}`",
            f"- Model Top1 / Top3: `{metrics.get('model_top1_accuracy')}` / `{metrics.get('model_top3_accuracy')}`",
            f"- Market Top1 / Top3: `{metrics.get('market_top1_accuracy')}` / `{metrics.get('market_top3_accuracy')}`",
            f"- Model mean winner rank: `{metrics.get('model_mean_winner_rank')}`",
            f"- Market mean winner rank: `{metrics.get('market_mean_winner_rank')}`",
            "",
            "## Action Counts",
            "",
            "```json",
            json.dumps(counts.get("action_counts") or {}, indent=2, sort_keys=True),
            "```",
            "",
            "## No-Write Guarantees",
            "",
            "```json",
            json.dumps(report.get("no_write_guarantees") or {}, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def build_packet(
    *,
    artifact_roots: Sequence[Path],
    db_path: Path,
    output_dir: Path,
    max_prediction_files: int | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    records: dict[str, dict[str, Any]] = {}
    official_artifact_summary = scan_official_artifacts(
        artifact_roots=artifact_roots,
        records=records,
    )
    shadow_summary = scan_shadow_predictions(
        artifact_roots=artifact_roots,
        records=records,
        max_prediction_files=max_prediction_files,
    )
    db_summary = scan_db(db_path=db_path, records=records)
    race_rows = build_race_rows(records)
    summary_counts = build_summary_counts(race_rows)
    scorecard_rows, scorecard_metrics = build_scorecard(records)
    decision = recommended_decision(summary_counts)
    db_available = (db_summary.get("db_status") or {}).get("status") == "AVAILABLE"
    final_status = (
        "DATA_MISSING"
        if not db_available
        else (
            "RACE_EVIDENCE_INVENTORY_READY_FOR_EVALUATION"
            if decision == "RUN_POST_BACKLOG_UNIFIED_EVALUATION"
            else "RACE_EVIDENCE_INVENTORY_GAPS_FOUND"
        )
    )

    report = {
        "schema_version": "race_evidence_inventory_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "recommended_decision": decision,
        "output_dir": relpath(output_dir),
        "artifact_roots": [relpath(root if root.is_absolute() else ROOT / root) for root in artifact_roots],
        "db_path": str(db_path),
        "official_artifact_summary": official_artifact_summary,
        "shadow_prediction_summary": shadow_summary,
        "db_summary": db_summary,
        "latest_backlog_append_report": latest_backlog_append_report(artifact_roots),
        "summary_counts": summary_counts,
        "scorecard_metrics": scorecard_metrics,
        "top_gap_races": top_gaps(race_rows),
        "inventory_csv": relpath(output_dir / CSV_FILE),
        "inventory_jsonl": relpath(output_dir / JSONL_FILE),
        "scorecard_csv": relpath(output_dir / SCORECARD_CSV_FILE),
        "scorecard_jsonl": relpath(output_dir / SCORECARD_JSONL_FILE),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_csv(output_dir / CSV_FILE, race_rows)
    write_jsonl(output_dir / JSONL_FILE, race_rows)
    write_scorecard_csv(output_dir / SCORECARD_CSV_FILE, scorecard_rows)
    write_jsonl(output_dir / SCORECARD_JSONL_FILE, scorecard_rows)
    write_json(output_dir / REPORT_FILE, report)
    write_text(output_dir / SUMMARY_FILE, summary_markdown(report))
    write_text(output_dir / "final_status.txt", final_status + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-prediction-files",
        type=int,
        default=None,
        help="Optional cap for prediction JSONL files scanned; default scans all.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    artifact_roots = args.artifact_root or [DEFAULT_EVIDENCE_ROOT]
    output_dir = (
        args.output_dir
        or DEFAULT_EVIDENCE_ROOT / f"race_evidence_inventory_{now_id(generated_at)}_report_only"
    )
    report = build_packet(
        artifact_roots=artifact_roots,
        db_path=args.db,
        output_dir=output_dir,
        max_prediction_files=args.max_prediction_files,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
