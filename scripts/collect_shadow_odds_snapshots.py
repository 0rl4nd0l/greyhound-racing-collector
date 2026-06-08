#!/usr/bin/env python3
"""Collect report-only odds diagnostics for forward-shadow predictions.

This lane is intentionally artifact-only. It reads completed shadow prediction
rows, looks for exact dog/box live-odds rows in SQLite read-only mode, classifies
odds eligibility with the existing EV safety interface, and writes diagnostics.
It does not write DB rows, labels, snapshots, registry entries, production
predictions, EV recommendations, betting output, or model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from accuracy_program.odds_coverage import normalize_dog_name, normalize_venue  # noqa: E402
from accuracy_program.snapshots import classify_odds_snapshot_for_ev  # noqa: E402
from scripts.refresh_prejump_upcoming import venue_exclusion_aliases  # noqa: E402


DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_"
EXPECTED_OFFICIAL_RACES = 214
EXPECTED_OFFICIAL_DOG_ROWS = 1493
DEFAULT_STALE_ODDS_AFTER_MINUTES = 30.0
PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
FINAL_COLLECTED = "SHADOW_ODDS_SNAPSHOT_COLLECTED"
FINAL_NO_MATCHES = "SHADOW_ODDS_SNAPSHOT_NO_MATCHES"
FINAL_NO_PREDICTIONS = "DATA_MISSING_NO_SHADOW_PREDICTIONS"
FINAL_ODDS_UNAVAILABLE = "ODDS_SOURCE_UNAVAILABLE"
FINAL_DB_BLOCKED = "BLOCKED_DB_STATE"
RACE_ID_RE = re.compile(r"^Race\s+(\d+)\s+-\s+(.+)\s+-\s+(\d{4}-\d{2}-\d{2})$")
ODDS_CSV_COLUMNS = (
    "race_id",
    "dog_name",
    "box",
    "predicted_rank",
    "shadow_rf_calibrated_probability",
    "odds_decimal",
    "odds_match_status",
    "odds_exclusion_reason",
    "odds_provenance_status",
    "odds_match_method",
    "odds_source_url",
    "odds_timestamp",
    "odds_stale_at_prediction",
    "stale_odds_after_minutes",
    "odds_captured_before_prediction",
    "odds_captured_before_feature_freeze",
    "odds_age_minutes_at_feature_freeze",
    "odds_captured_before_jump",
    "odds_age_minutes_at_jump",
)
NO_WRITE_GUARANTEES = {
    "db_write": False,
    "label_write": False,
    "registry_mutation": False,
    "production_model_mutation": False,
    "production_prediction_write": False,
    "prediction_snapshot_write": False,
    "ev_output": False,
    "betting_action": False,
    "tgr_enabled": False,
    "training": False,
}
POST_PREDICTION_ODDS_REASONS = {
    "timestamp_after_prediction",
    "odds_captured_after_prediction",
}
POST_FEATURE_FREEZE_ODDS_REASONS = {
    "timestamp_after_feature_freeze",
    "odds_captured_after_feature_freeze",
}
POST_JUMP_ODDS_REASONS = {
    "timestamp_after_jump",
    "odds_captured_after_jump",
}
ODDS_RESEARCH_GATE_POLICY = {
    "schema_version": "shadow_odds_research_gate_policy_v1",
    "candidate_scope": {
        "market_type": "win",
        "odds_level": "dog",
        "minimum_decimal_odds": ">1.0",
        "current_only_default": True,
    },
    "source_requirements": {
        "trusted_source_required": True,
        "source_url_required": True,
        "post_race_or_sp_markets_rejected": True,
        "post_race_source_tables_rejected": True,
    },
    "timing_requirements": {
        "odds_timestamp_required": True,
        "captured_before_prediction_required": True,
        "captured_before_feature_freeze_required_when_available": True,
        "captured_before_jump_required_when_available": True,
        "stale_beyond_ttl_rejected": True,
        "stale_odds_after_minutes_default": DEFAULT_STALE_ODDS_AFTER_MINUTES,
    },
    "identity_requirements": {
        "race_identity_required": "exact_or_canonical_equivalent",
        "dog_name_match_required": True,
        "box_match_required": True,
        "ambiguous_box_source_rejected": True,
        "duplicate_odds_rows_rejected": True,
        "low_confidence_match_rejected": True,
        "fuzzy_only_match_rejected": True,
    },
    "coverage_requirements": {
        "complete_valid_prejump_odds_required_for_odds_research_ready": True,
        "all_predicted_runners_must_have_valid_prejump_odds": True,
    },
    "ev_policy": {
        "ev_output_allowed": False,
        "betting_action_allowed": False,
        "odds_used_for_shadow_scoring": False,
        "status": "REPORT_ONLY_NO_EV_OUTPUT",
    },
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ODDS_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            snapshot = row.get("odds_snapshot") if isinstance(row.get("odds_snapshot"), Mapping) else {}
            provenance = (
                snapshot.get("odds_provenance")
                if isinstance(snapshot.get("odds_provenance"), Mapping)
                else {}
            )
            writer.writerow(
                {
                    "race_id": row.get("race_id"),
                    "dog_name": row.get("dog_name"),
                    "box": row.get("box"),
                    "predicted_rank": row.get("predicted_rank"),
                    "shadow_rf_calibrated_probability": row.get(
                        "shadow_rf_calibrated_probability"
                    ),
                    "odds_decimal": snapshot.get("market_odds_win"),
                    "odds_match_status": row.get("odds_match_status"),
                    "odds_exclusion_reason": row.get("odds_exclusion_reason"),
                    "odds_provenance_status": row.get("odds_provenance_status"),
                    "odds_match_method": row.get("odds_match_method"),
                    "odds_source_url": provenance.get("source_url"),
                    "odds_timestamp": snapshot.get("odds_timestamp"),
                    "odds_stale_at_prediction": snapshot.get(
                        "odds_stale_at_prediction"
                    ),
                    "stale_odds_after_minutes": snapshot.get(
                        "stale_odds_after_minutes"
                    ),
                    "odds_captured_before_prediction": snapshot.get(
                        "odds_captured_before_prediction"
                    ),
                    "odds_captured_before_feature_freeze": snapshot.get(
                        "odds_captured_before_feature_freeze"
                    ),
                    "odds_age_minutes_at_feature_freeze": snapshot.get(
                        "odds_age_minutes_at_feature_freeze"
                    ),
                    "odds_captured_before_jump": snapshot.get(
                        "odds_captured_before_jump"
                    ),
                    "odds_age_minutes_at_jump": snapshot.get(
                        "odds_age_minutes_at_jump"
                    ),
                }
            )


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(paths: Sequence[Path] = PROTECTED_PATHS) -> dict[str, str | None]:
    return {relpath(path) or str(path): sha256_file(path) for path in paths}


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_shadow_odds_snapshot_artifact:{relative}")
    return logical.absolute()


def parse_current_time(value: str | None) -> datetime:
    if not value:
        return datetime.now().astimezone()
    text = value.strip()
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    parsed = datetime.fromisoformat(text)
    return parsed.astimezone() if parsed.tzinfo is None else parsed


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def open_readonly_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def verify_db_state(db_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "shadow_odds_snapshot_db_state_v1",
        "db_path": relpath(db_path),
        "expected_official_races": EXPECTED_OFFICIAL_RACES,
        "expected_official_dog_rows": EXPECTED_OFFICIAL_DOG_ROWS,
        "status": "FAIL",
        "fail_reasons": [],
    }
    if not db_path.exists():
        report["fail_reasons"].append("db_missing")
        return report
    try:
        report["sha256"] = sha256_file(db_path)
        with open_readonly_db(db_path) as conn:
            report["quick_check"] = conn.execute("PRAGMA quick_check").fetchone()[0]
            report["official_races"] = conn.execute(
                "SELECT count(DISTINCT race_id) FROM race_metadata "
                "WHERE winner_source='thedogs_official'"
            ).fetchone()[0]
            report["official_dog_rows"] = conn.execute(
                "SELECT count(*) FROM dog_race_data "
                "WHERE data_source='thedogs_official'"
            ).fetchone()[0]
    except Exception as exc:
        report["fail_reasons"].append(f"db_read_failed:{type(exc).__name__}")
        return report
    if report.get("quick_check") != "ok":
        report["fail_reasons"].append("quick_check_not_ok")
    if report.get("official_races") != EXPECTED_OFFICIAL_RACES:
        report["fail_reasons"].append("official_race_count_mismatch")
    if report.get("official_dog_rows") != EXPECTED_OFFICIAL_DOG_ROWS:
        report["fail_reasons"].append("official_dog_row_count_mismatch")
    if not report["fail_reasons"]:
        report["status"] = "PASS"
    return report


def shadow_predictions_path(shadow_run_dir: Path) -> Path:
    return shadow_run_dir / "shadow_predictions.jsonl"


def parse_race_id(race_id: Any) -> dict[str, Any]:
    match = RACE_ID_RE.match(str(race_id or "").strip())
    if not match:
        return {"race_number": None, "venue": None, "race_date": None}
    race_number, venue, race_date = match.groups()
    return {
        "race_number": int(race_number),
        "venue": venue.strip(),
        "race_date": race_date,
    }


def stable_race_id(race_number: Any, venue: Any, race_date: Any) -> str | None:
    if race_number in (None, "") or venue in (None, "") or race_date in (None, ""):
        return None
    try:
        parsed_race_number = int(str(race_number).strip())
    except Exception:
        return None
    return f"Race {parsed_race_number} - {str(venue).strip().upper()} - {str(race_date).strip()[:10]}"


def load_race_contexts(shadow_run_dir: Path) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    report = read_json(shadow_run_dir / "prejump_metadata_report.json")
    for row in report.get("files") or []:
        if not isinstance(row, Mapping):
            continue
        race_id = stable_race_id(
            row.get("race_number"),
            row.get("venue"),
            row.get("race_date"),
        )
        if race_id:
            contexts[race_id] = {
                "race_id": race_id,
                "race_date": row.get("race_date"),
                "venue": row.get("venue"),
                "race_number": row.get("race_number"),
                "jump_datetime": row.get("jump_datetime"),
                "jump_time_iso": row.get("jump_time_iso"),
                "jump_time": row.get("jump_time"),
                "race_time": row.get("race_time"),
                "source_url": row.get("source_url"),
                "runner_count": row.get("runner_count"),
            }
    return contexts


def parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def parse_jump_datetime(context: Mapping[str, Any] | None) -> datetime | None:
    if not isinstance(context, Mapping):
        return None
    for key in (
        "jump_datetime",
        "jump_time_iso",
        "race_jump_datetime",
        "scheduled_jump_datetime",
        "start_datetime",
    ):
        parsed = parse_timestamp(context.get(key))
        if parsed is not None:
            return parsed

    race_date = str(context.get("race_date") or "").strip()
    jump_time = str(context.get("jump_time") or context.get("race_time") or "").strip()
    if not race_date or not jump_time:
        return None
    candidates = [
        f"{race_date[:10]} {jump_time}",
        f"{race_date[:10]}T{jump_time}",
    ]
    for text in candidates:
        parsed = parse_timestamp(text)
        if parsed is not None:
            return parsed
        for fmt in (
            "%Y-%m-%d %I:%M %p",
            "%Y-%m-%d %I:%M:%S %p",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def manifest_timestamp(
    manifest: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> datetime | None:
    for path in paths:
        current: Any = manifest
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        parsed = parse_timestamp(current)
        if parsed is not None:
            return parsed
    return None


def seconds_between(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    compare_later = later
    compare_earlier = earlier
    if compare_later.tzinfo is not None and compare_earlier.tzinfo is None:
        compare_earlier = compare_earlier.replace(tzinfo=compare_later.tzinfo)
    elif compare_later.tzinfo is None and compare_earlier.tzinfo is not None:
        compare_later = compare_later.replace(tzinfo=compare_earlier.tzinfo)
    return (compare_later - compare_earlier).total_seconds()


def candidate_rows_for_prediction(
    conn: sqlite3.Connection,
    prediction: Mapping[str, Any],
    *,
    current_only: bool,
) -> list[dict[str, Any]]:
    meta = parse_race_id(prediction.get("race_id"))
    box = safe_int(prediction.get("box"))
    dog_key = normalize_dog_name(prediction.get("dog_name"))
    if box is None or not dog_key:
        return []
    aliases = {
        normalize_venue(alias)
        for alias in venue_exclusion_aliases(meta.get("venue"))
        if alias not in (None, "")
    }
    if meta.get("venue"):
        aliases.add(normalize_venue(meta["venue"]))
    exact_race_id = str(prediction.get("race_id") or "").strip()
    race_date = meta.get("race_date")
    race_number = meta.get("race_number")
    current_clause = "AND (is_current = 1 OR is_current IS NULL)" if current_only else ""
    rows = conn.execute(
        f"""
        SELECT *
          FROM live_odds
         WHERE lower(coalesce(market_type, 'win')) = 'win'
           AND odds_decimal IS NOT NULL
           AND odds_decimal > 1
           AND CAST(box_number AS INTEGER) = ?
           AND (
                race_id = ?
             OR (race_date = ? AND CAST(race_number AS INTEGER) = ?)
           )
           {current_clause}
        """,
        (box, exact_race_id, race_date, race_number),
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        row_box = safe_int(record.get("box_number"))
        row_dog_key = normalize_dog_name(record.get("dog_clean_name") or record.get("dog_name"))
        if row_box != box or row_dog_key != dog_key:
            continue
        direct_race_id = exact_race_id and str(record.get("race_id") or "") == exact_race_id
        venue_date_race = (
            race_date
            and race_number is not None
            and str(record.get("race_date") or "")[:10] == str(race_date)
            and safe_int(record.get("race_number")) == race_number
            and normalize_venue(record.get("venue")) in aliases
        )
        if direct_race_id or venue_date_race:
            record["_match_basis"] = "race_id_box_name" if direct_race_id else "venue_date_race_box_name"
            candidates.append(record)
    return candidates


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(str(value).strip())
    except Exception:
        return None


def odds_snapshot_from_row(
    row: Mapping[str, Any],
    *,
    prediction_time: datetime,
    feature_freeze_time: datetime | None = None,
    jump_time: datetime | None = None,
    duplicate_count: int,
    stale_odds_after_minutes: float = DEFAULT_STALE_ODDS_AFTER_MINUTES,
) -> dict[str, Any]:
    odds_timestamp = row.get("timestamp") or row.get("capture_timestamp")
    odds_dt = parse_timestamp(odds_timestamp)
    age_at_prediction = seconds_between(prediction_time, odds_dt)
    provenance = {
        "source": row.get("source"),
        "source_url": row.get("source_url"),
        "source_table": "live_odds",
        "odds_id": row.get("id"),
        "odds_race_id": row.get("race_id"),
        "odds_dog_name": row.get("dog_clean_name") or row.get("dog_name"),
        "odds_box_number": row.get("box_number"),
        "match_type": row.get("_match_basis") or "race_id_box_name",
        "match_method": "race_id_box_name_exact",
        "match_confidence": 1.0,
        "candidate_count": duplicate_count,
        "duplicate_count": duplicate_count,
        "sportsbet_box_source": row.get("sportsbet_box_source") or "unknown",
        "sportsbet_list_position": row.get("sportsbet_list_position"),
        "sportsbet_raw_runner_text": row.get("sportsbet_raw_runner_text"),
        "capture_mode": row.get("capture_mode"),
    }
    snapshot = {
        "market_odds_win": row.get("odds_decimal"),
        "market_type": row.get("market_type") or "win",
        "odds_level": row.get("odds_level") or "dog",
        "odds_timestamp": odds_timestamp,
        "odds_age_seconds_at_prediction": age_at_prediction,
        "odds_age_minutes_at_prediction": (
            age_at_prediction / 60.0 if age_at_prediction is not None else None
        ),
        "odds_captured_before_prediction": (
            age_at_prediction is not None and age_at_prediction >= 0
        ),
        "odds_provenance": provenance,
    }
    if age_at_prediction is not None:
        snapshot["odds_stale_at_prediction"] = (
            age_at_prediction > stale_odds_after_minutes * 60.0
        )
        snapshot["stale_odds_after_minutes"] = stale_odds_after_minutes
    age_at_feature_freeze = seconds_between(feature_freeze_time, odds_dt)
    if age_at_feature_freeze is not None:
        snapshot["odds_age_seconds_at_feature_freeze"] = age_at_feature_freeze
        snapshot["odds_age_minutes_at_feature_freeze"] = age_at_feature_freeze / 60.0
        snapshot["odds_captured_before_feature_freeze"] = age_at_feature_freeze >= 0
    age_at_jump = seconds_between(jump_time, odds_dt)
    if age_at_jump is not None:
        snapshot["odds_age_seconds_at_jump"] = age_at_jump
        snapshot["odds_age_minutes_at_jump"] = age_at_jump / 60.0
        snapshot["odds_captured_before_jump"] = age_at_jump >= 0
    return {key: value for key, value in snapshot.items() if value is not None}


def _race_id_from_prediction(prediction: Mapping[str, Any]) -> str:
    return str(prediction.get("race_id") or "").strip()


def _odds_source_url(row: Mapping[str, Any]) -> str | None:
    snapshot = row.get("odds_snapshot") if isinstance(row.get("odds_snapshot"), Mapping) else {}
    provenance = (
        snapshot.get("odds_provenance")
        if isinstance(snapshot.get("odds_provenance"), Mapping)
        else {}
    )
    source_url = provenance.get("source_url")
    return str(source_url) if source_url not in (None, "") else None


def race_odds_coverage_report(
    *,
    predictions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    collection_status: str,
) -> dict[str, Any]:
    predictions_by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows_by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        race_id = _race_id_from_prediction(prediction)
        if race_id:
            predictions_by_race[race_id].append(prediction)
    for row in rows:
        race_id = _race_id_from_prediction(row)
        if race_id:
            rows_by_race[race_id].append(row)

    race_reports: list[dict[str, Any]] = []
    for race_id in sorted(predictions_by_race):
        race_rows = rows_by_race.get(race_id, [])
        prediction_count = len(predictions_by_race[race_id])
        runner_rows_with_candidates = sum(1 for row in race_rows if row.get("odds_candidate_count"))
        total_candidate_count = sum(int(row.get("odds_candidate_count") or 0) for row in race_rows)
        valid_rows = sum(
            1 for row in race_rows if row.get("odds_match_status") == "valid_pre_jump_dog_odds"
        )
        ev_eligible_rows = sum(1 for row in race_rows if row.get("is_ev_eligible") is True)
        exclusion_counts = Counter(
            str(row.get("odds_exclusion_reason") or "none") for row in race_rows
        )
        duplicate_rows = exclusion_counts.get("duplicate_odds_rows", 0)
        post_prediction_rows = sum(
            1
            for row in race_rows
            if row.get("odds_exclusion_reason") in POST_PREDICTION_ODDS_REASONS
        )
        post_feature_freeze_rows = sum(
            1
            for row in race_rows
            if row.get("odds_exclusion_reason") in POST_FEATURE_FREEZE_ODDS_REASONS
        )
        post_jump_rows = sum(
            1
            for row in race_rows
            if row.get("odds_exclusion_reason") in POST_JUMP_ODDS_REASONS
        )
        missing_rows = max(0, prediction_count - runner_rows_with_candidates)
        complete_candidate_coverage = (
            prediction_count > 0 and runner_rows_with_candidates == prediction_count
        )
        complete_valid_prejump_odds = prediction_count > 0 and valid_rows == prediction_count
        odds_analysis_blockers: list[str] = []
        seen_blockers: set[str] = set()

        def append_blocker(reason: str) -> None:
            if reason not in seen_blockers:
                odds_analysis_blockers.append(reason)
                seen_blockers.add(reason)

        if prediction_count <= 0:
            append_blocker("no_shadow_predictions")
        if missing_rows > 0:
            append_blocker("missing_odds_rows")
        for row in race_rows:
            reason = str(row.get("odds_exclusion_reason") or "")
            if reason and reason not in {"none", "no_odds_row"}:
                append_blocker(reason)
        if prediction_count > 0 and not complete_valid_prejump_odds:
            append_blocker("incomplete_valid_prejump_odds")
        odds_analysis_status = (
            "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED"
            if prediction_count > 0 and not odds_analysis_blockers
            else "ODDS_ANALYSIS_BLOCKED"
        )

        if collection_status == FINAL_DB_BLOCKED:
            coverage_status = "ODDS_NOT_CHECKED_DB_BLOCKED"
        elif collection_status == FINAL_ODDS_UNAVAILABLE:
            coverage_status = "ODDS_NOT_CHECKED_SOURCE_UNAVAILABLE"
        elif complete_valid_prejump_odds:
            coverage_status = "COMPLETE_VALID_PREJUMP_ODDS"
        elif complete_candidate_coverage:
            coverage_status = "COMPLETE_CANDIDATE_COVERAGE_WITH_REJECTIONS"
        elif runner_rows_with_candidates:
            coverage_status = "PARTIAL_ODDS_COVERAGE"
        else:
            coverage_status = "NO_ODDS_COVERAGE"

        source_urls = sorted({url for row in race_rows for url in [_odds_source_url(row)] if url})
        status_counts = Counter(str(row.get("odds_match_status") or "unknown") for row in race_rows)
        race_reports.append(
            {
                "race_id": race_id,
                "race_context": dict(contexts.get(race_id) or {}),
                "odds_coverage_status": coverage_status,
                "predicted_runner_count": prediction_count,
                "runner_rows_checked": len(race_rows),
                "runner_rows_with_odds_candidates": runner_rows_with_candidates,
                "total_odds_candidate_count": total_candidate_count,
                "valid_pre_jump_dog_odds_rows": valid_rows,
                "ev_eligible_rows": ev_eligible_rows,
                "missing_odds_rows": missing_rows,
                "duplicate_odds_rows": duplicate_rows,
                "post_prediction_odds_rows": post_prediction_rows,
                "post_feature_freeze_odds_rows": post_feature_freeze_rows,
                "post_jump_odds_rows": post_jump_rows,
                "stale_odds_rows": exclusion_counts.get("stale_beyond_ttl", 0),
                "missing_source_url_rows": exclusion_counts.get("missing_source_url", 0),
                "untrusted_source_rows": exclusion_counts.get("untrusted_source", 0),
                "post_race_or_sp_only_rows": exclusion_counts.get(
                    "post_race_or_sp_only", 0
                ),
                "complete_odds_candidate_coverage": complete_candidate_coverage,
                "complete_valid_prejump_odds": complete_valid_prejump_odds,
                "odds_analysis_status": odds_analysis_status,
                "odds_analysis_blockers": odds_analysis_blockers,
                "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
                "odds_match_status_distribution": dict(sorted(status_counts.items())),
                "odds_exclusion_reason_distribution": dict(sorted(exclusion_counts.items())),
                "odds_source_urls": source_urls,
                "odds_source_url_count": len(source_urls),
            }
        )

    return {
        "schema_version": "shadow_odds_race_coverage_v1",
        "collection_status": collection_status,
        "race_count": len(race_reports),
        "races": race_reports,
        "races_with_any_odds_candidates": sum(
            1 for race in race_reports if race["runner_rows_with_odds_candidates"] > 0
        ),
        "races_with_complete_odds_candidate_coverage": sum(
            1 for race in race_reports if race["complete_odds_candidate_coverage"]
        ),
        "races_with_complete_valid_prejump_odds": sum(
            1 for race in race_reports if race["complete_valid_prejump_odds"]
        ),
        "races_with_missing_odds_rows": sum(
            1 for race in race_reports if race["missing_odds_rows"] > 0
        ),
        "races_with_duplicate_odds_rows": sum(
            1 for race in race_reports if race["duplicate_odds_rows"] > 0
        ),
        "races_with_post_prediction_odds_rows": sum(
            1 for race in race_reports if race["post_prediction_odds_rows"] > 0
        ),
        "races_with_post_feature_freeze_odds_rows": sum(
            1 for race in race_reports if race["post_feature_freeze_odds_rows"] > 0
        ),
        "races_with_post_jump_odds_rows": sum(
            1 for race in race_reports if race["post_jump_odds_rows"] > 0
        ),
        "races_with_stale_odds_rows": sum(
            1 for race in race_reports if race["stale_odds_rows"] > 0
        ),
        "races_with_missing_source_url_rows": sum(
            1 for race in race_reports if race["missing_source_url_rows"] > 0
        ),
        "races_with_untrusted_source_rows": sum(
            1 for race in race_reports if race["untrusted_source_rows"] > 0
        ),
        "races_with_post_race_or_sp_only_rows": sum(
            1 for race in race_reports if race["post_race_or_sp_only_rows"] > 0
        ),
    }


def odds_research_readiness_report(
    *,
    predictions: Sequence[Mapping[str, Any]],
    race_coverage: Mapping[str, Any],
    collection_status: str,
    odds_source_report: Mapping[str, Any],
) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    race_status_counts: Counter[str] = Counter()
    blocked_races: list[dict[str, Any]] = []

    if not predictions:
        blocker_counts["no_shadow_predictions"] += 1
    if collection_status == FINAL_DB_BLOCKED:
        blocker_counts["db_state_blocked"] += 1
    if collection_status == FINAL_ODDS_UNAVAILABLE:
        blocker_counts["live_odds_source_unavailable"] += 1
    if not odds_source_report.get("live_odds_table_available") and predictions:
        blocker_counts["live_odds_table_missing"] += 1

    for race in race_coverage.get("races") or []:
        status = str(race.get("odds_analysis_status") or "UNKNOWN")
        race_status_counts[status] += 1
        blockers = [str(item) for item in race.get("odds_analysis_blockers") or []]
        for blocker in blockers:
            blocker_counts[blocker] += 1
        if blockers:
            blocked_races.append(
                {
                    "race_id": race.get("race_id"),
                    "odds_coverage_status": race.get("odds_coverage_status"),
                    "predicted_runner_count": race.get("predicted_runner_count"),
                    "valid_pre_jump_dog_odds_rows": race.get(
                        "valid_pre_jump_dog_odds_rows"
                    ),
                    "missing_odds_rows": race.get("missing_odds_rows"),
                    "duplicate_odds_rows": race.get("duplicate_odds_rows"),
                    "post_prediction_odds_rows": race.get(
                        "post_prediction_odds_rows"
                    ),
                    "post_feature_freeze_odds_rows": race.get(
                        "post_feature_freeze_odds_rows"
                    ),
                    "post_jump_odds_rows": race.get("post_jump_odds_rows"),
                    "blockers": blockers,
                }
            )

    race_count = int(race_coverage.get("race_count") or 0)
    complete_valid_races = int(
        race_coverage.get("races_with_complete_valid_prejump_odds") or 0
    )
    all_races_complete = race_count > 0 and complete_valid_races == race_count
    status = (
        "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED"
        if all_races_complete and not blocker_counts
        else "ODDS_ANALYSIS_BLOCKED"
    )
    if not predictions:
        next_action = "WAIT_FOR_SHADOW_PREDICTIONS"
    elif collection_status == FINAL_DB_BLOCKED:
        next_action = "RESTORE_DB_STATE_BEFORE_ODDS_RESEARCH"
    elif collection_status == FINAL_ODDS_UNAVAILABLE:
        next_action = "ADD_OR_VERIFY_LIVE_ODDS_SOURCE_TABLE"
    elif blocker_counts.get("missing_odds_rows"):
        next_action = "COLLECT_EXACT_PREJUMP_DOG_ODDS_FOR_ALL_RUNNERS"
    elif blocker_counts.get("duplicate_odds_rows"):
        next_action = "FIX_ODDS_DEDUPLICATION_OR_IDENTITY_PROVENANCE"
    elif blocker_counts.get("timestamp_after_prediction"):
        next_action = "CAPTURE_ODDS_BEFORE_SHADOW_PREDICTION_AND_FEATURE_FREEZE"
    elif blocker_counts.get("timestamp_after_feature_freeze"):
        next_action = "CAPTURE_ODDS_BEFORE_SHADOW_PREDICTION_AND_FEATURE_FREEZE"
    elif blocker_counts.get("timestamp_after_jump"):
        next_action = "CAPTURE_ODDS_BEFORE_RACE_JUMP"
    elif blocker_counts.get("stale_beyond_ttl"):
        next_action = "REFRESH_ODDS_WITHIN_TTL_BEFORE_FEATURE_FREEZE"
    elif (
        blocker_counts.get("missing_source_url")
        or blocker_counts.get("untrusted_source")
        or blocker_counts.get("post_race_or_sp_only")
    ):
        next_action = "FIX_ODDS_SOURCE_PROVENANCE"
    elif (
        blocker_counts.get("race_id_mismatch")
        or blocker_counts.get("box_mismatch")
        or blocker_counts.get("dog_name_mismatch")
        or blocker_counts.get("ambiguous_runner_identity")
        or blocker_counts.get("ambiguous_box_source")
    ):
        next_action = "FIX_ODDS_IDENTITY_PROVENANCE"
    elif all_races_complete:
        next_action = "REPORT_ONLY_REVIEW_ODDS_CALIBRATION_NO_EV_ACTION"
    else:
        next_action = "CONTINUE_ODDS_PROVENANCE_COLLECTION"
    return {
        "schema_version": "shadow_odds_research_readiness_v1",
        "status": status,
        "collection_status": collection_status,
        "race_count": race_count,
        "prediction_rows": len(predictions),
        "races_with_complete_valid_prejump_odds": complete_valid_races,
        "all_races_complete_valid_prejump_odds": all_races_complete,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "race_status_counts": dict(sorted(race_status_counts.items())),
        "blocked_races": blocked_races,
        "odds_research_gate_policy": ODDS_RESEARCH_GATE_POLICY,
        "odds_research_next_action": next_action,
        "ev_research_gate": {
            "status": "BLOCKED_REPORT_ONLY_NO_EV_OUTPUT"
            if status != "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED"
            else "READY_FOR_REPORT_ONLY_ODDS_REVIEW_NO_EV_OUTPUT",
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "ev_output_allowed": False,
            "betting_action_allowed": False,
            "odds_used_for_shadow_scoring": False,
            "requires_complete_valid_prejump_odds": True,
        },
        "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
        "ev_output_rows": 0,
        "odds_used_for_shadow_scoring": False,
        "no_write_guarantees": NO_WRITE_GUARANTEES,
    }


def collect_shadow_odds_snapshot(
    *,
    shadow_run_dir: Path,
    db_path: Path,
    output_dir: Path,
    current_time: datetime | None = None,
    current_only: bool = True,
    stale_odds_after_minutes: float = DEFAULT_STALE_ODDS_AFTER_MINUTES,
) -> dict[str, Any]:
    if not math.isfinite(stale_odds_after_minutes) or stale_odds_after_minutes <= 0:
        raise ValueError("stale_odds_after_minutes_must_be_positive")
    generated_at = current_time or datetime.now().astimezone()
    output_dir = assert_output_dir_safe(output_dir)
    protected_before = protected_hashes()
    db_state = verify_db_state(db_path)
    predictions = read_jsonl(shadow_predictions_path(shadow_run_dir))
    contexts = load_race_contexts(shadow_run_dir)
    manifest = read_json(shadow_run_dir / "shadow_manifest.json")
    prediction_time = manifest_timestamp(
        manifest,
        (
            ("prediction_timestamp",),
            ("score_live_manifest", "prediction_timestamp"),
            ("score_live_manifest", "generated_at"),
            ("generated_at",),
        ),
    ) or generated_at
    feature_freeze_time = manifest_timestamp(
        manifest,
        (
            ("feature_freeze_timestamp",),
            ("score_live_manifest", "feature_freeze_timestamp"),
        ),
    )

    rows: list[dict[str, Any]] = []
    status = FINAL_NO_PREDICTIONS if not predictions else FINAL_NO_MATCHES
    odds_source_report: dict[str, Any] = {"live_odds_table_available": False}

    if db_state.get("status") != "PASS":
        status = FINAL_DB_BLOCKED
    elif predictions:
        try:
            with open_readonly_db(db_path) as conn:
                live_odds_columns = table_columns(conn, "live_odds")
                odds_source_report = {
                    "live_odds_table_available": bool(live_odds_columns),
                    "live_odds_columns": sorted(live_odds_columns),
                }
                if not live_odds_columns:
                    status = FINAL_ODDS_UNAVAILABLE
                else:
                    for prediction in predictions:
                        candidates = candidate_rows_for_prediction(
                            conn,
                            prediction,
                            current_only=current_only,
                        )
                        duplicate_count = len(candidates)
                        selected = candidates[0] if candidates else None
                        odds_snapshot = (
                            odds_snapshot_from_row(
                                selected,
                                prediction_time=prediction_time,
                                feature_freeze_time=feature_freeze_time,
                                jump_time=parse_jump_datetime(
                                    contexts.get(str(prediction.get("race_id") or ""))
                                ),
                                duplicate_count=duplicate_count,
                                stale_odds_after_minutes=stale_odds_after_minutes,
                            )
                            if selected
                            else {}
                        )
                        runner = {
                            "dog_name": prediction.get("dog_name"),
                            "box_number": prediction.get("box"),
                        }
                        eligibility = classify_odds_snapshot_for_ev(
                            runner,
                            odds_snapshot,
                            snapshot_race_id=prediction.get("race_id"),
                        )
                        row = {
                            **prediction,
                            "schema_version": "shadow_odds_snapshot_runner_v1",
                            "race_context": contexts.get(str(prediction.get("race_id") or "")) or {},
                            "odds_candidate_count": duplicate_count,
                            "odds_snapshot": odds_snapshot,
                            "odds_match_status": eligibility.get("odds_match_status"),
                            "odds_match_method": eligibility.get("odds_match_method"),
                            "odds_exclusion_reason": eligibility.get("odds_exclusion_reason"),
                            "odds_provenance_status": eligibility.get("odds_provenance_status"),
                            "is_ev_eligible": eligibility.get("is_ev_eligible") is True,
                            "ev_win": None,
                            "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
                        }
                        rows.append(row)
                    if any(row["odds_candidate_count"] for row in rows):
                        status = FINAL_COLLECTED
        except Exception as exc:
            odds_source_report["error"] = f"odds_read_failed:{type(exc).__name__}"
            status = FINAL_ODDS_UNAVAILABLE

    protected_after = protected_hashes()
    status_counts = Counter(str(row.get("odds_match_status") or "unknown") for row in rows)
    exclusion_counts = Counter(str(row.get("odds_exclusion_reason") or "none") for row in rows)
    race_count = len({str(row.get("race_id") or "") for row in predictions if row.get("race_id")})
    race_coverage = race_odds_coverage_report(
        predictions=predictions,
        rows=rows,
        contexts=contexts,
        collection_status=status,
    )
    odds_research_readiness = odds_research_readiness_report(
        predictions=predictions,
        race_coverage=race_coverage,
        collection_status=status,
        odds_source_report=odds_source_report,
    )
    report = {
        "schema_version": "shadow_odds_snapshot_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": status,
        "shadow_run_dir": relpath(shadow_run_dir),
        "db_path": relpath(db_path),
        "db_state": db_state,
        "current_only": current_only,
        "stale_odds_after_minutes": stale_odds_after_minutes,
        "prediction_rows": len(predictions),
        "race_count": race_count,
        "runner_rows": len(rows),
        "odds_candidate_rows": sum(1 for row in rows if row.get("odds_candidate_count")),
        "valid_pre_jump_dog_odds_rows": sum(
            1 for row in rows if row.get("odds_match_status") == "valid_pre_jump_dog_odds"
        ),
        "ev_eligible_rows": sum(1 for row in rows if row.get("is_ev_eligible") is True),
        "ev_output_rows": 0,
        "odds_match_status_distribution": dict(sorted(status_counts.items())),
        "odds_exclusion_reason_distribution": dict(sorted(exclusion_counts.items())),
        "odds_source_report": odds_source_report,
        "odds_research_readiness": odds_research_readiness,
        "race_coverage_path": relpath(output_dir / "shadow_odds_race_coverage.json"),
        "races_with_any_odds_candidates": race_coverage["races_with_any_odds_candidates"],
        "races_with_complete_odds_candidate_coverage": race_coverage[
            "races_with_complete_odds_candidate_coverage"
        ],
        "races_with_complete_valid_prejump_odds": race_coverage[
            "races_with_complete_valid_prejump_odds"
        ],
        "races_with_missing_odds_rows": race_coverage["races_with_missing_odds_rows"],
        "races_with_duplicate_odds_rows": race_coverage["races_with_duplicate_odds_rows"],
        "races_with_post_prediction_odds_rows": race_coverage[
            "races_with_post_prediction_odds_rows"
        ],
        "races_with_post_feature_freeze_odds_rows": race_coverage[
            "races_with_post_feature_freeze_odds_rows"
        ],
        "races_with_post_jump_odds_rows": race_coverage[
            "races_with_post_jump_odds_rows"
        ],
        "races_with_stale_odds_rows": race_coverage["races_with_stale_odds_rows"],
        "races_with_missing_source_url_rows": race_coverage[
            "races_with_missing_source_url_rows"
        ],
        "races_with_untrusted_source_rows": race_coverage[
            "races_with_untrusted_source_rows"
        ],
        "races_with_post_race_or_sp_only_rows": race_coverage[
            "races_with_post_race_or_sp_only_rows"
        ],
        "no_write_guarantees": NO_WRITE_GUARANTEES,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_before == protected_after,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "shadow_odds_snapshot.jsonl", rows)
    write_csv(output_dir / "shadow_odds_snapshot.csv", rows)
    write_json(output_dir / "shadow_odds_race_coverage.json", race_coverage)
    write_json(output_dir / "shadow_odds_research_readiness.json", odds_research_readiness)
    write_json(output_dir / "shadow_odds_snapshot_report.json", report)
    write_text(output_dir / "final_status.txt", f"{status}\n")
    write_text(
        output_dir / "SUMMARY.md",
        "\n".join(
            [
                "# Shadow Odds Snapshot",
                "",
                f"- Final status: `{status}`",
                f"- Shadow run: `{relpath(shadow_run_dir)}`",
                f"- Prediction rows: `{len(predictions)}`",
                f"- Odds candidate rows: `{report['odds_candidate_rows']}`",
                f"- Valid pre-jump dog odds rows: `{report['valid_pre_jump_dog_odds_rows']}`",
                f"- Races with complete valid pre-jump odds: `{report['races_with_complete_valid_prejump_odds']}`",
                f"- Odds analysis status: `{odds_research_readiness['status']}`",
                f"- Odds analysis blockers: `{odds_research_readiness['blocker_counts']}`",
                f"- EV output rows: `0`",
                f"- Protected paths unchanged: `{report['protected_paths_unchanged']}`",
                "",
                "Report-only lane; no DB, label, registry, production model, production prediction, EV, betting, TGR, or training mutation.",
                "",
            ]
        ),
    )
    write_text(
        output_dir / "verification_results.txt",
        "\n".join(
            [
                f"db_status={db_state.get('status')}",
                f"quick_check={db_state.get('quick_check')}",
                f"official_races={db_state.get('official_races')}",
                f"official_dog_rows={db_state.get('official_dog_rows')}",
                f"protected_paths_unchanged={report['protected_paths_unchanged']}",
                "",
            ]
        ),
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-run-dir", required=True, type=Path)
    parser.add_argument("--db", default=DEFAULT_DB, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--current-time")
    parser.add_argument("--all-odds", action="store_true")
    parser.add_argument(
        "--stale-odds-after-minutes",
        type=float,
        default=DEFAULT_STALE_ODDS_AFTER_MINUTES,
    )
    args = parser.parse_args(argv)

    current_time = parse_current_time(args.current_time)
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_PARENT / f"shadow_odds_snapshot_{now_id(current_time)}"
    )
    report = collect_shadow_odds_snapshot(
        shadow_run_dir=args.shadow_run_dir,
        db_path=args.db,
        output_dir=output_dir,
        current_time=current_time,
        current_only=not args.all_odds,
        stale_odds_after_minutes=args.stale_odds_after_minutes,
    )
    print(
        json.dumps(
            {
                "final_status": report["final_status"],
                "output_dir": relpath(assert_output_dir_safe(output_dir)),
                "prediction_rows": report["prediction_rows"],
                "odds_candidate_rows": report["odds_candidate_rows"],
                "valid_pre_jump_dog_odds_rows": report["valid_pre_jump_dog_odds_rows"],
                "protected_paths_unchanged": report["protected_paths_unchanged"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["final_status"] not in {FINAL_DB_BLOCKED} else 2


if __name__ == "__main__":
    raise SystemExit(main())
