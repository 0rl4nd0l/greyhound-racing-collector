#!/usr/bin/env python3
"""Collect official-result evidence into append-only artifacts.

This wrapper runs the existing official-first result ingester in dry-run mode,
then converts clean TheDogs official results into source-backed JSONL datasets.
It does not write labels, mutate the DB, rewrite snapshots, or promote models.
Unsafe, partial, non-official, and failed attempts are kept visible in a
quarantine JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import ingest_results_for_date as ingest
from scripts.refresh_prejump_upcoming import venue_exclusion_aliases
from utils.report_output_dir_guard import assert_prefixed_report_output_dir
from utils.runner_completeness import RunnerRow, analyze_runner_rows, normalise_runner_name

try:
    from config.venue_mapping import normalize_venue
except Exception:
    def normalize_venue(value: Any) -> str:
        return str(value or "").strip().upper()


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_"
OUTPUT_ARTIFACT_PREFIX = "autonomous_official_result_capture_"
OFFICIAL_SOURCE = "thedogs_official"
RESULTED_STATUS = "resulted"
OFFICIAL_RESULT_EVIDENCE_RACES_TABLE = "autonomous_official_result_evidence_races"
OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE = "autonomous_official_result_evidence_runners"
DEFAULT_BACKLOG_LOOKBACK_DAYS = 2
DEFAULT_BACKLOG_LIMIT = 128
DEFAULT_BACKLOG_SHADOW_RUN_LIMIT = 200
NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "tgr_enabled": False,
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


def flush_official_result_progress(
    output_dir: Path,
    *,
    candidates: Sequence[Any],
    progress_rows: Sequence[Mapping[str, Any]],
    active_row: Mapping[str, Any] | None = None,
) -> None:
    rows = list(progress_rows)
    if active_row is not None:
        rows.append(dict(active_row))
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    write_jsonl(
        output_dir / "autonomous_official_result_capture_attempts.progress.jsonl",
        rows,
    )
    write_json(
        output_dir / "autonomous_official_result_capture_progress.json",
        {
            "schema_version": "autonomous_official_result_capture_progress_v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "candidate_count": len(candidates),
            "completed_count": len(progress_rows),
            "active_candidate": dict(active_row) if active_row is not None else None,
            "status_counts": dict(sorted(status_counts.items())),
            "no_write_guarantees": {
                **NO_WRITE_GUARANTEES,
                "db_write": False,
            },
        },
    )


def parse_current_time(value: str | None) -> datetime:
    if not value:
        return datetime.now().astimezone()
    text = value.strip()
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def assert_output_dir_safe(
    output_dir: Path,
    *,
    evidence_root: Path | None = None,
) -> Path:
    return assert_prefixed_report_output_dir(
        output_dir,
        repo_root=ROOT,
        repo_prefix=OUTPUT_PREFIX,
        artifact_prefix=OUTPUT_ARTIFACT_PREFIX,
        prefix_error="output_dir_must_be_autonomous_official_result_capture_artifact",
        evidence_root=evidence_root,
    )


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def ingest_dry_run_command(
    *,
    db_path: Path,
    target_date: str,
    upcoming_dir: Path,
    snapshot_dir: Path | None,
    output_path: Path,
    race_ids: Sequence[str],
    require_ready_snapshot: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/ingest_results_for_date.py"),
        "--dry-run",
        "--db",
        str(db_path),
        "--date",
        target_date,
        "--upcoming-dir",
        str(upcoming_dir),
        "--output",
        str(output_path),
    ]
    if snapshot_dir is not None:
        command.extend(["--snapshot-dir", str(snapshot_dir)])
    if require_ready_snapshot:
        command.append("--require-ready-snapshot")
    for race_id in race_ids:
        command.extend(["--race-id", race_id])
    return command


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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


def load_official_result_artifact_rows(
    *,
    race_rows_path: Path,
    runner_rows_path: Path,
    quarantine_rows_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "race_rows": load_jsonl(race_rows_path),
        "runner_rows": load_jsonl(runner_rows_path),
        "quarantine_rows": load_jsonl(quarantine_rows_path) if quarantine_rows_path else [],
    }


def shared_lock_status(lock_path: Path | None) -> dict[str, Any]:
    if lock_path is None:
        return {
            "schema_version": "shared_lock_status_v1",
            "lock_path": None,
            "status": "not_configured",
            "write_allowed": True,
        }
    if not lock_path.exists():
        return {
            "schema_version": "shared_lock_status_v1",
            "lock_path": str(lock_path),
            "status": "missing",
            "write_allowed": True,
        }
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "schema_version": "shared_lock_status_v1",
            "lock_path": str(lock_path),
            "status": "unreadable",
            "error": f"{type(exc).__name__}:{exc}",
            "write_allowed": False,
        }
    if not isinstance(payload, Mapping):
        return {
            "schema_version": "shared_lock_status_v1",
            "lock_path": str(lock_path),
            "status": "invalid_payload",
            "write_allowed": False,
        }
    pid = parse_int(payload.get("pid"))
    if pid is None or pid <= 0:
        return {
            "schema_version": "shared_lock_status_v1",
            "lock_path": str(lock_path),
            "status": "present_without_pid",
            "lock": dict(payload),
            "write_allowed": False,
        }
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {
            "schema_version": "shared_lock_status_v1",
            "lock_path": str(lock_path),
            "status": "stale_dead_pid",
            "pid": pid,
            "lock": dict(payload),
            "write_allowed": True,
        }
    except PermissionError:
        return {
            "schema_version": "shared_lock_status_v1",
            "lock_path": str(lock_path),
            "status": "present_pid_permission_unknown",
            "pid": pid,
            "lock": dict(payload),
            "write_allowed": False,
        }
    return {
        "schema_version": "shared_lock_status_v1",
        "lock_path": str(lock_path),
        "status": "present_live_pid",
        "pid": pid,
        "lock": dict(payload),
        "write_allowed": False,
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


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_race_identity(race_id: Any) -> dict[str, Any]:
    text = str(race_id or "").strip()
    match = re.match(r"^Race\s+(\d+)\s+-\s+(.+?)\s+-\s+(\d{4}-\d{2}-\d{2})$", text)
    if not match:
        return {"race_number": None, "venue": None, "race_date": None}
    return {
        "race_number": int(match.group(1)),
        "venue": match.group(2).strip(),
        "race_date": match.group(3),
    }


def race_time_from_minutes(value: Any) -> str | None:
    minutes = parse_int(value)
    if minutes is None or minutes < 0 or minutes >= 24 * 60:
        return None
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def prediction_rows_from_shadow_run(shadow_run_dir: Path) -> list[dict[str, Any]]:
    stage2 = load_jsonl(shadow_run_dir / "stage2_shadow_predictions.jsonl")
    if stage2:
        return stage2
    return load_jsonl(shadow_run_dir / "shadow_predictions.jsonl")


def feature_rows_path_candidates(shadow_run_dir: Path) -> list[Path]:
    candidates = [shadow_run_dir / "shadow_feature_rows.json"]
    manifest = load_json(shadow_run_dir / "shadow_manifest.json")
    score_manifest = manifest.get("score_live_manifest") if isinstance(manifest, Mapping) else {}
    if isinstance(score_manifest, Mapping):
        feature_rows = score_manifest.get("feature_rows")
        if feature_rows:
            path = Path(str(feature_rows))
            candidates.append(path if path.is_absolute() else ROOT / path)
    candidates.append(shadow_run_dir / "shadow_score_live" / "shadow_feature_rows.json")
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def feature_rows_by_race_id(shadow_run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    payload = None
    for path in feature_rows_path_candidates(shadow_run_dir):
        if path.exists():
            payload = load_json_any(path)
            break
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(payload, list):
        return grouped
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        race_id = str(row.get("race_id") or "").strip()
        if race_id:
            grouped[race_id].append(dict(row))
    return grouped


def _runner_rows_from_predictions(rows: Sequence[Mapping[str, Any]]) -> list[RunnerRow]:
    runners: list[RunnerRow] = []
    for row in rows:
        box = parse_int(row.get("box") if row.get("box") not in (None, "") else row.get("box_number"))
        dog_name = str(row.get("dog_name") or "").strip()
        if box is None or not dog_name:
            continue
        runners.append(RunnerRow(box_number=box, dog_name=dog_name))
    return runners


def _same_participant_set(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    def key(row: Mapping[str, Any]) -> tuple[int | None, str]:
        return (parse_int(row.get("box_number")), normalise_runner_name(row.get("dog_name")))

    return {key(row) for row in left} == {key(row) for row in right}


def write_shadow_run_candidate_csv(path: Path, participants: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Dog Name", "Box"])
        writer.writeheader()
        for participant in participants:
            writer.writerow(
                {
                    "Dog Name": participant.get("dog_name"),
                    "Box": participant.get("box_number"),
                }
            )


def shadow_run_checkout_root(shadow_run_dir: Path) -> Path | None:
    marker = Path("artifacts/full_evidence_orchestration_20260525")
    parts = shadow_run_dir.resolve().parts
    marker_parts = marker.parts
    for index in range(0, len(parts) - len(marker_parts) + 1):
        if parts[index : index + len(marker_parts)] == marker_parts:
            if index == 0:
                return Path(".").resolve()
            return Path(*parts[:index])
    return None


def resolve_shadow_run_source_csv(source_csv_value: str, *, shadow_run_dir: Path) -> Path:
    source_csv = Path(source_csv_value)
    if source_csv.is_absolute():
        return source_csv

    candidate_paths = [
        (shadow_run_dir / source_csv).resolve(),
    ]
    checkout_root = shadow_run_checkout_root(shadow_run_dir)
    if checkout_root is not None:
        candidate_paths.append((checkout_root / source_csv).resolve())
    candidate_paths.append((ROOT / source_csv).resolve())

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidate_paths:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return next((candidate for candidate in unique_candidates if candidate.exists()), unique_candidates[0])


def shadow_run_candidates(
    *,
    shadow_run_dir: Path,
    target_date: str,
    current_time: datetime,
    race_ids: Sequence[str],
    output_dir: Path,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    predictions = prediction_rows_from_shadow_run(shadow_run_dir)
    features = feature_rows_by_race_id(shadow_run_dir)
    race_filter = {str(race_id) for race_id in race_ids if race_id}
    grouped_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        race_id = str(row.get("race_id") or "").strip()
        if not race_id:
            continue
        if race_filter and race_id not in race_filter:
            continue
        grouped_predictions[race_id].append(row)

    candidates: list[Any] = []
    skipped: list[dict[str, Any]] = []
    for race_id in sorted(grouped_predictions):
        identity = parse_race_identity(race_id)
        if identity.get("race_date") != target_date:
            skipped.append(
                {
                    "race_id": race_id,
                    "reason": "shadow_run_race_date_mismatch",
                    "race_date": identity.get("race_date"),
                    "target_date": target_date,
                }
            )
            continue
        runner_rows = _runner_rows_from_predictions(grouped_predictions[race_id])
        runner_completeness = analyze_runner_rows(
            runner_rows,
            source=f"{shadow_run_dir}/stage2_shadow_predictions.jsonl",
        ).as_dict()
        if runner_completeness.get("status") != "COMPLETE":
            skipped.append(
                {
                    "race_id": race_id,
                    "reason": "shadow_run_prediction_runner_set_incomplete",
                    "runner_completeness": runner_completeness,
                }
            )
            continue

        feature_group = features.get(race_id) or []
        first_feature = feature_group[0] if feature_group else {}
        source_csv_value = str(first_feature.get("source_csv") or "").strip()
        csv_participants: list[dict[str, Any]] = []
        if not source_csv_value:
            skipped.append(
                {
                    "race_id": race_id,
                    "reason": "shadow_run_source_csv_missing",
                }
            )
            continue
        source_csv = resolve_shadow_run_source_csv(source_csv_value, shadow_run_dir=shadow_run_dir)
        if not source_csv.exists():
            skipped.append(
                {
                    "race_id": race_id,
                    "reason": "shadow_run_source_csv_missing",
                    "source_csv": str(source_csv),
                }
            )
            continue
        csv_path = source_csv
        csv_participants = ingest.parse_participants_from_csv(source_csv)
        if not csv_participants or not _same_participant_set(
            runner_completeness.get("participants") or [],
            csv_participants,
        ):
            skipped.append(
                {
                    "race_id": race_id,
                    "reason": "shadow_run_source_csv_participant_mismatch",
                    "prediction_participants": runner_completeness.get("participants") or [],
                    "csv_participants": csv_participants,
                    "source_csv": str(source_csv),
                }
            )
            continue

        race_time = race_time_from_minutes(first_feature.get("race_time_minutes_since_midnight"))
        record = {
            "race_id": race_id,
            "venue": identity.get("venue"),
            "race_number": identity.get("race_number"),
            "race_date": identity.get("race_date"),
            "race_time": race_time,
            "start_datetime": None,
            "results_status": None,
            "winner_name": None,
            "source_path": str(csv_path),
        }
        lifecycle = ingest.classify_race_record(
            record,
            now=current_time,
            source_context="shadow_run_prediction",
        )
        if lifecycle.status == ingest.UPCOMING_NOT_JUMPED:
            skipped.append(
                {
                    "race_id": race_id,
                    "reason": f"race_not_jumped:{lifecycle.status}",
                    "jump_datetime": lifecycle.jump_datetime,
                    "race_time": race_time,
                }
            )
            continue

        candidates.append(
            ingest.RaceCandidate(
                race_id=race_id,
                venue=str(identity.get("venue") or ""),
                race_number=int(identity.get("race_number") or 0),
                race_date=str(identity.get("race_date") or ""),
                race_time=race_time,
                start_datetime=lifecycle.jump_datetime,
                sportsbet_url=None,
                csv_path=csv_path,
                participants=list(runner_completeness.get("participants") or []),
                lifecycle_status=lifecycle.status,
                participant_source="shadow_run_predictions",
                csv_participants=csv_participants or list(runner_completeness.get("participants") or []),
                runner_completeness=runner_completeness,
                canonical_thedogs_url=first_feature.get("target_metadata_source_url"),
            )
        )

    source_report = {
        "schema_version": "shadow_run_official_result_candidate_source_v1",
        "shadow_run_dir": relpath(shadow_run_dir),
        "target_date": target_date,
        "prediction_rows": len(predictions),
        "prediction_race_count": len({str(row.get("race_id") or "") for row in predictions if row.get("race_id")}),
        "candidate_count": len(candidates),
        "skipped_count": len(skipped),
        "candidate_race_ids": sorted(candidate.race_id for candidate in candidates),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    return candidates, skipped, source_report


def source_backed_live_odds_backlog_race_ids(
    *,
    db_path: Path,
    target_date: str,
    limit: int,
    lookback_days: int = 0,
) -> list[str]:
    """Find source-backed odds races that still need official result evidence."""

    return [
        str(item.get("race_id"))
        for item in source_backed_live_odds_backlog_entries(
            db_path=db_path,
            target_date=target_date,
            limit=limit,
            lookback_days=lookback_days,
        )
    ]


def live_odds_backlog_target_dates(target_date: str, *, lookback_days: int) -> list[str]:
    try:
        base = date.fromisoformat(str(target_date))
    except ValueError:
        return [target_date]
    bounded_lookback = max(0, int(lookback_days or 0))
    return [
        (base - timedelta(days=offset)).isoformat()
        for offset in range(bounded_lookback + 1)
    ]


def source_backed_live_odds_backlog_entries(
    *,
    db_path: Path,
    target_date: str,
    limit: int,
    lookback_days: int = 0,
) -> list[dict[str, Any]]:
    """Find source-backed odds races that still need official result evidence."""

    if limit <= 0 or not db_path.exists():
        return []
    target_dates = live_odds_backlog_target_dates(
        target_date,
        lookback_days=lookback_days,
    )
    placeholders = ",".join("?" for _ in target_dates)
    try:
        with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            live_odds_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(live_odds)").fetchall()
            }
            venue_expr = (
                "MAX(COALESCE(lo.venue, ''))" if "venue" in live_odds_columns else "''"
            )
            race_number_expr = (
                "MAX(COALESCE(lo.race_number, ''))"
                if "race_number" in live_odds_columns
                else "''"
            )
            box_count_expr = (
                """
                COUNT(DISTINCT CASE
                    WHEN CAST(lo.box_number AS INTEGER) > 0
                    THEN CAST(lo.box_number AS INTEGER)
                END)
                """
                if "box_number" in live_odds_columns
                else "0"
            )
            box_sources_expr = (
                "GROUP_CONCAT(DISTINCT COALESCE(lo.sportsbet_box_source, ''))"
                if "sportsbet_box_source" in live_odds_columns
                else "''"
            )
            query = """
        SELECT
            lo.race_id,
            lo.race_date,
            MAX(COALESCE(lo.capture_timestamp, lo.timestamp, '')) AS latest_capture,
            {venue_expr} AS venue,
            {race_number_expr} AS race_number,
            MAX(COALESCE(lo.source_url, '')) AS source_url,
            COUNT(*) AS odds_row_count,
            {box_count_expr} AS box_count,
            {box_sources_expr} AS sportsbet_box_sources
        FROM live_odds lo
        WHERE lo.race_date IN ({placeholders})
          AND lo.race_id IS NOT NULL
          AND TRIM(lo.race_id) != ''
          AND lo.source_url IS NOT NULL
          AND TRIM(lo.source_url) != ''
          AND lo.capture_timestamp IS NOT NULL
          AND TRIM(lo.capture_timestamp) != ''
          AND LOWER(COALESCE(lo.market_type, '')) = 'win'
          AND lo.odds_decimal IS NOT NULL
          AND lo.odds_decimal > 1.0
          AND LOWER(COALESCE(lo.odds_level, 'dog')) IN ('dog', 'runner', '')
          AND COALESCE(lo.sportsbet_box_source, '') IN ('explicit_dom', 'runner_text')
          AND NOT EXISTS (
              SELECT 1
              FROM race_metadata rm
              WHERE rm.race_id = lo.race_id
                AND rm.winner_source = ?
          )
        GROUP BY lo.race_id
        ORDER BY lo.race_date DESC, latest_capture ASC, lo.race_id ASC
        LIMIT ?
    """.format(
                placeholders=placeholders,
                venue_expr=venue_expr,
                race_number_expr=race_number_expr,
                box_count_expr=box_count_expr,
                box_sources_expr=box_sources_expr,
            )
            rows = conn.execute(
                query,
                [*target_dates, OFFICIAL_SOURCE, int(limit)],
            ).fetchall()
    except sqlite3.Error:
        return []
    entries: list[dict[str, Any]] = []
    for row in rows:
        race_id = str(row[0] if row else "").strip()
        if not race_id:
            continue
        entries.append(
            {
                "race_id": race_id,
                "race_date": row[1],
                "latest_capture": row[2],
                "venue": row[3],
                "race_number": row[4],
                "source_url": row[5],
                "odds_row_count": row[6],
                "box_count": row[7],
                "sportsbet_box_sources": sorted(
                    {
                        str(value).strip()
                        for value in str(row[8] or "").split(",")
                        if str(value).strip()
                    }
                ),
            }
        )
    return entries


def sportsbet_venue_slug(source_url: Any) -> str | None:
    match = re.search(
        r"/greyhound-racing/(?:[^/?#]+/)*([^/?#]+)/race-",
        str(source_url or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip()


def shadow_identity_candidates_from_live_odds_entry(
    entry: Mapping[str, Any],
) -> list[str]:
    race_date = str(entry.get("race_date") or "").strip()[:10]
    race_number = parse_int(entry.get("race_number"))
    if not race_date or race_number is None:
        return []

    aliases: list[str] = []

    def add_alias(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        for candidate in [
            text.upper(),
            text.upper().replace("_", "-"),
            text.upper().replace("-", "_"),
            re.sub(r"[^A-Z0-9]+", "-", text.upper()).strip("-"),
        ]:
            if candidate and candidate not in aliases:
                aliases.append(candidate)

    venue = str(entry.get("venue") or "").strip()
    add_alias(venue)
    normalized = normalize_venue(venue)
    add_alias(normalized)
    slug = sportsbet_venue_slug(entry.get("source_url"))
    if slug:
        add_alias(slug)
        add_alias(slug.replace("-", " "))
    for alias in sorted(venue_exclusion_aliases(venue, source_url=entry.get("source_url"))):
        add_alias(alias)

    return [
        f"Race {race_number} - {alias} - {race_date}"
        for alias in aliases
    ]


def canonical_live_odds_race_id_from_entry(entry: Mapping[str, Any]) -> str | None:
    race_date = str(entry.get("race_date") or "").strip()[:10]
    race_number = parse_int(entry.get("race_number"))
    venue = str(entry.get("venue") or "").strip()
    normalized = normalize_venue(venue)
    if not normalized or not race_date or race_number is None:
        return None
    return f"{normalized}_{race_date}_{race_number}"


def shadow_artifact_matches_for_race_ids(
    candidate_race_ids: Sequence[str],
    shadow_run_dirs: Sequence[Path],
    *,
    max_matches_per_race_id: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    targets = {str(race_id) for race_id in candidate_race_ids if race_id}
    if not targets:
        return {}
    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shadow_run_dir in shadow_run_dirs:
        if all(
            len(matches.get(race_id, [])) >= max_matches_per_race_id
            for race_id in targets
        ):
            break
        sources_by_race_id: dict[str, set[str]] = defaultdict(set)
        for row in prediction_rows_from_shadow_run(shadow_run_dir):
            if not isinstance(row, Mapping):
                continue
            race_id = str(row.get("race_id") or "").strip()
            if race_id in targets:
                sources_by_race_id[race_id].add("shadow_predictions")
        for race_id in feature_rows_by_race_id(shadow_run_dir):
            if race_id in targets:
                sources_by_race_id[race_id].add("shadow_feature_rows")
        for race_id, sources in sources_by_race_id.items():
            if len(matches[race_id]) >= max_matches_per_race_id:
                continue
            matches[race_id].append(
                {
                    "race_id": race_id,
                    "shadow_run_dir": relpath(shadow_run_dir),
                    "artifact_sources": sorted(sources),
                }
            )
    return {race_id: rows for race_id, rows in sorted(matches.items())}


def alias_reconciliation_status(
    candidate_shadow_race_ids: Sequence[str],
    matches_by_race_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    if not candidate_shadow_race_ids:
        return "NO_CANDIDATE_SHADOW_RACE_IDS"
    if any(matches_by_race_id.get(str(race_id)) for race_id in candidate_shadow_race_ids):
        return "EXACT_SHADOW_ARTIFACT_MATCH_FOUND"
    return "NO_EXACT_SHADOW_ARTIFACT_MATCH"


def unresolved_live_odds_backlog_diagnostics(
    *,
    unresolved_race_ids: Sequence[str],
    backlog_entries: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    shadow_run_report_count: int,
    shadow_run_dirs: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    unresolved = [str(race_id) for race_id in unresolved_race_ids if race_id]
    entry_by_race_id = {
        str(entry.get("race_id")): entry
        for entry in backlog_entries
        if str(entry.get("race_id") or "").strip()
    }
    skipped_by_race_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unresolved_set = set(unresolved)
    for item in skipped:
        race_id = str(item.get("race_id") or "").strip()
        if race_id in unresolved_set:
            skipped_by_race_id[race_id].append(item)

    diagnostics: list[dict[str, Any]] = []
    for race_id in unresolved:
        identity = parse_race_identity(race_id)
        skip_rows = skipped_by_race_id.get(race_id, [])
        skip_reasons = sorted(
            {
                str(item.get("reason") or "skipped_before_fetch")
                for item in skip_rows
            }
        )
        if skip_reasons:
            reason = "shadow_run_candidate_rejected"
        elif identity.get("race_date") is None:
            reason = "live_odds_race_id_not_canonical_shadow_race_id"
        else:
            reason = "no_matching_shadow_run_candidate_found"

        entry = entry_by_race_id.get(race_id) or {}
        candidate_shadow_ids = shadow_identity_candidates_from_live_odds_entry(entry)
        canonical_live_odds_race_id = canonical_live_odds_race_id_from_entry(entry)
        shadow_matches = shadow_artifact_matches_for_race_ids(
            candidate_shadow_ids,
            shadow_run_dirs,
        )
        flat_shadow_matches = [
            match
            for candidate_id in candidate_shadow_ids
            for match in shadow_matches.get(candidate_id, [])
        ]
        reconciliation_status = alias_reconciliation_status(
            candidate_shadow_ids,
            shadow_matches,
        )
        diagnostics.append(
            {
                "race_id": race_id,
                "race_date": entry.get("race_date") or identity.get("race_date"),
                "latest_capture": entry.get("latest_capture"),
                "live_odds_venue": entry.get("venue"),
                "live_odds_race_number": entry.get("race_number"),
                "live_odds_source_url": entry.get("source_url"),
                "live_odds_row_count": entry.get("odds_row_count"),
                "live_odds_box_count": entry.get("box_count"),
                "live_odds_box_sources": list(
                    entry.get("sportsbet_box_sources") or []
                ),
                "canonical_live_odds_race_id": canonical_live_odds_race_id,
                "candidate_shadow_race_ids": candidate_shadow_ids,
                "candidate_shadow_race_id_match_count": len(flat_shadow_matches),
                "candidate_shadow_race_id_matches": flat_shadow_matches,
                "alias_reconciliation_status": reconciliation_status,
                "reason": reason,
                "parsed_identity": identity,
                "shadow_run_report_count": shadow_run_report_count,
                "shadow_run_skip_reasons": skip_reasons,
                "recovery_action": (
                    "validate_runner_set_then_alias_join"
                    if reconciliation_status == "EXACT_SHADOW_ARTIFACT_MATCH_FOUND"
                    else (
                        "recover_shadow_predictions_for_source_identity"
                        if reason == "live_odds_race_id_not_canonical_shadow_race_id"
                        else "inspect_shadow_run_candidate_coverage"
                    )
                ),
                "source": "source_backed_live_odds_without_official_results",
            }
        )
    return diagnostics


def reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def field_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or f"missing_{field}")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def retryable_exact_shadow_match_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len(retryable_exact_shadow_match_race_ids(rows))


def retryable_exact_shadow_match_race_ids(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    return sorted(
        str(row.get("race_id"))
        for row in rows
        if row.get("race_id")
        and row.get("recovery_action") == "validate_runner_set_then_alias_join"
        and row.get("alias_reconciliation_status") == "EXACT_SHADOW_ARTIFACT_MATCH_FOUND"
    )


def no_exact_shadow_match_race_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        str(row.get("race_id"))
        for row in rows
        if row.get("race_id")
        and row.get("alias_reconciliation_status") == "NO_EXACT_SHADOW_ARTIFACT_MATCH"
    )


def latest_shadow_run_dirs(evidence_root: Path, *, limit: int) -> list[Path]:
    if limit <= 0 or not evidence_root.exists():
        return []
    candidates = [
        path
        for path in evidence_root.glob("daily_race_ingest_shadow_*")
        if path.is_dir()
        and (
            (path / "stage2_shadow_predictions.jsonl").exists()
            or (path / "shadow_predictions.jsonl").exists()
        )
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[:limit]


def run_shadow_run_official_dry_run(
    *,
    db_path: Path,
    shadow_run_dir: Path,
    target_date: str,
    current_time: datetime,
    output_dir: Path,
    race_ids: Sequence[str],
    include_live_odds_backlog: bool = False,
    backlog_evidence_root: Path | None = None,
    backlog_limit: int = 0,
    backlog_shadow_run_limit: int = 0,
    backlog_lookback_days: int = 0,
) -> tuple[dict[str, Any], int]:
    candidates, skipped, source_report = shadow_run_candidates(
        shadow_run_dir=shadow_run_dir,
        target_date=target_date,
        current_time=current_time,
        race_ids=race_ids,
        output_dir=output_dir,
    )
    candidate_by_race_id = {candidate.race_id: candidate for candidate in candidates}
    backlog_report: dict[str, Any] = {
        "enabled": include_live_odds_backlog,
        "source": "source_backed_live_odds_without_official_results",
        "discovered_race_ids": [],
        "candidate_race_ids": [],
        "unresolved_race_ids": [],
        "unresolved_races": [],
        "unresolved_reason_counts": {},
        "shadow_run_reports": [],
        "backlog_limit": backlog_limit,
        "backlog_shadow_run_limit": backlog_shadow_run_limit,
        "backlog_lookback_days": backlog_lookback_days,
        "target_dates": live_odds_backlog_target_dates(
            target_date,
            lookback_days=backlog_lookback_days,
        ),
        "discovered_races": [],
    }
    if include_live_odds_backlog:
        backlog_entries = source_backed_live_odds_backlog_entries(
            db_path=db_path,
            target_date=target_date,
            limit=backlog_limit,
            lookback_days=backlog_lookback_days,
        )
        backlog_race_ids = [str(item.get("race_id")) for item in backlog_entries]
        backlog_race_dates = {
            str(item.get("race_id")): str(item.get("race_date") or target_date)
            for item in backlog_entries
        }
        backlog_report["discovered_race_ids"] = backlog_race_ids
        backlog_report["discovered_races"] = backlog_entries
        remaining = [
            race_id
            for race_id in backlog_race_ids
            if race_id not in candidate_by_race_id
        ]
        scanned_shadow_run_dirs = latest_shadow_run_dirs(
            backlog_evidence_root or DEFAULT_EVIDENCE_ROOT,
            limit=backlog_shadow_run_limit,
        )
        for backlog_dir in scanned_shadow_run_dirs:
            if not remaining:
                break
            remaining_dates = sorted(
                {backlog_race_dates.get(race_id, target_date) for race_id in remaining}
            )
            for race_date in remaining_dates:
                date_race_ids = [
                    race_id
                    for race_id in remaining
                    if backlog_race_dates.get(race_id, target_date) == race_date
                ]
                if not date_race_ids:
                    continue
                extra_candidates, extra_skipped, extra_report = shadow_run_candidates(
                    shadow_run_dir=backlog_dir,
                    target_date=race_date,
                    current_time=current_time,
                    race_ids=date_race_ids,
                    output_dir=output_dir,
                )
                extra_report["candidate_source"] = "source_backed_live_odds_backlog_shadow_run"
                extra_report["backlog_shadow_run_dir"] = relpath(backlog_dir)
                extra_report["backlog_target_date"] = race_date
                backlog_report["shadow_run_reports"].append(extra_report)
                for item in extra_skipped:
                    skipped.append(
                        {
                            **item,
                            "candidate_source": "source_backed_live_odds_backlog_shadow_run",
                            "backlog_shadow_run_dir": relpath(backlog_dir),
                            "backlog_target_date": race_date,
                        }
                    )
                for candidate in extra_candidates:
                    if candidate.race_id not in candidate_by_race_id:
                        candidate_by_race_id[candidate.race_id] = candidate
            remaining = [
                race_id
                for race_id in remaining
                if race_id not in candidate_by_race_id
            ]
        backlog_report["candidate_race_ids"] = sorted(
            race_id
            for race_id in backlog_race_ids
            if race_id in candidate_by_race_id
        )
        backlog_report["unresolved_race_ids"] = remaining
        unresolved_races = unresolved_live_odds_backlog_diagnostics(
            unresolved_race_ids=remaining,
            backlog_entries=backlog_entries,
            skipped=skipped,
            shadow_run_report_count=len(backlog_report["shadow_run_reports"]),
            shadow_run_dirs=scanned_shadow_run_dirs,
        )
        backlog_report["unresolved_races"] = unresolved_races
        backlog_report["unresolved_reason_counts"] = reason_counts(unresolved_races)
        backlog_report["unresolved_recovery_action_counts"] = field_counts(
            unresolved_races,
            "recovery_action",
        )
        backlog_report["unresolved_alias_status_counts"] = field_counts(
            unresolved_races,
            "alias_reconciliation_status",
        )
        backlog_report["retryable_exact_shadow_match_race_ids"] = (
            retryable_exact_shadow_match_race_ids(unresolved_races)
        )
        backlog_report["no_exact_shadow_match_race_ids"] = (
            no_exact_shadow_match_race_ids(unresolved_races)
        )
        backlog_report["retryable_exact_shadow_match_race_count"] = (
            len(backlog_report["retryable_exact_shadow_match_race_ids"])
        )
        backlog_report["no_exact_shadow_match_race_count"] = len(
            backlog_report["no_exact_shadow_match_race_ids"]
        )
    candidates = list(candidate_by_race_id.values())
    source_report["candidate_count"] = len(candidates)
    source_report["candidate_race_ids"] = sorted(candidate.race_id for candidate in candidates)
    source_report["live_odds_backlog"] = backlog_report
    write_json(output_dir / "shadow_run_candidate_source_report.json", source_report)

    ingested: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    flush_official_result_progress(
        output_dir,
        candidates=candidates,
        progress_rows=progress_rows,
        active_row=None,
    )
    if candidates:
        driver, By, browser_error = ingest.optional_browser_driver(headless=True)
        thedogs = ingest.TheDogsResultFetcher(
            driver,
            by=By,
            http_session=ingest._StatelessPublicHttpClient(),
        )
        sportsbet = ingest.SportsbetResultFetcher(driver, target_date, by=By) if driver else None
        if not db_path.exists():
            raise FileNotFoundError(f"db_path_not_found:{db_path}")
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            for index, candidate in enumerate(candidates, start=1):
                attempts: list[Any] = []
                active_row = {
                    "schema_version": "autonomous_official_result_capture_attempt_v1",
                    "candidate_index": index,
                    "candidate_count": len(candidates),
                    "race_id": candidate.race_id,
                    "race_date": candidate.race_date,
                    "venue": candidate.venue,
                    "race_number": candidate.race_number,
                    "status": "FETCH_IN_PROGRESS",
                    "started_at": datetime.now().astimezone().isoformat(),
                }
                flush_official_result_progress(
                    output_dir,
                    candidates=candidates,
                    progress_rows=progress_rows,
                    active_row=active_row,
                )
                try:
                    official = thedogs.fetch(candidate)
                    attempts.append(official)
                    selected = official
                    validation_error = ingest.result_validation_error(candidate, selected)
                    if validation_error and sportsbet is not None:
                        fallback = sportsbet.fetch(candidate)
                        attempts.append(fallback)
                        fallback_error = ingest.result_validation_error(candidate, fallback)
                        if fallback_error is None:
                            selected = fallback
                            validation_error = None
                        else:
                            validation_error = f"{validation_error}; fallback:{fallback_error}"
                except Exception as exc:
                    validation_error = f"fetch_exception:{type(exc).__name__}:{exc}"
                    selected = None
                if validation_error:
                    failed_row = {
                        "race_id": candidate.race_id,
                        **_candidate_participant_diagnostic(candidate),
                        "errors": [validation_error],
                        "attempted_sources": [
                            ingest._source_result_diagnostic(attempt)
                            for attempt in attempts
                        ],
                    }
                    failed.append(failed_row)
                    progress_rows.append(
                        {
                            **active_row,
                            "status": "FAILED_VALIDATION",
                            "completed_at": datetime.now().astimezone().isoformat(),
                            "errors": [validation_error],
                            "attempted_source_count": len(attempts),
                        }
                    )
                    flush_official_result_progress(
                        output_dir,
                        candidates=candidates,
                        progress_rows=progress_rows,
                        active_row=None,
                    )
                    continue
                ingested_row = ingest.write_result(
                    conn,
                    candidate,
                    selected,
                    attempts,
                    dry_run=True,
                )
                ingested.append(ingested_row)
                progress_rows.append(
                    {
                        **active_row,
                        "status": "INGESTED_DRY_RUN",
                        "completed_at": datetime.now().astimezone().isoformat(),
                        "attempted_source_count": len(attempts),
                        "source": ingested_row.get("source"),
                    }
                )
                flush_official_result_progress(
                    output_dir,
                    candidates=candidates,
                    progress_rows=progress_rows,
                    active_row=None,
                )
        finally:
            conn.close()
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass

        if browser_error:
            skipped.append(
                {
                    "race_id": "__browser__",
                    "reason": browser_error,
                }
            )

    status = "SUCCESS"
    if not candidates:
        status = "DATA_MISSING"
    if failed:
        status = "FAILED"
    report = {
        "schema_version": "official_result_ingest_report_v1",
        "generated_at": current_time.isoformat(),
        "status": status,
        "dry_run": True,
        "scope": {
            "db_path": str(db_path),
            "date": target_date,
            "shadow_run_dir": relpath(shadow_run_dir),
            "race_ids": sorted({str(race_id) for race_id in race_ids if race_id}),
            "candidate_source": "shadow_run_predictions",
            "live_odds_backlog_enabled": include_live_odds_backlog,
            "live_odds_backlog_lookback_days": backlog_lookback_days,
        },
        "candidate_count": len(candidates),
        "candidate_race_ids": sorted(candidate.race_id for candidate in candidates),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "ingested_count": len(ingested),
        "ingested": ingested,
        "failed_count": len(failed),
        "failed": failed,
        "label_write_blockers": [],
        "backup_path": None,
        "result_label_write_approval": {
            "approved": False,
            "status": "not_approved",
            "required_for": "official_result_label_writes",
        },
        "dry_run_report_gate": None,
        "clean_for_label_write": False,
        "shadow_run_candidate_source_report": relpath(
            output_dir / "shadow_run_candidate_source_report.json"
        ),
        "live_odds_backlog": backlog_report,
    }
    return report, 0


def _position_rows(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = item.get("positions")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _participant_rows(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = item.get("participants")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _candidate_participant_diagnostic(candidate: ingest.RaceCandidate) -> dict[str, Any]:
    participants: list[dict[str, Any]] = []
    participant_boxes: list[int] = []
    for participant in candidate.participants or []:
        if not isinstance(participant, Mapping):
            continue
        box_number = parse_int(participant.get("box_number"))
        row = {
            "box_number": box_number,
            "dog_name": participant.get("dog_name"),
        }
        participants.append(row)
        if box_number is not None:
            participant_boxes.append(box_number)
    return {
        "participant_source": candidate.participant_source,
        "participant_count": len(participants),
        "participant_boxes": sorted(set(participant_boxes)),
        "participants": participants,
    }


def _participant_boxes_from_item(item: Mapping[str, Any]) -> list[int]:
    boxes: list[int] = []
    raw_boxes = item.get("participant_boxes")
    if isinstance(raw_boxes, list):
        for value in raw_boxes:
            box_number = parse_int(value)
            if box_number is not None:
                boxes.append(box_number)
    for participant in _participant_rows(item):
        box_number = parse_int(participant.get("box_number"))
        if box_number is not None:
            boxes.append(box_number)
    return sorted(set(boxes))


def build_artifact_rows(
    ingest_report: Mapping[str, Any],
    *,
    generated_at: datetime,
) -> dict[str, Any]:
    race_rows: list[dict[str, Any]] = []
    runner_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    scope = ingest_report.get("scope") if isinstance(ingest_report.get("scope"), Mapping) else {}

    for item in ingest_report.get("ingested") or []:
        if not isinstance(item, Mapping):
            continue
        official = item.get("source") == OFFICIAL_SOURCE and item.get("status") == RESULTED_STATUS
        positions = _position_rows(item)
        if not official or not positions:
            quarantine_rows.append(
                {
                    "race_id": item.get("race_id"),
                    "reason": "non_official_or_missing_positions",
                    "source": item.get("source"),
                    "status": item.get("status"),
                    "item": dict(item),
                }
            )
            continue
        race_row = {
            "schema_version": "autonomous_official_result_race_v1",
            "captured_at": generated_at.isoformat(),
            "race_id": item.get("race_id"),
            "venue": item.get("venue"),
            "race_number": item.get("race_number"),
            "race_date": item.get("race_date"),
            "race_time": item.get("race_time"),
            "start_datetime": item.get("start_datetime"),
            "source": item.get("source"),
            "source_url": item.get("source_url"),
            "status": item.get("status"),
            "winner_name": item.get("winner_name"),
            "winner_box": item.get("winner_box"),
            "box_order": list(item.get("box_order") or []),
            "participant_source": item.get("participant_source"),
            "position_count": len(positions),
            "participant_count": len(_participant_rows(item)),
            "scope": dict(scope),
        }
        race_rows.append(race_row)
        for position in positions:
            runner_rows.append(
                {
                    "schema_version": "autonomous_official_result_runner_v1",
                    "captured_at": generated_at.isoformat(),
                    "race_id": item.get("race_id"),
                    "venue": item.get("venue"),
                    "race_number": item.get("race_number"),
                    "race_date": item.get("race_date"),
                    "source": item.get("source"),
                    "source_url": item.get("source_url"),
                    "box_number": position.get("box_number"),
                    "dog_name": position.get("dog_name"),
                    "finish_position": position.get("finish_position"),
                    "is_winner": position.get("finish_position") == 1,
                }
            )

    for item in ingest_report.get("failed") or []:
        if isinstance(item, Mapping):
            quarantine_rows.append(
                {
                    "race_id": item.get("race_id"),
                    "reason": "ingest_failed_or_unsafe_match",
                    "item": dict(item),
                }
            )
    for item in ingest_report.get("skipped") or []:
        if isinstance(item, Mapping):
            quarantine_rows.append(
                {
                    "race_id": item.get("race_id"),
                    "reason": item.get("reason") or "skipped_before_fetch",
                    "item": dict(item),
                }
            )

    return {
        "race_rows": race_rows,
        "runner_rows": runner_rows,
        "quarantine_rows": quarantine_rows,
    }


def summarize_quarantine_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    attempted_source_counts: Counter[str] = Counter()
    result_boxes_not_in_participants_counts: Counter[str] = Counter()
    race_ids: set[str] = set()
    samples: list[dict[str, Any]] = []
    runner_set_mismatch_samples: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        race_id = str(row.get("race_id") or "").strip()
        if race_id and race_id != "__browser__":
            race_ids.add(race_id)
        reason = str(row.get("reason") or "unknown").strip() or "unknown"
        reason_counts[reason] += 1
        item = row.get("item") if isinstance(row.get("item"), Mapping) else {}
        errors = [
            str(error)
            for error in item.get("errors") or []
            if str(error or "").strip()
        ]
        error_counts.update(errors)
        participant_boxes = _participant_boxes_from_item(item)
        participant_box_set = set(participant_boxes)
        result_boxes_not_in_participants: list[int] = []
        for error in errors:
            for match in re.finditer(r"result_boxes_not_in_participants:([0-9,]+)", error):
                for value in match.group(1).split(","):
                    if value.strip().isdigit():
                        box = int(value.strip())
                        result_boxes_not_in_participants.append(box)
                        result_boxes_not_in_participants_counts[str(box)] += 1
        attempted_sources: list[dict[str, Any]] = []
        for source in item.get("attempted_sources") or []:
            if not isinstance(source, Mapping):
                continue
            source_name = str(source.get("source") or "unknown").strip() or "unknown"
            attempted_source_counts[source_name] += 1
            attempted_sources.append(
                {
                    "source": source_name,
                    "status": source.get("status"),
                    "source_url": source.get("source_url"),
                    "raw_order": list(source.get("raw_order") or []),
                    "dog_names_by_box": dict(source.get("dog_names_by_box") or {}),
                    "terminal_statuses": list(source.get("terminal_statuses") or []),
                }
            )
        if result_boxes_not_in_participants and len(runner_set_mismatch_samples) < 25:
            attempted_source_box_sets = []
            for source in attempted_sources:
                terminal_statuses = [
                    terminal
                    for terminal in source.get("terminal_statuses") or []
                    if isinstance(terminal, Mapping)
                ]
                attempted_source_box_sets.append(
                    {
                        "source": source.get("source"),
                        "status": source.get("status"),
                        "source_url": source.get("source_url"),
                        "result_boxes": source.get("raw_order") or [],
                        "dog_names_by_box": dict(source.get("dog_names_by_box") or {}),
                        "terminal_status_boxes": [
                            terminal.get("box_number") for terminal in terminal_statuses
                        ],
                        "terminal_statuses": terminal_statuses,
                    }
                )
            runner_set_mismatch_samples.append(
                {
                    "race_id": race_id or row.get("race_id"),
                    "reason": reason,
                    "errors": errors,
                    "participant_source": item.get("participant_source"),
                    "participant_count": item.get("participant_count"),
                    "participant_boxes": participant_boxes,
                    "participants": _participant_rows(item),
                    "result_boxes_not_in_participants": sorted(
                        set(result_boxes_not_in_participants)
                    ),
                    "result_boxes_in_participants": sorted(
                        {
                            result_box
                            for source in attempted_source_box_sets
                            for box in source.get("result_boxes") or []
                            for result_box in [parse_int(box)]
                            if result_box in participant_box_set
                        }
                    ),
                    "attempted_source_box_sets": attempted_source_box_sets,
                }
            )
        if len(samples) < 25:
            samples.append(
                {
                    "race_id": race_id or row.get("race_id"),
                    "reason": reason,
                    "errors": errors,
                    "attempted_sources": attempted_sources,
                }
            )

    return {
        "race_ids": sorted(race_ids),
        "reason_counts": dict(sorted(reason_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "attempted_source_counts": dict(sorted(attempted_source_counts.items())),
        "result_boxes_not_in_participants_counts": dict(
            sorted(result_boxes_not_in_participants_counts.items())
        ),
        "samples": samples,
        "runner_set_mismatch_samples": runner_set_mismatch_samples,
    }


def build_capture_report(
    *,
    generated_at: datetime,
    dry_run_command: Sequence[str],
    dry_run_returncode: int,
    ingest_report: Mapping[str, Any],
    artifact_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    evidence_db_ingest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    official_race_count = len(artifact_rows.get("race_rows") or [])
    official_runner_count = len(artifact_rows.get("runner_rows") or [])
    quarantine_rows = [
        dict(row)
        for row in artifact_rows.get("quarantine_rows") or []
        if isinstance(row, Mapping)
    ]
    quarantine_count = len(quarantine_rows)
    quarantine_summary = summarize_quarantine_rows(quarantine_rows)
    backlog = (
        ingest_report.get("live_odds_backlog")
        if isinstance(ingest_report.get("live_odds_backlog"), Mapping)
        else {}
    )
    skipped_rows = [
        dict(row)
        for row in ingest_report.get("skipped") or []
        if isinstance(row, Mapping)
    ]
    skipped_reason_counts = reason_counts(skipped_rows)
    awaiting_jump_rows = sorted(
        [
            row
            for row in skipped_rows
            if str(row.get("reason") or "").startswith("race_not_jumped:")
        ],
        key=lambda row: (
            str(row.get("jump_datetime") or ""),
            str(row.get("race_id") or ""),
        ),
    )
    awaiting_jump_races = [
        {
            "race_id": row.get("race_id"),
            "race_time": row.get("race_time"),
            "jump_datetime": row.get("jump_datetime"),
            "reason": row.get("reason"),
        }
        for row in awaiting_jump_rows
    ]
    if dry_run_returncode != 0:
        final_status = "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_DRY_RUN_FAILED"
    elif official_race_count:
        final_status = "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    elif awaiting_jump_races and len(awaiting_jump_races) == quarantine_count:
        final_status = "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_AWAITING_JUMP"
    elif quarantine_count:
        final_status = "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_QUARANTINED"
    else:
        final_status = "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_NO_RESULTS"
    evidence_db_ingest = dict(evidence_db_ingest or evidence_db_ingest_not_executed())
    no_write_guarantees = dict(NO_WRITE_GUARANTEES)
    no_write_guarantees["db_write"] = bool(evidence_db_ingest.get("db_write_performed"))
    no_write_guarantees["label_write"] = False
    no_write_guarantees["canonical_result_label_write"] = False
    return {
        "schema_version": "autonomous_official_result_capture_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "dry_run_command": list(dry_run_command),
        "dry_run_returncode": dry_run_returncode,
        "ingest_report_status": ingest_report.get("status"),
        "candidate_count": int(ingest_report.get("candidate_count") or 0),
        "ingested_count": int(ingest_report.get("ingested_count") or 0),
        "failed_count": int(ingest_report.get("failed_count") or 0),
        "skipped_count": int(ingest_report.get("skipped_count") or 0),
        "skipped_reason_counts": skipped_reason_counts,
        "awaiting_jump_race_count": len(awaiting_jump_races),
        "awaiting_jump_race_ids": [
            str(row.get("race_id") or "") for row in awaiting_jump_races
        ],
        "awaiting_jump_next_recheck_after_local": (
            awaiting_jump_races[0].get("jump_datetime")
            if awaiting_jump_races
            else None
        ),
        "awaiting_jump_races": awaiting_jump_races[:25],
        "official_result_race_rows": official_race_count,
        "official_result_runner_rows": official_runner_count,
        "quarantine_rows": quarantine_count,
        "quarantined_race_ids": quarantine_summary["race_ids"],
        "quarantine_reason_counts": quarantine_summary["reason_counts"],
        "quarantine_error_counts": quarantine_summary["error_counts"],
        "quarantine_attempted_source_counts": quarantine_summary[
            "attempted_source_counts"
        ],
        "quarantine_result_boxes_not_in_participants_counts": quarantine_summary[
            "result_boxes_not_in_participants_counts"
        ],
        "quarantine_samples": quarantine_summary["samples"],
        "quarantine_runner_set_mismatch_samples": quarantine_summary[
            "runner_set_mismatch_samples"
        ],
        "live_odds_backlog_enabled": bool(backlog.get("enabled")),
        "live_odds_backlog_lookback_days": int(backlog.get("backlog_lookback_days") or 0),
        "live_odds_backlog_target_dates": list(backlog.get("target_dates") or []),
        "live_odds_backlog_discovered_race_count": len(
            backlog.get("discovered_race_ids") or []
        ),
        "live_odds_backlog_discovered_race_ids": list(
            backlog.get("discovered_race_ids") or []
        ),
        "live_odds_backlog_candidate_race_count": len(
            backlog.get("candidate_race_ids") or []
        ),
        "live_odds_backlog_candidate_race_ids": list(
            backlog.get("candidate_race_ids") or []
        ),
        "live_odds_backlog_unresolved_race_count": len(
            backlog.get("unresolved_race_ids") or []
        ),
        "live_odds_backlog_unresolved_race_ids": list(
            backlog.get("unresolved_race_ids") or []
        ),
        "live_odds_backlog_unresolved_races": list(
            backlog.get("unresolved_races") or []
        ),
        "live_odds_backlog_unresolved_reason_counts": dict(
            backlog.get("unresolved_reason_counts") or {}
        ),
        "live_odds_backlog_unresolved_recovery_action_counts": dict(
            backlog.get("unresolved_recovery_action_counts") or {}
        ),
        "live_odds_backlog_unresolved_alias_status_counts": dict(
            backlog.get("unresolved_alias_status_counts") or {}
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_ids": list(
            backlog.get("retryable_exact_shadow_match_race_ids") or []
        ),
        "live_odds_backlog_no_exact_shadow_match_race_ids": list(
            backlog.get("no_exact_shadow_match_race_ids") or []
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_count": int(
            backlog.get("retryable_exact_shadow_match_race_count") or 0
        ),
        "live_odds_backlog_no_exact_shadow_match_race_count": int(
            backlog.get("no_exact_shadow_match_race_count") or 0
        ),
        "live_odds_backlog_recovery_queue_path": relpath(
            output_dir / "live_odds_backlog_recovery_queue.json"
        ),
        "live_odds_backlog_runner_set_validation_path": relpath(
            output_dir / "live_odds_backlog_runner_set_validation.json"
        ),
        "live_odds_backlog_join_eligibility_packet_path": relpath(
            output_dir / "live_odds_backlog_join_eligibility_packet.json"
        ),
        "shadow_run_candidate_source_report": ingest_report.get(
            "shadow_run_candidate_source_report"
        ),
        "output_dir": relpath(output_dir),
        "race_rows_path": relpath(output_dir / "official_result_races.jsonl"),
        "runner_rows_path": relpath(output_dir / "official_result_runners.jsonl"),
        "quarantine_path": relpath(output_dir / "official_result_quarantine.jsonl"),
        "official_result_evidence_db_ingest": evidence_db_ingest,
        "no_write_guarantees": no_write_guarantees,
    }


def build_live_odds_backlog_recovery_queue(
    *,
    capture_report: Mapping[str, Any],
) -> dict[str, Any]:
    unresolved_races = [
        dict(item)
        for item in capture_report.get("live_odds_backlog_unresolved_races") or []
        if isinstance(item, Mapping)
    ]
    def race_id(row: Mapping[str, Any]) -> str:
        return str(row.get("race_id") or "")

    retryable_source_ids = capture_report.get(
        "live_odds_backlog_retryable_exact_shadow_match_race_ids"
    )
    if retryable_source_ids:
        retryable_ids = {str(race_id) for race_id in retryable_source_ids}
    else:
        retryable_ids = {
            race_id(row)
            for row in unresolved_races
            if row.get("recovery_action") == "validate_runner_set_then_alias_join"
            and row.get("alias_reconciliation_status")
            == "EXACT_SHADOW_ARTIFACT_MATCH_FOUND"
        }
    no_exact_source_ids = capture_report.get("live_odds_backlog_no_exact_shadow_match_race_ids")
    if no_exact_source_ids:
        no_exact_ids = {str(race_id) for race_id in no_exact_source_ids}
    else:
        no_exact_ids = {
            race_id(row)
            for row in unresolved_races
            if row.get("alias_reconciliation_status") == "NO_EXACT_SHADOW_ARTIFACT_MATCH"
        }

    def matching(ids: set[str]) -> list[dict[str, Any]]:
        return sorted(
            [dict(row) for row in unresolved_races if race_id(row) in ids],
            key=lambda row: race_id(row),
        )

    def parse_dt(value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        try:
            return parse_current_time(str(value))
        except Exception:
            return None

    generated_at_dt = parse_dt(capture_report.get("generated_at"))

    def awaiting_official_recheck_plan(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        stale_enough_count = 0
        for row in sorted(rows, key=race_id):
            latest_capture = str(row.get("latest_capture") or "").strip()
            latest_capture_dt = parse_dt(latest_capture)
            age_minutes = None
            if generated_at_dt is not None and latest_capture_dt is not None:
                age_minutes = round(
                    (generated_at_dt - latest_capture_dt).total_seconds() / 60.0,
                    3,
                )
            recheck_ready = age_minutes is None or age_minutes >= 5.0
            if recheck_ready:
                stale_enough_count += 1
            parsed_identity = (
                row.get("parsed_identity")
                if isinstance(row.get("parsed_identity"), Mapping)
                else {}
            )
            summaries.append(
                {
                    "race_id": race_id(row),
                    "race_date": row.get("race_date"),
                    "venue": row.get("live_odds_venue")
                    or parsed_identity.get("venue"),
                    "race_number": row.get("live_odds_race_number")
                    or parsed_identity.get("race_number"),
                    "latest_live_odds_capture": latest_capture or None,
                    "minutes_since_latest_live_odds_capture": age_minutes,
                    "source_url": row.get("live_odds_source_url"),
                    "candidate_shadow_race_id_match_count": int(
                        row.get("candidate_shadow_race_id_match_count") or 0
                    ),
                    "official_result_recheck_ready": recheck_ready,
                }
            )
        return {
            "schema_version": "awaiting_official_result_recheck_plan_v1",
            "diagnostic_only": True,
            "join_acceptance_changed": False,
            "db_write_performed": False,
            "authorized_action": "diagnostic_recheck_official_result_evidence_only",
            "minimum_minutes_since_latest_live_odds_capture_for_recheck": 5.0,
            "race_count": len(summaries),
            "recheck_ready_race_count": stale_enough_count,
            "race_ids": [str(row.get("race_id") or "") for row in summaries],
            "races": summaries,
        }

    awaiting_official_result_ids = {
        race_id(row)
        for row in unresolved_races
        if row.get("alias_reconciliation_status") == "EXACT_SHADOW_ARTIFACT_MATCH_FOUND"
        and row.get("source") == "source_backed_live_odds_without_official_results"
    }
    awaiting_official_result_rows = matching(awaiting_official_result_ids)

    by_recovery_action: dict[str, dict[str, Any]] = {}
    for row in sorted(unresolved_races, key=race_id):
        action = str(row.get("recovery_action") or "missing_recovery_action")
        bucket = by_recovery_action.setdefault(
            action,
            {
                "race_count": 0,
                "race_ids": [],
                "races": [],
            },
        )
        bucket["race_count"] += 1
        bucket["race_ids"].append(race_id(row))
        bucket["races"].append(dict(row))

    flat_items = [
        {
            "race_id": race_id(row),
            "race_date": row.get("race_date"),
            "venue": row.get("live_odds_venue"),
            "race_number": row.get("live_odds_race_number"),
            "canonical_live_odds_race_id": row.get("canonical_live_odds_race_id"),
            "latest_capture": row.get("latest_capture"),
            "live_odds_row_count": row.get("live_odds_row_count"),
            "live_odds_box_count": row.get("live_odds_box_count"),
            "alias_reconciliation_status": row.get("alias_reconciliation_status"),
            "candidate_shadow_race_id_match_count": row.get(
                "candidate_shadow_race_id_match_count"
            ),
            "reason": row.get("reason"),
            "recovery_action": row.get("recovery_action"),
            "authorized_action": (
                "diagnostic_recheck_official_result_evidence_only"
                if race_id(row) in awaiting_official_result_ids
                else "diagnostic_review_only"
            ),
            "db_write_performed": False,
            "join_acceptance_changed": False,
        }
        for row in sorted(unresolved_races, key=race_id)
    ]

    return {
        "schema_version": "live_odds_backlog_recovery_queue_v1",
        "generated_at": capture_report.get("generated_at"),
        "source_capture_report": relpath(
            Path(str(capture_report.get("output_dir") or ""))
            / "autonomous_official_result_capture_report.json"
        ),
        "diagnostic_only": True,
        "join_acceptance_changed": False,
        "db_write_performed": False,
        "promotion_or_registry_mutation": False,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "live_odds_backlog_enabled": bool(
            capture_report.get("live_odds_backlog_enabled")
        ),
        "unresolved_race_count": int(
            capture_report.get("live_odds_backlog_unresolved_race_count") or 0
        ),
        "unresolved_reason_counts": dict(
            capture_report.get("live_odds_backlog_unresolved_reason_counts") or {}
        ),
        "unresolved_recovery_action_counts": dict(
            capture_report.get("live_odds_backlog_unresolved_recovery_action_counts")
            or {}
        ),
        "unresolved_alias_status_counts": dict(
            capture_report.get("live_odds_backlog_unresolved_alias_status_counts")
            or {}
        ),
        "queue_count": len(flat_items),
        "items": flat_items,
        "queues": {
            "retryable_exact_shadow_match": {
                "description": (
                    "Exact shadow artifact match exists; still requires runner-set "
                    "validation before any future alias join."
                ),
                "race_count": len(retryable_ids),
                "race_ids": sorted(retryable_ids),
                "races": matching(retryable_ids),
                "authorized_action": "diagnostic_review_only",
            },
            "awaiting_official_result_evidence": {
                "description": (
                    "Source-backed live odds and an exact shadow artifact match exist, "
                    "but official result evidence was not available at capture time; "
                    "recheck the official source before any future runner-set join."
                ),
                "race_count": len(awaiting_official_result_ids),
                "race_ids": sorted(awaiting_official_result_ids),
                "races": awaiting_official_result_rows,
                "recheck_plan": awaiting_official_recheck_plan(
                    awaiting_official_result_rows
                ),
                "authorized_action": "diagnostic_recheck_official_result_evidence_only",
            },
            "no_exact_shadow_match": {
                "description": (
                    "No exact shadow artifact match was found; inspect shadow run "
                    "coverage or recover source-backed predictions."
                ),
                "race_count": len(no_exact_ids),
                "race_ids": sorted(no_exact_ids),
                "races": matching(no_exact_ids),
                "authorized_action": "diagnostic_review_only",
            },
            "by_recovery_action": by_recovery_action,
        },
    }


def runner_identity_key(row: Mapping[str, Any]) -> tuple[int | None, str]:
    return (
        parse_int(row.get("box_number") if row.get("box_number") not in (None, "") else row.get("box")),
        normalise_runner_name(row.get("dog_name")),
    )


def sorted_runner_key_rows(keys: set[tuple[Any, ...]]) -> list[list[Any]]:
    def sort_key(item: tuple[Any, ...]) -> tuple[int, int, str]:
        box = parse_int(item[0] if item else None)
        name = str(item[1] if len(item) > 1 else "")
        return (1 if box is None else 0, box if box is not None else 9999, name)

    return [list(key) for key in sorted(keys, key=sort_key)]


def live_odds_runner_rows_for_capture(
    *,
    db_path: Path,
    race_id: str,
    capture_timestamp: str | None,
) -> list[dict[str, Any]]:
    if not race_id or not db_path.exists():
        return []
    try:
        with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(live_odds)").fetchall()
            }
            timestamp_expr = (
                "COALESCE(capture_timestamp, timestamp, '')"
                if "capture_timestamp" in columns
                else "COALESCE(timestamp, '')"
            )
            if capture_timestamp:
                rows = conn.execute(
                    f"""
                    SELECT
                        dog_name,
                        box_number,
                        odds_decimal,
                        {timestamp_expr} AS captured_at,
                        source_url,
                        sportsbet_box_source,
                        sportsbet_raw_runner_text
                    FROM live_odds
                    WHERE race_id = ?
                      AND {timestamp_expr} = ?
                    ORDER BY CAST(box_number AS INTEGER), dog_name
                    """,
                    [race_id, capture_timestamp],
                ).fetchall()
            else:
                rows = []
            if not rows:
                latest = conn.execute(
                    f"""
                    SELECT MAX({timestamp_expr})
                    FROM live_odds
                    WHERE race_id = ?
                    """,
                    [race_id],
                ).fetchone()
                latest_timestamp = latest[0] if latest else None
                rows = conn.execute(
                    f"""
                    SELECT
                        dog_name,
                        box_number,
                        odds_decimal,
                        {timestamp_expr} AS captured_at,
                        source_url,
                        sportsbet_box_source,
                        sportsbet_raw_runner_text
                    FROM live_odds
                    WHERE race_id = ?
                      AND {timestamp_expr} = ?
                    ORDER BY CAST(box_number AS INTEGER), dog_name
                    """,
                    [race_id, latest_timestamp],
                ).fetchall() if latest_timestamp else []
    except sqlite3.Error:
        return []

    runner_rows: list[dict[str, Any]] = []
    for row in rows:
        dog_name = str(row[0] or "").strip()
        box_number = parse_int(row[1])
        if not dog_name or box_number is None:
            continue
        item = {
            "dog_name": dog_name,
            "box_number": box_number,
            "odds_decimal": row[2],
            "captured_at": row[3],
            "source_url": row[4],
            "sportsbet_box_source": row[5],
            "sportsbet_raw_runner_text": row[6],
        }
        item["runner_key"] = list(runner_identity_key(item))
        runner_rows.append(item)
    return runner_rows


def shadow_runner_rows_for_match(match: Mapping[str, Any]) -> list[dict[str, Any]]:
    shadow_run_dir_text = str(match.get("shadow_run_dir") or "").strip()
    race_id = str(match.get("race_id") or "").strip()
    if not shadow_run_dir_text or not race_id:
        return []
    shadow_run_dir = Path(shadow_run_dir_text)
    if not shadow_run_dir.is_absolute():
        shadow_run_dir = ROOT / shadow_run_dir
    feature_rows = feature_rows_by_race_id(shadow_run_dir).get(race_id) or []
    prediction_rows = [
        row
        for row in prediction_rows_from_shadow_run(shadow_run_dir)
        if str(row.get("race_id") or "").strip() == race_id
    ]
    source_rows = feature_rows or prediction_rows
    runner_rows: list[dict[str, Any]] = []
    for row in source_rows:
        dog_name = str(row.get("dog_name") or "").strip()
        box_number = parse_int(
            row.get("box_number") if row.get("box_number") not in (None, "") else row.get("box")
        )
        if not dog_name or box_number is None:
            continue
        item = {
            "dog_name": dog_name,
            "box_number": box_number,
            "source": "shadow_feature_rows" if feature_rows else "shadow_predictions",
        }
        item["runner_key"] = list(runner_identity_key(item))
        runner_rows.append(item)
    return runner_rows


def build_live_odds_backlog_runner_set_validation(
    *,
    recovery_queue: Mapping[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    retryable_queue = (
        (recovery_queue.get("queues") or {}).get("retryable_exact_shadow_match")
        if isinstance(recovery_queue.get("queues"), Mapping)
        else {}
    )
    retryable_races = (
        retryable_queue.get("races") if isinstance(retryable_queue, Mapping) else []
    ) or []
    validations: list[dict[str, Any]] = []
    exact_match_count = 0
    blocked_count = 0
    for race in retryable_races:
        if not isinstance(race, Mapping):
            continue
        race_id = str(race.get("race_id") or "").strip()
        live_rows = live_odds_runner_rows_for_capture(
            db_path=db_path,
            race_id=race_id,
            capture_timestamp=str(race.get("latest_capture") or "").strip() or None,
        )
        live_keys = {tuple(row.get("runner_key") or []) for row in live_rows}
        match_validations: list[dict[str, Any]] = []
        for match in race.get("candidate_shadow_race_id_matches") or []:
            if not isinstance(match, Mapping):
                continue
            shadow_rows = shadow_runner_rows_for_match(match)
            shadow_keys = {tuple(row.get("runner_key") or []) for row in shadow_rows}
            missing_from_shadow = sorted_runner_key_rows(live_keys - shadow_keys)
            missing_from_live_odds = sorted_runner_key_rows(shadow_keys - live_keys)
            exact_runner_set_match = bool(live_keys) and live_keys == shadow_keys
            match_validations.append(
                {
                    "shadow_run_dir": match.get("shadow_run_dir"),
                    "shadow_race_id": match.get("race_id"),
                    "artifact_sources": list(match.get("artifact_sources") or []),
                    "live_odds_runner_count": len(live_keys),
                    "shadow_runner_count": len(shadow_keys),
                    "exact_runner_set_match": exact_runner_set_match,
                    "missing_from_shadow": missing_from_shadow,
                    "missing_from_live_odds": missing_from_live_odds,
                }
            )
        has_exact_match = any(
            item.get("exact_runner_set_match") for item in match_validations
        )
        if has_exact_match:
            exact_match_count += 1
            status = "RUNNER_SET_EXACT_MATCH_DIAGNOSTIC_ONLY"
        else:
            blocked_count += 1
            status = "RUNNER_SET_VALIDATION_BLOCKED"
        validations.append(
            {
                "race_id": race_id,
                "canonical_live_odds_race_id": race.get("canonical_live_odds_race_id"),
                "latest_capture": race.get("latest_capture"),
                "live_odds_source_url": race.get("live_odds_source_url"),
                "validation_status": status,
                "join_authorized": False,
                "db_write_performed": False,
                "live_odds_runners": live_rows,
                "match_validations": match_validations,
            }
        )
    return {
        "schema_version": "live_odds_backlog_runner_set_validation_v1",
        "generated_at": recovery_queue.get("generated_at"),
        "source_recovery_queue": recovery_queue.get("source_capture_report"),
        "diagnostic_only": True,
        "join_authorized": False,
        "db_write_performed": False,
        "production_promotion": False,
        "retryable_race_count": len(validations),
        "exact_runner_set_match_race_count": exact_match_count,
        "blocked_race_count": blocked_count,
        "validations": validations,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def is_sportsbet_url(value: Any) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
    except Exception:
        return False
    host = parsed.netloc.lower()
    return parsed.scheme in {"http", "https"} and (
        host == "sportsbet.com.au" or host.endswith(".sportsbet.com.au")
    )


def parse_datetime_or_none(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return parse_current_time(text)
    except Exception:
        return None


def official_result_evidence_rows_for_race(
    *,
    db_path: Path,
    race_id: str,
) -> dict[str, Any]:
    empty = {
        "race_rows": [],
        "runner_rows": [],
    }
    if not race_id or not db_path.exists():
        return empty
    try:
        with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if (
                OFFICIAL_RESULT_EVIDENCE_RACES_TABLE not in tables
                or OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE not in tables
            ):
                return empty
            race_rows = [
                {
                    "race_id": row[0],
                    "race_date": row[1],
                    "venue": row[2],
                    "race_number": row[3],
                    "start_datetime": row[4],
                    "source": row[5],
                    "source_url": row[6],
                    "status": row[7],
                    "winner_name": row[8],
                    "winner_box": row[9],
                    "position_count": row[10],
                    "participant_count": row[11],
                    "captured_at": row[12],
                    "source_artifact_dir": row[13],
                }
                for row in conn.execute(
                    f"""
                    SELECT
                        race_id,
                        race_date,
                        venue,
                        race_number,
                        start_datetime,
                        source,
                        source_url,
                        status,
                        winner_name,
                        winner_box,
                        position_count,
                        participant_count,
                        captured_at,
                        source_artifact_dir
                    FROM {OFFICIAL_RESULT_EVIDENCE_RACES_TABLE}
                    WHERE race_id = ?
                    ORDER BY inserted_at DESC, id DESC
                    """,
                    [race_id],
                ).fetchall()
            ]
            runner_rows = [
                {
                    "race_id": row[0],
                    "race_date": row[1],
                    "venue": row[2],
                    "race_number": row[3],
                    "source": row[4],
                    "source_url": row[5],
                    "box_number": row[6],
                    "dog_name": row[7],
                    "finish_position": row[8],
                    "is_winner": bool(row[9]),
                    "captured_at": row[10],
                    "source_artifact_dir": row[11],
                    "runner_key": list(
                        runner_identity_key(
                            {"box_number": row[6], "dog_name": row[7]}
                        )
                    ),
                }
                for row in conn.execute(
                    f"""
                    SELECT
                        race_id,
                        race_date,
                        venue,
                        race_number,
                        source,
                        source_url,
                        box_number,
                        dog_name,
                        finish_position,
                        is_winner,
                        captured_at,
                        source_artifact_dir
                    FROM {OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE}
                    WHERE race_id = ?
                    ORDER BY CAST(box_number AS INTEGER), dog_name
                    """,
                    [race_id],
                ).fetchall()
            ]
            return {
                "race_rows": race_rows,
                "runner_rows": runner_rows,
            }
    except sqlite3.Error:
        return empty


def build_live_odds_backlog_join_eligibility_packet(
    *,
    runner_set_validation: Mapping[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    validations = [
        dict(row)
        for row in runner_set_validation.get("validations") or []
        if isinstance(row, Mapping)
    ]
    rows: list[dict[str, Any]] = []
    eligible_count = 0
    blocked_count = 0
    awaiting_official_result_evidence_count = 0
    awaiting_official_result_evidence_race_ids: list[str] = []
    blocker_counts: dict[str, int] = defaultdict(int)
    generated_at_dt = parse_datetime_or_none(runner_set_validation.get("generated_at"))
    awaiting_official_result_recheck_rows: list[dict[str, Any]] = []
    for validation in validations:
        race_id = str(validation.get("race_id") or "").strip()
        live_rows = [
            dict(row)
            for row in validation.get("live_odds_runners") or []
            if isinstance(row, Mapping)
        ]
        exact_shadow_matches = [
            dict(row)
            for row in validation.get("match_validations") or []
            if isinstance(row, Mapping) and row.get("exact_runner_set_match")
        ]
        official_rows = official_result_evidence_rows_for_race(
            db_path=db_path,
            race_id=race_id,
        )
        official_races = official_rows["race_rows"]
        official_runners = official_rows["runner_rows"]
        official_race = official_races[0] if official_races else {}

        live_keys = {tuple(row.get("runner_key") or []) for row in live_rows}
        official_keys = {tuple(row.get("runner_key") or []) for row in official_runners}
        source_urls = {
            str(row.get("source_url") or "").strip()
            for row in live_rows
            if str(row.get("source_url") or "").strip()
        }
        captured_at_values = {
            str(row.get("captured_at") or "").strip()
            for row in live_rows
            if str(row.get("captured_at") or "").strip()
        }
        latest_capture = str(validation.get("latest_capture") or "").strip()
        source_url = str(validation.get("live_odds_source_url") or "").strip()
        capture_dt = parse_datetime_or_none(latest_capture)
        official_start_dt = parse_datetime_or_none(official_race.get("start_datetime"))

        gates = {
            "race_id_present": bool(race_id),
            "canonical_live_odds_race_id_present": bool(
                validation.get("canonical_live_odds_race_id")
            ),
            "exact_shadow_runner_set_match": bool(exact_shadow_matches),
            "live_odds_runner_rows_present": bool(live_rows),
            "live_odds_source_url_valid": bool(source_url)
            and is_sportsbet_url(source_url),
            "live_odds_runner_source_urls_consistent": bool(source_urls)
            and len(source_urls) == 1
            and (not source_url or source_url in source_urls),
            "live_odds_capture_timestamp_present": bool(latest_capture),
            "live_odds_runner_timestamps_consistent": bool(captured_at_values)
            and len(captured_at_values) == 1
            and (not latest_capture or latest_capture in captured_at_values),
            "dog_level_win_odds_present": bool(live_rows)
            and all((parse_float(row.get("odds_decimal")) or 0.0) > 0 for row in live_rows),
            "sportsbet_runner_source_identity_present": bool(live_rows)
            and all(str(row.get("sportsbet_box_source") or "").strip() for row in live_rows),
            "official_result_race_row_present": bool(official_races),
            "official_result_runner_rows_present": bool(official_runners),
            "official_result_source_valid": bool(official_race)
            and official_race.get("source") == OFFICIAL_SOURCE
            and official_race.get("status") == RESULTED_STATUS
            and is_thedogs_official_url(official_race.get("source_url")),
            "official_result_runner_set_exact_live_odds_match": bool(live_keys)
            and live_keys == official_keys,
            "prejump_timing_verified": bool(capture_dt and official_start_dt)
            and capture_dt < official_start_dt,
        }
        identity = parse_race_identity(race_id)
        if official_race:
            gates["official_result_race_identity_exact"] = (
                identity.get("race_date") == str(official_race.get("race_date") or "")[:10]
                and identity.get("race_number") == parse_int(official_race.get("race_number"))
                and (
                    not identity.get("venue")
                    or not official_race.get("venue")
                    or normalize_venue(identity.get("venue"))
                    == normalize_venue(official_race.get("venue"))
                )
            )
        else:
            gates["official_result_race_identity_exact"] = False

        blockers = [gate for gate, passed in gates.items() if not passed]
        for blocker in blockers:
            blocker_counts[blocker] += 1
        if blockers:
            blocked_count += 1
            status = "JOIN_ELIGIBILITY_BLOCKED"
        else:
            eligible_count += 1
            status = "JOIN_ELIGIBLE_REPORT_ONLY"
        awaiting_official_result_evidence = (
            gates["race_id_present"]
            and gates["canonical_live_odds_race_id_present"]
            and gates["exact_shadow_runner_set_match"]
            and gates["live_odds_runner_rows_present"]
            and not gates["official_result_race_row_present"]
            and not gates["official_result_runner_rows_present"]
        )
        if awaiting_official_result_evidence:
            awaiting_official_result_evidence_count += 1
            awaiting_official_result_evidence_race_ids.append(race_id)
            minutes_since_latest_capture = None
            if generated_at_dt is not None and capture_dt is not None:
                minutes_since_latest_capture = round(
                    (generated_at_dt - capture_dt).total_seconds() / 60.0,
                    3,
                )
            official_result_recheck_ready = (
                minutes_since_latest_capture is None
                or minutes_since_latest_capture >= 5.0
            )
            awaiting_official_result_recheck_rows.append(
                {
                    "race_id": race_id,
                    "canonical_live_odds_race_id": validation.get(
                        "canonical_live_odds_race_id"
                    ),
                    "latest_live_odds_capture": latest_capture or None,
                    "minutes_since_latest_live_odds_capture": (
                        minutes_since_latest_capture
                    ),
                    "live_odds_runner_count": len(live_rows),
                    "exact_shadow_runner_set_match_count": len(exact_shadow_matches),
                    "official_result_recheck_ready": official_result_recheck_ready,
                    "next_authorized_action": (
                        "diagnostic_recheck_official_result_evidence_only"
                    ),
                    "join_authorized": False,
                    "db_write_performed": False,
                }
            )
        next_authorized_action = (
            "diagnostic_recheck_official_result_evidence_only"
            if awaiting_official_result_evidence
            else "diagnostic_review_join_blockers_only"
            if blockers
            else "diagnostic_review_join_eligible_report_only"
        )

        rows.append(
            {
                "race_id": race_id,
                "canonical_live_odds_race_id": validation.get(
                    "canonical_live_odds_race_id"
                ),
                "eligibility_status": status,
                "blockers": blockers,
                "blocker_category": (
                    "awaiting_official_result_evidence"
                    if awaiting_official_result_evidence
                    else "join_gate_blocked"
                    if blockers
                    else None
                ),
                "gates": gates,
                "awaiting_official_result_evidence": awaiting_official_result_evidence,
                "next_authorized_action": next_authorized_action,
                "join_authorized": False,
                "db_write_performed": False,
                "production_promotion": False,
                "latest_capture": latest_capture or None,
                "official_start_datetime": official_race.get("start_datetime"),
                "live_odds_source_url": source_url or None,
                "official_result_source_url": official_race.get("source_url"),
                "live_odds_runner_count": len(live_rows),
                "official_result_race_row_count": len(official_races),
                "official_result_runner_count": len(official_runners),
                "exact_shadow_runner_set_match_count": len(exact_shadow_matches),
                "missing_from_official_result": sorted_runner_key_rows(
                    live_keys - official_keys
                ),
                "missing_from_live_odds": sorted_runner_key_rows(
                    official_keys - live_keys
                ),
            }
        )

    awaiting_official_result_recheck_rows.sort(key=lambda row: str(row["race_id"]))
    awaiting_official_result_recheck_plan = {
        "schema_version": "join_eligibility_awaiting_official_result_recheck_plan_v1",
        "diagnostic_only": True,
        "join_acceptance_changed": False,
        "join_authorized": False,
        "db_write_performed": False,
        "authorized_action": "diagnostic_recheck_official_result_evidence_only",
        "minimum_minutes_since_latest_live_odds_capture_for_recheck": 5.0,
        "race_count": len(awaiting_official_result_recheck_rows),
        "recheck_ready_race_count": sum(
            1
            for row in awaiting_official_result_recheck_rows
            if row["official_result_recheck_ready"]
        ),
        "race_ids": [
            str(row["race_id"]) for row in awaiting_official_result_recheck_rows
        ],
        "races": awaiting_official_result_recheck_rows,
    }

    return {
        "schema_version": "live_odds_backlog_join_eligibility_packet_v1",
        "generated_at": runner_set_validation.get("generated_at"),
        "source_runner_set_validation": runner_set_validation.get(
            "source_recovery_queue"
        ),
        "diagnostic_only": True,
        "join_authorized": False,
        "db_write_performed": False,
        "production_promotion": False,
        "evaluated_race_count": len(rows),
        "eligible_report_only_race_count": eligible_count,
        "blocked_race_count": blocked_count,
        "awaiting_official_result_evidence_race_count": (
            awaiting_official_result_evidence_count
        ),
        "awaiting_official_result_evidence_race_ids": sorted(
            awaiting_official_result_evidence_race_ids
        ),
        "awaiting_official_result_recheck_plan": (
            awaiting_official_result_recheck_plan
        ),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "races": rows,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def add_live_odds_backlog_join_eligibility_report_fields(
    report: dict[str, Any],
    join_eligibility_packet: Mapping[str, Any],
) -> None:
    blocker_counts = join_eligibility_packet.get("blocker_counts")
    if not isinstance(blocker_counts, Mapping):
        blocker_counts = {}
    report["live_odds_backlog_join_eligibility_evaluated_race_count"] = int(
        join_eligibility_packet.get("evaluated_race_count") or 0
    )
    report["live_odds_backlog_join_eligibility_eligible_report_only_race_count"] = int(
        join_eligibility_packet.get("eligible_report_only_race_count") or 0
    )
    report["live_odds_backlog_join_eligibility_blocked_race_count"] = int(
        join_eligibility_packet.get("blocked_race_count") or 0
    )
    report["live_odds_backlog_join_eligibility_blocker_counts"] = dict(
        blocker_counts
    )
    report["live_odds_backlog_join_eligibility_diagnostic_only"] = bool(
        join_eligibility_packet.get("diagnostic_only", True)
    )
    report["live_odds_backlog_join_eligibility_join_authorized"] = bool(
        join_eligibility_packet.get("join_authorized")
    )
    report["live_odds_backlog_join_eligibility_db_write_performed"] = bool(
        join_eligibility_packet.get("db_write_performed")
    )
    join_recheck_plan = join_eligibility_packet.get(
        "awaiting_official_result_recheck_plan"
    )
    if not isinstance(join_recheck_plan, Mapping):
        join_recheck_plan = {}
    report[
        "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
    ] = int(join_recheck_plan.get("recheck_ready_race_count") or 0)
    report[
        "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_plan"
    ] = dict(join_recheck_plan)


def evidence_db_ingest_not_executed() -> dict[str, Any]:
    return {
        "schema_version": "official_result_evidence_db_ingest_status_v1",
        "execute": False,
        "status": "NOT_EXECUTED",
        "db_write_performed": False,
        "race_rows_seen": 0,
        "runner_rows_seen": 0,
        "valid_race_rows": 0,
        "valid_runner_rows": 0,
        "inserted_race_rows": 0,
        "inserted_runner_rows": 0,
        "blocked_race_rows": 0,
        "blocked_runner_rows": 0,
        "blocker_reason_counts": {},
        "table_names": {
            "races": OFFICIAL_RESULT_EVIDENCE_RACES_TABLE,
            "runners": OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE,
        },
        "write_scope": "append_only_official_result_evidence",
        "label_write_performed": False,
    }


def result_evidence_identity_blockers(
    race_row: Mapping[str, Any],
    runners: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    race_id = str(race_row.get("race_id") or "").strip()
    race_date = str(race_row.get("race_date") or "").strip()[:10]
    race_number = parse_int(race_row.get("race_number"))
    source_url = str(race_row.get("source_url") or "").strip()
    identity = parse_race_identity(race_id)
    if not race_id:
        blockers.append("race_id_missing")
    if identity.get("race_date") != race_date:
        blockers.append("race_date_identity_mismatch")
    if identity.get("race_number") != race_number:
        blockers.append("race_number_identity_mismatch")
    if identity.get("venue") and race_row.get("venue"):
        if normalize_venue(identity.get("venue")) != normalize_venue(race_row.get("venue")):
            blockers.append("venue_identity_mismatch")
    if race_row.get("source") != OFFICIAL_SOURCE:
        blockers.append("source_not_thedogs_official")
    if race_row.get("status") != RESULTED_STATUS:
        blockers.append("status_not_resulted")
    if not is_thedogs_official_url(source_url):
        blockers.append("official_source_url_missing_or_invalid")

    expected_box_order = [parse_int(value) for value in race_row.get("box_order") or []]
    expected_boxes = {box for box in expected_box_order if box is not None and box > 0}
    if not expected_boxes:
        blockers.append("box_order_missing")
    position_count = parse_int(race_row.get("position_count")) or 0
    if position_count <= 0:
        blockers.append("position_count_missing")
    if len(runners) != position_count:
        blockers.append("runner_count_mismatch")

    runner_boxes: set[int] = set()
    finish_positions: set[int] = set()
    for runner in runners:
        if str(runner.get("source") or "") != OFFICIAL_SOURCE:
            blockers.append("runner_source_not_thedogs_official")
        if str(runner.get("source_url") or "").strip() != source_url:
            blockers.append("runner_source_url_mismatch")
        if str(runner.get("race_id") or "").strip() != race_id:
            blockers.append("runner_race_id_mismatch")
        box = parse_int(runner.get("box_number"))
        finish = parse_int(runner.get("finish_position"))
        if box is None or box <= 0:
            blockers.append("runner_box_missing")
        else:
            runner_boxes.add(box)
        if finish is None or finish <= 0:
            blockers.append("runner_finish_position_missing")
        else:
            finish_positions.add(finish)
        if not str(runner.get("dog_name") or "").strip():
            blockers.append("runner_dog_name_missing")
    if runner_boxes and runner_boxes != expected_boxes:
        blockers.append("runner_box_order_mismatch")
    if finish_positions and len(finish_positions) != len(runners):
        blockers.append("duplicate_finish_positions")
    if finish_positions and finish_positions != set(range(1, len(runners) + 1)):
        blockers.append("finish_positions_not_contiguous")
    return list(dict.fromkeys(blockers))


def is_thedogs_official_url(value: Any) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
    except Exception:
        return False
    host = parsed.netloc.lower()
    return parsed.scheme in {"http", "https"} and (
        host == "thedogs.com.au" or host.endswith(".thedogs.com.au")
    )


def validate_official_result_evidence_rows(
    artifact_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    race_rows = [dict(row) for row in artifact_rows.get("race_rows") or []]
    runner_rows = [dict(row) for row in artifact_rows.get("runner_rows") or []]
    runners_by_race_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for runner in runner_rows:
        race_id = str(runner.get("race_id") or "").strip()
        runners_by_race_id[race_id].append(runner)

    valid_races: list[dict[str, Any]] = []
    valid_runners: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    valid_race_ids: set[str] = set()
    for race_row in race_rows:
        race_id = str(race_row.get("race_id") or "").strip()
        runners = runners_by_race_id.get(race_id) or []
        blockers = result_evidence_identity_blockers(race_row, runners)
        if blockers:
            blocked.append(
                {
                    "race_id": race_row.get("race_id"),
                    "source_url": race_row.get("source_url"),
                    "reasons": blockers,
                }
            )
            continue
        valid_races.append(race_row)
        valid_runners.extend(runners)
        valid_race_ids.add(race_id)

    blocked_runner_rows = [
        runner
        for runner in runner_rows
        if str(runner.get("race_id") or "").strip() not in valid_race_ids
    ]
    reason_count: dict[str, int] = {}
    for row in blocked:
        for reason in row.get("reasons") or ["unknown"]:
            reason_count[str(reason)] = reason_count.get(str(reason), 0) + 1
    return {
        "race_rows_seen": len(race_rows),
        "runner_rows_seen": len(runner_rows),
        "valid_race_rows": len(valid_races),
        "valid_runner_rows": len(valid_runners),
        "blocked_race_rows": len(blocked),
        "blocked_runner_rows": len(blocked_runner_rows),
        "blocker_reason_counts": dict(sorted(reason_count.items())),
        "blocked_races": blocked[:25],
        "race_rows": valid_races,
        "runner_rows": valid_runners,
    }


def ensure_official_result_evidence_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {OFFICIAL_RESULT_EVIDENCE_RACES_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            race_date TEXT NOT NULL,
            venue TEXT,
            race_number INTEGER,
            race_time TEXT,
            start_datetime TEXT,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            status TEXT NOT NULL,
            winner_name TEXT,
            winner_box INTEGER,
            position_count INTEGER NOT NULL,
            participant_count INTEGER,
            box_order_json TEXT NOT NULL,
            participant_source TEXT,
            captured_at TEXT NOT NULL,
            inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source_artifact_dir TEXT NOT NULL,
            row_json TEXT NOT NULL,
            UNIQUE(race_id, source_url, box_order_json)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            race_date TEXT NOT NULL,
            venue TEXT,
            race_number INTEGER,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            box_number INTEGER NOT NULL,
            dog_name TEXT NOT NULL,
            finish_position INTEGER NOT NULL,
            is_winner INTEGER NOT NULL,
            captured_at TEXT NOT NULL,
            inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source_artifact_dir TEXT NOT NULL,
            row_json TEXT NOT NULL,
            UNIQUE(race_id, source_url, box_number, dog_name, finish_position)
        )
        """
    )


def insert_official_result_evidence_rows(
    conn: sqlite3.Connection,
    *,
    race_rows: Sequence[Mapping[str, Any]],
    runner_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[int, int]:
    inserted_races = 0
    inserted_runners = 0
    source_artifact_dir = relpath(output_dir) or str(output_dir)
    for row in race_rows:
        before = conn.total_changes
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {OFFICIAL_RESULT_EVIDENCE_RACES_TABLE}
                (
                    race_id, race_date, venue, race_number, race_time,
                    start_datetime, source, source_url, status, winner_name,
                    winner_box, position_count, participant_count, box_order_json,
                    participant_source, captured_at, source_artifact_dir, row_json
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("race_id"),
                row.get("race_date"),
                row.get("venue"),
                parse_int(row.get("race_number")),
                row.get("race_time"),
                row.get("start_datetime"),
                row.get("source"),
                row.get("source_url"),
                row.get("status"),
                row.get("winner_name"),
                parse_int(row.get("winner_box")),
                parse_int(row.get("position_count")) or 0,
                parse_int(row.get("participant_count")),
                json.dumps(list(row.get("box_order") or []), sort_keys=True),
                row.get("participant_source"),
                row.get("captured_at"),
                source_artifact_dir,
                json.dumps(dict(row), sort_keys=True, default=str),
            ),
        )
        inserted_races += conn.total_changes - before
    for row in runner_rows:
        before = conn.total_changes
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE}
                (
                    race_id, race_date, venue, race_number, source, source_url,
                    box_number, dog_name, finish_position, is_winner,
                    captured_at, source_artifact_dir, row_json
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("race_id"),
                row.get("race_date"),
                row.get("venue"),
                parse_int(row.get("race_number")),
                row.get("source"),
                row.get("source_url"),
                parse_int(row.get("box_number")),
                row.get("dog_name"),
                parse_int(row.get("finish_position")),
                1 if row.get("is_winner") is True else 0,
                row.get("captured_at"),
                source_artifact_dir,
                json.dumps(dict(row), sort_keys=True, default=str),
            ),
        )
        inserted_runners += conn.total_changes - before
    return inserted_races, inserted_runners


def append_official_result_evidence_to_db(
    *,
    db_path: Path,
    artifact_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    execute: bool,
) -> dict[str, Any]:
    validation = validate_official_result_evidence_rows(artifact_rows)
    status = {
        **evidence_db_ingest_not_executed(),
        "execute": execute,
        "race_rows_seen": validation["race_rows_seen"],
        "runner_rows_seen": validation["runner_rows_seen"],
        "valid_race_rows": validation["valid_race_rows"],
        "valid_runner_rows": validation["valid_runner_rows"],
        "blocked_race_rows": validation["blocked_race_rows"],
        "blocked_runner_rows": validation["blocked_runner_rows"],
        "blocker_reason_counts": validation["blocker_reason_counts"],
        "blocked_races": validation["blocked_races"],
        "db_path": str(db_path),
    }
    has_valid_rows = validation["valid_race_rows"] > 0
    has_blocked_rows = validation["blocked_race_rows"] > 0
    if not execute:
        if has_valid_rows and has_blocked_rows:
            status["status"] = "READY_NOT_EXECUTED_WITH_QUARANTINE"
        elif has_valid_rows:
            status["status"] = "READY_NOT_EXECUTED"
        else:
            status["status"] = "NOT_EXECUTED"
        return status
    if validation["valid_race_rows"] <= 0 and validation["blocked_race_rows"]:
        status["status"] = "BLOCKED_UNSAFE_OFFICIAL_RESULT_EVIDENCE"
        return status
    if validation["valid_race_rows"] <= 0:
        status["status"] = "NO_VALID_OFFICIAL_RESULT_EVIDENCE"
        return status
    if not db_path.exists():
        status["status"] = "BLOCKED_DB_MISSING"
        status["blocker_reason_counts"] = {
            **dict(status.get("blocker_reason_counts") or {}),
            "db_missing": 1,
        }
        return status

    with sqlite3.connect(db_path) as conn:
        ensure_official_result_evidence_tables(conn)
        inserted_races, inserted_runners = insert_official_result_evidence_rows(
            conn,
            race_rows=validation["race_rows"],
            runner_rows=validation["runner_rows"],
            output_dir=output_dir,
        )
        conn.commit()
    status["inserted_race_rows"] = inserted_races
    status["inserted_runner_rows"] = inserted_runners
    status["db_write_performed"] = bool(inserted_races or inserted_runners)
    if inserted_races or inserted_runners:
        status["status"] = (
            "APPENDED_OFFICIAL_RESULT_EVIDENCE_WITH_QUARANTINE"
            if has_blocked_rows
            else "APPENDED_OFFICIAL_RESULT_EVIDENCE"
        )
    else:
        status["status"] = (
            "NOOP_ALREADY_PRESENT_WITH_QUARANTINE"
            if has_blocked_rows
            else "NOOP_ALREADY_PRESENT"
        )
    return status


def official_result_evidence_ingest_blocked_by_lock(
    *,
    db_path: Path,
    artifact_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    shared_lock: Mapping[str, Any],
) -> dict[str, Any]:
    status = append_official_result_evidence_to_db(
        db_path=db_path,
        artifact_rows=artifact_rows,
        output_dir=ROOT,
        execute=False,
    )
    status["execute"] = True
    status["status"] = "BLOCKED_SHARED_LOCK_HELD"
    status["shared_lock_status"] = dict(shared_lock)
    status["db_write_performed"] = False
    status["inserted_race_rows"] = 0
    status["inserted_runner_rows"] = 0
    blocker_counts = dict(status.get("blocker_reason_counts") or {})
    blocker_counts["shared_lock_held"] = 1
    status["blocker_reason_counts"] = blocker_counts
    return status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--upcoming-dir", type=Path)
    parser.add_argument("--shadow-run-dir", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "artifacts/prediction_snapshots")
    parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--existing-race-rows-jsonl", type=Path)
    parser.add_argument("--existing-runner-rows-jsonl", type=Path)
    parser.add_argument("--existing-quarantine-jsonl", type=Path)
    parser.add_argument("--race-id", action="append", default=[])
    parser.add_argument("--current-time")
    parser.add_argument("--require-ready-snapshot", action="store_true")
    parser.add_argument("--include-live-odds-backlog", action="store_true")
    parser.add_argument("--backlog-evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--backlog-limit", type=int, default=DEFAULT_BACKLOG_LIMIT)
    parser.add_argument(
        "--backlog-shadow-run-limit",
        type=int,
        default=DEFAULT_BACKLOG_SHADOW_RUN_LIMIT,
    )
    parser.add_argument("--backlog-lookback-days", type=int, default=DEFAULT_BACKLOG_LOOKBACK_DAYS)
    parser.add_argument(
        "--execute-db-ingest",
        action="store_true",
        help=(
            "Append validated official result evidence into append-only evidence "
            "tables. Does not write canonical labels."
        ),
    )
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument(
        "--require-lock-free",
        action="store_true",
        help=(
            "When --execute-db-ingest is set, validate the shared daemon lock "
            "before appending evidence rows and fail closed if a live lock is held."
        ),
    )
    args = parser.parse_args(argv)
    has_existing_artifacts = bool(args.existing_race_rows_jsonl or args.existing_runner_rows_jsonl)
    if has_existing_artifacts and not (
        args.existing_race_rows_jsonl and args.existing_runner_rows_jsonl
    ):
        parser.error(
            "--existing-race-rows-jsonl and --existing-runner-rows-jsonl must be provided together"
        )
    if args.upcoming_dir is None and args.shadow_run_dir is None and not has_existing_artifacts:
        parser.error(
            "--upcoming-dir, --shadow-run-dir, or existing official-result JSONL is required"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = assert_output_dir_safe(
        args.output_dir
        or args.evidence_root / f"autonomous_official_result_capture_{now_id(generated_at)}",
        evidence_root=args.evidence_root,
    )
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    ingest_report_path = output_dir / "official_result_ingest_dry_run_report.json"
    if args.existing_race_rows_jsonl and args.existing_runner_rows_jsonl:
        command = [
            sys.executable,
            str(ROOT / "scripts/autonomous_official_result_capture.py"),
            "--date",
            args.date,
            "--existing-race-rows-jsonl",
            str(args.existing_race_rows_jsonl),
            "--existing-runner-rows-jsonl",
            str(args.existing_runner_rows_jsonl),
            "--output-dir",
            str(output_dir),
            "--db",
            str(args.db),
        ]
        if args.existing_quarantine_jsonl:
            command.extend(["--existing-quarantine-jsonl", str(args.existing_quarantine_jsonl)])
        if args.execute_db_ingest:
            command.append("--execute-db-ingest")
        if args.lock_path:
            command.extend(["--lock-path", str(args.lock_path)])
        if args.require_lock_free:
            command.append("--require-lock-free")
        artifact_rows = load_official_result_artifact_rows(
            race_rows_path=args.existing_race_rows_jsonl,
            runner_rows_path=args.existing_runner_rows_jsonl,
            quarantine_rows_path=args.existing_quarantine_jsonl,
        )
        ingest_report = {
            "schema_version": "official_result_ingest_report_v1",
            "generated_at": generated_at.isoformat(),
            "status": "SUCCESS",
            "dry_run": True,
            "scope": {
                "date": args.date,
                "db_path": str(args.db),
                "candidate_source": "existing_official_result_artifact",
                "race_rows_path": str(args.existing_race_rows_jsonl),
                "runner_rows_path": str(args.existing_runner_rows_jsonl),
                "quarantine_rows_path": (
                    str(args.existing_quarantine_jsonl)
                    if args.existing_quarantine_jsonl
                    else None
                ),
            },
            "candidate_count": len(artifact_rows["race_rows"]),
            "candidate_race_ids": sorted(
                {
                    str(row.get("race_id") or "")
                    for row in artifact_rows["race_rows"]
                    if row.get("race_id")
                }
            ),
            "skipped_count": 0,
            "skipped": [],
            "ingested_count": len(artifact_rows["race_rows"]),
            "ingested": [],
            "failed_count": 0,
            "failed": [],
            "label_write_blockers": [],
            "backup_path": None,
            "result_label_write_approval": {
                "approved": False,
                "status": "not_approved",
                "required_for": "official_result_label_writes",
            },
            "dry_run_report_gate": None,
            "clean_for_label_write": False,
            "shadow_run_candidate_source_report": None,
            "live_odds_backlog": {},
        }
        returncode = 0
        write_json(ingest_report_path, ingest_report)
        write_text(output_dir / "official_result_ingest.stdout.txt", "")
        write_text(output_dir / "official_result_ingest.stderr.txt", "")
    elif args.shadow_run_dir is not None:
        command = [
            sys.executable,
            str(ROOT / "scripts/autonomous_official_result_capture.py"),
            "--date",
            args.date,
            "--shadow-run-dir",
            str(args.shadow_run_dir),
            "--output-dir",
            str(output_dir),
            "--db",
            str(args.db),
        ]
        if args.current_time:
            command.extend(["--current-time", args.current_time])
        if args.include_live_odds_backlog:
            command.extend(["--include-live-odds-backlog"])
            command.extend(["--backlog-evidence-root", str(args.backlog_evidence_root)])
            command.extend(["--backlog-limit", str(args.backlog_limit)])
            command.extend(["--backlog-shadow-run-limit", str(args.backlog_shadow_run_limit)])
            command.extend(["--backlog-lookback-days", str(args.backlog_lookback_days)])
        if args.execute_db_ingest:
            command.append("--execute-db-ingest")
        for race_id in args.race_id or []:
            command.extend(["--race-id", race_id])
        ingest_report, returncode = run_shadow_run_official_dry_run(
            db_path=args.db,
            shadow_run_dir=args.shadow_run_dir,
            target_date=args.date,
            current_time=parse_current_time(args.current_time),
            output_dir=output_dir,
            race_ids=args.race_id or [],
            include_live_odds_backlog=args.include_live_odds_backlog,
            backlog_evidence_root=args.backlog_evidence_root,
            backlog_limit=args.backlog_limit,
            backlog_shadow_run_limit=args.backlog_shadow_run_limit,
            backlog_lookback_days=args.backlog_lookback_days,
        )
        write_json(ingest_report_path, ingest_report)
        write_text(output_dir / "official_result_ingest.stdout.txt", "")
        write_text(output_dir / "official_result_ingest.stderr.txt", "")
        artifact_rows = build_artifact_rows(ingest_report, generated_at=generated_at)
    else:
        command = ingest_dry_run_command(
            db_path=args.db,
            target_date=args.date,
            upcoming_dir=args.upcoming_dir,
            snapshot_dir=args.snapshot_dir,
            output_path=ingest_report_path,
            race_ids=args.race_id or [],
            require_ready_snapshot=args.require_ready_snapshot,
        )
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        returncode = int(result.returncode or 0)
        write_text(output_dir / "official_result_ingest.stdout.txt", result.stdout or "")
        write_text(output_dir / "official_result_ingest.stderr.txt", result.stderr or "")
        ingest_report = load_json(ingest_report_path)
        artifact_rows = build_artifact_rows(ingest_report, generated_at=generated_at)
    write_jsonl(output_dir / "official_result_races.jsonl", artifact_rows["race_rows"])
    write_jsonl(output_dir / "official_result_runners.jsonl", artifact_rows["runner_rows"])
    write_jsonl(output_dir / "official_result_quarantine.jsonl", artifact_rows["quarantine_rows"])
    current_lock_status = (
        shared_lock_status(args.lock_path)
        if args.require_lock_free and args.execute_db_ingest
        else None
    )
    if (
        current_lock_status is not None
        and args.execute_db_ingest
        and not bool(current_lock_status.get("write_allowed"))
    ):
        evidence_db_ingest = official_result_evidence_ingest_blocked_by_lock(
            db_path=args.db,
            artifact_rows=artifact_rows,
            shared_lock=current_lock_status,
        )
    else:
        evidence_db_ingest = append_official_result_evidence_to_db(
            db_path=args.db,
            artifact_rows=artifact_rows,
            output_dir=output_dir,
            execute=args.execute_db_ingest,
        )
    report = build_capture_report(
        generated_at=generated_at,
        dry_run_command=command,
        dry_run_returncode=returncode,
        ingest_report=ingest_report,
        artifact_rows=artifact_rows,
        output_dir=output_dir,
        evidence_db_ingest=evidence_db_ingest,
    )
    if current_lock_status is not None:
        report["shared_lock_status"] = dict(current_lock_status)
    recovery_queue = build_live_odds_backlog_recovery_queue(capture_report=report)
    awaiting_official_result = (
        (recovery_queue.get("queues") or {}).get("awaiting_official_result_evidence")
        or {}
    )
    report["live_odds_backlog_awaiting_official_result_evidence_race_count"] = int(
        awaiting_official_result.get("race_count") or 0
    )
    report["live_odds_backlog_awaiting_official_result_evidence_race_ids"] = list(
        awaiting_official_result.get("race_ids") or []
    )
    report["live_odds_backlog_awaiting_official_result_evidence_authorized_action"] = (
        awaiting_official_result.get("authorized_action")
    )
    awaiting_recheck_plan = awaiting_official_result.get("recheck_plan")
    if not isinstance(awaiting_recheck_plan, Mapping):
        awaiting_recheck_plan = {}
    report["live_odds_backlog_awaiting_official_result_recheck_ready_race_count"] = int(
        awaiting_recheck_plan.get("recheck_ready_race_count") or 0
    )
    report["live_odds_backlog_awaiting_official_result_recheck_plan"] = dict(
        awaiting_recheck_plan
    )
    runner_set_validation = build_live_odds_backlog_runner_set_validation(
        recovery_queue=recovery_queue,
        db_path=args.db,
    )
    join_eligibility_packet = build_live_odds_backlog_join_eligibility_packet(
        runner_set_validation=runner_set_validation,
        db_path=args.db,
    )
    report["live_odds_backlog_runner_set_validation_retryable_race_count"] = int(
        runner_set_validation.get("retryable_race_count") or 0
    )
    report["live_odds_backlog_runner_set_validation_exact_match_race_count"] = int(
        runner_set_validation.get("exact_runner_set_match_race_count") or 0
    )
    report["live_odds_backlog_runner_set_validation_blocked_race_count"] = int(
        runner_set_validation.get("blocked_race_count") or 0
    )
    report["live_odds_backlog_runner_set_validation_diagnostic_only"] = bool(
        runner_set_validation.get("diagnostic_only", True)
    )
    report["live_odds_backlog_runner_set_validation_join_authorized"] = bool(
        runner_set_validation.get("join_authorized")
    )
    report["live_odds_backlog_runner_set_validation_db_write_performed"] = bool(
        runner_set_validation.get("db_write_performed")
    )
    report["live_odds_backlog_join_eligibility_evaluated_race_count"] = int(
        join_eligibility_packet.get("evaluated_race_count") or 0
    )
    report["live_odds_backlog_join_eligibility_eligible_report_only_race_count"] = int(
        join_eligibility_packet.get("eligible_report_only_race_count") or 0
    )
    report["live_odds_backlog_join_eligibility_blocked_race_count"] = int(
        join_eligibility_packet.get("blocked_race_count") or 0
    )
    join_eligibility_blocker_counts = join_eligibility_packet.get("blocker_counts")
    if not isinstance(join_eligibility_blocker_counts, Mapping):
        join_eligibility_blocker_counts = {}
    report["live_odds_backlog_join_eligibility_blocker_counts"] = dict(
        join_eligibility_blocker_counts
    )
    report["live_odds_backlog_join_eligibility_diagnostic_only"] = bool(
        join_eligibility_packet.get("diagnostic_only", True)
    )
    report["live_odds_backlog_join_eligibility_join_authorized"] = bool(
        join_eligibility_packet.get("join_authorized")
    )
    report["live_odds_backlog_join_eligibility_db_write_performed"] = bool(
        join_eligibility_packet.get("db_write_performed")
    )
    join_recheck_plan = join_eligibility_packet.get(
        "awaiting_official_result_recheck_plan"
    )
    if not isinstance(join_recheck_plan, Mapping):
        join_recheck_plan = {}
    report[
        "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
    ] = int(join_recheck_plan.get("recheck_ready_race_count") or 0)
    report[
        "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_plan"
    ] = dict(join_recheck_plan)
    write_json(output_dir / "autonomous_official_result_capture_report.json", report)
    write_json(output_dir / "live_odds_backlog_recovery_queue.json", recovery_queue)
    write_json(
        output_dir / "live_odds_backlog_runner_set_validation.json",
        runner_set_validation,
    )
    write_json(
        output_dir / "live_odds_backlog_join_eligibility_packet.json",
        join_eligibility_packet,
    )
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if returncode == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
