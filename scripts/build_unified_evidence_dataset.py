#!/usr/bin/env python3
"""Build a provenance-safe prediction/odds/result evidence dataset.

This is an artifact builder only. It reads shadow prediction artifacts plus
source-backed odds/result stores and writes versioned JSONL/CSV reports. It
does not write labels, odds, snapshots, manifests, registry state, or model
pointers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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

from utils.runner_completeness import normalise_runner_name  # noqa: E402
from utils.report_output_dir_guard import assert_prefixed_report_output_dir  # noqa: E402


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_"
OUTPUT_ARTIFACT_PREFIX = "unified_evidence_dataset_"
DATASET_FILE = "unified_evidence_dataset.jsonl"
CSV_FILE = "unified_evidence_dataset.csv"
REPORT_FILE = "unified_evidence_dataset_report.json"
SUMMARY_FILE = "SUMMARY.md"
ACCEPTED_SPORTSBET_BOX_SOURCES = {"explicit_dom", "runner_text"}
ACCEPTED_DOG_LEVEL_ODDS_LEVELS = {"dog", "runner"}
POST_RACE_SOURCE_URL_TOKENS = {"dividend", "payout", "result", "results"}
PREDICTION_TIMESTAMP_KEYS = (
    "prediction_timestamp",
    "effective_prediction_timestamp",
    "source_prediction_timestamp",
)
FEATURE_FREEZE_TIMESTAMP_KEYS = (
    "feature_freeze_timestamp",
    "effective_feature_freeze_timestamp",
    "source_feature_freeze_timestamp",
)
JOINED_SHADOW_EXACT_IDENTITY_STATUS = "exact_box_and_normalized_name"
OFFICIAL_RESULT_EVIDENCE_DB_SOURCE = "official_result_evidence_db"
GAP_CLASS_PRIORITY = (
    "source_set_missing",
    "identity_mismatch",
    "official_result_missing",
    "strict_prejump_odds_missing",
    "stage2_missing",
    "other_gate",
)
GAP_CLASS_ACTIONS = {
    "source_set_missing": "restore_rolling_source_set_inclusion",
    "identity_mismatch": "inspect_identity_match_or_join_artifact",
    "official_result_missing": "capture_or_join_official_result",
    "strict_prejump_odds_missing": "collect_strict_prejump_odds",
    "stage2_missing": "rebuild_stage2_shadow_predictions",
    "other_gate": "inspect_other_unified_evidence_gate",
}
NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "betting_or_ev_action": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
    "tgr_enabled": False,
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
        prefix_error="output_dir_must_be_unified_evidence_dataset_artifact",
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
        "schema_version": "unified_evidence_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def official_result_coverage_summary(
    *,
    official_result_runner_paths: Sequence[Path],
    official_result_evidence_db_audit: Mapping[str, Any],
    exclusion_reason_counts: Mapping[str, int],
) -> dict[str, Any]:
    requested_race_ids = [
        str(race_id)
        for race_id in official_result_evidence_db_audit.get("requested_race_ids") or []
        if str(race_id or "").strip()
    ]
    races_with_rows = [
        str(race_id)
        for race_id in official_result_evidence_db_audit.get("race_ids_with_rows") or []
        if str(race_id or "").strip()
    ]
    missing_race_ids = [
        str(race_id)
        for race_id in official_result_evidence_db_audit.get("missing_race_ids") or []
        if str(race_id or "").strip()
    ]
    return {
        "source": "unified_evidence_dataset",
        "requested_race_count": len(requested_race_ids),
        "requested_race_count_source": (
            "official_result_evidence_db_audit_requested_race_ids"
        ),
        "requested_race_ids": requested_race_ids,
        "races_with_rows_count": len(races_with_rows),
        "missing_race_count": len(missing_race_ids),
        "missing_race_ids": missing_race_ids,
        "races_with_rows": races_with_rows,
        "runner_path_count": len(official_result_runner_paths),
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": int(
            exclusion_reason_counts.get("official_result_missing") or 0
        ),
    }


def accepted_source_race_ids(
    join_eligibility_packet_audits: Sequence[Mapping[str, Any]],
) -> list[str]:
    return sorted(
        {
            str(race_id).strip()
            for audit in join_eligibility_packet_audits
            for race_id in audit.get("accepted_race_ids") or []
            if str(race_id or "").strip()
        }
    )


def joined_identity_mismatch_context(
    joined_shadow_prediction_audits: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    for audit in joined_shadow_prediction_audits:
        reason_map = audit.get("rejected_race_ids_by_reason") or {}
        if not isinstance(reason_map, Mapping):
            continue
        path = audit.get("path")
        for reason, race_ids in reason_map.items():
            if str(reason) != "identity_match_not_exact_box_and_normalized_name":
                continue
            if not isinstance(race_ids, Sequence) or isinstance(race_ids, (str, bytes)):
                continue
            for race_id in race_ids:
                race_key = str(race_id or "").strip()
                if not race_key:
                    continue
                entry = context.setdefault(
                    race_key,
                    {
                        "identity_mismatch_reasons": set(),
                        "identity_mismatch_source_paths": set(),
                    },
                )
                entry["identity_mismatch_reasons"].add(str(reason))
                if path:
                    entry["identity_mismatch_source_paths"].add(str(path))
    return {
        race_id: {
            "identity_mismatch_reasons": sorted(value["identity_mismatch_reasons"]),
            "identity_mismatch_source_paths": sorted(value["identity_mismatch_source_paths"]),
        }
        for race_id, value in sorted(context.items())
    }


def is_odds_detail_reason(reason: str) -> bool:
    return reason.startswith("odds_") or reason.startswith("unsupported_sportsbet_box_source")


def build_race_gap_prioritization(
    *,
    rows: Sequence[Mapping[str, Any]],
    join_eligibility_packet_audits: Sequence[Mapping[str, Any]],
    joined_shadow_prediction_audits: Sequence[Mapping[str, Any]],
    top_limit: int = 20,
) -> dict[str, Any]:
    rows_by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        race_id = str(row.get("race_id") or "").strip()
        if race_id:
            rows_by_race[race_id].append(row)

    dataset_race_ids = set(rows_by_race)
    source_race_ids = set(accepted_source_race_ids(join_eligibility_packet_audits))
    source_race_id_source = (
        "join_eligibility_packet_accepted_race_ids"
        if source_race_ids
        else "dataset_race_ids"
    )
    if not source_race_ids:
        source_race_ids = set(dataset_race_ids)

    identity_context = joined_identity_mismatch_context(joined_shadow_prediction_audits)
    source_race_ids.update(identity_context)
    source_race_ids.update(dataset_race_ids)

    gap_class_counts = {gap_class: 0 for gap_class in GAP_CLASS_PRIORITY}
    primary_gap_class_counts = {gap_class: 0 for gap_class in GAP_CLASS_PRIORITY}
    gap_rows: list[dict[str, Any]] = []

    for race_id in sorted(source_race_ids):
        race_rows = rows_by_race.get(race_id, [])
        row_count = len(race_rows)
        reason_counts = Counter()
        for row in race_rows:
            reason_counts.update(
                str(reason)
                for reason in row.get("excluded_from_unified_reason") or []
                if str(reason or "").strip()
            )

        if row_count == 0:
            gap_classes = ["source_set_missing"]
            official_missing_rows = 0
            strict_missing_rows = 0
            stage2_missing_rows = 0
            primary_missing_rows = 0
            unified_missing_rows = 0
            official_complete = False
            strict_complete = False
            stage2_complete = False
            primary_complete = False
            unified_complete = False
            race_date = None
            venue = None
            race_number = None
        else:
            official_rows = sum(1 for row in race_rows if row.get("official_result_available"))
            strict_rows = sum(1 for row in race_rows if row.get("strict_prejump_odds_available"))
            stage2_rows = sum(1 for row in race_rows if row.get("stage2_prediction_available"))
            primary_rows = sum(1 for row in race_rows if row.get("primary_prediction_available"))
            unified_rows = sum(1 for row in race_rows if row.get("unified_evidence_eligible"))
            official_missing_rows = max(0, row_count - official_rows)
            strict_missing_rows = max(0, row_count - strict_rows)
            stage2_missing_rows = max(0, row_count - stage2_rows)
            primary_missing_rows = max(0, row_count - primary_rows)
            unified_missing_rows = max(0, row_count - unified_rows)
            official_complete = row_count > 0 and official_missing_rows == 0
            strict_complete = row_count > 0 and strict_missing_rows == 0
            stage2_complete = row_count > 0 and stage2_missing_rows == 0
            primary_complete = row_count > 0 and primary_missing_rows == 0
            unified_complete = row_count > 0 and unified_missing_rows == 0
            race_date = next((row.get("race_date") for row in race_rows if row.get("race_date")), None)
            venue = next((row.get("venue") for row in race_rows if row.get("venue")), None)
            race_number = next(
                (row.get("race_number") for row in race_rows if row.get("race_number") is not None),
                None,
            )

            gap_classes = []
            if race_id in identity_context and not official_complete:
                gap_classes.append("identity_mismatch")
            if not official_complete:
                gap_classes.append("official_result_missing")
            if not strict_complete:
                gap_classes.append("strict_prejump_odds_missing")
            if not stage2_complete:
                gap_classes.append("stage2_missing")
            other_reasons = [
                reason
                for reason in sorted(reason_counts)
                if reason
                not in {
                    "official_result_missing",
                    "strict_prejump_odds_missing",
                    "stage2_shadow_prediction_missing",
                }
                and not (
                    "strict_prejump_odds_missing" in gap_classes
                    and is_odds_detail_reason(reason)
                )
            ]
            if primary_missing_rows or (unified_missing_rows and not gap_classes) or other_reasons:
                gap_classes.append("other_gate")

        gap_classes = [gap_class for gap_class in GAP_CLASS_PRIORITY if gap_class in gap_classes]
        if not gap_classes:
            continue

        for gap_class in gap_classes:
            gap_class_counts[gap_class] += 1
        primary_gap_class = gap_classes[0]
        primary_gap_class_counts[primary_gap_class] += 1
        identity_details = identity_context.get(race_id) or {}
        gap_rows.append(
            {
                "race_id": race_id,
                "race_date": race_date,
                "venue": venue,
                "race_number": race_number,
                "primary_gap_class": primary_gap_class,
                "gap_class": primary_gap_class,
                "gap_classes": gap_classes,
                "recommended_action": GAP_CLASS_ACTIONS[primary_gap_class],
                "source_set_present": row_count > 0,
                "row_count": row_count,
                "official_result_missing_rows": official_missing_rows,
                "strict_prejump_odds_missing_rows": strict_missing_rows,
                "stage2_missing_rows": stage2_missing_rows,
                "primary_prediction_missing_rows": primary_missing_rows,
                "unified_evidence_missing_rows": unified_missing_rows,
                "official_result_complete": official_complete,
                "strict_prejump_odds_complete": strict_complete,
                "stage2_complete": stage2_complete,
                "primary_prediction_complete": primary_complete,
                "unified_evidence_complete": unified_complete,
                "excluded_from_unified_reason_counts": dict(sorted(reason_counts.items())),
                "identity_mismatch_reasons": identity_details.get("identity_mismatch_reasons") or [],
                "identity_mismatch_source_paths": (
                    identity_details.get("identity_mismatch_source_paths") or []
                ),
            }
        )

    priority = {gap_class: index for index, gap_class in enumerate(GAP_CLASS_PRIORITY)}
    gap_rows.sort(
        key=lambda row: (
            priority.get(str(row.get("primary_gap_class")), len(priority)),
            -int(row.get("unified_evidence_missing_rows") or 0),
            str(row.get("race_id") or ""),
        )
    )
    return {
        "schema_version": "unified_evidence_race_gap_prioritization_v1",
        "scope": "source_set_or_dataset_race_level_gaps",
        "lineage_basis": source_race_id_source,
        "raw_db_count_basis": False,
        "source_race_id_source": source_race_id_source,
        "source_race_count": len(source_race_ids),
        "dataset_race_count": len(dataset_race_ids),
        "source_set_missing_race_count": sum(
            1 for race_id in source_race_ids if race_id not in dataset_race_ids
        ),
        "sample_blocking_gap_count": len(gap_rows),
        "primary_gap_class_counts": dict(sorted(primary_gap_class_counts.items())),
        "gap_class_counts": dict(sorted(gap_class_counts.items())),
        "top_gap_race_ids": [row["race_id"] for row in gap_rows[:top_limit]],
        "top_gap_races": gap_rows[:top_limit],
    }


def normalize_name(value: Any) -> str:
    return normalise_runner_name(value)


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


def runner_key(race_id: Any, box_number: Any, dog_name: Any) -> tuple[str, int | None, str]:
    return (str(race_id or "").strip(), parse_int(box_number), normalize_name(dog_name))


def sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def prediction_probability(row: Mapping[str, Any]) -> float | None:
    for key in (
        "shadow_rf_calibrated_probability",
        "calibrated_probability",
        "prediction_probability",
        "win_probability",
    ):
        value = parse_float(row.get(key))
        if value is not None:
            return value
    return None


def stage2_prediction_path(shadow_run_dir: Path) -> Path:
    root_path = shadow_run_dir / "stage2_shadow_predictions.jsonl"
    if root_path.exists() and load_jsonl(root_path):
        return root_path
    fallback_path = shadow_run_dir / "shadow_score_live" / "stage2_shadow_predictions.jsonl"
    if fallback_path.exists() and load_jsonl(fallback_path):
        return fallback_path
    return root_path if root_path.exists() else fallback_path


def parse_json_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def load_prediction_rows(shadow_run_dir: Path) -> dict[str, Any]:
    primary_path = shadow_run_dir / "shadow_predictions.jsonl"
    stage2_path = stage2_prediction_path(shadow_run_dir)
    primary_rows = load_jsonl(primary_path)
    stage2_rows = load_jsonl(stage2_path)
    primary_by_key = {
        runner_key(row.get("race_id"), row.get("box"), row.get("dog_name")): row
        for row in primary_rows
    }
    stage2_by_key = {
        runner_key(row.get("race_id"), row.get("box"), row.get("dog_name")): row
        for row in stage2_rows
    }
    keys = sorted(set(primary_by_key) | set(stage2_by_key))
    merged: list[dict[str, Any]] = []
    for key in keys:
        primary = primary_by_key.get(key) or {}
        stage2 = stage2_by_key.get(key) or {}
        source = primary or stage2
        race_identity = parse_race_identity(source.get("race_id"))
        merged.append(
            {
                "key": key,
                "race_id": key[0],
                "box_number": key[1],
                "dog_name": source.get("dog_name"),
                "dog_identity": key[2],
                "race_number": race_identity.get("race_number"),
                "venue": race_identity.get("venue"),
                "race_date": race_identity.get("race_date"),
                "primary": dict(primary),
                "stage2": dict(stage2),
            }
        )
    return {
        "primary_path": primary_path if primary_path.exists() else None,
        "stage2_path": stage2_path if stage2_path.exists() else None,
        "stage2_path_source": (
            "shadow_run_root"
            if stage2_path == shadow_run_dir / "stage2_shadow_predictions.jsonl"
            else "shadow_score_live_fallback"
        )
        if stage2_path.exists()
        else None,
        "primary_rows": primary_rows,
        "stage2_rows": stage2_rows,
        "merged_rows": merged,
    }


def join_eligibility_packet_audit(
    packet_paths: Sequence[Path],
) -> tuple[set[str], list[dict[str, Any]]]:
    eligible_race_ids: set[str] = set()
    audits: list[dict[str, Any]] = []
    for path in packet_paths:
        packet = load_json(path)
        rows = [row for row in packet.get("races") or [] if isinstance(row, Mapping)]
        accepted_ids: list[str] = []
        rejected_reasons = Counter()
        for row in rows:
            race_id = str(row.get("race_id") or "").strip()
            if not race_id:
                rejected_reasons["race_id_missing"] += 1
                continue
            if row.get("eligibility_status") != "JOIN_ELIGIBLE_REPORT_ONLY":
                rejected_reasons["eligibility_status_not_report_only"] += 1
                continue
            if row.get("blockers"):
                rejected_reasons["eligibility_blockers_present"] += 1
                continue
            if row.get("join_authorized"):
                rejected_reasons["join_authorized_not_report_only"] += 1
                continue
            if row.get("db_write_performed"):
                rejected_reasons["db_write_performed"] += 1
                continue
            eligible_race_ids.add(race_id)
            accepted_ids.append(race_id)
        audits.append(
            {
                "path": relpath(path),
                "schema_version": packet.get("schema_version"),
                "diagnostic_only": bool(packet.get("diagnostic_only", True)),
                "packet_join_authorized": bool(packet.get("join_authorized")),
                "packet_db_write_performed": bool(packet.get("db_write_performed")),
                "rows_seen": len(rows),
                "accepted_race_count": len(accepted_ids),
                "accepted_race_ids": sorted(accepted_ids),
                "rejected_race_count": sum(rejected_reasons.values()),
                "rejection_reason_counts": dict(sorted(rejected_reasons.items())),
            }
        )
    return eligible_race_ids, audits


def filter_prediction_info_by_race_ids(
    prediction_info: Mapping[str, Any],
    race_ids: set[str],
) -> dict[str, Any]:
    if not race_ids:
        return dict(prediction_info)
    filtered = dict(prediction_info)
    filtered["primary_rows"] = [
        row
        for row in prediction_info.get("primary_rows") or []
        if str(row.get("race_id") or "").strip() in race_ids
    ]
    filtered["stage2_rows"] = [
        row
        for row in prediction_info.get("stage2_rows") or []
        if str(row.get("race_id") or "").strip() in race_ids
    ]
    filtered["merged_rows"] = [
        row
        for row in prediction_info.get("merged_rows") or []
        if str(row.get("race_id") or "").strip() in race_ids
    ]
    return filtered


def db_official_results(db_path: Path, race_ids: set[str]) -> dict[tuple[str, int | None, str], dict[str, Any]]:
    if not db_path.exists() or not race_ids:
        return {}
    placeholders = ",".join("?" for _ in race_ids)
    query = f"""
        SELECT
            rm.race_id,
            rm.venue,
            rm.race_number,
            rm.race_date,
            rm.race_time,
            rm.start_datetime,
            rm.url,
            rm.winner_source,
            rm.results_status,
            d.dog_name,
            d.dog_clean_name,
            d.box_number,
            d.finish_position,
            d.data_source
        FROM race_metadata rm
        JOIN dog_race_data d ON d.race_id = rm.race_id
        WHERE rm.race_id IN ({placeholders})
          AND rm.winner_source = 'thedogs_official'
          AND d.data_source = 'thedogs_official'
          AND d.finish_position IS NOT NULL
    """
    results: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(query, sorted(race_ids)):
            item = dict(row)
            key = runner_key(item.get("race_id"), item.get("box_number"), item.get("dog_name"))
            results[key] = {
                "source": "thedogs_official",
                "source_url": item.get("url"),
                "finish_position": parse_int(item.get("finish_position")),
                "is_winner": parse_int(item.get("finish_position")) == 1,
                "dog_name": item.get("dog_name"),
                "box_number": parse_int(item.get("box_number")),
                "race_time": item.get("race_time"),
                "start_datetime": item.get("start_datetime"),
                "data_source": item.get("data_source"),
                "results_status": item.get("results_status"),
            }
    return results


def db_official_result_evidence_results(
    db_path: Path,
    race_ids: set[str],
) -> tuple[dict[tuple[str, int | None, str], dict[str, Any]], dict[str, Any]]:
    requested_race_ids = sorted(str(race_id) for race_id in race_ids)
    audit: dict[str, Any] = {
        "source": OFFICIAL_RESULT_EVIDENCE_DB_SOURCE,
        "db_path": str(db_path),
        "race_ids_requested": len(race_ids),
        "requested_race_ids": requested_race_ids,
        "race_ids_with_rows": [],
        "missing_race_ids": requested_race_ids,
        "race_table_present": False,
        "runner_table_present": False,
        "rows_seen": 0,
        "accepted_rows": 0,
        "duplicate_rows": 0,
        "conflict_rows": 0,
        "rejected_rows": 0,
        "rejection_reason_counts": {},
    }
    if not db_path.exists():
        audit["skipped_reason"] = "db_path_missing"
        return {}, audit
    if not race_ids:
        audit["skipped_reason"] = "no_prediction_race_ids"
        return {}, audit

    placeholders = ",".join("?" for _ in race_ids)
    query = f"""
        SELECT
            r.id AS runner_evidence_id,
            r.race_id,
            r.race_date,
            r.venue,
            r.race_number,
            r.source AS runner_source,
            r.source_url AS runner_source_url,
            r.box_number,
            r.dog_name,
            r.finish_position,
            r.is_winner,
            r.captured_at AS runner_captured_at,
            r.inserted_at AS runner_inserted_at,
            r.source_artifact_dir AS runner_source_artifact_dir,
            e.race_time,
            e.start_datetime,
            e.status AS race_status,
            e.source AS race_source,
            e.source_url AS race_source_url,
            e.captured_at AS race_captured_at,
            e.source_artifact_dir AS race_source_artifact_dir
        FROM autonomous_official_result_evidence_runners r
        LEFT JOIN autonomous_official_result_evidence_races e
          ON e.race_id = r.race_id
         AND e.source_url = r.source_url
        WHERE r.race_id IN ({placeholders})
    """

    results: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    conflicted_keys: set[tuple[str, int | None, str]] = set()
    seen_race_ids: set[str] = set()
    rejection_reasons = Counter()
    duplicate_rows = 0
    conflict_rows = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        audit["race_table_present"] = sqlite_table_exists(
            conn, "autonomous_official_result_evidence_races"
        )
        audit["runner_table_present"] = sqlite_table_exists(
            conn, "autonomous_official_result_evidence_runners"
        )
        if not audit["race_table_present"] or not audit["runner_table_present"]:
            audit["skipped_reason"] = "official_result_evidence_tables_missing"
            return {}, audit

        for row in conn.execute(query, sorted(race_ids)):
            audit["rows_seen"] += 1
            item = dict(row)
            race_id = str(item.get("race_id") or "").strip()
            if race_id:
                seen_race_ids.add(race_id)
            source = item.get("runner_source") or item.get("race_source")
            source_url = item.get("runner_source_url") or item.get("race_source_url")
            box_number = parse_int(item.get("box_number"))
            dog_name = item.get("dog_name")
            finish_position = parse_int(item.get("finish_position"))
            is_winner_int = parse_int(item.get("is_winner"))

            if not race_id:
                rejection_reasons["race_id_missing"] += 1
                continue
            if source != "thedogs_official":
                rejection_reasons["source_not_thedogs_official"] += 1
                continue
            if "thedogs.com.au" not in str(source_url or "").lower():
                rejection_reasons["source_url_not_thedogs"] += 1
                continue
            if box_number is None:
                rejection_reasons["box_number_missing"] += 1
                continue
            if not normalize_name(dog_name):
                rejection_reasons["dog_name_missing"] += 1
                continue
            if finish_position is None or finish_position <= 0:
                rejection_reasons["finish_position_invalid"] += 1
                continue
            if is_winner_int not in {0, 1}:
                rejection_reasons["is_winner_not_boolean_integer"] += 1
                continue
            is_winner = bool(is_winner_int)
            if is_winner != (finish_position == 1):
                rejection_reasons["finish_position_winner_flag_conflict"] += 1
                continue

            key = runner_key(race_id, box_number, dog_name)
            if key in conflicted_keys:
                conflict_rows += 1
                rejection_reasons["duplicate_evidence_result_conflict"] += 1
                continue

            result = {
                "source": "thedogs_official",
                "source_url": source_url,
                "finish_position": finish_position,
                "is_winner": is_winner,
                "dog_name": dog_name,
                "box_number": box_number,
                "race_time": item.get("race_time"),
                "start_datetime": item.get("start_datetime"),
                "data_source": OFFICIAL_RESULT_EVIDENCE_DB_SOURCE,
                "results_status": item.get("race_status") or "resulted",
                "official_result_evidence_runner_id": item.get("runner_evidence_id"),
                "official_result_evidence_captured_at": item.get("runner_captured_at")
                or item.get("race_captured_at"),
                "official_result_evidence_source_artifact_dir": item.get(
                    "runner_source_artifact_dir"
                )
                or item.get("race_source_artifact_dir"),
            }
            existing = results.get(key)
            if existing is not None:
                if (
                    existing.get("finish_position") != result.get("finish_position")
                    or existing.get("is_winner") != result.get("is_winner")
                ):
                    results.pop(key, None)
                    conflicted_keys.add(key)
                    conflict_rows += 1
                    rejection_reasons["duplicate_evidence_result_conflict"] += 1
                    continue
                duplicate_rows += 1
                continue

            results[key] = result

    audit["accepted_rows"] = len(results)
    audit["duplicate_rows"] = duplicate_rows
    audit["conflict_rows"] = conflict_rows
    audit["rejected_rows"] = sum(rejection_reasons.values())
    audit["race_ids_with_rows"] = sorted(seen_race_ids)
    audit["missing_race_ids"] = sorted(set(requested_race_ids) - seen_race_ids)
    audit["rejection_reason_counts"] = dict(sorted(rejection_reasons.items()))
    return results, audit


def artifact_official_results(
    runner_paths: Sequence[Path],
) -> dict[tuple[str, int | None, str], dict[str, Any]]:
    results: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    for path in runner_paths:
        for item in load_jsonl(path):
            if item.get("source") != "thedogs_official":
                continue
            finish_position = parse_int(item.get("finish_position"))
            if finish_position is None:
                continue
            key = runner_key(item.get("race_id"), item.get("box_number"), item.get("dog_name"))
            results[key] = {
                "source": "thedogs_official",
                "source_url": item.get("source_url"),
                "finish_position": finish_position,
                "is_winner": finish_position == 1,
                "dog_name": item.get("dog_name"),
                "box_number": parse_int(item.get("box_number")),
                "race_time": item.get("race_time"),
                "start_datetime": item.get("start_datetime"),
                "data_source": "official_result_artifact",
                "results_status": "resulted",
            }
    return results


def joined_shadow_official_results(
    joined_paths: Sequence[Path],
) -> tuple[dict[tuple[str, int | None, str], dict[str, Any]], list[dict[str, Any]]]:
    results: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    for path in joined_paths:
        rows = load_jsonl(path)
        rejection_reasons = Counter()
        rejected_race_ids_by_reason: dict[str, set[str]] = defaultdict(set)
        accepted_rows = 0
        duplicate_rows = 0
        for item in rows:
            race_id = str(item.get("race_id") or "").strip()

            def reject(reason: str) -> None:
                rejection_reasons[reason] += 1
                if race_id:
                    rejected_race_ids_by_reason[reason].add(race_id)

            if item.get("identity_match_status") != JOINED_SHADOW_EXACT_IDENTITY_STATUS:
                reject("identity_match_not_exact_box_and_normalized_name")
                continue
            box_number = parse_int(
                item.get("box_number") if item.get("box_number") not in (None, "") else item.get("box")
            )
            dog_name = item.get("dog_name") or item.get("official_dog_name")
            if not race_id:
                reject("race_id_missing")
                continue
            if box_number is None:
                reject("box_number_missing")
                continue
            if not normalize_name(dog_name):
                reject("dog_name_missing")
                continue
            finish_position = parse_int(item.get("finish_position"))
            if finish_position is None or finish_position <= 0:
                reject("finish_position_invalid")
                continue
            is_winner = parse_json_bool(item.get("is_winner"))
            if is_winner is None:
                reject("is_winner_not_json_bool")
                continue
            if is_winner != (finish_position == 1):
                reject("finish_position_winner_flag_conflict")
                continue
            source_url = item.get("result_url") or item.get("source_url") or item.get("race_url")
            if not str(source_url or "").strip():
                reject("result_source_url_missing")
                continue
            key = runner_key(race_id, box_number, dog_name)
            result = {
                "source": "forward_shadow_exact_join_artifact",
                "source_url": source_url,
                "finish_position": finish_position,
                "is_winner": is_winner,
                "dog_name": dog_name,
                "official_dog_name": item.get("official_dog_name"),
                "box_number": box_number,
                "race_time": item.get("race_time"),
                "start_datetime": item.get("start_datetime"),
                "data_source": "forward_shadow_exact_join_artifact",
                "results_status": "resulted",
                "identity_match_status": item.get("identity_match_status"),
                "joined_shadow_prediction_path": relpath(path),
            }
            existing = results.get(key)
            if existing is not None:
                if (
                    existing.get("finish_position") != result.get("finish_position")
                    or existing.get("is_winner") != result.get("is_winner")
                ):
                    reject("duplicate_joined_result_conflict")
                    continue
                duplicate_rows += 1
                continue
            results[key] = result
            accepted_rows += 1
        audits.append(
            {
                "path": relpath(path),
                "rows_seen": len(rows),
                "accepted_rows": accepted_rows,
                "duplicate_rows": duplicate_rows,
                "rejected_rows": sum(rejection_reasons.values()),
                "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
                "rejected_race_ids_by_reason": {
                    reason: sorted(race_ids)
                    for reason, race_ids in sorted(rejected_race_ids_by_reason.items())
                },
            }
        )
    return results, audits


def odds_rows_from_db(db_path: Path, race_ids: set[str]) -> list[dict[str, Any]]:
    if not db_path.exists() or not race_ids:
        return []
    placeholders = ",".join("?" for _ in race_ids)
    query = f"""
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
        WHERE race_id IN ({placeholders})
    """
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(query, sorted(race_ids)):
            rows.append(dict(row))
    return rows


def source_url_is_post_race(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    tokens = {token for token in re.split(r"[^a-z0-9]+", text) if token}
    return bool(tokens.intersection(POST_RACE_SOURCE_URL_TOKENS))


def parse_datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_race_time_value(value: Any) -> Any:
    text = str(value or "").strip().upper()
    if not text:
        return None
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def live_odds_jump_datetime(row: Mapping[str, Any]) -> datetime | None:
    for key in ("jump_datetime", "start_datetime", "race_datetime"):
        parsed = parse_datetime_value(row.get(key))
        if parsed is not None:
            return parsed

    race_time_datetime = parse_datetime_value(row.get("race_time"))
    if race_time_datetime is not None:
        return race_time_datetime

    race_date = parse_datetime_value(row.get("race_date"))
    race_time = parse_race_time_value(row.get("race_time"))
    if race_date is None or race_time is None:
        return None
    return datetime.combine(race_date.date(), race_time)


def has_timezone(value: datetime) -> bool:
    return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None


def datetime_before(value: datetime, reference: datetime) -> bool:
    left = value
    right = reference
    if has_timezone(left) and not has_timezone(right):
        right = right.replace(tzinfo=left.tzinfo)
    elif has_timezone(right) and not has_timezone(left):
        left = left.replace(tzinfo=right.tzinfo)
    return left < right


def first_datetime(row: Mapping[str, Any], keys: Sequence[str]) -> datetime | None:
    for key in keys:
        parsed = parse_datetime_value(row.get(key))
        if parsed is not None:
            return parsed
    return None


def first_datetime_from_sources(
    sources: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> datetime | None:
    for source in sources:
        parsed = first_datetime(source, keys)
        if parsed is not None:
            return parsed
    return None


def db_live_odds_temporal_reasons(
    row: Mapping[str, Any],
    *,
    generated_at: datetime | None,
    prediction_at: datetime | None = None,
    feature_freeze_at: datetime | None = None,
) -> list[str]:
    if generated_at is None or row.get("source_artifact_path"):
        return []

    capture_value = row.get("capture_timestamp") or row.get("timestamp")
    capture_dt = parse_datetime_value(capture_value)
    if capture_value and capture_dt is None:
        return ["odds_capture_timestamp_unparseable"]
    if capture_dt is None:
        return []

    prediction_dt = (
        first_datetime(row, PREDICTION_TIMESTAMP_KEYS)
        or prediction_at
        or generated_at
    )
    feature_freeze_dt = (
        first_datetime(row, FEATURE_FREEZE_TIMESTAMP_KEYS)
        or feature_freeze_at
        or prediction_dt
    )
    jump_dt = live_odds_jump_datetime(row)

    reasons: list[str] = []
    if not datetime_before(capture_dt, prediction_dt):
        reasons.append("odds_capture_not_before_prediction")
    if not datetime_before(capture_dt, feature_freeze_dt):
        reasons.append("odds_capture_not_before_feature_freeze")
    if jump_dt is None:
        reasons.append("odds_jump_timestamp_missing")
    elif not datetime_before(capture_dt, jump_dt):
        reasons.append("odds_capture_not_before_jump")
    return reasons


def validate_odds_row(
    row: Mapping[str, Any],
    *,
    generated_at: datetime | None = None,
    prediction_at: datetime | None = None,
    feature_freeze_at: datetime | None = None,
) -> list[str]:
    reasons: list[str] = []
    if str(row.get("market_type") or "").strip().lower() != "win":
        reasons.append("odds_market_not_win")
    if str(row.get("source") or "").strip().lower() != "sportsbet":
        reasons.append("odds_source_not_sportsbet")
    if parse_float(row.get("odds_decimal")) is None or float(row.get("odds_decimal") or 0) <= 1.0:
        reasons.append("odds_decimal_invalid")
    if not str(row.get("source_url") or "").strip():
        reasons.append("odds_source_url_missing")
    elif "sportsbet.com.au" not in str(row.get("source_url") or "").lower():
        reasons.append("odds_source_url_not_sportsbet")
    elif source_url_is_post_race(row.get("source_url")):
        reasons.append("odds_source_url_post_race")
    if not str(row.get("capture_timestamp") or "").strip():
        reasons.append("odds_capture_timestamp_missing")
    odds_level = str(row.get("odds_level") or "").strip().lower()
    if not odds_level:
        reasons.append("odds_level_missing")
    elif odds_level not in ACCEPTED_DOG_LEVEL_ODDS_LEVELS:
        reasons.append("odds_level_not_dog")
    box_source = str(row.get("sportsbet_box_source") or "").strip()
    if box_source not in ACCEPTED_SPORTSBET_BOX_SOURCES:
        reasons.append(f"unsupported_sportsbet_box_source:{box_source or 'missing'}")
    if parse_int(row.get("box_number") if row.get("box_number") not in (None, "") else row.get("box")) is None:
        reasons.append("odds_box_number_missing")
    if not normalize_name(row.get("dog_name") or row.get("dog_clean_name")):
        reasons.append("odds_dog_name_missing")
    reasons.extend(
        db_live_odds_temporal_reasons(
            row,
            generated_at=generated_at,
            prediction_at=prediction_at,
            feature_freeze_at=feature_freeze_at,
        )
    )
    return reasons


def capture_bucket(row: Mapping[str, Any]) -> str:
    mode = str(row.get("capture_mode") or "").lower()
    for token, bucket in (("t60", "t60"), ("t30", "t30"), ("t10", "t10"), ("t2", "t2")):
        if token in mode:
            return bucket
    return str(row.get("capture_mode") or "unknown").strip() or "unknown"


def odds_by_runner(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int | None, str], list[dict[str, Any]]]:
    by_key: dict[tuple[str, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = runner_key(
            row.get("race_id"),
            row.get("box_number") if row.get("box_number") not in (None, "") else row.get("box"),
            row.get("dog_name") or row.get("dog_clean_name"),
        )
        by_key[key].append(dict(row))
    return by_key


def best_strict_odds(
    rows: Sequence[Mapping[str, Any]],
    *,
    generated_at: datetime | None = None,
    prediction_at: datetime | None = None,
    feature_freeze_at: datetime | None = None,
) -> tuple[list[dict[str, Any]], Counter, list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    reasons = Counter()
    for row in rows:
        row_reasons = validate_odds_row(
            row,
            generated_at=generated_at,
            prediction_at=prediction_at,
            feature_freeze_at=feature_freeze_at,
        )
        if row_reasons:
            reasons.update(row_reasons)
            rejected = compact_odds(row)
            rejected["rejection_reasons"] = sorted(set(row_reasons))
            rejected_rows.append(rejected)
            continue
        valid_rows.append(dict(row))
    valid_rows.sort(key=lambda row: str(row.get("capture_timestamp") or row.get("timestamp") or ""))
    rejected_rows.sort(
        key=lambda row: str(row.get("capture_timestamp") or row.get("timestamp") or "")
    )
    return valid_rows, reasons, rejected_rows


def compact_odds(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "odds_decimal": parse_float(row.get("odds_decimal")),
        "odds_fractional": row.get("odds_fractional"),
        "source": row.get("source"),
        "source_table": row.get("source_table"),
        "source_artifact_path": row.get("source_artifact_path"),
        "source_url": row.get("source_url"),
        "capture_timestamp": row.get("capture_timestamp"),
        "capture_mode": row.get("capture_mode"),
        "odds_level": row.get("odds_level"),
        "sportsbet_box_source": row.get("sportsbet_box_source"),
        "sportsbet_raw_runner_text": row.get("sportsbet_raw_runner_text"),
    }


def artifact_shadow_odds_row(
    item: Mapping[str, Any],
    *,
    source_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert a validated shadow odds snapshot row into live_odds-like shape."""

    snapshot = item.get("odds_snapshot") if isinstance(item.get("odds_snapshot"), Mapping) else {}
    provenance = (
        snapshot.get("odds_provenance")
        if isinstance(snapshot.get("odds_provenance"), Mapping)
        else {}
    )
    if item.get("schema_version") != "shadow_odds_snapshot_runner_v1":
        return None, "unsupported_artifact_schema"
    if item.get("odds_match_status") != "valid_pre_jump_dog_odds":
        return None, "odds_match_status_not_valid_pre_jump_dog_odds"
    if item.get("odds_provenance_status") != "complete":
        return None, "odds_provenance_status_not_complete"
    if item.get("ev_calculation_status") not in {
        None,
        "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
    }:
        return None, "ev_calculation_status_not_report_only_disabled"
    if item.get("ev_win") is not None:
        return None, "ev_output_present"
    for key in (
        "odds_captured_before_feature_freeze",
        "odds_captured_before_jump",
        "odds_captured_before_prediction",
    ):
        if snapshot.get(key) is not True:
            return None, f"{key}_not_true"
    return {
        "race_id": item.get("race_id") or provenance.get("odds_race_id"),
        "venue": (item.get("race_context") or {}).get("venue")
        if isinstance(item.get("race_context"), Mapping)
        else None,
        "race_number": (item.get("race_context") or {}).get("race_number")
        if isinstance(item.get("race_context"), Mapping)
        else None,
        "race_date": (item.get("race_context") or {}).get("race_date")
        if isinstance(item.get("race_context"), Mapping)
        else None,
        "race_time": (item.get("race_context") or {}).get("race_time")
        if isinstance(item.get("race_context"), Mapping)
        else None,
        "dog_name": item.get("dog_name") or provenance.get("odds_dog_name"),
        "dog_clean_name": item.get("dog_name") or provenance.get("odds_dog_name"),
        "box_number": item.get("box") or provenance.get("odds_box_number"),
        "odds_decimal": snapshot.get("market_odds_win"),
        "odds_fractional": None,
        "market_type": snapshot.get("market_type"),
        "source": provenance.get("source"),
        "source_table": provenance.get("source_table") or "shadow_odds_snapshot",
        "timestamp": snapshot.get("odds_timestamp"),
        "is_current": 1,
        "topN": None,
        "source_url": provenance.get("source_url"),
        "capture_timestamp": snapshot.get("odds_timestamp"),
        "capture_mode": provenance.get("capture_mode"),
        "odds_level": snapshot.get("odds_level"),
        "sportsbet_box_source": provenance.get("sportsbet_box_source"),
        "sportsbet_list_position": provenance.get("sportsbet_list_position"),
        "sportsbet_raw_runner_text": provenance.get("sportsbet_raw_runner_text"),
        "source_artifact_path": relpath(source_path),
        "artifact_schema_version": item.get("schema_version"),
        "artifact_odds_match_status": item.get("odds_match_status"),
        "artifact_odds_provenance_status": item.get("odds_provenance_status"),
    }, None


def artifact_shadow_odds_rows(
    odds_jsonl_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for path in odds_jsonl_paths:
        rejection_reasons = Counter()
        accepted_rows = 0
        seen_rows = 0
        for item in load_jsonl(path):
            seen_rows += 1
            converted, reason = artifact_shadow_odds_row(item, source_path=path)
            if reason is not None or converted is None:
                rejection_reasons[reason or "artifact_odds_conversion_failed"] += 1
                continue
            row_reasons = validate_odds_row(converted)
            if row_reasons:
                rejection_reasons.update(row_reasons)
                continue
            rows.append(converted)
            accepted_rows += 1
        audits.append(
            {
                "path": relpath(path),
                "rows_seen": seen_rows,
                "accepted_rows": accepted_rows,
                "rejected_rows": sum(rejection_reasons.values()),
                "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
                "db_write": False,
                "odds_write": False,
                "betting_or_ev_action": False,
            }
        )
    return rows, audits


def build_dataset_rows(
    *,
    prediction_info: Mapping[str, Any],
    official_results: Mapping[tuple[str, int | None, str], Mapping[str, Any]],
    odds_index: Mapping[tuple[str, int | None, str], Sequence[Mapping[str, Any]]],
    generated_at: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in prediction_info.get("merged_rows") or []:
        key = item["key"]
        primary = item.get("primary") or {}
        stage2 = item.get("stage2") or {}
        result = dict(official_results.get(key) or {})
        odds_rows = list(odds_index.get(key) or [])
        prediction_at = (
            first_datetime_from_sources((primary, stage2), PREDICTION_TIMESTAMP_KEYS)
            or generated_at
        )
        feature_freeze_at = (
            first_datetime_from_sources(
                (primary, stage2),
                FEATURE_FREEZE_TIMESTAMP_KEYS,
            )
            or prediction_at
        )
        strict_odds, odds_reasons, rejected_odds = best_strict_odds(
            odds_rows,
            generated_at=generated_at,
            prediction_at=prediction_at,
            feature_freeze_at=feature_freeze_at,
        )
        odds_by_bucket = {
            capture_bucket(row): compact_odds(row)
            for row in strict_odds
        }
        artifact_shadow_odds_candidate_count = sum(
            1 for row in strict_odds if row.get("source_artifact_path")
        )
        artifact_shadow_odds_selected_bucket_count = sum(
            1
            for odds in odds_by_bucket.values()
            if isinstance(odds, Mapping) and odds.get("source_artifact_path")
        )

        exclusion_reasons: list[str] = []
        if not primary:
            exclusion_reasons.append("primary_shadow_prediction_missing")
        if not stage2:
            exclusion_reasons.append("stage2_shadow_prediction_missing")
        if not result:
            exclusion_reasons.append("official_result_missing")
        if not strict_odds:
            exclusion_reasons.append("strict_prejump_odds_missing")
            exclusion_reasons.extend(sorted(odds_reasons))

        official_available = bool(result)
        stage2_available = bool(stage2 and prediction_probability(stage2) is not None)
        primary_available = bool(primary and prediction_probability(primary) is not None)
        strict_odds_available = bool(strict_odds)
        row = {
            "schema_version": "unified_evidence_runner_v1",
            "generated_at": generated_at.isoformat(),
            "race_id": item.get("race_id"),
            "venue": item.get("venue"),
            "race_number": item.get("race_number"),
            "race_date": item.get("race_date"),
            "dog_name": item.get("dog_name"),
            "dog_identity": item.get("dog_identity"),
            "box_number": item.get("box_number"),
            "primary_shadow_probability": prediction_probability(primary),
            "primary_shadow_rank": parse_int(primary.get("predicted_rank")),
            "primary_model_version": primary.get("model_version"),
            "primary_model_source": primary.get("model_source"),
            "primary_calibration_method": primary.get("calibration_method"),
            "primary_prediction_schema_version": primary.get("schema_version"),
            "stage2_shadow_probability": prediction_probability(stage2),
            "stage2_shadow_uncalibrated_probability": parse_float(
                stage2.get("shadow_rf_uncalibrated_probability")
            ),
            "stage2_shadow_rank": parse_int(stage2.get("predicted_rank")),
            "stage2_challenger_key": stage2.get("stage2_challenger_key"),
            "stage2_forward_shadow_status": stage2.get("stage2_forward_shadow_status"),
            "stage2_model_version": stage2.get("model_version"),
            "stage2_model_source": stage2.get("model_source"),
            "official_result_source": result.get("source"),
            "official_result_source_url": result.get("source_url"),
            "finish_position": result.get("finish_position"),
            "is_winner": bool(result.get("is_winner")) if result else None,
            "official_result_status": result.get("results_status"),
            "official_result_data_source": result.get("data_source"),
            "strict_prejump_odds_available": strict_odds_available,
            "strict_prejump_odds_count": len(strict_odds),
            "artifact_shadow_odds_available": artifact_shadow_odds_candidate_count > 0,
            "artifact_shadow_odds_candidate_count": artifact_shadow_odds_candidate_count,
            "artifact_shadow_odds_selected_bucket_count": (
                artifact_shadow_odds_selected_bucket_count
            ),
            "all_live_odds_candidate_count": len(odds_rows),
            "odds_by_capture_bucket": odds_by_bucket,
            "odds_exclusion_reasons": dict(odds_reasons),
            "rejected_live_odds_candidate_count": len(rejected_odds),
            "rejected_live_odds_candidates": rejected_odds,
            "official_result_available": official_available,
            "primary_prediction_available": primary_available,
            "stage2_prediction_available": stage2_available,
            "label_evaluation_eligible": official_available and primary_available,
            "stage2_evaluation_eligible": official_available and stage2_available,
            "odds_evaluation_eligible": official_available and strict_odds_available,
            "unified_evidence_eligible": (
                official_available
                and primary_available
                and stage2_available
                and strict_odds_available
            ),
            "excluded_from_unified_reason": sorted(set(exclusion_reasons)),
        }
        rows.append(row)
    return rows


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "race_id",
        "venue",
        "race_number",
        "race_date",
        "dog_name",
        "box_number",
        "primary_shadow_probability",
        "primary_shadow_rank",
        "stage2_shadow_probability",
        "stage2_shadow_rank",
        "finish_position",
        "is_winner",
        "strict_prejump_odds_available",
        "strict_prejump_odds_count",
        "label_evaluation_eligible",
        "stage2_evaluation_eligible",
        "odds_evaluation_eligible",
        "unified_evidence_eligible",
        "excluded_from_unified_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def build_report(
    *,
    generated_at: datetime,
    rows: Sequence[Mapping[str, Any]],
    shadow_run_dir: Path,
    prediction_info: Mapping[str, Any],
    db_path: Path,
    output_dir: Path,
    official_result_runner_paths: Sequence[Path],
    official_result_evidence_db_audit: Mapping[str, Any],
    joined_shadow_prediction_paths: Sequence[Path],
    joined_shadow_prediction_audits: Sequence[Mapping[str, Any]],
    join_eligibility_packet_paths: Sequence[Path],
    join_eligibility_packet_audits: Sequence[Mapping[str, Any]],
    odds_jsonl_paths: Sequence[Path],
    artifact_odds_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    race_ids = {str(row.get("race_id")) for row in rows if row.get("race_id")}
    join_eligibility_accepted_race_ids = sorted(
        {
            str(race_id)
            for audit in join_eligibility_packet_audits
            for race_id in audit.get("accepted_race_ids") or []
            if str(race_id or "").strip()
        }
    )
    exclusion_counts = Counter()
    odds_reason_counts = Counter()
    rejected_odds_reason_counts = Counter()
    artifact_odds_rejection_reason_counts = Counter()
    rejected_odds_candidate_count = 0
    rows_with_rejected_live_odds_candidates = 0
    rejected_live_odds_candidate_samples: list[dict[str, Any]] = []
    for audit in artifact_odds_audits:
        artifact_odds_rejection_reason_counts.update(
            {
                str(reason): int(count)
                for reason, count in (
                    audit.get("rejection_reason_counts") or {}
                ).items()
                if reason and int(count or 0) > 0
            }
        )
    for row in rows:
        exclusion_counts.update(row.get("excluded_from_unified_reason") or [])
        odds_reason_counts.update((row.get("odds_exclusion_reasons") or {}).keys())
        rejected_candidates = row.get("rejected_live_odds_candidates") or []
        if rejected_candidates:
            rows_with_rejected_live_odds_candidates += 1
        rejected_odds_candidate_count += len(rejected_candidates)
        for candidate in rejected_candidates:
            rejected_odds_reason_counts.update(candidate.get("rejection_reasons") or [])
            if len(rejected_live_odds_candidate_samples) >= 20:
                continue
            rejected_live_odds_candidate_samples.append(
                {
                    "race_id": row.get("race_id"),
                    "dog_name": row.get("dog_name"),
                    "box_number": row.get("box_number"),
                    "rejection_reasons": candidate.get("rejection_reasons") or [],
                    "source": candidate.get("source"),
                    "source_url": candidate.get("source_url"),
                    "capture_timestamp": candidate.get("capture_timestamp"),
                    "capture_mode": candidate.get("capture_mode"),
                    "sportsbet_box_source": candidate.get("sportsbet_box_source"),
                }
            )
    official_result_coverage = official_result_coverage_summary(
        official_result_runner_paths=official_result_runner_paths,
        official_result_evidence_db_audit=official_result_evidence_db_audit,
        exclusion_reason_counts=exclusion_counts,
    )
    race_gap_prioritization = build_race_gap_prioritization(
        rows=rows,
        join_eligibility_packet_audits=join_eligibility_packet_audits,
        joined_shadow_prediction_audits=joined_shadow_prediction_audits,
    )
    return {
        "schema_version": "unified_evidence_dataset_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT" if rows else "UNIFIED_EVIDENCE_DATASET_EMPTY",
        "output_dir": relpath(output_dir),
        "dataset_jsonl": relpath(output_dir / DATASET_FILE),
        "dataset_csv": relpath(output_dir / CSV_FILE),
        "shadow_run_dir": relpath(shadow_run_dir),
        "primary_predictions_path": relpath(prediction_info.get("primary_path")),
        "stage2_predictions_path": relpath(prediction_info.get("stage2_path")),
        "stage2_predictions_path_source": prediction_info.get("stage2_path_source"),
        "db_path": str(db_path),
        "official_result_runner_paths": [relpath(path) for path in official_result_runner_paths],
        "official_result_evidence_db_audit": dict(official_result_evidence_db_audit),
        "official_result_coverage": official_result_coverage,
        "race_gap_prioritization": race_gap_prioritization,
        "joined_shadow_prediction_paths": [relpath(path) for path in joined_shadow_prediction_paths],
        "joined_shadow_prediction_audits": list(joined_shadow_prediction_audits),
        "joined_shadow_prediction_rows_seen": sum(
            int(audit.get("rows_seen") or 0) for audit in joined_shadow_prediction_audits
        ),
        "joined_shadow_prediction_rows_accepted": sum(
            int(audit.get("accepted_rows") or 0) for audit in joined_shadow_prediction_audits
        ),
        "joined_shadow_prediction_rows_rejected": sum(
            int(audit.get("rejected_rows") or 0) for audit in joined_shadow_prediction_audits
        ),
        "join_eligibility_packet_paths": [
            relpath(path) for path in join_eligibility_packet_paths
        ],
        "join_eligibility_packet_audits": list(join_eligibility_packet_audits),
        "join_eligibility_packet_rows_seen": sum(
            int(audit.get("rows_seen") or 0)
            for audit in join_eligibility_packet_audits
        ),
        "join_eligibility_packet_accepted_races": sum(
            int(audit.get("accepted_race_count") or 0)
            for audit in join_eligibility_packet_audits
        ),
        "join_eligibility_packet_accepted_race_ids": join_eligibility_accepted_race_ids,
        "join_eligibility_packet_accepted_races_present_in_shadow_run": len(
            set(join_eligibility_accepted_race_ids) & race_ids
        ),
        "join_eligibility_packet_accepted_race_ids_missing_from_shadow_run": sorted(
            set(join_eligibility_accepted_race_ids) - race_ids
        ),
        "join_eligibility_packet_rejected_races": sum(
            int(audit.get("rejected_race_count") or 0)
            for audit in join_eligibility_packet_audits
        ),
        "odds_jsonl_paths": [relpath(path) for path in odds_jsonl_paths],
        "artifact_odds_audits": list(artifact_odds_audits),
        "artifact_odds_rows_seen": sum(
            int(audit.get("rows_seen") or 0) for audit in artifact_odds_audits
        ),
        "artifact_odds_rows_accepted": sum(
            int(audit.get("accepted_rows") or 0) for audit in artifact_odds_audits
        ),
        "artifact_odds_rows_rejected": sum(
            int(audit.get("rejected_rows") or 0) for audit in artifact_odds_audits
        ),
        "artifact_odds_rejection_reason_counts": dict(
            sorted(artifact_odds_rejection_reason_counts.items())
        ),
        "row_count": len(rows),
        "race_count": len(race_ids),
        "primary_prediction_rows": len(prediction_info.get("primary_rows") or []),
        "stage2_prediction_rows": len(prediction_info.get("stage2_rows") or []),
        "rows_with_official_results": sum(1 for row in rows if row.get("official_result_available")),
        "rows_with_official_result_evidence_db": sum(
            1
            for row in rows
            if row.get("official_result_data_source") == OFFICIAL_RESULT_EVIDENCE_DB_SOURCE
        ),
        "rows_with_stage2_predictions": sum(1 for row in rows if row.get("stage2_prediction_available")),
        "rows_with_strict_prejump_odds": sum(
            1 for row in rows if row.get("strict_prejump_odds_available")
        ),
        "rows_with_artifact_shadow_odds": sum(
            1
            for row in rows
            if any(
                odds.get("source_artifact_path")
                for odds in (row.get("odds_by_capture_bucket") or {}).values()
                if isinstance(odds, Mapping)
            )
        ),
        "rows_with_artifact_shadow_odds_candidates": sum(
            1 for row in rows if row.get("artifact_shadow_odds_available")
        ),
        "artifact_shadow_odds_candidate_count": sum(
            int(row.get("artifact_shadow_odds_candidate_count") or 0)
            for row in rows
        ),
        "artifact_shadow_odds_selected_bucket_count": sum(
            int(row.get("artifact_shadow_odds_selected_bucket_count") or 0)
            for row in rows
        ),
        "label_evaluation_eligible_rows": sum(
            1 for row in rows if row.get("label_evaluation_eligible")
        ),
        "stage2_evaluation_eligible_rows": sum(
            1 for row in rows if row.get("stage2_evaluation_eligible")
        ),
        "odds_evaluation_eligible_rows": sum(
            1 for row in rows if row.get("odds_evaluation_eligible")
        ),
        "unified_evidence_eligible_rows": sum(
            1 for row in rows if row.get("unified_evidence_eligible")
        ),
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "odds_exclusion_reason_counts": dict(sorted(odds_reason_counts.items())),
        "rejected_live_odds_candidate_count": rejected_odds_candidate_count,
        "rows_with_rejected_live_odds_candidates": rows_with_rejected_live_odds_candidates,
        "rejected_live_odds_candidate_reason_counts": dict(
            sorted(rejected_odds_reason_counts.items())
        ),
        "rejected_live_odds_candidate_samples": rejected_live_odds_candidate_samples,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def summary_markdown(report: Mapping[str, Any]) -> str:
    official_result = (
        report.get("official_result_coverage")
        if isinstance(report.get("official_result_coverage"), Mapping)
        else {}
    )
    race_gap_prioritization = (
        report.get("race_gap_prioritization")
        if isinstance(report.get("race_gap_prioritization"), Mapping)
        else {}
    )
    return "\n".join(
        [
            "# Unified Evidence Dataset",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Rows: `{report.get('row_count')}`",
            f"- Races: `{report.get('race_count')}`",
            f"- Stage 2 prediction rows: `{report.get('stage2_prediction_rows')}`",
            f"- Rows with official results: `{report.get('rows_with_official_results')}`",
            f"- Rows with official-result evidence DB: `{report.get('rows_with_official_result_evidence_db')}`",
            f"- Official-result coverage requested races: `{official_result.get('requested_race_count')}`",
            f"- Official-result coverage requested race count source: `{official_result.get('requested_race_count_source')}`",
            f"- Official-result coverage races with rows: `{official_result.get('races_with_rows_count')}`",
            f"- Official-result coverage missing races: `{official_result.get('missing_race_count')}`",
            f"- Official-result missing exclusions: `{official_result.get('missing_exclusion_count')}`",
            f"- Official-result runner path count: `{official_result.get('runner_path_count')}`",
            f"- Official-result runner paths source field: `{official_result.get('runner_paths_source_field')}`",
            f"- Rows with strict pre-jump odds: `{report.get('rows_with_strict_prejump_odds')}`",
            f"- Rows with artifact shadow odds: `{report.get('rows_with_artifact_shadow_odds')}`",
            f"- Rows with artifact shadow odds candidates: `{report.get('rows_with_artifact_shadow_odds_candidates')}`",
            f"- Artifact odds accepted rows: `{report.get('artifact_odds_rows_accepted')}`",
            f"- Artifact odds rejected rows: `{report.get('artifact_odds_rows_rejected')}`",
            f"- Artifact odds rejection reasons: `{report.get('artifact_odds_rejection_reason_counts')}`",
            f"- Rejected live odds candidates: `{report.get('rejected_live_odds_candidate_count')}`",
            f"- Rows with rejected live odds candidates: `{report.get('rows_with_rejected_live_odds_candidates')}`",
            f"- Join-eligibility accepted races: `{report.get('join_eligibility_packet_accepted_races')}`",
            f"- Join-eligibility accepted races present: `{report.get('join_eligibility_packet_accepted_races_present_in_shadow_run')}`",
            f"- Join-eligibility accepted races missing: `{report.get('join_eligibility_packet_accepted_race_ids_missing_from_shadow_run')}`",
            f"- Join-eligibility rejected races: `{report.get('join_eligibility_packet_rejected_races')}`",
            f"- Label-evaluation eligible rows: `{report.get('label_evaluation_eligible_rows')}`",
            f"- Stage 2-evaluation eligible rows: `{report.get('stage2_evaluation_eligible_rows')}`",
            f"- Odds-evaluation eligible rows: `{report.get('odds_evaluation_eligible_rows')}`",
            f"- Unified-evidence eligible rows: `{report.get('unified_evidence_eligible_rows')}`",
            f"- Race gap source basis: `{race_gap_prioritization.get('source_race_id_source')}`",
            f"- Race gap raw DB count basis: `{race_gap_prioritization.get('raw_db_count_basis')}`",
            f"- Race gap source races: `{race_gap_prioritization.get('source_race_count')}`",
            f"- Race gap sample-blocking races: `{race_gap_prioritization.get('sample_blocking_gap_count')}`",
            f"- Race gap primary class counts: `{race_gap_prioritization.get('primary_gap_class_counts')}`",
            f"- Race gap all class counts: `{race_gap_prioritization.get('gap_class_counts')}`",
            f"- Race gap top race IDs: `{race_gap_prioritization.get('top_gap_race_ids')}`",
            "",
            "## Exclusions",
            "",
            json.dumps(report.get("exclusion_reason_counts") or {}, indent=2, sort_keys=True),
            "",
            "## Race Gap Prioritization",
            "",
            json.dumps(
                race_gap_prioritization.get("top_gap_races") or [],
                indent=2,
                sort_keys=True,
            ),
            "",
            "## Rejected Live Odds Candidate Reasons",
            "",
            json.dumps(
                report.get("rejected_live_odds_candidate_reason_counts") or {},
                indent=2,
                sort_keys=True,
            ),
            "",
            "No training, production promotion, registry mutation, production pointer update, DB write, label write, odds write, betting/EV action, snapshot rewrite, manifest rewrite, or TGR enablement was performed.",
            "",
        ]
    )


def build_dataset(
    *,
    shadow_run_dir: Path,
    db_path: Path,
    output_dir: Path,
    evidence_root: Path | None = None,
    official_result_runner_paths: Sequence[Path] = (),
    joined_shadow_prediction_paths: Sequence[Path] = (),
    join_eligibility_packet_paths: Sequence[Path] = (),
    odds_jsonl_paths: Sequence[Path] = (),
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir, evidence_root=evidence_root))
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_info = load_prediction_rows(shadow_run_dir)
    eligible_race_ids, join_eligibility_audits = join_eligibility_packet_audit(
        join_eligibility_packet_paths
    )
    if join_eligibility_packet_paths:
        prediction_info = filter_prediction_info_by_race_ids(
            prediction_info,
            eligible_race_ids,
        )
    race_ids = {row["race_id"] for row in prediction_info["merged_rows"] if row.get("race_id")}
    official_results = db_official_results(db_path, race_ids)
    official_result_evidence_results, official_result_evidence_audit = (
        db_official_result_evidence_results(db_path, race_ids)
    )
    for key, result in official_result_evidence_results.items():
        official_results.setdefault(key, result)
    official_results.update(artifact_official_results(official_result_runner_paths))
    joined_results, joined_audits = joined_shadow_official_results(joined_shadow_prediction_paths)
    for key, result in joined_results.items():
        official_results.setdefault(key, result)
    odds_rows = odds_rows_from_db(db_path, race_ids)
    artifact_odds_rows_list, artifact_odds_audits = artifact_shadow_odds_rows(
        odds_jsonl_paths
    )
    odds_rows.extend(artifact_odds_rows_list)
    odds_index = odds_by_runner(odds_rows)
    rows = build_dataset_rows(
        prediction_info=prediction_info,
        official_results=official_results,
        odds_index=odds_index,
        generated_at=generated_at,
    )
    write_jsonl(output_dir / DATASET_FILE, rows)
    write_csv(output_dir / CSV_FILE, rows)
    report = build_report(
        generated_at=generated_at,
        rows=rows,
        shadow_run_dir=shadow_run_dir,
        prediction_info=prediction_info,
        db_path=db_path,
        output_dir=output_dir,
        official_result_runner_paths=official_result_runner_paths,
        official_result_evidence_db_audit=official_result_evidence_audit,
        joined_shadow_prediction_paths=joined_shadow_prediction_paths,
        joined_shadow_prediction_audits=joined_audits,
        join_eligibility_packet_paths=join_eligibility_packet_paths,
        join_eligibility_packet_audits=join_eligibility_audits,
        odds_jsonl_paths=odds_jsonl_paths,
        artifact_odds_audits=artifact_odds_audits,
    )
    write_json(output_dir / REPORT_FILE, report)
    write_text(output_dir / SUMMARY_FILE, summary_markdown(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-run-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--official-result-runners-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--joined-shadow-predictions-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--join-eligibility-packet", action="append", type=Path, default=[])
    parser.add_argument("--odds-jsonl", action="append", type=Path, default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir
        or DEFAULT_EVIDENCE_ROOT / f"unified_evidence_dataset_{now_id(generated_at)}"
    )
    report = build_dataset(
        shadow_run_dir=args.shadow_run_dir,
        db_path=args.db,
        output_dir=output_dir,
        evidence_root=args.evidence_root,
        official_result_runner_paths=args.official_result_runners_jsonl,
        joined_shadow_prediction_paths=args.joined_shadow_predictions_jsonl,
        join_eligibility_packet_paths=args.join_eligibility_packet,
        odds_jsonl_paths=args.odds_jsonl,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("final_status") == "UNIFIED_EVIDENCE_DATASET_BUILT" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
