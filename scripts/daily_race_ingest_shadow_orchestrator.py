#!/usr/bin/env python3
"""Fail-closed daily/pre-jump race discovery and shadow-only scoring.

This orchestrator is intentionally shadow-only. It verifies the recovered
canonical DB state, classifies current/future race CSVs, runs the non-TGR
RandomForest shadow scorer only on eligible staged copies, and writes reports
inside a fresh daily shadow artifact directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import validate_upcoming_races  # noqa: E402
from scripts.run_feature_recovery_execution_v1 import (  # noqa: E402
    DEFAULT_CLEAN_DATASET,
    DEFAULT_DB,
    DEFAULT_REPAIRED_PACKET,
    DEFAULT_SCHEMA,
    load_json,
    sha256_file,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_shadow_non_tgr_rf_evaluation import (  # noqa: E402
    CALIBRATION_METHOD_KEY,
    SHADOW_OUTPUT_MODE,
    build_shadow_feature_matrix,
    inactive_feature_policy_report,
    output_file_manifest,
    protected_path_snapshot,
    protected_path_verification,
    shadow_relpath,
    train_eval_feature_parity_report,
)
from scripts.shadow_feature_audit_packet import (  # noqa: E402
    copy_shadow_feature_audit_reports,
    ensure_same_distance_history_provenance_report,
)
from utils.csv_metadata import load_safe_sidecar_target_metadata  # noqa: E402
from utils.runner_completeness import (  # noqa: E402
    normalise_runner_name,
    parse_runner_rows_from_csv,
)


DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
DEFAULT_INPUT_DIRS = (ROOT / "upcoming_races",)
DEFAULT_ALL_MISSING_TRAIN_POLICY = "quarantine_feature"
DEFAULT_SCORE_COMMAND_MODE = "auto"
# The locked shadow RandomForest artifact was pickled with sklearn 1.7.2.
# Floating scikit-learn breaks live scoring when uv resolves a newer version.
SHADOW_MODEL_SKLEARN_VERSION = "1.7.2"
UV_SCORE_LIVE_PACKAGES = ("joblib", f"scikit-learn=={SHADOW_MODEL_SKLEARN_VERSION}", "numpy")
DAILY_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_"
EXPECTED_OFFICIAL_RACES = 214
EXPECTED_OFFICIAL_DOG_ROWS = 1493
FINAL_STATUS_FORWARD_COMPLETE = "FORWARD_SHADOW_RUN_COMPLETE"
FINAL_STATUS_WAITING = "WAITING_FOR_UPCOMING_RACES"
FINAL_STATUS_MALFORMED = "BLOCKED_MALFORMED_INPUTS"
FINAL_STATUS_DB = "BLOCKED_DB_STATE"
FINAL_STATUS_RUN_FAILURE = "BLOCKED_SHADOW_RUN_FAILURE"
PREDICTION_COLUMNS = [
    "race_id",
    "dog_name",
    "box",
    "shadow_rf_uncalibrated_probability",
    "shadow_rf_calibrated_probability",
    "predicted_rank",
    "calibration_method",
    "model_version",
    "model_source",
    "tgr_enabled",
    "output_mode",
]
REQUIRED_PREJUMP_METADATA_FIELDS = (
    "race_date",
    "venue",
    "race_number",
    "jump_time",
    "metadata_captured_at",
    "target_distance",
    "target_grade",
    "source_url",
    "runner_box_name_list",
    "csv_sidecar_runner_identity",
    "canonical_final_runner_alignment",
    "canonical_runner_source_url",
)
AUXILIARY_REFRESH_DIR_NAMES = {"raw_exports", "quarantine"}
POST_RESULT_URL_MARKERS = {"result", "results", "dividend", "dividends", "payout", "payouts"}


def now_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


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


def assert_daily_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative_path = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative_path.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    relative = relative_path.as_posix()
    if not relative.startswith(DAILY_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_daily_shadow_artifact:{relative}")
    return logical.absolute()


def verify_db_state(db_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "daily_shadow_db_state_v1",
        "db_path": shadow_relpath(db_path),
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
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        try:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            race_count = connection.execute(
                "SELECT count(DISTINCT race_id) FROM race_metadata "
                "WHERE winner_source='thedogs_official'"
            ).fetchone()[0]
            dog_row_count = connection.execute(
                "SELECT count(*) FROM dog_race_data "
                "WHERE data_source='thedogs_official'"
            ).fetchone()[0]
        finally:
            connection.close()
    except Exception as exc:
        report["fail_reasons"].append(f"db_read_failed:{exc!r}")
        return report

    report.update(
        {
            "quick_check": quick_check,
            "official_races": int(race_count),
            "official_dog_rows": int(dog_row_count),
        }
    )
    if quick_check != "ok":
        report["fail_reasons"].append("quick_check_failed")
    if race_count != EXPECTED_OFFICIAL_RACES:
        report["fail_reasons"].append("official_race_count_mismatch")
    if dog_row_count != EXPECTED_OFFICIAL_DOG_ROWS:
        report["fail_reasons"].append("official_dog_row_count_mismatch")
    if not report["fail_reasons"]:
        report["status"] = "PASS"
    return report


def parse_date_value(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_int_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def is_thedogs_source_url(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    return host == "thedogs.com.au" or host.endswith(".thedogs.com.au")


def looks_post_result_source_url(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    try:
        parsed = urlparse(text)
        searchable = " ".join(
            part for part in (parsed.path, parsed.query, parsed.fragment) if part
        )
    except Exception:
        searchable = text
    tokens = {token for token in re.split(r"[^a-z0-9]+", searchable) if token}
    return bool(tokens.intersection(POST_RESULT_URL_MARKERS))


def parse_jump_datetime(
    *,
    race_date: date | None,
    jump_time: Any,
    current_time: datetime,
) -> tuple[datetime | None, str | None]:
    if race_date is None or jump_time in (None, ""):
        return None, "jump_time_missing"
    text = str(jump_time).strip()
    if not text:
        return None, "jump_time_missing"

    normalized = text
    if len(normalized) >= 5 and normalized[-5] in {"+", "-"} and normalized[-4:].isdigit():
        normalized = f"{normalized[:-5]}{normalized[-5:-2]}:{normalized[-2:]}"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None and current_time.tzinfo is not None:
            parsed = parsed.replace(tzinfo=current_time.tzinfo)
        return parsed, None

    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(text.upper(), fmt).time()
        except ValueError:
            continue
        return (
            datetime.combine(race_date, parsed_time).replace(tzinfo=current_time.tzinfo),
            None,
        )
    return None, "jump_time_unparseable"


def parse_sidecar_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def sidecar_metadata_capture_timestamp(
    payload: Mapping[str, Any],
    shadow_metadata: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    candidates = (
        ("prejump_shadow_metadata.metadata_captured_at", shadow_metadata.get("metadata_captured_at")),
        ("metadata_captured_at", payload.get("metadata_captured_at")),
        ("created_at", payload.get("created_at")),
        ("generated_at", payload.get("generated_at")),
        ("fetched_at", payload.get("fetched_at")),
        ("capture_timestamp", payload.get("capture_timestamp")),
    )
    for source, value in candidates:
        if value not in (None, ""):
            return str(value).strip(), source
    return None, None


def _sidecar_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    sidecar = path.with_name(path.name + ".metadata.json")
    if not sidecar.exists():
        return None, "sidecar_metadata_missing"
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"sidecar_metadata_unreadable:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "sidecar_metadata_not_object"
    return payload, None


def _runner_participants_from_sidecar(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    shadow_metadata = payload.get("prejump_shadow_metadata")
    if isinstance(shadow_metadata, Mapping) and isinstance(
        shadow_metadata.get("runner_box_name_list"), list
    ):
        return [
            dict(item)
            for item in shadow_metadata["runner_box_name_list"]
            if isinstance(item, Mapping)
        ]
    for key in ("runner_completeness_after_canonical_alignment", "runner_completeness"):
        section = payload.get(key)
        if isinstance(section, Mapping) and isinstance(section.get("participants"), list):
            return [dict(item) for item in section["participants"] if isinstance(item, Mapping)]
    return []


def _canonical_alignment_from_sidecar(payload: Mapping[str, Any]) -> dict[str, Any]:
    alignment = payload.get("canonical_runner_alignment")
    if not isinstance(alignment, Mapping):
        shadow_metadata = payload.get("prejump_shadow_metadata")
        if isinstance(shadow_metadata, Mapping):
            alignment = shadow_metadata.get("canonical_final_runner_alignment")
    if not isinstance(alignment, Mapping):
        alignment = {}
    return dict(alignment)


def _duplicate_values(values: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _runner_identity_by_box(
    participants: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for participant in participants:
        box = parse_int_value(participant.get("box_number") or participant.get("box"))
        dog_name = str(participant.get("dog_name") or participant.get("name") or "").strip()
        if box is None or not dog_name:
            continue
        output[box] = {
            "box_number": box,
            "dog_name": dog_name,
            "identity": normalise_runner_name(dog_name),
        }
    return output


def _csv_target_participants(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        rows = parse_runner_rows_from_csv(path)
    except Exception as exc:
        return [], [f"csv_target_runner_rows_unreadable:{type(exc).__name__}"]
    participants = [
        {"box_number": row.box_number, "dog_name": row.dog_name}
        for row in rows
    ]
    reasons: list[str] = []
    if not participants:
        reasons.append("csv_target_runner_rows_missing")
    duplicate_boxes = _duplicate_values([row["box_number"] for row in participants])
    if duplicate_boxes:
        reasons.append(
            "csv_target_runner_duplicate_boxes:"
            + ",".join(str(value) for value in duplicate_boxes)
        )
    duplicate_names = _duplicate_values(
        [normalise_runner_name(row["dog_name"]) for row in participants]
    )
    if duplicate_names:
        reasons.append("csv_target_runner_duplicate_dog_names:" + ",".join(duplicate_names))
    return participants, reasons


def _csv_sidecar_runner_identity_errors(
    *,
    csv_participants: Sequence[Mapping[str, Any]],
    sidecar_participants: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    csv_by_box = _runner_identity_by_box(csv_participants)
    sidecar_by_box = _runner_identity_by_box(sidecar_participants)
    csv_boxes = set(csv_by_box)
    sidecar_boxes = set(sidecar_by_box)
    missing_csv_boxes = sorted(sidecar_boxes - csv_boxes)
    extra_csv_boxes = sorted(csv_boxes - sidecar_boxes)
    name_mismatches = []
    for box in sorted(csv_boxes & sidecar_boxes):
        csv_row = csv_by_box[box]
        sidecar_row = sidecar_by_box[box]
        if csv_row["identity"] != sidecar_row["identity"]:
            name_mismatches.append(
                {
                    "box_number": box,
                    "csv_dog_name": csv_row["dog_name"],
                    "sidecar_dog_name": sidecar_row["dog_name"],
                    "csv_identity": csv_row["identity"],
                    "sidecar_identity": sidecar_row["identity"],
                }
            )

    errors: list[str] = []
    if missing_csv_boxes:
        errors.append(
            "runner_box_name_list_missing_csv_boxes:"
            + ",".join(str(value) for value in missing_csv_boxes)
        )
    if extra_csv_boxes:
        errors.append(
            "runner_box_name_list_extra_csv_boxes:"
            + ",".join(str(value) for value in extra_csv_boxes)
        )
    if name_mismatches:
        errors.append("runner_box_name_list_name_mismatch")
    if errors:
        errors.insert(0, "runner_box_name_list_does_not_match_csv_target_rows")
    return errors, {
        "missing_csv_boxes": missing_csv_boxes,
        "extra_csv_boxes": extra_csv_boxes,
        "name_mismatches": name_mismatches,
    }


def validate_prejump_sidecar_metadata(path: Path) -> dict[str, Any]:
    """Validate the pre-race sidecar fields needed for safe shadow scoring."""

    report: dict[str, Any] = {
        "schema_version": "daily_shadow_prejump_sidecar_metadata_v1",
        "sidecar_path": shadow_relpath(path.with_name(path.name + ".metadata.json")),
        "status": "FAIL",
        "fail_reasons": [],
        "metadata_is_leakage_safe": False,
        "target_distance": None,
        "target_distance_source": None,
        "target_grade": None,
        "target_grade_source": None,
        "race_date": None,
        "venue": None,
        "race_number": None,
        "jump_time": None,
        "metadata_captured_at": None,
        "metadata_capture_source": None,
        "metadata_capture_timing_status": None,
        "metadata_capture_seconds_before_jump": None,
        "source_url": None,
        "runner_count": 0,
        "participants": [],
        "csv_target_runner_count": 0,
        "csv_target_participants": [],
        "csv_sidecar_runner_identity_status": "FAIL",
        "csv_sidecar_runner_identity_mismatches": {},
        "canonical_runner_alignment_status": None,
        "canonical_runner_alignment_reason": None,
        "canonical_runner_set_status": None,
        "canonical_runner_count": None,
        "canonical_prediction_runner_count": None,
        "canonical_runner_source_url": None,
        "canonical_runner_alignment_verified": False,
        "remapped_participants": [],
        "dropped_participants": [],
    }
    payload, error = _sidecar_payload(path)
    if error:
        report["fail_reasons"].append(error)
        return report
    assert payload is not None

    shadow_metadata = (
        payload.get("prejump_shadow_metadata")
        if isinstance(payload.get("prejump_shadow_metadata"), Mapping)
        else {}
    )
    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    leakage_safe = payload.get("metadata_is_leakage_safe") is True or (
        shadow_metadata.get("status") == "PASS"
        and shadow_metadata.get("metadata_is_leakage_safe") is True
    )
    report["metadata_is_leakage_safe"] = leakage_safe
    if not leakage_safe:
        report["fail_reasons"].append("metadata_is_leakage_safe_not_true")

    safe_target = load_safe_sidecar_target_metadata(path)
    report["target_distance"] = safe_target.get("target_distance")
    report["target_distance_source"] = safe_target.get("target_distance_source")
    report["target_grade"] = safe_target.get("target_grade")
    report["target_grade_source"] = safe_target.get("target_grade_source")
    if not report["target_distance"]:
        report["fail_reasons"].append("target_distance_missing_or_unsafe")
    if not report["target_grade"]:
        report["fail_reasons"].append("target_grade_missing_or_unsafe")
    report["rejected_metadata_sources"] = safe_target.get("rejected_metadata_sources") or []

    race_date = (
        parse_date_value(shadow_metadata.get("race_date"))
        or parse_date_value(race_info.get("date"))
        or parse_date_value(payload.get("race_date"))
        or parse_date_value(payload.get("date"))
    )
    venue = shadow_metadata.get("venue") or race_info.get("venue") or payload.get("venue")
    race_number = parse_int_value(
        shadow_metadata.get("race_number") or race_info.get("race_number") or payload.get("race_number")
    )
    jump_time = (
        shadow_metadata.get("jump_time")
        or race_info.get("race_time")
        or race_info.get("jump_time")
        or payload.get("jump_time")
        or payload.get("jump_datetime")
    )
    source_url = (
        safe_target.get("metadata_source_url")
        or shadow_metadata.get("source_url")
        or payload.get("metadata_source_url")
        or payload.get("race_url")
        or race_info.get("url")
    )
    report["race_date"] = race_date.isoformat() if race_date else None
    report["venue"] = str(venue).strip().upper() if venue else None
    report["race_number"] = race_number
    report["jump_time"] = str(jump_time).strip() if jump_time not in (None, "") else None
    captured_at, capture_source = sidecar_metadata_capture_timestamp(payload, shadow_metadata)
    report["metadata_captured_at"] = captured_at
    report["metadata_capture_source"] = capture_source
    capture_dt = parse_sidecar_timestamp(captured_at)
    if not captured_at:
        report["fail_reasons"].append("metadata_captured_at_missing")
        report["metadata_capture_timing_status"] = "MISSING"
    elif capture_dt is None:
        report["fail_reasons"].append("metadata_captured_at_unparseable")
        report["metadata_capture_timing_status"] = "UNPARSEABLE"
    elif race_date and report["jump_time"]:
        jump_dt, jump_error = parse_jump_datetime(
            race_date=race_date,
            jump_time=report["jump_time"],
            current_time=datetime.now().astimezone(),
        )
        if jump_dt is not None:
            seconds_before_jump = (jump_dt - capture_dt).total_seconds()
            report["metadata_capture_seconds_before_jump"] = seconds_before_jump
            if seconds_before_jump <= 0:
                report["fail_reasons"].append("metadata_captured_at_not_before_jump")
                report["metadata_capture_timing_status"] = "AFTER_OR_AT_JUMP"
            else:
                report["metadata_capture_timing_status"] = "PRE_JUMP"
        elif jump_error:
            report["metadata_capture_timing_status"] = f"UNVERIFIED:{jump_error}"
            report["fail_reasons"].append(
                f"metadata_capture_timing_unverified:{jump_error}"
            )
    report["source_url"] = str(source_url).strip() if source_url not in (None, "") else None
    if not race_date:
        report["fail_reasons"].append("race_date_missing")
    if not report["venue"]:
        report["fail_reasons"].append("venue_missing")
    if race_number is None:
        report["fail_reasons"].append("race_number_missing")
    if not report["jump_time"]:
        report["fail_reasons"].append("jump_time_missing")
    if not report["source_url"]:
        report["fail_reasons"].append("source_url_missing")
    elif not is_thedogs_source_url(report["source_url"]):
        report["fail_reasons"].append("source_url_not_thedogs")
    elif looks_post_result_source_url(report["source_url"]):
        report["fail_reasons"].append("source_url_looks_post_result")

    participants = _runner_participants_from_sidecar(payload)
    valid_participants = []
    invalid_participants = []
    for participant in participants:
        box = parse_int_value(participant.get("box_number") or participant.get("box"))
        dog_name = str(participant.get("dog_name") or participant.get("name") or "").strip()
        if box is None or not dog_name:
            invalid_participants.append(participant)
            continue
        valid_participants.append({"box_number": box, "dog_name": dog_name})
    report["participants"] = valid_participants
    report["runner_count"] = len(valid_participants)
    if not valid_participants:
        report["fail_reasons"].append("runner_box_name_list_missing")
    if invalid_participants:
        report["fail_reasons"].append("runner_box_name_list_has_invalid_entries")
    sidecar_duplicate_boxes = _duplicate_values(
        [row["box_number"] for row in valid_participants]
    )
    if sidecar_duplicate_boxes:
        report["fail_reasons"].append(
            "runner_box_name_list_duplicate_boxes:"
            + ",".join(str(value) for value in sidecar_duplicate_boxes)
        )
    sidecar_duplicate_names = _duplicate_values(
        [normalise_runner_name(row["dog_name"]) for row in valid_participants]
    )
    if sidecar_duplicate_names:
        report["fail_reasons"].append(
            "runner_box_name_list_duplicate_dog_names:"
            + ",".join(sidecar_duplicate_names)
        )

    csv_participants, csv_runner_errors = _csv_target_participants(path)
    report["csv_target_participants"] = csv_participants
    report["csv_target_runner_count"] = len(csv_participants)
    report["fail_reasons"].extend(csv_runner_errors)
    identity_errors, identity_mismatches = _csv_sidecar_runner_identity_errors(
        csv_participants=csv_participants,
        sidecar_participants=valid_participants,
    )
    report["csv_sidecar_runner_identity_mismatches"] = identity_mismatches
    if identity_errors:
        report["fail_reasons"].extend(identity_errors)
    else:
        report["csv_sidecar_runner_identity_status"] = "PASS"

    alignment = _canonical_alignment_from_sidecar(payload)
    alignment_status = alignment.get("status")
    canonical_status = alignment.get("canonical_runner_set_status")
    report["canonical_runner_alignment_status"] = alignment_status
    report["canonical_runner_alignment_reason"] = alignment.get("reason")
    report["canonical_runner_set_status"] = canonical_status
    report["canonical_runner_count"] = parse_int_value(alignment.get("canonical_runner_count"))
    report["canonical_prediction_runner_count"] = parse_int_value(
        alignment.get("prediction_runner_count")
    )
    report["canonical_runner_source_url"] = (
        alignment.get("canonical_source_url")
        or alignment.get("canonical_runner_source_url")
        or alignment.get("canonical_runner_set_source_url")
        or alignment.get("source_url")
    )
    report["remapped_participants"] = [
        dict(row) for row in alignment.get("remapped_participants") or [] if isinstance(row, Mapping)
    ]
    report["dropped_participants"] = [
        dict(row) for row in alignment.get("dropped_participants") or [] if isinstance(row, Mapping)
    ]
    if not alignment:
        report["fail_reasons"].append("canonical_runner_alignment_missing")
    elif alignment_status != "aligned":
        report["fail_reasons"].append("canonical_runner_alignment_not_aligned")
    if canonical_status != "available":
        report["fail_reasons"].append("canonical_runner_set_not_available")
    if not report["canonical_runner_source_url"]:
        report["fail_reasons"].append("canonical_runner_source_url_missing")
    elif not is_thedogs_source_url(report["canonical_runner_source_url"]):
        report["fail_reasons"].append("canonical_runner_source_url_not_thedogs")
    elif looks_post_result_source_url(report["canonical_runner_source_url"]):
        report["fail_reasons"].append("canonical_runner_source_url_looks_post_result")
    if report["canonical_prediction_runner_count"] is not None and (
        report["runner_count"] != report["canonical_prediction_runner_count"]
    ):
        report["fail_reasons"].append("runner_count_mismatch_after_canonical_alignment")
    if not any(
        reason.startswith("canonical_runner_")
        or reason == "runner_count_mismatch_after_canonical_alignment"
        for reason in report["fail_reasons"]
    ):
        report["canonical_runner_alignment_verified"] = True

    if not report["fail_reasons"]:
        report["status"] = "PASS"
    return report


def race_date_from_sidecar(path: Path) -> date | None:
    sidecar = path.with_name(path.name + ".metadata.json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in (
        "race_date",
        "target_race_date",
        "meeting_date",
        "date",
    ):
        parsed = parse_date_value(data.get(key))
        if parsed is not None:
            return parsed
    target_metadata = data.get("target_metadata")
    if isinstance(target_metadata, Mapping):
        for key in ("race_date", "target_race_date", "meeting_date", "date"):
            parsed = parse_date_value(target_metadata.get(key))
            if parsed is not None:
                return parsed
    shadow_metadata = data.get("prejump_shadow_metadata")
    if isinstance(shadow_metadata, Mapping):
        for key in ("race_date", "target_race_date", "meeting_date", "date"):
            parsed = parse_date_value(shadow_metadata.get(key))
            if parsed is not None:
                return parsed
    race_info = data.get("race_info")
    if isinstance(race_info, Mapping):
        for key in ("date", "race_date", "target_race_date", "meeting_date"):
            parsed = parse_date_value(race_info.get(key))
            if parsed is not None:
                return parsed
    return None


def race_date_from_csv_first_row(path: Path) -> date | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.readline()
            if not sample:
                return None
            delimiter = "|" if "|" in sample else ","
            handle.seek(0)
            reader = csv.DictReader(handle, delimiter=delimiter)
            row = next(reader, None)
    except Exception:
        return None
    if not row:
        return None
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for key in (
        "race_date",
        "target_race_date",
        "meeting_date",
        "date",
    ):
        parsed = parse_date_value(lowered.get(key))
        if parsed is not None:
            return parsed
    return None


def extract_race_date(path: Path) -> tuple[date | None, str | None]:
    _race_no, _venue, race_date, _problems = validate_upcoming_races.validate_filename(path)
    if race_date is not None:
        return race_date, "filename"
    sidecar_date = race_date_from_sidecar(path)
    if sidecar_date is not None:
        return sidecar_date, "metadata_sidecar"
    row_date = race_date_from_csv_first_row(path)
    if row_date is not None:
        return row_date, "csv_first_row"
    return None, None


def validate_candidate_structure(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    _race_no, _venue, _race_date, filename_problems = validate_upcoming_races.validate_filename(path)
    errors.extend(filename_problems)
    try:
        header_problems = validate_upcoming_races.iter_csv_rows(path.resolve())
    except Exception as exc:
        errors.append(str(exc))
    else:
        for problem in header_problems:
            if "WARNING:" in problem:
                warnings.append(problem)
            else:
                errors.append(problem)
    return errors, warnings


def discover_csv_files(input_dirs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            continue
        if directory.is_file() and directory.suffix.lower() == ".csv":
            files.append(directory)
            continue
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*.csv")
                if path.is_file()
                and not path.name.startswith(".")
                and not AUXILIARY_REFRESH_DIR_NAMES.intersection(path.relative_to(directory).parts[:-1])
            )
    return sorted({path.resolve() for path in files}, key=lambda item: item.as_posix())


def file_record(path: Path, **extra: Any) -> dict[str, Any]:
    record = {
        "path": shadow_relpath(path),
        "source_path": str(path.resolve()),
        "basename": path.name,
    }
    record.update(extra)
    return record


def classify_candidate_csvs(
    input_dirs: Sequence[Path],
    current_date: date,
    *,
    current_time: datetime | None = None,
    require_prejump_sidecar_metadata: bool = True,
) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    files = discover_csv_files(input_dirs)

    for path in files:
        race_date, date_source = extract_race_date(path)
        if race_date is None:
            malformed.append(
                file_record(
                    path,
                    reason="race_date_not_found",
                    race_date=None,
                    date_source=None,
                    errors=["race date not found in filename, metadata sidecar, or CSV first row"],
                )
            )
            continue
        if race_date < current_date:
            stale.append(
                file_record(
                    path,
                    reason="stale_before_current_date",
                    race_date=race_date.isoformat(),
                    date_source=date_source,
                )
            )
            continue
        errors, structure_warnings = validate_candidate_structure(path)
        if errors:
            malformed.append(
                file_record(
                    path,
                    reason="malformed_current_or_future_csv",
                    race_date=race_date.isoformat(),
                    date_source=date_source,
                    errors=errors,
                    warnings=structure_warnings,
                )
            )
            continue
        sidecar_report = validate_prejump_sidecar_metadata(path)
        if require_prejump_sidecar_metadata and sidecar_report["status"] != "PASS":
            malformed.append(
                file_record(
                    path,
                    reason="prejump_sidecar_metadata_failed",
                    race_date=race_date.isoformat(),
                    date_source=date_source,
                    errors=sidecar_report["fail_reasons"],
                    warnings=structure_warnings,
                    sidecar_metadata_report=sidecar_report,
                )
            )
            continue
        if current_time is not None and race_date == current_date:
            jump_datetime, jump_error = parse_jump_datetime(
                race_date=race_date,
                jump_time=sidecar_report.get("jump_time"),
                current_time=current_time,
            )
            if jump_error:
                malformed.append(
                    file_record(
                        path,
                        reason="prejump_sidecar_jump_time_failed",
                        race_date=race_date.isoformat(),
                        date_source=date_source,
                        errors=[jump_error],
                        warnings=structure_warnings,
                        sidecar_metadata_report=sidecar_report,
                    )
                )
                continue
            if jump_datetime is not None and jump_datetime <= current_time:
                stale.append(
                    file_record(
                        path,
                        reason="stale_after_jump_time",
                        race_date=race_date.isoformat(),
                        date_source=date_source,
                        jump_datetime=jump_datetime.isoformat(),
                        current_time=current_time.isoformat(),
                        sidecar_metadata_report=sidecar_report,
                    )
                )
                continue
        record = file_record(
            path,
            race_date=race_date.isoformat(),
            date_source=date_source,
            warnings=structure_warnings,
            sidecar_metadata_report=sidecar_report,
        )
        eligible.append(record)
        if structure_warnings:
            warnings.append(record)

    return {
        "schema_version": "daily_shadow_input_classification_v1",
        "current_date": current_date.isoformat(),
        "current_time": current_time.isoformat() if current_time else None,
        "input_dirs": [shadow_relpath(path) for path in input_dirs],
        "prejump_sidecar_metadata_required": require_prejump_sidecar_metadata,
        "scanned_csv_count": len(files),
        "eligible_count": len(eligible),
        "stale_count": len(stale),
        "malformed_count": len(malformed),
        "warning_count": len(warnings),
        "eligible": eligible,
        "stale": stale,
        "malformed": malformed,
        "warnings": warnings,
    }


def _metadata_row_from_record(record: Mapping[str, Any], *, bucket: str) -> dict[str, Any]:
    sidecar_report = record.get("sidecar_metadata_report")
    if not isinstance(sidecar_report, Mapping):
        sidecar_report = {}
    return {
        "bucket": bucket,
        "path": record.get("path"),
        "basename": record.get("basename"),
        "reason": record.get("reason"),
        "sidecar_status": sidecar_report.get("status"),
        "fail_reasons": list(sidecar_report.get("fail_reasons") or []),
        "metadata_is_leakage_safe": sidecar_report.get("metadata_is_leakage_safe") is True,
        "race_date": sidecar_report.get("race_date"),
        "venue": sidecar_report.get("venue"),
        "race_number": sidecar_report.get("race_number"),
        "jump_time": sidecar_report.get("jump_time"),
        "metadata_captured_at": sidecar_report.get("metadata_captured_at"),
        "metadata_capture_source": sidecar_report.get("metadata_capture_source"),
        "metadata_capture_timing_status": sidecar_report.get(
            "metadata_capture_timing_status"
        ),
        "metadata_capture_seconds_before_jump": sidecar_report.get(
            "metadata_capture_seconds_before_jump"
        ),
        "target_distance": sidecar_report.get("target_distance"),
        "target_distance_source": sidecar_report.get("target_distance_source"),
        "target_grade": sidecar_report.get("target_grade"),
        "target_grade_source": sidecar_report.get("target_grade_source"),
        "source_url": sidecar_report.get("source_url"),
        "runner_count": int(sidecar_report.get("runner_count") or 0),
        "canonical_runner_alignment_status": sidecar_report.get(
            "canonical_runner_alignment_status"
        ),
        "canonical_runner_set_status": sidecar_report.get("canonical_runner_set_status"),
        "canonical_prediction_runner_count": sidecar_report.get(
            "canonical_prediction_runner_count"
        ),
        "canonical_runner_count": sidecar_report.get("canonical_runner_count"),
        "canonical_runner_source_url": sidecar_report.get("canonical_runner_source_url"),
        "csv_target_runner_count": sidecar_report.get("csv_target_runner_count"),
        "csv_sidecar_runner_identity_status": sidecar_report.get(
            "csv_sidecar_runner_identity_status"
        ),
        "csv_sidecar_runner_identity_verified": (
            sidecar_report.get("csv_sidecar_runner_identity_status") == "PASS"
        ),
        "canonical_runner_alignment_verified": (
            sidecar_report.get("canonical_runner_alignment_verified") is True
        ),
        "remapped_participant_count": len(sidecar_report.get("remapped_participants") or []),
        "dropped_participant_count": len(sidecar_report.get("dropped_participants") or []),
        "rejected_metadata_sources": list(sidecar_report.get("rejected_metadata_sources") or []),
    }


def _metadata_required_field_present(row: Mapping[str, Any], field: str) -> bool:
    if field == "runner_box_name_list":
        return int(row.get("runner_count") or 0) > 0
    if field == "csv_sidecar_runner_identity":
        return row.get("csv_sidecar_runner_identity_verified") is True
    if field == "canonical_final_runner_alignment":
        return row.get("canonical_runner_alignment_verified") is True
    if field == "canonical_runner_source_url":
        return is_thedogs_source_url(row.get("canonical_runner_source_url")) and not (
            looks_post_result_source_url(row.get("canonical_runner_source_url"))
        )
    return row.get(field) not in (None, "")


def target_metadata_readiness_report(
    *,
    eligible_rows: Sequence[Mapping[str, Any]],
    malformed_rows: Sequence[Mapping[str, Any]],
    stale_rows: Sequence[Mapping[str, Any]],
    verified_eligible_rows: Sequence[Mapping[str, Any]],
    rejected_metadata_sources: Sequence[str],
) -> dict[str, Any]:
    current_or_future_rows = list(eligible_rows) + list(malformed_rows)
    blocker_counts: Counter[str] = Counter()
    missing_required_field_counts: Counter[str] = Counter()
    blocked_rows: list[dict[str, Any]] = []

    for row in current_or_future_rows:
        row_blockers = [str(reason) for reason in row.get("fail_reasons") or []]
        for field in REQUIRED_PREJUMP_METADATA_FIELDS:
            if not _metadata_required_field_present(row, field):
                missing_required_field_counts[field] += 1
                if row.get("bucket") == "eligible":
                    row_blockers.append(f"{field}_missing_from_eligible_sidecar")
        if row.get("bucket") == "eligible":
            if row.get("metadata_is_leakage_safe") is not True:
                row_blockers.append("metadata_is_leakage_safe_not_true")
            if not is_thedogs_source_url(row.get("source_url")):
                row_blockers.append("source_url_not_thedogs")
            if row.get("csv_sidecar_runner_identity_verified") is not True:
                row_blockers.append("csv_sidecar_runner_identity_not_verified")
            if row.get("canonical_runner_alignment_verified") is not True:
                row_blockers.append("canonical_runner_alignment_not_verified")

        deduped_row_blockers = list(dict.fromkeys(row_blockers))
        for blocker in deduped_row_blockers:
            blocker_counts[blocker] += 1
        if deduped_row_blockers:
            blocked_rows.append(
                {
                    "bucket": row.get("bucket"),
                    "path": row.get("path"),
                    "basename": row.get("basename"),
                    "sidecar_status": row.get("sidecar_status"),
                    "race_date": row.get("race_date"),
                    "venue": row.get("venue"),
                    "race_number": row.get("race_number"),
                    "blockers": deduped_row_blockers,
                }
            )

    current_or_future_count = len(current_or_future_rows)
    verified_count = len(verified_eligible_rows)
    if current_or_future_count == 0:
        status = "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
        capture_status = "WAITING"
    elif malformed_rows or verified_count != len(eligible_rows) or blocker_counts:
        status = "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS"
        capture_status = "BLOCKED"
    else:
        status = "TARGET_METADATA_READY_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
        capture_status = "READY"

    return {
        "schema_version": "daily_shadow_target_metadata_readiness_v1",
        "status": status,
        "target_metadata_capture_status": capture_status,
        "current_or_future_input_count": current_or_future_count,
        "eligible_count": len(eligible_rows),
        "verified_eligible_count": verified_count,
        "malformed_prejump_metadata_count": len(malformed_rows),
        "stale_with_prejump_metadata_count": len(stale_rows),
        "all_current_future_inputs_verified": (
            current_or_future_count > 0
            and capture_status == "READY"
        ),
        "required_fields": list(REQUIRED_PREJUMP_METADATA_FIELDS),
        "missing_required_field_counts": dict(sorted(missing_required_field_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "blocked_rows": blocked_rows,
        "rejected_metadata_sources": list(rejected_metadata_sources),
        "future_train_row_target_metadata_status": (
            "PRE_RACE_SIDECAR_SAFE"
            if capture_status == "READY"
            else capture_status
        ),
        "historical_repair_policy": "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE",
        "no_write_guarantees": {
            "db_write": False,
            "label_write": False,
            "canonical_schema_mutation": False,
            "production_prediction_write": False,
            "betting_or_ev_output": False,
        },
    }


def prejump_metadata_report_from_classification(
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    eligible_rows = [
        _metadata_row_from_record(record, bucket="eligible")
        for record in classification.get("eligible") or []
    ]
    malformed_rows = [
        _metadata_row_from_record(record, bucket="malformed")
        for record in classification.get("malformed") or []
        if str(record.get("reason") or "").startswith("prejump_sidecar")
    ]
    stale_rows = [
        _metadata_row_from_record(record, bucket="stale")
        for record in classification.get("stale") or []
        if isinstance(record.get("sidecar_metadata_report"), Mapping)
    ]
    all_rows = eligible_rows + malformed_rows + stale_rows

    field_coverage: dict[str, dict[str, Any]] = {}
    eligible_count = len(eligible_rows)
    for field in REQUIRED_PREJUMP_METADATA_FIELDS:
        present = sum(1 for row in eligible_rows if _metadata_required_field_present(row, field))
        field_coverage[field] = {
            "eligible_present_rows": present,
            "eligible_present_pct": present / eligible_count if eligible_count else None,
        }

    rejected_sources = sorted(
        {
            str(source)
            for row in all_rows
            for source in row.get("rejected_metadata_sources", [])
            if source not in (None, "")
        }
    )
    unsafe_or_incomplete = [
        row
        for row in malformed_rows
        if row.get("fail_reasons")
    ]
    verified_eligible = [
        row
        for row in eligible_rows
        if row.get("sidecar_status") == "PASS"
        and row.get("metadata_is_leakage_safe") is True
        and is_thedogs_source_url(row.get("source_url"))
        and row.get("csv_sidecar_runner_identity_verified") is True
        and row.get("canonical_runner_alignment_verified") is True
    ]
    status = "PASS"
    if eligible_count and len(verified_eligible) != eligible_count:
        status = "FAIL"
    if malformed_rows:
        status = "FAIL"
    target_readiness = target_metadata_readiness_report(
        eligible_rows=eligible_rows,
        malformed_rows=malformed_rows,
        stale_rows=stale_rows,
        verified_eligible_rows=verified_eligible,
        rejected_metadata_sources=rejected_sources,
    )
    return {
        "schema_version": "daily_shadow_prejump_metadata_report_v1",
        "status": status,
        "required_fields": list(REQUIRED_PREJUMP_METADATA_FIELDS),
        "eligible_count": eligible_count,
        "eligible_with_verified_prejump_metadata": len(verified_eligible),
        "malformed_prejump_metadata_count": len(malformed_rows),
        "stale_with_prejump_metadata_count": len(stale_rows),
        "field_coverage": field_coverage,
        "target_metadata_readiness": target_readiness,
        "rejected_metadata_sources": rejected_sources,
        "unsafe_or_incomplete_metadata": unsafe_or_incomplete,
        "files": all_rows,
    }


def stage_eligible_inputs(classification: Mapping[str, Any], stage_dir: Path) -> list[Path]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for index, record in enumerate(classification.get("eligible") or [], start=1):
        source = Path(str(record.get("source_path") or (ROOT / str(record["path"])))).resolve()
        target_dir = stage_dir / f"source_{index:04d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        sidecar = source.with_name(source.name + ".metadata.json")
        if sidecar.exists():
            shutil.copy2(sidecar, target.with_name(target.name + ".metadata.json"))
        staged.append(target)
    return staged


def shadow_ml_dependencies_available() -> bool:
    try:
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def score_live_command_prefix(mode: str = DEFAULT_SCORE_COMMAND_MODE) -> list[str]:
    if mode not in {"auto", "python", "uv"}:
        raise ValueError(f"unknown_score_command_mode:{mode}")
    if mode == "python" or (mode == "auto" and shadow_ml_dependencies_available()):
        return [sys.executable]
    uv_path = shutil.which("uv")
    if uv_path:
        command = [uv_path, "run"]
        for package in UV_SCORE_LIVE_PACKAGES:
            command.extend(["--with", package])
        command.append("python")
        return command
    return [sys.executable]


def build_score_live_command(
    *,
    input_dir: Path,
    output_dir: Path,
    db_path: Path,
    schema_path: Path,
    clean_dataset: Path,
    repaired_packet: Path,
    all_missing_train_policy: str,
    shadow_model: Path | None = None,
    score_command_mode: str = DEFAULT_SCORE_COMMAND_MODE,
) -> list[str]:
    command = [
        *score_live_command_prefix(score_command_mode),
        str(ROOT / "scripts/run_shadow_non_tgr_rf_evaluation.py"),
        "score-live",
        "--input",
        str(input_dir),
    ]
    if shadow_model is not None:
        command.extend(["--model", str(shadow_model)])
    else:
        command.append("--train-if-missing")
    command.extend(
        [
            "--all-missing-train-policy",
            all_missing_train_policy,
            "--db",
            str(db_path),
            "--schema",
            str(schema_path),
            "--clean-dataset",
            str(clean_dataset),
            "--repaired-packet",
            str(repaired_packet),
            "--output-dir",
            str(output_dir),
        ]
    )
    return command


def score_live_subprocess_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    pythonpath = env.get("PYTHONPATH")
    if not pythonpath:
        return env
    filtered = []
    root = ROOT.resolve()
    for entry in pythonpath.split(os.pathsep):
        if not entry:
            continue
        try:
            if Path(entry).resolve() == root:
                continue
        except OSError:
            pass
        filtered.append(entry)
    if filtered:
        env["PYTHONPATH"] = os.pathsep.join(filtered)
    else:
        env.pop("PYTHONPATH", None)
    return env


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_prediction_rows(score_output_dir: Path) -> list[dict[str, Any]]:
    predictions_path = score_output_dir / "shadow_predictions.json"
    if not predictions_path.exists():
        return []
    data = load_json(predictions_path)
    if not isinstance(data, list):
        return []
    return [dict(row) for row in data if isinstance(row, Mapping)]


def probability_sum_report_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("race_id") or "")].append(row)
    per_race = []
    max_abs_error = 0.0
    for race_id, rows in sorted(grouped.items()):
        total = sum(float(row.get("shadow_rf_calibrated_probability") or 0.0) for row in rows)
        abs_error = abs(1.0 - total)
        max_abs_error = max(max_abs_error, abs_error)
        per_race.append(
            {
                "race_id": race_id,
                "runner_count": len(rows),
                "sum": total,
                "abs_error": abs_error,
            }
        )
    return {
        "schema_version": "daily_shadow_probability_sum_report_v1",
        "probability_key": "shadow_rf_calibrated_probability",
        "race_count": len(grouped),
        "prediction_rows": len(predictions),
        "max_abs_error": max_abs_error,
        "status": "PASS" if max_abs_error <= 1e-6 else "FAIL",
        "per_race": per_race,
    }


def box_distribution_report_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    top_pick_boxes = Counter()
    all_boxes = Counter()
    for row in predictions:
        box = str(row.get("box") or "missing")
        all_boxes[box] += 1
        try:
            rank = int(row.get("predicted_rank"))
        except (TypeError, ValueError):
            rank = None
        if rank == 1:
            top_pick_boxes[box] += 1
    top_pick_count = sum(top_pick_boxes.values())
    return {
        "schema_version": "daily_shadow_box_distribution_report_v1",
        "prediction_rows": len(predictions),
        "top_pick_count": top_pick_count,
        "box1_top_pick_share": (
            top_pick_boxes.get("1", 0) / top_pick_count if top_pick_count else None
        ),
        "top_pick_box_counts": dict(sorted(top_pick_boxes.items())),
        "all_prediction_box_counts": dict(sorted(all_boxes.items())),
    }


def pending_results_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    race_ids = sorted({str(row.get("race_id") or "") for row in predictions if row.get("race_id")})
    return {
        "schema_version": "daily_shadow_pending_results_v1",
        "result_join_performed": False,
        "pending_race_count": len(race_ids),
        "pending_results": [
            {"race_id": race_id, "status": "PENDING_OFFICIAL_OUTCOME"}
            for race_id in race_ids
        ],
    }


def write_empty_prediction_outputs(output_dir: Path) -> None:
    write_jsonl(output_dir / "shadow_predictions.jsonl", [])
    write_csv(output_dir / "shadow_predictions.csv", [], PREDICTION_COLUMNS)


def write_matrix_reports_for_waiting(
    *,
    output_dir: Path,
    clean_dataset: Path,
    repaired_packet: Path,
    schema_path: Path,
    db_path: Path,
    all_missing_train_policy: str,
) -> None:
    dataset, feature_audit, population = build_shadow_feature_matrix(
        clean_dataset=clean_dataset,
        repaired_packet=repaired_packet,
        schema_path=schema_path,
        db_path=db_path,
    )
    parity = train_eval_feature_parity_report(dataset, policy=all_missing_train_policy)
    policy = inactive_feature_policy_report(parity)
    write_json(output_dir / "feature_population_report.json", population)
    write_json(output_dir / "shadow_feature_matrix_audit.json", feature_audit)
    write_json(output_dir / "train_eval_feature_parity_report.json", parity)
    write_json(output_dir / "inactive_feature_policy_report.json", policy)


def write_manifest(
    *,
    output_dir: Path,
    generated_at: datetime,
    mode: str,
    db_report: Mapping[str, Any],
    classification: Mapping[str, Any],
    protected: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    score_output_dir: Path | None,
    final_status: str,
    all_missing_train_policy: str,
    shadow_model: Path | None,
) -> None:
    score_manifest = None
    if score_output_dir is not None and (score_output_dir / "shadow_manifest.json").exists():
        score_manifest = load_json(score_output_dir / "shadow_manifest.json")
    score_manifest_data = score_manifest if isinstance(score_manifest, Mapping) else {}
    metadata_report = prejump_metadata_report_from_classification(classification)
    manifest = {
        "schema_version": "daily_shadow_manifest_v1",
        "generated_at": generated_at.isoformat(),
        "mode": mode,
        "final_status": final_status,
        "output_dir": shadow_relpath(output_dir),
        "db_status": db_report.get("status"),
        "db_quick_check": db_report.get("quick_check"),
        "official_races": db_report.get("official_races"),
        "official_dog_rows": db_report.get("official_dog_rows"),
        "input_summary": {
            "scanned_csv_count": classification.get("scanned_csv_count", 0),
            "eligible_count": classification.get("eligible_count", 0),
            "stale_count": classification.get("stale_count", 0),
            "malformed_count": classification.get("malformed_count", 0),
        },
        "prejump_metadata_summary": {
            "status": metadata_report.get("status"),
            "eligible_with_verified_prejump_metadata": metadata_report.get(
                "eligible_with_verified_prejump_metadata"
            ),
            "target_metadata_readiness_status": (
                metadata_report.get("target_metadata_readiness") or {}
            ).get("status"),
            "target_metadata_capture_status": (
                metadata_report.get("target_metadata_readiness") or {}
            ).get("target_metadata_capture_status"),
        },
        "prediction_rows": len(predictions),
        "feature_freeze_timestamp": score_manifest_data.get("feature_freeze_timestamp"),
        "prediction_timestamp": (
            score_manifest_data.get("prediction_timestamp")
            or score_manifest_data.get("generated_at")
        ),
        "race_count": len({row.get("race_id") for row in predictions if row.get("race_id")}),
        "all_missing_train_policy": all_missing_train_policy,
        "shadow_model": shadow_relpath(shadow_model) if shadow_model else None,
        "shadow_training_allowed": shadow_model is None,
        "calibration_method": CALIBRATION_METHOD_KEY,
        "tgr_enabled": False,
        "output_mode": SHADOW_OUTPUT_MODE,
        "production_promotion": False,
        "registry_mutation": False,
        "production_pointer_update": False,
        "active_model_replacement": False,
        "db_writes": False,
        "label_writes": False,
        "production_prediction_overwrite": False,
        "betting_action": False,
        "ev_action": False,
        "protected_paths_unchanged": protected.get("protected_paths_unchanged"),
        "score_live_manifest": score_manifest,
    }
    write_json(output_dir / "shadow_manifest.json", manifest)


def write_summary(output_dir: Path, final_status: str, classification: Mapping[str, Any]) -> None:
    metadata_report = prejump_metadata_report_from_classification(classification)
    lines = [
        "# Daily Race Ingest Shadow Run",
        "",
        f"Final status: `{final_status}`",
        "",
        f"Scanned CSVs: `{classification.get('scanned_csv_count', 0)}`",
        f"Eligible current/future CSVs: `{classification.get('eligible_count', 0)}`",
        f"Stale CSVs: `{classification.get('stale_count', 0)}`",
        f"Malformed CSVs: `{classification.get('malformed_count', 0)}`",
        f"Pre-jump metadata status: `{metadata_report.get('status')}`",
        f"Eligible with verified pre-jump metadata: `{metadata_report.get('eligible_with_verified_prejump_metadata')}`",
        f"Target metadata readiness: `{(metadata_report.get('target_metadata_readiness') or {}).get('status')}`",
        f"Target metadata blockers: `{(metadata_report.get('target_metadata_readiness') or {}).get('blocker_counts')}`",
        "",
        "No production promotion, registry mutation, DB writes, label writes, TGR enablement, betting output, EV output, or production prediction overwrite was performed.",
        "",
    ]
    write_text(output_dir / "SUMMARY.md", "\n".join(lines))


def write_common_reports(
    *,
    output_dir: Path,
    final_status: str,
    classification: Mapping[str, Any],
    db_report: Mapping[str, Any],
    protected: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    score_output_dir: Path | None,
    generated_at: datetime,
    mode: str,
    all_missing_train_policy: str,
    shadow_model: Path | None,
) -> None:
    metadata_report = prejump_metadata_report_from_classification(classification)
    write_json(output_dir / "malformed_or_stale_inputs.json", classification)
    write_json(output_dir / "prejump_metadata_report.json", metadata_report)
    ensure_same_distance_history_provenance_report(
        output_dir=output_dir,
        score_output_dir=score_output_dir,
    )
    write_json(output_dir / "db_recovery_verification.json", db_report)
    write_json(output_dir / "protected_path_verification.json", protected)
    write_json(output_dir / "probability_sum_report.json", probability_sum_report_from_predictions(predictions))
    write_json(output_dir / "box_distribution_report.json", box_distribution_report_from_predictions(predictions))
    write_json(output_dir / "pending_results.json", pending_results_from_predictions(predictions))
    write_manifest(
        output_dir=output_dir,
        generated_at=generated_at,
        mode=mode,
        db_report=db_report,
        classification=classification,
        protected=protected,
        predictions=predictions,
        score_output_dir=score_output_dir,
        final_status=final_status,
        all_missing_train_policy=all_missing_train_policy,
        shadow_model=shadow_model,
    )
    write_summary(output_dir, final_status, classification)
    write_text(output_dir / "final_status.txt", final_status + "\n")
    write_text(
        output_dir / "verification_results.txt",
        "\n".join(
            [
                f"db_status={db_report.get('status')}",
                f"quick_check={db_report.get('quick_check')}",
                f"official_races={db_report.get('official_races')}",
                f"official_dog_rows={db_report.get('official_dog_rows')}",
                f"protected_paths_unchanged={protected.get('protected_paths_unchanged')}",
                f"final_status={final_status}",
                "",
            ]
        ),
    )
    write_json(output_dir / "evidence_manifest.json", output_file_manifest(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("discover-only", "shadow-only", "join-results", "full-dry-run"),
        default="full-dry-run",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        default=None,
        help="Upcoming/pre-jump CSV directory. May be repeated.",
    )
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--current-time", default=None)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN_DATASET)
    parser.add_argument("--repaired-packet", type=Path, default=DEFAULT_REPAIRED_PACKET)
    parser.add_argument(
        "--all-missing-train-policy",
        choices=("quarantine_feature",),
        default=DEFAULT_ALL_MISSING_TRAIN_POLICY,
    )
    parser.add_argument(
        "--score-command-mode",
        choices=("auto", "python", "uv"),
        default=DEFAULT_SCORE_COMMAND_MODE,
        help="How to launch the ML scorer. auto falls back to uv when this Python lacks ML deps.",
    )
    parser.add_argument(
        "--shadow-model",
        type=Path,
        default=None,
        help="Existing shadow RandomForest joblib to reuse. When set, live scoring does not train.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    current_time = parse_current_time(args.current_time)
    input_dirs = [path.resolve() for path in (args.input_dir or DEFAULT_INPUT_DIRS)]
    output_dir = (
        args.output_dir
        if args.output_dir
        else (args.output_parent / f"daily_race_ingest_shadow_{now_id()}")
    )
    output_dir = assert_daily_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    protected_before = protected_path_snapshot()
    db_report = verify_db_state(args.db)
    classification: dict[str, Any] = {
        "schema_version": "daily_shadow_input_classification_v1",
        "current_date": current_time.date().isoformat(),
        "input_dirs": [shadow_relpath(path) for path in input_dirs],
        "scanned_csv_count": 0,
        "eligible_count": 0,
        "stale_count": 0,
        "malformed_count": 0,
        "eligible": [],
        "stale": [],
        "malformed": [],
        "warnings": [],
    }
    predictions: list[dict[str, Any]] = []
    score_output_dir: Path | None = None
    final_status = FINAL_STATUS_RUN_FAILURE
    return_code = 2

    try:
        if db_report["status"] != "PASS":
            final_status = FINAL_STATUS_DB
            write_empty_prediction_outputs(output_dir)
        else:
            classification = classify_candidate_csvs(
                input_dirs,
                current_time.date(),
                current_time=current_time,
            )
            write_json(output_dir / "malformed_or_stale_inputs.json", classification)
            write_empty_prediction_outputs(output_dir)

        if db_report["status"] == "PASS" and args.mode == "join-results":
            final_status = FINAL_STATUS_WAITING
            return_code = 0
        elif db_report["status"] == "PASS" and args.mode == "discover-only":
            if classification["malformed_count"] and not classification["eligible_count"]:
                final_status = FINAL_STATUS_MALFORMED
                return_code = 2
            else:
                final_status = FINAL_STATUS_WAITING
                return_code = 0
        elif db_report["status"] == "PASS" and not classification["eligible"]:
            if classification["malformed_count"] and not classification["stale_count"]:
                final_status = FINAL_STATUS_MALFORMED
                return_code = 2
            else:
                if args.mode == "full-dry-run":
                    write_matrix_reports_for_waiting(
                        output_dir=output_dir,
                        clean_dataset=args.clean_dataset,
                        repaired_packet=args.repaired_packet,
                        schema_path=args.schema,
                        db_path=args.db,
                        all_missing_train_policy=args.all_missing_train_policy,
                    )
                final_status = FINAL_STATUS_WAITING
                return_code = 0
        elif db_report["status"] == "PASS":
            stage_dir = output_dir / "eligible_inputs"
            stage_eligible_inputs(classification, stage_dir)
            score_output_dir = output_dir / "shadow_score_live"
            command = build_score_live_command(
                input_dir=stage_dir,
                output_dir=score_output_dir,
                db_path=args.db,
                schema_path=args.schema,
                clean_dataset=args.clean_dataset,
                repaired_packet=args.repaired_packet,
                all_missing_train_policy=args.all_missing_train_policy,
                shadow_model=args.shadow_model,
                score_command_mode=args.score_command_mode,
            )
            write_json(
                output_dir / "shadow_score_live_command.json",
                {
                    "schema_version": "daily_shadow_score_live_command_v1",
                    "command": command,
                    "cwd": str(ROOT),
                },
            )
            env = score_live_subprocess_env()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            write_json(
                output_dir / "shadow_score_live_process.json",
                {
                    "schema_version": "daily_shadow_score_live_process_v1",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
            if completed.returncode != 0:
                final_status = FINAL_STATUS_RUN_FAILURE
                return_code = 2
            else:
                predictions = read_prediction_rows(score_output_dir)
                write_jsonl(output_dir / "shadow_predictions.jsonl", predictions)
                if (score_output_dir / "shadow_predictions.csv").exists():
                    shutil.copy2(
                        score_output_dir / "shadow_predictions.csv",
                        output_dir / "shadow_predictions.csv",
                    )
                else:
                    write_csv(output_dir / "shadow_predictions.csv", predictions, PREDICTION_COLUMNS)
                copy_shadow_feature_audit_reports(score_output_dir, output_dir)

                probability_report = probability_sum_report_from_predictions(predictions)
                if probability_report["status"] != "PASS":
                    final_status = FINAL_STATUS_RUN_FAILURE
                    return_code = 2
                else:
                    final_status = FINAL_STATUS_FORWARD_COMPLETE
                    return_code = 0
        else:
            return_code = 2
    except Exception as exc:
        write_json(
            output_dir / "daily_shadow_runtime_error.json",
            {"schema_version": "daily_shadow_runtime_error_v1", "error": repr(exc)},
        )
        final_status = FINAL_STATUS_RUN_FAILURE if final_status == FINAL_STATUS_RUN_FAILURE else final_status
        return_code = 2
    finally:
        protected = protected_path_verification(protected_before)
        if not protected.get("protected_paths_unchanged"):
            final_status = FINAL_STATUS_RUN_FAILURE
            return_code = 2
        write_common_reports(
            output_dir=output_dir,
            final_status=final_status,
            classification=classification,
            db_report=db_report,
            protected=protected,
            predictions=predictions,
            score_output_dir=score_output_dir,
            generated_at=current_time,
            mode=args.mode,
            all_missing_train_policy=args.all_missing_train_policy,
            shadow_model=args.shadow_model,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
