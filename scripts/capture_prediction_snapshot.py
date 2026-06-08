#!/usr/bin/env python3
"""Capture durable, result-free pre-jump prediction snapshots.

Default mode is dry-run. Use --persist to write JSON snapshots under the local
snapshot directory. The script never scrapes by default and refuses to persist
non-live lifecycle states.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.snapshots import (  # noqa: E402
    assert_no_result_fields,
    build_prediction_snapshot,
    persist_prediction_snapshot,
)
from utils.runner_completeness import (  # noqa: E402
    analyze_csv_runner_completeness,
    analyze_csv_text_runner_completeness,
    align_csv_text_to_canonical_final_runner_set,
    canonical_race_url_from_sidecar,
    fetch_canonical_runner_set,
    verify_final_runner_set,
)
from utils.csv_metadata import verify_canonical_sidecar_target_metadata  # noqa: E402
from utils.race_lifecycle import (  # noqa: E402
    STALE_FORM_GUIDE,
    UPCOMING_NOT_JUMPED,
    classify_race_file,
    melbourne_now,
)


SAFE_ENV_DEFAULTS = {
    "WATCH_DOWNLOADS": "0",
    "WATCH_UPCOMING": "0",
    "PREDICTION_IMPORT_MODE": "prediction_only",
    "ENABLE_LIVE_SCRAPING": "0",
    "ENABLE_RESULTS_SCRAPERS": "0",
    "ENABLE_AUTO_SCRAPE_ODDS": "0",
    "SPORTSBET_DOM_FALLBACK_ODDS": "0",
    "TGR_ENABLED": "0",
    "DISABLE_SPORTSBET_INTEGRATOR": "1",
    "INGEST_EMBEDDED_HISTORY_ON_PREDICT": "0",
    "MEM_LOGGER_DISABLED": "1",
    "MEM_WATCHDOG_DISABLED": "1",
}


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "approved",
    }


def _live_odds_capture_approval(args: argparse.Namespace) -> dict[str, Any]:
    flag_approved = bool(getattr(args, "approve_live_odds_capture", False))
    env_approved = _env_flag_enabled("APPROVE_LIVE_ODDS_CAPTURE")
    approved = flag_approved or env_approved
    return {
        "approved": approved,
        "status": "approved" if approved else "not_approved",
        "sources": {
            "cli_approve_live_odds_capture": flag_approved,
            "env_APPROVE_LIVE_ODDS_CAPTURE": env_approved,
        },
        "required_for": "--capture-live-odds",
    }


def _live_persist_approval(args: argparse.Namespace) -> dict[str, Any]:
    flag_approved = bool(getattr(args, "approve_live_persist", False))
    env_approved = _env_flag_enabled("APPROVE_LIVE_PERSIST")
    approved = flag_approved or env_approved
    return {
        "approved": approved,
        "status": "approved" if approved else "not_approved",
        "sources": {
            "cli_approve_live_persist": flag_approved,
            "env_APPROVE_LIVE_PERSIST": env_approved,
        },
        "required_for": "--persist",
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def _report_only_calibration_from_design(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError("calibration_design_missing")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("calibration_design_root_not_object")

    failures: list[str] = []
    schema_version = report.get("schema_version")
    status = report.get("status")
    runtime_transform = report.get("runtime_transform_spec")
    runtime_transform = (
        runtime_transform if isinstance(runtime_transform, dict) else {}
    )
    deployment_control = report.get("deployment_control")
    deployment_control = (
        deployment_control if isinstance(deployment_control, dict) else {}
    )
    runtime_scope = report.get("runtime_scope")
    runtime_scope = runtime_scope if isinstance(runtime_scope, dict) else {}
    alpha = _safe_float(runtime_transform.get("alpha"))

    if schema_version == "calibration_layer_design_v1":
        if status != "READY_FOR_OPERATOR_DESIGN_REVIEW":
            failures.append("calibration_design_not_ready")
        if report.get("failures"):
            failures.append("calibration_design_contains_failures")
        if deployment_control.get("promotion_allowed") is not False:
            failures.append("calibration_promotion_not_blocked")
        if deployment_control.get("registry_mutation_allowed") is not False:
            failures.append("calibration_registry_mutation_not_blocked")
        if deployment_control.get("model_artifact_written") is not False:
            failures.append("calibration_model_artifact_written")
        if deployment_control.get("production_config_write_allowed") is not False:
            failures.append("calibration_production_config_write_allowed")
        if deployment_control.get("betting_allowed") is not False:
            failures.append("calibration_betting_allowed")
        if deployment_control.get("required_gate") != "APPROVE_MODEL_PROMOTION":
            failures.append("calibration_required_gate_missing")
    elif schema_version == "runtime_calibration_config_v1":
        if status != "ACTIVE_REPORT_ONLY":
            failures.append("runtime_calibration_config_not_active")
        if runtime_scope.get("report_only") is not True:
            failures.append("runtime_calibration_config_not_report_only")
        if runtime_scope.get("canonical_probability_key_unchanged") != "win_prob_norm":
            failures.append("runtime_calibration_config_changes_canonical_probability")
        if runtime_scope.get("canonical_rank_unchanged") is not True:
            failures.append("runtime_calibration_config_changes_canonical_rank")
        if deployment_control.get("model_artifact_written") is not False:
            failures.append("runtime_calibration_model_artifact_written")
        if deployment_control.get("model_registry_index_mutated") is not False:
            failures.append("runtime_calibration_registry_index_mutated")
        if deployment_control.get("best_model_symlinks_mutated") is not False:
            failures.append("runtime_calibration_best_symlinks_mutated")
        if deployment_control.get("label_write") is not False:
            failures.append("runtime_calibration_label_write")
        if deployment_control.get("betting") is not False:
            failures.append("runtime_calibration_betting")
        if (
            deployment_control.get("required_env_var")
            != "APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR"
        ):
            failures.append("runtime_calibration_required_env_missing")
    else:
        failures.append("calibration_design_schema_mismatch")

    if runtime_transform.get("algorithm") != "power_normalize_per_race":
        failures.append("calibration_algorithm_mismatch")
    if alpha is None or alpha <= 0:
        failures.append("calibration_alpha_invalid")
    if runtime_transform.get("input_probability_key") != "win_prob_norm":
        failures.append("calibration_input_key_mismatch")
    if not runtime_transform.get("output_probability_key"):
        failures.append("calibration_output_key_missing")
    if runtime_transform.get("uses_labels_at_runtime") is not False:
        failures.append("calibration_uses_labels_at_runtime")
    if runtime_transform.get("uses_odds_at_runtime") is not False:
        failures.append("calibration_uses_odds_at_runtime")
    if runtime_transform.get("rank_preserving_when_alpha_positive") is not True:
        failures.append("calibration_not_rank_preserving")
    if runtime_transform.get("requires_runner_complete_race_group") is not True:
        failures.append("calibration_missing_complete_group_requirement")
    if failures:
        raise ValueError(",".join(failures))

    return {
        "algorithm": "power_normalize_per_race",
        "alpha": alpha,
        "input_probability_key": "win_prob_norm",
        "output_probability_key": runtime_transform.get(
            "output_probability_key"
        ),
        "source_design_path": str(path),
        "source_schema_version": schema_version,
        "source_status": status,
    }


def _safe_db_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _configure_safe_runtime(db_path: Path) -> None:
    for key, value in SAFE_ENV_DEFAULTS.items():
        os.environ[key] = value
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["DATABASE_PATH"] = str(db_path)
    os.environ["GREYHOUND_DB_PATH"] = str(db_path)
    os.environ["STAGING_DB_PATH"] = str(db_path)
    os.environ["ANALYTICS_DB_PATH"] = str(db_path)
    os.environ["SINGLE_DB_MODE"] = "1"


def _candidate_files(race_files: list[str], upcoming_dir: str) -> list[Path]:
    if race_files:
        out: list[Path] = []
        for raw in race_files:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            if path.exists():
                out.append(path)
                continue
            fallback = ROOT / upcoming_dir / raw
            if fallback.exists():
                out.append(fallback.resolve())
        return sorted(out)
    directory = Path(upcoming_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    return sorted(path.resolve() for path in directory.glob("*.csv"))


def _probability_sum(snapshot: dict[str, Any]) -> dict[str, Any]:
    probs = []
    for row in snapshot.get("predictions") or []:
        try:
            probs.append(float(row.get("win_prob_norm")))
        except Exception:
            continue
    return {
        "runner_count": len(snapshot.get("predictions") or []),
        "probability_sum": sum(probs) if probs else None,
        "abs_error": abs(sum(probs) - 1.0) if probs else None,
    }


def _preview_market_odds_win(row: dict[str, Any]) -> Any:
    if row.get("market_odds_win") is not None:
        return row.get("market_odds_win")
    if row.get("odds") is not None:
        return row.get("odds")
    odds_snapshot = row.get("odds_snapshot")
    if isinstance(odds_snapshot, dict):
        return odds_snapshot.get("market_odds_win")
    return None


def _prediction_preview(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for row in snapshot.get("predictions") or []:
        if not isinstance(row, dict):
            continue
        preview.append(
            {
                "predicted_rank": row.get("predicted_rank"),
                "box_number": row.get("box_number"),
                "dog_name": row.get("dog_name") or row.get("dog_clean_name"),
                "win_prob_norm": row.get("win_prob_norm"),
                "odds_match_status": row.get("odds_match_status"),
                "market_odds_win": _preview_market_odds_win(row),
                "ev_win": row.get("ev_win"),
                "quality_flags": row.get("quality_flags") or [],
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        rank = row.get("predicted_rank")
        try:
            return (int(rank), 0.0)
        except (TypeError, ValueError):
            pass
        try:
            return (9999, -float(row.get("win_prob_norm") or 0.0))
        except (TypeError, ValueError):
            return (9999, 0.0)

    return sorted(preview, key=sort_key)


def _set_readiness_requirement(
    snapshot: dict[str, Any],
    key: str,
    value: bool,
) -> None:
    readiness = snapshot.get("snapshot_readiness")
    if not isinstance(readiness, dict):
        return
    requirements = readiness.get("requirements")
    if not isinstance(requirements, dict):
        return
    requirements[key] = bool(value)
    readiness["status"] = "READY" if all(requirements.values()) else "NOT_READY"


def _apply_target_metadata_to_snapshot(
    snapshot: dict[str, Any],
    target_metadata: dict[str, Any],
) -> None:
    verified = target_metadata.get("target_metadata_status") == "verified"
    snapshot["target_metadata_status"] = target_metadata.get("target_metadata_status")
    snapshot["target_metadata_failure_reason"] = target_metadata.get(
        "target_metadata_failure_reason"
    )
    snapshot["target_distance"] = target_metadata.get("target_distance") if verified else None
    snapshot["target_grade"] = target_metadata.get("target_grade") if verified else None
    snapshot["target_distance_source"] = (
        target_metadata.get("target_distance_source") if verified else None
    )
    snapshot["target_grade_source"] = (
        target_metadata.get("target_grade_source") if verified else None
    )
    snapshot["metadata_is_leakage_safe"] = bool(verified)
    snapshot["metadata_source_detail"] = (
        target_metadata.get("metadata_source_detail") if verified else None
    )
    snapshot["canonical_race_url"] = target_metadata.get("canonical_race_url")
    snapshot["race_time_mapping_status"] = target_metadata.get("race_time_mapping_status")
    snapshot["race_time_source"] = target_metadata.get("race_time_source")
    snapshot["target_metadata_verification"] = target_metadata
    _set_readiness_requirement(snapshot, "target_metadata_verified", verified)


def _readiness_failure_categories(snapshot: dict[str, Any]) -> list[str]:
    readiness = snapshot.get("snapshot_readiness")
    requirements = (
        readiness.get("requirements")
        if isinstance(readiness, dict) and isinstance(readiness.get("requirements"), dict)
        else {}
    )
    categories: set[str] = set()
    for key, value in requirements.items():
        if value is True:
            continue
        if key in {"pre_jump_lifecycle"}:
            categories.add("lifecycle")
        elif key in {
            "source_runner_set_complete",
            "predictions_match_source_runner_set",
            "final_runner_set_verified",
        }:
            categories.add("runner_set")
        elif key == "target_metadata_verified":
            categories.add("metadata")
        elif "odds" in key:
            categories.add("odds_provenance")
        else:
            categories.add("data_integrity")
    return sorted(categories)


def _persistence_skip_category(
    *,
    live_lifecycle: bool,
    runner_set_complete: bool,
    final_runner_verified: bool,
    target_metadata_verified: bool,
    allow_unverified_runner_set: bool,
    mechanics_only: bool,
) -> str | None:
    if mechanics_only:
        return "lifecycle"
    if not live_lifecycle:
        return "lifecycle"
    if not runner_set_complete:
        return "runner_set"
    if not final_runner_verified:
        return "runner_set"
    if not target_metadata_verified:
        return "metadata"
    return None


def _should_write_snapshot(
    *,
    persist: bool,
    live_lifecycle: bool,
    runner_set_complete: bool,
    final_runner_verified: bool,
    target_metadata_verified: bool,
    allow_unverified_runner_set: bool,
    mechanics_only: bool,
) -> bool:
    return bool(
        persist
        and live_lifecycle
        and runner_set_complete
        and final_runner_verified
        and target_metadata_verified
        and not mechanics_only
    )


def _lifecycle_freshness_record(
    *,
    race_file: Path,
    lifecycle: Any,
    checked_at: datetime,
) -> dict[str, Any]:
    status = getattr(lifecycle, "status", None)
    record = {
        "checked_at": checked_at.isoformat(),
        "race_file": str(race_file),
        "lifecycle_status": status,
        "lifecycle_status_reason": getattr(lifecycle, "status_reason", None),
        "race_date": getattr(lifecycle, "race_date", None),
        "venue": getattr(lifecycle, "venue", None),
        "race_number": getattr(lifecycle, "race_number", None),
        "jump_time": getattr(lifecycle, "jump_time", None),
        "jump_datetime": getattr(lifecycle, "jump_datetime", None),
        "still_pre_jump": status == UPCOMING_NOT_JUMPED,
    }
    jump_datetime = record.get("jump_datetime")
    if jump_datetime:
        try:
            jump_dt = datetime.fromisoformat(str(jump_datetime))
            checked_dt = checked_at
            if checked_dt.tzinfo is None and jump_dt.tzinfo is not None:
                checked_dt = checked_dt.replace(tzinfo=jump_dt.tzinfo)
            elif checked_dt.tzinfo is not None and jump_dt.tzinfo is None:
                jump_dt = jump_dt.replace(tzinfo=checked_dt.tzinfo)
            record["seconds_to_jump"] = (jump_dt - checked_dt).total_seconds()
        except Exception:
            record["seconds_to_jump"] = None
    return record


def _pre_persist_freshness_check(
    *,
    race_file: Path,
    db_path: Path,
) -> dict[str, Any]:
    checked_at = melbourne_now()
    lifecycle = classify_race_file(
        race_file,
        db_path=str(db_path),
        source_context="csv_file",
        now=checked_at,
    )
    return _lifecycle_freshness_record(
        race_file=race_file,
        lifecycle=lifecycle,
        checked_at=checked_at,
    )


def _skipped_lifecycle_candidate(
    *,
    race_file: Path,
    lifecycle: Any,
    checked_at: datetime,
) -> dict[str, Any]:
    record = _lifecycle_freshness_record(
        race_file=race_file,
        lifecycle=lifecycle,
        checked_at=checked_at,
    )
    return {
        "status": "SKIPPED",
        "race_file": str(race_file),
        "lifecycle_status": record.get("lifecycle_status"),
        "lifecycle_status_reason": record.get("lifecycle_status_reason"),
        "race_date": record.get("race_date"),
        "venue": record.get("venue"),
        "race_number": record.get("race_number"),
        "jump_time": record.get("jump_time"),
        "jump_datetime": record.get("jump_datetime"),
        "pre_capture_freshness_check": record,
        "persistence_skip_category": "lifecycle",
        "persistence": {
            "status": "skipped_non_live_lifecycle_pre_capture",
            "reason": "candidate_not_upcoming_not_jumped_at_capture_start",
            "pre_capture_freshness_check": record,
        },
    }


def _apply_pre_persist_freshness_to_snapshot(
    snapshot: dict[str, Any],
    freshness: dict[str, Any],
) -> None:
    snapshot["pre_persist_freshness_check"] = freshness
    snapshot["lifecycle_status"] = freshness.get("lifecycle_status")
    snapshot["lifecycle_status_reason"] = freshness.get("lifecycle_status_reason")
    snapshot["jump_time"] = freshness.get("jump_time")
    snapshot["jump_datetime"] = freshness.get("jump_datetime")
    still_pre_jump = freshness.get("still_pre_jump") is True
    snapshot["is_pre_jump_snapshot"] = still_pre_jump
    snapshot["snapshot_state"] = (
        "pre_jump_feature_freeze" if still_pre_jump else "not_bet_qualified_lifecycle"
    )
    provenance = snapshot.get("snapshot_provenance")
    if isinstance(provenance, dict):
        provenance["pre_persist_freshness_check"] = freshness
    _set_readiness_requirement(snapshot, "pre_jump_lifecycle", still_pre_jump)
    _set_readiness_requirement(
        snapshot,
        "pre_persist_lifecycle_verified",
        still_pre_jump,
    )


def _copy_metadata_sidecar_for_prediction_input(
    *,
    source_csv: Path,
    prediction_csv: Path,
) -> dict[str, Any]:
    """Copy verified CSV metadata sidecar to an aligned temp prediction input.

    Canonical runner alignment can write a temporary CSV before prediction. The
    original CSV sidecar carries safe target distance/grade provenance, so the
    temp prediction CSV must receive the same sidecar or the model-facing mapper
    defaults target metadata back to missing.
    """
    source_sidecar = Path(f"{source_csv}.metadata.json")
    prediction_sidecar = Path(f"{prediction_csv}.metadata.json")
    if source_sidecar == prediction_sidecar:
        return {
            "status": "same_path",
            "source_sidecar": str(source_sidecar),
            "prediction_sidecar": str(prediction_sidecar),
        }
    if not source_sidecar.exists():
        return {
            "status": "source_sidecar_missing",
            "source_sidecar": str(source_sidecar),
            "prediction_sidecar": str(prediction_sidecar),
        }
    prediction_sidecar.write_bytes(source_sidecar.read_bytes())
    return {
        "status": "copied",
        "source_sidecar": str(source_sidecar),
        "prediction_sidecar": str(prediction_sidecar),
    }


def _capture_live_odds_for_lifecycle(
    *,
    db_path: Path,
    lifecycle: Any,
) -> dict[str, Any]:
    from odds_auto_integrator import ensure_odds_for_target_race

    venue = getattr(lifecycle, "venue", None)
    race_number = getattr(lifecycle, "race_number", None)
    race_date = getattr(lifecycle, "race_date", None)
    if not venue or not race_number or not race_date:
        return {
            "status": "DATA_MISSING",
            "success": False,
            "reason": "missing_lifecycle_target_identity",
            "venue": venue,
            "race_number": race_number,
            "race_date": race_date,
            "append_only": True,
        }
    return ensure_odds_for_target_race(
        str(db_path),
        venue,
        race_number,
        race_date,
        allow_auto_scrape_odds=True,
        append_only=True,
    )


def _capture_one(
    *,
    race_file: Path,
    lifecycle: Any,
    db_path: Path,
    snapshot_dir: Path,
    persist: bool,
    mechanics_only: bool,
    capture_live_odds_requested: bool,
    capture_live_odds_approved: bool,
    allow_unverified_runner_set: bool,
    report_only_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from app import enhance_prediction_with_csv_meta, run_prediction_for_race_file

    original_source_runner_completeness = analyze_csv_runner_completeness(race_file).as_dict()
    source_runner_completeness = dict(original_source_runner_completeness)
    canonical_url = canonical_race_url_from_sidecar(race_file)
    target_metadata = verify_canonical_sidecar_target_metadata(
        race_file,
        race_number=getattr(lifecycle, "race_number", None),
        canonical_url=canonical_url,
    )
    target_metadata_verified = (
        target_metadata.get("target_metadata_status") == "verified"
    )
    final_runner_set_verification: dict[str, Any] | None = None
    final_runner_verified = False
    runner_set_alignment: dict[str, Any] | None = None
    prediction_race_file = race_file
    prediction_input_sidecar: dict[str, Any] | None = None
    temp_prediction_dir: tempfile.TemporaryDirectory[str] | None = None
    if not mechanics_only and getattr(lifecycle, "status", None) == UPCOMING_NOT_JUMPED:
        canonical_runner_set = fetch_canonical_runner_set(canonical_url)
        try:
            original_csv_text = race_file.read_text(encoding="utf-8-sig", errors="replace")
            aligned_csv_text, runner_set_alignment = (
                align_csv_text_to_canonical_final_runner_set(
                    original_csv_text,
                    canonical_runner_set,
                    source=str(race_file),
                )
            )
        except Exception as exc:
            runner_set_alignment = {
                "schema_version": "canonical_runner_alignment_v1",
                "status": "not_aligned",
                "source": str(race_file),
                "reason": f"alignment_error:{type(exc).__name__}",
            }
            aligned_csv_text = None
        if runner_set_alignment.get("status") == "aligned" and aligned_csv_text is not None:
            temp_prediction_dir = tempfile.TemporaryDirectory(
                prefix="prejump_aligned_"
            )
            prediction_race_file = Path(temp_prediction_dir.name) / race_file.name
            prediction_race_file.write_text(aligned_csv_text, encoding="utf-8")
            prediction_input_sidecar = _copy_metadata_sidecar_for_prediction_input(
                source_csv=race_file,
                prediction_csv=prediction_race_file,
            )
            if isinstance(runner_set_alignment, dict):
                runner_set_alignment["prediction_input_sidecar"] = prediction_input_sidecar
            source_runner_completeness = analyze_csv_text_runner_completeness(
                aligned_csv_text,
                source=str(race_file),
            ).as_dict()
            source_runner_completeness["alignment_status"] = "canonical_aligned"
        final_runner_set_verification = verify_final_runner_set(
            source_runner_completeness,
            canonical_runner_set,
        )
        final_runner_verified = (
            final_runner_set_verification.get("final_runner_set_status") == "verified"
        )

    odds_capture: dict[str, Any] | None = None
    if capture_live_odds_requested and not capture_live_odds_approved:
        odds_capture = {
            "status": "APPROVAL_REQUIRED",
            "success": False,
            "reason": "live_odds_capture_not_approved",
            "required_approval": "APPROVE_LIVE_ODDS_CAPTURE or --approve-live-odds-capture",
            "append_only": True,
        }
    if (
        capture_live_odds_requested
        and capture_live_odds_approved
        and not mechanics_only
        and getattr(lifecycle, "status", None) == UPCOMING_NOT_JUMPED
        and final_runner_verified
        and target_metadata_verified
    ):
        odds_capture = _capture_live_odds_for_lifecycle(
            db_path=db_path,
            lifecycle=lifecycle,
        )

    prediction_timestamp = datetime.now().isoformat(timespec="seconds")
    try:
        result = run_prediction_for_race_file(str(prediction_race_file))
    finally:
        if temp_prediction_dir is not None:
            temp_prediction_dir.cleanup()
            temp_prediction_dir = None
    if not isinstance(result, dict) or not result.get("success"):
        return {
            "status": "FAILED",
            "race_file": str(race_file),
            "prediction_input_file": str(prediction_race_file),
            "lifecycle_status": getattr(lifecycle, "status", None),
            "odds_capture_requested": bool(capture_live_odds_requested),
            "odds_capture_approved": bool(capture_live_odds_approved),
            "odds_capture": odds_capture,
            "runner_set_alignment": runner_set_alignment,
            "original_source_runner_completeness": original_source_runner_completeness,
            "final_runner_set_verification": final_runner_set_verification,
            "target_metadata_status": target_metadata.get("target_metadata_status"),
            "target_metadata_failure_reason": target_metadata.get(
                "target_metadata_failure_reason"
            ),
            "target_metadata": target_metadata,
            "error": (
                (result or {}).get("error") if isinstance(result, dict) else "prediction_failed"
            ),
        }
    try:
        result = enhance_prediction_with_csv_meta(result, str(race_file))
    except Exception:
        pass
    snapshot = build_prediction_snapshot(
        result,
        source_file_path=str(race_file),
        lifecycle=lifecycle,
        prediction_timestamp=prediction_timestamp,
        feature_freeze_timestamp=prediction_timestamp,
        source_runner_completeness=source_runner_completeness,
        final_runner_set_verification=final_runner_set_verification,
        report_only_calibration=report_only_calibration,
    )
    if runner_set_alignment:
        snapshot["runner_set_alignment"] = runner_set_alignment
        snapshot["original_source_runner_completeness"] = (
            original_source_runner_completeness
        )
        snapshot["prediction_input_mode"] = (
            "canonical_aligned_temp_csv"
            if runner_set_alignment.get("status") == "aligned"
            else "original_csv"
        )
    _apply_target_metadata_to_snapshot(snapshot, target_metadata)
    assert_no_result_fields(snapshot)
    live_lifecycle = snapshot.get("lifecycle_status") == UPCOMING_NOT_JUMPED
    runner_set_complete = snapshot.get("runner_set_complete") is True
    final_runner_verified = snapshot.get("final_runner_set_status") == "verified"
    target_metadata_verified = snapshot.get("target_metadata_status") == "verified"
    write_snapshot = _should_write_snapshot(
        persist=persist,
        live_lifecycle=live_lifecycle,
        runner_set_complete=runner_set_complete,
        final_runner_verified=final_runner_verified,
        target_metadata_verified=target_metadata_verified,
        allow_unverified_runner_set=allow_unverified_runner_set,
        mechanics_only=mechanics_only,
    )
    pre_persist_freshness: dict[str, Any] | None = None
    if write_snapshot:
        pre_persist_freshness = _pre_persist_freshness_check(
            race_file=race_file,
            db_path=db_path,
        )
        _apply_pre_persist_freshness_to_snapshot(snapshot, pre_persist_freshness)
        live_lifecycle = pre_persist_freshness.get("still_pre_jump") is True
        write_snapshot = write_snapshot and live_lifecycle
    persistence = persist_prediction_snapshot(
        snapshot,
        snapshot_dir,
        dry_run=not write_snapshot,
        require_final_runner_verification=True,
    )
    if persist and not write_snapshot:
        if (
            pre_persist_freshness is not None
            and pre_persist_freshness.get("still_pre_jump") is not True
        ):
            persistence["status"] = "skipped_pre_persist_lifecycle_not_live"
            persistence["reason"] = "pre_persist_lifecycle_not_live"
            persistence["pre_persist_freshness_check"] = pre_persist_freshness
        elif not live_lifecycle:
            persistence["status"] = "skipped_non_live_lifecycle"
        elif not runner_set_complete:
            persistence["status"] = "skipped_incomplete_runner_set"
        elif not final_runner_verified:
            persistence["status"] = "skipped_pre_jump_runner_set_unverified"
            persistence["reason"] = "pre_jump_runner_set_unverified"
            persistence["final_runner_set_status"] = snapshot.get("final_runner_set_status")
            persistence["final_runner_set_mismatch_reason"] = snapshot.get(
                "final_runner_set_mismatch_reason"
            )
        elif not target_metadata_verified:
            persistence["status"] = "skipped_target_metadata_not_verified"
            persistence["reason"] = "target_metadata_not_verified"
            persistence["target_metadata_status"] = snapshot.get("target_metadata_status")
            persistence["target_metadata_failure_reason"] = snapshot.get(
                "target_metadata_failure_reason"
            )
        else:
            persistence["status"] = "skipped_not_persistable"
        persistence["skip_category"] = _persistence_skip_category(
            live_lifecycle=live_lifecycle,
            runner_set_complete=runner_set_complete,
            final_runner_verified=final_runner_verified,
            target_metadata_verified=target_metadata_verified,
            allow_unverified_runner_set=allow_unverified_runner_set,
            mechanics_only=mechanics_only,
        )
    if pre_persist_freshness is not None and persistence.get("status") == "persisted":
        persistence["pre_persist_freshness_check"] = pre_persist_freshness

    snapshot_readiness = snapshot.get("snapshot_readiness")
    ev_readiness = (
        snapshot_readiness.get("ev_readiness")
        if isinstance(snapshot_readiness, dict)
        and isinstance(snapshot_readiness.get("ev_readiness"), dict)
        else {}
    )
    readiness_failure_categories = _readiness_failure_categories(snapshot)
    return {
        "status": "SUCCESS",
        "race_file": str(race_file),
        "prediction_input_mode": snapshot.get("prediction_input_mode") or "original_csv",
        "mechanics_only_not_live": mechanics_only,
        "race_id": snapshot.get("race_id"),
        "stable_race_key": snapshot.get("stable_race_key"),
        "lifecycle_status": snapshot.get("lifecycle_status"),
        "snapshot_state": snapshot.get("snapshot_state"),
        "prediction_timestamp": snapshot.get("prediction_timestamp"),
        "feature_freeze_timestamp": snapshot.get("feature_freeze_timestamp"),
        "model_version": snapshot.get("model_version"),
        "runner_count": len(snapshot.get("predictions") or []),
        "runner_set_complete": runner_set_complete,
        "source_runner_completeness": source_runner_completeness,
        "original_source_runner_completeness": original_source_runner_completeness,
        "runner_set_alignment": runner_set_alignment,
        "final_runner_set_verified": final_runner_verified,
        "final_runner_set_status": snapshot.get("final_runner_set_status"),
        "final_runner_set_mismatch_reason": snapshot.get(
            "final_runner_set_mismatch_reason"
        ),
        "final_runner_set_verification": final_runner_set_verification,
        "pre_persist_freshness_check": pre_persist_freshness,
        "target_metadata_status": snapshot.get("target_metadata_status"),
        "target_distance": snapshot.get("target_distance"),
        "target_grade": snapshot.get("target_grade"),
        "target_distance_source": snapshot.get("target_distance_source"),
        "target_grade_source": snapshot.get("target_grade_source"),
        "target_metadata_failure_reason": snapshot.get(
            "target_metadata_failure_reason"
        ),
        "target_metadata": target_metadata,
        "odds_capture_requested": bool(capture_live_odds_requested),
        "odds_capture_approved": bool(capture_live_odds_approved),
        "odds_capture": odds_capture,
        "priced_ev_runner_count": int(ev_readiness.get("ev_present_runner_count") or 0),
        "priced_runner_count": int(ev_readiness.get("priced_runner_count") or 0),
        "ev_eligible_runner_count": int(
            ev_readiness.get("ev_eligible_runner_count") or 0
        ),
        "ev_readiness_status": ev_readiness.get("status"),
        "ev_readiness": ev_readiness,
        "report_only_calibration": snapshot.get("report_only_calibration"),
        "odds_exclusion_counts": ev_readiness.get("odds_exclusion_counts") or {},
        "snapshot_readiness": snapshot_readiness,
        "snapshot_readiness_failure_categories": readiness_failure_categories,
        "prediction_preview": _prediction_preview(snapshot),
        "persistence_skip_category": persistence.get("skip_category")
        or ("dry_run" if not persist else None),
        "probability_sum_check": _probability_sum(snapshot),
        "persistence": persistence,
        "leakage_check": "passed_result_free_snapshot",
    }


def _endpoint_health_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, url in {
        "api_health": "http://127.0.0.1:5002/api/health",
        "model_health": "http://127.0.0.1:5002/api/model_health",
    }.items():
        try:
            with urlopen(url, timeout=0.75) as response:
                checks[name] = {
                    "status": "reachable" if response.status == 200 else "degraded",
                    "http_status": response.status,
                }
        except URLError as exc:
            checks[name] = {"status": "not_running_or_unreachable", "error": str(exc)}
        except Exception as exc:
            checks[name] = {"status": "error", "error": f"{type(exc).__name__}:{exc}"}
    return checks


def _sqlite_quick_check(db_path: Path) -> dict[str, Any]:
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            row = conn.execute("PRAGMA quick_check").fetchone()
        value = row[0] if row else None
        return {"status": "ok" if value == "ok" else "degraded", "quick_check": value}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}:{exc}"}


def _regular_checks(db_path: Path, captures: list[dict[str, Any]]) -> dict[str, Any]:
    model_versions = sorted(
        {
            str(capture.get("model_version"))
            for capture in captures
            if capture.get("model_version") not in (None, "", "unknown")
        }
    )
    unknown_model_version_count = sum(
        1 for capture in captures if capture.get("model_version") in (None, "", "unknown")
    )
    leakage_passed = all(
        capture.get("leakage_check") == "passed_result_free_snapshot"
        for capture in captures
        if capture.get("status") == "SUCCESS"
    )
    return {
        "endpoint_health": _endpoint_health_checks(),
        "model_version": {
            "status": "ok" if model_versions and unknown_model_version_count == 0 else "warning",
            "versions": model_versions,
            "unknown_count": unknown_model_version_count,
        },
        "calibration_drift": {
            "status": "not_evaluated_no_result_ingestion",
            "reason": "capture fix is label-free and does not ingest results",
        },
        "data_integrity": _sqlite_quick_check(db_path),
        "temporal_leakage": {
            "status": "passed" if leakage_passed else "not_run_or_failed",
            "guard": "assert_no_result_fields",
        },
    }


def capture_snapshots(args: argparse.Namespace) -> dict[str, Any]:
    db_path = _safe_db_path(args.db)
    odds_approval = _live_odds_capture_approval(args)
    persist_approval = _live_persist_approval(args)
    capture_live_odds_requested = bool(getattr(args, "capture_live_odds", False))
    persist_requested = bool(getattr(args, "persist", False))
    persist_effective = persist_requested and bool(persist_approval["approved"])
    report_only_calibration: dict[str, Any] | None = None
    calibration_design_path = getattr(args, "report_only_calibration_design", None)
    if calibration_design_path:
        try:
            report_only_calibration = _report_only_calibration_from_design(
                Path(str(calibration_design_path))
            )
        except Exception as exc:
            return {
                "status": "DATA_MISSING",
                "reason": "report_only_calibration_design_not_ready",
                "error": str(exc),
                "report_only_calibration_design": str(calibration_design_path),
                "capture_count": 0,
                "captures": [],
                "data_missing": ["report_only_calibration_design_not_ready"],
                "persist_requested": persist_requested,
                "persist_approved": bool(persist_approval["approved"]),
                "persist_approval": persist_approval,
                "odds_capture_requested": capture_live_odds_requested,
                "odds_capture_approval": odds_approval,
            }
    if not db_path.exists():
        return {
            "status": "DATA_MISSING",
            "reason": "db_path_not_found",
            "db_path": str(db_path),
            "capture_count": 0,
            "captures": [],
            "data_missing": ["db_path_not_found"],
            "persist_requested": persist_requested,
            "persist_approved": bool(persist_approval["approved"]),
            "persist_approval": persist_approval,
            "odds_capture_requested": capture_live_odds_requested,
            "odds_capture_approval": odds_approval,
        }
    _configure_safe_runtime(db_path)
    files = _candidate_files(args.race_file or [], args.upcoming_dir)
    classification_checked_at = melbourne_now()
    lifecycles = [
        (
            path,
            classify_race_file(
                path,
                db_path=str(db_path),
                source_context="csv_file",
                now=classification_checked_at,
            ),
        )
        for path in files
    ]
    counts = Counter(lifecycle.status for _, lifecycle in lifecycles)
    live_targets = [
        (path, lifecycle)
        for path, lifecycle in lifecycles
        if lifecycle.status == UPCOMING_NOT_JUMPED
    ]
    skipped_lifecycle_candidates = [
        _skipped_lifecycle_candidate(
            race_file=path,
            lifecycle=lifecycle,
            checked_at=classification_checked_at,
        )
        for path, lifecycle in lifecycles
        if lifecycle.status != UPCOMING_NOT_JUMPED
    ]

    mechanics_only = False
    targets = live_targets
    data_missing: list[str] = []
    if not targets:
        data_missing.append("no_genuinely_upcoming_not_jumped_local_races")
        if args.mechanics_on_stale:
            mechanics_only = True
            targets = [
                (path, lifecycle)
                for path, lifecycle in lifecycles
                if lifecycle.status == STALE_FORM_GUIDE
            ][:1]

    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    captures = [
        _capture_one(
            race_file=path,
            lifecycle=lifecycle,
            db_path=db_path,
            snapshot_dir=Path(args.snapshot_dir),
            persist=persist_effective,
            mechanics_only=mechanics_only,
            capture_live_odds_requested=capture_live_odds_requested,
            capture_live_odds_approved=bool(odds_approval["approved"]),
            allow_unverified_runner_set=bool(args.allow_unverified_runner_set),
            report_only_calibration=report_only_calibration,
        )
        for path, lifecycle in targets
    ]
    final_runner_counts = Counter(
        str(capture.get("final_runner_set_status") or "not_checked")
        for capture in captures
    )
    target_metadata_counts = Counter(
        str(capture.get("target_metadata_status") or "not_checked")
        for capture in captures
    )
    ev_readiness_counts = Counter(
        str(capture.get("ev_readiness_status") or "not_checked")
        for capture in captures
        if capture.get("status") == "SUCCESS"
    )
    odds_exclusion_counts: Counter[str] = Counter()
    for capture in captures:
        for reason, count in (capture.get("odds_exclusion_counts") or {}).items():
            odds_exclusion_counts[str(reason)] += int(count or 0)
    persisted_with_top_level_metadata_count = sum(
        1
        for capture in captures
        if (capture.get("persistence") or {}).get("status") == "persisted"
        and capture.get("target_distance") not in (None, "")
        and capture.get("target_grade") not in (None, "")
    )

    if captures and mechanics_only:
        status = "MECHANICS_ONLY_NOT_LIVE"
    elif captures:
        status = "SUCCESS"
    else:
        status = "DATA_MISSING"

    return {
        "status": status,
        "dry_run": not persist_effective,
        "persist_requested": persist_requested,
        "persist_approved": bool(persist_approval["approved"]),
        "persist_approval": persist_approval,
        "odds_capture_requested": capture_live_odds_requested,
        "odds_capture_approved": bool(odds_approval["approved"]),
        "odds_capture_approval": odds_approval,
        "allow_unverified_runner_set": bool(args.allow_unverified_runner_set),
        "report_only_calibration": report_only_calibration,
        "report_only_calibration_design": str(calibration_design_path)
        if calibration_design_path
        else None,
        "db_path": str(db_path),
        "snapshot_dir": str(Path(args.snapshot_dir)),
        "candidate_files": len(files),
        "lifecycle_counts": dict(counts),
        "final_runner_set_counts": dict(final_runner_counts),
        "target_metadata_counts": dict(target_metadata_counts),
        "ev_readiness_counts": dict(ev_readiness_counts),
        "priced_runner_count": sum(
            int(capture.get("priced_runner_count") or 0) for capture in captures
        ),
        "priced_ev_runner_count": sum(
            int(capture.get("priced_ev_runner_count") or 0) for capture in captures
        ),
        "ev_eligible_runner_count": sum(
            int(capture.get("ev_eligible_runner_count") or 0) for capture in captures
        ),
        "odds_exclusion_counts": dict(sorted(odds_exclusion_counts.items())),
        "metadata_verified_count": target_metadata_counts.get("verified", 0),
        "metadata_missing_count": target_metadata_counts.get("missing", 0),
        "metadata_unsafe_count": target_metadata_counts.get("unsafe", 0),
        "metadata_mismatch_count": target_metadata_counts.get("mismatch", 0),
        "persisted_with_top_level_metadata_count": persisted_with_top_level_metadata_count,
        "capture_count": len(captures),
        "captures": captures,
        "skipped_lifecycle_candidate_count": len(skipped_lifecycle_candidates),
        "skipped_lifecycle_candidates": skipped_lifecycle_candidates,
        "data_missing": data_missing,
        "regular_checks": _regular_checks(db_path, captures),
        "safe_runtime_env": {key: os.environ.get(key) for key in sorted(SAFE_ENV_DEFAULTS)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="greyhound_racing_data_writable.db")
    parser.add_argument("--upcoming-dir", default="upcoming_races")
    parser.add_argument("--snapshot-dir", default="artifacts/prediction_snapshots")
    parser.add_argument("--race-file", action="append", help="Specific local race CSV")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--persist", action="store_true", help="Write result-free JSON snapshots")
    parser.add_argument(
        "--approve-live-persist",
        action="store_true",
        help=(
            "Approval gate for --persist. Without this flag or "
            "APPROVE_LIVE_PERSIST=true, snapshot persistence is converted to "
            "a dry-run report."
        ),
    )
    parser.add_argument(
        "--capture-live-odds",
        action="store_true",
        help="Explicitly capture append-only Sportsbet dog-level win odds before prediction snapshots",
    )
    parser.add_argument(
        "--approve-live-odds-capture",
        action="store_true",
        help=(
            "Approval gate for --capture-live-odds. Without this flag or "
            "APPROVE_LIVE_ODDS_CAPTURE=true, odds capture is reported as "
            "APPROVAL_REQUIRED and is not attempted."
        ),
    )
    parser.add_argument(
        "--mechanics-on-stale",
        action="store_true",
        help="If no live races exist, run one stale-form-guide mechanics test without persisting",
    )
    parser.add_argument(
        "--allow-unverified-runner-set",
        action="store_true",
        help=(
            "Deprecated diagnostic flag; persistence still requires verified "
            "canonical pre-race runner-set verification"
        ),
    )
    parser.add_argument(
        "--report-only-calibration-design",
        help=(
            "Optional calibration_layer_design_v1 JSON. When supplied, the "
            "validated report-only power calibration is attached as additive "
            "snapshot fields only; it never promotes a model or changes "
            "canonical ranks/probabilities."
        ),
    )
    parser.add_argument("--output", help="Optional report JSON path")
    args = parser.parse_args()

    report = capture_snapshots(args)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("status") in {"SUCCESS", "MECHANICS_ONLY_NOT_LIVE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
