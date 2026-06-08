#!/usr/bin/env python3
"""Shadow evidence accumulation autopilot V1.

This is a report-only orchestration wrapper. It refreshes pre-jump inputs into
an isolated artifact directory, runs shadow-only scoring with an existing
shadow model, attempts exact official result joins, aggregates cumulative
forward-shadow evidence, and writes the required status packet.

It must not train, promote, mutate registries, update production pointers,
write DB rows, write labels, enable TGR, or emit betting/EV actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.shadow_feature_audit_packet import feature_activation_gate_input_paths  # noqa: E402


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_v1_"
DEFAULT_TARGET_JOINED_RACES = 100
DEFAULT_MIN_JOINED_RACES_FOR_STATUS = 100
UV_REFRESH_PACKAGES = ("requests", "beautifulsoup4", "pandas")
WATCHED_QUARANTINED_FEATURES = (
    "same_distance_same_grade_best_time",
    "same_distance_same_grade_avg_time",
)
PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "model_registry/current_production.json",
    ROOT / "processed_manifest.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
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
    "calibration_method_change": False,
    "feature_engineering": False,
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


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


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
        raise ValueError(f"output_dir_must_be_shadow_autopilot_artifact:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def latest_artifact(root: Path, prefix: str, required_file: str) -> Path | None:
    candidates = [
        item
        for item in root.glob(f"{prefix}*")
        if item.is_dir() and (item / required_file).exists()
    ]
    return sorted(candidates)[-1] if candidates else None


def current_aggregate_safe_joined_race_count(aggregate_metrics_path: Path | None) -> int | None:
    aggregate = load_json(aggregate_metrics_path)
    if not aggregate:
        return None
    value = safe_number(aggregate.get("safe_joined_race_count"))
    return int(value) if value is not None else None


def challenger_metric_source_count(metric_path: Path) -> int | None:
    metric = load_json(metric_path)
    if not metric:
        return None
    for key in ("source_safe_exact_joined_race_count", "safe_exact_joined_race_count"):
        value = safe_number(metric.get(key))
        if value is not None:
            return int(value)
    return None


def latest_challenger_activation_metric_paths(
    root: Path,
    *,
    aggregate_metrics_path: Path | None = None,
) -> dict[str, Path | None]:
    challenger_dir = latest_artifact(
        root,
        "forward_shadow_challenger_calibration_",
        "candidate_eval_metrics_for_activation.json",
    )
    if challenger_dir is None:
        return {"baseline_metrics": None, "candidate_metrics": None}
    baseline = challenger_dir / "baseline_eval_metrics_for_activation.json"
    candidate = challenger_dir / "candidate_eval_metrics_for_activation.json"
    if not baseline.exists() or not candidate.exists():
        return {"baseline_metrics": None, "candidate_metrics": None}
    aggregate_count = current_aggregate_safe_joined_race_count(aggregate_metrics_path)
    source_count = challenger_metric_source_count(candidate)
    if aggregate_count is not None and (source_count is None or source_count < aggregate_count):
        return {"baseline_metrics": None, "candidate_metrics": None}
    return {"baseline_metrics": baseline, "candidate_metrics": candidate}


def shadow_prediction_count(daily_dir: Path | None) -> int:
    if daily_dir is None:
        return 0
    manifest = load_json(daily_dir / "shadow_manifest.json") or {}
    manifest_count = safe_number(manifest.get("prediction_rows"))
    jsonl_count = len(read_jsonl(daily_dir / "shadow_predictions.jsonl"))
    return max(int(manifest_count or 0), jsonl_count)


def should_collect_shadow_odds_snapshot(daily_dir: Path | None) -> tuple[bool, str, int]:
    if daily_dir is None:
        return False, "daily_shadow_run_missing", 0
    prediction_path = daily_dir / "shadow_predictions.jsonl"
    if not prediction_path.exists():
        return False, "shadow_predictions_missing", 0
    prediction_count = shadow_prediction_count(daily_dir)
    if prediction_count <= 0:
        return False, "no_shadow_predictions", prediction_count
    return True, "shadow_predictions_present", prediction_count


def shadow_odds_snapshot_command(
    *,
    daily_dir: Path,
    odds_dir: Path,
    db_path: Path,
    current_time: str,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/collect_shadow_odds_snapshots.py"),
        "--shadow-run-dir",
        str(daily_dir),
        "--output-dir",
        str(odds_dir),
        "--db",
        str(db_path),
        "--current-time",
        current_time,
    ]


def build_shadow_odds_snapshot_status(
    *,
    generated_at: datetime,
    odds_dir: Path | None,
    odds_report: Mapping[str, Any] | None,
    skipped_reason: str | None = None,
    prediction_rows: int = 0,
    attempted: bool | None = None,
    status_override: str | None = None,
) -> dict[str, Any]:
    report = odds_report or {}
    final_status = status_override or report.get("final_status") or "SKIPPED"
    return {
        "schema_version": "shadow_autopilot_odds_snapshot_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": final_status,
        "final_status": final_status,
        "collection_attempted": bool(odds_report) if attempted is None else attempted,
        "skipped_reason": skipped_reason,
        "output_dir": relpath(odds_dir),
        "prediction_rows": report.get("prediction_rows", prediction_rows),
        "race_count": report.get("race_count"),
        "runner_rows": report.get("runner_rows"),
        "odds_candidate_rows": report.get("odds_candidate_rows", 0),
        "valid_pre_jump_dog_odds_rows": report.get("valid_pre_jump_dog_odds_rows", 0),
        "race_coverage_path": report.get("race_coverage_path"),
        "races_with_any_odds_candidates": report.get("races_with_any_odds_candidates", 0),
        "races_with_complete_odds_candidate_coverage": report.get(
            "races_with_complete_odds_candidate_coverage",
            0,
        ),
        "races_with_complete_valid_prejump_odds": report.get(
            "races_with_complete_valid_prejump_odds",
            0,
        ),
        "races_with_missing_odds_rows": report.get("races_with_missing_odds_rows", 0),
        "races_with_duplicate_odds_rows": report.get("races_with_duplicate_odds_rows", 0),
        "races_with_post_prediction_odds_rows": report.get(
            "races_with_post_prediction_odds_rows",
            0,
        ),
        "races_with_post_feature_freeze_odds_rows": report.get(
            "races_with_post_feature_freeze_odds_rows",
            0,
        ),
        "odds_research_readiness": report.get("odds_research_readiness"),
        "odds_analysis_status": (report.get("odds_research_readiness") or {}).get(
            "status"
        )
        if isinstance(report.get("odds_research_readiness"), Mapping)
        else None,
        "odds_analysis_blocker_counts": (report.get("odds_research_readiness") or {}).get(
            "blocker_counts"
        )
        if isinstance(report.get("odds_research_readiness"), Mapping)
        else {},
        "ev_eligible_rows": report.get("ev_eligible_rows", 0),
        "ev_output_rows": report.get("ev_output_rows", 0),
        "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
        "protected_paths_unchanged": report.get("protected_paths_unchanged"),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def latest_shadow_model(evidence_root: Path = DEFAULT_EVIDENCE_ROOT) -> Path | None:
    candidates: list[Path] = []
    preferred_patterns = (
        "shadow_evaluation_implementation_*/shadow_randomforest_model.joblib",
        "shadow_reliability_resume_after_db_recovery_*/phase_5_shadow_rerun/quarantine_feature_shadow_run/shadow_randomforest_model.joblib",
        "shadow_reliability_population_hardening_v1_*/phase_5_shadow_rerun/quarantine_feature_shadow_run/shadow_randomforest_model.joblib",
    )
    for pattern in preferred_patterns:
        candidates.extend(path for path in evidence_root.glob(pattern) if path.is_file())
    return sorted(candidates)[-1] if candidates else None


def safe_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def step_command(
    *,
    name: str,
    command: Sequence[str],
    output_dir: Path,
    cwd: Path = ROOT,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    log_dir = output_dir / "logs"
    stdout_path = log_dir / f"{name}.stdout.txt"
    stderr_path = log_dir / f"{name}.stderr.txt"
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.monotonic() - started_monotonic
    write_text(stdout_path, completed.stdout)
    write_text(stderr_path, completed.stderr)
    return {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "duration_seconds": duration,
        "returncode": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "stdout_path": relpath(stdout_path),
        "stderr_path": relpath(stderr_path),
    }


def refresh_dependencies_available() -> bool:
    try:
        import bs4  # noqa: F401
        import pandas  # noqa: F401
        import requests  # noqa: F401
    except Exception:
        return False
    return True


def refresh_command_prefix(mode: str = "auto") -> list[str]:
    if mode not in {"auto", "python", "uv"}:
        raise ValueError(f"unknown_refresh_command_mode:{mode}")
    if mode == "python" or (mode == "auto" and refresh_dependencies_available()):
        return [sys.executable]
    uv_path = shutil.which("uv")
    if uv_path:
        command = [uv_path, "run"]
        for package in UV_REFRESH_PACKAGES:
            command.extend(["--with", package])
        command.append("python")
        return command
    raise RuntimeError("refresh_dependencies_missing_and_uv_unavailable")


def metric_from_source(
    *,
    aggregate_metrics: Mapping[str, Any] | None,
    join_metrics: Mapping[str, Any] | None,
    key: str,
) -> Any:
    if aggregate_metrics and aggregate_metrics.get(key) is not None:
        return aggregate_metrics.get(key)
    if join_metrics and join_metrics.get(key) is not None:
        return join_metrics.get(key)
    return None


def probability_sum_status(max_error: Any) -> dict[str, Any]:
    value = safe_number(max_error)
    return {
        "status": "PASS" if value is not None and value <= 1e-6 else "UNKNOWN_OR_FAIL",
        "max_abs_error": value,
        "threshold": 1e-6,
    }


def build_join_history(evidence_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for join_dir in sorted(evidence_root.glob("forward_shadow_result_join_*")):
        metrics = load_json(join_dir / "shadow_forward_metrics.json")
        identity = load_json(join_dir / "identity_match_report.json")
        if not metrics:
            continue
        rows.append(
            {
                "artifact": relpath(join_dir),
                "source_shadow_run": metrics.get("source_shadow_run"),
                "generated_at": (identity or {}).get("generated_at"),
                "final_status": (identity or {}).get("summary", {}).get("verdict")
                if isinstance((identity or {}).get("summary"), Mapping)
                else None,
                "safe_joined_race_count": metrics.get("safe_joined_race_count"),
                "pending_race_count": metrics.get("pending_race_count"),
                "unsafe_match_count": metrics.get("unsafe_match_count"),
                "top1": metrics.get("top1"),
                "top3": metrics.get("top3"),
                "mean_winner_rank": metrics.get("mean_winner_rank"),
                "brier": metrics.get("brier"),
                "logloss": metrics.get("logloss"),
                "probability_sum_max_error_joined_races": metrics.get(
                    "probability_sum_max_error_joined_races"
                ),
            }
        )
    return rows


def build_aggregate_timeseries(evidence_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aggregate_dir in sorted(evidence_root.glob("forward_shadow_result_aggregate_*")):
        metrics = load_json(aggregate_dir / "aggregate_forward_metrics.json")
        report = load_json(aggregate_dir / "forward_shadow_result_aggregate_report.json")
        calibration = load_json(aggregate_dir / "aggregate_calibration_review.json")
        box_bias = load_json(aggregate_dir / "aggregate_box_bias_review.json")
        if not metrics:
            continue
        slope_intercept = (calibration or {}).get("slope_intercept") or {}
        rows.append(
            {
                "artifact": relpath(aggregate_dir),
                "generated_at": (report or {}).get("generated_at"),
                "safe_joined_race_count": metrics.get("safe_joined_race_count"),
                "pending_race_count": metrics.get("pending_race_count"),
                "unsafe_match_count": metrics.get("unsafe_match_count"),
                "top1": metrics.get("top1"),
                "top3": metrics.get("top3"),
                "mean_winner_rank": metrics.get("mean_winner_rank"),
                "brier": metrics.get("brier"),
                "logloss": metrics.get("logloss"),
                "calibration_slope": slope_intercept.get("slope"),
                "calibration_intercept": slope_intercept.get("intercept"),
                "calibration_status": slope_intercept.get("status"),
                "box1_share": (box_bias or {}).get("safe_joined_box_1_top_pick_share"),
                "probability_sum_max_error_joined_races": metrics.get(
                    "probability_sum_max_error_joined_races"
                ),
            }
        )
    def sort_key(row: Mapping[str, Any]) -> tuple[str, float, int, str]:
        artifact = str(row.get("artifact") or "")
        phase_rank = 2 if artifact.endswith("_daemon") else 1 if artifact.endswith("_daemon_autopilot") else 0
        safe_joined = safe_number(row.get("safe_joined_race_count"))
        return (
            str(row.get("generated_at") or ""),
            safe_joined if safe_joined is not None else -1.0,
            phase_rank,
            artifact,
        )

    return sorted(rows, key=sort_key)


def build_daily_manifest_history(evidence_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(evidence_root.glob("daily_race_ingest_shadow_*/shadow_manifest.json")):
        manifest = load_json(manifest_path)
        if not manifest:
            continue
        rows.append(
            {
                "artifact": relpath(manifest_path.parent),
                "final_status": manifest.get("final_status"),
                "prediction_rows": manifest.get("prediction_rows"),
                "race_count": manifest.get("race_count"),
                "eligible_count": (manifest.get("input_summary") or {}).get("eligible_count"),
                "pending_result_count": manifest.get("race_count"),
                "calibration_method": manifest.get("calibration_method"),
                "all_missing_train_policy": manifest.get("all_missing_train_policy"),
                "tgr_enabled": manifest.get("tgr_enabled"),
                "shadow_training_allowed": manifest.get("shadow_training_allowed"),
                "protected_paths_unchanged": manifest.get("protected_paths_unchanged"),
            }
        )
    return rows


def latest_shadow_drift(evidence_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        path
        for path in evidence_root.glob("*/shadow_drift_report.json")
        if path.is_file()
    )
    if not candidates:
        return None, None
    latest = candidates[-1]
    return latest, load_json(latest)


def metric_delta(current: Mapping[str, Any], previous: Mapping[str, Any] | None, key: str) -> float | None:
    if not previous:
        return None
    current_value = safe_number(current.get(key))
    previous_value = safe_number(previous.get(key))
    if current_value is None or previous_value is None:
        return None
    return current_value - previous_value


def direction_label(key: str, delta: float | None) -> str:
    if delta is None:
        return "NO_COMPARISON"
    if abs(delta) < 1e-12:
        return "UNCHANGED"
    lower_is_better = {"brier", "logloss", "mean_winner_rank", "unsafe_match_count"}
    if key in lower_is_better:
        return "IMPROVED" if delta < 0 else "WORSENED"
    return "IMPROVED" if delta > 0 else "WORSENED"


def next_prejump_refresh_window_from_report(
    refresh_report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not refresh_report:
        return None
    window = refresh_report.get("next_preferred_window")
    if not isinstance(window, Mapping):
        return None
    next_race = window.get("next_race")
    if not isinstance(next_race, Mapping):
        next_race = {}
    selected_races = refresh_report.get("selected_races")
    if not isinstance(selected_races, list):
        selected_races = []
    return {
        "schema_version": "shadow_autopilot_next_prejump_refresh_window_v1",
        "status": window.get("status"),
        "reason": window.get("reason"),
        "generated_at": refresh_report.get("generated_at"),
        "recommended_rerun_after_local": window.get("recommended_rerun_after_local"),
        "next_window_opens_at": window.get("next_window_opens_at"),
        "next_window_closes_at": window.get("next_window_closes_at"),
        "minutes_until_window_opens": window.get("minutes_until_window_opens"),
        "minutes_until_window_closes": window.get("minutes_until_window_closes"),
        "selected_count": int(refresh_report.get("selected_count") or 0),
        "total_races_found": int(refresh_report.get("total_races_found") or 0),
        "selected_race_count": len(selected_races),
        "next_race": {
            "race_id": next_race.get("race_id"),
            "date": next_race.get("date"),
            "venue": next_race.get("venue"),
            "race_number": next_race.get("race_number"),
            "race_time": next_race.get("race_time"),
            "jump_datetime": next_race.get("jump_datetime"),
            "minutes_to_jump": next_race.get("minutes_to_jump"),
            "bucket": next_race.get("bucket"),
            "selected": next_race.get("selected"),
            "race_url": next_race.get("race_url"),
        },
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def build_dashboard(
    *,
    generated_at: datetime,
    aggregate_metrics: Mapping[str, Any] | None,
    join_metrics: Mapping[str, Any] | None,
    aggregate_calibration: Mapping[str, Any] | None,
    aggregate_box_bias: Mapping[str, Any] | None,
    status_report: Mapping[str, Any] | None,
    sources: Mapping[str, Any],
    refresh_report: Mapping[str, Any] | None = None,
    odds_snapshot_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calibration = (aggregate_calibration or {}).get("slope_intercept") or {}
    probability_error = metric_from_source(
        aggregate_metrics=aggregate_metrics,
        join_metrics=join_metrics,
        key="probability_sum_max_error_joined_races",
    )
    dashboard = {
        "schema_version": "shadow_dashboard_v1",
        "generated_at": generated_at.isoformat(),
        "current_status": (status_report or {}).get("final_status"),
        "status_reasons": (status_report or {}).get("status_reasons") or [],
        "safe_joined_races": metric_from_source(
            aggregate_metrics=aggregate_metrics,
            join_metrics=join_metrics,
            key="safe_joined_race_count",
        )
        or 0,
        "pending_races": metric_from_source(
            aggregate_metrics=aggregate_metrics,
            join_metrics=join_metrics,
            key="pending_race_count",
        )
        or 0,
        "unsafe_matches": metric_from_source(
            aggregate_metrics=aggregate_metrics,
            join_metrics=join_metrics,
            key="unsafe_match_count",
        )
        or 0,
        "top1": metric_from_source(aggregate_metrics=aggregate_metrics, join_metrics=join_metrics, key="top1"),
        "top3": metric_from_source(aggregate_metrics=aggregate_metrics, join_metrics=join_metrics, key="top3"),
        "winner_rank": {
            "mean": metric_from_source(
                aggregate_metrics=aggregate_metrics,
                join_metrics=join_metrics,
                key="mean_winner_rank",
            ),
            "values": metric_from_source(
                aggregate_metrics=aggregate_metrics,
                join_metrics=join_metrics,
                key="winner_ranks",
            )
            or [],
        },
        "brier": metric_from_source(aggregate_metrics=aggregate_metrics, join_metrics=join_metrics, key="brier"),
        "logloss": metric_from_source(
            aggregate_metrics=aggregate_metrics,
            join_metrics=join_metrics,
            key="logloss",
        ),
        "calibration": {
            "status": calibration.get("status"),
            "slope": calibration.get("slope"),
            "intercept": calibration.get("intercept"),
            "sample_size": calibration.get("sample_size"),
            "positive_labels": calibration.get("positive_labels"),
            "negative_labels": calibration.get("negative_labels"),
        },
        "box_1_share": (aggregate_box_bias or {}).get("safe_joined_box_1_top_pick_share"),
        "probability_sum_status": probability_sum_status(probability_error),
        "calibration_method": metric_from_source(
            aggregate_metrics=aggregate_metrics,
            join_metrics=join_metrics,
            key="calibration_method",
        )
        or "power_gamma_2.4",
        "quarantined_features": list(WATCHED_QUARANTINED_FEATURES),
        "odds_snapshot": {
            "status": (odds_snapshot_status or {}).get("status"),
            "output_dir": (odds_snapshot_status or {}).get("output_dir"),
            "prediction_rows": (odds_snapshot_status or {}).get("prediction_rows"),
            "odds_candidate_rows": (odds_snapshot_status or {}).get("odds_candidate_rows"),
            "valid_pre_jump_dog_odds_rows": (odds_snapshot_status or {}).get(
                "valid_pre_jump_dog_odds_rows"
            ),
            "races_with_complete_valid_prejump_odds": (odds_snapshot_status or {}).get(
                "races_with_complete_valid_prejump_odds"
            ),
            "races_with_missing_odds_rows": (odds_snapshot_status or {}).get(
                "races_with_missing_odds_rows"
            ),
            "ev_output_rows": (odds_snapshot_status or {}).get("ev_output_rows", 0),
            "ev_calculation_status": (odds_snapshot_status or {}).get(
                "ev_calculation_status"
            ),
            "odds_analysis_status": (odds_snapshot_status or {}).get(
                "odds_analysis_status"
            ),
            "odds_analysis_blocker_counts": (odds_snapshot_status or {}).get(
                "odds_analysis_blocker_counts"
            )
            or {},
        },
        "sources": dict(sources),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    next_window = next_prejump_refresh_window_from_report(refresh_report)
    if next_window:
        dashboard["next_prejump_refresh_window"] = next_window
        dashboard["next_prejump_refresh_status"] = next_window.get("status")
        dashboard["recommended_rerun_after_local"] = next_window.get(
            "recommended_rerun_after_local"
        )
    return dashboard


def build_result_join_status(
    *,
    generated_at: datetime,
    latest_join_dir: Path | None,
    aggregate_dir: Path | None,
    join_metrics: Mapping[str, Any] | None,
    aggregate_metrics: Mapping[str, Any] | None,
    pending_payload: Mapping[str, Any] | None,
    unsafe_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "shadow_autopilot_result_join_status_v1",
        "generated_at": generated_at.isoformat(),
        "latest_join_artifact": relpath(latest_join_dir),
        "aggregate_artifact": relpath(aggregate_dir),
        "exact_identity_join_only": True,
        "fuzzy_join_allowed": False,
        "latest_join": {
            "joined_count": (join_metrics or {}).get("safe_joined_race_count", 0),
            "pending_count": (join_metrics or {}).get("pending_race_count", 0),
            "unsafe_count": (join_metrics or {}).get("unsafe_match_count", 0),
            "status": (join_metrics or {}).get("status"),
        },
        "cumulative": {
            "joined_count": (aggregate_metrics or {}).get("safe_joined_race_count", 0),
            "pending_count": (aggregate_metrics or {}).get("pending_race_count", 0),
            "unsafe_count": (aggregate_metrics or {}).get("unsafe_match_count", 0),
            "status": (aggregate_metrics or {}).get("status"),
        },
        "pending_results": pending_payload or {"pending_race_count": 0, "pending_results": []},
        "unsafe_result_matches": unsafe_payload
        or {"unsafe_match_count": 0, "unsafe_result_matches": []},
    }


def build_drift_reports(
    *,
    generated_at: datetime,
    timeseries: Sequence[Mapping[str, Any]],
    latest_replay_drift_path: Path | None,
    latest_replay_drift: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = dict(timeseries[-1]) if timeseries else {}
    previous = dict(timeseries[-2]) if len(timeseries) >= 2 else None
    metric_deltas = {
        key: {
            "delta": metric_delta(current, previous, key),
            "direction": direction_label(key, metric_delta(current, previous, key)),
        }
        for key in ("safe_joined_race_count", "top1", "top3", "mean_winner_rank", "brier", "logloss")
    }
    alerts: list[str] = []
    if current.get("unsafe_match_count"):
        alerts.append("unsafe_result_matches_present")
    if probability_sum_status(current.get("probability_sum_max_error_joined_races"))["status"] != "PASS":
        alerts.append("probability_sum_not_pass")
    if current.get("calibration_status") not in (None, "computed"):
        alerts.append("calibration_not_computed_on_forward_shadow")

    drift_report = {
        "schema_version": "shadow_autopilot_drift_report_v1",
        "generated_at": generated_at.isoformat(),
        "status": "REPORT_ONLY",
        "source_replay_drift_report": relpath(latest_replay_drift_path),
        "feature_population_drift": (latest_replay_drift or {}).get("feature_population_drift", []),
        "missingness_drift": (latest_replay_drift or {}).get("missingness_drift", []),
        "box_distribution_drift": {
            "replay": (latest_replay_drift or {}).get("box_distribution_drift"),
            "forward_shadow_box1_share": [
                {
                    "artifact": row.get("artifact"),
                    "safe_joined_race_count": row.get("safe_joined_race_count"),
                    "box1_share": row.get("box1_share"),
                }
                for row in timeseries
            ],
        },
        "calibration_drift": {
            "replay": (latest_replay_drift or {}).get("calibration_drift"),
            "forward_shadow": [
                {
                    "artifact": row.get("artifact"),
                    "safe_joined_race_count": row.get("safe_joined_race_count"),
                    "slope": row.get("calibration_slope"),
                    "intercept": row.get("calibration_intercept"),
                    "status": row.get("calibration_status"),
                }
                for row in timeseries
            ],
        },
        "probability_distribution_drift": {
            "replay": (latest_replay_drift or {}).get("probability_distribution_drift"),
            "forward_shadow_probability_sum_error": [
                {
                    "artifact": row.get("artifact"),
                    "safe_joined_race_count": row.get("safe_joined_race_count"),
                    "probability_sum_max_error": row.get("probability_sum_max_error_joined_races"),
                }
                for row in timeseries
            ],
        },
        "metric_deltas_from_previous_aggregate": metric_deltas,
        "alerts": alerts,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    drift_timeseries = {
        "schema_version": "shadow_autopilot_drift_timeseries_v1",
        "generated_at": generated_at.isoformat(),
        "aggregate_metric_timeseries": list(timeseries),
    }
    return drift_report, drift_timeseries


def build_promotion_readiness(
    *,
    generated_at: datetime,
    dashboard: Mapping[str, Any],
    target_joined_races: int,
) -> dict[str, Any]:
    current_joined = int(dashboard.get("safe_joined_races") or 0)
    pending = int(dashboard.get("pending_races") or 0)
    unsafe = int(dashboard.get("unsafe_matches") or 0)
    calibration = dashboard.get("calibration") or {}
    probability_status = (dashboard.get("probability_sum_status") or {}).get("status")
    blockers: list[str] = []
    if current_joined < target_joined_races:
        blockers.append("insufficient_forward_shadow_joined_races")
    if pending:
        blockers.append("pending_official_results_remain")
    if unsafe:
        blockers.append("unsafe_result_matches_present")
    if probability_status != "PASS":
        blockers.append("probability_sum_status_not_pass")
    if list(dashboard.get("quarantined_features") or []):
        blockers.append("same_distance_same_grade_features_remain_quarantined")

    calibration_status = str(calibration.get("status") or "not_computed")
    if current_joined < target_joined_races:
        decision = "NEED_MORE_RESULTS"
    elif calibration_status != "computed":
        decision = "NEED_CALIBRATION_REVIEW"
    else:
        decision = "READY_FOR_RELIABILITY_REVIEW"

    return {
        "schema_version": "shadow_promotion_readiness_tracker_v1",
        "generated_at": generated_at.isoformat(),
        "current_joined_race_count": current_joined,
        "target_joined_race_count": target_joined_races,
        "calibration_status": calibration_status,
        "reliability_status": (
            "INSUFFICIENT_FORWARD_SHADOW_SAMPLE"
            if current_joined < target_joined_races
            else "READY_FOR_REPORT_ONLY_RELIABILITY_REVIEW"
        ),
        "box_bias_status": {
            "box_1_share": dashboard.get("box_1_share"),
            "status": "WATCH" if safe_number(dashboard.get("box_1_share")) and safe_number(dashboard.get("box_1_share")) > 0.35 else "REPORT_ONLY",
        },
        "leakage_status": "REPORT_ONLY_NO_NEW_FEATURES_OR_LABELS",
        "outstanding_blockers": blockers,
        "decision": decision,
        "promotion_allowed": False,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def count_values(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return counts


def build_feature_activation_provenance_audit(
    *,
    prejump_metadata_report: Mapping[str, Any] | None,
    same_distance_history_provenance: Mapping[str, Any] | None = None,
    protected_paths_unchanged: bool,
    generated_at: datetime,
) -> dict[str, Any]:
    files = list((prejump_metadata_report or {}).get("files") or [])
    eligible_rows = [row for row in files if row.get("bucket") == "eligible"]
    rejected_rows = [
        row
        for row in files
        if row.get("rejected_metadata_sources") or row.get("fail_reasons")
    ]
    field_coverage = (prejump_metadata_report or {}).get("field_coverage") or {}
    target_metadata_readiness = (prejump_metadata_report or {}).get(
        "target_metadata_readiness"
    )

    return {
        "schema_version": "shadow_autopilot_feature_activation_provenance_audit_v1",
        "generated_at": generated_at.isoformat(),
        "source": "latest_daily_prejump_metadata_report",
        "protected_paths_unchanged": protected_paths_unchanged,
        "prejump_metadata_status": (prejump_metadata_report or {}).get("status"),
        "eligible_count": len(eligible_rows),
        "rejected_source_rows": len(rejected_rows),
        "target_metadata_readiness": target_metadata_readiness,
        "target_distance_sources": count_values(eligible_rows, "target_distance_source"),
        "target_grade_sources": count_values(eligible_rows, "target_grade_source"),
        "by_feature": {
            "target_distance_safe": {
                "present_rows": (
                    (field_coverage.get("target_distance") or {}).get("eligible_present_rows")
                    or 0
                )
            },
            "target_grade_safe": {
                "present_rows": (
                    (field_coverage.get("target_grade") or {}).get("eligible_present_rows")
                    or 0
                )
            },
        },
        "same_distance_same_grade_history_provenance": same_distance_history_provenance
        or {
            "status": "NOT_VERIFIED",
            "reason": "no_prior_history_repair_packet_attached_to_autopilot_gate",
            "required_source": "prior_dog_history",
            "required_history_cutoff": "strictly_before_target_race",
            "target_race_rows_allowed": 0,
            "post_outcome_rows_allowed": 0,
            "by_feature": {},
        },
        "unsafe_or_incomplete_metadata": (prejump_metadata_report or {}).get(
            "unsafe_or_incomplete_metadata"
        )
        or [],
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def feature_activation_gate_inputs(
    *,
    daily_dir: Path | None,
    shadow_model: Path | None,
    baseline_metrics: Path | None = None,
    candidate_metrics: Path | None = None,
) -> dict[str, Path | None]:
    return feature_activation_gate_input_paths(
        daily_dir=daily_dir,
        shadow_model=shadow_model,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )


def feature_activation_gate_status_for_skip(
    *,
    generated_at: datetime,
    reason: str,
    inputs: Mapping[str, Path | None],
) -> dict[str, Any]:
    return {
        "schema_version": "shadow_autopilot_feature_activation_gate_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": "SKIPPED",
        "reason": reason,
        "inputs": {key: relpath(value) for key, value in inputs.items()},
        "activation_allowed_features": [],
        "kept_quarantined_features": list(WATCHED_QUARANTINED_FEATURES),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def build_daily_status(
    *,
    generated_at: datetime,
    daily_manifest: Mapping[str, Any] | None,
    result_join_status: Mapping[str, Any],
    dashboard: Mapping[str, Any],
    timeseries: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    odds_snapshot_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = dict(timeseries[-1]) if timeseries else {}
    previous = dict(timeseries[-2]) if len(timeseries) >= 2 else None
    comparisons = {
        key: {
            "current": current.get(key),
            "previous": previous.get(key) if previous else None,
            "delta": metric_delta(current, previous, key),
            "direction": direction_label(key, metric_delta(current, previous, key)),
        }
        for key in ("safe_joined_race_count", "top1", "top3", "mean_winner_rank", "brier", "logloss")
    }
    return {
        "schema_version": "shadow_autopilot_daily_status_v1",
        "generated_at": generated_at.isoformat(),
        "what_happened_today": "Ran shadow autopilot collection, odds diagnostics, exact result join, aggregate dashboard, drift report, and readiness tracker.",
        "races_scored_today": int((daily_manifest or {}).get("race_count") or 0),
        "prediction_rows_today": int((daily_manifest or {}).get("prediction_rows") or 0),
        "results_joined_this_run": (result_join_status.get("latest_join") or {}).get("joined_count", 0),
        "odds_snapshot_status": (odds_snapshot_status or {}).get("status"),
        "odds_candidate_rows": (odds_snapshot_status or {}).get("odds_candidate_rows", 0),
        "valid_pre_jump_dog_odds_rows": (odds_snapshot_status or {}).get(
            "valid_pre_jump_dog_odds_rows",
            0,
        ),
        "races_with_complete_valid_prejump_odds": (odds_snapshot_status or {}).get(
            "races_with_complete_valid_prejump_odds",
            0,
        ),
        "races_with_missing_odds_rows": (odds_snapshot_status or {}).get(
            "races_with_missing_odds_rows",
            0,
        ),
        "ev_output_rows": (odds_snapshot_status or {}).get("ev_output_rows", 0),
        "odds_analysis_status": (odds_snapshot_status or {}).get(
            "odds_analysis_status"
        ),
        "odds_analysis_blocker_counts": (odds_snapshot_status or {}).get(
            "odds_analysis_blocker_counts"
        )
        or {},
        "cumulative_safe_joined_races": dashboard.get("safe_joined_races"),
        "metrics_improved_or_worsened": comparisons,
        "closer_to_promotion_review": (
            "YES_RESULTS_ACCUMULATED"
            if (comparisons.get("safe_joined_race_count") or {}).get("delta", 0) and (comparisons.get("safe_joined_race_count") or {}).get("delta", 0) > 0
            else "NO_NEW_SAFE_JOINS_YET"
        ),
        "readiness_decision": readiness.get("decision"),
        "next_prejump_refresh_status": dashboard.get("next_prejump_refresh_status"),
        "recommended_rerun_after_local": dashboard.get("recommended_rerun_after_local"),
        "next_prejump_race": (
            (dashboard.get("next_prejump_refresh_window") or {}).get("next_race")
            if isinstance(dashboard.get("next_prejump_refresh_window"), Mapping)
            else None
        ),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def summary_markdown(
    *,
    final_verdict: str,
    dashboard: Mapping[str, Any],
    readiness: Mapping[str, Any],
    result_join_status: Mapping[str, Any],
    activation_gate_status: Mapping[str, Any] | None = None,
    odds_snapshot_status: Mapping[str, Any] | None = None,
) -> str:
    activation_status = activation_gate_status or {}
    odds_status = odds_snapshot_status or dashboard.get("odds_snapshot") or {}
    next_window = dashboard.get("next_prejump_refresh_window") or {}
    next_race = next_window.get("next_race") or {}
    return "\n".join(
        [
            "# Shadow Autopilot V1",
            "",
            f"Final verdict: `{final_verdict}`",
            "",
            "Scope: forward-shadow evidence accumulation and report-only observability.",
            "",
            f"- Safe joined races: `{dashboard.get('safe_joined_races')}`",
            f"- Pending races: `{dashboard.get('pending_races')}`",
            f"- Unsafe matches: `{dashboard.get('unsafe_matches')}`",
            f"- Top1: `{dashboard.get('top1')}`",
            f"- Top3: `{dashboard.get('top3')}`",
            f"- Brier: `{dashboard.get('brier')}`",
            f"- LogLoss: `{dashboard.get('logloss')}`",
            f"- Calibration: `{dashboard.get('calibration')}`",
            f"- Box-1 share: `{dashboard.get('box_1_share')}`",
            f"- Probability sum: `{dashboard.get('probability_sum_status')}`",
            f"- Next pre-jump refresh status: `{next_window.get('status')}`",
            f"- Recommended rerun after: `{next_window.get('recommended_rerun_after_local')}`",
            f"- Next pre-jump race: `{next_race.get('race_id')}` at `{next_race.get('jump_datetime')}`",
            "",
            "## Result Joins",
            f"- Latest joined: `{(result_join_status.get('latest_join') or {}).get('joined_count')}`",
            f"- Cumulative joined: `{(result_join_status.get('cumulative') or {}).get('joined_count')}`",
            "",
            "## Odds Snapshot",
            f"- Status: `{odds_status.get('status')}`",
            f"- Output: `{odds_status.get('output_dir')}`",
            f"- Odds candidate rows: `{odds_status.get('odds_candidate_rows')}`",
            f"- Valid pre-jump dog odds rows: `{odds_status.get('valid_pre_jump_dog_odds_rows')}`",
            f"- Races with complete valid pre-jump odds: `{odds_status.get('races_with_complete_valid_prejump_odds')}`",
            f"- Races with missing odds rows: `{odds_status.get('races_with_missing_odds_rows')}`",
            f"- EV output rows: `{odds_status.get('ev_output_rows')}`",
            "",
            "## Feature Activation Gate",
            f"- Status: `{activation_status.get('status')}`",
            f"- Output: `{activation_status.get('output_dir')}`",
            f"- Activation allowed: `{activation_status.get('activation_allowed_features')}`",
            f"- Kept quarantined: `{activation_status.get('kept_quarantined_features')}`",
            "",
            "## Readiness",
            f"- Decision: `{readiness.get('decision')}`",
            f"- Blockers: `{readiness.get('outstanding_blockers')}`",
            "",
            "No training, production promotion, registry mutation, production pointer update, DB write, label write, TGR enablement, betting action, EV action, feature engineering, or calibration-method change was performed.",
            "",
        ]
    )


def shadow_status_markdown(dashboard: Mapping[str, Any], readiness: Mapping[str, Any]) -> str:
    next_window = dashboard.get("next_prejump_refresh_window") or {}
    next_race = next_window.get("next_race") or {}
    odds_snapshot = dashboard.get("odds_snapshot") or {}
    return "\n".join(
        [
            "# Shadow Status",
            "",
            f"Status: `{dashboard.get('current_status')}`",
            "",
            f"Safe joined races: `{dashboard.get('safe_joined_races')}`",
            f"Pending races: `{dashboard.get('pending_races')}`",
            f"Unsafe matches: `{dashboard.get('unsafe_matches')}`",
            f"Top1: `{dashboard.get('top1')}`",
            f"Top3: `{dashboard.get('top3')}`",
            f"Mean winner rank: `{(dashboard.get('winner_rank') or {}).get('mean')}`",
            f"Brier: `{dashboard.get('brier')}`",
            f"LogLoss: `{dashboard.get('logloss')}`",
            f"Calibration slope/intercept: `{dashboard.get('calibration')}`",
            f"Box-1 share: `{dashboard.get('box_1_share')}`",
            f"Probability sum status: `{dashboard.get('probability_sum_status')}`",
            f"Odds snapshot status: `{odds_snapshot.get('status')}`",
            f"Valid pre-jump dog odds rows: `{odds_snapshot.get('valid_pre_jump_dog_odds_rows')}`",
            f"Next pre-jump refresh status: `{next_window.get('status')}`",
            f"Recommended rerun after: `{next_window.get('recommended_rerun_after_local')}`",
            f"Next pre-jump race: `{next_race.get('race_id')}` at `{next_race.get('jump_datetime')}`",
            "",
            f"Readiness decision: `{readiness.get('decision')}`",
            "",
        ]
    )


def readiness_markdown(readiness: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Readiness Summary",
            "",
            f"Decision: `{readiness.get('decision')}`",
            f"Joined races: `{readiness.get('current_joined_race_count')}` / `{readiness.get('target_joined_race_count')}`",
            f"Calibration status: `{readiness.get('calibration_status')}`",
            f"Reliability status: `{readiness.get('reliability_status')}`",
            f"Box-bias status: `{readiness.get('box_bias_status')}`",
            f"Leakage status: `{readiness.get('leakage_status')}`",
            f"Outstanding blockers: `{readiness.get('outstanding_blockers')}`",
            "",
            "This tracker is not a promotion approval.",
            "",
        ]
    )


def daily_status_markdown(daily_status: Mapping[str, Any]) -> str:
    comparisons = daily_status.get("metrics_improved_or_worsened") or {}
    next_race = daily_status.get("next_prejump_race") or {}
    lines = [
        "# Daily Status",
        "",
        f"What happened today: {daily_status.get('what_happened_today')}",
        "",
        f"Races scored: `{daily_status.get('races_scored_today')}`",
        f"Results joined this run: `{daily_status.get('results_joined_this_run')}`",
        f"Odds snapshot status: `{daily_status.get('odds_snapshot_status')}`",
        f"Valid pre-jump dog odds rows: `{daily_status.get('valid_pre_jump_dog_odds_rows')}`",
        f"Races with complete valid pre-jump odds: `{daily_status.get('races_with_complete_valid_prejump_odds')}`",
        f"Races with missing odds rows: `{daily_status.get('races_with_missing_odds_rows')}`",
        f"EV output rows: `{daily_status.get('ev_output_rows')}`",
        f"Cumulative safe joined races: `{daily_status.get('cumulative_safe_joined_races')}`",
        f"Closer to promotion review: `{daily_status.get('closer_to_promotion_review')}`",
        f"Readiness decision: `{daily_status.get('readiness_decision')}`",
        f"Next pre-jump refresh status: `{daily_status.get('next_prejump_refresh_status')}`",
        f"Recommended rerun after: `{daily_status.get('recommended_rerun_after_local')}`",
        f"Next pre-jump race: `{next_race.get('race_id')}` at `{next_race.get('jump_datetime')}`",
        "",
        "## Metric Movement",
    ]
    for key, value in comparisons.items():
        lines.append(
            f"- {key}: `{value.get('previous')}` -> `{value.get('current')}` (`{value.get('direction')}`)"
        )
    lines.append("")
    return "\n".join(lines)


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "shadow_autopilot_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def final_verdict_for(
    *,
    steps: Sequence[Mapping[str, Any]],
    protected_paths_unchanged: bool,
    required_outputs_present: bool,
) -> str:
    if not protected_paths_unchanged or not required_outputs_present:
        return "NEEDS_MORE_TOOLING"
    failed_required = [
        step
        for step in steps
        if step.get("name")
        in {
            "refresh_prejump_races",
            "daily_shadow_run",
            "result_join",
            "aggregate_results",
            "status_report",
            "feature_activation_gate",
        }
        and step.get("returncode") != 0
    ]
    if failed_required:
        return "PARTIAL_AUTOMATION_READY"
    return "AUTOPILOT_READY"


def run_autopilot(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    run_id = args.run_id or now_id(generated_at)
    evidence_root = args.evidence_root
    output_dir = assert_output_dir_safe(
        args.output_dir or evidence_root / f"shadow_autopilot_v1_{run_id}"
    )
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    current_time = args.current_time or generated_at.isoformat()
    protected_before = protected_hashes()
    steps: list[dict[str, Any]] = []
    refresh_dir = output_dir / "refreshed_upcoming"
    shadow_model = args.shadow_model or latest_shadow_model(evidence_root)
    if shadow_model is None and not args.skip_shadow_run:
        raise RuntimeError("shadow_model_required_for_no_training_autopilot")

    if not args.skip_refresh:
        refresh_command = [
            *refresh_command_prefix(args.refresh_command_mode),
            str(ROOT / "scripts/refresh_prejump_upcoming.py"),
            "--upcoming-dir",
            str(refresh_dir),
            "--days-ahead",
            str(args.days_ahead),
            "--min-minutes",
            str(args.min_minutes),
            "--max-minutes",
            str(args.max_minutes),
            "--limit",
            str(args.refresh_limit),
            "--output",
            str(output_dir / "refresh_prejump_report.json"),
        ]
        if args.refresh_dry_run:
            refresh_command.append("--dry-run")
        steps.append(
            step_command(name="refresh_prejump_races", command=refresh_command, output_dir=output_dir)
        )
    else:
        write_json(
            output_dir / "refresh_prejump_report.json",
            {
                "status": "SKIPPED",
                "reason": "skip_refresh_requested",
                "dry_run": None,
                "no_snapshot_persist": True,
                "no_odds_capture": True,
                "no_result_ingest": True,
                "no_label_write": True,
                "no_retrain_or_promotion": True,
            },
        )

    input_dirs = list(args.input_dir or [])
    if not input_dirs:
        input_dirs = [refresh_dir]

    daily_dir = evidence_root / f"daily_race_ingest_shadow_{run_id}_autopilot"
    if not args.skip_shadow_run:
        daily_command = [
            sys.executable,
            str(ROOT / "scripts/daily_race_ingest_shadow_orchestrator.py"),
            "--mode",
            "full-dry-run",
            "--output-dir",
            str(daily_dir),
            "--current-time",
            current_time,
            "--db",
            str(args.db),
            "--score-command-mode",
            args.score_command_mode,
            "--shadow-model",
            str(shadow_model),
        ]
        for input_dir in input_dirs:
            daily_command.extend(["--input-dir", str(input_dir)])
        steps.append(step_command(name="daily_shadow_run", command=daily_command, output_dir=output_dir))
    else:
        daily_dir = args.shadow_run_dir or latest_artifact(evidence_root, "daily_race_ingest_shadow_", "shadow_manifest.json") or latest_artifact(evidence_root, "forward_shadow_run_", "shadow_manifest.json")

    odds_dir = evidence_root / f"shadow_odds_snapshot_{run_id}_autopilot"
    odds_report: dict[str, Any] | None = None
    should_collect_odds, odds_skip_reason, odds_prediction_rows = should_collect_shadow_odds_snapshot(
        daily_dir
    )
    if args.skip_odds_snapshot:
        odds_snapshot_status = build_shadow_odds_snapshot_status(
            generated_at=generated_at,
            odds_dir=None,
            odds_report=None,
            skipped_reason="skip_odds_snapshot_requested",
            prediction_rows=odds_prediction_rows,
        )
    elif should_collect_odds and daily_dir is not None:
        odds_command = shadow_odds_snapshot_command(
            daily_dir=daily_dir,
            odds_dir=odds_dir,
            db_path=args.db,
            current_time=current_time,
        )
        odds_step = step_command(
            name="shadow_odds_snapshot",
            command=odds_command,
            output_dir=output_dir,
        )
        steps.append(odds_step)
        odds_report = load_json(odds_dir / "shadow_odds_snapshot_report.json") or {}
        odds_snapshot_status = build_shadow_odds_snapshot_status(
            generated_at=generated_at,
            odds_dir=odds_dir,
            odds_report=odds_report or None,
            skipped_reason=(
                f"odds_snapshot_report_missing_returncode_{odds_step.get('returncode')}"
                if not odds_report
                else None
            ),
            prediction_rows=odds_prediction_rows,
            attempted=True,
            status_override="SHADOW_ODDS_SNAPSHOT_FAILED_NO_REPORT" if not odds_report else None,
        )
    else:
        odds_snapshot_status = build_shadow_odds_snapshot_status(
            generated_at=generated_at,
            odds_dir=None,
            odds_report=None,
            skipped_reason=odds_skip_reason,
            prediction_rows=odds_prediction_rows,
        )
    write_json(output_dir / "shadow_odds_snapshot_status.json", odds_snapshot_status)

    latest_join_dir: Path | None = None
    if not args.skip_result_join and daily_dir is not None and (daily_dir / "shadow_manifest.json").exists():
        latest_join_dir = evidence_root / f"forward_shadow_result_join_{run_id}_autopilot"
        join_command = [
            sys.executable,
            str(ROOT / "scripts/join_forward_shadow_results.py"),
            "--shadow-run-dir",
            str(daily_dir),
            "--output-dir",
            str(latest_join_dir),
            "--db",
            str(args.db),
            "--current-time",
            current_time,
        ]
        steps.append(step_command(name="result_join", command=join_command, output_dir=output_dir))
    else:
        latest_join_dir = latest_artifact(evidence_root, "forward_shadow_result_join_", "shadow_forward_metrics.json")

    aggregate_dir = evidence_root / f"forward_shadow_result_aggregate_{run_id}_autopilot"
    if not args.skip_aggregate:
        aggregate_command = [
            sys.executable,
            str(ROOT / "scripts/aggregate_forward_shadow_results.py"),
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(aggregate_dir),
        ]
        steps.append(step_command(name="aggregate_results", command=aggregate_command, output_dir=output_dir))
    else:
        aggregate_dir = latest_artifact(evidence_root, "forward_shadow_result_aggregate_", "aggregate_forward_metrics.json") or aggregate_dir

    status_dir = evidence_root / f"forward_shadow_status_{run_id}_autopilot"
    if not args.skip_status:
        status_command = [
            sys.executable,
            str(ROOT / "scripts/forward_shadow_status_report.py"),
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(status_dir),
            "--db",
            str(args.db),
            "--min-joined-races",
            str(args.min_joined_races),
        ]
        steps.append(step_command(name="status_report", command=status_command, output_dir=output_dir))
    else:
        status_dir = latest_artifact(evidence_root, "forward_shadow_status_", "forward_shadow_status_report.json") or status_dir

    aggregate_metrics_path = aggregate_dir / "aggregate_forward_metrics.json"
    challenger_activation_metrics = latest_challenger_activation_metric_paths(
        evidence_root,
        aggregate_metrics_path=aggregate_metrics_path,
    )
    activation_gate_inputs = feature_activation_gate_inputs(
        daily_dir=daily_dir,
        shadow_model=shadow_model,
        baseline_metrics=challenger_activation_metrics.get("baseline_metrics") or aggregate_metrics_path,
        candidate_metrics=challenger_activation_metrics.get("candidate_metrics"),
    )
    activation_dir = evidence_root / f"shadow_feature_activation_gate_{run_id}_autopilot"
    activation_provenance_path = output_dir / "feature_activation_provenance_audit.json"
    if daily_dir is None or not (daily_dir / "prejump_metadata_report.json").exists():
        activation_gate_status = feature_activation_gate_status_for_skip(
            generated_at=generated_at,
            reason="prejump_metadata_report_missing",
            inputs=activation_gate_inputs,
        )
    elif not activation_gate_inputs.get("parity_report"):
        activation_gate_status = feature_activation_gate_status_for_skip(
            generated_at=generated_at,
            reason="train_eval_feature_parity_report_missing",
            inputs=activation_gate_inputs,
        )
    else:
        protected_mid_unchanged = protected_hashes() == protected_before
        provenance_audit = build_feature_activation_provenance_audit(
            prejump_metadata_report=load_json(daily_dir / "prejump_metadata_report.json"),
            same_distance_history_provenance=load_json(
                activation_gate_inputs["same_distance_history_provenance"]
            )
            if activation_gate_inputs.get("same_distance_history_provenance")
            else None,
            protected_paths_unchanged=protected_mid_unchanged,
            generated_at=generated_at,
        )
        write_json(activation_provenance_path, provenance_audit)
        activation_command = [
            sys.executable,
            str(ROOT / "scripts/shadow_feature_activation_gate.py"),
            "--parity-report",
            str(activation_gate_inputs["parity_report"]),
            "--provenance-audit",
            str(activation_provenance_path),
            "--output-dir",
            str(activation_dir),
            "--min-shadow-joined-races",
            str(args.min_joined_races),
        ]
        if activation_gate_inputs.get("inactive_policy_report"):
            activation_command.extend(
                ["--inactive-policy-report", str(activation_gate_inputs["inactive_policy_report"])]
            )
        if activation_gate_inputs.get("matrix_audit"):
            activation_command.extend(["--matrix-audit", str(activation_gate_inputs["matrix_audit"])])
        if activation_gate_inputs.get("baseline_metrics"):
            activation_command.extend(["--baseline-metrics", str(activation_gate_inputs["baseline_metrics"])])
        if activation_gate_inputs.get("candidate_metrics"):
            activation_command.extend(["--candidate-metrics", str(activation_gate_inputs["candidate_metrics"])])
        steps.append(
            step_command(
                name="feature_activation_gate",
                command=activation_command,
                output_dir=output_dir,
            )
        )
        activation_report = load_json(activation_dir / "feature_activation_gate_report.json") or {}
        activation_gate_status = {
            "schema_version": "shadow_autopilot_feature_activation_gate_status_v1",
            "generated_at": generated_at.isoformat(),
            "status": activation_report.get("final_status")
            or "FEATURE_ACTIVATION_GATE_FAILED",
            "output_dir": relpath(activation_dir),
            "inputs": {key: relpath(value) for key, value in activation_gate_inputs.items()},
            "provenance_audit": relpath(activation_provenance_path),
            "activation_allowed_features": activation_report.get("activation_allowed_features")
            or [],
            "kept_quarantined_features": activation_report.get("kept_quarantined_features")
            or list(WATCHED_QUARANTINED_FEATURES),
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    write_json(output_dir / "feature_activation_gate_status.json", activation_gate_status)

    join_metrics = load_json(latest_join_dir / "shadow_forward_metrics.json") if latest_join_dir else None
    join_pending = load_json(latest_join_dir / "pending_results.json") if latest_join_dir else None
    join_unsafe = load_json(latest_join_dir / "unsafe_result_matches.json") if latest_join_dir else None
    aggregate_metrics = load_json(aggregate_dir / "aggregate_forward_metrics.json")
    aggregate_calibration = load_json(aggregate_dir / "aggregate_calibration_review.json")
    aggregate_box_bias = load_json(aggregate_dir / "aggregate_box_bias_review.json")
    status_report = load_json(status_dir / "forward_shadow_status_report.json")
    daily_manifest = load_json(daily_dir / "shadow_manifest.json") if daily_dir else None
    refresh_report = load_json(output_dir / "refresh_prejump_report.json")

    sources = {
        "autopilot_output_dir": relpath(output_dir),
        "refresh_report": relpath(output_dir / "refresh_prejump_report.json"),
        "refreshed_upcoming_dir": relpath(refresh_dir),
        "daily_shadow_run_dir": relpath(daily_dir),
        "result_join_dir": relpath(latest_join_dir),
        "aggregate_dir": relpath(aggregate_dir),
        "status_dir": relpath(status_dir),
        "shadow_model": relpath(shadow_model),
        "shadow_odds_snapshot": odds_snapshot_status.get("output_dir"),
        "shadow_odds_snapshot_status": relpath(output_dir / "shadow_odds_snapshot_status.json"),
        "feature_activation_gate": activation_gate_status.get("output_dir"),
        "feature_activation_gate_status": relpath(output_dir / "feature_activation_gate_status.json"),
    }
    dashboard = build_dashboard(
        generated_at=generated_at,
        aggregate_metrics=aggregate_metrics,
        join_metrics=join_metrics,
        aggregate_calibration=aggregate_calibration,
        aggregate_box_bias=aggregate_box_bias,
        status_report=status_report,
        sources=sources,
        refresh_report=refresh_report,
        odds_snapshot_status=odds_snapshot_status,
    )
    result_join_status = build_result_join_status(
        generated_at=generated_at,
        latest_join_dir=latest_join_dir,
        aggregate_dir=aggregate_dir,
        join_metrics=join_metrics,
        aggregate_metrics=aggregate_metrics,
        pending_payload=join_pending,
        unsafe_payload=join_unsafe,
    )
    join_history = build_join_history(evidence_root)
    aggregate_timeseries = build_aggregate_timeseries(evidence_root)
    daily_history = build_daily_manifest_history(evidence_root)
    replay_drift_path, replay_drift = latest_shadow_drift(evidence_root)
    drift_report, drift_timeseries = build_drift_reports(
        generated_at=generated_at,
        timeseries=aggregate_timeseries,
        latest_replay_drift_path=replay_drift_path,
        latest_replay_drift=replay_drift,
    )
    readiness = build_promotion_readiness(
        generated_at=generated_at,
        dashboard=dashboard,
        target_joined_races=args.target_joined_races,
    )
    daily_status = build_daily_status(
        generated_at=generated_at,
        daily_manifest=daily_manifest,
        result_join_status=result_join_status,
        dashboard=dashboard,
        timeseries=aggregate_timeseries,
        readiness=readiness,
        odds_snapshot_status=odds_snapshot_status,
    )

    protected_after = protected_hashes()
    protected_paths_unchanged = protected_before == protected_after

    write_json(output_dir / "shadow_dashboard.json", dashboard)
    write_json(output_dir / "result_join_status.json", result_join_status)
    write_json(
        output_dir / "cumulative_join_history.json",
        {
            "schema_version": "shadow_autopilot_cumulative_join_history_v1",
            "generated_at": generated_at.isoformat(),
            "join_history": join_history,
            "aggregate_metric_timeseries": aggregate_timeseries,
        },
    )
    write_json(output_dir / "drift_report.json", drift_report)
    write_json(output_dir / "drift_timeseries.json", drift_timeseries)
    write_json(output_dir / "promotion_readiness_tracker.json", readiness)
    write_json(output_dir / "DAILY_STATUS.json", daily_status)
    write_text(output_dir / "SHADOW_STATUS.md", shadow_status_markdown(dashboard, readiness))
    write_text(output_dir / "readiness_summary.md", readiness_markdown(readiness))
    write_text(output_dir / "DAILY_STATUS.md", daily_status_markdown(daily_status))

    orchestration_report = {
        "schema_version": "shadow_autopilot_orchestration_report_v1",
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "current_time": current_time,
        "steps": steps,
        "refresh_report": refresh_report,
        "sources": sources,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_paths_unchanged,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "required_output_files": [
            "SUMMARY.md",
            "shadow_dashboard.json",
            "result_join_status.json",
            "shadow_odds_snapshot_status.json",
            "feature_activation_gate_status.json",
            "cumulative_join_history.json",
            "drift_report.json",
            "promotion_readiness_tracker.json",
            "DAILY_STATUS.md",
            "DAILY_STATUS.json",
            "verification_results.txt",
            "final_status.txt",
        ],
    }
    write_json(output_dir / "shadow_orchestration_report.json", orchestration_report)
    run_manifest = {
        "schema_version": "shadow_autopilot_run_manifest_v1",
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "output_dir": relpath(output_dir),
        "entrypoint": "scripts/shadow_autopilot_v1.py",
        "phase_outputs": {
            "phase_1_shadow_orchestration": relpath(output_dir / "shadow_orchestration_report.json"),
            "phase_2_result_join": relpath(output_dir / "result_join_status.json"),
            "phase_2b_odds_snapshot": relpath(output_dir / "shadow_odds_snapshot_status.json"),
            "phase_3_dashboard": relpath(output_dir / "shadow_dashboard.json"),
            "phase_4_drift": relpath(output_dir / "drift_report.json"),
            "phase_5_readiness": relpath(output_dir / "promotion_readiness_tracker.json"),
            "phase_6_daily_status": relpath(output_dir / "DAILY_STATUS.json"),
            "phase_7_feature_activation_gate": relpath(output_dir / "feature_activation_gate_status.json"),
        },
        "source_artifacts": sources,
        "daily_manifest_history_count": len(daily_history),
        "join_history_count": len(join_history),
        "aggregate_timeseries_count": len(aggregate_timeseries),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    required_outputs = [
        output_dir / "SUMMARY.md",
        output_dir / "shadow_dashboard.json",
        output_dir / "result_join_status.json",
        output_dir / "shadow_odds_snapshot_status.json",
        output_dir / "feature_activation_gate_status.json",
        output_dir / "cumulative_join_history.json",
        output_dir / "drift_report.json",
        output_dir / "promotion_readiness_tracker.json",
        output_dir / "DAILY_STATUS.md",
        output_dir / "DAILY_STATUS.json",
        output_dir / "verification_results.txt",
        output_dir / "final_status.txt",
    ]
    required_outputs_present = all(path.exists() for path in required_outputs if path.name not in {"SUMMARY.md", "verification_results.txt", "final_status.txt"})
    final_verdict = final_verdict_for(
        steps=steps,
        protected_paths_unchanged=protected_paths_unchanged,
        required_outputs_present=required_outputs_present,
    )
    write_text(
        output_dir / "verification_results.txt",
        "\n".join(
            [
                f"run_id={run_id}",
                f"shadow_model={relpath(shadow_model)}",
                f"training_performed=False",
                f"promotion_performed=False",
                f"registry_mutation=False",
                f"production_pointer_update=False",
                f"db_write=False",
                f"label_write=False",
                f"tgr_enabled=False",
                f"betting_or_ev_action=False",
                f"shadow_odds_snapshot_status={odds_snapshot_status.get('status')}",
                f"shadow_odds_snapshot_ev_output_rows={odds_snapshot_status.get('ev_output_rows')}",
                f"feature_activation_gate_status={activation_gate_status.get('status')}",
                f"next_prejump_refresh_status={dashboard.get('next_prejump_refresh_status')}",
                f"recommended_rerun_after_local={dashboard.get('recommended_rerun_after_local')}",
                f"protected_paths_unchanged={protected_paths_unchanged}",
                f"required_outputs_present={required_outputs_present}",
                f"final_verdict={final_verdict}",
                "",
            ]
        ),
    )
    write_text(output_dir / "final_status.txt", final_verdict + "\n")
    write_text(
        output_dir / "SUMMARY.md",
        summary_markdown(
            final_verdict=final_verdict,
            dashboard=dashboard,
            readiness=readiness,
            result_join_status=result_join_status,
            activation_gate_status=activation_gate_status,
            odds_snapshot_status=odds_snapshot_status,
        ),
    )
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))

    return {
        "output_dir": relpath(output_dir),
        "final_verdict": final_verdict,
        "safe_joined_races": dashboard.get("safe_joined_races"),
        "pending_races": dashboard.get("pending_races"),
        "unsafe_matches": dashboard.get("unsafe_matches"),
        "readiness_decision": readiness.get("decision"),
        "feature_activation_gate_status": activation_gate_status.get("status"),
        "shadow_odds_snapshot_status": odds_snapshot_status.get("status"),
        "shadow_odds_snapshot_valid_prejump_rows": odds_snapshot_status.get(
            "valid_pre_jump_dog_odds_rows"
        ),
        "next_prejump_refresh_status": dashboard.get("next_prejump_refresh_status"),
        "recommended_rerun_after_local": dashboard.get("recommended_rerun_after_local"),
        "protected_paths_unchanged": protected_paths_unchanged,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--current-time")
    parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    parser.add_argument("--shadow-model", type=Path)
    parser.add_argument("--input-dir", action="append", type=Path, default=None)
    parser.add_argument("--days-ahead", type=int, default=0)
    parser.add_argument("--min-minutes", type=float, default=20.0)
    parser.add_argument("--max-minutes", type=float, default=160.0)
    parser.add_argument("--refresh-limit", type=int, default=16)
    parser.add_argument("--refresh-dry-run", action="store_true")
    parser.add_argument("--refresh-command-mode", choices=("auto", "python", "uv"), default="auto")
    parser.add_argument("--score-command-mode", choices=("auto", "python", "uv"), default="auto")
    parser.add_argument("--target-joined-races", type=int, default=DEFAULT_TARGET_JOINED_RACES)
    parser.add_argument("--min-joined-races", type=int, default=DEFAULT_MIN_JOINED_RACES_FOR_STATUS)
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--skip-shadow-run", action="store_true")
    parser.add_argument("--skip-odds-snapshot", action="store_true")
    parser.add_argument("--skip-result-join", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-status", action="store_true")
    parser.add_argument("--shadow-run-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_autopilot(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("final_verdict") != "NEEDS_MORE_TOOLING" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
