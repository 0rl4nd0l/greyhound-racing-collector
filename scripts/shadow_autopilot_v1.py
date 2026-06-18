#!/usr/bin/env python3
"""Shadow evidence accumulation autopilot V1.

This is a shadow evidence orchestration wrapper. It refreshes pre-jump inputs
into an isolated artifact directory, runs shadow-only scoring with an existing
shadow model, attempts exact official result joins, aggregates cumulative
forward-shadow evidence, and writes the required status packet.

It must not train, promote, mutate registries, update production pointers,
write labels, enable TGR, or emit betting/EV actions. DB writes are restricted
to explicitly enabled append-only live odds and official-result evidence rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
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
DEFAULT_ODDS_CAPTURE_MIN_MINUTES = 0.0
DEFAULT_ODDS_CAPTURE_MAX_MINUTES = 60.0
DEFAULT_ODDS_CAPTURE_REFRESH_LIMIT = 8
DEFAULT_AUTONOMOUS_ODDS_CAPTURE_LIMIT: int | None = None
DEFAULT_HISTORICAL_UNIFIED_EVIDENCE_REPORT_LIMIT = 500
DEFAULT_RESULT_BACKLOG_LIMIT = 128
DEFAULT_RESULT_BACKLOG_SHADOW_RUN_LIMIT = 200
DEFAULT_RESULT_BACKLOG_LOOKBACK_DAYS = 2
UV_REFRESH_PACKAGES = ("requests", "beautifulsoup4", "pandas")
UV_ODDS_CAPTURE_PACKAGES = (
    "requests",
    "beautifulsoup4",
    "pandas",
    "selenium",
    "webdriver-manager",
)
ODDS_CAPTURE_REQUIRED_MODULES = (
    "requests",
    "bs4",
    "pandas",
    "selenium",
    "webdriver_manager",
)
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
FIXED_PREJUMP_ODDS_CAPTURE_WINDOWS_MINUTES = (60, 30, 10, 2)
LIVE_ODDS_CAPTURE_REQUIRED_PROVENANCE_FIELDS = (
    "canonical_race_identity",
    "sportsbet_source_url",
    "sportsbet_source_race_identity",
    "scrape_timestamp",
    "market_type",
    "dog_level_win_odds",
    "sportsbet_box_source",
    "runner_name_box_match_status",
)
PROMOTION_DISTANCE_STATUS_FIELD_MAP = {
    "status": "promotion_distance_status",
    "promotion_ready": "promotion_distance_promotion_ready",
    "blockers": "promotion_distance_blockers",
    "sample_race_count": "promotion_distance_sample_race_count",
    "sample_runner_rows": "promotion_distance_sample_runner_rows",
    "source_rejected_live_odds_candidate_count": "promotion_distance_source_rejected_live_odds_candidate_count",
    "source_rows_with_rejected_live_odds_candidates": "promotion_distance_source_rows_with_rejected_live_odds_candidates",
    "source_rejected_live_odds_candidate_reason_counts": "promotion_distance_source_rejected_live_odds_candidate_reason_counts",
    "source_exclusion_reason_counts": "promotion_distance_source_exclusion_reason_counts",
    "source_odds_exclusion_reason_counts": "promotion_distance_source_odds_exclusion_reason_counts",
    "source_official_result_evidence_db_missing_race_ids": "promotion_distance_source_official_result_evidence_db_missing_race_ids",
    "source_official_result_evidence_db_requested_race_count": "promotion_distance_source_official_result_evidence_db_requested_race_count",
    "source_official_result_evidence_db_races_with_rows": "promotion_distance_source_official_result_evidence_db_races_with_rows",
    "source_official_result_runner_paths": "promotion_distance_source_official_result_runner_paths",
    "official_result_coverage": "promotion_distance_official_result_coverage",
    "official_result_coverage_requested_race_count": "promotion_distance_official_result_coverage_requested_race_count",
    "official_result_coverage_requested_race_count_source": "promotion_distance_official_result_coverage_requested_race_count_source",
    "official_result_coverage_legacy_requested_race_count_without_ids": "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids",
    "official_result_coverage_races_with_rows_count": "promotion_distance_official_result_coverage_races_with_rows_count",
    "official_result_coverage_missing_race_count": "promotion_distance_official_result_coverage_missing_race_count",
    "official_result_coverage_missing_exclusion_count": "promotion_distance_official_result_coverage_missing_exclusion_count",
    "official_result_runner_path_count": "promotion_distance_official_result_runner_path_count",
    "official_result_runner_paths_source_field": "promotion_distance_official_result_runner_paths_source_field",
    "best_candidate_key": "promotion_distance_best_candidate_key",
    "best_non_market_candidate_key": "promotion_distance_best_non_market_candidate_key",
    "best_non_market_top1_margin_gap": "promotion_distance_best_non_market_top1_margin_gap",
    "predeclared_residual_candidate_status": "promotion_distance_predeclared_residual_candidate_status",
    "predeclared_residual_triggered_race_count": "promotion_distance_predeclared_residual_triggered_race_count",
    "report": "promotion_distance_report",
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


def rooted_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def timing_aligned_rerun_manifest_phase_outputs(
    *,
    output_dir: Path,
    execution_status: Mapping[str, Any] | None,
) -> dict[str, str | None]:
    status = execution_status or {}
    return {
        "phase_2b_timing_aligned_prediction_rerun_plan": relpath(
            output_dir / "timing_aligned_prediction_rerun_plan.json"
        ),
        "phase_2b_timing_aligned_prediction_rerun_execution": relpath(
            output_dir / "timing_aligned_prediction_rerun_execution_status.json"
        ),
        "phase_2b_timing_aligned_prediction_rerun_output_dir": relpath(
            rooted_path(status.get("rerun_daily_shadow_run_dir"))
        ),
        "phase_2b_timing_aligned_prediction_rerun_odds_snapshot_dir": relpath(
            rooted_path(status.get("rerun_odds_snapshot_dir"))
        ),
    }


def promotion_distance_status_projection(
    high_accuracy_refinement_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = high_accuracy_refinement_status or {}
    return {
        output_key: status.get(source_key)
        for output_key, source_key in PROMOTION_DISTANCE_STATUS_FIELD_MAP.items()
    }


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def promotion_distance_official_result_coverage_fields(
    promotion_distance: Mapping[str, Any],
) -> dict[str, Any]:
    direct = promotion_distance.get("official_result_coverage")
    if isinstance(direct, Mapping):
        coverage = {
            "source": direct.get("source"),
            "requested_race_count": int_or_zero(direct.get("requested_race_count")),
            "requested_race_count_source": direct.get("requested_race_count_source"),
            "legacy_requested_race_count_without_ids": direct.get(
                "legacy_requested_race_count_without_ids"
            ),
            "races_with_rows_count": int_or_zero(
                direct.get("races_with_rows_count")
            ),
            "missing_race_count": int_or_zero(direct.get("missing_race_count")),
            "missing_exclusion_count": int_or_zero(
                direct.get("missing_exclusion_count")
            ),
            "missing_race_ids": list(direct.get("missing_race_ids") or []),
            "races_with_rows": list(direct.get("races_with_rows") or []),
            "runner_path_count": int_or_zero(direct.get("runner_path_count")),
            "runner_paths_source_field": direct.get("runner_paths_source_field"),
        }
    else:
        missing_race_ids = list(
            promotion_distance.get(
                "source_official_result_evidence_db_missing_race_ids"
            )
            or []
        )
        races_with_rows = list(
            promotion_distance.get("source_official_result_evidence_db_races_with_rows")
            or []
        )
        runner_paths = list(
            promotion_distance.get("source_official_result_runner_paths") or []
        )
        source_exclusions = dict(
            promotion_distance.get("source_exclusion_reason_counts") or {}
        )
        coverage = {
            "source": "promotion_distance_summary",
            "requested_race_count": int_or_zero(
                promotion_distance.get(
                    "source_official_result_evidence_db_requested_race_count"
                )
            ),
            "requested_race_count_source": promotion_distance.get(
                "source_official_result_evidence_db_requested_race_count_source"
            ),
            "legacy_requested_race_count_without_ids": promotion_distance.get(
                "source_official_result_evidence_db_legacy_requested_race_count_without_ids"
            ),
            "races_with_rows_count": len(races_with_rows),
            "missing_race_count": len(missing_race_ids),
            "missing_exclusion_count": int_or_zero(
                source_exclusions.get("official_result_missing")
            ),
            "missing_race_ids": missing_race_ids,
            "races_with_rows": races_with_rows,
            "runner_path_count": len(runner_paths),
            "runner_paths_source_field": (
                "promotion_distance_summary.source_official_result_runner_paths"
            ),
        }
    return {
        "promotion_distance_official_result_coverage": coverage,
        "promotion_distance_official_result_coverage_requested_race_count": coverage[
            "requested_race_count"
        ],
        "promotion_distance_official_result_coverage_requested_race_count_source": coverage[
            "requested_race_count_source"
        ],
        "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": coverage[
            "legacy_requested_race_count_without_ids"
        ],
        "promotion_distance_official_result_coverage_races_with_rows_count": coverage[
            "races_with_rows_count"
        ],
        "promotion_distance_official_result_coverage_missing_race_count": coverage[
            "missing_race_count"
        ],
        "promotion_distance_official_result_coverage_missing_exclusion_count": coverage[
            "missing_exclusion_count"
        ],
        "promotion_distance_official_result_runner_path_count": coverage[
            "runner_path_count"
        ],
        "promotion_distance_official_result_runner_paths_source_field": coverage[
            "runner_paths_source_field"
        ],
    }


def unified_evidence_official_result_coverage_fields(
    unified_status: Mapping[str, Any],
) -> dict[str, Any]:
    direct = unified_status.get("official_result_coverage")
    if isinstance(direct, Mapping):
        coverage = {
            "source": direct.get("source"),
            "requested_race_count": int_or_zero(direct.get("requested_race_count")),
            "requested_race_count_source": direct.get("requested_race_count_source"),
            "requested_race_ids": list(direct.get("requested_race_ids") or []),
            "races_with_rows_count": int_or_zero(
                direct.get("races_with_rows_count")
            ),
            "missing_race_count": int_or_zero(direct.get("missing_race_count")),
            "missing_exclusion_count": int_or_zero(
                direct.get("missing_exclusion_count")
            ),
            "missing_race_ids": list(direct.get("missing_race_ids") or []),
            "races_with_rows": list(direct.get("races_with_rows") or []),
            "runner_path_count": int_or_zero(direct.get("runner_path_count")),
            "runner_paths_source_field": direct.get("runner_paths_source_field"),
        }
    else:
        audit = unified_status.get("official_result_evidence_db_audit")
        if not isinstance(audit, Mapping):
            audit = {}
        missing_race_ids = list(audit.get("missing_race_ids") or [])
        races_with_rows = list(audit.get("race_ids_with_rows") or [])
        runner_paths = list(unified_status.get("official_result_runner_paths") or [])
        exclusions = dict(unified_status.get("exclusion_reason_counts") or {})
        coverage = {
            "source": "unified_evidence_dataset_status",
            "requested_race_count": int_or_zero(audit.get("race_ids_requested")),
            "requested_race_count_source": (
                "official_result_evidence_db_audit_requested_race_ids"
                if audit.get("requested_race_ids")
                else None
            ),
            "requested_race_ids": list(audit.get("requested_race_ids") or []),
            "races_with_rows_count": len(races_with_rows),
            "missing_race_count": len(missing_race_ids),
            "missing_exclusion_count": int_or_zero(
                exclusions.get("official_result_missing")
            ),
            "missing_race_ids": missing_race_ids,
            "races_with_rows": races_with_rows,
            "runner_path_count": len(runner_paths),
            "runner_paths_source_field": "official_result_runner_paths",
        }
    return {
        "official_result_coverage": coverage,
        "official_result_coverage_requested_race_count": coverage[
            "requested_race_count"
        ],
        "official_result_coverage_requested_race_count_source": coverage[
            "requested_race_count_source"
        ],
        "official_result_coverage_races_with_rows_count": coverage[
            "races_with_rows_count"
        ],
        "official_result_coverage_missing_race_count": coverage[
            "missing_race_count"
        ],
        "official_result_coverage_missing_exclusion_count": coverage[
            "missing_exclusion_count"
        ],
        "official_result_runner_path_count": coverage["runner_path_count"],
        "official_result_runner_paths_source_field": coverage[
            "runner_paths_source_field"
        ],
    }


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


def load_json_after_timeout_grace(
    path: Path,
    *,
    grace_seconds: float = 10.0,
    poll_seconds: float = 0.25,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while True:
        payload = load_json(path)
        if payload is not None:
            return payload
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.01, poll_seconds))


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


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


def run_id_from_daily_shadow_dir(daily_dir: Path | None) -> str | None:
    if daily_dir is None:
        return None
    name = daily_dir.name
    if not name.startswith("daily_race_ingest_shadow_"):
        return None
    return name.removeprefix("daily_race_ingest_shadow_").removesuffix(
        "_daemon_autopilot"
    )


def shadow_odds_snapshot_paths_for_daily_dir(
    *,
    evidence_root: Path,
    daily_dir: Path | None,
) -> list[Path]:
    run_id = run_id_from_daily_shadow_dir(daily_dir)
    if not run_id:
        return []
    candidates = [
        evidence_root
        / f"shadow_odds_snapshot_{run_id}_daemon_autopilot"
        / "shadow_odds_snapshot.jsonl",
        evidence_root
        / f"shadow_odds_snapshot_{run_id}_autopilot"
        / "shadow_odds_snapshot.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def autonomous_live_odds_capture_command(
    *,
    input_dirs: Sequence[Path],
    capture_dir: Path,
    db_path: Path,
    current_time: str,
    limit: int | None,
    execute: bool,
    allow_auto_scrape_odds: bool,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    command = list(command_prefix or odds_capture_command_prefix())
    command.extend(
        [
            str(ROOT / "scripts/autonomous_live_odds_capture.py"),
            "--output-dir",
            str(capture_dir),
            "--db",
            str(db_path),
            "--current-time",
            current_time,
        ]
    )
    for input_dir in input_dirs:
        command.extend(["--input-dir", str(input_dir)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if execute:
        command.append("--execute")
    if allow_auto_scrape_odds:
        command.append("--allow-auto-scrape-odds")
    return command


def autonomous_official_result_capture_command(
    *,
    target_date: str,
    upcoming_dir: Path | None,
    shadow_run_dir: Path | None = None,
    snapshot_dir: Path | None,
    output_dir: Path,
    db_path: Path,
    current_time: str | None = None,
    race_ids: Sequence[str] | None = None,
    require_ready_snapshot: bool = True,
    include_live_odds_backlog: bool = False,
    backlog_evidence_root: Path | None = None,
    backlog_limit: int | None = None,
    backlog_shadow_run_limit: int | None = None,
    backlog_lookback_days: int | None = None,
    execute_db_ingest: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/autonomous_official_result_capture.py"),
        "--date",
        target_date,
        "--output-dir",
        str(output_dir),
        "--db",
        str(db_path),
    ]
    if upcoming_dir is not None:
        command.extend(["--upcoming-dir", str(upcoming_dir)])
    if shadow_run_dir is not None:
        command.extend(["--shadow-run-dir", str(shadow_run_dir)])
    if current_time:
        command.extend(["--current-time", current_time])
    if snapshot_dir is not None:
        command.extend(["--snapshot-dir", str(snapshot_dir)])
    if require_ready_snapshot:
        command.append("--require-ready-snapshot")
    if include_live_odds_backlog:
        command.append("--include-live-odds-backlog")
        if backlog_evidence_root is not None:
            command.extend(["--backlog-evidence-root", str(backlog_evidence_root)])
        if backlog_limit is not None:
            command.extend(["--backlog-limit", str(backlog_limit)])
        if backlog_shadow_run_limit is not None:
            command.extend(["--backlog-shadow-run-limit", str(backlog_shadow_run_limit)])
        if backlog_lookback_days is not None:
            command.extend(["--backlog-lookback-days", str(backlog_lookback_days)])
    if execute_db_ingest:
        command.append("--execute-db-ingest")
    for race_id in race_ids or []:
        command.extend(["--race-id", race_id])
    return command


def current_step_time_iso() -> str:
    return datetime.now().astimezone().isoformat()


def unified_evidence_dataset_command(
    *,
    shadow_run_dir: Path,
    output_dir: Path,
    db_path: Path,
    odds_jsonl_paths: Sequence[Path] | None = None,
    official_result_runner_paths: Sequence[Path] | None = None,
    joined_shadow_prediction_paths: Sequence[Path] | None = None,
    join_eligibility_packet_paths: Sequence[Path] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/build_unified_evidence_dataset.py"),
        "--shadow-run-dir",
        str(shadow_run_dir),
        "--output-dir",
        str(output_dir),
        "--db",
        str(db_path),
    ]
    for path in odds_jsonl_paths or []:
        if path.exists():
            command.extend(["--odds-jsonl", str(path)])
    for path in official_result_runner_paths or []:
        if path.exists():
            command.extend(["--official-result-runners-jsonl", str(path)])
    for path in joined_shadow_prediction_paths or []:
        if path.exists():
            command.extend(["--joined-shadow-predictions-jsonl", str(path)])
    for path in join_eligibility_packet_paths or []:
        if path.exists():
            command.extend(["--join-eligibility-packet", str(path)])
    return command


def backlog_official_result_runner_paths(
    official_result_path: Path,
    *,
    row_count: int,
) -> list[Path]:
    if row_count <= 0:
        return []
    return [official_result_path]


def backlog_unified_shadow_run_dirs(
    result_capture_dir: Path,
) -> list[Path]:
    return [
        item["shadow_run_dir"]
        for item in backlog_unified_shadow_run_candidates(result_capture_dir)
    ]


def backlog_unified_shadow_run_candidates(
    result_capture_dir: Path,
) -> list[dict[str, Any]]:
    source_report = load_json(result_capture_dir / "shadow_run_candidate_source_report.json") or {}
    backlog = source_report.get("live_odds_backlog")
    if not isinstance(backlog, Mapping):
        return []
    by_path: dict[str, dict[str, Any]] = {}
    for report in backlog.get("shadow_run_reports") or []:
        if not isinstance(report, Mapping):
            continue
        if int(report.get("candidate_count") or 0) <= 0:
            continue
        path = rooted_path(report.get("backlog_shadow_run_dir") or report.get("shadow_run_dir"))
        if path is None:
            continue
        candidate_race_ids = [
            str(race_id).strip()
            for race_id in report.get("candidate_race_ids") or []
            if str(race_id).strip()
        ]
        if not candidate_race_ids:
            continue
        key = str(path.resolve())
        existing = by_path.setdefault(
            key,
            {
                "shadow_run_dir": path,
                "candidate_race_ids": [],
                "source_candidate_count": 0,
            },
        )
        existing["source_candidate_count"] += int(report.get("candidate_count") or 0)
        seen_race_ids = set(existing["candidate_race_ids"])
        for race_id in candidate_race_ids:
            if race_id not in seen_race_ids:
                existing["candidate_race_ids"].append(race_id)
                seen_race_ids.add(race_id)
    return list(by_path.values())


def official_result_quarantine_context_by_race(
    quarantine_path: Path,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}

    def int_list(values: Any) -> list[int]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return []
        parsed: list[int] = []
        for value in values:
            try:
                parsed.append(int(value))
            except (TypeError, ValueError):
                continue
        return parsed

    def int_value(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def participant_rows(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for participant in item.get("participants") or []:
            if not isinstance(participant, Mapping):
                continue
            try:
                box_number = int(participant.get("box_number"))
            except (TypeError, ValueError):
                continue
            dog_name = str(participant.get("dog_name") or "").strip()
            row = {"box_number": box_number}
            if dog_name:
                row["dog_name"] = dog_name
            rows.append(row)
        return rows

    def attempted_source_box_sets(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source in item.get("attempted_sources") or []:
            if not isinstance(source, Mapping):
                continue
            terminal_statuses = [
                terminal
                for terminal in source.get("terminal_statuses") or []
                if isinstance(terminal, Mapping)
            ]
            rows.append(
                {
                    "source": source.get("source"),
                    "status": source.get("status"),
                    "source_url": source.get("source_url"),
                    "result_boxes": int_list(source.get("raw_order")),
                    "dog_names_by_box": dict(
                        source.get("dog_names_by_box")
                        if isinstance(source.get("dog_names_by_box"), Mapping)
                        else {}
                    ),
                    "terminal_status_boxes": int_list(
                        [terminal.get("box_number") for terminal in terminal_statuses]
                    ),
                }
            )
        return rows

    def reserve_substitution_diagnostic(
        *,
        participant_boxes: Sequence[int],
        result_boxes_not_in_participants: Sequence[int],
        result_boxes_in_participants: Sequence[int],
        source_box_sets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        participant_box_set = set(participant_boxes)
        terminal_status_boxes = sorted(
            {
                int(box)
                for source in source_box_sets
                for box in source.get("terminal_status_boxes") or []
                if isinstance(box, int)
            }
        )
        scratched_participant_boxes = sorted(
            set(terminal_status_boxes).intersection(participant_box_set)
        )
        terminal_status_boxes_outside_participants = sorted(
            set(terminal_status_boxes) - participant_box_set
        )
        outside_result_boxes = sorted(set(result_boxes_not_in_participants))
        reserve_like_result_boxes = sorted(
            box for box in outside_result_boxes if box >= 9
        )
        possible_reserve_substitution = bool(
            outside_result_boxes
            and outside_result_boxes == reserve_like_result_boxes
            and len(scratched_participant_boxes) >= len(outside_result_boxes)
        )
        return {
            "classification": (
                "possible_reserve_substitution_manual_review_required"
                if possible_reserve_substitution
                else "unclassified_runner_set_mismatch_manual_review_required"
            ),
            "acceptance_status": "not_accepted_report_only",
            "result_boxes_outside_participants": outside_result_boxes,
            "result_boxes_inside_participants": sorted(set(result_boxes_in_participants)),
            "candidate_reserve_boxes": reserve_like_result_boxes,
            "scratched_participant_boxes": scratched_participant_boxes,
            "terminal_status_boxes": terminal_status_boxes,
            "terminal_status_boxes_outside_participants": (
                terminal_status_boxes_outside_participants
            ),
        }

    for row in read_jsonl(quarantine_path):
        race_id = str(row.get("race_id") or "").strip()
        if not race_id or race_id == "__browser__":
            continue
        item = row.get("item") if isinstance(row.get("item"), Mapping) else {}
        errors = [
            str(error)
            for error in (item.get("errors") if isinstance(item, Mapping) else []) or []
            if str(error).strip()
        ]
        source_urls: list[str] = []
        for source in (item.get("attempted_sources") if isinstance(item, Mapping) else []) or []:
            if not isinstance(source, Mapping):
                continue
            source_url = str(source.get("source_url") or "").strip()
            if source_url and source_url not in source_urls:
                source_urls.append(source_url)
        participant_boxes = int_list(
            item.get("participant_boxes")
            or [participant.get("box_number") for participant in participant_rows(item)]
        )
        result_boxes = [
            box
            for source in attempted_source_box_sets(item)
            for box in source.get("result_boxes") or []
        ]
        participant_box_set = set(participant_boxes)
        source_box_sets = attempted_source_box_sets(item)
        result_boxes_not_in_participants = sorted(
            {box for box in result_boxes if box not in participant_box_set}
        )
        result_boxes_in_participants = sorted(
            {box for box in result_boxes if box in participant_box_set}
        )
        context = contexts.setdefault(
            race_id,
            {
                "official_result_quarantine_reason": str(
                    row.get("reason") or "official_result_quarantined"
                ),
                "official_result_quarantine_errors": [],
                "official_result_quarantine_source_urls": [],
            },
        )
        for error in errors:
            if error not in context["official_result_quarantine_errors"]:
                context["official_result_quarantine_errors"].append(error)
        for source_url in source_urls:
            if source_url not in context["official_result_quarantine_source_urls"]:
                context["official_result_quarantine_source_urls"].append(source_url)
        if result_boxes_not_in_participants:
            context["official_result_quarantine_participant_source"] = item.get(
                "participant_source"
            )
            context["official_result_quarantine_participant_count"] = int_value(
                item.get("participant_count"), len(participant_boxes)
            )
            context["official_result_quarantine_participant_boxes"] = participant_boxes
            context["official_result_quarantine_result_boxes_not_in_participants"] = (
                result_boxes_not_in_participants
            )
            context["official_result_quarantine_result_boxes_in_participants"] = (
                result_boxes_in_participants
            )
            context["official_result_quarantine_participants"] = participant_rows(item)
            context["official_result_quarantine_attempted_source_box_sets"] = (
                source_box_sets
            )
            context["official_result_quarantine_reserve_substitution_diagnostic"] = (
                reserve_substitution_diagnostic(
                    participant_boxes=participant_boxes,
                    result_boxes_not_in_participants=result_boxes_not_in_participants,
                    result_boxes_in_participants=result_boxes_in_participants,
                    source_box_sets=source_box_sets,
                )
            )
    return contexts


def filtered_official_result_rows_for_race_ids(
    source_path: Path,
    output_path: Path,
    race_ids: Sequence[str],
) -> int:
    race_id_set = {str(race_id).strip() for race_id in race_ids if str(race_id).strip()}
    rows = [
        row
        for row in read_jsonl(source_path)
        if str(row.get("race_id") or "").strip() in race_id_set
    ]
    write_jsonl(output_path, rows)
    return len(rows)


def _bool_count(value: Any) -> int:
    return 1 if bool(value) else 0


def backlog_unified_gap_action_plan(
    race_values: Sequence[Mapping[str, Any]],
    *,
    official_result_gap_context_by_race: Mapping[str, Mapping[str, Any]] | None = None,
    top_limit: int = 20,
) -> dict[str, Any]:
    action_counts: dict[str, int] = {
        "collect_future_strict_prejump_odds": 0,
        "retry_official_result_capture_or_join": 0,
        "investigate_join_or_stage2_gap": 0,
        "monitor_non_sample_blocking_completion_gap": 0,
    }
    evidence_missing_reason_counts: dict[str, int] = {
        "strict_prejump_odds_missing": 0,
        "official_result_missing_only": 0,
        "join_or_stage2_gap": 0,
    }
    rows: list[dict[str, Any]] = []
    for item in race_values:
        has_evidence = bool(item.get("has_unified_evidence_instance"))
        has_official = bool(item.get("has_complete_official_result_instance"))
        has_strict_odds = bool(
            item.get("has_complete_strict_prejump_odds_instance")
        )
        race_id = str(item.get("race_id") or "").strip()
        official_gap_context = dict(
            (official_result_gap_context_by_race or {}).get(race_id) or {}
        )
        if has_evidence and has_official and has_strict_odds:
            continue
        if not has_strict_odds:
            action = "collect_future_strict_prejump_odds"
            reason = "strict_prejump_odds_missing"
        elif not has_official and official_gap_context:
            action = "inspect_quarantined_official_result_runner_set"
            reason = "official_result_quarantined_unsafe_match"
        elif not has_official:
            action = "retry_official_result_capture_or_join"
            reason = "official_result_missing_only"
        elif not has_evidence:
            action = "investigate_join_or_stage2_gap"
            reason = "join_or_stage2_gap"
        else:
            action = "monitor_non_sample_blocking_completion_gap"
            reason = "completion_gap_not_sample_blocking"
        action_counts[action] = action_counts.get(action, 0) + 1
        if not has_evidence:
            evidence_missing_reason_counts[reason] = (
                evidence_missing_reason_counts.get(reason, 0) + 1
            )
        rows.append(
            {
                "race_id": item.get("race_id"),
                "race_date": item.get("race_date"),
                "venue": item.get("venue"),
                "recommended_action": action,
                "evidence_missing_reason": reason if not has_evidence else None,
                "has_unified_evidence_instance": has_evidence,
                "has_complete_official_result_instance": has_official,
                "has_complete_strict_prejump_odds_instance": has_strict_odds,
                "dataset_instance_count": item.get("dataset_instance_count"),
                "row_count": item.get("row_count"),
                "official_result_missing_rows": item.get(
                    "official_result_missing_rows"
                ),
                "strict_prejump_odds_missing_rows": item.get(
                    "strict_prejump_odds_missing_rows"
                ),
                "unified_evidence_missing_rows": item.get(
                    "unified_evidence_missing_rows"
                ),
                **official_gap_context,
            }
        )

    def sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
        sample_blocking = 0 if row.get("has_unified_evidence_instance") else 1
        missing_rows = int(row.get("unified_evidence_missing_rows") or 0)
        return (sample_blocking, missing_rows, str(row.get("race_id") or ""))

    rows = sorted(rows, key=sort_key, reverse=True)
    return {
        "scope": "deduped_race_id_gap_actions",
        "action_counts": dict(sorted(action_counts.items())),
        "evidence_missing_reason_counts": dict(
            sorted(evidence_missing_reason_counts.items())
        ),
        "sample_blocking_gap_count": sum(
            1 for row in rows if not row.get("has_unified_evidence_instance")
        ),
        "top_gap_races": rows[:top_limit],
    }


def compact_unified_gap_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    allowed_fields = (
        "race_id",
        "race_date",
        "venue",
        "recommended_action",
        "action",
        "evidence_missing_reason",
        "has_unified_evidence_instance",
        "has_complete_official_result_instance",
        "has_complete_strict_prejump_odds_instance",
        "official_result_quarantine_reason",
        "official_result_quarantine_errors",
        "official_result_quarantine_source_urls",
        "official_result_quarantine_participant_source",
        "official_result_quarantine_participant_count",
        "official_result_quarantine_participant_boxes",
        "official_result_quarantine_result_boxes_not_in_participants",
        "official_result_quarantine_result_boxes_in_participants",
        "official_result_quarantine_participants",
        "official_result_quarantine_attempted_source_box_sets",
        "official_result_quarantine_reserve_substitution_diagnostic",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        compact = {
            key: row.get(key)
            for key in allowed_fields
            if row.get(key) not in (None, "", [], {})
        }
        if compact.get("race_id"):
            compact_rows.append(compact)
    return compact_rows


def backlog_unified_race_coverage_summary(
    reports: Sequence[Mapping[str, Any]],
    *,
    official_result_gap_context_by_race: Mapping[str, Mapping[str, Any]] | None = None,
    top_limit: int = 20,
) -> dict[str, Any]:
    race_instances: list[dict[str, Any]] = []
    for report in reports:
        dataset_path = rooted_path(report.get("dataset_jsonl"))
        if dataset_path is None:
            output_dir = rooted_path(report.get("output_dir"))
            dataset_path = output_dir / "unified_evidence_dataset.jsonl" if output_dir else None
        if dataset_path is None or not dataset_path.exists():
            continue
        rows_by_race: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(dataset_path):
            race_id = str(row.get("race_id") or "").strip()
            if not race_id:
                continue
            item = rows_by_race.setdefault(
                race_id,
                {
                    "dataset_jsonl": relpath(dataset_path),
                    "output_dir": report.get("output_dir"),
                    "race_id": race_id,
                    "race_date": row.get("race_date"),
                    "venue": row.get("venue"),
                    "row_count": 0,
                    "official_result_rows": 0,
                    "strict_prejump_odds_rows": 0,
                    "unified_evidence_eligible_rows": 0,
                },
            )
            item["row_count"] += 1
            item["official_result_rows"] += _bool_count(row.get("official_result_available"))
            item["strict_prejump_odds_rows"] += _bool_count(
                row.get("strict_prejump_odds_available")
            )
            item["unified_evidence_eligible_rows"] += _bool_count(
                row.get("unified_evidence_eligible")
            )
        for item in rows_by_race.values():
            row_count = int(item.get("row_count") or 0)
            official_rows = int(item.get("official_result_rows") or 0)
            strict_rows = int(item.get("strict_prejump_odds_rows") or 0)
            eligible_rows = int(item.get("unified_evidence_eligible_rows") or 0)
            item["official_result_missing_rows"] = max(0, row_count - official_rows)
            item["strict_prejump_odds_missing_rows"] = max(0, row_count - strict_rows)
            item["unified_evidence_missing_rows"] = max(0, row_count - eligible_rows)
            item["official_result_complete"] = row_count > 0 and official_rows == row_count
            item["strict_prejump_odds_complete"] = row_count > 0 and strict_rows == row_count
            item["unified_evidence_complete"] = row_count > 0 and eligible_rows == row_count
            item["unified_evidence_present"] = eligible_rows > 0
            race_instances.append(item)

    by_race: dict[str, dict[str, Any]] = {}
    for item in race_instances:
        race_id = str(item.get("race_id") or "").strip()
        summary = by_race.setdefault(
            race_id,
            {
                "race_id": race_id,
                "race_date": item.get("race_date"),
                "venue": item.get("venue"),
                "dataset_instance_count": 0,
                "row_count": 0,
                "official_result_missing_rows": 0,
                "strict_prejump_odds_missing_rows": 0,
                "unified_evidence_missing_rows": 0,
                "has_complete_official_result_instance": False,
                "has_complete_strict_prejump_odds_instance": False,
                "has_unified_evidence_instance": False,
                "has_complete_unified_evidence_instance": False,
            },
        )
        summary["dataset_instance_count"] += 1
        summary["row_count"] += int(item.get("row_count") or 0)
        summary["official_result_missing_rows"] += int(
            item.get("official_result_missing_rows") or 0
        )
        summary["strict_prejump_odds_missing_rows"] += int(
            item.get("strict_prejump_odds_missing_rows") or 0
        )
        summary["unified_evidence_missing_rows"] += int(
            item.get("unified_evidence_missing_rows") or 0
        )
        summary["has_complete_official_result_instance"] = bool(
            summary["has_complete_official_result_instance"]
            or item.get("official_result_complete")
        )
        summary["has_complete_strict_prejump_odds_instance"] = bool(
            summary["has_complete_strict_prejump_odds_instance"]
            or item.get("strict_prejump_odds_complete")
        )
        summary["has_unified_evidence_instance"] = bool(
            summary["has_unified_evidence_instance"]
            or item.get("unified_evidence_present")
        )
        summary["has_complete_unified_evidence_instance"] = bool(
            summary["has_complete_unified_evidence_instance"]
            or item.get("unified_evidence_complete")
        )
    for race_id, context in (official_result_gap_context_by_race or {}).items():
        if race_id in by_race:
            by_race[race_id].update(dict(context))

    def top_items(items: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
        return [
            {
                "race_id": item.get("race_id"),
                "race_date": item.get("race_date"),
                "venue": item.get("venue"),
                "row_count": item.get("row_count"),
                "official_result_missing_rows": item.get("official_result_missing_rows"),
                "strict_prejump_odds_missing_rows": item.get(
                    "strict_prejump_odds_missing_rows"
                ),
                "unified_evidence_missing_rows": item.get("unified_evidence_missing_rows"),
                "dataset_instance_count": item.get("dataset_instance_count"),
                "official_result_quarantine_reason": item.get(
                    "official_result_quarantine_reason"
                ),
                "official_result_quarantine_errors": item.get(
                    "official_result_quarantine_errors"
                ),
            }
            for item in sorted(
                items,
                key=lambda row: (
                    int(row.get(key) or 0),
                    int(row.get("unified_evidence_missing_rows") or 0),
                    str(row.get("race_id") or ""),
                ),
                reverse=True,
            )
            if int(item.get(key) or 0) > 0
        ][:top_limit]

    race_values = list(by_race.values())
    official_missing_by_date: dict[str, int] = {}
    strict_missing_by_date: dict[str, int] = {}
    for item in race_values:
        race_date = str(item.get("race_date") or "UNKNOWN")
        if not item.get("has_complete_official_result_instance"):
            official_missing_by_date[race_date] = official_missing_by_date.get(race_date, 0) + 1
        if not item.get("has_complete_strict_prejump_odds_instance"):
            strict_missing_by_date[race_date] = strict_missing_by_date.get(race_date, 0) + 1
    gap_action_plan = backlog_unified_gap_action_plan(
        race_values,
        official_result_gap_context_by_race=official_result_gap_context_by_race,
        top_limit=top_limit,
    )

    return {
        "scope": "dataset_race_instances_and_deduped_race_id",
        "dataset_race_instance_count": len(race_instances),
        "deduped_race_count": len(by_race),
        "race_instances_with_unified_evidence": sum(
            1 for item in race_instances if item.get("unified_evidence_present")
        ),
        "race_instances_with_complete_unified_evidence": sum(
            1 for item in race_instances if item.get("unified_evidence_complete")
        ),
        "race_instances_with_official_result_missing": sum(
            1 for item in race_instances if int(item.get("official_result_missing_rows") or 0) > 0
        ),
        "race_instances_with_strict_prejump_odds_missing": sum(
            1 for item in race_instances if int(item.get("strict_prejump_odds_missing_rows") or 0) > 0
        ),
        "deduped_races_with_unified_evidence": sum(
            1 for item in race_values if item.get("has_unified_evidence_instance")
        ),
        "deduped_races_without_unified_evidence": sum(
            1 for item in race_values if not item.get("has_unified_evidence_instance")
        ),
        "deduped_races_with_complete_unified_evidence_instance": sum(
            1 for item in race_values if item.get("has_complete_unified_evidence_instance")
        ),
        "deduped_races_without_complete_official_result_instance": sum(
            1 for item in race_values if not item.get("has_complete_official_result_instance")
        ),
        "deduped_races_without_complete_strict_prejump_odds_instance": sum(
            1 for item in race_values if not item.get("has_complete_strict_prejump_odds_instance")
        ),
        "deduped_races_without_complete_official_result_instance_by_date": dict(
            sorted(official_missing_by_date.items())
        ),
        "deduped_races_without_complete_strict_prejump_odds_instance_by_date": dict(
            sorted(strict_missing_by_date.items())
        ),
        "gap_action_plan": gap_action_plan,
        "top_official_result_missing_races": top_items(
            race_values,
            "official_result_missing_rows",
        ),
        "top_strict_prejump_odds_missing_races": top_items(
            race_values,
            "strict_prejump_odds_missing_rows",
        ),
    }


def artifact_odds_rejection_reason_counts_for_backlog_report(
    report: Mapping[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    source_counts = report.get("artifact_odds_rejection_reason_counts")
    if isinstance(source_counts, Mapping):
        for reason, count in source_counts.items():
            reason_text = str(reason).strip()
            if reason_text:
                counts[reason_text] = counts.get(reason_text, 0) + int(count or 0)
        return dict(sorted(counts.items()))

    for audit in report.get("artifact_odds_audits") or []:
        if not isinstance(audit, Mapping):
            continue
        for reason, count in (audit.get("rejection_reason_counts") or {}).items():
            reason_text = str(reason).strip()
            if not reason_text:
                continue
            counts[reason_text] = counts.get(reason_text, 0) + int(count or 0)
    return dict(sorted(counts.items()))


def _string_sequence(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def backlog_unified_official_result_coverage_summary(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    requested_ids: set[str] = set()
    races_with_rows: set[str] = set()
    missing_ids: set[str] = set()
    runner_paths: set[str] = set()
    requested_count_without_ids = 0
    missing_exclusion_count = 0
    legacy_requested_count_without_ids = 0
    runner_path_count_without_paths = 0

    for report in reports:
        direct = report.get("official_result_coverage")
        coverage = direct if isinstance(direct, Mapping) else {}
        requested = _string_sequence(coverage.get("requested_race_ids"))
        if requested:
            requested_ids.update(requested)
        else:
            requested_count_without_ids += int(coverage.get("requested_race_count") or 0)
        races_with_rows.update(_string_sequence(coverage.get("races_with_rows")))
        missing_ids.update(_string_sequence(coverage.get("missing_race_ids")))
        missing_exclusion_count += int(coverage.get("missing_exclusion_count") or 0)
        legacy_requested_count_without_ids += int(
            coverage.get("legacy_requested_race_count_without_ids") or 0
        )
        report_runner_paths = _string_sequence(report.get("official_result_runner_paths"))
        if report_runner_paths:
            runner_paths.update(report_runner_paths)
        else:
            runner_path_count_without_paths += int(coverage.get("runner_path_count") or 0)

    requested_race_ids = sorted(requested_ids)
    requested_race_count = (
        len(requested_race_ids)
        if requested_race_ids
        else requested_count_without_ids
    )
    requested_source = (
        "deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids"
        if requested_race_ids
        else "backlog_unified_evidence_dataset_official_result_coverage_source_count"
    )
    return {
        "source": "backlog_unified_evidence_dataset_reports",
        "requested_race_count": requested_race_count,
        "requested_race_count_source": requested_source,
        "requested_race_ids": requested_race_ids,
        "legacy_requested_race_count_without_ids": (
            legacy_requested_count_without_ids or requested_count_without_ids
        ),
        "races_with_rows_count": len(races_with_rows),
        "missing_race_count": len(missing_ids),
        "missing_race_ids": sorted(missing_ids),
        "races_with_rows": sorted(races_with_rows),
        "runner_path_count": len(runner_paths) or runner_path_count_without_paths,
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": missing_exclusion_count,
    }


def build_backlog_unified_evidence_status(
    *,
    generated_at: datetime,
    reports: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]] | None = None,
    skipped_reason: str | None = None,
    official_result_gap_context_by_race: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    failures = list(failures or [])
    exclusion_reason_counts: dict[str, int] = {}
    odds_exclusion_reason_counts: dict[str, int] = {}
    artifact_odds_rejection_reason_counts: dict[str, int] = {}
    rejected_live_odds_candidate_reason_counts: dict[str, int] = {}
    for report in reports:
        for source_key, target in (
            ("exclusion_reason_counts", exclusion_reason_counts),
            ("odds_exclusion_reason_counts", odds_exclusion_reason_counts),
            (
                "rejected_live_odds_candidate_reason_counts",
                rejected_live_odds_candidate_reason_counts,
            ),
        ):
            source_counts = report.get(source_key)
            if not isinstance(source_counts, Mapping):
                continue
            for reason, count in source_counts.items():
                reason_text = str(reason).strip()
                if not reason_text:
                    continue
                target[reason_text] = target.get(reason_text, 0) + int(count or 0)
        for reason, count in artifact_odds_rejection_reason_counts_for_backlog_report(
            report
        ).items():
            artifact_odds_rejection_reason_counts[reason] = (
                artifact_odds_rejection_reason_counts.get(reason, 0) + count
            )
    if skipped_reason:
        status = "SKIPPED"
    elif failures and not reports:
        status = "BACKLOG_UNIFIED_EVIDENCE_DATASETS_FAILED"
    elif failures:
        status = "BACKLOG_UNIFIED_EVIDENCE_DATASETS_PARTIAL"
    elif reports:
        status = "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    else:
        status = "BACKLOG_UNIFIED_EVIDENCE_DATASETS_EMPTY"
    race_coverage = backlog_unified_race_coverage_summary(
        reports,
        official_result_gap_context_by_race=official_result_gap_context_by_race,
    )
    gap_action_plan = race_coverage.get("gap_action_plan") or {}
    top_gap_races = list(gap_action_plan.get("top_gap_races") or [])
    top_official_result_missing_races = list(
        race_coverage.get("top_official_result_missing_races") or []
    )
    official_result_coverage = backlog_unified_official_result_coverage_summary(
        reports
    )
    return {
        "schema_version": "shadow_autopilot_backlog_unified_evidence_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": status,
        "skipped_reason": skipped_reason,
        "aggregation_scope": "per_dataset_totals_not_cross_dataset_deduped",
        "attempted_dataset_count": len(reports) + len(failures),
        "dataset_count": len(reports),
        "failed_dataset_count": len(failures),
        "row_count": sum(int(report.get("row_count") or 0) for report in reports),
        "race_count": sum(int(report.get("race_count") or 0) for report in reports),
        "rows_with_official_results": sum(
            int(report.get("rows_with_official_results") or 0) for report in reports
        ),
        "rows_with_strict_prejump_odds": sum(
            int(report.get("rows_with_strict_prejump_odds") or 0) for report in reports
        ),
        "rows_with_artifact_shadow_odds": sum(
            int(report.get("rows_with_artifact_shadow_odds") or 0)
            for report in reports
        ),
        "rows_with_artifact_shadow_odds_candidates": sum(
            int(report.get("rows_with_artifact_shadow_odds_candidates") or 0)
            for report in reports
        ),
        "artifact_shadow_odds_candidate_count": sum(
            int(report.get("artifact_shadow_odds_candidate_count") or 0)
            for report in reports
        ),
        "artifact_shadow_odds_selected_bucket_count": sum(
            int(report.get("artifact_shadow_odds_selected_bucket_count") or 0)
            for report in reports
        ),
        "artifact_odds_rows_seen": sum(
            int(report.get("artifact_odds_rows_seen") or 0) for report in reports
        ),
        "artifact_odds_rows_accepted": sum(
            int(report.get("artifact_odds_rows_accepted") or 0) for report in reports
        ),
        "artifact_odds_rows_rejected": sum(
            int(report.get("artifact_odds_rows_rejected") or 0) for report in reports
        ),
        "unified_evidence_eligible_rows": sum(
            int(report.get("unified_evidence_eligible_rows") or 0) for report in reports
        ),
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "odds_exclusion_reason_counts": dict(
            sorted(odds_exclusion_reason_counts.items())
        ),
        "artifact_odds_rejection_reason_counts": dict(
            sorted(artifact_odds_rejection_reason_counts.items())
        ),
        "rejected_live_odds_candidate_count": sum(
            int(report.get("rejected_live_odds_candidate_count") or 0)
            for report in reports
        ),
        "rows_with_rejected_live_odds_candidates": sum(
            int(report.get("rows_with_rejected_live_odds_candidates") or 0)
            for report in reports
        ),
        "rejected_live_odds_candidate_reason_counts": dict(
            sorted(rejected_live_odds_candidate_reason_counts.items())
        ),
        "official_result_coverage": official_result_coverage,
        "official_result_coverage_requested_race_count": official_result_coverage[
            "requested_race_count"
        ],
        "official_result_coverage_requested_race_count_source": official_result_coverage[
            "requested_race_count_source"
        ],
        "official_result_coverage_legacy_requested_race_count_without_ids": (
            official_result_coverage["legacy_requested_race_count_without_ids"]
        ),
        "official_result_coverage_races_with_rows_count": official_result_coverage[
            "races_with_rows_count"
        ],
        "official_result_coverage_missing_race_count": official_result_coverage[
            "missing_race_count"
        ],
        "official_result_coverage_missing_exclusion_count": official_result_coverage[
            "missing_exclusion_count"
        ],
        "official_result_evidence_db_requested_race_count": official_result_coverage[
            "requested_race_count"
        ],
        "official_result_evidence_db_missing_race_ids": official_result_coverage[
            "missing_race_ids"
        ],
        "official_result_evidence_db_races_with_rows": official_result_coverage[
            "races_with_rows"
        ],
        "official_result_runner_paths": sorted(
            {
                path
                for report in reports
                for path in _string_sequence(report.get("official_result_runner_paths"))
            }
        ),
        "race_coverage": race_coverage,
        "race_coverage_summary": race_coverage,
        "gap_action_plan": gap_action_plan,
        "sample_blocking_gap_count": gap_action_plan.get(
            "sample_blocking_gap_count", 0
        ),
        "gap_action_counts": dict(gap_action_plan.get("action_counts") or {}),
        "evidence_missing_reason_counts": dict(
            gap_action_plan.get("evidence_missing_reason_counts") or {}
        ),
        "top_gap_race_ids": [
            str(row.get("race_id"))
            for row in top_gap_races
            if isinstance(row, Mapping) and row.get("race_id")
        ],
        "top_gap_races": top_gap_races,
        "top_official_result_missing_race_ids": [
            str(row.get("race_id"))
            for row in top_official_result_missing_races
            if isinstance(row, Mapping) and row.get("race_id")
        ],
        "top_official_result_missing_races": top_official_result_missing_races,
        "datasets": [
            {
                "output_dir": report.get("output_dir"),
                "shadow_run_dir": report.get("shadow_run_dir"),
                "row_count": report.get("row_count"),
                "candidate_race_count": report.get("candidate_race_count"),
                "candidate_race_ids": report.get("candidate_race_ids"),
                "filtered_official_result_runner_rows": report.get(
                    "filtered_official_result_runner_rows"
                ),
                "filtered_official_result_runners_jsonl": report.get(
                    "filtered_official_result_runners_jsonl"
                ),
                "filtered_official_result_runners_empty": report.get(
                    "filtered_official_result_runners_empty"
                ),
                "rows_with_official_results": report.get("rows_with_official_results"),
                "rows_with_strict_prejump_odds": report.get("rows_with_strict_prejump_odds"),
                "rows_with_artifact_shadow_odds": report.get(
                    "rows_with_artifact_shadow_odds"
                )
                or 0,
                "rows_with_artifact_shadow_odds_candidates": report.get(
                    "rows_with_artifact_shadow_odds_candidates"
                )
                or 0,
                "artifact_shadow_odds_candidate_count": report.get(
                    "artifact_shadow_odds_candidate_count"
                )
                or 0,
                "artifact_shadow_odds_selected_bucket_count": report.get(
                    "artifact_shadow_odds_selected_bucket_count"
                )
                or 0,
                "artifact_odds_rows_seen": report.get("artifact_odds_rows_seen") or 0,
                "artifact_odds_rows_accepted": report.get("artifact_odds_rows_accepted")
                or 0,
                "artifact_odds_rows_rejected": report.get("artifact_odds_rows_rejected")
                or 0,
                "artifact_odds_rejection_reason_counts": report.get(
                    "artifact_odds_rejection_reason_counts"
                )
                or artifact_odds_rejection_reason_counts_for_backlog_report(report),
                "unified_evidence_eligible_rows": report.get("unified_evidence_eligible_rows"),
                "exclusion_reason_counts": report.get("exclusion_reason_counts") or {},
                "odds_exclusion_reason_counts": report.get(
                    "odds_exclusion_reason_counts"
                )
                or {},
                "rejected_live_odds_candidate_count": report.get(
                    "rejected_live_odds_candidate_count"
                )
                or 0,
                "rows_with_rejected_live_odds_candidates": report.get(
                    "rows_with_rejected_live_odds_candidates"
                )
                or 0,
                "rejected_live_odds_candidate_reason_counts": report.get(
                    "rejected_live_odds_candidate_reason_counts"
                )
                or {},
            }
            for report in reports
        ],
        "failures": [
            {
                "output_dir": failure.get("output_dir"),
                "shadow_run_dir": failure.get("shadow_run_dir"),
                "returncode": failure.get("returncode"),
            }
            for failure in failures
        ],
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def build_autonomous_official_result_capture_status(
    *,
    generated_at: datetime,
    capture_dir: Path | None,
    capture_report: Mapping[str, Any] | None,
    progress_report: Mapping[str, Any] | None = None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    in_progress: bool = False,
    returncode: int | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    report = capture_report or {}
    progress = progress_report or {}
    evidence_db_ingest = report.get("official_result_evidence_db_ingest")
    if not isinstance(evidence_db_ingest, Mapping):
        evidence_db_ingest = {}
    progress_path = None
    progress_attempts_path = None
    if capture_dir is not None:
        progress_path = relpath(capture_dir / "autonomous_official_result_capture_progress.json")
        progress_attempts_path = relpath(
            capture_dir / "autonomous_official_result_capture_attempts.progress.jsonl"
        )
    recovery_queue_path = report.get("live_odds_backlog_recovery_queue_path")
    if not recovery_queue_path and capture_dir is not None:
        recovery_queue_path = relpath(capture_dir / "live_odds_backlog_recovery_queue.json")
    runner_set_validation_path = report.get(
        "live_odds_backlog_runner_set_validation_path"
    )
    if not runner_set_validation_path and capture_dir is not None:
        runner_set_validation_path = relpath(
            capture_dir / "live_odds_backlog_runner_set_validation.json"
        )
    join_eligibility_packet_path = report.get(
        "live_odds_backlog_join_eligibility_packet_path"
    )
    if not join_eligibility_packet_path and capture_dir is not None:
        join_eligibility_packet_path = relpath(
            capture_dir / "live_odds_backlog_join_eligibility_packet.json"
        )
    join_eligibility_blocker_counts = report.get(
        "live_odds_backlog_join_eligibility_blocker_counts"
    )
    if not isinstance(join_eligibility_blocker_counts, Mapping):
        join_eligibility_blocker_counts = {}
    if skipped_reason:
        status = "SKIPPED"
    elif in_progress:
        status = "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_IN_PROGRESS"
    else:
        status = (
            report.get("final_status")
            or "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_FAILED_NO_REPORT"
        )
    return {
        "schema_version": "shadow_autopilot_autonomous_official_result_capture_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": status,
        "output_dir": relpath(capture_dir),
        "attempted": attempted,
        "in_progress": in_progress,
        "returncode": returncode,
        "timed_out": timed_out,
        "skipped_reason": skipped_reason,
        "candidate_count": int(report.get("candidate_count") or 0),
        "ingested_count": int(report.get("ingested_count") or 0),
        "failed_count": int(report.get("failed_count") or 0),
        "skipped_count": int(report.get("skipped_count") or 0),
        "skipped_reason_counts": dict(report.get("skipped_reason_counts") or {}),
        "awaiting_jump_race_count": int(report.get("awaiting_jump_race_count") or 0),
        "awaiting_jump_race_ids": list(report.get("awaiting_jump_race_ids") or []),
        "awaiting_jump_next_recheck_after_local": report.get(
            "awaiting_jump_next_recheck_after_local"
        ),
        "awaiting_jump_races": list(report.get("awaiting_jump_races") or []),
        "progress_path": progress_path,
        "progress_attempts_path": progress_attempts_path,
        "progress_candidate_count": int(progress.get("candidate_count") or 0),
        "progress_completed_count": int(progress.get("completed_count") or 0),
        "progress_status_counts": dict(progress.get("status_counts") or {}),
        "progress_active_candidate": (
            dict(progress.get("active_candidate"))
            if isinstance(progress.get("active_candidate"), Mapping)
            else None
        ),
        "official_result_race_rows": int(report.get("official_result_race_rows") or 0),
        "official_result_runner_rows": int(report.get("official_result_runner_rows") or 0),
        "quarantine_rows": int(report.get("quarantine_rows") or 0),
        "quarantined_race_ids": list(report.get("quarantined_race_ids") or []),
        "quarantine_reason_counts": dict(
            report.get("quarantine_reason_counts") or {}
        ),
        "quarantine_error_counts": dict(report.get("quarantine_error_counts") or {}),
        "quarantine_result_boxes_not_in_participants_counts": dict(
            report.get("quarantine_result_boxes_not_in_participants_counts") or {}
        ),
        "quarantine_runner_set_mismatch_samples": list(
            report.get("quarantine_runner_set_mismatch_samples") or []
        ),
        "official_result_evidence_db_ingest": dict(evidence_db_ingest),
        "official_result_evidence_db_ingest_status": evidence_db_ingest.get("status"),
        "official_result_evidence_db_execute": bool(evidence_db_ingest.get("execute")),
        "official_result_evidence_db_write_performed": bool(
            evidence_db_ingest.get("db_write_performed")
        ),
        "official_result_evidence_valid_race_rows": int(
            evidence_db_ingest.get("valid_race_rows") or 0
        ),
        "official_result_evidence_valid_runner_rows": int(
            evidence_db_ingest.get("valid_runner_rows") or 0
        ),
        "official_result_evidence_blocked_race_rows": int(
            evidence_db_ingest.get("blocked_race_rows") or 0
        ),
        "official_result_evidence_blocked_runner_rows": int(
            evidence_db_ingest.get("blocked_runner_rows") or 0
        ),
        "official_result_evidence_inserted_race_rows": int(
            evidence_db_ingest.get("inserted_race_rows") or 0
        ),
        "official_result_evidence_inserted_runner_rows": int(
            evidence_db_ingest.get("inserted_runner_rows") or 0
        ),
        "official_result_evidence_blocker_reason_counts": dict(
            evidence_db_ingest.get("blocker_reason_counts") or {}
        ),
        "live_odds_backlog_enabled": bool(report.get("live_odds_backlog_enabled")),
        "live_odds_backlog_lookback_days": int(
            report.get("live_odds_backlog_lookback_days") or 0
        ),
        "live_odds_backlog_target_dates": list(
            report.get("live_odds_backlog_target_dates") or []
        ),
        "live_odds_backlog_discovered_race_count": int(
            report.get("live_odds_backlog_discovered_race_count") or 0
        ),
        "live_odds_backlog_discovered_race_ids": list(
            report.get("live_odds_backlog_discovered_race_ids") or []
        ),
        "live_odds_backlog_candidate_race_count": int(
            report.get("live_odds_backlog_candidate_race_count") or 0
        ),
        "live_odds_backlog_candidate_race_ids": list(
            report.get("live_odds_backlog_candidate_race_ids") or []
        ),
        "live_odds_backlog_unresolved_race_count": int(
            report.get("live_odds_backlog_unresolved_race_count") or 0
        ),
        "live_odds_backlog_unresolved_race_ids": list(
            report.get("live_odds_backlog_unresolved_race_ids") or []
        ),
        "live_odds_backlog_unresolved_races": list(
            report.get("live_odds_backlog_unresolved_races") or []
        ),
        "live_odds_backlog_unresolved_reason_counts": dict(
            report.get("live_odds_backlog_unresolved_reason_counts") or {}
        ),
        "live_odds_backlog_unresolved_recovery_action_counts": dict(
            report.get("live_odds_backlog_unresolved_recovery_action_counts") or {}
        ),
        "live_odds_backlog_unresolved_alias_status_counts": dict(
            report.get("live_odds_backlog_unresolved_alias_status_counts") or {}
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_ids": list(
            report.get("live_odds_backlog_retryable_exact_shadow_match_race_ids") or []
        ),
        "live_odds_backlog_no_exact_shadow_match_race_ids": list(
            report.get("live_odds_backlog_no_exact_shadow_match_race_ids") or []
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_count": int(
            report.get("live_odds_backlog_retryable_exact_shadow_match_race_count") or 0
        ),
        "live_odds_backlog_no_exact_shadow_match_race_count": int(
            report.get("live_odds_backlog_no_exact_shadow_match_race_count") or 0
        ),
        "live_odds_backlog_awaiting_official_result_evidence_race_count": int(
            report.get(
                "live_odds_backlog_awaiting_official_result_evidence_race_count"
            )
            or 0
        ),
        "live_odds_backlog_awaiting_official_result_evidence_race_ids": list(
            report.get("live_odds_backlog_awaiting_official_result_evidence_race_ids")
            or []
        ),
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
            report.get(
                "live_odds_backlog_awaiting_official_result_evidence_authorized_action"
            )
        ),
        "live_odds_backlog_awaiting_official_result_recheck_ready_race_count": int(
            report.get(
                "live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
            )
            or 0
        ),
        "live_odds_backlog_recovery_queue_path": recovery_queue_path,
        "live_odds_backlog_recovery_queue_diagnostic_only": True,
        "live_odds_backlog_recovery_queue_join_acceptance_changed": False,
        "live_odds_backlog_recovery_queue_db_write_performed": False,
        "live_odds_backlog_runner_set_validation_path": runner_set_validation_path,
        "live_odds_backlog_runner_set_validation_retryable_race_count": int(
            report.get("live_odds_backlog_runner_set_validation_retryable_race_count")
            or 0
        ),
        "live_odds_backlog_runner_set_validation_exact_match_race_count": int(
            report.get("live_odds_backlog_runner_set_validation_exact_match_race_count")
            or 0
        ),
        "live_odds_backlog_runner_set_validation_blocked_race_count": int(
            report.get("live_odds_backlog_runner_set_validation_blocked_race_count")
            or 0
        ),
        "live_odds_backlog_runner_set_validation_diagnostic_only": bool(
            report.get("live_odds_backlog_runner_set_validation_diagnostic_only", True)
        ),
        "live_odds_backlog_runner_set_validation_join_authorized": bool(
            report.get("live_odds_backlog_runner_set_validation_join_authorized")
        ),
        "live_odds_backlog_runner_set_validation_db_write_performed": bool(
            report.get("live_odds_backlog_runner_set_validation_db_write_performed")
        ),
        "live_odds_backlog_join_eligibility_packet_path": join_eligibility_packet_path,
        "live_odds_backlog_join_eligibility_evaluated_race_count": int(
            report.get("live_odds_backlog_join_eligibility_evaluated_race_count") or 0
        ),
        "live_odds_backlog_join_eligibility_eligible_report_only_race_count": int(
            report.get(
                "live_odds_backlog_join_eligibility_eligible_report_only_race_count"
            )
            or 0
        ),
        "live_odds_backlog_join_eligibility_blocked_race_count": int(
            report.get("live_odds_backlog_join_eligibility_blocked_race_count") or 0
        ),
        "live_odds_backlog_join_eligibility_blocker_counts": dict(
            join_eligibility_blocker_counts
        ),
        "live_odds_backlog_join_eligibility_diagnostic_only": bool(
            report.get("live_odds_backlog_join_eligibility_diagnostic_only", True)
        ),
        "live_odds_backlog_join_eligibility_join_authorized": bool(
            report.get("live_odds_backlog_join_eligibility_join_authorized")
        ),
        "live_odds_backlog_join_eligibility_db_write_performed": bool(
            report.get("live_odds_backlog_join_eligibility_db_write_performed")
        ),
        "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": int(
            report.get(
                "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
            )
            or 0
        ),
        "shadow_run_candidate_source_report": report.get(
            "shadow_run_candidate_source_report"
        ),
        "no_write_guarantees": dict(
            report.get("no_write_guarantees")
            or {
                **NO_WRITE_GUARANTEES,
                "db_write": False,
                "label_write": False,
                "snapshot_rewrite": False,
                "manifest_rewrite": False,
            }
        ),
    }


def build_unified_evidence_dataset_status(
    *,
    generated_at: datetime,
    dataset_dir: Path | None,
    dataset_report: Mapping[str, Any] | None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    returncode: int | None = None,
) -> dict[str, Any]:
    report = dataset_report or {}
    official_result_coverage = unified_evidence_official_result_coverage_fields(report)
    return {
        "schema_version": "shadow_autopilot_unified_evidence_dataset_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": report.get("final_status")
        or ("SKIPPED" if skipped_reason else "UNIFIED_EVIDENCE_DATASET_FAILED_NO_REPORT"),
        "output_dir": relpath(dataset_dir),
        "attempted": attempted,
        "returncode": returncode,
        "skipped_reason": skipped_reason,
        "row_count": int(report.get("row_count") or 0),
        "race_count": int(report.get("race_count") or 0),
        "rows_with_official_results": int(report.get("rows_with_official_results") or 0),
        "rows_with_stage2_predictions": int(report.get("rows_with_stage2_predictions") or 0),
        "rows_with_strict_prejump_odds": int(report.get("rows_with_strict_prejump_odds") or 0),
        "label_evaluation_eligible_rows": int(
            report.get("label_evaluation_eligible_rows") or 0
        ),
        "stage2_evaluation_eligible_rows": int(
            report.get("stage2_evaluation_eligible_rows") or 0
        ),
        "odds_evaluation_eligible_rows": int(
            report.get("odds_evaluation_eligible_rows") or 0
        ),
        "unified_evidence_eligible_rows": int(
            report.get("unified_evidence_eligible_rows") or 0
        ),
        "exclusion_reason_counts": dict(report.get("exclusion_reason_counts") or {}),
        "odds_exclusion_reason_counts": dict(
            report.get("odds_exclusion_reason_counts") or {}
        ),
        "rejected_live_odds_candidate_count": int(
            report.get("rejected_live_odds_candidate_count") or 0
        ),
        "rows_with_rejected_live_odds_candidates": int(
            report.get("rows_with_rejected_live_odds_candidates") or 0
        ),
        "rejected_live_odds_candidate_reason_counts": dict(
            report.get("rejected_live_odds_candidate_reason_counts") or {}
        ),
        **official_result_coverage,
        "no_write_guarantees": dict(report.get("no_write_guarantees") or NO_WRITE_GUARANTEES),
    }


def high_accuracy_refinement_packet_command(
    *,
    unified_evidence_report: Path,
    output_dir: Path,
    stage2_predictions: Path | None = None,
    odds_augmented_report: Path | None = None,
    odds_gate_report: Path | None = None,
    backlog_unified_evidence_status: Path | None = None,
    promotion_distance_report: Path | None = None,
    reserve_substitution_preflight: Path | None = None,
    timing_aligned_rerun_plan: Path | None = None,
    timing_aligned_rerun_execution_status: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/build_high_accuracy_refinement_packet.py"),
        "--unified-evidence-report",
        str(unified_evidence_report),
        "--output-dir",
        str(output_dir),
    ]
    if stage2_predictions is not None and stage2_predictions.exists():
        command.extend(["--stage2-predictions", str(stage2_predictions)])
    if odds_augmented_report is not None and odds_augmented_report.exists():
        command.extend(["--odds-augmented-report", str(odds_augmented_report)])
    if odds_gate_report is not None and odds_gate_report.exists():
        command.extend(["--odds-gate-report", str(odds_gate_report)])
    if (
        backlog_unified_evidence_status is not None
        and backlog_unified_evidence_status.exists()
    ):
        command.extend(
            ["--backlog-unified-evidence-status", str(backlog_unified_evidence_status)]
        )
    if promotion_distance_report is not None and promotion_distance_report.exists():
        command.extend(["--promotion-distance-report", str(promotion_distance_report)])
    if (
        reserve_substitution_preflight is not None
        and reserve_substitution_preflight.exists()
    ):
        command.extend(
            ["--reserve-substitution-preflight", str(reserve_substitution_preflight)]
        )
    if timing_aligned_rerun_plan is not None and timing_aligned_rerun_plan.exists():
        command.extend(
            ["--timing-aligned-rerun-plan", str(timing_aligned_rerun_plan)]
        )
    if (
        timing_aligned_rerun_execution_status is not None
        and timing_aligned_rerun_execution_status.exists()
    ):
        command.extend(
            [
                "--timing-aligned-rerun-execution-status",
                str(timing_aligned_rerun_execution_status),
            ]
        )
    return command


def reserve_substitution_preflight_command(
    *,
    backlog_unified_evidence_status: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_official_result_reserve_substitution_preflight.py"),
        "--backlog-unified-evidence-status",
        str(backlog_unified_evidence_status),
        "--output-dir",
        str(output_dir),
    ]


def pre_race_gated_challenger_command(
    *,
    runner_matrix_csv: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_pre_race_gated_challenger_packet.py"),
        "--runner-matrix-csv",
        str(runner_matrix_csv),
        "--output-dir",
        str(output_dir),
    ]


def promotion_distance_report_command(
    *,
    rolling_report: Path,
    pre_race_gated_report: Path,
    high_accuracy_gate: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_promotion_distance_report.py"),
        "--rolling-report",
        str(rolling_report),
        "--pre-race-gated-report",
        str(pre_race_gated_report),
        "--high-accuracy-gate",
        str(high_accuracy_gate),
        "--output-dir",
        str(output_dir),
    ]


def odds_research_gate_report_path_from_snapshot_status(
    odds_snapshot_status: Mapping[str, Any] | None,
) -> Path | None:
    if not isinstance(odds_snapshot_status, Mapping):
        return None
    output_dir = rooted_path(odds_snapshot_status.get("output_dir"))
    if output_dir is None:
        return None
    explicit_path = rooted_path(odds_snapshot_status.get("odds_research_gate_report_path"))
    if explicit_path is not None and explicit_path.exists():
        try:
            explicit_path.resolve().relative_to(output_dir.resolve())
        except ValueError:
            return None
        return explicit_path
    candidate = output_dir / "odds_research_gate_report.json"
    return candidate if candidate.exists() else None


def rolling_model_comparison_command(
    *,
    unified_evidence_reports: Sequence[Path],
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/build_rolling_model_comparison_packet.py"),
        "--output-dir",
        str(output_dir),
    ]
    for report in unified_evidence_reports:
        if report.exists():
            command.extend(["--unified-evidence-report", str(report)])
    return command


def unified_report_dataset_path(report_path: Path, report: Mapping[str, Any]) -> Path:
    dataset_value = report.get("dataset_jsonl")
    if dataset_value:
        dataset_path = Path(str(dataset_value))
        if dataset_path.is_absolute():
            return dataset_path
        rooted_dataset_path = ROOT / dataset_path
        if rooted_dataset_path.exists():
            return rooted_dataset_path
        return report_path.parent / dataset_path
    return report_path.parent / "unified_evidence_dataset.jsonl"


def unified_report_eligible_rows(report_path: Path) -> int:
    report = load_json(report_path) or {}
    try:
        return int(report.get("unified_evidence_eligible_rows") or 0)
    except (TypeError, ValueError):
        return 0


def is_automatic_unified_evidence_report_path(report_path: Path) -> bool:
    dirname = report_path.parent.name
    if "_manual" in dirname or "_probe" in dirname or "_validation" in dirname:
        return False
    if "_daemon_autopilot" in dirname:
        return True
    marker = "_daemon_rejoin_"
    if marker not in dirname:
        return False
    suffix = dirname.rsplit(marker, 1)[1]
    return len(suffix) >= 3 and suffix[:3].isdigit() and (
        len(suffix) == 3 or suffix[3] == "_"
    )


def historical_unified_evidence_report_paths(
    evidence_root: Path,
    *,
    exclude_paths: Sequence[Path] = (),
    max_reports: int = DEFAULT_HISTORICAL_UNIFIED_EVIDENCE_REPORT_LIMIT,
) -> list[Path]:
    excluded = {path.resolve() for path in exclude_paths if path.exists()}
    candidates: list[Path] = []
    for report_path in sorted(
        evidence_root.glob("unified_evidence_dataset_*/unified_evidence_dataset_report.json")
    ):
        if not is_automatic_unified_evidence_report_path(report_path):
            continue
        if report_path.resolve() in excluded:
            continue
        report = load_json(report_path) or {}
        if not report or report.get("final_status") != "UNIFIED_EVIDENCE_DATASET_BUILT":
            continue
        if unified_report_eligible_rows(report_path) <= 0:
            continue
        if not unified_report_dataset_path(report_path, report).exists():
            continue
        candidates.append(report_path)
    return candidates[-max_reports:] if max_reports > 0 else candidates


def unique_sorted_report_paths(paths: Sequence[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        if path.exists():
            unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda path: path.as_posix())


def best_unified_evidence_report_path(paths: Sequence[Path]) -> Path | None:
    best: tuple[int, str, Path] | None = None
    for path in paths:
        eligible_rows = unified_report_eligible_rows(path)
        if eligible_rows <= 0:
            continue
        candidate = (eligible_rows, path.as_posix(), path)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else None


def build_rolling_model_comparison_status(
    *,
    generated_at: datetime,
    comparison_dir: Path | None,
    comparison_report: Mapping[str, Any] | None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    returncode: int | None = None,
) -> dict[str, Any]:
    report = comparison_report or {}
    candidate_metrics = report.get("candidate_metrics")
    if not isinstance(candidate_metrics, Mapping):
        candidate_metrics = {}
    sample_race_count = int(report.get("sample_race_count") or 0)
    minimum_races_for_review = int(report.get("minimum_races_for_review") or 0)
    sample_floor_value = report.get("sample_floor_met")
    sample_floor_met = (
        sample_floor_value
        if isinstance(sample_floor_value, bool)
        else (
            minimum_races_for_review > 0
            and sample_race_count >= minimum_races_for_review
        )
    )
    rank_first_sort = report.get("rank_first_sort")
    if isinstance(rank_first_sort, Sequence) and not isinstance(
        rank_first_sort,
        (str, bytes),
    ):
        rank_first_sort = [str(item) for item in rank_first_sort]
    else:
        rank_first_sort = []
    return {
        "schema_version": "shadow_autopilot_rolling_model_comparison_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": report.get("final_status")
        or ("SKIPPED" if skipped_reason else "ROLLING_MODEL_COMPARISON_FAILED_NO_REPORT"),
        "output_dir": relpath(comparison_dir),
        "attempted": attempted,
        "returncode": returncode,
        "skipped_reason": skipped_reason,
        "sample_scope": report.get("sample_scope"),
        "dedupe_race_id": report.get("dedupe_race_id"),
        "sample_race_count": sample_race_count,
        "sample_runner_rows": int(report.get("sample_runner_rows") or 0),
        "minimum_races_for_review": minimum_races_for_review,
        "sample_floor_met": sample_floor_met,
        "races_needed_for_review": max(
            0,
            minimum_races_for_review - sample_race_count,
        ),
        "candidate_count": int(report.get("candidate_count") or 0),
        "best_candidate_key": report.get("best_candidate_key"),
        "best_non_baseline_candidate_key": report.get("best_non_baseline_candidate_key"),
        "rank_first_sort": rank_first_sort,
        "source_report_count": len(list(report.get("source_reports") or [])),
        "source_rejected_live_odds_candidate_count": int(
            report.get("source_rejected_live_odds_candidate_count") or 0
        ),
        "source_rows_with_rejected_live_odds_candidates": int(
            report.get("source_rows_with_rejected_live_odds_candidates") or 0
        ),
        "source_rejected_live_odds_candidate_reason_counts": dict(
            report.get("source_rejected_live_odds_candidate_reason_counts") or {}
        ),
        "best_candidate_top1": candidate_metrics.get("top1"),
        "best_candidate_top3": candidate_metrics.get("top3"),
        "best_candidate_mean_winner_rank": candidate_metrics.get("mean_winner_rank"),
        "promotion_ready": False,
        "blockers": list(report.get("blockers") or []),
        "no_write_guarantees": dict(report.get("no_write_guarantees") or NO_WRITE_GUARANTEES),
    }


def build_high_accuracy_refinement_status(
    *,
    generated_at: datetime,
    packet_dir: Path | None,
    packet_report: Mapping[str, Any] | None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    returncode: int | None = None,
) -> dict[str, Any]:
    report = packet_report or {}
    promotion_gate = report.get("promotion_pr_gate")
    if not isinstance(promotion_gate, Mapping):
        promotion_gate = {}
    unified_summary = report.get("unified_evidence_summary")
    if not isinstance(unified_summary, Mapping):
        unified_summary = {}
    backlog_unified_summary = report.get("backlog_unified_evidence_summary")
    if not isinstance(backlog_unified_summary, Mapping):
        backlog_unified_summary = {}
    odds_gate_summary = report.get("odds_research_gate_summary")
    if not isinstance(odds_gate_summary, Mapping):
        odds_gate_summary = {}
    source_artifacts = report.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        source_artifacts = {}
    promotion_distance = report.get("promotion_distance_summary")
    if not isinstance(promotion_distance, Mapping):
        promotion_distance = {}
    promotion_distance_coverage = promotion_distance_official_result_coverage_fields(
        promotion_distance
    )
    reserve_preflight = report.get("reserve_substitution_preflight_summary")
    if not isinstance(reserve_preflight, Mapping):
        reserve_preflight = {}
    reserve_manual_review = report.get("reserve_substitution_manual_review_summary")
    if not isinstance(reserve_manual_review, Mapping):
        reserve_manual_review = {}
    reserve_policy_impact = report.get("reserve_substitution_policy_impact_summary")
    if not isinstance(reserve_policy_impact, Mapping):
        reserve_policy_impact = {}
    stages = report.get("stages")
    if not isinstance(stages, Mapping):
        stages = {}
    stage2_stage = stages.get("non_tgr_model_challenger")
    if not isinstance(stage2_stage, Mapping):
        stage2_stage = {}
    stage2_gate = stage2_stage.get("gate")
    if not isinstance(stage2_gate, Mapping):
        stage2_gate = {}
    odds_augmented_stage = stages.get("odds_augmented_model")
    if not isinstance(odds_augmented_stage, Mapping):
        odds_augmented_stage = {}
    odds_augmented_gate = odds_augmented_stage.get("gate")
    if not isinstance(odds_augmented_gate, Mapping):
        odds_augmented_gate = {}
    rolling_summary = odds_augmented_stage.get("rolling_model_comparison")
    if not isinstance(rolling_summary, Mapping):
        rolling_summary = {}
    cumulative_odds_evidence = odds_augmented_stage.get("cumulative_odds_evidence")
    if not isinstance(cumulative_odds_evidence, Mapping):
        cumulative_odds_evidence = {}
    unified_eligible_rows = int(unified_summary.get("unified_evidence_eligible_rows") or 0)
    backlog_unified_eligible_rows = int(
        backlog_unified_summary.get("unified_evidence_eligible_rows") or 0
    )
    best_available_unified_rows = max(unified_eligible_rows, backlog_unified_eligible_rows)
    rolling_sample_floor_value = rolling_summary.get("sample_floor_met")
    rolling_sample_floor_met = (
        rolling_sample_floor_value
        if isinstance(rolling_sample_floor_value, bool)
        else cumulative_odds_evidence.get("ready")
    )
    rolling_rank_first_sort = rolling_summary.get("rank_first_sort")
    if isinstance(rolling_rank_first_sort, Sequence) and not isinstance(
        rolling_rank_first_sort,
        (str, bytes),
    ):
        rolling_rank_first_sort = [str(item) for item in rolling_rank_first_sort]
    else:
        rolling_rank_first_sort = []
    return {
        "schema_version": "shadow_autopilot_high_accuracy_refinement_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": report.get("final_status")
        or ("SKIPPED" if skipped_reason else "HIGH_ACCURACY_REFINEMENT_PACKET_FAILED_NO_REPORT"),
        "output_dir": relpath(packet_dir),
        "attempted": attempted,
        "returncode": returncode,
        "skipped_reason": skipped_reason,
        "promotion_pr_gate_status": promotion_gate.get("status"),
        "promotion_pr_blockers": list(promotion_gate.get("blockers") or []),
        "stage2_status": stage2_stage.get("status"),
        "stage2_source_status": stage2_stage.get("source_status"),
        "stage2_prediction_rows": int(stage2_stage.get("stage2_prediction_rows") or 0),
        "stage2_predictions_path": stage2_stage.get("stage2_predictions_path"),
        "stage2_gate_status": stage2_gate.get("status"),
        "stage2_gate_blockers": list(stage2_gate.get("blockers") or []),
        "unified_evidence_status": unified_summary.get("status"),
        "unified_evidence_eligible_rows": unified_eligible_rows,
        "unified_evidence_rows_with_artifact_shadow_odds": int(
            unified_summary.get("rows_with_artifact_shadow_odds") or 0
        ),
        "unified_evidence_artifact_odds_rows_seen": int(
            unified_summary.get("artifact_odds_rows_seen") or 0
        ),
        "unified_evidence_artifact_odds_rows_accepted": int(
            unified_summary.get("artifact_odds_rows_accepted") or 0
        ),
        "unified_evidence_artifact_odds_rows_rejected": int(
            unified_summary.get("artifact_odds_rows_rejected") or 0
        ),
        "unified_evidence_artifact_odds_rejection_reason_counts": dict(
            unified_summary.get("artifact_odds_rejection_reason_counts") or {}
        ),
        "minimum_eligible_rows_for_review": int(
            unified_summary.get("minimum_eligible_rows_for_review") or 0
        ),
        "backlog_unified_evidence_status": backlog_unified_summary.get("status"),
        "backlog_unified_evidence_source_status": backlog_unified_summary.get(
            "source_status"
        ),
        "backlog_unified_evidence_dataset_count": int(
            backlog_unified_summary.get("dataset_count") or 0
        ),
        "backlog_unified_evidence_failed_dataset_count": int(
            backlog_unified_summary.get("failed_dataset_count") or 0
        ),
        "backlog_unified_evidence_eligible_rows": backlog_unified_eligible_rows,
        "backlog_unified_evidence_rows_with_artifact_shadow_odds": int(
            backlog_unified_summary.get("rows_with_artifact_shadow_odds") or 0
        ),
        "backlog_unified_evidence_artifact_odds_rows_seen": int(
            backlog_unified_summary.get("artifact_odds_rows_seen") or 0
        ),
        "backlog_unified_evidence_artifact_odds_rows_accepted": int(
            backlog_unified_summary.get("artifact_odds_rows_accepted") or 0
        ),
        "backlog_unified_evidence_artifact_odds_rows_rejected": int(
            backlog_unified_summary.get("artifact_odds_rows_rejected") or 0
        ),
        "backlog_unified_evidence_artifact_odds_rejection_reason_counts": dict(
            backlog_unified_summary.get("artifact_odds_rejection_reason_counts") or {}
        ),
        "backlog_unified_evidence_aggregation_scope": backlog_unified_summary.get(
            "aggregation_scope"
        ),
        "best_available_unified_evidence_eligible_rows": best_available_unified_rows,
        "best_available_unified_evidence_scope": (
            "backlog"
            if backlog_unified_eligible_rows >= unified_eligible_rows
            and backlog_unified_eligible_rows > 0
            else "current"
        ),
        "odds_research_gate_status": odds_gate_summary.get("status"),
        "odds_research_gate_complete_valid_prejump_odds_races": odds_gate_summary.get(
            "complete_valid_prejump_odds_races"
        ),
        "odds_augmented_model_status": odds_augmented_stage.get("status"),
        "odds_augmented_model_source_final_status": odds_augmented_stage.get(
            "source_final_status"
        ),
        "odds_augmented_gate_status": odds_augmented_gate.get("status"),
        "odds_augmented_gate_blockers": list(
            odds_augmented_gate.get("blockers") or []
        ),
        "rolling_model_comparison_status": (
            rolling_summary.get("status")
            or cumulative_odds_evidence.get("status")
        ),
        "rolling_model_comparison_sample_scope": (
            rolling_summary.get("sample_scope")
            or cumulative_odds_evidence.get("sample_scope")
        ),
        "rolling_model_comparison_sample_race_count": int(
            rolling_summary.get("sample_race_count")
            or cumulative_odds_evidence.get("sample_race_count")
            or 0
        ),
        "rolling_model_comparison_minimum_races_for_review": int(
            rolling_summary.get("minimum_races_for_review")
            or cumulative_odds_evidence.get(
                "minimum_complete_valid_prejump_odds_races"
            )
            or 0
        ),
        "rolling_model_comparison_sample_floor_met": rolling_sample_floor_met,
        "rolling_model_comparison_races_needed_for_review": int(
            rolling_summary.get("races_needed_for_review")
            or cumulative_odds_evidence.get("races_needed_for_review")
            or 0
        ),
        "rolling_model_comparison_candidate_count": int(
            rolling_summary.get("candidate_count") or 0
        ),
        "rolling_model_comparison_best_candidate_key": rolling_summary.get(
            "best_candidate_key"
        ),
        "rolling_model_comparison_best_non_baseline_candidate_key": (
            rolling_summary.get("best_non_baseline_candidate_key")
        ),
        "rolling_model_comparison_rank_first_sort": rolling_rank_first_sort,
        "promotion_distance_status": promotion_distance.get("status"),
        "promotion_distance_promotion_ready": bool(
            promotion_distance.get("promotion_ready", False)
        ),
        "promotion_distance_blockers": list(
            promotion_distance.get("blockers") or []
        ),
        "promotion_distance_sample_race_count": int(
            promotion_distance.get("sample_race_count") or 0
        ),
        "promotion_distance_sample_runner_rows": int(
            promotion_distance.get("sample_runner_rows") or 0
        ),
        "promotion_distance_source_rejected_live_odds_candidate_count": int(
            promotion_distance.get("source_rejected_live_odds_candidate_count") or 0
        ),
        "promotion_distance_source_rows_with_rejected_live_odds_candidates": int(
            promotion_distance.get("source_rows_with_rejected_live_odds_candidates")
            or 0
        ),
        "promotion_distance_source_rejected_live_odds_candidate_reason_counts": dict(
            promotion_distance.get("source_rejected_live_odds_candidate_reason_counts")
            or {}
        ),
        "promotion_distance_source_exclusion_reason_counts": dict(
            promotion_distance.get("source_exclusion_reason_counts") or {}
        ),
        "promotion_distance_source_odds_exclusion_reason_counts": dict(
            promotion_distance.get("source_odds_exclusion_reason_counts") or {}
        ),
        "promotion_distance_source_official_result_evidence_db_missing_race_ids": list(
            promotion_distance.get(
                "source_official_result_evidence_db_missing_race_ids"
            )
            or []
        ),
        "promotion_distance_source_official_result_evidence_db_requested_race_count": int(
            promotion_distance.get(
                "source_official_result_evidence_db_requested_race_count"
            )
            or 0
        ),
        "promotion_distance_source_official_result_evidence_db_races_with_rows": list(
            promotion_distance.get(
                "source_official_result_evidence_db_races_with_rows"
            )
            or []
        ),
        "promotion_distance_source_official_result_runner_paths": list(
            promotion_distance.get("source_official_result_runner_paths") or []
        ),
        **promotion_distance_coverage,
        "promotion_distance_best_candidate_key": (
            promotion_distance.get("best_candidate_key")
        ),
        "promotion_distance_best_non_market_candidate_key": (
            promotion_distance.get("best_non_market_candidate_key")
        ),
        "promotion_distance_best_non_market_top1_margin_gap": (
            promotion_distance.get("best_non_market_top1_margin_gap")
        ),
        "promotion_distance_predeclared_residual_candidate_status": (
            promotion_distance.get("predeclared_residual_candidate_status")
        ),
        "promotion_distance_predeclared_residual_triggered_race_count": (
            promotion_distance.get("predeclared_residual_triggered_race_count")
        ),
        "reserve_substitution_preflight_status": reserve_preflight.get("status"),
        "reserve_substitution_preflight_candidate_count": int(
            reserve_preflight.get("candidate_count") or 0
        ),
        "reserve_substitution_preflight_ready_for_policy_review_count": int(
            reserve_preflight.get("ready_for_policy_review_count") or 0
        ),
        "reserve_substitution_preflight_blocked_candidate_count": int(
            reserve_preflight.get("blocked_candidate_count") or 0
        ),
        "reserve_substitution_preflight_readiness_blocker_counts": dict(
            reserve_preflight.get("readiness_blocker_counts") or {}
        ),
        "reserve_substitution_preflight_dataset_join_blocker_counts": dict(
            reserve_preflight.get("dataset_join_blocker_counts") or {}
        ),
        "reserve_substitution_preflight_ready_race_ids": list(
            reserve_preflight.get("ready_race_ids") or []
        ),
        "reserve_substitution_preflight_blocked_race_ids": list(
            reserve_preflight.get("blocked_race_ids") or []
        ),
        "reserve_substitution_preflight_report": source_artifacts.get(
            "reserve_substitution_preflight"
        ),
        "reserve_substitution_manual_review_status": reserve_manual_review.get(
            "status"
        ),
        "reserve_substitution_manual_review_candidate_count": int(
            reserve_manual_review.get("candidate_count") or 0
        ),
        "reserve_substitution_manual_review_ready_candidate_count": int(
            reserve_manual_review.get("ready_candidate_count") or 0
        ),
        "reserve_substitution_manual_review_blocked_candidate_count": int(
            reserve_manual_review.get("blocked_candidate_count") or 0
        ),
        "reserve_substitution_manual_review_mapping_pair_count": int(
            reserve_manual_review.get("mapping_pair_count") or 0
        ),
        "reserve_substitution_manual_review_dataset_join_allowed": (
            reserve_manual_review.get("dataset_join_allowed")
        ),
        "reserve_substitution_manual_review_official_result_acceptance_allowed": (
            reserve_manual_review.get("official_result_acceptance_allowed")
        ),
        "reserve_substitution_manual_review_db_write": reserve_manual_review.get(
            "db_write"
        ),
        "reserve_substitution_manual_review_blockers": list(
            reserve_manual_review.get("blockers") or []
        ),
        "reserve_substitution_manual_review_ready_race_ids": list(
            reserve_manual_review.get("ready_race_ids") or []
        ),
        "reserve_substitution_manual_review_report": source_artifacts.get(
            "reserve_substitution_manual_review"
        ),
        "reserve_substitution_policy_impact_status": reserve_policy_impact.get(
            "status"
        ),
        "reserve_substitution_policy_impact_candidate_count": int(
            reserve_policy_impact.get("candidate_count") or 0
        ),
        "reserve_substitution_policy_impact_ready_candidate_count": int(
            reserve_policy_impact.get("ready_candidate_count") or 0
        ),
        "reserve_substitution_policy_impact_mapping_pair_count": int(
            reserve_policy_impact.get("mapping_pair_count") or 0
        ),
        "reserve_substitution_policy_impact_potential_runner_rows_blocked": int(
            reserve_policy_impact.get(
                "potential_official_result_runner_rows_blocked_by_policy"
            )
            or 0
        ),
        "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count": int(
            reserve_policy_impact.get("matched_backlog_top_gap_race_count") or 0
        ),
        "reserve_substitution_policy_impact_matched_backlog_top_gap_race_ids": list(
            reserve_policy_impact.get("matched_backlog_top_gap_race_ids") or []
        ),
        "reserve_substitution_policy_impact_dataset_join_allowed": (
            reserve_policy_impact.get("dataset_join_allowed")
        ),
        "reserve_substitution_policy_impact_official_result_acceptance_allowed": (
            reserve_policy_impact.get("official_result_acceptance_allowed")
        ),
        "reserve_substitution_policy_impact_db_write": (
            reserve_policy_impact.get("db_write")
        ),
        "reserve_substitution_policy_impact_blockers": list(
            reserve_policy_impact.get("blockers") or []
        ),
        "reserve_substitution_policy_impact_report": source_artifacts.get(
            "reserve_substitution_policy_impact_preview"
        ),
        "odds_research_gate_report": source_artifacts.get("odds_research_gate_report"),
        "backlog_unified_evidence_status_report": source_artifacts.get(
            "backlog_unified_evidence_status"
        ),
        "promotion_distance_report": source_artifacts.get("promotion_distance_report"),
        "timing_aligned_rerun_plan": source_artifacts.get(
            "timing_aligned_rerun_plan"
        ),
        "timing_aligned_rerun_execution_status": source_artifacts.get(
            "timing_aligned_rerun_execution_status"
        ),
        "protected_paths_unchanged": report.get("protected_paths_unchanged"),
        "no_write_guarantees": dict(report.get("no_write_guarantees") or NO_WRITE_GUARANTEES),
    }


def high_accuracy_timing_source_verification_lines(
    high_accuracy_status: Mapping[str, Any],
) -> list[str]:
    status = high_accuracy_status or {}
    return [
        (
            "high_accuracy_timing_aligned_rerun_plan="
            f"{status.get('timing_aligned_rerun_plan')}"
        ),
        (
            "high_accuracy_timing_aligned_rerun_execution_status="
            f"{status.get('timing_aligned_rerun_execution_status')}"
        ),
    ]


def build_autonomous_live_odds_capture_status(
    *,
    generated_at: datetime,
    capture_dir: Path | None,
    capture_report: Mapping[str, Any] | None,
    odds_capture_refresh_report: Mapping[str, Any] | None = None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    returncode: int | None = None,
    timed_out: bool = False,
    recovered_from_step_failure: bool = False,
) -> dict[str, Any]:
    report = capture_report or {}
    next_prejump_window = next_prejump_refresh_window_from_report(
        odds_capture_refresh_report
    )
    next_race = (
        next_prejump_window.get("next_race")
        if isinstance(next_prejump_window, Mapping)
        else {}
    )
    if not isinstance(next_race, Mapping):
        next_race = {}
    window_coverage = report.get("capture_window_coverage") if report else {}
    if not isinstance(window_coverage, Mapping):
        window_coverage = {}
    window_coverage_status_counts = dict(window_coverage.get("status_counts") or {})
    window_coverage_race_count = int(window_coverage.get("race_count") or 0)
    window_coverage_window_count = int(window_coverage.get("window_count") or 0)
    has_window_coverage = (
        window_coverage_race_count > 0
        or window_coverage_window_count > 0
        or bool(window_coverage_status_counts)
    )
    final_status = report.get("final_status") or (
        "SKIPPED" if skipped_reason else "AUTONOMOUS_LIVE_ODDS_CAPTURE_FAILED_NO_REPORT"
    )
    return {
        "schema_version": "shadow_autopilot_autonomous_live_odds_capture_status_v1",
        "generated_at": generated_at.isoformat(),
        "run_id": report.get("run_id"),
        "status": final_status,
        "final_status": final_status,
        "operator_status": report.get("status"),
        "runtime_action": report.get("runtime_action"),
        "readiness_decision": report.get("readiness_decision"),
        "output_dir": report.get("output_dir") or relpath(capture_dir),
        "attempted": attempted,
        "returncode": returncode,
        "timed_out": timed_out,
        "recovered_from_step_failure": recovered_from_step_failure,
        "skipped_reason": skipped_reason,
        "execute": bool(report.get("execute")) if report else False,
        "allow_auto_scrape_odds": bool(report.get("allow_auto_scrape_odds")) if report else False,
        "ready_count": int(report.get("ready_count") or 0),
        "validation_pass_count": int(report.get("validation_pass_count") or 0),
        "inserted_live_odds_rows": int(report.get("inserted_live_odds_rows") or 0),
        "status_counts": dict(report.get("status_counts") or {}),
        "blocked_attempt_count": int(report.get("blocked_attempt_count") or 0),
        "blocked_attempts": list(report.get("blocked_attempts") or []),
        "next_prejump_window": next_prejump_window,
        "next_window_opens_at": (
            next_prejump_window.get("next_window_opens_at")
            if isinstance(next_prejump_window, Mapping)
            else None
        ),
        "recommended_rerun_after_local": (
            next_prejump_window.get("recommended_rerun_after_local")
            if isinstance(next_prejump_window, Mapping)
            else None
        ),
        "next_race_id": next_race.get("race_id"),
        "capture_window_coverage_status_counts": window_coverage_status_counts,
        "capture_window_coverage_race_count": window_coverage_race_count,
        "capture_window_coverage_window_count": window_coverage_window_count,
        "capture_window_coverage_report": relpath(
            capture_dir / "autonomous_live_odds_capture_window_coverage.json"
            if capture_dir is not None and has_window_coverage
            else None
        ),
        "no_write_guarantees": dict(
            report.get("no_write_guarantees")
            or {
                **NO_WRITE_GUARANTEES,
                "db_write": False,
                "odds_history_write": False,
                "race_metadata_write": False,
            }
        ),
    }


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
    odds_research_readiness = report.get("odds_research_readiness")
    if not isinstance(odds_research_readiness, Mapping):
        odds_research_readiness = {}
    odds_research_gate = report.get("odds_research_gate")
    if not isinstance(odds_research_gate, Mapping):
        odds_research_gate = {}
    approved_odds_augmented_predictions = report.get(
        "approved_odds_augmented_predictions"
    )
    if not isinstance(approved_odds_augmented_predictions, Mapping):
        approved_odds_augmented_predictions = {}
    odds_gate_report_path = report.get("odds_research_gate_report_path")
    if not odds_gate_report_path and odds_dir is not None:
        candidate_gate_path = odds_dir / "odds_research_gate_report.json"
        if candidate_gate_path.exists():
            odds_gate_report_path = relpath(candidate_gate_path)
    approved_odds_augmented_prediction_report_path = report.get(
        "approved_odds_augmented_prediction_report_path"
    )
    if (
        not approved_odds_augmented_prediction_report_path
        and odds_dir is not None
    ):
        candidate_approved_path = odds_dir / "approved_odds_augmented_prediction_report.json"
        if candidate_approved_path.exists():
            approved_odds_augmented_prediction_report_path = relpath(
                candidate_approved_path
            )
    return {
        "schema_version": "shadow_autopilot_odds_snapshot_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": final_status,
        "final_status": final_status,
        "collection_attempted": bool(odds_report) if attempted is None else attempted,
        "skipped_reason": skipped_reason,
        "output_dir": relpath(odds_dir),
        "source_shadow_run_dir": report.get("shadow_run_dir"),
        "db_path": report.get("db_path"),
        "effective_prediction_timestamp": report.get("effective_prediction_timestamp"),
        "effective_prediction_timestamp_source": report.get(
            "effective_prediction_timestamp_source"
        ),
        "effective_feature_freeze_timestamp": report.get(
            "effective_feature_freeze_timestamp"
        ),
        "effective_feature_freeze_timestamp_source": report.get(
            "effective_feature_freeze_timestamp_source"
        ),
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
        "odds_research_readiness": odds_research_readiness,
        "odds_analysis_status": odds_research_readiness.get("status"),
        "odds_analysis_blocker_counts": odds_research_readiness.get("blocker_counts")
        or {},
        "odds_research_next_action": odds_research_readiness.get(
            "odds_research_next_action"
        ),
        "timing_aligned_prediction_rerun_required": bool(
            odds_research_readiness.get("timing_aligned_prediction_rerun_required")
        ),
        "timing_aligned_prediction_rerun_race_count": int(
            odds_research_readiness.get("timing_aligned_prediction_rerun_race_count")
            or 0
        ),
        "timing_aligned_prediction_rerun_race_ids": [
            race.get("race_id")
            for race in odds_research_readiness.get(
                "timing_aligned_prediction_rerun_races"
            )
            or []
            if isinstance(race, Mapping) and race.get("race_id")
        ],
        "timing_aligned_prediction_rerun_races": list(
            odds_research_readiness.get("timing_aligned_prediction_rerun_races")
            or []
        ),
        "timing_aligned_prediction_rerun_reason_counts": dict(
            odds_research_readiness.get(
                "timing_aligned_prediction_rerun_reason_counts"
            )
            or {}
        ),
        "odds_research_gate_status": odds_research_gate.get("status"),
        "odds_research_gate_report_path": odds_gate_report_path,
        "odds_research_gate_complete_valid_prejump_odds_races": odds_research_gate.get(
            "complete_valid_prejump_odds_races"
        ),
        "odds_research_gate_minimum_complete_valid_prejump_odds_races": odds_research_gate.get(
            "minimum_complete_valid_prejump_odds_races"
        ),
        "odds_research_gate_source_url_coverage_pct": odds_research_gate.get(
            "source_url_coverage_pct"
        ),
        "odds_research_gate_source_url_rows_missing": odds_research_gate.get(
            "source_url_rows_missing"
        ),
        "odds_research_gate_blocker_counts": dict(
            odds_research_gate.get("blocker_counts") or {}
        ),
        "approved_odds_augmented_candidate_key": approved_odds_augmented_predictions.get(
            "candidate_key"
        ),
        "approved_odds_augmented_prediction_status": approved_odds_augmented_predictions.get(
            "status"
        ),
        "approved_odds_augmented_ready_race_count": int(
            approved_odds_augmented_predictions.get("ready_race_count") or 0
        ),
        "approved_odds_augmented_blocked_race_count": int(
            approved_odds_augmented_predictions.get("blocked_race_count") or 0
        ),
        "approved_odds_augmented_prediction_rows": int(
            approved_odds_augmented_predictions.get("prediction_rows") or 0
        ),
        "approved_odds_augmented_prediction_report_path": (
            approved_odds_augmented_prediction_report_path
        ),
        "ev_eligible_rows": report.get("ev_eligible_rows", 0),
        "ev_output_rows": report.get("ev_output_rows", 0),
        "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
        "protected_paths_unchanged": report.get("protected_paths_unchanged"),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def _race_date_from_race_id(race_id: str) -> str | None:
    parts = [part.strip() for part in str(race_id or "").split(" - ")]
    if not parts:
        return None
    candidate = parts[-1]
    if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
        return candidate
    return None


def _iso_offset_suffix(value: Any) -> str:
    text = str(value or "")
    if len(text) >= 6 and text[-6] in {"+", "-"} and text[-3] == ":":
        return text[-6:]
    return "+10:00"


def _jump_datetime_from_metadata_row(
    row: Mapping[str, Any],
    *,
    default_offset: str,
) -> datetime | None:
    for key in ("jump_datetime", "jump_time_iso"):
        parsed = parse_datetime_or_none(row.get(key))
        if parsed is not None:
            return parsed
    race_date = str(row.get("race_date") or "")
    jump_time = str(row.get("jump_time") or "").strip()
    if not race_date or not jump_time:
        return None
    try:
        parsed_time = datetime.strptime(jump_time, "%I:%M %p").time()
    except ValueError:
        return None
    return datetime.fromisoformat(f"{race_date}T{parsed_time.isoformat()}{default_offset}")


def build_timing_aligned_prediction_rerun_plan(
    *,
    generated_at: datetime,
    odds_snapshot_status: Mapping[str, Any] | None,
    output_dir: Path,
    db_path: Path,
    shadow_model: Path | None,
    score_command_mode: str,
) -> dict[str, Any]:
    status = odds_snapshot_status or {}
    race_ids = [
        str(race_id)
        for race_id in status.get("timing_aligned_prediction_rerun_race_ids") or []
        if str(race_id or "").strip()
    ]
    hard_stops: list[str] = []
    if not status:
        hard_stops.append("odds_snapshot_status_missing")
    if status.get("timing_aligned_prediction_rerun_required") is not True:
        hard_stops.append("timing_aligned_prediction_rerun_not_required")
    if not race_ids:
        hard_stops.append("timing_aligned_prediction_rerun_race_ids_missing")

    source_shadow_run_dir = rooted_path(status.get("source_shadow_run_dir"))
    if source_shadow_run_dir is None or not source_shadow_run_dir.exists():
        hard_stops.append("source_shadow_run_dir_missing")
        source_manifest: dict[str, Any] = {}
        prejump_metadata_report: dict[str, Any] = {}
    else:
        source_manifest = load_json(source_shadow_run_dir / "shadow_manifest.json") or {}
        prejump_metadata_report = (
            load_json(source_shadow_run_dir / "prejump_metadata_report.json") or {}
        )
        if not source_manifest:
            hard_stops.append("source_shadow_manifest_missing")

    score_manifest = source_manifest.get("score_live_manifest")
    if not isinstance(score_manifest, Mapping):
        score_manifest = {}
    input_file_values = list(score_manifest.get("input_files") or [])
    if not input_file_values:
        hard_stops.append("source_shadow_input_files_missing")

    matched_input_files: list[Path] = []
    missing_input_race_ids: list[str] = []
    for race_id in race_ids:
        matches: list[Path] = []
        for value in input_file_values:
            candidate = rooted_path(value)
            if candidate is not None and race_id in candidate.name:
                matches.append(candidate)
        existing_matches = [path for path in matches if path.exists()]
        if not existing_matches:
            missing_input_race_ids.append(race_id)
        matched_input_files.extend(existing_matches)
    if missing_input_race_ids:
        hard_stops.append("rerun_input_files_missing_for_race_ids")

    offset = _iso_offset_suffix(status.get("effective_prediction_timestamp"))
    metadata_rows = (
        prejump_metadata_report.get("files")
        if isinstance(prejump_metadata_report.get("files"), list)
        else []
    )
    race_jump_contexts: list[dict[str, Any]] = []
    race_jump_by_id: dict[str, datetime] = {}
    for row in metadata_rows:
        if not isinstance(row, Mapping):
            continue
        race_id = canonical_race_identity(row)
        if race_id not in race_ids:
            continue
        jump_dt = _jump_datetime_from_metadata_row(row, default_offset=offset)
        if jump_dt is not None:
            race_jump_by_id[race_id] = jump_dt
        race_jump_contexts.append(
            {
                "race_id": race_id,
                "race_date": row.get("race_date"),
                "venue": row.get("venue"),
                "race_number": row.get("race_number"),
                "jump_time": row.get("jump_time"),
                "jump_datetime": jump_dt.isoformat() if jump_dt is not None else None,
                "generated_at_after_jump": (
                    generated_at >= jump_dt if jump_dt is not None else None
                ),
            }
        )
    missing_jump_race_ids = [
        race_id for race_id in race_ids if race_id not in race_jump_by_id
    ]
    if missing_jump_race_ids:
        hard_stops.append("rerun_jump_time_missing_for_race_ids")
    if any(generated_at >= jump_dt for jump_dt in race_jump_by_id.values()):
        hard_stops.append("timing_aligned_rerun_window_already_closed_after_jump")

    input_dirs: list[Path] = []
    seen_input_dirs: set[str] = set()
    for path in matched_input_files:
        key = str(path.parent)
        if key not in seen_input_dirs:
            input_dirs.append(path.parent)
            seen_input_dirs.add(key)

    model_path = shadow_model
    if model_path is None:
        model_path = rooted_path(score_manifest.get("model_source"))
    if model_path is None or not model_path.exists():
        hard_stops.append("shadow_model_missing_for_no_training_rerun")

    race_dates = sorted({date for race_id in race_ids for date in [_race_date_from_race_id(race_id)] if date})
    if len(race_dates) != 1:
        hard_stops.append("rerun_race_ids_must_share_single_date")
    planned_current_time = generated_at.isoformat()

    rerun_output_dir = (
        output_dir.parent
        / f"daily_race_ingest_shadow_{now_id(generated_at)}_timing_aligned_rerun"
    )
    planned_command = [
        sys.executable,
        str(ROOT / "scripts/daily_race_ingest_shadow_orchestrator.py"),
        "--mode",
        "full-dry-run",
        "--output-dir",
        str(rerun_output_dir),
        "--current-time",
        planned_current_time,
        "--db",
        str(db_path),
        "--score-command-mode",
        score_command_mode,
    ]
    if model_path is not None:
        planned_command.extend(["--shadow-model", str(model_path)])
    for input_dir in input_dirs:
        planned_command.extend(["--input-dir", str(input_dir)])

    status_value = (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
        if hard_stops
        else "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_READY_FOR_GUARDED_EXECUTION"
    )
    return {
        "schema_version": "timing_aligned_prediction_rerun_plan_v1",
        "generated_at": generated_at.isoformat(),
        "status": status_value,
        "execution_performed": False,
        "approval_required_before_execution": False,
        "operator_review_recommended": True,
        "rerun_reason": (
            "raw_fixed_window_odds_complete_but_rejected_after_previous_prediction"
        ),
        "odds_research_next_action": status.get("odds_research_next_action"),
        "timing_aligned_prediction_rerun_required": bool(
            status.get("timing_aligned_prediction_rerun_required")
        ),
        "timing_aligned_prediction_rerun_race_count": len(race_ids),
        "timing_aligned_prediction_rerun_race_ids": race_ids,
        "timing_aligned_prediction_rerun_reason_counts": dict(
            status.get("timing_aligned_prediction_rerun_reason_counts") or {}
        ),
        "source_shadow_run_dir": relpath(source_shadow_run_dir),
        "source_shadow_manifest": relpath(
            source_shadow_run_dir / "shadow_manifest.json"
            if source_shadow_run_dir is not None
            else None
        ),
        "source_prediction_timestamp": status.get("effective_prediction_timestamp"),
        "source_prediction_timestamp_source": status.get(
            "effective_prediction_timestamp_source"
        ),
        "source_feature_freeze_timestamp": status.get(
            "effective_feature_freeze_timestamp"
        ),
        "source_feature_freeze_timestamp_source": status.get(
            "effective_feature_freeze_timestamp_source"
        ),
        "source_stage2_prediction_rows": int(
            source_manifest.get("stage2_prediction_rows") or 0
        ),
        "stage2_predictions_required_first_class": True,
        "matched_input_files": [relpath(path) for path in matched_input_files],
        "matched_input_dirs": [relpath(path) for path in input_dirs],
        "missing_input_race_ids": missing_input_race_ids,
        "race_jump_contexts": race_jump_contexts,
        "missing_jump_race_ids": missing_jump_race_ids,
        "planned_classification_current_time": planned_current_time,
        "planned_output_dir": relpath(rerun_output_dir),
        "planned_command": planned_command,
        "post_rerun_required_steps": [
            "materialize_root_stage2_predictions",
            "collect_shadow_odds_snapshots_against_rerun",
            "build_unified_evidence_dataset_against_rerun",
            "refresh_rolling_model_comparison",
            "refresh_high_accuracy_refinement_packet",
        ],
        "hard_stops": sorted(set(hard_stops)),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def execute_timing_aligned_prediction_rerun_plan(
    *,
    generated_at: datetime,
    plan: Mapping[str, Any],
    output_dir: Path,
    db_path: Path,
    current_time: str,
    timeout_seconds: int,
    steps: list[dict[str, Any]],
    skip_odds_snapshot: bool = False,
) -> dict[str, Any]:
    plan_status = str(plan.get("status") or "")
    plan_hard_stops = sorted({str(item) for item in plan.get("hard_stops") or []})
    base_status: dict[str, Any] = {
        "schema_version": "timing_aligned_prediction_rerun_execution_v1",
        "generated_at": generated_at.isoformat(),
        "plan_status": plan_status,
        "plan_hard_stops": plan_hard_stops,
        "hard_stops": plan_hard_stops,
        "execution_performed": False,
        "rerun_daily_shadow_run_dir": plan.get("planned_output_dir"),
        "rerun_odds_snapshot_dir": None,
        "rerun_odds_snapshot_status": None,
        "stage2_materialization_status": None,
        "returncode": None,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    if plan_status != "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_READY_FOR_GUARDED_EXECUTION":
        return {
            **base_status,
            "status": "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY",
            "skip_reason": "plan_not_ready_for_guarded_execution",
        }

    command = [str(item) for item in plan.get("planned_command") or []]
    rerun_daily_dir = rooted_path(plan.get("planned_output_dir"))
    if not command or rerun_daily_dir is None:
        return {
            **base_status,
            "status": "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_BLOCKED_INVALID_PLAN",
            "skip_reason": "planned_command_or_output_dir_missing",
        }

    rerun_step = step_command(
        name="timing_aligned_prediction_rerun",
        command=command,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )
    steps.append(rerun_step)
    execution_status = {
        **base_status,
        "execution_performed": True,
        "status": (
            "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_EXECUTED"
            if rerun_step.get("returncode") == 0
            else "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_FAILED"
        ),
        "returncode": rerun_step.get("returncode"),
        "timed_out": bool(rerun_step.get("timed_out")),
        "rerun_daily_shadow_run_dir": relpath(rerun_daily_dir),
    }
    if rerun_step.get("returncode") != 0 or not (
        rerun_daily_dir / "shadow_predictions.jsonl"
    ).exists():
        return execution_status

    stage2_status = materialize_root_stage2_predictions(
        rerun_daily_dir,
        output_dir=output_dir,
        generated_at=generated_at,
    )
    execution_status["stage2_materialization_status"] = stage2_status.get("status")
    steps.append(
        {
            "name": "timing_aligned_stage2_shadow_predictions_first_class",
            "command": [],
            "cwd": str(ROOT),
            "started_at": generated_at.isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "returncode": stage2_status.get("returncode"),
            "status": "PASS" if stage2_status.get("returncode") == 0 else "FAIL",
            "stage2_status": stage2_status.get("status"),
            "stage2_status_path": relpath(
                output_dir / "stage2_shadow_predictions_status.json"
            ),
            "root_materialized": stage2_status.get("root_materialized"),
        }
    )
    if skip_odds_snapshot:
        return {
            **execution_status,
            "status": "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_EXECUTED_ODDS_SKIPPED",
            "rerun_odds_snapshot_status": "SKIPPED",
        }

    should_collect_odds, odds_skip_reason, odds_prediction_rows = (
        should_collect_shadow_odds_snapshot(rerun_daily_dir)
    )
    rerun_odds_dir = (
        output_dir.parent
        / f"shadow_odds_snapshot_{now_id(generated_at)}_timing_aligned_rerun"
    )
    if should_collect_odds:
        odds_command = shadow_odds_snapshot_command(
            daily_dir=rerun_daily_dir,
            odds_dir=rerun_odds_dir,
            db_path=db_path,
            current_time=current_time,
        )
        odds_step = step_command(
            name="timing_aligned_shadow_odds_snapshot",
            command=odds_command,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )
        steps.append(odds_step)
        odds_report = load_json(rerun_odds_dir / "shadow_odds_snapshot_report.json") or {}
        odds_status = build_shadow_odds_snapshot_status(
            generated_at=generated_at,
            odds_dir=rerun_odds_dir,
            odds_report=odds_report or None,
            skipped_reason=(
                f"timing_aligned_odds_snapshot_report_missing_returncode_{odds_step.get('returncode')}"
                if not odds_report
                else None
            ),
            prediction_rows=odds_prediction_rows,
            attempted=True,
            status_override=(
                "TIMING_ALIGNED_SHADOW_ODDS_SNAPSHOT_FAILED_NO_REPORT"
                if not odds_report
                else None
            ),
        )
    else:
        odds_status = build_shadow_odds_snapshot_status(
            generated_at=generated_at,
            odds_dir=None,
            odds_report=None,
            skipped_reason=odds_skip_reason,
            prediction_rows=odds_prediction_rows,
        )
    return {
        **execution_status,
        "status": "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_EXECUTED_WITH_ODDS_REFRESH",
        "rerun_odds_snapshot_dir": relpath(rerun_odds_dir) if should_collect_odds else None,
        "rerun_odds_snapshot_status": odds_status.get("status"),
        "rerun_odds_snapshot": odds_status,
    }


def parse_datetime_or_none(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def compare_datetimes(left: datetime, right: datetime) -> tuple[datetime, datetime]:
    if left.tzinfo is None and right.tzinfo is not None:
        left = left.replace(tzinfo=right.tzinfo)
    elif left.tzinfo is not None and right.tzinfo is None:
        right = right.replace(tzinfo=left.tzinfo)
    elif left.tzinfo is not None and right.tzinfo is not None:
        right = right.astimezone(left.tzinfo)
    return left, right


def canonical_race_identity(row: Mapping[str, Any]) -> str | None:
    race_date = str(row.get("race_date") or "")[:10]
    venue = str(row.get("venue") or "").strip().upper()
    race_number = row.get("race_number")
    if not race_date or not venue or race_number in (None, ""):
        return None
    return f"Race {race_number} - {venue} - {race_date}"


def runner_set_validation_for_metadata_row(row: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if str(row.get("bucket") or "") != "eligible":
        reasons.append("not_eligible_prejump_file")
    if int(row.get("runner_count") or 0) <= 0:
        reasons.append("runner_count_missing_or_zero")
    if row.get("sidecar_status") != "PASS":
        reasons.append("csv_sidecar_status_not_pass")
    if row.get("metadata_is_leakage_safe") is not True:
        reasons.append("metadata_not_leakage_safe")
    if row.get("csv_sidecar_runner_identity_verified") is not True:
        reasons.append("csv_sidecar_runner_identity_not_verified")
    if row.get("canonical_runner_alignment_verified") is not True:
        reasons.append("canonical_runner_alignment_not_verified")
    if not row.get("source_url"):
        reasons.append("source_url_missing")
    if not row.get("canonical_runner_source_url"):
        reasons.append("canonical_runner_source_url_missing")
    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "runner_count": int(row.get("runner_count") or 0),
        "csv_sidecar_runner_identity_verified": row.get(
            "csv_sidecar_runner_identity_verified"
        )
        is True,
        "canonical_runner_alignment_verified": row.get(
            "canonical_runner_alignment_verified"
        )
        is True,
        "metadata_is_leakage_safe": row.get("metadata_is_leakage_safe") is True,
    }


def fixed_capture_windows_for_metadata_row(
    row: Mapping[str, Any],
    *,
    generated_at: datetime,
) -> list[dict[str, Any]]:
    jump_at = parse_datetime_or_none(row.get("jump_datetime") or row.get("jump_time_iso"))
    windows: list[dict[str, Any]] = []
    for offset in FIXED_PREJUMP_ODDS_CAPTURE_WINDOWS_MINUTES:
        window: dict[str, Any] = {"offset_minutes": offset}
        if jump_at is None:
            window.update(
                {
                    "target_capture_at": None,
                    "seconds_until_target": None,
                    "status": "DATA_MISSING_JUMP_TIME",
                }
            )
        else:
            target = jump_at - timedelta(minutes=offset)
            target_cmp, generated_cmp = compare_datetimes(target, generated_at)
            jump_cmp, generated_for_jump = compare_datetimes(jump_at, generated_at)
            if generated_for_jump >= jump_cmp:
                status = "MISSED_AFTER_JUMP"
            elif generated_cmp < target_cmp:
                status = "PENDING"
            else:
                status = "DUE_NOW_OR_PASSED_PRE_JUMP"
            window.update(
                {
                    "target_capture_at": target.isoformat(),
                    "seconds_until_target": round((target_cmp - generated_cmp).total_seconds()),
                    "status": status,
                }
            )
        windows.append(window)
    return windows


def build_live_odds_capture_approval_packet(
    *,
    generated_at: datetime,
    daily_shadow_run_dir: Path | None,
    upcoming_dir: Path,
    db_path: Path,
    output_path: Path,
    limit: int,
) -> dict[str, Any]:
    report_path = (
        daily_shadow_run_dir / "prejump_metadata_report.json"
        if daily_shadow_run_dir is not None
        else None
    )
    report = load_json(report_path)
    hard_stops: list[str] = []
    rows = report.get("files") if report else None
    if report is None:
        hard_stops.append("prejump_metadata_report_missing")
        rows = []
    elif not isinstance(rows, list):
        hard_stops.append("prejump_metadata_files_missing")
        rows = []

    readiness = report.get("target_metadata_readiness") if isinstance(report, Mapping) else {}
    if (
        not isinstance(readiness, Mapping)
        or readiness.get("target_metadata_capture_status") != "READY"
        or readiness.get("all_current_future_inputs_verified") is not True
    ):
        hard_stops.append("target_metadata_capture_not_ready")

    race_packets: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        validation = runner_set_validation_for_metadata_row(row)
        if validation.get("status") != "PASS":
            continue
        race_packets.append(
            {
                "race_id": canonical_race_identity(row),
                "canonical_race_identity": canonical_race_identity(row),
                "venue": row.get("venue"),
                "race_number": row.get("race_number"),
                "race_date": row.get("race_date"),
                "jump_datetime": row.get("jump_datetime") or row.get("jump_time_iso"),
                "thedogs_source_url": row.get("source_url"),
                "canonical_runner_source_url": row.get("canonical_runner_source_url"),
                "runner_count": int(row.get("runner_count") or 0),
                "runner_set_validation": validation,
                "capture_windows": fixed_capture_windows_for_metadata_row(
                    row,
                    generated_at=generated_at,
                ),
            }
        )

    if not race_packets:
        hard_stops.append("verified_prejump_race_count_zero")

    capture_output_dir = (
        DEFAULT_EVIDENCE_ROOT
        / f"autonomous_live_odds_capture_{now_id(generated_at)}_approval_packet"
    )
    planned_command = [
        sys.executable,
        str(ROOT / "scripts/autonomous_live_odds_capture.py"),
        "--input-dir",
        str(upcoming_dir),
        "--output-dir",
        str(capture_output_dir),
        "--db",
        str(db_path),
        "--current-time",
        generated_at.isoformat(),
        "--limit",
        str(limit or len(race_packets) or 1),
    ]
    approved_command = list(planned_command)
    approved_command.extend(["--execute", "--allow-auto-scrape-odds"])

    no_write_guarantees = dict(NO_WRITE_GUARANTEES)
    no_write_guarantees.update(
        {
            "odds_used_for_shadow_scoring": False,
            "betting_action": False,
            "ev_action": False,
            "snapshot_write": False,
            "model_training": False,
        }
    )
    return {
        "schema_version": "shadow_live_odds_capture_approval_packet_v1",
        "generated_at": generated_at.isoformat(),
        "status": "NOT_READY"
        if hard_stops
        else "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS",
        "approval_required": True,
        "approval_gate": "operator_explicit_live_odds_capture_approval",
        "required_cli_flag": "--approve-live-odds-capture",
        "required_env_var": "APPROVE_LIVE_ODDS_CAPTURE=true",
        "can_capture_live_odds_now": False,
        "hard_stops": sorted(set(hard_stops)),
        "write_scope": "append_only_live_odds_rows",
        "append_only_required": True,
        "fixed_capture_windows_required": True,
        "capture_window_offsets_minutes": list(FIXED_PREJUMP_ODDS_CAPTURE_WINDOWS_MINUTES),
        "prejump_metadata_report": relpath(report_path),
        "verified_prejump_race_count": len(race_packets),
        "races": race_packets,
        "required_provenance_fields": list(LIVE_ODDS_CAPTURE_REQUIRED_PROVENANCE_FIELDS),
        "provenance_requirements": {
            "canonical_race_identity": "Race number, venue, and date from verified target metadata.",
            "sportsbet_source_url": "Sportsbet source URL captured with every appended row.",
            "sportsbet_source_race_identity": "Sportsbet venue/date/race identity used for source matching.",
            "scrape_timestamp": "Timestamp of the odds scrape before jump.",
            "market_type": "WIN market only for dog-level odds capture.",
            "dog_level_win_odds": "One WIN price per final runner where available.",
            "sportsbet_box_source": "Box source extracted from Sportsbet runner text or metadata.",
            "runner_name_box_match_status": "Name and box validation status before append.",
        },
        "planned_live_odds_capture_command": planned_command,
        "approved_live_odds_capture_command_template": approved_command,
        "no_write_guarantees": no_write_guarantees,
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


def safe_count(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return 0


def subprocess_output_text(value: Any) -> str:
    if value in (None, b""):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def step_command(
    *,
    name: str,
    command: Sequence[str],
    output_dir: Path,
    cwd: Path = ROOT,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    log_dir = output_dir / "logs"
    stdout_path = log_dir / f"{name}.stdout.txt"
    stderr_path = log_dir / f"{name}.stderr.txt"
    timed_out = False
    timeout_note = ""
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_note = (
            f"\n[TIMEOUT] command exceeded step timeout of {timeout_seconds} seconds\n"
        )
        stdout = subprocess_output_text(exc.output)
        stderr = subprocess_output_text(exc.stderr) + timeout_note
        returncode = -9
    duration = time.monotonic() - started_monotonic
    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    return {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "duration_seconds": duration,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "returncode": returncode,
        "status": "PASS" if returncode == 0 else "FAIL",
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


def odds_capture_dependencies_available() -> bool:
    return all(
        importlib.util.find_spec(module) is not None
        for module in ODDS_CAPTURE_REQUIRED_MODULES
    )


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


def odds_capture_command_prefix(mode: str = "auto") -> list[str]:
    if mode not in {"auto", "python", "uv"}:
        raise ValueError(f"unknown_odds_capture_command_mode:{mode}")
    if mode == "python" or (mode == "auto" and odds_capture_dependencies_available()):
        return [sys.executable]
    uv_path = shutil.which("uv")
    if uv_path:
        command = [uv_path, "run"]
        for package in UV_ODDS_CAPTURE_PACKAGES:
            command.extend(["--with", package])
        command.append("python")
        return command
    raise RuntimeError("odds_capture_dependencies_missing_and_uv_unavailable")




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
    live_odds_capture_packet: Mapping[str, Any] | None = None,
    autonomous_live_odds_capture_status: Mapping[str, Any] | None = None,
    autonomous_official_result_capture_status: Mapping[str, Any] | None = None,
    unified_evidence_dataset_status: Mapping[str, Any] | None = None,
    backlog_unified_evidence_status: Mapping[str, Any] | None = None,
    rolling_model_comparison_status: Mapping[str, Any] | None = None,
    high_accuracy_refinement_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calibration = (aggregate_calibration or {}).get("slope_intercept") or {}
    promotion_distance = promotion_distance_status_projection(
        high_accuracy_refinement_status
    )
    current_unified_eligible_rows = safe_count(
        (unified_evidence_dataset_status or {}).get("unified_evidence_eligible_rows")
    )
    backlog_unified_eligible_rows = safe_count(
        (backlog_unified_evidence_status or {}).get("unified_evidence_eligible_rows")
    )
    high_accuracy_unified_eligible_rows = safe_count(
        (high_accuracy_refinement_status or {}).get("unified_evidence_eligible_rows")
    )
    max_observed_unified_eligible_rows = max(
        current_unified_eligible_rows,
        backlog_unified_eligible_rows,
        high_accuracy_unified_eligible_rows,
    )
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
        "live_odds_capture_approval": {
            "status": (live_odds_capture_packet or {}).get("status"),
            "packet_path": (live_odds_capture_packet or {}).get("packet_path"),
            "verified_prejump_race_count": (live_odds_capture_packet or {}).get(
                "verified_prejump_race_count",
                0,
            ),
            "capture_window_offsets_minutes": (live_odds_capture_packet or {}).get(
                "capture_window_offsets_minutes",
                [],
            ),
            "approval_required": (live_odds_capture_packet or {}).get(
                "approval_required"
            ),
            "can_capture_live_odds_now": (live_odds_capture_packet or {}).get(
                "can_capture_live_odds_now",
                False,
            ),
        },
        "autonomous_live_odds_capture": {
            "status": (autonomous_live_odds_capture_status or {}).get("status"),
            "output_dir": (autonomous_live_odds_capture_status or {}).get("output_dir"),
            "attempted": (autonomous_live_odds_capture_status or {}).get(
                "attempted",
                False,
            ),
            "execute": (autonomous_live_odds_capture_status or {}).get("execute", False),
            "ready_count": (autonomous_live_odds_capture_status or {}).get(
                "ready_count",
                0,
            ),
            "validation_pass_count": (autonomous_live_odds_capture_status or {}).get(
                "validation_pass_count",
                0,
            ),
            "inserted_live_odds_rows": (autonomous_live_odds_capture_status or {}).get(
                "inserted_live_odds_rows",
                0,
            ),
            "status_counts": (autonomous_live_odds_capture_status or {}).get(
                "status_counts",
                {},
            ),
            "capture_window_coverage_status_counts": (
                autonomous_live_odds_capture_status or {}
            ).get("capture_window_coverage_status_counts", {}),
            "capture_window_coverage_race_count": (
                autonomous_live_odds_capture_status or {}
            ).get("capture_window_coverage_race_count", 0),
            "capture_window_coverage_window_count": (
                autonomous_live_odds_capture_status or {}
            ).get("capture_window_coverage_window_count", 0),
            "capture_window_coverage_report": (
                autonomous_live_odds_capture_status or {}
            ).get("capture_window_coverage_report"),
        },
        "autonomous_official_result_capture": {
            "status": (autonomous_official_result_capture_status or {}).get("status"),
            "output_dir": (autonomous_official_result_capture_status or {}).get("output_dir"),
            "attempted": (autonomous_official_result_capture_status or {}).get(
                "attempted",
                False,
            ),
            "candidate_count": (autonomous_official_result_capture_status or {}).get(
                "candidate_count",
                0,
            ),
            "official_result_race_rows": (
                autonomous_official_result_capture_status or {}
            ).get("official_result_race_rows", 0),
            "official_result_runner_rows": (
                autonomous_official_result_capture_status or {}
            ).get("official_result_runner_rows", 0),
            "quarantine_rows": (autonomous_official_result_capture_status or {}).get(
                "quarantine_rows",
                0,
            ),
            "quarantined_race_ids": (
                autonomous_official_result_capture_status or {}
            ).get("quarantined_race_ids", []),
            "quarantine_reason_counts": (
                autonomous_official_result_capture_status or {}
            ).get("quarantine_reason_counts", {}),
            "quarantine_error_counts": (
                autonomous_official_result_capture_status or {}
            ).get("quarantine_error_counts", {}),
            "quarantine_result_boxes_not_in_participants_counts": (
                autonomous_official_result_capture_status or {}
            ).get("quarantine_result_boxes_not_in_participants_counts", {}),
            "quarantine_runner_set_mismatch_samples": (
                autonomous_official_result_capture_status or {}
            ).get("quarantine_runner_set_mismatch_samples", []),
            "skipped_reason_counts": (
                autonomous_official_result_capture_status or {}
            ).get("skipped_reason_counts", {}),
            "awaiting_jump_race_count": (
                autonomous_official_result_capture_status or {}
            ).get("awaiting_jump_race_count", 0),
            "awaiting_jump_race_ids": (
                autonomous_official_result_capture_status or {}
            ).get("awaiting_jump_race_ids", []),
            "awaiting_jump_next_recheck_after_local": (
                autonomous_official_result_capture_status or {}
            ).get("awaiting_jump_next_recheck_after_local"),
            "awaiting_jump_races": (
                autonomous_official_result_capture_status or {}
            ).get("awaiting_jump_races", []),
            "live_odds_backlog_recovery_queue_path": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_recovery_queue_path"),
            "live_odds_backlog_recovery_queue_diagnostic_only": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_recovery_queue_diagnostic_only"),
            "live_odds_backlog_recovery_queue_join_acceptance_changed": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_recovery_queue_join_acceptance_changed"),
            "live_odds_backlog_recovery_queue_db_write_performed": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_recovery_queue_db_write_performed"),
            "live_odds_backlog_runner_set_validation_path": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_runner_set_validation_path"),
            "live_odds_backlog_runner_set_validation_retryable_race_count": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_runner_set_validation_retryable_race_count", 0),
            "live_odds_backlog_runner_set_validation_exact_match_race_count": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_runner_set_validation_exact_match_race_count", 0),
            "live_odds_backlog_runner_set_validation_blocked_race_count": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_runner_set_validation_blocked_race_count", 0),
            "live_odds_backlog_runner_set_validation_diagnostic_only": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_runner_set_validation_diagnostic_only"),
            "live_odds_backlog_runner_set_validation_join_authorized": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_runner_set_validation_join_authorized"),
            "live_odds_backlog_runner_set_validation_db_write_performed": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_runner_set_validation_db_write_performed"),
            "live_odds_backlog_join_eligibility_packet_path": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_join_eligibility_packet_path"),
            "live_odds_backlog_join_eligibility_evaluated_race_count": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_join_eligibility_evaluated_race_count", 0),
            "live_odds_backlog_join_eligibility_eligible_report_only_race_count": (
                autonomous_official_result_capture_status or {}
            ).get(
                "live_odds_backlog_join_eligibility_eligible_report_only_race_count",
                0,
            ),
            "live_odds_backlog_join_eligibility_blocked_race_count": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_join_eligibility_blocked_race_count", 0),
            "live_odds_backlog_join_eligibility_blocker_counts": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_join_eligibility_blocker_counts", {}),
            "live_odds_backlog_join_eligibility_diagnostic_only": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_join_eligibility_diagnostic_only"),
            "live_odds_backlog_join_eligibility_join_authorized": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_join_eligibility_join_authorized"),
            "live_odds_backlog_join_eligibility_db_write_performed": (
                autonomous_official_result_capture_status or {}
            ).get("live_odds_backlog_join_eligibility_db_write_performed"),
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": (
                autonomous_official_result_capture_status or {}
            ).get(
                "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count",
                0,
            ),
        },
        "unified_evidence_dataset": {
            "status": (unified_evidence_dataset_status or {}).get("status"),
            "output_dir": (unified_evidence_dataset_status or {}).get("output_dir"),
            "attempted": (unified_evidence_dataset_status or {}).get("attempted", False),
            "row_count": (unified_evidence_dataset_status or {}).get("row_count", 0),
            "race_count": (unified_evidence_dataset_status or {}).get("race_count", 0),
            "rows_with_official_results": (unified_evidence_dataset_status or {}).get(
                "rows_with_official_results",
                0,
            ),
            "rows_with_stage2_predictions": (unified_evidence_dataset_status or {}).get(
                "rows_with_stage2_predictions",
                0,
            ),
            "rows_with_strict_prejump_odds": (unified_evidence_dataset_status or {}).get(
                "rows_with_strict_prejump_odds",
                0,
            ),
            "label_evaluation_eligible_rows": (unified_evidence_dataset_status or {}).get(
                "label_evaluation_eligible_rows",
                0,
            ),
            "stage2_evaluation_eligible_rows": (unified_evidence_dataset_status or {}).get(
                "stage2_evaluation_eligible_rows",
                0,
            ),
            "odds_evaluation_eligible_rows": (unified_evidence_dataset_status or {}).get(
                "odds_evaluation_eligible_rows",
                0,
            ),
            "unified_evidence_eligible_rows": (unified_evidence_dataset_status or {}).get(
                "unified_evidence_eligible_rows",
                0,
            ),
            "artifact_odds_rows_seen": (unified_evidence_dataset_status or {}).get(
                "artifact_odds_rows_seen",
                0,
            ),
            "artifact_odds_rows_accepted": (unified_evidence_dataset_status or {}).get(
                "artifact_odds_rows_accepted",
                0,
            ),
            "artifact_odds_rows_rejected": (unified_evidence_dataset_status or {}).get(
                "artifact_odds_rows_rejected",
                0,
            ),
            "artifact_odds_rejection_reason_counts": (
                unified_evidence_dataset_status or {}
            ).get("artifact_odds_rejection_reason_counts", {}),
            "exclusion_reason_counts": (unified_evidence_dataset_status or {}).get(
                "exclusion_reason_counts",
                {},
            ),
            "odds_exclusion_reason_counts": (
                unified_evidence_dataset_status or {}
            ).get("odds_exclusion_reason_counts", {}),
            "rejected_live_odds_candidate_count": (
                unified_evidence_dataset_status or {}
            ).get("rejected_live_odds_candidate_count", 0),
            "rows_with_rejected_live_odds_candidates": (
                unified_evidence_dataset_status or {}
            ).get("rows_with_rejected_live_odds_candidates", 0),
            "rejected_live_odds_candidate_reason_counts": (
                unified_evidence_dataset_status or {}
            ).get("rejected_live_odds_candidate_reason_counts", {}),
            "official_result_coverage": (unified_evidence_dataset_status or {}).get(
                "official_result_coverage",
                {},
            ),
            "official_result_coverage_requested_race_count": (
                unified_evidence_dataset_status or {}
            ).get("official_result_coverage_requested_race_count", 0),
            "official_result_coverage_requested_race_count_source": (
                unified_evidence_dataset_status or {}
            ).get("official_result_coverage_requested_race_count_source")
            or (
                (unified_evidence_dataset_status or {}).get(
                    "official_result_coverage"
                )
                or {}
            ).get("requested_race_count_source"),
            "official_result_coverage_races_with_rows_count": (
                unified_evidence_dataset_status or {}
            ).get("official_result_coverage_races_with_rows_count", 0),
            "official_result_coverage_missing_race_count": (
                unified_evidence_dataset_status or {}
            ).get("official_result_coverage_missing_race_count", 0),
            "official_result_coverage_missing_exclusion_count": (
                unified_evidence_dataset_status or {}
            ).get("official_result_coverage_missing_exclusion_count", 0),
            "official_result_runner_path_count": (
                unified_evidence_dataset_status or {}
            ).get("official_result_runner_path_count", 0),
            "official_result_runner_paths_source_field": (
                unified_evidence_dataset_status or {}
            ).get("official_result_runner_paths_source_field"),
        },
        "backlog_unified_evidence_datasets": {
            "status": (backlog_unified_evidence_status or {}).get("status"),
            "attempted_dataset_count": (backlog_unified_evidence_status or {}).get(
                "attempted_dataset_count",
                0,
            ),
            "dataset_count": (backlog_unified_evidence_status or {}).get(
                "dataset_count",
                0,
            ),
            "failed_dataset_count": (backlog_unified_evidence_status or {}).get(
                "failed_dataset_count",
                0,
            ),
            "row_count": (backlog_unified_evidence_status or {}).get("row_count", 0),
            "race_count": (backlog_unified_evidence_status or {}).get("race_count", 0),
            "rows_with_official_results": (backlog_unified_evidence_status or {}).get(
                "rows_with_official_results",
                0,
            ),
            "rows_with_strict_prejump_odds": (backlog_unified_evidence_status or {}).get(
                "rows_with_strict_prejump_odds",
                0,
            ),
            "unified_evidence_eligible_rows": (
                backlog_unified_evidence_status or {}
            ).get("unified_evidence_eligible_rows", 0),
            "artifact_odds_rows_seen": (
                backlog_unified_evidence_status or {}
            ).get("artifact_odds_rows_seen", 0),
            "artifact_odds_rows_accepted": (
                backlog_unified_evidence_status or {}
            ).get("artifact_odds_rows_accepted", 0),
            "artifact_odds_rows_rejected": (
                backlog_unified_evidence_status or {}
            ).get("artifact_odds_rows_rejected", 0),
            "artifact_odds_rejection_reason_counts": (
                backlog_unified_evidence_status or {}
            ).get("artifact_odds_rejection_reason_counts", {}),
            "exclusion_reason_counts": (backlog_unified_evidence_status or {}).get(
                "exclusion_reason_counts",
                {},
            ),
            "odds_exclusion_reason_counts": (
                backlog_unified_evidence_status or {}
            ).get("odds_exclusion_reason_counts", {}),
            "rejected_live_odds_candidate_count": (
                backlog_unified_evidence_status or {}
            ).get("rejected_live_odds_candidate_count", 0),
            "rows_with_rejected_live_odds_candidates": (
                backlog_unified_evidence_status or {}
            ).get("rows_with_rejected_live_odds_candidates", 0),
            "rejected_live_odds_candidate_reason_counts": (
                backlog_unified_evidence_status or {}
            ).get("rejected_live_odds_candidate_reason_counts", {}),
            "official_result_coverage": (backlog_unified_evidence_status or {}).get(
                "official_result_coverage",
                {},
            ),
            "official_result_coverage_requested_race_count": (
                backlog_unified_evidence_status or {}
            ).get("official_result_coverage_requested_race_count", 0),
            "official_result_coverage_requested_race_count_source": (
                backlog_unified_evidence_status or {}
            ).get("official_result_coverage_requested_race_count_source"),
            "official_result_coverage_legacy_requested_race_count_without_ids": (
                backlog_unified_evidence_status or {}
            ).get("official_result_coverage_legacy_requested_race_count_without_ids"),
            "official_result_coverage_races_with_rows_count": (
                backlog_unified_evidence_status or {}
            ).get("official_result_coverage_races_with_rows_count", 0),
            "official_result_coverage_missing_race_count": (
                backlog_unified_evidence_status or {}
            ).get("official_result_coverage_missing_race_count", 0),
            "official_result_coverage_missing_exclusion_count": (
                backlog_unified_evidence_status or {}
            ).get("official_result_coverage_missing_exclusion_count", 0),
            "race_coverage": (backlog_unified_evidence_status or {}).get(
                "race_coverage",
                {},
            ),
            "gap_action_plan": (
                (
                    (backlog_unified_evidence_status or {}).get("race_coverage")
                    or {}
                ).get("gap_action_plan")
                or {}
            ),
        },
        "unified_evidence_growth": {
            "current_cycle_unified_evidence_eligible_rows": (
                current_unified_eligible_rows
            ),
            "backlog_unified_evidence_eligible_rows": backlog_unified_eligible_rows,
            "high_accuracy_unified_evidence_eligible_rows": (
                high_accuracy_unified_eligible_rows
            ),
            "max_observed_unified_evidence_eligible_rows": (
                max_observed_unified_eligible_rows
            ),
            "existing_unified_evidence_eligible_rows_scope": (
                "current_cycle_unified_evidence_dataset"
            ),
        },
        "rolling_model_comparison": {
            "status": (rolling_model_comparison_status or {}).get("status"),
            "output_dir": (rolling_model_comparison_status or {}).get("output_dir"),
            "attempted": (rolling_model_comparison_status or {}).get(
                "attempted",
                False,
            ),
            "sample_scope": (rolling_model_comparison_status or {}).get("sample_scope"),
            "dedupe_race_id": (rolling_model_comparison_status or {}).get(
                "dedupe_race_id"
            ),
            "sample_race_count": (rolling_model_comparison_status or {}).get(
                "sample_race_count",
                0,
            ),
            "sample_runner_rows": (rolling_model_comparison_status or {}).get(
                "sample_runner_rows",
                0,
            ),
            "minimum_races_for_review": (rolling_model_comparison_status or {}).get(
                "minimum_races_for_review",
                0,
            ),
            "best_candidate_key": (rolling_model_comparison_status or {}).get(
                "best_candidate_key"
            ),
            "best_candidate_top1": (rolling_model_comparison_status or {}).get(
                "best_candidate_top1"
            ),
            "best_candidate_top3": (rolling_model_comparison_status or {}).get(
                "best_candidate_top3"
            ),
            "source_rejected_live_odds_candidate_count": (
                rolling_model_comparison_status or {}
            ).get("source_rejected_live_odds_candidate_count", 0),
            "source_rows_with_rejected_live_odds_candidates": (
                rolling_model_comparison_status or {}
            ).get("source_rows_with_rejected_live_odds_candidates", 0),
            "source_rejected_live_odds_candidate_reason_counts": (
                rolling_model_comparison_status or {}
            ).get("source_rejected_live_odds_candidate_reason_counts", {}),
            "blockers": (rolling_model_comparison_status or {}).get("blockers") or [],
        },
        "promotion_distance": promotion_distance,
        "high_accuracy_refinement_status": (
            high_accuracy_refinement_status or {}
        ).get("status"),
        "high_accuracy_promotion_pr_gate_status": (
            high_accuracy_refinement_status or {}
        ).get("promotion_pr_gate_status"),
        "high_accuracy_timing_aligned_rerun_plan": (
            high_accuracy_refinement_status or {}
        ).get("timing_aligned_rerun_plan"),
        "high_accuracy_timing_aligned_rerun_execution_status": (
            high_accuracy_refinement_status or {}
        ).get("timing_aligned_rerun_execution_status"),
        "reserve_substitution_preflight": {
            "status": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_preflight_status"
            ),
            "candidate_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_preflight_candidate_count",
                0,
            ),
            "ready_for_policy_review_count": (
                high_accuracy_refinement_status or {}
            ).get("reserve_substitution_preflight_ready_for_policy_review_count", 0),
            "blocked_candidate_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_preflight_blocked_candidate_count",
                0,
            ),
            "readiness_blocker_counts": (
                high_accuracy_refinement_status or {}
            ).get("reserve_substitution_preflight_readiness_blocker_counts", {}),
            "dataset_join_blocker_counts": (
                high_accuracy_refinement_status or {}
            ).get("reserve_substitution_preflight_dataset_join_blocker_counts", {}),
            "ready_race_ids": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_preflight_ready_race_ids",
                [],
            ),
            "blocked_race_ids": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_preflight_blocked_race_ids",
                [],
            ),
            "report": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_preflight_report"
            ),
        },
        "reserve_substitution_manual_review": {
            "status": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_status"
            ),
            "candidate_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_candidate_count",
                0,
            ),
            "ready_candidate_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_ready_candidate_count",
                0,
            ),
            "blocked_candidate_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_blocked_candidate_count",
                0,
            ),
            "mapping_pair_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_mapping_pair_count",
                0,
            ),
            "dataset_join_allowed": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_dataset_join_allowed"
            ),
            "official_result_acceptance_allowed": (
                high_accuracy_refinement_status or {}
            ).get(
                "reserve_substitution_manual_review_official_result_acceptance_allowed"
            ),
            "db_write": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_db_write"
            ),
            "blockers": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_blockers",
                [],
            ),
            "ready_race_ids": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_ready_race_ids",
                [],
            ),
            "report": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_manual_review_report"
            ),
        },
        "reserve_substitution_policy_impact": {
            "status": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_policy_impact_status"
            ),
            "candidate_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_policy_impact_candidate_count",
                0,
            ),
            "ready_candidate_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_policy_impact_ready_candidate_count",
                0,
            ),
            "mapping_pair_count": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_policy_impact_mapping_pair_count",
                0,
            ),
            "potential_runner_rows_blocked": (
                high_accuracy_refinement_status or {}
            ).get("reserve_substitution_policy_impact_potential_runner_rows_blocked"),
            "matched_backlog_top_gap_race_count": (
                high_accuracy_refinement_status or {}
            ).get(
                "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count",
                0,
            ),
            "matched_backlog_top_gap_race_ids": (
                high_accuracy_refinement_status or {}
            ).get(
                "reserve_substitution_policy_impact_matched_backlog_top_gap_race_ids",
                [],
            ),
            "dataset_join_allowed": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_policy_impact_dataset_join_allowed"
            ),
            "official_result_acceptance_allowed": (
                high_accuracy_refinement_status or {}
            ).get(
                "reserve_substitution_policy_impact_official_result_acceptance_allowed"
            ),
            "db_write": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_policy_impact_db_write"
            ),
            "blockers": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_policy_impact_blockers",
                [],
            ),
            "report": (high_accuracy_refinement_status or {}).get(
                "reserve_substitution_policy_impact_report"
            ),
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


def summarize_same_distance_history_provenance(
    same_distance_history_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = same_distance_history_provenance or {}
    by_feature = report.get("by_feature") if isinstance(report.get("by_feature"), Mapping) else {}
    return {
        "status": report.get("status") or "NOT_VERIFIED",
        "feature_rows": report.get("feature_rows"),
        "required_source": report.get("required_source") or "prior_dog_history",
        "required_history_cutoff": report.get("required_history_cutoff")
        or "strictly_before_target_race",
        "target_race_rows_allowed": report.get("target_race_rows_allowed", 0),
        "post_outcome_rows_allowed": report.get("post_outcome_rows_allowed", 0),
        "fail_reasons": list(report.get("fail_reasons") or []),
        "by_feature": {
            feature: {
                "status": (by_feature.get(feature) or {}).get("status"),
                "present_rows": (by_feature.get(feature) or {}).get("present_rows"),
                "prior_history_rows_used": (by_feature.get(feature) or {}).get(
                    "prior_history_rows_used"
                ),
                "source": (by_feature.get(feature) or {}).get("source"),
                "history_cutoff": (by_feature.get(feature) or {}).get("history_cutoff"),
                "target_race_rows_used": (by_feature.get(feature) or {}).get(
                    "target_race_rows_used"
                ),
                "post_outcome_rows_used": (by_feature.get(feature) or {}).get(
                    "post_outcome_rows_used"
                ),
                "fail_reasons": list((by_feature.get(feature) or {}).get("fail_reasons") or []),
            }
            for feature in WATCHED_QUARANTINED_FEATURES
        },
    }


def build_feature_activation_data_availability_status(
    *,
    activation_report: Mapping[str, Any] | None,
    same_distance_history_provenance: Mapping[str, Any] | None,
    inputs: Mapping[str, Path | None],
) -> dict[str, Any]:
    report = activation_report or {}
    thresholds = report.get("thresholds") if isinstance(report.get("thresholds"), Mapping) else {}
    features = {
        str(row.get("feature")): row
        for row in report.get("features") or []
        if isinstance(row, Mapping) and row.get("feature")
    }
    same_distance_summary = summarize_same_distance_history_provenance(
        same_distance_history_provenance
    )
    candidate_metrics_path = inputs.get("candidate_metrics")
    any_allowed = bool(report.get("activation_allowed_features"))
    any_quarantined = bool(report.get("kept_quarantined_features"))
    if not report:
        status = "FEATURE_ACTIVATION_GATE_NOT_RUN"
    elif any_allowed and not any_quarantined:
        status = "FEATURE_ACTIVATION_DATA_READY_REPORT_ONLY"
    else:
        status = "FEATURE_ACTIVATION_DATA_STILL_MISSING_KEEP_QUARANTINED"

    return {
        "schema_version": "shadow_autopilot_feature_activation_data_availability_v1",
        "status": status,
        "candidate_metric_comparison_status": (
            "AVAILABLE" if candidate_metrics_path else "MISSING_OR_STALE"
        ),
        "candidate_metrics_path": relpath(candidate_metrics_path),
        "fail_reason_summary": report.get("fail_reason_summary") or {},
        "same_distance_history": same_distance_summary,
        "next_data_requirement": {
            "mode": "report_only_training_eval_packet",
            "min_train_present_rows": thresholds.get("min_train_present_rows"),
            "min_train_present_pct": thresholds.get("min_train_present_pct"),
            "min_train_unique_present_values": thresholds.get(
                "min_train_unique_present_values"
            ),
            "min_holdout_present_rows": thresholds.get("min_holdout_present_rows"),
            "min_holdout_present_pct": thresholds.get("min_holdout_present_pct"),
            "min_holdout_unique_present_values": thresholds.get(
                "min_holdout_unique_present_values"
            ),
            "required_candidate_metric_comparison": True,
            "history_cutoff": "strictly_before_target_race",
            "source": "prior_dog_history",
        },
        "by_feature": {
            feature: {
                "decision": (features.get(feature) or {}).get("decision"),
                "fail_reasons": list((features.get(feature) or {}).get("fail_reasons") or []),
                "train_present_rows": (
                    ((features.get(feature) or {}).get("parity") or {}).get(
                        "train_present_rows"
                    )
                ),
                "train_rows": (
                    ((features.get(feature) or {}).get("parity") or {}).get("train_rows")
                ),
                "train_present_pct": (
                    ((features.get(feature) or {}).get("parity") or {}).get(
                        "train_present_pct"
                    )
                ),
                "train_unique_present_values": (
                    ((features.get(feature) or {}).get("parity") or {}).get(
                        "train_unique_present_values"
                    )
                ),
                "holdout_present_rows": (
                    ((features.get(feature) or {}).get("parity") or {}).get(
                        "holdout_present_rows"
                    )
                ),
                "holdout_rows": (
                    ((features.get(feature) or {}).get("parity") or {}).get("holdout_rows")
                ),
                "holdout_present_pct": (
                    ((features.get(feature) or {}).get("parity") or {}).get(
                        "holdout_present_pct"
                    )
                ),
                "holdout_unique_present_values": (
                    ((features.get(feature) or {}).get("parity") or {}).get(
                        "holdout_unique_present_values"
                    )
                ),
                "live_same_distance_history": (
                    same_distance_summary.get("by_feature") or {}
                ).get(feature)
                or {},
            }
            for feature in WATCHED_QUARANTINED_FEATURES
        },
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
    live_odds_capture_packet: Mapping[str, Any] | None = None,
    autonomous_live_odds_capture_status: Mapping[str, Any] | None = None,
    autonomous_official_result_capture_status: Mapping[str, Any] | None = None,
    unified_evidence_dataset_status: Mapping[str, Any] | None = None,
    backlog_unified_evidence_status: Mapping[str, Any] | None = None,
    rolling_model_comparison_status: Mapping[str, Any] | None = None,
    high_accuracy_refinement_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = dict(timeseries[-1]) if timeseries else {}
    previous = dict(timeseries[-2]) if len(timeseries) >= 2 else None
    backlog_race_coverage = (
        (backlog_unified_evidence_status or {}).get("race_coverage") or {}
    )
    backlog_gap_action_plan = backlog_race_coverage.get("gap_action_plan") or {}
    promotion_distance = promotion_distance_status_projection(
        high_accuracy_refinement_status
    )
    current_unified_eligible_rows = safe_count(
        (unified_evidence_dataset_status or {}).get("unified_evidence_eligible_rows")
    )
    backlog_unified_eligible_rows = safe_count(
        (backlog_unified_evidence_status or {}).get("unified_evidence_eligible_rows")
    )
    high_accuracy_unified_eligible_rows = safe_count(
        (high_accuracy_refinement_status or {}).get("unified_evidence_eligible_rows")
    )
    max_observed_unified_eligible_rows = max(
        current_unified_eligible_rows,
        backlog_unified_eligible_rows,
        high_accuracy_unified_eligible_rows,
    )
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
        "live_odds_capture_approval_status": (live_odds_capture_packet or {}).get(
            "status"
        ),
        "live_odds_capture_verified_prejump_races": (
            live_odds_capture_packet or {}
        ).get("verified_prejump_race_count", 0),
        "live_odds_capture_window_offsets_minutes": (
            live_odds_capture_packet or {}
        ).get("capture_window_offsets_minutes", []),
        "live_odds_capture_can_capture_now": (live_odds_capture_packet or {}).get(
            "can_capture_live_odds_now",
            False,
        ),
        "autonomous_live_odds_capture_status": (
            autonomous_live_odds_capture_status or {}
        ).get("status"),
        "autonomous_live_odds_capture_attempted": (
            autonomous_live_odds_capture_status or {}
        ).get("attempted", False),
        "autonomous_live_odds_capture_execute": (
            autonomous_live_odds_capture_status or {}
        ).get("execute", False),
        "autonomous_live_odds_capture_ready_count": (
            autonomous_live_odds_capture_status or {}
        ).get("ready_count", 0),
        "autonomous_live_odds_inserted_rows": (
            autonomous_live_odds_capture_status or {}
        ).get("inserted_live_odds_rows", 0),
        "autonomous_live_odds_capture_window_coverage_status_counts": (
            autonomous_live_odds_capture_status or {}
        ).get("capture_window_coverage_status_counts", {}),
        "autonomous_live_odds_capture_window_coverage_race_count": (
            autonomous_live_odds_capture_status or {}
        ).get("capture_window_coverage_race_count", 0),
        "autonomous_live_odds_capture_window_coverage_window_count": (
            autonomous_live_odds_capture_status or {}
        ).get("capture_window_coverage_window_count", 0),
        "autonomous_official_result_capture_status": (
            autonomous_official_result_capture_status or {}
        ).get("status"),
        "autonomous_official_result_capture_attempted": (
            autonomous_official_result_capture_status or {}
        ).get("attempted", False),
        "autonomous_official_result_candidate_count": (
            autonomous_official_result_capture_status or {}
        ).get("candidate_count", 0),
        "autonomous_official_result_race_rows": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_race_rows", 0),
        "autonomous_official_result_runner_rows": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_runner_rows", 0),
        "autonomous_official_result_quarantine_rows": (
            autonomous_official_result_capture_status or {}
        ).get("quarantine_rows", 0),
        "autonomous_official_result_quarantined_race_ids": (
            autonomous_official_result_capture_status or {}
        ).get("quarantined_race_ids", []),
        "autonomous_official_result_quarantine_reason_counts": (
            autonomous_official_result_capture_status or {}
        ).get("quarantine_reason_counts", {}),
        "autonomous_official_result_quarantine_error_counts": (
            autonomous_official_result_capture_status or {}
        ).get("quarantine_error_counts", {}),
        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": (
            autonomous_official_result_capture_status or {}
        ).get("quarantine_result_boxes_not_in_participants_counts", {}),
        "autonomous_official_result_quarantine_runner_set_mismatch_samples": (
            autonomous_official_result_capture_status or {}
        ).get("quarantine_runner_set_mismatch_samples", []),
        "autonomous_official_result_skipped_reason_counts": (
            autonomous_official_result_capture_status or {}
        ).get("skipped_reason_counts", {}),
        "autonomous_official_result_awaiting_jump_race_count": (
            autonomous_official_result_capture_status or {}
        ).get("awaiting_jump_race_count", 0),
        "autonomous_official_result_awaiting_jump_race_ids": (
            autonomous_official_result_capture_status or {}
        ).get("awaiting_jump_race_ids", []),
        "autonomous_official_result_awaiting_jump_next_recheck_after_local": (
            autonomous_official_result_capture_status or {}
        ).get("awaiting_jump_next_recheck_after_local"),
        "autonomous_official_result_awaiting_jump_races": (
            autonomous_official_result_capture_status or {}
        ).get("awaiting_jump_races", []),
        "autonomous_official_result_evidence_db_ingest_status": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_db_ingest_status"),
        "autonomous_official_result_evidence_db_execute": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_db_execute", False),
        "autonomous_official_result_evidence_db_write_performed": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_db_write_performed", False),
        "autonomous_official_result_evidence_valid_race_rows": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_valid_race_rows", 0),
        "autonomous_official_result_evidence_valid_runner_rows": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_valid_runner_rows", 0),
        "autonomous_official_result_evidence_blocked_race_rows": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_blocked_race_rows", 0),
        "autonomous_official_result_evidence_blocked_runner_rows": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_blocked_runner_rows", 0),
        "autonomous_official_result_evidence_inserted_race_rows": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_inserted_race_rows", 0),
        "autonomous_official_result_evidence_inserted_runner_rows": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_inserted_runner_rows", 0),
        "autonomous_official_result_evidence_blocker_reason_counts": (
            autonomous_official_result_capture_status or {}
        ).get("official_result_evidence_blocker_reason_counts", {}),
        "unified_evidence_dataset_status": (
            unified_evidence_dataset_status or {}
        ).get("status"),
        "unified_evidence_dataset_rows": (
            unified_evidence_dataset_status or {}
        ).get("row_count", 0),
        "unified_evidence_dataset_races": (
            unified_evidence_dataset_status or {}
        ).get("race_count", 0),
        "unified_label_evaluation_eligible_rows": (
            unified_evidence_dataset_status or {}
        ).get("label_evaluation_eligible_rows", 0),
        "unified_stage2_evaluation_eligible_rows": (
            unified_evidence_dataset_status or {}
        ).get("stage2_evaluation_eligible_rows", 0),
        "unified_odds_evaluation_eligible_rows": (
            unified_evidence_dataset_status or {}
        ).get("odds_evaluation_eligible_rows", 0),
        "unified_evidence_eligible_rows": (
            unified_evidence_dataset_status or {}
        ).get("unified_evidence_eligible_rows", 0),
        "current_cycle_unified_evidence_eligible_rows": current_unified_eligible_rows,
        "max_observed_unified_evidence_eligible_rows": (
            max_observed_unified_eligible_rows
        ),
        "unified_evidence_eligible_rows_scope": (
            "current_cycle_unified_evidence_dataset"
        ),
        "unified_evidence_artifact_odds_rows_seen": (
            unified_evidence_dataset_status or {}
        ).get("artifact_odds_rows_seen", 0),
        "unified_evidence_artifact_odds_rows_accepted": (
            unified_evidence_dataset_status or {}
        ).get("artifact_odds_rows_accepted", 0),
        "unified_evidence_artifact_odds_rows_rejected": (
            unified_evidence_dataset_status or {}
        ).get("artifact_odds_rows_rejected", 0),
        "unified_evidence_artifact_odds_rejection_reason_counts": (
            unified_evidence_dataset_status or {}
        ).get("artifact_odds_rejection_reason_counts", {}),
        "unified_rejected_live_odds_candidate_count": (
            unified_evidence_dataset_status or {}
        ).get("rejected_live_odds_candidate_count", 0),
        "unified_rows_with_rejected_live_odds_candidates": (
            unified_evidence_dataset_status or {}
        ).get("rows_with_rejected_live_odds_candidates", 0),
        "unified_rejected_live_odds_candidate_reason_counts": (
            unified_evidence_dataset_status or {}
        ).get("rejected_live_odds_candidate_reason_counts", {}),
        "unified_evidence_official_result_coverage": (
            unified_evidence_dataset_status or {}
        ).get("official_result_coverage", {}),
        "unified_evidence_official_result_coverage_requested_race_count": (
            unified_evidence_dataset_status or {}
        ).get("official_result_coverage_requested_race_count", 0),
        "unified_evidence_official_result_coverage_requested_race_count_source": (
            unified_evidence_dataset_status or {}
        ).get("official_result_coverage_requested_race_count_source")
        or (
            (unified_evidence_dataset_status or {}).get("official_result_coverage")
            or {}
        ).get("requested_race_count_source"),
        "unified_evidence_official_result_coverage_races_with_rows_count": (
            unified_evidence_dataset_status or {}
        ).get("official_result_coverage_races_with_rows_count", 0),
        "unified_evidence_official_result_coverage_missing_race_count": (
            unified_evidence_dataset_status or {}
        ).get("official_result_coverage_missing_race_count", 0),
        "unified_evidence_official_result_coverage_missing_exclusion_count": (
            unified_evidence_dataset_status or {}
        ).get("official_result_coverage_missing_exclusion_count", 0),
        "unified_evidence_official_result_runner_path_count": (
            unified_evidence_dataset_status or {}
        ).get("official_result_runner_path_count", 0),
        "unified_evidence_official_result_runner_paths_source_field": (
            unified_evidence_dataset_status or {}
        ).get("official_result_runner_paths_source_field"),
        "backlog_unified_evidence_status": (
            backlog_unified_evidence_status or {}
        ).get("status"),
        "backlog_unified_evidence_dataset_count": (
            backlog_unified_evidence_status or {}
        ).get("dataset_count", 0),
        "backlog_unified_evidence_failed_dataset_count": (
            backlog_unified_evidence_status or {}
        ).get("failed_dataset_count", 0),
        "backlog_unified_evidence_rows": (
            backlog_unified_evidence_status or {}
        ).get("row_count", 0),
        "backlog_unified_evidence_official_result_rows": (
            backlog_unified_evidence_status or {}
        ).get("rows_with_official_results", 0),
        "backlog_unified_evidence_strict_odds_rows": (
            backlog_unified_evidence_status or {}
        ).get("rows_with_strict_prejump_odds", 0),
        "backlog_unified_evidence_eligible_rows": (
            backlog_unified_evidence_status or {}
        ).get("unified_evidence_eligible_rows", 0),
        "backlog_unified_evidence_eligible_rows_scope": (
            "backlog_unified_evidence_datasets"
        ),
        "backlog_unified_evidence_artifact_odds_rows_seen": (
            backlog_unified_evidence_status or {}
        ).get("artifact_odds_rows_seen", 0),
        "backlog_unified_evidence_artifact_odds_rows_accepted": (
            backlog_unified_evidence_status or {}
        ).get("artifact_odds_rows_accepted", 0),
        "backlog_unified_evidence_artifact_odds_rows_rejected": (
            backlog_unified_evidence_status or {}
        ).get("artifact_odds_rows_rejected", 0),
        "backlog_unified_evidence_artifact_odds_rejection_reason_counts": (
            backlog_unified_evidence_status or {}
        ).get("artifact_odds_rejection_reason_counts", {}),
        "backlog_unified_evidence_exclusion_reason_counts": (
            backlog_unified_evidence_status or {}
        ).get("exclusion_reason_counts", {}),
        "backlog_unified_evidence_odds_exclusion_reason_counts": (
            backlog_unified_evidence_status or {}
        ).get("odds_exclusion_reason_counts", {}),
        "backlog_unified_rejected_live_odds_candidate_count": (
            backlog_unified_evidence_status or {}
        ).get("rejected_live_odds_candidate_count", 0),
        "backlog_unified_rows_with_rejected_live_odds_candidates": (
            backlog_unified_evidence_status or {}
        ).get("rows_with_rejected_live_odds_candidates", 0),
        "backlog_unified_rejected_live_odds_candidate_reason_counts": (
            backlog_unified_evidence_status or {}
        ).get("rejected_live_odds_candidate_reason_counts", {}),
        "backlog_unified_official_result_coverage": (
            backlog_unified_evidence_status or {}
        ).get("official_result_coverage", {}),
        "backlog_unified_official_result_coverage_requested_race_count": (
            backlog_unified_evidence_status or {}
        ).get("official_result_coverage_requested_race_count", 0),
        "backlog_unified_official_result_coverage_requested_race_count_source": (
            backlog_unified_evidence_status or {}
        ).get("official_result_coverage_requested_race_count_source"),
        "backlog_unified_official_result_coverage_legacy_requested_race_count_without_ids": (
            backlog_unified_evidence_status or {}
        ).get("official_result_coverage_legacy_requested_race_count_without_ids"),
        "backlog_unified_official_result_coverage_races_with_rows_count": (
            backlog_unified_evidence_status or {}
        ).get("official_result_coverage_races_with_rows_count", 0),
        "backlog_unified_official_result_coverage_missing_race_count": (
            backlog_unified_evidence_status or {}
        ).get("official_result_coverage_missing_race_count", 0),
        "backlog_unified_official_result_coverage_missing_exclusion_count": (
            backlog_unified_evidence_status or {}
        ).get("official_result_coverage_missing_exclusion_count", 0),
        "backlog_unified_race_instance_count": (
            (backlog_unified_evidence_status or {}).get("race_coverage") or {}
        ).get("dataset_race_instance_count", 0),
        "backlog_unified_deduped_race_count": (
            (backlog_unified_evidence_status or {}).get("race_coverage") or {}
        ).get("deduped_race_count", 0),
        "backlog_unified_deduped_races_with_evidence": (
            (backlog_unified_evidence_status or {}).get("race_coverage") or {}
        ).get("deduped_races_with_unified_evidence", 0),
        "backlog_unified_deduped_races_without_evidence": (
            (backlog_unified_evidence_status or {}).get("race_coverage") or {}
        ).get("deduped_races_without_unified_evidence", 0),
        "backlog_unified_deduped_races_without_complete_official_result": (
            backlog_race_coverage
        ).get("deduped_races_without_complete_official_result_instance", 0),
        "backlog_unified_deduped_races_without_complete_strict_odds": (
            backlog_race_coverage
        ).get("deduped_races_without_complete_strict_prejump_odds_instance", 0),
        "backlog_unified_gap_action_counts": (
            backlog_gap_action_plan.get("action_counts", {})
        ),
        "backlog_unified_gap_evidence_missing_reason_counts": (
            backlog_gap_action_plan.get("evidence_missing_reason_counts", {})
        ),
        "backlog_unified_sample_blocking_gap_count": (
            backlog_gap_action_plan.get("sample_blocking_gap_count", 0)
        ),
        "backlog_unified_top_gap_race_ids": [
            str(row.get("race_id"))
            for row in backlog_gap_action_plan.get("top_gap_races") or []
            if isinstance(row, Mapping) and row.get("race_id")
        ],
        "backlog_unified_top_gap_races": compact_unified_gap_rows(
            backlog_gap_action_plan.get("top_gap_races") or []
        ),
        "backlog_unified_top_official_result_missing_race_ids": [
            str(row.get("race_id"))
            for row in backlog_race_coverage.get("top_official_result_missing_races")
            or []
            if isinstance(row, Mapping) and row.get("race_id")
        ],
        "backlog_unified_top_official_result_missing_races": compact_unified_gap_rows(
            backlog_race_coverage.get("top_official_result_missing_races") or []
        ),
        "rolling_model_comparison_status": (
            rolling_model_comparison_status or {}
        ).get("status"),
        "rolling_model_comparison_sample_races": (
            rolling_model_comparison_status or {}
        ).get("sample_race_count", 0),
        "rolling_model_comparison_sample_runner_rows": (
            rolling_model_comparison_status or {}
        ).get("sample_runner_rows", 0),
        "rolling_model_comparison_minimum_races_for_review": (
            rolling_model_comparison_status or {}
        ).get("minimum_races_for_review", 0),
        "rolling_model_comparison_best_candidate": (
            rolling_model_comparison_status or {}
        ).get("best_candidate_key"),
        "rolling_model_comparison_best_top1": (
            rolling_model_comparison_status or {}
        ).get("best_candidate_top1"),
        "rolling_model_comparison_best_top3": (
            rolling_model_comparison_status or {}
        ).get("best_candidate_top3"),
        "rolling_model_comparison_source_rejected_live_odds_candidate_count": (
            rolling_model_comparison_status or {}
        ).get("source_rejected_live_odds_candidate_count", 0),
        "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": (
            rolling_model_comparison_status or {}
        ).get("source_rows_with_rejected_live_odds_candidates", 0),
        "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": (
            rolling_model_comparison_status or {}
        ).get("source_rejected_live_odds_candidate_reason_counts", {}),
        "rolling_model_comparison_blockers": (
            rolling_model_comparison_status or {}
        ).get("blockers", []),
        "high_accuracy_refinement_status": (
            high_accuracy_refinement_status or {}
        ).get("status"),
        "high_accuracy_promotion_pr_gate_status": (
            high_accuracy_refinement_status or {}
        ).get("promotion_pr_gate_status"),
        "high_accuracy_unified_evidence_eligible_rows": (
            high_accuracy_refinement_status or {}
        ).get("unified_evidence_eligible_rows", 0),
        "high_accuracy_unified_evidence_eligible_rows_scope": (
            "high_accuracy_refinement_packet"
        ),
        "reserve_substitution_preflight_status": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_status"),
        "reserve_substitution_preflight_candidate_count": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_candidate_count", 0),
        "reserve_substitution_preflight_ready_for_policy_review_count": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_ready_for_policy_review_count", 0),
        "reserve_substitution_preflight_blocked_candidate_count": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_blocked_candidate_count", 0),
        "reserve_substitution_preflight_readiness_blocker_counts": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_readiness_blocker_counts", {}),
        "reserve_substitution_preflight_dataset_join_blocker_counts": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_dataset_join_blocker_counts", {}),
        "reserve_substitution_preflight_ready_race_ids": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_ready_race_ids", []),
        "reserve_substitution_preflight_blocked_race_ids": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_blocked_race_ids", []),
        "reserve_substitution_preflight_report": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_preflight_report"),
        "reserve_substitution_manual_review_status": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_manual_review_status"),
        "reserve_substitution_manual_review_ready_candidate_count": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_manual_review_ready_candidate_count", 0),
        "reserve_substitution_manual_review_mapping_pair_count": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_manual_review_mapping_pair_count", 0),
        "reserve_substitution_manual_review_dataset_join_allowed": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_manual_review_dataset_join_allowed"),
        "reserve_substitution_manual_review_official_result_acceptance_allowed": (
            high_accuracy_refinement_status or {}
        ).get(
            "reserve_substitution_manual_review_official_result_acceptance_allowed"
        ),
        "reserve_substitution_manual_review_db_write": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_manual_review_db_write"),
        "reserve_substitution_manual_review_blockers": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_manual_review_blockers", []),
        "reserve_substitution_manual_review_ready_race_ids": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_manual_review_ready_race_ids", []),
        "reserve_substitution_manual_review_report": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_manual_review_report"),
        "reserve_substitution_policy_impact_status": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_status"),
        "reserve_substitution_policy_impact_candidate_count": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_candidate_count", 0),
        "reserve_substitution_policy_impact_ready_candidate_count": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_ready_candidate_count", 0),
        "reserve_substitution_policy_impact_mapping_pair_count": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_mapping_pair_count", 0),
        "reserve_substitution_policy_impact_potential_runner_rows_blocked": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_potential_runner_rows_blocked", 0),
        "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count": (
            high_accuracy_refinement_status or {}
        ).get(
            "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count",
            0,
        ),
        "reserve_substitution_policy_impact_matched_backlog_top_gap_race_ids": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_matched_backlog_top_gap_race_ids", []),
        "reserve_substitution_policy_impact_dataset_join_allowed": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_dataset_join_allowed"),
        "reserve_substitution_policy_impact_official_result_acceptance_allowed": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_official_result_acceptance_allowed"),
        "reserve_substitution_policy_impact_db_write": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_db_write"),
        "reserve_substitution_policy_impact_blockers": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_blockers", []),
        "reserve_substitution_policy_impact_report": (
            high_accuracy_refinement_status or {}
        ).get("reserve_substitution_policy_impact_report"),
        "promotion_distance_status": promotion_distance.get("status"),
        "promotion_distance_promotion_ready": promotion_distance.get(
            "promotion_ready"
        ),
        "promotion_distance_blockers": promotion_distance.get("blockers") or [],
        "promotion_distance_sample_race_count": promotion_distance.get(
            "sample_race_count"
        )
        or 0,
        "promotion_distance_sample_runner_rows": promotion_distance.get(
            "sample_runner_rows"
        )
        or 0,
        "promotion_distance_source_rejected_live_odds_candidate_count": (
            promotion_distance.get("source_rejected_live_odds_candidate_count") or 0
        ),
        "promotion_distance_source_rows_with_rejected_live_odds_candidates": (
            promotion_distance.get("source_rows_with_rejected_live_odds_candidates")
            or 0
        ),
        "promotion_distance_source_rejected_live_odds_candidate_reason_counts": (
            promotion_distance.get("source_rejected_live_odds_candidate_reason_counts")
            or {}
        ),
        "promotion_distance_source_exclusion_reason_counts": (
            promotion_distance.get("source_exclusion_reason_counts") or {}
        ),
        "promotion_distance_source_odds_exclusion_reason_counts": (
            promotion_distance.get("source_odds_exclusion_reason_counts") or {}
        ),
        "promotion_distance_source_official_result_evidence_db_missing_race_ids": (
            promotion_distance.get(
                "source_official_result_evidence_db_missing_race_ids"
            )
            or []
        ),
        "promotion_distance_source_official_result_evidence_db_requested_race_count": (
            promotion_distance.get(
                "source_official_result_evidence_db_requested_race_count"
            )
            or 0
        ),
        "promotion_distance_source_official_result_evidence_db_races_with_rows": (
            promotion_distance.get(
                "source_official_result_evidence_db_races_with_rows"
            )
            or []
        ),
        "promotion_distance_source_official_result_runner_paths": (
            promotion_distance.get("source_official_result_runner_paths") or []
        ),
        "promotion_distance_official_result_coverage": (
            promotion_distance.get("official_result_coverage") or {}
        ),
        "promotion_distance_official_result_coverage_requested_race_count": (
            promotion_distance.get("official_result_coverage_requested_race_count")
        ),
        "promotion_distance_official_result_coverage_requested_race_count_source": (
            promotion_distance.get(
                "official_result_coverage_requested_race_count_source"
            )
        ),
        "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": (
            promotion_distance.get(
                "official_result_coverage_legacy_requested_race_count_without_ids"
            )
        ),
        "promotion_distance_official_result_coverage_races_with_rows_count": (
            promotion_distance.get("official_result_coverage_races_with_rows_count")
        ),
        "promotion_distance_official_result_coverage_missing_race_count": (
            promotion_distance.get("official_result_coverage_missing_race_count")
        ),
        "promotion_distance_official_result_coverage_missing_exclusion_count": (
            promotion_distance.get("official_result_coverage_missing_exclusion_count")
        ),
        "promotion_distance_official_result_runner_path_count": (
            promotion_distance.get("official_result_runner_path_count")
        ),
        "promotion_distance_official_result_runner_paths_source_field": (
            promotion_distance.get("official_result_runner_paths_source_field")
        ),
        "promotion_distance_best_candidate_key": promotion_distance.get(
            "best_candidate_key"
        ),
        "promotion_distance_best_non_market_candidate_key": promotion_distance.get(
            "best_non_market_candidate_key"
        ),
        "promotion_distance_best_non_market_top1_margin_gap": promotion_distance.get(
            "best_non_market_top1_margin_gap"
        ),
        "promotion_distance_predeclared_residual_candidate_status": promotion_distance.get(
            "predeclared_residual_candidate_status"
        ),
        "promotion_distance_predeclared_residual_triggered_race_count": promotion_distance.get(
            "predeclared_residual_triggered_race_count"
        ),
        "promotion_distance_report": promotion_distance.get("report"),
        "timing_aligned_rerun_plan": (high_accuracy_refinement_status or {}).get(
            "timing_aligned_rerun_plan"
        ),
        "timing_aligned_rerun_execution_status": (
            high_accuracy_refinement_status or {}
        ).get("timing_aligned_rerun_execution_status"),
        "odds_analysis_status": (odds_snapshot_status or {}).get(
            "odds_analysis_status"
        ),
        "odds_analysis_blocker_counts": (odds_snapshot_status or {}).get(
            "odds_analysis_blocker_counts"
        )
        or {},
        "odds_research_gate_status": (odds_snapshot_status or {}).get(
            "odds_research_gate_status"
        ),
        "odds_research_gate_complete_valid_prejump_odds_races": (
            odds_snapshot_status or {}
        ).get("odds_research_gate_complete_valid_prejump_odds_races"),
        "odds_research_gate_minimum_complete_valid_prejump_odds_races": (
            odds_snapshot_status or {}
        ).get("odds_research_gate_minimum_complete_valid_prejump_odds_races"),
        "odds_research_gate_source_url_coverage_pct": (odds_snapshot_status or {}).get(
            "odds_research_gate_source_url_coverage_pct"
        ),
        "odds_research_gate_source_url_rows_missing": (odds_snapshot_status or {}).get(
            "odds_research_gate_source_url_rows_missing"
        ),
        "odds_research_gate_blocker_counts": (odds_snapshot_status or {}).get(
            "odds_research_gate_blocker_counts"
        )
        or {},
        "odds_research_next_action": (odds_snapshot_status or {}).get(
            "odds_research_next_action"
        ),
        "timing_aligned_prediction_rerun_required": (
            odds_snapshot_status or {}
        ).get("timing_aligned_prediction_rerun_required", False),
        "timing_aligned_prediction_rerun_race_count": (
            odds_snapshot_status or {}
        ).get("timing_aligned_prediction_rerun_race_count", 0),
        "timing_aligned_prediction_rerun_race_ids": (
            odds_snapshot_status or {}
        ).get("timing_aligned_prediction_rerun_race_ids", []),
        "timing_aligned_prediction_rerun_races": (odds_snapshot_status or {}).get(
            "timing_aligned_prediction_rerun_races",
            [],
        ),
        "timing_aligned_prediction_rerun_reason_counts": (
            odds_snapshot_status or {}
        ).get("timing_aligned_prediction_rerun_reason_counts", {}),
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
    live_odds_capture_packet: Mapping[str, Any] | None = None,
) -> str:
    activation_status = activation_gate_status or {}
    activation_data = activation_status.get("data_availability_status") or {}
    same_distance_history = activation_data.get("same_distance_history") or {}
    fail_summary = activation_data.get("fail_reason_summary") or {}
    odds_status = odds_snapshot_status or dashboard.get("odds_snapshot") or {}
    live_odds_status = (
        live_odds_capture_packet
        or dashboard.get("live_odds_capture_approval")
        or {}
    )
    autonomous_odds_status = dashboard.get("autonomous_live_odds_capture") or {}
    autonomous_result_status = dashboard.get("autonomous_official_result_capture") or {}
    unified_dataset_status = dashboard.get("unified_evidence_dataset") or {}
    backlog_unified_status = dashboard.get("backlog_unified_evidence_datasets") or {}
    backlog_race_coverage = backlog_unified_status.get("race_coverage") or {}
    backlog_gap_action_plan = backlog_unified_status.get("gap_action_plan") or (
        backlog_race_coverage.get("gap_action_plan") or {}
    )
    rolling_comparison_status = dashboard.get("rolling_model_comparison") or {}
    reserve_preflight_status = dashboard.get("reserve_substitution_preflight") or {}
    reserve_manual_review_status = (
        dashboard.get("reserve_substitution_manual_review") or {}
    )
    promotion_distance_status = dashboard.get("promotion_distance") or {}
    next_window = dashboard.get("next_prejump_refresh_window") or {}
    next_race = next_window.get("next_race") or {}
    return "\n".join(
        [
            "# Shadow Autopilot V1",
            "",
            f"Final verdict: `{final_verdict}`",
            "",
            "Scope: forward-shadow evidence accumulation, autonomous odds-capture status, and report observability.",
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
            f"- Odds research gate: `{odds_status.get('odds_research_gate_status')}`",
            f"- Odds research gate complete valid races: `{odds_status.get('odds_research_gate_complete_valid_prejump_odds_races')}`",
            f"- EV output rows: `{odds_status.get('ev_output_rows')}`",
            "",
            "## Live Odds Capture Approval",
            f"- Status: `{live_odds_status.get('status')}`",
            f"- Verified pre-jump races: `{live_odds_status.get('verified_prejump_race_count')}`",
            f"- Capture windows: `{live_odds_status.get('capture_window_offsets_minutes')}`",
            f"- Approval required: `{live_odds_status.get('approval_required')}`",
            f"- Can capture now: `{live_odds_status.get('can_capture_live_odds_now')}`",
            "",
            "## Autonomous Live Odds Capture",
            f"- Status: `{autonomous_odds_status.get('status')}`",
            f"- Attempted: `{autonomous_odds_status.get('attempted')}`",
            f"- Execute: `{autonomous_odds_status.get('execute')}`",
            f"- Ready races: `{autonomous_odds_status.get('ready_count')}`",
            f"- Inserted rows: `{autonomous_odds_status.get('inserted_live_odds_rows')}`",
            "",
            "## Autonomous Official Result Capture",
            f"- Status: `{autonomous_result_status.get('status')}`",
            f"- Attempted: `{autonomous_result_status.get('attempted')}`",
            f"- Official race rows: `{autonomous_result_status.get('official_result_race_rows')}`",
            f"- Official runner rows: `{autonomous_result_status.get('official_result_runner_rows')}`",
            f"- Evidence DB ingest: `{autonomous_result_status.get('official_result_evidence_db_ingest_status')}`",
            f"- Evidence DB execute: `{autonomous_result_status.get('official_result_evidence_db_execute')}`",
            f"- Evidence DB write performed: `{autonomous_result_status.get('official_result_evidence_db_write_performed')}`",
            f"- Evidence DB valid race rows: `{autonomous_result_status.get('official_result_evidence_valid_race_rows')}`",
            f"- Evidence DB valid runner rows: `{autonomous_result_status.get('official_result_evidence_valid_runner_rows')}`",
            f"- Evidence DB blocked race rows: `{autonomous_result_status.get('official_result_evidence_blocked_race_rows')}`",
            f"- Evidence DB blocked runner rows: `{autonomous_result_status.get('official_result_evidence_blocked_runner_rows')}`",
            f"- Evidence DB inserted race rows: `{autonomous_result_status.get('official_result_evidence_inserted_race_rows')}`",
            f"- Evidence DB inserted runner rows: `{autonomous_result_status.get('official_result_evidence_inserted_runner_rows')}`",
            f"- Evidence DB blocker reasons: `{autonomous_result_status.get('official_result_evidence_blocker_reason_counts')}`",
            f"- Quarantine rows: `{autonomous_result_status.get('quarantine_rows')}`",
            f"- Quarantined race IDs: `{autonomous_result_status.get('quarantined_race_ids')}`",
            f"- Quarantine result boxes not in participants: `{autonomous_result_status.get('quarantine_result_boxes_not_in_participants_counts')}`",
            f"- Quarantine runner-set mismatch samples: `{autonomous_result_status.get('quarantine_runner_set_mismatch_samples')}`",
            "",
            "## Unified Evidence Dataset",
            f"- Status: `{unified_dataset_status.get('status')}`",
            f"- Rows: `{unified_dataset_status.get('row_count')}`",
            f"- Races: `{unified_dataset_status.get('race_count')}`",
            f"- Rows with official results: `{unified_dataset_status.get('rows_with_official_results')}`",
            f"- Rows with Stage 2 predictions: `{unified_dataset_status.get('rows_with_stage2_predictions')}`",
            f"- Rows with strict pre-jump odds: `{unified_dataset_status.get('rows_with_strict_prejump_odds')}`",
            f"- Unified eligible rows: `{unified_dataset_status.get('unified_evidence_eligible_rows')}`",
            f"- Artifact odds rows seen: `{unified_dataset_status.get('artifact_odds_rows_seen')}`",
            f"- Artifact odds rows accepted: `{unified_dataset_status.get('artifact_odds_rows_accepted')}`",
            f"- Artifact odds rows rejected: `{unified_dataset_status.get('artifact_odds_rows_rejected')}`",
            f"- Artifact odds rejection reasons: `{unified_dataset_status.get('artifact_odds_rejection_reason_counts')}`",
            f"- Rejected live odds candidates: `{unified_dataset_status.get('rejected_live_odds_candidate_count')}`",
            f"- Rejected live odds candidate reasons: `{unified_dataset_status.get('rejected_live_odds_candidate_reason_counts')}`",
            "",
            "## Backlog Unified Evidence Datasets",
            f"- Status: `{backlog_unified_status.get('status')}`",
            f"- Datasets: `{backlog_unified_status.get('dataset_count')}`",
            f"- Failed datasets: `{backlog_unified_status.get('failed_dataset_count')}`",
            f"- Rows: `{backlog_unified_status.get('row_count')}`",
            f"- Rows with official results: `{backlog_unified_status.get('rows_with_official_results')}`",
            f"- Rows with strict pre-jump odds: `{backlog_unified_status.get('rows_with_strict_prejump_odds')}`",
            f"- Unified eligible rows: `{backlog_unified_status.get('unified_evidence_eligible_rows')}`",
            f"- Artifact odds rows seen: `{backlog_unified_status.get('artifact_odds_rows_seen')}`",
            f"- Artifact odds rows accepted: `{backlog_unified_status.get('artifact_odds_rows_accepted')}`",
            f"- Artifact odds rows rejected: `{backlog_unified_status.get('artifact_odds_rows_rejected')}`",
            f"- Artifact odds rejection reasons: `{backlog_unified_status.get('artifact_odds_rejection_reason_counts')}`",
            f"- Exclusion reason counts: `{backlog_unified_status.get('exclusion_reason_counts')}`",
            f"- Odds exclusion reason counts: `{backlog_unified_status.get('odds_exclusion_reason_counts')}`",
            f"- Rejected live odds candidates: `{backlog_unified_status.get('rejected_live_odds_candidate_count')}`",
            f"- Rejected live odds candidate reasons: `{backlog_unified_status.get('rejected_live_odds_candidate_reason_counts')}`",
            f"- Sample-blocking gap races: `{backlog_gap_action_plan.get('sample_blocking_gap_count')}`",
            f"- Gap actions: `{backlog_gap_action_plan.get('action_counts')}`",
            f"- Evidence-missing reasons: `{backlog_gap_action_plan.get('evidence_missing_reason_counts')}`",
            f"- Top gap race IDs: `{[str(row.get('race_id')) for row in backlog_gap_action_plan.get('top_gap_races') or [] if isinstance(row, Mapping) and row.get('race_id')]}`",
            f"- Top gap races: `{compact_unified_gap_rows(backlog_gap_action_plan.get('top_gap_races') or [])}`",
            f"- Top official-result-missing race IDs: `{[str(row.get('race_id')) for row in backlog_race_coverage.get('top_official_result_missing_races') or [] if isinstance(row, Mapping) and row.get('race_id')]}`",
            f"- Top official-result-missing races: `{compact_unified_gap_rows(backlog_race_coverage.get('top_official_result_missing_races') or [])}`",
            f"- Race coverage: `{backlog_unified_status.get('race_coverage')}`",
            "",
            "## Reserve Substitution Preflight",
            f"- Status: `{reserve_preflight_status.get('status')}`",
            f"- Candidates: `{reserve_preflight_status.get('candidate_count')}`",
            f"- Ready for policy review: `{reserve_preflight_status.get('ready_for_policy_review_count')}`",
            f"- Blocked candidates: `{reserve_preflight_status.get('blocked_candidate_count')}`",
            f"- Readiness blockers: `{reserve_preflight_status.get('readiness_blocker_counts')}`",
            f"- Dataset join blockers: `{reserve_preflight_status.get('dataset_join_blocker_counts')}`",
            f"- Ready race IDs: `{reserve_preflight_status.get('ready_race_ids')}`",
            f"- Report: `{reserve_preflight_status.get('report')}`",
            "",
            "## Reserve Substitution Manual Review",
            f"- Status: `{reserve_manual_review_status.get('status')}`",
            f"- Ready candidates: `{reserve_manual_review_status.get('ready_candidate_count')}`",
            f"- Mapping pairs: `{reserve_manual_review_status.get('mapping_pair_count')}`",
            f"- Dataset join allowed: `{reserve_manual_review_status.get('dataset_join_allowed')}`",
            f"- Official result acceptance allowed: `{reserve_manual_review_status.get('official_result_acceptance_allowed')}`",
            f"- DB write: `{reserve_manual_review_status.get('db_write')}`",
            f"- Blockers: `{reserve_manual_review_status.get('blockers')}`",
            f"- Report: `{reserve_manual_review_status.get('report')}`",
            "",
            "## Rolling Model Comparison",
            f"- Status: `{rolling_comparison_status.get('status')}`",
            f"- Output: `{rolling_comparison_status.get('output_dir')}`",
            f"- Sample races: `{rolling_comparison_status.get('sample_race_count')}` / `{rolling_comparison_status.get('minimum_races_for_review')}`",
            f"- Sample runner rows: `{rolling_comparison_status.get('sample_runner_rows')}`",
            f"- Best candidate: `{rolling_comparison_status.get('best_candidate_key')}`",
            f"- Best top1: `{rolling_comparison_status.get('best_candidate_top1')}`",
            f"- Best top3: `{rolling_comparison_status.get('best_candidate_top3')}`",
            f"- Source rejected live odds candidates: `{rolling_comparison_status.get('source_rejected_live_odds_candidate_count')}`",
            f"- Source rows with rejected live odds candidates: `{rolling_comparison_status.get('source_rows_with_rejected_live_odds_candidates')}`",
            f"- Source rejected live odds candidate reasons: `{rolling_comparison_status.get('source_rejected_live_odds_candidate_reason_counts')}`",
            f"- Blockers: `{rolling_comparison_status.get('blockers')}`",
            "",
            "## Promotion Distance",
            f"- Status: `{promotion_distance_status.get('status')}`",
            f"- Promotion ready: `{promotion_distance_status.get('promotion_ready')}`",
            f"- Sample races: `{promotion_distance_status.get('sample_race_count')}`",
            f"- Sample runner rows: `{promotion_distance_status.get('sample_runner_rows')}`",
            f"- Source rejected live odds candidates: `{promotion_distance_status.get('source_rejected_live_odds_candidate_count')}`",
            f"- Source rows with rejected live odds candidates: `{promotion_distance_status.get('source_rows_with_rejected_live_odds_candidates')}`",
            f"- Source rejected live odds candidate reasons: `{promotion_distance_status.get('source_rejected_live_odds_candidate_reason_counts')}`",
            f"- Source exclusion reasons: `{promotion_distance_status.get('source_exclusion_reason_counts')}`",
            f"- Source odds exclusion reasons: `{promotion_distance_status.get('source_odds_exclusion_reason_counts')}`",
            f"- Source official-result missing race IDs: `{promotion_distance_status.get('source_official_result_evidence_db_missing_race_ids')}`",
            f"- Official-result coverage requested races: `{promotion_distance_status.get('official_result_coverage_requested_race_count')}`",
            f"- Official-result coverage races with rows: `{promotion_distance_status.get('official_result_coverage_races_with_rows_count')}`",
            f"- Official-result coverage missing races: `{promotion_distance_status.get('official_result_coverage_missing_race_count')}`",
            f"- Official-result missing exclusions: `{promotion_distance_status.get('official_result_coverage_missing_exclusion_count')}`",
            f"- Official-result runner path count: `{promotion_distance_status.get('official_result_runner_path_count')}`",
            f"- Official-result runner paths source: `{promotion_distance_status.get('official_result_runner_paths_source_field')}`",
            f"- Best candidate: `{promotion_distance_status.get('best_candidate_key')}`",
            f"- Best non-market candidate: `{promotion_distance_status.get('best_non_market_candidate_key')}`",
            f"- Best non-market top1 margin gap: `{promotion_distance_status.get('best_non_market_top1_margin_gap')}`",
            f"- Predeclared residual status: `{promotion_distance_status.get('predeclared_residual_candidate_status')}`",
            f"- Predeclared residual triggered races: `{promotion_distance_status.get('predeclared_residual_triggered_race_count')}`",
            f"- Blockers: `{promotion_distance_status.get('blockers')}`",
            f"- Report: `{promotion_distance_status.get('report')}`",
            "",
            "## High-Accuracy Timing Sources",
            f"- Packet status: `{dashboard.get('high_accuracy_refinement_status')}`",
            f"- Promotion PR gate: `{dashboard.get('high_accuracy_promotion_pr_gate_status')}`",
            f"- Timing-aligned rerun plan: `{dashboard.get('high_accuracy_timing_aligned_rerun_plan')}`",
            f"- Timing-aligned rerun execution status: `{dashboard.get('high_accuracy_timing_aligned_rerun_execution_status')}`",
            "",
            "## Feature Activation Gate",
            f"- Status: `{activation_status.get('status')}`",
            f"- Output: `{activation_status.get('output_dir')}`",
            f"- Activation allowed: `{activation_status.get('activation_allowed_features')}`",
            f"- Kept quarantined: `{activation_status.get('kept_quarantined_features')}`",
            f"- Data availability: `{activation_data.get('status')}`",
            f"- Candidate metric comparison: `{activation_data.get('candidate_metric_comparison_status')}`",
            f"- Blocker counts: `{fail_summary.get('reason_counts')}`",
            f"- Same-distance history status: `{same_distance_history.get('status')}`",
            f"- Same-distance feature rows: `{same_distance_history.get('feature_rows')}`",
            "",
            "## Readiness",
            f"- Decision: `{readiness.get('decision')}`",
            f"- Blockers: `{readiness.get('outstanding_blockers')}`",
            "",
            "No training, production promotion, registry mutation, production pointer update, label write, TGR enablement, betting action, EV action, feature engineering, or calibration-method change was performed. Any DB write is restricted to explicitly enabled append-only live odds or official-result evidence capture and is reported above.",
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
        f"Odds research gate: `{daily_status.get('odds_research_gate_status')}`",
        f"Odds research gate complete valid races: `{daily_status.get('odds_research_gate_complete_valid_prejump_odds_races')}`",
        f"Odds research next action: `{daily_status.get('odds_research_next_action')}`",
        f"Timing-aligned prediction rerun required: `{daily_status.get('timing_aligned_prediction_rerun_required')}`",
        f"Timing-aligned prediction rerun races: `{daily_status.get('timing_aligned_prediction_rerun_race_count')}`",
        f"Timing-aligned prediction rerun race IDs: `{daily_status.get('timing_aligned_prediction_rerun_race_ids')}`",
        f"Timing-aligned prediction rerun plan: `{daily_status.get('timing_aligned_prediction_rerun_plan_status')}`",
        f"Timing-aligned prediction rerun execution: `{daily_status.get('timing_aligned_prediction_rerun_execution_status')}`",
        f"Timing-aligned prediction rerun execution hard stops: `{daily_status.get('timing_aligned_prediction_rerun_execution_hard_stops') or []}`",
        f"Timing-aligned prediction rerun executed: `{daily_status.get('timing_aligned_prediction_rerun_execution_performed')}`",
        f"Timing-aligned prediction rerun output: `{daily_status.get('timing_aligned_prediction_rerun_output_dir')}`",
        f"Timing-aligned prediction rerun odds snapshot dir: `{daily_status.get('timing_aligned_prediction_rerun_odds_snapshot_dir')}`",
        f"Timing-aligned prediction rerun odds snapshot: `{daily_status.get('timing_aligned_prediction_rerun_odds_snapshot_status')}`",
        f"EV output rows: `{daily_status.get('ev_output_rows')}`",
        f"Live odds capture approval: `{daily_status.get('live_odds_capture_approval_status')}`",
        f"Live odds verified races: `{daily_status.get('live_odds_capture_verified_prejump_races')}`",
        f"Live odds capture windows: `{daily_status.get('live_odds_capture_window_offsets_minutes')}`",
        f"Autonomous live odds capture: `{daily_status.get('autonomous_live_odds_capture_status')}`",
        f"Autonomous odds ready races: `{daily_status.get('autonomous_live_odds_capture_ready_count')}`",
        f"Autonomous odds inserted rows: `{daily_status.get('autonomous_live_odds_inserted_rows')}`",
        f"Autonomous odds window coverage: `{daily_status.get('autonomous_live_odds_capture_window_coverage_status_counts')}`",
        f"Autonomous official result capture: `{daily_status.get('autonomous_official_result_capture_status')}`",
        f"Autonomous official result races: `{daily_status.get('autonomous_official_result_race_rows')}`",
        f"Autonomous official result runners: `{daily_status.get('autonomous_official_result_runner_rows')}`",
        f"Autonomous official result evidence DB ingest: `{daily_status.get('autonomous_official_result_evidence_db_ingest_status')}`",
        f"Autonomous official result evidence DB execute: `{daily_status.get('autonomous_official_result_evidence_db_execute')}`",
        f"Autonomous official result evidence DB write performed: `{daily_status.get('autonomous_official_result_evidence_db_write_performed')}`",
        f"Autonomous official result evidence valid race rows: `{daily_status.get('autonomous_official_result_evidence_valid_race_rows')}`",
        f"Autonomous official result evidence valid runner rows: `{daily_status.get('autonomous_official_result_evidence_valid_runner_rows')}`",
        f"Autonomous official result evidence blocked race rows: `{daily_status.get('autonomous_official_result_evidence_blocked_race_rows')}`",
        f"Autonomous official result evidence blocked runner rows: `{daily_status.get('autonomous_official_result_evidence_blocked_runner_rows')}`",
        f"Autonomous official result evidence inserted race rows: `{daily_status.get('autonomous_official_result_evidence_inserted_race_rows')}`",
        f"Autonomous official result evidence inserted runner rows: `{daily_status.get('autonomous_official_result_evidence_inserted_runner_rows')}`",
        f"Autonomous official result evidence blocker reasons: `{daily_status.get('autonomous_official_result_evidence_blocker_reason_counts')}`",
        f"Autonomous official result quarantine rows: `{daily_status.get('autonomous_official_result_quarantine_rows')}`",
        f"Autonomous official result quarantined race IDs: `{daily_status.get('autonomous_official_result_quarantined_race_ids')}`",
        f"Autonomous official result quarantine result boxes not in participants: `{daily_status.get('autonomous_official_result_quarantine_result_boxes_not_in_participants_counts')}`",
        f"Autonomous official result quarantine runner-set mismatch samples: `{daily_status.get('autonomous_official_result_quarantine_runner_set_mismatch_samples')}`",
        f"Autonomous official result awaiting-jump races: `{daily_status.get('autonomous_official_result_awaiting_jump_race_count')}`",
        f"Autonomous official result next recheck: `{daily_status.get('autonomous_official_result_awaiting_jump_next_recheck_after_local')}`",
        f"Unified evidence dataset: `{daily_status.get('unified_evidence_dataset_status')}`",
        f"Unified evidence rows: `{daily_status.get('unified_evidence_dataset_rows')}`",
        f"Unified label-eligible rows: `{daily_status.get('unified_label_evaluation_eligible_rows')}`",
        f"Unified Stage 2-eligible rows: `{daily_status.get('unified_stage2_evaluation_eligible_rows')}`",
        f"Unified odds-eligible rows: `{daily_status.get('unified_odds_evaluation_eligible_rows')}`",
        f"Unified full-evidence rows: `{daily_status.get('unified_evidence_eligible_rows')}`",
        f"Unified full-evidence rows scope: `{daily_status.get('unified_evidence_eligible_rows_scope')}`",
        f"Current-cycle unified full-evidence rows: `{daily_status.get('current_cycle_unified_evidence_eligible_rows')}`",
        f"Max observed unified full-evidence rows: `{daily_status.get('max_observed_unified_evidence_eligible_rows')}`",
        f"Unified official-result coverage requested races: `{daily_status.get('unified_evidence_official_result_coverage_requested_race_count')}`",
        f"Unified official-result requested race count source: `{daily_status.get('unified_evidence_official_result_coverage_requested_race_count_source')}`",
        f"Unified official-result coverage races with rows: `{daily_status.get('unified_evidence_official_result_coverage_races_with_rows_count')}`",
        f"Unified official-result coverage missing races: `{daily_status.get('unified_evidence_official_result_coverage_missing_race_count')}`",
        f"Unified official-result missing exclusions: `{daily_status.get('unified_evidence_official_result_coverage_missing_exclusion_count')}`",
        f"Unified official-result runner path count: `{daily_status.get('unified_evidence_official_result_runner_path_count')}`",
        f"Unified official-result runner paths source: `{daily_status.get('unified_evidence_official_result_runner_paths_source_field')}`",
        f"Unified artifact odds rows seen: `{daily_status.get('unified_evidence_artifact_odds_rows_seen')}`",
        f"Unified artifact odds rows accepted: `{daily_status.get('unified_evidence_artifact_odds_rows_accepted')}`",
        f"Unified artifact odds rows rejected: `{daily_status.get('unified_evidence_artifact_odds_rows_rejected')}`",
        f"Unified artifact odds rejection reasons: `{daily_status.get('unified_evidence_artifact_odds_rejection_reason_counts')}`",
        f"Unified rejected live odds candidates: `{daily_status.get('unified_rejected_live_odds_candidate_count')}`",
        f"Unified rows with rejected live odds candidates: `{daily_status.get('unified_rows_with_rejected_live_odds_candidates')}`",
        f"Unified rejected live odds candidate reasons: `{daily_status.get('unified_rejected_live_odds_candidate_reason_counts')}`",
        f"Backlog unified evidence: `{daily_status.get('backlog_unified_evidence_status')}`",
        f"Backlog unified datasets: `{daily_status.get('backlog_unified_evidence_dataset_count')}`",
        f"Backlog unified failed datasets: `{daily_status.get('backlog_unified_evidence_failed_dataset_count')}`",
        f"Backlog unified rows: `{daily_status.get('backlog_unified_evidence_rows')}`",
        f"Backlog unified official-result rows: `{daily_status.get('backlog_unified_evidence_official_result_rows')}`",
        f"Backlog unified strict-odds rows: `{daily_status.get('backlog_unified_evidence_strict_odds_rows')}`",
        f"Backlog unified full-evidence rows: `{daily_status.get('backlog_unified_evidence_eligible_rows')}`",
        f"Backlog unified full-evidence rows scope: `{daily_status.get('backlog_unified_evidence_eligible_rows_scope')}`",
        f"Backlog unified artifact odds rows seen: `{daily_status.get('backlog_unified_evidence_artifact_odds_rows_seen')}`",
        f"Backlog unified artifact odds rows accepted: `{daily_status.get('backlog_unified_evidence_artifact_odds_rows_accepted')}`",
        f"Backlog unified artifact odds rows rejected: `{daily_status.get('backlog_unified_evidence_artifact_odds_rows_rejected')}`",
        f"Backlog unified artifact odds rejection reasons: `{daily_status.get('backlog_unified_evidence_artifact_odds_rejection_reason_counts')}`",
        f"Backlog unified exclusion reasons: `{daily_status.get('backlog_unified_evidence_exclusion_reason_counts')}`",
        f"Backlog unified odds exclusion reasons: `{daily_status.get('backlog_unified_evidence_odds_exclusion_reason_counts')}`",
        f"Backlog unified rejected live odds candidates: `{daily_status.get('backlog_unified_rejected_live_odds_candidate_count')}`",
        f"Backlog unified rows with rejected live odds candidates: `{daily_status.get('backlog_unified_rows_with_rejected_live_odds_candidates')}`",
        f"Backlog unified rejected live odds candidate reasons: `{daily_status.get('backlog_unified_rejected_live_odds_candidate_reason_counts')}`",
        f"Backlog unified official-result coverage requested races: `{daily_status.get('backlog_unified_official_result_coverage_requested_race_count')}`",
        f"Backlog unified official-result requested race count source: `{daily_status.get('backlog_unified_official_result_coverage_requested_race_count_source')}`",
        f"Backlog unified official-result legacy requested race count without IDs: `{daily_status.get('backlog_unified_official_result_coverage_legacy_requested_race_count_without_ids')}`",
        f"Backlog unified official-result coverage races with rows: `{daily_status.get('backlog_unified_official_result_coverage_races_with_rows_count')}`",
        f"Backlog unified official-result coverage missing races: `{daily_status.get('backlog_unified_official_result_coverage_missing_race_count')}`",
        f"Backlog unified official-result missing exclusions: `{daily_status.get('backlog_unified_official_result_coverage_missing_exclusion_count')}`",
        f"Backlog unified race instances: `{daily_status.get('backlog_unified_race_instance_count')}`",
        f"Backlog unified deduped races: `{daily_status.get('backlog_unified_deduped_race_count')}`",
        f"Backlog unified deduped races with evidence: `{daily_status.get('backlog_unified_deduped_races_with_evidence')}`",
        f"Backlog unified deduped races without evidence: `{daily_status.get('backlog_unified_deduped_races_without_evidence')}`",
        f"Backlog unified deduped races without complete official result: `{daily_status.get('backlog_unified_deduped_races_without_complete_official_result')}`",
        f"Backlog unified deduped races without complete strict odds: `{daily_status.get('backlog_unified_deduped_races_without_complete_strict_odds')}`",
        f"Backlog unified sample-blocking gap races: `{daily_status.get('backlog_unified_sample_blocking_gap_count')}`",
        f"Backlog unified gap actions: `{daily_status.get('backlog_unified_gap_action_counts')}`",
        f"Backlog unified evidence-missing reasons: `{daily_status.get('backlog_unified_gap_evidence_missing_reason_counts')}`",
        f"Backlog unified top gap race IDs: `{daily_status.get('backlog_unified_top_gap_race_ids')}`",
        f"Backlog unified top gap races: `{daily_status.get('backlog_unified_top_gap_races')}`",
        f"Backlog unified top official-result-missing race IDs: `{daily_status.get('backlog_unified_top_official_result_missing_race_ids')}`",
        f"Backlog unified top official-result-missing races: `{daily_status.get('backlog_unified_top_official_result_missing_races')}`",
        f"Rolling comparison: `{daily_status.get('rolling_model_comparison_status')}`",
        f"Rolling comparison sample races: `{daily_status.get('rolling_model_comparison_sample_races')}` / `{daily_status.get('rolling_model_comparison_minimum_races_for_review')}`",
        f"Rolling comparison sample runner rows: `{daily_status.get('rolling_model_comparison_sample_runner_rows')}`",
        f"Rolling comparison best candidate: `{daily_status.get('rolling_model_comparison_best_candidate')}`",
        f"Rolling comparison best top1: `{daily_status.get('rolling_model_comparison_best_top1')}`",
        f"Rolling comparison best top3: `{daily_status.get('rolling_model_comparison_best_top3')}`",
        f"Rolling comparison source rejected live odds candidates: `{daily_status.get('rolling_model_comparison_source_rejected_live_odds_candidate_count')}`",
        f"Rolling comparison source rows with rejected live odds candidates: `{daily_status.get('rolling_model_comparison_source_rows_with_rejected_live_odds_candidates')}`",
        f"Rolling comparison source rejected live odds candidate reasons: `{daily_status.get('rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts')}`",
        f"Rolling comparison blockers: `{daily_status.get('rolling_model_comparison_blockers')}`",
        f"High-accuracy packet: `{daily_status.get('high_accuracy_refinement_status')}`",
        f"High-accuracy PR gate: `{daily_status.get('high_accuracy_promotion_pr_gate_status')}`",
        f"High-accuracy unified full-evidence rows: `{daily_status.get('high_accuracy_unified_evidence_eligible_rows')}`",
        f"High-accuracy unified full-evidence rows scope: `{daily_status.get('high_accuracy_unified_evidence_eligible_rows_scope')}`",
        f"High-accuracy timing-aligned rerun plan: `{daily_status.get('timing_aligned_rerun_plan')}`",
        f"High-accuracy timing-aligned rerun execution status: `{daily_status.get('timing_aligned_rerun_execution_status')}`",
        f"Reserve substitution preflight: `{daily_status.get('reserve_substitution_preflight_status')}`",
        f"Reserve substitution ready for policy review: `{daily_status.get('reserve_substitution_preflight_ready_for_policy_review_count')}`",
        f"Reserve substitution dataset join blockers: `{daily_status.get('reserve_substitution_preflight_dataset_join_blocker_counts')}`",
        f"Reserve substitution ready race IDs: `{daily_status.get('reserve_substitution_preflight_ready_race_ids')}`",
        f"Reserve substitution manual review: `{daily_status.get('reserve_substitution_manual_review_status')}`",
        f"Reserve substitution manual review ready candidates: `{daily_status.get('reserve_substitution_manual_review_ready_candidate_count')}`",
        f"Reserve substitution manual review mapping pairs: `{daily_status.get('reserve_substitution_manual_review_mapping_pair_count')}`",
        f"Reserve substitution manual review dataset join allowed: `{daily_status.get('reserve_substitution_manual_review_dataset_join_allowed')}`",
        f"Reserve substitution manual review official-result acceptance allowed: `{daily_status.get('reserve_substitution_manual_review_official_result_acceptance_allowed')}`",
        f"Reserve substitution manual review DB write: `{daily_status.get('reserve_substitution_manual_review_db_write')}`",
        f"Reserve substitution manual review blockers: `{daily_status.get('reserve_substitution_manual_review_blockers')}`",
        f"Reserve substitution policy impact: `{daily_status.get('reserve_substitution_policy_impact_status')}`",
        f"Reserve substitution policy impact ready candidates: `{daily_status.get('reserve_substitution_policy_impact_ready_candidate_count')}`",
        f"Reserve substitution policy impact mapping pairs: `{daily_status.get('reserve_substitution_policy_impact_mapping_pair_count')}`",
        f"Reserve substitution policy impact potential runner rows blocked: `{daily_status.get('reserve_substitution_policy_impact_potential_runner_rows_blocked')}`",
        f"Reserve substitution policy impact matched backlog top-gap races: `{daily_status.get('reserve_substitution_policy_impact_matched_backlog_top_gap_race_count')}`",
        f"Reserve substitution policy impact dataset join allowed: `{daily_status.get('reserve_substitution_policy_impact_dataset_join_allowed')}`",
        f"Reserve substitution policy impact official-result acceptance allowed: `{daily_status.get('reserve_substitution_policy_impact_official_result_acceptance_allowed')}`",
        f"Reserve substitution policy impact DB write: `{daily_status.get('reserve_substitution_policy_impact_db_write')}`",
        f"Reserve substitution policy impact blockers: `{daily_status.get('reserve_substitution_policy_impact_blockers')}`",
        f"Promotion distance: `{daily_status.get('promotion_distance_status')}`",
        f"Promotion distance ready: `{daily_status.get('promotion_distance_promotion_ready')}`",
        f"Promotion distance sample races: `{daily_status.get('promotion_distance_sample_race_count')}`",
        f"Promotion distance sample runner rows: `{daily_status.get('promotion_distance_sample_runner_rows')}`",
        f"Promotion distance source rejected live odds candidates: `{daily_status.get('promotion_distance_source_rejected_live_odds_candidate_count')}`",
        f"Promotion distance source rejected live odds candidate reasons: `{daily_status.get('promotion_distance_source_rejected_live_odds_candidate_reason_counts')}`",
        f"Promotion distance source exclusion reasons: `{daily_status.get('promotion_distance_source_exclusion_reason_counts')}`",
        f"Promotion distance source odds exclusion reasons: `{daily_status.get('promotion_distance_source_odds_exclusion_reason_counts')}`",
        f"Promotion distance source official-result missing race IDs: `{daily_status.get('promotion_distance_source_official_result_evidence_db_missing_race_ids')}`",
        f"Promotion distance official-result coverage requested races: `{daily_status.get('promotion_distance_official_result_coverage_requested_race_count')}`",
        f"Promotion distance official-result requested race count source: `{daily_status.get('promotion_distance_official_result_coverage_requested_race_count_source')}`",
        f"Promotion distance official-result legacy requested race count without IDs: `{daily_status.get('promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids')}`",
        f"Promotion distance official-result coverage races with rows: `{daily_status.get('promotion_distance_official_result_coverage_races_with_rows_count')}`",
        f"Promotion distance official-result coverage missing races: `{daily_status.get('promotion_distance_official_result_coverage_missing_race_count')}`",
        f"Promotion distance official-result missing exclusions: `{daily_status.get('promotion_distance_official_result_coverage_missing_exclusion_count')}`",
        f"Promotion distance official-result runner path count: `{daily_status.get('promotion_distance_official_result_runner_path_count')}`",
        f"Promotion distance official-result runner paths source: `{daily_status.get('promotion_distance_official_result_runner_paths_source_field')}`",
        f"Promotion distance best candidate: `{daily_status.get('promotion_distance_best_candidate_key')}`",
        f"Promotion distance best non-market candidate: `{daily_status.get('promotion_distance_best_non_market_candidate_key')}`",
        f"Promotion distance best non-market top1 margin gap: `{daily_status.get('promotion_distance_best_non_market_top1_margin_gap')}`",
        f"Promotion distance predeclared residual status: `{daily_status.get('promotion_distance_predeclared_residual_candidate_status')}`",
        f"Promotion distance predeclared residual triggered races: `{daily_status.get('promotion_distance_predeclared_residual_triggered_race_count')}`",
        f"Promotion distance blockers: `{daily_status.get('promotion_distance_blockers')}`",
        f"Promotion distance report: `{daily_status.get('promotion_distance_report')}`",
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


def materialize_root_stage2_predictions(
    daily_dir: Path | None,
    *,
    output_dir: Path,
    generated_at: datetime,
) -> dict[str, Any]:
    status_path = output_dir / "stage2_shadow_predictions_status.json"
    base_status: dict[str, Any] = {
        "schema_version": "stage2_shadow_predictions_first_class_v1",
        "generated_at": generated_at.isoformat(),
        "daily_shadow_run_dir": relpath(daily_dir),
        "root_stage2_predictions_path": None,
        "fallback_stage2_predictions_path": None,
        "stage2_prediction_rows": 0,
        "root_materialized": False,
        "protected_paths_unchanged": True,
    }
    if daily_dir is None:
        status = {
            **base_status,
            "status": "STAGE2_SHADOW_RUN_DIR_MISSING",
            "returncode": 0,
        }
        write_json(status_path, status)
        return status

    root_path = daily_dir / "stage2_shadow_predictions.jsonl"
    fallback_path = daily_dir / "shadow_score_live" / "stage2_shadow_predictions.jsonl"
    base_status.update(
        {
            "root_stage2_predictions_path": relpath(root_path),
            "fallback_stage2_predictions_path": relpath(fallback_path),
        }
    )
    root_existed_before = root_path.exists()
    root_rows = read_jsonl(root_path)
    fallback_rows = read_jsonl(fallback_path)
    base_status.update(
        {
            "root_stage2_prediction_rows_before": len(root_rows),
            "fallback_stage2_prediction_rows": len(fallback_rows),
        }
    )
    if root_path.exists() and (root_rows or not fallback_rows):
        status = {
            **base_status,
            "status": "STAGE2_SHADOW_PREDICTIONS_ROOT_PRESENT",
            "stage2_prediction_rows": len(root_rows),
            "root_sha256": sha256_file(root_path),
            "returncode": 0,
        }
        write_json(status_path, status)
        return status

    if not fallback_rows:
        status = {
            **base_status,
            "status": "STAGE2_SHADOW_PREDICTIONS_MISSING",
            "returncode": 0,
        }
        write_json(status_path, status)
        return status

    root_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fallback_path, root_path)
    fallback_hash = sha256_file(fallback_path)
    root_hash = sha256_file(root_path)
    status = {
        **base_status,
        "status": (
            "STAGE2_SHADOW_PREDICTIONS_EMPTY_ROOT_REPLACED"
            if root_existed_before
            else "STAGE2_SHADOW_PREDICTIONS_ROOT_MATERIALIZED"
        ),
        "stage2_prediction_rows": len(fallback_rows),
        "fallback_sha256": fallback_hash,
        "root_sha256": root_hash,
        "hashes_match": fallback_hash == root_hash,
        "root_materialized": True,
        "returncode": 0 if fallback_hash == root_hash else 1,
    }
    write_json(status_path, status)
    return status


def final_verdict_for(
    *,
    steps: Sequence[Mapping[str, Any]],
    protected_paths_unchanged: bool,
    required_outputs_present: bool,
) -> str:
    if not protected_paths_unchanged or not required_outputs_present:
        return "NEEDS_MORE_TOOLING"
    required_step_names = {
        "refresh_prejump_races",
        "autonomous_live_odds_capture",
        "autonomous_official_result_capture",
        "unified_evidence_dataset",
        "rolling_model_comparison",
        "daily_shadow_run",
        "stage2_shadow_predictions_first_class",
        "result_join",
        "aggregate_results",
        "status_report",
        "feature_activation_gate",
    }
    if any(step.get("name") == "refresh_odds_capture_candidates" for step in steps):
        required_step_names.add("refresh_odds_capture_candidates")
    if any(step.get("name") == "timing_aligned_prediction_rerun" for step in steps):
        required_step_names.update(
            {
                "timing_aligned_prediction_rerun",
                "timing_aligned_stage2_shadow_predictions_first_class",
                "timing_aligned_shadow_odds_snapshot",
            }
        )
    failed_required = []
    for step in steps:
        name = str(step.get("name") or "")
        is_required = name in required_step_names or name.startswith(
            "backlog_unified_evidence_dataset_"
        )
        if is_required and step.get("returncode") != 0:
            failed_required.append(step)
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

    skip_primary_refresh = bool(getattr(args, "skip_primary_refresh", False))
    if skip_primary_refresh and not args.skip_shadow_run and not args.input_dir:
        raise RuntimeError("skip_primary_refresh_requires_skip_shadow_run_or_input_dir")
    if not args.skip_refresh and not skip_primary_refresh:
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
            step_command(
                name="refresh_prejump_races",
                command=refresh_command,
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
        )
    else:
        write_json(
            output_dir / "refresh_prejump_report.json",
            {
                "status": "SKIPPED",
                "reason": (
                    "skip_refresh_requested"
                    if args.skip_refresh
                    else "skip_primary_refresh_requested"
                ),
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

    odds_capture_input_dirs = list(input_dirs)
    odds_capture_limit = (
        args.odds_capture_refresh_limit
        if args.odds_capture_refresh_limit is not None
        else args.refresh_limit
    )
    autonomous_odds_capture_limit = (
        args.autonomous_odds_capture_limit
        if args.autonomous_odds_capture_limit is not None
        else odds_capture_limit
    )
    if (
        args.enable_autonomous_odds_capture
        and args.input_dir is None
        and not args.skip_refresh
    ):
        odds_capture_refresh_dir = output_dir / "odds_capture_refreshed_upcoming"
        odds_capture_refresh_command = [
            *refresh_command_prefix(args.refresh_command_mode),
            str(ROOT / "scripts/refresh_prejump_upcoming.py"),
            "--upcoming-dir",
            str(odds_capture_refresh_dir),
            "--days-ahead",
            str(args.days_ahead),
            "--min-minutes",
            str(args.odds_capture_min_minutes),
            "--max-minutes",
            str(args.odds_capture_max_minutes),
            "--limit",
            str(odds_capture_limit),
            "--output",
            str(output_dir / "odds_capture_refresh_report.json"),
        ]
        if args.refresh_dry_run:
            odds_capture_refresh_command.append("--dry-run")
        steps.append(
            step_command(
                name="refresh_odds_capture_candidates",
                command=odds_capture_refresh_command,
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
        )
        odds_capture_input_dirs = [odds_capture_refresh_dir]

    autonomous_odds_capture_dir = evidence_root / f"autonomous_live_odds_capture_{run_id}_autopilot"
    autonomous_odds_capture_report: dict[str, Any] | None = None
    odds_capture_refresh_report = (
        load_json(output_dir / "odds_capture_refresh_report.json")
        or load_json(output_dir / "refresh_prejump_report.json")
    )
    if args.enable_autonomous_odds_capture:
        autonomous_odds_command = autonomous_live_odds_capture_command(
            input_dirs=odds_capture_input_dirs,
            capture_dir=autonomous_odds_capture_dir,
            db_path=args.db,
            current_time=current_time,
            limit=autonomous_odds_capture_limit,
            execute=args.execute_autonomous_odds_capture,
            allow_auto_scrape_odds=args.allow_auto_scrape_odds,
        )
        autonomous_odds_step = step_command(
            name="autonomous_live_odds_capture",
            command=autonomous_odds_command,
            output_dir=output_dir,
            timeout_seconds=args.step_timeout_seconds,
        )
        steps.append(autonomous_odds_step)
        autonomous_odds_report_path = (
            autonomous_odds_capture_dir / "autonomous_live_odds_capture_report.json"
        )
        autonomous_odds_capture_report = (
            load_json(autonomous_odds_report_path)
            or {}
        )
        if not autonomous_odds_capture_report and autonomous_odds_step.get("timed_out"):
            autonomous_odds_capture_report = (
                load_json_after_timeout_grace(autonomous_odds_report_path)
                or {}
            )
        autonomous_odds_recovered = bool(
            autonomous_odds_capture_report
            and autonomous_odds_step.get("returncode") not in (0, None)
        )
        if autonomous_odds_recovered:
            autonomous_odds_step["recovered_output_report"] = relpath(
                autonomous_odds_report_path
            )
            autonomous_odds_step["recovered_final_status"] = (
                autonomous_odds_capture_report.get("final_status")
            )
        autonomous_odds_capture_status = build_autonomous_live_odds_capture_status(
            generated_at=generated_at,
            capture_dir=autonomous_odds_capture_dir,
            capture_report=autonomous_odds_capture_report or None,
            odds_capture_refresh_report=odds_capture_refresh_report,
            attempted=True,
            returncode=autonomous_odds_step.get("returncode"),
            timed_out=bool(autonomous_odds_step.get("timed_out")),
            recovered_from_step_failure=autonomous_odds_recovered,
        )
    else:
        autonomous_odds_capture_status = build_autonomous_live_odds_capture_status(
            generated_at=generated_at,
            capture_dir=None,
            capture_report=None,
            odds_capture_refresh_report=odds_capture_refresh_report,
            skipped_reason="enable_autonomous_odds_capture_not_set",
        )
    write_json(
        output_dir / "autonomous_live_odds_capture_status.json",
        autonomous_odds_capture_status,
    )

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
        steps.append(
            step_command(
                name="daily_shadow_run",
                command=daily_command,
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
        )
        stage2_materialization_status = materialize_root_stage2_predictions(
            daily_dir,
            output_dir=output_dir,
            generated_at=generated_at,
        )
        steps.append(
            {
                "name": "stage2_shadow_predictions_first_class",
                "command": [],
                "cwd": str(ROOT),
                "started_at": generated_at.isoformat(),
                "finished_at": datetime.now().astimezone().isoformat(),
                "returncode": stage2_materialization_status.get("returncode"),
                "status": (
                    "PASS"
                    if stage2_materialization_status.get("returncode") == 0
                    else "FAIL"
                ),
                "stage2_status": stage2_materialization_status.get("status"),
                "stage2_status_path": relpath(
                    output_dir / "stage2_shadow_predictions_status.json"
                ),
                "root_materialized": stage2_materialization_status.get(
                    "root_materialized"
                ),
            }
        )
    else:
        daily_dir = args.shadow_run_dir or latest_artifact(evidence_root, "daily_race_ingest_shadow_", "shadow_manifest.json") or latest_artifact(evidence_root, "forward_shadow_run_", "shadow_manifest.json")

    autonomous_result_capture_dir = (
        evidence_root / f"autonomous_official_result_capture_{run_id}_autopilot"
    )
    autonomous_result_capture_report: dict[str, Any] | None = None
    result_capture_upcoming_dir = input_dirs[0] if input_dirs else refresh_dir
    result_capture_current_time = parse_datetime_or_none(current_time)
    result_capture_target_date = (
        result_capture_current_time or generated_at
    ).date().isoformat()
    result_capture_shadow_dir = (
        daily_dir
        if daily_dir is not None
        and (
            (daily_dir / "shadow_predictions.jsonl").exists()
            or (daily_dir / "stage2_shadow_predictions.jsonl").exists()
        )
        else None
    )

    def add_autonomous_result_capture_context(status: dict[str, Any]) -> dict[str, Any]:
        status["target_date"] = result_capture_target_date
        status["upcoming_dir"] = relpath(result_capture_upcoming_dir)
        status["shadow_run_dir"] = relpath(result_capture_shadow_dir)
        status["candidate_source"] = (
            "shadow_run_predictions"
            if result_capture_shadow_dir is not None
            else "upcoming_or_snapshot"
        )
        return status

    if args.enable_autonomous_result_capture:
        result_capture_command_current_time = current_step_time_iso()
        autonomous_result_command = autonomous_official_result_capture_command(
            target_date=result_capture_target_date,
            upcoming_dir=result_capture_upcoming_dir if result_capture_shadow_dir is None else None,
            shadow_run_dir=result_capture_shadow_dir,
            snapshot_dir=ROOT / "artifacts/prediction_snapshots"
            if result_capture_shadow_dir is None
            else None,
            output_dir=autonomous_result_capture_dir,
            db_path=args.db,
            current_time=result_capture_command_current_time,
            require_ready_snapshot=result_capture_shadow_dir is None,
            include_live_odds_backlog=result_capture_shadow_dir is not None,
            backlog_evidence_root=evidence_root,
            backlog_limit=args.result_backlog_limit,
            backlog_shadow_run_limit=args.result_backlog_shadow_run_limit,
            backlog_lookback_days=args.result_backlog_lookback_days,
            execute_db_ingest=True,
        )
        autonomous_result_capture_status = add_autonomous_result_capture_context(
            build_autonomous_official_result_capture_status(
                generated_at=generated_at,
                capture_dir=autonomous_result_capture_dir,
                capture_report=None,
                progress_report=None,
                attempted=True,
                in_progress=True,
            )
        )
        write_json(
            output_dir / "autonomous_official_result_capture_status.json",
            autonomous_result_capture_status,
        )
        autonomous_result_step = step_command(
            name="autonomous_official_result_capture",
            command=autonomous_result_command,
            output_dir=output_dir,
            timeout_seconds=args.step_timeout_seconds,
        )
        steps.append(autonomous_result_step)
        autonomous_result_capture_report = (
            load_json(
                autonomous_result_capture_dir
                / "autonomous_official_result_capture_report.json"
            )
            or {}
        )
        autonomous_result_progress_report = load_json(
            autonomous_result_capture_dir
            / "autonomous_official_result_capture_progress.json"
        ) or {}
        autonomous_result_capture_status = build_autonomous_official_result_capture_status(
            generated_at=generated_at,
            capture_dir=autonomous_result_capture_dir,
            capture_report=autonomous_result_capture_report or None,
            progress_report=autonomous_result_progress_report or None,
            attempted=True,
            returncode=autonomous_result_step.get("returncode"),
            timed_out=bool(autonomous_result_step.get("timed_out")),
        )
    else:
        autonomous_result_capture_status = build_autonomous_official_result_capture_status(
            generated_at=generated_at,
            capture_dir=None,
            capture_report=None,
            skipped_reason="enable_autonomous_result_capture_not_set",
        )
    autonomous_result_capture_status = add_autonomous_result_capture_context(
        autonomous_result_capture_status
    )
    write_json(
        output_dir / "autonomous_official_result_capture_status.json",
        autonomous_result_capture_status,
    )

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
            timeout_seconds=args.step_timeout_seconds,
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
    timing_aligned_rerun_plan = build_timing_aligned_prediction_rerun_plan(
        generated_at=generated_at,
        odds_snapshot_status=odds_snapshot_status,
        output_dir=output_dir,
        db_path=args.db,
        shadow_model=shadow_model,
        score_command_mode=args.score_command_mode,
    )
    write_json(
        output_dir / "timing_aligned_prediction_rerun_plan.json",
        timing_aligned_rerun_plan,
    )
    timing_aligned_rerun_execution_status = execute_timing_aligned_prediction_rerun_plan(
        generated_at=generated_at,
        plan=timing_aligned_rerun_plan,
        output_dir=output_dir,
        db_path=args.db,
        current_time=current_time,
        timeout_seconds=args.step_timeout_seconds,
        steps=steps,
        skip_odds_snapshot=args.skip_odds_snapshot,
    )
    timing_aligned_rerun_plan = {
        **timing_aligned_rerun_plan,
        "execution_performed": timing_aligned_rerun_execution_status.get(
            "execution_performed",
            False,
        ),
        "execution_status": timing_aligned_rerun_execution_status.get("status"),
        "execution_returncode": timing_aligned_rerun_execution_status.get("returncode"),
        "execution": timing_aligned_rerun_execution_status,
    }
    write_json(
        output_dir / "timing_aligned_prediction_rerun_plan.json",
        timing_aligned_rerun_plan,
    )
    write_json(
        output_dir / "timing_aligned_prediction_rerun_execution_status.json",
        timing_aligned_rerun_execution_status,
    )
    rerun_odds_snapshot_status = timing_aligned_rerun_execution_status.get(
        "rerun_odds_snapshot"
    )
    rerun_daily_dir = rooted_path(
        timing_aligned_rerun_execution_status.get("rerun_daily_shadow_run_dir")
    )
    if (
        isinstance(rerun_odds_snapshot_status, Mapping)
        and rerun_odds_snapshot_status.get("status")
        and rerun_daily_dir is not None
        and (rerun_daily_dir / "shadow_predictions.jsonl").exists()
    ):
        daily_dir = rerun_daily_dir
        rerun_odds_dir = rooted_path(
            timing_aligned_rerun_execution_status.get("rerun_odds_snapshot_dir")
        )
        if rerun_odds_dir is not None:
            odds_dir = rerun_odds_dir
        odds_snapshot_status = dict(rerun_odds_snapshot_status)
        write_json(output_dir / "shadow_odds_snapshot_status.json", odds_snapshot_status)

    unified_dataset_dir = evidence_root / f"unified_evidence_dataset_{run_id}_autopilot"
    unified_dataset_report: dict[str, Any] | None = None
    if args.skip_unified_dataset:
        unified_dataset_status = build_unified_evidence_dataset_status(
            generated_at=generated_at,
            dataset_dir=None,
            dataset_report=None,
            skipped_reason="skip_unified_dataset_requested",
        )
    elif daily_dir is not None and (daily_dir / "shadow_predictions.jsonl").exists():
        odds_jsonl_paths = []
        if odds_dir is not None:
            odds_jsonl_paths.append(odds_dir / "shadow_odds_snapshot.jsonl")
        official_result_runner_paths = [
            autonomous_result_capture_dir / "official_result_runners.jsonl"
        ]
        unified_dataset_command = unified_evidence_dataset_command(
            shadow_run_dir=daily_dir,
            output_dir=unified_dataset_dir,
            db_path=args.db,
            odds_jsonl_paths=odds_jsonl_paths,
            official_result_runner_paths=official_result_runner_paths,
        )
        unified_dataset_step = step_command(
            name="unified_evidence_dataset",
            command=unified_dataset_command,
            output_dir=output_dir,
            timeout_seconds=args.step_timeout_seconds,
        )
        steps.append(unified_dataset_step)
        unified_dataset_report = (
            load_json(unified_dataset_dir / "unified_evidence_dataset_report.json")
            or {}
        )
        unified_dataset_status = build_unified_evidence_dataset_status(
            generated_at=generated_at,
            dataset_dir=unified_dataset_dir,
            dataset_report=unified_dataset_report or None,
            attempted=True,
            returncode=unified_dataset_step.get("returncode"),
        )
    else:
        unified_dataset_status = build_unified_evidence_dataset_status(
            generated_at=generated_at,
            dataset_dir=None,
            dataset_report=None,
            skipped_reason="shadow_predictions_missing",
        )
    write_json(output_dir / "unified_evidence_dataset_status.json", unified_dataset_status)

    backlog_unified_reports: list[dict[str, Any]] = []
    backlog_unified_failures: list[dict[str, Any]] = []
    official_result_gap_context_by_race = official_result_quarantine_context_by_race(
        autonomous_result_capture_dir / "official_result_quarantine.jsonl"
    )
    if args.skip_unified_dataset:
        backlog_unified_status = build_backlog_unified_evidence_status(
            generated_at=generated_at,
            reports=[],
            skipped_reason="skip_unified_dataset_requested",
            official_result_gap_context_by_race=official_result_gap_context_by_race,
        )
    else:
        backlog_candidates = backlog_unified_shadow_run_candidates(autonomous_result_capture_dir)
        official_result_runners_path = (
            autonomous_result_capture_dir / "official_result_runners.jsonl"
        )
        for index, backlog_candidate in enumerate(backlog_candidates, start=1):
            backlog_shadow_dir = backlog_candidate["shadow_run_dir"]
            candidate_race_ids = list(backlog_candidate.get("candidate_race_ids") or [])
            backlog_dataset_dir = (
                evidence_root
                / f"unified_evidence_dataset_{run_id}_autopilot_backlog_{index:03d}"
            )
            filtered_official_result_path = (
                output_dir
                / "backlog_unified_evidence_inputs"
                / f"official_result_runners_backlog_{index:03d}.jsonl"
            )
            filtered_official_result_rows = filtered_official_result_rows_for_race_ids(
                official_result_runners_path,
                filtered_official_result_path,
                candidate_race_ids,
            )
            backlog_dataset_command = unified_evidence_dataset_command(
                shadow_run_dir=backlog_shadow_dir,
                output_dir=backlog_dataset_dir,
                db_path=args.db,
                odds_jsonl_paths=shadow_odds_snapshot_paths_for_daily_dir(
                    evidence_root=evidence_root,
                    daily_dir=backlog_shadow_dir,
                ),
                official_result_runner_paths=backlog_official_result_runner_paths(
                    filtered_official_result_path,
                    row_count=filtered_official_result_rows,
                ),
            )
            backlog_step = step_command(
                name=f"backlog_unified_evidence_dataset_{index:03d}",
                command=backlog_dataset_command,
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
            steps.append(backlog_step)
            backlog_report = (
                load_json(backlog_dataset_dir / "unified_evidence_dataset_report.json")
                or {}
            )
            if backlog_report and backlog_step.get("returncode") == 0:
                backlog_report = dict(backlog_report)
                backlog_report.setdefault("output_dir", relpath(backlog_dataset_dir))
                backlog_report.setdefault("shadow_run_dir", relpath(backlog_shadow_dir))
                backlog_report["candidate_race_ids"] = candidate_race_ids
                backlog_report["candidate_race_count"] = len(candidate_race_ids)
                backlog_report["filtered_official_result_runner_rows"] = (
                    filtered_official_result_rows
                )
                backlog_report["filtered_official_result_runners_jsonl"] = relpath(
                    filtered_official_result_path
                )
                backlog_report["filtered_official_result_runners_empty"] = (
                    filtered_official_result_rows <= 0
                )
                backlog_unified_reports.append(backlog_report)
            else:
                backlog_unified_failures.append(
                    {
                        "output_dir": relpath(backlog_dataset_dir),
                        "shadow_run_dir": relpath(backlog_shadow_dir),
                        "returncode": backlog_step.get("returncode"),
                    }
                )
        backlog_unified_status = build_backlog_unified_evidence_status(
            generated_at=generated_at,
            reports=backlog_unified_reports,
            failures=backlog_unified_failures,
            official_result_gap_context_by_race=official_result_gap_context_by_race,
        )
    write_json(
        output_dir / "backlog_unified_evidence_datasets_status.json",
        backlog_unified_status,
    )
    reserve_substitution_preflight_dir = (
        evidence_root
        / f"official_result_reserve_substitution_preflight_{run_id}_autopilot"
    )
    reserve_substitution_preflight_report_path = (
        reserve_substitution_preflight_dir
        / "official_result_reserve_substitution_preflight.json"
    )
    if not args.skip_unified_dataset:
        reserve_preflight_step = step_command(
            name="official_result_reserve_substitution_preflight",
            command=reserve_substitution_preflight_command(
                backlog_unified_evidence_status=(
                    output_dir / "backlog_unified_evidence_datasets_status.json"
                ),
                output_dir=reserve_substitution_preflight_dir,
            ),
            output_dir=output_dir,
            timeout_seconds=args.step_timeout_seconds,
        )
        steps.append(reserve_preflight_step)

    rolling_comparison_dir = evidence_root / f"rolling_model_comparison_{run_id}_autopilot"
    rolling_comparison_report: dict[str, Any] | None = None
    unified_report_paths: list[Path] = []
    if args.skip_unified_dataset:
        rolling_comparison_status = build_rolling_model_comparison_status(
            generated_at=generated_at,
            comparison_dir=None,
            comparison_report=None,
            skipped_reason="skip_unified_dataset_requested",
        )
    else:
        current_unified_report_path = unified_dataset_dir / "unified_evidence_dataset_report.json"
        if current_unified_report_path.exists():
            unified_report_paths.append(current_unified_report_path)
        for report in backlog_unified_reports:
            dataset_dir = rooted_path(report.get("output_dir"))
            if dataset_dir is None:
                continue
            dataset_report = dataset_dir / "unified_evidence_dataset_report.json"
            if dataset_report.exists():
                unified_report_paths.append(dataset_report)
        unified_report_paths.extend(
            historical_unified_evidence_report_paths(
                evidence_root,
                exclude_paths=unified_report_paths,
            )
        )
        unified_report_paths = unique_sorted_report_paths(unified_report_paths)
        if unified_report_paths:
            rolling_comparison_command = rolling_model_comparison_command(
                unified_evidence_reports=unified_report_paths,
                output_dir=rolling_comparison_dir,
            )
            rolling_comparison_step = step_command(
                name="rolling_model_comparison",
                command=rolling_comparison_command,
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
            steps.append(rolling_comparison_step)
            rolling_comparison_report = (
                load_json(rolling_comparison_dir / "rolling_model_comparison_report.json")
                or {}
            )
            rolling_comparison_status = build_rolling_model_comparison_status(
                generated_at=generated_at,
                comparison_dir=rolling_comparison_dir,
                comparison_report=rolling_comparison_report or None,
                attempted=True,
                returncode=rolling_comparison_step.get("returncode"),
            )
        else:
            rolling_comparison_status = build_rolling_model_comparison_status(
                generated_at=generated_at,
                comparison_dir=None,
                comparison_report=None,
                skipped_reason="unified_evidence_reports_missing",
            )
    write_json(
        output_dir / "rolling_model_comparison_status.json",
        rolling_comparison_status,
    )

    pre_race_gated_dir = (
        evidence_root / f"pre_race_gated_challenger_{run_id}_autopilot"
    )
    pre_race_gated_report_path = (
        pre_race_gated_dir / "pre_race_gated_challenger_report.json"
    )
    rolling_comparison_report_path = (
        rolling_comparison_dir / "rolling_model_comparison_report.json"
    )
    rolling_runner_matrix_csv = rolling_comparison_dir / "market_residual_runner_matrix.csv"
    if (
        not args.skip_unified_dataset
        and rolling_comparison_report_path.exists()
        and rolling_runner_matrix_csv.exists()
    ):
        pre_race_gated_step = step_command(
            name="pre_race_gated_challenger",
            command=pre_race_gated_challenger_command(
                runner_matrix_csv=rolling_runner_matrix_csv,
                output_dir=pre_race_gated_dir,
            ),
            output_dir=output_dir,
            timeout_seconds=args.step_timeout_seconds,
        )
        steps.append(pre_race_gated_step)

    high_accuracy_dir = evidence_root / f"high_accuracy_refinement_packet_{run_id}_autopilot"
    high_accuracy_after_promotion_dir = (
        evidence_root
        / f"high_accuracy_refinement_packet_{run_id}_autopilot_post_promotion_distance"
    )
    promotion_distance_dir = (
        evidence_root / f"promotion_distance_report_{run_id}_autopilot"
    )
    promotion_distance_report_path = (
        promotion_distance_dir / "promotion_distance_report.json"
    )
    high_accuracy_report: dict[str, Any] | None = None
    unified_report_path = best_unified_evidence_report_path(unified_report_paths)
    if unified_report_path is None:
        unified_report_path = unified_dataset_dir / "unified_evidence_dataset_report.json"
    odds_gate_report_path = odds_research_gate_report_path_from_snapshot_status(
        odds_snapshot_status
    )
    stage2_predictions_path = (
        daily_dir / "stage2_shadow_predictions.jsonl"
        if daily_dir is not None
        else None
    )
    if args.skip_unified_dataset:
        high_accuracy_status = build_high_accuracy_refinement_status(
            generated_at=generated_at,
            packet_dir=None,
            packet_report=None,
            skipped_reason="skip_unified_dataset_requested",
        )
    elif unified_report_path.exists():
        high_accuracy_command = high_accuracy_refinement_packet_command(
            unified_evidence_report=unified_report_path,
            output_dir=high_accuracy_dir,
            stage2_predictions=stage2_predictions_path,
            odds_augmented_report=(
                rolling_comparison_dir / "rolling_model_comparison_report.json"
            ),
            odds_gate_report=odds_gate_report_path,
            backlog_unified_evidence_status=(
                output_dir / "backlog_unified_evidence_datasets_status.json"
            ),
            reserve_substitution_preflight=reserve_substitution_preflight_report_path,
            timing_aligned_rerun_plan=(
                output_dir / "timing_aligned_prediction_rerun_plan.json"
            ),
            timing_aligned_rerun_execution_status=(
                output_dir / "timing_aligned_prediction_rerun_execution_status.json"
            ),
        )
        high_accuracy_step = step_command(
            name="high_accuracy_refinement_packet",
            command=high_accuracy_command,
            output_dir=output_dir,
            timeout_seconds=args.step_timeout_seconds,
        )
        steps.append(high_accuracy_step)
        high_accuracy_report = (
            load_json(high_accuracy_dir / "high_accuracy_refinement_packet.json")
            or {}
        )
        high_accuracy_status = build_high_accuracy_refinement_status(
            generated_at=generated_at,
            packet_dir=high_accuracy_dir,
            packet_report=high_accuracy_report or None,
            attempted=True,
            returncode=high_accuracy_step.get("returncode"),
        )
        high_accuracy_gate_path = high_accuracy_dir / "promotion_pr_gate.json"
        if (
            rolling_comparison_report_path.exists()
            and pre_race_gated_report_path.exists()
            and high_accuracy_gate_path.exists()
        ):
            promotion_distance_step = step_command(
                name="promotion_distance_report",
                command=promotion_distance_report_command(
                    rolling_report=rolling_comparison_report_path,
                    pre_race_gated_report=pre_race_gated_report_path,
                    high_accuracy_gate=high_accuracy_gate_path,
                    output_dir=promotion_distance_dir,
                ),
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
            steps.append(promotion_distance_step)
            if promotion_distance_report_path.exists():
                high_accuracy_with_promotion_step = step_command(
                    name="high_accuracy_refinement_packet_with_promotion_distance",
                    command=high_accuracy_refinement_packet_command(
                        unified_evidence_report=unified_report_path,
                        output_dir=high_accuracy_after_promotion_dir,
                        stage2_predictions=stage2_predictions_path,
                        odds_augmented_report=rolling_comparison_report_path,
                        odds_gate_report=odds_gate_report_path,
                        backlog_unified_evidence_status=(
                            output_dir / "backlog_unified_evidence_datasets_status.json"
                        ),
                        promotion_distance_report=promotion_distance_report_path,
                        reserve_substitution_preflight=(
                            reserve_substitution_preflight_report_path
                        ),
                        timing_aligned_rerun_plan=(
                            output_dir / "timing_aligned_prediction_rerun_plan.json"
                        ),
                        timing_aligned_rerun_execution_status=(
                            output_dir
                            / "timing_aligned_prediction_rerun_execution_status.json"
                        ),
                    ),
                    output_dir=output_dir,
                    timeout_seconds=args.step_timeout_seconds,
                )
                steps.append(high_accuracy_with_promotion_step)
                high_accuracy_report = (
                    load_json(
                        high_accuracy_after_promotion_dir
                        / "high_accuracy_refinement_packet.json"
                    )
                    or {}
                )
                high_accuracy_status = build_high_accuracy_refinement_status(
                    generated_at=generated_at,
                    packet_dir=high_accuracy_after_promotion_dir,
                    packet_report=high_accuracy_report or None,
                    attempted=True,
                    returncode=high_accuracy_with_promotion_step.get("returncode"),
                )
    else:
        high_accuracy_status = build_high_accuracy_refinement_status(
            generated_at=generated_at,
            packet_dir=None,
            packet_report=None,
            skipped_reason="unified_evidence_report_missing",
        )
    write_json(output_dir / "high_accuracy_refinement_status.json", high_accuracy_status)

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
        steps.append(
            step_command(
                name="result_join",
                command=join_command,
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
        )
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
        steps.append(
            step_command(
                name="aggregate_results",
                command=aggregate_command,
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
        )
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
        steps.append(
            step_command(
                name="status_report",
                command=status_command,
                output_dir=output_dir,
                timeout_seconds=args.step_timeout_seconds,
            )
        )
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
        activation_gate_status["data_availability_status"] = (
            build_feature_activation_data_availability_status(
                activation_report=None,
                same_distance_history_provenance=None,
                inputs=activation_gate_inputs,
            )
        )
    elif not activation_gate_inputs.get("parity_report"):
        activation_gate_status = feature_activation_gate_status_for_skip(
            generated_at=generated_at,
            reason="train_eval_feature_parity_report_missing",
            inputs=activation_gate_inputs,
        )
        activation_gate_status["data_availability_status"] = (
            build_feature_activation_data_availability_status(
                activation_report=None,
                same_distance_history_provenance=None,
                inputs=activation_gate_inputs,
            )
        )
    else:
        protected_mid_unchanged = protected_hashes() == protected_before
        same_distance_history_provenance = load_json(
            activation_gate_inputs["same_distance_history_provenance"]
        ) if activation_gate_inputs.get("same_distance_history_provenance") else None
        provenance_audit = build_feature_activation_provenance_audit(
            prejump_metadata_report=load_json(daily_dir / "prejump_metadata_report.json"),
            same_distance_history_provenance=same_distance_history_provenance,
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
                timeout_seconds=args.step_timeout_seconds,
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
            "fail_reason_summary": activation_report.get("fail_reason_summary") or {},
            "data_availability_status": build_feature_activation_data_availability_status(
                activation_report=activation_report,
                same_distance_history_provenance=same_distance_history_provenance,
                inputs=activation_gate_inputs,
            ),
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
    live_odds_capture_packet_path = output_dir / "live_odds_capture_approval_packet.json"
    live_odds_capture_packet = build_live_odds_capture_approval_packet(
        generated_at=generated_at,
        daily_shadow_run_dir=daily_dir,
        upcoming_dir=refresh_dir,
        db_path=args.db,
        output_path=output_dir / "live_odds_capture_report.json",
        limit=args.refresh_limit,
    )
    live_odds_capture_packet["packet_path"] = relpath(live_odds_capture_packet_path)
    write_json(live_odds_capture_packet_path, live_odds_capture_packet)

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
        "timing_aligned_prediction_rerun_plan": relpath(
            output_dir / "timing_aligned_prediction_rerun_plan.json"
        ),
        "timing_aligned_prediction_rerun_execution_status": relpath(
            output_dir / "timing_aligned_prediction_rerun_execution_status.json"
        ),
        "timing_aligned_prediction_rerun_output_dir": (
            timing_aligned_rerun_execution_status.get("rerun_daily_shadow_run_dir")
        ),
        "timing_aligned_prediction_rerun_odds_snapshot": (
            timing_aligned_rerun_execution_status.get("rerun_odds_snapshot_dir")
        ),
        "autonomous_live_odds_capture": autonomous_odds_capture_status.get("output_dir"),
        "autonomous_live_odds_capture_status": relpath(
            output_dir / "autonomous_live_odds_capture_status.json"
        ),
        "autonomous_official_result_capture": autonomous_result_capture_status.get("output_dir"),
        "autonomous_official_result_capture_status": relpath(
            output_dir / "autonomous_official_result_capture_status.json"
        ),
        "unified_evidence_dataset": unified_dataset_status.get("output_dir"),
        "unified_evidence_dataset_status": relpath(
            output_dir / "unified_evidence_dataset_status.json"
        ),
        "backlog_unified_evidence_datasets_status": relpath(
            output_dir / "backlog_unified_evidence_datasets_status.json"
        ),
        "official_result_reserve_substitution_preflight": relpath(
            reserve_substitution_preflight_report_path
        )
        if reserve_substitution_preflight_report_path.exists()
        else None,
        "rolling_model_comparison": rolling_comparison_status.get("output_dir"),
        "rolling_model_comparison_status": relpath(
            output_dir / "rolling_model_comparison_status.json"
        ),
        "pre_race_gated_challenger": relpath(pre_race_gated_dir)
        if pre_race_gated_report_path.exists()
        else None,
        "promotion_distance_report": relpath(promotion_distance_report_path)
        if promotion_distance_report_path.exists()
        else None,
        "high_accuracy_refinement_packet": high_accuracy_status.get("output_dir"),
        "high_accuracy_refinement_status": relpath(
            output_dir / "high_accuracy_refinement_status.json"
        ),
        "feature_activation_gate": activation_gate_status.get("output_dir"),
        "feature_activation_gate_status": relpath(output_dir / "feature_activation_gate_status.json"),
        "live_odds_capture_approval_packet": relpath(live_odds_capture_packet_path),
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
        live_odds_capture_packet=live_odds_capture_packet,
        autonomous_live_odds_capture_status=autonomous_odds_capture_status,
        autonomous_official_result_capture_status=autonomous_result_capture_status,
        unified_evidence_dataset_status=unified_dataset_status,
        backlog_unified_evidence_status=backlog_unified_status,
        rolling_model_comparison_status=rolling_comparison_status,
        high_accuracy_refinement_status=high_accuracy_status,
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
        live_odds_capture_packet=live_odds_capture_packet,
        autonomous_live_odds_capture_status=autonomous_odds_capture_status,
        autonomous_official_result_capture_status=autonomous_result_capture_status,
        unified_evidence_dataset_status=unified_dataset_status,
        backlog_unified_evidence_status=backlog_unified_status,
        rolling_model_comparison_status=rolling_comparison_status,
        high_accuracy_refinement_status=high_accuracy_status,
    )
    daily_status.update(
        {
            "timing_aligned_prediction_rerun_plan_status": (
                timing_aligned_rerun_plan.get("status")
            ),
            "timing_aligned_prediction_rerun_plan_hard_stops": (
                timing_aligned_rerun_plan.get("hard_stops") or []
            ),
            "timing_aligned_prediction_rerun_execution_status": (
                timing_aligned_rerun_execution_status.get("status")
            ),
            "timing_aligned_prediction_rerun_execution_hard_stops": (
                timing_aligned_rerun_execution_status.get("hard_stops") or []
            ),
            "timing_aligned_prediction_rerun_execution_performed": (
                timing_aligned_rerun_execution_status.get("execution_performed")
                is True
            ),
            "timing_aligned_prediction_rerun_output_dir": (
                timing_aligned_rerun_execution_status.get("rerun_daily_shadow_run_dir")
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_dir": (
                timing_aligned_rerun_execution_status.get("rerun_odds_snapshot_dir")
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_status": (
                timing_aligned_rerun_execution_status.get("rerun_odds_snapshot_status")
            ),
            "timing_aligned_prediction_rerun_returncode": (
                timing_aligned_rerun_execution_status.get("returncode")
            ),
        }
    )

    protected_after = protected_hashes()
    protected_paths_unchanged = protected_before == protected_after
    protected_changed_paths = sorted(
        key
        for key, before_value in protected_before.items()
        if protected_after.get(key) != before_value
    )
    autonomous_odds_inserted_rows = int(
        autonomous_odds_capture_status.get("inserted_live_odds_rows") or 0
    )
    autonomous_official_result_evidence_inserted_rows = int(
        autonomous_result_capture_status.get("official_result_evidence_inserted_race_rows")
        or 0
    ) + int(
        autonomous_result_capture_status.get("official_result_evidence_inserted_runner_rows")
        or 0
    )
    allowed_odds_db_change = (
        bool(protected_changed_paths)
        and autonomous_odds_inserted_rows > 0
        and set(protected_changed_paths).issubset({relpath(args.db)})
    )
    allowed_official_result_evidence_db_change = (
        bool(protected_changed_paths)
        and autonomous_official_result_evidence_inserted_rows > 0
        and set(protected_changed_paths).issubset({relpath(args.db)})
    )
    protected_paths_unchanged_or_allowed = (
        protected_paths_unchanged
        or allowed_odds_db_change
        or allowed_official_result_evidence_db_change
    )

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
        "protected_changed_paths": protected_changed_paths,
        "allowed_odds_db_change": allowed_odds_db_change,
        "allowed_official_result_evidence_db_change": (
            allowed_official_result_evidence_db_change
        ),
        "protected_paths_unchanged_or_allowed": protected_paths_unchanged_or_allowed,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "required_output_files": [
            "SUMMARY.md",
            "shadow_dashboard.json",
            "result_join_status.json",
            "shadow_odds_snapshot_status.json",
            "timing_aligned_prediction_rerun_plan.json",
            "timing_aligned_prediction_rerun_execution_status.json",
            "autonomous_live_odds_capture_status.json",
            "autonomous_official_result_capture_status.json",
            "unified_evidence_dataset_status.json",
            "backlog_unified_evidence_datasets_status.json",
            "rolling_model_comparison_status.json",
            "high_accuracy_refinement_status.json",
            "feature_activation_gate_status.json",
            "live_odds_capture_approval_packet.json",
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
            **timing_aligned_rerun_manifest_phase_outputs(
                output_dir=output_dir,
                execution_status=timing_aligned_rerun_execution_status,
            ),
            "phase_2c_autonomous_live_odds_capture": relpath(
                output_dir / "autonomous_live_odds_capture_status.json"
            ),
            "phase_2d_autonomous_official_result_capture": relpath(
                output_dir / "autonomous_official_result_capture_status.json"
            ),
            "phase_2e_unified_evidence_dataset": relpath(
                output_dir / "unified_evidence_dataset_status.json"
            ),
            "phase_2f_backlog_unified_evidence_datasets": relpath(
                output_dir / "backlog_unified_evidence_datasets_status.json"
            ),
            "phase_2g_rolling_model_comparison": relpath(
                output_dir / "rolling_model_comparison_status.json"
            ),
            "phase_2g_pre_race_gated_challenger": relpath(
                pre_race_gated_report_path
            )
            if pre_race_gated_report_path.exists()
            else None,
            "phase_2g_promotion_distance_report": relpath(
                promotion_distance_report_path
            )
            if promotion_distance_report_path.exists()
            else None,
            "phase_2h_high_accuracy_refinement_packet": relpath(
                output_dir / "high_accuracy_refinement_status.json"
            ),
            "phase_2i_live_odds_capture_approval_packet": relpath(live_odds_capture_packet_path),
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
        output_dir / "timing_aligned_prediction_rerun_plan.json",
        output_dir / "timing_aligned_prediction_rerun_execution_status.json",
        output_dir / "autonomous_live_odds_capture_status.json",
        output_dir / "autonomous_official_result_capture_status.json",
        output_dir / "unified_evidence_dataset_status.json",
        output_dir / "backlog_unified_evidence_datasets_status.json",
        output_dir / "rolling_model_comparison_status.json",
        output_dir / "high_accuracy_refinement_status.json",
        output_dir / "feature_activation_gate_status.json",
        output_dir / "live_odds_capture_approval_packet.json",
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
        protected_paths_unchanged=protected_paths_unchanged_or_allowed,
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
                f"db_write={bool(autonomous_odds_inserted_rows or autonomous_official_result_evidence_inserted_rows)}",
                f"db_write_scope=append_only_live_odds_or_official_result_evidence_rows_if_true",
                f"label_write=False",
                f"tgr_enabled=False",
                f"betting_or_ev_action=False",
                f"shadow_odds_snapshot_status={odds_snapshot_status.get('status')}",
                f"shadow_odds_snapshot_ev_output_rows={odds_snapshot_status.get('ev_output_rows')}",
                f"timing_aligned_prediction_rerun_plan_status={timing_aligned_rerun_plan.get('status')}",
                f"timing_aligned_prediction_rerun_execution_status={timing_aligned_rerun_execution_status.get('status')}",
                f"timing_aligned_prediction_rerun_execution_hard_stops={timing_aligned_rerun_execution_status.get('hard_stops') or []}",
                f"timing_aligned_prediction_rerun_execution_performed={timing_aligned_rerun_execution_status.get('execution_performed')}",
                f"timing_aligned_prediction_rerun_output_dir={timing_aligned_rerun_execution_status.get('rerun_daily_shadow_run_dir')}",
                f"timing_aligned_prediction_rerun_odds_snapshot_dir={timing_aligned_rerun_execution_status.get('rerun_odds_snapshot_dir')}",
                f"timing_aligned_prediction_rerun_odds_snapshot_status={timing_aligned_rerun_execution_status.get('rerun_odds_snapshot_status')}",
                f"autonomous_live_odds_capture_status={autonomous_odds_capture_status.get('status')}",
                f"autonomous_live_odds_capture_ready_count={autonomous_odds_capture_status.get('ready_count')}",
                f"autonomous_live_odds_capture_inserted_rows={autonomous_odds_capture_status.get('inserted_live_odds_rows')}",
                f"autonomous_official_result_capture_status={autonomous_result_capture_status.get('status')}",
                f"autonomous_official_result_candidate_count={autonomous_result_capture_status.get('candidate_count')}",
                f"autonomous_official_result_race_rows={autonomous_result_capture_status.get('official_result_race_rows')}",
                f"autonomous_official_result_runner_rows={autonomous_result_capture_status.get('official_result_runner_rows')}",
                f"autonomous_official_result_quarantine_rows={autonomous_result_capture_status.get('quarantine_rows')}",
                f"autonomous_official_result_evidence_db_ingest_status={autonomous_result_capture_status.get('official_result_evidence_db_ingest_status')}",
                f"autonomous_official_result_evidence_inserted_race_rows={autonomous_result_capture_status.get('official_result_evidence_inserted_race_rows')}",
                f"autonomous_official_result_evidence_inserted_runner_rows={autonomous_result_capture_status.get('official_result_evidence_inserted_runner_rows')}",
                f"unified_evidence_dataset_status={unified_dataset_status.get('status')}",
                f"unified_evidence_dataset_rows={unified_dataset_status.get('row_count')}",
                f"unified_evidence_dataset_races={unified_dataset_status.get('race_count')}",
                f"unified_label_evaluation_eligible_rows={unified_dataset_status.get('label_evaluation_eligible_rows')}",
                f"unified_stage2_evaluation_eligible_rows={unified_dataset_status.get('stage2_evaluation_eligible_rows')}",
                f"unified_odds_evaluation_eligible_rows={unified_dataset_status.get('odds_evaluation_eligible_rows')}",
                f"unified_evidence_eligible_rows={unified_dataset_status.get('unified_evidence_eligible_rows')}",
                f"unified_evidence_official_result_coverage_requested_race_count={unified_dataset_status.get('official_result_coverage_requested_race_count')}",
                f"unified_evidence_official_result_coverage_requested_race_count_source={unified_dataset_status.get('official_result_coverage_requested_race_count_source') or (unified_dataset_status.get('official_result_coverage') or {}).get('requested_race_count_source')}",
                f"unified_evidence_official_result_coverage_races_with_rows_count={unified_dataset_status.get('official_result_coverage_races_with_rows_count')}",
                f"unified_evidence_official_result_coverage_missing_race_count={unified_dataset_status.get('official_result_coverage_missing_race_count')}",
                f"unified_evidence_official_result_coverage_missing_exclusion_count={unified_dataset_status.get('official_result_coverage_missing_exclusion_count')}",
                f"unified_evidence_official_result_runner_path_count={unified_dataset_status.get('official_result_runner_path_count')}",
                f"unified_evidence_official_result_runner_paths_source_field={unified_dataset_status.get('official_result_runner_paths_source_field')}",
                f"backlog_unified_evidence_status={backlog_unified_status.get('status')}",
                f"backlog_unified_evidence_dataset_count={backlog_unified_status.get('dataset_count')}",
                f"backlog_unified_evidence_failed_dataset_count={backlog_unified_status.get('failed_dataset_count')}",
                f"backlog_unified_evidence_rows={backlog_unified_status.get('row_count')}",
                f"backlog_unified_evidence_official_result_rows={backlog_unified_status.get('rows_with_official_results')}",
                f"backlog_unified_evidence_strict_odds_rows={backlog_unified_status.get('rows_with_strict_prejump_odds')}",
                f"backlog_unified_evidence_eligible_rows={backlog_unified_status.get('unified_evidence_eligible_rows')}",
                f"rolling_model_comparison_status={rolling_comparison_status.get('status')}",
                f"rolling_model_comparison_sample_races={rolling_comparison_status.get('sample_race_count')}",
                f"rolling_model_comparison_sample_runner_rows={rolling_comparison_status.get('sample_runner_rows')}",
                f"rolling_model_comparison_best_candidate={rolling_comparison_status.get('best_candidate_key')}",
                f"rolling_model_comparison_best_top1={rolling_comparison_status.get('best_candidate_top1')}",
                f"rolling_model_comparison_best_top3={rolling_comparison_status.get('best_candidate_top3')}",
                f"rolling_model_comparison_source_rejected_live_odds_candidate_count={rolling_comparison_status.get('source_rejected_live_odds_candidate_count')}",
                f"rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts={rolling_comparison_status.get('source_rejected_live_odds_candidate_reason_counts')}",
                f"high_accuracy_refinement_status={high_accuracy_status.get('status')}",
                f"high_accuracy_promotion_pr_gate_status={high_accuracy_status.get('promotion_pr_gate_status')}",
                *high_accuracy_timing_source_verification_lines(high_accuracy_status),
                f"reserve_substitution_preflight_status={high_accuracy_status.get('reserve_substitution_preflight_status')}",
                f"reserve_substitution_preflight_ready_for_policy_review_count={high_accuracy_status.get('reserve_substitution_preflight_ready_for_policy_review_count')}",
                f"reserve_substitution_preflight_dataset_join_blocker_counts={high_accuracy_status.get('reserve_substitution_preflight_dataset_join_blocker_counts')}",
                f"reserve_substitution_preflight_ready_race_ids={high_accuracy_status.get('reserve_substitution_preflight_ready_race_ids')}",
                f"reserve_substitution_manual_review_status={high_accuracy_status.get('reserve_substitution_manual_review_status')}",
                f"reserve_substitution_manual_review_ready_candidate_count={high_accuracy_status.get('reserve_substitution_manual_review_ready_candidate_count')}",
                f"reserve_substitution_manual_review_mapping_pair_count={high_accuracy_status.get('reserve_substitution_manual_review_mapping_pair_count')}",
                f"reserve_substitution_manual_review_dataset_join_allowed={high_accuracy_status.get('reserve_substitution_manual_review_dataset_join_allowed')}",
                f"reserve_substitution_manual_review_official_result_acceptance_allowed={high_accuracy_status.get('reserve_substitution_manual_review_official_result_acceptance_allowed')}",
                f"reserve_substitution_manual_review_db_write={high_accuracy_status.get('reserve_substitution_manual_review_db_write')}",
                f"reserve_substitution_policy_impact_status={high_accuracy_status.get('reserve_substitution_policy_impact_status')}",
                f"reserve_substitution_policy_impact_ready_candidate_count={high_accuracy_status.get('reserve_substitution_policy_impact_ready_candidate_count')}",
                f"reserve_substitution_policy_impact_mapping_pair_count={high_accuracy_status.get('reserve_substitution_policy_impact_mapping_pair_count')}",
                f"reserve_substitution_policy_impact_potential_runner_rows_blocked={high_accuracy_status.get('reserve_substitution_policy_impact_potential_runner_rows_blocked')}",
                f"reserve_substitution_policy_impact_dataset_join_allowed={high_accuracy_status.get('reserve_substitution_policy_impact_dataset_join_allowed')}",
                f"reserve_substitution_policy_impact_official_result_acceptance_allowed={high_accuracy_status.get('reserve_substitution_policy_impact_official_result_acceptance_allowed')}",
                f"reserve_substitution_policy_impact_db_write={high_accuracy_status.get('reserve_substitution_policy_impact_db_write')}",
                f"promotion_distance_status={high_accuracy_status.get('promotion_distance_status')}",
                f"promotion_distance_promotion_ready={high_accuracy_status.get('promotion_distance_promotion_ready')}",
                f"promotion_distance_sample_races={high_accuracy_status.get('promotion_distance_sample_race_count')}",
                f"promotion_distance_sample_runner_rows={high_accuracy_status.get('promotion_distance_sample_runner_rows')}",
                f"promotion_distance_source_rejected_live_odds_candidate_count={high_accuracy_status.get('promotion_distance_source_rejected_live_odds_candidate_count')}",
                f"promotion_distance_source_rejected_live_odds_candidate_reason_counts={high_accuracy_status.get('promotion_distance_source_rejected_live_odds_candidate_reason_counts')}",
                f"promotion_distance_source_exclusion_reason_counts={high_accuracy_status.get('promotion_distance_source_exclusion_reason_counts')}",
                f"promotion_distance_source_odds_exclusion_reason_counts={high_accuracy_status.get('promotion_distance_source_odds_exclusion_reason_counts')}",
                f"promotion_distance_source_official_result_evidence_db_missing_race_ids={high_accuracy_status.get('promotion_distance_source_official_result_evidence_db_missing_race_ids')}",
                f"promotion_distance_official_result_coverage_requested_race_count={high_accuracy_status.get('promotion_distance_official_result_coverage_requested_race_count')}",
                f"promotion_distance_official_result_coverage_requested_race_count_source={high_accuracy_status.get('promotion_distance_official_result_coverage_requested_race_count_source')}",
                f"promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids={high_accuracy_status.get('promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids')}",
                f"promotion_distance_official_result_coverage_races_with_rows_count={high_accuracy_status.get('promotion_distance_official_result_coverage_races_with_rows_count')}",
                f"promotion_distance_official_result_coverage_missing_race_count={high_accuracy_status.get('promotion_distance_official_result_coverage_missing_race_count')}",
                f"promotion_distance_official_result_coverage_missing_exclusion_count={high_accuracy_status.get('promotion_distance_official_result_coverage_missing_exclusion_count')}",
                f"promotion_distance_official_result_runner_path_count={high_accuracy_status.get('promotion_distance_official_result_runner_path_count')}",
                f"promotion_distance_official_result_runner_paths_source_field={high_accuracy_status.get('promotion_distance_official_result_runner_paths_source_field')}",
                f"promotion_distance_best_candidate={high_accuracy_status.get('promotion_distance_best_candidate_key')}",
                f"promotion_distance_best_non_market_candidate={high_accuracy_status.get('promotion_distance_best_non_market_candidate_key')}",
                f"promotion_distance_best_non_market_top1_margin_gap={high_accuracy_status.get('promotion_distance_best_non_market_top1_margin_gap')}",
                f"promotion_distance_predeclared_residual_status={high_accuracy_status.get('promotion_distance_predeclared_residual_candidate_status')}",
                f"promotion_distance_predeclared_residual_triggered_races={high_accuracy_status.get('promotion_distance_predeclared_residual_triggered_race_count')}",
                f"promotion_distance_blockers={high_accuracy_status.get('promotion_distance_blockers')}",
                f"feature_activation_gate_status={activation_gate_status.get('status')}",
                f"live_odds_capture_approval_status={live_odds_capture_packet.get('status')}",
                f"live_odds_capture_verified_prejump_races={live_odds_capture_packet.get('verified_prejump_race_count')}",
                f"live_odds_capture_db_write={live_odds_capture_packet.get('no_write_guarantees', {}).get('db_write')}",
                f"next_prejump_refresh_status={dashboard.get('next_prejump_refresh_status')}",
                f"recommended_rerun_after_local={dashboard.get('recommended_rerun_after_local')}",
                f"protected_paths_unchanged={protected_paths_unchanged}",
                f"protected_paths_unchanged_or_allowed={protected_paths_unchanged_or_allowed}",
                f"allowed_official_result_evidence_db_change={allowed_official_result_evidence_db_change}",
                f"protected_changed_paths={protected_changed_paths}",
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
            live_odds_capture_packet=live_odds_capture_packet,
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
        "live_odds_capture_approval_status": live_odds_capture_packet.get("status"),
        "live_odds_capture_verified_prejump_race_count": live_odds_capture_packet.get(
            "verified_prejump_race_count"
        ),
        "autonomous_live_odds_capture_status": autonomous_odds_capture_status.get("status"),
        "autonomous_live_odds_capture_ready_count": autonomous_odds_capture_status.get(
            "ready_count"
        ),
        "autonomous_live_odds_capture_inserted_rows": autonomous_odds_capture_status.get(
            "inserted_live_odds_rows"
        ),
        "autonomous_official_result_capture_status": autonomous_result_capture_status.get("status"),
        "autonomous_official_result_race_rows": autonomous_result_capture_status.get(
            "official_result_race_rows"
        ),
        "autonomous_official_result_runner_rows": autonomous_result_capture_status.get(
            "official_result_runner_rows"
        ),
        "unified_evidence_dataset_status": unified_dataset_status.get("status"),
        "unified_evidence_dataset_rows": unified_dataset_status.get("row_count"),
        "unified_label_evaluation_eligible_rows": unified_dataset_status.get(
            "label_evaluation_eligible_rows"
        ),
        "unified_stage2_evaluation_eligible_rows": unified_dataset_status.get(
            "stage2_evaluation_eligible_rows"
        ),
        "unified_odds_evaluation_eligible_rows": unified_dataset_status.get(
            "odds_evaluation_eligible_rows"
        ),
        "unified_evidence_eligible_rows": unified_dataset_status.get(
            "unified_evidence_eligible_rows"
        ),
        "backlog_unified_evidence_status": backlog_unified_status.get("status"),
        "backlog_unified_evidence_dataset_count": backlog_unified_status.get(
            "dataset_count"
        ),
        "backlog_unified_evidence_failed_dataset_count": backlog_unified_status.get(
            "failed_dataset_count"
        ),
        "backlog_unified_evidence_eligible_rows": backlog_unified_status.get(
            "unified_evidence_eligible_rows"
        ),
        "rolling_model_comparison_status": rolling_comparison_status.get("status"),
        "rolling_model_comparison_sample_races": rolling_comparison_status.get(
            "sample_race_count"
        ),
        "rolling_model_comparison_best_candidate": rolling_comparison_status.get(
            "best_candidate_key"
        ),
        "promotion_distance_status": high_accuracy_status.get(
            "promotion_distance_status"
        ),
        "promotion_distance_promotion_ready": high_accuracy_status.get(
            "promotion_distance_promotion_ready"
        ),
        "promotion_distance_sample_races": high_accuracy_status.get(
            "promotion_distance_sample_race_count"
        ),
        "promotion_distance_sample_runner_rows": high_accuracy_status.get(
            "promotion_distance_sample_runner_rows"
        ),
        "promotion_distance_best_candidate": high_accuracy_status.get(
            "promotion_distance_best_candidate_key"
        ),
        "promotion_distance_best_non_market_candidate": high_accuracy_status.get(
            "promotion_distance_best_non_market_candidate_key"
        ),
        "promotion_distance_blockers": high_accuracy_status.get(
            "promotion_distance_blockers"
        ),
        "reserve_substitution_preflight_status": high_accuracy_status.get(
            "reserve_substitution_preflight_status"
        ),
        "reserve_substitution_preflight_ready_for_policy_review_count": (
            high_accuracy_status.get(
                "reserve_substitution_preflight_ready_for_policy_review_count"
            )
        ),
        "reserve_substitution_preflight_dataset_join_blocker_counts": (
            high_accuracy_status.get(
                "reserve_substitution_preflight_dataset_join_blocker_counts"
            )
        ),
        "reserve_substitution_preflight_ready_race_ids": high_accuracy_status.get(
            "reserve_substitution_preflight_ready_race_ids"
        ),
        "reserve_substitution_manual_review_status": high_accuracy_status.get(
            "reserve_substitution_manual_review_status"
        ),
        "reserve_substitution_manual_review_ready_candidate_count": (
            high_accuracy_status.get(
                "reserve_substitution_manual_review_ready_candidate_count"
            )
        ),
        "reserve_substitution_manual_review_mapping_pair_count": (
            high_accuracy_status.get(
                "reserve_substitution_manual_review_mapping_pair_count"
            )
        ),
        "reserve_substitution_manual_review_dataset_join_allowed": (
            high_accuracy_status.get(
                "reserve_substitution_manual_review_dataset_join_allowed"
            )
        ),
        "reserve_substitution_manual_review_official_result_acceptance_allowed": (
            high_accuracy_status.get(
                "reserve_substitution_manual_review_official_result_acceptance_allowed"
            )
        ),
        "reserve_substitution_manual_review_db_write": high_accuracy_status.get(
            "reserve_substitution_manual_review_db_write"
        ),
        "reserve_substitution_manual_review_blockers": high_accuracy_status.get(
            "reserve_substitution_manual_review_blockers"
        ),
        "reserve_substitution_policy_impact_status": high_accuracy_status.get(
            "reserve_substitution_policy_impact_status"
        ),
        "reserve_substitution_policy_impact_ready_candidate_count": (
            high_accuracy_status.get(
                "reserve_substitution_policy_impact_ready_candidate_count"
            )
        ),
        "reserve_substitution_policy_impact_mapping_pair_count": (
            high_accuracy_status.get(
                "reserve_substitution_policy_impact_mapping_pair_count"
            )
        ),
        "reserve_substitution_policy_impact_potential_runner_rows_blocked": (
            high_accuracy_status.get(
                "reserve_substitution_policy_impact_potential_runner_rows_blocked"
            )
        ),
        "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count": (
            high_accuracy_status.get(
                "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count"
            )
        ),
        "reserve_substitution_policy_impact_dataset_join_allowed": (
            high_accuracy_status.get(
                "reserve_substitution_policy_impact_dataset_join_allowed"
            )
        ),
        "reserve_substitution_policy_impact_official_result_acceptance_allowed": (
            high_accuracy_status.get(
                "reserve_substitution_policy_impact_official_result_acceptance_allowed"
            )
        ),
        "reserve_substitution_policy_impact_db_write": high_accuracy_status.get(
            "reserve_substitution_policy_impact_db_write"
        ),
        "reserve_substitution_policy_impact_blockers": high_accuracy_status.get(
            "reserve_substitution_policy_impact_blockers"
        ),
        "shadow_odds_snapshot_status": odds_snapshot_status.get("status"),
        "shadow_odds_snapshot_valid_prejump_rows": odds_snapshot_status.get(
            "valid_pre_jump_dog_odds_rows"
        ),
        "next_prejump_refresh_status": dashboard.get("next_prejump_refresh_status"),
        "recommended_rerun_after_local": dashboard.get("recommended_rerun_after_local"),
        "protected_paths_unchanged": protected_paths_unchanged,
        "protected_paths_unchanged_or_allowed": protected_paths_unchanged_or_allowed,
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
    parser.add_argument(
        "--odds-capture-min-minutes",
        type=float,
        default=DEFAULT_ODDS_CAPTURE_MIN_MINUTES,
    )
    parser.add_argument(
        "--odds-capture-max-minutes",
        type=float,
        default=DEFAULT_ODDS_CAPTURE_MAX_MINUTES,
    )
    parser.add_argument(
        "--odds-capture-refresh-limit",
        type=int,
        default=DEFAULT_ODDS_CAPTURE_REFRESH_LIMIT,
    )
    parser.add_argument(
        "--autonomous-odds-capture-limit",
        type=int,
        default=DEFAULT_AUTONOMOUS_ODDS_CAPTURE_LIMIT,
        help=(
            "Maximum races to execute in the autonomous live-odds capture step. "
            "Defaults to the odds-capture refresh limit."
        ),
    )
    parser.add_argument("--refresh-limit", type=int, default=16)
    parser.add_argument("--refresh-dry-run", action="store_true")
    parser.add_argument("--refresh-command-mode", choices=("auto", "python", "uv"), default="auto")
    parser.add_argument("--score-command-mode", choices=("auto", "python", "uv"), default="auto")
    parser.add_argument("--target-joined-races", type=int, default=DEFAULT_TARGET_JOINED_RACES)
    parser.add_argument("--min-joined-races", type=int, default=DEFAULT_MIN_JOINED_RACES_FOR_STATUS)
    parser.add_argument("--step-timeout-seconds", type=int, default=840)
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--skip-primary-refresh", action="store_true")
    parser.add_argument("--skip-shadow-run", action="store_true")
    parser.add_argument("--skip-odds-snapshot", action="store_true")
    parser.add_argument("--skip-result-join", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-status", action="store_true")
    parser.add_argument("--skip-unified-dataset", action="store_true")
    parser.add_argument("--shadow-run-dir", type=Path)
    parser.add_argument("--enable-autonomous-odds-capture", action="store_true")
    parser.add_argument("--execute-autonomous-odds-capture", action="store_true")
    parser.add_argument("--allow-auto-scrape-odds", action="store_true")
    parser.add_argument("--enable-autonomous-result-capture", action="store_true")
    parser.add_argument("--result-backlog-limit", type=int, default=DEFAULT_RESULT_BACKLOG_LIMIT)
    parser.add_argument(
        "--result-backlog-shadow-run-limit",
        type=int,
        default=DEFAULT_RESULT_BACKLOG_SHADOW_RUN_LIMIT,
    )
    parser.add_argument(
        "--result-backlog-lookback-days",
        type=int,
        default=DEFAULT_RESULT_BACKLOG_LOOKBACK_DAYS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_autopilot(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("final_verdict") != "NEEDS_MORE_TOOLING" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
