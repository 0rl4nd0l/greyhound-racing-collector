#!/usr/bin/env python3
"""Build a rolling report-only status for forward shadow reliability.

The report combines result-join metrics, sidecar metadata gates, live/training
coverage, and activation-gate decisions into one evidence packet. It does not
write predictions, labels, DB rows, registry entries, EV, betting output, or
production model pointers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from accuracy_program.odds_coverage import (
    summarize_read_only_odds_coverage_report,
)
from utils.report_output_dir_guard import assert_prefixed_report_output_dir  # noqa: E402

DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/forward_shadow_status_"
OUTPUT_ARTIFACT_PREFIX = "forward_shadow_status_"
DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
EXPECTED_OFFICIAL_RACES = 214
EXPECTED_OFFICIAL_DOG_ROWS = 1493
DEFAULT_MIN_JOINED_RACES = 20

STATUS_DB_BLOCKED = "BLOCKED_DB_STATE"
STATUS_COLLECT_MORE = "CONTINUE_FORWARD_SHADOW_COLLECTION"
STATUS_READY_REVIEW = "READY_FOR_FORWARD_SHADOW_REVIEW_REPORT_ONLY"
STATUS_REVIEW_KEEP_QUARANTINED = "FORWARD_REVIEW_READY_KEEP_QUARANTINED"
ARTIFACT_TIMESTAMP_RE = re.compile(r"20\d{6}T\d{6}[+-]\d{4}")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(paths: Sequence[Path] = DEFAULT_PROTECTED_PATHS) -> dict[str, str | None]:
    return {relpath(path) or str(path): sha256_file(path) for path in paths}


def db_state(db_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "db_path": relpath(db_path),
        "status": "FAIL",
        "fail_reasons": [],
    }
    if not db_path.exists():
        report["fail_reasons"].append("db_missing")
        return report
    try:
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        try:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            official_races = connection.execute(
                "SELECT count(DISTINCT race_id) FROM race_metadata "
                "WHERE winner_source='thedogs_official'"
            ).fetchone()[0]
            official_dog_rows = connection.execute(
                "SELECT count(*) FROM dog_race_data WHERE data_source='thedogs_official'"
            ).fetchone()[0]
        finally:
            connection.close()
    except Exception as exc:  # pragma: no cover - defensive artifact reporting
        report["fail_reasons"].append(f"db_read_failed:{type(exc).__name__}")
        return report
    report.update(
        {
            "quick_check": quick_check,
            "official_races": int(official_races),
            "official_dog_rows": int(official_dog_rows),
        }
    )
    if quick_check != "ok":
        report["fail_reasons"].append("quick_check_not_ok")
    if official_races != EXPECTED_OFFICIAL_RACES:
        report["fail_reasons"].append("official_race_count_mismatch")
    if official_dog_rows != EXPECTED_OFFICIAL_DOG_ROWS:
        report["fail_reasons"].append("official_dog_row_count_mismatch")
    if not report["fail_reasons"]:
        report["status"] = "PASS"
    return report


def latest_artifact(root: Path, prefix: str, required_file: str) -> Path | None:
    candidates = [
        item
        for item in root.glob(f"{prefix}*")
        if item.is_dir() and (item / required_file).exists()
    ]
    return sorted(candidates)[-1] if candidates else None


def artifact_timestamp(path: Path | None) -> str:
    if path is None:
        return ""
    for part in reversed(path.parts):
        match = ARTIFACT_TIMESTAMP_RE.search(part)
        if match:
            return match.group(0)
    return ""


def latest_artifact_file(root: Path, prefixes: Sequence[str], filename: str) -> Path | None:
    candidates: list[Path] = []
    for prefix in prefixes:
        candidates.extend(
            item / filename
            for item in root.glob(f"{prefix}*")
            if item.is_dir() and (item / filename).exists()
        )
    return sorted(candidates, key=lambda path: (artifact_timestamp(path.parent), str(path)))[-1] if candidates else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "safe_joined_race_count": int(metrics.get("safe_joined_race_count") or 0),
        "pending_race_count": int(metrics.get("pending_race_count") or 0),
        "unsafe_match_count": int(metrics.get("unsafe_match_count") or 0),
        "top1": metrics.get("top1"),
        "top3": metrics.get("top3"),
        "mean_winner_rank": metrics.get("mean_winner_rank"),
        "brier": metrics.get("brier"),
        "logloss": metrics.get("logloss"),
        "probability_sum_max_error_joined_races": metrics.get(
            "probability_sum_max_error_joined_races"
        ),
        "winner_ranks": metrics.get("winner_ranks") or [],
    }


def activation_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    return {
        "final_status": report.get("final_status"),
        "activation_allowed_features": report.get("activation_allowed_features") or [],
        "kept_quarantined_features": report.get("kept_quarantined_features") or [],
    }


def odds_snapshot_summary(
    report: Mapping[str, Any] | None,
    readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = report or {}
    readiness = readiness or report.get("odds_research_readiness") or {}
    if not isinstance(readiness, Mapping):
        readiness = {}
    blocker_counts = readiness.get("blocker_counts")
    if not isinstance(blocker_counts, Mapping):
        blocker_counts = report.get("odds_analysis_blocker_counts")
    skipped_reason = report.get("skipped_reason")
    if not isinstance(blocker_counts, Mapping) or (not blocker_counts and skipped_reason):
        blocker_counts = {str(skipped_reason): 1} if skipped_reason else {}
    odds_analysis_status = (
        readiness.get("status")
        or report.get("odds_analysis_status")
        or ("ODDS_ANALYSIS_BLOCKED" if blocker_counts else None)
    )
    odds_next_action = readiness.get("odds_research_next_action")
    if not odds_next_action and skipped_reason == "no_shadow_predictions":
        odds_next_action = "WAIT_FOR_SHADOW_PREDICTIONS"
    approved_predictions = report.get("approved_odds_augmented_predictions")
    if not isinstance(approved_predictions, Mapping):
        approved_predictions = {}
    return {
        "final_status": report.get("final_status") or report.get("status"),
        "collection_attempted": report.get("collection_attempted"),
        "skipped_reason": report.get("skipped_reason"),
        "prediction_rows": int(report.get("prediction_rows") or 0),
        "race_count": int(report.get("race_count") or 0),
        "odds_candidate_rows": int(report.get("odds_candidate_rows") or 0),
        "valid_pre_jump_dog_odds_rows": int(
            report.get("valid_pre_jump_dog_odds_rows") or 0
        ),
        "races_with_complete_valid_prejump_odds": int(
            report.get("races_with_complete_valid_prejump_odds") or 0
        ),
        "races_with_missing_odds_rows": int(
            report.get("races_with_missing_odds_rows") or 0
        ),
        "races_with_stale_odds_rows": int(
            report.get("races_with_stale_odds_rows") or 0
        ),
        "races_with_post_feature_freeze_odds_rows": int(
            report.get("races_with_post_feature_freeze_odds_rows") or 0
        ),
        "races_with_post_jump_odds_rows": int(
            report.get("races_with_post_jump_odds_rows") or 0
        ),
        "odds_analysis_status": odds_analysis_status,
        "odds_analysis_blocker_counts": dict(blocker_counts),
        "odds_research_next_action": odds_next_action,
        "ev_output_rows": int(report.get("ev_output_rows") or 0),
        "ev_calculation_status": readiness.get(
            "ev_calculation_status",
            "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
        ),
        "odds_used_for_shadow_scoring": bool(readiness.get("odds_used_for_shadow_scoring")),
        "approved_odds_augmented_candidate_key": approved_predictions.get(
            "candidate_key"
        ),
        "approved_odds_augmented_prediction_status": approved_predictions.get("status"),
        "approved_odds_augmented_ready_race_count": int(
            approved_predictions.get("ready_race_count") or 0
        ),
        "approved_odds_augmented_blocked_race_count": int(
            approved_predictions.get("blocked_race_count") or 0
        ),
        "approved_odds_augmented_prediction_rows": int(
            approved_predictions.get("prediction_rows") or 0
        ),
        "approved_odds_augmented_prediction_report_path": report.get(
            "approved_odds_augmented_prediction_report_path"
        ),
    }


def latest_odds_coverage_evidence(evidence_root: Path) -> dict[str, Any]:
    report_path = latest_artifact_file(
        evidence_root,
        ("shadow_autopilot_daemonization_v1_",),
        "odds_coverage_report.json",
    )
    return {
        "report_path": report_path,
        "report": load_json(report_path),
    }


def daemon_runtime_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    next_race = report.get("last_next_prejump_race")
    if not isinstance(next_race, Mapping):
        next_race = None
    return {
        "last_verdict": report.get("last_verdict"),
        "last_cycle_activity_status": report.get("last_cycle_activity_status"),
        "last_next_prejump_refresh_status": report.get(
            "last_next_prejump_refresh_status"
        ),
        "last_next_prejump_race": dict(next_race) if next_race else None,
        "last_recommended_rerun_after_local": report.get(
            "last_recommended_rerun_after_local"
        ),
        "last_safe_joined_delta": report.get("last_safe_joined_delta"),
        "last_safe_joined_races": report.get("last_safe_joined_races"),
        "last_prejump_metadata_status": report.get("last_prejump_metadata_status"),
        "last_shadow_odds_snapshot_status": report.get("last_shadow_odds_snapshot_status"),
        "last_shadow_odds_snapshot_ev_output_rows": report.get(
            "last_shadow_odds_snapshot_ev_output_rows"
        ),
        "last_systemd_deployment_status": report.get(
            "last_systemd_deployment_status"
        ),
        "updated_at": report.get("updated_at"),
    }


def latest_odds_snapshot_evidence(evidence_root: Path) -> dict[str, Any]:
    standalone_dir = latest_artifact(
        evidence_root,
        "shadow_odds_snapshot_",
        "shadow_odds_snapshot_report.json",
    )
    status_path = latest_artifact_file(
        evidence_root,
        ("shadow_autopilot_daemonization_v1_", "shadow_autopilot_v1_"),
        "shadow_odds_snapshot_status.json",
    )
    standalone_timestamp = artifact_timestamp(standalone_dir)
    status_timestamp = artifact_timestamp(status_path.parent if status_path else None)

    use_status = bool(status_path) and (
        not standalone_dir or status_timestamp >= standalone_timestamp
    )
    if use_status and status_path is not None:
        return {
            "source_kind": "autopilot_or_daemon_status",
            "snapshot_dir": None,
            "status_path": status_path,
            "report": load_json(status_path),
            "readiness": None,
        }
    return {
        "source_kind": "standalone_snapshot_report" if standalone_dir else None,
        "snapshot_dir": standalone_dir,
        "status_path": None,
        "report": load_json(standalone_dir / "shadow_odds_snapshot_report.json")
        if standalone_dir
        else None,
        "readiness": load_json(standalone_dir / "shadow_odds_research_readiness.json")
        if standalone_dir
        else None,
    }


def sidecar_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    classification = report.get("classification") or {}
    return {
        "final_status": report.get("final_status"),
        "eligible_count": classification.get("eligible_count"),
        "malformed_count": classification.get("malformed_count"),
        "stale_count": classification.get("stale_count"),
        "prejump_sidecar_metadata_required": classification.get(
            "prejump_sidecar_metadata_required"
        ),
    }


def artifact_final_status(directory: Path | None) -> str | None:
    if directory is None:
        return None
    path = directory / "final_status.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def coverage_summary(
    report: Mapping[str, Any] | None,
    selected_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = report or {}
    latest_metrics = (
        dict(selected_metrics)
        if selected_metrics is not None
        else report.get("latest_forward_metrics_summary") or {}
    )
    return {
        "final_status": report.get("final_status"),
        "blocked_reasons": report.get("blocked_reasons") or [],
        "latest_forward_metrics_summary": latest_metrics,
        "training_feature_coverage": report.get("training_feature_coverage") or {},
        "live_sidecar_feature_coverage": report.get("live_sidecar_feature_coverage") or {},
    }


def decide_status(
    *,
    db_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    activation: Mapping[str, Any],
    min_joined_races: int,
) -> tuple[str, list[str]]:
    if db_report.get("status") != "PASS":
        return STATUS_DB_BLOCKED, ["db_state_not_pass"]

    collection_reasons: list[str] = []
    if int(metrics.get("safe_joined_race_count") or 0) < min_joined_races:
        collection_reasons.append("safe_joined_race_count_below_review_min")
    if int(metrics.get("pending_race_count") or 0) > 0:
        collection_reasons.append("pending_official_results_remain")
    if int(metrics.get("unsafe_match_count") or 0) > 0:
        collection_reasons.append("unsafe_identity_matches_present")
    probability_error = metrics.get("probability_sum_max_error_joined_races")
    if probability_error is None or float(probability_error) > 1e-6:
        collection_reasons.append("probability_sum_error_not_pass")

    kept_quarantined = activation.get("kept_quarantined_features") or []
    if kept_quarantined:
        collection_reasons.append("features_remain_quarantined")
    if collection_reasons:
        only_quarantine_blocks = (
            set(collection_reasons) == {"features_remain_quarantined"}
            and int(metrics.get("safe_joined_race_count") or 0) >= min_joined_races
        )
        if only_quarantine_blocks:
            return STATUS_REVIEW_KEEP_QUARANTINED, collection_reasons
        return STATUS_COLLECT_MORE, collection_reasons
    return STATUS_READY_REVIEW, []


def build_status_report(
    *,
    evidence_root: Path,
    db_path: Path,
    aggregate_result_dir: Path | None = None,
    result_join_dir: Path | None = None,
    sidecar_gate_dir: Path | None = None,
    live_feature_audit_dir: Path | None = None,
    activation_gate_dir: Path | None = None,
    coverage_gap_dir: Path | None = None,
    min_joined_races: int = DEFAULT_MIN_JOINED_RACES,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    aggregate_result_dir = aggregate_result_dir or latest_artifact(
        evidence_root, "forward_shadow_result_aggregate_", "aggregate_forward_metrics.json"
    )
    result_join_dir = result_join_dir or latest_artifact(
        evidence_root, "forward_shadow_result_join_", "shadow_forward_metrics.json"
    )
    sidecar_gate_dir = sidecar_gate_dir or latest_artifact(
        evidence_root, "prejump_sidecar_gate_audit_", "prejump_sidecar_gate_audit.json"
    )
    live_feature_audit_dir = live_feature_audit_dir or latest_artifact(
        evidence_root,
        "sidecar_target_metadata_live_feature_audit_",
        "sidecar_target_metadata_live_feature_audit.json",
    )
    activation_gate_dir = activation_gate_dir or latest_artifact(
        evidence_root, "shadow_feature_activation_gate_", "feature_activation_gate_report.json"
    )
    odds_evidence = latest_odds_snapshot_evidence(evidence_root)
    odds_snapshot_dir = odds_evidence.get("snapshot_dir")
    odds_status_path = odds_evidence.get("status_path")
    odds_coverage_evidence = latest_odds_coverage_evidence(evidence_root)
    odds_coverage_path = odds_coverage_evidence.get("report_path")
    daemon_runtime_state_path = evidence_root / "shadow_autopilot_daemon_runtime/state.json"
    coverage_gap_dir = coverage_gap_dir or latest_artifact(
        evidence_root,
        "train_live_feature_coverage_gap_audit_",
        "train_live_feature_coverage_gap_audit.json",
    )

    db_report = db_state(db_path)
    if aggregate_result_dir is not None:
        metrics = metric_summary(load_json(aggregate_result_dir / "aggregate_forward_metrics.json"))
        result_metric_source = "aggregate_forward_metrics"
    else:
        metrics = metric_summary(
            load_json(result_join_dir / "shadow_forward_metrics.json") if result_join_dir else None
        )
        result_metric_source = "latest_single_result_join"
    activation = activation_summary(
        load_json(activation_gate_dir / "feature_activation_gate_report.json")
        if activation_gate_dir
        else None
    )
    sidecar_gate = sidecar_summary(
        load_json(sidecar_gate_dir / "prejump_sidecar_gate_audit.json")
        if sidecar_gate_dir
        else None
    )
    if not sidecar_gate.get("final_status"):
        sidecar_gate["final_status"] = artifact_final_status(sidecar_gate_dir)
    odds_report = odds_evidence.get("report")
    odds_readiness = odds_evidence.get("readiness")
    odds_snapshot = odds_snapshot_summary(odds_report, odds_readiness)
    odds_coverage = summarize_read_only_odds_coverage_report(
        odds_coverage_evidence.get("report")
    )
    daemon_runtime = daemon_runtime_summary(load_json(daemon_runtime_state_path))
    final_status, reasons = decide_status(
        db_report=db_report,
        metrics=metrics,
        activation=activation,
        min_joined_races=min_joined_races,
    )
    return {
        "schema_version": "forward_shadow_status_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "status_reasons": reasons,
        "min_joined_races_for_review": min_joined_races,
        "db_state": db_report,
        "forward_metrics": metrics,
        "activation_gate": activation,
        "shadow_odds_snapshot": odds_snapshot,
        "shadow_odds_coverage": odds_coverage,
        "daemon_runtime": daemon_runtime,
        "prejump_sidecar_gate": sidecar_gate,
        "live_feature_audit": load_json(
            live_feature_audit_dir / "sidecar_target_metadata_live_feature_audit.json"
        )
        if live_feature_audit_dir
        else None,
        "coverage_gap": coverage_summary(
            load_json(coverage_gap_dir / "train_live_feature_coverage_gap_audit.json")
            if coverage_gap_dir
            else None,
            selected_metrics=metrics,
        ),
        "source_dirs": {
            "aggregate_result_dir": relpath(aggregate_result_dir),
            "result_join_dir": relpath(result_join_dir),
            "sidecar_gate_dir": relpath(sidecar_gate_dir),
            "live_feature_audit_dir": relpath(live_feature_audit_dir),
            "activation_gate_dir": relpath(activation_gate_dir),
            "odds_snapshot_dir": relpath(odds_snapshot_dir),
            "odds_status_path": relpath(odds_status_path),
            "odds_snapshot_source_kind": odds_evidence.get("source_kind"),
            "odds_coverage_report_path": relpath(odds_coverage_path),
            "daemon_runtime_state_path": relpath(daemon_runtime_state_path)
            if daemon_runtime_state_path.exists()
            else None,
            "coverage_gap_dir": relpath(coverage_gap_dir),
        },
        "result_metric_source": result_metric_source,
        "no_write_guarantees": {
            "production_promotion": False,
            "registry_mutation": False,
            "production_pointer_update": False,
            "production_prediction_write": False,
            "db_write": False,
            "label_write": False,
            "tgr_enabled": False,
            "betting_or_ev_output": False,
        },
    }


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
        prefix_error="output_dir_must_be_forward_shadow_status_artifact",
        evidence_root=evidence_root,
    )


def build_summary(report: Mapping[str, Any]) -> str:
    metrics = report.get("forward_metrics") or {}
    activation = report.get("activation_gate") or {}
    odds = report.get("shadow_odds_snapshot") or {}
    odds_coverage = report.get("shadow_odds_coverage") or {}
    daemon = report.get("daemon_runtime") or {}
    next_race = daemon.get("last_next_prejump_race") or {}
    lines = [
        "# Forward Shadow Status",
        "",
        f"- Final status: `{report.get('final_status')}`",
        f"- Reasons: `{report.get('status_reasons')}`",
        f"- Safe joined races: `{metrics.get('safe_joined_race_count')}`",
        f"- Pending races: `{metrics.get('pending_race_count')}`",
        f"- Unsafe matches: `{metrics.get('unsafe_match_count')}`",
        f"- Top1: `{metrics.get('top1')}`",
        f"- Top3: `{metrics.get('top3')}`",
        f"- Mean winner rank: `{metrics.get('mean_winner_rank')}`",
        f"- Brier: `{metrics.get('brier')}`",
        f"- LogLoss: `{metrics.get('logloss')}`",
        f"- Quarantined features: `{activation.get('kept_quarantined_features')}`",
        f"- Odds analysis status: `{odds.get('odds_analysis_status')}`",
        f"- Odds blockers: `{odds.get('odds_analysis_blocker_counts')}`",
        f"- Odds EV output rows: `{odds.get('ev_output_rows')}`",
        f"- Odds coverage status: `{odds_coverage.get('status')}`",
        f"- Odds coverage readiness: `{odds_coverage.get('readiness_status')}`",
        f"- Odds coverage next action: `{odds_coverage.get('next_action')}`",
        f"- Dog-level odds rows: `{odds_coverage.get('dog_level_win_odds_rows')}`",
        f"- Odds rows missing source URL: `{odds_coverage.get('source_url_rows_missing')}`",
        f"- Stale current odds rows: `{odds_coverage.get('stale_current_win_rows')}`",
        f"- Daemon cycle activity: `{daemon.get('last_cycle_activity_status')}`",
        f"- Next pre-jump refresh status: `{daemon.get('last_next_prejump_refresh_status')}`",
        f"- Next pre-jump race: `{next_race.get('race_id')}` at `{next_race.get('jump_datetime')}`",
        f"- Recommended rerun after: `{daemon.get('last_recommended_rerun_after_local')}`",
        "",
        "## Decision",
        "- Continue collecting forward shadow results.",
        "- Keep `quarantine_feature` for same-distance/same-grade timing fields.",
        "- Do not promote, enable TGR, mutate registry, write DB labels, or write betting/EV outputs.",
    ]
    return "\n".join(lines) + "\n"


def run_status_report(
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    output_dir: Path | None = None,
    db_path: Path = ROOT / "greyhound_racing_data.db",
    min_joined_races: int = DEFAULT_MIN_JOINED_RACES,
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or evidence_root / f"forward_shadow_status_{generated_at.strftime('%Y%m%dT%H%M%S%z')}"
    output_dir = assert_output_dir_safe(output_dir, evidence_root=evidence_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes()
    report = build_status_report(
        evidence_root=evidence_root,
        db_path=db_path,
        min_joined_races=min_joined_races,
        generated_at=generated_at,
    )
    protected_after = protected_hashes()
    report["protected_hashes_before"] = protected_before
    report["protected_hashes_after"] = protected_after
    report["protected_paths_unchanged"] = protected_before == protected_after
    if not report["protected_paths_unchanged"]:
        report["final_status"] = "BLOCKED_PROTECTED_PATH_MUTATION"
        report.setdefault("status_reasons", []).append("protected_paths_changed")
    write_json(output_dir / "forward_shadow_status_report.json", report)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["final_status"],
        "status_reasons": report["status_reasons"],
        "safe_joined_race_count": report["forward_metrics"]["safe_joined_race_count"],
        "pending_race_count": report["forward_metrics"]["pending_race_count"],
        "protected_paths_unchanged": report["protected_paths_unchanged"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    parser.add_argument("--min-joined-races", type=int, default=DEFAULT_MIN_JOINED_RACES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_status_report(
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
        db_path=args.db,
        min_joined_races=args.min_joined_races,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
