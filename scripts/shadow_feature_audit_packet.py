#!/usr/bin/env python3
"""Shared report helpers for shadow feature audit packets.

The functions here are intentionally report-only. They centralize the file
contract used by daily shadow runs, live shadow scoring, and activation-gate
checks without changing feature schemas, model artifacts, DB state, labels,
snapshots, odds, EV, or betting outputs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from scripts.run_shadow_non_tgr_rf_evaluation import (
    same_distance_same_grade_history_provenance_report,
)


SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME = (
    "same_distance_same_grade_history_provenance.json"
)
SCORE_REPORT_COPIES = {
    "shadow_feature_population_report.json": "feature_population_report.json",
    "train_eval_feature_parity_report.json": "train_eval_feature_parity_report.json",
    "inactive_feature_policy_report.json": "inactive_feature_policy_report.json",
    SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME: SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME,
}
WAITING_LIVE_INPUT_STATUS = "NO_ELIGIBLE_PREJUMP_RACES"
WAITING_REASON = "no_live_feature_rows_available_for_same_distance_same_grade_history_audit"
REPORT_ONLY_NO_WRITE_GUARANTEES = {
    "db_write": False,
    "label_write": False,
    "canonical_schema_mutation": False,
    "production_prediction_write": False,
    "betting_or_ev_output": False,
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_shadow_feature_audit_reports(score_output_dir: Path, output_dir: Path) -> None:
    """Copy score-live audit reports into the daily shadow packet."""

    for source_name, target_name in SCORE_REPORT_COPIES.items():
        source = score_output_dir / source_name
        if source.exists():
            shutil.copy2(source, output_dir / target_name)


def waiting_same_distance_history_provenance_report(
    *,
    report_scope: str = "daily_shadow_run",
) -> dict[str, Any]:
    """Build an explicit no-eligible-inputs provenance report.

    This keeps waiting runs auditable: the same-distance/same-grade features are
    not silently missing from the artifact packet, and the report states that no
    target-race rows or post-outcome rows are permitted.
    """

    report = same_distance_same_grade_history_provenance_report([])
    report.update(
        {
            "report_scope": report_scope,
            "live_input_status": WAITING_LIVE_INPUT_STATUS,
            "reason": WAITING_REASON,
            "no_write_guarantees": dict(REPORT_ONLY_NO_WRITE_GUARANTEES),
        }
    )
    return report


def ensure_same_distance_history_provenance_report(
    *,
    output_dir: Path,
    score_output_dir: Path | None,
) -> Path:
    """Guarantee the daily packet has a same-distance provenance report."""

    target = output_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME
    if target.exists():
        return target
    if score_output_dir is not None:
        source = score_output_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME
        if source.exists():
            shutil.copy2(source, target)
            return target
    write_json(target, waiting_same_distance_history_provenance_report())
    return target


def first_existing_path(candidates: Sequence[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def feature_activation_gate_input_paths(
    *,
    daily_dir: Path | None,
    shadow_model: Path | None,
    baseline_metrics: Path | None = None,
    candidate_metrics: Path | None = None,
) -> dict[str, Path | None]:
    """Resolve report inputs needed by the feature activation gate."""

    score_live_dir = daily_dir / "shadow_score_live" if daily_dir else None
    model_dir = shadow_model.parent if shadow_model else None
    return {
        "parity_report": first_existing_path(
            [
                score_live_dir / "train_eval_feature_parity_report.json"
                if score_live_dir
                else None,
                model_dir / "train_eval_feature_parity_report.json" if model_dir else None,
            ]
        ),
        "inactive_policy_report": first_existing_path(
            [
                score_live_dir / "inactive_feature_policy_report.json"
                if score_live_dir
                else None,
                score_live_dir / "active_feature_policy_report.json" if score_live_dir else None,
                model_dir / "inactive_feature_policy_report.json" if model_dir else None,
            ]
        ),
        "matrix_audit": first_existing_path(
            [model_dir / "shadow_feature_matrix_audit.json" if model_dir else None]
        ),
        "same_distance_history_provenance": first_existing_path(
            [
                score_live_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME
                if score_live_dir
                else None,
                daily_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME if daily_dir else None,
            ]
        ),
        "baseline_metrics": first_existing_path([baseline_metrics]),
        "candidate_metrics": first_existing_path([candidate_metrics]),
    }
