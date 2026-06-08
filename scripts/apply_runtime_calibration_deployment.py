#!/usr/bin/env python3
"""Stage a gated runtime calibration deployment.

Default mode is dry-run: validate the no-write deployment plan and emit the
exact write targets, rollback path, and command templates without touching
runtime config, registry state, model artifacts, labels, or betting surfaces.

Write mode is intentionally narrow and requires both ``--write-approved`` and
``APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR``. It writes only a report-only
runtime calibration config and a refresh signal. It never mutates model_index,
best-model symlinks, model artifacts, labels, or betting state.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "runtime_calibration_deployment_execution_v1"
CONFIG_SCHEMA_VERSION = "runtime_calibration_config_v1"
REFRESH_SIGNAL_SCHEMA_VERSION = "runtime_calibration_refresh_signal_v1"
DEPLOYMENT_PLAN_SCHEMA_VERSION = "calibration_deployment_plan_v1"
DEPLOYMENT_PLAN_READY_STATUS = (
    "READY_FOR_SEPARATE_PROMOTION_IMPLEMENTATION_REVIEW"
)
CALIBRATION_DESIGN_SCHEMA_VERSION = "calibration_layer_design_v1"
DESIGN_READY_STATUS = "READY_FOR_OPERATOR_DESIGN_REVIEW"
APPROVAL_ENV = "APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR"
APPROVAL_FLAG = "--write-approved"
DEFAULT_CONFIG_PATH = Path("model_registry/runtime_calibration_report_only.json")
DEFAULT_REFRESH_SIGNAL_PATH = Path("model_registry/refresh_signal.json")
DEFAULT_BACKUP_DIR = Path("model_registry/runtime_calibration_backups")


WRITE_SURFACES = {
    "runtime_calibration_config": False,
    "refresh_signal": False,
    "model_artifact_write": False,
    "model_registry_index_mutation": False,
    "best_model_symlink_mutation": False,
    "label_write": False,
    "betting": False,
}


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _approval_env_enabled() -> bool:
    value = str(os.environ.get(APPROVAL_ENV) or "").strip().lower()
    return value in {"1", "true", "yes", "on", "approved"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_if(condition: bool, failures: list[str], name: str) -> None:
    if condition:
        failures.append(name)


def _load_plan_and_design(
    deployment_plan_path: Path,
    calibration_design_path: Path | None,
    failures: list[str],
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    try:
        plan = _json(deployment_plan_path)
    except Exception as exc:
        failures.append(f"deployment_plan_unreadable:{type(exc).__name__}")
        return {}, {}, calibration_design_path

    source_evidence = _mapping(plan.get("source_evidence"))
    if calibration_design_path is None and source_evidence.get("calibration_design"):
        calibration_design_path = Path(str(source_evidence["calibration_design"]))

    if calibration_design_path is None:
        failures.append("calibration_design_path_missing")
        return plan, {}, None

    try:
        design = _json(calibration_design_path)
    except Exception as exc:
        failures.append(f"calibration_design_unreadable:{type(exc).__name__}")
        design = {}

    return plan, design, calibration_design_path


def _validate_plan(plan: Mapping[str, Any], failures: list[str]) -> None:
    controls = _mapping(plan.get("deployment_controls"))
    writes_performed = _mapping(plan.get("writes_performed"))
    loop_pass_through = _mapping(plan.get("loop_pass_through"))
    _append_if(
        plan.get("schema_version") != DEPLOYMENT_PLAN_SCHEMA_VERSION,
        failures,
        "deployment_plan_schema_mismatch",
    )
    _append_if(
        plan.get("status") != DEPLOYMENT_PLAN_READY_STATUS,
        failures,
        "deployment_plan_not_ready",
    )
    _append_if(bool(plan.get("failures")), failures, "deployment_plan_has_failures")
    _append_if(
        controls.get("actual_promotion_command_ready") is not False,
        failures,
        "deployment_plan_exposes_promotion_command",
    )
    _append_if(
        plan.get("actual_promotion_command") is not None,
        failures,
        "deployment_plan_actual_promotion_command_present",
    )
    for key, value in writes_performed.items():
        _append_if(value is not False, failures, f"prior_write_performed:{key}")
    for key in (
        "dry_run_capture",
        "approved_persist_capture",
        "live_odds_capture",
        "persist_same_run",
        "live_odds_same_run",
    ):
        _append_if(
            loop_pass_through.get(key) is not True,
            failures,
            f"loop_pass_through_missing:{key}",
        )


def _validate_design(design: Mapping[str, Any], failures: list[str]) -> None:
    transform = _mapping(design.get("runtime_transform_spec"))
    deployment = _mapping(design.get("deployment_control"))
    _append_if(
        design.get("schema_version") != CALIBRATION_DESIGN_SCHEMA_VERSION,
        failures,
        "calibration_design_schema_mismatch",
    )
    _append_if(
        design.get("status") != DESIGN_READY_STATUS,
        failures,
        "calibration_design_not_ready",
    )
    _append_if(bool(design.get("failures")), failures, "calibration_design_has_failures")
    _append_if(
        transform.get("algorithm") != "power_normalize_per_race",
        failures,
        "calibration_algorithm_mismatch",
    )
    _append_if(
        transform.get("input_probability_key") != "win_prob_norm",
        failures,
        "calibration_input_key_mismatch",
    )
    _append_if(
        transform.get("output_probability_key")
        != "calibrated_win_prob_report_only",
        failures,
        "calibration_output_key_mismatch",
    )
    _append_if(
        transform.get("uses_labels_at_runtime") is not False,
        failures,
        "calibration_uses_labels_at_runtime",
    )
    _append_if(
        transform.get("uses_odds_at_runtime") is not False,
        failures,
        "calibration_uses_odds_at_runtime",
    )
    _append_if(
        transform.get("rank_preserving_when_alpha_positive") is not True,
        failures,
        "calibration_not_rank_preserving",
    )
    for key in (
        "model_artifact_written",
        "registry_mutation_allowed",
        "production_config_write_allowed",
        "promotion_allowed",
        "betting_allowed",
    ):
        _append_if(
            deployment.get(key) is not False,
            failures,
            f"calibration_control_not_blocked:{key}",
        )
    _append_if(
        deployment.get("required_gate") != "APPROVE_MODEL_PROMOTION",
        failures,
        "calibration_required_gate_mismatch",
    )


def _runtime_config(
    *,
    deployment_plan_path: Path,
    calibration_design_path: Path,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "status": "ACTIVE_REPORT_ONLY",
        "generated_at": _iso_now(),
        "source_evidence": {
            "deployment_plan": str(deployment_plan_path.resolve()),
            "calibration_design": str(calibration_design_path.resolve()),
            "model_review_packet": _mapping(plan.get("source_evidence")).get(
                "model_review_packet"
            ),
            "prejump_loop_plan": _mapping(plan.get("source_evidence")).get(
                "prejump_loop_plan"
            ),
        },
        "runtime_transform_spec": dict(
            _mapping(design.get("runtime_transform_spec"))
        ),
        "runtime_scope": {
            "canonical_probability_key_unchanged": "win_prob_norm",
            "calibrated_probability_key": "calibrated_win_prob_report_only",
            "canonical_rank_unchanged": True,
            "report_only": True,
            "uses_labels_at_runtime": False,
            "uses_odds_at_runtime": False,
        },
        "deployment_control": {
            "activated_by_executor": True,
            "required_cli_flag": APPROVAL_FLAG,
            "required_env_var": APPROVAL_ENV,
            "model_artifact_written": False,
            "model_registry_index_mutated": False,
            "best_model_symlinks_mutated": False,
            "label_write": False,
            "betting": False,
        },
    }


def _refresh_signal(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REFRESH_SIGNAL_SCHEMA_VERSION,
        "timestamp": _iso_now(),
        "event": "runtime_calibration_report_only_config_updated",
        "runtime_calibration_config": str(config_path.resolve()),
        "selection_policy": "unchanged",
        "prediction_type": "win",
        "model_registry_index_mutated": False,
        "model_artifact_written": False,
        "best_model_symlinks_mutated": False,
        "runtime_transform_spec": dict(
            _mapping(config.get("runtime_transform_spec"))
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _backup_existing(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak"
    shutil.copy2(path, backup_path)
    return backup_path


def build_execution(
    *,
    deployment_plan_path: Path,
    calibration_design_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    refresh_signal_path: Path = DEFAULT_REFRESH_SIGNAL_PATH,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    write_approved: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    deployment_plan_path = deployment_plan_path.resolve()
    config_path = config_path.resolve()
    refresh_signal_path = refresh_signal_path.resolve()
    backup_dir = backup_dir.resolve()
    plan, design, resolved_design_path = _load_plan_and_design(
        deployment_plan_path,
        calibration_design_path.resolve() if calibration_design_path else None,
        failures,
    )
    if plan:
        _validate_plan(plan, failures)
    if design:
        _validate_design(design, failures)

    env_approved = _approval_env_enabled()
    write_requested = bool(write_approved)
    if write_requested and not env_approved:
        failures.append("write_requested_without_executor_env_approval")

    config = (
        _runtime_config(
            deployment_plan_path=deployment_plan_path,
            calibration_design_path=resolved_design_path,
            plan=plan,
            design=design,
        )
        if resolved_design_path and design
        else {}
    )
    signal = _refresh_signal(config_path, config) if config else {}

    rollback = {
        "config_path": str(config_path),
        "backup_path": None,
        "rollback_action_if_no_backup": "remove_runtime_calibration_config",
        "refresh_signal_previous_state_not_restored": True,
    }
    writes_performed = dict(WRITE_SURFACES)

    ready = not failures
    status = "DRY_RUN_READY" if ready else "NOT_READY"
    if ready and write_requested:
        backup_path = _backup_existing(config_path, backup_dir)
        _write_json(config_path, config)
        _write_json(refresh_signal_path, signal)
        rollback["backup_path"] = str(backup_path) if backup_path else None
        writes_performed["runtime_calibration_config"] = True
        writes_performed["refresh_signal"] = True
        status = "ACTIVE_REPORT_ONLY"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso_now(),
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "dry_run": not write_requested,
        "write_requested": write_requested,
        "write_approved_by_cli": write_requested,
        "write_approved_by_env": env_approved,
        "required_cli_flag": APPROVAL_FLAG,
        "required_env_var": APPROVAL_ENV,
        "source_evidence": {
            "deployment_plan": str(deployment_plan_path),
            "calibration_design": str(resolved_design_path)
            if resolved_design_path
            else None,
        },
        "write_targets": {
            "runtime_calibration_config": str(config_path),
            "refresh_signal": str(refresh_signal_path),
            "backup_dir": str(backup_dir),
            "model_registry_index": "not_touched",
            "best_model_symlinks": "not_touched",
            "model_artifacts": "not_touched",
        },
        "rollback": rollback,
        "runtime_calibration_config_preview": config,
        "refresh_signal_preview": signal,
        "writes_performed": writes_performed,
        "approved_loop_command_template": [
            ".venv/bin/python",
            "scripts/prejump_prediction_loop.py",
            "--report-only-calibration-design",
            str(config_path),
            "--execute-ready",
            "--approve-live-persist",
            "--approve-live-odds-capture",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-plan", required=True)
    parser.add_argument("--calibration-design")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--refresh-signal-path",
        default=str(DEFAULT_REFRESH_SIGNAL_PATH),
    )
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--write-approved",
        action="store_true",
        help=(
            "Actually write the runtime calibration config and refresh signal. "
            f"Requires {APPROVAL_ENV}=approved/true/1 in the same environment."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_execution(
        deployment_plan_path=Path(args.deployment_plan),
        calibration_design_path=Path(args.calibration_design)
        if args.calibration_design
        else None,
        config_path=Path(args.config_path),
        refresh_signal_path=Path(args.refresh_signal_path),
        backup_dir=Path(args.backup_dir),
        write_approved=bool(args.write_approved),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("status") in {"DRY_RUN_READY", "ACTIVE_REPORT_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
