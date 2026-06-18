#!/usr/bin/env python3
"""Shadow-only RandomForest evaluation for the repaired non-TGR schema.

This tool intentionally writes only under shadow artifact roots. It never
registers a model, updates production pointers, writes predictions, writes EV
or betting files, mutates snapshots, rewrites manifests, or enables TGR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.run_feature_recovery_execution_v1 import (  # noqa: E402
    DEFAULT_CLEAN_DATASET,
    DEFAULT_DB,
    DEFAULT_REPAIRED_PACKET,
    DEFAULT_SCHEMA,
    HARD_EXCLUDED_FEATURE_NAMES,
    IDENTITY_FEATURE_NAMES,
    RANDOM_SEED,
    add_history_features,
    build_repaired_dataset,
    clean_name,
    git_output,
    group_key,
    load_csv,
    load_db_history,
    load_json,
    load_jsonl,
    mean,
    matrix_gate,
    normalize_grade,
    parse_date,
    parse_datetime_minutes,
    parse_race_number,
    prepare_xy,
    relpath,
    safe_float,
    safe_int,
    serialize_cell,
    sha256_file,
    sqlite_ro,
    write_csv,
    write_json,
    write_text,
)
from utils.expert_form_metadata import safe_expert_form_metadata_from_payload  # noqa: E402
from utils.csv_metadata import (  # noqa: E402
    load_safe_sidecar_target_metadata,
    load_safe_weather_track_metadata,
)
from utils.race_lifecycle import extract_target_metadata_from_filename  # noqa: E402


CALIBRATION_METHOD_KEY = "power_gamma_2.4"
POWER_GAMMA = 2.4
SHADOW_MODEL_FAMILY = "RandomForest"
SHADOW_OUTPUT_MODE = "shadow_only"
STAGE2_FORWARD_SHADOW_COLLECTING = "STAGE2_FORWARD_SHADOW_COLLECTING"
STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW = "STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW"
ALL_MISSING_TRAIN_POLICIES = ("report_only", "quarantine_feature", "fail")
WATCHED_PARITY_FEATURES = (
    "same_distance_same_grade_best_time",
    "same_distance_same_grade_avg_time",
)
SAME_DISTANCE_HISTORY_SOURCE = "prior_dog_history"
SAME_DISTANCE_HISTORY_CUTOFF = "strictly_before_target_race"
SAME_DISTANCE_HISTORY_CUTOFF_BASIS = "race_date_less_than_target_race_date"

DEFAULT_FULL_EVIDENCE_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
DEFAULT_LIVE_PARENT = ROOT / "artifacts/shadow_evaluation"

ALLOWED_OUTPUT_PREFIXES = (
    "artifacts/shadow_evaluation",
    "artifacts/full_evidence_orchestration_20260525/shadow_evaluation_",
    "artifacts/full_evidence_orchestration_20260525/shadow_reliability_population_hardening_v1_",
    "artifacts/full_evidence_orchestration_20260525/shadow_reliability_resume_after_db_recovery_",
    "artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_",
)
PROTECTED_OUTPUT_PREFIXES = (
    "artifacts/prediction_snapshots",
    "artifacts/eval",
    "model_registry",
    "docs/model_contracts",
    "docs/model_registry",
    "ml_models_v4",
    "advanced_models",
    "models",
    "predictions",
)
PROTECTED_PATHS = (
    "greyhound_racing_data.db",
    "greyhound_racing_data_writable.db",
    "model_registry/best_metadata.json",
    "docs/model_contracts/v4_feature_contract.json",
    "model_registry/current_production.json",
    "processed_manifest.json",
    "artifacts/prediction_snapshots/manifest.jsonl",
    "models",
    "predictions",
)
FORBIDDEN_APPROVAL_ENV_VARS = (
    "APPROVE_RESULT_LABEL_WRITE",
    "APPROVE_LIVE_DB_WRITE",
    "APPROVE_PRODUCTION_MODEL_PROMOTION",
    "ENABLE_TGR",
    "TGR_ENABLED",
)
POST_OUTCOME_PREFIXES = (
    "actual_",
    "official_",
    "result_",
    "results_",
    "winner_",
    "scraped_",
    "target_finish",
)
LIVE_RUNNER_PREFIX_RE = re.compile(r"^\s*(\d{1,2})\.\s*(.+?)\s*$")


def now_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def ensure_shadow_runtime_guard() -> None:
    enabled = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if os.environ.get(name)]
    if enabled:
        raise SystemExit(f"refusing_shadow_run_with_forbidden_env:{','.join(sorted(enabled))}")


def repo_relative_text(path: Path, root: Path = ROOT) -> str:
    logical = path if path.is_absolute() else root / path
    logical = logical.absolute()
    try:
        return logical.relative_to(root.absolute()).as_posix()
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc


def shadow_relpath(path: Path) -> str:
    try:
        return repo_relative_text(path)
    except ValueError:
        return relpath(path)


def assert_shadow_output_dir_safe(output_dir: Path, root: Path = ROOT) -> Path:
    relative = repo_relative_text(output_dir, root)
    allowed = False
    for prefix in ALLOWED_OUTPUT_PREFIXES:
        if relative == prefix or relative.startswith(prefix + "/"):
            allowed = True
            break
        if prefix.endswith("_") and relative.startswith(prefix):
            allowed = True
            break
    if not allowed:
        raise ValueError(f"output_dir_must_be_shadow_artifact:{relative}")

    for prefix in PROTECTED_OUTPUT_PREFIXES:
        if relative == prefix or relative.startswith(prefix + "/"):
            raise ValueError(f"output_dir_protected:{prefix}")
    return output_dir


def prepare_output_dir(output_dir: Path) -> Path:
    output_dir = assert_shadow_output_dir_safe(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output_dir_already_exists:{relpath(output_dir)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def default_implementation_output_dir() -> Path:
    return DEFAULT_FULL_EVIDENCE_PARENT / f"shadow_evaluation_implementation_{now_id()}"


def default_live_output_dir() -> Path:
    return DEFAULT_LIVE_PARENT / now_id()


def path_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "type": "file",
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        digest = hashlib.sha256()
        file_count = 0
        total_bytes = 0
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            try:
                relative = child.relative_to(path).as_posix()
            except ValueError:
                relative = str(child)
            stat = child.stat()
            file_count += 1
            total_bytes += stat.st_size
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        return {
            "type": "directory",
            "exists": True,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "listing_sha256": digest.hexdigest(),
        }
    return {"type": "missing", "exists": False}


def protected_path_snapshot() -> dict[str, dict[str, Any]]:
    return {path: path_state(ROOT / path) for path in PROTECTED_PATHS}


def protected_path_verification(before: Mapping[str, Any]) -> dict[str, Any]:
    after = protected_path_snapshot()
    unchanged = dict(before) == after
    return {
        "schema_version": "shadow_protected_path_verification_v1",
        "protected_paths": list(PROTECTED_PATHS),
        "before": before,
        "after": after,
        "protected_paths_unchanged": unchanged,
        "forbidden_actions": {
            "db_writes": False,
            "label_writes": False,
            "registry_mutation": False,
            "production_pointer_update": False,
            "active_model_replacement": False,
            "snapshot_rewrite": False,
            "existing_manifest_rewrite": False,
            "tgr_enablement": False,
            "betting_action": False,
            "ev_action": False,
            "production_prediction_endpoint_mutation": False,
            "champion_artifact_overwrite": False,
        },
    }


def write_jsonl_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def stage2_shadow_prediction_rows(
    predictions: Sequence[Mapping[str, Any]],
    *,
    stage2_status: str,
) -> list[dict[str, Any]]:
    rows = []
    for prediction in predictions:
        item = dict(prediction)
        item.update(
            {
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_forward_shadow_status": stage2_status,
                "stage2_challenger_family": SHADOW_MODEL_FAMILY,
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
                "odds_used_for_shadow_scoring": False,
                "ev_output": False,
                "betting_action": False,
                "production_prediction_write": False,
                "registry_mutation": False,
                "production_pointer_update": False,
            }
        )
        rows.append(item)
    return rows


def validate_schema_contract(schema: Mapping[str, Any]) -> dict[str, Any]:
    features = list(schema.get("feature_columns") or [])
    tgr_columns = [feature for feature in features if str(feature).startswith("tgr_")]
    hard_exclusions = sorted(set(features).intersection(HARD_EXCLUDED_FEATURE_NAMES))
    identity_features = sorted(set(features).intersection(IDENTITY_FEATURE_NAMES))
    post_outcome_features = sorted(
        feature
        for feature in features
        if feature in hard_exclusions
        or str(feature).lower() in HARD_EXCLUDED_FEATURE_NAMES
        or str(feature).lower().startswith(POST_OUTCOME_PREFIXES)
        or str(feature).lower().endswith("_result")
        or str(feature).lower().endswith("_results")
    )
    duplicate_features = sorted(feature for feature, count in Counter(features).items() if count > 1)
    fail_reasons: list[str] = []
    if len(features) != 78:
        fail_reasons.append(f"feature_count_not_78:{len(features)}")
    if tgr_columns:
        fail_reasons.append(f"tgr_columns_present:{len(tgr_columns)}")
    if hard_exclusions:
        fail_reasons.append(f"hard_exclusions_present:{hard_exclusions}")
    if identity_features:
        fail_reasons.append(f"identity_features_present:{identity_features}")
    if post_outcome_features:
        fail_reasons.append(f"post_outcome_features_present:{post_outcome_features}")
    if duplicate_features:
        fail_reasons.append(f"duplicate_features_present:{duplicate_features}")
    return {
        "status": "PASS" if not fail_reasons else "FAIL",
        "feature_count": len(features),
        "tgr_columns": tgr_columns,
        "hard_exclusions_present": hard_exclusions,
        "identity_columns_present_as_features": identity_features,
        "post_outcome_columns_present_as_features": post_outcome_features,
        "duplicate_features": duplicate_features,
        "fail_reasons": fail_reasons,
        "features": features,
    }


def write_shadow_candidate_definition(
    output_dir: Path,
    schema: Mapping[str, Any],
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    schema_audit = validate_schema_contract(schema)
    definition = {
        "schema_version": "shadow_candidate_definition_v1",
        "model_family": SHADOW_MODEL_FAMILY,
        "schema": {
            "path": shadow_relpath(schema_path),
            "schema_version": schema.get("schema_version"),
            "feature_count": len(schema.get("feature_columns") or []),
            "feature_columns_sha256": hashlib.sha256(
                json.dumps(list(schema.get("feature_columns") or []), sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "calibration": {
            "method_key": CALIBRATION_METHOD_KEY,
            "method": "power",
            "gamma": POWER_GAMMA,
            "rank_preserving": True,
            "formula": "p_i_cal = p_i^2.4 / sum_j(p_j^2.4)",
        },
        "tgr_enabled": False,
        "registry_mutation": False,
        "promotion_allowed": False,
        "production_pointer_update_allowed": False,
        "active_model_replacement_allowed": False,
        "output_mode": SHADOW_OUTPUT_MODE,
        "source_label_policy": "canonical clean official labels only",
        "schema_audit_status": schema_audit["status"],
    }
    write_json(output_dir / "shadow_candidate_definition.json", definition)
    return definition


def canonical_label_audit(clean_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bad_rows = []
    quality_counts: Counter[str] = Counter()
    result_quality_counts: Counter[str] = Counter()
    for row in clean_rows:
        quality = str(row.get("label_quality") or "DATA_MISSING")
        result_quality = str(row.get("result_detail_quality") or "DATA_MISSING")
        quality_counts[quality] += 1
        result_quality_counts[result_quality] += 1
        has_label = safe_int(row.get("actual_win")) in (0, 1)
        official = "official" in quality.lower() or "complete_result" in quality.lower()
        if not has_label or not official:
            bad_rows.append(
                {
                    "race_id": row.get("race_id"),
                    "dog_name": row.get("dog_name"),
                    "label_quality": row.get("label_quality"),
                    "actual_win": row.get("actual_win"),
                }
            )
    return {
        "status": "PASS" if not bad_rows else "FAIL",
        "row_count": len(clean_rows),
        "label_quality_counts": dict(sorted(quality_counts.items())),
        "result_detail_quality_counts": dict(sorted(result_quality_counts.items())),
        "bad_rows": bad_rows[:25],
        "bad_row_count": len(bad_rows),
        "policy": "canonical clean official labels only",
    }


def family_population_report(
    population: Mapping[str, Any],
    features: Sequence[str],
    family_name: str,
    wanted_features: Sequence[str],
) -> dict[str, Any]:
    by_feature = population.get("by_feature") or {}
    selected = [feature for feature in features if feature in wanted_features]
    present_rows = {
        feature: (by_feature.get(feature) or {}).get("present_rows", 0) for feature in selected
    }
    return {
        "family": family_name,
        "feature_count": len(selected),
        "features": selected,
        "present_rows_by_feature": present_rows,
        "populated_feature_count": sum(1 for value in present_rows.values() if value),
    }


def present_values(rows: Sequence[Mapping[str, Any]], feature: str) -> list[Any]:
    return [row.get(feature) for row in rows if row.get(feature) not in (None, "")]


def train_eval_feature_parity_report(
    dataset: Mapping[str, Any],
    *,
    policy: str,
) -> dict[str, Any]:
    if policy not in ALL_MISSING_TRAIN_POLICIES:
        raise ValueError(f"unknown_all_missing_train_policy:{policy}")
    features = list(dataset["features"])
    train_rows = list(dataset["train_rows"])
    holdout_rows = list(dataset["holdout_rows"])
    by_feature: dict[str, Any] = {}
    all_missing_train_features: list[str] = []
    all_missing_train_present_holdout_features: list[str] = []

    for feature in features:
        train_present = present_values(train_rows, feature)
        holdout_present = present_values(holdout_rows, feature)
        all_missing_in_train = not train_present
        all_missing_in_holdout = not holdout_present
        present_in_holdout = bool(holdout_present)
        if all_missing_in_train:
            all_missing_train_features.append(feature)
        if all_missing_in_train and present_in_holdout:
            all_missing_train_present_holdout_features.append(feature)

        if all_missing_in_train and present_in_holdout:
            parity_status = "ALL_MISSING_IN_TRAIN_PRESENT_IN_HOLDOUT"
        elif all_missing_in_train and all_missing_in_holdout:
            parity_status = "ALL_MISSING_BOTH_SPLITS"
        elif not all_missing_in_train and all_missing_in_holdout:
            parity_status = "PRESENT_IN_TRAIN_ALL_MISSING_IN_HOLDOUT"
        else:
            parity_status = "PRESENT_IN_BOTH_SPLITS"

        by_feature[feature] = {
            "feature": feature,
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "train_present_rows": len(train_present),
            "train_present_pct": len(train_present) / len(train_rows) if train_rows else 0.0,
            "holdout_present_rows": len(holdout_present),
            "holdout_present_pct": len(holdout_present) / len(holdout_rows) if holdout_rows else 0.0,
            "train_unique_present_values": len({serialize_cell(value) for value in train_present}),
            "holdout_unique_present_values": len({serialize_cell(value) for value in holdout_present}),
            "all_missing_in_train": all_missing_in_train,
            "all_missing_in_holdout": all_missing_in_holdout,
            "present_in_holdout": present_in_holdout,
            "watched_feature": feature in WATCHED_PARITY_FEATURES,
            "parity_status": parity_status,
        }

    inactive = list(all_missing_train_features) if policy == "quarantine_feature" else []
    policy_status = "FAIL" if policy == "fail" and all_missing_train_features else "WARN" if all_missing_train_features else "PASS"
    if policy == "quarantine_feature" and all_missing_train_features:
        policy_action = "quarantine_train_all_missing_features_for_this_run_only"
    elif policy == "fail" and all_missing_train_features:
        policy_action = "fail_before_training_or_scoring"
    elif policy == "report_only" and all_missing_train_features:
        policy_action = "report_warning_keep_features_active"
    else:
        policy_action = "no_inactive_features"

    watched = {
        feature: by_feature.get(feature, {"feature": feature, "missing_from_schema": True})
        for feature in WATCHED_PARITY_FEATURES
    }
    return {
        "schema_version": "train_eval_feature_parity_report_v1",
        "policy": policy,
        "policy_status": policy_status,
        "policy_action": policy_action,
        "feature_count": len(features),
        "train_rows": len(train_rows),
        "train_races": len({row.get("race_id") for row in train_rows}),
        "holdout_rows": len(holdout_rows),
        "holdout_races": len({row.get("race_id") for row in holdout_rows}),
        "all_missing_train_feature_count": len(all_missing_train_features),
        "all_missing_train_features": all_missing_train_features,
        "all_missing_train_present_holdout_feature_count": len(
            all_missing_train_present_holdout_features
        ),
        "all_missing_train_present_holdout_features": all_missing_train_present_holdout_features,
        "inactive_features_due_to_train_all_missing": inactive,
        "watched_features": watched,
        "by_feature": by_feature,
    }


def inactive_feature_policy_report(parity_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "inactive_feature_policy_report_v1",
        "policy": parity_report.get("policy"),
        "policy_status": parity_report.get("policy_status"),
        "policy_action": parity_report.get("policy_action"),
        "canonical_schema_mutation": False,
        "model_registry_mutation": False,
        "inactive_features_due_to_train_all_missing": list(
            parity_report.get("inactive_features_due_to_train_all_missing") or []
        ),
        "active_feature_count_after_policy": int(parity_report.get("feature_count") or 0)
        - len(parity_report.get("inactive_features_due_to_train_all_missing") or []),
        "original_schema_feature_count": parity_report.get("feature_count"),
        "all_missing_train_features": list(parity_report.get("all_missing_train_features") or []),
        "all_missing_train_present_holdout_features": list(
            parity_report.get("all_missing_train_present_holdout_features") or []
        ),
        "warning": (
            "features all-missing in train are explicit; report_only keeps them active, "
            "quarantine_feature removes them only from this run, fail aborts before training/scoring"
        ),
    }


def dataset_with_all_missing_train_policy(
    dataset: Mapping[str, Any],
    parity_report: Mapping[str, Any],
) -> dict[str, Any]:
    policy = str(parity_report.get("policy") or "")
    inactive = list(parity_report.get("inactive_features_due_to_train_all_missing") or [])
    if policy == "fail" and parity_report.get("all_missing_train_features"):
        raise RuntimeError(
            "all_missing_train_policy_failed:"
            + ",".join(str(feature) for feature in parity_report["all_missing_train_features"])
        )
    output = dict(dataset)
    original_features = list(dataset["features"])
    if inactive:
        inactive_set = set(inactive)
        output["features"] = [feature for feature in original_features if feature not in inactive_set]
        output["categorical_features"] = [
            feature
            for feature in list(dataset.get("categorical_features") or [])
            if feature not in inactive_set
        ]
    else:
        output["features"] = original_features
        output["categorical_features"] = list(dataset.get("categorical_features") or [])
    output["schema_features"] = original_features
    output["all_missing_train_policy"] = policy
    output["inactive_features_due_to_train_all_missing"] = inactive
    return output


def build_shadow_feature_matrix(
    *,
    clean_dataset: Path,
    repaired_packet: Path,
    schema_path: Path,
    db_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    clean_rows = load_jsonl(clean_dataset)
    packet_rows = load_csv(repaired_packet)
    schema = load_json(schema_path)
    schema_audit = validate_schema_contract(schema)
    if schema_audit["status"] != "PASS":
        raise RuntimeError(f"schema_contract_failed:{schema_audit['fail_reasons']}")

    connection = sqlite_ro(db_path)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        dataset = build_repaired_dataset(
            clean_rows=clean_rows,
            packet_rows=packet_rows,
            schema=schema,
            connection=connection,
        )
    finally:
        connection.close()

    gate = matrix_gate(dataset)
    label_audit = canonical_label_audit(clean_rows)
    train_ids = {str(row.get("race_id") or "") for row in dataset["train_rows"]}
    holdout_ids = {str(row.get("race_id") or "") for row in dataset["holdout_rows"]}
    race_id_overlap = sorted(train_ids.intersection(holdout_ids))
    train_dates = [str(row.get("race_date") or "") for row in dataset["train_rows"] if row.get("race_date")]
    holdout_dates = [str(row.get("race_date") or "") for row in dataset["holdout_rows"] if row.get("race_date")]
    temporal_status = "PASS"
    if race_id_overlap:
        temporal_status = "FAIL"
    elif train_dates and holdout_dates and max(train_dates) >= min(holdout_dates):
        temporal_status = "FAIL"

    audit_failures: list[str] = []
    if gate["status"] != "PASS":
        audit_failures.append("matrix_gate_failed")
    if label_audit["status"] != "PASS":
        audit_failures.append("label_policy_failed")
    if temporal_status != "PASS":
        audit_failures.append("temporal_split_failed")
    if dataset["leakage_audit"]["status"] != "PASS":
        audit_failures.append("leakage_audit_failed")

    audit = {
        "schema_version": "shadow_feature_matrix_audit_v1",
        "status": "PASS" if not audit_failures else "FAIL",
        "fail_reasons": audit_failures,
        "schema_contract": schema_audit,
        "matrix_gate": gate,
        "sqlite_quick_check": quick_check,
        "source_paths": {
            "clean_dataset": shadow_relpath(clean_dataset),
            "repaired_packet": shadow_relpath(repaired_packet),
            "schema": shadow_relpath(schema_path),
            "db": shadow_relpath(db_path),
        },
        "label_audit": label_audit,
        "feature_columns_exactly_78": len(dataset["features"]) == 78,
        "tgr_columns_present": [feature for feature in dataset["features"] if feature.startswith("tgr_")],
        "hard_exclusions_present": sorted(
            set(dataset["features"]).intersection(HARD_EXCLUDED_FEATURE_NAMES)
        ),
        "identity_columns_present_as_features": sorted(
            set(dataset["features"]).intersection(IDENTITY_FEATURE_NAMES)
        ),
        "post_outcome_columns_present_as_features": schema_audit[
            "post_outcome_columns_present_as_features"
        ],
        "temporal_evaluation": {
            "status": temporal_status,
            "train_races": len(train_ids),
            "holdout_races": len(holdout_ids),
            "race_id_overlap": race_id_overlap,
            "train_max_date": max(train_dates) if train_dates else None,
            "holdout_min_date": min(holdout_dates) if holdout_dates else None,
            "all_dogs_in_race_kept_together": True,
        },
        "source_label_policy": "canonical clean official labels only",
    }

    features = list(dataset["features"])
    population = {
        "schema_version": "shadow_feature_population_report_v1",
        "overall": dataset["feature_population"],
        "train": dataset["train_population"],
        "holdout": dataset["holdout_population"],
        "target_resolution_counts": dataset["target_resolution_counts"],
        "history_status_counts": dataset["history_status_counts"],
        "target_distance_safe": (dataset["feature_population"].get("by_feature") or {}).get(
            "target_distance_safe"
        ),
        "target_grade_safe": (dataset["feature_population"].get("by_feature") or {}).get(
            "target_grade_safe"
        ),
        "same_distance_family": family_population_report(
            dataset["feature_population"],
            features,
            "same_distance_family",
            [feature for feature in features if "same_distance" in feature],
        ),
        "same_grade_family": family_population_report(
            dataset["feature_population"],
            features,
            "same_grade_family",
            [feature for feature in features if "same_grade" in feature],
        ),
    }
    return dataset, audit, population


def sklearn_imports() -> dict[str, Any]:
    try:
        from joblib import dump, load
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception as exc:
        return {"status": "MISSING", "error": repr(exc)}
    return {
        "status": "OK",
        "dump": dump,
        "load": load,
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
    }


def make_one_hot_encoder(factory: Any) -> Any:
    try:
        return factory(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return factory(handle_unknown="ignore", sparse=False)


def train_or_load_shadow_rf(
    *,
    dataset: Mapping[str, Any],
    output_dir: Path,
    load_model: Path | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    deps = sklearn_imports()
    if deps["status"] != "OK":
        return None, {
            "schema_version": "shadow_training_report_v1",
            "status": "FAIL",
            "reason": "missing_ml_dependencies",
            "error": deps["error"],
            "model_family": SHADOW_MODEL_FAMILY,
            "registry_mutation": False,
            "promotion_allowed": False,
        }

    features = list(dataset["features"])
    schema_features = list(dataset.get("schema_features") or features)
    inactive = list(dataset.get("inactive_features_due_to_train_all_missing") or [])
    categorical = set(dataset.get("categorical_features") or [])
    categorical_indices = [index for index, feature in enumerate(features) if feature in categorical]
    numeric_indices = [index for index, feature in enumerate(features) if feature not in categorical]
    train_rows = list(dataset["train_rows"])
    holdout_rows = list(dataset["holdout_rows"])
    x_train, y_train = prepare_xy(train_rows, features)

    if load_model:
        pipeline = deps["load"](load_model)
        shadow_model_path = output_dir / "shadow_randomforest_model.joblib"
        if load_model.resolve() != shadow_model_path.resolve():
            shutil.copy2(load_model, shadow_model_path)
        status = "LOADED"
    else:
        transformer = deps["ColumnTransformer"](
            transformers=[
                ("num", deps["SimpleImputer"](strategy="median"), numeric_indices),
                (
                    "cat",
                    deps["Pipeline"](
                        steps=[
                            ("imputer", deps["SimpleImputer"](strategy="most_frequent")),
                            ("onehot", make_one_hot_encoder(deps["OneHotEncoder"])),
                        ]
                    ),
                    categorical_indices,
                ),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        estimator = deps["RandomForestClassifier"](
            n_estimators=300,
            min_samples_leaf=4,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        )
        pipeline = deps["Pipeline"](steps=[("prep", transformer), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        shadow_model_path = output_dir / "shadow_randomforest_model.joblib"
        deps["dump"](pipeline, shadow_model_path)
        status = "TRAINED"

    report = {
        "schema_version": "shadow_training_report_v1",
        "status": "PASS",
        "training_mode": status,
        "model_family": SHADOW_MODEL_FAMILY,
        "random_seed": RANDOM_SEED,
        "train_rows": len(train_rows),
        "train_races": len({row.get("race_id") for row in train_rows}),
        "holdout_rows": len(holdout_rows),
        "holdout_races": len({row.get("race_id") for row in holdout_rows}),
        "feature_count": len(features),
        "schema_feature_count": len(schema_features),
        "active_feature_count": len(features),
        "inactive_feature_count": len(inactive),
        "inactive_features_due_to_train_all_missing": inactive,
        "all_missing_train_policy": dataset.get("all_missing_train_policy") or "report_only",
        "categorical_feature_count": len(categorical_indices),
        "numeric_or_boolean_feature_count": len(numeric_indices),
        "model_artifact_path": shadow_relpath(shadow_model_path),
        "model_artifact_sha256": sha256_file(shadow_model_path),
        "registry_mutation": False,
        "promotion_allowed": False,
        "production_pointer_update": False,
        "active_model_replacement": False,
        "output_mode": SHADOW_OUTPUT_MODE,
    }
    return pipeline, report


def normalize_probabilities_by_race(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    probability_key: str,
    rank_key: str,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    grouped: dict[str, list[int]] = defaultdict(list)
    parsed_probs: list[float] = []
    for index, (row, probability) in enumerate(zip(output, probabilities)):
        group_id = group_key(row)
        row["shadow_race_group_id"] = group_id
        grouped[group_id].append(index)
        parsed_probs.append(max(float(probability), 0.0))

    for group_id, indexes in grouped.items():
        total = sum(parsed_probs[index] for index in indexes)
        if total <= 0:
            normalized = {index: 1.0 / len(indexes) for index in indexes}
        else:
            normalized = {index: parsed_probs[index] / total for index in indexes}
        ordered = sorted(indexes, key=lambda index: (-normalized[index], index))
        for rank, index in enumerate(ordered, start=1):
            output[index][probability_key] = normalized[index]
            output[index][rank_key] = rank
    return output


def apply_power_gamma_by_race(
    rows: Sequence[Mapping[str, Any]],
    *,
    gamma: float = POWER_GAMMA,
    input_key: str,
    output_key: str,
    output_rank_key: str,
    race_key: str = "shadow_race_group_id",
) -> list[dict[str, Any]]:
    if not math.isfinite(float(gamma)) or gamma <= 0:
        raise ValueError("gamma_must_be_positive_finite")
    if not rows:
        raise ValueError("rows_missing")
    output = [dict(row) for row in rows]
    grouped: dict[str, list[int]] = defaultdict(list)
    powered: list[float] = []
    for index, row in enumerate(output):
        group_id = row.get(race_key)
        if group_id in (None, ""):
            raise ValueError(f"{race_key}_missing")
        probability = safe_float(row.get(input_key))
        if probability is None:
            raise ValueError(f"{input_key}_invalid")
        if probability < 0:
            raise ValueError(f"{input_key}_negative")
        grouped[str(group_id)].append(index)
        powered.append(float(probability) ** gamma)

    for indexes in grouped.values():
        total = sum(powered[index] for index in indexes)
        if total <= 0:
            calibrated = {index: 1.0 / len(indexes) for index in indexes}
        else:
            calibrated = {index: powered[index] / total for index in indexes}
        ordered = sorted(indexes, key=lambda index: (-calibrated[index], index))
        for rank, index in enumerate(ordered, start=1):
            output[index][output_key] = calibrated[index]
            output[index][output_rank_key] = rank
    return output


def probability_sum_report(rows: Sequence[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("shadow_race_group_id") or group_key(row))].append(row)
    per_race = []
    errors = []
    for group_id, group_rows in sorted(grouped.items()):
        total = sum(safe_float(row.get(probability_key)) or 0.0 for row in group_rows)
        error = abs(1.0 - total)
        errors.append(error)
        per_race.append(
            {
                "group": group_id,
                "race_id": group_rows[0].get("race_id") if group_rows else None,
                "runner_count": len(group_rows),
                "sum": total,
                "abs_error": error,
            }
        )
    return {
        "probability_key": probability_key,
        "groups_checked": len(grouped),
        "max_abs_error": max(errors) if errors else None,
        "mean_abs_error": mean(errors),
        "per_race": per_race,
        "status": "PASS" if errors and max(errors) <= 1e-9 else "FAIL" if errors else "NO_GROUPS",
    }


def ranking_signature(row: Mapping[str, Any], fallback_index: int) -> tuple[str, str, int]:
    return (clean_name(row.get("dog_name")), str(row.get("box_number") or ""), fallback_index)


def ranking_preservation_report(
    before_rows: Sequence[Mapping[str, Any]],
    after_rows: Sequence[Mapping[str, Any]],
    *,
    before_key: str,
    after_key: str,
) -> dict[str, Any]:
    before_groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    after_groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(before_rows):
        before_groups[str(row.get("shadow_race_group_id") or group_key(row))].append((index, row))
    for index, row in enumerate(after_rows):
        after_groups[str(row.get("shadow_race_group_id") or group_key(row))].append((index, row))

    changed = []
    unexpected = []
    for group_id in sorted(before_groups):
        before_order = [
            ranking_signature(row, index)
            for index, row in sorted(
                before_groups[group_id],
                key=lambda item: (-(safe_float(item[1].get(before_key)) or 0.0), item[0]),
            )
        ]
        after_order = [
            ranking_signature(row, index)
            for index, row in sorted(
                after_groups[group_id],
                key=lambda item: (-(safe_float(item[1].get(after_key)) or 0.0), item[0]),
            )
        ]
        if before_order != after_order:
            before_probs = [safe_float(row.get(before_key)) or 0.0 for _, row in before_groups[group_id]]
            after_probs = [safe_float(row.get(after_key)) or 0.0 for _, row in after_groups[group_id]]
            tie_only = len(before_probs) != len(set(before_probs)) or len(after_probs) != len(set(after_probs))
            item = {
                "group": group_id,
                "race_id": before_groups[group_id][0][1].get("race_id"),
                "before_order": before_order,
                "after_order": after_order,
                "tie_only_explanation": tie_only,
            }
            changed.append(item)
            if not tie_only:
                unexpected.append(item)
    status = "PASS" if not unexpected else "FAIL"
    return {
        "schema_version": "shadow_ranking_preservation_report_v1",
        "status": status,
        "groups_checked": len(before_groups),
        "changed_group_count": len(changed),
        "unexpected_changed_group_count": len(unexpected),
        "changed_groups": changed[:25],
        "method": CALIBRATION_METHOD_KEY,
        "tie_policy": "stable input-order tie break; exact probability ties are documented",
    }


def logistic_calibration_slope_intercept(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    pairs = []
    eps = 1e-12
    for label, probability in zip(labels, probabilities):
        p = min(max(float(probability), eps), 1.0 - eps)
        pairs.append((int(label), math.log(p / (1.0 - p))))
    positive = sum(label for label, _ in pairs)
    if len(pairs) < 10 or positive == 0 or positive == len(pairs):
        return {
            "status": "NOT_RUN",
            "reason": "insufficient_label_variation",
            "rows": len(pairs),
            "positive_labels": positive,
            "method": "pure_python_logistic_regression_y_on_logit_p",
            "slope": None,
            "intercept": None,
        }

    beta0 = 0.0
    beta1 = 1.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for y, x in pairs:
            z = beta0 + beta1 * x
            if z >= 0:
                ez = math.exp(-z)
                pred = 1.0 / (1.0 + ez)
            else:
                ez = math.exp(z)
                pred = ez / (1.0 + ez)
            weight = max(pred * (1.0 - pred), 1e-12)
            residual = y - pred
            g0 += residual
            g1 += residual * x
            h00 -= weight
            h01 -= weight * x
            h11 -= weight * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            return {
                "status": "NOT_RUN",
                "reason": "singular_hessian",
                "rows": len(pairs),
                "positive_labels": positive,
                "method": "pure_python_logistic_regression_y_on_logit_p",
                "slope": None,
                "intercept": None,
            }
        step0 = (h11 * g0 - h01 * g1) / det
        step1 = (-h01 * g0 + h00 * g1) / det
        beta0 -= step0
        beta1 -= step1
        if abs(step0) + abs(step1) < 1e-9:
            break
    return {
        "status": "RUN",
        "reason": None,
        "rows": len(pairs),
        "positive_labels": positive,
        "method": "pure_python_logistic_regression_y_on_logit_p",
        "slope": beta1,
        "intercept": beta0,
    }


def reliability_bins(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bin_count: int = 10,
) -> list[dict[str, Any]]:
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for label, probability in zip(labels, probabilities):
        p = min(max(float(probability), 0.0), 1.0)
        index = min(int(p * bin_count), bin_count - 1)
        buckets[index].append((p, int(label)))
    output = []
    for index in range(bin_count):
        items = buckets.get(index) or []
        if not items:
            continue
        probs = [probability for probability, _ in items]
        ys = [label for _, label in items]
        avg_prob = mean(probs)
        win_rate = mean([float(label) for label in ys])
        output.append(
            {
                "bin": f"{index / bin_count:.1f}-{(index + 1) / bin_count:.1f}",
                "count": len(items),
                "fraction": len(items) / len(probabilities) if probabilities else 0.0,
                "mean_probability": avg_prob,
                "win_rate": win_rate,
                "abs_error": abs((avg_prob or 0.0) - (win_rate or 0.0)),
            }
        )
    return output


def score_grouped_metrics(rows: Sequence[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("shadow_race_group_id") or group_key(row))].append(row)
    top1 = 0
    top3 = 0
    winner_ranks: list[int] = []
    winner_rank_distribution: Counter[str] = Counter()
    top_boxes: Counter[str] = Counter()
    labels: list[int] = []
    probabilities: list[float] = []
    eps = 1e-12
    brier_values: list[float] = []
    log_values: list[float] = []
    for group_rows in groups.values():
        ordered = sorted(
            group_rows,
            key=lambda row: (-(safe_float(row.get(probability_key)) or 0.0), str(row.get("dog_name") or "")),
        )
        if not ordered:
            continue
        top_boxes[str(ordered[0].get("box_number"))] += 1
        for index, row in enumerate(ordered, start=1):
            y = int(row.get("actual_win") or 0)
            p = min(max(safe_float(row.get(probability_key)) or 0.0, eps), 1.0 - eps)
            labels.append(y)
            probabilities.append(p)
            brier_values.append((p - y) ** 2)
            log_values.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
            if y == 1:
                winner_ranks.append(index)
                winner_rank_distribution[str(index)] += 1
        if int(ordered[0].get("actual_win") or 0) == 1:
            top1 += 1
        if any(int(row.get("actual_win") or 0) == 1 for row in ordered[:3]):
            top3 += 1

    race_count = len(groups)
    bins = reliability_bins(labels, probabilities)
    ece = sum((item["fraction"] or 0.0) * (item["abs_error"] or 0.0) for item in bins)
    mce = max([item["abs_error"] or 0.0 for item in bins], default=0.0)
    return {
        "race_count": race_count,
        "row_count": len(rows),
        "positive_labels": sum(labels),
        "top1": top1 / race_count if race_count else 0.0,
        "top3": top3 / race_count if race_count else 0.0,
        "winner_rank": winner_ranks,
        "winner_rank_distribution": dict(sorted(winner_rank_distribution.items(), key=lambda item: int(item[0]))),
        "mean_winner_rank": mean([float(rank) for rank in winner_ranks]),
        "brier": mean(brier_values),
        "log_loss": mean(log_values),
        "calibration_slope_intercept": logistic_calibration_slope_intercept(labels, probabilities),
        "reliability_bins": bins,
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "probability_sum_error": probability_sum_report(rows, probability_key),
        "box_bias": {
            "box1_top_pick_share": top_boxes.get("1", 0) / race_count if race_count else 0.0,
            "top_pick_box_distribution": dict(sorted(top_boxes.items(), key=lambda item: item[0])),
            "ranking_concentration_top_box_share": top_boxes.most_common(1)[0][1] / race_count
            if race_count and top_boxes
            else 0.0,
        },
    }


def compare_metrics(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    metrics_higher_better = ("top1", "top3")
    metrics_lower_better = ("brier", "log_loss", "expected_calibration_error", "maximum_calibration_error", "mean_winner_rank")
    deltas: dict[str, Any] = {}
    beats: dict[str, bool] = {}
    for key in metrics_higher_better:
        av = safe_float(a.get(key))
        bv = safe_float(b.get(key))
        deltas[key] = None if av is None or bv is None else av - bv
        beats[key] = bool(av is not None and bv is not None and av >= bv)
    for key in metrics_lower_better:
        av = safe_float(a.get(key))
        bv = safe_float(b.get(key))
        deltas[key] = None if av is None or bv is None else av - bv
        beats[key] = bool(av is not None and bv is not None and av <= bv)
    a_box1 = safe_float((a.get("box_bias") or {}).get("box1_top_pick_share"))
    b_box1 = safe_float((b.get("box_bias") or {}).get("box1_top_pick_share"))
    return {
        "deltas_a_minus_b": deltas,
        "a_beats_or_matches_b": beats,
        "box1_share_delta": None if a_box1 is None or b_box1 is None else a_box1 - b_box1,
        "box1_not_worse": bool(a_box1 is not None and b_box1 is not None and a_box1 <= b_box1),
    }


def model_metadata(
    *,
    output_dir: Path,
    schema: Mapping[str, Any],
    training_report: Mapping[str, Any],
) -> dict[str, Any]:
    features = list(schema.get("feature_columns") or [])
    model_path = output_dir / "shadow_randomforest_model.joblib"
    return {
        "schema_version": "shadow_model_metadata_v1",
        "model_version": f"shadow_randomforest_{CALIBRATION_METHOD_KEY}_{now_id()}",
        "model_family": SHADOW_MODEL_FAMILY,
        "output_mode": SHADOW_OUTPUT_MODE,
        "shadow_only": True,
        "tgr_enabled": False,
        "calibration_method": CALIBRATION_METHOD_KEY,
        "power_gamma": POWER_GAMMA,
        "feature_count": len(features),
        "schema_feature_count": len(features),
        "active_feature_count": training_report.get("active_feature_count")
        or training_report.get("feature_count"),
        "inactive_feature_count": training_report.get("inactive_feature_count", 0),
        "inactive_features_due_to_train_all_missing": list(
            training_report.get("inactive_features_due_to_train_all_missing") or []
        ),
        "all_missing_train_policy": training_report.get("all_missing_train_policy") or "report_only",
        "feature_columns": features,
        "schema_version_source": schema.get("schema_version"),
        "schema_feature_columns_sha256": hashlib.sha256(
            json.dumps(features, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "artifact_path": shadow_relpath(model_path) if model_path.exists() else None,
        "artifact_sha256": sha256_file(model_path) if model_path.exists() else None,
        "training_report_status": training_report.get("status"),
        "train_rows": training_report.get("train_rows"),
        "train_races": training_report.get("train_races"),
        "holdout_rows": training_report.get("holdout_rows"),
        "holdout_races": training_report.get("holdout_races"),
        "registry_mutation": False,
        "promotion_allowed": False,
        "production_pointer_update": False,
        "active_model_replacement": False,
    }


def active_features_for_loaded_model(
    *,
    model_path: Path,
    schema: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    features = list(schema["feature_columns"])
    training_report_path = model_path.parent / "shadow_training_report.json"
    if not training_report_path.exists():
        return features, {
            "schema_version": "loaded_shadow_model_feature_policy_v1",
            "source": "canonical_schema_default",
            "reason": "shadow_training_report_missing_next_to_model",
            "model_path": shadow_relpath(model_path),
            "training_report_path": shadow_relpath(training_report_path),
            "active_feature_count": len(features),
            "schema_feature_count": len(features),
            "inactive_features_due_to_train_all_missing": [],
        }

    report = load_json(training_report_path)
    inactive = list(report.get("inactive_features_due_to_train_all_missing") or [])
    inactive_set = set(inactive)
    active = [feature for feature in features if feature not in inactive_set]
    expected_active = safe_int(report.get("active_feature_count"))
    if expected_active is not None and expected_active != len(active):
        raise RuntimeError(
            "loaded_model_active_feature_metadata_mismatch:"
            f"expected={expected_active}:derived={len(active)}"
        )
    return active, {
        "schema_version": "loaded_shadow_model_feature_policy_v1",
        "source": shadow_relpath(training_report_path),
        "reason": "shadow_training_report_loaded_next_to_model",
        "model_path": shadow_relpath(model_path),
        "active_feature_count": len(active),
        "schema_feature_count": len(features),
        "inactive_features_due_to_train_all_missing": inactive,
        "all_missing_train_policy": report.get("all_missing_train_policy") or "report_only",
    }


def replay_shadow_evaluation(
    *,
    pipeline: Any,
    dataset: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    features = list(dataset["features"])
    holdout_rows = list(dataset["holdout_rows"])
    x_holdout, _ = prepare_xy(holdout_rows, features)
    raw_probabilities = [float(value) for value in pipeline.predict_proba(x_holdout)[:, 1]]

    uncalibrated_rows = normalize_probabilities_by_race(
        holdout_rows,
        raw_probabilities,
        "shadow_rf_uncalibrated_probability",
        "shadow_rf_uncalibrated_rank",
    )
    calibrated_rows = apply_power_gamma_by_race(
        uncalibrated_rows,
        input_key="shadow_rf_uncalibrated_probability",
        output_key="shadow_rf_calibrated_probability",
        output_rank_key="shadow_rf_calibrated_rank",
    )
    champion_rows = normalize_probabilities_by_race(
        holdout_rows,
        [safe_float(row.get("champion_win_prob_norm")) or 0.0 for row in holdout_rows],
        "champion_probability",
        "champion_rank",
    )

    ranking_report = ranking_preservation_report(
        uncalibrated_rows,
        calibrated_rows,
        before_key="shadow_rf_uncalibrated_probability",
        after_key="shadow_rf_calibrated_probability",
    )
    if ranking_report["status"] != "PASS":
        raise RuntimeError("calibration_changed_rankings_unexpectedly")

    replay_predictions = []
    calibrated_by_key = {
        (row.get("shadow_race_group_id"), clean_name(row.get("dog_name")), str(row.get("box_number") or "")): row
        for row in calibrated_rows
    }
    champion_by_key = {
        (row.get("shadow_race_group_id"), clean_name(row.get("dog_name")), str(row.get("box_number") or "")): row
        for row in champion_rows
    }
    for row in uncalibrated_rows:
        key = (row.get("shadow_race_group_id"), clean_name(row.get("dog_name")), str(row.get("box_number") or ""))
        cal = calibrated_by_key[key]
        champ = champion_by_key[key]
        replay_predictions.append(
            {
                "race_id": row.get("race_id"),
                "shadow_race_group_id": row.get("shadow_race_group_id"),
                "race_date": row.get("race_date"),
                "dog_name": row.get("dog_name"),
                "box_number": row.get("box_number"),
                "actual_win": row.get("actual_win"),
                "finish_position": row.get("finish_position"),
                "champion_probability": champ.get("champion_probability"),
                "champion_rank": champ.get("champion_rank"),
                "shadow_rf_uncalibrated_probability": row.get("shadow_rf_uncalibrated_probability"),
                "shadow_rf_uncalibrated_rank": row.get("shadow_rf_uncalibrated_rank"),
                "shadow_rf_calibrated_probability": cal.get("shadow_rf_calibrated_probability"),
                "shadow_rf_calibrated_rank": cal.get("shadow_rf_calibrated_rank"),
                "calibration_method": CALIBRATION_METHOD_KEY,
                "tgr_enabled": False,
            }
        )
    write_csv(
        output_dir / "shadow_replay_predictions.csv",
        replay_predictions,
        [
            "race_id",
            "shadow_race_group_id",
            "race_date",
            "dog_name",
            "box_number",
            "actual_win",
            "finish_position",
            "champion_probability",
            "champion_rank",
            "shadow_rf_uncalibrated_probability",
            "shadow_rf_uncalibrated_rank",
            "shadow_rf_calibrated_probability",
            "shadow_rf_calibrated_rank",
            "calibration_method",
            "tgr_enabled",
        ],
    )
    write_json(output_dir / "shadow_replay_predictions.json", replay_predictions)

    champion_metrics = score_grouped_metrics(champion_rows, "champion_probability")
    uncalibrated_metrics = score_grouped_metrics(
        uncalibrated_rows, "shadow_rf_uncalibrated_probability"
    )
    calibrated_metrics = score_grouped_metrics(
        calibrated_rows, "shadow_rf_calibrated_probability"
    )
    return {
        "champion_rows": champion_rows,
        "uncalibrated_rows": uncalibrated_rows,
        "calibrated_rows": calibrated_rows,
        "ranking_report": ranking_report,
        "probability_sum_report": {
            "champion": champion_metrics["probability_sum_error"],
            "shadow_uncalibrated_rf": uncalibrated_metrics["probability_sum_error"],
            "shadow_calibrated_rf_power_gamma_2_4": calibrated_metrics["probability_sum_error"],
        },
        "metrics": {
            "champion_baseline": champion_metrics,
            "shadow_uncalibrated_rf": uncalibrated_metrics,
            "shadow_calibrated_rf_power_gamma_2_4": calibrated_metrics,
        },
    }


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {"count": 0}
    def pick(q: float) -> float:
        if len(clean) == 1:
            return clean[0]
        index = q * (len(clean) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return clean[lower]
        weight = index - lower
        return clean[lower] * (1 - weight) + clean[upper] * weight
    return {
        "count": len(clean),
        "min": clean[0],
        "p10": pick(0.10),
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "max": clean[-1],
        "mean": mean(clean),
    }


def box_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("box_number") or "") for row in rows).items()))


def feature_population_drift(
    train_population: Mapping[str, Any],
    holdout_population: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train = train_population.get("by_feature") or {}
    holdout = holdout_population.get("by_feature") or {}
    output = []
    for feature in sorted(set(train).union(holdout)):
        train_pct = safe_float((train.get(feature) or {}).get("present_pct")) or 0.0
        holdout_pct = safe_float((holdout.get(feature) or {}).get("present_pct")) or 0.0
        output.append(
            {
                "feature": feature,
                "train_present_pct": train_pct,
                "holdout_present_pct": holdout_pct,
                "holdout_minus_train_present_pct": holdout_pct - train_pct,
            }
        )
    return output


def build_monitoring_reports(
    *,
    dataset: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_rows = list(dataset["train_rows"])
    holdout_rows = list(dataset["holdout_rows"])
    metrics = replay["metrics"]
    probability_distributions = {
        "champion": quantiles(
            [safe_float(row.get("champion_probability")) or 0.0 for row in replay["champion_rows"]]
        ),
        "shadow_uncalibrated_rf": quantiles(
            [
                safe_float(row.get("shadow_rf_uncalibrated_probability")) or 0.0
                for row in replay["uncalibrated_rows"]
            ]
        ),
        "shadow_calibrated_rf_power_gamma_2_4": quantiles(
            [
                safe_float(row.get("shadow_rf_calibrated_probability")) or 0.0
                for row in replay["calibrated_rows"]
            ]
        ),
    }
    monitoring = {
        "schema_version": "shadow_monitoring_report_v1",
        "output_mode": SHADOW_OUTPUT_MODE,
        "race_count_summary": {
            "train_races": len({row.get("race_id") for row in train_rows}),
            "holdout_races": len({row.get("race_id") for row in holdout_rows}),
        },
        "runner_count_summary": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
        },
        "feature_population_drift": feature_population_drift(
            dataset["train_population"], dataset["holdout_population"]
        ),
        "missingness_drift": feature_population_drift(
            dataset["train_population"], dataset["holdout_population"]
        ),
        "box_distribution_drift": {
            "train": box_distribution(train_rows),
            "holdout": box_distribution(holdout_rows),
        },
        "calibration_drift": {
            "champion": metrics["champion_baseline"]["calibration_slope_intercept"],
            "shadow_uncalibrated_rf": metrics["shadow_uncalibrated_rf"]["calibration_slope_intercept"],
            "shadow_calibrated_rf_power_gamma_2_4": metrics[
                "shadow_calibrated_rf_power_gamma_2_4"
            ]["calibration_slope_intercept"],
        },
        "probability_distribution_drift": probability_distributions,
    }
    drift = {
        "schema_version": "shadow_drift_report_v1",
        "status": "REPORT_ONLY",
        "feature_population_drift": monitoring["feature_population_drift"],
        "missingness_drift": monitoring["missingness_drift"],
        "box_distribution_drift": monitoring["box_distribution_drift"],
        "calibration_drift": monitoring["calibration_drift"],
        "probability_distribution_drift": probability_distributions,
        "race_count_summary": monitoring["race_count_summary"],
        "runner_count_summary": monitoring["runner_count_summary"],
    }
    return monitoring, drift


def output_file_manifest(output_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "implementation_file_manifest.json":
            continue
        files[shadow_relpath(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "shadow_implementation_file_manifest_v1",
        "output_dir": shadow_relpath(output_dir),
        "git_head": git_output(["rev-parse", "--short=12", "HEAD"]),
        "git_branch": git_output(["branch", "--show-current"]),
        "implementation_files": [
            "scripts/run_shadow_non_tgr_rf_evaluation.py",
            "tests/test_run_shadow_non_tgr_rf_evaluation.py",
        ],
        "artifact_files": files,
    }


def write_summary(
    output_dir: Path,
    *,
    final_status: str,
    feature_audit: Mapping[str, Any] | None,
    parity_report: Mapping[str, Any] | None,
    training_report: Mapping[str, Any] | None,
    replay_metrics: Mapping[str, Any] | None,
    protected: Mapping[str, Any],
) -> None:
    lines = [
        "# Shadow Evaluation Implementation",
        "",
        f"Final verdict: `{final_status}`",
        "",
        "Scope: shadow-only repaired non-TGR RandomForest with locked `power_gamma_2.4` calibration.",
        "",
        f"Feature matrix audit: `{None if feature_audit is None else feature_audit.get('status')}`.",
        f"Train/eval parity policy: `{None if parity_report is None else parity_report.get('policy')}`.",
        f"All-missing train features: `{None if parity_report is None else parity_report.get('all_missing_train_feature_count')}`.",
        f"All-missing train / present holdout features: `{None if parity_report is None else parity_report.get('all_missing_train_present_holdout_feature_count')}`.",
        f"Training status: `{None if training_report is None else training_report.get('status')}`.",
        f"Protected paths unchanged: `{protected.get('protected_paths_unchanged')}`.",
        "",
        "Forbidden production actions performed: `False`.",
        "",
        "No production promotion, registry mutation, production pointer update, active model replacement, DB write, label write, snapshot rewrite, existing manifest rewrite, TGR enablement, betting action, EV action, production endpoint mutation, or champion artifact overwrite was performed.",
    ]
    if replay_metrics:
        cal = replay_metrics.get("shadow_calibrated_rf_power_gamma_2_4") or {}
        champ = replay_metrics.get("champion_baseline") or {}
        lines.extend(
            [
                "",
                "Replay headline:",
                "",
                f"- Champion Top1 `{champ.get('top1')}`, Top3 `{champ.get('top3')}`, Brier `{champ.get('brier')}`, LogLoss `{champ.get('log_loss')}`.",
                f"- Shadow calibrated RF Top1 `{cal.get('top1')}`, Top3 `{cal.get('top3')}`, Brier `{cal.get('brier')}`, LogLoss `{cal.get('log_loss')}`.",
            ]
        )
    write_text(output_dir / "SUMMARY.md", "\n".join(lines) + "\n")


def rollback_plan_text() -> str:
    return "\n".join(
        [
            "# Rollback / No-Op Plan",
            "",
            "This implementation is shadow-only.",
            "",
            "- Remove the generated shadow output directory if the evidence artifacts are not needed.",
            "- Revert `scripts/run_shadow_non_tgr_rf_evaluation.py` and `tests/test_run_shadow_non_tgr_rf_evaluation.py` from Git if the implementation is not accepted.",
            "- No DB restore is needed.",
            "- No registry restore is needed.",
            "- No production pointer restore is needed.",
            "- No snapshot or manifest restore is needed.",
            "",
        ]
    )


def run_shadow(args: argparse.Namespace) -> int:
    ensure_shadow_runtime_guard()
    output_dir = prepare_output_dir(args.output_dir or default_implementation_output_dir())
    protected_before = protected_path_snapshot()
    final_status = "IMPLEMENTATION_ABORTED"
    feature_audit: dict[str, Any] | None = None
    parity_report: dict[str, Any] | None = None
    training_report: dict[str, Any] | None = None
    replay_metrics: dict[str, Any] | None = None
    protected = protected_path_verification(protected_before)

    try:
        schema = load_json(args.schema)
        write_shadow_candidate_definition(output_dir, schema, args.schema)
        if args.stop_after_definition:
            final_status = "PARTIAL_SHADOW_IMPLEMENTATION_NEEDS_FIXES"
            return 0

        dataset, feature_audit, population = build_shadow_feature_matrix(
            clean_dataset=args.clean_dataset,
            repaired_packet=args.repaired_packet,
            schema_path=args.schema,
            db_path=args.db,
        )
        write_json(output_dir / "shadow_feature_matrix_audit.json", feature_audit)
        write_json(output_dir / "shadow_feature_population_report.json", population)
        parity_report = train_eval_feature_parity_report(
            dataset,
            policy=args.all_missing_train_policy,
        )
        policy_report = inactive_feature_policy_report(parity_report)
        write_json(output_dir / "train_eval_feature_parity_report.json", parity_report)
        write_json(output_dir / "inactive_feature_policy_report.json", policy_report)
        if feature_audit["status"] != "PASS":
            final_status = "BLOCKED_BY_FEATURE_MATRIX"
            return 2
        if args.stop_after_audit:
            final_status = "PARTIAL_SHADOW_IMPLEMENTATION_NEEDS_FIXES"
            return 0
        dataset_for_model = dataset_with_all_missing_train_policy(dataset, parity_report)

        pipeline, training_report = train_or_load_shadow_rf(
            dataset=dataset_for_model,
            output_dir=output_dir,
            load_model=args.load_model,
        )
        write_json(output_dir / "shadow_training_report.json", training_report)
        metadata = model_metadata(output_dir=output_dir, schema=schema, training_report=training_report)
        write_json(output_dir / "shadow_model_metadata.json", metadata)
        if pipeline is None or training_report["status"] != "PASS":
            final_status = "PARTIAL_SHADOW_IMPLEMENTATION_NEEDS_FIXES"
            return 2

        replay = replay_shadow_evaluation(pipeline=pipeline, dataset=dataset_for_model, output_dir=output_dir)
        replay_metrics = replay["metrics"]
        comparison = {
            "schema_version": "champion_vs_shadow_comparison_v1",
            "shadow_calibrated_vs_champion": compare_metrics(
                replay_metrics["shadow_calibrated_rf_power_gamma_2_4"],
                replay_metrics["champion_baseline"],
            ),
            "shadow_uncalibrated_vs_champion": compare_metrics(
                replay_metrics["shadow_uncalibrated_rf"],
                replay_metrics["champion_baseline"],
            ),
            "shadow_calibrated_vs_uncalibrated": compare_metrics(
                replay_metrics["shadow_calibrated_rf_power_gamma_2_4"],
                replay_metrics["shadow_uncalibrated_rf"],
            ),
            "promotion_allowed": False,
            "registry_mutation": False,
        }
        calibration_report = {
            "schema_version": "shadow_calibration_report_v1",
            "method_key": CALIBRATION_METHOD_KEY,
            "gamma": POWER_GAMMA,
            "formula": "p_i_cal = p_i^2.4 / sum_j(p_j^2.4)",
            "rank_preserving_by_construction": True,
            "ranking_preservation_status": replay["ranking_report"]["status"],
            "probability_sum_status": replay["probability_sum_report"][
                "shadow_calibrated_rf_power_gamma_2_4"
            ]["status"],
        }
        diagnostics = {
            "schema_version": "shadow_calibration_diagnostics_v1",
            "champion": {
                "calibration_slope_intercept": replay_metrics["champion_baseline"][
                    "calibration_slope_intercept"
                ],
                "reliability_bins": replay_metrics["champion_baseline"]["reliability_bins"],
            },
            "shadow_uncalibrated_rf": {
                "calibration_slope_intercept": replay_metrics["shadow_uncalibrated_rf"][
                    "calibration_slope_intercept"
                ],
                "reliability_bins": replay_metrics["shadow_uncalibrated_rf"]["reliability_bins"],
            },
            "shadow_calibrated_rf_power_gamma_2_4": {
                "calibration_slope_intercept": replay_metrics[
                    "shadow_calibrated_rf_power_gamma_2_4"
                ]["calibration_slope_intercept"],
                "reliability_bins": replay_metrics["shadow_calibrated_rf_power_gamma_2_4"][
                    "reliability_bins"
                ],
            },
        }
        box_bias = {
            "schema_version": "shadow_box_bias_report_v1",
            "champion": replay_metrics["champion_baseline"]["box_bias"],
            "shadow_uncalibrated_rf": replay_metrics["shadow_uncalibrated_rf"]["box_bias"],
            "shadow_calibrated_rf_power_gamma_2_4": replay_metrics[
                "shadow_calibrated_rf_power_gamma_2_4"
            ]["box_bias"],
        }
        monitoring, drift = build_monitoring_reports(dataset=dataset_for_model, replay=replay)
        write_json(output_dir / "shadow_calibration_report.json", calibration_report)
        write_json(output_dir / "shadow_probability_sum_report.json", replay["probability_sum_report"])
        write_json(output_dir / "shadow_ranking_preservation_report.json", replay["ranking_report"])
        write_json(output_dir / "shadow_replay_metrics.json", replay_metrics)
        write_json(output_dir / "champion_vs_shadow_comparison.json", comparison)
        write_json(output_dir / "shadow_box_bias_report.json", box_bias)
        write_json(output_dir / "shadow_calibration_diagnostics.json", diagnostics)
        write_json(output_dir / "shadow_monitoring_report.json", monitoring)
        write_json(output_dir / "shadow_drift_report.json", drift)
        final_status = "SHADOW_EVALUATION_IMPLEMENTED_READY_FOR_SHADOW_RUNS"
        return 0
    except RuntimeError as exc:
        text = str(exc)
        if text.startswith("schema_contract_failed"):
            final_status = "BLOCKED_BY_SCHEMA_REPRODUCTION"
        elif text.startswith("all_missing_train_policy_failed"):
            final_status = "BLOCKED_BY_TRAIN_EVAL_FEATURE_PARITY"
        elif "calibration_changed_rankings" in text:
            final_status = "PARTIAL_SHADOW_IMPLEMENTATION_NEEDS_FIXES"
        else:
            final_status = "PARTIAL_SHADOW_IMPLEMENTATION_NEEDS_FIXES"
        write_json(output_dir / "shadow_runtime_error.json", {"error": repr(exc)})
        return 2
    finally:
        protected = protected_path_verification(protected_before)
        write_json(output_dir / "protected_path_verification.json", protected)
        if not protected.get("protected_paths_unchanged"):
            final_status = "IMPLEMENTATION_ABORTED"
        write_text(output_dir / "rollback_or_noop_plan.md", rollback_plan_text())
        write_text(
            output_dir / "verification_results.txt",
            "\n".join(
                [
                    "py_compile=NOT_RUN_BY_SCRIPT",
                    "targeted_pytest=NOT_RUN_BY_SCRIPT",
                    "static_forbidden_path_grep=NOT_RUN_BY_SCRIPT",
                    f"protected_paths_unchanged={protected.get('protected_paths_unchanged')}",
                    f"candidate_definition_exists={(output_dir / 'shadow_candidate_definition.json').exists()}",
                ]
            )
            + "\n",
        )
        write_text(output_dir / "test_results.txt", "NOT_RUN_BY_SCRIPT\n")
        write_text(output_dir / "final_status.txt", final_status + "\n")
        write_summary(
            output_dir,
            final_status=final_status,
            feature_audit=feature_audit,
            parity_report=parity_report,
            training_report=training_report,
            replay_metrics=replay_metrics,
            protected=protected,
        )
        write_json(output_dir / "implementation_file_manifest.json", output_file_manifest(output_dir))


def csv_value(row: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower().strip(): value for key, value in row.items()}
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def parse_live_runner_identity(raw_dog_name: Any, raw_box: Any) -> tuple[str, int | None]:
    dog_name_text = str(raw_dog_name or "").strip()
    if not dog_name_text:
        return "", None
    prefix_match = LIVE_RUNNER_PREFIX_RE.match(dog_name_text)
    if prefix_match:
        return prefix_match.group(2).strip(), safe_int(prefix_match.group(1))
    return dog_name_text, safe_int(raw_box)


def load_live_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header = handle.readline()
        delimiter = "|" if "|" in header else ","
        handle.seek(0)
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def live_form_history_by_dog(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_race_date: str | None,
) -> dict[str, list[dict[str, Any]]]:
    """Parse pre-jump expert-form history rows embedded in the live CSV."""

    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current_dog = ""
    for raw in rows:
        dog_cell = csv_value(raw, "dog_name", "Dog Name", "dog", "runner", "runner_name")
        if dog_cell not in (None, ""):
            current_dog, _box = parse_live_runner_identity(dog_cell, csv_value(raw, "box", "BOX"))
        if not current_dog:
            continue
        row_date = parse_date(csv_value(raw, "DATE", "date", "race_date"))
        if not row_date:
            continue
        if target_race_date and row_date >= target_race_date:
            continue
        dog_key = clean_name(current_dog)
        history_row = {
            "dog_name": current_dog,
            "dog_key": dog_key,
            "race_date": row_date,
            "venue": str(csv_value(raw, "TRACK", "track", "venue") or "").strip(),
            "distance": csv_value(raw, "DIST", "distance"),
            "distance_num": safe_float(csv_value(raw, "DIST", "distance")),
            "grade": csv_value(raw, "G", "grade"),
            "grade_normalized": normalize_grade(csv_value(raw, "G", "grade")),
            "finish_position": csv_value(raw, "PLC", "placing", "finish_position"),
            "finish_num": safe_int(csv_value(raw, "PLC", "placing", "finish_position")),
            "box_number": safe_int(csv_value(raw, "BOX", "box")),
            "weight": csv_value(raw, "WGT", "weight"),
            "weight_num": safe_float(csv_value(raw, "WGT", "weight")),
            "individual_time": csv_value(raw, "TIME", "time"),
            "time_num": safe_float(csv_value(raw, "TIME", "time")),
            "sectional_1st": csv_value(raw, "1 SEC", "sectional_1st"),
            "sectional_1st_num": safe_float(csv_value(raw, "1 SEC", "sectional_1st")),
            "beaten_margin": csv_value(raw, "MGN", "margin", "beaten_margin"),
            "margin_num": safe_float(csv_value(raw, "MGN", "margin", "beaten_margin")),
            "history_source": "prejump_form_history",
        }
        history[dog_key].append(history_row)
    for dog_rows in history.values():
        dog_rows.sort(
            key=lambda row: (
                str(row.get("race_date") or ""),
                safe_int(row.get("box_number")) or 0,
                str(row.get("venue") or ""),
            )
        )
    return dict(history)


def merge_prior_history_rows(
    db_rows: Sequence[Mapping[str, Any]],
    form_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    merged: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in list(db_rows) + list(form_rows):
        key = (
            row.get("race_date"),
            str(row.get("venue") or "").upper(),
            safe_float(row.get("distance_num")),
            normalize_grade(row.get("grade_normalized") or row.get("grade")),
            safe_float(row.get("time_num")),
            safe_int(row.get("finish_num")),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = row
            continue
        if safe_float(existing.get("time_num")) is None and safe_float(row.get("time_num")) is not None:
            merged[key] = row
    out = list(merged.values())
    out.sort(
        key=lambda row: (
            str(row.get("race_date") or ""),
            safe_int(row.get("race_number")) or 0,
            str(row.get("race_id") or ""),
            str(row.get("venue") or ""),
        )
    )
    return out


def race_time_iso_from_sidecar(race_date: Any, race_time: Any) -> str | None:
    date_text = str(race_date or "").strip()
    time_text = str(race_time or "").strip()
    if not time_text:
        return None

    iso_candidate = time_text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone(timedelta(hours=10)))
        return parsed.isoformat(timespec="seconds")
    except Exception:
        pass

    if not date_text:
        return None
    for fmt in ("%I:%M %p", "%H:%M", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(time_text.upper(), fmt).time()
            parsed = datetime.fromisoformat(date_text).replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=parsed_time.second,
                microsecond=0,
                tzinfo=timezone(timedelta(hours=10)),
            )
            return parsed.isoformat(timespec="seconds")
        except Exception:
            continue
    return None


def load_live_sidecar_context(path: Path) -> dict[str, Any]:
    safe_target = load_safe_sidecar_target_metadata(path)
    safe_weather_track = load_safe_weather_track_metadata(path)
    context: dict[str, Any] = {
        "target_distance": safe_target.get("target_distance"),
        "target_distance_source": safe_target.get("target_distance_source"),
        "target_grade": safe_target.get("target_grade"),
        "target_grade_source": safe_target.get("target_grade_source"),
        "metadata_is_leakage_safe": bool(safe_target.get("metadata_is_leakage_safe")),
        "metadata_source_url": safe_target.get("metadata_source_url"),
        "race_info": {},
        "rejected_metadata_sources": list(safe_target.get("rejected_metadata_sources") or []),
        "track_condition": safe_weather_track.get("track_condition"),
        "weather": safe_weather_track.get("weather"),
        "weather_condition": safe_weather_track.get("weather_condition"),
        "weather_track_metadata_source": safe_weather_track.get(
            "weather_track_metadata_source"
        ),
        "weather_track_metadata_is_leakage_safe": bool(
            safe_weather_track.get("weather_track_metadata_is_leakage_safe")
        ),
        "weather_track_metadata_source_url": safe_weather_track.get(
            "weather_track_metadata_source_url"
        )
        or safe_weather_track.get("metadata_source_url"),
        "weather_track_metadata_captured_at": safe_weather_track.get(
            "metadata_captured_at"
        ),
        "weather_track_metadata_race_date": safe_weather_track.get("race_date"),
        "weather_track_metadata_race_time": safe_weather_track.get("race_time"),
        "rejected_weather_track_metadata_sources": list(
            safe_weather_track.get("rejected_weather_track_metadata_sources") or []
        ),
    }
    sidecar_path = Path(f"{path}.metadata.json")
    if not sidecar_path.exists():
        return context
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        context["rejected_metadata_sources"].append("sidecar_unreadable")
        return context
    if not isinstance(payload, Mapping):
        context["rejected_metadata_sources"].append("sidecar_not_object")
        return context
    context["expert_form_metadata"] = safe_expert_form_metadata_from_payload(payload)
    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    if payload.get("metadata_is_leakage_safe") is True:
        context["race_info"] = {
            key: race_info.get(key)
            for key in (
                "date",
                "venue",
                "race_number",
                "race_time",
                "url",
                "race_time_mapping_status",
                "race_time_source",
            )
            if race_info.get(key) not in (None, "")
        }
        context["metadata_source_url"] = (
            context.get("metadata_source_url")
            or payload.get("metadata_source_url")
            or payload.get("race_url")
            or race_info.get("url")
        )
    elif race_info:
        context["rejected_metadata_sources"].append("sidecar_race_info_not_leakage_safe")
    return context


def expert_form_runner_features(
    *,
    expert_form_metadata: Mapping[str, Any],
    dog_name: str,
    box_number: int | None,
) -> dict[str, Any]:
    """Flatten safe Expert Form sidecar metadata for one live runner row."""

    rejected = list(expert_form_metadata.get("rejected_reasons") or [])
    base: dict[str, Any] = {
        "expert_form_metadata_from_sidecar": False,
        "expert_form_metadata_source_url": expert_form_metadata.get("source_url"),
        "expert_form_metadata_captured_at": expert_form_metadata.get("captured_at"),
        "expert_form_metadata_rejected_reasons": rejected,
    }
    if expert_form_metadata.get("metadata_is_leakage_safe") is not True:
        return base
    runners = expert_form_metadata.get("runners")
    if not isinstance(runners, Sequence):
        return {
            **base,
            "expert_form_metadata_rejected_reasons": [
                *rejected,
                "expert_form_runners_not_sequence",
            ],
        }
    target_key = clean_name(dog_name)
    runner = next(
        (
            item
            for item in runners
            if isinstance(item, Mapping)
            and clean_name(item.get("dog_name")) == target_key
        ),
        None,
    )
    if not isinstance(runner, Mapping):
        return {
            **base,
            "expert_form_metadata_rejected_reasons": [
                *rejected,
                "expert_form_runner_not_found",
            ],
        }

    career = runner.get("career") if isinstance(runner.get("career"), Mapping) else {}
    track_distance = (
        runner.get("track_distance")
        if isinstance(runner.get("track_distance"), Mapping)
        else {}
    )
    greyhound = (
        runner.get("greyhound") if isinstance(runner.get("greyhound"), Mapping) else {}
    )
    trainer = runner.get("trainer") if isinstance(runner.get("trainer"), Mapping) else {}
    best_other = runner.get("best_win_times_other_tracks")
    if not isinstance(best_other, list):
        best_other = []
    best_other_times = [
        safe_float(item.get("time"))
        for item in best_other
        if isinstance(item, Mapping) and safe_float(item.get("time")) is not None
    ]
    distance_counts = (
        runner.get("winning_distance_counts")
        if isinstance(runner.get("winning_distance_counts"), Mapping)
        else {}
    )
    box_history = (
        runner.get("box_history")
        if isinstance(runner.get("box_history"), Mapping)
        else {}
    )
    current_box_history = (
        box_history.get(str(box_number))
        if box_number is not None and isinstance(box_history.get(str(box_number)), Mapping)
        else {}
    )
    return {
        **base,
        "expert_form_metadata_from_sidecar": True,
        "expert_form_grade": runner.get("grade"),
        "expert_form_trainer_name": trainer.get("name"),
        "expert_form_trainer_district": trainer.get("district"),
        "expert_form_owner": runner.get("owner"),
        "expert_form_colour": greyhound.get("colour"),
        "expert_form_sex": greyhound.get("sex"),
        "expert_form_date_of_birth": greyhound.get("date_of_birth"),
        "expert_form_sire": greyhound.get("sire"),
        "expert_form_dam": greyhound.get("dam"),
        "expert_form_career_starts": safe_int(career.get("starts")),
        "expert_form_career_wins": safe_int(career.get("wins")),
        "expert_form_career_seconds": safe_int(career.get("seconds")),
        "expert_form_career_thirds": safe_int(career.get("thirds")),
        "expert_form_track_distance_starts": safe_int(track_distance.get("starts")),
        "expert_form_track_distance_wins": safe_int(track_distance.get("wins")),
        "expert_form_track_distance_seconds": safe_int(track_distance.get("seconds")),
        "expert_form_track_distance_thirds": safe_int(track_distance.get("thirds")),
        "expert_form_win_percent": safe_float(runner.get("win_percent")),
        "expert_form_place_percent": safe_float(runner.get("place_percent")),
        "expert_form_prize_money": safe_float(runner.get("prize_money")),
        "expert_form_track_distance_best_time": safe_float(track_distance.get("best_time")),
        "expert_form_track_distance_best_time_date": track_distance.get("best_time_date"),
        "expert_form_track_distance_best_first_split": safe_float(
            track_distance.get("best_first_split")
        ),
        "expert_form_best_other_track_count": len(best_other),
        "expert_form_best_other_track_min_time": min(best_other_times)
        if best_other_times
        else None,
        "expert_form_distance_wins_under_400": safe_int(distance_counts.get("<400")),
        "expert_form_distance_wins_400_plus": safe_int(distance_counts.get("400+")),
        "expert_form_distance_wins_500_plus": safe_int(distance_counts.get("500+")),
        "expert_form_distance_wins_600_plus": safe_int(distance_counts.get("600+")),
        "expert_form_distance_wins_700_plus": safe_int(distance_counts.get("700+")),
        "expert_form_current_box_starts": safe_int(current_box_history.get("starts")),
        "expert_form_current_box_wins": safe_int(current_box_history.get("wins")),
        "expert_form_current_box_places": safe_int(current_box_history.get("places")),
    }


def input_files_from_path(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*.csv") if item.is_file())
    raise ValueError(f"input_path_missing:{path}")


def same_distance_same_grade_history_rows(
    history_rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    target_distance = safe_float(target.get("distance"))
    target_grade = normalize_grade(target.get("grade"))
    if target_distance is None or not target_grade:
        return []
    matched: list[Mapping[str, Any]] = []
    for row in history_rows:
        distance = safe_float(row.get("distance_num"))
        if distance is None or abs(distance - target_distance) > 50.0:
            continue
        if normalize_grade(row.get("grade_normalized")) != target_grade:
            continue
        matched.append(row)
    return matched


def same_distance_same_grade_live_row_provenance(
    *,
    history_rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    matched_rows = same_distance_same_grade_history_rows(history_rows, target)
    time_rows = [row for row in matched_rows if safe_float(row.get("time_num")) is not None]
    status = "PASS" if time_rows else "NOT_POPULATED"
    return {
        "same_distance_same_grade_history_status": status,
        "same_distance_same_grade_history_source": SAME_DISTANCE_HISTORY_SOURCE
        if time_rows
        else None,
        "same_distance_same_grade_history_cutoff": SAME_DISTANCE_HISTORY_CUTOFF,
        "same_distance_same_grade_history_cutoff_basis": SAME_DISTANCE_HISTORY_CUTOFF_BASIS,
        "same_distance_same_grade_prior_history_rows_matched": len(matched_rows),
        "same_distance_same_grade_prior_history_rows_used": len(time_rows),
        "same_distance_same_grade_target_race_rows_used": 0,
        "same_distance_same_grade_post_outcome_rows_used": 0,
        "same_distance_same_grade_post_outcome_fields_used": [],
    }


def _present(value: Any) -> bool:
    return value not in (None, "")


def _int_from_row(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def same_distance_same_grade_history_provenance_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_feature: dict[str, Any] = {}
    report_failures: list[str] = []
    report_partial = False

    for feature in WATCHED_PARITY_FEATURES:
        populated_rows = [row for row in rows if _present(row.get(feature))]
        sources = sorted(
            {
                str(row.get("same_distance_same_grade_history_source") or "")
                for row in populated_rows
                if row.get("same_distance_same_grade_history_source") not in (None, "")
            }
        )
        cutoffs = sorted(
            {
                str(row.get("same_distance_same_grade_history_cutoff") or "")
                for row in populated_rows
                if row.get("same_distance_same_grade_history_cutoff") not in (None, "")
            }
        )
        cutoff_bases = sorted(
            {
                str(row.get("same_distance_same_grade_history_cutoff_basis") or "")
                for row in populated_rows
                if row.get("same_distance_same_grade_history_cutoff_basis") not in (None, "")
            }
        )
        post_outcome_fields = sorted(
            {
                str(field)
                for row in populated_rows
                for field in (row.get("same_distance_same_grade_post_outcome_fields_used") or [])
                if field not in (None, "")
            }
        )
        prior_history_rows_used = sum(
            _int_from_row(row, "same_distance_same_grade_prior_history_rows_used")
            for row in populated_rows
        )
        target_race_rows_used = sum(
            _int_from_row(row, "same_distance_same_grade_target_race_rows_used")
            for row in populated_rows
        )
        post_outcome_rows_used = sum(
            _int_from_row(row, "same_distance_same_grade_post_outcome_rows_used")
            for row in populated_rows
        )

        reasons: list[str] = []
        if not populated_rows:
            reasons.append("feature_not_populated")
        if populated_rows and sources != [SAME_DISTANCE_HISTORY_SOURCE]:
            reasons.append("history_source_not_prior_dog_history")
        if populated_rows and cutoffs != [SAME_DISTANCE_HISTORY_CUTOFF]:
            reasons.append("history_cutoff_not_strictly_before_target_race")
        if populated_rows and prior_history_rows_used <= 0:
            reasons.append("prior_history_rows_used_missing")
        if target_race_rows_used:
            reasons.append("target_race_rows_used")
        if post_outcome_rows_used:
            reasons.append("post_outcome_rows_used")
        if post_outcome_fields:
            reasons.append("post_outcome_fields_used")

        if not reasons:
            status = "PASS"
        elif reasons == ["feature_not_populated"]:
            status = "NOT_POPULATED"
            report_partial = True
        else:
            status = "FAIL"
            report_failures.extend(f"{feature}:{reason}" for reason in reasons)

        by_feature[feature] = {
            "status": status,
            "fail_reasons": reasons,
            "present_rows": len(populated_rows),
            "source": sources[0] if len(sources) == 1 else None,
            "sources": sources,
            "history_cutoff": cutoffs[0] if len(cutoffs) == 1 else None,
            "history_cutoffs": cutoffs,
            "history_cutoff_basis": cutoff_bases[0] if len(cutoff_bases) == 1 else None,
            "history_cutoff_bases": cutoff_bases,
            "prior_history_rows_used": prior_history_rows_used,
            "target_race_rows_used": target_race_rows_used,
            "post_outcome_rows_used": post_outcome_rows_used,
            "post_outcome_fields_used": post_outcome_fields,
        }

    statuses = [entry["status"] for entry in by_feature.values()]
    if report_failures:
        status = "FAIL"
    elif statuses and all(item == "PASS" for item in statuses):
        status = "PASS"
    elif report_partial:
        status = "NOT_POPULATED"
    else:
        status = "UNKNOWN"

    return {
        "schema_version": "same_distance_same_grade_history_provenance_v1",
        "status": status,
        "fail_reasons": report_failures,
        "required_source": SAME_DISTANCE_HISTORY_SOURCE,
        "required_history_cutoff": SAME_DISTANCE_HISTORY_CUTOFF,
        "cutoff_basis": SAME_DISTANCE_HISTORY_CUTOFF_BASIS,
        "target_race_rows_allowed": 0,
        "post_outcome_rows_allowed": 0,
        "feature_rows": len(rows),
        "by_feature": by_feature,
    }


def build_live_feature_rows(
    *,
    input_paths: Sequence[Path],
    schema: Mapping[str, Any],
    db_path: Path,
) -> list[dict[str, Any]]:
    features = list(schema["feature_columns"])
    connection = sqlite_ro(db_path)
    try:
        history_index = load_db_history(connection)
    finally:
        connection.close()
    output: list[dict[str, Any]] = []
    for path in input_paths:
        rows = load_live_csv(path)
        filename_metadata = extract_target_metadata_from_filename(path.name)
        sidecar_context = load_live_sidecar_context(path)
        sidecar_race_info = sidecar_context.get("race_info") or {}
        expert_form_metadata = sidecar_context.get("expert_form_metadata") or {}
        weather_track_safe = bool(
            sidecar_context.get("weather_track_metadata_is_leakage_safe")
        )
        safe_track_condition = (
            sidecar_context.get("track_condition") if weather_track_safe else None
        )
        safe_weather = sidecar_context.get("weather") if weather_track_safe else None
        runner_rows = []
        for raw in rows:
            dog_name, box = parse_live_runner_identity(
                csv_value(raw, "dog_name", "Dog Name", "dog", "runner", "runner_name"),
                csv_value(raw, "box_number", "box", "Box"),
            )
            if dog_name:
                runner_rows.append((raw, dog_name, box))
        field_size = len(runner_rows)
        race_id_default = path.stem
        embedded_history_cache: dict[str | None, dict[str, list[dict[str, Any]]]] = {}
        for raw, dog_name, box in runner_rows:
            race_id = str(csv_value(raw, "race_id", "Race ID", "race") or race_id_default)
            race_date = parse_date(
                csv_value(raw, "target_race_date", "race_date_safe")
                or sidecar_race_info.get("date")
                or filename_metadata.get("race_date")
                or path.name
            )
            if race_date not in embedded_history_cache:
                embedded_history_cache[race_date] = live_form_history_by_dog(
                    rows,
                    target_race_date=race_date,
                )
            embedded_history_by_dog = embedded_history_cache[race_date]
            venue = str(
                csv_value(raw, "target_venue", "venue_safe", "race_venue")
                or sidecar_race_info.get("venue")
                or filename_metadata.get("venue")
                or ""
            )
            race_number = (
                safe_int(csv_value(raw, "target_race_number", "race_number_safe"))
                or safe_int(sidecar_race_info.get("race_number"))
                or safe_int(filename_metadata.get("race_number"))
                or parse_race_number(race_id)
            )
            target_distance = safe_float(
                csv_value(raw, "target_distance_safe", "target_distance", "race_distance")
                or sidecar_context.get("target_distance")
            )
            target_grade = normalize_grade(
                csv_value(raw, "target_grade_safe", "target_grade", "race_grade")
                or sidecar_context.get("target_grade")
            )
            target_distance_source = (
                "explicit_csv_target_field"
                if csv_value(raw, "target_distance_safe", "target_distance", "race_distance")
                not in (None, "")
                else sidecar_context.get("target_distance_source")
            )
            target_grade_source = (
                "explicit_csv_target_field"
                if csv_value(raw, "target_grade_safe", "target_grade", "race_grade")
                not in (None, "")
                else sidecar_context.get("target_grade_source")
            )
            weather_track_race_time = race_time_iso_from_sidecar(
                sidecar_context.get("weather_track_metadata_race_date") or race_date,
                sidecar_context.get("weather_track_metadata_race_time")
                or sidecar_race_info.get("race_time"),
            )
            target = {"distance": target_distance, "grade": target_grade}
            row_features = {feature: None for feature in features}
            row_features.update(
                {
                    "field_size": field_size,
                    "box_number": box,
                    "box_band_inside": 1 if box in (1, 2) else 0,
                    "box_band_middle": 1 if box in (3, 4, 5, 6) else 0,
                    "box_band_outside": 1 if box is not None and box >= 7 else 0,
                    "target_distance_safe": target_distance,
                    "target_distance_source_is_safe": 1 if target_distance is not None else 0,
                    "target_distance_missing": 0 if target_distance is not None else 1,
                    "target_grade_safe": target_grade,
                    "target_grade_normalized": target_grade,
                    "target_grade_missing": 0 if target_grade else 1,
                    "target_grade_vocab_known": 1 if target_grade else 0,
                    "target_grade_provenance_safe": 1 if target_grade else 0,
                    "venue": venue,
                    "race_number": race_number,
                    "race_time_minutes_since_midnight": parse_datetime_minutes(
                        csv_value(raw, "race_time", "jump_time", "Race Time")
                        or sidecar_race_info.get("race_time")
                    ),
                    "track_condition": safe_track_condition,
                    "weather": safe_weather,
                    "target_month": safe_int((race_date or "").split("-")[1]) if race_date else None,
                    "target_day_of_week": datetime.fromisoformat(race_date).weekday()
                    if race_date
                    else None,
                    "target_distance_band_sprint": 1
                    if target_distance is not None and target_distance < 450
                    else 0
                    if target_distance is not None
                    else None,
                    "target_distance_band_middle": 1
                    if target_distance is not None and 450 <= target_distance < 650
                    else 0
                    if target_distance is not None
                    else None,
                    "target_distance_band_staying": 1
                    if target_distance is not None and target_distance >= 650
                    else 0
                    if target_distance is not None
                    else None,
                }
            )
            db_history = [
                row
                for row in history_index.get(clean_name(dog_name), [])
                if race_date and str(row.get("race_date") or "") < race_date
            ]
            embedded_history = embedded_history_by_dog.get(clean_name(dog_name), [])
            dog_history = merge_prior_history_rows(db_history, embedded_history)
            add_history_features(row_features, dog_history, target, race_date, venue)
            history_provenance = same_distance_same_grade_live_row_provenance(
                history_rows=dog_history,
                target=target,
            )
            expert_features = expert_form_runner_features(
                expert_form_metadata=expert_form_metadata,
                dog_name=dog_name,
                box_number=box,
            )
            for feature_name, feature_value in expert_features.items():
                if feature_name in row_features:
                    row_features[feature_name] = feature_value
            output_row = {
                "race_id": race_id,
                "shadow_race_group_id": f"{path.as_posix()}::{race_id}",
                "source_csv": shadow_relpath(path),
                "race_date": race_date,
                "venue": venue,
                "dog_name": dog_name,
                "box_number": box,
                "target_distance_source": target_distance_source,
                "target_grade_source": target_grade_source,
                "target_metadata_source_url": sidecar_context.get("metadata_source_url"),
                "target_metadata_from_sidecar": bool(
                    sidecar_context.get("metadata_is_leakage_safe")
                    and (
                        sidecar_context.get("target_distance")
                        or sidecar_context.get("target_grade")
                    )
                ),
                "target_metadata_rejected_sources": sidecar_context.get(
                    "rejected_metadata_sources", []
                ),
                "metadata_is_leakage_safe": bool(
                    weather_track_safe and (safe_track_condition or safe_weather)
                ),
                "source_url": sidecar_context.get("weather_track_metadata_source_url"),
                "collection_timestamp": sidecar_context.get(
                    "weather_track_metadata_captured_at"
                ),
                "race_time": weather_track_race_time,
                "track_condition_source_backed": bool(
                    weather_track_safe
                    and safe_track_condition
                    and sidecar_context.get("weather_track_metadata_source_url")
                ),
                "weather_source_backed": bool(
                    weather_track_safe
                    and safe_weather
                    and sidecar_context.get("weather_track_metadata_source_url")
                ),
                "weather_track_metadata_from_sidecar": bool(
                    weather_track_safe and (safe_track_condition or safe_weather)
                ),
                "weather_track_metadata_source": sidecar_context.get(
                    "weather_track_metadata_source"
                ),
                "weather_track_metadata_rejected_sources": sidecar_context.get(
                    "rejected_weather_track_metadata_sources", []
                ),
            }
            output_row.update(history_provenance)
            output_row.update(row_features)
            output_row.update(expert_features)
            output.append(output_row)
    return output


def score_live(args: argparse.Namespace) -> int:
    ensure_shadow_runtime_guard()
    run_started_at = datetime.now().astimezone()
    output_dir = prepare_output_dir(args.output_dir or default_live_output_dir())
    protected_before = protected_path_snapshot()
    final_status = "IMPLEMENTATION_ABORTED"
    try:
        schema = load_json(args.schema)
        schema_audit = validate_schema_contract(schema)
        if schema_audit["status"] != "PASS":
            raise RuntimeError(f"schema_contract_failed:{schema_audit['fail_reasons']}")
        write_shadow_candidate_definition(output_dir, schema, args.schema)
        input_paths = input_files_from_path(args.input)
        deps = sklearn_imports()
        if deps["status"] != "OK":
            raise RuntimeError(f"missing_ml_dependencies:{deps['error']}")
        active_features = list(schema["feature_columns"])
        active_feature_policy = {
            "schema_version": "loaded_shadow_model_feature_policy_v1",
            "source": "canonical_schema_default",
            "reason": "initialized_before_model_selection",
            "active_feature_count": len(active_features),
            "schema_feature_count": len(active_features),
            "inactive_features_due_to_train_all_missing": [],
        }
        if args.model:
            pipeline = deps["load"](args.model)
            model_source = shadow_relpath(args.model)
            model_version = f"shadow_loaded_{args.model.stem}"
            active_features, active_feature_policy = active_features_for_loaded_model(
                model_path=args.model,
                schema=schema,
            )
        elif args.train_if_missing:
            dataset, feature_audit, population = build_shadow_feature_matrix(
                clean_dataset=args.clean_dataset,
                repaired_packet=args.repaired_packet,
                schema_path=args.schema,
                db_path=args.db,
            )
            write_json(output_dir / "shadow_feature_matrix_audit.json", feature_audit)
            write_json(output_dir / "shadow_feature_population_report.json", population)
            parity_report = train_eval_feature_parity_report(
                dataset,
                policy=args.all_missing_train_policy,
            )
            policy_report = inactive_feature_policy_report(parity_report)
            write_json(output_dir / "train_eval_feature_parity_report.json", parity_report)
            write_json(output_dir / "inactive_feature_policy_report.json", policy_report)
            dataset_for_model = dataset_with_all_missing_train_policy(dataset, parity_report)
            active_features = list(dataset_for_model["features"])
            pipeline, training_report = train_or_load_shadow_rf(dataset=dataset_for_model, output_dir=output_dir)
            write_json(output_dir / "shadow_training_report.json", training_report)
            if pipeline is None:
                raise RuntimeError("shadow_model_training_failed")
            model_source = shadow_relpath(output_dir / "shadow_randomforest_model.joblib")
            model_version = f"shadow_randomforest_{CALIBRATION_METHOD_KEY}_{now_id()}"
            active_feature_policy = {
                "schema_version": "loaded_shadow_model_feature_policy_v1",
                "source": "trained_in_current_shadow_score_live_run",
                "reason": "train_if_missing_active_feature_policy",
                "model_path": model_source,
                "active_feature_count": len(active_features),
                "schema_feature_count": len(schema["feature_columns"]),
                "inactive_features_due_to_train_all_missing": list(
                    training_report.get("inactive_features_due_to_train_all_missing") or []
                ),
                "all_missing_train_policy": training_report.get("all_missing_train_policy")
                or "report_only",
            }
        else:
            raise RuntimeError("shadow_model_required_or_train_if_missing")

        write_json(output_dir / "active_feature_policy_report.json", active_feature_policy)
        feature_freeze_at = datetime.now().astimezone()
        rows = build_live_feature_rows(input_paths=input_paths, schema=schema, db_path=args.db)
        write_json(output_dir / "shadow_feature_rows.json", rows)
        same_distance_history_report = same_distance_same_grade_history_provenance_report(rows)
        write_json(
            output_dir / "same_distance_same_grade_history_provenance.json",
            same_distance_history_report,
        )
        x_live, _ = prepare_xy(rows, active_features)
        raw_probs = [float(value) for value in pipeline.predict_proba(x_live)[:, 1]]
        uncalibrated = normalize_probabilities_by_race(
            rows,
            raw_probs,
            "shadow_rf_uncalibrated_probability",
            "shadow_rf_uncalibrated_rank",
        )
        calibrated = apply_power_gamma_by_race(
            uncalibrated,
            input_key="shadow_rf_uncalibrated_probability",
            output_key="shadow_rf_calibrated_probability",
            output_rank_key="predicted_rank",
        )
        predictions = []
        for row in calibrated:
            predictions.append(
                {
                    "race_id": row.get("race_id"),
                    "dog_name": row.get("dog_name"),
                    "box": row.get("box_number"),
                    "shadow_rf_uncalibrated_probability": row.get(
                        "shadow_rf_uncalibrated_probability"
                    ),
                    "shadow_rf_calibrated_probability": row.get(
                        "shadow_rf_calibrated_probability"
                    ),
                    "predicted_rank": row.get("predicted_rank"),
                    "calibration_method": CALIBRATION_METHOD_KEY,
                    "model_version": model_version,
                    "model_source": model_source,
                    "tgr_enabled": False,
                    "output_mode": SHADOW_OUTPUT_MODE,
                }
            )
        prediction_timestamp = datetime.now().astimezone()
        stage2_status = STAGE2_FORWARD_SHADOW_COLLECTING
        stage2_predictions = stage2_shadow_prediction_rows(
            predictions,
            stage2_status=stage2_status,
        )
        write_json(output_dir / "shadow_predictions.json", predictions)
        write_jsonl_file(output_dir / "shadow_predictions.jsonl", stage2_predictions)
        write_jsonl_file(output_dir / "stage2_shadow_predictions.jsonl", stage2_predictions)
        write_csv(
            output_dir / "shadow_predictions.csv",
            predictions,
            [
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
            ],
        )
        manifest = {
            "schema_version": "shadow_live_scoring_manifest_v1",
            "generated_at": prediction_timestamp.isoformat(),
            "run_started_at": run_started_at.isoformat(),
            "feature_freeze_timestamp": feature_freeze_at.isoformat(),
            "prediction_timestamp": prediction_timestamp.isoformat(),
            "output_mode": SHADOW_OUTPUT_MODE,
            "input_files": [shadow_relpath(path) for path in input_paths],
            "prediction_rows": len(predictions),
            "feature_rows": shadow_relpath(output_dir / "shadow_feature_rows.json"),
            "shadow_predictions_jsonl": shadow_relpath(output_dir / "shadow_predictions.jsonl"),
            "stage2_shadow_predictions_jsonl": shadow_relpath(
                output_dir / "stage2_shadow_predictions.jsonl"
            ),
            "same_distance_same_grade_history_provenance": shadow_relpath(
                output_dir / "same_distance_same_grade_history_provenance.json"
            ),
            "model_source": model_source,
            "model_version": model_version,
            "calibration_method": CALIBRATION_METHOD_KEY,
            "active_feature_count": len(active_features),
            "schema_feature_count": len(schema["feature_columns"]),
            "inactive_features_due_to_train_all_missing": active_feature_policy.get(
                "inactive_features_due_to_train_all_missing"
            )
            or [],
            "tgr_enabled": False,
            "registry_mutation": False,
            "production_prediction_write": False,
            "stage2_forward_shadow_status": stage2_status,
            "stage2_forward_shadow_ready_for_review_status": STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW,
            "stage2_forward_shadow_collecting_status": STAGE2_FORWARD_SHADOW_COLLECTING,
            "odds_used_for_shadow_scoring": False,
            "betting_output": False,
            "ev_output": False,
        }
        write_json(output_dir / "shadow_manifest.json", manifest)
        write_text(
            output_dir / "shadow_run_summary.md",
            "\n".join(
                [
                    "# Shadow Live Scoring Run",
                    "",
                    f"Input files: `{len(input_paths)}`.",
                    f"Prediction rows: `{len(predictions)}`.",
                    f"Calibration: `{CALIBRATION_METHOD_KEY}`.",
                    "",
                    "No production predictions, betting files, EV files, registry entries, DB writes, snapshots, or manifests outside this shadow directory were written.",
                    "",
                ]
            ),
        )
        final_status = "SHADOW_EVALUATION_IMPLEMENTED_READY_FOR_SHADOW_RUNS"
        return 0
    except RuntimeError as exc:
        write_json(output_dir / "shadow_runtime_error.json", {"error": repr(exc)})
        if str(exc).startswith("all_missing_train_policy_failed"):
            final_status = "BLOCKED_BY_TRAIN_EVAL_FEATURE_PARITY"
        else:
            final_status = "PARTIAL_SHADOW_IMPLEMENTATION_NEEDS_FIXES"
        return 2
    finally:
        protected = protected_path_verification(protected_before)
        if not protected.get("protected_paths_unchanged"):
            final_status = "IMPLEMENTATION_ABORTED"
        write_json(output_dir / "protected_path_verification.json", protected)
        write_json(output_dir / "implementation_file_manifest.json", output_file_manifest(output_dir))
        write_text(output_dir / "final_status.txt", final_status + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args_list = list(argv if argv is not None else sys.argv[1:])
    if not args_list or args_list[0] not in {"run", "score-live"}:
        args_list.insert(0, "run")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="train/load and replay shadow evaluation")
    run_parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN_DATASET)
    run_parser.add_argument("--repaired-packet", type=Path, default=DEFAULT_REPAIRED_PACKET)
    run_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    run_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    run_parser.add_argument("--output-dir", type=Path, default=None)
    run_parser.add_argument("--load-model", type=Path, default=None)
    run_parser.add_argument(
        "--all-missing-train-policy",
        choices=ALL_MISSING_TRAIN_POLICIES,
        default="report_only",
        help="How to handle features with no observed train values.",
    )
    run_parser.add_argument("--stop-after-definition", action="store_true")
    run_parser.add_argument("--stop-after-audit", action="store_true")

    live_parser = subparsers.add_parser("score-live", help="shadow-score upcoming/pre-jump CSVs")
    live_parser.add_argument("--input", type=Path, required=True)
    live_parser.add_argument("--model", type=Path, default=None)
    live_parser.add_argument("--train-if-missing", action="store_true")
    live_parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN_DATASET)
    live_parser.add_argument("--repaired-packet", type=Path, default=DEFAULT_REPAIRED_PACKET)
    live_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    live_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    live_parser.add_argument("--output-dir", type=Path, default=None)
    live_parser.add_argument(
        "--all-missing-train-policy",
        choices=ALL_MISSING_TRAIN_POLICIES,
        default="report_only",
        help="How to handle features with no observed train values when --train-if-missing is used.",
    )
    return parser.parse_args(args_list)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return run_shadow(args)
    if args.command == "score-live":
        return score_live(args)
    raise SystemExit(f"unknown_command:{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
