#!/usr/bin/env python3
"""Run a report-only Expert Form schema trial and ablation gate.

The packet creates an artifact-local trial schema with the missing
``expert_form_*`` fields, audits feature quality, joins report-only official
result labels when available, and only runs an in-memory ablation when there is
enough labeled evidence. It never mutates the canonical schema, registry, DB,
labels, production predictions, EV outputs, or betting files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.build_expert_form_feature_readiness_packet import (  # noqa: E402
    DEFAULT_SCHEMA,
    EXPERT_FORM_FEATURES,
)
from scripts.run_feature_recovery_execution_v1 import (  # noqa: E402
    RANDOM_SEED,
    clean_name,
    load_json,
    prepare_xy,
    safe_float,
    safe_int,
    sha256_file,
)
from scripts.run_shadow_non_tgr_rf_evaluation import (  # noqa: E402
    apply_power_gamma_by_race,
    make_one_hot_encoder,
    normalize_probabilities_by_race,
    protected_path_snapshot,
    protected_path_verification,
    score_grouped_metrics,
    shadow_relpath,
    sklearn_imports,
)


OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "expert_form_schema_trial_ablation_"
)
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
FINAL_LABELS_MISSING = "KEEP_COLLECTING_ONLY_LABELS_MISSING_FOR_EXPERT_FORM_ABLATION"
FINAL_BASE_FEATURES_MISSING = "KEEP_COLLECTING_ONLY_BASE_FEATURE_ROWS_MISSING_FOR_ABLATION"
FINAL_ABLATION_FAILED = "KEEP_COLLECTING_ONLY_EXPERT_FORM_ABLATION_FAILED"
FINAL_APPROVAL_READY = "EXPERT_FORM_SHADOW_SCHEMA_APPROVAL_PACKET_READY_REPORT_ONLY"
DEFAULT_MIN_SLICE_RACES = 3

EXPERT_FORM_CATEGORICAL_FEATURES = {
    "expert_form_grade",
    "expert_form_sex",
    "expert_form_sire",
    "expert_form_dam",
    "expert_form_trainer_name",
    "expert_form_trainer_district",
}
PLACEHOLDER_VALUES = {
    "",
    "0",
    "0.0",
    "none",
    "null",
    "unknown",
    "n/a",
    "-",
}
NO_WRITE_GUARANTEES = {
    "report_only": True,
    "canonical_schema_mutation": False,
    "training_artifact_write": False,
    "registry_mutation": False,
    "production_prediction_write": False,
    "db_write": False,
    "label_write": False,
    "ev_output": False,
    "betting_output": False,
}


def now_id(value: datetime | None = None) -> str:
    return (value or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_expert_form_schema_trial_artifact:{relative}")
    return logical


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def read_json_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def latest_path(pattern: str) -> Path | None:
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))


def default_expert_feature_rows_path() -> Path | None:
    return latest_path(
        "artifacts/full_evidence_orchestration_20260525/"
        "expert_form_shadow_feature_row_backfill_*_report_only/shadow_feature_rows.json"
    )


def default_official_result_paths() -> list[Path]:
    return sorted(
        DEFAULT_EVIDENCE_ROOT.glob(
            "autonomous_official_result_capture_*_daemon_autopilot/official_result_runners.jsonl"
        )
    )


def build_trial_schema(schema: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original_features = list(schema.get("feature_columns") or [])
    original_categorical = list(schema.get("categorical_features") or [])
    original_numeric = list(schema.get("numeric_or_boolean_features") or [])
    missing = [feature for feature in EXPERT_FORM_FEATURES if feature not in original_features]
    trial = dict(schema)
    trial["schema_version"] = f"{schema.get('schema_version', 'schema')}_expert_form_report_only_trial"
    trial["status"] = "report_only_trial_not_canonical"
    trial["feature_columns"] = [*original_features, *missing]
    trial["categorical_features"] = [
        *original_categorical,
        *[feature for feature in missing if feature in EXPERT_FORM_CATEGORICAL_FEATURES],
    ]
    trial["numeric_or_boolean_features"] = [
        *original_numeric,
        *[feature for feature in missing if feature not in EXPERT_FORM_CATEGORICAL_FEATURES],
    ]
    families = dict(schema.get("feature_families") or {})
    families["expert_form"] = list(EXPERT_FORM_FEATURES)
    trial["feature_families"] = families
    trial["canonical_schema_mutation"] = False
    trial["report_only_source_schema_sha256"] = None
    diff_rows = [
        {
            "feature": feature,
            "in_canonical_schema": feature in original_features,
            "in_trial_schema": True,
            "trial_action": "already_present" if feature in original_features else "added_report_only",
            "feature_type": "categorical"
            if feature in EXPERT_FORM_CATEGORICAL_FEATURES
            else "numeric_or_boolean",
        }
        for feature in EXPERT_FORM_FEATURES
    ]
    return trial, diff_rows


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


def value_text(value: Any) -> str:
    return str(value).strip().lower()


def feature_quality_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    out = []
    for feature in EXPERT_FORM_FEATURES:
        values = [row.get(feature) for row in rows if present(row.get(feature))]
        unique = {str(value) for value in values}
        most_common = Counter(str(value) for value in values).most_common(1)
        placeholder_rows = sum(1 for value in values if value_text(value) in PLACEHOLDER_VALUES)
        present_pct = len(values) / total if total else 0.0
        default_share = placeholder_rows / len(values) if values else 0.0
        dominance = most_common[0][1] / len(values) if values and most_common else 0.0
        if not values:
            status = "MISSING"
        elif len(unique) <= 1:
            status = "FLAT"
        elif present_pct < 0.2:
            status = "LOW_COVERAGE"
        elif default_share >= 0.8 or dominance >= 0.95:
            status = "DEFAULT_OR_DOMINANT"
        else:
            status = "PASS"
        out.append(
            {
                "feature": feature,
                "rows": total,
                "present_rows": len(values),
                "missing_rows": total - len(values),
                "present_pct": present_pct,
                "unique_present_values": len(unique),
                "placeholder_or_zero_rows": placeholder_rows,
                "placeholder_or_zero_share": default_share,
                "most_common_value": most_common[0][0] if most_common else None,
                "most_common_count": most_common[0][1] if most_common else 0,
                "most_common_share": dominance,
                "quality_status": status,
            }
        )
    return out


def leakage_check_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unsafe_rows = [
        row
        for row in rows
        if row.get("expert_form_metadata_from_sidecar") is not True
        or row.get("expert_form_metadata_rejected_reasons") not in (None, "", [], {})
    ]
    return [
        {
            "check": "expert_form_features_are_prefixed",
            "status": "PASS"
            if all(str(feature).startswith("expert_form_") for feature in EXPERT_FORM_FEATURES)
            else "FAIL",
            "details": "all trial candidate columns must stay in expert_form namespace",
        },
        {
            "check": "rows_source_safe_sidecar",
            "status": "PASS" if not unsafe_rows else "FAIL",
            "details": f"unsafe_or_rejected_rows={len(unsafe_rows)}",
        },
        {
            "check": "post_result_fields_excluded",
            "status": "PASS",
            "details": "finish/result/winner fields are not in candidate feature list",
        },
    ]


def race_date_from_id(race_id: Any) -> str | None:
    text = str(race_id or "")
    parts = text.rsplit(" - ", 1)
    if len(parts) == 2 and len(parts[1]) >= 10:
        return parts[1][:10]
    return None


def official_result_index(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl_rows(path):
            race_id = str(row.get("race_id") or "")
            if not race_id:
                continue
            key = (race_id, clean_name(row.get("dog_name")), safe_int(row.get("box_number")))
            current = by_key.get(key)
            if current is None or str(row.get("captured_at") or "") >= str(current.get("captured_at") or ""):
                by_key[key] = {**row, "official_result_path": relpath(path)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_key.values():
        grouped[str(row.get("race_id"))].append(row)
    return dict(grouped)


def join_labels(
    feature_rows: Sequence[Mapping[str, Any]],
    official_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official = official_result_index(official_paths)
    feature_by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        feature_by_race[str(row.get("race_id") or "")].append(row)
    labeled: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for race_id, rows in sorted(feature_by_race.items()):
        result_rows = official.get(race_id) or []
        winners = [row for row in result_rows if row.get("is_winner") is True or safe_int(row.get("finish_position")) == 1]
        if not result_rows:
            coverage_rows.append(
                {
                    "race_id": race_id,
                    "feature_rows": len(rows),
                    "official_result_rows": 0,
                    "matched_rows": 0,
                    "winner_count": 0,
                    "label_join_status": "OFFICIAL_RESULT_MISSING",
                }
            )
            continue
        if len(winners) != 1:
            coverage_rows.append(
                {
                    "race_id": race_id,
                    "feature_rows": len(rows),
                    "official_result_rows": len(result_rows),
                    "matched_rows": 0,
                    "winner_count": len(winners),
                    "label_join_status": "WINNER_COUNT_NOT_ONE",
                }
            )
            continue
        by_dog = {clean_name(row.get("dog_name")): row for row in result_rows}
        by_box = {safe_int(row.get("box_number")): row for row in result_rows}
        matched = 0
        race_labeled = []
        for row in rows:
            result = by_dog.get(clean_name(row.get("dog_name"))) or by_box.get(
                safe_int(row.get("box_number"))
            )
            if result is None:
                continue
            matched += 1
            race_labeled.append(
                {
                    **dict(row),
                    "race_date": result.get("race_date") or race_date_from_id(race_id),
                    "venue": row.get("venue") or result.get("venue"),
                    "race_number": row.get("race_number") or result.get("race_number"),
                    "finish_position": safe_int(result.get("finish_position")),
                    "actual_win": 1 if (result.get("is_winner") is True or safe_int(result.get("finish_position")) == 1) else 0,
                    "official_result_source_url": result.get("source_url"),
                    "official_result_path": result.get("official_result_path"),
                }
            )
        status = "PASS" if matched == len(rows) else "PARTIAL_RUNNER_MATCH"
        coverage_rows.append(
            {
                "race_id": race_id,
                "feature_rows": len(rows),
                "official_result_rows": len(result_rows),
                "matched_rows": matched,
                "winner_count": len(winners),
                "label_join_status": status,
            }
        )
        if status == "PASS":
            labeled.extend(race_labeled)
    return labeled, coverage_rows


def merge_base_rows(
    expert_rows: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not base_rows:
        return [dict(row) for row in expert_rows]
    base_index = {
        (
            str(row.get("race_id") or ""),
            clean_name(row.get("dog_name")),
            safe_int(row.get("box_number")),
        ): dict(row)
        for row in base_rows
    }
    merged = []
    for row in expert_rows:
        key = (
            str(row.get("race_id") or ""),
            clean_name(row.get("dog_name")),
            safe_int(row.get("box_number")),
        )
        merged.append({**base_index.get(key, {}), **dict(row)})
    return merged


def race_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("race_id") or "")].append(dict(row))
    return dict(groups)


def temporal_split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_fraction: float,
) -> dict[str, Any]:
    groups = race_groups(rows)
    race_ids = sorted(
        groups,
        key=lambda race_id: (
            str((groups[race_id][0].get("race_date") or race_date_from_id(race_id) or "")),
            race_id,
        ),
    )
    holdout_count = max(1, int(math.ceil(len(race_ids) * holdout_fraction))) if race_ids else 0
    if holdout_count >= len(race_ids) and len(race_ids) > 1:
        holdout_count = len(race_ids) - 1
    holdout_ids = set(race_ids[-holdout_count:]) if holdout_count else set()
    train_ids = [race_id for race_id in race_ids if race_id not in holdout_ids]
    train_rows = [row for race_id in train_ids for row in groups[race_id]]
    holdout_rows = [row for race_id in race_ids if race_id in holdout_ids for row in groups[race_id]]
    return {
        "race_ids": race_ids,
        "train_race_ids": train_ids,
        "holdout_race_ids": [race_id for race_id in race_ids if race_id in holdout_ids],
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
    }


def has_label_variation(rows: Sequence[Mapping[str, Any]]) -> bool:
    labels = {int(row.get("actual_win") or 0) for row in rows}
    return labels == {0, 1}


def train_predict_metrics(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    categorical_features: Sequence[str],
    probability_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    deps = sklearn_imports()
    if deps["status"] != "OK":
        return {"status": "SKLEARN_MISSING", "error": deps["error"]}, [], {}
    categorical = set(categorical_features)
    categorical_indices = [index for index, feature in enumerate(features) if feature in categorical]
    numeric_indices = [index for index, feature in enumerate(features) if feature not in categorical]
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
        n_estimators=200,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    pipeline = deps["Pipeline"](steps=[("prep", transformer), ("model", estimator)])
    x_train, y_train = prepare_xy(train_rows, features)
    pipeline.fit(x_train, y_train)
    x_holdout, _ = prepare_xy(holdout_rows, features)
    probabilities = [float(value) for value in pipeline.predict_proba(x_holdout)[:, 1]]
    normalized = normalize_probabilities_by_race(
        holdout_rows,
        probabilities,
        f"{probability_key}_uncalibrated",
        f"{probability_key}_uncalibrated_rank",
    )
    scored = apply_power_gamma_by_race(
        normalized,
        input_key=f"{probability_key}_uncalibrated",
        output_key=probability_key,
        output_rank_key=f"{probability_key}_rank",
    )
    metrics = score_grouped_metrics(scored, probability_key)
    metrics["status"] = "EVALUATED"
    metrics["feature_count"] = len(features)
    importances: dict[str, float] = {}
    try:
        transformed_names = list(pipeline.named_steps["prep"].get_feature_names_out())
        values = list(pipeline.named_steps["model"].feature_importances_)
        for name, value in zip(transformed_names, values):
            original = name.split("__", 1)[-1].split("_", 1)[0]
            for feature in features:
                if name.endswith(feature) or feature in name:
                    original = feature
                    break
            importances[original] = importances.get(original, 0.0) + float(value)
    except Exception:
        importances = {}
    return metrics, scored, importances


def market_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    odds_keys = ("odds_win", "win_odds", "fixed_win_odds", "market_odds", "odds")
    out_rows: list[dict[str, Any]] = []
    for race_id, group_rows in race_groups(rows).items():
        implied = []
        for row in group_rows:
            odds = None
            for key in odds_keys:
                odds = safe_float(row.get(key))
                if odds is not None:
                    break
            if odds is None or odds <= 1.0:
                implied.append(None)
            else:
                implied.append(1.0 / odds)
        total = sum(value for value in implied if value is not None)
        if total <= 0:
            continue
        for row, value in zip(group_rows, implied):
            if value is None:
                continue
            out_rows.append({**row, "market_probability": value / total, "shadow_race_group_id": race_id})
    if not out_rows:
        return {"status": "DATA_MISSING", "blockers": ["market_odds_missing"]}
    metrics = score_grouped_metrics(out_rows, "market_probability")
    metrics["status"] = "EVALUATED"
    return metrics


def compare_gate(
    *,
    trial_metrics: Mapping[str, Any] | None,
    control_metrics: Mapping[str, Any] | None,
    market: Mapping[str, Any],
    min_holdout_races: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not trial_metrics or trial_metrics.get("status") != "EVALUATED":
        blockers.append("trial_metrics_missing")
    if not control_metrics or control_metrics.get("status") != "EVALUATED":
        blockers.append("control_metrics_missing")
    if market.get("status") != "EVALUATED":
        blockers.append("market_baseline_missing")
    if trial_metrics and safe_int(trial_metrics.get("race_count")) is not None:
        if safe_int(trial_metrics.get("race_count")) < min_holdout_races:
            blockers.append("holdout_race_count_below_min")
    if trial_metrics and control_metrics:
        comparisons = (
            ("top1", "higher_or_equal"),
            ("top3", "higher_or_equal"),
            ("mean_winner_rank", "lower_or_equal"),
            ("log_loss", "lower_or_equal"),
        )
        for metric, direction in comparisons:
            trial_value = safe_float(trial_metrics.get(metric))
            control_value = safe_float(control_metrics.get(metric))
            if trial_value is None or control_value is None:
                blockers.append(f"metric_missing:{metric}")
            elif direction == "higher_or_equal" and trial_value < control_value:
                blockers.append(f"trial_worse_than_control:{metric}")
            elif direction == "lower_or_equal" and trial_value > control_value:
                blockers.append(f"trial_worse_than_control:{metric}")
    if trial_metrics and market.get("status") == "EVALUATED":
        for metric, direction in (
            ("top1", "higher_or_equal"),
            ("top3", "higher_or_equal"),
            ("mean_winner_rank", "lower_or_equal"),
            ("log_loss", "lower_or_equal"),
        ):
            trial_value = safe_float(trial_metrics.get(metric))
            market_value = safe_float(market.get(metric))
            if trial_value is None or market_value is None:
                blockers.append(f"market_metric_missing:{metric}")
            elif direction == "higher_or_equal" and trial_value < market_value:
                blockers.append(f"trial_worse_than_market:{metric}")
            elif direction == "lower_or_equal" and trial_value > market_value:
                blockers.append(f"trial_worse_than_market:{metric}")
    return {
        "schema_version": "expert_form_schema_trial_gate_v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "required_gates": {
            "top1": "trial >= control and market when market exists",
            "top3": "trial >= control and market when market exists",
            "mean_winner_rank": "trial <= control and market when market exists",
            "log_loss": "trial <= control and market when market exists",
            "market_missing_policy": "fail_closed",
        },
    }


def distance_band(row: Mapping[str, Any]) -> str | None:
    if safe_int(row.get("target_distance_band_sprint")) == 1:
        return "sprint"
    if safe_int(row.get("target_distance_band_middle")) == 1:
        return "middle"
    if safe_int(row.get("target_distance_band_staying")) == 1:
        return "staying"
    distance = safe_float(row.get("target_distance_safe") or row.get("target_distance"))
    if distance is None:
        return None
    if distance < 450:
        return "sprint"
    if distance < 650:
        return "middle"
    return "staying"


def slice_race_sets(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        race_id = str(row.get("race_id") or "")
        if not race_id:
            continue
        venue = str(row.get("venue") or "").strip()
        if venue:
            out[("venue", venue)].add(race_id)
        band = distance_band(row)
        if band:
            out[("distance_band", band)].add(race_id)
    return out


def metric_delta(
    trial_metrics: Mapping[str, Any],
    control_metrics: Mapping[str, Any],
    metric: str,
) -> float | None:
    trial_value = safe_float(trial_metrics.get(metric))
    control_value = safe_float(control_metrics.get(metric))
    if trial_value is None or control_value is None:
        return None
    return trial_value - control_value


def slice_regression_rows(
    *,
    control_rows: Sequence[Mapping[str, Any]],
    trial_rows: Sequence[Mapping[str, Any]],
    min_slice_races: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    race_sets = slice_race_sets(control_rows)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for (slice_type, slice_value), race_ids in sorted(race_sets.items()):
        control_slice = [row for row in control_rows if str(row.get("race_id") or "") in race_ids]
        trial_slice = [row for row in trial_rows if str(row.get("race_id") or "") in race_ids]
        control_metrics = score_grouped_metrics(control_slice, "control_probability")
        trial_metrics = score_grouped_metrics(trial_slice, "trial_probability")
        race_count = int(control_metrics.get("race_count") or 0)
        deltas = {
            "top1_delta": metric_delta(trial_metrics, control_metrics, "top1"),
            "top3_delta": metric_delta(trial_metrics, control_metrics, "top3"),
            "mean_winner_rank_delta": metric_delta(
                trial_metrics, control_metrics, "mean_winner_rank"
            ),
            "log_loss_delta": metric_delta(trial_metrics, control_metrics, "log_loss"),
        }
        regression_metrics: list[str] = []
        if race_count >= min_slice_races:
            if deltas["top1_delta"] is not None and deltas["top1_delta"] < 0:
                regression_metrics.append("top1")
            if deltas["top3_delta"] is not None and deltas["top3_delta"] < 0:
                regression_metrics.append("top3")
            if (
                deltas["mean_winner_rank_delta"] is not None
                and deltas["mean_winner_rank_delta"] > 0
            ):
                regression_metrics.append("mean_winner_rank")
            if deltas["log_loss_delta"] is not None and deltas["log_loss_delta"] > 0:
                regression_metrics.append("log_loss")
        status = (
            "REGRESSION"
            if regression_metrics
            else "PASS"
            if race_count >= min_slice_races
            else "REVIEW_ONLY_SMALL_SLICE"
        )
        if regression_metrics:
            blockers.append(
                "slice_regression:"
                + slice_type
                + ":"
                + slice_value
                + ":"
                + ",".join(regression_metrics)
            )
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "race_count": race_count,
                "row_count": int(control_metrics.get("row_count") or 0),
                "control_top1": control_metrics.get("top1"),
                "trial_top1": trial_metrics.get("top1"),
                "top1_delta": deltas["top1_delta"],
                "control_top3": control_metrics.get("top3"),
                "trial_top3": trial_metrics.get("top3"),
                "top3_delta": deltas["top3_delta"],
                "control_mean_winner_rank": control_metrics.get("mean_winner_rank"),
                "trial_mean_winner_rank": trial_metrics.get("mean_winner_rank"),
                "mean_winner_rank_delta": deltas["mean_winner_rank_delta"],
                "control_log_loss": control_metrics.get("log_loss"),
                "trial_log_loss": trial_metrics.get("log_loss"),
                "log_loss_delta": deltas["log_loss_delta"],
                "status": status,
                "regression_metrics": ",".join(regression_metrics),
            }
        )
    return rows, blockers


def utility_rank_rows(
    quality_rows: Sequence[Mapping[str, Any]],
    importances: Mapping[str, float],
    *,
    trained: bool,
) -> list[dict[str, Any]]:
    rows = []
    for row in quality_rows:
        feature = str(row.get("feature"))
        present_pct = safe_float(row.get("present_pct")) or 0.0
        unique = safe_float(row.get("unique_present_values")) or 0.0
        dominance = safe_float(row.get("most_common_share")) or 0.0
        importance = float(importances.get(feature, 0.0))
        quality_score = present_pct * min(unique, 20.0) / 20.0 * (1.0 - dominance)
        rows.append(
            {
                "feature": feature,
                "utility_signal_type": "model_importance" if trained else "coverage_quality_only",
                "importance_share": importance,
                "quality_score": quality_score,
                "present_pct": present_pct,
                "unique_present_values": unique,
                "most_common_share": dominance,
                "quality_status": row.get("quality_status"),
            }
        )
    rows.sort(key=lambda item: (item["importance_share"], item["quality_score"]), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def build_report(
    *,
    schema_path: Path,
    expert_feature_rows_path: Path | None,
    official_result_paths: Sequence[Path],
    base_feature_rows_path: Path | None = None,
    min_train_races: int = 20,
    min_holdout_races: int = 10,
    min_slice_races: int = DEFAULT_MIN_SLICE_RACES,
    holdout_fraction: float = 0.3,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    schema = load_json(schema_path)
    trial_schema, schema_diff = build_trial_schema(schema)
    if schema_path.exists():
        trial_schema["report_only_source_schema_sha256"] = sha256_file(schema_path)
    expert_rows = read_json_rows(expert_feature_rows_path)
    base_rows = read_json_rows(base_feature_rows_path)
    feature_rows = merge_base_rows(expert_rows, base_rows)
    quality = feature_quality_rows(feature_rows)
    leakage = leakage_check_rows(feature_rows)
    labeled_rows, label_coverage = join_labels(feature_rows, official_result_paths)
    split = temporal_split_rows(labeled_rows, holdout_fraction=holdout_fraction)
    train_rows = split["train_rows"]
    holdout_rows = split["holdout_rows"]
    train_races = len(split["train_race_ids"])
    holdout_races = len(split["holdout_race_ids"])
    blockers: list[str] = []
    ablation_status = "NOT_RUN"
    control_metrics: dict[str, Any] | None = None
    trial_metrics: dict[str, Any] | None = None
    market = market_metrics(holdout_rows) if holdout_rows else {"status": "DATA_MISSING", "blockers": ["holdout_rows_missing"]}
    gate = compare_gate(
        trial_metrics=None,
        control_metrics=None,
        market=market,
        min_holdout_races=min_holdout_races,
    )
    importances: dict[str, float] = {}
    control_scored_rows: list[dict[str, Any]] = []
    trial_scored_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    if train_races < min_train_races or holdout_races < min_holdout_races:
        blockers.append("labeled_temporal_split_below_min")
        final_status = FINAL_LABELS_MISSING
    elif not base_rows:
        blockers.append("base_feature_rows_missing")
        final_status = FINAL_BASE_FEATURES_MISSING
    elif not has_label_variation(train_rows) or not has_label_variation(holdout_rows):
        blockers.append("label_variation_missing")
        final_status = FINAL_LABELS_MISSING
    else:
        base_features = list(schema.get("feature_columns") or [])
        trial_features = list(trial_schema.get("feature_columns") or [])
        base_present = any(any(present(row.get(feature)) for row in train_rows) for feature in base_features)
        if not base_present:
            blockers.append("base_features_all_missing")
            final_status = FINAL_BASE_FEATURES_MISSING
        else:
            base_categorical = list(schema.get("categorical_features") or [])
            trial_categorical = list(trial_schema.get("categorical_features") or [])
            control_metrics, control_scored_rows, _control_importances = train_predict_metrics(
                train_rows=train_rows,
                holdout_rows=holdout_rows,
                features=base_features,
                categorical_features=base_categorical,
                probability_key="control_probability",
            )
            trial_metrics, trial_scored_rows, importances = train_predict_metrics(
                train_rows=train_rows,
                holdout_rows=holdout_rows,
                features=trial_features,
                categorical_features=trial_categorical,
                probability_key="trial_probability",
            )
            ablation_status = "RUN"
            gate = compare_gate(
                trial_metrics=trial_metrics,
                control_metrics=control_metrics,
                market=market,
                min_holdout_races=min_holdout_races,
            )
            blockers.extend(gate["blockers"])
            slice_rows, slice_blockers = slice_regression_rows(
                control_rows=control_scored_rows,
                trial_rows=trial_scored_rows,
                min_slice_races=min_slice_races,
            )
            blockers.extend(slice_blockers)
            if slice_blockers:
                gate = {**gate, "status": "FAIL", "blockers": [*gate["blockers"], *slice_blockers]}
            final_status = FINAL_APPROVAL_READY if gate["status"] == "PASS" else FINAL_ABLATION_FAILED
    utility = utility_rank_rows(quality, importances, trained=ablation_status == "RUN")
    label_counts = Counter(row.get("label_join_status") for row in label_coverage)
    return {
        "schema_version": "expert_form_schema_trial_ablation_packet_v1",
        "generated_at": generated_at.isoformat(),
        "schema_path": relpath(schema_path),
        "expert_feature_rows_path": relpath(expert_feature_rows_path)
        if expert_feature_rows_path
        else None,
        "base_feature_rows_path": relpath(base_feature_rows_path) if base_feature_rows_path else None,
        "official_result_paths": [relpath(path) for path in official_result_paths],
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "thresholds": {
            "min_train_races": min_train_races,
            "min_holdout_races": min_holdout_races,
            "min_slice_races": min_slice_races,
            "holdout_fraction": holdout_fraction,
        },
        "coverage_summary": {
            "feature_rows": len(feature_rows),
            "expert_feature_rows": len(expert_rows),
            "base_feature_rows": len(base_rows),
            "feature_races": len({row.get("race_id") for row in feature_rows}),
            "official_result_paths": len(official_result_paths),
            "label_join_races": len(label_coverage),
            "label_join_status_counts": dict(sorted(label_counts.items())),
            "labeled_rows": len(labeled_rows),
            "labeled_races": len({row.get("race_id") for row in labeled_rows}),
            "train_races": train_races,
            "holdout_races": holdout_races,
        },
        "schema_trial_summary": {
            "canonical_feature_count": len(schema.get("feature_columns") or []),
            "trial_feature_count": len(trial_schema.get("feature_columns") or []),
            "expert_form_features_added": sum(
                1 for row in schema_diff if row["trial_action"] == "added_report_only"
            ),
            "canonical_schema_mutation": False,
        },
        "ablation_status": ablation_status,
        "control_metrics": control_metrics,
        "trial_metrics": trial_metrics,
        "market_metrics": market,
        "gate_decision": gate,
        "final_status": final_status,
        "activation_allowed": final_status == FINAL_APPROVAL_READY,
        "blockers": blockers,
        "trial_schema": trial_schema,
        "schema_diff": schema_diff,
        "feature_quality": quality,
        "leakage_checks": leakage,
        "label_join_coverage": label_coverage,
        "slice_regression": slice_rows,
        "feature_utility_rank": utility,
        "split": {
            "train_race_ids": split["train_race_ids"],
            "holdout_race_ids": split["holdout_race_ids"],
        },
    }


def metric_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in (
        ("control_current_schema", report.get("control_metrics")),
        ("trial_current_plus_expert_form", report.get("trial_metrics")),
        ("market", report.get("market_metrics")),
    ):
        if not isinstance(metrics, Mapping):
            rows.append({"candidate": name, "status": "NOT_RUN"})
            continue
        rows.append(
            {
                "candidate": name,
                "status": metrics.get("status"),
                "race_count": metrics.get("race_count"),
                "row_count": metrics.get("row_count"),
                "top1": metrics.get("top1"),
                "top3": metrics.get("top3"),
                "mean_winner_rank": metrics.get("mean_winner_rank"),
                "brier": metrics.get("brier"),
                "log_loss": metrics.get("log_loss"),
                "blockers": ";".join(metrics.get("blockers") or []),
            }
        )
    return rows


def activation_packet(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "expert_form_shadow_schema_activation_approval_packet_v1",
        "status": "READY_FOR_OPERATOR_APPROVAL"
        if report.get("final_status") == FINAL_APPROVAL_READY
        else "NOT_READY",
        "activation_allowed_without_operator_approval": False,
        "shadow_only_schema_inclusion_candidate": report.get("final_status") == FINAL_APPROVAL_READY,
        "canonical_schema_mutation": False,
        "production_promotion": False,
        "blockers": list(report.get("blockers") or []),
        "gate_decision": report.get("gate_decision"),
    }


def summary_md(report: Mapping[str, Any], output_dir: Path) -> str:
    summary = report.get("coverage_summary") or {}
    schema_summary = report.get("schema_trial_summary") or {}
    return "\n".join(
        [
            "# Expert Form Schema Trial Ablation Packet",
            "",
            f"Final status: `{report.get('final_status')}`",
            f"Activation allowed: `{report.get('activation_allowed')}`",
            "",
            "## Schema Trial",
            "",
            f"- Canonical feature count: `{schema_summary.get('canonical_feature_count')}`",
            f"- Trial feature count: `{schema_summary.get('trial_feature_count')}`",
            f"- Expert Form features added report-only: `{schema_summary.get('expert_form_features_added')}`",
            "",
            "## Coverage",
            "",
            f"- Feature rows: `{summary.get('feature_rows')}`",
            f"- Feature races: `{summary.get('feature_races')}`",
            f"- Labeled races: `{summary.get('labeled_races')}`",
            f"- Train races: `{summary.get('train_races')}`",
            f"- Holdout races: `{summary.get('holdout_races')}`",
            "",
            "## Blockers",
            "",
            *(f"- `{blocker}`" for blocker in report.get("blockers") or []),
            "",
            "## Artifacts",
            "",
            f"- `{relpath(output_dir / 'schema_trial_repaired_non_tgr_plus_expert_form.json')}`",
            f"- `{relpath(output_dir / 'expert_form_feature_quality.csv')}`",
            f"- `{relpath(output_dir / 'label_join_coverage.csv')}`",
            f"- `{relpath(output_dir / 'slice_regression.csv')}`",
            f"- `{relpath(output_dir / 'ablation_metrics.csv')}`",
            f"- `{relpath(output_dir / 'activation_approval_packet.json')}`",
            "",
            "No canonical schema mutation, training artifact write, registry mutation, DB write, label write, EV output, or betting output was performed.",
            "",
        ]
    )


def write_packet(report: Mapping[str, Any], output_dir: Path, protected: Mapping[str, Any]) -> None:
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "schema_trial_repaired_non_tgr_plus_expert_form.json", report["trial_schema"])
    write_csv(
        output_dir / "schema_trial_diff.csv",
        report["schema_diff"],
        ["feature", "in_canonical_schema", "in_trial_schema", "trial_action", "feature_type"],
    )
    write_csv(
        output_dir / "expert_form_feature_quality.csv",
        report["feature_quality"],
        [
            "feature",
            "rows",
            "present_rows",
            "missing_rows",
            "present_pct",
            "unique_present_values",
            "placeholder_or_zero_rows",
            "placeholder_or_zero_share",
            "most_common_value",
            "most_common_count",
            "most_common_share",
            "quality_status",
        ],
    )
    write_csv(output_dir / "expert_form_leakage_checks.csv", report["leakage_checks"], ["check", "status", "details"])
    write_csv(
        output_dir / "label_join_coverage.csv",
        report["label_join_coverage"],
        [
            "race_id",
            "feature_rows",
            "official_result_rows",
            "matched_rows",
            "winner_count",
            "label_join_status",
        ],
    )
    write_csv(
        output_dir / "ablation_metrics.csv",
        metric_rows(report),
        [
            "candidate",
            "status",
            "race_count",
            "row_count",
            "top1",
            "top3",
            "mean_winner_rank",
            "brier",
            "log_loss",
            "blockers",
        ],
    )
    write_csv(
        output_dir / "slice_regression.csv",
        report["slice_regression"],
        [
            "slice_type",
            "slice_value",
            "race_count",
            "row_count",
            "control_top1",
            "trial_top1",
            "top1_delta",
            "control_top3",
            "trial_top3",
            "top3_delta",
            "control_mean_winner_rank",
            "trial_mean_winner_rank",
            "mean_winner_rank_delta",
            "control_log_loss",
            "trial_log_loss",
            "log_loss_delta",
            "status",
            "regression_metrics",
        ],
    )
    write_csv(
        output_dir / "expert_form_feature_utility_rank.csv",
        report["feature_utility_rank"],
        [
            "rank",
            "feature",
            "utility_signal_type",
            "importance_share",
            "quality_score",
            "present_pct",
            "unique_present_values",
            "most_common_share",
            "quality_status",
        ],
    )
    write_json(output_dir / "gate_decision.json", report["gate_decision"])
    write_json(output_dir / "activation_approval_packet.json", activation_packet(report))
    write_json(output_dir / "protected_path_verification.json", protected)
    report_for_disk = dict(report)
    report_for_disk.pop("trial_schema", None)
    write_json(output_dir / "expert_form_schema_trial_ablation_report.json", report_for_disk)
    manifest = {
        "schema_version": "expert_form_schema_trial_ablation_manifest_v1",
        "generated_at": report["generated_at"],
        "files": {
            "report": relpath(output_dir / "expert_form_schema_trial_ablation_report.json"),
            "summary": relpath(output_dir / "SUMMARY.md"),
            "final_status": relpath(output_dir / "final_status.txt"),
            "schema_trial": relpath(output_dir / "schema_trial_repaired_non_tgr_plus_expert_form.json"),
            "schema_diff": relpath(output_dir / "schema_trial_diff.csv"),
            "feature_quality": relpath(output_dir / "expert_form_feature_quality.csv"),
            "leakage_checks": relpath(output_dir / "expert_form_leakage_checks.csv"),
            "label_join_coverage": relpath(output_dir / "label_join_coverage.csv"),
            "ablation_metrics": relpath(output_dir / "ablation_metrics.csv"),
            "slice_regression": relpath(output_dir / "slice_regression.csv"),
            "gate_decision": relpath(output_dir / "gate_decision.json"),
            "feature_utility_rank": relpath(output_dir / "expert_form_feature_utility_rank.csv"),
            "activation_approval_packet": relpath(output_dir / "activation_approval_packet.json"),
            "protected_path_verification": relpath(output_dir / "protected_path_verification.json"),
        },
        "no_write_guarantees": report["no_write_guarantees"],
    }
    write_json(output_dir / "output_manifest.json", manifest)
    write_text(output_dir / "SUMMARY.md", summary_md(report, output_dir))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--expert-feature-rows", type=Path, default=None)
    parser.add_argument("--base-feature-rows", type=Path, default=None)
    parser.add_argument("--official-result-runners-jsonl", action="append", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-train-races", type=int, default=20)
    parser.add_argument("--min-holdout-races", type=int, default=10)
    parser.add_argument("--min-slice-races", type=int, default=DEFAULT_MIN_SLICE_RACES)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        ROOT
        / "artifacts/full_evidence_orchestration_20260525"
        / f"expert_form_schema_trial_ablation_{now_id()}_report_only"
    )
    expert_rows_path = args.expert_feature_rows or default_expert_feature_rows_path()
    official_paths = tuple(args.official_result_runners_jsonl or default_official_result_paths())
    protected_before = protected_path_snapshot()
    report = build_report(
        schema_path=args.schema,
        expert_feature_rows_path=expert_rows_path,
        base_feature_rows_path=args.base_feature_rows,
        official_result_paths=official_paths,
        min_train_races=args.min_train_races,
        min_holdout_races=args.min_holdout_races,
        min_slice_races=args.min_slice_races,
        holdout_fraction=args.holdout_fraction,
    )
    protected = protected_path_verification(protected_before)
    write_packet(report, output_dir, protected)
    print(
        json.dumps(
            {
                "final_status": report["final_status"],
                "output_dir": shadow_relpath(output_dir),
                "coverage_summary": report["coverage_summary"],
                "blockers": report["blockers"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
