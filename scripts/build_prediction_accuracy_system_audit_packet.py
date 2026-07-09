#!/usr/bin/env python3
"""Build a report-only prediction accuracy system audit packet.

The packet consolidates shadow predictions, feature rows, strict pre-jump odds,
official result joins, aggregate metrics, and recent activation/promotion
reports. It does not train, promote, mutate registries, write labels, write DB
rows, persist production predictions, emit EV, or emit betting instructions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/prediction_accuracy_system_audit_"
SCHEMA_VERSION = "prediction_accuracy_system_audit_v1"
PROBABILITY_COLUMN = "shadow_rf_calibrated_probability"
DEFAULT_MIN_MEANINGFUL_RACES = 100

DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "model_registry/current_production.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
    ROOT / "predictions",
)

NO_WRITE_GUARANTEES = {
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "production_prediction_write": False,
    "db_write": False,
    "label_write": False,
    "model_training": False,
    "model_artifact_write": False,
    "tgr_enabled": False,
    "betting_or_ev_output": False,
    "live_odds_capture": False,
    "daemon_control": False,
}

FEATURE_FAMILIES = {
    "weather_track": (
        "weather",
        "track_condition",
        "race_time_minutes_since_midnight",
    ),
    "same_distance_time": (
        "best_time_same_distance",
        "avg_time_same_distance",
        "median_time_same_distance",
        "recent_avg_time_same_distance_5",
        "prior_same_distance_start_count",
    ),
    "same_venue_time": (
        "best_time_same_venue",
        "avg_time_same_venue",
        "place_rate_same_venue",
    ),
    "same_distance_same_grade": (
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
    ),
    "sectional_weight": (
        "last_start_sectional_1st",
        "last_start_weight",
        "recent_avg_sectional_1st_5",
        "recent_best_sectional_1st_5",
        "recent_avg_weight_5",
    ),
    "history_rates": (
        "career_win_rate",
        "career_place_rate",
        "prior_start_count",
        "place_rate_same_distance",
        "career_avg_finish",
        "career_best_time",
    ),
    "target_context": (
        "box_number",
        "field_size",
        "grade_change_indicator",
        "grade_change_direction",
        "grade_strength_delta",
        "last_start_grade_normalized",
    ),
    "venue_distance": (
        "race_id",
        "race_date",
        "race_number",
        "race_time",
    ),
    "expert_form": (
        "expert_form_metadata_from_sidecar",
        "expert_form_career_starts",
        "expert_form_career_wins",
        "expert_form_win_percent",
        "expert_form_place_percent",
        "expert_form_track_distance_starts",
        "expert_form_track_distance_wins",
        "expert_form_track_distance_best_time",
        "expert_form_current_box_starts",
        "expert_form_current_box_wins",
    ),
}


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def normalize_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^\s*\d{1,2}\s*[\.)\-:]\s*", "", text)
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def runner_key(row: Mapping[str, Any]) -> tuple[str, int | None, str]:
    box_value = row.get("box", row.get("box_number"))
    try:
        box = int(box_value)
    except (TypeError, ValueError):
        box = None
    return (str(row.get("race_id") or "").strip(), box, normalize_name(row.get("dog_name")))


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return stripped != "" and stripped.casefold() not in {"none", "null", "nan", "n/a"}
    return True


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def pct(count: int, total: int) -> float | None:
    return count / total if total else None


def sha256_file(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "type": "file",
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        entries: list[dict[str, Any]] = []
        for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
            if item.is_file():
                entries.append(
                    {
                        "type": "file",
                        "path": relpath(item),
                        "bytes": item.stat().st_size,
                        "sha256": sha256_file(item),
                    }
                )
            elif item.is_dir():
                entries.append({"type": "directory", "path": relpath(item)})
        digest_input = "\n".join(json.dumps(entry, sort_keys=True) for entry in entries)
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        return {
            "type": "directory",
            "exists": True,
            "file_count": sum(1 for entry in entries if entry["type"] == "file"),
            "entry_count": len(entries),
            "listing_sha256": digest,
        }
    return {"type": "missing", "exists": False}


def protected_path_states(paths: Sequence[Path] | None = None) -> dict[str, dict[str, Any]]:
    paths = DEFAULT_PROTECTED_PATHS if paths is None else paths
    return {relpath(path) or str(path): path_state(path) for path in paths}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path | None, failures: list[str], label: str) -> dict[str, Any]:
    if path is None:
        failures.append(f"{label}_path_missing")
        return {}
    if not path.exists():
        failures.append(f"{label}_missing:{relpath(path)}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - packet records exact failure.
        failures.append(f"{label}_unreadable:{type(exc).__name__}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"{label}_root_not_object")
        return {}
    return value


def read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - optional sidecar, main report records absence elsewhere.
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path | None, failures: list[str], label: str) -> list[dict[str, Any]]:
    if path is None:
        failures.append(f"{label}_path_missing")
        return []
    if not path.exists():
        failures.append(f"{label}_missing:{relpath(path)}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
                else:
                    failures.append(f"{label}_non_object_row:{line_number}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{label}_unreadable:{type(exc).__name__}")
        return []
    return rows


def read_feature_rows(path: Path | None, failures: list[str]) -> list[dict[str, Any]]:
    if path is None:
        failures.append("feature_rows_path_missing")
        return []
    if not path.exists():
        failures.append(f"feature_rows_missing:{relpath(path)}")
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"feature_rows_unreadable:{type(exc).__name__}")
        return []
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        return [row for row in value["rows"] if isinstance(row, dict)]
    failures.append("feature_rows_root_not_list")
    return []


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_prediction_accuracy_system_audit_artifact:{relative}")
    return logical.absolute()


def latest_path(paths: Sequence[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def discover_latest_shadow_run(evidence_root: Path) -> Path | None:
    candidates = [
        path.parent
        for path in evidence_root.rglob("shadow_predictions.jsonl")
        if (path.parent / "shadow_manifest.json").exists()
    ]
    return latest_path(candidates)


def discover_latest_file(evidence_root: Path, pattern: str) -> Path | None:
    return latest_path(list(evidence_root.glob(pattern)))


def artifact_paths(
    *,
    evidence_root: Path,
    shadow_run_dir: Path | None,
    result_join_path: Path | None,
    odds_snapshot_path: Path | None,
    aggregate_report_path: Path | None,
    promotion_distance_report_path: Path | None,
    feature_activation_report_path: Path | None,
    expert_form_ablation_report_path: Path | None,
) -> dict[str, Path | None]:
    shadow_run_dir = shadow_run_dir or discover_latest_shadow_run(evidence_root)
    active_policy_path = None
    if shadow_run_dir:
        direct_policy = shadow_run_dir / "active_feature_policy_report.json"
        nested_policy = shadow_run_dir / "shadow_score_live/active_feature_policy_report.json"
        active_policy_path = direct_policy if direct_policy.exists() else nested_policy
    resolved_odds_snapshot = odds_snapshot_path or discover_latest_file(
        evidence_root,
        "shadow_odds_snapshot_*/shadow_odds_snapshot.jsonl",
    )
    odds_snapshot_report = (
        resolved_odds_snapshot.parent / "shadow_odds_snapshot_report.json"
        if resolved_odds_snapshot is not None
        else None
    )
    return {
        "shadow_run_dir": shadow_run_dir,
        "shadow_predictions": shadow_run_dir / "shadow_predictions.jsonl" if shadow_run_dir else None,
        "shadow_feature_rows": shadow_run_dir / "shadow_feature_rows.json" if shadow_run_dir else None,
        "shadow_manifest": shadow_run_dir / "shadow_manifest.json" if shadow_run_dir else None,
        "active_feature_policy_report": active_policy_path,
        "result_join_rows": result_join_path
        or discover_latest_file(evidence_root, "forward_shadow_result_join_*/joined_shadow_predictions.jsonl"),
        "aggregate_report": aggregate_report_path
        or discover_latest_file(evidence_root, "forward_shadow_result_aggregate_*/forward_shadow_result_aggregate_report.json"),
        "odds_snapshot": resolved_odds_snapshot,
        "odds_snapshot_report": odds_snapshot_report,
        "promotion_distance_report": promotion_distance_report_path
        or discover_latest_file(evidence_root, "promotion_distance_report_*/promotion_distance_report.json"),
        "feature_activation_report": feature_activation_report_path
        or discover_latest_file(evidence_root, "shadow_feature_activation_gate_*/feature_activation_gate_report.json"),
        "expert_form_ablation_report": expert_form_ablation_report_path
        or discover_latest_file(evidence_root, "expert_form_schema_trial_ablation_*/expert_form_schema_trial_ablation_report.json"),
    }


def index_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int | None, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, int | None, str], Mapping[str, Any]] = {}
    for row in rows:
        key = runner_key(row)
        if key[0] and key[1] is not None and key[2]:
            indexed[key] = row
    return indexed


def odds_win_price(row: Mapping[str, Any] | None) -> float | None:
    if not row:
        return None
    snapshot = row.get("odds_snapshot")
    if isinstance(snapshot, Mapping):
        value = safe_float(snapshot.get("market_odds_win"))
        if value is not None:
            return value
    return safe_float(row.get("market_odds_win"))


def build_joined_runner_rows(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    odds_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    features_by_key = index_by_key(feature_rows)
    results_by_key = index_by_key(result_rows)
    odds_by_key = index_by_key(odds_rows)

    raw_implied_by_race: dict[str, dict[tuple[str, int | None, str], float]] = defaultdict(dict)
    for key, odds_row in odds_by_key.items():
        price = odds_win_price(odds_row)
        if price and price > 0:
            raw_implied_by_race[key[0]][key] = 1.0 / price
    implied_sum_by_race = {
        race_id: sum(values.values()) for race_id, values in raw_implied_by_race.items()
    }

    joined: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        key = runner_key(prediction)
        feature = features_by_key.get(key)
        result = results_by_key.get(key)
        odds = odds_by_key.get(key)
        price = odds_win_price(odds)
        raw_implied = 1.0 / price if price and price > 0 else None
        implied_sum = implied_sum_by_race.get(key[0])
        normalized_implied = raw_implied / implied_sum if raw_implied is not None and implied_sum else None
        feature_safe = feature.get("metadata_is_leakage_safe") if isinstance(feature, Mapping) else None
        joined.append(
            {
                "race_id": prediction.get("race_id"),
                "box": prediction.get("box"),
                "dog_name": prediction.get("dog_name"),
                "predicted_rank": prediction.get("predicted_rank"),
                "shadow_rf_calibrated_probability": prediction.get(PROBABILITY_COLUMN),
                "shadow_rf_uncalibrated_probability": prediction.get("shadow_rf_uncalibrated_probability"),
                "is_winner": result.get("is_winner") if result else None,
                "finish_position": result.get("finish_position") if result else None,
                "label_join_status": "JOINED_OFFICIAL_RESULT" if result else "DATA_MISSING_LABEL",
                "result_identity_match_status": result.get("identity_match_status") if result else None,
                "odds_join_status": (
                    str(odds.get("odds_match_status")) if odds else "DATA_MISSING_ODDS"
                ),
                "odds_provenance_status": odds.get("odds_provenance_status") if odds else None,
                "market_odds_win": price,
                "market_implied_probability_raw": raw_implied,
                "market_implied_probability_normalized": normalized_implied,
                "feature_row_join_status": "JOINED_FEATURE_ROW" if feature else "DATA_MISSING_FEATURE_ROW",
                "metadata_is_leakage_safe": feature_safe,
                "weather_present": bool(feature and is_present(feature.get("weather"))),
                "track_condition_present": bool(feature and is_present(feature.get("track_condition"))),
                "race_time_minutes_since_midnight_present": bool(
                    feature and is_present(feature.get("race_time_minutes_since_midnight"))
                ),
                "expert_form_sidecar_present": bool(
                    feature and feature.get("expert_form_metadata_from_sidecar") is True
                ),
            }
        )
    return joined


def grouped_by_race(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        race_id = str(row.get("race_id") or "")
        if race_id:
            grouped[race_id].append(row)
    return dict(grouped)


def metric_summary(
    rows: Sequence[Mapping[str, Any]],
    probability_key: str,
    rank_key: str | None = None,
) -> dict[str, Any]:
    safe_races = 0
    top1_hits = 0
    top3_hits = 0
    winner_ranks: list[int] = []
    logloss_terms: list[float] = []
    brier_terms: list[float] = []
    skipped: Counter[str] = Counter()
    safe_runner_count = 0

    for race_id, race_rows in grouped_by_race(rows).items():
        labeled = [row for row in race_rows if row.get("is_winner") is not None]
        if not labeled:
            skipped["missing_labels"] += 1
            continue
        winners = [row for row in labeled if row.get("is_winner") is True]
        if len(winners) != 1:
            skipped["winner_row_count_not_one"] += 1
            continue
        scored = [
            row for row in labeled if safe_float(row.get(probability_key)) is not None
        ]
        if len(scored) != len(labeled):
            skipped["missing_probability"] += 1
            continue

        if rank_key:
            scored.sort(key=lambda row: int(row.get(rank_key) or 999))
        else:
            scored.sort(key=lambda row: safe_float(row.get(probability_key)) or 0.0, reverse=True)

        safe_races += 1
        safe_runner_count += len(scored)
        winner = winners[0]
        winner_index = next(
            index for index, row in enumerate(scored, start=1) if row is winner
        )
        winner_ranks.append(winner_index)
        top1_hits += int(winner_index == 1)
        top3_hits += int(winner_index <= 3)
        winner_probability = safe_float(winner.get(probability_key)) or 0.0
        logloss_terms.append(-math.log(min(max(winner_probability, 1e-15), 1 - 1e-15)))
        for row in scored:
            label = 1.0 if row.get("is_winner") is True else 0.0
            probability = safe_float(row.get(probability_key)) or 0.0
            brier_terms.append((probability - label) ** 2)

    return {
        "status": "COMPUTED" if safe_races else "DATA_MISSING_LABELS_OR_PROBABILITIES",
        "safe_race_count": safe_races,
        "safe_runner_count": safe_runner_count,
        "top1": top1_hits / safe_races if safe_races else None,
        "top3": top3_hits / safe_races if safe_races else None,
        "mean_winner_rank": sum(winner_ranks) / len(winner_ranks) if winner_ranks else None,
        "logloss": sum(logloss_terms) / len(logloss_terms) if logloss_terms else None,
        "brier": sum(brier_terms) / len(brier_terms) if brier_terms else None,
        "winner_ranks": winner_ranks,
        "skipped_race_reason_counts": dict(sorted(skipped.items())),
    }


def data_point_provenance_rows(
    *,
    paths: Mapping[str, Path | None],
    counts: Mapping[str, int],
    reports: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    aggregate_metrics = reports.get("aggregate", {}).get("aggregate_forward_metrics", {})
    promotion = reports.get("promotion", {})
    expert = reports.get("expert_form", {})
    feature_activation = reports.get("feature_activation", {})
    odds_report = reports.get("odds_report", {})
    return [
        {
            "data_point": "shadow_predictions",
            "source_path": relpath(paths.get("shadow_predictions")),
            "rows": counts.get("prediction_rows", 0),
            "status": "AVAILABLE" if counts.get("prediction_rows", 0) else "DATA_MISSING",
            "join_key": "race_id + box + normalized dog_name",
            "notes": "report-only shadow scorer output",
        },
        {
            "data_point": "shadow_feature_rows",
            "source_path": relpath(paths.get("shadow_feature_rows")),
            "rows": counts.get("feature_rows", 0),
            "status": "AVAILABLE" if counts.get("feature_rows", 0) else "DATA_MISSING",
            "join_key": "race_id + box_number + normalized dog_name",
            "notes": "feature rows include sidecar provenance flags where present",
        },
        {
            "data_point": "official_result_labels",
            "source_path": relpath(paths.get("result_join_rows")),
            "rows": counts.get("result_rows", 0),
            "status": "AVAILABLE" if counts.get("result_rows", 0) else "DATA_MISSING",
            "join_key": "race_id + box + normalized dog_name",
            "notes": "TheDogs result join rows only; no label writes",
        },
        {
            "data_point": "strict_prejump_odds",
            "source_path": relpath(paths.get("odds_snapshot")),
            "rows": counts.get("odds_rows", 0),
            "status": "AVAILABLE" if counts.get("odds_rows", 0) else "DATA_MISSING",
            "join_key": "race_id + box + normalized dog_name",
            "notes": f"latest odds report status={odds_report.get('final_status')}",
        },
        {
            "data_point": "aggregate_accuracy_metrics",
            "source_path": relpath(paths.get("aggregate_report")),
            "rows": aggregate_metrics.get("safe_joined_runner_count"),
            "status": "AVAILABLE"
            if aggregate_metrics.get("safe_joined_race_count")
            else "DATA_MISSING",
            "join_key": "selected unique safe result-joined races",
            "notes": f"safe_races={aggregate_metrics.get('safe_joined_race_count')}",
        },
        {
            "data_point": "market_benchmark",
            "source_path": relpath(paths.get("promotion_distance_report")),
            "rows": promotion.get("rolling_sample", {}).get("sample_runner_rows"),
            "status": "AVAILABLE" if promotion.get("market_benchmark") else "DATA_MISSING",
            "join_key": "promotion-distance rolling sample",
            "notes": f"best_candidate={promotion.get('market_benchmark', {}).get('best_candidate_key')}",
        },
        {
            "data_point": "feature_activation_gate",
            "source_path": relpath(paths.get("feature_activation_report")),
            "rows": len(feature_activation.get("features") or []),
            "status": feature_activation.get("final_status") or "DATA_MISSING",
            "join_key": "feature name",
            "notes": "activation remains report-only",
        },
        {
            "data_point": "expert_form_ablation",
            "source_path": relpath(paths.get("expert_form_ablation_report")),
            "rows": expert.get("coverage_summary", {}).get("feature_rows"),
            "status": expert.get("final_status") or "DATA_MISSING",
            "join_key": "ablation train/holdout rows",
            "notes": f"activation_allowed={expert.get('activation_allowed')}",
        },
    ]


def missingness_rows(joined_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = len(joined_rows)

    def row(metric: str, count: int, reason: str = "") -> dict[str, Any]:
        return {
            "metric": metric,
            "present_count": count,
            "total_rows": total,
            "present_pct": pct(count, total),
            "status": "PASS" if total and count == total else ("PARTIAL" if count else "DATA_MISSING"),
            "reason": reason,
        }

    return [
        row("feature_row_joined", sum(r.get("feature_row_join_status") == "JOINED_FEATURE_ROW" for r in joined_rows)),
        row("official_label_joined", sum(r.get("label_join_status") == "JOINED_OFFICIAL_RESULT" for r in joined_rows)),
        row("strict_prejump_odds_joined", sum(r.get("odds_join_status") == "valid_pre_jump_dog_odds" for r in joined_rows)),
        row("weather_present", sum(bool(r.get("weather_present")) for r in joined_rows), "feature inactive if train all-missing"),
        row("track_condition_present", sum(bool(r.get("track_condition_present")) for r in joined_rows), "feature inactive if train all-missing"),
        row(
            "race_time_minutes_since_midnight_present",
            sum(bool(r.get("race_time_minutes_since_midnight_present")) for r in joined_rows),
            "feature inactive if train all-missing",
        ),
        row(
            "expert_form_sidecar_present",
            sum(bool(r.get("expert_form_sidecar_present")) for r in joined_rows),
        ),
    ]


def feature_family_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    feature_activation_report: Mapping[str, Any],
    expert_form_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    total = len(feature_rows)
    inactive = set(str(item) for item in manifest.get("inactive_features_due_to_train_all_missing") or [])
    activation_by_feature = {
        str(item.get("feature")): item
        for item in feature_activation_report.get("features") or []
        if isinstance(item, Mapping)
    }
    expert_control = expert_form_report.get("control_metrics") or {}
    expert_trial = expert_form_report.get("trial_metrics") or {}
    output = []
    for family, fields in FEATURE_FAMILIES.items():
        rows_with_any = 0
        values: list[str] = []
        present_by_field = {}
        for field in fields:
            present = [row.get(field) for row in feature_rows if is_present(row.get(field))]
            present_by_field[field] = len(present)
            values.extend(str(value) for value in present)
        for feature_row in feature_rows:
            if any(is_present(feature_row.get(field)) for field in fields):
                rows_with_any += 1
        most_common_count = Counter(values).most_common(1)[0][1] if values else 0
        activation_reasons = []
        for field in fields:
            gate = activation_by_feature.get(field)
            if gate:
                activation_reasons.extend(str(reason) for reason in gate.get("fail_reasons") or [])
        utility_notes = ""
        if family == "expert_form" and expert_control and expert_trial:
            utility_notes = (
                f"control_top1={expert_control.get('top1')}; "
                f"trial_top1={expert_trial.get('top1')}; "
                f"activation_allowed={expert_form_report.get('activation_allowed')}"
            )
        output.append(
            {
                "feature_family": family,
                "field_count": len(fields),
                "rows_with_any_present": rows_with_any,
                "total_feature_rows": total,
                "coverage_pct": pct(rows_with_any, total),
                "unique_present_values": len(set(values)),
                "default_dominance_pct": (most_common_count / len(values)) if values else None,
                "inactive_train_all_missing_fields": ";".join(field for field in fields if field in inactive),
                "activation_decision": (
                    "KEEP_QUARANTINED"
                    if any(field in activation_by_feature for field in fields)
                    else "REVIEW_ONLY"
                ),
                "activation_fail_reasons": ";".join(sorted(set(activation_reasons))),
                "present_by_field_json": json.dumps(present_by_field, sort_keys=True),
                "utility_notes": utility_notes,
            }
        )
    return output


def odds_market_ledger_rows(
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    odds_report: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    current_market_metrics = metric_summary(
        joined_rows,
        "market_implied_probability_normalized",
        rank_key=None,
    )
    shadow_current_metrics = metric_summary(joined_rows, PROBABILITY_COLUMN, "predicted_rank")
    expert_form_rows = [row for row in joined_rows if row.get("expert_form_sidecar_present")]
    expert_form_market_metrics = metric_summary(
        expert_form_rows,
        "market_implied_probability_normalized",
        rank_key=None,
    )
    expert_form_shadow_metrics = metric_summary(
        expert_form_rows,
        PROBABILITY_COLUMN,
        "predicted_rank",
    )
    benchmark = promotion_report.get("market_benchmark") or {}
    rolling = promotion_report.get("rolling_sample") or {}
    return [
        {
            "scope": "current_shadow_rows",
            "runner_rows": len(joined_rows),
            "race_count": len(grouped_by_race(joined_rows)),
            "valid_prejump_odds_rows": sum(
                row.get("odds_join_status") == "valid_pre_jump_dog_odds" for row in joined_rows
            ),
            "complete_valid_odds_races": odds_report.get("races_with_complete_valid_prejump_odds"),
            "market_metrics_status": current_market_metrics["status"],
            "market_top1": current_market_metrics["top1"],
            "market_top3": current_market_metrics["top3"],
            "market_logloss": current_market_metrics["logloss"],
            "shadow_top1_same_rows": shadow_current_metrics["top1"],
            "shadow_top3_same_rows": shadow_current_metrics["top3"],
            "shadow_logloss_same_rows": shadow_current_metrics["logloss"],
            "notes": "current row labels may be pending",
        },
        {
            "scope": "current_expert_form_sidecar_rows",
            "runner_rows": len(expert_form_rows),
            "race_count": len(grouped_by_race(expert_form_rows)),
            "valid_prejump_odds_rows": sum(
                row.get("odds_join_status") == "valid_pre_jump_dog_odds"
                for row in expert_form_rows
            ),
            "complete_valid_odds_races": None,
            "market_metrics_status": expert_form_market_metrics["status"],
            "market_top1": expert_form_market_metrics["top1"],
            "market_top3": expert_form_market_metrics["top3"],
            "market_logloss": expert_form_market_metrics["logloss"],
            "shadow_top1_same_rows": expert_form_shadow_metrics["top1"],
            "shadow_top3_same_rows": expert_form_shadow_metrics["top3"],
            "shadow_logloss_same_rows": expert_form_shadow_metrics["logloss"],
            "notes": "Expert Form sidecar rows use the same strict pre-jump odds join",
        },
        {
            "scope": "promotion_distance_rolling_sample",
            "runner_rows": rolling.get("sample_runner_rows"),
            "race_count": rolling.get("sample_race_count"),
            "valid_prejump_odds_rows": None,
            "complete_valid_odds_races": None,
            "market_metrics_status": "AVAILABLE" if benchmark else "DATA_MISSING",
            "market_top1": None,
            "market_top3": None,
            "market_logloss": None,
            "shadow_top1_same_rows": None,
            "shadow_top3_same_rows": None,
            "shadow_logloss_same_rows": None,
            "notes": f"best_candidate={benchmark.get('best_candidate_key')}; best_non_market={benchmark.get('best_non_market_candidate_key')}",
        },
    ]


def calibration_report(
    *,
    joined_rows: Sequence[Mapping[str, Any]],
    aggregate_report: Mapping[str, Any],
) -> dict[str, Any]:
    probability_sum_errors = []
    for race_rows in grouped_by_race(joined_rows).values():
        values = [safe_float(row.get(PROBABILITY_COLUMN)) for row in race_rows]
        if values and all(value is not None for value in values):
            probability_sum_errors.append(abs(sum(value for value in values if value is not None) - 1.0))
    aggregate_calibration = aggregate_report.get("aggregate_calibration_review") or {}
    aggregate_metrics = aggregate_report.get("aggregate_forward_metrics") or {}
    return {
        "schema_version": "prediction_accuracy_system_calibration_review_v1",
        "current_probability_sum_max_error": max(probability_sum_errors) if probability_sum_errors else None,
        "current_probability_sum_status": (
            "PASS" if probability_sum_errors and max(probability_sum_errors) <= 1e-6 else "DATA_MISSING"
        ),
        "aggregate_safe_joined_race_count": aggregate_metrics.get("safe_joined_race_count"),
        "aggregate_safe_joined_runner_count": aggregate_metrics.get("safe_joined_runner_count"),
        "aggregate_brier": aggregate_calibration.get("brier"),
        "aggregate_logloss": aggregate_calibration.get("logloss"),
        "aggregate_slope_intercept": aggregate_calibration.get("slope_intercept"),
        "aggregate_reliability_bins": aggregate_calibration.get("reliability_bins") or [],
    }


def slice_metrics(joined_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slices: list[dict[str, Any]] = []
    for slice_name, predicate in {
        "all_current_rows": lambda row: True,
        "current_rows_with_strict_prejump_odds": lambda row: row.get("odds_join_status")
        == "valid_pre_jump_dog_odds",
        "current_rows_with_expert_form_sidecar": lambda row: bool(row.get("expert_form_sidecar_present")),
        "current_rows_with_weather_or_track": lambda row: bool(row.get("weather_present"))
        or bool(row.get("track_condition_present")),
    }.items():
        rows = [row for row in joined_rows if predicate(row)]
        metrics = metric_summary(rows, PROBABILITY_COLUMN, "predicted_rank")
        slices.append(
            {
                "slice": slice_name,
                "runner_rows": len(rows),
                "race_count": len(grouped_by_race(rows)),
                "metrics": metrics,
            }
        )
    return {"schema_version": "prediction_accuracy_system_slice_metrics_v1", "slices": slices}


def blocker_rows(
    *,
    joined_rows: Sequence[Mapping[str, Any]],
    aggregate_report: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
    feature_activation_report: Mapping[str, Any],
    expert_form_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    aggregate_metrics = aggregate_report.get("aggregate_forward_metrics") or {}
    calibration = aggregate_report.get("aggregate_calibration_review") or {}
    promotion_blockers = promotion_report.get("blockers") or []
    inactive_features = [str(item) for item in manifest.get("inactive_features_due_to_train_all_missing") or []]
    label_joined = sum(row.get("label_join_status") == "JOINED_OFFICIAL_RESULT" for row in joined_rows)
    odds_joined = sum(row.get("odds_join_status") == "valid_pre_jump_dog_odds" for row in joined_rows)
    blockers = [
        {
            "rank": 1,
            "blocker": "market_and_promotion_gate",
            "status": promotion_report.get("final_status") or "DATA_MISSING",
            "evidence": ";".join(str(item) for item in promotion_blockers),
            "next_action": "run report-only model tournament on identical label/odds/prediction rows",
        },
        {
            "rank": 2,
            "blocker": "aggregate_model_quality",
            "status": aggregate_report.get("final_status") or "DATA_MISSING",
            "evidence": (
                f"safe_races={aggregate_metrics.get('safe_joined_race_count')}; "
                f"top1={aggregate_metrics.get('top1')}; top3={aggregate_metrics.get('top3')}; "
                f"logloss={aggregate_metrics.get('logloss')}; "
                f"calibration_slope={((calibration.get('slope_intercept') or {}).get('slope'))}"
            ),
            "next_action": "compare current model, market-only, blends, and candidate feature sets on one joined table",
        },
        {
            "rank": 3,
            "blocker": "current_result_label_join",
            "status": "JOINED" if label_joined else "DATA_MISSING",
            "evidence": f"joined_label_rows={label_joined}; current_runner_rows={len(joined_rows)}",
            "next_action": "wait for official results or run safe result join when races have resulted",
        },
        {
            "rank": 4,
            "blocker": "strict_odds_join",
            "status": "JOINED" if odds_joined else "DATA_MISSING",
            "evidence": f"valid_prejump_odds_rows={odds_joined}; current_runner_rows={len(joined_rows)}",
            "next_action": "repair odds join only if rows exist but fail exact race/box/name matching",
        },
        {
            "rank": 5,
            "blocker": "inactive_train_all_missing_features",
            "status": "BLOCKED" if inactive_features else "CLEAR",
            "evidence": ";".join(inactive_features),
            "next_action": "backfill or retrain only after provenance-safe coverage and report-only ablation pass",
        },
        {
            "rank": 6,
            "blocker": "feature_activation_gate",
            "status": feature_activation_report.get("final_status") or "DATA_MISSING",
            "evidence": json.dumps(feature_activation_report.get("fail_reason_summary") or {}, sort_keys=True),
            "next_action": "keep quarantined until train/holdout coverage and metric comparison pass",
        },
        {
            "rank": 7,
            "blocker": "expert_form_activation",
            "status": expert_form_report.get("final_status") or "DATA_MISSING",
            "evidence": (
                f"activation_allowed={expert_form_report.get('activation_allowed')}; "
                f"control_top1={(expert_form_report.get('control_metrics') or {}).get('top1')}; "
                f"trial_top1={(expert_form_report.get('trial_metrics') or {}).get('top1')}"
            ),
            "next_action": "do not activate until rank-first gate passes on meaningful sample",
        },
    ]
    return blockers


def decide_next_status(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    joined_rows: Sequence[Mapping[str, Any]],
    aggregate_report: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
    min_meaningful_races: int,
) -> dict[str, Any]:
    if not prediction_rows:
        return {
            "runtime_status": "SHADOW_SCORER_RUNTIME_BLOCKED",
            "next_decision": "DATA_MISSING",
            "reason": "current shadow scorer emitted zero prediction rows",
        }

    current_label_races = len(
        {
            str(row.get("race_id"))
            for row in joined_rows
            if row.get("label_join_status") == "JOINED_OFFICIAL_RESULT"
        }
    )
    current_odds_races = len(
        {
            str(row.get("race_id"))
            for row in joined_rows
            if row.get("odds_join_status") == "valid_pre_jump_dog_odds"
        }
    )
    current_odds_evidence_races = len(
        {
            str(row.get("race_id"))
            for row in joined_rows
            if row.get("odds_join_status")
            and row.get("odds_join_status") != "DATA_MISSING_ODDS"
        }
    )
    aggregate_metrics = aggregate_report.get("aggregate_forward_metrics") or {}
    aggregate_safe_races = int(aggregate_metrics.get("safe_joined_race_count") or 0)
    rolling = promotion_report.get("rolling_sample") or {}
    rolling_races = int(rolling.get("sample_race_count") or 0)
    market_available = bool((promotion_report.get("market_benchmark") or {}).get("best_candidate_key"))

    if (
        current_label_races >= min_meaningful_races
        and current_odds_races >= min_meaningful_races
        and market_available
    ):
        return {
            "runtime_status": "SHADOW_SCORER_RUNTIME_REPAIRED",
            "next_decision": "READY_FOR_REPORT_ONLY_MODEL_TOURNAMENT",
            "reason": "current predictions, labels, strict odds, and market evidence align on a meaningful sample",
        }
    if current_odds_evidence_races and current_odds_races == 0:
        return {
            "runtime_status": "SHADOW_SCORER_RUNTIME_REPAIRED",
            "next_decision": "DATA_JOIN_REPAIR_NEXT",
            "reason": (
                "current odds evidence exists but zero current rows passed the strict "
                "pre-jump odds join"
            ),
        }
    if current_label_races == 0 and current_odds_races > 0:
        return {
            "runtime_status": "SHADOW_SCORER_RUNTIME_REPAIRED",
            "next_decision": "KEEP_COLLECTING_ONLY",
            "reason": "current strict odds/features joined but current official result labels are not joined yet",
        }
    if aggregate_safe_races >= min_meaningful_races and rolling_races >= min_meaningful_races and market_available:
        return {
            "runtime_status": "SHADOW_SCORER_RUNTIME_REPAIRED",
            "next_decision": "DATA_JOIN_REPAIR_NEXT",
            "reason": (
                "historical aggregate and market samples are meaningful, but current "
                "labels/strict odds/features do not yet align on identical runner rows"
            ),
        }
    return {
        "runtime_status": "SHADOW_SCORER_RUNTIME_REPAIRED",
        "next_decision": "DATA_JOIN_REPAIR_NEXT",
        "reason": "predictions exist but labels, strict odds, features, or market baseline do not align",
    }


def output_manifest(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        rows.append({"path": relpath(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def build_summary(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    counts = report["counts"]
    aggregate = report.get("aggregate_metrics") or {}
    return "\n".join(
        [
            "# Prediction Accuracy System Audit",
            "",
            f"- Runtime status: `{decision.get('runtime_status')}`",
            f"- Next decision: `{decision.get('next_decision')}`",
            f"- Reason: `{decision.get('reason')}`",
            f"- Current prediction rows: `{counts.get('prediction_rows')}`",
            f"- Current feature rows: `{counts.get('feature_rows')}`",
            f"- Current result-join rows: `{counts.get('result_rows')}`",
            f"- Current odds rows: `{counts.get('odds_rows')}`",
            f"- Aggregate safe joined races: `{aggregate.get('safe_joined_race_count')}`",
            f"- Aggregate Top1: `{aggregate.get('top1')}`",
            f"- Aggregate Top3: `{aggregate.get('top3')}`",
            f"- Aggregate LogLoss: `{aggregate.get('logloss')}`",
            f"- Protected paths unchanged: `{report.get('protected_paths_unchanged')}`",
            "",
            "No production promotion, registry mutation, DB writes, label writes, TGR enablement, betting output, EV output, model training, or production prediction write was performed.",
            "",
        ]
    )


def board_ready_recommendation(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    blockers = report.get("blocker_ranking") or []
    lines = [
        "# Board-Ready Recommendation",
        "",
        f"Decision: `{decision.get('next_decision')}`.",
        "",
        f"Runtime: `{decision.get('runtime_status')}`.",
        "",
        "Top blockers:",
    ]
    for blocker in blockers[:5]:
        lines.append(
            f"- `{blocker.get('blocker')}`: `{blocker.get('status')}` - {blocker.get('next_action')}"
        )
    lines.extend(
        [
            "",
            "Recommended next action: run a report-only model tournament/evaluation contract on identical rows before any activation, promotion, EV, betting, registry, DB, or label mutation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_packet(
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    shadow_run_dir: Path | None = None,
    result_join_path: Path | None = None,
    odds_snapshot_path: Path | None = None,
    aggregate_report_path: Path | None = None,
    promotion_distance_report_path: Path | None = None,
    feature_activation_report_path: Path | None = None,
    expert_form_ablation_report_path: Path | None = None,
    min_meaningful_races: int = DEFAULT_MIN_MEANINGFUL_RACES,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    failures: list[str] = []
    paths = artifact_paths(
        evidence_root=evidence_root,
        shadow_run_dir=shadow_run_dir,
        result_join_path=result_join_path,
        odds_snapshot_path=odds_snapshot_path,
        aggregate_report_path=aggregate_report_path,
        promotion_distance_report_path=promotion_distance_report_path,
        feature_activation_report_path=feature_activation_report_path,
        expert_form_ablation_report_path=expert_form_ablation_report_path,
    )
    prediction_rows = read_jsonl(paths["shadow_predictions"], failures, "shadow_predictions")
    feature_rows = read_feature_rows(paths["shadow_feature_rows"], failures)
    result_rows = read_jsonl(paths["result_join_rows"], failures, "result_join_rows")
    odds_rows = read_jsonl(paths["odds_snapshot"], failures, "odds_snapshot_rows")
    manifest = read_json(paths["shadow_manifest"], failures, "shadow_manifest")
    active_policy = read_optional_json(paths["active_feature_policy_report"])
    feature_policy = active_policy or manifest
    aggregate = read_json(paths["aggregate_report"], failures, "aggregate_report")
    odds_report = read_json(paths["odds_snapshot_report"], failures, "odds_snapshot_report")
    promotion = read_json(paths["promotion_distance_report"], failures, "promotion_distance_report")
    feature_activation = read_json(paths["feature_activation_report"], failures, "feature_activation_report")
    expert_form = read_json(paths["expert_form_ablation_report"], failures, "expert_form_ablation_report")

    joined_rows = build_joined_runner_rows(
        prediction_rows=prediction_rows,
        feature_rows=feature_rows,
        result_rows=result_rows,
        odds_rows=odds_rows,
    )
    counts = {
        "prediction_rows": len(prediction_rows),
        "feature_rows": len(feature_rows),
        "result_rows": len(result_rows),
        "odds_rows": len(odds_rows),
        "joined_runner_rows": len(joined_rows),
        "joined_feature_rows": sum(row.get("feature_row_join_status") == "JOINED_FEATURE_ROW" for row in joined_rows),
        "joined_label_rows": sum(row.get("label_join_status") == "JOINED_OFFICIAL_RESULT" for row in joined_rows),
        "joined_valid_prejump_odds_rows": sum(row.get("odds_join_status") == "valid_pre_jump_dog_odds" for row in joined_rows),
    }
    reports = {
        "aggregate": aggregate,
        "odds_report": odds_report,
        "promotion": promotion,
        "feature_activation": feature_activation,
        "expert_form": expert_form,
    }
    provenance = data_point_provenance_rows(paths=paths, counts=counts, reports=reports)
    missingness = missingness_rows(joined_rows)
    family_coverage = feature_family_rows(
        feature_rows,
        manifest=feature_policy,
        feature_activation_report=feature_activation,
        expert_form_report=expert_form,
    )
    odds_ledger = odds_market_ledger_rows(
        joined_rows,
        odds_report=odds_report,
        promotion_report=promotion,
    )
    calibration = calibration_report(joined_rows=joined_rows, aggregate_report=aggregate)
    slices = slice_metrics(joined_rows)
    blockers = blocker_rows(
        joined_rows=joined_rows,
        aggregate_report=aggregate,
        promotion_report=promotion,
        feature_activation_report=feature_activation,
        expert_form_report=expert_form,
        manifest=feature_policy,
    )
    decision = decide_next_status(
        prediction_rows=prediction_rows,
        joined_rows=joined_rows,
        aggregate_report=aggregate,
        promotion_report=promotion,
        min_meaningful_races=min_meaningful_races,
    )
    aggregate_metrics = aggregate.get("aggregate_forward_metrics") or {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_paths": {key: relpath(path) for key, path in paths.items()},
        "counts": counts,
        "failures": failures,
        "decision": decision,
        "aggregate_metrics": {
            key: aggregate_metrics.get(key)
            for key in (
                "safe_joined_race_count",
                "safe_joined_runner_count",
                "pending_race_count",
                "unsafe_match_count",
                "top1",
                "top3",
                "mean_winner_rank",
                "logloss",
                "brier",
            )
        },
        "promotion_distance": {
            "final_status": promotion.get("final_status"),
            "promotion_ready": promotion.get("promotion_ready"),
            "blockers": promotion.get("blockers") or [],
            "market_benchmark": promotion.get("market_benchmark"),
            "rolling_sample": promotion.get("rolling_sample"),
        },
        "feature_activation": {
            "final_status": feature_activation.get("final_status"),
            "activation_allowed_features": feature_activation.get("activation_allowed_features") or [],
            "kept_quarantined_features": feature_activation.get("kept_quarantined_features") or [],
        },
        "expert_form_ablation": {
            "final_status": expert_form.get("final_status"),
            "activation_allowed": expert_form.get("activation_allowed"),
            "control_metrics": expert_form.get("control_metrics"),
            "trial_metrics": expert_form.get("trial_metrics"),
            "market_metrics": expert_form.get("market_metrics"),
        },
        "runtime_manifest": {
            "active_feature_count": feature_policy.get("active_feature_count"),
            "schema_feature_count": feature_policy.get("schema_feature_count"),
            "inactive_features_due_to_train_all_missing": feature_policy.get(
                "inactive_features_due_to_train_all_missing"
            )
            or [],
            "all_missing_train_policy": feature_policy.get("all_missing_train_policy")
            or manifest.get("all_missing_train_policy"),
            "feature_policy_source": relpath(paths.get("active_feature_policy_report")),
            "calibration_method": manifest.get("calibration_method"),
            "betting_output": manifest.get("betting_output"),
            "ev_output": manifest.get("ev_output"),
            "production_prediction_write": manifest.get("production_prediction_write"),
            "registry_mutation": manifest.get("registry_mutation"),
            "tgr_enabled": manifest.get("tgr_enabled"),
        },
        "calibration_report": calibration,
        "slice_metrics": slices,
        "blocker_ranking": blockers,
        "no_write_guarantees": NO_WRITE_GUARANTEES,
    }
    ledgers = {
        "joined_runner_evaluation": joined_rows,
        "data_point_provenance_ledger": provenance,
        "missingness_ledger": missingness,
        "feature_family_coverage_utility": family_coverage,
        "odds_market_baseline_ledger": odds_ledger,
        "blocker_ranking": blockers,
    }
    return report, ledgers


def run_packet(
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    output_dir: Path | None = None,
    shadow_run_dir: Path | None = None,
    result_join_path: Path | None = None,
    odds_snapshot_path: Path | None = None,
    aggregate_report_path: Path | None = None,
    promotion_distance_report_path: Path | None = None,
    feature_activation_report_path: Path | None = None,
    expert_form_ablation_report_path: Path | None = None,
    min_meaningful_races: int = DEFAULT_MIN_MEANINGFUL_RACES,
) -> dict[str, Any]:
    output_dir = output_dir or evidence_root / f"prediction_accuracy_system_audit_{now_id()}_report_only"
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_path_states()
    report, ledgers = build_packet(
        evidence_root=evidence_root,
        shadow_run_dir=shadow_run_dir,
        result_join_path=result_join_path,
        odds_snapshot_path=odds_snapshot_path,
        aggregate_report_path=aggregate_report_path,
        promotion_distance_report_path=promotion_distance_report_path,
        feature_activation_report_path=feature_activation_report_path,
        expert_form_ablation_report_path=expert_form_ablation_report_path,
        min_meaningful_races=min_meaningful_races,
    )
    protected_after = protected_path_states()
    report["protected_paths_before"] = protected_before
    report["protected_paths_after"] = protected_after
    report["protected_paths_unchanged"] = protected_before == protected_after
    if not report["protected_paths_unchanged"]:
        report["decision"] = {
            "runtime_status": report["decision"].get("runtime_status"),
            "next_decision": "BLOCKED_PROTECTED_PATH_MUTATION",
            "reason": "protected path hashes changed during report-only packet build",
        }

    write_csv(
        output_dir / "joined_runner_evaluation.csv",
        ledgers["joined_runner_evaluation"],
        [
            "race_id",
            "box",
            "dog_name",
            "predicted_rank",
            "shadow_rf_calibrated_probability",
            "shadow_rf_uncalibrated_probability",
            "is_winner",
            "finish_position",
            "label_join_status",
            "result_identity_match_status",
            "odds_join_status",
            "odds_provenance_status",
            "market_odds_win",
            "market_implied_probability_raw",
            "market_implied_probability_normalized",
            "feature_row_join_status",
            "metadata_is_leakage_safe",
            "weather_present",
            "track_condition_present",
            "race_time_minutes_since_midnight_present",
            "expert_form_sidecar_present",
        ],
    )
    write_csv(
        output_dir / "data_point_provenance_ledger.csv",
        ledgers["data_point_provenance_ledger"],
        ["data_point", "source_path", "rows", "status", "join_key", "notes"],
    )
    write_csv(
        output_dir / "missingness_ledger.csv",
        ledgers["missingness_ledger"],
        ["metric", "present_count", "total_rows", "present_pct", "status", "reason"],
    )
    write_csv(
        output_dir / "feature_family_coverage_utility.csv",
        ledgers["feature_family_coverage_utility"],
        [
            "feature_family",
            "field_count",
            "rows_with_any_present",
            "total_feature_rows",
            "coverage_pct",
            "unique_present_values",
            "default_dominance_pct",
            "inactive_train_all_missing_fields",
            "activation_decision",
            "activation_fail_reasons",
            "present_by_field_json",
            "utility_notes",
        ],
    )
    write_csv(
        output_dir / "odds_market_baseline_ledger.csv",
        ledgers["odds_market_baseline_ledger"],
        [
            "scope",
            "runner_rows",
            "race_count",
            "valid_prejump_odds_rows",
            "complete_valid_odds_races",
            "market_metrics_status",
            "market_top1",
            "market_top3",
            "market_logloss",
            "shadow_top1_same_rows",
            "shadow_top3_same_rows",
            "shadow_logloss_same_rows",
            "notes",
        ],
    )
    write_csv(
        output_dir / "blocker_ranking.csv",
        ledgers["blocker_ranking"],
        ["rank", "blocker", "status", "evidence", "next_action"],
    )
    write_json(output_dir / "calibration_report.json", report["calibration_report"])
    write_json(output_dir / "slice_metrics.json", report["slice_metrics"])
    write_json(output_dir / "prediction_accuracy_system_audit_report.json", report)
    write_text(output_dir / "BOARD_READY_RECOMMENDATION.md", board_ready_recommendation(report))
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["decision"]["next_decision"]) + "\n")
    manifest = output_manifest(output_dir)
    write_json(output_dir / "output_manifest.json", manifest)
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["decision"]["next_decision"],
        "runtime_status": report["decision"]["runtime_status"],
        "prediction_rows": report["counts"]["prediction_rows"],
        "joined_label_rows": report["counts"]["joined_label_rows"],
        "joined_valid_prejump_odds_rows": report["counts"]["joined_valid_prejump_odds_rows"],
        "protected_paths_unchanged": report["protected_paths_unchanged"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shadow-run-dir", type=Path)
    parser.add_argument("--result-join-path", type=Path)
    parser.add_argument("--odds-snapshot-path", type=Path)
    parser.add_argument("--aggregate-report", type=Path)
    parser.add_argument("--promotion-distance-report", type=Path)
    parser.add_argument("--feature-activation-report", type=Path)
    parser.add_argument("--expert-form-ablation-report", type=Path)
    parser.add_argument("--min-meaningful-races", type=int, default=DEFAULT_MIN_MEANINGFUL_RACES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_packet(
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
        shadow_run_dir=args.shadow_run_dir,
        result_join_path=args.result_join_path,
        odds_snapshot_path=args.odds_snapshot_path,
        aggregate_report_path=args.aggregate_report,
        promotion_distance_report_path=args.promotion_distance_report,
        feature_activation_report_path=args.feature_activation_report,
        expert_form_ablation_report_path=args.expert_form_ablation_report,
        min_meaningful_races=args.min_meaningful_races,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
