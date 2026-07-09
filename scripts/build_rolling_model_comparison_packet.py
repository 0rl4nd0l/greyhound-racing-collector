#!/usr/bin/env python3
"""Build report-only rolling model comparisons from unified evidence.

This consumes provenance-safe unified evidence datasets and evaluates candidate
score families on exact official-result races. It writes artifacts only; it
does not train, promote, mutate registries, write DB labels, emit EV, or place
any betting action.
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
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.join_forward_shadow_results import logistic_calibration_review  # noqa: E402
from utils.report_output_dir_guard import assert_prefixed_report_output_dir  # noqa: E402


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "rolling_model_comparison_"
)
OUTPUT_ARTIFACT_PREFIX = "rolling_model_comparison_"
REPORT_FILE = "rolling_model_comparison_report.json"
SUMMARY_FILE = "SUMMARY.md"
CANDIDATE_CSV_FILE = "candidate_metrics.csv"
MARKET_RESIDUAL_CASES_CSV_FILE = "market_residual_cases.csv"
MARKET_RESIDUAL_RUNNER_MATRIX_CSV_FILE = "market_residual_runner_matrix.csv"
MIN_RACES_FOR_REVIEW = 100
DEFAULT_HISTORICAL_UNIFIED_EVIDENCE_REPORT_LIMIT = 500
POWER_GAMMAS = (0.85, 1.2, 1.5, 2.0)
BLEND_MARKET_WEIGHTS = tuple(weight / 100 for weight in range(5, 100, 5))
REFINED_BLEND_MARKET_WEIGHTS = (0.63, 0.66, 0.89, 0.91)
CANDIDATE_DENOMINATOR_MISMATCH_BLOCKER = (
    "candidate_denominator_mismatch_primary_shadow"
)
NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "model_artifact_overwrite": False,
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "betting_or_ev_action": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
    "tgr_enabled": False,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


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
        prefix_error="output_dir_must_be_rolling_model_comparison",
        evidence_root=evidence_root,
    )


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "rolling_model_comparison_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def evidence_repo_root(evidence_root: Path | None) -> Path | None:
    if evidence_root is None:
        return None
    root = evidence_root.resolve()
    if root.name == "full_evidence_orchestration_20260525" and root.parent.name == "artifacts":
        return root.parent.parent
    return root


def resolve_report_relative_path(
    report_path: Path,
    path_value: Any,
    *,
    evidence_root: Path | None = None,
) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    candidates: list[Path] = []
    repo_root = evidence_repo_root(evidence_root)
    if repo_root is not None:
        candidates.append(repo_root / path)
    candidates.append(ROOT / path)
    candidates.append(report_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else report_path.parent / path


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def finite_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def safe_int(value: Any) -> int:
    parsed = finite_int(value)
    return parsed if parsed is not None else 0


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def int_count_mapping(value: Any) -> dict[str, int]:
    return {
        str(reason): safe_int(count)
        for reason, count in sorted(mapping(value).items())
    }


def artifact_odds_rejection_reason_counts(report: Mapping[str, Any]) -> dict[str, int]:
    direct_counts = int_count_mapping(
        report.get("artifact_odds_rejection_reason_counts")
    )
    if direct_counts:
        return direct_counts
    counts = Counter()
    for audit in report.get("artifact_odds_audits") or []:
        if not isinstance(audit, Mapping):
            continue
        counts.update(int_count_mapping(audit.get("rejection_reason_counts")))
    return dict(sorted(counts.items()))


def extend_unique(target: list[str], values: Sequence[str]) -> None:
    seen = set(target)
    for value in values:
        if value in seen:
            continue
        target.append(value)
        seen.add(value)


def gamma_key(prefix: str, gamma: float) -> str:
    return f"{prefix}_power_gamma_{str(gamma).replace('.', '_')}"


def selected_odds_decimal(row: Mapping[str, Any]) -> float | None:
    bucket = selected_odds_bucket(row)
    if bucket is None:
        return None
    odds = finite_float(bucket.get("odds_decimal"))
    if odds is None or odds <= 1.0:
        return None
    return odds


def selected_odds_bucket(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    buckets = row.get("odds_by_capture_bucket")
    if not isinstance(buckets, Mapping):
        return None
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for bucket in buckets.values():
        if not isinstance(bucket, Mapping):
            continue
        odds = finite_float(bucket.get("odds_decimal"))
        if odds is None or odds <= 1.0:
            continue
        candidates.append((str(bucket.get("capture_timestamp") or ""), bucket))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def market_raw_score(row: Mapping[str, Any]) -> float | None:
    odds = selected_odds_decimal(row)
    if odds is None or odds <= 1.0:
        return None
    return 1.0 / odds


def normalize_scores(raw_scores: Sequence[float]) -> list[float] | None:
    if not raw_scores:
        return None
    if any(score < 0 or not math.isfinite(score) for score in raw_scores):
        return None
    total = sum(raw_scores)
    if total <= 0:
        return None
    return [score / total for score in raw_scores]


def base_scores(rows: Sequence[Mapping[str, Any]], key: str) -> list[float] | None:
    scores: list[float] = []
    for row in rows:
        score = finite_float(row.get(key))
        if score is None or score < 0:
            return None
        scores.append(score)
    return normalize_scores(scores)


def market_scores(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    scores: list[float] = []
    for row in rows:
        score = market_raw_score(row)
        if score is None:
            return None
        scores.append(score)
    return normalize_scores(scores)


def power_scores(rows: Sequence[Mapping[str, Any]], key: str, gamma: float) -> list[float] | None:
    raw = base_scores(rows, key)
    if raw is None:
        return None
    return normalize_scores([score**gamma for score in raw])


def blend_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    market_weight: float,
) -> list[float] | None:
    model = base_scores(rows, model_key)
    market = market_scores(rows)
    if model is None or market is None:
        return None
    return [
        ((1.0 - market_weight) * model_score) + (market_weight * market_score)
        for model_score, market_score in zip(model, market, strict=True)
    ]


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "candidate_key": "primary_shadow",
            "family": "baseline",
            "score_function": lambda rows: base_scores(rows, "primary_shadow_probability"),
        },
        {
            "candidate_key": "stage2_shadow",
            "family": "stage2",
            "score_function": lambda rows: base_scores(rows, "stage2_shadow_probability"),
        },
        {
            "candidate_key": "stage2_shadow_uncalibrated",
            "family": "stage2_raw_rf",
            "score_function": lambda rows: base_scores(
                rows,
                "stage2_shadow_uncalibrated_probability",
            ),
        },
        {
            "candidate_key": "market_only_implied",
            "family": "market_only",
            "score_function": market_scores,
        },
    ]
    for gamma in POWER_GAMMAS:
        specs.append(
            {
                "candidate_key": gamma_key("primary_shadow", gamma),
                "family": "calibration_variant",
                "calibration_method": "power_gamma",
                "gamma": gamma,
                "score_function": (
                    lambda rows, gamma=gamma: power_scores(
                        rows,
                        "primary_shadow_probability",
                        gamma,
                    )
                ),
            }
        )
        specs.append(
            {
                "candidate_key": gamma_key("stage2_shadow", gamma),
                "family": "calibration_variant",
                "calibration_method": "power_gamma",
                "gamma": gamma,
                "score_function": (
                    lambda rows, gamma=gamma: power_scores(
                        rows,
                        "stage2_shadow_probability",
                        gamma,
                    )
                ),
            }
        )
        specs.append(
            {
                "candidate_key": gamma_key("stage2_shadow_uncalibrated", gamma),
                "family": "stage2_raw_rf_calibration_variant",
                "calibration_method": "power_gamma",
                "gamma": gamma,
                "score_function": (
                    lambda rows, gamma=gamma: power_scores(
                        rows,
                        "stage2_shadow_uncalibrated_probability",
                        gamma,
                    )
                ),
            }
        )
    for weight in BLEND_MARKET_WEIGHTS:
        specs.append(
            {
                "candidate_key": f"stage2_market_blend_{int(weight * 100)}",
                "family": "odds_augmented_blend",
                "market_weight": weight,
                "score_function": (
                    lambda rows, weight=weight: blend_scores(
                        rows,
                        model_key="stage2_shadow_probability",
                        market_weight=weight,
                    )
                ),
            }
        )
        specs.append(
            {
                "candidate_key": f"stage2_uncalibrated_market_blend_{int(weight * 100)}",
                "family": "stage2_raw_rf_odds_augmented_blend",
                "market_weight": weight,
                "score_function": (
                    lambda rows, weight=weight: blend_scores(
                        rows,
                        model_key="stage2_shadow_uncalibrated_probability",
                        market_weight=weight,
                    )
                ),
            }
        )
    return specs


def report_dataset_path(
    report_path: Path,
    report: Mapping[str, Any],
    *,
    evidence_root: Path | None = None,
) -> Path | None:
    dataset = report.get("dataset_jsonl")
    if dataset:
        return resolve_report_relative_path(
            report_path,
            dataset,
            evidence_root=evidence_root,
        )
    fallback = report_path.parent / "unified_evidence_dataset.jsonl"
    return fallback if fallback.exists() else None


def unified_report_eligible_rows(report: Mapping[str, Any]) -> int:
    return safe_int(report.get("unified_evidence_eligible_rows"))


def is_automatic_unified_evidence_report_path(report_path: Path) -> bool:
    dirname = report_path.parent.name
    if (
        "_manual" in dirname
        or "_probe" in dirname
        or "_validation" in dirname
        or "_odds_only" in dirname
        or "_lock_wait" in dirname
    ):
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
        if report_path.exists() and report_path.resolve() in excluded:
            continue
        report = load_json(report_path)
        if not report or report.get("final_status") != "UNIFIED_EVIDENCE_DATASET_BUILT":
            continue
        if unified_report_eligible_rows(report) <= 0:
            continue
        dataset_path = report_dataset_path(
            report_path,
            report,
            evidence_root=evidence_root,
        )
        if dataset_path is None or not dataset_path.exists():
            continue
        candidates.append(report_path)
    return candidates[-max_reports:] if max_reports > 0 else candidates


def unique_report_paths(paths: Sequence[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        try:
            key = str(path.resolve()) if path.exists() else path.as_posix()
        except OSError:
            key = path.as_posix()
        if key in unique:
            del unique[key]
        unique[key] = path
    return list(unique.values())


def resolve_unified_evidence_report_paths(
    report_paths: Sequence[Path],
    *,
    evidence_root: Path | None = None,
    historical_limit: int = DEFAULT_HISTORICAL_UNIFIED_EVIDENCE_REPORT_LIMIT,
) -> tuple[list[Path], dict[str, Any]]:
    explicit_paths = list(report_paths)
    historical_paths: list[Path] = []
    if evidence_root is not None and evidence_root.exists():
        historical_paths = historical_unified_evidence_report_paths(
            evidence_root,
            exclude_paths=explicit_paths,
            max_reports=historical_limit,
        )
    resolved_paths = unique_report_paths([*historical_paths, *explicit_paths])
    return resolved_paths, {
        "schema_version": "rolling_model_source_discovery_v1",
        "evidence_root": relpath(evidence_root),
        "explicit_report_count": len(explicit_paths),
        "historical_report_limit": historical_limit,
        "historical_report_count": len(historical_paths),
        "effective_report_count": len(resolved_paths),
        "historical_first_report_path": relpath(historical_paths[0])
        if historical_paths
        else None,
        "historical_last_report_path": relpath(historical_paths[-1])
        if historical_paths
        else None,
    }


def collect_race_groups(
    report_paths: Sequence[Path],
    *,
    evidence_root: Path | None = None,
    source_discovery: Mapping[str, Any] | None = None,
    sample_scope: str,
    dedupe_race_id: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    skipped = Counter()
    source_reports: list[dict[str, Any]] = []
    for source_index, report_path in enumerate(report_paths):
        report = load_json(report_path)
        dataset_path = report_dataset_path(
            report_path,
            report,
            evidence_root=evidence_root,
        )
        official_audit = mapping(report.get("official_result_evidence_db_audit"))
        source_reports.append(
            {
                "source_index": source_index,
                "report_path": relpath(report_path),
                "dataset_jsonl": relpath(dataset_path),
                "final_status": report.get("final_status"),
                "row_count": report.get("row_count"),
                "race_count": report.get("race_count"),
                "unified_evidence_eligible_rows": report.get(
                    "unified_evidence_eligible_rows"
                ),
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
                "artifact_odds_rows_seen": report.get("artifact_odds_rows_seen") or 0,
                "artifact_odds_rows_accepted": report.get("artifact_odds_rows_accepted")
                or 0,
                "artifact_odds_rows_rejected": report.get("artifact_odds_rows_rejected")
                or 0,
                "artifact_odds_rejection_reason_counts": (
                    artifact_odds_rejection_reason_counts(report)
                ),
                "exclusion_reason_counts": int_count_mapping(
                    report.get("exclusion_reason_counts")
                ),
                "odds_exclusion_reason_counts": int_count_mapping(
                    report.get("odds_exclusion_reason_counts")
                ),
                "official_result_evidence_db_missing_race_ids": string_list(
                    official_audit.get("missing_race_ids")
                ),
                "official_result_evidence_db_requested_race_ids": string_list(
                    official_audit.get("requested_race_ids")
                ),
                "official_result_evidence_db_requested_race_count": safe_int(
                    official_audit.get("race_ids_requested")
                ),
                "official_result_evidence_db_races_with_rows": string_list(
                    official_audit.get("race_ids_with_rows")
                ),
                "official_result_runner_paths": string_list(
                    report.get("official_result_runner_paths")
                ),
            }
        )
        rows = load_jsonl(dataset_path)
        by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            race_id = str(row.get("race_id") or "").strip()
            if not race_id:
                continue
            if row.get("official_result_available") is not True:
                continue
            if row.get("primary_prediction_available") is not True:
                continue
            by_race[race_id].append(row)
        for race_id, race_rows in by_race.items():
            if sample_scope == "unified" and not all(
                row.get("unified_evidence_eligible") is True for row in race_rows
            ):
                skipped["race_not_fully_unified_evidence_eligible"] += 1
                continue
            winner_count = sum(1 for row in race_rows if row.get("is_winner") is True)
            if winner_count != 1:
                skipped["race_winner_count_not_one"] += 1
                continue
            value = {
                "race_id": race_id,
                "source_index": source_index,
                "source_report": relpath(report_path),
                "runner_rows": race_rows,
            }
            if dedupe_race_id:
                chosen[race_id] = value
            else:
                chosen[f"{source_index}:{race_id}"] = value
    race_groups = sorted(
        chosen.values(),
        key=lambda item: (item.get("source_index", 0), item.get("race_id") or ""),
    )
    return race_groups, {
        "source_reports": source_reports,
        "skipped_race_counts": dict(sorted(skipped.items())),
        "dedupe_race_id": dedupe_race_id,
        "sample_scope": sample_scope,
        "source_discovery": dict(source_discovery or {}),
    }


def evaluate_candidate(
    race_groups: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    score_function: Callable[[Sequence[Mapping[str, Any]]], list[float] | None] = spec[
        "score_function"
    ]
    rank_hits_top1 = 0
    rank_hits_top3 = 0
    winner_ranks: list[int] = []
    brier_values: list[float] = []
    logloss_values: list[float] = []
    probability_sum_errors: list[float] = []
    box1_top_picks = 0
    skipped = Counter()
    evaluated_race_ids: list[str] = []
    calibration_labels: list[int] = []
    calibration_probabilities: list[float] = []
    for race in race_groups:
        rows = list(race.get("runner_rows") or [])
        scores = score_function(rows)
        if scores is None or len(scores) != len(rows):
            skipped["score_missing_or_invalid"] += 1
            continue
        probability_sum_errors.append(abs(sum(scores) - 1.0))
        order = sorted(
            range(len(rows)),
            key=lambda index: (
                -scores[index],
                finite_int(rows[index].get("box_number")) or 999,
                str(rows[index].get("dog_name") or ""),
            ),
        )
        winner_indexes = [
            index for index, row in enumerate(rows) if row.get("is_winner") is True
        ]
        if len(winner_indexes) != 1:
            skipped["race_winner_count_not_one"] += 1
            continue
        winner_index = winner_indexes[0]
        winner_rank = order.index(winner_index) + 1
        winner_ranks.append(winner_rank)
        rank_hits_top1 += int(winner_rank == 1)
        rank_hits_top3 += int(winner_rank <= 3)
        winner_score = max(scores[winner_index], 1e-15)
        logloss_values.append(-math.log(winner_score))
        brier_values.append(
            sum(
                (scores[index] - (1.0 if index == winner_index else 0.0)) ** 2
                for index in range(len(rows))
            )
        )
        top_pick_row = rows[order[0]]
        box1_top_picks += int(finite_int(top_pick_row.get("box_number")) == 1)
        evaluated_race_ids.append(str(race.get("race_id") or ""))
        for index, row in enumerate(rows):
            calibration_labels.append(1 if index == winner_index else 0)
            calibration_probabilities.append(scores[index])
    race_count = len(winner_ranks)
    if race_count <= 0:
        return {
            "candidate_key": spec.get("candidate_key"),
            "family": spec.get("family"),
            "status": "NO_EVALUABLE_RACES",
            "race_count": 0,
            "blockers": ["no_evaluable_races"],
            "skipped_race_counts": dict(sorted(skipped.items())),
        }
    return {
        "candidate_key": spec.get("candidate_key"),
        "family": spec.get("family"),
        "calibration_method": spec.get("calibration_method"),
        "gamma": spec.get("gamma"),
        "market_weight": spec.get("market_weight"),
        "status": "EVALUATED",
        "race_count": race_count,
        "evaluated_race_ids_hash": hashlib.sha256(
            "\n".join(sorted(evaluated_race_ids)).encode("utf-8")
        ).hexdigest(),
        "top1": rank_hits_top1 / race_count,
        "top3": rank_hits_top3 / race_count,
        "mean_winner_rank": sum(winner_ranks) / race_count,
        "winner_ranks": winner_ranks,
        "brier": sum(brier_values) / race_count,
        "logloss": sum(logloss_values) / race_count,
        "box1_top_pick_share": box1_top_picks / race_count,
        "probability_sum_max_error_joined_races": max(probability_sum_errors),
        "calibration_slope_intercept": logistic_calibration_review(
            calibration_labels,
            calibration_probabilities,
        ),
        "skipped_race_counts": dict(sorted(skipped.items())),
        "blockers": [],
    }


def compact_candidate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "candidate_key",
            "family",
            "status",
            "race_count",
            "evaluated_race_ids_hash",
            "baseline_denominator_match",
            "top1",
            "top3",
            "mean_winner_rank",
            "brier",
            "logloss",
            "box1_top_pick_share",
            "calibration_slope_intercept",
            "skipped_race_counts",
            "blockers",
        )
        if key in metrics
    }


def candidate_sort_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        finite_float(metrics.get("top1")) or -1.0,
        finite_float(metrics.get("top3")) or -1.0,
        -(finite_float(metrics.get("mean_winner_rank")) or 999.0),
        -(finite_float(metrics.get("brier")) or 999.0),
        -(finite_float(metrics.get("logloss")) or 999.0),
    )


def is_market_only_candidate(metrics: Mapping[str, Any]) -> bool:
    return (
        str(metrics.get("candidate_key") or "") == "market_only_implied"
        or str(metrics.get("family") or "") == "market_only"
    )


def with_baseline_denominator_guard(
    metrics: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    guarded = dict(metrics)
    if guarded.get("status") != "EVALUATED":
        return guarded
    baseline_count = safe_int(baseline.get("race_count"))
    baseline_hash = str(baseline.get("evaluated_race_ids_hash") or "")
    candidate_count = safe_int(guarded.get("race_count"))
    candidate_hash = str(guarded.get("evaluated_race_ids_hash") or "")
    denominator_match = (
        bool(baseline)
        and baseline.get("status") == "EVALUATED"
        and candidate_count == baseline_count
        and bool(candidate_hash)
        and candidate_hash == baseline_hash
    )
    guarded["baseline_denominator_candidate_key"] = "primary_shadow"
    guarded["baseline_denominator_race_count"] = baseline_count if baseline else None
    guarded["baseline_denominator_race_ids_hash"] = baseline_hash or None
    guarded["baseline_denominator_match"] = denominator_match
    if not denominator_match:
        blockers = list(guarded.get("blockers") or [])
        if CANDIDATE_DENOMINATOR_MISMATCH_BLOCKER not in blockers:
            blockers.append(CANDIDATE_DENOMINATOR_MISMATCH_BLOCKER)
        guarded["blockers"] = blockers
    return guarded


def metric_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    key: str,
) -> float | None:
    left = finite_float(baseline.get(key))
    right = finite_float(candidate.get(key))
    if left is None or right is None:
        return None
    return right - left


def ranking_order(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
) -> list[int]:
    return sorted(
        range(len(rows)),
        key=lambda index: (
            -scores[index],
            finite_int(rows[index].get("box_number")) or 999,
            str(rows[index].get("dog_name") or ""),
        ),
    )


def rank_band(rank: int | None, *, prefix: str) -> str:
    if rank is None or rank <= 0:
        return f"{prefix}_unknown"
    if rank == 1:
        return f"{prefix}_1"
    if rank <= 3:
        return f"{prefix}_2_3"
    return f"{prefix}_4_plus"


def odds_band(odds: float | None, *, prefix: str) -> str:
    if odds is None:
        return f"{prefix}_unknown"
    if odds <= 2.0:
        return f"{prefix}_lte_2"
    if odds <= 4.0:
        return f"{prefix}_2_01_to_4"
    if odds <= 8.0:
        return f"{prefix}_4_01_to_8"
    return f"{prefix}_gt_8"


def race_rows(race: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in (race.get("runner_rows") or []) if isinstance(row, Mapping)]


def race_first_row(race: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = race_rows(race)
    return rows[0] if rows else {}


def race_venue(race: Mapping[str, Any]) -> str:
    venue = str(race_first_row(race).get("venue") or "").strip()
    return venue or "UNKNOWN"


def race_number_band(race: Mapping[str, Any]) -> str:
    number = finite_int(race_first_row(race).get("race_number"))
    if number is None or number <= 0:
        return "race_number_unknown"
    if number <= 4:
        return "race_number_1_4"
    if number <= 8:
        return "race_number_5_8"
    return "race_number_9_plus"


def market_order_and_scores(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[float]] | tuple[None, None]:
    scores = market_scores(rows)
    if scores is None:
        return None, None
    return ranking_order(rows, scores), scores


def selected_order_and_scores(
    rows: Sequence[Mapping[str, Any]],
    selected_spec: Mapping[str, Any],
) -> tuple[list[int], list[float]] | tuple[None, None]:
    score_function: Callable[[Sequence[Mapping[str, Any]]], list[float] | None] = (
        selected_spec["score_function"]
    )
    scores = score_function(rows)
    if scores is None or len(scores) != len(rows):
        return None, None
    return ranking_order(rows, scores), scores


def market_favourite_odds_band(race: Mapping[str, Any]) -> str:
    rows = race_rows(race)
    return odds_band(
        market_favourite_odds_decimal(rows),
        prefix="market_favourite_odds",
    )


def market_favourite_odds_decimal(rows: Sequence[Mapping[str, Any]]) -> float | None:
    market_order, _ = market_order_and_scores(rows)
    if market_order is None or not market_order:
        return None
    return selected_odds_decimal(rows[market_order[0]])


def conditional_market_blend_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    market_weight: float,
    market_favourite_odds_gt: float,
) -> list[float] | None:
    favourite_odds = market_favourite_odds_decimal(rows)
    if favourite_odds is None:
        return None
    if favourite_odds > market_favourite_odds_gt:
        return blend_scores(
            rows,
            model_key=model_key,
            market_weight=market_weight,
        )
    return market_scores(rows)


def selected_top_market_rank_band(
    race: Mapping[str, Any],
    selected_spec: Mapping[str, Any],
) -> str:
    rows = race_rows(race)
    market_order, _ = market_order_and_scores(rows)
    selected_order, _ = selected_order_and_scores(rows, selected_spec)
    if market_order is None or selected_order is None or not selected_order:
        return rank_band(None, prefix="selected_top_market_rank")
    selected_top = selected_order[0]
    return rank_band(
        market_order.index(selected_top) + 1 if selected_top in market_order else None,
        prefix="selected_top_market_rank",
    )


def selected_market_agreement(
    race: Mapping[str, Any],
    selected_spec: Mapping[str, Any],
) -> str:
    rows = race_rows(race)
    market_order, _ = market_order_and_scores(rows)
    selected_order, _ = selected_order_and_scores(rows, selected_spec)
    if (
        market_order is None
        or selected_order is None
        or not market_order
        or not selected_order
    ):
        return "selected_market_agreement_unknown"
    if selected_order[0] == market_order[0]:
        return "selected_top_matches_market_favourite"
    return "selected_top_differs_from_market_favourite"


def slice_metric_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float | None]:
    return {
        key: metric_delta(baseline, candidate, key)
        for key in (
            "top1",
            "top3",
            "mean_winner_rank",
            "brier",
            "logloss",
            "box1_top_pick_share",
        )
    }


def build_edge_diagnostics(
    race_groups: Sequence[Mapping[str, Any]],
    *,
    selected_candidate_key: str | None,
    minimum_slice_races_for_directional_read: int = 10,
) -> dict[str, Any]:
    specs_by_key = {str(spec.get("candidate_key")): spec for spec in candidate_specs()}
    if selected_candidate_key not in specs_by_key:
        selected_candidate_key = "market_only_implied"
    selected_spec = specs_by_key[selected_candidate_key]
    candidate_keys = [
        key
        for key in (
            "primary_shadow",
            "stage2_shadow",
            "stage2_shadow_uncalibrated",
            "market_only_implied",
            selected_candidate_key,
        )
        if key in specs_by_key
    ]
    candidate_keys = list(dict.fromkeys(candidate_keys))

    dimensions: dict[str, Callable[[Mapping[str, Any]], str]] = {
        "venue": race_venue,
        "race_number_band": race_number_band,
        "market_favourite_odds_band": market_favourite_odds_band,
        "selected_top_market_rank_band": (
            lambda race: selected_top_market_rank_band(race, selected_spec)
        ),
        "selected_market_agreement": (
            lambda race: selected_market_agreement(race, selected_spec)
        ),
    }
    diagnostics: dict[str, Any] = {
        "schema_version": "rolling_model_edge_diagnostics_v1",
        "selected_candidate_key": selected_candidate_key,
        "market_candidate_key": "market_only_implied",
        "baseline_candidate_key": "primary_shadow",
        "minimum_slice_races_for_directional_read": (
            minimum_slice_races_for_directional_read
        ),
        "dimensions": {},
    }
    for dimension, classifier in dimensions.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for race in race_groups:
            grouped[classifier(race)].append(race)
        slices: list[dict[str, Any]] = []
        for slice_key, races in sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            metrics_by_key = {
                key: compact_candidate_metrics(
                    evaluate_candidate(races, specs_by_key[key])
                )
                for key in candidate_keys
            }
            selected_metrics = metrics_by_key.get(selected_candidate_key) or {}
            market_metrics = metrics_by_key.get("market_only_implied") or {}
            baseline_metrics = metrics_by_key.get("primary_shadow") or {}
            blockers: list[str] = []
            if len(races) < minimum_slice_races_for_directional_read:
                blockers.append("slice_race_count_below_directional_floor")
            slices.append(
                {
                    "slice_key": slice_key,
                    "race_count": len(races),
                    "runner_rows": sum(len(race_rows(race)) for race in races),
                    "candidate_metrics_by_key": metrics_by_key,
                    "selected_minus_market": slice_metric_delta(
                        market_metrics,
                        selected_metrics,
                    ),
                    "selected_minus_baseline": slice_metric_delta(
                        baseline_metrics,
                        selected_metrics,
                    ),
                    "directional_blockers": blockers,
                }
            )
        diagnostics["dimensions"][dimension] = slices
    return diagnostics


def race_brier(scores: Sequence[float], winner_index: int) -> float:
    return sum(
        (scores[index] - (1.0 if index == winner_index else 0.0)) ** 2
        for index in range(len(scores))
    )


def race_top_pick_summary(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    index: int,
) -> dict[str, Any]:
    row = rows[index]
    return {
        "dog_name": row.get("dog_name"),
        "box_number": finite_int(row.get("box_number")),
        "odds_decimal": selected_odds_decimal(row),
        "score": scores[index],
    }


def build_market_residual_diagnostics(
    race_groups: Sequence[Mapping[str, Any]],
    *,
    selected_candidate_key: str | None,
    max_examples: int = 10,
) -> dict[str, Any]:
    specs_by_key = {str(spec.get("candidate_key")): spec for spec in candidate_specs()}
    market_key = "market_only_implied"
    if selected_candidate_key not in specs_by_key:
        return {
            "schema_version": "rolling_model_market_residual_diagnostics_v1",
            "candidate_key": selected_candidate_key,
            "market_candidate_key": market_key,
            "status": "CANDIDATE_MISSING",
            "blockers": ["candidate_missing"],
        }
    if market_key not in specs_by_key:
        return {
            "schema_version": "rolling_model_market_residual_diagnostics_v1",
            "candidate_key": selected_candidate_key,
            "market_candidate_key": market_key,
            "status": "MARKET_CANDIDATE_MISSING",
            "blockers": ["market_candidate_missing"],
        }

    selected_spec = specs_by_key[selected_candidate_key]
    selected_score_function: Callable[
        [Sequence[Mapping[str, Any]]], list[float] | None
    ] = selected_spec["score_function"]
    skipped = Counter()
    outcome_counts = Counter()
    examples: list[dict[str, Any]] = []
    selected_top1 = 0
    market_top1 = 0
    selected_top3 = 0
    market_top3 = 0
    selected_ranks: list[int] = []
    market_ranks: list[int] = []
    logloss_deltas: list[float] = []
    brier_deltas: list[float] = []

    for race in race_groups:
        rows = race_rows(race)
        selected_scores = selected_score_function(rows)
        market_scores_for_race = market_scores(rows)
        if (
            selected_scores is None
            or market_scores_for_race is None
            or len(selected_scores) != len(rows)
            or len(market_scores_for_race) != len(rows)
        ):
            skipped["score_missing_or_invalid"] += 1
            continue
        winner_indexes = [
            index for index, row in enumerate(rows) if row.get("is_winner") is True
        ]
        if len(winner_indexes) != 1:
            skipped["race_winner_count_not_one"] += 1
            continue

        winner_index = winner_indexes[0]
        selected_order = ranking_order(rows, selected_scores)
        market_order = ranking_order(rows, market_scores_for_race)
        selected_rank = selected_order.index(winner_index) + 1
        market_rank = market_order.index(winner_index) + 1
        selected_correct = selected_rank == 1
        market_correct = market_rank == 1
        selected_top1 += int(selected_correct)
        market_top1 += int(market_correct)
        selected_top3 += int(selected_rank <= 3)
        market_top3 += int(market_rank <= 3)
        selected_ranks.append(selected_rank)
        market_ranks.append(market_rank)

        if selected_correct and not market_correct:
            outcome_counts["candidate_top1_market_miss"] += 1
        elif market_correct and not selected_correct:
            outcome_counts["market_top1_candidate_miss"] += 1
        elif market_correct and selected_correct:
            outcome_counts["both_top1"] += 1
        else:
            outcome_counts["both_miss_top1"] += 1

        selected_logloss = -math.log(max(selected_scores[winner_index], 1e-15))
        market_logloss = -math.log(max(market_scores_for_race[winner_index], 1e-15))
        logloss_delta = selected_logloss - market_logloss
        selected_brier = race_brier(selected_scores, winner_index)
        market_brier = race_brier(market_scores_for_race, winner_index)
        brier_delta = selected_brier - market_brier
        logloss_deltas.append(logloss_delta)
        brier_deltas.append(brier_delta)

        if logloss_delta < 0:
            outcome_counts["candidate_better_logloss"] += 1
        elif logloss_delta > 0:
            outcome_counts["market_better_logloss"] += 1
        else:
            outcome_counts["logloss_tie"] += 1

        first_row = race_first_row(race)
        winner_row = rows[winner_index]
        selected_top = selected_order[0]
        market_top = market_order[0]
        examples.append(
            {
                "race_id": race.get("race_id"),
                "venue": first_row.get("venue"),
                "race_number": finite_int(first_row.get("race_number")),
                "winner": {
                    "dog_name": winner_row.get("dog_name"),
                    "box_number": finite_int(winner_row.get("box_number")),
                    "odds_decimal": selected_odds_decimal(winner_row),
                },
                "candidate_winner_rank": selected_rank,
                "market_winner_rank": market_rank,
                "candidate_top_pick": race_top_pick_summary(
                    rows,
                    selected_scores,
                    selected_top,
                ),
                "market_top_pick": race_top_pick_summary(
                    rows,
                    market_scores_for_race,
                    market_top,
                ),
                "candidate_top_market_rank": (
                    market_order.index(selected_top) + 1
                    if selected_top in market_order
                    else None
                ),
                "candidate_logloss": selected_logloss,
                "market_logloss": market_logloss,
                "candidate_minus_market_logloss": logloss_delta,
                "candidate_brier": selected_brier,
                "market_brier": market_brier,
                "candidate_minus_market_brier": brier_delta,
            }
        )

    race_count = len(selected_ranks)
    if race_count <= 0:
        return {
            "schema_version": "rolling_model_market_residual_diagnostics_v1",
            "candidate_key": selected_candidate_key,
            "market_candidate_key": market_key,
            "status": "NO_EVALUABLE_RACES",
            "blockers": ["no_evaluable_races"],
            "skipped_race_counts": dict(sorted(skipped.items())),
            "race_count": 0,
        }

    strongest_candidate_edges = sorted(
        [item for item in examples if item["candidate_minus_market_logloss"] < 0],
        key=lambda item: item["candidate_minus_market_logloss"],
    )[:max_examples]
    strongest_market_edges = sorted(
        [item for item in examples if item["candidate_minus_market_logloss"] > 0],
        key=lambda item: item["candidate_minus_market_logloss"],
        reverse=True,
    )[:max_examples]

    return {
        "schema_version": "rolling_model_market_residual_diagnostics_v1",
        "candidate_key": selected_candidate_key,
        "market_candidate_key": market_key,
        "status": "EVALUATED",
        "race_count": race_count,
        "skipped_race_counts": dict(sorted(skipped.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "candidate_minus_market": {
            "top1": (selected_top1 / race_count) - (market_top1 / race_count),
            "top3": (selected_top3 / race_count) - (market_top3 / race_count),
            "mean_winner_rank": (
                sum(selected_ranks) / race_count
                - sum(market_ranks) / race_count
            ),
            "logloss": sum(logloss_deltas) / race_count,
            "brier": sum(brier_deltas) / race_count,
        },
        "strongest_candidate_logloss_edges": strongest_candidate_edges,
        "strongest_market_logloss_edges": strongest_market_edges,
        "blockers": [],
    }


def build_market_residual_case_rows(
    race_groups: Sequence[Mapping[str, Any]],
    *,
    selected_candidate_key: str | None,
) -> list[dict[str, Any]]:
    specs_by_key = {str(spec.get("candidate_key")): spec for spec in candidate_specs()}
    market_key = "market_only_implied"
    if selected_candidate_key not in specs_by_key or market_key not in specs_by_key:
        return []
    selected_spec = specs_by_key[selected_candidate_key]
    selected_score_function: Callable[
        [Sequence[Mapping[str, Any]]], list[float] | None
    ] = selected_spec["score_function"]
    cases: list[dict[str, Any]] = []
    for race in race_groups:
        rows = race_rows(race)
        selected_scores = selected_score_function(rows)
        market_scores_for_race = market_scores(rows)
        if (
            selected_scores is None
            or market_scores_for_race is None
            or len(selected_scores) != len(rows)
            or len(market_scores_for_race) != len(rows)
        ):
            continue
        winner_indexes = [
            index for index, row in enumerate(rows) if row.get("is_winner") is True
        ]
        if len(winner_indexes) != 1:
            continue
        winner_index = winner_indexes[0]
        selected_order = ranking_order(rows, selected_scores)
        market_order = ranking_order(rows, market_scores_for_race)
        selected_top_index = selected_order[0]
        market_top_index = market_order[0]
        selected_rank = selected_order.index(winner_index) + 1
        market_rank = market_order.index(winner_index) + 1
        selected_logloss = -math.log(max(selected_scores[winner_index], 1e-15))
        market_logloss = -math.log(max(market_scores_for_race[winner_index], 1e-15))
        selected_brier = race_brier(selected_scores, winner_index)
        market_brier = race_brier(market_scores_for_race, winner_index)
        first_row = race_first_row(race)
        winner_row = rows[winner_index]
        selected_top_row = rows[selected_top_index]
        market_top_row = rows[market_top_index]
        market_favourite_odds = market_favourite_odds_decimal(rows)
        selected_top_market_rank = (
            market_order.index(selected_top_index) + 1
            if selected_top_index in market_order
            else None
        )
        cases.append(
            {
                "candidate_key": selected_candidate_key,
                "market_candidate_key": market_key,
                "race_id": race.get("race_id"),
                "source_report": race.get("source_report"),
                "venue": first_row.get("venue"),
                "race_number": finite_int(first_row.get("race_number")),
                "race_date": first_row.get("race_date"),
                "market_favourite_odds_decimal": market_favourite_odds,
                "market_favourite_odds_band": odds_band(
                    market_favourite_odds,
                    prefix="market_favourite_odds",
                ),
                "selected_top_market_rank": selected_top_market_rank,
                "selected_market_agreement": (
                    "selected_top_matches_market_favourite"
                    if selected_top_index == market_top_index
                    else "selected_top_differs_from_market_favourite"
                ),
                "winner_dog_name": winner_row.get("dog_name"),
                "winner_box_number": finite_int(winner_row.get("box_number")),
                "winner_odds_decimal": selected_odds_decimal(winner_row),
                "candidate_winner_rank": selected_rank,
                "market_winner_rank": market_rank,
                "candidate_winner_score": selected_scores[winner_index],
                "market_winner_score": market_scores_for_race[winner_index],
                "candidate_top_pick_dog_name": selected_top_row.get("dog_name"),
                "candidate_top_pick_box_number": finite_int(
                    selected_top_row.get("box_number")
                ),
                "candidate_top_pick_odds_decimal": selected_odds_decimal(
                    selected_top_row
                ),
                "candidate_top_pick_score": selected_scores[selected_top_index],
                "market_top_pick_dog_name": market_top_row.get("dog_name"),
                "market_top_pick_box_number": finite_int(
                    market_top_row.get("box_number")
                ),
                "market_top_pick_odds_decimal": selected_odds_decimal(market_top_row),
                "market_top_pick_score": market_scores_for_race[market_top_index],
                "candidate_logloss": selected_logloss,
                "market_logloss": market_logloss,
                "candidate_minus_market_logloss": selected_logloss - market_logloss,
                "candidate_brier": selected_brier,
                "market_brier": market_brier,
                "candidate_minus_market_brier": selected_brier - market_brier,
                "candidate_top1_market_miss": (
                    selected_rank == 1 and market_rank != 1
                ),
                "market_top1_candidate_miss": (
                    market_rank == 1 and selected_rank != 1
                ),
                "both_top1": selected_rank == 1 and market_rank == 1,
                "both_miss_top1": selected_rank != 1 and market_rank != 1,
                "candidate_better_logloss": selected_logloss < market_logloss,
                "market_better_logloss": market_logloss < selected_logloss,
            }
        )
    return cases


def optional_score(scores: Sequence[float] | None, index: int) -> float | None:
    if scores is None or index >= len(scores):
        return None
    return scores[index]


def optional_rank(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float] | None,
    index: int,
) -> int | None:
    if scores is None or len(scores) != len(rows):
        return None
    order = ranking_order(rows, scores)
    return order.index(index) + 1 if index in order else None


def build_market_residual_runner_matrix_rows(
    race_groups: Sequence[Mapping[str, Any]],
    *,
    selected_candidate_key: str | None,
) -> list[dict[str, Any]]:
    specs_by_key = {str(spec.get("candidate_key")): spec for spec in candidate_specs()}
    market_key = "market_only_implied"
    if selected_candidate_key not in specs_by_key or market_key not in specs_by_key:
        return []
    selected_spec = specs_by_key[selected_candidate_key]
    selected_score_function: Callable[
        [Sequence[Mapping[str, Any]]], list[float] | None
    ] = selected_spec["score_function"]
    matrix_rows: list[dict[str, Any]] = []
    for race in race_groups:
        rows = race_rows(race)
        selected_scores = selected_score_function(rows)
        market_scores_for_race = market_scores(rows)
        if (
            selected_scores is None
            or market_scores_for_race is None
            or len(selected_scores) != len(rows)
            or len(market_scores_for_race) != len(rows)
        ):
            continue
        primary_scores = base_scores(rows, "primary_shadow_probability")
        stage2_scores = base_scores(rows, "stage2_shadow_probability")
        stage2_uncalibrated_scores = base_scores(
            rows,
            "stage2_shadow_uncalibrated_probability",
        )
        market_order = ranking_order(rows, market_scores_for_race)
        selected_order = ranking_order(rows, selected_scores)
        market_favourite_odds = market_favourite_odds_decimal(rows)
        first_row = race_first_row(race)
        for index, row in enumerate(rows):
            odds_bucket = selected_odds_bucket(row) or {}
            market_probability = market_scores_for_race[index]
            candidate_probability = selected_scores[index]
            matrix_rows.append(
                {
                    "candidate_key": selected_candidate_key,
                    "market_candidate_key": market_key,
                    "race_id": race.get("race_id"),
                    "source_report": race.get("source_report"),
                    "venue": first_row.get("venue"),
                    "race_number": finite_int(first_row.get("race_number")),
                    "race_date": first_row.get("race_date"),
                    "dog_name": row.get("dog_name"),
                    "box_number": finite_int(row.get("box_number")),
                    "is_winner": row.get("is_winner") is True,
                    "finish_position": finite_int(row.get("finish_position")),
                    "odds_decimal": selected_odds_decimal(row),
                    "odds_source_url": odds_bucket.get("source_url"),
                    "odds_capture_timestamp": odds_bucket.get("capture_timestamp"),
                    "odds_capture_mode": odds_bucket.get("capture_mode"),
                    "odds_level": odds_bucket.get("odds_level"),
                    "market_favourite_odds_decimal": market_favourite_odds,
                    "market_favourite_odds_band": odds_band(
                        market_favourite_odds,
                        prefix="market_favourite_odds",
                    ),
                    "market_probability": market_probability,
                    "candidate_probability": candidate_probability,
                    "candidate_minus_market_probability": (
                        candidate_probability - market_probability
                    ),
                    "primary_shadow_probability_norm": optional_score(
                        primary_scores,
                        index,
                    ),
                    "stage2_shadow_probability_norm": optional_score(
                        stage2_scores,
                        index,
                    ),
                    "stage2_shadow_uncalibrated_probability_norm": optional_score(
                        stage2_uncalibrated_scores,
                        index,
                    ),
                    "market_rank": market_order.index(index) + 1,
                    "candidate_rank": selected_order.index(index) + 1,
                    "primary_shadow_rank": optional_rank(rows, primary_scores, index),
                    "stage2_shadow_rank": optional_rank(rows, stage2_scores, index),
                    "stage2_shadow_uncalibrated_rank": optional_rank(
                        rows,
                        stage2_uncalibrated_scores,
                        index,
                    ),
                    "market_top_pick": market_order[0] == index,
                    "candidate_top_pick": selected_order[0] == index,
                    "market_favourite": market_order[0] == index,
                    "candidate_agrees_with_market_top": (
                        selected_order[0] == market_order[0]
                    ),
                    "runner_count": len(rows),
                }
            )
    return matrix_rows


def build_residual_hypothesis_backtests(
    race_groups: Sequence[Mapping[str, Any]],
    *,
    minimum_triggered_races_for_directional_read: int = 10,
) -> list[dict[str, Any]]:
    market_metrics = evaluate_candidate(
        race_groups,
        {
            "candidate_key": "market_only_implied",
            "family": "market_only",
            "score_function": market_scores,
        },
    )
    hypotheses: list[dict[str, Any]] = [
        {
            "candidate_key": (
                "market_anchor_stage2_uncalibrated_blend75_"
                "when_market_favourite_gt4"
            ),
            "family": "exploratory_market_residual_hypothesis",
            "model_key": "stage2_shadow_uncalibrated_probability",
            "market_weight": 0.75,
            "market_favourite_odds_gt": 4.0,
            "selection_basis": (
                "post_hoc_edge_diagnostic_market_favourite_odds_4_01_to_8"
            ),
        }
    ]
    results: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        threshold = float(hypothesis["market_favourite_odds_gt"])
        model_key = str(hypothesis["model_key"])
        market_weight = float(hypothesis["market_weight"])
        spec = {
            "candidate_key": hypothesis["candidate_key"],
            "family": hypothesis["family"],
            "market_weight": market_weight,
            "score_function": (
                lambda rows,
                model_key=model_key,
                market_weight=market_weight,
                threshold=threshold: conditional_market_blend_scores(
                    rows,
                    model_key=model_key,
                    market_weight=market_weight,
                    market_favourite_odds_gt=threshold,
                )
            ),
        }
        metrics = evaluate_candidate(race_groups, spec)
        triggered_race_count = sum(
            1
            for race in race_groups
            if (
                (market_favourite_odds_decimal(race_rows(race)) or -1.0)
                > threshold
            )
        )
        blockers = ["post_hoc_residual_hypothesis_not_promotion_eligible"]
        if triggered_race_count < minimum_triggered_races_for_directional_read:
            blockers.append("triggered_race_count_below_directional_floor")
        results.append(
            {
                "schema_version": "rolling_model_residual_hypothesis_backtest_v1",
                "candidate_key": hypothesis["candidate_key"],
                "family": hypothesis["family"],
                "status": (
                    "EXPLORATORY_EVALUATED"
                    if metrics.get("status") == "EVALUATED"
                    else metrics.get("status")
                ),
                "selection_basis": hypothesis["selection_basis"],
                "promotion_eligible": False,
                "trigger_condition": {
                    "market_favourite_odds_gt": threshold,
                    "model_key": model_key,
                    "market_weight": market_weight,
                },
                "triggered_race_count": triggered_race_count,
                "minimum_triggered_races_for_directional_read": (
                    minimum_triggered_races_for_directional_read
                ),
                "metrics": compact_candidate_metrics(metrics),
                "candidate_minus_market": (
                    slice_metric_delta(market_metrics, metrics)
                    if market_metrics and metrics.get("status") == "EVALUATED"
                    else {}
                ),
                "blockers": blockers,
            }
        )
    return results


def build_refined_blend_frontier_backtests(
    race_groups: Sequence[Mapping[str, Any]],
    *,
    baseline_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for weight in REFINED_BLEND_MARKET_WEIGHTS:
        weight_percent = int(round(weight * 100))
        spec = {
            "candidate_key": f"stage2_market_blend_{weight_percent}",
            "family": "report_only_refined_odds_augmented_blend",
            "market_weight": weight,
            "score_function": (
                lambda rows, weight=weight: blend_scores(
                    rows,
                    model_key="stage2_shadow_probability",
                    market_weight=weight,
                )
            ),
        }
        metrics = with_baseline_denominator_guard(
            evaluate_candidate(race_groups, spec),
            baseline=baseline_metrics,
        )
        blockers = ["report_only_refined_frontier_not_promotion_eligible"]
        if metrics.get("baseline_denominator_match") is False:
            blockers.append(CANDIDATE_DENOMINATOR_MISMATCH_BLOCKER)
        results.append(
            {
                "schema_version": "rolling_model_refined_blend_frontier_backtest_v1",
                "candidate_key": spec["candidate_key"],
                "family": spec["family"],
                "status": (
                    "REPORT_ONLY_EVALUATED"
                    if metrics.get("status") == "EVALUATED"
                    else metrics.get("status")
                ),
                "selection_basis": (
                    "small_refined_frontier_around_prior_stage2_market_blend_candidates"
                ),
                "promotion_eligible": False,
                "market_weight": weight,
                "metrics": compact_candidate_metrics(metrics),
                "blockers": blockers,
            }
        )
    return results


def build_report(
    *,
    generated_at: datetime,
    report_paths: Sequence[Path],
    race_groups: Sequence[Mapping[str, Any]],
    collection: Mapping[str, Any],
    candidate_metrics: Sequence[Mapping[str, Any]],
    min_races_for_review: int,
    output_dir: Path,
) -> dict[str, Any]:
    raw_evaluated = [
        item for item in candidate_metrics if item.get("status") == "EVALUATED"
    ]
    raw_baseline = next(
        (item for item in raw_evaluated if item.get("candidate_key") == "primary_shadow"),
        {},
    )
    guarded_candidate_metrics = [
        with_baseline_denominator_guard(item, baseline=raw_baseline)
        for item in candidate_metrics
    ]
    evaluated = [
        item for item in guarded_candidate_metrics if item.get("status") == "EVALUATED"
    ]
    baseline = next(
        (item for item in evaluated if item.get("candidate_key") == "primary_shadow"),
        {},
    )
    rankable = [
        item for item in evaluated if item.get("baseline_denominator_match") is True
    ]
    denominator_mismatches = [
        item for item in evaluated if item.get("baseline_denominator_match") is False
    ]
    best = max(rankable, key=candidate_sort_key) if rankable else {}
    best_non_baseline = max(
        [item for item in rankable if item.get("candidate_key") != "primary_shadow"],
        key=candidate_sort_key,
        default={},
    )
    market = next(
        (item for item in evaluated if is_market_only_candidate(item)),
        {},
    )
    rankable_market = (
        market if market.get("baseline_denominator_match") is True else {}
    )
    best_non_market = max(
        [item for item in rankable if not is_market_only_candidate(item)],
        key=candidate_sort_key,
        default={},
    )
    sample_race_count = len(race_groups)
    sample_floor_met = sample_race_count >= min_races_for_review
    source_reports = list(collection.get("source_reports") or [])
    rejected_live_odds_candidate_reason_counts = Counter()
    artifact_odds_rejection_reason_counts = Counter()
    source_exclusion_reason_counts = Counter()
    source_odds_exclusion_reason_counts = Counter()
    source_official_result_missing_race_ids: list[str] = []
    source_official_result_requested_race_ids: list[str] = []
    source_official_result_races_with_rows: list[str] = []
    source_official_result_runner_paths: list[str] = []
    source_official_result_requested_race_count_fallback = 0
    for source_report in source_reports:
        if not isinstance(source_report, Mapping):
            continue
        for reason, count in (
            source_report.get("rejected_live_odds_candidate_reason_counts") or {}
        ).items():
            rejected_live_odds_candidate_reason_counts[str(reason)] += safe_int(count)
        for reason, count in (
            source_report.get("artifact_odds_rejection_reason_counts") or {}
        ).items():
            artifact_odds_rejection_reason_counts[str(reason)] += safe_int(count)
        for reason, count in (
            source_report.get("exclusion_reason_counts") or {}
        ).items():
            source_exclusion_reason_counts[str(reason)] += safe_int(count)
        for reason, count in (
            source_report.get("odds_exclusion_reason_counts") or {}
        ).items():
            source_odds_exclusion_reason_counts[str(reason)] += safe_int(count)
        missing_race_ids = string_list(
            source_report.get("official_result_evidence_db_missing_race_ids")
        )
        races_with_rows = string_list(
            source_report.get("official_result_evidence_db_races_with_rows")
        )
        requested_race_ids = string_list(
            source_report.get("official_result_evidence_db_requested_race_ids")
        )
        if not requested_race_ids:
            requested_race_ids = list(dict.fromkeys([*races_with_rows, *missing_race_ids]))
        if requested_race_ids:
            extend_unique(source_official_result_requested_race_ids, requested_race_ids)
        else:
            source_official_result_requested_race_count_fallback += safe_int(
                source_report.get("official_result_evidence_db_requested_race_count")
            )
        extend_unique(
            source_official_result_missing_race_ids,
            missing_race_ids,
        )
        extend_unique(
            source_official_result_races_with_rows,
            races_with_rows,
        )
        extend_unique(
            source_official_result_runner_paths,
            string_list(source_report.get("official_result_runner_paths")),
        )
    source_official_result_requested_race_count = (
        len(source_official_result_requested_race_ids)
        if source_official_result_requested_race_ids
        else source_official_result_requested_race_count_fallback
    )
    official_result_coverage = {
        "source": "unified_evidence_reports",
        "requested_race_count": source_official_result_requested_race_count,
        "requested_race_count_source": (
            "deduped_requested_or_inferred_race_ids"
            if source_official_result_requested_race_ids
            else "legacy_count_fallback"
        ),
        "requested_race_ids": source_official_result_requested_race_ids,
        "legacy_requested_race_count_without_ids": (
            source_official_result_requested_race_count_fallback
        ),
        "races_with_rows_count": len(source_official_result_races_with_rows),
        "missing_race_count": len(source_official_result_missing_race_ids),
        "missing_race_ids": source_official_result_missing_race_ids,
        "races_with_rows": source_official_result_races_with_rows,
        "runner_path_count": len(source_official_result_runner_paths),
        "runner_paths_source_field": "source_official_result_runner_paths",
        "missing_exclusion_count": safe_int(
            source_exclusion_reason_counts.get("official_result_missing")
        ),
    }
    blockers: list[str] = []
    if not sample_floor_met:
        blockers.append("comparison_race_count_below_review_floor")
    if not baseline:
        blockers.append("baseline_primary_shadow_metrics_missing")
    if not best_non_baseline:
        blockers.append("non_baseline_candidate_metrics_missing")
    selected = best_non_baseline or best
    selected_candidate_key = selected.get("candidate_key")
    best_non_market_candidate_key = best_non_market.get("candidate_key")
    edge_candidate_key = (
        str(best_non_market_candidate_key)
        if best_non_market_candidate_key
        else (str(selected_candidate_key) if selected_candidate_key else None)
    )
    return {
        "schema_version": "rolling_model_comparison_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": (
            "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
            if not blockers
            else "ROLLING_MODEL_COMPARISON_COLLECTING"
        ),
        "output_dir": relpath(output_dir),
        "market_residual_cases_csv": relpath(
            output_dir / MARKET_RESIDUAL_CASES_CSV_FILE
        ),
        "market_residual_runner_matrix_csv": relpath(
            output_dir / MARKET_RESIDUAL_RUNNER_MATRIX_CSV_FILE
        ),
        "source_discovery": collection.get("source_discovery") or {},
        "source_unified_evidence_reports": [relpath(path) for path in report_paths],
        "sample_scope": collection.get("sample_scope"),
        "dedupe_race_id": collection.get("dedupe_race_id"),
        "source_reports": source_reports,
        "source_rejected_live_odds_candidate_count": sum(
            int(source_report.get("rejected_live_odds_candidate_count") or 0)
            for source_report in source_reports
            if isinstance(source_report, Mapping)
        ),
        "source_rows_with_rejected_live_odds_candidates": sum(
            int(source_report.get("rows_with_rejected_live_odds_candidates") or 0)
            for source_report in source_reports
            if isinstance(source_report, Mapping)
        ),
        "source_rejected_live_odds_candidate_reason_counts": dict(
            sorted(rejected_live_odds_candidate_reason_counts.items())
        ),
        "source_artifact_odds_rows_seen": sum(
            int(source_report.get("artifact_odds_rows_seen") or 0)
            for source_report in source_reports
            if isinstance(source_report, Mapping)
        ),
        "source_artifact_odds_rows_accepted": sum(
            int(source_report.get("artifact_odds_rows_accepted") or 0)
            for source_report in source_reports
            if isinstance(source_report, Mapping)
        ),
        "source_artifact_odds_rows_rejected": sum(
            int(source_report.get("artifact_odds_rows_rejected") or 0)
            for source_report in source_reports
            if isinstance(source_report, Mapping)
        ),
        "source_artifact_odds_rejection_reason_counts": dict(
            sorted(artifact_odds_rejection_reason_counts.items())
        ),
        "source_exclusion_reason_counts": dict(
            sorted(source_exclusion_reason_counts.items())
        ),
        "source_odds_exclusion_reason_counts": dict(
            sorted(source_odds_exclusion_reason_counts.items())
        ),
        "source_official_result_evidence_db_missing_race_ids": (
            source_official_result_missing_race_ids
        ),
        "source_official_result_evidence_db_requested_race_ids": (
            source_official_result_requested_race_ids
        ),
        "source_official_result_evidence_db_requested_race_count": (
            source_official_result_requested_race_count
        ),
        "source_official_result_evidence_db_legacy_requested_race_count_without_ids": (
            source_official_result_requested_race_count_fallback
        ),
        "source_official_result_evidence_db_races_with_rows": (
            source_official_result_races_with_rows
        ),
        "source_official_result_runner_paths": source_official_result_runner_paths,
        "official_result_coverage": official_result_coverage,
        "skipped_race_counts": collection.get("skipped_race_counts") or {},
        "sample_race_count": sample_race_count,
        "sample_runner_rows": sum(len(race.get("runner_rows") or []) for race in race_groups),
        "candidate_count": len(guarded_candidate_metrics),
        "rankable_candidate_count": len(rankable),
        "candidate_denominator_mismatch_count": len(denominator_mismatches),
        "candidate_denominator_mismatch_keys": [
            item.get("candidate_key") for item in denominator_mismatches
        ],
        "baseline_denominator": {
            "candidate_key": "primary_shadow",
            "race_count": baseline.get("race_count"),
            "evaluated_race_ids_hash": baseline.get("evaluated_race_ids_hash"),
        },
        "minimum_races_for_review": min_races_for_review,
        "sample_floor_met": sample_floor_met,
        "races_needed_for_review": max(0, min_races_for_review - sample_race_count),
        "blockers": blockers,
        "baseline_candidate_key": "primary_shadow",
        "market_candidate_key": "market_only_implied",
        "best_candidate_key": best.get("candidate_key"),
        "best_non_baseline_candidate_key": best_non_baseline.get("candidate_key"),
        "best_non_market_candidate_key": best_non_market_candidate_key,
        "baseline_metrics": dict(baseline),
        "market_metrics": dict(market),
        "candidate_metrics": dict(selected),
        "best_rank_accuracy_candidate_metrics": dict(selected),
        "best_non_market_candidate_metrics": dict(best_non_market),
        "candidate_minus_baseline": {
            key: metric_delta(baseline, selected, key)
            for key in (
                "top1",
                "top3",
                "mean_winner_rank",
                "brier",
                "logloss",
                "box1_top_pick_share",
            )
        },
        "best_non_market_minus_market": (
            slice_metric_delta(rankable_market, best_non_market)
            if rankable_market and best_non_market
            else {}
        ),
        "best_non_market_minus_baseline": (
            slice_metric_delta(baseline, best_non_market)
            if baseline and best_non_market
            else {}
        ),
        "candidate_metrics_by_key": {
            str(item.get("candidate_key")): dict(item)
            for item in guarded_candidate_metrics
        },
        "rank_first_sort": [
            item.get("candidate_key")
            for item in sorted(rankable, key=candidate_sort_key, reverse=True)
        ],
        "edge_diagnostics": build_edge_diagnostics(
            race_groups,
            selected_candidate_key=edge_candidate_key,
        ),
        "market_residual_diagnostics": build_market_residual_diagnostics(
            race_groups,
            selected_candidate_key=edge_candidate_key,
        ),
        "market_residual_case_count": len(
            build_market_residual_case_rows(
                race_groups,
                selected_candidate_key=edge_candidate_key,
            )
        ),
        "market_residual_runner_matrix_row_count": len(
            build_market_residual_runner_matrix_rows(
                race_groups,
                selected_candidate_key=edge_candidate_key,
            )
        ),
        "residual_hypothesis_backtests": build_residual_hypothesis_backtests(
            race_groups,
        ),
        "refined_blend_frontier_backtests": build_refined_blend_frontier_backtests(
            race_groups,
            baseline_metrics=baseline,
        ),
        "ev_metrics_used_for_promotion": False,
        "ev_improved": False,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def write_candidate_csv(path: Path, candidate_metrics: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_key",
        "family",
        "status",
        "race_count",
        "top1",
        "top3",
        "mean_winner_rank",
        "brier",
        "logloss",
        "box1_top_pick_share",
        "probability_sum_max_error_joined_races",
        "evaluated_race_ids_hash",
        "baseline_denominator_match",
        "baseline_denominator_race_count",
        "baseline_denominator_race_ids_hash",
        "calibration_status",
        "calibration_slope",
        "calibration_intercept",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in candidate_metrics:
            calibration = row.get("calibration_slope_intercept")
            if not isinstance(calibration, Mapping):
                calibration = {}
            output = {field: row.get(field) for field in fields}
            output["calibration_status"] = calibration.get("status")
            output["calibration_slope"] = calibration.get("slope")
            output["calibration_intercept"] = calibration.get("intercept")
            writer.writerow(output)


def write_market_residual_cases_csv(
    path: Path,
    case_rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_key",
        "market_candidate_key",
        "race_id",
        "source_report",
        "venue",
        "race_number",
        "race_date",
        "market_favourite_odds_decimal",
        "market_favourite_odds_band",
        "selected_top_market_rank",
        "selected_market_agreement",
        "winner_dog_name",
        "winner_box_number",
        "winner_odds_decimal",
        "candidate_winner_rank",
        "market_winner_rank",
        "candidate_winner_score",
        "market_winner_score",
        "candidate_top_pick_dog_name",
        "candidate_top_pick_box_number",
        "candidate_top_pick_odds_decimal",
        "candidate_top_pick_score",
        "market_top_pick_dog_name",
        "market_top_pick_box_number",
        "market_top_pick_odds_decimal",
        "market_top_pick_score",
        "candidate_logloss",
        "market_logloss",
        "candidate_minus_market_logloss",
        "candidate_brier",
        "market_brier",
        "candidate_minus_market_brier",
        "candidate_top1_market_miss",
        "market_top1_candidate_miss",
        "both_top1",
        "both_miss_top1",
        "candidate_better_logloss",
        "market_better_logloss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in case_rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_market_residual_runner_matrix_csv(
    path: Path,
    matrix_rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_key",
        "market_candidate_key",
        "race_id",
        "source_report",
        "venue",
        "race_number",
        "race_date",
        "dog_name",
        "box_number",
        "is_winner",
        "finish_position",
        "odds_decimal",
        "odds_source_url",
        "odds_capture_timestamp",
        "odds_capture_mode",
        "odds_level",
        "market_favourite_odds_decimal",
        "market_favourite_odds_band",
        "market_probability",
        "candidate_probability",
        "candidate_minus_market_probability",
        "primary_shadow_probability_norm",
        "stage2_shadow_probability_norm",
        "stage2_shadow_uncalibrated_probability_norm",
        "market_rank",
        "candidate_rank",
        "primary_shadow_rank",
        "stage2_shadow_rank",
        "stage2_shadow_uncalibrated_rank",
        "market_top_pick",
        "candidate_top_pick",
        "market_favourite",
        "candidate_agrees_with_market_top",
        "runner_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in matrix_rows:
            writer.writerow({field: row.get(field) for field in fields})


def summary_markdown(report: Mapping[str, Any]) -> str:
    official_result = report.get("official_result_coverage") or {}
    source_discovery = report.get("source_discovery") or {}
    return "\n".join(
        [
            "# Rolling Model Comparison",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Source discovery explicit reports: `{source_discovery.get('explicit_report_count')}`",
            f"- Source discovery historical reports: `{source_discovery.get('historical_report_count')}`",
            f"- Source discovery effective reports: `{source_discovery.get('effective_report_count')}`",
            f"- Sample scope: `{report.get('sample_scope')}`",
            f"- Dedupe race id: `{report.get('dedupe_race_id')}`",
            f"- Sample races: `{report.get('sample_race_count')}` / `{report.get('minimum_races_for_review')}`",
            f"- Sample floor met: `{report.get('sample_floor_met')}`",
            f"- Races needed for review: `{report.get('races_needed_for_review')}`",
            f"- Sample runner rows: `{report.get('sample_runner_rows')}`",
            f"- Source rejected live odds candidates: `{report.get('source_rejected_live_odds_candidate_count')}`",
            f"- Source rows with rejected live odds candidates: `{report.get('source_rows_with_rejected_live_odds_candidates')}`",
            f"- Source rejected live odds candidate reasons: `{report.get('source_rejected_live_odds_candidate_reason_counts')}`",
            f"- Source artifact odds rows seen: `{report.get('source_artifact_odds_rows_seen')}`",
            f"- Source artifact odds rows accepted: `{report.get('source_artifact_odds_rows_accepted')}`",
            f"- Source artifact odds rows rejected: `{report.get('source_artifact_odds_rows_rejected')}`",
            f"- Source artifact odds rejection reasons: `{report.get('source_artifact_odds_rejection_reason_counts')}`",
            f"- Source exclusion reasons: `{report.get('source_exclusion_reason_counts')}`",
            f"- Source odds exclusion reasons: `{report.get('source_odds_exclusion_reason_counts')}`",
            f"- Source official-result missing race IDs: `{report.get('source_official_result_evidence_db_missing_race_ids')}`",
            f"- Official-result coverage requested races: `{official_result.get('requested_race_count')}`",
            f"- Official-result coverage requested race count source: `{official_result.get('requested_race_count_source')}`",
            f"- Official-result legacy requested race count without IDs: `{official_result.get('legacy_requested_race_count_without_ids')}`",
            f"- Official-result coverage races with rows: `{official_result.get('races_with_rows_count')}`",
            f"- Official-result coverage missing races: `{official_result.get('missing_race_count')}`",
            f"- Official-result missing exclusion count: `{official_result.get('missing_exclusion_count')}`",
            f"- Official-result runner path count: `{official_result.get('runner_path_count')}`",
            f"- Official-result runner paths source field: `{official_result.get('runner_paths_source_field')}`",
            f"- Candidate count: `{report.get('candidate_count')}`",
            f"- Rankable candidate count: `{report.get('rankable_candidate_count')}`",
            f"- Candidate denominator mismatch count: `{report.get('candidate_denominator_mismatch_count')}`",
            f"- Candidate denominator mismatch keys: `{report.get('candidate_denominator_mismatch_keys')}`",
            f"- Best candidate: `{report.get('best_candidate_key')}`",
            f"- Best non-baseline candidate: `{report.get('best_non_baseline_candidate_key')}`",
            f"- Best non-market candidate: `{report.get('best_non_market_candidate_key')}`",
            f"- Blockers: `{report.get('blockers')}`",
            f"- Edge diagnostics dimensions: `{sorted((report.get('edge_diagnostics') or {}).get('dimensions', {}).keys())}`",
            f"- Market residual diagnostic candidate: `{(report.get('market_residual_diagnostics') or {}).get('candidate_key')}`",
            f"- Market residual case rows: `{report.get('market_residual_case_count')}`",
            f"- Market residual runner matrix rows: `{report.get('market_residual_runner_matrix_row_count')}`",
            f"- Residual hypothesis backtests: `{len(report.get('residual_hypothesis_backtests') or [])}`",
            f"- Refined blend frontier backtests: `{len(report.get('refined_blend_frontier_backtests') or [])}`",
            "",
            "No training, production promotion, registry mutation, production pointer update, DB write, label write, odds write, betting/EV action, snapshot rewrite, manifest rewrite, or TGR enablement was performed.",
            "",
        ]
    )


def build_comparison(
    *,
    unified_evidence_report_paths: Sequence[Path],
    output_dir: Path,
    evidence_root: Path | None = None,
    sample_scope: str = "unified",
    dedupe_race_id: bool = True,
    min_races_for_review: int = MIN_RACES_FOR_REVIEW,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir, evidence_root=evidence_root))
    output_dir.mkdir(parents=True, exist_ok=False)
    effective_report_paths, source_discovery = resolve_unified_evidence_report_paths(
        unified_evidence_report_paths,
        evidence_root=evidence_root,
    )
    race_groups, collection = collect_race_groups(
        effective_report_paths,
        evidence_root=evidence_root,
        source_discovery=source_discovery,
        sample_scope=sample_scope,
        dedupe_race_id=dedupe_race_id,
    )
    metrics = [evaluate_candidate(race_groups, spec) for spec in candidate_specs()]
    report = build_report(
        generated_at=generated_at,
        report_paths=effective_report_paths,
        race_groups=race_groups,
        collection=collection,
        candidate_metrics=metrics,
        min_races_for_review=min_races_for_review,
        output_dir=output_dir,
    )
    write_json(output_dir / REPORT_FILE, report)
    write_candidate_csv(output_dir / CANDIDATE_CSV_FILE, metrics)
    residual_candidate_key = (
        (report.get("market_residual_diagnostics") or {}).get("candidate_key")
    )
    write_market_residual_cases_csv(
        output_dir / MARKET_RESIDUAL_CASES_CSV_FILE,
        build_market_residual_case_rows(
            race_groups,
            selected_candidate_key=(
                str(residual_candidate_key) if residual_candidate_key else None
            ),
        ),
    )
    write_market_residual_runner_matrix_csv(
        output_dir / MARKET_RESIDUAL_RUNNER_MATRIX_CSV_FILE,
        build_market_residual_runner_matrix_rows(
            race_groups,
            selected_candidate_key=(
                str(residual_candidate_key) if residual_candidate_key else None
            ),
        ),
    )
    write_text(output_dir / SUMMARY_FILE, summary_markdown(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unified-evidence-report",
        action="append",
        type=Path,
        default=[],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--sample-scope",
        choices=("unified", "label"),
        default="unified",
    )
    parser.add_argument("--no-dedupe-race-id", action="store_true")
    parser.add_argument("--min-races-for-review", type=int, default=MIN_RACES_FOR_REVIEW)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir
        or DEFAULT_EVIDENCE_ROOT / f"rolling_model_comparison_{now_id(generated_at)}"
    )
    report = build_comparison(
        unified_evidence_report_paths=args.unified_evidence_report,
        output_dir=output_dir,
        evidence_root=args.evidence_root,
        sample_scope=args.sample_scope,
        dedupe_race_id=not args.no_dedupe_race_id,
        min_races_for_review=args.min_races_for_review,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
