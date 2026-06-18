#!/usr/bin/env python3
"""Build a report-only audit of promotion gate contracts.

This packet compares the current primary-shadow-relative promotion gate with
market-relative and dual-baseline alternatives over an existing rolling model
comparison report. It is deliberately report-only: it never trains, promotes,
mutates registries, writes DB labels, emits EV, or places betting actions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "promotion_gate_contract_audit_"
)
REPORT_FILE = "promotion_gate_contract_audit_report.json"
CANDIDATE_MATRIX_CSV = "candidate_gate_matrix.csv"
POLICY_SUMMARY_CSV = "gate_policy_summary.csv"
SUMMARY_FILE = "SUMMARY.md"
FINAL_STATUS_FILE = "final_status.txt"
OUTPUT_MANIFEST_FILE = "output_manifest.json"
ROLLING_READY_STATUS = "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
MARKET_ONLY_CANDIDATE_KEYS = {"market_only_implied"}
BASELINE_CANDIDATE_KEYS = {"primary_shadow", "champion_baseline"}


@dataclass(frozen=True)
class GateAuditThresholds:
    min_safe_joined_races: int = 100
    current_min_top1_delta: float = 0.02
    min_market_top1_delta: float = 0.0
    min_top3_delta: float = 0.0
    max_mean_winner_rank_delta: float = 0.0
    max_brier_delta: float = 0.0
    max_logloss_delta: float = 0.0
    max_calibration_distance_delta: float = 0.0
    max_box1_top_pick_share: float = 0.35
    max_box1_share_delta: float = 0.0
    max_probability_sum_error: float = 1e-6


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
    "ev_or_betting_action": False,
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


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
        "schema_version": "promotion_gate_contract_audit_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_promotion_gate_contract_audit:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def finite_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return 0


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def metric(metrics: Mapping[str, Any], key: str) -> float | None:
    if key == "logloss":
        return finite_float(metrics.get("logloss", metrics.get("log_loss")))
    if key == "box1_top_pick_share":
        direct = finite_float(metrics.get("box1_top_pick_share"))
        if direct is not None:
            return direct
        return finite_float(mapping(metrics.get("box_bias")).get("box1_top_pick_share"))
    return finite_float(metrics.get(key))


def race_count(metrics: Mapping[str, Any]) -> int:
    return finite_int(
        metrics.get("safe_joined_race_count")
        or metrics.get("race_count")
        or metrics.get("holdout_races")
        or metrics.get("source_safe_exact_joined_race_count")
    )


def probability_sum_error(metrics: Mapping[str, Any]) -> float | None:
    direct = finite_float(metrics.get("probability_sum_max_error_joined_races"))
    if direct is not None:
        return direct
    report = metrics.get("probability_sum_error")
    if isinstance(report, Mapping):
        return finite_float(report.get("max_abs_error") or report.get("max_error"))
    return None


def calibration_distance(metrics: Mapping[str, Any]) -> float | None:
    calibration = mapping(
        metrics.get("calibration_slope_intercept") or metrics.get("slope_intercept")
    )
    slope = finite_float(calibration.get("slope"))
    intercept = finite_float(calibration.get("intercept"))
    if slope is None or intercept is None:
        return None
    return abs(slope - 1.0) + abs(intercept)


def delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    key: str,
) -> float | None:
    baseline_value = metric(baseline, key)
    candidate_value = metric(candidate, key)
    if baseline_value is None or candidate_value is None:
        return None
    return candidate_value - baseline_value


def calibration_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> float | None:
    baseline_value = calibration_distance(baseline)
    candidate_value = calibration_distance(candidate)
    if baseline_value is None or candidate_value is None:
        return None
    return candidate_value - baseline_value


def is_market_only_candidate(candidate_key: Any) -> bool:
    return str(candidate_key or "") in MARKET_ONLY_CANDIDATE_KEYS


def is_baseline_candidate(candidate_key: Any) -> bool:
    return str(candidate_key or "") in BASELINE_CANDIDATE_KEYS


def candidate_sort_tuple(
    row: Mapping[str, Any],
    rank_lookup: Mapping[str, int],
) -> tuple[Any, ...]:
    candidate_key = str(row.get("candidate_key") or "")
    rank = rank_lookup.get(candidate_key, 10_000)
    top1 = metric(row, "top1")
    top3 = metric(row, "top3")
    mean_rank = metric(row, "mean_winner_rank")
    logloss = metric(row, "logloss")
    brier = metric(row, "brier")
    return (
        rank,
        -(top1 if top1 is not None else -1.0),
        -(top3 if top3 is not None else -1.0),
        mean_rank if mean_rank is not None else 10_000.0,
        logloss if logloss is not None else 10_000.0,
        brier if brier is not None else 10_000.0,
        candidate_key,
    )


def add_common_candidate_blockers(
    *,
    blockers: list[str],
    candidate: Mapping[str, Any],
    rolling_report: Mapping[str, Any],
    thresholds: GateAuditThresholds,
) -> None:
    candidate_key = candidate.get("candidate_key")
    if rolling_report.get("final_status") != ROLLING_READY_STATUS:
        blockers.append(f"source_status_not_ready:{rolling_report.get('final_status')}")
    if str(rolling_report.get("sample_scope") or "") != "unified":
        blockers.append(f"sample_scope_not_unified:{rolling_report.get('sample_scope')}")
    if rolling_report.get("sample_floor_met") is not True:
        blockers.append("sample_floor_not_met")
    if candidate.get("status") != "EVALUATED":
        blockers.append(f"candidate_status_not_evaluated:{candidate.get('status')}")
    if is_market_only_candidate(candidate_key):
        blockers.append("market_only_candidate_not_promotable")
    if is_baseline_candidate(candidate_key):
        blockers.append("baseline_candidate_not_promotable")
    if race_count(candidate) < thresholds.min_safe_joined_races:
        blockers.append("candidate_race_sample_below_min")
    sum_error = probability_sum_error(candidate)
    if sum_error is not None and sum_error > thresholds.max_probability_sum_error:
        blockers.append("candidate_probability_sum_error_failed")


def require_delta(
    blockers: list[str],
    *,
    value: float | None,
    metric_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    if value is None:
        blockers.append(f"metric_missing:{metric_name}")
        return
    if min_value is not None and value < min_value:
        blockers.append(f"metric_delta_below_min:{metric_name}")
    if max_value is not None and value > max_value:
        blockers.append(f"metric_regressed:{metric_name}")


def gate_current_primary_relative(
    *,
    candidate: Mapping[str, Any],
    primary: Mapping[str, Any],
    rolling_report: Mapping[str, Any],
    thresholds: GateAuditThresholds,
) -> list[str]:
    blockers: list[str] = []
    add_common_candidate_blockers(
        blockers=blockers,
        candidate=candidate,
        rolling_report=rolling_report,
        thresholds=thresholds,
    )
    require_delta(
        blockers,
        value=delta(primary, candidate, "top1"),
        metric_name="top1",
        min_value=thresholds.current_min_top1_delta,
    )
    require_delta(
        blockers,
        value=delta(primary, candidate, "top3"),
        metric_name="top3",
        min_value=thresholds.min_top3_delta,
    )
    require_delta(
        blockers,
        value=delta(primary, candidate, "mean_winner_rank"),
        metric_name="mean_winner_rank",
        max_value=thresholds.max_mean_winner_rank_delta,
    )
    require_delta(
        blockers,
        value=delta(primary, candidate, "brier"),
        metric_name="brier",
        max_value=thresholds.max_brier_delta,
    )
    require_delta(
        blockers,
        value=delta(primary, candidate, "logloss"),
        metric_name="logloss",
        max_value=thresholds.max_logloss_delta,
    )
    require_delta(
        blockers,
        value=calibration_delta(primary, candidate),
        metric_name="calibration_slope_intercept",
        max_value=thresholds.max_calibration_distance_delta,
    )
    candidate_box1 = metric(candidate, "box1_top_pick_share")
    if candidate_box1 is None:
        blockers.append("metric_missing:box1_top_pick_share_candidate")
    elif candidate_box1 > thresholds.max_box1_top_pick_share:
        blockers.append("candidate_box1_top_pick_share_above_max")
    require_delta(
        blockers,
        value=delta(primary, candidate, "box1_top_pick_share"),
        metric_name="box1_top_pick_share",
        max_value=thresholds.max_box1_share_delta,
    )
    return list(dict.fromkeys(blockers))


def gate_market_relative(
    *,
    candidate: Mapping[str, Any],
    market: Mapping[str, Any],
    rolling_report: Mapping[str, Any],
    thresholds: GateAuditThresholds,
    require_box_not_above_market: bool,
    require_calibration_not_worse_than_market: bool,
) -> list[str]:
    blockers: list[str] = []
    add_common_candidate_blockers(
        blockers=blockers,
        candidate=candidate,
        rolling_report=rolling_report,
        thresholds=thresholds,
    )
    require_delta(
        blockers,
        value=delta(market, candidate, "top1"),
        metric_name="top1",
        min_value=thresholds.min_market_top1_delta,
    )
    require_delta(
        blockers,
        value=delta(market, candidate, "top3"),
        metric_name="top3",
        min_value=thresholds.min_top3_delta,
    )
    require_delta(
        blockers,
        value=delta(market, candidate, "mean_winner_rank"),
        metric_name="mean_winner_rank",
        max_value=thresholds.max_mean_winner_rank_delta,
    )
    require_delta(
        blockers,
        value=delta(market, candidate, "brier"),
        metric_name="brier",
        max_value=thresholds.max_brier_delta,
    )
    require_delta(
        blockers,
        value=delta(market, candidate, "logloss"),
        metric_name="logloss",
        max_value=thresholds.max_logloss_delta,
    )
    candidate_box1 = metric(candidate, "box1_top_pick_share")
    if candidate_box1 is None:
        blockers.append("metric_missing:box1_top_pick_share_candidate")
    elif candidate_box1 > thresholds.max_box1_top_pick_share:
        blockers.append("candidate_box1_top_pick_share_above_max")
    if require_box_not_above_market:
        require_delta(
            blockers,
            value=delta(market, candidate, "box1_top_pick_share"),
            metric_name="box1_top_pick_share",
            max_value=thresholds.max_box1_share_delta,
        )
    if require_calibration_not_worse_than_market:
        require_delta(
            blockers,
            value=calibration_delta(market, candidate),
            metric_name="calibration_slope_intercept",
            max_value=thresholds.max_calibration_distance_delta,
        )
    return list(dict.fromkeys(blockers))


def gate_dual_baseline(
    *,
    candidate: Mapping[str, Any],
    market: Mapping[str, Any],
    primary: Mapping[str, Any],
    rolling_report: Mapping[str, Any],
    thresholds: GateAuditThresholds,
) -> list[str]:
    blockers = gate_market_relative(
        candidate=candidate,
        market=market,
        rolling_report=rolling_report,
        thresholds=thresholds,
        require_box_not_above_market=False,
        require_calibration_not_worse_than_market=False,
    )
    candidate_calibration = calibration_distance(candidate)
    market_calibration = calibration_distance(market)
    primary_calibration = calibration_distance(primary)
    if (
        candidate_calibration is None
        or market_calibration is None
        or primary_calibration is None
    ):
        blockers.append("metric_missing:calibration_slope_intercept")
    elif candidate_calibration > market_calibration and candidate_calibration > primary_calibration:
        blockers.append("calibration_worse_than_market_and_primary")
    return list(dict.fromkeys(blockers))


POLICIES = (
    {
        "policy_key": "current_primary_relative",
        "baseline": "primary_shadow",
        "description": (
            "Current gate shape: candidate must beat primary shadow on rank, "
            "loss, calibration distance, and box-one share delta."
        ),
    },
    {
        "policy_key": "market_relative_rank_safe_box_cap_only",
        "baseline": "market_only_implied",
        "description": (
            "Market-relative rank and loss non-regression with hard box-one cap; "
            "calibration and box-one-vs-market are review fields."
        ),
    },
    {
        "policy_key": "market_relative_rank_safe_box_not_above_market",
        "baseline": "market_only_implied",
        "description": (
            "Market-relative rank and loss non-regression, hard box-one cap, "
            "and box-one share not above market."
        ),
    },
    {
        "policy_key": "market_relative_strict_calibration_box_not_above_market",
        "baseline": "market_only_implied",
        "description": (
            "Strict market-relative policy: rank, loss, box-one share, and "
            "calibration distance must not regress versus market."
        ),
    },
    {
        "policy_key": "dual_baseline_market_rank_primary_safety",
        "baseline": "market_only_implied+primary_shadow",
        "description": (
            "Dual-baseline policy: rank and loss must not regress versus market; "
            "box-one uses a hard cap; calibration is blocked only if worse than "
            "both market and primary shadow."
        ),
    },
)


def resolve_baselines(
    rolling_report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    by_key_raw = mapping(rolling_report.get("candidate_metrics_by_key"))
    by_key: dict[str, Mapping[str, Any]] = {
        str(key): mapping(value)
        for key, value in by_key_raw.items()
        if isinstance(value, Mapping)
    }
    market = mapping(rolling_report.get("market_metrics")) or mapping(
        by_key.get("market_only_implied")
    )
    primary = mapping(rolling_report.get("baseline_metrics")) or mapping(
        by_key.get("primary_shadow")
    )
    return primary, market, by_key


def evaluate_policy(
    policy_key: str,
    *,
    candidate: Mapping[str, Any],
    primary: Mapping[str, Any],
    market: Mapping[str, Any],
    rolling_report: Mapping[str, Any],
    thresholds: GateAuditThresholds,
) -> list[str]:
    if policy_key == "current_primary_relative":
        return gate_current_primary_relative(
            candidate=candidate,
            primary=primary,
            rolling_report=rolling_report,
            thresholds=thresholds,
        )
    if policy_key == "market_relative_rank_safe_box_cap_only":
        return gate_market_relative(
            candidate=candidate,
            market=market,
            rolling_report=rolling_report,
            thresholds=thresholds,
            require_box_not_above_market=False,
            require_calibration_not_worse_than_market=False,
        )
    if policy_key == "market_relative_rank_safe_box_not_above_market":
        return gate_market_relative(
            candidate=candidate,
            market=market,
            rolling_report=rolling_report,
            thresholds=thresholds,
            require_box_not_above_market=True,
            require_calibration_not_worse_than_market=False,
        )
    if policy_key == "market_relative_strict_calibration_box_not_above_market":
        return gate_market_relative(
            candidate=candidate,
            market=market,
            rolling_report=rolling_report,
            thresholds=thresholds,
            require_box_not_above_market=True,
            require_calibration_not_worse_than_market=True,
        )
    if policy_key == "dual_baseline_market_rank_primary_safety":
        return gate_dual_baseline(
            candidate=candidate,
            market=market,
            primary=primary,
            rolling_report=rolling_report,
            thresholds=thresholds,
        )
    raise ValueError(f"unknown_policy:{policy_key}")


def candidate_gate_row(
    candidate: Mapping[str, Any],
    *,
    primary: Mapping[str, Any],
    market: Mapping[str, Any],
    rolling_report: Mapping[str, Any],
    rank_lookup: Mapping[str, int],
    thresholds: GateAuditThresholds,
) -> dict[str, Any]:
    candidate_key = str(candidate.get("candidate_key") or "")
    row: dict[str, Any] = {
        "candidate_key": candidate_key,
        "family": candidate.get("family"),
        "status": candidate.get("status"),
        "rank_first_order": rank_lookup.get(candidate_key),
        "race_count": race_count(candidate),
        "top1": metric(candidate, "top1"),
        "top3": metric(candidate, "top3"),
        "mean_winner_rank": metric(candidate, "mean_winner_rank"),
        "brier": metric(candidate, "brier"),
        "logloss": metric(candidate, "logloss"),
        "box1_top_pick_share": metric(candidate, "box1_top_pick_share"),
        "calibration_distance": calibration_distance(candidate),
        "probability_sum_max_error_joined_races": probability_sum_error(candidate),
        "market_top1_delta": delta(market, candidate, "top1"),
        "market_top3_delta": delta(market, candidate, "top3"),
        "market_mean_winner_rank_delta": delta(market, candidate, "mean_winner_rank"),
        "market_brier_delta": delta(market, candidate, "brier"),
        "market_logloss_delta": delta(market, candidate, "logloss"),
        "market_box1_top_pick_share_delta": delta(
            market, candidate, "box1_top_pick_share"
        ),
        "market_calibration_distance_delta": calibration_delta(market, candidate),
        "primary_top1_delta": delta(primary, candidate, "top1"),
        "primary_top3_delta": delta(primary, candidate, "top3"),
        "primary_mean_winner_rank_delta": delta(
            primary, candidate, "mean_winner_rank"
        ),
        "primary_brier_delta": delta(primary, candidate, "brier"),
        "primary_logloss_delta": delta(primary, candidate, "logloss"),
        "primary_box1_top_pick_share_delta": delta(
            primary, candidate, "box1_top_pick_share"
        ),
        "primary_calibration_distance_delta": calibration_delta(primary, candidate),
    }
    for policy in POLICIES:
        policy_key = str(policy["policy_key"])
        blockers = evaluate_policy(
            policy_key,
            candidate=candidate,
            primary=primary,
            market=market,
            rolling_report=rolling_report,
            thresholds=thresholds,
        )
        row[f"{policy_key}_status"] = "PASS" if not blockers else "BLOCKED"
        row[f"{policy_key}_blockers"] = ";".join(blockers)
    return row


def policy_summary_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    rank_lookup: Mapping[str, int],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for policy in POLICIES:
        policy_key = str(policy["policy_key"])
        passing = [
            row
            for row in candidate_rows
            if row.get(f"{policy_key}_status") == "PASS"
        ]
        selected = (
            sorted(passing, key=lambda row: candidate_sort_tuple(row, rank_lookup))[0]
            if passing
            else {}
        )
        summaries.append(
            {
                "policy_key": policy_key,
                "baseline": policy["baseline"],
                "description": policy["description"],
                "status": "PASS" if passing else "BLOCKED",
                "passing_candidate_count": len(passing),
                "selected_candidate": selected.get("candidate_key"),
                "selected_rank_first_order": selected.get("rank_first_order"),
                "selected_top1": selected.get("top1"),
                "selected_top3": selected.get("top3"),
                "selected_mean_winner_rank": selected.get("mean_winner_rank"),
                "selected_logloss": selected.get("logloss"),
                "selected_brier": selected.get("brier"),
                "selected_box1_top_pick_share": selected.get("box1_top_pick_share"),
                "selected_market_top1_delta": selected.get("market_top1_delta"),
                "selected_market_top3_delta": selected.get("market_top3_delta"),
                "selected_market_mean_winner_rank_delta": selected.get(
                    "market_mean_winner_rank_delta"
                ),
                "selected_market_logloss_delta": selected.get("market_logloss_delta"),
                "selected_market_brier_delta": selected.get("market_brier_delta"),
                "selected_market_box1_top_pick_share_delta": selected.get(
                    "market_box1_top_pick_share_delta"
                ),
                "selected_market_calibration_distance_delta": selected.get(
                    "market_calibration_distance_delta"
                ),
            }
        )
    return summaries


def final_status_from_policy_summaries(
    summaries: Sequence[Mapping[str, Any]],
    rolling_report: Mapping[str, Any],
) -> str:
    if rolling_report.get("final_status") != ROLLING_READY_STATUS:
        return "DATA_MISSING"
    by_key = {str(row.get("policy_key")): row for row in summaries}
    if not any(row.get("status") == "PASS" for row in summaries):
        return "KEEP_BASELINE_GATE_VALID"
    strict = by_key.get("market_relative_strict_calibration_box_not_above_market", {})
    dual = by_key.get("dual_baseline_market_rank_primary_safety", {})
    if strict.get("status") == "PASS" and dual.get("status") == "PASS":
        return "REPORT_ONLY_GATE_CHANGE_CANDIDATE"
    return "GATE_POLICY_REVIEW_REQUIRED"


def build_report(
    *,
    rolling_report: Mapping[str, Any],
    rolling_report_path: Path | None = None,
    high_accuracy_report: Mapping[str, Any] | None = None,
    high_accuracy_report_path: Path | None = None,
    output_dir: Path | None = None,
    thresholds: GateAuditThresholds = GateAuditThresholds(),
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    primary, market, by_key = resolve_baselines(rolling_report)
    rank_first_sort = [str(item) for item in rolling_report.get("rank_first_sort") or []]
    rank_lookup = {candidate_key: index + 1 for index, candidate_key in enumerate(rank_first_sort)}
    blockers: list[str] = []
    if not primary:
        blockers.append("primary_shadow_baseline_missing")
    if not market:
        blockers.append("market_only_baseline_missing")
    if not by_key:
        blockers.append("candidate_metrics_by_key_missing")
    candidate_rows: list[dict[str, Any]] = []
    if not blockers:
        for candidate_key, candidate in sorted(
            by_key.items(),
            key=lambda item: candidate_sort_tuple(item[1], rank_lookup),
        ):
            candidate_with_key = (
                candidate
                if candidate.get("candidate_key")
                else {**dict(candidate), "candidate_key": candidate_key}
            )
            candidate_rows.append(
                candidate_gate_row(
                    candidate_with_key,
                    primary=primary,
                    market=market,
                    rolling_report=rolling_report,
                    rank_lookup=rank_lookup,
                    thresholds=thresholds,
                )
            )
    summaries = policy_summary_rows(candidate_rows, rank_lookup=rank_lookup)
    final_status = (
        "DATA_MISSING"
        if blockers
        else final_status_from_policy_summaries(summaries, rolling_report)
    )
    high_accuracy = mapping(high_accuracy_report)
    return {
        "schema_version": "promotion_gate_contract_audit_packet_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "blockers": blockers,
        "rolling_report_summary": {
            "path": relpath(rolling_report_path),
            "final_status": rolling_report.get("final_status"),
            "sample_scope": rolling_report.get("sample_scope"),
            "sample_floor_met": rolling_report.get("sample_floor_met"),
            "sample_race_count": rolling_report.get("sample_race_count"),
            "sample_runner_rows": rolling_report.get("sample_runner_rows"),
            "candidate_count": rolling_report.get("candidate_count"),
            "best_candidate_key": rolling_report.get("best_candidate_key"),
            "best_non_market_candidate_key": rolling_report.get(
                "best_non_market_candidate_key"
            ),
            "rank_first_sort_top10": rank_first_sort[:10],
        },
        "high_accuracy_gate_summary": {
            "path": relpath(high_accuracy_report_path),
            "final_status": high_accuracy.get("final_status"),
            "promotion_pr_gate_status": mapping(
                high_accuracy.get("promotion_pr_gate")
            ).get("status"),
            "promotion_pr_gate_blockers": mapping(
                high_accuracy.get("promotion_pr_gate")
            ).get("blockers"),
            "selected_candidate": mapping(high_accuracy.get("promotion_pr_gate")).get(
                "selected_candidate"
            ),
        },
        "baseline_metrics": {
            "primary_shadow": compact_metrics(primary),
            "market_only_implied": compact_metrics(market),
        },
        "thresholds": asdict(thresholds),
        "gate_policies": list(POLICIES),
        "policy_summaries": summaries,
        "candidate_gate_matrix": candidate_rows,
        "recommended_next_action": recommended_next_action(final_status),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "output_dir": relpath(output_dir),
    }


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_key": metrics.get("candidate_key"),
        "family": metrics.get("family"),
        "status": metrics.get("status"),
        "race_count": race_count(metrics),
        "top1": metric(metrics, "top1"),
        "top3": metric(metrics, "top3"),
        "mean_winner_rank": metric(metrics, "mean_winner_rank"),
        "brier": metric(metrics, "brier"),
        "logloss": metric(metrics, "logloss"),
        "box1_top_pick_share": metric(metrics, "box1_top_pick_share"),
        "calibration_distance": calibration_distance(metrics),
        "probability_sum_max_error_joined_races": probability_sum_error(metrics),
    }


def recommended_next_action(final_status: str) -> str:
    if final_status == "REPORT_ONLY_GATE_CHANGE_CANDIDATE":
        return (
            "Open a separate reviewed implementation to make high-accuracy "
            "selection evaluate predeclared market-relative/dual-baseline "
            "passing candidates instead of only the rank-first candidate."
        )
    if final_status == "GATE_POLICY_REVIEW_REQUIRED":
        return (
            "Keep baseline active; review the relaxed market-relative pass set "
            "before changing any gate."
        )
    if final_status == "KEEP_BASELINE_GATE_VALID":
        return "Keep baseline active; no audited policy found a passing candidate."
    return "Collect or repair missing rolling comparison evidence before gate changes."


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["status"]
        rows = [{"status": "DATA_MISSING"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(report: Mapping[str, Any]) -> str:
    rolling = mapping(report.get("rolling_report_summary"))
    high_accuracy = mapping(report.get("high_accuracy_gate_summary"))
    policy_summaries = report.get("policy_summaries") or []
    lines = [
        "# Promotion Gate Contract Audit",
        "",
        f"- Final status: `{report.get('final_status')}`",
        f"- Rolling report status: `{rolling.get('final_status')}`",
        f"- Sample scope: `{rolling.get('sample_scope')}`",
        f"- Sample races: `{rolling.get('sample_race_count')}`",
        f"- Candidate count: `{rolling.get('candidate_count')}`",
        f"- Rolling best candidate: `{rolling.get('best_candidate_key')}`",
        f"- Rolling best non-market candidate: `{rolling.get('best_non_market_candidate_key')}`",
        f"- Current high-accuracy PR gate: `{high_accuracy.get('promotion_pr_gate_status')}`",
        f"- Current high-accuracy PR blockers: `{high_accuracy.get('promotion_pr_gate_blockers')}`",
        f"- Recommended next action: `{report.get('recommended_next_action')}`",
        "",
        "## Gate Policies",
        "",
    ]
    for row in policy_summaries:
        lines.extend(
            [
                f"- `{row.get('policy_key')}`: `{row.get('status')}`",
                f"  - Passing candidates: `{row.get('passing_candidate_count')}`",
                f"  - Selected candidate: `{row.get('selected_candidate')}`",
                f"  - Selected rank-first order: `{row.get('selected_rank_first_order')}`",
            ]
        )
    lines.extend(
        [
            "",
            "No training, registry, DB, label, snapshot, EV action, betting action, TGR, or production pointer write was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_packet(output_dir: Path, report: Mapping[str, Any]) -> None:
    write_csv(output_dir / CANDIDATE_MATRIX_CSV, report.get("candidate_gate_matrix") or [])
    write_csv(output_dir / POLICY_SUMMARY_CSV, report.get("policy_summaries") or [])
    write_json(output_dir / REPORT_FILE, report)
    write_text(output_dir / SUMMARY_FILE, build_summary(report))
    write_text(output_dir / FINAL_STATUS_FILE, str(report["final_status"]) + "\n")
    write_json(output_dir / OUTPUT_MANIFEST_FILE, output_manifest(output_dir))


def run_packet(
    *,
    rolling_report_path: Path,
    high_accuracy_report_path: Path | None = None,
    output_dir: Path | None = None,
    thresholds: GateAuditThresholds = GateAuditThresholds(),
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or (
        DEFAULT_EVIDENCE_ROOT
        / f"promotion_gate_contract_audit_{now_id(generated_at)}_report_only"
    )
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_report(
        rolling_report=load_json(rolling_report_path),
        rolling_report_path=rolling_report_path,
        high_accuracy_report=(
            load_json(high_accuracy_report_path) if high_accuracy_report_path else None
        ),
        high_accuracy_report_path=high_accuracy_report_path,
        output_dir=output_dir,
        thresholds=thresholds,
        generated_at=generated_at,
    )
    write_packet(output_dir, report)
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["final_status"],
        "policy_summaries": report["policy_summaries"],
        "no_write_guarantees": report["no_write_guarantees"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-report", required=True, type=Path)
    parser.add_argument("--high-accuracy-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-safe-joined-races", type=int, default=100)
    parser.add_argument("--current-min-top1-delta", type=float, default=0.02)
    parser.add_argument("--min-market-top1-delta", type=float, default=0.0)
    parser.add_argument("--max-box1-top-pick-share", type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_packet(
        rolling_report_path=args.rolling_report,
        high_accuracy_report_path=args.high_accuracy_report,
        output_dir=args.output_dir,
        thresholds=GateAuditThresholds(
            min_safe_joined_races=args.min_safe_joined_races,
            current_min_top1_delta=args.current_min_top1_delta,
            min_market_top1_delta=args.min_market_top1_delta,
            max_box1_top_pick_share=args.max_box1_top_pick_share,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
