#!/usr/bin/env python3
"""Report-only residual weight and calibration sweep.

This helper consumes frozen and later OOS ``market_residual_runner_matrix.csv``
files. It predeclares small residual weights, optional movement caps, and
blend modes, selects candidates using only the freeze matrix, then validates
the freeze-selected candidate on later OOS races. It writes report artifacts
only when an output directory is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "residual_weight_calibration_sweep_v1"
REPORT_FILE = "residual_weight_calibration_sweep_report.json"
CANDIDATE_METRICS_CSV = "candidate_metrics.csv"
CALIBRATION_BINS_CSV = "calibration_bins.csv"
FROZEN_MANIFEST_FILE = "frozen_candidate_manifest.json"
SUMMARY_FILE = "SUMMARY.md"
FINAL_DATA_MISSING = "DATA_MISSING"
FINAL_BLOCKED = "BLOCKED_KEEP_BASELINE"
FINAL_SEGMENT_DESIGN_ONLY = "SEGMENT_DESIGN_ONLY"
FINAL_READY_FOR_OWNER = "VALIDATION_READY_FOR_OWNER_REVIEW"

DEFAULT_WEIGHTS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25)
DEFAULT_CAPS: tuple[float | None, ...] = (None, 0.02, 0.05)
DEFAULT_MODES = ("linear_residual", "logit_residual")
DEFAULT_BIN_EDGES = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0000001)
PROTECTED_OUTPUT_PREFIXES = (
    "artifacts/full_evidence_orchestration_20260525",
    "artifacts/prediction_snapshots",
    "model_registry",
    "docs/model_registry",
    "ml_models_v4",
    "advanced_models",
)
NO_WRITE_GUARANTEES = {
    "live_db_write": False,
    "official_result_capture": False,
    "live_odds_capture": False,
    "model_fit": False,
    "model_artifact_write": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "promotion": False,
    "betting": False,
    "ev_output": False,
    "tgr_enabled": False,
}


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _normalize(scores: Sequence[float | None]) -> list[float] | None:
    if not scores or any(score is None for score in scores):
        return None
    cleaned = [max(float(score), 1e-12) for score in scores if score is not None]
    total = sum(cleaned)
    if total <= 0 or not math.isfinite(total):
        return None
    return [score / total for score in cleaned]


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-12), 1.0 - 1e-12)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clip_delta(value: float, cap: float | None) -> float:
    if cap is None:
        return value
    return max(-cap, min(cap, value))


def _group_by_race(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        race_id = str(row.get("race_id") or "").strip()
        if race_id:
            grouped[race_id].append(dict(row))
    return {
        race_id: sorted(
            race_rows,
            key=lambda row: (
                _safe_int(row.get("box_number")) or 999,
                str(row.get("dog_name") or ""),
            ),
        )
        for race_id, race_rows in grouped.items()
    }


def _accepted_races(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key: str,
    market_candidate_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = _group_by_race(rows)
    skipped: Counter[str] = Counter()
    races: list[dict[str, Any]] = []
    for race_id, race_rows in grouped.items():
        winners = [row for row in race_rows if _truthy(row.get("is_winner"))]
        if len(winners) != 1:
            skipped["missing_or_ambiguous_winner"] += 1
            continue
        if {str(row.get("candidate_key") or "").strip() for row in race_rows} != {
            candidate_key
        }:
            skipped["candidate_key_mismatch_requires_new_freeze"] += 1
            continue
        if {str(row.get("market_candidate_key") or "").strip() for row in race_rows} != {
            market_candidate_key
        }:
            skipped["market_candidate_key_mismatch"] += 1
            continue
        if any(_safe_float(row.get("market_probability")) is None for row in race_rows):
            skipped["missing_market_probability"] += 1
            continue
        if any(_safe_float(row.get("candidate_probability")) is None for row in race_rows):
            skipped["missing_candidate_probability"] += 1
            continue
        first = race_rows[0]
        races.append(
            {
                "race_id": race_id,
                "race_date": first.get("race_date"),
                "venue": first.get("venue"),
                "race_number": _safe_int(first.get("race_number")),
                "runner_count": _safe_int(first.get("runner_count")) or len(race_rows),
                "rows": race_rows,
            }
        )
    races.sort(
        key=lambda race: (
            str(race.get("race_date") or ""),
            str(race.get("venue") or ""),
            _safe_int(race.get("race_number")) or 999,
            str(race.get("race_id") or ""),
        )
    )
    return races, {
        "input_rows": len(rows),
        "input_races": len(grouped),
        "accepted_races": len(races),
        "skipped_counts": dict(sorted(skipped.items())),
    }


def _market_scores(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    return _normalize([_safe_float(row.get("market_probability")) for row in rows])


def _candidate_scores(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    return _normalize([_safe_float(row.get("candidate_probability")) for row in rows])


def _candidate_specs(
    *,
    weights: Sequence[float],
    caps: Sequence[float | None],
    modes: Sequence[str],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "candidate_key": "market_only_implied",
            "family": "market_baseline",
            "mode": "market_only",
            "weight": 0.0,
            "movement_cap": None,
        }
    ]
    for mode in modes:
        if mode not in {"linear_residual", "logit_residual"}:
            raise ValueError(f"mode_not_supported:{mode}")
        for weight in weights:
            if weight <= 0:
                continue
            for cap in caps:
                cap_key = "uncapped" if cap is None else str(cap).replace(".", "_")
                weight_key = str(weight).replace(".", "_")
                specs.append(
                    {
                        "candidate_key": f"{mode}_w{weight_key}_cap_{cap_key}",
                        "family": "residual_weight_sweep",
                        "mode": mode,
                        "weight": weight,
                        "movement_cap": cap,
                    }
                )
    return specs


def _blend_scores(
    rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[float] | None:
    market = _market_scores(rows)
    candidate = _candidate_scores(rows)
    if market is None or candidate is None:
        return None
    mode = str(spec.get("mode"))
    weight = float(spec.get("weight") or 0.0)
    cap = _safe_float(spec.get("movement_cap"))
    if mode == "market_only" or weight == 0:
        return market
    if mode == "linear_residual":
        blended = [
            m + _clip_delta(weight * (c - m), cap)
            for m, c in zip(market, candidate, strict=True)
        ]
        return _normalize(blended)
    if mode == "logit_residual":
        raw = [
            _sigmoid(_logit(m) + weight * (_logit(c) - _logit(m)))
            for m, c in zip(market, candidate, strict=True)
        ]
        blended = _normalize(raw)
        if blended is None:
            return None
        capped = [
            m + _clip_delta(b - m, cap)
            for m, b in zip(market, blended, strict=True)
        ]
        return _normalize(capped)
    raise ValueError(f"mode_not_supported:{mode}")


def _ranking_order(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> list[int]:
    return sorted(
        range(len(rows)),
        key=lambda index: (
            -scores[index],
            _safe_int(rows[index].get("box_number")) or 999,
            str(rows[index].get("dog_name") or ""),
        ),
    )


def _evaluate_spec(
    races: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    skipped: Counter[str] = Counter()
    race_count = 0
    top1 = 0
    top3 = 0
    rank_sum = 0.0
    brier_sum = 0.0
    logloss_sum = 0.0
    promoted = 0
    demoted = 0
    same = 0
    prediction_rows: list[dict[str, Any]] = []

    for race in races:
        rows = list(race.get("rows") or [])
        scores = _blend_scores(rows, spec)
        market = _market_scores(rows)
        if scores is None or market is None:
            skipped["score_missing"] += 1
            continue
        winners = [index for index, row in enumerate(rows) if _truthy(row.get("is_winner"))]
        if len(winners) != 1:
            skipped["missing_or_ambiguous_winner"] += 1
            continue
        winner_index = winners[0]
        candidate_order = _ranking_order(rows, scores)
        market_order = _ranking_order(rows, market)
        winner_rank = candidate_order.index(winner_index) + 1
        market_winner_rank = market_order.index(winner_index) + 1

        race_count += 1
        top1 += int(winner_rank == 1)
        top3 += int(winner_rank <= 3)
        rank_sum += winner_rank
        brier_sum += sum(
            (scores[index] - (1.0 if index == winner_index else 0.0)) ** 2
            for index in range(len(rows))
        )
        logloss_sum += -math.log(max(scores[winner_index], 1e-15))
        if winner_rank < market_winner_rank:
            promoted += 1
        elif winner_rank > market_winner_rank:
            demoted += 1
        else:
            same += 1
        prediction_rows.extend(
            {
                "race_id": race.get("race_id"),
                "race_date": race.get("race_date"),
                "venue": race.get("venue"),
                "candidate_key": spec.get("candidate_key"),
                "dog_name": row.get("dog_name"),
                "predicted_probability": scores[index],
                "actual_win": 1 if index == winner_index else 0,
            }
            for index, row in enumerate(rows)
        )

    if race_count == 0:
        return {
            "status": FINAL_DATA_MISSING,
            "race_count": 0,
            "blockers": ["no_evaluable_races"],
            "skipped_counts": dict(sorted(skipped.items())),
        }, prediction_rows

    return {
        "status": "EVALUATED",
        "race_count": race_count,
        "top1": top1 / race_count,
        "top3": top3 / race_count,
        "mean_winner_rank": rank_sum / race_count,
        "brier": brier_sum / race_count,
        "logloss": logloss_sum / race_count,
        "candidate_promoted_winner_count": promoted,
        "candidate_demoted_winner_count": demoted,
        "candidate_same_winner_rank_count": same,
        "skipped_counts": dict(sorted(skipped.items())),
        "blockers": [],
    }, prediction_rows


def _metric_deltas(market: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("top1", "top3", "mean_winner_rank", "brier", "logloss"):
        market_value = _safe_float(market.get(key))
        candidate_value = _safe_float(candidate.get(key))
        output[key] = (
            candidate_value - market_value
            if market_value is not None and candidate_value is not None
            else None
        )
    return output


def _float_or(value: Any, default: float) -> float:
    parsed = _safe_float(value)
    return parsed if parsed is not None else default


def _gate_decision(
    *,
    race_count: int,
    deltas: Mapping[str, Any],
    metrics: Mapping[str, Any],
    min_oos_races: int,
    promotion_review_races: int,
    concentration: Mapping[str, Any],
    min_race_dates_for_stability: int,
    min_venues_for_stability: int,
    max_single_date_share: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    if race_count < min_oos_races:
        return {
            "final_status": FINAL_DATA_MISSING,
            "hard_gates_pass": False,
            "materiality_gates_pass": False,
            "promotion_review_floor_met": False,
            "concentration_guard_pass": False,
            "blockers": ["oos_race_count_below_floor"],
        }
    hard_gates = {
        "top1_delta_gte_0": _float_or(deltas.get("top1"), 0.0) >= 0.0,
        "top3_delta_gte_0": _float_or(deltas.get("top3"), 0.0) >= 0.0,
        "mean_winner_rank_delta_lte_0": _float_or(
            deltas.get("mean_winner_rank"), 999.0
        )
        <= 0.0,
        "brier_delta_lte_0": _float_or(deltas.get("brier"), 999.0) <= 0.0,
        "logloss_delta_lte_0": _float_or(deltas.get("logloss"), 999.0) <= 0.0,
        "promoted_winners_gte_demoted_winners": int(
            metrics.get("candidate_promoted_winner_count") or 0
        )
        >= int(metrics.get("candidate_demoted_winner_count") or 0),
    }
    for key, passed in hard_gates.items():
        if not passed:
            blockers.append(key.replace("_gte_0", "_failed").replace("_lte_0", "_failed"))

    rank_materiality = (
        _float_or(deltas.get("top1"), 0.0) >= 0.02
        or _float_or(deltas.get("mean_winner_rank"), 999.0) <= -0.02
    )
    probability_materiality = (
        _float_or(deltas.get("brier"), 999.0) <= -0.001
        or _float_or(deltas.get("logloss"), 999.0) <= -0.005
    )
    if not rank_materiality:
        blockers.append("rank_materiality_gate_failed")
    if not probability_materiality:
        blockers.append("probability_materiality_gate_failed")

    if blockers:
        return {
            "final_status": FINAL_BLOCKED,
            "hard_gates": hard_gates,
            "hard_gates_pass": all(hard_gates.values()),
            "rank_materiality_gate_pass": rank_materiality,
            "probability_materiality_gate_pass": probability_materiality,
            "materiality_gates_pass": rank_materiality and probability_materiality,
            "promotion_review_floor_met": race_count >= promotion_review_races,
            "concentration_guard_pass": False,
            "blockers": blockers,
        }

    concentration_guard_pass = True
    if int(concentration.get("race_date_count") or 0) < min_race_dates_for_stability:
        concentration_guard_pass = False
        blockers.append("race_date_count_below_stability_minimum")
    if int(concentration.get("venue_count") or 0) < min_venues_for_stability:
        concentration_guard_pass = False
        blockers.append("venue_count_below_stability_minimum")
    share = _safe_float(concentration.get("max_single_race_date_share"))
    if share is not None and share > max_single_date_share:
        concentration_guard_pass = False
        blockers.append("single_race_date_concentration_above_maximum")

    promotion_review_floor_met = race_count >= promotion_review_races
    if promotion_review_floor_met and concentration_guard_pass:
        final_status = FINAL_READY_FOR_OWNER
    else:
        final_status = FINAL_SEGMENT_DESIGN_ONLY
        if not promotion_review_floor_met:
            blockers.append("promotion_review_floor_not_met")
        if not concentration_guard_pass:
            blockers.append("concentration_guard_failed")

    return {
        "final_status": final_status,
        "hard_gates": hard_gates,
        "hard_gates_pass": True,
        "rank_materiality_gate_pass": True,
        "probability_materiality_gate_pass": True,
        "materiality_gates_pass": True,
        "promotion_review_floor_met": promotion_review_floor_met,
        "concentration_guard_pass": concentration_guard_pass,
        "blockers": blockers,
    }


def _concentration(races: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    race_dates = Counter(str(race.get("race_date") or "DATA_MISSING") for race in races)
    venues = Counter(str(race.get("venue") or "DATA_MISSING") for race in races)
    race_count = len(races)
    max_date_count = max(race_dates.values(), default=0)
    return {
        "race_date_counts": dict(sorted(race_dates.items())),
        "venue_counts": dict(sorted(venues.items())),
        "race_date_count": len(race_dates),
        "venue_count": len(venues),
        "max_single_race_date_count": max_date_count,
        "max_single_race_date_share": max_date_count / race_count if race_count else None,
    }


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _float_or(row.get("freeze_top1_delta"), -999.0),
        _float_or(row.get("freeze_logloss_delta"), 999.0) * -1.0,
        _float_or(row.get("freeze_brier_delta"), 999.0) * -1.0,
        _float_or(row.get("freeze_mean_winner_rank_delta"), 999.0) * -1.0,
        _float_or(row.get("freeze_top3_delta"), -999.0),
    )


def _calibration_bins(
    *,
    split: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        probability = _safe_float(row.get("predicted_probability"))
        if probability is None:
            continue
        label = None
        for index in range(len(bin_edges) - 1):
            lower = bin_edges[index]
            upper = bin_edges[index + 1]
            if lower <= probability < upper:
                label = f"{lower:.2f}_{upper:.2f}"
                break
        if label is None:
            label = "out_of_range"
        by_key[(str(row.get("candidate_key")), label)].append(row)

    output: list[dict[str, Any]] = []
    for (candidate_key, label), rows in sorted(by_key.items()):
        probs = [_safe_float(row.get("predicted_probability")) or 0.0 for row in rows]
        wins = [1 if _truthy(row.get("actual_win")) else 0 for row in rows]
        avg_probability = sum(probs) / len(probs)
        actual_win_rate = sum(wins) / len(wins)
        output.append(
            {
                "split": split,
                "candidate_key": candidate_key,
                "probability_bin": label,
                "runner_count": len(rows),
                "winner_count": sum(wins),
                "avg_predicted_probability": avg_probability,
                "actual_win_rate": actual_win_rate,
                "calibration_error": avg_probability - actual_win_rate,
            }
        )
    return output


def _nearest_git_root(path: Path) -> Path | None:
    resolved = path.resolve()
    candidates = [resolved] if resolved.is_dir() else []
    candidates.extend(resolved.parents)
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def _assert_output_dir_safe(output_dir: Path, repo_root: Path | None = None) -> Path:
    resolved = output_dir.resolve()
    roots = []
    if repo_root is not None:
        roots.append(repo_root.resolve())
    else:
        roots.extend(
            root
            for root in (
                _nearest_git_root(resolved),
                _nearest_git_root(Path.cwd()),
                Path.cwd().resolve(),
            )
            if root is not None
        )
    for root in dict.fromkeys(roots):
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        relative_text = relative.as_posix()
        for prefix in PROTECTED_OUTPUT_PREFIXES:
            if relative_text == prefix or relative_text.startswith(prefix + "/"):
                raise ValueError(f"output_dir_protected:{prefix}")
    return resolved


def _assert_not_input_packet_output(output_dir: Path, input_paths: Sequence[Path]) -> None:
    for input_path in input_paths:
        input_dir = input_path.resolve().parent
        try:
            output_dir.resolve().relative_to(input_dir)
        except ValueError:
            continue
        raise ValueError(f"output_dir_must_not_write_inside_input_packet:{input_dir}")


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[str(path.relative_to(output_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return {
        "schema_version": "residual_weight_calibration_sweep_output_manifest_v1",
        "output_dir": str(output_dir),
        "files": files,
    }


def _write_candidate_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "candidate_key",
        "mode",
        "weight",
        "movement_cap",
        "freeze_status",
        "freeze_design_status",
        "freeze_race_count",
        "freeze_top1_delta",
        "freeze_top3_delta",
        "freeze_mean_winner_rank_delta",
        "freeze_brier_delta",
        "freeze_logloss_delta",
        "oos_status",
        "oos_race_count",
        "oos_final_status",
        "oos_top1_delta",
        "oos_top3_delta",
        "oos_mean_winner_rank_delta",
        "oos_brier_delta",
        "oos_logloss_delta",
        "oos_blockers",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field), sort_keys=True)
                        if field == "oos_blockers"
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def _write_calibration_bins_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "split",
        "candidate_key",
        "probability_bin",
        "runner_count",
        "winner_count",
        "avg_predicted_probability",
        "actual_win_rate",
        "calibration_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _summary_markdown(report: Mapping[str, Any]) -> str:
    selected = report.get("freeze_selected_candidate") or {}
    validation = report.get("oos_validation") or {}
    return "\n".join(
        [
            "# Residual Weight Calibration Sweep",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Freeze races: `{report.get('freeze_race_count')}`",
            f"- OOS races after freeze exclusion: `{report.get('oos_race_count')}`",
            f"- Candidate count: `{report.get('candidate_count')}`",
            f"- Freeze-selected candidate: `{selected.get('candidate_key')}`",
            f"- OOS decision: `{validation.get('decision')}`",
            f"- OOS candidate minus market: `{validation.get('candidate_minus_market')}`",
            f"- Blockers: `{validation.get('blockers')}`",
            "",
            "Selection used freeze metrics only. OOS metrics were not used to choose weights, caps, or modes.",
            "",
            "No runtime, DB, capture, training, promotion, registry, dependency, TGR, EV/betting, PR/GitHub, git-history, rolling-packet, or production pointer mutation was performed.",
            "",
        ]
    )


def _parse_float_list(value: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if not value:
        return tuple(default)
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("float_list_empty")
    return parsed


def _parse_caps(value: str | None) -> tuple[float | None, ...]:
    if not value:
        return DEFAULT_CAPS
    caps: list[float | None] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        caps.append(None if text in {"none", "uncapped"} else float(text))
    if not caps:
        raise ValueError("caps_empty")
    return tuple(caps)


def _parse_modes(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_MODES
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not modes:
        raise ValueError("modes_empty")
    return modes


def build_report(
    *,
    freeze_runner_matrix_csv: Path,
    oos_runner_matrix_csv: Path,
    candidate_key: str,
    market_candidate_key: str = "market_only_implied",
    freeze_report_json: Path | None = None,
    oos_report_json: Path | None = None,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    movement_caps: Sequence[float | None] = DEFAULT_CAPS,
    modes: Sequence[str] = DEFAULT_MODES,
    output_dir: Path | None = None,
    min_oos_races: int = 30,
    promotion_review_races: int = 100,
    min_race_dates_for_stability: int = 3,
    min_venues_for_stability: int = 3,
    max_single_date_share: float = 0.5,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    freeze_rows = _load_csv(freeze_runner_matrix_csv)
    oos_rows_all = _load_csv(oos_runner_matrix_csv)
    freeze_report = _load_json(freeze_report_json)
    oos_report = _load_json(oos_report_json)

    freeze_races, freeze_collection = _accepted_races(
        freeze_rows,
        candidate_key=candidate_key,
        market_candidate_key=market_candidate_key,
    )
    freeze_race_ids = {str(race.get("race_id")) for race in freeze_races}
    oos_rows = [row for row in oos_rows_all if str(row.get("race_id") or "") not in freeze_race_ids]
    oos_races, oos_collection = _accepted_races(
        oos_rows,
        candidate_key=candidate_key,
        market_candidate_key=market_candidate_key,
    )
    specs = _candidate_specs(weights=weights, caps=movement_caps, modes=modes)

    freeze_market, freeze_market_predictions = _evaluate_spec(
        freeze_races,
        {"candidate_key": "market_only_implied", "mode": "market_only", "weight": 0.0},
    )
    oos_market, oos_market_predictions = _evaluate_spec(
        oos_races,
        {"candidate_key": "market_only_implied", "mode": "market_only", "weight": 0.0},
    )
    candidate_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    _ = freeze_market_predictions, oos_market_predictions

    for spec in specs:
        freeze_metrics, freeze_predictions = _evaluate_spec(freeze_races, spec)
        oos_metrics, oos_predictions = _evaluate_spec(oos_races, spec)
        freeze_deltas = _metric_deltas(freeze_market, freeze_metrics)
        oos_deltas = _metric_deltas(oos_market, oos_metrics)
        freeze_decision = _gate_decision(
            race_count=int(freeze_metrics.get("race_count") or 0),
            deltas=freeze_deltas,
            metrics=freeze_metrics,
            min_oos_races=min_oos_races,
            promotion_review_races=promotion_review_races,
            concentration=_concentration(freeze_races),
            min_race_dates_for_stability=min_race_dates_for_stability,
            min_venues_for_stability=min_venues_for_stability,
            max_single_date_share=max_single_date_share,
        )
        oos_decision = _gate_decision(
            race_count=int(oos_metrics.get("race_count") or 0),
            deltas=oos_deltas,
            metrics=oos_metrics,
            min_oos_races=min_oos_races,
            promotion_review_races=promotion_review_races,
            concentration=_concentration(oos_races),
            min_race_dates_for_stability=min_race_dates_for_stability,
            min_venues_for_stability=min_venues_for_stability,
            max_single_date_share=max_single_date_share,
        )
        candidate_rows.append(
            {
                "candidate_key": spec.get("candidate_key"),
                "mode": spec.get("mode"),
                "weight": spec.get("weight"),
                "movement_cap": spec.get("movement_cap"),
                "freeze_status": freeze_metrics.get("status"),
                "freeze_design_status": freeze_decision.get("final_status"),
                "freeze_race_count": freeze_metrics.get("race_count"),
                "freeze_top1_delta": freeze_deltas.get("top1"),
                "freeze_top3_delta": freeze_deltas.get("top3"),
                "freeze_mean_winner_rank_delta": freeze_deltas.get("mean_winner_rank"),
                "freeze_brier_delta": freeze_deltas.get("brier"),
                "freeze_logloss_delta": freeze_deltas.get("logloss"),
                "freeze_blockers": freeze_decision.get("blockers"),
                "oos_status": oos_metrics.get("status"),
                "oos_race_count": oos_metrics.get("race_count"),
                "oos_final_status": oos_decision.get("final_status"),
                "oos_top1_delta": oos_deltas.get("top1"),
                "oos_top3_delta": oos_deltas.get("top3"),
                "oos_mean_winner_rank_delta": oos_deltas.get("mean_winner_rank"),
                "oos_brier_delta": oos_deltas.get("brier"),
                "oos_logloss_delta": oos_deltas.get("logloss"),
                "oos_blockers": oos_decision.get("blockers"),
                "freeze_metrics": freeze_metrics,
                "oos_metrics": oos_metrics,
                "freeze_decision": freeze_decision,
                "oos_decision": oos_decision,
            }
        )
        calibration_rows.extend(
            _calibration_bins(split="freeze", prediction_rows=freeze_predictions)
        )
        calibration_rows.extend(_calibration_bins(split="oos", prediction_rows=oos_predictions))

    non_market_freeze_passes = [
        row
        for row in candidate_rows
        if row.get("candidate_key") != "market_only_implied"
        and row.get("freeze_design_status")
        in {FINAL_SEGMENT_DESIGN_ONLY, FINAL_READY_FOR_OWNER}
    ]
    selected = (
        max(non_market_freeze_passes, key=_candidate_sort_key)
        if non_market_freeze_passes
        else None
    )
    if selected is None:
        blockers = ["no_non_market_candidate_passed_freeze_design_gates"]
        if len(oos_races) < min_oos_races:
            blockers.insert(0, "oos_race_count_below_floor")
        final_status = (
            FINAL_DATA_MISSING
            if not freeze_races or len(oos_races) < min_oos_races
            else FINAL_BLOCKED
        )
        oos_validation = {
            "decision": final_status,
            "candidate_key": None,
            "candidate_minus_market": None,
            "blockers": blockers,
        }
    else:
        final_status = str(selected["oos_final_status"])
        oos_validation = {
            "decision": final_status,
            "candidate_key": selected.get("candidate_key"),
            "candidate_minus_market": {
                "top1": selected.get("oos_top1_delta"),
                "top3": selected.get("oos_top3_delta"),
                "mean_winner_rank": selected.get("oos_mean_winner_rank_delta"),
                "brier": selected.get("oos_brier_delta"),
                "logloss": selected.get("oos_logloss_delta"),
            },
            "blockers": selected.get("oos_blockers"),
        }

    manifest = {
        "schema_version": "frozen_residual_weight_candidate_manifest_v1",
        "selection_rule": "freeze_metrics_only; oos_metrics_not_used_for_selection",
        "baseline": "market_only_implied",
        "weights": list(weights),
        "movement_caps": list(movement_caps),
        "modes": list(modes),
        "selected_candidate": {
            key: selected.get(key) if selected else None
            for key in (
                "candidate_key",
                "mode",
                "weight",
                "movement_cap",
                "freeze_design_status",
                "freeze_top1_delta",
                "freeze_brier_delta",
                "freeze_logloss_delta",
            )
        },
        "freeze_ranked_candidates": [
            {
                key: row.get(key)
                for key in (
                    "candidate_key",
                    "mode",
                    "weight",
                    "movement_cap",
                    "freeze_design_status",
                    "freeze_top1_delta",
                    "freeze_top3_delta",
                    "freeze_mean_winner_rank_delta",
                    "freeze_brier_delta",
                    "freeze_logloss_delta",
                    "freeze_blockers",
                )
            }
            for row in sorted(candidate_rows, key=_candidate_sort_key, reverse=True)
        ],
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "candidate_key": candidate_key,
        "market_candidate_key": market_candidate_key,
        "freeze_runner_matrix_csv": str(freeze_runner_matrix_csv),
        "oos_runner_matrix_csv": str(oos_runner_matrix_csv),
        "freeze_report_json": str(freeze_report_json) if freeze_report_json else None,
        "oos_report_json": str(oos_report_json) if oos_report_json else None,
        "freeze_report_generated_at": freeze_report.get("generated_at"),
        "oos_report_generated_at": oos_report.get("generated_at"),
        "freeze_race_count": len(freeze_races),
        "oos_race_count": len(oos_races),
        "candidate_count": len(specs),
        "sample_floors": {
            "min_oos_races": min_oos_races,
            "promotion_review_races": promotion_review_races,
            "min_race_dates_for_stability": min_race_dates_for_stability,
            "min_venues_for_stability": min_venues_for_stability,
            "max_single_date_share": max_single_date_share,
        },
        "collection": {
            "freeze": freeze_collection,
            "oos_after_freeze_exclusion": oos_collection,
            "freeze_race_id_excluded_from_oos": len(freeze_race_ids),
        },
        "freeze_market_metrics": freeze_market,
        "oos_market_metrics": oos_market,
        "candidate_metrics": candidate_rows,
        "freeze_selected_candidate": manifest["selected_candidate"],
        "oos_validation": oos_validation,
        "frozen_candidate_manifest": manifest,
        "calibration_bin_count": len(calibration_rows),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }

    if output_dir is not None:
        resolved_output = _assert_output_dir_safe(output_dir)
        _assert_not_input_packet_output(
            resolved_output,
            [freeze_runner_matrix_csv, oos_runner_matrix_csv],
        )
        resolved_output.mkdir(parents=True, exist_ok=True)
        _write_json(resolved_output / REPORT_FILE, report)
        _write_candidate_metrics_csv(resolved_output / CANDIDATE_METRICS_CSV, candidate_rows)
        _write_calibration_bins_csv(resolved_output / CALIBRATION_BINS_CSV, calibration_rows)
        _write_json(resolved_output / FROZEN_MANIFEST_FILE, manifest)
        _write_text(resolved_output / SUMMARY_FILE, _summary_markdown(report))
        _write_text(resolved_output / "final_status.txt", final_status + "\n")
        _write_json(resolved_output / "output_manifest.json", _output_manifest(resolved_output))

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-runner-matrix-csv", type=Path, required=True)
    parser.add_argument("--oos-runner-matrix-csv", type=Path, required=True)
    parser.add_argument("--candidate-key", default="stage2_uncalibrated_market_blend_25")
    parser.add_argument("--market-candidate-key", default="market_only_implied")
    parser.add_argument("--freeze-report-json", type=Path)
    parser.add_argument("--oos-report-json", type=Path)
    parser.add_argument("--weights", help="Comma-separated residual weights.")
    parser.add_argument("--movement-caps", help="Comma-separated caps, use none for uncapped.")
    parser.add_argument("--modes", help="Comma-separated modes: linear_residual,logit_residual.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-oos-races", type=int, default=30)
    parser.add_argument("--promotion-review-races", type=int, default=100)
    parser.add_argument("--min-race-dates-for-stability", type=int, default=3)
    parser.add_argument("--min-venues-for-stability", type=int, default=3)
    parser.add_argument("--max-single-date-share", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        freeze_runner_matrix_csv=args.freeze_runner_matrix_csv,
        oos_runner_matrix_csv=args.oos_runner_matrix_csv,
        candidate_key=args.candidate_key,
        market_candidate_key=args.market_candidate_key,
        freeze_report_json=args.freeze_report_json,
        oos_report_json=args.oos_report_json,
        weights=_parse_float_list(args.weights, DEFAULT_WEIGHTS),
        movement_caps=_parse_caps(args.movement_caps),
        modes=_parse_modes(args.modes),
        output_dir=args.output_dir,
        min_oos_races=args.min_oos_races,
        promotion_review_races=args.promotion_review_races,
        min_race_dates_for_stability=args.min_race_dates_for_stability,
        min_venues_for_stability=args.min_venues_for_stability,
        max_single_date_share=args.max_single_date_share,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
