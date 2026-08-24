#!/usr/bin/env python3
"""Freeze, evaluate, and verify a report-only early-pace topology audit.

Freeze never reads result values.  It reproduces the sealed latent experiment's
causal early-pace state with the exact frozen implementation, projects only
pre-jump fields, fixes outcome-blind thresholds/groups/folds, and seals the
matrix.  Evaluate opens approved development labels only after that seal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0")
LATENT_ROOT = SOURCE_ROOT / "greyhound-latent-ability-signal-20260818"
LATENT_SCRIPT = LATENT_ROOT / "scripts/run_latent_ability_market_residual_experiment.py"
LATENT_MATRIX = LATENT_ROOT / "artifacts/latent_ability_market_residual_20260818_report_only/feature_matrix.jsonl"
TARGETS = ROOT / "artifacts/form_speed_market_residual_20260818_report_only/feature_matrix.jsonl"
HISTORY = SOURCE_ROOT / "greyhound-fast-nonfavourite-native-enrichment-20260818/artifacts/fast_nonfavourite_native_history_enrichment_20260818_report_only/recovered_history.jsonl"
BETFAIR = ROOT / "artifacts/betfair_historical_surface_20260817_report_only/sportsbet_betfair_joined_surface.jsonl"
DEFAULT_OUT = ROOT / "artifacts/pace_topology_mechanism_audit_20260818_report_only"

START = "2026-06-10"
END = "2026-07-18"
FORWARD_BOUNDARY = "2026-08-18"
SEED = 20260818
BOOTSTRAPS = 5000
MIN_N = 75
MIN_MEETING_DATES = 15
EXPECTED_HASHES = {
    LATENT_SCRIPT: "4057d1c4867b5473794158b57e23f2caf865aaab19ea01542d5d1d74f3fd47c1",
    LATENT_MATRIX: "31f0ecf8fca359c28a903fed8cf1c3cf7a39251300d0eebc83652ebd1da4dcf6",
    TARGETS: "01cbff10bf162d6b7b712c6c5c14a29bdb3cafaa2ca962d704a0634ec54bd148",
    HISTORY: "f087195d07320501b9c57b436f2a784f3ae86425bae8327e37fc00dc1fafc3f6",
    BETFAIR: "86fabb05556160e555f076322eb8786b6166e369a6a8ec57d475c0e4a06e67f7",
}
PACE_FIELDS = (
    "race_id", "race_date", "box_number", "field_size", "jump_at",
    "odds_capture_timestamp", "native_thedogs_dog_id",
    "market_implied_probability", "canonical_sportsbet_win_odds",
    "rating_cutoff_exclusive", "pace_rank_fraction", "pace_gap_to_best",
    "pace_uncertainty", "pace_effective_starts",
    "pace_log_effective_starts", "pace_missing",
)
DIRECTIONAL_GROUPS = {
    "LONE_LEADER_POSITIVE": "positive",
    "ADJACENT_PRESSURE_ADVERSE": "negative",
    "PRESSURED_FAVOURITE_ADVERSE": "negative",
    "CLEAR_PATH_NONFAV_POSITIVE": "positive",
}
DIAGNOSTIC_GROUPS = (
    "FAVOURITE_INSIDE_PACE_IMBALANCE_DIAGNOSTIC",
    "FAVOURITE_OUTSIDE_PACE_IMBALANCE_DIAGNOSTIC",
)
ALL_GROUPS = tuple(DIRECTIONAL_GROUPS) + DIAGNOSTIC_GROUPS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def canonical_line(row: Mapping[str, Any]) -> bytes:
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_line(row))


def rows_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_line(row))
    return digest.hexdigest()


def write_checksums(out: Path, names: Sequence[str], manifest: str) -> None:
    (out / manifest).write_text("".join(f"{sha256(out / name)}  {name}\n" for name in names), encoding="utf-8")


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def exact_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["race_id"]), int(row["box_number"]), str(row["odds_capture_timestamp"])


def race_box_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["race_id"]), int(row["box_number"])


def verify_inputs() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"frozen_input_hash_mismatch:{path}")


def outcome_blind_validate_targets(rows: Sequence[Mapping[str, Any]], module: Any) -> None:
    """Replacement for the latent builder's label-aware target validation."""
    if not rows or any(not (START <= str(row.get("race_date")) <= END) for row in rows):
        raise SystemExit("target_date_boundary_breached")
    if any(str(row.get("race_date")) >= FORWARD_BOUNDARY for row in rows):
        raise SystemExit("forward_target_boundary_breached")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("race_id"))].append(row)
    for race_rows in grouped.values():
        if any(not row.get("native_thedogs_dog_id") for row in race_rows):
            raise SystemExit("native_dog_identity_missing")
        boxes = [int(row["box_number"]) for row in race_rows]
        if len(boxes) != len(set(boxes)) or any(box < 1 or box > 8 for box in boxes):
            raise SystemExit("target_box_invalid_or_duplicate")
        probabilities = [float(row["market_implied_probability"]) for row in race_rows]
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            raise SystemExit("sportsbet_probability_not_normalized")
        if any(module.epoch(str(row["odds_capture_timestamp"])) >= module.epoch(str(row["jump_at"])) for row in race_rows):
            raise SystemExit("sportsbet_not_prejump")


def load_frozen_latent_module() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_latent_ability_v1", LATENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit("latent_implementation_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_targets = lambda rows: outcome_blind_validate_targets(rows, module)
    return module


def project_pace_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = [{field: row[field] for field in PACE_FIELDS} for row in rows]
    projected.sort(key=lambda row: (str(row["race_date"]), str(row["race_id"]), int(row["box_number"])))
    return projected


def chronological_folds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    races_by_date: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        races_by_date[str(row["race_date"])].add(str(row["race_id"]))
    dates = sorted(races_by_date)
    total = sum(len(races_by_date[day]) for day in dates)
    targets = (total / 3.0, 2.0 * total / 3.0)
    boundaries: list[int] = []
    cumulative = 0
    for index, day in enumerate(dates):
        cumulative += len(races_by_date[day])
        if len(boundaries) < 2 and cumulative >= targets[len(boundaries)]:
            boundaries.append(index)
    if len(boundaries) != 2 or boundaries[0] >= boundaries[1] or boundaries[1] >= len(dates) - 1:
        raise SystemExit("chronological_fold_construction_failed")
    ranges = ((0, boundaries[0]), (boundaries[0] + 1, boundaries[1]), (boundaries[1] + 1, len(dates) - 1))
    return [
        {
            "id": fold_id,
            "start": dates[first],
            "end": dates[last],
            "races": sum(len(races_by_date[day]) for day in dates[first : last + 1]),
        }
        for fold_id, (first, last) in enumerate(ranges, 1)
    ]


def competition_rank(row: Mapping[str, Any], race_rows: Sequence[Mapping[str, Any]]) -> int:
    value = float(row["market_implied_probability"])
    return 1 + sum(float(other["market_implied_probability"]) > value for other in race_rows)


def quantile(values: Sequence[float], percentile: float, name: str) -> float:
    if not values:
        raise SystemExit(f"threshold_distribution_empty:{name}")
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def topology_distributions(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    races: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        races[str(row["race_id"])].append(row)
    leader_gaps: list[float] = []
    adjacent_differences: list[float] = []
    positive_adjacent_pressure: list[float] = []
    imbalance_magnitudes: list[float] = []
    source_complete = pace_complete = full_eight = spatial_gap_races = 0
    missing_pace_rows = 0
    for race_rows in races.values():
        boxes = [int(row["box_number"]) for row in race_rows]
        declared_sizes = {int(row["field_size"]) for row in race_rows}
        complete = (
            len(declared_sizes) == 1
            and next(iter(declared_sizes)) == len(race_rows)
            and len(boxes) == len(set(boxes))
            and all(1 <= box <= 8 for box in boxes)
        )
        source_complete += int(complete)
        spatial_gap_races += int(set(boxes) != set(range(min(boxes), max(boxes) + 1)))
        full_eight += int(complete and set(boxes) == set(range(1, 9)))
        missing_pace_rows += sum(int(row["pace_missing"]) == 1 for row in race_rows)
        if not complete or len(race_rows) < 3 or any(int(row["pace_missing"]) != 0 for row in race_rows):
            continue
        pace_complete += 1
        ordered = sorted((float(row["pace_gap_to_best"]), int(row["box_number"])) for row in race_rows)
        best, second = ordered[-1][0], ordered[-2][0]
        if best > second:
            leader_gaps.append(best - second)
        by_box = {int(row["box_number"]): float(row["pace_gap_to_best"]) for row in race_rows}
        for box, score in by_box.items():
            for neighbour in (box - 1, box + 1):
                if neighbour not in by_box or neighbour < box:
                    continue
                difference = abs(score - by_box[neighbour])
                if difference > 0:
                    adjacent_differences.append(difference)
            pressures = [by_box[neighbour] - score for neighbour in (box - 1, box + 1) if neighbour in by_box]
            if pressures and max(pressures) > 0:
                positive_adjacent_pressure.append(max(pressures))
        if set(by_box) == set(range(1, 9)):
            inside = float(np.mean([by_box[box] for box in range(1, 5)]))
            outside = float(np.mean([by_box[box] for box in range(5, 9)]))
            if not math.isclose(inside, outside, abs_tol=1e-15):
                imbalance_magnitudes.append(abs(inside - outside))
    coverage = {
        "races": len(races),
        "runner_rows": len(rows),
        "source_complete_unique_box_races": source_complete,
        "complete_pace_topology_races": pace_complete,
        "incomplete_pace_topology_races": len(races) - pace_complete,
        "pace_missing_runner_rows": missing_pace_rows,
        "full_eight_box_races": full_eight,
        "races_with_internal_box_gaps": spatial_gap_races,
        "scratch_flag_available": False,
        "vacancy_flag_available": False,
        "scratch_vacancy_interpretation": "absent boxes are spatial gaps only; no cause, collision, rail preference, or wide preference is inferred",
    }
    distributions = {
        "leader_gap_positive_n": len(leader_gaps),
        "adjacent_difference_positive_n": len(adjacent_differences),
        "positive_adjacent_pressure_n": len(positive_adjacent_pressure),
        "full_eight_nonzero_imbalance_n": len(imbalance_magnitudes),
    }
    thresholds = {
        "large_leader_gap_q75": quantile(leader_gaps, 75, "leader_gap"),
        "comparable_adjacent_difference_q25": quantile(adjacent_differences, 25, "adjacent_difference"),
        "high_adjacent_pressure_q75": quantile(positive_adjacent_pressure, 75, "adjacent_pressure"),
        "large_inside_outside_imbalance_q75": quantile(imbalance_magnitudes, 75, "inside_outside_imbalance"),
        "distributions": distributions,
    }
    return coverage, thresholds


def assign_topology(rows: Sequence[dict[str, Any]], thresholds: Mapping[str, Any], folds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    races: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        races[str(row["race_id"])].append(dict(row))
    output: list[dict[str, Any]] = []
    for race_rows in races.values():
        boxes = [int(row["box_number"]) for row in race_rows]
        complete = (
            len({int(row["field_size"]) for row in race_rows}) == 1
            and int(race_rows[0]["field_size"]) == len(race_rows)
            and len(boxes) == len(set(boxes))
            and len(race_rows) >= 3
            and all(1 <= box <= 8 for box in boxes)
            and all(int(row["pace_missing"]) == 0 for row in race_rows)
        )
        fold = next(int(item["id"]) for item in folds if str(item["start"]) <= str(race_rows[0]["race_date"]) <= str(item["end"]))
        if complete:
            by_box = {int(row["box_number"]): float(row["pace_gap_to_best"]) for row in race_rows}
            ordered = sorted(race_rows, key=lambda row: (-float(row["pace_gap_to_best"]), int(row["box_number"])))
            leader = ordered[0]
            leader_gap = float(leader["pace_gap_to_best"]) - float(ordered[1]["pace_gap_to_best"])
            imbalance = None
            if set(by_box) == set(range(1, 9)):
                imbalance = float(np.mean([by_box[b] for b in range(1, 5)]) - np.mean([by_box[b] for b in range(5, 9)]))
        else:
            by_box, leader, leader_gap, imbalance = {}, None, None, None
        for row in race_rows:
            box = int(row["box_number"])
            market_rank = competition_rank(row, race_rows)
            neighbours = [by_box[b] for b in (box - 1, box + 1) if b in by_box] if complete else []
            score = float(row["pace_gap_to_best"]) if complete else None
            maximum_pressure = max((value - float(score) for value in neighbours), default=None) if complete else None
            comparable_adjacent = bool(
                complete and any(abs(value - float(score)) <= float(thresholds["comparable_adjacent_difference_q25"]) for value in neighbours)
            )
            is_leader = complete and row is leader
            large_lead = bool(is_leader and float(leader_gap) >= float(thresholds["large_leader_gap_q75"]))
            high_pressure = bool(
                maximum_pressure is not None and maximum_pressure >= float(thresholds["high_adjacent_pressure_q75"])
            )
            large_imbalance = bool(
                imbalance is not None and abs(imbalance) >= float(thresholds["large_inside_outside_imbalance_q75"])
            )
            groups = {
                "LONE_LEADER_POSITIVE": bool(large_lead and not comparable_adjacent),
                "ADJACENT_PRESSURE_ADVERSE": high_pressure,
                "PRESSURED_FAVOURITE_ADVERSE": bool(market_rank == 1 and high_pressure),
                "CLEAR_PATH_NONFAV_POSITIVE": bool(market_rank >= 2 and large_lead and not comparable_adjacent),
                "FAVOURITE_INSIDE_PACE_IMBALANCE_DIAGNOSTIC": bool(market_rank == 1 and large_imbalance and float(imbalance) > 0),
                "FAVOURITE_OUTSIDE_PACE_IMBALANCE_DIAGNOSTIC": bool(market_rank == 1 and large_imbalance and float(imbalance) < 0),
            }
            item = dict(row)
            item.update(
                {
                    "evaluation_fold_id": fold,
                    "topology_complete": complete,
                    "market_rank": market_rank,
                    "leader_gap": leader_gap if is_leader else None,
                    "adjacent_boxes_present": [b for b in (box - 1, box + 1) if b in by_box],
                    "maximum_adjacent_pace_pressure": maximum_pressure,
                    "comparable_adjacent": comparable_adjacent,
                    "inside_minus_outside_mean_pace": imbalance,
                    "mechanisms": groups,
                }
            )
            output.append(item)
    output.sort(key=lambda row: (str(row["race_date"]), str(row["race_id"]), int(row["box_number"])))
    return output


def freeze(out: Path) -> None:
    if out.exists():
        raise SystemExit("output_exists")
    verify_inputs()
    module = load_frozen_latent_module()
    rebuilt, latent_population = module.build_feature_rows(TARGETS, HISTORY)
    replay = project_pace_rows(rebuilt)
    sealed_projection = project_pace_rows(load_jsonl(LATENT_MATRIX))
    if rows_digest(replay) != rows_digest(sealed_projection) or replay != sealed_projection:
        raise SystemExit("early_pace_state_byte_replay_mismatch")
    if len(replay) != 7000 or len({str(row["race_id"]) for row in replay}) != 989:
        raise SystemExit("development_population_mismatch")
    coverage, thresholds = topology_distributions(replay)
    folds = chronological_folds(replay)
    topology = assign_topology(replay, thresholds, folds)
    group_sizes = {name: sum(bool(row["mechanisms"][name]) for row in topology) for name in ALL_GROUPS}
    group_dates = {name: len({str(row["race_date"]) for row in topology if row["mechanisms"][name]}) for name in ALL_GROUPS}
    blockers = []
    if not any(group_sizes[name] >= MIN_N and group_dates[name] >= MIN_MEETING_DATES for name in DIRECTIONAL_GROUPS):
        blockers.append("NO_DIRECTIONAL_GROUP_MEETS_PREDECLARED_MINIMUM_COVERAGE")
    out.mkdir(parents=True)
    write_jsonl(out / "pace_state_replay.jsonl", replay)
    write_jsonl(out / "topology_matrix.jsonl", topology)
    coverage.update({"pre_outcome_group_sizes": group_sizes, "pre_outcome_group_meeting_dates": group_dates})
    write_json(out / "coverage_audit.json", coverage)
    protocol = {
        "schema_version": "pace_topology_mechanism_protocol_v1",
        "status": "PROTOCOL_FROZEN_READY_TO_EVALUATE" if not blockers else "PROTOCOL_FROZEN_COVERAGE_BLOCKED",
        "authority": "research_only_no_model_fit_no_tuning_no_betting_no_deployment_no_promotion",
        "population": {"start": START, "end": END, "races": 989, "runner_rows": 7000},
        "early_pace_state": {
            "source_implementation": str(LATENT_SCRIPT),
            "source_implementation_sha256": sha256(LATENT_SCRIPT),
            "source_matrix": str(LATENT_MATRIX),
            "source_matrix_sha256": sha256(LATENT_MATRIX),
            "replay_projection_sha256": sha256(out / "pace_state_replay.jsonl"),
            "sealed_projection_sha256": rows_digest(sealed_projection),
            "byte_for_byte_projection_match": True,
            "definition": "unchanged frozen causal first_sectional_seconds rating state; exact track-distance expanding prior moments, race-relative robust score, pre-race opposition, 180-day decay, prior precision 2, and simultaneous post-race updates",
            "history_rows": latent_population["history_rows"],
            "history_first_sectional_rows": latent_population["history_first_sectional_rows"],
            "pace_rating_rows": latent_population["pace_rating_rows"],
        },
        "identity_timing": "exact race_id+box_number+odds_capture_timestamp and native TheDogs dog ID; history native race+dog ID and immutable receipt hashes; all history strictly before target jump",
        "box_scratch_vacancy": coverage,
        "thresholds": {
            **thresholds,
            "source": "outcome-blind distributions over complete frozen pace topology only",
            "post_outcome_search": False,
        },
        "mechanisms": {
            "LONE_LEADER_POSITIVE": "pace rank 1; leader gap >= frozen q75; no present box+/-1 starter within frozen comparable-adjacent q25 difference",
            "ADJACENT_PRESSURE_ADVERSE": "a present box+/-1 starter has faster pace by at least frozen positive-pressure q75",
            "PRESSURED_FAVOURITE_ADVERSE": "Sportsbet competition rank 1 and ADJACENT_PRESSURE_ADVERSE",
            "CLEAR_PATH_NONFAV_POSITIVE": "Sportsbet competition rank >=2, pace rank 1, leader gap >= frozen q75, and no comparable present adjacent starter",
            "FAVOURITE_INSIDE_PACE_IMBALANCE_DIAGNOSTIC": "full boxes 1..8; favourite in a race where mean pace boxes 1..4 minus 5..8 >= frozen absolute-imbalance q75; no directional gate",
            "FAVOURITE_OUTSIDE_PACE_IMBALANCE_DIAGNOSTIC": "full boxes 1..8; favourite in a race where mean pace boxes 1..4 minus 5..8 <= negative frozen absolute-imbalance q75; no directional gate",
            "pace_rank_ties": "pace leader uses descending frozen pace score then box only for deterministic ordering; zero best-second gaps do not meet large-lead definition",
            "market_rank_ties": "competition rank: 1 plus count with strictly higher normalized Sportsbet probability",
        },
        "minimum_coverage": {"runner_rows": MIN_N, "meeting_dates": MIN_MEETING_DATES, "fixed_before_labels": True},
        "folds": folds,
        "metrics": {
            "baseline": "corrected normalized Sportsbet WIN probability",
            "primary": "mean(label_is_winner - Sportsbet normalized probability)",
            "uncertainty": {"unit": "meeting date", "method": "percentile cluster bootstrap", "repetitions": BOOTSTRAPS, "seed": SEED},
            "reported": ["N", "wins", "summed expected wins", "mean residual", "cluster CI95", "calibration ratio", "fold residuals", "fixed 1u P&L/ROI CI/max drawdown"],
            "betfair": "strict complete scheduled-off overlap only; p95_5=0.95*p_sportsbet+0.05*normalized(1/scheduled_off_back_price); probability diagnostic only",
        },
        "decision": "PACE_TOPOLOGY_SIGNAL_WORTH_MODELLING iff a directional group meets frozen minimum coverage, residual CI excludes zero in its expected direction, and the expected residual direction holds in >=2/3 folds; otherwise NO_PACE_TOPOLOGY_SIGNAL; definitional coverage failure yields DATA_COVERAGE_BLOCKED",
        "blockers": blockers,
        "forward_exclusions": {"outcomes_on_or_after": FORWARD_BOUNDARY, "october_outcomes": "unopened", "forward_cohorts": "untouched"},
        "inputs": {str(path): sha256(path) for path in EXPECTED_HASHES},
        "topology_matrix_sha256": sha256(out / "topology_matrix.jsonl"),
    }
    write_json(out / "protocol.json", protocol)
    write_checksums(out, ("pace_state_replay.jsonl", "topology_matrix.jsonl", "coverage_audit.json", "protocol.json"), "SEALED_SHA256SUMS")


def cluster_ci(rows: Sequence[Mapping[str, Any]], value_key: str) -> list[float | None]:
    if not rows:
        return [None, None]
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row["race_date"])].append(float(row[value_key]))
    dates = sorted(clusters)
    arrays = [np.asarray(clusters[day], dtype=float) for day in dates]
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(arrays), len(arrays))
        draws[index] = float(np.concatenate([arrays[item] for item in sampled]).mean())
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def max_drawdown(values: Sequence[float]) -> float:
    cumulative = peak = drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def group_metrics(name: str, rows: Sequence[dict[str, Any]], probability_key: str) -> dict[str, Any]:
    enriched = []
    for row in rows:
        item = dict(row)
        probability = float(row[probability_key])
        item["_residual"] = int(row["label_is_winner"]) - probability
        item["_pnl"] = float(row["canonical_sportsbet_win_odds"]) - 1.0 if int(row["label_is_winner"]) else -1.0
        enriched.append(item)
    enriched.sort(key=lambda row: (str(row["jump_at"]), str(row["race_id"]), int(row["box_number"])))
    n = len(enriched)
    wins = sum(int(row["label_is_winner"]) for row in enriched)
    expected = sum(float(row[probability_key]) for row in enriched)
    folds = []
    for fold_id in (1, 2, 3):
        selected = [row for row in enriched if int(row["evaluation_fold_id"]) == fold_id]
        folds.append(
            {
                "id": fold_id,
                "N": len(selected),
                "residual_mean": float(np.mean([row["_residual"] for row in selected])) if selected else None,
            }
        )
    pnl = [float(row["_pnl"]) for row in enriched]
    return {
        "group": name,
        "N": n,
        "meeting_dates": len({str(row["race_date"]) for row in enriched}),
        "wins": wins,
        "summed_expected_wins": expected,
        "mean_observed_minus_market": float(np.mean([row["_residual"] for row in enriched])) if enriched else None,
        "meeting_date_cluster_ci95": cluster_ci(enriched, "_residual"),
        "calibration_ratio": wins / expected if expected > 0 else None,
        "folds": folds,
        "economic": {
            "stake_units": n,
            "pnl_units": sum(pnl),
            "roi": float(np.mean(pnl)) if pnl else None,
            "roi_meeting_date_cluster_ci95": cluster_ci(enriched, "_pnl"),
            "maximum_drawdown_units": max_drawdown(pnl),
            "interpretation": "fixed 1u descriptive only; no selection or betting authority",
        },
    }


def strict_betfair_runner_is_eligible(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    status = row.get("betfair_scheduled_off_back_price_status")
    clock_precedes = row.get("scheduled_clock_precedes_provider_actual_off_clock")
    price = row.get("betfair_scheduled_off_back_price")
    if type(status) is not str or status != "PRESENT":
        return False
    if type(clock_precedes) is not bool or clock_precedes is not True:
        return False
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        return False
    return math.isfinite(float(price)) and float(price) > 1.0


def strict_betfair_probabilities(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int], float]:
    source = load_jsonl(BETFAIR)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in source:
        key = race_box_key(row)
        if key in by_key:
            raise SystemExit("duplicate_betfair_join_identity")
        by_key[key] = row
    races: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        races[str(row["race_id"])].append(row)
    probabilities: dict[tuple[str, int], float] = {}
    for race_rows in races.values():
        joined = [by_key.get(race_box_key(row)) for row in race_rows]
        if not all(strict_betfair_runner_is_eligible(item) for item in joined):
            continue
        prices = [float(item["betfair_scheduled_off_back_price"]) for item in joined if item is not None]
        raw = np.asarray([1.0 / float(price) for price in prices], dtype=float)
        normalized = raw / raw.sum()
        for row, betfair_probability in zip(race_rows, normalized):
            probabilities[race_box_key(row)] = 0.95 * float(row["market_implied_probability"]) + 0.05 * float(betfair_probability)
    return probabilities


def repo_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "head_tree": git("rev-parse", "HEAD^{tree}"),
        "index_tree": git("write-tree"),
        "dirty": bool(git("status", "--porcelain=v1", "--untracked-files=all")),
    }


def evaluate(out: Path) -> None:
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    matrix_path = out / "topology_matrix.jsonl"
    if sha256(matrix_path) != protocol["topology_matrix_sha256"]:
        raise SystemExit("sealed_topology_matrix_hash_mismatch")
    if protocol["status"] not in {"PROTOCOL_FROZEN_READY_TO_EVALUATE", "PROTOCOL_FROZEN_COVERAGE_BLOCKED"}:
        raise SystemExit("protocol_not_frozen")
    topology = load_jsonl(matrix_path)
    if any("label_is_winner" in row or "label_finish_position" in row for row in topology):
        raise SystemExit("labels_present_in_preoutcome_matrix")
    labels_source = load_jsonl(LATENT_MATRIX)
    labels = {exact_key(row): {"label_is_winner": int(row["label_is_winner"]), "label_finish_position": int(row["label_finish_position"])} for row in labels_source}
    if len(labels) != len(labels_source):
        raise SystemExit("duplicate_label_identity")
    evaluated = []
    for row in topology:
        if str(row["race_date"]) >= FORWARD_BOUNDARY:
            raise SystemExit("forward_outcome_boundary_breached")
        label = labels.get(exact_key(row))
        if label is None:
            raise SystemExit("approved_label_identity_missing")
        evaluated.append({**row, **label})
    race_labels: dict[str, int] = defaultdict(int)
    for row in evaluated:
        race_labels[str(row["race_id"])] += int(row["label_is_winner"])
    if any(value != 1 for value in race_labels.values()) or len(race_labels) != 989:
        raise SystemExit("winner_identity_invalid")

    group_reports = []
    passed = []
    adequately_covered = []
    for name in ALL_GROUPS:
        selected = [row for row in evaluated if bool(row["mechanisms"][name])]
        metrics = group_metrics(name, selected, "market_implied_probability")
        direction = DIRECTIONAL_GROUPS.get(name)
        adequate = metrics["N"] >= MIN_N and metrics["meeting_dates"] >= MIN_MEETING_DATES
        if direction and adequate:
            adequately_covered.append(name)
        fold_values = [fold["residual_mean"] for fold in metrics["folds"]]
        correct_folds = sum(
            value is not None and ((direction == "positive" and value > 0) or (direction == "negative" and value < 0))
            for value in fold_values
        ) if direction else None
        lower, upper = metrics["meeting_date_cluster_ci95"]
        ci_pass = bool(
            direction == "positive" and lower is not None and lower > 0
            or direction == "negative" and upper is not None and upper < 0
        )
        gate_pass = bool(direction and adequate and ci_pass and correct_folds >= 2)
        metrics.update({"expected_direction": direction or "diagnostic_only", "minimum_coverage_met": adequate, "folds_in_expected_direction": correct_folds, "decision_gate_pass": gate_pass})
        if gate_pass:
            passed.append(name)
        group_reports.append(metrics)

    betfair_probability = strict_betfair_probabilities(evaluated)
    betfair_reports = []
    for name in ALL_GROUPS:
        selected = []
        for row in evaluated:
            key = race_box_key(row)
            if row["mechanisms"][name] and key in betfair_probability:
                selected.append({**row, "p_95_5": betfair_probability[key]})
        diagnostic = group_metrics(name, selected, "p_95_5")
        diagnostic.pop("economic")
        diagnostic["probability_only"] = True
        betfair_reports.append(diagnostic)

    if protocol["blockers"] or not adequately_covered:
        decision = "DATA_COVERAGE_BLOCKED"
    elif passed:
        decision = "PACE_TOPOLOGY_SIGNAL_WORTH_MODELLING"
    else:
        decision = "NO_PACE_TOPOLOGY_SIGNAL"
    strongest_supported = (
        f"At least one frozen topology mechanism ({', '.join(passed)}) has a cluster-robust market residual in its predeclared direction and is stable in at least two folds."
        if passed
        else "No adequately covered, predeclared early-pace topology mechanism passed both the meeting-date clustered residual gate and the two-of-three-fold direction gate."
        if decision == "NO_PACE_TOPOLOGY_SIGNAL"
        else "The frozen pace state replays exactly, but the predeclared topology groups do not provide enough independent coverage for the signal decision."
    )
    unsupported = "This audit does not establish collision behaviour, rail or wide preference, causal race interference, an optimal odds band, a fitted winner model, forward profitability, or betting value."
    report = {
        "schema_version": "pace_topology_mechanism_report_v1",
        "decision": decision,
        "passed_mechanisms": passed,
        "adequately_covered_directional_mechanisms": adequately_covered,
        "protocol_sha256": sha256(protocol_path),
        "repo": repo_metadata(),
        "population": protocol["population"],
        "early_pace_state": protocol["early_pace_state"],
        "coverage": protocol["box_scratch_vacancy"],
        "thresholds": protocol["thresholds"],
        "mechanism_definitions": protocol["mechanisms"],
        "groups": group_reports,
        "betfair_95_5_probability_diagnostic": betfair_reports,
        "strongest_supported_claim": strongest_supported,
        "strongest_unsupported_claim": unsupported,
        "findings": {
            "BLOCKING": protocol["blockers"],
            "IMPORTANT": [
                "Only complete pace fields define topology; partial fields never manufacture a leader or pressure state.",
                "Absent boxes are treated as spatial gaps only because the frozen target matrix has no direct scratch or vacancy flag.",
                "Inside/outside imbalance groups are diagnostics and cannot trigger the decision because no direct preference or collision source is present.",
            ],
            "OPTIONAL": [],
        },
        "boundaries": {
            "outcomes_2026_08_18_or_later_opened": False,
            "october_outcomes_opened": False,
            "forward_cohorts_touched": False,
            "model_fit": False,
            "hyperparameter_search": False,
            "deployment": False,
            "promotion": False,
            "betting_recommendation": False,
        },
    }
    write_json(out / "report.json", report)
    lines = [
        "# Early-pace race-topology mechanism audit",
        "",
        f"Decision: `{decision}`.",
        "",
        f"The frozen causal pace projection replayed byte-for-byte for {protocol['population']['runner_rows']} runners in {protocol['population']['races']} races. Complete pace topology was available for {protocol['box_scratch_vacancy']['complete_pace_topology_races']} races.",
        "",
        "## Frozen definitions",
        "",
        f"- Large leader gap: at least `{protocol['thresholds']['large_leader_gap_q75']:.12f}` frozen pace-rating units.",
        f"- Comparable adjacent starter: absolute pace difference at most `{protocol['thresholds']['comparable_adjacent_difference_q25']:.12f}` in a present box +/-1.",
        f"- High adjacent pressure: a present box +/-1 starter is faster by at least `{protocol['thresholds']['high_adjacent_pressure_q75']:.12f}`.",
        f"- Large inside/outside imbalance: absolute boxes 1..4 minus boxes 5..8 mean pace at least `{protocol['thresholds']['large_inside_outside_imbalance_q75']:.12f}`; full boxes 1..8 only and diagnostic-only.",
        f"- Decision coverage: at least `{MIN_N}` runner rows across `{MIN_MEETING_DATES}` meeting dates.",
        "- Missing boxes are spatial gaps only. Scratch/vacancy cause, collision behaviour, and rail/wide preference are not inferred.",
        "",
        "## Sportsbet residuals and descriptive economics",
        "",
        "| Mechanism | N | Wins | Expected | Residual | 95% CI | Calibration | P&L | ROI | ROI 95% CI | Max DD | Fold residuals | Gate |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|",
    ]
    for item in group_reports:
        ci = item["meeting_date_cluster_ci95"]
        fold_text = ", ".join("NA" if fold["residual_mean"] is None else f"{fold['residual_mean']:.6f}" for fold in item["folds"])
        residual_text = "NA" if item["mean_observed_minus_market"] is None else f"{item['mean_observed_minus_market']:.6f}"
        lower_text = "NA" if ci[0] is None else f"{ci[0]:.6f}"
        upper_text = "NA" if ci[1] is None else f"{ci[1]:.6f}"
        calibration_text = "NA" if item["calibration_ratio"] is None else f"{item['calibration_ratio']:.6f}"
        roi_text = "NA" if item["economic"]["roi"] is None else f"{item['economic']['roi']:.6f}"
        roi_ci = item["economic"]["roi_meeting_date_cluster_ci95"]
        roi_lower_text = "NA" if roi_ci[0] is None else f"{roi_ci[0]:.6f}"
        roi_upper_text = "NA" if roi_ci[1] is None else f"{roi_ci[1]:.6f}"
        lines.append(
            f"| {item['group']} | {item['N']} | {item['wins']} | {item['summed_expected_wins']:.6f} | "
            f"{residual_text} | [{lower_text}, {upper_text}] | {calibration_text} | {item['economic']['pnl_units']:.2f} | "
            f"{roi_text} | [{roi_lower_text}, {roi_upper_text}] | {item['economic']['maximum_drawdown_units']:.2f} | "
            f"{fold_text} | {item['decision_gate_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Strict Betfair 95/5 probability diagnostic",
            "",
            "| Mechanism | N | Wins | Expected | Residual | 95% CI |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in betfair_reports:
        ci = item["meeting_date_cluster_ci95"]
        residual_text = "NA" if item["mean_observed_minus_market"] is None else f"{item['mean_observed_minus_market']:.6f}"
        lower_text = "NA" if ci[0] is None else f"{ci[0]:.6f}"
        upper_text = "NA" if ci[1] is None else f"{ci[1]:.6f}"
        lines.append(
            f"| {item['group']} | {item['N']} | {item['wins']} | {item['summed_expected_wins']:.6f} | {residual_text} | [{lower_text}, {upper_text}] |"
        )
    lines.extend(["", "Probability diagnostic only; no Betfair economics or selection was computed.", "", "## Claims", "", strongest_supported, "", unsupported, ""])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_checksums(out, ("pace_state_replay.jsonl", "topology_matrix.jsonl", "coverage_audit.json", "protocol.json", "report.json", "REPORT.md"), "SHA256SUMS")


def verify(out: Path) -> None:
    verify_inputs()
    for manifest in ("SEALED_SHA256SUMS", "SHA256SUMS"):
        for line in (out / manifest).read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if sha256(out / name) != digest:
                raise SystemExit(f"checksum_mismatch:{name}")
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    topology = load_jsonl(out / "topology_matrix.jsonl")
    if any("label_is_winner" in row or "label_finish_position" in row for row in topology):
        raise SystemExit("label_leakage_into_sealed_topology")
    if not protocol["early_pace_state"]["byte_for_byte_projection_match"]:
        raise SystemExit("early_pace_replay_not_exact")
    if protocol["early_pace_state"]["replay_projection_sha256"] != protocol["early_pace_state"]["sealed_projection_sha256"]:
        raise SystemExit("early_pace_projection_hash_disagreement")
    if report["decision"] not in {"PACE_TOPOLOGY_SIGNAL_WORTH_MODELLING", "NO_PACE_TOPOLOGY_SIGNAL", "DATA_COVERAGE_BLOCKED"}:
        raise SystemExit("invalid_terminal_decision")
    if any(str(row["race_date"]) >= FORWARD_BOUNDARY for row in topology):
        raise SystemExit("forward_boundary_breached")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.output)
    elif args.evaluate:
        evaluate(args.output)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
