#!/usr/bin/env python3
"""Freeze a report-only Sportsbet/Betfair scheduled-off consensus candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "sportsbet_betfair_consensus_freeze_v1"
EXPECTED_JOIN_SCHEMA_VERSION = "sportsbet_betfair_win_join_v1"
EXPECTED_JOINED_SHA256 = "86fabb05556160e555f076322eb8786b6166e369a6a8ec57d475c0e4a06e67f7"
EXPECTED_AUDIT_REPORT_SHA256 = "b573d7d8512ba9cfa11373425e465f0b1b87901764966cb845e0ecd8fc0c9491"
EXPECTED_SPORTSBET_SHA256 = "eb1783d4cc07e6980463a097c97fdac9f5370b08f493ca15addf768aa0b014b6"
EXPECTED_PROTOCOL_SHA256 = "2147a3181336326cb6df3222e9d39aba162db45be1cc614a7c59c0518432d2c8"
EXPECTED_BETFAIR_SOURCE_HASHES = {
    "ANZ_Greyhounds_2026_06.csv": "304085cec9dd7930c505f9b45d33835bdf1d2223dbb6f5c0723087c813114748",
    "ANZ_Greyhounds_2026_07.csv": "f150a95b7ebd323d7626bb3653cae04a9a3165d04ebbf8c4611c22d2a647944f",
}
FIT_START = date(2026, 6, 10)
FIT_END = date(2026, 6, 30)
VALIDATION_START = date(2026, 7, 1)
VALIDATION_END = date(2026, 7, 18)


class ContractError(RuntimeError):
    """Raised when a frozen provenance or modelling contract fails."""


@dataclass(frozen=True)
class Race:
    race_id: str
    race_date: str
    venue: str
    race_number: int
    scheduled_race_time_raw: str
    win_market_id: str
    split: str
    boxes: tuple[int, ...]
    selection_ids: tuple[str, ...]
    sportsbet_probabilities: tuple[float, ...]
    betfair_probabilities: tuple[float, ...]
    betfair_prices: tuple[float, ...]
    winner_index: int
    sportsbet_matrix_row_indices: tuple[int, ...]
    betfair_source_file: str
    betfair_source_file_sha256: str

    @property
    def cluster_key(self) -> str:
        return f"{self.race_date}|{self.venue}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def normalize(values: Sequence[float], label: str) -> tuple[float, ...]:
    if not values:
        raise ContractError(f"{label} is empty")
    parsed = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed):
        raise ContractError(f"{label} must contain finite positive values")
    total = math.fsum(parsed)
    if not math.isfinite(total) or total <= 0.0:
        raise ContractError(f"{label} has invalid total")
    return tuple(value / total for value in parsed)


def score_consensus(
    sportsbet_probabilities: Sequence[float],
    betfair_scheduled_off_prices: Sequence[float],
    betfair_weight: float,
) -> tuple[float, ...]:
    """Apply the frozen scorer contract without outcome or BSP inputs."""
    if len(sportsbet_probabilities) != len(betfair_scheduled_off_prices):
        raise ContractError("runner count mismatch between sources")
    weight = float(betfair_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ContractError("betfair_weight must be finite and within [0, 1]")
    sportsbet = normalize(sportsbet_probabilities, "sportsbet probabilities")
    prices = tuple(float(value) for value in betfair_scheduled_off_prices)
    if any(not math.isfinite(value) or value <= 1.0 for value in prices):
        raise ContractError("Betfair scheduled-off prices must be finite and > 1")
    betfair = normalize(tuple(1.0 / value for value in prices), "Betfair implied probabilities")
    combined = tuple((1.0 - weight) * sb + weight * bf for sb, bf in zip(sportsbet, betfair))
    return normalize(combined, "consensus probabilities")


def project_runner(row: dict[str, Any]) -> dict[str, Any]:
    """Whitelist development fields; BSP and actual-off values never enter scoring."""
    required = (
        "schema_version",
        "race_id",
        "race_date",
        "sportsbet_venue",
        "race_number",
        "scheduled_race_time_raw",
        "win_market_id",
        "box_number",
        "selection_id",
        "sportsbet_normalized_probability",
        "betfair_scheduled_off_back_price",
        "betfair_source_file",
        "betfair_source_file_sha256",
        "sportsbet_matrix_sha256",
        "sportsbet_matrix_row_index",
        "scheduled_clock_precedes_provider_actual_off_clock",
        "sportsbet_runner_name",
        "betfair_runner_name",
        "win_result",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ContractError(f"joined row missing required fields: {missing}")
    projected = {key: row[key] for key in required}
    if projected["schema_version"] != EXPECTED_JOIN_SCHEMA_VERSION:
        raise ContractError("joined row schema_version mismatch")
    return projected


def record_global_runner_identity(
    row: dict[str, Any],
    race_to_market: dict[str, str],
    market_to_race: dict[str, str],
    sportsbet_matrix_row_indices: set[int],
) -> None:
    """Fail closed on cross-race identity reuse and duplicate Sportsbet rows."""
    race_id = str(row["race_id"])
    win_market_id = str(row["win_market_id"])
    previous_market = race_to_market.get(race_id)
    if previous_market is not None and previous_market != win_market_id:
        raise ContractError(f"race_id maps to multiple win_market_id values: {race_id}")
    previous_race = market_to_race.get(win_market_id)
    if previous_race is not None and previous_race != race_id:
        raise ContractError(f"win_market_id maps to multiple race_id values: {win_market_id}")
    race_to_market[race_id] = win_market_id
    market_to_race[win_market_id] = race_id

    try:
        row_index = int(row["sportsbet_matrix_row_index"])
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid sportsbet_matrix_row_index") from exc
    if row_index in sportsbet_matrix_row_indices:
        raise ContractError(f"duplicate sportsbet_matrix_row_index: {row_index}")
    sportsbet_matrix_row_indices.add(row_index)


def _one_value(rows: Sequence[dict[str, Any]], key: str) -> Any:
    values = {row[key] for row in rows}
    if len(values) != 1:
        raise ContractError(f"race has inconsistent {key}: {sorted(map(str, values))}")
    return next(iter(values))


def _split_for_date(value: date) -> str:
    if FIT_START <= value <= FIT_END:
        return "fit"
    if VALIDATION_START <= value <= VALIDATION_END:
        return "validation"
    raise ContractError(f"joined source contains out-of-protocol race date {value.isoformat()}")


def validate_betfair_source_month(source_file: str, race_date: date) -> None:
    expected_source_file = f"ANZ_Greyhounds_{race_date.year}_{race_date.month:02d}.csv"
    if source_file != expected_source_file:
        raise ContractError("Betfair source file month mismatch")


def normalized_name(value: Any) -> str:
    """Mirror the audited corroboration rule; never use names as identity."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def load_races(joined_path: Path) -> tuple[list[Race], dict[str, Any]]:
    if sha256_file(joined_path) != EXPECTED_JOINED_SHA256:
        raise ContractError("joined surface SHA-256 mismatch")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    race_to_market: dict[str, str] = {}
    market_to_race: dict[str, str] = {}
    sportsbet_matrix_row_indices: set[int] = set()
    runner_rows = 0
    with joined_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ContractError(f"blank joined row at line {line_number}")
            row = project_runner(json.loads(raw))
            record_global_runner_identity(
                row,
                race_to_market,
                market_to_race,
                sportsbet_matrix_row_indices,
            )
            grouped[(str(row["race_id"]), str(row["win_market_id"]))].append(row)
            runner_rows += 1
    if runner_rows != 7142 or len(grouped) != 1008:
        raise ContractError(f"joined surface count mismatch: {len(grouped)} races/{runner_rows} runners")

    races: list[Race] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["box_number"]))
        race_date = str(_one_value(rows, "race_date"))
        parsed_date = date.fromisoformat(race_date)
        split = _split_for_date(parsed_date)
        boxes = tuple(int(row["box_number"]) for row in rows)
        if len(rows) < 2 or len(set(boxes)) != len(boxes) or any(box < 1 or box > 8 for box in boxes):
            raise ContractError(f"invalid complete box set for {key}")
        selection_ids = tuple(str(row["selection_id"]) for row in rows)
        if any(not value for value in selection_ids) or len(set(selection_ids)) != len(selection_ids):
            raise ContractError(f"invalid Betfair selection IDs for {key}")
        if any(row["sportsbet_matrix_sha256"] != EXPECTED_SPORTSBET_SHA256 for row in rows):
            raise ContractError(f"Sportsbet source hash drift in {key}")
        source_file = str(_one_value(rows, "betfair_source_file"))
        source_hash = str(_one_value(rows, "betfair_source_file_sha256"))
        if EXPECTED_BETFAIR_SOURCE_HASHES.get(source_file) != source_hash:
            raise ContractError(f"Betfair source hash drift in {key}")
        validate_betfair_source_month(source_file, parsed_date)
        if any(row["scheduled_clock_precedes_provider_actual_off_clock"] is not True for row in rows):
            raise ContractError(f"scheduled-off timing check failed in {key}")
        if any(
            normalized_name(row["sportsbet_runner_name"])
            != normalized_name(row["betfair_runner_name"])
            for row in rows
        ):
            raise ContractError(f"runner-name corroboration mismatch in {key}")
        sportsbet_raw = tuple(float(row["sportsbet_normalized_probability"]) for row in rows)
        if not math.isclose(math.fsum(sportsbet_raw), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ContractError(f"Sportsbet probabilities do not sum to one in {key}")
        sportsbet = normalize(sportsbet_raw, "Sportsbet probabilities")
        prices = tuple(float(row["betfair_scheduled_off_back_price"]) for row in rows)
        if any(not math.isfinite(value) or value <= 1.0 for value in prices):
            raise ContractError(f"invalid Betfair scheduled-off price in {key}")
        betfair = normalize(tuple(1.0 / value for value in prices), "Betfair implied probabilities")
        results = tuple(str(row["win_result"]) for row in rows)
        if any(value not in {"WINNER", "LOSER"} for value in results) or results.count("WINNER") != 1:
            raise ContractError(f"race does not have exactly one winner in {key}")
        races.append(
            Race(
                race_id=str(_one_value(rows, "race_id")),
                race_date=race_date,
                venue=str(_one_value(rows, "sportsbet_venue")),
                race_number=int(_one_value(rows, "race_number")),
                scheduled_race_time_raw=str(_one_value(rows, "scheduled_race_time_raw")),
                win_market_id=str(_one_value(rows, "win_market_id")),
                split=split,
                boxes=boxes,
                selection_ids=selection_ids,
                sportsbet_probabilities=sportsbet,
                betfair_probabilities=betfair,
                betfair_prices=prices,
                winner_index=results.index("WINNER"),
                sportsbet_matrix_row_indices=tuple(int(row["sportsbet_matrix_row_index"]) for row in rows),
                betfair_source_file=source_file,
                betfair_source_file_sha256=source_hash,
            )
        )
    counts = {
        "races": len(races),
        "runner_rows": sum(len(race.boxes) for race in races),
        "fit_races": sum(race.split == "fit" for race in races),
        "fit_runner_rows": sum(len(race.boxes) for race in races if race.split == "fit"),
        "validation_races": sum(race.split == "validation" for race in races),
        "validation_runner_rows": sum(len(race.boxes) for race in races if race.split == "validation"),
        "meeting_date_clusters": len({race.cluster_key for race in races}),
        "fit_meeting_date_clusters": len({race.cluster_key for race in races if race.split == "fit"}),
        "validation_meeting_date_clusters": len(
            {race.cluster_key for race in races if race.split == "validation"}
        ),
    }
    return races, counts


def probabilities_for(race: Race, model: str, weight: float | None = None) -> tuple[float, ...]:
    if model == "sportsbet":
        return race.sportsbet_probabilities
    if model == "betfair_only":
        return race.betfair_probabilities
    if model == "consensus" and weight is not None:
        return score_consensus(race.sportsbet_probabilities, race.betfair_prices, weight)
    raise ContractError(f"unknown model {model}")


def race_metric_values(race: Race, probabilities: Sequence[float]) -> dict[str, float]:
    winner_probability = float(probabilities[race.winner_index])
    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], race.boxes[index]))
    winner_rank = order.index(race.winner_index) + 1
    top_index = order[0]
    brier = math.fsum(
        (probability - (1.0 if index == race.winner_index else 0.0)) ** 2
        for index, probability in enumerate(probabilities)
    )
    return {
        "log_loss": -math.log(winner_probability),
        "brier": brier,
        "top1": 1.0 if winner_rank <= 1 else 0.0,
        "top2": 1.0 if winner_rank <= 2 else 0.0,
        "top3": 1.0 if winner_rank <= 3 else 0.0,
        "winner_rank": float(winner_rank),
        "winner_probability": winner_probability,
        "top_confidence": float(probabilities[top_index]),
        "top_correct": 1.0 if top_index == race.winner_index else 0.0,
    }


def evaluate(races: Sequence[Race], model: str, weight: float | None = None) -> dict[str, Any]:
    if not races:
        raise ContractError("cannot evaluate an empty race population")
    rows: list[dict[str, float]] = []
    ranks: list[float] = []
    calibration_bins = [dict(count=0, confidence_sum=0.0, accuracy_sum=0.0) for _ in range(10)]
    for race in races:
        values = race_metric_values(race, probabilities_for(race, model, weight))
        rows.append(values)
        ranks.append(values["winner_rank"])
        bin_index = min(int(values["top_confidence"] * 10.0), 9)
        calibration_bins[bin_index]["count"] += 1
        calibration_bins[bin_index]["confidence_sum"] += values["top_confidence"]
        calibration_bins[bin_index]["accuracy_sum"] += values["top_correct"]
    count = len(rows)
    mean = lambda key: math.fsum(row[key] for row in rows) / count
    ranks_sorted = sorted(ranks)
    midpoint = count // 2
    median_rank = (
        ranks_sorted[midpoint]
        if count % 2
        else (ranks_sorted[midpoint - 1] + ranks_sorted[midpoint]) / 2.0
    )
    calibration: list[dict[str, Any]] = []
    ece = 0.0
    for index, bucket in enumerate(calibration_bins):
        bucket_count = int(bucket["count"])
        if bucket_count:
            mean_confidence = bucket["confidence_sum"] / bucket_count
            accuracy = bucket["accuracy_sum"] / bucket_count
            ece += (bucket_count / count) * abs(mean_confidence - accuracy)
        else:
            mean_confidence = None
            accuracy = None
        calibration.append(
            {
                "bin_lower_inclusive": index / 10.0,
                "bin_upper_exclusive_except_last": (index + 1) / 10.0,
                "races": bucket_count,
                "mean_top_confidence": mean_confidence,
                "top1_accuracy": accuracy,
            }
        )
    top_confidence = mean("top_confidence")
    top1 = mean("top1")
    return {
        "races": count,
        "log_loss": mean("log_loss"),
        "brier": mean("brier"),
        "top1_accuracy": top1,
        "top2_accuracy": mean("top2"),
        "top3_accuracy": mean("top3"),
        "mean_winner_rank": mean("winner_rank"),
        "median_winner_rank": median_rank,
        "mean_winner_probability": mean("winner_probability"),
        "mean_top_confidence": top_confidence,
        "top_confidence_minus_accuracy": top_confidence - top1,
        "top_label_ece_10_bin": ece,
        "top_label_calibration_bins": calibration,
    }


BOOTSTRAP_METRICS = (
    "log_loss",
    "brier",
    "top1",
    "top2",
    "top3",
    "winner_rank",
    "winner_probability",
    "top_confidence",
)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def bootstrap_delta(
    races: Sequence[Race],
    alternative_model: str,
    weight: float | None,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    cluster_rows: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = defaultdict(list)
    for race in races:
        baseline = race_metric_values(race, probabilities_for(race, "sportsbet"))
        alternative = race_metric_values(race, probabilities_for(race, alternative_model, weight))
        cluster_rows[race.cluster_key].append((baseline, alternative))
    cluster_keys = sorted(cluster_rows)
    cluster_sums: list[tuple[int, tuple[float, ...]]] = []
    for key in cluster_keys:
        pairs = cluster_rows[key]
        sums = tuple(
            math.fsum(alternative[metric] - baseline[metric] for baseline, alternative in pairs)
            for metric in BOOTSTRAP_METRICS
        )
        cluster_sums.append((len(pairs), sums))
    rng = random.Random(seed)
    distributions = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(replicates):
        sampled = [cluster_sums[rng.randrange(len(cluster_sums))] for _ in cluster_sums]
        denominator = sum(count for count, _ in sampled)
        for metric_index, metric in enumerate(BOOTSTRAP_METRICS):
            value = math.fsum(sums[metric_index] for _, sums in sampled) / denominator
            distributions[metric].append(value)
    names = {
        "top1": "top1_accuracy",
        "top2": "top2_accuracy",
        "top3": "top3_accuracy",
        "winner_rank": "mean_winner_rank",
    }
    intervals = {}
    for metric, values in distributions.items():
        intervals[names.get(metric, metric)] = {
            "lower_95": _quantile(values, 0.025),
            "upper_95": _quantile(values, 0.975),
        }
    return {
        "cluster_count": len(cluster_keys),
        "cluster_key": "race_date|sportsbet_venue",
        "replicates": replicates,
        "seed": seed,
        "delta_direction": "alternative_minus_sportsbet; negative improves log_loss/brier/rank, positive improves accuracy",
        "intervals": intervals,
    }


def select_weight(fit_races: Sequence[Race], selection_weights: Iterable[float]) -> tuple[float, list[dict[str, float]]]:
    diagnostics = []
    for weight in selection_weights:
        metrics = evaluate(fit_races, "consensus", float(weight))
        diagnostics.append(
            {
                "betfair_weight": float(weight),
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
            }
        )
    selected = min(diagnostics, key=lambda row: (round(row["log_loss"], 15), row["betfair_weight"]))
    return selected["betfair_weight"], diagnostics


def validate_source_contract(audit_report_path: Path, sportsbet_matrix_path: Path) -> dict[str, Any]:
    if sha256_file(audit_report_path) != EXPECTED_AUDIT_REPORT_SHA256:
        raise ContractError("Betfair audit report SHA-256 mismatch")
    if sha256_file(sportsbet_matrix_path) != EXPECTED_SPORTSBET_SHA256:
        raise ContractError("corrected Sportsbet WIN matrix SHA-256 mismatch")
    audit = json.loads(audit_report_path.read_text(encoding="utf-8"))
    overlap = audit.get("overlap", {})
    required = {
        "terminal_state": audit.get("terminal_state") == "BETFAIR_HISTORICAL_SURFACE_PARTIAL",
        "matched_races": overlap.get("matched_races") == 1008,
        "matched_runner_rows": overlap.get("matched_runner_rows") == 7142,
        "ambiguous_races_zero": overlap.get("ambiguous_races") == 0,
        "scheduled_prices_complete": overlap.get("matched_rows_missing_scheduled_off_back") == 0,
        "result_conflicts_zero": overlap.get("result_corroboration_conflicts") == 0,
        "runner_name_conflicts_zero": overlap.get("runner_name_corroboration_conflicts") == 0,
        "reserve_tab_races_zero": overlap.get("matched_races_with_tab_9_or_10") == 0,
        "scheduled_before_actual_check": overlap.get(
            "matched_races_not_proven_scheduled_clock_before_provider_actual_off_clock"
        )
        == 0,
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise ContractError(f"audited surface contract failed: {failed}")
    return required


def population_record(race: Race) -> dict[str, Any]:
    return {
        "schema_version": "sportsbet_betfair_consensus_population_v1",
        "split": race.split,
        "race_id": race.race_id,
        "race_date": race.race_date,
        "sportsbet_venue": race.venue,
        "race_number": race.race_number,
        "scheduled_race_time_raw": race.scheduled_race_time_raw,
        "win_market_id": race.win_market_id,
        "runner_count": len(race.boxes),
        "boxes": list(race.boxes),
        "selection_ids": list(race.selection_ids),
        "sportsbet_matrix_row_indices": list(race.sportsbet_matrix_row_indices),
        "betfair_source_file": race.betfair_source_file,
        "betfair_source_file_sha256": race.betfair_source_file_sha256,
        "meeting_date_cluster": race.cluster_key,
    }


def build_report(
    races: Sequence[Race],
    counts: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    scorer_sha256: str,
    source_checks: dict[str, bool],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fit = [race for race in races if race.split == "fit"]
    validation = [race for race in races if race.split == "validation"]
    selection_weights = protocol["candidate"]["selection_pool_betfair_weights"]
    selected_weight, fit_grid = select_weight(fit, selection_weights)

    fit_metrics = {
        "sportsbet": evaluate(fit, "sportsbet"),
        "betfair_only": evaluate(fit, "betfair_only"),
        "consensus": evaluate(fit, "consensus", selected_weight),
    }
    validation_metrics = {
        "sportsbet": evaluate(validation, "sportsbet"),
        "betfair_only": evaluate(validation, "betfair_only"),
        "consensus": evaluate(validation, "consensus", selected_weight),
    }
    fit_delta = fit_metrics["consensus"]["log_loss"] - fit_metrics["sportsbet"]["log_loss"]
    validation_delta = (
        validation_metrics["consensus"]["log_loss"] - validation_metrics["sportsbet"]["log_loss"]
    )
    freeze_checks = {
        "selected_weight_is_interior": 0.0 < selected_weight < 1.0,
        "fit_log_loss_improves_sportsbet": fit_delta < 0.0,
        "validation_log_loss_improves_sportsbet": validation_delta < 0.0,
        "provenance_checks_pass": all(source_checks.values()),
        "no_future_rows_loaded_or_scored": True,
        "BSP_not_used_as_predictor": True,
        "actual_off_not_used_as_predictor": True,
    }
    frozen = all(freeze_checks.values())
    terminal_state = "CONSENSUS_CANDIDATE_FROZEN" if frozen else "CONSENSUS_CANDIDATE_NOT_FROZEN"
    readiness = "READY_TO_FREEZE" if frozen else "NOT_READY_TO_FREEZE"
    paired_deltas = {}
    for alternative in ("betfair_only", "consensus"):
        paired_deltas[alternative] = {
            key: validation_metrics[alternative][key] - validation_metrics["sportsbet"][key]
            for key in (
                "log_loss",
                "brier",
                "top1_accuracy",
                "top2_accuracy",
                "top3_accuracy",
                "mean_winner_rank",
                "mean_winner_probability",
                "mean_top_confidence",
                "top_confidence_minus_accuracy",
                "top_label_ece_10_bin",
            )
        }
    bootstrap = {
        "betfair_only_vs_sportsbet": bootstrap_delta(
            validation,
            "betfair_only",
            None,
            int(protocol["bootstrap"]["replicates"]),
            int(protocol["bootstrap"]["seed"]),
        ),
        "consensus_vs_sportsbet": bootstrap_delta(
            validation,
            "consensus",
            selected_weight,
            int(protocol["bootstrap"]["replicates"]),
            int(protocol["bootstrap"]["seed"]),
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "terminal_state": terminal_state,
        "prospective_test_readiness": readiness,
        "analysis_as_of_date": "2026-08-17",
        "inputs": {
            "joined_surface_sha256": EXPECTED_JOINED_SHA256,
            "betfair_audit_report_sha256": EXPECTED_AUDIT_REPORT_SHA256,
            "corrected_sportsbet_win_matrix_sha256": EXPECTED_SPORTSBET_SHA256,
            "Betfair_source_hashes": EXPECTED_BETFAIR_SOURCE_HASHES,
        },
        "code_and_protocol": {
            "scorer_sha256": scorer_sha256,
            "protocol_sha256": protocol_sha256,
        },
        "population": counts,
        "selection": {
            "selected_betfair_weight": selected_weight,
            "selected_sportsbet_weight": 1.0 - selected_weight,
            "selection_metric": "fit mean multiclass race log loss",
            "fit_grid": fit_grid,
            "fit_log_loss_delta_consensus_minus_sportsbet": fit_delta,
            "validation_log_loss_delta_consensus_minus_sportsbet": validation_delta,
            "freeze_checks": freeze_checks,
        },
        "fit_metrics": fit_metrics,
        "validation": {
            "metrics": validation_metrics,
            "paired_deltas_alternative_minus_sportsbet": paired_deltas,
            "meeting_date_cluster_bootstrap": bootstrap,
        },
        "provenance_checks": source_checks,
        "leakage_controls": {
            "predictor_fields": [
                "sportsbet_normalized_probability",
                "betfair_scheduled_off_back_price",
            ],
            "BSP_as_predictor": False,
            "actual_off_as_predictor": False,
            "name_only_identity": False,
            "future_rows_loaded": 0,
            "future_rows_scored": 0,
            "August_2026_rows_in_joined_surface": 0,
            "post_hoc_exclusions": 0,
        },
        "future_protocol": {
            "start_date_inclusive": "2026-08-18",
            "end_date_inclusive": "2026-09-30",
            "population_rows_materialized_now": 0,
            "outcomes_inspected_now": 0,
            "interim_scoring": "FORBIDDEN",
            "analysis_count": 1,
        },
        "findings": {
            "BLOCKING": [] if frozen else ["predeclared freeze gate did not pass"],
            "IMPORTANT": [
                "validation evidence is development screening only, not prospective confirmation",
                "Betfair scheduled-off quote is not proof of executable liquidity",
            ],
            "OPTIONAL": ["seek Betfair clarification of scheduled-off quote sampling semantics"],
        },
        "claims": {
            "supported": [
                "descriptive performance on the frozen June fit and July validation intersection",
                "deterministic replay of the frozen convex scorer",
                "untouched fixed-calendar prospective eligibility protocol",
            ],
            "unsupported": [
                "prospective confirmation",
                "profitability or betting value",
                "promotion, deployment, live scoring, or market-wide generalization",
            ],
        },
    }
    frozen_rule = {
        "schema_version": "sportsbet_betfair_frozen_consensus_rule_v1",
        "terminal_state": terminal_state,
        "frozen": frozen,
        "selected_betfair_weight": selected_weight if frozen else None,
        "selected_sportsbet_weight": (1.0 - selected_weight) if frozen else None,
        "screened_weight_if_not_frozen": None if frozen else selected_weight,
        "formula": protocol["candidate"]["formula"],
        "normalization": protocol["candidate"]["normalization"],
        "tie_break": "probability_descending_then_box_ascending",
        "source_hashes": report["inputs"],
        "protocol_sha256": protocol_sha256,
        "scorer_sha256": scorer_sha256,
        "freeze_checks": freeze_checks,
        "claim_boundary": "development-screening artifact only; no promotion, deployment, EV, betting, or future scoring",
    }
    first_validation = validation[0]
    replay_fixture = {
        "schema_version": "sportsbet_betfair_consensus_replay_fixture_v1",
        "race_id": first_validation.race_id,
        "boxes": list(first_validation.boxes),
        "sportsbet_probabilities": list(first_validation.sportsbet_probabilities),
        "betfair_scheduled_off_prices": list(first_validation.betfair_prices),
        "betfair_weight": selected_weight,
        "expected_consensus_probabilities": list(
            score_consensus(
                first_validation.sportsbet_probabilities,
                first_validation.betfair_prices,
                selected_weight,
            )
        ),
    }
    return report, frozen_rule, replay_fixture


def report_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Sportsbet Betfair consensus freeze report",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "terminal_state",
            "prospective_test_readiness",
            "analysis_as_of_date",
            "inputs",
            "code_and_protocol",
            "population",
            "selection",
            "fit_metrics",
            "validation",
            "provenance_checks",
            "leakage_controls",
            "future_protocol",
            "findings",
            "claims",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "terminal_state": {
                "enum": ["CONSENSUS_CANDIDATE_FROZEN", "CONSENSUS_CANDIDATE_NOT_FROZEN"]
            },
            "prospective_test_readiness": {"enum": ["READY_TO_FREEZE", "NOT_READY_TO_FREEZE"]},
            "analysis_as_of_date": {"type": "string"},
            "inputs": {"type": "object"},
            "code_and_protocol": {"type": "object"},
            "population": {"type": "object"},
            "selection": {"type": "object"},
            "fit_metrics": {"type": "object"},
            "validation": {"type": "object"},
            "provenance_checks": {"type": "object"},
            "leakage_controls": {"type": "object"},
            "future_protocol": {"type": "object"},
            "findings": {"type": "object"},
            "claims": {"type": "object"},
        },
    }


def report_markdown(report: dict[str, Any]) -> str:
    selection = report["selection"]
    validation = report["validation"]
    metrics = validation["metrics"]
    delta = validation["paired_deltas_alternative_minus_sportsbet"]["consensus"]
    ci = validation["meeting_date_cluster_bootstrap"]["consensus_vs_sportsbet"]["intervals"]
    population = report["population"]
    lines = [
        "# Sportsbet + Betfair scheduled-off consensus freeze",
        "",
        f"Terminal state: `{report['terminal_state']}`.",
        f"Prospective test: `{report['prospective_test_readiness']}`.",
        "",
        "## Population",
        "",
        f"- Strict joined surface: {population['races']} races / {population['runner_rows']} runners.",
        f"- Fit: {population['fit_races']} races / {population['fit_runner_rows']} runners.",
        f"- Validation: {population['validation_races']} races / {population['validation_runner_rows']} runners.",
        "- August 2026 rows read or scored: 0.",
        "",
        "## Frozen rule",
        "",
        f"- Betfair weight: {selection['selected_betfair_weight']:.2f}.",
        f"- Sportsbet weight: {selection['selected_sportsbet_weight']:.2f}.",
        "- Both sources are normalized within race before the convex combination.",
        "- Selection used fit log loss only; validation screened that one weight once.",
        "",
        "## Validation",
        "",
        f"- Sportsbet log loss: {metrics['sportsbet']['log_loss']:.9f}.",
        f"- Betfair-only log loss: {metrics['betfair_only']['log_loss']:.9f}.",
        f"- Consensus log loss: {metrics['consensus']['log_loss']:.9f}.",
        f"- Consensus - Sportsbet log-loss delta: {delta['log_loss']:.9f} "
        f"(cluster bootstrap 95% CI {ci['log_loss']['lower_95']:.9f} to {ci['log_loss']['upper_95']:.9f}).",
        f"- Consensus - Sportsbet Brier delta: {delta['brier']:.9f} "
        f"(95% CI {ci['brier']['lower_95']:.9f} to {ci['brier']['upper_95']:.9f}).",
        f"- Consensus top-1/top-2/top-3: {metrics['consensus']['top1_accuracy']:.6f} / "
        f"{metrics['consensus']['top2_accuracy']:.6f} / {metrics['consensus']['top3_accuracy']:.6f}.",
        f"- Consensus mean winner rank: {metrics['consensus']['mean_winner_rank']:.6f}.",
        "",
        "## Claim boundary",
        "",
        "This is development screening evidence only. It does not confirm prospective edge, profitability,",
        "betting value, promotion readiness, deployment readiness, or live scoring readiness. The fixed",
        "2026-08-18 through 2026-09-30 cohort has not been ingested, labelled, or scored.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    races: Sequence[Race],
    report: dict[str, Any],
    frozen_rule: dict[str, Any],
    replay_fixture: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    scorer_sha256: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    population_path = output_dir / "development_population.jsonl"
    with population_path.open("w", encoding="utf-8", newline="\n") as handle:
        for race in races:
            handle.write(json.dumps(population_record(race), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    write_json(output_dir / "report.json", report)
    write_json(output_dir / "report.schema.json", report_schema())
    write_json(output_dir / "frozen_consensus_rule.json", frozen_rule)
    write_json(output_dir / "replay_fixture.json", replay_fixture)
    write_json(
        output_dir / "input_manifest.json",
        {
            "schema_version": "sportsbet_betfair_consensus_input_manifest_v1",
            "joined_surface_sha256": EXPECTED_JOINED_SHA256,
            "betfair_audit_report_sha256": EXPECTED_AUDIT_REPORT_SHA256,
            "corrected_sportsbet_win_matrix_sha256": EXPECTED_SPORTSBET_SHA256,
            "Betfair_source_hashes": EXPECTED_BETFAIR_SOURCE_HASHES,
            "protocol_sha256": protocol_sha256,
            "scorer_sha256": scorer_sha256,
            "eligible_population_sha256": sha256_file(population_path),
        },
    )
    write_json(
        output_dir / "future_eligibility_protocol.json",
        {
            "schema_version": "sportsbet_betfair_future_eligibility_protocol_v1",
            "terminal_state": report["prospective_test_readiness"],
            "window": protocol["future_evaluation"],
            "eligibility": protocol["eligibility"],
            "frozen_rule_sha256": sha256_file(output_dir / "frozen_consensus_rule.json"),
            "protocol_sha256": protocol_sha256,
            "scorer_sha256": scorer_sha256,
            "population_rows_materialized": 0,
            "outcome_rows_inspected": 0,
            "scored_races": 0,
            "reopen_rule": "input/hash/timing/leakage ambiguity stops; no post-hoc exclusions or BSP substitution",
        },
    )
    (output_dir / "REPORT.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")
    members = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in members]
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-surface", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--sportsbet-matrix", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_sha256 = sha256_file(args.protocol)
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ContractError("protocol SHA-256 mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_checks = validate_source_contract(args.audit_report, args.sportsbet_matrix)
    races, counts = load_races(args.joined_surface)
    scorer_sha256 = sha256_file(Path(__file__).resolve())
    report, frozen_rule, replay_fixture = build_report(
        races,
        counts,
        protocol,
        protocol_sha256,
        scorer_sha256,
        source_checks,
    )
    write_outputs(
        args.output_dir,
        races,
        report,
        frozen_rule,
        replay_fixture,
        protocol,
        protocol_sha256,
        scorer_sha256,
    )
    print(json.dumps({
        "terminal_state": report["terminal_state"],
        "prospective_test_readiness": report["prospective_test_readiness"],
        "selected_betfair_weight": report["selection"]["selected_betfair_weight"],
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
