#!/usr/bin/env python3
"""Deterministic one-shot scorer for the frozen forward-overround successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

EXPECTED_PROTOCOL_SHA256 = "4978163d1dd9c0e4ced5eb1d4cb9425d3994379d8c617fb3306a489b838073be"
ASSET_HASHES = {
    "final_model.json": "c81b4b3047b7840ba31269504e0c5ceb6c54d742a82a4e01cca52b11fdaa471e",
    "preprocessing.json": "ad85722337d80360e1745f75fe57ff6b3fbd1e80deac57af318c898372b01998",
    "protocol.json": "2b20704e41574d5557eb1d6381bc314212b382a195ca5bffea95b832d4a5fb4a",
    "scorer_contract.json": "c119feea4f67baad73dc9e23ff7f98d755a34054c32a0a98dea4f44cdafa2576",
}
MELBOURNE = ZoneInfo("Australia/Melbourne")


class FinalizationError(ValueError):
    """Raised when exact fixed-N evidence cannot be scored safely."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, payload: Mapping[str, Any]) -> str:
    raw = canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise FinalizationError(f"write_once_conflict:{path}") from None
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    return sha256_bytes(raw)


def load_frozen_assets(protocol_path: Path, asset_dir: Path) -> tuple[dict[str, Any], ...]:
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise FinalizationError("successor_protocol_hash_drift")
    protocol = json.loads(protocol_path.read_bytes())
    loaded: list[dict[str, Any]] = []
    for name, expected in ASSET_HASHES.items():
        path = asset_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise FinalizationError(f"frozen_asset_hash_drift:{name}")
        loaded.append(json.loads(path.read_bytes()))
    model, preprocessing, development_protocol, scorer = loaded
    if model.get("model") != "linear_overround_allocation":
        raise FinalizationError("frozen_model_identity_mismatch")
    if scorer.get("model") != "linear_overround_allocation":
        raise FinalizationError("frozen_scorer_identity_mismatch")
    if model.get("feature_names") != preprocessing.get("feature_names"):
        raise FinalizationError("frozen_feature_order_mismatch")
    if protocol["model"]["hashes"]["protocol.json"] != ASSET_HASHES["protocol.json"]:
        raise FinalizationError("successor_development_protocol_binding_mismatch")
    return protocol, model, preprocessing, development_protocol, scorer


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizationError(f"invalid_number:{field}")
    number = float(value)
    if not math.isfinite(number):
        raise FinalizationError(f"non_finite_number:{field}")
    return number


def runner_set_payload(runners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for runner in runners:
        box = runner.get("box_number")
        dog = runner.get("dog_name")
        if not isinstance(box, int) or isinstance(box, bool) or box <= 0:
            raise FinalizationError("invalid_box_number")
        if not isinstance(dog, str) or not dog or dog != dog.strip():
            raise FinalizationError("invalid_exact_dog_name")
        identities.append({"box_number": box, "dog_name": dog})
    identities.sort(key=lambda item: item["box_number"])
    if len(identities) < 2 or len({item["box_number"] for item in identities}) != len(identities):
        raise FinalizationError("incomplete_or_duplicate_runner_set")
    if len({item["dog_name"] for item in identities}) != len(identities):
        raise FinalizationError("duplicate_exact_dog_name")
    return identities


def runner_set_sha256(runners: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes(runner_set_payload(runners)))


def score_race(
    runners: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
) -> dict[int, dict[str, float]]:
    runner_set_payload(runners)
    ordered = sorted(
        runners,
        key=lambda row: (-1.0 / _finite_number(row.get("decimal_win_odds"), "decimal_win_odds"), row["box_number"]),
    )
    odds = np.asarray(
        [_finite_number(row.get("decimal_win_odds"), "decimal_win_odds") for row in ordered],
        dtype=float,
    )
    if np.any(odds <= 1.0):
        raise FinalizationError("invalid_decimal_win_odds")
    raw = 1.0 / odds
    baseline = raw / raw.sum()
    size = len(raw)
    rank = np.arange(size, dtype=float) / max(size - 1, 1)
    favourite = np.zeros(size, dtype=float)
    favourite[0] = 1.0
    shorter = np.zeros(size, dtype=float)
    shorter[1:] = raw[:-1] - raw[1:]
    longer = np.zeros(size, dtype=float)
    longer[:-1] = raw[:-1] - raw[1:]
    favourite_gap = raw[0] - raw
    overround = float(raw.sum())
    concentration = float(np.square(baseline).sum())
    entropy = float(
        -(baseline * np.log(np.clip(baseline, 1e-15, 1.0))).sum() / math.log(size)
    )
    values = np.column_stack(
        (
            raw,
            baseline,
            rank,
            favourite,
            shorter,
            longer,
            favourite_gap,
            overround * baseline,
            overround * rank,
            float(size) * baseline,
            float(size) * rank,
            concentration * baseline,
            concentration * rank,
            entropy * baseline,
            entropy * rank,
        )
    )
    mean = np.asarray(preprocessing["mean"], dtype=float)
    scale = np.asarray(preprocessing["scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if values.shape[1] != len(mean) or len(mean) != len(scale) or len(scale) != len(coefficients):
        raise FinalizationError("frozen_model_dimension_mismatch")
    if np.any(scale < _finite_number(preprocessing["scale_floor"], "scale_floor")):
        raise FinalizationError("frozen_preprocessing_scale_invalid")
    logits = np.log(baseline) + ((values - mean) / scale) @ coefficients
    logits -= float(np.max(logits))
    candidate = np.exp(logits)
    candidate /= candidate.sum()
    if (
        not np.all(np.isfinite(candidate))
        or np.any(candidate <= 0.0)
        or not math.isclose(float(candidate.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise FinalizationError("candidate_probability_contract_failed")
    return {
        int(row["box_number"]): {
            "baseline_probability": float(baseline[index]),
            "candidate_probability": float(candidate[index]),
        }
        for index, row in enumerate(ordered)
    }


def _calibration(probability: np.ndarray, outcome: np.ndarray, bands: Sequence[Sequence[float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    for index, pair in enumerate(bands):
        lower, upper = map(float, pair)
        mask = (probability >= lower) & ((probability <= upper) if index == len(bands) - 1 else (probability < upper))
        count = int(mask.sum())
        mean_probability = float(probability[mask].mean()) if count else None
        observed = float(outcome[mask].mean()) if count else None
        gap = None if count == 0 else observed - mean_probability
        if gap is not None:
            weighted_gap += count * abs(gap)
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "runner_count": count,
                "mean_probability": mean_probability,
                "observed_win_rate": observed,
                "observed_minus_probability": gap,
            }
        )
    return {"bands": rows, "ece": weighted_gap / len(probability)}


def _metric_summary(races: Sequence[dict[str, Any]], probability_key: str, bands: Sequence[Sequence[float]]) -> dict[str, Any]:
    losses: list[float] = []
    briers: list[float] = []
    all_probability: list[float] = []
    all_outcome: list[int] = []
    winner_ranks: list[int] = []
    top_1 = 0
    for race in races:
        rows = race["runners"]
        probabilities = np.asarray([row[probability_key] for row in rows], dtype=float)
        winner_index = next(index for index, row in enumerate(rows) if row["is_winner"])
        outcome = np.zeros(len(rows), dtype=float)
        outcome[winner_index] = 1.0
        losses.append(-math.log(max(float(probabilities[winner_index]), 1e-15)))
        briers.append(float(np.square(probabilities - outcome).sum()))
        order = sorted(
            range(len(rows)),
            key=lambda index: (-float(probabilities[index]), int(rows[index]["box_number"])),
        )
        rank = order.index(winner_index) + 1
        winner_ranks.append(rank)
        top_1 += int(rank == 1)
        all_probability.extend(map(float, probabilities))
        all_outcome.extend(map(int, outcome))
    calibration = _calibration(np.asarray(all_probability), np.asarray(all_outcome), bands)
    return {
        "race_count": len(races),
        "runner_count": len(all_probability),
        "mean_multiclass_race_log_loss": float(np.mean(losses)),
        "mean_multiclass_brier": float(np.mean(briers)),
        "runner_calibration": calibration,
        "top_1_accuracy": top_1 / len(races),
        "mean_winner_rank": float(np.mean(winner_ranks)),
        "mean_reciprocal_winner_rank": float(np.mean([1.0 / rank for rank in winner_ranks])),
        "race_log_losses": losses,
    }


def _percentile_interval(values: np.ndarray, interval: float) -> tuple[float, float]:
    tail = (1.0 - interval) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def _race_bootstrap(deltas: np.ndarray, replicates: int, seed: int, interval: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    batch_size = 256
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indexes = rng.integers(0, len(deltas), size=(stop - start, len(deltas)))
        values[start:stop] = deltas[indexes].mean(axis=1)
    lower, upper = _percentile_interval(values, interval)
    return {"replicates": replicates, "seed": seed, "lower": lower, "upper": upper}


def _cluster_bootstrap(
    deltas: np.ndarray,
    dates: Sequence[str],
    replicates: int,
    seed: int,
    interval: float,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for date, delta in zip(dates, deltas, strict=True):
        grouped[date].append(float(delta))
    keys = sorted(grouped)
    sums = np.asarray([sum(grouped[key]) for key in keys], dtype=float)
    counts = np.asarray([len(grouped[key]) for key in keys], dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    batch_size = 256
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indexes = rng.integers(0, len(keys), size=(stop - start, len(keys)))
        values[start:stop] = sums[indexes].sum(axis=1) / counts[indexes].sum(axis=1)
    lower, upper = _percentile_interval(values, interval)
    return {
        "cluster_count": len(keys),
        "replicates": replicates,
        "seed": seed,
        "lower": lower,
        "upper": upper,
    }


def _load_receipt(path: Path, expected_sha256: str, kind: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise FinalizationError(f"sealed_{kind}_receipt_hash_drift:{path.name}")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise FinalizationError(f"invalid_{kind}_receipt:{path.name}")
    return payload


def _melbourne_race_date(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FinalizationError("invalid_jump_at")
    try:
        jump_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinalizationError("invalid_jump_at") from exc
    if jump_at.tzinfo is None or jump_at.utcoffset() is None:
        raise FinalizationError("naive_jump_at")
    return jump_at.astimezone(MELBOURNE).date().isoformat()


def finalize(
    snapshot: Mapping[str, Any],
    cohort_root: Path,
    protocol_path: Path,
    asset_dir: Path,
) -> dict[str, Any]:
    protocol, model, preprocessing, _, _ = load_frozen_assets(protocol_path, asset_dir)
    target = protocol["cohort"]["target_races"]
    if (
        snapshot.get("state") != "FINALIZATION_LOCKED"
        or snapshot.get("score_invocation_count") != 1
        or len(snapshot.get("predictions", {})) != target
        or len(snapshot.get("results", {})) != target
    ):
        raise FinalizationError("exact_fixed_n_finalization_not_locked_once")
    ordered_predictions = sorted(
        snapshot["predictions"].values(),
        key=lambda row: (row["jump_at"], row["captured_at"], row["race_id"]),
    )
    manifest = [
        {
            "member_id": prediction["member_id"],
            "prediction_receipt_sha256": prediction["prediction_receipt_sha256"],
            "result_receipt_sha256": snapshot["results"][prediction["member_id"]]["result_receipt_sha256"],
        }
        for prediction in ordered_predictions
    ]
    manifest_sha256 = sha256_bytes(canonical_bytes(manifest))
    if manifest_sha256 != snapshot.get("finalization_member_manifest_sha256"):
        raise FinalizationError("fixed_n_member_manifest_drift")

    races: list[dict[str, Any]] = []
    for prediction_event in ordered_predictions:
        member_id = prediction_event["member_id"]
        result_event = snapshot["results"][member_id]
        prediction = _load_receipt(
            cohort_root / "predictions" / f"{member_id}.json",
            prediction_event["prediction_receipt_sha256"],
            "prediction",
        )
        result = _load_receipt(
            cohort_root / "results" / f"{member_id}.json",
            result_event["result_receipt_sha256"],
            "result",
        )
        if (
            prediction.get("member_id") != member_id
            or result.get("member_id") != member_id
            or prediction.get("race_id") != result.get("race_id")
            or prediction.get("race_id") != prediction_event["race_id"]
        ):
            raise FinalizationError(f"sealed_race_identity_drift:{member_id}")
        if runner_set_sha256(prediction["runners"]) != prediction_event["runner_set_sha256"]:
            raise FinalizationError(f"sealed_prediction_runner_set_drift:{member_id}")
        if runner_set_sha256(result["runners"]) != prediction_event["runner_set_sha256"]:
            raise FinalizationError(f"sealed_result_runner_set_drift:{member_id}")
        scored = score_race(prediction["runners"], model, preprocessing)
        result_by_box = {row["box_number"]: row for row in result["runners"]}
        rows: list[dict[str, Any]] = []
        winners = 0
        for runner in sorted(prediction["runners"], key=lambda row: row["box_number"]):
            box = runner["box_number"]
            result_runner = result_by_box.get(box)
            if result_runner is None or result_runner.get("dog_name") != runner.get("dog_name"):
                raise FinalizationError(f"sealed_runner_identity_drift:{member_id}:{box}")
            is_winner = result_runner.get("finish_position") == 1
            winners += int(is_winner)
            for key in ("baseline_probability", "candidate_probability"):
                if not math.isclose(
                    _finite_number(runner.get(key), key),
                    scored[box][key],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise FinalizationError(f"sealed_prediction_probability_drift:{member_id}:{box}:{key}")
            rows.append(
                {
                    "box_number": box,
                    "dog_name": runner["dog_name"],
                    "baseline_probability": scored[box]["baseline_probability"],
                    "candidate_probability": scored[box]["candidate_probability"],
                    "is_winner": is_winner,
                }
            )
        if winners != 1 or result.get("winner_box") != next(row["box_number"] for row in rows if row["is_winner"]):
            raise FinalizationError(f"sealed_winner_conflict:{member_id}")
        races.append(
            {
                "member_id": member_id,
                "race_id": prediction["race_id"],
                "race_date": _melbourne_race_date(prediction.get("jump_at")),
                "runners": rows,
            }
        )

    bands = protocol["evaluation"]["calibration_bands"]
    baseline = _metric_summary(races, "baseline_probability", bands)
    candidate = _metric_summary(races, "candidate_probability", bands)
    deltas = np.asarray(candidate.pop("race_log_losses"), dtype=float) - np.asarray(
        baseline.pop("race_log_losses"), dtype=float
    )
    mean_delta = float(deltas.mean())
    uncertainty = protocol["evaluation"]["uncertainty"]
    race_spec = uncertainty["race_bootstrap"]
    cluster_spec = uncertainty["race_date_cluster_bootstrap"]
    race_interval = _race_bootstrap(
        deltas, race_spec["replicates"], race_spec["seed"], race_spec["interval"]
    )
    cluster_interval = _cluster_bootstrap(
        deltas,
        [race["race_date"] for race in races],
        cluster_spec["replicates"],
        cluster_spec["seed"],
        cluster_spec["interval"],
    )
    blocks = np.array_split(deltas, protocol["evaluation"]["stability"]["chronological_blocks"])
    block_rows = [
        {"block": index + 1, "race_count": len(block), "mean_log_loss_delta": float(block.mean())}
        for index, block in enumerate(blocks)
    ]
    negative_blocks = sum(row["mean_log_loss_delta"] < 0.0 for row in block_rows)
    confirmed = (
        mean_delta < 0.0
        and race_interval["upper"] < 0.0
        and cluster_interval["upper"] < 0.0
        and negative_blocks >= protocol["evaluation"]["stability"]["minimum_negative_blocks"]
    )
    verdict = (
        protocol["evaluation"]["confirmation_rule"]["valid_evidence_gate_pass_verdict"]
        if confirmed
        else protocol["evaluation"]["confirmation_rule"]["valid_evidence_gate_failure_verdict"]
    )
    return {
        "schema_version": "forward_overround_successor_final_report_v1",
        "verdict": verdict,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "member_manifest_sha256": manifest_sha256,
        "race_count": target,
        "identical_races_compared": True,
        "score_invocation_count": 1,
        "metrics": {
            "primary": {
                "name": "mean_multiclass_race_log_loss",
                "candidate_minus_baseline": mean_delta,
                "baseline": baseline["mean_multiclass_race_log_loss"],
                "candidate": candidate["mean_multiclass_race_log_loss"],
            },
            "baseline": baseline,
            "candidate": candidate,
            "race_bootstrap_95pct": race_interval,
            "race_date_cluster_bootstrap_95pct": cluster_interval,
            "chronological_blocks": block_rows,
            "negative_chronological_blocks": negative_blocks,
        },
        "profitability": {"roi_computed": False, "betting_analysis_performed": False},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--cohort-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = json.loads(args.snapshot.read_bytes())
    report = finalize(snapshot, args.cohort_root, args.protocol, args.asset_dir)
    write_once(args.output, report)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
