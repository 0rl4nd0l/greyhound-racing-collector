#!/usr/bin/env python3
"""Rerun only experiments affected by the sealed Sportsbet WIN repair.

The driver is deliberately report-only.  It reads the immutable original
experiment bundles and the canonical WIN recovery sidecars, intersects whole
races fail-closed, and reuses the frozen experiment implementations for model
families, metrics, splits, seeds, and paired race bootstrap calculations.
It never opens August outcomes, writes a database, or overwrites an existing
artifact directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ANALYSIS_DATE = "2026-08-15"
EXPECTED_MATRIX_SHA256 = "eb1783d4cc07e6980463a097c97fdac9f5370b08f493ca15addf768aa0b014b6"
EXPECTED_SIDECAR_SHA256 = "880ae93680e56991fa2c9eb316732cbc71bc7ff713525efcf83750ceace4493d"
EXPECTED_DEPENDENCY_SHA256 = "8623a791ee3b573ef8f191f16f412a3285888b94caf3d86be5f4c7f7cca545bf"
EXPECTED_SOURCE_DB_SHA256 = "0d2b84c923d1380864c0d19495b1279d0a64764ab6fd92b20eacfdffd02ad1c7"
EXPECTED_TIER_A_CLASSIFICATIONS = {
    "VERIFIED_WIN": 3692,
    "PLACE_MISLABEL": 968,
    "UNPARSABLE": 92,
}
CANONICAL_CLASSES = {"VERIFIED_WIN", "PLACE_MISLABEL", "RECOVERABLE_WIN"}
DEFAULT_SOURCE_ROOT = Path(
    "/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector"
)
DEFAULT_TIER_A_ROOT = Path(
    "/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/"
    "greyhound-frozen-rf-win-market-evaluation-20260716/reports/agent_jobs/"
    "frozen_rf_win_market_evaluation_20260716"
)
DEFAULT_OUTPUT = Path(
    "artifacts/corrected_sportsbet_win_reruns_20260815_report_only"
)
TRAIN_END = "2026-07-08"
VALID_START = "2026-07-09"
VALID_END = "2026-07-18"
DEV_END = "2026-08-02"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tier_a_capture_timestamps(path: Path) -> dict[tuple[str, int], str]:
    return {
        key: str(row["odds_capture_timestamp"])
        for key, row in load_tier_a_runner_bindings(path).items()
    }


def load_tier_a_runner_bindings(
    path: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    bindings: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("strongest_tier") != "A":
                continue
            key = (str(row["race_id"]), int(row["box_number"]))
            if key in bindings:
                raise ValueError(f"duplicate Tier-A runner identity: {key}")
            timestamp = str(row.get("odds_capture_timestamp") or "")
            if not timestamp:
                raise ValueError(f"Tier-A runner missing capture timestamp: {key}")
            odds = float(row["strict_win_odds"])
            if odds <= 1:
                raise ValueError(f"Tier-A runner invalid stored WIN odds: {key}")
            bindings[key] = {
                **row,
                "box_number": key[1],
                "odds_capture_timestamp": timestamp,
                "strict_win_odds": odds,
            }
    return bindings


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            )


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runner_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["race_id"]),
        int(row["box_number"]),
        str(row["odds_capture_timestamp"]),
    )


def race_ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    race_ids = sorted({str(row["race_id"]) for row in rows})
    return hashlib.sha256("".join(f"{race_id}\n" for race_id in race_ids).encode()).hexdigest()


def intersection_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = sorted(
        (
            str(row["race_id"]),
            int(row["box_number"]),
            str(row["odds_capture_timestamp"]),
            float(row["market_implied_probability"]),
        )
        for row in rows
    )
    return digest_json(identities)


def population(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    races = {str(row["race_id"]) for row in rows}
    return {
        "races": len(races),
        "runner_rows": len(rows),
        "race_ids_sha256": race_ids_sha256(rows),
        "canonical_intersection_sha256": intersection_sha256(rows),
    }


def canonical_index(
    matrix_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in matrix_rows:
        key = runner_key(row)
        if key in index:
            raise ValueError(f"canonical runner collision: {key}")
        if str(row.get("sportsbet_win_evidence_classification")) not in CANONICAL_CLASSES:
            raise ValueError(f"noncanonical row in canonical matrix: {key}")
        probability = float(row["market_implied_probability"])
        odds = float(row["canonical_sportsbet_win_odds"])
        if not (0 < probability < 1 and odds > 1):
            raise ValueError(f"canonical value outside domain: {key}")
        index[key] = dict(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index.values():
        grouped[str(row["race_id"])].append(row)
    for race_id, rows in grouped.items():
        if len(rows) != int(rows[0]["field_size"]):
            raise ValueError(f"canonical incomplete field: {race_id}")
        if not math.isclose(
            sum(float(row["market_implied_probability"]) for row in rows),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"canonical probabilities not normalized: {race_id}")
    return index


def corrected_intersection(
    original_rows: Sequence[Mapping[str, Any]],
    canonical: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Substitute canonical WIN values and retain complete original race fields."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in original_rows:
        grouped[str(row["race_id"])].append(row)
    corrected: list[dict[str, Any]] = []
    for race_id in sorted(grouped):
        race_rows = grouped[race_id]
        matched = [canonical.get(runner_key(row)) for row in race_rows]
        if any(row is None for row in matched):
            continue
        if len({int(row["box_number"]) for row in race_rows}) != len(race_rows):
            raise ValueError(f"duplicate original box identity: {race_id}")
        if sum(int(row["label_is_winner"]) for row in race_rows) != 1:
            raise ValueError(f"invalid winner field: {race_id}")
        for original, repaired in zip(race_rows, matched):
            assert repaired is not None
            item = dict(original)
            item["market_implied_probability_original"] = float(
                original["market_implied_probability"]
            )
            item["market_implied_probability"] = float(
                repaired["market_implied_probability"]
            )
            item["canonical_sportsbet_win_odds"] = float(
                repaired["canonical_sportsbet_win_odds"]
            )
            item["sportsbet_win_source_row_id"] = int(
                repaired["sportsbet_win_source_row_id"]
            )
            item["sportsbet_win_evidence_classification"] = str(
                repaired["sportsbet_win_evidence_classification"]
            )
            corrected.append(item)
    return corrected


def normalize_frame(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(scores, dtype=float), 1e-15, None)
    totals = pd.Series(values).groupby(frame["race_id"], sort=False).transform("sum")
    return values / totals.to_numpy()


def per_race_losses(
    frame: pd.DataFrame, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    data = frame[["race_id", "label_is_winner"]].copy()
    data["p"] = probabilities
    grouped = data.groupby("race_id", sort=False)
    log_loss = grouped.apply(
        lambda race: -math.log(
            max(float(race.loc[race["label_is_winner"] == 1, "p"].iloc[0]), 1e-15)
        ),
        include_groups=False,
    ).to_numpy(float)
    brier = grouped.apply(
        lambda race: float(
            ((race["p"] - race["label_is_winner"].astype(float)) ** 2).sum()
        ),
        include_groups=False,
    ).to_numpy(float)
    return log_loss, brier


def paired_uncertainty(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    reference: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    reference_loss, reference_brier = per_race_losses(frame, predictions[reference])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(reference_loss), size=(repetitions, len(reference_loss)))
    output: dict[str, Any] = {}
    for name, values in predictions.items():
        if name == reference:
            continue
        loss, brier = per_race_losses(frame, values)
        delta_loss = loss - reference_loss
        delta_brier = brier - reference_brier
        loss_samples = delta_loss[draws].mean(axis=1)
        brier_samples = delta_brier[draws].mean(axis=1)
        output[name] = {
            "reference": reference,
            "unit": "race",
            "repetitions": repetitions,
            "seed": seed,
            "delta_log_loss": float(delta_loss.mean()),
            "log_loss_lower_95": float(np.percentile(loss_samples, 2.5)),
            "log_loss_upper_95": float(np.percentile(loss_samples, 97.5)),
            "probability_log_loss_delta_lt_zero": float(np.mean(loss_samples < 0)),
            "delta_brier": float(delta_brier.mean()),
            "brier_lower_95": float(np.percentile(brier_samples, 2.5)),
            "brier_upper_95": float(np.percentile(brier_samples, 97.5)),
            "probability_brier_delta_lt_zero": float(np.mean(brier_samples < 0)),
        }
    return output


def prediction_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def run_classifier_protocol(
    *,
    branch: str,
    rows: Sequence[Mapping[str, Any]],
    module: ModuleType,
    candidates: Mapping[str, tuple[list[str], str]],
    market_metric_name: str,
    repetitions: int,
    seed: int,
    output_dir: Path,
    original_report: Path,
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    train = frame[frame["race_date"] <= TRAIN_END].reset_index(drop=True)
    valid = frame[
        frame["race_date"].between(VALID_START, VALID_END)
    ].reset_index(drop=True)
    development = frame[frame["race_date"] <= DEV_END].reset_index(drop=True)
    predictions: dict[str, np.ndarray] = {
        "market_baseline": normalize_frame(
            valid, valid["market_implied_probability"].to_numpy(float)
        )
    }
    results: dict[str, Any] = {
        "market_baseline": {
            "features": ["market_implied_probability"],
            "validation": getattr(module, market_metric_name)(
                valid, predictions["market_baseline"]
            ),
        }
    }
    fitted: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    refit: dict[str, Any] = {}
    for name, (features, family) in candidates.items():
        model = module.estimator(features, family)
        model.fit(train[features], train["label_is_winner"])
        values = normalize_frame(valid, model.predict_proba(valid[features])[:, 1])
        predictions[name] = values
        results[name] = {
            "features": features,
            "validation": getattr(module, market_metric_name)(valid, values),
        }
        fitted[name] = model
        replayed = normalize_frame(
            valid, pickle.loads(pickle.dumps(model, protocol=5)).predict_proba(valid[features])[:, 1]
        )
        second = module.estimator(features, family)
        second.fit(train[features], train["label_is_winner"])
        refitted = normalize_frame(valid, second.predict_proba(valid[features])[:, 1])
        replay[name] = {
            "max_abs_probability_delta": float(np.max(np.abs(values - replayed))),
            "prediction_sha256": prediction_sha256(replayed),
        }
        refit[name] = {
            "max_abs_probability_delta": float(np.max(np.abs(values - refitted))),
            "prediction_sha256": prediction_sha256(refitted),
        }
    key = lambda item: (
        item[1]["validation"]["log_loss"],
        item[1]["validation"]["brier_multiclass"],
        item[1]["validation"]["calibration_error"],
        item[1]["validation"]["mean_winner_rank"],
        item[0],
    )
    selected = min(results.items(), key=key)[0]
    paired = paired_uncertainty(
        valid,
        predictions,
        "market_baseline",
        repetitions=repetitions,
        seed=seed,
    )
    selected_paired = paired.get(selected)
    signal = (
        selected != "market_baseline"
        and selected_paired is not None
        and selected_paired["log_loss_upper_95"] < 0
    )
    if branch == "basic Sportsbet history v2":
        status = (
            "DEVELOPMENT_SIGNAL_READY_FOR_FORWARD_TEST"
            if signal
            else "NO_INCREMENTAL_SIGNAL"
        )
    elif branch == "clean speed/context rerun":
        status = (
            "DEVELOPMENT_SIGNAL_READY_FOR_FORWARD_TEST"
            if signal
            else "NO_INCREMENTAL_SIGNAL_SPEED_CONTEXT"
        )
    else:
        status = "CORRECTED_RERUN_COMPLETE"
    output_dir.mkdir(parents=True)
    with (output_dir / "validation_models.pkl").open("wb") as handle:
        pickle.dump(fitted, handle, protocol=5)
    report = {
        "schema_version": "corrected_win_classifier_rerun_v1",
        "branch": branch,
        "status": status,
        "original_report": str(original_report),
        "original_report_sha256": sha256_file(original_report),
        "canonical_population": population(rows),
        "split": {
            "train_races": int(train["race_id"].nunique()),
            "train_runner_rows": len(train),
            "validation_races": int(valid["race_id"].nunique()),
            "validation_runner_rows": len(valid),
            "development_races": int(development["race_id"].nunique()),
            "development_runner_rows": len(development),
        },
        "candidate_results": results,
        "paired_vs_market": paired,
        "selection": {
            "selected_candidate_id": selected,
            "selection_key": list(key((selected, results[selected]))),
        },
        "replay": replay,
        "deterministic_refit": refit,
        "gates": {
            "serialization_replay_exact": all(
                item["max_abs_probability_delta"] == 0 for item in replay.values()
            ),
            "deterministic_refit_exact": all(
                item["max_abs_probability_delta"] == 0 for item in refit.values()
            ),
        },
        "boundaries": {
            "august_opened": False,
            "forward_cohort_created": False,
            "roi_run": False,
            "database_written": False,
        },
    }
    write_json(output_dir / "report.json", report)
    return report


def run_canonical_training(
    rows: Sequence[Mapping[str, Any]], module: ModuleType, output_dir: Path, original: Path
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    train = frame[frame["race_date"] < module.SPLIT_DATE].reset_index(drop=True)
    holdout = frame[frame["race_date"] >= module.SPLIT_DATE].reset_index(drop=True)
    market = module.normalize_by_race(
        holdout, holdout["market_implied_probability"].to_numpy(float)
    )
    predictions = {"normalized_market": market}
    results = {
        "normalized_market": {
            "features": ["market_implied_probability"],
            "holdout": module.metrics(holdout, market),
        }
    }
    replay: dict[str, Any] = {}
    refit: dict[str, Any] = {}
    for candidate in module.candidates():
        candidate.estimator.fit(train[candidate.features], train["label_is_winner"])
        values = module.normalize_by_race(
            holdout,
            candidate.estimator.predict_proba(holdout[candidate.features])[:, 1],
        )
        predictions[candidate.name] = values
        results[candidate.name] = {
            "features": candidate.features,
            "holdout": module.metrics(holdout, values),
            "advancement": module.advancement(
                module.metrics(holdout, values), results["normalized_market"]["holdout"]
            ),
        }
        replayed = module.normalize_by_race(
            holdout,
            pickle.loads(pickle.dumps(candidate.estimator, protocol=5)).predict_proba(
                holdout[candidate.features]
            )[:, 1],
        )
        second = next(x for x in module.candidates() if x.name == candidate.name)
        second.estimator.fit(train[second.features], train["label_is_winner"])
        refitted = module.normalize_by_race(
            holdout, second.estimator.predict_proba(holdout[second.features])[:, 1]
        )
        replay[candidate.name] = {
            "max_abs_probability_delta": float(np.max(np.abs(values - replayed))),
            "prediction_sha256": prediction_sha256(replayed),
        }
        refit[candidate.name] = {
            "max_abs_probability_delta": float(np.max(np.abs(values - refitted))),
            "prediction_sha256": prediction_sha256(refitted),
        }
    paired = paired_uncertainty(
        holdout,
        predictions,
        "normalized_market",
        repetitions=2000,
        seed=module.SEED + 1,
    )
    report = {
        "schema_version": "corrected_canonical_training_model_rerun_v1",
        "branch": "canonical training model experiment 20260812",
        "status": "CORRECTED_RERUN_COMPLETE",
        "original_report": str(original),
        "original_report_sha256": sha256_file(original),
        "canonical_population": population(rows),
        "split": {
            "training_races": int(train["race_id"].nunique()),
            "training_runner_rows": len(train),
            "evaluation_races": int(holdout["race_id"].nunique()),
            "evaluation_runner_rows": len(holdout),
        },
        "models": results,
        "paired_vs_market": paired,
        "replay": replay,
        "deterministic_refit": refit,
        "gates": {
            "serialization_replay_exact": all(
                item["max_abs_probability_delta"] == 0 for item in replay.values()
            ),
            "deterministic_refit_exact": all(
                item["max_abs_probability_delta"] == 0 for item in refit.values()
            ),
        },
        "boundaries": {"model_serialized": False, "august_opened": False, "roi_run": False},
    }
    output_dir.mkdir(parents=True)
    write_json(output_dir / "report.json", report)
    return report


def tier_a_core_metrics(races: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_rows: list[dict[str, float]] = []
    market_rows: list[dict[str, float]] = []
    for race in races:
        rows = list(race["rows"])
        winner = next(i for i, row in enumerate(rows) if int(row["actual_win"]) == 1)
        for label, key, target in (
            ("model", "rf_probability", model_rows),
            ("market", "market_probability", market_rows),
        ):
            values = np.asarray([float(row[key]) for row in rows])
            order = sorted(
                range(len(rows)),
                key=lambda i: (-values[i], int(rows[i]["box_number"])),
            )
            rank = order.index(winner) + 1
            target.append(
                {
                    "top_1": float(rank == 1),
                    "top_3": float(rank <= 3),
                    "mean_winner_rank": float(rank),
                    "brier_multiclass": float(
                        sum(
                            (value - float(i == winner)) ** 2
                            for i, value in enumerate(values)
                        )
                    ),
                    "log_loss": float(-math.log(max(values[winner], 1e-15))),
                }
            )
    def aggregate(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
        return {key: float(np.mean([row[key] for row in values])) for key in values[0]}
    return {"model": aggregate(model_rows), "market": aggregate(market_rows)}


def corrected_tier_a_races(
    *,
    races: Sequence[Mapping[str, Any]],
    runners_path: Path,
    source_db: Path,
    audit_module: ModuleType,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bindings = load_tier_a_runner_bindings(runners_path)
    expected_identities = {
        (str(row["race_id"]), int(row["box_number"]))
        for race in races
        for row in race["rows"]
    }
    if set(bindings) != expected_identities:
        raise ValueError("Tier-A validated runner population does not match source bindings")

    race_ids = sorted({race_id for race_id, _ in bindings})
    placeholders = ",".join("?" for _ in race_ids)
    connection = sqlite3.connect(
        f"{source_db.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        source_rows = [
            dict(row)
            for row in connection.execute(
                f"SELECT * FROM live_odds WHERE source = 'sportsbet' "
                f"AND market_type = 'win' AND race_id IN ({placeholders})",
                race_ids,
            )
        ]
    finally:
        connection.close()

    indexed: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        indexed[
            (
                str(row["race_id"]),
                int(row["box_number"]),
                str(row["capture_timestamp"]),
            )
        ].append(row)

    audited: list[dict[str, Any]] = []
    evidence_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for identity, binding in bindings.items():
        key = (*identity, str(binding["odds_capture_timestamp"]))
        candidates = indexed.get(key, [])
        if len(candidates) != 1:
            raise ValueError(f"Tier-A source row cardinality mismatch: {key}:{len(candidates)}")
        source = candidates[0]
        if audit_module.normalize_name(source["dog_name"]) != audit_module.normalize_name(
            binding["dog_name"]
        ):
            raise ValueError(f"Tier-A source runner conflict: {key}")
        if not math.isclose(
            float(source["odds_decimal"]),
            float(binding["strict_win_odds"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Tier-A stored price conflict: {key}")
        evidence = audit_module.classify_win_evidence(
            raw_text=source.get("sportsbet_raw_runner_text"),
            expected_box=int(source["box_number"]),
            stored_odds=float(source["odds_decimal"]),
        )
        item = {
            "race_id": identity[0],
            "box_number": identity[1],
            "odds_capture_timestamp": key[2],
            "source_row_id": int(source["id"]),
            "stored_odds_decimal": float(source["odds_decimal"]),
            "classification": str(evidence.classification),
            "canonical_win_odds": evidence.canonical_win_odds,
            "paired_win_odds": evidence.paired_win_odds,
            "paired_place_odds": evidence.paired_place_odds,
            "reason": str(evidence.reason),
        }
        audited.append(item)
        evidence_by_identity[identity] = item

    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for race in races:
        race_id = str(race["race_id"])
        evidence_rows = [
            evidence_by_identity[(race_id, int(row["box_number"]))]
            for row in race["rows"]
        ]
        if any(row["classification"] not in CANONICAL_CLASSES for row in evidence_rows):
            excluded.append(
                {"race_id": race_id, "reason": "noncanonical_tier_a_win_evidence"}
            )
            continue
        repaired_rows = []
        for row, evidence in zip(race["rows"], evidence_rows):
            if evidence["canonical_win_odds"] is None:
                raise ValueError(f"Tier-A canonical WIN odds missing: {race_id}")
            repaired = dict(row)
            repaired["strict_win_odds"] = float(evidence["canonical_win_odds"])
            repaired["odds_capture_timestamp"] = evidence["odds_capture_timestamp"]
            repaired_rows.append(repaired)
        retained.append({**race, "rows": repaired_rows})
    return retained, audited, excluded


def run_tier_a(
    module: ModuleType,
    audit_module: ModuleType,
    source_db: Path,
    source_db_sha256: str,
    output_dir: Path,
    original: Path,
) -> dict[str, Any]:
    races, source_validation, _ = module.validate_and_load()
    if len(races) != 678 or sum(len(race["rows"]) for race in races) != 4752:
        raise ValueError("Tier-A frozen input population mismatch")
    retained, audited, exclusions = corrected_tier_a_races(
        races=races,
        runners_path=module.RUNNERS_PATH,
        source_db=source_db,
        audit_module=audit_module,
    )
    classification_counts = dict(
        sorted(Counter(row["classification"] for row in audited).items())
    )
    if classification_counts != EXPECTED_TIER_A_CLASSIFICATIONS:
        raise ValueError(
            f"Tier-A WIN classification count mismatch: {classification_counts}"
        )
    if len(retained) != 655 or sum(len(race["rows"]) for race in retained) != 4660:
        raise ValueError("Tier-A corrected intersection population mismatch")
    compact_rows: list[dict[str, Any]] = []
    for race in retained:
        repaired_rows = race["rows"]
        inverse = np.asarray([1.0 / float(row["strict_win_odds"]) for row in repaired_rows])
        probabilities = inverse / inverse.sum()
        for row, value in zip(repaired_rows, probabilities):
            row["market_probability"] = float(value)
            compact_rows.append(
                {
                    "race_id": row["race_id"],
                    "box_number": row["box_number"],
                    "odds_capture_timestamp": row["odds_capture_timestamp"],
                    "market_implied_probability": float(value),
                }
            )
    metrics = tier_a_core_metrics(retained)
    differences = []
    for race in retained:
        race_metric = tier_a_core_metrics([race])
        differences.append(
            {
                name: race_metric["model"][name] - race_metric["market"][name]
                for name in metrics["model"]
            }
        )
    rng = np.random.default_rng(20260716)
    indices = rng.integers(0, len(differences), size=(10000, len(differences)))
    paired: dict[str, Any] = {}
    for name in differences[0]:
        values = np.asarray([row[name] for row in differences])
        samples = values[indices].mean(axis=1)
        paired[name] = {
            "model_minus_market": float(values.mean()),
            "ci95_lower": float(np.percentile(samples, 2.5)),
            "ci95_upper": float(np.percentile(samples, 97.5)),
        }
    market_wins = (
        paired["top_1"]["ci95_upper"] < 0
        and paired["top_3"]["ci95_upper"] < 0
        and paired["mean_winner_rank"]["ci95_lower"] > 0
        and paired["brier_multiclass"]["ci95_lower"] > 0
        and paired["log_loss"]["ci95_lower"] > 0
    )
    output_dir.mkdir(parents=True)
    evidence_path = output_dir / "corrected_tier_a_win_evidence.jsonl"
    exclusions_path = output_dir / "exclusions.jsonl"
    write_jsonl(evidence_path, audited)
    write_jsonl(exclusions_path, exclusions)
    report = {
        "schema_version": "corrected_tier_a_frozen_rf_vs_win_market_v1",
        "branch": "historical Tier-A frozen-RF vs WIN-market evaluation 20260716",
        "status": "FROZEN_RF_DOES_NOT_BEAT_CORRECTED_MARKET" if market_wins else "MIXED_OR_UNSTABLE_RESULT_KEEP_BASELINE",
        "original_report": str(original),
        "original_report_sha256": sha256_file(original),
        "canonical_population": population(compact_rows),
        "source_population": {
            "races": len(races),
            "runner_rows": len(audited),
            "classification_counts": classification_counts,
            "excluded_races": len(exclusions),
        },
        "corrected_evidence_sidecar": {
            "path": evidence_path.name,
            "sha256": sha256_file(evidence_path),
        },
        "exclusions": {
            "path": exclusions_path.name,
            "sha256": sha256_file(exclusions_path),
        },
        "source_db": {"path": str(source_db), "sha256": source_db_sha256},
        "model_sha256": module.EXPECTED_MODEL_SHA256,
        "metrics": metrics,
        "paired_race_bootstrap": {
            "unit": "race",
            "repetitions": 10000,
            "seed": 20260716,
            "metrics": paired,
        },
        "source_validation_original": source_validation["status"],
        "boundaries": {
            "model_refit": False,
            "model_predictions_changed": False,
            "roi_run": False,
            "august_opened": False,
        },
    }
    write_json(output_dir / "report.json", report)
    return report


def run_calibration(
    rows: Sequence[Mapping[str, Any]], module: ModuleType, output_dir: Path, original: Path
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    frame["race_date"] = frame["race_date"].astype(str).str[:10]
    calibration = frame[
        frame["race_date"].between(module.CAL_START, module.CAL_END)
    ].reset_index(drop=True)
    validation = frame[
        frame["race_date"].between(module.VALID_START, module.VALID_END)
    ].reset_index(drop=True)
    predictions = {
        "raw_market": module.normalize(
            validation, validation[module.PROBABILITY].to_numpy(float)
        )
    }
    fitted = {}
    replay = {}
    refit = {}
    for name in ("platt", "beta", "isotonic"):
        model = module.fit_transform(calibration, name)
        fitted[name] = model
        predictions[name] = module.apply_transform(validation, model, name)
        loaded = pickle.loads(pickle.dumps(model, protocol=5))
        replayed = module.apply_transform(validation, loaded, name)
        second = module.fit_transform(calibration, name)
        refitted = module.apply_transform(validation, second, name)
        replay[name] = {
            "max_abs_probability_delta": float(
                np.max(np.abs(predictions[name] - replayed))
            ),
            "prediction_sha256": prediction_sha256(replayed),
        }
        refit[name] = {
            "max_abs_probability_delta": float(
                np.max(np.abs(predictions[name] - refitted))
            ),
            "prediction_sha256": prediction_sha256(refitted),
        }
    results = {
        name: {"validation": module.metric_values(validation, values)}
        for name, values in predictions.items()
    }
    key = lambda item: (
        item[1]["validation"]["log_loss"],
        item[1]["validation"]["brier_multiclass"],
        item[1]["validation"]["calibration_error"],
        item[1]["validation"]["mean_winner_rank"],
        item[0],
    )
    selected = min(results.items(), key=key)[0]
    paired = paired_uncertainty(
        validation,
        predictions,
        "raw_market",
        repetitions=module.BOOTSTRAP_REPS,
        seed=module.SEED + 1,
    )
    selected_paired = paired.get(selected)
    signal = (
        selected != "raw_market"
        and selected_paired is not None
        and selected_paired["log_loss_upper_95"] < 0
        and selected_paired["brier_upper_95"] < 0
    )
    status = (
        "CALIBRATION_SIGNAL_READY_FOR_FORWARD_TEST"
        if signal
        else "NO_STABLE_CALIBRATION_IMPROVEMENT"
    )
    report = {
        "schema_version": "corrected_sportsbet_calibration_rerun_v1",
        "branch": "Sportsbet calibration",
        "status": status,
        "original_report": str(original),
        "original_report_sha256": sha256_file(original),
        "canonical_population": population(rows),
        "split": {
            "calibration_races": int(calibration["race_id"].nunique()),
            "calibration_runner_rows": len(calibration),
            "validation_races": int(validation["race_id"].nunique()),
            "validation_runner_rows": len(validation),
        },
        "candidate_results": results,
        "paired_vs_raw_market": paired,
        "selection": {"selected_candidate_id": selected, "selection_key": list(key((selected, results[selected])))},
        "replay": replay,
        "deterministic_refit": refit,
        "gates": {
            "serialization_replay_exact": all(x["max_abs_probability_delta"] == 0 for x in replay.values()),
            "deterministic_refit_exact": all(x["max_abs_probability_delta"] == 0 for x in refit.values()),
        },
        "boundaries": {"august_opened": False, "roi_run": False},
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "calibrators.pkl").open("wb") as handle:
        pickle.dump(fitted, handle, protocol=5)
    write_json(output_dir / "report.json", report)
    return report


def run_raw_shape(
    rows: Sequence[Mapping[str, Any]], module: ModuleType, output_dir: Path, original: Path
) -> dict[str, Any]:
    fold_results = []
    replay = {}
    refit = {}
    final_models = {}
    for fold in module.FOLDS:
        train = [row for row in rows if fold["train_start"] <= row["race_date"] <= fold["train_end"]]
        valid = [row for row in rows if fold["valid_start"] <= row["race_date"] <= fold["valid_end"]]
        market = np.asarray([float(row["market_implied_probability"]) for row in valid])
        models = {
            "B_market_plus_availability": module.fit_offset(train, module.AVAILABILITY_FEATURES),
            "C_market_plus_availability_plus_values": module.fit_offset(train, module.AVAILABILITY_FEATURES + module.VALUE_FEATURES),
        }
        predictions = {
            "A_market_baseline": module.race_probabilities(valid, market),
            "B_market_plus_availability": module.predict_offset(valid, models["B_market_plus_availability"]),
            "C_market_plus_availability_plus_values": module.predict_offset(valid, models["C_market_plus_availability_plus_values"]),
        }
        seed = int(module.SEED + list(module.FOLDS).index(fold) + 1)
        comparisons = {
            "B_vs_A": module.paired(valid, predictions["B_market_plus_availability"], predictions["A_market_baseline"], seed, "A_market_baseline"),
            "C_vs_A": module.paired(valid, predictions["C_market_plus_availability_plus_values"], predictions["A_market_baseline"], seed + 100, "A_market_baseline"),
            "C_vs_B": module.paired(valid, predictions["C_market_plus_availability_plus_values"], predictions["B_market_plus_availability"], seed + 200, "B_market_plus_availability"),
        }
        fold_results.append({
            "fold": fold,
            "population": {"train_races": len(module.grouped_indices(train)), "train_runner_rows": len(train), "validation_races": len(module.grouped_indices(valid)), "validation_runner_rows": len(valid)},
            "candidate_results": {name: module.metrics(valid, values) for name, values in predictions.items()},
            "paired_comparisons": comparisons,
        })
        if fold["id"] == "rolling_3_final":
            final_models = models
            for name, model in models.items():
                values = module.predict_offset(valid, model)
                loaded = pickle.loads(pickle.dumps(model, protocol=4))
                replayed = module.predict_offset(valid, loaded)
                feature_set = module.AVAILABILITY_FEATURES if name.startswith("B_") else module.AVAILABILITY_FEATURES + module.VALUE_FEATURES
                second = module.fit_offset(train, feature_set)
                refitted = module.predict_offset(valid, second)
                replay[name] = {"max_abs_probability_delta": float(np.max(np.abs(values - replayed))), "prediction_sha256": prediction_sha256(replayed)}
                refit[name] = {"max_abs_probability_delta": float(np.max(np.abs(values - refitted))), "prediction_sha256": prediction_sha256(refitted)}
    final = fold_results[-1]
    c_a = final["paired_comparisons"]["C_vs_A"]
    c_b = final["paired_comparisons"]["C_vs_B"]
    metrics = final["candidate_results"]
    survivor = (
        c_a["delta_log_loss"] < 0 and c_a["log_loss_upper_95"] < 0
        and c_b["delta_log_loss"] < 0 and c_b["log_loss_upper_95"] < 0
        and metrics["C_market_plus_availability_plus_values"]["brier_multiclass"] <= metrics["A_market_baseline"]["brier_multiclass"]
        and metrics["C_market_plus_availability_plus_values"]["brier_multiclass"] <= metrics["B_market_plus_availability"]["brier_multiclass"]
        and sum(f["paired_comparisons"]["C_vs_A"]["delta_log_loss"] < 0 for f in fold_results) >= 2
        and sum(f["paired_comparisons"]["C_vs_B"]["delta_log_loss"] < 0 for f in fold_results) >= 2
    )
    report = {
        "schema_version": "corrected_raw_race_shape_rerun_v1",
        "branch": "raw race-shape Sportsbet baseline",
        "status": "RAW_RACE_SHAPE_SIGNAL_READY_FOR_FORWARD_TEST" if survivor else "NO_INCREMENTAL_RAW_RACE_SHAPE_SIGNAL",
        "original_report": str(original), "original_report_sha256": sha256_file(original),
        "canonical_population": population(rows), "fold_results": fold_results,
        "replay": replay, "deterministic_refit": refit,
        "gates": {"serialization_replay_exact": all(x["max_abs_probability_delta"] == 0 for x in replay.values()), "deterministic_refit_exact": all(x["max_abs_probability_delta"] == 0 for x in refit.values())},
        "boundaries": {"august_opened": False, "roi_run": False, "forward_artifact_created": False},
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "final_fold_models.pkl").open("wb") as handle:
        pickle.dump(final_models, handle, protocol=4)
    write_json(output_dir / "report.json", report)
    return report


def corrected_fixed_window(
    *,
    old_rows: Sequence[Mapping[str, Any]],
    source_extract: Sequence[Mapping[str, Any]],
    source_db: Path,
    audit_module: ModuleType,
    market_module: ModuleType,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_ids = [int(row["id"]) for row in source_extract]
    placeholders = ",".join("?" for _ in source_ids)
    connection = sqlite3.connect(
        f"{source_db.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    db_rows = {
        int(row["id"]): dict(row)
        for row in connection.execute(
            f"SELECT * FROM live_odds WHERE id IN ({placeholders})", source_ids
        )
    }
    connection.close()
    if set(db_rows) != set(source_ids):
        raise ValueError("fixed-window source row missing")
    audited = []
    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in source_extract:
        raw = db_rows[int(source["id"])]
        evidence = audit_module.classify_win_evidence(
            raw_text=raw.get("sportsbet_raw_runner_text"),
            expected_box=int(raw["box_number"]),
            stored_odds=float(raw["odds_decimal"]),
        )
        item = {
            **dict(source),
            "classification": evidence.classification,
            "canonical_win_odds": evidence.canonical_win_odds,
            "paired_win_odds": evidence.paired_win_odds,
            "paired_place_odds": evidence.paired_place_odds,
            "reason": evidence.reason,
        }
        audited.append(item)
        by_race[str(source["race_id"])].append(item)
    qualified = {
        race_id
        for race_id, rows in by_race.items()
        if rows and all(row["classification"] in CANONICAL_CLASSES for row in rows)
    }
    old_by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in old_rows:
        old_by_race[str(row["race_id"])].append(row)
    corrected = []
    for race_id in sorted(qualified):
        target = sorted(old_by_race[race_id], key=lambda row: int(row["box_number"]))
        modes: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in by_race[race_id]:
            mode = str(row["capture_mode"])
            box = int(row["box_number"])
            if box in modes[mode]:
                raise ValueError(f"fixed-window duplicate box: {race_id}:{mode}:{box}")
            modes[mode][box] = row
        early = modes[market_module.EARLY_MODE]
        latest = modes[market_module.LATEST_MODE]
        boxes = [int(row["box_number"]) for row in target]
        if sorted(early) != boxes or sorted(latest) != boxes:
            raise ValueError(f"fixed-window incomplete corrected field: {race_id}")
        early_odds = [float(early[box]["canonical_win_odds"]) for box in boxes]
        latest_odds = [float(latest[box]["canonical_win_odds"]) for box in boxes]
        early_p = market_module.normalize(1 / np.asarray(early_odds))
        latest_p = market_module.normalize(1 / np.asarray(latest_odds))
        features = market_module.feature_vector(early_p.tolist(), latest_p.tolist())
        for index, row in enumerate(target):
            item = dict(row)
            item["market_implied_probability_original"] = float(row["market_implied_probability"])
            item["market_implied_probability"] = float(latest_p[index])
            item["early_market_implied_probability"] = float(early_p[index])
            item["canonical_early_win_odds"] = early_odds[index]
            item["canonical_latest_win_odds"] = latest_odds[index]
            for column, name in enumerate(market_module.FEATURES):
                item[name] = float(features[index, column])
            corrected.append(item)
    excluded = [
        {"race_id": race_id, "reason": "unparseable_fixed_window_win_evidence"}
        for race_id in sorted(set(old_by_race) - qualified)
    ]
    corrected_source = [row for row in audited if str(row["race_id"]) in qualified]
    return corrected, corrected_source, excluded


def run_fixed_window(
    rows: Sequence[Mapping[str, Any]],
    module: ModuleType,
    output_dir: Path,
    original: Path,
    source_db: Path,
    source_db_sha256: str,
) -> dict[str, Any]:
    folds = []
    replay = {}
    refit = {}
    final_artifacts = {}
    for fold in module.FOLDS:
        train = [row for row in rows if fold["train_start"] <= row["race_date"] <= fold["train_end"]]
        valid = [row for row in rows if fold["valid_start"] <= row["race_date"] <= fold["valid_end"]]
        market = np.asarray([float(row["market_implied_probability"]) for row in valid])
        alpha = module.fit_temperature(train)
        residual = module.fit_residual(train)
        predictions = {
            "market_baseline": module.race_probabilities(valid, market),
            "market_temperature": module.race_probabilities(valid, np.power(market, alpha)),
            "market_movement_residual_l2": module.predict_residual(valid, residual),
        }
        seed = int(module.BOOTSTRAP_SEED + list(module.FOLDS).index(fold))
        comparisons = {name: module.paired(valid, values, predictions["market_baseline"], seed) for name, values in predictions.items() if name != "market_baseline"}
        folds.append({"fold": fold, "population": {"train_races": len(module.grouped_indices(train)), "train_runner_rows": len(train), "validation_races": len(module.grouped_indices(valid)), "validation_runner_rows": len(valid)}, "candidate_results": {name: module.metrics(valid, values) for name, values in predictions.items()}, "paired_vs_market": comparisons})
        if fold["id"] == "rolling_3_final":
            final_artifacts = {"market_temperature": {"alpha": alpha}, "market_movement_residual_l2": residual}
            for name, artifact in final_artifacts.items():
                if name == "market_temperature":
                    values = module.race_probabilities(valid, np.power(market, artifact["alpha"]))
                    loaded = pickle.loads(pickle.dumps(artifact, protocol=4))
                    replayed = module.race_probabilities(valid, np.power(market, loaded["alpha"]))
                    second = module.fit_temperature(train)
                    refitted = module.race_probabilities(valid, np.power(market, second))
                else:
                    values = module.predict_residual(valid, artifact)
                    loaded = pickle.loads(pickle.dumps(artifact, protocol=4))
                    replayed = module.predict_residual(valid, loaded)
                    second = module.fit_residual(train)
                    refitted = module.predict_residual(valid, second)
                replay[name] = {"max_abs_probability_delta": float(np.max(np.abs(values - replayed))), "prediction_sha256": prediction_sha256(replayed)}
                refit[name] = {"max_abs_probability_delta": float(np.max(np.abs(values - refitted))), "prediction_sha256": prediction_sha256(refitted)}
    final = folds[-1]
    survivors = []
    for candidate in ("market_temperature", "market_movement_residual_l2"):
        comparison = final["paired_vs_market"][candidate]
        improving = sum(fold["paired_vs_market"][candidate]["delta_log_loss"] < 0 for fold in folds)
        if comparison["delta_log_loss"] < 0 and comparison["log_loss_upper_95"] < 0 and comparison["delta_brier"] <= 0 and improving >= 2:
            survivors.append(candidate)
    report = {
        "schema_version": "corrected_fixed_window_market_structure_rerun_v1",
        "branch": "fixed-window T-30/T-10 market structure",
        "status": "EXPLORATORY_SIGNAL_REQUIRES_INDEPENDENT_REPLICATION" if survivors else "NO_ROBUST_INCREMENTAL_SIGNAL",
        "original_report": str(original), "original_report_sha256": sha256_file(original),
        "source_db": {
            "path": str(source_db),
            "sha256": source_db_sha256,
        },
        "canonical_population": population(rows), "fold_results": folds, "survivors": survivors,
        "replay": replay, "deterministic_refit": refit,
        "gates": {"serialization_replay_exact": all(x["max_abs_probability_delta"] == 0 for x in replay.values()), "deterministic_refit_exact": all(x["max_abs_probability_delta"] == 0 for x in refit.values())},
        "boundaries": {"independent_fixed_window_win_repair": True, "development_repair_assumed": False, "august_opened": False, "roi_run": False},
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "final_fold_models.pkl").open("wb") as handle:
        pickle.dump(final_artifacts, handle, protocol=4)
    write_json(output_dir / "report.json", report)
    return report


def original_metric_summary(branch: str, report: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if branch == "canonical_training":
        return None, {name: value["holdout"] for name, value in report["models"].items()}
    if branch == "tier_a":
        return str(report["disposition"]), {"model": report["metrics"]["model"], "market": report["metrics"]["market"]}
    if branch == "raw_shape":
        return str(report["status"]), report["fold_results"][-1]["candidate_results"]
    if branch == "fixed_window":
        return str(report["status"]), report["fold_results"][-1]["candidate_results"]
    if branch == "fresh_v1":
        return str(report["status"]), {
            name: value["validation"] for name, value in report["candidates"].items()
        }
    return str(report.get("status")), {
        name: value["validation"] for name, value in report["candidate_results"].items()
    }


def corrected_metric_summary(branch: str, report: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if branch == "canonical_training":
        return None, {name: value["holdout"] for name, value in report["models"].items()}
    if branch == "tier_a":
        return str(report["status"]), dict(report["metrics"])
    if branch in {"raw_shape", "fixed_window"}:
        return str(report["status"]), report["fold_results"][-1]["candidate_results"]
    return str(report.get("status")), {
        name: value["validation"] for name, value in report["candidate_results"].items()
    }


def conclusion_signature(branch: str, report: Mapping[str, Any]) -> dict[str, Any]:
    status = (
        report.get("disposition", report.get("status"))
        if branch == "tier_a"
        else report.get("status")
    )
    status_aliases = {
        "SPEED_CONTEXT_CLEAN_NO_INCREMENTAL_SIGNAL": "NO_INCREMENTAL_SIGNAL_SPEED_CONTEXT",
        "FROZEN_RF_DOES_NOT_BEAT_MARKET": "FROZEN_RF_DOES_NOT_BEAT_CORRECTED_MARKET",
    }
    if branch in {"canonical_training", "fresh_v1", "fresh_v2"}:
        status = None
    elif status is not None:
        status = status_aliases.get(str(status), str(status))
    selection = report.get("selection", {}).get("selected_candidate_id")
    if branch == "canonical_training":
        selection = sorted(
            name
            for name, result in report["models"].items()
            if result.get("advancement", {}).get("advanced") is True
        )
    elif branch == "fixed_window":
        selection = sorted(report.get("survivors", []))
    return {"status": status, "selection": selection}


def conclusion_display(signature: Mapping[str, Any]) -> tuple[str | None, str | None]:
    status = signature.get("status")
    selection = signature.get("selection")
    if isinstance(selection, list):
        selection = ",".join(str(value) for value in selection) or "none"
    return (
        None if status is None else str(status),
        None if selection is None else str(selection),
    )


def metric_rows(branch: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in sorted(set(before) | set(after)):
        for metric in ("log_loss", "brier_multiclass", "top_1", "top_3", "mean_winner_rank"):
            old = before.get(candidate, {}).get(metric)
            new = after.get(candidate, {}).get(metric)
            if old is None and metric == "top_1":
                old = before.get(candidate, {}).get("top1")
            if new is None and metric == "top_1":
                new = after.get(candidate, {}).get("top1")
            if old is None and metric == "top_3":
                old = before.get(candidate, {}).get("top3")
            if new is None and metric == "top_3":
                new = after.get(candidate, {}).get("top3")
            if old is None and metric == "brier_multiclass":
                old = before.get(candidate, {}).get("multiclass_brier")
            if new is None and metric == "brier_multiclass":
                new = after.get(candidate, {}).get("multiclass_brier")
            if old is None and metric == "log_loss":
                old = before.get(candidate, {}).get("multiclass_log_loss")
            if new is None and metric == "log_loss":
                new = after.get(candidate, {}).get("multiclass_log_loss")
            if old is not None or new is not None:
                rows.append({"branch": branch, "candidate": candidate, "metric": metric, "before": old, "corrected": new, "delta": None if old is None or new is None else float(new) - float(old)})
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    recovery = args.recovery_root.resolve()
    source = args.source_root.resolve()
    expected = {
        "canonical_win_matrix.jsonl": EXPECTED_MATRIX_SHA256,
        "canonical_win_sidecar.jsonl": EXPECTED_SIDECAR_SHA256,
        "experiment_dependency_audit.csv": EXPECTED_DEPENDENCY_SHA256,
    }
    actual = {name: sha256_file(recovery / name) for name in expected}
    if actual != expected:
        raise ValueError(f"sealed recovery hash mismatch: {actual}")
    matrix_rows = load_jsonl(recovery / "canonical_win_matrix.jsonl")
    sidecar_rows = load_jsonl(recovery / "canonical_win_sidecar.jsonl")
    if len(matrix_rows) != 8234 or len({row["race_id"] for row in matrix_rows}) != 1153 or len(sidecar_rows) != 8350:
        raise ValueError("sealed recovery population mismatch")
    canonical = canonical_index(matrix_rows)
    scripts = source / "scripts"
    module_paths = {
        "canonical": scripts / "run_canonical_training_report.py",
        "v2": scripts / "fresh_model_protocol_v2.py",
        "history": scripts / "run_sportsbet_history_experiment.py",
        "speed": scripts / "run_sportsbet_speed_context_experiment.py",
        "calibration": scripts / "run_sportsbet_calibration_experiment.py",
        "raw": scripts / "run_raw_race_shape_experiment.py",
        "fixed": scripts / "run_sportsbet_market_structure_experiment.py",
        "audit": Path(__file__).with_name("audit_sportsbet_win_market_surface.py"),
        "tier": args.tier_a_root / "evaluate_frozen_rf_vs_market.py",
    }
    source_script_hashes = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in module_paths.items()
    }
    modules = {
        name: load_module(f"frozen_{name}", path)
        for name, path in module_paths.items()
    }
    artifacts = source / "artifacts"
    originals = {
        "canonical_training": artifacts / "full_evidence_orchestration_20260525/canonical_training_model_experiment_20260812T_report_only/report.json",
        "tier_a": args.tier_a_root / "evaluation_results.json",
        "fresh_v1": artifacts / "fresh_model_protocol_v1_model_frozen_20260814_v2_report_only/report.json",
        "fresh_v2": artifacts / "fresh_model_protocol_v2_report_only/report.json",
        "basic_history": artifacts / "sportsbet_history_experiment_20260815_v2_report_only/report.json",
        "speed_context": artifacts / "sportsbet_speed_context_experiment_20260815_clean_rerun_report_only/report.json",
        "calibration": artifacts / "sportsbet_calibration_experiment_20260815_report_only/report.json",
        "raw_shape": artifacts / "raw_race_shape_experiment_20260815_report_only/report.json",
        "fixed_window": artifacts / "prejump_market_structure_experiment_20260815_report_only/report.json",
    }
    for path in originals.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    source_db = source / ".scratch/development_source_20260802.db"
    source_db_sha256 = sha256_file(source_db)
    if source_db_sha256 != EXPECTED_SOURCE_DB_SHA256:
        raise ValueError(f"sealed source DB hash mismatch: {source_db_sha256}")
    reports: dict[str, dict[str, Any]] = {}
    reports["canonical_training"] = run_canonical_training(matrix_rows, modules["canonical"], output / "01_canonical_training", originals["canonical_training"])
    reports["tier_a"] = run_tier_a(
        modules["tier"],
        modules["audit"],
        source_db,
        source_db_sha256,
        output / "02_tier_a_frozen_rf",
        originals["tier_a"],
    )
    if sha256_file(source_db) != source_db_sha256:
        raise ValueError("sealed source DB changed during Tier-A correction")
    fresh_v1_original = json.loads(originals["fresh_v1"].read_text())
    v1_candidates = {
        name: (
            list(spec["features"]),
            "logistic" if name.endswith("logistic") else "hist_gradient_boosting",
        )
        for name, spec in fresh_v1_original["candidates"].items()
        if name != "market_baseline"
    }
    reports["fresh_v1"] = run_classifier_protocol(branch="fresh model protocol v1 frozen selection / market baseline", rows=matrix_rows, module=modules["v2"], candidates=v1_candidates, market_metric_name="metric", repetitions=500, seed=20260813, output_dir=output / "03_fresh_model_v1", original_report=originals["fresh_v1"])
    v2_candidates = {
        "history_only_logistic": (modules["v2"].HISTORY, "logistic"),
        "market_plus_history_logistic": (modules["v2"].ALL_FEATURES, "logistic"),
        "market_plus_history_hist_gradient_boosting": (
            modules["v2"].ALL_FEATURES,
            "hist_gradient_boosting",
        ),
    }
    reports["fresh_v2"] = run_classifier_protocol(branch="fresh model protocol v2 frozen selection", rows=matrix_rows, module=modules["v2"], candidates=v2_candidates, market_metric_name="metric", repetitions=500, seed=20260813, output_dir=output / "04_fresh_model_v2", original_report=originals["fresh_v2"])
    history_old = load_jsonl(artifacts / "sportsbet_history_experiment_20260815_v2_report_only/enriched_development_matrix.jsonl")
    history_rows = corrected_intersection(history_old, canonical)
    history_candidates = {
        "history_only_logistic": (modules["history"].HISTORY_FEATURES, "logistic"),
        "market_plus_history_logistic": (modules["history"].ALL_FEATURES, "logistic"),
        "market_plus_history_hist_gradient_boosting": (
            modules["history"].ALL_FEATURES,
            "hist_gradient_boosting",
        ),
    }
    reports["basic_history"] = run_classifier_protocol(branch="basic Sportsbet history v2", rows=history_rows, module=modules["history"], candidates=history_candidates, market_metric_name="metrics", repetitions=2000, seed=20260816, output_dir=output / "05_basic_history", original_report=originals["basic_history"])
    speed_old = load_jsonl(artifacts / "sportsbet_speed_context_experiment_20260815_clean_rerun_report_only/enriched_development_matrix.jsonl")
    speed_rows = corrected_intersection(speed_old, canonical)
    speed_candidates = {
        "speed_context_logistic": (modules["speed"].SPEED, "logistic"),
        "market_plus_speed_context_logistic": (
            modules["speed"].MARKET + modules["speed"].SPEED,
            "logistic",
        ),
        "market_plus_speed_context_hist_gradient_boosting": (
            modules["speed"].MARKET + modules["speed"].NUMERIC,
            "hist_gradient_boosting",
        ),
    }
    reports["speed_context"] = run_classifier_protocol(branch="clean speed/context rerun", rows=speed_rows, module=modules["speed"], candidates=speed_candidates, market_metric_name="metrics", repetitions=2000, seed=20260816, output_dir=output / "06_speed_context", original_report=originals["speed_context"])
    reports["calibration"] = run_calibration(history_rows, modules["calibration"], output / "07_calibration", originals["calibration"])
    raw_old = load_jsonl(artifacts / "raw_race_shape_experiment_20260815_report_only/feature_matrix.jsonl")
    raw_rows = corrected_intersection(raw_old, canonical)
    reports["raw_shape"] = run_raw_shape(raw_rows, modules["raw"], output / "08_raw_race_shape", originals["raw_shape"])
    fixed_root = artifacts / "prejump_market_structure_experiment_20260815_report_only"
    fixed_old = load_jsonl(fixed_root / "movement_development_matrix.jsonl")
    fixed_source = load_jsonl(fixed_root / "sportsbet_source_extract.jsonl")
    if sha256_file(source_db) != source_db_sha256:
        raise ValueError("sealed source DB changed before fixed-window correction")
    fixed_rows, corrected_source, fixed_exclusions = corrected_fixed_window(old_rows=fixed_old, source_extract=fixed_source, source_db=source_db, audit_module=modules["audit"], market_module=modules["fixed"])
    if sha256_file(source_db) != source_db_sha256:
        raise ValueError("fixed-window source DB changed during correction")
    fixed_out = output / "09_fixed_window_market_structure"
    reports["fixed_window"] = run_fixed_window(
        fixed_rows,
        modules["fixed"],
        fixed_out,
        originals["fixed_window"],
        source_db,
        source_db_sha256,
    )
    write_jsonl(fixed_out / "corrected_sportsbet_source_extract.jsonl", corrected_source)
    write_jsonl(fixed_out / "corrected_movement_development_matrix.jsonl", fixed_rows)
    write_jsonl(fixed_out / "exclusions.jsonl", fixed_exclusions)
    dependency_rows = list(csv.DictReader((recovery / "experiment_dependency_audit.csv").open()))
    branch_lookup = {
        "canonical training model experiment 20260812": "canonical_training",
        "historical Tier-A frozen-RF vs WIN-market evaluation 20260716": "tier_a",
        "fresh model protocol v1 frozen selection / market baseline": "fresh_v1",
        "fresh model protocol v2 frozen selection": "fresh_v2",
        "basic Sportsbet history v2": "basic_history",
        "clean speed/context rerun": "speed_context",
        "Sportsbet calibration": "calibration",
        "raw race-shape Sportsbet baseline": "raw_shape",
        "fixed-window T-30/T-10 market structure": "fixed_window",
    }
    refit_policy = {
        "canonical_training": "YES_population_and_market_feature_changed",
        "tier_a": "NO_frozen_model_predictions_reused",
        "fresh_v1": "YES_population_changed",
        "fresh_v2": "YES_population_and_market_feature_changed",
        "basic_history": "YES_population_and_market_feature_changed",
        "speed_context": "YES_population_and_market_feature_changed",
        "calibration": "YES_calibrators_consume_market_probability",
        "raw_shape": "YES_market_offset_models_consume_market_probability",
        "fixed_window": "YES_independently_corrected_T30_T10_features",
    }
    manifest = []
    for row in dependency_rows:
        experiment = row["experiment"]
        branch = branch_lookup.get(experiment)
        if branch is None:
            action = "PRESERVE_NOT_RERUN_SUPERSEDED" if experiment in {"basic Sportsbet history initial run", "speed/context initial run"} else "PRESERVE_NOT_RERUN_VALID_UNAFFECTED"
            canonical_population = None
            refit_required = "NO"
            corrected_input = None
        else:
            action = "RERUN_COMPLETE"
            canonical_population = reports[branch]["canonical_population"]
            refit_required = refit_policy[branch]
            if branch == "fixed_window":
                corrected_input = "independent_fixed_window_corrected_source"
            elif branch == "tier_a":
                corrected_input = "independent_tier_a_corrected_win_evidence"
            else:
                corrected_input = "canonical_win_matrix"
        manifest.append({
            "experiment": experiment,
            "dependency_classification": row["classification"],
            "action": action,
            "original_races": int(row["exact_input_races"]),
            "original_runner_rows": int(row["exact_input_runner_rows"]),
            "canonical_races": None if canonical_population is None else canonical_population["races"],
            "canonical_runner_rows": None if canonical_population is None else canonical_population["runner_rows"],
            "canonical_race_ids_sha256": None if canonical_population is None else canonical_population["race_ids_sha256"],
            "canonical_intersection_sha256": None if canonical_population is None else canonical_population["canonical_intersection_sha256"],
            "corrected_win_input": corrected_input,
            "model_refit_required": refit_required,
        })
    write_csv(output / "rerun_manifest.csv", manifest, list(manifest[0]))
    before_after = []
    verdicts = []
    for branch, corrected in reports.items():
        original_report = json.loads(originals[branch].read_text())
        old_verdict, old_metrics = original_metric_summary(branch, original_report)
        new_verdict, new_metrics = corrected_metric_summary(branch, corrected)
        before_after.extend(metric_rows(branch, old_metrics, new_metrics))
        old_signature = conclusion_signature(branch, original_report)
        new_signature = conclusion_signature(branch, corrected)
        _, old_selection = conclusion_display(old_signature)
        _, new_selection = conclusion_display(new_signature)
        verdicts.append({
            "branch": branch,
            "before_verdict": old_verdict,
            "corrected_verdict": new_verdict,
            "before_selection": old_selection,
            "corrected_selection": new_selection,
            "changed": old_signature != new_signature,
        })
    write_csv(output / "before_vs_corrected_metrics.csv", before_after, ["branch", "candidate", "metric", "before", "corrected", "delta"])
    write_csv(
        output / "model_selection_verdicts.csv",
        verdicts,
        [
            "branch",
            "before_verdict",
            "corrected_verdict",
            "before_selection",
            "corrected_selection",
            "changed",
        ],
    )
    all_gates = all(all(report.get("gates", {}).values()) for report in reports.values() if report.get("gates"))
    conclusions_changed = any(row["changed"] for row in verdicts)
    summary = {
        "schema_version": "corrected_sportsbet_win_rerun_programme_v1",
        "analysis_date": ANALYSIS_DATE,
        "status": (
            "BLOCKED_RERUN_EVIDENCE"
            if not all_gates
            else "CORRECTED_MARKET_CHANGES_MODEL_CONCLUSIONS"
            if conclusions_changed
            else "CORRECTED_MARKET_BASELINE_CONFIRMED"
        ),
        "sealed_recovery_hashes": actual,
        "frozen_source_script_hashes": source_script_hashes,
        "fixed_window_source_db": {
            "path": str(source_db.resolve()),
            "sha256": source_db_sha256,
        },
        "canonical_population": population(matrix_rows),
        "tier_a_population": reports["tier_a"]["canonical_population"],
        "fixed_window_population": reports["fixed_window"]["canonical_population"],
        "executed_branches": list(reports),
        "preserved_not_rerun": ["basic Sportsbet history initial run", "speed/context initial run", "favourite benchmark"],
        "branch_reports": {name: report["status"] for name, report in reports.items()},
        "replay_and_refit_gates_passed": all_gates,
        "favourite_roi_run": False,
        "august_opened_or_rescored": False,
        "new_forward_cohort_opened": False,
        "programme_status_rule": "CONFIRMED only if corrected baseline conclusions remain non-promotional and all replay/refit gates pass; otherwise CHANGES or BLOCKED",
    }
    write_json(output / "report.json", summary)
    report_lines = [
        "# Corrected Sportsbet WIN experiment reruns", "",
        f"Programme status: `{summary['status']}`", "",
        f"Canonical development surface: {summary['canonical_population']['races']:,} races / {summary['canonical_population']['runner_rows']:,} runners.",
        f"Fixed-window corrected surface: {summary['fixed_window_population']['races']:,} races / {summary['fixed_window_population']['runner_rows']:,} runners.",
        "", "Every prior artifact remains unchanged. August was not opened or rescored, no forward cohort was created, and favourite ROI was not run.",
        "", "See `rerun_manifest.csv`, `before_vs_corrected_metrics.csv`, `model_selection_verdicts.csv`, and the nine branch reports for exact populations, metrics, paired uncertainty, and replay evidence.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    seal(output)
    return summary


def seal(output: Path) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output)}\n" for path in files),
        encoding="utf-8",
    )


def verify(output: Path) -> dict[str, Any]:
    manifest = output / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = sha256_file(output / relative)
        if actual != expected:
            raise ValueError(f"sealed artifact drift: {relative}")
        checked += 1
    report = json.loads((output / "report.json").read_text())
    if not report["replay_and_refit_gates_passed"]:
        raise ValueError("embedded replay/refit gate failed")
    return {"status": "DETERMINISTIC_REPLAY_VERIFIED", "sealed_files": checked, "programme_status": report["status"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--tier-a-root", type=Path, default=DEFAULT_TIER_A_ROOT)
    parser.add_argument(
        "--recovery-root",
        type=Path,
        default=Path("artifacts/sportsbet_win_market_surface_audit_20260815_report_only"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_all(args) if args.action == "run" else verify(args.output.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
