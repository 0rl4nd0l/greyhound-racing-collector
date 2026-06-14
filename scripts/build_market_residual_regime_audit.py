#!/usr/bin/env python3
"""Build a report-only market residual regime audit.

This consumes the runner-level market residual matrix and the cross-validated
race prediction CSV emitted by the residual challenger packet. It summarizes
where market-only, the CV challenger, current non-market blend, and raw Stage 2
signals help or hurt.

The output is diagnostic only. It does not train, promote, mutate registries,
update pointers, write DB labels/odds, emit EV, place bets, rewrite snapshots or
manifests, or enable TGR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "market_residual_regime_audit_"
)
REPORT_FILE = "market_residual_regime_audit_report.json"
LEDGER_CSV = "race_regime_ledger.csv"
SUMMARY_CSV = "regime_summary.csv"
HYPOTHESES_JSON = "next_hypotheses.json"
SUMMARY_MD = "SUMMARY.md"
FINAL_READY = "MARKET_RESIDUAL_REGIME_AUDIT_READY"
FINAL_COLLECTING = "MARKET_RESIDUAL_REGIME_AUDIT_COLLECTING"
MIN_RACES_FOR_REVIEW = 100
MIN_REGIME_RACES = 5
NO_WRITE_GUARANTEES = {
    "training_production_model": False,
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

DIMENSIONS = [
    {
        "name": "market_favourite_odds_band",
        "pre_race_usable": True,
        "description": "Market favourite price band from pre-jump odds.",
    },
    {
        "name": "market_favourite_odds_group",
        "pre_race_usable": True,
        "description": "Computed market favourite price group.",
    },
    {
        "name": "runner_count",
        "pre_race_usable": True,
        "description": "Runner count after exact runner-set validation.",
    },
    {
        "name": "selected_candidate_key",
        "pre_race_usable": True,
        "description": "Fold-selected CV challenger family.",
    },
    {
        "name": "current_candidate_agrees_market_top",
        "pre_race_usable": True,
        "description": "Whether current non-market top pick matches market top pick.",
    },
    {
        "name": "stage2_uncalibrated_agrees_market_top",
        "pre_race_usable": True,
        "description": "Whether raw Stage 2 top pick matches market top pick.",
    },
    {
        "name": "venue",
        "pre_race_usable": True,
        "description": "Venue code from the exact race identity.",
    },
    {
        "name": "winner_odds_band",
        "pre_race_usable": False,
        "description": "Outcome-observed winner price band; diagnostic only.",
    },
    {
        "name": "market_winner_rank_band",
        "pre_race_usable": False,
        "description": "Outcome-observed rank of the winner under market-only.",
    },
    {
        "name": "winner_box_number",
        "pre_race_usable": False,
        "description": "Outcome-observed winning box; diagnostic only.",
    },
]


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_market_residual_regime_audit:{relative}")
    return logical.absolute()


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
        "schema_version": "market_residual_regime_audit_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def odds_group(value: Any, *, prefix: str) -> str:
    odds = finite_float(value)
    if odds is None:
        return f"{prefix}_missing"
    if odds <= 2.0:
        return f"{prefix}_lte_2"
    if odds <= 4.0:
        return f"{prefix}_2_4"
    if odds <= 8.0:
        return f"{prefix}_4_8"
    return f"{prefix}_gt_8"


def rank_band(value: Any, *, prefix: str) -> str:
    rank = finite_int(value)
    if rank is None:
        return f"{prefix}_missing"
    if rank == 1:
        return f"{prefix}_1"
    if rank <= 3:
        return f"{prefix}_2_3"
    return f"{prefix}_gt_3"


def probability_logloss_delta(candidate_probability: Any, market_probability: Any) -> float | None:
    candidate = finite_float(candidate_probability)
    market = finite_float(market_probability)
    if candidate is None or market is None:
        return None
    if candidate <= 0 or market <= 0:
        return None
    return -math.log(max(candidate, 1e-15)) + math.log(max(market, 1e-15))


def rank_delta(candidate_rank: Any, market_rank: Any) -> int | None:
    candidate = finite_int(candidate_rank)
    market = finite_int(market_rank)
    if candidate is None or market is None:
        return None
    return candidate - market


def top1_delta(candidate_rank: Any, market_rank: Any) -> int | None:
    candidate = finite_int(candidate_rank)
    market = finite_int(market_rank)
    if candidate is None or market is None:
        return None
    return int(candidate == 1) - int(market == 1)


def top_pick(rows: Sequence[Mapping[str, Any]], rank_key: str, flag_key: str | None = None) -> Mapping[str, Any] | None:
    if flag_key:
        flagged = [row for row in rows if parse_bool(row.get(flag_key))]
        if flagged:
            return sorted(
                flagged,
                key=lambda row: (finite_int(row.get("box_number")) or 999, str(row.get("dog_name") or "")),
            )[0]
    ranked = [row for row in rows if finite_int(row.get(rank_key)) is not None]
    if not ranked:
        return None
    return sorted(
        ranked,
        key=lambda row: (
            finite_int(row.get(rank_key)) or 999,
            finite_int(row.get("box_number")) or 999,
            str(row.get("dog_name") or ""),
        ),
    )[0]


def build_race_ledger(
    *,
    matrix_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions_by_race = {
        str(row.get("race_id") or ""): dict(row)
        for row in prediction_rows
        if str(row.get("race_id") or "")
    }
    matrix_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix_rows:
        race_id = str(row.get("race_id") or "").strip()
        if race_id:
            matrix_by_race[race_id].append(dict(row))

    skipped = Counter()
    ledger: list[dict[str, Any]] = []
    for race_id, rows in sorted(matrix_by_race.items()):
        prediction = predictions_by_race.get(race_id)
        if not prediction:
            skipped["race_missing_cv_prediction_row"] += 1
            continue
        winners = [row for row in rows if parse_bool(row.get("is_winner"))]
        if len(winners) != 1:
            skipped["race_winner_count_not_one"] += 1
            continue
        winner = winners[0]
        market_top = top_pick(rows, "market_rank", "market_top_pick")
        current_top = top_pick(rows, "candidate_rank", "candidate_top_pick")
        stage2_uncalibrated_top = top_pick(rows, "stage2_shadow_uncalibrated_rank")
        if market_top is None:
            skipped["race_missing_market_top_pick"] += 1
            continue

        market_rank = finite_int(prediction.get("market_winner_rank")) or finite_int(
            winner.get("market_rank")
        )
        challenger_rank = finite_int(prediction.get("challenger_winner_rank"))
        current_rank = finite_int(winner.get("candidate_rank"))
        stage2_uncalibrated_rank = finite_int(winner.get("stage2_shadow_uncalibrated_rank"))
        stage2_rank = finite_int(winner.get("stage2_shadow_rank"))
        primary_rank = finite_int(winner.get("primary_shadow_rank"))
        current_top_box = finite_int((current_top or {}).get("box_number"))
        market_top_box = finite_int(market_top.get("box_number"))
        stage2_top_box = finite_int((stage2_uncalibrated_top or {}).get("box_number"))

        ledger.append(
            {
                "race_id": race_id,
                "race_date": rows[0].get("race_date"),
                "venue": rows[0].get("venue"),
                "race_number": finite_int(rows[0].get("race_number")),
                "runner_count": finite_int(rows[0].get("runner_count")) or len(rows),
                "selected_candidate_key": prediction.get("selected_candidate_key"),
                "market_favourite_odds_decimal": finite_float(
                    rows[0].get("market_favourite_odds_decimal")
                ),
                "market_favourite_odds_band": rows[0].get("market_favourite_odds_band")
                or odds_group(
                    rows[0].get("market_favourite_odds_decimal"),
                    prefix="market_favourite_odds",
                ),
                "market_favourite_odds_group": odds_group(
                    rows[0].get("market_favourite_odds_decimal"),
                    prefix="market_favourite_odds",
                ),
                "winner_dog_name": winner.get("dog_name"),
                "winner_box_number": finite_int(winner.get("box_number")),
                "winner_odds_decimal": finite_float(winner.get("odds_decimal")),
                "winner_odds_band": odds_group(winner.get("odds_decimal"), prefix="winner_odds"),
                "market_winner_rank": market_rank,
                "market_winner_rank_band": rank_band(market_rank, prefix="market_winner_rank"),
                "challenger_winner_rank": challenger_rank,
                "current_candidate_winner_rank": current_rank,
                "stage2_uncalibrated_winner_rank": stage2_uncalibrated_rank,
                "stage2_shadow_winner_rank": stage2_rank,
                "primary_shadow_winner_rank": primary_rank,
                "challenger_minus_market_logloss": finite_float(
                    prediction.get("challenger_minus_market_logloss")
                ),
                "current_candidate_minus_market_logloss": probability_logloss_delta(
                    winner.get("candidate_probability"),
                    winner.get("market_probability"),
                ),
                "stage2_uncalibrated_minus_market_logloss": probability_logloss_delta(
                    winner.get("stage2_shadow_uncalibrated_probability_norm"),
                    winner.get("market_probability"),
                ),
                "stage2_shadow_minus_market_logloss": probability_logloss_delta(
                    winner.get("stage2_shadow_probability_norm"),
                    winner.get("market_probability"),
                ),
                "primary_shadow_minus_market_logloss": probability_logloss_delta(
                    winner.get("primary_shadow_probability_norm"),
                    winner.get("market_probability"),
                ),
                "challenger_top1_delta": top1_delta(challenger_rank, market_rank),
                "current_candidate_top1_delta": top1_delta(current_rank, market_rank),
                "stage2_uncalibrated_top1_delta": top1_delta(stage2_uncalibrated_rank, market_rank),
                "stage2_shadow_top1_delta": top1_delta(stage2_rank, market_rank),
                "primary_shadow_top1_delta": top1_delta(primary_rank, market_rank),
                "challenger_rank_delta": rank_delta(challenger_rank, market_rank),
                "current_candidate_rank_delta": rank_delta(current_rank, market_rank),
                "stage2_uncalibrated_rank_delta": rank_delta(stage2_uncalibrated_rank, market_rank),
                "stage2_shadow_rank_delta": rank_delta(stage2_rank, market_rank),
                "primary_shadow_rank_delta": rank_delta(primary_rank, market_rank),
                "market_top_box_number": market_top_box,
                "current_candidate_top_box_number": current_top_box,
                "stage2_uncalibrated_top_box_number": stage2_top_box,
                "current_candidate_agrees_market_top": current_top_box == market_top_box,
                "stage2_uncalibrated_agrees_market_top": stage2_top_box == market_top_box,
            }
        )

    return ledger, {
        "matrix_races": len(matrix_by_race),
        "prediction_races": len(predictions_by_race),
        "accepted_races": len(ledger),
        "skipped_counts": dict(sorted(skipped.items())),
    }


def mean(values: Sequence[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def positive_count(values: Sequence[float | int | None]) -> int:
    return sum(1 for value in values if value is not None and float(value) > 0)


def negative_count(values: Sequence[float | int | None]) -> int:
    return sum(1 for value in values if value is not None and float(value) < 0)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    challenger_logloss = [finite_float(row.get("challenger_minus_market_logloss")) for row in rows]
    current_logloss = [
        finite_float(row.get("current_candidate_minus_market_logloss")) for row in rows
    ]
    stage2_uncalibrated_logloss = [
        finite_float(row.get("stage2_uncalibrated_minus_market_logloss")) for row in rows
    ]
    challenger_top1 = [finite_float(row.get("challenger_top1_delta")) for row in rows]
    current_top1 = [finite_float(row.get("current_candidate_top1_delta")) for row in rows]
    stage2_top1 = [finite_float(row.get("stage2_uncalibrated_top1_delta")) for row in rows]
    challenger_rank = [finite_float(row.get("challenger_rank_delta")) for row in rows]
    current_rank = [finite_float(row.get("current_candidate_rank_delta")) for row in rows]
    stage2_rank = [finite_float(row.get("stage2_uncalibrated_rank_delta")) for row in rows]
    market_top1 = [
        1.0 if finite_int(row.get("market_winner_rank")) == 1 else 0.0
        for row in rows
        if finite_int(row.get("market_winner_rank")) is not None
    ]
    return {
        "race_count": len(rows),
        "market_top1_rate": mean(market_top1),
        "market_miss_top1_count": sum(
            1 for row in rows if finite_int(row.get("market_winner_rank")) not in {None, 1}
        ),
        "challenger_mean_logloss_delta": mean(challenger_logloss),
        "challenger_mean_top1_delta": mean(challenger_top1),
        "challenger_mean_rank_delta": mean(challenger_rank),
        "challenger_better_logloss_count": negative_count(challenger_logloss),
        "challenger_worse_logloss_count": positive_count(challenger_logloss),
        "current_candidate_mean_logloss_delta": mean(current_logloss),
        "current_candidate_mean_top1_delta": mean(current_top1),
        "current_candidate_mean_rank_delta": mean(current_rank),
        "current_candidate_better_logloss_count": negative_count(current_logloss),
        "current_candidate_worse_logloss_count": positive_count(current_logloss),
        "stage2_uncalibrated_mean_logloss_delta": mean(stage2_uncalibrated_logloss),
        "stage2_uncalibrated_mean_top1_delta": mean(stage2_top1),
        "stage2_uncalibrated_mean_rank_delta": mean(stage2_rank),
        "stage2_uncalibrated_better_logloss_count": negative_count(stage2_uncalibrated_logloss),
        "stage2_uncalibrated_worse_logloss_count": positive_count(stage2_uncalibrated_logloss),
    }


def summarize_regimes(
    ledger: Sequence[Mapping[str, Any]],
    *,
    min_regime_races: int,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    dimension_meta = {item["name"]: item for item in DIMENSIONS}
    for dimension in DIMENSIONS:
        name = str(dimension["name"])
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in ledger:
            value = row.get(name)
            grouped[str(value if value not in {None, ""} else "missing")].append(row)
        for value, rows in grouped.items():
            if len(rows) < min_regime_races:
                continue
            summary = summarize_rows(rows)
            row_out = {
                "dimension": name,
                "dimension_value": value,
                "pre_race_usable": bool(dimension_meta[name]["pre_race_usable"]),
                "description": dimension_meta[name]["description"],
            }
            row_out.update(summary)
            rows_out.append(row_out)
    rows_out.sort(
        key=lambda row: (
            str(row["dimension"]),
            -(finite_int(row.get("race_count")) or 0),
            str(row["dimension_value"]),
        )
    )
    return rows_out


def sort_help(row: Mapping[str, Any], key: str) -> tuple[float, int, str]:
    value = finite_float(row.get(key))
    return (
        value if value is not None else 999.0,
        -(finite_int(row.get("race_count")) or 0),
        str(row.get("dimension_value") or ""),
    )


def max_rank_first_delta(row: Mapping[str, Any]) -> float:
    values = [
        finite_float(row.get("current_candidate_mean_top1_delta")),
        finite_float(row.get("stage2_uncalibrated_mean_top1_delta")),
        finite_float(row.get("challenger_mean_top1_delta")),
    ]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return 0.0
    return max(finite_values)


def best_logloss_delta(row: Mapping[str, Any]) -> float:
    values = [
        finite_float(row.get("current_candidate_mean_logloss_delta")),
        finite_float(row.get("stage2_uncalibrated_mean_logloss_delta")),
        finite_float(row.get("challenger_mean_logloss_delta")),
    ]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return 999.0
    return min(finite_values)


def derive_hypotheses(
    regime_rows: Sequence[Mapping[str, Any]],
    *,
    min_regime_races: int,
) -> dict[str, Any]:
    pre_race = [
        row
        for row in regime_rows
        if row.get("pre_race_usable") is True
        and (finite_int(row.get("race_count")) or 0) >= min_regime_races
    ]
    outcome_only = [
        row
        for row in regime_rows
        if row.get("pre_race_usable") is False
        and (finite_int(row.get("race_count")) or 0) >= min_regime_races
    ]

    candidate_help = [
        row
        for row in pre_race
        if (finite_float(row.get("current_candidate_mean_logloss_delta")) or 999.0) < 0.0
        or (finite_float(row.get("stage2_uncalibrated_mean_logloss_delta")) or 999.0) < 0.0
    ]
    candidate_harm = [
        row
        for row in pre_race
        if (finite_float(row.get("current_candidate_mean_logloss_delta")) or -999.0) > 0.0
        or (finite_float(row.get("stage2_uncalibrated_mean_logloss_delta")) or -999.0) > 0.0
    ]
    cv_help = [
        row
        for row in pre_race
        if (finite_float(row.get("challenger_mean_logloss_delta")) or 999.0) < 0.0
    ]
    diagnostic_help = [
        row
        for row in outcome_only
        if (finite_float(row.get("stage2_uncalibrated_mean_logloss_delta")) or 999.0) < 0.0
        or (finite_float(row.get("current_candidate_mean_logloss_delta")) or 999.0) < 0.0
    ]
    rank_first_help = [
        row
        for row in pre_race
        if max_rank_first_delta(row) > 0.0
    ]
    logloss_only_help = [
        row
        for row in candidate_help
        if max_rank_first_delta(row) <= 0.0
    ]
    rank_first_status = (
        "PRE_RACE_RANK_FIRST_EDGE_CANDIDATE_FOUND"
        if rank_first_help
        else "NO_PRE_RACE_RANK_FIRST_EDGE_FOUND"
    )
    rank_first_blockers = []
    if not rank_first_help:
        rank_first_blockers.append("no_pre_race_usable_positive_top1_delta")

    return {
        "schema_version": "market_residual_regime_next_hypotheses_v1",
        "minimum_regime_races": min_regime_races,
        "promotion_ready": False,
        "rank_first_readiness": {
            "schema_version": "market_residual_rank_first_readiness_v1",
            "status": rank_first_status,
            "promotion_ready": False,
            "minimum_regime_races": min_regime_races,
            "pre_race_rank_first_help_regime_count": len(rank_first_help),
            "pre_race_logloss_only_help_regime_count": len(logloss_only_help),
            "blockers": rank_first_blockers,
        },
        "notes": [
            "Outcome-only diagnostic dimensions must not be used as deployment gates.",
            "Pre-race hypotheses require a new out-of-sample packet before promotion review.",
        ],
        "pre_race_rank_first_help_regimes": sorted(
            rank_first_help,
            key=lambda row: (
                -max_rank_first_delta(row),
                -(finite_int(row.get("race_count")) or 0),
                best_logloss_delta(row),
                str(row.get("dimension") or ""),
                str(row.get("dimension_value") or ""),
            ),
        )[:10],
        "pre_race_logloss_only_help_regimes": sorted(
            logloss_only_help,
            key=lambda row: (
                best_logloss_delta(row),
                -(finite_int(row.get("race_count")) or 0),
                str(row.get("dimension") or ""),
                str(row.get("dimension_value") or ""),
            ),
        )[:10],
        "pre_race_candidate_help_regimes": sorted(
            candidate_help,
            key=lambda row: min(
                finite_float(row.get("current_candidate_mean_logloss_delta")) or 999.0,
                finite_float(row.get("stage2_uncalibrated_mean_logloss_delta")) or 999.0,
            ),
        )[:10],
        "pre_race_candidate_harm_regimes": sorted(
            candidate_harm,
            key=lambda row: max(
                finite_float(row.get("current_candidate_mean_logloss_delta")) or -999.0,
                finite_float(row.get("stage2_uncalibrated_mean_logloss_delta")) or -999.0,
            ),
            reverse=True,
        )[:10],
        "pre_race_cv_challenger_help_regimes": sorted(
            cv_help,
            key=lambda row: sort_help(row, "challenger_mean_logloss_delta"),
        )[:10],
        "outcome_only_diagnostic_help_regimes": sorted(
            diagnostic_help,
            key=lambda row: min(
                finite_float(row.get("current_candidate_mean_logloss_delta")) or 999.0,
                finite_float(row.get("stage2_uncalibrated_mean_logloss_delta")) or 999.0,
            ),
        )[:10],
        "recommended_next_tests": [
            {
                "name": "pre_race_gate_from_positive_regime_only",
                "status": (
                    "REQUIRES_NEW_REPORT_ONLY_PACKET"
                    if rank_first_help
                    else "WAIT_FOR_RANK_FIRST_EDGE_OR_NEW_HYPOTHESIS"
                ),
                "description": (
                    "Only test a bounded pre-race gate when a pre-race regime has "
                    "positive Top1 delta; logloss-only improvements are not enough "
                    "for the current rank-first promotion objective."
                ),
            },
            {
                "name": "do_not_use_outcome_only_winner_odds_gate",
                "status": "FORBIDDEN_FOR_DEPLOYMENT",
                "description": (
                    "Winner odds/rank bands explain residuals after the race but are "
                    "not available as a deployment selector."
                ),
            },
        ],
    }


LEDGER_FIELDS = [
    "race_id",
    "race_date",
    "venue",
    "race_number",
    "runner_count",
    "selected_candidate_key",
    "market_favourite_odds_decimal",
    "market_favourite_odds_band",
    "market_favourite_odds_group",
    "winner_dog_name",
    "winner_box_number",
    "winner_odds_decimal",
    "winner_odds_band",
    "market_winner_rank",
    "market_winner_rank_band",
    "challenger_winner_rank",
    "current_candidate_winner_rank",
    "stage2_uncalibrated_winner_rank",
    "stage2_shadow_winner_rank",
    "primary_shadow_winner_rank",
    "challenger_minus_market_logloss",
    "current_candidate_minus_market_logloss",
    "stage2_uncalibrated_minus_market_logloss",
    "stage2_shadow_minus_market_logloss",
    "primary_shadow_minus_market_logloss",
    "challenger_top1_delta",
    "current_candidate_top1_delta",
    "stage2_uncalibrated_top1_delta",
    "stage2_shadow_top1_delta",
    "primary_shadow_top1_delta",
    "challenger_rank_delta",
    "current_candidate_rank_delta",
    "stage2_uncalibrated_rank_delta",
    "stage2_shadow_rank_delta",
    "primary_shadow_rank_delta",
    "market_top_box_number",
    "current_candidate_top_box_number",
    "stage2_uncalibrated_top_box_number",
    "current_candidate_agrees_market_top",
    "stage2_uncalibrated_agrees_market_top",
]

SUMMARY_FIELDS = [
    "dimension",
    "dimension_value",
    "pre_race_usable",
    "description",
    "race_count",
    "market_top1_rate",
    "market_miss_top1_count",
    "challenger_mean_logloss_delta",
    "challenger_mean_top1_delta",
    "challenger_mean_rank_delta",
    "challenger_better_logloss_count",
    "challenger_worse_logloss_count",
    "current_candidate_mean_logloss_delta",
    "current_candidate_mean_top1_delta",
    "current_candidate_mean_rank_delta",
    "current_candidate_better_logloss_count",
    "current_candidate_worse_logloss_count",
    "stage2_uncalibrated_mean_logloss_delta",
    "stage2_uncalibrated_mean_top1_delta",
    "stage2_uncalibrated_mean_rank_delta",
    "stage2_uncalibrated_better_logloss_count",
    "stage2_uncalibrated_worse_logloss_count",
]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summarize_markdown(report: Mapping[str, Any]) -> str:
    overall = report.get("overall_metrics") or {}
    return "\n".join(
        [
            "# Market Residual Regime Audit",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Accepted races: `{report.get('accepted_race_count')}` / `{report.get('minimum_races_for_review')}`",
            f"- Matrix runner rows: `{report.get('matrix_row_count')}`",
            f"- CV prediction rows: `{report.get('prediction_row_count')}`",
            f"- Regime summaries: `{report.get('regime_summary_count')}`",
            f"- Challenger logloss delta: `{overall.get('challenger_mean_logloss_delta')}`",
            f"- Current candidate logloss delta: `{overall.get('current_candidate_mean_logloss_delta')}`",
            f"- Raw Stage 2 uncalibrated logloss delta: `{overall.get('stage2_uncalibrated_mean_logloss_delta')}`",
            f"- Rank-first hypothesis status: `{report.get('rank_first_hypothesis_status')}`",
            f"- Pre-race rank-first help regimes: `{report.get('pre_race_rank_first_help_regime_count')}`",
            f"- Pre-race logloss-only help regimes: `{report.get('pre_race_logloss_only_help_regime_count')}`",
            f"- Promotion ready: `{report.get('promotion_ready')}`",
            "",
            "This is a diagnostic artifact only. It marks outcome-only dimensions as non-deployable and performs no production mutation.",
            "",
        ]
    )


def build_audit(
    *,
    runner_matrix_csv: Path,
    race_predictions_csv: Path,
    output_dir: Path,
    min_races_for_review: int = MIN_RACES_FOR_REVIEW,
    min_regime_races: int = MIN_REGIME_RACES,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    matrix_rows = load_csv(runner_matrix_csv)
    prediction_rows = load_csv(race_predictions_csv)
    ledger, collection = build_race_ledger(
        matrix_rows=matrix_rows,
        prediction_rows=prediction_rows,
    )
    overall_metrics = summarize_rows(ledger)
    regime_rows = summarize_regimes(ledger, min_regime_races=min_regime_races)
    hypotheses = derive_hypotheses(regime_rows, min_regime_races=min_regime_races)
    rank_first_readiness = hypotheses.get("rank_first_readiness")
    if not isinstance(rank_first_readiness, Mapping):
        rank_first_readiness = {}

    blockers: list[str] = []
    if len(ledger) < min_races_for_review:
        blockers.append("accepted_race_count_below_review_floor")
    if not regime_rows:
        blockers.append("no_regime_summaries_meet_minimum")

    report = {
        "schema_version": "market_residual_regime_audit_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": FINAL_READY if not blockers else FINAL_COLLECTING,
        "output_dir": relpath(output_dir),
        "runner_matrix_csv": relpath(runner_matrix_csv),
        "race_predictions_csv": relpath(race_predictions_csv),
        "race_regime_ledger_csv": relpath(output_dir / LEDGER_CSV),
        "regime_summary_csv": relpath(output_dir / SUMMARY_CSV),
        "next_hypotheses_json": relpath(output_dir / HYPOTHESES_JSON),
        "matrix_row_count": len(matrix_rows),
        "prediction_row_count": len(prediction_rows),
        "accepted_race_count": len(ledger),
        "minimum_races_for_review": min_races_for_review,
        "minimum_regime_races": min_regime_races,
        "regime_summary_count": len(regime_rows),
        "collection": collection,
        "dimension_contract": DIMENSIONS,
        "overall_metrics": overall_metrics,
        "rank_first_hypothesis_status": rank_first_readiness.get("status"),
        "rank_first_hypothesis_blockers": list(
            rank_first_readiness.get("blockers") or []
        ),
        "pre_race_rank_first_help_regime_count": int(
            rank_first_readiness.get("pre_race_rank_first_help_regime_count") or 0
        ),
        "pre_race_logloss_only_help_regime_count": int(
            rank_first_readiness.get("pre_race_logloss_only_help_regime_count") or 0
        ),
        "blockers": blockers,
        "promotion_ready": False,
        "promotion_blockers": [
            "report_only_residual_regime_audit_not_promotion_eligible",
            "requires_new_out_of_sample_packet_for_any_pre_race_gate",
        ],
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }

    write_json(output_dir / REPORT_FILE, report)
    write_csv(output_dir / LEDGER_CSV, ledger, LEDGER_FIELDS)
    write_csv(output_dir / SUMMARY_CSV, regime_rows, SUMMARY_FIELDS)
    write_json(output_dir / HYPOTHESES_JSON, hypotheses)
    write_text(output_dir / SUMMARY_MD, summarize_markdown(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-matrix-csv", type=Path, required=True)
    parser.add_argument("--race-predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-races-for-review", type=int, default=MIN_RACES_FOR_REVIEW)
    parser.add_argument("--min-regime-races", type=int, default=MIN_REGIME_RACES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir
        or DEFAULT_EVIDENCE_ROOT / f"market_residual_regime_audit_{now_id(generated_at)}"
    )
    report = build_audit(
        runner_matrix_csv=args.runner_matrix_csv,
        race_predictions_csv=args.race_predictions_csv,
        output_dir=output_dir,
        min_races_for_review=args.min_races_for_review,
        min_regime_races=args.min_regime_races,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
