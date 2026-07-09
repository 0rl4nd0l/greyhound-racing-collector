#!/usr/bin/env python3
"""Build a report-only failure attribution packet for a race evidence scorecard.

The packet summarizes where the model trails the market on an already-certified
race-level scorecard. It does not train, promote, mutate registries, write DB
rows, write labels, emit EV, place bets, rewrite snapshots/manifests, or control
daemons.
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
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/" "scorecard_failure_attribution_"
REPORT_FILE = "scorecard_failure_attribution_report.json"
SUMMARY_CSV = "dimension_summary.csv"
ERROR_CSV = "top_error_clusters.csv"
SUMMARY_MD = "SUMMARY.md"
FINAL_READY = "SCORECARD_FAILURE_ATTRIBUTION_READY"
FINAL_DATA_MISSING = "SCORECARD_FAILURE_ATTRIBUTION_DATA_MISSING"
MIN_CLUSTER_RACES = 10
NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "official_result_write": False,
    "daemon_control": False,
    "betting_or_ev_action": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
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


def assert_output_dir_safe(output_dir: Path) -> Path:
    root = ROOT.expanduser().resolve(strict=False)
    logical = output_dir.expanduser()
    if not logical.is_absolute():
        logical = root / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_scorecard_failure_attribution:{relative}")
    return resolved


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
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
        "schema_version": "scorecard_failure_attribution_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def probability_band(value: Any, *, prefix: str) -> str:
    probability = finite_float(value)
    if probability is None:
        return f"{prefix}_missing"
    if probability < 0.10:
        return f"{prefix}_lt_10pct"
    if probability < 0.15:
        return f"{prefix}_10_15pct"
    if probability < 0.20:
        return f"{prefix}_15_20pct"
    if probability < 0.30:
        return f"{prefix}_20_30pct"
    return f"{prefix}_gte_30pct"


def rank_band(value: Any, *, prefix: str) -> str:
    rank = finite_int(value)
    if rank is None:
        return f"{prefix}_missing"
    if rank == 1:
        return f"{prefix}_rank_1"
    if rank <= 3:
        return f"{prefix}_rank_2_3"
    return f"{prefix}_rank_4_plus"


def race_number_band(value: Any) -> str:
    race_number = finite_int(value)
    if race_number is None:
        return "race_number_missing"
    if race_number <= 4:
        return "race_number_1_4"
    if race_number <= 8:
        return "race_number_5_8"
    return "race_number_9_plus"


def boolean_band(value: bool, *, true_label: str, false_label: str) -> str:
    return true_label if value else false_label


def enrich_rows(
    scorecard_rows: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    inventory_by_race = {str(row.get("race_id")): row for row in inventory_rows}
    enriched: list[dict[str, Any]] = []
    for score in scorecard_rows:
        row = dict(score)
        inventory = inventory_by_race.get(str(row.get("race_id"))) or {}
        model_top_box = str(row.get("model_top_box") or "")
        market_top_box = str(row.get("market_top_box") or "")
        row.update(
            {
                "runner_count_int": finite_int(row.get("runner_count")) or 0,
                "winner_box_int": finite_int(row.get("winner_box")),
                "model_winner_rank_int": finite_int(row.get("model_winner_rank")),
                "market_winner_rank_int": finite_int(row.get("market_winner_rank")),
                "model_top1_bool": parse_bool(row.get("model_top1_correct")),
                "model_top3_bool": parse_bool(row.get("model_top3_correct")),
                "market_top1_bool": parse_bool(row.get("market_top1_correct")),
                "market_top3_bool": parse_bool(row.get("market_top3_correct")),
                "model_logloss_float": finite_float(row.get("model_logloss")),
                "market_logloss_float": finite_float(row.get("market_logloss")),
                "model_winner_probability_float": finite_float(row.get("model_winner_probability")),
                "market_winner_probability_float": finite_float(
                    row.get("market_winner_probability")
                ),
                "model_market_top_agree": bool(model_top_box and model_top_box == market_top_box),
                "race_number_band": race_number_band(row.get("race_number")),
                "winner_market_rank_band": rank_band(
                    row.get("market_winner_rank"), prefix="winner_market"
                ),
                "winner_model_rank_band": rank_band(
                    row.get("model_winner_rank"), prefix="winner_model"
                ),
                "model_winner_probability_band": probability_band(
                    row.get("model_winner_probability"), prefix="model_winner_probability"
                ),
                "market_winner_probability_band": probability_band(
                    row.get("market_winner_probability"), prefix="market_winner_probability"
                ),
                "model_market_top_agreement_band": boolean_band(
                    bool(model_top_box and model_top_box == market_top_box),
                    true_label="model_market_top_agree",
                    false_label="model_market_top_disagree",
                ),
                "official_result_duplicate_certification": inventory.get(
                    "official_result_duplicate_certification"
                )
                or "unknown",
            }
        )
        enriched.append(row)
    return enriched


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "race_count": 0,
            "model_top1_accuracy": None,
            "market_top1_accuracy": None,
            "model_top3_accuracy": None,
            "market_top3_accuracy": None,
            "model_mean_winner_rank": None,
            "market_mean_winner_rank": None,
            "model_logloss": None,
            "market_logloss": None,
            "model_minus_market_top1": None,
            "model_minus_market_top3": None,
            "model_minus_market_mean_winner_rank": None,
            "model_minus_market_logloss": None,
        }
    race_count = len(rows)
    model_top1 = sum(1 for row in rows if row.get("model_top1_bool")) / race_count
    market_top1 = sum(1 for row in rows if row.get("market_top1_bool")) / race_count
    model_top3 = sum(1 for row in rows if row.get("model_top3_bool")) / race_count
    market_top3 = sum(1 for row in rows if row.get("market_top3_bool")) / race_count
    model_rank = mean(row.get("model_winner_rank_int") for row in rows)
    market_rank = mean(row.get("market_winner_rank_int") for row in rows)
    model_logloss = mean(row.get("model_logloss_float") for row in rows)
    market_logloss = mean(row.get("market_logloss_float") for row in rows)
    return {
        "race_count": race_count,
        "model_top1_accuracy": model_top1,
        "market_top1_accuracy": market_top1,
        "model_top3_accuracy": model_top3,
        "market_top3_accuracy": market_top3,
        "model_mean_winner_rank": model_rank,
        "market_mean_winner_rank": market_rank,
        "model_logloss": model_logloss,
        "market_logloss": market_logloss,
        "model_minus_market_top1": model_top1 - market_top1,
        "model_minus_market_top3": model_top3 - market_top3,
        "model_minus_market_mean_winner_rank": (
            model_rank - market_rank if model_rank is not None and market_rank is not None else None
        ),
        "model_minus_market_logloss": (
            model_logloss - market_logloss
            if model_logloss is not None and market_logloss is not None
            else None
        ),
    }


DIMENSIONS = {
    "venue": lambda row: row.get("venue") or "venue_missing",
    "race_date": lambda row: row.get("race_date") or "race_date_missing",
    "race_number_band": lambda row: row.get("race_number_band"),
    "runner_count": lambda row: str(row.get("runner_count") or "runner_count_missing"),
    "winner_box": lambda row: str(row.get("winner_box") or "winner_box_missing"),
    "model_top_box": lambda row: str(row.get("model_top_box") or "model_top_box_missing"),
    "market_top_box": lambda row: str(row.get("market_top_box") or "market_top_box_missing"),
    "winner_market_rank_band": lambda row: row.get("winner_market_rank_band"),
    "winner_model_rank_band": lambda row: row.get("winner_model_rank_band"),
    "model_market_top_agreement": lambda row: row.get("model_market_top_agreement_band"),
    "model_winner_probability_band": lambda row: row.get("model_winner_probability_band"),
    "market_winner_probability_band": lambda row: row.get("market_winner_probability_band"),
    "official_result_duplicate_certification": lambda row: row.get(
        "official_result_duplicate_certification"
    )
    or "unknown",
}


def dimension_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dimension, getter in DIMENSIONS.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(getter(row))].append(row)
        for value, group_rows in sorted(grouped.items()):
            row = {"dimension": dimension, "dimension_value": value}
            row.update(metrics(group_rows))
            summaries.append(row)
    return summaries


def top_error_clusters(
    summaries: Sequence[Mapping[str, Any]],
    *,
    min_cluster_races: int,
    limit: int = 25,
) -> list[dict[str, Any]]:
    candidates = [
        dict(row)
        for row in summaries
        if int(row.get("race_count") or 0) >= min_cluster_races
        and row.get("dimension") not in {"winner_market_rank_band", "winner_model_rank_band"}
    ]
    return sorted(
        candidates,
        key=lambda row: (
            float(row.get("model_minus_market_top1") or 0.0),
            -float(row.get("model_minus_market_logloss") or 0.0),
            -int(row.get("race_count") or 0),
        ),
    )[:limit]


def box_bias_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    race_count = len(rows)
    model_top = Counter(str(row.get("model_top_box") or "missing") for row in rows)
    market_top = Counter(str(row.get("market_top_box") or "missing") for row in rows)
    winner = Counter(str(row.get("winner_box") or "missing") for row in rows)
    model_share = {
        box: count / race_count for box, count in sorted(model_top.items()) if race_count
    }
    market_share = {
        box: count / race_count for box, count in sorted(market_top.items()) if race_count
    }
    winner_share = {box: count / race_count for box, count in sorted(winner.items()) if race_count}
    model_minus_winner = {
        box: model_share.get(box, 0.0) - winner_share.get(box, 0.0)
        for box in sorted(set(model_share) | set(winner_share))
        if box != "missing"
    }
    max_overpick_box = None
    max_overpick_share = None
    if model_minus_winner:
        max_overpick_box, max_overpick_share = max(
            model_minus_winner.items(), key=lambda item: item[1]
        )
    return {
        "race_count": race_count,
        "model_top_box_share": model_share,
        "market_top_box_share": market_share,
        "winner_box_share": winner_share,
        "model_top_box_minus_winner_box_share": model_minus_winner,
        "max_model_top_box_overpick_box": max_overpick_box,
        "max_model_top_box_overpick_share": max_overpick_share,
        "model_box1_top_pick_share": model_top.get("1", 0) / race_count if race_count else None,
        "market_box1_top_pick_share": market_top.get("1", 0) / race_count if race_count else None,
        "winner_box1_share": winner.get("1", 0) / race_count if race_count else None,
    }


def blocker_decision(overall: Mapping[str, Any], box_bias: Mapping[str, Any]) -> str:
    race_count = int(overall.get("race_count") or 0)
    if race_count < MIN_CLUSTER_RACES:
        return "DATA_INTEGRITY_BLOCKER"
    top1_gap = float(overall.get("model_minus_market_top1") or 0.0)
    top3_gap = float(overall.get("model_minus_market_top3") or 0.0)
    rank_gap = float(overall.get("model_minus_market_mean_winner_rank") or 0.0)
    logloss_gap = float(overall.get("model_minus_market_logloss") or 0.0)
    max_model_box_overpick = float(box_bias.get("max_model_top_box_overpick_share") or 0.0)
    if max_model_box_overpick >= 0.10 and top1_gap < -0.10:
        return "FEATURE_COVERAGE_BLOCKER"
    if logloss_gap > 0.20 and abs(top1_gap) <= 0.10:
        return "MODEL_CALIBRATION_BLOCKER"
    if top1_gap <= -0.10 and top3_gap <= -0.10 and rank_gap > 0.50:
        return "MARKET_BASELINE_DOMINATES"
    return "REPORT_ONLY_CHALLENGER_READY"


def remediation_target(decision: str, top_clusters: Sequence[Mapping[str, Any]]) -> str:
    if decision == "FEATURE_COVERAGE_BLOCKER":
        return (
            "Audit non-box feature coverage and box/draw shortcut reliance before challenger work."
        )
    if decision == "MODEL_CALIBRATION_BLOCKER":
        return "Run a rank-preserving calibration/de-biasing study only after feature provenance passes."
    if decision == "MARKET_BASELINE_DOMINATES":
        cluster = top_clusters[0] if top_clusters else {}
        if cluster:
            return (
                "Prioritize market-baseline residual analysis in "
                f"{cluster.get('dimension')}={cluster.get('dimension_value')}."
            )
        return "Keep market as baseline and collect/repair richer pre-race features."
    if decision == "REPORT_ONLY_CHALLENGER_READY":
        return (
            "Proceed to report-only challenger only with temporal split and market baseline gates."
        )
    return "Repair data integrity before accuracy attribution."


SUMMARY_FIELDS = [
    "dimension",
    "dimension_value",
    "race_count",
    "model_top1_accuracy",
    "market_top1_accuracy",
    "model_top3_accuracy",
    "market_top3_accuracy",
    "model_mean_winner_rank",
    "market_mean_winner_rank",
    "model_logloss",
    "market_logloss",
    "model_minus_market_top1",
    "model_minus_market_top3",
    "model_minus_market_mean_winner_rank",
    "model_minus_market_logloss",
]


def summary_markdown(report: Mapping[str, Any]) -> str:
    overall = (
        report.get("overall_metrics") if isinstance(report.get("overall_metrics"), Mapping) else {}
    )
    return "\n".join(
        [
            "# Scorecard Failure Attribution",
            "",
            f"Final status: `{report.get('final_status')}`",
            f"Decision: `{report.get('decision')}`",
            f"Recommended next remediation target: `{report.get('recommended_next_remediation_target')}`",
            "",
            f"- Scorecard races: `{overall.get('race_count')}`",
            f"- Model Top1 / Top3: `{overall.get('model_top1_accuracy')}` / `{overall.get('model_top3_accuracy')}`",
            f"- Market Top1 / Top3: `{overall.get('market_top1_accuracy')}` / `{overall.get('market_top3_accuracy')}`",
            f"- Model minus market Top1 / Top3: `{overall.get('model_minus_market_top1')}` / `{overall.get('model_minus_market_top3')}`",
            f"- Model minus market mean winner rank: `{overall.get('model_minus_market_mean_winner_rank')}`",
            f"- Model minus market logloss: `{overall.get('model_minus_market_logloss')}`",
            "",
            "## Top Error Clusters",
            "",
            "```json",
            json.dumps(report.get("top_error_clusters") or [], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def build_packet(
    *,
    scorecard_csv: Path,
    inventory_csv: Path,
    report_json: Path | None,
    output_dir: Path,
    min_cluster_races: int = MIN_CLUSTER_RACES,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    generated_at = generated_at or datetime.now().astimezone()

    scorecard_rows = load_csv(scorecard_csv)
    inventory_rows = load_csv(inventory_csv)
    source_report = load_json(report_json) if report_json else {}
    enriched = enrich_rows(scorecard_rows, inventory_rows)
    overall = metrics(enriched)
    summaries = dimension_summaries(enriched)
    clusters = top_error_clusters(
        summaries,
        min_cluster_races=int(min_cluster_races),
    )
    box_bias = box_bias_summary(enriched)
    decision = blocker_decision(overall, box_bias)
    final_status = FINAL_READY if enriched else FINAL_DATA_MISSING
    report = {
        "schema_version": "scorecard_failure_attribution_packet_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "decision": decision,
        "recommended_next_remediation_target": remediation_target(decision, clusters),
        "scorecard_csv": relpath(scorecard_csv),
        "inventory_csv": relpath(inventory_csv),
        "source_report_json": relpath(report_json) if report_json else None,
        "source_report_final_status": source_report.get("final_status"),
        "source_report_recommended_decision": source_report.get("recommended_decision"),
        "output_dir": relpath(output_dir),
        "overall_metrics": overall,
        "box_bias_summary": box_bias,
        "top_error_clusters": clusters,
        "dimension_summary_csv": relpath(output_dir / SUMMARY_CSV),
        "top_error_clusters_csv": relpath(output_dir / ERROR_CSV),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_csv(output_dir / SUMMARY_CSV, summaries, SUMMARY_FIELDS)
    write_csv(output_dir / ERROR_CSV, clusters, SUMMARY_FIELDS)
    write_json(output_dir / REPORT_FILE, report)
    write_text(output_dir / SUMMARY_MD, summary_markdown(report))
    write_text(output_dir / "final_status.txt", final_status + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard-csv", type=Path, required=True)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-cluster-races", type=int, default=MIN_CLUSTER_RACES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (ROOT / f"{OUTPUT_PREFIX}{now_id()}_report_only")
    report = build_packet(
        scorecard_csv=args.scorecard_csv,
        inventory_csv=args.inventory_csv,
        report_json=args.report_json,
        output_dir=output_dir,
        min_cluster_races=args.min_cluster_races,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
