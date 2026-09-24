#!/usr/bin/env python3
"""Report-only audit of exact-track/distance fast non-favourites.

The two-phase interface freezes the source-bound score definition and matrix
before any group outcome is evaluated. Missing runner history fails closed:
"speed rank 1" is asserted only for a complete active field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector")
SOURCE_MATRIX = ROOT / "artifacts/form_speed_market_residual_20260818_report_only/feature_matrix.jsonl"
TARGET_MATRIX = SOURCE_ROOT / "artifacts/sportsbet_speed_context_experiment_20260815_clean_rerun_report_only/enriched_development_matrix.jsonl"
HISTORY = SOURCE_ROOT / "artifacts/raw_race_shape_experiment_20260815_report_only/deduplicated_raw_history_sidecar.jsonl"
BETFAIR = ROOT / "artifacts/betfair_historical_surface_20260817_report_only/sportsbet_betfair_joined_surface.jsonl"
DEFAULT_OUT = ROOT / "artifacts/fast_nonfavourite_mechanism_audit_20260818_report_only"

START = "2026-06-10"
CALIBRATION_END = "2026-06-24"
END = "2026-07-18"
FORWARD_START = "2026-08-18"
LOOKBACK_DAYS = 365
LAST_N = 5
MIN_STARTS = 2
SEED = 20260818
BOOTSTRAPS = 5000
FOLDS = (
    {"id": 1, "start": "2026-06-25", "end": "2026-07-02"},
    {"id": 2, "start": "2026-07-03", "end": "2026-07-10"},
    {"id": 3, "start": "2026-07-11", "end": END},
)
GROUP_NAMES = ("A_MARKET_RANK_1", "B_MARKET_RANK_2", "C_MARKET_RANK_3_PLUS", "D_Q75_NONFAV", "D_Q90_NONFAV")


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def exact_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["race_id"]), int(row["box_number"]), str(row["odds_capture_timestamp"])


def finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def fold_id(race_date: str) -> int | None:
    return next((int(fold["id"]) for fold in FOLDS if fold["start"] <= race_date <= fold["end"]), None)


def history_times(
    history: Sequence[Mapping[str, Any]], *, cutoff: str, track: str, distance_m: int
) -> list[dict[str, Any]]:
    cutoff_date = date.fromisoformat(cutoff)
    earliest = cutoff_date - timedelta(days=LOOKBACK_DAYS)
    accepted = []
    for row in history:
        when_text = str(row.get("source_performance_date") or "")[:10]
        try:
            when = date.fromisoformat(when_text)
        except ValueError:
            continue
        elapsed = finite_positive(row.get("individual_time_seconds"))
        source_distance = finite_positive(row.get("source_distance_m"))
        if (
            earliest <= when < cutoff_date
            and str(row.get("source_track") or "") == track
            and source_distance is not None
            and int(source_distance) == distance_m
            and elapsed is not None
        ):
            accepted.append({"date": when_text, "time": elapsed, "source_row_id": int(row["source_row_id"])})
    accepted.sort(key=lambda row: (row["date"], row["source_row_id"]))
    return accepted[-LAST_N:]


def build_matrix() -> tuple[list[dict[str, Any]], dict[str, Any], list[float]]:
    source_rows = [row for row in load_jsonl(SOURCE_MATRIX) if START <= str(row["race_date"]) <= END]
    if any(str(row["race_date"]) >= FORWARD_START for row in source_rows):
        raise SystemExit("forward_target_boundary_breached")
    targets_list = load_jsonl(TARGET_MATRIX)
    targets = {exact_key(row): row for row in targets_list}
    if len(targets) != len(targets_list):
        raise SystemExit("duplicate_target_identity")
    history_by_native: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_content: set[str] = set()
    for row in load_jsonl(HISTORY):
        content_hash = str(row.get("canonical_content_sha256") or "")
        native_id = str(row.get("native_thedogs_dog_id") or "")
        if not content_hash or content_hash in seen_content or not native_id:
            continue
        seen_content.add(content_hash)
        history_by_native[native_id].append(row)

    output = []
    for row in source_rows:
        target = targets.get(exact_key(row))
        if target is None or str(target.get("native_thedogs_dog_id") or "") != str(row.get("native_thedogs_dog_id") or ""):
            raise SystemExit("target_native_identity_mismatch")
        track = str(target.get("venue") or "")
        distance = finite_positive(target.get("target_distance_m"))
        if not track or distance is None or not distance.is_integer():
            raise SystemExit("target_context_missing")
        times = history_times(
            history_by_native[str(row["native_thedogs_dog_id"])],
            cutoff=str(row["race_date"]),
            track=track,
            distance_m=int(distance),
        )
        item = dict(row)
        item.update(
            {
                "target_track_exact": track,
                "target_distance_m": int(distance),
                "exact_td_prior_start_count": len(times),
                "exact_td_source_dates": [entry["date"] for entry in times],
                "exact_td_median_time_seconds": float(np.median([entry["time"] for entry in times])) if len(times) >= MIN_STARTS else None,
                "speed_score": -float(np.median([entry["time"] for entry in times])) if len(times) >= MIN_STARTS else None,
                "evaluation_fold_id": fold_id(str(row["race_date"])),
            }
        )
        output.append(item)

    races: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        races[str(row["race_id"])].append(row)
    complete = 0
    incomplete = 0
    some_coverage = 0
    calibration_margins: list[float] = []
    for race_rows in races.values():
        scores = [row["speed_score"] for row in race_rows]
        available = sum(score is not None for score in scores)
        some_coverage += available > 0
        if available != len(race_rows):
            incomplete += 1
            continue
        ordered = sorted(race_rows, key=lambda row: (-float(row["speed_score"]), int(row["box_number"])))
        if len(ordered) < 2 or math.isclose(float(ordered[0]["speed_score"]), float(ordered[1]["speed_score"]), rel_tol=0, abs_tol=1e-12):
            incomplete += 1
            continue
        complete += 1
        best_time = float(ordered[0]["exact_td_median_time_seconds"])
        second_time = float(ordered[1]["exact_td_median_time_seconds"])
        margin = 100.0 * (second_time - best_time) / second_time
        for row in race_rows:
            row["complete_field_speed_rank"] = 1 + sum(float(other["speed_score"]) > float(row["speed_score"]) for other in race_rows)
            row["speed_margin_percent_if_rank1"] = margin if row is ordered[0] else None
        if str(race_rows[0]["race_date"]) <= CALIBRATION_END:
            calibration_margins.append(margin)

    summary = {
        "source_races": len(races),
        "source_runner_rows": len(output),
        "runner_rows_with_score": sum(row["speed_score"] is not None for row in output),
        "races_with_any_scored_runner": some_coverage,
        "complete_field_races": complete,
        "incomplete_or_tied_field_races": incomplete,
        "calibration_complete_field_races": len(calibration_margins),
        "history_source_date_min": min((entry["source_performance_date"] for entries in history_by_native.values() for entry in entries), default=None),
        "history_source_date_max": max((entry["source_performance_date"] for entries in history_by_native.values() for entry in entries), default=None),
    }
    return output, summary, calibration_margins


def write_checksums(out: Path, names: Sequence[str], manifest: str) -> None:
    (out / manifest).write_text("".join(f"{sha256(out / name)}  {name}\n" for name in names), encoding="utf-8")


def freeze(out: Path) -> None:
    if out.exists():
        raise SystemExit("output_exists")
    out.mkdir(parents=True)
    rows, coverage, margins = build_matrix()
    matrix_path = out / "mechanism_matrix.jsonl"
    write_jsonl(matrix_path, rows)
    thresholds = None
    blockers = []
    if margins:
        thresholds = {"q75_percent": float(np.percentile(margins, 75)), "q90_percent": float(np.percentile(margins, 90))}
    else:
        blockers.append("NO_COMPLETE_FIELD_CALIBRATION_DISTRIBUTION")
    if coverage["complete_field_races"] == 0:
        blockers.append("NO_COMPLETE_FIELD_SPEED_RANKS")
    protocol = {
        "schema_version": "fast_nonfavourite_mechanism_protocol_v1",
        "status": "DATA_COVERAGE_BLOCKED" if blockers else "PROTOCOL_FROZEN_READY_TO_EVALUATE",
        "authority": "report_only_no_model_fit_no_betting_no_deployment_no_promotion",
        "population": {"start": START, "calibration_end": CALIBRATION_END, "evaluation_start": FOLDS[0]["start"], "end": END, **coverage},
        "score": {
            "name": "negative_median_exact_track_distance_time_seconds",
            "definition": "negative median individual time of the last up to five source-bound starts at exact target venue and exact target distance",
            "direction": "higher is faster because elapsed-time median is negated",
            "lookback_days": LOOKBACK_DAYS,
            "last_n": LAST_N,
            "minimum_prior_starts_per_runner": MIN_STARTS,
            "recency_order": "source performance date then immutable source row id",
            "missing_history": "runner score missing; race excluded unless every active runner has a score",
            "ties": "race excluded if the two best scores tie",
            "cross_track_comparison": False,
        },
        "groups": {
            "A_MARKET_RANK_1": "complete-field speed rank 1 and Sportsbet rank 1",
            "B_MARKET_RANK_2": "complete-field speed rank 1 and Sportsbet rank 2",
            "C_MARKET_RANK_3_PLUS": "complete-field speed rank 1 and Sportsbet rank at least 3",
            "D_Q75_NONFAV": "B or C and within-race time-margin percent at least calibration-distribution q75",
            "D_Q90_NONFAV": "B or C and within-race time-margin percent at least calibration-distribution q90",
            "sportsbet_rank_ties": "competition rank: one plus count with strictly greater normalized probability",
        },
        "large_margin": {
            "unit": "100 * (second-fastest median time - fastest median time) / second-fastest median time",
            "calibration_distribution": f"complete-field races {START}..{CALIBRATION_END}; outcomes unused",
            "thresholds": thresholds,
            "post_outcome_search": False,
        },
        "betfair_corroboration": "within strict complete Betfair race overlap, normalized Betfair probability exceeds Sportsbet and Betfair competition rank is strictly better",
        "metrics": {
            "primary": "mean(label_is_winner - Sportsbet normalized probability)",
            "uncertainty": {"unit": "meeting date", "method": "percentile cluster bootstrap", "repetitions": BOOTSTRAPS, "seed": SEED},
            "other": ["observed wins", "summed Sportsbet probability", "binary log loss", "multiclass race-log-score contribution", "odds distribution", "fixed-1u P&L", "ROI CI", "maximum drawdown"],
        },
        "folds": list(FOLDS),
        "decision": "FAST_NONFAV_SIGNAL_WORTH_PROSPECTIVE_TEST only if a predeclared non-favourite group has residual CI lower bound >0 and positive residual in at least 2/3 folds; otherwise FAST_NONFAV_HYPOTHESIS_NOT_SUPPORTED; missing definitional coverage yields DATA_COVERAGE_BLOCKED",
        "forward_exclusions": {"outcomes_on_or_after": FORWARD_START, "betfair_95_5": "2026-08-18..2026-09-30 untouched"},
        "blockers": blockers,
        "inputs": {str(path): sha256(path) for path in (SOURCE_MATRIX, TARGET_MATRIX, HISTORY, BETFAIR)},
        "matrix_sha256": sha256(matrix_path),
    }
    write_json(out / "protocol.json", protocol)
    write_checksums(out, ("mechanism_matrix.jsonl", "protocol.json"), "SEALED_SHA256SUMS")


def empty_group(name: str) -> dict[str, Any]:
    return {
        "group": name,
        "count": 0,
        "observed_wins": 0,
        "summed_sportsbet_probability": 0.0,
        "probability_residual_sum": 0.0,
        "calibration_residual_mean": None,
        "meeting_date_cluster_ci95": [None, None],
        "binary_log_loss_mean": None,
        "race_log_score_contribution_mean": None,
        "sportsbet_odds_distribution": None,
        "folds": [],
        "economic": {"stake_units": 0, "pnl_units": 0.0, "roi": None, "roi_cluster_ci95": [None, None], "max_drawdown_units": 0.0},
    }


def repo_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"head_commit": git("rev-parse", "HEAD"), "head_tree": git("rev-parse", "HEAD^{tree}"), "index_tree": git("write-tree"), "dirty": bool(git("status", "--porcelain=v1", "--untracked-files=all"))}


def evaluate(out: Path) -> None:
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    matrix = out / "mechanism_matrix.jsonl"
    if sha256(matrix) != protocol["matrix_sha256"]:
        raise SystemExit("sealed_matrix_hash_mismatch")
    rows = load_jsonl(matrix)
    if any(str(row["race_date"]) >= FORWARD_START for row in rows):
        raise SystemExit("forward_outcome_boundary_breached")
    if protocol["status"] == "DATA_COVERAGE_BLOCKED":
        report = {
            "schema_version": "fast_nonfavourite_mechanism_report_v1",
            "decision": "DATA_COVERAGE_BLOCKED",
            "protocol_sha256": sha256(out / "protocol.json"),
            "repo": repo_metadata(),
            "population": protocol["population"],
            "score": protocol["score"],
            "large_margin": protocol["large_margin"],
            "groups": [empty_group(name) for name in GROUP_NAMES],
            "betfair_corroboration": {"eligible_races": 0, "corroborated_count": 0, "metrics": empty_group("BETFAIR_CORROBORATED_NONFAV")},
            "runner_examples": [],
            "findings": {
                "BLOCKING": protocol["blockers"],
                "IMPORTANT": [
                    "The complete-field rule prevents a partial-history subset from being mislabeled as speed rank 1.",
                    "The available exact-track/distance history ends before the target period and does not cover every runner in any race under the frozen recency and minimum-start rules.",
                ],
                "OPTIONAL": [],
            },
            "strongest_supported_claim": "The current development corpus cannot test whether a race-wide, robust exact-track/distance speed leader is underpriced by Sportsbet.",
            "boundaries": {"outcomes_2026_08_18_or_later_opened": False, "frozen_forward_cohort_touched": False, "model_fit": False, "deployment": False, "betting_recommendation": False},
        }
        write_json(out / "report.json", report)
        (out / "REPORT.md").write_text(
            "# Fast non-favourite mechanism audit\n\n"
            "Decision: `DATA_COVERAGE_BLOCKED`.\n\n"
            f"The frozen population contains {protocol['population']['source_races']} races and {protocol['population']['source_runner_rows']} runners, but zero complete fields under the exact-track/distance score. "
            "No outcome group, economic diagnostic, or Betfair corroboration was estimated.\n",
            encoding="utf-8",
        )
        write_checksums(out, ("mechanism_matrix.jsonl", "protocol.json", "report.json", "REPORT.md"), "SHA256SUMS")
        return
    raise SystemExit("ready_evaluation_not_implemented_without_a_new_frozen_protocol")


def verify(out: Path) -> None:
    for manifest in ("SEALED_SHA256SUMS", "SHA256SUMS"):
        for line in (out / manifest).read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if sha256(out / name) != digest:
                raise SystemExit(f"checksum_mismatch:{name}")
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    if report["decision"] != "DATA_COVERAGE_BLOCKED" or protocol["status"] != "DATA_COVERAGE_BLOCKED":
        raise SystemExit("unexpected_terminal_state")
    if protocol["population"]["complete_field_races"] != 0:
        raise SystemExit("blocked_state_with_complete_fields")
    if any(str(row["race_date"]) >= FORWARD_START for row in load_jsonl(out / "mechanism_matrix.jsonl")):
        raise SystemExit("forward_outcome_boundary_breached")


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
