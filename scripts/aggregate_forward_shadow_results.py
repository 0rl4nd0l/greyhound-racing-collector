#!/usr/bin/env python3
"""Aggregate forward shadow result joins across shadow batches.

This is report-only. It reads existing forward-shadow result-join artifacts,
deduplicates repeated race IDs, computes metrics on the selected unique joined
races, and writes a fresh aggregate artifact. It does not write labels, DB rows,
registry entries, models, production predictions, EV, or betting output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/forward_shadow_result_aggregate_"
PROBABILITY_COLUMN = "shadow_rf_calibrated_probability"
FINAL_AGGREGATED = "FORWARD_SHADOW_RESULTS_AGGREGATED"
FINAL_PARTIAL = "PARTIAL_AGGREGATE_PENDING_MORE_RESULTS"
FINAL_WAITING = "WAITING_FOR_OFFICIAL_RESULTS"
FINAL_UNSAFE = "BLOCKED_IDENTITY_MATCH_FAILURE"
DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
SOURCE_SHADOW_RUN_NAME_RE = re.compile(r"^(daily_race_ingest_shadow_|forward_shadow_run_).+")

from scripts.join_forward_shadow_results import (  # noqa: E402
    clip_probability,
    logistic_calibration_review,
    probability_reliability_bins,
)


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(paths: Sequence[Path] = DEFAULT_PROTECTED_PATHS) -> dict[str, str | None]:
    return {relpath(path) or str(path): sha256_file(path) for path in paths}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_forward_shadow_result_aggregate_artifact:{relative}")
    return logical.absolute()


def discovered_result_join_dirs(evidence_root: Path) -> list[Path]:
    return sorted(
        item
        for item in evidence_root.glob("forward_shadow_result_join_*")
        if item.is_dir() and (item / "shadow_forward_metrics.json").exists()
    )


def source_shadow_run_key(source_shadow_run: object, *, fallback_join_dir: Path | None = None) -> str:
    """Return a stable key for grouping result joins by source shadow run.

    Result-join artifacts can refer to the same shadow run through different
    path aliases, such as repo-relative `artifacts/...` paths and storage-root
    relative paths. Known shadow run directory names are stable identifiers, so
    use those before falling back to the previous raw path behavior.
    """

    text = str(source_shadow_run or "").strip()
    if text:
        normalized = text.replace("\\", "/").rstrip("/")
        for part in reversed([part for part in normalized.split("/") if part]):
            if SOURCE_SHADOW_RUN_NAME_RE.match(part):
                return f"shadow_run_dir:{part}"

        path = Path(text)
        try:
            logical = path if path.is_absolute() else ROOT / path
            if logical.exists():
                return f"path:{logical.resolve()}"
        except OSError:
            pass
        return f"raw:{normalized}"

    if fallback_join_dir is not None:
        return f"join_dir:{fallback_join_dir.name}"
    return "raw:"


def result_join_dirs(evidence_root: Path) -> list[Path]:
    latest_by_shadow_run: dict[str, Path] = {}
    for join_dir in discovered_result_join_dirs(evidence_root):
        metrics = load_json(join_dir / "shadow_forward_metrics.json")
        source_key = source_shadow_run_key(metrics.get("source_shadow_run"), fallback_join_dir=join_dir)
        latest_by_shadow_run[source_key] = join_dir
    return sorted(latest_by_shadow_run.values())


def race_key(value: Mapping[str, Any]) -> str:
    return str(value.get("race_id") or "").strip()


def grouped_joined_rows(join_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(join_dir / "joined_shadow_predictions.jsonl"):
        key = race_key(row)
        if key:
            grouped[key].append(row)
    return dict(grouped)


def selected_unique_joined_races(join_dirs: Sequence[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    duplicate_records: list[dict[str, Any]] = []
    seen_sources: dict[str, list[str]] = defaultdict(list)
    for join_dir in join_dirs:
        for key, rows in grouped_joined_rows(join_dir).items():
            if key in selected:
                duplicate_records.append(
                    {
                        "race_id": key,
                        "previous_sources": list(seen_sources[key]),
                        "selected_source": relpath(join_dir),
                        "selection_policy": "latest_join_artifact_per_unique_race",
                    }
                )
            selected[key] = rows
            seen_sources[key].append(relpath(join_dir) or str(join_dir))
    return selected, duplicate_records


def collect_pending_and_unsafe(join_dirs: Sequence[Path], selected_race_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending_by_race: dict[str, dict[str, Any]] = {}
    unsafe_by_race: dict[str, dict[str, Any]] = {}
    for join_dir in join_dirs:
        pending_path = join_dir / "pending_results.json"
        if pending_path.exists():
            for row in load_json(pending_path).get("pending_results") or []:
                if isinstance(row, Mapping):
                    key = race_key(row)
                    if key and key not in selected_race_ids:
                        record = dict(row)
                        record["source_join_artifact"] = relpath(join_dir)
                        pending_by_race[key] = record
        unsafe_path = join_dir / "unsafe_result_matches.json"
        if unsafe_path.exists():
            for row in load_json(unsafe_path).get("unsafe_result_matches") or []:
                if isinstance(row, Mapping):
                    key = race_key(row) or str(row.get("race_url") or row.get("source_csv") or "")
                    if key:
                        record = dict(row)
                        record["source_join_artifact"] = relpath(join_dir)
                        unsafe_by_race[key] = record
    return sorted(pending_by_race.values(), key=lambda row: str(row.get("race_id"))), sorted(
        unsafe_by_race.values(), key=lambda row: str(row.get("race_id"))
    )


def safe_race_summary(race_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    winners = [row for row in rows if row.get("is_winner")]
    if len(winners) != 1:
        return None
    winner = winners[0]
    top_pick = sorted(rows, key=lambda row: int(row.get("predicted_rank") or 999))[0]
    winner_rank = int(winner["predicted_rank"])
    return {
        "race_id": race_id,
        "race_date": winner.get("race_date"),
        "venue": winner.get("venue"),
        "race_number": winner.get("race_number"),
        "runner_count": len(rows),
        "top_pick_box": top_pick.get("box"),
        "top_pick_dog_name": top_pick.get("dog_name"),
        "top_pick_won": bool(top_pick.get("is_winner")),
        "winner_box": winner.get("box"),
        "winner_predicted_rank": winner_rank,
        "winner_in_top3": winner_rank <= 3,
        "selected_result_url": winner.get("result_url"),
    }


def compute_aggregate_metrics(
    *,
    selected_by_race: Mapping[str, Sequence[Mapping[str, Any]]],
    pending_count: int,
    unsafe_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    safe_races = []
    joined_rows = []
    skipped = []
    for key, rows in sorted(selected_by_race.items()):
        summary = safe_race_summary(key, rows)
        if summary is None:
            skipped.append({"race_id": key, "reason": "winner_row_count_not_exactly_one"})
            continue
        safe_races.append(summary)
        joined_rows.extend(dict(row) for row in rows)

    labels = [1 if row.get("is_winner") else 0 for row in joined_rows]
    probabilities = [float(row[PROBABILITY_COLUMN]) for row in joined_rows]
    joined_by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        joined_by_race[str(row.get("race_id"))].append(row)

    if safe_races:
        winner_ranks = [int(row["winner_predicted_rank"]) for row in safe_races]
        top1 = sum(1 for row in safe_races if row.get("top_pick_won")) / len(safe_races)
        top3 = sum(1 for row in safe_races if row.get("winner_in_top3")) / len(safe_races)
        brier = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)
        logloss = sum(
            -math.log(clip_probability(float(next(row for row in rows if row.get("is_winner"))[PROBABILITY_COLUMN])))
            for rows in joined_by_race.values()
        ) / len(joined_by_race)
        probability_sum_max_error = max(
            abs(sum(float(row[PROBABILITY_COLUMN]) for row in rows) - 1.0)
            for rows in joined_by_race.values()
        )
    else:
        winner_ranks = []
        top1 = top3 = brier = logloss = probability_sum_max_error = None

    calibration_methods = sorted({str(row.get("calibration_method")) for row in joined_rows if row.get("calibration_method")})
    tgr_values = sorted({bool(row.get("tgr_enabled")) for row in joined_rows})
    metrics = {
        "schema_version": "forward_shadow_aggregate_metrics_v1",
        "status": "COMPUTED_FOR_UNIQUE_SAFE_JOINED_RACES" if safe_races else "NO_SAFE_JOINED_RACES",
        "selection_policy": "latest_join_artifact_per_unique_race",
        "safe_joined_race_count": len(safe_races),
        "safe_joined_runner_count": len(joined_rows),
        "pending_race_count": pending_count,
        "unsafe_match_count": unsafe_count,
        "top1": top1,
        "top3": top3,
        "winner_ranks": winner_ranks,
        "mean_winner_rank": sum(winner_ranks) / len(winner_ranks) if winner_ranks else None,
        "brier": brier,
        "logloss": logloss,
        "logloss_method": "mean_negative_log_calibrated_probability_assigned_to_winner_per_race",
        "probability_sum_max_error_joined_races": probability_sum_max_error,
        "joined_races": safe_races,
        "calibration_methods": calibration_methods,
        "tgr_enabled_values": tgr_values,
    }
    calibration = {
        "schema_version": "forward_shadow_aggregate_calibration_review_v1",
        "safe_joined_race_count": len(safe_races),
        "safe_joined_runner_count": len(joined_rows),
        "brier": metrics["brier"],
        "logloss": metrics["logloss"],
        "reliability_bins": probability_reliability_bins(labels, probabilities) if joined_rows else [],
        "slope_intercept": (
            logistic_calibration_review(labels, probabilities)
            if joined_rows
            else {"status": "no_safe_joined_rows", "slope": None, "intercept": None}
        ),
    }
    top_pick_counts = Counter(str(row.get("top_pick_box")) for row in safe_races)
    box_bias = {
        "schema_version": "forward_shadow_aggregate_box_bias_review_v1",
        "safe_joined_race_count": len(safe_races),
        "safe_joined_top_pick_box_distribution": dict(sorted(top_pick_counts.items())),
        "safe_joined_box_1_top_pick_share": top_pick_counts.get("1", 0) / len(safe_races) if safe_races else None,
    }
    return metrics, calibration, box_bias, skipped


def final_status(metrics: Mapping[str, Any]) -> str:
    safe = int(metrics.get("safe_joined_race_count") or 0)
    pending = int(metrics.get("pending_race_count") or 0)
    unsafe = int(metrics.get("unsafe_match_count") or 0)
    if unsafe and not safe:
        return FINAL_UNSAFE
    if safe and not pending and not unsafe:
        return FINAL_AGGREGATED
    if safe:
        return FINAL_PARTIAL
    return FINAL_WAITING


def build_aggregate_report(*, evidence_root: Path, generated_at: datetime | None = None) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    join_dirs = result_join_dirs(evidence_root)
    selected, duplicates = selected_unique_joined_races(join_dirs)
    pending, unsafe = collect_pending_and_unsafe(join_dirs, set(selected))
    metrics, calibration, box_bias, skipped = compute_aggregate_metrics(
        selected_by_race=selected,
        pending_count=len(pending),
        unsafe_count=len(unsafe),
    )
    return {
        "schema_version": "forward_shadow_result_aggregate_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status(metrics),
        "source_join_artifacts": [relpath(path) for path in join_dirs],
        "source_join_artifact_count": len(join_dirs),
        "discovered_join_artifact_count": len(discovered_result_join_dirs(evidence_root)),
        "join_artifact_selection_policy": "latest_result_join_per_source_shadow_run",
        "aggregate_forward_metrics": metrics,
        "aggregate_calibration_review": calibration,
        "aggregate_box_bias_review": box_bias,
        "duplicate_joined_races": duplicates,
        "duplicate_joined_race_count": len(duplicates),
        "pending_results": {
            "schema_version": "forward_shadow_aggregate_pending_results_v1",
            "pending_race_count": len(pending),
            "pending_results": pending,
        },
        "unsafe_result_matches": {
            "schema_version": "forward_shadow_aggregate_unsafe_result_matches_v1",
            "unsafe_match_count": len(unsafe),
            "unsafe_result_matches": unsafe,
        },
        "skipped_joined_races": skipped,
        "no_write_guarantees": {
            "production_promotion": False,
            "registry_mutation": False,
            "production_pointer_update": False,
            "production_prediction_write": False,
            "db_write": False,
            "label_write": False,
            "tgr_enabled": False,
            "betting_or_ev_output": False,
        },
    }


def build_summary(report: Mapping[str, Any]) -> str:
    metrics = report["aggregate_forward_metrics"]
    return "\n".join(
        [
            "# Forward Shadow Result Aggregate",
            "",
            f"- Final status: `{report.get('final_status')}`",
            f"- Source join artifacts: `{report.get('source_join_artifact_count')}`",
            f"- Unique safe joined races: `{metrics.get('safe_joined_race_count')}`",
            f"- Pending races: `{metrics.get('pending_race_count')}`",
            f"- Unsafe matches: `{metrics.get('unsafe_match_count')}`",
            f"- Duplicate joined races de-duplicated: `{report.get('duplicate_joined_race_count')}`",
            f"- Top1: `{metrics.get('top1')}`",
            f"- Top3: `{metrics.get('top3')}`",
            f"- Mean winner rank: `{metrics.get('mean_winner_rank')}`",
            f"- Brier: `{metrics.get('brier')}`",
            f"- LogLoss: `{metrics.get('logloss')}`",
            "",
            "No production promotion, registry mutation, DB writes, label writes, TGR enablement, betting output, EV output, or production prediction overwrite was performed.",
            "",
        ]
    )


def run_aggregate(
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or evidence_root / f"forward_shadow_result_aggregate_{generated_at.strftime('%Y%m%dT%H%M%S%z')}"
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes()
    report = build_aggregate_report(evidence_root=evidence_root, generated_at=generated_at)
    protected_after = protected_hashes()
    report["protected_hashes_before"] = protected_before
    report["protected_hashes_after"] = protected_after
    report["protected_paths_unchanged"] = protected_before == protected_after
    if not report["protected_paths_unchanged"]:
        report["final_status"] = "BLOCKED_PROTECTED_PATH_MUTATION"

    write_json(output_dir / "aggregate_forward_metrics.json", report["aggregate_forward_metrics"])
    write_json(output_dir / "aggregate_calibration_review.json", report["aggregate_calibration_review"])
    write_json(output_dir / "aggregate_box_bias_review.json", report["aggregate_box_bias_review"])
    write_json(output_dir / "duplicate_joined_races.json", report["duplicate_joined_races"])
    write_json(output_dir / "pending_results.json", report["pending_results"])
    write_json(output_dir / "unsafe_result_matches.json", report["unsafe_result_matches"])
    write_json(output_dir / "forward_shadow_result_aggregate_report.json", report)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["final_status"],
        "safe_joined_race_count": report["aggregate_forward_metrics"]["safe_joined_race_count"],
        "pending_race_count": report["aggregate_forward_metrics"]["pending_race_count"],
        "unsafe_match_count": report["aggregate_forward_metrics"]["unsafe_match_count"],
        "duplicate_joined_race_count": report["duplicate_joined_race_count"],
        "protected_paths_unchanged": report["protected_paths_unchanged"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_aggregate(evidence_root=args.evidence_root, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
