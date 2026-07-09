#!/usr/bin/env python3
"""Build a report-only longitudinal rank-first hypothesis watchlist.

This scans daemon-emitted rank-first hypothesis gated challenger packets and
summarizes whether any pre-race-usable hypothesis has repeated support across
distinct rolling evidence samples.

It writes artifacts only. It does not train, promote, mutate registries,
update pointers, write DB labels/odds, emit EV, place bets, rewrite snapshots
or manifests, or enable TGR.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
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
    "rank_first_hypothesis_watchlist_"
)
REPORT_FILE = "rank_first_hypothesis_watchlist_report.json"
SUMMARY_FILE = "SUMMARY.md"
WATCHLIST_CSV = "rank_first_hypothesis_watchlist.csv"
EVALUATIONS_CSV = "rank_first_hypothesis_evaluations.csv"
FINAL_READY = "RANK_FIRST_HYPOTHESIS_WATCHLIST_READY"
FINAL_COLLECTING = "RANK_FIRST_HYPOTHESIS_WATCHLIST_COLLECTING"
DEFAULT_MIN_TRIGGERED_RACES = 10
DEFAULT_MIN_DISTINCT_SAMPLES = 2
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
        raise ValueError(f"output_dir_must_be_rank_first_hypothesis_watchlist:{relative}")
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
        "schema_version": "rank_first_hypothesis_watchlist_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
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


def parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def sample_signature(
    *,
    runner_matrix_csv: Path,
    rolling_report: Mapping[str, Any],
) -> str:
    race_ids: list[str] = []
    if runner_matrix_csv.exists():
        with runner_matrix_csv.open("r", encoding="utf-8", newline="") as handle:
            race_ids = sorted(
                {
                    str(row.get("race_id") or "")
                    for row in csv.DictReader(handle)
                    if row.get("race_id")
                }
            )
    source_reports = rolling_report.get("source_unified_evidence_reports") or []
    payload = {
        "sample_race_count": rolling_report.get("sample_race_count"),
        "sample_runner_rows": rolling_report.get("sample_runner_rows"),
        "runner_matrix_race_ids": race_ids,
        "source_unified_evidence_reports": sorted(str(item) for item in source_reports),
    }
    if race_ids:
        payload.pop("source_unified_evidence_reports")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def discover_rank_first_packets(evidence_root: Path) -> list[Path]:
    return sorted(
        path
        for path in evidence_root.glob("pre_race_gated_challenger_*rank_first_hypothesis_review*")
        if (path / "pre_race_gated_challenger_report.json").exists()
    )


def load_candidate_metric_rows(packet_dir: Path) -> list[dict[str, Any]]:
    path = packet_dir / "rank_first_hypothesis_candidate_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def evaluation_rows_for_packet(packet_dir: Path) -> list[dict[str, Any]]:
    report_path = packet_dir / "pre_race_gated_challenger_report.json"
    report = load_json(report_path)
    review = report.get("rank_first_hypothesis_gate_review") or {}
    if not isinstance(review, Mapping):
        review = {}
    runner_matrix = Path(str(report.get("runner_matrix_csv") or ""))
    if not runner_matrix.is_absolute():
        runner_matrix = ROOT / runner_matrix
    rolling_report = load_json(runner_matrix.parent / "rolling_model_comparison_report.json")
    if not rolling_report:
        rolling_report = load_json(runner_matrix.parent / "rolling_model_comparison_report.json")
    signature = (
        sample_signature(
            runner_matrix_csv=runner_matrix,
            rolling_report=rolling_report,
        )
        if rolling_report
        else "missing_rolling_sample"
    )
    rows: list[dict[str, Any]] = []
    for metric in load_candidate_metric_rows(packet_dir):
        deltas = parse_mapping(metric.get("candidate_minus_market"))
        candidate_key = str(metric.get("candidate_key") or "")
        rows.append(
            {
                "packet_dir": relpath(packet_dir),
                "report_path": relpath(report_path),
                "generated_at": report.get("generated_at"),
                "review_status": review.get("status"),
                "candidate_key": candidate_key,
                "hypothesis_dimension": metric.get("hypothesis_dimension"),
                "hypothesis_dimension_value": metric.get("hypothesis_dimension_value"),
                "hypothesis_source_race_count": finite_int(
                    metric.get("hypothesis_source_race_count")
                ),
                "gate_triggered_race_count": finite_int(
                    metric.get("gate_triggered_race_count")
                )
                or 0,
                "race_count": finite_int(metric.get("race_count")) or 0,
                "top1": finite_float(metric.get("top1")),
                "top3": finite_float(metric.get("top3")),
                "mean_winner_rank": finite_float(metric.get("mean_winner_rank")),
                "brier": finite_float(metric.get("brier")),
                "logloss": finite_float(metric.get("logloss")),
                "top1_delta_vs_market": finite_float(deltas.get("top1")),
                "top3_delta_vs_market": finite_float(deltas.get("top3")),
                "mean_winner_rank_delta_vs_market": finite_float(
                    deltas.get("mean_winner_rank")
                ),
                "brier_delta_vs_market": finite_float(deltas.get("brier")),
                "logloss_delta_vs_market": finite_float(deltas.get("logloss")),
                "sample_signature": signature,
                "rolling_sample_race_count": rolling_report.get("sample_race_count"),
                "rolling_sample_runner_rows": rolling_report.get("sample_runner_rows"),
            }
        )
    return rows


def readiness_for_candidate(
    *,
    latest: Mapping[str, Any],
    distinct_sample_count: int,
    min_triggered_races: int,
    min_distinct_samples: int,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if distinct_sample_count < min_distinct_samples:
        blockers.append("needs_distinct_future_sample")
    if int(latest.get("gate_triggered_race_count") or 0) < min_triggered_races:
        blockers.append("triggered_race_count_below_directional_floor")
    if (latest.get("top1_delta_vs_market") or 0.0) <= 0:
        blockers.append("top1_not_above_market")
    if (latest.get("top3_delta_vs_market") or 0.0) < 0:
        blockers.append("top3_below_market")
    if (latest.get("mean_winner_rank_delta_vs_market") or 999.0) > 0:
        blockers.append("mean_winner_rank_worse_than_market")
    if (latest.get("brier_delta_vs_market") or 999.0) > 0:
        blockers.append("brier_worse_than_market")
    if (latest.get("logloss_delta_vs_market") or 999.0) > 0:
        blockers.append("logloss_worse_than_market")
    if not blockers:
        return "RANK_FIRST_HYPOTHESIS_DIRECTIONAL_READY", []
    if blockers == ["needs_distinct_future_sample"]:
        return "RANK_FIRST_HYPOTHESIS_WAITING_FOR_FRESH_SAMPLE", blockers
    return "RANK_FIRST_HYPOTHESIS_COLLECTING", blockers


def summarize_candidates(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    min_triggered_races: int,
    min_distinct_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluations:
        grouped[str(row.get("candidate_key") or "")].append(row)
    summaries: list[dict[str, Any]] = []
    for candidate_key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: str(row.get("generated_at") or ""))
        latest = ordered[-1]
        distinct_signatures = sorted({str(row.get("sample_signature")) for row in rows})
        status, blockers = readiness_for_candidate(
            latest=latest,
            distinct_sample_count=len(distinct_signatures),
            min_triggered_races=min_triggered_races,
            min_distinct_samples=min_distinct_samples,
        )
        summaries.append(
            {
                "candidate_key": candidate_key,
                "hypothesis_dimension": latest.get("hypothesis_dimension"),
                "hypothesis_dimension_value": latest.get("hypothesis_dimension_value"),
                "status": status,
                "blockers": blockers,
                "evaluation_count": len(rows),
                "distinct_sample_signature_count": len(distinct_signatures),
                "latest_generated_at": latest.get("generated_at"),
                "latest_gate_triggered_race_count": latest.get(
                    "gate_triggered_race_count"
                ),
                "minimum_triggered_races_for_directional_read": min_triggered_races,
                "latest_top1_delta_vs_market": latest.get("top1_delta_vs_market"),
                "latest_top3_delta_vs_market": latest.get("top3_delta_vs_market"),
                "latest_mean_winner_rank_delta_vs_market": latest.get(
                    "mean_winner_rank_delta_vs_market"
                ),
                "latest_brier_delta_vs_market": latest.get("brier_delta_vs_market"),
                "latest_logloss_delta_vs_market": latest.get("logloss_delta_vs_market"),
                "latest_packet_dir": latest.get("packet_dir"),
                "latest_sample_signature": latest.get("sample_signature"),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            item.get("status") != "RANK_FIRST_HYPOTHESIS_DIRECTIONAL_READY",
            -(int(item.get("distinct_sample_signature_count") or 0)),
            -(float(item.get("latest_top1_delta_vs_market") or -999.0)),
            str(item.get("candidate_key") or ""),
        ),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summary_markdown(report: Mapping[str, Any]) -> str:
    best = report.get("best_candidate") or {}
    return "\n".join(
        [
            "# Rank-First Hypothesis Watchlist",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Packet count: `{report.get('packet_count')}`",
            f"- Evaluation count: `{report.get('evaluation_count')}`",
            f"- Candidate count: `{report.get('candidate_count')}`",
            f"- Directional-ready candidates: `{report.get('directional_ready_candidate_count')}`",
            f"- Best candidate: `{best.get('candidate_key')}`",
            f"- Best candidate status: `{best.get('status')}`",
            f"- Best candidate distinct samples: `{best.get('distinct_sample_signature_count')}` / `{report.get('minimum_distinct_samples_for_directional_read')}`",
            f"- Best candidate triggered races: `{best.get('latest_gate_triggered_race_count')}` / `{best.get('minimum_triggered_races_for_directional_read')}`",
            f"- Best candidate Top1 delta vs market: `{best.get('latest_top1_delta_vs_market')}`",
            f"- Best candidate LogLoss delta vs market: `{best.get('latest_logloss_delta_vs_market')}`",
            f"- Best candidate blockers: `{best.get('blockers')}`",
            "",
            "No production training, promotion, registry mutation, pointer update, DB write, label write, odds write, betting/EV action, snapshot rewrite, manifest rewrite, or TGR enablement was performed.",
            "",
        ]
    )


def build_packet(
    *,
    evidence_root: Path,
    output_dir: Path,
    min_triggered_races: int = DEFAULT_MIN_TRIGGERED_RACES,
    min_distinct_samples: int = DEFAULT_MIN_DISTINCT_SAMPLES,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    packet_dirs = discover_rank_first_packets(evidence_root)
    evaluations: list[dict[str, Any]] = []
    for packet_dir in packet_dirs:
        evaluations.extend(evaluation_rows_for_packet(packet_dir))
    candidates = summarize_candidates(
        evaluations,
        min_triggered_races=min_triggered_races,
        min_distinct_samples=min_distinct_samples,
    )
    best_candidate = candidates[0] if candidates else {}
    directional_ready_count = sum(
        1
        for item in candidates
        if item.get("status") == "RANK_FIRST_HYPOTHESIS_DIRECTIONAL_READY"
    )
    blockers: list[str] = []
    if not packet_dirs:
        blockers.append("rank_first_hypothesis_packets_missing")
    if not evaluations:
        blockers.append("rank_first_hypothesis_evaluations_missing")
    if directional_ready_count == 0:
        blockers.append("no_directional_ready_rank_first_hypotheses")
    report = {
        "schema_version": "rank_first_hypothesis_watchlist_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": FINAL_READY if evaluations else FINAL_COLLECTING,
        "output_dir": relpath(output_dir),
        "evidence_root": relpath(evidence_root),
        "watchlist_csv": relpath(output_dir / WATCHLIST_CSV),
        "evaluations_csv": relpath(output_dir / EVALUATIONS_CSV),
        "packet_count": len(packet_dirs),
        "evaluation_count": len(evaluations),
        "candidate_count": len(candidates),
        "directional_ready_candidate_count": directional_ready_count,
        "minimum_triggered_races_for_directional_read": min_triggered_races,
        "minimum_distinct_samples_for_directional_read": min_distinct_samples,
        "best_candidate": best_candidate,
        "candidates": candidates,
        "blockers": blockers,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_json(output_dir / REPORT_FILE, report)
    write_csv(
        output_dir / WATCHLIST_CSV,
        candidates,
        [
            "candidate_key",
            "hypothesis_dimension",
            "hypothesis_dimension_value",
            "status",
            "evaluation_count",
            "distinct_sample_signature_count",
            "latest_gate_triggered_race_count",
            "minimum_triggered_races_for_directional_read",
            "latest_top1_delta_vs_market",
            "latest_top3_delta_vs_market",
            "latest_mean_winner_rank_delta_vs_market",
            "latest_brier_delta_vs_market",
            "latest_logloss_delta_vs_market",
            "blockers",
        ],
    )
    write_csv(
        output_dir / EVALUATIONS_CSV,
        evaluations,
        [
            "packet_dir",
            "generated_at",
            "candidate_key",
            "hypothesis_dimension",
            "hypothesis_dimension_value",
            "gate_triggered_race_count",
            "top1_delta_vs_market",
            "top3_delta_vs_market",
            "mean_winner_rank_delta_vs_market",
            "brier_delta_vs_market",
            "logloss_delta_vs_market",
            "sample_signature",
            "rolling_sample_race_count",
            "rolling_sample_runner_rows",
        ],
    )
    write_text(output_dir / SUMMARY_FILE, summary_markdown(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--min-triggered-races",
        type=int,
        default=DEFAULT_MIN_TRIGGERED_RACES,
    )
    parser.add_argument(
        "--min-distinct-samples",
        type=int,
        default=DEFAULT_MIN_DISTINCT_SAMPLES,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir
        or args.evidence_root / f"rank_first_hypothesis_watchlist_{now_id(generated_at)}"
    )
    report = build_packet(
        evidence_root=args.evidence_root,
        output_dir=output_dir,
        min_triggered_races=args.min_triggered_races,
        min_distinct_samples=args.min_distinct_samples,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
