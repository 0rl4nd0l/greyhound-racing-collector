#!/usr/bin/env python3
"""Evaluate a report-only no-box actual-win smoke packet.

This script consumes rows produced by build_winner_only_no_box_rehearsal_packet.
It does not train or promote models. It verifies the actual-win/no-box data
contract and scores deterministic baselines so the smoke gate has measurable
Top1/Top3/Brier evidence before any feature-model work is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "winner_only_no_box_actual_win_smoke_eval_v1"
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}
FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL = [
    "write_official_safe_labels",
    "mutate_db",
    "regenerate_canonical_dataset",
    "train_or_promote_model",
    "update_registry",
    "enable_tgr",
    "betting_or_ev_action",
]
FORBIDDEN_ROW_FIELDS = {
    "box_number",
    "official_box_number",
    "db_box_number",
    "finish_position",
    "official_finish_position",
    "db_finish_position",
    "db_result_position",
    "result_position",
    "placing",
    "scraped_finish_position",
}
REQUIRED_ROW_FIELDS = {
    "race_id",
    "dog_name_key",
    "dog_name",
    "actual_win",
    "box_features_allowed",
    "finish_order_labels_allowed",
    "top3_labels_allowed",
    "label_write_approved",
}
PREDICTION_FIELDS = [
    "baseline",
    "race_id",
    "race_date",
    "venue",
    "race_number",
    "dog_name_key",
    "dog_name",
    "score",
    "probability",
    "predicted_rank",
    "actual_win",
    "candidate_kind",
    "field_scope",
    "field_complete_for_ranking",
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_row_not_object:{line_number}")
            rows.append(row)
    return rows


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root = root or ROOT
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{logical}") from exc
    return resolved, relative


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    return _repo_output_path(path, root)[1]


def _assert_output_dir_safe(output_dir: Path) -> Path:
    resolved, relative = _repo_output_path(output_dir)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _race_sort_key(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    first = rows[0] if rows else {}
    return (str(first.get("race_date") or ""), str(first.get("race_id") or ""))


def _dog_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("dog_name_key") or ""), str(row.get("dog_name") or ""))


def _group_by_race(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("race_id") or "")].append(dict(row))
    return dict(grouped)


def _brier(rows: Sequence[Mapping[str, Any]], probability_key: str = "probability") -> float | None:
    if not rows:
        return None
    return sum(
        (float(row.get(probability_key) or 0.0) - int(row.get("actual_win") or 0)) ** 2
        for row in rows
    ) / len(rows)


def _log_loss(rows: Sequence[Mapping[str, Any]], probability_key: str = "probability") -> float | None:
    if not rows:
        return None
    eps = 1e-12
    total = 0.0
    for row in rows:
        prob = min(1.0 - eps, max(eps, float(row.get(probability_key) or 0.0)))
        actual = int(row.get("actual_win") or 0)
        total += -(actual * math.log(prob) + (1 - actual) * math.log(1.0 - prob))
    return total / len(rows)


def _rank_predictions(
    *,
    race_rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    baseline: str,
) -> list[dict[str, Any]]:
    score_values = [float(scores.get(str(row.get("dog_name_key") or ""), 0.0)) for row in race_rows]
    if score_values:
        minimum = min(score_values)
        if minimum <= 0:
            score_values = [value - minimum + 1e-9 for value in score_values]
    score_sum = sum(score_values)
    if score_sum <= 0:
        score_values = [1.0 for _ in race_rows]
        score_sum = float(len(score_values))
    scored = []
    for row, score in zip(race_rows, score_values, strict=False):
        scored.append(
            {
                **dict(row),
                "baseline": baseline,
                "score": score,
                "probability": score / score_sum,
            }
        )
    scored.sort(key=lambda row: (-float(row["probability"]), _dog_sort_key(row)))
    for rank, row in enumerate(scored, start=1):
        row["predicted_rank"] = rank
    return scored


def _uniform_predictions(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    predictions = []
    for _, race_rows in sorted(grouped.items(), key=lambda item: _race_sort_key(item[1])):
        scores = {str(row.get("dog_name_key") or ""): 1.0 for row in race_rows}
        predictions.extend(_rank_predictions(race_rows=race_rows, scores=scores, baseline="uniform_field"))
    return predictions


def _rolling_dog_prior_predictions(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    predictions = []
    starts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    for _, race_rows in sorted(grouped.items(), key=lambda item: _race_sort_key(item[1])):
        scores = {}
        for row in race_rows:
            dog_key = str(row.get("dog_name_key") or "")
            scores[dog_key] = (wins[dog_key] + 1.0) / (starts[dog_key] + 2.0)
        race_predictions = _rank_predictions(
            race_rows=race_rows,
            scores=scores,
            baseline="rolling_dog_name_prior",
        )
        predictions.extend(race_predictions)
        for row in race_rows:
            dog_key = str(row.get("dog_name_key") or "")
            starts[dog_key] += 1
            wins[dog_key] += int(row.get("actual_win") or 0)
    return predictions


def _dog_form_score(row: Mapping[str, Any]) -> float:
    score = 0.0

    def add(name: str, weight: float, *, cap: float | None = None) -> None:
        nonlocal score
        value = _safe_float(row.get(f"feature_{name}"))
        if value is None:
            return
        if cap is not None:
            value = max(-cap, min(cap, value))
        score += weight * value

    for name in (
        "recent_win_rate_5",
        "recent_place_rate_5",
        "career_win_rate",
        "career_place_rate",
        "win_rate_same_distance",
        "place_rate_same_distance",
        "win_rate_same_venue",
        "place_rate_same_venue",
        "same_grade_win_rate",
        "same_grade_place_rate",
    ):
        add(name, 2.0)
    for name in (
        "prior_start_count",
        "prior_same_distance_start_count",
        "starts_same_distance",
        "starts_same_venue",
        "same_distance_same_grade_start_count",
        "same_distance_venue_start_count",
        "same_grade_start_count",
    ):
        add(name, 0.03, cap=20.0)
    for name in (
        "recent_finish_mean_3",
        "recent_finish_mean_5",
        "recent_finish_best_5",
        "career_avg_finish",
        "career_best_finish",
    ):
        add(name, -0.12)
    for name in (
        "days_since_last_start",
        "days_since_last_same_distance_start",
    ):
        add(name, -0.004, cap=90.0)
    for name in (
        "sectional_missing_rate_5",
        "recent_time_std_5",
        "career_time_std",
    ):
        add(name, -0.05)
    return score


def _dog_form_heuristic_predictions(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    predictions = []
    for _, race_rows in sorted(grouped.items(), key=lambda item: _race_sort_key(item[1])):
        scores = {
            str(row.get("dog_name_key") or ""): _dog_form_score(row)
            for row in race_rows
        }
        predictions.extend(
            _rank_predictions(
                race_rows=race_rows,
                scores=scores,
                baseline="dog_form_heuristic",
            )
        )
    return predictions


def _baseline_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = _group_by_race(predictions)
    per_race = []
    for race_id, race_rows in sorted(grouped.items(), key=lambda item: _race_sort_key(item[1])):
        ordered = sorted(race_rows, key=lambda row: int(row.get("predicted_rank") or 999999))
        winners = [row for row in ordered if int(row.get("actual_win") or 0) == 1]
        winner_rank = int(winners[0].get("predicted_rank")) if winners else None
        field_size = len(ordered)
        per_race.append(
            {
                "race_id": race_id,
                "field_size": field_size,
                "winner_rank": winner_rank,
                "top1_hit": winner_rank == 1,
                "top3_hit": winner_rank is not None and winner_rank <= min(3, field_size),
                "winner_probability": float(winners[0].get("probability")) if winners else None,
                "probability_sum": sum(float(row.get("probability") or 0.0) for row in ordered),
                "field_complete_for_ranking": all(
                    row.get("field_complete_for_ranking") is True for row in ordered
                ),
            }
        )
    race_count = len(per_race)
    top1_hits = sum(1 for row in per_race if row["top1_hit"])
    top3_hits = sum(1 for row in per_race if row["top3_hit"])
    random_top1 = [
        1.0 / row["field_size"]
        for row in per_race
        if row["field_size"]
    ]
    random_top3 = [
        min(3, row["field_size"]) / row["field_size"]
        for row in per_race
        if row["field_size"]
    ]
    return {
        "race_count": race_count,
        "row_count": len(predictions),
        "top1_accuracy": top1_hits / race_count if race_count else None,
        "top3_hit_rate": top3_hits / race_count if race_count else None,
        "mean_winner_rank": (
            sum(row["winner_rank"] for row in per_race if row["winner_rank"] is not None) / race_count
            if race_count
            else None
        ),
        "mean_winner_probability": (
            sum(row["winner_probability"] for row in per_race if row["winner_probability"] is not None)
            / race_count
            if race_count
            else None
        ),
        "brier": _brier(predictions),
        "log_loss": _log_loss(predictions),
        "expected_random_top1": sum(random_top1) / len(random_top1) if random_top1 else None,
        "expected_random_top3": sum(random_top3) / len(random_top3) if random_top3 else None,
        "probability_sum_check": {
            "status": (
                "PASS"
                if all(abs(float(row["probability_sum"]) - 1.0) <= 1e-9 for row in per_race)
                else "FAIL"
            ),
            "max_abs_error": max(
                (abs(float(row["probability_sum"]) - 1.0) for row in per_race),
                default=0.0,
            ),
        },
        "per_race": per_race,
    }


def _validate_rows(
    *,
    rehearsal_packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    expected_races: int | None,
    min_smoke_races: int,
) -> dict[str, Any]:
    failures = []
    warnings = []
    packet_summary = rehearsal_packet.get("summary") or {}
    grouped = _group_by_race(rows)
    if expected_races is not None and len(grouped) != expected_races:
        failures.append(f"expected_races_mismatch:{expected_races}:{len(grouped)}")
    if len(grouped) < min_smoke_races:
        failures.append(f"insufficient_smoke_races:{len(grouped)}:{min_smoke_races}")
    if packet_summary.get("can_evaluate_model") is not True:
        warnings.append("source_packet_can_evaluate_model_not_true")
    if packet_summary.get("no_box_row_policy_pass") is not True:
        failures.append("source_packet_no_box_row_policy_not_pass")

    candidate_kind_counts: Counter[str] = Counter()
    venue_counts: Counter[str] = Counter()
    race_field_sizes: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        missing = sorted(field for field in REQUIRED_ROW_FIELDS if field not in row)
        if missing:
            failures.append(f"row_{index}_missing_required_fields:{','.join(missing)}")
        forbidden = sorted(FORBIDDEN_ROW_FIELDS & set(row))
        if forbidden:
            failures.append(f"row_{index}_forbidden_fields_present:{','.join(forbidden)}")
        if row.get("box_features_allowed") is not False:
            failures.append(f"row_{index}_box_features_allowed_not_false")
        if row.get("finish_order_labels_allowed") is not False:
            failures.append(f"row_{index}_finish_order_labels_allowed_not_false")
        if row.get("top3_labels_allowed") is not False:
            failures.append(f"row_{index}_top3_labels_allowed_not_false")
        if row.get("label_write_approved") is not False:
            failures.append(f"row_{index}_label_write_approved_not_false")
        if _safe_int(row.get("actual_win")) not in (0, 1):
            failures.append(f"row_{index}_actual_win_not_binary")
        candidate_kind_counts[str(row.get("candidate_kind") or "UNKNOWN")] += 1
        venue_counts[str(row.get("venue") or "UNKNOWN")] += 1

    for race_id, race_rows in grouped.items():
        positive_count = sum(int(row.get("actual_win") or 0) for row in race_rows)
        race_field_sizes[race_id] = len(race_rows)
        if positive_count != 1:
            failures.append(f"race_{race_id}_actual_win_positive_count:{positive_count}")

    feature_columns = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith("feature_") and _safe_float(value) is not None
        }
    )
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "race_count": len(grouped),
        "row_count": len(rows),
        "candidate_kind_counts": dict(sorted(candidate_kind_counts.items())),
        "venue_counts": dict(sorted(venue_counts.items())),
        "field_size_counts": dict(sorted(Counter(race_field_sizes.values()).items())),
        "feature_columns_present": feature_columns,
        "feature_model_status": (
            "READY_FOR_FEATURE_MODEL"
            if feature_columns
            else "SKIPPED_NO_PREDICTIVE_FEATURE_COLUMNS_IN_REHEARSAL_ROWS"
        ),
        "complete_field_races": sum(
            1
            for race_rows in grouped.values()
            if all(row.get("field_complete_for_ranking") is True for row in race_rows)
        ),
        "partial_field_races": sum(
            1
            for race_rows in grouped.values()
            if any(row.get("field_complete_for_ranking") is not True for row in race_rows)
        ),
    }


def evaluate_smoke_packet(
    *,
    rehearsal_packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    rehearsal_packet_path: str | None = None,
    rows_path: str | None = None,
    expected_races: int | None = None,
    min_smoke_races: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation = _validate_rows(
        rehearsal_packet=rehearsal_packet,
        rows=rows,
        expected_races=expected_races,
        min_smoke_races=min_smoke_races,
    )
    grouped = _group_by_race(rows)
    baseline_predictions = {
        "uniform_field": _uniform_predictions(grouped),
        "rolling_dog_name_prior": _rolling_dog_prior_predictions(grouped),
    }
    if validation["feature_columns_present"]:
        baseline_predictions["dog_form_heuristic"] = _dog_form_heuristic_predictions(grouped)
    baselines = {
        name: _baseline_metrics(predictions)
        for name, predictions in baseline_predictions.items()
    }
    all_predictions = [
        row
        for name in sorted(baseline_predictions)
        for row in baseline_predictions[name]
    ]
    ranking_ready = validation["complete_field_races"]
    status = "REPORT_ONLY_NO_BOX_ACTUAL_WIN_SMOKE_EVALUATED"
    if validation["status"] != "PASS":
        status = "REPORT_ONLY_NO_BOX_ACTUAL_WIN_SMOKE_FAILED_CONTRACT"
    elif validation["race_count"] < min_smoke_races:
        status = "REPORT_ONLY_NO_BOX_ACTUAL_WIN_SMOKE_UNDERPOWERED"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "report_only": True,
        "writes_performed": dict(WRITES_PERFORMED),
        "rehearsal_packet": rehearsal_packet_path,
        "rows_jsonl": rows_path,
        "validation": validation,
        "baselines": baselines,
        "model_training_status": "SKIPPED_REPORT_ONLY_BASELINES_ONLY",
        "feature_model_status": validation["feature_model_status"],
        "race_grouped_ranking_status": (
            "SKIPPED_INSUFFICIENT_COMPLETE_FIELD_RACES"
            if ranking_ready < 100
            else "READY_FOR_RACE_GROUPED_RANKING_EXPERIMENT"
        ),
        "race_grouped_ranking_ready_candidate_count": ranking_ready,
        "minimums": {
            "smoke_actual_win_races": min_smoke_races,
            "rolling_temporal_actual_win_races": 50,
            "race_grouped_ranking_complete_field_races": 100,
        },
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "recommended_next_action": (
            "join_pre_race_dog_form_features_without_box_race_calendar_leakage"
            if not validation["feature_columns_present"]
            else "run_report_only_no_box_feature_model_smoke"
        ),
    }, all_predictions


def write_outputs(output_dir: Path, report: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "no_box_actual_win_smoke_eval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "no_box_actual_win_smoke_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "no_box_actual_win_smoke_predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in predictions:
            writer.writerow({field: row.get(field) for field in PREDICTION_FIELDS})
    summary = [
        "# No-Box Actual-Win Smoke Evaluation",
        "",
        f"Status: `{report.get('status')}`.",
        "",
        "No DB writes, label writes, snapshot mutations, manifest mutations, model training, model promotion, registry updates, TGR enablement, betting decisions, or EV actions occurred.",
        "",
        "## Contract",
        "",
        f"- Validation status: `{report['validation'].get('status')}`",
        f"- Races: `{report['validation'].get('race_count')}`",
        f"- Rows: `{report['validation'].get('row_count')}`",
        f"- Complete-field races: `{report['validation'].get('complete_field_races')}`",
        f"- Partial-field races: `{report['validation'].get('partial_field_races')}`",
        f"- Feature model status: `{report.get('feature_model_status')}`",
        f"- Race-grouped ranking status: `{report.get('race_grouped_ranking_status')}`",
        "",
        "## Baselines",
    ]
    for name, metrics in report.get("baselines", {}).items():
        summary.extend(
            [
                "",
                f"### {name}",
                "",
                f"- Top1: `{metrics.get('top1_accuracy')}`",
                f"- Top3: `{metrics.get('top3_hit_rate')}`",
                f"- Brier: `{metrics.get('brier')}`",
                f"- Mean winner rank: `{metrics.get('mean_winner_rank')}`",
            ]
        )
    summary.extend(
        [
            "",
            "## Next",
            "",
            str(report.get("recommended_next_action")),
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal-packet", required=True)
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-races", type=int)
    parser.add_argument("--min-smoke-races", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet_path = Path(args.rehearsal_packet).expanduser().resolve()
    rows_path = Path(args.rows_jsonl).expanduser().resolve()
    report, predictions = evaluate_smoke_packet(
        rehearsal_packet=_load_json(packet_path),
        rows=_load_jsonl(rows_path),
        rehearsal_packet_path=str(packet_path),
        rows_path=str(rows_path),
        expected_races=args.expected_races,
        min_smoke_races=args.min_smoke_races,
    )
    write_outputs(Path(args.output_dir), report, predictions)
    print(json.dumps({"status": report["status"], "validation": report["validation"], "baselines": report["baselines"]}, indent=2, sort_keys=True))
    return 1 if report["validation"]["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
