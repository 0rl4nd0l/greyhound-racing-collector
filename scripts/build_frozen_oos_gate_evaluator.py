#!/usr/bin/env python3
"""Evaluate a frozen source-safe gate on future out-of-sample matrix rows.

This report-only helper compares a freeze runner matrix with a later runner
matrix, excludes freeze race IDs, applies one source-safe selector, and checks
whether the frozen challenger beats ``market_only_implied`` on the selected OOS
races. It writes artifacts only when an output directory is supplied. It does
not train, capture data, write databases, mutate registries, promote models,
enable TGR, or perform EV/betting work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "frozen_oos_gate_evaluator_v1"
REPORT_FILE = "frozen_oos_gate_report.json"
RACE_METRICS_CSV = "oos_segment_race_metrics.csv"
SUMMARY_FILE = "SUMMARY.md"
GRID_REPORT_FILE = "frozen_oos_grid_report.json"
GRID_RESULTS_CSV = "frozen_oos_grid_results.csv"
GRID_SUMMARY_FILE = "GRID_SUMMARY.md"
FINAL_DATA_MISSING = "DATA_MISSING"
FINAL_BLOCKED = "BLOCKED_KEEP_BASELINE"
FINAL_SEGMENT_DESIGN_ONLY = "SEGMENT_DESIGN_ONLY"
FINAL_READY_FOR_OWNER = "VALIDATION_READY_FOR_OWNER_REVIEW"

NO_WRITE_GUARANTEES = {
    "live_db_write": False,
    "official_result_capture": False,
    "live_odds_capture": False,
    "model_fit": False,
    "model_artifact_write": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "promotion": False,
    "betting": False,
    "ev_output": False,
    "tgr_enabled": False,
}

SOURCE_SAFE_SELECTOR_FIELDS = {
    "runner_count",
    "venue",
    "race_number",
    "odds_capture_mode",
    "odds_level",
    "market_favourite_odds_band",
    "market_favourite_odds_decimal",
}
DEFAULT_GRID_SELECTOR_FIELDS = (
    "runner_count",
    "market_favourite_odds_band",
    "odds_capture_mode",
    "odds_level",
    "venue",
    "race_number",
)
SELECTOR_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in"}
PROTECTED_OUTPUT_PREFIXES = (
    "artifacts/full_evidence_orchestration_20260525",
    "artifacts/prediction_snapshots",
    "model_registry",
    "docs/model_registry",
    "ml_models_v4",
    "advanced_models",
)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "race_id",
        "race_date",
        "venue",
        "race_number",
        "runner_count",
        "market_winner_rank",
        "candidate_winner_rank",
        "market_winner_probability",
        "candidate_winner_probability",
        "market_logloss",
        "candidate_logloss",
        "candidate_minus_market_logloss",
        "winner_promoted",
        "winner_demoted",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_grid_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "freeze_packet",
        "freeze_generated_at",
        "freeze_race_count",
        "oos_packet",
        "selector_field",
        "selector_value",
        "final_status",
        "selected_race_count",
        "evaluable_race_count",
        "top1_delta",
        "top3_delta",
        "mean_winner_rank_delta",
        "brier_delta",
        "logloss_delta",
        "candidate_promoted_winner_count",
        "candidate_demoted_winner_count",
        "blockers",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field), sort_keys=True)
                        if field == "blockers"
                        else row.get(field)
                    )
                    for field in fieldnames
                }
            )


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[str(path.relative_to(output_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return {
        "schema_version": "frozen_oos_gate_output_manifest_v1",
        "output_dir": str(output_dir),
        "files": files,
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or(value: Any, default: float) -> float:
    parsed = _safe_float(value)
    return parsed if parsed is not None else default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _group_by_race(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        race_id = str(row.get("race_id") or "").strip()
        if race_id:
            grouped[race_id].append(dict(row))
    return dict(grouped)


def _race_value(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    first = rows[0] if rows else {}
    if field == "runner_count":
        direct = _safe_int(first.get("runner_count"))
        return direct if direct is not None else len(rows)
    return first.get(field)


def _parse_selector_value(value: str) -> Any:
    text = str(value)
    if "," in text:
        return [_parse_selector_value(item.strip()) for item in text.split(",")]
    as_int = _safe_int(text)
    as_float = _safe_float(text)
    if as_float is not None and "." in text:
        return as_float
    if as_int is not None and str(as_int) == text:
        return as_int
    return text


def validate_selector(field: str, operator: str) -> None:
    if field not in SOURCE_SAFE_SELECTOR_FIELDS:
        raise ValueError(f"selector_field_not_source_safe:{field}")
    if operator not in SELECTOR_OPERATORS:
        raise ValueError(f"selector_operator_not_supported:{operator}")


def _value_matches(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "in":
        values = expected if isinstance(expected, list) else [expected]
        return any(_value_matches(actual, "eq", item) for item in values)

    actual_float = _safe_float(actual)
    expected_float = _safe_float(expected)
    if operator in {"gt", "gte", "lt", "lte"}:
        if actual_float is None or expected_float is None:
            return False
        if operator == "gt":
            return actual_float > expected_float
        if operator == "gte":
            return actual_float >= expected_float
        if operator == "lt":
            return actual_float < expected_float
        return actual_float <= expected_float

    if actual_float is not None and expected_float is not None:
        equal = actual_float == expected_float
    else:
        equal = str(actual).strip() == str(expected).strip()
    return equal if operator == "eq" else not equal


def _candidate_keys(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("candidate_key") or "").strip() for row in rows}


def _market_keys(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("market_candidate_key") or "").strip() for row in rows}


def _evaluate_candidate(
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    probability_field: str,
    rank_field: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    skipped: Counter[str] = Counter()
    race_count = 0
    top1 = 0
    top3 = 0
    rank_sum = 0.0
    brier_sum = 0.0
    logloss_sum = 0.0
    race_rows_out: list[dict[str, Any]] = []

    for race_id in sorted(grouped_rows):
        rows = list(grouped_rows[race_id])
        winners = [row for row in rows if _truthy(row.get("is_winner"))]
        if len(winners) != 1:
            skipped["missing_or_ambiguous_winner"] += 1
            continue
        winner = winners[0]
        probabilities = [_safe_float(row.get(probability_field)) for row in rows]
        if any(value is None for value in probabilities):
            skipped[f"missing_{probability_field}"] += 1
            continue
        winner_index = next(index for index, row in enumerate(rows) if row is winner)
        winner_probability = probabilities[winner_index]
        winner_rank = _safe_int(winner.get(rank_field))
        if winner_probability is None or winner_rank is None:
            skipped[f"missing_{rank_field}"] += 1
            continue

        race_count += 1
        top1 += 1 if winner_rank == 1 else 0
        top3 += 1 if winner_rank <= 3 else 0
        rank_sum += winner_rank
        brier_sum += sum(
            (probability - (1.0 if row is winner else 0.0)) ** 2
            for row, probability in zip(rows, probabilities)
            if probability is not None
        )
        logloss = -math.log(max(winner_probability, 1e-15))
        logloss_sum += logloss
        race_rows_out.append(
            {
                "race_id": race_id,
                "race_date": winner.get("race_date"),
                "venue": winner.get("venue"),
                "race_number": winner.get("race_number"),
                "runner_count": _race_value(rows, "runner_count"),
                f"{rank_field}": winner_rank,
                f"{probability_field}": winner_probability,
                f"{probability_field}_logloss": logloss,
            }
        )

    if race_count == 0:
        return {
            "status": FINAL_DATA_MISSING,
            "race_count": 0,
            "blockers": ["no_evaluable_races"],
        }, race_rows_out, skipped

    return {
        "status": "EVALUATED",
        "race_count": race_count,
        "top1": top1 / race_count,
        "top3": top3 / race_count,
        "mean_winner_rank": rank_sum / race_count,
        "brier": brier_sum / race_count,
        "logloss": logloss_sum / race_count,
    }, race_rows_out, skipped


def _metric_deltas(market: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("top1", "top3", "mean_winner_rank", "brier", "logloss"):
        market_value = _safe_float(market.get(key))
        candidate_value = _safe_float(candidate.get(key))
        output[key] = (
            candidate_value - market_value
            if market_value is not None and candidate_value is not None
            else None
        )
    return output


def _winner_movements(grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    promoted = 0
    demoted = 0
    same = 0
    for rows in grouped_rows.values():
        winners = [row for row in rows if _truthy(row.get("is_winner"))]
        if len(winners) != 1:
            continue
        winner = winners[0]
        market_rank = _safe_int(winner.get("market_rank"))
        candidate_rank = _safe_int(winner.get("candidate_rank"))
        if market_rank is None or candidate_rank is None:
            continue
        if candidate_rank < market_rank:
            promoted += 1
        elif candidate_rank > market_rank:
            demoted += 1
        else:
            same += 1
    return {
        "candidate_promoted_winner_count": promoted,
        "candidate_demoted_winner_count": demoted,
        "candidate_same_winner_rank_count": same,
    }


def _concentration(grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    race_dates = Counter()
    venues = Counter()
    for rows in grouped_rows.values():
        first = rows[0] if rows else {}
        race_dates[str(first.get("race_date") or "DATA_MISSING")] += 1
        venues[str(first.get("venue") or "DATA_MISSING")] += 1
    race_count = len(grouped_rows)
    max_date_count = max(race_dates.values(), default=0)
    return {
        "race_date_counts": dict(sorted(race_dates.items())),
        "venue_counts": dict(sorted(venues.items())),
        "race_date_count": len(race_dates),
        "venue_count": len(venues),
        "max_single_race_date_count": max_date_count,
        "max_single_race_date_share": (max_date_count / race_count) if race_count else None,
    }


def _combined_race_rows(
    market_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    market_by_race = {str(row.get("race_id")): row for row in market_rows}
    candidate_by_race = {str(row.get("race_id")): row for row in candidate_rows}
    output: list[dict[str, Any]] = []
    for race_id in sorted(grouped_rows):
        market = market_by_race.get(race_id)
        candidate = candidate_by_race.get(race_id)
        if not market or not candidate:
            continue
        market_rank = _safe_int(market.get("market_rank"))
        candidate_rank = _safe_int(candidate.get("candidate_rank"))
        market_logloss = _safe_float(market.get("market_probability_logloss"))
        candidate_logloss = _safe_float(candidate.get("candidate_probability_logloss"))
        output.append(
            {
                "race_id": race_id,
                "race_date": market.get("race_date"),
                "venue": market.get("venue"),
                "race_number": market.get("race_number"),
                "runner_count": market.get("runner_count"),
                "market_winner_rank": market_rank,
                "candidate_winner_rank": candidate_rank,
                "market_winner_probability": market.get("market_probability"),
                "candidate_winner_probability": candidate.get("candidate_probability"),
                "market_logloss": market_logloss,
                "candidate_logloss": candidate_logloss,
                "candidate_minus_market_logloss": (
                    candidate_logloss - market_logloss
                    if candidate_logloss is not None and market_logloss is not None
                    else None
                ),
                "winner_promoted": bool(
                    market_rank is not None
                    and candidate_rank is not None
                    and candidate_rank < market_rank
                ),
                "winner_demoted": bool(
                    market_rank is not None
                    and candidate_rank is not None
                    and candidate_rank > market_rank
                ),
            }
        )
    return output


def _gate_decision(
    *,
    race_count: int,
    deltas: Mapping[str, Any],
    movements: Mapping[str, int],
    concentration: Mapping[str, Any],
    min_oos_races: int,
    stability_races: int,
    promotion_review_races: int,
    min_race_dates_for_stability: int,
    min_venues_for_stability: int,
    max_single_date_share: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    if race_count < min_oos_races:
        blockers.append("oos_race_count_below_floor")
        return {
            "final_status": FINAL_DATA_MISSING,
            "gate_aligned": False,
            "hard_gates_pass": False,
            "materiality_gates_pass": False,
            "concentration_guard_pass": False,
            "promotion_review_floor_met": False,
            "blockers": blockers,
        }

    hard_gates = {
        "top1_delta_gte_0": _float_or(deltas.get("top1"), 0.0) >= 0.0,
        "top3_delta_gte_0": _float_or(deltas.get("top3"), 0.0) >= 0.0,
        "mean_winner_rank_delta_lte_0": _float_or(
            deltas.get("mean_winner_rank"), 999.0
        )
        <= 0.0,
        "brier_delta_lte_0": _float_or(deltas.get("brier"), 999.0) <= 0.0,
        "logloss_delta_lte_0": _float_or(deltas.get("logloss"), 999.0) <= 0.0,
        "promoted_winners_gte_demoted_winners": (
            int(movements.get("candidate_promoted_winner_count") or 0)
            >= int(movements.get("candidate_demoted_winner_count") or 0)
        ),
    }
    for key, passed in hard_gates.items():
        if not passed:
            blockers.append(key.replace("_gte_0", "_failed").replace("_lte_0", "_failed"))

    rank_materiality = (
        _float_or(deltas.get("top1"), 0.0) >= 0.02
        or _float_or(deltas.get("mean_winner_rank"), 999.0) <= -0.02
    )
    probability_materiality = (
        _float_or(deltas.get("brier"), 999.0) <= -0.001
        or _float_or(deltas.get("logloss"), 999.0) <= -0.005
    )
    if not rank_materiality:
        blockers.append("rank_materiality_gate_failed")
    if not probability_materiality:
        blockers.append("probability_materiality_gate_failed")

    if blockers:
        return {
            "final_status": FINAL_BLOCKED,
            "gate_aligned": False,
            "hard_gates": hard_gates,
            "hard_gates_pass": all(hard_gates.values()),
            "rank_materiality_gate_pass": rank_materiality,
            "probability_materiality_gate_pass": probability_materiality,
            "materiality_gates_pass": rank_materiality and probability_materiality,
            "concentration_guard_pass": False,
            "promotion_review_floor_met": race_count >= promotion_review_races,
            "blockers": blockers,
        }

    promotion_review_floor_met = race_count >= promotion_review_races
    stability_floor_met = race_count >= stability_races
    concentration_guard_pass = True
    if stability_floor_met:
        if int(concentration.get("race_date_count") or 0) < min_race_dates_for_stability:
            concentration_guard_pass = False
            blockers.append("race_date_count_below_stability_minimum")
        if int(concentration.get("venue_count") or 0) < min_venues_for_stability:
            concentration_guard_pass = False
            blockers.append("venue_count_below_stability_minimum")
        share = _safe_float(concentration.get("max_single_race_date_share"))
        if share is not None and share > max_single_date_share:
            concentration_guard_pass = False
            blockers.append("single_race_date_concentration_above_maximum")

    if promotion_review_floor_met and concentration_guard_pass:
        final_status = FINAL_READY_FOR_OWNER
    else:
        final_status = FINAL_SEGMENT_DESIGN_ONLY
        if not promotion_review_floor_met:
            blockers.append("promotion_review_floor_not_met")
        if not concentration_guard_pass:
            blockers.append("concentration_guard_failed")

    return {
        "final_status": final_status,
        "gate_aligned": True,
        "hard_gates": hard_gates,
        "hard_gates_pass": True,
        "rank_materiality_gate_pass": rank_materiality,
        "probability_materiality_gate_pass": probability_materiality,
        "materiality_gates_pass": True,
        "concentration_guard_pass": concentration_guard_pass,
        "promotion_review_floor_met": promotion_review_floor_met,
        "stability_floor_met": stability_floor_met,
        "blockers": blockers,
    }


def _nearest_git_root(path: Path) -> Path | None:
    resolved = path.resolve()
    candidates = [resolved] if resolved.is_dir() else []
    candidates.extend(resolved.parents)
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def _assert_output_dir_safe(output_dir: Path, repo_root: Path | None = None) -> Path:
    resolved = output_dir.resolve()
    roots = []
    if repo_root is not None:
        roots.append(repo_root.resolve())
    else:
        roots.extend(
            root
            for root in (
                _nearest_git_root(resolved),
                _nearest_git_root(Path.cwd()),
                Path.cwd().resolve(),
            )
            if root is not None
        )

    for root in dict.fromkeys(roots):
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        relative_text = relative.as_posix()
        for prefix in PROTECTED_OUTPUT_PREFIXES:
            if relative_text == prefix or relative_text.startswith(prefix + "/"):
                raise ValueError(f"output_dir_protected:{prefix}")
    return resolved


def _assert_not_input_packet_output(output_dir: Path, input_paths: Sequence[Path]) -> None:
    for input_path in input_paths:
        input_dir = input_path.resolve().parent
        try:
            output_dir.resolve().relative_to(input_dir)
        except ValueError:
            continue
        raise ValueError(f"output_dir_must_not_write_inside_input_packet:{input_dir}")


def _selector_fields_from_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_GRID_SELECTOR_FIELDS
    fields = tuple(field.strip() for field in value.split(",") if field.strip())
    if not fields:
        raise ValueError("grid_selector_fields_empty")
    for field in fields:
        validate_selector(field, "eq")
    return fields


def _packet_sort_key(packet: Mapping[str, Any]) -> tuple[str, str]:
    generated_at = str(packet.get("generated_at") or "")
    return generated_at, str(packet.get("name") or "")


def discover_grid_packets(packet_root: Path, packet_name_regex: str) -> list[dict[str, Any]]:
    pattern = re.compile(packet_name_regex)
    packets: list[dict[str, Any]] = []
    for packet_dir in sorted(path for path in packet_root.iterdir() if path.is_dir()):
        if not pattern.match(packet_dir.name):
            continue
        report_path = packet_dir / "rolling_model_comparison_report.json"
        matrix_path = packet_dir / "market_residual_runner_matrix.csv"
        if not report_path.exists() or not matrix_path.exists():
            continue
        report = _load_json(report_path)
        packets.append(
            {
                "name": packet_dir.name,
                "path": packet_dir,
                "report_path": report_path,
                "matrix_path": matrix_path,
                "generated_at": report.get("generated_at"),
                "report": report,
            }
        )
    return sorted(packets, key=_packet_sort_key)


def _packet_by_name(packets: Sequence[Mapping[str, Any]], name: str | None) -> Mapping[str, Any]:
    if not packets:
        raise ValueError("grid_no_matching_packets")
    if name is None:
        return packets[-1]
    matches = [packet for packet in packets if packet.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"grid_oos_packet_not_found:{name}")
    return matches[0]


def _grid_values_by_field(
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    selector_fields: Sequence[str],
) -> dict[str, list[Any]]:
    values_by_field: dict[str, list[Any]] = {}
    for field in selector_fields:
        validate_selector(field, "eq")
        values: list[Any] = []
        seen: set[str] = set()
        for rows in grouped_rows.values():
            value = _race_value(rows, field)
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                values.append(value)
        values_by_field[field] = values
    return values_by_field


def _evaluate_grouped_gate(
    *,
    freeze_grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    oos_grouped_all: Mapping[str, Sequence[Mapping[str, Any]]],
    selector_field: str,
    selector_value: Any,
    candidate_key: str,
    market_candidate_key: str,
    min_oos_races: int,
    stability_races: int,
    promotion_review_races: int,
    min_race_dates_for_stability: int,
    min_venues_for_stability: int,
    max_single_date_share: float,
) -> dict[str, Any]:
    validate_selector(selector_field, "eq")
    freeze_race_ids = set(freeze_grouped)
    selected_grouped = {
        race_id: rows
        for race_id, rows in oos_grouped_all.items()
        if race_id not in freeze_race_ids
        and _value_matches(_race_value(rows, selector_field), "eq", selector_value)
    }
    selected_rows = [row for rows in selected_grouped.values() for row in rows]
    schema_blockers: list[str] = []
    if selected_rows and _candidate_keys(selected_rows) != {candidate_key}:
        schema_blockers.append("candidate_key_mismatch_requires_new_freeze")
    if selected_rows and _market_keys(selected_rows) != {market_candidate_key}:
        schema_blockers.append("market_candidate_key_mismatch")

    market_metrics, _, market_skipped = _evaluate_candidate(
        selected_grouped,
        probability_field="market_probability",
        rank_field="market_rank",
    )
    candidate_metrics, _, candidate_skipped = _evaluate_candidate(
        selected_grouped,
        probability_field="candidate_probability",
        rank_field="candidate_rank",
    )
    race_count = int(candidate_metrics.get("race_count") or 0)
    deltas = _metric_deltas(market_metrics, candidate_metrics)
    movements = _winner_movements(selected_grouped)
    concentration = _concentration(selected_grouped)

    if schema_blockers:
        decision = {
            "final_status": FINAL_DATA_MISSING,
            "gate_aligned": False,
            "hard_gates_pass": False,
            "materiality_gates_pass": False,
            "concentration_guard_pass": False,
            "promotion_review_floor_met": False,
            "blockers": schema_blockers,
        }
    else:
        decision = _gate_decision(
            race_count=race_count,
            deltas=deltas,
            movements=movements,
            concentration=concentration,
            min_oos_races=min_oos_races,
            stability_races=stability_races,
            promotion_review_races=promotion_review_races,
            min_race_dates_for_stability=min_race_dates_for_stability,
            min_venues_for_stability=min_venues_for_stability,
            max_single_date_share=max_single_date_share,
        )

    return {
        "final_status": decision["final_status"],
        "selected_race_count": len(selected_grouped),
        "evaluable_race_count": race_count,
        "market": market_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_market": deltas,
        "winner_movements": movements,
        "concentration": concentration,
        "skipped_counts": {
            "freeze_race_id_excluded": len(freeze_race_ids & set(oos_grouped_all)),
            "market": dict(sorted(market_skipped.items())),
            "candidate": dict(sorted(candidate_skipped.items())),
        },
        "decision": decision,
    }


def _grid_final_status(rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(row.get("final_status")) for row in rows}
    if FINAL_READY_FOR_OWNER in statuses:
        return FINAL_READY_FOR_OWNER
    if FINAL_SEGMENT_DESIGN_ONLY in statuses:
        return FINAL_SEGMENT_DESIGN_ONLY
    if FINAL_BLOCKED in statuses:
        return FINAL_BLOCKED
    return FINAL_DATA_MISSING


def _grid_summary_markdown(report: Mapping[str, Any]) -> str:
    best = report.get("best_eligible_by_logloss") or {}
    return "\n".join(
        [
            "# Frozen OOS Grid Evaluation",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Packet count: `{report.get('packet_count')}`",
            f"- OOS packet: `{report.get('oos_packet')}`",
            f"- Evaluated gates: `{report.get('evaluated_gate_count')}`",
            f"- Eligible gates: `{report.get('eligible_gate_count')}`",
            f"- Status counts: `{report.get('status_counts')}`",
            f"- Eligible status counts: `{report.get('eligible_status_counts')}`",
            f"- Best eligible by logloss: `{best}`",
            "",
            "No runtime, DB, capture, training, promotion, registry, dependency, TGR, EV/betting, or production pointer mutation was performed.",
            "",
        ]
    )


def build_grid_report(
    *,
    packet_root: Path,
    packet_name_regex: str,
    candidate_key: str,
    market_candidate_key: str = "market_only_implied",
    oos_packet_name: str | None = None,
    selector_fields: Sequence[str] = DEFAULT_GRID_SELECTOR_FIELDS,
    output_dir: Path | None = None,
    min_oos_races: int = 30,
    stability_races: int = 60,
    promotion_review_races: int = 100,
    min_race_dates_for_stability: int = 3,
    min_venues_for_stability: int = 3,
    max_single_date_share: float = 0.5,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    selector_fields = tuple(selector_fields)
    for field in selector_fields:
        validate_selector(field, "eq")

    packets = discover_grid_packets(packet_root, packet_name_regex)
    oos_packet = _packet_by_name(packets, oos_packet_name)
    oos_rows = _load_csv(Path(oos_packet["matrix_path"]))
    oos_grouped_all = _group_by_race(oos_rows)
    values_by_field = _grid_values_by_field(oos_grouped_all, selector_fields)

    grid_rows: list[dict[str, Any]] = []
    input_paths: list[Path] = [Path(oos_packet["matrix_path"]), Path(oos_packet["report_path"])]
    for packet in packets:
        if packet["name"] == oos_packet["name"]:
            continue
        freeze_rows = _load_csv(Path(packet["matrix_path"]))
        freeze_grouped = _group_by_race(freeze_rows)
        input_paths.extend([Path(packet["matrix_path"]), Path(packet["report_path"])])
        for field, values in values_by_field.items():
            for value in values:
                gate = _evaluate_grouped_gate(
                    freeze_grouped=freeze_grouped,
                    oos_grouped_all=oos_grouped_all,
                    selector_field=field,
                    selector_value=value,
                    candidate_key=candidate_key,
                    market_candidate_key=market_candidate_key,
                    min_oos_races=min_oos_races,
                    stability_races=stability_races,
                    promotion_review_races=promotion_review_races,
                    min_race_dates_for_stability=min_race_dates_for_stability,
                    min_venues_for_stability=min_venues_for_stability,
                    max_single_date_share=max_single_date_share,
                )
                deltas = gate["candidate_minus_market"]
                movements = gate["winner_movements"]
                decision = gate["decision"]
                grid_rows.append(
                    {
                        "freeze_packet": packet["name"],
                        "freeze_generated_at": packet.get("generated_at"),
                        "freeze_race_count": len(freeze_grouped),
                        "oos_packet": oos_packet["name"],
                        "selector_field": field,
                        "selector_value": value,
                        "final_status": gate["final_status"],
                        "selected_race_count": gate["selected_race_count"],
                        "evaluable_race_count": gate["evaluable_race_count"],
                        "top1_delta": deltas.get("top1"),
                        "top3_delta": deltas.get("top3"),
                        "mean_winner_rank_delta": deltas.get("mean_winner_rank"),
                        "brier_delta": deltas.get("brier"),
                        "logloss_delta": deltas.get("logloss"),
                        "candidate_promoted_winner_count": movements.get(
                            "candidate_promoted_winner_count"
                        ),
                        "candidate_demoted_winner_count": movements.get(
                            "candidate_demoted_winner_count"
                        ),
                        "blockers": decision.get("blockers"),
                    }
                )

    eligible_rows = [
        row for row in grid_rows if int(row.get("evaluable_race_count") or 0) >= min_oos_races
    ]
    status_counts = Counter(str(row.get("final_status")) for row in grid_rows)
    eligible_status_counts = Counter(str(row.get("final_status")) for row in eligible_rows)
    best_eligible_by_logloss = None
    if eligible_rows:
        best_eligible_by_logloss = min(
            eligible_rows,
            key=lambda row: (
                _safe_float(row.get("logloss_delta"))
                if _safe_float(row.get("logloss_delta")) is not None
                else 999.0,
                _safe_float(row.get("brier_delta"))
                if _safe_float(row.get("brier_delta")) is not None
                else 999.0,
            ),
        )

    report = {
        "schema_version": "frozen_oos_grid_evaluator_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": _grid_final_status(grid_rows),
        "packet_root": str(packet_root),
        "packet_name_regex": packet_name_regex,
        "packet_count": len(packets),
        "oos_packet": oos_packet["name"],
        "oos_report_generated_at": oos_packet.get("generated_at"),
        "oos_input_race_count": len(oos_grouped_all),
        "candidate_key": candidate_key,
        "market_candidate_key": market_candidate_key,
        "selector_fields": list(selector_fields),
        "selector_values_by_field": values_by_field,
        "sample_floors": {
            "min_oos_races": min_oos_races,
            "stability_races": stability_races,
            "promotion_review_races": promotion_review_races,
            "min_race_dates_for_stability": min_race_dates_for_stability,
            "min_venues_for_stability": min_venues_for_stability,
            "max_single_date_share": max_single_date_share,
        },
        "evaluated_gate_count": len(grid_rows),
        "eligible_gate_count": len(eligible_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "eligible_status_counts": dict(sorted(eligible_status_counts.items())),
        "best_eligible_by_logloss": best_eligible_by_logloss,
        "gate_results": grid_rows,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }

    if output_dir is not None:
        resolved_output = _assert_output_dir_safe(output_dir)
        _assert_not_input_packet_output(resolved_output, input_paths)
        resolved_output.mkdir(parents=True, exist_ok=True)
        _write_json(resolved_output / GRID_REPORT_FILE, report)
        _write_grid_csv(resolved_output / GRID_RESULTS_CSV, grid_rows)
        _write_text(resolved_output / GRID_SUMMARY_FILE, _grid_summary_markdown(report))
        _write_text(resolved_output / "grid_final_status.txt", str(report["final_status"]) + "\n")
        _write_json(resolved_output / "output_manifest.json", _output_manifest(resolved_output))

    return report


def summary_markdown(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    deltas = metrics.get("candidate_minus_market") or {}
    decision = report.get("decision") or {}
    return "\n".join(
        [
            "# Frozen OOS Gate Evaluation",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Gate id: `{report.get('gate_id')}`",
            f"- Selector: `{report.get('selector')}`",
            f"- Freeze race IDs: `{report.get('freeze_race_count')}`",
            f"- OOS selected races: `{metrics.get('race_count')}`",
            f"- Candidate minus market: `{deltas}`",
            f"- Winner movement: `{metrics.get('winner_movements')}`",
            f"- Decision blockers: `{decision.get('blockers')}`",
            "",
            "No runtime, DB, capture, training, promotion, registry, dependency, TGR, EV/betting, or production pointer mutation was performed.",
            "",
        ]
    )


def build_report(
    *,
    freeze_runner_matrix_csv: Path,
    oos_runner_matrix_csv: Path,
    gate_id: str,
    selector_field: str,
    selector_operator: str,
    selector_value: Any,
    candidate_key: str,
    market_candidate_key: str = "market_only_implied",
    freeze_report_json: Path | None = None,
    oos_report_json: Path | None = None,
    output_dir: Path | None = None,
    min_oos_races: int = 30,
    stability_races: int = 60,
    promotion_review_races: int = 100,
    min_race_dates_for_stability: int = 3,
    min_venues_for_stability: int = 3,
    max_single_date_share: float = 0.5,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    validate_selector(selector_field, selector_operator)

    freeze_rows = _load_csv(freeze_runner_matrix_csv)
    oos_rows = _load_csv(oos_runner_matrix_csv)
    freeze_report = _load_json(freeze_report_json)
    oos_report = _load_json(oos_report_json)

    freeze_grouped = _group_by_race(freeze_rows)
    oos_grouped_all = _group_by_race(oos_rows)
    freeze_race_ids = set(freeze_grouped)
    selected_grouped: dict[str, Sequence[Mapping[str, Any]]] = {}
    skipped = Counter()
    for race_id, rows in oos_grouped_all.items():
        if race_id in freeze_race_ids:
            skipped["freeze_race_id_excluded"] += 1
            continue
        if _value_matches(_race_value(rows, selector_field), selector_operator, selector_value):
            selected_grouped[race_id] = rows

    selected_rows = [row for rows in selected_grouped.values() for row in rows]
    candidate_keys = _candidate_keys(selected_rows)
    market_keys = _market_keys(selected_rows)
    schema_blockers: list[str] = []
    if selected_rows and candidate_keys != {candidate_key}:
        schema_blockers.append("candidate_key_mismatch_requires_new_freeze")
    if selected_rows and market_keys != {market_candidate_key}:
        schema_blockers.append("market_candidate_key_mismatch")

    market_metrics, market_race_rows, market_skipped = _evaluate_candidate(
        selected_grouped,
        probability_field="market_probability",
        rank_field="market_rank",
    )
    candidate_metrics, candidate_race_rows, candidate_skipped = _evaluate_candidate(
        selected_grouped,
        probability_field="candidate_probability",
        rank_field="candidate_rank",
    )
    race_count = int(candidate_metrics.get("race_count") or 0)
    deltas = _metric_deltas(market_metrics, candidate_metrics)
    movements = _winner_movements(selected_grouped)
    concentration = _concentration(selected_grouped)

    if schema_blockers:
        decision = {
            "final_status": FINAL_DATA_MISSING,
            "gate_aligned": False,
            "hard_gates_pass": False,
            "materiality_gates_pass": False,
            "concentration_guard_pass": False,
            "promotion_review_floor_met": False,
            "blockers": schema_blockers,
        }
    else:
        decision = _gate_decision(
            race_count=race_count,
            deltas=deltas,
            movements=movements,
            concentration=concentration,
            min_oos_races=min_oos_races,
            stability_races=stability_races,
            promotion_review_races=promotion_review_races,
            min_race_dates_for_stability=min_race_dates_for_stability,
            min_venues_for_stability=min_venues_for_stability,
            max_single_date_share=max_single_date_share,
        )

    combined_rows = _combined_race_rows(market_race_rows, candidate_race_rows, selected_grouped)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "final_status": decision["final_status"],
        "gate_id": gate_id,
        "selector": {
            "field": selector_field,
            "operator": selector_operator,
            "value": selector_value,
            "source_safe": True,
        },
        "candidate_key": candidate_key,
        "market_candidate_key": market_candidate_key,
        "freeze_runner_matrix_csv": str(freeze_runner_matrix_csv),
        "oos_runner_matrix_csv": str(oos_runner_matrix_csv),
        "freeze_report_json": str(freeze_report_json) if freeze_report_json else None,
        "oos_report_json": str(oos_report_json) if oos_report_json else None,
        "freeze_report_generated_at": freeze_report.get("generated_at"),
        "oos_report_generated_at": oos_report.get("generated_at"),
        "freeze_race_count": len(freeze_race_ids),
        "oos_input_race_count": len(oos_grouped_all),
        "oos_selected_race_count": len(selected_grouped),
        "sample_floors": {
            "min_oos_races": min_oos_races,
            "stability_races": stability_races,
            "promotion_review_races": promotion_review_races,
            "min_race_dates_for_stability": min_race_dates_for_stability,
            "min_venues_for_stability": min_venues_for_stability,
            "max_single_date_share": max_single_date_share,
        },
        "metrics": {
            "race_count": race_count,
            "market": market_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_market": deltas,
            "winner_movements": movements,
            "concentration": concentration,
            "skipped_counts": {
                **dict(sorted(skipped.items())),
                "market": dict(sorted(market_skipped.items())),
                "candidate": dict(sorted(candidate_skipped.items())),
            },
        },
        "decision": decision,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }

    if output_dir is not None:
        resolved_output = _assert_output_dir_safe(output_dir)
        _assert_not_input_packet_output(
            resolved_output,
            [freeze_runner_matrix_csv, oos_runner_matrix_csv],
        )
        resolved_output.mkdir(parents=True, exist_ok=True)
        _write_json(resolved_output / REPORT_FILE, report)
        _write_csv(resolved_output / RACE_METRICS_CSV, combined_rows)
        _write_text(resolved_output / SUMMARY_FILE, summary_markdown(report))
        _write_text(resolved_output / "final_status.txt", str(report["final_status"]) + "\n")
        _write_json(resolved_output / "output_manifest.json", _output_manifest(resolved_output))

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-runner-matrix-csv", type=Path)
    parser.add_argument("--oos-runner-matrix-csv", type=Path)
    parser.add_argument("--gate-id")
    parser.add_argument("--selector-field")
    parser.add_argument("--selector-operator", choices=sorted(SELECTOR_OPERATORS), default="eq")
    parser.add_argument("--selector-value")
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--market-candidate-key", default="market_only_implied")
    parser.add_argument("--freeze-report-json", type=Path)
    parser.add_argument("--oos-report-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--grid-root", type=Path)
    parser.add_argument(
        "--grid-packet-name-regex",
        default=r"^rolling_model_comparison_\d{8}T\d{6}\+1000_daemon_autopilot$",
    )
    parser.add_argument("--grid-oos-packet-name")
    parser.add_argument(
        "--grid-selector-fields",
        help="Comma-separated source-safe selector fields. Defaults to low-cardinality pre-race fields.",
    )
    parser.add_argument("--grid-output-dir", type=Path)
    parser.add_argument("--min-oos-races", type=int, default=30)
    parser.add_argument("--stability-races", type=int, default=60)
    parser.add_argument("--promotion-review-races", type=int, default=100)
    parser.add_argument("--min-race-dates-for-stability", type=int, default=3)
    parser.add_argument("--min-venues-for-stability", type=int, default=3)
    parser.add_argument("--max-single-date-share", type=float, default=0.5)
    return parser.parse_args(argv)


def _require_single_gate_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "freeze_runner_matrix_csv",
            "oos_runner_matrix_csv",
            "gate_id",
            "selector_field",
            "selector_value",
        )
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise ValueError(f"missing_single_gate_args:{','.join(missing)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.grid_root is not None:
        report = build_grid_report(
            packet_root=args.grid_root,
            packet_name_regex=args.grid_packet_name_regex,
            oos_packet_name=args.grid_oos_packet_name,
            selector_fields=_selector_fields_from_csv(args.grid_selector_fields),
            candidate_key=args.candidate_key,
            market_candidate_key=args.market_candidate_key,
            output_dir=args.grid_output_dir,
            min_oos_races=args.min_oos_races,
            stability_races=args.stability_races,
            promotion_review_races=args.promotion_review_races,
            min_race_dates_for_stability=args.min_race_dates_for_stability,
            min_venues_for_stability=args.min_venues_for_stability,
            max_single_date_share=args.max_single_date_share,
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0

    _require_single_gate_args(args)
    report = build_report(
        freeze_runner_matrix_csv=args.freeze_runner_matrix_csv,
        oos_runner_matrix_csv=args.oos_runner_matrix_csv,
        freeze_report_json=args.freeze_report_json,
        oos_report_json=args.oos_report_json,
        gate_id=args.gate_id,
        selector_field=args.selector_field,
        selector_operator=args.selector_operator,
        selector_value=_parse_selector_value(args.selector_value),
        candidate_key=args.candidate_key,
        market_candidate_key=args.market_candidate_key,
        output_dir=args.output_dir,
        min_oos_races=args.min_oos_races,
        stability_races=args.stability_races,
        promotion_review_races=args.promotion_review_races,
        min_race_dates_for_stability=args.min_race_dates_for_stability,
        min_venues_for_stability=args.min_venues_for_stability,
        max_single_date_share=args.max_single_date_share,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
