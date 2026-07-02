#!/usr/bin/env python3
"""Adapt a cumulative rolling runner matrix into a report-only challenger packet.

This helper reads existing rolling-model artifacts only. It does not fetch
results, capture odds, write databases, fit models, promote models, mutate
registries, or place bets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "cumulative_runner_matrix_challenger_packet_v1"

NO_WRITE_GUARANTEES = {
    "live_db_write": False,
    "official_result_capture": False,
    "live_odds_capture": False,
    "model_fit": False,
    "model_artifact_write": False,
    "registry_mutation": False,
    "promotion": False,
    "betting": False,
}

PROTECTED_OUTPUT_PREFIXES = (
    "artifacts/prediction_snapshots",
    "model_registry",
    "docs/model_registry",
    "ml_models_v4",
    "advanced_models",
)

REQUIRED_RUNNER_FIELDS = (
    "race_id",
    "source_report",
    "dog_name",
    "box_number",
    "finish_position",
    "odds_decimal",
    "market_probability",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _truthy_number(value: Any) -> bool:
    parsed = _safe_float(value)
    return parsed is not None


def _resolve_existing_path(raw_path: str, *, path_base: Path, report_path: Path) -> Path:
    candidates: list[Path] = []
    candidate = Path(raw_path)
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                (path_base / candidate).resolve(),
                (report_path.parent / candidate).resolve(),
                Path.cwd() / candidate,
            ]
        )

    for item in candidates:
        if item.exists():
            return item
    return candidates[0] if candidates else candidate


def _assert_output_dir_safe(output_dir: Path, repo_root: Path | None = None) -> Path:
    resolved = output_dir.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return resolved

    relative_text = relative.as_posix()
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        if relative_text == prefix or relative_text.startswith(prefix + "/"):
            raise ValueError(f"output_dir_protected:{prefix}")
    return resolved


def _status_from_counts(ready_count: int, total_count: int) -> str:
    if total_count <= 0:
        return "DATA_MISSING"
    if ready_count == total_count:
        return "READY"
    if ready_count > 0:
        return "PARTIAL"
    return "DATA_MISSING"


def _race_rows(rows: list[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        race_id = str(row.get("race_id") or "")
        if race_id:
            grouped[race_id].append(row)
    return dict(grouped)


def _count_complete_races(
    grouped: Mapping[str, list[Mapping[str, Any]]],
    field: str,
) -> int:
    return sum(
        1
        for race_rows in grouped.values()
        if race_rows and all(str(row.get(field) or "") != "" for row in race_rows)
    )


def _count_numeric_complete_races(
    grouped: Mapping[str, list[Mapping[str, Any]]],
    field: str,
) -> int:
    return sum(
        1
        for race_rows in grouped.values()
        if race_rows and all(_truthy_number(row.get(field)) for row in race_rows)
    )


def _probability_row_count(rows: list[Mapping[str, Any]], field: str) -> int:
    return sum(1 for row in rows if _truthy_number(row.get(field)))


def _race_table(grouped: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for race_id in sorted(grouped):
        rows = grouped[race_id]
        first = rows[0]
        output.append(
            {
                "race_id": race_id,
                "race_date": first.get("race_date"),
                "venue": first.get("venue"),
                "race_number": first.get("race_number"),
                "source_report": first.get("source_report"),
                "runner_count": len(rows),
                "complete_valid_odds": all(
                    _truthy_number(row.get("odds_decimal")) for row in rows
                ),
                "official_result_joined": all(
                    str(row.get("finish_position") or "") != "" for row in rows
                ),
                "market_probability_rows": _probability_row_count(
                    rows, "market_probability"
                ),
                "primary_probability_rows": _probability_row_count(
                    rows, "primary_shadow_probability_norm"
                ),
                "stage2_probability_rows": _probability_row_count(
                    rows, "stage2_shadow_probability_norm"
                ),
                "stage2_uncalibrated_probability_rows": _probability_row_count(
                    rows, "stage2_shadow_uncalibrated_probability_norm"
                ),
            }
        )
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_packet(
    *,
    rolling_report_path: Path,
    output_dir: Path,
    path_base: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    path_base = path_base or Path.cwd()
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rolling_report = _load_json(rolling_report_path)
    runner_matrix_raw = rolling_report.get("market_residual_runner_matrix_csv")
    if not runner_matrix_raw:
        raise ValueError("rolling_report_missing_market_residual_runner_matrix_csv")
    runner_matrix_path = _resolve_existing_path(
        str(runner_matrix_raw),
        path_base=path_base,
        report_path=rolling_report_path,
    )
    if not runner_matrix_path.exists():
        raise FileNotFoundError(f"runner_matrix_missing:{runner_matrix_path}")

    runner_rows = _load_csv(runner_matrix_path)
    grouped = _race_rows(runner_rows)
    race_rows = _race_table(grouped)

    missing_required_fields = sorted(
        {
            field
            for field in REQUIRED_RUNNER_FIELDS
            if any(str(row.get(field) or "") == "" for row in runner_rows)
        }
    )
    runner_race_count = len(grouped)
    rolling_sample_race_count = _safe_int(rolling_report.get("sample_race_count"))
    rolling_sample_runner_rows = _safe_int(rolling_report.get("sample_runner_rows"))
    complete_valid_odds_races = _count_numeric_complete_races(grouped, "odds_decimal")
    official_result_joined_races = _count_complete_races(grouped, "finish_position")
    source_reports = sorted(
        {str(row.get("source_report")) for row in runner_rows if row.get("source_report")}
    )

    row_count_matches = len(runner_rows) == rolling_sample_runner_rows
    race_count_matches = runner_race_count == rolling_sample_race_count
    complete_market_comparable = (
        runner_race_count > 0
        and complete_valid_odds_races == runner_race_count
        and official_result_joined_races == runner_race_count
    )
    status = "READY_FOR_REPORT_ONLY_CHALLENGER"
    blockers: list[str] = []
    if missing_required_fields:
        blockers.append("runner_matrix_required_fields_missing")
    if not row_count_matches:
        blockers.append("runner_matrix_row_count_mismatch")
    if not race_count_matches:
        blockers.append("runner_matrix_race_count_mismatch")
    if not complete_market_comparable:
        blockers.append("runner_matrix_not_complete_market_comparable")
    if blockers:
        status = "DATA_MISSING"

    race_table_csv = output_dir / "current_cumulative_race_table.csv"
    race_table_jsonl = output_dir / "current_cumulative_race_table.jsonl"
    _write_csv(race_table_csv, race_rows)
    _write_jsonl(race_table_jsonl, race_rows)

    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(),
        "status": status,
        "blockers": blockers,
        "input_surface": "current_cumulative_rolling_runner_matrix",
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "paths": {
            "rolling_report": str(rolling_report_path),
            "runner_matrix": str(runner_matrix_path),
            "race_table_csv": str(race_table_csv),
            "race_table_jsonl": str(race_table_jsonl),
        },
        "counts": {
            "rolling_sample_race_count": rolling_sample_race_count,
            "runner_matrix_race_count": runner_race_count,
            "rolling_sample_runner_rows": rolling_sample_runner_rows,
            "runner_matrix_rows": len(runner_rows),
            "complete_valid_odds_races": complete_valid_odds_races,
            "official_result_joined_races": official_result_joined_races,
            "source_report_count": len(source_reports),
            "source_unified_evidence_report_count": len(
                rolling_report.get("source_unified_evidence_reports") or []
            ),
            "market_probability_rows": _probability_row_count(
                runner_rows, "market_probability"
            ),
            "primary_probability_rows": _probability_row_count(
                runner_rows, "primary_shadow_probability_norm"
            ),
            "stage2_probability_rows": _probability_row_count(
                runner_rows, "stage2_shadow_probability_norm"
            ),
            "stage2_uncalibrated_probability_rows": _probability_row_count(
                runner_rows, "stage2_shadow_uncalibrated_probability_norm"
            ),
        },
        "readiness": {
            "race_count_match": race_count_matches,
            "runner_row_count_match": row_count_matches,
            "complete_market_comparable_status": _status_from_counts(
                min(complete_valid_odds_races, official_result_joined_races),
                runner_race_count,
            ),
            "missing_required_runner_fields": missing_required_fields,
        },
        "rolling_context": {
            "rolling_final_status": rolling_report.get("final_status"),
            "sample_floor_met": rolling_report.get("sample_floor_met"),
            "minimum_races_for_review": rolling_report.get("minimum_races_for_review"),
            "best_non_market_candidate_key": rolling_report.get(
                "best_non_market_candidate_key"
            ),
            "best_non_market_minus_market": rolling_report.get(
                "best_non_market_minus_market"
            ),
            "market_candidate_key": rolling_report.get("market_candidate_key"),
            "candidate_count": rolling_report.get("candidate_count"),
        },
        "runner_row_date_counts": dict(
            sorted(
                Counter(
                    row.get("race_date") or "DATA_MISSING" for row in runner_rows
                ).items()
            )
        ),
        "race_date_counts": dict(
            sorted(
                Counter(
                    row.get("race_date") or "DATA_MISSING" for row in race_rows
                ).items()
            )
        ),
        "challenger_adapter_note": (
            "This packet proves the current cumulative rolling sample can be consumed "
            "directly by report-only challenger logic. It intentionally does not reuse "
            "older recovered clean-official/history-feature inputs."
        ),
    }
    packet_path = output_dir / "CUMULATIVE_RUNNER_MATRIX_CHALLENGER_PACKET.json"
    _write_json(packet_path, packet)
    return packet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--path-base",
        type=Path,
        default=Path.cwd(),
        help="Base used to resolve relative runner-matrix paths.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    packet = build_packet(
        rolling_report_path=args.rolling_report,
        output_dir=args.output_dir,
        path_base=args.path_base,
    )
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
