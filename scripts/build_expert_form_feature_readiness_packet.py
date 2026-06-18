#!/usr/bin/env python3
"""Build a report-only Expert Form feature readiness packet.

This packet answers whether source-safe TheDogs Expert Form metadata is now
available enough to justify a later report-only schema/training trial. It does
not train, mutate schemas, write DB rows, update registries, emit EV, or bet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.run_feature_recovery_execution_v1 import clean_name  # noqa: E402
from utils.expert_form_metadata import safe_expert_form_metadata_from_payload  # noqa: E402


OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "expert_form_feature_readiness_"
)
DEFAULT_SCHEMA = ROOT / "accuracy_program/repaired_non_tgr_schema.json"
DEFAULT_ROOTS = (
    ROOT / "artifacts/full_evidence_orchestration_20260525",
    ROOT / "artifacts/shadow_evaluation",
)
FINAL_SOURCE_LOW = "KEEP_COLLECTING_ONLY_EXPERT_FORM_SOURCE_COVERAGE_LOW"
FINAL_FEATURE_ROWS_LOW = "READY_FOR_SHADOW_FEATURE_ROW_BACKFILL_REPORT_ONLY"
FINAL_SCHEMA_TRIAL = "EXPERT_FORM_SCHEMA_TRIAL_READY_REPORT_ONLY_NO_ACTIVATION"
FINAL_ABLATION_NEEDED = "EXPERT_FORM_ABLATION_NEEDED_REPORT_ONLY"

NO_WRITE_GUARANTEES = {
    "report_only": True,
    "training_run": False,
    "schema_mutation": False,
    "registry_mutation": False,
    "production_prediction_write": False,
    "db_write": False,
    "label_write": False,
    "ev_output": False,
    "betting_output": False,
}

EXPERT_FORM_FEATURES = (
    "expert_form_career_starts",
    "expert_form_career_wins",
    "expert_form_career_seconds",
    "expert_form_career_thirds",
    "expert_form_track_distance_starts",
    "expert_form_track_distance_wins",
    "expert_form_track_distance_seconds",
    "expert_form_track_distance_thirds",
    "expert_form_win_percent",
    "expert_form_place_percent",
    "expert_form_prize_money",
    "expert_form_track_distance_best_time",
    "expert_form_track_distance_best_first_split",
    "expert_form_best_other_track_count",
    "expert_form_best_other_track_min_time",
    "expert_form_distance_wins_under_400",
    "expert_form_distance_wins_400_plus",
    "expert_form_distance_wins_500_plus",
    "expert_form_distance_wins_600_plus",
    "expert_form_distance_wins_700_plus",
    "expert_form_current_box_starts",
    "expert_form_current_box_wins",
    "expert_form_current_box_places",
    "expert_form_grade",
    "expert_form_sex",
    "expert_form_sire",
    "expert_form_dam",
    "expert_form_trainer_name",
    "expert_form_trainer_district",
)


def relpath(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def now_id(value: datetime | None = None) -> str:
    return (value or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_expert_form_readiness_artifact:{relative}")
    return logical


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def is_present(value: Any) -> bool:
    return value not in (None, "", [], {})


def stable_race_id(payload: Mapping[str, Any], sidecar_path: Path) -> str:
    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    race_number = race_info.get("race_number")
    venue = race_info.get("venue") or race_info.get("venue_name")
    race_date = race_info.get("date")
    if race_number and venue and race_date:
        return f"Race {race_number} - {venue} - {str(race_date)[:10]}"
    return sidecar_path.name.replace(".csv.metadata.json", "")


def flatten_runner_metadata(runner: Mapping[str, Any]) -> dict[str, Any]:
    career = runner.get("career") if isinstance(runner.get("career"), Mapping) else {}
    td = runner.get("track_distance") if isinstance(runner.get("track_distance"), Mapping) else {}
    greyhound = runner.get("greyhound") if isinstance(runner.get("greyhound"), Mapping) else {}
    trainer = runner.get("trainer") if isinstance(runner.get("trainer"), Mapping) else {}
    best_other = runner.get("best_win_times_other_tracks")
    if not isinstance(best_other, list):
        best_other = []
    best_other_times = [
        item.get("time")
        for item in best_other
        if isinstance(item, Mapping) and item.get("time") not in (None, "")
    ]
    distance_counts = (
        runner.get("winning_distance_counts")
        if isinstance(runner.get("winning_distance_counts"), Mapping)
        else {}
    )
    box_history = runner.get("box_history") if isinstance(runner.get("box_history"), Mapping) else {}
    return {
        "dog_name": runner.get("dog_name"),
        "dog_key": clean_name(runner.get("dog_name")),
        "expert_form_grade": runner.get("grade"),
        "expert_form_trainer_name": trainer.get("name"),
        "expert_form_trainer_district": trainer.get("district"),
        "expert_form_owner": runner.get("owner"),
        "expert_form_colour": greyhound.get("colour"),
        "expert_form_sex": greyhound.get("sex"),
        "expert_form_sire": greyhound.get("sire"),
        "expert_form_dam": greyhound.get("dam"),
        "expert_form_career_starts": career.get("starts"),
        "expert_form_career_wins": career.get("wins"),
        "expert_form_career_seconds": career.get("seconds"),
        "expert_form_career_thirds": career.get("thirds"),
        "expert_form_track_distance_starts": td.get("starts"),
        "expert_form_track_distance_wins": td.get("wins"),
        "expert_form_track_distance_seconds": td.get("seconds"),
        "expert_form_track_distance_thirds": td.get("thirds"),
        "expert_form_win_percent": runner.get("win_percent"),
        "expert_form_place_percent": runner.get("place_percent"),
        "expert_form_prize_money": runner.get("prize_money"),
        "expert_form_track_distance_best_time": td.get("best_time"),
        "expert_form_track_distance_best_first_split": td.get("best_first_split"),
        "expert_form_best_other_track_count": len(best_other),
        "expert_form_best_other_track_min_time": min(best_other_times)
        if best_other_times
        else None,
        "expert_form_distance_wins_under_400": distance_counts.get("<400"),
        "expert_form_distance_wins_400_plus": distance_counts.get("400+"),
        "expert_form_distance_wins_500_plus": distance_counts.get("500+"),
        "expert_form_distance_wins_600_plus": distance_counts.get("600+"),
        "expert_form_distance_wins_700_plus": distance_counts.get("700+"),
        "expert_form_box_history_available": bool(box_history),
    }


def sidecar_runner_records(sidecar_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], {
            "sidecar_path": relpath(sidecar_path),
            "status": "unreadable",
            "rejected_reasons": [f"sidecar_unreadable:{type(exc).__name__}"],
            "runner_count": 0,
        }
    if not isinstance(payload, Mapping):
        return [], {
            "sidecar_path": relpath(sidecar_path),
            "status": "not_object",
            "rejected_reasons": ["sidecar_not_object"],
            "runner_count": 0,
        }
    metadata = safe_expert_form_metadata_from_payload(payload)
    race_id = stable_race_id(payload, sidecar_path)
    source_row = {
        "race_id": race_id,
        "sidecar_path": relpath(sidecar_path),
        "source_url": metadata.get("source_url"),
        "captured_at": metadata.get("captured_at"),
        "metadata_is_leakage_safe": metadata.get("metadata_is_leakage_safe"),
        "runner_count": metadata.get("runner_count") or 0,
        "rejected_reasons": ";".join(metadata.get("rejected_reasons") or []),
    }
    if metadata.get("metadata_is_leakage_safe") is not True:
        return [], source_row
    rows = []
    for runner in metadata.get("runners") or []:
        if not isinstance(runner, Mapping):
            continue
        rows.append(
            {
                **source_row,
                **flatten_runner_metadata(runner),
            }
        )
    return rows, source_row


def scan_sidecars(roots: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runner_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for sidecar_path in sorted(root.rglob("*.metadata.json")):
            resolved = sidecar_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows, source = sidecar_runner_records(sidecar_path)
            source_rows.append(source)
            runner_rows.extend(rows)
    return runner_rows, source_rows


def load_shadow_feature_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def scan_shadow_feature_rows(roots: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("shadow_feature_rows.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            for row in load_shadow_feature_rows(path):
                rows.append({"feature_rows_path": relpath(path), **row})
    return rows


def feature_coverage(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    *,
    source: str,
) -> list[dict[str, Any]]:
    total = len(rows)
    out = []
    for feature in features:
        present_values = [row.get(feature) for row in rows if is_present(row.get(feature))]
        unique_values = {str(value) for value in present_values}
        most_common = Counter(str(value) for value in present_values).most_common(1)
        out.append(
            {
                "source": source,
                "feature": feature,
                "rows": total,
                "present_rows": len(present_values),
                "present_pct": len(present_values) / total if total else 0.0,
                "unique_present_values": len(unique_values),
                "most_common_value": most_common[0][0] if most_common else None,
                "most_common_count": most_common[0][1] if most_common else 0,
                "most_common_share": most_common[0][1] / len(present_values)
                if present_values and most_common
                else 0.0,
            }
        )
    return out


def schema_gap_rows(schema_path: Path, features: Sequence[str]) -> list[dict[str, Any]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_features = set(schema.get("feature_columns") or [])
    categorical = set(schema.get("categorical_features") or [])
    numeric = set(schema.get("numeric_or_boolean_features") or [])
    rows = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "in_schema_feature_columns": feature in schema_features,
                "in_categorical_features": feature in categorical,
                "in_numeric_or_boolean_features": feature in numeric,
                "schema_action": "already_present" if feature in schema_features else "schema_trial_required",
            }
        )
    return rows


def decide(
    *,
    safe_sidecar_races: int,
    safe_sidecar_runner_rows: int,
    safe_shadow_feature_rows: int,
    schema_gap: Sequence[Mapping[str, Any]],
    min_source_races: int,
    min_source_runner_rows: int,
    min_shadow_feature_rows: int,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if safe_sidecar_races < min_source_races:
        blockers.append("safe_expert_form_sidecar_races_below_min")
    if safe_sidecar_runner_rows < min_source_runner_rows:
        blockers.append("safe_expert_form_runner_rows_below_min")
    if blockers:
        return FINAL_SOURCE_LOW, blockers
    if safe_shadow_feature_rows < min_shadow_feature_rows:
        return FINAL_FEATURE_ROWS_LOW, ["safe_shadow_feature_rows_below_min"]
    if any(row.get("in_schema_feature_columns") is not True for row in schema_gap):
        return FINAL_SCHEMA_TRIAL, ["expert_form_features_not_in_canonical_schema"]
    return FINAL_ABLATION_NEEDED, ["report_only_ablation_metrics_missing"]


def build_report(
    *,
    artifact_roots: Sequence[Path],
    schema_path: Path,
    min_source_races: int,
    min_source_runner_rows: int,
    min_shadow_feature_rows: int,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    sidecar_rows, source_rows = scan_sidecars(artifact_roots)
    feature_rows = scan_shadow_feature_rows(artifact_roots)
    safe_source_rows = [
        row for row in source_rows if row.get("metadata_is_leakage_safe") is True
    ]
    safe_feature_rows = [
        row for row in feature_rows if row.get("expert_form_metadata_from_sidecar") is True
    ]
    race_ids = {row.get("race_id") for row in sidecar_rows if row.get("race_id")}
    schema_gap = schema_gap_rows(schema_path, EXPERT_FORM_FEATURES)
    decision, blockers = decide(
        safe_sidecar_races=len(race_ids),
        safe_sidecar_runner_rows=len(sidecar_rows),
        safe_shadow_feature_rows=len(safe_feature_rows),
        schema_gap=schema_gap,
        min_source_races=min_source_races,
        min_source_runner_rows=min_source_runner_rows,
        min_shadow_feature_rows=min_shadow_feature_rows,
    )
    sidecar_coverage = feature_coverage(
        sidecar_rows,
        EXPERT_FORM_FEATURES,
        source="safe_expert_form_sidecars",
    )
    shadow_feature_coverage = feature_coverage(
        safe_feature_rows,
        EXPERT_FORM_FEATURES,
        source="shadow_feature_rows",
    )
    return {
        "schema_version": "expert_form_feature_readiness_packet_v1",
        "generated_at": generated_at.isoformat(),
        "artifact_roots": [relpath(path) for path in artifact_roots],
        "schema_path": relpath(schema_path),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "candidate_features": list(EXPERT_FORM_FEATURES),
        "thresholds": {
            "min_source_races": min_source_races,
            "min_source_runner_rows": min_source_runner_rows,
            "min_shadow_feature_rows": min_shadow_feature_rows,
        },
        "coverage_summary": {
            "source_sidecars_scanned": len(source_rows),
            "safe_source_sidecars": len(safe_source_rows),
            "safe_source_races": len(race_ids),
            "safe_source_runner_rows": len(sidecar_rows),
            "shadow_feature_rows_scanned": len(feature_rows),
            "safe_shadow_feature_rows": len(safe_feature_rows),
        },
        "schema_gap_count": sum(
            1 for row in schema_gap if row.get("in_schema_feature_columns") is not True
        ),
        "final_status": decision,
        "activation_allowed": False,
        "training_run": False,
        "blockers": blockers,
        "source_rows": source_rows,
        "sidecar_runner_rows": sidecar_rows,
        "sidecar_feature_coverage": sidecar_coverage,
        "shadow_feature_coverage": shadow_feature_coverage,
        "schema_gap": schema_gap,
    }


def summary_md(report: Mapping[str, Any], output_dir: Path) -> str:
    summary = report.get("coverage_summary") or {}
    blockers = report.get("blockers") or []
    return "\n".join(
        [
            "# Expert Form Feature Readiness Packet",
            "",
            f"Final status: `{report.get('final_status')}`",
            f"Activation allowed: `{report.get('activation_allowed')}`",
            "",
            "## Coverage",
            "",
            f"- Safe source races: `{summary.get('safe_source_races')}`",
            f"- Safe source runner rows: `{summary.get('safe_source_runner_rows')}`",
            f"- Safe shadow feature rows: `{summary.get('safe_shadow_feature_rows')}`",
            f"- Schema gap count: `{report.get('schema_gap_count')}`",
            "",
            "## Blockers",
            "",
            *(f"- `{blocker}`" for blocker in blockers),
            "",
            "## Artifacts",
            "",
            f"- `{relpath(output_dir / 'expert_form_source_coverage.csv')}`",
            f"- `{relpath(output_dir / 'expert_form_feature_coverage.csv')}`",
            f"- `{relpath(output_dir / 'expert_form_schema_gap.csv')}`",
            f"- `{relpath(output_dir / 'expert_form_feature_readiness_report.json')}`",
            "",
            "No training, schema mutation, registry mutation, DB write, label write, EV output, or betting output was performed.",
            "",
        ]
    )


def write_packet(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(
        output_dir / "expert_form_source_coverage.csv",
        report["sidecar_runner_rows"],
        [
            "race_id",
            "sidecar_path",
            "source_url",
            "captured_at",
            "dog_name",
            "expert_form_career_starts",
            "expert_form_career_wins",
            "expert_form_track_distance_starts",
            "expert_form_track_distance_wins",
            "expert_form_win_percent",
            "expert_form_place_percent",
            "expert_form_prize_money",
            "expert_form_track_distance_best_time",
            "expert_form_best_other_track_count",
            "expert_form_box_history_available",
        ],
    )
    write_csv(
        output_dir / "expert_form_feature_coverage.csv",
        [*report["sidecar_feature_coverage"], *report["shadow_feature_coverage"]],
        [
            "source",
            "feature",
            "rows",
            "present_rows",
            "present_pct",
            "unique_present_values",
            "most_common_value",
            "most_common_count",
            "most_common_share",
        ],
    )
    write_csv(
        output_dir / "expert_form_schema_gap.csv",
        report["schema_gap"],
        [
            "feature",
            "in_schema_feature_columns",
            "in_categorical_features",
            "in_numeric_or_boolean_features",
            "schema_action",
        ],
    )
    manifest = {
        "schema_version": "expert_form_feature_readiness_manifest_v1",
        "generated_at": report["generated_at"],
        "files": {
            "report": relpath(output_dir / "expert_form_feature_readiness_report.json"),
            "summary": relpath(output_dir / "SUMMARY.md"),
            "final_status": relpath(output_dir / "final_status.txt"),
            "source_coverage": relpath(output_dir / "expert_form_source_coverage.csv"),
            "feature_coverage": relpath(output_dir / "expert_form_feature_coverage.csv"),
            "schema_gap": relpath(output_dir / "expert_form_schema_gap.csv"),
        },
        "no_write_guarantees": report["no_write_guarantees"],
    }
    write_json(output_dir / "expert_form_feature_readiness_report.json", report)
    write_json(output_dir / "output_manifest.json", manifest)
    write_text(output_dir / "SUMMARY.md", summary_md(report, output_dir))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", action="append", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-source-races", type=int, default=20)
    parser.add_argument("--min-source-runner-rows", type=int, default=100)
    parser.add_argument("--min-shadow-feature-rows", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_roots = tuple(args.artifact_root or DEFAULT_ROOTS)
    output_dir = args.output_dir or (
        ROOT
        / "artifacts/full_evidence_orchestration_20260525"
        / f"expert_form_feature_readiness_{now_id()}_report_only"
    )
    report = build_report(
        artifact_roots=artifact_roots,
        schema_path=args.schema,
        min_source_races=args.min_source_races,
        min_source_runner_rows=args.min_source_runner_rows,
        min_shadow_feature_rows=args.min_shadow_feature_rows,
    )
    write_packet(report, output_dir)
    print(json.dumps({"final_status": report["final_status"], "output_dir": relpath(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
