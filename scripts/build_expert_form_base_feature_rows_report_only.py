#!/usr/bin/env python3
"""Build current-schema base feature rows for the Expert Form race set.

This packet is report-only. It reads accepted Expert Form shadow feature rows,
dedupes their source pre-jump CSVs, and uses the existing live feature-row
builder to materialize the current canonical feature surface for those same
races. It does not train, score, mutate schemas, write DB rows, write labels,
rewrite snapshots, emit EV, or bet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.build_expert_form_feature_readiness_packet import DEFAULT_SCHEMA  # noqa: E402
from scripts.collect_expert_form_official_result_labels_report_only import (  # noqa: E402
    DEFAULT_FEATURE_ROWS_GLOB,
)
from scripts.run_feature_recovery_execution_v1 import load_json, sha256_file  # noqa: E402
from scripts.run_shadow_non_tgr_rf_evaluation import (  # noqa: E402
    build_live_feature_rows,
    protected_path_snapshot,
    protected_path_verification,
    shadow_relpath,
)


OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "expert_form_base_feature_rows_"
)
FINAL_READY = "BASE_FEATURE_ROWS_READY_REPORT_ONLY"
FINAL_DATA_MISSING = "DATA_MISSING_BASE_FEATURE_ROWS"

NO_WRITE_GUARANTEES = {
    "report_only": True,
    "training_run": False,
    "model_scoring": False,
    "canonical_schema_mutation": False,
    "registry_mutation": False,
    "production_prediction_write": False,
    "db_write": False,
    "label_write": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
    "ev_output": False,
    "betting_output": False,
}


def now_id(value: datetime | None = None) -> str:
    return (value or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


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
        relative = logical.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_expert_form_base_feature_rows_artifact:{relative}")
    return logical


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
            writer.writerow({key: row.get(key) for key in fieldnames})


def read_json_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def latest_feature_rows_path() -> Path | None:
    matches = sorted(ROOT.glob(DEFAULT_FEATURE_ROWS_GLOB))
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))


def source_csv_paths(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Path], list[dict[str, Any]]]:
    selected: dict[str, Path] = {}
    rejected: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source_csv_path") or row.get("source_csv") or "").strip()
        race_id = str(row.get("race_id") or "").strip()
        if not source:
            rejected.append({"race_id": race_id, "reason": "source_csv_missing"})
            continue
        path = ROOT / source
        if not path.exists():
            rejected.append({"race_id": race_id, "source_csv_path": source, "reason": "source_csv_not_found"})
            continue
        selected.setdefault(str(path), path)
    return [selected[key] for key in sorted(selected)], rejected


def coverage_rows(base_rows: Sequence[Mapping[str, Any]], expert_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expert_counts: Counter[str] = Counter(str(row.get("race_id") or "") for row in expert_rows)
    base_counts: Counter[str] = Counter(str(row.get("race_id") or "") for row in base_rows)
    rows = []
    for race_id in sorted(set(expert_counts) | set(base_counts)):
        rows.append(
            {
                "race_id": race_id,
                "expert_feature_rows": expert_counts.get(race_id, 0),
                "base_feature_rows": base_counts.get(race_id, 0),
                "coverage_status": "PASS"
                if expert_counts.get(race_id, 0) and base_counts.get(race_id, 0) == expert_counts.get(race_id, 0)
                else "MISMATCH",
            }
        )
    return rows


def build_report(
    *,
    expert_feature_rows_path: Path | None,
    schema_path: Path,
    db_path: Path,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    expert_rows = read_json_rows(expert_feature_rows_path)
    schema = load_json(schema_path)
    input_paths, rejected_sources = source_csv_paths(expert_rows)
    build_error = None
    base_rows: list[dict[str, Any]] = []
    if input_paths and db_path.exists() and db_path.stat().st_size > 0:
        try:
            base_rows = build_live_feature_rows(
                input_paths=input_paths,
                schema=schema,
                db_path=db_path,
            )
        except Exception as exc:  # noqa: BLE001 - packet records fail-closed reason.
            build_error = f"{type(exc).__name__}:{exc}"
    elif not input_paths:
        build_error = "source_csvs_missing"
    elif not db_path.exists():
        build_error = "db_path_missing"
    else:
        build_error = "db_path_zero_bytes"

    rows_by_race = coverage_rows(base_rows, expert_rows)
    pass_races = sum(1 for row in rows_by_race if row["coverage_status"] == "PASS")
    expert_races = {str(row.get("race_id") or "") for row in expert_rows if row.get("race_id")}
    final_status = FINAL_READY if base_rows and pass_races == len(expert_races) else FINAL_DATA_MISSING
    return {
        "schema_version": "expert_form_base_feature_rows_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "expert_feature_rows_path": relpath(expert_feature_rows_path) if expert_feature_rows_path else None,
        "schema_path": relpath(schema_path),
        "schema_sha256": sha256_file(schema_path) if schema_path.exists() else None,
        "db_path": str(db_path),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "build_error": build_error,
        "coverage_summary": {
            "expert_feature_rows": len(expert_rows),
            "expert_races": len(expert_races),
            "input_csvs": len(input_paths),
            "rejected_source_rows": len(rejected_sources),
            "base_feature_rows": len(base_rows),
            "base_races": len({row.get("race_id") for row in base_rows}),
            "coverage_pass_races": pass_races,
            "coverage_mismatch_races": sum(
                1 for row in rows_by_race if row["coverage_status"] != "PASS"
            ),
        },
        "input_csvs": [relpath(path) for path in input_paths],
        "rejected_sources": rejected_sources,
        "coverage_rows": rows_by_race,
        "base_feature_rows": base_rows,
    }


def summary_md(report: Mapping[str, Any], output_dir: Path) -> str:
    summary = report.get("coverage_summary") or {}
    return "\n".join(
        [
            "# Expert Form Base Feature Rows Packet",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            "## Coverage",
            "",
            f"- Expert races: `{summary.get('expert_races')}`",
            f"- Input CSVs: `{summary.get('input_csvs')}`",
            f"- Base rows: `{summary.get('base_feature_rows')}`",
            f"- Base races: `{summary.get('base_races')}`",
            f"- Coverage pass races: `{summary.get('coverage_pass_races')}`",
            f"- Build error: `{report.get('build_error')}`",
            "",
            "## Artifacts",
            "",
            f"- `{relpath(output_dir / 'base_feature_rows.json')}`",
            f"- `{relpath(output_dir / 'base_feature_row_coverage.csv')}`",
            f"- `{relpath(output_dir / 'expert_form_base_feature_rows_report.json')}`",
            "",
            "No training, model scoring, DB write, label write, schema mutation, registry mutation, EV output, or betting output was performed.",
            "",
        ]
    )


def write_packet(report: Mapping[str, Any], output_dir: Path, protected: Mapping[str, Any]) -> None:
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "base_feature_rows.json", report["base_feature_rows"])
    write_csv(
        output_dir / "base_feature_row_coverage.csv",
        report["coverage_rows"],
        ["race_id", "expert_feature_rows", "base_feature_rows", "coverage_status"],
    )
    write_json(output_dir / "protected_path_verification.json", protected)
    report_for_disk = dict(report)
    report_for_disk.pop("base_feature_rows", None)
    report_for_disk.pop("coverage_rows", None)
    report_for_disk["artifacts"] = {
        "base_feature_rows": relpath(output_dir / "base_feature_rows.json"),
        "coverage": relpath(output_dir / "base_feature_row_coverage.csv"),
    }
    write_json(output_dir / "expert_form_base_feature_rows_report.json", report_for_disk)
    manifest = {
        "schema_version": "expert_form_base_feature_rows_manifest_v1",
        "generated_at": report["generated_at"],
        "files": {
            "report": relpath(output_dir / "expert_form_base_feature_rows_report.json"),
            "summary": relpath(output_dir / "SUMMARY.md"),
            "final_status": relpath(output_dir / "final_status.txt"),
            "base_feature_rows": relpath(output_dir / "base_feature_rows.json"),
            "coverage": relpath(output_dir / "base_feature_row_coverage.csv"),
            "protected_path_verification": relpath(output_dir / "protected_path_verification.json"),
        },
        "no_write_guarantees": report["no_write_guarantees"],
    }
    write_json(output_dir / "output_manifest.json", manifest)
    write_text(output_dir / "SUMMARY.md", summary_md(report, output_dir))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-feature-rows", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        ROOT
        / "artifacts/full_evidence_orchestration_20260525"
        / f"expert_form_base_feature_rows_{now_id()}_report_only"
    )
    output_dir = assert_output_dir_safe(output_dir)
    expert_rows_path = args.expert_feature_rows or latest_feature_rows_path()
    protected_before = protected_path_snapshot()
    report = build_report(
        expert_feature_rows_path=expert_rows_path,
        schema_path=args.schema,
        db_path=args.db,
    )
    protected = protected_path_verification(protected_before)
    write_packet(report, output_dir, protected)
    print(
        json.dumps(
            {
                "final_status": report["final_status"],
                "output_dir": shadow_relpath(output_dir),
                "coverage_summary": report["coverage_summary"],
                "build_error": report["build_error"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
