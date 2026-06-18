#!/usr/bin/env python3
"""Backfill report-only Expert Form shadow feature rows from safe sidecars.

This packet is deliberately narrower than live scoring. It reads already
source-safe TheDogs Expert Form sidecars and their matching pre-race CSVs,
flattens Expert Form columns into ``shadow_feature_rows.json``, and writes only
under the packet artifact directory. It does not train, score, mutate schemas,
write DB rows, emit EV, or bet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.build_expert_form_feature_readiness_packet import (  # noqa: E402
    DEFAULT_ROOTS,
    stable_race_id,
)
from scripts.run_shadow_non_tgr_rf_evaluation import (  # noqa: E402
    csv_value,
    expert_form_runner_features,
    load_live_csv,
    parse_live_runner_identity,
    protected_path_snapshot,
    protected_path_verification,
    shadow_relpath,
)
from utils.expert_form_metadata import safe_expert_form_metadata_from_payload  # noqa: E402


OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "expert_form_shadow_feature_row_backfill_"
)
FINAL_SOURCE_LOW = "KEEP_COLLECTING_ONLY_EXPERT_FORM_SOURCE_COVERAGE_LOW"
FINAL_BACKFILL_READY = "READY_FOR_SCHEMA_TRIAL_REPORT_ONLY"
FINAL_BACKFILL_INCOMPLETE = "EXPERT_FORM_FEATURE_ROW_BACKFILL_INCOMPLETE"

NO_WRITE_GUARANTEES = {
    "report_only": True,
    "training_run": False,
    "model_scoring": False,
    "schema_mutation": False,
    "registry_mutation": False,
    "production_prediction_write": False,
    "db_write": False,
    "label_write": False,
    "ev_output": False,
    "betting_output": False,
}


def now_id(value: datetime | None = None) -> str:
    return (value or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path) -> str:
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
        raise ValueError(
            f"output_dir_must_be_expert_form_shadow_feature_row_backfill_artifact:{relative}"
        )
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


def csv_path_for_sidecar(sidecar_path: Path) -> Path:
    text = str(sidecar_path)
    suffix = ".metadata.json"
    if text.endswith(suffix):
        return Path(text[: -len(suffix)])
    return sidecar_path.with_suffix("")


def sidecar_candidate(sidecar_path: Path) -> dict[str, Any]:
    base = {
        "race_id": sidecar_path.name.replace(".csv.metadata.json", ""),
        "sidecar_path": relpath(sidecar_path),
        "csv_path": relpath(csv_path_for_sidecar(sidecar_path)),
        "status": "DATA_MISSING",
        "runner_count": 0,
        "captured_at": None,
        "rejected_reasons": [],
    }
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            **base,
            "status": "REJECTED",
            "rejected_reasons": [f"sidecar_unreadable:{type(exc).__name__}"],
        }
    if not isinstance(payload, Mapping):
        return {**base, "status": "REJECTED", "rejected_reasons": ["sidecar_not_object"]}
    metadata = safe_expert_form_metadata_from_payload(payload)
    csv_path = csv_path_for_sidecar(sidecar_path)
    rejected = list(metadata.get("rejected_reasons") or [])
    if metadata.get("metadata_is_leakage_safe") is not True:
        return {
            **base,
            "race_id": stable_race_id(payload, sidecar_path),
            "status": "REJECTED",
            "source_url": metadata.get("source_url"),
            "captured_at": metadata.get("captured_at"),
            "rejected_reasons": rejected,
        }
    if not csv_path.exists():
        return {
            **base,
            "race_id": stable_race_id(payload, sidecar_path),
            "status": "CSV_MISSING",
            "source_url": metadata.get("source_url"),
            "captured_at": metadata.get("captured_at"),
            "runner_count": metadata.get("runner_count") or 0,
            "rejected_reasons": ["matching_csv_missing"],
        }
    return {
        **base,
        "race_id": stable_race_id(payload, sidecar_path),
        "status": "ACCEPTED",
        "source_url": metadata.get("source_url"),
        "captured_at": metadata.get("captured_at"),
        "runner_count": metadata.get("runner_count") or 0,
        "metadata": metadata,
        "payload": payload,
    }


def discover_candidates(artifact_roots: Sequence[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in artifact_roots:
        if not root.exists():
            continue
        for sidecar_path in sorted(root.rglob("*.csv.metadata.json")):
            resolved = sidecar_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(sidecar_candidate(sidecar_path))
    return candidates


def selected_candidates_by_race(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if candidate.get("status") != "ACCEPTED":
            continue
        race_id = str(candidate.get("race_id") or candidate.get("sidecar_path") or "")
        current = selected.get(race_id)
        if current is None:
            selected[race_id] = candidate
            continue
        current_key = (
            str(current.get("captured_at") or ""),
            str(current.get("sidecar_path") or ""),
        )
        candidate_key = (
            str(candidate.get("captured_at") or ""),
            str(candidate.get("sidecar_path") or ""),
        )
        if candidate_key > current_key:
            selected[race_id] = candidate
    return [dict(item) for item in sorted(selected.values(), key=lambda row: str(row.get("race_id")))]


def feature_rows_from_candidate(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    csv_path = ROOT / str(candidate["csv_path"])
    rows = load_live_csv(csv_path)
    metadata = candidate.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for raw in rows:
        dog_name, box = parse_live_runner_identity(
            csv_value(raw, "dog_name", "Dog Name", "dog", "runner", "runner_name"),
            csv_value(raw, "box_number", "box", "Box", "BOX"),
        )
        if not dog_name:
            continue
        expert_features = expert_form_runner_features(
            expert_form_metadata=metadata,
            dog_name=dog_name,
            box_number=box,
        )
        out.append(
            {
                "race_id": candidate.get("race_id"),
                "dog_name": dog_name,
                "box_number": box,
                "source_csv_path": candidate.get("csv_path"),
                "source_sidecar_path": candidate.get("sidecar_path"),
                **expert_features,
            }
        )
    return out


def build_report(
    *,
    artifact_roots: Sequence[Path],
    min_source_races: int,
    min_source_runner_rows: int,
    min_feature_rows: int,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    candidates = discover_candidates(artifact_roots)
    selected = selected_candidates_by_race(candidates)
    feature_rows: list[dict[str, Any]] = []
    build_errors: list[dict[str, Any]] = []
    for candidate in selected:
        try:
            feature_rows.extend(feature_rows_from_candidate(candidate))
        except Exception as exc:
            build_errors.append(
                {
                    "race_id": candidate.get("race_id"),
                    "sidecar_path": candidate.get("sidecar_path"),
                    "csv_path": candidate.get("csv_path"),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
    safe_feature_rows = [
        row for row in feature_rows if row.get("expert_form_metadata_from_sidecar") is True
    ]
    safe_source_runner_rows = sum(int(row.get("runner_count") or 0) for row in selected)
    blockers: list[str] = []
    if len(selected) < min_source_races:
        blockers.append("safe_expert_form_source_races_below_min")
    if safe_source_runner_rows < min_source_runner_rows:
        blockers.append("safe_expert_form_source_runner_rows_below_min")
    if blockers:
        final_status = FINAL_SOURCE_LOW
    elif len(safe_feature_rows) < min_feature_rows:
        blockers.append("safe_expert_form_feature_rows_below_min")
        final_status = FINAL_BACKFILL_INCOMPLETE
    elif build_errors:
        blockers.append("feature_row_build_errors_present")
        final_status = FINAL_BACKFILL_INCOMPLETE
    else:
        final_status = FINAL_BACKFILL_READY
    status_counts: dict[str, int] = {}
    for candidate in candidates:
        status = str(candidate.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": "expert_form_shadow_feature_row_backfill_packet_v1",
        "generated_at": generated_at.isoformat(),
        "artifact_roots": [relpath(path) for path in artifact_roots],
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "thresholds": {
            "min_source_races": min_source_races,
            "min_source_runner_rows": min_source_runner_rows,
            "min_feature_rows": min_feature_rows,
        },
        "coverage_summary": {
            "sidecars_scanned": len(candidates),
            "candidate_status_counts": dict(sorted(status_counts.items())),
            "selected_safe_source_races": len(selected),
            "selected_safe_source_runner_rows": safe_source_runner_rows,
            "feature_rows_written": len(feature_rows),
            "safe_expert_form_feature_rows": len(safe_feature_rows),
            "feature_row_build_error_count": len(build_errors),
        },
        "final_status": final_status,
        "activation_allowed": False,
        "training_run": False,
        "model_scoring": False,
        "blockers": blockers,
        "selected_sources": [
            {
                key: candidate.get(key)
                for key in (
                    "race_id",
                    "sidecar_path",
                    "csv_path",
                    "source_url",
                    "captured_at",
                    "runner_count",
                    "status",
                )
            }
            for candidate in selected
        ],
        "rejected_or_missing_sources": [
            {
                key: candidate.get(key)
                for key in (
                    "race_id",
                    "sidecar_path",
                    "csv_path",
                    "status",
                    "source_url",
                    "captured_at",
                    "runner_count",
                    "rejected_reasons",
                )
            }
            for candidate in candidates
            if candidate.get("status") != "ACCEPTED"
        ],
        "build_errors": build_errors,
        "feature_rows": feature_rows,
    }


def summary_md(report: Mapping[str, Any], output_dir: Path) -> str:
    summary = report.get("coverage_summary") or {}
    blockers = list(report.get("blockers") or [])
    return "\n".join(
        [
            "# Expert Form Shadow Feature Row Backfill Packet",
            "",
            f"Final status: `{report.get('final_status')}`",
            f"Activation allowed: `{report.get('activation_allowed')}`",
            "",
            "## Coverage",
            "",
            f"- Selected safe source races: `{summary.get('selected_safe_source_races')}`",
            f"- Selected safe source runner rows: `{summary.get('selected_safe_source_runner_rows')}`",
            f"- Feature rows written: `{summary.get('feature_rows_written')}`",
            f"- Safe Expert Form feature rows: `{summary.get('safe_expert_form_feature_rows')}`",
            "",
            "## Blockers",
            "",
            *(f"- `{blocker}`" for blocker in blockers),
            "",
            "## Artifacts",
            "",
            f"- `{relpath(output_dir / 'shadow_feature_rows.json')}`",
            f"- `{relpath(output_dir / 'expert_form_backfill_sources.csv')}`",
            f"- `{relpath(output_dir / 'expert_form_shadow_feature_row_backfill_report.json')}`",
            f"- `{relpath(output_dir / 'protected_path_verification.json')}`",
            "",
            "No training, model scoring, schema mutation, registry mutation, DB write, label write, EV output, or betting output was performed.",
            "",
        ]
    )


def write_packet(report: Mapping[str, Any], output_dir: Path, protected: Mapping[str, Any]) -> None:
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "shadow_feature_rows.json", report["feature_rows"])
    write_csv(
        output_dir / "expert_form_backfill_sources.csv",
        report["selected_sources"],
        [
            "race_id",
            "sidecar_path",
            "csv_path",
            "source_url",
            "captured_at",
            "runner_count",
            "status",
        ],
    )
    write_csv(
        output_dir / "expert_form_backfill_rejected_or_missing_sources.csv",
        report["rejected_or_missing_sources"],
        [
            "race_id",
            "sidecar_path",
            "csv_path",
            "status",
            "source_url",
            "captured_at",
            "runner_count",
            "rejected_reasons",
        ],
    )
    report_for_disk = dict(report)
    report_for_disk["feature_rows"] = {
        "path": relpath(output_dir / "shadow_feature_rows.json"),
        "rows": len(report["feature_rows"]),
    }
    write_json(
        output_dir / "expert_form_shadow_feature_row_backfill_report.json",
        report_for_disk,
    )
    write_json(output_dir / "protected_path_verification.json", protected)
    manifest = {
        "schema_version": "expert_form_shadow_feature_row_backfill_manifest_v1",
        "generated_at": report["generated_at"],
        "files": {
            "report": relpath(output_dir / "expert_form_shadow_feature_row_backfill_report.json"),
            "summary": relpath(output_dir / "SUMMARY.md"),
            "final_status": relpath(output_dir / "final_status.txt"),
            "feature_rows": relpath(output_dir / "shadow_feature_rows.json"),
            "sources": relpath(output_dir / "expert_form_backfill_sources.csv"),
            "rejected_or_missing_sources": relpath(
                output_dir / "expert_form_backfill_rejected_or_missing_sources.csv"
            ),
            "protected_path_verification": relpath(output_dir / "protected_path_verification.json"),
        },
        "no_write_guarantees": report["no_write_guarantees"],
    }
    write_json(output_dir / "output_manifest.json", manifest)
    write_text(output_dir / "SUMMARY.md", summary_md(report, output_dir))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", action="append", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-source-races", type=int, default=20)
    parser.add_argument("--min-source-runner-rows", type=int, default=100)
    parser.add_argument("--min-feature-rows", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_roots = tuple(args.artifact_root or DEFAULT_ROOTS)
    output_dir = args.output_dir or (
        ROOT
        / "artifacts/full_evidence_orchestration_20260525"
        / f"expert_form_shadow_feature_row_backfill_{now_id()}_report_only"
    )
    protected_before = protected_path_snapshot()
    report = build_report(
        artifact_roots=artifact_roots,
        min_source_races=args.min_source_races,
        min_source_runner_rows=args.min_source_runner_rows,
        min_feature_rows=args.min_feature_rows,
    )
    protected = protected_path_verification(protected_before)
    write_packet(report, output_dir, protected)
    print(
        json.dumps(
            {
                "final_status": report["final_status"],
                "output_dir": shadow_relpath(output_dir),
                "coverage_summary": report["coverage_summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
