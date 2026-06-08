#!/usr/bin/env python3
"""Audit pre-jump CSV sidecar metadata contracts.

The audit is report-only. It checks that each CSV has a sidecar with a
pre-race-safe `prejump_shadow_metadata` block containing target race date,
venue, race number, jump time, distance, grade, source URL, and runner
box/name list. It does not write DB rows, labels, model artifacts, registry
entries, snapshots, EV, or betting output.
"""

from __future__ import annotations

import argparse
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

from scripts.daily_race_ingest_shadow_orchestrator import (  # noqa: E402
    REQUIRED_PREJUMP_METADATA_FIELDS,
    extract_race_date,
    is_thedogs_source_url,
    looks_post_result_source_url,
    parse_current_time,
    parse_date_value,
    parse_jump_datetime,
    validate_prejump_sidecar_metadata,
)


DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "prejump_sidecar_metadata_audit_"
)
FINAL_PASS = "PREJUMP_SIDECAR_METADATA_AUDIT_PASS"
FINAL_FAIL = "PREJUMP_SIDECAR_METADATA_AUDIT_FAIL"


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_prejump_sidecar_metadata_audit_artifact:{relative}")
    return logical.absolute()


def csv_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir_not_found:{input_dir}")
    return sorted(
        path
        for path in input_dir.glob("*.csv")
        if path.is_file()
        and not path.name.endswith(".metadata.json")
        and path.parent.name != "raw_exports"
    )


def _contract_field_presence(report: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "race_date": report.get("race_date") not in (None, ""),
        "venue": report.get("venue") not in (None, ""),
        "race_number": report.get("race_number") not in (None, ""),
        "jump_time": report.get("jump_time") not in (None, ""),
        "metadata_captured_at": report.get("metadata_captured_at") not in (None, ""),
        "target_distance": report.get("target_distance") not in (None, ""),
        "target_grade": report.get("target_grade") not in (None, ""),
        "source_url": bool(report.get("source_url"))
        and is_thedogs_source_url(report.get("source_url")),
        "runner_box_name_list": int(report.get("runner_count") or 0) > 0,
        "csv_sidecar_runner_identity": (
            report.get("csv_sidecar_runner_identity_status") == "PASS"
        ),
        "canonical_final_runner_alignment": (
            report.get("canonical_runner_alignment_verified") is True
        ),
        "canonical_runner_source_url": bool(report.get("canonical_runner_source_url"))
        and is_thedogs_source_url(report.get("canonical_runner_source_url"))
        and not looks_post_result_source_url(report.get("canonical_runner_source_url")),
    }


def _freshness_status(
    report: Mapping[str, Any],
    *,
    current_time: datetime,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    race_date = parse_date_value(report.get("race_date"))
    race_date_source = "sidecar_report" if race_date else None
    if race_date is None and csv_path is not None:
        race_date, race_date_source = extract_race_date(csv_path)
    status = "freshness_unverified"
    jump_datetime = None
    minutes_to_jump = None
    error = None
    current_or_future_input = False

    if race_date is None:
        error = "race_date_missing"
    elif race_date < current_time.date():
        status = "stale_before_current_date"
    elif race_date > current_time.date():
        status = "future_date"
        current_or_future_input = True
    else:
        jump_datetime, error = parse_jump_datetime(
            race_date=race_date,
            jump_time=report.get("jump_time"),
            current_time=current_time,
        )
        if jump_datetime is None:
            status = (
                f"current_date_{error}"
                if error
                else "current_date_freshness_unverified"
            )
            current_or_future_input = True
        else:
            minutes_to_jump = (jump_datetime - current_time).total_seconds() / 60.0
            status = "current_prejump" if jump_datetime > current_time else "stale_after_jump_time"
            current_or_future_input = jump_datetime > current_time

    return {
        "status": status,
        "current_time": current_time.isoformat(),
        "race_date": race_date.isoformat() if race_date else None,
        "race_date_source": race_date_source,
        "jump_datetime": jump_datetime.isoformat() if jump_datetime else None,
        "minutes_to_jump": minutes_to_jump,
        "freshness_error": error,
        "is_current_or_future_prejump": status in {"current_prejump", "future_date"},
        "is_current_or_future_input": current_or_future_input,
    }


def _collection_status(records: Sequence[Mapping[str, Any]]) -> str:
    current_or_future = [
        record
        for record in records
        if (record.get("freshness") or {}).get("is_current_or_future_input") is True
    ]
    if not current_or_future:
        return "NO_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
    actionable = [record for record in current_or_future if record.get("status") == "PASS"]
    if actionable:
        return "PREJUMP_INPUTS_READY"
    return "PREJUMP_INPUTS_BLOCKED_BY_METADATA"


def _metadata_readiness(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current_or_future = [
        record
        for record in records
        if (record.get("freshness") or {}).get("is_current_or_future_input") is True
    ]
    blocker_counts: dict[str, int] = {}
    missing_required_field_counts: dict[str, int] = {}
    blocked_records: list[dict[str, Any]] = []
    for record in current_or_future:
        blockers = list(record.get("errors") or [])
        for field in record.get("missing_contract_fields") or []:
            field_name = str(field)
            missing_required_field_counts[field_name] = (
                missing_required_field_counts.get(field_name, 0) + 1
            )
            blockers.append(f"{field_name}_missing")
        blockers = list(dict.fromkeys(str(item) for item in blockers))
        for blocker in blockers:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        if blockers:
            blocked_records.append(
                {
                    "csv_path": record.get("csv_path"),
                    "sidecar_path": record.get("sidecar_path"),
                    "freshness_status": (record.get("freshness") or {}).get("status"),
                    "blockers": blockers,
                }
            )

    ready_records = [record for record in current_or_future if record.get("status") == "PASS"]
    if not current_or_future:
        status = "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
        capture_status = "WAITING"
    elif len(ready_records) == len(current_or_future) and not blocker_counts:
        status = "TARGET_METADATA_READY_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
        capture_status = "READY"
    else:
        status = "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS"
        capture_status = "BLOCKED"
    return {
        "schema_version": "prejump_sidecar_target_metadata_readiness_v1",
        "status": status,
        "target_metadata_capture_status": capture_status,
        "current_or_future_input_count": len(current_or_future),
        "ready_current_or_future_input_count": len(ready_records),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "missing_required_field_counts": dict(sorted(missing_required_field_counts.items())),
        "blocked_records": blocked_records,
        "historical_repair_policy": "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE",
        "no_write_guarantees": {
            "db_write": False,
            "label_write": False,
            "canonical_schema_mutation": False,
            "production_prediction_write": False,
            "betting_or_ev_output": False,
        },
    }


def audit_sidecars(
    input_dir: Path,
    *,
    generated_at: datetime | None = None,
    current_time: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    current_time = current_time or generated_at
    records = []
    pass_count = 0
    fail_count = 0
    for csv_path in csv_files(input_dir):
        report = validate_prejump_sidecar_metadata(csv_path)
        status = str(report.get("status") or "FAIL")
        field_presence = _contract_field_presence(report)
        freshness = _freshness_status(report, current_time=current_time, csv_path=csv_path)
        if status == "PASS":
            pass_count += 1
        else:
            fail_count += 1
        records.append(
            {
                "csv_path": relpath(csv_path),
                "sidecar_path": relpath(Path(f"{csv_path}.metadata.json")),
                "status": status,
                "errors": report.get("errors") or report.get("fail_reasons") or [],
                "contract_required_fields": list(REQUIRED_PREJUMP_METADATA_FIELDS),
                "contract_field_presence": field_presence,
                "missing_contract_fields": [
                    field for field in REQUIRED_PREJUMP_METADATA_FIELDS if not field_presence[field]
                ],
                "freshness": freshness,
                "required_fields": {
                    "race_date": report.get("race_date"),
                    "venue": report.get("venue"),
                    "race_number": report.get("race_number"),
                    "jump_time": report.get("jump_time"),
                    "metadata_captured_at": report.get("metadata_captured_at"),
                    "distance": report.get("distance") or report.get("target_distance"),
                    "grade": report.get("grade") or report.get("target_grade"),
                    "source_url": report.get("source_url"),
                    "canonical_runner_source_url": report.get(
                        "canonical_runner_source_url"
                    ),
                    "runner_count": report.get("runner_count"),
                },
                "source_url_is_thedogs": is_thedogs_source_url(report.get("source_url")),
                "target_distance_source": report.get("target_distance_source"),
                "target_grade_source": report.get("target_grade_source"),
                "metadata_is_leakage_safe": report.get("metadata_is_leakage_safe"),
                "runner_box_name_list": report.get("participants") or [],
                "csv_sidecar_runner_identity_status": report.get(
                    "csv_sidecar_runner_identity_status"
                ),
                "csv_sidecar_runner_identity_mismatches": report.get(
                    "csv_sidecar_runner_identity_mismatches"
                )
                or {},
                "canonical_runner_alignment_status": report.get(
                    "canonical_runner_alignment_status"
                ),
                "canonical_runner_alignment_verified": (
                    report.get("canonical_runner_alignment_verified") is True
                ),
                "canonical_runner_set_status": report.get("canonical_runner_set_status"),
                "canonical_runner_count": report.get("canonical_runner_count"),
                "canonical_prediction_runner_count": report.get(
                    "canonical_prediction_runner_count"
                ),
                "canonical_runner_source_url": report.get("canonical_runner_source_url"),
                "rejected_metadata_sources": list(
                    report.get("rejected_metadata_sources") or []
                ),
            }
        )
    current_or_future_records = [
        record
        for record in records
        if record["freshness"]["is_current_or_future_input"] is True
    ]
    actionable_records = [
        record for record in current_or_future_records if record["status"] == "PASS"
    ]
    stale_before_count = sum(
        1 for record in records if record["freshness"]["status"] == "stale_before_current_date"
    )
    stale_after_count = sum(
        1 for record in records if record["freshness"]["status"] == "stale_after_jump_time"
    )
    freshness_unverified_count = sum(
        1 for record in records if record["freshness"]["status"] == "freshness_unverified"
    )
    metadata_readiness = _metadata_readiness(records)
    return {
        "schema_version": "prejump_sidecar_metadata_audit_v1",
        "generated_at": generated_at.isoformat(),
        "current_time": current_time.isoformat(),
        "input_dir": relpath(input_dir),
        "csv_count": len(records),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "current_or_future_prejump_count": len(current_or_future_records),
        "current_or_future_input_count": len(current_or_future_records),
        "current_or_future_prejump_pass_count": len(actionable_records),
        "current_or_future_input_pass_count": len(actionable_records),
        "stale_count": stale_before_count + stale_after_count,
        "stale_before_current_date_count": stale_before_count,
        "stale_after_jump_time_count": stale_after_count,
        "freshness_unverified_count": freshness_unverified_count,
        "collection_status": _collection_status(records),
        "target_metadata_readiness": metadata_readiness,
        "final_status": FINAL_PASS if records and fail_count == 0 else FINAL_FAIL,
        "records": records,
        "no_write_guarantees": {
            "production_promotion": False,
            "registry_mutation": False,
            "production_pointer_update": False,
            "production_prediction_write": False,
            "db_write": False,
            "label_write": False,
            "canonical_schema_mutation": False,
            "tgr_enabled": False,
            "betting_or_ev_output": False,
        },
    }


def build_summary(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Pre-Jump Sidecar Metadata Audit",
            "",
            f"- Final status: `{report.get('final_status')}`",
            f"- Input dir: `{report.get('input_dir')}`",
            f"- CSV count: `{report.get('csv_count')}`",
            f"- Pass count: `{report.get('pass_count')}`",
            f"- Fail count: `{report.get('fail_count')}`",
            f"- Current/future pre-jump count: `{report.get('current_or_future_prejump_count')}`",
            f"- Current/future pre-jump pass count: `{report.get('current_or_future_prejump_pass_count')}`",
            f"- Stale count: `{report.get('stale_count')}`",
            f"- Collection status: `{report.get('collection_status')}`",
            f"- Target metadata readiness: `{(report.get('target_metadata_readiness') or {}).get('status')}`",
            f"- Target metadata blockers: `{(report.get('target_metadata_readiness') or {}).get('blocker_counts')}`",
            "",
            "This audit is report-only and writes no production state.",
            "",
        ]
    )


def run_audit(
    *,
    input_dir: Path,
    output_dir: Path | None = None,
    current_time: datetime | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    current_time = current_time or generated_at
    output_dir = output_dir or DEFAULT_OUTPUT_PARENT / f"prejump_sidecar_metadata_audit_{now_id(generated_at)}"
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = audit_sidecars(input_dir, generated_at=generated_at, current_time=current_time)
    write_json(output_dir / "prejump_sidecar_metadata_audit.json", report)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["final_status"],
        "csv_count": report["csv_count"],
        "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "current_or_future_prejump_count": report["current_or_future_prejump_count"],
        "current_or_future_prejump_pass_count": report["current_or_future_prejump_pass_count"],
        "collection_status": report["collection_status"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--current-time")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        current_time=parse_current_time(args.current_time) if args.current_time else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
