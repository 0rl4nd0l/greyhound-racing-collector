#!/usr/bin/env python3
"""Build a no-write official repair plan for one reviewed race.

This helper consumes a single-race official gap review packet and emits exact
candidate operation shapes for a future approved repair. It does not write DB
rows, labels, snapshots, manifests, datasets, model files, registries, betting
decisions, or EV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "single_race_official_repair_plan_v1"
GAP_SCHEMA_VERSION = "single_race_official_gap_review_packet_v1"
STATUS_OK = "REPORT_ONLY_SINGLE_RACE_OFFICIAL_REPAIR_PLAN"
STATUS_FAILURES = "REPORT_ONLY_SINGLE_RACE_OFFICIAL_REPAIR_PLAN_WITH_FAILURES"
OFFICIAL_SOURCE = "thedogs_official"

WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "dataset_regeneration": False,
    "model_training": False,
    "model_persistence": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}

FORBIDDEN_APPROVAL_ENV_VARS = (
    "APPROVE_RESULT_LABEL_WRITE",
    "APPROVE_GREYHOUND_DB_WRITE",
    "GREYHOUND_ALLOW_DB_WRITE",
    "GREYHOUND_ALLOW_TGR",
)

OPERATION_FIELDS = [
    "operation_id",
    "operation_type",
    "status",
    "table_name",
    "race_id",
    "name_key",
    "official_dog_name",
    "selector_json",
    "before_json",
    "after_json",
    "changed_fields",
    "blockers",
    "sql_shape",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _name_key(value: Any) -> str:
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(value or "").strip())
    text = text.replace('"', "").replace("'", "").replace("`", "")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root_path = (root or ROOT).expanduser().resolve(strict=False)
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root_path / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{logical}") from exc
    return resolved, relative


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    return _repo_output_path(path, root)[1]


def _assert_output_dir_safe(output_dir: Path, root: Path | None = None) -> Path:
    resolved, relative = _repo_output_path(output_dir, root)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _rows_by_name(rows: Iterable[Mapping[str, Any]], *, field: str = "dog_name") -> dict[str, Mapping[str, Any]]:
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = _name_key(row.get(field) or row.get("official_dog_name") or row.get("name_key"))
        if key and key not in by_name:
            by_name[key] = row
    return by_name


def _source_url(packet: Mapping[str, Any]) -> str | None:
    failure_review = _mapping(packet.get("failure_review_row"))
    if failure_review.get("source_url"):
        return str(failure_review.get("source_url"))
    return None


def _official_winner(official_rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in official_rows:
        if _safe_int(row.get("finish_position")) == 1:
            return row
    return {}


def _changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    return sorted(key for key, value in after.items() if before.get(key) != value)


def _validate_gap_packet(
    *,
    packet: Mapping[str, Any],
    runner_rows: list[Mapping[str, Any]],
    failures: list[str],
) -> None:
    if packet.get("schema_version") != GAP_SCHEMA_VERSION:
        failures.append("gap_packet_schema_mismatch")
    if packet.get("report_only") is not True:
        failures.append("gap_packet_not_report_only")
    writes = _mapping(packet.get("writes_performed"))
    for key, value in writes.items():
        if value is not False:
            failures.append(f"gap_packet_write_flag_true:{key}")
    summary = _mapping(packet.get("summary"))
    if not summary.get("race_id"):
        failures.append("gap_packet_race_id_missing")
    official_count = _safe_int(summary.get("official_runner_count"))
    if official_count is not None and official_count != len(runner_rows):
        failures.append("runner_csv_count_mismatch")


def _metadata_update_candidate(
    *,
    packet: Mapping[str, Any],
    race_id: str,
    official_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = dict(_mapping(packet.get("db_metadata")))
    official_count = len(official_rows)
    winner = _official_winner(official_rows)
    source_url = _source_url(packet)

    before = {
        "winner_name": metadata.get("winner_name"),
        "winner_source": metadata.get("winner_source"),
        "results_status": metadata.get("results_status"),
        "field_size": metadata.get("field_size"),
        "actual_field_size": metadata.get("actual_field_size"),
        "url": metadata.get("url"),
        "data_source": metadata.get("data_source"),
    }
    after_patch: dict[str, Any] = {}
    if metadata.get("winner_source") != OFFICIAL_SOURCE:
        after_patch["winner_source"] = OFFICIAL_SOURCE
    if _safe_int(metadata.get("actual_field_size")) != official_count:
        after_patch["actual_field_size"] = official_count
    if source_url and not metadata.get("url"):
        after_patch["url"] = source_url
    if winner and _name_key(metadata.get("winner_name")) != _name_key(winner.get("dog_name")):
        after_patch["winner_name"] = winner.get("dog_name")

    deferred_policy_candidates = []
    if _safe_int(metadata.get("field_size")) != official_count:
        deferred_policy_candidates.append(
            {
                "field": "field_size",
                "before": metadata.get("field_size"),
                "candidate_after": official_count,
                "status": "POLICY_DECISION_REQUIRED",
                "reason": (
                    "current field_size equals legacy DB runner count; official runner "
                    "count differs and changing this field may affect model features"
                ),
            }
        )

    return {
        "operation_id": "metadata_update_01",
        "operation_type": "race_metadata_update",
        "status": "REQUIRES_EXPLICIT_APPROVAL",
        "table_name": "race_metadata",
        "race_id": race_id,
        "selector": {"race_id": race_id},
        "before": before,
        "after_patch": after_patch,
        "deferred_policy_candidates": deferred_policy_candidates,
        "blockers": [
            "explicit_operator_approval_required",
            "db_backup_required_before_apply",
            "exact_row_allowlist_required",
        ],
        "write_sql_shape_if_later_approved": (
            "UPDATE race_metadata SET winner_source=?, actual_field_size=?, "
            "url=COALESCE(?, url) WHERE race_id=?"
        ),
    }


def _dog_update_candidates(
    *,
    race_id: str,
    official_rows: list[Mapping[str, Any]],
    db_rows: list[Mapping[str, Any]],
    runner_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    db_by_name = _rows_by_name(db_rows)
    review_by_name = _rows_by_name(runner_rows, field="official_dog_name")
    candidates = []
    for official in sorted(
        official_rows,
        key=lambda row: (_safe_int(row.get("finish_position")) or 999, _safe_int(row.get("box_number")) or 999),
    ):
        key = _name_key(official.get("dog_name"))
        db_row = _mapping(db_by_name.get(key))
        if not db_row:
            continue
        review_row = _mapping(review_by_name.get(key))
        before = {
            "box_number": _safe_int(db_row.get("box_number")),
            "finish_position": _safe_int(db_row.get("finish_position")),
            "placing": _safe_int(db_row.get("placing")),
            "scraped_finish_position": db_row.get("scraped_finish_position"),
            "data_source": db_row.get("data_source"),
        }
        official_finish = _safe_int(official.get("finish_position"))
        after = {
            "box_number": _safe_int(official.get("box_number")),
            "finish_position": official_finish,
            "placing": official_finish,
            "scraped_finish_position": str(official_finish) if official_finish is not None else None,
            "data_source": OFFICIAL_SOURCE,
        }
        candidates.append(
            {
                "operation_id": f"dog_update_{len(candidates) + 1:02d}",
                "operation_type": "dog_row_update_by_name",
                "status": "REQUIRES_EXPLICIT_APPROVAL",
                "table_name": "dog_race_data",
                "race_id": race_id,
                "name_key": key,
                "db_dog_name": db_row.get("dog_name"),
                "official_dog_name": official.get("dog_name"),
                "selector": {"race_id": race_id, "dog_name": db_row.get("dog_name")},
                "before": before,
                "after": after,
                "changed_fields": _changed_fields(before, after),
                "gap_flags": str(review_row.get("gap_flags") or ""),
                "blockers": [
                    "explicit_operator_approval_required",
                    "db_backup_required_before_apply",
                    "exact_row_allowlist_required",
                ],
                "write_sql_shape_if_later_approved": (
                    "UPDATE dog_race_data SET box_number=?, finish_position=?, placing=?, "
                    "scraped_finish_position=?, data_source=? WHERE race_id=? AND dog_name=?"
                ),
            }
        )
    return candidates


def _missing_runner_insert_candidates(
    *,
    race_id: str,
    official_rows: list[Mapping[str, Any]],
    db_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    db_by_name = _rows_by_name(db_rows)
    candidates = []
    for official in sorted(
        official_rows,
        key=lambda row: (_safe_int(row.get("finish_position")) or 999, _safe_int(row.get("box_number")) or 999),
    ):
        key = _name_key(official.get("dog_name"))
        if key in db_by_name:
            continue
        official_finish = _safe_int(official.get("finish_position"))
        values = {
            "race_id": race_id,
            "dog_name": official.get("dog_name"),
            "dog_clean_name": official.get("dog_name"),
            "box_number": _safe_int(official.get("box_number")),
            "finish_position": official_finish,
            "placing": official_finish,
            "scraped_finish_position": str(official_finish) if official_finish is not None else None,
            "extraction_timestamp": "<APPLY_TIME_UTC>",
            "data_source": OFFICIAL_SOURCE,
        }
        candidates.append(
            {
                "operation_id": f"dog_insert_{len(candidates) + 1:02d}",
                "operation_type": "missing_dog_row_insert",
                "status": "BLOCKED_REQUIRES_SCHEMA_DEFAULT_POLICY_AND_APPROVAL",
                "table_name": "dog_race_data",
                "race_id": race_id,
                "name_key": key,
                "official_dog_name": official.get("dog_name"),
                "selector": None,
                "before": None,
                "after": values,
                "changed_fields": sorted(values),
                "blockers": [
                    "schema_default_policy_required_for_missing_runner_insert",
                    "explicit_operator_approval_required",
                    "db_backup_required_before_apply",
                    "exact_row_allowlist_required",
                ],
                "write_sql_shape_if_later_approved": (
                    "INSERT INTO dog_race_data (race_id, dog_name, dog_clean_name, "
                    "box_number, finish_position, placing, scraped_finish_position, "
                    "extraction_timestamp, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
            }
        )
    return candidates


def _operation_csv_rows(packet: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    operations = []
    metadata = _mapping(packet.get("metadata_update_candidate"))
    if metadata.get("after_patch") or metadata.get("deferred_policy_candidates"):
        operations.append(metadata)
    operations.extend(_list(packet.get("dog_row_update_candidates")))
    operations.extend(_list(packet.get("missing_runner_insert_candidates")))
    for operation in operations:
        op = _mapping(operation)
        rows.append(
            {
                "operation_id": str(op.get("operation_id") or ""),
                "operation_type": str(op.get("operation_type") or ""),
                "status": str(op.get("status") or ""),
                "table_name": str(op.get("table_name") or ""),
                "race_id": str(op.get("race_id") or ""),
                "name_key": str(op.get("name_key") or ""),
                "official_dog_name": str(op.get("official_dog_name") or ""),
                "selector_json": _json_cell(op.get("selector")),
                "before_json": _json_cell(op.get("before")),
                "after_json": _json_cell(op.get("after") or op.get("after_patch")),
                "changed_fields": "|".join(_list(op.get("changed_fields"))) or "|".join(
                    sorted(_mapping(op.get("after_patch")))
                ),
                "blockers": "|".join(_list(op.get("blockers"))),
                "sql_shape": str(op.get("write_sql_shape_if_later_approved") or ""),
            }
        )
    return rows


def build_repair_plan(
    *,
    gap_review_packet_path: Path,
    runner_review_csv_path: Path,
) -> dict[str, Any]:
    gap_resolved = gap_review_packet_path.expanduser().resolve()
    csv_resolved = runner_review_csv_path.expanduser().resolve()
    packet = _load_json(gap_resolved)
    runner_rows = _load_csv_rows(csv_resolved)
    failures: list[str] = []
    _validate_gap_packet(packet=packet, runner_rows=runner_rows, failures=failures)

    summary = _mapping(packet.get("summary"))
    race_id = str(summary.get("race_id") or "")
    official_rows = [dict(_mapping(row)) for row in _list(packet.get("official_rows"))]
    db_rows = [dict(_mapping(row)) for row in _list(packet.get("db_rows"))]
    metadata_candidate = _metadata_update_candidate(
        packet=packet,
        race_id=race_id,
        official_rows=official_rows,
    )
    dog_updates = _dog_update_candidates(
        race_id=race_id,
        official_rows=official_rows,
        db_rows=db_rows,
        runner_rows=runner_rows,
    )
    missing_inserts = _missing_runner_insert_candidates(
        race_id=race_id,
        official_rows=official_rows,
        db_rows=db_rows,
    )

    field_size_policy_required = bool(metadata_candidate.get("deferred_policy_candidates"))
    metadata_patch_count = 1 if metadata_candidate.get("after_patch") else 0
    source_evidence = {
        "gap_review_packet": str(gap_resolved),
        "runner_review_csv": str(csv_resolved),
        "official_source_url": _source_url(packet),
        "inherited": packet.get("source_evidence"),
    }
    status = STATUS_OK if not failures else STATUS_FAILURES
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "failures": failures,
        "report_only": True,
        "write_ready": False,
        "safe_to_write_now": False,
        "label_write_ready": False,
        "label_write_approved": False,
        "approval_required_before_db_write": True,
        "approval_required_before_label_write": True,
        "approval_required_before_dataset_regeneration": True,
        "source_evidence": source_evidence,
        "summary": {
            "race_id": race_id,
            "official_runner_count": len(official_rows),
            "db_runner_count": len(db_rows),
            "dog_row_update_candidate_count": len(dog_updates),
            "missing_runner_insert_candidate_count": len(missing_inserts),
            "metadata_update_candidate_count": metadata_patch_count,
            "field_size_policy_decision_required": field_size_policy_required,
            "safe_to_write_now_count": 0,
            "direct_label_write_ready": False,
            "can_expand_training_without_approval": False,
            "recommended_next_action": (
                "operator_review_exact_no_write_plan_then_explicitly_approve_or_reject_db_repair"
            ),
        },
        "metadata_update_candidate": metadata_candidate,
        "dog_row_update_candidates": dog_updates,
        "missing_runner_insert_candidates": missing_inserts,
        "post_repair_required_checks": [
            "backup_db_before_any_apply",
            "apply_only_exact_race_id_and_name_allowlist_if_approved",
            "rerun_single_race_gap_review",
            "rerun_official_label_preflight",
            "only_then_decide_whether_label_expansion_is_allowed",
        ],
        "approval_gate": {
            "required_before_any_apply": True,
            "approved_here": False,
            "backup_required_before_apply": True,
            "schema_default_policy_required_for_inserts": bool(missing_inserts),
            "field_size_policy_decision_required": field_size_policy_required,
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": [
            "db_write",
            "label_write",
            "metadata_write",
            "dog_row_insert",
            "field_size_update",
            "dataset_regeneration",
            "model_training_or_promotion",
            "registry_update",
            "enable_tgr",
            "betting_or_ev_action",
        ],
    }


def write_outputs(
    output_dir: Path,
    packet: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> None:
    output_dir = _assert_output_dir_safe(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "single_race_official_repair_plan.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "single_race_official_repair_operations.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OPERATION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_operation_csv_rows(packet))
    _write_report(output_dir / "SUMMARY.md", packet)


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    source = _mapping(packet.get("source_evidence"))
    lines = [
        "# Single Race Official Repair Plan",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Race: `{summary.get('race_id')}`",
        f"- Official source URL: `{source.get('official_source_url')}`",
        f"- Official runners: `{summary.get('official_runner_count')}`",
        f"- DB runners: `{summary.get('db_runner_count')}`",
        f"- Existing dog update candidates: `{summary.get('dog_row_update_candidate_count')}`",
        f"- Missing dog insert candidates: `{summary.get('missing_runner_insert_candidate_count')}`",
        f"- Metadata update candidates: `{summary.get('metadata_update_candidate_count')}`",
        f"- Field-size policy decision required: `{summary.get('field_size_policy_decision_required')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Gate",
        "",
        "Any apply step still requires explicit approval, a DB backup, an exact row allowlist, and a separate post-repair label preflight. Missing-runner inserts also require a schema/default policy before execution.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-review-packet", required=True)
    parser.add_argument("--runner-review-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None, *, root: Path | None = None) -> int:
    active = [name for name in FORBIDDEN_APPROVAL_ENV_VARS if str(os.environ.get(name) or "").strip()]
    if active:
        raise SystemExit("refusing report-only repair plan while approval flags are set:" + ",".join(active))
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_repair_plan(
        gap_review_packet_path=Path(args.gap_review_packet),
        runner_review_csv_path=Path(args.runner_review_csv),
    )
    write_outputs(Path(args.output_dir), packet, root=root)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
