#!/usr/bin/env python3
"""Build a no-write closeout packet for a copied-DB label-write rehearsal.

This packet proves the official label-write command has been rehearsed against
an isolated SQLite copy while the live writable DB remains result-free. It does
not write labels, does not fetch results, does not mutate model/config/registry
state, and does not bless a live label write without the exact live approval
gate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "label_write_rehearsal_packet_v1"
READY_STATUS = "READY_FOR_LIVE_LABEL_WRITE_APPROVAL"
LIVE_PREFLIGHT_SCHEMA = "label_write_preflight_packet_v1"
RESULT_REPORT_SCHEMA = "official_result_ingest_report_v1"
READINESS_SCHEMA = "result_label_write_readiness_validation_v1"
MODEL_REVIEW_SCHEMA = "model_review_packet_v1"
OFFICIAL_SOURCE = "thedogs_official"
REQUIRED_LABEL_GATE = "APPROVE_RESULT_LABEL_WRITE"

WRITES_PERFORMED = {
    "live_db_result_label_write": False,
    "copied_db_result_label_write": True,
    "model_artifact_write": False,
    "registry_mutation": False,
    "production_config_write": False,
    "refresh_signal_write": False,
    "retrain": False,
    "betting": False,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _append_failure(condition: bool, failures: list[str], name: str) -> None:
    if condition:
        failures.append(name)


def _normalise_path(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    return str(Path(str(path)).expanduser().resolve())


def _same_path(left: Any, right: Path) -> bool:
    return _normalise_path(left) == _normalise_path(right)


def _race_ids(scope: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(race_id).strip()
            for race_id in _list(scope.get("race_ids"))
            if str(race_id).strip()
        }
    )


def _scope_summary(scope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "date": scope.get("date"),
        "db_path": _normalise_path(scope.get("db_path")),
        "race_ids": _race_ids(scope),
        "require_ready_snapshot": bool(scope.get("require_ready_snapshot")),
        "snapshot_dir": _normalise_path(scope.get("snapshot_dir")),
        "upcoming_dir": _normalise_path(scope.get("upcoming_dir")),
    }


def _open_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db_path.expanduser().resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _db_state(db_path: Path, race_ids: list[str], failures: list[str], label: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "db_path": _normalise_path(db_path),
        "quick_check": None,
        "race_ids": race_ids,
        "dog_race_data_rows_by_race": {},
        "race_metadata_by_race": {},
    }
    if not db_path.exists():
        failures.append(f"{label}_db_missing")
        return state
    try:
        with _open_ro(db_path) as conn:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            state["quick_check"] = quick_check[0] if quick_check else None
            _append_failure(state["quick_check"] != "ok", failures, f"{label}_db_quick_check_failed")
            for table in ("dog_race_data", "race_metadata"):
                if not _table_exists(conn, table):
                    failures.append(f"{label}_db_table_missing:{table}")
                    return state
            for race_id in race_ids:
                count = conn.execute(
                    "SELECT COUNT(*) FROM dog_race_data WHERE race_id = ?",
                    (race_id,),
                ).fetchone()[0]
                state["dog_race_data_rows_by_race"][race_id] = int(count)
                row = conn.execute(
                    """
                    SELECT results_status, winner_name, winner_source
                    FROM race_metadata
                    WHERE race_id = ?
                    """,
                    (race_id,),
                ).fetchone()
                state["race_metadata_by_race"][race_id] = dict(row) if row else None
    except sqlite3.Error as exc:
        failures.append(f"{label}_db_read_failed:{type(exc).__name__}")
    return state


def _assert_result_free(
    state: Mapping[str, Any],
    failures: list[str],
    *,
    label: str,
) -> None:
    for race_id, count in _mapping(state.get("dog_race_data_rows_by_race")).items():
        if _safe_int(count) != 0:
            failures.append(f"{label}_db_has_result_rows:{race_id}")
    for race_id, metadata in _mapping(state.get("race_metadata_by_race")).items():
        metadata_map = _mapping(metadata)
        if not metadata_map:
            failures.append(f"{label}_race_metadata_missing:{race_id}")
            continue
        if metadata_map.get("results_status") != "pending":
            failures.append(f"{label}_race_metadata_not_pending:{race_id}")
        if metadata_map.get("winner_name") not in (None, ""):
            failures.append(f"{label}_race_metadata_winner_present:{race_id}")
        if metadata_map.get("winner_source") not in (None, ""):
            failures.append(f"{label}_race_metadata_winner_source_present:{race_id}")


def _read_labeled_rows(db_path: Path, race_id: str, failures: list[str]) -> list[dict[str, Any]]:
    if not db_path.exists():
        failures.append("copied_db_missing_for_labeled_rows")
        return []
    try:
        with _open_ro(db_path) as conn:
            rows = conn.execute(
                """
                SELECT box_number, dog_name, finish_position, placing,
                       scraped_finish_position, data_source
                FROM dog_race_data
                WHERE race_id = ?
                ORDER BY CAST(box_number AS INTEGER)
                """,
                (race_id,),
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        failures.append(f"copied_labeled_rows_read_failed:{type(exc).__name__}")
        return []


def _validate_live_preflight(
    preflight: Mapping[str, Any],
    *,
    live_db_path: Path,
    failures: list[str],
) -> dict[str, Any]:
    approval = _mapping(preflight.get("approval_gate"))
    writes = _mapping(preflight.get("writes_performed"))
    pre_write_state = _mapping(preflight.get("pre_write_db_state"))
    race_ids = [str(item) for item in _list(pre_write_state.get("race_ids"))]
    _append_failure(
        preflight.get("schema_version") != LIVE_PREFLIGHT_SCHEMA,
        failures,
        "live_preflight_schema_mismatch",
    )
    _append_failure(
        preflight.get("status") != "READY_FOR_EXPLICIT_LABEL_WRITE_APPROVAL",
        failures,
        "live_preflight_not_ready",
    )
    _append_failure(
        approval.get("approved") is not False,
        failures,
        "live_label_write_already_approved",
    )
    _append_failure(
        approval.get("required_env_var") != REQUIRED_LABEL_GATE,
        failures,
        "live_preflight_required_gate_mismatch",
    )
    _append_failure(
        writes.get("result_label_write") is not False,
        failures,
        "live_preflight_result_label_write_performed",
    )
    _append_failure(
        not _same_path(pre_write_state.get("db_path"), live_db_path),
        failures,
        "live_preflight_db_path_mismatch",
    )
    _append_failure(
        pre_write_state.get("result_free_before_write") is not True,
        failures,
        "live_preflight_not_result_free",
    )
    return {
        "status": preflight.get("status"),
        "approval_approved": approval.get("approved"),
        "required_env_var": approval.get("required_env_var"),
        "race_ids": race_ids,
        "result_free_before_write": pre_write_state.get("result_free_before_write"),
    }


def _validate_copy_reports(
    *,
    dry_run: Mapping[str, Any],
    readiness: Mapping[str, Any],
    write_report: Mapping[str, Any],
    copied_db_path: Path,
    backup_db_path: Path,
    failures: list[str],
) -> dict[str, Any]:
    dry_scope = _mapping(dry_run.get("scope"))
    readiness_scope = _mapping(readiness.get("scope"))
    write_scope = _mapping(write_report.get("scope"))
    dry_approval = _mapping(dry_run.get("result_label_write_approval"))
    readiness_approval = _mapping(readiness.get("result_label_write_approval"))
    write_approval = _mapping(write_report.get("result_label_write_approval"))
    ingested = [item for item in _list(write_report.get("ingested")) if isinstance(item, Mapping)]

    _append_failure(dry_run.get("schema_version") != RESULT_REPORT_SCHEMA, failures, "copy_dry_run_schema_mismatch")
    _append_failure(dry_run.get("status") != "SUCCESS", failures, "copy_dry_run_not_success")
    _append_failure(dry_run.get("dry_run") is not True, failures, "copy_dry_run_not_dry_run")
    _append_failure(dry_run.get("clean_for_label_write") is not True, failures, "copy_dry_run_not_clean")
    _append_failure(dry_approval.get("approved") is not False, failures, "copy_dry_run_approval_present")

    _append_failure(readiness.get("schema_version") != READINESS_SCHEMA, failures, "copy_readiness_schema_mismatch")
    _append_failure(readiness.get("status") != "READY_FOR_EXPLICIT_APPROVAL", failures, "copy_readiness_not_ready")
    _append_failure(readiness.get("write_performed") is not False, failures, "copy_readiness_write_performed")
    _append_failure(readiness_approval.get("approved") is not False, failures, "copy_readiness_approval_present")

    _append_failure(write_report.get("schema_version") != RESULT_REPORT_SCHEMA, failures, "copy_write_schema_mismatch")
    _append_failure(write_report.get("status") != "SUCCESS", failures, "copy_write_not_success")
    _append_failure(write_report.get("dry_run") is not False, failures, "copy_write_still_dry_run")
    _append_failure(_safe_int(write_report.get("failed_count")) != 0, failures, "copy_write_failed_count_nonzero")
    _append_failure(_safe_int(write_report.get("ingested_count")) <= 0, failures, "copy_write_no_ingested_results")
    _append_failure(write_approval.get("approved") is not True, failures, "copy_write_not_approved")
    _append_failure(not _same_path(write_report.get("backup_path"), backup_db_path), failures, "copy_write_backup_path_mismatch")
    _append_failure(not backup_db_path.exists(), failures, "copy_write_backup_missing")

    for label, scope in (
        ("copy_dry_run", dry_scope),
        ("copy_readiness", readiness_scope),
        ("copy_write", write_scope),
    ):
        _append_failure(not _same_path(scope.get("db_path"), copied_db_path), failures, f"{label}_db_path_mismatch")

    race_ids = _race_ids(write_scope)
    for item in ingested:
        race_id = item.get("race_id") or "unknown"
        _append_failure(item.get("source") != OFFICIAL_SOURCE, failures, f"copy_write_not_official:{race_id}")
        _append_failure(item.get("status") != "resulted", failures, f"copy_write_not_resulted:{race_id}")
        _append_failure(not item.get("winner_name"), failures, f"copy_write_missing_winner:{race_id}")
        _append_failure(not isinstance(item.get("box_order"), list) or not item.get("box_order"), failures, f"copy_write_missing_box_order:{race_id}")

    return {
        "dry_run_status": dry_run.get("status"),
        "readiness_status": readiness.get("status"),
        "write_status": write_report.get("status"),
        "write_approved": write_approval.get("approved"),
        "backup_path": _normalise_path(write_report.get("backup_path")),
        "race_ids": race_ids,
        "ingested": [
            {
                "race_id": item.get("race_id"),
                "source": item.get("source"),
                "status": item.get("status"),
                "winner_name": item.get("winner_name"),
                "box_order": item.get("box_order"),
            }
            for item in ingested
        ],
    }


def _validate_copied_rows(
    *,
    copied_db_path: Path,
    copy_write_summary: Mapping[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"rows_by_race": {}, "metadata_by_race": {}}
    for item in _list(copy_write_summary.get("ingested")):
        if not isinstance(item, Mapping):
            continue
        race_id = str(item.get("race_id") or "")
        if not race_id:
            continue
        box_order = [int(box) for box in _list(item.get("box_order"))]
        rows = _read_labeled_rows(copied_db_path, race_id, failures)
        result["rows_by_race"][race_id] = rows
        if len(rows) != len(box_order):
            failures.append(f"copied_db_row_count_mismatch:{race_id}")
        expected_positions = {box: index + 1 for index, box in enumerate(box_order)}
        for row in rows:
            box_number = _safe_int(row.get("box_number"))
            expected_position = expected_positions.get(box_number)
            if expected_position is None:
                failures.append(f"copied_db_unexpected_box:{race_id}:{box_number}")
                continue
            if _safe_int(row.get("finish_position")) != expected_position:
                failures.append(f"copied_db_finish_position_mismatch:{race_id}:{box_number}")
            if _safe_int(row.get("placing")) != expected_position:
                failures.append(f"copied_db_placing_mismatch:{race_id}:{box_number}")
            if str(row.get("scraped_finish_position") or "") != str(expected_position):
                failures.append(f"copied_db_scraped_finish_position_mismatch:{race_id}:{box_number}")
            if row.get("data_source") != OFFICIAL_SOURCE:
                failures.append(f"copied_db_data_source_mismatch:{race_id}:{box_number}")
    return result


def _evaluation_summary(report: Mapping[str, Any], failures: list[str]) -> dict[str, Any]:
    diagnosis = _mapping(report.get("model_quality_diagnosis"))
    retrain_gate = _mapping(diagnosis.get("retrain_gate"))
    _append_failure(report.get("status") != "SUCCESS", failures, "copy_evaluation_not_success")
    _append_failure(retrain_gate.get("action_taken") not in (None, "none"), failures, "copy_evaluation_retrain_action_taken")
    return {
        "status": report.get("status"),
        "runner_rows_scored": report.get("runner_rows_scored"),
        "clean_official_races_evaluated": diagnosis.get("clean_official_races_evaluated"),
        "clean_official_runner_rows_evaluated": diagnosis.get("clean_official_runner_rows_evaluated"),
        "clean_official_snapshot_instances_evaluated": diagnosis.get("clean_official_snapshot_instances_evaluated"),
        "retrain_gate": dict(retrain_gate),
    }


def _model_review_summary(packet: Mapping[str, Any], failures: list[str]) -> dict[str, Any]:
    review_gate = _mapping(packet.get("review_gate"))
    promotion_control = _mapping(packet.get("promotion_control"))
    _append_failure(packet.get("schema_version") != MODEL_REVIEW_SCHEMA, failures, "copy_model_review_schema_mismatch")
    _append_failure(promotion_control.get("promotion_allowed") is not False, failures, "copy_model_review_promotion_not_blocked")
    _append_failure(promotion_control.get("registry_mutation_allowed") is not False, failures, "copy_model_review_registry_not_blocked")
    _append_failure(promotion_control.get("action_taken") not in (None, "none"), failures, "copy_model_review_action_taken")
    return {
        "status": packet.get("status"),
        "failures": _list(packet.get("failures")),
        "clean_official_evaluated_races": review_gate.get("clean_official_evaluated_races"),
        "clean_official_runner_rows": review_gate.get("clean_official_runner_rows"),
        "clean_official_snapshot_instances": review_gate.get("clean_official_snapshot_instances"),
        "promotion_allowed": promotion_control.get("promotion_allowed"),
        "registry_mutation_allowed": promotion_control.get("registry_mutation_allowed"),
    }


def build_packet(
    *,
    live_db_path: Path,
    copied_db_path: Path,
    backup_db_path: Path,
    live_preflight_path: Path,
    copy_dry_run_report_path: Path,
    copy_readiness_path: Path,
    copy_write_report_path: Path,
    copy_evaluation_report_path: Path,
    copy_model_review_packet_path: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    live_preflight = _load_json(live_preflight_path)
    copy_dry_run = _load_json(copy_dry_run_report_path)
    copy_readiness = _load_json(copy_readiness_path)
    copy_write_report = _load_json(copy_write_report_path)
    copy_evaluation = _load_json(copy_evaluation_report_path)
    copy_model_review = _load_json(copy_model_review_packet_path)

    live_summary = _validate_live_preflight(
        live_preflight,
        live_db_path=live_db_path,
        failures=failures,
    )
    copy_summary = _validate_copy_reports(
        dry_run=copy_dry_run,
        readiness=copy_readiness,
        write_report=copy_write_report,
        copied_db_path=copied_db_path,
        backup_db_path=backup_db_path,
        failures=failures,
    )
    race_ids = sorted({*live_summary.get("race_ids", []), *copy_summary.get("race_ids", [])})
    live_state = _db_state(live_db_path, race_ids, failures, "live")
    backup_state = _db_state(backup_db_path, race_ids, failures, "backup")
    copied_state = _db_state(copied_db_path, race_ids, failures, "copied")
    _assert_result_free(live_state, failures, label="live")
    _assert_result_free(backup_state, failures, label="backup")
    copied_rows = _validate_copied_rows(
        copied_db_path=copied_db_path,
        copy_write_summary=copy_summary,
        failures=failures,
    )
    evaluation = _evaluation_summary(copy_evaluation, failures)
    model_review = _model_review_summary(copy_model_review, failures)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": READY_STATUS if not failures else "NOT_READY",
        "failures": failures,
        "source_evidence": {
            "live_db": _normalise_path(live_db_path),
            "copied_db": _normalise_path(copied_db_path),
            "copy_pre_op_backup": _normalise_path(backup_db_path),
            "live_preflight_packet": _normalise_path(live_preflight_path),
            "copy_dry_run_report": _normalise_path(copy_dry_run_report_path),
            "copy_label_write_readiness": _normalise_path(copy_readiness_path),
            "copy_label_write_report": _normalise_path(copy_write_report_path),
            "copy_evaluation_report": _normalise_path(copy_evaluation_report_path),
            "copy_model_review_packet": _normalise_path(copy_model_review_packet_path),
        },
        "approval_gate": {
            "approved": False,
            "required": True,
            "required_env_var": REQUIRED_LABEL_GATE,
            "required_cli_flag": "--write-labels-approved",
            "scope": "live_db_result_label_write",
        },
        "live_preflight_summary": live_summary,
        "copied_db_rehearsal_summary": copy_summary,
        "db_state": {
            "live": live_state,
            "copied": copied_state,
            "copy_pre_op_backup": backup_state,
            "copied_labeled_rows": copied_rows,
        },
        "copy_evaluation_summary": evaluation,
        "copy_model_review_summary": model_review,
        "writes_performed": dict(WRITES_PERFORMED),
        "next_safe_step": "await_exact_APPROVE_RESULT_LABEL_WRITE_before_live_db_label_write",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-db", required=True)
    parser.add_argument("--copied-db", required=True)
    parser.add_argument("--copy-pre-op-backup", required=True)
    parser.add_argument("--live-preflight-packet", required=True)
    parser.add_argument("--copy-dry-run-report", required=True)
    parser.add_argument("--copy-label-write-readiness", required=True)
    parser.add_argument("--copy-label-write-report", required=True)
    parser.add_argument("--copy-evaluation-report", required=True)
    parser.add_argument("--copy-model-review-packet", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packet = build_packet(
        live_db_path=Path(args.live_db),
        copied_db_path=Path(args.copied_db),
        backup_db_path=Path(args.copy_pre_op_backup),
        live_preflight_path=Path(args.live_preflight_packet),
        copy_dry_run_report_path=Path(args.copy_dry_run_report),
        copy_readiness_path=Path(args.copy_label_write_readiness),
        copy_write_report_path=Path(args.copy_label_write_report),
        copy_evaluation_report_path=Path(args.copy_evaluation_report),
        copy_model_review_packet_path=Path(args.copy_model_review_packet),
    )
    text = json.dumps(packet, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if packet.get("status") == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
