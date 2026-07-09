#!/usr/bin/env python3
"""Crosswalk rolling no-box ranking errors to source and label-expansion queues.

This helper is report-only. It reads the rolling stratified error analysis and
manual official-label verification packets, optionally enriches the rolling
races from the DB in read-only/query-only mode, and writes packet artifacts
only. It does not fetch official results, write labels, mutate databases,
regenerate datasets, train or persist models, update registries, enable TGR, or
produce betting/EV actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "no_box_pairwise_rolling_source_label_expansion_packet_v1"
STATUS_OK = "REPORT_ONLY_ROLLING_SOURCE_LABEL_EXPANSION_PACKET"
STATUS_FAILURES = "REPORT_ONLY_ROLLING_SOURCE_LABEL_EXPANSION_PACKET_WITH_FAILURES"
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
FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL = [
    "write_official_safe_labels",
    "mutate_db",
    "regenerate_canonical_dataset",
    "train_or_promote_model",
    "update_registry",
    "enable_tgr",
    "betting_or_ev_action",
]
QUEUE_PRIORITY = {
    "canonical_identity_ready": 0,
    "identity_lookup_required": 1,
    "other_manual_flags": 2,
}
CROSSWALK_FIELDS = [
    "race_id",
    "identity_key",
    "race_date",
    "venue",
    "race_number",
    "window_id",
    "winner_rank",
    "top1_hit",
    "top3_hit",
    "field_size",
    "field_scope",
    "field_complete_for_ranking",
    "distance_bucket",
    "winner_box_bucket",
    "source_bucket",
    "db_results_status",
    "db_winner_source",
    "db_race_metadata_present",
    "dog_row_count",
    "dog_data_source_counts",
    "source_gap_status",
    "queue_match_count",
    "best_queue_key",
    "best_queue_policy_key",
    "best_queue_identity_key",
    "best_queue_projected_strict_train_if_approved",
    "best_queue_required_action",
]
QUEUE_SUMMARY_FIELDS = [
    "packet_path",
    "selected_policy_key",
    "queue_key",
    "candidate_count",
    "strict_protocol_train_candidate_count",
    "rows_with_manual_review_flags",
    "approval_required_before_label_write",
    "approval_request_possible_after_manual_review",
    "projected_official_safe_races",
    "projected_strict_protocol_official_train_races",
    "second_holdout_untouched",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


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


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _select_existing_columns(
    conn: sqlite3.Connection,
    *,
    table: str,
    desired: Sequence[str],
    race_ids: Sequence[str],
) -> list[dict[str, Any]]:
    columns = _table_columns(conn, table)
    selected = [column for column in desired if column in columns]
    if "race_id" not in selected:
        selected.insert(0, "race_id")
    if not race_ids or "race_id" not in columns:
        return []
    placeholders = ",".join("?" for _ in race_ids)
    sql = (
        f"SELECT {', '.join(selected)} FROM {table} "
        f"WHERE race_id IN ({placeholders})"
    )
    return [dict(row) for row in conn.execute(sql, list(race_ids))]


def _fetch_db_metadata(
    conn: sqlite3.Connection,
    race_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not race_ids:
        return {}, {"quick_check": None, "race_metadata_rows": 0, "dog_race_data_rows": 0}
    quick_check = conn.execute("PRAGMA quick_check").fetchone()
    race_rows = _select_existing_columns(
        conn,
        table="race_metadata",
        desired=[
            "race_id",
            "race_date",
            "venue",
            "race_number",
            "distance",
            "grade",
            "winner_name",
            "winner_source",
            "results_status",
            "data_source",
        ],
        race_ids=race_ids,
    )
    dog_rows = _select_existing_columns(
        conn,
        table="dog_race_data",
        desired=[
            "race_id",
            "dog_name",
            "box_number",
            "finish_position",
            "placing",
            "scraped_finish_position",
            "data_source",
        ],
        race_ids=race_ids,
    )
    by_race = {str(row.get("race_id")): dict(row) for row in race_rows}
    dog_rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dog_rows:
        dog_rows_by_race[str(row.get("race_id"))].append(dict(row))
    metadata = {}
    for race_id in race_ids:
        race_meta = dict(by_race.get(str(race_id)) or {})
        dogs = dog_rows_by_race.get(str(race_id), [])
        source_counts: Counter[str] = Counter(
            str(row.get("data_source") or "DATA_MISSING")
            for row in dogs
        )
        metadata[str(race_id)] = {
            **race_meta,
            "race_metadata_present": bool(race_meta),
            "dog_rows": dogs,
            "dog_row_count": len(dogs),
            "dog_data_source_counts": dict(sorted(source_counts.items())),
        }
    return metadata, {
        "quick_check": quick_check[0] if quick_check else None,
        "race_metadata_rows": len(race_rows),
        "dog_race_data_rows": len(dog_rows),
    }


def _race_number_from_race_id(race_id: Any) -> int | None:
    match = re.search(r"_(\d+)$", str(race_id or ""))
    return _safe_int(match.group(1)) if match else None


def _identity_key_for_record(record: Mapping[str, Any]) -> str:
    existing = str(record.get("identity_key") or "").strip()
    if existing:
        return existing
    race_date = str(record.get("race_date") or "").strip()
    venue = str(record.get("venue") or "").strip()
    race_number = _safe_int(record.get("race_number")) or _race_number_from_race_id(
        record.get("race_id")
    )
    if race_date and venue and race_number is not None:
        return f"{race_date}|{venue}|R{race_number:02d}"
    return str(record.get("race_id") or "")


def _record_lookup_keys(record: Mapping[str, Any]) -> set[str]:
    keys = {str(record.get("race_id") or ""), _identity_key_for_record(record)}
    return {key for key in keys if key}


def _queue_row_lookup_keys(row: Mapping[str, Any]) -> set[str]:
    keys = {
        str(row.get("identity_key") or ""),
        str(row.get("selected_source_race_id") or ""),
        str(row.get("selected_metadata_race_id") or ""),
    }
    return {key for key in keys if key}


def _validate_report_only_packet(
    *,
    path: Path,
    packet: Mapping[str, Any],
    failures: list[str],
) -> None:
    if packet.get("report_only") is not True:
        failures.append(f"packet_not_report_only:{path}")
    for flag in (
        "label_write_approved",
        "label_writes_performed",
        "model_promotion_allowed",
    ):
        if packet.get(flag) not in (False, None):
            failures.append(f"packet_flag_not_false:{path}:{flag}")
    writes = _mapping(packet.get("writes_performed"))
    for key, value in writes.items():
        if value is True:
            failures.append(f"packet_write_flag_true:{path}:{key}")


def _validate_rolling_analysis(analysis: Mapping[str, Any], failures: list[str]) -> None:
    if analysis.get("report_only") is not True:
        failures.append("rolling_analysis_not_report_only")
    for flag in (
        "label_write_approved",
        "model_training_performed",
        "model_promotion_allowed",
        "write_ready",
    ):
        if analysis.get(flag) not in (False, None):
            failures.append(f"rolling_analysis_flag_not_false:{flag}")
    for key, value in _mapping(analysis.get("writes_performed")).items():
        if value is True:
            failures.append(f"rolling_analysis_write_flag_true:{key}")


def _queue_projection(queue: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(queue.get("projected_if_queue_reviewed_and_explicitly_approved"))


def _queue_rows(
    *,
    packet_path: Path,
    packet: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_policy_key = str(packet.get("selected_policy_key") or "")
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for queue_key, queue in sorted(_mapping(packet.get("queues")).items()):
        queue_map = _mapping(queue)
        projection = _queue_projection(queue_map)
        packet_rows = _list(queue_map.get("packet_rows"))
        summary_rows.append(
            {
                "packet_path": str(packet_path),
                "selected_policy_key": selected_policy_key,
                "queue_key": queue_key,
                "candidate_count": queue_map.get("candidate_count"),
                "strict_protocol_train_candidate_count": queue_map.get(
                    "strict_protocol_train_candidate_count"
                ),
                "rows_with_manual_review_flags": queue_map.get(
                    "rows_with_manual_review_flags"
                ),
                "approval_required_before_label_write": queue_map.get(
                    "approval_required_before_label_write"
                ),
                "approval_request_possible_after_manual_review": queue_map.get(
                    "approval_request_possible_after_manual_review"
                ),
                "current_official_safe_races": projection.get("current_official_safe_races"),
                "current_strict_protocol_official_train_races": projection.get(
                    "current_strict_protocol_official_train_races"
                ),
                "projected_official_safe_races": projection.get(
                    "projected_official_safe_races"
                ),
                "projected_strict_protocol_official_train_races": projection.get(
                    "projected_strict_protocol_official_train_races"
                ),
                "second_holdout_untouched": projection.get("second_holdout_untouched"),
            }
        )
        for row in packet_rows:
            row_map = dict(_mapping(row))
            row_map["_packet_path"] = str(packet_path)
            row_map["_queue_key"] = queue_key
            row_map["_selected_policy_key"] = selected_policy_key
            all_rows.append(row_map)
    return all_rows, summary_rows


def _queue_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for key in _queue_row_lookup_keys(row):
            index[key].append(row)
    return index


def _best_queue_match(matches: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda row: (
            QUEUE_PRIORITY.get(str(row.get("_queue_key") or ""), 99),
            _safe_int(row.get("policy_rank")) or 999999,
            str(row.get("identity_key") or ""),
        ),
    )[0]


def _source_gap_status(
    *,
    record: Mapping[str, Any],
    db_meta: Mapping[str, Any],
) -> str:
    if str(record.get("source_bucket") or "") not in {"", "DATA_MISSING"}:
        return "SOURCE_BUCKET_PRESENT"
    if not db_meta.get("race_metadata_present"):
        return "RACE_METADATA_MISSING"
    if not db_meta.get("winner_source"):
        return "WINNER_SOURCE_MISSING"
    return "SOURCE_BUCKET_MISSING_IN_STRATIFIED_ANALYSIS"


def _json_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True)


def _build_crosswalk_rows(
    *,
    race_records: Sequence[Mapping[str, Any]],
    db_metadata: Mapping[str, Mapping[str, Any]],
    candidate_index: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(
        race_records,
        key=lambda item: (
            item.get("top1_hit") is True,
            str(item.get("race_date") or ""),
            str(item.get("race_id") or ""),
        ),
    ):
        race_id = str(record.get("race_id") or "")
        db_meta = _mapping(db_metadata.get(race_id))
        lookup_keys = _record_lookup_keys(record)
        matches_by_id: dict[int, Mapping[str, Any]] = {}
        for key in lookup_keys:
            for match in candidate_index.get(key, []):
                matches_by_id[id(match)] = match
        matches = list(matches_by_id.values())
        best = _best_queue_match(matches)
        race_number = _safe_int(record.get("race_number")) or _safe_int(
            db_meta.get("race_number")
        ) or _race_number_from_race_id(race_id)
        rows.append(
            {
                "race_id": race_id,
                "identity_key": _identity_key_for_record({**record, "race_number": race_number}),
                "race_date": record.get("race_date") or db_meta.get("race_date"),
                "venue": record.get("venue") or db_meta.get("venue"),
                "race_number": race_number,
                "window_id": record.get("window_id"),
                "winner_rank": record.get("winner_rank"),
                "top1_hit": record.get("top1_hit"),
                "top3_hit": record.get("top3_hit"),
                "field_size": record.get("field_size"),
                "field_scope": record.get("field_scope"),
                "field_complete_for_ranking": record.get("field_complete_for_ranking"),
                "distance_bucket": record.get("distance_bucket"),
                "winner_box_bucket": record.get("winner_box_bucket"),
                "source_bucket": record.get("source_bucket"),
                "db_results_status": db_meta.get("results_status"),
                "db_winner_source": db_meta.get("winner_source"),
                "db_race_metadata_present": db_meta.get("race_metadata_present"),
                "dog_row_count": db_meta.get("dog_row_count"),
                "dog_data_source_counts": _json_cell(db_meta.get("dog_data_source_counts")),
                "source_gap_status": _source_gap_status(record=record, db_meta=db_meta),
                "queue_match_count": len(matches),
                "best_queue_key": best.get("_queue_key") if best else "",
                "best_queue_policy_key": best.get("_selected_policy_key") if best else "",
                "best_queue_identity_key": best.get("identity_key") if best else "",
                "best_queue_projected_strict_train_if_approved": (
                    best.get("projected_strict_protocol_train_if_approved")
                    if best
                    else ""
                ),
                "best_queue_required_action": best.get("required_action") if best else "",
            }
        )
    return rows


def _unique_strict_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    queue_key: str,
) -> dict[str, Mapping[str, Any]]:
    unique = {}
    for row in rows:
        if row.get("_queue_key") != queue_key:
            continue
        if row.get("projected_strict_protocol_train_if_approved") is not True:
            continue
        identity = str(row.get("identity_key") or "")
        if not identity:
            continue
        unique.setdefault(identity, row)
    return unique


def _current_count_from_queue_summaries(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> int | None:
    values = [
        _safe_int(row.get(key))
        for row in rows
        if _safe_int(row.get(key)) is not None
    ]
    if not values:
        return None
    # All current artifacts use one base count. The minimum is conservative if
    # stale packets disagree.
    return min(values)


def build_source_label_packet(
    *,
    rolling_analysis: Mapping[str, Any],
    manual_subpacket_paths: Sequence[Path],
    db_path: Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    _validate_rolling_analysis(rolling_analysis, failures)
    race_records = list(_list(rolling_analysis.get("race_records")))
    race_ids = sorted({str(row.get("race_id") or "") for row in race_records if row.get("race_id")})
    db_summary: dict[str, Any] = {
        "db_path": str(db_path.expanduser().resolve()) if db_path else None,
        "read_only": bool(db_path),
        "query_only": bool(db_path),
        "quick_check": None,
        "race_metadata_rows": 0,
        "dog_race_data_rows": 0,
    }
    db_metadata: dict[str, dict[str, Any]] = {}
    if db_path:
        with _connect_read_only(db_path) as conn:
            db_metadata, db_update = _fetch_db_metadata(conn, race_ids)
            db_summary.update(db_update)
            if db_summary.get("quick_check") != "ok":
                failures.append("db_quick_check_failed")

    candidate_rows: list[dict[str, Any]] = []
    queue_summary_rows: list[dict[str, Any]] = []
    packet_status_counts: Counter[str] = Counter()
    for path in manual_subpacket_paths:
        packet_path = path.expanduser().resolve()
        packet = _load_json(packet_path)
        _validate_report_only_packet(path=packet_path, packet=packet, failures=failures)
        packet_status_counts[str(packet.get("status") or "DATA_MISSING")] += 1
        rows, summaries = _queue_rows(packet_path=packet_path, packet=packet)
        candidate_rows.extend(rows)
        queue_summary_rows.extend(summaries)

    candidate_index = _queue_index(candidate_rows)
    crosswalk_rows = _build_crosswalk_rows(
        race_records=race_records,
        db_metadata=db_metadata,
        candidate_index=candidate_index,
    )
    canonical_strict = _unique_strict_rows(candidate_rows, queue_key="canonical_identity_ready")
    lookup_strict = _unique_strict_rows(candidate_rows, queue_key="identity_lookup_required")
    current_official = _current_count_from_queue_summaries(
        queue_summary_rows,
        "current_official_safe_races",
    )
    current_strict = _current_count_from_queue_summaries(
        queue_summary_rows,
        "current_strict_protocol_official_train_races",
    )

    recommended = sorted(
        queue_summary_rows,
        key=lambda row: (
            -(_safe_int(row.get("strict_protocol_train_candidate_count")) or 0),
            str(row.get("packet_path") or ""),
        ),
    )[0] if queue_summary_rows else {}
    source_gap_counts: Counter[str] = Counter(
        str(row.get("source_gap_status") or "DATA_MISSING")
        for row in crosswalk_rows
    )
    overlap_count = sum(1 for row in crosswalk_rows if _safe_int(row.get("queue_match_count")) or 0)
    miss_overlap_count = sum(
        1
        for row in crosswalk_rows
        if row.get("top1_hit") is not True and (_safe_int(row.get("queue_match_count")) or 0)
    )
    all_second_holdout_values = [
        row.get("second_holdout_untouched")
        for row in queue_summary_rows
        if row.get("second_holdout_untouched") is not None
    ]
    summary = {
        "rolling_race_count": len(crosswalk_rows),
        "rolling_top1_miss_count": sum(1 for row in crosswalk_rows if row.get("top1_hit") is not True),
        "rolling_top3_miss_count": sum(1 for row in crosswalk_rows if row.get("top3_hit") is not True),
        "rolling_source_bucket_missing_count": sum(
            1 for row in crosswalk_rows if row.get("source_bucket") == "DATA_MISSING"
        ),
        "source_gap_status_counts": dict(sorted(source_gap_counts.items())),
        "manual_subpacket_count": len(manual_subpacket_paths),
        "packet_status_counts": dict(sorted(packet_status_counts.items())),
        "manual_queue_candidate_row_count": len(candidate_rows),
        "unique_canonical_ready_strict_train_candidates": len(canonical_strict),
        "unique_identity_lookup_strict_train_candidates": len(lookup_strict),
        "rolling_queue_overlap_count": overlap_count,
        "rolling_top1_miss_queue_overlap_count": miss_overlap_count,
        "recommended_first_review_queue": {
            "packet_path": recommended.get("packet_path"),
            "selected_policy_key": recommended.get("selected_policy_key"),
            "queue_key": recommended.get("queue_key"),
            "strict_protocol_train_candidate_count": recommended.get(
                "strict_protocol_train_candidate_count"
            ),
            "candidate_count": recommended.get("candidate_count"),
        },
        "current_official_safe_races_inferred": current_official,
        "current_strict_protocol_official_train_races_inferred": current_strict,
        "projected_if_all_unique_canonical_ready_rows_reviewed_and_explicitly_approved": {
            "projection_not_approval": True,
            "approval_required_before_label_write": True,
            "label_write_approved": False,
            "label_writes_performed": False,
            "added_strict_protocol_train_races": len(canonical_strict),
            "projected_strict_protocol_official_train_races": (
                current_strict + len(canonical_strict)
                if current_strict is not None
                else None
            ),
        },
        "second_holdout_untouched_across_queue_projections": (
            all(value is True for value in all_second_holdout_values)
            if all_second_holdout_values
            else None
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": STATUS_OK if not failures else STATUS_FAILURES,
        "failures": failures,
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "label_writes_performed": False,
        "approval_required_before_label_write": True,
        "approval_required_before_db_write": True,
        "approval_required_before_dataset_regeneration": True,
        "model_training_performed": False,
        "model_promotion_allowed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "rolling_analysis_status": rolling_analysis.get("status"),
        "rolling_analysis_path": rolling_analysis.get("rolling_report_path"),
        "rolling_predictions_path": rolling_analysis.get("predictions_path"),
        "source_sample_size_status": rolling_analysis.get("source_sample_size_status"),
        "source_reserved_final_races": rolling_analysis.get("source_reserved_final_races"),
        "source_reserved_races_predicted": rolling_analysis.get("source_reserved_races_predicted"),
        "db_enrichment": db_summary,
        "summary": summary,
        "crosswalk_rows": crosswalk_rows,
        "queue_summary_rows": queue_summary_rows,
    }


def write_outputs(
    output_dir: Path,
    packet: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> None:
    output_dir = _assert_output_dir_safe(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        key: value
        for key, value in packet.items()
        if key not in {"crosswalk_rows", "queue_summary_rows"}
    }
    (output_dir / "no_box_pairwise_rolling_source_label_expansion_packet.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "rolling_source_label_crosswalk.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CROSSWALK_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(packet.get("crosswalk_rows") or [])
    with (output_dir / "official_review_queue_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(packet.get("queue_summary_rows") or [])
    summary = _mapping(packet.get("summary"))
    recommended = _mapping(summary.get("recommended_first_review_queue"))
    projection = _mapping(
        summary.get("projected_if_all_unique_canonical_ready_rows_reviewed_and_explicitly_approved")
    )
    lines = [
        "# No-Box Pairwise Rolling Source/Label Expansion Packet",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, or EV actions were changed.",
        "",
        "## Rolling Source Gaps",
        "",
        f"- Rolling races: `{summary.get('rolling_race_count')}`",
        f"- Top1 misses: `{summary.get('rolling_top1_miss_count')}`",
        f"- Source bucket missing: `{summary.get('rolling_source_bucket_missing_count')}`",
        f"- Queue overlaps with rolling races: `{summary.get('rolling_queue_overlap_count')}`",
        "",
        "## Official-Safe Expansion Queue",
        "",
        f"- Manual subpackets: `{summary.get('manual_subpacket_count')}`",
        f"- Unique canonical-ready strict-train candidates: `{summary.get('unique_canonical_ready_strict_train_candidates')}`",
        f"- Unique identity-lookup strict-train candidates: `{summary.get('unique_identity_lookup_strict_train_candidates')}`",
        f"- Recommended first queue: `{recommended.get('selected_policy_key')}` / `{recommended.get('queue_key')}`",
        f"- Projected strict train after all canonical-ready review+explicit approval: `{projection.get('projected_strict_protocol_official_train_races')}`",
        f"- Second holdout untouched across queue projections: `{summary.get('second_holdout_untouched_across_queue_projections')}`",
        "",
        "## Next Safe Action",
        "",
        "Review the recommended canonical-ready queue against official results. Any label or DB write still requires explicit approval, an exact row allowlist, and a pre-op backup.",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-stratified-analysis", required=True)
    parser.add_argument(
        "--manual-subpacket",
        action="append",
        required=True,
        help="Path to official_label_manual_verification*_subpackets.json. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--db")
    return parser


def main(argv: Iterable[str] | None = None, *, root: Path | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    analysis_path = Path(args.rolling_stratified_analysis).expanduser().resolve()
    packet = build_source_label_packet(
        rolling_analysis=_load_json(analysis_path),
        manual_subpacket_paths=[Path(path) for path in args.manual_subpacket],
        db_path=Path(args.db) if args.db else None,
    )
    write_outputs(Path(args.output_dir), packet, root=root)
    print(
        json.dumps(
            {"status": packet["status"], "summary": packet["summary"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
