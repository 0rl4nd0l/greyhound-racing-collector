#!/usr/bin/env python3
"""Build a report-only no-box actual-win rehearsal packet.

This script does not train models, write labels, mutate databases, update
registries, enable TGR, or create betting/EV actions. It materializes only the
data-contract rows that are safe under the winner-only policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "winner_only_no_box_actual_win_rehearsal_v1"
ROWS_SCHEMA_VERSION = "winner_only_no_box_actual_win_rows_v1"
RECOVERY_QUEUE_SCHEMA_VERSION = "winner_only_no_box_recovery_queue_v1"
ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"

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
    "rekey_race_ids",
    "strict_full_finish_training",
    "top3_or_finish_order_training",
    "box_feature_training",
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

SAMPLE_SIZE_THRESHOLDS = {
    "minimum_smoke_actual_win_eval": 20,
    "minimum_rolling_temporal_eval": 50,
    "minimum_ranking_model_comparison": 100,
}

COMPLETE_FIELD_CANDIDATE_STATUS = "winner_only_no_box_research_candidate_metadata_confirmed"
PARTIAL_FIELD_CANDIDATE_STATUS = (
    "partial_field_winner_only_no_box_research_candidate_metadata_confirmed"
)
PARTIAL_FIELD_ALLOWED_SKIP_REASONS = {
    "official_positions_incomplete_for_legacy_runner_count",
    "official_terminal_statuses_present",
}
PARTIAL_FIELD_ALLOWED_SCOPES = {
    "partial_db_name_subset_of_official_finishers",
    "partial_db_name_subset_after_nonstarter_terminal_exclusions",
}
COMPLETE_FIELD_TERMINAL_EXCLUSION_SCOPE = (
    "complete_name_set_after_nonstarter_terminal_exclusions"
)

CSV_FIELDS = [
    "race_id",
    "legacy_race_id",
    "identity_key",
    "race_date",
    "venue",
    "race_number",
    "dog_name_key",
    "dog_name",
    "actual_win",
    "candidate_kind",
    "field_scope",
    "field_complete_for_ranking",
    "race_grouped_actual_win_ranking_allowed",
    "target_source",
    "label_scope",
    "box_features_allowed",
    "finish_order_labels_allowed",
    "top3_labels_allowed",
    "official_safe_label_candidate",
    "label_write_approved",
]

RECOVERY_QUEUE_FIELDS = [
    "priority",
    "recovery_lane",
    "race_id",
    "legacy_race_id",
    "identity_key",
    "status",
    "winner_alignment_summary",
    "name_set_result",
    "lookup_skip_reasons",
    "primary_bucket",
    "official_rows",
    "db_rows",
    "next_report_only_action",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _clean_display_name(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", text)
    return text.strip()


def _name_key(value: Any) -> str:
    text = _clean_display_name(value).lower()
    text = text.replace("'", "").replace('"', "").replace("`", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _race_keys(record: Mapping[str, Any]) -> list[str]:
    keys = []
    for value in (record.get("race_id"), record.get("legacy_race_id")):
        text = str(value or "").strip()
        if text and text not in keys:
            keys.append(text)
    return keys


def _index_identity_records(identity_packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for record in _list(identity_packet.get("records")):
        mapped = _mapping(record)
        for key in _race_keys(mapped):
            index.setdefault(key, mapped)
    return index


def _candidate_blockers(record: Mapping[str, Any]) -> list[str]:
    blockers = []
    winner_alignment = _mapping(record.get("winner_alignment"))
    candidate_kind = _candidate_kind(record)
    if candidate_kind is None:
        blockers.append("not_winner_only_no_box_research_candidate")

    lookup_skip_reasons = [str(reason) for reason in _list(record.get("lookup_skip_reasons"))]
    name_set_result = str(record.get("name_set_result") or "")
    if candidate_kind == "complete_field":
        subset_scope = _partial_field_subset_scope(record)
        db_only_count = _safe_int(subset_scope.get("db_only_name_count")) or 0
        terminal_exclusion_count = (
            _safe_int(subset_scope.get("db_only_terminal_exclusion_count")) or 0
        )
        duplicate_db_name_keys = _list(subset_scope.get("duplicate_db_name_keys"))
        duplicate_official_name_keys = _list(subset_scope.get("duplicate_official_name_keys"))
        terminal_complete_scope_ready = (
            str(record.get("field_scope") or "") == COMPLETE_FIELD_TERMINAL_EXCLUSION_SCOPE
            and record.get(
                "terminal_exclusion_complete_field_winner_only_no_box_research_candidate"
            )
            is True
            and subset_scope.get("db_name_set_complete_after_terminal_exclusions") is True
            and db_only_count > 0
            and terminal_exclusion_count == db_only_count
            and (_safe_int(subset_scope.get("official_only_name_count")) or 0) == 0
            and not duplicate_db_name_keys
            and not duplicate_official_name_keys
            and bool(lookup_skip_reasons)
            and set(lookup_skip_reasons).issubset(PARTIAL_FIELD_ALLOWED_SKIP_REASONS)
            and name_set_result == "mismatch"
        )
        if terminal_complete_scope_ready:
            if record.get("full_name_identity_ready") is True:
                blockers.append("terminal_complete_candidate_marked_full_name_identity_ready")
        else:
            if record.get("full_name_identity_ready") is not True:
                blockers.append("full_name_identity_not_ready")
            if name_set_result != "exact_match":
                blockers.append("name_set_not_exact_match")
            if lookup_skip_reasons:
                blockers.append("lookup_skip_reasons_present")
        if record.get("field_complete_for_ranking") is not True:
            blockers.append("complete_field_candidate_not_marked_ranking_ready")
    elif candidate_kind == "partial_field":
        subset_scope = _partial_field_subset_scope(record)
        db_only_count = _safe_int(subset_scope.get("db_only_name_count")) or 0
        terminal_exclusion_count = (
            _safe_int(subset_scope.get("db_only_terminal_exclusion_count")) or 0
        )
        db_subset_scope_ready = subset_scope.get("db_name_subset_of_official") is True
        terminal_exclusion_scope_ready = (
            subset_scope.get("db_name_subset_after_terminal_exclusions") is True
            and db_only_count > 0
            and terminal_exclusion_count == db_only_count
        )
        if record.get("full_name_identity_ready") is True:
            blockers.append("partial_field_candidate_marked_full_name_identity_ready")
        if name_set_result != "mismatch":
            blockers.append("partial_field_name_set_not_mismatch")
        if not lookup_skip_reasons:
            blockers.append("partial_field_lookup_skip_reasons_missing")
        if not set(lookup_skip_reasons).issubset(PARTIAL_FIELD_ALLOWED_SKIP_REASONS):
            blockers.append("partial_field_lookup_skip_reasons_not_allowed")
        if not (db_subset_scope_ready or terminal_exclusion_scope_ready):
            blockers.append("partial_field_db_names_not_subset_of_official")
        if db_only_count and not terminal_exclusion_scope_ready:
            blockers.append("partial_field_db_only_names_present")
        if (_safe_int(subset_scope.get("official_only_name_count")) or 0) <= 0:
            blockers.append("partial_field_official_only_names_missing")
        if str(record.get("field_scope") or "") not in PARTIAL_FIELD_ALLOWED_SCOPES:
            blockers.append("partial_field_scope_missing")
        if record.get("field_complete_for_ranking") is not False:
            blockers.append("partial_field_candidate_marked_ranking_ready")
    else:
        if record.get("full_name_identity_ready") is not True:
            blockers.append("full_name_identity_not_ready")
        if name_set_result != "exact_match":
            blockers.append("name_set_not_exact_match")
        if lookup_skip_reasons:
            blockers.append("lookup_skip_reasons_present")

    if str(record.get("primary_bucket") or "") != "box_identity_drift":
        blockers.append("primary_bucket_not_box_identity_drift")
    if winner_alignment.get("official_winner_matches_db_winner") is not True:
        blockers.append("official_winner_not_confirmed_against_db_winner")
    if winner_alignment.get("official_winner_matches_metadata_winner") is not True:
        blockers.append("official_winner_not_confirmed_against_metadata_winner")
    if record.get("requires_actual_win_only_target") is not True:
        blockers.append("actual_win_only_target_not_required")
    if record.get("requires_no_box_feature_policy") is not True:
        blockers.append("no_box_feature_policy_not_required")
    if record.get("official_safe_label_candidate") is not False:
        blockers.append("official_safe_label_candidate_not_false")
    if record.get("strict_full_finish_label_candidate") is not False:
        blockers.append("strict_full_finish_label_candidate_not_false")
    if record.get("label_write_approved") is not False:
        blockers.append("label_write_approved_not_false")
    if record.get("forbidden_for_top3_or_finish_order_training") is not True:
        blockers.append("top3_or_finish_order_training_not_forbidden")
    if record.get("forbidden_for_box_feature_training") is not True:
        blockers.append("box_feature_training_not_forbidden")
    if candidate_kind == "complete_field" and str(record.get("status") or "") != COMPLETE_FIELD_CANDIDATE_STATUS:
        blockers.append("metadata_confirmed_status_missing")
    if candidate_kind == "partial_field" and str(record.get("status") or "") != PARTIAL_FIELD_CANDIDATE_STATUS:
        blockers.append("partial_field_metadata_confirmed_status_missing")
    if candidate_kind is None and str(record.get("status") or "") != COMPLETE_FIELD_CANDIDATE_STATUS:
        blockers.append("metadata_confirmed_status_missing")
    return blockers


def _eligible_candidates(winner_only_packet: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    candidates = []
    blocked = []
    for record in _list(winner_only_packet.get("records")):
        mapped = _mapping(record)
        blockers = _candidate_blockers(mapped)
        if blockers:
            blocked.append(
                {
                    "race_id": mapped.get("race_id"),
                    "legacy_race_id": mapped.get("legacy_race_id"),
                    "status": mapped.get("status"),
                    "blockers": blockers,
                }
            )
        else:
            candidates.append(mapped)
    return candidates, blocked


def _lookup_key(identity_record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(identity_record.get("lookup_key"))


def _identity_record_for(
    record: Mapping[str, Any],
    identity_by_race: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    for key in _race_keys(record):
        identity_record = _mapping(identity_by_race.get(key))
        if identity_record:
            return identity_record
    return {}


def _winner_alignment_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    winner_alignment = _mapping(record.get("winner_alignment"))
    return {
        "official_winner_name": winner_alignment.get("official_winner_name"),
        "db_winner_name": winner_alignment.get("db_winner_name"),
        "metadata_winner_name": winner_alignment.get("metadata_winner_name"),
        "official_winner_matches_db_winner": (
            winner_alignment.get("official_winner_matches_db_winner") is True
        ),
        "official_winner_matches_metadata_winner": (
            winner_alignment.get("official_winner_matches_metadata_winner") is True
        ),
    }


def _candidate_kind(record: Mapping[str, Any]) -> str | None:
    status = str(record.get("status") or "")
    if (
        record.get("winner_only_no_box_research_candidate") is True
        and status == COMPLETE_FIELD_CANDIDATE_STATUS
    ):
        return "complete_field"
    if (
        record.get("partial_field_winner_only_no_box_research_candidate") is True
        and status == PARTIAL_FIELD_CANDIDATE_STATUS
    ):
        return "partial_field"
    return None


def _partial_field_subset_scope(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("name_set_subset_scope"))


def _recovery_lane_for(record: Mapping[str, Any], identity_record: Mapping[str, Any]) -> dict[str, Any]:
    alignment = _winner_alignment_summary(record)
    lookup_skip_reasons = [str(reason) for reason in _list(record.get("lookup_skip_reasons"))]
    name_set_result = str(record.get("name_set_result") or "")
    primary_bucket = str(record.get("primary_bucket") or "")
    official_db_match = alignment["official_winner_matches_db_winner"] is True
    official_metadata_match = alignment["official_winner_matches_metadata_winner"] is True
    name_set_exact = name_set_result == "exact_match"
    official_rows = len(_list(identity_record.get("official_runner_rows")))
    db_rows = len(_list(identity_record.get("db_runner_rows")))

    if _candidate_kind(record):
        lane = "already_confirmed_winner_only_candidate"
        priority = "P0"
        action = "use_existing_no_box_actual_win_rehearsal_row; do_not_write_labels"
    elif primary_bucket == "duplicate_or_ambiguous_db_identity":
        lane = "blocked_duplicate_or_ambiguous_identity"
        priority = "P4"
        action = "manual_identity_deduplication_review_only"
    elif not official_db_match:
        lane = "blocked_official_db_winner_mismatch"
        priority = "P4"
        action = "manual_official_vs_db_winner_audit_before_any_actual_win_use"
    elif official_metadata_match and not name_set_exact and lookup_skip_reasons:
        lane = "name_set_and_parser_repair_candidate"
        priority = "P1"
        action = "repair_official_name_set_or_terminal_status_parse_then_rerun_winner_only_gate"
    elif name_set_exact and lookup_skip_reasons:
        lane = "parser_terminal_status_repair_candidate"
        priority = "P2" if official_metadata_match else "P3"
        action = "repair_lookup_skip_reason_then_rerun_winner_only_gate"
    elif official_db_match and not official_metadata_match and not name_set_exact:
        lane = "metadata_and_name_set_repair_candidate"
        priority = "P3"
        action = "reconcile_metadata_winner_and_name_set_before_actual_win_use"
    elif official_db_match and not official_metadata_match:
        lane = "metadata_winner_recheck_candidate"
        priority = "P3"
        action = "recheck_metadata_winner_against_official_before_actual_win_use"
    elif official_db_match and not name_set_exact:
        lane = "name_set_repair_candidate"
        priority = "P2"
        action = "reconcile_name_set_before_actual_win_use"
    else:
        lane = "manual_review_required"
        priority = "P5"
        action = "manual_review_only"

    blockers = _candidate_blockers(record)
    return {
        "race_id": record.get("race_id"),
        "legacy_race_id": record.get("legacy_race_id"),
        "identity_key": record.get("identity_key"),
        "status": record.get("status"),
        "priority": priority,
        "recovery_lane": lane,
        "candidate_blockers": blockers,
        "winner_alignment_summary": alignment,
        "name_set_result": name_set_result,
        "full_name_identity_ready": record.get("full_name_identity_ready") is True,
        "lookup_skip_reasons": lookup_skip_reasons,
        "primary_bucket": primary_bucket,
        "official_rows": official_rows,
        "db_rows": db_rows,
        "manual_drift_classification": record.get("manual_drift_classification"),
        "next_report_only_action": action,
        "forbidden_without_explicit_approval": [
            "write_official_safe_labels",
            "mutate_db",
            "strict_full_finish_training",
            "box_feature_training",
            "model_training_or_promotion",
        ],
    }


def build_recovery_queue(
    winner_only_packet: Mapping[str, Any],
    identity_packet: Mapping[str, Any],
) -> dict[str, Any]:
    identity_by_race = _index_identity_records(identity_packet)
    queue = []
    for record in _list(winner_only_packet.get("records")):
        mapped = _mapping(record)
        if _candidate_kind(mapped):
            continue
        identity_record = _identity_record_for(mapped, identity_by_race)
        queue.append(_recovery_lane_for(mapped, identity_record))

    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
    queue.sort(
        key=lambda item: (
            priority_order.get(str(item.get("priority")), 99),
            str(item.get("recovery_lane") or ""),
            str(item.get("race_id") or ""),
        )
    )
    lane_counts = Counter(str(item.get("recovery_lane") or "DATA_MISSING") for item in queue)
    priority_counts = Counter(str(item.get("priority") or "DATA_MISSING") for item in queue)
    return {
        "schema_version": RECOVERY_QUEUE_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "model_training_performed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "summary": {
            "records_reviewed": len(_list(winner_only_packet.get("records"))),
            "recovery_queue_count": len(queue),
            "recovery_lane_counts": dict(sorted(lane_counts.items())),
            "priority_counts": dict(sorted(priority_counts.items())),
            "p1_name_set_and_parser_repair_candidates": lane_counts.get(
                "name_set_and_parser_repair_candidate",
                0,
            ),
            "next_recommended_gate": (
                "repair_official_name_set_or_terminal_status_parse_for_P1_candidates"
                if lane_counts.get("name_set_and_parser_repair_candidate")
                else "continue_manual_recovery_review"
            ),
        },
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "records": queue,
    }


def _materialize_rows_for_candidate(
    candidate: Mapping[str, Any],
    identity_record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    winner_alignment = _mapping(candidate.get("winner_alignment"))
    official_winner_key = _name_key(winner_alignment.get("official_winner_name"))
    metadata_winner_key = _name_key(winner_alignment.get("metadata_winner_name"))
    matches = [_mapping(row) for row in _list(identity_record.get("matches_by_name"))]
    lookup_key = _lookup_key(identity_record)
    candidate_kind = _candidate_kind(candidate) or "unknown"
    field_scope = str(candidate.get("field_scope") or "not_eligible")
    field_complete_for_ranking = candidate.get("field_complete_for_ranking") is True
    rows = []
    for match in sorted(matches, key=lambda item: str(item.get("dog_name_key") or "")):
        dog_key = str(match.get("dog_name_key") or "").strip()
        if not dog_key:
            continue
        official_name = _clean_display_name(match.get("official_dog_name"))
        db_name = _clean_display_name(match.get("db_dog_name"))
        rows.append(
            {
                "race_id": candidate.get("race_id"),
                "legacy_race_id": candidate.get("legacy_race_id"),
                "identity_key": candidate.get("identity_key"),
                "race_date": lookup_key.get("race_date"),
                "venue": lookup_key.get("venue"),
                "race_number": _safe_int(lookup_key.get("race_number")),
                "dog_name_key": dog_key,
                "dog_name": official_name or db_name,
                "actual_win": 1 if dog_key == official_winner_key else 0,
                "candidate_kind": candidate_kind,
                "field_scope": field_scope,
                "field_complete_for_ranking": field_complete_for_ranking,
                "race_grouped_actual_win_ranking_allowed": field_complete_for_ranking,
                "target_source": "official_winner_name_metadata_confirmed",
                "label_scope": "actual_win_only",
                "box_features_allowed": False,
                "finish_order_labels_allowed": False,
                "top3_labels_allowed": False,
                "official_safe_label_candidate": False,
                "label_write_approved": False,
            }
        )

    row_keys = set().union(*(set(row) for row in rows)) if rows else set()
    forbidden_fields_present = sorted(row_keys & FORBIDDEN_ROW_FIELDS)
    positive_count = sum(int(row.get("actual_win") or 0) for row in rows)
    validation_failures = []
    if not rows:
        validation_failures.append("no_rows_materialized")
    dog_keys = [str(row.get("dog_name_key") or "") for row in rows]
    if len(dog_keys) != len(set(dog_keys)):
        validation_failures.append("duplicate_dog_name_keys_materialized")
    if positive_count != 1:
        validation_failures.append("actual_win_positive_count_not_one")
    if official_winner_key != metadata_winner_key:
        validation_failures.append("official_winner_metadata_winner_key_mismatch")
    if official_winner_key and official_winner_key not in {str(row.get("dog_name_key")) for row in rows}:
        validation_failures.append("official_winner_key_not_materialized")
    if forbidden_fields_present:
        validation_failures.append("forbidden_row_fields_present")
    if candidate_kind == "partial_field" and field_complete_for_ranking:
        validation_failures.append("partial_field_candidate_marked_ranking_ready")

    return rows, {
        "race_id": candidate.get("race_id"),
        "legacy_race_id": candidate.get("legacy_race_id"),
        "identity_key": candidate.get("identity_key"),
        "status": "MATERIALIZED_ACTUAL_WIN_ONLY_ROWS" if not validation_failures else "BLOCKED",
        "row_count": len(rows),
        "actual_win_positive_rows": positive_count,
        "actual_win_negative_rows": len(rows) - positive_count,
        "candidate_kind": candidate_kind,
        "field_scope": field_scope,
        "field_complete_for_ranking": field_complete_for_ranking,
        "race_grouped_actual_win_ranking_allowed": field_complete_for_ranking,
        "official_winner_name": winner_alignment.get("official_winner_name"),
        "metadata_winner_name": winner_alignment.get("metadata_winner_name"),
        "db_winner_name": winner_alignment.get("db_winner_name"),
        "forbidden_row_fields_present": forbidden_fields_present,
        "validation_failures": validation_failures,
        "partial_field_official_only_names": _list(
            _partial_field_subset_scope(candidate).get("official_not_in_db")
        ),
        "partial_field_db_only_terminal_exclusions": _list(
            _partial_field_subset_scope(candidate).get("db_only_terminal_exclusions")
        ),
        "full_finish_order_forbidden": True,
        "top3_training_forbidden": True,
        "box_feature_training_forbidden": True,
    }


def _sample_size_gate(actual_win_candidate_count: int, ranking_ready_candidate_count: int) -> dict[str, Any]:
    gates = {}
    for key, minimum in SAMPLE_SIZE_THRESHOLDS.items():
        count_basis = (
            "ranking_ready_complete_field_races"
            if key == "minimum_ranking_model_comparison"
            else "actual_win_no_box_research_races"
        )
        current = ranking_ready_candidate_count if key == "minimum_ranking_model_comparison" else actual_win_candidate_count
        gates[key] = {
            "minimum_confirmed_races": minimum,
            "current_confirmed_races": current,
            "count_basis": count_basis,
            "additional_confirmed_candidates_needed": max(0, minimum - current),
            "status": "PASS" if current >= minimum else "INSUFFICIENT_CONFIRMED_RACES",
        }
    return gates


def build_winner_only_no_box_rehearsal_packet(
    *,
    winner_only_packet: Mapping[str, Any],
    identity_packet: Mapping[str, Any],
    winner_only_packet_path: str | None = None,
    identity_packet_path: str | None = None,
    expected_candidates: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates, blocked = _eligible_candidates(winner_only_packet)
    identity_by_race = _index_identity_records(identity_packet)
    all_rows: list[dict[str, Any]] = []
    race_summaries = []
    failures = []

    for candidate in candidates:
        identity_record = {}
        for key in _race_keys(candidate):
            identity_record = _mapping(identity_by_race.get(key))
            if identity_record:
                break
        if not identity_record:
            failures.append(f"identity_record_missing:{candidate.get('race_id')}")
            race_summaries.append(
                {
                    "race_id": candidate.get("race_id"),
                    "legacy_race_id": candidate.get("legacy_race_id"),
                    "status": "BLOCKED",
                    "validation_failures": ["identity_record_missing"],
                }
            )
            continue
        rows, summary = _materialize_rows_for_candidate(candidate, identity_record)
        all_rows.extend(rows)
        race_summaries.append(summary)
        for failure in summary["validation_failures"]:
            failures.append(f"{failure}:{candidate.get('race_id')}")

    if expected_candidates is not None and len(candidates) != expected_candidates:
        failures.append(f"candidate_count_mismatch:expected_{expected_candidates}:actual_{len(candidates)}")

    status_counts = Counter(str(item.get("status") or "DATA_MISSING") for item in race_summaries)
    candidate_kind_counts = Counter(_candidate_kind(candidate) or "DATA_MISSING" for candidate in candidates)
    candidate_count = len(candidates)
    ranking_ready_candidate_count = sum(
        1 for candidate in candidates if candidate.get("field_complete_for_ranking") is True
    )
    sample_gate = _sample_size_gate(candidate_count, ranking_ready_candidate_count)
    recovery_queue = build_recovery_queue(winner_only_packet, identity_packet)
    recovery_summary = _mapping(recovery_queue.get("summary"))
    if failures:
        status = "REPORT_ONLY_WITH_FAILURES"
    elif candidate_count == 0:
        status = "REPORT_ONLY_NO_CONFIRMED_WINNER_ONLY_CANDIDATES"
    elif sample_gate["minimum_smoke_actual_win_eval"]["status"] != "PASS":
        status = "REPORT_ONLY_SINGLE_CANDIDATE_CONTRACT_REHEARSAL"
    else:
        status = "REPORT_ONLY_READY_FOR_NO_BOX_ACTUAL_WIN_EVALUATION"
    if sample_gate["minimum_smoke_actual_win_eval"]["status"] == "PASS":
        next_recommended_gate = "run_report_only_no_box_actual_win_eval"
    elif recovery_summary.get("p1_name_set_and_parser_repair_candidates"):
        next_recommended_gate = recovery_summary.get("next_recommended_gate")
    else:
        next_recommended_gate = "collect_more_metadata_confirmed_winner_only_candidates_before_model_eval"

    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "failures": failures,
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "model_training_performed": False,
        "model_promotion_allowed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_packets": {
            "winner_only_packet": winner_only_packet_path,
            "identity_packet": identity_packet_path,
        },
        "scope": {
            "allowed_use": "single_candidate_no_box_actual_win_data_contract_rehearsal",
            "not_allowed_use": [
                "official_safe_label_write",
                "strict_full_finish_training",
                "top3_or_finish_order_training",
                "box_feature_training",
                "model_promotion",
                "betting_or_ev_action",
            ],
            "row_schema_version": ROWS_SCHEMA_VERSION,
            "forbidden_row_fields": sorted(FORBIDDEN_ROW_FIELDS),
        },
        "summary": {
            "confirmed_winner_only_candidate_count": candidate_count,
            "complete_field_confirmed_winner_only_candidate_count": candidate_kind_counts.get(
                "complete_field",
                0,
            ),
            "partial_field_confirmed_winner_only_candidate_count": candidate_kind_counts.get(
                "partial_field",
                0,
            ),
            "race_grouped_ranking_ready_candidate_count": ranking_ready_candidate_count,
            "blocked_or_non_candidate_records_seen": len(blocked),
            "materialized_races": len(race_summaries),
            "materialized_rows": len(all_rows),
            "actual_win_positive_rows": sum(int(row.get("actual_win") or 0) for row in all_rows),
            "actual_win_negative_rows": sum(1 for row in all_rows if int(row.get("actual_win") or 0) == 0),
            "race_status_counts": dict(sorted(status_counts.items())),
            "no_box_row_policy_pass": not any(
                set(row).intersection(FORBIDDEN_ROW_FIELDS) for row in all_rows
            ),
            "strict_full_finish_label_candidate_count": 0,
            "official_safe_label_candidate_count": 0,
            "can_evaluate_model": sample_gate["minimum_smoke_actual_win_eval"]["status"] == "PASS",
            "sample_size_gate": sample_gate,
            "recovery_queue_count": recovery_summary.get("recovery_queue_count"),
            "p1_recovery_candidates": recovery_summary.get(
                "p1_name_set_and_parser_repair_candidates"
            ),
            "next_recommended_gate": next_recommended_gate,
            "can_evaluate_race_grouped_model": (
                sample_gate["minimum_ranking_model_comparison"]["status"] == "PASS"
            ),
        },
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "race_summaries": race_summaries,
        "blocked_or_non_candidate_records": blocked[:25],
        "recovery_queue": recovery_queue,
    }
    return packet, all_rows


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


def _recovery_queue_csv_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "priority": record.get("priority"),
                "recovery_lane": record.get("recovery_lane"),
                "race_id": record.get("race_id"),
                "legacy_race_id": record.get("legacy_race_id"),
                "identity_key": record.get("identity_key"),
                "status": record.get("status"),
                "winner_alignment_summary": json.dumps(
                    record.get("winner_alignment_summary") or {},
                    sort_keys=True,
                ),
                "name_set_result": record.get("name_set_result"),
                "lookup_skip_reasons": "|".join(
                    str(reason) for reason in _list(record.get("lookup_skip_reasons"))
                ),
                "primary_bucket": record.get("primary_bucket"),
                "official_rows": record.get("official_rows"),
                "db_rows": record.get("db_rows"),
                "next_report_only_action": record.get("next_report_only_action"),
            }
        )
    return rows


def write_rehearsal_outputs(
    output_dir: Path,
    packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path | None = None,
) -> None:
    output_dir = _assert_output_dir_safe(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "winner_only_no_box_actual_win_rehearsal_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "winner_only_no_box_actual_win_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "winner_only_no_box_actual_win_rows.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    recovery_queue = _mapping(packet.get("recovery_queue"))
    recovery_records = [_mapping(record) for record in _list(recovery_queue.get("records"))]
    (output_dir / "winner_only_no_box_recovery_queue.json").write_text(
        json.dumps(recovery_queue, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "winner_only_no_box_recovery_queue.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RECOVERY_QUEUE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_recovery_queue_csv_rows(recovery_records))
    summary = _mapping(packet.get("summary"))
    recovery_summary = _mapping(recovery_queue.get("summary"))
    (output_dir / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# Winner-Only No-Box Actual-Win Rehearsal",
                "",
                f"- Status: `{packet.get('status')}`",
                f"- Confirmed winner-only candidates: `{summary.get('confirmed_winner_only_candidate_count')}`",
                f"- Complete-field candidates: `{summary.get('complete_field_confirmed_winner_only_candidate_count')}`",
                f"- Partial-field candidates: `{summary.get('partial_field_confirmed_winner_only_candidate_count')}`",
                f"- Race-grouped ranking-ready candidates: `{summary.get('race_grouped_ranking_ready_candidate_count')}`",
                f"- Materialized rows: `{summary.get('materialized_rows')}`",
                f"- Can evaluate model now: `{summary.get('can_evaluate_model')}`",
                f"- Can evaluate race-grouped model now: `{summary.get('can_evaluate_race_grouped_model')}`",
                f"- Recovery queue records: `{recovery_summary.get('recovery_queue_count')}`",
                f"- P1 recovery candidates: `{recovery_summary.get('p1_name_set_and_parser_repair_candidates')}`",
                f"- Next gate: `{summary.get('next_recommended_gate')}`",
                "",
                "No labels, DB rows, registries, model pointers, TGR settings, or betting/EV actions were changed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-only-packet", required=True)
    parser.add_argument("--identity-packet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-candidates", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    winner_only_path = Path(args.winner_only_packet).expanduser().resolve()
    identity_path = Path(args.identity_packet).expanduser().resolve()
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_load_json(winner_only_path),
        identity_packet=_load_json(identity_path),
        winner_only_packet_path=str(winner_only_path),
        identity_packet_path=str(identity_path),
        expected_candidates=args.expected_candidates,
    )
    write_rehearsal_outputs(Path(args.output_dir), packet, rows)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2))
    return 0 if not packet["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
