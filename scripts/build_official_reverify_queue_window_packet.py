#!/usr/bin/env python3
"""Build a report-only official reverify queue for a manual-review window.

This helper slices the already generated manual-verification candidate queue so
later ranks can be looked up without changing the default first-50 evaluation
packet. It writes artifacts only; it does not fetch official results, write
labels, mutate databases, train models, update registries, enable TGR, or
produce betting/EV actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "expanded_historical_official_reverify_queue_window_packet_v1"
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL = [
    "write_official_safe_labels",
    "mutate_db",
    "regenerate_canonical_dataset",
    "promote_model",
    "update_registry",
    "enable_tgr",
    "betting_or_ev_action",
]
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "promotion": False,
    "betting_decision": False,
}
MANUAL_VERIFICATION_BATCH_CSV_FIELDS = [
    "identity_key",
    "race_date",
    "venue",
    "race_number",
    "consensus_sub_bucket",
    "consensus_sub_bucket_tags",
    "target_distance",
    "winner_name",
    "winner_key",
    "runner_count",
    "source_count",
    "matching_source_count",
    "matching_source_roles",
    "all_source_roles",
    "full_finish_signature",
    "full_finish_signature_agreement_status",
    "distance_values",
    "distance_agreement_status",
    "source_agreement_status",
    "selected_source_role",
    "selected_source_path",
    "source_paths",
    "projected_strict_protocol_train_if_approved",
    "manual_verification_required",
    "policy_key",
    "policy_rank",
    "target_distance_bucket",
]
MANUAL_VERIFICATION_PACKET_CSV_FIELDS = [
    *MANUAL_VERIFICATION_BATCH_CSV_FIELDS,
    "manual_packet_policy_key",
    "manual_packet_rank",
    "verification_status",
    "required_action",
    "approval_write_allowed",
    "official_result_checked",
    "finish_order_checked",
    "distance_checked",
    "source_paths_checked",
    "manual_review_flags",
    "identity_resolution_status",
    "identity_resolution_hints",
    "selected_source_race_id",
    "selected_source_initial_bucket",
    "selected_source_initial_reasons",
    "selected_metadata_race_id",
    "selected_metadata_grade",
    "selected_metadata_results_status",
    "selected_metadata_data_source",
    "source_observation_ids",
    "reviewer_decision",
    "reviewer_notes",
]
OFFICIAL_REVERIFY_QUEUE_CSV_FIELDS = [
    "legacy_race_id",
    "identity_key",
    "manual_packet_policy_key",
    "manual_packet_rank",
    "queue_key",
    "lookup_status",
    "blockers",
    "label_safety_precheck_reasons",
    "next_action",
    "race_date",
    "venue",
    "race_number",
    "target_distance",
    "selected_metadata_grade",
    "winner_name",
    "winner_key",
    "legacy_runner_rows",
    "lookup_key",
    "partial_lookup_key",
    "selected_source_path",
    "source_paths",
]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def manual_verification_csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "consensus_sub_bucket_tags": "|".join(
                str(item)
                for item in row.get("consensus_sub_bucket_tags") or []
            ),
            "distance_values": "|".join(str(item) for item in row.get("distance_values") or []),
            "matching_source_roles": "|".join(
                str(item)
                for item in row.get("matching_source_roles") or []
            ),
            "all_source_roles": "|".join(str(item) for item in row.get("all_source_roles") or []),
            "source_paths": "|".join(str(item) for item in row.get("source_paths") or []),
            "manual_review_flags": "|".join(str(item) for item in row.get("manual_review_flags") or []),
            "identity_resolution_hints": "|".join(str(item) for item in row.get("identity_resolution_hints") or []),
            "selected_source_initial_reasons": "|".join(
                str(item)
                for item in row.get("selected_source_initial_reasons") or []
            ),
            "source_observation_ids": "|".join(str(item) for item in row.get("source_observation_ids") or []),
        }
        for row in rows
    ]


def official_reverify_queue_csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "blockers": "|".join(str(item) for item in row.get("blockers") or []),
            "label_safety_precheck_reasons": "|".join(
                str(item) for item in row.get("label_safety_precheck_reasons") or []
            ),
            "lookup_key": json.dumps(row.get("lookup_key"), sort_keys=True)
            if row.get("lookup_key") is not None
            else "",
            "partial_lookup_key": json.dumps(row.get("partial_lookup_key"), sort_keys=True),
            "source_paths": "|".join(str(item) for item in row.get("source_paths") or []),
        }
        for row in rows
    ]


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def manual_verification_row_race_date(row: Mapping[str, Any]) -> str | None:
    text = str(row.get("race_date") or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text).date().isoformat()
        except ValueError:
            return text
    identity_parts = str(row.get("identity_key") or "").split("|")
    if identity_parts:
        candidate = identity_parts[0]
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except ValueError:
            return None
    return None


def manual_verification_distance_bucket(row: Mapping[str, Any]) -> str:
    distance = safe_float(row.get("target_distance"))
    if distance is None:
        return "missing"
    if distance < 450:
        return "sprint"
    if distance < 650:
        return "middle"
    return "staying"


def manual_verification_holdout_context(
    shared_holdout_protocol: Mapping[str, Any] | None,
) -> dict[str, Any]:
    protocol = shared_holdout_protocol or {}
    calibration_ids = set(protocol.get("calibration_holdout_race_ids") or [])
    second_ids = set(protocol.get("second_holdout_race_ids") or [])
    return {
        "protocol": protocol,
        "calibration_ids": calibration_ids,
        "second_ids": second_ids,
        "holdout_ids": calibration_ids | second_ids,
        "holdout_min_date": str(protocol.get("holdout_min_date") or ""),
    }


def projected_strict_protocol_train_for_manual_row(
    row: Mapping[str, Any],
    *,
    holdout_min_date: str,
    holdout_ids: set[str],
) -> bool:
    identity_key = str(row.get("identity_key") or "")
    date_text = manual_verification_row_race_date(row)
    return bool(
        identity_key
        and date_text
        and holdout_min_date
        and identity_key not in holdout_ids
        and date_text < holdout_min_date
    )


def enrich_manual_verification_batch_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_key: str | None = None,
    holdout_min_date: str,
    holdout_ids: set[str],
) -> list[dict[str, Any]]:
    batch = []
    for index, row in enumerate(rows, start=1):
        enriched = dict(row)
        if policy_key is not None:
            enriched["policy_key"] = policy_key
            enriched["policy_rank"] = index
        enriched["target_distance_bucket"] = manual_verification_distance_bucket(row)
        enriched["projected_strict_protocol_train_if_approved"] = (
            projected_strict_protocol_train_for_manual_row(
                row,
                holdout_min_date=holdout_min_date,
                holdout_ids=holdout_ids,
            )
        )
        batch.append(enriched)
    return batch


def manual_verification_queue_projection(
    rows: Sequence[Mapping[str, Any]],
    policy_projection: Mapping[str, Any],
) -> dict[str, Any]:
    current_official = safe_int(policy_projection.get("current_official_safe_races"))
    current_strict = safe_int(policy_projection.get("current_strict_protocol_official_train_races"))
    strict_rows = [
        row
        for row in rows
        if row.get("projected_strict_protocol_train_if_approved") is True
    ]
    strict_identity_keys = [
        str(row.get("identity_key"))
        for row in strict_rows
        if row.get("identity_key") not in (None, "")
    ]
    strict_dates = sorted(
        {
            str(row.get("race_date"))
            for row in strict_rows
            if row.get("race_date") not in (None, "")
        }
    )
    projection = {
        "status": "PASS" if rows else "NO_QUEUE_ROWS",
        "projection_not_approval": True,
        "approval_required_before_label_write": True,
        "label_write_approved": False,
        "label_writes_performed": False,
        "added_official_safe_races": len(rows),
        "added_strict_protocol_train_races": len(strict_rows),
        "current_official_safe_races": current_official,
        "current_strict_protocol_official_train_races": current_strict,
        "second_holdout_untouched": policy_projection.get("second_holdout_untouched"),
        "excluded_holdout_race_ids": policy_projection.get("excluded_holdout_race_ids") or [],
        "holdout_min_date": policy_projection.get("holdout_min_date"),
        "projected_strict_protocol_train_dates": strict_dates,
        "projected_strict_protocol_train_identity_keys": strict_identity_keys,
    }
    if current_official is not None:
        projection["projected_official_safe_races"] = current_official + len(rows)
    if current_strict is not None:
        projection["projected_strict_protocol_official_train_races"] = current_strict + len(strict_rows)
    return projection


def identity_resolution_for_packet_row(row: Mapping[str, Any], flags: Sequence[str]) -> dict[str, Any]:
    hints = []
    selected_source_race_id = row.get("selected_source_race_id")
    selected_metadata_race_id = row.get("selected_metadata_race_id")
    selected_metadata_grade = row.get("selected_metadata_grade")
    selected_metadata_results_status = row.get("selected_metadata_results_status")
    selected_metadata_data_source = row.get("selected_metadata_data_source")
    selected_source_initial_bucket = row.get("selected_source_initial_bucket")
    selected_source_initial_reasons = list(row.get("selected_source_initial_reasons") or [])
    if selected_source_race_id:
        hints.append(f"selected_source_race_id:{selected_source_race_id}")
    if selected_metadata_race_id and selected_metadata_race_id != selected_source_race_id:
        hints.append(f"selected_metadata_race_id:{selected_metadata_race_id}")
    if selected_metadata_grade:
        hints.append(f"selected_metadata_grade:{selected_metadata_grade}")
    if selected_metadata_results_status:
        hints.append(f"selected_metadata_results_status:{selected_metadata_results_status}")
    if selected_metadata_data_source:
        hints.append(f"selected_metadata_data_source:{selected_metadata_data_source}")
    if selected_source_initial_bucket:
        hints.append(f"selected_source_initial_bucket:{selected_source_initial_bucket}")
    for reason in selected_source_initial_reasons:
        hints.append(f"selected_source_initial_reason:{reason}")
    if "ambiguous_identity_key" in flags or "missing_race_number" in flags:
        hints.extend(
            [
                "lookup_official_race_number_for_date_venue_distance_grade",
                "confirm_official_result_identity_before_any_label_write",
                "do_not_promote_or_write_from_correlated_db_consensus_alone",
            ]
        )
        status = "NEEDS_OFFICIAL_RACE_NUMBER_LOOKUP"
    elif flags:
        hints.append("resolve_manual_review_flags_before_any_label_write")
        status = "NEEDS_MANUAL_FLAG_REVIEW"
    else:
        hints.append("confirm_canonical_identity_key_matches_official_result")
        status = "CANONICAL_IDENTITY_READY_FOR_MANUAL_CONFIRMATION"
    return {
        "identity_resolution_status": status,
        "identity_resolution_hints": hints,
    }


def manual_packet_rows_for_policy(policy_key: str, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(policy.get("manual_verification_batch") or [], start=1):
        item = dict(row)
        flags = []
        identity_key = str(item.get("identity_key") or "")
        if identity_key.startswith("AMBIGUOUS|"):
            flags.append("ambiguous_identity_key")
        if safe_int(item.get("race_number")) is None:
            flags.append("missing_race_number")
        if item.get("source_agreement_status") != "PASS":
            flags.append("source_agreement_not_pass")
        if item.get("distance_agreement_status") != "PASS":
            flags.append("distance_agreement_not_pass")
        if item.get("full_finish_signature_agreement_status") != "PASS":
            flags.append("full_finish_signature_agreement_not_pass")
        if item.get("projected_strict_protocol_train_if_approved") is not True:
            flags.append("not_projected_strict_protocol_train")
        if not item.get("selected_source_path"):
            flags.append("missing_selected_source_path")
        if not item.get("source_paths"):
            flags.append("missing_source_paths")
        if item.get("target_distance") in (None, ""):
            flags.append("missing_target_distance")
        if not item.get("winner_key"):
            flags.append("missing_winner_key")
        if safe_int(item.get("runner_count")) is None:
            flags.append("missing_runner_count")
        resolution = identity_resolution_for_packet_row(item, flags)
        item["manual_packet_policy_key"] = policy_key
        item["manual_packet_rank"] = index
        item["verification_status"] = (
            "PENDING_MANUAL_OFFICIAL_RESULT_REVIEW_WITH_FLAGS"
            if flags
            else "PENDING_MANUAL_OFFICIAL_RESULT_REVIEW"
        )
        item["required_action"] = "verify_official_result_distance_and_full_finish_order_before_any_label_write"
        item["approval_write_allowed"] = False
        item["official_result_checked"] = False
        item["finish_order_checked"] = False
        item["distance_checked"] = False
        item["source_paths_checked"] = False
        item["manual_review_flags"] = flags
        item["identity_resolution_status"] = resolution["identity_resolution_status"]
        item["identity_resolution_hints"] = resolution["identity_resolution_hints"]
        item["reviewer_decision"] = "PENDING"
        item["reviewer_notes"] = ""
        item["verification_checklist"] = [
            "open_selected_source_path_or_official_source",
            "confirm_race_date_venue_race_number",
            "confirm_target_distance",
            "confirm_winner_key_and_winner_name",
            "confirm_full_finish_signature_for_all_runners",
            "confirm_no_holdout_or_second_holdout_overlap",
            "record_reviewer_decision_before_any_write",
        ]
        return_required = item.get("manual_verification_required")
        if return_required is not True:
            item["verification_status"] = "NEEDS_MANUAL_VERIFICATION_FLAG_REVIEW"
            if "manual_verification_required_not_true" not in flags:
                flags.append("manual_verification_required_not_true")
        rows.append(item)
    return rows


def manual_verification_queue_summary(
    *,
    queue_key: str,
    rows: Sequence[Mapping[str, Any]],
    policy_key: str,
    csv_path: str,
    policy_projection: Mapping[str, Any],
    ready_status: str,
    empty_status: str,
    required_next_action: str,
    approval_request_possible_after_manual_review: bool,
) -> dict[str, Any]:
    flag_counts: Counter[str] = Counter(
        flag
        for row in rows
        for flag in row.get("manual_review_flags", [])
    )
    identity_resolution_status_counts: Counter[str] = Counter(
        str(row.get("identity_resolution_status") or "UNKNOWN")
        for row in rows
    )
    return {
        "queue_key": queue_key,
        "schema_version": "expanded_historical_official_label_manual_verification_subqueue_v1",
        "status": ready_status if rows else empty_status,
        "report_only": True,
        "label_writes_performed": False,
        "label_write_approved": False,
        "approval_write_allowed": False,
        "approval_required_before_label_write": True,
        "approval_request_possible_after_manual_review": (
            approval_request_possible_after_manual_review and bool(rows)
        ),
        "selected_policy_key": policy_key,
        "candidate_count": len(rows),
        "strict_protocol_train_candidate_count": sum(
            1
            for row in rows
            if row.get("projected_strict_protocol_train_if_approved") is True
        ),
        "rows_with_manual_review_flags": sum(
            1
            for row in rows
            if row.get("manual_review_flags")
        ),
        "manual_review_flag_counts": dict(sorted(flag_counts.items())),
        "identity_resolution_status_counts": dict(sorted(identity_resolution_status_counts.items())),
        "csv": csv_path,
        "packet_rows": list(rows),
        "projected_if_queue_reviewed_and_explicitly_approved": manual_verification_queue_projection(
            rows,
            policy_projection,
        ),
        "required_next_action": required_next_action,
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
    }


def official_label_manual_verification_subpackets(
    manual_verification_packet: Mapping[str, Any],
) -> dict[str, Any]:
    policy_key = str(manual_verification_packet.get("selected_policy_key") or "unselected")
    packet_rows = list(manual_verification_packet.get("packet_rows") or [])
    policy_projection = manual_verification_packet.get("policy_projection_if_approved") or {}
    canonical_rows = [
        row
        for row in packet_rows
        if row.get("identity_resolution_status") == "CANONICAL_IDENTITY_READY_FOR_MANUAL_CONFIRMATION"
    ]
    lookup_rows = [
        row
        for row in packet_rows
        if row.get("identity_resolution_status") == "NEEDS_OFFICIAL_RACE_NUMBER_LOOKUP"
    ]
    other_rows = [
        row
        for row in packet_rows
        if row.get("identity_resolution_status")
        not in {
            "CANONICAL_IDENTITY_READY_FOR_MANUAL_CONFIRMATION",
            "NEEDS_OFFICIAL_RACE_NUMBER_LOOKUP",
        }
    ]
    canonical_csv = f"official_label_manual_verification_canonical_ready_{policy_key}.csv"
    lookup_csv = f"official_label_manual_verification_identity_lookup_{policy_key}.csv"
    other_csv = f"official_label_manual_verification_other_flags_{policy_key}.csv"
    queues = {
        "canonical_identity_ready": manual_verification_queue_summary(
            queue_key="canonical_identity_ready",
            rows=canonical_rows,
            policy_key=policy_key,
            csv_path=canonical_csv,
            policy_projection=policy_projection,
            ready_status="READY_FOR_MANUAL_OFFICIAL_RESULT_REVIEW",
            empty_status="NO_CANONICAL_IDENTITY_ROWS",
            required_next_action=(
                "manual_review_against_official_result_then_request_explicit_approval_before_label_write"
            ),
            approval_request_possible_after_manual_review=True,
        ),
        "identity_lookup_required": manual_verification_queue_summary(
            queue_key="identity_lookup_required",
            rows=lookup_rows,
            policy_key=policy_key,
            csv_path=lookup_csv,
            policy_projection=policy_projection,
            ready_status="NEEDS_OFFICIAL_RACE_NUMBER_LOOKUP",
            empty_status="NO_IDENTITY_LOOKUP_ROWS",
            required_next_action=(
                "resolve_official_race_number_before_manual_label_review_or_approval_request"
            ),
            approval_request_possible_after_manual_review=False,
        ),
        "other_manual_flags": manual_verification_queue_summary(
            queue_key="other_manual_flags",
            rows=other_rows,
            policy_key=policy_key,
            csv_path=other_csv,
            policy_projection=policy_projection,
            ready_status="NEEDS_MANUAL_FLAG_REVIEW",
            empty_status="NO_OTHER_MANUAL_FLAG_ROWS",
            required_next_action="resolve_manual_review_flags_before_approval_request",
            approval_request_possible_after_manual_review=False,
        ),
    }
    return {
        "schema_version": "expanded_historical_official_label_manual_verification_subpackets_v1",
        "status": "READY_FOR_QUEUE_REVIEW" if packet_rows else "NO_PACKET_ROWS_AVAILABLE",
        "report_only": True,
        "label_writes_performed": False,
        "label_write_approved": False,
        "approval_required_before_label_write": True,
        "approval_required_before_db_write": True,
        "approval_required_before_dataset_regeneration": True,
        "model_promotion_allowed": False,
        "selected_policy_key": policy_key,
        "source_packet_status": manual_verification_packet.get("status"),
        "source_packet_csv": manual_verification_packet.get("packet_csv"),
        "total_packet_rows": len(packet_rows),
        "queue_counts": {
            key: queue["candidate_count"]
            for key, queue in queues.items()
        },
        "recommended_review_order": [
            "canonical_identity_ready",
            "identity_lookup_required",
            "other_manual_flags",
        ],
        "queues": queues,
        "recommended_next_action": (
            "review_canonical_identity_ready_queue_first; lookup_required_queue_needs_official_race_number_resolution"
        ),
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
    }


def safe_file_token(value: Any) -> str:
    text = str(value or "unselected").strip()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return token or "unselected"


def official_reverify_queue_row_for_manual_packet_row(
    *,
    row: Mapping[str, Any],
    queue_key: str,
) -> dict[str, Any]:
    race_number = safe_int(row.get("race_number"))
    race_date_value = str(row.get("race_date") or "").strip()
    venue = str(row.get("venue") or "").strip().upper().replace(" ", "_")
    identity_status = str(row.get("identity_resolution_status") or "UNKNOWN")
    runner_count = safe_int(row.get("runner_count"))
    blockers: list[str] = []
    label_safety_precheck_reasons: list[str] = []
    if identity_status != "CANONICAL_IDENTITY_READY_FOR_MANUAL_CONFIRMATION":
        blockers.append(f"identity_resolution_{identity_status.lower()}")
    if not race_date_value:
        blockers.append("race_date_missing")
    if not venue:
        blockers.append("venue_missing")
    if race_number is None:
        blockers.append("race_number_missing")
    if runner_count is None:
        label_safety_precheck_reasons.append("legacy_runner_count_missing")

    lookup_key = (
        {"venue": venue, "race_number": race_number, "race_date": race_date_value}
        if not blockers
        else None
    )
    partial_lookup_key = {
        "venue": venue or None,
        "race_date": race_date_value or None,
        "race_number": race_number,
        "target_distance": row.get("target_distance"),
        "selected_metadata_grade": row.get("selected_metadata_grade"),
        "selected_metadata_race_id": row.get("selected_metadata_race_id"),
        "selected_source_race_id": row.get("selected_source_race_id"),
    }
    lookup_status = "PARSE_READY" if lookup_key else "PARSE_BLOCKED"
    legacy_race_id = (
        row.get("selected_metadata_race_id")
        or row.get("selected_source_race_id")
        or row.get("identity_key")
    )
    return {
        "schema_version": "expanded_historical_official_reverify_candidate_v1",
        "legacy_race_id": legacy_race_id,
        "legacy_runner_rows": runner_count,
        "legacy_source": "expanded_historical_manual_verification_packet",
        "lookup_status": lookup_status,
        "lookup_key": lookup_key,
        "partial_lookup_key": partial_lookup_key,
        "blockers": sorted(set(blockers)),
        "next_action": (
            "official_result_dry_run_lookup"
            if lookup_status == "PARSE_READY"
            else "manual_identifier_mapping_required_before_official_lookup"
        ),
        "identity_key": row.get("identity_key"),
        "manual_packet_policy_key": row.get("manual_packet_policy_key"),
        "manual_packet_rank": row.get("manual_packet_rank"),
        "queue_key": queue_key,
        "race_date": race_date_value or None,
        "venue": venue or None,
        "race_number": race_number,
        "target_distance": row.get("target_distance"),
        "selected_metadata_grade": row.get("selected_metadata_grade"),
        "winner_name": row.get("winner_name"),
        "winner_key": row.get("winner_key"),
        "selected_source_path": row.get("selected_source_path"),
        "source_paths": list(row.get("source_paths") or []),
        "manual_review_flags": list(row.get("manual_review_flags") or []),
        "identity_resolution_status": identity_status,
        "label_safety_precheck_reasons": label_safety_precheck_reasons,
        "writes_performed": dict(WRITES_PERFORMED),
    }


def official_reverify_queue_for_manual_subpackets(
    manual_verification_subpackets: Mapping[str, Any],
) -> dict[str, Any]:
    selected_policy_key = str(manual_verification_subpackets.get("selected_policy_key") or "unselected")
    policy_token = safe_file_token(selected_policy_key)
    queues = manual_verification_subpackets.get("queues") or {}
    review_order = list(manual_verification_subpackets.get("recommended_review_order") or queues.keys())
    rows: list[dict[str, Any]] = []
    for queue_key in review_order:
        queue = queues.get(queue_key) if isinstance(queues, Mapping) else None
        if not isinstance(queue, Mapping):
            continue
        for packet_row in queue.get("packet_rows") or []:
            if isinstance(packet_row, Mapping):
                rows.append(
                    official_reverify_queue_row_for_manual_packet_row(
                        row=packet_row,
                        queue_key=str(queue_key),
                    )
                )

    lookup_status_counts: Counter[str] = Counter(
        str(row.get("lookup_status") or "DATA_MISSING")
        for row in rows
    )
    blocker_counts: Counter[str] = Counter(
        blocker
        for row in rows
        for blocker in row.get("blockers", [])
    )
    queue_counts: Counter[str] = Counter(str(row.get("queue_key") or "DATA_MISSING") for row in rows)
    queue_jsonl = f"official_label_manual_verification_reverify_queue_{policy_token}.jsonl"
    queue_csv = f"official_label_manual_verification_reverify_queue_{policy_token}.csv"
    return {
        "schema_version": "expanded_historical_official_reverify_queue_report_v1",
        "status": "REPORT_ONLY_REVERIFY_QUEUE_READY" if rows else "NO_REVERIFY_QUEUE_ROWS",
        "report_only": True,
        "selected_policy_key": selected_policy_key,
        "source_subpacket_status": manual_verification_subpackets.get("status"),
        "queue_jsonl": queue_jsonl,
        "queue_csv": queue_csv,
        "candidate_count": len(rows),
        "parse_ready_count": lookup_status_counts.get("PARSE_READY", 0),
        "parse_blocked_count": lookup_status_counts.get("PARSE_BLOCKED", 0),
        "lookup_status_counts": dict(sorted(lookup_status_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "queue_counts": dict(sorted(queue_counts.items())),
        "writes_performed": dict(WRITES_PERFORMED),
        "queue_rows": rows,
        "recommended_next_actions": [
            "Run scripts/run_official_reverify_lookup_dry_run.py on PARSE_READY rows only as a dry run.",
            "Resolve PARSE_BLOCKED rows by official race number before any official lookup attempt.",
            "Do not write labels without complete official positions, identity reconciliation, and explicit approval.",
        ],
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root is not an object")
    return payload


def _safe_output_path(path: Path, *, root: Path) -> Path:
    root = root.expanduser().resolve()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{resolved}") from exc
    if not str(relative).startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _date_desc_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    date_text = manual_verification_row_race_date(row) or ""
    try:
        date_order = -datetime.fromisoformat(date_text).toordinal()
    except ValueError:
        date_order = 0
    return (
        date_order,
        str(row.get("identity_key") or ""),
        str(row.get("winner_name") or ""),
    )


def _ordered_rows(
    *,
    official_expansion_report: Mapping[str, Any],
    shared_holdout_protocol: Mapping[str, Any],
    selection_policy: str,
) -> list[Mapping[str, Any]]:
    context = manual_verification_holdout_context(shared_holdout_protocol)
    holdout_min_date = context["holdout_min_date"]
    holdout_ids = context["holdout_ids"]
    queue = list(official_expansion_report.get("distance_known_manual_verification_queue") or [])
    strict_rows = [
        row
        for row in queue
        if row.get("source_agreement_status") == "PASS"
        and projected_strict_protocol_train_for_manual_row(
            row,
            holdout_min_date=holdout_min_date,
            holdout_ids=holdout_ids,
        )
    ]
    if selection_policy == "latest_source_agreement_strict_train":
        return sorted(strict_rows, key=_date_desc_key)
    if selection_policy == "source_agreement_strict_train":
        return strict_rows
    if selection_policy == "identity_order":
        return queue
    raise ValueError(f"unsupported selection_policy: {selection_policy}")


def _count_by(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter(str(row.get(field) or "UNKNOWN") for row in rows)
    return dict(sorted(counts.items()))


def _window_policy_key(selection_policy: str, start_rank: int, limit: int) -> str:
    end_rank = start_rank + limit - 1
    return f"{selection_policy}_ranks_{start_rank}_{end_rank}"


def build_window_packet(
    *,
    evaluation_dir: Path,
    output_dir: Path,
    selection_policy: str = "latest_source_agreement_strict_train",
    start_rank: int = 51,
    limit: int = 50,
) -> dict[str, Any]:
    if start_rank < 1:
        raise ValueError("start_rank must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    official_expansion_report = _load_json(evaluation_dir / "official_label_expansion_report.json")
    temporal_report = _load_json(evaluation_dir / "temporal_holdout_report.json")
    shared_holdout_protocol = temporal_report.get("shared_holdout_protocol") or {}
    context = manual_verification_holdout_context(shared_holdout_protocol)
    holdout_min_date = context["holdout_min_date"]
    holdout_ids = context["holdout_ids"]

    ordered = _ordered_rows(
        official_expansion_report=official_expansion_report,
        shared_holdout_protocol=shared_holdout_protocol,
        selection_policy=selection_policy,
    )
    end_rank = start_rank + limit - 1
    selected_raw = ordered[start_rank - 1 : end_rank]
    policy_key = _window_policy_key(selection_policy, start_rank, limit)
    enriched = enrich_manual_verification_batch_rows(
        selected_raw,
        policy_key=policy_key,
        holdout_min_date=holdout_min_date,
        holdout_ids=holdout_ids,
    )
    for offset, row in enumerate(enriched):
        row["policy_rank"] = start_rank + offset
        row["window_rank"] = offset + 1

    current_official = official_expansion_report.get("current_official_safe_races")
    current_strict = official_expansion_report.get("strict_protocol_official_train_races")
    policy_projection = {
        "status": "PASS" if enriched else "NO_QUEUE_ROWS",
        "projection_not_approval": True,
        "approval_required_before_label_write": True,
        "label_write_approved": False,
        "label_writes_performed": False,
        "current_official_safe_races": current_official,
        "current_strict_protocol_official_train_races": current_strict,
        "second_holdout_untouched": True,
        "excluded_holdout_race_ids": [],
        "holdout_min_date": holdout_min_date or None,
    }
    policy_projection.update(
        manual_verification_queue_projection(
            enriched,
            policy_projection,
        )
    )

    packet_rows = manual_packet_rows_for_policy(policy_key, {"manual_verification_batch": enriched})
    policy_token = safe_file_token(policy_key)
    packet = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY_FOR_MANUAL_VERIFICATION" if packet_rows else "NO_CANDIDATES_AVAILABLE",
        "report_only": True,
        "label_writes_performed": False,
        "approval_required_before_label_write": True,
        "approval_required_before_db_write": True,
        "approval_required_before_dataset_regeneration": True,
        "model_promotion_allowed": False,
        "label_write_approved": False,
        "selected_policy_key": policy_key,
        "selection_basis": {
            "selection_policy": selection_policy,
            "start_rank": start_rank,
            "end_rank": end_rank,
            "eligible_candidates_for_policy": len(ordered),
            "source_evaluation_dir": str(evaluation_dir),
            "selection_note": (
                "windowed_manual_review_queue_for_official_lookup_only; "
                "does_not_approve_label_write_or_model_promotion"
            ),
        },
        "policy_projection_if_approved": policy_projection,
        "candidate_counts": {
            "selected_candidates": len(packet_rows),
            "eligible_candidates_for_policy": len(ordered),
            "rows_with_manual_review_flags": sum(
                1 for row in packet_rows if row.get("manual_review_flags")
            ),
        },
        "distribution": {
            "venue": _count_by(enriched, "venue"),
            "target_distance_bucket": _count_by(enriched, "target_distance_bucket"),
            "consensus_sub_bucket": _count_by(enriched, "consensus_sub_bucket"),
            "source_agreement_status": _count_by(enriched, "source_agreement_status"),
            "projected_strict_protocol_train_if_approved": _count_by(
                enriched,
                "projected_strict_protocol_train_if_approved",
            ),
        },
        "packet_csv": f"official_label_manual_verification_packet_{policy_token}.csv",
        "packet_rows": packet_rows,
        "required_manual_checks": [
            "verify_race_date_venue_race_number_against_official_result",
            "verify_distance_against_official_result",
            "verify_winner_key_and_winner_name",
            "verify_full_finish_signature_for_all_runners",
            "confirm_selected_source_path_and_source_paths_are_reviewed",
            "record_reviewer_decision_for_each_candidate",
        ],
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "recommended_next_action": (
            "run_official_reverify_lookup_dry_run_on_parse_ready_rows_only; "
            "do_not_write_labels_without_explicit_approval"
        ),
    }

    subpackets = official_label_manual_verification_subpackets(packet)
    reverify_queue = official_reverify_queue_for_manual_subpackets(subpackets)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": packet["status"],
        "report_only": True,
        "selection_policy": selection_policy,
        "start_rank": start_rank,
        "end_rank": end_rank,
        "output_dir": str(output_dir),
        "candidate_count": len(packet_rows),
        "eligible_candidates_for_policy": len(ordered),
        "queue_counts": subpackets.get("queue_counts") or {},
        "parse_ready_count": reverify_queue.get("parse_ready_count", 0),
        "parse_blocked_count": reverify_queue.get("parse_blocked_count", 0),
        "packet_file": "official_label_manual_verification_window_packet.json",
        "subpackets_file": "official_label_manual_verification_window_subpackets.json",
        "reverify_queue_report_file": "official_label_manual_verification_window_reverify_queue_report.json",
        "reverify_queue_jsonl": reverify_queue.get("queue_jsonl"),
        "reverify_queue_csv": reverify_queue.get("queue_csv"),
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
    }
    return {
        "summary": summary,
        "manual_verification_packet": packet,
        "manual_verification_subpackets": subpackets,
        "reverify_queue_report": reverify_queue,
    }


def write_window_outputs(output_dir: Path, packet_bundle: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = packet_bundle["manual_verification_packet"]
    subpackets = packet_bundle["manual_verification_subpackets"]
    reverify_queue = packet_bundle["reverify_queue_report"]

    write_json(output_dir / "official_label_manual_verification_window_summary.json", packet_bundle["summary"])
    write_json(output_dir / "official_label_manual_verification_window_packet.json", packet)
    write_json(output_dir / "official_label_manual_verification_window_subpackets.json", subpackets)
    write_json(
        output_dir / "official_label_manual_verification_window_reverify_queue_report.json",
        reverify_queue,
    )
    if packet.get("packet_csv"):
        write_csv_rows(
            output_dir / str(packet["packet_csv"]),
            manual_verification_csv_rows(packet.get("packet_rows", [])),
            MANUAL_VERIFICATION_PACKET_CSV_FIELDS,
        )
    for queue in (subpackets.get("queues") or {}).values():
        queue_csv = queue.get("csv")
        if queue_csv:
            write_csv_rows(
                output_dir / str(queue_csv),
                manual_verification_csv_rows(queue.get("packet_rows", [])),
                MANUAL_VERIFICATION_PACKET_CSV_FIELDS,
            )
    reverify_rows = reverify_queue.get("queue_rows", [])
    reverify_jsonl = reverify_queue.get("queue_jsonl")
    if reverify_jsonl:
        write_jsonl(output_dir / str(reverify_jsonl), reverify_rows)
    reverify_csv = reverify_queue.get("queue_csv")
    if reverify_csv:
        write_csv_rows(
            output_dir / str(reverify_csv),
            official_reverify_queue_csv_rows(reverify_rows),
            OFFICIAL_REVERIFY_QUEUE_CSV_FIELDS,
        )
    write_csv_rows(
        output_dir / "official_label_manual_verification_window_batch.csv",
        manual_verification_csv_rows(
            packet.get("packet_rows", []),
        ),
        MANUAL_VERIFICATION_BATCH_CSV_FIELDS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--selection-policy",
        default="latest_source_agreement_strict_train",
        choices=[
            "latest_source_agreement_strict_train",
            "source_agreement_strict_train",
            "identity_order",
        ],
    )
    parser.add_argument("--start-rank", default=51, type=int)
    parser.add_argument("--limit", default=50, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output_dir = _safe_output_path(args.output_dir, root=root)
    bundle = build_window_packet(
        evaluation_dir=evaluation_dir,
        output_dir=output_dir,
        selection_policy=args.selection_policy,
        start_rank=args.start_rank,
        limit=args.limit,
    )
    write_window_outputs(output_dir, bundle)
    print(json.dumps(bundle["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
