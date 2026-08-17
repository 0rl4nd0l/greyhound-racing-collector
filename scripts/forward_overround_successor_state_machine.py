#!/usr/bin/env python3
"""Pure replay model for the prepared forward-overround successor protocol.

This module does not collect, score, schedule, or write evidence. It models the
append-only event rules so the fixed-N finalization path can be tested before a
separately authorized implementation exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_STATES = frozenset({"FINALIZED_SCORED", "FINALIZED_ABORTED_NO_METRICS"})
TEMPORARY_REJECTION_REASONS = frozenset(
    {
        "source_unavailable",
        "candidate_incomplete_field",
        "candidate_timing_outside_window",
        "candidate_identity_ambiguous",
        "candidate_win_provenance_ambiguous",
    }
)


class ProtocolViolation(ValueError):
    """Raised for a malformed synthetic journal or invalid transition."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProtocolViolation(f"invalid_sha256:{field}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolViolation(f"invalid_text:{field}")
    return value


def _time(value: Any, field: str) -> datetime:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolViolation(f"invalid_timestamp:{field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolViolation(f"naive_timestamp:{field}")
    return parsed


def load_protocol(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != "forward_overround_successor_protocol_v1":
        raise ProtocolViolation("protocol_schema_mismatch")
    if protocol["authorization"] != {
        "activation_receipt_required": True,
        "collection_authorized": False,
        "scheduler_install_authorized": False,
        "status": "PREPARED_NOT_AUTHORIZED",
    }:
        raise ProtocolViolation("protocol_is_not_prepared_only")
    target = protocol["cohort"].get("target_races")
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        raise ProtocolViolation("invalid_target_races")
    if protocol["evaluation"]["finalization"] != {
        "identical_races_required": True,
        "one_shot": True,
        "requires_approved_results": target,
        "requires_sealed_predictions": target,
    }:
        raise ProtocolViolation("finalization_target_mismatch")
    protocol["_document_sha256"] = sha256_bytes(raw)
    return protocol


def initial_snapshot(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "forward_overround_successor_state_v1",
        "protocol_sha256": protocol["_document_sha256"],
        "state": "PREPARED_NOT_AUTHORIZED",
        "activation_at": None,
        "activation_receipt_sha256": None,
        "active_admission_id": None,
        "active_admission_sha256": None,
        "admissions": {},
        "admission_payloads": {},
        "admission_times": {},
        "predictions": {},
        "race_members": {},
        "candidate_files": {},
        "last_member_order_key": None,
        "results": {},
        "rejections": {},
        "rejection_files": {},
        "event_hashes": {},
        "paused_reason": None,
        "paused_at": None,
        "fatal_reason": None,
        "finalization_requested": False,
        "score_invocation_count": 0,
        "finalization_member_manifest_sha256": None,
        "metrics_receipt_sha256": None,
        "actions": [],
    }


def _fatal(state: dict[str, Any], reason: str) -> None:
    state["fatal_reason"] = reason
    state["paused_reason"] = None
    state["paused_at"] = None
    state["state"] = "FINALIZED_ABORTED_NO_METRICS"
    state["actions"] = ["SEAL_TERMINAL_NO_METRICS"]


def _derive_state(state: dict[str, Any], target: int) -> None:
    if state["state"] in TERMINAL_STATES or state["finalization_requested"]:
        return
    if state["activation_at"] is None:
        state["state"] = "PREPARED_NOT_AUTHORIZED"
    elif state["paused_reason"] is not None:
        state["state"] = "ADMISSION_PAUSED"
    elif len(state["predictions"]) < target:
        state["state"] = "COLLECTING"
    elif len(state["results"]) < target:
        state["state"] = "RESULT_CLOSURE"
    else:
        state["state"] = "READY_TO_FINALIZE"


def _admission_payload(event: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    admission = event.get("admission")
    if not isinstance(admission, Mapping):
        raise ProtocolViolation("missing_admission")
    payload = {
        "admission_id": _text(admission.get("admission_id"), "admission_id"),
        "admitted_at": _time(admission.get("admitted_at"), "admitted_at").isoformat(),
        "capture_code_sha256": _hash(admission.get("capture_code_sha256"), "capture_code_sha256"),
        "capture_unit_sha256": _hash(admission.get("capture_unit_sha256"), "capture_unit_sha256"),
        "finalizer_code_sha256": _hash(
            admission.get("finalizer_code_sha256"), "finalizer_code_sha256"
        ),
        "semantic_contract_sha256": _hash(
            admission.get("semantic_contract_sha256"), "semantic_contract_sha256"
        ),
        "protocol_sha256": _hash(admission.get("protocol_sha256"), "protocol_sha256"),
        "reviewed": admission.get("reviewed"),
        "model_and_config_unchanged": admission.get("model_and_config_unchanged"),
        "protocol_semantics_unchanged": admission.get("protocol_semantics_unchanged"),
        "predecessor_admission_sha256": admission.get("predecessor_admission_sha256"),
    }
    if (
        payload["semantic_contract_sha256"]
        != protocol["runtime_admission"]["semantic_contract_sha256"]
    ):
        raise ProtocolViolation("semantic_contract_drift")
    if payload["protocol_sha256"] != protocol["_document_sha256"]:
        raise ProtocolViolation("admission_protocol_hash_mismatch")
    if (
        payload["reviewed"] is not True
        or payload["model_and_config_unchanged"] is not True
        or payload["protocol_semantics_unchanged"] is not True
    ):
        raise ProtocolViolation("admission_review_incomplete")
    predecessor = payload["predecessor_admission_sha256"]
    if predecessor is not None:
        _hash(predecessor, "predecessor_admission_sha256")
    return payload


def _install_admission(
    state: dict[str, Any], event: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    payload = _admission_payload(event, protocol)
    admission_id = payload["admission_id"]
    admission_sha = sha256_bytes(canonical_bytes(payload))
    existing = state["admissions"].get(admission_id)
    if existing is not None and existing != admission_sha:
        _fatal(state, f"conflicting_admission:{admission_id}")
        return
    expected_predecessor = state["active_admission_sha256"]
    if payload["predecessor_admission_sha256"] != expected_predecessor:
        raise ProtocolViolation("admission_lineage_mismatch")
    state["admissions"][admission_id] = admission_sha
    state["admission_payloads"][admission_id] = payload
    state["admission_times"][admission_id] = payload["admitted_at"]
    state["active_admission_id"] = admission_id
    state["active_admission_sha256"] = admission_sha
    state["paused_reason"] = None
    state["paused_at"] = None


def _authorize(
    state: dict[str, Any], event: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    if state["activation_at"] is not None:
        raise ProtocolViolation("already_authorized")
    if event.get("authority") != "separate_owner_authorization":
        raise ProtocolViolation("missing_separate_owner_authority")
    activation = _time(event.get("activation_at"), "activation_at")
    earliest = _time(protocol["eligibility"]["earliest_jump_local"], "earliest_jump_local")
    if activation < earliest:
        raise ProtocolViolation("activation_before_forward_population_boundary")
    payload = _admission_payload(event, protocol)
    if payload["predecessor_admission_sha256"] is not None:
        raise ProtocolViolation("initial_admission_has_predecessor")
    if _time(payload["admitted_at"], "admitted_at") != activation:
        raise ProtocolViolation("initial_admission_activation_mismatch")
    activation_receipt_sha256 = _hash(
        event.get("activation_receipt_sha256"), "activation_receipt_sha256"
    )
    _install_admission(state, event, protocol)
    state["activation_at"] = activation.isoformat()
    state["activation_receipt_sha256"] = activation_receipt_sha256


def _admission_failed(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    if state["activation_at"] is None:
        raise ProtocolViolation("admission_check_before_authorization")
    if state["state"] != "COLLECTING":
        raise ProtocolViolation("admission_check_outside_collection")
    reason = _text(event.get("reason"), "reason")
    if reason != "capture_code_or_unit_hash_unadmitted_before_seal":
        raise ProtocolViolation("admission_failure_not_temporary")
    observed_at = _time(event.get("observed_at"), "observed_at")
    active_admission_at = _time(
        state["admission_times"][state["active_admission_id"]], "active_admission_at"
    )
    if observed_at <= active_admission_at:
        raise ProtocolViolation("admission_failure_not_after_active_admission")
    state["paused_reason"] = reason
    state["paused_at"] = observed_at.isoformat()
    state["actions"] = ["HALT_SEALING_NO_WRITE"]


def _accept_admission(
    state: dict[str, Any], event: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    if state["paused_reason"] != "capture_code_or_unit_hash_unadmitted_before_seal":
        raise ProtocolViolation("admission_acceptance_without_pause")
    payload = event.get("admission")
    if not isinstance(payload, Mapping):
        raise ProtocolViolation("missing_admission")
    admitted_at = _time(payload.get("admitted_at"), "admitted_at")
    if admitted_at <= _time(state["paused_at"], "paused_at"):
        raise ProtocolViolation("successor_admission_not_after_observed_drift")
    parsed = _admission_payload(event, protocol)
    active = state["admission_payloads"][state["active_admission_id"]]
    if parsed["finalizer_code_sha256"] != active["finalizer_code_sha256"]:
        raise ProtocolViolation("finalizer_code_hash_drift")
    _install_admission(state, event, protocol)


def _prediction_sealed(
    state: dict[str, Any], event: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    target = protocol["cohort"]["target_races"]
    if state["state"] != "COLLECTING" or state["paused_reason"] is not None:
        _fatal(state, "sealed_prediction_while_not_admitted")
        return
    if len(state["predictions"]) >= target:
        _fatal(state, "sealed_prediction_after_fixed_n")
        return
    member_id = _text(event.get("member_id"), "member_id")
    race_id = _text(event.get("race_id"), "race_id")
    candidate_file = _text(event.get("candidate_file"), "candidate_file")
    candidate_content_sha256 = _hash(
        event.get("candidate_content_sha256"), "candidate_content_sha256"
    )
    prior_file_identity = state["candidate_files"].get(candidate_file)
    if prior_file_identity is not None and prior_file_identity != candidate_content_sha256:
        _fatal(state, f"candidate_file_identity_changed:{candidate_file}")
        return
    admission_id = _text(event.get("admission_id"), "admission_id")
    if admission_id != state["active_admission_id"]:
        _fatal(state, f"seal_under_unadmitted_runtime:{member_id}")
        return
    captured_at = _time(event.get("captured_at"), "captured_at")
    observed_at = _time(event.get("observed_at"), "observed_at")
    jump_at = _time(event.get("jump_at"), "jump_at")
    activation_at = _time(state["activation_at"], "activation_at")
    earliest_jump = _time(protocol["eligibility"]["earliest_jump_local"], "earliest_jump_local")
    if jump_at <= activation_at or jump_at < earliest_jump:
        _fatal(state, f"sealed_member_not_forward_eligible:{member_id}")
        return
    if not activation_at <= captured_at < jump_at:
        _fatal(state, f"sealed_member_not_forward_prejump:{member_id}")
        return
    if captured_at > observed_at or observed_at >= jump_at:
        _fatal(state, f"sealed_member_observation_not_prejump:{member_id}")
        return
    if not jump_at - timedelta(minutes=33) <= captured_at < jump_at - timedelta(minutes=10):
        _fatal(state, f"sealed_member_outside_frozen_window:{member_id}")
        return
    if _time(state["admission_times"][admission_id], "admitted_at") > captured_at:
        _fatal(state, f"admission_did_not_precede_capture:{member_id}")
        return
    order_key = (
        jump_at.astimezone(timezone.utc).isoformat(timespec="microseconds"),
        captured_at.astimezone(timezone.utc).isoformat(timespec="microseconds"),
        race_id,
    )
    prior_order_key = state["last_member_order_key"]
    if prior_order_key is not None and order_key <= tuple(prior_order_key):
        _fatal(state, f"membership_order_violation:{member_id}")
        return
    prediction = {
        "member_id": member_id,
        "race_id": race_id,
        "candidate_file": candidate_file,
        "candidate_content_sha256": candidate_content_sha256,
        "admission_id": admission_id,
        "admission_sha256": state["active_admission_sha256"],
        "captured_at": captured_at.isoformat(),
        "observed_at": observed_at.isoformat(),
        "jump_at": jump_at.isoformat(),
        "runner_set_sha256": _hash(event.get("runner_set_sha256"), "runner_set_sha256"),
        "odds_receipt_sha256": _hash(event.get("odds_receipt_sha256"), "odds_receipt_sha256"),
        "prediction_receipt_sha256": _hash(
            event.get("prediction_receipt_sha256"), "prediction_receipt_sha256"
        ),
    }
    existing = state["predictions"].get(member_id)
    existing_member = state["race_members"].get(race_id)
    if existing is not None:
        if existing != prediction:
            _fatal(state, f"conflicting_prediction_receipt:{member_id}")
        return
    if existing_member is not None:
        _fatal(state, f"duplicate_race_membership:{race_id}")
        return
    state["predictions"][member_id] = prediction
    state["race_members"][race_id] = member_id
    state["candidate_files"][candidate_file] = candidate_content_sha256
    state["last_member_order_key"] = list(order_key)


def _result_pending(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    member_id = _text(event.get("member_id"), "member_id")
    prediction = state["predictions"].get(member_id)
    if prediction is None:
        raise ProtocolViolation("pending_result_for_nonmember")
    if member_id in state["results"]:
        raise ProtocolViolation("pending_result_already_closed")
    if event.get("reason") != "approved_result_not_yet_available":
        raise ProtocolViolation("invalid_result_pending_reason")
    observed_at = _time(event.get("observed_at"), "observed_at")
    if observed_at <= _time(prediction["jump_at"], "jump_at"):
        raise ProtocolViolation("pending_result_not_after_jump")
    state["actions"] = ["WAIT_FOR_APPROVED_RESULT"]


def _candidate_rejected(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    reason = _text(event.get("reason"), "reason")
    if reason not in TEMPORARY_REJECTION_REASONS:
        raise ProtocolViolation("candidate_rejection_not_temporary")
    observed_at = _time(event.get("observed_at"), "observed_at")
    if observed_at < _time(state["activation_at"], "activation_at"):
        raise ProtocolViolation("candidate_rejection_before_activation")
    candidate_id = _hash(event.get("candidate_id"), "candidate_id")
    candidate_file = _text(event.get("candidate_file"), "candidate_file")
    candidate_content_sha256 = _hash(
        event.get("candidate_content_sha256"), "candidate_content_sha256"
    )
    race_id = event.get("race_id")
    if race_id is not None:
        race_id = _text(race_id, "race_id")
    source_receipt_sha256 = event.get("source_receipt_sha256")
    if source_receipt_sha256 is not None:
        source_receipt_sha256 = _hash(
            source_receipt_sha256, "source_receipt_sha256"
        )
    rejection = {
        "candidate_id": candidate_id,
        "candidate_file": candidate_file,
        "candidate_content_sha256": candidate_content_sha256,
        "race_id": race_id,
        "observed_at": observed_at.isoformat(),
        "reason": reason,
        "source_receipt_sha256": source_receipt_sha256,
        "detail": _text(event.get("detail"), "detail"),
    }
    prior_file_candidate = state["rejection_files"].get(candidate_file)
    if prior_file_candidate is not None and prior_file_candidate != candidate_id:
        _fatal(state, f"candidate_file_identity_changed:{candidate_file}")
        return
    existing = state["rejections"].get(candidate_id)
    if existing is not None and existing != rejection:
        _fatal(state, f"conflicting_candidate_rejection:{candidate_id}")
        return
    state["rejections"][candidate_id] = rejection
    state["rejection_files"][candidate_file] = candidate_id


def _result_appended(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    member_id = _text(event.get("member_id"), "member_id")
    prediction = state["predictions"].get(member_id)
    if prediction is None:
        raise ProtocolViolation("result_for_nonmember")
    result = {
        "member_id": member_id,
        "race_id": _text(event.get("race_id"), "race_id"),
        "runner_set_sha256": _hash(event.get("runner_set_sha256"), "runner_set_sha256"),
        "result_receipt_sha256": _hash(event.get("result_receipt_sha256"), "result_receipt_sha256"),
        "captured_at": _time(event.get("captured_at"), "captured_at").isoformat(),
        "observed_at": _time(event.get("observed_at"), "observed_at").isoformat(),
        "winner_box": event.get("winner_box"),
    }
    if (
        result["race_id"] != prediction["race_id"]
        or result["runner_set_sha256"] != prediction["runner_set_sha256"]
        or _time(result["captured_at"], "captured_at") <= _time(prediction["jump_at"], "jump_at")
        or _time(result["captured_at"], "captured_at")
        > _time(result["observed_at"], "observed_at")
        or not isinstance(result["winner_box"], int)
        or isinstance(result["winner_box"], bool)
        or result["winner_box"] <= 0
    ):
        _fatal(state, f"result_identity_conflict:{member_id}")
        return
    existing = state["results"].get(member_id)
    if existing is not None and existing != result:
        _fatal(state, f"conflicting_result_receipt:{member_id}")
        return
    state["results"][member_id] = result


def _request_finalization(state: dict[str, Any], target: int) -> None:
    if state["state"] != "READY_TO_FINALIZE":
        raise ProtocolViolation(
            f"fixed_n_not_ready:predictions={len(state['predictions'])}/{target}:"
            f"results={len(state['results'])}/{target}"
        )
    state["finalization_requested"] = True
    state["score_invocation_count"] += 1
    members = []
    ordered_predictions = sorted(
        state["predictions"].values(),
        key=lambda row: (row["jump_at"], row["captured_at"], row["race_id"]),
    )
    for prediction in ordered_predictions:
        result = state["results"][prediction["member_id"]]
        members.append(
            {
                "member_id": prediction["member_id"],
                "prediction_receipt_sha256": prediction["prediction_receipt_sha256"],
                "result_receipt_sha256": result["result_receipt_sha256"],
            }
        )
    state["finalization_member_manifest_sha256"] = sha256_bytes(canonical_bytes(members))
    state["state"] = "FINALIZATION_LOCKED"
    state["actions"] = ["RUN_PAIRED_SCORER_ON_EXACT_FIXED_N_MEMBERS"]


def _commit_score(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    if state["state"] != "FINALIZATION_LOCKED" or state["score_invocation_count"] != 1:
        raise ProtocolViolation("paired_score_not_requested_once")
    member_manifest = _hash(event.get("member_manifest_sha256"), "member_manifest_sha256")
    if member_manifest != state["finalization_member_manifest_sha256"]:
        raise ProtocolViolation("paired_score_member_manifest_mismatch")
    state["metrics_receipt_sha256"] = _hash(
        event.get("metrics_receipt_sha256"), "metrics_receipt_sha256"
    )
    state["state"] = "FINALIZED_SCORED"
    state["actions"] = ["SEAL_FINAL_REPORT_AND_CONSUMED_RECEIPT"]


def apply_event(
    snapshot: Mapping[str, Any], event: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise ProtocolViolation("snapshot_must_be_mutable_dict")
    state = snapshot
    state["actions"] = []
    event_id = _text(event.get("event_id"), "event_id")
    event_sha = sha256_bytes(canonical_bytes(event))
    prior = state["event_hashes"].get(event_id)
    if prior is not None:
        if prior != event_sha:
            _fatal(state, f"conflicting_event_id:{event_id}")
        return state
    if state["state"] in TERMINAL_STATES:
        raise ProtocolViolation(f"event_after_terminal:{event_id}")
    event_type = _text(event.get("type"), "type")
    try:
        if event_type == "AUTHORIZE":
            _authorize(state, event, protocol)
        elif event_type == "ADMISSION_CHECK_FAILED":
            _admission_failed(state, event)
        elif event_type == "ADMISSION_ACCEPTED":
            try:
                _accept_admission(state, event, protocol)
            except ProtocolViolation as exc:
                _fatal(state, str(exc))
        elif event_type == "CANDIDATE_REJECTED":
            if state["state"] != "COLLECTING":
                raise ProtocolViolation("candidate_rejection_outside_collection")
            _candidate_rejected(state, event)
        elif event_type == "PREDICTION_SEALED":
            try:
                _prediction_sealed(state, event, protocol)
            except ProtocolViolation as exc:
                _fatal(state, str(exc))
        elif event_type == "RESULT_APPENDED":
            try:
                _result_appended(state, event)
            except ProtocolViolation as exc:
                _fatal(state, str(exc))
        elif event_type == "RESULT_PENDING":
            _result_pending(state, event)
        elif event_type == "SEALED_EVIDENCE_INVALID":
            _fatal(state, _text(event.get("reason"), "reason"))
        elif event_type == "OPERATOR_ABORT":
            _fatal(state, f"operator_abort:{_text(event.get('reason'), 'reason')}")
        elif event_type == "FINALIZE_REQUESTED":
            _request_finalization(state, protocol["cohort"]["target_races"])
        elif event_type == "PAIRED_SCORE_COMMITTED":
            try:
                _commit_score(state, event)
            except ProtocolViolation as exc:
                _fatal(state, str(exc))
        else:
            raise ProtocolViolation(f"unknown_event_type:{event_type}")
    except ProtocolViolation:
        raise
    state["event_hashes"][event_id] = event_sha
    _derive_state(state, protocol["cohort"]["target_races"])
    return state


def replay(protocol: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state = initial_snapshot(protocol)
    for event in events:
        state = apply_event(state, event, protocol)
    return state


def public_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return status fields only; never expose interim outcomes or candidate loss."""
    rejection_reason_counts: dict[str, int] = {}
    for rejection in state["rejections"].values():
        reason = rejection["reason"]
        rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
    return {
        "schema_version": state["schema_version"],
        "protocol_sha256": state["protocol_sha256"],
        "state": state["state"],
        "sealed_prediction_races": len(state["predictions"]),
        "approved_result_races": len(state["results"]),
        "excluded_candidate_events": len(state["rejections"]),
        "exclusion_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "active_admission_id": state["active_admission_id"],
        "paused_reason": state["paused_reason"],
        "paused_at": state["paused_at"],
        "fatal_reason": state["fatal_reason"],
        "score_invocation_count": state["score_invocation_count"],
        "finalization_member_manifest_sha256": state["finalization_member_manifest_sha256"],
        "metrics_receipt_sha256": state["metrics_receipt_sha256"],
        "actions": list(state["actions"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    events = json.loads(args.events.read_text())
    if not isinstance(events, list):
        raise ProtocolViolation("events_must_be_a_list")
    print(json.dumps(public_snapshot(replay(protocol, events)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
