from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.forward_overround_successor_state_machine import (
    ProtocolViolation,
    apply_event,
    initial_snapshot,
    load_protocol,
    public_snapshot,
    replay,
)

PROTOCOL_PATH = (
    Path(__file__).parents[1] / "configs/prediction/forward_overround_successor_v1_protocol.json"
)
H = "a" * 64


def protocol_with_target(target: int) -> dict:
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["cohort"]["target_races"] = target
    protocol["evaluation"]["finalization"].update(
        requires_approved_results=target,
        requires_sealed_predictions=target,
    )
    return protocol


def admission(
    protocol: dict,
    admission_id: str,
    admitted_at: str,
    *,
    predecessor: str | None = None,
    code_hash: str = "b" * 64,
) -> dict:
    return {
        "admission_id": admission_id,
        "admitted_at": admitted_at,
        "capture_code_sha256": code_hash,
        "capture_unit_sha256": "c" * 64,
        "finalizer_code_sha256": "d" * 64,
        "semantic_contract_sha256": protocol["runtime_admission"]["semantic_contract_sha256"],
        "protocol_sha256": protocol["_document_sha256"],
        "reviewed": True,
        "model_and_config_unchanged": True,
        "protocol_semantics_unchanged": True,
        "predecessor_admission_sha256": predecessor,
    }


def authorize(protocol: dict, admitted_at: str = "2026-09-01T00:00:00+10:00") -> dict:
    return {
        "event_id": "authorize",
        "type": "AUTHORIZE",
        "authority": "separate_owner_authorization",
        "activation_at": admitted_at,
        "admission": admission(protocol, "admission-1", admitted_at),
    }


def prediction(
    index: int,
    *,
    admission_id: str = "admission-1",
    jump_at: datetime | None = None,
    runner_hash: str = H,
) -> dict:
    jump = jump_at or datetime(2026, 9, 1, 1, 0, tzinfo=timezone(timedelta(hours=10))) + timedelta(
        minutes=index
    )
    return {
        "event_id": f"prediction-{index}",
        "type": "PREDICTION_SEALED",
        "member_id": f"member-{index}",
        "race_id": f"race-{index}",
        "admission_id": admission_id,
        "captured_at": (jump - timedelta(minutes=20)).isoformat(),
        "jump_at": jump.isoformat(),
        "runner_set_sha256": runner_hash,
        "odds_receipt_sha256": f"{index % 16:x}" * 64,
        "prediction_receipt_sha256": f"{(index + 1) % 16:x}" * 64,
    }


def result(
    index: int,
    *,
    runner_hash: str = H,
    captured_at: datetime | None = None,
) -> dict:
    captured = captured_at or datetime(
        2026, 9, 1, 1, 5, tzinfo=timezone(timedelta(hours=10))
    ) + timedelta(minutes=index)
    return {
        "event_id": f"result-{index}",
        "type": "RESULT_APPENDED",
        "member_id": f"member-{index}",
        "race_id": f"race-{index}",
        "runner_set_sha256": runner_hash,
        "result_receipt_sha256": f"{(index + 2) % 16:x}" * 64,
        "captured_at": captured.isoformat(),
        "winner_box": index % 8 + 1,
    }


def test_protocol_is_prepared_fixed_n_and_preserves_frozen_model() -> None:
    protocol = load_protocol(PROTOCOL_PATH)

    assert protocol["authorization"]["status"] == "PREPARED_NOT_AUTHORIZED"
    assert protocol["authorization"]["collection_authorized"] is False
    assert protocol["authorization"]["scheduler_install_authorized"] is False
    assert protocol["cohort"]["target_races"] == 1000
    assert protocol["cohort"]["administrative_deadline"] is None
    assert protocol["eligibility"]["earliest_jump_local"] == "2026-09-01T00:00:00+10:00"
    assert protocol["model"]["hashes"]["final_model.json"] == (
        "c81b4b3047b7840ba31269504e0c5ceb6c54d742a82a4e01cca52b11fdaa471e"
    )
    assert public_snapshot(initial_snapshot(protocol))["state"] == ("PREPARED_NOT_AUTHORIZED")


def test_exact_1000_member_path_reaches_one_paired_scoring_action() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    activation = authorize(protocol)
    first_jump = datetime(2026, 9, 1, 1, 0, tzinfo=timezone(timedelta(hours=10)))
    events = [activation]
    events.extend(prediction(i, jump_at=first_jump + timedelta(minutes=i)) for i in range(1000))
    events.extend(result(i, captured_at=first_jump + timedelta(minutes=i + 5)) for i in range(1000))
    state = replay(protocol, events)

    assert state["state"] == "READY_TO_FINALIZE"
    state = apply_event(
        state,
        {"event_id": "finalize", "type": "FINALIZE_REQUESTED"},
        protocol,
    )
    assert state["state"] == "FINALIZATION_LOCKED"
    assert state["score_invocation_count"] == 1
    assert state["actions"] == ["RUN_PAIRED_SCORER_ON_EXACT_FIXED_N_MEMBERS"]

    state = apply_event(
        state,
        {
            "event_id": "score-committed",
            "type": "PAIRED_SCORE_COMMITTED",
            "member_manifest_sha256": state["finalization_member_manifest_sha256"],
            "metrics_receipt_sha256": "e" * 64,
        },
        protocol,
    )
    assert state["state"] == "FINALIZED_SCORED"
    assert state["score_invocation_count"] == 1


def test_reviewed_code_drift_admission_resumes_without_poisoning_finalization() -> None:
    protocol = protocol_with_target(3)
    state = replay(protocol, [authorize(protocol), prediction(0), result(0)])
    state = apply_event(
        state,
        {
            "event_id": "pause-1",
            "type": "ADMISSION_CHECK_FAILED",
            "reason": "capture_code_or_unit_hash_unadmitted_before_seal",
            "observed_at": "2026-09-01T01:20:00+10:00",
        },
        protocol,
    )
    assert state["state"] == "ADMISSION_PAUSED"
    assert state["actions"] == ["HALT_SEALING_NO_WRITE"]
    prior_admission = state["active_admission_sha256"]

    state = apply_event(
        state,
        {
            "event_id": "admit-2",
            "type": "ADMISSION_ACCEPTED",
            "admission": admission(
                protocol,
                "admission-2",
                "2026-09-01T01:30:00+10:00",
                predecessor=prior_admission,
                code_hash="f" * 64,
            ),
        },
        protocol,
    )
    assert state["state"] == "COLLECTING"
    state = apply_event(
        state,
        prediction(
            1,
            admission_id="admission-2",
            jump_at=datetime(2026, 9, 1, 2, 0, tzinfo=timezone(timedelta(hours=10))),
        ),
        protocol,
    )
    state = apply_event(
        state,
        prediction(
            2,
            admission_id="admission-2",
            jump_at=datetime(2026, 9, 1, 2, 30, tzinfo=timezone(timedelta(hours=10))),
        ),
        protocol,
    )
    state = replay_from(
        state,
        protocol,
        [
            result(
                1,
                captured_at=datetime(2026, 9, 1, 2, 5, tzinfo=timezone(timedelta(hours=10))),
            ),
            result(
                2,
                captured_at=datetime(2026, 9, 1, 2, 35, tzinfo=timezone(timedelta(hours=10))),
            ),
        ],
    )
    assert state["state"] == "READY_TO_FINALIZE"

    state = apply_event(
        state,
        {"event_id": "finalize", "type": "FINALIZE_REQUESTED"},
        protocol,
    )
    assert state["actions"] == ["RUN_PAIRED_SCORER_ON_EXACT_FIXED_N_MEMBERS"]
    assert state["fatal_reason"] is None


def replay_from(state: dict, protocol: dict, events: list[dict]) -> dict:
    for event in events:
        state = apply_event(state, event, protocol)
    return state


def test_seal_while_admission_paused_is_fatal_with_no_metrics() -> None:
    protocol = protocol_with_target(2)
    state = replay(
        protocol,
        [
            authorize(protocol),
            {
                "event_id": "pause-1",
                "type": "ADMISSION_CHECK_FAILED",
                "reason": "capture_code_or_unit_hash_unadmitted_before_seal",
                "observed_at": "2026-09-01T01:20:00+10:00",
            },
        ],
    )

    state = apply_event(state, prediction(0), protocol)

    assert state["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert state["fatal_reason"] == "sealed_prediction_while_not_admitted"
    assert state["metrics_receipt_sha256"] is None
    assert state["score_invocation_count"] == 0


def test_re_admission_cannot_change_frozen_semantic_contract() -> None:
    protocol = protocol_with_target(1)
    state = replay(
        protocol,
        [
            authorize(protocol),
            {
                "event_id": "pause-1",
                "type": "ADMISSION_CHECK_FAILED",
                "reason": "capture_code_or_unit_hash_unadmitted_before_seal",
                "observed_at": "2026-09-01T01:20:00+10:00",
            },
        ],
    )
    changed = admission(
        protocol,
        "admission-2",
        "2026-09-01T01:30:00+10:00",
        predecessor=state["active_admission_sha256"],
    )
    changed["semantic_contract_sha256"] = "0" * 64

    state = apply_event(
        state,
        {
            "event_id": "admit-invalid",
            "type": "ADMISSION_ACCEPTED",
            "admission": changed,
        },
        protocol,
    )

    assert state["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert state["fatal_reason"] == "semantic_contract_drift"
    assert state["metrics_receipt_sha256"] is None


def test_re_admission_cannot_change_frozen_finalizer_code() -> None:
    protocol = protocol_with_target(1)
    state = replay(
        protocol,
        [
            authorize(protocol),
            {
                "event_id": "pause-1",
                "type": "ADMISSION_CHECK_FAILED",
                "reason": "capture_code_or_unit_hash_unadmitted_before_seal",
                "observed_at": "2026-09-01T01:20:00+10:00",
            },
        ],
    )
    changed = admission(
        protocol,
        "admission-2",
        "2026-09-01T01:30:00+10:00",
        predecessor=state["active_admission_sha256"],
    )
    changed["finalizer_code_sha256"] = "1" * 64

    state = apply_event(
        state,
        {
            "event_id": "admit-finalizer-drift",
            "type": "ADMISSION_ACCEPTED",
            "admission": changed,
        },
        protocol,
    )

    assert state["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert state["fatal_reason"] == "finalizer_code_hash_drift"
    assert state["metrics_receipt_sha256"] is None


def test_candidate_rejection_is_nonmember_and_does_not_poison_path() -> None:
    protocol = protocol_with_target(1)
    rejected = {
        "event_id": "reject-1",
        "type": "CANDIDATE_REJECTED",
        "candidate_id": "candidate-1",
        "race_id": "rejected-race-1",
        "observed_at": "2026-09-01T00:30:00+10:00",
        "reason": "candidate_identity_ambiguous",
        "source_receipt_sha256": "7" * 64,
    }
    state = replay(protocol, [authorize(protocol), rejected])

    assert state["state"] == "COLLECTING"
    assert state["predictions"] == {}
    assert public_snapshot(state)["excluded_candidate_events"] == 1
    state = replay_from(state, protocol, [prediction(0), result(0)])
    assert state["state"] == "READY_TO_FINALIZE"


def test_result_identity_conflict_is_fatal_and_never_scores() -> None:
    protocol = protocol_with_target(1)
    state = replay(protocol, [authorize(protocol), prediction(0)])

    state = apply_event(state, result(0, runner_hash="9" * 64), protocol)

    assert state["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert state["fatal_reason"] == "result_identity_conflict:member-0"
    assert state["metrics_receipt_sha256"] is None
    assert state["score_invocation_count"] == 0


def test_result_observed_before_jump_is_fatal_and_never_scores() -> None:
    protocol = protocol_with_target(1)
    state = replay(protocol, [authorize(protocol), prediction(0)])
    too_early = result(
        0,
        captured_at=datetime(2026, 9, 1, 0, 59, tzinfo=timezone(timedelta(hours=10))),
    )

    state = apply_event(state, too_early, protocol)

    assert state["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert state["fatal_reason"] == "result_identity_conflict:member-0"
    assert state["score_invocation_count"] == 0


def test_result_pending_is_temporary_until_the_same_member_closes() -> None:
    protocol = protocol_with_target(1)
    state = replay(protocol, [authorize(protocol), prediction(0)])

    state = apply_event(
        state,
        {
            "event_id": "result-pending-0",
            "type": "RESULT_PENDING",
            "member_id": "member-0",
            "reason": "approved_result_not_yet_available",
            "observed_at": "2026-09-01T01:01:00+10:00",
        },
        protocol,
    )

    assert state["state"] == "RESULT_CLOSURE"
    assert state["actions"] == ["WAIT_FOR_APPROVED_RESULT"]
    assert state["fatal_reason"] is None
    state = apply_event(state, result(0), protocol)
    assert state["state"] == "READY_TO_FINALIZE"


def test_restart_replay_is_idempotent_but_conflicting_event_id_is_fatal() -> None:
    protocol = protocol_with_target(2)
    event = prediction(0)
    state = replay(protocol, [authorize(protocol), event, event])

    assert state["state"] == "COLLECTING"
    assert len(state["predictions"]) == 1
    conflicting = {**event, "prediction_receipt_sha256": "8" * 64}
    state = apply_event(state, conflicting, protocol)
    assert state["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert state["fatal_reason"] == "conflicting_event_id:prediction-0"


def test_finalization_before_fixed_n_or_result_closure_is_rejected() -> None:
    protocol = protocol_with_target(2)
    state = replay(protocol, [authorize(protocol), prediction(0), result(0)])

    with pytest.raises(ProtocolViolation, match="fixed_n_not_ready"):
        apply_event(
            state,
            {"event_id": "finalize", "type": "FINALIZE_REQUESTED"},
            protocol,
        )
    assert state["score_invocation_count"] == 0
    assert state["metrics_receipt_sha256"] is None


def test_public_status_never_contains_predictions_results_or_candidate_loss() -> None:
    protocol = protocol_with_target(2)
    state = replay(protocol, [authorize(protocol), prediction(0), result(0)])
    public = public_snapshot(state)

    assert "predictions" not in public
    assert "results" not in public
    assert "loss" not in public
    assert public["metrics_receipt_sha256"] is None


def test_operator_abort_is_explicit_terminal_no_metrics() -> None:
    protocol = protocol_with_target(2)
    state = replay(
        protocol,
        [
            authorize(protocol),
            {
                "event_id": "abort",
                "type": "OPERATOR_ABORT",
                "reason": "approved_result_permanently_unavailable",
            },
        ],
    )

    assert state["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert state["metrics_receipt_sha256"] is None
    assert state["actions"] == ["SEAL_TERMINAL_NO_METRICS"]
