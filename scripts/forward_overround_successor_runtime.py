#!/usr/bin/env python3
"""Append-only collector/sealer runtime for the prepared overround successor.

The runtime cannot create or authorize a cohort. It operates only when an
existing cohort directory contains a separately supplied ACTIVATION.json.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

try:
    from scripts.finalize_forward_overround_successor import (
        ASSET_HASHES,
        EXPECTED_PROTOCOL_SHA256,
        FinalizationError,
        finalize,
        load_frozen_assets,
        runner_set_sha256,
        score_race,
        validate_finish_positions,
    )
    from scripts.forward_overround_successor_state_machine import (
        ProtocolViolation,
        apply_event,
        canonical_bytes,
        initial_snapshot,
        public_snapshot,
        sha256_bytes,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from finalize_forward_overround_successor import (  # type: ignore[no-redef]
        ASSET_HASHES,
        EXPECTED_PROTOCOL_SHA256,
        FinalizationError,
        finalize,
        load_frozen_assets,
        runner_set_sha256,
        score_race,
        validate_finish_positions,
    )
    from forward_overround_successor_state_machine import (  # type: ignore[no-redef]
        ProtocolViolation,
        apply_event,
        canonical_bytes,
        initial_snapshot,
        public_snapshot,
        sha256_bytes,
    )

TERMINAL_STATES = frozenset({"FINALIZED_SCORED", "FINALIZED_ABORTED_NO_METRICS"})
ObservationClock = Callable[[], datetime]


class RuntimeEvidenceError(ValueError):
    """Raised for runtime evidence that must fail closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeEvidenceError(f"invalid_json:{path}:{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeEvidenceError(f"json_object_required:{path}")
    return payload


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeEvidenceError(f"invalid_text:{field}")
    return value


def _sha(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeEvidenceError(f"invalid_sha256:{field}")
    return text


def _aware_time(value: Any, field: str) -> datetime:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeEvidenceError(f"invalid_timestamp:{field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeEvidenceError(f"naive_timestamp:{field}")
    return parsed


def _utc_iso(value: Any, field: str) -> str:
    return _aware_time(value, field).astimezone(timezone.utc).isoformat()


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


def _observe_utc(clock: ObservationClock) -> datetime:
    observed_at = clock()
    if not isinstance(observed_at, datetime):
        raise RuntimeEvidenceError("observation_clock_did_not_return_datetime")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise RuntimeEvidenceError("observation_clock_returned_naive_datetime")
    return observed_at.astimezone(timezone.utc)


def _write_once(path: Path, payload: Mapping[str, Any]) -> str:
    raw = canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise RuntimeEvidenceError(f"write_once_conflict:{path}") from None
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    return sha256_bytes(raw)


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class CohortStore:
    def __init__(self, root: Path, protocol: Mapping[str, Any]):
        self.root = root
        self.protocol = protocol
        if not root.is_dir():
            raise RuntimeEvidenceError("cohort_root_must_preexist")
        if not (root / "ACTIVATION.json").is_file():
            raise RuntimeEvidenceError("activation_receipt_absent")
        self.journal = root / "EVENTS.jsonl"

    def _journal_rows(self) -> list[dict[str, Any]]:
        if not self.journal.exists():
            return []
        rows: list[dict[str, Any]] = []
        previous: str | None = None
        for line_number, raw_line in enumerate(self.journal.read_bytes().splitlines(), start=1):
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeEvidenceError(f"journal_json_invalid:{line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("event"), dict):
                raise RuntimeEvidenceError(f"journal_row_invalid:{line_number}")
            event_hash = sha256_bytes(canonical_bytes(row["event"]))
            if row.get("event_sha256") != event_hash:
                raise RuntimeEvidenceError(f"journal_event_hash_drift:{line_number}")
            if row.get("previous_event_sha256") != previous:
                raise RuntimeEvidenceError(f"journal_hash_chain_drift:{line_number}")
            if row.get("journal_index") != line_number:
                raise RuntimeEvidenceError(f"journal_index_drift:{line_number}")
            previous = event_hash
            rows.append(row)
        return rows

    def replay(
        self, *, verify_receipts: bool = True
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rows = self._journal_rows()
        state = initial_snapshot(self.protocol)
        for row in rows:
            try:
                state = apply_event(state, row["event"], self.protocol)
            except ProtocolViolation as exc:
                raise RuntimeEvidenceError(f"journal_protocol_violation:{exc}") from exc
        if verify_receipts:
            self.verify_receipts(state)
        return state, rows

    def verify_receipts(self, state: Mapping[str, Any]) -> None:
        activation_path = self.root / "ACTIVATION.json"
        expected_activation_sha256 = state.get("activation_receipt_sha256")
        if state.get("activation_at") is not None:
            if (
                not activation_path.is_file()
                or not isinstance(expected_activation_sha256, str)
                or sha256_file(activation_path) != expected_activation_sha256
            ):
                raise RuntimeEvidenceError("activation_receipt_hash_drift")
        for member_id, item in state["predictions"].items():
            path = self.root / "predictions" / f"{member_id}.json"
            if not path.is_file() or sha256_file(path) != item["prediction_receipt_sha256"]:
                raise RuntimeEvidenceError(f"sealed_prediction_receipt_hash_drift:{member_id}")
        for member_id, item in state["results"].items():
            path = self.root / "results" / f"{member_id}.json"
            if not path.is_file() or sha256_file(path) != item["result_receipt_sha256"]:
                raise RuntimeEvidenceError(f"sealed_result_receipt_hash_drift:{member_id}")

    def append(self, state: dict[str, Any], rows: list[dict[str, Any]], event: Mapping[str, Any]) -> dict[str, Any]:
        try:
            state = apply_event(state, event, self.protocol)
        except ProtocolViolation as exc:
            raise RuntimeEvidenceError(str(exc)) from exc
        event_hash = sha256_bytes(canonical_bytes(event))
        row = {
            "event": dict(event),
            "event_sha256": event_hash,
            "journal_index": len(rows) + 1,
            "previous_event_sha256": rows[-1]["event_sha256"] if rows else None,
        }
        raw = canonical_bytes(row)
        descriptor = os.open(self.journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        rows.append(row)
        return state


def runtime_identity(finalizer_path: Path, service_unit_path: Path) -> dict[str, str]:
    paths = {
        "capture_code_sha256": Path(__file__).resolve(),
        "finalizer_code_sha256": finalizer_path.resolve(),
        "capture_unit_sha256": service_unit_path.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeEvidenceError(f"runtime_identity_path_missing:{path}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _admission_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "admission_id": _text(receipt.get("admission_id"), "admission_id"),
        "admitted_at": _utc_iso(receipt.get("admitted_at"), "admitted_at"),
        "capture_code_sha256": _sha(receipt.get("capture_code_sha256"), "capture_code_sha256"),
        "capture_unit_sha256": _sha(receipt.get("capture_unit_sha256"), "capture_unit_sha256"),
        "finalizer_code_sha256": _sha(receipt.get("finalizer_code_sha256"), "finalizer_code_sha256"),
        "semantic_contract_sha256": _sha(receipt.get("semantic_contract_sha256"), "semantic_contract_sha256"),
        "protocol_sha256": _sha(receipt.get("protocol_sha256"), "protocol_sha256"),
        "reviewed": receipt.get("reviewed"),
        "model_and_config_unchanged": receipt.get("model_and_config_unchanged"),
        "protocol_semantics_unchanged": receipt.get("protocol_semantics_unchanged"),
        "predecessor_admission_sha256": receipt.get("predecessor_admission_sha256"),
    }


def _verify_asset_receipt(receipt: Mapping[str, Any]) -> None:
    asset_hashes = receipt.get("asset_hashes")
    if not isinstance(asset_hashes, Mapping):
        raise RuntimeEvidenceError("activation_asset_hashes_missing")
    for name, expected in ASSET_HASHES.items():
        if asset_hashes.get(name) != expected:
            raise RuntimeEvidenceError(f"activation_frozen_asset_mismatch:{name}")


def _authorize(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    identity: Mapping[str, str],
    observed_at: datetime,
) -> dict[str, Any]:
    receipt_path = store.root / "ACTIVATION.json"
    receipt = _load_json(receipt_path)
    if receipt.get("schema_version") != "forward_overround_successor_activation_v1":
        raise RuntimeEvidenceError("activation_schema_mismatch")
    if receipt.get("authority") != "separate_owner_authorization":
        raise RuntimeEvidenceError("activation_authority_missing")
    if receipt.get("collection_authorized") is not True or receipt.get("scheduler_authorized") is not True:
        raise RuntimeEvidenceError("activation_scope_incomplete")
    _verify_asset_receipt(receipt)
    admission = _admission_payload(receipt)
    for field, actual in identity.items():
        if admission[field] != actual:
            raise RuntimeEvidenceError(f"activation_runtime_identity_mismatch:{field}")
    activation_at = _aware_time(receipt.get("activation_at"), "activation_at")
    if activation_at > observed_at:
        raise RuntimeEvidenceError("activation_receipt_not_yet_effective")
    receipt_sha = sha256_file(receipt_path)
    event = {
        "event_id": f"authorize-{receipt_sha}",
        "type": "AUTHORIZE",
        "authority": "separate_owner_authorization",
        "activation_at": activation_at.astimezone(timezone.utc).isoformat(),
        "activation_receipt_sha256": receipt_sha,
        "admission": admission,
    }
    return store.append(state, rows, event)


def _abort(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    if state["state"] not in TERMINAL_STATES:
        event = {
            "event_id": f"fatal-{len(rows) + 1}-{sha256_bytes(reason.encode())}",
            "type": "SEALED_EVIDENCE_INVALID",
            "reason": reason,
        }
        state = store.append(state, rows, event)
    report = {
        "schema_version": "forward_overround_successor_final_report_v1",
        "verdict": "BLOCKED_FORWARD_EVIDENCE",
        "blocking_reason": state.get("fatal_reason") or reason,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "metrics": None,
        "sealed_prediction_races": len(state["predictions"]),
        "approved_result_races": len(state["results"]),
        "score_invocation_count": state["score_invocation_count"],
    }
    report_sha = _write_once(store.root / "FINAL_REPORT.json", report)
    _write_once(
        store.root / "CONSUMED.json",
        {
            "schema_version": "forward_overround_successor_consumed_v1",
            "verdict": "BLOCKED_FORWARD_EVIDENCE",
            "final_report_sha256": report_sha,
        },
    )
    return state


def _independent_terminal(
    cohort_root: Path,
    reason: str,
) -> dict[str, Any]:
    """Seal an untrusted-store failure without reading or changing its journal."""
    sentinel = {
        "schema_version": "forward_overround_successor_independent_terminal_v1",
        "state": "FINALIZED_ABORTED_NO_METRICS",
        "blocking_reason": reason,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "journal_mutated": False,
        "metrics": None,
        "score_invocation_count": 0,
    }
    sentinel_sha = _write_once(cohort_root / "TERMINAL_SENTINEL.json", sentinel)
    report = {
        "schema_version": "forward_overround_successor_final_report_v1",
        "verdict": "BLOCKED_FORWARD_EVIDENCE",
        "blocking_reason": reason,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "evidence_counts_trusted": False,
        "metrics": None,
        "sealed_prediction_races": None,
        "approved_result_races": None,
        "score_invocation_count": 0,
        "terminal_sentinel_sha256": sentinel_sha,
    }
    report_sha = _write_once(cohort_root / "FINAL_REPORT.json", report)
    _write_once(
        cohort_root / "CONSUMED.json",
        {
            "schema_version": "forward_overround_successor_consumed_v1",
            "verdict": "BLOCKED_FORWARD_EVIDENCE",
            "final_report_sha256": report_sha,
            "terminal_sentinel_sha256": sentinel_sha,
        },
    )
    status = {
        "schema_version": "forward_overround_successor_state_v1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "state": "FINALIZED_ABORTED_NO_METRICS",
        "sealed_prediction_races": None,
        "approved_result_races": None,
        "excluded_candidate_events": None,
        "exclusion_reason_counts": {},
        "active_admission_id": None,
        "paused_reason": None,
        "paused_at": None,
        "fatal_reason": reason,
        "score_invocation_count": 0,
        "finalization_member_manifest_sha256": None,
        "metrics_receipt_sha256": None,
        "actions": ["SEAL_TERMINAL_NO_METRICS"],
        "target_races": 1000,
        "interim_aggregate_performance_emitted": False,
        "evidence_counts_trusted": False,
        "terminal_sentinel_sha256": sentinel_sha,
    }
    _write_status(cohort_root / "STATUS.json", status)
    return status


def _resume_independent_terminal(cohort_root: Path) -> dict[str, Any] | None:
    sentinel_path = cohort_root / "TERMINAL_SENTINEL.json"
    if not sentinel_path.exists():
        return None
    sentinel = _load_json(sentinel_path)
    if (
        sentinel.get("schema_version")
        != "forward_overround_successor_independent_terminal_v1"
        or sentinel.get("state") != "FINALIZED_ABORTED_NO_METRICS"
        or sentinel.get("metrics") is not None
        or sentinel.get("score_invocation_count") != 0
        or not isinstance(sentinel.get("blocking_reason"), str)
    ):
        raise RuntimeEvidenceError("independent_terminal_sentinel_invalid")
    return _independent_terminal(cohort_root, sentinel["blocking_reason"])


def _accept_reviewed_admission(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    identity: Mapping[str, str],
    observed_at: datetime,
) -> dict[str, Any]:
    admission_dir = store.root / "admission_inbox"
    if not admission_dir.is_dir():
        return state
    for path in sorted(admission_dir.glob("*.json")):
        receipt = _load_json(path)
        if receipt.get("schema_version") != "forward_overround_successor_admission_v1":
            continue
        _verify_asset_receipt(receipt)
        admission = _admission_payload(receipt)
        if _aware_time(admission["admitted_at"], "admitted_at") > observed_at:
            continue
        if any(admission[field] != actual for field, actual in identity.items()):
            continue
        if admission["predecessor_admission_sha256"] != state["active_admission_sha256"]:
            continue
        event = {
            "event_id": f"admission-{sha256_file(path)}",
            "type": "ADMISSION_ACCEPTED",
            "admission_receipt_sha256": sha256_file(path),
            "admission": admission,
        }
        state = store.append(state, rows, event)
        break
    return state


def _check_runtime_admission(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    identity: Mapping[str, str],
    observed_at: datetime,
) -> dict[str, Any]:
    active = state["admission_payloads"].get(state["active_admission_id"])
    if active is None:
        return _abort(store, state, rows, "active_runtime_admission_missing")
    if active["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        return _abort(store, state, rows, "successor_protocol_hash_drift")
    if active["semantic_contract_sha256"] != store.protocol["runtime_admission"]["semantic_contract_sha256"]:
        return _abort(store, state, rows, "semantic_contract_hash_drift")
    if active["finalizer_code_sha256"] != identity["finalizer_code_sha256"]:
        return _abort(store, state, rows, "finalizer_code_hash_drift")
    capture_drift = any(
        active[field] != identity[field]
        for field in ("capture_code_sha256", "capture_unit_sha256")
    )
    if capture_drift and state["state"] == "COLLECTING":
        state = store.append(
            state,
            rows,
            {
                "event_id": f"admission-pause-{len(rows) + 1}",
                "type": "ADMISSION_CHECK_FAILED",
                "reason": "capture_code_or_unit_hash_unadmitted_before_seal",
                "observed_at": observed_at.isoformat(),
            },
        )
    if state["state"] == "ADMISSION_PAUSED":
        state = _accept_reviewed_admission(store, state, rows, identity, observed_at)
    return state


def _prediction_receipt(
    candidate: Mapping[str, Any],
    state: Mapping[str, Any],
    protocol: Mapping[str, Any],
    model: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    observed_at: datetime,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if candidate.get("schema_version") != "forward_overround_successor_candidate_v1":
        raise RuntimeEvidenceError("candidate_schema_mismatch")
    race_id = _text(candidate.get("race_id"), "race_id")
    captured_at_source = _text(candidate.get("captured_at"), "captured_at")
    jump_at_source = _text(candidate.get("jump_at"), "jump_at")
    captured_at = _aware_time(captured_at_source, "captured_at")
    jump_at = _aware_time(jump_at_source, "jump_at")
    activation_at = _aware_time(state.get("activation_at"), "activation_at")
    earliest_jump = _aware_time(protocol["eligibility"]["earliest_jump_local"], "earliest_jump_local")
    active_admission_at = _aware_time(
        state["admission_times"][state["active_admission_id"]], "active_admission_at"
    )
    if (
        jump_at <= activation_at
        or jump_at < earliest_jump
        or not activation_at <= captured_at < jump_at
        or not jump_at - timedelta(minutes=33) <= captured_at < jump_at - timedelta(minutes=10)
        or captured_at > observed_at
        or active_admission_at > captured_at
    ):
        raise RuntimeEvidenceError("candidate_timing_outside_window")
    if candidate.get("source") != "sportsbet" or candidate.get("market_type") != "win":
        raise RuntimeEvidenceError("candidate_sportsbet_win_provenance_ambiguous")
    if candidate.get("raw_paired_column_win_proof") is not True:
        raise RuntimeEvidenceError("candidate_raw_paired_win_proof_missing")
    source_receipt_sha = _sha(candidate.get("source_receipt_sha256"), "source_receipt_sha256")
    runners = candidate.get("runners")
    if not isinstance(runners, list) or candidate.get("active_runner_count") != len(runners):
        raise RuntimeEvidenceError("candidate_incomplete_active_runner_set")
    scores = score_race(runners, model, preprocessing)
    sealed_runners: list[dict[str, Any]] = []
    for runner in sorted(runners, key=lambda row: row.get("box_number", 0)):
        box = runner.get("box_number")
        if box not in scores:
            raise RuntimeEvidenceError("candidate_runner_identity_invalid")
        sealed_runners.append(
            {
                "box_number": box,
                "dog_name": runner.get("dog_name"),
                "decimal_win_odds": runner.get("decimal_win_odds"),
                "source_row_sha256": _sha(runner.get("source_row_sha256"), "source_row_sha256"),
                **scores[box],
            }
        )
    runner_hash = runner_set_sha256(sealed_runners)
    member_id = sha256_bytes(canonical_bytes({"race_id": race_id, "runner_set_sha256": runner_hash}))
    receipt = {
        "schema_version": "forward_overround_successor_prediction_receipt_v1",
        "member_id": member_id,
        "race_id": race_id,
        "captured_at": captured_at_source,
        "jump_at": jump_at_source,
        "source": "sportsbet",
        "market_type": "win",
        "raw_paired_column_win_proof": True,
        "source_receipt_sha256": source_receipt_sha,
        "runner_set_sha256": runner_hash,
        "active_runner_count": len(sealed_runners),
        "admission_id": state["active_admission_id"],
        "admission_sha256": state["active_admission_sha256"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runners": sealed_runners,
    }
    event = {
        "event_id": f"prediction-{member_id}",
        "type": "PREDICTION_SEALED",
        "member_id": member_id,
        "race_id": race_id,
        "admission_id": state["active_admission_id"],
        "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
        "jump_at": jump_at.astimezone(timezone.utc).isoformat(),
        "runner_set_sha256": runner_hash,
        "odds_receipt_sha256": source_receipt_sha,
    }
    return member_id, receipt, event


def _candidate_rejection_reason(exc: Exception) -> str:
    detail = str(exc)
    if detail == "candidate_timing_outside_window":
        return "candidate_timing_outside_window"
    if detail in {
        "candidate_sportsbet_win_provenance_ambiguous",
        "candidate_raw_paired_win_proof_missing",
    }:
        return "candidate_win_provenance_ambiguous"
    if detail.startswith(
        (
            "candidate_runner_identity_invalid",
            "duplicate_exact_dog_name",
            "incomplete_or_duplicate_runner_set",
            "invalid_box_number",
            "invalid_exact_dog_name",
        )
    ):
        return "candidate_identity_ambiguous"
    return "candidate_incomplete_field"


def _sealed_receipt_matches(
    path: Path,
    expected_sha256: str,
    incoming: Mapping[str, Any],
) -> bool:
    return (
        path.is_file()
        and sha256_file(path) == expected_sha256
        and path.read_bytes() == canonical_bytes(incoming)
    )


def _seal_candidates(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    model: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    inbox = store.root / "candidate_inbox"
    if not inbox.is_dir() or state["state"] != "COLLECTING":
        return state
    candidates = [_load_json(path) for path in inbox.glob("*.json")]
    candidates.sort(key=lambda item: (str(item.get("jump_at")), str(item.get("captured_at")), str(item.get("race_id"))))
    target = store.protocol["cohort"]["target_races"]
    for candidate in candidates:
        if len(state["predictions"]) >= target or state["state"] != "COLLECTING":
            break
        try:
            member_id, receipt, event = _prediction_receipt(
                candidate, state, store.protocol, model, preprocessing, observed_at
            )
        except (RuntimeEvidenceError, FinalizationError) as exc:
            source_sha = candidate.get("source_receipt_sha256")
            race_id = candidate.get("race_id")
            if (
                isinstance(source_sha, str)
                and len(source_sha) == 64
                and all(character in "0123456789abcdef" for character in source_sha)
                and isinstance(race_id, str)
                and race_id
                and state["state"] == "COLLECTING"
            ):
                state = store.append(
                    state,
                    rows,
                    {
                        "event_id": f"rejection-{sha256_bytes(canonical_bytes(candidate))}",
                        "type": "CANDIDATE_REJECTED",
                        "candidate_id": sha256_bytes(canonical_bytes(candidate)),
                        "race_id": race_id,
                        "observed_at": observed_at.isoformat(),
                        "reason": _candidate_rejection_reason(exc),
                        "source_receipt_sha256": source_sha,
                        "detail": str(exc),
                    },
                )
            else:
                continue
            continue
        if member_id in state["predictions"]:
            existing_path = store.root / "predictions" / f"{member_id}.json"
            if not _sealed_receipt_matches(
                existing_path,
                state["predictions"][member_id]["prediction_receipt_sha256"],
                receipt,
            ):
                return _abort(store, state, rows, f"conflicting_prediction_receipt:{member_id}")
            continue
        if event["race_id"] in state["race_members"]:
            return _abort(store, state, rows, f"duplicate_race_membership:{event['race_id']}")
        try:
            receipt_sha = _write_once(store.root / "predictions" / f"{member_id}.json", receipt)
        except RuntimeEvidenceError as exc:
            return _abort(store, state, rows, f"prediction_receipt_write_conflict:{member_id}:{exc}")
        event["prediction_receipt_sha256"] = receipt_sha
        state = store.append(state, rows, event)
    return state


def _result_receipt(result: Mapping[str, Any], prediction: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if result.get("schema_version") != "forward_overround_successor_official_result_v1":
        raise RuntimeEvidenceError("result_schema_mismatch")
    if result.get("source") != "thedogs" or result.get("official") is not True:
        raise RuntimeEvidenceError("result_official_source_ambiguous")
    member_id = prediction["member_id"]
    if result.get("member_id") != member_id or result.get("race_id") != prediction["race_id"]:
        raise RuntimeEvidenceError("result_member_identity_conflict")
    runners = result.get("runners")
    if not isinstance(runners, list) or runner_set_sha256(runners) != prediction["runner_set_sha256"]:
        raise RuntimeEvidenceError("result_runner_set_conflict")
    validate_finish_positions(runners)
    winners = [runner for runner in runners if runner.get("finish_position") == 1]
    if len(winners) != 1 or result.get("winner_box") != winners[0].get("box_number"):
        raise RuntimeEvidenceError("result_winner_conflict")
    captured_at_source = _text(result.get("captured_at"), "captured_at")
    captured_at = _aware_time(captured_at_source, "captured_at")
    if captured_at <= _aware_time(prediction["jump_at"], "jump_at"):
        raise RuntimeEvidenceError("result_not_observed_after_jump")
    receipt = {
        "schema_version": "forward_overround_successor_result_receipt_v1",
        "member_id": member_id,
        "race_id": prediction["race_id"],
        "captured_at": captured_at_source,
        "source": "thedogs",
        "official": True,
        "source_receipt_sha256": _sha(result.get("source_receipt_sha256"), "source_receipt_sha256"),
        "runner_set_sha256": prediction["runner_set_sha256"],
        "winner_box": result["winner_box"],
        "runners": sorted(runners, key=lambda row: row["box_number"]),
    }
    event = {
        "event_id": f"result-{member_id}",
        "type": "RESULT_APPENDED",
        "member_id": member_id,
        "race_id": prediction["race_id"],
        "runner_set_sha256": prediction["runner_set_sha256"],
        "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
        "winner_box": result["winner_box"],
    }
    return receipt, event


def _seal_results(store: CohortStore, state: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    inbox = store.root / "result_inbox"
    if not inbox.is_dir() or not state["predictions"]:
        return state
    result_payloads = [_load_json(path) for path in inbox.glob("*.json")]
    result_payloads.sort(key=lambda item: (str(item.get("captured_at")), str(item.get("race_id"))))
    for result in result_payloads:
        member_id = result.get("member_id")
        if not isinstance(member_id, str) or member_id not in state["predictions"]:
            continue
        prediction_path = store.root / "predictions" / f"{member_id}.json"
        prediction = _load_json(prediction_path)
        try:
            receipt, event = _result_receipt(result, prediction)
        except (RuntimeEvidenceError, FinalizationError) as exc:
            return _abort(store, state, rows, f"result_identity_or_winner_conflict:{member_id}:{exc}")
        if member_id in state["results"]:
            existing_path = store.root / "results" / f"{member_id}.json"
            if not _sealed_receipt_matches(
                existing_path,
                state["results"][member_id]["result_receipt_sha256"],
                receipt,
            ):
                return _abort(store, state, rows, f"conflicting_result_receipt:{member_id}")
            continue
        try:
            receipt_sha = _write_once(store.root / "results" / f"{member_id}.json", receipt)
        except RuntimeEvidenceError as exc:
            return _abort(store, state, rows, f"result_receipt_write_conflict:{member_id}:{exc}")
        event["result_receipt_sha256"] = receipt_sha
        state = store.append(state, rows, event)
    return state


def _finalize_if_ready(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    protocol_path: Path,
    asset_dir: Path,
) -> dict[str, Any]:
    if state["state"] not in {"READY_TO_FINALIZE", "FINALIZATION_LOCKED", "FINALIZED_SCORED"}:
        return state
    final_paths = [
        store.root / "METRICS.json",
        store.root / "FINAL_REPORT.json",
        store.root / "CONSUMED.json",
    ]
    if state["state"] == "READY_TO_FINALIZE":
        if any(path.exists() for path in final_paths):
            return _abort(store, state, rows, "finalization_artifact_preexists_request")
        state = store.append(
            state,
            rows,
            {"event_id": "fixed-n-finalize-requested", "type": "FINALIZE_REQUESTED"},
        )

    metrics_path = store.root / "METRICS.json"
    if metrics_path.is_file():
        try:
            report = _load_json(metrics_path)
        except RuntimeEvidenceError as exc:
            if state["state"] == "FINALIZATION_LOCKED":
                return _abort(store, state, rows, f"finalizer_evidence_failure:{exc}")
            raise
        metrics_sha = sha256_file(metrics_path)
        confirmation = store.protocol["evaluation"]["confirmation_rule"]
        valid_verdicts = {
            confirmation["valid_evidence_gate_pass_verdict"],
            confirmation["valid_evidence_gate_failure_verdict"],
        }
        if (
            report.get("schema_version") != "forward_overround_successor_final_report_v1"
            or report.get("verdict") not in valid_verdicts
            or report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
            or report.get("member_manifest_sha256")
            != state["finalization_member_manifest_sha256"]
            or report.get("race_count") != store.protocol["cohort"]["target_races"]
            or report.get("identical_races_compared") is not True
            or report.get("score_invocation_count") != 1
            or not isinstance(report.get("metrics"), Mapping)
            or report.get("profitability")
            != {"roi_computed": False, "betting_analysis_performed": False}
        ):
            if state["state"] == "FINALIZATION_LOCKED":
                return _abort(store, state, rows, "finalizer_evidence_failure:metrics_receipt_invalid")
            raise RuntimeEvidenceError("metrics_receipt_invalid_after_score_commit")
    elif state["state"] == "FINALIZATION_LOCKED":
        try:
            report = finalize(state, store.root, protocol_path, asset_dir)
        except FinalizationError as exc:
            return _abort(store, state, rows, f"finalizer_evidence_failure:{exc}")
        metrics_sha = _write_once(metrics_path, report)
    else:
        raise RuntimeEvidenceError("metrics_receipt_missing_after_score_commit")

    if state["state"] == "FINALIZATION_LOCKED":
        state = store.append(
            state,
            rows,
            {
                "event_id": f"paired-score-{metrics_sha}",
                "type": "PAIRED_SCORE_COMMITTED",
                "member_manifest_sha256": state["finalization_member_manifest_sha256"],
                "metrics_receipt_sha256": metrics_sha,
            },
        )
    elif state.get("metrics_receipt_sha256") != metrics_sha:
        raise RuntimeEvidenceError("metrics_receipt_hash_drift_after_score_commit")

    report_sha = _write_once(store.root / "FINAL_REPORT.json", report)
    _write_once(
        store.root / "CONSUMED.json",
        {
            "schema_version": "forward_overround_successor_consumed_v1",
            "verdict": report["verdict"],
            "final_report_sha256": report_sha,
            "metrics_receipt_sha256": metrics_sha,
            "member_manifest_sha256": state["finalization_member_manifest_sha256"],
        },
    )
    return state


def run_once(
    cohort_root: Path,
    protocol_path: Path,
    asset_dir: Path,
    finalizer_path: Path,
    service_unit_path: Path,
    *,
    observation_clock: ObservationClock = _system_clock,
) -> dict[str, Any]:
    terminal = _resume_independent_terminal(cohort_root)
    if terminal is not None:
        return terminal
    journal_path = cohort_root / "EVENTS.jsonl"
    try:
        protocol = _load_json(protocol_path)
    except RuntimeEvidenceError as exc:
        if journal_path.is_file() and journal_path.stat().st_size > 0:
            return _independent_terminal(cohort_root, f"untrusted_protocol:{exc}")
        raise
    if (
        sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256
        and journal_path.is_file()
        and journal_path.stat().st_size > 0
    ):
        return _independent_terminal(cohort_root, "untrusted_protocol:successor_protocol_hash_drift")
    protocol["_document_sha256"] = EXPECTED_PROTOCOL_SHA256
    try:
        store = CohortStore(cohort_root, protocol)
    except RuntimeEvidenceError as exc:
        if journal_path.is_file() and journal_path.stat().st_size > 0:
            return _independent_terminal(cohort_root, f"untrusted_cohort_store:{exc}")
        raise
    with _exclusive_lock(cohort_root / "runtime" / "successor.lock"):
        observed_at = _observe_utc(observation_clock)
        try:
            state, rows = store.replay(verify_receipts=False)
        except (RuntimeEvidenceError, OSError, KeyError, TypeError, ValueError) as exc:
            return _independent_terminal(cohort_root, f"untrusted_journal_or_protocol:{exc}")
        try:
            store.verify_receipts(state)
        except RuntimeEvidenceError as exc:
            if state["state"] in TERMINAL_STATES:
                raise
            try:
                state = _abort(store, state, rows, f"sealed_receipt_validation_failed:{exc}")
            except (RuntimeEvidenceError, OSError) as append_exc:
                return _independent_terminal(
                    cohort_root,
                    f"sealed_receipt_validation_failed:{exc}:fatal_append_failed:{append_exc}",
                )
            status = public_snapshot(state)
            status["target_races"] = protocol["cohort"]["target_races"]
            status["interim_aggregate_performance_emitted"] = False
            _write_status(cohort_root / "STATUS.json", status)
            return status
        try:
            _, model, preprocessing, _, _ = load_frozen_assets(protocol_path, asset_dir)
        except (FinalizationError, OSError, json.JSONDecodeError) as exc:
            if rows and state["state"] not in TERMINAL_STATES:
                state = _abort(store, state, rows, f"frozen_asset_validation_failed:{exc}")
                status = public_snapshot(state)
                status["target_races"] = protocol["cohort"]["target_races"]
                status["interim_aggregate_performance_emitted"] = False
                _write_status(cohort_root / "STATUS.json", status)
                return status
            raise
        identity = runtime_identity(finalizer_path, service_unit_path)
        if not rows:
            state = _authorize(store, state, rows, identity, observed_at)
        if state["state"] not in TERMINAL_STATES:
            state = _check_runtime_admission(store, state, rows, identity, observed_at)
        if state["state"] == "COLLECTING":
            state = _seal_candidates(store, state, rows, model, preprocessing, observed_at)
        if state["state"] not in TERMINAL_STATES:
            state = _seal_results(store, state, rows)
        if state["state"] in {"READY_TO_FINALIZE", "FINALIZATION_LOCKED", "FINALIZED_SCORED"}:
            state = _finalize_if_ready(store, state, rows, protocol_path, asset_dir)
        status = public_snapshot(state)
        status["target_races"] = protocol["cohort"]["target_races"]
        status["interim_aggregate_performance_emitted"] = False
        _write_status(cohort_root / "STATUS.json", status)
        return status


def read_status(cohort_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_bytes())
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeEvidenceError("successor_protocol_hash_drift")
    protocol["_document_sha256"] = EXPECTED_PROTOCOL_SHA256
    store = CohortStore(cohort_root, protocol)
    state, _ = store.replay()
    return public_snapshot(state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-once")
    run.add_argument("--cohort-root", required=True, type=Path)
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--asset-dir", required=True, type=Path)
    run.add_argument("--finalizer", required=True, type=Path)
    run.add_argument("--service-unit", required=True, type=Path)
    status = subparsers.add_parser("status")
    status.add_argument("--cohort-root", required=True, type=Path)
    status.add_argument("--protocol", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run-once":
        status = run_once(
            args.cohort_root,
            args.protocol,
            args.asset_dir,
            args.finalizer,
            args.service_unit,
        )
    else:
        status = read_status(args.cohort_root, args.protocol)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
