#!/usr/bin/env python3
"""Append-only collector/sealer runtime for the prepared overround successor.

The runtime cannot create or authorize a cohort. It operates only when an
existing cohort directory contains a separately supplied ACTIVATION.json.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
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
STATE_MACHINE_PATH = Path(__file__).with_name("forward_overround_successor_state_machine.py")
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


def _exact_keys(payload: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        raise RuntimeEvidenceError(f"invalid_keys:{field}")


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeEvidenceError(f"invalid_nonnegative_int:{field}")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeEvidenceError(f"invalid_number:{field}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeEvidenceError(f"non_finite_number:{field}")
    return number


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


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, payload: Mapping[str, Any]) -> str:
    raw = canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise RuntimeEvidenceError(f"write_once_existing_read_failed:{path}:{exc}") from exc
            if existing != raw:
                raise RuntimeEvidenceError(f"write_once_conflict:{path}") from None
        else:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
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
        _fsync_directory(path.parent)
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


def runtime_identity(
    finalizer_path: Path,
    service_unit_path: Path,
    state_machine_path: Path = STATE_MACHINE_PATH,
) -> dict[str, str]:
    paths = {
        "capture_code_sha256": Path(__file__).resolve(),
        "finalizer_code_sha256": finalizer_path.resolve(),
        "capture_unit_sha256": service_unit_path.resolve(),
        "state_machine_code_sha256": state_machine_path.resolve(),
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
        "state_machine_code_sha256": _sha(
            receipt.get("state_machine_code_sha256"), "state_machine_code_sha256"
        ),
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
    _discard_unconsumed_score_artifacts(store.root)
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
    _require_terminal_commit(store.root)
    return state


def _validate_metric_summary(summary: Any, field: str) -> None:
    if not isinstance(summary, Mapping):
        raise RuntimeEvidenceError(f"mapping_required:{field}")
    _exact_keys(
        summary,
        {
            "race_count",
            "runner_count",
            "mean_multiclass_race_log_loss",
            "mean_multiclass_brier",
            "runner_calibration",
            "top_1_accuracy",
            "mean_winner_rank",
            "mean_reciprocal_winner_rank",
        },
        field,
    )
    if _nonnegative_int(summary["race_count"], f"{field}.race_count") != 1000:
        raise RuntimeEvidenceError(f"invalid_race_count:{field}")
    runner_count = _nonnegative_int(summary["runner_count"], f"{field}.runner_count")
    if runner_count < 2000:
        raise RuntimeEvidenceError(f"invalid_runner_count:{field}")
    for name in (
        "mean_multiclass_race_log_loss",
        "mean_multiclass_brier",
        "top_1_accuracy",
        "mean_winner_rank",
        "mean_reciprocal_winner_rank",
    ):
        _finite_number(summary[name], f"{field}.{name}")
    calibration = summary["runner_calibration"]
    if not isinstance(calibration, Mapping):
        raise RuntimeEvidenceError(f"mapping_required:{field}.runner_calibration")
    _exact_keys(calibration, {"bands", "ece"}, f"{field}.runner_calibration")
    _finite_number(calibration["ece"], f"{field}.runner_calibration.ece")
    bands = calibration["bands"]
    if not isinstance(bands, list) or len(bands) != 5:
        raise RuntimeEvidenceError(f"invalid_calibration_bands:{field}")
    band_runner_count = 0
    for index, band in enumerate(bands):
        band_field = f"{field}.runner_calibration.bands[{index}]"
        if not isinstance(band, Mapping):
            raise RuntimeEvidenceError(f"mapping_required:{band_field}")
        _exact_keys(
            band,
            {
                "lower",
                "upper",
                "runner_count",
                "mean_probability",
                "observed_win_rate",
                "observed_minus_probability",
            },
            band_field,
        )
        _finite_number(band["lower"], f"{band_field}.lower")
        _finite_number(band["upper"], f"{band_field}.upper")
        count = _nonnegative_int(band["runner_count"], f"{band_field}.runner_count")
        band_runner_count += count
        for name in ("mean_probability", "observed_win_rate", "observed_minus_probability"):
            value = band[name]
            if count == 0:
                if value is not None:
                    raise RuntimeEvidenceError(f"nonempty_zero_count_band:{band_field}.{name}")
            else:
                _finite_number(value, f"{band_field}.{name}")
    if band_runner_count != runner_count:
        raise RuntimeEvidenceError(f"calibration_runner_count_mismatch:{field}")


def _validate_interval(
    value: Any, field: str, *, seed: int, cluster_count_required: bool = False
) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeEvidenceError(f"mapping_required:{field}")
    expected_keys = {"replicates", "seed", "lower", "upper"}
    if cluster_count_required:
        expected_keys.add("cluster_count")
    _exact_keys(value, expected_keys, field)
    if value["replicates"] != 20000 or value["seed"] != seed:
        raise RuntimeEvidenceError(f"frozen_interval_contract_mismatch:{field}")
    lower = _finite_number(value["lower"], f"{field}.lower")
    upper = _finite_number(value["upper"], f"{field}.upper")
    if lower > upper:
        raise RuntimeEvidenceError(f"invalid_interval_order:{field}")
    if cluster_count_required and _nonnegative_int(
        value["cluster_count"], f"{field}.cluster_count"
    ) == 0:
        raise RuntimeEvidenceError(f"invalid_cluster_count:{field}")


def _validate_scored_report(report: Mapping[str, Any]) -> None:
    _exact_keys(
        report,
        {
            "schema_version",
            "verdict",
            "protocol_sha256",
            "member_manifest_sha256",
            "race_count",
            "identical_races_compared",
            "score_invocation_count",
            "metrics",
            "profitability",
        },
        "scored_report",
    )
    if report.get("schema_version") != "forward_overround_successor_final_report_v1":
        raise RuntimeEvidenceError("scored_report_schema_invalid")
    if report.get("verdict") not in {
        "FORWARD_OVERROUND_SIGNAL_CONFIRMED",
        "FORWARD_OVERROUND_SIGNAL_NOT_CONFIRMED",
    }:
        raise RuntimeEvidenceError("scored_report_verdict_invalid")
    if (
        report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or report.get("race_count") != 1000
        or report.get("identical_races_compared") is not True
        or report.get("score_invocation_count") != 1
        or report.get("profitability")
        != {"roi_computed": False, "betting_analysis_performed": False}
    ):
        raise RuntimeEvidenceError("scored_report_frozen_contract_invalid")
    _sha(report.get("member_manifest_sha256"), "scored_report.member_manifest_sha256")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeEvidenceError("scored_report_metrics_invalid")
    _exact_keys(
        metrics,
        {
            "primary",
            "baseline",
            "candidate",
            "race_bootstrap_95pct",
            "race_date_cluster_bootstrap_95pct",
            "chronological_blocks",
            "negative_chronological_blocks",
        },
        "scored_report.metrics",
    )
    _validate_metric_summary(metrics["baseline"], "scored_report.metrics.baseline")
    _validate_metric_summary(metrics["candidate"], "scored_report.metrics.candidate")
    primary = metrics["primary"]
    if not isinstance(primary, Mapping):
        raise RuntimeEvidenceError("scored_report_primary_invalid")
    _exact_keys(
        primary,
        {"name", "candidate_minus_baseline", "baseline", "candidate"},
        "scored_report.metrics.primary",
    )
    if primary["name"] != "mean_multiclass_race_log_loss":
        raise RuntimeEvidenceError("scored_report_primary_name_invalid")
    baseline = _finite_number(primary["baseline"], "scored_report.metrics.primary.baseline")
    candidate = _finite_number(primary["candidate"], "scored_report.metrics.primary.candidate")
    delta = _finite_number(
        primary["candidate_minus_baseline"],
        "scored_report.metrics.primary.candidate_minus_baseline",
    )
    if not math.isclose(delta, candidate - baseline, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeEvidenceError("scored_report_primary_delta_invalid")
    if baseline != metrics["baseline"]["mean_multiclass_race_log_loss"]:
        raise RuntimeEvidenceError("scored_report_baseline_primary_mismatch")
    if candidate != metrics["candidate"]["mean_multiclass_race_log_loss"]:
        raise RuntimeEvidenceError("scored_report_candidate_primary_mismatch")
    _validate_interval(
        metrics["race_bootstrap_95pct"],
        "scored_report.metrics.race_bootstrap_95pct",
        seed=20260817,
    )
    _validate_interval(
        metrics["race_date_cluster_bootstrap_95pct"],
        "scored_report.metrics.race_date_cluster_bootstrap_95pct",
        seed=20260818,
        cluster_count_required=True,
    )
    blocks = metrics["chronological_blocks"]
    if not isinstance(blocks, list) or len(blocks) != 5:
        raise RuntimeEvidenceError("scored_report_chronological_blocks_invalid")
    negative_blocks = 0
    for index, block in enumerate(blocks, start=1):
        if not isinstance(block, Mapping):
            raise RuntimeEvidenceError("scored_report_chronological_block_invalid")
        _exact_keys(
            block,
            {"block", "race_count", "mean_log_loss_delta"},
            f"scored_report.metrics.chronological_blocks[{index - 1}]",
        )
        delta_value = _finite_number(
            block["mean_log_loss_delta"],
            f"scored_report.metrics.chronological_blocks[{index - 1}].mean_log_loss_delta",
        )
        if block["block"] != index or block["race_count"] != 200:
            raise RuntimeEvidenceError("scored_report_chronological_block_contract_invalid")
        negative_blocks += int(delta_value < 0.0)
    reported_negative_blocks = _nonnegative_int(
        metrics["negative_chronological_blocks"],
        "scored_report.metrics.negative_chronological_blocks",
    )
    if reported_negative_blocks != negative_blocks:
        raise RuntimeEvidenceError("scored_report_negative_block_count_invalid")
    confirmed = (
        delta < 0.0
        and metrics["race_bootstrap_95pct"]["upper"] < 0.0
        and metrics["race_date_cluster_bootstrap_95pct"]["upper"] < 0.0
        and negative_blocks >= 4
    )
    expected_verdict = (
        "FORWARD_OVERROUND_SIGNAL_CONFIRMED"
        if confirmed
        else "FORWARD_OVERROUND_SIGNAL_NOT_CONFIRMED"
    )
    if report["verdict"] != expected_verdict:
        raise RuntimeEvidenceError("scored_report_verdict_metrics_mismatch")


def _validate_independent_sentinel(sentinel: Mapping[str, Any]) -> None:
    _exact_keys(
        sentinel,
        {
            "schema_version",
            "state",
            "blocking_reason",
            "protocol_sha256",
            "journal_mutated",
            "metrics",
            "score_invocation_count",
        },
        "independent_terminal_sentinel",
    )
    if (
        sentinel.get("schema_version")
        != "forward_overround_successor_independent_terminal_v1"
        or sentinel.get("state") != "FINALIZED_ABORTED_NO_METRICS"
        or sentinel.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or sentinel.get("journal_mutated") is not False
        or sentinel.get("metrics") is not None
    ):
        raise RuntimeEvidenceError("independent_terminal_sentinel_invalid")
    _text(sentinel.get("blocking_reason"), "sentinel.blocking_reason")
    _nonnegative_int(sentinel.get("score_invocation_count"), "sentinel.score_invocation_count")


def _validate_no_metrics_report(
    report: Mapping[str, Any],
    consumed: Mapping[str, Any],
    sentinel: Mapping[str, Any] | None,
) -> None:
    common = {
        "schema_version",
        "verdict",
        "blocking_reason",
        "protocol_sha256",
        "metrics",
        "sealed_prediction_races",
        "approved_result_races",
        "score_invocation_count",
    }
    expected_report_keys = common | (
        {"evidence_counts_trusted", "terminal_sentinel_sha256"}
        if sentinel is not None
        else set()
    )
    _exact_keys(report, expected_report_keys, "no_metrics_report")
    if (
        report.get("schema_version") != "forward_overround_successor_final_report_v1"
        or report.get("verdict") != "BLOCKED_FORWARD_EVIDENCE"
        or report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or report.get("metrics") is not None
    ):
        raise RuntimeEvidenceError("no_metrics_report_schema_invalid")
    _text(report.get("blocking_reason"), "no_metrics_report.blocking_reason")
    score_count = _nonnegative_int(
        report.get("score_invocation_count"), "no_metrics_report.score_invocation_count"
    )
    if sentinel is None:
        _nonnegative_int(
            report.get("sealed_prediction_races"), "no_metrics_report.sealed_prediction_races"
        )
        _nonnegative_int(
            report.get("approved_result_races"), "no_metrics_report.approved_result_races"
        )
        _exact_keys(
            consumed,
            {"schema_version", "verdict", "final_report_sha256"},
            "no_metrics_consumed",
        )
        return
    _validate_independent_sentinel(sentinel)
    if (
        report.get("evidence_counts_trusted") is not False
        or report.get("sealed_prediction_races") is not None
        or report.get("approved_result_races") is not None
        or report.get("blocking_reason") != sentinel.get("blocking_reason")
        or score_count != sentinel.get("score_invocation_count")
        or report.get("terminal_sentinel_sha256")
        != consumed.get("terminal_sentinel_sha256")
    ):
        raise RuntimeEvidenceError("independent_terminal_report_binding_invalid")
    _exact_keys(
        consumed,
        {
            "schema_version",
            "verdict",
            "final_report_sha256",
            "terminal_sentinel_sha256",
        },
        "independent_terminal_consumed",
    )


def _validated_terminal_commit(cohort_root: Path) -> dict[str, Any] | None:
    consumed_path = cohort_root / "CONSUMED.json"
    if not consumed_path.exists():
        return None
    consumed = _load_json(consumed_path)
    if consumed.get("schema_version") != "forward_overround_successor_consumed_v1":
        raise RuntimeEvidenceError("consumed_schema_invalid")
    verdict = _text(consumed.get("verdict"), "consumed.verdict")
    final_report_sha256 = _sha(
        consumed.get("final_report_sha256"), "consumed.final_report_sha256"
    )
    final_report_path = cohort_root / "FINAL_REPORT.json"
    if (
        not final_report_path.is_file()
        or sha256_file(final_report_path) != final_report_sha256
    ):
        raise RuntimeEvidenceError("consumed_final_report_hash_invalid")
    report = _load_json(final_report_path)
    if report.get("verdict") != verdict:
        raise RuntimeEvidenceError("consumed_final_report_verdict_invalid")

    metrics_sha256 = consumed.get("metrics_receipt_sha256")
    metrics_path = cohort_root / "METRICS.json"
    if metrics_sha256 is None:
        if report.get("metrics") is not None or metrics_path.exists():
            raise RuntimeEvidenceError("no_metrics_commit_contains_metrics")
        terminal_sentinel_sha256 = consumed.get("terminal_sentinel_sha256")
        sentinel: dict[str, Any] | None = None
        if terminal_sentinel_sha256 is not None:
            terminal_sentinel_sha256 = _sha(
                terminal_sentinel_sha256, "consumed.terminal_sentinel_sha256"
            )
            sentinel_path = cohort_root / "TERMINAL_SENTINEL.json"
            if (
                not sentinel_path.is_file()
                or sha256_file(sentinel_path) != terminal_sentinel_sha256
            ):
                raise RuntimeEvidenceError("consumed_terminal_sentinel_hash_invalid")
            sentinel = _load_json(sentinel_path)
        _validate_no_metrics_report(report, consumed, sentinel)
    else:
        _exact_keys(
            consumed,
            {
                "schema_version",
                "verdict",
                "final_report_sha256",
                "metrics_receipt_sha256",
                "member_manifest_sha256",
            },
            "scored_consumed",
        )
        metrics_sha256 = _sha(metrics_sha256, "consumed.metrics_receipt_sha256")
        member_manifest_sha256 = _sha(
            consumed.get("member_manifest_sha256"),
            "consumed.member_manifest_sha256",
        )
        if not metrics_path.is_file() or sha256_file(metrics_path) != metrics_sha256:
            raise RuntimeEvidenceError("consumed_metrics_hash_invalid")
        metrics_report = _load_json(metrics_path)
        if metrics_report != report or report.get("member_manifest_sha256") != member_manifest_sha256:
            raise RuntimeEvidenceError("consumed_metrics_schema_invalid")
        _validate_scored_report(report)
    return consumed


def _require_terminal_commit(cohort_root: Path) -> dict[str, Any]:
    consumed = _validated_terminal_commit(cohort_root)
    if consumed is None:
        raise RuntimeEvidenceError("terminal_commit_marker_absent_after_publication")
    return consumed


def _discard_unconsumed_score_artifacts(cohort_root: Path) -> None:
    """Remove score publications unless a complete cross-hash commit validates."""
    invalid_consumed = False
    try:
        if _validated_terminal_commit(cohort_root) is not None:
            return
    except RuntimeEvidenceError:
        invalid_consumed = True
    names = ["METRICS.json", "FINAL_REPORT.json"]
    if invalid_consumed:
        names.append("CONSUMED.json")
    removed = False
    for name in names:
        try:
            (cohort_root / name).unlink()
        except FileNotFoundError:
            continue
        else:
            removed = True
    if removed:
        _fsync_directory(cohort_root)


def _discard_terminal_publication(cohort_root: Path) -> None:
    """Remove a terminal publication proven inconsistent with durable state."""
    removed = False
    for name in ("METRICS.json", "FINAL_REPORT.json", "CONSUMED.json"):
        try:
            (cohort_root / name).unlink()
        except FileNotFoundError:
            continue
        else:
            removed = True
    if removed:
        _fsync_directory(cohort_root)


def _discard_invalid_independent_terminal(cohort_root: Path) -> None:
    """Remove an invalid sentinel and every publication that depends on it."""
    _discard_terminal_publication(cohort_root)
    try:
        (cohort_root / "TERMINAL_SENTINEL.json").unlink()
    except FileNotFoundError:
        return
    _fsync_directory(cohort_root)


def _preserve_commit_without_unreferenced_sentinel(cohort_root: Path) -> bool:
    """Drop only a stray invalid sentinel beside an otherwise valid commit."""
    try:
        committed = _validated_terminal_commit(cohort_root)
    except RuntimeEvidenceError:
        return False
    if committed is None or committed.get("terminal_sentinel_sha256") is not None:
        return False
    try:
        (cohort_root / "TERMINAL_SENTINEL.json").unlink()
    except FileNotFoundError:
        return True
    _fsync_directory(cohort_root)
    return True


def _resume_or_recover_independent_terminal(cohort_root: Path) -> dict[str, Any] | None:
    """Run under the cohort lock and preserve any already valid commit."""
    try:
        return _resume_independent_terminal(cohort_root)
    except (RuntimeEvidenceError, OSError):
        if _preserve_commit_without_unreferenced_sentinel(cohort_root):
            return None
        _discard_invalid_independent_terminal(cohort_root)
        return _independent_terminal(
            cohort_root, "independent_terminal_sentinel_invalid"
        )


def _independent_terminal(
    cohort_root: Path,
    reason: str,
    *,
    score_invocation_count: int = 0,
) -> dict[str, Any]:
    """Seal an untrusted-store failure without reading or changing its journal."""
    _discard_unconsumed_score_artifacts(cohort_root)
    sentinel = {
        "schema_version": "forward_overround_successor_independent_terminal_v1",
        "state": "FINALIZED_ABORTED_NO_METRICS",
        "blocking_reason": reason,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "journal_mutated": False,
        "metrics": None,
        "score_invocation_count": score_invocation_count,
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
        "score_invocation_count": score_invocation_count,
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
    _require_terminal_commit(cohort_root)
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
        "score_invocation_count": score_invocation_count,
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
    _validate_independent_sentinel(sentinel)
    return _independent_terminal(
        cohort_root,
        sentinel["blocking_reason"],
        score_invocation_count=sentinel["score_invocation_count"],
    )


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
    if active["state_machine_code_sha256"] != identity["state_machine_code_sha256"]:
        return _abort(store, state, rows, "state_machine_code_hash_drift")
    capture_drift = any(
        active[field] != identity[field]
        for field in ("capture_code_sha256", "capture_unit_sha256")
    )
    if capture_drift:
        if state["state"] == "COLLECTING":
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
        elif state["state"] != "ADMISSION_PAUSED":
            return _abort(
                store,
                state,
                rows,
                "capture_code_or_unit_hash_drift_after_membership_freeze",
            )
    if state["state"] == "ADMISSION_PAUSED":
        state = _accept_reviewed_admission(store, state, rows, identity, observed_at)
    return state


def _prediction_receipt(
    candidate: Mapping[str, Any],
    candidate_file: str,
    candidate_content_sha256: str,
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
        or observed_at >= jump_at
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
        "candidate_file": candidate_file,
        "candidate_content_sha256": candidate_content_sha256,
        "captured_at": captured_at_source,
        "observed_at": observed_at.isoformat(),
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
        "candidate_file": candidate_file,
        "candidate_content_sha256": candidate_content_sha256,
        "admission_id": state["active_admission_id"],
        "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
        "observed_at": observed_at.isoformat(),
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


def _optional_candidate_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value and value == value.strip() else None


def _optional_candidate_sha(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        return None
    return value


def _candidate_file_identity(path: Path, raw: bytes) -> tuple[str, str]:
    content_sha256 = sha256_bytes(raw)
    candidate_id = sha256_bytes(
        canonical_bytes(
            {
                "candidate_file": path.name,
                "candidate_content_sha256": content_sha256,
            }
        )
    )
    return candidate_id, content_sha256


def _append_candidate_rejection(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    path: Path,
    candidate_id: str,
    content_sha256: str,
    candidate: Mapping[str, Any] | None,
    observed_at: datetime,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    return store.append(
        state,
        rows,
        {
            "event_id": f"rejection-{candidate_id}",
            "type": "CANDIDATE_REJECTED",
            "candidate_id": candidate_id,
            "candidate_file": path.name,
            "candidate_content_sha256": content_sha256,
            "race_id": (
                _optional_candidate_text(candidate.get("race_id"))
                if candidate is not None
                else None
            ),
            "observed_at": observed_at.isoformat(),
            "reason": reason,
            "source_receipt_sha256": (
                _optional_candidate_sha(candidate.get("source_receipt_sha256"))
                if candidate is not None
                else None
            ),
            "detail": detail,
        },
    )


def _seal_candidates(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    model: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    observation_clock: ObservationClock,
) -> dict[str, Any]:
    inbox = store.root / "candidate_inbox"
    if not inbox.is_dir() or state["state"] != "COLLECTING":
        return state
    candidate_items: list[
        tuple[Path, str, str, dict[str, Any] | None, RuntimeEvidenceError | None]
    ] = []
    for path in sorted(inbox.glob("*.json"), key=lambda item: item.name):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return _abort(store, state, rows, f"candidate_inbox_read_failed:{path.name}:{exc}")
        candidate_id, content_sha256 = _candidate_file_identity(path, raw)
        prior_accepted_content_sha256 = state["candidate_files"].get(path.name)
        if prior_accepted_content_sha256 is not None:
            if prior_accepted_content_sha256 != content_sha256:
                return _abort(store, state, rows, f"candidate_file_identity_changed:{path.name}")
        prior_candidate_id = state["rejection_files"].get(path.name)
        if prior_candidate_id is not None:
            if prior_candidate_id != candidate_id:
                return _abort(store, state, rows, f"candidate_file_identity_changed:{path.name}")
            continue
        try:
            decoded = json.loads(raw)
            if not isinstance(decoded, dict):
                raise RuntimeEvidenceError("candidate_json_object_required")
        except (json.JSONDecodeError, RuntimeEvidenceError, UnicodeDecodeError) as exc:
            error = RuntimeEvidenceError(f"candidate_json_invalid:{type(exc).__name__}")
            candidate_items.append((path, candidate_id, content_sha256, None, error))
        else:
            candidate_items.append((path, candidate_id, content_sha256, decoded, None))
    candidate_items.sort(
        key=lambda item: (
            str(item[3].get("jump_at")) if item[3] is not None else "",
            str(item[3].get("captured_at")) if item[3] is not None else "",
            str(item[3].get("race_id")) if item[3] is not None else "",
            item[0].name,
        )
    )
    target = store.protocol["cohort"]["target_races"]
    for path, candidate_id, content_sha256, candidate, parse_error in candidate_items:
        if len(state["predictions"]) >= target or state["state"] != "COLLECTING":
            break
        if parse_error is not None or candidate is None:
            observed_at = _observe_utc(observation_clock)
            state = _append_candidate_rejection(
                store,
                state,
                rows,
                path=path,
                candidate_id=candidate_id,
                content_sha256=content_sha256,
                candidate=None,
                observed_at=observed_at,
                reason="candidate_incomplete_field",
                detail=str(parse_error),
            )
            continue
        receipt_observed_at: datetime | None = None
        existing_race_member = state["race_members"].get(candidate.get("race_id"))
        if isinstance(existing_race_member, str):
            existing_receipt_path = (
                store.root / "predictions" / f"{existing_race_member}.json"
            )
            try:
                existing_receipt = _load_json(existing_receipt_path)
                receipt_observed_at = _aware_time(
                    existing_receipt.get("observed_at"), "observed_at"
                ).astimezone(timezone.utc)
            except RuntimeEvidenceError as exc:
                return _abort(
                    store,
                    state,
                    rows,
                    f"sealed_prediction_receipt_invalid:{existing_race_member}:{exc}",
                )
        observed_at = _observe_utc(observation_clock)
        if receipt_observed_at is None:
            receipt_observed_at = observed_at
        try:
            member_id, receipt, event = _prediction_receipt(
                candidate,
                path.name,
                content_sha256,
                state,
                store.protocol,
                model,
                preprocessing,
                receipt_observed_at,
            )
        except (RuntimeEvidenceError, FinalizationError) as exc:
            state = _append_candidate_rejection(
                store,
                state,
                rows,
                path=path,
                candidate_id=candidate_id,
                content_sha256=content_sha256,
                candidate=candidate,
                observed_at=observed_at,
                reason=_candidate_rejection_reason(exc),
                detail=str(exc),
            )
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
        receipt_sha = sha256_bytes(canonical_bytes(receipt))
        event["prediction_receipt_sha256"] = receipt_sha
        preview = apply_event(copy.deepcopy(state), event, store.protocol)
        if preview["state"] == "FINALIZED_ABORTED_NO_METRICS":
            state = store.append(state, rows, event)
            break
        try:
            written_sha = _write_once(store.root / "predictions" / f"{member_id}.json", receipt)
        except RuntimeEvidenceError as exc:
            return _abort(store, state, rows, f"prediction_receipt_write_conflict:{member_id}:{exc}")
        if written_sha != receipt_sha:
            return _abort(store, state, rows, f"prediction_receipt_hash_mismatch:{member_id}")
        state = store.append(state, rows, event)
    return state


def _result_receipt(
    result: Mapping[str, Any],
    prediction: Mapping[str, Any],
    observed_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    if captured_at > observed_at:
        raise RuntimeEvidenceError("result_observed_in_future")
    receipt = {
        "schema_version": "forward_overround_successor_result_receipt_v1",
        "member_id": member_id,
        "race_id": prediction["race_id"],
        "captured_at": captured_at_source,
        "observed_at": observed_at.isoformat(),
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
        "observed_at": observed_at.isoformat(),
        "winner_box": result["winner_box"],
    }
    return receipt, event


def _seal_results(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    observed_at: datetime,
) -> dict[str, Any]:
    inbox = store.root / "result_inbox"
    if not inbox.is_dir():
        return state
    result_items: list[tuple[Path, str, dict[str, Any]]] = []
    for path in sorted(inbox.glob("*.json"), key=lambda item: item.name):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return _abort(store, state, rows, f"result_inbox_read_failed:{path.name}:{exc}")
        content_sha256 = sha256_bytes(raw)
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _abort(
                store,
                state,
                rows,
                f"result_inbox_invalid_json:{path.name}:{content_sha256}",
            )
        if not isinstance(result, dict):
            return _abort(
                store,
                state,
                rows,
                f"result_inbox_object_required:{path.name}:{content_sha256}",
            )
        result_items.append((path, content_sha256, result))
    result_items.sort(
        key=lambda item: (
            str(item[2].get("captured_at")),
            str(item[2].get("race_id")),
            item[0].name,
        )
    )
    for path, content_sha256, result in result_items:
        member_id = result.get("member_id")
        if not isinstance(member_id, str) or member_id not in state["predictions"]:
            return _abort(
                store,
                state,
                rows,
                f"result_before_prediction_or_nonmember:{path.name}:{content_sha256}",
            )
        prediction_path = store.root / "predictions" / f"{member_id}.json"
        prediction = _load_json(prediction_path)
        receipt_observed_at = observed_at
        if member_id in state["results"]:
            existing_result_path = store.root / "results" / f"{member_id}.json"
            try:
                existing_result_receipt = _load_json(existing_result_path)
                receipt_observed_at = _aware_time(
                    existing_result_receipt.get("observed_at"), "observed_at"
                ).astimezone(timezone.utc)
            except RuntimeEvidenceError as exc:
                return _abort(
                    store,
                    state,
                    rows,
                    f"sealed_result_receipt_invalid:{member_id}:{exc}",
                )
        try:
            receipt, event = _result_receipt(result, prediction, receipt_observed_at)
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
        receipt_sha = sha256_bytes(canonical_bytes(receipt))
        event["result_receipt_sha256"] = receipt_sha
        preview = apply_event(copy.deepcopy(state), event, store.protocol)
        if preview["state"] == "FINALIZED_ABORTED_NO_METRICS":
            state = store.append(state, rows, event)
            break
        try:
            written_sha = _write_once(store.root / "results" / f"{member_id}.json", receipt)
        except RuntimeEvidenceError as exc:
            return _abort(store, state, rows, f"result_receipt_write_conflict:{member_id}:{exc}")
        if written_sha != receipt_sha:
            return _abort(store, state, rows, f"result_receipt_hash_mismatch:{member_id}")
        state = store.append(state, rows, event)
    return state


def _abort_if_result_present_before_fixed_n(
    store: CohortStore,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target = store.protocol["cohort"]["target_races"]
    if len(state["predictions"]) >= target:
        return state
    inbox = store.root / "result_inbox"
    if not inbox.is_dir():
        return state
    paths = sorted(inbox.glob("*.json"), key=lambda item: item.name)
    if not paths:
        return state
    path = paths[0]
    try:
        content_sha256 = sha256_file(path)
    except OSError as exc:
        return _abort(
            store,
            state,
            rows,
            f"result_present_before_fixed_n_membership_freeze:{path.name}:read_failed:{exc}",
        )
    return _abort(
        store,
        state,
        rows,
        f"result_present_before_fixed_n_membership_freeze:{path.name}:{content_sha256}",
    )


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
    scorer_started_now = False
    if state["state"] == "READY_TO_FINALIZE":
        if any(path.exists() for path in final_paths):
            return _abort(store, state, rows, "finalization_artifact_preexists_request")
        state = store.append(
            state,
            rows,
            {"event_id": "fixed-n-finalize-requested", "type": "FINALIZE_REQUESTED"},
        )
        scorer_started_now = True

    metrics_path = store.root / "METRICS.json"
    if metrics_path.is_file():
        try:
            report = _load_json(metrics_path)
        except RuntimeEvidenceError as exc:
            if state["state"] == "FINALIZATION_LOCKED":
                return _abort(store, state, rows, f"finalizer_evidence_failure:{exc}")
            raise
        try:
            _validate_scored_report(report)
            if (
                report.get("member_manifest_sha256")
                != state["finalization_member_manifest_sha256"]
            ):
                raise RuntimeEvidenceError("metrics_receipt_member_manifest_invalid")
        except RuntimeEvidenceError as exc:
            if state["state"] == "FINALIZATION_LOCKED":
                return _abort(store, state, rows, f"finalizer_evidence_failure:{exc}")
            raise RuntimeEvidenceError(f"metrics_receipt_invalid_after_score_commit:{exc}") from exc
        metrics_sha = sha256_file(metrics_path)
    elif state["state"] == "FINALIZATION_LOCKED" and not scorer_started_now:
        return _abort(
            store,
            state,
            rows,
            "scorer_start_precommit_without_durable_metrics",
        )
    elif state["state"] == "FINALIZATION_LOCKED":
        try:
            report = finalize(state, store.root, protocol_path, asset_dir)
        except FinalizationError as exc:
            return _abort(store, state, rows, f"finalizer_evidence_failure:{exc}")
        try:
            _validate_scored_report(report)
            if (
                report.get("member_manifest_sha256")
                != state["finalization_member_manifest_sha256"]
            ):
                raise RuntimeEvidenceError("metrics_receipt_member_manifest_invalid")
        except RuntimeEvidenceError as exc:
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
    _require_terminal_commit(store.root)
    return state


def run_once(
    cohort_root: Path,
    protocol_path: Path,
    asset_dir: Path,
    finalizer_path: Path,
    service_unit_path: Path,
    *,
    state_machine_path: Path = STATE_MACHINE_PATH,
    observation_clock: ObservationClock = _system_clock,
) -> dict[str, Any]:
    terminal_names = (
        "TERMINAL_SENTINEL.json",
        "CONSUMED.json",
        "FINAL_REPORT.json",
        "METRICS.json",
    )
    if cohort_root.is_dir() and any(
        (cohort_root / name).exists() for name in terminal_names
    ):
        with _exclusive_lock(cohort_root / "runtime" / "successor.lock"):
            terminal = _resume_or_recover_independent_terminal(cohort_root)
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
        terminal = _resume_or_recover_independent_terminal(cohort_root)
        if terminal is not None:
            return terminal
        observed_at = _observe_utc(observation_clock)
        try:
            state, rows = store.replay(verify_receipts=False)
        except (RuntimeEvidenceError, OSError, KeyError, TypeError, ValueError) as exc:
            return _independent_terminal(cohort_root, f"untrusted_journal_or_protocol:{exc}")
        try:
            committed = _validated_terminal_commit(cohort_root)
            if committed is not None:
                report = _load_json(cohort_root / "FINAL_REPORT.json")
                if state["state"] == "FINALIZED_SCORED":
                    if (
                        committed.get("metrics_receipt_sha256")
                        != state["metrics_receipt_sha256"]
                        or committed.get("member_manifest_sha256")
                        != state["finalization_member_manifest_sha256"]
                    ):
                        raise RuntimeEvidenceError("terminal_commit_state_binding_invalid")
                elif state["state"] == "FINALIZED_ABORTED_NO_METRICS":
                    if (
                        committed.get("metrics_receipt_sha256") is not None
                        or committed.get("terminal_sentinel_sha256") is not None
                        or report.get("blocking_reason") != state["fatal_reason"]
                        or report.get("sealed_prediction_races")
                        != len(state["predictions"])
                        or report.get("approved_result_races") != len(state["results"])
                        or report.get("score_invocation_count")
                        != state["score_invocation_count"]
                    ):
                        raise RuntimeEvidenceError(
                            "terminal_commit_state_binding_invalid"
                        )
                else:
                    raise RuntimeEvidenceError("terminal_commit_before_terminal_journal_state")
        except RuntimeEvidenceError as exc:
            _discard_terminal_publication(cohort_root)
            return _independent_terminal(
                cohort_root,
                f"terminal_commit_validation_failed:{exc}",
                score_invocation_count=state["score_invocation_count"],
            )
        try:
            store.verify_receipts(state)
        except RuntimeEvidenceError as exc:
            if state["state"] in TERMINAL_STATES:
                if state["state"] == "FINALIZED_SCORED" and not (
                    cohort_root / "CONSUMED.json"
                ).exists():
                    _discard_unconsumed_score_artifacts(cohort_root)
                    return _independent_terminal(
                        cohort_root,
                        f"sealed_receipt_validation_failed_after_score_commit:{exc}",
                        score_invocation_count=state["score_invocation_count"],
                    )
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
        identity = runtime_identity(finalizer_path, service_unit_path, state_machine_path)
        if not rows:
            state = _authorize(store, state, rows, identity, observed_at)
        if state["state"] not in TERMINAL_STATES:
            state = _check_runtime_admission(store, state, rows, identity, observed_at)
        if state["state"] not in TERMINAL_STATES:
            state = _abort_if_result_present_before_fixed_n(store, state, rows)
        if state["state"] == "COLLECTING":
            state = _seal_candidates(
                store,
                state,
                rows,
                model,
                preprocessing,
                observation_clock,
            )
        if state["state"] == "RESULT_CLOSURE":
            state = _seal_results(store, state, rows, observed_at)
        if state["state"] in {"READY_TO_FINALIZE", "FINALIZATION_LOCKED", "FINALIZED_SCORED"}:
            try:
                state = _finalize_if_ready(store, state, rows, protocol_path, asset_dir)
            except RuntimeEvidenceError as exc:
                if state["state"] == "FINALIZED_SCORED" and not (
                    cohort_root / "CONSUMED.json"
                ).exists():
                    _discard_unconsumed_score_artifacts(cohort_root)
                    return _independent_terminal(
                        cohort_root,
                        f"score_artifact_validation_failed_after_score_commit:{exc}",
                        score_invocation_count=state["score_invocation_count"],
                    )
                raise
        if state["state"] == "FINALIZED_ABORTED_NO_METRICS":
            state = _abort(
                store,
                state,
                rows,
                state.get("fatal_reason") or "terminal_state_without_fatal_reason",
            )
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
