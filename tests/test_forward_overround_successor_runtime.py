from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import scripts.forward_overround_successor_runtime as successor_runtime
from scripts.finalize_forward_overround_successor import (
    ASSET_HASHES,
    EXPECTED_PROTOCOL_SHA256,
    runner_set_sha256,
)
from scripts.forward_overround_successor_runtime import (
    CohortStore,
    RuntimeEvidenceError,
    canonical_bytes,
    read_status,
    run_once,
    runtime_identity,
    sha256_bytes,
    sha256_file,
)

REPO_ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs/prediction/forward_overround_successor_v1_protocol.json"
RUNTIME_PATH = REPO_ROOT / "scripts/forward_overround_successor_runtime.py"
FINALIZER_PATH = REPO_ROOT / "scripts/finalize_forward_overround_successor.py"
SERVICE_PATH = REPO_ROOT / "ops/systemd/forward-overround-successor.service"
ASSET_DIR = Path(
    "/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-overround-structure-20260816/"
    "reports/agent_jobs/OVERROUND_STRUCTURE_TRANSFORM_V1_20260816"
)
MELBOURNE = timezone(timedelta(hours=10))
ACTIVATION = datetime(2026, 9, 1, 0, 0, tzinfo=MELBOURNE)
OBSERVED = datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))


def _protocol() -> dict:
    payload = json.loads(PROTOCOL_PATH.read_bytes())
    payload["_document_sha256"] = EXPECTED_PROTOCOL_SHA256
    return payload


def _activation_receipt(finalizer_path: Path, service_path: Path) -> dict:
    identity = runtime_identity(finalizer_path, service_path)
    return {
        "schema_version": "forward_overround_successor_activation_v1",
        "authority": "separate_owner_authorization",
        "collection_authorized": True,
        "scheduler_authorized": True,
        "activation_at": ACTIVATION.isoformat(),
        "admission_id": "initial-reviewed-admission",
        "admitted_at": ACTIVATION.isoformat(),
        **identity,
        "semantic_contract_sha256": ASSET_HASHES["scorer_contract.json"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "reviewed": True,
        "model_and_config_unchanged": True,
        "protocol_semantics_unchanged": True,
        "predecessor_admission_sha256": None,
        "asset_hashes": dict(ASSET_HASHES),
    }


def _prepare_root(root: Path, finalizer_path: Path = FINALIZER_PATH, service_path: Path = SERVICE_PATH) -> None:
    root.mkdir()
    _write_json(root / "ACTIVATION.json", _activation_receipt(finalizer_path, service_path))


def _candidate(index: int, *, jump_at: datetime | None = None) -> dict:
    jump = jump_at or datetime(2026, 9, 2, 0, 0, tzinfo=MELBOURNE) + timedelta(minutes=5 * index)
    race_id = f"synthetic-forward-race-{index:04d}"
    return {
        "schema_version": "forward_overround_successor_candidate_v1",
        "race_id": race_id,
        "captured_at": (jump - timedelta(minutes=20)).isoformat(),
        "jump_at": jump.isoformat(),
        "source": "sportsbet",
        "market_type": "win",
        "raw_paired_column_win_proof": True,
        "source_receipt_sha256": hashlib.sha256(f"odds-{index}".encode()).hexdigest(),
        "active_runner_count": 2,
        "runners": [
            {
                "box_number": 1,
                "dog_name": f"Synthetic Alpha {index:04d}",
                "decimal_win_odds": 2.0 + (index % 5) * 0.05,
                "source_row_sha256": hashlib.sha256(f"odds-{index}-1".encode()).hexdigest(),
            },
            {
                "box_number": 2,
                "dog_name": f"Synthetic Beta {index:04d}",
                "decimal_win_odds": 3.6 + (index % 7) * 0.05,
                "source_row_sha256": hashlib.sha256(f"odds-{index}-2".encode()).hexdigest(),
            },
        ],
    }


def _member_id(candidate: dict) -> str:
    runner_hash = runner_set_sha256(candidate["runners"])
    return sha256_bytes(
        canonical_bytes({"race_id": candidate["race_id"], "runner_set_sha256": runner_hash})
    )


def _result(index: int, candidate: dict) -> dict:
    winner_box = 1 if index % 2 == 0 else 2
    return {
        "schema_version": "forward_overround_successor_official_result_v1",
        "member_id": _member_id(candidate),
        "race_id": candidate["race_id"],
        "captured_at": (
            datetime.fromisoformat(candidate["jump_at"]) + timedelta(minutes=5)
        ).isoformat(),
        "source": "thedogs",
        "official": True,
        "source_receipt_sha256": hashlib.sha256(f"result-{index}".encode()).hexdigest(),
        "winner_box": winner_box,
        "runners": [
            {
                "box_number": runner["box_number"],
                "dog_name": runner["dog_name"],
                "finish_position": 1 if runner["box_number"] == winner_box else 2,
            }
            for runner in candidate["runners"]
        ],
    }


def _run(
    root: Path,
    *,
    finalizer_path: Path = FINALIZER_PATH,
    service_path: Path = SERVICE_PATH,
    observed: datetime = OBSERVED,
) -> dict:
    return run_once(
        root,
        PROTOCOL_PATH,
        ASSET_DIR,
        finalizer_path,
        service_path,
        observation_clock=lambda: observed,
    )


def test_frozen_inputs_are_exact_and_prepared_only() -> None:
    assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256
    protocol = json.loads(PROTOCOL_PATH.read_bytes())
    assert protocol["cohort"]["target_races"] == 1000
    assert protocol["authorization"] == {
        "activation_receipt_required": True,
        "collection_authorized": False,
        "scheduler_install_authorized": False,
        "status": "PREPARED_NOT_AUTHORIZED",
    }
    assert protocol["boundaries"]["interim_candidate_loss_peeking"] is False
    for name, expected in ASSET_HASHES.items():
        assert sha256_file(ASSET_DIR / name) == expected


def test_runtime_never_creates_missing_cohort_or_activation(tmp_path: Path) -> None:
    missing = tmp_path / "successor-cohort"
    with unittest.TestCase().assertRaisesRegex(RuntimeEvidenceError, "cohort_root_must_preexist"):
        _run(missing)
    assert not missing.exists()

    missing.mkdir()
    with unittest.TestCase().assertRaisesRegex(RuntimeEvidenceError, "activation_receipt_absent"):
        _run(missing)
    assert list(missing.iterdir()) == []


def test_future_activation_receipt_cannot_start_collection(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _prepare_root(root)
    before_activation = ACTIVATION.astimezone(timezone.utc) - timedelta(seconds=1)
    with unittest.TestCase().assertRaisesRegex(RuntimeEvidenceError, "activation_receipt_not_yet_effective"):
        _run(root, observed=before_activation)
    assert not (root / "EVENTS.jsonl").exists()
    assert not (root / "predictions").exists()


def test_out_of_window_candidate_is_nonmember_rejection(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _prepare_root(root)
    candidate = _candidate(0)
    candidate["captured_at"] = (
        datetime.fromisoformat(candidate["jump_at"]) - timedelta(minutes=9)
    ).isoformat()
    _write_json(root / "candidate_inbox" / "candidate.json", candidate)

    status = _run(root)

    assert status["state"] == "COLLECTING"
    assert status["sealed_prediction_races"] == 0
    assert status["excluded_candidate_events"] == 1
    assert status["exclusion_reason_counts"] == {"candidate_timing_outside_window": 1}
    assert not (root / "predictions").exists()


def test_unit_drift_pauses_and_reviewed_hash_chained_admission_resumes(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    unit_copy = tmp_path / "successor.service"
    shutil.copyfile(SERVICE_PATH, unit_copy)
    _prepare_root(root, service_path=unit_copy)
    assert _run(root, service_path=unit_copy)["state"] == "COLLECTING"

    unit_copy.write_bytes(unit_copy.read_bytes() + b"# reviewed transport-only change\n")
    paused_at = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc)
    paused = _run(root, service_path=unit_copy, observed=paused_at)
    assert paused["state"] == "ADMISSION_PAUSED"
    assert paused["sealed_prediction_races"] == 0
    assert paused["fatal_reason"] is None

    store = CohortStore(root, _protocol())
    state, _ = store.replay()
    admission = {
        "schema_version": "forward_overround_successor_admission_v1",
        "admission_id": "reviewed-unit-admission-2",
        "admitted_at": (paused_at + timedelta(minutes=1)).isoformat(),
        **runtime_identity(FINALIZER_PATH, unit_copy),
        "semantic_contract_sha256": ASSET_HASHES["scorer_contract.json"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "reviewed": True,
        "model_and_config_unchanged": True,
        "protocol_semantics_unchanged": True,
        "predecessor_admission_sha256": state["active_admission_sha256"],
        "asset_hashes": dict(ASSET_HASHES),
    }
    _write_json(root / "admission_inbox" / "reviewed.json", admission)

    resumed = _run(
        root,
        service_path=unit_copy,
        observed=paused_at + timedelta(minutes=2),
    )
    assert resumed["state"] == "COLLECTING"
    assert resumed["active_admission_id"] == "reviewed-unit-admission-2"
    assert resumed["fatal_reason"] is None


def test_finalizer_drift_is_terminal_no_metrics(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    finalizer_copy = tmp_path / "finalizer.py"
    shutil.copyfile(FINALIZER_PATH, finalizer_copy)
    _prepare_root(root, finalizer_path=finalizer_copy)
    assert _run(root, finalizer_path=finalizer_copy)["state"] == "COLLECTING"

    finalizer_copy.write_bytes(finalizer_copy.read_bytes() + b"# drift\n")
    status = _run(root, finalizer_path=finalizer_copy)
    report = json.loads((root / "FINAL_REPORT.json").read_bytes())

    assert status["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert status["score_invocation_count"] == 0
    assert status["metrics_receipt_sha256"] is None
    assert report["verdict"] == "BLOCKED_FORWARD_EVIDENCE"
    assert report["metrics"] is None
    assert "finalizer_code_hash_drift" in report["blocking_reason"]


def test_changed_packet_for_immutable_member_is_fatal(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _prepare_root(root)
    candidate = _candidate(0)
    packet_path = root / "candidate_inbox" / "candidate.json"
    _write_json(packet_path, candidate)
    first = _run(root)
    assert first["sealed_prediction_races"] == 1
    assert first["state"] == "COLLECTING"

    candidate["runners"][0]["decimal_win_odds"] = 2.75
    _write_json(packet_path, candidate)
    terminal = _run(root)

    assert terminal["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert terminal["sealed_prediction_races"] == 1
    assert terminal["score_invocation_count"] == 0
    assert terminal["metrics_receipt_sha256"] is None
    assert terminal["fatal_reason"].startswith("conflicting_prediction_receipt:")


def test_tampered_sealed_receipt_appends_terminal_no_metrics(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _prepare_root(root)
    candidate = _candidate(0)
    _write_json(root / "candidate_inbox" / "candidate.json", candidate)
    first = _run(root)
    assert first["sealed_prediction_races"] == 1

    receipt_path = root / "predictions" / f"{_member_id(candidate)}.json"
    receipt_path.chmod(0o644)
    tampered = json.loads(receipt_path.read_bytes())
    tampered["source_receipt_sha256"] = "f" * 64
    _write_json(receipt_path, tampered)
    receipt_path.chmod(0o444)

    terminal = _run(root)
    report = json.loads((root / "FINAL_REPORT.json").read_bytes())
    consumed = json.loads((root / "CONSUMED.json").read_bytes())
    events = [
        json.loads(line)["event"]
        for line in (root / "EVENTS.jsonl").read_text().splitlines()
    ]

    assert terminal["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert terminal["score_invocation_count"] == 0
    assert terminal["metrics_receipt_sha256"] is None
    assert terminal["fatal_reason"].startswith("sealed_receipt_validation_failed:")
    assert events[-1]["type"] == "SEALED_EVIDENCE_INVALID"
    assert report["metrics"] is None
    assert consumed["verdict"] == "BLOCKED_FORWARD_EVIDENCE"
    assert not (root / "METRICS.json").exists()


def test_activation_receipt_byte_drift_after_initialization_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _prepare_root(root)
    assert _run(root)["state"] == "COLLECTING"

    activation_path = root / "ACTIVATION.json"
    activation = json.loads(activation_path.read_bytes())
    activation["unexpected_byte_drift"] = True
    _write_json(activation_path, activation)

    terminal = _run(root)
    report = json.loads((root / "FINAL_REPORT.json").read_bytes())

    assert terminal["state"] == "FINALIZED_ABORTED_NO_METRICS"
    assert terminal["score_invocation_count"] == 0
    assert terminal["metrics_receipt_sha256"] is None
    assert "activation_receipt_hash_drift" in report["blocking_reason"]


def test_invalid_complete_finish_order_fails_closed(tmp_path: Path) -> None:
    invalid_orders = {
        "missing": [1, 2, None],
        "duplicate": [1, 2, 2],
        "non-integer": [1, 2, "3"],
        "boolean": [1, 2, False],
        "out-of-range": [1, 2, 0],
    }
    for case, finish_positions in invalid_orders.items():
        root = tmp_path / case
        _prepare_root(root)
        candidate = _candidate(0)
        candidate["active_runner_count"] = 3
        candidate["runners"].append(
            {
                "box_number": 3,
                "dog_name": "Synthetic Gamma 0000",
                "decimal_win_odds": 5.5,
                "source_row_sha256": hashlib.sha256(b"odds-0-3").hexdigest(),
            }
        )
        _write_json(root / "candidate_inbox" / "candidate.json", candidate)
        assert _run(root)["sealed_prediction_races"] == 1

        result = _result(0, candidate)
        for runner, finish_position in zip(result["runners"], finish_positions, strict=True):
            runner["finish_position"] = finish_position
        _write_json(root / "result_inbox" / "result.json", result)

        terminal = _run(root)
        report = json.loads((root / "FINAL_REPORT.json").read_bytes())

        assert terminal["state"] == "FINALIZED_ABORTED_NO_METRICS"
        assert terminal["approved_result_races"] == 0
        assert terminal["score_invocation_count"] == 0
        assert terminal["metrics_receipt_sha256"] is None
        assert "result_finish_positions_invalid" in report["blocking_reason"]


def test_restart_at_every_finalization_write_boundary_is_idempotent(tmp_path: Path) -> None:
    boundaries = [
        "finalize-requested",
        "metrics",
        "score-committed",
        "final-report",
        "consumed",
        "status",
    ]
    for boundary in boundaries:
        root = tmp_path / boundary
        _prepare_root(root)
        candidates = [_candidate(index) for index in range(1000)]
        for index, candidate in enumerate(candidates):
            _write_json(root / "candidate_inbox" / f"{index:04d}.json", candidate)
            _write_json(root / "result_inbox" / f"{index:04d}.json", _result(index, candidate))

        finalize_calls = 0
        actual_finalize = successor_runtime.finalize
        actual_append = CohortStore.append
        actual_write_once = successor_runtime._write_once
        actual_write_status = successor_runtime._write_status
        injected = False

        def counted_finalize(*args: object, **kwargs: object) -> dict:
            nonlocal finalize_calls
            finalize_calls += 1
            return actual_finalize(*args, **kwargs)

        if boundary in {"finalize-requested", "score-committed"}:
            target_event = {
                "finalize-requested": "FINALIZE_REQUESTED",
                "score-committed": "PAIRED_SCORE_COMMITTED",
            }[boundary]

            def faulting_append(
                self: CohortStore,
                state: dict,
                rows: list[dict],
                event: dict,
            ) -> dict:
                nonlocal injected
                updated = actual_append(self, state, rows, event)
                if event.get("type") == target_event and not injected:
                    injected = True
                    raise SystemExit(f"fault_after_{boundary}")
                return updated

            boundary_patch = patch.object(CohortStore, "append", faulting_append)
        elif boundary in {"metrics", "final-report", "consumed"}:
            target_name = {
                "metrics": "METRICS.json",
                "final-report": "FINAL_REPORT.json",
                "consumed": "CONSUMED.json",
            }[boundary]

            def faulting_write_once(path: Path, payload: dict) -> str:
                nonlocal injected
                digest = actual_write_once(path, payload)
                if path.name == target_name and not injected:
                    injected = True
                    raise SystemExit(f"fault_after_{boundary}")
                return digest

            boundary_patch = patch.object(successor_runtime, "_write_once", faulting_write_once)
        else:

            def faulting_write_status(path: Path, payload: dict) -> None:
                nonlocal injected
                actual_write_status(path, payload)
                if not injected:
                    injected = True
                    raise SystemExit("fault_after_status")

            boundary_patch = patch.object(successor_runtime, "_write_status", faulting_write_status)

        with patch.object(successor_runtime, "finalize", counted_finalize):
            with boundary_patch:
                with unittest.TestCase().assertRaisesRegex(
                    SystemExit, f"fault_after_{boundary}"
                ):
                    _run(root)
            assert injected is True

            restarted = _run(root)
            events = [
                json.loads(line)["event"]
                for line in (root / "EVENTS.jsonl").read_text().splitlines()
            ]
            report_sha = sha256_file(root / "FINAL_REPORT.json")
            consumed_sha = sha256_file(root / "CONSUMED.json")

            assert restarted["state"] == "FINALIZED_SCORED"
            assert restarted["score_invocation_count"] == 1
            assert finalize_calls == 1
            assert (root / "FINAL_REPORT.json").is_file()
            assert (root / "CONSUMED.json").is_file()
            assert sum(event["type"] == "FINALIZE_REQUESTED" for event in events) == 1
            assert sum(event["type"] == "PAIRED_SCORE_COMMITTED" for event in events) == 1

            replayed = _run(root)
            assert replayed == restarted
            assert finalize_calls == 1
            assert sha256_file(root / "FINAL_REPORT.json") == report_sha
            assert sha256_file(root / "CONSUMED.json") == consumed_sha


def test_synthetic_empty_to_exact_1000_one_shot_paired_finalization(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _prepare_root(root)
    candidates = [_candidate(index) for index in range(1000)]
    for index, candidate in enumerate(candidates):
        _write_json(root / "candidate_inbox" / f"{index:04d}.json", candidate)
        _write_json(root / "result_inbox" / f"{index:04d}.json", _result(index, candidate))

    status = _run(root)
    report_path = root / "FINAL_REPORT.json"
    consumed_path = root / "CONSUMED.json"
    report_sha = sha256_file(report_path)
    consumed_sha = sha256_file(consumed_path)
    report = json.loads(report_path.read_bytes())
    consumed = json.loads(consumed_path.read_bytes())

    assert status["state"] == "FINALIZED_SCORED"
    assert status["sealed_prediction_races"] == 1000
    assert status["approved_result_races"] == 1000
    assert status["score_invocation_count"] == 1
    assert status["metrics_receipt_sha256"] == sha256_file(root / "METRICS.json")
    assert status["finalization_member_manifest_sha256"] == report["member_manifest_sha256"]
    assert report["race_count"] == 1000
    assert report["identical_races_compared"] is True
    assert report["score_invocation_count"] == 1
    assert report["metrics"]["race_bootstrap_95pct"]["replicates"] == 20000
    assert report["metrics"]["race_bootstrap_95pct"]["seed"] == 20260817
    assert report["metrics"]["race_date_cluster_bootstrap_95pct"]["replicates"] == 20000
    assert report["metrics"]["race_date_cluster_bootstrap_95pct"]["seed"] == 20260818
    assert [row["race_count"] for row in report["metrics"]["chronological_blocks"]] == [200] * 5
    assert report["profitability"] == {
        "roi_computed": False,
        "betting_analysis_performed": False,
    }
    assert consumed["final_report_sha256"] == report_sha

    replayed = _run(root)
    assert replayed == status
    assert sha256_file(report_path) == report_sha
    assert sha256_file(consumed_path) == consumed_sha
    journal_events = [json.loads(line)["event"] for line in (root / "EVENTS.jsonl").read_text().splitlines()]
    assert sum(event["type"] == "FINALIZE_REQUESTED" for event in journal_events) == 1
    assert sum(event["type"] == "PAIRED_SCORE_COMMITTED" for event in journal_events) == 1
    assert len(list((root / "predictions").glob("*.json"))) == 1000
    assert len(list((root / "results").glob("*.json"))) == 1000


class ForwardOverroundSuccessorRuntimeTests(unittest.TestCase):
    def _run_with_temp(self, function: object) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            function(Path(temporary))  # type: ignore[operator]

    def test_frozen_inputs(self) -> None:
        test_frozen_inputs_are_exact_and_prepared_only()

    def test_missing_cohort_or_activation(self) -> None:
        self._run_with_temp(test_runtime_never_creates_missing_cohort_or_activation)

    def test_future_activation(self) -> None:
        self._run_with_temp(test_future_activation_receipt_cannot_start_collection)

    def test_out_of_window_rejection(self) -> None:
        self._run_with_temp(test_out_of_window_candidate_is_nonmember_rejection)

    def test_unit_drift_pause_resume(self) -> None:
        self._run_with_temp(test_unit_drift_pauses_and_reviewed_hash_chained_admission_resumes)

    def test_finalizer_drift(self) -> None:
        self._run_with_temp(test_finalizer_drift_is_terminal_no_metrics)

    def test_immutable_member_conflict(self) -> None:
        self._run_with_temp(test_changed_packet_for_immutable_member_is_fatal)

    def test_tampered_sealed_receipt(self) -> None:
        self._run_with_temp(test_tampered_sealed_receipt_appends_terminal_no_metrics)

    def test_activation_receipt_drift(self) -> None:
        self._run_with_temp(test_activation_receipt_byte_drift_after_initialization_fails_closed)

    def test_invalid_finish_orders(self) -> None:
        self._run_with_temp(test_invalid_complete_finish_order_fails_closed)

    def test_finalization_restart_boundaries(self) -> None:
        self._run_with_temp(test_restart_at_every_finalization_write_boundary_is_idempotent)

    def test_synthetic_exact_1000(self) -> None:
        self._run_with_temp(test_synthetic_empty_to_exact_1000_one_shot_paired_finalization)


if __name__ == "__main__":
    unittest.main()
