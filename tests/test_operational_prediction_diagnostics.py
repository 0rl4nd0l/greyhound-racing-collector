"""Keep the cause of a pre-claim rejection without exposing exception payloads."""
import pytest

from race_collection.operational_prediction import failure_reason
from src.operator_ui.prediction_worker import WorkerRejected
from src.predictor.on_demand import PredictionBlocked


@pytest.mark.parametrize("code", ["RECEIPT_INVALID", "CURRENT_INDEX_PROVENANCE_CHANGED",
                                  "CURRENT_INDEX_SOURCE_INVALID", "RUNNER_SET_CHANGED"])
def test_worker_rejection_retains_machine_code(code):
    assert failure_reason(WorkerRejected(code)) == code


@pytest.mark.parametrize("message", ["https://example.invalid/private?token=secret",
                                     "RECEIPT_INVALID\nprivate payload", "A" * 97, ""])
def test_worker_rejection_does_not_retain_unstructured_payload(message):
    assert failure_reason(WorkerRejected(message)) == "WorkerRejected"


def test_os_error_does_not_expose_private_path():
    assert failure_reason(OSError("/private/input/result.json")) == "OSError"


def test_prediction_blocker_retains_only_code():
    assert failure_reason(PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE",
        private_path="/private/result.json", token="secret")) == "EXACT_RACE_IDENTITY_UNAVAILABLE"


@pytest.mark.parametrize("code", ["private?token=secret", "BLOCKED\nprivate", "A" * 97, ""])
def test_prediction_blocker_rejects_unstructured_code(code):
    assert failure_reason(PredictionBlocked(code)) == "PredictionBlocked"
