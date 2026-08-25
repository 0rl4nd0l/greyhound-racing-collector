import hashlib
import json
from pathlib import Path

from utils.runner_completeness import extract_canonical_runner_set_from_html

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "thedogs_live_20260825"
_BODY_SHA256 = "8151dedd4cf52bfe409791fa3286f1ac5b6ba56b2f41237f57faa87e35f7adbf"
_RECEIPT_SHA256 = "a087abb1f72d735037222bc5ae8976c9d60ca7719d36d47c48a03e74e5500dc8"
_EXPECTED_ACTIVE_RUNNER_IDS = [
    "7583527",
    "7583528",
    "7583529",
    "7583530",
    "7583531",
    "7583532",
    "7583534",
]


def _captured_race_page():
    body_path = _FIXTURE_ROOT / f"{_BODY_SHA256}.race-page.html"
    receipt_path = (
        _FIXTURE_ROOT
        / "32e86006c6d64e9d873825f2dded0a55ab366adb67fe2c89ed19a8c4c185ed28.race-page.receipt.json"
    )
    body = body_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert hashlib.sha256(body).hexdigest() == _BODY_SHA256
    assert hashlib.sha256(receipt_bytes).hexdigest() == _RECEIPT_SHA256
    assert receipt["body_sha256"] == _BODY_SHA256
    assert receipt["content_length"] == len(body)
    return body.decode("utf-8"), receipt


def test_captured_live_page_preserves_native_runner_entry_ids():
    html, receipt = _captured_race_page()

    canonical = extract_canonical_runner_set_from_html(
        html,
        source_url=receipt["final_url"],
        extraction_timestamp=receipt["capture_timestamp"],
    )

    assert canonical["canonical_runner_set_status"] == "available"
    assert canonical["final_runner_boxes"] == [1, 2, 3, 4, 5, 6, 8]
    assert canonical["scratched_boxes"] == [7]
    assert [
        runner["source_native_runner_id"]
        for runner in canonical["final_runner_participants"]
    ] == _EXPECTED_ACTIVE_RUNNER_IDS
    assert canonical["native_identity_reasons"] == ["source_native_race_id_missing"]
