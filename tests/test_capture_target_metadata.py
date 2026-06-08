import json
import shutil
from pathlib import Path

import pytest

from accuracy_program.snapshots import assert_no_result_fields, persist_prediction_snapshot
from scripts.capture_prediction_snapshot import (
    _apply_target_metadata_to_snapshot,
    _persistence_skip_category,
    _should_write_snapshot,
)
from utils.csv_metadata import verify_canonical_sidecar_target_metadata


ROOT = Path(__file__).resolve().parents[1]
LIVE_BATCH_DIR = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525/final_runner_verified_live_batch/upcoming_races"
)
BAL_CSV = LIVE_BATCH_DIR / "Race 1 - BAL - 2026-05-27.csv"
BAL_SIDECAR = Path(f"{BAL_CSV}.metadata.json")
BAL_SNAPSHOT_DIR = ROOT / "artifacts/prediction_snapshots/2026-05-27/BAL"


def _require_live_batch_fixture() -> None:
    if not BAL_CSV.exists() or not BAL_SIDECAR.exists():
        pytest.skip("local verified live-batch BAL CSV sidecar fixture is not present")


def _load_real_bal_snapshot() -> dict:
    candidates = sorted(BAL_SNAPSHOT_DIR.glob("race-1_*.json"))
    if not candidates:
        pytest.skip("local BAL frozen snapshot fixture is not present")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _copy_bal_fixture(tmp_path: Path) -> Path:
    _require_live_batch_fixture()
    csv_path = tmp_path / BAL_CSV.name
    shutil.copy2(BAL_CSV, csv_path)
    shutil.copy2(BAL_SIDECAR, Path(f"{csv_path}.metadata.json"))
    return csv_path


def test_verified_canonical_sidecar_metadata_copies_to_snapshot_top_level():
    _require_live_batch_fixture()
    snapshot = _load_real_bal_snapshot()

    metadata = verify_canonical_sidecar_target_metadata(BAL_CSV, race_number=1)
    _apply_target_metadata_to_snapshot(snapshot, metadata)

    assert metadata["target_metadata_status"] == "verified"
    assert snapshot["target_distance"] == "450m"
    assert snapshot["target_grade"] == "Maiden"
    assert snapshot["target_distance_source"] == "canonical_pre_race_page"
    assert snapshot["target_grade_source"] == "canonical_pre_race_page"
    assert snapshot["metadata_is_leakage_safe"] is True
    assert snapshot["snapshot_readiness"]["requirements"]["target_metadata_verified"] is True
    assert _should_write_snapshot(
        persist=True,
        live_lifecycle=True,
        runner_set_complete=True,
        final_runner_verified=True,
        target_metadata_verified=True,
        allow_unverified_runner_set=False,
        mechanics_only=False,
    )
    assert_no_result_fields(snapshot)


def test_unsafe_sidecar_target_metadata_fails_closed_for_persist(tmp_path):
    csv_path = _copy_bal_fixture(tmp_path)
    sidecar_path = Path(f"{csv_path}.metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata_is_leakage_safe"] = False
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    metadata = verify_canonical_sidecar_target_metadata(csv_path, race_number=1)
    snapshot = _load_real_bal_snapshot()
    _apply_target_metadata_to_snapshot(snapshot, metadata)

    assert metadata["target_metadata_status"] == "unsafe"
    assert snapshot["target_distance"] is None
    assert snapshot["target_grade"] is None
    assert not _should_write_snapshot(
        persist=True,
        live_lifecycle=True,
        runner_set_complete=True,
        final_runner_verified=True,
        target_metadata_verified=False,
        allow_unverified_runner_set=False,
        mechanics_only=False,
    )
    assert (
        _persistence_skip_category(
            live_lifecycle=True,
            runner_set_complete=True,
            final_runner_verified=True,
            target_metadata_verified=False,
            allow_unverified_runner_set=False,
            mechanics_only=False,
        )
        == "metadata"
    )


@pytest.mark.parametrize("missing_key", ["target_distance", "target_grade"])
def test_missing_target_distance_or_grade_fails_closed_for_persist(tmp_path, missing_key):
    csv_path = _copy_bal_fixture(tmp_path)
    sidecar_path = Path(f"{csv_path}.metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[missing_key] = None
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    metadata = verify_canonical_sidecar_target_metadata(csv_path, race_number=1)

    assert metadata["target_metadata_status"] == "missing"
    assert missing_key in metadata["target_metadata_failure_reason"]
    assert not _should_write_snapshot(
        persist=True,
        live_lifecycle=True,
        runner_set_complete=True,
        final_runner_verified=True,
        target_metadata_verified=False,
        allow_unverified_runner_set=False,
        mechanics_only=False,
    )


def test_canonical_url_race_number_mismatch_fails_closed(tmp_path):
    csv_path = _copy_bal_fixture(tmp_path)

    metadata = verify_canonical_sidecar_target_metadata(csv_path, race_number=2)

    assert metadata["target_metadata_status"] == "mismatch"
    assert "canonical_url_race_number_mismatch:1!=2" in metadata[
        "target_metadata_failure_reason"
    ]
    assert not _should_write_snapshot(
        persist=True,
        live_lifecycle=True,
        runner_set_complete=True,
        final_runner_verified=True,
        target_metadata_verified=False,
        allow_unverified_runner_set=False,
        mechanics_only=False,
    )


def test_dry_run_snapshot_persistence_never_writes_snapshot_files(tmp_path):
    _require_live_batch_fixture()
    snapshot = _load_real_bal_snapshot()
    metadata = verify_canonical_sidecar_target_metadata(BAL_CSV, race_number=1)
    _apply_target_metadata_to_snapshot(snapshot, metadata)

    report = persist_prediction_snapshot(snapshot, tmp_path, dry_run=True)

    assert report["status"] == "dry_run"
    assert not list(tmp_path.rglob("*.json"))
    assert not (tmp_path / "manifest.jsonl").exists()


def test_final_runner_set_verified_remains_mandatory_for_writes():
    assert not _should_write_snapshot(
        persist=True,
        live_lifecycle=True,
        runner_set_complete=True,
        final_runner_verified=False,
        target_metadata_verified=True,
        allow_unverified_runner_set=False,
        mechanics_only=False,
    )
    assert not _should_write_snapshot(
        persist=True,
        live_lifecycle=True,
        runner_set_complete=True,
        final_runner_verified=False,
        target_metadata_verified=True,
        allow_unverified_runner_set=True,
        mechanics_only=False,
    )
    assert (
        _persistence_skip_category(
            live_lifecycle=True,
            runner_set_complete=True,
            final_runner_verified=False,
            target_metadata_verified=True,
            allow_unverified_runner_set=False,
            mechanics_only=False,
        )
        == "runner_set"
    )
