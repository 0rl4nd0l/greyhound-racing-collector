import argparse
import json
import shutil
import sqlite3
import sys
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from accuracy_program.snapshots import assert_no_result_fields, persist_prediction_snapshot
from scripts import capture_prediction_snapshot as capture_module
from scripts.capture_prediction_snapshot import (
    _apply_target_metadata_to_snapshot,
    _persistence_skip_category,
    _should_write_snapshot,
)
from utils.csv_metadata import verify_canonical_sidecar_target_metadata
from utils.runner_completeness import analyze_csv_runner_completeness
from utils.race_lifecycle import JUMPED_PENDING_RESULTS, RaceLifecycle, UPCOMING_NOT_JUMPED


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


class _CompleteRunnerSet:
    def as_dict(self) -> dict:
        return {
            "status": "COMPLETE",
            "runner_count": 2,
            "participants": [
                {"box_number": 1, "dog_clean_name": "Alpha Runner"},
                {"box_number": 2, "dog_clean_name": "Bravo Runner"},
            ],
        }


def _install_lightweight_capture_stubs(
    monkeypatch,
    *,
    stale_races: set[int] | None = None,
    stale_on_recheck: set[int] | None = None,
    prediction_rows: list[dict] | None = None,
) -> dict[int, int]:
    stale_races = stale_races or set()
    stale_on_recheck = stale_on_recheck or set()
    calls: dict[int, int] = {}
    fixed_now = datetime(2026, 5, 29, 14, 0, tzinfo=ZoneInfo("Australia/Melbourne"))

    def race_number_from_path(path: Path) -> int:
        return 2 if "Race 2" in path.name else 1

    def fake_classify(path, **_kwargs):
        race_number = race_number_from_path(Path(path))
        calls[race_number] = calls.get(race_number, 0) + 1
        if race_number in stale_races or (
            race_number in stale_on_recheck and calls[race_number] > 1
        ):
            return RaceLifecycle(
                status=JUMPED_PENDING_RESULTS,
                status_reason="jump_time_passed_no_official_result",
                race_date="2026-05-29",
                venue="TEST",
                race_number=race_number,
                jump_time="13:00",
                jump_datetime="2026-05-29T13:00:00+10:00",
                source_path=str(path),
            )
        return RaceLifecycle(
            status=UPCOMING_NOT_JUMPED,
            status_reason="jump_time_after_now_no_result",
            race_date="2026-05-29",
            venue="TEST",
            race_number=race_number,
            jump_time="15:00",
            jump_datetime="2026-05-29T15:00:00+10:00",
            source_path=str(path),
        )

    default_prediction_rows = [
        {
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "win_prob_norm": 0.6,
            "predicted_rank": 1,
        },
        {
            "dog_clean_name": "Bravo Runner",
            "box_number": 2,
            "win_prob_norm": 0.4,
            "predicted_rank": 2,
        },
    ]

    fake_app = types.SimpleNamespace(
        run_prediction_for_race_file=lambda path: {
            "success": True,
            "race_id": Path(path).stem,
            "model_version": "test-model",
            "predictions": prediction_rows or default_prediction_rows,
        },
        enhance_prediction_with_csv_meta=lambda result, _path: result,
    )
    monkeypatch.setitem(sys.modules, "app", fake_app)
    monkeypatch.setattr(capture_module, "melbourne_now", lambda: fixed_now)
    monkeypatch.setattr(capture_module, "classify_race_file", fake_classify)
    monkeypatch.setattr(
        capture_module,
        "analyze_csv_runner_completeness",
        lambda _path: _CompleteRunnerSet(),
    )
    monkeypatch.setattr(
        capture_module,
        "canonical_race_url_from_sidecar",
        lambda _path: "https://www.thedogs.com.au/racing/test/2026-05-29/1",
    )
    monkeypatch.setattr(
        capture_module,
        "fetch_canonical_runner_set",
        lambda _url: {"status": "ok"},
    )
    monkeypatch.setattr(
        capture_module,
        "verify_final_runner_set",
        lambda _source, _canonical: {
            "final_runner_set_status": "verified",
            "final_runner_set_source": "test",
            "final_runner_set_source_url": "https://www.thedogs.com.au/racing/test",
            "source_active_boxes": [1, 2],
            "canonical_active_boxes": [1, 2],
        },
    )
    monkeypatch.setattr(
        capture_module,
        "verify_canonical_sidecar_target_metadata",
        lambda *_args, **_kwargs: {
            "target_metadata_status": "verified",
            "target_distance": "450m",
            "target_grade": "Grade 5",
            "target_distance_source": "canonical_pre_race_page",
            "target_grade_source": "canonical_pre_race_page",
            "metadata_source_detail": "test",
            "canonical_race_url": "https://www.thedogs.com.au/racing/test",
            "race_time_mapping_status": "exact_url_match",
            "race_time_source": "canonical_race_url",
        },
    )
    return calls


def _write_test_race(tmp_path: Path, race_number: int) -> Path:
    path = tmp_path / f"Race {race_number} - TEST - 2026-05-29.csv"
    path.write_text(
        "Dog Name|BOX\nAlpha Runner|1\nBravo Runner|2\n",
        encoding="utf-8",
    )
    return path


def _capture_args(tmp_path: Path, db_path: Path, snapshot_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        db=str(db_path),
        upcoming_dir=str(tmp_path),
        snapshot_dir=str(snapshot_dir),
        race_file=[],
        limit=0,
        persist=True,
        approve_live_persist=True,
        mechanics_on_stale=False,
        capture_live_odds=False,
        allow_unverified_runner_set=False,
    )


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


def test_partial_persist_skips_pre_capture_stale_candidate_and_writes_fresh_ready(
    tmp_path, monkeypatch
):
    _write_test_race(tmp_path, 1)
    _write_test_race(tmp_path, 2)
    db_path = tmp_path / "greyhound.sqlite"
    sqlite3.connect(db_path).close()
    snapshot_dir = tmp_path / "snapshots"
    _install_lightweight_capture_stubs(monkeypatch, stale_races={2})

    report = capture_module.capture_snapshots(
        _capture_args(tmp_path, db_path, snapshot_dir)
    )

    assert report["status"] == "SUCCESS"
    assert report["capture_count"] == 1
    assert report["skipped_lifecycle_candidate_count"] == 1
    assert report["skipped_lifecycle_candidates"][0]["race_number"] == 2
    assert report["skipped_lifecycle_candidates"][0]["persistence"]["status"] == (
        "skipped_non_live_lifecycle_pre_capture"
    )
    assert report["captures"][0]["persistence"]["status"] == "persisted"
    assert report["captures"][0]["pre_persist_freshness_check"]["still_pre_jump"] is True
    assert report["persisted_with_top_level_metadata_count"] == 1
    assert len(list(snapshot_dir.rglob("*.json"))) == 1
    assert len((snapshot_dir / "manifest.jsonl").read_text().splitlines()) == 1


def test_partial_persist_rechecks_immediately_and_skips_race_that_jumped(
    tmp_path, monkeypatch
):
    _write_test_race(tmp_path, 1)
    _write_test_race(tmp_path, 2)
    db_path = tmp_path / "greyhound.sqlite"
    sqlite3.connect(db_path).close()
    snapshot_dir = tmp_path / "snapshots"
    calls = _install_lightweight_capture_stubs(monkeypatch, stale_on_recheck={2})

    report = capture_module.capture_snapshots(
        _capture_args(tmp_path, db_path, snapshot_dir)
    )

    persisted = [
        capture
        for capture in report["captures"]
        if capture["persistence"]["status"] == "persisted"
    ]
    skipped = [
        capture
        for capture in report["captures"]
        if capture["persistence"]["status"]
        == "skipped_pre_persist_lifecycle_not_live"
    ]
    assert report["status"] == "SUCCESS"
    assert report["capture_count"] == 2
    assert calls[1] == 2
    assert calls[2] == 2
    assert len(persisted) == 1
    assert len(skipped) == 1
    assert persisted[0]["pre_persist_freshness_check"]["still_pre_jump"] is True
    assert skipped[0]["pre_persist_freshness_check"]["still_pre_jump"] is False
    assert skipped[0]["persistence_skip_category"] == "lifecycle"
    assert len(list(snapshot_dir.rglob("*.json"))) == 1
    assert len((snapshot_dir / "manifest.jsonl").read_text().splitlines()) == 1


def test_persist_recheck_skips_all_newly_stale_ready_races_without_manifest(
    tmp_path, monkeypatch
):
    _write_test_race(tmp_path, 1)
    _write_test_race(tmp_path, 2)
    db_path = tmp_path / "greyhound.sqlite"
    sqlite3.connect(db_path).close()
    snapshot_dir = tmp_path / "snapshots"
    calls = _install_lightweight_capture_stubs(monkeypatch, stale_on_recheck={1, 2})

    report = capture_module.capture_snapshots(
        _capture_args(tmp_path, db_path, snapshot_dir)
    )

    skipped = [
        capture
        for capture in report["captures"]
        if capture["persistence"]["status"]
        == "skipped_pre_persist_lifecycle_not_live"
    ]
    assert report["status"] == "SUCCESS"
    assert report["capture_count"] == 2
    assert calls[1] == 2
    assert calls[2] == 2
    assert len(skipped) == 2
    assert all(
        capture["pre_persist_freshness_check"]["still_pre_jump"] is False
        for capture in skipped
    )
    assert report["persisted_with_top_level_metadata_count"] == 0
    assert not list(snapshot_dir.rglob("*.json"))
    assert not (snapshot_dir / "manifest.jsonl").exists()


def test_capture_predicts_from_canonical_aligned_promoted_reserve_csv(
    tmp_path, monkeypatch
):
    race_file = tmp_path / "Race 4 - TEST - 2026-05-29.csv"
    race_file.write_text(
        "\n".join(
            [
                "Dog Name|Sex|PLC|BOX|DATE",
                "1. Alpha Runner|D|1|1|2026-05-01",
                "2. Bravo Runner|D|1|2|2026-05-01",
                "3. Charlie Runner|D|1|3|2026-05-01",
                "4. Scratched Runner|D|1|4|2026-05-01",
                "9. Reserve Runner|D|1|8|2026-05-01",
            ]
        ),
        encoding="utf-8",
    )
    canonical = {
        "schema_version": "canonical_pre_race_runner_set_v1",
        "canonical_runner_set_status": "available",
        "final_runner_source_url": "https://www.thedogs.com.au/racing/test/2026-05-29/4",
        "final_runner_boxes": [1, 2, 3, 4],
        "final_runner_names": [
            "Alpha Runner",
            "Bravo Runner",
            "Charlie Runner",
            "Reserve Runner",
        ],
        "final_runner_participants": [
            {"box_number": 1, "dog_name": "Alpha Runner"},
            {"box_number": 2, "dog_name": "Bravo Runner"},
            {"box_number": 3, "dog_name": "Charlie Runner"},
            {
                "box_number": 4,
                "dog_name": "Reserve Runner",
                "original_box_number": 9,
            },
        ],
        "scratched_boxes": [4],
        "scratched_participants": [
            {"box_number": 4, "dog_name": "Scratched Runner"}
        ],
        "reserve_boxes": [9],
        "vacant_boxes": [],
        "extraction_timestamp": "2026-05-29T05:00:00Z",
    }
    seen: dict[str, str] = {}

    def fake_run_prediction(path: str) -> dict:
        seen["path"] = path
        seen["text"] = Path(path).read_text(encoding="utf-8")
        runners = analyze_csv_runner_completeness(path).as_dict()["participants"]
        probability = round(1.0 / len(runners), 4)
        return {
            "success": True,
            "race_id": Path(path).stem,
            "model_version": "test-model",
            "predictions": [
                {
                    "dog_clean_name": row["dog_name"],
                    "box_number": row["box_number"],
                    "win_prob_norm": probability,
                    "predicted_rank": idx + 1,
                }
                for idx, row in enumerate(runners)
            ],
        }

    fake_app = types.SimpleNamespace(
        run_prediction_for_race_file=fake_run_prediction,
        enhance_prediction_with_csv_meta=lambda result, _path: result,
    )
    monkeypatch.setitem(sys.modules, "app", fake_app)
    monkeypatch.setattr(
        capture_module,
        "canonical_race_url_from_sidecar",
        lambda _path: canonical["final_runner_source_url"],
    )
    monkeypatch.setattr(
        capture_module,
        "fetch_canonical_runner_set",
        lambda _url: canonical,
    )
    monkeypatch.setattr(
        capture_module,
        "verify_canonical_sidecar_target_metadata",
        lambda *_args, **_kwargs: {
            "target_metadata_status": "verified",
            "target_distance": "450m",
            "target_grade": "Grade 5",
            "target_distance_source": "canonical_pre_race_page",
            "target_grade_source": "canonical_pre_race_page",
            "metadata_source_detail": "test",
            "canonical_race_url": canonical["final_runner_source_url"],
            "race_time_mapping_status": "exact_url_match",
            "race_time_source": "canonical_race_url",
        },
    )
    lifecycle = RaceLifecycle(
        status=UPCOMING_NOT_JUMPED,
        status_reason="jump_time_after_now_no_result",
        race_date="2026-05-29",
        venue="TEST",
        race_number=4,
        jump_time="15:00",
        jump_datetime="2026-05-29T15:00:00+10:00",
        source_path=str(race_file),
    )

    capture = capture_module._capture_one(
        race_file=race_file,
        lifecycle=lifecycle,
        db_path=tmp_path / "greyhound.sqlite",
        snapshot_dir=tmp_path / "snapshots",
        persist=False,
        mechanics_only=False,
        capture_live_odds_requested=False,
        capture_live_odds_approved=False,
        allow_unverified_runner_set=False,
    )

    assert seen["path"] != str(race_file)
    assert "4. Reserve Runner" in seen["text"]
    assert "4. Scratched Runner" not in seen["text"]
    assert "9. Reserve Runner" not in seen["text"]
    assert capture["prediction_input_mode"] == "canonical_aligned_temp_csv"
    assert capture["final_runner_set_status"] == "verified"
    assert capture["snapshot_readiness"]["status"] == "READY"
    assert capture["runner_count"] == 4
    assert capture["runner_set_alignment"]["dropped_participants"] == [
        {"box_number": 4, "dog_name": "Scratched Runner"}
    ]
    assert capture["runner_set_alignment"]["remapped_participants"] == [
        {
            "dog_name": "Reserve Runner",
            "source_box_number": 9,
            "final_box_number": 4,
            "original_box_number": 9,
        }
    ]


def test_live_odds_capture_requires_explicit_approval(tmp_path, monkeypatch):
    _write_test_race(tmp_path, 1)
    db_path = tmp_path / "greyhound.sqlite"
    sqlite3.connect(db_path).close()
    snapshot_dir = tmp_path / "snapshots"
    _install_lightweight_capture_stubs(monkeypatch)
    monkeypatch.delenv("APPROVE_LIVE_ODDS_CAPTURE", raising=False)
    monkeypatch.setattr(
        capture_module,
        "_capture_live_odds_for_lifecycle",
        lambda **_kwargs: pytest.fail("live odds capture must not run without approval"),
    )
    args = _capture_args(tmp_path, db_path, snapshot_dir)
    args.capture_live_odds = True

    report = capture_module.capture_snapshots(args)

    assert report["odds_capture_requested"] is True
    assert report["odds_capture_approved"] is False
    assert report["odds_capture_approval"]["status"] == "not_approved"
    assert report["ev_readiness_counts"] == {"EV_NOT_READY": 1}
    assert report["priced_runner_count"] == 0
    assert report["priced_ev_runner_count"] == 0
    assert report["ev_eligible_runner_count"] == 0
    assert report["odds_exclusion_counts"] == {"missing_live_odds": 2}
    assert report["captures"][0]["odds_capture_requested"] is True
    assert report["captures"][0]["odds_capture_approved"] is False
    assert report["captures"][0]["odds_capture"] == {
        "status": "APPROVAL_REQUIRED",
        "success": False,
        "reason": "live_odds_capture_not_approved",
        "required_approval": "APPROVE_LIVE_ODDS_CAPTURE or --approve-live-odds-capture",
        "append_only": True,
    }


def test_approved_live_odds_capture_still_reports_ev_not_ready_for_bad_provenance(
    tmp_path, monkeypatch
):
    race_file = _write_test_race(tmp_path, 1)
    db_path = tmp_path / "greyhound.sqlite"
    sqlite3.connect(db_path).close()
    snapshot_dir = tmp_path / "snapshots"

    _install_lightweight_capture_stubs(
        monkeypatch,
        prediction_rows=[
            {
                "dog_clean_name": "Alpha Runner",
                "box_number": 1,
                "win_prob_norm": 0.6,
                "predicted_rank": 1,
                "odds_win": 3.0,
                "odds_timestamp": "2026-05-29T13:55:00",
                "odds_source": "untrusted-book",
                "odds_source_url": "https://example.invalid/greyhound-racing/test",
                "odds_race_id": "Race 1 - TEST - 2026-05-29",
                "odds_dog_name": "Alpha Runner",
                "odds_box_number": 1,
                "odds_match_method": "race_id_box_name",
                "odds_match_confidence": 1.0,
            },
            {
                "dog_clean_name": "Bravo Runner",
                "box_number": 2,
                "win_prob_norm": 0.4,
                "predicted_rank": 2,
            },
        ],
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2026, 5, 29, 14, 0, tzinfo=tz)
            return value

    monkeypatch.setattr(capture_module, "datetime", FixedDateTime)
    monkeypatch.setattr(
        capture_module,
        "_capture_live_odds_for_lifecycle",
        lambda **_kwargs: {
            "status": "SUCCESS",
            "success": True,
            "append_only": True,
            "source": "sportsbet",
        },
    )
    lifecycle = RaceLifecycle(
        status=UPCOMING_NOT_JUMPED,
        status_reason="jump_time_after_now_no_result",
        race_date="2026-05-29",
        venue="TEST",
        race_number=1,
        jump_time="15:00",
        jump_datetime="2026-05-29T15:00:00+10:00",
        source_path=str(race_file),
    )

    capture = capture_module._capture_one(
        race_file=race_file,
        lifecycle=lifecycle,
        db_path=db_path,
        snapshot_dir=snapshot_dir,
        persist=False,
        mechanics_only=False,
        capture_live_odds_requested=True,
        capture_live_odds_approved=True,
        allow_unverified_runner_set=False,
    )

    assert capture["status"] == "SUCCESS"
    assert capture["odds_capture"]["status"] == "SUCCESS"
    assert capture["odds_capture_requested"] is True
    assert capture["odds_capture_approved"] is True
    assert capture["ev_readiness_status"] == "EV_NOT_READY"
    assert capture["priced_runner_count"] == 1
    assert capture["ev_eligible_runner_count"] == 0
    assert capture["priced_ev_runner_count"] == 0
    assert capture["odds_exclusion_counts"] == {
        "missing_live_odds": 1,
        "untrusted_source": 1,
    }
    assert capture["snapshot_readiness"]["ev_readiness"]["requirements"][
        "ev_null_for_unpriced_or_ineligible"
    ] is True


def test_prediction_timestamp_is_recorded_after_approved_live_odds_capture(
    tmp_path, monkeypatch
):
    race_file = _write_test_race(tmp_path, 1)
    db_path = tmp_path / "greyhound.sqlite"
    sqlite3.connect(db_path).close()
    snapshot_dir = tmp_path / "snapshots"
    prediction_rows = [
        {
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "win_prob_norm": 0.6,
            "predicted_rank": 1,
            "odds_win": 3.0,
            "odds_source": "sportsbet",
            "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/test/r1",
            "odds_race_id": "Race 1 - TEST - 2026-05-29",
            "odds_dog_name": "Alpha Runner",
            "odds_box_number": 1,
            "odds_match_method": "race_id_box_name",
            "odds_match_confidence": 1.0,
        },
        {
            "dog_clean_name": "Bravo Runner",
            "box_number": 2,
            "win_prob_norm": 0.4,
            "predicted_rank": 2,
            "odds_win": 2.5,
            "odds_source": "sportsbet",
            "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/test/r1",
            "odds_race_id": "Race 1 - TEST - 2026-05-29",
            "odds_dog_name": "Bravo Runner",
            "odds_box_number": 2,
            "odds_match_method": "race_id_box_name",
            "odds_match_confidence": 1.0,
        },
    ]
    _install_lightweight_capture_stubs(monkeypatch, prediction_rows=prediction_rows)

    class SequencedDateTime(datetime):
        values = [
            datetime(2026, 5, 29, 14, 0, 0),
            datetime(2026, 5, 29, 14, 0, 10),
        ]

        @classmethod
        def now(cls, tz=None):
            value = cls.values.pop(0)
            if tz is not None:
                value = value.replace(tzinfo=tz)
            return value

    monkeypatch.setattr(capture_module, "datetime", SequencedDateTime)

    def fake_capture_live_odds(**_kwargs):
        captured_at = capture_module.datetime.now().isoformat(timespec="seconds")
        for row in prediction_rows:
            row["odds_timestamp"] = captured_at
        return {
            "status": "SUCCESS",
            "success": True,
            "append_only": True,
            "source": "sportsbet",
        }

    monkeypatch.setattr(
        capture_module,
        "_capture_live_odds_for_lifecycle",
        fake_capture_live_odds,
    )
    lifecycle = RaceLifecycle(
        status=UPCOMING_NOT_JUMPED,
        status_reason="jump_time_after_now_no_result",
        race_date="2026-05-29",
        venue="TEST",
        race_number=1,
        jump_time="15:00",
        jump_datetime="2026-05-29T15:00:00+10:00",
        source_path=str(race_file),
    )

    capture = capture_module._capture_one(
        race_file=race_file,
        lifecycle=lifecycle,
        db_path=db_path,
        snapshot_dir=snapshot_dir,
        persist=False,
        mechanics_only=False,
        capture_live_odds_requested=True,
        capture_live_odds_approved=True,
        allow_unverified_runner_set=False,
    )

    assert capture["status"] == "SUCCESS"
    assert capture["prediction_timestamp"] == "2026-05-29T14:00:10"
    assert capture["ev_readiness_status"] == "EV_READY"
    assert capture["priced_runner_count"] == 2
    assert capture["ev_eligible_runner_count"] == 2
    assert capture["priced_ev_runner_count"] == 2
    assert capture["odds_exclusion_counts"] == {}
    assert "timestamp_after_prediction" not in capture["snapshot_readiness"][
        "ev_readiness"
    ]["odds_exclusion_counts"]


def test_persist_requires_explicit_approval(tmp_path, monkeypatch):
    _write_test_race(tmp_path, 1)
    db_path = tmp_path / "greyhound.sqlite"
    sqlite3.connect(db_path).close()
    snapshot_dir = tmp_path / "snapshots"
    _install_lightweight_capture_stubs(monkeypatch)
    monkeypatch.delenv("APPROVE_LIVE_PERSIST", raising=False)
    args = _capture_args(tmp_path, db_path, snapshot_dir)
    args.approve_live_persist = False

    report = capture_module.capture_snapshots(args)

    assert report["persist_requested"] is True
    assert report["persist_approved"] is False
    assert report["persist_approval"]["status"] == "not_approved"
    assert report["dry_run"] is True
    assert report["captures"][0]["persistence"]["status"] == "dry_run"
    assert not list(snapshot_dir.rglob("*.json"))
    assert not (snapshot_dir / "manifest.jsonl").exists()


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
