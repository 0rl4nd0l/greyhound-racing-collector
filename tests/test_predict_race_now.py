from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest

import scripts.predict_race_now as predict_now
import src.predictor.on_demand as on_demand
from scripts.predict_market_form_residual import score_from_artifacts
from scripts.predict_race_now import (
    CollectorLockBusy,
    _acquire_collector_lock_no_steal,
    _release_owned_collector_lock,
    main,
    replay_bundle,
    run_prediction,
)
from scripts.run_priority_race_prediction import CaptureHandoffError
from src.predictor.on_demand import (
    Dependencies,
    PredictionBlocked,
    canonical_bytes,
    resolve_model,
    seal_history_database,
    sha256_bytes,
    sha256_file,
    write_exact_bytes,
)


NOW = datetime.fromisoformat("2026-07-19T12:00:00+10:00")
RACE_ID = "Race 5 - GUNN - 2026-07-19"


def race(race_time: str = "13:00") -> dict[str, Any]:
    return {
        "venue": "GUNN",
        "race_number": 5,
        "date": "2026-07-19",
        "race_time": race_time,
        "url": "https://thedogs.com.au/racing/gunnedah/2026-07-19/5",
    }


def create_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            race_date TEXT,
            venue TEXT,
            race_number INTEGER,
            grade TEXT,
            distance REAL,
            race_time TEXT,
            start_datetime TEXT,
            track_condition TEXT,
            weather TEXT,
            data_source TEXT,
            url TEXT
        );
        CREATE TABLE dog_race_data (
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            placing INTEGER,
            individual_time REAL,
            sectional_1st REAL,
            weight REAL,
            beaten_margin REAL,
            margin REAL,
            data_source TEXT
        );
        """
    )
    rows = [
        ("past", "2026-07-18", "GUNN", 1),
        (RACE_ID, "2026-07-19", "GUNN", 5),
        ("future", "2026-07-20", "GUNN", 2),
    ]
    connection.executemany(
        "INSERT INTO race_metadata (race_id,race_date,venue,race_number,grade,distance) VALUES (?,?,?,?,?,?)",
        [(rid, day, venue, number, "5", 400) for rid, day, venue, number in rows],
    )
    for rid, _, _, _ in rows:
        connection.executemany(
            "INSERT INTO dog_race_data (race_id,dog_name,dog_clean_name,box_number,finish_position,individual_time) VALUES (?,?,?,?,?,?)",
            [(rid, "Alpha", "ALPHA", 1, 1, 22.9), (rid, "Beta", "BETA", 2, 2, 23.1)],
        )
    connection.commit()
    connection.close()


def validation(status: str = "PASS") -> dict[str, Any]:
    rows = [
        {
            "dog_name": "Alpha",
            "dog_clean_name": "Alpha",
            "box_number": 1,
            "identity": "ALPHA",
            "odds_decimal": 2.5,
            "sportsbet_box_source": "explicit_dom",
        },
        {
            "dog_name": "Beta",
            "dog_clean_name": "Beta",
            "box_number": 2,
            "identity": "BETA",
            "odds_decimal": 4.0,
            "sportsbet_box_source": "explicit_dom",
        },
    ]
    reasons = [] if status == "PASS" else ["sportsbet_place_accepted_runner_rows_zero"]
    return {
        "schema_version": "autonomous_live_odds_capture_validation_v1",
        "status": status,
        "source_url": "https://www.sportsbet.com.au/betting/greyhound-racing/gunnedah/race-5-999",
        "accepted_rows": rows,
        "accepted_row_count": len(rows),
        "rejected_rows": [],
        "accepted_place_rows": rows,
        "accepted_place_row_count": len(rows),
        "rejected_place_rows": [],
        "expected_runner_count": len(rows),
        "active_expected_runner_count": len(rows),
        "scratched_expected_runner_count": 0,
        "scratched_expected_runners": [],
        "scratched_expected_runners_with_odds": [],
        "missing_expected_runners": [],
        "extra_unexpected_runners": [],
        "failure_root_cause": None,
        "reasons": reasons,
    }


def handoff(captured_at: datetime = NOW) -> dict[str, Any]:
    report = {
        "schema_version": "autonomous_live_odds_capture_report_v1",
        "attempts": [
            {
                "schema_version": "autonomous_live_odds_capture_attempt_v1",
                "race_id": RACE_ID,
                "status": "APPENDED",
                "reasons": [],
                "fetch_time": captured_at.isoformat(),
                "append_time": captured_at.isoformat(),
                "validation": validation(),
            }
        ],
    }
    report_raw = canonical_bytes(report)
    form_raw = b"dog_name,box_number\nAlpha,1\nBeta,2\n"
    sidecar_raw = canonical_bytes(
        {"participants": [{"dog_name": "Alpha"}, {"dog_name": "Beta"}]}
    )
    return {
        "schema_version": "manual_priority_capture_handoff_v1",
        "race_id": RACE_ID,
        "append_timestamp": captured_at.isoformat(),
        "source_report_sha256": sha256_bytes(report_raw),
        "source_form_sha256": sha256_bytes(form_raw),
        "source_sidecar_sha256": sha256_bytes(sidecar_raw),
        "_report_bytes": report_raw,
        "_form_bytes": form_raw,
        "_sidecar_bytes": sidecar_raw,
        "_form_name": "gunnedah-r5.csv",
    }


class Busy(RuntimeError):
    def __init__(self) -> None:
        self.payload = {"owner": "collector"}


def fake_seal_features(**kwargs: Any) -> Mapping[str, Path]:
    db = sqlite3.connect(kwargs["db_path"])
    race_ids = [
        row[0]
        for row in db.execute("SELECT race_id FROM race_metadata ORDER BY race_id")
    ]
    db.close()
    assert race_ids == ["past"]
    output = Path(kwargs["output_dir"])
    output.mkdir(parents=True)
    rows = output / "shadow_feature_rows.json"
    manifest = output / "shadow_manifest.json"
    implementation = output / "implementation_file_manifest.json"
    rows.write_bytes(
        canonical_bytes(
            [
                {
                    "same_distance_same_grade_target_race_rows_used": 0,
                    "same_distance_same_grade_post_outcome_rows_used": 0,
                }
            ]
        )
    )
    manifest.write_bytes(canonical_bytes({"safe": True}))
    implementation.write_bytes(canonical_bytes({"safe": True}))
    return {
        "feature_rows": rows,
        "feature_manifest": manifest,
        "implementation_manifest": implementation,
    }


def fake_score_residual(**kwargs: Any) -> Mapping[str, Any]:
    return {
        "model_sha256": sha256_file(Path(kwargs["model_path"])),
        "manifest_sha256": sha256_file(Path(kwargs["manifest_path"])),
        "predictions": [
            {
                "box_number": 1,
                "dog_name": "Alpha",
                "full_probability": 0.65,
                "half_probability": 0.58,
                "market_probability": 0.6,
                "win_odds": 2.5,
            },
            {
                "box_number": 2,
                "dog_name": "Beta",
                "full_probability": 0.35,
                "half_probability": 0.42,
                "market_probability": 0.4,
                "win_odds": 4.0,
            },
        ],
    }


def refresh(
    target: Mapping[str, Any], bundle: Path, now: datetime, days: int
) -> tuple[Path, Path]:
    del target, now, days
    form = bundle / "source" / "fixture.csv"
    form.parent.mkdir(parents=True, exist_ok=True)
    form.write_text("dog_name,box_number\nAlpha,1\nBeta,2\n", encoding="utf-8")
    sidecar = form.with_name(form.name + ".metadata.json")
    sidecar.write_bytes(
        canonical_bytes({"participants": [{"dog_name": "Alpha"}, {"dog_name": "Beta"}]})
    )
    return form, sidecar


def dependencies(
    *,
    discover=lambda **kwargs: handoff(),
    fetch=lambda context, db, timeout: {
        "captured_at": NOW.isoformat(),
        "validation": validation(),
        "plan_item": {"race_id": RACE_ID},
    },
    acquire=lambda: "lock",
    release=lambda handle: None,
    now=lambda: NOW,
    monotonic=lambda: 0.0,
    sleep=lambda seconds: None,
) -> Dependencies:
    return Dependencies(
        schedule=lambda days: [race()],
        refresh=refresh,
        discover_receipt=discover,
        fetch_odds=fetch,
        acquire_lock=acquire,
        release_lock=release,
        lock_busy_type=Busy,
        seal_features=fake_seal_features,
        score_residual=fake_score_residual,
        now=now,
        monotonic=monotonic,
        sleep=sleep,
    )


def args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    db = tmp_path / "source.db"
    if not db.exists():
        create_db(db)
    values = {
        "race": "gunnedah r5",
        "model": "latest-research",
        "config": Path("configs/prediction/manual-default.json"),
        "odds_source": "auto",
        "db": db,
        "output_root": tmp_path / "bundles",
        "days_ahead": 1,
        "current_time": NOW.isoformat(),
        "fetch_timeout_seconds": 1.0,
        "capture_evidence_root": [tmp_path / "evidence"],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fixture_e2e_reuses_receipt_seals_features_selects_model_and_bundles(
    tmp_path: Path,
):
    result = run_prediction(args(tmp_path), dependencies())

    assert result["status"] == "PREDICTION_READY"
    assert result["odds_source"] == "verified_autonomous_receipt"
    assert result["model"]["resolved"] == "market_form_residual_v1"
    assert result["model"]["alias_resolved"] is True
    assert result["prediction"]["variant"] == "full_strength"
    assert result["history_seal"]["safe_race_count"] == 1
    assert result["history_seal"]["excluded_target_metadata_rows"] == 1
    assert result["history_seal"]["excluded_at_or_after_cutoff_metadata_rows"] == 1
    bundle = Path(result["bundle"])
    assert (bundle / "bundle_manifest.json").is_file()
    assert json.loads((bundle / "result.json").read_bytes()) == result


def test_operator_cli_emits_one_canonical_fixture_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    source_db = tmp_path / "source.db"
    create_db(source_db)
    monkeypatch.setattr(
        predict_now, "default_dependencies", lambda parsed: dependencies()
    )

    exit_code = main(
        [
            "--race",
            "gunnedah r5",
            "--model",
            "latest-research",
            "--config",
            "configs/prediction/manual-default.json",
            "--odds-source",
            "auto",
            "--db",
            str(source_db),
            "--output-root",
            str(tmp_path / "bundles"),
            "--current-time",
            NOW.isoformat(),
        ]
    )
    stdout = capsys.readouterr().out.encode()
    result = json.loads(stdout)
    assert exit_code == 0
    assert stdout == canonical_bytes(result)
    assert result["status"] == "PREDICTION_READY"
    assert result["research_only"] is True
    assert result["production_persisted"] is False


def test_unexpected_dependency_error_still_prints_one_canonical_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    deps = dependencies()
    deps.schedule = lambda days: (_ for _ in ()).throw(OSError("schedule unavailable"))
    monkeypatch.setattr(predict_now, "default_dependencies", lambda parsed: deps)

    exit_code = main(
        [
            "--race",
            "gunnedah r5",
            "--output-root",
            str(tmp_path / "bundles"),
        ]
    )
    stdout = capsys.readouterr().out.encode()
    result = json.loads(stdout)
    assert exit_code == 2
    assert stdout == canonical_bytes(result)
    assert result["status"] == "PREDICTION_INTERNAL_ERROR"
    assert result["blockers"] == [
        {"code": "PREDICTION_INTERNAL_ERROR", "error": "OSError"}
    ]


def test_operator_dependency_surface_excludes_shadow_writer_and_timer_control():
    operator_source = Path("scripts/predict_race_now.py").read_text(encoding="utf-8")
    safety_source = Path("src/predictor/on_demand.py").read_text(encoding="utf-8")
    scorer_source = inspect.getsource(score_from_artifacts)

    for source in (operator_source, safety_source, scorer_source):
        assert "append_shadow_record" not in source
        assert "shadow_output_path" not in source
        assert "systemctl" not in source
        assert "subprocess" not in source


def test_immediate_capture_is_isolated_and_releases_lock(tmp_path: Path):
    released: list[str] = []
    source_hash = sha256_file(args(tmp_path).db)
    result = run_prediction(
        args(
            tmp_path,
            model="market-only",
            config=Path("configs/prediction/market-only.json"),
        ),
        dependencies(discover=lambda **kwargs: None, release=released.append),
    )

    assert result["odds_source"] == "isolated_immediate_capture"
    assert result["prediction"]["variant"] == "market_only_implied"
    assert released == ["lock"]
    assert sha256_file(args(tmp_path).db) == source_hash
    assert result["production_persisted"] is False


def test_busy_collector_may_complete_receipt_during_bounded_wait(tmp_path: Path):
    clock = {"value": 0.0}
    calls = {"discover": 0}

    def discover(**kwargs: Any) -> Mapping[str, Any] | None:
        calls["discover"] += 1
        return handoff() if calls["discover"] >= 3 else None

    def sleep(seconds: float) -> None:
        clock["value"] += seconds

    result = run_prediction(
        args(tmp_path),
        dependencies(
            discover=discover,
            acquire=lambda: (_ for _ in ()).throw(Busy()),
            monotonic=lambda: clock["value"],
            sleep=sleep,
        ),
    )
    assert result["odds_source"] == "verified_autonomous_receipt"
    assert calls["discover"] == 3


def test_busy_without_receipt_returns_smallest_blocker(tmp_path: Path):
    clock = {"value": 0.0}

    def sleep(seconds: float) -> None:
        clock["value"] += seconds

    with pytest.raises(PredictionBlocked, match="BUSY") as captured:
        run_prediction(
            args(tmp_path),
            dependencies(
                discover=lambda **kwargs: None,
                acquire=lambda: (_ for _ in ()).throw(Busy()),
                monotonic=lambda: clock["value"],
                sleep=sleep,
            ),
        )
    assert captured.value.code == "BUSY"


def test_default_lock_path_never_reclaims_existing_unreadable_lock(tmp_path: Path):
    lock_path = tmp_path / "collector.lock"
    original = b"not-json\n"
    lock_path.write_bytes(original)

    with pytest.raises(CollectorLockBusy) as captured:
        _acquire_collector_lock_no_steal(
            lock_path, run_id="on_demand_test", output_dir=tmp_path / "bundle"
        )
    assert captured.value.payload["reason"] == "existing_lock_present_no_steal"
    assert lock_path.read_bytes() == original


def test_default_lock_release_requires_exact_owner_and_inode(tmp_path: Path):
    lock_path = tmp_path / "collector.lock"
    lock = _acquire_collector_lock_no_steal(
        lock_path, run_id="on_demand_test", output_dir=tmp_path / "bundle"
    )
    lock_path.write_bytes(b"not-json\n")

    with pytest.raises(PredictionBlocked) as captured:
        _release_owned_collector_lock(lock)
    assert captured.value.code == "LOCK_RELEASE_FAILED"
    assert lock_path.read_bytes() == b"not-json\n"


def test_default_lock_acquire_closes_and_removes_own_inode_when_fstat_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    lock_path = tmp_path / "collector.lock"

    def fail_fstat(descriptor: int) -> Any:
        del descriptor
        raise OSError("injected fstat failure")

    monkeypatch.setattr(predict_now.os, "fstat", fail_fstat)
    with pytest.raises(PredictionBlocked) as captured:
        _acquire_collector_lock_no_steal(
            lock_path, run_id="on_demand_test", output_dir=tmp_path / "bundle"
        )
    assert captured.value.code == "LOCK_ACQUIRE_FAILED"
    assert captured.value.details["reason"] == "descriptor_stat_failed"
    assert not lock_path.exists()


@pytest.mark.parametrize(
    ("discover", "code"),
    [
        (lambda **kwargs: handoff(NOW - timedelta(hours=1)), "RECEIPT_STALE"),
        (
            lambda **kwargs: (_ for _ in ()).throw(
                CaptureHandoffError("accepted_capture_attempt_ambiguous")
            ),
            "RECEIPT_AMBIGUOUS",
        ),
    ],
)
def test_stale_and_ambiguous_receipts_fail_closed(
    tmp_path: Path, discover: Any, code: str
):
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), dependencies(discover=discover))
    assert captured.value.code == code


def test_unavailable_market_fails_without_source_write(tmp_path: Path):
    command_args = args(tmp_path, odds_source="capture")
    before = sha256_file(command_args.db)
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(
            command_args,
            dependencies(
                discover=lambda **kwargs: None,
                fetch=lambda context, db, timeout: {
                    "captured_at": NOW.isoformat(),
                    "validation": validation("FAIL"),
                },
            ),
        )
    assert captured.value.code == "MARKET_UNAVAILABLE"
    assert sha256_file(command_args.db) == before
    bundle = Path(captured.value.details["bundle"])
    blocked_result = json.loads((bundle / "result.json").read_bytes())
    manifest = json.loads((bundle / "bundle_manifest.json").read_bytes())
    assert blocked_result["status"] == "MARKET_UNAVAILABLE"
    assert blocked_result["research_only"] is True
    assert blocked_result["blockers"] == [
        {"code": "MARKET_UNAVAILABLE", "reasons": validation("FAIL")["reasons"]}
    ]
    assert "result.json" in manifest["files"]


def test_market_fetch_exception_becomes_canonical_blocker_and_releases_lock(
    tmp_path: Path,
):
    released: list[str] = []

    def fail_fetch(
        context: Mapping[str, Any], db: Path, timeout: float
    ) -> Mapping[str, Any]:
        del context, db, timeout
        raise RuntimeError("injected market failure")

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(
            args(tmp_path, odds_source="capture"),
            dependencies(
                discover=lambda **kwargs: None,
                fetch=fail_fetch,
                release=released.append,
            ),
        )
    assert captured.value.code == "MARKET_UNAVAILABLE"
    assert captured.value.details["error"] == "RuntimeError"
    assert released == ["lock"]
    result = json.loads(
        Path(captured.value.details["bundle"], "result.json").read_bytes()
    )
    assert result["blockers"] == [
        {"code": "MARKET_UNAVAILABLE", "error": "RuntimeError"}
    ]


@pytest.mark.parametrize(
    ("boundary", "code"),
    [("features", "FEATURE_SEAL_FAILED"), ("scorer", "RESIDUAL_SCORER_FAILED")],
)
def test_dependency_failures_become_canonical_persisted_blockers(
    tmp_path: Path, boundary: str, code: str
):
    deps = dependencies()

    def fail(**kwargs: Any) -> Mapping[str, Any]:
        del kwargs
        raise RuntimeError("injected dependency failure")

    if boundary == "features":
        deps.seal_features = fail
    else:
        deps.score_residual = fail
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == code
    result = json.loads(
        Path(captured.value.details["bundle"], "result.json").read_bytes()
    )
    assert result["status"] == code
    assert result["blockers"] == [{"code": code, "error": "RuntimeError"}]


def test_post_jump_blocks_before_bundle_or_lock(tmp_path: Path):
    deps = dependencies()
    deps.schedule = lambda days: [race("11:59")]
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == "POST_JUMP"
    assert not (tmp_path / "bundles").exists()


def test_odds_captured_at_or_after_jump_fails_closed(tmp_path: Path):
    deps = dependencies(discover=lambda **kwargs: None)

    def late_fetch(
        context: Mapping[str, Any], db: Path, timeout: float
    ) -> Mapping[str, Any]:
        del context, db, timeout
        return {
            "captured_at": (NOW + timedelta(hours=1)).isoformat(),
            "validation": validation(),
        }

    deps.fetch_odds = late_fetch
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path, odds_source="capture"), deps)
    assert captured.value.code == "POST_JUMP"
    assert Path(captured.value.details["bundle"], "result.json").is_file()


def test_history_seal_excludes_target_and_all_same_or_future_dates(tmp_path: Path):
    source = tmp_path / "source.db"
    create_db(source)
    target = tmp_path / "sealed.db"
    audit = seal_history_database(
        source=source,
        target=target,
        target_race_id=RACE_ID,
        cutoff=NOW + timedelta(hours=1),
        runner_names=["Alpha", "Beta"],
    )
    connection = sqlite3.connect(target)
    ids = [row[0] for row in connection.execute("SELECT race_id FROM race_metadata")]
    dog_ids = {
        row[0] for row in connection.execute("SELECT race_id FROM dog_race_data")
    }
    connection.close()
    assert ids == ["past"]
    assert dog_ids == {"past"}
    assert audit["target_rows_materialized"] == 0
    assert audit["at_or_after_cutoff_rows_materialized"] == 0


def test_history_seal_rejects_malformed_date_for_target_runner(tmp_path: Path):
    source = tmp_path / "source.db"
    create_db(source)
    connection = sqlite3.connect(source)
    connection.execute(
        "INSERT INTO race_metadata (race_id, race_date, venue, race_number) VALUES (?, ?, ?, ?)",
        ("ambiguous", "not-a-date", "GUNN", 3),
    )
    connection.execute(
        "INSERT INTO dog_race_data (race_id, dog_name, dog_clean_name, box_number, finish_position, individual_time) VALUES (?, ?, ?, ?, ?, ?)",
        ("ambiguous", "Alpha", "ALPHA", 1, 1, 30.0),
    )
    connection.commit()
    connection.close()

    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=tmp_path / "sealed.db",
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )
    assert captured.value.code == "HISTORY_CUTOFF_AMBIGUOUS"
    assert captured.value.details["race_ids"] == ["ambiguous"]


def test_model_config_mismatch_and_alias_resolution_fail_or_resolve_exactly(
    tmp_path: Path,
):
    assert resolve_model("latest-research").resolved == "market_form_residual_v1"
    assert resolve_model("market-only").resolved == "market_only_v1"
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(
            args(tmp_path, config=Path("configs/prediction/market-only.json")),
            dependencies(),
        )
    assert captured.value.code == "MODEL_CONFIG_MISMATCH"


def test_bundle_replay_is_deterministic_and_tampering_fails(tmp_path: Path):
    result = run_prediction(
        args(
            tmp_path,
            model="market-only",
            config=Path("configs/prediction/market-only.json"),
        ),
        dependencies(),
    )
    bundle = Path(result["bundle"])
    assert replay_bundle(bundle) == result
    receipt_path = bundle / "odds_receipt.json"
    receipt_path.write_bytes(receipt_path.read_bytes().replace(b"2.5", b"2.6", 1))
    with pytest.raises(PredictionBlocked) as captured:
        replay_bundle(bundle)
    assert captured.value.code == "REPLAY_TAMPERED"


def test_residual_bundle_replay_reruns_scorer_at_original_timestamp(tmp_path: Path):
    result = run_prediction(args(tmp_path), dependencies())
    calls: list[dict[str, Any]] = []

    def replay_score(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(kwargs)
        return fake_score_residual(**kwargs)

    assert replay_bundle(Path(result["bundle"]), replay_score) == result
    assert len(calls) == 1
    assert calls[0]["score_timestamp"] == datetime.fromisoformat(
        result["score_timestamp"]
    )
    replay_paths = result["feature_identity"]["replay_paths"]
    assert set(replay_paths) == {
        "capture",
        "feature_manifest",
        "feature_rows",
        "form_csv",
        "implementation_manifest",
        "manifest",
        "model",
        "sidecar",
    }
    assert all(not Path(path).is_absolute() for path in replay_paths.values())


def test_refreshed_sources_must_be_regular_adjacent_bundle_files(tmp_path: Path):
    outside = tmp_path / "outside.csv"
    outside.write_text("dog_name,box_number\nAlpha,1\nBeta,2\n", encoding="utf-8")

    def unsafe_refresh(
        target: Mapping[str, Any], bundle: Path, now: datetime, days: int
    ) -> tuple[Path, Path]:
        del target, now, days
        source_dir = bundle / "source"
        source_dir.mkdir(parents=True)
        form_link = source_dir / "fixture.csv"
        os.symlink(outside, form_link)
        sidecar = form_link.with_name(form_link.name + ".metadata.json")
        sidecar.write_bytes(canonical_bytes({"participants": []}))
        return form_link, sidecar

    deps = dependencies(discover=lambda **kwargs: None)
    deps.refresh = unsafe_refresh
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path, odds_source="capture"), deps)
    assert captured.value.code == "BUNDLE_SOURCE_UNSAFE"
    bundle = Path(captured.value.details["bundle"])
    blocked_result = json.loads((bundle / "result.json").read_bytes())
    assert blocked_result["status"] == "BUNDLE_SOURCE_UNSAFE"


def test_output_symlink_write_attempt_is_rejected(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "bundle-link"
    os.symlink(real, link)
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path, output_root=link), dependencies())
    assert captured.value.code == "OUTPUT_ROOT_UNSAFE"


def test_bundle_writer_closes_descriptor_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "result.json"
    real_close = os.close
    closed: list[int] = []

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    def fail_fdopen(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(on_demand.os, "close", record_close)
    monkeypatch.setattr(on_demand.os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="injected fdopen failure"):
        write_exact_bytes(target, b"payload")
    assert len(closed) == 1
    assert not target.exists()


def test_frozen_artifact_tampering_is_detected(tmp_path: Path):
    def bad_score(**kwargs: Any) -> Mapping[str, Any]:
        result = dict(fake_score_residual(**kwargs))
        result["model_sha256"] = hashlib.sha256(b"tampered").hexdigest()
        return result

    deps = dependencies()
    deps.score_residual = bad_score
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == "FROZEN_MODEL_DRIFT"
