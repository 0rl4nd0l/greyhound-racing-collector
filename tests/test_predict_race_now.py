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
from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
)
from scripts.predict_market_form_residual import score_from_artifacts
from scripts.predict_race_now import (
    CaptureHandoffError,
    CollectorLockBusy,
    _acquire_collector_lock_no_steal,
    _release_owned_collector_lock,
    main,
    replay_bundle,
    resolve_target_race,
    run_prediction,
)
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
        "collector_request_root": tmp_path / "collector-requests",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def config_with_response_wait(tmp_path: Path, value: Any) -> Path:
    config = json.loads(Path("configs/prediction/manual-default.json").read_bytes())
    config["bundle"]["collector_response_wait_seconds"] = value
    path = tmp_path / "prediction-config.json"
    path.write_bytes(canonical_bytes(config))
    return path


def protocol_handoff(captured_at: datetime) -> dict[str, Any]:
    value = handoff(captured_at)
    value.update(
        {
            "schema_version": "on_demand_verified_master_packet_v1",
            "packet_record_schema_version": "market_form_residual_shadow_record_v3",
            "packet_record_checksum_sha256": "d" * 64,
            "packet_effective_state_schema_version": (
                "market_form_residual_effective_state_v2"
            ),
            "packet_effective_state_sha256": "e" * 64,
        }
    )
    return value


def test_master_packet_adapter_reuses_pr56_validated_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    form = tmp_path / "gunnedah-r5.csv"
    sidecar = form.with_name(form.name + ".metadata.json")
    feature_rows = tmp_path / "shadow_feature_rows.json"
    feature_manifest = tmp_path / "shadow_manifest.json"
    implementation_manifest = tmp_path / "implementation_file_manifest.json"
    capture = tmp_path / "autonomous_live_odds_capture_report.json"
    payloads = {
        form: b"dog_name,box_number\nAlpha,1\nBeta,2\n",
        sidecar: canonical_bytes({"prejump_shadow_metadata": {"status": "PASS"}}),
        feature_rows: canonical_bytes([]),
        feature_manifest: canonical_bytes({"safe": True}),
        implementation_manifest: canonical_bytes({"safe": True}),
        capture: canonical_bytes({"attempts": []}),
    }
    for path, raw in payloads.items():
        path.write_bytes(raw)
    packet = {
        "race_id": RACE_ID,
        "form_csv_path": form,
        "sidecar_path": sidecar,
        "feature_rows_path": feature_rows,
        "feature_manifest_path": feature_manifest,
        "implementation_manifest_path": implementation_manifest,
        "capture_path": capture,
    }
    score_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        predict_now,
        "discover_race_artifacts",
        lambda **kwargs: packet,
    )

    def validated_score(**kwargs: Any) -> Mapping[str, Any]:
        score_calls.append(kwargs)
        return {
            "jump_timestamp": (NOW + timedelta(hours=1)).isoformat(),
            "odds_append_timestamp": NOW.isoformat(),
            "record_schema_version": "market_form_residual_shadow_record_v3",
            "record_checksum_sha256": "record-checksum",
            "effective_state_schema_version": "market_form_residual_effective_state_v2",
            "effective_state_sha256": "effective-state",
            "input_hashes": {
                "capture_artifact_sha256": sha256_bytes(payloads[capture]),
                "form_csv_sha256": sha256_bytes(payloads[form]),
                "sidecar_sha256": sha256_bytes(payloads[sidecar]),
            },
        }

    monkeypatch.setattr(predict_now, "score_from_artifacts", validated_score)
    result = predict_now.discover_capture_handoff(
        evidence_roots=[tmp_path],
        db_path=tmp_path / "unused.db",
        race_id=RACE_ID,
        jump_datetime=NOW + timedelta(hours=1),
        capture_window_minutes=60,
        current_time=NOW,
    )

    assert result is not None
    assert result["schema_version"] == "on_demand_verified_master_packet_v1"
    assert result["packet_record_schema_version"].endswith("record_v3")
    assert result["packet_effective_state_schema_version"].endswith(
        "effective_state_v2"
    )
    assert result["_report_bytes"] == payloads[capture]
    assert score_calls[0]["score_timestamp"] == NOW
    assert score_calls[0]["capture_path"] == capture


def test_default_schedule_uses_current_browser_api_and_cleans_scratch(
    monkeypatch: pytest.MonkeyPatch,
):
    import upcoming_race_browser

    scratch_paths: list[Path] = []

    class Browser:
        def __init__(self) -> None:
            scratch = Path(os.environ["UPCOMING_RACES_DIR"])
            assert scratch.is_dir()
            scratch_paths.append(scratch)

        def get_upcoming_races(self, *, days_ahead: int):
            assert days_ahead == 2
            return [race()]

    monkeypatch.setattr(upcoming_race_browser, "UpcomingRaceBrowser", Browser)
    monkeypatch.setenv("UPCOMING_RACES_DIR", "owner-value")

    assert predict_now._default_schedule(2) == [race()]
    assert os.environ["UPCOMING_RACES_DIR"] == "owner-value"
    assert len(scratch_paths) == 1
    assert not scratch_paths[0].exists()


def test_default_refresh_downloads_only_exact_target_into_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.predict_market_form_residual as residual
    import upcoming_race_browser

    downloads: list[tuple[str, Mapping[str, Any]]] = []
    runtime_paths: list[Path] = []

    class Browser:
        def __init__(self) -> None:
            self.output = Path(os.environ["UPCOMING_RACES_DIR"])

        def download_race_csv(
            self, url: str, *, race_info_hint: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            runtime_paths.append(Path.cwd())
            downloads.append((url, race_info_hint))
            form = self.output / "Race 5 - GUNN - 2026-07-19.csv"
            form.write_bytes(b"dog_name,box_number\nAlpha,1\nBeta,2\n")
            form.with_name(form.name + ".metadata.json").write_bytes(
                canonical_bytes({"safe": True})
            )
            return {"success": True, "filepath": str(form)}

    monkeypatch.setattr(upcoming_race_browser, "UpcomingRaceBrowser", Browser)
    monkeypatch.setattr(
        residual,
        "_sidecar_context",
        lambda value: {
            "expected_race_id": RACE_ID,
            "jump_timestamp": NOW + timedelta(hours=1),
        },
    )
    binding_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        residual,
        "_validate_form_binding",
        lambda *args, **kwargs: binding_calls.append(kwargs),
    )
    monkeypatch.setenv("UPCOMING_RACES_DIR", "owner-value")

    form, sidecar = predict_now._default_refresh(race(), tmp_path / "bundle", NOW, 1)

    assert downloads == [(race()["url"], race())]
    assert form.parent == tmp_path / "bundle/source/upcoming"
    assert sidecar == form.with_name(form.name + ".metadata.json")
    assert binding_calls[0]["form_csv_path"] == form
    assert os.environ["UPCOMING_RACES_DIR"] == "owner-value"
    assert len(runtime_paths) == 1
    assert runtime_paths[0].parent == (tmp_path / "bundle").resolve()
    assert not runtime_paths[0].exists()


def test_source_backed_refresh_payload_emits_complete_exact_grade_proof(tmp_path: Path):
    from utils.csv_metadata import build_csv_download_provenance_payload

    race_url = "https://www.thedogs.com.au/racing/mandurah/2030-06-09/1/test"
    grade_proof = {
        "target_grade_context_schema": "thedogs_meeting_card_exact_race_v1",
        "target_grade_equivalence_key": "MAIDEN",
        "target_grade_exact_value": "Maiden",
        "target_grade_race_date": "2030-06-09",
        "target_grade_race_number": 1,
        "target_grade_race_url": race_url,
        "target_grade_source_url": ("https://www.thedogs.com.au/racing/2030-06-09"),
        "target_grade_source_sha256": "a" * 64,
        "target_grade_venue": "MAND",
    }
    participants = [
        {"box_number": 1, "dog_name": "Alpha"},
        {"box_number": 2, "dog_name": "Beta"},
    ]
    payload = build_csv_download_provenance_payload(
        filepath=tmp_path / "Race 1 - MAND - 2030-06-09.csv",
        race_url=race_url,
        csv_info={"type": "direct_csv", "url": race_url},
        content="Dog Name|BOX\n1. Alpha|1\n2. Beta|2\n",
        completeness={
            "status": "COMPLETE",
            "runner_count": 2,
            "participants": participants,
        },
        race_info={
            "date": "2030-06-09",
            "venue": "MAND",
            "race_number": 1,
            "race_time": "1:00 PM",
            "distance": "400m",
            "grade": "Maiden",
            "target_grade": "Maiden",
            "target_grade_source": "thedogs_meeting_card_exact_race",
            "url": race_url,
            **grade_proof,
        },
        normalization={
            "canonical_runner_alignment": {
                "status": "aligned",
                "canonical_runner_set_status": "available",
                "canonical_source_url": race_url,
            },
            "runner_completeness_after_canonical_alignment": {
                "status": "COMPLETE",
                "runner_count": 2,
                "participants": participants,
            },
        },
        filename="Race 1 - MAND - 2030-06-09.csv",
    )

    assert {key: payload[key] for key in grade_proof} == grade_proof
    assert {key: payload["race_info"][key] for key in grade_proof} == grade_proof
    assert payload["prejump_shadow_metadata"]["status"] == "PASS"


def test_murray_bridge_meetings_use_distinct_authoritative_url_identities():
    common = {
        "venue": "MURR",
        "race_number": 1,
        "date": "2030-06-09",
        "race_time": "13:00",
    }
    bridge = {
        **common,
        "venue_name": "Murray Bridge",
        "url": ("https://www.thedogs.com.au/racing/murray-bridge/2030-06-09/1/test"),
    }
    straight = {
        **common,
        "venue_name": "Murray Bridge Straight",
        "url": (
            "https://www.thedogs.com.au/racing/murray-bridge-straight/2030-06-09/1/test"
        ),
    }

    assert (
        resolve_target_race(
            [bridge, straight], race_id=None, race_query="murray bridge r1"
        )[1]
        == bridge
    )
    assert (
        resolve_target_race(
            [bridge, straight], race_id=None, race_query="murray bridge straight r1"
        )[1]
        == straight
    )
    status, selected, matches = resolve_target_race(
        [bridge, straight], race_id=None, race_query="murr r1"
    )
    assert status == "BLOCKED_RACE_AMBIGUOUS"
    assert selected is None
    assert matches == ["Race 1 - MURR - 2030-06-09"]


def test_live_feature_seal_hashes_exact_current_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.run_feature_recovery_execution_v1 as recovery
    import scripts.run_shadow_non_tgr_rf_evaluation as feature_builder
    from scripts.predict_market_form_residual import FEATURE_GENERATOR_FILES

    form = tmp_path / "Race 5 - GUNN - 2026-07-19.csv"
    form.write_text("Dog Name|BOX\n1. Alpha|1\n2. Beta|2\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "load_json", lambda path: {"schema": "fixture"})
    monkeypatch.setattr(
        feature_builder, "validate_schema_contract", lambda schema: {"status": "PASS"}
    )
    monkeypatch.setattr(
        feature_builder,
        "build_live_feature_rows",
        lambda **kwargs: [{"race_id": RACE_ID, "box_number": 1}],
    )
    monkeypatch.setattr(
        feature_builder,
        "same_distance_same_grade_history_provenance_report",
        lambda rows: {"status": "PASS", "row_count": len(rows)},
    )
    monkeypatch.setattr(
        feature_builder, "shadow_relpath", lambda path: str(Path(path).resolve())
    )

    sealed = predict_now.seal_live_features(
        form_csv=form,
        db_path=tmp_path / "fixture.db",
        output_dir=tmp_path / "sealed",
        current_time=NOW,
    )
    implementation = json.loads(sealed["implementation_manifest"].read_bytes())

    assert implementation["implementation_files"] == list(FEATURE_GENERATOR_FILES)
    assert implementation["implementation_file_hashes"] == {
        relative: sha256_file(predict_now.ROOT / relative)
        for relative in FEATURE_GENERATOR_FILES
    }


def test_master_packet_adapter_rejects_pr56_jump_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    packet = {
        "race_id": RACE_ID,
        "form_csv_path": tmp_path / "form.csv",
        "sidecar_path": tmp_path / "form.csv.metadata.json",
        "feature_rows_path": tmp_path / "shadow_feature_rows.json",
        "feature_manifest_path": tmp_path / "shadow_manifest.json",
        "implementation_manifest_path": tmp_path / "implementation_file_manifest.json",
        "capture_path": tmp_path / "capture.json",
    }
    monkeypatch.setattr(predict_now, "discover_race_artifacts", lambda **kwargs: packet)
    monkeypatch.setattr(
        predict_now,
        "score_from_artifacts",
        lambda **kwargs: {
            "jump_timestamp": (NOW + timedelta(hours=2)).isoformat(),
            "odds_append_timestamp": NOW.isoformat(),
            "input_hashes": {},
        },
    )

    with pytest.raises(CaptureHandoffError, match="capture_packet_jump_mismatch"):
        predict_now.discover_capture_handoff(
            evidence_roots=[tmp_path],
            db_path=tmp_path / "unused.db",
            race_id=RACE_ID,
            jump_datetime=NOW + timedelta(hours=1),
            capture_window_minutes=60,
            current_time=NOW,
        )


def test_master_packet_adapter_treats_mismatched_exact_race_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def discover(**kwargs: Any) -> Mapping[str, Any]:
        if kwargs.get("exact_race_id") == RACE_ID:
            raise CaptureHandoffError("race_feature_packet_not_found")
        return {"race_id": "Race 5 - GUNN - 2026-07-18"}

    monkeypatch.setattr(predict_now, "discover_race_artifacts", discover)

    assert (
        predict_now.discover_capture_handoff(
            evidence_roots=[tmp_path],
            db_path=tmp_path / "unused.db",
            race_id=RACE_ID,
            jump_datetime=NOW + timedelta(hours=1),
            capture_window_minutes=60,
            current_time=NOW,
        )
        is None
    )


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


def test_default_fetch_preserves_master_fixed_window_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.autonomous_live_odds_capture as capture

    form = tmp_path / "form.csv"
    form.write_text("dog_name,box_number\nAlpha,1\nBeta,2\n", encoding="utf-8")
    fetch_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        capture,
        "build_plan_item",
        lambda path, current_time: {
            "race_id": RACE_ID,
            "blockers": ["outside_capture_windows"],
        },
    )
    monkeypatch.setattr(
        capture,
        "fetch_odds_for_target_race_with_timeout",
        lambda *call_args, **call_kwargs: fetch_calls.append(call_args),
    )

    with pytest.raises(PredictionBlocked) as captured:
        predict_now._default_fetch(
            {
                "race_id": RACE_ID,
                "form_csv": str(form),
                "jump_timestamp": datetime.now().astimezone() + timedelta(hours=2),
            },
            tmp_path / "sealed.db",
            1.0,
        )

    assert captured.value.code == "CAPTURE_WINDOW_UNAVAILABLE"
    assert fetch_calls == []


def test_existing_receipt_bypasses_request_and_collector_lock(tmp_path: Path):
    calls = {"acquire": 0}
    deps = dependencies()
    deps.acquire_lock = lambda: calls.__setitem__("acquire", calls["acquire"] + 1)

    result = run_prediction(args(tmp_path), deps)

    assert result["odds_source"] == "verified_autonomous_receipt"
    assert calls["acquire"] == 0
    assert not (tmp_path / "collector-requests").exists()


def test_request_response_receipt_continues_existing_scoring_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = {"value": 0.0}
    calls = {"discover": 0, "score": 0, "published": False}
    request_root = tmp_path / "collector-requests"
    protocol = ManualPredictionCollectorProtocol(request_root)
    captured_handoff: dict[str, Any] | None = None

    def discover(**kwargs: Any) -> Mapping[str, Any] | None:
        calls["discover"] += 1
        return captured_handoff

    def sleep(seconds: float) -> None:
        nonlocal captured_handoff
        clock["value"] += seconds
        if calls["published"]:
            return
        context = protocol.prepare_collector_request(
            now=NOW + timedelta(seconds=clock["value"]),
            collector_run_id="scheduled-run-1",
            active_capture=False,
        )
        assert context is not None
        protocol.begin_attempt(
            context,
            now=NOW + timedelta(seconds=clock["value"]),
            collector_run_id="scheduled-run-1",
        )
        captured_handoff = protocol_handoff(
            NOW + timedelta(seconds=clock["value"])
        )
        from scripts.shadow_autopilot_v1 import (
            finalize_manual_collector_request,
        )

        monkeypatch.setattr(
            predict_now,
            "discover_capture_handoff",
            lambda **kwargs: captured_handoff,
        )
        response = finalize_manual_collector_request(
            protocol=protocol,
            context=context,
            capture_report={
                "attempts": [
                    {
                        "race_id": RACE_ID,
                        "status": "APPENDED",
                        "capture_window_minutes": 60,
                    }
                ]
            },
            evidence_root=tmp_path / "evidence",
            db_path=tmp_path / "source.db",
            current_time=NOW + timedelta(seconds=clock["value"]),
        )
        assert response["status"] == "RECEIPT_READY"
        calls["published"] = True

    deps = dependencies(
        discover=discover,
        acquire=lambda: pytest.fail("manual predictor must not acquire lock"),
        monotonic=lambda: clock["value"],
        sleep=sleep,
    )
    original_score = deps.score_residual

    def score(**kwargs: Any) -> Mapping[str, Any]:
        calls["score"] += 1
        return original_score(**kwargs)

    deps.score_residual = score
    result = run_prediction(args(tmp_path), deps)

    assert result["odds_source"] == "verified_autonomous_receipt"
    assert calls == {"discover": 2, "score": 1, "published": True}
    consumes = list((request_root / "consumed").glob("*.json"))
    assert len(consumes) == 1


def test_request_wait_has_finite_deadline_without_lock_attempt(tmp_path: Path):
    clock = {"value": 0.0}
    calls = {"acquire": 0}

    def sleep(seconds: float) -> None:
        clock["value"] += seconds

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(
            args(
                tmp_path,
                config=config_with_response_wait(tmp_path, 2),
            ),
            dependencies(
                discover=lambda **kwargs: None,
                acquire=lambda: calls.__setitem__(
                    "acquire", calls["acquire"] + 1
                ),
                monotonic=lambda: clock["value"],
                sleep=sleep,
            ),
        )
    assert captured.value.code == "COLLECTOR_RESPONSE_TIMEOUT"
    assert clock["value"] == 2.0
    assert calls["acquire"] == 0


def test_capture_source_cannot_create_second_capture_authority(tmp_path: Path):
    calls = {"acquire": 0, "fetch": 0}
    deps = dependencies(discover=lambda **kwargs: None)
    deps.acquire_lock = lambda: calls.__setitem__("acquire", calls["acquire"] + 1)
    deps.fetch_odds = lambda *args, **kwargs: calls.__setitem__(
        "fetch", calls["fetch"] + 1
    )

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path, odds_source="capture"), deps)

    assert captured.value.code == "CAPTURE_AUTHORITY_FORBIDDEN"
    assert calls == {"acquire": 0, "fetch": 0}


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


def test_manual_predictor_does_not_mutate_or_interfere_with_lock_owner(
    tmp_path: Path,
):
    lock_path = tmp_path / "collector.lock"
    original = canonical_bytes({"pid": 1234, "run_id": "scheduled_collector"})
    lock_path.write_bytes(original)
    lock_path.chmod(0o640)
    before = lock_path.stat(follow_symlinks=False)

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(
            args(tmp_path, odds_source="capture"),
            dependencies(discover=lambda **kwargs: None),
        )
    assert captured.value.code == "CAPTURE_AUTHORITY_FORBIDDEN"

    after = lock_path.stat(follow_symlinks=False)
    assert lock_path.read_bytes() == original
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )


def test_default_dependencies_reject_db_lock_root_mismatch(tmp_path: Path):
    namespace = argparse.Namespace(
        db=tmp_path / "canonical/greyhound_racing_data.db",
        lock_path=tmp_path / "different/shadow_autopilot.lock",
        lock_output_dir=tmp_path / "bundles",
    )
    dependency_set = predict_now.default_dependencies(namespace)

    with pytest.raises(PredictionBlocked) as captured:
        dependency_set.acquire_lock()

    assert captured.value.code == "LOCK_PATH_DB_ROOT_MISMATCH"
    assert captured.value.details["expected_lock_path"].endswith(
        "canonical/artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_daemon_runtime/shadow_autopilot.lock"
    )


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


@pytest.mark.parametrize(
    "reason",
    [
        "sidecar_target_grade_context_schema_missing",
        "sidecar_target_grade_exact_value_missing",
        "sidecar_target_grade_equivalence_key_missing",
        "sidecar_target_grade_race_url_missing",
        "sidecar_target_grade_source_url_missing",
        "sidecar_target_grade_source_sha256_missing",
        "sidecar_target_grade_race_date_missing",
        "sidecar_target_grade_race_number_missing",
        "sidecar_target_grade_venue_missing",
        "feature_generator_implementation_hash_mismatch",
    ],
)
def test_auto_rejects_precurrent_packet_then_requests_collector_without_effects(
    tmp_path: Path, reason: str
):
    clock = {"value": 0.0}
    calls = {"acquire": 0, "refresh": 0, "score": 0}

    def rejected_packet(**kwargs: Any) -> Mapping[str, Any]:
        del kwargs
        raise CaptureHandoffError(reason)

    deps = dependencies(
        discover=rejected_packet,
        monotonic=lambda: clock["value"],
        sleep=lambda seconds: clock.__setitem__(
            "value", clock["value"] + seconds
        ),
    )
    deps.refresh = lambda *args, **kwargs: calls.__setitem__(
        "refresh", calls["refresh"] + 1
    )
    deps.score_residual = lambda **kwargs: calls.__setitem__(
        "score", calls["score"] + 1
    )
    deps.acquire_lock = lambda: calls.__setitem__(
        "acquire", calls["acquire"] + 1
    )
    command_args = args(
        tmp_path,
        config=config_with_response_wait(tmp_path, 2),
    )
    database_before = sha256_file(command_args.db)

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(command_args, deps)

    assert captured.value.code == "COLLECTOR_RESPONSE_TIMEOUT"
    assert calls == {"acquire": 0, "refresh": 0, "score": 0}
    assert sha256_file(command_args.db) == database_before
    assert len(list((tmp_path / "collector-requests" / "requests").glob("*.json"))) == 1
    assert not list(tmp_path.rglob("*.service"))
    assert not list(tmp_path.rglob("*prediction_history*"))


def test_receipt_only_mode_does_not_fallback_from_precurrent_packet(tmp_path: Path):
    calls = {"acquire": 0, "score": 0}

    def rejected_packet(**kwargs: Any) -> Mapping[str, Any]:
        del kwargs
        raise CaptureHandoffError("sidecar_target_grade_context_schema_missing")

    deps = dependencies(discover=rejected_packet)
    deps.acquire_lock = lambda: calls.__setitem__("acquire", calls["acquire"] + 1)
    deps.score_residual = lambda **kwargs: calls.__setitem__(
        "score", calls["score"] + 1
    )
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path, odds_source="receipt"), deps)

    assert captured.value.code == "RECEIPT_INVALID"
    assert captured.value.details["reason"] == (
        "sidecar_target_grade_context_schema_missing"
    )
    assert calls == {"acquire": 0, "score": 0}


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        ("accepted_capture_attempt_ambiguous", "RECEIPT_AMBIGUOUS"),
        ("target_grade_proof_mismatch", "RECEIPT_INVALID"),
    ],
)
def test_auto_keeps_ambiguous_or_conflicting_receipt_evidence_terminal(
    tmp_path: Path, reason: str, code: str
):
    calls = {"acquire": 0, "score": 0}

    def rejected_packet(**kwargs: Any) -> Mapping[str, Any]:
        del kwargs
        raise CaptureHandoffError(reason)

    deps = dependencies(discover=rejected_packet)
    deps.acquire_lock = lambda: calls.__setitem__("acquire", calls["acquire"] + 1)
    deps.score_residual = lambda **kwargs: calls.__setitem__(
        "score", calls["score"] + 1
    )
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)

    assert captured.value.code == code
    assert captured.value.details["reason"] == reason
    assert calls == {"acquire": 0, "score": 0}


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
    deps = dependencies(
        discover=lambda **kwargs: handoff(NOW + timedelta(hours=1))
    )
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == "RECEIPT_STALE"
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


def test_history_seal_never_selects_target_or_future_outcome_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    create_db(source)
    statements: list[str] = []
    original_connect = on_demand.sqlite3.connect

    class ReadProxy:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        @property
        def row_factory(self):
            return self.connection.row_factory

        @row_factory.setter
        def row_factory(self, value: Any) -> None:
            self.connection.row_factory = value

        def execute(self, sql: str, parameters: Any = ()):
            statements.append(" ".join(sql.split()))
            return self.connection.execute(sql, parameters)

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

    def traced_connect(database: Any, *args: Any, **kwargs: Any):
        connection = original_connect(database, *args, **kwargs)
        return (
            ReadProxy(connection) if str(database).startswith("file:") else connection
        )

    monkeypatch.setattr(on_demand.sqlite3, "connect", traced_connect)
    seal_history_database(
        source=source,
        target=tmp_path / "sealed.db",
        target_race_id=RACE_ID,
        cutoff=NOW + timedelta(hours=1),
        runner_names=["Alpha", "Beta"],
    )

    assert 'SELECT "race_id", "race_date" FROM race_metadata' in statements
    assert 'SELECT * FROM "race_metadata" WHERE "race_id" IN (?)' in statements
    assert 'SELECT * FROM "dog_race_data" WHERE "race_id" IN (?)' in statements
    assert "SELECT * FROM dog_race_data" not in statements
    assert "SELECT * FROM race_metadata" not in statements


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


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_config_is_rejected_before_any_prediction_side_effect(
    tmp_path: Path, constant: str
):
    config = tmp_path / "nonfinite.json"
    raw = (
        '{"bundle":{"collector_response_wait_seconds":'
        f"{constant}"
        ',"poll_seconds":1,"receipt_max_age_seconds":900},'
        '"model":"market_only_v1",'
        '"schema_version":"on_demand_prediction_config_v1",'
        '"variant":"market_only_implied"}\n'
    ).encode()
    config.write_bytes(raw)
    assert canonical_bytes(json.loads(raw)) == raw

    command_args = args(
        tmp_path,
        model="market-only",
        config=config,
        odds_source="capture",
    )
    database_before = sha256_file(command_args.db)
    files_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    calls = {
        "schedule": 0,
        "refresh": 0,
        "discover_receipt": 0,
        "fetch_odds": 0,
        "acquire_lock": 0,
        "release_lock": 0,
        "seal_features": 0,
        "score_residual": 0,
        "sleep": 0,
    }
    deps = dependencies()
    for name in calls:
        original = getattr(deps, name)

        def counted(
            *call_args: Any, _name=name, _original=original, **call_kwargs: Any
        ):
            calls[_name] += 1
            return _original(*call_args, **call_kwargs)

        setattr(deps, name, counted)

    for _ in range(2):
        with pytest.raises(PredictionBlocked) as captured:
            run_prediction(command_args, deps)
        assert captured.value.code == "CONFIG_INVALID_JSON"

    assert calls == {name: 0 for name in calls}
    assert sha256_file(command_args.db) == database_before
    assert (
        sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
        == files_before
    )


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (-1, "CONFIG_SCHEMA_MISMATCH"),
        (0, "CONFIG_SCHEMA_MISMATCH"),
        (901, "CONFIG_SCHEMA_MISMATCH"),
        (True, "CONFIG_SCHEMA_MISMATCH"),
        ("1", "CONFIG_SCHEMA_MISMATCH"),
        (float("nan"), "CONFIG_INVALID_JSON"),
        (float("inf"), "CONFIG_INVALID_JSON"),
        (float("-inf"), "CONFIG_INVALID_JSON"),
    ],
)
def test_invalid_response_wait_values_are_rejected(
    tmp_path: Path, value: Any, code: str
):
    calls = {"schedule": 0}
    deps = dependencies()
    deps.schedule = lambda days: calls.__setitem__("schedule", calls["schedule"] + 1)

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(
            args(tmp_path, config=config_with_response_wait(tmp_path, value)),
            deps,
        )

    assert captured.value.code == code
    assert calls["schedule"] == 0


def test_checked_in_response_wait_default_and_max_are_bounded():
    for name in ("manual-default.json", "market-only.json"):
        config = json.loads((Path("configs/prediction") / name).read_bytes())
        assert config["bundle"]["collector_response_wait_seconds"] == 600

    for name in ("market_form_residual_v1.schema.json", "market_only_v1.schema.json"):
        schema = json.loads((Path("configs/prediction/schemas") / name).read_bytes())
        wait_schema = schema["properties"]["bundle"]["properties"][
            "collector_response_wait_seconds"
        ]
        assert wait_schema == {"maximum": 900, "minimum": 1, "type": "number"}


def test_list_configs_is_finite_validated_and_deterministic(
    capsys: pytest.CaptureFixture[str],
):
    first = predict_now.list_configs()
    second = predict_now.list_configs()
    assert canonical_bytes(first) == canonical_bytes(second)
    assert [row["name"] for row in first["configs"]] == [
        "market-form-residual-v1",
        "market-only",
    ]
    assert [row["model"]["resolved"] for row in first["configs"]] == [
        "market_form_residual_v1",
        "market_only_v1",
    ]

    assert main(["--list-configs"]) == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == first
    assert stdout == canonical_bytes(first).decode()


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
