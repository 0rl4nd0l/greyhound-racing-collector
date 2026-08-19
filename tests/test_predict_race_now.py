from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sqlite3
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import race_collection.manual_prediction_collector_request as collector_protocol
import race_collection.synchronous_manual_capture as synchronous_capture
import scripts.predict_race_now as predict_now
import src.predictor.on_demand as on_demand
from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
)
from race_collection.synchronous_manual_capture import (
    CollectorBusy as CollectorLockBusy,
)
from race_collection.synchronous_manual_capture import (
    acquire_collector_lock_no_steal as _acquire_collector_lock_no_steal,
)
from race_collection.synchronous_manual_capture import (
    release_owned_collector_lock as _release_owned_collector_lock,
)
from scripts.predict_market_form_residual import score_from_artifacts
from scripts.predict_race_now import (
    CaptureHandoffError,
    main,
    replay_bundle,
    resolve_target_race,
    run_prediction,
)
from src.predictor.on_demand import (
    Dependencies,
    PredictionBlocked,
    canonical_bytes,
    receipt_from_handoff,
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
        "url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5",
        "participants": [
            {"box": 1, "display_name": "Alpha", "identity": "ALPHA", "source_native_runner_id": "101"},
            {"box": 2, "display_name": "Beta", "identity": "BETA", "source_native_runner_id": None},
        ],
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
    now=lambda: NOW,
    monotonic=lambda: 0.0,
    capture_one=None,
) -> Dependencies:
    def seal_synchronous_receipt(**values: Any) -> Mapping[str, Any]:
        protocol = ManualPredictionCollectorProtocol(values["protocol_root"])
        context = protocol.claim_request(
            values["request_id"],
            now=NOW,
            collector_run_id="fixture_capture_one",
        )
        protocol.begin_attempt(
            context,
            now=NOW,
            collector_run_id="fixture_capture_one",
        )
        value = handoff()
        output = Path(values["output_dir"])
        output.mkdir(parents=True)
        paths = {
            label: output / name
            for label, name in (
                ("report", "capture.json"),
                ("form", "gunnedah-r5.csv"),
                ("sidecar", "gunnedah-r5.csv.metadata.json"),
            )
        }
        for label, path in paths.items():
            path.write_bytes(value[f"_{label}_bytes"])
            value[f"_{label}_path"] = path.resolve()
        normalized, _, _, _ = receipt_from_handoff(
            value,
            current_time=NOW,
            max_age_seconds=900,
        )
        value.update(
            {
                "schema_version": "on_demand_verified_collector_capture_v2",
                "race": dict(context.request["race"]),
                "runner_set_sha256": normalized["runner_set_sha256"],
                "capture_attempt_sha256": "a" * 64,
                "append_report_sha256": "b" * 64,
            }
        )
        response = protocol.publish_receipt_ready(
            context,
            now=NOW,
            handoff=value,
            normalized_receipt=normalized,
        )
        return {
            "schema_version": "collector_capture_one_result_v1",
            "status": response["status"],
            "request_id": values["request_id"],
            "appended_attempt_count": 1,
        }

    return Dependencies(
        schedule=lambda *values: [race()],
        seal_features=fake_seal_features,
        score_residual=fake_score_residual,
        now=now,
        capture_one=capture_one or seal_synchronous_receipt,
        monotonic=monotonic,
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
        "job_id": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def config_with_discovery_budget(tmp_path: Path, value: Any) -> Path:
    config = json.loads(Path("configs/prediction/manual-default.json").read_bytes())
    config["bundle"]["latency_budget"]["discovery_seconds"] = value
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


def test_legacy_refresh_report_is_not_substituted_for_collector_owned_bounded_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    evidence_root = tmp_path / "evidence"
    source = evidence_root / "shadow_autopilot_v1_fixture/odds_capture_refresh_report.json"
    state = evidence_root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        canonical_bytes(
            {
                "generated_at": NOW.isoformat(),
                "selected_count": 1,
                "selected_races": [
                    {
                        "date": "2026-07-19",
                        "jump_datetime": "2026-07-19T13:00:00+10:00",
                        "race_id": RACE_ID,
                        "race_id_aliases": [RACE_ID],
                        "race_number": 5,
                        "race_time": "13:00",
                        "race_url": (
                            "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
                        ),
                        "venue": "GUNN",
                    }
                ],
            }
        )
    )
    published = synchronous_capture.publish_current_race_index(
        state_path=state,
        evidence_root=evidence_root,
        source_refresh_report_path=source,
        run_id="fixture",
    )
    browser_sentinel = object()
    monkeypatch.setitem(sys.modules, "upcoming_race_browser", browser_sentinel)

    with pytest.raises(synchronous_capture.CaptureOneRejected) as captured:
        predict_now._default_schedule(
            NOW,
            12,
            synchronous_capture.current_race_index_path(state),
            evidence_root,
            900,
        )

    assert published["status"] == "REJECTED"
    assert captured.value.code == "CURRENT_INDEX_UNAVAILABLE"
    assert sys.modules["upcoming_race_browser"] is browser_sentinel


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

    from race_collection.synchronous_manual_capture import refresh_exact_race

    target = {
        **race(),
        "race_id": RACE_ID,
        "jump_timestamp": (NOW + timedelta(hours=1)).isoformat(),
    }
    form, sidecar = refresh_exact_race(target, tmp_path / "bundle", NOW)

    assert downloads == [(race()["url"], target)]
    assert form.parent == tmp_path / "bundle/exact_upcoming"
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
    assert (
        resolve_target_race(
            [bridge, straight],
            race_id=None,
            race_query=None,
            race_url=straight["url"],
        )[1]
        == straight
    )


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

    assert result["schema_version"] == "on_demand_race_prediction_v2"
    assert result["status"] == "PREDICTION_READY"
    assert result["model"]["resolved"] == "market_form_residual_v1"
    assert result["model"]["alias_resolved"] is True
    assert sum(row["probability"] for row in result["prediction"]["predictions"]) == pytest.approx(1.0)
    index = json.loads((tmp_path / "bundles/prediction_bundle_index_v1.json").read_bytes())
    bundle = tmp_path / "bundles" / index["entries"][0]["directory"]
    assert (bundle / "bundle_manifest.json").is_file()
    assert json.loads((bundle / "result.json").read_bytes()) == result


def scheduled_exact_receipt(
    tmp_path: Path,
) -> tuple[
    ManualPredictionCollectorProtocol,
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
]:
    protocol_root = tmp_path / "collector-requests"
    protocol = ManualPredictionCollectorProtocol(protocol_root)
    value = handoff()
    collector_run_id = "20260719T115500+1000_odds_capture"
    source_attempt = {
        **json.loads(value["_report_bytes"])["attempts"][0],
        "append_report": {
            "status": "SUCCESS",
            "append_only": True,
            "inserted_rows": 2,
        },
    }
    source_plan_item = {
        "schema_version": "autonomous_live_odds_capture_plan_item_v1",
        "status": "READY_TO_CAPTURE",
        "race_id": RACE_ID,
        "venue": "GUNN",
        "race_number": 5,
        "race_date": "2026-07-19",
        "race_time": "13:00",
        "jump_datetime": "2026-07-19T13:00:00+10:00",
        "thedogs_source_url": race()["url"],
        "expected_runners": [
            {"box_number": 1, "dog_name": "Alpha", "identity": "ALPHA"},
            {"box_number": 2, "dog_name": "Beta", "identity": "BETA"},
        ],
    }
    value["_report_bytes"] = canonical_bytes(
        {
            "schema_version": "collector_exact_capture_source_v1",
            "collector_run_id": collector_run_id,
            "generated_at": NOW.isoformat(),
            "race_id": RACE_ID,
            "source_race_id": RACE_ID,
            "source_plan_item": source_plan_item,
            "source_attempt": source_attempt,
            "attempts": [source_attempt],
        }
    )
    value["source_report_sha256"] = sha256_bytes(value["_report_bytes"])
    paths = {
        label: tmp_path / name
        for label, name in (
            ("report", "scheduled-capture.json"),
            ("form", "gunnedah-r5.csv"),
            ("sidecar", "gunnedah-r5.csv.metadata.json"),
        )
    }
    for label, path in paths.items():
        path.write_bytes(value[f"_{label}_bytes"])
        value[f"_{label}_path"] = path.resolve()
    normalized, _, _, _ = receipt_from_handoff(
        value,
        current_time=NOW,
        max_age_seconds=900,
    )
    value.update(
        {
            "schema_version": "on_demand_verified_collector_capture_v2",
            "race": {
                "race_id": RACE_ID,
                "url": race()["url"],
                "venue": "GUNN",
                "race_number": 5,
                "race_date": "2026-07-19",
                "jump_timestamp": "2026-07-19T13:00:00+10:00",
            },
            "runner_set_sha256": normalized["runner_set_sha256"],
            "capture_attempt_sha256": sha256_bytes(
                canonical_bytes(source_attempt)
            ),
            "append_report_sha256": sha256_bytes(
                canonical_bytes(source_attempt["append_report"])
            ),
        }
    )
    protocol.publish_collector_exact_receipt(
        collector_run_id=collector_run_id,
        emitted_at=NOW,
        handoff=value,
    )
    return protocol, value, paths, source_attempt


def test_scheduled_exact_receipt_reuses_while_capture_authority_is_busy(
    tmp_path: Path,
):
    protocol, value, paths, _ = scheduled_exact_receipt(tmp_path)

    calls = {"capture": 0, "score": 0}
    deps = dependencies(
        capture_one=lambda **_: calls.__setitem__(
            "capture", calls["capture"] + 1
        )
    )
    original_score = deps.score_residual

    def score(**kwargs: Any) -> Mapping[str, Any]:
        calls["score"] += 1
        return original_score(**kwargs)

    deps.score_residual = score
    result = run_prediction(
        args(tmp_path, collector_request_root=protocol.root),
        deps,
    )

    assert result["status"] == "PREDICTION_READY"
    assert calls == {"capture": 0, "score": 1}
    assert not protocol.outstanding_request_ids()
    assert result["evidence"]["protocol_chain"]["protocol_kind"] == (
        "collector_exact_capture_v1"
    )
    index = on_demand.verify_prediction_bundle_index(tmp_path / "bundles")
    assert len(index["entries"]) == 1
    bundle = tmp_path / "bundles" / index["entries"][0]["directory"]
    verified = on_demand.verify_indexed_prediction_bundle(
        tmp_path / "bundles", index["entries"][0]
    )
    assert verified.result == result
    assert (bundle / "protocol/collector_exact_receipt.json").is_file()
    assert not (bundle / "protocol/request.json").exists()
    contents = {
        path.relative_to(bundle).as_posix(): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    tampered = json.loads(contents["protocol/collector_exact_receipt.json"])
    tampered["collector_run_id"] = "substituted-run"
    contents["protocol/collector_exact_receipt.json"] = canonical_bytes(tampered)
    with pytest.raises(PredictionBlocked):
        on_demand._validate_sealed_protocol(contents, result)

    contents = {
        path.relative_to(bundle).as_posix(): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    contents["protocol/request.json"] = canonical_bytes({})
    with pytest.raises(PredictionBlocked) as contradictory_protocol:
        on_demand._validate_sealed_protocol(contents, result)
    assert contradictory_protocol.value.code == "PREDICTION_BUNDLE_INVALID"
    assert contradictory_protocol.value.details == {
        "reason": "sealed_protocol_required"
    }

    contents.pop("protocol/request.json")
    alias_result = json.loads(canonical_bytes(result))
    alias_receipt = json.loads(contents["protocol/collector_exact_receipt.json"])
    report_hash = alias_receipt["artifacts"]["report"]["sha256"]
    alias_receipt["form_name"] = "capture.json"
    alias_receipt["artifacts"]["form"] = {
        "path": "capture.json",
        "sha256": report_hash,
    }
    alias_receipt["sealed_handoff"]["source_form_sha256"] = report_hash
    alias_raw = canonical_bytes(alias_receipt)
    contents["protocol/collector_exact_receipt.json"] = alias_raw
    alias_result["evidence"]["protocol_chain"][
        "collector_exact_receipt_sha256"
    ] = sha256_bytes(alias_raw)
    with pytest.raises(PredictionBlocked) as aliased_source:
        on_demand._validate_sealed_protocol(contents, alias_result)
    assert aliased_source.value.code == "PREDICTION_BUNDLE_INVALID"
    assert aliased_source.value.details == {"reason": "sealed_protocol_required"}

    paths["form"].write_bytes(b"tampered after verified snapshot")
    public_handoff = {
        key: item for key, item in value.items() if not key.startswith("_")
    }
    with pytest.raises(ProtocolRejected):
        protocol.snapshot_collector_exact_handoff(public_handoff)


def test_scheduled_source_report_rejects_mismatched_source_identity(
    tmp_path: Path,
):
    _, value, _, source_attempt = scheduled_exact_receipt(tmp_path)
    report = json.loads(value["_report_bytes"])
    report["source_race_id"] = "Race 5 - OTHER - 2026-07-19"

    with pytest.raises(ProtocolRejected) as mismatch:
        collector_protocol._validate_collector_source_report(
            canonical_bytes(report),
            race_id=RACE_ID,
            collector_run_id=report["collector_run_id"],
            emitted_at=NOW.isoformat(),
            capture_attempt_sha256=sha256_bytes(canonical_bytes(source_attempt)),
            append_report_sha256=sha256_bytes(
                canonical_bytes(source_attempt["append_report"])
            ),
        )
    assert mismatch.value.code == "EXACT_RECEIPT_MALFORMED"


def test_scheduled_snapshot_rejects_fifo_without_blocking(tmp_path: Path):
    protocol, value, paths, _ = scheduled_exact_receipt(tmp_path)
    paths["form"].unlink()
    os.mkfifo(paths["form"])
    public_handoff = {
        key: item for key, item in value.items() if not key.startswith("_")
    }

    with pytest.raises(ProtocolRejected) as fifo:
        protocol.snapshot_collector_exact_handoff(public_handoff)
    assert fifo.value.code == "PROTOCOL_PATH_UNSAFE"


def test_scheduled_snapshot_rejects_hardlinked_source_members(tmp_path: Path):
    protocol, value, paths, _ = scheduled_exact_receipt(tmp_path)
    paths["sidecar"].unlink()
    os.link(paths["form"], paths["sidecar"])
    public_handoff = {
        key: item for key, item in value.items() if not key.startswith("_")
    }
    public_handoff["source_sidecar_sha256"] = public_handoff[
        "source_form_sha256"
    ]
    receipt_path = protocol.collector_exact_receipt_path(
        RACE_ID, value["capture_attempt_sha256"]
    )
    receipt = json.loads(receipt_path.read_bytes())
    receipt["sealed_handoff"] = public_handoff
    receipt["artifacts"]["sidecar"]["sha256"] = public_handoff[
        "source_sidecar_sha256"
    ]
    receipt_path.write_bytes(canonical_bytes(receipt))

    with pytest.raises(ProtocolRejected) as hardlink:
        protocol.snapshot_collector_exact_handoff(public_handoff)
    assert hardlink.value.code == "EXACT_RECEIPT_MALFORMED"


def test_scheduled_snapshot_rejects_unbounded_artifact_path(tmp_path: Path):
    protocol, value, _, _ = scheduled_exact_receipt(tmp_path)
    receipt_path = protocol.collector_exact_receipt_path(
        RACE_ID, value["capture_attempt_sha256"]
    )
    receipt = json.loads(receipt_path.read_bytes())
    receipt["artifacts"]["form"]["path"] = "/".join(
        ["nested"] * 33 + ["gunnedah-r5.csv"]
    )
    receipt_path.write_bytes(canonical_bytes(receipt))
    public_handoff = {
        key: item for key, item in value.items() if not key.startswith("_")
    }

    with pytest.raises(ProtocolRejected) as unbounded:
        protocol.snapshot_collector_exact_handoff(public_handoff)
    assert unbounded.value.code == "PROTOCOL_PATH_UNSAFE"


def test_operator_cli_canonicalizes_source_url_query_and_emits_one_v2_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    source_db = tmp_path / "source.db"
    create_db(source_db)
    deps = dependencies()
    target = race()
    target["url"] += "?trial=false"
    deps.schedule = lambda *values: [target]
    monkeypatch.setattr(
        predict_now, "default_dependencies", lambda parsed: deps
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
            "--capture-evidence-root",
            str(tmp_path / "evidence"),
            "--collector-request-root",
            str(tmp_path / "collector-requests"),
            "--current-time",
            NOW.isoformat(),
        ]
    )
    stdout = capsys.readouterr().out.encode()
    result = json.loads(stdout)
    assert exit_code == 0
    assert stdout == canonical_bytes(result)
    assert result["schema_version"] == "on_demand_race_prediction_v2"
    assert result["status"] == "PREDICTION_READY"
    assert result["race"]["url"] == target["url"].split("?", 1)[0]
    assert result["research_only"] is True
    assert result["production_persisted"] is False


def test_unexpected_dependency_error_still_prints_one_canonical_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    deps = dependencies()
    deps.schedule = lambda *values: (_ for _ in ()).throw(
        OSError("schedule unavailable")
    )
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


def test_predictor_has_no_browser_capture_or_lock_implementation():
    source = Path("scripts/predict_race_now.py").read_text(encoding="utf-8")

    assert not hasattr(predict_now, "_default_fetch")
    assert not hasattr(predict_now, "_default_refresh")
    assert "UpcomingRaceBrowser" not in source
    assert "execute_capture_plan" not in source
    assert "os.open(" not in source


def test_synchronous_capture_does_not_use_predictor_lock_hook(tmp_path: Path):
    calls = {"acquire": 0}
    deps = dependencies()
    deps.acquire_lock = lambda: calls.__setitem__("acquire", calls["acquire"] + 1)

    result = run_prediction(args(tmp_path), deps)

    assert result["schema_version"] == "on_demand_race_prediction_v2"
    assert result["status"] == "PREDICTION_READY"
    assert calls["acquire"] == 0
    assert len(list((tmp_path / "collector-requests/requests").glob("*.json"))) == 1


def test_request_response_receipt_continues_existing_scoring_once(
    tmp_path: Path,
):
    calls = {"capture": 0, "score": 0}
    deps = dependencies()
    original_capture = deps.capture_one
    original_score = deps.score_residual

    def capture(**kwargs: Any) -> Mapping[str, Any]:
        calls["capture"] += 1
        assert original_capture is not None
        return original_capture(**kwargs)

    def score(**kwargs: Any) -> Mapping[str, Any]:
        calls["score"] += 1
        return original_score(**kwargs)

    deps.capture_one = capture
    deps.score_residual = score
    result = run_prediction(args(tmp_path), deps)

    assert result["schema_version"] == "on_demand_race_prediction_v2"
    assert result["status"] == "PREDICTION_READY"
    assert calls == {"capture": 1, "score": 1}
    consumes = list((tmp_path / "collector-requests/consumed").glob("*.json"))
    assert len(consumes) == 1


def test_slow_discovery_fails_before_capture_or_bundle(tmp_path: Path):
    clock = {"value": 0.0}
    calls = {"capture": 0}
    deps = dependencies(monotonic=lambda: clock["value"])

    def slow_schedule(*values: Any) -> list[dict[str, Any]]:
        del values
        clock["value"] = 13.0
        return [race()]

    deps.schedule = slow_schedule
    deps.capture_one = lambda **kwargs: calls.__setitem__(
        "capture", calls["capture"] + 1
    )
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == "DISCOVERY_TIMEOUT"
    assert calls["capture"] == 0
    assert not (tmp_path / "bundles").exists()


def test_parser_rejects_caller_controlled_current_race_index():
    with pytest.raises(SystemExit) as captured:
        predict_now.build_parser().parse_args(
            ["--race-id", "race", "--current-race-index", "/tmp/attacker.json"]
        )
    assert captured.value.code == 2


def test_prediction_discovery_uses_only_fixed_current_index_locator(tmp_path: Path):
    observed: list[Path] = []
    deps = dependencies()

    def schedule(*values: Any) -> list[dict[str, Any]]:
        observed.append(values[2])
        return [race()]

    deps.schedule = schedule
    run_prediction(args(tmp_path, capture_evidence_root=[tmp_path / "caller-root"]), deps)
    assert observed == [
        predict_now.DEFAULT_CAPTURE_EVIDENCE_ROOTS[0]
        / "shadow_autopilot_daemon_runtime"
        / "manual_prediction_current_race_index.json"
    ]


def test_prediction_discovery_uses_worker_bound_evidence_root_index_locator(tmp_path: Path):
    observed: list[Path] = []
    deps = dependencies()
    bound_root = tmp_path / "bound-evidence"

    def schedule(*values: Any) -> list[dict[str, Any]]:
        observed.append(values[2])
        return [race()]

    deps.schedule = schedule
    run_prediction(
        args(
            tmp_path,
            job_id="job_" + "a" * 32,
            capture_evidence_root=[bound_root],
        ),
        deps,
    )
    assert observed == [
        bound_root
        / "shadow_autopilot_daemon_runtime"
        / "manual_prediction_current_race_index.json"
    ]


def test_invalid_worker_job_id_fails_before_discovery_or_bundle_creation(tmp_path: Path):
    calls = {"schedule": 0}
    deps = dependencies()
    deps.schedule = lambda *values: calls.__setitem__("schedule", calls["schedule"] + 1)

    with pytest.raises(PredictionBlocked,match="PREDICTION_BUNDLE_INVALID"):
        run_prediction(args(tmp_path,job_id="not-a-worker-job"),deps)

    assert calls["schedule"] == 0
    assert not (tmp_path / "bundles").exists()


def test_request_race_failure_preserves_original_pre_bundle_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = race()
    deps = dependencies()
    deps.schedule = lambda *values: [target]
    monkeypatch.setattr(
        predict_now,
        "_request_race",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE")
        ),
    )

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)

    assert captured.value.code == "EXACT_RACE_IDENTITY_UNAVAILABLE"
    assert "bundle_persistence_error" not in captured.value.details
    assert not (tmp_path / "bundles").exists()


def test_empty_runner_set_preserves_original_pre_bundle_blocker(tmp_path: Path):
    target = race()
    target["participants"] = []
    deps = dependencies()
    deps.schedule = lambda *values: [target]

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)

    assert captured.value.code == "RUNNER_SET_AMBIGUOUS"
    assert "bundle_persistence_error" not in captured.value.details
    assert not (tmp_path / "bundles").exists()


def test_cancelled_collector_child_leaves_no_live_request(tmp_path: Path):
    deps = dependencies()

    def cancelled(**kwargs: Any) -> Mapping[str, Any]:
        del kwargs
        raise synchronous_capture.CaptureOneRejected("CANCELLED")

    deps.capture_one = cancelled
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)

    assert captured.value.code == "CANCELLED"
    protocol = ManualPredictionCollectorProtocol(tmp_path / "collector-requests")
    assert not protocol.outstanding_request_ids()


def test_insufficient_margin_rejects_before_capture_process(tmp_path: Path):
    calls = {"capture": 0}
    deps = dependencies()
    deps.schedule = lambda *values: [race("12:01")]
    deps.capture_one = lambda **kwargs: calls.__setitem__(
        "capture", calls["capture"] + 1
    )
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == "INSUFFICIENT_PREJUMP_MARGIN"
    assert calls["capture"] == 0
    assert not (tmp_path / "collector-requests/requests").exists()


def test_capture_source_uses_collector_child_not_predictor_hooks(tmp_path: Path):
    calls = {"acquire": 0, "fetch": 0}
    deps = dependencies()
    deps.acquire_lock = lambda: calls.__setitem__("acquire", calls["acquire"] + 1)
    deps.fetch_odds = lambda *args, **kwargs: calls.__setitem__(
        "fetch", calls["fetch"] + 1
    )

    result = run_prediction(args(tmp_path, odds_source="capture"), deps)

    assert result["status"] == "PREDICTION_READY"
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

    result = run_prediction(
        args(tmp_path, odds_source="capture"),
        dependencies(),
    )
    assert result["status"] == "PREDICTION_READY"

    after = lock_path.stat(follow_symlinks=False)
    assert lock_path.read_bytes() == original
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )


def test_default_dependencies_delegate_exact_capture_to_collector_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    commands: list[tuple[list[str], float]] = []
    monkeypatch.setattr(
        synchronous_capture,
        "invoke_capture_one",
        lambda *, command, timeout_seconds: (
            commands.append((list(command), timeout_seconds))
            or {"status": "RECEIPT_READY"}
        ),
    )
    namespace = argparse.Namespace(
        db=tmp_path / "canonical/greyhound_racing_data.db",
        lock_path=tmp_path / "different/shadow_autopilot.lock",
        fetch_timeout_seconds=45,
    )
    dependency_set = predict_now.default_dependencies(namespace)

    assert not hasattr(dependency_set, "acquire_lock")
    assert not hasattr(dependency_set, "release_lock")
    assert not hasattr(dependency_set, "fetch_odds")
    assert not hasattr(dependency_set, "refresh")
    assert dependency_set.capture_one is not None
    result = dependency_set.capture_one(
        protocol_root=tmp_path / "protocol",
        evidence_root=tmp_path / "evidence",
        request_id="request-id",
        output_dir=tmp_path / "evidence/capture",
        minimum_margin_seconds=114,
        minimum_post_lock_margin_seconds=113,
        minimum_fetch_margin_seconds=98,
        timeout_seconds=84,
    )
    assert result["status"] == "RECEIPT_READY"
    command, timeout = commands[0]
    assert command[2] == "capture-one"
    assert "--minimum-post-lock-margin-seconds" in command
    assert "--minimum-fetch-margin-seconds" in command
    assert timeout == 84


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

    monkeypatch.setattr(synchronous_capture.os, "fstat", fail_fstat)
    with pytest.raises(PredictionBlocked) as captured:
        _acquire_collector_lock_no_steal(
            lock_path, run_id="on_demand_test", output_dir=tmp_path / "bundle"
        )
    assert captured.value.code == "LOCK_ACQUIRE_FAILED"
    assert captured.value.details["reason"] == "descriptor_stat_failed"
    assert not lock_path.exists()


def test_exact_valid_receipt_is_reused_without_second_capture(tmp_path: Path):
    calls = {"capture": 0, "score": 0}
    deps = dependencies()
    original_capture = deps.capture_one
    original_score = deps.score_residual

    def capture(**kwargs: Any) -> Mapping[str, Any]:
        calls["capture"] += 1
        assert original_capture is not None
        return original_capture(**kwargs)

    def score(**kwargs: Any) -> Mapping[str, Any]:
        calls["score"] += 1
        return original_score(**kwargs)

    deps.capture_one = capture
    deps.score_residual = score
    run_prediction(args(tmp_path), deps)
    run_prediction(args(tmp_path), deps)

    assert calls == {"capture": 1, "score": 2}
    assert len(list((tmp_path / "collector-requests/requests").glob("*.json"))) == 1


def test_receipt_only_mode_does_not_scan_legacy_evidence(tmp_path: Path):
    deps = dependencies()

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path, odds_source="receipt"), deps)

    assert captured.value.code == "RECEIPT_UNAVAILABLE"


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
        (
            tmp_path
            / "bundles"
            / json.loads(
                (tmp_path / "bundles/prediction_bundle_index_v1.json").read_bytes()
            )["entries"][0]["directory"]
            / "result.json"
        ).read_bytes()
    )
    assert result["status"] == "PREDICTION_BLOCKED"
    assert result["blocker_stage"] == "SCORING"
    assert result["blocker"] == {"code": code}


@pytest.mark.parametrize(
    "publication_code",
    [
        "PREDICTION_BUNDLE_INDEX_LOCK_UNAVAILABLE",
        "PREDICTION_BUNDLE_INDEX_LOCK_INVALID",
        "PREDICTION_BUNDLE_INDEX_LOCK_REPLACED",
        "PREDICTION_BUNDLE_INDEX_LOCK_RELEASE_FAILED",
        "PREDICTION_BUNDLE_INDEX_WRITE_FAILED",
        "PREDICTION_BUNDLE_REPLACED",
    ],
)
def test_post_creation_publication_failure_seals_once_without_retry_or_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publication_code: str
):
    calls = 0

    def fail_publication(*values: Any, **kwargs: Any) -> None:
        nonlocal calls
        del values, kwargs
        calls += 1
        raise PredictionBlocked(publication_code)

    monkeypatch.setattr(predict_now, "publish_prediction_bundle_index_entry", fail_publication)
    deps = dependencies()
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == publication_code
    assert calls == 1
    bundles = list((tmp_path / "bundles").glob("prediction_*"))
    assert len(bundles) == 1
    assert (bundles[0] / "result.json").is_file()
    assert (bundles[0] / "bundle_manifest.json").is_file()
    assert not (tmp_path / "bundles/prediction_bundle_index_v1.json").exists()
    assert len(list((tmp_path / "collector-requests/requests").glob("*.json"))) == 1


def test_post_jump_blocks_before_bundle_or_lock(tmp_path: Path):
    deps = dependencies()
    deps.schedule = lambda *values: [race("11:59")]
    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == "POST_JUMP"
    assert not (tmp_path / "bundles").exists()


def test_exact_receipt_is_rejected_after_jump(tmp_path: Path):
    deps = dependencies()
    run_prediction(args(tmp_path), deps)
    deps.now = lambda: NOW + timedelta(hours=1)

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(args(tmp_path), deps)
    assert captured.value.code == "POST_JUMP"


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


def test_history_seal_never_opens_canonical_database_through_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    create_db(source)
    target = tmp_path / "sealed.db"
    original_connect = on_demand.sqlite3.connect
    opened: list[str] = []

    def reject_canonical_open(database: Any, *values: Any, **kwargs: Any):
        location = str(database)
        opened.append(location)
        if location.startswith(f"file:{source.resolve()}"):
            raise sqlite3.OperationalError("read-only WAL sidecars unavailable")
        return original_connect(database, *values, **kwargs)

    monkeypatch.setattr(on_demand.sqlite3, "connect", reject_canonical_open)
    audit = seal_history_database(
        source=source,
        target=target,
        target_race_id=RACE_ID,
        cutoff=NOW + timedelta(hours=1),
        runner_names=["Alpha", "Beta"],
    )

    assert target.is_file()
    assert audit["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert any("immutable=1" in location for location in opened)
    assert not list(tmp_path.glob(".history-source-*.db"))


def test_history_seal_fails_closed_while_wal_has_uncheckpointed_bytes(
    tmp_path: Path,
):
    source = tmp_path / "source.db"
    create_db(source)
    Path(f"{source}-wal").write_bytes(b"uncheckpointed")
    target = tmp_path / "sealed.db"

    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=target,
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_DATABASE_BUSY"
    assert not target.exists()
    assert not list(tmp_path.glob(".history-source-*.db"))


def test_history_seal_rejects_atomic_source_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    replacement = tmp_path / "replacement.db"
    create_db(source)
    create_db(replacement)
    target = tmp_path / "sealed.db"
    original_check = on_demand._history_sidecars_clear
    calls = 0

    def replace_before_named_revalidation(path: Path) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            os.replace(replacement, source)
        return original_check(path)

    monkeypatch.setattr(
        on_demand, "_history_sidecars_clear", replace_before_named_revalidation
    )
    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=target,
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_DATABASE_CHANGED"
    assert not target.exists()
    assert not list(tmp_path.glob(".history-source-*.db"))


def test_history_seal_normalizes_copy_failure_and_removes_temporary_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    create_db(source)
    target = tmp_path / "sealed.db"

    def fail_read(*values: Any, **kwargs: Any) -> bytes:
        del values, kwargs
        raise OSError("simulated source read failure")

    monkeypatch.setattr(on_demand.os, "read", fail_read)
    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=target,
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_DATABASE_UNAVAILABLE"
    assert not target.exists()
    assert not list(tmp_path.glob(".history-source-*.db"))


def test_history_seal_normalizes_snapshot_sqlite_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    create_db(source)
    target = tmp_path / "sealed.db"
    original_connect = on_demand.sqlite3.connect

    def fail_snapshot_open(database: Any, *values: Any, **kwargs: Any):
        if "immutable=1" in str(database):
            raise sqlite3.OperationalError("simulated snapshot open failure")
        return original_connect(database, *values, **kwargs)

    monkeypatch.setattr(on_demand.sqlite3, "connect", fail_snapshot_open)
    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=target,
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_DATABASE_INTEGRITY_FAILED"
    assert not target.exists()
    assert not list(tmp_path.glob(".history-source-*.db"))


def test_history_seal_normalizes_workspace_creation_failure(tmp_path: Path):
    source = tmp_path / "source.db"
    create_db(source)
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")

    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=blocked_parent / "sealed.db",
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_SEAL_WRITE_FAILED"


def test_history_seal_removes_partial_target_after_target_sqlite_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    create_db(source)
    target = tmp_path / "sealed.db"
    original_connect = on_demand.sqlite3.connect

    class FailingTarget:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, sql: str, parameters: Any = ()):
            if sql.startswith("CREATE TABLE"):
                raise sqlite3.OperationalError("simulated target write failure")
            return self.connection.execute(sql, parameters)

        def close(self) -> None:
            self.connection.close()

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

    def fail_target(database: Any, *values: Any, **kwargs: Any):
        connection = original_connect(database, *values, **kwargs)
        return FailingTarget(connection) if Path(str(database)) == target else connection

    monkeypatch.setattr(on_demand.sqlite3, "connect", fail_target)
    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=target,
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_SEAL_WRITE_FAILED"
    assert not target.exists()
    assert not list(tmp_path.glob(".history-source-*.db"))


def test_history_seal_cleanup_failure_does_not_mask_primary_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    create_db(source)
    target = tmp_path / "sealed.db"
    original_unlink = Path.unlink

    def fail_read(*values: Any, **kwargs: Any) -> bytes:
        del values, kwargs
        raise OSError("simulated source read failure")

    def fail_snapshot_unlink(path: Path, *values: Any, **kwargs: Any) -> None:
        if path.name.startswith(".history-source-"):
            raise OSError("simulated cleanup failure")
        original_unlink(path, *values, **kwargs)

    monkeypatch.setattr(on_demand.os, "read", fail_read)
    monkeypatch.setattr(Path, "unlink", fail_snapshot_unlink)
    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=target,
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_DATABASE_UNAVAILABLE"
    assert captured.value.details["snapshot_cleanup_error"] == "OSError"


def test_history_seal_descriptor_close_failure_fails_closed_and_cleans_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    create_db(source)
    target = tmp_path / "sealed.db"
    original_close = on_demand.os.close
    calls = 0

    def fail_first_close(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated snapshot descriptor close failure")
        original_close(descriptor)

    monkeypatch.setattr(on_demand.os, "close", fail_first_close)
    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=target,
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_SEAL_WRITE_FAILED"
    assert captured.value.details["snapshot_descriptor_close_error"] == "OSError"
    assert not target.exists()
    assert not list(tmp_path.glob(".history-source-*.db"))


def test_history_seal_target_connection_close_failure_removes_complete_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.db"
    create_db(source)
    target = tmp_path / "sealed.db"
    original_connect = on_demand.sqlite3.connect

    class CloseFailTarget:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def close(self) -> None:
            self.connection.close()
            raise sqlite3.OperationalError("simulated target close failure")

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

    def fail_target_close(database: Any, *values: Any, **kwargs: Any):
        connection = original_connect(database, *values, **kwargs)
        return CloseFailTarget(connection) if Path(str(database)) == target else connection

    monkeypatch.setattr(on_demand.sqlite3, "connect", fail_target_close)
    with pytest.raises(PredictionBlocked) as captured:
        seal_history_database(
            source=source,
            target=target,
            target_race_id=RACE_ID,
            cutoff=NOW + timedelta(hours=1),
            runner_names=["Alpha", "Beta"],
        )

    assert captured.value.code == "HISTORY_SEAL_WRITE_FAILED"
    assert captured.value.details["target_connection_close_error"] == "OperationalError"
    assert not target.exists()
    assert not list(tmp_path.glob(".history-source-*.db"))


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
        '{"bundle":{"latency_budget":{"capture_seconds":45,'
        f'"discovery_seconds":{constant},'
        '"lock_seconds":1,"safety_seconds":15,"scoring_seconds":30,'
        '"validation_seconds":8},"receipt_max_age_seconds":900},'
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
        "seal_features": 0,
        "score_residual": 0,
        "capture_one": 0,
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
def test_invalid_discovery_budget_values_are_rejected(
    tmp_path: Path, value: Any, code: str
):
    calls = {"schedule": 0}
    deps = dependencies()
    deps.schedule = lambda *values: calls.__setitem__(
        "schedule", calls["schedule"] + 1
    )

    with pytest.raises(PredictionBlocked) as captured:
        run_prediction(
            args(tmp_path, config=config_with_discovery_budget(tmp_path, value)),
            deps,
        )

    assert captured.value.code == code
    assert calls["schedule"] == 0


def test_checked_in_latency_budget_is_declared_and_bounded():
    for name in ("manual-default.json", "market-only.json"):
        config = json.loads((Path("configs/prediction") / name).read_bytes())
        assert config["bundle"]["latency_budget"] == {
            "capture_seconds": 60,
            "discovery_seconds": 12,
            "lock_seconds": 1,
            "safety_seconds": 15,
            "scoring_seconds": 30,
            "validation_seconds": 8,
        }
        assert config["bundle"]["current_index_max_age_seconds"] == 1200

    for name in ("market_form_residual_v1.schema.json", "market_only_v1.schema.json"):
        schema = json.loads((Path("configs/prediction/schemas") / name).read_bytes())
        budget_schema = schema["properties"]["bundle"]["properties"][
            "latency_budget"
        ]
        assert budget_schema["additionalProperties"] is False
        assert set(budget_schema["required"]) == {
            "discovery_seconds",
            "lock_seconds",
            "capture_seconds",
            "validation_seconds",
            "scoring_seconds",
            "safety_seconds",
        }


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
    bundle = tmp_path / "legacy-v1"
    bundle.mkdir()
    receipt, *_ = receipt_from_handoff(
        handoff(NOW - timedelta(minutes=1)),
        current_time=NOW,
        max_age_seconds=900,
    )
    result = {
        "schema_version": "on_demand_race_prediction_v1",
        "prediction": on_demand.market_only_prediction(receipt),
    }
    (bundle / "result.json").write_bytes(canonical_bytes(result))
    (bundle / "odds_receipt.json").write_bytes(canonical_bytes(receipt))
    (bundle / "bundle_manifest.json").write_bytes(
        canonical_bytes(on_demand.bundle_manifest(bundle))
    )
    assert replay_bundle(bundle) == result
    receipt_path = bundle / "odds_receipt.json"
    receipt_path.write_bytes(receipt_path.read_bytes().replace(b"2.5", b"2.6", 1))
    with pytest.raises(PredictionBlocked) as captured:
        replay_bundle(bundle)
    assert captured.value.code == "REPLAY_TAMPERED"


def test_residual_bundle_replay_reruns_scorer_at_original_timestamp(tmp_path: Path):
    result = run_prediction(args(tmp_path), dependencies())
    index = on_demand.verify_prediction_bundle_index(tmp_path / "bundles")
    verified = on_demand.verify_indexed_prediction_bundle(
        tmp_path / "bundles", index["entries"][0]
    )
    assert verified.result == result

    bundle = tmp_path / "bundles" / index["entries"][0]["directory"]
    contents = {
        path.relative_to(bundle).as_posix(): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    contents["protocol/collector_exact_receipt.json"] = canonical_bytes({})
    with pytest.raises(PredictionBlocked) as contradictory_protocol:
        on_demand._validate_sealed_protocol(contents, result)
    assert contradictory_protocol.value.code == "PREDICTION_BUNDLE_INVALID"
    assert contradictory_protocol.value.details == {
        "reason": "sealed_protocol_required"
    }


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
