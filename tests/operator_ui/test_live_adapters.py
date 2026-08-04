from __future__ import annotations

import inspect
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from flask import Flask
from types import SimpleNamespace
from werkzeug.security import generate_password_hash

import src.operator_ui.live_adapters as live_module
from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
from race_collection.synchronous_manual_capture import CaptureOneRejected
from src.predictor.on_demand import (
    PREDICTION_BUNDLE_INDEX_NAME,
    VerifiedPredictionBundleIndex,
    canonical_bytes as prediction_canonical_bytes,
)

from scripts.shadow_autopilot_daemon import (
    completed_daemon_run_report_envelope,
    initial_daemon_run_report,
    lock_held_daemon_run_report,
    odds_capture_service_file_text,
    odds_capture_timer_file_text,
    service_file_text,
    timer_file_text,
)
from src.operator_ui.foundation import (
    JsonSerializationPolicy,
    JsonSource,
    OperatorEvidenceReader,
    RawSourceConfig,
    SourceConfig,
    TimestampSyntax,
)
from src.operator_ui.live_adapters import (
    InstalledUnits, LiveEvidenceAdapters, PredictionBundleSource,
    UpcomingRaceSource, _calendar_gap, _finite_metric,
)
from src.operator_ui.api import install_level_1_api
from src.operator_ui.bootstrap import CONFIG_KEY, bind_configured_live_evidence
from src.operator_ui.security import install_connected_mode

NOW = datetime(2026, 7, 31, 2, tzinfo=timezone.utc)


def canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def actual_payloads(at=NOW - timedelta(seconds=30)):
    stamp = at.isoformat().replace("+00:00", "Z")
    full_report = completed_daemon_run_report_envelope(
        run_id="full-1", generated_at=at, current_time=stamp,
        output_dir=Path("artifacts/full-1"), final_verdict="DAEMON_READY",
    )
    full_state = {
        "schema_version": "shadow_autopilot_daemon_state_v1",
        "last_run_id": "full-1", "last_output_dir": "artifacts/full-1",
        "last_verdict": "DAEMON_READY", "updated_at": stamp,
    }
    # Exact refresh shape written by shadow_autopilot_v1: it has producer time
    # and status but intentionally has no invented schema_version or outer run_id.
    refresh = {"generated_at": stamp, "status": "SUCCESS", "selected_count": 0, "selected_races": []}
    odds_report = {
        "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "run_id": "odds-9", "generated_at": stamp,
        "final_status": "ODDS_CAPTURE_ONLY_READY", "status": "READY", "output_dir": "artifacts/odds-9",
        "autopilot_output_dir": "artifacts/autopilot-9",
        "odds_capture_refresh_report": refresh,
    }
    odds_state = {
        "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
        "run_id": "odds-9", "updated_at": stamp,
        "final_status": "ODDS_CAPTURE_ONLY_READY", "output_dir": "artifacts/odds-9",
        "autopilot_output_dir": "artifacts/autopilot-9",
        "odds_capture_refresh_status": "SUCCESS",
    }
    corpus = {
        "schema_version": "race_evidence_inventory_report_v1", "generated_at": stamp,
        "final_status": "RACE_EVIDENCE_INVENTORY_READY_FOR_EVALUATION",
        "recommended_decision": "RUN_POST_BACKLOG_UNIFIED_EVALUATION",
        "output_dir": "packet", "artifact_roots": ["artifacts/evidence"],
        "db_path": "greyhound_racing_data.db",
        "inventory_csv": "packet/race_evidence_inventory.csv",
        "inventory_jsonl": "packet/race_evidence_inventory.jsonl",
        "scorecard_csv": "packet/race_evidence_scorecard.csv",
        "scorecard_jsonl": "packet/race_evidence_scorecard.jsonl",
        "official_artifact_summary": {"input_artifact_root_count": 1, "artifact_discovery": [{"input_artifact_dir": "artifacts/evidence", "mode": "missing_artifact_dir", "direct_match": False, "discovered_child_artifact_count": 0, "discovered_child_artifact_dirs": [], "discovered_child_artifact_dirs_truncated": False}], "official_result_artifact_dir_count": 0, "official_result_artifact_race_rows": 0, "official_result_artifact_runner_rows": 0, "official_result_artifact_race_count": 0},
        "shadow_prediction_summary": {"prediction_file_count": 1, "prediction_file_limit": None, "prediction_file_scan_truncated": False, "prediction_file_kind_counts": {"stage2_shadow_predictions.jsonl": 1}, "shadow_prediction_rows": 2, "shadow_prediction_race_count": 2},
        "db_summary": {"db_status": {"status": "AVAILABLE", "db_path": "greyhound_racing_data.db", "bytes": 1}, "table_status": {"autonomous_official_result_evidence_races": {"present": True}, "autonomous_official_result_evidence_runners": {"present": True}, "live_odds": {"present": True}}, "counts": {"official_result_evidence_race_rows": 1, "official_result_evidence_race_count": 1, "official_result_evidence_runner_rows": 1, "official_result_evidence_runner_race_count": 1, "live_odds_rows": 1, "live_odds_race_count": 1, "strict_live_odds_rows": 1, "strict_live_odds_race_count": 1}},
        "latest_backlog_append_report": {"status": "DATA_MISSING", "reason": "no_backlog_append_report_found"},
        "summary_counts": {
            "race_union_count": 2, "shadow_prediction_race_count": 2,
            "official_result_artifact_race_count": 0, "official_result_evidence_db_race_count": 1,
            "live_odds_race_count": 1, "strict_prejump_odds_race_count": 1,
            "shadow_races_with_official_result_evidence_db": 1,
            "shadow_races_with_strict_prejump_odds": 1,
            "shadow_races_complete_official_and_strict_odds": 1,
            "action_counts": {"ready_for_unified_evidence_evaluation": 1, "capture_official_result": 1},
        },
        "scorecard_metrics": {"schema_version": "race_evidence_scorecard_metrics_v1", "evaluation_race_count": 1, "model_top1_accuracy": 1.0, "model_top3_accuracy": 1.0, "model_mean_winner_rank": 1.0, "model_logloss": 0.5, "market_top1_accuracy": 1.0, "market_top3_accuracy": 1.0, "market_mean_winner_rank": 1.0, "market_logloss": 0.5, "skipped_race_reason_counts": {"official_result_incomplete_for_shadow_boxes": 1}, "skipped_race_action_counts": {"capture_official_result": 1}, "official_result_gap_action_counts": {"capture_official_result": 1}, "strict_odds_gap_action_counts": {}, "metric_notes": ["report_only_latest_shadow_prediction_per_race_box", "official_results_from_append_only_evidence_db", "market_baseline_from_latest_strict_sportsbet_odds_per_box", "scorecard_gap_action_counts_use_recommended_next_action"]},
        "top_gap_races": [],
        "no_write_guarantees": {name: False for name in ("training", "production_promotion", "registry_mutation", "production_pointer_update", "active_model_replacement", "db_write", "label_write", "odds_write", "official_result_write", "daemon_control", "betting_or_ev_action", "snapshot_rewrite", "manifest_rewrite")},
    }
    repo = Path(__file__).resolve().parents[2]
    config_specs = (
        ("market-form-residual-v1", "latest-research", "manual-default.json", "market_form_residual_v1"),
        ("market-only", "market-only", "market-only.json", "market_only_v1"),
    )
    catalog_configs = []
    for name, selector, filename, resolved in config_specs:
        config_path = repo / "configs/prediction" / filename
        schema_path = repo / "configs/prediction/schemas" / f"{resolved}.schema.json"
        artifact = repo / "artifacts/frozen_models/market_form_residual_v1/model.json" if resolved != "market_only_v1" else None
        model_manifest = repo / "artifacts/frozen_models/market_form_residual_v1/manifest.json" if artifact else None
        catalog_configs.append({
            "name": name, "selector": selector, "config": f"configs/prediction/{filename}",
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "resolved_config": json.loads(config_path.read_bytes()),
            "model": {"requested": selector, "resolved": resolved, "alias_resolved": True,
                "model_sha256": None if artifact is None else hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "manifest_sha256": None if model_manifest is None else hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
                "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest()},
        })
    return {
        "full_report": full_report, "full_state": full_state, "odds_report": odds_report,
        "odds_state": odds_state, "odds_refresh": refresh, "corpus_report": corpus,
        "corpus_manifest": {"schema_version": "race_evidence_inventory_output_manifest_v1", "output_dir": "packet", "files": {}},
        "deployment_manifest": {"schema_version": "operator_ui_deployment_manifest_v1", "generated_at": stamp, "source_commit": "b" * 40, "source_tree": "c" * 40, "deployed_commit": "b" * 40, "deployed_tree": "c" * 40},
        "model_catalog": {"schema_version": "on_demand_prediction_config_catalog_v1", "status": "CONFIGS_AVAILABLE", "configs": catalog_configs},
    }


def make_live(
    root: Path,
    values=None,
    *,
    now=NOW,
    full_timer=None,
    odds_timer=None,
    full_status=("inactive", "dead", 0),
    odds_status=("active", "waiting", 0),
    units_observed_at=None,
    serialization_policies=None,
    raw_overrides=None,
    upcoming_races=None,
    prediction_bundles=None,
):
    values = values or actual_payloads()
    unit_bytes = {
        "full_timer": (full_timer or timer_file_text()).encode(),
        "full_service": service_file_text(repo_path=Path("/srv/app"), timeout_seconds=840).encode(),
        "odds_timer": (odds_timer or odds_capture_timer_file_text()).encode(),
        "odds_service": odds_capture_service_file_text(repo_path=Path("/srv/app"), timeout_seconds=600).encode(),
    }
    if "deployment_manifest" in values:
        values["deployment_manifest"]["working_directory"] = "/srv/app"
        values["deployment_manifest"]["installed_unit_sha256"] = {
            name: hashlib.sha256(raw).hexdigest() for name, raw in unit_bytes.items()
        }
    raw_sources = {}
    packet_files = {
        "race_evidence_inventory.csv": b"race_id\n1\n",
        "race_evidence_inventory.jsonl": b'{"race_id":"1"}\n',
        "race_evidence_scorecard.csv": b"race_id\n1\n",
        "race_evidence_scorecard.jsonl": b'{"race_id":"1"}\n',
        "SUMMARY.md": b"# Inventory\n",
        "final_status.txt": b"RACE_EVIDENCE_INVENTORY_READY_FOR_EVALUATION\n",
    }
    packet = root / "packet"
    packet.mkdir(parents=True, exist_ok=True)
    report_raw = canonical(values["corpus_report"])
    packet_files["race_evidence_inventory_report.json"] = report_raw
    for filename, raw in packet_files.items():
        target = packet / filename
        target.write_bytes(raw)
        output_locator = values["corpus_manifest"]["output_dir"].rstrip("/")
        values["corpus_manifest"]["files"][f"{output_locator}/{filename}"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    raw_key_files = {
        "corpus_inventory_csv": packet / "race_evidence_inventory.csv",
        "corpus_inventory_jsonl": packet / "race_evidence_inventory.jsonl",
        "corpus_scorecard_csv": packet / "race_evidence_scorecard.csv",
        "corpus_scorecard_jsonl": packet / "race_evidence_scorecard.jsonl",
        "corpus_report_bytes": packet / "race_evidence_inventory_report.json",
        "corpus_summary": packet / "SUMMARY.md", "corpus_final_status": packet / "final_status.txt",
    }
    repo = Path(__file__).resolve().parents[2]
    raw_key_files.update({
        "model_latest_config": repo / "configs/prediction/manual-default.json",
        "model_latest_schema": repo / "configs/prediction/schemas/market_form_residual_v1.schema.json",
        "model_latest_artifact": repo / "artifacts/frozen_models/market_form_residual_v1/model.json",
        "model_latest_manifest": repo / "artifacts/frozen_models/market_form_residual_v1/manifest.json",
        "model_baseline_config": repo / "configs/prediction/market-only.json",
        "model_baseline_schema": repo / "configs/prediction/schemas/market_only_v1.schema.json",
    })
    for key, raw in (raw_overrides or {}).items():
        override = root / "raw-overrides" / f"{key}.json"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_bytes(raw)
        raw_key_files[key] = override
    for key, path in raw_key_files.items():
        allowlisted = root if path.is_relative_to(root) else repo
        raw_sources[key] = RawSourceConfig(
            locator=path, allowlisted_root=allowlisted, source_kind="fixed_file",
            source_identity=key, source_locator=f"operator_ui.{key}", policy="P-CATALOG-60" if key.startswith("model_") else "P-REPORT-24H",
            supported_claim="Exact fixed bytes only.", max_bytes=64*1024*1024 if key in {"corpus_inventory_csv","corpus_inventory_jsonl"} else 16_777_216,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), expected_bytes=path.stat().st_size if key in {"corpus_inventory_csv","corpus_inventory_jsonl"} else None,
            digest_only=key in {"corpus_inventory_csv","corpus_inventory_jsonl"},
        )
    sources = {}
    for key, payload in values.items():
        path = (
            root / "artifacts/autopilot-9/odds_capture_refresh_report.json"
            if key == "odds_refresh"
            else root / f"{key}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        policy = (serialization_policies or {}).get(
            key, JsonSerializationPolicy.COMPACT_CANONICAL
        )
        if policy is JsonSerializationPolicy.PRODUCER_PRETTY_SORTED:
            path.write_bytes(
                (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode()
            )
        else:
            path.write_bytes(canonical(payload))
        time_field = (
            None
            if key in {"full_state", "corpus_manifest", "model_catalog"}
            else ("updated_at" if key.endswith("state") else "generated_at")
        )
        schema_value = payload.get("schema_version")
        sources[key] = SourceConfig(
            locator=path, allowlisted_root=root, source_kind="producer_report",
            source_identity=str(schema_value or "shadow_autopilot_refresh_report"),
            source_locator=f"operator_ui.{key}",
            policy="P-COLLECTOR-AGGREGATE" if key.startswith(("full_", "odds_")) else ("P-REPORT-24H" if key.startswith("corpus_") else ("P-CATALOG-60" if key == "model_catalog" else "P-DEPLOY-60")),
            supported_claim="Exact producer evidence only.",
            json=JsonSource(
                schema_field="schema_version" if schema_value else None,
                schema_value=schema_value,
                top_level_fields=tuple(payload),
                time_field=time_field,
                identity_fields=("schema_version",) if schema_value else (),
                max_items=1000,
                serialization_policy=policy,
                timestamp_syntax=TimestampSyntax.AWARE_ISO8601,
                authority_observed_at=(now.isoformat() if key == "model_catalog" else None),
            ),
        )
    reader = OperatorEvidenceReader(sources, raw_sources=raw_sources, clock=lambda: now)
    units = InstalledUnits(
        **unit_bytes, observed_at=units_observed_at or now, working_directory="/srv/app",
        full_unit_name="shadow-autopilot.service",
        full_active_state=full_status[0], full_sub_state=full_status[1],
        full_exec_main_pid=full_status[2],
        odds_unit_name="shadow-autopilot-odds-capture.service",
        odds_active_state=odds_status[0],
        odds_sub_state=odds_status[1], odds_exec_main_pid=odds_status[2],
        **{
            f"{name}_sha256": hashlib.sha256(raw).hexdigest()
            for name, raw in unit_bytes.items()
        },
    )
    return LiveEvidenceAdapters(
        reader, units=units, upcoming_races=upcoming_races,
        prediction_bundles=prediction_bundles,
    )


def test_actual_generated_units_and_distinct_lane_ids(tmp_path):
    result = make_live(tmp_path).collector(NOW)
    assert result.evidence.status == "AVAILABLE/FRESH"
    assert result.evidence.freshness_policy == "P-COLLECTOR-AGGREGATE"
    assert [lane["run_id"] for lane in result.data["lanes"]] == ["full-1", "odds-9"]
    assert result.data["lanes"][0]["component_identity"]["cadence_seconds"] == "900.0"
    assert result.data["lanes"][1]["component_identity"]["cadence_seconds"] == "120.0"
    assert result.data["lanes"][0]["component_identity"]["timeout_seconds"] == "3360.0"
    assert result.data["lanes"][1]["component_identity"]["timeout_seconds"] == "1200.0"
    assert result.data["lanes"][0]["phase"] == "DAEMON_READY"
    assert result.data["lanes"][1]["phase"] == "ODDS_CAPTURE_ONLY_READY"
    assert {lane["cycle_state"] for lane in result.data["lanes"]} == {"COMPLETED"}
    assert all("next_action" not in lane for lane in result.data["lanes"])
    for lane in result.data["lanes"]:
        assert set(lane["operational_context"]) == {
            "final_status", "final_verdict", "status",
            "next_meaningful_action", "next_meaningful_action_at",
            "lock_owner", "recent_capture",
        }


def test_realistic_producer_payloads_are_bound_at_authenticated_collector_endpoint(tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True, OPERATOR_UI_CONNECTED_MODE=True, OPERATOR_UI_LEVEL=2,
        OPERATOR_UI_SECRET_KEY="endpoint-secret-" + "x" * 40,
        OPERATOR_UI_USERNAME="operator",
        OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"),
        OPERATOR_UI_AUDIT_DB_PATH=str(tmp_path / "audit.sqlite3"),
        DATABASE_PATH=str(tmp_path / "canonical.sqlite3"),
        OPERATOR_UI_DEPLOYED_COMMIT="b" * 40,
        OPERATOR_UI_DEPLOYED_TREE="c" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="operator-ui-v1",
        OPERATOR_UI_CLOCK=lambda: NOW,
    )
    install_connected_mode(app); assert install_level_1_api(app)
    app.config[CONFIG_KEY] = make_live(tmp_path / "live")
    assert bind_configured_live_evidence(app)
    client = app.test_client()
    token = client.get("/operator-ui/login").get_json()["csrf_token"]
    assert client.post("/operator-ui/login", data={"username":"operator","password":"correct horse","csrf_token":token}).status_code == 200
    response = client.get("/operator-ui/api/v1/collector")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["classification"] == "AVAILABLE/FRESH"
    assert [lane["run_id"] for lane in payload["data"]["lanes"]] == ["full-1", "odds-9"]


def test_complete_calendar_and_nondefault_values(tmp_path):
    assert _calendar_gap("*:00,15,30,45") == 900
    live = make_live(tmp_path, full_timer="[Timer]\nOnUnitInactiveSec=2h\nAccuracySec=7s\n", odds_timer="[Timer]\nOnCalendar=*:00,10,40\nAccuracySec=3s\n")
    lanes = live.collector(NOW).data["lanes"]
    assert lanes[0]["component_identity"]["cadence_seconds"] == "7200.0"
    assert lanes[0]["component_identity"]["accuracy_seconds"] == "7.0"
    assert lanes[1]["component_identity"]["cadence_seconds"] == "1800.0"
    assert lanes[1]["component_identity"]["accuracy_seconds"] == "3.0"


@pytest.mark.parametrize("timer", ["[Timer]\nAccuracySec=30s\n", "[Timer]\nOnUnitInactiveSec=0s\nAccuracySec=30s\n"])
def test_missing_nonpositive_conflicting_deadline_inputs_fail_closed(tmp_path, timer):
    lane = make_live(tmp_path, full_timer=timer).collector(NOW).data["lanes"][0]
    assert lane["status"] == "DATA_MISSING"
    assert lane["deadline_utc"] is None
    assert lane["state_age_seconds"] is None


@pytest.mark.parametrize("evidence", ["state", "report"])
def test_missing_lane_evidence_never_invents_freshness(tmp_path, evidence):
    live = make_live(tmp_path)
    (tmp_path / f"full_{evidence}.json").unlink()

    lane = live.collector(NOW).data["lanes"][0]

    assert lane["status"] == "DATA_MISSING"
    assert lane["deadline_utc"] is None
    assert lane["state_age_seconds"] is None


@pytest.mark.parametrize("evidence", ["state", "report"])
def test_malformed_lane_evidence_never_invents_freshness(tmp_path, evidence):
    values = actual_payloads()
    values[f"full_{evidence}"]["schema_version"] = "wrong"

    lane = make_live(tmp_path, values).collector(NOW).data["lanes"][0]

    assert lane["status"] == "INTEGRITY_FAILED"
    assert lane["deadline_utc"] is None
    assert lane["state_age_seconds"] is None


def test_conflicting_deadline_input_diverges(tmp_path):
    timer = "[Timer]\nOnUnitInactiveSec=1min\nOnUnitInactiveSec=2min\nAccuracySec=30s\n"
    lane = make_live(tmp_path, full_timer=timer).collector(NOW).data["lanes"][0]
    assert lane["status"] == "DIVERGENT"


def test_exact_completed_boundary_and_worst_child_propagation(tmp_path):
    values = actual_payloads(NOW - timedelta(seconds=930))
    assert make_live(tmp_path / "at", values).collector(NOW).data["lanes"][0]["status"] == "RECEIPT_READY"
    after = NOW + timedelta(microseconds=1)
    result = make_live(tmp_path / "after", values, now=after).collector(after)
    assert result.evidence.status == "STALE"
    assert result.data["lanes"][0]["status"] == "STALE"


def test_intra_lane_run_mismatch_diverges_but_cross_lane_ids_do_not(tmp_path):
    values = actual_payloads()
    values["odds_state"]["run_id"] = "different"
    result = make_live(tmp_path, values).collector(NOW)
    assert result.evidence.status == "DIVERGENT"
    assert result.data["lanes"][0]["status"] == "RECEIPT_READY"
    assert result.data["lanes"][1]["status"] == "DIVERGENT"


def test_full_last_run_id_and_output_are_bound(tmp_path):
    values = actual_payloads()
    values["full_state"]["last_run_id"] = "previous-full"
    result = make_live(tmp_path, values).collector(NOW)
    assert result.evidence.status == "DIVERGENT"
    assert result.data["lanes"][0]["status"] == "DIVERGENT"


def test_active_full_report_accepts_prior_state_with_installed_corroboration(tmp_path):
    values = actual_payloads()
    values["full_report"] = initial_daemon_run_report(
        run_id="full-active",
        generated_at=NOW - timedelta(seconds=10),
        current_time=(NOW - timedelta(seconds=10)).isoformat(),
        output_dir=Path("artifacts/full-active"),
        lock_path=Path("runtime/collector.lock"),
        state_path=Path("runtime/state.json"),
        odds_capture_state_path=Path("runtime/odds_capture_state.json"),
        autonomous_odds_capture_enabled=True,
        autonomous_result_capture_enabled=True,
    )
    # Persisted full state is intentionally the previous completed lifecycle.
    values["full_state"].update(
        last_run_id="full-previous",
        last_output_dir="artifacts/full-previous",
        last_verdict="DAEMON_READY",
    )
    lane = make_live(
        tmp_path,
        values,
        full_status=("active", "running", 4321),
    ).collector(NOW).data["lanes"][0]
    assert lane["status"] == "ACTIVE"
    assert lane["run_id"] == "full-active"


def test_completed_full_report_requires_exact_state_identity(tmp_path):
    values = actual_payloads()
    assert make_live(tmp_path / "match", values).collector(NOW).data["lanes"][0]["status"] == "RECEIPT_READY"
    conflict = actual_payloads()
    conflict["full_state"]["last_output_dir"] = "artifacts/another-full"
    assert make_live(tmp_path / "conflict", conflict).collector(NOW).data["lanes"][0]["status"] == "DIVERGENT"


def test_actual_lock_held_full_report_pair_is_known_completed_lifecycle(tmp_path):
    values = actual_payloads()
    values["full_report"] = lock_held_daemon_run_report(
        run_id="full-1",
        generated_at=NOW - timedelta(seconds=30),
        current_time=(NOW - timedelta(seconds=30)).isoformat(),
        output_dir=Path("artifacts/full-1"),
        lock_path=Path("runtime/shadow.lock"),
        lock_details={"reason": "active_lock", "existing_lock": {
            "run_id": "owner_odds_capture", "output_dir": "private/runtime-output",
            "pid": 123, "hostname": "private-host", "started_at": (NOW - timedelta(minutes=1)).isoformat(),
        }},
        odds_capture_state={
            "run_id": "prior-odds", "final_status": "ODDS_CAPTURE_ONLY_READY",
            "odds_capture_status": "APPENDED", "status": "READY",
            "inserted_live_odds_rows": 12, "ready_count": 3,
            "status_counts": {"APPENDED": 3}, "blocked_attempt_count": 0,
            "next_meaningful_action": "WAIT_UNTIL_NEXT_FIXED_WINDOW",
            "next_meaningful_action_at": (NOW + timedelta(minutes=1)).isoformat(),
        },
    )
    # The lock-held producer returns before updating the completed-run state;
    # the state therefore legitimately remains the previous lifecycle.
    lane = make_live(tmp_path, values).collector(NOW).data["lanes"][0]
    assert lane["status"] == "CAPTURE_WINDOW_CLOSED"
    assert lane["operational_context"] == {
        "final_status": None, "final_verdict": "PARTIAL_DAEMONIZATION",
        "status": "SKIPPED_LOCK_HELD",
        "next_meaningful_action": "WAIT_UNTIL_NEXT_FIXED_WINDOW",
        "next_meaningful_action_at": (NOW + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "lock_owner": {"kind": "odds_capture", "run_id": "owner_odds_capture", "started_at": (NOW - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")},
        "recent_capture": {"inserted_live_odds_rows": 12, "ready_count": 3, "status_counts": {"APPENDED": 3}, "blocked_attempt_count": 0},
    }
    rendered = json.dumps(lane)
    for hidden in ("private/runtime-output", "private-host", '"pid"', "runtime/shadow.lock"):
        assert hidden not in rendered


def test_actual_completed_full_report_projects_current_odds_capture_fields(tmp_path):
    values = actual_payloads()
    values["full_report"].update({
        "odds_capture_next_meaningful_action": "WAIT_UNTIL_NEXT_FIXED_WINDOW",
        "odds_capture_next_meaningful_action_at": (
            NOW + timedelta(minutes=1)
        ).isoformat(),
        "autonomous_live_odds_capture_inserted_rows": 7,
        "autonomous_live_odds_capture_ready_count": 4,
    })

    context = make_live(tmp_path, values).collector(NOW).data["lanes"][0][
        "operational_context"
    ]
    assert context["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert context["next_meaningful_action_at"] == (
        NOW + timedelta(minutes=1)
    ).isoformat().replace("+00:00", "Z")
    assert context["recent_capture"] == {
        "inserted_live_odds_rows": 7,
        "ready_count": 4,
        "status_counts": None,
        "blocked_attempt_count": None,
    }


def test_odds_same_run_refresh_source_and_status_are_bound(tmp_path):
    values = actual_payloads()
    values["odds_report"]["odds_capture_refresh_report"] = {}
    assert make_live(tmp_path / "external", values).collector(NOW).data["lanes"][1]["status"] == "INTEGRITY_FAILED"

    conflicting = actual_payloads()
    conflicting["odds_state"]["autopilot_output_dir"] = "artifacts/another"
    assert make_live(tmp_path / "conflict", conflicting).collector(NOW).data["lanes"][1]["status"] == "DIVERGENT"


def test_odds_refresh_publishes_exact_producer_raw_byte_hash(tmp_path):
    values = actual_payloads()
    live = make_live(
        tmp_path,
        values,
        serialization_policies={
            "odds_refresh": JsonSerializationPolicy.PRODUCER_PRETTY_SORTED
        },
    )
    refresh_path = tmp_path / "artifacts/autopilot-9/odds_capture_refresh_report.json"
    from scripts.shadow_autopilot_daemon import write_json
    write_json(refresh_path, values["odds_refresh"])
    lane = live.collector(NOW).data["lanes"][1]
    assert lane["status"] == "RECEIPT_READY"
    assert lane["reference_hashes"]["odds_refresh"] == hashlib.sha256(
        refresh_path.read_bytes()
    ).hexdigest()
    assert lane["reference_hashes"]["odds_refresh"] != hashlib.sha256(
        canonical(values["odds_refresh"])
    ).hexdigest()
    observation = live.collector(NOW)
    rendered = json.dumps({"evidence": observation.evidence.to_dict(), "data": observation.data})
    assert "operator_ui.odds_refresh" not in rendered
    assert "artifacts/autopilot-9/odds_capture_refresh_report.json" not in rendered
    assert str(tmp_path) not in rendered


def test_odds_refresh_fixed_source_locator_substitution_fails_closed(tmp_path):
    values = actual_payloads()
    values["odds_report"]["autopilot_output_dir"] = "artifacts/substituted"
    values["odds_state"]["autopilot_output_dir"] = "artifacts/substituted"
    observation = make_live(tmp_path, values).collector(NOW)
    lane = observation.data["lanes"][1]
    assert lane["status"] == "DIVERGENT"
    assert observation.evidence.source_locator == "operator_ui.full_report"
    rendered = json.dumps({"evidence": observation.evidence.to_dict(), "data": observation.data})
    assert "artifacts/substituted/odds_capture_refresh_report.json" not in rendered
    assert str(tmp_path) not in rendered


def test_odds_refresh_failure_envelope_keeps_public_symbolic_locator(tmp_path):
    values = actual_payloads()
    values["odds_refresh"] = {**values["odds_refresh"], "selected_count": 1}
    live = make_live(tmp_path, values)
    envelope, lane = live._lane(lane="ODDS_ONLY", now=NOW)
    assert lane["status"] == "DIVERGENT"
    assert envelope.source_locator == "operator_ui.odds_refresh"
    rendered = envelope.to_json()
    assert "artifacts/autopilot-9" not in rendered
    assert str(tmp_path) not in rendered


@pytest.mark.parametrize(
    ("final_status", "status", "lane_status"),
    [
        ("ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW", "WAITING", "CAPTURE_WINDOW_CLOSED"),
        ("SKIPPED_LOCK_HELD", "SKIPPED_LOCK_HELD", "CAPTURE_WINDOW_CLOSED"),
        ("SKIPPED_FULL_DAEMON_LOCK_HANDOFF", "SKIPPED_FULL_DAEMON_LOCK_HANDOFF", "CAPTURE_WINDOW_CLOSED"),
    ],
)
def test_no_refresh_producer_lifecycles_are_truthful_not_integrity_failures(
    tmp_path, final_status, status, lane_status
):
    values = actual_payloads()
    report = values["odds_report"]
    report.update(final_status=final_status, status=status, autopilot_output_dir=None)
    report["odds_capture_refresh_report"] = {}
    state = values["odds_state"]
    if final_status == "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW":
        state.update(final_status=final_status, status=status, autopilot_output_dir=None)
    lane = make_live(tmp_path, values).collector(NOW).data["lanes"][1]
    assert lane["status"] == lane_status


def test_ready_with_blocked_attempts_is_not_receipt_ready(tmp_path):
    values = actual_payloads()
    values["odds_report"]["status"] = "READY_WITH_BLOCKED_ATTEMPTS"
    values["odds_state"]["status"] = "READY_WITH_BLOCKED_ATTEMPTS"
    assert make_live(tmp_path, values).collector(NOW).data["lanes"][1]["status"] == "CAPTURE_FAILED"


@pytest.mark.parametrize("value", ["-1s", "0s", "nan", "inf"])
def test_supported_nonpositive_or_nonfinite_duration_is_data_missing(tmp_path, value):
    timer = f"[Timer]\nOnUnitInactiveSec={value}\nAccuracySec=30s\n"
    assert make_live(tmp_path, full_timer=timer).collector(NOW).data["lanes"][0]["status"] == "DATA_MISSING"


def test_unknown_lifecycle_and_full_status_verdict_disagreement_fail_closed(tmp_path):
    unknown = actual_payloads()
    unknown["odds_report"].update(final_status="SOMETHING_NEW", status="READY")
    assert make_live(tmp_path / "unknown", unknown).collector(NOW).data["lanes"][1]["status"] == "INTEGRITY_FAILED"
    conflict = actual_payloads()
    conflict["full_report"]["status"] = "PARTIAL_DAEMONIZATION"
    assert make_live(tmp_path / "conflict", conflict).collector(NOW).data["lanes"][0]["status"] == "DIVERGENT"


@pytest.mark.parametrize("path", ["../prior", "/tmp/report"])
def test_refresh_path_traversal_fails_closed(tmp_path, path):
    values = actual_payloads()
    values["odds_state"]["autopilot_output_dir"] = path
    assert make_live(tmp_path, values).collector(NOW).data["lanes"][1]["status"] == "INTEGRITY_FAILED"


def test_refresh_timestamp_and_status_disagreement_fail_closed(tmp_path):
    future = actual_payloads()
    future["odds_report"]["odds_capture_refresh_report"]["generated_at"] = (NOW + timedelta(seconds=1)).isoformat()
    assert make_live(tmp_path / "future", future).collector(NOW).data["lanes"][1]["status"] == "DIVERGENT"
    conflict = actual_payloads()
    conflict["odds_state"]["odds_capture_refresh_status"] = "FAILED"
    assert make_live(tmp_path / "status", conflict).collector(NOW).data["lanes"][1]["status"] == "DIVERGENT"


def test_calendar_rejects_arbitrary_prefix_but_accepts_complete_date_token():
    assert _calendar_gap("*-*-* *:00,15,30,45") == 900
    with pytest.raises(ValueError):
        _calendar_gap("Mon *:00,15,30,45")


def test_real_corpus_catalog_and_deployment_shapes(tmp_path):
    live = make_live(tmp_path)
    assert live.corpus(NOW).evidence.status == "UNAVAILABLE/DATA_MISSING"
    models = live.models(NOW).data["models"]
    assert [(item["model_id"], item["role"]) for item in models] == [
        ("market_form_residual_v1", "LATEST_RESEARCH"),
        ("market_only_v1", "BASELINE"),
    ]
    assert models[1]["model_sha256"] is None
    assert models[1]["manifest_sha256"] is None
    system = live.system(NOW)
    assert system.evidence.status == "AVAILABLE/FRESH"
    assert system.data["components"][0]["service_status"]["full"] == {
        "active_state": "inactive", "sub_state": "dead", "exec_main_pid": 0,
    }
    rendered = json.dumps({"evidence": system.evidence.to_dict(), "data": system.data})
    assert "/srv/app" not in rendered
    assert str(tmp_path) not in rendered


def test_corpus_uses_report_time_and_exposes_admission_gap(tmp_path):
    values = actual_payloads(NOW - timedelta(seconds=86400))
    result = make_live(tmp_path, values).corpus(NOW)
    report = result.data["reports"][0]
    assert result.evidence.status == "UNAVAILABLE/DATA_MISSING"
    assert report["status"] == "UNAVAILABLE"
    assert "population_id" not in report
    assert "population_count" not in report
    assert "official-result publication/closure" in report["admission_gap"]
    assert "generated_at" not in values["corpus_manifest"]


def test_corpus_report_age_boundary_and_future_fail_closed(tmp_path):
    stale = make_live(tmp_path / "stale", actual_payloads(NOW - timedelta(seconds=86400, microseconds=1))).corpus(NOW)
    assert stale.evidence.status == "STALE"
    future = make_live(tmp_path / "future", actual_payloads(NOW + timedelta(microseconds=1))).corpus(NOW)
    assert future.evidence.status == "INVALID/INTEGRITY_FAILED"


def test_corpus_exact_manifest_and_bound_file_tamper_fail_closed(tmp_path):
    extra = actual_payloads()
    extra["corpus_manifest"]["files"]["packet/extra.csv"] = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    assert make_live(tmp_path / "extra", extra).corpus(NOW).evidence.status == "DIVERGENT"
    live = make_live(tmp_path / "replace")
    (tmp_path / "replace/packet/race_evidence_inventory.csv").write_bytes(b"changed")
    assert live.corpus(NOW).evidence.status == "DIVERGENT"


def test_corpus_inventory_is_digest_only_but_final_status_retains_bytes(tmp_path, monkeypatch):
    values = actual_payloads()
    live = make_live(tmp_path, values)
    observed = {}
    original = live._reader.read_raw_authenticated
    def capture(key):
        result = original(key); observed[key] = result; return result
    monkeypatch.setattr(live._reader, "read_raw_authenticated", capture)
    assert live.corpus(NOW).evidence.status == "UNAVAILABLE/DATA_MISSING"
    csv_manifest = values["corpus_manifest"]["files"]["packet/race_evidence_inventory.csv"]
    assert observed["corpus_inventory_csv"][0].content_sha256 == hashlib.sha256(
        b"race_id\n1\n"
    ).hexdigest() == csv_manifest["sha256"]
    assert observed["corpus_inventory_csv"][2] == csv_manifest["bytes"]
    assert observed["corpus_inventory_csv"][1] is None
    jsonl_manifest = values["corpus_manifest"]["files"]["packet/race_evidence_inventory.jsonl"]
    assert observed["corpus_inventory_jsonl"][0].content_sha256 == hashlib.sha256(
        b'{"race_id":"1"}\n'
    ).hexdigest() == jsonl_manifest["sha256"]
    assert observed["corpus_inventory_jsonl"][2] == jsonl_manifest["bytes"]
    assert observed["corpus_inventory_jsonl"][1] is None
    assert observed["corpus_final_status"][1] == b"RACE_EVIDENCE_INVENTORY_READY_FOR_EVALUATION\n"


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_corpus_manifest_rejects_nonexact_top_level_envelope(tmp_path, mutation):
    values = actual_payloads()
    if mutation == "unknown":
        values["corpus_manifest"]["population_id"] = "identity-transplant"
        live = make_live(tmp_path, values)
    else:
        live = make_live(tmp_path, values)
        values["corpus_manifest"].pop("files")
        (tmp_path / "corpus_manifest.json").write_bytes(canonical(values["corpus_manifest"]))
    assert live.corpus(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.update(final_status="BOGUS"),
        lambda report: report.update(recommended_decision="BOGUS"),
        lambda report: report["summary_counts"].update(invented_count=1),
        lambda report: report["summary_counts"].update(race_union_count=2**53),
        lambda report: report["summary_counts"].update(race_union_count=True),
        lambda report: report["summary_counts"].update(race_union_count=2.0),
        lambda report: report["summary_counts"].update(race_union_count=-1),
        lambda report: report["summary_counts"].pop("live_odds_race_count"),
        lambda report: report["summary_counts"].update(
            shadow_races_complete_official_and_strict_odds=0
        ),
        lambda report: report["summary_counts"]["action_counts"].update(invented_action=1),
    ],
)
def test_corpus_rejects_unknown_malformed_or_contradictory_status_counts(tmp_path, mutate):
    values = actual_payloads()
    mutate(values["corpus_report"])
    assert make_live(tmp_path, values).corpus(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["official_artifact_summary"].update(source_sha256="0" * 64),
        lambda report: report["shadow_prediction_summary"].update(artifact_sha256="0" * 64),
        lambda report: report["db_summary"]["db_status"].update(db_path="substituted.db"),
        lambda report: report.update(population_id="transplanted"),
        lambda report: report["latest_backlog_append_report"].update(report_sha256="0" * 64),
    ],
)
def test_corpus_rejects_uncontracted_nested_identity_or_population_transplant(tmp_path, mutate):
    values = actual_payloads()
    mutate(values["corpus_report"])
    assert make_live(tmp_path, values).corpus(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"


def test_corpus_rejects_backlog_closure_disagreement(tmp_path):
    values = actual_payloads()
    values["corpus_report"]["latest_backlog_append_report"] = {
        "status": "FOUND",
        "path": "artifacts/evidence/official_result_evidence_append_backlog_x/official_result_evidence_append_backlog_report.json",
        "final_status": "APPENDED_OFFICIAL_RESULT_EVIDENCE_BACKLOG",
        "artifact_count": 1,
        "processed_count": 1,
        "status_counts": {"APPENDED_OFFICIAL_RESULT_EVIDENCE": 1},
        "inserted_race_rows": 1,
        "inserted_runner_rows": 8,
        "db_write_performed": False,
        "shared_lock_status": None,
        "shared_lock_release": None,
    }
    assert make_live(tmp_path, values).corpus(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"


@pytest.mark.parametrize("absolute", [False, True])
def test_corpus_accepts_genuine_relative_and_absolute_producer_locators_without_disclosure(tmp_path, absolute):
    values = actual_payloads()
    prefix = "/configured/evidence" if absolute else "configured/evidence"
    output = f"{prefix}/packet"
    values["corpus_report"].update(
        output_dir=output, artifact_roots=[prefix], db_path=f"{prefix}/greyhound.sqlite",
        inventory_csv=f"{output}/race_evidence_inventory.csv",
        inventory_jsonl=f"{output}/race_evidence_inventory.jsonl",
        scorecard_csv=f"{output}/race_evidence_scorecard.csv",
        scorecard_jsonl=f"{output}/race_evidence_scorecard.jsonl",
    )
    values["corpus_report"]["official_artifact_summary"]["artifact_discovery"][0]["input_artifact_dir"] = prefix
    values["corpus_report"]["db_summary"]["db_status"]["db_path"] = f"{prefix}/greyhound.sqlite"
    values["corpus_manifest"]["output_dir"] = output
    result = make_live(tmp_path, values).corpus(NOW)
    assert result.evidence.status == "UNAVAILABLE/DATA_MISSING"
    rendered = json.dumps(result.data)
    assert prefix not in rendered
    assert "population_count" not in rendered


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["official_artifact_summary"]["artifact_discovery"][0].update(mode="recursive_parent_discovery"),
        lambda r: r["official_artifact_summary"]["artifact_discovery"][0].update(discovered_child_artifact_count=True),
        lambda r: r["shadow_prediction_summary"].update(prediction_file_kind_counts={"shadow_predictions.jsonl": 2}),
        lambda r: r["shadow_prediction_summary"].update(prediction_file_scan_truncated=True),
        lambda r: r["db_summary"]["table_status"].pop("live_odds"),
        lambda r: r["db_summary"]["counts"].update(strict_live_odds_rows=2),
        lambda r: r["scorecard_metrics"].update(model_top1_accuracy=-0.1),
        lambda r: r["scorecard_metrics"].update(model_top1_accuracy=True),
        lambda r: r["scorecard_metrics"].update(evaluation_race_count=0),
        lambda r: r["scorecard_metrics"]["official_result_gap_action_counts"].update(capture_official_result=3),
        lambda r: r.update(top_gap_races=[{"race_id": "race-1", "race_date": None, "venue": None, "race_number": 1, "recommended_next_action": "ready_for_unified_evidence_evaluation", "shadow_box_count": 1, "official_result_db_box_count": 0, "strict_live_odds_box_count": 0}]),
        lambda r: r["no_write_guarantees"].update(db_write=True),
    ],
)
def test_corpus_rejects_adversarial_nested_producer_families(tmp_path, mutate):
    values = actual_payloads()
    mutate(values["corpus_report"])
    assert make_live(tmp_path, values).corpus(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_corpus_metric_validator_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError):
        _finite_metric(value)


def test_model_catalog_is_exact_finite_order_and_hash_bound(tmp_path):
    reordered = actual_payloads()
    reordered["model_catalog"]["configs"].reverse()
    assert make_live(tmp_path / "order", reordered).models(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"
    mismatch = actual_payloads()
    mismatch["model_catalog"]["configs"][0]["config_sha256"] = "0" * 64
    assert make_live(tmp_path / "hash", mismatch).models(NOW).evidence.status == "DIVERGENT"


def test_model_catalog_observation_age_and_unavailable_evaluation(tmp_path):
    live = make_live(tmp_path, now=NOW)
    fresh = live.models(NOW + timedelta(seconds=60))
    assert fresh.evidence.status == "AVAILABLE/FRESH"
    assert all(item["evaluation_status"] == "UNAVAILABLE" for item in fresh.data["models"])
    assert all(item["evaluation_claim"] is None and item["evaluation_hashes"] == {} for item in fresh.data["models"])
    assert live.models(NOW + timedelta(seconds=60, microseconds=1)).evidence.status == "STALE"


def test_model_catalog_age_is_bound_to_immutable_authority_observation(tmp_path):
    values = actual_payloads()
    live = make_live(tmp_path, values, now=NOW - timedelta(minutes=10))
    source = live._reader._sources["model_catalog"]
    live = LiveEvidenceAdapters(
        OperatorEvidenceReader(
            {**live._reader._sources, "model_catalog": replace(
                source, json=replace(
                    source.json,
                    authority_observed_at=(NOW - timedelta(minutes=10)).isoformat(),
                )
            )},
            raw_sources=live._reader._raw_sources,
            clock=lambda: NOW,
        ),
        units=live._units,
    )
    assert live.models(NOW).evidence.status == "STALE"


@pytest.mark.parametrize("lane", ["full", "odds"])
def test_system_rejects_wrong_observed_service_unit_identity(tmp_path, lane):
    live = make_live(tmp_path)
    units = replace(
        live._units,
        **{f"{lane}_unit_name": "shadow-autopilot.service" if lane == "odds" else "shadow-autopilot-odds-capture.service"},
    )
    assert LiveEvidenceAdapters(live._reader, units=units).system(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"


@pytest.mark.parametrize("family", ["config_schema", "artifact_manifest"])
def test_model_catalog_rejects_consistently_resealed_identity_transplant(tmp_path, family):
    values = actual_payloads()
    repo = Path(__file__).resolve().parents[2]
    overrides = {}
    entry = values["model_catalog"]["configs"][0]
    if family == "config_schema":
        config = json.loads((repo / "configs/prediction/manual-default.json").read_bytes())
        config["model"] = "transplanted_model_v1"
        schema = json.loads((repo / "configs/prediction/schemas/market_form_residual_v1.schema.json").read_bytes())
        schema["properties"]["model"]["const"] = "transplanted_model_v1"
        config_raw, schema_raw = canonical(config), canonical(schema)
        overrides.update(model_latest_config=config_raw, model_latest_schema=schema_raw)
        entry["resolved_config"] = config
        entry["config_sha256"] = hashlib.sha256(config_raw).hexdigest()
        entry["model"]["schema_sha256"] = hashlib.sha256(schema_raw).hexdigest()
    else:
        model = json.loads((repo / "artifacts/frozen_models/market_form_residual_v1/model.json").read_bytes())
        manifest = json.loads((repo / "artifacts/frozen_models/market_form_residual_v1/manifest.json").read_bytes())
        model["schema_version"] = "transplanted_frozen_model_v1"
        model_raw = canonical(model)
        manifest["model_schema_version"] = model["schema_version"]
        manifest["model_sha256"] = hashlib.sha256(model_raw).hexdigest()
        manifest_raw = canonical(manifest)
        overrides.update(model_latest_artifact=model_raw, model_latest_manifest=manifest_raw)
        entry["model"]["model_sha256"] = hashlib.sha256(model_raw).hexdigest()
        entry["model"]["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    assert make_live(tmp_path, values, raw_overrides=overrides).models(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"


def test_model_catalog_rejects_resealed_manifest_dataset_identity_transplant(tmp_path):
    values = actual_payloads()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (repo / "artifacts/frozen_models/market_form_residual_v1/manifest.json").read_bytes()
    )
    manifest["input_manifest"]["dataset_id"] = "identity-transplant"
    manifest_raw = canonical(manifest)
    values["model_catalog"]["configs"][0]["model"]["manifest_sha256"] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result = make_live(
        tmp_path,
        values,
        raw_overrides={"model_latest_manifest": manifest_raw},
    ).models(NOW)
    assert result.evidence.status == "INVALID/INTEGRITY_FAILED"


def test_live_adapter_contains_no_side_effect_authority():
    source = inspect.getsource(LiveEvidenceAdapters)
    for forbidden in ("subprocess", "systemctl", "requests", "sqlite", ".glob(", ".rglob(", ".walk(", ".write", ".unlink("):
        assert forbidden not in source


@pytest.mark.parametrize(
    ("offset", "expected"),
    [(timedelta(seconds=60), "HEALTHY"), (timedelta(seconds=60, microseconds=1), "STALE")],
)
def test_installed_unit_freshness_boundary(tmp_path, offset, expected):
    result = make_live(tmp_path, units_observed_at=NOW - offset).system(NOW)
    assert result.data["components"][0]["status"] == expected


@pytest.mark.parametrize("observed_at", [NOW + timedelta(seconds=1), NOW.replace(tzinfo=None)])
def test_future_or_naive_installed_unit_observation_fails_closed(tmp_path, observed_at):
    result = make_live(tmp_path, units_observed_at=observed_at).system(NOW)
    assert result.evidence.status == "INVALID/INTEGRITY_FAILED"
    rendered = json.dumps({"evidence": result.evidence.to_dict(), "data": result.data})
    assert "/srv/app" not in rendered
    assert str(tmp_path) not in rendered


def test_odds_operational_context_preserves_exact_producer_fields(tmp_path):
    values = actual_payloads()
    fields = {
        "next_meaningful_action": "WAIT_UNTIL_NEXT_FIXED_WINDOW",
        "next_meaningful_action_at": (NOW + timedelta(minutes=1)).isoformat(),
        "inserted_live_odds_rows": 0, "ready_count": 2,
        "status_counts": {"SKIPPED_ALREADY_CAPTURED": 2},
        "blocked_attempt_count": 0,
    }
    values["odds_report"].update(fields)
    values["odds_state"].update(fields)
    context = make_live(tmp_path, values).collector(NOW).data["lanes"][1]["operational_context"]
    assert context["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert context["final_verdict"] is None
    assert context["status"] == "READY"
    assert context["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert context["recent_capture"] == {
        "inserted_live_odds_rows": 0, "ready_count": 2,
        "status_counts": {"SKIPPED_ALREADY_CAPTURED": 2},
        "blocked_attempt_count": 0,
    }


@pytest.mark.parametrize("field", [
    "next_meaningful_action", "next_meaningful_action_at",
    "inserted_live_odds_rows", "ready_count", "status_counts",
    "blocked_attempt_count",
])
def test_same_run_odds_operational_mismatch_diverges(tmp_path, field):
    values = actual_payloads()
    matching = {
        "next_meaningful_action": "WAIT_UNTIL_NEXT_FIXED_WINDOW",
        "next_meaningful_action_at": (NOW + timedelta(minutes=1)).isoformat(),
        "inserted_live_odds_rows": 1, "ready_count": 2,
        "status_counts": {"APPENDED": 1}, "blocked_attempt_count": 0,
    }
    values["odds_report"].update(matching)
    values["odds_state"].update(matching)
    values["odds_state"][field] = {
        "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
        "next_meaningful_action_at": (NOW + timedelta(minutes=2)).isoformat(),
        "inserted_live_odds_rows": 2,
        "ready_count": 3,
        "status_counts": {"APPENDED": 2},
        "blocked_attempt_count": 1,
    }[field]
    lane = make_live(tmp_path, values).collector(NOW).data["lanes"][1]
    assert lane["status"] == "DIVERGENT"


def test_missing_operational_values_are_unavailable_not_zero(tmp_path):
    lanes = make_live(tmp_path).collector(NOW).data["lanes"]
    for lane in lanes:
        capture = lane["operational_context"]["recent_capture"]
        assert capture == {
            "inserted_live_odds_rows": None, "ready_count": None,
            "status_counts": None, "blocked_attempt_count": None,
        }


@pytest.mark.parametrize("field,value", [
    ("inserted_live_odds_rows", -1),
    ("ready_count", 1.5),
    ("status_counts", {"APPENDED": -1}),
    ("blocked_attempt_count", 1_000_000_001),
])
def test_invalid_operational_counts_fail_closed(tmp_path, field, value):
    values = actual_payloads()
    values["odds_report"][field] = value
    result = make_live(tmp_path, values).collector(NOW)
    assert result.data["lanes"][1]["status"] == "INTEGRITY_FAILED"


def test_deployed_identity_missing_and_mismatch_propagate_to_collector(tmp_path):
    missing = actual_payloads()
    missing["deployment_manifest"]["deployed_tree"] = None
    assert make_live(tmp_path / "missing", missing).collector(NOW).evidence.status == "UNAVAILABLE/DATA_MISSING"

    mismatch = actual_payloads()
    mismatch["deployment_manifest"]["deployed_commit"] = "f" * 40
    live = make_live(tmp_path / "mismatch", mismatch)
    assert live.system(NOW).evidence.status == "DIVERGENT"
    assert live.collector(NOW).evidence.status == "DIVERGENT"


@pytest.mark.parametrize(
    "missing",
    ["identity", "manifest_unit_hashes", "working_directory", "supplied_unit_hash"],
)
def test_valid_incomplete_deployment_discloses_bounded_component(tmp_path, missing):
    values = actual_payloads()
    if missing == "identity":
        values["deployment_manifest"]["deployed_tree"] = None
    elif missing == "manifest_unit_hashes":
        # make_live adds the normal map; an explicit null remains a valid
        # observation that the adapter must classify as incomplete.
        values["deployment_manifest"]["installed_unit_sha256"] = None
    live = make_live(tmp_path, values)
    if missing == "manifest_unit_hashes":
        manifest = tmp_path / "deployment_manifest.json"
        payload = json.loads(manifest.read_text())
        payload["installed_unit_sha256"] = None
        manifest.write_bytes(canonical(payload))
    elif missing == "working_directory":
        live._units = replace(live._units, working_directory=None)
    elif missing == "supplied_unit_hash":
        live._units = replace(live._units, full_timer_sha256=None)

    result = live.system(NOW)

    assert result.evidence.status == "UNAVAILABLE/DATA_MISSING"
    component = result.data["components"][0]
    assert component["status"] == "DEGRADED"
    assert component["source_commit"] == "b" * 40
    if missing == "identity":
        assert component["deployed_tree"] is None
    else:
        assert component["deployed_tree"] == "c" * 40
    assert component["reference_hashes"] is None
    assert component["service_status"]["odds"] == {
        "active_state": "active", "sub_state": "waiting", "exec_main_pid": 0,
    }


def test_malformed_incomplete_deployment_suppresses_component(tmp_path):
    values = actual_payloads()
    values["deployment_manifest"]["deployed_tree"] = "NOT-A-GIT-IDENTITY"
    result = make_live(tmp_path, values).system(NOW)
    assert result.evidence.status == "INVALID/INTEGRITY_FAILED"
    assert result.data == {}


@pytest.mark.parametrize("variant", ["wrong_schema", "missing_field", "unknown_field", "empty"])
def test_deployment_manifest_structure_is_exact(tmp_path, variant):
    values = actual_payloads()
    manifest = values["deployment_manifest"]
    if variant == "wrong_schema":
        manifest["schema_version"] = "operator_ui_deployment_manifest_v0"
    elif variant == "missing_field":
        del manifest["deployed_tree"]
    elif variant == "unknown_field":
        manifest["unexpected"] = "value"
    else:
        live = make_live(tmp_path, values)
        (tmp_path / "deployment_manifest.json").write_bytes(canonical({}))

    result = (live if variant == "empty" else make_live(tmp_path, values)).system(NOW)

    assert result.evidence.status == "INVALID/INTEGRITY_FAILED"
    assert result.data == {}


def test_malformed_deployment_manifest_structure_is_invalid(tmp_path):
    live = make_live(tmp_path)
    (tmp_path / "deployment_manifest.json").write_bytes(canonical([]))

    result = live.system(NOW)

    assert result.evidence.status == "INVALID/INTEGRITY_FAILED"
    assert result.data == {}


@pytest.mark.parametrize("lane", ["full", "odds"])
@pytest.mark.parametrize("field", ["active_state", "sub_state", "exec_main_pid"])
def test_missing_service_observation_is_valid_incomplete(tmp_path, lane, field):
    live = make_live(tmp_path)
    live._units = replace(live._units, **{f"{lane}_{field}": None})

    result = live.system(NOW)

    assert result.evidence.status == "UNAVAILABLE/DATA_MISSING"
    component = result.data["components"][0]
    assert component["status"] == "DEGRADED"
    assert component["service_status"][lane][field] is None


def _set_deployment_divergence(live, root, kind):
    manifest = root / "deployment_manifest.json"
    payload = json.loads(manifest.read_text())
    if kind == "source_commit":
        payload["deployed_commit"] = "f" * 40
    elif kind == "source_tree":
        payload["deployed_tree"] = "f" * 40
    elif kind == "working_directory":
        payload["working_directory"] = "/srv/other"
    elif kind == "manifest_unit_hash":
        payload["installed_unit_sha256"]["full_timer"] = "f" * 64
    else:
        live._units = replace(live._units, full_timer_sha256="f" * 64)
        return
    manifest.write_bytes(canonical(payload))


@pytest.mark.parametrize("lane", ["full", "odds"])
@pytest.mark.parametrize("field", ["active_state", "sub_state", "exec_main_pid"])
@pytest.mark.parametrize(
    "divergence",
    ["source_commit", "source_tree", "working_directory", "manifest_unit_hash", "supplied_unit_hash"],
)
def test_missing_service_observation_never_masks_deployment_divergence(
    tmp_path, lane, field, divergence
):
    live = make_live(tmp_path)
    live._units = replace(live._units, **{f"{lane}_{field}": None})
    _set_deployment_divergence(live, tmp_path, divergence)

    result = live.system(NOW)

    assert result.evidence.status == "DIVERGENT"
    component = result.data["components"][0]
    assert component["status"] == "DIVERGENT"
    assert component["service_status"][lane][field] is None
    assert component["reference_hashes"] is None


@pytest.mark.parametrize("lane", ["full", "odds"])
@pytest.mark.parametrize("field", ["active_state", "sub_state", "exec_main_pid"])
@pytest.mark.parametrize(
    ("age", "outer", "component_status"),
    [
        (timedelta(seconds=60), "UNAVAILABLE/DATA_MISSING", "DEGRADED"),
        (timedelta(seconds=60, microseconds=1), "STALE", "STALE"),
    ],
)
def test_incomplete_service_observation_respects_deployment_freshness_boundary(
    tmp_path, lane, field, age, outer, component_status
):
    observed = NOW - age
    live = make_live(tmp_path, actual_payloads(at=observed), units_observed_at=observed)
    live._units = replace(live._units, **{f"{lane}_{field}": None})

    result = live.system(NOW)

    assert result.evidence.status == outer
    component = result.data["components"][0]
    assert component["status"] == component_status
    assert component["age_seconds"] == age.total_seconds()
    assert component["service_status"][lane][field] is None


@pytest.mark.parametrize("lane", ["full", "odds"])
@pytest.mark.parametrize("field", ["active_state", "sub_state", "exec_main_pid"])
def test_stale_incomplete_service_observation_remains_divergent_on_proved_drift(
    tmp_path, lane, field
):
    observed = NOW - timedelta(seconds=60, microseconds=1)
    live = make_live(tmp_path, actual_payloads(at=observed), units_observed_at=observed)
    live._units = replace(live._units, **{f"{lane}_{field}": None})
    _set_deployment_divergence(live, tmp_path, "source_commit")

    assert live.system(NOW).evidence.status == "DIVERGENT"


@pytest.mark.parametrize(
    "variant",
    ["empty", "missing_full_timer", "missing_full_service", "missing_odds_timer", "missing_odds_service", "extra", "malformed"],
)
def test_present_deployment_unit_hash_mapping_requires_exact_valid_shape(tmp_path, variant):
    live = make_live(tmp_path)
    manifest = tmp_path / "deployment_manifest.json"
    payload = json.loads(manifest.read_text())
    hashes = payload["installed_unit_sha256"]
    if variant == "empty":
        payload["installed_unit_sha256"] = {}
    elif variant.startswith("missing_"):
        del hashes[variant.removeprefix("missing_")]
    elif variant == "extra":
        hashes["other"] = "a" * 64
    else:
        hashes["full_timer"] = "not-a-digest"
    manifest.write_bytes(canonical(payload))

    result = live.system(NOW)

    assert result.evidence.status == "INVALID/INTEGRITY_FAILED"
    assert result.data == {}


@pytest.mark.parametrize(
    "field,value",
    [("full_active_state", 3), ("odds_sub_state", ""), ("full_exec_main_pid", -1)],
)
def test_malformed_service_observation_suppresses_component(tmp_path, field, value):
    live = make_live(tmp_path)
    live._units = replace(live._units, **{field: value})

    result = live.system(NOW)

    assert result.evidence.status == "INVALID/INTEGRITY_FAILED"
    assert result.data == {}


def test_no_browser_path_shell_scan_write_service_or_database_surface():
    assert tuple(inspect.signature(LiveEvidenceAdapters.collector).parameters) == ("self", "now")
    source = inspect.getsource(__import__("src.operator_ui.live_adapters", fromlist=["x"]))
    for forbidden in ("subprocess", "systemctl", "sqlite3", "requests", "glob(", "rglob(", "write_text(", "write_bytes(", "unlink("):
        assert forbidden not in source


@pytest.mark.parametrize(
    ("age", "jump_offset", "count"),
    [(300, 1, 1), (300.000001, 1, 0), (0, 1, 1), (0, 0, 0), (0, -1, 0)],
)
def test_upcoming_verified_view_exact_boundaries_and_identity(
    tmp_path, monkeypatch, age, jump_offset, count
):
    generated = NOW - timedelta(seconds=age)
    race_id = "Race 5 - GUNN - 2026-07-19"
    row = {
        "race_id": race_id, "race_url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5",
        "date": "2026-07-19", "venue": "GUNN", "race_number": 5,
        "jump_datetime": (NOW + timedelta(seconds=jump_offset)).isoformat(),
        "runner_set_sha256": "a" * 64,
        "runners": [
            {"box_number": 1, "dog_name": "ONE", "identity": "ONE", "source_native_runner_id": None, "scratch_state": "ACTIVE"},
            {"box_number": 2, "dog_name": "TWO", "identity": "TWO", "source_native_runner_id": "22", "scratch_state": "ACTIVE"},
        ],
    }
    view = VerifiedCurrentRaceIndex(
        "collector_current_race_index_v2", "run-1", generated.isoformat(),
        "1" * 64, b"packet", (row,), "refresh.json", "2" * 64,
        "3" * 64, "4" * 64, "5" * 64,
    )
    monkeypatch.setattr(live_module, "bounded_current_race_index", lambda **kwargs: view)
    adapter = make_live(
        tmp_path, upcoming_races=UpcomingRaceSource(tmp_path / "index", tmp_path)
    )
    result = adapter.upcoming(NOW)
    expected_status = "STALE" if age > 300 else "AVAILABLE/FRESH"
    assert result.evidence.status == expected_status
    assert result.data == {} if expected_status == "STALE" else len(result.data["races"]) == count
    if count:
        race = result.data["races"][0]
        assert race["race_id"] == race["source_race_id"] == race_id
        assert race["route_id"].startswith("r1.")
        assert race["meeting_slug"] is None and race["venue"] == "GUNN"
        assert [runner["runner_id"] for runner in race["runners"]] == ["ONE", "TWO"]
        assert [runner["source_runner_id"] for runner in race["runners"]] == [None, "22"]
        assert {runner["scratch_state"] for runner in race["runners"]} == {"ACTIVE"}
        assert adapter.race_detail(race["route_id"], NOW).data["race"] == race
        assert adapter.race_detail("hostile..route", NOW).data == {}


def test_fresh_source_with_only_post_jump_rows_is_verified_empty(tmp_path, monkeypatch):
    row = {
        "race_id": "Race 5 - GUNN - 2026-07-19",
        "race_url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5",
        "date": "2026-07-19", "venue": "GUNN", "race_number": 5,
        "jump_datetime": NOW.isoformat(), "runner_set_sha256": "a" * 64,
        "runners": ({"box_number": 1, "dog_name": "ONE", "identity": "ONE", "source_native_runner_id": None, "scratch_state": "ACTIVE"},),
    }
    view = VerifiedCurrentRaceIndex(
        "collector_current_race_index_v2", "run-1", NOW.isoformat(), "1" * 64,
        b"packet", (row,), "refresh.json", "2" * 64, "3" * 64, "4" * 64, "5" * 64,
    )
    monkeypatch.setattr(live_module, "bounded_current_race_index", lambda **kwargs: view)
    result = make_live(tmp_path, upcoming_races=UpcomingRaceSource(tmp_path / "index", tmp_path)).upcoming(NOW)
    assert result.evidence.status == "AVAILABLE/FRESH"
    assert result.data == {"races": []}


@pytest.mark.parametrize("status", ["STALE", "INVALID/INTEGRITY_FAILED"])
def test_race_detail_preserves_nonfresh_snapshot_envelope(tmp_path, monkeypatch, status):
    upstream = make_live(tmp_path)._verified_envelope(
        now=NOW, policy="P-UPCOMING-300-PREJUMP",
        identity="collector_current_race_index_v2",
        locator="operator_ui.current_race_index", status=status,
        source_at=NOW - timedelta(seconds=1200), content_sha256="1" * 64,
        references={"publication": "2" * 64},
        evidence_identity={"run_id": "run-1"},
    )
    adapter = make_live(tmp_path)
    monkeypatch.setattr(adapter, "_race_snapshot", lambda _now: (upstream, []))
    detail = adapter.race_detail("r1.missing", NOW)
    assert detail.evidence == upstream
    assert detail.data == {}


def test_ui_stale_snapshot_beyond_predictor_window_retains_bound_identity(tmp_path, monkeypatch):
    generated = NOW - timedelta(seconds=1200, microseconds=1)
    view = VerifiedCurrentRaceIndex(
        "collector_current_race_index_v2", "run-1", generated.isoformat(),
        "1" * 64, b"packet", (), "refresh.json", "2" * 64,
        "3" * 64, "4" * 64, "5" * 64,
    )
    monkeypatch.setattr(live_module, "bounded_current_race_index", lambda **kwargs: view)
    result = make_live(
        tmp_path, upcoming_races=UpcomingRaceSource(tmp_path / "index", tmp_path)
    ).upcoming(NOW)
    assert result.evidence.status == "STALE"
    assert result.evidence.content_sha256 == "1" * 64
    assert dict(result.evidence.evidence_identity) == {
        "run_id": "run-1", "schema_version": "collector_current_race_index_v2"
    }
    assert result.evidence.source_at == generated.isoformat().replace("+00:00", "Z")
    assert result.data == {}


@pytest.mark.parametrize(
    ("code", "status", "availability", "integrity"),
    [
        ("CURRENT_INDEX_UNAVAILABLE", "UNAVAILABLE/DATA_MISSING", "missing", "unknown"),
        ("DISCOVERY_TIMEOUT", "UNAVAILABLE/DATA_MISSING", "error", "unknown"),
        ("CURRENT_INDEX_SOURCE_CHANGED", "INVALID/INTEGRITY_FAILED", "error", "failed"),
        ("CURRENT_INDEX_PUBLICATION_INVALID", "INVALID/INTEGRITY_FAILED", "error", "failed"),
    ],
)
def test_real_collector_rejection_codes_map_truthfully(tmp_path, monkeypatch, code, status, availability, integrity):
    def rejected(**kwargs):
        raise CaptureOneRejected(code)
    monkeypatch.setattr(live_module, "bounded_current_race_index", rejected)
    result = make_live(tmp_path, upcoming_races=UpcomingRaceSource(tmp_path / "index", tmp_path)).upcoming(NOW)
    assert (result.evidence.status, result.evidence.availability, result.evidence.schema_integrity) == (status, availability, integrity)
    assert "no operational claim" in result.evidence.supported_claim


@pytest.mark.parametrize(("age", "status"), [(60, "AVAILABLE/FRESH"), (60.000001, "STALE")])
def test_prediction_empty_index_uses_only_producer_publication_time(
    tmp_path, monkeypatch, age, status
):
    published = NOW - timedelta(seconds=age)
    view = VerifiedPredictionBundleIndex(
        "on_demand_prediction_bundle_index_v1", published.isoformat(), (),
        b"index", "a" * 64,
    )
    monkeypatch.setattr(live_module, "verify_prediction_bundle_index", lambda *args, **kwargs: view)
    result = make_live(
        tmp_path, prediction_bundles=PredictionBundleSource(tmp_path)
    ).recent_predictions(NOW)
    assert result.evidence.status == status
    assert result.evidence.source_at == published.isoformat().replace("+00:00", "Z")
    assert result.data == ({"predictions": []} if status == "AVAILABLE/FRESH" else {})


def test_real_missing_prediction_index_differs_from_malformed_index(tmp_path):
    adapter = make_live(tmp_path, prediction_bundles=PredictionBundleSource(tmp_path))
    missing = adapter.recent_predictions(NOW)
    assert missing.evidence.status == "UNAVAILABLE/DATA_MISSING"
    assert missing.evidence.availability == "missing"
    assert missing.evidence.schema_integrity == "unknown"
    (tmp_path / PREDICTION_BUNDLE_INDEX_NAME).write_bytes(b"not-json")
    malformed = adapter.recent_predictions(NOW)
    assert malformed.evidence.status == "INVALID/INTEGRITY_FAILED"
    assert malformed.evidence.schema_integrity == "failed"
    assert all("Exact producer-verified" not in item.evidence.supported_claim for item in (missing, malformed))


def test_unreadable_prediction_index_is_unavailable_for_listing_and_detail(tmp_path, monkeypatch):
    def unreadable(*args, **kwargs):
        raise PermissionError("configured prediction evidence is unreadable")
    monkeypatch.setattr(live_module, "verify_prediction_bundle_index", unreadable)
    adapter = make_live(tmp_path, prediction_bundles=PredictionBundleSource(tmp_path))
    for result in (
        adapter.recent_predictions(NOW),
        adapter.prediction_detail("prediction-1", NOW),
    ):
        assert result.evidence.status == "UNAVAILABLE/DATA_MISSING"
        assert result.evidence.availability == "unreadable"
        assert result.evidence.schema_integrity == "unknown"
        assert "no operational claim" in result.evidence.supported_claim
        assert result.data == {}


def test_legacy_index_blocks_listing_but_allows_verified_historical_detail(tmp_path, monkeypatch):
    prediction_id = "11111111-1111-4111-8111-111111111111"
    entry = {
        "directory": "prediction_20260719T120000000000+0000_aaaaaaaaaaaa",
        "prediction_id": prediction_id, "job_id": None,
        "generated_at": NOW.isoformat(), "status": "PREDICTION_BLOCKED",
        "blocker_stage": "PROTOCOL", "manifest_sha256": "b" * 64,
        "logical_bundle_sha256": "c" * 64,
    }
    legacy = {"schema_version": live_module.PREDICTION_BUNDLE_INDEX_SCHEMA, "entries": [entry]}
    (tmp_path / PREDICTION_BUNDLE_INDEX_NAME).write_bytes(prediction_canonical_bytes(legacy))
    bundle = SimpleNamespace(
        index_entry=entry, directory=entry["directory"],
        manifest={"files": {"result.json": {}}},
        result={
            "prediction_id": prediction_id, "job_id": None,
            "race": {"race_id": "Race 5 - GUNN - 2026-07-19"},
            "model": {"resolved": "market_only_v1", "artifact_sha256": None},
            "config": {"sha256": "d" * 64}, "status": "PREDICTION_BLOCKED",
            "blocker_stage": "PROTOCOL", "blocker": {"code": "PROTOCOL_FAILED"},
            "prediction": None, "evidence": {"runner_set_sha256": "e" * 64},
        },
    )
    monkeypatch.setattr(live_module, "verify_indexed_prediction_bundle", lambda root, selected: bundle)
    adapter = make_live(tmp_path, prediction_bundles=PredictionBundleSource(tmp_path))
    assert adapter.recent_predictions(NOW).evidence.status == "INVALID/INTEGRITY_FAILED"
    detail = adapter.prediction_detail(prediction_id, NOW)
    assert detail.evidence.status == "AVAILABLE/FRESH"
    assert detail.data["prediction"]["prediction_id"] == prediction_id


def test_prediction_verified_records_preserve_terminal_distinctions(tmp_path, monkeypatch):
    entries = tuple({
        "directory": f"bundle-{stage.lower()}", "prediction_id": f"prediction-{stage.lower()}",
        "job_id": None, "generated_at": NOW.isoformat(), "status": "PREDICTION_BLOCKED",
        "blocker_stage": stage, "manifest_sha256": "b" * 64,
        "logical_bundle_sha256": "c" * 64,
    } for stage in ("PROTOCOL", "VALIDATION", "SCORING"))
    view = VerifiedPredictionBundleIndex(
        "on_demand_prediction_bundle_index_v1", NOW.isoformat(), entries,
        b"index", "a" * 64,
    )
    monkeypatch.setattr(live_module, "verify_prediction_bundle_index", lambda *args, **kwargs: view)
    def verify(_root, entry):
        stage = entry["blocker_stage"]
        return SimpleNamespace(
            index_entry=entry, directory=entry["directory"],
            manifest={"files": {"result.json": {}, "request.json": {}, "config.json": {}, "model/config.schema.json": {}}},
            result={
                "prediction_id": entry["prediction_id"], "job_id": None,
                "race": {"race_id": "Race 5 - GUNN - 2026-07-19"},
                "model": {"resolved": "market_only_v1", "artifact_sha256": None, "requested": "market-only"},
                "config": {"sha256": "d" * 64}, "status": "PREDICTION_BLOCKED",
                "blocker_stage": stage, "blocker": {"code": f"{stage}_FAILED"},
                "prediction": None, "evidence": {"runner_set_sha256": "e" * 64},
            },
        )
    monkeypatch.setattr(live_module, "verify_indexed_prediction_bundle", verify)
    result = make_live(tmp_path, prediction_bundles=PredictionBundleSource(tmp_path)).recent_predictions(NOW)
    assert [item["blocker_stage"] for item in result.data["predictions"]] == ["PROTOCOL", "VALIDATION", "SCORING"]
    assert all(item["probabilities"] is None and item["job_id"] is None and item["model_sha256"] is None for item in result.data["predictions"])
    assert all(len(item["evidence_names"]) == len(set(item["evidence_names"])) for item in result.data["predictions"])



@pytest.mark.parametrize("lane", ["full", "odds"])
@pytest.mark.parametrize(
    "missing_fields",
    [
        ("active_state",),
        ("sub_state",),
        ("exec_main_pid",),
        ("active_state", "sub_state", "exec_main_pid"),
    ],
    ids=["active-state", "sub-state", "pid", "all-service-fields"],
)
def test_foreign_active_lock_identity_outranks_missing_service_observation(
    tmp_path, lane, missing_fields
):
    values = actual_payloads()
    if lane == "full":
        values["full_report"] = initial_daemon_run_report(
            run_id="full-active",
            generated_at=NOW - timedelta(seconds=10),
            current_time=(NOW - timedelta(seconds=10)).isoformat(),
            output_dir=Path("artifacts/full-active"),
            lock_path=Path("runtime/collector.lock"),
            state_path=Path("runtime/state.json"),
            odds_capture_state_path=Path("runtime/odds_capture_state.json"),
            autonomous_odds_capture_enabled=True,
            autonomous_result_capture_enabled=True,
        )
        report = values["full_report"]
        run_id = "full-active"
    else:
        report = values["odds_report"]
        report.update(
            final_status="ODDS_CAPTURE_ONLY_RUNNING",
            status="RUNNING",
            odds_capture_refresh_report={},
            autopilot_output_dir=None,
        )
        values["odds_state"].update(
            final_status="ODDS_CAPTURE_ONLY_RUNNING",
            status="RUNNING",
            autopilot_output_dir=None,
        )
        run_id = "odds-9"
    report.update(
        lock_owner_kind="full_daemon" if lane == "full" else "odds_capture",
        lock_owner_run_id="foreign-run",
        lock_owner_started_at=(NOW - timedelta(seconds=20)).isoformat(),
    )
    status = {"active_state": "active", "sub_state": "running", "exec_main_pid": 4321}
    for field in missing_fields:
        status[field] = None
    kwargs = {
        f"{lane}_status": (
            status["active_state"], status["sub_state"], status["exec_main_pid"]
        )
    }

    result = make_live(tmp_path, values, **kwargs).collector(NOW)
    selected = result.data["lanes"][0 if lane == "full" else 1]

    assert selected["run_id"] == run_id
    assert selected["status"] == "DIVERGENT"
    assert result.evidence.status == "DIVERGENT"
