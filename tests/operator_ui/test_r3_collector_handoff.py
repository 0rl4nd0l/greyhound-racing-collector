"""Invented collector publication through the outcome-free R3 admission gate."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from race_collection import synchronous_manual_capture as capture
from scripts import shadow_autopilot_v1 as autopilot
from src.operator_ui.job_store import JobInput, OperationalIndexProvenance
from src.operator_ui.journal_readiness import ResultAcquisitionReadiness
from src.operator_ui.r3_api import R3Rejected
from src.predictor.on_demand import canonical_bytes
from tests.race_collection.test_synchronous_manual_capture import _runner_coverage


def published_handoff(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    state = root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    source = root / "refresh/refresh_prejump_report.json"
    source.parent.mkdir(parents=True)
    now = datetime.fromisoformat("2026-07-19T12:55:00+10:00")
    url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    coverage = _runner_coverage(root, url, now)
    source.write_bytes(canonical_bytes({
        "status": "SUCCESS", "generated_at": now.isoformat(),
        "sidecar_metadata_coverage": coverage, "selected_count": 1,
        "selected_races": [{
            "date": "2026-07-19", "jump_datetime": "2026-07-19T13:05:00+10:00",
            "race_id": "Race 5 - GUNN - 2026-07-19",
            "race_id_aliases": ["Race 5 - GUNN - 2026-07-19", "Race 5 - GUNNEDAH - 2026-07-19"],
            "race_number": 5, "source_native_race_id": "15900", "race_time": "13:05",
            "race_url": url, "venue": "GUNN",
        }],
    }))
    published = autopilot.publish_current_race_index_after_refresh(
        state_path=state, evidence_root=root, output_dir=source.parent,
        run_id="invented-collector-run", source_refresh_report_path=source,
    )
    assert published["status"] == "PUBLISHED"
    monkeypatch.setattr(capture, "ROOT", tmp_path)

    def current():
        return capture.bounded_current_race_index(
            current_time=now, timeout_seconds=2,
            index_path=capture.current_race_index_path(state), evidence_root=root,
            max_age_seconds=300, return_verified_view=True,
        )

    view = current()
    race = view.races[0]
    runners = tuple({"box": r["box"], "name": r["display_name"],
                     "identity": r["identity"], "source_native_runner_id": r["source_native_runner_id"]}
                    for r in race["runners"])
    digest = "a" * 64
    job_input = JobInput(
        race["race_id"], race["jump_datetime"], race["runner_set_sha256"],
        "latest-research", "market_form_residual_v1", digest, digest, digest,
        "manual-default", digest, "receipt", runners,
        OperationalIndexProvenance.from_verified_current_race_index(view),
    )
    collector = tmp_path / "collector"
    collector.mkdir()
    unit = tmp_path / "collector.service"
    jobs, bundles = tmp_path / "jobs.sqlite3", tmp_path / "bundles"
    unit.write_text(
        f"[Service]\nWorkingDirectory={collector}\n"
        f"ExecStart=/python {collector}/scripts/shadow_autopilot_daemon.py run-once "
        f"--evidence-root {root} --enable-autonomous-result-capture "
        f"--r3-job-store {jobs} --r3-prediction-bundles {bundles}\n"
    )
    authority = {"working_directory": str(collector), "units": {"full_service": {
        "path": str(unit), "sha256": hashlib.sha256(unit.read_bytes()).hexdigest(),
    }}}
    readiness = ResultAcquisitionReadiness(
        root, authority=authority, races=lambda: current().races, verified_index=current,
        result_job_store=jobs, result_prediction_bundles=bundles,
    )
    return readiness, job_input, now, source


def test_collector_only_publication_can_admit_without_shadow_predictions(tmp_path, monkeypatch):
    readiness, job_input, now, _ = published_handoff(tmp_path, monkeypatch)
    readiness.require(job_input, now=now)
    assert not list(readiness.root.glob("daily_race_ingest_shadow_*"))


@pytest.mark.parametrize("mismatch", [
    "race", "runner", "runner_hash", "run", "source", "publication", "jump",
    "stale", "future", "job_store", "bundles", "missing_flag", "unit_drift",
    "unverified_index",
])
def test_configured_r3_result_lane_rejects_mismatched_prerequisites(tmp_path, monkeypatch, mismatch):
    readiness, inp, now, _ = published_handoff(tmp_path, monkeypatch)
    if mismatch == "race":
        inp = replace(inp, race_id="Race 6 - GUNN - 2026-07-19")
    elif mismatch == "runner":
        inp = replace(inp, ordered_runners=({**inp.ordered_runners[0], "source_native_runner_id": "999"}, *inp.ordered_runners[1:]))
    elif mismatch == "runner_hash":
        inp = replace(inp, runner_set_sha256="b" * 64)
    elif mismatch in {"run", "source", "publication"}:
        field = {"run": "run_id", "source": "source_refresh_sha256", "publication": "publication_sha256"}[mismatch]
        inp = replace(inp, operational_index_provenance=replace(inp.operational_index_provenance, **{field: "b" * 64}))
    elif mismatch == "jump":
        inp = replace(inp, jump_timestamp=(now + timedelta(minutes=11)).isoformat())
    elif mismatch == "stale":
        now += timedelta(seconds=301)
    elif mismatch == "future":
        now -= timedelta(seconds=1)
    elif mismatch == "job_store":
        readiness.result_job_store = tmp_path / "other-jobs.sqlite3"
    elif mismatch == "bundles":
        readiness.result_prediction_bundles = tmp_path / "other-bundles"
    elif mismatch in {"missing_flag", "unit_drift"}:
        unit = Path(readiness.authority["units"]["full_service"]["path"])
        unit.write_text(unit.read_text().replace(f"--r3-job-store {readiness.result_job_store}", ""))
        if mismatch == "missing_flag":
            readiness.authority["units"]["full_service"]["sha256"] = hashlib.sha256(unit.read_bytes()).hexdigest()
    else:
        readiness.verified_index = lambda: {"races": []}
    with pytest.raises(R3Rejected, match="RESULT_ACQUISITION_NOT_READY"):
        readiness.require(inp, now=now)


def test_r3_result_admission_does_not_fallback_when_collector_source_changes(tmp_path, monkeypatch):
    readiness, inp, now, source = published_handoff(tmp_path, monkeypatch)
    # The actual index reader, also used by bootstrap, authenticates the source
    # bytes. A fresh timestamp alone cannot make this altered publication usable.
    payload = json.loads(source.read_bytes())
    payload["selected_races"][0]["source_native_race_id"] = "999"
    source.write_bytes(canonical_bytes(payload))
    with pytest.raises(capture.CaptureOneRejected, match="CURRENT_INDEX_SOURCE_CHANGED"):
        readiness.require(inp, now=now)


@pytest.mark.parametrize("mismatch", [False, True])
def test_journal_allocates_only_after_exact_collector_handoff(tmp_path, monkeypatch, mismatch):
    from src.operator_ui.journal import JournalActivation, JournalCoordinator
    from src.operator_ui.job_store import JobStore
    from src.operator_ui.r3_api import R3Services, ResolvedSubmission
    from src.operator_ui.security import AuditStore

    readiness, inp, now, _ = published_handoff(tmp_path, monkeypatch)
    if mismatch:
        inp = replace(inp, operational_index_provenance=replace(
            inp.operational_index_provenance, source_refresh_sha256="b" * 64))
    clock = [now - timedelta(seconds=1)]
    store = JobStore(tmp_path / "jobs.sqlite3")
    audit = AuditStore(tmp_path / "audit.sqlite3")
    dispatched = []
    services = R3Services(
        store, lambda selected, observed: ResolvedSubmission(inp, inp.ordered_runners),
        lambda job_id, confirm: dispatched.append(job_id), lambda job, confirm: job,
        lambda job: None, clock=lambda: clock[0],
    )
    activation = JournalActivation(
        "12345678-1234-4123-8123-123456789abc", now, now + timedelta(minutes=5),
        1, "a" * 40, "a" * 64, inp.model_sha256, inp.config_sha256, (),
    )
    coordinator = JournalCoordinator(
        activation=activation, root=tmp_path / "journal", services=services,
        audit=audit, races=lambda: ({"race_id": inp.race_id, "jump_datetime": inp.jump_timestamp},),
        result_readiness=readiness, clock=lambda: clock[0],
    )
    assert coordinator.tick()["state"] == "WAITING_START"
    clock[0] = now
    report = coordinator.tick()
    if mismatch:
        assert report["admissions"][inp.race_id] == "RESULT_ACQUISITION_NOT_READY"
        assert store.recorded_jobs() == ()
        assert dispatched == []
    else:
        assert len(store.recorded_jobs()) == 1
        assert dispatched == [store.recorded_jobs()[0].job_id]
    assert store.verify() and audit.verify_chain()
