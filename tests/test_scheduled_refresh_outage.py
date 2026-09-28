import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts import run_freshness_rehearsal as run
from utils.sportsbet_access import SportsbetAccess


def outage(tmp_path, *, status=502, run_id="20990101T120000+0000_odds_capture"):
    evidence = tmp_path / "evidence"
    output = tmp_path / "output"
    gate = tmp_path / "gate.json"
    if not gate.exists():
        SportsbetAccess(gate).initialize(access_basis={"status": "permitted", "reference": "synthetic"})
    plan = {"operational_predictions": {"enabled": True}, "evidence_root": str(evidence),
            "sportsbet_access_state": str(gate)}
    root = evidence / ("shadow_autopilot_daemonization_v1_" + run_id)
    result = root / "phase-0-result.json"
    run.atomic_json(result, {"collection_phase": "refresh", "final_verdict": "COLLECTION_PHASE_BLOCKED"})
    run.atomic_json(root / "phase-checkpoint.json", {
        "cycle_id": run_id, "phases": [{"kind": "refresh", "status": "COMPLETE",
            "budget_exceeded": False, "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest()}]})
    refresh = evidence / ("shadow_autopilot_v1_" + run_id + "_phase_0") / "odds_capture_refresh_report.json"
    run.atomic_json(refresh, {"status": "METADATA_COVERAGE_INCOMPLETE",
        "reason": "no_selected_race_csv_sidecars", "accepted_csv_count": 0, "sidecar_count": 0,
        "downloads": [{"success": False, "result": {"success": False, "source_http_status": status}},
                      {"success": False, "result": {"error": "No CSV download link found",
                                                      "source_failure_category": "observed_export_absent"}}]})
    return output, plan, run_id, root, refresh


def test_two_failed_cycles_are_retained_without_request_or_success(tmp_path):
    failed = set()
    for n in range(3):
        output, plan, rid, root, refresh = outage(tmp_path, run_id=f"fixture{n}_odds_capture")
        assert run.record_refresh_outage(output, plan, rid, failed) is (n < 2)
        if n < 2:
            assert run.record_refresh_outage(output, plan, rid, failed)
            record = json.loads((output / "refresh-deferrals" / (rid + ".json")).read_text())
            assert record["request_retries_added"] == 0
            assert record["status"] == "FAILED_REFRESH_AWAITING_NORMAL_TIMER"
            assert record["refresh_sha256"] == hashlib.sha256(refresh.read_bytes()).hexdigest()
        assert json.loads(refresh.read_text())["status"] == "METADATA_COVERAGE_INCOMPLETE"
    assert len(failed) == 2
    assert SportsbetAccess(plan["sportsbet_access_state"]).read()["active"] is None


@pytest.mark.parametrize("change", ["403", "429", "500", "retry", "reset", "integrity", "budget", "capture", "accepted", "unknown", "stop", "ordinary"])
def test_nontransient_or_untrusted_failure_never_defers(tmp_path, change):
    output, plan, rid, root, refresh = outage(tmp_path)
    report = json.loads(refresh.read_text())
    phase = json.loads((root / "phase-checkpoint.json").read_text())
    if change.isdigit():
        report["downloads"][0]["result"]["source_http_status"] = int(change)
    elif change in {"retry", "reset"}:
        report["downloads"][0]["result"]["source_retry_after" if change == "retry" else "source_rate_limit_reset"] = "60"
    elif change == "integrity":
        (root / "phase-0-result.json").write_text("{}")
    elif change == "budget":
        phase["phases"][0]["budget_exceeded"] = True
    elif change == "capture":
        phase["phases"][0]["kind"] = "capture"
    elif change == "accepted":
        report["accepted_csv_count"] = 1
    elif change == "unknown":
        report["downloads"][1]["result"]["error"] = "unclassified source error"
        report["downloads"][1]["result"].pop("source_failure_category")
    elif change == "stop":
        gate = SportsbetAccess(plan["sportsbet_access_state"])
        value = gate.read()
        value["phase"] = "STOP"
        run.atomic_json(gate.path, value)
    elif change == "ordinary":
        plan.pop("operational_predictions")
    run.atomic_json(refresh, report)
    run.atomic_json(root / "phase-checkpoint.json", phase)
    assert not run.record_refresh_outage(output, plan, rid, set())
    assert not (output / "refresh-deferrals").exists()


@pytest.mark.parametrize("minute", [2, 25])
@pytest.mark.parametrize("terminal_present", [False, True])
@pytest.mark.parametrize("age,status,continued", [(114, "CAPTURE_FAILED", True), (270, "CAPTURE_FAILED", False), (114, "INTEGRITY_FAILED", False), (114, "ACTIVE", True)])
def test_actual_observer_keeps_failed_lane_visible_and_requires_fresh_index(tmp_path, monkeypatch, age, status, continued, minute, terminal_present):
    output, plan, rid, root, refresh = outage(tmp_path)
    start = datetime(2099, 1, 1, tzinfo=timezone.utc)
    plan.update(starts_at=start.isoformat(), ends_at=(start + timedelta(hours=1)).isoformat(),
        first_index_deadline_seconds=180, readiness_warmup_seconds=1200, sample_period_seconds=0)
    monkeypatch.setattr(run, "now", lambda: start + timedelta(minutes=minute))
    if terminal_present:
        run.atomic_json(root / "terminal-timing.json", {
            "runtime_action": "LIVE_PHASE_FAILED", "timing": {"lock_wait_seconds": 0},
        })
    scope = SimpleNamespace(session=tmp_path / "session", campaign=None)
    monkeypatch.setattr(run, "AttemptAllowance", lambda scope: SimpleNamespace(claims=lambda: []))
    monkeypatch.setattr(run, "window_accounting", lambda *args: {})
    import race_collection.freshness_rehearsal as policy
    monkeypatch.setattr(policy, "TimerAccounting", lambda start: SimpleNamespace(observe=lambda current: None, summary=lambda now: {}))
    event_dir = root.parent / "shadow_autopilot_daemon_runtime/live-publication-events"
    run.atomic_json(event_dir / "000000.json", {"previous_event_sha256": None, "packet_sha256": "packet"})
    value = {"index_status": "AVAILABLE/FRESH", "authority_status": "AVAILABLE/FRESH",
        "collector_status": ("INVALID/INTEGRITY_FAILED" if status == "INTEGRITY_FAILED" else
                             "AVAILABLE/FRESH" if status == "ACTIVE" else "UNAVAILABLE/DATA_MISSING"),
        "source_age_seconds": age,
        "source_at": (start + timedelta(minutes=minute, seconds=-age)).isoformat(),
        "packet_sha256": "packet", "external_service_overhead_seconds": {},
        "lanes": [{"status": status, "run_id": "replacement_odds_capture" if status == "ACTIVE" else rid},
                  {"status": "RECEIPT_READY", "run_id": "full"}]}
    monkeypatch.setattr(run, "sample", lambda *args: value.copy())
    class AfterFirstSample(Exception):
        pass
    ticks = []
    def tick():
        ticks.append(1)
        if len(ticks) == 2:
            raise AfterFirstSample
    with pytest.raises(AfterFirstSample if continued else ValueError):
        run.observe(output, plan, None, scope, predictions=SimpleNamespace(tick=tick))
    sample = json.loads((output / "samples/000000.json").read_text())
    assert sample["collector_status"] == value["collector_status"]
    assert sample["source_age_seconds"] == age
    if continued:
        progress = json.loads((output / "progress.json").read_text())
        assert progress["completed_cycles"] == {"full": 0, "odds": 0}
        expected_failures = [rid] if terminal_present or status == "CAPTURE_FAILED" else []
        assert progress["failed_refresh_cycles"] == expected_failures
        deferrals = list((output / "refresh-deferrals").glob("*.json"))
        assert len(deferrals) == len(expected_failures)
        if expected_failures:
            assert json.loads(deferrals[0].read_text())["failed_cycle_count"] == 1
            # A replacement activation still points at the old index and must
            # not turn the retained terminal failure into proven recovery.
            assert not run.refresh_recovery_proven(output, sample, set(expected_failures))


def test_recovery_requires_new_successful_index_not_merely_active_lane(tmp_path, monkeypatch):
    output, plan, rid, root, refresh = outage(tmp_path)
    observed = datetime(2099, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(run, "now", lambda: observed)
    failed = set()
    assert run.record_refresh_outage(output, plan, rid, failed)
    current = {"collector_status": "AVAILABLE/FRESH", "source_at": (observed - timedelta(seconds=30)).isoformat()}
    assert not run.refresh_recovery_proven(output, current, failed)
    current["source_at"] = (observed + timedelta(seconds=60)).isoformat()
    assert run.refresh_recovery_proven(output, current, failed)
    current["collector_status"] = "UNAVAILABLE/DATA_MISSING"
    assert not run.refresh_recovery_proven(output, current, failed)
