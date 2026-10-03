"""Real generated service/refresh subprocesses, fabricated HTTP, denied network."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tests.test_freshness_campaign import make_campaign
from tests.test_refresh_shared_sportsbet_snapshot import fixture, access

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scenario", ["recovery", "cap", "untyped", "guidance", "denial", "no_index",
                                     "discovery_recovery", "discovery_cap", "discovery_no_index", "empty_close", "empty_future", "mixed_recovery", "mixed_cap"])
def test_actual_service_preserves_bounded_scheduled_refresh_recovery(tmp_path, monkeypatch, scenario):
    from scripts import prepare_freshness_rehearsal as packaging
    from scripts.check_freshness_service import service_command
    from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest

    gate = access(tmp_path)
    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(gate))
    http = fixture(tmp_path, 2 if scenario.startswith("mixed_") else 1)
    good = json.loads(http.read_text())
    installed = tmp_path / "installed"
    installed.mkdir()
    for name in (*packaging.UNITS, "greyhound-operator-ui-r3.service"):
        (installed / name).write_text("synthetic original " + name)
    db = tmp_path / "empty-history.sqlite"
    sqlite3.connect(db).close()
    campaign = make_campaign(tmp_path / "campaign")
    stamp = datetime.now(ZoneInfo("Australia/Melbourne"))
    package = tmp_path / "package"
    packaging.prepare(output=package, start=stamp-timedelta(seconds=5), python=Path(sys.executable),
        db=db, lock=tmp_path / "collector.lock", reconciliation_roots={}, installed_dir=installed,
        campaign_root=campaign.root, operational_predictions=True, observation_minutes=60)
    plan = json.loads((package / "plan.json").read_text())
    accounting = dict(schema_version="freshness_attempt_reconciliation_v1", complete=True,
                      consumed=[], sources=[{"sha256": "a"*64}])
    keys = ("profile", "rehearsal_id", "starts_at", "ends_at", "lock_path", "evidence_root", "db_path",
            "cleanup_seconds", "max_capture_attempts", "max_logical_requests", "source_identity_sha256",
            "runtime_sha256", "campaign_root", "campaign_authorization_sha256", "operational_predictions")
    contract = {key: plan[key] for key in keys}
    contract.update(schema_version="freshness_rehearsal_contract_v1", source_date=stamp.date().isoformat(),
                    reconciliation_sha256=digest(accounting))
    (package / "contract.json").write_text(json.dumps(contract))
    scope = FreshnessContract(contract)
    AttemptAllowance(scope).initialize(accounting)
    campaign.begin(plan["rehearsal_id"], now=stamp, deadline=stamp+timedelta(minutes=100))
    command, cwd, env = service_command(package / "units/shadow-autopilot-odds-capture.service")
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT / "tests/fixtures/shared_snapshot_transport"), str(package / "source"))),
               GREYHOUND_SHARED_SNAPSHOT_FIXTURE=str(http), LIVE_ENHANCE_LIMIT="0")
    launcher = "from scripts.check_freshness_service import deny_network; import os,sys; deny_network(); os.execv(sys.argv[1],sys.argv[1:])"
    evidence = Path(plan["evidence_root"])
    def invoke(number):
        run_id = f"fixture{number}_odds_capture"
        result = subprocess.run([sys.executable, "-c", launcher, *command, "--run-id", run_id],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=100)
        (tmp_path / f"service{number}.log").write_text(result.stdout + result.stderr)
        path = evidence / ("shadow_autopilot_daemonization_v1_" + run_id) / "terminal-timing.json"
        return json.loads(path.read_text()) if path.exists() else None

    if not scenario.endswith("no_index"):
        seeded = invoke(0)
        assert seeded and seeded["runtime_action"] == "LIVE_COLLECTION_COMPLETE", (tmp_path / "service0.log").read_text()[-2000:]
        publications = evidence / "shadow_autopilot_daemon_runtime/live-publication-events"
        old_events = len(list(publications.glob("*.json")))
    bad = json.loads(json.dumps(good))
    for url, row in bad["responses"].items():
        if scenario.startswith("empty_"):
            import re
            jump = (stamp + timedelta(minutes=5 if scenario == "empty_close" else 80)).strftime("%H:%M")
            row["body"] = re.sub(r'(<formatted-time data-format="time_24">)[^<]+',
                                 lambda match: match[1] + jump, row.get("body", ""))
        elif scenario.startswith("discovery_"):
            if url == "www.thedogs.com.au/racing/" + stamp.date().isoformat():
                row["error_type"] = "ConnectionError"
        elif url.endswith("export.csv"):
            if scenario.startswith("mixed_") and "/2/" in url:
                continue
            row.update(status=200 if scenario == "untyped" else 429 if scenario == "denial" else 502,
                       body="not a csv", headers={"Retry-After": "60"} if scenario == "guidance" else {})
    http.write_text(json.dumps(bad))
    traffic_before_failure = len((tmp_path / "transport.jsonl").read_text().splitlines()) if (tmp_path / "transport.jsonl").exists() else 0
    failed = invoke(1)
    if scenario.startswith("empty_"):
        assert failed and failed["runtime_action"] == "LIVE_COLLECTION_COMPLETE"
        assert not (scope.session / "STOP.json").exists()
        assert not AttemptAllowance(scope).claims()
        index = evidence / "shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json"
        assert json.loads(index.read_bytes())["races"] == []
        assert len(list(publications.glob("*.json"))) > old_events
        traffic = [json.loads(line) for line in (tmp_path / "transport.jsonl").read_text().splitlines()][traffic_before_failure:]
        assert not any(row["path"].endswith("export.csv") for row in traffic)
        return
    expected_action = "LIVE_COLLECTION_BLOCKED" if scenario.endswith("no_index") else "LIVE_PHASE_FAILED"
    assert failed and failed["runtime_action"] == expected_action
    assert failed["status"] == "FAILED"
    assert not AttemptAllowance(scope).claims()
    if scenario.startswith("discovery_"):
        # Exactly one failed discovery request; no hidden request retry or
        # cached-race download is created by the recovery classifier.
        assert len((tmp_path / "transport.jsonl").read_text().splitlines()) == traffic_before_failure + 1
        report = json.loads((evidence / "shadow_autopilot_v1_fixture1_odds_capture_phase_0/odds_capture_refresh_report.json").read_bytes())
        assert report["status"] == "DISCOVERY_FAILED" and report["current_index_race_count"] == 0
        if scenario == "discovery_no_index":
            assert not (evidence / "shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json").exists()
            assert failed["current_race_index_publish"]["failure_detail"]["reason"] == "refresh_not_accepted_success"
    if scenario in {"untyped", "guidance", "denial", "no_index", "discovery_no_index"}:
        assert (scope.session / "STOP.json").exists()
        assert not list((package / "refresh-deferrals").glob("*.json"))
        return
    assert not (scope.session / "STOP.json").exists()
    records = list((package / "refresh-deferrals").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["status"] == "FAILED_REFRESH_AWAITING_NORMAL_TIMER"
    if scenario.startswith("mixed_"):
        report = json.loads((evidence / "shadow_autopilot_v1_fixture1_odds_capture_phase_0/odds_capture_refresh_report.json").read_bytes())
        assert report["status"] == "ACQUISITION_INCOMPLETE"
        assert report["selected_count"] == 2 and report["accepted_csv_count"] == 1
        assert len(list(publications.glob("*.json"))) == old_events
    if scenario in {"recovery", "discovery_recovery", "mixed_recovery"}:
        http.write_text(json.dumps(good))
        recovered = invoke(2)
        assert recovered and recovered["runtime_action"] == "LIVE_COLLECTION_COMPLETE"
        assert len(list(publications.glob("*.json"))) > old_events
        latest = json.loads(sorted(publications.glob("*.json"))[-1].read_text())
        observed = json.loads(records[0].read_text())["observed_at"]
        assert datetime.fromisoformat(latest["source_generated_at"]) > datetime.fromisoformat(observed)
        assert not (scope.session / "STOP.json").exists()
    else:
        second = invoke(2)
        assert second and second["runtime_action"] == "LIVE_PHASE_FAILED"
        assert len(list((package / "refresh-deferrals").glob("*.json"))) == 2
        traffic_before = (tmp_path / "transport.jsonl").read_bytes()
        invoke(3)
        assert (scope.session / "STOP.json").exists()
        assert (tmp_path / "transport.jsonl").read_bytes() == traffic_before
