"""Actual exported service, retention and frozen predictor; only source transport is invented."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
from datetime import timedelta

from tests.test_freshness_capture_e2e import fixture_data
from tests.test_refresh_shared_sportsbet_snapshot import fixture as http_fixture, access
from tests.test_freshness_campaign import make_campaign

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("landing_missing", [False, True, "paired_missing"])
def test_packaged_capture_retention_frozen_prediction(tmp_path, monkeypatch, landing_missing):
    from scripts.prepare_freshness_rehearsal import prepare, UNITS
    from scripts.check_freshness_service import service_command
    from sportsbet_odds_integrator import SportsbetOddsIntegrator
    from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest

    gate = access(tmp_path)
    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(gate))
    stamp, browser = fixture_data(tmp_path, "canonical_alias")
    from datetime import datetime
    operational_jump = (stamp + timedelta(minutes=9)).replace(second=0, microsecond=0)
    browser["sidecar"]["prejump_shadow_metadata"]["jump_time"] = operational_jump.isoformat()
    browser["race"]["race_time"] = operational_jump.strftime("%H:%M")
    browser["sidecar"]["race_info"]["race_time"] = operational_jump.strftime("%H:%M")
    http = http_fixture(tmp_path, 1)
    payload = json.loads(http.read_bytes())
    jump = browser['sidecar']['prejump_shadow_metadata']['jump_time']
    from datetime import datetime
    jump_dt = datetime.fromisoformat(jump)
    # The invented HTTP and browser observations refer to the same race/runners.
    responses = {}
    import re
    for key, value in payload['responses'].items():
        key = key.replace('/sale/', '/murray-bridge-straight/').replace('/1/invented', '/9/fabricated')
        body = value['body'].replace('/sale/', '/murray-bridge-straight/').replace('/1/invented', '/9/fabricated')
        body = body.replace('Race 1', 'Race 9').replace('>R1<', '>R9<')
        body = re.sub(r'(<formatted-time[^>]*>).*?(</formatted-time>)', r'\g<1>'+jump_dt.strftime('%H:%M')+r'\g<2>', body)
        if 'NextEvents' in key:
            events = json.loads(body)
            events[0].update(competitionName='Murray Bridge Straight', raceNumber=9, startTime=int(jump_dt.timestamp()))
            body = json.dumps(events)
        if 'open-meteo' in key:
            weather = json.loads(body)
            weather['hourly']['time'] = [jump_dt.strftime('%Y-%m-%dT%H:%M')]
            body = json.dumps(weather)
        if key.count('/') == 2 and '/racing/' in key:
            body = body.replace('</a>', '<formatted-time data-format="time_24">'+jump_dt.strftime('%H:%M')+'</formatted-time></a>')
        responses[key] = {**value, 'body': body}
    payload['responses'] = responses
    http.write_text(json.dumps(payload))
    for name in ('Alpha', 'Bravo', 'Charlie', 'Delta'):
        browser['race_html'] = browser['race_html'].replace(name, 'Synthetic '+name)
    if landing_missing == "paired_missing":
        browser["race_html"] = browser["race_html"].replace("<span>1.50</span><span>EW</span>", "")
    if landing_missing is True:
        browser['landing_html'] = '<a href="https://www.sportsbet.com.au/greyhound-racing/australia-nz/murray-bridge-straight">Murray Bridge Straight</a>'
    browser_path = tmp_path / 'browser.json'
    browser_path.write_text(json.dumps(browser))
    installed = tmp_path / 'installed'
    installed.mkdir()
    for name in (*UNITS, 'greyhound-operator-ui-r3.service'):
        (installed/name).write_text('synthetic original '+name)
    db = tmp_path/'history.sqlite'
    SportsbetOddsIntegrator(str(db), allow_auto_scrape_odds=False)
    with sqlite3.connect(db) as conn:
        conn.executescript('CREATE TABLE IF NOT EXISTS race_metadata(race_id TEXT,race_date TEXT,data_source TEXT,url TEXT); CREATE TABLE IF NOT EXISTS dog_race_data(race_id TEXT,dog_name TEXT,finish_position INTEGER,data_source TEXT);')
    campaign = make_campaign(tmp_path/'campaign')
    if not landing_missing:
        import hashlib
        from race_collection.live_freshness_contract import create_once
        create_once(campaign.root/'prospective-authorization-amendment.json', dict(
            schema_version='collector_engineering_amendment_v1', campaign_id='synthetic',
            prior_authorization_sha256=hashlib.sha256((campaign.root/'authorization.json').read_bytes()).hexdigest(),
            authority_reference='synthetic-explicit-user', rationale='sustained comparison', max_capture_attempts=64))
    package = tmp_path/'package'
    prepare(output=package, start=stamp-timedelta(seconds=5), python=Path(sys.executable), db=db,
        lock=tmp_path/'collector.lock', reconciliation_roots={}, installed_dir=installed,
        campaign_root=campaign.root, operational_predictions=True, observation_minutes=60)
    plan = json.loads((package/'plan.json').read_bytes())
    assert (datetime.fromisoformat(plan['ends_at']) - datetime.fromisoformat(plan['starts_at'])).total_seconds() == 3600
    assert plan['max_capture_attempts'] == (12 if landing_missing else 64)
    from race_collection.synchronous_manual_capture import _atomic_replace_canonical
    from race_collection.live_phase_checkpoint import atomic_json
    # The real supervisor replaces this sibling on every observation tick.
    # Force that overlap at the publisher's final retained-directory check.
    evidence = Path(plan['evidence_root'])
    _atomic_replace_canonical(evidence/'runtime/overlap-probe.json', {'synthetic': True},
        evidence_root=evidence,
        _pre_replace=lambda: atomic_json(package/'progress.json', {'synthetic_tick': 1}))
    assert Path(plan['db_path']).resolve() != db.resolve()
    with sqlite3.connect(db) as history_conn:
        initial_odds = history_conn.execute('SELECT count(*) FROM live_odds').fetchone()[0]
    accounting = dict(schema_version='freshness_attempt_reconciliation_v1',complete=True,consumed=[],sources=[{'sha256':'a'*64}])
    keys = ('profile','rehearsal_id','starts_at','ends_at','lock_path','evidence_root','db_path','cleanup_seconds',
        'max_capture_attempts','max_logical_requests','source_identity_sha256','runtime_sha256',
        'campaign_root','campaign_authorization_sha256','operational_predictions')
    contract = {k:plan[k] for k in keys}
    contract.update(schema_version='freshness_rehearsal_contract_v1',source_date=stamp.date().isoformat(),reconciliation_sha256=digest(accounting))
    (package/'contract.json').write_text(json.dumps(contract))
    scope = FreshnessContract(contract)
    allowance = AttemptAllowance(scope)
    allowance.initialize(accounting)
    campaign.begin(plan['rehearsal_id'],now=stamp,deadline=stamp+timedelta(minutes=110))
    command, cwd, env = service_command(package/'units/shadow-autopilot.service')
    env.update(PYTHONPATH=os.pathsep.join(str(p) for p in (ROOT/'tests/fixtures/freshness_transport',ROOT/'tests/fixtures/shared_snapshot_transport',package/'source')),
        FRESHNESS_FABRICATED_SOURCE=str(browser_path),GREYHOUND_SHARED_SNAPSHOT_FIXTURE=str(http))
    launcher = 'from scripts.check_freshness_service import deny_network; import os,sys; deny_network(); os.execv(sys.argv[1],sys.argv[1:])'
    service = subprocess.run([sys.executable,'-c',launcher,*command],cwd=cwd,env=env,capture_output=True,text=True,timeout=100)
    (tmp_path/'collector.log').write_text(service.stdout+service.stderr)
    inspection_path = allowance.claims()[0].with_suffix('.requests.responses.json')
    inspection = json.loads(inspection_path.read_bytes())
    assert inspection['operation_id'].startswith('browser:')
    assert any(mark['name'] == 'browser_ready' for mark in inspection['marks'])
    if landing_missing:
        assert service.returncode == 0, (tmp_path/"collector.log").read_text()[-1500:]
        assert not (scope.session/"STOP.json").exists()
        assert len(allowance.claims()) == 1
        terminal_capture=json.loads(allowance.claims()[0].with_suffix(".terminal.json").read_bytes())["result"]
        assert terminal_capture["operational_capture_outcome"]["status"] == "UNREADY_NO_CAPTURE"
        assert json.loads(gate.read_bytes())['phase'] == 'OPEN'
        expected_marker = ("target_race_not_visible_within_navigation_allowance" if landing_missing is True
                           else "required_paired_markets_not_ready_within_readiness_budget")
        assert terminal_capture["operational_capture_outcome"]["reason"] == expected_marker
        if landing_missing == "paired_missing":
            assert inspection["paired_readiness"][-1]["card_count"] == 4
            assert inspection["paired_readiness"][-1]["paired_card_count"] == 0
            assert any(m["name"] == "paired_readiness_expired" for m in inspection["marks"])
        assert not list((campaign.root/"operational-predictions/races").glob("*/terminal.json"))
        with sqlite3.connect(plan['db_path']) as capture_conn:
            assert capture_conn.execute('SELECT count(*) FROM live_odds').fetchone()[0] == 0
        return
    assert service.returncode == 0, (tmp_path/'collector.log').read_text()[-3000:]
    assert inspection['paired_readiness'][-1]['card_count'] == 4
    assert inspection['paired_readiness'][-1]['paired_card_count'] == 4
    claims = allowance.claims()
    assert len(claims) == 1
    result = subprocess.run([sys.executable,'-B','-m','race_collection.operational_prediction',str(package/'plan.json'),str(claims[0])],
        cwd=cwd,env=env,capture_output=True,text=True,timeout=180)
    (tmp_path/'prediction.log').write_text(result.stdout+result.stderr)
    terminals = list((campaign.root/'operational-predictions/races').glob('*/terminal.json'))
    assert len(terminals) == 1, result.stderr
    terminal = json.loads(terminals[0].read_bytes())
    with sqlite3.connect(db) as history_conn:
        assert history_conn.execute('SELECT count(*) FROM live_odds').fetchone()[0] == initial_odds
    assert terminal['status'] == 'PREDICTION_READY', terminal
    assert terminal['seconds_to_jump_at_verification'] > 60
    from src.operator_ui.job_store import JobStore
    store = JobStore(campaign.root/'operational-predictions/jobs.sqlite3',readonly=True)
    jobs = store.recorded_jobs()
    assert len(jobs) == 1 and jobs[0].attempt_claimed and jobs[0].operation == 'operational_prediction'
    # The real result nomination interface must reject operational jobs even if
    # a future operator accidentally points it at this store.
    from scripts.r3_official_result_candidates import r3_prediction_candidates
    candidates, skipped, _ = r3_prediction_candidates(job_store_path=store.path,
        prediction_bundles=campaign.root/'operational-predictions/bundles',result_database=db,
        target_date=stamp.date().isoformat(),current_time=jump_dt+timedelta(minutes=1),race_ids=[],output_dir=tmp_path/'unused')
    assert candidates == [] and skipped[0]['reason'] == 'OPERATIONAL_RESULT_ACCESS_FORBIDDEN'
    repeated = subprocess.run([sys.executable,'-B','-m','race_collection.operational_prediction',str(package/'plan.json'),str(claims[0])],
        cwd=cwd,env=env,capture_output=True,text=True,timeout=20)
    assert repeated.returncode != 0 and len(store.recorded_jobs()) == 1
