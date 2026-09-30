"""Actual exported pilot adapter under invented transports and a fixture-only future clock.

Reads approved allocation/control metadata only. All writes are in the labelled
synthetic output, and kernel network denial is inherited by collector children.
"""
from datetime import datetime,timedelta
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import pytest
from tests.test_freshness_capture_e2e import fixture_data
from tests.test_refresh_shared_sportsbet_snapshot import fixture as http_fixture, access
from tests.test_freshness_campaign import make_campaign
from scripts.prepare_freshness_rehearsal import prepare, UNITS
from sportsbet_odds_integrator import SportsbetOddsIntegrator
from race_collection.live_freshness_contract import digest
from race_collection.development_examples import put
ROOT=Path(__file__).resolve().parents[2]


def seed(tmp_path,monkeypatch,allocation):
    landing_missing=False;venue_case='murray'
    gate = access(tmp_path, previous_operations=0)
    if venue_case == "allocated_murray":
        import hashlib
        import time
        from utils.sportsbet_access import SportsbetAccess
        source_gate = SportsbetAccess(gate)
        prior_operations = source_gate.read()["operations"]
        source_gate.authorize_diagnostic(reference="synthetic finite continuation",
            expected_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
            expires_at=time.time()+3600, max_operations=94,
            rationale="Exercise exported collector and prediction above the legacy ceiling")
    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(gate))
    stamp, browser = fixture_data(tmp_path, "canonical_alias")
    if venue_case == "sandown_park":
        browser = json.loads(json.dumps(browser).replace("murray-bridge-straight", "sandown")
            .replace("MURRAY-BRIDGE-STRAIGHT", "SANDOWN").replace("Murray Bridge Straight", "Sandown Park")
            .replace('"MURR"', '"SAN"'))
        # The two normally observed providers use different venue spellings.
        browser = json.loads(json.dumps(browser).replace(
            "sportsbet.com.au/betting/greyhound-racing/australia-nz/sandown/",
            "sportsbet.com.au/betting/greyhound-racing/australia-nz/sandown-park/"))
    elif venue_case == "angle_park":
        browser = json.loads(json.dumps(browser).replace("murray-bridge-straight", "angle-park")
            .replace("MURRAY-BRIDGE-STRAIGHT", "AP_K").replace("Murray Bridge Straight", "Angle Park")
            .replace('"MURR"', '"AP_K"').replace('"race_number": 9', '"race_number": 7')
            .replace("Race 9", "Race 7").replace("R9", "R7")
            .replace("race-9-", "race-7-").replace("/9/fabricated", "/7/fabricated"))
    repaired_venues = {'maitland': ('Maitland', 'MAITLAND', 3),
                       'grafton': ('Grafton', 'GRAF', 4),
                       'launceston': ('Launceston', 'LCTN', 2)}
    if venue_case in repaired_venues:
        name, code, number = repaired_venues[venue_case]
        browser = json.loads(json.dumps(browser).replace('murray-bridge-straight', venue_case)
            .replace('MURRAY-BRIDGE-STRAIGHT', code).replace('Murray Bridge Straight', name)
            .replace('"MURR"', '"'+code+'"').replace('"race_number": 9', '"race_number": '+str(number))
            .replace('Race 9', f'Race {number}').replace('R9', f'R{number}')
            .replace('race-9-', f'race-{number}-').replace('/9/fabricated', f'/{number}/fabricated'))
    from datetime import datetime
    operational_jump = stamp.replace(hour=13,minute=10,second=0,microsecond=0)
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
    venue_slug = {'sandown_park': 'sandown', 'angle_park': 'angle-park'}.get(venue_case, 'murray-bridge-straight')
    venue_name = {'sandown_park': 'Sandown Park', 'angle_park': 'Angle Park'}.get(venue_case, 'Murray Bridge Straight')
    race_number = 7 if venue_case == 'angle_park' else 9
    if venue_case in repaired_venues:
        venue_slug = venue_case
        venue_name, _, race_number = repaired_venues[venue_case]
    for key, value in payload['responses'].items():
        key = key.replace('/sale/', '/' + venue_slug + '/').replace('/1/invented', f'/{race_number}/fabricated')
        body = value['body'].replace('/sale/', '/' + venue_slug + '/').replace('/1/invented', f'/{race_number}/fabricated')
        body = body.replace('Race 1', f'Race {race_number}').replace('>R1<', f'>R{race_number}<')
        body = re.sub(r'(<formatted-time[^>]*>).*?(</formatted-time>)', r'\g<1>'+jump_dt.strftime('%H:%M')+r'\g<2>', body)
        if 'NextEvents' in key:
            events = json.loads(body)
            events[0].update(competitionName=venue_name,
                             raceNumber=race_number, startTime=int(jump_dt.timestamp()))
            body = json.dumps(events)
        if 'open-meteo' in key:
            weather = json.loads(body)
            weather['hourly']['time'] = [jump_dt.strftime('%Y-%m-%dT%H:%M')]
            body = json.dumps(weather)
        if key.count('/') == 2 and '/racing/' in key:
            body = body.replace('</a>', '<formatted-time data-format="time_24">'+jump_dt.strftime('%H:%M')+'</formatted-time></a>')
        responses[key] = {**value, 'body': body}
        if 'open-meteo' in key and landing_missing == "weather_guidance":
            responses[key].update(status=503, headers={"Retry-After": "120"})
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
    authority = None
    if venue_case == 'preprogramme_murray':
        from race_collection.freshness_campaign import Campaign
        campaign = Campaign(campaign.root)
        create_once(campaign.root/'persistent-programme-authority.json', dict(
            schema_version='collector_persistent_programme_v1', status='AUTHORIZED_PERSISTENT_PROGRAMME',
            campaign_id='synthetic', programme_id='future-study', authority_reference='synthetic-study',
            prior_effective_authorization_sha256=digest(campaign.value),
            starts_at=(stamp+timedelta(days=2)).isoformat(), expires_at=(stamp+timedelta(days=126)).isoformat(),
            max_capture_attempts=1064, max_logical_requests=1352000, max_live_seconds=591600,
            initial_counters=dict(capture_attempts=0, logical_requests=0, live_seconds=0)))
        authority = 'synthetic-separate-operational'
        campaign = Campaign(campaign.root, engineering_authority=authority)
    from race_collection.development_pilot import reference
    from race_collection.development_source_authority import CAPS, DATES
    from race_collection.freshness_campaign import Campaign
    from race_collection.development_examples import put
    allocation_ref=reference(allocation)
    campaign=Campaign(campaign.root)
    runtime=tmp_path/'runtime';runtime.mkdir(mode=0o700)
    prediction_root=tmp_path/'operational-predictions'
    authority={'schema_version':'collector_development_pilot_authority_v1','status':'AUTHORIZED_DEVELOPMENT_PILOT',
        'campaign_id':campaign.value['campaign_id'],'allocation_id':'development-single-snapshot-20261003-v1',
        'authority_reference':'synthetic:pilot-runtime-export-only','allocation_sha256':allocation_ref['sha256'],
        'prior_effective_authorization_sha256':digest(campaign.value),'state_root':str(runtime),
        'prediction_root':str(prediction_root),'dates':DATES,**CAPS,
        'result_closure_at':'2026-10-25T12:00:00+11:00'}
    authority_path=campaign.root/'development-pilot-authority.json';put(authority_path,authority)
    roots={key:[str(tmp_path/'reconciliation'/key)] for key in ('scheduled_progress','scheduled_reports','manual_claims','manual_attempts','phase_checkpoints','prior_rehearsals')}
    for paths in roots.values():
        p=Path(paths[0]);p.mkdir(parents=True)
        for name in ('claims','attempts','requests'):(p/name).mkdir()
    roots_path=tmp_path/'roots.json';put(roots_path,roots)
    prepare(output=package, start=stamp.replace(hour=12,minute=40,second=0,microsecond=0),python=Path(sys.executable),db=db,
        lock=tmp_path/'collector.lock',reconciliation_roots=roots,installed_dir=installed,
        campaign_root=campaign.root,operational_predictions=True,observation_minutes=110,
        prediction_root=prediction_root,development_authority=reference(authority_path))
    plan=json.loads((package/'plan.json').read_bytes())
    source=json.loads(gate.read_bytes())
    config={'schema_version':'development_pilot_runtime_v1','status':'SYNTHETIC_FIXTURE',
        'authority_reference':authority['authority_reference'],'state_root':str(runtime),'campaign_root':str(campaign.root),
        'allocation':allocation_ref,'pilot_campaign_authority':reference(authority_path),
        'prediction_root':str(prediction_root),'python':str(Path(sys.executable)),'source_state':str(gate),
        'lock_path':str(tmp_path/'collector.lock'),'reconciliation_roots':reference(roots_path),
        'source_baseline':{'access_basis_sha256':digest(source['access_basis']),'denials_sha256':digest(source['denials']),
            'operating_policy_sha256':digest(source.get('operating_policy')),'recovery_attempts':source['recovery_attempts']},
        'session_packages':{'2026-10-03':reference(package/'plan.json')}}
    schedule=tmp_path/'study.json';put(schedule,{'slots':[],'session_minutes':90});config['study_schedule']=reference(schedule)
    return config,plan,http,browser_path,stamp


def demonstrate(output,allocation):
    from scripts.check_freshness_service import deny_network
    deny_network()
    from race_collection.development_pilot import Collector,reference,selected_population
    from race_collection.development_examples import seal,join_result,verify_package
    from tests.fixtures.development_pipeline import prepare_result
    started=time.monotonic()
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not os.environ.get('SYNTHETIC_DEVELOPMENT_CLOCK'):raise ValueError('fixture_clock_required')
    def clock(at):
        Path(os.environ['SYNTHETIC_DEVELOPMENT_CLOCK']).write_text(json.dumps({'synthetic':True,'at':at,'monotonic':time.monotonic()}))
    def health(config,at):
        for name,status in [('study','NO_SLOT_DUE'),('result','CYCLE_COMPLETE')]:
            p=output/(name+'-'+at[11:16].replace(':','')+'.json')
            put(p,{'status':status,'at':at,'counts':{},'oldest_due':None,'synthetic':True})
            config['synthetic_'+('study' if name=='study' else 'result')+'_health']=reference(p)
    with pytest.MonkeyPatch.context() as patch:
        config,plan,http,browser,start=seed(output,patch,allocation)
        patch.setenv('PYTHONPATH',os.pathsep.join(str(p) for p in (ROOT/'tests/fixtures/development_pilot_transport',
            ROOT/'tests/fixtures/freshness_transport',ROOT/'tests/fixtures/shared_snapshot_transport',Path(plan['source_root']))))
        patch.setenv('FRESHNESS_FABRICATED_SOURCE',str(browser))
        patch.setenv('GREYHOUND_SHARED_SNAPSHOT_FIXTURE',str(http))
        clock('2026-10-03T12:40:05+10:00');health(config,'2026-10-03T12:40:05+10:00')
        collector=Collector(config)
        session=Path(config['state_root'])/'sessions/2026-10-03';session.mkdir(parents=True,mode=0o700)
        collector.prepare(session,start.replace(hour=12,minute=40,second=0,microsecond=0))
        clock('2026-10-03T12:48:00+10:00');health(config,'2026-10-03T12:48:00+10:00')
        collector.refresh(session,'preparation-refresh')
        clock('2026-10-03T12:50:00+10:00')
        collector.freeze(session)
        selected=selected_population(session,config['allocation']['sha256'])
        if len(selected)!=1:raise ValueError('fixture_expected_one_selected')
        clock('2026-10-03T12:59:00+10:00');health(config,'2026-10-03T12:59:00+10:00')
        collector.refresh(session,'race-refresh-0')
        clock('2026-10-03T13:00:05+10:00');health(config,'2026-10-03T13:00:05+10:00')
        claim=collector.capture(session,selected[0])
        verification=collector.predict(session,claim)
        # The same actual captured/predicted bundle enters the explicitly
        # synthetic development access seam. Never label invented data real.
        rid=selected[0]['race_id'];directory=output/'development-synthetic';directory.mkdir(mode=0o700)
        root=Path(config['prediction_root'])/'bundles'
        entry,=json.loads((root/'prediction_bundle_index_v1.json').read_bytes())['entries']
        put(directory/'registry.json',{'schema_version':'development_reservation_registry_v1','sources':[]})
        put(directory/'allocation.json',{'schema_version':'development_allocation_v1','status':'SYNTHETIC_FIXTURE',
            'authority_reference':'synthetic:offline-only','allocation_id':'synthetic_development','mode':'single_snapshot',
            'operational_owner':'primary_orchestrator','dates':['2026-10-03'],'starts_at':'2026-10-03T00:00:00+10:00',
            'ends_at':'2026-10-04T00:00:00+10:00','max_capture_attempts':1,'max_attempts_per_date':1,
            'reservation_registry':reference(directory/'registry.json'),'denied_history_intervals':[]})
        put(directory/'opportunities.json',{'schema_version':'development_opportunities_v1','complete':True,
            'opportunities':[{'race_id':rid,'race_key':selected[0]['race_key'],'qualified':True,'attempt_consumed':True,
                'disposition':'VERIFIED_FORECAST','reason':'synthetic_actual_pilot_collector_verified'}]})
        put(directory/'access.json',{'schema_version':'development_access_v1','status':'SYNTHETIC_FIXTURE','enabled':True,
            'allocation':reference(directory/'allocation.json'),'opportunities':reference(directory/'opportunities.json'),
            'members':{rid:{'race_key':selected[0]['race_key'],'jump_at':selected[0]['jump_at'],'bundle_root':str(root),
                'entry':entry,'verification':reference(verification[1]),'history_access':'machine_only_pre_target',
                'target_label_access':'separate_result_authority'}}})
        access=reference(directory/'access.json');example=directory/'example'
        def cli(action,*extra):
            command=[sys.executable,'-B','-m','scripts.development_examples',action,'--access',access['path'],
                '--access-sha256',access['sha256'],'--race-id',rid,'--output',str(example),*extra]
            proc=subprocess.run(command,cwd=plan['source_root'],capture_output=True,text=True,timeout=45)
            (output/(action+'.log')).write_text(proc.stdout+proc.stderr)
            if proc.returncode:raise RuntimeError('synthetic_'+action+'_failed')
            return json.loads(proc.stdout)
        sealed=cli('seal')
        authority,pin=prepare_result(directory,example,rid)
        joined=cli('join','--result-authority',str(authority),'--result-authority-sha256',pin)
        verified=cli('verify')
        before=(example/'example.json').read_bytes()
        cli('join','--result-authority',str(authority),'--result-authority-sha256',pin)
        assert (example/'example.json').read_bytes()==before
        assert json.loads((example/'pre_result.json').read_bytes())['synthetic'] is True
        collector.close(session)
        ledger=json.loads((Path(config['campaign_root'])/'ledger.json').read_bytes())
        record={'status':'SYNTHETIC_ACTUAL_PILOT_ADAPTER_COMPLETE','synthetic':True,
            'source_commit':plan['commit'],'kernel_network':'denied','real_provider_operations':0,
            'protected_outcomes_read':0,'actual_refresh_only_phases':2,'actual_selected_capture_attempts':len(ledger['attempts']),
            'actual_prediction_status':verification[0]['status'],'seal':sealed,'join':joined,'verify':verified,
            'idempotent_join':True,'campaign_lease_closed':all(r.get('closed_at') for r in ledger['launches'].values()),
            'elapsed_seconds':time.monotonic()-started,'interpreter':sys.executable,
            'scope_note':'All source responses, browser pages, clocks and target labels are invented. Existing approved allocation metadata was read only for native freeze; final development access and examples are SYNTHETIC_FIXTURE.'}
        put(output/'demonstration.json',record)
        print(json.dumps(record,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allocation',type=Path,required=True)
    args=parser.parse_args();demonstrate(args.output,args.allocation)
