"""Scheduling/authority boundaries; systemd and providers never invoked."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from race_collection.live_freshness_contract import digest
from src.predictor.on_demand import canonical_bytes
from scripts import run_comparison_schedule as schedule


def config(tmp_path,monkeypatch,slots):
    cfg={'state_root':str(tmp_path/'state'),'programme_id':'invented','slots':slots,
         'authority_reference':'SYNTHETIC','python':'unused','source_commit':'unused'}
    plan={'ends_at':(datetime.now(timezone.utc)+timedelta(days=112)).isoformat()}
    monkeypatch.setattr(schedule,'load_config',lambda _: (cfg,plan))
    return cfg


def test_missed_slots_are_recorded_once_no_makeup(tmp_path,monkeypatch):
    cfg=config(tmp_path,monkeypatch,[(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()])
    monkeypatch.setattr(schedule,'renew_source',lambda *a,**k:pytest.fail('missed slot must not renew'))
    assert schedule.tick(tmp_path/'config')['status']=='NO_SLOT_DUE'
    terminal=Path(cfg['state_root'])/'slots/001/terminal.json';before=terminal.read_bytes()
    assert json.loads(before)['status']=='MISSED_SLOT'
    schedule.tick(tmp_path/'config')
    assert terminal.read_bytes()==before


def test_restore_owed_even_when_admission_paused_and_already_restored(tmp_path,monkeypatch):
    cfg=config(tmp_path,monkeypatch,[])
    root=Path(cfg['state_root']);claim=root/'slots/001';package=claim/'invented-001';package.mkdir(parents=True)
    for name,value in [('plan.json',{'source_root':str(tmp_path),'rehearsal_id':'invented-001'}),('restoration.json',{}),('restored.json',{})]:
        (package/name).write_bytes(canonical_bytes(value))
    (root/'PAUSE_ADMISSIONS').touch()
    calls=[]
    def child(command,log):calls.append(command);return 0
    monkeypatch.setattr(schedule,'child',child)
    assert schedule.tick(tmp_path/'config')['status']=='ADMISSIONS_PAUSED'
    assert len(calls)==1 and '--restore-only' in calls[0]
    assert json.loads((claim/'terminal.json').read_bytes())['status']=='INTERRUPTED_CONSUMED'
    schedule.tick(tmp_path/'config');assert len(calls)==1


def test_unknown_reboot_restoration_holds_new_admission(tmp_path,monkeypatch):
    cfg=config(tmp_path,monkeypatch,[])
    package=Path(cfg['state_root'])/'slots/001/invented-001';package.mkdir(parents=True)
    (package/'plan.json').write_bytes(canonical_bytes({'source_root':str(tmp_path)}))
    (package/'restoration.json').write_bytes(b'{}')
    monkeypatch.setattr(schedule,'child',lambda *a:2)
    assert schedule.tick(tmp_path/'config')['status']=='RESTORATION_HELD'
    assert not (package.parent/'terminal.json').exists()


def test_open_only_renewal_preserves_denials_and_consumed_slot(tmp_path,monkeypatch):
    from race_collection.freshness_campaign import Campaign
    from utils.sportsbet_access import SportsbetAccess
    root=tmp_path/'campaign';root.mkdir()
    initial={'schema_version':'collector_engineering_campaign_v1','campaign_id':'SYNTHETIC',
             'max_capture_attempts':12,'max_logical_requests':48000,'max_live_seconds':10800}
    (root/'authorization.json').write_bytes(canonical_bytes(initial))
    (root/'ledger.json').write_bytes(canonical_bytes({'campaign_id':'SYNTHETIC','launches':{},'attempts':[],'logical_requests':0}))
    now=datetime.now(timezone.utc)
    programme={'schema_version':'collector_persistent_programme_v1','status':'AUTHORIZED_PERSISTENT_PROGRAMME',
        'campaign_id':'SYNTHETIC','programme_id':'SYNTHETIC','authority_reference':'SYNTHETIC',
        'prior_effective_authorization_sha256':digest(initial),'starts_at':(now-timedelta(hours=1)).isoformat(),
        'expires_at':(now+timedelta(days=126)).isoformat(),'max_capture_attempts':1000,
        'max_logical_requests':1304000,'max_live_seconds':580800,
        'initial_counters':{'capture_attempts':0,'logical_requests':0,'live_seconds':0}}
    ap=root/'persistent-programme-authority.json';ap.write_bytes(canonical_bytes(programme))
    access=SportsbetAccess(tmp_path/'source.json');access.initialize(access_basis={'status':'permitted','reference':'SYNTHETIC'})
    # Freeze the existing operating policy as activation does.
    access.authorize_diagnostic(reference='SYNTHETIC_OLD',expected_sha256=__import__('hashlib').sha256(access.path.read_bytes()).hexdigest(),
                                expires_at=now.timestamp()+600,max_operations=1,rationale='SYNTHETIC')
    value=access.read()
    cfg={'campaign_root':str(root),'programme_authority_sha256':__import__('hashlib').sha256(ap.read_bytes()).hexdigest(),
         'source_state':str(access.path),'lock_path':str(tmp_path/'collector.lock'),'authority_reference':'NEW_SYNTHETIC',
         'slots':[1,2],'max_source_operations':384,
         'source_baseline':{'denials_sha256':digest(value['denials']),'recovery_attempts':0,
           'access_basis_sha256':digest(value['access_basis']),'operating_policy_sha256':digest(value['operating_policy']),'operation_count':0}}
    schedule.renew_source(cfg,'1',now=now)
    assert len(access.read()['diagnostic_authorizations'])==2
    with pytest.raises(ValueError,match='already_consumed'):schedule.renew_source(cfg,'1',now=now)
    access.retain_denial(403,reason='SYNTHETIC')
    with pytest.raises(ValueError,match='explicit_disposition'):schedule.renew_source(cfg,'2',now=now)
    assert access.read()['phase']=='STOP' and len(access.read()['denials'])==1

# Reuse environment stubs, but execute the real package preparer and the exact
# supervisor contract constructor. This caught the NVMe path and Path/JSON bugs.
from tests.test_short_operational_observation import prepared


def test_real_prepare_and_supervisor_contract_accept_prospective_root(prepared,monkeypatch):
    from scripts.prepare_freshness_rehearsal import prepare
    from scripts.run_freshness_rehearsal import execution_contract
    from race_collection.live_freshness_contract import FreshnessContract
    from race_collection.freshness_campaign import Campaign
    campaign=Campaign(prepared['campaign_root'])
    prediction=prepared['output'].parent/'new-nvme-predictions'
    campaign.programme={'prediction_root':str(prediction)}
    # This test isolates root propagation; cumulative accounting is covered by
    # native Campaign tests, so its existing fake provides the read-only seam.
    for method in ('development_usage', 'incident_usage', 'persistent_usage', 'programme_usage'):
        setattr(campaign, method, lambda _: {'capture_attempts': 0})
    roots={k:[str(prepared['output'].parent)] for k in ('scheduled_progress','scheduled_reports','phase_checkpoints','prior_rehearsals','manual_claims','manual_attempts')}
    prepared['reconciliation_roots']=roots
    prepare(**prepared,prediction_root=prediction)
    plan=json.loads((prepared['output']/'plan.json').read_bytes())
    assert plan['reconciliation_roots']==roots
    assert plan['max_logical_requests']==16000
    contract=execution_contract(plan,{'source_date':'2026-09-24'})
    scope=FreshnessContract(contract)
    assert scope.value['db_path']==str(prediction/'capture.sqlite3')
    assert scope.value['prediction_root']==str(prediction)
