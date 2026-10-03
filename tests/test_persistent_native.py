"""Daily backend lifecycle with native authority/reconciliation and no network."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import pytest

from race_collection import persistent_native as native
from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import create_once, digest
from race_collection.persistent_authority import stamp
from tests.fixtures.persistent_operation_case import make_persistent, put


@pytest.fixture
def backend(tmp_path,monkeypatch):
    standing,ref,_,_=make_persistent(tmp_path/'authority')
    study=json.loads(Path(standing['study_plan']['path']).read_bytes())
    study.update(schema_version='frozen_four_way_comparison_plan_v2',decision_seconds_before_jump=120,
        quote_lead_seconds=[120,600],denied_history_intervals=[],history_policy='strictly_earlier_machine_features_only',
        machine_history_authority_reference='synthetic',fixed_closure_days=14,missing_result_policy='bounded_paired_losses_v1')
    standing['study_plan']=put(Path(standing['study_plan']['path']),study)
    standing['daily_caps']=dict(max_python_requests=500000,max_browser_navigations=1000,
        max_capture_attempts=500,max_source_operations=4000,max_result_requests=0)
    ref=put(Path(ref['path']),standing)
    root=tmp_path/'campaign';put(root/'authorization.json',dict(schema_version='collector_engineering_campaign_v1',
        campaign_id='SYNTHETIC',max_capture_attempts=12,max_logical_requests=48000,max_live_seconds=10800))
    put(root/'ledger.json',dict(campaign_id='SYNTHETIC',attempts=[],launches={},logical_requests=0))
    history=tmp_path/'history.sqlite3'
    with sqlite3.connect(history) as db:db.execute('CREATE TABLE live_odds(race_id TEXT,capture_mode TEXT)')
    prior=tmp_path/'prior';prior.mkdir()
    manual=tmp_path/'manual'
    for sub in ('claims','attempts','requests'):(manual/sub).mkdir(parents=True,exist_ok=True)
    roots={k:[str(manual if k.startswith('manual') else prior)] for k in
           ('scheduled_progress','scheduled_reports','phase_checkpoints','prior_rehearsals','manual_claims','manual_attempts')}
    cfg=dict(python=sys.executable,history_database=str(history),lock_path=str(tmp_path/'collector.lock'),
        reconciliation_roots=roots,installed_dir=str(tmp_path/'units'),campaign_root=str(root),source_state=str(tmp_path/'source.json'))
    calls=[]
    def prepare(**kw):
        calls.append(kw);out=kw['output'];out.mkdir()
        source=out/'source';source.mkdir();(source/'fixture.py').write_text('# immutable synthetic package\n')
        identity=dict(commit='f'*40,files={'fixture.py':hashlib.sha256((source/'fixture.py').read_bytes()).hexdigest()})
        create_once(source/'SOURCE_IDENTITY.json',identity)
        (out/'source.tar').write_bytes(b'synthetic archive')
        (out/'units').mkdir();(out/'units/fixture.service').write_text('synthetic unit')
        allocation=json.loads(Path(kw['persistent_allocation']['path']).read_bytes())
        c=Campaign(kw['campaign_root'],persistent_allocation=kw['persistent_allocation'])
        plan=dict(schema_version='freshness_scheduled_rehearsal_plan_v1',profile='bounded80-v1',
            rehearsal_id=out.name,starts_at=allocation['starts_at'],ends_at=allocation['ends_at'],cleanup_seconds=1860,
            lock_path=str(kw['lock']),db_path=str(kw['prediction_root']/'capture.sqlite3'),evidence_root=str(out/'evidence'),
            max_capture_attempts=allocation['caps']['max_capture_attempts'],max_logical_requests=allocation['caps']['max_python_requests'],
            source_root=str(source),source_identity_sha256=digest(identity),commit='f'*40,
            source_archive_sha256=hashlib.sha256((out/'source.tar').read_bytes()).hexdigest(),
            python=sys.executable,python_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),runtime_sha256=digest({}),
            unit_sha256={'fixture.service':hashlib.sha256((out/'units/fixture.service').read_bytes()).hexdigest()},
            campaign_root=str(kw['campaign_root']),campaign_authorization_sha256=digest(c.value),
            persistent_allocation=kw['persistent_allocation'],racing_date=allocation['racing_date'],
            frozen_comparison=native._ref(kw['comparison_plan']),prediction_root=str(kw['prediction_root']),
            operational_predictions=dict(capture_db_path=str(kw['prediction_root']/'capture.sqlite3'),
                history_db_path=str(kw['db']),result_access=False,research_activation=False))
        create_once(out/'plan.json',plan)
        return dict(plan=str(out/'plan.json'),plan_sha256=digest(plan))
    monkeypatch.setattr(native,'prepare',prepare)
    monkeypatch.setattr('scripts.check_freshness_runtime.probe_runtime',lambda **kw:{})
    return cfg,standing,ref,calls


def run(backend,now='2026-10-03T12:00:00+10:00'):
    cfg,standing,ref,calls=backend
    return native.prepare_day(cfg,ref,'2026-10-03',stamp(now))


def test_prepare_and_restart_reuse_native_receipts_without_lease_or_requests(backend):
    cfg,standing,ref,calls=backend
    before=Path(cfg['campaign_root'],'ledger.json').read_bytes()
    first=run(backend)
    assert len(calls)==1
    assert stamp(first['plan']['starts_at'])==stamp('2026-10-03T12:02:00+10:00')
    assert stamp(first['plan']['ends_at'])==stamp('2026-10-04T01:20:00+10:00')
    assert first['plan']['rehearsal_id'].startswith('native-2026-10-03-')
    second=run(backend,'2026-10-03T13:00:00+10:00')
    assert first==second and len(calls)==1
    assert Path(cfg['campaign_root'],'ledger.json').read_bytes()==before
    assert not Path(cfg['lock_path']).exists() and not Path(cfg['source_state']).exists()


def test_partial_preparation_is_preserved_and_never_retried(backend,monkeypatch):
    cfg,standing,ref,calls=backend
    def broken(**kw):raise RuntimeError('synthetic package interruption')
    monkeypatch.setattr(native,'prepare',broken)
    with pytest.raises(RuntimeError,match='interruption'):run(backend)
    day=Path(standing['state_root'])/'days/2026-10-03'
    assert (day/'preparation-started.json').is_file() and (day/'allocation.json').is_file()
    with pytest.raises(ValueError,match='incomplete_preserved'):run(backend)
    assert not (day/'native-prepared.json').exists()


def test_foreign_collector_lock_is_not_stolen(backend):
    cfg,standing,ref,calls=backend
    Path(cfg['lock_path']).write_text('{"run_id":"foreign","pid":123}')
    before=Path(cfg['lock_path']).read_bytes()
    with pytest.raises(Exception):run(backend)
    assert Path(cfg['lock_path']).read_bytes()==before
    with pytest.raises(ValueError,match='incomplete_preserved'):run(backend)


def test_reconciliation_failure_releases_only_owned_lock(backend):
    cfg,standing,ref,calls=backend
    cfg['reconciliation_roots']['manual_claims']=[str(Path(cfg['history_database']).parent/'missing')]
    with pytest.raises(FileNotFoundError):run(backend)
    assert not Path(cfg['lock_path']).exists()
    with pytest.raises(ValueError,match='incomplete_preserved'):run(backend)


@pytest.mark.parametrize('change',['plan','contract','source','unit','configuration','reconciliation'])
def test_restart_rejects_modified_bindings_without_repreparing(backend,change):
    first=run(backend);cfg,standing,ref,calls=backend
    if change=='plan':Path(first['plan_path']).write_text('{}')
    if change=='contract':Path(first['contract_path']).write_text('{}')
    if change=='source':Path(first['plan']['source_root'],'fixture.py').write_text('# changed')
    if change=='unit':Path(first['output'],'units/fixture.service').write_text('changed')
    if change=='configuration':cfg['source_state']+='changed'
    if change=='reconciliation':
        receipt=json.loads(Path(first['receipt_ref']['path']).read_bytes())
        Path(receipt['reconciliation']['path']).write_text('{}')
    with pytest.raises(ValueError):run(backend)
    assert len(calls)==1


def test_date_or_last_minute_start_cannot_backdate_new_allocation(backend):
    cfg,standing,ref,calls=backend
    for now in ['2026-10-04T00:00:00+10:00','2026-10-03T23:59:00+10:00']:
        with pytest.raises(ValueError,match='source_date'):run(backend,now)
    assert not (Path(standing['state_root'])/'days').exists()


def test_after_midnight_restart_keeps_previous_source_day_allocation(backend):
    first=run(backend)
    resumed=run(backend,'2026-10-04T00:13:00+10:00')
    assert resumed==first and len(backend[3])==1
    assert resumed['contract']['source_date']=='2026-10-03'


def test_large_real_reconciliation_preserves_consumption_on_prepare_and_restart(backend):
    cfg,standing,ref,calls=backend
    identities=[f'fixture-{index:05d}-2026-10-03' for index in range(3000)]
    with sqlite3.connect(cfg['history_database']) as db:
        db.executemany('INSERT INTO live_odds VALUES (?,?)',
            [(identity,'autonomous_prejump_t2m') for identity in identities])
    before=Path(cfg['campaign_root'],'ledger.json').read_bytes()
    first=run(backend)
    receipt=json.loads(Path(first['receipt_ref']['path']).read_bytes())
    reconciliation=Path(receipt['reconciliation']['path'])
    retained=reconciliation.read_bytes()
    assert 262144<len(retained)<64*1024*1024
    accounting=json.loads(retained)
    assert accounting['complete'] is True
    assert {row['race_id'] for row in accounting['consumed']}==set(identities)
    assert all(row['capture_window_minutes']==2 for row in accounting['consumed'])
    assert len(accounting['consumed'])==len(identities)
    # Only native reconciliation may use the larger bound. Generic authority
    # reference validation must continue rejecting exactly these larger bytes.
    with pytest.raises(ValueError,match='persistent_reference_unsafe'):
        native.checked(receipt['reconciliation'])
    assert run(backend,'2026-10-03T13:00:00+10:00')==first
    assert len(calls)==1 and reconciliation.read_bytes()==retained
    assert Path(cfg['campaign_root'],'ledger.json').read_bytes()==before
    assert not Path(cfg['source_state']).exists() and not Path(cfg['lock_path']).exists()


def test_native_reconciliation_reader_keeps_path_hash_and_finite_size_checks(tmp_path):
    path=tmp_path/'reconciliation.json'
    path.write_text('{"fixture":true}')
    ref=native._ref(path)
    alias=tmp_path/'alias.json';alias.symlink_to(path)
    with pytest.raises(ValueError,match='persistent_reconciliation_reference_unsafe'):
        native._checked_reconciliation({**ref,'path':str(alias)})
    path.write_text('{"fixture":false}')
    with pytest.raises(ValueError,match='persistent_reconciliation_changed'):
        native._checked_reconciliation(ref)
    with path.open('wb') as stream:stream.truncate(64*1024*1024+1)
    with pytest.raises(ValueError,match='persistent_reconciliation_reference_unsafe'):
        native._checked_reconciliation(ref)
