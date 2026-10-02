"""Successor-only writes using invented retained bodies and real native validators."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3

import pytest

from tests.test_engineering_result_renewal import renewal
from tests.test_reconcile_comparison_result_identity import case, put, ref


def successor_value(renewal_ref, approval, now, jobs):
    return dict(schema_version='renewed_engineering_identity_successor_v1',
        status='AUTHORIZED_RETAINED_IDENTITY_SUCCESSOR', authority_reference='SYNTHETIC_SUCCESSOR',
        issued_at=now.isoformat(), expires_at=(now+timedelta(minutes=30)).isoformat(),
        renewed_authority=renewal_ref, renewed_authority_approval=approval,
        network_requests_allowed=False, max_additional_requests=0, outcomes_released=False,
        preserve_originals=True, jobs=jobs)


def test_exact_renewal_allows_expired_original_metadata_without_reviving_it(renewal, tmp_path):
    from scripts.reconcile_renewed_engineering_identity import load_scope
    from src.predictor.comparison_result_runtime import load_runtime
    path, renewed, binding, old_ref, cfg = renewal
    now = datetime.fromisoformat('2026-10-02T13:00:00+10:00')
    old_binding = json.loads(Path(binding['path']).read_bytes())
    before = {p:Path(p).read_bytes() for p in (binding['path'],old_ref['path'])}
    with pytest.raises(ValueError):
        load_runtime(old_binding, now=now)
    value = successor_value(ref(path),'SYNTHETIC_NEW',now,renewed['jobs'])
    receipt = put(tmp_path/'successor.json',value)
    loaded = load_scope(receipt,'SYNTHETIC_SUCCESSOR',now=now)
    assert loaded[-1]['expires_at'] == cfg['expires_at']
    assert all(Path(p).read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize('mutation',['absent_renewal','wrong_approval','unlisted_job','unlisted_race',
    'duplicate_job','expanded_deadline','network','requests','changed_renewal_hash'])
def test_renewal_membership_and_limits_fail_closed(renewal,tmp_path,mutation):
    from scripts.reconcile_renewed_engineering_identity import load_scope
    path, renewed, *_ = renewal
    now = datetime.fromisoformat('2026-10-02T13:00:00+10:00')
    value = successor_value(ref(path),'SYNTHETIC_NEW',now,[dict(renewed['jobs'][0])])
    if mutation=='absent_renewal':del value['renewed_authority']
    elif mutation=='wrong_approval':value['renewed_authority_approval']='NOT_AUTHORIZED'
    elif mutation=='unlisted_job':value['jobs'][0]['job_id']='unlisted'
    elif mutation=='unlisted_race':value['jobs'][0]['race_id']='unlisted'
    elif mutation=='duplicate_job':value['jobs']*=2
    elif mutation=='expanded_deadline':value['expires_at']='2026-10-04T12:00:01+11:00'
    elif mutation=='network':value['network_requests_allowed']=True
    elif mutation=='requests':value['max_additional_requests']=1
    else:value['renewed_authority']['sha256']='0'*64
    with pytest.raises((ValueError,KeyError)):
        load_scope(put(tmp_path/'successor.json',value),'SYNTHETIC_SUCCESSOR',now=now)


@pytest.fixture
def successor(case, tmp_path, monkeypatch):
    import scripts.reconcile_renewed_engineering_identity as module
    import scripts.reconcile_comparison_result_identity as original_module
    # The separate load_scope tests above exercise the real renewed engineering
    # validator. This integration fixture reuses an existing native sealed job;
    # only its already-tested renewal admission boundary is supplied here.
    monkeypatch.setattr(module,'datetime',original_module.datetime)
    binding=json.loads(Path(case['authority']['binding']['path']).read_bytes())
    plan=json.loads(Path(binding['plan']).read_bytes())
    original=json.loads(Path(binding['authority']).read_bytes())
    renewal_value=dict(issued_at=case['now'].isoformat(),
        expires_at=(case['now']+timedelta(hours=2)).isoformat(),
        original_binding=case['authority']['binding'],
        jobs=[dict(job_id=case['job'].job_id,race_id=case['job'].input.race_id)])
    renewal_ref=put(tmp_path/'renewal.json',renewal_value)
    monkeypatch.setattr(module,'load_renewal',lambda *args,**kwargs:
        (renewal_value,binding,plan,original,case['runtime']))
    closure=case['root']/'closure';closure.mkdir()
    put(closure/'closure.json',dict(status='SEALED_ORIGINAL',quarantines=1))
    (closure/'official-results.sqlite3').write_bytes((case['root']/'official-results.sqlite3').read_bytes())
    member={key:case['authority'][key] for key in ('job_id','race_id','attempt_directory','body','request','response','failed_report')}
    value=successor_value(renewal_ref,'SYNTHETIC_NEW',case['now'],[member])
    value.update(source_commit=case['authority']['source_commit'],output_directory=str(tmp_path/'successor'))
    paths=[Path(renewal_ref['path']),Path(case['authority']['binding']['path']),
        Path(binding['authority']),Path(binding['plan']),case['root']/'queue.sqlite3',
        case['root']/'official-results.sqlite3',Path(case['runtime']['campaign_root'])/'ledger.json',
        Path(case['runtime']['source_state']),*closure.iterdir()]
    value['preserved_originals']=[ref(path) for path in paths]
    path=tmp_path/'successor-authority.json';put(path,value)
    return case,path,value,{str(p):p.read_bytes() for p in paths}


def execute(successor):
    from scripts.reconcile_renewed_engineering_identity import reconcile
    case,path,value,_=successor
    put(path,value)
    return reconcile(ref(path),'SYNTHETIC_SUCCESSOR',now=case['now'])


def test_native_successor_closure_preserves_original_queue_database_and_closure(successor,monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session,'request',lambda *a,**k:pytest.fail('network forbidden'))
    case,path,value,before=successor
    result=execute(successor)
    assert result['closed']==1 and result['provider_requests']==0
    assert result['rows'][0]['state']=='IDENTITY_VERIFIED_CLOSED_IN_SUCCESSOR'
    assert result['rows'][0]['original_state']=='QUARANTINED'
    assert all(Path(p).read_bytes()==data for p,data in before.items())
    with sqlite3.connect(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone()==('QUARANTINED',1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0]==1
    assert (Path(value['output_directory'])/'closure.json').is_file()
    with pytest.raises((ValueError,FileExistsError)):
        execute(successor)


@pytest.mark.parametrize('mutation',['original_output','missing_closure_pin','changed_original','body_mismatch','transport_denial'])
def test_successor_refuses_changed_originals_or_unverified_retained_evidence(successor,mutation):
    case,path,value,before=successor
    if mutation=='original_output':value['output_directory']=str(case['root']/'successor')
    elif mutation=='missing_closure_pin':value['preserved_originals']=[r for r in value['preserved_originals'] if not r['path'].endswith('/closure/closure.json')]
    elif mutation=='changed_original':value['preserved_originals'][0]['sha256']='0'*64
    elif mutation=='body_mismatch':value['jobs'][0]['body']['sha256']='0'*64
    else:
        response_path=Path(value['jobs'][0]['response']['path'])
        response=json.loads(response_path.read_bytes());response['status']=429
        value['jobs'][0]['response']=put(response_path,response)
    with pytest.raises(ValueError):execute(successor)
    assert all(Path(p).read_bytes()==data for p,data in before.items())
    assert not (Path(value['output_directory'])/'closure.json').exists()
