"""An exact reviewed publication overlap never becomes a generic PATH_UNSAFE bypass."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_collection import persistent_native as native
from tests.test_persistent_native import backend
from tests.test_persistent_recovery import recovery
from tests.fixtures.persistent_operation_case import put


@pytest.fixture
def publication_recovery(recovery, monkeypatch):
    cfg, standing, old, now, item, failed_terminal, calls = recovery
    selection = native.checked(cfg['recovery_selection'])
    cleanup = native.checked(selection['cleanup'])
    out = Path(old['output']); evidence = Path(old['plan']['evidence_root'])
    runtime = evidence/'shadow_autopilot_daemon_runtime'
    inv = 'a'*32; run = 'synthetic_odds_capture'
    t = lambda sec: f'2026-10-03T02:04:{sec:02d}+00:00'
    terminal = put(runtime/'service-terminals'/(inv+'.json'), dict(invocation_id=inv,
        allocation_sha256=old['allocation_ref']['sha256'],status='READY',
        runtime_action='LIVE_COLLECTION_COMPLETE',final_verdict='DAEMON_READY',
        run_id=run,output_dir=str(evidence/('shadow_autopilot_daemonization_v1_'+run)),at=t(22)))
    lifecycle = put(runtime/'service-lifecycles'/(inv+'.json'), dict(invocation_id=inv,
        status='COMPLETE',returncode=0,children_reaped=True,interrupted=False))
    owner = json.loads((out/'persistent-owner-state.json').read_bytes())
    owner['dispatches'] = [dict(invocation_id=inv,lane='odds',returncode=0,
        native_disposition='COMPLETED',started_at=t(10),completed_at=t(23))]
    put(out/'persistent-owner-state.json',owner)
    cleanup['halt'] = put(out/'HALT.json',dict(reason='CURRENT_INDEX_PATH_UNSAFE',at=t(24)))
    cleanup['at'] = t(25)
    selection['cleanup'] = put(Path(selection['cleanup']['path']),cleanup)
    selection['baseline'] = native.recovery_baseline(old,cfg)
    index = put(runtime/'manual_prediction_current_race_index.json',dict(run_id=run))
    os.utime(index['path'],(native.stamp(t(20)).timestamp(),native.stamp(t(20)).timestamp()))
    view = dict(run_id=run,packet_sha256=index['sha256'],
        source_refresh_report_path=str(Path('shadow_autopilot_v1_'+run+'_phase_0')/'odds_capture_refresh_report.json'),
        source_refresh_report_sha256='1'*64,publication_sha256='2'*64,state_sha256='3'*64,report_sha256='4'*64)
    monkeypatch.setattr(native,'_quiet_publication_view',lambda *a:SimpleNamespace(**view))
    review = dict(schema_version='persistent_reviewed_publication_failure_v1',
        disposition='PROSPECTIVE_OWNED_PUBLICATION_CORRECTION',authority_reference='SYNTHETIC_REVIEW',
        source_commit=selection['source_commit'],cleanup=selection['cleanup'],
        original_exception_detail='NOT_RETAINED',correlation_only=True,
        failure=put(out/'failure-synthetic.json',dict(at=t(21),failure_class='CaptureOneRejected',reason='CURRENT_INDEX_PATH_UNSAFE')),
        last_owner_health=put(out/'persistent-health.json',dict(at=t(19),status='ACTIVE_COLLECTION',children=['odds'])),
        service_terminal=terminal,service_lifecycle=lifecycle,index=index,
        index_mtime_ns=Path(index['path']).stat().st_mtime_ns,
        quiet_replay=put(out/'quiet-replay.json',dict(at=t(26),status='PASS_HISTORICAL_PUBLICATION_INTEGRITY_ONLY',
            index=index,source_evidence_root=str(evidence),verified_view=view)))
    selection['reviewed_failure'] = put(out/'reviewed-publication.json',review)
    cfg['recovery_selection'] = put(Path(cfg['recovery_selection']['path']),selection)
    return recovery,review,view


def rebind(case,review):
    cfg=case[0][0];selection=native.checked(cfg['recovery_selection'])
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)


def test_reviewed_overlap_reuses_grant_and_preserves_failure(publication_recovery):
    case,review,view=publication_recovery
    cfg,standing,old,now,item,failed_terminal,calls=case
    paths=[Path(cfg['campaign_root'])/'ledger.json',Path(cfg['source_state']),
           Path(old['output'])/'HALT.json',Path(review['failure']['path']),Path(failed_terminal['path'])]
    before=[p.read_bytes() for p in paths]
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    assert new['allocation_ref']==old['allocation_ref']
    assert [p.read_bytes() for p in paths]==before
    assert new['plan']['prediction_root']==old['plan']['prediction_root']
    assert native.prepare_day(cfg,standing,'2026-10-03',now)==new


@pytest.mark.parametrize('change',[
    'unreviewed','detail_claim','wrong_halt','wrong_stop','wrong_allocation','unowned',
    'unreaped','nonzero','outside_overlap','old_publication','changed_index',
    'fake_pass','changed_view','precleanup_replay','unsafe_replay', 'other_report_root'])
def test_publication_recovery_rejects_missing_or_changed_proof(publication_recovery,monkeypatch,change):
    case,review,view=publication_recovery
    cfg,standing,old,now,item,failed_terminal,calls=case
    selection=native.checked(cfg['recovery_selection'])
    if change=='unreviewed':selection.pop('reviewed_failure');cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    elif change=='detail_claim':review['original_exception_detail']='path_replaced'
    elif change in ('wrong_halt','wrong_stop'):
        cleanup=native.checked(selection['cleanup'])
        if change=='wrong_halt':
            cleanup['halt']=put(Path(cleanup['halt']['path']),{'reason':'other'})
            selection['cleanup']=put(Path(selection['cleanup']['path']),cleanup);review['cleanup']=selection['cleanup']
        else:selection['prior_stop']=put(Path(selection['prior_stop']['path']),{'reason':'other'})
        cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    elif change in ('wrong_allocation','unowned','nonzero','unreaped','outside_overlap'):
        key='service_lifecycle' if change in ('nonzero','unreaped') else 'service_terminal'
        row=native.checked(review[key])
        field,value={'wrong_allocation':('allocation_sha256','0'*64),'unowned':('invocation_id','b'*32),
            'nonzero':('returncode',2),'unreaped':('children_reaped',False),'outside_overlap':('at','2026-10-03T02:04:19+00:00')}[change]
        row[field]=value;review[key]=put(Path(review[key]['path']),row)
    elif change=='old_publication':
        p=Path(review['index']['path']);os.utime(p,(1,1));review['index_mtime_ns']=p.stat().st_mtime_ns
    elif change=='changed_index':Path(review['index']['path']).write_text('{}')
    elif change in ('fake_pass','precleanup_replay'):
        row=native.checked(review['quiet_replay']);row['status' if change=='fake_pass' else 'at']='PASS' if change=='fake_pass' else '2026-10-03T02:04:20+00:00'
        review['quiet_replay']=put(Path(review['quiet_replay']['path']),row)
    elif change=='changed_view':view['state_sha256']='5'*64
    elif change=='other_report_root':view['source_refresh_report_path']='/tmp/other/report.json'
    elif change=='unsafe_replay':
        from race_collection.synchronous_manual_capture import CaptureOneRejected
        def unsafe(*args):raise CaptureOneRejected('CURRENT_INDEX_PATH_UNSAFE')
        monkeypatch.setattr(native,'_quiet_publication_view',unsafe)
    if change!='unreviewed':rebind(publication_recovery,review)
    with pytest.raises(Exception):native.prepare_day(cfg,standing,'2026-10-03',now)
    assert len(calls)==1
