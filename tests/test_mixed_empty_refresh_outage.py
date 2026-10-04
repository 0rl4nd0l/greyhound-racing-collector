"""Zero accepted CSVs do not hide an authenticated local exclusion beside 502s."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from race_collection.live_freshness_contract import classify_refresh_outage
from race_collection import persistent_native as native
from scripts import run_freshness_rehearsal as run
from tests.test_race_local_quarantine_refresh import mixed_report, reselect
from tests.test_empty_eligible_refresh import publish
from tests.test_scheduled_refresh_outage import outage
from tests.test_persistent_recovery_terminal import terminal_recovery
from tests.test_persistent_recovery import recovery
from tests.test_persistent_native import backend
from tests.fixtures.persistent_operation_case import put


def empty_mixed(root):
    report = mixed_report(root)
    for key in ('selected_races', 'downloads'):
        report[key] = report[key][1:]
    report['sidecar_metadata_coverage']['races'] = report['sidecar_metadata_coverage']['races'][1:]
    first = report['selected_races'][0]
    for number in range(7, 10):
        candidate = dict(first, race_url=f'https://www.thedogs.com.au/racing/gunnedah/2026-07-19/{number}',
                         race_id=f'Race {number} - GUNN - 2026-07-19', race_number=number)
        report['selected_races'].append(candidate)
        report['downloads'].append(dict(race_url=candidate['race_url'], success=False,
            result=dict(success=False, source_http_status=502, error='Source HTTP status 502')))
        report['sidecar_metadata_coverage']['races'].append(dict(race_url=candidate['race_url'],
            race_id=candidate['race_id'], csv_path=None, sidecar_path=None,
            weather_track_rejected_reasons=['accepted_csv_missing']))
    report.update(status='METADATA_COVERAGE_INCOMPLETE', reason='no_selected_race_csv_sidecars',
        selected_count=4, accepted_csv_count=0, sidecar_count=0,
        shared_sportsbet_snapshot={'status':'VALIDATED','payload_sha256':'a'*64})
    reselect(report)
    return report


def retained(tmp_path):
    output, plan, rid, root, path = outage(tmp_path)
    report = empty_mixed(tmp_path)
    path.write_text(json.dumps(report))
    return output, plan, rid, root, path, report


def test_three502_and_local_quarantine_wait_without_success_or_rewriting_evidence(tmp_path):
    output, plan, rid, root, path, report = retained(tmp_path)
    before = path.read_bytes()
    classified = classify_refresh_outage(plan['evidence_root'], rid)
    assert classified and classified['upstream_statuses'] == [502, 502, 502]
    assert classified['request_retries_added'] == 0
    assert publish(tmp_path, tmp_path/'runtime/index.json', report, 'mixed')['status'] == 'REJECTED'
    failures = set()
    assert run.record_refresh_outage(output, plan, rid, failures)
    assert run.record_refresh_outage(output, plan, rid, failures)
    record = json.loads((output/'refresh-deferrals'/(rid+'.json')).read_bytes())
    assert record['failed_cycle_count'] == 1
    assert failures == {rid} and path.read_bytes() == before


@pytest.mark.parametrize('defect', ['403','429','500','unknown','untyped','retry','reset','header',
    'challenge','budget','snapshot','missing_candidate','partial_csv','false_accepted','exposed_index',
    'local_raw_tampered','local_identity','local_unknown','reason','phase_hash','phase_budget'])
def test_mixed_empty_does_not_excuse_denial_unknown_or_incomplete_proof(tmp_path, defect):
    output, plan, rid, root, path, report = retained(tmp_path)
    item = report['downloads'][1]['result']
    if defect.isdigit(): item.update(source_http_status=int(defect), error=f'Source HTTP status {defect}')
    elif defect=='unknown': item['error']='unknown'
    elif defect=='untyped': item.pop('source_http_status')
    elif defect=='retry': item['source_retry_after']='60'
    elif defect=='reset': item['source_rate_limit_reset']='60'
    elif defect=='header': item['source_retry_headers']={'Retry-After':'60'}
    elif defect in ('challenge','budget'): item['source_failure_category']=defect
    elif defect=='snapshot': report['shared_sportsbet_snapshot']['status']='DENIED'
    elif defect=='missing_candidate': report['selected_races'].pop()
    elif defect=='partial_csv': report['sidecar_metadata_coverage']['races'][1]['csv_path']='partial.csv'
    elif defect=='false_accepted': report['accepted_csv_count']=1
    elif defect=='exposed_index': report['current_index_races']=[report['selected_races'][1]]
    elif defect=='local_raw_tampered': Path(report['downloads'][0]['result']['raw_export_path']).write_text('changed')
    elif defect=='local_identity': report['downloads'][0]['result']['normalization']['canonical_runner_alignment']['native_identity_status']='unavailable'
    elif defect=='local_unknown': report['downloads'][0]['result']['error']='unknown'
    elif defect=='reason': report['reason']='unknown'
    else:
        cp=json.loads((root/'phase-checkpoint.json').read_bytes())
        cp['phases'][0]['result_sha256' if defect=='phase_hash' else 'budget_exceeded']='0'*64 if defect=='phase_hash' else True
        (root/'phase-checkpoint.json').write_text(json.dumps(cp))
    path.write_text(json.dumps(report))
    assert classify_refresh_outage(plan['evidence_root'], rid) is None
    assert not run.record_refresh_outage(output, plan, rid, set())


def test_reviewed_empty_mixed_recovery_keeps_same_grant_and_consumes_one_cycle(terminal_recovery, tmp_path):
    case, review = terminal_recovery
    cfg, standing, old, now, item, terminal, calls = case
    selection = native.checked(cfg['recovery_selection'])
    report = empty_mixed(tmp_path)
    path = Path(review['refresh_report']['path']).with_name('refresh_prejump_report.json')
    review['refresh_report'] = put(path, report)
    phase = native.checked(review['phase_result'])
    phase['current_race_index_publish']['source_refresh_report_path'] = str(path)
    review['phase_result'] = put(Path(review['phase_result']['path']), phase)
    cp = native.checked(review['checkpoint']); cp['phases'][0]['result_sha256'] = review['phase_result']['sha256']
    review['checkpoint'] = put(Path(review['checkpoint']['path']), cp)
    service = native.checked(review['service_terminal']); service['at'] = now.isoformat()
    review['service_terminal'] = put(Path(review['service_terminal']['path']), service)
    classified = classify_refresh_outage(old['plan']['evidence_root'], service['run_id'])
    assert classified is not None
    review['classified_outage'] = put(Path(old['output'])/'classified-outage.json', classified)
    review['disposition'] = 'PROSPECTIVE_TYPED_UPSTREAM_OUTAGE_RECOVERY'
    selection['reviewed_failure'] = put(Path(selection['reviewed_failure']['path']), review)
    cfg['recovery_selection'] = put(Path(cfg['recovery_selection']['path']), selection)
    paths = [Path(cfg['campaign_root'])/'ledger.json', Path(cfg['source_state']), Path(old['output'])/'HALT.json', path]
    before = {p:p.read_bytes() for p in paths}
    new = native.prepare_day(cfg, standing, '2026-10-03', now)
    assert new['allocation_ref'] == old['allocation_ref']
    assert new['plan']['prediction_root'] == old['plan']['prediction_root']
    records = list((Path(new['output'])/'refresh-deferrals').glob('*.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_bytes())
    assert record['failed_cycle_count'] == 1 and record['upstream_statuses'] == [502,502,502]
    assert {p:p.read_bytes() for p in paths} == before
    assert native.prepare_day(cfg, standing, '2026-10-03', now) == new
