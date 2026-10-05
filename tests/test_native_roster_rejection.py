"""Authenticated contradictory source rosters exclude only that race."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest
from scripts import capture_thedogs_market_history as capture
from scripts import refresh_prejump_upcoming as refresh
from tests.test_mixed_completed_field_exclusions import mixed_case
from tests.test_empty_eligible_refresh import publish
from utils.native_roster_rejection import verify_native_roster_rejection
from utils.runner_completeness import extract_canonical_runner_set_from_html


def roster_case(tmp_path, monkeypatch):
    report, worker, old_ref = mixed_case(tmp_path)
    candidate = report['selected_races'][0]
    original_url = candidate['race_url']
    new_url = 'https://www.thedogs.com.au/racing/meadows/2026-07-19/6/test-race'
    def replace_url(value):
        if isinstance(value, dict):
            return {k: replace_url(v) for k,v in value.items()}
        if isinstance(value, list):
            return [replace_url(v) for v in value]
        return value.replace(original_url, new_url) if isinstance(value, str) else value
    report = replace_url(report)
    candidate = report['selected_races'][0]
    normal = report['downloads'][0]['result']['normalization']
    observed = datetime.fromisoformat(report['generated_at'])
    jump = datetime.fromisoformat(candidate['jump_datetime'])
    body = (worker/old_ref['raw_path']).read_bytes()
    body += f'<formatted-time data-format="datetime_short" data-timestamp="{int(jump.timestamp())}"></formatted-time>'.encode()
    for key in ('raw_path', 'receipt_path'):
        (worker/old_ref[key]).unlink()
    headers = {'content-type':'text/html', 'date':format_datetime(observed.astimezone(timezone.utc), usegmt=True), 'set-cookie':'never retain'}
    primary = capture.TimedResponse(requested_url=candidate['race_url'],final_url=candidate['race_url'],
        request_start_utc=observed-timedelta(seconds=3),request_end_utc=observed-timedelta(seconds=2),
        status_code=200,headers=headers,body=body)
    canonical = extract_canonical_runner_set_from_html(body.decode(),source_url=candidate['race_url'],
        expected_race_number=candidate['race_number'],extraction_timestamp=primary.request_end_utc.isoformat())
    capture.persist_primary_race_page_evidence(artifact_root=worker,race_discovery_key=candidate['race_id'],
        response=primary,canonical_runner_set=canonical)
    from bs4 import BeautifulSoup
    odds_soup = BeautifulSoup(body, 'html.parser')
    for row in odds_soup.select('tr.race-runner'):
        native_id = row.select_one('runner-odd')['data-runner-id']
        wrapper = odds_soup.new_tag('tbody', attrs={'data-content-url':f'/dogs/runner/{native_id}/odds'})
        row.wrap(wrapper)
    odds_body = str(odds_soup).encode().replace(b'159004', b'159999')
    odds = capture.TimedResponse(requested_url=candidate['race_url']+'/odds',final_url=candidate['race_url']+'/odds',
        request_start_utc=observed-timedelta(seconds=1),request_end_utc=observed,
        status_code=200,headers=headers,body=odds_body)
    calls=[]
    def get(session,url,**kwargs):
        calls.append(url)
        return odds
    monkeypatch.setattr(capture,'timed_get',get)
    with pytest.raises(capture.CaptureError,match='expected_native_runner_set_mismatch') as caught:
        capture.capture_native_identity_from_retained_race_page(session=object(),race_page=primary,
            expected_active_runner_boxes=[(str(159000+i),i) for i in range(1,5)],
            expected_jump_utc=jump,current_time=observed,rejection_artifact_root=worker)
    assert calls==[candidate['race_url']+'/odds']  # No native odds API call after contradiction.
    reference=caught.value.native_roster_rejection
    normal['native_roster_rejection']=reference
    canonical.update(native_identity_status='unavailable', source_native_race_id=None,
        native_identity_reasons=['native_identity_evidence_rejected:expected_native_runner_set_mismatch'])
    from utils.runner_completeness import align_csv_text_to_canonical_final_runner_set
    _, normal['canonical_runner_alignment'] = align_csv_text_to_canonical_final_runner_set(
        Path(normal['raw_export_path']).read_text(), canonical, source=normal['accepted_csv_path'])
    assert 'never retain' not in Path(reference['path']).read_text()
    return report,worker,reference


def test_producer_retains_contradiction_and_publishes_only_other_verified_race(tmp_path,monkeypatch):
    report,worker,reference=roster_case(tmp_path,monkeypatch)
    verify_native_roster_rejection(worker,reference,report['selected_races'][0])
    assert refresh.complete_mixed_field_exclusions(report)
    assert not refresh.has_unisolated_refresh_failure(report)
    result=publish(tmp_path,tmp_path/'state.json',report,'prospective')
    assert result['status']=='PUBLISHED' and result['race_count']==1
    assert report['selected_count']==2 and report['quarantine_count']==1
    assert report['current_index_races'][0]['race_id']==report['selected_races'][1]['race_id']


@pytest.mark.parametrize('defect',['missing_original_odds','tampered','denial','shared','expected_set','retry_after'])
def test_unproved_or_shared_failure_never_becomes_local_exclusion(tmp_path,monkeypatch,defect):
    report,worker,reference=roster_case(tmp_path,monkeypatch)
    normal=report['downloads'][0]['result']['normalization']
    if defect=='missing_original_odds':normal.pop('native_roster_rejection')
    elif defect=='tampered':
        Path(reference['path']).chmod(0o600);Path(reference['path']).write_bytes(b'changed')
    elif defect=='denial':report['downloads'][0]['result']['source_http_status']=429
    elif defect=='shared':report['shared_sportsbet_snapshot']['status']='FAILED'
    else:
        value=json.loads(Path(reference['path']).read_bytes())
        if defect=='expected_set':value['expected_active_runner_boxes']['159004']=8
        else:value['odds']['headers']['retry-after']='60'
        raw=capture.canonical_json_bytes(value);digest=hashlib.sha256(raw).hexdigest()
        path=Path(reference['path']).with_name(digest+'.json');path.write_bytes(raw)
        normal['native_roster_rejection']={**reference,'path':str(path),'sha256':digest}
    assert not refresh.complete_mixed_field_exclusions(report)
    assert refresh.has_unisolated_refresh_failure(report)
    assert publish(tmp_path,tmp_path/'state.json',report,'failed')['status']=='REJECTED'
