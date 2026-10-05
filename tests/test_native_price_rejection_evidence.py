"""Failure evidence is private, immutable and cannot admit a rejected race."""
import base64
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from scripts import capture_thedogs_market_history as capture
from tests.test_capture_thedogs_market_history import (
    FakeClock, FakeSession, JUMP, replacement_api_payload, replacement_source_html,
    retained_primary_race_page,
)


def rejected(tmp_path, *, session=None):
    observed=JUMP-timedelta(minutes=20)
    body=replacement_source_html(); api=replacement_api_payload(original_price=4.0)
    session=session or FakeSession(server_time=observed,source_body=body,api_body=api)
    primary=replace(retained_primary_race_page(observed),body=body,
        headers={**retained_primary_race_page(observed).headers,'set-cookie':'PRIVATE_COOKIE','authorization':'PRIVATE_AUTH'},
        request_headers={'authorization':'PRIVATE_REQUEST_AUTH'})
    with pytest.raises(capture.CaptureError,match='^scratched_runner_has_active_price$') as caught:
        capture.capture_native_identity_from_retained_race_page(session=session,race_page=primary,
            expected_active_runner_boxes=[('101',1),('104',2)],expected_jump_utc=JUMP,
            current_time=observed+timedelta(seconds=1),clock=FakeClock(observed+timedelta(milliseconds=100)),
            rejection_artifact_root=tmp_path)
    return caught.value,session,body,api


def test_actual_identity_failure_retains_exact_both_responses_without_new_requests(tmp_path):
    error,session,body,api=rejected(tmp_path)
    ref=error.native_price_rejection
    raw=Path(ref['path']).read_bytes();value=json.loads(raw)
    assert base64.b64decode(value['odds']['body_base64'])==body
    assert base64.b64decode(value['api']['body_base64'])==api
    assert len(session.calls)==2
    assert value['reason']=='scratched_runner_has_active_price'
    assert value['disposition']=='REJECTED_DIAGNOSTIC_ONLY_NOT_ADMISSIBLE'
    assert Path(ref['path']).stat().st_mode & 0o777 == 0o400
    assert Path(ref['path']).parent.stat().st_mode & 0o777 == 0o700
    for secret in (b'PRIVATE_COOKIE',b'PRIVATE_AUTH',b'PRIVATE_REQUEST_AUTH'):
        assert secret not in raw


def test_replay_category_and_idempotent_same_bytes(tmp_path):
    from utils.native_price_rejection import inspect_native_price_rejection
    error,_,_,_=rejected(tmp_path)
    ref=error.native_price_rejection
    before=Path(ref['path']).read_bytes()
    repeated,_,_,_=rejected(tmp_path)
    assert repeated.native_price_rejection==ref
    assert Path(ref['path']).read_bytes()==before
    summary=inspect_native_price_rejection(tmp_path,ref)
    assert summary['status']=='REJECTION_REPRODUCED_NOT_ADMISSIBLE'
    assert summary['additional_requests']==0
    assert 'price' not in summary and 'runners' not in summary


@pytest.mark.parametrize('defect',['body_limit','receipt_limit','symlink_directory','permissive_directory'])
def test_failed_retention_preserves_original_identity_error_and_request_count(tmp_path,monkeypatch,defect):
    from utils import native_price_rejection as evidence
    if defect=='body_limit':monkeypatch.setattr(evidence,'MAX_BODY',1)
    elif defect=='receipt_limit':monkeypatch.setattr(evidence,'MAX_RECEIPT',1)
    else:
        parent=tmp_path/'source_evidence';parent.mkdir()
        if defect=='symlink_directory':
            elsewhere=tmp_path/'elsewhere';elsewhere.mkdir();(parent/'native_price_rejections').symlink_to(elsewhere,target_is_directory=True)
        else:(parent/'native_price_rejections').mkdir(mode=0o755)
    error,session,_,_=rejected(tmp_path)
    assert not hasattr(error,'native_price_rejection')
    assert error.native_price_rejection_retention_status=='FAILED_NO_REJECTION_EVIDENCE'
    assert len(session.calls)==2
    assert not list(tmp_path.rglob('*.json'))


@pytest.mark.parametrize('defect',['changed_bytes','wrong_hash','wrong_path','symlink_file','hardlink','permissions'])
def test_inspection_rejects_tamper_or_unsafe_storage(tmp_path,defect):
    from utils.native_price_rejection import inspect_native_price_rejection
    error,_,_,_=rejected(tmp_path);ref=error.native_price_rejection;p=Path(ref['path'])
    if defect=='changed_bytes':p.chmod(0o600);p.write_bytes(p.read_bytes()+b' ');p.chmod(0o400)
    elif defect=='wrong_hash':ref={**ref,'sha256':'0'*64}
    elif defect=='wrong_path':ref={**ref,'path':str(tmp_path/p.name)}
    elif defect=='symlink_file':p.rename(p.with_suffix('.original'));p.symlink_to(p.with_suffix('.original'))
    elif defect=='hardlink':
        import os
        os.link(p,tmp_path/'second-link')
    else:p.chmod(0o644)
    with pytest.raises((ValueError,OSError)):inspect_native_price_rejection(tmp_path,ref)


def test_denial_and_unknown_failure_do_not_create_diagnostic_local_evidence(tmp_path,monkeypatch):
    observed=JUMP-timedelta(minutes=20)
    class Denial(FakeSession):
        def get(self,url,**kwargs):
            response=super().get(url,**kwargs)
            if '/api/' in url:response.status_code=429
            return response
    session=Denial(server_time=observed,source_body=replacement_source_html(),api_body=replacement_api_payload(original_price=4.0))
    primary=replace(retained_primary_race_page(observed),body=replacement_source_html())
    def invoke():
        return capture.capture_native_identity_from_retained_race_page(session=session,race_page=primary,
            expected_active_runner_boxes=[('101',1),('104',2)],expected_jump_utc=JUMP,
            current_time=observed+timedelta(seconds=1),clock=FakeClock(observed+timedelta(milliseconds=100)),
            rejection_artifact_root=tmp_path)
    with pytest.raises(capture.CaptureError,match='source_http_status_429') as caught:invoke()
    assert caught.value.source_http_status==429 and len(session.calls)==2
    assert not list(tmp_path.rglob('*.json'))
    session=FakeSession(server_time=observed,source_body=replacement_source_html(),api_body=replacement_api_payload(original_price=4.0))
    def unknown(*args):raise capture.CaptureError('shared_unknown_integrity_failure')
    monkeypatch.setattr(capture,'normalize_api_snapshot',unknown)
    with pytest.raises(capture.CaptureError,match='shared_unknown_integrity_failure'):invoke()
    assert not list(tmp_path.rglob('*.json'))


def test_diagnostic_reference_does_not_qualify_rejected_race(tmp_path):
    from tests.test_native_identity_exclusion_continuation import local_identity_case
    from tests.test_empty_eligible_refresh import publish
    from scripts import refresh_prejump_upcoming as refresh
    case=tmp_path/'case';case.mkdir();_,_,_,report,_=local_identity_case(case)
    evidence=tmp_path/'evidence';evidence.mkdir();error,_,_,_=rejected(evidence)
    report['downloads'][0]['result']['normalization']['native_price_rejection']=error.native_price_rejection
    assert refresh.complete_unavailable_metadata_selection(report)
    assert publish(case,case/'state.json',report,'diagnostic-only')['status']=='REJECTED'
    assert report['current_index_metadata_selection']['exclusions'][0]['missing_safe_metadata']==['native_source_identity']
