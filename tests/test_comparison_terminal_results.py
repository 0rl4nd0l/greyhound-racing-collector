"""Known non-finish evidence is terminal without becoming a complete order."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.reconcile_comparison_result_identity import RetainedDeadline
from src.predictor.comparison_terminal_results import known_nonfinish_evidence


@pytest.fixture
def terminal_case(tmp_path):
    jump = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
    captured = jump + timedelta(minutes=20)
    now = captured + timedelta(minutes=1)
    url = 'https://www.thedogs.com.au/racing/richmond/2026-10-02/1/invented'
    job = SimpleNamespace(job_id='synthetic-job', input=SimpleNamespace(
        race_id='synthetic-race', jump_timestamp=jump.isoformat(), ordered_runners=[
            {'box': 1, 'name': 'Alpha', 'source_native_runner_id': '101'},
            {'box': 2, 'name': 'Beta', 'source_native_runner_id': '102'}]))
    bundle = SimpleNamespace(directory='sealed', manifest={'files': {}}, result={
        'race': {'race_id': 'synthetic-race', 'url': url, 'race_date': '2026-10-02',
                 'venue': 'Richmond', 'race_number': 1},
        'generated_at': (jump-timedelta(minutes=9)).isoformat()})
    body = (b'<table class="race-runners--result">'
            b'<tr class="race-runner"><td class="race-runners__finish-position">1st</td>'
            b'<td class="race-runners__box"><sprite-svg name="rug_1"></sprite-svg></td>'
            b'<td class="race-runners__name"><a href="/dogs/501/alpha" data-dog-id="501">Alpha</a></td></tr>'
            b'<tr class="race-runner"><td class="race-runners__finish-position">DNF</td>'
            b'<td class="race-runners__box"><sprite-svg name="rug_2"></sprite-svg></td>'
            b'<td class="race-runners__name"><a href="/dogs/502/beta" data-dog-id="502">Beta</a></td></tr></table>')
    def invoke(body=body, *, url=url, captured=captured, job=job, bundle=bundle,
               mutate=None, deadline=None):
        body_ref = tmp_path/'request.body'; body_ref.write_bytes(body)
        request = {'at': captured.isoformat(), 'url': url}
        response = {'observed_at': captured.isoformat(), 'final_url': url, 'status': 200,
                    'host': 'www.thedogs.com.au', 'retry_headers': {}, 'content_type': 'text/html',
                    'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}
        if mutate: mutate(request, response)
        refs = {}
        for name, value in [('request', request), ('response', response)]:
            path = tmp_path/(name+'.json'); path.write_text(json.dumps(value))
            refs[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        refs['body'] = {'path': str(body_ref), 'sha256': hashlib.sha256(body).hexdigest()}
        return known_nonfinish_evidence(job, bundle, body, url, captured, now,
            deadline=deadline or RetainedDeadline({'expires_at': '2030-10-04T01:00:00+00:00'}, None),
            prediction_bundles=tmp_path, source_evidence=refs)
    return invoke, body, job, bundle


def test_explicit_nonfinish_is_known_without_inventing_a_position(terminal_case):
    invoke, _, _, _ = terminal_case
    record = invoke()
    assert record['state'] == 'RESULT_KNOWN_NON_FINISH'
    assert record['result_known'] is True and record['identity_verified'] is True
    assert record['full_order_eligible'] is False
    assert record['runner_results'][1]['finish_position'] is None
    assert record['runner_results'][1]['terminal_status'] == 'DNF'
    assert 'race_rows' not in record and 'runner_rows' not in record
    assert record['outcomes_released'] is False


@pytest.mark.parametrize('marker', [b'FELL', b'DNF', b'DISQ'])
def test_only_recognized_nonfinish_taxonomy(terminal_case, marker):
    invoke, body, _, _ = terminal_case
    assert invoke(body.replace(b'>DNF<', b'>'+marker+b'<'))['state'] == 'RESULT_KNOWN_NON_FINISH'


@pytest.mark.parametrize('marker', [b'', b'F', b'UNKNOWN', b'SCR', b'LSCR', b'L/SCR', b'2nd'])
def test_absent_unknown_nonstarter_or_complete_order_is_not_this_state(terminal_case, marker):
    invoke, body, _, _ = terminal_case
    with pytest.raises(ValueError):
        invoke(body.replace(b'>DNF<', b'>'+marker+b'<'))


@pytest.mark.parametrize('mutation', ['name', 'duplicate', 'gap', 'foreign_url', 'before_jump',
                                     'request_url', 'response_hash', 'response_status', 'future_capture'])
def test_wrong_identity_or_provenance_never_becomes_known(terminal_case, mutation):
    invoke, body, _, _ = terminal_case
    kwargs = {}
    if mutation == 'name': body = body.replace(b'>Beta<', b'>Wrong<')
    elif mutation == 'duplicate': body = body.replace(b'rug_2', b'rug_1')
    elif mutation == 'gap': body = body.replace(b'>1st<', b'>2nd<')
    elif mutation == 'foreign_url': kwargs['url'] = 'https://example.org/racing/richmond/2026-10-02/1/invented'
    elif mutation == 'before_jump': kwargs['captured'] = datetime(2026,10,2,8,0,tzinfo=timezone.utc)
    elif mutation == 'future_capture': kwargs['captured'] = datetime(2026,10,2,10,0,tzinfo=timezone.utc)
    elif mutation == 'request_url': kwargs['mutate'] = lambda req,res: req.update(url='https://example.org/')
    elif mutation == 'response_hash': kwargs['mutate'] = lambda req,res: res.update(sha256='0'*64)
    else: kwargs['mutate'] = lambda req,res: res.update(status=429)
    with pytest.raises(ValueError): invoke(body, **kwargs)


def test_expired_private_read_authority_is_not_bypassed(terminal_case):
    invoke, _, _, _ = terminal_case
    with pytest.raises(ValueError, match='RESULT_DEADLINE_EXPIRED'):
        invoke(deadline=RetainedDeadline({'expires_at':'2020-01-01T00:00:00+00:00'},None))
