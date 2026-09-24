"""Prospective finite authorization preserves denial and consumption history."""
import hashlib,json
import pytest
from utils.sportsbet_access import SportsbetAccess,SportsbetAccessBlocked


def test_prospective_authority_keeps_prior_stop_and_honors_cooldown(tmp_path):
    now=[10000.0]
    gate=SportsbetAccess(tmp_path/'access.json',clock=lambda:now[0])
    gate.initialize(access_basis={'status':'permitted','reference':'synthetic'})
    value=gate.read();value.update(phase='STOP',recovery_attempts=1,not_before=10100,
        denials=[{'status':429,'retry_headers':{'retry-after':'100'}}],operations=[{'kind':'python','at':1}])
    gate.write(value)
    def authorize():
        gate.authorize_diagnostic(reference='synthetic-user-authority',
            expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
            expires_at=10500,max_operations=2,rationale='record exact resource on new candidate')
    with pytest.raises(SportsbetAccessBlocked): authorize()
    assert gate.read()==value
    now[0]=10101;authorize()
    reopened=gate.read()
    assert reopened['recovery_attempts']==1 and reopened['denials']==value['denials']
    assert reopened['operations']==value['operations']
    assert reopened['diagnostic_authorizations'][-1]['prior_phase']=='STOP'
    with gate.operation('python'): pass
    with gate.operation('browser'): pass
    with pytest.raises(SportsbetAccessBlocked):
        with gate.operation('python'): pytest.fail('exceeded finite sequence')
    assert len(gate.read()['operations'])==3


def test_new_denial_stops_diagnostic_without_automatic_recovery(tmp_path):
    gate=SportsbetAccess(tmp_path/'access.json',clock=lambda:10000)
    gate.initialize(access_basis={'status':'permitted','reference':'synthetic'})
    gate.authorize_diagnostic(reference='user',expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
        expires_at=10500,max_operations=2,rationale='new observation')
    with gate.operation('browser') as operation:
        operation.response(429,{'Retry-After':'9000'})
    assert gate.read()['phase']=='STOP'
    assert gate.read()['not_before']==19000
    with pytest.raises(SportsbetAccessBlocked):
        with gate.operation('python'): pytest.fail('recovery must be reassessed')


@pytest.mark.parametrize('limit', [0, True, 193, float('inf')])
def test_sustained_authority_has_finite_upper_bound(tmp_path, limit):
    gate=SportsbetAccess(tmp_path/'access.json',clock=lambda:10000)
    gate.initialize(access_basis={'status':'permitted','reference':'synthetic'})
    before=gate.path.read_bytes()
    with pytest.raises(ValueError, match='invalid_finite'):
        gate.authorize_diagnostic(reference='user',expected_sha256=hashlib.sha256(before).hexdigest(),
            expires_at=10500,max_operations=limit,rationale='sustained comparison')
    assert gate.path.read_bytes()==before
    gate.authorize_diagnostic(reference='user',expected_sha256=hashlib.sha256(before).hexdigest(),
        expires_at=10500,max_operations=192,rationale='sustained comparison')
    assert gate.read()['diagnostic_authority']['max_operations']==192


def test_explicit_quiet_revision_preserves_old_deadline_and_stops_on_next_denial(tmp_path):
    now=[10000.0]
    gate=SportsbetAccess(tmp_path/'access.json',clock=lambda:now[0])
    gate.initialize(access_basis={'status':'permitted','reference':'synthetic'})
    for _ in range(3):gate.retain_denial(429)
    prior=gate.read()
    def authorize():
        gate.authorize_diagnostic(reference='explicit-revised-policy',
            expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
            expires_at=now[0]+5400,max_operations=10,rationale='tested different candidate',
            engineering_quiet_seconds=2700)
    now[0]=12699
    with pytest.raises(SportsbetAccessBlocked):authorize()
    assert gate.read()==prior
    now[0]=12700;authorize()
    revised=gate.read()
    assert revised['not_before']==prior['not_before']==17200
    assert revised['denials']==prior['denials']
    assert revised['diagnostic_authority']['cooldown_revision']['effective_not_before']==12700
    with gate.operation('browser') as operation:operation.response(429,{})
    assert gate.read()['phase']=='STOP'


@pytest.mark.parametrize('headers', [{'Retry-After':'9999'}, {'Retry-After':'unparsed'}, {'RateLimit-Reset':'99999'}])
def test_engineering_revision_cannot_override_provider_guidance(tmp_path, headers):
    now=[10000.0]
    gate=SportsbetAccess(tmp_path/'access.json',clock=lambda:now[0])
    gate.initialize(access_basis={'status':'permitted','reference':'synthetic'})
    gate.retain_denial(429,headers)
    before=gate.path.read_bytes();now[0]=12700
    with pytest.raises(SportsbetAccessBlocked, match='provider_guidance'):
        gate.authorize_diagnostic(reference='user',expected_sha256=hashlib.sha256(before).hexdigest(),
            expires_at=18000,max_operations=10,rationale='new candidate',engineering_quiet_seconds=2700)
    assert gate.path.read_bytes()==before


@pytest.mark.parametrize('field', ['recorded_at_epoch','provider_not_before_epoch'])
@pytest.mark.parametrize('bad', [True, '123', float('nan'), float('inf')])
def test_quiet_revision_rejects_malformed_denial_clock(tmp_path, field, bad):
    gate=SportsbetAccess(tmp_path/'access.json',clock=lambda:10000)
    gate.initialize(access_basis={'status':'permitted','reference':'synthetic'})
    gate.retain_denial(429)
    value=gate.read();value['denials'][-1][field]=bad
    # Simulate a malformed external file; the normal writer rejects NaN/Inf.
    gate.path.write_text(json.dumps(value))
    before=gate.path.read_bytes()
    with pytest.raises(SportsbetAccessBlocked):
        gate.authorize_diagnostic(reference='user',expected_sha256=hashlib.sha256(before).hexdigest(),
            expires_at=18000,max_operations=10,rationale='new candidate',engineering_quiet_seconds=2700)
    assert gate.path.read_bytes()==before
