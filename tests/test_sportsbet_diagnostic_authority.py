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
