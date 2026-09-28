"""Only an authenticated empty readiness observation can continue collection."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from race_collection.operational_prediction import classify_unready_capture
from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked


@pytest.mark.parametrize('mutation', [None, 'rows', 'win', 'success', 'warning', 'identity', 'denial'])
def test_unready_capture_preserves_denial_and_no_write_boundary(tmp_path, monkeypatch, mutation):
    gate_path = tmp_path / 'access.json'
    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(gate_path))
    gate = SportsbetAccess(gate_path)
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'synthetic fixture'})
    stamp = datetime.now(timezone.utc)
    item = dict(race_id='Race 1 - SYNTHETIC', capture_window_minutes=10,
                race_identity={'jump_datetime': (stamp + timedelta(minutes=9)).isoformat()})
    claim = tmp_path / 'claim.json'
    claim.write_text(json.dumps({'item': item, 'reserved_at': (stamp-timedelta(seconds=1)).isoformat()}))
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    fetch = dict(success=False, write_performed=False, win_count=0, place_count=0,
                 warnings=['required_paired_markets_not_ready_within_readiness_budget'],
                 discovery_method='sportsbet_exact_race_paired_markets_unready')
    attempt = dict(race_id=item['race_id'], capture_window_minutes=10,
                   status='BLOCKED_VALIDATION_FAILED', inserted_rows=0,
                   fetch_time=stamp.isoformat(), fetch_result=fetch)
    if mutation == 'rows': attempt['inserted_rows'] = 1
    if mutation == 'win': fetch['win_count'] = 1
    if mutation == 'success': fetch['success'] = True
    if mutation == 'warning': fetch['warnings'] = ['unrecognized error']
    if mutation == 'identity': attempt['race_id'] = 'Race 2 - SYNTHETIC'
    if mutation == 'denial': gate.retain_denial(429)
    (evidence / 'autonomous_live_odds_capture_report.json').write_text(json.dumps({'attempts': [attempt]}))
    result = {'autonomous_live_odds_capture_status': {'output_dir': str(evidence)}}
    before = claim.read_bytes()
    if mutation == 'denial':
        with pytest.raises(SportsbetAccessBlocked):
            classify_unready_capture(claim, result, evidence, tmp_path)
        assert len(gate.read()['denials']) == 1
    else:
        outcome = classify_unready_capture(claim, result, evidence, tmp_path)
        if mutation is None:
            assert outcome['status'] == 'UNREADY_NO_CAPTURE'
            assert outcome['capture_consumed'] and not outcome['prediction_started']
        else:
            assert outcome is None
    assert claim.read_bytes() == before
