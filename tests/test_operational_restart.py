"""Restart admission preserves even partially written dispatch consumption."""
import hashlib
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from race_collection import operational_prediction as prediction
from race_collection.live_freshness_contract import create_once


@pytest.mark.parametrize('boundary', ['dispatch', 'partial_dispatch', 'race_started', 'terminal'])
def test_restarted_supervisor_never_reclaims_consumed_race(tmp_path, monkeypatch, boundary):
    root = tmp_path / 'operational-predictions'
    race_id = 'Race 1 - SYNTHETIC - 2099-01-01'
    identity = hashlib.sha256(race_id.encode()).hexdigest()
    claim = tmp_path / 'captures/one/capture-reservation.json'
    create_once(claim, {'item': {'race_id': race_id}})
    output = tmp_path / 'package'
    create_once(output / 'capture-verifications/one.json', {'synthetic': True})
    if boundary in {'dispatch', 'partial_dispatch'}:
        path = root / 'dispatches' / (identity + '.json')
        path.parent.mkdir(parents=True)
        path.write_bytes(b'{' if boundary == 'partial_dispatch' else b'{}')
    else:
        path = root / 'races' / identity
        path.mkdir(parents=True)
        if boundary == 'terminal':
            create_once(path / 'terminal.json', {'status': 'PREDICTION_READY'})
    scope = SimpleNamespace(end=prediction.now()+timedelta(minutes=10),
                            campaign=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr('race_collection.live_freshness_contract.AttemptAllowance',
                        lambda _: SimpleNamespace(claims=lambda: [claim]))
    def forbidden(*args, **kwargs):
        pytest.fail('restart attempted duplicate prediction')
    monkeypatch.setattr('subprocess.Popen', forbidden)
    for _ in range(2):
        restarted = prediction.Supervisor(output, {'operational_predictions': True}, scope)
        restarted.tick()
        restarted.drain()
        assert restarted.child is None
    assert claim.read_bytes() == (json.dumps({'item': {'race_id': race_id}}, sort_keys=True, separators=(',', ':'))+'\n').encode()
