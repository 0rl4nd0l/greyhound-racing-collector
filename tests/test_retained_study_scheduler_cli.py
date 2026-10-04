"""CLI exit status follows the real amended readiness gate, including failures."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import pytest

from scripts import run_comparison_schedule as schedule
from tests.test_retained_study_readiness import fixture, put
from race_collection.retained_study_readiness import checked
from race_collection.live_freshness_contract import digest


@pytest.mark.parametrize('condition,expected_status,expected_exit', [
    ('ready', 'RETAINED_STUDY_OBSERVER_READY', 0),
    ('not_effective', 'CANARY_NOT_VERIFIED', 2),
    ('changed_evidence', 'SCHEDULE_FAILED', 2),
])
def test_scheduler_cli_uses_actual_readiness_gate(tmp_path, monkeypatch, capsys,
                                                 condition, expected_status, expected_exit):
    cfg, at = fixture(tmp_path, monkeypatch)
    amendment = checked(cfg['study_amendment'])
    root = Path(cfg['state_root'])
    identity = put(root/'config-identity.json', {'sha256': digest(checked(amendment['predecessor_config']))})
    original_identity = Path(identity['path']).read_bytes()
    original_terminal = (root/'slots/001/terminal.json').read_bytes()
    if condition == 'not_effective':
        at = datetime.fromisoformat(amendment['issued_at'])
    if condition == 'changed_evidence':
        (tmp_path/'manifest.json').write_bytes(b'changed')
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return at
    monkeypatch.setattr(schedule, 'datetime', Clock)
    # Existing synthetic configuration fixture supplies the external config loader;
    # tick, bound amendment, evidence verifier, health writer and argparse are real.
    monkeypatch.setattr(schedule, 'load_config', lambda *a: (cfg, {'ends_at': (at+timedelta(days=100)).isoformat()}))
    monkeypatch.setattr(schedule, 'renew_source', lambda *a, **k: pytest.fail('no provider ownership'))
    monkeypatch.setattr(schedule, 'child', lambda *a, **k: pytest.fail('no collector child'))
    monkeypatch.setattr(sys, 'argv', ['run_comparison_schedule.py', '--config', str(tmp_path/'fixture.json')])
    code = schedule.main()
    emitted = json.loads(capsys.readouterr().out)
    assert emitted['status'] == expected_status
    assert emitted['outcomes_released'] is False
    assert code == expected_exit
    assert json.loads((root/'health.json').read_bytes())['status'] == expected_status
    assert Path(identity['path']).read_bytes() == original_identity
    assert (root/'slots/001/terminal.json').read_bytes() == original_terminal
    assert not (root/'canary.json').exists()
    assert sorted(p.name for p in (root/'slots').iterdir()) == ['001']
