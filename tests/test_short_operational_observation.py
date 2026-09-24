from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from scripts import prepare_freshness_rehearsal as prep
from race_collection.live_freshness_contract import FreshnessContract, digest


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    root.mkdir()
    (root / 'ledger.json').write_text('{"attempts": []}')
    campaign = SimpleNamespace(root=root, value={'max_capture_attempts': 64})
    monkeypatch.setattr('race_collection.freshness_campaign.Campaign', lambda _: campaign)
    monkeypatch.setitem(sys.modules, 'sportsbet_odds_integrator', SimpleNamespace(
        SportsbetOddsIntegrator=lambda path, **kw: Path(path).touch()))
    monkeypatch.setattr('race_collection.operational_prediction.prepare_retention', lambda *a: 'a'*64)
    monkeypatch.setattr(prep, 'probe_runtime', lambda **kw: {})
    def git(command, **kw):
        if command[1] == 'rev-parse': return 'a'*40+'\n'
        if command[1] == 'ls-tree': return 'placeholder.py\n'
        return b'# fixture\n'
    monkeypatch.setattr(prep.subprocess, 'check_output', git)
    monkeypatch.setattr(prep.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0,stdout='{}',stderr=''))
    installed = tmp_path / 'installed'
    installed.mkdir()
    for name in (*prep.UNITS, 'greyhound-operator-ui-r3.service'):
        (installed / name).write_text('baseline')
    history = tmp_path / 'history.sqlite'
    history.touch()
    return dict(output=tmp_path/'package',start=datetime(2026,9,24,11,tzinfo=timezone.utc),
        python=Path(sys.executable),db=history,lock=tmp_path/'collector.lock',
        reconciliation_roots=[],installed_dir=installed,campaign_root=root,operational_predictions=True)


@pytest.mark.parametrize('minutes', [10, 59, 60, 61, 90])
def test_prepared_short_scope_timer_and_native_warmup(prepared, minutes):
    prep.prepare(**prepared, observation_minutes=minutes)
    output = prepared['output']
    plan = json.loads((output/'plan.json').read_text())
    short = minutes < 60
    timer = (output/'units/shadow-autopilot.timer').read_text()
    assert ('OnActiveSec=1s\n' if short else 'OnActiveSec=15min\n') in timer
    assert 'OnUnitInactiveSec=15min\n' in timer
    assert plan['readiness_warmup_seconds'] == (180 if short else 1200)
    assert plan.get('minimum_completed_full_cycles', 3) == (1 if short else 3)
    assert plan.get('minimum_distinct_captures', 3) == (1 if short else 3)
    assert plan['cleanup_seconds'] == 1860
    assert plan['first_index_deadline_seconds'] == 180
    value = dict(plan, schema_version='freshness_rehearsal_contract_v1',source_date='2026-09-24',reconciliation_sha256='b'*64)
    assert (FreshnessContract(value).end - prepared['start']).total_seconds() == minutes * 60


@pytest.mark.parametrize('minutes', [0, 9, 10.5, True, 91])
def test_prepare_rejects_invalid_operational_duration(prepared, minutes):
    with pytest.raises(ValueError, match='invalid_operational_observation_duration'):
        prep.prepare(**prepared, observation_minutes=minutes)
    assert not prepared['output'].exists()


@pytest.mark.parametrize('campaign, operational', [(False,False),(True,False),(False,True)])
def test_short_prepare_requires_both_authorities(prepared,campaign,operational):
    prepared.update(campaign_root=prepared['campaign_root'] if campaign else None,
                    operational_predictions=operational)
    with pytest.raises(ValueError, match='invalid_operational_observation_duration'):
        prep.prepare(**prepared, observation_minutes=10)
