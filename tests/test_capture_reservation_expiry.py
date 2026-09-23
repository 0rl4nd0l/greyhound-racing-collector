"""Fabricated reproduction of the 16:11 T-10 admission boundary."""
from datetime import datetime, timedelta

import pytest

from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest
from tests.test_live_freshness_candidate import contract_value


def allowance_and_item(tmp_path):
    value = contract_value(tmp_path)
    accounting = dict(schema_version='freshness_attempt_reconciliation_v1', complete=True,
                      consumed=[], sources=[{'sha256': 'b' * 64}])
    value['reconciliation_sha256'] = digest(accounting)
    allowance = AttemptAllowance(FreshnessContract(value))
    allowance.initialize(accounting)
    item = dict(race_id='Race 7 - SYNTHETIC - 2026-09-23', capture_window_minutes=10,
                race_identity={'jump_datetime': '2026-09-23T01:13:00+00:00'})
    return allowance, item


@pytest.mark.parametrize('offset', [0, 0.927768])
def test_expired_plan_rejected_before_consumption(tmp_path, offset):
    allowance, item = allowance_and_item(tmp_path)
    now = datetime.fromisoformat('2026-09-23T01:11:00+00:00') + timedelta(seconds=offset)
    with pytest.raises(ValueError, match='capture_reservation_expired'):
        allowance.reserve(item, now=now)
    assert allowance.available()
    assert not allowance.consumed(item)


def test_delayed_start_preserves_consumption_and_rejects_fetch(tmp_path):
    allowance, item = allowance_and_item(tmp_path)
    close = datetime.fromisoformat('2026-09-23T01:11:00+00:00')
    claim = allowance.reserve(item, now=close - timedelta(seconds=60))
    with pytest.raises(ValueError, match='capture_reservation_expired'):
        allowance.start_fetch(claim, item, now=close)
    assert not allowance.available()
    assert allowance.consumed(item)
    assert not claim.with_suffix('.fetch.json').exists()


@pytest.mark.parametrize("window,smaller", [(60, 30), (30, 10), (10, 2), (2, 0)])
def test_all_native_window_boundaries_are_exclusive(window, smaller):
    jump = datetime.fromisoformat('2026-09-23T02:00:00+00:00')
    item = dict(race_id='synthetic', capture_window_minutes=window, race_identity={'jump_datetime': jump.isoformat()})
    opens, closes = jump - timedelta(minutes=window), jump - timedelta(minutes=smaller)
    with pytest.raises(ValueError, match='not_open'):
        AttemptAllowance.check_window(item, now=opens - timedelta(microseconds=1))
    assert AttemptAllowance.check_window(item, now=opens) == closes
    assert AttemptAllowance.check_window(item, now=closes - timedelta(microseconds=1)) == closes
    with pytest.raises(ValueError, match='expired'):
        AttemptAllowance.check_window(item, now=closes)
    with pytest.raises(ValueError, match='insufficient_time'):
        AttemptAllowance.check_window(item, now=closes - timedelta(seconds=50), required_seconds=50)
