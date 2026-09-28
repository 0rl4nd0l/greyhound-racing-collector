"""Calendar delay must exclude an observed running predecessor, not idle delay."""
from datetime import datetime, timedelta, timezone

import pytest

from race_collection.freshness_rehearsal import TimerAccounting


START = datetime(2026, 9, 28, 0, 48, 44, tzinfo=timezone.utc)


def sample(seconds, trigger, started, active, overhead=None):
    return {
        'read_start': (START+timedelta(seconds=seconds)).isoformat(),
        'monotonic_start': 100+seconds,
        'timer_status': {'full': {}, 'odds': {'LastTriggerUSecMonotonic': str(int(trigger*1e6))}},
        'unit_status': {
            'full': {'ActiveState': 'inactive', 'ExecMainStartTimestampMonotonic': '0'},
            'odds': {'ActiveState': active, 'ExecMainStartTimestampMonotonic': str(int(started*1e6)),
                     'ExecMainExitTimestampMonotonic': str(int((100+seconds)*1e6)) if active=='inactive' else '0'}},
        'external_service_overhead_seconds': {'odds': overhead},
    }


def test_actual_observed_predecessor_overlap_is_not_dispatch_overhead():
    accounting = TimerAccounting(START)
    accounting.observe(sample(1, 100, 100.1, 'active'))
    accounting.observe(sample(36, 100, 100.1, 'active'))
    accounting.observe(sample(40, 138, 138.1, 'active'))
    accounting.observe(sample(72, 138, 138.1, 'inactive', 3.2))
    row = accounting.summary(START+timedelta(seconds=72))['activations']['odds'][-1]
    assert row['calendar_delay_seconds'] == 22
    assert row['calendar_blocked_until_monotonic'] == 136
    assert row['complete_overhead_seconds'] == pytest.approx(5.3)


def test_persistent_startup_trigger_uses_scope_start_as_earliest_admission():
    accounting = TimerAccounting(START)
    accounting.observe(sample(1, 100, 100.1, 'active'))
    accounting.observe(sample(8, 100, 100.1, 'inactive', 3.2))
    row = accounting.summary(START+timedelta(seconds=8))['activations']['odds'][0]
    assert row['calendar_delay_seconds'] == 44
    assert row['complete_overhead_seconds'] == pytest.approx(3.3)


@pytest.mark.parametrize('active_through,overhead', [(10,3.2),(36,10.1)])
def test_idle_calendar_delay_and_genuine_overhead_still_reject(active_through, overhead):
    accounting = TimerAccounting(START)
    accounting.observe(sample(1, 100, 100.1, 'active'))
    accounting.observe(sample(active_through, 100, 100.1, 'active'))
    accounting.observe(sample(40, 138, 138.1, 'active'))
    with pytest.raises(ValueError, match='dispatch_plus_process_overhead_exceeded'):
        accounting.observe(sample(72, 138, 138.1, 'inactive', overhead))
