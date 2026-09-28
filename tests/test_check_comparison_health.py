"""Monitoring detects silence/holds without touching target data."""
from datetime import datetime, timedelta, timezone
import pytest
from scripts.check_comparison_health import evaluate

NOW = datetime(2026, 10, 5, 5, tzinfo=timezone.utc)


def row(status, age=0, **fields):
    return {'status': status, 'at': (NOW - timedelta(minutes=age)).isoformat(), **fields}


def check(schedule, results, active='inactive'):
    return evaluate(schedule, results, now=NOW, schedule_active=active)


def test_running_session_allows_natural_drain_but_not_missing_process():
    schedule = row('SESSION_RUNNING', 130)
    results = row('CAMPAIGN_OWNER_BUSY', 20, counts={'PENDING': 3})
    assert check(schedule, results, 'active') == []
    assert check(schedule, results) == ['schedule:running_without_service']
    assert 'schedule:health_stale_or_future' in check(row('SESSION_RUNNING', 136), results, 'active')


@pytest.mark.parametrize('schedule,results,expected', [
    (None, row('CYCLE_COMPLETE'), 'schedule:health_missing'),
    (row('NO_SLOT_DUE', 16), row('CYCLE_COMPLETE'), 'schedule:health_stale_or_future'),
    (row('NO_SLOT_DUE'), row('CYCLE_COMPLETE', 46), 'results:health_stale_or_future'),
    (row('RESTORATION_HELD'), row('CYCLE_COMPLETE'), 'schedule:worker_hold_or_failure'),
    (row('NO_SLOT_DUE'), row('CYCLE_COMPLETE', counts={'QUARANTINED': 1}), 'results:unresolved_terminal_members'),
    (row('NO_SLOT_DUE'), row('CYCLE_COMPLETE', oldest_due=(NOW-timedelta(days=2)).isoformat()), 'results:overdue_more_than_24h'),
])
def test_structural_monitor_reports_lost_progress(schedule, results, expected):
    assert expected in check(schedule, results)


def test_busy_worker_is_healthy_without_burning_an_attempt():
    assert check(row('NO_SLOT_DUE'), row('COLLECTOR_LOCK_BUSY', 20, request_attempts=3)) == []
