from datetime import datetime, timedelta, timezone

from race_collection.live_phase_budget import LiveBudget


def test_admission_reserves_refresh_and_timer_slack():
    budget = LiveBudget()
    observed = datetime(2026, 7, 19, tzinfo=timezone.utc)
    assert budget.admit_work(observed, observed + timedelta(seconds=130))
    assert not budget.admit_work(observed, observed + timedelta(seconds=130.01))
    assert budget.safe_to_yield(observed, observed + timedelta(seconds=75))
    assert not budget.safe_to_yield(observed, observed + timedelta(seconds=75.01))
    assert not budget.admit_work(None, observed)
    assert not budget.admit_work(observed, observed - timedelta(seconds=1))


def test_long_phase_is_an_overrun_not_extra_freshness():
    budget = LiveBudget()
    assert budget.overrun("refresh", 65) is False
    assert budget.overrun("refresh", 65.01) is True
    assert budget.overrun("capture", 50.01) is True


def test_overhead_is_one_aggregate_reservation_not_per_operation():
    budget = LiveBudget()
    assert not budget.overhead_exceeded(10)
    assert budget.overhead_exceeded(10.01)
    assert budget.publication_age == 270
    assert budget.refresh_seconds == 65
    assert budget.work_seconds == 50
    assert budget.next_trigger_seconds == 75
    assert budget.handoff_overhead_seconds == 10
    assert budget.handoff_poll_seconds == 5
