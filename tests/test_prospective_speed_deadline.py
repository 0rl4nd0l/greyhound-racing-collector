import signal
import time

import pytest

from race_collection.prospective_speed_deadline import bounded


def test_whole_cycle_deadline_survives_completed_inner_fetch():
    with pytest.raises(TimeoutError, match='RESULT_CYCLE_DEADLINE'):
        with bounded(0.04):
            with bounded(0.02):
                pass
            time.sleep(0.1)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_caught_shorter_fetch_timeout_leaves_outer_budget():
    with bounded(0.1):
        with pytest.raises(TimeoutError):
            with bounded(0.01):
                time.sleep(0.05)
        left, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0 < left < 0.1 and interval == 0
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_longer_inner_deadline_does_not_extend_parent():
    with pytest.raises(TimeoutError):
        with bounded(0.02):
            with bounded(1):
                time.sleep(0.05)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
