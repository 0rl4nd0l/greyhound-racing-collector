"""Nested wall-clock limits that preserve an outer result-cycle deadline."""
from contextlib import contextmanager
import signal
import time


@contextmanager
def bounded(seconds):
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    if interval:
        raise ValueError('PERIODIC_ALARM_NOT_SUPPORTED')
    limit = min(seconds, remaining) if remaining else seconds
    if limit <= 0:
        raise TimeoutError('RESULT_CYCLE_DEADLINE')

    def expired(_signum, _frame):
        raise TimeoutError('RESULT_CYCLE_DEADLINE')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if remaining:
            # Even a caught inner timeout must not silently cancel its parent.
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, remaining-(time.monotonic()-started)))
