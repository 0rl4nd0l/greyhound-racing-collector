"""Subprocess fixture: invented clock/HTTP, native wrapper and owner checks."""
import datetime
import json
import os
from pathlib import Path
import sys
import time

if os.environ.get('GREYHOUND_INCIDENT_SERVICE_FIXTURE'):
    real_datetime, real_date = datetime.datetime, datetime.date
    base = float(os.environ['GREYHOUND_FIXTURE_EPOCH'])
    started = float(os.environ['GREYHOUND_FIXTURE_MONOTONIC'])

    def epoch():
        return base + time.monotonic() - started

    class FixtureDateTime(real_datetime):
        __slots__ = ()
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(epoch(), tz)

        @classmethod
        def utcnow(cls):
            return cls.fromtimestamp(epoch(), datetime.timezone.utc).replace(tzinfo=None)

    class FixtureDate(real_date):
        __slots__ = ()
        @classmethod
        def today(cls):
            return FixtureDateTime.now().date()

    time.time = epoch
    datetime.datetime, datetime.date = FixtureDateTime, FixtureDate
    name = Path(sys.argv[0]).name
    if name in {'run_freshness_service.py', 'shadow_autopilot_daemon.py'}:
        path = Path(os.environ['GREYHOUND_INCIDENT_SERVICE_FIXTURE'])
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, (json.dumps({'script':name, 'pid':os.getpid(),
                'authority':os.environ.get('GREYHOUND_INCIDENT_AUTHORITY_SHA256'),
                'slot':os.environ.get('GREYHOUND_INCIDENT_SLOT')})+'\n').encode())
        finally:
            os.close(descriptor)
    from snapshot_transport import install
    install()
