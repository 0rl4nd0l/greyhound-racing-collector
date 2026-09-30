"""Fixture-only future wall clock plus the existing invented source transports.

Never shipped by the runtime exporter (tests are excluded). The parent installs
kernel network denial before importing/execing this fixture.
"""
import datetime
import json
import os
from pathlib import Path
import time

clock = os.environ.get('SYNTHETIC_DEVELOPMENT_CLOCK')
if clock:
    real_datetime=datetime.datetime
    real_monotonic=time.monotonic
    def read_clock():
        value=json.loads(Path(clock).read_bytes())
        if value.get('synthetic') is not True:
            raise ValueError('synthetic_clock_label_required')
        return real_datetime.fromisoformat(value['at'])+datetime.timedelta(seconds=real_monotonic()-value['monotonic'])
    class SyntheticDateTime(real_datetime):
        @classmethod
        def now(cls,tz=None):
            current=read_clock()
            return cls.fromisoformat((current.astimezone(tz) if tz else current.astimezone().replace(tzinfo=None)).isoformat())
        @classmethod
        def utcnow(cls):return cls.fromisoformat(read_clock().astimezone(datetime.timezone.utc).replace(tzinfo=None).isoformat())
    datetime.datetime=SyntheticDateTime
    time.time=lambda:read_clock().timestamp()
if os.environ.get('FRESHNESS_FABRICATED_SOURCE'):
    import freshness_transport
    freshness_transport.install()
if os.environ.get('GREYHOUND_SHARED_SNAPSHOT_FIXTURE'):
    from snapshot_transport import install
    install()
