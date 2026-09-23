"""Opt-in acquisition accounting; never caches, retries, or changes a request.

One file per discovery process / race worker directory. Request spans include
the existing HTTP client's retries; they are logical calls, not wire attempts.
Nested/parallel spans must not be summed as elapsed refresh time. Unmatched
starts are incomplete work, including uncatchable process termination.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit


class RefreshRequestTiming:
    def __init__(self, directory):
        self.path = Path(directory) / "refresh-request-timing.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sequence = itertools.count()

    def _write(self, event):
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema_version": "refresh_request_timing_v1",
                        "pid": os.getpid(),
                        "utc": datetime.now(timezone.utc).isoformat(),
                        **event,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    @contextmanager
    def span(self, kind, **details):
        identity = next(self._sequence)
        started = time.monotonic()
        self._write(
            {"event": "start", "id": identity, "kind": kind, "monotonic": started, **details}
        )
        outcome = {}
        try:
            yield outcome
        except BaseException as error:
            outcome["error_type"] = type(error).__name__
            raise
        finally:
            finished = time.monotonic()
            self._write(
                {
                    "event": "end",
                    "id": identity,
                    "kind": kind,
                    "monotonic": finished,
                    "elapsed_seconds": finished - started,
                    **details,
                    **outcome,
                }
            )

    def call(self, kind, function, *args, **kwargs):
        with self.span(kind):
            return function(*args, **kwargs)

    def attach(self, browser):
        browser.session = _TimedSession(browser.session, self)


class _TimedSession:
    def __init__(self, session, timing):
        self._session = session
        self._timing = timing

    def __getattr__(self, name):
        original = getattr(self._session, name)
        if name not in {"get", "post", "head"}:
            return original

        def request(url, *args, **kwargs):
            # Keep endpoint identity without credentials, query parameters,
            # request/response headers, bodies, or exception messages.
            parts = urlsplit(url)
            endpoint = f"{parts.scheme}://{parts.hostname or ''}{parts.path}"
            response = None
            try:
                with self._timing.span("request", method=name.upper(), endpoint=endpoint) as result:
                    response = original(url, *args, **kwargs)
                    result["status_code"] = response.status_code
                return response
            except BaseException:
                # An end-record failure must not leak a response which the
                # caller never received. Ordinary responses remain caller-owned.
                if response is not None:
                    response.close()
                raise

        return request
