import os
import threading
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session_lock = threading.Lock()
_shared_session: Optional[requests.Session] = None


class AccessDenialAwareRetry(Retry):
    """Expose access stops to callers even when urllib3 sees Retry-After.

    A retry header is timing guidance, not permission to hide a source denial
    behind an adapter retry. Existing transient-server-error retries remain.
    """

    def is_retry(self, method, status_code, has_retry_after=False):
        if status_code in {401, 403, 429}:
            return False
        return super().is_retry(method, status_code, has_retry_after)


def source_retry_headers(headers):
    """Retain bounded retry metadata without cookies, credentials or URLs."""
    allowed = {
        "retry-after", "date", "ratelimit-limit", "ratelimit-remaining",
        "ratelimit-reset", "x-ratelimit-limit", "x-ratelimit-remaining",
        "x-ratelimit-reset",
    }
    return {
        str(key).lower(): str(value)[:256].replace("\r", "").replace("\n", "")
        for key, value in (headers or {}).items()
        if str(key).lower() in allowed
    }


class SourceCoordinatedSession(requests.Session):
    """Each Sportsbet transport attempt participates in the durable source hold."""

    def __init__(self):
        super().__init__()
        self.sportsbet_adapter = HTTPAdapter(max_retries=0)

    def get_adapter(self, url):
        from utils.sportsbet_access import is_sportsbet

        return self.sportsbet_adapter if is_sportsbet(url) else super().get_adapter(url)

    def send(self, request, **kwargs):
        from utils.sportsbet_access import SportsbetAccess, is_sportsbet

        if not is_sportsbet(request.url):
            return super().send(request, **kwargs)
        # A redirect is not permission to change the route during recovery.
        kwargs["allow_redirects"] = False
        with SportsbetAccess().operation("python") as operation:
            response = super().send(request, **kwargs)
            operation.response(response.status_code, response.headers, source_url=request.url, resource_type="python")
            if operation.recovery and 200 <= response.status_code < 300:
                from utils.prejump_sportsbet import usable_recovery_snapshot

                if usable_recovery_snapshot(request.url, response):
                    operation.accept_data()
            return response

    def close(self):
        self.sportsbet_adapter.close()
        super().close()


def get_shared_session() -> requests.Session:
    """Return a process-wide shared requests.Session configured with a larger
    connection pool and light retries. Safe for concurrent use.
    """
    global _shared_session
    if _shared_session is not None:
        return _shared_session
    with _session_lock:
        if _shared_session is not None:
            return _shared_session
        s = SourceCoordinatedSession()
        retry = AccessDenialAwareRetry(
            total=0 if os.environ.get("GREYHOUND_LIVE_EXECUTION") else 2,
            backoff_factor=0.1,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "POST", "PUT", "DELETE"),
        )
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _shared_session = s
        return _shared_session
