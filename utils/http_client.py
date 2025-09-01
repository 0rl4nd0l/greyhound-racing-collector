import threading
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session_lock = threading.Lock()
_shared_session: Optional[requests.Session] = None


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
        s = requests.Session()
        retry = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "POST", "PUT", "DELETE"),
        )
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _shared_session = s
        return _shared_session

