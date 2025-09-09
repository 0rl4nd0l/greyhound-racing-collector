# Gunicorn configuration for greyhound_racing_collector
# - Defaults are conservative and SSE-friendly (threaded workers)
# - Tunable via environment variables without editing this file
#
# Key env vars:
#   PORT                      (default: 5002)
#   GUNICORN_WORKERS          (default: max(2, cpu_count()))
#   GUNICORN_THREADS          (default: 4)
#   GUNICORN_WORKER_CLASS     (default: gthread)
#   GUNICORN_TIMEOUT          (default: 120)
#   GUNICORN_GRACEFUL_TIMEOUT (default: 30)
#   GUNICORN_KEEPALIVE        (default: 30)
#   GUNICORN_MAX_REQUESTS     (default: 0)
#   GUNICORN_MAX_REQUESTS_JITTER (default: 0)
#   GUNICORN_ACCESSLOG        (default: '-')
#   GUNICORN_ERRORLOG         (default: '-')
#   GUNICORN_LOGLEVEL         (default: info)
#   GUNICORN_PRELOAD          (default: 0)
#   GUNICORN_WORKER_TMP_DIR   (default: None)

import multiprocessing
import os


def _to_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


_port = os.getenv('PORT', '5002')
bind = os.getenv('GUNICORN_BIND', '') or f":{_port}"
workers = _to_int("GUNICORN_WORKERS", max(2, multiprocessing.cpu_count()))
threads = _to_int("GUNICORN_THREADS", 4)
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread")

# Timeouts and connection handling
timeout = _to_int("GUNICORN_TIMEOUT", 120)
graceful_timeout = _to_int("GUNICORN_GRACEFUL_TIMEOUT", 30)
keepalive = _to_int("GUNICORN_KEEPALIVE", 30)

# Request recycling (disabled by default)
max_requests = _to_int("GUNICORN_MAX_REQUESTS", 0)
max_requests_jitter = _to_int("GUNICORN_MAX_REQUESTS_JITTER", 0)

# Logging
accesslog = os.getenv("GUNICORN_ACCESSLOG", "-")
errorlog = os.getenv("GUNICORN_ERRORLOG", "-")
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

# Preload for faster worker startup (beware of DB connections and caches)
preload_app = os.getenv("GUNICORN_PRELOAD", "0").lower() in ("1", "true", "yes")

# Temp directory (can help on some filesystems)
_worker_tmp_dir_env = os.getenv("GUNICORN_WORKER_TMP_DIR", "")
worker_tmp_dir = _worker_tmp_dir_env or None

# If behind a reverse proxy that passes X-Forwarded-* headers
forwarded_allow_ips = os.getenv("GUNICORN_FORWARDED_ALLOW_IPS", "*")
proxy_protocol = os.getenv("GUNICORN_PROXY_PROTOCOL", "0").lower() in ("1", "true", "yes")

