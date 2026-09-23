"""Controlled local transport only; never contact a source provider."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def test_shared_http_client_surfaces_429_before_any_transport_retry(monkeypatch):
    from utils import http_client

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(429 if len(calls) == 1 else 200)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    monkeypatch.setattr(http_client, "_shared_session", None)
    client = http_client.get_shared_session()
    client.trust_env = False
    try:
        response = client.get(f"http://127.0.0.1:{server.server_port}/fixture", timeout=2)
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "0"
        assert len(calls) == 1
    finally:
        client.close()
        server.shutdown()
        worker.join()
        server.server_close()


def test_browser_denial_retains_only_retry_guidance(tmp_path, monkeypatch):
    from race_collection.live_execution import BrowserNetworkAccounting

    monkeypatch.delenv("GREYHOUND_LIVE_CONTRACT", raising=False)

    class Driver:
        def get(self, url):
            pass

        def get_log(self, name):
            return [{"message": json.dumps({"message": {
                "method": "Network.responseReceived", "params": {"response": {
                    "url": "https://www.sportsbet.com.au/fixture?secret=hidden",
                    "status": 429,
                    "headers": {"Retry-After": "120", "Date": "Wed, 23 Sep 2026 07:00:00 GMT",
                                "Set-Cookie": "secret", "Authorization": "secret"},
                }}}})}]

    path = tmp_path / "browser.json"
    meter = BrowserNetworkAccounting(Driver(), path)
    with pytest.raises(ValueError, match="source_access_denied"):
        meter.drain()
    evidence = json.loads(path.read_text())["source_access_denied"]
    assert evidence["retry_headers"] == {
        "retry-after": "120", "date": "Wed, 23 Sep 2026 07:00:00 GMT"}
    assert evidence["observed_at"]
    assert "secret" not in path.read_text()


def test_python_guard_records_guidance_and_stops_next_logical_request(tmp_path, monkeypatch):
    import requests
    from datetime import datetime
    from tests.test_live_capture_binding import reserved_alias_plan
    from race_collection import live_freshness_contract as contracts

    allowance, _, _, stamp = reserved_alias_plan(tmp_path)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return stamp

    calls = []

    def response(session, method, url, **kwargs):
        calls.append(url)
        result = requests.Response()
        result.status_code = 429
        result.headers = {"Retry-After": "120", "Set-Cookie": "secret"}
        return result

    monkeypatch.setattr(contracts, "datetime", Clock)
    monkeypatch.setattr(requests.Session, "request", response)
    restore = contracts.install_request_guard(allowance.scope)
    try:
        with pytest.raises(ValueError, match="source_access_denied"):
            requests.get("https://www.sportsbet.com.au/fixture")
        with pytest.raises(ValueError):
            requests.get("https://www.sportsbet.com.au/fixture")
    finally:
        restore()
    assert len(calls) == 1
    path, = allowance.scope.session.glob("source-access-denied-*.json")
    assert json.loads(path.read_text())["retry_headers"] == {"retry-after": "120"}
    assert "secret" not in path.read_text()


def test_access_stop_retry_preserves_existing_server_error_policy():
    from utils.http_client import AccessDenialAwareRetry

    retry = AccessDenialAwareRetry(total=2, backoff_factor=0.1,
                                  status_forcelist=(500, 502, 503, 504))
    for status in (401, 403, 429):
        assert not retry.is_retry("GET", status, has_retry_after=True)
    assert retry.is_retry("GET", 503, has_retry_after=True)
    assert retry.respect_retry_after_header
