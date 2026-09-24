"""All requests here are fabricated; execute with outbound networking denied."""

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import inspect_odds_com_au_public as probe

URL = "https://www.odds.com.au/greyhounds/fabricated-20990101/fixture-race-1/"


def test_public_host_rejects_bookmakers_and_lookalikes():
    assert probe.public_host(URL)
    for url in (
        "https://www.sportsbet.com.au/",
        "https://odds.com.au.evil/",
        "https://www.odds.com.au@www.sportsbet.com.au/",
    ):
        assert not probe.public_host(url)


def test_schedule_projection_excludes_results_and_runner_values():
    raw = json.dumps(
        {
            "props": {
                "startTime": "2099-01-01T05:00:00Z",
                "hasResults": False,
                "results": {"startTime": "2001-01-01", "winner": "SECRET"},
                "runners": [{"name": "SECRET", "position": 1, "price": 8.8}],
            }
        }
    )
    assert probe.metadata(raw) == [
        {"path": ["props", "startTime"], "time": "2099-01-01T05:00:00Z"},
        {"path": ["props", "hasResults"], "flag": False},
    ]
    assert "SECRET" not in json.dumps(probe.response_shape(raw))


def test_inline_json_and_query_redaction():
    parser = probe.InlineJSON()
    parser.feed('<script id="__NEXT_DATA__">{"hasResults":false}</script><script>SECRET</script>')
    assert parser.items == ['{"hasResults":false}']
    assert "SECRET" not in json.dumps(probe.route_metadata(URL + "?token=SECRET"))


class Request:
    def __init__(self, url, resource_type="xhr", redirected=False):
        self.url, self.resource_type = url, resource_type
        self.method = "GET"
        self.redirected_from = object() if redirected else None
        self.allowed = False
        self.aborted = False

    def is_navigation_request(self):
        return self.resource_type == "document"


class Route:
    def __init__(self, request):
        self.request = request

    def continue_(self):
        self.request.allowed = True

    def abort(self):
        self.request.aborted = True


def run_fake_browser(
    monkeypatch, tmp_path, *, status=403, extra_requests=(), body="", websocket=False
):
    import playwright.sync_api

    requests = []
    body_reads = []
    closed = []

    class Context:
        def route(self, pattern, callback):
            self.route_callback = callback

        def route_web_socket(self, pattern, callback):
            self.ws_callback = callback

        def new_page(self):
            return self

        def on(self, name, callback):
            self.response_callback = callback

        def goto(self, url, **kwargs):
            request = Request(url, "document")
            requests.append(request)
            self.route_callback(Route(request))
            assert request.allowed

            def text():
                body_reads.append(url)
                return body

            self.response_callback(
                SimpleNamespace(
                    url=url,
                    request=request,
                    status=status,
                    headers={"content-type": "text/html"},
                    text=text,
                )
            )
            for req in extra_requests:
                requests.append(req)
                self.route_callback(Route(req))
            if websocket:
                self.ws_callback(
                    SimpleNamespace(
                        url="wss://www.sportsbet.com.au/stream",
                        close=lambda: closed.append("websocket"),
                    )
                )

        def wait_for_timeout(self, delay):
            # Advance local monotonic time without a sleep or provider activity.
            clock[0] += 20

        def close(self):
            closed.append("context")

    context = Context()

    class Browser:
        def new_context(self, **kwargs):
            assert kwargs["service_workers"] == "block"
            return context

        def close(self):
            closed.append("browser")

    class Manager:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(launch=lambda **kwargs: Browser()))

        def __exit__(self, *args):
            pass

    clock = [0.0]
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(playwright.sync_api, "sync_playwright", Manager)
    path = tmp_path / "observation.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--url", URL, "--output", str(path)])
    probe.main()
    return json.loads(path.read_text()), requests, body_reads, closed, path


@pytest.mark.parametrize("status", [401, 403, 429])
def test_denial_stops_following_requests_without_reading_body(monkeypatch, tmp_path, status):
    following = Request("https://www.odds.com.au/api/racing")
    evidence, requests, reads, closed, path = run_fake_browser(
        monkeypatch, tmp_path, status=status, extra_requests=[following], body="SECRET RESULTS"
    )
    assert evidence["stop"] == "denial"
    assert len(evidence["requests_allowed"]) == 1
    assert reads == [] and following.aborted
    assert evidence["provider_operations"]["sportsbet"] == 0
    assert evidence["provider_operations"]["bookmaker_redirects"] == 0
    assert closed == ["context", "browser"]
    assert "SECRET" not in path.read_text()
    with pytest.raises(ValueError, match="no_retry"):
        probe.main()


def test_normal_page_blocks_bookmakers_results_redirects_and_websockets(monkeypatch, tmp_path):
    blocked = [
        Request("https://www.sportsbet.com.au/"),
        Request("https://www.odds.com.au/results/"),
        Request("https://www.odds.com.au/redirect", redirected=True),
        Request("https://www.odds.com.au/greyhounds/another/", "document"),
    ]
    evidence, requests, reads, closed, path = run_fake_browser(
        monkeypatch,
        tmp_path,
        status=200,
        extra_requests=blocked,
        websocket=True,
        body='<script id="__NEXT_DATA__">{"startTime":"2099-01-01T05:00:00Z","runners":[{"name":"SECRET","win":9.9}]}</script>',
    )
    assert all(request.aborted for request in blocked)
    assert len(evidence["requests_allowed"]) == 1
    assert len(evidence["requests_blocked"]) == 5
    assert (
        evidence["responses"][0]["inline_json"][0]["schedule_state"][0]["time"]
        == "2099-01-01T05:00:00Z"
    )
    assert "SECRET" not in path.read_text()
    assert closed == ["websocket", "context", "browser"]


def test_request_cap_is_durable_and_challenge_stops(monkeypatch, tmp_path):
    extra = [Request(f"https://www.odds.com.au/api/{i}") for i in range(65)]
    evidence, requests, _, _, _ = run_fake_browser(
        monkeypatch, tmp_path, status=200, extra_requests=extra
    )
    assert len(evidence["requests_allowed"]) == 60
    assert evidence["stop"] == "request_cap"
    assert sum(request.allowed for request in requests) == 60
    other = tmp_path / "other"
    other.mkdir()
    evidence, requests, _, _, _ = run_fake_browser(
        monkeypatch,
        other,
        status=200,
        body="Verify you are human",
        extra_requests=[Request("https://www.odds.com.au/api/")],
    )
    assert evidence["stop"] == "challenge_indicator"
    assert requests[-1].aborted


def test_honest_odds_source_fails_actual_capture_and_receipt_contract():
    from datetime import datetime, timezone
    from scripts.autonomous_live_odds_capture import validate_fetched_odds
    from src.predictor.on_demand import normalize_validation_receipt, PredictionBlocked

    # Fabricated complete UI rows isolate the acquisition-source boundary. A
    # Sportsbet bookmaker label must not rewrite an Odds.com.au acquisition URL.
    rows = [
        {
            "box_number": box,
            "dog_name": name,
            "odds_decimal": 2.4,
            "sportsbet_box_source": "runner_text",
        }
        for box, name in ((1, "Alpha"), (2, "Bravo"))
    ]
    plan = {
        "race_number": 1,
        "expected_runners": [
            {
                "box_number": row["box_number"],
                "dog_name": row["dog_name"],
                "identity": row["dog_name"].upper(),
            }
            for row in rows
        ],
    }
    fetched = {
        "success": True,
        "source_url": URL,
        "acquisition_source": "odds.com.au",
        "price_origin_bookmaker": "sportsbet",
        "odds_data": rows,
        "odds_data_place": [dict(row, odds_decimal=1.4) for row in rows],
    }
    validated = validate_fetched_odds(plan, fetched)
    assert validated["status"] == "FAIL"
    assert "sportsbet_source_url_not_sportsbet" in validated["reasons"]
    assert validated["accepted_row_count"] == validated["accepted_place_row_count"] == 2
    assert validated["source_url"] == URL
    with pytest.raises(PredictionBlocked, match="MARKET_UNAVAILABLE"):
        normalize_validation_receipt(
            race_id="Race 1 - WPK - 2099-01-01",
            captured_at=datetime.now(timezone.utc),
            validation=validated,
            source_kind="odds_com_au",
        )
