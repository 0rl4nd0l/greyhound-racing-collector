"""Synthetic browser events only; run in an outbound-denied network namespace."""

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from utils.sportsbet_response_inspection import ResponseInspection, response_shape, safe_route

URL = "https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/Events?eventId=123"


def inspector():
    return ResponseInspection(expires_at=datetime.now(timezone.utc) + timedelta(seconds=45))


def emit(recorder, method, **params):
    recorder.observe({"method": "Network." + method, "params": {"requestId": "1", **params}})


def complete(recorder, **overrides):
    emit(
        recorder,
        "requestWillBeSent",
        request={"url": URL, "method": "GET", "headers": {"Cookie": "secret"}},
    )
    emit(
        recorder,
        "responseReceived",
        response={"url": URL, "status": 200, "mimeType": "application/json", **overrides},
    )
    emit(recorder, "loadingFinished", encodedDataLength=100)


def test_shape_never_exposes_values_or_unknown_field_names():
    raw = json.dumps(
        {
            "eventId": 123,
            "runners": [
                {
                    "id": 456,
                    "name": "protected dog",
                    "place": 1,
                    "finishingPosition": 2,
                    "secret dynamic key": "token",
                    "win": 9.3,
                }
            ],
            "Authorization": "secret",
        }
    )
    shape = response_shape(raw)
    text = json.dumps(shape)
    for private in (
        "123",
        "456",
        "protected",
        "finishingPosition",
        "secret",
        "token",
        "9.3",
        "Authorization",
    ):
        assert private not in text
    assert shape["shape"]["fields"]["eventId"] == {"type": "number"}
    assert shape["state"] == "shape_only"


def test_document_response_distinguishes_main_child_and_unknown_without_urls():
    rec = inspector()
    rec.observe({"method": "Page.frameNavigated", "params": {"frame": {
        "id": "private-main", "url": "https://www.sportsbet.com.au/private?token=secret",
    }}})
    rec.observe({"method": "Page.frameAttached", "params": {
        "frameId": "private-child", "parentFrameId": "private-main",
    }})
    for frame in ["private-main", "private-child", "not-seen"]:
        emit(rec, "responseReceived", frameId=frame, type="Document", response={
            "url": "https://www.sportsbet.com.au/private?token=secret", "status": 429,
            "fromDiskCache": True,
        })
    rows = rec.report()["network_responses"]
    assert [row["frame_role"] for row in rows] == ["top_level", "child", "unknown"]
    assert all(row["from_cache"] for row in rows)
    assert len({row["frame_id_sha256"] for row in rows}) == 3
    assert "private" not in json.dumps(rows)
    assert "secret" not in json.dumps(rows)


def test_route_redacts_secrets_and_marks_incomplete_route():
    route = safe_route(URL + "&token=secret")
    assert route["query"] == {"eventId": "123"}
    assert route["omitted_query_fields"] == 1
    assert safe_route(URL.replace("/Events", "/secret"))["path_redacted"]
    assert "secret" not in json.dumps(safe_route(URL.replace("/Events", "/secret")))
    assert safe_route("https://account.sportsbet.com.au/private") is None
    assert safe_route("https://www.sportsbet.com.au.evil/apigw/") is None


@pytest.mark.parametrize("suffix", [
    "AllRacing/2099-01-01", "Events/12345678/Racecard",
    "Events/12345678/RacecardWithContext", "Events/MultipleRacecards",
])
def test_documented_racing_routes_keep_names_but_not_target_values(suffix):
    route = safe_route(
        "https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/"
        + suffix + "?eventIds=12345678,87654321&selectionNames=PRIVATE_RUNNER"
    )
    text = json.dumps(route)
    for secret in ("2099-01-01", "12345678", "87654321", "PRIVATE_RUNNER"):
        assert secret not in text
    assert route["omitted_query_fields"] == 2
    assert any(name in route["path"] for name in (
        "AllRacing", "Racecard", "MultipleRacecards"
    ))


def test_racing_shape_distinguishes_price_lists_without_exposing_values():
    # Fabricated, documentation-shaped horse example; no greyhound claim.
    raw = json.dumps({"racecardEvent": {
        "id": 12345678, "type": "horse", "bettingStatus": "PRICED",
        "markets": [{"numPlaces": 3, "statusCode": "A", "selections": [{
            "runnerNumber": 1, "drawNumber": 3, "isOut": False,
            "prices": [{"priceCode": "L", "winPrice": 8.75, "placePrice": 2.65}],
            "result": "PRIVATE_RESULT", "shortForm": "PRIVATE_FORM",
            "statistics": {"PRIVATE_KEY": 998877},
        }]}], "results": ["PRIVATE_OUTCOME"], "Authorization": "PRIVATE_TOKEN",
    }})
    report = response_shape(raw)
    event = report["shape"]["fields"]["racecardEvent"]["fields"]
    market = event["markets"]["sample"][0]["fields"]
    runner = market["selections"]["sample"][0]["fields"]
    assert market["numPlaces"] == {"type": "number"}
    assert runner["isOut"] == {"type": "boolean"}
    assert runner["prices"]["sample"][0]["fields"] == {
        "priceCode": {"type": "string"}, "winPrice": {"type": "number"},
        "placePrice": {"type": "number"},
    }
    text = json.dumps(report)
    for private in ("PRIVATE", "12345678", "horse", "PRICED", '"L"',
                    "8.75", "2.65", "998877", "shortForm", "statistics", "results"):
        assert private not in text
    assert report["truncated"] is False


def test_complete_response_is_schema_only_and_body_is_read_once():
    rec = inspector()
    rec.navigate()
    complete(rec)
    calls = []

    def read(rid):
        calls.append(rid)
        return {"body": '{"markets":[{"price":3.4}]}'}

    rec.inspect_bodies(read)
    rec.inspect_bodies(read)
    assert calls == ["1"]
    report = rec.report()
    assert report["provider_timestamp"] is None
    assert report["responses"][0]["body"]["state"] == "shape_only"
    assert report["responses"][0]["request"]["observed_at"]
    assert report["responses"][0]["completed"]["elapsed_seconds"] >= 0
    assert "3.4" not in json.dumps(report["responses"][0]["body"])


@pytest.mark.parametrize(
    "case",
    [
        "missing_request",
        "partial",
        "out_of_order",
        "navigation",
        "redirect",
        "expired",
        "cached",
        "worker",
        "denied",
    ],
)
def test_ambiguous_response_never_reads_body(case):
    rec = inspector()
    if case == "missing_request":
        emit(rec, "responseReceived", response={"url": URL, "status": 200})
    elif case == "partial":
        emit(rec, "requestWillBeSent", request={"url": URL, "method": "GET"})
    elif case == "out_of_order":
        emit(rec, "loadingFinished", encodedDataLength=100)
        emit(rec, "requestWillBeSent", request={"url": URL, "method": "GET"})
    else:
        complete(
            rec,
            **(
                {"fromDiskCache": True}
                if case == "cached"
                else (
                    {"fromServiceWorker": True}
                    if case == "worker"
                    else {"status": 403} if case == "denied" else {}
                )
            ),
        )
        if case == "navigation":
            rec.navigate()
        elif case == "redirect":
            emit(rec, "requestWillBeSent", request={"url": URL, "method": "GET"})
        elif case == "expired":
            rec.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    rec.inspect_bodies(lambda _: pytest.fail("ambiguous body read"))
    assert rec.report()["body_reads"] == 0


def test_bounded_payload_and_stream_evidence():
    assert response_shape("x" * 1_000_001)["state"] == "body_limit"
    assert response_shape("not json")["state"] == "not_json"
    assert response_shape(json.dumps([1] * 30))["truncated"]
    rec = inspector()
    emit(rec, "webSocketFrameReceived", response={"payloadData": "protected result"})
    for i in range(70):
        rec.observe(
            {
                "method": "Network.requestWillBeSent",
                "params": {"requestId": str(i), "request": {"url": URL, "method": "GET"}},
            }
        )
    assert len(rec.report()["responses"]) == 64
    assert rec.report()["dropped_requests"] == 6
    assert rec.report()["websocket_frame_count"] == 1
    assert "protected" not in json.dumps(rec.report())


def test_body_errors_and_base64_are_finite_and_do_not_leak():
    rec = inspector()
    complete(rec)

    def read(_):
        raise RuntimeError("secret payload")

    rec.inspect_bodies(read)
    assert rec.report()["responses"][0]["body"] == {"state": "body_unavailable"}
    assert "secret" not in json.dumps(rec.report())
    rec = inspector()
    complete(rec)
    rec.inspect_bodies(lambda _: {"body": "c2VjcmV0", "base64Encoded": True})
    assert rec.report()["responses"][0]["body"]["state"] == "base64_not_inspected"


@pytest.mark.parametrize("denial", [False, True])
def test_actual_guarded_browser_keeps_one_operation_and_denial_hold(tmp_path, monkeypatch, denial):
    from tests.fixtures.freshness_transport.fake_cdp import BrowserTransport
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked
    from utils.sportsbet_browser import create_sportsbet_driver

    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(tmp_path / "access.json"))
    gate = SportsbetAccess()
    gate.initialize(access_basis={"status": "permitted", "reference": "synthetic fixture only"})
    with gate.locked():
        state = gate.read()
        state["operating_policy"] = {
            "reference": "synthetic fixture only",
            "python_per_60_seconds": 10,
            "browser_per_60_seconds": 1,
            "browser_navigation_cap": 2,
        }
        gate.write(state)
    transport = BrowserTransport()
    commands = []
    original_execute = transport.execute

    def execute(command):
        # Inspect and reconstruct the generator so the real guard seam is used.
        from utils.sportsbet_browser import cdp

        request = next(command)
        commands.append(request["method"])
        if request["method"] == "Network.getResponseBody":
            return {"body": '{"runners":[{"win":2.2,"place":1.1}]}'}
        return original_execute(cdp(request["method"], request["params"]))

    transport.execute = execute

    class Driver:
        service = SimpleNamespace(process=transport.process)

        def start_devtools(self):
            return None, transport

        def get(self, url):
            for method, params in [
                ("requestWillBeSent", {"request": {"url": URL, "method": "GET"}}),
                (
                    "responseReceived",
                    {
                        "type": "XHR",
                        "response": {
                            "url": URL,
                            "status": 429 if denial else 200,
                            "mimeType": "application/json",
                            "headers": {"Set-Cookie": "secret"},
                        },
                    },
                ),
                ("loadingFinished", {"encodedDataLength": 100}),
            ]:
                transport._ws.on_message(
                    None,
                    json.dumps(
                        {"method": "Network." + method, "params": {"requestId": "1", **params}}
                    ),
                )

        def get_log(self, name):
            return []

        def quit(self):
            transport.quit()

    rec = inspector()
    driver = create_sportsbet_driver(Driver, response_inspection=rec)
    try:
        if denial:
            with pytest.raises(SportsbetAccessBlocked):
                driver.get("https://www.sportsbet.com.au/fixture")
            with pytest.raises(SportsbetAccessBlocked):
                driver.sportsbet_inspect_response_shapes()
            assert "Network.getResponseBody" not in commands
        else:
            driver.get("https://www.sportsbet.com.au/fixture")
            report = driver.sportsbet_inspect_response_shapes()
            assert report["responses"][0]["body"]["state"] == "shape_only"
            assert commands.count("Network.getResponseBody") == 1
    finally:
        driver.quit()
    state = gate.read()
    assert len(state["operations"]) == 1
    assert state["operations"][0]["kind"] == "browser"
    assert state["recovery_attempts"] == 0
    if denial:
        assert state["phase"] == "COOLDOWN"
        with pytest.raises(SportsbetAccessBlocked):
            create_sportsbet_driver(
                lambda: pytest.fail("held factory"), response_inspection=inspector()
            )
    assert "secret" not in json.dumps(rec.report())


def test_body_finishing_after_window_expiry_is_explicit():
    rec = inspector()
    complete(rec)

    def read(_):
        rec.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        return {"body": '{"win":2.2}'}

    rec.inspect_bodies(read)
    assert rec.report()["responses"][0]["body"]["state"] == "expired_during_inspection"


def test_schema_report_cannot_pass_actual_receipt_verifier():
    from src.predictor.on_demand import normalize_validation_receipt, PredictionBlocked

    rec = inspector()
    complete(rec)
    rec.inspect_bodies(lambda _: {"body": '{"runners":[{"win":2.2,"place":1.1}]}'})
    with pytest.raises(PredictionBlocked, match="MARKET_UNAVAILABLE"):
        normalize_validation_receipt(
            race_id="synthetic-race",
            captured_at=datetime.now(timezone.utc),
            validation=rec.report(),
            source_kind="browser_response_inspection",
        )


def test_four_body_cap_never_adds_a_replay_or_retries():
    rec = inspector()
    for i in range(6):
        for method, params in [
            ("requestWillBeSent", {"request": {"url": URL, "method": "GET"}}),
            (
                "responseReceived",
                {"response": {"url": URL, "status": 200, "mimeType": "application/json"}},
            ),
            ("loadingFinished", {"encodedDataLength": 100}),
        ]:
            rec.observe({"method": "Network." + method, "params": {"requestId": str(i), **params}})
    calls = []

    def read(rid):
        calls.append(rid)
        return {"body": "{}"}

    rec.inspect_bodies(read)
    rec.inspect_bodies(read)
    assert len(calls) == rec.report()["body_reads"] == 4


@pytest.mark.parametrize("case", ["duplicate_response", "failed_then_response", "foreign_redirect"])
def test_response_identity_or_order_change_is_terminal(case):
    rec = inspector()
    complete(rec)
    if case == "foreign_redirect":
        emit(
            rec,
            "requestWillBeSent",
            request={"url": "https://other.invalid/account", "method": "GET"},
        )
    else:
        if case == "failed_then_response":
            emit(rec, "loadingFailed")
        emit(
            rec,
            "responseReceived",
            response={"url": URL, "status": 200, "mimeType": "application/json"},
        )
        emit(rec, "loadingFinished", encodedDataLength=100)
    rec.inspect_bodies(lambda _: pytest.fail("invalid sequence read"))
    assert rec.report()["body_reads"] == 0
