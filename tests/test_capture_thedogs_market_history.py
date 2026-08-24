import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest

from scripts import capture_thedogs_market_history as subject
from scripts.capture_thedogs_market_history import (
    CaptureError,
    TimedResponse,
    canonical_json_bytes,
    capture_snapshot,
    parse_source_runners,
    sha256_bytes,
    validate_response,
)

RACE_ID = "race-immutable-0001"
RACE_URL = "https://www.thedogs.com.au/racing/meadows/2026-06-01/1/test-race"
ODDS_URL = f"{RACE_URL}/odds"
JUMP = datetime(2026, 6, 1, 1, 0, tzinfo=timezone.utc)


def source_html(jump: datetime = JUMP) -> bytes:
    return f"""
    <html><body>
      <formatted-time data-format="datetime_short" data-timestamp="{int(jump.timestamp())}">11:00</formatted-time>
      <table class="race-runners"><tbody data-content-url="/dogs/runner/101/odds">
        <tr class="race-runner"><td><sprite-svg name="rug_1"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Alpha<span>29.1</span></div></td>
        <td><runner-odd data-runner-id="101"></runner-odd></td></tr>
      </tbody><tbody data-content-url="/dogs/runner/102/odds">
        <tr class="race-runner"><td><sprite-svg name="rug_2"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Beta<span>29.2</span></div></td>
        <td><runner-odd data-runner-id="102"></runner-odd></td></tr>
      </tbody><tbody data-content-url="/dogs/runner/103/odds">
        <tr class="race-runner race-runner--scratched"><td><sprite-svg name="rug_3"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Gamma<span>29.3</span></div></td>
        <td><runner-odd data-runner-id="103"></runner-odd></td><td>SCR</td></tr>
      </tbody><tbody>
        <tr class="race-runner race-runner--scratched"><td><sprite-svg name="rug_4"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Vacant Box</div></td><td>SCR</td></tr>
      </tbody></table>
    </body></html>
    """.encode()


def api_payload(
    *,
    provider: bool = True,
    missing_runner: str | None = None,
    mismatched_quote_id: bool = False,
) -> bytes:
    def quote(runner_id: str, box: int | None, price: float):
        return {
            "runner_id": int(runner_id),
            "run_box": box,
            "price": price,
            "preferred_bookmaker_id": 63 if provider else None,
            "bookmaker": (
                {"id": 63, "code": "ladbrokes", "name": "Ladbrokes"}
                if provider
                else None
            ),
            "market": {"code": "fixed_win", "race_id": 9001},
        }

    rows = {
        "101": [quote("101", 1, 2.5)],
        "102": [quote("102", 2, 3.5)],
        "103": [quote("103", None, 0.0)],
    }
    if missing_runner:
        rows.pop(missing_runner)
    if mismatched_quote_id:
        rows["102"][0]["runner_id"] = 999
    return json.dumps({"runner_odds": rows}).encode()


def replacement_source_html(
    *,
    from_box_text: str = "(from box 2)",
    replacement_active: bool = True,
) -> bytes:
    replacement_classes = "race-runner" if replacement_active else (
        "race-runner race-runner--scratched"
    )
    return f"""
    <html><body>
      <formatted-time data-format="datetime_short" data-timestamp="{int(JUMP.timestamp())}">11:00</formatted-time>
      <table class="race-runners"><tbody data-content-url="/dogs/runner/101/odds">
        <tr class="race-runner"><td><sprite-svg name="rug_1"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Alpha<span>29.1</span></div></td>
        <td><runner-odd data-runner-id="101"></runner-odd></td></tr>
      </tbody><tbody data-content-url="/dogs/runner/102/odds">
        <tr class="race-runner race-runner--scratched"><td><sprite-svg name="rug_2"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Beta<span>29.2</span></div></td>
        <td><runner-odd data-runner-id="102"></runner-odd></td><td>SCR</td></tr>
      </tbody><tbody data-content-url="/dogs/runner/104/odds">
        <tr class="{replacement_classes}"><td><sprite-svg name="rug_9"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Reserve<span>29.4</span>
        <span class="race-runners__name__box">{from_box_text}</span></div></td>
        <td><runner-odd data-runner-id="104"></runner-odd></td></tr>
      </tbody></table>
    </body></html>
    """.encode()


def replacement_api_payload(
    *,
    replacement_run_box: int | None = 2,
    original_price: float = 0.0,
) -> bytes:
    def quote(runner_id: str, box: int | None, price: float):
        return {
            "runner_id": int(runner_id),
            "run_box": box,
            "price": price,
            "preferred_bookmaker_id": 63,
            "bookmaker": {"id": 63, "code": "ladbrokes", "name": "Ladbrokes"},
            "market": {"code": "fixed_win", "race_id": 9001},
        }

    return json.dumps(
        {
            "runner_odds": {
                "101": [quote("101", 1, 2.5)],
                "102": [quote("102", 2, original_price)],
                "104": [quote("104", replacement_run_box, 3.5)],
            }
        }
    ).encode()


def mismatched_normal_api_payload() -> bytes:
    payload = json.loads(api_payload())
    payload["runner_odds"]["102"][0]["run_box"] = 3
    return json.dumps(payload).encode()


class FakeResponse:
    def __init__(
        self, url: str, content: bytes, content_type: str, server_time: datetime
    ):
        self.url = url
        self.content = content
        self.status_code = 200
        self.headers = {
            "Content-Type": content_type,
            "Date": format_datetime(server_time, usegmt=True),
            "X-Request-Id": "fixture-request",
        }


class FakeSession:
    def __init__(
        self,
        *,
        provider: bool = True,
        missing_runner: str | None = None,
        mismatched_quote_id: bool = False,
        server_time: datetime | None = None,
        source_body: bytes | None = None,
        api_body: bytes | None = None,
    ):
        self.provider = provider
        self.missing_runner = missing_runner
        self.server_time = server_time or (JUMP - timedelta(minutes=120))
        self.mismatched_quote_id = mismatched_quote_id
        self.source_body = source_body or source_html()
        self.api_body = api_body
        self.calls: list[str] = []
        self.request_headers: list[dict[str, str]] = []

    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False
        self.calls.append(url)
        self.request_headers.append(dict(kwargs["headers"]))
        if url.endswith("/racing/2026-06-01"):
            return FakeResponse(
                url,
                b"<html>meeting</html>",
                "text/html; charset=utf-8",
                self.server_time,
            )
        if url == RACE_URL:
            return FakeResponse(
                url, self.source_body, "text/html; charset=utf-8", self.server_time
            )
        if url == ODDS_URL:
            return FakeResponse(
                url, self.source_body, "text/html; charset=utf-8", self.server_time
            )
        if "/api/runners/odds?" in url:
            return FakeResponse(
                url,
                self.api_body
                or api_payload(
                        provider=self.provider,
                        missing_runner=self.missing_runner,
                        mismatched_quote_id=self.mismatched_quote_id,
                    ),
                "application/json; charset=utf-8",
                self.server_time,
            )
        raise AssertionError(url)


class FakeClock:
    def __init__(self, start: datetime):
        self.value = start

    def __call__(self):
        value = self.value
        self.value += timedelta(milliseconds=100)
        return value


def plan(**overrides):
    payload = {
        "schema_version": "thedogs_market_snapshot_plan_v1",
        "race_id": RACE_ID,
        "race_url": RACE_URL,
        "odds_url": ODDS_URL,
        "jump_timestamp": JUMP.isoformat(),
        "nominal_window": "T-120",
        "expected_active_runner_ids": ["101", "102"],
    }
    payload.update(overrides)
    return payload


def capture(
    tmp_path: Path,
    *,
    plan_payload=None,
    provider: bool = True,
    missing_runner: str | None = None,
    mismatched_quote_id: bool = False,
    source_body: bytes | None = None,
    api_body: bytes | None = None,
):
    current = JUMP - timedelta(minutes=120)
    session = FakeSession(
        provider=provider,
        missing_runner=missing_runner,
        server_time=current,
        mismatched_quote_id=mismatched_quote_id,
        source_body=source_body,
        api_body=api_body,
    )
    result = capture_snapshot(
        plan_payload or plan(),
        tmp_path / "capture" / "snapshot",
        session=session,
        current_time=current,
        clock=FakeClock(current),
        repo_root=tmp_path,
    )
    return result, session


def test_exact_odds_url_is_required(tmp_path):
    with pytest.raises(CaptureError, match="exact_thedogs_odds_url_required"):
        capture(tmp_path, plan_payload=plan(odds_url=RACE_URL))


def test_http_date_and_age_interval_binds_cached_response_time():
    request_time = JUMP - timedelta(minutes=120)
    response = TimedResponse(
        requested_url=ODDS_URL,
        request_start_utc=request_time,
        request_end_utc=request_time + timedelta(milliseconds=100),
        final_url=ODDS_URL,
        status_code=200,
        headers={
            "content-type": "text/html; charset=utf-8",
            "date": format_datetime(request_time - timedelta(seconds=18), usegmt=True),
            "age": "18",
        },
        body=b"source",
    )

    validate_response(response, exact_url=ODDS_URL, content_type_prefix="text/html")


def test_internal_race_id_is_preserved_separately_from_source_alias(tmp_path):
    result, _session = capture(tmp_path)

    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert receipt["race_id"] == RACE_ID
    assert receipt["race_identity"]["venue_slug"] == "meadows"
    assert receipt["race_identity"]["race_number"] == 1


def test_expected_active_native_runner_set_is_required(tmp_path):
    with pytest.raises(CaptureError, match="expected_active_runner_ids_required"):
        capture(tmp_path, plan_payload=plan(expected_active_runner_ids=[]))


def test_complete_active_field_and_native_ids_are_captured(tmp_path):
    result, session = capture(tmp_path)

    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert result["status"] == "CAPTURE_ACCEPTED"
    assert receipt["active_native_runner_ids"] == ["101", "102"]
    assert receipt["all_native_runner_ids"] == ["101", "102", "103"]
    assert [row["current_price"] for row in receipt["runners"]] == [2.5, 3.5, None]
    assert receipt["runners"][0]["page_box"] == 1
    assert receipt["runners"][0]["page_effective_box"] is None
    assert receipt["runners"][0]["api_run_box"] == 1
    assert receipt["runners"][0]["effective_box"] == 1
    assert receipt["provider"]["classification"] == "provider_explicit"
    assert receipt["provider"]["code"] == "ladbrokes"
    assert receipt["open_low_high_are_temporal_observations"] is False
    assert len(session.calls) == 4


def test_retained_race_page_identity_binding_uses_exactly_two_requests():
    observed = JUMP - timedelta(minutes=20)
    session = FakeSession(server_time=observed)
    race_page = TimedResponse(
        requested_url=RACE_URL,
        request_start_utc=observed - timedelta(milliseconds=100),
        request_end_utc=observed,
        final_url=RACE_URL,
        status_code=200,
        headers={
            "content-type": "text/html; charset=utf-8",
            "date": format_datetime(observed, usegmt=True),
        },
        body=source_html(),
    )

    evidence = subject.capture_native_identity_from_retained_race_page(
        session=session,
        race_page=race_page,
        expected_active_runner_boxes=[("101", 1), ("102", 2)],
        expected_jump_utc=JUMP,
        current_time=observed + timedelta(seconds=1),
        clock=FakeClock(observed + timedelta(milliseconds=100)),
    )

    assert session.calls == [ODDS_URL, subject._api_url(parse_source_runners(source_html()))]
    assert evidence["source_native_race_id"] == "9001"
    assert evidence["active_native_runner_ids"] == ["101", "102"]
    assert evidence["request_accounting"] == {
        "logical_requests": 2,
        "shared_session_max_attempts_per_request": 3,
        "worst_case_wire_attempts": 6,
    }
    assert evidence["evidence_sha256"] == sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        )
    )


def test_primary_identity_request_accounting_matches_shared_retry_bound():
    from utils.http_client import get_shared_session

    retries = get_shared_session().get_adapter("https://").max_retries
    assert retries.total == 2
    assert 1 + retries.total == 3
    assert 2 * (1 + retries.total) == 6


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        (
            "capture_start_utc",
            "2026-06-01T08:39:00Z",
            "persisted_capture_start_mismatch",
        ),
        (
            "capture_end_utc",
            "2026-06-01T08:41:00Z",
            "persisted_capture_end_mismatch",
        ),
        (
            "provider",
            {
                "classification": "provider_unknown",
                "id": None,
                "code": None,
                "name": None,
                "source": "provider_not_explicit_in_source_payload",
            },
            "persisted_provider_projection_mismatch",
        ),
    ],
)
def test_primary_identity_revalidation_rejects_rehashed_provenance_tampering(
    field,
    replacement,
    reason,
):
    observed = JUMP - timedelta(minutes=20)
    evidence = subject.capture_native_identity_from_retained_race_page(
        session=FakeSession(server_time=observed),
        race_page=TimedResponse(
            requested_url=RACE_URL,
            request_start_utc=observed - timedelta(milliseconds=100),
            request_end_utc=observed,
            final_url=RACE_URL,
            status_code=200,
            headers={
                "content-type": "text/html; charset=utf-8",
                "date": format_datetime(observed, usegmt=True),
            },
            body=source_html(),
        ),
        expected_active_runner_boxes=[("101", 1), ("102", 2)],
        expected_jump_utc=JUMP,
        current_time=observed + timedelta(seconds=1),
        clock=FakeClock(observed + timedelta(milliseconds=100)),
    )
    evidence[field] = replacement
    evidence["evidence_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        )
    )

    valid, observed_reason = subject.validate_primary_native_identity_evidence(
        evidence,
        expected_race_url=RACE_URL,
        expected_native_race_id="9001",
        expected_active_runner_boxes=[("101", 1), ("102", 2)],
        metadata_captured_at=evidence["odds_api_http"]["request_end_utc"],
    )

    assert valid is False
    assert observed_reason == reason


def test_retained_race_page_identity_rejects_non_numeric_native_race_id():
    observed = JUMP - timedelta(minutes=20)
    payload = json.loads(api_payload())
    for quotes in payload["runner_odds"].values():
        quotes[0]["market"]["race_id"] = "race-nine-thousand-one"
    session = FakeSession(
        server_time=observed,
        api_body=json.dumps(payload).encode(),
    )
    race_page = TimedResponse(
        requested_url=RACE_URL,
        request_start_utc=observed - timedelta(milliseconds=100),
        request_end_utc=observed,
        final_url=RACE_URL,
        status_code=200,
        headers={
            "content-type": "text/html; charset=utf-8",
            "date": format_datetime(observed, usegmt=True),
        },
        body=source_html(),
    )

    with pytest.raises(CaptureError, match="native_race_identity_invalid"):
        subject.capture_native_identity_from_retained_race_page(
            session=session,
            race_page=race_page,
            expected_active_runner_boxes=[("101", 1), ("102", 2)],
            expected_jump_utc=JUMP,
            current_time=observed + timedelta(seconds=1),
            clock=FakeClock(observed + timedelta(milliseconds=100)),
        )


def test_retained_race_page_identity_rejects_partial_native_race_id():
    observed = JUMP - timedelta(minutes=20)
    payload = json.loads(api_payload())
    del payload["runner_odds"]["102"][0]["market"]["race_id"]
    session = FakeSession(
        server_time=observed,
        api_body=json.dumps(payload).encode(),
    )
    race_page = TimedResponse(
        requested_url=RACE_URL,
        request_start_utc=observed - timedelta(milliseconds=100),
        request_end_utc=observed,
        final_url=RACE_URL,
        status_code=200,
        headers={
            "content-type": "text/html; charset=utf-8",
            "date": format_datetime(observed, usegmt=True),
        },
        body=source_html(),
    )

    with pytest.raises(CaptureError, match="native_race_identity_incomplete"):
        subject.capture_native_identity_from_retained_race_page(
            session=session,
            race_page=race_page,
            expected_active_runner_boxes=[("101", 1), ("102", 2)],
            expected_jump_utc=JUMP,
            current_time=observed + timedelta(seconds=1),
            clock=FakeClock(observed + timedelta(milliseconds=100)),
        )


@pytest.mark.parametrize(
    ("expected_runner_boxes", "reason"),
    [
        ([("101", 1)], "expected_active_runner_boxes_invalid"),
        (
            [("101", 1), ("101", 2)],
            "expected_active_runner_boxes_invalid",
        ),
        (
            [("101", 1), ("runner-two", 2)],
            "expected_active_runner_boxes_invalid",
        ),
        (
            [("101", 1), ("999", 2)],
            "expected_native_runner_set_mismatch",
        ),
    ],
)
def test_retained_race_page_identity_rejects_partial_or_mismatched_runner_ids(
    expected_runner_boxes,
    reason,
):
    observed = JUMP - timedelta(minutes=20)
    session = FakeSession(server_time=observed)
    race_page = TimedResponse(
        requested_url=RACE_URL,
        request_start_utc=observed - timedelta(milliseconds=100),
        request_end_utc=observed,
        final_url=RACE_URL,
        status_code=200,
        headers={
            "content-type": "text/html; charset=utf-8",
            "date": format_datetime(observed, usegmt=True),
        },
        body=source_html(),
    )

    with pytest.raises(CaptureError, match=reason):
        subject.capture_native_identity_from_retained_race_page(
            session=session,
            race_page=race_page,
            expected_active_runner_boxes=expected_runner_boxes,
            expected_jump_utc=JUMP,
            current_time=observed + timedelta(seconds=1),
            clock=FakeClock(observed + timedelta(milliseconds=100)),
        )


def test_retained_race_page_identity_rejects_ambiguous_native_race_id():
    observed = JUMP - timedelta(minutes=20)
    payload = json.loads(api_payload())
    payload["runner_odds"]["102"][0]["market"]["race_id"] = 9002
    session = FakeSession(
        server_time=observed,
        api_body=json.dumps(payload).encode(),
    )
    race_page = TimedResponse(
        requested_url=RACE_URL,
        request_start_utc=observed - timedelta(milliseconds=100),
        request_end_utc=observed,
        final_url=RACE_URL,
        status_code=200,
        headers={
            "content-type": "text/html; charset=utf-8",
            "date": format_datetime(observed, usegmt=True),
        },
        body=source_html(),
    )

    with pytest.raises(CaptureError, match="native_race_identity_not_unique"):
        subject.capture_native_identity_from_retained_race_page(
            session=session,
            race_page=race_page,
            expected_active_runner_boxes=[("101", 1), ("102", 2)],
            expected_jump_utc=JUMP,
            current_time=observed + timedelta(seconds=1),
            clock=FakeClock(observed + timedelta(milliseconds=100)),
        )


def test_retained_race_page_identity_rejects_primary_effective_box_mismatch():
    observed = JUMP - timedelta(minutes=20)
    race_page = TimedResponse(
        requested_url=RACE_URL,
        request_start_utc=observed - timedelta(milliseconds=100),
        request_end_utc=observed,
        final_url=RACE_URL,
        status_code=200,
        headers={
            "content-type": "text/html; charset=utf-8",
            "date": format_datetime(observed, usegmt=True),
        },
        body=source_html(),
    )

    with pytest.raises(
        CaptureError,
        match="expected_native_runner_effective_box_mismatch",
    ):
        subject.capture_native_identity_from_retained_race_page(
            session=FakeSession(server_time=observed),
            race_page=race_page,
            expected_active_runner_boxes=[("101", 1), ("102", 3)],
            expected_jump_utc=JUMP,
            current_time=observed + timedelta(seconds=1),
            clock=FakeClock(observed + timedelta(milliseconds=100)),
        )


def test_retained_race_page_identity_rejects_stale_or_postjump_evidence():
    observed = JUMP - timedelta(minutes=20)
    race_page = TimedResponse(
        requested_url=RACE_URL,
        request_start_utc=observed - timedelta(milliseconds=100),
        request_end_utc=observed,
        final_url=RACE_URL,
        status_code=200,
        headers={
            "content-type": "text/html; charset=utf-8",
            "date": format_datetime(observed, usegmt=True),
        },
        body=source_html(),
    )
    with pytest.raises(CaptureError, match="native_identity_evidence_stale"):
        subject.capture_native_identity_from_retained_race_page(
            session=FakeSession(server_time=observed),
            race_page=race_page,
            expected_active_runner_boxes=[("101", 1), ("102", 2)],
            expected_jump_utc=JUMP,
            current_time=observed + timedelta(seconds=1201),
            clock=FakeClock(observed + timedelta(milliseconds=100)),
        )

    with pytest.raises(CaptureError, match="capture_not_strictly_prejump"):
        subject.capture_native_identity_from_retained_race_page(
            session=FakeSession(server_time=JUMP),
            race_page=race_page,
            expected_active_runner_boxes=[("101", 1), ("102", 2)],
            expected_jump_utc=JUMP,
            current_time=observed + timedelta(seconds=1),
            clock=FakeClock(JUMP),
        )

def test_native_odds_api_uses_frontend_xhr_request_header(tmp_path):
    _result, session = capture(tmp_path)

    assert all(
        "X-Requested-With" not in headers for headers in session.request_headers[:3]
    )
    assert session.request_headers[3]["X-Requested-With"] == "XMLHttpRequest"


def test_incomplete_active_field_is_rejected(tmp_path):
    with pytest.raises(CaptureError, match="active_runner_current_price_missing"):
        capture(tmp_path, missing_runner="102")


def test_unexpected_api_native_runner_is_rejected(tmp_path):
    payload = json.loads(api_payload())
    payload["runner_odds"]["999"] = [
        {
            "runner_id": 999,
            "run_box": 8,
            "price": 9.0,
            "preferred_bookmaker_id": 63,
            "bookmaker": {"id": 63, "code": "ladbrokes", "name": "Ladbrokes"},
            "market": {"code": "fixed_win", "race_id": 9001},
        }
    ]

    with pytest.raises(CaptureError, match="odds_api_native_runner_set_mismatch"):
        capture(tmp_path, api_body=json.dumps(payload).encode())


def test_native_id_expectation_mismatch_is_rejected(tmp_path):
    with pytest.raises(CaptureError, match="expected_native_runner_set_mismatch"):
        capture(
            tmp_path,
            plan_payload=plan(expected_active_runner_ids=["101", "999"]),
        )


def test_api_native_runner_id_mismatch_is_rejected(tmp_path):
    with pytest.raises(CaptureError, match="native_runner_quote_id_mismatch"):
        capture(tmp_path, mismatched_quote_id=True)


def test_native_id_replacement_resolves_only_from_explicit_matching_boxes(tmp_path):
    result, _session = capture(
        tmp_path,
        plan_payload=plan(expected_active_runner_ids=["101", "104"]),
        source_body=replacement_source_html(),
        api_body=replacement_api_payload(),
    )

    receipt = json.loads(Path(result["receipt_path"]).read_text())
    rows = {row["native_runner_id"]: row for row in receipt["runners"]}
    assert rows["104"] == {
        "active": True,
        "api_run_box": 2,
        "box": 9,
        "current_price": 3.5,
        "effective_box": 2,
        "effective_box_provenance": {
            "api_source": "thedogs_runner_odds_api_fixed_win_run_box",
            "page_source": "thedogs_odds_page_race_runners_name_box_from_box",
            "resolution": "explicit_replacement_box_match",
        },
        "native_runner_id": "104",
        "page_box": 9,
        "page_effective_box": 2,
        "provider": {"code": "ladbrokes", "id": "63", "name": "Ladbrokes"},
        "runner_name": "Reserve",
    }
    assert rows["102"]["active"] is False
    assert rows["102"]["api_run_box"] == 2
    assert rows["102"]["effective_box"] is None


@pytest.mark.parametrize(
    ("source_body", "api_body"),
    [
        (replacement_source_html(), replacement_api_payload(replacement_run_box=3)),
        (source_html(), mismatched_normal_api_payload()),
    ],
)
def test_contradictory_page_and_api_effective_boxes_fail_closed(
    tmp_path, source_body, api_body
):
    expected_ids = ["101", "104"] if b"runner/104" in source_body else ["101", "102"]
    with pytest.raises(CaptureError, match="effective_box_source_conflict"):
        capture(
            tmp_path,
            plan_payload=plan(expected_active_runner_ids=expected_ids),
            source_body=source_body,
            api_body=api_body,
        )


def test_duplicate_active_effective_box_from_replacement_fails_closed(tmp_path):
    body = replacement_source_html().replace(
        b'race-runner race-runner--scratched"><td><sprite-svg name="rug_2"',
        b'race-runner"><td><sprite-svg name="rug_2"',
    )
    payload = json.loads(replacement_api_payload())
    payload["runner_odds"]["102"][0]["price"] = 4.0

    with pytest.raises(CaptureError, match="active_effective_box_not_unique"):
        capture(
            tmp_path,
            plan_payload=plan(expected_active_runner_ids=["101", "102", "104"]),
            source_body=body,
            api_body=json.dumps(payload).encode(),
        )


def test_scratched_runner_with_active_price_fails_closed(tmp_path):
    with pytest.raises(CaptureError, match="scratched_runner_has_active_price"):
        capture(
            tmp_path,
            plan_payload=plan(expected_active_runner_ids=["101", "104"]),
            source_body=replacement_source_html(),
            api_body=replacement_api_payload(original_price=4.0),
        )


def test_missing_native_id_on_active_runner_fails_closed():
    body = source_html().replace(
        b'<tbody data-content-url="/dogs/runner/101/odds">', b"<tbody>"
    )
    with pytest.raises(
        CaptureError, match="native_runner_identity_missing_from_odds_page"
    ):
        parse_source_runners(body)


def test_ambiguous_native_id_mapping_in_page_fails_closed():
    body = source_html().replace(
        b'<runner-odd data-runner-id="101">',
        b'<runner-odd data-runner-id="999">',
    )
    with pytest.raises(
        CaptureError, match="native_runner_identity_mismatch_in_odds_page"
    ):
        parse_source_runners(body)


def test_malformed_explicit_replacement_box_fails_closed():
    with pytest.raises(CaptureError, match="effective_box_source_invalid_in_odds_page"):
        parse_source_runners(replacement_source_html(from_box_text="(box 2)"))


def test_provider_unknown_is_preserved_without_rejection(tmp_path):
    result, _session = capture(tmp_path, provider=False)

    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert receipt["provider"] == {
        "classification": "provider_unknown",
        "code": None,
        "id": None,
        "name": None,
        "source": "provider_not_explicit_in_source_payload",
    }


def test_identical_rerun_is_idempotent_without_network(tmp_path):
    first, _session = capture(tmp_path)
    second_session = FakeSession()

    second = capture_snapshot(
        plan(),
        tmp_path / "capture" / "snapshot",
        session=second_session,
        current_time=JUMP - timedelta(minutes=120),
        repo_root=tmp_path,
    )

    assert first["raw_html_sha256"] == second["raw_html_sha256"]
    assert second["status"] == "SKIPPED_IDENTICAL_SNAPSHOT"
    assert second_session.calls == []


def test_identical_rerun_still_requires_expected_native_runner_set(tmp_path):
    capture(tmp_path)
    replay_plan = plan()
    replay_plan.pop("expected_active_runner_ids")

    with pytest.raises(CaptureError, match="expected_active_runner_ids_required"):
        capture_snapshot(
            replay_plan,
            tmp_path / "capture" / "snapshot",
            session=FakeSession(),
            current_time=JUMP - timedelta(minutes=120),
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt["runners"][0].update({"effective_box": 8}),
        lambda receipt: receipt["provider"].update({"code": "altered"}),
        lambda receipt: receipt["odds_api_http"].update(
            {"request_end_utc": (JUMP - timedelta(minutes=119)).isoformat()}
        ),
        lambda receipt: receipt.update(
            {"nominal_capture_utc": (JUMP - timedelta(minutes=60)).isoformat()}
        ),
    ],
)
def test_semantically_conflicting_existing_receipt_fails_closed(tmp_path, mutate):
    result, _session = capture(tmp_path)
    receipt_path = Path(result["receipt_path"])
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text())
    mutate(receipt)
    core = {key: value for key, value in receipt.items() if key != "receipt_core_sha256"}
    receipt["receipt_core_sha256"] = sha256_bytes(canonical_json_bytes(core))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_path.chmod(0o444)

    with pytest.raises(CaptureError, match="conflicting_snapshot"):
        capture_snapshot(
            plan(),
            tmp_path / "capture" / "snapshot",
            session=FakeSession(),
            current_time=JUMP - timedelta(minutes=120),
            repo_root=tmp_path,
        )


def test_legacy_v1_receipt_remains_idempotent_without_network(tmp_path):
    first, _session = capture(tmp_path)
    receipt_path = Path(first["receipt_path"])
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text())
    receipt["schema_version"] = "thedogs_market_snapshot_receipt_v1"
    legacy_keys = {
        "native_runner_id",
        "runner_name",
        "box",
        "active",
        "current_price",
        "provider",
    }
    receipt["runners"] = [
        {key: value for key, value in row.items() if key in legacy_keys}
        for row in receipt["runners"]
    ]
    core = {key: value for key, value in receipt.items() if key != "receipt_core_sha256"}
    receipt["receipt_core_sha256"] = sha256_bytes(canonical_json_bytes(core))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_path.chmod(0o444)
    second_session = FakeSession()

    second = capture_snapshot(
        plan(),
        tmp_path / "capture" / "snapshot",
        session=second_session,
        current_time=JUMP - timedelta(minutes=120),
        repo_root=tmp_path,
    )

    assert second["status"] == "SKIPPED_IDENTICAL_SNAPSHOT"
    assert second_session.calls == []


def test_conflicting_existing_snapshot_fails_closed(tmp_path):
    result, _session = capture(tmp_path)
    raw_path = Path(result["raw_html_path"])
    raw_path.chmod(0o644)
    raw_path.write_bytes(b"conflict")

    with pytest.raises(CaptureError, match="conflicting_snapshot"):
        capture_snapshot(
            plan(),
            tmp_path / "capture" / "snapshot",
            session=FakeSession(),
            current_time=JUMP - timedelta(minutes=120),
            repo_root=tmp_path,
        )


def test_missed_window_is_rejected_before_network(tmp_path):
    session = FakeSession()

    with pytest.raises(CaptureError, match="nominal_window_missed"):
        capture_snapshot(
            plan(),
            tmp_path / "capture" / "snapshot",
            session=session,
            current_time=JUMP - timedelta(minutes=118),
            repo_root=tmp_path,
        )

    assert session.calls == []


@pytest.mark.parametrize(
    "plan_payload",
    [
        plan(early_tolerance_seconds=31),
        plan(late_tolerance_seconds=91),
        plan(early_tolerance_seconds=-1),
        plan(late_tolerance_seconds=-1),
    ],
)
def test_window_tolerance_cannot_weaken_prescribed_interval(tmp_path, plan_payload):
    session = FakeSession()

    with pytest.raises(CaptureError, match="window_tolerance_invalid"):
        capture_snapshot(
            plan_payload,
            tmp_path / "capture" / "snapshot",
            session=session,
            current_time=JUMP - timedelta(minutes=120),
            repo_root=tmp_path,
        )

    assert session.calls == []
