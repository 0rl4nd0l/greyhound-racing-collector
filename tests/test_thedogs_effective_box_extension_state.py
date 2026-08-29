from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.thedogs_effective_box_extension_state import (
    CLAIM_NAME,
    RESPONSE_NAME,
    STATE_NAME,
    DiscoveryAcquisition,
    DiscoveryStateError,
    load_or_acquire_discovery,
)

NOW = datetime(2026, 8, 17, 0, 0, tzinfo=timezone.utc)
DISCOVERY_URL = "https://www.thedogs.com.au/racing/2026-08-18"


def contract(**overrides):
    payload = {
        "cohort_id": "effective-box-extension-2026-08-18-v1",
        "cohort_date": "2026-08-18",
        "discovery_url": DISCOVERY_URL,
        "candidate_selection_rule": "fixture-three-race-pool",
        "preflight_lead_minutes": 140,
        "nominal_windows": ["T-120", "T-60", "T-30", "T-10", "T-2"],
    }
    payload.update(overrides)
    return payload


def acquisition():
    return DiscoveryAcquisition(
        requested_url=DISCOVERY_URL,
        final_url=DISCOVERY_URL,
        request_start_utc=NOW,
        request_end_utc=NOW + timedelta(milliseconds=125),
        status_code=200,
        headers={"Content-Type": "text/html; charset=utf-8", "Date": "fixture"},
        body=b"<html>immutable discovery fixture</html>",
    )


def candidate_rows():
    return [
        {
            "race_url": "https://www.thedogs.com.au/racing/alpha/2026-08-18/1/a",
            "odds_url": "https://www.thedogs.com.au/racing/alpha/2026-08-18/1/a/odds",
            "jump_timestamp": "2026-08-18T04:00:00Z",
        },
        {
            "race_url": "https://www.thedogs.com.au/racing/bravo/2026-08-18/2/b",
            "odds_url": "https://www.thedogs.com.au/racing/bravo/2026-08-18/2/b/odds",
            "jump_timestamp": "2026-08-18T04:05:00Z",
        },
        {
            "race_url": "https://www.thedogs.com.au/racing/charlie/2026-08-18/3/c",
            "odds_url": "https://www.thedogs.com.au/racing/charlie/2026-08-18/3/c/odds",
            "jump_timestamp": "2026-08-18T04:10:00Z",
        },
    ]


def parse_candidates(_body):
    return candidate_rows()


def attributes(path: Path):
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "mode": path.stat().st_mode & 0o777,
    }


def test_interruption_restart_before_readiness_reuses_discovery_without_network(tmp_path):
    calls = []

    def first_acquire():
        calls.append("network")
        return acquisition()

    first = load_or_acquire_discovery(
        tmp_path,
        contract=contract(),
        acquire=first_acquire,
        parse_candidates=parse_candidates,
        clock=lambda: NOW,
    )
    claim_before = attributes(tmp_path / CLAIM_NAME)
    response_before = attributes(tmp_path / RESPONSE_NAME)
    state_before = attributes(tmp_path / STATE_NAME)

    # Simulated manual interruption: readiness is never called after persistence.
    def duplicate_network_is_forbidden():
        calls.append("duplicate")
        raise AssertionError("restart attempted duplicate discovery")

    resumed = load_or_acquire_discovery(
        tmp_path,
        contract=contract(),
        acquire=duplicate_network_is_forbidden,
        parse_candidates=parse_candidates,
        clock=lambda: NOW + timedelta(hours=1),
    )

    assert calls == ["network"]
    assert first.network_requests == 1
    assert first.resumed is False
    assert resumed.network_requests == 0
    assert resumed.resumed is True
    assert resumed.state == first.state
    assert resumed.state["contract"]["cohort_id"] == "effective-box-extension-2026-08-18-v1"
    assert resumed.state["preflight_at_utc"] == "2026-08-18T01:40:00Z"
    assert resumed.response["request_attempts"] == 1
    assert resumed.response["request_retries"] == 0
    assert attributes(tmp_path / CLAIM_NAME) == claim_before
    assert attributes(tmp_path / RESPONSE_NAME) == response_before
    assert attributes(tmp_path / STATE_NAME) == state_before
    assert claim_before["mode"] == response_before["mode"] == state_before["mode"] == 0o444


def test_existing_discovery_with_contract_drift_fails_before_network(tmp_path):
    load_or_acquire_discovery(
        tmp_path,
        contract=contract(),
        acquire=acquisition,
        parse_candidates=parse_candidates,
        clock=lambda: NOW,
    )
    calls = []

    with pytest.raises(DiscoveryStateError, match="discovery_contract_mismatch"):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(preflight_lead_minutes=139),
            acquire=lambda: calls.append("network"),
            parse_candidates=lambda _body: calls.append("parse"),
            clock=lambda: NOW,
        )

    assert calls == []


def test_orphan_request_claim_blocks_restart_without_network(tmp_path):
    calls = []

    def interrupted_acquire():
        calls.append("network")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=interrupted_acquire,
            parse_candidates=parse_candidates,
            clock=lambda: NOW,
        )
    assert (tmp_path / CLAIM_NAME).exists()
    assert not (tmp_path / STATE_NAME).exists()

    with pytest.raises(DiscoveryStateError, match="discovery_request_outcome_ambiguous"):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=lambda: calls.append("duplicate"),
            parse_candidates=parse_candidates,
            clock=lambda: NOW + timedelta(minutes=1),
        )

    assert calls == ["network"]


def test_parser_failure_retains_raw_response_and_restarts_without_network(tmp_path):
    calls = []

    def first_acquire():
        calls.append("network")
        return acquisition()

    def ambiguous_parser(_body):
        raise DiscoveryStateError("meeting_discovery_race_ambiguous")

    with pytest.raises(DiscoveryStateError, match="meeting_discovery_race_ambiguous"):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=first_acquire,
            parse_candidates=ambiguous_parser,
            clock=lambda: NOW,
        )

    response_path = tmp_path / RESPONSE_NAME
    assert response_path.exists()
    response_before = attributes(response_path)
    assert not (tmp_path / STATE_NAME).exists()

    with pytest.raises(DiscoveryStateError, match="meeting_discovery_race_ambiguous"):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=lambda: calls.append("duplicate"),
            parse_candidates=ambiguous_parser,
            clock=lambda: NOW + timedelta(minutes=1),
        )

    assert calls == ["network"]
    assert attributes(response_path) == response_before
    assert response_before["mode"] == 0o444


def test_interruption_after_response_write_restarts_parser_without_network(tmp_path):
    calls = []

    def first_acquire():
        calls.append("network")
        return acquisition()

    def interrupted_parser(_body):
        calls.append("interrupted_parse")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=first_acquire,
            parse_candidates=interrupted_parser,
            clock=lambda: NOW,
        )

    claim_before = attributes(tmp_path / CLAIM_NAME)
    response_before = attributes(tmp_path / RESPONSE_NAME)
    assert not (tmp_path / STATE_NAME).exists()

    resumed = load_or_acquire_discovery(
        tmp_path,
        contract=contract(),
        acquire=lambda: calls.append("duplicate"),
        parse_candidates=lambda body: calls.append(body) or candidate_rows(),
        clock=lambda: NOW + timedelta(hours=1),
    )

    assert calls == [
        "network",
        "interrupted_parse",
        b"<html>immutable discovery fixture</html>",
    ]
    assert resumed.network_requests == 0
    assert resumed.resumed is True
    assert resumed.state["preflight_at_utc"] == "2026-08-18T01:40:00Z"
    assert resumed.state["response_core_sha256"] == resumed.response["response_core_sha256"]
    assert attributes(tmp_path / CLAIM_NAME) == claim_before
    assert attributes(tmp_path / RESPONSE_NAME) == response_before


def test_tampered_raw_response_fails_before_network_or_parse(tmp_path):
    load_or_acquire_discovery(
        tmp_path,
        contract=contract(),
        acquire=acquisition,
        parse_candidates=parse_candidates,
        clock=lambda: NOW,
    )
    response_path = tmp_path / RESPONSE_NAME
    response_path.chmod(0o644)
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload["response"]["body_sha256"] = "0" * 64
    response_path.write_text(json.dumps(payload), encoding="utf-8")
    calls = []

    with pytest.raises(DiscoveryStateError, match="discovery_raw_body_hash_mismatch"):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=lambda: calls.append("network"),
            parse_candidates=lambda _body: calls.append("parse"),
            clock=lambda: NOW,
        )

    assert calls == []


def test_rejected_response_is_persisted_and_restart_does_not_reacquire(tmp_path):
    calls = []

    def rejected_acquire():
        calls.append("network")
        return replace(acquisition(), status_code=503, body=b"temporarily unavailable")

    with pytest.raises(DiscoveryStateError, match="discovery_response_status_invalid"):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=rejected_acquire,
            parse_candidates=lambda _body: calls.append("parse"),
            clock=lambda: NOW,
        )

    response_path = tmp_path / RESPONSE_NAME
    response_before = attributes(response_path)
    assert not (tmp_path / STATE_NAME).exists()

    with pytest.raises(DiscoveryStateError, match="discovery_response_status_invalid"):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=lambda: calls.append("duplicate"),
            parse_candidates=lambda _body: calls.append("parse"),
            clock=lambda: NOW + timedelta(minutes=1),
        )

    assert calls == ["network"]
    assert attributes(response_path) == response_before


def test_naive_claim_clock_fails_before_request(tmp_path):
    calls = []

    with pytest.raises(DiscoveryStateError, match="request_authorized_at_utc_timezone_required"):
        load_or_acquire_discovery(
            tmp_path,
            contract=contract(),
            acquire=lambda: calls.append("network"),
            parse_candidates=parse_candidates,
            clock=lambda: NOW.replace(tzinfo=None),
        )

    assert calls == []
    assert not (tmp_path / CLAIM_NAME).exists()
