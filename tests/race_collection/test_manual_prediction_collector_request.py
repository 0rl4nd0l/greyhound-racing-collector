import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from race_collection.manual_prediction_collector_request import (
    RECEIPT_READY,
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
    canonical_bytes,
    runner_set_sha256,
    sha256_bytes,
)

NOW = datetime.fromisoformat("2026-07-29T14:00:00+10:00")
JUMP = NOW + timedelta(minutes=30)
RACE = {
    "race_id": "Race 5 - GUNN - 2026-07-29",
    "url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-29/5/example",
    "venue": "GUNN",
    "race_number": 5,
    "race_date": "2026-07-29",
    "jump_timestamp": JUMP.isoformat(),
}
RUNNERS = [
    {"box_number": 1, "dog_name": "Alpha", "identity": "ALPHA"},
    {"box_number": 2, "dog_name": "Bravo", "identity": "BRAVO"},
]


def protocol(tmp_path):
    return ManualPredictionCollectorProtocol(tmp_path / "requests")


def publish(store, *, request_id="00000000-0000-4000-8000-000000000001", **overrides):
    values = {
        "race": RACE,
        "expected_runners": RUNNERS,
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=10),
        "request_id": request_id,
    }
    values.update(overrides)
    return store.publish_request(**values)


def ready_handoff():
    report_raw = b'{"attempts":[]}\n'
    form_raw = b"Dog Name,BOX\n1. Alpha,\n2. Bravo,\n"
    sidecar_raw = b'{"metadata_is_leakage_safe":true}\n'
    return {
        "schema_version": "on_demand_verified_master_packet_v1",
        "race_id": RACE["race_id"],
        "append_timestamp": NOW.isoformat(),
        "source_report_sha256": sha256_bytes(report_raw),
        "source_form_sha256": sha256_bytes(form_raw),
        "source_sidecar_sha256": sha256_bytes(sidecar_raw),
        "packet_record_schema_version": "market_form_residual_shadow_record_v3",
        "packet_record_checksum_sha256": "d" * 64,
        "packet_effective_state_schema_version": "market_form_residual_effective_state_v2",
        "packet_effective_state_sha256": "e" * 64,
        "_report_bytes": report_raw,
        "_form_bytes": form_raw,
        "_sidecar_bytes": sidecar_raw,
    }


def ready_receipt():
    return {
        "schema_version": "on_demand_odds_receipt_v1",
        "race_id": RACE["race_id"],
        "captured_at": NOW.isoformat(),
        "source_kind": "verified_autonomous_receipt",
        "source_url": "https://www.sportsbet.com.au/example",
        "runner_set_sha256": runner_set_sha256(RUNNERS),
        "markets": {"win": RUNNERS, "place": RUNNERS},
    }


def claim(store):
    publish(store)
    context = store.prepare_collector_request(
        now=NOW,
        collector_run_id="collector-1",
        active_capture=False,
    )
    assert context is not None
    return context


def test_request_claim_ready_response_and_consume_once(tmp_path):
    store = protocol(tmp_path)
    context = claim(store)
    store.begin_attempt(context, now=NOW, collector_run_id="collector-1")
    response = store.publish_receipt_ready(
        context,
        now=NOW + timedelta(seconds=5),
        handoff=ready_handoff(),
        normalized_receipt=ready_receipt(),
    )

    assert response["status"] == RECEIPT_READY
    consumed = store.consume_response(
        context.request["request_id"],
        now=NOW + timedelta(seconds=6),
    )
    assert consumed["response"]["status"] == RECEIPT_READY
    assert consumed["receipt"]["race"] == RACE
    with pytest.raises(ProtocolRejected, match="RESPONSE_ALREADY_CONSUMED"):
        store.consume_response(
            context.request["request_id"],
            now=NOW + timedelta(seconds=7),
        )


def test_duplicate_request_claim_and_response_fail_closed(tmp_path):
    store = protocol(tmp_path)
    published = publish(store)
    with pytest.raises(ProtocolRejected, match="REPLAYED_REQUEST"):
        publish(store)

    context = store.claim_request(
        published["request_id"],
        now=NOW,
        collector_run_id="collector-1",
    )
    with pytest.raises(ProtocolRejected, match="DUPLICATE_CLAIM"):
        store.claim_request(
            context.request["request_id"],
            now=NOW,
            collector_run_id="collector-2",
        )

    store.publish_terminal(
        context,
        status="CAPTURE_FAILED",
        now=NOW + timedelta(seconds=1),
        reason="synthetic_failure",
    )
    with pytest.raises(ProtocolRejected, match="DUPLICATE_RESPONSE"):
        store.publish_terminal(
            context,
            status="CAPTURE_FAILED",
            now=NOW + timedelta(seconds=1),
            reason="synthetic_failure",
        )


def test_only_one_unexpired_manual_request_may_be_active(tmp_path):
    store = protocol(tmp_path)
    publish(store)
    with pytest.raises(ProtocolRejected, match="ACTIVE_REQUEST_EXISTS"):
        publish(
            store,
            request_id="00000000-0000-4000-8000-000000000002",
        )

    second = publish(
        store,
        request_id="00000000-0000-4000-8000-000000000002",
        created_at=NOW + timedelta(minutes=11),
        expires_at=NOW + timedelta(minutes=20),
    )
    assert second["request_id"].endswith("2")


def test_concurrent_publish_admits_only_one_active_request(tmp_path):
    store = protocol(tmp_path)

    def attempt(request_id):
        try:
            return publish(store, request_id=request_id)["request_id"]
        except ProtocolRejected as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                attempt,
                (
                    "00000000-0000-4000-8000-000000000001",
                    "00000000-0000-4000-8000-000000000002",
                ),
            )
        )

    assert sorted(result == "ACTIVE_REQUEST_EXISTS" for result in results) == [
        False,
        True,
    ]
    assert len(list((store.root / "requests").glob("*.json"))) == 1


@pytest.mark.parametrize(
    ("created_at", "expires_at", "now", "status"),
    [
        (
            NOW - timedelta(minutes=5),
            NOW - timedelta(seconds=1),
            NOW,
            "REQUEST_EXPIRED",
        ),
        (
            NOW - timedelta(minutes=5),
            JUMP + timedelta(minutes=5),
            JUMP,
            "CAPTURE_WINDOW_CLOSED",
        ),
    ],
)
def test_expired_and_post_jump_requests_get_one_terminal_response(
    tmp_path, created_at, expires_at, now, status
):
    store = protocol(tmp_path)
    publish(store, created_at=created_at, expires_at=expires_at)

    assert (
        store.prepare_collector_request(
            now=now,
            collector_run_id="collector-1",
            active_capture=False,
        )
        is None
    )
    assert (
        store.read_response("00000000-0000-4000-8000-000000000001")["status"] == status
    )


def test_future_created_timestamp_terminalizes_without_attempt(tmp_path):
    store = protocol(tmp_path)
    published = publish(
        store,
        created_at=NOW + timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=5),
    )

    context = store.prepare_collector_request(
        now=NOW,
        collector_run_id="collector-1",
        active_capture=False,
    )

    assert context is None
    assert store.read_response(published["request_id"])["status"] == "CAPTURE_FAILED"
    assert not store.attempt_path(published["request_id"]).exists()


def test_active_capture_defers_without_claiming(tmp_path):
    store = protocol(tmp_path)
    published = publish(store)

    assert (
        store.prepare_collector_request(
            now=NOW,
            collector_run_id="collector-1",
            active_capture=True,
        )
        is None
    )
    assert not store.claim_path(published["request_id"]).exists()


def test_crash_recovery_resumes_before_attempt_but_never_retries_started_attempt(
    tmp_path,
):
    store = protocol(tmp_path)
    context = claim(store)

    recovered = store.prepare_collector_request(
        now=NOW + timedelta(seconds=1),
        collector_run_id="collector-2",
        active_capture=False,
    )
    assert recovered is not None
    assert recovered.recovered is True
    store.begin_attempt(
        recovered, now=NOW + timedelta(seconds=2), collector_run_id="collector-2"
    )

    assert (
        store.prepare_collector_request(
            now=NOW + timedelta(seconds=3),
            collector_run_id="collector-3",
            active_capture=False,
        )
        is None
    )
    response = store.read_response(context.request["request_id"])
    assert response["status"] == "CAPTURE_FAILED"
    assert response["reason"] == "collector_recovered_after_started_attempt"


def test_plan_priority_and_exact_identity_mismatch_fail_closed(tmp_path):
    store = protocol(tmp_path)
    context = claim(store)
    other = {
        "status": "READY_TO_CAPTURE",
        "race_id": "Race 1 - WPK - 2026-07-29",
    }
    target = {
        "status": "READY_TO_CAPTURE",
        "race_id": RACE["race_id"],
        "thedogs_source_url": RACE["url"],
        "venue": RACE["venue"],
        "race_number": RACE["race_number"],
        "race_date": RACE["race_date"],
        "jump_datetime": RACE["jump_timestamp"],
        "expected_runners": RUNNERS,
    }
    prioritized = store.prioritize_capture_plan(
        context,
        {"races": [other, target], "limit": 2},
        now=NOW,
    )
    assert prioritized["races"][0]["race_id"] == RACE["race_id"]
    assert prioritized["manual_request_id"] == context.request["request_id"]

    mismatched = dict(target, jump_datetime=(JUMP + timedelta(minutes=1)).isoformat())
    with pytest.raises(ProtocolRejected, match="IDENTITY_MISMATCH"):
        store.prioritize_capture_plan(
            context,
            {"races": [mismatched], "limit": 1},
            now=NOW,
        )


def test_malformed_timestamp_unknown_status_and_receipt_hash_drift_fail_closed(
    tmp_path,
):
    store = protocol(tmp_path)
    with pytest.raises(ProtocolRejected, match="TIMESTAMP_INVALID"):
        publish(store, expires_at="not-a-time")

    context = claim(store)
    store.begin_attempt(context, now=NOW, collector_run_id="collector-1")
    response = store.publish_receipt_ready(
        context,
        now=NOW + timedelta(seconds=1),
        handoff=ready_handoff(),
        normalized_receipt=ready_receipt(),
    )
    receipt_path = store.root / response["receipt"]["path"]
    receipt_path.write_bytes(canonical_bytes({"changed": True}))
    with pytest.raises(ProtocolRejected, match="HASH_DRIFT"):
        store.consume_response(
            context.request["request_id"],
            now=NOW + timedelta(seconds=2),
        )

    other = protocol(tmp_path / "other")
    context = claim(other)
    response_path = other.response_path(context.request["request_id"])
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_bytes(
        canonical_bytes(
            {
                "schema_version": "manual-prediction-collector-response-v1",
                "request_id": context.request["request_id"],
                "request_sha256": context.request_sha256,
                "claim_sha256": context.claim_sha256,
                "attempt_sha256": None,
                "status": "UNKNOWN",
                "responded_at": NOW.isoformat(),
                "race": RACE,
                "reason": None,
                "receipt": None,
            }
        )
    )
    with pytest.raises(ProtocolRejected, match="STATUS_UNKNOWN"):
        other.read_response(context.request["request_id"])


def test_partial_temporary_write_is_ignored_deterministically(tmp_path):
    store = protocol(tmp_path)
    incoming = store.root / "requests" / ".incoming-crash"
    incoming.parent.mkdir(parents=True)
    incoming.write_text('{"partial":', encoding="utf-8")

    published = publish(store)

    assert published["request_id"] == "00000000-0000-4000-8000-000000000001"
    assert incoming.read_text(encoding="utf-8") == '{"partial":'


def test_protocol_root_symlink_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "requests-link"
    os.symlink(target, link)

    with pytest.raises(ProtocolRejected, match="PROTOCOL_PATH_UNSAFE"):
        publish(ManualPredictionCollectorProtocol(link))


def test_request_and_response_bytes_are_canonical_and_hash_bound(tmp_path):
    store = protocol(tmp_path)
    context = claim(store)
    request_raw = store.request_path(context.request["request_id"]).read_bytes()
    claim_raw = store.claim_path(context.request["request_id"]).read_bytes()

    assert request_raw == canonical_bytes(json.loads(request_raw))
    assert claim_raw == canonical_bytes(json.loads(claim_raw))
    assert context.request_sha256 in claim_raw.decode()
