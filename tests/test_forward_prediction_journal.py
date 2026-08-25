from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
)
from scripts.forward_prediction_journal import (
    PENDING_RECEIPT,
    PREFLIGHT_EXCLUDED,
    READY,
    observe_receipt_ready_races,
    validate_exact_receipt_for_race,
)
from src.predictor.on_demand import PredictionBlocked, canonical_bytes, sha256_bytes

NOW = datetime(2026, 8, 25, 10, 0, tzinfo=timezone(timedelta(hours=10)))


def race(venue: str, slug: str, number: int, minutes: int) -> dict[str, Any]:
    race_date = NOW.date().isoformat()
    jump = NOW + timedelta(minutes=minutes)
    return {
        "race_id": f"Race {number} - {venue} - {race_date}",
        "date": race_date,
        "venue": venue,
        "race_number": number,
        "race_time": jump.strftime("%H:%M"),
        "jump_datetime": jump.isoformat(),
        "race_url": (
            f"https://www.thedogs.com.au/racing/{slug}/{race_date}/{number}/race"
        ),
        "runners": [
            {
                "box": 1,
                "display_name": f"{venue} Alpha",
                "identity": f"{venue} ALPHA",
                "source_native_runner_id": f"{number}01",
            },
            {
                "box": 2,
                "display_name": f"{venue} Beta",
                "identity": f"{venue} BETA",
                "source_native_runner_id": f"{number}02",
            },
        ],
    }


RACES = (
    race("GEE", "geelong", 13, 20),
    race("TWN", "townsville", 3, 18),
    race("QOT", "ladbrokes-q1-lakeside", 13, 15),
    race("WRGL", "warragul", 2, 22),
)


def validation(value: Mapping[str, Any], captured_at: datetime) -> dict[str, Any]:
    rows = [
        {
            "box_number": row["box"],
            "dog_name": row["display_name"],
            "identity": row["identity"],
            "odds_decimal": 2.0 + offset,
        }
        for offset, row in enumerate(value["runners"], start=1)
    ]
    del captured_at
    return {
        "schema_version": "autonomous_live_odds_capture_validation_v1",
        "status": "PASS",
        "source_url": (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            f"{str(value['race_url']).split('/racing/', 1)[1].split('/', 1)[0]}"
            f"/race-{value['race_number']}-999"
        ),
        "accepted_rows": rows,
        "accepted_row_count": len(rows),
        "rejected_rows": [],
        "accepted_place_rows": rows,
        "accepted_place_row_count": len(rows),
        "rejected_place_rows": [],
        "expected_runner_count": len(rows),
        "active_expected_runner_count": len(rows),
        "scratched_expected_runner_count": 0,
        "scratched_expected_runners": [],
        "scratched_expected_runners_with_odds": [],
        "missing_expected_runners": [],
        "extra_unexpected_runners": [],
        "failure_root_cause": None,
        "reasons": [],
    }


def handoff(value: Mapping[str, Any], captured_at: datetime) -> dict[str, Any]:
    source_validation = validation(value, captured_at)
    report = {
        "schema_version": "autonomous_live_odds_capture_report_v1",
        "attempts": [
            {
                "schema_version": "autonomous_live_odds_capture_attempt_v1",
                "race_id": value["race_id"],
                "status": "APPENDED",
                "reasons": [],
                "fetch_time": captured_at.isoformat(),
                "append_time": captured_at.isoformat(),
                "validation": source_validation,
            }
        ],
    }
    report_raw = canonical_bytes(report)
    form_raw = canonical_bytes({"race_id": value["race_id"], "kind": "form"})
    sidecar_raw = canonical_bytes(
        {"race_id": value["race_id"], "participants": value["runners"]}
    )
    return {
        "schema_version": "on_demand_verified_collector_capture_v2",
        "race_id": value["race_id"],
        "race": {
            "race_id": value["race_id"],
            "url": value["race_url"],
            "venue": value["venue"],
            "race_number": value["race_number"],
            "race_date": value["date"],
            "jump_timestamp": value["jump_datetime"],
        },
        "append_timestamp": captured_at.isoformat(),
        "runner_set_sha256": "0" * 64,
        "source_report_sha256": sha256_bytes(report_raw),
        "source_form_sha256": sha256_bytes(form_raw),
        "source_sidecar_sha256": sha256_bytes(sidecar_raw),
        "capture_attempt_sha256": "1" * 64,
        "append_report_sha256": "2" * 64,
        "_report_bytes": report_raw,
        "_form_bytes": form_raw,
        "_sidecar_bytes": sidecar_raw,
        "_form_name": f"{value['race_id']}.csv",
        "_scheduled_exact_receipt": True,
    }


def publish_manual_receipt(
    protocol: ManualPredictionCollectorProtocol,
    value: Mapping[str, Any],
    captured_at: datetime,
    root: Path,
) -> None:
    expected = [
        {
            "box_number": row["box"],
            "dog_name": row["display_name"],
            "identity": row["identity"],
        }
        for row in value["runners"]
    ]
    request = protocol.publish_request(
        race={
            "race_id": value["race_id"],
            "url": value["race_url"],
            "venue": value["venue"],
            "race_number": value["race_number"],
            "race_date": value["date"],
            "jump_timestamp": value["jump_datetime"],
        },
        expected_runners=expected,
        created_at=NOW - timedelta(minutes=1),
        expires_at=datetime.fromisoformat(str(value["jump_datetime"])),
    )
    context = protocol.claim_request(
        request["request_id"], now=captured_at, collector_run_id="fixture-collector"
    )
    protocol.begin_attempt(
        context, now=captured_at, collector_run_id="fixture-collector"
    )
    value_handoff = handoff(value, captured_at)
    from src.predictor.on_demand import receipt_from_handoff

    normalized, _, _, _ = receipt_from_handoff(
        value_handoff, current_time=captured_at, max_age_seconds=900
    )
    value_handoff["runner_set_sha256"] = normalized["runner_set_sha256"]
    source = root / str(value["venue"])
    source.mkdir()
    paths = {
        "report": source / "capture.json",
        "form": source / value_handoff["_form_name"],
        "sidecar": source / f"{value_handoff['_form_name']}.metadata.json",
    }
    for label, path in paths.items():
        path.write_bytes(value_handoff[f"_{label}_bytes"])
        value_handoff[f"_{label}_path"] = path.resolve()
    protocol.publish_receipt_ready(
        context,
        now=captured_at,
        handoff=value_handoff,
        normalized_receipt=normalized,
    )
    protocol.consume_response(request["request_id"], now=captured_at)


def test_four_races_wait_for_independent_staggered_exact_receipts(tmp_path: Path):
    protocol = ManualPredictionCollectorProtocol(tmp_path / "collector-requests")
    invocations: list[str] = []
    outputs = tmp_path / "outputs"

    def invoke(value: Mapping[str, Any]) -> str:
        race_id = str(value["race_id"])
        invocations.append(race_id)
        (outputs / race_id).mkdir(parents=True)
        return f"job-{len(invocations)}"

    first = observe_receipt_ready_races(
        RACES,
        protocol=protocol,
        current_time=NOW,
        receipt_max_age_seconds=900,
        minimum_prejump_margin_seconds=53,
        recorded_race_ids=frozenset(),
        invoke_ready=invoke,
    )
    assert [row.state for row in first] == [PENDING_RECEIPT] * 4
    assert invocations == []
    assert not outputs.exists()

    first_arrival = NOW + timedelta(seconds=70)
    publish_manual_receipt(protocol, RACES[1], first_arrival, tmp_path)
    publish_manual_receipt(protocol, RACES[3], first_arrival, tmp_path)
    second = observe_receipt_ready_races(
        RACES,
        protocol=protocol,
        current_time=first_arrival,
        receipt_max_age_seconds=900,
        minimum_prejump_margin_seconds=53,
        recorded_race_ids=frozenset(),
        invoke_ready=invoke,
    )
    assert [row.race_id for row in second if row.state == READY] == [
        RACES[1]["race_id"],
        RACES[3]["race_id"],
    ]
    assert invocations == [RACES[1]["race_id"], RACES[3]["race_id"]]

    second_arrival = NOW + timedelta(seconds=130)
    publish_manual_receipt(protocol, RACES[0], second_arrival, tmp_path)
    publish_manual_receipt(protocol, RACES[2], second_arrival, tmp_path)
    recorded = frozenset(invocations)
    third = observe_receipt_ready_races(
        RACES,
        protocol=protocol,
        current_time=second_arrival,
        receipt_max_age_seconds=900,
        minimum_prejump_margin_seconds=53,
        recorded_race_ids=recorded,
        invoke_ready=invoke,
    )
    assert [row.race_id for row in third if row.state == READY] == [
        RACES[0]["race_id"],
        RACES[2]["race_id"],
    ]
    assert invocations == [
        RACES[1]["race_id"],
        RACES[3]["race_id"],
        RACES[2]["race_id"],
        RACES[0]["race_id"],
    ]
    assert len(list(outputs.iterdir())) == 4


def test_expired_pending_race_never_allocates_or_invokes(tmp_path: Path):
    value = race("GEE", "geelong", 13, 1)
    invoked: list[str] = []
    result = observe_receipt_ready_races(
        [value],
        protocol=ManualPredictionCollectorProtocol(tmp_path / "requests"),
        current_time=NOW + timedelta(seconds=7),
        receipt_max_age_seconds=900,
        minimum_prejump_margin_seconds=53,
        recorded_race_ids=frozenset(),
        invoke_ready=lambda row: invoked.append(str(row["race_id"])),
    )
    assert result[0].state == PREFLIGHT_EXCLUDED
    assert result[0].reason == "INSUFFICIENT_PREJUMP_MARGIN"
    assert invoked == []


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("stale", "RECEIPT_STALE"),
        ("wrong_race", "RECEIPT_INVALID"),
        ("wrong_date", "RECEIPT_INVALID"),
        ("wrong_number", "RECEIPT_INVALID"),
        ("wrong_runner", "RUNNER_SET_AMBIGUOUS"),
        ("tampered", "RECEIPT_TAMPERED"),
        ("ambiguous", "RECEIPT_AMBIGUOUS"),
        ("alias_mismatch", "RECEIPT_INVALID"),
        ("malformed_url", "RECEIPT_INVALID"),
        ("unknown_alias", "RECEIPT_INVALID"),
    ],
)
def test_exact_receipt_negative_controls_fail_closed(mutation: str, reason: str):
    value = RACES[2]
    captured_at = NOW - timedelta(seconds=30)
    value_handoff = handoff(value, captured_at)
    if mutation == "stale":
        captured_at = NOW - timedelta(seconds=901)
        value_handoff = handoff(value, captured_at)
    elif mutation == "wrong_race":
        value_handoff["race_id"] = RACES[0]["race_id"]
    elif mutation == "wrong_date":
        value_handoff["race"] = {
            **value_handoff["race"],
            "race_id": "Race 13 - QOT - 2026-08-24",
            "race_date": "2026-08-24",
            "url": "https://www.thedogs.com.au/racing/ladbrokes-q1-lakeside/2026-08-24/13/race",
        }
    elif mutation == "wrong_number":
        value_handoff["race"] = {
            **value_handoff["race"],
            "race_id": f"Race 12 - QOT - {value['date']}",
            "race_number": 12,
            "url": f"https://www.thedogs.com.au/racing/ladbrokes-q1-lakeside/{value['date']}/12/race",
        }
    elif mutation == "wrong_runner":
        report = json.loads(value_handoff["_report_bytes"])
        report["attempts"][0]["validation"]["accepted_rows"][0]["identity"] = "WRONG"
        report["attempts"][0]["validation"]["accepted_place_rows"][0]["identity"] = (
            "WRONG"
        )
        value_handoff["_report_bytes"] = canonical_bytes(report)
        value_handoff["source_report_sha256"] = sha256_bytes(
            value_handoff["_report_bytes"]
        )
    elif mutation == "tampered":
        value_handoff["_form_bytes"] += b"tampered"
    elif mutation == "ambiguous":
        report = json.loads(value_handoff["_report_bytes"])
        report["attempts"].append(dict(report["attempts"][0]))
        value_handoff["_report_bytes"] = canonical_bytes(report)
        value_handoff["source_report_sha256"] = sha256_bytes(
            value_handoff["_report_bytes"]
        )
    elif mutation == "alias_mismatch":
        value_handoff["race"] = {
            **value_handoff["race"],
            "race_id": f"Race 13 - GEE - {value['date']}",
            "venue": "GEE",
            "url": str(RACES[0]["race_url"]),
        }
    elif mutation in {"malformed_url", "unknown_alias"}:
        report = json.loads(value_handoff["_report_bytes"])
        report["attempts"][0]["validation"]["source_url"] = (
            "not-a-url"
            if mutation == "malformed_url"
            else "https://www.sportsbet.com.au/greyhound-racing/australia-nz/unknown-track/race-13-999"
        )
        value_handoff["_report_bytes"] = canonical_bytes(report)
        value_handoff["source_report_sha256"] = sha256_bytes(
            value_handoff["_report_bytes"]
        )

    with pytest.raises(PredictionBlocked) as captured:
        validate_exact_receipt_for_race(
            value,
            value_handoff,
            current_time=NOW,
            receipt_max_age_seconds=900,
        )
    assert captured.value.code == reason


def test_recorded_race_is_never_retried_or_backfilled(tmp_path: Path):
    invoked: list[str] = []
    result = observe_receipt_ready_races(
        [RACES[0]],
        protocol=ManualPredictionCollectorProtocol(tmp_path / "requests"),
        current_time=NOW,
        receipt_max_age_seconds=900,
        minimum_prejump_margin_seconds=53,
        recorded_race_ids={str(RACES[0]["race_id"])},
        invoke_ready=lambda value: invoked.append(str(value["race_id"])),
    )
    assert result[0].state == "ALREADY_RECORDED"
    assert invoked == []
