import base64
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

from scripts.audit_thedogs_market_history_snapshots import audit_manifest
from scripts.capture_thedogs_market_history import (
    canonical_json_bytes,
    capture_snapshot,
    sha256_bytes,
)

RACE_ID = "race-immutable-0001"
RACE_URL = "https://www.thedogs.com.au/racing/meadows/2026-06-01/1/test-race"
ODDS_URL = f"{RACE_URL}/odds"
JUMP = datetime(2026, 6, 1, 1, 0, tzinfo=timezone.utc)


def source_html() -> bytes:
    return f"""
    <html><body>
      <formatted-time data-format="datetime_short" data-timestamp="{int(JUMP.timestamp())}">11:00</formatted-time>
      <table class="race-runners"><tbody data-content-url="/dogs/runner/101/odds">
        <tr class="race-runner"><td><sprite-svg name="rug_1"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Alpha<span>29.1</span></div></td>
        <td class="runner-odds-fluctuation--price">2.60</td>
        <td><runner-odd data-runner-id="101"></runner-odd></td></tr>
      </tbody><tbody data-content-url="/dogs/runner/102/odds">
        <tr class="race-runner"><td><sprite-svg name="rug_2"></sprite-svg></td>
        <td><div class="race-runners__name__dog">Beta<span>29.2</span></div></td>
        <td class="runner-odds-fluctuation--price">3.60</td>
        <td><runner-odd data-runner-id="102"></runner-odd></td></tr>
      </tbody></table>
    </body></html>
    """.encode()


def api_payload(*, provider: bool = True, native_race_id: int = 9001) -> bytes:
    def quote(runner_id: int, box: int, price: float):
        return {
            "runner_id": runner_id,
            "run_box": box,
            "price": price,
            "bookmaker": (
                {"id": 63, "code": "ladbrokes", "name": "Ladbrokes"}
                if provider
                else None
            ),
            "market": {"code": "fixed_win", "race_id": native_race_id},
        }

    return json.dumps(
        {"runner_odds": {"101": [quote(101, 1, 2.5)], "102": [quote(102, 2, 3.5)]}}
    ).encode()


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
        self, *, provider: bool = True, server_time: datetime, native_race_id: int = 9001
    ):
        self.provider = provider
        self.server_time = server_time
        self.native_race_id = native_race_id

    def get(self, url, **_kwargs):
        if url.endswith("/racing/2026-06-01"):
            return FakeResponse(
                url,
                b"<html>meeting</html>",
                "text/html; charset=utf-8",
                self.server_time,
            )
        if url in {RACE_URL, ODDS_URL}:
            return FakeResponse(
                url, source_html(), "text/html; charset=utf-8", self.server_time
            )
        if "/api/runners/odds?" in url:
            return FakeResponse(
                url,
                api_payload(
                    provider=self.provider, native_race_id=self.native_race_id
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


def plan(window: str):
    return {
        "schema_version": "thedogs_market_snapshot_plan_v1",
        "race_id": RACE_ID,
        "race_url": RACE_URL,
        "odds_url": ODDS_URL,
        "jump_timestamp": JUMP.isoformat(),
        "nominal_window": window,
        "expected_active_runner_ids": ["101", "102"],
    }


def make_snapshot(
    tmp_path: Path,
    window: str = "T-120",
    *,
    provider: bool = True,
    native_race_id: int = 9001,
):
    minutes = int(window.removeprefix("T-"))
    current = JUMP - timedelta(minutes=minutes)
    output = tmp_path / "capture" / window
    result = capture_snapshot(
        plan(window),
        output,
        session=FakeSession(
            provider=provider,
            server_time=current,
            native_race_id=native_race_id,
        ),
        current_time=current,
        clock=FakeClock(current),
        repo_root=tmp_path,
    )
    return {
        "race_id": RACE_ID,
        "odds_url": ODDS_URL,
        "nominal_window": window,
        "expected_active_runner_ids": ["101", "102"],
        "raw_html_path": result["raw_html_path"],
        "receipt_path": result["receipt_path"],
    }


def write_manifest(tmp_path: Path, entries) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def mutate_receipt(path: Path, mutate) -> None:
    path.chmod(0o644)
    receipt = json.loads(path.read_text())
    mutate(receipt)
    core = {key: value for key, value in receipt.items() if key != "receipt_core_sha256"}
    receipt["receipt_core_sha256"] = sha256_bytes(canonical_json_bytes(core))
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def test_one_complete_field_snapshot_counts_as_one_temporal_observation(tmp_path):
    entry = make_snapshot(tmp_path)
    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert report["final_status"] == "THEDOGS_MARKET_HISTORY_CAPTURE_PARTIAL"
    assert report["accepted_snapshot_count"] == 1
    assert report["race_summary"][0]["temporal_observation_count"] == 1
    assert report["race_summary"][0]["runner_temporal_depth"] == {"101": 1, "102": 1}
    assert report["snapshots"][0]["active_runner_count"] == 2
    assert report["snapshots"][0]["runners"][0]["current_price"] == 2.5
    assert report["open_low_high_are_temporal_observations"] is False


def test_missing_receipt_is_rejected(tmp_path):
    entry = make_snapshot(tmp_path)
    Path(entry["receipt_path"]).unlink()

    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert report["final_status"] == "BLOCKED_LIVE_CAPTURE_OR_PROVENANCE"
    assert report["accepted_snapshot_count"] == 0
    assert "receipt_path_missing" in report["snapshots"][0]["blockers"][0]


def test_manifest_must_bind_exact_odds_url(tmp_path):
    entry = make_snapshot(tmp_path)
    entry.pop("odds_url")

    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert report["accepted_snapshot_count"] == 0
    assert "manifest_exact_odds_url_required" in report["snapshots"][0]["blockers"]


def test_receipt_source_alias_identity_tamper_is_rejected(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])
    mutate_receipt(
        receipt_path,
        lambda receipt: receipt["race_identity"].update({"race_number": 2}),
    )

    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert "receipt_race_identity_mismatch" in report["snapshots"][0]["blockers"]


def test_capture_at_jump_is_rejected(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])
    jump = JUMP.isoformat().replace("+00:00", "Z")
    mutate_receipt(
        receipt_path,
        lambda receipt: receipt.update(
            {"request_end_utc": jump, "capture_end_utc": jump}
        ),
    )

    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert "capture_not_strictly_prejump" in report["snapshots"][0]["blockers"]


def test_post_jump_capture_is_rejected(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])
    after = (JUMP + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    mutate_receipt(
        receipt_path,
        lambda receipt: receipt.update(
            {"request_end_utc": after, "capture_end_utc": after}
        ),
    )

    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert "capture_not_strictly_prejump" in report["snapshots"][0]["blockers"]


def test_receipt_native_id_mismatch_is_rejected(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])
    mutate_receipt(
        receipt_path,
        lambda receipt: receipt.update(
            {"active_native_runner_ids": ["101", "999"]}
        ),
    )

    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert "receipt_active_native_runner_set_mismatch" in report["snapshots"][0]["blockers"]


def test_unexpected_api_native_runner_is_rejected(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])

    def add_unexpected_runner(receipt):
        api_http = receipt["odds_api_http"]
        payload = json.loads(base64.b64decode(api_http["body_base64"]))
        payload["runner_odds"]["999"] = [
            {
                "runner_id": 999,
                "run_box": 8,
                "price": 9.0,
                "bookmaker": {"id": 63, "code": "ladbrokes", "name": "Ladbrokes"},
                "market": {"code": "fixed_win", "race_id": 9001},
            }
        ]
        body = json.dumps(payload).encode()
        api_http["body_base64"] = base64.b64encode(body).decode("ascii")
        api_http["body_sha256"] = sha256_bytes(body)
        api_http["body_bytes"] = len(body)

    mutate_receipt(receipt_path, add_unexpected_runner)
    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert "odds_api_native_runner_set_mismatch" in report["snapshots"][0]["blockers"]


def test_overlapping_warmed_request_chain_is_rejected(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])

    def overlap_meeting_and_race(receipt):
        meeting_start = receipt["warm_meeting_http"]["request_start_utc"]
        receipt["jump_source"]["request_start_utc"] = meeting_start

    mutate_receipt(receipt_path, overlap_meeting_and_race)
    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert "request_chain_time_order_invalid" in report["snapshots"][0]["blockers"]


def test_receipt_window_tolerance_cannot_exceed_prescribed_interval(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])
    mutate_receipt(
        receipt_path,
        lambda receipt: receipt.update({"late_tolerance_seconds": 91}),
    )

    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert "window_tolerance_invalid" in report["snapshots"][0]["blockers"]


def test_receipt_effective_box_projection_tamper_is_rejected(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])

    def tamper(receipt):
        receipt["runners"][0]["effective_box"] = 8

    mutate_receipt(receipt_path, tamper)
    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert "receipt_runner_projection_mismatch" in report["snapshots"][0]["blockers"]


def test_legacy_v1_runner_projection_remains_auditable(tmp_path):
    entry = make_snapshot(tmp_path)
    receipt_path = Path(entry["receipt_path"])

    def downgrade(receipt):
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

    mutate_receipt(receipt_path, downgrade)
    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert report["accepted_snapshot_count"] == 1
    assert "effective_box" not in report["snapshots"][0]["runners"][0]


def test_provider_unknown_is_a_valid_source_classification(tmp_path):
    entry = make_snapshot(tmp_path, provider=False)

    report = audit_manifest(write_manifest(tmp_path, [entry]))

    assert report["accepted_snapshot_count"] == 1
    assert report["snapshots"][0]["provider_classification"] == "provider_unknown"


def test_all_five_distinct_windows_are_required_for_trajectory_ready(tmp_path):
    entries = [
        make_snapshot(tmp_path / f"snapshot-{window}", window)
        for window in ("T-120", "T-60", "T-30", "T-10", "T-2")
    ]

    report = audit_manifest(write_manifest(tmp_path, entries))

    assert report["final_status"] == "THEDOGS_MARKET_HISTORY_CAPTURE_READY"
    assert report["trajectory_ready_race_count"] == 1
    assert report["race_summary"][0]["temporal_observation_count"] == 5
    assert report["race_summary"][0]["missing_windows"] == []
    assert report["race_summary"][0]["runner_temporal_depth"] == {"101": 5, "102": 5}


def test_five_windows_with_conflicting_native_race_identity_are_not_ready(tmp_path):
    entries = [
        make_snapshot(
            tmp_path / f"snapshot-{window}",
            window,
            native_race_id=9002 if window == "T-2" else 9001,
        )
        for window in ("T-120", "T-60", "T-30", "T-10", "T-2")
    ]

    report = audit_manifest(write_manifest(tmp_path, entries))

    assert report["final_status"] == "THEDOGS_MARKET_HISTORY_CAPTURE_PARTIAL"
    assert report["trajectory_ready_race_count"] == 0
    assert report["race_summary"][0]["source_identity_conflict"] is True


def test_duplicate_manifest_entry_does_not_inflate_temporal_depth(tmp_path):
    entry = make_snapshot(tmp_path)

    report = audit_manifest(write_manifest(tmp_path, [entry, dict(entry)]))

    assert report["accepted_snapshot_count"] == 2
    assert report["race_summary"][0]["temporal_observation_count"] == 1
    assert report["race_summary"][0]["runner_temporal_depth"] == {"101": 1, "102": 1}
