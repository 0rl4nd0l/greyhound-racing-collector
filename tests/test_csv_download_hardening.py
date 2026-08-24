import hashlib
import io
import os
import sys
import types
import tempfile
import json
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest

from utils.csv_metadata import (
    THEDOGS_EXPERT_FORM_COLUMNS,
    build_prejump_shadow_metadata_payload,
    build_safe_target_metadata_payload,
    existing_prejump_sidecar_contract_status,
    load_safe_sidecar_target_metadata,
    verify_canonical_sidecar_payload,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_COMMA_EXPORT = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_live_batch/quarantine/20260527T092141Z_non_pipe_delimited_Race 13 - BAL - 2026-05-27.csv"
)
REAL_COMMA_SIDECAR = Path(f"{REAL_COMMA_EXPORT}.metadata.json")


@pytest.fixture(autouse=True)
def _isolate_upcoming_dir(monkeypatch):
    # Use a temp directory for UPCOMING_RACES_DIR in these tests
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("UPCOMING_RACES_DIR", tmpdir)
        yield tmpdir


def test_normalize_race_url_strips_expert_form_and_fragments():
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()
    base, expert = br._normalize_race_url(
        "https://www.thedogs.com.au/racing/grafton/2025-09-02/5/expert-form?foo=1#frag"
    )
    assert base.endswith("/racing/grafton/2025-09-02/5")
    assert expert.endswith("/racing/grafton/2025-09-02/5/expert-form")

    # Duplicated expert-form segments are collapsed
    base2, expert2 = br._normalize_race_url(
        "https://www.thedogs.com.au/racing/angle-park/2025-09-02/1/expert-form/expert-form/"
    )
    assert base2.endswith("/racing/angle-park/2025-09-02/1")
    assert expert2.endswith("/racing/angle-park/2025-09-02/1/expert-form")


def test_meeting_card_grade_source_label_without_proof_is_never_safe(tmp_path):
    race_url = "https://www.thedogs.com.au/racing/mandurah/2026-07-17/10/test"
    forged = {
        "date": "2026-07-17",
        "venue": "MAND",
        "race_number": 10,
        "target_distance": "520m",
        "target_distance_source": "canonical_pre_race_page",
        "target_grade": "Maiden",
        "target_grade_source": "thedogs_meeting_card_exact_race",
    }

    built = build_safe_target_metadata_payload(
        forged,
        source_url=race_url,
        allow_generic_fields=False,
    )

    assert built["target_grade"] is None
    assert built["target_grade_source"] == "default_missing_target"
    assert built["metadata_is_leakage_safe"] is False

    payload = {
        **forged,
        "metadata_is_leakage_safe": True,
        "race_url": race_url,
        "race_time_mapping_status": "exact_url_match",
        "race_time_source": "canonical_race_url",
    }
    verified = verify_canonical_sidecar_payload(
        payload,
        csv_path=tmp_path / "Race 10 - MAND - 2026-07-17.csv",
    )

    assert verified["target_metadata_status"] == "unsafe"
    assert "noncanonical_target_grade_source" in verified[
        "target_metadata_failure_reason"
    ]

    csv_path = tmp_path / "Race 10 - MAND - 2026-07-17.csv"
    Path(f"{csv_path}.metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    loaded = load_safe_sidecar_target_metadata(csv_path)
    assert loaded["target_grade"] is None
    assert "unsafe_sidecar_target_grade:thedogs_meeting_card_exact_race" in loaded[
        "rejected_metadata_sources"
    ]


def test_meeting_card_grade_proof_rejects_mismatched_sidecar_venue(tmp_path):
    race_url = "https://www.thedogs.com.au/racing/mandurah/2026-07-17/10/test"
    forged = {
        "date": "2026-07-17",
        "venue": "CANN",
        "race_number": 10,
        "target_distance": "520m",
        "target_distance_source": "canonical_pre_race_page",
        "target_grade": "Maiden",
        "target_grade_source": "thedogs_meeting_card_exact_race",
        "target_grade_context_schema": "thedogs_meeting_card_exact_race_v1",
        "target_grade_equivalence_key": "MAIDEN",
        "target_grade_exact_value": "Maiden",
        "target_grade_race_date": "2026-07-17",
        "target_grade_race_number": 10,
        "target_grade_race_url": race_url,
        "target_grade_source_url": "https://www.thedogs.com.au/racing/2026-07-17",
        "target_grade_source_sha256": "a" * 64,
        "target_grade_venue": "MAND",
    }

    built = build_safe_target_metadata_payload(
        forged,
        source_url=race_url,
        allow_generic_fields=False,
    )
    assert built["target_grade"] is None
    assert built["metadata_is_leakage_safe"] is False

    payload = {
        **forged,
        "metadata_is_leakage_safe": True,
        "race_url": race_url,
        "race_time_mapping_status": "exact_url_match",
        "race_time_source": "canonical_race_url",
    }
    verified = verify_canonical_sidecar_payload(
        payload,
        csv_path=tmp_path / "Race 10 - MAND - 2026-07-17.csv",
    )
    assert verified["target_metadata_status"] == "unsafe"

    shadow = build_prejump_shadow_metadata_payload(payload)
    assert shadow["status"] == "FAIL"
    assert "target_grade_missing_or_unsafe" in shadow["fail_reasons"]

    csv_path = tmp_path / "Race 10 - MAND - 2026-07-17.csv"
    Path(f"{csv_path}.metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    loaded = load_safe_sidecar_target_metadata(csv_path)
    assert loaded["target_grade"] is None


def test_download_rejects_html_masquerading_as_csv(monkeypatch, _isolate_upcoming_dir):
    """download_race_csv should NOT save files when the body is HTML."""
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()

    # Patch link finder to return direct CSV content which is actually HTML
    def _fake_find_csv_download_link(soup, race_url, **_kwargs):
        return {"type": "direct_csv", "data": "<html><body>Not CSV</body></html>"}

    monkeypatch.setattr(br, "find_csv_download_link", _fake_find_csv_download_link)

    # Stub network GET for the race page to avoid external calls
    class _Resp:
        def __init__(self, status_code=200, text="", content=b""):
            self.status_code = status_code
            self.text = text
            self.content = content
            self.headers = {}
        def close(self):
            pass
    
    fake_session = types.SimpleNamespace(
        get=lambda url, timeout=30, **_kwargs: _Resp(200, content=b"<html><title>Race 5</title></html>")
    )
    monkeypatch.setattr(br, "session", fake_session)

    res = br.download_race_csv("https://www.thedogs.com.au/racing/grafton/2025-09-02/5")
    assert not res.get("success"), f"Expected failure, got: {res}"
    # Ensure nothing was written (no filepath in result)
    assert "filepath" not in res


def test_download_accepts_verified_thedogs_export_and_writes_pipe_file(monkeypatch, _isolate_upcoming_dir):
    from upcoming_race_browser import UpcomingRaceBrowser

    if not REAL_COMMA_EXPORT.exists() or not REAL_COMMA_SIDECAR.exists():
        pytest.skip("real TheDogs comma export fixture is not present")

    br = UpcomingRaceBrowser()
    csv_content = REAL_COMMA_EXPORT.read_text(encoding="utf-8")
    sidecar = json.loads(REAL_COMMA_SIDECAR.read_text(encoding="utf-8"))

    def _fake_find_csv_download_link(soup, race_url, **_kwargs):
        return {"type": "direct_csv", "data": csv_content}

    monkeypatch.setattr(br, "find_csv_download_link", _fake_find_csv_download_link)
    monkeypatch.setattr(
        br,
        "extract_detailed_race_info",
        lambda soup, race_url: dict(sidecar["race_info"]),
    )
    monkeypatch.setattr(
        br,
        "_extract_safe_target_metadata_from_page",
        lambda soup, race_url: {
            "target_distance": sidecar["target_distance"],
            "target_distance_source": sidecar["target_distance_source"],
            "target_grade": sidecar["target_grade"],
            "target_grade_source": sidecar["target_grade_source"],
            "metadata_is_leakage_safe": sidecar["metadata_is_leakage_safe"],
            "metadata_source_url": sidecar["metadata_source_url"],
        },
    )

    # Stub network GET for the race page to avoid external calls
    class _Resp:
        def __init__(self, status_code=200, text="", content=b""):
            self.status_code = status_code
            self.text = text
            self.content = content
            self.headers = {}
        def close(self):
            pass

    fake_session = types.SimpleNamespace(
        get=lambda url, timeout=30, **_kwargs: _Resp(200, content=b"<html><title>Race 5</title></html>")
    )
    monkeypatch.setattr(br, "session", fake_session)

    res = br.download_race_csv(sidecar["race_url"])
    assert res.get("success"), f"Expected success, got: {res}"
    fp = res.get("filepath")
    assert fp and os.path.exists(fp)
    with open(fp, "r", encoding="utf-8") as f:
        data = f.read()
    assert data.splitlines()[0].startswith("Dog Name|Sex|PLC|BOX")
    assert res["runner_completeness"]["status"] == "COMPLETE"
    assert res["normalization"]["original_delimiter"] == ","
    assert res["normalization"]["normalized_delimiter"] == "|"
    assert os.path.exists(res["raw_export_path"])
    assert os.path.exists(f"{fp}.metadata.json")


def _synthetic_thedogs_export(runners):
    rows = [list(THEDOGS_EXPERT_FORM_COLUMNS)]
    for box_number, dog_name in runners:
        rows.append(
            [
                f"{box_number}. {dog_name}",
                "D",
                "1",
                str(box_number),
                "30.0",
                "400",
                "2026-05-01",
                "TEST",
                "M",
                "22.10",
                "2.00",
                "22.00",
                "5.00",
                "1.00",
                "111",
                "1",
                "$2.00",
            ]
        )
    return "\n".join(",".join(row) for row in rows) + "\n"


def test_primary_download_reallocates_duplicate_refetches_to_native_identity(
    monkeypatch,
    _isolate_upcoming_dir,
):
    from upcoming_race_browser import UpcomingRaceBrowser
    from scripts.refresh_prejump_upcoming import (
        current_index_metadata_selection,
        sidecar_metadata_coverage,
    )

    jump = datetime(2099, 6, 9, 13, 15, tzinfo=timezone.utc)
    race_url = "https://www.thedogs.com.au/racing/the-meadows/2099-06-09/1/test-race"
    odds_url = f"{race_url}/odds"
    expert_url = f"{race_url}/expert-form"
    export_url = f"{expert_url}/export.csv"
    race_html = f"""
    <html><head><title>Race 1</title></head><body>
      <formatted-time data-format="datetime_short" data-timestamp="{int(jump.timestamp())}">11:15 PM</formatted-time>
      <section class="race-card"><dl>
        <dt>Race Distance</dt><dd>400m</dd><dt>Race Grade</dt><dd>Maiden</dd>
        <dt>Track Condition</dt><dd>Good</dd><dt>Weather</dt><dd>Fine</dd>
      </dl></section>
      <table class="race-runners">
        <tbody data-content-url="/dogs/runner/159001/odds"><tr class="race-runner">
          <td class="race-runners__box"><sprite-svg name="rug_1"></sprite-svg></td>
          <td><div class="race-runners__name__dog">Alpha Runner<span>29.1</span></div></td>
          <td><runner-odd data-runner-id="159001"></runner-odd></td>
        </tr></tbody>
        <tbody data-content-url="/dogs/runner/159002/odds"><tr class="race-runner">
          <td class="race-runners__box"><sprite-svg name="rug_2"></sprite-svg></td>
          <td><div class="race-runners__name__dog">Bravo Runner<span>29.2</span></div></td>
          <td><runner-odd data-runner-id="159002"></runner-odd></td>
        </tr></tbody>
        <tbody data-content-url="/dogs/runner/159003/odds"><tr class="race-runner">
          <td class="race-runners__box"><sprite-svg name="rug_3"></sprite-svg></td>
          <td><div class="race-runners__name__dog">Charlie Runner<span>29.3</span></div></td>
          <td><runner-odd data-runner-id="159003"></runner-odd></td>
        </tr></tbody>
        <tbody data-content-url="/dogs/runner/159004/odds"><tr class="race-runner">
          <td class="race-runners__box"><sprite-svg name="rug_4"></sprite-svg></td>
          <td><div class="race-runners__name__dog">Delta Runner<span>29.4</span></div></td>
          <td><runner-odd data-runner-id="159004"></runner-odd></td>
        </tr></tbody>
      </table>
    </body></html>
    """.encode()
    expert_html = f"""
    <html><body><a href="{export_url}">Download CSV</a>
      <div class="layout--sidebar--expert">
        <div class="expert-form-runner__details__dog__name">Alpha Runner<span>(M)</span></div>
      </div>
    </body></html>
    """.encode()
    csv_content = _synthetic_thedogs_export(
        [
            (1, "Alpha Runner"),
            (2, "Bravo Runner"),
            (3, "Charlie Runner"),
            (4, "Delta Runner"),
        ]
    )
    api_url_prefix = "https://www.thedogs.com.au/api/runners/odds?"
    api_body = json.dumps(
        {
            "runner_odds": {
                runner_id: [
                    {
                        "runner_id": int(runner_id),
                        "run_box": box,
                        "price": 2.5 + box,
                        "bookmaker": {"id": 63, "code": "ladbrokes", "name": "Ladbrokes"},
                        "market": {"code": "fixed_win", "race_id": 15900},
                    }
                ]
                for box, runner_id in (
                    (1, "159001"),
                    (2, "159002"),
                    (3, "159003"),
                    (4, "159004"),
                )
            }
        },
        sort_keys=True,
    ).encode()

    class Response:
        def __init__(self, url, body, content_type):
            self.url = url
            self.content = body
            self.text = body.decode()
            self.status_code = 200
            self.headers = {
                "Content-Type": content_type,
                "Date": format_datetime(datetime.now(timezone.utc), usegmt=True),
                "Set-Cookie": "session=must-not-be-retained",
            }

        def close(self):
            pass

        def json(self):
            return json.loads(self.text)

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append(url)
            if url == race_url:
                return Response(url, race_html, "text/html; charset=utf-8")
            if url == expert_url:
                return Response(url, expert_html, "text/html; charset=utf-8")
            if url == export_url:
                return Response(url, csv_content.encode(), "text/csv")
            if url == odds_url:
                return Response(url, race_html, "text/html; charset=utf-8")
            if url.startswith(api_url_prefix):
                return Response(url, api_body, "application/json; charset=utf-8")
            if url.startswith("https://api.open-meteo.com/v1/forecast?"):
                weather = {
                    "hourly": {
                        "time": ["2099-06-09T23:00"],
                        "temperature_2m": [18.0],
                        "relative_humidity_2m": [70],
                        "precipitation": [0.0],
                        "pressure_msl": [1014.0],
                        "weather_code": [0],
                        "wind_speed_10m": [12.0],
                        "wind_direction_10m": [180],
                        "visibility": [10000],
                    }
                }
                return Response(url, json.dumps(weather).encode(), "application/json")
            raise AssertionError(f"unexpected request: {url}")

    browser = UpcomingRaceBrowser()
    session = Session()
    browser.session = session
    result = browser.download_race_csv(
        race_url,
        race_info_hint={
            "url": race_url,
            "date": "2099-06-09",
            "venue": "MEA",
            "race_number": 1,
            "jump_datetime": jump.isoformat(),
        },
    )

    assert result["success"] is True, result
    assert session.calls[0] == race_url
    assert session.calls[1].startswith("https://api.open-meteo.com/v1/forecast?")
    assert len(session.calls) == 6
    assert session.calls[2:5] == [expert_url, export_url, odds_url]
    assert session.calls[-1].startswith(api_url_prefix)
    assert session.calls.count(race_url) == 1
    assert session.calls.count(expert_url) == 1
    sidecar = json.loads(Path(f"{result['filepath']}.metadata.json").read_text())
    assert sidecar["source_native_race_id"] == "15900"
    evidence = sidecar["native_identity_evidence"]
    assert evidence["active_native_runner_ids"] == [
        "159001",
        "159002",
        "159003",
        "159004",
    ]
    assert evidence["request_accounting"]["logical_requests"] == 2
    evidence_core = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    assert evidence["evidence_sha256"] == hashlib.sha256(
        json.dumps(
            evidence_core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    assert evidence["odds_page_http"]["body_sha256"]
    assert evidence["odds_api_http"]["body_sha256"]
    assert "set-cookie" not in evidence["race_page_http"]["headers"]
    assert sidecar["expert_form_metadata"]["source_final_url"] == expert_url
    assert sidecar["expert_form_metadata"]["source_sha256"] == hashlib.sha256(
        expert_html
    ).hexdigest()
    assert sidecar["prejump_shadow_metadata"]["source_native_race_id"] == "15900"
    selected = [
        {
            "race_id": "Race 1 - MEA - 2099-06-09",
            "race_id_aliases": ["Race 1 - MEA - 2099-06-09"],
            "race_url": race_url,
            "race_number": 1,
            "venue": "MEA",
            "date": "2099-06-09",
            "jump_datetime": jump.isoformat(),
        }
    ]
    coverage = sidecar_metadata_coverage(Path(browser.upcoming_dir), selected)
    identity_coverage = json.loads(json.dumps(coverage))
    identity_coverage["races"][0].update(
        {
            "safe_weather_present": True,
            "safe_track_condition_present": True,
            "safe_expert_form_present": True,
            "safe_all_weather_track_expert_form_present": True,
        }
    )
    eligible, selection = current_index_metadata_selection(
        selected,
        identity_coverage,
        source_generated_at=datetime.now(timezone.utc),
    )
    assert selection["status"] == "READY", (selection, identity_coverage)
    assert eligible[0]["source_native_race_id"] == "15900"

    verified_sidecar = json.loads(json.dumps(sidecar))
    sidecar["native_identity_evidence"]["source_native_race_id"] = "15999"
    Path(f"{result['filepath']}.metadata.json").write_text(json.dumps(sidecar))
    tampered = sidecar_metadata_coverage(Path(browser.upcoming_dir), selected)
    assert tampered["races"][0]["source_native_race_id"] is None
    assert tampered["races"][0]["source_native_runner_ids"] == []

    del verified_sidecar["native_identity_evidence"]
    Path(f"{result['filepath']}.metadata.json").write_text(
        json.dumps(verified_sidecar)
    )
    missing = sidecar_metadata_coverage(Path(browser.upcoming_dir), selected)
    assert missing["races"][0]["source_native_race_id"] is None
    assert missing["races"][0]["native_identity_evidence_reason"] == (
        "native_identity_evidence_missing"
    )


def _canonical_runner_set_for_test(race_url, runners):
    participants = [
        {
            "box_number": box_number,
            "dog_name": dog_name,
            "source_native_runner_id": str(159000 + box_number),
        }
        for box_number, dog_name in runners
    ]
    return {
        "schema_version": "canonical_pre_race_runner_set_v1",
        "canonical_runner_set_status": "available",
        "final_runner_source": "canonical_pre_race_page",
        "final_runner_source_url": race_url,
        "final_runner_boxes": [box_number for box_number, _dog_name in runners],
        "final_runner_names": [dog_name for _box_number, dog_name in runners],
        "final_runner_participants": participants,
        "source_native_race_id": "15900",
        "native_identity_status": "available",
        "scratched_boxes": [],
        "scratched_participants": [],
        "reserve_boxes": [],
        "vacant_boxes": [],
        "race_number": 1,
        "expected_race_number": 1,
        "extraction_timestamp": "2026-06-09T07:00:00+10:00",
        "ambiguous_reasons": [],
    }


def test_download_refreshes_existing_csv_with_stale_sidecar_contract(
    monkeypatch,
    _isolate_upcoming_dir,
):
    import upcoming_race_browser as browser_module
    from scripts.daily_race_ingest_shadow_orchestrator import (
        validate_prejump_sidecar_metadata,
    )
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()
    race_url = "https://www.thedogs.com.au/racing/test/2030-06-09/1/test?trial=false"
    runners = [
        (1, "Alpha Runner"),
        (2, "Bravo Runner"),
        (3, "Charlie Runner"),
        (4, "Delta Runner"),
    ]
    csv_content = _synthetic_thedogs_export(runners)
    filename = "Race 1 - TEST - 2030-06-09.csv"
    existing_path = Path(br.upcoming_dir) / filename
    existing_path.write_text("Dog Name|BOX\n1. Old Runner|1\n", encoding="utf-8")
    existing_sidecar = Path(f"{existing_path}.metadata.json")
    existing_sidecar.write_text(
        json.dumps({"schema_version": "old_sidecar_without_prejump_contract"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        br,
        "find_csv_download_link",
        lambda soup, url, **_kwargs: {"type": "direct_csv", "data": csv_content},
    )
    monkeypatch.setattr(
        br,
        "extract_detailed_race_info",
        lambda soup, url, **_kwargs: {
            "race_number": "1",
            "venue": "TEST",
            "date": "2030-06-09",
            "race_time": "11:15 PM",
            "race_time_source": "canonical_race_url",
            "race_time_mapping_status": "exact_url_match",
        },
    )
    monkeypatch.setattr(
        br,
        "_extract_safe_target_metadata_from_page",
        lambda soup, url, **_kwargs: {
            "target_distance": "400m",
            "target_distance_source": "canonical_pre_race_page",
            "metadata_source_url": race_url,
        },
    )
    monkeypatch.setattr(
        browser_module,
        "extract_canonical_runner_set_from_html",
        lambda html, source_url=None, **_kwargs: _canonical_runner_set_for_test(
            source_url, runners
        ),
    )

    class _Resp:
        status_code = 200
        content = b"<html><title>Race 1</title></html>"
        text = "<html><title>Race 1</title></html>"
        headers = {}

        def close(self):
            pass

    monkeypatch.setattr(
        br,
        "session",
        types.SimpleNamespace(get=lambda url, timeout=30, **_kwargs: _Resp()),
    )

    result = br.download_race_csv(
        race_url,
        race_info_hint={
            "url": race_url,
            "date": "2030-06-09",
            "venue": "TEST",
            "race_number": "1",
            "grade": "Maiden",
            "target_grade_context_schema": "thedogs_meeting_card_exact_race_v1",
            "target_grade_equivalence_key": "MAIDEN",
            "target_grade_exact_value": "Maiden",
            "target_grade_race_url": (
                "https://www.thedogs.com.au/racing/test/2030-06-09/1/test"
            ),
            "target_grade_source_url": (
                "https://www.thedogs.com.au/racing/2030-06-09"
            ),
            "target_grade_source_sha256": "b" * 64,
        },
    )

    assert result["success"] is True, result
    assert result.get("already_exists") is not True
    assert result["existing_quarantine"]["csv_quarantine_path"]
    assert result["existing_quarantine"]["sidecar_quarantine_path"]
    assert Path(result["existing_quarantine"]["csv_quarantine_path"]).exists()
    assert Path(result["existing_quarantine"]["sidecar_quarantine_path"]).exists()
    assert existing_path.exists()
    sidecar = json.loads(existing_sidecar.read_text(encoding="utf-8"))
    assert sidecar["metadata_captured_at"]
    assert sidecar["target_grade"] == "Maiden"
    assert sidecar["target_grade_source"] == "thedogs_meeting_card_exact_race"
    assert sidecar["target_grade_context_schema"] == (
        "thedogs_meeting_card_exact_race_v1"
    )
    assert sidecar["target_grade_exact_value"] == "Maiden"
    assert sidecar["target_grade_source_url"] == (
        "https://www.thedogs.com.au/racing/2030-06-09"
    )
    assert sidecar["target_grade_source_sha256"] == "b" * 64
    assert sidecar["target_grade_race_url"] == (
        "https://www.thedogs.com.au/racing/test/2030-06-09/1/test"
    )
    assert sidecar["prejump_shadow_metadata"]["status"] == "PASS"
    assert sidecar["prejump_shadow_metadata"]["metadata_captured_at"]
    assert sidecar["prejump_shadow_metadata"]["runner_box_name_list"] == [
        {
            "box_number": 1, "dog_name": "Alpha Runner",
            "scratch_state": "ACTIVE",
            "source_native_runner_id": "159001",
        },
        {
            "box_number": 2, "dog_name": "Bravo Runner",
            "scratch_state": "ACTIVE",
            "source_native_runner_id": "159002",
        },
        {
            "box_number": 3, "dog_name": "Charlie Runner",
            "scratch_state": "ACTIVE",
            "source_native_runner_id": "159003",
        },
        {
            "box_number": 4, "dog_name": "Delta Runner",
            "scratch_state": "ACTIVE",
            "source_native_runner_id": "159004",
        },
    ]
    assert sidecar["prejump_shadow_metadata"]["source_native_race_id"] == "15900"
    validation = validate_prejump_sidecar_metadata(existing_path)
    assert validation["status"] == "PASS", validation
    contract = existing_prejump_sidecar_contract_status(existing_path)
    assert contract["status"] == "PASS", contract
    sidecar["metadata_captured_at"] = "2030-06-09T23:16:00+10:00"
    sidecar["prejump_shadow_metadata"]["metadata_captured_at"] = (
        "2030-06-09T23:16:00+10:00"
    )
    existing_sidecar.write_text(json.dumps(sidecar), encoding="utf-8")
    post_jump_contract = existing_prejump_sidecar_contract_status(existing_path)
    assert post_jump_contract["status"] == "FAIL"
    assert "metadata_captured_at_not_before_jump" in post_jump_contract["reasons"]


def test_download_fallback_quarantines_csv_without_valid_prejump_sidecar(
    monkeypatch,
    _isolate_upcoming_dir,
):
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()
    race_url = "https://www.thedogs.com.au/racing/test/2026-06-09/1/test?trial=false"
    filename = "Race 1 - TEST - 2026-06-09.csv"

    monkeypatch.setattr(
        br, "find_csv_download_link", lambda soup, url, **_kwargs: None
    )
    monkeypatch.setattr(
        br,
        "extract_detailed_race_info",
        lambda soup, url, **_kwargs: {
            "race_number": "1",
            "venue": "TEST",
            "date": "2026-06-09",
            "race_time": "11:15 PM",
            "race_time_source": "canonical_race_url",
            "race_time_mapping_status": "exact_url_match",
        },
    )
    monkeypatch.setattr(
        br,
        "_extract_safe_target_metadata_from_page",
        lambda soup, url, **_kwargs: {
            "target_distance": "400m",
            "target_distance_source": "canonical_pre_race_page",
            "target_grade": "Maiden",
            "target_grade_source": "canonical_pre_race_page",
            "metadata_is_leakage_safe": True,
            "metadata_source_url": race_url,
        },
    )

    class _Resp:
        status_code = 200
        content = b"<html><title>Race 1</title></html>"
        text = "<html><title>Race 1</title></html>"
        headers = {}

        def close(self):
            pass

    monkeypatch.setattr(
        br,
        "session",
        types.SimpleNamespace(get=lambda url, timeout=30, **_kwargs: _Resp()),
    )

    class _FakeExpertFormCsvScraper:
        def __init__(self, *args, **kwargs):
            pass

        def download_csv_from_expert_form(self, base_race_url, target_filename, race_info=None):
            assert target_filename == filename
            filepath = Path(os.environ["UPCOMING_RACES_DIR"]) / target_filename
            filepath.write_text(
                "Dog Name|BOX\n1. Alpha Runner|1\n2. Bravo Runner|2\n",
                encoding="utf-8",
            )
            Path(f"{filepath}.metadata.json").write_text(
                json.dumps({"schema_version": "stale_fallback_sidecar"}),
                encoding="utf-8",
            )
            return True

    fake_module = types.ModuleType("expert_form_csv_scraper")
    fake_module.ExpertFormCsvScraper = _FakeExpertFormCsvScraper
    monkeypatch.setitem(sys.modules, "expert_form_csv_scraper", fake_module)

    result = br.download_race_csv(race_url)

    assert result["success"] is False, result
    assert result["error"] == (
        "Expert-form fallback produced CSV without valid pre-jump sidecar contract"
    )
    assert result["existing_prejump_sidecar_contract"]["status"] == "FAIL"
    assert "prejump_shadow_metadata_missing" in result[
        "existing_prejump_sidecar_contract"
    ]["reasons"]
    assert Path(result["quarantine"]["csv_quarantine_path"]).exists()
    assert Path(result["quarantine"]["sidecar_quarantine_path"]).exists()
    assert not (Path(br.upcoming_dir) / filename).exists()


def test_download_pdf_masquerading_as_csv_tries_expert_form_fallback(
    monkeypatch,
    _isolate_upcoming_dir,
):
    import upcoming_race_browser as browser_module
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()
    race_url = "https://www.thedogs.com.au/racing/taree/2026-06-13/9/example?trial=false"
    filename = "Race 9 - TAREE - 2026-06-13.csv"

    monkeypatch.setattr(
        br,
        "find_csv_download_link",
        lambda soup, url, **_kwargs: {
            "type": "direct_csv",
            "data": "%PDF-1.5\nnot a csv\n",
        },
    )
    monkeypatch.setattr(
        br,
        "extract_detailed_race_info",
        lambda soup, url, **_kwargs: {
            "race_number": "9",
            "venue": "TAREE",
            "date": "2026-06-13",
            "race_time": "10:15 AM",
            "race_time_source": "canonical_race_url",
            "race_time_mapping_status": "exact_url_match",
        },
    )
    monkeypatch.setattr(
        br,
        "_extract_safe_target_metadata_from_page",
        lambda soup, url, **_kwargs: {
            "target_distance": "300m",
            "target_distance_source": "canonical_pre_race_page",
            "target_grade": "5th Grade",
            "target_grade_source": "canonical_pre_race_page",
            "metadata_is_leakage_safe": True,
            "metadata_source_url": race_url,
        },
    )
    monkeypatch.setattr(
        br,
        "_existing_prejump_sidecar_contract_status",
        lambda filepath: {
            "status": "PASS",
            "runner_completeness": {"status": "COMPLETE"},
        },
    )
    monkeypatch.setattr(
        browser_module,
        "bs4",
        types.SimpleNamespace(BeautifulSoup=lambda content, parser: object()),
    )

    class _Resp:
        status_code = 200
        content = b"<html><title>Race 9</title></html>"
        text = "<html><title>Race 9</title></html>"
        headers = {}

        def close(self):
            pass

    monkeypatch.setattr(
        br,
        "session",
        types.SimpleNamespace(get=lambda url, timeout=30, **_kwargs: _Resp()),
    )

    calls = []

    class _FakeExpertFormCsvScraper:
        def __init__(self, *args, **kwargs):
            pass

        def download_csv_from_expert_form(self, base_race_url, target_filename, race_info=None):
            calls.append((base_race_url, target_filename, race_info))
            assert target_filename == filename
            filepath = Path(os.environ["UPCOMING_RACES_DIR"]) / target_filename
            filepath.write_text("Dog Name|BOX\n1. Alpha Runner|1\n", encoding="utf-8")
            Path(f"{filepath}.metadata.json").write_text(
                json.dumps({"schema_version": "valid_prejump_sidecar"}),
                encoding="utf-8",
            )
            return True

    fake_module = types.ModuleType("expert_form_csv_scraper")
    fake_module.ExpertFormCsvScraper = _FakeExpertFormCsvScraper
    monkeypatch.setitem(sys.modules, "expert_form_csv_scraper", fake_module)

    result = br.download_race_csv(race_url)

    assert result["success"] is True, result
    assert calls
    assert result["fallback_trigger"] == "pdf_masquerading_as_csv"
    assert result["rejected_export"]["reason"] == "pdf_masquerading_as_csv"
    assert Path(result["rejected_export"]["raw_export_path"]).exists()
    assert Path(result["rejected_export"]["quarantine_path"]).exists()
    assert Path(result["filepath"]).exists()


def test_expert_form_scraper_rejects_success_without_valid_prejump_sidecar(
    _isolate_upcoming_dir,
):
    from expert_form_csv_scraper import ExpertFormCsvScraper

    scraper = ExpertFormCsvScraper.__new__(ExpertFormCsvScraper)
    scraper.output_dir = _isolate_upcoming_dir
    scraper.verbose = True
    scraper.collected_races = set()
    scraper.stats = {
        "races_requested": 0,
        "cache_hits": 0,
        "fetches_attempted": 0,
        "fetches_failed": 0,
        "successful_saves": 0,
    }
    scraper.safe_log = lambda *args, **kwargs: None

    filename = "Race 1 - TEST - 2026-06-09.csv"
    output_path = Path(_isolate_upcoming_dir) / filename
    output_path.write_text("Dog Name|BOX\n1. Old Runner|1\n", encoding="utf-8")
    Path(f"{output_path}.metadata.json").write_text(
        json.dumps({"schema_version": "old_sidecar_without_prejump_contract"}),
        encoding="utf-8",
    )
    calls = []

    def _fake_download(race_url, target_filename, race_info=None):
        calls.append((race_url, target_filename, race_info))
        assert target_filename == filename
        output_path.write_text(
            "Dog Name|BOX\n1. Alpha Runner|1\n2. Bravo Runner|2\n",
            encoding="utf-8",
        )
        Path(f"{output_path}.metadata.json").write_text(
            json.dumps({"schema_version": "still_missing_prejump_contract"}),
            encoding="utf-8",
        )
        return True

    scraper.download_csv_from_expert_form = _fake_download

    result = scraper.download_race_csv(
        {
            "date": "2026-06-09",
            "venue": "TEST",
            "race_number": 1,
            "url": "https://www.thedogs.com.au/racing/test/2026-06-09/1/test",
        }
    )

    assert result is False
    assert len(calls) == 1
    assert scraper.stats["successful_saves"] == 0
    assert scraper.stats["fetches_failed"] == 1
    assert not output_path.exists()
    quarantine_files = list((Path(_isolate_upcoming_dir) / "quarantine").glob("*"))
    assert any("stale_prejump_sidecar_contract" in path.name for path in quarantine_files)
    assert any(
        "download_prejump_sidecar_contract_failed" in path.name
        for path in quarantine_files
    )


def test_download_rejects_partial_runner_set(monkeypatch, _isolate_upcoming_dir):
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()
    csv_content = "Dog Name,Box\n2. Shima Lexie,2\n4. Sekiro,4\n"

    def _fake_find_csv_download_link(soup, race_url, **_kwargs):
        return {"type": "direct_csv", "data": csv_content}

    monkeypatch.setattr(br, "find_csv_download_link", _fake_find_csv_download_link)

    class _Resp:
        def __init__(self, status_code=200, text="", content=b""):
            self.status_code = status_code
            self.text = text
            self.content = content
            self.headers = {}

        def close(self):
            pass

    fake_session = types.SimpleNamespace(
        get=lambda url, timeout=30, **_kwargs: _Resp(200, content=b"<html><title>Race 5</title></html>")
    )
    monkeypatch.setattr(br, "session", fake_session)

    res = br.download_race_csv("https://www.thedogs.com.au/racing/grafton/2025-09-02/5")

    assert res.get("success") is not True
    assert res["error"] == "Incomplete runner set in downloaded CSV"
    assert res["runner_completeness"]["status"] == "INCOMPLETE"
    assert os.path.exists(res["quarantine_path"])


@pytest.fixture
def flask_client():
    # Importing app initializes the Flask app
    from app import app as flask_app

    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client


def test_api_download_quarantines_html_file(monkeypatch, flask_client, _isolate_upcoming_dir):
    """
    The /api/download_upcoming_race endpoint should 502 if the saved file contains HTML
    and the guardian flags it for quarantine.
    """
    # Create a fake browser that pretends to download a file by writing HTML
    class FakeBrowser:
        def __init__(self, *a, **kw):
            pass

        def download_race_csv(self, race_url):
            # Write an HTML file into upcoming dir
            upcoming = os.environ.get("UPCOMING_RACES_DIR")
            fn = "Race 5 - GRAFTON - 2025-09-02.csv"
            fp = os.path.join(upcoming, fn)
            with open(fp, "w", encoding="utf-8") as f:
                f.write("<html><body>oops</body></html>")
            return {"success": True, "filename": fn, "filepath": fp}

    # Ensure the endpoint imports our fake browser implementation at call time
    fake_upcoming = types.ModuleType("upcoming_race_browser")
    fake_upcoming.UpcomingRaceBrowser = FakeBrowser
    sys.modules["upcoming_race_browser"] = fake_upcoming

    # Patch the guardian at import site to always quarantine
    # Create a proper fake module so `from utils.file_integrity_guardian import FileIntegrityGuardian` resolves correctly
    if "utils" not in sys.modules:
        sys.modules["utils"] = types.ModuleType("utils")
    fake_mod = types.ModuleType("utils.file_integrity_guardian")

    class _VR:
        def __init__(self):
            self.should_quarantine = True
            self.issues = ["CSV file contains HTML content"]

    class _Guardian:
        def validate_file(self, path):
            return _VR()

    fake_mod.FileIntegrityGuardian = _Guardian

    sys.modules["utils.file_integrity_guardian"] = fake_mod

    resp = flask_client.post(
        "/api/download_upcoming_race",
        data=json.dumps({"race_url": "https://www.thedogs.com.au/racing/grafton/2025-09-02/5"}),
        content_type="application/json",
    )
    assert resp.status_code == 502
    data = resp.get_json()
    assert data and data.get("error") == "Upstream returned HTML instead of CSV"


def test_api_download_success_passthrough(monkeypatch, flask_client, _isolate_upcoming_dir):
    """Happy path: guardian allows file, endpoint returns success JSON."""
    # Fake browser writes a valid CSV file
    class FakeBrowserOK:
        def __init__(self, *a, **kw):
            pass

        def download_race_csv(self, race_url):
            upcoming = os.environ.get("UPCOMING_RACES_DIR")
            fn = "Race 2 - GRAFTON - 2025-09-02.csv"
            fp = os.path.join(upcoming, fn)
            with open(fp, "w", encoding="utf-8") as f:
                f.write("Dog Name,Box\nRunner,1\n")
            return {"success": True, "filename": fn, "filepath": fp}

    # Ensure the endpoint imports our fake browser implementation at call time
    fake_upcoming_ok = types.ModuleType("upcoming_race_browser")
    fake_upcoming_ok.UpcomingRaceBrowser = FakeBrowserOK
    sys.modules["upcoming_race_browser"] = fake_upcoming_ok

    # Patch guardian to allow
    if "utils" not in sys.modules:
        sys.modules["utils"] = types.ModuleType("utils")
    fake_mod_ok = types.ModuleType("utils.file_integrity_guardian")

    class _VR2:
        def __init__(self):
            self.should_quarantine = False
            self.issues = []

    class _GuardianOK:
        def validate_file(self, path):
            return _VR2()

    fake_mod_ok.FileIntegrityGuardian = _GuardianOK
    sys.modules["utils.file_integrity_guardian"] = fake_mod_ok

    resp = flask_client.post(
        "/api/download_upcoming_race",
        data=json.dumps({"race_url": "https://www.thedogs.com.au/racing/grafton/2025-09-02/2"}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.data
    payload = resp.get_json()
    assert payload and payload.get("success") is True
    assert payload.get("filename", "").endswith(".csv")
