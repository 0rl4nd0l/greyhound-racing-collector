from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import shadow_autopilot_daemon as daemon
from src.operator_ui.live_adapters import _calendar_gap, _duration, _one, _unit


def test_default_odds_timer_remains_byte_identical():
    retained = daemon.ROOT / "ops/systemd/shadow-autopilot-odds-capture.timer"
    assert daemon.odds_capture_timer_file_text().encode() == retained.read_bytes()
    assert (
        daemon.odds_capture_timer_file_text(live_freshness=False).encode() == retained.read_bytes()
    )


def test_live_odds_timer_changes_only_calendar_and_description():
    original = daemon.odds_capture_timer_file_text()
    candidate = daemon.odds_capture_timer_file_text(live_freshness=True)
    expected = original.replace(
        "Description=Run greyhound autonomous live odds capture except full-daemon minutes",
        "Description=Run greyhound autonomous live odds capture every minute",
    ).replace(
        f"OnCalendar={daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR}",
        "OnCalendar=*:*",
    )
    assert candidate == expected
    timer = _unit(candidate.encode(), "Timer")
    assert _one(timer, "OnCalendar") == "*:*"
    assert _one(timer, "AccuracySec") == "15s"
    assert "OnUnitInactiveSec" not in timer


@pytest.mark.parametrize("live_freshness", [False, True])
def test_writer_propagates_profile_and_reports_actual_calendar(tmp_path, live_freshness):
    result = daemon.write_odds_capture_service_files(
        service_dir=tmp_path / "odds",
        repo_path=Path("/fixture"),
        live_freshness=live_freshness,
    )
    timer_bytes = Path(result["timer_path"]).read_bytes()
    assert (
        timer_bytes == daemon.odds_capture_timer_file_text(live_freshness=live_freshness).encode()
    )
    timer = _unit(timer_bytes, "Timer")
    assert result["timer_calendar"] == _one(timer, "OnCalendar")
    assert result["timer_accuracy"] == _one(timer, "AccuracySec")
    assert result["timer_frequency"] == (
        "1min" if live_freshness else daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMER_FREQUENCY
    )
    assert ("--live-freshness" in Path(result["service_path"]).read_text()) == live_freshness
    full = daemon.write_service_files(
        service_dir=tmp_path / "full",
        repo_path=Path("/fixture"),
        live_freshness=live_freshness,
    )
    assert Path(full["timer_path"]).read_bytes() == daemon.timer_file_text().encode()


@pytest.mark.parametrize("live_freshness,expected_gap", [(False, 135), (True, 75)])
def test_native_r3_parser_accepts_timer_and_bounds_gap(live_freshness, expected_gap):
    timer = _unit(
        daemon.odds_capture_timer_file_text(live_freshness=live_freshness).encode(), "Timer"
    )
    assert (
        _calendar_gap(_one(timer, "OnCalendar")) + _duration(_one(timer, "AccuracySec"))
        == expected_gap
    )


def test_generated_live_calendar_keeps_idle_odds_fresh_until_next_activation(tmp_path, monkeypatch):
    from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
    from src.operator_ui import live_adapters
    from src.operator_ui.live_adapters import UpcomingRaceSource
    from tests.operator_ui.test_live_adapters import actual_payloads, make_live

    observed = datetime(2026, 7, 19, 12, tzinfo=timezone.utc)
    completed = observed + timedelta(seconds=65)
    idle = observed + timedelta(seconds=76)
    values = actual_payloads(completed, include_models=False)
    values["odds_refresh"]["generated_at"] = observed.isoformat()
    view = VerifiedCurrentRaceIndex(
        "collector_current_race_index_v2",
        "odds-9",
        observed.isoformat(),
        "1" * 64,
        b"packet",
        (),
        "refresh.json",
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )
    monkeypatch.setattr(live_adapters, "bounded_current_race_index", lambda **kwargs: view)
    timer = daemon.odds_capture_timer_file_text(live_freshness=True)
    parsed = _unit(timer.encode(), "Timer")
    assert _calendar_gap(_one(parsed, "OnCalendar")) == 60
    assert _duration(_one(parsed, "AccuracySec")) == 15
    adapter = make_live(
        tmp_path,
        values,
        now=completed,
        odds_timer=timer,
        include_models=False,
        odds_status=("inactive", "dead", 0),
        upcoming_races=UpcomingRaceSource(tmp_path / "index.json", tmp_path),
    )
    assert adapter.collector(completed).data["lanes"][1]["status"] == "RECEIPT_READY"
    assert adapter.collector(idle).data["lanes"][1]["status"] == "RECEIPT_READY"
    due = completed + timedelta(seconds=75)
    assert adapter.collector(due).data["lanes"][1]["status"] == "RECEIPT_READY"
    assert adapter.collector(due + timedelta(microseconds=1)).data["lanes"][1]["status"] == "STALE"
    assert adapter.upcoming(idle).evidence.status == "AVAILABLE/FRESH"
    assert adapter.upcoming(idle).evidence.age_seconds == 76


@pytest.mark.parametrize("age,status", [(300, "RECEIPT_READY"), (300.000001, "STALE")])
def test_completed_odds_cannot_rejuvenate_source_with_a_fresh_report(tmp_path, age, status):
    from tests.operator_ui.test_live_adapters import actual_payloads, make_live

    completed = datetime(2026, 7, 19, 12, tzinfo=timezone.utc)
    values = actual_payloads(completed, include_models=False)
    values["odds_refresh"]["generated_at"] = (completed - timedelta(seconds=age)).isoformat()
    adapter = make_live(
        tmp_path,
        values,
        now=completed,
        odds_timer=daemon.odds_capture_timer_file_text(live_freshness=True),
        include_models=False,
        odds_status=("inactive", "dead", 0),
    )
    assert adapter.collector(completed).data["lanes"][1]["status"] == status
