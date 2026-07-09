import sqlite3
from datetime import datetime
from pathlib import Path

from scripts import autonomous_live_odds_capture as capture
from sportsbet_odds_integrator import SportsbetOddsIntegrator


SOURCE_URL = "https://www.sportsbet.com.au/greyhound-racing/test/race-1"
RACE_ID = "Race 1 - TEST - 2026-07-09"
CAPTURE_MODE = "autonomous_prejump_t10m"


def _runner_rows() -> list[dict]:
    return [
        {"box_number": 1, "dog_name": "Alpha"},
        {"box_number": 2, "dog_name": "Bravo"},
    ]


def _odds_rows(*, market: str = "win", runners: list[dict] | None = None) -> list[dict]:
    base_price = 2.0 if market == "win" else 1.4
    return [
        {
            "box_number": runner["box_number"],
            "dog_name": runner["dog_name"],
            "dog_clean_name": runner["dog_name"],
            "odds_decimal": base_price + (runner["box_number"] / 10),
            "odds_fractional": "",
            "sportsbet_box_source": "runner_text",
            "sportsbet_list_position": runner["box_number"],
            "sportsbet_raw_runner_text": (
                f"{runner['box_number']}. {runner['dog_name']}"
            ),
        }
        for runner in (runners or _runner_rows())
    ]


def _plan_item() -> dict:
    return {
        "status": "READY_TO_CAPTURE",
        "skip_reasons": [],
        "canonical_race_identity": RACE_ID,
        "venue": "TEST",
        "race_number": 1,
        "race_date": "2026-07-09",
        "jump_datetime": "2026-07-09T12:30:00+10:00",
        "capture_window_minutes": 10,
        "capture_mode": CAPTURE_MODE,
        "expected_runners": _runner_rows(),
    }


def _plan() -> dict:
    return {
        "schema_version": "autonomous_live_odds_capture_plan_v1",
        "candidate_race_count": 1,
        "ready_to_capture_race_count": 1,
        "items": [_plan_item()],
    }


def _fetch_result(*, include_place: bool = True) -> dict:
    race_info = {
        "venue_url": SOURCE_URL,
        "race_number": 1,
    }
    if include_place:
        race_info["odds_data_place"] = _odds_rows(market="place")
    return {
        "success": True,
        "race_id": "TEST_2026-07-09_1",
        "alias_race_id": RACE_ID,
        "win_count": 2,
        "place_count": 2 if include_place else 0,
        "discovery_method": "sportsbet_landing",
        "warnings": [],
        "race_info": race_info,
        "odds_data": _odds_rows(market="win"),
    }


def _race_info() -> dict:
    return {
        "race_id": RACE_ID,
        "preserve_race_id": True,
        "venue": "TEST",
        "race_number": 1,
        "race_date": "2026-07-09",
        "venue_url": SOURCE_URL,
    }


def _insert_existing_capture(
    db_path: Path,
    *,
    markets: tuple[str, ...],
    rows_by_market: dict[str, list[dict]] | None = None,
) -> None:
    integrator = SportsbetOddsIntegrator(str(db_path), setup_database=True)
    for market in markets:
        report = integrator.append_pre_jump_odds_snapshot(
            _race_info(),
            (rows_by_market or {}).get(market) or _odds_rows(market=market),
            market_type=market,
            topN=3 if market == "place" else None,
            capture_mode=CAPTURE_MODE,
            capture_timestamp="2026-07-09T12:20:00+10:00",
            write_race_metadata=False,
        )
        assert report["status"] == "SUCCESS"


def test_execute_capture_plan_persists_balanced_win_and_place_rows(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "odds.db"
    monkeypatch.setattr(
        capture,
        "fetch_odds_for_target_race",
        lambda *_args, **_kwargs: _fetch_result(include_place=True),
    )

    report = capture.execute_capture_plan(
        plan=_plan(),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-07-09T12:20:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
    )

    assert report["status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    assert report["captured_race_count"] == 1
    assert report["inserted_live_odds_rows"] == 4
    assert report["attempts"][0]["append_report"]["win_inserted_rows"] == 2
    assert report["attempts"][0]["append_report"]["place_inserted_rows"] == 2
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT market_type, COUNT(*), MIN(topN), MAX(topN)
            FROM live_odds
            WHERE race_id = ? AND capture_mode = ?
            GROUP BY market_type
            ORDER BY market_type
            """,
            (RACE_ID, CAPTURE_MODE),
        ).fetchall()

    assert rows == [
        ("place", 2, 3, 3),
        ("win", 2, None, None),
    ]


def test_execute_capture_plan_skips_existing_complete_win_place_capture(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "odds.db"
    _insert_existing_capture(db_path, markets=("place", "win"))

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch should not run for a complete existing capture")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fail_fetch)

    report = capture.execute_capture_plan(
        plan=_plan(),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-07-09T12:20:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
    )

    attempt = report["attempts"][0]
    assert attempt["status"] == "SKIPPED_ALREADY_CAPTURED"
    assert attempt["existing_capture"]["status"] == "COMPLETE"
    assert attempt["existing_capture_rows"] == 4


def test_existing_win_place_capture_with_extra_runner_is_not_complete(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "odds.db"
    runners = _runner_rows() + [{"box_number": 3, "dog_name": "Charlie"}]
    _insert_existing_capture(
        db_path,
        markets=("place", "win"),
        rows_by_market={
            "place": _odds_rows(market="place", runners=runners),
            "win": _odds_rows(market="win", runners=runners),
        },
    )
    fetch_count = 0

    def fake_fetch(*_args, **_kwargs):
        nonlocal fetch_count
        fetch_count += 1
        return _fetch_result(include_place=True)

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)

    report = capture.execute_capture_plan(
        plan=_plan(),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-07-09T12:21:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
    )

    attempt = report["attempts"][0]
    assert fetch_count == 1
    assert attempt["status"] == "CAPTURED"
    assert attempt["existing_capture"]["status"] == "INCOMPLETE"
    assert attempt["existing_capture"]["extra_unexpected_runners_by_market"] == {
        "place": [{"box_number": 3, "identity": "CHARLIE"}],
        "win": [{"box_number": 3, "identity": "CHARLIE"}],
    }


def test_existing_win_place_capture_with_duplicate_runner_is_not_complete(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "odds.db"
    win_rows = _odds_rows(market="win")
    place_rows = _odds_rows(market="place")
    _insert_existing_capture(
        db_path,
        markets=("place", "win"),
        rows_by_market={
            "place": place_rows + [dict(place_rows[0])],
            "win": win_rows + [dict(win_rows[0])],
        },
    )
    fetch_count = 0

    def fake_fetch(*_args, **_kwargs):
        nonlocal fetch_count
        fetch_count += 1
        return _fetch_result(include_place=True)

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)

    report = capture.execute_capture_plan(
        plan=_plan(),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-07-09T12:21:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
    )

    attempt = report["attempts"][0]
    assert fetch_count == 1
    assert attempt["status"] == "CAPTURED"
    assert attempt["existing_capture"]["status"] == "INCOMPLETE"
    assert attempt["existing_capture"]["duplicate_runner_keys_by_market"] == {
        "place": [{"box_number": 1, "identity": "ALPHA"}],
        "win": [{"box_number": 1, "identity": "ALPHA"}],
    }


def test_execute_capture_plan_recaptures_existing_win_only_window(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "odds.db"
    _insert_existing_capture(db_path, markets=("win",))
    fetch_count = 0

    def fake_fetch(*_args, **_kwargs):
        nonlocal fetch_count
        fetch_count += 1
        return _fetch_result(include_place=True)

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)

    report = capture.execute_capture_plan(
        plan=_plan(),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-07-09T12:21:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
    )

    attempt = report["attempts"][0]
    assert fetch_count == 1
    assert attempt["status"] == "CAPTURED"
    assert attempt["existing_capture"]["status"] == "INCOMPLETE"
    assert attempt["existing_capture"]["missing_required_markets"] == ["place"]
    assert attempt["inserted_rows"] == 4


def test_missing_place_market_blocks_append(tmp_path, monkeypatch):
    db_path = tmp_path / "odds.db"
    monkeypatch.setattr(
        capture,
        "fetch_odds_for_target_race",
        lambda *_args, **_kwargs: _fetch_result(include_place=False),
    )

    report = capture.execute_capture_plan(
        plan=_plan(),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-07-09T12:20:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
    )

    attempt = report["attempts"][0]
    assert attempt["status"] == "BLOCKED_VALIDATION_FAILED"
    assert "sportsbet_place_market_missing" in attempt["reasons"]
    assert attempt["validation"]["validation_failure_root_cause"] == (
        "sportsbet_place_market_missing"
    )
    assert report["inserted_live_odds_rows"] == 0
    assert not db_path.exists()


def test_failed_combined_append_is_not_reported_as_captured(tmp_path, monkeypatch):
    monkeypatch.setattr(
        capture,
        "fetch_odds_for_target_race",
        lambda *_args, **_kwargs: _fetch_result(include_place=True),
    )
    monkeypatch.setattr(
        capture,
        "append_validated_capture",
        lambda **_kwargs: {
            "status": "FAILED",
            "inserted_rows": 2,
            "warnings": ["win:database_locked"],
        },
    )

    report = capture.execute_capture_plan(
        plan=_plan(),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-07-09T12:20:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
    )

    attempt = report["attempts"][0]
    assert report["status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert report["captured_race_count"] == 0
    assert report["inserted_live_odds_rows"] == 2
    assert report["db_write_performed"] is True
    assert attempt["status"] == "BLOCKED_APPEND_FAILED"
    assert attempt["db_write_performed"] is True


def test_append_pre_jump_odds_snapshot_accepts_place_and_rejects_other_markets(
    tmp_path,
):
    db_path = tmp_path / "odds.db"
    integrator = SportsbetOddsIntegrator(str(db_path), setup_database=True)

    place = integrator.append_pre_jump_odds_snapshot(
        _race_info(),
        _odds_rows(market="place"),
        market_type="place",
        topN=3,
        capture_mode=CAPTURE_MODE,
        capture_timestamp="2026-07-09T12:20:00+10:00",
        write_race_metadata=False,
    )
    rejected = integrator.append_pre_jump_odds_snapshot(
        _race_info(),
        _odds_rows(market="win"),
        market_type="show",
        capture_mode=CAPTURE_MODE,
        capture_timestamp="2026-07-09T12:20:00+10:00",
        write_race_metadata=False,
    )

    assert place["status"] == "SUCCESS"
    assert rejected["status"] == "REJECTED"
    assert "unsupported_market_type:show" in rejected["warnings"]
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT market_type, COUNT(*), MIN(topN), MAX(topN) FROM live_odds GROUP BY market_type"
        ).fetchall()

    assert rows == [("place", 2, 3, 3)]
