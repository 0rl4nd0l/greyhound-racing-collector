import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from scripts import autonomous_live_odds_capture as capture


def test_output_guard_accepts_configured_external_evidence_root(
    tmp_path, monkeypatch
):
    repo_root = tmp_path / "release_repo"
    evidence_root = tmp_path / "runtime_artifacts" / "full_evidence_orchestration_20260525"
    output_dir = evidence_root / "autonomous_live_odds_capture_external"
    repo_root.mkdir()

    monkeypatch.setattr(capture, "ROOT", repo_root)

    assert (
        capture.assert_output_dir_safe(output_dir, evidence_root=evidence_root)
        == output_dir.absolute()
    )

    try:
        capture.assert_output_dir_safe(
            evidence_root / "not_an_odds_capture_output",
            evidence_root=evidence_root,
        )
    except ValueError as exc:
        assert str(exc).startswith(
            "output_dir_must_be_autonomous_live_odds_capture_artifact"
        )
    else:
        raise AssertionError("external evidence root must still enforce capture prefix")


def _write_capture_input(
    input_dir: Path,
    *,
    jump_time: str = "2026-06-10T15:00:00+10:00",
    venue: str = "WPK",
    race_number: int = 1,
) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    csv_path = input_dir / f"Race {race_number} - {venue} - 2026-06-10.csv"
    csv_path.write_text("Dog Name,BOX\n1. Alpha,\n2. Bravo,\n", encoding="utf-8")
    sidecar = {
        "metadata_is_leakage_safe": True,
        "prejump_shadow_metadata": {
            "status": "PASS",
            "metadata_is_leakage_safe": True,
            "race_date": "2026-06-10",
            "venue": venue,
            "race_number": str(race_number),
            "jump_time": jump_time,
            "source_url": (
                "https://www.thedogs.com.au/racing/wentworth-park/"
                f"2026-06-10/{race_number}/example"
            ),
            "runner_box_name_list": [
                {"box_number": 1, "dog_name": "Alpha"},
                {"box_number": 2, "dog_name": "Bravo"},
            ],
            "canonical_final_runner_alignment": {
                "status": "aligned",
                "canonical_runner_set_status": "available",
            },
        },
    }
    capture.write_json(capture.sidecar_path_for(csv_path), sidecar)
    return csv_path


def _write_shepparton_eight_runner_input(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    csv_path = input_dir / "Race 7 - SHEP - 2026-06-10.csv"
    rows = [
        (1, "Shep Alpha"),
        (2, "Shep Bravo"),
        (3, "Shep Charlie"),
        (4, "Shep Delta"),
        (5, "Shep Echo"),
        (6, "Shep Foxtrot"),
        (7, "Shep Golf"),
        (8, "Shep Hotel"),
    ]
    csv_path.write_text(
        "Dog Name,BOX\n" + "".join(f"{box}. {name},\n" for box, name in rows),
        encoding="utf-8",
    )
    sidecar = {
        "metadata_is_leakage_safe": True,
        "prejump_shadow_metadata": {
            "status": "PASS",
            "metadata_is_leakage_safe": True,
            "race_date": "2026-06-10",
            "venue": "SHEP",
            "race_number": "7",
            "jump_time": "2026-06-10T15:00:00+10:00",
            "source_url": (
                "https://www.thedogs.com.au/racing/shepparton/"
                "2026-06-10/7/example"
            ),
            "runner_box_name_list": [
                {"box_number": box, "dog_name": name} for box, name in rows
            ],
            "canonical_final_runner_alignment": {
                "status": "aligned",
                "canonical_runner_set_status": "available",
            },
        },
    }
    capture.write_json(capture.sidecar_path_for(csv_path), sidecar)
    return csv_path


def _plan(input_dir: Path) -> dict:
    return capture.build_capture_plan(
        [input_dir],
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )


def _insert_live_odds_rows(
    db_path: Path,
    rows: list[dict],
    *,
    race_id: str = "Race 1 - WPK - 2026-06-10",
    capture_mode: str = "autonomous_prejump_t30m",
    capture_timestamp: str = "2026-06-10T14:40:00+10:00",
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE live_odds (
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                odds_decimal REAL,
                source_url TEXT,
                capture_timestamp TEXT,
                capture_mode TEXT,
                market_type TEXT,
                source TEXT,
                odds_level TEXT,
                sportsbet_box_source TEXT
            )
            """
        )
        for row in rows:
            conn.execute(
                """
                INSERT INTO live_odds (
                    race_id, dog_name, dog_clean_name, box_number, odds_decimal,
                    source_url, capture_timestamp, capture_mode, market_type,
                    source, odds_level, sportsbet_box_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    race_id,
                    row["dog_name"],
                    row.get("dog_clean_name", row["dog_name"]),
                    row["box_number"],
                    row.get("odds_decimal", 2.0),
                    row.get(
                        "source_url",
                        "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/test/race-1",
                    ),
                    row.get("capture_timestamp", capture_timestamp),
                    capture_mode,
                    row.get("market_type", "win"),
                    row.get("source", "sportsbet"),
                    row.get("odds_level", "dog"),
                    row.get("sportsbet_box_source", "runner_text"),
                ),
            )
        conn.commit()


def test_build_capture_plan_selects_due_window_for_verified_sidecar(tmp_path):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)

    plan = _plan(input_dir)

    item = plan["races"][0]
    assert plan["ready_count"] == 1
    assert item["status"] == "READY_TO_CAPTURE"
    assert item["race_id"] == "Race 1 - WPK - 2026-06-10"
    assert item["capture_window_minutes"] == 30
    assert item["runner_set_validation"]["status"] == "PASS"
    assert item["expected_runners"] == [
        {"box_number": 1, "dog_name": "Alpha", "identity": "ALPHA"},
        {"box_number": 2, "dog_name": "Bravo", "identity": "BRAVO"},
    ]


def test_build_capture_plan_discovers_nested_daemon_eligible_inputs(tmp_path):
    input_dir = tmp_path / "eligible_inputs"
    _write_capture_input(input_dir / "source_0001")

    plan = _plan(input_dir)

    assert plan["ready_count"] == 1
    assert plan["status_counts"] == {"READY_TO_CAPTURE": 1}
    assert plan["races"][0]["race_id"] == "Race 1 - WPK - 2026-06-10"
    assert plan["races"][0]["csv_path"].endswith(
        "eligible_inputs/source_0001/Race 1 - WPK - 2026-06-10.csv"
    )


def test_build_capture_plan_prioritizes_imminent_windows_before_limit(tmp_path):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(
        input_dir,
        race_number=10,
        venue="AAA",
        jump_time="2026-06-10T15:35:00+10:00",
    )
    _write_capture_input(
        input_dir,
        race_number=8,
        venue="BBB",
        jump_time="2026-06-10T14:42:00+10:00",
    )
    _write_capture_input(
        input_dir,
        race_number=9,
        venue="CCC",
        jump_time="2026-06-10T15:05:00+10:00",
    )

    plan = capture.build_capture_plan(
        [input_dir],
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        limit=2,
    )

    assert [item["race_id"] for item in plan["races"]] == [
        "Race 8 - BBB - 2026-06-10",
        "Race 9 - CCC - 2026-06-10",
    ]
    assert [item["capture_window_minutes"] for item in plan["races"]] == [2, 30]


def test_execute_capture_plan_reports_fixed_window_coverage(tmp_path):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=False,
        allow_auto_scrape_odds=False,
    )

    coverage = report["capture_window_coverage"]
    assert coverage["capture_window_offsets_minutes"] == [60, 30, 10, 2]
    assert coverage["race_count"] == 1
    assert coverage["window_count"] == 4
    assert coverage["status_counts"] == {"DUE": 1, "MISSED": 1, "PENDING": 2}
    assert report["candidate_count"] == 1
    assert report["completed_count"] == 1
    assert report["appended_attempt_count"] == 0
    assert report["skipped_already_captured_count"] == 0
    assert report["next_meaningful_action"] == "RUN_ODDS_CAPTURE_NOW"
    assert report["next_meaningful_action_reason"] == "due_capture_windows_present"
    assert report["next_due_capture_window_count"] == 1
    assert report["next_pending_capture_window_count"] == 2
    windows = {row["offset_minutes"]: row for row in coverage["windows"]}
    assert windows[60]["status"] == "MISSED"
    assert windows[60]["reason"] == "earlier_window_passed_without_complete_capture"
    assert windows[30]["status"] == "DUE"
    assert windows[30]["capture_mode"] == "autonomous_prejump_t30m"
    assert windows[10]["status"] == "PENDING"
    assert windows[2]["status"] == "PENDING"


def test_main_writes_report_runtime_identity(tmp_path, monkeypatch):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_live_odds_capture_20260610T144000+1000_unit"
    )
    monkeypatch.setattr(capture, "ROOT", tmp_path)

    result = capture.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--db",
            str(tmp_path / "odds.db"),
            "--current-time",
            "2026-06-10T14:40:00+10:00",
        ]
    )

    report = json.loads(
        (output_dir / "autonomous_live_odds_capture_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert result == 0
    assert report["run_id"] == "20260610T144000+1000_unit"
    assert (
        report["output_dir"]
        == "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_live_odds_capture_20260610T144000+1000_unit"
    )
    assert report["status"] == "READY_REPORT_ONLY"
    assert report["candidate_count"] == 1
    assert report["completed_count"] == 1
    assert report["ready_count"] == 1
    assert report["ready_race_count"] == 1
    assert report["ready_race_ids"] == ["Race 1 - WPK - 2026-06-10"]
    assert report["next_meaningful_action"] == "RUN_ODDS_CAPTURE_NOW"
    assert report["next_meaningful_action_at"] == "2026-06-10T14:40:00+10:00"
    assert report["next_due_capture_window_count"] == 1
    assert report["next_pending_capture_window_count"] == 2


def test_execute_capture_plan_skips_existing_superset_capture_without_retry(
    tmp_path,
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir, jump_time="2026-06-10T15:00:00+10:00")
    db_path = tmp_path / "odds.db"
    _insert_live_odds_rows(
        db_path,
        [
            {"dog_name": "Alpha", "box_number": 1, "odds_decimal": 2.4},
            {"dog_name": "Bravo", "box_number": 2, "odds_decimal": 3.5},
            {"dog_name": "Charlie", "box_number": 3, "odds_decimal": 6.0},
        ],
        capture_mode="autonomous_prejump_t60m",
    )

    plan = capture.build_capture_plan(
        [input_dir],
        current_time=datetime.fromisoformat("2026-06-10T14:15:00+10:00"),
    )
    report = capture.execute_capture_plan(
        plan,
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-06-10T14:15:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:15:00+10:00"),
    )

    assert plan["ready_count"] == 1
    assert plan["races"][0]["status"] == "READY_TO_CAPTURE"
    assert plan["races"][0]["capture_window_minutes"] == 60
    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
    assert report["status"] == "READY"
    assert report["runtime_action"] == "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW"
    assert report["readiness_decision"] == "CONTINUE_ODDS_CAPTURE"
    assert report["attempts"][0]["status"] == "SKIPPED_EXISTING_CAPTURE_SUPERSET"
    assert report["attempts"][0]["reasons"] == [
        "existing_capture_extra_unexpected_runners:3:CHARLIE"
    ]
    windows = {
        row["offset_minutes"]: row for row in report["capture_window_coverage"]["windows"]
    }
    assert windows[60]["status"] == "BLOCKED_EXISTING_CAPTURE_INVALID"
    assert windows[60]["existing_capture_reasons"] == [
        "existing_capture_extra_unexpected_runners:3:CHARLIE"
    ]
    assert windows[30]["status"] == "PENDING"


def test_execute_capture_plan_reports_complete_existing_window_coverage(tmp_path):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)
    db_path = tmp_path / "odds.db"
    _insert_live_odds_rows(
        db_path,
        [
            {"dog_name": "Alpha", "box_number": 1, "odds_decimal": 2.4},
            {"dog_name": "Bravo", "box_number": 2, "odds_decimal": 3.5},
        ],
        capture_mode="autonomous_prejump_t60m",
        capture_timestamp="2026-06-10T14:05:00+10:00",
    )

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=False,
        allow_auto_scrape_odds=False,
    )

    coverage = report["capture_window_coverage"]
    assert coverage["status_counts"] == {
        "CAPTURED": 1,
        "DUE": 1,
        "PENDING": 2,
    }
    windows = {row["offset_minutes"]: row for row in coverage["windows"]}
    assert windows[60]["status"] == "CAPTURED"
    assert windows[60]["existing_capture_status"] == "COMPLETE"
    assert windows[60]["existing_capture_count"] == 2
    assert windows[30]["status"] == "DUE"


def test_execute_capture_plan_recaptures_stale_existing_fixed_window_group(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)
    db_path = tmp_path / "odds.db"
    _insert_live_odds_rows(
        db_path,
        [
            {"dog_name": "Alpha", "box_number": 1, "odds_decimal": 2.4},
            {"dog_name": "Bravo", "box_number": 2, "odds_decimal": 3.5},
        ],
        capture_mode="autonomous_prejump_t30m",
        capture_timestamp="2026-06-10T14:10:00+10:00",
    )
    appended = {}

    def fake_fetch(db_path, venue, race_number, race_date, allow_auto_scrape_odds):
        return {
            "success": True,
            "race_info": {
                "venue_url": (
                    "https://www.sportsbet.com.au/betting/greyhound-racing/"
                    "australia-nz/wentworth-park/race-1"
                ),
                "race_number": 1,
            },
            "odds_data": [
                {
                    "dog_name": "Alpha",
                    "box_number": 1,
                    "odds_decimal": 2.8,
                    "sportsbet_box_source": "runner_text",
                },
                {
                    "dog_name": "Bravo",
                    "box_number": 2,
                    "odds_decimal": 3.1,
                    "sportsbet_box_source": "runner_text",
                },
            ],
        }

    def fake_append(*, db_path, plan_item, validation, current_time):
        appended["capture_mode"] = f"autonomous_prejump_t{plan_item['capture_window_minutes']}m"
        appended["capture_timestamp"] = current_time.isoformat()
        return {
            "status": "SUCCESS",
            "inserted_rows": len(validation["accepted_rows"]),
            "warnings": [],
            "append_only": True,
        }

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)
    monkeypatch.setattr(capture, "append_validated_capture", fake_append)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    assert report["status"] == "APPENDED"
    assert report["runtime_action"] == "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW"
    assert report["readiness_decision"] == "CONTINUE_ODDS_CAPTURE"
    assert report["status_counts"] == {"APPENDED": 1}
    assert report["attempts"][0]["stale_existing_capture"]["status"] == "STALE"
    assert report["attempts"][0]["stale_existing_capture"]["reasons"] == [
        "existing_capture_before_fixed_window"
    ]
    assert appended == {
        "capture_mode": "autonomous_prejump_t30m",
        "capture_timestamp": "2026-06-10T14:40:00+10:00",
    }


def test_execute_capture_plan_defaults_to_report_only(tmp_path):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=False,
        allow_auto_scrape_odds=False,
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_READY_NOT_EXECUTED"
    assert report["status"] == "READY_REPORT_ONLY"
    assert (
        report["runtime_action"]
        == "RUN_WITH_EXECUTE_AND_ALLOW_AUTO_SCRAPE_ODDS_TO_APPEND"
    )
    assert report["readiness_decision"] == "REPORT_ONLY_NO_WRITE"
    assert report["inserted_live_odds_rows"] == 0
    assert report["attempts"][0]["status"] == "PLANNED_NOT_EXECUTED"
    assert report["capture_window_coverage"]["status_counts"] == {
        "DUE": 1,
        "MISSED": 1,
        "PENDING": 2,
    }


def test_execute_capture_plan_appends_after_exact_sportsbet_validation(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)
    appended = {}

    def fake_fetch(db_path, venue, race_number, race_date, allow_auto_scrape_odds):
        return {
            "success": True,
            "race_id": "sportsbet-race-1",
            "race_info": {
                "venue_url": (
                    "https://www.sportsbet.com.au/betting/greyhound-racing/"
                    "australia-nz/wentworth-park/race-1"
                ),
                "race_number": 1,
            },
            "odds_data": [
                {
                    "dog_name": "Alpha",
                    "dog_clean_name": "ALPHA",
                    "box_number": 1,
                    "odds_decimal": 2.4,
                    "sportsbet_box_source": "runner_text",
                    "sportsbet_raw_runner_text": "1. Alpha",
                },
                {
                    "dog_name": "Bravo",
                    "dog_clean_name": "BRAVO",
                    "box_number": 2,
                    "odds_decimal": 3.5,
                    "sportsbet_box_source": "runner_text",
                    "sportsbet_raw_runner_text": "2. Bravo",
                },
            ],
        }

    def fake_append(*, db_path, plan_item, validation, current_time):
        appended["capture_mode"] = f"autonomous_prejump_t{plan_item['capture_window_minutes']}m"
        appended["rows"] = validation["accepted_rows"]
        return {
            "status": "SUCCESS",
            "inserted_rows": len(validation["accepted_rows"]),
            "warnings": [],
            "append_only": True,
        }

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)
    monkeypatch.setattr(capture, "append_validated_capture", fake_append)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    assert report["candidate_count"] == 1
    assert report["completed_count"] == 1
    assert report["appended_attempt_count"] == 1
    assert report["skipped_already_captured_count"] == 0
    assert report["validation_pass_count"] == 1
    assert report["inserted_live_odds_rows"] == 2
    assert appended["capture_mode"] == "autonomous_prejump_t30m"
    assert [row["box_number"] for row in appended["rows"]] == [1, 2]


def test_execute_capture_plan_skips_complete_existing_capture_without_fetch(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)
    db_path = tmp_path / "odds.db"
    _insert_live_odds_rows(
        db_path,
        [
            {"dog_name": "Alpha", "box_number": 1, "odds_decimal": 2.4},
            {"dog_name": "Bravo", "box_number": 2, "odds_decimal": 3.5},
        ],
    )

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch must not run for complete existing capture")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fail_fetch)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    attempt = report["attempts"][0]
    assert attempt["status"] == "SKIPPED_ALREADY_CAPTURED"
    assert attempt["existing_capture_count"] == 2
    assert attempt["existing_capture"]["status"] == "COMPLETE"
    assert attempt["existing_capture"]["missing_expected_runners"] == []


def test_existing_capture_allows_missing_explicit_scratched_expected_runner(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    csv_path = _write_capture_input(input_dir)
    sidecar_path = capture.sidecar_path_for(csv_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["prejump_shadow_metadata"]["runner_box_name_list"][1]["status"] = "SCR"
    capture.write_json(sidecar_path, sidecar)
    db_path = tmp_path / "odds.db"
    _insert_live_odds_rows(
        db_path,
        [{"dog_name": "Alpha", "box_number": 1, "odds_decimal": 2.4}],
    )

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch must not run for complete active existing capture")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fail_fetch)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    attempt = report["attempts"][0]
    assert attempt["status"] == "SKIPPED_ALREADY_CAPTURED"
    assert attempt["existing_capture"]["status"] == "COMPLETE"
    assert attempt["existing_capture"]["expected_runner_count"] == 2
    assert attempt["existing_capture"]["active_expected_runner_count"] == 1
    assert attempt["existing_capture"]["scratched_expected_runners"] == [
        {"box_number": 2, "identity": "BRAVO"}
    ]
    assert attempt["existing_capture"]["missing_expected_runners"] == []


def test_existing_capture_blocks_priced_explicit_scratched_expected_runner(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    csv_path = _write_capture_input(input_dir)
    sidecar_path = capture.sidecar_path_for(csv_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["prejump_shadow_metadata"]["runner_box_name_list"][1]["status"] = "SCR"
    capture.write_json(sidecar_path, sidecar)
    db_path = tmp_path / "odds.db"
    _insert_live_odds_rows(
        db_path,
        [
            {"dog_name": "Alpha", "box_number": 1, "odds_decimal": 2.4},
            {"dog_name": "Bravo", "box_number": 2, "odds_decimal": 3.5},
        ],
    )

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch must not run when existing capture conflicts")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fail_fetch)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    attempt = report["attempts"][0]
    assert attempt["status"] == "BLOCKED_EXISTING_CAPTURE_INVALID"
    assert attempt["existing_capture"]["scratched_expected_runners_with_odds"] == [
        {"box_number": 2, "identity": "BRAVO"}
    ]
    assert (
        "existing_capture_odds_present_for_inactive_expected_runners:2:BRAVO"
        in attempt["reasons"]
    )


def test_execute_capture_plan_blocks_existing_capture_with_incomplete_provenance(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)
    db_path = tmp_path / "odds.db"
    _insert_live_odds_rows(
        db_path,
        [
            {
                "dog_name": "Alpha",
                "box_number": 1,
                "odds_decimal": 2.4,
                "source_url": "",
                "odds_level": "",
            },
            {"dog_name": "Bravo", "box_number": 2, "odds_decimal": 3.5},
        ],
    )

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch must not run while existing capture is invalid")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fail_fetch)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert report["status"] == "BLOCKED"
    assert report["runtime_action"] == "REVIEW_CAPTURE_BLOCKERS_BEFORE_RETRY"
    assert report["readiness_decision"] == "CHECK_BLOCKED_ATTEMPTS"
    attempt = report["attempts"][0]
    assert attempt["status"] == "BLOCKED_EXISTING_CAPTURE_INVALID"
    assert attempt["existing_capture"]["status"] == "INVALID"
    assert attempt["existing_capture"]["invalid_rows"] == [
        {
            "row_index": 1,
            "box_number": 1,
            "identity": "ALPHA",
            "reasons": ["odds_level_missing", "source_url_missing"],
        }
    ]
    assert attempt["reasons"] == [
        "existing_capture_invalid_rows:1",
        "existing_capture_missing_expected_runners:1:ALPHA",
    ]


def test_execute_capture_plan_blocks_partial_existing_capture_without_fetch(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)
    db_path = tmp_path / "odds.db"
    _insert_live_odds_rows(
        db_path,
        [{"dog_name": "Alpha", "box_number": 1, "odds_decimal": 2.4}],
    )

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch must not run for incomplete existing capture")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fail_fetch)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=db_path,
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    attempt = report["attempts"][0]
    assert attempt["status"] == "BLOCKED_EXISTING_CAPTURE_INCOMPLETE"
    assert attempt["existing_capture_count"] == 1
    assert attempt["existing_capture"]["status"] == "INCOMPLETE"
    assert attempt["existing_capture"]["missing_expected_runners"] == [
        {"box_number": 2, "identity": "BRAVO"}
    ]
    assert attempt["reasons"] == ["existing_capture_missing_expected_runners:2:BRAVO"]


def test_execute_capture_plan_records_fetch_exception_and_continues(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(
        input_dir,
        venue="AAA",
        race_number=1,
        jump_time="2026-06-10T14:50:00+10:00",
    )
    _write_capture_input(
        input_dir,
        venue="BBB",
        race_number=2,
        jump_time="2026-06-10T15:00:00+10:00",
    )
    fetch_calls = []

    def fake_fetch(db_path, venue, race_number, race_date, allow_auto_scrape_odds):
        fetch_calls.append((venue, race_number))
        if venue == "AAA":
            raise RuntimeError("stale element")
        return {
            "success": True,
            "race_info": {
                "venue_url": (
                    "https://www.sportsbet.com.au/betting/greyhound-racing/"
                    "australia-nz/test/race-2"
                ),
                "race_number": 2,
            },
            "odds_data": [
                {
                    "dog_name": "Alpha",
                    "box_number": 1,
                    "odds_decimal": 2.4,
                    "sportsbet_box_source": "runner_text",
                },
                {
                    "dog_name": "Bravo",
                    "box_number": 2,
                    "odds_decimal": 3.5,
                    "sportsbet_box_source": "runner_text",
                },
            ],
        }

    def fake_append(*, db_path, plan_item, validation, current_time):
        return {
            "status": "SUCCESS",
            "inserted_rows": len(validation["accepted_rows"]),
            "warnings": [],
            "append_only": True,
        }

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)
    monkeypatch.setattr(capture, "append_validated_capture", fake_append)

    report = capture.execute_capture_plan(
        capture.build_capture_plan(
            [input_dir],
            current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        ),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    assert fetch_calls == [("AAA", 1), ("BBB", 2)]
    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    assert report["status"] == "APPENDED_WITH_BLOCKED_ATTEMPTS"
    assert report["runtime_action"] == "REVIEW_CAPTURE_BLOCKERS_AFTER_APPEND"
    assert report["readiness_decision"] == (
        "CONTINUE_ODDS_CAPTURE_WITH_BLOCKER_REVIEW"
    )
    assert report["status_counts"] == {"BLOCKED_FETCH_EXCEPTION": 1, "APPENDED": 1}
    assert report["inserted_live_odds_rows"] == 2
    assert report["blocked_attempt_count"] == 1
    assert report["attempts"][0]["status"] == "BLOCKED_FETCH_EXCEPTION"
    assert report["attempts"][0]["reasons"] == ["fetch_exception:RuntimeError"]
    assert report["attempts"][1]["status"] == "APPENDED"


def test_execute_capture_plan_blocks_fetch_timeout_and_flushes_progress(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)

    def slow_fetch(db_path, venue, race_number, race_date, allow_auto_scrape_odds):
        time.sleep(2)
        raise AssertionError("timeout should interrupt the fetch")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", slow_fetch)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        fetch_timeout_seconds=0.1,
        progress_dir=tmp_path,
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert report["inserted_live_odds_rows"] == 0
    assert report["status_counts"] == {"BLOCKED_FETCH_TIMEOUT": 1}
    attempt = report["attempts"][0]
    assert attempt["status"] == "BLOCKED_FETCH_TIMEOUT"
    assert attempt["reasons"] == ["fetch_timeout:0.1s"]
    progress_text = (
        tmp_path / "autonomous_live_odds_capture_attempts.progress.jsonl"
    ).read_text(encoding="utf-8")
    assert "BLOCKED_FETCH_TIMEOUT" in progress_text


def test_validate_fetched_odds_accepts_reserve_runner_text_final_box():
    plan_item = {
        "race_id": "Race 12 - MAND - 2026-06-11",
        "race_number": 12,
        "expected_runners": [
            {"box_number": 6, "dog_name": "High Rollin", "identity": "HIGHROLLIN"},
        ],
    }
    fetch_result = {
        "success": True,
        "race_info": {
            "race_number": 12,
            "venue_url": (
                "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
                "mandurah/race-12-10573646"
            ),
        },
        "odds_data": [
            {
                "dog_name": "High Rollin'",
                "dog_clean_name": "HIGH ROLLIN",
                "box_number": 9,
                "odds_decimal": 8.0,
                "sportsbet_box_source": "runner_text",
                "sportsbet_list_position": 8,
                "sportsbet_raw_runner_text": "9. High Rollin' (6)\nF: 388638\n8.00",
            },
        ],
    }

    validation = capture.validate_fetched_odds(plan_item, fetch_result)

    assert validation["status"] == "PASS"
    assert validation["reasons"] == []
    assert validation["accepted_rows"][0]["box_number"] == 6
    assert validation["accepted_rows"][0]["identity"] == "HIGHROLLIN"


def test_validate_fetched_odds_accepts_abbreviated_name_period_runner_text():
    plan_item = {
        "race_id": "Race 4 - BEN - 2026-06-12",
        "race_number": 4,
        "expected_runners": [
            {"box_number": 3, "dog_name": "Dr. Will", "identity": "DRWILL"},
        ],
    }
    fetch_result = {
        "success": True,
        "race_info": {
            "race_number": 4,
            "venue_url": (
                "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
                "bendigo/race-4-10577353"
            ),
        },
        "odds_data": [
            {
                "dog_name": "Dr. Will",
                "dog_clean_name": "DR WILL",
                "box_number": 3,
                "odds_decimal": 8.5,
                "sportsbet_box_source": "runner_text",
                "sportsbet_list_position": 3,
                "sportsbet_raw_runner_text": "3. Dr. Will (3)\nF: 241112\n8.50",
            },
        ],
    }

    validation = capture.validate_fetched_odds(plan_item, fetch_result)

    assert validation["status"] == "PASS"
    assert validation["reasons"] == []
    assert validation["accepted_rows"][0]["box_number"] == 3
    assert validation["accepted_rows"][0]["identity"] == "DRWILL"


def test_validate_fetched_odds_allows_missing_explicit_scratched_expected_runner():
    plan_item = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "race_number": 1,
        "expected_runners": [
            {"box_number": 1, "dog_name": "Alpha", "identity": "ALPHA"},
            {
                "box_number": 2,
                "dog_name": "Bravo",
                "identity": "BRAVO",
                "status": "SCR",
            },
        ],
    }
    fetch_result = {
        "success": True,
        "race_info": {
            "race_number": 1,
            "venue_url": (
                "https://www.sportsbet.com.au/betting/greyhound-racing/"
                "australia-nz/wentworth-park/race-1"
            ),
        },
        "odds_data": [
            {
                "dog_name": "Alpha",
                "box_number": 1,
                "odds_decimal": 2.4,
                "sportsbet_box_source": "runner_text",
            },
        ],
    }

    validation = capture.validate_fetched_odds(plan_item, fetch_result)

    assert validation["status"] == "PASS"
    assert validation["expected_runner_count"] == 2
    assert validation["active_expected_runner_count"] == 1
    assert validation["scratched_expected_runners"] == [
        {"box_number": 2, "identity": "BRAVO"}
    ]
    assert validation["missing_expected_runners"] == []
    assert validation["reasons"] == []


def test_validate_fetched_odds_missing_unmarked_runner_still_blocks():
    plan_item = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "race_number": 1,
        "expected_runners": [
            {"box_number": 1, "dog_name": "Alpha", "identity": "ALPHA"},
            {"box_number": 2, "dog_name": "Bravo", "identity": "BRAVO"},
        ],
    }
    fetch_result = {
        "success": True,
        "race_info": {
            "race_number": 1,
            "venue_url": (
                "https://www.sportsbet.com.au/betting/greyhound-racing/"
                "australia-nz/wentworth-park/race-1"
            ),
        },
        "odds_data": [
            {
                "dog_name": "Alpha",
                "box_number": 1,
                "odds_decimal": 2.4,
                "sportsbet_box_source": "runner_text",
            },
        ],
    }

    validation = capture.validate_fetched_odds(plan_item, fetch_result)

    assert validation["status"] == "FAIL"
    assert validation["missing_expected_runners"] == [
        {"box_number": 2, "identity": "BRAVO"}
    ]
    assert validation["scratched_expected_runners"] == []
    assert "sportsbet_missing_expected_runners:2:BRAVO" in validation["reasons"]


def test_validate_fetched_odds_classifies_partial_win_market_when_place_complete():
    plan_item = {
        "race_id": "Race 7 - SHEP - 2026-06-10",
        "race_number": 7,
        "expected_runners": [
            {"box_number": 1, "dog_name": "Shep Alpha", "identity": "SHEPALPHA"},
            {"box_number": 2, "dog_name": "Shep Bravo", "identity": "SHEPBRAVO"},
            {"box_number": 3, "dog_name": "Shep Charlie", "identity": "SHEPCHARLIE"},
            {"box_number": 4, "dog_name": "Shep Delta", "identity": "SHEPDELTA"},
            {"box_number": 5, "dog_name": "Shep Echo", "identity": "SHEPECHO"},
            {"box_number": 6, "dog_name": "Shep Foxtrot", "identity": "SHEPFOXTROT"},
            {"box_number": 7, "dog_name": "Shep Golf", "identity": "SHEPGOLF"},
            {"box_number": 8, "dog_name": "Shep Hotel", "identity": "SHEPHOTEL"},
        ],
    }
    fetch_result = {
        "success": True,
        "win_count": 4,
        "place_count": 8,
        "race_info": {
            "race_number": 7,
            "venue_url": (
                "https://www.sportsbet.com.au/betting/greyhound-racing/"
                "australia-nz/shepparton/race-7"
            ),
        },
        "odds_data": [
            {
                "dog_name": "Shep Alpha",
                "box_number": 1,
                "odds_decimal": 2.4,
                "sportsbet_box_source": "runner_text",
            },
            {
                "dog_name": "Shep Bravo",
                "box_number": 2,
                "odds_decimal": 3.5,
                "sportsbet_box_source": "runner_text",
            },
            {
                "dog_name": "Shep Charlie",
                "box_number": 3,
                "odds_decimal": 4.8,
                "sportsbet_box_source": "runner_text",
            },
            {
                "dog_name": "Shep Delta",
                "box_number": 4,
                "odds_decimal": 6.0,
                "sportsbet_box_source": "runner_text",
            },
        ],
    }

    validation = capture.validate_fetched_odds(plan_item, fetch_result)

    assert validation["status"] == "FAIL"
    assert validation["active_expected_runner_count"] == 8
    assert validation["accepted_row_count"] == 4
    assert validation["failure_root_cause"] == "sportsbet_win_market_partial_but_place_complete"
    assert validation["failure_detail"] == {
        "active_expected_runner_count": 8,
        "accepted_win_row_count": 4,
        "missing_active_runner_count": 4,
        "extra_unexpected_runner_count": 0,
        "fetch_win_count": 4,
        "fetch_place_count": 8,
        "root_cause": "sportsbet_win_market_partial_but_place_complete",
    }
    assert validation["missing_expected_runners"] == [
        {"box_number": 5, "identity": "SHEPECHO"},
        {"box_number": 6, "identity": "SHEPFOXTROT"},
        {"box_number": 7, "identity": "SHEPGOLF"},
        {"box_number": 8, "identity": "SHEPHOTEL"},
    ]
    assert validation["extra_unexpected_runners"] == []
    assert any(
        reason.startswith("sportsbet_win_market_partial_but_place_complete:")
        for reason in validation["reasons"]
    )


def test_validate_fetched_odds_does_not_classify_zero_accepted_rows_as_partial_market():
    plan_item = {
        "race_id": "Race 7 - SHEP - 2026-06-10",
        "race_number": 7,
        "expected_runners": [
            {"box_number": 1, "dog_name": "Shep Alpha", "identity": "SHEPALPHA"},
            {"box_number": 2, "dog_name": "Shep Bravo", "identity": "SHEPBRAVO"},
        ],
    }
    fetch_result = {
        "success": False,
        "win_count": 0,
        "place_count": 0,
        "race_info": {
            "race_number": 7,
            "venue_url": (
                "https://www.sportsbet.com.au/betting/greyhound-racing/"
                "australia-nz/shepparton/race-7"
            ),
        },
        "odds_data": [],
    }

    validation = capture.validate_fetched_odds(plan_item, fetch_result)

    assert validation["status"] == "FAIL"
    assert validation["accepted_row_count"] == 0
    assert validation["failure_root_cause"] is None
    assert "sportsbet_accepted_runner_rows_zero" in validation["reasons"]
    assert not any(
        reason.startswith("partial_same_race_win_market:")
        or reason.startswith("sportsbet_win_market_partial_but_place_complete:")
        for reason in validation["reasons"]
    )


def test_validate_fetched_odds_keeps_extra_identity_mismatch_distinct():
    plan_item = {
        "race_id": "Race 7 - SHEP - 2026-06-10",
        "race_number": 7,
        "expected_runners": [
            {"box_number": 1, "dog_name": "Shep Alpha", "identity": "SHEPALPHA"},
            {"box_number": 2, "dog_name": "Shep Bravo", "identity": "SHEPBRAVO"},
        ],
    }
    fetch_result = {
        "success": True,
        "win_count": 2,
        "place_count": 2,
        "race_info": {
            "race_number": 7,
            "venue_url": (
                "https://www.sportsbet.com.au/betting/greyhound-racing/"
                "australia-nz/shepparton/race-7"
            ),
        },
        "odds_data": [
            {
                "dog_name": "Shep Alpha",
                "box_number": 1,
                "odds_decimal": 2.4,
                "sportsbet_box_source": "runner_text",
            },
            {
                "dog_name": "Wrong Race Dog",
                "box_number": 8,
                "odds_decimal": 3.5,
                "sportsbet_box_source": "runner_text",
            },
        ],
    }

    validation = capture.validate_fetched_odds(plan_item, fetch_result)

    assert validation["status"] == "FAIL"
    assert validation["failure_root_cause"] == "sportsbet_unexpected_runner_identity_mismatch"
    assert validation["failure_root_cause"] != "sportsbet_win_market_partial_but_place_complete"
    assert validation["extra_unexpected_runners"] == [
        {"box_number": 8, "identity": "WRONGRACEDOG"}
    ]
    assert any(
        reason.startswith("sportsbet_unexpected_runner_identity_mismatch:")
        for reason in validation["reasons"]
    )


def test_validate_fetched_odds_blocks_priced_explicit_scratched_expected_runner():
    plan_item = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "race_number": 1,
        "expected_runners": [
            {"box_number": 1, "dog_name": "Alpha", "identity": "ALPHA"},
            {"box_number": 2, "dog_name": "Bravo", "identity": "BRAVO", "status": "SCR"},
        ],
    }
    fetch_result = {
        "success": True,
        "race_info": {
            "race_number": 1,
            "venue_url": (
                "https://www.sportsbet.com.au/betting/greyhound-racing/"
                "australia-nz/wentworth-park/race-1"
            ),
        },
        "odds_data": [
            {
                "dog_name": "Alpha",
                "box_number": 1,
                "odds_decimal": 2.4,
                "sportsbet_box_source": "runner_text",
            },
            {
                "dog_name": "Bravo",
                "box_number": 2,
                "odds_decimal": 3.5,
                "sportsbet_box_source": "runner_text",
            },
        ],
    }

    validation = capture.validate_fetched_odds(plan_item, fetch_result)

    assert validation["status"] == "FAIL"
    assert validation["missing_expected_runners"] == []
    assert validation["scratched_expected_runners_with_odds"] == [
        {"box_number": 2, "identity": "BRAVO"}
    ]
    assert (
        "sportsbet_odds_present_for_scratched_expected_runners:2:BRAVO"
        in validation["reasons"]
    )


def test_execute_capture_plan_rechecks_prejump_time_before_append(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)

    def fake_fetch(db_path, venue, race_number, race_date, allow_auto_scrape_odds):
        return {
            "success": True,
            "race_info": {
                "venue_url": (
                    "https://www.sportsbet.com.au/betting/greyhound-racing/"
                    "australia-nz/wentworth-park/race-1"
                ),
                "race_number": 1,
            },
            "odds_data": [
                {
                    "dog_name": "Alpha",
                    "box_number": 1,
                    "odds_decimal": 2.4,
                    "sportsbet_box_source": "runner_text",
                },
                {
                    "dog_name": "Bravo",
                    "box_number": 2,
                    "odds_decimal": 3.5,
                    "sportsbet_box_source": "runner_text",
                },
            ],
        }

    def fail_append(**_kwargs):
        raise AssertionError("append must not run after jump time")

    times = iter(
        [
            datetime.fromisoformat("2026-06-10T14:50:00+10:00"),
            datetime.fromisoformat("2026-06-10T15:00:01+10:00"),
        ]
    )
    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)
    monkeypatch.setattr(capture, "append_validated_capture", fail_append)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: next(times),
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert report["validation_pass_count"] == 1
    assert report["inserted_live_odds_rows"] == 0
    assert report["blocked_attempt_count"] == 1
    blocked = report["blocked_attempts"][0]
    assert blocked["status"] == "BLOCKED_TIME_GATE_BEFORE_APPEND"
    assert blocked["fetch_time"] == "2026-06-10T14:50:00+10:00"
    assert blocked["append_time"] == "2026-06-10T15:00:01+10:00"
    assert blocked["fresh_plan_status"] == "NO_DUE_WINDOW"
    assert blocked["fresh_minutes_to_jump"] < 0
    assert blocked["validation_status"] == "PASS"
    attempt = report["attempts"][0]
    assert attempt["status"] == "BLOCKED_TIME_GATE_BEFORE_APPEND"
    assert "race_already_jumped" in attempt["reasons"]


def test_execute_capture_plan_reports_t2_late_time_gate_miss(tmp_path, monkeypatch):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir, jump_time="2026-06-10T14:42:00+10:00")

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch must not run after jump time")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fail_fetch)

    report = capture.execute_capture_plan(
        capture.build_capture_plan(
            [input_dir],
            current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        ),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat(
            "2026-06-10T14:42:01+10:00"
        ),
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert report["t2_miss_attempt_count"] == 1
    assert report["t2_miss_cause_counts"] == {"t2_miss_late_time_gate": 1}
    assert report["t2_miss_examples"] == [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "capture_window_minutes": 2,
            "status": "BLOCKED_TIME_GATE_BEFORE_FETCH",
            "cause": "t2_miss_late_time_gate",
            "reasons": ["race_already_jumped"],
            "fetch_time": "2026-06-10T14:42:01+10:00",
            "append_time": None,
            "fresh_plan_status": "NO_DUE_WINDOW",
            "fresh_minutes_to_jump": -1 / 60,
        }
    ]


def test_execute_capture_plan_blocks_mismatched_or_ambiguous_sportsbet_rows(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_capture_input(input_dir)

    def fake_fetch(db_path, venue, race_number, race_date, allow_auto_scrape_odds):
        return {
            "success": True,
            "race_info": {
                "venue_url": (
                    "https://www.sportsbet.com.au/betting/greyhound-racing/"
                    "australia-nz/wentworth-park/race-1"
                ),
                "race_number": 1,
            },
            "odds_data": [
                {
                    "dog_name": "Alpha",
                    "box_number": 1,
                    "odds_decimal": 2.4,
                    "sportsbet_box_source": "runner_text",
                },
                {
                    "dog_name": "Bravo",
                    "box_number": 3,
                    "odds_decimal": 3.5,
                    "sportsbet_box_source": "list_position_fallback",
                },
            ],
        }

    def fail_append(**_kwargs):
        raise AssertionError("append must not run after failed validation")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)
    monkeypatch.setattr(capture, "append_validated_capture", fail_append)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert report["inserted_live_odds_rows"] == 0
    assert report["blocked_attempt_count"] == 1
    assert report["blocked_attempts"][0]["race_id"] == "Race 1 - WPK - 2026-06-10"
    assert report["blocked_attempts"][0]["status"] == "BLOCKED_VALIDATION_FAILED"
    assert report["blocked_attempts"][0]["validation_status"] == "FAIL"
    assert report["blocked_attempts"][0]["validation_expected_runner_count"] == 2
    assert report["blocked_attempts"][0]["validation_accepted_row_count"] == 1
    assert (
        report["blocked_attempts"][0]["validation_missing_expected_runner_count"] == 1
    )
    attempt = report["attempts"][0]
    assert attempt["status"] == "BLOCKED_VALIDATION_FAILED"
    assert "sportsbet_rejected_runner_rows:1" in attempt["reasons"]
    assert any(
        reason.startswith("sportsbet_missing_expected_runners:")
        for reason in attempt["reasons"]
    )


def test_execute_capture_plan_reports_partial_win_market_place_complete(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "upcoming"
    _write_shepparton_eight_runner_input(input_dir)

    def fake_fetch(db_path, venue, race_number, race_date, allow_auto_scrape_odds):
        return {
            "success": True,
            "win_count": 4,
            "place_count": 8,
            "race_info": {
                "venue_url": (
                    "https://www.sportsbet.com.au/betting/greyhound-racing/"
                    "australia-nz/shepparton/race-7"
                ),
                "race_number": 7,
            },
            "odds_data": [
                {
                    "dog_name": "Shep Alpha",
                    "box_number": 1,
                    "odds_decimal": 2.4,
                    "sportsbet_box_source": "runner_text",
                },
                {
                    "dog_name": "Shep Bravo",
                    "box_number": 2,
                    "odds_decimal": 3.5,
                    "sportsbet_box_source": "runner_text",
                },
                {
                    "dog_name": "Shep Charlie",
                    "box_number": 3,
                    "odds_decimal": 4.8,
                    "sportsbet_box_source": "runner_text",
                },
                {
                    "dog_name": "Shep Delta",
                    "box_number": 4,
                    "odds_decimal": 6.0,
                    "sportsbet_box_source": "runner_text",
                },
            ],
        }

    def fail_append(**_kwargs):
        raise AssertionError("append must not run after failed validation")

    monkeypatch.setattr(capture, "fetch_odds_for_target_race", fake_fetch)
    monkeypatch.setattr(capture, "append_validated_capture", fail_append)

    report = capture.execute_capture_plan(
        _plan(input_dir),
        db_path=tmp_path / "odds.db",
        current_time=datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
        execute=True,
        allow_auto_scrape_odds=True,
        current_time_provider=lambda: datetime.fromisoformat("2026-06-10T14:40:00+10:00"),
    )

    assert report["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert report["inserted_live_odds_rows"] == 0
    blocked = report["blocked_attempts"][0]
    assert blocked["race_id"] == "Race 7 - SHEP - 2026-06-10"
    assert blocked["fetch_win_count"] == 4
    assert blocked["fetch_place_count"] == 8
    assert blocked["validation_active_expected_runner_count"] == 8
    assert blocked["validation_accepted_row_count"] == 4
    assert blocked["validation_missing_expected_runner_count"] == 4
    assert (
        blocked["validation_failure_root_cause"]
        == "sportsbet_win_market_partial_but_place_complete"
    )
    assert blocked["validation_failure_detail"]["fetch_place_count"] == 8
    attempt = report["attempts"][0]
    assert attempt["status"] == "BLOCKED_VALIDATION_FAILED"
    assert (
        attempt["validation"]["failure_root_cause"]
        == "sportsbet_win_market_partial_but_place_complete"
    )
    assert any(
        reason.startswith("sportsbet_win_market_partial_but_place_complete:")
        for reason in attempt["reasons"]
    )
