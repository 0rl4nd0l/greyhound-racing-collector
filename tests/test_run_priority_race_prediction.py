from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from upcoming_race_browser import UpcomingRaceBrowser
from scripts.refresh_prejump_upcoming import select_prejump_races
from scripts.run_priority_race_prediction import (
    CaptureHandoffError,
    FIXED_CAPTURE_WINDOWS_MINUTES,
    _fixed_window_source_matches,
    acquire_with_bounded_wait,
    canonical_json,
    discover_capture_handoff,
    main,
    resolve_target_race,
    run_command,
    wait_for_lock_or_handoff,
)


NOW = datetime.fromisoformat("2026-07-18T12:00:00+10:00")


def race(*, venue="SAN", number=7, race_time="13:00", url="https://thedogs.com.au/racing/sandown/2026-07-18/7"):
    return {
        "venue": venue,
        "race_number": number,
        "date": "2026-07-18",
        "race_time": race_time,
        "url": url,
    }


def args(tmp_path: Path, **overrides):
    values = {
        "race_id": "Race 7 - SAN - 2026-07-18",
        "race": None,
        "execute_collection": False,
        "allow_auto_scrape_odds": False,
        "require_autonomous_handoff": False,
        "days_ahead": 1,
        "max_wait_seconds": 0.0,
        "poll_seconds": 0.1,
        "fetch_timeout_seconds": 1.0,
        "db": tmp_path / "db.sqlite",
        "model_dir": tmp_path / "model",
        "lock_path": tmp_path / "runtime.lock",
        "lock_output_dir": tmp_path,
        "lock_stale_seconds": 60,
        "capture_evidence_root": [tmp_path / "evidence"],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_exact_target_resolution_fails_closed_for_missing_and_ambiguous():
    status, selected, matches = resolve_target_race(
        [race()], race_id="Race 1 - SAN - 2026-07-18", race_query=None
    )
    assert (status, selected, matches) == ("BLOCKED_RACE_NOT_FOUND", None, [])

    duplicate = race(url="https://thedogs.com.au/racing/sandown/2026-07-18/7?copy=1")
    status, selected, matches = resolve_target_race(
        [race(), duplicate], race_id="Race 7 - SAN - 2026-07-18", race_query=None
    )
    assert status == "BLOCKED_RACE_AMBIGUOUS"
    assert selected is None
    assert matches == ["Race 7 - SAN - 2026-07-18"]


def test_named_race_query_is_exact_and_unambiguous():
    status, selected, _ = resolve_target_race(
        [race()], race_id=None, race_query="Sandown race 7"
    )
    assert status == "RESOLVED"
    assert selected == race()


def test_plan_only_is_default_write_free_and_deterministic(tmp_path):
    command_args = args(tmp_path)
    first = run_command(command_args, races=[race()], current_time=NOW)
    second = run_command(command_args, races=[race()], current_time=NOW)

    assert first == second
    assert first["status"] == "PLAN_ONLY"
    assert first["fixed_capture_windows_minutes"] == list(
        FIXED_CAPTURE_WINDOWS_MINUTES
    )
    assert first["next_capture_window_minutes"] == 60
    assert first["persisted"] is False
    assert not tmp_path.joinpath("runtime.lock").exists()
    assert canonical_json(first) == canonical_json(second)


def test_schedule_browser_can_read_without_creating_default_upcoming_dir(
    tmp_path, monkeypatch
):
    upcoming = tmp_path / "must-not-be-created"
    monkeypatch.setenv("UPCOMING_RACES_DIR", str(upcoming))
    UpcomingRaceBrowser(create_upcoming_dir=False)
    assert not upcoming.exists()


def test_already_jumped_target_is_blocked_before_execution(tmp_path):
    output = run_command(
        args(tmp_path, execute_collection=True, allow_auto_scrape_odds=True),
        races=[race(race_time="11:59")],
        current_time=NOW,
    )
    assert output["status"] == "BLOCKED_RACE_ALREADY_JUMPED"


def test_collection_requires_second_explicit_scrape_gate(tmp_path):
    output = run_command(
        args(tmp_path, execute_collection=True), races=[race()], current_time=NOW
    )
    assert output["status"] == "BLOCKED_ODDS_CAPTURE"
    assert output["reason"] == "allow_auto_scrape_odds_flag_not_set"


class Busy(RuntimeError):
    def __init__(self):
        self.payload = {"reason": "active_lock_present"}


def test_lock_wait_is_bounded_and_reports_owner():
    clock = iter([0.0, 0.0, 0.5, 1.0, 1.0])
    slept = []

    lock, waited, details = acquire_with_bounded_wait(
        acquire=lambda: (_ for _ in ()).throw(Busy()),
        busy_type=Busy,
        max_wait_seconds=1.0,
        poll_seconds=0.5,
        monotonic=lambda: next(clock),
        sleeper=slept.append,
    )
    assert lock is None
    assert waited == 1.0
    assert details == {"reason": "active_lock_present"}
    assert slept == [0.5, 0.5]


def _execution_dependencies(tmp_path, *, plan_status="READY_TO_CAPTURE", attempt_status="APPENDED", reasons=None):
    released = []

    def refresh(refresh_args):
        csv_path = Path(refresh_args.upcoming_dir) / "Race 7 - SAN - 2026-07-18.csv"
        sidecar = csv_path.with_name(csv_path.name + ".metadata.json")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("runner\n", encoding="utf-8")
        sidecar.write_text("{}\n", encoding="utf-8")
        return {
            "status": "SUCCESS",
            "selected_count": 1,
            "sidecar_metadata_coverage": {
                "status": "READY",
                "races": [{"csv_path": str(csv_path), "sidecar_path": str(sidecar)}],
            },
        }

    def capture_plan(input_dirs, *, current_time, limit):
        return {
            "races": [
                {
                    "race_id": "Race 7 - SAN - 2026-07-18",
                    "status": plan_status,
                    "capture_window_minutes": 60,
                }
            ],
            "ready_count": 1 if plan_status == "READY_TO_CAPTURE" else 0,
        }

    def capture_execute(*_args, **_kwargs):
        return {
            "attempts": [
                {
                    "race_id": "Race 7 - SAN - 2026-07-18",
                    "status": attempt_status,
                    "reasons": list(reasons or []),
                }
            ],
            "status_counts": {attempt_status: 1},
            "inserted_live_odds_rows": 16 if attempt_status == "APPENDED" else 0,
        }

    def seal(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for key, filename in (
            ("feature_rows", "shadow_feature_rows.json"),
            ("feature_manifest", "shadow_manifest.json"),
            ("implementation_manifest", "implementation_file_manifest.json"),
        ):
            paths[key] = output_dir / filename
            paths[key].write_text("{}\n", encoding="utf-8")
        return paths

    def score(**kwargs):
        assert kwargs["score_timestamp"] == NOW
        return {
            "status": "MANUAL_PREJUMP_FROZEN_RESIDUAL_PREDICTION",
            "probability_sums": {"market": 1.0, "half": 1.0, "full": 1.0},
            "predictions": [
                {"box": 1, "half_probability": 0.4, "full_probability": 0.5},
                {"box": 2, "half_probability": 0.6, "full_probability": 0.5},
            ],
            "persisted": False,
            "outcomes_present": False,
        }

    return {
        "refresh_fn": refresh,
        "capture_plan_fn": capture_plan,
        "capture_execute_fn": capture_execute,
        "feature_seal_fn": seal,
        "score_fn": score,
        "acquire_fn": lambda **_kwargs: {"run_id": "test"},
        "release_fn": lambda path, run_id: released.append((path, run_id)) or {"released": True},
        "busy_type": Busy,
        "released": released,
    }


def test_explicit_execution_produces_only_nonpersisted_normalized_stdout_payload(tmp_path):
    dependencies = _execution_dependencies(tmp_path)
    released = dependencies.pop("released")
    output = run_command(
        args(tmp_path, execute_collection=True, allow_auto_scrape_odds=True),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: NOW,
        **dependencies,
    )
    assert output["status"] == "PREDICTION_READY"
    assert output["inserted_live_odds_rows"] == 16
    assert output["prediction"]["probability_sums"] == {
        "market": 1.0,
        "half": 1.0,
        "full": 1.0,
    }
    assert output["prediction"]["persisted"] is False
    assert output["feature_packet_ephemeral"] is True
    assert len(released) == 1


def test_feature_and_score_use_fresh_post_capture_timestamps(tmp_path):
    dependencies = _execution_dependencies(tmp_path)
    dependencies.pop("released")
    collection_time = NOW.replace(second=30)
    feature_time = NOW.replace(minute=1)
    score_time = NOW.replace(minute=2)
    seen = {}
    original_refresh = dependencies["refresh_fn"]
    original_capture_plan = dependencies["capture_plan_fn"]
    original_seal = dependencies["feature_seal_fn"]
    original_score = dependencies["score_fn"]

    def refresh(refresh_args):
        seen["refresh_time"] = refresh_args.current_time
        return original_refresh(refresh_args)

    def capture_plan(input_dirs, *, current_time, limit):
        seen["capture_plan_time"] = current_time
        return original_capture_plan(
            input_dirs, current_time=current_time, limit=limit
        )

    def seal(**kwargs):
        seen["feature_time"] = kwargs["current_time"]
        return original_seal(**kwargs)

    def score(**kwargs):
        seen["score_time"] = kwargs["score_timestamp"]
        kwargs["score_timestamp"] = NOW
        return original_score(**kwargs)

    dependencies["refresh_fn"] = refresh
    dependencies["capture_plan_fn"] = capture_plan
    dependencies["feature_seal_fn"] = seal
    dependencies["score_fn"] = score
    times = iter([collection_time, feature_time, score_time])
    output = run_command(
        args(tmp_path, execute_collection=True, allow_auto_scrape_odds=True),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: next(times),
        **dependencies,
    )
    assert output["status"] == "PREDICTION_READY"
    assert seen == {
        "refresh_time": collection_time.isoformat(),
        "capture_plan_time": collection_time,
        "feature_time": feature_time,
        "score_time": score_time,
    }


def test_not_due_fixed_window_releases_lock_and_returns_wait_status(tmp_path):
    dependencies = _execution_dependencies(tmp_path, plan_status="TOO_EARLY_FOR_FIXED_WINDOW")
    released = dependencies.pop("released")
    output = run_command(
        args(tmp_path, execute_collection=True, allow_auto_scrape_odds=True),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: NOW,
        **dependencies,
    )
    assert output["status"] == "WAITING_FOR_CAPTURE_WINDOW"
    assert output["next_capture_window_minutes"] == 60
    assert len(released) == 1


@pytest.mark.parametrize(
    ("attempt_status", "reasons", "expected"),
    [
        ("BLOCKED_VALIDATION_FAILED", ["runner_identity_mismatch"], "BLOCKED_RUNNER_IDENTITY"),
        ("BLOCKED_FETCH_TIMEOUT", ["fetch_timeout:1s"], "BLOCKED_ODDS_CAPTURE"),
        ("SKIPPED_ALREADY_CAPTURED", [], "BLOCKED_ODDS_CAPTURE"),
    ],
)
def test_capture_failures_and_idempotent_retry_fail_closed(
    tmp_path, attempt_status, reasons, expected
):
    dependencies = _execution_dependencies(
        tmp_path, attempt_status=attempt_status, reasons=reasons
    )
    dependencies.pop("released")
    output = run_command(
        args(tmp_path, execute_collection=True, allow_auto_scrape_odds=True),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: NOW,
        **dependencies,
    )
    assert output["status"] == expected
    assert int(output.get("inserted_live_odds_rows") or 0) == 0
    if attempt_status == "SKIPPED_ALREADY_CAPTURED":
        assert output["idempotent_existing_capture"] is True


def test_refresh_selector_never_admits_unrelated_races():
    selected, records = select_prejump_races(
        [race(), race(venue="MEA", number=4, race_time="13:10", url="https://thedogs.com.au/racing/the-meadows/2026-07-18/4")],
        now=NOW,
        min_minutes=0,
        max_minutes=180,
        include_race_ids={"Race 7 - SAN - 2026-07-18"},
    )
    assert selected == [race()]
    assert [row["bucket"] for row in records] == [
        "preferred_window",
        "not_exact_included_race",
    ]


def test_cli_routes_dependency_noise_to_stderr_and_emits_one_canonical_stdout(
    monkeypatch, capsys
):
    class NoisyBrowser:
        def __init__(self, *, create_upcoming_dir):
            assert create_upcoming_dir is False

        def get_upcoming_races(self, *, days_ahead):
            assert days_ahead == 1
            print("browser diagnostic")
            return [race()]

    monkeypatch.setattr(
        "upcoming_race_browser.UpcomingRaceBrowser", NoisyBrowser
    )
    exit_code = main(
        [
            "--race-id",
            "Race 7 - SAN - 2026-07-18",
            "--current-time",
            NOW.isoformat(),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == "browser diagnostic\n"
    stdout_lines = captured.out.splitlines()
    assert len(stdout_lines) == 1
    payload = json.loads(stdout_lines[0])
    assert payload["status"] == "PLAN_ONLY"
    assert payload["persisted"] is False


def _handoff_fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "evidence"
    source_dir = root / "shadow_autopilot_v1_20260718T120000+1000"
    source_dir.mkdir(parents=True)
    form_path = source_dir / "Race 7 - SAN - 2026-07-18.csv"
    form_bytes = b"box,dog\n1,Alpha\n2,Bravo\n"
    form_path.write_bytes(form_bytes)
    sidecar_path = form_path.with_name(form_path.name + ".metadata.json")
    sidecar_bytes = json.dumps(
        {
            "filename": form_path.name,
            "content_length": len(form_bytes),
            "content_sha256": hashlib.sha256(form_bytes).hexdigest(),
            "metadata_captured_at": "2026-07-18T11:59:50+10:00",
        },
        sort_keys=True,
    ).encode()
    sidecar_path.write_bytes(sidecar_bytes)

    capture_dir = (
        root
        / "autonomous_live_odds_capture_20260718T120000+1000_odds_capture_autopilot"
    )
    capture_dir.mkdir()
    race_id = "Race 7 - SAN - 2026-07-18"
    generated_at = "2026-07-18T12:00:00+10:00"
    fetch_time = "2026-07-18T12:00:10+10:00"
    append_time = "2026-07-18T12:00:20+10:00"
    source_url = (
        "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
        "sandown/race-7-123456"
    )
    runners = [
        {"box_number": 1, "dog_name": "Alpha", "identity": "ALPHA"},
        {"box_number": 2, "dog_name": "Bravo", "identity": "BRAVO"},
    ]
    win_rows = [
        {**runners[0], "odds_decimal": 2.5, "sportsbet_box_source": "explicit_dom"},
        {**runners[1], "odds_decimal": 3.5, "sportsbet_box_source": "explicit_dom"},
    ]
    place_rows = [
        {**runners[0], "odds_decimal": 1.4, "sportsbet_box_source": "explicit_dom"},
        {**runners[1], "odds_decimal": 1.8, "sportsbet_box_source": "explicit_dom"},
    ]
    plan = {
        "schema_version": "autonomous_live_odds_capture_plan_v1",
        "generated_at": generated_at,
        "races": [
            {
                "schema_version": "autonomous_live_odds_capture_plan_item_v1",
                "status": "READY_TO_CAPTURE",
                "race_id": race_id,
                "venue": "SAN",
                "race_number": 7,
                "race_date": "2026-07-18",
                "jump_datetime": "2026-07-18T13:00:00+10:00",
                "capture_window_minutes": 60,
                "csv_path": str(form_path),
                "sidecar_path": str(sidecar_path),
                "thedogs_source_url": (
                    "https://www.thedogs.com.au/racing/sandown/2026-07-18/7"
                ),
                "expected_runners": runners,
            }
        ],
    }
    validation = {
        "schema_version": "autonomous_live_odds_capture_validation_v1",
        "status": "PASS",
        "source_url": source_url,
        "accepted_rows": win_rows,
        "accepted_row_count": 2,
        "accepted_place_rows": place_rows,
        "accepted_place_row_count": 2,
        "rejected_rows": [],
        "rejected_place_rows": [],
        "expected_runner_count": 2,
        "active_expected_runner_count": 2,
        "scratched_expected_runner_count": 0,
        "scratched_expected_runners": [],
        "scratched_expected_runners_with_odds": [],
        "missing_expected_runners": [],
        "extra_unexpected_runners": [],
        "place_missing_expected_runners": [],
        "place_extra_unexpected_runners": [],
        "failure_root_cause": None,
        "reasons": [],
    }
    market_reports = {
        market: {
            "status": "SUCCESS",
            "race_id": race_id,
            "source_url": source_url,
            "capture_mode": "autonomous_prejump_t60m",
            "capture_timestamp": append_time,
            "market_type": market,
            "inserted_rows": 2,
            "skipped_rows": 0,
            "warnings": [],
            "append_only": True,
        }
        for market in ("win", "place")
    }
    attempt = {
        "schema_version": "autonomous_live_odds_capture_attempt_v1",
        "race_id": race_id,
        "status": "APPENDED",
        "plan_status": "READY_TO_CAPTURE",
        "capture_window_minutes": 60,
        "fetch_time": fetch_time,
        "append_time": append_time,
        "inserted_rows": 4,
        "reasons": [],
        "fetch_result": {
            "success": True,
            "write_performed": False,
            "warnings": [],
            "alias_race_id": race_id,
            "race_id": "SAN_2026-07-18_7",
            "opt_in_source": "explicit argument allow_auto_scrape_odds",
            "discovery_method": "sportsbet_meeting_exact_race",
            "win_count": 2,
            "place_count": 2,
        },
        "validation": validation,
        "append_report": {
            "status": "SUCCESS",
            "race_id": race_id,
            "source_url": source_url,
            "capture_mode": "autonomous_prejump_t60m",
            "capture_timestamp": append_time,
            "market_types": ["win", "place"],
            "inserted_rows": 4,
            "win_inserted_rows": 2,
            "place_inserted_rows": 2,
            "warnings": [],
            "append_only": True,
            "market_reports": market_reports,
        },
    }
    report = {
        "schema_version": "autonomous_live_odds_capture_report_v1",
        "generated_at": generated_at,
        "run_id": capture_dir.name.removeprefix("autonomous_live_odds_capture_"),
        "output_dir": str(capture_dir),
        "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
        "execute": True,
        "allow_auto_scrape_odds": True,
        "attempts": [attempt],
    }
    plan_path = capture_dir / "autonomous_live_odds_capture_plan.json"
    report_path = capture_dir / "autonomous_live_odds_capture_report.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    capture_dir.joinpath("final_status.txt").write_text(
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED\n", encoding="utf-8"
    )

    db_path = tmp_path / "db.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE live_odds ("
            "race_id TEXT, box_number INTEGER, dog_name TEXT, dog_clean_name TEXT, "
            "odds_decimal REAL, source_url TEXT, capture_timestamp TEXT, "
            "market_type TEXT, source TEXT, odds_level TEXT, "
            "sportsbet_box_source TEXT, capture_mode TEXT)"
        )
        for market, rows in (("win", win_rows), ("place", place_rows)):
            for row in rows:
                conn.execute(
                    "INSERT INTO live_odds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        race_id,
                        row["box_number"],
                        row["dog_name"],
                        row["dog_name"].upper(),
                        row["odds_decimal"],
                        source_url,
                        append_time,
                        market,
                        "Sportsbet",
                        "dog",
                        row["sportsbet_box_source"],
                        "autonomous_prejump_t60m",
                    ),
                )
    return {
        "root": root,
        "db_path": db_path,
        "plan_path": plan_path,
        "report_path": report_path,
        "form_path": form_path,
        "sidecar_path": sidecar_path,
        "race_id": race_id,
        "jump": datetime.fromisoformat("2026-07-18T13:00:00+10:00"),
        "current_time": datetime.fromisoformat("2026-07-18T12:01:00+10:00"),
    }


def test_finalized_handoff_binds_report_plan_and_exact_db_rows(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    receipt = discover_capture_handoff(
        evidence_roots=[fixture["root"]],
        db_path=fixture["db_path"],
        race_id=fixture["race_id"],
        jump_datetime=fixture["jump"],
        capture_window_minutes=60,
        current_time=fixture["current_time"],
    )

    assert receipt is not None
    assert receipt["capture_window_minutes"] == 60
    assert receipt["db_row_count"] == 4
    assert receipt["source_report_sha256"] == hashlib.sha256(
        fixture["report_path"].read_bytes()
    ).hexdigest()
    assert receipt["source_plan_sha256"] == hashlib.sha256(
        fixture["plan_path"].read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("market", ["win", "place"])
def test_handoff_rejects_report_odds_that_do_not_match_db(tmp_path, market):
    fixture = _handoff_fixture(tmp_path)
    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    key = "accepted_rows" if market == "win" else "accepted_place_rows"
    report["attempts"][0]["validation"][key][0]["odds_decimal"] += 0.25
    fixture["report_path"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="db_rows_mismatch"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_rejects_changed_or_missing_db_rows(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    with sqlite3.connect(fixture["db_path"]) as conn:
        conn.execute(
            "UPDATE live_odds SET odds_decimal = odds_decimal + 0.5 "
            "WHERE market_type = 'place' AND box_number = 1"
        )

    with pytest.raises(CaptureHandoffError, match="db_rows_mismatch"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_compares_odds_without_lossy_decimal_rounding(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    with sqlite3.connect(fixture["db_path"]) as conn:
        conn.execute(
            "UPDATE live_odds SET odds_decimal = ? "
            "WHERE market_type = 'win' AND box_number = 1",
            (2.5000000000004,),
        )

    with pytest.raises(CaptureHandoffError, match="db_rows_mismatch"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )

def test_handoff_rejects_append_report_integrity_mismatch(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    report["attempts"][0]["append_report"]["place_inserted_rows"] = 1
    fixture["report_path"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="append_report_count_or_integrity"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


@pytest.mark.parametrize(
    "wrong_url",
    [
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "the-meadows/race-7-123456"
        ),
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "sand/race-7-123456"
        ),
        (
            "https://www.sportsbet.com.au/greyhound-racing/results/"
            "sandown/race-7-123456"
        ),
    ],
)
def test_handoff_rejects_wrong_or_post_race_source_url(tmp_path, wrong_url):
    fixture = _handoff_fixture(tmp_path)
    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    attempt = report["attempts"][0]
    attempt["validation"]["source_url"] = wrong_url
    attempt["append_report"]["source_url"] = wrong_url
    for market_report in attempt["append_report"]["market_reports"].values():
        market_report["source_url"] = wrong_url
    fixture["report_path"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="capture_source_race_identity_mismatch"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_source_identity_accepts_canonical_code_to_full_venue_mapping():
    assert _fixed_window_source_matches(
        source_url=(
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "wentworth-park/race-2-123456"
        ),
        thedogs_url=(
            "https://www.thedogs.com.au/racing/wentworth-park/"
            "2026-07-18/2/example"
        ),
        venue="WPK",
        race_date="2026-07-18",
        race_number=2,
    )


def test_handoff_rejects_non_fixed_window_and_ignores_skip_only_report(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    with pytest.raises(CaptureHandoffError, match="capture_window_not_fixed"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=15,
            current_time=fixture["current_time"],
        )

    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    report["attempts"][0]["status"] = "SKIPPED_ALREADY_CAPTURED"
    fixture["report_path"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    assert (
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )
        is None
    )


@pytest.mark.parametrize(
    "key",
    [
        "outcome",
        "results",
        "finish_positions",
        "race_result",
        "winner_details",
        "finishPosition",
    ],
)
def test_handoff_rejects_outcome_bearing_report(tmp_path, key):
    fixture = _handoff_fixture(tmp_path)
    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    report[key] = "must-not-be-consumed"
    fixture["report_path"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="capture_report_contains_outcome"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_waits_for_final_marker_before_parsing_report(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    marker = fixture["report_path"].with_name("final_status.txt")
    marker.unlink()
    fixture["report_path"].write_text("{partial", encoding="utf-8")

    assert (
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )
        is None
    )

    marker.write_text("AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED\n", encoding="utf-8")
    with pytest.raises(CaptureHandoffError, match="capture_report_invalid_json"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_rejects_source_symlink_outside_evidence_root(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("box,dog\n1,UNTRUSTED\n", encoding="utf-8")
    fixture["form_path"].unlink()
    fixture["form_path"].symlink_to(outside)

    with pytest.raises(CaptureHandoffError, match="capture_form_outside_evidence_root"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_rejects_runner_identity_drift(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    plan = json.loads(fixture["plan_path"].read_text(encoding="utf-8"))
    plan["races"][0]["expected_runners"][0]["dog_name"] = "DIFFERENT DOG"
    fixture["plan_path"].write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="validation_runner_set_mismatch"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_rejects_duplicate_box_binding(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    plan = json.loads(fixture["plan_path"].read_text(encoding="utf-8"))
    plan["races"][0]["expected_runners"][1]["box_number"] = 1
    fixture["plan_path"].write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="plan_expected_runner_invalid"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("race_id", "MEA_2026-07-18_7"),
        ("opt_in_source", "environment default"),
        ("discovery_method", "unbounded_search"),
    ],
)
def test_handoff_rejects_fetch_provenance_drift(tmp_path, field, value):
    fixture = _handoff_fixture(tmp_path)
    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    report["attempts"][0]["fetch_result"][field] = value
    fixture["report_path"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="capture_fetch_result_mismatch"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.thedogs.com.au/racing/sandown/2026-07-17/7/example",
        "https://www.thedogs.com.au/racing/sandown/2026-07-18/6/example",
        "https://www.thedogs.com.au/racing/sand/2026-07-18/7/example",
        "https://www.thedogs.com.au/racing/results/sandown/2026-07-18/7/example",
    ],
)
def test_handoff_rejects_wrong_thedogs_source_identity(tmp_path, url):
    fixture = _handoff_fixture(tmp_path)
    plan = json.loads(fixture["plan_path"].read_text(encoding="utf-8"))
    plan["races"][0]["thedogs_source_url"] = url
    fixture["plan_path"].write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="capture_source_race_identity_mismatch"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def _copy_capture_candidate(fixture, *, run_id: str) -> tuple[Path, Path]:
    capture_dir = fixture["root"] / f"autonomous_live_odds_capture_{run_id}"
    capture_dir.mkdir()
    plan = json.loads(fixture["plan_path"].read_text(encoding="utf-8"))
    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    report["run_id"] = run_id
    report["output_dir"] = str(capture_dir)
    plan_path = capture_dir / "autonomous_live_odds_capture_plan.json"
    report_path = capture_dir / "autonomous_live_odds_capture_report.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    capture_dir.joinpath("final_status.txt").write_text(
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED\n", encoding="utf-8"
    )
    return plan_path, report_path


def test_handoff_rejects_two_equally_latest_accepted_receipts(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    _copy_capture_candidate(fixture, run_id="20260718T120500000000Z")

    with pytest.raises(CaptureHandoffError, match="accepted_capture_attempt_ambiguous"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_fails_closed_when_newer_target_candidate_is_invalid(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    plan_path, report_path = _copy_capture_candidate(
        fixture, run_id="20260718T120500000000Z"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plan["generated_at"] = "2026-07-18T12:01:00+10:00"
    report["generated_at"] = plan["generated_at"]
    report["outcome"] = "must-not-be-consumed"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="newer_capture_candidate_invalid"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_fails_closed_on_newer_finalized_truncated_target_plan(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    plan_path, report_path = _copy_capture_candidate(
        fixture, run_id="20260718T120500000000Z"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["generated_at"] = "2026-07-18T12:01:00+10:00"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    plan_path.write_text("{truncated", encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="newer_capture_candidate_invalid"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_rejects_post_jump_use_and_out_of_window_timestamp(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    with pytest.raises(CaptureHandoffError, match="race_already_jumped"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["jump"],
        )

    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    attempt = report["attempts"][0]
    attempt["append_time"] = "2026-07-18T12:31:00+10:00"
    attempt["append_report"]["capture_timestamp"] = attempt["append_time"]
    for market_report in attempt["append_report"]["market_reports"].values():
        market_report["capture_timestamp"] = attempt["append_time"]
    fixture["report_path"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="fixed_window"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=datetime.fromisoformat("2026-07-18T12:32:00+10:00"),
        )


def test_handoff_rejects_fetch_started_before_fixed_window(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    plan = json.loads(fixture["plan_path"].read_text(encoding="utf-8"))
    report = json.loads(fixture["report_path"].read_text(encoding="utf-8"))
    plan["generated_at"] = "2026-07-18T11:56:58+10:00"
    report["generated_at"] = plan["generated_at"]
    report["attempts"][0]["fetch_time"] = "2026-07-18T11:56:59+10:00"
    fixture["plan_path"].write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    fixture["report_path"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(CaptureHandoffError, match="fixed_window"):
        discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )


def test_handoff_reads_database_query_only_without_side_files(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    fixture["db_path"].chmod(0o444)
    try:
        receipt = discover_capture_handoff(
            evidence_roots=[fixture["root"]],
            db_path=fixture["db_path"],
            race_id=fixture["race_id"],
            jump_datetime=fixture["jump"],
            capture_window_minutes=60,
            current_time=fixture["current_time"],
        )
    finally:
        fixture["db_path"].chmod(0o644)

    assert receipt is not None
    assert not Path(str(fixture["db_path"]) + "-journal").exists()
    assert not Path(str(fixture["db_path"]) + "-wal").exists()


def test_lock_wait_polling_yields_to_handoff_after_busy_writer():
    receipt = {"race_id": "Race 7 - SAN - 2026-07-18"}
    calls = iter([None, receipt])
    clock = iter([0.0, 0.0, 0.0, 0.5])
    slept = []

    lock, selected, waited, details = wait_for_lock_or_handoff(
        acquire=lambda: (_ for _ in ()).throw(Busy()),
        handoff=lambda _elapsed: next(calls),
        busy_type=Busy,
        max_wait_seconds=1.0,
        poll_seconds=0.5,
        monotonic=lambda: next(clock),
        sleeper=slept.append,
    )

    assert lock is None
    assert selected is receipt
    assert waited == 0.5
    assert details == {"reason": "active_lock_present"}
    assert slept == [0.5]


def test_busy_writer_without_handoff_keeps_bounded_wait_status(tmp_path):
    dependencies = _execution_dependencies(tmp_path)
    dependencies.pop("released")
    dependencies["capture_handoff_fn"] = lambda **_kwargs: None
    dependencies["acquire_fn"] = lambda **_kwargs: (_ for _ in ()).throw(Busy())
    dependencies["release_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "unacquired lock must not be released"
    )
    dependencies["refresh_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "busy path must not refresh"
    )
    dependencies["capture_execute_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "busy path must not capture"
    )

    output = run_command(
        args(
            tmp_path,
            execute_collection=True,
            allow_auto_scrape_odds=True,
            max_wait_seconds=0.0,
        ),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: NOW,
        **dependencies,
    )

    assert output["status"] == "WAITING_FOR_DAEMON_LOCK"
    assert output["lock_details"] == {"reason": "active_lock_present"}


def test_required_handoff_never_acquires_writer_lock_or_captures(tmp_path):
    dependencies = _execution_dependencies(tmp_path)
    dependencies.pop("released")
    dependencies["capture_handoff_fn"] = lambda **_kwargs: None
    dependencies["acquire_fn"] = lambda **_kwargs: pytest.fail(
        "receipt-only proof must never acquire the writer lock"
    )
    dependencies["release_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "receipt-only proof must never release the writer lock"
    )
    dependencies["refresh_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "receipt-only proof must never refresh"
    )
    dependencies["capture_execute_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "receipt-only proof must never capture"
    )

    output = run_command(
        args(
            tmp_path,
            execute_collection=True,
            allow_auto_scrape_odds=True,
            require_autonomous_handoff=True,
            max_wait_seconds=0.0,
        ),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: NOW,
        **dependencies,
    )

    assert output["status"] == "BLOCKED_ODDS_CAPTURE"
    assert output["reason"] == "exact_autonomous_capture_handoff_missing"
    assert output["capture_window_minutes"] == 60


def test_busy_writer_yields_to_verified_handoff_without_lock_or_capture(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    receipt = discover_capture_handoff(
        evidence_roots=[fixture["root"]],
        db_path=fixture["db_path"],
        race_id=fixture["race_id"],
        jump_datetime=fixture["jump"],
        capture_window_minutes=60,
        current_time=fixture["current_time"],
    )
    assert receipt is not None
    with sqlite3.connect(fixture["db_path"]) as conn:
        original_count = conn.execute("SELECT COUNT(*) FROM live_odds").fetchone()[0]
    fixture["form_path"].write_text("mutated after discovery", encoding="utf-8")
    fixture["report_path"].write_text("mutated after discovery", encoding="utf-8")
    dependencies = _execution_dependencies(tmp_path)
    dependencies.pop("released")
    dependencies["refresh_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "reuse must not refresh"
    )
    dependencies["capture_plan_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "reuse must not plan a new capture"
    )
    dependencies["capture_execute_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "reuse must not capture"
    )
    dependencies["acquire_fn"] = lambda **_kwargs: pytest.fail(
        "reuse must not acquire the writer lock"
    )
    dependencies["release_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "reuse must not release the writer lock"
    )
    dependencies["capture_handoff_fn"] = lambda **_kwargs: receipt
    dependencies["score_fn"] = lambda **_kwargs: {
        "status": "MANUAL_PREJUMP_FROZEN_RESIDUAL_PREDICTION",
        "probability_sums": {"market": 1.0, "half": 1.0, "full": 1.0},
        "predictions": [],
        "persisted": False,
        "outcomes_present": False,
    }

    output = run_command(
        args(
            tmp_path,
            execute_collection=True,
            allow_auto_scrape_odds=True,
            db=fixture["db_path"],
        ),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: fixture["current_time"],
        **dependencies,
    )

    assert output["status"] == "PREDICTION_READY"
    assert output["inserted_live_odds_rows"] == 0
    assert output["capture_reused"] is True
    assert output["capture_handoff"]["db_row_count"] == 4
    assert "_report_bytes" not in output["capture_handoff"]

    repeated = run_command(
        args(
            tmp_path,
            execute_collection=True,
            allow_auto_scrape_odds=True,
            db=fixture["db_path"],
        ),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: fixture["current_time"],
        **dependencies,
    )
    assert canonical_json(repeated) == canonical_json(output)
    with sqlite3.connect(fixture["db_path"]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM live_odds").fetchone()[0] == original_count

    tampered = dict(receipt)
    tampered["_plan_bytes"] = bytes(receipt["_plan_bytes"]) + b" "
    dependencies["capture_handoff_fn"] = lambda **_kwargs: tampered
    blocked = run_command(
        args(
            tmp_path,
            execute_collection=True,
            allow_auto_scrape_odds=True,
            db=fixture["db_path"],
        ),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: fixture["current_time"],
        **dependencies,
    )
    assert blocked["status"] == "BLOCKED_ODDS_CAPTURE"
    assert "handoff_staged_hash_mismatch" in blocked["reason"]


def test_reused_handoff_blocks_if_fixed_window_changes_before_score(tmp_path):
    fixture = _handoff_fixture(tmp_path)
    receipt = discover_capture_handoff(
        evidence_roots=[fixture["root"]],
        db_path=fixture["db_path"],
        race_id=fixture["race_id"],
        jump_datetime=fixture["jump"],
        capture_window_minutes=60,
        current_time=fixture["current_time"],
    )
    assert receipt is not None
    dependencies = _execution_dependencies(tmp_path)
    dependencies.pop("released")
    dependencies["capture_handoff_fn"] = lambda **_kwargs: receipt
    dependencies["acquire_fn"] = lambda **_kwargs: pytest.fail(
        "reuse must not acquire the writer lock"
    )
    dependencies["release_fn"] = lambda *_args, **_kwargs: pytest.fail(
        "reuse must not release the writer lock"
    )
    score_times = iter(
        [
            fixture["current_time"],
            datetime.fromisoformat("2026-07-18T12:31:00+10:00"),
        ]
    )

    output = run_command(
        args(
            tmp_path,
            execute_collection=True,
            allow_auto_scrape_odds=True,
            db=fixture["db_path"],
        ),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: next(score_times),
        **dependencies,
    )

    assert output["status"] == "BLOCKED_MANUAL_PREDICTION"
    assert "capture_window_changed_before_manual_score" in output["reason"]
