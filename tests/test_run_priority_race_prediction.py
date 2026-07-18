from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pytest

from upcoming_race_browser import UpcomingRaceBrowser
from scripts.refresh_prejump_upcoming import select_prejump_races
from scripts.run_priority_race_prediction import (
    FIXED_CAPTURE_WINDOWS_MINUTES,
    acquire_with_bounded_wait,
    canonical_json,
    resolve_target_race,
    run_command,
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
        "days_ahead": 1,
        "max_wait_seconds": 0.0,
        "poll_seconds": 0.1,
        "fetch_timeout_seconds": 1.0,
        "db": tmp_path / "db.sqlite",
        "model_dir": tmp_path / "model",
        "lock_path": tmp_path / "runtime.lock",
        "lock_output_dir": tmp_path,
        "lock_stale_seconds": 60,
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
    feature_time = NOW.replace(minute=1)
    score_time = NOW.replace(minute=2)
    seen = {}
    original_seal = dependencies["feature_seal_fn"]
    original_score = dependencies["score_fn"]

    def seal(**kwargs):
        seen["feature_time"] = kwargs["current_time"]
        return original_seal(**kwargs)

    def score(**kwargs):
        seen["score_time"] = kwargs["score_timestamp"]
        kwargs["score_timestamp"] = NOW
        return original_score(**kwargs)

    dependencies["feature_seal_fn"] = seal
    dependencies["score_fn"] = score
    times = iter([feature_time, score_time])
    output = run_command(
        args(tmp_path, execute_collection=True, allow_auto_scrape_odds=True),
        races=[race()],
        current_time=NOW,
        now_provider=lambda: next(times),
        **dependencies,
    )
    assert output["status"] == "PREDICTION_READY"
    assert seen == {"feature_time": feature_time, "score_time": score_time}


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
