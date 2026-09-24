import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from types import SimpleNamespace

import pytest

from scripts import refresh_prejump_upcoming as refresh


@pytest.fixture(autouse=True)
def isolated_metadata_snapshot(tmp_path, monkeypatch):
    """Timing-only fixtures must never acquire provider metadata."""
    from utils import prejump_sportsbet

    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(tmp_path / "never-live-access.json"))
    monkeypatch.setattr(prejump_sportsbet, "fetch_sportsbet_next_events_snapshot",
                        lambda **kwargs: {"events": []})


@pytest.mark.parametrize(
    "observed,wall,max_minutes,requested,expected",
    [
        ("2026-09-22T14:22:33+10:00", "2026-09-22T14:22:34+10:00", 160, 1, 0),
        ("2026-09-22T23:00:00+10:00", "2026-09-22T23:00:01+10:00", 160, 1, 1),
        ("2026-09-22T23:00:00+10:00", "2026-09-22T23:00:01+10:00", 60, 1, 1),
        ("2026-09-22T23:00:00+10:00", "2026-09-22T23:00:01+10:00", 160, 0, 0),
        ("2026-09-22T14:00:00+10:00", "2026-09-23T14:00:00+10:00", 160, 1, 1),
        ("2026-09-22T14:00:00+10:00", "2026-09-22T04:00:00+00:00", 160, 1, 1),
        ("2026-09-22T14:00:00+10:00", "2026-09-22T14:00:01+10:00", 3000, 1, 1),
    ],
)
def test_budgeted_discovery_preserves_every_date_in_window(
    observed, wall, max_minutes, requested, expected
):
    from datetime import datetime

    assert (
        refresh.bounded_discovery_days_ahead(
            now=datetime.fromisoformat(observed),
            wall_now=datetime.fromisoformat(wall),
            max_minutes=max_minutes,
            requested_days_ahead=requested,
        )
        == expected
    )


def install_fixture_browser(path):
    import importlib.util
    import sys

    specification = importlib.util.spec_from_file_location("upcoming_race_browser", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    sys.modules["upcoming_race_browser"] = module


@pytest.mark.parametrize("budgeted,legacy_discovery", [(False, False), (True, False), (True, True)])
def test_refresh_avoids_out_of_window_day_before_download_admission(
    tmp_path, monkeypatch, budgeted, legacy_discovery
):
    from datetime import datetime
    import upcoming_race_browser as browser_module

    observed = datetime.now().astimezone().replace(hour=14, minute=0, second=0, microsecond=0)
    elapsed = [0.0]
    requested_dates = []
    downloaded = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed

    def discover(browser, date):
        requested_dates.append(date)
        elapsed[0] += 38
        return [
            {
                "date": date.isoformat(),
                "race_time": f"14:{30 + number:02d}",
                "race_number": number + 1,
                "venue": "HOR",
                "url": f"https://www.thedogs.com.au/racing/horsham/{date}/{number + 1}/fixture",
            }
            for number in range(6)
        ]

    def download(browser, url, **kwargs):
        downloaded.append(url)
        elapsed[0] += 1
        return {"success": True}

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, worker, tasks):
            return map(worker, tasks)

    monkeypatch.setenv("UPCOMING_RACES_DIR", str(tmp_path))
    monkeypatch.setattr(refresh, "datetime", Clock)
    monkeypatch.setattr(browser_module, "datetime", Clock)
    monkeypatch.setattr(browser_module, "time", SimpleNamespace(sleep=lambda seconds: None))
    monkeypatch.setattr(refresh, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    monkeypatch.setattr(browser_module.UpcomingRaceBrowser, "get_races_for_date", discover)
    monkeypatch.setattr(browser_module.UpcomingRaceBrowser, "download_race_csv", download)
    monkeypatch.setattr(refresh, "ProcessPoolExecutor", Pool)
    if legacy_discovery:
        monkeypatch.setattr(
            refresh,
            "bounded_discovery_days_ahead",
            lambda **kwargs: kwargs["requested_days_ahead"],
        )
    args = refresh.build_parser().parse_args(
        [
            "--upcoming-dir",
            str(tmp_path),
            "--days-ahead",
            "1",
            "--limit",
            "4",
            "--workers",
            "2",
            "--current-time",
            observed.isoformat(),
            "--require-safe-metadata",
            *(["--refresh-budget-seconds", "65"] if budgeted else []),
        ]
    )
    report = refresh.refresh_prejump_upcoming(args)
    assert report["generated_at"] == observed.isoformat()
    assert report["days_ahead"] == 1
    bounded = budgeted and not legacy_discovery
    assert report["discovery_days_ahead"] == (0 if bounded else 1)
    assert len(requested_dates) == (1 if bounded else 2)
    assert report["selected_count"] == 4
    assert len(downloaded) == (0 if legacy_discovery else 4)
    assert {item["date"] for item in report["selected_races"]} == {observed.date().isoformat()}
    decisions = report["considered_races"]
    assert len(decisions) == len(requested_dates) * 6
    assert sum(row["selection_decision"] == "selected_for_download" for row in decisions) == 4
    assert sum(row["selection_decision"] == "download_limit" for row in decisions) == 2
    assert all(row["selection_decision"] == "future_outside_preferred_window"
               for row in decisions if row["date"] != observed.date().isoformat())
    assert report["metadata_collection_status"] != "PASS"
    assert report["current_index_race_count"] == 0
    assert report["status"] != "SUCCESS"
    if legacy_discovery:
        assert report["status"] == "REFRESH_BUDGET_EXCEEDED"
        assert report["refresh_elapsed_seconds"] == 76
        assert all(item["reason"] == "refresh_budget_exhausted" for item in report["downloads"])
    elif budgeted:
        assert report["refresh_elapsed_seconds"] == 42
        assert report["refresh_phase_seconds"] == {
            "discovery_and_browser_startup": 38,
            "selection_and_downloads": 4,
            "metadata_validation": 0,
        }


def test_worker_report_fallback_is_limited_to_its_exact_race(tmp_path):
    import json
    from scripts.autonomous_live_odds_capture import refresh_report_for_input_dir

    source_dir = tmp_path / "refreshed"
    workers = [source_dir / "workers" / str(number) for number in range(2)]
    for directory in workers:
        directory.mkdir(parents=True)
    report = {
        "generated_at": "2026-07-19T12:00:00+10:00",
        "selected_races": [{"race_url": f"race-{number}"} for number in range(2)],
        "downloads": [
            {
                "race_url": f"race-{number}",
                "result": {"raw_export_path": str(directory / "raw_exports/race.csv")},
            }
            for number, directory in enumerate(workers)
        ],
    }
    (tmp_path / "refresh_prejump_report.json").write_text(json.dumps(report))
    selected = refresh_report_for_input_dir(workers[1])
    assert selected["generated_at"] == report["generated_at"]
    assert selected["selected_races"] == [{"race_url": "race-1"}]
    assert len(selected["downloads"]) == 1
    assert selected["downloads"][0]["race_url"] == "race-1"


def test_expired_tasks_do_not_import_browser_or_request_source(tmp_path):
    tasks = [
        {
            "race_url": f"fixture-{number}",
            "deadline": 0,
            "directory": str(tmp_path / str(number)),
            "hint": {},
        }
        for number in range(4)
    ]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
        results = list(executor.map(refresh.download_selected_race, tasks))
    assert [result["race_url"] for result in results] == [task["race_url"] for task in tasks]
    assert all(result["reason"] == "refresh_budget_exhausted" for result in results)
    assert list(tmp_path.iterdir()) == []


def test_successful_workers_have_distinct_processes_and_directories(tmp_path, monkeypatch):
    import time

    module = tmp_path / "upcoming_race_browser.py"
    module.write_text(
        "import os,time\nfrom pathlib import Path\n"
        "class UpcomingRaceBrowser:\n"
        " def download_race_csv(self, url, **kwargs):\n"
        "  directory=Path(os.environ['UPCOMING_RACES_DIR'])\n"
        "  directory.mkdir(parents=True)\n"
        "  (directory.parent / str(os.getpid())).touch()\n"
        "  deadline=time.monotonic()+10\n"
        "  while len(list(directory.parent.glob('[0-9]*'))) < 2:\n"
        "   if time.monotonic()>deadline: raise RuntimeError('second worker did not start')\n"
        "   time.sleep(0.01)\n"
        "  (directory / 'fixture.csv').write_text(url)\n"
        "  return {'success':True,'pid':os.getpid(),'directory':str(directory)}\n"
    )
    tasks = [
        {
            "race_url": f"fixture-{number}",
            "deadline": time.monotonic() + 30,
            "directory": str(tmp_path / "outputs" / f"race-{number}"),
            "hint": {},
        }
        for number in range(2)
    ]
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=get_context("spawn"),
        initializer=install_fixture_browser,
        initargs=(str(module),),
    ) as executor:
        results = list(executor.map(refresh.download_selected_race, tasks))
    assert all(result["success"] for result in results)
    assert len({result["result"]["pid"] for result in results}) == 2
    assert len({result["result"]["directory"] for result in results}) == 2
    assert [
        (tmp_path / "outputs" / f"race-{number}" / "fixture.csv").read_text() for number in range(2)
    ] == ["fixture-0", "fixture-1"]


@pytest.mark.parametrize("workers", [1, 2])
def test_refresh_accounts_for_every_download_and_does_not_redate_overrun(
    tmp_path, monkeypatch, workers
):
    import sys

    observed = "2026-07-19T12:00:00+10:00"
    clock = iter([0, 20, *([20] if workers == 2 else []), 90, 101])
    monkeypatch.setattr(refresh, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setenv("UPCOMING_RACES_DIR", str(tmp_path))
    selected = [{"url": f"fixture-{number}"} for number in range(3)]
    records = [
        {"race_url": row["url"], "selected": True, "bucket": "preferred"} for row in selected
    ]
    downloads = []

    class Browser:
        session = None

        def get_upcoming_races(self, **kwargs):
            return selected

        def download_race_csv(self, url, **kwargs):
            downloads.append((url, os.environ["UPCOMING_RACES_DIR"]))
            return {"success": True}

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, worker, tasks):
            for task in tasks:
                downloads.append((task["race_url"], task["directory"]))
                yield {"race_url": task["race_url"], "success": True}

    monkeypatch.setitem(
        sys.modules, "upcoming_race_browser", SimpleNamespace(UpcomingRaceBrowser=Browser)
    )
    monkeypatch.setattr(refresh, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        refresh, "select_prejump_races", lambda *args, **kwargs: (selected, records)
    )
    monkeypatch.setattr(refresh, "selected_prejump_records", lambda *args, **kwargs: records)
    monkeypatch.setattr(refresh, "sidecar_metadata_coverage", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(
        refresh,
        "current_index_metadata_selection",
        lambda *args, **kwargs: ([], {"status": "PASS"}),
    )
    monkeypatch.setattr(refresh, "refresh_timing_summary", lambda *args, **kwargs: {})
    args = refresh.build_parser().parse_args(
        [
            "--upcoming-dir",
            str(tmp_path),
            "--workers",
            str(workers),
            "--refresh-budget-seconds",
            "100",
            "--current-time",
            observed,
        ]
    )
    report = refresh.refresh_prejump_upcoming(args)
    assert len(report["downloads"]) == 3
    assert [item[0] for item in downloads] == [row["url"] for row in selected]
    if workers == 2:
        assert len({item[1] for item in downloads}) == 3
    assert report["status"] == "REFRESH_BUDGET_EXCEEDED"
    assert report["generated_at"] == observed
    assert report["refresh_elapsed_seconds"] == 101
