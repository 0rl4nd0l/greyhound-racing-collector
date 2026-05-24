import time
from pathlib import Path

import pytest

# Integration tests simulate flows that call the ingestor programmatically

FORM_GUIDE_CSV_CONTENT = """Race Date,Venue,Race Number,Dog Name,Box,Trainer
2025-08-20,SAND,7,Dog Z,5,Trainer Z
"""


@pytest.fixture()
def temp_env_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    upcoming_dir = data_dir / "upcoming_races"
    downloads_dir = tmp_path / "Downloads"
    archive_dir = tmp_path / "archive"
    data_dir.mkdir(parents=True, exist_ok=True)
    upcoming_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("UPCOMING_RACES_DIR", str(upcoming_dir))
    monkeypatch.setenv("ARCHIVE_DIR", str(archive_dir))
    monkeypatch.setenv("DOWNLOADS_WATCH_DIR", str(downloads_dir))

    # Reload config.paths for env changes
    import importlib

    if "config.paths" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("config.paths"))
    yield {
        "DATA_DIR": data_dir,
        "UPCOMING": upcoming_dir,
        "DL": downloads_dir,
        "ARCHIVE": archive_dir,
    }


def _fresh_ingestor_and_watchers():
    import importlib

    if "ingestion.ingest_race_csv" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("ingestion.ingest_race_csv"))
    if "utils.download_watcher" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("utils.download_watcher"))
    if "utils.upcoming_watcher" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("utils.upcoming_watcher"))
    from ingestion.ingest_race_csv import ingest_form_guide_csv
    from utils.download_watcher import start_download_watcher, wait_for_stable
    from utils.upcoming_watcher import start_upcoming_watcher

    return (
        ingest_form_guide_csv,
        start_download_watcher,
        start_upcoming_watcher,
        wait_for_stable,
    )


def test_programmatic_manual_flow_updates_upcoming_and_index(temp_env_dirs):
    ingest_form_guide_csv, _, _, _ = _fresh_ingestor_and_watchers()

    # Simulate a downloader writing a temp file and then finishing
    tmp_src = temp_env_dirs["DL"] / "manual_form.csv"
    tmp_src.write_text(FORM_GUIDE_CSV_CONTENT, encoding="utf-8")

    # Direct call (programmatic manual flow)
    published = ingest_form_guide_csv(str(tmp_src))

    assert published.exists()
    assert published.parent == temp_env_dirs["UPCOMING"]
    assert published.name == "2025-08-20_sand_R7.csv"

    # Predictions index presence: minimal check is that file is in UPCOMING; a broader index test would call API layer
    all_files = {p.name for p in temp_env_dirs["UPCOMING"].glob("*.csv")}
    assert published.name in all_files


try:
    _HAS_WATCHDOG = True
except Exception:
    _HAS_WATCHDOG = False


@pytest.mark.skipif(not _HAS_WATCHDOG, reason="watchdog not installed")
def test_watcher_flow_ingests_from_downloads_and_triggers_callback(
    monkeypatch, temp_env_dirs
):
    # Provide a UI refresh callback spy
    callback_calls = []

    def ui_refresh(paths):
        callback_calls.append([Path(p).name for p in paths])

    (
        ingest_form_guide_csv,
        start_download_watcher,
        start_upcoming_watcher,
        wait_for_stable,
    ) = _fresh_ingestor_and_watchers()

    # Start upcoming watcher to observe UPCOMING for new CSVs
    obs1 = start_upcoming_watcher(
        upcoming_dir=temp_env_dirs["UPCOMING"],
        on_change=ui_refresh,
        debounce_seconds=0.2,
    )

    # Start downloads watcher to auto-ingest
    from utils.download_watcher import start_download_watcher as _sdw

    obs2 = _sdw(
        downloads_dir=temp_env_dirs["DL"]
    )  # use default callback that invokes ingestor

    try:
        # Simulate browser download: write a .tmp then rename to .csv
        partial = temp_env_dirs["DL"] / "Race 7 - SAND - 2025-08-20.csv.part"
        final = temp_env_dirs["DL"] / "Race 7 - SAND - 2025-08-20.csv"
        partial.write_text(FORM_GUIDE_CSV_CONTENT, encoding="utf-8")
        partial.rename(final)

        # Allow some time for watchers to process
        time.sleep(1.0)

        published = temp_env_dirs["UPCOMING"] / "2025-08-20_sand_R7.csv"
        assert published.exists()

        # UI refresh callback should have been called at least once referencing the published file
        joined = ",".join(name for batch in callback_calls for name in batch)
        assert "2025-08-20_sand_R7.csv" in joined
    finally:
        # Stop observers if available
        for obs in (locals().get("obs1"), locals().get("obs2")):
            try:
                if obs is not None:
                    obs.stop()
                    obs.join(timeout=2)
            except Exception:
                pass
