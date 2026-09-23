"""Exercise the exported entrypoint; only systemd, time and input delivery are simulated."""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from datetime import datetime

import pytest


@pytest.mark.parametrize("scenario", ["empty", "publication", "unavailable", "stale", "malformed"])
def test_packaged_supervisor_reader_monitor_and_restoration(tmp_path, scenario, monkeypatch):
    from scripts.prepare_freshness_rehearsal import prepare, UNITS
    from race_collection.freshness_attempt_reconciliation import DOMAINS
    from utils.sportsbet_access import SportsbetAccess

    access = tmp_path / "sportsbet-access.json"
    SportsbetAccess(access).initialize(access_basis={"status": "permitted", "reference": "fabricated test"})
    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(access))

    installed = tmp_path / "installed"
    installed.mkdir()
    for name in (*UNITS, "greyhound-operator-ui-r3.service"):
        (installed / name).write_text("synthetic original " + name)
    operational = tmp_path / "operational"
    for name in ("requests", "claims", "attempts"):
        (operational / name).mkdir(parents=True)
    db = tmp_path / "operational.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE live_odds (race_id TEXT, capture_mode TEXT)")
    output = tmp_path / "package"
    prepared = prepare(
        output=output,
        start=datetime.fromisoformat("2026-07-19T12:00:00+10:00"),
        python=Path(sys.executable),
        db=db,
        lock=tmp_path / "collector.lock",
        reconciliation_roots={key: [str(operational)] for key in DOMAINS if key != "live_odds"},
        installed_dir=installed,
    )
    harness = Path(__file__).with_name("fixtures") / "freshness_packaged_harness.py"
    completed = subprocess.run(
        [sys.executable, "-B", str(harness), str(output), scenario, prepared["plan_sha256"]],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "fixture-result.json").read_text())
    assert result["restored"] and result["actual_packaged_entrypoint"]
    assert result["samples"] >= 1
