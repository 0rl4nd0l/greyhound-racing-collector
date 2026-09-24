"""Generated bounded services stay inside their sealed source date after 21:20."""
import shlex
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scripts import shadow_autopilot_daemon as daemon
from scripts.refresh_prejump_upcoming import bounded_discovery_days_ahead


@pytest.mark.parametrize("renderer", [daemon.service_file_text, daemon.odds_capture_service_file_text])
@pytest.mark.parametrize("profile", [None, "bounded80-v1"])
def test_generated_discovery_respects_single_date_contract(tmp_path, renderer, profile):
    options = dict(repo_path=tmp_path, timeout_seconds=600)
    if profile:
        options.update(live_freshness=True, live_freshness_profile=profile,
                       live_freshness_contract=tmp_path / "contract.json")
    service = renderer(**options)
    command = shlex.split(next(line.removeprefix("ExecStart=") for line in service.splitlines()
                              if line.startswith("ExecStart=")))
    days = int(command[command.index("--days-ahead") + 1])
    assert days == (0 if profile else 1)
    now = datetime(2026, 9, 24, 21, 30, tzinfo=ZoneInfo("Australia/Melbourne"))
    assert bounded_discovery_days_ahead(now=now, wall_now=now, max_minutes=160,
                                       requested_days_ahead=days) == days
