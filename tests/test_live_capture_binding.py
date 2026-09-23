import hashlib
import json
from datetime import datetime, timedelta

import pytest


def reserved_alias_plan(tmp_path):
    from tests.test_autonomous_live_odds_capture import _write_capture_input
    from tests.test_live_freshness_candidate import contract_value
    from scripts import autonomous_live_odds_capture as capture
    from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest
    from race_collection.synchronous_manual_capture import runner_set_sha256

    stamp = datetime.fromisoformat("2026-06-10T14:40:00+10:00")
    csv = _write_capture_input(tmp_path / "input", venue="MURRAY-BRIDGE-STRAIGHT", race_number=9)
    sidecar = capture.sidecar_path_for(csv)
    metadata = json.loads(sidecar.read_bytes())
    metadata["prejump_shadow_metadata"]["source_native_race_id"] = "synthetic-9"
    sidecar.write_text(json.dumps(metadata))
    plan = capture.build_capture_plan([csv.parent], current_time=stamp)
    native = plan["races"][0]
    canonical = "Race 9 - MURR - 2026-06-10"
    accounting = {
        "schema_version": "freshness_attempt_reconciliation_v1",
        "complete": True,
        "consumed": [],
        "sources": [{"sha256": "a" * 64}],
    }
    scope = FreshnessContract(
        {
            **contract_value(tmp_path),
            "starts_at": stamp.isoformat(),
            "ends_at": (stamp + timedelta(minutes=90)).isoformat(),
            "source_date": "2026-06-10",
            "reconciliation_sha256": digest(accounting),
        }
    )
    allowance = AttemptAllowance(scope)
    allowance.initialize(accounting)
    inputs = {
        "capture_runner_set_sha256": runner_set_sha256(native["expected_runners"]),
        "race_id": canonical,
        "race_id_aliases": [canonical, native["race_id"]],
        "capture_window_minutes": native["capture_window_minutes"],
        "input_files": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (csv, capture.sidecar_path_for(csv))
        },
        "race_identity": {
            "race_id": canonical,
            "race_url": native["thedogs_source_url"],
            "jump_datetime": native["jump_datetime"],
            "source_native_race_id": "synthetic-9",
            "runner_set_sha256": runner_set_sha256(native["expected_runners"]),
        },
    }
    claim = allowance.reserve(inputs, now=stamp)
    return allowance, claim, plan, stamp


def test_reserved_alias_is_explicitly_bound_to_canonical_identity_before_fetch(tmp_path):
    allowance, claim, plan, stamp = reserved_alias_plan(tmp_path)
    bound = allowance.bind_capture_plan(claim, plan)
    row = bound["races"][0]
    assert row["race_id"] == "Race 9 - MURR - 2026-06-10"
    assert row["planner_race_id"] == "Race 9 - MURRAY-BRIDGE-STRAIGHT - 2026-06-10"
    allowance.start_fetch(claim, row, now=stamp)
    assert claim.with_suffix(".fetch.json").exists()
    assert not allowance.available()


@pytest.mark.parametrize("change", ["race", "url", "jump", "runner", "ambiguous", "input"])
def test_different_or_ambiguous_capture_is_rejected_without_fetch(tmp_path, change):
    allowance, claim, plan, stamp = reserved_alias_plan(tmp_path)
    row = plan["races"][0]
    if change == "race":
        row["race_id"] = "Race 9 - ELSEWHERE - 2026-06-10"
    elif change == "url":
        row["thedogs_source_url"] += "/different"
    elif change == "jump":
        row["jump_datetime"] = "2026-06-10T15:01:00+10:00"
    elif change == "runner":
        row["expected_runners"][0]["dog_name"] = "Different Dog"
    elif change == "ambiguous":
        plan["races"].append(dict(row))
    else:
        from pathlib import Path

        Path(row["csv_path"]).write_text("changed")
    before = claim.read_bytes()
    with pytest.raises(ValueError):
        allowance.bind_capture_plan(claim, plan)
    assert not claim.with_suffix(".fetch.json").exists()
    assert claim.read_bytes() == before
    assert not allowance.available()


def test_unexpected_dependency_http_is_recorded_and_blocked_before_network(tmp_path, monkeypatch):
    import requests
    from race_collection.live_freshness_contract import install_request_guard

    allowance, _, _, stamp = reserved_alias_plan(tmp_path)
    import race_collection.live_freshness_contract as contract_module

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return stamp

    monkeypatch.setattr(contract_module, "datetime", Clock)
    restore = install_request_guard(allowance.scope)
    try:
        with pytest.raises(ValueError, match="unexpected_network"):
            requests.get("https://pypi.org/simple/selenium/")
    finally:
        restore()
    counters = json.loads((allowance.scope.session / "network-count.json").read_bytes())
    assert counters["unexpected_blocked"] == 1
    assert counters["provider_started"] == 0


def test_browser_network_observations_do_not_hide_other_hosts(tmp_path):
    from race_collection.live_execution import BrowserNetworkAccounting

    class Driver:
        def get(self, url):
            pass

        def get_log(self, name):
            return [
                {
                    "message": json.dumps(
                        {
                            "message": {
                                "method": "Network.requestWillBeSent",
                                "params": {"requestId": str(n), "request": {"url": url}},
                            }
                        }
                    )
                }
                for n, url in enumerate(
                    ("https://www.sportsbet.com.au/race-9", "https://unexpected.example/asset")
                )
            ]

    driver = Driver()
    meter = BrowserNetworkAccounting(driver, tmp_path / "browser.json")
    driver.get("https://www.sportsbet.com.au/race-9")
    meter.drain()
    assert meter.value["browser_navigation_attempts"] == 1
    assert meter.value["observed_provider_requests"] == 1
    assert meter.value["observed_other_requests"] == 1
    assert "unexpected.example" in meter.value["observed_by_host"]
    with pytest.raises(ValueError, match="unexpected_browser_navigation"):
        driver.get("https://pypi.org/")
