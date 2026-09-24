"""Actual CLI/discovery/download/spawn/policy, using only invented HTTP payloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("REFRESH_TEST_SOURCE_ROOT", ROOT))
TRANSPORT = ROOT / "tests/fixtures/shared_snapshot_transport"
PYTHON = os.environ.get("FRESHNESS_TEST_PYTHON", sys.executable)


def fixture(root, count, *, mismatch=False):
    now = datetime.now(ZoneInfo("Australia/Melbourne"))
    jump = (now + timedelta(minutes=40)).replace(second=0, microsecond=0)
    date = now.date().isoformat()
    names = ("Synthetic Alpha", "Synthetic Bravo", "Synthetic Charlie", "Synthetic Delta")
    responses, links, events = {}, [], []
    for i in range(count):
        venue, competition = ("sale", "Sale") if i < 8 else ("sandown", "Sandown")
        number = i % 8 + 1
        url = f"https://www.thedogs.com.au/racing/{venue}/{date}/{number}/invented"
        path = url.split("www.thedogs.com.au", 1)[1]
        links.append(f'<a href="{path}">Race {number} Grade 5 400m</a>')
        race_id = 99000 + i
        runners = "".join(f'''<tr class="race-runner"><td class="race-runners__box"><sprite-svg name="rug_{box}"></sprite-svg></td><td class="race-runners__name"><div class="race-runners__name__dog">{name}<span class="race-runners__name__time">24.00</span></div></td><td class="race-runners__odds"><a href="/dogs/runner/{race_id}{box}"><runner-odd data-runner-id="{race_id}{box}"></runner-odd></a></td></tr>''' for box, name in enumerate(names, 1))
        page = f'''<html><title>Race {number}</title><body><div class="race-header" data-race-id="{race_id}"><span class="race-box__number">R{number}</span><div class="race-header__info__grade">Grade 5 400m</div></div><formatted-time data-format="time_24">{jump:%H:%M}</formatted-time><section><dl><dt>Weather</dt><dd>Fine</dd></dl></section><table class="race-runners"><tbody>{runners}</tbody></table></body></html>'''
        expert = f'<a href="{url}/expert-form/export.csv">Download CSV</a>' + "".join(f'<div class="layout--sidebar--expert"><div class="expert-form-runner__details__dog__name">{name}<span>(5)</span></div></div>' for name in names)
        csv = "Dog Name,Sex,PLC,BOX,WGT,DIST,DATE,TRACK,G,TIME,WIN,BON,1 SEC,MGN,W/2G,PIR,SP\n" + "".join(f"{box}. {name},D,1,{box},30,400,2026-01-01,TEST,5,22.10,2.00,22.00,5.00,1.00,111,1,$2.00\n" for box, name in enumerate(names, 1))
        for suffix, body in (("", page), ("/expert-form", expert), ("/expert-form/export.csv", csv)):
            responses["www.thedogs.com.au" + path + suffix] = {"body": body, "content_type": "text/csv" if suffix.endswith("csv") else "text/html"}
        events.append(dict(id=80000+i, classId="4", competitionName=competition,
                           raceNumber=number, startTime=int(jump.timestamp()), trackStatus="Good", distance="400"))
    if mismatch:
        events[0]["raceNumber"] = 99
    responses["api.open-meteo.com/v1/forecast"] = {"body": json.dumps({"hourly": {"time": [jump.strftime("%Y-%m-%dT%H:%M")], "weather_code": [0]}}), "content_type": "application/json"}
    responses["www.thedogs.com.au/racing/" + date] = {"body": "<html>" + "".join(links) + "</html>"}
    from utils.prejump_sportsbet import SPORTSBET_NEXT_EVENTS_ENDPOINT
    from urllib.parse import urlsplit
    endpoint = urlsplit(SPORTSBET_NEXT_EVENTS_ENDPOINT)
    responses[endpoint.netloc + endpoint.path] = {"body": json.dumps(events), "content_type": "application/json"}
    path = root / "fixture.json"
    path.write_text(json.dumps(dict(responses=responses, log=str(root / "transport.jsonl"))))
    return path


def access(root, *, previous_operations=0):
    from utils.sportsbet_access import SportsbetAccess
    path = root / "access.json"
    gate = SportsbetAccess(path)
    gate.initialize(access_basis={"status": "permitted", "reference": "invented offline workload"})
    with gate.locked():
        state = gate.read()
        state["operating_policy"] = dict(reference="bounded 10/1/2 policy fixture", python_per_60_seconds=10, browser_per_60_seconds=1, browser_navigation_cap=2)
        # Synthetic preceding operations represent persisted history, not resetting a ledger.
        state["operations"] = [{"at": time.time()-120, "kind": "browser" if i % 3 == 0 else "python"} for i in range(previous_operations)]
        gate.write(state)
    return path


def refresh(root, fixture_path, access_path, count, *, lane, workers=2):
    output = root / (lane + ".json")
    env = dict(os.environ, PYTHONPATH=str(TRANSPORT) + os.pathsep + str(SOURCE), PYTHONDONTWRITEBYTECODE="1", GREYHOUND_SHARED_SNAPSHOT_FIXTURE=str(fixture_path), GREYHOUND_SPORTSBET_ACCESS_STATE=str(access_path), LIVE_ENHANCE_LIMIT="0")
    command = [PYTHON, str(SOURCE / "scripts/refresh_prejump_upcoming.py"), "--upcoming-dir", str(root / lane), "--workers", str(workers), "--limit", str(count), "--days-ahead", "0", "--min-minutes", "20", "--max-minutes", "160", "--require-safe-metadata", "--refresh-budget-seconds", "80", "--output", str(output)]
    launcher = "from scripts.check_freshness_service import deny_network; import os,sys; deny_network(); os.execv(sys.argv[1],sys.argv[1:])"
    result = subprocess.run([PYTHON, "-c", launcher, *command], cwd=root, env=env, text=True, capture_output=True, timeout=60)
    (root / (lane + ".stdout")).write_text(result.stdout + result.stderr)
    assert output.exists(), result.stdout + result.stderr
    return json.loads(output.read_text())


@pytest.mark.parametrize("odds_count", [9, 16])
def test_actual_spawned_two_lane_refresh_shares_metadata_snapshot(tmp_path, odds_count):
    payload = fixture(tmp_path, odds_count)
    gate = access(tmp_path)
    full = refresh(tmp_path, payload, gate, 6, lane="full")
    from utils.sportsbet_access import SportsbetAccess
    # The separate browser lane owns the same durable policy between refreshes.
    with SportsbetAccess(gate).operation("browser") as operation:
        operation.response(200, {})
        operation.accept_data()
    odds = refresh(tmp_path, payload, gate, odds_count, lane="odds")
    state = json.loads(gate.read_text())
    assert state["phase"] == "OPEN", state
    assert [row["kind"] for row in state["operations"]] == ["python", "browser", "python"], state
    assert full["selected_count"] == 6 and odds["selected_count"] == odds_count
    for report, lane in ((full, "full"), (odds, "odds")):
        assert report["current_index_race_count"] == report["selected_count"], report
        assert all(row["success"] for row in report["downloads"]), report
        sidecars = [json.loads(path.read_text()) for path in (tmp_path / lane / "workers").glob("*/*.csv.metadata.json")]
        assert len(sidecars) == report["selected_count"]
        details = [row["weather_track_metadata_detail"]["sportsbet_pre_race_page"] for row in sidecars]
        assert len({row["snapshot_payload_sha256"] for row in details}) == 1
        assert len({row["snapshot_observed_at"] for row in details}) == 1
    lines = [json.loads(line) for line in (tmp_path / "transport.jsonl").read_text().splitlines()]
    downloads = [row for row in lines if row["path"].endswith("export.csv")]
    assert len({row["pid"] for row in downloads}) >= 4, "both lanes must use actual two-process workers"


def test_two_lanes_share_persisted_512_ceiling_and_restart_hold(tmp_path):
    payload = fixture(tmp_path, 9)
    gate = access(tmp_path, previous_operations=511)
    first = refresh(tmp_path, payload, gate, 6, lane="full")
    assert first["current_index_race_count"] == 6
    second = refresh(tmp_path, payload, gate, 9, lane="odds")
    state = json.loads(gate.read_text())
    assert state["phase"] == "STOP" and len(state["operations"]) == 512
    assert second["current_index_race_count"] == 0
    third = refresh(tmp_path, payload, gate, 9, lane="restart")
    assert third["current_index_race_count"] == 0
    assert len(json.loads(gate.read_text())["operations"]) == 512


def test_shared_snapshot_preserves_exact_per_race_rejection(tmp_path):
    payload = fixture(tmp_path, 9, mismatch=True)
    gate = access(tmp_path)
    report = refresh(tmp_path, payload, gate, 9, lane="odds")
    assert report["selected_count"] == 9
    assert report["current_index_race_count"] == 8, report
    assert report["current_index_metadata_selection"]["excluded_race_count"] == 1
    assert len(json.loads(gate.read_text())["operations"]) == 1


@pytest.mark.parametrize("failure", ["unexpected_payload", "source_denial"])
def test_failed_shared_observation_is_not_retried_per_race(tmp_path, failure):
    payload = fixture(tmp_path, 16)
    data = json.loads(payload.read_text())
    key = next(key for key in data["responses"] if "NextEvents" in key)
    data["responses"][key] = {"body": "{}", "status": 429 if failure == "source_denial" else 200,
                              "content_type": "application/json"}
    payload.write_text(json.dumps(data))
    gate = access(tmp_path)
    report = refresh(tmp_path, payload, gate, 16, lane="odds")
    assert report["selected_count"] == 16
    assert report["current_index_race_count"] == 0
    state = json.loads(gate.read_text())
    assert len(state["operations"]) == 1
    assert state["phase"] == ("COOLDOWN" if failure == "source_denial" else "OPEN")
    requests = [json.loads(line) for line in (tmp_path / "transport.jsonl").read_text().splitlines()]
    assert sum("NextEvents" in row["path"] for row in requests) == 1


def test_exact_meeting_link_time_avoids_duplicate_discovery_fetch(tmp_path):
    payload = fixture(tmp_path, 6)
    data = json.loads(payload.read_text())
    date_key = next(k for k in data['responses'] if k.startswith('www.thedogs.com.au/racing/') and k.count('/') == 2)
    race_page = next(v['body'] for k, v in data['responses'].items() if k.endswith('/invented'))
    import re
    clock = re.search(r'<formatted-time.*?</formatted-time>', race_page).group()
    data['responses'][date_key]['body'] = data['responses'][date_key]['body'].replace('</a>', clock+'</a>')
    payload.write_text(json.dumps(data))
    report = refresh(tmp_path, payload, access(tmp_path), 6, lane='full')
    assert report['current_index_race_count'] == 6
    calls = [json.loads(x) for x in (tmp_path/'transport.jsonl').read_text().splitlines()]
    canonical = [x for x in calls if x['path'].endswith('/invented')]
    assert len(canonical) == 6, 'exact meeting times must not cause discovery plus download page fetches'
