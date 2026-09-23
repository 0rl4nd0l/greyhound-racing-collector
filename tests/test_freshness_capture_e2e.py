"""Run the exported generated service through real planner/child/append/receipt."""

import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest


def fixture_data(root, scenario):
    now = datetime.now(ZoneInfo("Australia/Melbourne"))
    jump = (now + timedelta(minutes=27)).replace(second=0, microsecond=0)
    date = now.date().isoformat()
    url = f"https://www.thedogs.com.au/racing/murray-bridge-straight/{date}/9/fabricated"
    sportsbet = "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/murray-bridge-straight/race-9-12345678"
    runners = [
        dict(
            box_number=n,
            dog_name=name,
            scratch_state="ACTIVE",
            source_native_runner_id=str(900 + n),
        )
        for n, name in enumerate(("Alpha", "Bravo", "Charlie", "Delta"), 1)
    ]
    shadow = dict(
        status="PASS",
        metadata_is_leakage_safe=True,
        race_date=date,
        venue="MURRAY-BRIDGE-STRAIGHT",
        race_number=9,
        jump_time=jump.isoformat(),
        source_url=url,
        metadata_captured_at=now.isoformat(),
        source_native_race_id="900",
        runner_box_name_list=runners,
        canonical_final_runner_alignment=dict(
            status="aligned", canonical_runner_set_status="available"
        ),
    )
    race = dict(date=date, race_time=jump.strftime("%H:%M"), venue="MURR", race_number=9, url=url)
    sidecar = dict(
        metadata_is_leakage_safe=True,
        metadata_captured_at=now.isoformat(),
        prejump_shadow_metadata=shadow,
        runner_completeness_after_canonical_alignment=dict(
            status="COMPLETE", runner_count=4, participants=runners
        ),
        race_info=race,
        weather="Clear",
        track_condition="Good",
        weather_track_metadata_source="open_meteo_forecast_api+sportsbet_pre_race_page",
        weather_track_metadata_source_url={
            "open_meteo_forecast_api": "https://api.open-meteo.com/v1/forecast",
            "sportsbet_pre_race_page": sportsbet,
        },
        weather_track_metadata_is_leakage_safe=True,
        expert_form_metadata=dict(
            source="thedogs_expert_form_page",
            source_url=url + "/expert-form",
            captured_at=now.isoformat(),
            metadata_is_leakage_safe=True,
            runners=runners,
        ),
    )
    cards = "".join(
        f'<div data-automation-id="racecard-outcome-{r["box_number"]}"><div data-automation-id="racecard-outcome-name"><span>{r["box_number"]}. {r["dog_name"]}</span></div><span data-automation-id="price-text">4.00</span><span>1.50</span><span>EW</span></div>'
        for r in runners
    )
    # A different race returned by the source must fail native capture validation.
    if scenario == "mismatch":
        sportsbet = sportsbet.replace("race-9-", "race-8-")
    return now, dict(
        race=race,
        filename=f"Race 9 - MURRAY-BRIDGE-STRAIGHT - {date}.csv",
        csv="box|dog_name\n" + "".join(f'{r["box_number"]}|{r["dog_name"]}\n' for r in runners),
        sidecar=sidecar,
        landing_html=f'<a href="{sportsbet}">R9 Murray Bridge Straight\n27m</a>',
        race_html="<h1>Murray Bridge Straight Race 9</h1>" + cards,
        scenario=scenario,
        transport_marker=str(root / "transport-started"),
        cleanup_marker=str(root / "transport-cleaned"),
    )


@pytest.mark.parametrize("campaign_mode", [False, True])
@pytest.mark.parametrize("scenario", ["canonical_alias", "mismatch", "interrupted", "delayed_start", "expired_append", "source_denial_full", "source_denial_odds", "source_recovery", "source_denial_python_full", "source_denial_python_odds"])
def test_actual_packaged_service_capture(tmp_path, scenario, campaign_mode, monkeypatch):
    from scripts.prepare_freshness_rehearsal import prepare, UNITS
    from scripts.check_freshness_service import service_command
    from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest
    from sportsbet_odds_integrator import SportsbetOddsIntegrator
    from utils.sportsbet_access import SportsbetAccess

    access = tmp_path / "sportsbet-access.json"
    SportsbetAccess(access).initialize(access_basis={"status": "permitted", "reference": "fabricated test"})
    if scenario == "source_recovery":
        SportsbetAccess(access, clock=lambda: time.time() - 1801).retain_denial(429)
    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(access))

    stamp, data = fixture_data(tmp_path, scenario)
    campaign = None
    if campaign_mode:
        from tests.test_freshness_campaign import make_campaign
        campaign = make_campaign(tmp_path / "campaign")
    installed = tmp_path / "installed"
    installed.mkdir()
    for name in (*UNITS, "greyhound-operator-ui-r3.service"):
        (installed / name).write_text("fabricated original " + name)
    db = tmp_path / "synthetic.sqlite"
    SportsbetOddsIntegrator(str(db), allow_auto_scrape_odds=False)
    package = tmp_path / "package"
    prepared = prepare(
        campaign_root=campaign.root if campaign else None,
        output=package,
        start=stamp - timedelta(seconds=5),
        python=Path(sys.executable),
        db=db,
        lock=tmp_path / "collector.lock",
        reconciliation_roots={},
        installed_dir=installed,
    )
    plan = json.loads((package / "plan.json").read_bytes())
    accounting = dict(
        schema_version="freshness_attempt_reconciliation_v1",
        complete=True,
        consumed=[],
        sources=[{"sha256": "a" * 64}],
    )
    contract = {
        key: plan[key]
        for key in (
            "profile",
            "rehearsal_id",
            "starts_at",
            "ends_at",
            "lock_path",
            "evidence_root",
            "db_path",
            "cleanup_seconds",
            "max_capture_attempts",
            "max_logical_requests",
            "source_identity_sha256",
            "runtime_sha256",
        )
    }
    if campaign:
        contract.update({key: plan[key] for key in ("campaign_root", "campaign_authorization_sha256")})
        campaign.begin(plan["rehearsal_id"], now=stamp, deadline=stamp + timedelta(minutes=110))
    contract.update(
        schema_version="freshness_rehearsal_contract_v1",
        source_date=stamp.date().isoformat(),
        reconciliation_sha256=digest(accounting),
    )
    (package / "contract.json").write_text(json.dumps(contract))
    allowance = AttemptAllowance(FreshnessContract(contract))
    allowance.initialize(accounting)
    fixture = tmp_path / "fabricated.json"
    fixture.write_text(json.dumps(data))
    first_unit = "shadow-autopilot-odds-capture.service" if scenario.endswith("_odds") else "shadow-autopilot.service"
    command, cwd, env = service_command(package / "units" / first_unit)
    env.update(
        PYTHONPATH=str(Path(__file__).parent / "fixtures/freshness_transport")
        + os.pathsep
        + str(package / "source"),
        FRESHNESS_FABRICATED_SOURCE=str(fixture),
    )
    # This launcher installs the kernel filter before the real ExecStart is exec'd.
    launcher = "from scripts.check_freshness_service import deny_network; import os,sys; deny_network(); os.execv(sys.argv[1],sys.argv[1:])"
    monitor = Path(__file__).parent / "fixtures/freshness_capture_monitor.py"
    before = subprocess.run([sys.executable, str(monitor), str(package), "startup"], text=True, capture_output=True, timeout=20)
    assert before.returncode == 0, before.stdout + before.stderr
    with (tmp_path / "service.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", launcher, *command],
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if scenario == "interrupted":
            deadline = time.monotonic() + 45
            while (
                not Path(data["transport_marker"]).exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
            assert Path(data["transport_marker"]).exists(), (tmp_path / "service.log").read_text()
            process.send_signal(signal.SIGTERM)
            time.sleep(0.25)
            assert (tmp_path / "collector.lock").exists(), "lock released before child cleanup"
        process.wait(timeout=90)
    log = (tmp_path / "service.log").read_text()
    claims = allowance.claims()
    if scenario.startswith("source_denial_python"):
        assert claims == [], log
        assert Path(data["transport_marker"]).exists(), log
        assert SportsbetAccess(access).read()["phase"] == "COOLDOWN"
        for name in ("shadow-autopilot.service", "shadow-autopilot-odds-capture.service"):
            blocked_command, blocked_cwd, _ = service_command(package / "units" / name)
            blocked = subprocess.run([sys.executable, "-c", launcher, *blocked_command], cwd=blocked_cwd, env=env, text=True, capture_output=True, timeout=20)
            assert blocked.returncode != 0 and "sportsbet_source_hold" in blocked.stderr
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM live_odds").fetchone()[0] == 0
        if campaign:
            with campaign.ledger() as ledger:
                assert ledger["attempts"] == [] and ledger["logical_requests"] == 1
        restored = subprocess.run([sys.executable, str(monitor), str(package), "restore"], text=True, capture_output=True, timeout=20)
        assert restored.returncode == 0, restored.stdout + restored.stderr
        return
    assert len(claims) == 1, log
    claim_bytes = claims[0].read_bytes()
    claim = json.loads(claim_bytes)
    canonical = f"Race 9 - MURR - {stamp.date().isoformat()}"
    assert claim["item"]["race_id"] == canonical
    assert allowance.available() == campaign_mode
    assert allowance.consumed(claim["item"])
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT race_id,market_type FROM live_odds").fetchall()
    lifecycles = list(
        Path(plan["evidence_root"]).glob(
            "shadow_autopilot_daemon_runtime/service-lifecycles/*.json"
        )
    )
    assert len(lifecycles) == 1
    lifecycle = json.loads(lifecycles[0].read_bytes())
    assert lifecycle["children_reaped"]
    assert not (tmp_path / "collector.lock").exists()
    if scenario in {"canonical_alias", "source_recovery"}:
        assert process.returncode == 0, log
        assert len(rows) == 8 and {r[0] for r in rows} == {canonical}, log
        assert {r[1] for r in rows} == {"win", "place"}
        consumer = """
import json, sys
from pathlib import Path
from datetime import datetime
from scripts.check_freshness_service import deny_network
deny_network()
from race_collection.manual_prediction_collector_request import ManualPredictionCollectorProtocol
root, canonical, alias = sys.argv[1:]
protocol = ManualPredictionCollectorProtocol(Path(root)/'manual_prediction_collector_requests_v1')
for race_id in (canonical, alias):
    handoff = protocol.discover_collector_exact_handoff(race_id=race_id,current_time=datetime.now().astimezone(),max_age_seconds=300)
    assert handoff is not None
    source = json.loads(handoff['_report_bytes'])
    assert source['source_race_id'] == canonical
    assert source['source_plan_item']['canonical_race_id'] == canonical
    assert source['source_attempt']['append_report']['race_id'] == canonical
print('BOTH_ALIASES_VERIFIED')
"""
        consumed = subprocess.run(
            [
                sys.executable,
                "-c",
                consumer,
                plan["evidence_root"],
                canonical,
                data["filename"][:-4],
            ],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert consumed.returncode == 0 and "BOTH_ALIASES_VERIFIED" in consumed.stdout, (
            consumed.stdout + consumed.stderr
        )
        from scripts.run_freshness_rehearsal import verify_claim_receipt
        from race_collection.manual_prediction_collector_request import ManualPredictionCollectorProtocol
        handoff = ManualPredictionCollectorProtocol(Path(plan["evidence_root"]) / 'manual_prediction_collector_requests_v1').discover_collector_exact_handoff(
            race_id=canonical, current_time=datetime.now().astimezone(), max_age_seconds=300)
        verify_claim_receipt(claims[0], handoff, Path(plan["evidence_root"]), Path(cwd))
        # An earlier-window receipt cannot certify a later-window reservation.
        wrong = json.loads(claim_bytes)
        wrong["item"]["capture_window_minutes"] = 10
        claims[0].write_text(json.dumps(wrong))
        with pytest.raises(ValueError, match="receipt_reservation_mismatch"):
            verify_claim_receipt(claims[0], handoff, Path(plan["evidence_root"]), Path(cwd))
        claims[0].write_bytes(claim_bytes)
        # The other real generated lane progresses, but the shared spent claim
        # prevents a second acquisition even though the window remains eligible.
        odds_command, odds_cwd, _ = service_command(
            package / "units/shadow-autopilot-odds-capture.service"
        )
        odds = subprocess.run(
            [sys.executable, "-c", launcher, *odds_command],
            cwd=odds_cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
        )
        (tmp_path / "odds-service.log").write_text(odds.stdout + odds.stderr)
        assert odds.returncode == 0, odds.stdout + odds.stderr
        assert claims[0].read_bytes() == claim_bytes
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM live_odds").fetchone()[0] == 8
        metrics = json.loads((claims[0].with_suffix(".requests.json") if campaign else allowance.scope.session / "capture-requests.json").read_bytes())
        assert metrics["browser_navigation_attempts"] == 2
        assert metrics["observed_provider_requests"] == 2
        assert metrics["observed_other_requests"] == 0
        if campaign:
            second = json.loads(json.dumps(data).replace("Race 9", "Race 10").replace("R9", "R10").replace("race-9-", "race-10-").replace("/9/", "/10/").replace('"race_number": 9', '"race_number": 10'))
            fixture.write_text(json.dumps(second))
            next_capture = subprocess.run([sys.executable, "-c", launcher, *odds_command], cwd=odds_cwd, env=env, text=True, capture_output=True, timeout=60)
            assert next_capture.returncode == 0, next_capture.stdout + next_capture.stderr
            all_claims = allowance.claims()
            assert len(all_claims) == 2
            with sqlite3.connect(db) as conn:
                assert conn.execute("SELECT COUNT(*) FROM live_odds").fetchone()[0] == 16
            with campaign.ledger() as ledger:
                assert len(ledger["attempts"]) == 2 and ledger["logical_requests"] == 4

        if scenario == "source_recovery":
            assert SportsbetAccess(access).read()["recovery_attempts"] == 1
            assert SportsbetAccess(access).read()["phase"] == "OPEN"
            # A renewed denial permanently stops automatic source recovery.
            SportsbetAccess(access).retain_denial(429, {"Retry-After": "0"})
            denied = subprocess.run([sys.executable, "-c", launcher, *odds_command], cwd=odds_cwd, env=env, text=True, capture_output=True, timeout=20)
            assert denied.returncode != 0 and "sportsbet_source_hold" in denied.stderr
            assert SportsbetAccess(access).read()["phase"] == "STOP"

    else:
        assert rows == [], log
    if scenario.startswith("source_denial"):
        assert SportsbetAccess(access).read()["phase"] == "COOLDOWN"
        other = "shadow-autopilot.service" if first_unit.endswith("odds-capture.service") else "shadow-autopilot-odds-capture.service"
        other_command, other_cwd, _ = service_command(package / "units" / other)
        before = SportsbetAccess(access).read()
        restarted = subprocess.run([sys.executable, "-c", launcher, *other_command], cwd=other_cwd, env=env, text=True, capture_output=True, timeout=20)
        assert restarted.returncode != 0 and "sportsbet_source_hold" in restarted.stderr
        assert SportsbetAccess(access).read() == before
        assert claims[0].read_bytes() == claim_bytes
    if scenario in {"delayed_start", "expired_append"}:
        errors = "\n".join(path.read_text() for path in Path(plan["evidence_root"]).glob("**/autonomous_live_odds_capture.stderr.txt"))
        assert "capture_reservation_identity_changed" in errors, errors
        assert claim["item"]["capture_window_minutes"] == 30
        assert (allowance.scope.session / "STOP.json").exists()
    if scenario == "interrupted":
        assert lifecycle["interrupted"]
        timing_file = next(
            Path(plan["evidence_root"]).glob("shadow_autopilot_daemonization*/terminal-timing.json")
        )
        timing = json.loads(timing_file.read_bytes())["timing"]
        assert timing["phase_seconds"] >= 5, "interrupted acquisition charged as overhead"
        assert claims[0].with_suffix(".terminal.json").exists()
        finished = list(
            Path(plan["evidence_root"]).glob(
                "shadow_autopilot_daemonization*/phase-1/logs/*.finished.json"
            )
        )
        assert len(finished) == 1 and json.loads(finished[0].read_bytes())["interrupted"] is True
        assert Path(data["cleanup_marker"]).exists()
        assert (allowance.scope.session / "STOP.json").exists()
    assert claims[0].read_bytes() == claim_bytes
    restored = subprocess.run([sys.executable, str(monitor), str(package), "restore"], text=True, capture_output=True, timeout=20)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    (tmp_path / "result.json").write_text(
        json.dumps(
            dict(
                scenario=scenario,
                commit=prepared["commit"],
                rows=len(rows),
                lifecycle=lifecycle,
                network="KERNEL_IPV4_IPV6_DENIED",
                command=command,
            ),
            indent=2,
        )
    )
