"""Subprocess-only synthetic adapters around the real packaged supervisor main()."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

PACKAGE, SCENARIO, APPROVAL = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "source"))
sys.path.append(str(WORKTREE))
from scripts.check_freshness_candidate_offline import guard

sys.addaudithook(guard)
from scripts import run_freshness_rehearsal as run
from scripts import shadow_autopilot_v1 as autopilot
from race_collection.live_phase_checkpoint import atomic_json
from race_collection.live_freshness_contract import FreshnessContract
from race_collection.synchronous_manual_capture import current_race_index_path
from tests.race_collection.test_synchronous_manual_capture import _runner_coverage

PLAN = json.loads((PACKAGE / "plan.json").read_bytes())
START = datetime.fromisoformat(PLAN["starts_at"])
EVIDENCE = Path(PLAN["evidence_root"])
STATE = EVIDENCE / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
assert run.ROOT == PACKAGE / "source"


def publish(stamp):
    output = EVIDENCE / "synthetic-refresh"
    output.mkdir()
    source = output / "odds_capture_refresh_report.json"
    url = "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"
    atomic_json(
        source,
        {
            "status": "SUCCESS",
            "generated_at": stamp.isoformat(),
            "sidecar_metadata_coverage": _runner_coverage(output, url, stamp),
            "selected_count": 1,
            "selected_races": [
                {
                    "date": "2026-07-19",
                    "jump_datetime": "2026-07-19T13:00:00+10:00",
                    "race_id": "Race 5 - GUNN - 2026-07-19",
                    "race_id_aliases": [
                        "Race 5 - GUNN - 2026-07-19",
                        "Race 5 - GUNNEDAH - 2026-07-19",
                    ],
                    "race_number": 5,
                    "source_native_race_id": "15900",
                    "race_time": "13:00",
                    "race_url": url,
                    "venue": "GUNN",
                }
            ],
        },
    )
    result = autopilot.publish_current_race_index_after_refresh(
        state_path=STATE,
        evidence_root=EVIDENCE,
        output_dir=output,
        run_id="synthetic-refresh",
        source_refresh_report_path=source,
        enforce_monotonic=True,
    )
    assert result["status"] == "PUBLISHED", result
    if SCENARIO == "publication":
        from tests.operator_ui.test_live_adapters import actual_payloads

        values = actual_payloads(stamp, include_models=False)
        refresh = json.loads(source.read_bytes())
        values["odds_report"]["odds_capture_refresh_report"] = refresh
        for lane, filename in (
            ("full", "daemon_run_report.json"),
            ("odds", "odds_capture_only_daemon_report.json"),
        ):
            directory = EVIDENCE / ("shadow_autopilot_daemonization_v1_" + lane)
            values[lane + "_state"]["last_output_dir" if lane == "full" else "output_dir"] = str(
                directory
            )
            values[lane + "_report"]["output_dir"] = str(directory)
            if lane == "odds":
                values["odds_report"]["autopilot_output_dir"] = str(output)
                values["odds_state"]["autopilot_output_dir"] = str(output)
            atomic_json(directory / filename, values[lane + "_report"])
            atomic_json(
                STATE if lane == "odds" else STATE.parent / "state.json", values[lane + "_state"]
            )


class Clock:
    elapsed = -2.0
    delivered = False

    def now(self):
        return START + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return 1000 + self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds
        if self.elapsed >= 2 and not self.delivered:
            self.delivered = True
            if SCENARIO in {"publication", "stale", "malformed"}:
                publish(START - timedelta(seconds=301) if SCENARIO == "stale" else self.now())
            if SCENARIO == "malformed":
                current_race_index_path(STATE).write_text("{broken")
        if self.elapsed >= 8 and SCENARIO in {"empty", "publication", "stale"}:
            FreshnessContract(json.loads((PACKAGE / "contract.json").read_bytes())).stop(
                "SYNTHETIC_STOP"
            )


class Control:
    active = {timer: True for timer in run.TIMERS}

    def show(self, name):
        return {
            "ActiveState": (
                "active"
                if self.active.get(name, name == "greyhound-operator-ui-r3.service")
                else "inactive"
            ),
            "SubState": "dead",
            "MainPID": "123" if name == "greyhound-operator-ui-r3.service" else "0",
            "DropInPaths": "",
            "ControlGroup": "",
            "WorkingDirectory": PLAN["source_root"],
            "ExecMainStartTimestampMonotonic": "0",
            "ExecMainExitTimestampMonotonic": "0",
        }

    def idle(self):
        return True

    def command(self, action, *args):
        if action in {"start", "stop"}:
            self.active[args[0]] = action == "start"
        elif action == "is-enabled":
            return "enabled\n"
        elif action == "show":
            return "LastTriggerUSecMonotonic=0\nNextElapseUSecMonotonic=0\nNextElapseUSecRealtime=0\nActiveState=active\n"
        elif action != "daemon-reload":
            raise AssertionError((action, args))
        return ""


clock = Clock()
run.now = clock.now
run.time = SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
control = Control()
run.SystemdControl = lambda: control
sys.argv = [
    str(run.__file__),
    "--plan",
    str(PACKAGE / "plan.json"),
    "--plan-sha256",
    APPROVAL,
    "--approval-id",
    "OFFLINE-SYNTHETIC",
]
try:
    run.main()
except ValueError as error:
    expected = {
        "empty": "candidate_scope_stopped",
        "publication": "candidate_scope_stopped",
        "stale": "candidate_scope_stopped",
        "unavailable": "native_index_not_available",
        "malformed": "native_integrity_or_authority_failed",
    }[SCENARIO]
    assert str(error) == expected, (SCENARIO, str(error), expected)
else:
    raise AssertionError("synthetic stop was not enforced")
samples = [json.loads(path.read_bytes()) for path in sorted((PACKAGE / "samples").glob("*.json"))]
assert samples[0]["index_status"] == "UNAVAILABLE/DATA_MISSING", samples[0]
assert samples[0]["collector_status"] == "UNAVAILABLE/DATA_MISSING"
assert all(row["authority_status"] == "AVAILABLE/FRESH" for row in samples)
if SCENARIO == "publication":
    assert any(
        row["index_status"] == row["collector_status"] == "AVAILABLE/FRESH" for row in samples
    ), samples
if SCENARIO == "stale":
    assert any("STALE" in row["index_status"] for row in samples), samples
if SCENARIO == "unavailable":
    assert clock.elapsed == 180
if SCENARIO == "malformed":
    assert samples[-1]["index_status"] == "INVALID/INTEGRITY_FAILED", samples[-1]
restored = json.loads((PACKAGE / "restored.json").read_bytes())
assert restored["hashes"] == PLAN["baseline_unit_sha256"]
assert all(control.active[timer] for timer in run.TIMERS)
scope = FreshnessContract(json.loads((PACKAGE / "contract.json").read_bytes()))
assert not (scope.session / "capture-reservation.json").exists()
assert not (scope.session / "request-count.json").exists()
assert (scope.session / "STOP.json").exists()
atomic_json(
    PACKAGE / "fixture-result.json",
    {
        "restored": True,
        "actual_packaged_entrypoint": True,
        "samples": len(samples),
        "scenario": SCENARIO,
        "source_commit": PLAN["commit"],
    },
)
