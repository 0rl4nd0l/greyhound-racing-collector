#!/usr/bin/env python3
"""Bound the actual service process and naturally reap all owned descendants."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from race_collection.live_phase_checkpoint import atomic_json


def main():
    arguments = sys.argv[1:]
    command = [sys.executable, str(ROOT / "scripts/shadow_autopilot_daemon.py"), *arguments]
    if "--verify-live-runtime" in arguments:
        os.execv(sys.executable, command)
    from utils.sportsbet_access import SportsbetAccess
    # The daemon acquires the collector lock before requesting source ownership.
    SportsbetAccess().check_admission(allow_active=True)
    contract = Path(arguments[arguments.index("--live-freshness-contract") + 1])
    from race_collection.live_execution import configure_profile_execution

    configure_profile_execution(contract)
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("child_subreaper_unavailable")
    invocation = os.environ.get("INVOCATION_ID") or uuid.uuid4().hex
    if len(invocation) != 32 or any(c not in "0123456789abcdef" for c in invocation):
        raise ValueError("invalid_service_invocation")
    os.environ["GREYHOUND_SERVICE_INVOCATION"] = invocation
    value = json.loads(contract.read_bytes())
    output = (
        Path(value["evidence_root"])
        / "shadow_autopilot_daemon_runtime/service-lifecycles"
        / f"{invocation}.json"
    )
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    tick = 1 / os.sysconf("SC_CLK_TCK")
    # /proc rounds process birth down to a tick: use the conservative lower bound.
    lower = int(fields[19]) * tick - (time.clock_gettime(time.CLOCK_BOOTTIME) - time.monotonic())
    record = {
        "invocation_id": invocation,
        "wrapper_pid": os.getpid(),
        "process_start_lower_bound": lower,
        "process_start_tick_seconds": tick,
        "launch_started_monotonic": time.monotonic(),
        "interrupted": False,
        "children_reaped": False,
        "status": "STARTED",
    }
    child = None

    def interrupted(signum, frame):
        record["interrupted"] = True
        from race_collection.live_freshness_contract import FreshnessContract

        FreshnessContract(value).stop("SERVICE_INTERRUPTED")
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    atomic_json(output, record)
    child = subprocess.Popen(command, start_new_session=True)
    record["child_pid"] = child.pid
    atomic_json(output, record)
    result = child.wait()
    # PR_SET_CHILD_SUBREAPER makes orphaned descendants ours even if they made
    # another process group. Completion never means merely that the leader exited.
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break
        except InterruptedError:
            continue
    record.update(
        completed_monotonic=time.monotonic(),
        children_reaped=True,
        returncode=result,
        status="INTERRUPTED" if record["interrupted"] else "COMPLETE",
    )
    atomic_json(output, record)
    return result if result >= 0 else 128 - result


if __name__ == "__main__":
    raise SystemExit(main())
