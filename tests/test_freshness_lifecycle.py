import pytest


def test_invocation_binding_includes_pre_exec_start_and_final_process_exit():
    from race_collection.freshness_rehearsal import completed_service_overhead

    status = {
        "ActiveState": "inactive",
        "InvocationID": "a" * 32,
        "ExecMainPID": "12",
        "ExecMainStartTimestampMonotonic": "100020000",
        "ExecMainExitTimestampMonotonic": "153000000",
    }
    report = {
        "timing": {
            "service_invocation_id": "a" * 32,
            "process_pid": 13,
            "process_started_monotonic": 100.3,
            "phase_seconds": 50.5,
            "lock_wait_seconds": 0,
        }
    }
    lifecycle = {
        "invocation_id": "a" * 32,
        "wrapper_pid": 12,
        "child_pid": 13,
        "process_start_lower_bound": 100.0,
        "completed_monotonic": 152.9,
        "children_reaped": True,
    }
    assert completed_service_overhead(status, report, lifecycle) == 2.5
    with pytest.raises(ValueError, match="identity"):
        completed_service_overhead({**status, "InvocationID": "b" * 32}, report, lifecycle)


def test_phase_lifecycle_waits_for_detached_child_cleanup(tmp_path):
    import json
    import sys
    from scripts.shadow_autopilot_daemon import run_command

    marker = tmp_path / "child-cleaned"
    child = (
        "import time; from pathlib import Path; time.sleep(.5); Path("
        + repr(str(marker))
        + ").write_text('cleaned')"
    )
    parent = (
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c',"
        + repr(child)
        + "],start_new_session=True)"
    )
    result = run_command(
        name="fabricated-detached-cleanup",
        command=[sys.executable, "-c", parent],
        output_dir=tmp_path,
        timeout_seconds=5,
        cwd=tmp_path,
        wait_for_descendants=True,
    )
    assert marker.read_text() == "cleaned"
    assert result["status"] == "PASS" and result["duration_seconds"] >= 0.5


def test_detached_child_cannot_extend_phase_past_existing_timeout(tmp_path):
    import sys
    from scripts.shadow_autopilot_daemon import run_command

    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(4)"
    parent = (
        'import subprocess,sys; subprocess.Popen([sys.executable,"-c",'
        + repr(child)
        + "],start_new_session=True)"
    )
    result = run_command(
        name="detached-timeout",
        command=[sys.executable, "-c", parent],
        output_dir=tmp_path,
        timeout_seconds=1,
        cwd=tmp_path,
        wait_for_descendants=True,
    )
    assert result["timed_out"] and result["duration_seconds"] < 3
