"""Synthetic configuration/read-only contracts for R3 result candidate discovery."""
from pathlib import Path
import shlex

import pytest

from scripts import autonomous_official_result_capture as capture
from scripts import shadow_autopilot_daemon as daemon
from scripts import shadow_autopilot_v1 as autopilot
from scripts.r3_official_result_candidates import r3_prediction_candidates
from src.operator_ui.job_store import JobStore, JobStoreError
from tests.operator_ui.test_job_store import NOW, create, store


def test_readonly_store_never_creates_migrates_or_mutates(tmp_path):
    missing = tmp_path / "missing" / "jobs.db"
    with pytest.raises(JobStoreError):
        JobStore(missing, readonly=True)
    assert not missing.parent.exists()
    writable = store(tmp_path)
    job = create(writable)
    writable.path.chmod(0o400)
    before = writable.path.stat()
    raw = writable.path.read_bytes()
    readonly = JobStore(writable.path, readonly=True)
    assert readonly.recorded_jobs() == (job,)
    with pytest.raises(JobStoreError):
        create(readonly, key="different-key-12345")
    after = writable.path.stat()
    assert raw == writable.path.read_bytes()
    assert (before.st_mode, before.st_mtime_ns, before.st_ctime_ns) == (after.st_mode, after.st_mtime_ns, after.st_ctime_ns)
    assert not list(tmp_path.glob("jobs.db-*"))
    writable.path.chmod(0o600)


def test_corrupt_or_busy_store_cannot_discover_or_acquire(tmp_path, monkeypatch):
    writable = store(tmp_path)
    create(writable)
    args = dict(job_store_path=writable.path, prediction_bundles=tmp_path / "bundles",
                result_database=tmp_path / "results.db", target_date="2026-08-01",
                current_time=NOW, race_ids=[], output_dir=tmp_path / "out")
    # No READY job is a legitimate empty source.
    assert r3_prediction_candidates(**args)[0] == []
    sidecar = Path(str(writable.path) + "-wal")
    sidecar.write_bytes(b"busy")
    assert r3_prediction_candidates(**args)[1][0]["reason"] == "R3_JOB_STORE_UNAVAILABLE"
    sidecar.unlink()
    writable.path.write_bytes(b"corrupt fixture")
    assert r3_prediction_candidates(**args)[1][0]["reason"] == "R3_JOB_STORE_UNAVAILABLE"


def test_result_opt_in_round_trips_service_daemon_autopilot_capture(tmp_path):
    jobs = tmp_path / "R3 jobs.db"
    bundles = tmp_path / "R3 bundles"
    text = daemon.service_file_text(repo_path=tmp_path, timeout_seconds=60,
                                    r3_job_store=jobs, r3_prediction_bundles=bundles, skip_shadow_run=True)
    argv = shlex.split(next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("ExecStart=")))
    parsed = daemon.parse_args(argv[2:])
    assert parsed.r3_job_store == jobs and parsed.r3_prediction_bundles == bundles
    assert parsed.enable_autonomous_result_capture and parsed.skip_shadow_run
    default_unit = daemon.service_file_text(repo_path=tmp_path, timeout_seconds=60)
    assert "--r3-job-store" not in default_unit and "--skip-shadow-run" not in default_unit
    auto = autopilot.parse_args(["--enable-autonomous-result-capture", "--r3-job-store", str(jobs), "--r3-prediction-bundles", str(bundles)])
    command = autopilot.autonomous_official_result_capture_command(
        target_date="2026-08-01", upcoming_dir=None, snapshot_dir=None,
        output_dir=tmp_path / "output", evidence_root=tmp_path, db_path=tmp_path / "results.db",
        r3_job_store=auto.r3_job_store, r3_prediction_bundles=auto.r3_prediction_bundles)
    final = capture.parse_args(command[2:])
    assert final.r3_job_store == jobs and final.r3_prediction_bundles == bundles
    assert final.shadow_run_dir is None and final.upcoming_dir is None


@pytest.mark.parametrize("parse,prefix", [(daemon.parse_args, ["run-once"]), (autopilot.parse_args, []), (capture.parse_args, [])])
def test_partial_bindings_fail_closed(parse, prefix, tmp_path):
    with pytest.raises(SystemExit):
        parse([*prefix, "--r3-job-store", str(tmp_path / "jobs.db")])


@pytest.mark.parametrize("parse,prefix", [(daemon.parse_args, ["run-once"]), (autopilot.parse_args, [])])
def test_result_pair_needs_explicit_result_activation(parse, prefix, tmp_path):
    with pytest.raises(SystemExit):
        parse([*prefix, "--r3-job-store", str(tmp_path / "jobs.db"),
               "--r3-prediction-bundles", str(tmp_path / "bundles")])
