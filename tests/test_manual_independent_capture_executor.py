from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.predictor.manual_independent_capture import (
    PROTECTED_PATH_KEYS,
    canonical_sha256,
    parse_canonical_json,
    validate_terminal_artifact,
)
from src.predictor.manual_independent_capture_executor import (
    FixtureChildLaunch,
    execute_manual_capture_fixture,
)
from src.predictor.on_demand import sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "tests/fixtures/manual_independent_capture_child.py"
SOURCE_COMMIT = "e4f3699986237aad265b34e77d06d536f6046ee4"
SOURCE_TREE = "ac9cdde82d3a0ede953e36c9a87d9afd216c5826"
MODEL_BYTES = b'{"model":"fixture-only-no-scoring"}\n'
NOW = datetime(2026, 8, 5, 1, 0, 0, tzinfo=timezone.utc)


def _race(*, scheduled: datetime | None = None) -> dict:
    scheduled = scheduled or NOW + timedelta(hours=1)
    return {
        "url": "https://www.thedogs.com.au/racing/richmond/2026-08-05/1/race-name",
        "race_id": "Race 1 - RICH - 2026-08-05",
        "race_date": "2026-08-05",
        "venue": "RICH",
        "venue_slug": "richmond",
        "race_number": 1,
        "scheduled_start": scheduled.isoformat(),
    }


def _config(tmp_path: Path, **timing: int) -> dict:
    operations_root = tmp_path / "manual-operations"
    operations_root.mkdir()
    manual_root = operations_root / "manual-independent-capture-v1"
    return {
        "schema_version": "manual_independent_capture_config_v1",
        "contract_version": "manual-independent-capture-v1",
        "safety": {
            "research_only": True,
            "canonical": False,
            "phase7_excluded": True,
            "phase7_eligible": False,
            "phase7_exclusion_reason": "manual_research_only_noncanonical",
        },
        "authority_profile": "manual_independent_capture_research_only_v1",
        "paths": {
            "operations_root": str(operations_root),
            "manual_root": str(manual_root),
            "browser_profile": str(manual_root / "browser-profile"),
            "runs_root": str(manual_root / "runs"),
            "manual_lock": str(manual_root / "manual-capture.lock"),
        },
        "timing": {
            "minimum_prejump_margin_seconds": timing.get("minimum", 120),
            "hard_timeout_seconds": timing.get("timeout", 5),
            "cancellation_grace_seconds": timing.get("grace", 1),
        },
        "attempt_policy": {
            "max_concurrent_manual_runs": 1,
            "max_capture_attempts": 1,
            "retries_allowed": False,
            "replay_allowed": False,
        },
    }


def _forbidden(tmp_path: Path, *, sentinel: Path | None = None) -> dict[str, str]:
    protected = tmp_path / "protected"
    protected.mkdir(exist_ok=True)
    values = {
        name: str(protected / name.replace("_", "-")) for name in PROTECTED_PATH_KEYS
    }
    if sentinel is not None:
        values["autonomous_shared_lock"] = str(sentinel)
    return values


def _command(mode: str, source_timestamp: str = NOW.isoformat()):
    def build(launch: FixtureChildLaunch) -> list[str]:
        return [
            sys.executable,
            str(CHILD),
            mode,
            "--source-timestamp",
            source_timestamp,
            "--race-sha",
            canonical_sha256(launch.selected_race),
        ]

    return build


def _execute(
    tmp_path: Path,
    *,
    cfg: dict | None = None,
    race: dict | None = None,
    command=None,
    forbidden: dict[str, str] | None = None,
    **kwargs,
):
    selected = race or _race()
    return execute_manual_capture_fixture(
        config=cfg or _config(tmp_path),
        forbidden_paths=forbidden or _forbidden(tmp_path),
        requested_race_url=selected["url"],
        selected_race=selected,
        model_bytes=MODEL_BYTES,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        fixture_child_command=command or _command("success"),
        now=lambda: NOW,
        **kwargs,
    )


def _revalidate(result, cfg: dict, forbidden: dict[str, str]) -> dict:
    artifact = parse_canonical_json(result.terminal_path.read_bytes())
    members = {
        row["path"]: (result.run_dir / row["path"]).read_bytes()
        for row in [
            *artifact["provenance"]["source_files"],
            *artifact["provenance"]["artifact_hashes"],
        ]
    }
    return validate_terminal_artifact(
        artifact,
        config=cfg,
        forbidden_paths=forbidden,
        member_bytes=members,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_tree=SOURCE_TREE,
        expected_model_sha256=sha256_bytes(MODEL_BYTES),
        expected_source_files=deepcopy(artifact["provenance"]["source_files"]),
        expected_runner_set_sha256=artifact["provenance"]["runner_set_sha256"],
        expected_odds_sha256=artifact["provenance"]["odds_sha256"],
        expected_run_id=artifact["run_id"],
        expected_request_id=artifact["request"]["request_id"],
        expected_request_sha256=canonical_sha256(artifact["request"]),
        seen_run_ids=set(),
        seen_request_ids=set(),
        seen_request_sha256s=set(),
    )


def test_success_is_one_isolated_fixture_attempt_and_valid_ghu050_artifact(tmp_path: Path):
    cfg = _config(tmp_path)
    forbidden = _forbidden(tmp_path)
    calls = []

    def tracked_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.Popen(*args, **kwargs)

    result = _execute(tmp_path, cfg=cfg, forbidden=forbidden, popen=tracked_popen)

    assert result.artifact["terminal"] == {
        "status": "CAPTURE_READY",
        "failure_code": None,
    }
    assert result.artifact["attempt"] == {
        "attempt_count": 1,
        "source_attempt_count": 1,
    }
    assert len(calls) == 1
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["cwd"] == result.run_dir
    assert calls[0][1]["env"]["MANUAL_CAPTURE_PROFILE"] == cfg["paths"][
        "browser_profile"
    ]
    assert "canonical" not in " ".join(calls[0][1]["env"]).lower()
    assert result.pid == result.pgid
    assert result.cleanup.confirmed is True
    assert result.cleanup.process_group_absent is True
    assert _revalidate(result, cfg, forbidden) == result.artifact
    manual_root = Path(cfg["paths"]["manual_root"]).resolve()
    assert all(path.resolve().is_relative_to(manual_root) for path in result.run_dir.rglob("*"))
    assert {path.relative_to(result.run_dir).as_posix() for path in result.run_dir.rglob("*") if path.is_file()} == {
        "capture/odds.json",
        "config/config.json",
        "model/model.json",
        "sources/form.csv",
        "terminal.json",
    }


def test_second_invocation_is_manual_busy_before_second_child_launch(tmp_path: Path):
    cfg = _config(tmp_path, timeout=5, grace=1)
    forbidden = _forbidden(tmp_path)
    cancel = threading.Event()
    command_built = threading.Event()
    first_result = []
    first_error = []

    def first_command(launch: FixtureChildLaunch):
        command_built.set()
        return _command("hang")(launch)

    def run_first():
        try:
            first_result.append(
                _execute(
                    tmp_path,
                    cfg=cfg,
                    forbidden=forbidden,
                    command=first_command,
                    cancellation_token=cancel,
                )
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - diagnostic capture
            first_error.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert command_built.wait(timeout=2)
    time.sleep(0.05)
    second_launches = []

    def forbidden_second_launch(*args, **kwargs):
        second_launches.append((args, kwargs))
        raise AssertionError("busy invocation launched a child")

    second = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        popen=forbidden_second_launch,
    )
    cancel.set()
    thread.join(timeout=3)

    assert not first_error
    assert first_result and first_result[0].artifact["terminal"]["failure_code"] == "CANCELLED"
    assert second.artifact["terminal"] == {
        "status": "BLOCKED",
        "failure_code": "MANUAL_BUSY",
    }
    assert second.artifact["attempt"]["source_attempt_count"] == 0
    assert second_launches == []
    assert _revalidate(second, cfg, forbidden) == second.artifact


class _StepClock:
    def __init__(self, step: float = 0.6):
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


class _FakeProcess:
    def __init__(self):
        self.pid = 54321
        self.returncode = None
        self.reaped = False

    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired(["fixture"], timeout)

    def wait(self, timeout=None):
        if not self.reaped:
            raise subprocess.TimeoutExpired(["fixture"], timeout)
        self.returncode = -signal.SIGKILL
        return self.returncode

    def poll(self):
        return self.returncode


class _KillController:
    def __init__(self, process: _FakeProcess, *, confirm: bool = True):
        self.process = process
        self.group_exists = True
        self.confirm = confirm
        self.signals = []

    def __call__(self, pgid: int, signum: int):
        assert pgid == self.process.pid
        if signum == 0:
            if not self.group_exists:
                raise ProcessLookupError
            return
        self.signals.append(signum)
        if signum == signal.SIGKILL and self.confirm:
            self.group_exists = False
            self.process.reaped = True


def test_timeout_uses_monotonic_deadline_and_term_then_kill_cleanup(tmp_path: Path):
    cfg = _config(tmp_path, timeout=1, grace=1)
    forbidden = _forbidden(tmp_path)
    process = _FakeProcess()
    controller = _KillController(process)
    popen_calls = []

    def popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return process

    result = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        popen=popen,
        killpg=controller,
        monotonic=_StepClock(),
        sleep=lambda _: None,
    )

    assert len(popen_calls) == 1
    assert popen_calls[0][1]["start_new_session"] is True
    assert controller.signals == [signal.SIGTERM, signal.SIGKILL]
    assert result.cleanup.confirmed is True
    assert result.artifact["terminal"]["failure_code"] == "TIMED_OUT"
    assert result.artifact["timing"]["terminal_at"] == result.artifact["timing"][
        "deadline_at"
    ]
    assert _revalidate(result, cfg, forbidden) == result.artifact


def test_real_hanging_child_times_out_only_after_confirmed_group_cleanup(
    tmp_path: Path,
):
    cfg = _config(tmp_path, timeout=3, grace=2)
    forbidden = _forbidden(tmp_path)
    started = time.monotonic()
    result = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        command=_command("ignore-term"),
    )
    elapsed = time.monotonic() - started

    assert elapsed >= 3
    assert result.artifact["terminal"]["failure_code"] == "TIMED_OUT"
    assert result.artifact["timing"]["cleanup_deadline_at"] == result.artifact[
        "timing"
    ]["deadline_at"]
    assert result.artifact["timing"]["terminal_at"] == result.artifact["timing"][
        "deadline_at"
    ]
    assert result.cleanup.term_sent is True
    assert result.cleanup.kill_sent is True
    assert result.cleanup.leader_reaped is True
    assert result.cleanup.process_group_absent is True
    assert result.cleanup.confirmed is True
    assert _revalidate(result, cfg, forbidden) == result.artifact


class _CancelAfterLaunch:
    def __init__(self):
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls >= 2


def test_external_cancellation_reaps_whole_process_group(tmp_path: Path):
    cfg = _config(tmp_path, timeout=5, grace=1)
    forbidden = _forbidden(tmp_path)
    process = _FakeProcess()
    controller = _KillController(process)
    result = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        popen=lambda *args, **kwargs: process,
        killpg=controller,
        monotonic=_StepClock(step=0.1),
        sleep=lambda _: None,
        cancellation_token=_CancelAfterLaunch(),
    )

    assert controller.signals == [signal.SIGTERM, signal.SIGKILL]
    assert result.cleanup.confirmed is True
    assert result.artifact["terminal"]["failure_code"] == "CANCELLED"
    assert result.artifact["attempt"]["source_attempt_count"] == 1
    assert _revalidate(result, cfg, forbidden) == result.artifact


class _DelayedCancellation:
    def __init__(self):
        self.deadline = None

    def is_set(self):
        if self.deadline is None:
            self.deadline = time.monotonic() + 0.3
            return False
        return time.monotonic() >= self.deadline


def test_real_process_fixture_kills_term_ignoring_child_and_descendant(tmp_path: Path):
    cfg = _config(tmp_path, timeout=5, grace=2)
    forbidden = _forbidden(tmp_path)
    result = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        command=_command("ignore-term"),
        cancellation_token=_DelayedCancellation(),
    )

    assert result.artifact["terminal"]["failure_code"] == "CANCELLED"
    assert result.cleanup.term_sent is True
    assert result.cleanup.kill_sent is True
    assert result.cleanup.leader_reaped is True
    assert result.cleanup.process_group_absent is True
    assert result.cleanup.confirmed is True
    assert _revalidate(result, cfg, forbidden) == result.artifact


def test_unconfirmed_cleanup_can_never_succeed(tmp_path: Path):
    cfg = _config(tmp_path, timeout=1, grace=1)
    forbidden = _forbidden(tmp_path)
    process = _FakeProcess()
    controller = _KillController(process, confirm=False)
    result = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        popen=lambda *args, **kwargs: process,
        killpg=controller,
        monotonic=_StepClock(),
        sleep=lambda _: None,
    )

    assert result.cleanup.confirmed is False
    assert result.artifact["terminal"] == {
        "status": "FAILED",
        "failure_code": "PROCESS_REAP_UNCONFIRMED",
    }
    assert result.artifact["capture"]["runner_set"] == []
    assert not (result.run_dir / "capture/odds.json").exists()
    assert _revalidate(result, cfg, forbidden) == result.artifact


def test_malformed_child_output_is_one_attempt_and_no_capture(tmp_path: Path):
    cfg = _config(tmp_path)
    forbidden = _forbidden(tmp_path)
    launches = 0

    def popen(*args, **kwargs):
        nonlocal launches
        launches += 1
        return subprocess.Popen(*args, **kwargs)

    result = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        command=_command("malformed"),
        popen=popen,
    )
    assert launches == 1
    assert result.artifact["terminal"]["failure_code"] == "SOURCE_MALFORMED"
    assert result.artifact["attempt"]["source_attempt_count"] == 1
    assert result.artifact["capture"]["runner_set"] == []
    assert _revalidate(result, cfg, forbidden) == result.artifact


def test_invalid_utf8_is_parsed_only_after_term_kill_group_cleanup(tmp_path: Path):
    cfg = _config(tmp_path, timeout=5, grace=2)
    forbidden = _forbidden(tmp_path)
    result = _execute(
        tmp_path,
        cfg=cfg,
        forbidden=forbidden,
        command=_command("invalid-bytes-descendant"),
    )

    assert result.artifact["terminal"]["failure_code"] == "SOURCE_MALFORMED"
    assert result.artifact["attempt"]["source_attempt_count"] == 1
    assert result.cleanup.term_sent is True
    assert result.cleanup.kill_sent is True
    assert result.cleanup.leader_reaped is True
    assert result.cleanup.process_group_absent is True
    assert result.cleanup.confirmed is True
    assert result.artifact["capture"]["runner_set"] == []
    assert _revalidate(result, cfg, forbidden) == result.artifact


class _ExplodingProcess(_FakeProcess):
    def communicate(self, timeout=None):
        raise RuntimeError("unexpected communicate failure")


def test_unexpected_post_launch_exception_attempts_group_cleanup_before_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import src.predictor.manual_independent_capture_executor as executor_module

    cfg = _config(tmp_path, timeout=5, grace=1)
    process = _ExplodingProcess()
    controller = _KillController(process)
    events = []
    real_flock = executor_module.fcntl.flock

    def tracked_flock(descriptor, operation):
        if operation == executor_module.fcntl.LOCK_UN:
            events.append("unlock")
        return real_flock(descriptor, operation)

    def tracked_killpg(pgid, signum):
        if signum != 0:
            events.append(f"signal:{signum}")
        return controller(pgid, signum)

    monkeypatch.setattr(executor_module.fcntl, "flock", tracked_flock)

    with pytest.raises(RuntimeError, match="unexpected communicate failure"):
        _execute(
            tmp_path,
            cfg=cfg,
            forbidden=_forbidden(tmp_path),
            popen=lambda *args, **kwargs: process,
            killpg=tracked_killpg,
            monotonic=_StepClock(step=0.1),
            sleep=lambda _: None,
        )

    assert controller.signals == [signal.SIGTERM, signal.SIGKILL]
    assert events[-1] == "unlock"
    assert events.index(f"signal:{signal.SIGTERM}") < events.index("unlock")
    assert events.index(f"signal:{signal.SIGKILL}") < events.index("unlock")
    assert process.poll() is not None


def test_insufficient_margin_rechecked_immediately_before_launch(tmp_path: Path):
    cfg = _config(tmp_path, minimum=120)
    forbidden = _forbidden(tmp_path)
    selected = _race(scheduled=NOW + timedelta(seconds=180))
    clock_values = iter(
        (NOW, NOW, NOW + timedelta(seconds=61), NOW + timedelta(seconds=61))
    )
    command_builds = []
    launches = []

    def command(launch: FixtureChildLaunch):
        command_builds.append(launch)
        return _command("success")(launch)

    result = execute_manual_capture_fixture(
        config=cfg,
        forbidden_paths=forbidden,
        requested_race_url=selected["url"],
        selected_race=selected,
        model_bytes=MODEL_BYTES,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        fixture_child_command=command,
        now=lambda: next(clock_values),
        popen=lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    assert len(command_builds) == 1
    assert launches == []
    assert result.artifact["terminal"]["failure_code"] == "INSUFFICIENT_PREJUMP_MARGIN"
    assert result.artifact["timing"]["readiness_prejump_margin_seconds"] == 119
    assert result.artifact["attempt"]["source_attempt_count"] == 0
    assert _revalidate(result, cfg, forbidden) == result.artifact


def test_non_exact_thedogs_url_is_blocked_without_child_launch(tmp_path: Path):
    cfg = _config(tmp_path)
    forbidden = _forbidden(tmp_path)
    launches = []
    result = execute_manual_capture_fixture(
        config=cfg,
        forbidden_paths=forbidden,
        requested_race_url="https://www.thedogs.com.au/racing/richmond/2026-08-05",
        selected_race=_race(),
        model_bytes=MODEL_BYTES,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        fixture_child_command=_command("success"),
        popen=lambda *args, **kwargs: launches.append((args, kwargs)),
        now=lambda: NOW,
    )
    assert launches == []
    assert result.artifact["terminal"]["failure_code"] == "EXACT_RACE_INVALID"
    assert result.artifact["request"]["selected_race"] is None
    assert _revalidate(result, cfg, forbidden) == result.artifact


def test_unsafe_operations_root_symlink_fails_before_child_launch(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "manual-operations"
    link.symlink_to(target, target_is_directory=True)
    alternate = tmp_path / "other"
    alternate.mkdir()
    cfg = _config(alternate)
    cfg["paths"]["operations_root"] = str(link)
    manual = link / "manual-independent-capture-v1"
    cfg["paths"].update(
        {
            "manual_root": str(manual),
            "browser_profile": str(manual / "browser-profile"),
            "runs_root": str(manual / "runs"),
            "manual_lock": str(manual / "manual-capture.lock"),
        }
    )
    with pytest.raises(ValueError, match="UNSAFE_PATH:operations_root"):
        execute_manual_capture_fixture(
            config=cfg,
            forbidden_paths=_forbidden(tmp_path),
            requested_race_url=_race()["url"],
            selected_race=_race(),
            model_bytes=MODEL_BYTES,
            source_commit=SOURCE_COMMIT,
            source_tree=SOURCE_TREE,
            fixture_child_command=_command("success"),
            popen=lambda *args, **kwargs: pytest.fail("unsafe path launched child"),
            now=lambda: NOW,
        )


def test_forbidden_apis_explode_and_autonomous_sentinel_is_byte_and_metadata_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import race_collection.synchronous_manual_capture as synchronous
    import scripts.autonomous_live_odds_capture as autonomous

    sentinel = tmp_path / "autonomous-shared.lock"
    sentinel.write_bytes(b"AUTONOMOUS-SENTINEL\n")
    before = sentinel.stat()
    before_bytes = sentinel.read_bytes()

    def forbidden_call(*args, **kwargs):
        raise AssertionError("forbidden canonical/autonomous API called")

    monkeypatch.setattr(synchronous, "run_capture_one", forbidden_call)
    monkeypatch.setattr(autonomous, "execute_capture_plan", forbidden_call)
    monkeypatch.setattr(autonomous, "append_validated_capture", forbidden_call)
    monkeypatch.setattr(sqlite3, "connect", forbidden_call)

    cfg = _config(tmp_path)
    forbidden = _forbidden(tmp_path, sentinel=sentinel)
    result = _execute(tmp_path, cfg=cfg, forbidden=forbidden)
    after = sentinel.stat()

    assert result.artifact["terminal"]["status"] == "CAPTURE_READY"
    assert sentinel.read_bytes() == before_bytes
    assert (after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    assert _revalidate(result, cfg, forbidden) == result.artifact
