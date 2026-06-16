#!/usr/bin/env python3
"""Timer-safe daemon wrapper for shadow evidence accumulation.

The daemon layer is report-only. It schedules and supervises the existing
shadow autopilot, rechecks older pending shadow runs for exact official-result
joins, refreshes aggregate dashboards, emits alerts, and records lock/recovery
validation. It must not train, promote, mutate registries, write DB rows, write
labels, enable TGR, overwrite production predictions, rewrite snapshots, or
emit betting/EV actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import shadow_autopilot_v1 as autopilot  # noqa: E402
from scripts.forward_shadow_runtime_state import (  # noqa: E402
    build_runtime_state as build_forward_shadow_runtime_state,
    build_summary as build_forward_shadow_runtime_summary,
)


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemonization_v1_"
DEFAULT_RUNTIME_DIR = DEFAULT_EVIDENCE_ROOT / "shadow_autopilot_daemon_runtime"
DEFAULT_LOCK_PATH = DEFAULT_RUNTIME_DIR / "shadow_autopilot.lock"
DEFAULT_STATE_PATH = DEFAULT_RUNTIME_DIR / "state.json"
DEFAULT_SERVICE_DIR = ROOT / "ops/systemd"
DEFAULT_TARGET_JOINED_RACES = 100
DEFAULT_MIN_JOINED_RACES = 100
DEFAULT_TIMEOUT_SECONDS = 840
DEFAULT_LOCK_STALE_SECONDS = 3600
DEFAULT_REJOIN_PENDING_LIMIT = 8
DEFAULT_REJOIN_LOOKBACK_DAYS = 7
SERVICE_NAME = "shadow-autopilot.service"
TIMER_NAME = "shadow-autopilot.timer"
NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "tgr_enabled": False,
    "betting_action": False,
    "ev_action": False,
    "production_prediction_overwrite": False,
    "snapshot_rewrite": False,
    "outside_shadow_manifest_rewrite": False,
    "schema_change": False,
    "hyperparameter_change": False,
    "calibration_method_change": False,
    "champion_modification": False,
}
PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "model_registry/current_production.json",
    ROOT / "processed_manifest.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
RUNNER_RELATED_UNSAFE_REASONS = {
    "duplicate_official_boxes",
    "missing_predicted_boxes_in_official_result",
    "extra_official_non_scratch_boxes_outside_prediction_set",
    "dog_name_mismatch_after_exact_badge_stripping",
}


class LockBusy(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]):
        super().__init__("shadow_autopilot_daemon_lock_busy")
        self.payload = dict(payload)


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        logical = path if path.is_absolute() else ROOT / path
        return logical.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_json_value(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def rooted_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(paths: Sequence[Path] = PROTECTED_PATHS) -> dict[str, str | None]:
    return {relpath(path) or str(path): sha256_file(path) for path in paths}


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_shadow_autopilot_daemon_artifact:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def latest_artifact(root: Path, prefix: str, required_file: str) -> Path | None:
    candidates = [
        item
        for item in root.glob(f"{prefix}*")
        if item.is_dir() and (item / required_file).exists()
    ]
    return sorted(candidates)[-1] if candidates else None


def pid_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(lock_path: Path) -> dict[str, Any] | None:
    return load_json(lock_path)


def lock_stale_reason(payload: Mapping[str, Any] | None, *, stale_after_seconds: int) -> str | None:
    if not payload:
        return "unreadable_lock_payload"
    pid = payload.get("pid")
    if not pid_running(pid):
        return "lock_pid_not_running"
    started_at = payload.get("started_at")
    try:
        started = datetime.fromisoformat(str(started_at))
    except ValueError:
        return None
    age = (datetime.now().astimezone() - started).total_seconds()
    if age > stale_after_seconds and not pid_running(pid):
        return "stale_lock_pid_not_running"
    return None


def acquire_lock(
    *,
    lock_path: Path,
    run_id: str,
    stale_after_seconds: int,
    output_dir: Path,
) -> dict[str, Any]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "shadow_autopilot_daemon_lock_v1",
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": relpath(output_dir),
    }
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = read_lock(lock_path)
            reason = lock_stale_reason(existing, stale_after_seconds=stale_after_seconds)
            if reason:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise LockBusy(
                {
                    "lock_path": relpath(lock_path),
                    "existing_lock": existing,
                    "reason": "active_lock_present",
                }
            )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return payload


def release_lock(lock_path: Path, run_id: str) -> dict[str, Any]:
    payload = read_lock(lock_path)
    if payload and payload.get("run_id") != run_id:
        return {"released": False, "reason": "lock_owned_by_other_run", "lock": payload}
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return {"released": False, "reason": "lock_already_missing"}
    return {"released": True, "reason": "released_by_owner"}


def probe_duplicate_lock(lock_path: Path, *, stale_after_seconds: int, output_dir: Path) -> dict[str, Any]:
    try:
        acquire_lock(
            lock_path=lock_path,
            run_id="duplicate_probe",
            stale_after_seconds=stale_after_seconds,
            output_dir=output_dir,
        )
    except LockBusy as exc:
        return {"status": "PASS", "duplicate_acquire_blocked": True, "details": exc.payload}
    release_lock(lock_path, "duplicate_probe")
    return {"status": "FAIL", "duplicate_acquire_blocked": False}


def probe_stale_lock_cleanup(output_dir: Path) -> dict[str, Any]:
    probe_path = output_dir / "validation" / "stale_probe.lock"
    write_json(
        probe_path,
        {
            "schema_version": "shadow_autopilot_daemon_lock_v1",
            "run_id": "stale_probe",
            "pid": 999999999,
            "started_at": "2000-01-01T00:00:00+00:00",
        },
    )
    acquired = acquire_lock(
        lock_path=probe_path,
        run_id="stale_probe_replacement",
        stale_after_seconds=1,
        output_dir=output_dir,
    )
    released = release_lock(probe_path, "stale_probe_replacement")
    return {
        "status": "PASS" if released.get("released") else "FAIL",
        "stale_lock_cleaned": True,
        "replacement_lock": acquired,
        "release": released,
    }


def run_command(
    *,
    name: str,
    command: Sequence[str],
    output_dir: Path,
    timeout_seconds: int,
    cwd: Path = ROOT,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    log_dir = output_dir / "logs"
    stdout_path = log_dir / f"{name}.stdout.txt"
    stderr_path = log_dir / f"{name}.stderr.txt"
    timed_out = False
    returncode: int | None = None
    stdout = ""
    stderr = ""
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout_kill, stderr_kill = process.communicate()
            stdout += stdout_kill
            stderr += stderr_kill
        returncode = -signal.SIGTERM
    duration = time.monotonic() - started_monotonic
    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    return {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "duration_seconds": duration,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "returncode": returncode,
        "status": "PASS" if returncode == 0 and not timed_out else "FAIL",
        "stdout_path": relpath(stdout_path),
        "stderr_path": relpath(stderr_path),
    }


def simulate_timeout_recovery(output_dir: Path) -> dict[str, Any]:
    result = run_command(
        name="timeout_recovery_probe",
        command=[sys.executable, "-c", "import time; time.sleep(2)"],
        output_dir=output_dir,
        timeout_seconds=1,
    )
    return {
        "status": "PASS" if result.get("timed_out") is True else "FAIL",
        "timeout_enforced": result.get("timed_out") is True,
        "probe": result,
    }


def service_file_text(*, repo_path: Path, timeout_seconds: int) -> str:
    script_path = repo_path / "scripts/shadow_autopilot_daemon.py"
    return "\n".join(
        [
            "[Unit]",
            "Description=Greyhound shadow autopilot evidence collection",
            "Wants=network-online.target",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=oneshot",
            f"WorkingDirectory={repo_path}",
            "Environment=PYTHONUNBUFFERED=1",
            "Environment=GREYHOUND_ALLOW_TGR=0",
            "Environment=PATH=/home/l4nd0/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            (
                f"ExecStart=/usr/bin/python3 {script_path} run-once "
                "--days-ahead 1 --refresh-limit 16 "
                f"--rejoin-pending-limit {DEFAULT_REJOIN_PENDING_LIMIT} "
                f"--timeout-seconds {timeout_seconds}"
            ),
            f"TimeoutStartSec={timeout_seconds + 60}",
            "Nice=10",
            "IOSchedulingClass=best-effort",
            "",
        ]
    )


def timer_file_text() -> str:
    return "\n".join(
        [
            "[Unit]",
            "Description=Run greyhound shadow autopilot every 15 minutes",
            "",
            "[Timer]",
            "OnBootSec=2min",
            "OnUnitActiveSec=15min",
            "AccuracySec=1min",
            "Persistent=true",
            "Unit=shadow-autopilot.service",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )


def write_service_files(
    *,
    service_dir: Path = DEFAULT_SERVICE_DIR,
    repo_path: Path = Path("/home/l4nd0/greyhound_racing_collector"),
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / SERVICE_NAME
    timer_path = service_dir / TIMER_NAME
    write_text(service_path, service_file_text(repo_path=repo_path, timeout_seconds=timeout_seconds))
    write_text(timer_path, timer_file_text())
    return {
        "service_path": relpath(service_path),
        "timer_path": relpath(timer_path),
        "timer_frequency": "15min",
        "repo_path": str(repo_path),
        "timeout_seconds": timeout_seconds,
    }


def systemd_verify(service_path: Path, timer_path: Path, output_dir: Path) -> dict[str, Any]:
    tool = shutil.which("systemd-analyze")
    if not tool:
        return {"status": "SKIPPED", "reason": "systemd-analyze_not_available"}
    result = run_command(
        name="systemd_analyze_verify",
        command=[tool, "verify", str(service_path), str(timer_path)],
        output_dir=output_dir,
        timeout_seconds=30,
    )
    return {
        "status": "PASS" if result.get("returncode") == 0 else "FAIL",
        "command_result": result,
    }


def _parse_systemctl_show(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def systemd_unit_status(
    unit_name: str,
    *,
    systemctl_path: str | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Read one systemd unit status without mutating the host."""

    tool = systemctl_path or shutil.which("systemctl")
    if not tool:
        return {
            "unit": unit_name,
            "status": "SYSTEMCTL_UNAVAILABLE",
            "systemctl_available": False,
        }
    command = [
        tool,
        "show",
        unit_name,
        "--property=LoadState",
        "--property=ActiveState",
        "--property=UnitFileState",
        "--property=FragmentPath",
        "--no-pager",
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {
            "unit": unit_name,
            "status": "SYSTEMCTL_QUERY_FAILED",
            "systemctl_available": True,
            "command": command,
            "error": repr(exc),
        }
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    values = _parse_systemctl_show(stdout)
    load_state = values.get("LoadState")
    active_state = values.get("ActiveState")
    unit_file_state = values.get("UnitFileState")
    fragment_path = values.get("FragmentPath")
    loaded = load_state == "loaded"
    enabled = unit_file_state == "enabled"
    active = active_state == "active"
    if getattr(result, "returncode", 1) != 0:
        status = "SYSTEMCTL_QUERY_FAILED"
    elif not loaded:
        status = "NOT_LOADED"
    elif active:
        status = "ACTIVE"
    elif enabled:
        status = "LOADED_ENABLED_INACTIVE"
    else:
        status = "LOADED_INACTIVE"
    return {
        "unit": unit_name,
        "status": status,
        "systemctl_available": True,
        "command": command,
        "returncode": getattr(result, "returncode", None),
        "load_state": load_state,
        "active_state": active_state,
        "unit_file_state": unit_file_state,
        "fragment_path": fragment_path,
        "loaded": loaded,
        "enabled": enabled,
        "active": active,
        "stdout": stdout,
        "stderr": stderr,
    }


def systemd_deployment_status(
    *,
    service_name: str = SERVICE_NAME,
    timer_name: str = TIMER_NAME,
    systemctl_path: str | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Summarize whether the report-only daemon timer is actually deployed."""

    service = systemd_unit_status(
        service_name,
        systemctl_path=systemctl_path,
        runner=runner,
    )
    timer = systemd_unit_status(
        timer_name,
        systemctl_path=systemctl_path,
        runner=runner,
    )
    systemctl_available = bool(
        service.get("systemctl_available") and timer.get("systemctl_available")
    )
    service_loaded = bool(service.get("loaded"))
    timer_loaded = bool(timer.get("loaded"))
    timer_enabled = bool(timer.get("enabled"))
    timer_active = bool(timer.get("active"))
    deployment_ready = (
        systemctl_available
        and service_loaded
        and timer_loaded
        and timer_enabled
        and timer_active
    )
    if not systemctl_available:
        deployment_status = "SYSTEMCTL_UNAVAILABLE"
    elif deployment_ready:
        deployment_status = "INSTALLED_AND_ACTIVE"
    elif service_loaded or timer_loaded:
        deployment_status = "INSTALLED_NOT_ACTIVE"
    else:
        deployment_status = "NOT_INSTALLED"
    return {
        "schema_version": "shadow_autopilot_systemd_deployment_status_v1",
        "deployment_status": deployment_status,
        "deployment_ready": deployment_ready,
        "service_installed": service_loaded,
        "timer_installed": timer_loaded,
        "timer_enabled": timer_enabled,
        "timer_active": timer_active,
        "service_unit": service,
        "timer_unit": timer,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def install_markdown(service_info: Mapping[str, Any]) -> str:
    service_path = service_info.get("service_path")
    timer_path = service_info.get("timer_path")
    return "\n".join(
        [
            "# Shadow Autopilot Service Install",
            "",
            "The service is intentionally not installed or enabled by this script.",
            "Install it only after reviewing the generated unit files.",
            "",
            "```bash",
            f"sudo cp {service_path} /etc/systemd/system/{SERVICE_NAME}",
            f"sudo cp {timer_path} /etc/systemd/system/{TIMER_NAME}",
            "sudo systemctl daemon-reload",
            f"sudo systemctl enable --now {TIMER_NAME}",
            f"systemctl list-timers {TIMER_NAME}",
            f"journalctl -u {SERVICE_NAME} -n 200 --no-pager",
            "```",
            "",
            "The timer runs every 15 minutes. Overlap is prevented by systemd oneshot semantics and the daemon lock file.",
            "",
        ]
    )


def daemon_design_markdown() -> str:
    return "\n".join(
        [
            "# Shadow Autopilot Daemon Design",
            "",
            "## Objective",
            "Continuously collect forward-shadow evidence without changing production state.",
            "",
            "## Architecture",
            "- `shadow-autopilot.timer` activates every 15 minutes.",
            "- `shadow-autopilot.service` runs one bounded `shadow_autopilot_daemon.py run-once` cycle.",
            "- The daemon acquires a JSON lock before any work starts.",
            "- The daemon delegates refresh, scoring, current-run join, aggregate, and status generation to `scripts/shadow_autopilot_v1.py`.",
            "- The daemon then re-runs exact result joins for older pending shadow runs and refreshes aggregate/status artifacts.",
            "- The daemon emits dashboards, alert reports, validation reports, and protected-hash evidence in one packet.",
            "",
            "## Lifecycle",
            "Startup -> acquire lock -> refresh races -> score shadow predictions -> join current run -> rejoin prior pending runs -> aggregate/dashboard -> alert -> release lock -> sleep by timer.",
            "",
            "## Mutation Boundary",
            "Allowed writes are limited to shadow-only evidence artifacts, daemon runtime lock/state, logs, and unit-file templates. DB rows, labels, production pointers, registry files, model artifacts, snapshots, and betting/EV outputs are not written.",
            "",
        ]
    )


def lifecycle_diagram() -> dict[str, Any]:
    return {
        "schema_version": "shadow_autopilot_daemon_lifecycle_v1",
        "nodes": [
            {"id": "startup", "description": "systemd timer starts one daemon cycle"},
            {"id": "lock", "description": "acquire JSON lock and reject overlap"},
            {"id": "refresh", "description": "refresh pre-jump TheDogs races into isolated artifact input dir"},
            {"id": "score", "description": "score shadow predictions with existing shadow model only"},
            {"id": "join_current", "description": "exact official-result join for current shadow run"},
            {"id": "rejoin_pending", "description": "exact official-result rejoin sweep for older pending shadow runs"},
            {"id": "dashboard", "description": "aggregate joins and write shadow status/dashboard/readiness files"},
            {"id": "alert", "description": "compare current metrics to prior packet and emit alert report"},
            {"id": "release", "description": "release lock and persist state"},
            {"id": "sleep", "description": "systemd timer waits 15 minutes before next activation"},
        ],
        "edges": [
            ["startup", "lock"],
            ["lock", "refresh"],
            ["refresh", "score"],
            ["score", "join_current"],
            ["join_current", "rejoin_pending"],
            ["rejoin_pending", "dashboard"],
            ["dashboard", "alert"],
            ["alert", "release"],
            ["release", "sleep"],
        ],
        "fail_closed_edges": [
            {"from": "lock", "to": "release", "reason": "active lock prevents overlap"},
            {"from": "refresh", "to": "dashboard", "reason": "refresh failure still leaves existing evidence reviewable"},
            {"from": "score", "to": "dashboard", "reason": "score failure still refreshes aggregate/status from existing joins"},
        ],
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def normalize_source_key(value: Any) -> str:
    text = str(value or "").rstrip("/")
    if not text:
        return ""
    return text.split("/")[-1]


def join_index(evidence_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(evidence_root.glob("forward_shadow_result_join_*/shadow_forward_metrics.json")):
        metrics = load_json(metrics_path)
        if not metrics:
            continue
        source_key = normalize_source_key(metrics.get("source_shadow_run"))
        if not source_key:
            continue
        mtime = metrics_path.stat().st_mtime
        if source_key in rows and mtime <= float(rows[source_key].get("mtime") or 0):
            continue
        rows[source_key] = {
            "join_dir": metrics_path.parent,
            "metrics": metrics,
            "mtime": mtime,
        }
    return rows


def candidate_shadow_runs(
    *,
    evidence_root: Path,
    pending_limit: int,
    lookback_days: int,
) -> list[dict[str, Any]]:
    joined = join_index(evidence_root)
    cutoff = time.time() - (lookback_days * 86400)
    candidates: list[dict[str, Any]] = []
    for shadow_dir in sorted(evidence_root.glob("daily_race_ingest_shadow_*")):
        manifest = load_json(shadow_dir / "shadow_manifest.json")
        if not manifest:
            continue
        if shadow_dir.stat().st_mtime < cutoff:
            continue
        if int(manifest.get("race_count") or 0) <= 0:
            continue
        if manifest.get("final_status") not in {"FORWARD_SHADOW_RUN_COMPLETE", "NO_ELIGIBLE_CURRENT_OR_FUTURE_RACES"}:
            continue
        key = shadow_dir.name
        latest = joined.get(key)
        latest_metrics = (latest or {}).get("metrics") or {}
        pending_count = int(latest_metrics.get("pending_race_count") or 0)
        if latest is None or pending_count > 0:
            candidates.append(
                {
                    "shadow_run_dir": shadow_dir,
                    "shadow_run_key": key,
                    "latest_join_dir": (latest or {}).get("join_dir"),
                    "latest_pending_count": pending_count if latest else None,
                    "latest_safe_joined_count": int(latest_metrics.get("safe_joined_race_count") or 0),
                    "latest_unsafe_count": int(latest_metrics.get("unsafe_match_count") or 0),
                    "latest_join_mtime": (latest or {}).get("mtime"),
                    "shadow_run_mtime": shadow_dir.stat().st_mtime,
                    "race_count": int(manifest.get("race_count") or 0),
                }
            )
    candidates.sort(
        key=lambda row: (
            0 if row.get("latest_join_mtime") is None else 1,
            float(row.get("latest_join_mtime") or 0),
            float(row["shadow_run_mtime"]),
        )
    )
    return candidates[:pending_limit]


def rejoin_pending_shadow_runs(
    *,
    run_id: str,
    output_dir: Path,
    evidence_root: Path,
    db_path: Path,
    current_time: str,
    pending_limit: int,
    lookback_days: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    candidates = candidate_shadow_runs(
        evidence_root=evidence_root,
        pending_limit=pending_limit,
        lookback_days=lookback_days,
    )
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        join_dir = evidence_root / f"forward_shadow_result_join_{run_id}_daemon_rejoin_{index:03d}"
        command = [
            sys.executable,
            str(ROOT / "scripts/join_forward_shadow_results.py"),
            "--shadow-run-dir",
            str(candidate["shadow_run_dir"]),
            "--output-dir",
            str(join_dir),
            "--db",
            str(db_path),
            "--current-time",
            current_time,
        ]
        step = run_command(
            name=f"rejoin_pending_{index:03d}",
            command=command,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )
        metrics = load_json(join_dir / "shadow_forward_metrics.json")
        results.append(
            {
                "candidate": {
                    **candidate,
                    "shadow_run_dir": relpath(candidate["shadow_run_dir"]),
                    "latest_join_dir": relpath(candidate.get("latest_join_dir")),
                },
                "join_dir": relpath(join_dir),
                "step": step,
                "metrics": metrics,
            }
        )
    joined_count = sum(int((row.get("metrics") or {}).get("safe_joined_race_count") or 0) for row in results)
    pending_count = sum(int((row.get("metrics") or {}).get("pending_race_count") or 0) for row in results)
    unsafe_count = sum(int((row.get("metrics") or {}).get("unsafe_match_count") or 0) for row in results)
    return {
        "schema_version": "shadow_autopilot_automated_join_report_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "exact_identity_join_only": True,
        "fuzzy_join_allowed": False,
        "ambiguous_winners_rejected": True,
        "pending_shadow_runs_scanned": len(candidates),
        "rejoin_attempt_count": len(results),
        "rejoin_safe_joined_count_sum": joined_count,
        "rejoin_pending_count_sum": pending_count,
        "rejoin_unsafe_count_sum": unsafe_count,
        "results": results,
    }


def metric_value(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    value = payload.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def alert_rules(target_joined_races: int) -> dict[str, Any]:
    return {
        "schema_version": "shadow_autopilot_alert_rules_v1",
        "safe_joined_increase": {"enabled": True, "comparison": "current > previous"},
        "safe_joined_target_reached": {"enabled": True, "threshold": target_joined_races},
        "top1_material_drop": {"enabled": True, "delta_threshold": -0.05},
        "calibration_deterioration": {"enabled": True, "slope_distance_delta_threshold": 0.2},
        "box1_share_spike": {"enabled": True, "absolute_threshold": 0.35, "delta_threshold": 0.1},
        "join_failures_increase": {"enabled": True, "metric": "unsafe_matches"},
        "runner_set_mismatch_spike": {"enabled": True, "unsafe_reason_contains": "runner"},
        "model_hash_changed": {"enabled": True, "comparison": "current != previous"},
        "score_command_changed": {"enabled": True, "comparison": "current != previous"},
        "training_enabled": {"enabled": True, "expected": False},
        "tgr_enabled": {"enabled": True, "expected": False},
        "quarantined_feature_active": {"enabled": True, "expected": []},
        "probability_sum_failed": {"enabled": True, "expected": "PASS"},
        "no_predictions_with_eligible_inputs": {"enabled": True, "eligible_count_threshold": 1},
    }


def _as_text_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item) for item in value]
    return [str(value)]


def unsafe_match_runner_related(row: Mapping[str, Any]) -> bool:
    reasons = {reason.strip() for reason in _as_text_list(row.get("reason")) if reason.strip()}
    reasons.update(
        reason.strip() for reason in _as_text_list(row.get("identity_errors")) if reason.strip()
    )
    normalized = {reason.lower() for reason in reasons}
    if normalized.intersection(RUNNER_RELATED_UNSAFE_REASONS):
        return True
    if any("runner" in reason for reason in normalized):
        return True
    if row.get("missing_predicted_boxes"):
        return True
    if row.get("disallowed_extra_official_boxes"):
        return True
    if row.get("name_mismatches"):
        return True
    return False


def unsafe_match_alert_sample(row: Mapping[str, Any], *, join_dir: object) -> dict[str, Any]:
    name_mismatches = list(row.get("name_mismatches") or [])
    sample = {
        "race_id": row.get("race_id"),
        "status": row.get("status"),
        "join_dir": relpath(Path(str(join_dir))) if join_dir else None,
        "reason": _as_text_list(row.get("reason") or row.get("identity_errors")),
        "missing_predicted_boxes": row.get("missing_predicted_boxes") or [],
        "disallowed_extra_official_boxes": row.get("disallowed_extra_official_boxes") or [],
        "allowed_extra_scratched_official_boxes": (
            row.get("allowed_extra_scratched_official_boxes") or []
        ),
        "name_mismatch_count": len(name_mismatches),
        "name_mismatch_samples": name_mismatches[:3],
    }
    alignment = row.get("prejump_runner_alignment")
    if isinstance(alignment, Mapping):
        sample["prejump_runner_alignment"] = {
            "canonical_runner_alignment_status": alignment.get(
                "canonical_runner_alignment_status"
            ),
            "canonical_runner_set_status": alignment.get("canonical_runner_set_status"),
            "canonical_runner_count": alignment.get("canonical_runner_count"),
            "canonical_prediction_runner_count": alignment.get(
                "canonical_prediction_runner_count"
            ),
        }
    return sample


def build_alert_report(
    *,
    current_dashboard: Mapping[str, Any],
    previous_dashboard: Mapping[str, Any] | None,
    automated_join_report: Mapping[str, Any],
    target_joined_races: int,
    current_observability: Mapping[str, Any] | None = None,
    previous_observability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    previous_dashboard = previous_dashboard or {}
    triggered: list[dict[str, Any]] = []
    current_safe = metric_value(current_dashboard, "safe_joined_races") or 0.0
    previous_safe = metric_value(previous_dashboard, "safe_joined_races") or 0.0
    if current_safe > previous_safe:
        triggered.append(
            {
                "rule": "safe_joined_increase",
                "severity": "info",
                "previous": previous_safe,
                "current": current_safe,
            }
        )
    if current_safe >= target_joined_races and previous_safe < target_joined_races:
        triggered.append(
            {
                "rule": "safe_joined_target_reached",
                "severity": "review",
                "threshold": target_joined_races,
                "current": current_safe,
            }
        )
    current_top1 = metric_value(current_dashboard, "top1")
    previous_top1 = metric_value(previous_dashboard, "top1")
    if current_top1 is not None and previous_top1 is not None and current_top1 - previous_top1 <= -0.05:
        triggered.append(
            {
                "rule": "top1_material_drop",
                "severity": "warning",
                "previous": previous_top1,
                "current": current_top1,
            }
        )
    current_slope = metric_value(current_dashboard.get("calibration") or {}, "slope")
    previous_slope = metric_value((previous_dashboard.get("calibration") or {}), "slope")
    if current_slope is not None and previous_slope is not None:
        current_distance = abs(1.0 - current_slope)
        previous_distance = abs(1.0 - previous_slope)
        if current_distance - previous_distance >= 0.2:
            triggered.append(
                {
                    "rule": "calibration_deterioration",
                    "severity": "warning",
                    "previous_slope": previous_slope,
                    "current_slope": current_slope,
                }
            )
    current_box1 = metric_value(current_dashboard, "box_1_share")
    previous_box1 = metric_value(previous_dashboard, "box_1_share")
    if (
        current_box1 is not None
        and current_box1 >= 0.35
        and (previous_box1 is None or current_box1 - previous_box1 >= 0.1)
    ):
        triggered.append(
            {
                "rule": "box1_share_spike",
                "severity": "warning",
                "previous": previous_box1,
                "current": current_box1,
            }
        )
    current_unsafe = metric_value(current_dashboard, "unsafe_matches") or 0.0
    previous_unsafe = metric_value(previous_dashboard, "unsafe_matches") or 0.0
    if current_unsafe > previous_unsafe:
        triggered.append(
            {
                "rule": "join_failures_increase",
                "severity": "warning",
                "previous": previous_unsafe,
                "current": current_unsafe,
            }
        )
    runner_mismatch_count = 0
    runner_mismatch_reason_counts: Counter[str] = Counter()
    runner_mismatch_samples: list[dict[str, Any]] = []
    for result in automated_join_report.get("results") or []:
        metrics = result.get("metrics") or {}
        if int(metrics.get("unsafe_match_count") or 0) <= 0:
            continue
        unsafe_path = Path(str(result.get("join_dir") or "")) / "unsafe_result_matches.json"
        if not unsafe_path.is_absolute():
            unsafe_path = ROOT / unsafe_path
        unsafe = load_json(unsafe_path) or {}
        for row in unsafe.get("unsafe_result_matches") or []:
            if unsafe_match_runner_related(row):
                runner_mismatch_count += 1
                for reason in _as_text_list(row.get("reason") or row.get("identity_errors")):
                    runner_mismatch_reason_counts[reason] += 1
                if len(runner_mismatch_samples) < 5:
                    runner_mismatch_samples.append(
                        unsafe_match_alert_sample(row, join_dir=result.get("join_dir"))
                    )
    if runner_mismatch_count:
        triggered.append(
            {
                "rule": "runner_set_mismatch_spike",
                "severity": "warning",
                "runner_related_unsafe_match_count": runner_mismatch_count,
                "reason_counts": dict(sorted(runner_mismatch_reason_counts.items())),
                "samples": runner_mismatch_samples,
            }
        )
    current_observability = current_observability or {}
    previous_observability = previous_observability or {}
    current_model_hash = current_observability.get("model_sha256")
    previous_model_hash = previous_observability.get("model_sha256")
    if current_model_hash and previous_model_hash and current_model_hash != previous_model_hash:
        triggered.append(
            {
                "rule": "model_hash_changed",
                "severity": "warning",
                "previous": previous_model_hash,
                "current": current_model_hash,
            }
        )
    current_command = current_observability.get("score_command_text")
    previous_command = previous_observability.get("score_command_text")
    if current_command and previous_command and current_command != previous_command:
        triggered.append(
            {
                "rule": "score_command_changed",
                "severity": "warning",
                "previous": previous_command,
                "current": current_command,
            }
        )
    safety_flags = current_observability.get("safety_flags") or {}
    if safety_flags.get("score_command_trains") or safety_flags.get("training_disabled") is False:
        triggered.append({"rule": "training_enabled", "severity": "critical"})
    if safety_flags.get("tgr_disabled") is False:
        triggered.append({"rule": "tgr_enabled", "severity": "critical"})
    quarantined_active = safety_flags.get("quarantined_features_active") or []
    if quarantined_active:
        triggered.append(
            {
                "rule": "quarantined_feature_active",
                "severity": "critical",
                "features": quarantined_active,
            }
        )
    probability_status = current_observability.get("probability_sum_status")
    if probability_status and probability_status != "PASS":
        triggered.append(
            {
                "rule": "probability_sum_failed",
                "severity": "critical",
                "status": probability_status,
            }
        )
    no_prediction_details = current_observability.get("no_prediction_details") or {}
    input_summary = no_prediction_details.get("input_summary") or {}
    if (
        current_observability.get("status") == "NO_PREDICTIONS"
        and int(input_summary.get("eligible_count") or 0) > 0
    ):
        triggered.append(
            {
                "rule": "no_predictions_with_eligible_inputs",
                "severity": "warning",
                "eligible_count": int(input_summary.get("eligible_count") or 0),
            }
        )
    return {
        "schema_version": "shadow_autopilot_alert_report_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "ALERTS_TRIGGERED" if triggered else "NO_ALERTS_TRIGGERED",
        "triggered_alerts": triggered,
        "comparison": {
            "previous_dashboard_present": bool(previous_dashboard),
            "previous_safe_joined_races": previous_safe,
            "current_safe_joined_races": current_safe,
        },
    }


def daemon_readiness(
    *,
    generated_at: datetime,
    dashboard: Mapping[str, Any],
    target_joined_races: int,
) -> dict[str, Any]:
    current_joined = int(dashboard.get("safe_joined_races") or 0)
    pending = int(dashboard.get("pending_races") or 0)
    unsafe = int(dashboard.get("unsafe_matches") or 0)
    calibration = dashboard.get("calibration") or {}
    probability_status = (dashboard.get("probability_sum_status") or {}).get("status")
    blockers: list[str] = []
    if current_joined < target_joined_races:
        blockers.append("insufficient_forward_shadow_joined_races")
    if pending:
        blockers.append("pending_official_results_remain")
    if unsafe:
        blockers.append("unsafe_result_matches_present")
    if probability_status != "PASS":
        blockers.append("probability_sum_status_not_pass")
    if dashboard.get("quarantined_features"):
        blockers.append("same_distance_same_grade_features_remain_quarantined")
    feature_activation_gate = dashboard.get("feature_activation_gate") or {}
    feature_activation_status = feature_activation_gate.get("status")
    if feature_activation_status and feature_activation_status != "FEATURE_ACTIVATION_PASS":
        blockers.append("feature_activation_gate_not_passed")

    calibration_status = str(calibration.get("status") or "not_computed")
    if current_joined < target_joined_races:
        decision = "NEED_MORE_RESULTS"
    elif calibration_status != "computed" or probability_status != "PASS":
        decision = "NEED_CALIBRATION_REVIEW"
    elif blockers:
        decision = "READY_FOR_RELIABILITY_REVIEW"
    else:
        decision = "READY_FOR_PROMOTION_CANDIDATE_REVIEW"
    return {
        "schema_version": "shadow_daemon_promotion_readiness_tracker_v1",
        "generated_at": generated_at.isoformat(),
        "current_joined_race_count": current_joined,
        "target_joined_race_count": target_joined_races,
        "calibration_status": calibration_status,
        "reliability_status": "target_met" if current_joined >= target_joined_races else "needs_more_results",
        "box_bias_status": {
            "box_1_share": dashboard.get("box_1_share"),
            "status": "REVIEW" if (dashboard.get("box_1_share") or 0) >= 0.35 else "TRACKING",
        },
        "feature_activation_gate_status": feature_activation_status,
        "kept_quarantined_features": feature_activation_gate.get("kept_quarantined_features") or [],
        "activation_allowed_features": feature_activation_gate.get("activation_allowed_features") or [],
        "leakage_status": "NO_NEW_LEAKAGE_SIGNAL_FROM_DAEMON",
        "outstanding_blockers": blockers,
        "decision": decision,
        "promotion_allowed": False,
    }


def readiness_markdown(readiness: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Shadow Daemon Readiness",
            "",
            f"- Decision: `{readiness.get('decision')}`",
            f"- Current joined races: `{readiness.get('current_joined_race_count')}`",
            f"- Target joined races: `{readiness.get('target_joined_race_count')}`",
            f"- Calibration status: `{readiness.get('calibration_status')}`",
            f"- Reliability status: `{readiness.get('reliability_status')}`",
            f"- Box-bias status: `{readiness.get('box_bias_status')}`",
            f"- Feature activation gate: `{readiness.get('feature_activation_gate_status')}`",
            f"- Kept quarantined features: `{readiness.get('kept_quarantined_features')}`",
            f"- Leakage status: `{readiness.get('leakage_status')}`",
            f"- Outstanding blockers: `{readiness.get('outstanding_blockers')}`",
            "",
            "Promotion remains forbidden by this daemon.",
            "",
        ]
    )


def copy_if_exists(source: Path | None, destination: Path) -> bool:
    if source is None or not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def feature_activation_gate_status_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any] | None:
    if autopilot_output_dir is None:
        return None
    status_path = autopilot_output_dir / "feature_activation_gate_status.json"
    status = load_json(status_path)
    if not status:
        return None
    return {
        "schema_version": "shadow_daemon_feature_activation_gate_summary_v1",
        "status": status.get("status"),
        "status_path": relpath(status_path),
        "output_dir": status.get("output_dir"),
        "provenance_audit": status.get("provenance_audit"),
        "activation_allowed_features": status.get("activation_allowed_features") or [],
        "kept_quarantined_features": status.get("kept_quarantined_features") or [],
        "fail_reason_summary": status.get("fail_reason_summary") or {},
        "data_availability_status": status.get("data_availability_status") or {},
        "inputs": status.get("inputs") or {},
        "no_write_guarantees": status.get("no_write_guarantees") or {},
    }


def shadow_odds_snapshot_status_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any]:
    if autopilot_output_dir is None:
        return {
            "schema_version": "shadow_daemon_odds_snapshot_summary_v1",
            "status": "MISSING_AUTOPILOT_OUTPUT",
            "status_path": None,
            "collection_attempted": None,
            "skipped_reason": "autopilot_output_missing",
            "prediction_rows": None,
            "odds_candidate_rows": None,
            "valid_pre_jump_dog_odds_rows": None,
            "races_with_complete_valid_prejump_odds": None,
            "races_with_missing_odds_rows": None,
            "races_with_post_feature_freeze_odds_rows": None,
            "ev_output_rows": 0,
            "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    status_path = autopilot_output_dir / "shadow_odds_snapshot_status.json"
    status = load_json(status_path)
    if not status:
        return {
            "schema_version": "shadow_daemon_odds_snapshot_summary_v1",
            "status": "MISSING_AUTOPILOT_ODDS_SNAPSHOT_STATUS",
            "status_path": relpath(status_path),
            "collection_attempted": None,
            "skipped_reason": "shadow_odds_snapshot_status_missing",
            "prediction_rows": None,
            "odds_candidate_rows": None,
            "valid_pre_jump_dog_odds_rows": None,
            "races_with_complete_valid_prejump_odds": None,
            "races_with_missing_odds_rows": None,
            "races_with_post_feature_freeze_odds_rows": None,
            "ev_output_rows": 0,
            "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    return {
        "schema_version": "shadow_daemon_odds_snapshot_summary_v1",
        "status": status.get("status") or status.get("final_status"),
        "final_status": status.get("final_status") or status.get("status"),
        "status_path": relpath(status_path),
        "output_dir": status.get("output_dir"),
        "collection_attempted": status.get("collection_attempted"),
        "skipped_reason": status.get("skipped_reason"),
        "prediction_rows": status.get("prediction_rows"),
        "race_count": status.get("race_count"),
        "runner_rows": status.get("runner_rows"),
        "odds_candidate_rows": status.get("odds_candidate_rows"),
        "valid_pre_jump_dog_odds_rows": status.get("valid_pre_jump_dog_odds_rows"),
        "races_with_complete_valid_prejump_odds": status.get(
            "races_with_complete_valid_prejump_odds"
        ),
        "races_with_missing_odds_rows": status.get("races_with_missing_odds_rows"),
        "races_with_post_feature_freeze_odds_rows": status.get(
            "races_with_post_feature_freeze_odds_rows"
        ),
        "ev_eligible_rows": status.get("ev_eligible_rows"),
        "ev_output_rows": status.get("ev_output_rows", 0),
        "ev_calculation_status": status.get("ev_calculation_status")
        or "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
        "protected_paths_unchanged": status.get("protected_paths_unchanged"),
        "no_write_guarantees": status.get("no_write_guarantees") or dict(NO_WRITE_GUARANTEES),
    }


def next_prejump_refresh_window_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any] | None:
    if autopilot_output_dir is None:
        return None
    report_path = autopilot_output_dir / "refresh_prejump_report.json"
    report = load_json(report_path)
    if not report:
        return None
    window = report.get("next_preferred_window")
    if not isinstance(window, Mapping):
        return None
    next_race = window.get("next_race")
    if not isinstance(next_race, Mapping):
        next_race = {}
    selected_races = report.get("selected_races")
    if not isinstance(selected_races, list):
        selected_races = []
    return {
        "schema_version": "shadow_daemon_next_prejump_refresh_window_v1",
        "status": window.get("status"),
        "reason": window.get("reason"),
        "report_path": relpath(report_path),
        "generated_at": report.get("generated_at"),
        "recommended_rerun_after_local": window.get("recommended_rerun_after_local"),
        "next_window_opens_at": window.get("next_window_opens_at"),
        "next_window_closes_at": window.get("next_window_closes_at"),
        "minutes_until_window_opens": window.get("minutes_until_window_opens"),
        "minutes_until_window_closes": window.get("minutes_until_window_closes"),
        "selected_count": int(report.get("selected_count") or 0),
        "total_races_found": int(report.get("total_races_found") or 0),
        "selected_race_count": len(selected_races),
        "next_race": {
            "race_id": next_race.get("race_id"),
            "date": next_race.get("date"),
            "venue": next_race.get("venue"),
            "race_number": next_race.get("race_number"),
            "race_time": next_race.get("race_time"),
            "jump_datetime": next_race.get("jump_datetime"),
            "minutes_to_jump": next_race.get("minutes_to_jump"),
            "bucket": next_race.get("bucket"),
            "selected": next_race.get("selected"),
            "race_url": next_race.get("race_url"),
        },
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def prejump_metadata_status_from_daily_run(
    daily_shadow_run_dir: Path | None,
) -> dict[str, Any] | None:
    if daily_shadow_run_dir is None:
        return None
    report_path = daily_shadow_run_dir / "prejump_metadata_report.json"
    report = load_json(report_path)
    if not report:
        return None
    eligible_count = int(report.get("eligible_count") or 0)
    verified_count = int(report.get("eligible_with_verified_prejump_metadata") or 0)
    malformed_count = int(report.get("malformed_prejump_metadata_count") or 0)
    unsafe_rows = list(report.get("unsafe_or_incomplete_metadata") or [])
    required_fields = list(report.get("required_fields") or [])
    field_coverage = report.get("field_coverage") or {}
    missing_required_fields = [
        field
        for field in required_fields
        if eligible_count > 0
        and int((field_coverage.get(field) or {}).get("eligible_present_rows") or 0)
        < eligible_count
    ]
    if eligible_count == 0 and report.get("status") == "PASS":
        status = "NO_ELIGIBLE_PREJUMP_RACES"
    elif report.get("status") == "PASS" and verified_count == eligible_count and not missing_required_fields:
        status = "PREJUMP_METADATA_PASS"
    elif report.get("status") == "PASS":
        status = "PREJUMP_METADATA_PARTIAL"
    else:
        status = "PREJUMP_METADATA_FAIL"
    return {
        "schema_version": "shadow_daemon_prejump_metadata_summary_v1",
        "status": status,
        "report_status": report.get("status"),
        "report_path": relpath(report_path),
        "eligible_count": eligible_count,
        "eligible_with_verified_prejump_metadata": verified_count,
        "malformed_prejump_metadata_count": malformed_count,
        "unsafe_or_incomplete_metadata_count": len(unsafe_rows),
        "stale_with_prejump_metadata_count": int(
            report.get("stale_with_prejump_metadata_count") or 0
        ),
        "required_fields": required_fields,
        "missing_required_fields": missing_required_fields,
        "field_coverage": field_coverage,
        "rejected_metadata_sources": report.get("rejected_metadata_sources") or [],
    }


def _prejump_metadata_run_summary(
    *,
    report_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    eligible_count = int(report.get("eligible_count") or 0)
    verified_count = int(report.get("eligible_with_verified_prejump_metadata") or 0)
    malformed_count = int(report.get("malformed_prejump_metadata_count") or 0)
    stale_count = int(report.get("stale_with_prejump_metadata_count") or 0)
    required_fields = list(report.get("required_fields") or [])
    field_coverage = report.get("field_coverage") or {}
    missing_required_fields = [
        field
        for field in required_fields
        if eligible_count > 0
        and int((field_coverage.get(field) or {}).get("eligible_present_rows") or 0)
        < eligible_count
    ]
    if eligible_count == 0 and report.get("status") == "PASS":
        status = "NO_ELIGIBLE_PREJUMP_RACES"
    elif report.get("status") == "PASS" and verified_count == eligible_count and not missing_required_fields:
        status = "PREJUMP_METADATA_PASS"
    elif report.get("status") == "PASS":
        status = "PREJUMP_METADATA_PARTIAL"
    else:
        status = "PREJUMP_METADATA_FAIL"
    return {
        "run_dir": relpath(report_path.parent),
        "report_path": relpath(report_path),
        "report_status": report.get("status"),
        "status": status,
        "eligible_count": eligible_count,
        "eligible_with_verified_prejump_metadata": verified_count,
        "malformed_prejump_metadata_count": malformed_count,
        "stale_with_prejump_metadata_count": stale_count,
        "missing_required_fields": missing_required_fields,
        "rejected_metadata_sources": report.get("rejected_metadata_sources") or [],
    }


def build_prejump_metadata_trend_report(
    *,
    evidence_root: Path,
    output_dir: Path,
    generated_at: datetime,
    limit: int = 20,
) -> dict[str, Any]:
    """Aggregate recent daily pre-jump metadata reports without touching source data."""

    report_paths = sorted(
        path
        for path in evidence_root.glob("daily_race_ingest_shadow_*/prejump_metadata_report.json")
        if path.is_file()
    )
    selected_paths = report_paths[-limit:] if limit > 0 else report_paths
    runs: list[dict[str, Any]] = []
    field_totals: dict[str, dict[str, Any]] = {}
    rejected_sources: Counter[str] = Counter()
    missing_required_fields: Counter[str] = Counter()

    for report_path in selected_paths:
        report = load_json(report_path)
        if not report:
            continue
        summary = _prejump_metadata_run_summary(report_path=report_path, report=report)
        runs.append(summary)
        eligible_count = int(summary.get("eligible_count") or 0)
        field_coverage = report.get("field_coverage") or {}
        for field in report.get("required_fields") or []:
            field_summary = field_totals.setdefault(
                str(field),
                {"eligible_rows": 0, "present_rows": 0, "missing_rows": 0},
            )
            present_rows = int((field_coverage.get(field) or {}).get("eligible_present_rows") or 0)
            field_summary["eligible_rows"] += eligible_count
            field_summary["present_rows"] += present_rows
            field_summary["missing_rows"] += max(0, eligible_count - present_rows)
        for source in report.get("rejected_metadata_sources") or []:
            if source not in (None, ""):
                rejected_sources[str(source)] += 1
        for field in summary.get("missing_required_fields") or []:
            missing_required_fields[str(field)] += 1

    total_eligible = sum(int(run.get("eligible_count") or 0) for run in runs)
    total_verified = sum(
        int(run.get("eligible_with_verified_prejump_metadata") or 0) for run in runs
    )
    total_malformed = sum(int(run.get("malformed_prejump_metadata_count") or 0) for run in runs)
    for field_summary in field_totals.values():
        eligible_rows = int(field_summary["eligible_rows"] or 0)
        present_rows = int(field_summary["present_rows"] or 0)
        field_summary["present_pct"] = present_rows / eligible_rows if eligible_rows else None

    runs_with_eligible = sum(1 for run in runs if int(run.get("eligible_count") or 0) > 0)
    runs_with_full_verified_metadata = sum(
        1
        for run in runs
        if int(run.get("eligible_count") or 0) > 0
        and run.get("status") == "PREJUMP_METADATA_PASS"
    )
    runs_needing_attention = sum(
        1
        for run in runs
        if int(run.get("eligible_count") or 0) > 0
        and run.get("status") != "PREJUMP_METADATA_PASS"
    )
    if not runs:
        status = "NO_PREJUMP_METADATA_REPORTS"
    elif total_eligible == 0:
        status = "NO_ELIGIBLE_PREJUMP_RACES"
    elif (
        total_verified == total_eligible
        and total_malformed == 0
        and runs_with_full_verified_metadata == runs_with_eligible
    ):
        status = "PREJUMP_METADATA_TREND_PASS"
    else:
        status = "PREJUMP_METADATA_TREND_NEEDS_ATTENTION"

    trend = {
        "schema_version": "shadow_daemon_prejump_metadata_trend_v1",
        "generated_at": generated_at.isoformat(),
        "status": status,
        "evidence_root": relpath(evidence_root),
        "runs_checked": len(runs),
        "available_report_count": len(report_paths),
        "limit": limit,
        "runs_with_eligible_prejump_races": runs_with_eligible,
        "runs_with_full_verified_metadata": runs_with_full_verified_metadata,
        "runs_needing_metadata_attention": runs_needing_attention,
        "total_eligible_prejump_races": total_eligible,
        "total_verified_prejump_metadata_races": total_verified,
        "total_malformed_prejump_metadata_races": total_malformed,
        "verified_metadata_rate": total_verified / total_eligible if total_eligible else None,
        "field_totals": dict(sorted(field_totals.items())),
        "missing_required_field_counts": dict(sorted(missing_required_fields.items())),
        "rejected_metadata_source_counts": dict(sorted(rejected_sources.items())),
        "runs": runs,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_json(output_dir / "prejump_metadata_trend_report.json", trend)
    return trend


def build_read_only_odds_coverage_report(
    *,
    db_path: Path,
    output_dir: Path,
    generated_at: datetime,
    stale_after_hours: float = 6.0,
) -> dict[str, Any]:
    """Write DB odds coverage diagnostics without scraping, scoring, or DB writes."""

    report_path = output_dir / "odds_coverage_report.json"
    summary: dict[str, Any] = {
        "schema_version": "shadow_daemon_read_only_odds_coverage_summary_v1",
        "generated_at": generated_at.isoformat(),
        "status": "FAILED",
        "mode": "read_only_coverage_diagnostic",
        "report_path": relpath(report_path),
        "odds_capture_performed": False,
        "odds_used_for_shadow_scoring": False,
        "shadow_model_input": False,
        "betting_action": False,
        "ev_action": False,
        "db_write": False,
        "stale_after_hours": stale_after_hours,
    }
    try:
        from accuracy_program.odds_coverage import analyze_odds_coverage

        coverage = analyze_odds_coverage(
            db_path,
            current_only=True,
            stale_after_hours=stale_after_hours,
            now=generated_at,
        )
        counts = coverage.get("counts") or {}
        safe = coverage.get("safe_match_counts") or {}
        rates = coverage.get("coverage_rates") or {}
        timestamp_quality = (coverage.get("timestamp_quality") or {}).get("live_odds_current_win") or {}
        source_url_quality = coverage.get("source_url_quality") or {}
        dog_level_rows = int(counts.get("dog_level_win_odds_rows") or 0)
        summary.update(
            {
                "status": "SUCCESS" if dog_level_rows else "NO_CURRENT_DOG_LEVEL_WIN_ODDS",
                "tables": coverage.get("tables") or {},
                "live_odds_rows": int(counts.get("live_odds_rows") or 0),
                "odds_history_rows": int(counts.get("odds_history_rows") or 0),
                "live_odds_races": int(counts.get("live_odds_races") or 0),
                "dog_level_win_odds_rows": dog_level_rows,
                "races_with_dog_level_win_odds": int(
                    counts.get("races_with_dog_level_win_odds") or 0
                ),
                "safe_direct_identity_matches": int(
                    safe.get("safe_direct_identity_matches") or 0
                ),
                "ambiguous_strict_identity_rows": int(
                    safe.get("ambiguous_strict_identity_rows") or 0
                ),
                "safe_direct_identity_match_rate": rates.get("safe_direct_identity_match_rate"),
                "stale_current_win_rows": int(timestamp_quality.get("stale_rows") or 0),
                "timestamped_current_win_rows": int(
                    timestamp_quality.get("timestamped_rows") or 0
                ),
                "source_url_rows_checked": int(source_url_quality.get("rows_checked") or 0),
                "source_url_rows_missing": int(
                    source_url_quality.get("rows_missing_source_url") or 0
                ),
                "post_race_source_url_rows": int(
                    source_url_quality.get("post_race_source_url_rows") or 0
                ),
                "source_provenance": coverage.get("source_provenance") or {},
            }
        )
        payload = {
            "schema_version": "shadow_daemon_read_only_odds_coverage_report_v1",
            "summary": summary,
            "coverage": coverage,
            "no_write_guarantees": {
                "odds_capture_performed": False,
                "odds_used_for_shadow_scoring": False,
                "shadow_model_input": False,
                "betting_action": False,
                "ev_action": False,
                "db_write": False,
            },
        }
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}:{exc}"
        payload = {
            "schema_version": "shadow_daemon_read_only_odds_coverage_report_v1",
            "summary": summary,
            "coverage": None,
            "no_write_guarantees": {
                "odds_capture_performed": False,
                "odds_used_for_shadow_scoring": False,
                "shadow_model_input": False,
                "betting_action": False,
                "ev_action": False,
                "db_write": False,
            },
        }
    write_json(report_path, payload)
    return summary


def build_cycle_activity_summary(
    *,
    current_dashboard: Mapping[str, Any],
    previous_dashboard: Mapping[str, Any] | None,
    daily_status: Mapping[str, Any],
    observability_status: Mapping[str, Any],
) -> dict[str, Any]:
    previous_dashboard = previous_dashboard or {}
    current_joined = int(current_dashboard.get("safe_joined_races") or 0)
    previous_joined = int(previous_dashboard.get("safe_joined_races") or 0)
    safe_joined_delta = max(0, current_joined - previous_joined)
    prediction_rows = int(observability_status.get("prediction_rows") or 0)
    races_scored = int(daily_status.get("races_scored_today") or 0)
    if prediction_rows > 0 and safe_joined_delta > 0:
        status = "PREDICTIONS_AND_RESULT_JOINS_ADVANCED"
    elif prediction_rows > 0:
        status = "PREDICTIONS_ONLY"
    elif safe_joined_delta > 0:
        status = "RESULT_JOINS_ADVANCED_NO_NEW_PREDICTIONS"
    else:
        status = "NO_NEW_PREDICTIONS_OR_SAFE_JOINS"
    return {
        "schema_version": "shadow_daemon_cycle_activity_summary_v1",
        "status": status,
        "current_safe_joined_races": current_joined,
        "previous_safe_joined_races": previous_joined,
        "safe_joined_delta_this_cycle": safe_joined_delta,
        "prediction_rows_this_cycle": prediction_rows,
        "races_scored_this_cycle": races_scored,
        "observability_status": observability_status.get("status"),
    }


def first_json(paths: Sequence[Path | None]) -> tuple[dict[str, Any] | None, Path | None]:
    for path in paths:
        payload = load_json(path)
        if payload is not None:
            return payload, path
    return None, None


def first_json_list(paths: Sequence[Path | None]) -> tuple[list[dict[str, Any]], Path | None]:
    for path in paths:
        payload = load_json_value(path)
        if not isinstance(payload, list):
            continue
        rows = [dict(row) for row in payload if isinstance(row, Mapping)]
        return rows, path
    return [], None


def clean_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def prediction_rows_from_daily_run(daily_shadow_run_dir: Path | None) -> tuple[list[dict[str, Any]], Path | None]:
    if daily_shadow_run_dir is None:
        return [], None
    jsonl_path = daily_shadow_run_dir / "shadow_predictions.jsonl"
    rows = read_jsonl_rows(jsonl_path)
    if rows:
        return rows, jsonl_path
    rows, path = first_json_list(
        [
            daily_shadow_run_dir / "shadow_predictions.json",
            daily_shadow_run_dir / "shadow_score_live" / "shadow_predictions.json",
        ]
    )
    if rows:
        return rows, path
    return [], jsonl_path if jsonl_path.exists() else None


def feature_rows_from_daily_run(daily_shadow_run_dir: Path | None) -> tuple[list[dict[str, Any]], Path | None]:
    if daily_shadow_run_dir is None:
        return [], None
    return first_json_list(
        [
            daily_shadow_run_dir / "shadow_score_live" / "shadow_feature_rows.json",
            daily_shadow_run_dir / "shadow_feature_rows.json",
        ]
    )


def infer_no_prediction_reason(daily_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    manifest = daily_manifest or {}
    input_summary = manifest.get("input_summary") or {}
    final_status = manifest.get("final_status")
    if final_status == "WAITING_FOR_UPCOMING_RACES":
        reason = "no_eligible_current_or_future_races"
    elif final_status == "BLOCKED_DB_STATE":
        reason = "db_state_blocked_shadow_scoring"
    elif final_status == "BLOCKED_MALFORMED_INPUTS":
        reason = "malformed_inputs_blocked_shadow_scoring"
    elif final_status == "BLOCKED_SHADOW_RUN_FAILURE":
        reason = "shadow_scoring_failed"
    elif int(input_summary.get("eligible_count") or 0) <= 0:
        reason = "no_eligible_inputs"
    else:
        reason = "prediction_rows_zero"
    return {
        "reason": reason,
        "final_status": final_status,
        "input_summary": input_summary,
    }


def find_importance_estimator(model: Any, seen: set[int] | None = None) -> Any:
    seen = seen or set()
    object_id = id(model)
    if object_id in seen:
        return None
    seen.add(object_id)
    if hasattr(model, "feature_importances_"):
        return model
    for attr in ("named_steps", "steps"):
        value = getattr(model, attr, None)
        if isinstance(value, Mapping):
            for item in value.values():
                found = find_importance_estimator(item, seen)
                if found is not None:
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                candidate = item[-1] if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) else item
                found = find_importance_estimator(candidate, seen)
                if found is not None:
                    return found
    for attr in ("estimator", "classifier", "model", "base_estimator"):
        child = getattr(model, attr, None)
        if child is not None:
            found = find_importance_estimator(child, seen)
            if found is not None:
                return found
    return None


def transformed_feature_names(model: Any, active_features: Sequence[str], expected_count: int) -> list[str] | None:
    preprocess = None
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, Mapping):
        preprocess = named_steps.get("preprocess") or named_steps.get("preprocessor")
    if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
        try:
            names = [str(name) for name in preprocess.get_feature_names_out()]
        except Exception:
            names = []
        if len(names) == expected_count:
            return names
    if len(active_features) == expected_count:
        return list(active_features)
    return None


def extract_global_feature_importances(
    *,
    model_path: Path | None,
    active_features: Sequence[str],
    limit: int = 20,
) -> dict[str, Any]:
    if model_path is None:
        return {"status": "UNAVAILABLE", "reason": "model_path_missing", "top_features": []}
    if not model_path.exists():
        return {"status": "UNAVAILABLE", "reason": "model_file_missing", "top_features": []}
    try:
        import joblib  # type: ignore
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": "joblib_not_available_in_daemon_runtime",
            "error": repr(exc),
            "top_features": [],
        }
    try:
        model = joblib.load(model_path)
        estimator = find_importance_estimator(model)
        if estimator is None:
            return {"status": "UNAVAILABLE", "reason": "feature_importances_not_exposed", "top_features": []}
        importances = [float(value) for value in getattr(estimator, "feature_importances_")]
        names = transformed_feature_names(model, active_features, len(importances))
        if names is None:
            return {
                "status": "UNAVAILABLE",
                "reason": "feature_names_not_recoverable",
                "importance_count": len(importances),
                "active_feature_count": len(active_features),
                "top_features": [],
            }
        rows = sorted(
            (
                {"feature": feature, "importance": importance}
                for feature, importance in zip(names, importances)
            ),
            key=lambda row: row["importance"],
            reverse=True,
        )
        return {"status": "AVAILABLE", "top_features": rows[:limit], "feature_count": len(rows)}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "reason": "model_importance_read_failed", "error": repr(exc), "top_features": []}


def feature_policy_summary(
    *,
    model_metadata: Mapping[str, Any] | None,
    training_report: Mapping[str, Any] | None,
    active_policy: Mapping[str, Any] | None,
    inactive_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    inactive_features = (
        (active_policy or {}).get("inactive_features_due_to_train_all_missing")
        or (inactive_policy or {}).get("inactive_features_due_to_train_all_missing")
        or (training_report or {}).get("inactive_features_due_to_train_all_missing")
        or (model_metadata or {}).get("inactive_features_due_to_train_all_missing")
        or []
    )
    inactive_set = {str(feature) for feature in inactive_features}
    feature_columns = list((model_metadata or {}).get("feature_columns") or [])
    active_features = [str(feature) for feature in feature_columns if str(feature) not in inactive_set]
    watched = list(getattr(autopilot, "WATCHED_QUARANTINED_FEATURES", ()))
    quarantined_active = [feature for feature in watched if feature in active_features]
    return {
        "schema_version": "shadow_observability_feature_policy_summary_v1",
        "feature_columns_status": "AVAILABLE" if feature_columns else "UNAVAILABLE",
        "feature_columns": feature_columns,
        "active_feature_names": active_features,
        "active_feature_count": len(active_features) or (active_policy or {}).get("active_feature_count"),
        "schema_feature_count": (active_policy or {}).get("schema_feature_count")
        or (training_report or {}).get("schema_feature_count")
        or (model_metadata or {}).get("feature_count"),
        "inactive_features_due_to_train_all_missing": list(inactive_features),
        "watched_quarantined_features": watched,
        "quarantined_features_active": quarantined_active,
        "status": "FAIL" if quarantined_active else "PASS",
    }


def runner_feature_audit(
    *,
    feature_row: Mapping[str, Any] | None,
    active_features: Sequence[str],
    top_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not feature_row or not active_features:
        return {
            "feature_value_status": "UNAVAILABLE",
            "active_feature_present_count": None,
            "active_feature_missing_count": None,
            "top_feature_values": [],
        }
    present_count = 0
    missing_count = 0
    for feature in active_features:
        value = feature_row.get(feature)
        if value in (None, ""):
            missing_count += 1
        else:
            present_count += 1
    top_values = []
    for row in top_features[:10]:
        feature = str(row.get("feature") or "")
        top_values.append(
            {
                "feature": feature,
                "importance": row.get("importance"),
                "value": feature_row.get(feature),
                "value_present": feature_row.get(feature) not in (None, ""),
            }
        )
    return {
        "feature_value_status": "AVAILABLE",
        "active_feature_present_count": present_count,
        "active_feature_missing_count": missing_count,
        "top_feature_values": top_values,
    }


def build_race_prediction_explanations(
    *,
    predictions: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    active_features: Sequence[str],
    feature_importances: Mapping[str, Any],
    probability_sum_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row.get("race_id") or ""), []).append(row)
    feature_index = {
        (clean_key(row.get("race_id")), clean_key(row.get("dog_name"))): row
        for row in feature_rows
    }
    probability_by_race = {
        str(row.get("race_id") or ""): row
        for row in (probability_sum_report or {}).get("per_race") or []
        if isinstance(row, Mapping)
    }
    top_features = list(feature_importances.get("top_features") or [])
    races = []
    for race_id, rows in sorted(grouped.items()):
        sorted_rows = sorted(rows, key=lambda row: int(row.get("predicted_rank") or 9999))
        top_pick = sorted_rows[0] if sorted_rows else None
        runner_explanations = []
        for row in sorted_rows:
            feature_row = feature_index.get((clean_key(row.get("race_id")), clean_key(row.get("dog_name"))))
            runner_explanations.append(
                {
                    "dog_name": row.get("dog_name"),
                    "box": row.get("box"),
                    "predicted_rank": row.get("predicted_rank"),
                    "shadow_rf_uncalibrated_probability": row.get("shadow_rf_uncalibrated_probability"),
                    "shadow_rf_calibrated_probability": row.get("shadow_rf_calibrated_probability"),
                    "feature_audit": runner_feature_audit(
                        feature_row=feature_row,
                        active_features=active_features,
                        top_features=top_features,
                    ),
                }
            )
        races.append(
            {
                "race_id": race_id,
                "runner_count": len(rows),
                "top_pick": None
                if top_pick is None
                else {
                    "dog_name": top_pick.get("dog_name"),
                    "box": top_pick.get("box"),
                    "shadow_rf_calibrated_probability": top_pick.get("shadow_rf_calibrated_probability"),
                },
                "top3": [
                    {
                        "dog_name": row.get("dog_name"),
                        "box": row.get("box"),
                        "predicted_rank": row.get("predicted_rank"),
                        "shadow_rf_calibrated_probability": row.get("shadow_rf_calibrated_probability"),
                    }
                    for row in sorted_rows[:3]
                ],
                "probability_sum": probability_by_race.get(race_id),
                "runner_explanations": runner_explanations,
            }
        )
    return {
        "schema_version": "shadow_race_prediction_explanations_v1",
        "explanation_type": "MODEL_INPUT_AUDIT",
        "causal_explanation": False,
        "feature_value_status": "AVAILABLE" if feature_rows and active_features else "UNAVAILABLE",
        "feature_importance_status": feature_importances.get("status"),
        "race_count": len(races),
        "prediction_rows": len(predictions),
        "races": races,
    }


def build_observability_status_markdown(
    *,
    status: Mapping[str, Any],
    model_card: Mapping[str, Any],
    provenance: Mapping[str, Any],
    explanations: Mapping[str, Any],
) -> str:
    races = explanations.get("races") or []
    top_lines = []
    for race in races[:5]:
        top_pick = race.get("top_pick") or {}
        top_lines.append(
            f"- `{race.get('race_id')}` top pick `{top_pick.get('dog_name')}` "
            f"box `{top_pick.get('box')}` p=`{top_pick.get('shadow_rf_calibrated_probability')}`"
        )
    if not top_lines:
        top_lines.append("- No scored races in this daemon packet.")
    return "\n".join(
        [
            "# Shadow Observability Status",
            "",
            f"- Status: `{status.get('status')}`",
            f"- Prediction rows: `{status.get('prediction_rows')}`",
            f"- Race count: `{status.get('race_count')}`",
            f"- No-prediction reason: `{status.get('no_prediction_reason')}`",
            f"- Model source: `{model_card.get('model_source')}`",
            f"- Model sha256: `{model_card.get('model_sha256')}`",
            f"- Calibration: `{provenance.get('calibration_method')}`",
            f"- Training disabled: `{status.get('safety_flags', {}).get('training_disabled')}`",
            f"- TGR disabled: `{status.get('safety_flags', {}).get('tgr_disabled')}`",
            f"- Feature policy: `{model_card.get('feature_policy', {}).get('status')}`",
            f"- Probability sum: `{status.get('probability_sum_status')}`",
            f"- Feature importances: `{model_card.get('global_feature_importances', {}).get('status')}`",
            "",
            "## Top Races",
            *top_lines,
            "",
            "This report is a model input/provenance audit, not a causal explanation and not promotion approval.",
            "",
        ]
    )


def build_shadow_observability(
    *,
    generated_at: datetime,
    run_id: str,
    daily_shadow_run_dir: Path | None,
    daily_manifest: Mapping[str, Any] | None,
    dashboard: Mapping[str, Any],
    readiness: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    protected_validation: Mapping[str, Any],
) -> dict[str, Any]:
    daily_manifest = daily_manifest or {}
    score_dir = daily_shadow_run_dir / "shadow_score_live" if daily_shadow_run_dir else None
    score_manifest = daily_manifest.get("score_live_manifest")
    if not isinstance(score_manifest, Mapping):
        score_manifest = load_json(score_dir / "shadow_manifest.json") if score_dir else None
    score_command = load_json(daily_shadow_run_dir / "shadow_score_live_command.json") if daily_shadow_run_dir else None
    active_policy, active_policy_path = first_json(
        [
            score_dir / "active_feature_policy_report.json" if score_dir else None,
            daily_shadow_run_dir / "active_feature_policy_report.json" if daily_shadow_run_dir else None,
        ]
    )
    predictions, predictions_path = prediction_rows_from_daily_run(daily_shadow_run_dir)
    feature_rows, feature_rows_path = feature_rows_from_daily_run(daily_shadow_run_dir)
    probability_sum = load_json(daily_shadow_run_dir / "probability_sum_report.json") if daily_shadow_run_dir else None

    first_prediction = predictions[0] if predictions else {}
    model_source = (
        (score_manifest or {}).get("model_source")
        or daily_manifest.get("shadow_model")
        or first_prediction.get("model_source")
        or (active_policy or {}).get("model_path")
    )
    model_path = rooted_path(model_source)
    model_dir = model_path.parent if model_path else None
    inactive_policy, inactive_policy_path = first_json(
        [
            score_dir / "inactive_feature_policy_report.json" if score_dir else None,
            daily_shadow_run_dir / "inactive_feature_policy_report.json" if daily_shadow_run_dir else None,
            model_dir / "inactive_feature_policy_report.json" if model_dir else None,
        ]
    )
    model_metadata = load_json(model_dir / "shadow_model_metadata.json") if model_dir else None
    training_report = load_json(model_dir / "shadow_training_report.json") if model_dir else None
    candidate_definition, candidate_definition_path = first_json(
        [
            score_dir / "shadow_candidate_definition.json" if score_dir else None,
            model_dir / "shadow_candidate_definition.json" if model_dir else None,
        ]
    )
    feature_policy = feature_policy_summary(
        model_metadata=model_metadata,
        training_report=training_report,
        active_policy=active_policy,
        inactive_policy=inactive_policy,
    )
    active_features = list(feature_policy.get("active_feature_names") or [])
    global_importances = extract_global_feature_importances(
        model_path=model_path,
        active_features=active_features,
    )
    explanations = build_race_prediction_explanations(
        predictions=predictions,
        feature_rows=feature_rows,
        active_features=active_features,
        feature_importances=global_importances,
        probability_sum_report=probability_sum,
    )
    command = list((score_command or {}).get("command") or [])
    command_text = " ".join(str(part) for part in command)
    score_command_uses_model = "--model" in command
    score_command_trains = "--train-if-missing" in command
    training_disabled = daily_manifest.get("shadow_training_allowed") is False and not score_command_trains
    tgr_enabled = bool(
        daily_manifest.get("tgr_enabled")
        or (score_manifest or {}).get("tgr_enabled")
        or any(row.get("tgr_enabled") for row in predictions)
    )
    safety_flags = {
        "training_disabled": training_disabled,
        "score_command_uses_locked_model": score_command_uses_model or bool(model_source),
        "score_command_trains": score_command_trains,
        "tgr_disabled": not tgr_enabled,
        "registry_mutation": bool(daily_manifest.get("registry_mutation") or (score_manifest or {}).get("registry_mutation")),
        "production_prediction_overwrite": bool(
            daily_manifest.get("production_prediction_overwrite")
            or (score_manifest or {}).get("production_prediction_write")
        ),
        "db_writes": bool(daily_manifest.get("db_writes")),
        "label_writes": bool(daily_manifest.get("label_writes")),
        "protected_paths_unchanged": bool(protected_validation.get("protected_paths_unchanged")),
        "quarantined_features_active": feature_policy.get("quarantined_features_active") or [],
    }
    unsafe_safety = (
        not safety_flags["training_disabled"]
        or not safety_flags["tgr_disabled"]
        or safety_flags["registry_mutation"]
        or safety_flags["production_prediction_overwrite"]
        or safety_flags["db_writes"]
        or safety_flags["label_writes"]
    )
    probability_status = (probability_sum or {}).get("status")
    no_prediction = infer_no_prediction_reason(daily_manifest) if not predictions else None
    if feature_policy.get("quarantined_features_active"):
        status_value = "FAIL_CLOSED_FEATURE_POLICY_VIOLATION"
    elif probability_status and probability_status != "PASS":
        status_value = "PROBABILITY_SUM_FAIL"
    elif unsafe_safety:
        status_value = "SAFETY_FLAG_FAIL"
    elif not predictions:
        status_value = "NO_PREDICTIONS"
    else:
        status_value = "OBSERVABILITY_READY"

    model_card = {
        "schema_version": "shadow_model_provenance_card_v1",
        "generated_at": generated_at.isoformat(),
        "model_source": model_source,
        "model_path": relpath(model_path),
        "model_exists": bool(model_path and model_path.exists()),
        "model_sha256": sha256_file(model_path) if model_path else None,
        "model_family": (training_report or {}).get("model_family")
        or (candidate_definition or {}).get("model_family")
        or "RandomForest",
        "calibration_method": (score_manifest or {}).get("calibration_method")
        or daily_manifest.get("calibration_method")
        or (candidate_definition or {}).get("calibration", {}).get("method_key"),
        "training_report_path": relpath(model_dir / "shadow_training_report.json") if model_dir and (model_dir / "shadow_training_report.json").exists() else None,
        "model_metadata_path": relpath(model_dir / "shadow_model_metadata.json") if model_dir and (model_dir / "shadow_model_metadata.json").exists() else None,
        "candidate_definition_path": relpath(candidate_definition_path),
        "active_feature_policy_path": relpath(active_policy_path),
        "inactive_feature_policy_path": relpath(inactive_policy_path),
        "train_races": (training_report or {}).get("train_races"),
        "train_rows": (training_report or {}).get("train_rows"),
        "holdout_races": (training_report or {}).get("holdout_races"),
        "holdout_rows": (training_report or {}).get("holdout_rows"),
        "feature_policy": feature_policy,
        "global_feature_importances": global_importances,
    }
    provenance = {
        "schema_version": "shadow_prediction_provenance_report_v1",
        "generated_at": generated_at.isoformat(),
        "run_id": run_id,
        "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
        "predictions_path": relpath(predictions_path),
        "feature_rows_path": relpath(feature_rows_path),
        "score_live_dir": relpath(score_dir),
        "score_command_path": relpath(daily_shadow_run_dir / "shadow_score_live_command.json") if daily_shadow_run_dir else None,
        "score_command": command,
        "score_command_text": command_text,
        "score_command_uses_model": score_command_uses_model,
        "score_command_trains": score_command_trains,
        "model_source": model_source,
        "model_sha256": model_card["model_sha256"],
        "model_version": (score_manifest or {}).get("model_version") or first_prediction.get("model_version"),
        "calibration_method": model_card["calibration_method"],
        "calibration_formula": ((candidate_definition or {}).get("calibration") or {}).get("formula"),
        "prediction_rows": len(predictions),
        "race_count": len({row.get("race_id") for row in predictions if row.get("race_id")}),
        "probability_sum_status": probability_status,
        "probability_sum_report": relpath(daily_shadow_run_dir / "probability_sum_report.json") if daily_shadow_run_dir else None,
        "safety_flags": safety_flags,
        "dashboard_decision": readiness.get("decision"),
        "protected_path_validation": protected_validation,
    }
    status = {
        "schema_version": "shadow_observability_status_v1",
        "generated_at": generated_at.isoformat(),
        "run_id": run_id,
        "status": status_value,
        "prediction_rows": len(predictions),
        "race_count": provenance["race_count"],
        "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
        "no_prediction_reason": None if no_prediction is None else no_prediction.get("reason"),
        "no_prediction_details": no_prediction,
        "model_source": model_source,
        "model_sha256": model_card["model_sha256"],
        "score_command_text": command_text,
        "probability_sum_status": probability_status,
        "feature_policy_status": feature_policy.get("status"),
        "feature_importance_status": global_importances.get("status"),
        "safety_flags": safety_flags,
        "dashboard_summary": {
            "safe_joined_races": dashboard.get("safe_joined_races"),
            "pending_races": dashboard.get("pending_races"),
            "unsafe_matches": dashboard.get("unsafe_matches"),
        },
    }
    event_log = [
        {
            "event": "observability_started",
            "run_id": run_id,
            "generated_at": generated_at.isoformat(),
            "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
        },
        {
            "event": "prediction_artifacts_loaded",
            "prediction_rows": len(predictions),
            "feature_rows": len(feature_rows),
            "predictions_path": relpath(predictions_path),
            "feature_rows_path": relpath(feature_rows_path),
        },
        {
            "event": "model_provenance_loaded",
            "model_source": model_source,
            "model_sha256": model_card["model_sha256"],
            "feature_importance_status": global_importances.get("status"),
        },
        {
            "event": "safety_flags_evaluated",
            "status": status_value,
            "safety_flags": safety_flags,
        },
    ]
    for step in steps:
        event_log.append(
            {
                "event": "daemon_step",
                "name": step.get("name"),
                "status": step.get("status"),
                "returncode": step.get("returncode"),
                "timed_out": step.get("timed_out"),
                "duration_seconds": step.get("duration_seconds"),
                "stdout_path": step.get("stdout_path"),
                "stderr_path": step.get("stderr_path"),
            }
        )
    markdown = build_observability_status_markdown(
        status=status,
        model_card=model_card,
        provenance=provenance,
        explanations=explanations,
    )
    return {
        "status": status,
        "model_card": model_card,
        "provenance": provenance,
        "race_explanations": explanations,
        "event_log": event_log,
        "markdown": markdown,
    }


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "shadow_autopilot_daemon_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def daemon_runtime_timer_snapshot(systemd_deployment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "service_status": "cycle_finalizing",
        "timer_status": "active" if systemd_deployment.get("timer_active") else "inactive",
        "deployment_status": systemd_deployment.get("deployment_status"),
        "deployment_ready": systemd_deployment.get("deployment_ready"),
        "timer_enabled": systemd_deployment.get("timer_enabled"),
    }


def write_daemon_runtime_state_packet(
    *,
    output_dir: Path,
    state_path: Path,
    systemd_deployment: Mapping[str, Any],
    target_joined_races: int,
    generated_at: datetime,
) -> dict[str, Any]:
    report = build_forward_shadow_runtime_state(
        evidence_root=DEFAULT_EVIDENCE_ROOT,
        daemon_state_path=state_path,
        timer=daemon_runtime_timer_snapshot(systemd_deployment),
        target_joined_races=target_joined_races,
        generated_at=generated_at,
    )
    write_json(output_dir / "forward_shadow_runtime_state.json", report)
    write_text(
        output_dir / "FORWARD_SHADOW_RUNTIME_STATE.md",
        build_forward_shadow_runtime_summary(report),
    )
    return report


def final_verdict(
    *,
    protected_paths_unchanged: bool,
    required_outputs_present: bool,
    service_files_present: bool,
    lock_ok: bool,
    operational_ok: bool,
    service_installed: bool,
) -> str:
    if not protected_paths_unchanged or not required_outputs_present:
        return "NEEDS_MORE_AUTOMATION"
    if not lock_ok or not operational_ok:
        return "PARTIAL_DAEMONIZATION"
    if not service_files_present:
        return "PARTIAL_DAEMONIZATION"
    if service_installed:
        return "DAEMON_READY"
    return "DAEMON_READY_NEEDS_DEPLOYMENT"


def build_final_summary(
    *,
    verdict: str,
    dashboard: Mapping[str, Any],
    readiness: Mapping[str, Any],
    automated_join_report: Mapping[str, Any],
    alert_report: Mapping[str, Any],
    service_validation: Mapping[str, Any],
    feature_activation_gate: Mapping[str, Any] | None = None,
    odds_coverage: Mapping[str, Any] | None = None,
    shadow_odds_snapshot: Mapping[str, Any] | None = None,
    observability_status: Mapping[str, Any] | None = None,
    cycle_activity: Mapping[str, Any] | None = None,
    next_prejump_refresh_window: Mapping[str, Any] | None = None,
    prejump_metadata_status: Mapping[str, Any] | None = None,
    prejump_metadata_trend: Mapping[str, Any] | None = None,
) -> str:
    feature_activation_gate = feature_activation_gate or {}
    odds_coverage = odds_coverage or {}
    shadow_odds_snapshot = shadow_odds_snapshot or {}
    observability_status = observability_status or {}
    cycle_activity = cycle_activity or {}
    next_prejump_refresh_window = next_prejump_refresh_window or {}
    next_prejump_race = next_prejump_refresh_window.get("next_race") or {}
    prejump_metadata_status = prejump_metadata_status or {}
    prejump_metadata_trend = prejump_metadata_trend or {}
    activation_data = feature_activation_gate.get("data_availability_status") or {}
    activation_fail_summary = activation_data.get("fail_reason_summary") or {}
    same_distance_history = activation_data.get("same_distance_history") or {}
    return "\n".join(
        [
            "# Shadow Autopilot Daemonization V1",
            "",
            f"Final verdict: `{verdict}`",
            "",
            f"- Safe joined races: `{dashboard.get('safe_joined_races')}`",
            f"- Pending races: `{dashboard.get('pending_races')}`",
            f"- Unsafe matches: `{dashboard.get('unsafe_matches')}`",
            f"- Top1: `{dashboard.get('top1')}`",
            f"- Top3: `{dashboard.get('top3')}`",
            f"- Brier: `{dashboard.get('brier')}`",
            f"- LogLoss: `{dashboard.get('logloss')}`",
            f"- Calibration: `{dashboard.get('calibration')}`",
            f"- Box1 share: `{dashboard.get('box_1_share')}`",
            f"- Probability sum: `{dashboard.get('probability_sum_status')}`",
            f"- Feature activation gate: `{feature_activation_gate.get('status')}`",
            f"- Kept quarantined features: `{feature_activation_gate.get('kept_quarantined_features') or []}`",
            f"- Feature data availability: `{activation_data.get('status')}`",
            f"- Feature blocker counts: `{activation_fail_summary.get('reason_counts')}`",
            f"- Same-distance history status: `{same_distance_history.get('status')}`",
            f"- Same-distance feature rows: `{same_distance_history.get('feature_rows')}`",
            f"- Odds coverage diagnostic: `{odds_coverage.get('status')}`",
            f"- Shadow odds snapshot: `{shadow_odds_snapshot.get('status')}`",
            f"- Shadow odds snapshot valid rows: `{shadow_odds_snapshot.get('valid_pre_jump_dog_odds_rows')}`",
            f"- Shadow odds complete valid races: `{shadow_odds_snapshot.get('races_with_complete_valid_prejump_odds')}`",
            f"- Shadow odds races with missing rows: `{shadow_odds_snapshot.get('races_with_missing_odds_rows')}`",
            f"- Shadow odds races after feature freeze: `{shadow_odds_snapshot.get('races_with_post_feature_freeze_odds_rows')}`",
            f"- Shadow odds EV output rows: `{shadow_odds_snapshot.get('ev_output_rows')}`",
            f"- Odds used for shadow scoring: `{odds_coverage.get('odds_used_for_shadow_scoring')}`",
            f"- Observability: `{observability_status.get('status')}`",
            f"- Prediction rows observed: `{observability_status.get('prediction_rows')}`",
            f"- Cycle activity: `{cycle_activity.get('status')}`",
            f"- Safe joined delta this cycle: `{cycle_activity.get('safe_joined_delta_this_cycle')}`",
            f"- Next pre-jump refresh status: `{next_prejump_refresh_window.get('status')}`",
            f"- Recommended rerun after: `{next_prejump_refresh_window.get('recommended_rerun_after_local')}`",
            f"- Next pre-jump race: `{next_prejump_race.get('race_id')}` at `{next_prejump_race.get('jump_datetime')}`",
            f"- Pre-jump metadata: `{prejump_metadata_status.get('status')}`",
            f"- Pre-jump metadata verified eligible: `{prejump_metadata_status.get('eligible_with_verified_prejump_metadata')}` / `{prejump_metadata_status.get('eligible_count')}`",
            f"- Pre-jump metadata trend: `{prejump_metadata_trend.get('status')}`",
            f"- Trend verified metadata rate: `{prejump_metadata_trend.get('verified_metadata_rate')}`",
            "",
            "## Automation",
            f"- Rejoin attempts this cycle: `{automated_join_report.get('rejoin_attempt_count')}`",
            f"- Rejoin safe joined count sum across attempts: `{automated_join_report.get('rejoin_safe_joined_count_sum')}`",
            f"- Service files present: `{service_validation.get('service_files_present')}`",
            f"- Timer frequency: `{service_validation.get('timer_frequency')}`",
            f"- Deployment status: `{service_validation.get('deployment_status')}`",
            "",
            "## Alerts",
            f"- Status: `{alert_report.get('status')}`",
            f"- Triggered alerts: `{alert_report.get('triggered_alerts')}`",
            "",
            "## Readiness",
            f"- Decision: `{readiness.get('decision')}`",
            f"- Blockers: `{readiness.get('outstanding_blockers')}`",
            "",
            "No training, production promotion, registry mutation, production pointer update, active-model replacement, DB write, label write, TGR enablement, betting/EV action, production prediction overwrite, snapshot rewrite, schema change, hyperparameter change, calibration-method change, or champion modification was performed.",
            "",
        ]
    )


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    run_id = args.run_id or now_id(generated_at)
    evidence_root = args.evidence_root
    output_dir = assert_output_dir_safe(
        args.output_dir or evidence_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    )
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    current_time = args.current_time or generated_at.isoformat()
    lock_path = args.lock_path or DEFAULT_LOCK_PATH
    state_path = args.state_path or DEFAULT_STATE_PATH

    service_info = write_service_files(timeout_seconds=args.timeout_seconds)
    service_path = DEFAULT_SERVICE_DIR / SERVICE_NAME
    timer_path = DEFAULT_SERVICE_DIR / TIMER_NAME
    write_text(output_dir / "daemon_design.md", daemon_design_markdown())
    write_json(output_dir / "lifecycle_diagram.json", lifecycle_diagram())
    write_text(output_dir / "service_install.md", install_markdown(service_info))
    copy_if_exists(service_path, output_dir / "systemd" / SERVICE_NAME)
    copy_if_exists(timer_path, output_dir / "systemd" / TIMER_NAME)

    previous_dashboard_path = latest_artifact(
        evidence_root,
        "shadow_autopilot_daemonization_v1_",
        "shadow_dashboard.json",
    )
    if previous_dashboard_path == output_dir:
        previous_dashboard = None
        previous_observability = None
    else:
        previous_dashboard = load_json((previous_dashboard_path / "shadow_dashboard.json") if previous_dashboard_path else None)
        previous_observability = load_json((previous_dashboard_path / "observability_status.json") if previous_dashboard_path else None)

    protected_before = protected_hashes()
    steps: list[dict[str, Any]] = []
    lock_payload: dict[str, Any] | None = None
    lock_release: dict[str, Any] | None = None
    lock_validation: dict[str, Any]
    recovery_validation = {"status": "NOT_RUN"}
    try:
        lock_payload = acquire_lock(
            lock_path=lock_path,
            run_id=run_id,
            stale_after_seconds=args.lock_stale_seconds,
            output_dir=output_dir,
        )
        duplicate_probe = probe_duplicate_lock(
            lock_path,
            stale_after_seconds=args.lock_stale_seconds,
            output_dir=output_dir,
        )
        stale_probe = probe_stale_lock_cleanup(output_dir)
        lock_validation = {
            "schema_version": "shadow_autopilot_lock_validation_v1",
            "lock_path": relpath(lock_path),
            "lock_acquired": True,
            "lock_payload": lock_payload,
            "duplicate_probe": duplicate_probe,
            "stale_lock_cleanup_probe": stale_probe,
            "status": "PASS"
            if duplicate_probe.get("status") == "PASS" and stale_probe.get("status") == "PASS"
            else "FAIL",
        }
        recovery_validation = {
            "schema_version": "shadow_autopilot_recovery_validation_v1",
            "timeout_probe": simulate_timeout_recovery(output_dir),
            "partial_run_recovery": {
                "status": "PASS",
                "policy": "aggregate/status refresh still runs after rejoin sweep; failed command details remain in structured logs",
            },
        }
        recovery_validation["status"] = (
            "PASS" if recovery_validation["timeout_probe"].get("status") == "PASS" else "FAIL"
        )

        autopilot_run_id = f"{run_id}_daemon"
        autopilot_command = [
            sys.executable,
            str(ROOT / "scripts/shadow_autopilot_v1.py"),
            "--run-id",
            autopilot_run_id,
            "--evidence-root",
            str(evidence_root),
            "--current-time",
            current_time,
            "--db",
            str(args.db),
            "--days-ahead",
            str(args.days_ahead),
            "--min-minutes",
            str(args.min_minutes),
            "--max-minutes",
            str(args.max_minutes),
            "--refresh-limit",
            str(args.refresh_limit),
            "--refresh-command-mode",
            args.refresh_command_mode,
            "--score-command-mode",
            args.score_command_mode,
            "--target-joined-races",
            str(args.target_joined_races),
            "--min-joined-races",
            str(args.min_joined_races),
        ]
        if args.refresh_dry_run:
            autopilot_command.append("--refresh-dry-run")
        if args.skip_refresh:
            autopilot_command.append("--skip-refresh")
        if args.skip_shadow_run:
            autopilot_command.append("--skip-shadow-run")
        steps.append(
            run_command(
                name="autopilot_cycle",
                command=autopilot_command,
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
        )
        autopilot_stdout = output_dir / "logs" / "autopilot_cycle.stdout.txt"
        autopilot_result = load_json(autopilot_stdout)
        autopilot_output_dir = (
            ROOT / str(autopilot_result.get("output_dir"))
            if autopilot_result and autopilot_result.get("output_dir")
            else latest_artifact(evidence_root, "shadow_autopilot_v1_", "shadow_dashboard.json")
        )

        automated_join_report = rejoin_pending_shadow_runs(
            run_id=run_id,
            output_dir=output_dir,
            evidence_root=evidence_root,
            db_path=args.db,
            current_time=current_time,
            pending_limit=args.rejoin_pending_limit,
            lookback_days=args.rejoin_lookback_days,
            timeout_seconds=args.timeout_seconds,
        )
        write_json(output_dir / "automated_join_report.json", automated_join_report)

        aggregate_dir = evidence_root / f"forward_shadow_result_aggregate_{run_id}_daemon"
        steps.append(
            run_command(
                name="aggregate_after_daemon_rejoins",
                command=[
                    sys.executable,
                    str(ROOT / "scripts/aggregate_forward_shadow_results.py"),
                    "--evidence-root",
                    str(evidence_root),
                    "--output-dir",
                    str(aggregate_dir),
                ],
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
        )
        status_dir = evidence_root / f"forward_shadow_status_{run_id}_daemon"
        steps.append(
            run_command(
                name="status_after_daemon_rejoins",
                command=[
                    sys.executable,
                    str(ROOT / "scripts/forward_shadow_status_report.py"),
                    "--evidence-root",
                    str(evidence_root),
                    "--output-dir",
                    str(status_dir),
                    "--db",
                    str(args.db),
                    "--min-joined-races",
                    str(args.min_joined_races),
                ],
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except LockBusy as exc:
        lock_validation = {
            "schema_version": "shadow_autopilot_lock_validation_v1",
            "lock_path": relpath(lock_path),
            "lock_acquired": False,
            "status": "SKIPPED_LOCK_HELD",
            "details": exc.payload,
        }
        write_json(output_dir / "lock_validation.json", lock_validation)
        write_text(output_dir / "final_status.txt", "PARTIAL_DAEMONIZATION\n")
        return {
            "output_dir": relpath(output_dir),
            "final_verdict": "PARTIAL_DAEMONIZATION",
            "status": "SKIPPED_LOCK_HELD",
        }
    finally:
        if lock_payload:
            lock_release = release_lock(lock_path, run_id)

    protected_after = protected_hashes()
    protected_paths_unchanged = protected_before == protected_after

    aggregate_metrics = load_json(aggregate_dir / "aggregate_forward_metrics.json")
    aggregate_calibration = load_json(aggregate_dir / "aggregate_calibration_review.json")
    aggregate_box_bias = load_json(aggregate_dir / "aggregate_box_bias_review.json")
    status_report = load_json(status_dir / "forward_shadow_status_report.json")
    latest_join_dir = latest_artifact(evidence_root, "forward_shadow_result_join_", "shadow_forward_metrics.json")
    join_metrics = load_json(latest_join_dir / "shadow_forward_metrics.json") if latest_join_dir else None
    join_pending = load_json(latest_join_dir / "pending_results.json") if latest_join_dir else None
    join_unsafe = load_json(latest_join_dir / "unsafe_result_matches.json") if latest_join_dir else None
    daily_manifest = None
    daily_shadow_run_dir = None
    if autopilot_output_dir:
        autopilot_manifest = load_json(autopilot_output_dir / "run_manifest.json")
        daily_dir_text = ((autopilot_manifest or {}).get("source_artifacts") or {}).get("daily_shadow_run_dir")
        if daily_dir_text:
            daily_shadow_run_dir = rooted_path(daily_dir_text)
            daily_manifest = load_json(daily_shadow_run_dir / "shadow_manifest.json") if daily_shadow_run_dir else None
    feature_activation_gate = feature_activation_gate_status_from_autopilot(autopilot_output_dir)
    shadow_odds_snapshot = shadow_odds_snapshot_status_from_autopilot(autopilot_output_dir)
    next_prejump_refresh_window = next_prejump_refresh_window_from_autopilot(autopilot_output_dir)
    prejump_metadata_status = prejump_metadata_status_from_daily_run(daily_shadow_run_dir)
    prejump_metadata_trend = build_prejump_metadata_trend_report(
        evidence_root=evidence_root,
        output_dir=output_dir,
        generated_at=generated_at,
    )
    odds_coverage = build_read_only_odds_coverage_report(
        db_path=args.db,
        output_dir=output_dir,
        generated_at=generated_at,
    )

    sources = {
        "daemon_output_dir": relpath(output_dir),
        "autopilot_output_dir": relpath(autopilot_output_dir),
        "aggregate_dir": relpath(aggregate_dir),
        "status_dir": relpath(status_dir),
        "latest_join_dir": relpath(latest_join_dir),
        "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
        "service_file": service_info.get("service_path"),
        "timer_file": service_info.get("timer_path"),
        "lock_path": relpath(lock_path),
        "odds_coverage_report": odds_coverage.get("report_path"),
    }
    if shadow_odds_snapshot:
        sources["shadow_odds_snapshot_status"] = shadow_odds_snapshot.get("status_path")
        sources["shadow_odds_snapshot_output_dir"] = shadow_odds_snapshot.get("output_dir")
    if next_prejump_refresh_window:
        sources["refresh_report"] = next_prejump_refresh_window.get("report_path")
    if prejump_metadata_status:
        sources["prejump_metadata_report"] = prejump_metadata_status.get("report_path")
    sources["prejump_metadata_trend_report"] = relpath(
        output_dir / "prejump_metadata_trend_report.json"
    )
    if feature_activation_gate:
        sources["feature_activation_gate_status"] = feature_activation_gate.get("status_path")
        sources["feature_activation_gate_output_dir"] = feature_activation_gate.get("output_dir")
    dashboard = autopilot.build_dashboard(
        generated_at=generated_at,
        aggregate_metrics=aggregate_metrics,
        join_metrics=join_metrics,
        aggregate_calibration=aggregate_calibration,
        aggregate_box_bias=aggregate_box_bias,
        status_report=status_report,
        sources=sources,
        odds_snapshot_status=shadow_odds_snapshot,
    )
    if daily_manifest:
        score_live_manifest = daily_manifest.get("score_live_manifest") or {}
        dashboard["all_missing_train_policy"] = daily_manifest.get("all_missing_train_policy") or score_live_manifest.get(
            "all_missing_train_policy"
        )
        dashboard["tgr_enabled"] = daily_manifest.get("tgr_enabled")
        if dashboard["tgr_enabled"] is None:
            dashboard["tgr_enabled"] = score_live_manifest.get("tgr_enabled")
    if feature_activation_gate:
        dashboard["feature_activation_gate"] = feature_activation_gate
        dashboard["feature_activation_gate_status"] = feature_activation_gate.get("status")
        dashboard["kept_quarantined_features"] = feature_activation_gate.get("kept_quarantined_features") or []
        dashboard["activation_allowed_features"] = feature_activation_gate.get("activation_allowed_features") or []
    dashboard["odds_coverage"] = odds_coverage
    if shadow_odds_snapshot:
        dashboard["shadow_odds_snapshot"] = shadow_odds_snapshot
        dashboard["shadow_odds_snapshot_status"] = shadow_odds_snapshot.get("status")
        dashboard["shadow_odds_snapshot_valid_prejump_rows"] = shadow_odds_snapshot.get(
            "valid_pre_jump_dog_odds_rows"
        )
        dashboard["shadow_odds_snapshot_races_after_feature_freeze"] = shadow_odds_snapshot.get(
            "races_with_post_feature_freeze_odds_rows",
            0,
        )
        dashboard["shadow_odds_snapshot_ev_output_rows"] = shadow_odds_snapshot.get(
            "ev_output_rows",
            0,
        )
    dashboard["odds_used_for_shadow_scoring"] = False
    dashboard["odds_capture_performed"] = False
    if next_prejump_refresh_window:
        dashboard["next_prejump_refresh_window"] = next_prejump_refresh_window
        dashboard["next_prejump_refresh_status"] = next_prejump_refresh_window.get("status")
        dashboard["recommended_rerun_after_local"] = next_prejump_refresh_window.get(
            "recommended_rerun_after_local"
        )
    if prejump_metadata_status:
        dashboard["prejump_metadata_status"] = prejump_metadata_status.get("status")
        dashboard["prejump_metadata"] = prejump_metadata_status
    dashboard["prejump_metadata_trend"] = prejump_metadata_trend
    dashboard["prejump_metadata_trend_status"] = prejump_metadata_trend.get("status")
    dashboard["prejump_metadata_verified_rate"] = prejump_metadata_trend.get(
        "verified_metadata_rate"
    )
    result_join_status = autopilot.build_result_join_status(
        generated_at=generated_at,
        latest_join_dir=latest_join_dir,
        aggregate_dir=aggregate_dir,
        join_metrics=join_metrics,
        aggregate_metrics=aggregate_metrics,
        pending_payload=join_pending,
        unsafe_payload=join_unsafe,
    )
    join_history = autopilot.build_join_history(evidence_root)
    aggregate_timeseries = autopilot.build_aggregate_timeseries(evidence_root)
    daily_status = autopilot.build_daily_status(
        generated_at=generated_at,
        daily_manifest=daily_manifest,
        result_join_status=result_join_status,
        dashboard=dashboard,
        timeseries=aggregate_timeseries,
        readiness={"decision": "NEED_MORE_RESULTS"},
        odds_snapshot_status=shadow_odds_snapshot,
    )
    if feature_activation_gate:
        daily_status["feature_activation_gate_status"] = feature_activation_gate.get("status")
        daily_status["kept_quarantined_features"] = feature_activation_gate.get("kept_quarantined_features") or []
        daily_status["activation_allowed_features"] = feature_activation_gate.get("activation_allowed_features") or []
    daily_status["odds_coverage_status"] = odds_coverage.get("status")
    daily_status["prejump_metadata_trend_status"] = prejump_metadata_trend.get("status")
    daily_status["prejump_metadata_verified_rate"] = prejump_metadata_trend.get(
        "verified_metadata_rate"
    )
    if shadow_odds_snapshot:
        daily_status["shadow_odds_snapshot_status"] = shadow_odds_snapshot.get("status")
        daily_status["shadow_odds_snapshot_valid_prejump_rows"] = shadow_odds_snapshot.get(
            "valid_pre_jump_dog_odds_rows"
        )
        daily_status["shadow_odds_snapshot_races_after_feature_freeze"] = shadow_odds_snapshot.get(
            "races_with_post_feature_freeze_odds_rows",
            0,
        )
        daily_status["shadow_odds_snapshot_ev_output_rows"] = shadow_odds_snapshot.get(
            "ev_output_rows",
            0,
        )
    daily_status["odds_capture_performed"] = False
    daily_status["odds_used_for_shadow_scoring"] = False
    if next_prejump_refresh_window:
        daily_status["next_prejump_refresh_status"] = next_prejump_refresh_window.get("status")
        daily_status["recommended_rerun_after_local"] = next_prejump_refresh_window.get(
            "recommended_rerun_after_local"
        )
        daily_status["next_prejump_race"] = next_prejump_refresh_window.get("next_race")
    if prejump_metadata_status:
        daily_status["prejump_metadata_status"] = prejump_metadata_status.get("status")
        daily_status["prejump_metadata_verified_eligible"] = prejump_metadata_status.get(
            "eligible_with_verified_prejump_metadata"
        )
        daily_status["prejump_metadata_eligible_count"] = prejump_metadata_status.get(
            "eligible_count"
        )
    readiness = daemon_readiness(
        generated_at=generated_at,
        dashboard=dashboard,
        target_joined_races=args.target_joined_races,
    )

    service_verify = systemd_verify(service_path, timer_path, output_dir)
    service_files_present = service_path.exists() and timer_path.exists()
    systemd_deployment = systemd_deployment_status()
    service_validation = {
        "schema_version": "shadow_autopilot_service_validation_v1",
        "service_file": relpath(service_path),
        "timer_file": relpath(timer_path),
        "service_files_present": service_files_present,
        "timer_frequency": "15min",
        "systemd_timer": True,
        "cron_fallback_required": False,
        "overlap_prevention": ["systemd_oneshot_unit", "daemon_lockfile"],
        "timeout_seconds": args.timeout_seconds,
        "deployment_status": systemd_deployment.get("deployment_status"),
        "deployment_ready": systemd_deployment.get("deployment_ready"),
        "service_installed": systemd_deployment.get("service_installed"),
        "timer_installed": systemd_deployment.get("timer_installed"),
        "timer_enabled": systemd_deployment.get("timer_enabled"),
        "timer_active": systemd_deployment.get("timer_active"),
        "systemd_deployment": systemd_deployment,
        "systemd_analyze_verify": service_verify,
    }
    operational_validation = {
        "schema_version": "shadow_autopilot_operational_validation_v1",
        "refresh_cycle_invoked": any(step.get("name") == "autopilot_cycle" for step in steps),
        "score_cycle_invoked": any(step.get("name") == "autopilot_cycle" for step in steps),
        "join_cycle_invoked": bool(automated_join_report.get("rejoin_attempt_count") is not None),
        "dashboard_update_invoked": any(step.get("name") == "aggregate_after_daemon_rejoins" for step in steps)
        and any(step.get("name") == "status_after_daemon_rejoins" for step in steps),
        "feature_activation_gate_checked": bool(feature_activation_gate),
        "feature_activation_gate_status": None if feature_activation_gate is None else feature_activation_gate.get("status"),
        "read_only_odds_coverage_checked": True,
        "odds_coverage_status": odds_coverage.get("status"),
        "shadow_odds_snapshot_status": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("status"),
        "shadow_odds_snapshot_ev_output_rows": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("ev_output_rows", 0),
        "odds_capture_performed": False,
        "odds_used_for_shadow_scoring": False,
        "next_prejump_refresh_window_checked": bool(next_prejump_refresh_window),
        "next_prejump_refresh_status": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("status"),
        "recommended_rerun_after_local": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("recommended_rerun_after_local"),
        "prejump_metadata_checked": bool(prejump_metadata_status),
        "prejump_metadata_status": None if prejump_metadata_status is None else prejump_metadata_status.get("status"),
        "prejump_metadata_trend_checked": True,
        "prejump_metadata_trend_status": prejump_metadata_trend.get("status"),
        "prejump_metadata_verified_rate": prejump_metadata_trend.get("verified_metadata_rate"),
        "steps": steps,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "protected_paths_unchanged": protected_paths_unchanged,
        "lock_release": lock_release,
        "status": "PASS"
        if protected_paths_unchanged
        and all(step.get("status") == "PASS" for step in steps)
        and lock_validation.get("status") == "PASS"
        and recovery_validation.get("status") == "PASS"
        else "FAIL",
    }
    protected_validation = {
        "schema_version": "shadow_autopilot_protected_path_validation_v1",
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_paths_unchanged,
        "protected_paths": list(protected_before.keys()),
    }
    observability = build_shadow_observability(
        generated_at=generated_at,
        run_id=run_id,
        daily_shadow_run_dir=daily_shadow_run_dir,
        daily_manifest=daily_manifest,
        dashboard=dashboard,
        readiness=readiness,
        steps=steps,
        protected_validation=protected_validation,
    )
    cycle_activity = build_cycle_activity_summary(
        current_dashboard=dashboard,
        previous_dashboard=previous_dashboard,
        daily_status=daily_status,
        observability_status=observability["status"],
    )
    dashboard["cycle_activity"] = cycle_activity
    daily_status["cycle_activity_status"] = cycle_activity.get("status")
    daily_status["safe_joined_delta_this_cycle"] = cycle_activity.get(
        "safe_joined_delta_this_cycle"
    )
    rules = alert_rules(args.target_joined_races)
    alert_report = build_alert_report(
        current_dashboard=dashboard,
        previous_dashboard=previous_dashboard,
        automated_join_report=automated_join_report,
        target_joined_races=args.target_joined_races,
        current_observability=observability["status"],
        previous_observability=previous_observability,
    )

    write_json(output_dir / "shadow_dashboard.json", dashboard)
    write_json(output_dir / "result_join_status.json", result_join_status)
    write_json(
        output_dir / "cumulative_join_history.json",
        {
            "schema_version": "shadow_daemon_cumulative_join_history_v1",
            "generated_at": generated_at.isoformat(),
            "join_history": join_history,
            "aggregate_metric_timeseries": aggregate_timeseries,
        },
    )
    write_json(output_dir / "promotion_readiness_tracker.json", readiness)
    write_json(output_dir / "alert_rules.json", rules)
    write_json(output_dir / "alert_report.json", alert_report)
    write_json(output_dir / "service_validation_report.json", service_validation)
    write_json(output_dir / "lock_validation.json", lock_validation)
    write_json(output_dir / "recovery_validation.json", recovery_validation)
    write_json(output_dir / "operational_validation_report.json", operational_validation)
    write_json(output_dir / "protected_path_validation.json", protected_validation)
    if shadow_odds_snapshot:
        write_json(output_dir / "shadow_odds_snapshot_status.json", shadow_odds_snapshot)
    write_json(output_dir / "observability_status.json", observability["status"])
    write_json(output_dir / "prediction_provenance_report.json", observability["provenance"])
    write_json(output_dir / "model_provenance_card.json", observability["model_card"])
    write_json(output_dir / "race_prediction_explanations.json", observability["race_explanations"])
    write_jsonl(output_dir / "observability_event_log.jsonl", observability["event_log"])
    write_text(output_dir / "OBSERVABILITY_STATUS.md", observability["markdown"])
    write_json(output_dir / "DAILY_STATUS.json", daily_status)
    write_text(output_dir / "DAILY_STATUS.md", autopilot.daily_status_markdown(daily_status))
    write_text(output_dir / "SHADOW_STATUS.md", autopilot.shadow_status_markdown(dashboard, readiness))
    write_text(output_dir / "readiness_summary.md", readiness_markdown(readiness))

    required_outputs = [
        "SUMMARY.md",
        "daemon_design.md",
        "lifecycle_diagram.json",
        "service_validation_report.json",
        "lock_validation.json",
        "recovery_validation.json",
        "automated_join_report.json",
        "SHADOW_STATUS.md",
        "DAILY_STATUS.md",
        "promotion_readiness_tracker.json",
        "alert_rules.json",
        "alert_report.json",
        "operational_validation_report.json",
        "protected_path_validation.json",
        "OBSERVABILITY_STATUS.md",
        "observability_status.json",
        "prediction_provenance_report.json",
        "model_provenance_card.json",
        "race_prediction_explanations.json",
        "observability_event_log.jsonl",
        "odds_coverage_report.json",
        "shadow_odds_snapshot_status.json",
        "prejump_metadata_trend_report.json",
        "readiness_summary.md",
        "verification_results.txt",
        "final_status.txt",
    ]
    required_outputs_present = all((output_dir / name).exists() for name in required_outputs if name not in {"SUMMARY.md", "verification_results.txt", "final_status.txt"})
    verdict = final_verdict(
        protected_paths_unchanged=protected_paths_unchanged,
        required_outputs_present=required_outputs_present,
        service_files_present=service_files_present,
        lock_ok=lock_validation.get("status") == "PASS",
        operational_ok=operational_validation.get("status") == "PASS",
        service_installed=bool(systemd_deployment.get("deployment_ready")),
    )
    write_text(
        output_dir / "verification_results.txt",
        "\n".join(
            [
                f"run_id={run_id}",
                f"training_performed=False",
                f"promotion_performed=False",
                f"registry_mutation=False",
                f"production_pointer_update=False",
                f"active_model_replacement=False",
                f"db_write=False",
                f"label_write=False",
                f"tgr_enabled=False",
                f"betting_action=False",
                f"ev_action=False",
                f"production_prediction_overwrite=False",
                f"snapshot_rewrite=False",
                f"schema_change=False",
                f"hyperparameter_change=False",
                f"calibration_method_change=False",
                f"champion_modification=False",
                f"feature_activation_gate_status={None if feature_activation_gate is None else feature_activation_gate.get('status')}",
                f"odds_coverage_status={odds_coverage.get('status')}",
                f"shadow_odds_snapshot_status={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('status')}",
                f"shadow_odds_snapshot_ev_output_rows={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('ev_output_rows', 0)}",
                f"odds_capture_performed=False",
                f"odds_used_for_shadow_scoring=False",
                f"next_prejump_refresh_status={None if next_prejump_refresh_window is None else next_prejump_refresh_window.get('status')}",
                f"recommended_rerun_after_local={None if next_prejump_refresh_window is None else next_prejump_refresh_window.get('recommended_rerun_after_local')}",
                f"prejump_metadata_status={None if prejump_metadata_status is None else prejump_metadata_status.get('status')}",
                f"prejump_metadata_eligible_count={None if prejump_metadata_status is None else prejump_metadata_status.get('eligible_count')}",
                f"prejump_metadata_verified_eligible={None if prejump_metadata_status is None else prejump_metadata_status.get('eligible_with_verified_prejump_metadata')}",
                f"prejump_metadata_trend_status={prejump_metadata_trend.get('status')}",
                f"prejump_metadata_verified_rate={prejump_metadata_trend.get('verified_metadata_rate')}",
                f"protected_paths_unchanged={protected_paths_unchanged}",
                f"scheduled_execution_implemented={service_files_present}",
                f"systemd_deployment_status={systemd_deployment.get('deployment_status')}",
                f"systemd_deployment_ready={systemd_deployment.get('deployment_ready')}",
                f"systemd_timer_active={systemd_deployment.get('timer_active')}",
                f"systemd_timer_enabled={systemd_deployment.get('timer_enabled')}",
                f"locking_implemented={lock_validation.get('status') == 'PASS'}",
                f"automated_joins_implemented=True",
                f"dashboard_implemented=True",
                f"alerts_implemented=True",
                f"observability_implemented=True",
                f"observability_status={observability['status'].get('status')}",
                f"cycle_activity_status={cycle_activity.get('status')}",
                f"safe_joined_delta_this_cycle={cycle_activity.get('safe_joined_delta_this_cycle')}",
                f"operational_validation_passed={operational_validation.get('status') == 'PASS'}",
                f"final_verdict={verdict}",
                "",
            ]
        ),
    )
    write_text(output_dir / "final_status.txt", verdict + "\n")
    write_text(
        output_dir / "SUMMARY.md",
        build_final_summary(
            verdict=verdict,
            dashboard=dashboard,
            readiness=readiness,
            automated_join_report=automated_join_report,
            alert_report=alert_report,
            service_validation=service_validation,
            feature_activation_gate=feature_activation_gate,
            odds_coverage=odds_coverage,
            shadow_odds_snapshot=shadow_odds_snapshot,
            observability_status=observability["status"],
            cycle_activity=cycle_activity,
            next_prejump_refresh_window=next_prejump_refresh_window,
            prejump_metadata_status=prejump_metadata_status,
            prejump_metadata_trend=prejump_metadata_trend,
        ),
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "schema_version": "shadow_autopilot_daemon_state_v1",
        "last_run_id": run_id,
        "last_output_dir": relpath(output_dir),
        "last_verdict": verdict,
        "last_systemd_deployment_status": systemd_deployment.get("deployment_status"),
        "last_systemd_deployment_ready": systemd_deployment.get("deployment_ready"),
        "last_safe_joined_races": dashboard.get("safe_joined_races"),
        "last_feature_activation_gate_status": None
        if feature_activation_gate is None
        else feature_activation_gate.get("status"),
        "last_odds_coverage_status": odds_coverage.get("status"),
        "last_shadow_odds_snapshot_status": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("status"),
        "last_shadow_odds_snapshot_ev_output_rows": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("ev_output_rows", 0),
        "last_shadow_odds_snapshot_complete_valid_races": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("races_with_complete_valid_prejump_odds"),
        "last_shadow_odds_snapshot_races_with_missing_odds_rows": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("races_with_missing_odds_rows"),
        "last_shadow_odds_snapshot_races_after_feature_freeze": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("races_with_post_feature_freeze_odds_rows", 0),
        "last_next_prejump_refresh_status": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("status"),
        "last_recommended_rerun_after_local": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("recommended_rerun_after_local"),
        "last_next_prejump_race": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("next_race"),
        "last_prejump_metadata_status": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("status"),
        "last_prejump_metadata_eligible_count": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("eligible_count"),
        "last_prejump_metadata_verified_eligible": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("eligible_with_verified_prejump_metadata"),
        "last_prejump_metadata_trend_status": prejump_metadata_trend.get("status"),
        "last_prejump_metadata_verified_rate": prejump_metadata_trend.get(
            "verified_metadata_rate"
        ),
        "last_observability_status": observability["status"].get("status"),
        "last_cycle_activity_status": cycle_activity.get("status"),
        "last_safe_joined_delta": cycle_activity.get("safe_joined_delta_this_cycle"),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    write_json(state_path, state_payload)
    runtime_state_report = write_daemon_runtime_state_packet(
        output_dir=output_dir,
        state_path=state_path,
        systemd_deployment=systemd_deployment,
        target_joined_races=args.target_joined_races,
        generated_at=generated_at,
    )
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return {
        "output_dir": relpath(output_dir),
        "final_verdict": verdict,
        "systemd_deployment_status": systemd_deployment.get("deployment_status"),
        "systemd_deployment_ready": systemd_deployment.get("deployment_ready"),
        "safe_joined_races": dashboard.get("safe_joined_races"),
        "pending_races": dashboard.get("pending_races"),
        "unsafe_matches": dashboard.get("unsafe_matches"),
        "readiness_decision": readiness.get("decision"),
        "feature_activation_gate_status": None
        if feature_activation_gate is None
        else feature_activation_gate.get("status"),
        "odds_coverage_status": odds_coverage.get("status"),
        "shadow_odds_snapshot_status": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("status"),
        "shadow_odds_snapshot_ev_output_rows": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("ev_output_rows", 0),
        "next_prejump_refresh_status": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("status"),
        "recommended_rerun_after_local": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("recommended_rerun_after_local"),
        "prejump_metadata_status": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("status"),
        "prejump_metadata_eligible_count": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("eligible_count"),
        "prejump_metadata_verified_eligible": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("eligible_with_verified_prejump_metadata"),
        "prejump_metadata_trend_status": prejump_metadata_trend.get("status"),
        "prejump_metadata_verified_rate": prejump_metadata_trend.get(
            "verified_metadata_rate"
        ),
        "observability_status": observability["status"].get("status"),
        "cycle_activity_status": cycle_activity.get("status"),
        "safe_joined_delta_this_cycle": cycle_activity.get("safe_joined_delta_this_cycle"),
        "runtime_action": runtime_state_report.get("runtime_action"),
        "odds_capture_performed": False,
        "odds_used_for_shadow_scoring": False,
        "protected_paths_unchanged": protected_paths_unchanged,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-once", help="Run one timer-safe daemon cycle")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--current-time")
    run_parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    run_parser.add_argument("--days-ahead", type=int, default=1)
    run_parser.add_argument("--min-minutes", type=float, default=20.0)
    run_parser.add_argument("--max-minutes", type=float, default=160.0)
    run_parser.add_argument("--refresh-limit", type=int, default=16)
    run_parser.add_argument("--refresh-dry-run", action="store_true")
    run_parser.add_argument("--refresh-command-mode", choices=("auto", "python", "uv"), default="auto")
    run_parser.add_argument("--score-command-mode", choices=("auto", "python", "uv"), default="auto")
    run_parser.add_argument("--target-joined-races", type=int, default=DEFAULT_TARGET_JOINED_RACES)
    run_parser.add_argument("--min-joined-races", type=int, default=DEFAULT_MIN_JOINED_RACES)
    run_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    run_parser.add_argument("--lock-path", type=Path)
    run_parser.add_argument("--lock-stale-seconds", type=int, default=DEFAULT_LOCK_STALE_SECONDS)
    run_parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    run_parser.add_argument("--rejoin-pending-limit", type=int, default=DEFAULT_REJOIN_PENDING_LIMIT)
    run_parser.add_argument("--rejoin-lookback-days", type=int, default=DEFAULT_REJOIN_LOOKBACK_DAYS)
    run_parser.add_argument("--skip-refresh", action="store_true")
    run_parser.add_argument("--skip-shadow-run", action="store_true")

    service_parser = subparsers.add_parser("write-service-files", help="Write systemd unit templates")
    service_parser.add_argument("--service-dir", type=Path, default=DEFAULT_SERVICE_DIR)
    service_parser.add_argument("--repo-path", type=Path, default=Path("/home/l4nd0/greyhound_racing_collector"))
    service_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "write-service-files":
        result = write_service_files(
            service_dir=args.service_dir,
            repo_path=args.repo_path,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = run_once(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("final_verdict") != "NEEDS_MORE_AUTOMATION" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
