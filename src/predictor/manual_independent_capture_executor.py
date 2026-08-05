"""Fixture-only executor for one isolated manual research capture.

The executor owns process, lock, path, timeout, and terminal-artifact control.
It deliberately has no live fetch implementation, database argument, retry
surface, autonomous lock locator, or persistence-capable dependency.  A caller
must supply one reviewed fixture child command.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import json
import math
import os
import signal
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from scripts.refresh_prejump_upcoming import stable_race_id, stable_race_id_variants
from src.predictor.manual_independent_capture import (
    ARTIFACT_PATH_BY_ROLE,
    AUTHORITY_PROFILE,
    CONTRACT_VERSION,
    DOWNSTREAM_ADMISSIBILITY,
    SAFETY_FIELDS,
    SOURCE_PATH_BY_CLASS,
    TERMINAL_ARTIFACT_SCHEMA_VERSION,
    TERMINAL_STATUS_BY_FAILURE_CODE,
    canonical_bytes,
    canonical_sha256,
    validate_config,
    validate_terminal_artifact,
)
from src.predictor.on_demand import (
    PredictionBlocked,
    canonical_runner_set,
    sealed_runner_set_sha256,
    sha256_bytes,
)
from utils.csv_metadata import (
    canonical_thedogs_race_identity,
    canonical_thedogs_venue_identity,
)

CHILD_SCHEMA_VERSION = "manual_independent_capture_child_fixture_v2"
TERMINAL_FILENAME = "terminal.json"
_MAX_CHILD_OUTPUT_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_POLL_SECONDS = 0.01


class CancellationToken(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class FixtureChildLaunch:
    requested_race_url: str
    selected_race: Mapping[str, Any]
    browser_profile: Path
    run_dir: Path


@dataclass(frozen=True)
class CleanupProof:
    pid: int | None
    pgid: int | None
    reason: str
    term_sent: bool
    kill_sent: bool
    leader_reaped: bool
    process_group_absent: bool
    confirmed: bool


@dataclass(frozen=True)
class SourceResponseProof:
    final_url: str
    status_code: int
    content_type: str
    body_sha256: str


@dataclass(frozen=True)
class ManualCaptureExecution:
    artifact: Mapping[str, Any]
    terminal_path: Path
    run_dir: Path
    pid: int | None
    pgid: int | None
    cleanup: CleanupProof
    source_response: SourceResponseProof | None


def _aware_seconds(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.replace(microsecond=0)


def _stamp(value: datetime) -> str:
    return _aware_seconds(value, "timestamp").isoformat()


def _is_safe_directory(path: Path, parent: Path | None = None) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return False
    if parent is not None:
        try:
            path.resolve(strict=True).relative_to(parent.resolve(strict=True))
        except (OSError, ValueError):
            return False
    return True


def _ensure_directory(path: Path, *, parent: Path | None = None) -> None:
    if not path.exists():
        os.mkdir(path, 0o700)
    if not _is_safe_directory(path, parent):
        raise ValueError(f"UNSAFE_PATH:{path.name}")


def _prepare_roots(config: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    paths = config["paths"]
    operations_root = Path(paths["operations_root"])
    manual_root = Path(paths["manual_root"])
    runs_root = Path(paths["runs_root"])
    browser_profile = Path(paths["browser_profile"])
    manual_lock = Path(paths["manual_lock"])
    if not _is_safe_directory(operations_root):
        raise ValueError("UNSAFE_PATH:operations_root")
    _ensure_directory(manual_root, parent=operations_root)
    _ensure_directory(runs_root, parent=manual_root)
    return manual_root, runs_root, browser_profile, manual_lock


def _open_manual_lock(path: Path, manual_root: Path) -> int:
    if path.parent != manual_root:
        raise ValueError("UNSAFE_PATH:manual_lock")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("UNSAFE_PATH:manual_lock")
    return descriptor


def _new_run_dir(runs_root: Path, run_id: str) -> Path:
    run_dir = runs_root / run_id
    os.mkdir(run_dir, 0o700)
    if not _is_safe_directory(run_dir, runs_root):
        raise ValueError("UNSAFE_PATH:run_dir")
    return run_dir


def _write_once(run_dir: Path, relative: str, raw: bytes) -> Path:
    member = Path(relative)
    if (
        not relative
        or member.is_absolute()
        or member.as_posix() != relative
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise ValueError("UNSAFE_PATH:artifact")
    parent = run_dir
    for part in member.parts[:-1]:
        parent = parent / part
        if not parent.exists():
            os.mkdir(parent, 0o700)
        if not _is_safe_directory(parent, run_dir):
            raise ValueError("UNSAFE_PATH:artifact_parent")
    target = run_dir / member
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def _validated_race(url: str, value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "url",
        "race_id",
        "race_date",
        "venue",
        "venue_slug",
        "race_number",
        "scheduled_start",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("url") != url:
        raise ValueError("EXACT_RACE_INVALID")
    race = deepcopy(dict(value))
    identity = canonical_thedogs_race_identity(url)
    if identity is None or identity.get("canonical_url") != url:
        raise ValueError("EXACT_RACE_INVALID")
    venue = canonical_thedogs_venue_identity(race.get("venue"))
    url_venue = canonical_thedogs_venue_identity(identity.get("venue_slug"))
    projection = {
        "race_number": race.get("race_number"),
        "venue": race.get("venue"),
        "race_date": race.get("race_date"),
        "url": race.get("url"),
    }
    try:
        scheduled = datetime.fromisoformat(str(race["scheduled_start"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("EXACT_RACE_INVALID") from exc
    if (
        isinstance(race.get("race_number"), bool)
        or not isinstance(race.get("race_number"), int)
        or identity.get("race_date") != race.get("race_date")
        or identity.get("race_number") != race.get("race_number")
        or identity.get("venue_slug") != race.get("venue_slug")
        or venue is None
        or venue != url_venue
        or race.get("venue") != url_venue
        or race.get("race_id") != stable_race_id(projection)
        or race.get("race_id") not in stable_race_id_variants(projection)
        or scheduled.tzinfo is None
        or scheduled.utcoffset() is None
        or scheduled.isoformat() != race.get("scheduled_start")
        or scheduled.date().isoformat() != race.get("race_date")
    ):
        raise ValueError("EXACT_RACE_INVALID")
    return race


def _request(
    *,
    request_id: str,
    submitted_at: datetime,
    requested_race_url: str,
    selected_race: Mapping[str, Any] | None,
    minimum_margin: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "requested_at": _stamp(submitted_at),
        "requested_race_url": requested_race_url,
        "selected_race": deepcopy(selected_race),
        "minimum_prejump_margin_seconds": minimum_margin,
        "attempt_authority": "one_attempt",
        "manual_concurrency": "one_manual_run",
        "safety": dict(SAFETY_FIELDS),
    }


def _group_absent(pgid: int, killpg: Callable[[int, int], None]) -> bool:
    try:
        killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False


def _cleanup_process_group(
    process: subprocess.Popen[bytes],
    *,
    pgid: int,
    reason: str,
    deadline: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    killpg: Callable[[int, int], None],
) -> CleanupProof:
    term_sent = False
    kill_sent = False

    def absent() -> bool:
        return _group_absent(pgid, killpg)

    if not absent():
        try:
            killpg(pgid, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass
        remaining = max(0.0, deadline - monotonic())
        term_deadline = min(deadline, monotonic() + min(1.0, remaining / 2.0))
        while monotonic() < term_deadline and not absent():
            sleep(min(_POLL_SECONDS, max(0.0, term_deadline - monotonic())))
        if not absent():
            try:
                killpg(pgid, signal.SIGKILL)
                kill_sent = True
            except ProcessLookupError:
                pass

    wait_remaining = max(0.0, deadline - monotonic())
    try:
        process.wait(timeout=wait_remaining)
    except (subprocess.TimeoutExpired, OSError):
        pass
    while monotonic() < deadline and not absent():
        sleep(min(_POLL_SECONDS, max(0.0, deadline - monotonic())))
        try:
            process.poll()
        except OSError:
            pass
    try:
        leader_reaped = process.poll() is not None
    except OSError:
        leader_reaped = False
    process_group_absent = absent()
    return CleanupProof(
        pid=process.pid,
        pgid=pgid,
        reason=reason,
        term_sent=term_sent,
        kill_sent=kill_sent,
        leader_reaped=leader_reaped,
        process_group_absent=process_group_absent,
        confirmed=leader_reaped and process_group_absent,
    )


def _no_process_cleanup(reason: str) -> CleanupProof:
    return CleanupProof(
        pid=None,
        pgid=None,
        reason=reason,
        term_sent=False,
        kill_sent=False,
        leader_reaped=True,
        process_group_absent=True,
        confirmed=True,
    )


def _child_value(
    raw: bytes, *, expected_url: str, expected_race_sha: str
) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_CHILD_OUTPUT_BYTES:
        raise ValueError("SOURCE_MALFORMED")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("SOURCE_MALFORMED") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "requested_race_url",
        "race_identity_sha256",
        "runners",
        "source",
    }:
        raise ValueError("SOURCE_MALFORMED")
    if (
        value["schema_version"] != CHILD_SCHEMA_VERSION
        or value["requested_race_url"] != expected_url
        or value["race_identity_sha256"] != expected_race_sha
        or not isinstance(value["runners"], list)
        or not isinstance(value["source"], Mapping)
        or set(value["source"]) != {
            "content_class",
            "source_timestamp",
            "final_url",
            "status_code",
            "content_type",
            "bytes_base64",
        }
        or value["source"]["content_class"] not in SOURCE_PATH_BY_CLASS
        or not isinstance(value["source"]["source_timestamp"], str)
        or value["source"]["final_url"] != expected_url
        or value["source"]["status_code"] != 200
        or isinstance(value["source"]["status_code"], bool)
        or not isinstance(value["source"]["content_type"], str)
        or not value["source"]["content_type"]
        or len(value["source"]["content_type"]) > 256
        or value["source"]["content_type"]
        != value["source"]["content_type"].strip()
        or any(
            not 32 <= ord(character) <= 126
            for character in value["source"]["content_type"]
        )
        or not isinstance(value["source"]["bytes_base64"], str)
    ):
        raise ValueError("SOURCE_MALFORMED")
    runners = value["runners"]
    if not 1 <= len(runners) <= 10:
        raise ValueError("SOURCE_MALFORMED")
    canonical_runners = []
    for row in runners:
        if not isinstance(row, Mapping) or set(row) != {
            "box_number",
            "display_name",
            "identity",
            "source_native_runner_id",
            "decimal_odds",
        }:
            raise ValueError("SOURCE_MALFORMED")
        native_id = row["source_native_runner_id"]
        odds = row["decimal_odds"]
        if (
            isinstance(row["box_number"], bool)
            or not isinstance(row["box_number"], int)
            or not 1 <= row["box_number"] <= 10
            or not isinstance(row["display_name"], str)
            or not row["display_name"]
            or row["display_name"] != row["display_name"].strip()
            or not isinstance(row["identity"], str)
            or not row["identity"]
            or row["identity"] != row["identity"].strip()
            or row["identity"] != row["identity"].upper()
            or not row["identity"].isascii()
            or (
                native_id is not None
                and (
                    not isinstance(native_id, str)
                    or not native_id
                    or native_id != native_id.strip()
                )
            )
            or isinstance(odds, bool)
            or not isinstance(odds, (int, float))
            or not math.isfinite(float(odds))
            or odds <= 1
        ):
            raise ValueError("SOURCE_MALFORMED")
        canonical_runners.append(
            {key: item for key, item in row.items() if key != "decimal_odds"}
        )
    try:
        canonical_runner_set(canonical_runners, "fixture_child.runners")
    except PredictionBlocked as exc:
        raise ValueError("SOURCE_MALFORMED") from exc
    try:
        source_bytes = base64.b64decode(value["source"]["bytes_base64"], validate=True)
        source_time = datetime.fromisoformat(value["source"]["source_timestamp"])
    except (ValueError, TypeError, binascii.Error) as exc:
        raise ValueError("SOURCE_MALFORMED") from exc
    if (
        not source_bytes
        or len(source_bytes) > _MAX_SOURCE_BYTES
        or source_time.tzinfo is None
        or source_time.utcoffset() is None
        or source_time.isoformat() != value["source"]["source_timestamp"]
    ):
        raise ValueError("SOURCE_MALFORMED")
    return {
        "runners": deepcopy(runners),
        "source_content_class": value["source"]["content_class"],
        "source_timestamp": value["source"]["source_timestamp"],
        "source_bytes": source_bytes,
        "source_response": SourceResponseProof(
            final_url=value["source"]["final_url"],
            status_code=value["source"]["status_code"],
            content_type=value["source"]["content_type"],
            body_sha256=sha256_bytes(source_bytes),
        ),
    }


def _terminal_artifact(
    *,
    config: Mapping[str, Any],
    forbidden_paths: Mapping[str, str],
    request: Mapping[str, Any],
    run_id: str,
    source_commit: str,
    source_tree: str,
    model_bytes: bytes,
    readiness_at: datetime,
    deadline_at: datetime,
    terminal_at: datetime,
    cleanup_deadline_at: datetime | None,
    cancel_requested_at: datetime | None,
    readiness_margin: int | None,
    capture_at: datetime | None,
    failure_code: str | None,
    source_attempt_count: int,
    child: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    status = (
        "CAPTURE_READY"
        if failure_code is None
        else TERMINAL_STATUS_BY_FAILURE_CODE[failure_code]
    )
    selected_race = request["selected_race"]
    capture_rows = [] if child is None or failure_code is not None else child["runners"]
    capture_timestamp = None if not capture_rows else _stamp(capture_at)  # type: ignore[arg-type]
    capture_margin = None
    if capture_at is not None and capture_rows and selected_race is not None:
        scheduled = datetime.fromisoformat(selected_race["scheduled_start"])
        capture_margin = int((scheduled - capture_at).total_seconds())

    config_raw = canonical_bytes(config)
    members: dict[str, bytes] = {
        ARTIFACT_PATH_BY_ROLE["config"]: config_raw,
        ARTIFACT_PATH_BY_ROLE["model"]: model_bytes,
    }
    source_files: list[dict[str, Any]] = []
    runner_sha: str | None = None
    odds_sha: str | None = None
    race_sha = None if selected_race is None else canonical_sha256(selected_race)
    if capture_rows:
        source_class = child["source_content_class"]  # type: ignore[index]
        source_path = SOURCE_PATH_BY_CLASS[source_class]
        source_raw = child["source_bytes"]  # type: ignore[index]
        members[source_path] = source_raw
        capture_value = {"runner_set": capture_rows}
        members[ARTIFACT_PATH_BY_ROLE["capture"]] = canonical_bytes(capture_value)
        source_files = [
            {
                "path": source_path,
                "content_class": source_class,
                "outcome_scope": "target_same_future_outcomes_excluded",
                "race_url": selected_race["url"],
                "race_identity_sha256": race_sha,
                "source_timestamp": child["source_timestamp"],  # type: ignore[index]
                "bytes": len(source_raw),
                "sha256": sha256_bytes(source_raw),
            }
        ]
        canonical_runners = [
            {key: item for key, item in row.items() if key != "decimal_odds"}
            for row in capture_rows
        ]
        runner_sha = sealed_runner_set_sha256(selected_race, canonical_runners)
        odds_sha = canonical_sha256(
            {
                "capture_timestamp": capture_timestamp,
                "odds": [
                    {
                        "box_number": row["box_number"],
                        "decimal_odds": row["decimal_odds"],
                    }
                    for row in capture_rows
                ],
            }
        )

    artifact_hashes = []
    roles = ("capture", "config", "model") if capture_rows else ("config", "model")
    for role in roles:
        path = ARTIFACT_PATH_BY_ROLE[role]
        raw = members[path]
        artifact_hashes.append(
            {"role": role, "path": path, "bytes": len(raw), "sha256": sha256_bytes(raw)}
        )

    artifact = {
        "schema_version": TERMINAL_ARTIFACT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "safety": dict(SAFETY_FIELDS),
        "authority_profile": AUTHORITY_PROFILE,
        "request": deepcopy(dict(request)),
        "timing": {
            "submitted_at": request["requested_at"],
            "readiness_checked_at": _stamp(readiness_at),
            "deadline_at": _stamp(deadline_at),
            "cleanup_deadline_at": (
                None if cleanup_deadline_at is None else _stamp(cleanup_deadline_at)
            ),
            "capture_timestamp": capture_timestamp,
            "readiness_prejump_margin_seconds": readiness_margin,
            "capture_prejump_margin_seconds": capture_margin,
            "cancel_requested_at": (
                None if cancel_requested_at is None else _stamp(cancel_requested_at)
            ),
            "terminal_at": _stamp(terminal_at),
        },
        "attempt": {"attempt_count": 1, "source_attempt_count": source_attempt_count},
        "terminal": {"status": status, "failure_code": failure_code},
        "provenance": {
            "source_commit": source_commit,
            "source_tree": source_tree,
            "config_sha256": canonical_sha256(config),
            "model_sha256": sha256_bytes(model_bytes),
            "request_sha256": canonical_sha256(request),
            "race_identity_sha256": race_sha,
            "runner_set_sha256": runner_sha,
            "odds_sha256": odds_sha,
            "source_files": source_files,
            "artifact_hashes": artifact_hashes,
        },
        "capture": {"runner_set": capture_rows},
        "closure": {
            "bundle_closed": True,
            "closed_at": _stamp(terminal_at),
            "phase7_accessed": False,
            "outcome_accessed": False,
            "canonical_write_claimed": False,
            "downstream_admissibility": DOWNSTREAM_ADMISSIBILITY,
        },
    }
    validated = validate_terminal_artifact(
        artifact,
        config=config,
        forbidden_paths=forbidden_paths,
        member_bytes=members,
        expected_source_commit=source_commit,
        expected_source_tree=source_tree,
        expected_model_sha256=sha256_bytes(model_bytes),
        expected_source_files=deepcopy(source_files),
        expected_runner_set_sha256=runner_sha,
        expected_odds_sha256=odds_sha,
        expected_run_id=run_id,
        expected_request_id=request["request_id"],
        expected_request_sha256=canonical_sha256(request),
        seen_run_ids=set(),
        seen_request_ids=set(),
        seen_request_sha256s=set(),
    )
    return validated, members


def _emit(run_dir: Path, artifact: Mapping[str, Any], members: Mapping[str, bytes]) -> Path:
    for path in sorted(members):
        _write_once(run_dir, path, members[path])
    return _write_once(run_dir, TERMINAL_FILENAME, canonical_bytes(artifact))


def execute_manual_capture_fixture(
    *,
    config: Mapping[str, Any],
    forbidden_paths: Mapping[str, str],
    requested_race_url: str,
    selected_race: Mapping[str, Any],
    model_bytes: bytes,
    source_commit: str,
    source_tree: str,
    fixture_child_command: Callable[[FixtureChildLaunch], Sequence[str]],
    cancellation_token: CancellationToken | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    killpg: Callable[[int, int], None] = os.killpg,
    uuid4: Callable[[], uuid.UUID] = uuid.uuid4,
) -> ManualCaptureExecution:
    """Run one fixture child and emit one validator-accepted terminal bundle."""

    validated_config = validate_config(config, forbidden_paths=forbidden_paths)
    if not isinstance(model_bytes, bytes) or not model_bytes:
        raise ValueError("SOURCE_MALFORMED:model_bytes")
    submitted_at = _aware_seconds(now(), "submitted_at")
    request_id = str(uuid4())
    run_id = str(uuid4())
    minimum_margin = validated_config["timing"]["minimum_prejump_margin_seconds"]
    hard_timeout = validated_config["timing"]["hard_timeout_seconds"]
    cleanup_grace = validated_config["timing"]["cancellation_grace_seconds"]

    manual_root, runs_root, browser_profile, manual_lock = _prepare_roots(
        validated_config
    )
    lock_descriptor: int | None = None
    lock_owned = False
    process: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    deadline_mono: float | None = None
    try:
        try:
            race = _validated_race(requested_race_url, selected_race)
            exact_race_valid = True
        except ValueError:
            race = None
            exact_race_valid = False

        lock_descriptor = _open_manual_lock(manual_lock, manual_root)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_owned = True
        except BlockingIOError:
            lock_owned = False

        run_dir = _new_run_dir(runs_root, run_id)
        readiness_at = _aware_seconds(now(), "readiness_checked_at")
        deadline_at = readiness_at + timedelta(seconds=hard_timeout)
        deadline_mono = monotonic() + hard_timeout
        request = _request(
            request_id=request_id,
            submitted_at=submitted_at,
            requested_race_url=requested_race_url,
            selected_race=race,
            minimum_margin=minimum_margin,
        )

        failure_code: str | None = None
        source_attempt_count = 0
        capture_at: datetime | None = None
        cleanup_deadline_at: datetime | None = None
        cancel_requested_at: datetime | None = None
        child: Mapping[str, Any] | None = None
        cleanup = _no_process_cleanup("no_child_launched")
        readiness_margin: int | None = None

        if not exact_race_valid:
            failure_code = "EXACT_RACE_INVALID"
        elif not lock_owned:
            scheduled = datetime.fromisoformat(race["scheduled_start"])
            readiness_margin = int((scheduled - readiness_at).total_seconds())
            failure_code = "MANUAL_BUSY"
        else:
            _ensure_directory(browser_profile, parent=manual_root)
            scheduled = datetime.fromisoformat(race["scheduled_start"])
            readiness_margin = int((scheduled - readiness_at).total_seconds())
            launch = FixtureChildLaunch(
                requested_race_url=requested_race_url,
                selected_race=deepcopy(race),
                browser_profile=browser_profile,
                run_dir=run_dir,
            )
            command = list(fixture_child_command(launch))
            if not command or any(not isinstance(item, str) or not item for item in command):
                raise ValueError("SOURCE_MALFORMED:fixture_child_command")
            if cancellation_token is not None and cancellation_token.is_set():
                cancel_requested_at = readiness_at
                cleanup_deadline_at = min(
                    deadline_at, cancel_requested_at + timedelta(seconds=cleanup_grace)
                )
                failure_code = "CANCELLED"
            elif readiness_margin < minimum_margin:
                failure_code = "INSUFFICIENT_PREJUMP_MARGIN"
            else:
                child_env = {
                    key: value
                    for key, value in os.environ.items()
                    if key in {"PATH", "PYTHONPATH", "LANG", "LC_ALL", "TMPDIR"}
                }
                child_env.update(
                    {
                        "MANUAL_CAPTURE_PROFILE": str(browser_profile),
                        "MANUAL_CAPTURE_RUN_DIR": str(run_dir),
                        "MANUAL_CAPTURE_EXACT_URL": requested_race_url,
                        "MANUAL_CAPTURE_RACE_ID": race["race_id"],
                        "PYTHONDONTWRITEBYTECODE": "1",
                    }
                )
                # Deliberately perform no other preparation between this final
                # wall-clock margin check and the single Popen call.
                readiness_at = _aware_seconds(now(), "readiness_checked_at")
                deadline_at = readiness_at + timedelta(seconds=hard_timeout)
                deadline_mono = monotonic() + hard_timeout
                readiness_margin = int((scheduled - readiness_at).total_seconds())
                if readiness_margin < minimum_margin:
                    failure_code = "INSUFFICIENT_PREJUMP_MARGIN"
                else:
                    process = popen(
                        command,
                        cwd=run_dir,
                        env=child_env,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                    source_attempt_count = 1
                    pgid = process.pid
                    stdout = b""
                    stop_reason: str | None = None
                    cleanup_budget = min(
                        float(cleanup_grace), max(_POLL_SECONDS, hard_timeout / 2.0)
                    )
                    attempt_deadline_mono = deadline_mono - cleanup_budget
                    while True:
                        if cancellation_token is not None and cancellation_token.is_set():
                            stop_reason = "external_cancellation"
                            cancel_requested_at = _aware_seconds(
                                now(), "cancel_requested_at"
                            )
                            cleanup_deadline_at = min(
                                deadline_at,
                                cancel_requested_at + timedelta(seconds=cleanup_grace),
                            )
                            break
                        remaining = attempt_deadline_mono - monotonic()
                        if remaining <= 0:
                            stop_reason = "monotonic_timeout"
                            cleanup_deadline_at = deadline_at
                            break
                        try:
                            stdout, _stderr = process.communicate(
                                timeout=min(_POLL_SECONDS, remaining)
                            )
                            break
                        except subprocess.TimeoutExpired:
                            continue

                    if stop_reason is not None:
                        cleanup_mono = (
                            deadline_mono
                            if stop_reason == "monotonic_timeout"
                            else min(deadline_mono, monotonic() + cleanup_grace)
                        )
                        cleanup = _cleanup_process_group(
                            process,
                            pgid=pgid,
                            reason=stop_reason,
                            deadline=cleanup_mono,
                            monotonic=monotonic,
                            sleep=sleep,
                            killpg=killpg,
                        )
                        failure_code = (
                            "PROCESS_REAP_UNCONFIRMED"
                            if not cleanup.confirmed
                            else "TIMED_OUT"
                            if stop_reason == "monotonic_timeout"
                            else "CANCELLED"
                        )
                        if failure_code == "TIMED_OUT":
                            while monotonic() < deadline_mono:
                                sleep(
                                    min(
                                        _POLL_SECONDS,
                                        max(0.0, deadline_mono - monotonic()),
                                    )
                                )
                    else:
                        capture_at = _aware_seconds(now(), "capture_timestamp")
                        cleanup = _cleanup_process_group(
                            process,
                            pgid=pgid,
                            reason="child_completed",
                            deadline=deadline_mono,
                            monotonic=monotonic,
                            sleep=sleep,
                            killpg=killpg,
                        )
                        if not cleanup.confirmed:
                            failure_code = "PROCESS_REAP_UNCONFIRMED"
                            cleanup_deadline_at = deadline_at
                            capture_at = None
                        elif process.returncode != 0 or len(stdout) > _MAX_CHILD_OUTPUT_BYTES:
                            failure_code = "SOURCE_MALFORMED"
                            capture_at = None
                        else:
                            try:
                                child = _child_value(
                                    stdout,
                                    expected_url=requested_race_url,
                                    expected_race_sha=canonical_sha256(race),
                                )
                            except ValueError:
                                failure_code = "SOURCE_MALFORMED"
                                capture_at = None
                            if child is not None:
                                source_at = datetime.fromisoformat(
                                    child["source_timestamp"]
                                )
                                if (
                                    not readiness_at <= source_at <= capture_at
                                    or source_at >= scheduled
                                ):
                                    failure_code = "SOURCE_MALFORMED"
                                    capture_at = None
                                    child = None
                            if (
                                child is not None
                                and int((scheduled - capture_at).total_seconds())
                                < minimum_margin
                            ):
                                failure_code = "SOURCE_TIMEOUT"
                                capture_at = None
                                child = None

        if failure_code == "PROCESS_REAP_UNCONFIRMED":
            cleanup_deadline_at = cleanup_deadline_at or deadline_at
            terminal_at = cleanup_deadline_at
        elif failure_code == "TIMED_OUT":
            cleanup_deadline_at = deadline_at
            terminal_at = deadline_at
        else:
            terminal_at = _aware_seconds(now(), "terminal_at")

        artifact, members = _terminal_artifact(
            config=validated_config,
            forbidden_paths=forbidden_paths,
            request=request,
            run_id=run_id,
            source_commit=source_commit,
            source_tree=source_tree,
            model_bytes=model_bytes,
            readiness_at=readiness_at,
            deadline_at=deadline_at,
            terminal_at=terminal_at,
            cleanup_deadline_at=cleanup_deadline_at,
            cancel_requested_at=cancel_requested_at,
            readiness_margin=readiness_margin,
            capture_at=capture_at,
            failure_code=failure_code,
            source_attempt_count=source_attempt_count,
            child=child,
        )
        terminal_path = _emit(run_dir, artifact, members)
        return ManualCaptureExecution(
            artifact=artifact,
            terminal_path=terminal_path,
            run_dir=run_dir,
            pid=None if process is None else process.pid,
            pgid=pgid,
            cleanup=cleanup,
            source_response=(
                None if child is None else child["source_response"]
            ),
        )
    except BaseException:
        if process is not None and pgid is not None:
            try:
                cleanup_limit = (
                    deadline_mono if deadline_mono is not None else monotonic()
                )
                _cleanup_process_group(
                    process,
                    pgid=pgid,
                    reason="unexpected_exception",
                    deadline=cleanup_limit,
                    monotonic=monotonic,
                    sleep=sleep,
                    killpg=killpg,
                )
            except BaseException:  # noqa: BLE001, S110
                pass
        raise
    finally:
        if lock_descriptor is not None:
            if lock_owned:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
