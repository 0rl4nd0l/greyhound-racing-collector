"""Bounded collector-owned capture for one exact manual prediction request."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config.venue_mapping import normalize_venue
from race_collection.manual_prediction_collector_request import (
    RECEIPT_READY,
    CollectorRequest,
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
    canonical_bytes,
    runner_set_sha256,
)
from src.predictor.on_demand import (
    PredictionBlocked,
    normalize_validation_receipt,
    receipt_from_handoff,
    sha256_bytes,
)
from utils.runner_completeness import normalise_runner_name

ROOT = Path(__file__).resolve().parents[1]
HANDOFF_SCHEMA = "on_demand_verified_collector_capture_v2"
CURRENT_RACE_INDEX_V1_SCHEMA = "collector_current_race_index_v1"
CURRENT_RACE_INDEX_SCHEMA = "collector_current_race_index_v2"
CURRENT_RACE_INDEX_FILENAME = "manual_prediction_current_race_index.json"
CURRENT_RACE_INDEX_PUBLICATION_FILENAME = (
    "manual_prediction_current_race_index.publication.json"
)
ODDS_CAPTURE_ONLY_STATE_FILENAME = "odds_capture_state.json"
ODDS_CAPTURE_ONLY_REPORT_FILENAME = "odds_capture_only_daemon_report.json"
MAX_CURRENT_INDEX_RACES = 32
MAX_CURRENT_INDEX_BYTES = 2 * 1024 * 1024
CANONICAL_LOCK_RELATIVE_PATH = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "shadow_autopilot_daemon_runtime/shadow_autopilot.lock"
)


class CaptureCancelled(RuntimeError):
    """The caller cancelled capture before the collector completed."""


class CollectorBusy(RuntimeError):
    """The canonical collector lock is already owned and is never stolen."""

    def __init__(self, evidence: Mapping[str, Any]) -> None:
        super().__init__("collector_lock_busy_no_steal")
        self.evidence = dict(evidence)
        self.payload = self.evidence


class CaptureOneRejected(PredictionBlocked):
    """A stable collector capture-one terminal result."""

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code, **details)


@dataclass(frozen=True, slots=True)
class VerifiedCurrentRaceIndex:
    """Immutable operator view of the exact retained, verified v2 snapshot."""

    schema_version: str
    run_id: str
    source_generated_at: str
    packet_sha256: str
    packet_bytes: bytes
    races: tuple[Mapping[str, Any], ...]
    source_refresh_report_path: str
    source_refresh_report_sha256: str
    publication_sha256: str
    state_sha256: str
    report_sha256: str


class _DiscoveryTimedOut(BaseException):
    pass


@dataclass(frozen=True, slots=True)
class LatencyBudget:
    discovery_seconds: float
    lock_seconds: float
    capture_seconds: float
    validation_seconds: float
    scoring_seconds: float
    safety_seconds: float

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> LatencyBudget:
        expected = {
            "discovery_seconds",
            "lock_seconds",
            "capture_seconds",
            "validation_seconds",
            "scoring_seconds",
            "safety_seconds",
        }
        if set(value) != expected:
            raise PredictionBlocked("LATENCY_BUDGET_INVALID")
        numbers: dict[str, float] = {}
        for key in sorted(expected):
            raw = value[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise PredictionBlocked("LATENCY_BUDGET_INVALID", field=key)
            number = float(raw)
            if not 0 < number <= 300:
                raise PredictionBlocked("LATENCY_BUDGET_INVALID", field=key)
            numbers[key] = number
        return cls(**numbers)

    @property
    def capture_margin_seconds(self) -> float:
        return (
            self.lock_seconds
            + self.capture_seconds
            + self.validation_seconds
            + self.scoring_seconds
            + self.safety_seconds
        )

    @property
    def total_margin_seconds(self) -> float:
        return self.discovery_seconds + self.capture_margin_seconds

    @property
    def reuse_margin_seconds(self) -> float:
        return (
            self.validation_seconds + self.scoring_seconds + self.safety_seconds
        )

    @property
    def post_lock_margin_seconds(self) -> float:
        return self.capture_seconds + self.reuse_margin_seconds

    def pre_fetch_margin_seconds(self, fetch_timeout_seconds: float) -> float:
        if not 0 < fetch_timeout_seconds <= self.capture_seconds:
            raise PredictionBlocked("FETCH_TIMEOUT_EXCEEDS_CAPTURE_BUDGET")
        return fetch_timeout_seconds + self.reuse_margin_seconds


@dataclass(frozen=True, slots=True)
class OwnedCollectorLock:
    path: Path
    run_id: str
    device: int
    inode: int


@dataclass(slots=True)
class CaptureOneDependencies:
    now: Callable[[], datetime]
    refresh_exact: Callable[
        [Mapping[str, Any], Path, datetime], tuple[Path, Path]
    ]
    build_plan_item: Callable[[Path, datetime], Mapping[str, Any]]
    execute_capture_plan: Callable[..., Mapping[str, Any]]
    acquire_lock: Callable[..., Any]
    release_lock: Callable[[Any], None]
    phase_hook: Callable[[str], None] = lambda phase: None


def _read_lock_evidence(lock_path: Path) -> dict[str, Any]:
    try:
        if lock_path.is_symlink():
            raise OSError("lock_is_symlink")
        payload = json.loads(lock_path.read_bytes())
    except (OSError, TypeError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    run_id = str(payload.get("run_id") or "")
    return {
        "lock_path": str(lock_path),
        "lock_owner_run_id": payload.get("run_id"),
        "lock_owner_pid": payload.get("pid"),
        "lock_owner_hostname": payload.get("hostname"),
        "lock_owner_started_at": payload.get("started_at"),
        "lock_owner_output_dir": payload.get("output_dir"),
        "lock_owner_phase": payload.get("phase")
        or ("odds_capture" if run_id.endswith("_odds_capture") else "collector_cycle"),
        "reason": "existing_lock_present_no_steal",
    }


def _unlink_owned_lock(lock: OwnedCollectorLock) -> bool:
    try:
        current = lock.path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if lock.path.is_symlink() or (current.st_dev, current.st_ino) != (
        lock.device,
        lock.inode,
    ):
        return False
    lock.path.unlink()
    return True


def acquire_collector_lock_no_steal(
    lock_path: Path,
    *,
    run_id: str,
    output_dir: Path,
    phase: str = "manual_capture_one",
    acquisition_policy: str = "collector_capture_one_no_steal_v1",
) -> OwnedCollectorLock:
    lock_path = lock_path.absolute()
    if (
        lock_path.is_symlink()
        or lock_path.parent != lock_path.parent.resolve()
        or not lock_path.parent.is_dir()
    ):
        raise CaptureOneRejected("LOCK_PATH_UNSAFE", path=str(lock_path))
    payload = {
        "schema_version": "shadow_autopilot_daemon_lock_v1",
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir.resolve()),
        "phase": phase,
        "acquisition_policy": acquisition_policy,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise CollectorBusy(_read_lock_evidence(lock_path)) from exc
    except OSError as exc:
        raise CaptureOneRejected(
            "LOCK_ACQUIRE_FAILED", error=type(exc).__name__
        ) from exc
    try:
        try:
            opened = os.fstat(descriptor)
        except OSError as exc:
            try:
                opened = os.stat(descriptor)
            except OSError:
                os.close(descriptor)
                raise CaptureOneRejected(
                    "LOCK_ACQUIRE_FAILED",
                    reason="descriptor_identity_unavailable",
                ) from exc
            failed_lock = OwnedCollectorLock(
                path=lock_path,
                run_id=run_id,
                device=opened.st_dev,
                inode=opened.st_ino,
            )
            os.close(descriptor)
            _unlink_owned_lock(failed_lock)
            raise CaptureOneRejected(
                "LOCK_ACQUIRE_FAILED", reason="descriptor_stat_failed"
            ) from exc
        lock = OwnedCollectorLock(
            path=lock_path,
            run_id=run_id,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        return lock
    except CaptureOneRejected:
        raise
    except Exception as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if "lock" in locals():
            _unlink_owned_lock(lock)
        raise CaptureOneRejected(
            "LOCK_ACQUIRE_FAILED",
            reason="descriptor_or_write_failed",
            error=type(exc).__name__,
        ) from exc


def current_race_index_path(state_path: Path) -> Path:
    """Return the fixed collector-owned current-index path for one runtime."""

    return Path(state_path).parent / CURRENT_RACE_INDEX_FILENAME


def _evidence_locator_path(locator: object, *, evidence_root: Path) -> Path:
    """Resolve one producer locator without searching or accepting absolutes."""

    if not isinstance(locator, str) or not locator:
        raise CaptureOneRejected("CURRENT_INDEX_REPORT_INVALID")
    relative = Path(locator)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise CaptureOneRejected("CURRENT_INDEX_REPORT_INVALID")
    root = evidence_root.absolute()
    try:
        root_from_repo = root.relative_to(ROOT.absolute())
    except ValueError:
        root_from_repo = None
    if root_from_repo is not None and relative.parts[: len(root_from_repo.parts)] == root_from_repo.parts:
        resolved = ROOT.absolute() / relative
    else:
        resolved = root / relative
    try:
        resolved.absolute().relative_to(root)
    except ValueError as exc:
        raise CaptureOneRejected("CURRENT_INDEX_REPORT_INVALID") from exc
    return resolved


def _safe_file_bytes(
    path: Path,
    *,
    evidence_root: Path,
    missing_code: str,
) -> bytes:
    return _safe_files_bytes(
        [path], evidence_root=evidence_root, missing_code=missing_code,
    )[0]


def _safe_files_bytes(
    paths: list[Path],
    *,
    evidence_root: Path,
    missing_code: str,
) -> list[bytes]:
    """Read finite regular files while retaining and revalidating their chain."""

    root = evidence_root.absolute()
    try:
        root_named = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(root)) from exc
    if not stat.S_ISDIR(root_named.st_mode):
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(root))
    relatives: list[Path] = []
    for path in paths:
        logical = path.absolute()
        try:
            relative = logical.relative_to(root)
        except ValueError as exc:
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path)) from exc
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path))
        relatives.append(relative)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, directory_flags)
    directory_fds: dict[tuple[str, ...], int] = {(): root_fd}
    directory_stats: dict[tuple[str, ...], os.stat_result] = {}
    file_records: list[tuple[Path, tuple[str, ...], str, int, os.stat_result]] = []
    try:
        root_opened = os.fstat(root_fd)
        if (root_opened.st_dev, root_opened.st_ino) != (root_named.st_dev, root_named.st_ino):
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(root), reason="root_replaced")
        directory_stats[()] = root_opened
        for path, relative in zip(paths, relatives):
            parent_key: tuple[str, ...] = ()
            for component in relative.parts[:-1]:
                child_key = (*parent_key, component)
                if child_key not in directory_fds:
                    parent_fd = directory_fds[parent_key]
                    named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                    child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(named.st_mode)
                        or not stat.S_ISDIR(opened.st_mode)
                        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
                    ):
                        os.close(child_fd)
                        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="directory_replaced")
                    directory_fds[child_key] = child_fd
                    directory_stats[child_key] = opened
                parent_key = child_key
            parent_fd = directory_fds[parent_key]
            name = relative.parts[-1]
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            file_fd = os.open(name, flags, dir_fd=parent_fd)
            opened = os.fstat(file_fd)
            if (
                not stat.S_ISREG(named.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                os.close(file_fd)
                raise CaptureOneRejected(missing_code, path=str(path))
            file_records.append((path, parent_key, name, file_fd, opened))
    except CaptureOneRejected:
        for _, _, _, file_fd, _ in file_records:
            os.close(file_fd)
        for directory_fd in reversed(list(directory_fds.values())):
            os.close(directory_fd)
        raise
    except OSError as exc:
        for _, _, _, file_fd, _ in file_records:
            os.close(file_fd)
        for directory_fd in reversed(list(directory_fds.values())):
            os.close(directory_fd)
        raise CaptureOneRejected(missing_code, path=str(paths[0] if paths else root)) from exc
    try:
        payloads: list[bytes] = []
        for path, parent_key, name, file_fd, opened in file_records:
            if opened.st_size <= 0 or opened.st_size > MAX_CURRENT_INDEX_BYTES:
                raise CaptureOneRejected("CURRENT_INDEX_SIZE_INVALID", path=str(path), size_bytes=opened.st_size, max_bytes=MAX_CURRENT_INDEX_BYTES)
            chunks: list[bytes] = []
            remaining = MAX_CURRENT_INDEX_BYTES + 1
            while remaining:
                chunk = os.read(file_fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            try:
                named_after_read = os.stat(
                    name,
                    dir_fd=directory_fds[parent_key],
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise CaptureOneRejected(
                    "CURRENT_INDEX_PATH_UNSAFE",
                    path=str(path),
                    reason="path_replaced",
                ) from exc
            if (
                not stat.S_ISREG(named_after_read.st_mode)
                or (named_after_read.st_dev, named_after_read.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise CaptureOneRejected(
                    "CURRENT_INDEX_PATH_UNSAFE",
                    path=str(path),
                    reason="path_replaced",
                )
            after = os.fstat(file_fd)
            if (
                not stat.S_ISREG(after.st_mode)
                or _retained_read_identity(after)
                != _retained_read_identity(opened)
            ):
                raise CaptureOneRejected(
                    "CURRENT_INDEX_PATH_UNSAFE",
                    path=str(path),
                    reason="file_mutated",
                )
            if (
                not raw
                or len(raw) > MAX_CURRENT_INDEX_BYTES
                or len(raw) != opened.st_size
            ):
                raise CaptureOneRejected("CURRENT_INDEX_SIZE_INVALID", path=str(path), size_bytes=len(raw), max_bytes=MAX_CURRENT_INDEX_BYTES)
            payloads.append(raw)

        root_after = os.stat(root, follow_symlinks=False)
        root_current = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_after.st_mode)
            or _retained_read_identity(root_after) != _retained_read_identity(root_opened)
            or _retained_read_identity(root_current) != _retained_read_identity(root_opened)
        ):
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(root), reason="root_replaced")
        for key, opened in directory_stats.items():
            current = os.fstat(directory_fds[key])
            if not stat.S_ISDIR(current.st_mode) or _retained_read_identity(current) != _retained_read_identity(opened):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", reason="directory_replaced")
            if key:
                named = os.stat(key[-1], dir_fd=directory_fds[key[:-1]], follow_symlinks=False)
                if not stat.S_ISDIR(named.st_mode) or _retained_read_identity(named) != _retained_read_identity(opened):
                    raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", reason="directory_replaced")
        for path, parent_key, name, file_fd, opened in file_records:
            named = os.stat(name, dir_fd=directory_fds[parent_key], follow_symlinks=False)
            current = os.fstat(file_fd)
            if not stat.S_ISREG(named.st_mode) or not stat.S_ISREG(current.st_mode) or _retained_read_identity(named) != _retained_read_identity(opened) or _retained_read_identity(current) != _retained_read_identity(opened):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="path_replaced")
        return payloads
    except CaptureOneRejected:
        raise
    except OSError as exc:
        raise CaptureOneRejected(missing_code, path=str(path)) from exc
    finally:
        for _, _, _, file_fd, _ in file_records:
            os.close(file_fd)
        for directory_fd in reversed(list(directory_fds.values())):
            os.close(directory_fd)


def _retained_read_identity(value: os.stat_result) -> tuple[int, ...]:
    """Identity plus mutation witnesses used to reject swap-and-restore reads."""

    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _retained_directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    """Directory type and inode identity, excluding mutable child metadata."""

    return (stat.S_IFMT(value.st_mode), value.st_dev, value.st_ino)


class _RetainedSafeFiles:
    """One bounded no-follow snapshot whose descriptors live through validation."""

    def __init__(self, evidence_root: Path) -> None:
        self.root = evidence_root.absolute()
        self.directory_fds: dict[tuple[str, ...], int] = {}
        self.directory_stats: dict[tuple[str, ...], os.stat_result] = {}
        self.files: list[tuple[Path, tuple[str, ...], str, int, os.stat_result]] = []
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            named = os.stat(self.root, follow_symlinks=False)
            root_fd = os.open(self.root, flags)
            opened = os.fstat(root_fd)
        except OSError as exc:
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(self.root)) from exc
        if (
            not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _retained_directory_identity(named)
            != _retained_directory_identity(opened)
        ):
            os.close(root_fd)
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(self.root))
        self.directory_fds[()] = root_fd
        self.directory_stats[()] = opened
        self._directory_flags = flags

    def __enter__(self) -> _RetainedSafeFiles:
        return self

    def __exit__(self, *ignored: object) -> None:
        del ignored
        for _, _, _, descriptor, _ in self.files:
            os.close(descriptor)
        for descriptor in reversed(list(self.directory_fds.values())):
            os.close(descriptor)

    def read(self, path: Path, *, missing_code: str) -> bytes:
        logical = path.absolute()
        try:
            relative = logical.relative_to(self.root)
        except ValueError as exc:
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path)) from exc
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path))
        parent_key: tuple[str, ...] = ()
        try:
            for component in relative.parts[:-1]:
                child_key = (*parent_key, component)
                if child_key not in self.directory_fds:
                    parent_fd = self.directory_fds[parent_key]
                    named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                    child_fd = os.open(component, self._directory_flags, dir_fd=parent_fd)
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(named.st_mode)
                        or not stat.S_ISDIR(opened.st_mode)
                        or _retained_directory_identity(named)
                        != _retained_directory_identity(opened)
                    ):
                        os.close(child_fd)
                        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path))
                    self.directory_fds[child_key] = child_fd
                    self.directory_stats[child_key] = opened
                parent_key = child_key
            name = relative.parts[-1]
            parent_fd = self.directory_fds[parent_key]
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, flags, dir_fd=parent_fd)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(named.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or _retained_read_identity(named) != _retained_read_identity(opened)
            ):
                os.close(descriptor)
                raise CaptureOneRejected(missing_code, path=str(path))
            self.files.append((path, parent_key, name, descriptor, opened))
            if opened.st_size <= 0 or opened.st_size > MAX_CURRENT_INDEX_BYTES:
                raise CaptureOneRejected("CURRENT_INDEX_SIZE_INVALID", path=str(path))
            chunks: list[bytes] = []
            remaining = MAX_CURRENT_INDEX_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            if _retained_read_identity(after) != _retained_read_identity(opened):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="file_mutated")
            if not raw or len(raw) != opened.st_size or len(raw) > MAX_CURRENT_INDEX_BYTES:
                raise CaptureOneRejected("CURRENT_INDEX_SIZE_INVALID", path=str(path))
            return raw
        except CaptureOneRejected:
            raise
        except OSError as exc:
            raise CaptureOneRejected(missing_code, path=str(path)) from exc

    def validate(self) -> None:
        try:
            root_named = os.stat(self.root, follow_symlinks=False)
        except OSError as exc:
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(self.root), reason="root_replaced") from exc
        for path, parent_key, name, descriptor, opened in self.files:
            try:
                named = os.stat(name, dir_fd=self.directory_fds[parent_key], follow_symlinks=False)
            except OSError as exc:
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="path_replaced") from exc
            current = os.fstat(descriptor)
            if (
                not stat.S_ISREG(named.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or _retained_read_identity(named) != _retained_read_identity(opened)
                or _retained_read_identity(current) != _retained_read_identity(opened)
            ):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="path_replaced")
        for key, opened in self.directory_stats.items():
            current = os.fstat(self.directory_fds[key])
            if (
                not stat.S_ISDIR(current.st_mode)
                or _retained_directory_identity(current)
                != _retained_directory_identity(opened)
            ):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", reason="directory_replaced")
            if key:
                try:
                    named = os.stat(key[-1], dir_fd=self.directory_fds[key[:-1]], follow_symlinks=False)
                except OSError as exc:
                    raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", reason="directory_replaced") from exc
            else:
                named = root_named
            if (
                not stat.S_ISDIR(named.st_mode)
                or _retained_directory_identity(named)
                != _retained_directory_identity(opened)
            ):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", reason="directory_replaced")


def _time_matches(value: str, pattern: str) -> bool:
    try:
        datetime.strptime(value, pattern)
    except ValueError:
        return False
    return True


def _normalize_current_index_rows(
    source: Mapping[str, Any],
    *,
    max_races: int,
) -> list[dict[str, Any]]:
    selected = source.get("selected_races")
    selected_count = source.get("selected_count")
    if (
        not isinstance(selected, list)
        or isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or selected_count != len(selected)
        or len(selected) > max_races
    ):
        raise CaptureOneRejected(
            "CURRENT_INDEX_UNBOUNDED",
            race_count=len(selected) if isinstance(selected, list) else None,
            selected_count=selected_count,
            max_races=max_races,
        )
    from scripts.refresh_prejump_upcoming import stable_race_id, stable_race_id_variants
    from utils.csv_metadata import canonical_thedogs_race_identity

    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
    aliases_seen: set[str] = set()
    for raw in selected:
        if not isinstance(raw, Mapping):
            raise CaptureOneRejected("CURRENT_INDEX_INVALID", reason="race_not_mapping")
        race_url = str(raw.get("race_url") or "")
        identity = canonical_thedogs_race_identity(race_url)
        try:
            race_number = int(raw["race_number"])
            jump = datetime.fromisoformat(str(raw["jump_datetime"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureOneRejected(
                "CURRENT_INDEX_INVALID", reason="race_identity_invalid"
            ) from exc
        race_date = str(raw.get("date") or "")
        venue = str(raw.get("venue") or "")
        race_id = str(raw.get("race_id") or "")
        aliases = raw.get("race_id_aliases")
        race_time = jump.strftime("%H:%M")
        supplied_race_time = str(raw.get("race_time") or "").strip()
        try:
            parsed_race_time = next(
                datetime.strptime(supplied_race_time, pattern).time()
                for pattern in ("%H:%M", "%I:%M %p")
                if _time_matches(supplied_race_time, pattern)
            )
        except StopIteration:
            parsed_race_time = None
        if (
            identity is None
            or identity["race_date"] != race_date
            or identity["race_number"] != race_number
            or jump.tzinfo is None
            or jump.utcoffset() is None
            or jump.date().isoformat() != race_date
            or not venue
            or not race_id
            or not isinstance(aliases, list)
            or len(aliases) > 16
            or any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in aliases
            )
            or parsed_race_time is None
            or (parsed_race_time.hour, parsed_race_time.minute)
            != (jump.hour, jump.minute)
        ):
            raise CaptureOneRejected(
                "CURRENT_INDEX_INVALID", reason="race_identity_invalid"
            )
        row = {
            "date": race_date,
            "jump_datetime": jump.isoformat(),
            "race_id": race_id,
            "race_id_aliases": list(aliases),
            "race_number": race_number,
            "race_time": race_time,
            "race_url": race_url,
            "venue": venue,
        }
        alias_set = set(aliases)
        canonical_aliases = sorted(stable_race_id_variants(row))
        if (
            stable_race_id(row) != race_id
            or aliases != canonical_aliases
            or race_id in identities
            or len(alias_set) != len(aliases)
            or alias_set & identities
            or race_id in aliases_seen
            or alias_set & aliases_seen
        ):
            raise CaptureOneRejected(
                "CURRENT_INDEX_INVALID", reason="race_id_mismatch_or_duplicate"
            )
        identities.add(race_id)
        aliases_seen.update(alias_set)
        normalized.append(row)
    return normalized


def _v2_runner_rows(
    race: Mapping[str, Any], source: Mapping[str, Any], *, evidence_root: Path,
    snapshot: _RetainedSafeFiles | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    coverage = source.get("sidecar_metadata_coverage")
    if not isinstance(coverage, Mapping) or coverage.get("schema_version") != "prejump_sidecar_metadata_coverage_v1":
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_coverage_missing")
    matches = [
        item for item in coverage.get("races", [])
        if isinstance(item, Mapping) and item.get("race_url") == race["race_url"]
    ]
    if len(matches) != 1:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_source_misaligned")
    record = matches[0]
    csv_path = Path(str(record.get("csv_path") or ""))
    sidecar_path = Path(str(record.get("sidecar_path") or ""))
    if sidecar_path != csv_path.with_name(csv_path.name + ".metadata.json"):
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", reason="sidecar_not_adjacent")
    if snapshot is None:
        csv_raw, sidecar_raw = _safe_files_bytes(
            [csv_path, sidecar_path], evidence_root=evidence_root,
            missing_code="CURRENT_INDEX_SOURCE_MISSING",
        )
    else:
        csv_raw = snapshot.read(csv_path, missing_code="CURRENT_INDEX_SOURCE_MISSING")
        sidecar_raw = snapshot.read(sidecar_path, missing_code="CURRENT_INDEX_SOURCE_MISSING")
    try:
        sidecar = json.loads(sidecar_raw)
    except json.JSONDecodeError as exc:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID") from exc
    if not isinstance(sidecar, Mapping):
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="sidecar_invalid")
    shadow = sidecar.get("prejump_shadow_metadata")
    if not isinstance(shadow, Mapping) or shadow.get("status") != "PASS" or shadow.get("metadata_is_leakage_safe") is not True:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_source_not_accepted")
    alignment = shadow.get("canonical_final_runner_alignment")
    if not isinstance(alignment, Mapping) or alignment.get("status") != "aligned" or alignment.get("canonical_runner_set_status") != "available":
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_source_not_aligned")
    if (
        shadow.get("source_url") != race["race_url"]
        or shadow.get("race_date") != race["date"]
        or not isinstance(shadow.get("venue"), str)
        or normalize_venue(shadow["venue"]) != normalize_venue(race["venue"])
        or shadow.get("race_number") != race["race_number"]
    ):
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_race_identity_mismatch")
    observed = datetime.fromisoformat(str(shadow.get("metadata_captured_at") or ""))
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_observation_invalid")
    generated = datetime.fromisoformat(str(source.get("generated_at") or ""))
    jump = datetime.fromisoformat(str(race["jump_datetime"]))
    if generated.tzinfo is None or generated.utcoffset() is None:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="source_generated_at_invalid")
    source_distance = abs((generated - observed).total_seconds())
    if source_distance > 1200 or generated >= jump or observed >= jump:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_source_stale_or_postjump")
    participants = shadow.get("runner_box_name_list")
    if not isinstance(participants, list) or not participants:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_set_empty")
    aligned_details = sidecar.get("runner_completeness_after_canonical_alignment")
    if (
        not isinstance(aligned_details, Mapping)
        or aligned_details.get("status") != "COMPLETE"
        or not isinstance(aligned_details.get("participants"), list)
    ):
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_status_missing_or_inactive")
    detailed_participants = aligned_details["participants"]
    detailed_by_runner = {
        (item.get("box_number"), normalise_runner_name(item.get("dog_name"))): item
        for item in detailed_participants
        if isinstance(item, Mapping)
    }

    def admitted_native_id(item: Mapping[str, Any]) -> str | None:
        value = item.get("source_native_runner_id", item.get("runner_id"))
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (str, int))
            or not str(value).strip()
        ):
            raise CaptureOneRejected(
                "CURRENT_INDEX_SOURCE_INVALID", reason="runner_id_invalid"
            )
        return str(value).strip()
    canonical_active_projection = []
    for item in detailed_participants:
        if not isinstance(item, Mapping):
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_status_missing_or_inactive")
        declared = [
            item[key]
            for key in ("scratch_state", "activity_state", "status")
            if key in item
        ]
        if not declared or any(value != "ACTIVE" for value in declared):
            raise CaptureOneRejected(
                "CURRENT_INDEX_SOURCE_INVALID", reason="runner_status_missing_or_inactive"
            )
        canonical_active_projection.append(
            (item.get("box_number"), normalise_runner_name(item.get("dog_name")))
        )
    participant_projection = []
    for item in participants:
        if not isinstance(item, Mapping):
            raise CaptureOneRejected(
                "CURRENT_INDEX_SOURCE_INVALID", reason="runner_status_ambiguous"
            )
        listed_states = [
            item[key]
            for key in ("scratch_state", "activity_state", "status")
            if key in item
        ]
        if any(value != "ACTIVE" for value in listed_states):
            raise CaptureOneRejected(
                "CURRENT_INDEX_SOURCE_INVALID", reason="runner_status_ambiguous"
            )
        participant_projection.append(
            (item.get("box_number"), normalise_runner_name(item.get("dog_name")))
        )
    if (
        canonical_active_projection != participant_projection
        or len(canonical_active_projection) != len(detailed_participants)
    ):
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_status_ambiguous")

    rows: list[dict[str, Any]] = []
    boxes: set[int] = set()
    identities: set[str] = set()
    native_identities: set[str] = set()
    for item in participants:
        if not isinstance(item, Mapping):
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_invalid")
        try:
            box_value = item["box_number"]
            if isinstance(box_value, bool):
                raise ValueError("boolean box")
            box = int(box_value)
            if isinstance(box_value, float) and box_value != box:
                raise ValueError("fractional box")
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_box_invalid") from exc
        name = str(item.get("dog_name") or "").strip()
        identity = normalise_runner_name(name)
        if box <= 0 or not name or box in boxes or identity in identities:
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_duplicate_or_invalid")
        shadow_native_id = admitted_native_id(item)
        detailed = detailed_by_runner.get((box, identity))
        detailed_native_id = admitted_native_id(detailed) if detailed is not None else None
        if (
            shadow_native_id is not None
            and detailed_native_id is not None
            and shadow_native_id != detailed_native_id
        ):
            raise CaptureOneRejected(
                "CURRENT_INDEX_SOURCE_INVALID", reason="runner_id_conflict"
            )
        native_id = shadow_native_id or detailed_native_id
        if native_id is not None and native_id in native_identities:
            raise CaptureOneRejected(
                "CURRENT_INDEX_SOURCE_INVALID", reason="runner_id_duplicate"
            )
        boxes.add(box)
        identities.add(identity)
        if native_id is not None:
            native_identities.add(native_id)
        rows.append({"box": box, "display_name": name, "identity": identity, "scratch_state": "ACTIVE", "source_native_runner_id": native_id})
    if rows != sorted(rows, key=lambda item: (item["box"], item["identity"])):
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="runner_order_noncanonical")

    # Parse the accepted bytes, strictly.  The repository supports comma,
    # pipe, semicolon and tab form exports; malformed UTF-8 and mixed/partial
    # records are not recoverable publication evidence.
    try:
        csv_text = csv_raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="csv_encoding_invalid") from exc
    first = next((line for line in csv_text.splitlines() if line.strip()), "")
    delimiter_counts = {value: first.count(value) for value in (",", "|", ";", "\t")}
    best = max(delimiter_counts.values(), default=0)
    delimiters = [value for value, count in delimiter_counts.items() if count == best and count > 0]
    if len(delimiters) != 1:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="csv_format_unsupported_or_ambiguous")
    try:
        reader = csv.DictReader(io.StringIO(csv_text, newline=""), delimiter=delimiters[0], strict=True)
        headers = [str(value or "").lstrip("\ufeff").strip() for value in (reader.fieldnames or [])]
        lowered = {value.lower(): value for value in headers}
        expert_history = {"dog name", "plc", "box", "date", "track"}.issubset(lowered)
        name_header = next((lowered[value] for value in ("dog_name", "dog name", "runner", "name") if value in lowered), None)
        box_header = next((lowered[value] for value in ("box", "box_number") if value in lowered), None)
        if name_header is None:
            raise ValueError("name_header_missing")
        csv_runners: list[tuple[int, str]] = []
        for record in reader:
            if None in record or any(value is None for value in record.values()):
                raise ValueError("partial_row")
            name_cell = str(record[name_header] or "").strip()
            if not name_cell:
                continue
            if expert_history:
                import re
                match = re.match(r"^([0-9]{1,2})\.\s+(.+)$", name_cell)
                if match is None:
                    raise ValueError("expert_history_runner_prefix_invalid")
                box, name = int(match.group(1)), match.group(2).strip()
                if re.match(r"^[0-9]", name):
                    raise ValueError("expert_history_runner_prefix_ambiguous")
            elif box_header is None:
                import re
                match = re.match(r"^([0-9]{1,2})\s*[\.\):-]\s*(.+)$", name_cell)
                if match is None:
                    raise ValueError("target_runner_missing_box_prefix")
                box, name = int(match.group(1)), match.group(2).strip()
            else:
                box_text = str(record[box_header] or "").strip()
                if not box_text:
                    raise ValueError("runner_box_missing")
                box, name = int(box_text), name_cell
                import re
                prefix = re.match(r"^([0-9]{1,2})\.\s+(.+)$", name_cell)
                if prefix is not None:
                    if int(prefix.group(1)) != box:
                        raise ValueError("runner_box_prefix_mismatch")
                    name = prefix.group(2).strip()
                    if re.match(r"^[0-9]", name):
                        raise ValueError("runner_box_prefix_ambiguous")
                elif re.match(r"^[0-9]", name_cell):
                    raise ValueError("runner_box_prefix_malformed")
            csv_runners.append((box, normalise_runner_name(name)))
    except (csv.Error, TypeError, ValueError) as exc:
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="csv_runner_rows_invalid") from exc
    accepted = [(item["box"], item["identity"]) for item in rows]
    if csv_runners != accepted or len(csv_runners) != len(set(csv_runners)):
        raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID", reason="csv_sidecar_runner_mismatch")

    root = evidence_root.absolute()
    provenance = {
        "source_url": str(shadow["source_url"]),
        "observed_at": observed.isoformat(),
        "source_generated_at": generated.isoformat(),
        "csv_path": csv_path.absolute().relative_to(root).as_posix(),
        "csv_sha256": sha256_bytes(csv_raw),
        "sidecar_path": sidecar_path.absolute().relative_to(root).as_posix(),
        "sidecar_sha256": sha256_bytes(sidecar_raw),
    }
    runner_hash = sha256_bytes(canonical_bytes({
        "protocol": "collector_current_race_runner_set_sha256_v2",
        "race": {
            "race_url": race["race_url"], "date": race["date"],
            "venue": race["venue"], "race_number": race["race_number"],
            "jump_datetime": race["jump_datetime"],
        },
        "observed_at": provenance["observed_at"],
        "source_generated_at": provenance["source_generated_at"],
        "sources": [
            {"locator": provenance["csv_path"], "sha256": provenance["csv_sha256"]},
            {"locator": provenance["sidecar_path"], "sha256": provenance["sidecar_sha256"]},
        ],
        "active_runners": rows,
    }))
    return rows, provenance, runner_hash


def _seal_current_index_races(
    source: Mapping[str, Any],
    *,
    evidence_root: Path,
    snapshot: _RetainedSafeFiles,
    max_races: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    """Seal safe races and make race-local metadata gaps explicit."""

    races = _normalize_current_index_rows(source, max_races=max_races)
    status = source.get("status")
    if status == "SUCCESS":
        sealed = []
        for race in races:
            runners, runner_source, runner_hash = _v2_runner_rows(
                race, source, evidence_root=evidence_root, snapshot=snapshot
            )
            sealed.append(
                {
                    **race,
                    "runners": runners,
                    "runner_set_sha256": runner_hash,
                    "runner_source": runner_source,
                }
            )
        return sealed, [], len(races)
    if status != "METADATA_COVERAGE_INCOMPLETE":
        raise CaptureOneRejected(
            "CURRENT_INDEX_SOURCE_INVALID", reason="refresh_not_accepted_status"
        )

    coverage = source.get("sidecar_metadata_coverage")
    flags = (
        "safe_weather_present",
        "safe_track_condition_present",
        "safe_both_weather_track_present",
        "safe_expert_form_present",
        "safe_all_weather_track_expert_form_present",
    )
    counters = {
        "safe_weather_present": "safe_weather_race_count",
        "safe_track_condition_present": "safe_track_condition_race_count",
        "safe_both_weather_track_present": "safe_both_weather_track_race_count",
        "safe_expert_form_present": "safe_expert_form_race_count",
        "safe_all_weather_track_expert_form_present": (
            "safe_all_weather_track_expert_form_race_count"
        ),
    }
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("schema_version")
        != "prejump_sidecar_metadata_coverage_v1"
        or coverage.get("status") not in {"PARTIAL", "DATA_MISSING"}
        or coverage.get("selected_race_count") != len(races)
        or not isinstance(coverage.get("races"), list)
        or len(coverage["races"]) != len(races)
    ):
        raise CaptureOneRejected(
            "CURRENT_INDEX_SOURCE_INVALID", reason="runner_coverage_invalid"
        )
    coverage_rows = coverage["races"]
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("race_url"), str)
        or any(not isinstance(row.get(flag), bool) for flag in flags)
        or row.get("safe_both_weather_track_present")
        is not (
            row.get("safe_weather_present") is True
            and row.get("safe_track_condition_present") is True
        )
        or row.get("safe_all_weather_track_expert_form_present")
        is not (
            row.get("safe_both_weather_track_present") is True
            and row.get("safe_expert_form_present") is True
        )
        for row in coverage_rows
    ):
        raise CaptureOneRejected(
            "CURRENT_INDEX_SOURCE_INVALID", reason="runner_coverage_invalid"
        )
    for flag, counter in counters.items():
        value = coverage.get(counter)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != sum(row[flag] is True for row in coverage_rows)
        ):
            raise CaptureOneRejected(
                "CURRENT_INDEX_SOURCE_INVALID", reason="runner_coverage_invalid"
            )
    accepted_count = coverage.get("accepted_selected_csv_count")
    if (
        isinstance(accepted_count, bool)
        or not isinstance(accepted_count, int)
        or accepted_count
        != sum(
            isinstance(row.get("csv_path"), str) and bool(row["csv_path"])
            for row in coverage_rows
        )
        or coverage["safe_all_weather_track_expert_form_race_count"]
        >= len(races)
    ):
        raise CaptureOneRejected(
            "CURRENT_INDEX_SOURCE_INVALID", reason="runner_coverage_invalid"
        )
    by_url: dict[str, Mapping[str, Any]] = {}
    for row in coverage_rows:
        race_url = str(row["race_url"])
        if race_url in by_url:
            raise CaptureOneRejected(
                "CURRENT_INDEX_SOURCE_INVALID", reason="runner_source_misaligned"
            )
        by_url[race_url] = row
    if set(by_url) != {race["race_url"] for race in races}:
        raise CaptureOneRejected(
            "CURRENT_INDEX_SOURCE_INVALID", reason="runner_source_misaligned"
        )

    sealed_races: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for race in races:
        record = by_url[race["race_url"]]
        if (
            record["safe_all_weather_track_expert_form_present"] is not True
            or not isinstance(record.get("csv_path"), str)
            or not record["csv_path"]
            or not isinstance(record.get("sidecar_path"), str)
            or not record["sidecar_path"]
        ):
            exclusions.append(
                {
                    "race_id": race["race_id"],
                    "race_url": race["race_url"],
                    "reason": "runner_source_missing",
                }
            )
            continue
        try:
            runners, runner_source, runner_hash = _v2_runner_rows(
                race, source, evidence_root=evidence_root, snapshot=snapshot
            )
        except CaptureOneRejected as exc:
            if exc.code == "CURRENT_INDEX_PATH_UNSAFE":
                raise
            exclusions.append(
                {
                    "race_id": race["race_id"],
                    "race_url": race["race_url"],
                    "reason": str(exc.details.get("reason") or exc.code),
                }
            )
            continue
        sealed_races.append(
            {
                **race,
                "runners": runners,
                "runner_set_sha256": runner_hash,
                "runner_source": runner_source,
            }
        )
    return sealed_races, exclusions, len(races)


def _atomic_replace_canonical(
    path: Path, payload: Mapping[str, Any], *, evidence_root: Path,
    _pre_replace: Callable[[], None] | None = None,
    _on_replace_failure: Callable[[int, str], None] | None = None,
) -> None:
    root = evidence_root.absolute()
    target = path.absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path)) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path))
    flags_dir = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_parent = root.parent
    root_name = root.name
    root_parent_fd = os.open(root_parent, flags_dir)
    root_parent_identity = os.fstat(root_parent_fd)
    root_named = os.stat(root_name, dir_fd=root_parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(root_named.st_mode):
        os.close(root_parent_fd)
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(root))
    root_fd = os.open(root, flags_dir)
    descriptors = [root_fd]
    identities = [os.fstat(root_fd)]
    if (identities[0].st_dev, identities[0].st_ino) != (root_named.st_dev, root_named.st_ino):
        os.close(root_fd)
        os.close(root_parent_fd)
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(root), reason="root_replaced")
    try:
        for component in relative.parts[:-1]:
            parent_fd = descriptors[-1]
            try:
                named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                os.mkdir(component, mode=0o700, dir_fd=parent_fd)
                named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            child_fd = os.open(component, flags_dir, dir_fd=parent_fd)
            opened = os.fstat(child_fd)
            if not stat.S_ISDIR(named.st_mode) or not stat.S_ISDIR(opened.st_mode) or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
                os.close(child_fd)
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="publish_component_replaced")
            descriptors.append(child_fd)
            identities.append(opened)
        identities[:] = [os.fstat(retained) for retained in descriptors]
        parent_fd = descriptors[-1]
        temporary = f".{relative.parts[-1]}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        replaced = False
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        try:
            raw = canonical_bytes(payload)
            written = 0
            while written < len(raw):
                written += os.write(descriptor, raw[written:])
            os.fsync(descriptor)
            temporary_stat = os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
            temporary_opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(temporary_stat.st_mode)
                or _retained_read_identity(temporary_stat)
                != _retained_read_identity(temporary_opened)
            ):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="publish_temp_invalid")
            identities[-1] = os.fstat(parent_fd)
            _recheck_directory_chain(root_parent_fd, root_name, relative.parts[:-1], descriptors, identities, root_parent_identity, path)
            if _pre_replace is not None:
                _pre_replace()
                identities[-1] = os.fstat(parent_fd)
                _recheck_directory_chain(root_parent_fd, root_name, relative.parts[:-1], descriptors, identities, root_parent_identity, path)
            os.replace(temporary, relative.parts[-1], src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            replaced = True
            identities[-1] = os.fstat(parent_fd)
            published = os.stat(relative.parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            published_opened = os.fstat(descriptor)
            if not stat.S_ISREG(published.st_mode) or _retained_read_identity(published) != _retained_read_identity(published_opened):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="publish_final_replaced")
            published_identity = published_opened
            _recheck_directory_chain(root_parent_fd, root_name, relative.parts[:-1], descriptors, identities, root_parent_identity, path)
            os.fsync(descriptor)
            published = os.stat(relative.parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if _retained_read_identity(published) != _retained_read_identity(published_identity) or _retained_read_identity(os.fstat(descriptor)) != _retained_read_identity(published_identity):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="publish_final_mutated")
            _recheck_directory_chain(root_parent_fd, root_name, relative.parts[:-1], descriptors, identities, root_parent_identity, path)
            os.fsync(parent_fd)
            published = os.stat(relative.parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if _retained_read_identity(published) != _retained_read_identity(published_identity) or _retained_read_identity(os.fstat(descriptor)) != _retained_read_identity(published_identity):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path), reason="publish_final_mutated")
            _recheck_directory_chain(root_parent_fd, root_name, relative.parts[:-1], descriptors, identities, root_parent_identity, path)
        except (CaptureOneRejected, OSError, TypeError, ValueError):
            if replaced and _on_replace_failure is not None:
                try:
                    _on_replace_failure(parent_fd, relative.parts[-1])
                except OSError as rollback_error:
                    del rollback_error
            raise
        finally:
            os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
    finally:
        for retained in reversed(descriptors):
            os.close(retained)
        os.close(root_parent_fd)


def _recheck_directory_chain(
    root_parent_fd: int,
    root_name: str,
    components: tuple[str, ...],
    descriptors: list[int],
    identities: list[os.stat_result],
    root_parent_identity: os.stat_result,
    target: Path,
) -> None:
    parent_current = os.fstat(root_parent_fd)
    root_named = os.stat(root_name, dir_fd=root_parent_fd, follow_symlinks=False)
    if _retained_read_identity(parent_current) != _retained_read_identity(root_parent_identity):
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(target), reason="publish_root_parent_mutated")
    if not stat.S_ISDIR(root_named.st_mode) or _retained_read_identity(root_named) != _retained_read_identity(identities[0]):
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(target), reason="publish_root_replaced")
    for index, identity in enumerate(identities):
        opened = os.fstat(descriptors[index])
        if not stat.S_ISDIR(opened.st_mode) or _retained_read_identity(opened) != _retained_read_identity(identity):
            raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(target), reason="publish_component_replaced")
        if index:
            named = os.stat(components[index - 1], dir_fd=descriptors[index - 1], follow_symlinks=False)
            if not stat.S_ISDIR(named.st_mode) or _retained_read_identity(named) != _retained_read_identity(identity):
                raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(target), reason="publish_component_replaced")


def publish_current_race_index(
    *,
    state_path: Path,
    evidence_root: Path,
    source_refresh_report_path: Path,
    run_id: str,
    max_races: int = MAX_CURRENT_INDEX_RACES,
) -> dict[str, Any]:
    """Seal one finite scheduled-refresh selection at a fixed runtime path."""

    index_path = current_race_index_path(state_path)
    report: dict[str, Any] = {
        "schema_version": "collector_current_race_index_publish_v2",
        "status": "REJECTED",
        "index_path": str(index_path),
        "source_refresh_report_path": str(source_refresh_report_path),
        "run_id": run_id,
    }
    try:
        with _RetainedSafeFiles(evidence_root) as retained:
            source_raw = retained.read(
                source_refresh_report_path,
                missing_code="CURRENT_INDEX_SOURCE_MISSING",
            )
            source = json.loads(source_raw)
            if not isinstance(source, Mapping):
                raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID")
            if source.get("dry_run") is True:
                raise CaptureOneRejected(
                    "CURRENT_INDEX_SOURCE_INVALID", reason="refresh_not_accepted_dry_run"
                )
            source_generated_at = datetime.fromisoformat(str(source["generated_at"]))
            if (
                source_generated_at.tzinfo is None
                or source_generated_at.utcoffset() is None
            ):
                raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID")
            sealed_races, excluded_races, source_selected_race_count = (
                _seal_current_index_races(
                    source,
                    evidence_root=evidence_root,
                    snapshot=retained,
                    max_races=max_races,
                )
            )
            root = evidence_root.absolute()
            refresh_locator = source_refresh_report_path.absolute().relative_to(root).as_posix()
            packet = {
                "schema_version": CURRENT_RACE_INDEX_SCHEMA,
                "run_id": run_id,
                "source_generated_at": source_generated_at.isoformat(),
                "source_refresh_report_path": refresh_locator,
                "source_refresh_report_sha256": sha256_bytes(source_raw),
                "race_count": len(sealed_races),
                "max_races": max_races,
                "races": sealed_races,
            }
            if excluded_races:
                packet.update(
                    {
                        "source_selected_race_count": source_selected_race_count,
                        "excluded_races": excluded_races,
                    }
                )
            _atomic_replace_canonical(index_path, packet, evidence_root=evidence_root)
            packet_raw = canonical_bytes(packet)
            publication = {
                "schema_version": "collector_current_race_index_publication_v1",
                "status": "PUBLISHED",
                "packet_schema_version": CURRENT_RACE_INDEX_SCHEMA,
                "packet_sha256": sha256_bytes(packet_raw),
                "run_id": run_id,
                "source_refresh_report_path": refresh_locator,
                "source_refresh_report_sha256": packet["source_refresh_report_sha256"],
                "source_generated_at": source_generated_at.isoformat(),
                "race_count": len(sealed_races),
                "race_identities": [
                    {
                        "race_id": race["race_id"],
                        "race_url": race["race_url"],
                        "date": race["date"],
                        "venue": race["venue"],
                        "race_number": race["race_number"],
                        "jump_datetime": race["jump_datetime"],
                    }
                    for race in sealed_races
                ],
                "runner_set_sha256": [race["runner_set_sha256"] for race in sealed_races],
                "runner_sources": [race["runner_source"] for race in sealed_races],
            }
            if excluded_races:
                publication.update(
                    {
                        "source_selected_race_count": source_selected_race_count,
                        "excluded_races": excluded_races,
                    }
                )
            _atomic_replace_canonical(
                index_path.parent / CURRENT_RACE_INDEX_PUBLICATION_FILENAME,
                publication,
                evidence_root=evidence_root,
                _pre_replace=retained.validate,
            )
    except (
        CaptureOneRejected,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report["reason"] = (
            exc.code if isinstance(exc, CaptureOneRejected) else type(exc).__name__
        )
        return report
    report.update(
        {
            "status": "PUBLISHED",
            "schema_version": "collector_current_race_index_publish_v2",
            "packet_schema_version": CURRENT_RACE_INDEX_SCHEMA,
            "packet_sha256": sha256_bytes(packet_raw),
            "race_count": len(sealed_races),
            "source_generated_at": source_generated_at.isoformat(),
            "source_refresh_report_sha256": packet[
                "source_refresh_report_sha256"
            ],
            "runner_set_sha256": [race["runner_set_sha256"] for race in sealed_races],
            "runner_sources": [race["runner_source"] for race in sealed_races],
        }
    )
    if excluded_races:
        report.update(
            {
                "source_selected_race_count": source_selected_race_count,
                "excluded_races": excluded_races,
            }
        )
    return report


def _aware_datetime(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _validate_current_odds_evidence(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    packet: Mapping[str, Any],
    publication: Mapping[str, Any],
    current_time: datetime,
    max_age_seconds: int,
    enforce_max_age: bool = True,
) -> None:
    """Bind the producer's current state/report lifecycle to this publication."""

    state_time = _aware_datetime(state.get("updated_at"))
    report_time = _aware_datetime(report.get("generated_at"))
    source_time = _aware_datetime(packet.get("source_generated_at"))
    allowed = {
        "ODDS_CAPTURE_ONLY_READY": {"READY", "READY_WITH_BLOCKED_ATTEMPTS"},
        "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE": {"HANDLED_NO_WRITE"},
    }
    final_status = report.get("final_status")
    if (
        set(publication).isdisjoint({"packet_sha256"})
        or state_time is None
        or report_time is None
        or source_time is None
        or final_status not in allowed
        or report.get("status") not in allowed[final_status]
        or state.get("final_status") != final_status
        or state.get("status") != report.get("status")
        or state.get("output_dir") != report.get("output_dir")
        or state.get("autopilot_output_dir") != report.get("autopilot_output_dir")
        or state.get("run_id") != report.get("run_id")
        or report.get("run_id") != packet.get("run_id")
        or state_time != report_time
        or any(
            (current_time - moment).total_seconds() < 0
            or (
                enforce_max_age
                and (current_time - moment).total_seconds() > max_age_seconds
            )
            for moment in (state_time, report_time, source_time)
        )
    ):
        raise CaptureOneRejected("CURRENT_INDEX_REPORT_INVALID")


def bounded_current_race_index(
    *,
    current_time: datetime,
    timeout_seconds: float,
    index_path: Path,
    evidence_root: Path,
    max_age_seconds: int,
    max_races: int = MAX_CURRENT_INDEX_RACES,
    return_verified_view: bool = False,
) -> list[Mapping[str, Any]] | VerifiedCurrentRaceIndex:
    """Read one finite collector-owned index with a hard wall-clock deadline."""

    if (
        current_time.tzinfo is None
        or current_time.utcoffset() is None
        or timeout_seconds <= 0
        or isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, int)
        or max_age_seconds <= 0
        or max_races <= 0
    ):
        raise CaptureOneRejected("DISCOVERY_BUDGET_INVALID")

    previous_handler = signal.getsignal(signal.SIGALRM)

    def timed_out(signum: int, frame: object) -> None:
        del signum, frame
        raise _DiscoveryTimedOut

    snapshot: _RetainedSafeFiles | None = None
    try:
        signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        snapshot = _RetainedSafeFiles(evidence_root)
        packet_raw = snapshot.read(index_path, missing_code="CURRENT_INDEX_UNAVAILABLE")
        packet = json.loads(packet_raw)
        packet_keys = {
            "schema_version", "run_id", "source_generated_at",
            "source_refresh_report_path", "source_refresh_report_sha256",
            "race_count", "max_races", "races",
        }
        partial_packet_keys = packet_keys | {
            "source_selected_race_count", "excluded_races",
        }
        if (
            not isinstance(packet, Mapping)
            or packet.get("schema_version") not in {CURRENT_RACE_INDEX_V1_SCHEMA, CURRENT_RACE_INDEX_SCHEMA}
            or canonical_bytes(packet) != packet_raw
            or packet.get("max_races") != max_races
            or frozenset(packet)
            not in {frozenset(packet_keys), frozenset(partial_packet_keys)}
        ):
            raise CaptureOneRejected("CURRENT_INDEX_INVALID")
        source_generated_at = datetime.fromisoformat(
            str(packet["source_generated_at"])
        )
        if (
            source_generated_at.tzinfo is None
            or source_generated_at.utcoffset() is None
        ):
            raise CaptureOneRejected("CURRENT_INDEX_INVALID")
        age_seconds = (current_time - source_generated_at).total_seconds()
        if age_seconds < 0 or (
            not return_verified_view and age_seconds > max_age_seconds
        ):
            raise CaptureOneRejected(
                "CURRENT_INDEX_STALE",
                age_seconds=age_seconds,
                max_age_seconds=max_age_seconds,
            )
        source_locator = str(packet["source_refresh_report_path"])
        source_path = Path(source_locator)
        publication_raw = state_raw = report_raw = None
        if packet.get("schema_version") == CURRENT_RACE_INDEX_SCHEMA:
            if source_path.is_absolute() or not source_locator or any(
                part in {"", ".", ".."} for part in source_path.parts
            ):
                raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID")
            source_path = evidence_root.absolute() / source_path
        source_raw = snapshot.read(source_path, missing_code="CURRENT_INDEX_SOURCE_MISSING")
        if sha256_bytes(source_raw) != packet.get("source_refresh_report_sha256"):
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_CHANGED")
        source = json.loads(source_raw)
        if not isinstance(source, Mapping):
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID")
        if packet.get("schema_version") == CURRENT_RACE_INDEX_SCHEMA and source.get(
            "dry_run"
        ) is True:
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID")
        races = _normalize_current_index_rows(source, max_races=max_races)
        expected_races: list[Mapping[str, Any]] = races
        expected_exclusions: list[dict[str, str]] = []
        source_selected_race_count = len(races)
        if packet.get("schema_version") == CURRENT_RACE_INDEX_SCHEMA:
            expected_races, expected_exclusions, source_selected_race_count = (
                _seal_current_index_races(
                    source,
                    evidence_root=evidence_root,
                    snapshot=snapshot,
                    max_races=max_races,
                )
            )
        if (
            packet.get("race_count") != len(expected_races)
            or packet.get("races") != expected_races
            or bool(expected_exclusions) != (set(packet) == partial_packet_keys)
            or (
                expected_exclusions
                and (
                    packet.get("excluded_races") != expected_exclusions
                    or packet.get("source_selected_race_count")
                    != source_selected_race_count
                )
            )
        ):
            raise CaptureOneRejected("CURRENT_INDEX_INVALID")
        if packet.get("schema_version") == CURRENT_RACE_INDEX_SCHEMA:
            publication_raw = snapshot.read(
                index_path.parent / CURRENT_RACE_INDEX_PUBLICATION_FILENAME,
                missing_code="CURRENT_INDEX_PUBLICATION_MISSING",
            )
            publication = json.loads(publication_raw)
            expected_publication = {
                "schema_version": "collector_current_race_index_publication_v1",
                "status": "PUBLISHED",
                "packet_schema_version": packet["schema_version"],
                "packet_sha256": sha256_bytes(packet_raw),
                "run_id": packet["run_id"],
                "source_refresh_report_path": packet["source_refresh_report_path"],
                "source_refresh_report_sha256": packet["source_refresh_report_sha256"],
                "source_generated_at": packet["source_generated_at"],
                "race_count": packet["race_count"],
                "race_identities": [
                    {
                        "race_id": race["race_id"], "race_url": race["race_url"],
                        "date": race["date"], "venue": race["venue"],
                        "race_number": race["race_number"],
                        "jump_datetime": race["jump_datetime"],
                    }
                    for race in packet["races"]
                ],
                "runner_set_sha256": [race["runner_set_sha256"] for race in packet["races"]],
                "runner_sources": [race["runner_source"] for race in packet["races"]],
            }
            if expected_exclusions:
                expected_publication.update(
                    {
                        "source_selected_race_count": source_selected_race_count,
                        "excluded_races": expected_exclusions,
                    }
                )
            if (
                not isinstance(publication, Mapping)
                or canonical_bytes(publication) != publication_raw
                or publication != expected_publication
            ):
                raise CaptureOneRejected("CURRENT_INDEX_PUBLICATION_INVALID")
            state_path = index_path.parent / ODDS_CAPTURE_ONLY_STATE_FILENAME
            state_raw = snapshot.read(state_path, missing_code="CURRENT_INDEX_REPORT_MISSING")
            state = json.loads(state_raw)
            if (
                not isinstance(state, Mapping)
                or state.get("schema_version")
                != "shadow_autopilot_odds_capture_only_state_v1"
            ):
                raise CaptureOneRejected("CURRENT_INDEX_REPORT_INVALID")
            index_state = state.get("current_race_index_state", state)
            if index_state is not state and (
                not isinstance(index_state, Mapping)
                or set(index_state) != {
                    "schema_version", "updated_at", "run_id", "output_dir",
                    "autopilot_output_dir", "final_status", "status",
                }
                or index_state.get("schema_version")
                != "collector_current_race_index_state_v1"
            ):
                raise CaptureOneRejected("CURRENT_INDEX_REPORT_INVALID")
            if index_state.get("run_id") != packet["run_id"]:
                raise CaptureOneRejected("CURRENT_INDEX_REPORT_INVALID")
            output_dir = _evidence_locator_path(
                index_state.get("output_dir"), evidence_root=evidence_root
            )
            report_raw = snapshot.read(
                output_dir / ODDS_CAPTURE_ONLY_REPORT_FILENAME,
                missing_code="CURRENT_INDEX_REPORT_MISSING",
            )
            report = json.loads(report_raw)
            expected_publish = {
                "schema_version": "collector_current_race_index_publish_v2",
                "status": "PUBLISHED",
                "index_path": str(index_path),
                "source_refresh_report_path": str(source_path),
                "packet_schema_version": publication["packet_schema_version"],
                "packet_sha256": publication["packet_sha256"],
                "run_id": publication["run_id"],
                "source_refresh_report_sha256": publication[
                    "source_refresh_report_sha256"
                ],
                "source_generated_at": publication["source_generated_at"],
                "race_count": publication["race_count"],
                "runner_set_sha256": publication["runner_set_sha256"],
                "runner_sources": publication["runner_sources"],
            }
            if expected_exclusions:
                expected_publish.update(
                    {
                        "source_selected_race_count": source_selected_race_count,
                        "excluded_races": expected_exclusions,
                    }
                )
            if (
                not isinstance(report, Mapping)
                or report.get("schema_version")
                != "shadow_autopilot_odds_capture_only_daemon_report_v1"
                or report.get("run_id") != packet["run_id"]
                or report.get("output_dir") != index_state.get("output_dir")
                or report.get("current_race_index_publish") != expected_publish
            ):
                raise CaptureOneRejected("CURRENT_INDEX_REPORT_INVALID")
            _validate_current_odds_evidence(
                state=index_state,
                report=report,
                packet=packet,
                publication=publication,
                current_time=current_time,
                max_age_seconds=max_age_seconds,
                enforce_max_age=not return_verified_view,
            )
        snapshot.validate()
        if return_verified_view:
            if packet["schema_version"] != CURRENT_RACE_INDEX_SCHEMA:
                raise CaptureOneRejected("CURRENT_INDEX_CATALOG_INELIGIBLE")
            assert publication_raw is not None and state_raw is not None and report_raw is not None
            return VerifiedCurrentRaceIndex(
                schema_version=packet["schema_version"],
                run_id=packet["run_id"],
                source_generated_at=packet["source_generated_at"],
                packet_sha256=sha256_bytes(packet_raw),
                packet_bytes=packet_raw,
                races=tuple(expected_races),
                source_refresh_report_path=packet["source_refresh_report_path"],
                source_refresh_report_sha256=packet["source_refresh_report_sha256"],
                publication_sha256=sha256_bytes(publication_raw),
                state_sha256=sha256_bytes(state_raw),
                report_sha256=sha256_bytes(report_raw),
            )
        return expected_races
    except _DiscoveryTimedOut as exc:
        raise CaptureOneRejected(
            "DISCOVERY_TIMEOUT", budget_seconds=timeout_seconds
        ) from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CaptureOneRejected("CURRENT_INDEX_INVALID") from exc
    finally:
        if snapshot is not None:
            snapshot.__exit__()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def release_owned_collector_lock(lock: Any) -> None:
    if not isinstance(lock, OwnedCollectorLock):
        raise CaptureOneRejected("LOCK_RELEASE_FAILED", reason="handle_invalid")
    evidence = _read_lock_evidence(lock.path)
    if evidence.get("lock_owner_run_id") != lock.run_id:
        raise CaptureOneRejected(
            "LOCK_RELEASE_FAILED", reason="ownership_unverified"
        )
    if not _unlink_owned_lock(lock):
        raise CaptureOneRejected("LOCK_RELEASE_FAILED", reason="inode_changed")


@contextlib.contextmanager
def _isolated_runtime_cwd(output_dir: Path):
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".runtime_", dir=output_dir) as scratch:
        try:
            os.chdir(scratch)
            yield
        finally:
            os.chdir(previous)


def refresh_exact_race(
    race: Mapping[str, Any], output_dir: Path, current_time: datetime
) -> tuple[Path, Path]:
    """Download only the resolved exact race and validate its source binding."""

    upcoming_dir = output_dir / "exact_upcoming"
    upcoming_dir.mkdir(parents=True, exist_ok=False)
    previous = os.environ.get("UPCOMING_RACES_DIR")
    try:
        os.environ["UPCOMING_RACES_DIR"] = str(upcoming_dir)
        with _isolated_runtime_cwd(output_dir):
            from scripts.predict_market_form_residual import (
                _sidecar_context,
                _validate_form_binding,
            )
            from upcoming_race_browser import UpcomingRaceBrowser

            with contextlib.redirect_stdout(sys.stderr):
                result = UpcomingRaceBrowser().download_race_csv(
                    str(race.get("url") or ""),
                    race_info_hint=race,
                )
        if not isinstance(result, Mapping) or result.get("success") is not True:
            raise CaptureOneRejected(
                "EXACT_METADATA_UNAVAILABLE",
                reason=result.get("error") if isinstance(result, Mapping) else None,
            )
        form = Path(str(result.get("filepath") or ""))
        if not form.is_absolute():
            form = ROOT / form
        sidecar = form.with_name(form.name + ".metadata.json")
        if (
            form.is_symlink()
            or sidecar.is_symlink()
            or not form.is_file()
            or not sidecar.is_file()
            or not form.resolve().is_relative_to(upcoming_dir.resolve())
            or not sidecar.resolve().is_relative_to(upcoming_dir.resolve())
        ):
            raise CaptureOneRejected("EXACT_METADATA_UNAVAILABLE")
        form_raw = form.read_bytes()
        sidecar_value = json.loads(sidecar.read_bytes())
        if not isinstance(sidecar_value, Mapping):
            raise TypeError("sidecar_not_mapping")
        _validate_form_binding(
            sidecar_value,
            form_csv_path=form,
            form_raw=form_raw,
            form_sha=sha256_bytes(form_raw),
        )
        context = _sidecar_context(sidecar_value)
        expected_jump = datetime.fromisoformat(str(race["jump_timestamp"]))
        if (
            context.get("expected_race_id") != race["race_id"]
            or context.get("jump_timestamp") != expected_jump
        ):
            raise CaptureOneRejected(
                "EXACT_METADATA_UNAVAILABLE", reason="identity_mismatch"
            )
        return form, sidecar
    except CaptureOneRejected:
        raise
    except Exception as exc:
        raise CaptureOneRejected(
            "EXACT_METADATA_UNAVAILABLE", error=type(exc).__name__
        ) from exc
    finally:
        if previous is None:
            os.environ.pop("UPCOMING_RACES_DIR", None)
        else:
            os.environ["UPCOMING_RACES_DIR"] = previous


def _contains_outcome(value: Any) -> bool:
    outcome_keys = {
        "actual_win",
        "finish_position",
        "official_result",
        "outcome",
        "placing",
        "result",
        "winner",
        "winner_name",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in outcome_keys or _contains_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_outcome(item) for item in value)
    return False


def _seal_capture_handoff(
    *,
    expected: Mapping[str, Any],
    expected_runner_hash: str | None,
    plan_item: Mapping[str, Any],
    capture_report: Mapping[str, Any],
    report_path: Path,
    form_path: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    """Seal one exact append-only capture without invoking prediction scoring."""

    if (
        plan_item.get("race_id") != expected["race_id"]
        or plan_item.get("thedogs_source_url") != expected["url"]
        or plan_item.get("venue") != expected["venue"]
        or plan_item.get("race_number") != expected["race_number"]
        or plan_item.get("race_date") != expected["race_date"]
        or plan_item.get("jump_datetime") != expected["jump_timestamp"]
    ):
        raise CaptureOneRejected("IDENTITY_MISMATCH")
    if _contains_outcome(capture_report):
        raise CaptureOneRejected("RECEIPT_CONTAINS_OUTCOME")
    attempts = [
        row
        for row in capture_report.get("attempts") or []
        if isinstance(row, Mapping) and row.get("race_id") == expected["race_id"]
    ]
    if len(attempts) != 1:
        raise CaptureOneRejected("CAPTURE_FAILED", reason="exact_attempt_count")
    attempt = attempts[0]
    append_report = attempt.get("append_report")
    validation = attempt.get("validation")
    if (
        attempt.get("status") != "APPENDED"
        or not isinstance(append_report, Mapping)
        or append_report.get("status") != "SUCCESS"
        or append_report.get("append_only") is not True
        or int(append_report.get("inserted_rows") or 0) <= 0
        or not isinstance(validation, Mapping)
        or validation.get("status") != "PASS"
    ):
        raise CaptureOneRejected(
            "CAPTURE_FAILED", reason=str(attempt.get("status") or "UNKNOWN")
        )
    try:
        appended_at = datetime.fromisoformat(str(attempt["append_time"]))
        jump = datetime.fromisoformat(str(expected["jump_timestamp"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureOneRejected("CAPTURE_FAILED", reason="timestamp_invalid") from exc
    if (
        appended_at.tzinfo is None
        or jump.tzinfo is None
        or appended_at >= jump
    ):
        raise CaptureOneRejected("CAPTURE_WINDOW_CLOSED")
    normalized = normalize_validation_receipt(
        race_id=str(expected["race_id"]),
        captured_at=appended_at,
        validation=validation,
        source_kind="verified_collector_capture_one",
    )
    if (
        expected_runner_hash is not None
        and normalized["runner_set_sha256"] != expected_runner_hash
    ):
        raise CaptureOneRejected("IDENTITY_MISMATCH")
    raw_by_label: dict[str, bytes] = {}
    for label, path in (
        ("report", report_path),
        ("form", form_path),
        ("sidecar", sidecar_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise CaptureOneRejected("SOURCE_FILE_UNSAFE", label=label)
        raw_by_label[label] = path.read_bytes()
    return {
        "schema_version": HANDOFF_SCHEMA,
        "race_id": expected["race_id"],
        "race": dict(expected),
        "append_timestamp": appended_at.isoformat(),
        "runner_set_sha256": normalized["runner_set_sha256"],
        "source_report_sha256": sha256_bytes(raw_by_label["report"]),
        "source_form_sha256": sha256_bytes(raw_by_label["form"]),
        "source_sidecar_sha256": sha256_bytes(raw_by_label["sidecar"]),
        "capture_attempt_sha256": sha256_bytes(canonical_bytes(attempt)),
        "append_report_sha256": sha256_bytes(canonical_bytes(append_report)),
        "_report_bytes": raw_by_label["report"],
        "_form_bytes": raw_by_label["form"],
        "_sidecar_bytes": raw_by_label["sidecar"],
        "_report_path": report_path.resolve(),
        "_form_path": form_path.resolve(),
        "_sidecar_path": sidecar_path.resolve(),
        "_form_name": form_path.name,
    }


def seal_capture_handoff(
    *,
    context: CollectorRequest,
    plan_item: Mapping[str, Any],
    capture_report: Mapping[str, Any],
    report_path: Path,
    form_path: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    """Seal one exact manual capture through the shared collector primitive."""

    return _seal_capture_handoff(
        expected=context.request["race"],
        expected_runner_hash=context.request["expected_runner_set_sha256"],
        plan_item=plan_item,
        capture_report=capture_report,
        report_path=report_path,
        form_path=form_path,
        sidecar_path=sidecar_path,
    )


def _publish_once_canonical(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def publish_scheduled_capture_receipts(
    *,
    protocol: ManualPredictionCollectorProtocol,
    evidence_root: Path,
    collector_run_id: str,
    plan_item: Mapping[str, Any],
    attempt: Mapping[str, Any],
    output_dir: Path,
    emitted_at: datetime,
) -> dict[str, Any]:
    """Publish bounded alias-indexed receipts immediately after one append."""

    evidence_root = evidence_root.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(evidence_root):
        raise CaptureOneRejected("SOURCE_FILE_UNSAFE", label="output_dir")
    if attempt.get("status") != "APPENDED":
        raise CaptureOneRejected(
            "CAPTURE_FAILED", reason="scheduled_attempt_not_appended"
        )
    from scripts.refresh_prejump_upcoming import stable_race_id_variants

    aliases = set(stable_race_id_variants(plan_item))
    aliases.update(
        value
        for value in plan_item.get("race_id_aliases") or []
        if isinstance(value, str) and value
    )
    source_race_id = str(plan_item.get("race_id") or "")
    if source_race_id and source_race_id not in aliases:
        aliases.add(source_race_id)
    if not aliases or len(aliases) > 16:
        raise CaptureOneRejected(
            "IDENTITY_MISMATCH",
            reason="scheduled_receipt_aliases_unbounded",
            alias_count=len(aliases),
        )
    expected_runner_hash = runner_set_sha256(
        [
            dict(row)
            for row in plan_item.get("expected_runners") or []
            if isinstance(row, Mapping)
        ]
    )
    if expected_runner_hash is None:
        raise CaptureOneRejected(
            "IDENTITY_MISMATCH", reason="scheduled_receipt_runners_missing"
        )

    receipts: list[dict[str, Any]] = []
    for race_id in sorted(aliases):
        parts = race_id.split(" - ", 2)
        if (
            len(parts) != 3
            or parts[0] != f"Race {plan_item.get('race_number')}"
            or parts[2] != str(plan_item.get("race_date"))
        ):
            raise CaptureOneRejected(
                "IDENTITY_MISMATCH", reason="scheduled_receipt_alias_invalid"
            )
        adapted_attempt = dict(attempt)
        adapted_attempt["race_id"] = race_id
        source_report = {
            "schema_version": "collector_exact_capture_source_v1",
            "collector_run_id": collector_run_id,
            "generated_at": emitted_at.isoformat(),
            "race_id": race_id,
            "source_race_id": source_race_id,
            "source_plan_item": dict(plan_item),
            "source_attempt": dict(attempt),
            "attempts": [adapted_attempt],
        }
        source_raw = canonical_bytes(source_report)
        source_path = (
            output_dir
            / "collector_exact_receipt_sources"
            / f"{sha256_bytes(source_raw)}.json"
        )
        try:
            _publish_once_canonical(source_path, source_report)
        except FileExistsError:
            if source_path.is_symlink() or source_path.read_bytes() != source_raw:
                raise CaptureOneRejected("HASH_DRIFT", field="source_report")

        expected = {
            "race_id": race_id,
            "url": str(plan_item.get("thedogs_source_url") or ""),
            "venue": parts[1],
            "race_number": int(plan_item["race_number"]),
            "race_date": str(plan_item["race_date"]),
            "jump_timestamp": str(plan_item["jump_datetime"]),
        }
        adapted_plan = dict(plan_item)
        adapted_plan["race_id"] = race_id
        adapted_plan["venue"] = parts[1]
        handoff = _seal_capture_handoff(
            expected=expected,
            expected_runner_hash=expected_runner_hash,
            plan_item=adapted_plan,
            capture_report=source_report,
            report_path=source_path,
            form_path=Path(str(plan_item["csv_path"])).resolve(),
            sidecar_path=Path(str(plan_item["sidecar_path"])).resolve(),
        )
        published = protocol.publish_collector_exact_receipt(
            collector_run_id=collector_run_id,
            emitted_at=emitted_at,
            handoff=handoff,
        )
        receipts.append(
            {
                "race_id": race_id,
                "captured_at": published["captured_at"],
                "capture_attempt_sha256": handoff["capture_attempt_sha256"],
            }
        )
    return {
        "schema_version": "collector_exact_capture_receipt_publish_v1",
        "status": "PUBLISHED",
        "source_race_id": source_race_id,
        "receipt_count": len(receipts),
        "receipts": receipts,
    }


def _terminal_result(
    *,
    protocol: ManualPredictionCollectorProtocol,
    context: CollectorRequest,
    status: str,
    now: datetime,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol_status = (
        status
        if status
        in {
            "REQUEST_EXPIRED",
            "RACE_NOT_FOUND",
            "CAPTURE_WINDOW_CLOSED",
            "IDENTITY_MISMATCH",
            "CAPTURE_FAILED",
        }
        else "CAPTURE_FAILED"
    )
    response = protocol.read_response(str(context.request["request_id"]))
    if response is None:
        response = protocol.publish_terminal(
            context,
            status=protocol_status,
            now=now,
            reason=reason,
        )
    result_status = (
        str(response["status"])
        if response["status"] != protocol_status
        else status
    )
    return {
        "schema_version": "collector_capture_one_result_v1",
        "status": result_status,
        "request_id": str(context.request["request_id"]),
        "reason": response.get("reason"),
        **({"busy": dict(details)} if status == "BUSY" and details else {}),
    }


def run_capture_one(
    *,
    protocol_root: Path,
    evidence_root: Path,
    request_id: str,
    db_path: Path,
    lock_path: Path,
    output_dir: Path,
    minimum_margin_seconds: float,
    minimum_post_lock_margin_seconds: float,
    minimum_fetch_margin_seconds: float,
    fetch_timeout_seconds: float,
    dependencies: CaptureOneDependencies | None = None,
) -> dict[str, Any]:
    """Run one claimed exact request under the collector's canonical lock."""

    from scripts import autonomous_live_odds_capture as capture

    deps = dependencies or CaptureOneDependencies(
        now=lambda: datetime.now().astimezone(),
        refresh_exact=refresh_exact_race,
        build_plan_item=capture.build_plan_item,
        execute_capture_plan=capture.execute_capture_plan,
        acquire_lock=acquire_collector_lock_no_steal,
        release_lock=release_owned_collector_lock,
    )
    protocol = ManualPredictionCollectorProtocol(protocol_root)
    run_id = f"manual_capture_one_{uuid.uuid4().hex}"
    now = deps.now()
    context = protocol.claim_request(
        request_id,
        now=now,
        collector_run_id=run_id,
    )
    evidence_root = evidence_root.resolve()
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(evidence_root)
    except ValueError:
        return _terminal_result(
            protocol=protocol,
            context=context,
            status="CAPTURE_FAILED",
            now=now,
            reason="output_dir_outside_evidence_root",
        )
    lock_handle: Any | None = None
    try:
        expected_lock = db_path.resolve().parent / CANONICAL_LOCK_RELATIVE_PATH
        if lock_path.absolute() != expected_lock:
            raise CaptureOneRejected(
                "LOCK_PATH_DB_ROOT_MISMATCH",
                expected_lock_path=str(expected_lock),
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        jump = datetime.fromisoformat(
            str(context.request["race"]["jump_timestamp"])
        )
        remaining = (jump - now).total_seconds()
        if remaining <= 0:
            return _terminal_result(
                protocol=protocol,
                context=context,
                status="CAPTURE_WINDOW_CLOSED",
                now=now,
                reason="race_jump_reached_before_capture_one",
            )
        if remaining <= minimum_margin_seconds:
            return _terminal_result(
                protocol=protocol,
                context=context,
                status="INSUFFICIENT_PREJUMP_MARGIN",
                now=now,
                reason=(
                    "computed_capture_margin_not_met:"
                    f"remaining={remaining:.6f},required={minimum_margin_seconds:.6f}"
                ),
            )
        deps.phase_hook("before_lock")
        try:
            lock_handle = deps.acquire_lock(
                lock_path=lock_path,
                run_id=run_id,
                output_dir=output_dir,
            )
        except CollectorBusy as exc:
            return _terminal_result(
                protocol=protocol,
                context=context,
                status="BUSY",
                now=deps.now(),
                reason=json.dumps(exc.evidence, sort_keys=True, separators=(",", ":")),
                details=exc.evidence,
            )
        deps.phase_hook("after_lock")
        remaining = (jump - deps.now()).total_seconds()
        if remaining <= minimum_post_lock_margin_seconds:
            return _terminal_result(
                protocol=protocol,
                context=context,
                status="INSUFFICIENT_PREJUMP_MARGIN",
                now=deps.now(),
                reason=(
                    "computed_post_lock_margin_not_met:"
                    f"remaining={remaining:.6f},"
                    f"required={minimum_post_lock_margin_seconds:.6f}"
                ),
            )
        form_path, sidecar_path = deps.refresh_exact(
            context.request["race"],
            output_dir,
            deps.now(),
        )
        item = dict(deps.build_plan_item(form_path, deps.now()))
        plan = {
            "schema_version": "autonomous_live_odds_capture_plan_v1",
            "generated_at": deps.now().isoformat(),
            "races": [item],
            "ready_count": 1 if item.get("status") == "READY_TO_CAPTURE" else 0,
            "status_counts": {str(item.get("status") or "UNKNOWN"): 1},
        }
        plan = protocol.prioritize_capture_plan(context, plan, now=deps.now())
        protocol.begin_attempt(
            context,
            now=deps.now(),
            collector_run_id=run_id,
        )
        remaining = (jump - deps.now()).total_seconds()
        if remaining <= minimum_fetch_margin_seconds:
            return _terminal_result(
                protocol=protocol,
                context=context,
                status="INSUFFICIENT_PREJUMP_MARGIN",
                now=deps.now(),
                reason=(
                    "computed_pre_fetch_margin_not_met:"
                    f"remaining={remaining:.6f},"
                    f"required={minimum_fetch_margin_seconds:.6f}"
                ),
            )
        deps.phase_hook("before_capture")
        capture_report = dict(
            deps.execute_capture_plan(
                plan,
                db_path=db_path,
                current_time=deps.now(),
                execute=True,
                allow_auto_scrape_odds=True,
                fetch_timeout_seconds=fetch_timeout_seconds,
                progress_dir=output_dir,
            )
        )
        report_path = output_dir / "autonomous_live_odds_capture_report.json"
        report_path.write_bytes(canonical_bytes(capture_report))
        if int(capture_report.get("appended_attempt_count") or 0) != 1:
            return _terminal_result(
                protocol=protocol,
                context=context,
                status="CAPTURE_FAILED",
                now=deps.now(),
                reason="collector_exact_capture_not_appended_once",
            )
        deps.phase_hook("before_seal")
        handoff = seal_capture_handoff(
            context=context,
            plan_item=item,
            capture_report=capture_report,
            report_path=report_path,
            form_path=form_path,
            sidecar_path=sidecar_path,
        )
        appended_at = datetime.fromisoformat(str(handoff["append_timestamp"]))
        receipt_now = max(deps.now(), appended_at)
        normalized, _, _, _ = receipt_from_handoff(
            handoff,
            current_time=receipt_now,
            max_age_seconds=max(1, int(minimum_margin_seconds)),
        )
        response = protocol.publish_receipt_ready(
            context,
            now=receipt_now,
            handoff=handoff,
            normalized_receipt=normalized,
        )
        result = {
            "schema_version": "collector_capture_one_result_v1",
            "status": str(response["status"]),
            "request_id": request_id,
            "appended_attempt_count": 1,
            "inserted_live_odds_rows": int(
                capture_report.get("inserted_live_odds_rows") or 0
            ),
            "capture_report": str(report_path),
        }
        deps.phase_hook("after_seal")
        return result
    except CaptureCancelled:
        if (
            protocol.receipt_path(request_id).is_file()
            and not protocol.response_path(request_id).exists()
            and "handoff" in locals()
            and "normalized" in locals()
            and "receipt_now" in locals()
        ):
            protocol.publish_receipt_ready(
                context,
                now=receipt_now,
                handoff=handoff,
                normalized_receipt=normalized,
            )
        existing = protocol.read_response(request_id)
        if existing is not None:
            return {
                "schema_version": "collector_capture_one_result_v1",
                "status": str(existing["status"]),
                "request_id": request_id,
                "appended_attempt_count": 1
                if existing["status"] == RECEIPT_READY
                else 0,
            }
        return _terminal_result(
            protocol=protocol,
            context=context,
            status="CANCELLED",
            now=deps.now(),
            reason="capture_one_cancelled_before_receipt_seal",
        )
    except (CaptureOneRejected, ProtocolRejected, PredictionBlocked) as exc:
        code = getattr(exc, "code", "CAPTURE_FAILED")
        status = (
            code
            if code
            in {
                "CAPTURE_WINDOW_CLOSED",
                "IDENTITY_MISMATCH",
                "INSUFFICIENT_PREJUMP_MARGIN",
            }
            else "CAPTURE_FAILED"
        )
        return _terminal_result(
            protocol=protocol,
            context=context,
            status=status,
            now=deps.now(),
            reason=f"collector_capture_one_rejected:{code}",
        )
    except Exception as exc:
        return _terminal_result(
            protocol=protocol,
            context=context,
            status="CAPTURE_FAILED",
            now=deps.now(),
            reason=f"collector_capture_one_failed:{type(exc).__name__}",
        )
    finally:
        if lock_handle is not None:
            deps.release_lock(lock_handle)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def invoke_capture_one(
    *,
    command: Sequence[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Invoke collector capture-one synchronously and reap its process group."""

    previous = install_cancellation_handlers()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except BaseException as exc:
            _terminate_process_group(process)
            if isinstance(
                exc,
                (
                    CaptureCancelled,
                    KeyboardInterrupt,
                    SystemExit,
                    subprocess.TimeoutExpired,
                ),
            ):
                raise CaptureOneRejected(
                    "CANCELLED",
                    reason="collector_process_group_terminated_and_reaped",
                ) from exc
            raise
    finally:
        restore_signal_handlers(previous)
    if process.returncode not in (0, 2):
        raise CaptureOneRejected(
            "COLLECTOR_PROCESS_FAILED",
            returncode=process.returncode,
            stderr=stderr[-2000:],
        )
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CaptureOneRejected(
            "COLLECTOR_PROCESS_INVALID", stderr=stderr[-2000:]
        ) from exc
    if not isinstance(value, dict):
        raise CaptureOneRejected("COLLECTOR_PROCESS_INVALID")
    return value


def install_cancellation_handlers() -> dict[int, Any]:
    """Turn termination into an exception so browser contexts unwind cleanly."""

    previous: dict[int, Any] = {}

    def cancel(signum: int, frame: object) -> None:
        del frame
        raise CaptureCancelled(f"signal:{signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, cancel)
    return previous


def restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)
