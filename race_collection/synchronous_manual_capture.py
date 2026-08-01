"""Bounded collector-owned capture for one exact manual prediction request."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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

ROOT = Path(__file__).resolve().parents[1]
HANDOFF_SCHEMA = "on_demand_verified_collector_capture_v2"
CURRENT_RACE_INDEX_SCHEMA = "collector_current_race_index_v1"
CURRENT_RACE_INDEX_FILENAME = "manual_prediction_current_race_index.json"
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


def _safe_file_bytes(
    path: Path,
    *,
    evidence_root: Path,
    missing_code: str,
) -> bytes:
    root = evidence_root.resolve()
    logical = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise CaptureOneRejected(missing_code, path=str(path))
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path))
    try:
        size = path.stat(follow_symlinks=False).st_size
    except OSError as exc:
        raise CaptureOneRejected(missing_code, path=str(path)) from exc
    if size <= 0 or size > MAX_CURRENT_INDEX_BYTES:
        raise CaptureOneRejected(
            "CURRENT_INDEX_SIZE_INVALID",
            path=str(path),
            size_bytes=size,
            max_bytes=MAX_CURRENT_INDEX_BYTES,
        )
    return logical.read_bytes()


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
    from scripts.refresh_prejump_upcoming import stable_race_id
    from utils.csv_metadata import canonical_thedogs_race_identity

    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
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
        aliases = raw.get("race_id_aliases") or []
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
            or any(not isinstance(value, str) or not value for value in aliases)
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
            "race_time": str(raw.get("race_time") or ""),
            "race_url": race_url,
            "venue": venue,
        }
        if stable_race_id(row) != race_id or race_id in identities:
            raise CaptureOneRejected(
                "CURRENT_INDEX_INVALID", reason="race_id_mismatch_or_duplicate"
            )
        identities.add(race_id)
        normalized.append(row)
    return normalized


def _atomic_replace_canonical(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise CaptureOneRejected("CURRENT_INDEX_PATH_UNSAFE", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


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
        "schema_version": "collector_current_race_index_publish_v1",
        "status": "REJECTED",
        "index_path": str(index_path),
        "source_refresh_report_path": str(source_refresh_report_path),
        "run_id": run_id,
    }
    try:
        source_raw = _safe_file_bytes(
            source_refresh_report_path,
            evidence_root=evidence_root,
            missing_code="CURRENT_INDEX_SOURCE_MISSING",
        )
        source = json.loads(source_raw)
        if not isinstance(source, Mapping):
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID")
        source_generated_at = datetime.fromisoformat(str(source["generated_at"]))
        if (
            source_generated_at.tzinfo is None
            or source_generated_at.utcoffset() is None
        ):
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID")
        races = _normalize_current_index_rows(source, max_races=max_races)
        packet = {
            "schema_version": CURRENT_RACE_INDEX_SCHEMA,
            "run_id": run_id,
            "source_generated_at": source_generated_at.isoformat(),
            "source_refresh_report_path": str(source_refresh_report_path.resolve()),
            "source_refresh_report_sha256": sha256_bytes(source_raw),
            "race_count": len(races),
            "max_races": max_races,
            "races": races,
        }
        root = evidence_root.resolve()
        if not index_path.absolute().parent.resolve().is_relative_to(root):
            raise CaptureOneRejected(
                "CURRENT_INDEX_PATH_UNSAFE", path=str(index_path)
            )
        _atomic_replace_canonical(index_path, packet)
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
            "race_count": len(races),
            "source_generated_at": source_generated_at.isoformat(),
            "source_refresh_report_sha256": packet[
                "source_refresh_report_sha256"
            ],
        }
    )
    return report


def bounded_current_race_index(
    *,
    current_time: datetime,
    timeout_seconds: float,
    index_path: Path,
    evidence_root: Path,
    max_age_seconds: int,
    max_races: int = MAX_CURRENT_INDEX_RACES,
) -> list[Mapping[str, Any]]:
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

    try:
        signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        packet_raw = _safe_file_bytes(
            index_path,
            evidence_root=evidence_root,
            missing_code="CURRENT_INDEX_UNAVAILABLE",
        )
        packet = json.loads(packet_raw)
        if (
            not isinstance(packet, Mapping)
            or packet.get("schema_version") != CURRENT_RACE_INDEX_SCHEMA
            or canonical_bytes(packet) != packet_raw
            or packet.get("max_races") != max_races
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
        if age_seconds < -60 or age_seconds > max_age_seconds:
            raise CaptureOneRejected(
                "CURRENT_INDEX_STALE",
                age_seconds=age_seconds,
                max_age_seconds=max_age_seconds,
            )
        source_path = Path(str(packet["source_refresh_report_path"]))
        source_raw = _safe_file_bytes(
            source_path,
            evidence_root=evidence_root,
            missing_code="CURRENT_INDEX_SOURCE_MISSING",
        )
        if sha256_bytes(source_raw) != packet.get("source_refresh_report_sha256"):
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_CHANGED")
        source = json.loads(source_raw)
        if not isinstance(source, Mapping):
            raise CaptureOneRejected("CURRENT_INDEX_SOURCE_INVALID")
        races = _normalize_current_index_rows(source, max_races=max_races)
        if packet.get("race_count") != len(races) or packet.get("races") != races:
            raise CaptureOneRejected("CURRENT_INDEX_INVALID")
    except _DiscoveryTimedOut as exc:
        raise CaptureOneRejected(
            "DISCOVERY_TIMEOUT", budget_seconds=timeout_seconds
        ) from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CaptureOneRejected("CURRENT_INDEX_INVALID") from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    return races


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
