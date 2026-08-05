"""Immutable atomic evidence sealing for the isolated manual capture lane."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from src.predictor.manual_independent_capture import (
    ARTIFACT_PATH_BY_ROLE,
    AUTHORITY_PROFILE,
    CONTRACT_VERSION,
    DOWNSTREAM_ADMISSIBILITY,
    SAFETY_FIELDS,
    SOURCE_PATH_BY_CLASS,
    canonical_bytes,
    canonical_sha256,
    parse_canonical_json,
    validate_config,
    validate_terminal_artifact,
)
from src.predictor.manual_independent_capture_executor import ManualCaptureExecution
from src.predictor.on_demand import (
    PredictionBlocked,
    canonical_runner_set,
    sealed_runner_set_sha256,
    sha256_bytes,
)

EVIDENCE_BUNDLE_SCHEMA_VERSION = "manual_independent_evidence_bundle_v1"
EVIDENCE_MANIFEST_SCHEMA_VERSION = "manual_independent_evidence_manifest_v1"
NORMALIZED_ODDS_SCHEMA_VERSION = "manual_independent_normalized_odds_v1"
SEALER_IMPLEMENTATION_VERSION = "manual_independent_evidence_sealer_v1"
PUBLICATION_PROTOCOL = "same_filesystem_fsync_atomic_directory_rename_v1"
SEALED_ROOT_NAME = "sealed-evidence"
MANIFEST_FILENAME = "manifest.json"
_LOCK_FILENAME = ".evidence-seal.lock"
_STAGE_PREFIX = ".tmp-evidence-"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_OUTCOME_KEYS = frozenset(
    {
        "final_position",
        "final_positions",
        "finish",
        "finished",
        "finishes",
        "finish_position",
        "finish_positions",
        "finishing",
        "finishing_order",
        "finishing_orders",
        "finishing_position",
        "finishing_positions",
        "outcome",
        "outcomes",
        "place",
        "placed",
        "places",
        "placing",
        "placings",
        "position",
        "positions",
        "rank",
        "ranks",
        "result",
        "results",
        "win",
        "wins",
        "won",
        "winner",
        "winners",
        "winning_box",
        "winning_boxes",
    }
)
_SCHEMA_RELATIVES = {
    "config_schema_sha256": Path(
        "configs/prediction/manual-independent-capture-v1/config.schema.json"
    ),
    "terminal_schema_sha256": Path(
        "configs/prediction/manual-independent-capture-v1/terminal-artifact.schema.json"
    ),
    "evidence_bundle_schema_sha256": Path(
        "configs/prediction/manual-independent-capture-v1/evidence-bundle.schema.json"
    ),
    "evidence_manifest_schema_sha256": Path(
        "configs/prediction/manual-independent-capture-v1/evidence-manifest.schema.json"
    ),
}
_FIXED_MEMBER_PATHS = (
    "bundle.json",
    "normalized/odds.json",
    "producer/terminal.json",
    "source/raw.bin",
)


class CancellationToken(Protocol):
    def is_set(self) -> bool: ...


class ManualEvidenceRejected(RuntimeError):
    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class SealExpectations:
    source_commit: str
    source_tree: str
    model_sha256: str
    config_sha256: str
    run_id: str
    request_id: str
    request_sha256: str
    race_identity_sha256: str
    runner_set_sha256: str
    odds_sha256: str
    source_sha256: str
    source_timestamp: str
    final_url: str
    status_code: int
    content_type: str
    terminal_sha256: str
    cleanup_sha256: str


@dataclass(frozen=True)
class SealingIdentity:
    source_commit: str
    source_tree: str
    executor_sha256: str
    sealer_sha256: str
    config_schema_sha256: str
    terminal_schema_sha256: str
    evidence_bundle_schema_sha256: str
    evidence_manifest_schema_sha256: str


@dataclass(frozen=True)
class SealedManualEvidence:
    bundle_dir: Path
    bundle: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_sha256: str
    replayed: bool


def _reject(code: str, **details: Any) -> ManualEvidenceRejected:
    return ManualEvidenceRejected(code, **details)


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _reject("CONTRACT_FIELDS_INVALID", field=label)
    return value


def _hash(value: Any, field: str, *, git_object: bool = False) -> str:
    pattern = _GIT_OBJECT_RE if git_object else _SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _reject("HASH_INVALID", field=field)
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise _reject("TIMESTAMP_INVALID", field=field)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _reject("TIMESTAMP_INVALID", field=field) from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.microsecond != 0
        or parsed.isoformat() != value
    ):
        raise _reject("TIMESTAMP_INVALID", field=field)
    return parsed


def _is_plain_directory(path: Path, *, parent: Path | None = None) -> bool:
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


def _read_regular(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _reject("UNSAFE_PATH", path=str(path)) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > max_bytes
        ):
            raise _reject("UNSAFE_PATH", path=str(path))
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw or len(raw) > max_bytes:
            raise _reject("MEMBER_BYTES_INVALID", path=str(path))
        return raw
    finally:
        os.close(descriptor)


def _read_relative_regular(root: Path, relative: str, *, max_bytes: int) -> bytes:
    member = Path(relative)
    if (
        not relative
        or member.is_absolute()
        or member.as_posix() != relative
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise _reject("UNSAFE_PATH", path=relative)
    directory_flags = os.O_RDONLY | os.O_CLOEXEC
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        descriptors.append(os.open(root, directory_flags))
        for part in member.parts[:-1]:
            descriptors.append(
                os.open(part, directory_flags, dir_fd=descriptors[-1])
            )
        file_flags = os.O_RDONLY | os.O_CLOEXEC
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            member.parts[-1], file_flags, dir_fd=descriptors[-1]
        )
        descriptors.append(descriptor)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > max_bytes
        ):
            raise _reject("UNSAFE_PATH", path=relative)
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw or len(raw) > max_bytes:
            raise _reject("MEMBER_BYTES_INVALID", path=relative)
        return raw
    except OSError as exc:
        raise _reject("UNSAFE_PATH", path=relative) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _write_once(root: Path, relative: str, raw: bytes) -> None:
    member = Path(relative)
    if (
        member.is_absolute()
        or member.as_posix() != relative
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise _reject("UNSAFE_PATH", path=relative)
    parent = root
    for part in member.parts[:-1]:
        parent = parent / part
        try:
            os.mkdir(parent, 0o700)
        except FileExistsError:
            pass
        if not _is_plain_directory(parent, parent=root):
            raise _reject("UNSAFE_PATH", path=str(parent))
    target = root / member
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o400)
    except OSError as exc:
        raise _reject("PUBLICATION_FAILED", path=relative) from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(target, 0o444, follow_symlinks=False)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_and_fsync_tree(root: Path) -> None:
    directories = [root]
    for current, child_dirs, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in [*child_dirs, *files]:
            candidate = current_path / name
            if candidate.is_symlink():
                raise _reject("UNSAFE_PATH", path=str(candidate))
        for name in child_dirs:
            directories.append(current_path / name)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(directory, 0o555, follow_symlinks=False)
        _fsync_directory(directory)


def _remove_stale_tree(path: Path, *, parent: Path) -> None:
    if not _is_plain_directory(path, parent=parent):
        raise _reject("UNSAFE_STALE_TEMP", path=str(path))
    os.chmod(path, 0o700, follow_symlinks=False)
    for entry in os.scandir(path):
        child = Path(entry.path)
        if entry.is_symlink():
            raise _reject("UNSAFE_STALE_TEMP", path=str(child))
        if entry.is_dir(follow_symlinks=False):
            _remove_stale_tree(child, parent=path)
        elif entry.is_file(follow_symlinks=False):
            if entry.stat(follow_symlinks=False).st_nlink != 1:
                raise _reject("UNSAFE_STALE_TEMP", path=str(child))
            os.chmod(child, 0o600, follow_symlinks=False)
            os.unlink(child)
        else:
            raise _reject("UNSAFE_STALE_TEMP", path=str(child))
    os.chmod(path, 0o700, follow_symlinks=False)
    os.rmdir(path)


def _clean_stale_stages(sealed_root: Path) -> None:
    for entry in os.scandir(sealed_root):
        if not entry.name.startswith(_STAGE_PREFIX):
            continue
        _remove_stale_tree(Path(entry.path), parent=sealed_root)


def _open_seal_lock(run_dir: Path) -> int:
    path = run_dir / _LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise _reject("UNSAFE_PATH", path=str(path))
    return descriptor


def _check_cancelled(token: CancellationToken | None) -> None:
    if token is not None and token.is_set():
        raise _reject("CANCELLED")


def _schema_root(repo_root: Path) -> Path:
    if not _is_plain_directory(repo_root):
        raise _reject("UNSAFE_PATH", path=str(repo_root))
    return repo_root.resolve(strict=True)


def build_sealing_identity(
    *, repo_root: Path, source_commit: str, source_tree: str
) -> SealingIdentity:
    """Hash the reviewed implementation and schema bytes used by the seal."""

    _hash(source_commit, "source_commit", git_object=True)
    _hash(source_tree, "source_tree", git_object=True)
    root = _schema_root(repo_root)
    executor_path = root / "src/predictor/manual_independent_capture_executor.py"
    sealer_path = root / "src/predictor/manual_independent_capture_sealer.py"
    values: dict[str, str] = {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "executor_sha256": sha256_bytes(
            _read_relative_regular(
                root,
                executor_path.relative_to(root).as_posix(),
                max_bytes=_MAX_JSON_BYTES,
            )
        ),
        "sealer_sha256": sha256_bytes(
            _read_relative_regular(
                root,
                sealer_path.relative_to(root).as_posix(),
                max_bytes=_MAX_JSON_BYTES,
            )
        ),
    }
    for field, relative in _SCHEMA_RELATIVES.items():
        values[field] = sha256_bytes(
            _read_relative_regular(
                root, relative.as_posix(), max_bytes=_MAX_JSON_BYTES
            )
        )
    return SealingIdentity(**values)


def expectations_from_execution(execution: ManualCaptureExecution) -> SealExpectations:
    artifact = execution.artifact
    provenance = artifact["provenance"]
    request = artifact["request"]
    source_row = provenance["source_files"][0] if provenance["source_files"] else {}
    response = execution.source_response
    return SealExpectations(
        source_commit=provenance["source_commit"],
        source_tree=provenance["source_tree"],
        model_sha256=provenance["model_sha256"],
        config_sha256=provenance["config_sha256"],
        run_id=artifact["run_id"],
        request_id=request["request_id"],
        request_sha256=provenance["request_sha256"],
        race_identity_sha256=provenance["race_identity_sha256"] or "0" * 64,
        runner_set_sha256=provenance["runner_set_sha256"] or "0" * 64,
        odds_sha256=provenance["odds_sha256"] or "0" * 64,
        source_sha256=source_row.get("sha256", "0" * 64),
        source_timestamp=source_row.get(
            "source_timestamp", artifact["timing"]["readiness_checked_at"]
        ),
        final_url=(
            response.final_url if response is not None else request["requested_race_url"]
        ),
        status_code=response.status_code if response is not None else 0,
        content_type=response.content_type if response is not None else "unavailable",
        terminal_sha256=sha256_bytes(canonical_bytes(execution.artifact)),
        cleanup_sha256=canonical_sha256(asdict(execution.cleanup)),
    )


def _validate_expectations(expected: SealExpectations) -> None:
    _hash(expected.source_commit, "expected.source_commit", git_object=True)
    _hash(expected.source_tree, "expected.source_tree", git_object=True)
    for field in (
        "model_sha256",
        "config_sha256",
        "request_sha256",
        "race_identity_sha256",
        "runner_set_sha256",
        "odds_sha256",
        "source_sha256",
        "terminal_sha256",
        "cleanup_sha256",
    ):
        _hash(getattr(expected, field), f"expected.{field}")
    for field in ("run_id", "request_id"):
        try:
            parsed = uuid.UUID(getattr(expected, field))
        except (AttributeError, ValueError) as exc:
            raise _reject("IDENTIFIER_INVALID", field=f"expected.{field}") from exc
        if parsed.version != 4 or str(parsed) != getattr(expected, field):
            raise _reject("IDENTIFIER_INVALID", field=f"expected.{field}")
    _timestamp(expected.source_timestamp, "expected.source_timestamp")
    if (
        not isinstance(expected.final_url, str)
        or not expected.final_url
        or expected.final_url != expected.final_url.strip()
        or isinstance(expected.status_code, bool)
        or not isinstance(expected.status_code, int)
        or not isinstance(expected.content_type, str)
        or not expected.content_type
        or expected.content_type != expected.content_type.strip()
        or len(expected.content_type) > 256
        or any(
            not 32 <= ord(character) <= 126
            for character in expected.content_type
        )
    ):
        raise _reject("SOURCE_PROVENANCE_INCOMPLETE")


def _validate_identity(identity: SealingIdentity) -> dict[str, str]:
    values = asdict(identity)
    _hash(values["source_commit"], "identity.source_commit", git_object=True)
    _hash(values["source_tree"], "identity.source_tree", git_object=True)
    for field in set(values) - {"source_commit", "source_tree"}:
        _hash(values[field], f"identity.{field}")
    return values


def _producer_members(run_dir: Path, artifact: Mapping[str, Any]) -> dict[str, bytes]:
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        raise _reject("CONTRACT_FIELDS_INVALID", field="artifact.provenance")
    source_rows = provenance.get("source_files")
    artifact_rows = provenance.get("artifact_hashes")
    if not isinstance(source_rows, list) or not isinstance(artifact_rows, list):
        raise _reject("CONTRACT_FIELDS_INVALID", field="artifact.provenance.members")
    paths: list[str] = []
    for index, item in enumerate(source_rows):
        row = _exact(
            item,
            {
                "path",
                "content_class",
                "outcome_scope",
                "race_url",
                "race_identity_sha256",
                "source_timestamp",
                "bytes",
                "sha256",
            },
            f"artifact.provenance.source_files[{index}]",
        )
        content_class = row["content_class"]
        if (
            not isinstance(content_class, str)
            or content_class not in SOURCE_PATH_BY_CLASS
            or row["path"] != SOURCE_PATH_BY_CLASS[content_class]
        ):
            raise _reject("UNSAFE_PATH", path=row.get("path"))
        paths.append(row["path"])
    for index, item in enumerate(artifact_rows):
        row = _exact(
            item,
            {"role", "path", "bytes", "sha256"},
            f"artifact.provenance.artifact_hashes[{index}]",
        )
        role = row["role"]
        if (
            not isinstance(role, str)
            or role not in ARTIFACT_PATH_BY_ROLE
            or row["path"] != ARTIFACT_PATH_BY_ROLE[role]
        ):
            raise _reject("UNSAFE_PATH", path=row.get("path"))
        paths.append(row["path"])
    if len(paths) != len(set(paths)):
        raise _reject("ARTIFACT_MEMBERS_INVALID")
    members = {
        relative: _read_relative_regular(
            run_dir, relative, max_bytes=_MAX_SOURCE_BYTES
        )
        for relative in paths
    }
    expected_files = {*paths, "terminal.json", _LOCK_FILENAME}
    for current, directories, filenames in os.walk(run_dir, followlinks=False):
        current_path = Path(current)
        relative_root = current_path.relative_to(run_dir)
        if relative_root.parts and relative_root.parts[0] == SEALED_ROOT_NAME:
            directories[:] = []
            continue
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise _reject("UNSAFE_PATH", path=str(path))
        for name in filenames:
            path = current_path / name
            relative = path.relative_to(run_dir).as_posix()
            if relative not in expected_files or path.is_symlink():
                raise _reject("UNEXPECTED_PRODUCER_MEMBER", path=relative)
    return members


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _has_outcome_token(value: str) -> bool:
    normalized = _normalized_key(value)
    if normalized in _OUTCOME_KEYS:
        return True
    return any(token in _OUTCOME_KEYS for token in normalized.split("_") if token)


def _json_has_outcome_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _has_outcome_token(str(key)) or _json_has_outcome_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_json_has_outcome_key(item) for item in value)
    return False


def _reject_outcome_material(raw: bytes, content_type: str) -> None:
    media_type = content_type.split(";", 1)[0].strip().lower()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _reject("SOURCE_CONTENT_INVALID", reason="utf8") from exc
    if media_type in {"application/json", "text/json"}:
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise _reject("SOURCE_CONTENT_INVALID", reason="json") from exc
        if _json_has_outcome_key(value):
            raise _reject("OUTCOME_MATERIAL_FORBIDDEN")
    elif media_type in {"text/csv", "application/csv"}:
        try:
            header = next(csv.reader(text.splitlines()))
        except (csv.Error, StopIteration) as exc:
            raise _reject("SOURCE_CONTENT_INVALID", reason="csv") from exc
        if any(_has_outcome_token(field) for field in header):
            raise _reject("OUTCOME_MATERIAL_FORBIDDEN")
    elif media_type == "text/html":
        if any(
            _has_outcome_token(token)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
        ):
            raise _reject("OUTCOME_MATERIAL_FORBIDDEN")
    else:
        raise _reject("SOURCE_CONTENT_TYPE_INVALID")


def _source_decimal(value: str, *, field: str) -> Decimal:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _reject("ODDS_PROVENANCE_AMBIGUOUS", field=field)
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise _reject("ODDS_PROVENANCE_AMBIGUOUS", field=field) from exc
    if not parsed.is_finite() or parsed <= 1:
        raise _reject("ODDS_PROVENANCE_AMBIGUOUS", field=field)
    return parsed


def _validate_parsed_odds(
    raw: bytes, content_type: str, capture_rows: Any
) -> tuple[str, str]:
    if not isinstance(capture_rows, list) or not capture_rows:
        raise _reject("ODDS_PROVENANCE_AMBIGUOUS")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _reject("ODDS_PROVENANCE_AMBIGUOUS", reason="utf8") from exc
    media_type = content_type.split(";", 1)[0].strip().lower()
    parsed_rows: list[dict[str, Any]] = []
    parser: str
    if media_type in {"text/csv", "application/csv"}:
        try:
            reader = csv.DictReader(text.splitlines())
            if reader.fieldnames != ["box", "dog", "decimal_odds"]:
                raise _reject("ODDS_PROVENANCE_AMBIGUOUS", reason="csv_header")
            raw_rows = list(reader)
        except csv.Error as exc:
            raise _reject("ODDS_PROVENANCE_AMBIGUOUS", reason="csv") from exc
        parser = "manual_fixture_csv_odds_v1"
        for index, row in enumerate(raw_rows):
            if set(row) != {"box", "dog", "decimal_odds"} or any(
                value is None for value in row.values()
            ):
                raise _reject("ODDS_PROVENANCE_AMBIGUOUS", field=f"row[{index}]")
            try:
                box = int(row["box"])
            except (TypeError, ValueError) as exc:
                raise _reject(
                    "ODDS_PROVENANCE_AMBIGUOUS", field=f"row[{index}].box"
                ) from exc
            if str(box) != row["box"] or not row["dog"] or row["dog"] != row[
                "dog"
            ].strip():
                raise _reject("ODDS_PROVENANCE_AMBIGUOUS", field=f"row[{index}]")
            parsed_rows.append(
                {
                    "box_number": box,
                    "display_name": row["dog"],
                    "decimal_odds": _source_decimal(
                        row["decimal_odds"], field=f"row[{index}].decimal_odds"
                    ),
                }
            )
    elif media_type in {"application/json", "text/json"}:
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise _reject("ODDS_PROVENANCE_AMBIGUOUS", reason="json") from exc
        envelope = _exact(value, {"runners"}, "source.odds")
        if not isinstance(envelope["runners"], list):
            raise _reject("ODDS_PROVENANCE_AMBIGUOUS", reason="json_runners")
        parser = "manual_fixture_json_odds_v1"
        for index, item in enumerate(envelope["runners"]):
            row = _exact(
                item,
                {"box_number", "display_name", "decimal_odds"},
                f"source.odds.runners[{index}]",
            )
            if (
                isinstance(row["box_number"], bool)
                or not isinstance(row["box_number"], int)
                or not isinstance(row["display_name"], str)
                or not row["display_name"]
                or row["display_name"] != row["display_name"].strip()
                or isinstance(row["decimal_odds"], bool)
                or not isinstance(row["decimal_odds"], (int, float))
            ):
                raise _reject("ODDS_PROVENANCE_AMBIGUOUS", field=f"row[{index}]")
            parsed_rows.append(
                {
                    "box_number": row["box_number"],
                    "display_name": row["display_name"],
                    "decimal_odds": _source_decimal(
                        str(row["decimal_odds"]),
                        field=f"row[{index}].decimal_odds",
                    ),
                }
            )
    else:
        raise _reject("ODDS_PROVENANCE_AMBIGUOUS", reason="unsupported_content_type")
    if len(parsed_rows) != len(capture_rows):
        raise _reject("ODDS_PROVENANCE_AMBIGUOUS", reason="runner_count")
    bound_rows = []
    for index, (parsed, captured) in enumerate(
        zip(parsed_rows, capture_rows, strict=True)
    ):
        if (
            not isinstance(captured, Mapping)
            or parsed["box_number"] != captured.get("box_number")
            or parsed["display_name"] != captured.get("display_name")
            or isinstance(captured.get("decimal_odds"), bool)
            or not isinstance(captured.get("decimal_odds"), (int, float))
            or parsed["decimal_odds"]
            != _source_decimal(
                str(captured.get("decimal_odds")),
                field=f"capture_rows[{index}].decimal_odds",
            )
        ):
            raise _reject("ODDS_PROVENANCE_AMBIGUOUS", field=f"row[{index}]")
        bound_rows.append(
            {
                "box_number": parsed["box_number"],
                "display_name": parsed["display_name"],
                "decimal_odds": captured["decimal_odds"],
            }
        )
    return parser, canonical_sha256(bound_rows)


def _validate_response(
    execution: ManualCaptureExecution,
    *,
    artifact: Mapping[str, Any],
    source_raw: bytes,
    expected: SealExpectations,
) -> dict[str, Any]:
    response = execution.source_response
    race = artifact["request"]["selected_race"]
    source_row = artifact["provenance"]["source_files"][0]
    if response is None or race is None:
        raise _reject("SOURCE_PROVENANCE_INCOMPLETE")
    media_type = response.content_type.split(";", 1)[0].strip().lower()
    permitted = {
        "prejump_form": {"text/csv", "application/csv"},
        "prejump_sidecar": {"application/json", "text/json"},
        "prejump_race_source": {"application/json", "text/html", "text/json"},
    }
    if (
        response.final_url != race["url"]
        or response.final_url != source_row["race_url"]
        or isinstance(response.status_code, bool)
        or response.status_code != 200
        or not response.content_type
        or response.content_type != response.content_type.strip()
        or len(response.content_type) > 256
        or any(
            not 32 <= ord(character) <= 126
            for character in response.content_type
        )
        or media_type not in permitted[source_row["content_class"]]
        or response.body_sha256 != sha256_bytes(source_raw)
        or response.body_sha256 != source_row["sha256"]
        or response.final_url != expected.final_url
        or response.status_code != expected.status_code
        or response.content_type != expected.content_type
        or response.body_sha256 != expected.source_sha256
        or source_row["source_timestamp"] != expected.source_timestamp
    ):
        raise _reject("SOURCE_PROVENANCE_MISMATCH")
    _reject_outcome_material(source_raw, response.content_type)
    odds_parser, parsed_odds_sha256 = _validate_parsed_odds(
        source_raw,
        response.content_type,
        artifact["capture"]["runner_set"],
    )
    metadata = {
        "final_url": response.final_url,
        "status_code": response.status_code,
        "content_type": response.content_type,
    }
    return {
        **metadata,
        "metadata_sha256": canonical_sha256(metadata),
        "odds_parser": odds_parser,
        "parsed_odds_sha256": parsed_odds_sha256,
    }


def _validate_execution(
    execution: ManualCaptureExecution,
    *,
    config: Mapping[str, Any],
    forbidden_paths: Mapping[str, str],
    expected: SealExpectations,
) -> tuple[dict[str, Any], bytes, dict[str, bytes], dict[str, Any]]:
    if (
        execution.artifact.get("terminal")
        != {"status": "CAPTURE_READY", "failure_code": None}
        or not execution.cleanup.confirmed
        or not execution.cleanup.leader_reaped
        or not execution.cleanup.process_group_absent
    ):
        raise _reject("CAPTURE_NOT_SEALABLE")
    _validate_expectations(expected)
    validated_config = validate_config(config, forbidden_paths=forbidden_paths)
    run_dir = execution.run_dir
    runs_root = Path(validated_config["paths"]["runs_root"])
    if (
        not run_dir.is_absolute()
        or run_dir.parent != runs_root
        or not _is_plain_directory(run_dir, parent=runs_root)
        or execution.terminal_path != run_dir / "terminal.json"
    ):
        raise _reject("UNSAFE_PATH", path=str(run_dir))
    try:
        parsed_run_id = uuid.UUID(run_dir.name)
    except ValueError as exc:
        raise _reject("IDENTIFIER_INVALID", field="run_dir") from exc
    if parsed_run_id.version != 4 or str(parsed_run_id) != run_dir.name:
        raise _reject("IDENTIFIER_INVALID", field="run_dir")
    terminal_raw = _read_relative_regular(
        run_dir, "terminal.json", max_bytes=_MAX_JSON_BYTES
    )
    if sha256_bytes(terminal_raw) != expected.terminal_sha256:
        raise _reject("TERMINAL_ARTIFACT_MISMATCH")
    artifact = parse_canonical_json(terminal_raw, max_bytes=_MAX_JSON_BYTES)
    if artifact != execution.artifact:
        raise _reject("TERMINAL_ARTIFACT_MISMATCH")
    members = _producer_members(run_dir, artifact)
    validated = validate_terminal_artifact(
        artifact,
        config=validated_config,
        forbidden_paths=forbidden_paths,
        member_bytes=members,
        expected_source_commit=expected.source_commit,
        expected_source_tree=expected.source_tree,
        expected_model_sha256=expected.model_sha256,
        expected_source_files=deepcopy(artifact["provenance"]["source_files"]),
        expected_runner_set_sha256=expected.runner_set_sha256,
        expected_odds_sha256=expected.odds_sha256,
        expected_run_id=expected.run_id,
        expected_request_id=expected.request_id,
        expected_request_sha256=expected.request_sha256,
        seen_run_ids=set(),
        seen_request_ids=set(),
        seen_request_sha256s=set(),
    )
    if (
        validated["terminal"]
        != {"status": "CAPTURE_READY", "failure_code": None}
        or validated["attempt"]
        != {"attempt_count": 1, "source_attempt_count": 1}
        or validated["timing"]["cancel_requested_at"] is not None
        or not execution.cleanup.confirmed
        or not execution.cleanup.leader_reaped
        or not execution.cleanup.process_group_absent
        or len(validated["provenance"]["source_files"]) != 1
        or validated["provenance"]["config_sha256"] != expected.config_sha256
        or validated["provenance"]["race_identity_sha256"]
        != expected.race_identity_sha256
        or canonical_sha256(asdict(execution.cleanup)) != expected.cleanup_sha256
    ):
        raise _reject("CAPTURE_NOT_SEALABLE")
    source_row = validated["provenance"]["source_files"][0]
    source_raw = members[source_row["path"]]
    response = _validate_response(
        execution,
        artifact=validated,
        source_raw=source_raw,
        expected=expected,
    )
    return validated, terminal_raw, members, response


def _build_documents(
    *,
    artifact: Mapping[str, Any],
    terminal_raw: bytes,
    source_raw: bytes,
    response: Mapping[str, Any],
    identity: SealingIdentity,
    execution: ManualCaptureExecution,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, bytes]]:
    race = artifact["request"]["selected_race"]
    timing = artifact["timing"]
    source_row = artifact["provenance"]["source_files"][0]
    runners = deepcopy(artifact["capture"]["runner_set"])
    normalized_seed = {
        "schema_version": NORMALIZED_ODDS_SCHEMA_VERSION,
        "race_identity_sha256": artifact["provenance"]["race_identity_sha256"],
        "capture_timestamp": timing["capture_timestamp"],
        "source_timestamp": source_row["source_timestamp"],
        "runner_set_sha256": artifact["provenance"]["runner_set_sha256"],
        "odds_sha256": artifact["provenance"]["odds_sha256"],
        "runners": runners,
    }
    cleanup = asdict(execution.cleanup)
    bundle_seed = {
        "run_id": artifact["run_id"],
        "request_sha256": artifact["provenance"]["request_sha256"],
        "race_identity_sha256": artifact["provenance"]["race_identity_sha256"],
        "runner_set_sha256": artifact["provenance"]["runner_set_sha256"],
        "odds_sha256": artifact["provenance"]["odds_sha256"],
        "source_sha256": source_row["sha256"],
        "terminal_sha256": sha256_bytes(terminal_raw),
        "response_metadata_sha256": response["metadata_sha256"],
        "parsed_odds_sha256": response["parsed_odds_sha256"],
        "cleanup": cleanup,
        "implementation": _validate_identity(identity),
    }
    bundle_id = canonical_sha256(bundle_seed)
    normalized = {**normalized_seed, "bundle_id": bundle_id}
    bundle = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "implementation_version": SEALER_IMPLEMENTATION_VERSION,
        "bundle_id": bundle_id,
        "safety": dict(SAFETY_FIELDS),
        "authority_profile": AUTHORITY_PROFILE,
        "race": deepcopy(race),
        "race_identity_sha256": artifact["provenance"]["race_identity_sha256"],
        "runner_set": [
            {key: value for key, value in row.items() if key != "decimal_odds"}
            for row in runners
        ],
        "runner_set_sha256": artifact["provenance"]["runner_set_sha256"],
        "timing": {
            "capture_started_at": timing["submitted_at"],
            "readiness_checked_at": timing["readiness_checked_at"],
            "capture_completed_at": timing["terminal_at"],
            "capture_timestamp": timing["capture_timestamp"],
            "source_timestamp": source_row["source_timestamp"],
            "minimum_prejump_margin_seconds": artifact["request"][
                "minimum_prejump_margin_seconds"
            ],
            "final_prejump_margin_seconds": timing[
                "capture_prejump_margin_seconds"
            ],
        },
        "attempt": deepcopy(artifact["attempt"]),
        "source": {
            "content_class": source_row["content_class"],
            **deepcopy(dict(response)),
            "raw_path": "source/raw.bin",
            "bytes": len(source_raw),
            "sha256": source_row["sha256"],
        },
        "normalized_odds": {
            "path": "normalized/odds.json",
            "sha256": sha256_bytes(canonical_bytes(normalized)),
            "odds_sha256": artifact["provenance"]["odds_sha256"],
        },
        "producer": {
            "run_id": artifact["run_id"],
            "request_id": artifact["request"]["request_id"],
            "request_sha256": artifact["provenance"]["request_sha256"],
            "terminal_path": "producer/terminal.json",
            "terminal_sha256": sha256_bytes(terminal_raw),
            "config_sha256": artifact["provenance"]["config_sha256"],
            "model_sha256": artifact["provenance"]["model_sha256"],
        },
        "implementation": _validate_identity(identity),
        "cleanup": {**cleanup, "cancellation_status": "not_requested"},
        "closure": {
            "bundle_closed": True,
            "outcome_accessed": False,
            "canonical_accessed": False,
            "canonical_write_claimed": False,
            "phase7_accessed": False,
            "phase7_eligible": False,
            "downstream_admissibility": DOWNSTREAM_ADMISSIBILITY,
        },
        "publication": {
            "protocol": PUBLICATION_PROTOCOL,
            "manifest_path": MANIFEST_FILENAME,
        },
    }
    members = {
        "bundle.json": canonical_bytes(bundle),
        "normalized/odds.json": canonical_bytes(normalized),
        "producer/terminal.json": terminal_raw,
        "source/raw.bin": source_raw,
    }
    manifest_members = [
        {
            "path": path,
            "bytes": len(members[path]),
            "sha256": sha256_bytes(members[path]),
        }
        for path in _FIXED_MEMBER_PATHS
    ]
    manifest = {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "bundle_id": bundle_id,
        "safety": dict(SAFETY_FIELDS),
        "publication_protocol": PUBLICATION_PROTOCOL,
        "members": manifest_members,
    }
    return bundle_id, bundle, manifest, members


def _validate_bundle_document(
    bundle: Any,
    *,
    expected: SealExpectations,
    expected_identity: SealingIdentity,
) -> dict[str, Any]:
    row = _exact(
        bundle,
        {
            "schema_version",
            "contract_version",
            "implementation_version",
            "bundle_id",
            "safety",
            "authority_profile",
            "race",
            "race_identity_sha256",
            "runner_set",
            "runner_set_sha256",
            "timing",
            "attempt",
            "source",
            "normalized_odds",
            "producer",
            "implementation",
            "cleanup",
            "closure",
            "publication",
        },
        "bundle",
    )
    race = _exact(
        row["race"],
        {
            "url",
            "race_id",
            "race_date",
            "venue",
            "venue_slug",
            "race_number",
            "scheduled_start",
        },
        "bundle.race",
    )
    producer = _exact(
        row["producer"],
        {
            "run_id",
            "request_id",
            "request_sha256",
            "terminal_path",
            "terminal_sha256",
            "config_sha256",
            "model_sha256",
        },
        "bundle.producer",
    )
    source = _exact(
        row["source"],
        {
            "content_class",
            "final_url",
            "status_code",
            "content_type",
            "metadata_sha256",
            "odds_parser",
            "parsed_odds_sha256",
            "raw_path",
            "bytes",
            "sha256",
        },
        "bundle.source",
    )
    normalized_odds = _exact(
        row["normalized_odds"],
        {"path", "sha256", "odds_sha256"},
        "bundle.normalized_odds",
    )
    cleanup = _exact(
        row["cleanup"],
        {
            "pid",
            "pgid",
            "reason",
            "term_sent",
            "kill_sent",
            "leader_reaped",
            "process_group_absent",
            "confirmed",
            "cancellation_status",
        },
        "bundle.cleanup",
    )
    closure = _exact(
        row["closure"],
        {
            "bundle_closed",
            "outcome_accessed",
            "canonical_accessed",
            "canonical_write_claimed",
            "phase7_accessed",
            "phase7_eligible",
            "downstream_admissibility",
        },
        "bundle.closure",
    )
    publication = _exact(
        row["publication"], {"protocol", "manifest_path"}, "bundle.publication"
    )
    if not isinstance(row["runner_set"], list) or not 1 <= len(
        row["runner_set"]
    ) <= 10:
        raise _reject("RUNNER_SET_INVALID")
    runners = [
        dict(
            _exact(
                runner,
                {"box_number", "display_name", "identity", "source_native_runner_id"},
                f"bundle.runner_set[{index}]",
            )
        )
        for index, runner in enumerate(row["runner_set"])
    ]
    try:
        canonical_runner_set(runners, "bundle.runner_set")
    except PredictionBlocked as exc:
        raise _reject("RUNNER_SET_INVALID", source_code=exc.code) from exc
    cleanup_without_status = {
        key: value for key, value in cleanup.items() if key != "cancellation_status"
    }
    cleanup_identifiers_valid = all(
        value is None or (type(value) is int and value > 0)
        for value in (cleanup["pid"], cleanup["pgid"])
    )
    cleanup_reason_valid = (
        isinstance(cleanup["reason"], str)
        and re.fullmatch(r"[a-z0-9_]+", cleanup["reason"]) is not None
    )
    cleanup_flags_valid = all(
        type(cleanup[field]) is bool for field in ("term_sent", "kill_sent")
    )
    permitted_media = {
        "prejump_form": {"text/csv", "application/csv"},
        "prejump_sidecar": {"application/json", "text/json"},
        "prejump_race_source": {"application/json", "text/html", "text/json"},
    }
    if not isinstance(source["content_type"], str) or not isinstance(
        source["content_class"], str
    ):
        raise _reject("BUNDLE_CONTRACT_INVALID")
    media_type = source["content_type"].split(";", 1)[0].strip().lower()
    if (
        row["schema_version"] != EVIDENCE_BUNDLE_SCHEMA_VERSION
        or row["contract_version"] != CONTRACT_VERSION
        or row["implementation_version"] != SEALER_IMPLEMENTATION_VERSION
        or row["safety"] != dict(SAFETY_FIELDS)
        or row["authority_profile"] != AUTHORITY_PROFILE
        or canonical_sha256(race) != expected.race_identity_sha256
        or row["race_identity_sha256"] != expected.race_identity_sha256
        or sealed_runner_set_sha256(race, runners) != expected.runner_set_sha256
        or row["runner_set_sha256"] != expected.runner_set_sha256
        or producer["run_id"] != expected.run_id
        or producer["request_id"] != expected.request_id
        or producer["request_sha256"] != expected.request_sha256
        or producer["terminal_path"] != "producer/terminal.json"
        or producer["terminal_sha256"] != expected.terminal_sha256
        or producer["config_sha256"] != expected.config_sha256
        or producer["model_sha256"] != expected.model_sha256
        or row["implementation"] != _validate_identity(expected_identity)
        or row["attempt"] != {"attempt_count": 1, "source_attempt_count": 1}
        or source["final_url"] != expected.final_url
        or source["final_url"] != race["url"]
        or source["status_code"] != expected.status_code
        or source["status_code"] != 200
        or source["content_type"] != expected.content_type
        or media_type not in permitted_media.get(source["content_class"], set())
        or source["odds_parser"]
        not in {"manual_fixture_csv_odds_v1", "manual_fixture_json_odds_v1"}
        or not isinstance(source["parsed_odds_sha256"], str)
        or _SHA256_RE.fullmatch(source["parsed_odds_sha256"]) is None
        or source["raw_path"] != "source/raw.bin"
        or source["sha256"] != expected.source_sha256
        or isinstance(source["bytes"], bool)
        or not isinstance(source["bytes"], int)
        or not 1 <= source["bytes"] <= _MAX_SOURCE_BYTES
        or canonical_sha256(
            {
                "final_url": source["final_url"],
                "status_code": source["status_code"],
                "content_type": source["content_type"],
            }
        )
        != source["metadata_sha256"]
        or normalized_odds["path"] != "normalized/odds.json"
        or normalized_odds["odds_sha256"] != expected.odds_sha256
        or not cleanup_identifiers_valid
        or not cleanup_reason_valid
        or not cleanup_flags_valid
        or canonical_sha256(cleanup_without_status) != expected.cleanup_sha256
        or cleanup["confirmed"] is not True
        or cleanup["leader_reaped"] is not True
        or cleanup["process_group_absent"] is not True
        or cleanup["cancellation_status"] != "not_requested"
        or closure
        != {
            "bundle_closed": True,
            "outcome_accessed": False,
            "canonical_accessed": False,
            "canonical_write_claimed": False,
            "phase7_accessed": False,
            "phase7_eligible": False,
            "downstream_admissibility": DOWNSTREAM_ADMISSIBILITY,
        }
        or publication
        != {"protocol": PUBLICATION_PROTOCOL, "manifest_path": MANIFEST_FILENAME}
    ):
        raise _reject("BUNDLE_CONTRACT_INVALID")
    timing = _exact(
        row["timing"],
        {
            "capture_started_at",
            "readiness_checked_at",
            "capture_completed_at",
            "capture_timestamp",
            "source_timestamp",
            "minimum_prejump_margin_seconds",
            "final_prejump_margin_seconds",
        },
        "bundle.timing",
    )
    started = _timestamp(timing["capture_started_at"], "capture_started_at")
    readiness = _timestamp(timing["readiness_checked_at"], "readiness_checked_at")
    completed = _timestamp(timing["capture_completed_at"], "capture_completed_at")
    captured = _timestamp(timing["capture_timestamp"], "capture_timestamp")
    source_at = _timestamp(timing["source_timestamp"], "source_timestamp")
    scheduled = _timestamp(race["scheduled_start"], "race.scheduled_start")
    if (
        not started <= readiness <= source_at <= captured <= completed < scheduled
        or isinstance(timing["final_prejump_margin_seconds"], bool)
        or not isinstance(timing["final_prejump_margin_seconds"], int)
        or timing["final_prejump_margin_seconds"]
        != int((scheduled - captured).total_seconds())
        or timing["final_prejump_margin_seconds"]
        < timing["minimum_prejump_margin_seconds"]
        or timing["source_timestamp"] != expected.source_timestamp
    ):
        raise _reject("TIMING_INVALID")
    _hash(row["bundle_id"], "bundle.bundle_id")
    _hash(row["race_identity_sha256"], "bundle.race_identity_sha256")
    return deepcopy(dict(row))


def _manifest_document(value: Any, *, bundle_id: str) -> dict[str, Any]:
    manifest = _exact(
        value,
        {
            "schema_version",
            "contract_version",
            "bundle_id",
            "safety",
            "publication_protocol",
            "members",
        },
        "manifest",
    )
    if (
        manifest["schema_version"] != EVIDENCE_MANIFEST_SCHEMA_VERSION
        or manifest["contract_version"] != CONTRACT_VERSION
        or manifest["bundle_id"] != bundle_id
        or manifest["safety"] != dict(SAFETY_FIELDS)
        or manifest["publication_protocol"] != PUBLICATION_PROTOCOL
        or not isinstance(manifest["members"], list)
        or len(manifest["members"]) != len(_FIXED_MEMBER_PATHS)
    ):
        raise _reject("MANIFEST_INVALID")
    members = []
    for index, item in enumerate(manifest["members"]):
        member = _exact(item, {"path", "bytes", "sha256"}, f"members[{index}]")
        if (
            member["path"] != _FIXED_MEMBER_PATHS[index]
            or isinstance(member["bytes"], bool)
            or not isinstance(member["bytes"], int)
            or member["bytes"] <= 0
        ):
            raise _reject("MANIFEST_INVALID")
        _hash(member["sha256"], f"members[{index}].sha256")
        members.append(dict(member))
    return {**dict(manifest), "members": members}


def _enumerate_bundle_files(bundle_dir: Path) -> set[str]:
    files: set[str] = set()
    for current, directories, filenames in os.walk(bundle_dir, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise _reject("UNSAFE_PATH", path=str(path))
        for name in filenames:
            path = current_path / name
            if path.is_symlink():
                raise _reject("UNSAFE_PATH", path=str(path))
            files.add(path.relative_to(bundle_dir).as_posix())
    return files


def verify_manual_evidence_bundle(
    bundle_dir: Path,
    *,
    run_dir: Path,
    expected: SealExpectations,
    expected_identity: SealingIdentity,
) -> SealedManualEvidence:
    """Read and verify one fully published bundle without mutating it."""

    _validate_expectations(expected)
    _validate_identity(expected_identity)
    if (
        bundle_dir.name.startswith(_STAGE_PREFIX)
        or bundle_dir.parent != run_dir / SEALED_ROOT_NAME
        or not _is_plain_directory(run_dir)
        or not _is_plain_directory(bundle_dir, parent=run_dir)
    ):
        raise _reject("UNSAFE_PATH", path=str(bundle_dir))
    manifest_raw = _read_relative_regular(
        bundle_dir, MANIFEST_FILENAME, max_bytes=_MAX_JSON_BYTES
    )
    manifest_value = parse_canonical_json(manifest_raw, max_bytes=_MAX_JSON_BYTES)
    manifest = _manifest_document(manifest_value, bundle_id=bundle_dir.name)
    expected_files = {*_FIXED_MEMBER_PATHS, MANIFEST_FILENAME}
    if _enumerate_bundle_files(bundle_dir) != expected_files:
        raise _reject("PARTIAL_OR_EXTRA_OUTPUT")
    member_bytes: dict[str, bytes] = {}
    for member in manifest["members"]:
        raw = _read_relative_regular(
            bundle_dir,
            member["path"],
            max_bytes=_MAX_SOURCE_BYTES
            if member["path"] == "source/raw.bin"
            else _MAX_JSON_BYTES,
        )
        if len(raw) != member["bytes"] or sha256_bytes(raw) != member["sha256"]:
            raise _reject("HASH_DRIFT", path=member["path"])
        member_bytes[member["path"]] = raw
    bundle = _validate_bundle_document(
        parse_canonical_json(member_bytes["bundle.json"], max_bytes=_MAX_JSON_BYTES),
        expected=expected,
        expected_identity=expected_identity,
    )
    if bundle["bundle_id"] != bundle_dir.name or bundle["bundle_id"] != manifest[
        "bundle_id"
    ]:
        raise _reject("BUNDLE_ID_MISMATCH")
    normalized = _exact(
        parse_canonical_json(
            member_bytes["normalized/odds.json"], max_bytes=_MAX_JSON_BYTES
        ),
        {
            "schema_version",
            "bundle_id",
            "race_identity_sha256",
            "capture_timestamp",
            "source_timestamp",
            "runner_set_sha256",
            "odds_sha256",
            "runners",
        },
        "normalized_odds",
    )
    if (
        normalized["schema_version"] != NORMALIZED_ODDS_SCHEMA_VERSION
        or normalized["bundle_id"] != bundle["bundle_id"]
        or normalized["race_identity_sha256"] != bundle["race_identity_sha256"]
        or normalized["capture_timestamp"] != bundle["timing"]["capture_timestamp"]
        or normalized["source_timestamp"] != bundle["timing"]["source_timestamp"]
        or normalized["runner_set_sha256"] != expected.runner_set_sha256
        or normalized["odds_sha256"] != expected.odds_sha256
        or [
            {key: value for key, value in row.items() if key != "decimal_odds"}
            for row in normalized["runners"]
        ]
        != bundle["runner_set"]
        or sha256_bytes(member_bytes["normalized/odds.json"])
        != bundle["normalized_odds"]["sha256"]
        or bundle["normalized_odds"]["odds_sha256"] != expected.odds_sha256
    ):
        raise _reject("ODDS_PROVENANCE_MISMATCH")
    terminal = parse_canonical_json(
        member_bytes["producer/terminal.json"], max_bytes=_MAX_JSON_BYTES
    )
    terminal_source = terminal["provenance"]["source_files"]
    if (
        sha256_bytes(member_bytes["producer/terminal.json"])
        != expected.terminal_sha256
        or terminal["run_id"] != expected.run_id
        or terminal["request"]["request_id"] != expected.request_id
        or terminal["provenance"]["request_sha256"] != expected.request_sha256
        or terminal["provenance"]["config_sha256"] != expected.config_sha256
        or terminal["provenance"]["race_identity_sha256"]
        != expected.race_identity_sha256
        or terminal["provenance"]["runner_set_sha256"]
        != expected.runner_set_sha256
        or terminal["provenance"]["odds_sha256"] != expected.odds_sha256
        or terminal["provenance"]["model_sha256"] != expected.model_sha256
        or terminal["request"]["selected_race"] != bundle["race"]
        or terminal["capture"]["runner_set"] != normalized["runners"]
        or len(terminal_source) != 1
        or terminal_source[0]["sha256"] != expected.source_sha256
        or terminal_source[0]["source_timestamp"] != expected.source_timestamp
        or terminal_source[0]["race_url"] != expected.final_url
        or terminal_source[0]["content_class"] != bundle["source"]["content_class"]
        or terminal["timing"]["readiness_checked_at"]
        != bundle["timing"]["readiness_checked_at"]
        or terminal["timing"]["submitted_at"]
        != bundle["timing"]["capture_started_at"]
        or terminal["timing"]["terminal_at"]
        != bundle["timing"]["capture_completed_at"]
        or terminal["timing"]["capture_timestamp"]
        != bundle["timing"]["capture_timestamp"]
        or terminal["timing"]["capture_prejump_margin_seconds"]
        != bundle["timing"]["final_prejump_margin_seconds"]
        or terminal["request"]["minimum_prejump_margin_seconds"]
        != bundle["timing"]["minimum_prejump_margin_seconds"]
        or sha256_bytes(member_bytes["producer/terminal.json"])
        != bundle["producer"]["terminal_sha256"]
    ):
        raise _reject("PRODUCER_PROVENANCE_MISMATCH")
    source_raw = member_bytes["source/raw.bin"]
    if (
        len(source_raw) != bundle["source"]["bytes"]
        or sha256_bytes(source_raw) != bundle["source"]["sha256"]
    ):
        raise _reject("SOURCE_PROVENANCE_MISMATCH")
    metadata = {
        "final_url": bundle["source"]["final_url"],
        "status_code": bundle["source"]["status_code"],
        "content_type": bundle["source"]["content_type"],
    }
    if canonical_sha256(metadata) != bundle["source"]["metadata_sha256"]:
        raise _reject("SOURCE_PROVENANCE_MISMATCH")
    _reject_outcome_material(source_raw, bundle["source"]["content_type"])
    odds_parser, parsed_odds_sha256 = _validate_parsed_odds(
        source_raw,
        bundle["source"]["content_type"],
        normalized["runners"],
    )
    if (
        odds_parser != bundle["source"]["odds_parser"]
        or parsed_odds_sha256 != bundle["source"]["parsed_odds_sha256"]
    ):
        raise _reject("ODDS_PROVENANCE_MISMATCH")
    bundle_seed = {
        "run_id": bundle["producer"]["run_id"],
        "request_sha256": bundle["producer"]["request_sha256"],
        "race_identity_sha256": bundle["race_identity_sha256"],
        "runner_set_sha256": bundle["runner_set_sha256"],
        "odds_sha256": bundle["normalized_odds"]["odds_sha256"],
        "source_sha256": bundle["source"]["sha256"],
        "terminal_sha256": bundle["producer"]["terminal_sha256"],
        "response_metadata_sha256": bundle["source"]["metadata_sha256"],
        "parsed_odds_sha256": bundle["source"]["parsed_odds_sha256"],
        "cleanup": {
            key: value
            for key, value in bundle["cleanup"].items()
            if key != "cancellation_status"
        },
        "implementation": bundle["implementation"],
    }
    if canonical_sha256(bundle_seed) != bundle["bundle_id"]:
        raise _reject("BUNDLE_ID_MISMATCH")
    return SealedManualEvidence(
        bundle_dir=bundle_dir,
        bundle=bundle,
        manifest=manifest,
        manifest_sha256=sha256_bytes(manifest_raw),
        replayed=False,
    )


def seal_manual_capture(
    execution: ManualCaptureExecution,
    *,
    config: Mapping[str, Any],
    forbidden_paths: Mapping[str, str],
    expected: SealExpectations,
    identity: SealingIdentity,
    repo_root: Path,
    cancellation_token: CancellationToken | None = None,
    stage_hook: Callable[[str, Path], None] | None = None,
) -> SealedManualEvidence:
    """Validate, stage, fsync and atomically publish one immutable bundle."""

    _check_cancelled(cancellation_token)
    current_identity = build_sealing_identity(
        repo_root=repo_root,
        source_commit=expected.source_commit,
        source_tree=expected.source_tree,
    )
    if identity != current_identity:
        raise _reject("IMPLEMENTATION_IDENTITY_MISMATCH")
    artifact, terminal_raw, members, response = _validate_execution(
        execution,
        config=config,
        forbidden_paths=forbidden_paths,
        expected=expected,
    )
    source_row = artifact["provenance"]["source_files"][0]
    bundle_id, bundle, manifest, publication_members = _build_documents(
        artifact=artifact,
        terminal_raw=terminal_raw,
        source_raw=members[source_row["path"]],
        response=response,
        identity=identity,
        execution=execution,
    )
    run_dir = execution.run_dir
    sealed_root = run_dir / SEALED_ROOT_NAME
    try:
        os.mkdir(sealed_root, 0o700)
    except FileExistsError:
        pass
    if not _is_plain_directory(sealed_root, parent=run_dir):
        raise _reject("UNSAFE_PATH", path=str(sealed_root))
    lock_descriptor = _open_seal_lock(run_dir)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        _clean_stale_stages(sealed_root)
        destination = sealed_root / bundle_id
        if destination.exists() or destination.is_symlink():
            verified = verify_manual_evidence_bundle(
                destination,
                run_dir=run_dir,
                expected=expected,
                expected_identity=identity,
            )
            if (
                verified.bundle != bundle
                or verified.manifest != manifest
                or any(
                    _read_relative_regular(
                        destination,
                        path,
                        max_bytes=_MAX_SOURCE_BYTES
                        if path == "source/raw.bin"
                        else _MAX_JSON_BYTES,
                    )
                    != raw
                    for path, raw in publication_members.items()
                )
            ):
                raise _reject("REPLAY_MISMATCH")
            _fsync_directory(sealed_root)
            return SealedManualEvidence(
                bundle_dir=destination,
                bundle=verified.bundle,
                manifest=verified.manifest,
                manifest_sha256=verified.manifest_sha256,
                replayed=True,
            )
        _check_cancelled(cancellation_token)
        stage = sealed_root / f"{_STAGE_PREFIX}{bundle_id}-{uuid.uuid4()}"
        os.mkdir(stage, 0o700)
        if stage_hook is not None:
            stage_hook("stage_created", stage)
        _check_cancelled(cancellation_token)
        for path in _FIXED_MEMBER_PATHS:
            _write_once(stage, path, publication_members[path])
            if stage_hook is not None:
                stage_hook(f"member_written:{path}", stage)
            _check_cancelled(cancellation_token)
        _write_once(stage, MANIFEST_FILENAME, canonical_bytes(manifest))
        if stage_hook is not None:
            stage_hook("manifest_written", stage)
        _check_cancelled(cancellation_token)
        if stage_hook is not None:
            stage_hook("members_written", stage)
        _check_cancelled(cancellation_token)
        _freeze_and_fsync_tree(stage)
        if stage_hook is not None:
            stage_hook("stage_fsynced", stage)
        _check_cancelled(cancellation_token)
        os.rename(stage, destination)
        if stage_hook is not None:
            stage_hook("renamed", destination)
        _fsync_directory(sealed_root)
        if stage_hook is not None:
            stage_hook("parent_fsynced", destination)
        verified = verify_manual_evidence_bundle(
            destination,
            run_dir=run_dir,
            expected=expected,
            expected_identity=identity,
        )
        return verified
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


__all__ = [
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "NORMALIZED_ODDS_SCHEMA_VERSION",
    "PUBLICATION_PROTOCOL",
    "SEALED_ROOT_NAME",
    "ManualEvidenceRejected",
    "SealExpectations",
    "SealedManualEvidence",
    "SealingIdentity",
    "build_sealing_identity",
    "expectations_from_execution",
    "seal_manual_capture",
    "verify_manual_evidence_bundle",
]
