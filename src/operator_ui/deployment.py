"""Generate the finite, default-off Operator UI R3 deployment package."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class DeploymentRejected(RuntimeError):
    """The requested package cannot safely bind the repository deployment."""


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SYSTEMD_SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./+-]+$")
_SECRET_VALUE = re.compile(r"^(?=[!-~]+$)[^\"'\\]+$")
_REQUIRED_SECRETS = {
    "OPERATOR_UI_SECRET_KEY",
    "OPERATOR_UI_USERNAME",
    "OPERATOR_UI_PASSWORD_HASH",
}
_ARTIFACTS = {
    "prediction_script": "scripts/predict_race_now.py",
    "prediction_config": "configs/prediction/manual-default.json",
    "model_artifact": "artifacts/frozen_models/market_form_residual_v1/model.json",
    "model_manifest": "artifacts/frozen_models/market_form_residual_v1/manifest.json",
    "model_schema": "configs/prediction/schemas/market_form_residual_v1.schema.json",
}
_LIVE_JSON_KEYS = {
    "full_state", "full_report", "odds_state", "odds_report", "odds_refresh",
    "corpus_report", "corpus_manifest", "deployment_manifest", "model_catalog",
}
_LIVE_RAW_KEYS = {
    "corpus_inventory_csv", "corpus_inventory_jsonl", "corpus_scorecard_csv",
    "corpus_scorecard_jsonl", "corpus_report_bytes", "corpus_summary",
    "corpus_final_status", "model_latest_config", "model_latest_schema",
    "model_latest_artifact", "model_latest_manifest", "model_baseline_config",
    "model_baseline_schema",
}
_UNIT_KEYS = {"full_timer", "full_service", "odds_timer", "odds_service"}
_DIGEST_ONLY_RAW_KEYS = {"corpus_inventory_csv", "corpus_inventory_jsonl"}
_DIGEST_ONLY_MAX_BYTES = 64 * 1024 * 1024
_DIGEST_ONLY_DEADLINE_SECONDS = 30.0
_READ_CHUNK_BYTES = 64 * 1024
_UNIT_BASENAMES = {
    "full_timer": "shadow-autopilot.timer",
    "full_service": "shadow-autopilot.service",
    "odds_timer": "shadow-autopilot-odds-capture.timer",
    "odds_service": "shadow-autopilot-odds-capture.service",
}


def _safe_existing(path: Path, *, directory: bool, executable: bool = False) -> Path:
    path = Path(path).absolute()
    try:
        current = Path(path.anchor)
        for component in path.parts[1:]:
            current /= component
            if current.is_symlink():
                raise DeploymentRejected(f"symlinked input rejected: {path}")
        info = path.stat()
    except (OSError, ValueError) as error:
        raise DeploymentRejected(f"required input unavailable: {path}") from error
    if stat.S_ISDIR(info.st_mode) != directory or (not directory and not stat.S_ISREG(info.st_mode)):
        raise DeploymentRejected(f"input has wrong type: {path}")
    if info.st_mode & 0o002 or (info.st_mode & 0o020 and info.st_uid != os.geteuid()):
        raise DeploymentRejected(f"permission-unsafe input rejected: {path}")
    if executable and not info.st_mode & 0o100:
        raise DeploymentRejected(f"pinned Python is not owner-executable: {path}")
    return path


def _systemd_safe(path: Path) -> None:
    if not _SYSTEMD_SAFE_PATH.fullmatch(str(path)):
        raise DeploymentRejected(f"authority path is not systemd-safe: {path}")


def _verify_source_identity(source: Path, commit: str, tree: str) -> None:
    try:
        status = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        identity = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(source), "rev-parse", "HEAD", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DeploymentRejected("source Git identity unavailable") from error
    observed = identity.stdout.splitlines()
    if status.stdout or observed != [commit, tree]:
        raise DeploymentRejected("source Git identity is dirty or mismatched")


def _separate(roots: tuple[Path, ...]) -> None:
    for offset, left in enumerate(roots):
        for right in roots[offset + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise DeploymentRejected("deployment roots must not overlap")


def _read(path: Path, maximum: int = 256 * 1024) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise DeploymentRejected(f"required input unreadable: {path}") from error
    if len(data) > maximum:
        raise DeploymentRejected(f"bounded deployment input oversized: {path}")
    return data


def _retained_file_read(path: Path, maximum: int = 256 * 1024) -> bytes:
    """Bounded no-follow read retaining and rechecking every absolute component."""
    path = path.absolute(); opened: list[tuple[int, Path, tuple[int, int, int, int, int]]] = []
    descriptor = os.open("/", os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        root_info = os.fstat(descriptor)
        opened.append((descriptor, Path("/"), (root_info.st_dev, root_info.st_ino, root_info.st_mode, root_info.st_size, root_info.st_mtime_ns)))
        current = Path("/")
        for offset, component in enumerate(path.parts[1:]):
            current /= component; directory = offset < len(path.parts[1:]) - 1
            descriptor = os.open(component, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else 0), dir_fd=descriptor)
            info = os.fstat(descriptor)
            opened.append((descriptor, current, (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)))
            if directory and not stat.S_ISDIR(info.st_mode) or not directory and not stat.S_ISREG(info.st_mode):
                raise DeploymentRejected(f"authority input has wrong type: {path}")
        chunks=[];remaining=maximum+1
        while remaining:
            chunk=os.read(descriptor,min(65536,remaining))
            if not chunk:break
            chunks.append(chunk);remaining-=len(chunk)
        data=b"".join(chunks)
        if len(data)>maximum:raise DeploymentRejected(f"bounded deployment input oversized: {path}")
        for item,item_path,identity in opened:
            observed=os.fstat(item);named=os.stat(item_path,follow_symlinks=False)
            if (observed.st_dev,observed.st_ino,observed.st_mode,observed.st_size,observed.st_mtime_ns)!=identity or (named.st_dev,named.st_ino,named.st_mode,named.st_size,named.st_mtime_ns)!=identity:
                raise DeploymentRejected(f"authority identity changed during retained read: {path}")
        return data
    except DeploymentRejected:raise
    except OSError as error:raise DeploymentRejected(f"required input unreadable: {path}") from error
    finally:
        for item,_,_ in reversed(opened):
            try:os.close(item)
            except OSError:pass


def _retained_file_digest(path: Path) -> tuple[str, int]:
    """Stream one large fixed file to a digest without retaining its bytes."""
    path = path.absolute(); opened: list[tuple[int, Path, tuple[int, ...]]] = []
    descriptor = os.open("/", os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    started = time.monotonic()
    try:
        current = Path("/")
        parts = path.parts[1:]
        for offset, component in enumerate(parts):
            info = os.fstat(descriptor)
            opened.append((descriptor, current, (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)))
            current /= component; directory = offset < len(parts) - 1
            descriptor = os.open(component, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else 0), dir_fd=descriptor)
            info = os.fstat(descriptor)
            if directory and not stat.S_ISDIR(info.st_mode) or not directory and not stat.S_ISREG(info.st_mode):
                raise DeploymentRejected(f"authority input has wrong type: {path}")
        info = os.fstat(descriptor)
        identity = (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        opened.append((descriptor, current, identity))
        if info.st_size > _DIGEST_ONLY_MAX_BYTES:
            raise DeploymentRejected(f"bounded deployment input oversized: {path}")
        digest = hashlib.sha256(); total = 0
        while True:
            if time.monotonic() - started > _DIGEST_ONLY_DEADLINE_SECONDS:
                raise DeploymentRejected(f"bounded deployment input timed out: {path}")
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk: break
            total += len(chunk)
            if total > _DIGEST_ONLY_MAX_BYTES:
                raise DeploymentRejected(f"bounded deployment input oversized: {path}")
            digest.update(chunk)
        if time.monotonic() - started > _DIGEST_ONLY_DEADLINE_SECONDS:
            raise DeploymentRejected(f"bounded deployment input timed out: {path}")
        for item, item_path, expected in opened:
            observed = os.fstat(item); named = os.stat(item_path, follow_symlinks=False)
            actual = (observed.st_dev, observed.st_ino, observed.st_mode, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns)
            named_identity = (named.st_dev, named.st_ino, named.st_mode, named.st_size, named.st_mtime_ns, named.st_ctime_ns)
            if actual != expected or named_identity != expected:
                raise DeploymentRejected(f"authority identity changed during retained read: {path}")
        if total != identity[3]:
            raise DeploymentRejected(f"authority identity changed during retained read: {path}")
        return digest.hexdigest(), total
    except DeploymentRejected: raise
    except OSError as error: raise DeploymentRejected(f"required input unreadable: {path}") from error
    finally:
        for item, _, _ in reversed(opened):
            try: os.close(item)
            except OSError: pass


def _strict_json(raw: bytes) -> Any:
    def exact(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    return json.loads(
        raw.decode("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        object_pairs_hook=exact,
    )


def _live_authority(path: Path) -> dict[str, Any]:
    """Validate one deployer-owned, finite observation without discovering anything."""
    try:
        value = _strict_json(_retained_file_read(path))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DeploymentRejected("live authority observation is malformed") from error
    if not isinstance(value, dict) or set(value) != {"schema_version", "observed_at", "working_directory", "sources", "raw_sources", "units", "service_status"} or value["schema_version"] != "operator_ui_live_authority_v1":
        raise DeploymentRejected("live authority observation is incomplete")
    if set(value.get("sources", {})) != _LIVE_JSON_KEYS or set(value.get("raw_sources", {})) != _LIVE_RAW_KEYS or set(value.get("units", {})) != _UNIT_KEYS or set(value.get("service_status", {})) != {"full", "odds"}:
        raise DeploymentRejected("live authority observation is incomplete")
    try:
        observed = __import__("datetime").datetime.fromisoformat(value["observed_at"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise DeploymentRejected("live authority observation time is invalid") from error
    if observed.tzinfo is None or not isinstance(value.get("working_directory"), str) or not value["working_directory"].startswith("/"):
        raise DeploymentRejected("live authority observation is invalid")
    sealed: dict[str, Any] = {"schema_version": value["schema_version"], "observed_at": value["observed_at"], "working_directory": value["working_directory"]}
    snapshots: dict[tuple[str, str], bytes] = {}
    unit_paths: set[Path] = set()
    for group in ("sources", "raw_sources", "units"):
        sealed[group] = {}
        for name, locator in value[group].items():
            if not isinstance(locator, str) or not Path(locator).is_absolute():
                raise DeploymentRejected("live authority locator is invalid")
            file_path = _safe_existing(Path(locator), directory=False)
            if group == "units":
                if file_path in unit_paths:
                    raise DeploymentRejected("live authority unit paths must be distinct")
                unit_paths.add(file_path)
                if file_path != file_path.resolve() or file_path.name != _UNIT_BASENAMES[name]:
                    raise DeploymentRejected("live authority unit path is invalid")
            if group == "raw_sources" and name in _DIGEST_ONLY_RAW_KEYS:
                digest, byte_count = _retained_file_digest(file_path)
                sealed[group][name] = {"path": str(file_path), "sha256": digest, "bytes": byte_count, "authentication": "sha256_size_only_v1"}
            else:
                raw = _retained_file_read(file_path, 16 * 1024 * 1024)
                snapshots[(group, name)] = raw
                sealed[group][name] = {"path": str(file_path), "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        odds_report = _strict_json(snapshots[("sources", "odds_report")])
        refresh_path = Path(value["sources"]["odds_refresh"])
        output_dir = odds_report["autopilot_output_dir"]
        if refresh_path.name != "odds_capture_refresh_report.json":
            raise ValueError
        if output_dir is None:
            if not isinstance(odds_report.get("generated_at"), str):
                raise ValueError
            generated_at = __import__("datetime").datetime.fromisoformat(
                odds_report["generated_at"].replace("Z", "+00:00")
            )
            if (
                odds_report.get("schema_version")
                != "shadow_autopilot_odds_capture_only_daemon_report_v1"
                or generated_at.tzinfo is None
                or odds_report.get("final_status")
                != "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW"
                or odds_report.get("status") != "WAITING"
                or odds_report.get("odds_capture_refresh_report") != {}
            ):
                raise ValueError
            refresh_root = refresh_path.parent
        else:
            relative = Path(output_dir) / "odds_capture_refresh_report.json"
            if ".." in relative.parts or len(relative.parts) < 2:
                raise ValueError
            if relative.is_absolute():
                refresh_root = Path(output_dir)
                if refresh_path != relative:
                    raise ValueError
            else:
                refresh_root = refresh_path.parents[len(relative.parts) - 1]
                if refresh_path.relative_to(refresh_root) != relative:
                    raise ValueError
        _safe_existing(refresh_root, directory=True)
    except (KeyError, TypeError, ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentRejected("odds refresh authority is contradictory") from error
    sealed["sources"]["odds_refresh"]["allowlisted_root"] = str(refresh_root)
    sealed["service_status"] = value["service_status"]
    for lane in ("full", "odds"):
        status = sealed["service_status"][lane]
        expected_unit = {"full": "shadow-autopilot.service", "odds": "shadow-autopilot-odds-capture.service"}[lane]
        if not isinstance(status, dict) or set(status) != {"unit_name", "active_state", "sub_state", "exec_main_pid"} or status["unit_name"] != expected_unit or not isinstance(status["active_state"], str) or not isinstance(status["sub_state"], str) or type(status["exec_main_pid"]) is not int or status["exec_main_pid"] < 0:
            raise DeploymentRejected("live authority service status is invalid")
    return sealed


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _authority_identity(info: os.stat_result, *, directory: bool) -> tuple[int, ...]:
    if directory:
        return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))
    return _file_identity(info)


@contextmanager
def _retained_authority_reads(
    source: Path, relatives: tuple[str, ...], maximum: int = 256 * 1024
) -> Iterator[dict[str, bytes]]:
    """Read fixed source files through retained, no-follow component descriptors."""
    retained: list[tuple[int, Path, tuple[int, ...], bool]] = []
    leaves: dict[str, int] = {}
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        for relative in relatives:
            target = source / relative
            parts = target.absolute().parts
            descriptor = os.open(parts[0], flags | os.O_DIRECTORY)
            try:
                info = os.fstat(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
            retained.append((descriptor, Path(parts[0]), _authority_identity(info, directory=True), True))
            current = Path(parts[0])
            for index, component in enumerate(parts[1:]):
                current /= component
                directory = index < len(parts[1:]) - 1
                descriptor = os.open(
                    component,
                    flags | (os.O_DIRECTORY if directory else 0),
                    dir_fd=descriptor,
                )
                try:
                    info = os.fstat(descriptor)
                except BaseException:
                    os.close(descriptor)
                    raise
                if directory and not stat.S_ISDIR(info.st_mode):
                    raise DeploymentRejected(f"authority component has wrong type: {current}")
                if not directory and not stat.S_ISREG(info.st_mode):
                    raise DeploymentRejected(f"authority input has wrong type: {current}")
                retained.append((descriptor, current, _authority_identity(info, directory=directory), directory))
            leaves[relative] = descriptor

        def verify_unchanged() -> None:
            for descriptor, path, identity, directory in retained:
                descriptor_info = os.fstat(descriptor)
                path_info = os.stat(path, follow_symlinks=False)
                expected_type = stat.S_ISDIR if directory else stat.S_ISREG
                if (
                    _authority_identity(descriptor_info, directory=directory) != identity
                    or _authority_identity(path_info, directory=directory) != identity
                    or not expected_type(path_info.st_mode)
                ):
                    raise DeploymentRejected(f"authority identity changed during retained read: {path}")

        verify_unchanged()
        contents: dict[str, bytes] = {}
        for relative, descriptor in leaves.items():
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > maximum:
                raise DeploymentRejected(f"bounded deployment input oversized: {source / relative}")
            contents[relative] = data
        verify_unchanged()
        yield contents
        verify_unchanged()
    except DeploymentRejected:
        raise
    except OSError as error:
        raise DeploymentRejected("authority input changed or became unavailable") from error
    finally:
        for descriptor, _, _, _ in reversed(retained):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _validate_secrets_file(path: Path) -> None:
    try:
        text = _read(path, 64 * 1024).decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise DeploymentRejected("secrets file is not UTF-8") from error
    present: set[str] = set()
    for line in text.split("\n"):
        if not line or (line.startswith("#") and all(" " <= char <= "~" for char in line)):
            continue
        if line.count("=") < 1:
            raise DeploymentRejected("secrets file contains unsupported syntax")
        name, value = line.split("=", 1)
        if name not in _REQUIRED_SECRETS or name in present:
            raise DeploymentRejected("secrets file contains an extra or duplicate assignment")
        if not value or not _SECRET_VALUE.fullmatch(value):
            raise DeploymentRejected("secrets file contains ambiguous quoting, escaping, or control syntax")
        present.add(name)
    if present != _REQUIRED_SECRETS:
        raise DeploymentRejected("secrets file is incomplete")


def _stage_file(path: Path, content: bytes, mode: int, suffix: str) -> Path:
    temporary = path.with_name(f".{path.name}.{suffix}-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return temporary


def _publish_transaction(outputs: tuple[tuple[Path, bytes, int], ...]) -> None:
    originals: dict[Path, tuple[bytes, int] | None] = {}
    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for index, (target, content, mode) in enumerate(outputs):
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists():
                originals[target] = (target.read_bytes(), stat.S_IMODE(target.stat().st_mode))
            else:
                originals[target] = None
            staged[target] = _stage_file(target, content, mode, f"tmp-{index}")
        for target, _, _ in outputs:
            os.replace(staged[target], target)
            published.append(target)
    except BaseException:
        for index, target in reversed(list(enumerate(published))):
            original = originals[target]
            if original is None:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            else:
                content, mode = original
                recovery = _stage_file(target, content, mode, f"restore-{index}")
                try:
                    os.replace(recovery, target)
                finally:
                    try:
                        recovery.unlink()
                    except FileNotFoundError:
                        pass
        raise
    finally:
        for temporary in staged.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _validate_target(path: Path, boundary: Path) -> None:
    """Reject pre-existing symlinks/types without following outside a fixed root."""
    try:
        relative = path.absolute().relative_to(boundary.absolute())
    except ValueError as error:
        raise DeploymentRejected("generated target escaped its fixed root") from error
    current = boundary.absolute()
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise DeploymentRejected(f"symlinked generated target rejected: {path}")
        if current.exists() and current != path and not current.is_dir():
            raise DeploymentRejected(f"generated target parent is not a directory: {path}")
    if path.exists() and not path.is_file():
        raise DeploymentRejected(f"unsafe generated target: {path}")


def generate_package(*, source_root: Path, pinned_python: Path, evidence_root: Path,
                     producer_root: Path, canonical_db: Path, operations_root: Path,
                     secrets_file: Path, output_dir: Path, source_commit: str,
                     source_tree: str, ui_version: str, profile_id: str,
                     bind_address: str = "127.0.0.1", port: int = 5055,
                     live_authority: Path | None = None, enabled: bool = False) -> dict[str, Any]:
    """Validate every authority input, then write one finite generated package."""
    if not _COMMIT.fullmatch(source_commit) or not _COMMIT.fullmatch(source_tree):
        raise DeploymentRejected("source commit/tree identity is invalid")
    if profile_id != "repository-v1" or ui_version != "operator-ui-v1":
        raise DeploymentRejected("deployment identity is not the finite repository-v1 profile")
    if not _VERSION.fullmatch(ui_version) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise DeploymentRejected("deployment version or port is invalid")
    try:
        address = ipaddress.ip_address(bind_address)
    except ValueError as error:
        raise DeploymentRejected("bind address is invalid") from error
    if address.is_unspecified or address.is_multicast or not (address.is_loopback or address.is_private):
        raise DeploymentRejected("bind address must be loopback or private")

    for authority_path in (
        source_root, pinned_python, evidence_root, producer_root, canonical_db,
        operations_root, secrets_file, output_dir,
    ):
        _systemd_safe(Path(authority_path).absolute())

    source = _safe_existing(source_root, directory=True)
    _verify_source_identity(source, source_commit, source_tree)
    python = _safe_existing(pinned_python, directory=False, executable=True)
    evidence = _safe_existing(evidence_root, directory=True)
    producer = _safe_existing(producer_root, directory=True)
    operations = _safe_existing(operations_root, directory=True)
    output = _safe_existing(output_dir, directory=True)
    database = _safe_existing(canonical_db, directory=False)
    secrets = _safe_existing(secrets_file, directory=False)
    secrets_info = secrets.stat()
    if secrets_info.st_uid != os.geteuid():
        raise DeploymentRejected("secrets file must be owned by the current service user")
    if stat.S_IMODE(secrets_info.st_mode) != 0o600:
        raise DeploymentRejected("secrets file must have exact mode 0600")
    _separate((source, evidence, producer, operations, output))
    if any(database == root or root in database.parents for root in (source, evidence, producer, operations, output)):
        raise DeploymentRejected("canonical database must be separate from deployment roots")
    if secrets.samefile(database):
        raise DeploymentRejected("secrets file must not be the canonical database")
    if any(secrets == root or root in secrets.parents for root in (source, evidence, producer, operations, output)):
        raise DeploymentRejected("secrets file must be separate from deployment roots")

    profile_path = _safe_existing(source / "configs/operator_ui/repository-v1.toml", directory=False)
    app_path = _safe_existing(source / "app.py", directory=False)
    index = _safe_existing(evidence / "shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json", directory=False)
    protocol = _safe_existing(evidence / "manual_prediction_collector_requests_v1", directory=True)
    bundles = _safe_existing(producer / "artifacts/on_demand_prediction_runs", directory=True)
    artifact_paths = {name: _safe_existing(source / relative, directory=False) for name, relative in _ARTIFACTS.items()}
    _validate_secrets_file(secrets)

    authority_relatives = ("configs/operator_ui/repository-v1.toml", *_ARTIFACTS.values())
    with _retained_authority_reads(source, authority_relatives) as authority_bytes:
        _verify_source_identity(source, source_commit, source_tree)
        binding = {
            "schema_version": "operator_ui_repository_binding_v1",
            "profile_id": profile_id,
            "generator": {"generator_id": "GHU-036-repository-v1-generator", "schema_version": "operator_ui_repository_binding_generator_v1", "version": "1"},
            "deployment": {"source_commit": source_commit, "source_tree": source_tree, "ui_version": ui_version, "profile_id": profile_id},
            "profile_sha256": hashlib.sha256(authority_bytes["configs/operator_ui/repository-v1.toml"]).hexdigest(),
            "artifacts": {
                name: hashlib.sha256(authority_bytes[relative]).hexdigest()
                for name, relative in _ARTIFACTS.items()
            },
            "roots": {"source_root": str(source), "pinned_python": str(python), "evidence_root": str(evidence), "producer_root": str(producer), "canonical_db": str(database), "operations_root": str(operations)},
        }
    active = bool(enabled)
    live = _live_authority(_safe_existing(live_authority, directory=False)) if active and live_authority is not None else None
    if active and live is None:
        raise DeploymentRejected("enabled package requires live authority observation")
    if live is not None:
        binding["live_evidence"] = live
    environment = "\n".join((
        f"OPERATOR_UI_CONNECTED_MODE={int(active)}",
        f"OPERATOR_UI_LEVEL={2 if active else 1}",
        f"OPERATOR_UI_R3_PROFILE={'repository-v1' if active else 'disabled'}",
        f"OPERATOR_UI_DEPLOYED_COMMIT={source_commit}", f"OPERATOR_UI_DEPLOYED_TREE={source_tree}",
        f"OPERATOR_UI_DEPLOYED_VERSION={ui_version}", f"OPERATOR_UI_DEPLOYED_PROFILE={profile_id}",
        "ENABLE_SCRAPING_DEFAULT=0", "ENABLE_LIVE_SCRAPING=0", "ENABLE_RESULTS_SCRAPERS=0", "TGR_ENABLED=0", "PREDICTION_IMPORT_MODE=prediction_only", ""))
    service = "\n".join((
        "[Unit]", "Description=Greyhound Operator UI R3 (generated, private)", "After=network-online.target", "Wants=network-online.target", "",
        "[Service]", "Type=simple", f"WorkingDirectory={source}", f"EnvironmentFile={output / 'operator-ui-r3.env'}", f"EnvironmentFile={secrets}",
        f"ExecStart={python} -m src.operator_ui.deployment serve --source-root {source} --host {address} --port {port}",
        "Restart=no", "UMask=0077", "NoNewPrivileges=true", "PrivateTmp=true", "PrivateUsers=true", "ProtectSystem=strict",
        f"ReadOnlyPaths={source} {evidence} {producer} {database}", f"ReadWritePaths={operations}", "",
        "[Install]", "WantedBy=default.target", ""))
    rollback = f"""# Operator UI R3 rollback

Regenerate this package without `--enable`, stop/disable only
`greyhound-operator-ui-r3.service`, and verify the existing UI remains available.
Do not delete `{operations}`: it contains Operator UI audit/job evidence. Do not
delete `{producer}` or `{evidence}`: prediction and collector evidence is retained.
Rollback changes the feature gate only; it does not edit installed/generated files
by hand and does not alter the canonical database `{database}`.
"""
    binding_target = source / "var/operator_ui/generated/repository-v1.binding.json"
    environment_target = output / "operator-ui-r3.env"
    service_target = output / "greyhound-operator-ui-r3.service"
    rollback_target = output / "ROLLBACK.md"
    for target, boundary in ((binding_target, source), (environment_target, output), (service_target, output), (rollback_target, output)):
        _validate_target(target, boundary)
    binding_bytes = (json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _publish_transaction((
        (binding_target, binding_bytes, 0o600),
        (environment_target, environment.encode(), 0o600),
        (service_target, service.encode(), 0o644),
        (rollback_target, rollback.encode(), 0o644),
    ))
    return {"enabled": active, "binding": str(binding_target), "service": str(service_target)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    for name in ("source-root", "pinned-python", "evidence-root", "producer-root", "canonical-db", "operations-root", "secrets-file", "output-dir"):
        generate.add_argument(f"--{name}", required=True, type=Path)
    generate.add_argument("--source-commit", required=True); generate.add_argument("--source-tree", required=True)
    generate.add_argument("--ui-version", default="operator-ui-v1"); generate.add_argument("--profile-id", default="repository-v1")
    generate.add_argument("--bind-address", default="127.0.0.1"); generate.add_argument("--port", type=int, default=5055); generate.add_argument("--enable", action="store_true", dest="enabled")
    generate.add_argument("--live-authority", type=Path)
    manual = commands.add_parser("generate-manual")
    for name in (
        "source-root", "pinned-python", "manual-root", "browser-profile-root",
        "manual-runs-root", "manual-lock", "model", "model-manifest", "config", "output-dir",
    ):
        manual.add_argument(f"--{name}", required=True, type=Path)
    manual.add_argument("--source-commit", required=True)
    manual.add_argument("--source-tree", required=True)
    manual.add_argument("--timeout-seconds", type=int, default=900)
    manual.add_argument("--margin-seconds", type=int, default=120)
    manual.add_argument("--protected-path", action="append", required=True, metavar="NAME=PATH")
    serve = commands.add_parser("serve")
    serve.add_argument("--source-root", required=True, type=Path); serve.add_argument("--host", required=True); serve.add_argument("--port", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate-manual":
        from src.predictor.manual_research_deployment import generate_manual_package

        values = vars(args)
        raw_protected = values.pop("protected_path")
        protected: dict[str, Path] = {}
        for item in raw_protected:
            name, separator, value = item.partition("=")
            if not separator or name in protected:
                raise DeploymentRejected("invalid --protected-path")
            protected[name] = Path(value)
        result = generate_manual_package(
            **{key: value for key, value in values.items() if key != "command"},
            protected_paths=protected,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "serve":
        if os.environ.get("OPERATOR_UI_CONNECTED_MODE") != "1" or os.environ.get("OPERATOR_UI_R3_PROFILE") != "repository-v1":
            return 0
        source = _safe_existing(args.source_root, directory=True)
        deployed_commit = os.environ.get("OPERATOR_UI_DEPLOYED_COMMIT", "")
        deployed_tree = os.environ.get("OPERATOR_UI_DEPLOYED_TREE", "")
        if not _COMMIT.fullmatch(deployed_commit) or not _COMMIT.fullmatch(deployed_tree):
            raise DeploymentRejected("deployed source commit/tree identity is invalid")
        _verify_source_identity(source, deployed_commit, deployed_tree)
        os.execv(sys.executable, [sys.executable, str(source / "app.py"), "--host", args.host, "--port", str(args.port)])
    values = vars(args); values.pop("command")
    result = generate_package(**values)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
