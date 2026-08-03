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


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


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
            retained.append((descriptor, Path(parts[0]), _file_identity(info), True))
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
                retained.append((descriptor, current, _file_identity(info), directory))
            leaves[relative] = descriptor

        def verify_unchanged() -> None:
            for descriptor, path, identity, directory in retained:
                descriptor_info = os.fstat(descriptor)
                path_info = os.stat(path, follow_symlinks=False)
                expected_type = stat.S_ISDIR if directory else stat.S_ISREG
                if (
                    _file_identity(descriptor_info) != identity
                    or _file_identity(path_info) != identity
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
                     enabled: bool = False) -> dict[str, Any]:
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
    environment = "\n".join((
        f"OPERATOR_UI_CONNECTED_MODE={int(active)}",
        f"OPERATOR_UI_R3_PROFILE={'repository-v1' if active else 'disabled'}",
        f"OPERATOR_UI_DEPLOYED_COMMIT={source_commit}", f"OPERATOR_UI_DEPLOYED_TREE={source_tree}",
        f"OPERATOR_UI_DEPLOYED_VERSION={ui_version}", f"OPERATOR_UI_DEPLOYED_PROFILE={profile_id}",
        "ENABLE_SCRAPING_DEFAULT=0", "ENABLE_LIVE_SCRAPING=0", "ENABLE_RESULTS_SCRAPERS=0", "TGR_ENABLED=0", "PREDICTION_IMPORT_MODE=prediction_only", ""))
    service = "\n".join((
        "[Unit]", "Description=Greyhound Operator UI R3 (generated, private)", "After=network-online.target", "Wants=network-online.target", "",
        "[Service]", "Type=simple", f"WorkingDirectory={source}", f"EnvironmentFile={output / 'operator-ui-r3.env'}", f"EnvironmentFile={secrets}",
        f"ExecStart={python} -m src.operator_ui.deployment serve --source-root {source} --host {address} --port {port}",
        "Restart=no", "UMask=0077", "NoNewPrivileges=true", "PrivateTmp=true", "ProtectSystem=strict",
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
    serve = commands.add_parser("serve")
    serve.add_argument("--source-root", required=True, type=Path); serve.add_argument("--host", required=True); serve.add_argument("--port", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        if os.environ.get("OPERATOR_UI_CONNECTED_MODE") != "1" or os.environ.get("OPERATOR_UI_R3_PROFILE") != "repository-v1":
            return 0
        source = _safe_existing(args.source_root, directory=True)
        os.execv(sys.executable, [sys.executable, str(source / "app.py"), "--host", args.host, "--port", str(args.port)])
    values = vars(args); values.pop("command")
    result = generate_package(**values)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
