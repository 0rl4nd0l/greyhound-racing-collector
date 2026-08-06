"""Generate the default-off deployment package for the manual research lane."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template


class ManualDeploymentRejected(RuntimeError):
    """The manual deployment cannot be bound without widening its authority."""


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_SYSTEMD_SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./+-]+$")
_PROTECTED_NAMES = frozenset(
    {
        "autonomous_browser_profile",
        "autonomous_shared_lock",
        "canonical_database",
        "canonical_history",
        "live_odds",
        "forward_corpus",
        "collector_requests",
        "collector_state",
        "result_evidence",
        "services",
        "timers",
    }
)
_TEMPLATE_RELATIVE = Path("ops/systemd/manual-research-api.service.in")


def _reject(message: str) -> ManualDeploymentRejected:
    return ManualDeploymentRejected(message)


def _absolute(value: Path | str, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or not _SYSTEMD_SAFE_PATH.fullmatch(str(path))
    ):
        raise _reject(f"{label} must be a normalized absolute path")
    return path


def _existing(path: Path, *, directory: bool, label: str, executable: bool = False) -> Path:
    path = _absolute(path, label=label)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                raise _reject(f"{label} contains a symlink")
        except OSError as exc:
            raise _reject(f"{label} is unavailable") from exc
    try:
        info = path.lstat()
    except OSError as exc:
        raise _reject(f"{label} is unavailable") from exc
    if (stat.S_ISDIR(info.st_mode) != directory) or (not directory and not stat.S_ISREG(info.st_mode)):
        raise _reject(f"{label} has the wrong type")
    if info.st_mode & 0o002 or (info.st_mode & 0o020 and info.st_uid != os.geteuid()):
        raise _reject(f"{label} has unsafe permissions")
    if executable and not info.st_mode & 0o100:
        raise _reject(f"{label} is not executable")
    return path


def _protected(path: Path, *, label: str) -> Path:
    path = _absolute(path, label=label)
    try:
        directory = stat.S_ISDIR(path.lstat().st_mode)
    except OSError as exc:
        raise _reject(f"{label} is unavailable") from exc
    return _existing(path, directory=directory, label=label)


def _separate(named: Mapping[str, Path]) -> None:
    items = tuple(named.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise _reject(f"deployment paths overlap: {left_name}/{right_name}")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise _reject(f"cannot read deployment artifact: {path}") from exc


def _source_identity(source: Path, commit: str, tree: str) -> None:
    if not _HEX40.fullmatch(commit) or not _HEX40.fullmatch(tree):
        raise _reject("source commit/tree identity is invalid")
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
    except (OSError, subprocess.SubprocessError) as exc:
        raise _reject("source Git identity unavailable") from exc
    if status.stdout or identity.stdout.splitlines() != [commit, tree]:
        raise _reject("source Git identity is dirty or mismatched")


def _retained_file_read(path: Path, maximum: int = 256 * 1024) -> bytes:
    """Read a fixed regular file while retaining no-follow component identity."""
    path = path.absolute()
    opened: list[tuple[int, Path, tuple[int, int, int, int, int]]] = []
    descriptor = os.open("/", os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        root = os.fstat(descriptor)
        opened.append(
            (descriptor, Path("/"), (root.st_dev, root.st_ino, root.st_mode, root.st_size, root.st_mtime_ns))
        )
        current = Path("/")
        for offset, component in enumerate(path.parts[1:]):
            current /= component
            directory = offset < len(path.parts[1:]) - 1
            descriptor = os.open(
                component,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
                | (os.O_DIRECTORY if directory else 0),
                dir_fd=descriptor,
            )
            info = os.fstat(descriptor)
            identity = (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)
            opened.append((descriptor, current, identity))
            if (directory and not stat.S_ISDIR(info.st_mode)) or (
                not directory and not stat.S_ISREG(info.st_mode)
            ):
                raise _reject(f"{path} has the wrong type")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > maximum:
            raise _reject(f"deployment artifact is oversized: {path}")
        for item, item_path, identity in opened:
            observed = os.fstat(item)
            named = os.stat(item_path, follow_symlinks=False)
            observed_identity = (
                observed.st_dev, observed.st_ino, observed.st_mode,
                observed.st_size, observed.st_mtime_ns,
            )
            named_identity = (
                named.st_dev, named.st_ino, named.st_mode,
                named.st_size, named.st_mtime_ns,
            )
            if observed_identity != identity or named_identity != identity:
                raise _reject(f"deployment identity changed during retained read: {path}")
        return data
    except ManualDeploymentRejected:
        raise
    except OSError as exc:
        raise _reject(f"cannot read deployment artifact: {path}") from exc
    finally:
        for item, _, _ in reversed(opened):
            try:
                os.close(item)
            except OSError:
                pass


def _publish(outputs: tuple[tuple[Path, bytes, int], ...]) -> None:
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, bytes, int] | None] = []
    try:
        for index, (target, content, mode) in enumerate(outputs):
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists() and (target.is_symlink() or not target.is_file()):
                raise _reject(f"generated target is unsafe: {target}")
            prior = (target.read_bytes(), stat.S_IMODE(target.stat().st_mode)) if target.exists() else None
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{index}")
            if temporary.exists():
                raise _reject(f"temporary generated target already exists: {temporary}")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), mode)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((target, temporary))
            published.append((target, prior[0], prior[1]) if prior is not None else None)
        for target, temporary in staged:
            os.replace(temporary, target)
    except BaseException:
        for (target, _), prior in zip(staged, published):
            try:
                if prior is None:
                    target.unlink()
                else:
                    target.write_bytes(prior[1])
                    target.chmod(prior[2])
            except FileNotFoundError:
                pass
        raise
    finally:
        for _, temporary in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def generate_manual_package(
    *,
    source_root: Path,
    pinned_python: Path,
    manual_root: Path,
    browser_profile_root: Path,
    manual_runs_root: Path,
    manual_lock: Path,
    model: Path,
    model_manifest: Path,
    config: Path,
    output_dir: Path,
    source_commit: str,
    source_tree: str,
    timeout_seconds: int = 900,
    margin_seconds: int = 120,
    protected_paths: Mapping[str, Path] | None = None,
) -> dict[str, str | bool]:
    """Validate and publish one disabled manual deployment package."""
    source = _existing(source_root, directory=True, label="source_root")
    python = _existing(pinned_python, directory=False, label="pinned_python", executable=True)
    manual = _existing(manual_root, directory=True, label="manual_root")
    profile = _existing(browser_profile_root, directory=True, label="browser_profile_root")
    runs = _existing(manual_runs_root, directory=True, label="manual_runs_root")
    lock = _absolute(manual_lock, label="manual_lock")
    model_path = _existing(model, directory=False, label="model")
    manifest_path = _existing(model_manifest, directory=False, label="model_manifest")
    config_path = _existing(config, directory=False, label="config")
    output = _existing(output_dir, directory=True, label="output_dir")
    _existing(source / "src/predictor/manual_research_cli.py", directory=False, label="manual adapter")
    _existing(source / "src/predictor/manual_research_worker.py", directory=False, label="manual worker")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= 900:
        raise _reject("timeout_seconds is invalid")
    if not isinstance(margin_seconds, int) or isinstance(margin_seconds, bool) or not 1 <= margin_seconds <= 7200:
        raise _reject("margin_seconds is invalid")
    if lock.parent != manual or manual not in profile.parents or manual not in runs.parents:
        raise _reject("manual profile, runs, and lock must be beneath manual_root")
    protected = dict(protected_paths or {})
    if set(protected) != _PROTECTED_NAMES:
        raise _reject("protected path inventory is incomplete")
    checked_protected = {
        name: _protected(path, label=f"protected_paths.{name}")
        for name, path in protected.items()
    }
    if os.path.lexists(lock):
        lock = _existing(lock, directory=False, label="manual_lock")
        for name, path in checked_protected.items():
            try:
                if lock.samefile(path):
                    raise _reject(f"manual lock aliases protected path: {name}")
            except FileNotFoundError:
                raise _reject("manual_lock is unavailable") from None
    _separate({
        "manual_root": manual,
        "output_dir": output,
        **checked_protected,
    })
    if output == source or source in output.parents or output in source.parents:
        raise _reject("output_dir must be separate from source_root")
    if any(path == source for path in checked_protected.values()):
        raise _reject("protected path cannot be the source_root")
    _source_identity(source, source_commit, source_tree)
    for path in (model_path, manifest_path, config_path):
        if source not in path.parents:
            raise _reject("model/config artifacts must be beneath source_root")
    template_path = _existing(source / _TEMPLATE_RELATIVE, directory=False, label="manual service template")
    template_bytes = _retained_file_read(template_path)
    model_bytes = _retained_file_read(model_path)
    manifest_bytes = _retained_file_read(manifest_path)
    config_bytes = _retained_file_read(config_path)
    _source_identity(source, source_commit, source_tree)
    hashes = {
        "model": hashlib.sha256(model_bytes).hexdigest(),
        "model_manifest": hashlib.sha256(manifest_bytes).hexdigest(),
        "config": hashlib.sha256(config_bytes).hexdigest(),
    }
    binding = {
        "schema_version": "manual_research_deployment_binding_v1",
        "deployment": {"source_commit": source_commit, "source_tree": source_tree, "lane": "manual-research-v1"},
        "executable": str(python),
        "entrypoint": "src.predictor.manual_research_cli:main",
        "manual": {
            "operations_root": str(manual),
            "browser_profile_root": str(profile),
            "runs_root": str(runs),
            "lock": str(lock),
            "timeout_seconds": timeout_seconds,
            "margin_seconds": margin_seconds,
        },
        "artifacts": hashes,
        "default_enabled": False,
        "research_only": True,
        "canonical": False,
        "phase7_excluded": True,
    }
    binding_path = output / "manual-research.binding.json"
    environment_path = output / "manual-research.env"
    service_path = output / "greyhound-manual-research.service"
    rollback_path = output / "ROLLBACK.md"
    environment = "\n".join(
        (
            "MANUAL_RESEARCH_ENABLED=0",
            "MANUAL_RESEARCH_PROFILE=manual-research-v1",
            f"MANUAL_RESEARCH_OPERATIONS_ROOT={manual}",
            f"MANUAL_RESEARCH_BROWSER_PROFILE={profile}",
            f"MANUAL_RESEARCH_RUNS_ROOT={runs}",
            f"MANUAL_RESEARCH_LOCK={lock}",
            f"MANUAL_RESEARCH_TIMEOUT_SECONDS={timeout_seconds}",
            f"MANUAL_RESEARCH_MARGIN_SECONDS={margin_seconds}",
            f"MANUAL_RESEARCH_SOURCE_COMMIT={source_commit}",
            f"MANUAL_RESEARCH_SOURCE_TREE={source_tree}",
            f"MANUAL_RESEARCH_MODEL_SHA256={hashes['model']}",
            f"MANUAL_RESEARCH_MODEL_MANIFEST_SHA256={hashes['model_manifest']}",
            f"MANUAL_RESEARCH_CONFIG_SHA256={hashes['config']}",
            "",
        )
    ).encode()
    service = Template(template_bytes.decode("utf-8")).substitute(
        SOURCE_ROOT=source,
        ENVIRONMENT_FILE=environment_path,
        PYTHON_EXECUTABLE=python,
        BINDING_PATH=output / "manual-research.binding.json",
        MANUAL_ROOT=manual,
        ENABLE_MARKER=manual / ".manual-research-enabled",
        PROTECTED_PATHS=" ".join(
            str(checked_protected[name]) for name in sorted(checked_protected)
        ),
        TIMEOUT_SECONDS=timeout_seconds,
    ).encode()
    rollback = (
        b"# Manual research deployment rollback\n\n"
        b"This package is default-off. Under separate deployment authority, remove only "
        b"the manual unit and generated package; do not touch autonomous units, timers, "
        b"locks, databases, evidence, or model artifacts.\n"
    )
    binding_bytes = (json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _publish(
        (
            (binding_path, binding_bytes, 0o600),
            (environment_path, environment, 0o600),
            (service_path, service, 0o644),
            (rollback_path, rollback, 0o644),
        )
    )
    return {"enabled": False, "binding": str(binding_path), "service": str(service_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manual-research-deployment")
    for name in (
        "source-root", "pinned-python", "manual-root", "browser-profile-root",
        "manual-runs-root", "manual-lock", "model", "model-manifest", "config", "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--margin-seconds", type=int, default=120)
    parser.add_argument("--protected-path", action="append", required=True, metavar="NAME=PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = vars(_parser().parse_args(argv))
    raw_protected = args.pop("protected_path")
    protected: dict[str, Path] = {}
    for item in raw_protected:
        name, separator, value = item.partition("=")
        if not separator or not _SAFE_NAME.fullmatch(name) or name in protected:
            raise SystemExit("invalid --protected-path")
        protected[name] = Path(value)
    result = generate_manual_package(**args, protected_paths=protected)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ManualDeploymentRejected", "generate_manual_package", "main"]
