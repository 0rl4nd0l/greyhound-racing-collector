#!/usr/bin/env python3
"""Run heavy read-only filesystem work in a fail-closed bounded cgroup.

The wrapper intentionally supports a small command vocabulary.  It mounts one
resolved root read-only into a transient Docker container and verifies the
container's cgroup-v2 hard read limits before executing the requested work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_IMAGE = "alpine:3.20"
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 3600
DEFAULT_READ_MIB_PER_SEC = 8
MAX_READ_MIB_PER_SEC = 16
DEFAULT_READ_IOPS = 64
MAX_READ_IOPS = 128
CONTAINER_ROOT = "/scan"
CONTAINER_RG = "/opt/bounded/rg"
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
class ConfigurationError(RuntimeError):
    """Raised when the wrapper cannot prove that an invocation is bounded."""


@dataclass(frozen=True)
class HostSupport:
    docker: Path
    image: str
    rg_binary: Path
    device: Path
    device_id: str


def _directory_exclusions(*names: str) -> tuple[str, ...]:
    patterns: list[str] = []
    for name in names:
        patterns.extend((f"!{name}", f"!{name}/**", f"!**/{name}", f"!**/{name}/**"))
    return tuple(patterns)


EXCLUSION_GROUPS: dict[str, tuple[str, ...]] = {
    "vcs": _directory_exclusions(".git", ".hg", ".svn"),
    "worktrees": _directory_exclusions(".worktrees", "worktrees"),
    "archives": _directory_exclusions(
        "archive",
        "archives",
        "archive_old_apps",
        "archive_unused_scripts",
        "cleanup_archive",
        "system_backup_*",
    ),
    "caches": _directory_exclusions(
        "__pycache__",
        ".cache",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".npm",
        ".yarn",
        "node_modules",
    ),
    "environments": _directory_exclusions(
        ".venv",
        ".venv311",
        "venv",
        "env",
        "ENV",
        ".virtualenv",
        "virtualenv",
    ),
    "generated": _directory_exclusions(
        "artifacts",
        "reports",
        "logs",
        "output",
        "outputs",
        "generated",
        "htmlcov",
        "build",
        "dist",
        "tmp",
        "temp",
        "test-results",
        "playwright-report",
    ),
    "large_data": _directory_exclusions(
        "data",
        "datasets",
        "dog_records",
        "exports",
        "model_registry",
        "predictions",
        "processed",
        "results",
        "samples",
        "snapshots",
        "unprocessed",
    )
    + (
        "!*.db*",
        "!*.sqlite*",
        "!*.csv",
        "!*.parquet",
        "!*.feather",
        "!*.joblib",
        "!*.pkl",
        "!*.pickle",
        "!*.npy",
        "!*.npz",
        "!*.onnx",
        "!*.h5",
        "!*.zip",
        "!*.tar",
        "!*.gz",
        "!*.bz2",
        "!*.xz",
        "!*.zst",
        "!*.7z",
    ),
}


HARD_LIMIT_BOOTSTRAP = r"""
expected_device=$1
expected_bps=$2
expected_iops=$3
shift 3

io_max=/sys/fs/cgroup/io.max
if [ ! -r "$io_max" ]; then
    echo "bounded-offline: hard limit unavailable: $io_max is not readable" >&2
    exit 78
fi
limit_line=$(awk -v device="$expected_device" '$1 == device { print; exit }' "$io_max")
case "$limit_line" in
    *"rbps=$expected_bps"*"riops=$expected_iops"*) ;;
    *)
        echo "bounded-offline: hard limit mismatch for $expected_device: ${limit_line:-missing}" >&2
        exit 78
        ;;
esac

io_weight=$(awk '$1 == "default" { print $2; exit }' /sys/fs/cgroup/io.weight)
case "$io_weight" in
    ''|*[!0-9]*)
        echo "bounded-offline: low I/O weight is not verifiable" >&2
        exit 78
        ;;
esac
if [ "$io_weight" -gt 10 ]; then
    echo "bounded-offline: I/O weight is not low: $io_weight" >&2
    exit 78
fi

cpu_weight=$(cat /sys/fs/cgroup/cpu.weight)
case "$cpu_weight" in
    ''|*[!0-9]*)
        echo "bounded-offline: low CPU weight is not verifiable" >&2
        exit 78
        ;;
esac
if [ "$cpu_weight" -gt 10 ]; then
    echo "bounded-offline: CPU weight is not low: $cpu_weight" >&2
    exit 78
fi

if ! ionice -c 3 -p $$ >/dev/null; then
    echo "bounded-offline: could not set idle I/O priority" >&2
    exit 78
fi
if ! ionice_state=$(ionice -p $$); then
    echo "bounded-offline: idle I/O priority is not verifiable" >&2
    exit 78
fi
case "$ionice_state" in
    idle*) ;;
    *)
        echo "bounded-offline: I/O priority is not idle: ${ionice_state:-missing}" >&2
        exit 78
        ;;
esac
renice 15 $$ >/dev/null
exec "$@"
""".strip()


HASH_SCRIPT = r"""
manifest=/tmp/bounded-files
"$@" > "$manifest"
sort -z "$manifest" -o "$manifest"
if [ -s "$manifest" ]; then
    xargs -0 -r sha256sum < "$manifest"
fi
""".strip()


def _bounded_int(label: str, minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"{label} must be between {minimum} and {maximum}"
            )
        return number

    return parse


def _positive_glob(value: str) -> str:
    if not value or value.startswith("!") or "\x00" in value:
        raise argparse.ArgumentTypeError("include globs must be non-empty positive globs")
    return value


def _non_empty_pattern(value: str) -> str:
    if not value:
        raise argparse.ArgumentTypeError("rg pattern must be non-empty")
    return value


def _validated_image_id(value: str) -> str:
    image_id = value.strip()
    digest = image_id.removeprefix("sha256:")
    if (
        not image_id.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ConfigurationError(
            f"local image {DEFAULT_IMAGE!r} returned an invalid immutable image ID"
        )
    return image_id


def _mount_source(path: Path, label: str) -> str:
    value = str(path)
    if "," in value:
        raise ConfigurationError(f"{label} must not contain a comma (Docker mount delimiter)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a heavy read-only command in a hard-limited transient cgroup.",
    )
    parser.add_argument("--root", required=True, help="one absolute existing directory")
    parser.add_argument(
        "--timeout-seconds",
        type=_bounded_int("timeout", 1, MAX_TIMEOUT_SECONDS),
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--read-mib-per-sec",
        type=_bounded_int("read MiB/s", 1, MAX_READ_MIB_PER_SEC),
        default=DEFAULT_READ_MIB_PER_SEC,
    )
    parser.add_argument(
        "--read-iops",
        type=_bounded_int("read IOPS", 1, MAX_READ_IOPS),
        default=DEFAULT_READ_IOPS,
    )
    parser.add_argument("--include-vcs", action="store_true")
    parser.add_argument("--include-worktrees", action="store_true")
    parser.add_argument("--include-archives", action="store_true")
    parser.add_argument("--include-caches", action="store_true")
    parser.add_argument("--include-environments", action="store_true")
    parser.add_argument("--include-generated", action="store_true")
    parser.add_argument("--include-large-data", action="store_true")
    parser.add_argument("--include-hidden", action="store_true")

    commands = parser.add_subparsers(dest="operation", required=True)

    search = commands.add_parser("rg", help="fixed-string search by default")
    search.add_argument("pattern", type=_non_empty_pattern)
    search.add_argument("--regex", action="store_true", help="interpret the pattern as regex")
    search.add_argument("--ignore-case", action="store_true")
    search.add_argument("--glob", action="append", default=[], type=_positive_glob)

    files = commands.add_parser("files", help="list matching files without reading contents")
    files.add_argument("--glob", action="append", default=[], type=_positive_glob)

    hashes = commands.add_parser("hash", help="SHA-256 files sequentially")
    hashes.add_argument("--glob", action="append", default=[], type=_positive_glob)

    commands.add_parser("tests", help="list test_*.py and *_test.py files without importing")
    return parser


def resolve_root(raw_root: str, *, repo_root: Path = SCRIPT_REPO_ROOT) -> Path:
    candidate = Path(raw_root).expanduser()
    if not candidate.is_absolute():
        raise ConfigurationError("--root must be an absolute path")
    if "," in str(candidate):
        raise ConfigurationError("--root must not contain a comma (Docker mount delimiter)")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise ConfigurationError(f"--root does not resolve to an existing directory: {exc}") from exc
    if not root.is_dir():
        raise ConfigurationError("--root must resolve to a directory")
    _mount_source(root, "resolved --root")

    resolved_repo = repo_root.resolve()
    if root != resolved_repo and resolved_repo.is_relative_to(root):
        raise ConfigurationError(
            "--root must not be an ancestor of the wrapper repository; name one exact worktree "
            "or a narrower directory"
        )

    prohibited = {Path("/"), Path("/home"), Path("/mnt"), Path("/var"), Path("/tmp")}
    try:
        prohibited.add(Path.home().resolve())
    except OSError:
        pass
    if root in prohibited:
        raise ConfigurationError(f"--root is too broad: {root}")
    return root


def exclusion_globs(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for group in (
        "vcs",
        "worktrees",
        "archives",
        "caches",
        "environments",
        "generated",
        "large_data",
    ):
        if not getattr(args, f"include_{group}"):
            result.extend(EXCLUSION_GROUPS[group])
    return result


def _include_hidden(args: argparse.Namespace) -> bool:
    return bool(
        args.include_hidden
        or args.include_vcs
        or args.include_caches
        or args.include_environments
    )


def _rg_file_command(args: argparse.Namespace, positive_globs: Sequence[str]) -> list[str]:
    command = [CONTAINER_RG, "--threads", "1", "--no-ignore"]
    if _include_hidden(args):
        command.append("--hidden")
    command.append("--files")
    for pattern in positive_globs:
        command.extend(("--glob", pattern))
    for pattern in exclusion_globs(args):
        command.extend(("--glob", pattern))
    command.append(CONTAINER_ROOT)
    return command


def workload_command(args: argparse.Namespace) -> list[str]:
    if args.operation == "rg":
        command = [CONTAINER_RG, "--threads", "1", "--no-ignore"]
        if _include_hidden(args):
            command.append("--hidden")
        if not args.regex:
            command.append("--fixed-strings")
        if args.ignore_case:
            command.append("--ignore-case")
        for pattern in args.glob:
            command.extend(("--glob", pattern))
        for pattern in exclusion_globs(args):
            command.extend(("--glob", pattern))
        command.extend(("--", args.pattern, CONTAINER_ROOT))
        return command

    if args.operation == "tests":
        return _rg_file_command(args, ("test_*.py", "*_test.py"))

    file_command = _rg_file_command(args, args.glob)
    if args.operation == "files":
        return file_command
    if args.operation == "hash":
        null_index = len(file_command) - 1
        file_command.insert(null_index, "--null")
        return ["/bin/sh", "-eu", "-c", HASH_SCRIPT, "bounded-hash", *file_command]
    raise ConfigurationError(f"unsupported operation: {args.operation}")


def _capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _require_success(result: subprocess.CompletedProcess[str], label: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ConfigurationError(f"{label} failed: {detail or f'exit {result.returncode}'}")
    return result.stdout.strip()


def detect_host_support(
    root: Path,
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _capture,
) -> HostSupport:
    docker_raw = shutil.which("docker")
    if not docker_raw:
        raise ConfigurationError("Docker is required for cgroup-v2 hard I/O enforcement")
    docker = Path(docker_raw).resolve()

    info = _require_success(
        runner([str(docker), "info", "--format", "{{.CgroupVersion}} {{.CgroupDriver}}"]),
        "Docker cgroup support check",
    ).split()
    if info != ["2", "systemd"]:
        raise ConfigurationError(
            "Docker must use cgroup v2 with the systemd driver; refusing unconstrained execution"
        )
    image = _validated_image_id(
        _require_success(
            runner(
                [str(docker), "image", "inspect", "--format", "{{.Id}}", DEFAULT_IMAGE]
            ),
            f"local image {DEFAULT_IMAGE!r} check (the wrapper never pulls)",
        )
    )

    rg_raw = shutil.which("rg")
    if not rg_raw:
        raise ConfigurationError("rg is required")
    rg_binary = Path(rg_raw).resolve(strict=True)
    if not rg_binary.is_file() or not os.access(rg_binary, os.X_OK):
        raise ConfigurationError(f"rg is not an executable file: {rg_binary}")
    linked = runner(["/usr/bin/ldd", str(rg_binary)])
    linked_text = f"{linked.stdout}\n{linked.stderr}".lower()
    if "statically linked" not in linked_text and "not a dynamic executable" not in linked_text:
        raise ConfigurationError(
            f"rg must be statically linked so only the bounded root is host-mounted: {rg_binary}"
        )

    mount_text = _require_success(
        runner(["/usr/bin/findmnt", "-J", "-T", str(root), "-o", "SOURCE,TARGET"]),
        "root backing-device discovery",
    )
    try:
        filesystems = json.loads(mount_text)["filesystems"]
        source = Path(filesystems[0]["source"])
        mount_target = Path(filesystems[0]["target"]).resolve()
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("root backing-device discovery returned invalid data") from exc
    if root == mount_target:
        raise ConfigurationError(
            f"--root is an entire mount ({mount_target}); name one exact worktree or subtree"
        )
    if not str(source).startswith("/dev/"):
        raise ConfigurationError(f"root is not backed by a directly limitable block device: {source}")

    parent_name = _require_success(
        runner(["/usr/bin/lsblk", "-ndo", "PKNAME", str(source)]),
        "physical-device discovery",
    )
    device = Path("/dev") / parent_name if parent_name else source
    try:
        device_stat = device.stat()
    except OSError as exc:
        raise ConfigurationError(f"physical device is unavailable: {device}: {exc}") from exc
    if not stat.S_ISBLK(device_stat.st_mode):
        raise ConfigurationError(f"physical device is not a block device: {device}")
    device_id = f"{os.major(device_stat.st_rdev)}:{os.minor(device_stat.st_rdev)}"
    return HostSupport(
        docker=docker,
        image=image,
        rg_binary=rg_binary,
        device=device,
        device_id=device_id,
    )


def docker_command(
    args: argparse.Namespace,
    *,
    root: Path,
    host: HostSupport,
    workload: Sequence[str],
    container_name: str,
) -> list[str]:
    read_bps = args.read_mib_per_sec * 1024 * 1024
    root_source = _mount_source(root, "resolved --root")
    rg_source = _mount_source(host.rg_binary, "resolved rg path")
    root_mount = f"type=bind,src={root_source},dst={CONTAINER_ROOT},readonly"
    rg_mount = f"type=bind,src={rg_source},dst={CONTAINER_RG},readonly"
    return [
        str(host.docker),
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        container_name,
        "--read-only",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--pids-limit",
        "64",
        "--cpu-shares",
        "32",
        "--cpus",
        "1",
        "--blkio-weight",
        "10",
        "--device-read-bps",
        f"{host.device}:{read_bps}",
        "--device-read-iops",
        f"{host.device}:{args.read_iops}",
        "--mount",
        root_mount,
        "--mount",
        rg_mount,
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--stop-timeout",
        "2",
        host.image,
        "/bin/sh",
        "-eu",
        "-c",
        HARD_LIMIT_BOOTSTRAP,
        "bounded-bootstrap",
        host.device_id,
        str(read_bps),
        str(args.read_iops),
        *workload,
    ]


def cleanup_container(container_name: str) -> None:
    docker = shutil.which("docker")
    if not docker:
        return
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "timeout": 10}
    subprocess.run([docker, "stop", "--time", "2", container_name], check=False, **quiet)
    subprocess.run([docker, "rm", "--force", container_name], check=False, **quiet)


def _settle_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def execute_docker(
    command: Sequence[str],
    *,
    container_name: str,
    timeout_seconds: int,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    cleanup: Callable[[str], None] = cleanup_container,
) -> int:
    process = popen_factory(list(command))
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"bounded-offline: timeout after {timeout_seconds}s; removing {container_name}",
            file=sys.stderr,
        )
        cleanup(container_name)
        _settle_process(process)
        return 124
    except KeyboardInterrupt:
        print(f"bounded-offline: interrupted; removing {container_name}", file=sys.stderr)
        cleanup(container_name)
        _settle_process(process)
        return 130


def _interrupt_as_keyboard(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = resolve_root(args.root)
        host = detect_host_support(root)
        workload = workload_command(args)
        container_name = f"greyhound-bounded-offline-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        command = docker_command(
            args,
            root=root,
            host=host,
            workload=workload,
            container_name=container_name,
        )
    except ConfigurationError as exc:
        print(f"bounded-offline: refused: {exc}", file=sys.stderr)
        return 2

    print(
        "bounded-offline: "
        f"container={container_name} root={root} device={host.device} "
        f"rbps={args.read_mib_per_sec}MiB/s riops={args.read_iops} "
        f"timeout={args.timeout_seconds}s workers=1",
        file=sys.stderr,
        flush=True,
    )
    previous = signal.signal(signal.SIGTERM, _interrupt_as_keyboard)
    try:
        return execute_docker(
            command,
            container_name=container_name,
            timeout_seconds=args.timeout_seconds,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
