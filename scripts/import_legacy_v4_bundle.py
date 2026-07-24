#!/usr/bin/env python3
"""Reproducibly verify and atomically copy the approved legacy V4 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_at(parent_fd: int, name: str) -> tuple[int, str]:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise SystemExit("publication winner is not a regular file")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return details.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def checksum_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _validated_path(value: object, name: str) -> Path:
    if type(value) is not str or not value.strip():
        raise SystemExit(f"{name} must be a nonblank path")
    path = Path(value)
    if ".." in path.parts:
        raise SystemExit(f"{name} must not contain traversal")
    return path


def _open_parent(root: Path, parts: tuple[str, ...]) -> int:
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts:
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                os.mkdir(part, 0o755, dir_fd=descriptor)
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _copy(entry: dict[str, object], destination_root: Path) -> Path:
    source = _validated_path(entry.get("source"), "source")
    relative = _validated_path(entry.get("destination"), "destination")
    if relative.is_absolute():
        raise SystemExit("destination must be repo-relative")
    root = destination_root.resolve(strict=True)
    destination = root / relative
    expected_hash = entry.get("sha256")
    expected_size = entry.get("size")
    if (
        type(expected_hash) is not str
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        raise SystemExit("manifest checksum or size is invalid")
    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise SystemExit(f"source must be a regular non-symlink: {source}") from error
    source_details = os.fstat(source_fd)
    if (
        not stat.S_ISREG(source_details.st_mode)
        or source_details.st_size != expected_size
        or checksum_fd(source_fd) != expected_hash
    ):
        os.close(source_fd)
        raise SystemExit(f"source checksum or size mismatch: {source}")
    parent_fd = _open_parent(root, relative.parent.parts)
    try:
        Path(f"/proc/self/fd/{parent_fd}").resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        os.close(source_fd)
        os.close(parent_fd)
        raise SystemExit("opened destination parent escapes destination root") from error
    try:
        winner_size, winner_hash = checksum_at(parent_fd, destination.name)
    except FileNotFoundError:
        pass
    else:
        os.close(source_fd)
        os.close(parent_fd)
        if winner_size != expected_size or winner_hash != expected_hash:
            raise SystemExit(f"destination exists with different content: {destination}")
        return destination
    temporary_name = f".{destination.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(descriptor, "wb") as target, os.fdopen(os.dup(source_fd), "rb") as origin:
            for chunk in iter(lambda: origin.read(1024 * 1024), b""):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        copied_size, copied_hash = checksum_at(parent_fd, temporary_name)
        if copied_size != expected_size or copied_hash != expected_hash:
            raise SystemExit(f"copied artifact failed verification: {source}")
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            winner_size, winner_hash = checksum_at(parent_fd, destination.name)
            if winner_size != expected_size or winner_hash != expected_hash:
                raise SystemExit(f"publication conflict at destination: {destination}")
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(source_fd)
        os.close(parent_fd)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("model_import/legacy_v4_manifest.json")
    )
    parser.add_argument("--destination-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.manifest.is_symlink() or not args.manifest.is_file():
        raise SystemExit("manifest must be a regular non-symlink")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    destinations: set[Path] = set()
    copied: list[Path] = []
    for name in ("artifact", "metadata", "index"):
        entry = manifest.get(name)
        if not isinstance(entry, dict) or entry.get("present", True) is not True:
            raise SystemExit(f"required manifest entry is absent: {name}")
        destination = _validated_path(entry.get("destination"), f"{name} destination")
        if destination in destinations:
            raise SystemExit(f"conflicting destination: {destination}")
        destinations.add(destination)
        copied.append(_copy(entry, args.destination_root))
    for path in copied:
        print(f"verified sha256:{checksum(path)} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
