"""Content-addressed immutable artifact storage."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Protocol

from .domain import ArtifactChecksum, EvidenceArtifact


class ArtifactStoreError(RuntimeError):
    """Base artifact-store failure."""


class ChecksumMismatch(ArtifactStoreError):
    """Artifact bytes do not match their claimed content identity."""


class UnsafeArtifactPath(ArtifactStoreError):
    """A resolved artifact path leaves or aliases the configured root."""


class ArtifactStore(Protocol):
    def put(
        self,
        content: bytes,
        *,
        media_type: str,
        expected_checksum: ArtifactChecksum | None = None,
    ) -> EvidenceArtifact: ...

    def read(self, checksum: ArtifactChecksum) -> bytes: ...

    def verify(self, checksum: ArtifactChecksum) -> EvidenceArtifact: ...


@dataclass(frozen=True, slots=True)
class LocalArtifactStore:
    """Safe local SHA-256 store using atomic same-filesystem publication."""

    root: Path

    def __init__(self, root: str | Path):
        object.__setattr__(self, "root", Path(root).resolve())

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise UnsafeArtifactPath("artifact root must be a real directory")

    def path_for(self, checksum: ArtifactChecksum) -> Path:
        digest = checksum.hex_digest
        path = self.root / "sha256" / digest[:2] / digest[2:4] / digest
        try:
            path.resolve(strict=False).relative_to(self.root)
        except ValueError as error:
            raise UnsafeArtifactPath("artifact path escapes configured root") from error
        return path

    @staticmethod
    def checksum(content: bytes) -> ArtifactChecksum:
        return ArtifactChecksum(f"sha256:{hashlib.sha256(content).hexdigest()}")

    def put(
        self,
        content: bytes,
        *,
        media_type: str,
        expected_checksum: ArtifactChecksum | None = None,
    ) -> EvidenceArtifact:
        actual = self.checksum(content)
        if expected_checksum is not None and actual != expected_checksum:
            raise ChecksumMismatch(f"expected {expected_checksum}, computed {actual}")
        self._ensure_root()
        target = self.path_for(actual)
        self._prepare_parent(target.parent)
        if target.exists():
            return self.verify(actual, media_type=media_type)

        descriptor, temporary_name = tempfile.mkstemp(prefix=".incoming-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            if self.checksum(temporary.read_bytes()) != actual:
                raise ChecksumMismatch("temporary artifact failed pre-publication verification")
            if target.exists():
                temporary.unlink()
                return self.verify(actual, media_type=media_type)
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return self.verify(actual, media_type=media_type)

    def _reject_symlink_components(self, directory: Path) -> None:
        relative = directory.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise UnsafeArtifactPath(f"artifact directory is a symlink: {current}")

    def _prepare_parent(self, directory: Path) -> None:
        relative = directory.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise UnsafeArtifactPath(f"artifact directory is a symlink: {current}")
            current.mkdir(exist_ok=True)

    def read(self, checksum: ArtifactChecksum) -> bytes:
        target = self.path_for(checksum)
        self._reject_symlink_components(target.parent)
        if target.is_symlink():
            raise UnsafeArtifactPath("artifact object must not be a symlink")
        try:
            content = target.read_bytes()
        except FileNotFoundError as error:
            raise ArtifactStoreError(f"artifact not found: {checksum}") from error
        actual = self.checksum(content)
        if actual != checksum:
            raise ChecksumMismatch(f"stored artifact {checksum} computed as {actual}")
        return content

    def verify(
        self, checksum: ArtifactChecksum, *, media_type: str = "application/octet-stream"
    ) -> EvidenceArtifact:
        content = self.read(checksum)
        return EvidenceArtifact(
            checksum=checksum,
            media_type=media_type,
            byte_size=len(content),
            created_at=datetime.now(timezone.utc),
        )
