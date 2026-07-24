import os

import pytest

from race_collection.artifacts import (
    ChecksumMismatch,
    LocalArtifactStore,
    UnsafeArtifactPath,
)
from race_collection.domain import ArtifactChecksum, DomainValidationError


def test_content_is_addressed_by_verified_sha256(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    record = store.put(b"sealed evidence", media_type="application/json")
    assert str(record.checksum).startswith("sha256:")
    assert store.read(record.checksum) == b"sealed evidence"
    assert store.path_for(record.checksum).name == record.checksum.hex_digest


def test_duplicate_content_reuses_immutable_object(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = store.put(b"same bytes", media_type="application/octet-stream")
    target = store.path_for(first.checksum)
    first_stat = target.stat()
    second = store.put(b"same bytes", media_type="application/octet-stream")
    second_stat = target.stat()
    assert first.checksum == second.checksum
    assert first_stat.st_ino == second_stat.st_ino
    assert first_stat.st_mtime_ns == second_stat.st_mtime_ns


def test_claimed_checksum_mismatch_writes_nothing(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    wrong = ArtifactChecksum(f"sha256:{'0' * 64}")
    with pytest.raises(ChecksumMismatch):
        store.put(b"different", media_type="text/plain", expected_checksum=wrong)
    assert not store.root.exists()


def test_corrupted_stored_object_fails_closed(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    record = store.put(b"original", media_type="text/plain")
    store.path_for(record.checksum).write_bytes(b"corrupt")
    with pytest.raises(ChecksumMismatch):
        store.read(record.checksum)


def test_failed_atomic_publication_leaves_no_partial_object(tmp_path, monkeypatch):
    store = LocalArtifactStore(tmp_path / "artifacts")
    checksum = store.checksum(b"payload")

    def fail_replace(source, target):
        raise OSError("simulated crash before publication")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.put(b"payload", media_type="application/octet-stream")
    assert not store.path_for(checksum).exists()
    assert list(store.root.rglob(".incoming-*")) == []


def test_path_identity_rejects_traversal_and_symlink_directories(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(DomainValidationError):
        ArtifactChecksum("sha256:../../outside")

    checksum = store.checksum(b"payload")
    store.root.mkdir()
    (store.root / "sha256").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(UnsafeArtifactPath):
        store.put(b"payload", media_type="application/octet-stream", expected_checksum=checksum)
