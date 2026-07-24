import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/import_legacy_v4_bundle.py").resolve()


def _manifest(tmp_path, *, duplicate=False):
    entries = {}
    for number, name in enumerate(("artifact", "metadata", "index"), 1):
        source = tmp_path / f"{name}.source"
        content = f"fixture-{name}".encode()
        source.write_bytes(content)
        entries[name] = {
            "present": True,
            "source": str(source),
            "destination": "copied/artifact" if duplicate else f"copied/{name}",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "purpose": name,
        }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path, entries


def _run(manifest, root):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--destination-root",
            str(root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_importer_copies_all_files_atomically_and_replays(tmp_path):
    manifest, entries = _manifest(tmp_path)
    first = _run(manifest, tmp_path)
    second = _run(manifest, tmp_path)
    assert first.returncode == second.returncode == 0
    for entry in entries.values():
        destination = tmp_path / entry["destination"]
        assert destination.read_bytes() == Path(entry["source"]).read_bytes()
    assert not list((tmp_path / "copied").glob("*.tmp"))


def test_importer_rejects_mismatch_and_preserves_unrelated_files(tmp_path):
    manifest, entries = _manifest(tmp_path)
    unrelated = tmp_path / "copied" / "unrelated.tmp"
    unrelated.parent.mkdir()
    unrelated.write_text("owned", encoding="utf-8")
    entries["metadata"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    result = _run(manifest, tmp_path)
    assert result.returncode != 0
    assert unrelated.read_text(encoding="utf-8") == "owned"


def test_importer_rejects_symlink_traversal_and_duplicate_destinations(tmp_path):
    manifest, entries = _manifest(tmp_path)
    entries["artifact"]["destination"] = "../escape"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    assert _run(manifest, tmp_path).returncode != 0
    manifest, _ = _manifest(tmp_path, duplicate=True)
    assert _run(manifest, tmp_path).returncode != 0
    manifest, entries = _manifest(tmp_path)
    source = Path(entries["artifact"]["source"])
    target = tmp_path / "target"
    target.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target)
    assert _run(manifest, tmp_path).returncode != 0


def test_importer_rejects_parent_symlink_escape_and_publication_conflict(tmp_path, monkeypatch):
    manifest, entries = _manifest(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    copied = tmp_path / "copied"
    copied.symlink_to(outside, target_is_directory=True)
    assert _run(manifest, tmp_path).returncode != 0
    copied.unlink()

    spec = importlib.util.spec_from_file_location("phase3_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    entry = entries["artifact"]
    destination = tmp_path / entry["destination"]
    real_link = module.os.link

    def conflicting_link(source, target, **kwargs):
        destination.write_bytes(b"competitor")
        raise FileExistsError

    monkeypatch.setattr(module.os, "link", conflicting_link)
    try:
        with pytest.raises(SystemExit, match="publication conflict"):
            module._copy(entry, tmp_path)
    finally:
        monkeypatch.setattr(module.os, "link", real_link)
    assert destination.read_bytes() == b"competitor"


def test_importer_parent_swap_creates_nothing_outside_root(tmp_path, monkeypatch):
    _, entries = _manifest(tmp_path)
    spec = importlib.util.spec_from_file_location("phase3_importer_swap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    entry = entries["artifact"]
    entry["destination"] = "level1/level2/artifact"
    copied = tmp_path / "level1"
    copied.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original = tmp_path / "level1-original"
    real_open = module.os.open
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "level2" and not swapped:
            swapped = True
            copied.rename(original)
            copied.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", swapping_open)
    module._copy(entry, tmp_path)
    assert list(outside.iterdir()) == []
    assert (original / "level2/artifact").is_file()
