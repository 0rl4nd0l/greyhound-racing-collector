#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config.paths import DATA_DIR
from utils.checksum import file_sha256
from utils.manifest import IngestionManifest


def load_manifest(path: Path | None = None) -> IngestionManifest:
    mpath = path or (DATA_DIR / "ingestion_manifest.json")
    return IngestionManifest.load(mpath)


def cmd_list(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    data = {
        k: {"file_path": v.file_path, "checksum": v.checksum}
        for k, v in manifest._entries.items()  # internal read for convenience
    }
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        if not data:
            print("No entries in manifest.")
            return 0
        for k in sorted(data.keys()):
            v = data[k]
            print(f"{k} -> {v['file_path']} | {v['checksum']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    entry = manifest.get(args.key)
    if not entry:
        print(f"Key not found: {args.key}", file=sys.stderr)
        return 1
    payload = {"file_path": entry.file_path, "checksum": entry.checksum}
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if args.json
        else f"{args.key} -> {entry.file_path} | {entry.checksum}"
    )
    return 0


def verify_entry(key: str, file_path: str, checksum: str) -> tuple[bool, str]:
    p = Path(file_path)
    if not p.exists():
        return False, "missing_file"
    actual = file_sha256(p)
    return (actual == checksum), ("ok" if actual == checksum else "checksum_mismatch")


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    keys = [args.key] if args.key else list(manifest._entries.keys())
    if not keys:
        print("No entries to verify.")
        return 0
    failures = 0
    results = {}
    for k in keys:
        entry = manifest.get(k)
        if not entry:
            results[k] = {"status": "missing_key"}
            failures += 1
            continue
        ok, status = verify_entry(k, entry.file_path, entry.checksum)
        results[k] = {
            "status": status,
            "file_path": entry.file_path,
            "expected_checksum": entry.checksum,
        }
        if not ok:
            failures += 1
            if args.autofix and status == "checksum_mismatch":
                # recompute and update
                actual = file_sha256(Path(entry.file_path))
                manifest.update_entry(k, entry.file_path, actual)
    if args.autofix and failures:
        manifest.save(args.manifest)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for k in sorted(results.keys()):
            r = results[k]
            status = r["status"]
            if status == "ok":
                print(f"[OK]  {k}")
            elif status == "checksum_mismatch":
                print(f"[MISMATCH] {k} -> {r['file_path']}")
            elif status == "missing_file":
                print(f"[MISSING FILE] {k} -> {r.get('file_path')}")
            else:
                print(f"[MISSING KEY] {k}")
    return 0 if failures == 0 else 2


def cmd_set(args: argparse.Namespace) -> int:
    p = Path(args.file)
    if not p.exists():
        print(f"File not found: {p}", file=sys.stderr)
        return 1
    checksum = file_sha256(p)
    manifest = load_manifest(args.manifest)
    manifest.update_entry(args.key, str(p), checksum)
    manifest.save(args.manifest)
    if args.json:
        print(
            json.dumps(
                {"key": args.key, "file_path": str(p), "checksum": checksum},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"Updated {args.key} -> {p} | {checksum}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingestion manifest CLI")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to manifest JSON (default: DATA_DIR/ingestion_manifest.json)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all manifest entries")
    p_list.add_argument("--json", action="store_true", help="Output JSON")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show a specific manifest entry")
    p_show.add_argument("key", help="Canonical key (e.g., 2025-07-28_gosf_R4.csv)")
    p_show.add_argument("--json", action="store_true", help="Output JSON")
    p_show.set_defaults(func=cmd_show)

    p_verify = sub.add_parser("verify", help="Verify checksum(s) against files")
    p_verify.add_argument("--key", help="Canonical key to verify (omit to verify all)")
    p_verify.add_argument(
        "--autofix",
        action="store_true",
        help="Update manifest with recomputed checksums on mismatch",
    )
    p_verify.add_argument("--json", action="store_true", help="Output JSON")
    p_verify.set_defaults(func=cmd_verify)

    p_set = sub.add_parser(
        "set", help="Set or update an entry from a file (recomputes checksum)"
    )
    p_set.add_argument("key", help="Canonical key to set/update")
    p_set.add_argument("file", help="Path to file for this key")
    p_set.add_argument("--json", action="store_true", help="Output JSON")
    p_set.set_defaults(func=cmd_set)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
