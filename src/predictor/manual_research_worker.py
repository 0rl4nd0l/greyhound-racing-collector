"""Default-off process guard for the request-scoped manual research adapter."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

_REQUIRED_KEYS = {
    "schema_version",
    "deployment",
    "executable",
    "entrypoint",
    "live_capture",
    "manual",
    "artifacts",
    "default_enabled",
    "research_only",
    "canonical",
    "phase7_excluded",
}
_LIVE_CAPTURE_KEYS = {
    "entrypoint",
    "child",
    "entrypoint_sha256",
    "child_sha256",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manual-research-worker")
    parser.add_argument("--binding", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        binding = json.loads(args.binding.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 78
    if not isinstance(binding, dict) or set(binding) != _REQUIRED_KEYS:
        return 78
    live_capture = binding["live_capture"]
    artifacts = binding["artifacts"]
    if (
        not isinstance(live_capture, dict)
        or set(live_capture) != _LIVE_CAPTURE_KEYS
        or not isinstance(artifacts, dict)
        or not isinstance(live_capture["entrypoint"], str)
        or not isinstance(live_capture["child"], str)
        or not isinstance(live_capture["entrypoint_sha256"], str)
        or not isinstance(live_capture["child_sha256"], str)
        or _SHA256_RE.fullmatch(live_capture["entrypoint_sha256"]) is None
        or _SHA256_RE.fullmatch(live_capture["child_sha256"]) is None
        or artifacts.get("live_capture") != live_capture["entrypoint_sha256"]
        or artifacts.get("live_capture_child") != live_capture["child_sha256"]
        or binding["default_enabled"] is not False
        or binding["research_only"] is not True
        or binding["canonical"] is not False
        or binding["phase7_excluded"] is not True
    ):
        return 78
    if os.environ.get("MANUAL_RESEARCH_ENABLED") != "1":
        return 0
    # GHU-054 is a request-scoped adapter; activation needs a separately
    # authorized caller and must not turn this package into a retry loop.
    return 78


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
