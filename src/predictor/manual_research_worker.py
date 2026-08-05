"""Default-off process guard for the request-scoped manual research adapter."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

_REQUIRED_KEYS = {
    "schema_version",
    "deployment",
    "executable",
    "entrypoint",
    "manual",
    "artifacts",
    "default_enabled",
    "research_only",
    "canonical",
    "phase7_excluded",
}


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
    if os.environ.get("MANUAL_RESEARCH_ENABLED") != "1":
        return 0
    # GHU-054 is a request-scoped adapter; activation needs a separately
    # authorized caller and must not turn this package into a retry loop.
    return 78


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
