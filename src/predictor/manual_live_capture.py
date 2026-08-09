"""Explicit, request-scoped entrypoint for the GHU-060 live child.

The CLI accepts one caller-bound race document and delegates all process,
timeout, lock, path, identity, runner, odds, and artifact validation to the
existing GHU-051 executor. It has no discovery or autonomous mode.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.predictor.manual_independent_capture import (
    canonical_bytes,
    parse_canonical_json,
)
from src.predictor.manual_independent_capture_executor import (
    execute_manual_capture_fixture,
)

_HASH40_RE = re.compile(r"^[0-9a-f]{40}$")
_PROTECTED_NAMES = {
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


class LiveCaptureCliRejected(RuntimeError):
    """Invalid explicit input; no alternate race or source is attempted."""


def live_capture_child_command(_launch: Any) -> Sequence[str]:
    """Return the only child command accepted by this entrypoint."""

    return (sys.executable, "-m", "src.predictor.manual_live_capture_child")


def _canonical_file(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LiveCaptureCliRejected("INPUT_MISSING") from exc
    try:
        return parse_canonical_json(raw)
    except (RuntimeError, ValueError) as exc:
        raise LiveCaptureCliRejected("CANONICAL_INPUT_INVALID") from exc


def _model_bytes(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise LiveCaptureCliRejected("MODEL_INPUT_INVALID")
        raw = path.read_bytes()
    except OSError as exc:
        raise LiveCaptureCliRejected("MODEL_INPUT_INVALID") from exc
    if not raw:
        raise LiveCaptureCliRejected("MODEL_INPUT_INVALID")
    return raw


def _race_input(value: Any) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if not isinstance(value, Mapping) or set(value) != {"race", "runners"}:
        raise LiveCaptureCliRejected("EXACT_INPUT_INVALID")
    race = value["race"]
    runners = value["runners"]
    if not isinstance(race, Mapping) or not isinstance(runners, list) or not runners:
        raise LiveCaptureCliRejected("EXACT_INPUT_INVALID")
    return dict(race), runners


def _forbidden(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise LiveCaptureCliRejected("PROTECTED_INPUT_INVALID")
        name, raw_path = item.split("=", 1)
        if name not in _PROTECTED_NAMES or name in result or not raw_path:
            raise LiveCaptureCliRejected("PROTECTED_INPUT_INVALID")
        result[name] = raw_path
    if set(result) != _PROTECTED_NAMES:
        raise LiveCaptureCliRejected("PROTECTED_INPUT_INVALID")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manual-live-capture")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--race-json", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--forbidden-path", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if not _HASH40_RE.fullmatch(args.source_commit) or not _HASH40_RE.fullmatch(
            args.source_tree
        ):
            raise LiveCaptureCliRejected("SOURCE_IDENTITY_INVALID")
        config = _canonical_file(args.config)
        race_document = _canonical_file(args.race_json)
        race, runners = _race_input(race_document)
        model_bytes = _model_bytes(args.model)
        forbidden_paths = _forbidden(args.forbidden_path)
        requested_url = race.get("url")
        if not isinstance(requested_url, str) or not requested_url:
            raise LiveCaptureCliRejected("EXACT_INPUT_INVALID")
        execution = execute_manual_capture_fixture(
            config=config,
            forbidden_paths=forbidden_paths,
            requested_race_url=requested_url,
            selected_race=race,
            expected_runner_set=runners,
            model_bytes=model_bytes,
            source_commit=args.source_commit,
            source_tree=args.source_tree,
            fixture_child_command=live_capture_child_command,
        )
        print(canonical_bytes(execution.artifact).decode("utf-8"))
        return 0 if execution.artifact["terminal"]["status"] == "CAPTURE_READY" else 78
    except LiveCaptureCliRejected as exc:
        print(str(exc), file=sys.stderr)
        return 78
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"LIVE_CAPTURE_REJECTED:{type(exc).__name__}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["live_capture_child_command", "main"]
