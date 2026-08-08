"""Classify changed paths for the stable backend test checks."""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = ROOT / ".github/backend-test-paths.ini"
DESTRUCTIVE_STATUS_PREFIXES = frozenset({"C", "D", "R", "T"})


class ClassificationError(RuntimeError):
    """The changed paths or classifier rules cannot be classified safely."""


@dataclass(frozen=True)
class Change:
    status: str
    paths: tuple[str, ...]


def _is_known_status(status: str) -> bool:
    if status in {"A", "D", "M", "T"}:
        return True
    score = status[1:]
    return (
        status[:1] in {"C", "R"}
        and 1 <= len(score) <= 3
        and score.isdigit()
        and 0 <= int(score) <= 100
    )


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ClassificationError(f"unsafe changed path: {path!r}")
    return candidate.as_posix()


def load_rules(path: Path = DEFAULT_RULES) -> dict[str, tuple[str, ...]]:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8") as rules_file:
            parser.read_file(rules_file)
    except (OSError, configparser.Error) as exc:
        raise ClassificationError(f"unable to load classifier rules: {exc}") from exc

    if parser.get("metadata", "schema_version", fallback=None) != (
        "backend-test-routing-v1"
    ):
        raise ClassificationError("unsupported classifier rules schema")
    if set(parser.sections()) - {"metadata"} != {
        "backend_excluded",
        "ui_only",
        "ui_backend",
    }:
        raise ClassificationError("classifier rules must define exactly three sections")

    rules: dict[str, tuple[str, ...]] = {}
    for section in ("backend_excluded", "ui_only", "ui_backend"):
        patterns = tuple(
            line.strip()
            for line in parser.get(section, "patterns", fallback="").splitlines()
            if line.strip()
        )
        if not patterns:
            raise ClassificationError(f"invalid patterns for {section}")
        rules[section] = patterns
    return rules


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _fail_closed(reason: str, paths: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    return {
        "backend_required": True,
        "ui_e2e_required": True,
        "trusted": False,
        "reason": reason,
        "paths": list(paths),
    }


def classify_changes(
    changes: Iterable[Change], rules: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    change_list = list(changes)
    if not change_list:
        return _fail_closed("empty_change_set_defaults_to_backend")

    classified_paths: list[dict[str, Any]] = []
    backend_required = False
    ui_e2e_required = False
    for change in change_list:
        if not _is_known_status(change.status) or not change.paths:
            return _fail_closed(
                f"unknown_change_status:{change.status or 'empty'}",
                classified_paths,
            )
        destructive_change = change.status[:1] in DESTRUCTIVE_STATUS_PREFIXES
        for path in change.paths:
            try:
                normalized = _normalize_path(path)
            except ClassificationError as exc:
                return _fail_closed(f"uncertain_path:{exc}", classified_paths)

            ui_only = _matches(normalized, rules["ui_only"])
            ui_match = ui_only or _matches(normalized, rules["ui_backend"])
            excluded = ui_only or _matches(normalized, rules["backend_excluded"])
            if destructive_change or not excluded:
                backend_required = True
            ui_e2e_required = ui_e2e_required or ui_match
            classified_paths.append(
                {
                    "path": normalized,
                    "status": change.status,
                    "backend_excluded": excluded,
                    "ui_e2e_match": ui_match,
                    "defaulted_to_backend": not excluded,
                }
            )

    if backend_required:
        reason = "backend_risk_path_or_destructive_change"
    elif ui_e2e_required:
        reason = "ui_e2e_only_paths"
    else:
        reason = "known_non_backend_paths"
    return {
        "backend_required": backend_required,
        "ui_e2e_required": ui_e2e_required,
        "trusted": True,
        "reason": reason,
        "paths": classified_paths,
    }


def parse_name_status(raw: bytes) -> list[Change]:
    tokens = raw.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    changes: list[Change] = []
    index = 0
    try:
        while index < len(tokens):
            status = tokens[index].decode("utf-8")
            index += 1
            path_count = 2 if status[:1] in {"R", "C"} else 1
            paths = tuple(
                tokens[position].decode("utf-8")
                for position in range(index, index + path_count)
            )
            index += path_count
            changes.append(Change(status=status, paths=paths))
    except (IndexError, UnicodeDecodeError) as exc:
        raise ClassificationError("malformed git name-status output") from exc
    if index != len(tokens):
        raise ClassificationError("trailing git name-status fields")
    return changes


def git_changes(base: str, head: str) -> list[Change]:
    command = [
        "git",
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        "--find-copies",
        f"{base}...{head}",
    ]
    try:
        raw = subprocess.check_output(command, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ClassificationError(f"git diff failed: {exc}") from exc
    return parse_name_status(raw)


def write_github_output(path: Path, result: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key in ("backend_required", "ui_e2e_required", "trusted"):
            output.write(f"{key}={str(bool(result[key])).lower()}\n")
        output.write(f"reason={result['reason']}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    if not args.force_all and not (args.base and args.head):
        parser.error("--base and --head are required unless --force-all is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.force_all:
            result = {
                "backend_required": True,
                "ui_e2e_required": True,
                "trusted": True,
                "reason": "non_pull_request_defaults_to_all_backend_checks",
                "paths": [],
            }
        else:
            result = classify_changes(git_changes(args.base, args.head), load_rules(args.rules))
    except ClassificationError as exc:
        result = _fail_closed(f"classifier_error:{exc}")
    if args.github_output:
        write_github_output(args.github_output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
