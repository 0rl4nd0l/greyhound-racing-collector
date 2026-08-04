"""Select trusted proportional validation for the stable Forecasting gate."""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = ROOT / ".github/forecasting-paths.ini"
SCHEMA_VERSION = "forecasting-change-rules-v2"
FULL_COMMAND = (
    "uv run --no-project --with-requirements requirements/all.in "
    "python -m pytest -q --noconftest tests/race_collection"
)
TRUSTED_COMMANDS = {
    "full_forecasting": FULL_COMMAND,
    "manual_prediction": (
        "uv run --no-project --with-requirements requirements/all.in "
        "python -m pytest -q --noconftest tests/test_predict_race_now.py "
        "tests/test_predict_market_form_residual.py"
    ),
    "official_results": (
        "uv run --no-project --with-requirements requirements/all.in "
        "python -m pytest -q tests/test_results_ingest_official_first.py "
        "tests/test_autonomous_official_result_capture.py "
        "tests/test_expert_form_official_result_labels_packet.py"
    ),
    "race_collection_inventory": (
        "uv run --no-project --with-requirements requirements/all.in "
        "python -m pytest -q --noconftest "
        "tests/race_collection/test_phase2_domain.py"
    ),
}
FOCUSED_SUITES = (
    "manual_prediction",
    "official_results",
    "race_collection_inventory",
)
SECTIONS = (
    "metadata",
    "full_forecasting",
    *(f"focused.{suite}" for suite in FOCUSED_SUITES),
    "non_forecasting",
)
KNOWN_STATUS_PREFIXES = frozenset({"A", "C", "D", "M", "R", "T"})
UNSAFE_STATUS_PREFIXES = frozenset({"C", "D", "R", "T"})
VALID_STATUS = re.compile(r"(?:A|D|M|T|[CR](?:[1-9][0-9]?|100))\Z")


class ClassificationError(RuntimeError):
    """The change set or rules cannot be classified safely."""


@dataclass(frozen=True)
class Change:
    status: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class RuleGroup:
    tier: str
    suite: str
    command: str
    patterns: tuple[str, ...]


def _validated_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ClassificationError(f"unsafe or non-POSIX changed path: {path!r}")
    return path


def _patterns(parser: configparser.ConfigParser, section: str) -> tuple[str, ...]:
    patterns = tuple(
        line.strip()
        for line in parser.get(section, "patterns", fallback="").splitlines()
        if line.strip()
    )
    if not patterns:
        raise ClassificationError(f"invalid patterns for {section}")
    for pattern in patterns:
        if (
            "\\" in pattern
            or pattern.startswith(("/", "!", "-"))
            or ".." in PurePosixPath(pattern).parts
            or "\0" in pattern
        ):
            raise ClassificationError(f"unsafe pattern in {section}: {pattern!r}")
    return patterns


def load_rules(path: Path = DEFAULT_RULES) -> tuple[RuleGroup, ...]:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8") as rules_file:
            parser.read_file(rules_file)
    except (OSError, configparser.Error, UnicodeError) as exc:
        raise ClassificationError(f"unable to load classifier rules: {exc}") from exc
    if tuple(parser.sections()) != SECTIONS:
        raise ClassificationError("classifier rules sections or order are invalid")
    if set(parser["metadata"]) != {"schema_version"}:
        raise ClassificationError("metadata options are invalid")
    if parser["metadata"]["schema_version"] != SCHEMA_VERSION:
        raise ClassificationError("unsupported classifier rules schema")

    groups: list[RuleGroup] = []
    configured = (
        ("full_forecasting", "full_forecasting", "full_forecasting"),
        *(
            (f"focused.{suite}", "focused_forecasting", suite)
            for suite in FOCUSED_SUITES
        ),
        ("non_forecasting", "non_forecasting", ""),
    )
    for section, tier, expected_suite in configured:
        if set(parser[section]) != {"suite", "command", "patterns"}:
            raise ClassificationError(f"invalid options for {section}")
        suite = parser[section]["suite"].strip()
        command = " ".join(parser[section]["command"].split())
        expected_command = TRUSTED_COMMANDS.get(expected_suite, "")
        if suite != expected_suite or command != expected_command:
            raise ClassificationError(f"untrusted suite or command in {section}")
        groups.append(RuleGroup(tier, suite, command, _patterns(parser, section)))
    return tuple(groups)


def _classify_path(path: str, rules: Sequence[RuleGroup]) -> dict[str, Any]:
    exact_path = _validated_path(path)
    matches = [
        group
        for group in rules
        if any(fnmatch.fnmatchcase(exact_path, pattern) for pattern in group.patterns)
    ]
    if not matches:
        return {
            "path": exact_path,
            "rule_tier": "full_forecasting",
            "rule_suite": "full_forecasting",
            "matched_groups": [],
            "defaulted_to_full": True,
        }
    rank = {"non_forecasting": 0, "focused_forecasting": 1, "full_forecasting": 2}
    selected = max(matches, key=lambda group: rank[group.tier])
    return {
        "path": exact_path,
        "rule_tier": selected.tier,
        "rule_suite": selected.suite,
        "matched_groups": [group.suite or "non_forecasting" for group in matches],
        "defaulted_to_full": False,
    }


def _selection(tier: str, suite: str, reason: str, paths: list[dict[str, Any]]):
    return {
        "tier": tier,
        "suite": suite,
        "command": TRUSTED_COMMANDS.get(suite, ""),
        "reason": reason,
        "paths": paths,
    }


def force_full_result() -> dict[str, Any]:
    return _selection(
        "full_forecasting", "full_forecasting", "manual_dispatch_forces_full", []
    )


def classify_changes(
    changes: Iterable[Change], rules: Sequence[RuleGroup]
) -> dict[str, Any]:
    change_list = list(changes)
    if not change_list:
        return _selection(
            "full_forecasting",
            "full_forecasting",
            "empty_change_set_forces_full",
            [],
        )

    classified: list[dict[str, Any]] = []
    unsafe_status = False
    malformed_status = False
    for change in change_list:
        prefix = change.status[:1]
        status_is_valid = (
            bool(change.paths)
            and prefix in KNOWN_STATUS_PREFIXES
            and bool(VALID_STATUS.fullmatch(change.status))
        )
        if not status_is_valid:
            malformed_status = True
        if not change.paths:
            continue
        if status_is_valid and prefix in UNSAFE_STATUS_PREFIXES:
            unsafe_status = True
        for path in change.paths:
            try:
                item = _classify_path(path, rules)
            except ClassificationError:
                item = {
                    "path": path,
                    "rule_tier": "full_forecasting",
                    "rule_suite": "full_forecasting",
                    "matched_groups": [],
                    "defaulted_to_full": True,
                }
            item["status"] = change.status
            classified.append(item)

    if malformed_status:
        return _selection(
            "full_forecasting",
            "full_forecasting",
            "invalid_git_status_forces_full",
            classified,
        )
    if unsafe_status:
        return _selection(
            "full_forecasting",
            "full_forecasting",
            "unsafe_git_status_forces_full",
            classified,
        )
    if any(item["rule_tier"] == "full_forecasting" for item in classified):
        reason = (
            "unknown_or_ambiguous_path_forces_full"
            if any(item["defaulted_to_full"] for item in classified)
            else "full_rule_match"
        )
        return _selection("full_forecasting", "full_forecasting", reason, classified)
    focused = {
        item["rule_suite"]
        for item in classified
        if item["rule_tier"] == "focused_forecasting"
    }
    if len(focused) > 1:
        return _selection(
            "full_forecasting",
            "full_forecasting",
            "multiple_focused_subsystems_force_full",
            classified,
        )
    if focused:
        suite = focused.pop()
        return _selection(
            "focused_forecasting", suite, "single_focused_subsystem", classified
        )
    return _selection("non_forecasting", "", "only_non_forecasting_paths", classified)


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
            changes.append(Change(status, paths))
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
        return parse_name_status(subprocess.check_output(command))
    except subprocess.CalledProcessError as exc:
        raise ClassificationError(
            f"git diff failed with exit {exc.returncode}"
        ) from exc


def _summary_code(value: object) -> str:
    encoded = "".join(f"&#{ord(character)};" for character in str(value))
    return f"<code>{encoded}</code>"


def github_summary(result: Mapping[str, Any]) -> str:
    lines = [
        "### Forecasting change classification",
        "",
        "#### Rule classification",
        "",
        "| Status | Exact changed path | Rule tier | Rule suite |",
        "| --- | --- | --- | --- |",
    ]
    for item in result["paths"]:
        lines.append(
            "| "
            + " | ".join(
                _summary_code(value)
                for value in (
                    item["status"],
                    item["path"],
                    item["rule_tier"],
                    item["rule_suite"] or "none",
                )
            )
            + " |"
        )
    if not result["paths"]:
        lines.append("| none | none | none | none |")
    lines.extend(
        [
            "",
            "#### Effective trusted execution selection",
            "",
            f"- Tier: {_summary_code(result['tier'])}",
            f"- Focused suite: {_summary_code(result['suite'] or 'none')}",
            f"- Exact command: {_summary_code(result['command'] or 'none')}",
            f"- Reason: {_summary_code(result['reason'])}",
            "",
        ]
    )
    return "\n".join(lines)


def write_github_output(path: Path, result: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"tier={result['tier']}\n")
        output.write(f"suite={result['suite']}\n")
        output.write(f"reason={result['reason']}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--github-summary", type=Path)
    args = parser.parse_args(argv)
    if not args.force_full and not (args.base and args.head):
        parser.error("--base and --head are required unless --force-full is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rules = load_rules(args.rules)
        result = (
            force_full_result()
            if args.force_full
            else classify_changes(git_changes(args.base, args.head), rules)
        )
    except ClassificationError as exc:
        print(f"classifier error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.github_output:
        write_github_output(args.github_output, result)
    if args.github_summary:
        with args.github_summary.open("a", encoding="utf-8") as summary:
            summary.write(github_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
