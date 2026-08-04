"""Classify changed paths for the stable Forecasting acceptance gate."""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = ROOT / ".github/forecasting-paths.ini"
TIERS = (
    "non_forecasting",
    "ci_contract",
    "manual_prediction",
    "official_results",
    "forward_corpus",
    "operator_ui",
    "forecasting_core",
    "full_forecasting",
)
TIER_RANK = {tier: rank for rank, tier in enumerate(TIERS)}
KNOWN_STATUS_PREFIXES = frozenset({"A", "C", "D", "M", "R", "T"})
DESTRUCTIVE_STATUS_PREFIXES = frozenset({"C", "D", "R", "T"})
FOCUSED_TIERS = frozenset(TIERS) - {"non_forecasting", "full_forecasting"}


class ClassificationError(RuntimeError):
    """The change set or rules cannot be classified safely."""


@dataclass(frozen=True)
class Change:
    status: str
    paths: tuple[str, ...]


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
        "forecasting-change-rules-v2"
    ):
        raise ClassificationError("unsupported classifier rules schema")
    tier_sections = set(parser.sections()) - {"metadata"}
    if tier_sections != set(TIERS):
        raise ClassificationError(
            "classifier rules must define exactly the configured tiers"
        )
    validated: dict[str, tuple[str, ...]] = {}
    for tier in TIERS:
        patterns = tuple(
            line.strip()
            for line in parser.get(tier, "patterns", fallback="").splitlines()
            if line.strip()
        )
        if not patterns:
            raise ClassificationError(f"invalid patterns for {tier}")
        validated[tier] = patterns
    return validated


def _path_tier(
    path: str, rules: Mapping[str, Sequence[str]]
) -> tuple[str, tuple[str, ...]]:
    normalized = _normalize_path(path)
    matched = tuple(
        tier
        for tier in TIERS
        if any(fnmatch.fnmatchcase(normalized, pattern) for pattern in rules[tier])
    )
    if not matched:
        return "full_forecasting", ()
    return max(matched, key=TIER_RANK.__getitem__), matched


def classify_changes(
    changes: Iterable[Change], rules: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    change_list = list(changes)
    if not change_list:
        return {
            "tier": "full_forecasting",
            "reason": "empty_change_set_defaults_to_full",
            "paths": [],
        }

    classified_paths: list[dict[str, Any]] = []
    selected_tiers: set[str] = set()
    destructive_change = False
    for change in change_list:
        prefix = change.status[:1]
        if prefix not in KNOWN_STATUS_PREFIXES or not change.paths:
            return {
                "tier": "full_forecasting",
                "reason": f"unknown_change_status:{change.status or 'empty'}",
                "paths": classified_paths,
            }
        destructive_change = (
            destructive_change or prefix in DESTRUCTIVE_STATUS_PREFIXES
        )
        for path in change.paths:
            try:
                tier, matched = _path_tier(path, rules)
                normalized = _normalize_path(path)
            except ClassificationError as exc:
                return {
                    "tier": "full_forecasting",
                    "reason": f"uncertain_path:{exc}",
                    "paths": classified_paths,
                }
            classified_paths.append(
                {
                    "path": normalized,
                    "status": change.status,
                    "tier": tier,
                    "matched_tiers": list(matched),
                    "defaulted_to_full": not matched,
                }
            )
            selected_tiers.add(tier)

    risk_tiers = selected_tiers - {"non_forecasting"}
    if destructive_change:
        selected = "full_forecasting"
        reason = "destructive_change_defaults_to_full"
    elif any(item["defaulted_to_full"] for item in classified_paths):
        selected = "full_forecasting"
        reason = "unknown_path_defaults_to_full"
    elif "full_forecasting" in risk_tiers:
        selected = "full_forecasting"
        reason = "shared_or_high_risk_path_requires_full"
    elif len(risk_tiers & FOCUSED_TIERS) > 1:
        selected = "full_forecasting"
        reason = "incompatible_mixed_tiers_default_to_full"
    elif risk_tiers:
        selected = next(iter(risk_tiers))
        reason = "single_trusted_tier"
    else:
        selected = "non_forecasting"
        reason = "known_non_forecasting_paths"
    return {"tier": selected, "reason": reason, "paths": classified_paths}


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
        raw = subprocess.check_output(command)
    except subprocess.CalledProcessError as exc:
        raise ClassificationError(
            f"git diff failed with exit {exc.returncode}"
        ) from exc
    return parse_name_status(raw)


def write_github_output(path: Path, result: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"tier={result['tier']}\n")
        output.write(f"reason={result['reason']}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    if not args.force_full and not (args.base and args.head):
        parser.error("--base and --head are required unless --force-full is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rules = load_rules(args.rules)
        if args.force_full:
            result = {
                "tier": "full_forecasting",
                "reason": "explicit_full_validation",
                "paths": [],
            }
        else:
            result = classify_changes(git_changes(args.base, args.head), rules)
    except ClassificationError as exc:
        print(f"classifier error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.github_output:
        write_github_output(args.github_output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
