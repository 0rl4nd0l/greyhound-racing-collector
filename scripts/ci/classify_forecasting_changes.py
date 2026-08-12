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
DESTRUCTIVE_STATUS_PREFIXES = frozenset({"C", "D", "R", "T"})
FOCUSED_TIERS = frozenset(TIERS) - {"non_forecasting", "full_forecasting"}
PRODUCT_TIERS = FOCUSED_TIERS - {"ci_contract"}

# GHU-058 changes this shared helper only to add a readiness-publication
# rollback hook. Keep unrelated future changes to the shared file full-risk by
# admitting only this exact base/head blob pair into manual_prediction.
EXACT_MANUAL_PREDICTION_BLOB_PAIRS = {
    "race_collection/synchronous_manual_capture.py": (
        "77b8aa8dd7391203162876c384e823e5c696d47d",
        "6a984e546c08494cbaa561b5aedd202087b7be6f",
    ),
}


class ClassificationError(RuntimeError):
    """The change set or rules cannot be classified safely."""


@dataclass(frozen=True)
class Change:
    status: str
    paths: tuple[str, ...]
    manual_prediction_paths: tuple[str, ...] = ()


def _is_known_status(status: str) -> bool:
    if status in {"A", "D", "M", "T"}:
        return True
    score = status[1:]
    if status[:1] not in {"C", "R"} or not 1 <= len(score) <= 3 or not score.isdigit():
        return False
    return 0 <= int(score) <= 100


def _full_result(
    reason: str,
    paths: Sequence[Mapping[str, Any]] = (),
    *,
    ci_contract_changed: bool = False,
) -> dict[str, Any]:
    return {
        "tier": "full_forecasting",
        "reason": reason,
        "ci_contract_changed": ci_contract_changed,
        "paths": list(paths),
    }


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
        return _full_result("empty_change_set_defaults_to_full")

    classified_paths: list[dict[str, Any]] = []
    selected_tiers: set[str] = set()
    destructive_change = False
    for change in change_list:
        prefix = change.status[:1]
        if not _is_known_status(change.status) or not change.paths:
            return _full_result(
                f"unknown_change_status:{change.status or 'empty'}",
                classified_paths,
                ci_contract_changed="ci_contract" in selected_tiers,
            )
        destructive_change = (
            destructive_change or prefix in DESTRUCTIVE_STATUS_PREFIXES
        )
        for path in change.paths:
            try:
                normalized = _normalize_path(path)
                if (
                    change.status == "M"
                    and normalized in change.manual_prediction_paths
                    and normalized in EXACT_MANUAL_PREDICTION_BLOB_PAIRS
                ):
                    tier, matched = "manual_prediction", ("manual_prediction",)
                else:
                    tier, matched = _path_tier(normalized, rules)
            except ClassificationError as exc:
                return _full_result(
                    f"uncertain_path:{exc}",
                    classified_paths,
                    ci_contract_changed="ci_contract" in selected_tiers,
                )
            classified_paths.append(
                {
                    "path": normalized,
                    "status": change.status,
                    "tier": tier,
                    "matched_tiers": list(matched),
                    "defaulted_to_full": not matched,
                }
            )
            selected_tiers.update(matched or (tier,))

    risk_tiers = selected_tiers - {"non_forecasting"}
    ci_contract_changed = "ci_contract" in risk_tiers
    product_tiers = risk_tiers & PRODUCT_TIERS
    if destructive_change:
        selected = "full_forecasting"
        reason = "destructive_change_defaults_to_full"
    elif any(item["defaulted_to_full"] for item in classified_paths):
        selected = "full_forecasting"
        reason = "unknown_path_defaults_to_full"
    elif "full_forecasting" in risk_tiers:
        selected = "full_forecasting"
        reason = "shared_or_high_risk_path_requires_full"
    elif len(product_tiers) > 1:
        selected = "full_forecasting"
        reason = "incompatible_mixed_tiers_default_to_full"
    elif product_tiers:
        selected = next(iter(product_tiers))
        reason = (
            "single_product_tier_with_ci_contract"
            if ci_contract_changed
            else "single_trusted_tier"
        )
    elif ci_contract_changed:
        selected = "ci_contract"
        reason = "single_trusted_tier"
    else:
        selected = "non_forecasting"
        reason = "known_non_forecasting_paths"
    return {
        "tier": selected,
        "reason": reason,
        "ci_contract_changed": ci_contract_changed,
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


def _exact_manual_prediction_paths(
    base: str, head: str, changes: Sequence[Change]
) -> frozenset[str]:
    eligible: set[str] = set()
    for change in changes:
        if change.status != "M":
            continue
        for path in change.paths:
            expected = EXACT_MANUAL_PREDICTION_BLOB_PAIRS.get(path)
            if expected is None:
                continue
            try:
                base_blob = subprocess.check_output(
                    ["git", "rev-parse", f"{base}:{path}"],
                    cwd=ROOT,
                    text=True,
                ).strip()
                head_blob = subprocess.check_output(
                    ["git", "rev-parse", f"{head}:{path}"],
                    cwd=ROOT,
                    text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ClassificationError(
                    f"unable to verify exact shared-path change: {path}"
                ) from exc
            if (base_blob, head_blob) == expected:
                eligible.add(path)
    return frozenset(eligible)


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
    changes = parse_name_status(raw)
    manual_prediction_paths = _exact_manual_prediction_paths(base, head, changes)
    return [
        Change(
            status=change.status,
            paths=change.paths,
            manual_prediction_paths=tuple(
                path for path in change.paths if path in manual_prediction_paths
            ),
        )
        for change in changes
    ]


def write_github_output(path: Path, result: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"tier={result['tier']}\n")
        output.write(
            "classified_tier="
            f"{result.get('classified_tier', result['tier'])}\n"
        )
        output.write(f"reason={result['reason']}\n")
        output.write(
            "ci_contract_changed="
            f"{str(bool(result['ci_contract_changed'])).lower()}\n"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument(
        "--pr-fast",
        action="store_true",
        help="Use the bounded PR fallback instead of the long suite for full-risk changes",
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    if args.force_full and args.pr_fast:
        parser.error("--force-full and --pr-fast are mutually exclusive")
    if not args.force_full and not (args.base and args.head):
        parser.error("--base and --head are required unless --force-full is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.force_full:
        result = _full_result("explicit_full_validation")
    else:
        try:
            rules = load_rules(args.rules)
            result = classify_changes(git_changes(args.base, args.head), rules)
        except ClassificationError as exc:
            print(f"classifier warning: {exc}; defaulting to full", file=sys.stderr)
            result = _full_result("classifier_error_defaults_to_full")
    classified_tier = result["tier"]
    if args.pr_fast and result["tier"] == "full_forecasting":
        result = {
            **result,
            "tier": "pr_fast",
            "classified_tier": classified_tier,
            "reason": f"pr_fast_fallback:{result['reason']}",
        }
    else:
        result = {**result, "classified_tier": classified_tier}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.github_output:
        write_github_output(args.github_output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
