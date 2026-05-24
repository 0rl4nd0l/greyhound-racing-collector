"""Leakage guards for target-race prediction and feature matrices."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


FORBIDDEN_TARGET_FEATURES = {
    "plc",
    "place",
    "placing",
    "position",
    "finish",
    "finish_position",
    "finishing_position",
    "time",
    "individual_time",
    "race_time_result",
    "bon",
    "bonus_time",
    "win_time",
    "winning_time",
    "mgn",
    "margin",
    "beaten_margin",
    "winner",
    "winner_name",
    "winner_odds",
    "winner_margin",
    "result",
    "results",
    "result_status",
    "results_status",
    "scraped_raw_result",
    "scraped_finish_position",
    "sp",
    "starting_price",
    "odds",
    "odds_decimal",
    "market_odds",
    "payout",
    "dividend",
    "future_result",
    "future_position",
    "future_time",
}


ALLOWED_LABEL_COLUMNS = {"target", "label", "is_winner", "is_placer"}


def normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def find_forbidden_target_columns(
    columns: Iterable[object],
    *,
    allow_labels: bool = False,
) -> list[str]:
    forbidden: list[str] = []
    for column in columns:
        normalized = normalize_column_name(column)
        if allow_labels and normalized in ALLOWED_LABEL_COLUMNS:
            continue
        if normalized in FORBIDDEN_TARGET_FEATURES:
            forbidden.append(str(column))
    return sorted(set(forbidden))


def strip_target_leakage_columns(
    df: pd.DataFrame,
    *,
    allow_labels: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    dropped = find_forbidden_target_columns(df.columns, allow_labels=allow_labels)
    if not dropped:
        return df, []
    return df.drop(columns=dropped, errors="ignore"), dropped


def assert_no_target_leakage_columns(
    columns: Iterable[object],
    *,
    allow_labels: bool = False,
) -> None:
    forbidden = find_forbidden_target_columns(columns, allow_labels=allow_labels)
    if forbidden:
        raise AssertionError(
            "Target-race leakage fields present in feature matrix: "
            + ", ".join(forbidden)
        )


def audit_feature_matrix(
    df: pd.DataFrame,
    *,
    allow_labels: bool = False,
) -> dict[str, object]:
    forbidden = find_forbidden_target_columns(df.columns, allow_labels=allow_labels)
    return {
        "passed": not forbidden,
        "forbidden_columns": forbidden,
        "feature_count": int(len(df.columns)),
    }
