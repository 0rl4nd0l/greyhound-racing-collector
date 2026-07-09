#!/usr/bin/env python3
"""Join dog-form pre-race features onto no-box actual-win smoke rows.

The output remains report-only. It copies only dog-form feature families from
the expanded historical dataset and explicitly excludes box, race-number,
calendar, finish-order, and target metadata feature groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "no_box_actual_win_dog_form_feature_join_v1"
ROWS_SCHEMA_VERSION = "no_box_actual_win_dog_form_feature_rows_v1"
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}
FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL = [
    "write_official_safe_labels",
    "mutate_db",
    "regenerate_canonical_dataset",
    "train_or_promote_model",
    "update_registry",
    "enable_tgr",
    "betting_or_ev_action",
]
FORBIDDEN_ROW_FIELDS = {
    "box_number",
    "official_box_number",
    "db_box_number",
    "finish_position",
    "official_finish_position",
    "db_finish_position",
    "db_result_position",
    "result_position",
    "placing",
    "scraped_finish_position",
}
FORBIDDEN_FEATURE_NAMES = {
    "box_number",
    "box_band_inside",
    "box_band_middle",
    "box_band_outside",
    "race_number",
    "target_day_of_week",
    "target_month",
    "field_size",
}
DOG_FORM_FEATURE_PREFIXES = (
    "prior_",
    "days_since_",
    "recent_",
    "career_",
    "starts_same_",
    "win_rate_same_",
    "place_rate_same_",
    "best_time_same_",
    "avg_time_same_",
    "median_time_same_",
    "same_distance_",
    "same_grade_",
    "grade_change_",
    "grade_strength_",
    "last_start_",
    "weight_",
    "sectional_",
)
HISTORY_FILL_POLICIES = {"all", "no_outcome_proxy_fields"}
TERMINAL_HISTORY_STATUS_TAGS = {"nbt", "ntt", "nbtt", "na"}
HISTORY_OUTCOME_PROXY_FEATURES = {
    "career_avg_finish",
    "career_best_finish",
    "career_place_rate",
    "career_win_rate",
    "place_rate_same_distance",
    "place_rate_same_venue",
    "recent_finish_best_5",
    "recent_finish_mean_3",
    "recent_finish_mean_5",
    "recent_place_rate_5",
    "recent_win_rate_5",
    "same_grade_place_rate",
    "same_grade_win_rate",
    "win_rate_same_distance",
    "win_rate_same_venue",
}
CSV_BASE_FIELDS = [
    "race_id",
    "legacy_race_id",
    "identity_key",
    "race_date",
    "venue",
    "race_number",
    "dog_name_key",
    "dog_name",
    "actual_win",
    "candidate_kind",
    "field_scope",
    "field_complete_for_ranking",
    "feature_join_status",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_row_not_object:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root = root or ROOT
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{logical}") from exc
    return resolved, relative


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    return _repo_output_path(path, root)[1]


def _assert_output_dir_safe(output_dir: Path) -> Path:
    resolved, relative = _repo_output_path(output_dir)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _name_key(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", text)
    text = text.lower().replace("'", "").replace('"', "").replace("`", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _strip_terminal_history_status_tag(value: Any) -> str:
    parts = _name_key(value).split()
    while parts:
        if parts[-1] in TERMINAL_HISTORY_STATUS_TAGS:
            parts = parts[:-1]
            continue
        if len(parts) >= 2 and parts[-2:] == ["n", "a"]:
            parts = parts[:-2]
            continue
        break
    return " ".join(parts)


def _history_lookup_key_candidates(
    row: Mapping[str, Any],
    primary_dog_key: str,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(value: Any, status: str) -> None:
        key = _name_key(value)
        if key and key not in seen:
            seen.add(key)
            candidates.append((key, status))

    add(primary_dog_key, "MATCHED")
    add(row.get("dog_name_key"), "MATCHED")
    add(row.get("dog_name"), "MATCHED")

    for value in (primary_dog_key, row.get("dog_name_key"), row.get("dog_name")):
        stripped = _strip_terminal_history_status_tag(value)
        if stripped != _name_key(value):
            add(stripped, "MATCHED_SUFFIX_STRIPPED_TARGET_NAME")

    return candidates


def _safe_number(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _safe_float(value: Any) -> float | None:
    number = _safe_number(value)
    return float(number) if number is not None else None


def _safe_int(value: Any) -> int | None:
    number = _safe_number(value)
    return int(number) if number is not None else None


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _stddev(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    avg = sum(clean) / len(clean)
    return (sum((value - avg) ** 2 for value in clean) / len(clean)) ** 0.5


def _rate(values: Sequence[int], predicate) -> float | None:
    clean = [int(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(1 for value in clean if predicate(value)) / len(clean)


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _normalize_venue(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")


def _grade_rank(value: Any) -> int | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))
    if text in {"M", "MAIDEN"}:
        return 10
    if text in {"I", "INV", "INVITATIONAL"}:
        return 1
    return None


def _history_time(row: Mapping[str, Any], raw: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
        parsed = _safe_float(raw.get(name))
        if parsed is not None:
            return parsed
    return None


def _history_feature_bundle(
    *,
    target_meta: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_date = _parse_date(target_meta.get("race_date"))
    target_distance = _safe_float(target_meta.get("distance"))
    target_venue = _normalize_venue(target_meta.get("venue"))
    target_grade = str(target_meta.get("grade") or "").strip().upper()
    recent_3 = list(history_rows[:3])
    recent_5 = list(history_rows[:5])
    positions = [_safe_int(row.get("finish_position")) for row in history_rows]
    positions = [value for value in positions if value is not None]
    recent_positions_3 = [_safe_int(row.get("finish_position")) for row in recent_3]
    recent_positions_3 = [value for value in recent_positions_3 if value is not None]
    recent_positions_5 = [_safe_int(row.get("finish_position")) for row in recent_5]
    recent_positions_5 = [value for value in recent_positions_5 if value is not None]
    time_values = [
        value for value in (_safe_float(row.get("individual_time")) for row in history_rows)
        if value is not None
    ]
    recent_time_values = [
        value for value in (_safe_float(row.get("individual_time")) for row in recent_5)
        if value is not None
    ]
    recent_margin_values = [
        value for value in (_safe_float(row.get("margin")) for row in recent_5)
        if value is not None
    ]
    recent_weight_values = [
        value for value in (_safe_float(row.get("weight")) for row in recent_5)
        if value is not None
    ]
    recent_sectional_values = [
        value for value in (_safe_float(row.get("sectional_1st")) for row in recent_5)
        if value is not None
    ]
    same_venue = [
        row for row in history_rows if _normalize_venue(row.get("venue")) == target_venue
    ]
    same_distance = [
        row
        for row in history_rows
        if target_distance is not None
        and _safe_float(row.get("distance")) is not None
        and abs((_safe_float(row.get("distance")) or 0.0) - target_distance) < 0.5
    ]
    same_distance_recent_5 = same_distance[:5]
    same_distance_venue = [
        row for row in same_distance if _normalize_venue(row.get("venue")) == target_venue
    ]
    same_grade = [
        row
        for row in history_rows
        if target_grade and str(row.get("grade") or "").strip().upper() == target_grade
    ]
    same_distance_same_grade = [
        row
        for row in same_distance
        if target_grade and str(row.get("grade") or "").strip().upper() == target_grade
    ]
    last_start_date = _parse_date(history_rows[0].get("race_date")) if history_rows else None
    last_same_distance_date = _parse_date(same_distance[0].get("race_date")) if same_distance else None
    last_weight = _safe_float(history_rows[0].get("weight")) if history_rows else None
    recent_avg_weight = _mean(recent_weight_values)
    last_grade = str(history_rows[0].get("grade") or "").strip().upper() if history_rows else ""
    target_rank = _grade_rank(target_grade)
    last_rank = _grade_rank(last_grade)
    grade_delta = (
        target_rank - last_rank
        if target_rank is not None and last_rank is not None
        else None
    )
    same_distance_times = [
        value for value in (_safe_float(row.get("individual_time")) for row in same_distance)
        if value is not None
    ]
    same_distance_recent_times = [
        value
        for value in (_safe_float(row.get("individual_time")) for row in same_distance_recent_5)
        if value is not None
    ]
    same_venue_times = [
        value for value in (_safe_float(row.get("individual_time")) for row in same_venue)
        if value is not None
    ]
    same_distance_venue_times = [
        value
        for value in (_safe_float(row.get("individual_time")) for row in same_distance_venue)
        if value is not None
    ]
    same_distance_same_grade_times = [
        value
        for value in (_safe_float(row.get("individual_time")) for row in same_distance_same_grade)
        if value is not None
    ]
    sectional_missing = sum(
        1 for row in recent_5 if _safe_float(row.get("sectional_1st")) is None
    )
    recent_grade_counts = Counter(
        str(row.get("grade") or "").strip().upper()
        for row in recent_5
        if str(row.get("grade") or "").strip()
    )
    recent_grade_mode = (
        sorted(recent_grade_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if recent_grade_counts
        else None
    )
    return {
        "prior_start_count": len(history_rows),
        "days_since_last_start": (
            (target_date - last_start_date).days
            if target_date is not None and last_start_date is not None
            else None
        ),
        "recent_finish_mean_3": _mean([float(value) for value in recent_positions_3]),
        "recent_finish_mean_5": _mean([float(value) for value in recent_positions_5]),
        "recent_finish_best_5": min(recent_positions_5) if recent_positions_5 else None,
        "recent_win_rate_5": _rate(recent_positions_5, lambda value: value == 1),
        "recent_place_rate_5": _rate(recent_positions_5, lambda value: value <= 3),
        "recent_avg_margin_5": _mean(recent_margin_values),
        "recent_avg_time_5": _mean(recent_time_values),
        "recent_best_time_5": min(recent_time_values) if recent_time_values else None,
        "recent_time_std_5": _stddev(recent_time_values),
        "career_win_rate": _rate(positions, lambda value: value == 1),
        "career_place_rate": _rate(positions, lambda value: value <= 3),
        "career_avg_finish": _mean([float(value) for value in positions]),
        "career_best_finish": min(positions) if positions else None,
        "career_avg_time": _mean(time_values),
        "career_best_time": min(time_values) if time_values else None,
        "career_time_std": _stddev(time_values),
        "last_start_weight": last_weight,
        "recent_avg_weight_5": recent_avg_weight,
        "weight_delta_last_to_recent": (
            last_weight - recent_avg_weight
            if last_weight is not None and recent_avg_weight is not None
            else None
        ),
        "starts_same_venue": len(same_venue),
        "win_rate_same_venue": _rate(
            [_safe_int(row.get("finish_position")) for row in same_venue],
            lambda value: value == 1,
        ),
        "place_rate_same_venue": _rate(
            [_safe_int(row.get("finish_position")) for row in same_venue],
            lambda value: value <= 3,
        ),
        "best_time_same_venue": min(same_venue_times) if same_venue_times else None,
        "avg_time_same_venue": _mean(same_venue_times),
        "starts_same_distance": len(same_distance) if target_distance is not None else None,
        "prior_same_distance_start_count": (
            len(same_distance) if target_distance is not None else None
        ),
        "best_time_same_distance": min(same_distance_times) if same_distance_times else None,
        "avg_time_same_distance": _mean(same_distance_times),
        "median_time_same_distance": (
            float(median(same_distance_times)) if same_distance_times else None
        ),
        "recent_best_time_same_distance_5": (
            min(same_distance_recent_times) if same_distance_recent_times else None
        ),
        "recent_avg_time_same_distance_5": _mean(same_distance_recent_times),
        "days_since_last_same_distance_start": (
            (target_date - last_same_distance_date).days
            if target_date is not None and last_same_distance_date is not None
            else None
        ),
        "win_rate_same_distance": _rate(
            [_safe_int(row.get("finish_position")) for row in same_distance],
            lambda value: value == 1,
        ),
        "place_rate_same_distance": _rate(
            [_safe_int(row.get("finish_position")) for row in same_distance],
            lambda value: value <= 3,
        ),
        "same_distance_venue_start_count": (
            len(same_distance_venue) if target_distance is not None else None
        ),
        "same_distance_venue_best_time": (
            min(same_distance_venue_times) if same_distance_venue_times else None
        ),
        "same_distance_same_grade_start_count": (
            len(same_distance_same_grade)
            if target_distance is not None and target_grade
            else None
        ),
        "same_grade_start_count": len(same_grade) if target_grade else None,
        "same_grade_win_rate": _rate(
            [_safe_int(row.get("finish_position")) for row in same_grade],
            lambda value: value == 1,
        ),
        "same_grade_place_rate": _rate(
            [_safe_int(row.get("finish_position")) for row in same_grade],
            lambda value: value <= 3,
        ),
        "last_start_grade_normalized": _safe_number(last_grade),
        "recent_grade_mode_5": _safe_number(recent_grade_mode),
        "grade_change_indicator": (
            0 if grade_delta == 0 else 1 if grade_delta is not None else None
        ),
        "grade_change_direction": (
            1 if grade_delta and grade_delta > 0 else -1 if grade_delta and grade_delta < 0 else 0 if grade_delta == 0 else None
        ),
        "grade_strength_delta": grade_delta,
        "last_start_sectional_1st": (
            _safe_float(history_rows[0].get("sectional_1st")) if history_rows else None
        ),
        "sectional_missing_rate_5": (
            sectional_missing / len(recent_5) if recent_5 else None
        ),
        "recent_avg_sectional_1st_5": _mean(recent_sectional_values),
        "recent_best_sectional_1st_5": (
            min(recent_sectional_values) if recent_sectional_values else None
        ),
        "recent_sectional_std_5": _stddev(recent_sectional_values),
        "sectional_time_delta_recent": (
            _safe_float(history_rows[0].get("sectional_1st")) - _mean(recent_sectional_values)
            if history_rows
            and _safe_float(history_rows[0].get("sectional_1st")) is not None
            and _mean(recent_sectional_values) is not None
            else None
        ),
    }


def load_history_feature_index_from_db(
    db_path: Path,
    smoke_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    race_ids = sorted({str(row.get("race_id") or "") for row in smoke_rows if row.get("race_id")})
    if not race_ids:
        return {}, {"history_db_rows_seen": 0, "history_db_rows_used": 0, "target_races_with_history": 0}
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    placeholders = ",".join("?" for _ in race_ids)
    try:
        target_meta = {
            str(row["race_id"]): dict(row)
            for row in conn.execute(
                f"""
                select race_id, race_date, venue, race_number, distance, grade
                from race_metadata
                where race_id in ({placeholders})
                """,
                race_ids,
            )
        }
        history_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        rows_seen = 0
        rows_used = 0
        skipped_no_prior_date = 0
        skipped_not_prior = 0
        for row in conn.execute(
            f"""
            select race_id, dog_name, dog_clean_name, finish_position, weight,
                   individual_time, sectional_1st, margin, raw_row_json
            from csv_dog_history_staging
            where race_id in ({placeholders})
            """,
            race_ids,
        ):
            rows_seen += 1
            mapped = dict(row)
            raw = json.loads(mapped.get("raw_row_json") or "{}")
            target = target_meta.get(str(mapped.get("race_id") or ""), {})
            target_date = _parse_date(target.get("race_date"))
            prior_date = _parse_date(raw.get("DATE"))
            if prior_date is None:
                skipped_no_prior_date += 1
                continue
            if target_date is not None and prior_date >= target_date:
                skipped_not_prior += 1
                continue
            dog_key = _name_key(mapped.get("dog_clean_name") or mapped.get("dog_name"))
            if not dog_key:
                continue
            history_row = {
                "race_date": prior_date.isoformat(),
                "venue": raw.get("TRACK"),
                "distance": _safe_float(raw.get("DIST")),
                "grade": str(raw.get("G") or "").strip().upper(),
                "finish_position": _safe_int(mapped.get("finish_position") or raw.get("PLC")),
                "individual_time": _history_time(mapped, raw, "individual_time", "TIME"),
                "sectional_1st": _history_time(mapped, raw, "sectional_1st", "1 SEC"),
                "margin": _history_time(mapped, raw, "margin", "MGN"),
                "weight": _history_time(mapped, raw, "weight", "WGT"),
            }
            history_by_key.setdefault((str(mapped.get("race_id")), dog_key), []).append(history_row)
            rows_used += 1
        feature_index: dict[tuple[str, str], dict[str, Any]] = {}
        for key, rows in history_by_key.items():
            rows.sort(key=lambda item: _parse_date(item.get("race_date")) or date.min, reverse=True)
            feature_index[key] = _history_feature_bundle(
                target_meta=target_meta.get(key[0], {}),
                history_rows=rows,
            )
    finally:
        conn.close()
    return feature_index, {
        "history_db_path": str(resolved),
        "target_races_requested": len(race_ids),
        "target_races_with_metadata": len(target_meta),
        "target_races_with_history": len({key[0] for key in history_by_key}),
        "history_db_rows_seen": rows_seen,
        "history_db_rows_used": rows_used,
        "history_db_rows_skipped_no_prior_date": skipped_no_prior_date,
        "history_db_rows_skipped_not_prior_to_target": skipped_not_prior,
        "history_db_dog_targets_with_features": len(feature_index),
    }


def _is_dog_form_feature(name: str) -> bool:
    return name.startswith(DOG_FORM_FEATURE_PREFIXES)


def _allowed_feature(name: str) -> bool:
    if name in FORBIDDEN_FEATURE_NAMES:
        return False
    if name.startswith("target_"):
        return False
    if "box" in name:
        return False
    if name in {"venue", "weather", "track_condition", "race_time_minutes_since_midnight"}:
        return False
    return _is_dog_form_feature(name)


def _index_dataset(dataset_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in dataset_rows:
        race_id = str(row.get("race_id") or "")
        dog_key = _name_key(row.get("dog_name") or row.get("dog_key"))
        if race_id and dog_key:
            index.setdefault((race_id, dog_key), row)
    return index


def _selected_feature_names(dataset_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names = {
        str(name)
        for row in dataset_rows
        for name in (row.get("features") or {}).keys()
        if _allowed_feature(str(name))
    }
    return sorted(names)


def _single_feature_top1(
    rows: Sequence[Mapping[str, Any]],
    feature_key: str,
    *,
    higher_better: bool,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("race_id") or ""), []).append(row)
    hits = 0
    scored_races = 0
    for race_rows in grouped.values():
        usable = [row for row in race_rows if _safe_float(row.get(feature_key)) is not None]
        if not usable:
            continue
        scored_races += 1
        if higher_better:
            ordered = sorted(
                race_rows,
                key=lambda row: (
                    _safe_float(row.get(feature_key)) is None,
                    -(_safe_float(row.get(feature_key)) or 0.0),
                    str(row.get("dog_name_key") or ""),
                ),
            )
        else:
            ordered = sorted(
                race_rows,
                key=lambda row: (
                    _safe_float(row.get(feature_key)) is None,
                    _safe_float(row.get(feature_key)) or 0.0,
                    str(row.get("dog_name_key") or ""),
                ),
            )
        hits += int(int(ordered[0].get("actual_win") or 0) == 1)
    return {
        "scored_races": scored_races,
        "top1_hits": hits,
        "top1_accuracy": hits / scored_races if scored_races else None,
    }


def _label_proxy_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feature_specs = {
        "feature_recent_win_rate_5": True,
        "feature_career_win_rate": True,
        "feature_recent_finish_best_5": False,
        "feature_recent_finish_mean_3": False,
    }
    single_feature_results = {
        feature: _single_feature_top1(rows, feature, higher_better=higher_better)
        for feature, higher_better in feature_specs.items()
    }
    risk_features = [
        feature
        for feature, result in single_feature_results.items()
        if (result.get("scored_races") or 0) >= 20
        and (result.get("top1_accuracy") or 0.0) >= 0.9
    ]
    return {
        "status": "POTENTIAL_LABEL_PROXY" if risk_features else "PASS",
        "risk_features": risk_features,
        "single_feature_top1": single_feature_results,
        "risk_rule": "flag_if_any_single_history_feature_top1_accuracy_ge_0_90_over_at_least_20_races",
    }


def build_feature_join_packet(
    *,
    smoke_rows: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    smoke_rows_path: str | None = None,
    dataset_path: str | None = None,
    history_feature_index: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    history_feature_summary: Mapping[str, Any] | None = None,
    history_fill_policy: str = "all",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if history_fill_policy not in HISTORY_FILL_POLICIES:
        raise ValueError(f"unsupported_history_fill_policy:{history_fill_policy}")
    dataset_index = _index_dataset(dataset_rows)
    feature_names = _selected_feature_names(dataset_rows)
    history_feature_index = history_feature_index or {}
    history_feature_summary = history_feature_summary or {}
    joined_rows = []
    failures = []
    match_status_counts: Counter[str] = Counter()
    history_match_status_counts: Counter[str] = Counter()
    non_null_counts: Counter[str] = Counter()
    source_non_null_counts: Counter[str] = Counter()
    history_filled_counts: Counter[str] = Counter()
    history_policy_skipped_counts: Counter[str] = Counter()
    candidate_kind_counts: Counter[str] = Counter()

    for index, row in enumerate(smoke_rows, start=1):
        race_id = str(row.get("race_id") or "")
        dog_key = str(row.get("dog_name_key") or _name_key(row.get("dog_name")))
        source = dataset_index.get((race_id, dog_key))
        history_features = None
        history_feature_match_status = (
            "DISABLED" if not history_feature_index else "NO_HISTORY_DB_ROW"
        )
        history_feature_matched_key = None
        if history_feature_index:
            for history_dog_key, match_status in _history_lookup_key_candidates(row, dog_key):
                history_features = history_feature_index.get((race_id, history_dog_key))
                if history_features is not None:
                    history_feature_match_status = match_status
                    history_feature_matched_key = history_dog_key
                    break
        joined = dict(row)
        joined["schema_version"] = ROWS_SCHEMA_VERSION
        forbidden = sorted(FORBIDDEN_ROW_FIELDS & set(joined))
        if forbidden:
            failures.append(f"smoke_row_{index}_forbidden_fields:{','.join(forbidden)}")
        if source is None:
            joined["feature_join_status"] = "MISSING_DATASET_ROW"
            failures.append(f"smoke_row_{index}_dataset_match_missing:{race_id}:{dog_key}")
        else:
            joined["feature_join_status"] = "MATCHED"
            source_features = source.get("features") or {}
            for name in feature_names:
                value = _safe_number(source_features.get(name))
                joined[f"feature_{name}"] = value
                if value is not None:
                    source_non_null_counts[name] += 1
        history_filled_for_row = 0
        if history_features is None:
            joined["history_feature_join_status"] = history_feature_match_status
        else:
            joined["history_feature_join_status"] = history_feature_match_status
            joined["history_feature_matched_key"] = history_feature_matched_key
            for name in feature_names:
                feature_key = f"feature_{name}"
                if joined.get(feature_key) is not None:
                    continue
                if (
                    history_fill_policy == "no_outcome_proxy_fields"
                    and name in HISTORY_OUTCOME_PROXY_FEATURES
                ):
                    if _safe_number(history_features.get(name)) is not None:
                        history_policy_skipped_counts[name] += 1
                    continue
                value = _safe_number(history_features.get(name))
                if value is None:
                    continue
                joined[feature_key] = value
                history_filled_counts[name] += 1
                history_filled_for_row += 1
            joined["history_feature_prior_start_count"] = _safe_int(
                history_features.get("prior_start_count")
            )
        joined["history_feature_values_filled"] = history_filled_for_row
        for name in feature_names:
            if joined.get(f"feature_{name}") is not None:
                non_null_counts[name] += 1
        joined_rows.append(joined)
        match_status_counts[joined["feature_join_status"]] += 1
        history_match_status_counts[joined["history_feature_join_status"]] += 1
        candidate_kind_counts[str(row.get("candidate_kind") or "UNKNOWN")] += 1

    feature_coverage = {
        name: {
            "non_null_rows": int(non_null_counts.get(name, 0)),
            "row_count": len(smoke_rows),
            "non_null_rate": (
                non_null_counts.get(name, 0) / len(smoke_rows)
                if smoke_rows
                else None
            ),
        }
        for name in feature_names
    }
    forbidden_selected = sorted(name for name in feature_names if not _allowed_feature(name))
    if forbidden_selected:
        failures.append(f"forbidden_features_selected:{','.join(forbidden_selected)}")
    label_proxy_audit = _label_proxy_audit(joined_rows) if history_feature_index else {"status": "NOT_RUN"}
    if failures:
        status = "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_FAILED"
    elif label_proxy_audit.get("status") == "POTENTIAL_LABEL_PROXY":
        status = "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_LEAKAGE_RISK"
    else:
        status = "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"
    packet = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "report_only": True,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_rows_jsonl": smoke_rows_path,
        "expanded_dataset_jsonl": dataset_path,
        "feature_policy": {
            "policy_key": "dog_form_only_no_box_no_race_number_no_calendar",
            "included_feature_prefixes": list(DOG_FORM_FEATURE_PREFIXES),
            "forbidden_feature_names": sorted(FORBIDDEN_FEATURE_NAMES),
            "forbidden_feature_prefixes": ["target_"],
            "excluded_metadata_features": ["venue", "weather", "track_condition", "race_time_minutes_since_midnight"],
        },
        "summary": {
            "smoke_rows_seen": len(smoke_rows),
            "joined_rows": len(joined_rows),
            "match_status_counts": dict(sorted(match_status_counts.items())),
            "history_feature_match_status_counts": dict(
                sorted(history_match_status_counts.items())
            ),
            "candidate_kind_counts": dict(sorted(candidate_kind_counts.items())),
            "feature_column_count": len(feature_names),
            "features_with_non_null_values": sum(1 for count in non_null_counts.values() if count > 0),
            "all_null_feature_count": sum(1 for name in feature_names if non_null_counts.get(name, 0) == 0),
            "feature_coverage": feature_coverage,
            "source_feature_non_null_counts": dict(sorted(source_non_null_counts.items())),
            "history_db_features_enabled": bool(history_feature_index),
            "history_db_fill_policy": history_fill_policy,
            "history_db_outcome_proxy_features_excluded": (
                sorted(HISTORY_OUTCOME_PROXY_FEATURES)
                if history_fill_policy == "no_outcome_proxy_fields"
                else []
            ),
            "history_db_feature_summary": dict(history_feature_summary),
            "history_db_filled_feature_counts": dict(sorted(history_filled_counts.items())),
            "history_db_policy_skipped_feature_counts": dict(
                sorted(history_policy_skipped_counts.items())
            ),
            "history_db_policy_skipped_feature_value_count": sum(
                history_policy_skipped_counts.values()
            ),
            "history_db_filled_feature_value_count": sum(history_filled_counts.values()),
            "history_db_filled_rows": sum(
                1
                for row in joined_rows
                if int(row.get("history_feature_values_filled") or 0) > 0
            ),
            "label_proxy_audit": label_proxy_audit,
            "failures": failures,
            "no_box_features_selected": not any("box" in name for name in feature_names),
            "no_race_number_feature_selected": "race_number" not in feature_names,
            "no_calendar_features_selected": not bool({"target_day_of_week", "target_month"} & set(feature_names)),
        },
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "recommended_next_action": (
            "do_not_use_history_db_enriched_metrics_until_history_source_label_proxy_risk_is_reviewed"
            if label_proxy_audit.get("status") == "POTENTIAL_LABEL_PROXY"
            else
            "run_no_box_actual_win_smoke_eval_on_feature_rows_then_add_report_only_feature_model"
            if not failures
            else "resolve_feature_join_failures_before_model_eval"
        ),
    }
    return packet, joined_rows


def write_outputs(output_dir: Path, packet: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "no_box_actual_win_feature_join_packet.json", packet)
    _write_jsonl(output_dir / "no_box_actual_win_feature_rows.jsonl", rows)
    feature_fields = sorted({key for row in rows for key in row if key.startswith("feature_")})
    with (output_dir / "no_box_actual_win_feature_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*CSV_BASE_FIELDS, *feature_fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in [*CSV_BASE_FIELDS, *feature_fields]})
    summary = packet.get("summary") or {}
    lines = [
        "# No-Box Actual-Win Dog-Form Feature Join",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot or manifest mutations, model training, registry updates, promotions, TGR enablement, betting decisions, or EV actions occurred.",
        "",
        f"- Smoke rows seen: `{summary.get('smoke_rows_seen')}`",
        f"- Joined rows: `{summary.get('joined_rows')}`",
        f"- Feature columns: `{summary.get('feature_column_count')}`",
        f"- Features with non-null values: `{summary.get('features_with_non_null_values')}`",
        f"- History DB features enabled: `{summary.get('history_db_features_enabled')}`",
        f"- History DB fill policy: `{summary.get('history_db_fill_policy')}`",
        f"- Rows filled from history DB: `{summary.get('history_db_filled_rows')}`",
        f"- Feature values filled from history DB: `{summary.get('history_db_filled_feature_value_count')}`",
        f"- Feature values skipped by history policy: `{summary.get('history_db_policy_skipped_feature_value_count')}`",
        f"- Match status counts: `{summary.get('match_status_counts')}`",
        f"- History match status counts: `{summary.get('history_feature_match_status_counts')}`",
        f"- No box features selected: `{summary.get('no_box_features_selected')}`",
        f"- No race-number feature selected: `{summary.get('no_race_number_feature_selected')}`",
        f"- No calendar features selected: `{summary.get('no_calendar_features_selected')}`",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-rows", required=True)
    parser.add_argument("--expanded-dataset", required=True)
    parser.add_argument("--history-db")
    parser.add_argument(
        "--history-fill-policy",
        choices=sorted(HISTORY_FILL_POLICIES),
        default="all",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    smoke_rows_path = Path(args.smoke_rows).expanduser().resolve()
    dataset_path = Path(args.expanded_dataset).expanduser().resolve()
    smoke_rows = _load_jsonl(smoke_rows_path)
    history_feature_index = None
    history_feature_summary = None
    if args.history_db:
        history_feature_index, history_feature_summary = load_history_feature_index_from_db(
            Path(args.history_db),
            smoke_rows,
        )
    packet, rows = build_feature_join_packet(
        smoke_rows=smoke_rows,
        dataset_rows=_load_jsonl(dataset_path),
        smoke_rows_path=str(smoke_rows_path),
        dataset_path=str(dataset_path),
        history_feature_index=history_feature_index,
        history_feature_summary=history_feature_summary,
        history_fill_policy=args.history_fill_policy,
    )
    write_outputs(Path(args.output_dir), packet, rows)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["status"].endswith("_FAILED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
