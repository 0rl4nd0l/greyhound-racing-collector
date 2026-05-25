"""Runner-set completeness checks for live form-guide evidence.

The expert-form CSV format stores the target race field as the rows whose
``Dog Name`` value starts with a box number, while following rows contain that
dog's historical starts. These helpers deliberately count only target runners.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


MIN_COMPLETE_RUNNERS = 4
PRIMARY_RUNNER_RE = re.compile(r"^\s*(\d{1,2})\s*[\.\):-]\s*(.+?)\s*$")


def normalise_runner_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def clean_runner_name(raw: Any) -> str:
    name = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(raw or "").strip())
    name = name.replace('"', "").replace("'", "").replace("`", "")
    return re.sub(r"\s+", " ", name).strip().title()


@dataclass(frozen=True)
class RunnerRow:
    box_number: int
    dog_name: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunnerCompleteness:
    schema_version: str
    status: str
    source: str | None
    runner_count: int
    min_complete_runners: int
    boxes: list[int]
    dog_names: list[str]
    participants: list[dict[str, Any]]
    duplicate_boxes: list[int]
    duplicate_dog_names: list[str]
    invalid_runner_rows: int
    reasons: list[str]

    @property
    def is_complete(self) -> bool:
        return self.status == "COMPLETE"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _header_value(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    lookup = {str(key).strip().lower(): key for key in row.keys()}
    for name in names:
        key = lookup.get(name.lower())
        if key is not None:
            return row.get(key)
    return None


def parse_runner_rows_from_text(content: str) -> tuple[list[RunnerRow], int]:
    """Return target-race runner rows and invalid target-like row count."""

    reader = csv.DictReader(io.StringIO(content))
    prefixed: list[RunnerRow] = []
    fallback: list[RunnerRow] = []
    invalid = 0

    for row in reader:
        raw_name = str(_header_value(row, ("Dog Name", "dog_name", "runner", "name")) or "").strip()
        if not raw_name or raw_name == '""':
            continue

        match = PRIMARY_RUNNER_RE.match(raw_name)
        if match:
            prefixed.append(
                RunnerRow(
                    box_number=int(match.group(1)),
                    dog_name=clean_runner_name(raw_name),
                )
            )
            continue

        box_value = _header_value(row, ("Box", "BOX", "box_number", "box"))
        if box_value not in (None, ""):
            try:
                fallback.append(
                    RunnerRow(
                        box_number=int(str(box_value).strip()),
                        dog_name=clean_runner_name(raw_name),
                    )
                )
            except (TypeError, ValueError):
                invalid += 1

    return (prefixed or fallback), invalid


def parse_runner_rows_from_csv(path: str | Path) -> list[RunnerRow]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    rows, _invalid = parse_runner_rows_from_text(text)
    return rows


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    dupes: set[Any] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return sorted(dupes)


def analyze_runner_rows(
    rows: Iterable[RunnerRow],
    *,
    source: str | None = None,
    invalid_runner_rows: int = 0,
    min_complete_runners: int = MIN_COMPLETE_RUNNERS,
) -> RunnerCompleteness:
    unique: dict[tuple[int, str], RunnerRow] = {}
    for row in rows:
        unique.setdefault((row.box_number, normalise_runner_name(row.dog_name)), row)

    deduped = list(unique.values())
    boxes = [row.box_number for row in deduped]
    dog_names = [row.dog_name for row in deduped]
    duplicate_boxes = _duplicates(boxes)
    duplicate_dog_names = [
        name
        for name in _duplicates(normalise_runner_name(name) for name in dog_names)
        if name
    ]

    reasons: list[str] = []
    if not deduped:
        reasons.append("no_target_runner_rows")
    if len(deduped) < min_complete_runners:
        reasons.append(f"runner_count_below_min:{len(deduped)}<{min_complete_runners}")
    if duplicate_boxes:
        reasons.append(
            "duplicate_box_numbers:" + ",".join(str(value) for value in duplicate_boxes)
        )
    if duplicate_dog_names:
        reasons.append("duplicate_dog_names:" + ",".join(duplicate_dog_names))
    if invalid_runner_rows:
        reasons.append(f"invalid_runner_rows:{invalid_runner_rows}")

    return RunnerCompleteness(
        schema_version="runner_completeness_v1",
        status="COMPLETE" if not reasons else "INCOMPLETE",
        source=source,
        runner_count=len(deduped),
        min_complete_runners=min_complete_runners,
        boxes=sorted(boxes),
        dog_names=dog_names,
        participants=[row.as_dict() for row in deduped],
        duplicate_boxes=duplicate_boxes,
        duplicate_dog_names=duplicate_dog_names,
        invalid_runner_rows=invalid_runner_rows,
        reasons=reasons,
    )


def analyze_csv_runner_completeness(
    path: str | Path,
    *,
    min_complete_runners: int = MIN_COMPLETE_RUNNERS,
) -> RunnerCompleteness:
    path_obj = Path(path)
    try:
        text = path_obj.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as exc:
        return RunnerCompleteness(
            schema_version="runner_completeness_v1",
            status="INCOMPLETE",
            source=str(path_obj),
            runner_count=0,
            min_complete_runners=min_complete_runners,
            boxes=[],
            dog_names=[],
            participants=[],
            duplicate_boxes=[],
            duplicate_dog_names=[],
            invalid_runner_rows=0,
            reasons=[f"csv_unreadable:{type(exc).__name__}"],
        )
    return analyze_csv_text_runner_completeness(
        text,
        source=str(path_obj),
        min_complete_runners=min_complete_runners,
    )


def analyze_csv_text_runner_completeness(
    content: str,
    *,
    source: str | None = None,
    min_complete_runners: int = MIN_COMPLETE_RUNNERS,
) -> RunnerCompleteness:
    rows, invalid = parse_runner_rows_from_text(content)
    return analyze_runner_rows(
        rows,
        source=source,
        invalid_runner_rows=invalid,
        min_complete_runners=min_complete_runners,
    )


def participants_from_runner_rows(rows: Iterable[RunnerRow]) -> list[dict[str, Any]]:
    participants: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (row.box_number, normalise_runner_name(row.dog_name))
        if key in seen:
            continue
        seen.add(key)
        participants.append({"box_number": row.box_number, "dog_name": row.dog_name})
    return participants


def analyze_prediction_runner_match(
    predictions: Iterable[Mapping[str, Any]],
    source_report: Mapping[str, Any] | RunnerCompleteness | None,
) -> dict[str, Any]:
    if isinstance(source_report, RunnerCompleteness):
        source_data = source_report.as_dict()
    else:
        source_data = dict(source_report or {})

    prediction_boxes: list[int] = []
    for row in predictions:
        box = row.get("box_number")
        try:
            if box is not None:
                prediction_boxes.append(int(box))
        except (TypeError, ValueError):
            continue

    source_boxes = [int(box) for box in source_data.get("boxes") or []]
    reasons: list[str] = []
    if source_boxes:
        missing = sorted(set(source_boxes) - set(prediction_boxes))
        extra = sorted(set(prediction_boxes) - set(source_boxes))
        if missing:
            reasons.append("prediction_missing_source_boxes:" + ",".join(map(str, missing)))
        if extra:
            reasons.append("prediction_extra_boxes:" + ",".join(map(str, extra)))
        if len(prediction_boxes) != len(source_boxes):
            reasons.append(
                f"prediction_runner_count_mismatch:{len(prediction_boxes)}!={len(source_boxes)}"
            )

    duplicate_prediction_boxes = _duplicates(prediction_boxes)
    if duplicate_prediction_boxes:
        reasons.append(
            "duplicate_prediction_boxes:"
            + ",".join(str(value) for value in duplicate_prediction_boxes)
        )

    return {
        "schema_version": "prediction_runner_match_v1",
        "status": "MATCHED" if not reasons else "MISMATCH",
        "source_runner_count": len(source_boxes) if source_boxes else None,
        "prediction_runner_count": len(prediction_boxes),
        "source_boxes": sorted(source_boxes),
        "prediction_boxes": sorted(prediction_boxes),
        "reasons": reasons,
    }


def quarantine_existing_file(path: str | Path, *, reason: str) -> Path:
    path_obj = Path(path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir = path_obj.parent / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{timestamp}_{path_obj.stem}_{reason}{path_obj.suffix}"
    shutil.move(str(path_obj), str(target))
    return target


def quarantine_csv_content(
    content: str,
    destination_dir: str | Path,
    filename: str,
    *,
    reason: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir = Path(destination_dir) / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{timestamp}_{Path(filename).stem}_{reason}.csv"
    target.write_text(content, encoding="utf-8", newline="")
    return target
