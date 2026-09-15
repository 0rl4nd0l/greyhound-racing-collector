"""Bounded, read-only joins against collector-owned official-result evidence.

This adapter never acquires results or opens a live WAL as an immutable snapshot.
It returns source rows with their content identity, not interim model metrics.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import time
from pathlib import Path
from datetime import datetime

from utils.csv_metadata import canonical_thedogs_race_identity, canonical_thedogs_venue_identity
from .job_store import canonical, utc_text


class OfficialResultSource:
    def __init__(self, database: Path):
        self.database = Path(database).absolute()

    def read(self, job, bundle, *, now):
        """Read only this completed prediction's race; reject ambiguous evidence."""
        jump = datetime.fromisoformat(job.input.jump_timestamp.replace("Z", "+00:00"))
        if now <= jump:
            return {"state": "RESULT_PENDING", "reason": "RACE_NOT_OFF"}
        path = self.database
        try:
            if path.resolve() != path or not stat.S_ISREG(path.stat().st_mode):
                return {"state": "RESULT_PENDING", "reason": "RESULT_SOURCE_UNSAFE"}
            before = path.stat()
            sidecars = tuple(Path(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal"))
            if any(p.exists() or p.is_symlink() for p in sidecars):
                return {"state": "RESULT_PENDING", "reason": "RESULT_SOURCE_BUSY"}
            deadline = time.monotonic() + 1.0
            db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1.0)
            try:
                db.execute("PRAGMA query_only=ON")
                db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                rows = []
                for table in (
                    "autonomous_official_result_evidence_races",
                    "autonomous_official_result_evidence_runners",
                ):
                    selected = db.execute(
                        f"SELECT row_json FROM {table} WHERE race_id=? LIMIT 33",
                        (job.input.race_id,),
                    ).fetchall()
                    if any(
                        not isinstance(row[0], str) or len(row[0].encode()) > 65536
                        for row in selected
                    ):
                        return {"state": "RESULT_REJECTED", "reason": "RESULT_ROWS_OVERSIZED"}
                    rows.append([json.loads(row[0]) for row in selected])
            finally:
                db.close()
            after = path.stat()
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(before) != identity(after) or any(
                p.exists() or p.is_symlink() for p in sidecars
            ):
                return {"state": "RESULT_PENDING", "reason": "RESULT_SOURCE_CHANGED"}
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return {"state": "RESULT_PENDING", "reason": "RESULT_SOURCE_UNAVAILABLE"}
        race_rows, runner_rows = rows
        if not race_rows or not runner_rows:
            return {"state": "RESULT_PENDING", "reason": "OFFICIAL_RESULT_UNAVAILABLE"}
        try:
            self._validate(job, bundle, race_rows, runner_rows, now)
        except (KeyError, TypeError, ValueError) as exc:
            return {"state": "RESULT_REJECTED", "reason": str(exc)}
        evidence = {
            "race_rows": race_rows,
            "runner_rows": sorted(runner_rows, key=lambda r: r["box_number"]),
        }
        return {
            "state": "RESULT_AVAILABLE",
            "evidence": evidence,
            "evidence_sha256": hashlib.sha256(canonical(evidence)).hexdigest(),
        }

    @staticmethod
    def _validate(job, bundle, race_rows, runner_rows, now):
        if len(race_rows) != 1 or len(runner_rows) != len(job.input.ordered_runners):
            raise ValueError("OFFICIAL_RESULT_MISSING_OR_AMBIGUOUS")
        race, expected = race_rows[0], bundle.result["race"]
        if race["source"] != "thedogs_official" or race["status"] != "resulted":
            raise ValueError("OFFICIAL_RESULT_NOT_TERMINAL")
        url = race["source_url"]
        if (
            url not in {expected["url"], expected["url"] + "?trial=false"}
            or canonical_thedogs_race_identity(url) is None
        ):
            raise ValueError("OFFICIAL_RESULT_URL_MISMATCH")
        if (race["race_id"], race["race_date"], race["race_number"]) != (
            job.input.race_id,
            expected["race_date"],
            expected["race_number"],
        ):
            raise ValueError("OFFICIAL_RESULT_RACE_MISMATCH")
        if canonical_thedogs_venue_identity(race["venue"]) != canonical_thedogs_venue_identity(
            expected["venue"]
        ):
            raise ValueError("OFFICIAL_RESULT_VENUE_MISMATCH")

        def parse(value):
            if not isinstance(value, str):
                raise ValueError("OFFICIAL_RESULT_TIMESTAMP_INVALID")
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            utc_text(parsed)
            return parsed

        jump = parse(job.input.jump_timestamp)
        captured = parse(race["captured_at"])
        utc_text(captured)
        if (
            parse(race["start_datetime"]) != jump
            or not max(jump, parse(bundle.result["generated_at"])) < captured <= now
        ):
            raise ValueError("OFFICIAL_RESULT_TIMESTAMP_MISMATCH")
        expected_rows = {r["box"]: r for r in job.input.ordered_runners}
        boxes, finishes = set(), set()
        for row in runner_rows:
            for key in (
                "race_id",
                "race_date",
                "race_number",
                "venue",
                "source",
                "source_url",
                "captured_at",
            ):
                if row[key] != race[key]:
                    raise ValueError("OFFICIAL_RESULT_RUNNER_PROVENANCE_MISMATCH")
            box, finish = row["box_number"], row["finish_position"]
            if (
                type(box) is not int
                or box in boxes
                or box not in expected_rows
                or type(finish) is not int
                or finish in finishes
            ):
                raise ValueError("OFFICIAL_RESULT_FINISH_AMBIGUOUS")
            if row["dog_name"] != expected_rows[box]["name"]:
                raise ValueError("OFFICIAL_RESULT_RUNNER_IDENTITY_MISMATCH")
            if row.get("source_native_runner_id") is not None and row[
                "source_native_runner_id"
            ] != expected_rows[box].get("source_native_runner_id"):
                raise ValueError("OFFICIAL_RESULT_NATIVE_ID_MISMATCH")
            if type(row["is_winner"]) is not bool or row["is_winner"] != (finish == 1):
                raise ValueError("OFFICIAL_RESULT_WINNER_MISMATCH")
            if finish == 1 and (race["winner_box"], race["winner_name"]) != (box, row["dog_name"]):
                raise ValueError("OFFICIAL_RESULT_WINNER_MISMATCH")
            boxes.add(box)
            finishes.add(finish)
        if (
            boxes != set(expected_rows)
            or finishes != set(range(1, len(expected_rows) + 1))
            or race["position_count"] != len(expected_rows)
            or race["participant_count"] != len(expected_rows)
            or race["box_order"]
            != [
                row["box_number"] for row in sorted(runner_rows, key=lambda r: r["finish_position"])
            ]
        ):
            raise ValueError("OFFICIAL_RESULT_FINISH_AMBIGUOUS")
