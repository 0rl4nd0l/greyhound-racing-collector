"""Outcome-blind admission prerequisite, not a result fetcher or reservation.

Only the current index run's collector-owned shadow precursor is supported.
Snapshot-only and older/fallback runs stay pending: do not scan the research
corpus, inspect result tables, or manufacture evidence to gain admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
from datetime import datetime
from pathlib import Path

from utils.csv_metadata import canonical_thedogs_race_identity
from utils.race_identity_equivalence import race_identity_equivalent
from utils.runner_completeness import analyze_csv_text_runner_completeness, normalise_runner_name
from .r3_api import R3Rejected


class ResultAcquisitionReadiness:
    def __init__(self, evidence_root, *, authority, races):
        self.root = Path(evidence_root).absolute()
        self.authority, self.races = authority, races

    def require(self, job_input, *, now):
        """Fail before allocation unless retained collector precursors match.

        Reads at most three source files (2 MiB each) and the pinned service
        definition. No directory enumeration, database access, or acquisition.
        """
        from scripts import autonomous_official_result_capture as collector
        from .live_adapters import _unit

        retained = []
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

        def read(path, *, source=True):
            path = Path(path).absolute()
            if source:
                path.relative_to(self.root)
            if path.resolve(strict=True) != path:
                raise ValueError("indirect source")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_size > 2 * 1024 * 1024:
                    raise ValueError("unsafe source")
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    raw = handle.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024 or identity(os.fstat(descriptor)) != identity(before):
                    raise ValueError("changing source")
            finally:
                os.close(descriptor)
            retained.append((path, before))
            return raw

        try:
            unit = self.authority["units"]["full_service"]
            raw = read(unit["path"], source=False)
            if hashlib.sha256(raw).hexdigest() != unit["sha256"]:
                raise ValueError("collector deployment changed")
            service = _unit(raw, "Service")
            working = self.authority["working_directory"]
            if (
                service.get("WorkingDirectory") != [working]
                or len(service.get("ExecStart", [])) != 1
            ):
                raise ValueError("collector binding unavailable")
            command = shlex.split(service["ExecStart"][0])
            script = str(Path(working) / "scripts/shadow_autopilot_daemon.py")
            if (
                len(command) < 3
                or command[1:3] != [script, "run-once"]
                or command.count("--enable-autonomous-result-capture") != 1
                or command.count("--evidence-root") != 1
                or command[command.index("--evidence-root") + 1] != str(self.root)
            ):
                raise ValueError("result acquisition not configured")
            run_id = job_input.operational_index_provenance.run_id
            if not re.fullmatch(r"[A-Za-z0-9_+-]{1,128}", run_id):
                raise ValueError("unsafe run identity")
            directory = self.root / f"daily_race_ingest_shadow_{run_id}_daemon_autopilot"
            # Intentionally require these direct files. Do not follow manifests
            # or the collector's historical fallback paths during admission.
            predictions = [
                json.loads(line)
                for line in read(directory / "stage2_shadow_predictions.jsonl").splitlines()
                if line.strip()
            ]
            features = json.loads(read(directory / "shadow_feature_rows.json"))
            if (
                not isinstance(features, list)
                or len(predictions) > 1024
                or len(features) > 1024
                or any(not isinstance(row, dict) for row in (*predictions, *features))
            ):
                raise ValueError("invalid shadow precursor")
            collector.ingest.assert_no_result_fields(predictions)
            collector.ingest.assert_no_result_fields(features)
            races = tuple(self.races())
            if len(races) > 64:
                raise ValueError("current race unavailable")
            matches = [r for r in races if r.get("race_id") == job_input.race_id]
            if len(matches) != 1:
                raise ValueError("current race unavailable")
            race = matches[0]
            url = canonical_thedogs_race_identity(race["race_url"])
            rows = [
                r
                for r in predictions
                if race_identity_equivalent(
                    job_input.race_id,
                    r.get("race_id"),
                    source_url=url["canonical_url"],
                )
            ]
            feature_rows = [
                r
                for r in features
                if race_identity_equivalent(
                    job_input.race_id,
                    r.get("race_id"),
                    source_url=url["canonical_url"],
                )
            ]
            if not rows or not feature_rows:
                raise ValueError("race not collector-owned")
            if len({r.get("race_id") for r in rows}) != 1 or len(
                {r.get("race_id") for r in feature_rows}
            ) != 1:
                raise ValueError("ambiguous race identity")
            first = feature_rows[0]
            source = Path(first["source_csv"])
            if not source.is_absolute():
                # This is the collector resolver's first preference. Ambiguous
                # checkout-relative fallbacks are deliberately not readiness.
                source = directory / source
            csv_report = analyze_csv_text_runner_completeness(read(source).decode("utf-8-sig"))
            report = collector.analyze_runner_rows(collector._runner_rows_from_predictions(rows))
            expected = {
                (r["box"], normalise_runner_name(r["name"])) for r in job_input.ordered_runners
            }
            participants = lambda values: {
                (r["box_number"], normalise_runner_name(r["dog_name"])) for r in values
            }
            if (
                report.status != "COMPLETE"
                or csv_report.status != "COMPLETE"
                or participants(report.participants) != expected
                or participants(csv_report.participants) != expected
                or participants(feature_rows) != expected
                or len(feature_rows) != len(expected)
            ):
                raise ValueError("collector participant mismatch")
            if url is None or any(
                canonical_thedogs_race_identity(r.get("target_metadata_source_url")) != url
                for r in feature_rows
            ):
                raise ValueError("collector URL mismatch")
            race_identity = collector.parse_race_identity(job_input.race_id)
            if (race_identity["race_date"], race_identity["race_number"]) != (
                url["race_date"],
                url["race_number"],
            ):
                raise ValueError("collector race mismatch")
            minutes = first["race_time_minutes_since_midnight"]
            if any(
                r.get("race_time_minutes_since_midnight") != minutes
                or r.get("source_csv") != first["source_csv"]
                for r in feature_rows
            ):
                raise ValueError("ambiguous source identity")
            jump = datetime.fromisoformat(job_input.jump_timestamp.replace("Z", "+00:00"))
            lifecycle = collector.ingest.classify_race_record(
                {
                    **race_identity,
                    "race_id": job_input.race_id,
                    "race_time": collector.race_time_from_minutes(minutes),
                },
                now=now,
                source_context="shadow_run_prediction",
            )
            if (
                datetime.fromisoformat(lifecycle.jump_datetime) != jump
                or datetime.fromisoformat(race["jump_datetime"].replace("Z", "+00:00")) != jump
                or participants(race["runners"]) != expected
            ):
                raise ValueError("collector jump or index mismatch")
            native = {r["box"]: r.get("source_native_runner_id") for r in job_input.ordered_runners}
            if (
                race.get("runner_set_sha256") != job_input.runner_set_sha256
                or {r["box_number"]: r.get("source_native_runner_id") for r in race["runners"]}
                != native
            ):
                raise ValueError("current native runner identity changed")
            for path, before in retained:
                if path.resolve(strict=True) != path or identity(path.stat()) != identity(before):
                    raise ValueError("source changed during admission")
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise R3Rejected("RESULT_ACQUISITION_NOT_READY") from exc
