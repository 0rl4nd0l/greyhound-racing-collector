#!/usr/bin/env python3
"""Run one adapter-fed forward-sealed corpus iteration or report status."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_corpus_module = importlib.import_module("race_collection.forward_sealed_corpus")
ForwardCorpusRejected = _corpus_module.ForwardCorpusRejected
ForwardSealedCorpus = _corpus_module.ForwardSealedCorpus
canonical_json = _corpus_module.canonical_json
_lock_module = importlib.import_module("scripts.shadow_autopilot_daemon")
LockBusy = _lock_module.LockBusy
acquire_lock = _lock_module.acquire_lock
release_lock = _lock_module.release_lock

_PREJUMP_KEYS = {
    "action",
    "race_id",
    "racing_date",
    "raw_source_path",
    "sealed_evidence_path",
    "feature_schema_path",
    "missingness_policy_path",
    "source_name",
    "canonical_source_url",
    "source_native_race_id",
    "runners",
    "meeting_metadata",
    "race_metadata",
    "source_observed_at",
    "feature_frozen_at",
    "scheduled_jump_at",
}
_RESULT_KEYS = {
    "action",
    "race_id",
    "raw_result_path",
    "collector_id",
    "session_id",
    "run_id",
    "request_id",
    "source_name",
    "request_url",
    "final_url",
    "http_status",
    "content_type",
    "source_document_last_modified",
}
_CLOSE_KEYS = {"action", "race_id"}


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ForwardCorpusRejected("iteration file is not valid JSON") from error
    if type(value) is not dict:
        raise ForwardCorpusRejected("iteration must be one JSON object")
    return value


def _input_bytes(iteration_path: Path, value: Any, name: str) -> bytes:
    if type(value) is not str or not value.strip():
        raise ForwardCorpusRejected(f"{name} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = iteration_path.parent / path
    try:
        return path.read_bytes()
    except OSError as error:
        raise ForwardCorpusRejected(f"{name} bytes are unavailable") from error


def _strict(value: Mapping[str, Any], expected: set[str], action: str) -> None:
    if set(value) != expected:
        raise ForwardCorpusRejected(f"{action} iteration has unknown or missing keys")


def run_iteration(
    corpus: ForwardSealedCorpus,
    iteration_path: Path,
) -> tuple[dict[str, Any], int]:
    value = _object(iteration_path)
    action = value.get("action")
    if action == "prejump":
        _strict(value, _PREJUMP_KEYS, action)
        receipt = corpus.capture_prejump(
            race_id=value["race_id"],
            racing_date=value["racing_date"],
            raw_source_bytes=_input_bytes(iteration_path, value["raw_source_path"], "raw source"),
            sealed_evidence_bytes=_input_bytes(
                iteration_path, value["sealed_evidence_path"], "sealed evidence"
            ),
            feature_schema_bytes=_input_bytes(
                iteration_path, value["feature_schema_path"], "feature schema"
            ),
            missingness_policy_bytes=_input_bytes(
                iteration_path,
                value["missingness_policy_path"],
                "missingness policy",
            ),
            source_name=value["source_name"],
            canonical_source_url=value["canonical_source_url"],
            source_native_race_id=value["source_native_race_id"],
            runners=value["runners"],
            meeting_metadata=value["meeting_metadata"],
            race_metadata=value["race_metadata"],
            source_observed_at=value["source_observed_at"],
            feature_frozen_at=value["feature_frozen_at"],
            scheduled_jump_at=value["scheduled_jump_at"],
        )
        return {
            "action": action,
            "decision": "PREJUMP_CAPTURED",
            "receipt": receipt,
            "status": corpus.status(),
        }, 0
    if action == "result":
        _strict(value, _RESULT_KEYS, action)
        raw_result = _input_bytes(iteration_path, value["raw_result_path"], "raw result")

        def transport(_request_url: str) -> Mapping[str, Any]:
            return {
                "body": raw_result,
                "status_code": value["http_status"],
                "content_type": value["content_type"],
                "final_url": value["final_url"],
                "source_document_last_modified": value["source_document_last_modified"],
            }

        receipt = corpus.capture_result(
            race_id=value["race_id"],
            collector_id=value["collector_id"],
            session_id=value["session_id"],
            run_id=value["run_id"],
            request_id=value["request_id"],
            source_name=value["source_name"],
            request_url=value["request_url"],
            transport=transport,
        )
        status = corpus.status()
        race_state = next(
            row["state"] for row in status["races"] if row["race_id"] == value["race_id"]
        )
        return {
            "action": action,
            "decision": race_state,
            "receipt": receipt,
            "status": status,
        }, 2 if race_state == "RESULT_CHANGED_BEFORE_CLOSURE" else 0
    if action == "close":
        _strict(value, _CLOSE_KEYS, action)
        receipt = corpus.close(race_id=value["race_id"])
        package = corpus.build_package()
        return {
            "action": action,
            "decision": "EXAMPLE_CLOSED",
            "receipt": receipt,
            "package_checksum": str(package.package_checksum),
            "manifest_checksum": str(package.manifest_checksum),
            "status": corpus.status(),
        }, 0
    raise ForwardCorpusRejected("iteration action must be prejump, result, or close")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--iteration", type=Path)
    mode.add_argument("--status", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    corpus = ForwardSealedCorpus(args.root)
    if args.status:
        print(canonical_json(corpus.status()).decode())
        return 0

    run_id = "forward_sealed_corpus_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    lock_path = corpus.root / "forward-sealed-corpus.lock"
    lock_payload = None
    try:
        lock_payload = acquire_lock(
            lock_path=lock_path,
            run_id=run_id,
            stale_after_seconds=3600,
            output_dir=corpus.root,
        )
        result, exit_code = run_iteration(corpus, args.iteration)
        print(canonical_json(result).decode())
        return exit_code
    except LockBusy as error:
        print(
            canonical_json(
                {
                    "decision": "LOCK_BUSY",
                    "details": error.payload,
                }
            ).decode(),
            file=sys.stderr,
        )
        return 3
    except ForwardCorpusRejected as error:
        print(
            canonical_json(
                {
                    "decision": "REJECTED",
                    "error": str(error),
                }
            ).decode(),
            file=sys.stderr,
        )
        return 1
    finally:
        if lock_payload is not None:
            release_lock(lock_path, run_id)


if __name__ == "__main__":
    raise SystemExit(main())
