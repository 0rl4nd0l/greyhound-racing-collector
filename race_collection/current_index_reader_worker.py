"""Private fixed-frame worker for threaded current-index readers."""

from __future__ import annotations

import argparse
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from race_collection.synchronous_manual_capture import (
    _CURRENT_INDEX_READER_MAX_REQUEST_BYTES,
    _CURRENT_INDEX_READER_MAX_RESPONSE_BYTES,
    _CURRENT_INDEX_READER_WIRE_SCHEMA,
    _bounded_current_race_index_main_thread,
    _serialize_current_index_reader_result,
    _socket_receive_frame,
    _socket_send_frame,
    CaptureOneRejected,
)


def _arguments(payload: object) -> Mapping[str, Any]:
    expected = {
        "current_time", "timeout_seconds", "index_path", "evidence_root",
        "max_age_seconds", "max_races", "return_verified_view",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("current index reader arguments are invalid")
    return {
        **payload,
        "current_time": datetime.fromisoformat(str(payload["current_time"])),
        "index_path": Path(str(payload["index_path"])),
        "evidence_root": Path(str(payload["evidence_root"])),
    }


def serve(connection: socket.socket) -> int:
    _socket_send_frame(
        connection,
        {"schema_version": _CURRENT_INDEX_READER_WIRE_SCHEMA, "status": "READY"},
        deadline=None,
        maximum=4096,
    )
    while True:
        try:
            request = _socket_receive_frame(
                connection, deadline=None,
                maximum=_CURRENT_INDEX_READER_MAX_REQUEST_BYTES,
            )
        except EOFError:
            return 0
        request_id = request.get("request_id")
        if (
            request.get("schema_version") != _CURRENT_INDEX_READER_WIRE_SCHEMA
            or not isinstance(request_id, str)
            or len(request_id) != 32
            or set(request) != {"schema_version", "request_id", "arguments"}
        ):
            return 2
        try:
            result = _bounded_current_race_index_main_thread(
                **_arguments(request.get("arguments"))
            )
        except CaptureOneRejected as exc:
            response = {
                "schema_version": _CURRENT_INDEX_READER_WIRE_SCHEMA,
                "request_id": request_id,
                "status": "rejected",
                "code": exc.code,
                "details": exc.details,
            }
        except BaseException:
            response = {
                "schema_version": _CURRENT_INDEX_READER_WIRE_SCHEMA,
                "request_id": request_id,
                "status": "failed",
                "code": "CURRENT_INDEX_INVALID",
                "details": {"reason": "reader_process_failed"},
            }
        else:
            response = {
                "schema_version": _CURRENT_INDEX_READER_WIRE_SCHEMA,
                "request_id": request_id,
                "status": "ok",
                "result": _serialize_current_index_reader_result(result),
            }
        try:
            _socket_send_frame(
                connection, response, deadline=None,
                maximum=_CURRENT_INDEX_READER_MAX_RESPONSE_BYTES,
            )
        except (BrokenPipeError, ConnectionError, OSError):
            return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-fd", type=int, required=True)
    args = parser.parse_args()
    connection = socket.socket(fileno=args.socket_fd)
    try:
        return serve(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
