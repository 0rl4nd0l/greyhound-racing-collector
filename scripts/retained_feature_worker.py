"""Private feature worker: retained source ZIP + inputs, never a live scorer."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

PHASE = "INPUT_SETUP"


def main() -> None:
    global PHASE
    root = Path(sys.argv[1]).resolve()
    request = json.loads(sys.stdin.buffer.read())

    def path(role):
        value = (root / request[role]).resolve()
        if root not in value.parents:
            raise ValueError("input path")
        return value

    PHASE = "ENVIRONMENT_CHECK"
    lock = json.loads(path("environment_lock").read_bytes())
    if lock["python"] != platform.python_version():
        raise ValueError("python drift")
    for package, version in lock["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("dependency drift")
    archive = path("generator_source_archive")
    # Isolated (-I) interpreter: remove all ordinary repository locations.
    sys.path[:] = [str(archive)] + [p for p in sys.path if p and (
        Path(p).is_relative_to(sys.base_prefix) or Path(p).is_relative_to(sys.prefix)
    )]
    sys.dont_write_bytecode = True
    # Load dependencies before restricting data I/O; generator source itself is
    # supplied by the approved archive. Nothing is allowed to contact a network.
    def no_network(event, args):
        if event.startswith("socket.") or event in {"subprocess.Popen", "os.system"}:
            raise PermissionError("offline feature worker")
    sys.addaudithook(no_network)
    PHASE = "GENERATOR_IMPORT"
    from scripts.run_shadow_non_tgr_rf_evaluation import build_live_feature_rows
    import encodings.utf_8_sig  # prewarm before the exact data-read boundary
    from zoneinfo import ZoneInfo
    from utils.prejump_weather import VENUE_WEATHER_LOCATIONS
    # Keep strong references: the feature validator may use any supported venue.
    timezones = [ZoneInfo(name) for name in sorted({v.timezone for v in VENUE_WEATHER_LOCATIONS.values()} | {"Australia/Melbourne"})]

    form = path("normalized_form")
    if path("form_metadata") != Path(str(form) + ".metadata.json"):
        raise ValueError("sidecar adjacency")
    schema = json.loads(path("feature_schema").read_bytes())
    names = json.loads(path("model").read_bytes())["feature_contract"]["feature_order"]
    allowed = {form, path("form_metadata"), root / "history.db"}

    def data_boundary(event, args):
        if event == "open":
            if isinstance(args[0], int):
                raise PermissionError("descriptor read")
            candidate = Path(os.fsdecode(args[0])).resolve()
            if candidate not in allowed:
                raise PermissionError("unretained input")
            mode = args[1]
            if isinstance(mode, str) and any(c in mode for c in "wa+"):
                raise PermissionError("write forbidden")
        if event == "sqlite3.connect":
            if str(args[0]) != f"file:{root / 'history.db'}?mode=ro":
                raise PermissionError("unretained database")
    sys.addaudithook(data_boundary)
    PHASE = "FEATURE_GENERATION"
    rows = build_live_feature_rows(input_paths=[form], schema=schema, db_path=root / "history.db")
    # Only path-dependent provenance is omitted; every model input and its null
    # value is retained, keyed by the exact generated race/runner identity.
    PHASE = "FEATURE_PROJECTION"
    projected = [{"race_id": r["race_id"], "dog_name": r["dog_name"],
                  "box_number": r["box_number"],
                  "features": {name: r[name] for name in names}} for r in rows]
    projected.sort(key=lambda r: (r["race_id"], r["box_number"], r["dog_name"]))
    sys.stdout.write(json.dumps(projected, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never print payloads or source exception details, including to stderr.
        sys.stdout.write(json.dumps({"failure_phase": PHASE}))
        raise SystemExit(2)
