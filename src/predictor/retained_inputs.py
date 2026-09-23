"""Explicit retained-input binding for the existing on-demand prediction consumer."""
from __future__ import annotations

import io
import importlib.metadata
import platform
import json
import re
import stat
import zipfile
from datetime import datetime
from pathlib import Path

from race_collection.prospective_input_retention import REQUIRED_ROLES
from src.predictor.on_demand import PredictionBlocked, canonical_bytes, sha256_bytes, write_exact_bytes

MAX_BYTES = 64 * 1024 * 1024


def validate_bindings(value):
    """Finite per-race operator choice, frozen in the generated release binding."""
    if not isinstance(value, dict) or not value or len(value) > 256:
        raise ValueError("invalid retained input bindings")
    for race_id, entry in value.items():
        if (not isinstance(race_id, str) or not race_id or len(race_id) > 200
                or not isinstance(entry, dict) or set(entry) != {"path", "manifest_sha256"}
                or not isinstance(entry["path"], str) or not Path(entry["path"]).is_absolute()
                or not isinstance(entry["manifest_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["manifest_sha256"])):
            raise ValueError("invalid retained input binding")
    return json.loads(json.dumps(value))


def _read(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("retained path invalid")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("retained symlink")
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > MAX_BYTES:
        raise ValueError("retained file invalid")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("retained file oversized")
    return raw


def consume_retained_inputs(*, root, expected_manifest_sha256, bundle, race_id, jump,
                            now, model, config_sha256, ready_receipt, repository_root):
    """Copy verified bytes once; never open the original source/history paths."""
    try:
        root = Path(root)
        manifest_raw = _read(root, "manifest.json")
        if sha256_bytes(manifest_raw) != expected_manifest_sha256:
            raise ValueError("manifest identity")
        manifest = json.loads(manifest_raw)
        completion_raw = _read(root, "completion.json")
        completion = json.loads(completion_raw)
        cutoff = datetime.fromisoformat(manifest["prediction_cutoff"])
        if (manifest["schema_version"] != "prospective_input_retention_v1"
                or manifest["race_id"] != race_id or datetime.fromisoformat(manifest["jump_at"]) != jump
                or not datetime.fromisoformat(manifest["source_observed_at"])
                <= datetime.fromisoformat(manifest["capture_started_at"])
                <= datetime.fromisoformat(manifest["capture_completed_at"])
                <= datetime.fromisoformat(completion["inputs_sealed_at"]) <= now < cutoff < jump
                or completion["status"] != "INPUTS_RETAINED_NOT_QUALIFIED"
                or completion["manifest_sha256"] != expected_manifest_sha256
                or set(manifest["files"]) != REQUIRED_ROLES):
            raise ValueError("retained identity or timing")
        archive = {"bundle/manifest.json": manifest_raw, "bundle/completion.json": completion_raw}
        if manifest["authorization_config_sha256"] is not None:
            raw = _read(root.parent, "terminal.json")
            terminal = json.loads(raw)
            if (terminal["status"] != "RETAINED" or terminal["manifest_sha256"] != expected_manifest_sha256
                    or terminal["config_sha256"] != manifest["authorization_config_sha256"]
                    or not datetime.fromisoformat(completion["inputs_sealed_at"])
                    <= datetime.fromisoformat(terminal["accepted_at"]) <= now < cutoff):
                raise ValueError("retention parent acceptance")
            archive["terminal.json"] = raw
        contents = {}
        for role, entry in {**manifest["files"], "history": manifest["history"], "history_seal": manifest["history_seal"]}.items():
            raw = _read(root, entry["path"])
            if sha256_bytes(raw) != entry["sha256"]:
                raise ValueError("retained file changed")
            contents[role] = raw
            archive["bundle/" + entry["path"]] = raw
            if sum(map(len, archive.values())) > MAX_BYTES:
                raise ValueError("retained archive oversized")
        feature_raw = _read(root, "feature_values.json")
        if sha256_bytes(feature_raw) != manifest["feature_values_sha256"]:
            raise ValueError("retained feature changed")
        archive["bundle/feature_values.json"] = feature_raw
        expected = {"model": model.model_sha256, "model_manifest": model.manifest_sha256,
                    "configuration": config_sha256}
        expected.update({role: sha256_bytes(ready_receipt.handoff[key]) for role, key in (
            ("normalized_form", "_form_bytes"), ("form_metadata", "_sidecar_bytes"), ("odds_report", "_report_bytes"))})
        expected["exact_odds_receipt"] = sha256_bytes(ready_receipt.protocol_members["collector_exact_receipt"])
        if any(sha256_bytes(contents[role]) != digest for role, digest in expected.items()):
            raise ValueError("retained source receipt or model identity")
        # The unchanged generator must have exactly the retained source bytes.
        # No archived code is executed by this adapter.
        lock = json.loads(contents["environment_lock"])
        if lock["python"] != platform.python_version() or any(
                importlib.metadata.version(name) != version for name, version in lock["packages"].items()):
            raise ValueError("retained environment changed")
        from scripts.predict_market_form_residual import FEATURE_GENERATOR_FILES
        required_source = {name for name in FEATURE_GENERATOR_FILES if not name.startswith("tests/")}
        required_source.update({"scripts/__init__.py", "scripts/utils.py", "utils/__init__.py", "config/__init__.py"})
        with zipfile.ZipFile(io.BytesIO(contents["generator_source_archive"])) as source:
            names = source.namelist()
            if len(set(names)) != len(names) or not required_source.issubset(names):
                raise ValueError("generator closure incomplete")
            for name in names:
                if not name.endswith(".py") or Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("generator archive invalid")
                if source.read(name) != _read(repository_root, name):
                    raise ValueError("generator identity")
        if contents["feature_replay_worker"] != _read(repository_root, "scripts/retained_feature_worker.py"):
            raise ValueError("retained worker changed")
        if contents["feature_schema"] != _read(repository_root, "accuracy_program/repaired_non_tgr_schema.json"):
            raise ValueError("schema identity")
        history = json.loads(contents["history_seal"])
        if (history["sealed_sha256"] != sha256_bytes(contents["history"])
                or history["source_sha256"] != manifest["history"]["source_sha256"]
                or history["target_race_id"] != race_id
                or datetime.fromisoformat(history["cutoff_timestamp"]) != jump
                or history["target_rows_materialized"] != 0
                or history["at_or_after_cutoff_rows_materialized"] != 0):
            raise ValueError("retained history identity")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as sealed:
            for name, raw in sorted(archive.items()):
                sealed.writestr(name, raw)
        if len(buffer.getvalue()) > MAX_BYTES:
            raise ValueError("retained archive oversized")
        write_exact_bytes(bundle / "retained_inputs.zip", buffer.getvalue())
        write_exact_bytes(bundle / "features/sealed_history.db", contents["history"])
        write_exact_bytes(bundle / "features/history_seal.json", contents["history_seal"])
        return {"history": history, "features": json.loads(feature_raw), "cutoff": cutoff}
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, importlib.metadata.PackageNotFoundError) as exc:
        raise PredictionBlocked("RETAINED_INPUT_INVALID") from exc


def verify_retained_features(retained, rows, model_path):
    try:
        names = json.loads(model_path.read_bytes())["feature_contract"]["feature_order"]
        projected = [{"race_id": row["race_id"], "dog_name": row["dog_name"],
                      "box_number": row["box_number"], "features": {name: row[name] for name in names}}
                     for row in rows]
        projected.sort(key=lambda row: (row["race_id"], row["box_number"], row["dog_name"]))
        if canonical_bytes(projected) != canonical_bytes(retained["features"]):
            raise ValueError("retained features differ")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PredictionBlocked("RETAINED_INPUT_INVALID") from exc


def verify_retained_prediction_bundle(contents, result, request):
    """Independently join the sealed prediction to its persisted retained choice."""
    try:
        with zipfile.ZipFile(io.BytesIO(contents["retained_inputs.zip"])) as archive:
            names = archive.namelist()
            if len(names) > 32 or len(names) != len(set(names)) or sum(info.file_size for info in archive.infolist()) > MAX_BYTES:
                raise ValueError("retained archive bounds")
            raw = {name: archive.read(name) for name in names}
        manifest_raw = raw["bundle/manifest.json"]
        if sha256_bytes(manifest_raw) != request["retained_input_manifest_sha256"]:
            raise ValueError("retained manifest binding")
        manifest = json.loads(manifest_raw)
        completion = json.loads(raw["bundle/completion.json"])
        cutoff = datetime.fromisoformat(manifest["prediction_cutoff"])
        sealed_at = datetime.fromisoformat(completion["inputs_sealed_at"])
        generated = datetime.fromisoformat(result["generated_at"])
        if (completion["manifest_sha256"] != request["retained_input_manifest_sha256"]
                or manifest["race_id"] != result["race"]["race_id"]
                or datetime.fromisoformat(manifest["jump_at"]) != datetime.fromisoformat(result["race"]["jump_timestamp"])
                or not sealed_at <= generated < cutoff < datetime.fromisoformat(manifest["jump_at"])):
            raise ValueError("retained prediction identity")
        if manifest["authorization_config_sha256"] is not None:
            terminal = json.loads(raw["terminal.json"])
            if (terminal["status"] != "RETAINED" or terminal["manifest_sha256"] != request["retained_input_manifest_sha256"]
                    or terminal["config_sha256"] != manifest["authorization_config_sha256"]
                    or not sealed_at <= datetime.fromisoformat(terminal["accepted_at"]) <= generated):
                raise ValueError("retained parent acceptance")
        retained = {}
        for role, entry in {**manifest["files"], "history": manifest["history"], "history_seal": manifest["history_seal"]}.items():
            value = raw["bundle/" + entry["path"]]
            if sha256_bytes(value) != entry["sha256"]:
                raise ValueError("retained member changed")
            retained[role] = value
        for role, member in {"history":"features/sealed_history.db", "history_seal":"features/history_seal.json",
                "model":"model/model.json", "model_manifest":"model/manifest.json", "configuration":"config.json",
                "exact_odds_receipt":"protocol/collector_exact_receipt.json", "odds_report":"source/capture.json"}.items():
            if retained[role] != contents[member]:
                raise ValueError("retained prediction member differs")
        for role in ("normalized_form", "form_metadata"):
            if retained[role] != contents["source/" + Path(manifest["files"][role]["path"]).name]:
                raise ValueError("retained prediction source differs")
        features_raw = raw["bundle/feature_values.json"]
        if sha256_bytes(features_raw) != manifest["feature_values_sha256"]:
            raise ValueError("retained features changed")
        names = json.loads(retained["model"])["feature_contract"]["feature_order"]
        rows = json.loads(contents["features/sealed/shadow_feature_rows.json"])
        projected = [{"race_id": row["race_id"], "dog_name": row["dog_name"], "box_number": row["box_number"],
                      "features": {name: row[name] for name in names}} for row in rows]
        projected.sort(key=lambda row: (row["race_id"], row["box_number"], row["dog_name"]))
        if canonical_bytes(projected) != features_raw:
            raise ValueError("retained prediction features differ")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise PredictionBlocked("RETAINED_INPUT_INVALID") from exc
