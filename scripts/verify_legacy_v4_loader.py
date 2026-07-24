#!/usr/bin/env python3
"""Prove the copied legacy estimator loads through the checked-in registry path."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (repository / "model_import/legacy_v4_manifest.json").read_text(encoding="utf-8")
    )
    registry_root = repository / ".phase3-artifacts/model_registry"
    os.environ["MPLCONFIGDIR"] = str(repository / ".phase3-artifacts/matplotlib")
    os.environ.pop("V4_MODEL_PATH", None)
    os.environ.pop("TESTING", None)
    os.environ.pop("ML_V4_ALLOW_MOCK_MODEL", None)

    import model_registry
    from model_registry import ModelRegistry

    selected = manifest["model_id"]
    model_path = repository / manifest["artifact"]["destination"]
    for name in ("artifact", "metadata", "index"):
        copied = repository / manifest[name]["destination"]
        assert copied.stat().st_size == manifest[name]["size"]
        assert _sha256(copied) == manifest[name]["sha256"]
    copied_index = json.loads(
        (repository / manifest["index"]["destination"]).read_text(encoding="utf-8")
    )
    selected_entry = copied_index[selected]
    assert selected_entry["model_id"] == selected
    assert selected_entry["is_best"] is True
    assert selected_entry["is_active"] is True
    assert selected_entry["file_hash"] == manifest["artifact"]["sha256"]
    assert selected_entry["features_count"] == 49
    assert len(selected_entry["feature_names"]) == 49
    registry = ModelRegistry(str(registry_root))
    registry.model_index = {selected: dict(registry.model_index[selected])}
    registry.model_index[selected]["model_file_path"] = str(model_path)
    registry.model_index[selected]["scaler_file_path"] = ""
    model_registry._registry_instance = registry

    from ml_system_v4 import MLSystemV4

    system = MLSystemV4.__new__(MLSystemV4)
    system._create_lightweight_mock_model = lambda: (_ for _ in ()).throw(
        AssertionError("mock fallback invoked")
    )
    system.train_model = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retraining invoked")
    )
    system._try_load_latest_model()
    model = system.calibrated_pipeline
    assert type(model).__module__ == "sklearn.calibration"
    assert type(model).__name__ == "CalibratedClassifierCV"
    assert callable(model.predict_proba)
    assert system.model_info["source"] == "model_registry"
    assert system.model_info["model_id"] == selected
    assert len(system.feature_columns) == 49
    print(f"verified {selected} sha256:{manifest['artifact']['sha256']} predict_proba 49 features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
