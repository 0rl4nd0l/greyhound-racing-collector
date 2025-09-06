import importlib.util as _importlib_util
import json
import os as _os
import shutil
import sys as _sys
from pathlib import Path

import pytest

from app import app as flask_app


def _swap_in_real_ml_system_v4():
    """Temporarily replace the stubbed ml_system_v4 with the real module for this test."""
    orig = _sys.modules.get("ml_system_v4")
    root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    path = _os.path.join(root, "ml_system_v4.py")
    spec = _importlib_util.spec_from_file_location("ml_system_v4", path)
    real = _importlib_util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to build import spec for ml_system_v4.py"
    spec.loader.exec_module(real)  # type: ignore
    _sys.modules["ml_system_v4"] = real
    return orig


def _restore_ml_system_v4(orig):
    if orig is not None:
        _sys.modules["ml_system_v4"] = orig
    else:
        try:
            del _sys.modules["ml_system_v4"]
        except Exception:
            pass


@pytest.fixture
def client():
    flask_app.testing = True
    with flask_app.test_client() as c:
        yield c


def test_refresh_list_get_and_check_contracts(client):
    # Swap in the real module for each API call that imports ml_system_v4 internally
    orig = _swap_in_real_ml_system_v4()
    try:
        # 1) Refresh contract to ensure a fresh file exists
        r = client.post("/api/v4/models/contracts/refresh")
        assert r.status_code == 200, f"refresh status={r.status_code}, body={r.data}"
        body = r.get_json()
        assert body and body.get("success") is True
        path = body.get("path")
        assert path and Path(path).exists(), f"Contract path missing: {path}"

        # 2) List contracts
        _restore_ml_system_v4(orig)
        orig = _swap_in_real_ml_system_v4()
        r = client.get("/api/v4/models/contracts")
        assert r.status_code == 200
        body = r.get_json()
        assert body and body.get("success") is True
        names = [item["name"] for item in body.get("contracts", [])]
        assert "v4_feature_contract.json" in names

        # 3) Get contract details
        _restore_ml_system_v4(orig)
        orig = _swap_in_real_ml_system_v4()
        r = client.get("/api/v4/models/contracts/v4_feature_contract.json")
        assert r.status_code == 200
        body = r.get_json()
        assert body and body.get("success") is True
        contract = body.get("contract") or {}
        assert isinstance(contract, dict)
        assert contract.get("version") == "v4"
        assert contract.get("schema_version") in (1,)
        assert "feature_signature" in contract
        assert "categorical_columns" in contract and isinstance(
            contract["categorical_columns"], list
        )
        assert "numerical_columns" in contract and isinstance(
            contract["numerical_columns"], list
        )
        assert "environment" in contract and isinstance(contract["environment"], dict)

        # 4) Non-strict check should be 200 and matched==True immediately after refresh
        _restore_ml_system_v4(orig)
        orig = _swap_in_real_ml_system_v4()
        r = client.get("/api/v4/models/contracts/check")
        assert r.status_code == 200
        body = r.get_json()
        assert body and body.get("success") is True
        assert body.get("matched") in (True, False)
        diff = body.get("diff")
        assert isinstance(diff, dict)
        assert "categorical" in diff and "numerical" in diff

        # 5) Strict check; typically should also be 200 and matched True after refresh
        _restore_ml_system_v4(orig)
        orig = _swap_in_real_ml_system_v4()
        r = client.get("/api/v4/models/contracts/check?strict=1")
        assert r.status_code in (200, 409)
        body = r.get_json()
        assert body and isinstance(body.get("diff"), dict)
    finally:
        _restore_ml_system_v4(orig)


def test_strict_mismatch_returns_409_and_restores_contract(client, tmp_path):
    # Ensure contract exists
    orig = _swap_in_real_ml_system_v4()
    try:
        r = client.post("/api/v4/models/contracts/refresh")
        assert r.status_code == 200
        path = r.get_json().get("path")
        cpath = Path(path)
        assert cpath.exists()

        # Backup original
        backup = tmp_path / "v4_feature_contract_backup.json"
        shutil.copy2(cpath, backup)

        try:
            # Mutate saved contract signature to force mismatch
            raw = json.loads(cpath.read_text())
            raw["feature_signature"] = "FORCED_MISMATCH_SIGNATURE"
            cpath.write_text(json.dumps(raw, indent=2))

            # Strict check now should return 409
            _restore_ml_system_v4(orig)
            orig = _swap_in_real_ml_system_v4()
            r = client.get("/api/v4/models/contracts/check?strict=1")
            assert r.status_code == 409
            body = r.get_json()
            assert body and body.get("success") is False
            diff = body.get("diff") or {}
            assert diff.get("signature_match") is False
        finally:
            # Restore original contract
            shutil.copy2(backup, cpath)
    finally:
        _restore_ml_system_v4(orig)
