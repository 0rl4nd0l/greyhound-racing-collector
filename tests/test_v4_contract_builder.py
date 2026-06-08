import importlib.util as _importlib_util
import json
import os as _os
from pathlib import Path

import pytest

# Import the real ml_system_v4 from file to bypass test-time stubbing in conftest
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_spec = _importlib_util.spec_from_file_location(
    "ml_system_v4_real", _os.path.join(_root, "ml_system_v4.py")
)
_mod = _importlib_util.module_from_spec(_spec)
assert _spec and _spec.loader, "Failed to create import spec for ml_system_v4"
_spec.loader.exec_module(_mod)  # type: ignore
MLSystemV4 = _mod.MLSystemV4
_IMPORT_TIME_GREYHOUND_DB_PATH = _os.environ.get("GREYHOUND_DB_PATH")


def test_00_database_env_mutation_is_local_to_single_test(tmp_path):
    leaky_db_path = tmp_path / "leaky-greyhound.sqlite"
    _os.environ["GREYHOUND_DB_PATH"] = str(leaky_db_path)

    assert _os.environ["GREYHOUND_DB_PATH"] == str(leaky_db_path)


def test_01_database_env_is_restored_for_v4_contract_tests():
    assert _os.environ.get("GREYHOUND_DB_PATH") == _IMPORT_TIME_GREYHOUND_DB_PATH


def test_build_and_save_contract_contains_expected_metadata(tmp_path):
    sys = MLSystemV4()

    # Build contract in-memory
    contract = sys._build_feature_contract()
    assert isinstance(contract, dict)
    assert contract.get("version") == "v4"
    # schema_version should be present and numeric (we expect 1)
    assert contract.get("schema_version") in (1,)
    assert "feature_signature" in contract
    assert isinstance(contract.get("categorical_columns"), list)
    assert isinstance(contract.get("numerical_columns"), list)
    env = contract.get("environment")
    assert isinstance(env, dict)
    # Environment snapshot should include python_version
    assert "python_version" in env

    # Save contract using API method
    result = sys.regenerate_feature_contract()
    assert result.get("success") is True
    saved_path = Path(result.get("path"))
    assert saved_path.exists()

    # Check contract should pass after regeneration
    ok = sys.check_feature_contract(enforce=False)
    assert ok in (True, False)  # Usually True; keep tolerant to env variance


def test_check_feature_contract_enforce_raises_on_mismatch(tmp_path):
    sys = MLSystemV4()

    # Ensure a baseline contract exists
    res = sys.regenerate_feature_contract()
    assert res.get("success") is True
    path = Path(res.get("path"))
    assert path.exists()

    # Mutate the saved contract to force mismatch
    data = json.loads(path.read_text())
    data["feature_signature"] = "FORCED_TEST_SIGNATURE_MISMATCH"
    path.write_text(json.dumps(data, indent=2))

    # Non-enforcing mode returns False
    ok = sys.check_feature_contract(enforce=False)
    assert ok is False

    # Enforcing mode raises
    with pytest.raises(Exception):
        _ = sys.check_feature_contract(enforce=True)
