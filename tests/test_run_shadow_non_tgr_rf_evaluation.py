import json
import shutil
from pathlib import Path

import pytest

from scripts.run_shadow_non_tgr_rf_evaluation import (
    ALLOWED_OUTPUT_PREFIXES,
    POWER_GAMMA,
    apply_power_gamma_by_race,
    assert_shadow_output_dir_safe,
    main,
    probability_sum_report,
    ranking_preservation_report,
    validate_schema_contract,
)


def _schema(features):
    return {
        "schema_version": "test_schema",
        "feature_columns": features,
        "categorical_features": [],
        "numeric_or_boolean_features": features,
    }


def test_default_repaired_schema_has_78_features_and_no_tgr():
    schema_path = Path(
        "outputs/milestone_6a_non_tgr_challenger_training_design/repaired_non_tgr_schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    audit = validate_schema_contract(schema)

    assert audit["status"] == "PASS"
    assert audit["feature_count"] == 78
    assert audit["tgr_columns"] == []


def test_schema_rejects_tgr_identity_and_post_outcome_features():
    features = [f"feature_{index}" for index in range(75)]
    features.extend(["tgr_speed", "race_id", "finish_position"])

    audit = validate_schema_contract(_schema(features))

    assert audit["status"] == "FAIL"
    assert audit["tgr_columns"] == ["tgr_speed"]
    assert audit["identity_columns_present_as_features"] == ["race_id"]
    assert audit["post_outcome_columns_present_as_features"] == ["finish_position"]


def test_power_gamma_2p4_normalizes_and_preserves_ranking():
    rows = [
        {
            "shadow_race_group_id": "race-a",
            "race_id": "Race A",
            "dog_name": "Alpha",
            "box_number": 1,
            "shadow_rf_uncalibrated_probability": 0.60,
        },
        {
            "shadow_race_group_id": "race-a",
            "race_id": "Race A",
            "dog_name": "Bravo",
            "box_number": 2,
            "shadow_rf_uncalibrated_probability": 0.30,
        },
        {
            "shadow_race_group_id": "race-a",
            "race_id": "Race A",
            "dog_name": "Charlie",
            "box_number": 3,
            "shadow_rf_uncalibrated_probability": 0.10,
        },
        {
            "shadow_race_group_id": "race-b",
            "race_id": "Race B",
            "dog_name": "Delta",
            "box_number": 1,
            "shadow_rf_uncalibrated_probability": 0.55,
        },
        {
            "shadow_race_group_id": "race-b",
            "race_id": "Race B",
            "dog_name": "Echo",
            "box_number": 2,
            "shadow_rf_uncalibrated_probability": 0.45,
        },
    ]

    calibrated = apply_power_gamma_by_race(
        rows,
        gamma=POWER_GAMMA,
        input_key="shadow_rf_uncalibrated_probability",
        output_key="shadow_rf_calibrated_probability",
        output_rank_key="shadow_rf_calibrated_rank",
    )

    sums = probability_sum_report(calibrated, "shadow_rf_calibrated_probability")
    ranking = ranking_preservation_report(
        rows,
        calibrated,
        before_key="shadow_rf_uncalibrated_probability",
        after_key="shadow_rf_calibrated_probability",
    )
    assert sums["status"] == "PASS"
    assert sums["max_abs_error"] == pytest.approx(0.0)
    assert ranking["status"] == "PASS"
    assert [row["shadow_rf_calibrated_rank"] for row in calibrated[:3]] == [1, 2, 3]


def test_shadow_output_path_rejects_production_paths(tmp_path):
    assert_shadow_output_dir_safe(
        tmp_path / "artifacts" / "shadow_evaluation" / "run",
        root=tmp_path,
    )
    assert_shadow_output_dir_safe(
        tmp_path
        / "artifacts"
        / "full_evidence_orchestration_20260525"
        / "shadow_evaluation_implementation_test",
        root=tmp_path,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_shadow_artifact"):
        assert_shadow_output_dir_safe(tmp_path / "predictions" / "shadow", root=tmp_path)
    with pytest.raises(ValueError, match="output_dir_must_be_shadow_artifact"):
        assert_shadow_output_dir_safe(tmp_path / "model_registry" / "shadow", root=tmp_path)


def test_cli_stop_after_definition_writes_only_shadow_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv("APPROVE_RESULT_LABEL_WRITE", raising=False)
    schema = _schema([f"feature_{index}" for index in range(78)])
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    output_dir = Path("artifacts/shadow_evaluation") / f"pytest_candidate_{tmp_path.name}"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        rc = main(
            [
                "run",
                "--schema",
                str(schema_path),
                "--output-dir",
                str(output_dir),
                "--stop-after-definition",
            ]
        )

        assert rc == 0
        assert (output_dir / "shadow_candidate_definition.json").exists()
        assert not (output_dir / "shadow_predictions.csv").exists()
        assert not any(
            "bet" in path.name.lower() or "ev" == path.stem.lower()
            for path in output_dir.rglob("*")
        )
        definition = json.loads((output_dir / "shadow_candidate_definition.json").read_text())
        assert definition["registry_mutation"] is False
        assert definition["promotion_allowed"] is False
        assert definition["tgr_enabled"] is False
        assert definition["output_mode"] == "shadow_only"
        assert definition["calibration"]["method_key"] == "power_gamma_2.4"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_allowed_output_prefixes_are_shadow_only():
    assert ALLOWED_OUTPUT_PREFIXES == (
        "artifacts/shadow_evaluation",
        "artifacts/full_evidence_orchestration_20260525/shadow_evaluation_",
    )
