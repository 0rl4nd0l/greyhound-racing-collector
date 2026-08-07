from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.predictor.market_form_residual import FEATURES
from src.predictor.scoring_parity import (
    SCORING_CORE_OUTPUT_SCHEMA,
    SCORING_INPUT_SCHEMA,
    ScoringParityRejected,
    build_core_output,
    build_scoring_input,
    compare_parity,
    first_difference_diagnostic,
)

HASHES = {
    "runner_set": "a" * 64,
    "feature": "b" * 64,
    "odds": "c" * 64,
    "model": "d" * 64,
    "manifest": "e" * 64,
    "effective": "f" * 64,
}
TIMESTAMPS = {
    "cutoff_timestamp": "2026-08-06T10:00:00+10:00",
    "capture_timestamp": "2026-08-06T10:01:00+10:00",
    "score_timestamp": "2026-08-06T10:02:00+10:00",
    "jump_timestamp": "2026-08-06T10:05:00+10:00",
}
RACE_ID = "Race 3 - TEST - 2026-08-06"


def _runner(box: int, name: str, odds: float, *, feature_values=None) -> dict:
    values = dict(
        zip(
            FEATURES,
            feature_values
            or (1, None, 2.0, 1.0, 0.5, 0.75, None, 0.2, 0.4, 3.0, 5, 0.2, 4, 0.25, 2, 0.5),
        )
    )
    token = "".join(character for character in name.upper() if character.isalnum())
    return {
        "runner_id": f"{RACE_ID}|box:{box}|dog:{token}",
        "box_number": box,
        "dog_name": name,
        "features": values,
        "feature_source_sha256": HASHES["feature"],
        "strict_win_odds": odds,
        "odds_source_sha256": HASHES["odds"],
        "feature_freeze_timestamp": TIMESTAMPS["cutoff_timestamp"],
        "odds_capture_timestamp": TIMESTAMPS["capture_timestamp"],
    }


def _input(runners=None, **changes):
    rows = runners or [
        _runner(1, "Alpha Dog", 2.5),
        _runner(2, "Beta Dog", 2.5),
        _runner(4, "Gamma Dog", 5.0),
    ]
    ids = [row["runner_id"] for row in rows]
    values = {
        "race_id": RACE_ID,
        "runner_set_sha256": hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest(),
        "runners": rows,
        **TIMESTAMPS,
        "model_sha256": HASHES["model"],
        "manifest_sha256": HASHES["manifest"],
        "effective_state_sha256": HASHES["effective"],
    }
    values.update(changes)
    return build_scoring_input(**values)


def _record(input_artifact, *, probabilities=(0.4, 0.4, 0.2)) -> dict:
    identities = input_artifact.document["identities"]
    return {
        "race_id": input_artifact.document["race_id"],
        "runner_set_sha256": input_artifact.document["runner_set_sha256"],
        "model_sha256": identities["model_sha256"],
        "manifest_sha256": identities["manifest_sha256"],
        "effective_state_sha256": identities["effective_state_sha256"],
        "outcomes_present": False,
        "predictions": [
            {
                "runner_id": row["runner_id"],
                "market_probability": probabilities[index],
                "residual_adjustment": 0.0,
                "full_probability": probabilities[index],
                "half_probability": probabilities[index],
            }
            for index, row in enumerate(input_artifact.document["runners"])
        ],
    }


def test_identical_lane_inputs_are_byte_identical_and_rank_box_ties():
    automatic = _input()
    manual = _input(runners=deepcopy(automatic.document["runners"]))
    automatic_core = build_core_output(automatic, _record(automatic))
    manual_core = build_core_output(manual, _record(manual))

    assert automatic.raw == manual.raw
    assert automatic.sha256 == manual.sha256
    assert automatic_core.raw == manual_core.raw
    assert automatic_core.sha256 == manual_core.sha256
    assert [row["runner_id"] for row in automatic_core.document["ranks"][:2]] == [
        automatic.document["runners"][0]["runner_id"],
        automatic.document["runners"][1]["runner_id"],
    ]
    assert compare_parity(automatic, manual, automatic_core, manual_core) == {
        "input_hash_equal": True,
        "core_output_hash_equal": True,
        "input_first_difference": {"equal": True},
        "core_output_first_difference": {"equal": True},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0]["features"].__setitem__(FEATURES[0], 9.0),
        lambda rows: rows[0].__setitem__("strict_win_odds", 3.75),
        lambda rows: rows[0].__setitem__("runner_id", "wrong"),
        lambda rows: rows.reverse(),
        lambda rows: rows[0]["features"].__setitem__(FEATURES[1], 4.0),
    ],
)
def test_input_mutations_never_silently_reconcile(mutation):
    original = _input()
    rows = deepcopy(original.document["runners"])
    mutation(rows)
    try:
        changed = _input(runners=rows)
    except ScoringParityRejected:
        return
    assert changed.sha256 != original.sha256


def test_missing_extra_or_reordered_features_fail_closed():
    original = _input()
    for mutation in (
        lambda row: row["features"].pop(FEATURES[0]),
        lambda row: row["features"].__setitem__("unexpected", 1.0),
        lambda row: row.__setitem__("features", dict(reversed(list(row["features"].items())))),
    ):
        rows = deepcopy(original.document["runners"])
        mutation(rows[0])
        with pytest.raises(ScoringParityRejected):
            _input(runners=rows)


def test_timestamp_identity_and_outcome_mutations_fail_closed():
    original = _input()
    with pytest.raises(ScoringParityRejected):
        _input(score_timestamp="2026-08-06T10:06:00+10:00")
    changed_config = _input(config_sha256="0" * 64)
    assert changed_config.sha256 != original.sha256
    assert _input(cutoff_timestamp="2026-08-06T10:00:30+10:00").sha256 != original.sha256
    assert _input(model_sha256="1" * 64).sha256 != original.sha256
    rows = deepcopy(original.document["runners"])
    rows[0]["outcome"] = "must reject"
    with pytest.raises(ScoringParityRejected):
        _input(runners=rows)


def test_first_difference_is_deterministic():
    assert first_difference_diagnostic(b"abc", b"abX") == {
        "equal": False,
        "offset": 2,
        "expected_length": 3,
        "actual_length": 3,
        "expected_byte": 99,
        "actual_byte": 88,
        "expected_context": "abc",
        "actual_context": "abX",
    }
    assert first_difference_diagnostic(b"abc", b"abcde")["offset"] == 3


def test_contract_schema_versions_are_versioned():
    artifact = _input()
    core = build_core_output(artifact, _record(artifact))
    assert artifact.document["schema_version"] == SCORING_INPUT_SCHEMA
    assert core.document["schema_version"] == SCORING_CORE_OUTPUT_SCHEMA


def test_versioned_input_and_core_schemas_validate():
    root = Path(__file__).resolve().parents[1] / "configs/prediction/market-form-residual-v1"
    input_schema = json.loads((root / "scoring-input.schema.json").read_bytes())
    core_schema = json.loads((root / "scoring-core-output.schema.json").read_bytes())
    artifact = _input()
    core = build_core_output(artifact, _record(artifact))
    Draft202012Validator(input_schema).validate(artifact.document)
    Draft202012Validator(core_schema).validate(core.document)


def test_contract_module_has_no_persistence_or_live_access_surface():
    import src.predictor.scoring_parity as module

    assert not {"sqlite3", "requests", "httpx", "subprocess", "playwright"} & set(module.__dict__)
    assert not any(
        key in module.__dict__
        for key in ("phase7", "promotion", "ev", "stake", "bet", "outcome", "result")
    )
