from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import pytest

from src.predictor import market_form_residual as residual


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/frozen_models/market_form_residual_v1"


def load_fixture():
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text())
    frozen = residual.load_frozen_model(
        ARTIFACT_DIR / "model.json", ARTIFACT_DIR / "manifest.json"
    )
    fixture = manifest["fixed_fixture"]
    return (
        frozen,
        copy.deepcopy(fixture["runners"]),
        copy.deepcopy(fixture["provenance"]),
    )


def canonical_sha256(record):
    return hashlib.sha256(residual._canonical_bytes(record)).hexdigest()


def deterministic_sweep_rows(frozen, base_runners, case_count):
    rng = random.Random(7462026)
    scales = [1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12]
    odds_sets = [
        [1.01, 1.2, 1.5, 2.0, 4.0, 1000.0],
        [2.0] * 6,
        [1.0000000000000002, 1.0000000000000004, 2.0, 3.0, 1e100, 1e308],
        [1.1, 1.1000000000000003, 1.1000000000000005, 10.0, 100.0, 10000.0],
    ]
    for case in range(case_count):
        rows = copy.deepcopy(base_runners)
        scale = scales[case % len(scales)]
        odds = odds_sets[(case // len(scales)) % len(odds_sets)]
        for runner_index, runner in enumerate(rows):
            runner["strict_win_odds"] = odds[runner_index]
            for feature_index, name in enumerate(frozen.feature_order):
                selector = (case + runner_index * 7 + feature_index * 11) % 19
                if selector == 0:
                    runner["features"][name] = None
                elif selector == 1:
                    runner["features"][name] = float(frozen.medians[feature_index])
                else:
                    sign = -1.0 if selector % 2 else 1.0
                    runner["features"][name] = (
                        float(frozen.medians[feature_index])
                        + sign * (runner_index + 1) * (feature_index + 1) * scale
                        + (rng.random() - 0.5) * scale
                    )
        yield rows


def boundary_straddling_fixture(frozen, base_runners):
    return list(deterministic_sweep_rows(frozen, base_runners, 14))[-1]


def test_portable_output_contract_is_bound_and_versioned():
    frozen, runners, provenance = load_fixture()
    record = residual.score_race(frozen, runners, provenance)

    assert residual.SHADOW_RECORD_SCHEMA == "market_form_residual_shadow_record_v3"
    assert residual.EFFECTIVE_STATE_SCHEMA == "market_form_residual_effective_state_v2"
    assert record["schema_version"] == residual.SHADOW_RECORD_SCHEMA
    assert residual._effective_state_payload(frozen)["output_contract"] == {
        "boundary": "ordered_scalar_scoring_before_record_construction",
        "calculation": "python_binary64_scalar_v1",
        "decimal_places": 15,
        "negative_zero": "normalize_to_positive_zero",
        "reductions": "math_fsum",
        "rounding": "decimal_round_half_even",
        "schema_version": "market_form_residual_numeric_canonicalization_v1",
        "transcendentals": "math_log_exp_tanh",
    }


def test_output_contract_is_immutable_and_effective_state_bound(monkeypatch):
    frozen, runners, provenance = load_fixture()

    with pytest.raises(TypeError):
        residual.NUMERICAL_CANONICALIZATION_CONTRACT["decimal_places"] = 16

    tampered = dict(residual.NUMERICAL_CANONICALIZATION_CONTRACT)
    tampered["decimal_places"] = 16
    monkeypatch.setattr(residual, "NUMERICAL_CANONICALIZATION_CONTRACT", tampered)

    with pytest.raises(residual.ResidualContractError, match="effective_state_sha256"):
        residual.score_race(frozen, runners, provenance)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0.0),
        (-0.0, 0.0),
        (1.2345678901234565, 1.234567890123456),
        (1.2345678901234575, 1.234567890123458),
        (-1.2345678901234565, -1.234567890123456),
        (-1.2345678901234575, -1.234567890123458),
        (0.34999999999999994, 0.35),
        (-0.34999999999999994, -0.35),
    ],
)
def test_canonicalization_boundaries(value, expected):
    actual = residual._canonicalize_residual_adjustment(value)
    assert actual == expected
    if actual == 0.0:
        assert math.copysign(1.0, actual) == 1.0


def test_fixed_fixture_portable_record_is_pinned():
    frozen, runners, provenance = load_fixture()
    record = residual.score_race(frozen, runners, provenance)

    assert canonical_sha256(record) == (
        "9568e516579a0c08aa77650fafb8bfd04ccf7986a3cf89e870e528b8abffb032"
    )
    assert record["record_checksum_sha256"] == (
        "20f358f997f3209d19b196ac11e8e9e3893428960389d50939782e21ef780cbc"
    )
    assert record["record_key"] == (
        "07b5630f1657883aebec337855b2730d0827c783350757824750d11546660f4a"
    )
    assert record["effective_state_sha256"] == (
        "f747bf8bf83766365f7a1bb8ddd83f40bc924c6722f1275b57623da38ecff35f"
    )


def test_boundary_straddling_fixture_record_is_pinned():
    frozen, runners, provenance = load_fixture()
    boundary_runners = boundary_straddling_fixture(frozen, runners)

    assert hashlib.sha256(residual._canonical_bytes(boundary_runners)).hexdigest() == (
        "4704c9385ee91a1a7619677d810e2dc1786165f41d30358f2bb6712a02fe248a"
    )
    record = residual.score_race(frozen, boundary_runners, provenance)
    assert canonical_sha256(record) == (
        "0789a5348029abbf65bb7cc0584de1d5e74f8f9dbc9d7f9fb96805e9834a2c86"
    )
    assert record["record_checksum_sha256"] == (
        "2e38780b7e5e15ed688bed92f255af85cc848f0cc1935102b1721816cc76edec"
    )
    assert record["record_key"] == (
        "c9d431151ca98eba91e582293371aa18a133c439372c4fc4a58f2848a27c3566"
    )
    assert record["effective_state_sha256"] == (
        "f747bf8bf83766365f7a1bb8ddd83f40bc924c6722f1275b57623da38ecff35f"
    )


def test_complete_record_is_pinned_across_broad_deterministic_sweep():
    frozen, runners, provenance = load_fixture()
    digest = hashlib.sha256()

    for sweep_runners in deterministic_sweep_rows(frozen, runners, 3000):
        record_bytes = residual._canonical_bytes(
            residual.score_race(frozen, sweep_runners, provenance)
        )
        digest.update(len(record_bytes).to_bytes(8, "big"))
        digest.update(record_bytes)

    assert digest.hexdigest() == (
        "34b92610bc5932f00f69a10985a8368bf7c1d31e96c58dcf1f3c6cd750d2ee06"
    )


@pytest.mark.parametrize("mixed_signs", [False, True])
def test_extreme_finite_features_fail_closed(mixed_signs):
    frozen, runners, provenance = load_fixture()
    for runner_index, runner in enumerate(runners):
        runner["features"] = {
            name: (
                -1e308 if mixed_signs and (runner_index + feature_index) % 2 else 1e308
            )
            for feature_index, name in enumerate(frozen.feature_order)
        }

    with pytest.raises(
        residual.ResidualContractError, match="^scoring_arithmetic_invalid$"
    ):
        residual.score_race(frozen, runners, provenance)


def test_broader_deterministic_fixtures_preserve_semantics():
    frozen, base_runners, provenance = load_fixture()
    fixtures = {}

    odds_curve = copy.deepcopy(base_runners)
    for runner, odds in zip(odds_curve, [1.01, 1.2, 1.5, 2.0, 4.0, 1000.0]):
        runner["strict_win_odds"] = odds
    fixtures["odds_curve"] = odds_curve

    feature_perturb = copy.deepcopy(base_runners)
    for runner_index, runner in enumerate(feature_perturb):
        for feature_index, name in enumerate(frozen.feature_order):
            if runner["features"].get(name) is not None:
                runner["features"][name] = float(runner["features"][name]) + (
                    (runner_index + 1) * (feature_index + 1) * 0.123456789012345
                )
    fixtures["feature_perturb"] = feature_perturb

    zero = copy.deepcopy(base_runners)
    for runner in zero:
        runner["features"] = {
            name: float(frozen.medians[index])
            for index, name in enumerate(frozen.feature_order)
        }
    fixtures["zero"] = zero

    ties = copy.deepcopy(zero)
    for runner in ties:
        runner["strict_win_odds"] = 2.0
    fixtures["ties"] = ties

    near_cap = copy.deepcopy(base_runners)
    for runner_index, runner in enumerate(near_cap):
        sign = -1.0 if runner_index % 2 else 1.0
        runner["features"] = {
            name: sign * (runner_index + 1) * (feature_index + 1) * 1e12
            for feature_index, name in enumerate(frozen.feature_order)
        }
    fixtures["near_cap"] = near_cap

    fixtures["fixed"] = copy.deepcopy(base_runners)
    expected_ranks = {
        "feature_perturb": [0, 5, 2, 3, 4, 1],
        "fixed": [0, 2, 5, 3, 1, 4],
        "near_cap": [0, 2, 5, 4, 3, 1],
        "odds_curve": [0, 1, 2, 3, 4, 5],
        "ties": [0, 1, 2, 3, 4, 5],
        "zero": [0, 2, 5, 3, 1, 4],
    }

    for name, runners in fixtures.items():
        first = residual.score_race(frozen, runners, provenance)
        second = residual.score_race(frozen, list(reversed(runners)), provenance)
        assert first == second
        assert canonical_sha256(first) == canonical_sha256(second)
        predictions = first["predictions"]
        assert [row["runner_id"] for row in predictions] == sorted(
            row["runner_id"] for row in predictions
        )
        for field in ("market_probability", "full_probability", "half_probability"):
            assert sum(row[field] for row in predictions) == pytest.approx(
                1.0, abs=1e-15
            )
        assert all(
            -frozen.residual_cap <= row["residual_adjustment"] <= frozen.residual_cap
            for row in predictions
        )
        if name == "zero":
            assert [row["residual_adjustment"] for row in predictions] == [0.0] * 6
            assert all(
                math.copysign(1.0, row["residual_adjustment"]) == 1.0
                for row in predictions
            )
        if name == "near_cap":
            assert max(abs(row["residual_adjustment"]) for row in predictions) == (
                frozen.residual_cap
            )
        if name == "ties":
            assert len({row["full_probability"] for row in predictions}) == 1
            assert len({row["half_probability"] for row in predictions}) == 1
        ranks = sorted(
            range(len(predictions)),
            key=lambda index: (-predictions[index]["full_probability"], index),
        )
        assert ranks == expected_ranks[name]
