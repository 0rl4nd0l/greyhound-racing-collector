import importlib.util
from pathlib import Path

import pytest

from accuracy_program.odds_provenance import classify_odds_snapshot_for_ev
from accuracy_program.snapshots import (
    classify_odds_snapshot_for_ev as classify_snapshot_odds_for_ev,
)


SNAPSHOT_RACE_ID = "Race 4 - WRGL - 2026-05-21"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_real_prediction_pipeline_v4():
    spec = importlib.util.spec_from_file_location(
        "_real_prediction_pipeline_v4_for_odds_provenance_tests",
        REPO_ROOT / "prediction_pipeline_v4.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _runner(**overrides):
    runner = {
        "dog_name": "Alpha Runner",
        "box_number": 1,
    }
    runner.update(overrides)
    return runner


def _odds_snapshot(**overrides):
    snapshot = {
        "market_odds_win": 3.0,
        "market_type": "win",
        "odds_level": "dog",
        "odds_timestamp": "2026-05-21T15:44:00",
        "odds_captured_before_prediction": True,
        "odds_captured_before_feature_freeze": True,
        "odds_captured_before_jump": True,
        "odds_stale_at_prediction": False,
        "odds_provenance": {
            "source": "SportsBet",
            "source_url": "https://www.sportsbet.com.au/greyhound-racing/race-4",
            "source_table": "live_odds",
            "odds_race_id": SNAPSHOT_RACE_ID,
            "odds_dog_name": "Alpha Runner",
            "odds_box_number": 1,
            "match_method": "race_id_box_name",
            "match_confidence": 1.0,
        },
    }
    provenance_overrides = overrides.pop("odds_provenance", None)
    snapshot.update(overrides)
    if provenance_overrides is not None:
        snapshot["odds_provenance"] = {
            **snapshot.get("odds_provenance", {}),
            **provenance_overrides,
        }
    return snapshot


@pytest.mark.parametrize(
    ("expected_status", "snapshot_overrides", "runner_overrides"),
    [
        ("valid_pre_jump_dog_odds", {}, {}),
        ("no_odds_row", {"market_odds_win": None}, {}),
        ("post_race_or_sp_only", {"market_type": "sp"}, {}),
        ("race_level_only_odds", {"market_type": "place"}, {}),
        ("race_level_only_odds", {"odds_level": "race"}, {}),
        ("missing_timestamp", {"odds_timestamp": None}, {}),
        ("timestamp_after_prediction", {"odds_captured_before_prediction": False}, {}),
        (
            "timestamp_after_feature_freeze",
            {"odds_captured_before_feature_freeze": False},
            {},
        ),
        ("timestamp_after_jump", {"odds_captured_before_jump": False}, {}),
        ("stale_beyond_ttl", {"odds_stale_at_prediction": True}, {}),
        ("untrusted_source", {"odds_provenance": {"source": "unknownbook"}}, {}),
        ("missing_source_url", {"odds_provenance": {"source_url": ""}}, {}),
        (
            "post_race_or_sp_only",
            {"odds_provenance": {"source_table": "race_results"}},
            {},
        ),
        (
            "post_race_or_sp_only",
            {
                "odds_provenance": {
                    "source_url": "https://sportsbet.com.au/greyhound-racing/results/race-4"
                }
            },
            {},
        ),
        (
            "race_id_mismatch",
            {"odds_provenance": {"odds_race_id": "WRGL_2026-05-22_4"}},
            {},
        ),
        (
            "ambiguous_box_source",
            {"odds_provenance": {"sportsbet_box_source": "list_position_fallback"}},
            {},
        ),
        ("box_mismatch", {"odds_provenance": {"odds_box_number": 2}}, {}),
        ("dog_name_mismatch", {"odds_provenance": {"odds_dog_name": "Beta Runner"}}, {}),
        ("duplicate_odds_rows", {"odds_provenance": {"duplicate_count": 2}}, {}),
        ("duplicate_odds_rows", {"odds_provenance": {"candidate_count": 2}}, {}),
        (
            "ambiguous_runner_identity",
            {
                "odds_provenance": {
                    "odds_box_number": None,
                    "match_method": "race_id_name",
                }
            },
            {},
        ),
        (
            "ambiguous_runner_identity",
            {"odds_provenance": {"match_confidence": 0.5}},
            {},
        ),
    ],
)
def test_odds_provenance_classifier_status_matrix(
    expected_status,
    snapshot_overrides,
    runner_overrides,
):
    decision = classify_odds_snapshot_for_ev(
        _runner(**runner_overrides),
        _odds_snapshot(**snapshot_overrides),
        snapshot_race_id=SNAPSHOT_RACE_ID,
    )

    assert decision["odds_match_status"] == expected_status
    assert decision["is_ev_eligible"] is (expected_status == "valid_pre_jump_dog_odds")
    if expected_status == "valid_pre_jump_dog_odds":
        assert decision["odds_exclusion_reason"] is None
        assert decision["odds_provenance_status"] == "complete"
    else:
        assert decision["odds_exclusion_reason"] == expected_status
        assert decision["odds_provenance_status"] == "excluded"


def test_odds_provenance_classifier_accepts_canonical_race_id_equivalence():
    decision = classify_odds_snapshot_for_ev(
        _runner(),
        _odds_snapshot(odds_provenance={"odds_race_id": "WRGL_2026-05-21_4"}),
        snapshot_race_id=SNAPSHOT_RACE_ID,
    )

    assert decision["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert decision["odds_match_method"] == "canonical_race_id_box_dog"
    assert decision["is_ev_eligible"] is True


def test_snapshot_compatibility_export_uses_central_odds_provenance():
    central = classify_odds_snapshot_for_ev(
        _runner(),
        _odds_snapshot(odds_provenance={"source_url": ""}),
        snapshot_race_id=SNAPSHOT_RACE_ID,
    )
    compatibility = classify_snapshot_odds_for_ev(
        _runner(),
        _odds_snapshot(odds_provenance={"source_url": ""}),
        snapshot_race_id=SNAPSHOT_RACE_ID,
    )

    assert compatibility["odds_match_status"] == "missing_source_url"
    assert compatibility == central


def test_live_market_context_clears_raw_ev_when_strict_provenance_fails():
    pipeline = _load_real_prediction_pipeline_v4()

    predictions = [
        {
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "win_prob_norm": 0.4,
            "ev_win": 0.2,
            "ev_win_positive": True,
        }
    ]
    market_odds = {"Alpha Runner": 3.0}
    market_records = {
        "Alpha Runner": {
            "id": 10,
            "race_id": SNAPSHOT_RACE_ID,
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "odds_decimal": 3.0,
            "market_type": "win",
            "source": "sportsbet",
            "timestamp": "2026-05-21T15:44:00",
            "source_url": "",
            "odds_level": "dog",
        }
    }

    pipeline._annotate_market_context(
        predictions,
        market_odds,
        market_records,
        prediction_timestamp="2026-05-21T15:45:00",
        feature_freeze_timestamp="2026-05-21T15:45:00",
        jump_datetime="2026-05-21T15:58:00",
        snapshot_race_id=SNAPSHOT_RACE_ID,
    )

    runner = predictions[0]
    assert runner["odds_match_status"] == "missing_source_url"
    assert runner["odds_provenance_status"] == "excluded"
    assert runner["ev_win"] is None
    assert runner["ev_win_positive"] is False
    assert "invalid_pre_jump_odds" in runner["quality_flags"]


def test_live_market_context_preserves_raw_ev_when_strict_provenance_passes():
    pipeline = _load_real_prediction_pipeline_v4()

    predictions = [
        {
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "win_prob_norm": 0.4,
            "ev_win": 0.2,
            "ev_win_positive": True,
        }
    ]
    market_odds = {"Alpha Runner": 3.0}
    market_records = {
        "Alpha Runner": {
            "id": 10,
            "race_id": SNAPSHOT_RACE_ID,
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "odds_decimal": 3.0,
            "market_type": "win",
            "source": "sportsbet",
            "timestamp": "2026-05-21T15:44:00",
            "source_url": "https://www.sportsbet.com.au/greyhound-racing/race-4",
            "odds_level": "dog",
        }
    }

    pipeline._annotate_market_context(
        predictions,
        market_odds,
        market_records,
        prediction_timestamp="2026-05-21T15:45:00",
        feature_freeze_timestamp="2026-05-21T15:45:00",
        jump_datetime="2026-05-21T15:58:00",
        snapshot_race_id=SNAPSHOT_RACE_ID,
    )

    runner = predictions[0]
    assert runner["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert runner["odds_provenance_status"] == "complete"
    assert runner["ev_win"] == pytest.approx(0.2)
    assert runner["ev_win_positive"] is True


def test_live_market_context_fails_closed_when_price_has_no_provenance_record():
    pipeline = _load_real_prediction_pipeline_v4()

    predictions = [
        {
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "win_prob_norm": 0.4,
            "ev_win": 0.2,
            "ev_win_positive": True,
        }
    ]
    market_odds = {"Alpha Runner": 3.0}

    pipeline._annotate_market_context(
        predictions,
        market_odds,
        None,
        prediction_timestamp="2026-05-21T15:45:00",
        feature_freeze_timestamp="2026-05-21T15:45:00",
        jump_datetime="2026-05-21T15:58:00",
        snapshot_race_id=SNAPSHOT_RACE_ID,
    )

    runner = predictions[0]
    assert runner["odds_match_status"] == "missing_timestamp"
    assert runner["odds_provenance_status"] == "excluded"
    assert runner["ev_win"] is None
    assert runner["ev_win_positive"] is False
    assert "invalid_pre_jump_odds" in runner["quality_flags"]
