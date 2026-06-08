import copy
import importlib.util
import json
import sqlite3
from pathlib import Path

import pandas as pd

from enhanced_accuracy_optimizer import AdvancedEnsemblePredictor
from src.parsers.csv_ingestion import CsvIngestion
from utils.csv_metadata import parse_race_csv_meta


REPO_ROOT = Path(__file__).resolve().parents[1]
RACE11_CSV = REPO_ROOT / "upcoming_races" / "Race 11 - WRGL - 2026-05-21.csv"
RACE11_PREDICTION = (
    REPO_ROOT / "predictions" / "Race 11 - WRGL - 2026-05-21_20260521T183444.json"
)
DB_PATH = REPO_ROOT / "greyhound_racing_data.db"


def _load_real_prediction_pipeline_v4():
    path = REPO_ROOT / "prediction_pipeline_v4.py"
    spec = importlib.util.spec_from_file_location(
        "_real_prediction_pipeline_v4_for_regression_tests", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_race11_dataframe() -> pd.DataFrame:
    parsed_race, validation_report = CsvIngestion(str(RACE11_CSV)).parse_csv()
    assert validation_report.is_valid, validation_report.errors
    return pd.DataFrame(parsed_race.records, columns=parsed_race.headers)


def test_optimizer_ranks_real_race11_on_normalized_probability():
    predictor = AdvancedEnsemblePredictor.__new__(AdvancedEnsemblePredictor)
    with RACE11_PREDICTION.open() as f:
        predictions = copy.deepcopy(json.load(f)["predictions"])

    ranked = predictor._normalize_race_probabilities(predictions)

    rank_probs = [p["rank_sort_probability"] for p in ranked]
    assert rank_probs == sorted(rank_probs, reverse=True)
    assert ranked[0]["dog_clean_name"] == "Rumour Not Fact"
    assert all(p["predicted_rank"] == index for index, p in enumerate(ranked, start=1))
    assert 0.999 <= sum(p["win_prob_norm_unrounded"] for p in ranked) <= 1.001


def test_optimizer_single_model_does_not_report_ensemble_agreement():
    predictor = AdvancedEnsemblePredictor.__new__(AdvancedEnsemblePredictor)
    with RACE11_PREDICTION.open() as f:
        predictions = copy.deepcopy(json.load(f)["predictions"])

    ranked = predictor._normalize_race_probabilities(predictions)

    assert all(p["model_agreement"] is None for p in ranked)
    assert all(p["confidence_score"] <= 0.55 for p in ranked)
    assert all(
        "single_model_no_ensemble_agreement" in p["quality_flags"] for p in ranked
    )


def test_csv_enrichment_marks_prefixed_first_form_row_as_explicit_history():
    pipeline_module = _load_real_prediction_pipeline_v4()
    pipeline = pipeline_module.PredictionPipelineV4.__new__(
        pipeline_module.PredictionPipelineV4
    )
    raw_csv = _load_race11_dataframe()
    participants = pd.DataFrame([{"dog_clean_name": "Rumour Not Fact", "box_number": 1}])

    enriched = pipeline._enrich_with_csv_historical_data(participants, raw_csv)
    row = enriched.iloc[0]

    assert row["csv_historical_races"] == 5
    assert row["csv_recent_form"] == 3
    assert row["csv_best_finish_position"] == 1
    assert row["csv_prefixed_history_rows"] == 1
    assert "prefixed_form_row" in row["csv_historical_sources"]
    assert "forward_filled_prefixed_row" in row["csv_historical_sources"]


def test_csv_enrichment_drops_history_rows_on_or_after_target_date():
    pipeline_module = _load_real_prediction_pipeline_v4()
    pipeline = pipeline_module.PredictionPipelineV4.__new__(
        pipeline_module.PredictionPipelineV4
    )
    raw_csv = pd.DataFrame(
        [
            {
                "Dog Name": "1. Alpha Runner",
                "PLC": 1,
                "TIME": 20.1,
                "DIST": 400,
                "DATE": "2026-05-25",
                "TRACK": "WPK",
            },
            {
                "Dog Name": '""',
                "PLC": 2,
                "TIME": 20.3,
                "DIST": 400,
                "DATE": "2026-05-24",
                "TRACK": "WPK",
            },
        ]
    )
    participants = pd.DataFrame(
        [
            {
                "dog_clean_name": "Alpha Runner",
                "box_number": 1,
                "race_date": "2026-05-25",
            }
        ]
    )

    enriched = pipeline._enrich_with_csv_historical_data(participants, raw_csv)
    row = enriched.iloc[0]

    assert row["csv_historical_races"] == 1
    assert row["csv_recent_form"] == 2
    assert row["csv_history_rows_dropped_post_target"] == 1


def test_map_csv_keeps_target_fields_separate_from_form_history():
    pipeline_module = _load_real_prediction_pipeline_v4()
    pipeline = pipeline_module.PredictionPipelineV4.__new__(
        pipeline_module.PredictionPipelineV4
    )
    raw_csv = _load_race11_dataframe()

    mapped = pipeline._map_csv_to_v4_format(
        raw_csv, str(RACE11_CSV)
    )

    assert mapped["dog_clean_name"].tolist() == [
        "Rumour Not Fact",
        "Mr. Fahrenheit",
        "Tootsie",
        "Offensive Lady",
        "Socks And Slides",
        "Rhine Star",
        "Chorus Line",
        "Valve Bounce",
    ]
    assert mapped["box_number"].tolist() == list(range(1, 9))
    assert mapped["weight"].tolist() == [30.0] * 8
    assert mapped["weight_source"].unique().tolist() == ["default_missing_target"]
    assert mapped["starting_price"].tolist() == [3.0] * 8
    assert mapped["starting_price_source"].unique().tolist() == [
        "default_missing_target"
    ]
    assert mapped["distance"].tolist() == [500.0] * 8
    assert mapped["distance_source"].unique().tolist() == [
        "default_missing_target"
    ]
    assert mapped["grade"].unique().tolist() == ["G5"]
    assert mapped["parser_context"].unique().tolist() == ["embedded_form_history"]
    assert "embedded_form_history_detected" in mapped["target_field_warning"].iloc[0]
    assert "historical_form_distance_mode_available:400" in mapped[
        "target_field_warning"
    ].iloc[0]


def test_odds_name_key_matches_punctuation_and_case_variants():
    pipeline_module = _load_real_prediction_pipeline_v4()
    odds = {}
    pipeline_module._store_market_odds(odds, "MR FAHRENHEIT", 1.33)

    assert pipeline_module._normalize_odds_name_key("Mr. Fahrenheit") == "MRFAHRENHEIT"
    assert odds[pipeline_module._normalize_odds_name_key("Mr. Fahrenheit")] == 1.33
    assert odds["MR FAHRENHEIT"] == 1.33


def test_race11_market_disagreement_annotations_do_not_change_ranking():
    pipeline_module = _load_real_prediction_pipeline_v4()
    with RACE11_PREDICTION.open() as f:
        predictions = copy.deepcopy(json.load(f)["predictions"])
    original_order = [p["dog_clean_name"] for p in predictions]

    odds = {}
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT dog_clean_name, odds_decimal
            FROM live_odds
            WHERE race_id = ?
              AND market_type = 'win'
              AND (is_current = 1 OR is_current IS NULL)
            """,
            ("Race 11 - WRGL - 2026-05-21",),
        ).fetchall()
    for dog_name, odds_decimal in rows:
        pipeline_module._store_market_odds(odds, dog_name, odds_decimal)

    summary = pipeline_module._annotate_market_context(predictions, odds)

    assert [p["dog_clean_name"] for p in predictions] == original_order
    assert summary["market_odds_count"] == 8
    assert summary["large_disagreement_count"] >= 2
    assert predictions[0]["market_odds_win"] == 21.0
    assert predictions[0]["odds_implied_prob"] == 1 / 21.0
    assert "large_model_market_disagreement" in predictions[0]["quality_flags"]


def test_csv_metadata_does_not_override_identity_from_real_form_history():
    meta = parse_race_csv_meta(str(RACE11_CSV))

    assert meta["venue"] == "WRGL"
    assert meta["race_date"] == "2026-05-21"
    assert meta["field_size"] == 8
    assert meta["distance"] == "Unknown"
    assert meta["grade"] == "Unknown"
    assert meta["csv_row_context"] == "embedded_form_history"
    assert meta["target_metadata_from_csv"] is False
