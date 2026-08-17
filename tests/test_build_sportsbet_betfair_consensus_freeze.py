import json
import math
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from scripts import build_sportsbet_betfair_consensus_freeze as consensus


def make_race(
    race_id: str = "Race 1 - TEST - 2026-07-01",
    race_date: str = "2026-07-01",
    venue: str = "TEST",
    sportsbet=(0.6, 0.4),
    betfair=(0.5, 0.5),
    winner_index: int = 0,
) -> consensus.Race:
    prices = tuple(1.0 / value for value in betfair)
    return consensus.Race(
        race_id=race_id,
        race_date=race_date,
        venue=venue,
        race_number=1,
        scheduled_race_time_raw="12:00:00.000",
        win_market_id=f"market-{race_id}",
        split="validation",
        boxes=(1, 2),
        selection_ids=("a", "b"),
        sportsbet_probabilities=tuple(sportsbet),
        betfair_probabilities=tuple(betfair),
        betfair_prices=prices,
        winner_index=winner_index,
        sportsbet_matrix_row_indices=(1, 2),
        betfair_source_file="ANZ_Greyhounds_2026_07.csv",
        betfair_source_file_sha256=consensus.EXPECTED_BETFAIR_SOURCE_HASHES[
            "ANZ_Greyhounds_2026_07.csv"
        ],
    )


def joined_row(**overrides):
    row = {
        "schema_version": consensus.EXPECTED_JOIN_SCHEMA_VERSION,
        "race_id": "race",
        "race_date": "2026-07-01",
        "sportsbet_venue": "TEST",
        "race_number": 1,
        "scheduled_race_time_raw": "12:00:00.000",
        "win_market_id": "market",
        "box_number": 1,
        "selection_id": "selection",
        "sportsbet_normalized_probability": 1.0,
        "betfair_scheduled_off_back_price": 2.0,
        "betfair_source_file": "ANZ_Greyhounds_2026_07.csv",
        "betfair_source_file_sha256": "hash",
        "sportsbet_matrix_sha256": "hash",
        "sportsbet_matrix_row_index": 1,
        "scheduled_clock_precedes_provider_actual_off_clock": True,
        "sportsbet_runner_name": "Runner",
        "betfair_runner_name": "Runner",
        "win_result": "WINNER",
    }
    row.update(overrides)
    return row


class TestSportsbetBetfairConsensusFreeze(unittest.TestCase):
    def test_score_consensus_is_bounded_normalized_convex_combination(self):
        scored = consensus.score_consensus((0.6, 0.4), (4.0, 2.0), 0.25)

        self.assertAlmostEqual(scored[0], 0.5333333333)
        self.assertAlmostEqual(scored[1], 0.4666666667)
        self.assertAlmostEqual(math.fsum(scored), 1.0)

    def test_score_consensus_fails_closed(self):
        cases = [
            ((0.5,), (2.0, 3.0), 0.5, "runner count mismatch"),
            ((0.5, 0.5), (1.0, 3.0), 0.5, "prices must be finite and > 1"),
            ((0.5, 0.5), (2.0, 3.0), -0.1, r"within \[0, 1\]"),
            ((0.5, float("nan")), (2.0, 3.0), 0.5, "finite positive"),
        ]
        for sportsbet, prices, weight, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(consensus.ContractError, message):
                    consensus.score_consensus(sportsbet, prices, weight)

    def test_ranking_ties_break_by_box_ascending(self):
        race = make_race(winner_index=1)
        values = consensus.race_metric_values(race, (0.5, 0.5))

        self.assertEqual(values["winner_rank"], 2.0)
        self.assertEqual(values["top1"], 0.0)
        self.assertEqual(values["top2"], 1.0)

    def test_select_weight_uses_fit_log_loss_and_lower_weight_tie(self):
        def fake_evaluate(_races, _model, weight=None):
            return {"log_loss": 1.0 if weight in {0.2, 0.3} else 2.0, "brier": 0.5}

        with mock.patch.object(consensus, "evaluate", side_effect=fake_evaluate):
            selected, diagnostics = consensus.select_weight([make_race()], [0.3, 0.2, 0.4])

        self.assertEqual(selected, 0.2)
        self.assertEqual(len(diagnostics), 3)

    def test_bootstrap_is_deterministic_and_meeting_date_clustered(self):
        races = [
            make_race(race_id="a", race_date="2026-07-01", venue="X"),
            make_race(race_id="b", race_date="2026-07-01", venue="X", winner_index=1),
            make_race(race_id="c", race_date="2026-07-02", venue="Y"),
        ]

        first = consensus.bootstrap_delta(races, "consensus", 0.5, 100, 42)
        second = consensus.bootstrap_delta(races, "consensus", 0.5, 100, 42)

        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 2)

    def test_project_runner_drops_bsp_actual_off_and_other_unapproved_fields(self):
        row = joined_row(
            betfair_bsp=1.01,
            actual_off_time_raw="23:59:59.999",
            post_jump_value=999,
        )

        projected = consensus.project_runner(row)

        self.assertNotIn("betfair_bsp", projected)
        self.assertNotIn("actual_off_time_raw", projected)
        self.assertNotIn("post_jump_value", projected)

    def test_project_runner_requires_expected_join_schema_version(self):
        with self.assertRaisesRegex(consensus.ContractError, "schema_version mismatch"):
            consensus.project_runner(joined_row(schema_version="wrong_schema"))

    def test_global_runner_identity_is_one_to_one_and_row_indices_are_unique(self):
        race_to_market = {}
        market_to_race = {}
        row_indices = set()
        first = {
            "race_id": "race-1",
            "win_market_id": "market-1",
            "sportsbet_matrix_row_index": 1,
        }
        consensus.record_global_runner_identity(
            first,
            race_to_market,
            market_to_race,
            row_indices,
        )

        with self.assertRaisesRegex(consensus.ContractError, "race_id maps to multiple"):
            consensus.record_global_runner_identity(
                {
                    "race_id": "race-1",
                    "win_market_id": "market-2",
                    "sportsbet_matrix_row_index": 2,
                },
                race_to_market,
                market_to_race,
                row_indices,
            )
        with self.assertRaisesRegex(consensus.ContractError, "win_market_id maps to multiple"):
            consensus.record_global_runner_identity(
                {
                    "race_id": "race-2",
                    "win_market_id": "market-1",
                    "sportsbet_matrix_row_index": 3,
                },
                race_to_market,
                market_to_race,
                row_indices,
            )
        with self.assertRaisesRegex(
            consensus.ContractError,
            "duplicate sportsbet_matrix_row_index",
        ):
            consensus.record_global_runner_identity(
                first,
                race_to_market,
                market_to_race,
                row_indices,
            )

    def test_betfair_source_filename_month_must_match_race_date(self):
        consensus.validate_betfair_source_month(
            "ANZ_Greyhounds_2026_07.csv",
            date(2026, 7, 18),
        )

        with self.assertRaisesRegex(consensus.ContractError, "source file month mismatch"):
            consensus.validate_betfair_source_month(
                "ANZ_Greyhounds_2026_06.csv",
                date(2026, 7, 1),
            )

    def test_name_corroboration_normalization_matches_audited_rule(self):
        self.assertEqual(
            consensus.normalized_name("Mr. Graceland"),
            consensus.normalized_name("Mr Graceland"),
        )
        self.assertNotEqual(
            consensus.normalized_name("Mr Graceland"),
            consensus.normalized_name("Ms Graceland"),
        )

    def test_protocol_is_hash_bound_and_future_population_is_untouched(self):
        root = Path(__file__).resolve().parents[1]
        protocol_path = (
            root
            / "artifacts"
            / "sportsbet_betfair_consensus_freeze_20260817_report_only"
            / "protocol.json"
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

        self.assertEqual(
            consensus.sha256_file(protocol_path),
            consensus.EXPECTED_PROTOCOL_SHA256,
        )
        self.assertIs(
            protocol["future_evaluation"]["future_rows_loaded_or_scored_during_freeze"],
            False,
        )
        self.assertEqual(protocol["future_evaluation"]["interim_peeking"], "forbidden")
        self.assertEqual(
            protocol["future_evaluation"]["start_date_inclusive"],
            "2026-08-18",
        )
        self.assertEqual(
            protocol["future_evaluation"]["end_date_inclusive"],
            "2026-09-30",
        )

    def test_report_schema_has_closed_top_level_contract(self):
        schema = consensus.report_schema()

        self.assertIs(schema["additionalProperties"], False)
        self.assertIn("terminal_state", schema["required"])
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            consensus.SCHEMA_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
