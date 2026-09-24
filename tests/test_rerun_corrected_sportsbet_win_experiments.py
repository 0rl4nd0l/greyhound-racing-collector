import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.rerun_corrected_sportsbet_win_experiments import (
    conclusion_signature,
    corrected_fixed_window,
    corrected_intersection,
    corrected_tier_a_races,
    load_tier_a_capture_timestamps,
    original_metric_summary,
)


class CorrectedSportsbetWinRerunTest(unittest.TestCase):
    def test_fresh_v1_original_candidate_schema_is_supported(self):
        verdict, metrics = original_metric_summary(
            "fresh_v1",
            {
                "status": "MODEL_FROZEN_READY_FOR_TEST",
                "candidates": {
                    "market_baseline": {
                        "validation": {"log_loss": 1.5},
                    }
                },
            },
        )
        self.assertEqual(verdict, "MODEL_FROZEN_READY_FOR_TEST")
        self.assertEqual(metrics, {"market_baseline": {"log_loss": 1.5}})

    def test_conclusion_signature_detects_selection_change_not_label_change(self):
        old_tier = {"disposition": "FROZEN_RF_DOES_NOT_BEAT_MARKET"}
        corrected_tier = {"status": "FROZEN_RF_DOES_NOT_BEAT_CORRECTED_MARKET"}
        old_fresh = {"status": "MODEL_FROZEN_READY_FOR_TEST", "selection": {"selected_candidate_id": "market_baseline"}}
        corrected_fresh = {"status": "CORRECTED_RERUN_COMPLETE", "selection": {"selected_candidate_id": "history_only_logistic"}}

        self.assertEqual(
            conclusion_signature("tier_a", old_tier),
            conclusion_signature("tier_a", corrected_tier),
        )
        self.assertNotEqual(
            conclusion_signature("fresh_v1", old_fresh),
            conclusion_signature("fresh_v1", corrected_fresh),
        )

    def test_corrected_intersection_retains_only_exact_whole_races(self):
        timestamp = "2026-08-01T00:00:00+00:00"
        original = [
            {
                "race_id": race_id,
                "box_number": box,
                "odds_capture_timestamp": timestamp,
                "market_implied_probability": 0.5,
                "label_is_winner": int(box == 1),
            }
            for race_id in ("R1", "R2")
            for box in (1, 2)
        ]
        canonical = {
            ("R1", box, timestamp): {
                "market_implied_probability": probability,
                "canonical_sportsbet_win_odds": odds,
                "sportsbet_win_source_row_id": box,
                "sportsbet_win_evidence_classification": "VERIFIED_WIN",
            }
            for box, probability, odds in ((1, 0.6, 2.0), (2, 0.4, 3.0))
        }
        canonical[("R2", 1, timestamp)] = {
            "market_implied_probability": 1.0,
            "canonical_sportsbet_win_odds": 2.0,
            "sportsbet_win_source_row_id": 3,
            "sportsbet_win_evidence_classification": "VERIFIED_WIN",
        }

        corrected = corrected_intersection(original, canonical)

        self.assertEqual({row["race_id"] for row in corrected}, {"R1"})
        self.assertEqual(
            [row["market_implied_probability"] for row in corrected], [0.6, 0.4]
        )

    def test_tier_a_capture_timestamp_map_rejects_duplicate_native_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tier_a.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "race_id",
                        "box_number",
                        "strongest_tier",
                        "odds_capture_timestamp",
                        "strict_win_odds",
                        "dog_name",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "race_id": "R1",
                            "box_number": 1,
                            "strongest_tier": "A",
                            "odds_capture_timestamp": "T1",
                            "strict_win_odds": 2.0,
                            "dog_name": "Dog One",
                        },
                        {
                            "race_id": "R1",
                            "box_number": 1,
                            "strongest_tier": "A",
                            "odds_capture_timestamp": "T2",
                            "strict_win_odds": 2.0,
                            "dog_name": "Dog One",
                        },
                    ]
                )

            with self.assertRaisesRegex(ValueError, "duplicate Tier-A runner identity"):
                load_tier_a_capture_timestamps(path)

    def test_tier_a_correction_classifies_source_and_excludes_whole_race(self):
        class AuditModule:
            @staticmethod
            def normalize_name(value):
                return str(value).casefold().replace(" ", "")

            @staticmethod
            def classify_win_evidence(*, raw_text, expected_box, stored_odds):
                classification = {
                    "win": "VERIFIED_WIN",
                    "place": "PLACE_MISLABEL",
                    "bad": "UNPARSABLE",
                }[raw_text]
                return SimpleNamespace(
                    classification=classification,
                    canonical_win_odds={
                        "win": stored_odds,
                        "place": 4.0,
                        "bad": None,
                    }[raw_text],
                    paired_win_odds=None,
                    paired_place_odds=None,
                    reason="test",
                )

        races = [
            {
                "race_id": race_id,
                "rows": [
                    {
                        "race_id": race_id,
                        "box_number": box,
                        "strict_win_odds": 2.0 if box == 1 else 1.5,
                    }
                    for box in (1, 2)
                ],
            }
            for race_id in ("R1", "R2")
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runners_path = root / "runners.csv"
            with runners_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "race_id",
                        "box_number",
                        "dog_name",
                        "strongest_tier",
                        "strict_win_odds",
                        "odds_capture_timestamp",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "race_id": race_id,
                        "box_number": box,
                        "dog_name": f"Dog {race_id} {box}",
                        "strongest_tier": "A",
                        "strict_win_odds": 2.0 if box == 1 else 1.5,
                        "odds_capture_timestamp": f"{race_id}-T",
                    }
                    for race_id in ("R1", "R2")
                    for box in (1, 2)
                )
            database = root / "source.db"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE live_odds "
                "(id INTEGER PRIMARY KEY, race_id TEXT, box_number INTEGER, "
                "dog_name TEXT, odds_decimal REAL, capture_timestamp TEXT, "
                "source TEXT, market_type TEXT, sportsbet_raw_runner_text TEXT)"
            )
            database_rows = []
            row_id = 0
            for race_id in ("R1", "R2"):
                for box in (1, 2):
                    row_id += 1
                    raw_text = "win"
                    if race_id == "R1" and box == 2:
                        raw_text = "place"
                    if race_id == "R2" and box == 2:
                        raw_text = "bad"
                    database_rows.append(
                        (
                            row_id,
                            race_id,
                            box,
                            f"Dog {race_id} {box}",
                            2.0 if box == 1 else 1.5,
                            f"{race_id}-T",
                            "sportsbet",
                            "win",
                            raw_text,
                        )
                    )
            connection.executemany(
                "INSERT INTO live_odds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                database_rows,
            )
            connection.commit()
            connection.close()

            retained, audited, excluded = corrected_tier_a_races(
                races=races,
                runners_path=runners_path,
                source_db=database,
                audit_module=AuditModule,
            )

        self.assertEqual([race["race_id"] for race in retained], ["R1"])
        self.assertEqual(len(audited), 4)
        self.assertEqual(retained[0]["rows"][1]["strict_win_odds"], 4.0)
        self.assertEqual(
            excluded,
            [{"race_id": "R2", "reason": "noncanonical_tier_a_win_evidence"}],
        )

    def test_fixed_window_uses_independently_qualified_win_rows(self):
        class AuditModule:
            @staticmethod
            def classify_win_evidence(*, raw_text, expected_box, stored_odds):
                classification = (
                    "UNPARSABLE" if raw_text == "bad" else "VERIFIED_WIN"
                )
                return SimpleNamespace(
                    classification=classification,
                    canonical_win_odds=None
                    if classification == "UNPARSABLE"
                    else stored_odds,
                    paired_win_odds=None,
                    paired_place_odds=None,
                    reason="test",
                )

        class MarketModule:
            EARLY_MODE = "T-30"
            LATEST_MODE = "T-10"
            FEATURES = ["market_move"]

            @staticmethod
            def normalize(values):
                values = np.asarray(values, dtype=float)
                return values / values.sum()

            @staticmethod
            def feature_vector(early, latest):
                return np.asarray(latest, dtype=float)[:, None] - np.asarray(
                    early, dtype=float
                )[:, None]

        old_rows = [
            {
                "race_id": race_id,
                "box_number": box,
                "market_implied_probability": 0.5,
            }
            for race_id in ("R1", "R2")
            for box in (1, 2)
        ]
        source_extract = []
        database_rows = []
        row_id = 0
        for race_id in ("R1", "R2"):
            for mode, odds in (("T-30", (2.0, 4.0)), ("T-10", (3.0, 3.0))):
                for box, price in enumerate(odds, start=1):
                    row_id += 1
                    source_extract.append(
                        {
                            "id": row_id,
                            "race_id": race_id,
                            "box_number": box,
                            "capture_mode": mode,
                        }
                    )
                    database_rows.append(
                        (
                            row_id,
                            box,
                            price,
                            "bad" if race_id == "R2" and box == 2 else "good",
                        )
                    )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "source.db"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE live_odds "
                "(id INTEGER PRIMARY KEY, box_number INTEGER, odds_decimal REAL, "
                "sportsbet_raw_runner_text TEXT)"
            )
            connection.executemany(
                "INSERT INTO live_odds VALUES (?, ?, ?, ?)", database_rows
            )
            connection.commit()
            connection.close()

            corrected, corrected_source, excluded = corrected_fixed_window(
                old_rows=old_rows,
                source_extract=source_extract,
                source_db=database,
                audit_module=AuditModule,
                market_module=MarketModule,
            )

        self.assertEqual({row["race_id"] for row in corrected}, {"R1"})
        self.assertEqual(len(corrected_source), 4)
        self.assertEqual(excluded, [{"race_id": "R2", "reason": "unparseable_fixed_window_win_evidence"}])
        self.assertEqual(
            [row["market_implied_probability"] for row in corrected], [0.5, 0.5]
        )
        self.assertAlmostEqual(corrected[0]["early_market_implied_probability"], 2 / 3)
        self.assertAlmostEqual(corrected[1]["early_market_implied_probability"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
