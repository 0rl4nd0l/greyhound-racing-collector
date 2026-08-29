import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_sportsbet_win_market_surface import (
    CONFLICTING,
    PLACE_MISLABEL,
    UNPARSABLE,
    VERIFIED_WIN,
    bind_legacy_training_matrix,
    classify_win_evidence,
    qualify_races,
)


def _paired_text(*, box: int = 3, win: str = "4.20", place: str = "1.57") -> str:
    return (
        f"{box}. Example Runner ({box})\n"
        "F: 241112\n"
        "Early Speed:\n"
        "5.00\n4.40\n4.00\n"
        f"{win}\nFav\n{place}\nEW\nIn\nForm"
    )


class SportsbetWinMarketSurfaceTest(unittest.TestCase):
    def test_classification_binds_win_and_place_columns_from_explicit_pair(self):
        verified = classify_win_evidence(
            raw_text=_paired_text(), expected_box=3, stored_odds=4.20
        )
        mislabeled = classify_win_evidence(
            raw_text=_paired_text(), expected_box=3, stored_odds=1.57
        )

        self.assertEqual(verified.classification, VERIFIED_WIN)
        self.assertEqual(verified.canonical_win_odds, 4.20)
        self.assertEqual(verified.paired_place_odds, 1.57)
        self.assertEqual(mislabeled.classification, PLACE_MISLABEL)
        self.assertEqual(mislabeled.canonical_win_odds, 4.20)
        self.assertEqual(
            mislabeled.reason, "stored_price_matches_source_paired_place"
        )

    def test_malformed_pair_fails_closed_without_price_fallback(self):
        no_ew = classify_win_evidence(
            raw_text=_paired_text().replace("\nEW", ""),
            expected_box=3,
            stored_odds=4.20,
        )
        reversed_columns = classify_win_evidence(
            raw_text=_paired_text(win="1.57", place="4.20"),
            expected_box=3,
            stored_odds=1.57,
        )

        self.assertEqual(no_ew.classification, UNPARSABLE)
        self.assertIsNone(no_ew.canonical_win_odds)
        self.assertEqual(no_ew.reason, "ew_control_missing")
        self.assertEqual(reversed_columns.classification, CONFLICTING)
        self.assertIsNone(reversed_columns.canonical_win_odds)
        self.assertEqual(
            reversed_columns.reason, "paired_market_order_conflict"
        )

    def test_runner_identity_conflict_fails_closed(self):
        result = classify_win_evidence(
            raw_text=_paired_text(box=4), expected_box=3, stored_odds=4.20
        )

        self.assertEqual(result.classification, CONFLICTING)
        self.assertIsNone(result.canonical_win_odds)
        self.assertEqual(result.reason, "raw_runner_box_conflict")

    def test_complete_field_qualification_excludes_entire_race(self):
        rows = [
            {"race_id": "R1", "classification": VERIFIED_WIN},
            {"race_id": "R1", "classification": PLACE_MISLABEL},
            {"race_id": "R2", "classification": VERIFIED_WIN},
            {"race_id": "R2", "classification": UNPARSABLE},
        ]

        qualified = qualify_races(rows)

        self.assertEqual(qualified, {"R1": True, "R2": False})

    def test_legacy_matrix_binding_requires_exact_identity_and_probability(self):
        sidecar = [
            {
                "race_id": "R1",
                "box_number": 1,
                "capture_timestamp": "2026-01-01T00:00:00+00:00",
                "stored_odds_decimal": 2.0,
                "classification": VERIFIED_WIN,
            },
            {
                "race_id": "R1",
                "box_number": 2,
                "capture_timestamp": "2026-01-01T00:00:00+00:00",
                "stored_odds_decimal": 4.0,
                "classification": PLACE_MISLABEL,
            },
        ]
        rows = [
            {
                "race_id": "R1",
                "box_number": 1,
                "odds_capture_timestamp": "2026-01-01T00:00:00+00:00",
                "market_implied_probability": 2 / 3,
            },
            {
                "race_id": "R1",
                "box_number": 2,
                "odds_capture_timestamp": "2026-01-01T00:00:00+00:00",
                "market_implied_probability": 1 / 3,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_matrix.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            bound = bind_legacy_training_matrix(path, sidecar)
            self.assertEqual(len(bound), 2)

            rows[1]["market_implied_probability"] = 0.34
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "probability_mismatch"):
                bind_legacy_training_matrix(path, sidecar)


if __name__ == "__main__":
    unittest.main()
