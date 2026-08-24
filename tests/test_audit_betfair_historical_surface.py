import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.audit_betfair_historical_surface import (
    EXPECTED_COLUMNS,
    average_ranks,
    build_report,
    join_surfaces,
    load_sportsbet_matrix,
    parse_betfair_sources,
    parse_optional_price,
    summarize_price_surface,
)


def _sportsbet_row(box, name, *, winner=False, clock="19:04:00+10:00"):
    return {
        "race_id": "Race 1 - AP_K - 2026-06-11",
        "race_date": "2026-06-11",
        "race_number": 1,
        "venue": "AP K",
        "jump_at": f"2026-06-11T{clock}",
        "box_number": box,
        "dog_name": name,
        "canonical_sportsbet_win_odds": 2.0 if box == 1 else 4.0,
        "market_implied_probability": 2 / 3 if box == 1 else 1 / 3,
        "label_is_winner": int(winner),
        "_matrix_row_index": box - 1,
    }


def _betfair_row(box, name, *, winner=False, selection_id=None, clock="19:04:00"):
    price = 2.2 if box == 1 else 4.4
    return {
        "local_meeting_date": "2026-06-11",
        "track": "Angle Park",
        "race_number": 1,
        "win_market_id": "259056256",
        "win_market_name": "R1 530m Gr5",
        "selection_id": str(selection_id or 1000 + box),
        "tab_number": box,
        "runner_name": name,
        "win_result": "WINNER" if winner else "LOSER",
        "scheduled_race_time_raw": f"{clock}.000",
        "scheduled_race_clock": clock,
        "actual_off_time_raw": "19:05:12.000",
        "actual_off_clock": "19:05:12",
        "scheduled_off_back_price_raw": str(price),
        "scheduled_off_back_price": price,
        "scheduled_off_back_price_status": "PRESENT",
        "win_bsp_raw": str(price + 0.1),
        "win_bsp": price + 0.1,
        "win_bsp_status": "PRESENT",
        "source_file": "ANZ_Greyhounds_2026_06.csv",
        "source_file_sha256": "a" * 64,
        "source_rows": [
            {
                "source_file": "ANZ_Greyhounds_2026_06.csv",
                "source_row_number": box + 1,
            }
        ],
        "win_projection_sha256": "b" * 64,
    }


def _raw_csv_row(**overrides):
    row = {field: "" for field in EXPECTED_COLUMNS}
    row.update(
        {
            "LOCAL_MEETING_DATE": "2026-06-11",
            "SCHEDULED_RACE_TIME": "19:04:00.000",
            "ACTUAL_OFF_TIME": "19:05:12.000",
            "TRACK": "Angle Park",
            "STATE_CODE": "SA",
            "RACE_NO": "1",
            "WIN_MARKET_ID": "259056256",
            "WIN_MARKET_NAME": "R1 530m Gr5",
            "PLACE_MARKET_ID": "259056257",
            "RACING_TYPE": "Greyhounds",
            "DISTANCE": "530",
            "RACE_TYPE": "Gr5",
            "SELECTION_ID": "84912662",
            "TAB_NUMBER": "1",
            "SELECTION_NAME": "El Rey",
            "WIN_RESULT": "LOSER",
            "WIN_BSP": "2.45",
            "PLACE_RESULT": "LOSER",
            "PLACE_BSP": "1.59",
            "BEST_AVAIL_BACK_AT_SCHEDULED_OFF": "2.36",
        }
    )
    row.update(overrides)
    return row


def _joined_row(box, *, scheduled_price=2.0):
    return {
        "race_id": "Race 1 - AP_K - 2026-06-11",
        "sportsbet_normalized_probability": 0.6 if box == 1 else 0.4,
        "betfair_scheduled_off_back_price": scheduled_price,
        "betfair_scheduled_off_back_price_status": (
            "PRESENT" if scheduled_price is not None else "MISSING_BLANK"
        ),
        "betfair_bsp": 2.1 if box == 1 else 4.1,
        "betfair_bsp_status": "PRESENT",
    }


def _matched_audit():
    return {
        "race_id": "Race 1 - AP_K - 2026-06-11",
        "status": "MATCHED",
        "exclusion_reason": None,
        "sportsbet_runner_count": 2,
        "betfair_tab_numbers": [1, 2],
        "scheduled_clock_precedes_provider_actual_off_clock": True,
    }


class BetfairHistoricalSurfaceTest(unittest.TestCase):
    def test_parser_preserves_distinct_times_and_prices_and_collapses_win_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.csv"
            receipt_path = root / "headers.txt"
            rows = [
                _raw_csv_row(),
                _raw_csv_row(
                    PLACE_MARKET_ID="259056299", PLACE_RESULT="", PLACE_BSP=""
                ),
            ]
            with raw_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            receipt_path.write_text("HTTP/2 200\n", encoding="utf-8")
            raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            manifest = {
                "sources": [
                    {
                        "raw_path": "raw.csv",
                        "receipt_path": "headers.txt",
                        "sha256": raw_hash,
                        "receipt_sha256": receipt_hash,
                        "byte_size": raw_path.stat().st_size,
                        "filename": "raw.csv",
                        "retrieved_at_utc": "2026-08-17T00:00:00Z",
                        "source_url": "https://example.invalid/raw.csv",
                    }
                ]
            }

            parsed, sidecar, metadata = parse_betfair_sources(root, manifest)

            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["scheduled_race_time_raw"], "19:04:00.000")
            self.assertEqual(parsed[0]["actual_off_time_raw"], "19:05:12.000")
            self.assertEqual(parsed[0]["scheduled_off_back_price"], 2.36)
            self.assertEqual(parsed[0]["win_bsp"], 2.45)
            self.assertEqual(parsed[0]["duplicate_win_projection_count"], 2)
            self.assertEqual(len(sidecar[0]["source_rows"]), 2)
            self.assertEqual(metadata["counts"]["duplicate_extra_rows_collapsed"], 1)

    def test_join_requires_metadata_time_and_complete_tab_box_identity(self):
        sportsbet = [
            _sportsbet_row(1, "Alpha", winner=True),
            _sportsbet_row(2, "Bravo"),
        ]
        betfair = [
            _betfair_row(1, "Alpha", winner=True),
            _betfair_row(2, "Bravo"),
        ]

        joined, audit = join_surfaces(sportsbet, betfair, {"2026-06"})

        self.assertEqual(len(joined), 2)
        self.assertEqual(audit[0]["status"], "MATCHED")
        self.assertEqual(joined[0]["selection_id"], "1001")

        wrong_time = [dict(row, scheduled_race_clock="19:09:00") for row in betfair]
        joined, audit = join_surfaces(sportsbet, wrong_time, {"2026-06"})
        self.assertEqual(joined, [])
        self.assertEqual(audit[0]["exclusion_reason"], "SCHEDULED_TIME_MISMATCH")

        reserve = [betfair[0], _betfair_row(9, "Bravo", selection_id=1002)]
        joined, audit = join_surfaces(sportsbet, reserve, {"2026-06"})
        self.assertEqual(joined, [])
        self.assertEqual(
            audit[0]["exclusion_reason"],
            "RUNNER_SET_MISMATCH_RESERVE_OR_SCRATCH",
        )
        self.assertEqual(audit[0]["betfair_reserve_tabs"], [9])

    def test_join_rejects_inconsistent_provider_actual_off_time(self):
        sportsbet = [
            _sportsbet_row(1, "Alpha", winner=True),
            _sportsbet_row(2, "Bravo"),
        ]
        inconsistent = [
            _betfair_row(1, "Alpha", winner=True),
            dict(
                _betfair_row(2, "Bravo"),
                actual_off_time_raw="19:05:13.000",
                actual_off_clock="19:05:13",
            ),
        ]

        joined, audit = join_surfaces(sportsbet, inconsistent, {"2026-06"})

        self.assertEqual(joined, [])
        self.assertEqual(
            audit[0]["exclusion_reason"], "PROVIDER_ACTUAL_OFF_TIME_CONFLICT"
        )
        self.assertEqual(
            audit[0]["provider_actual_off_clocks"], ["19:05:12", "19:05:13"]
        )

        all_blank = [
            dict(row, actual_off_time_raw="", actual_off_clock=None)
            for row in inconsistent
        ]
        joined, audit = join_surfaces(sportsbet, all_blank, {"2026-06"})
        self.assertEqual(len(joined), 2)
        self.assertEqual(audit[0]["status"], "MATCHED")
        self.assertIsNone(audit[0]["actual_off_clock"])
        self.assertFalse(
            audit[0]["scheduled_clock_precedes_provider_actual_off_clock"]
        )

    def test_name_is_only_corroboration_and_cannot_repair_box_identity(self):
        sportsbet = [
            _sportsbet_row(1, "Alpha", winner=True),
            _sportsbet_row(2, "Bravo"),
        ]
        swapped_names = [
            _betfair_row(1, "Bravo", winner=True),
            _betfair_row(2, "Alpha"),
        ]

        joined, audit = join_surfaces(sportsbet, swapped_names, {"2026-06"})

        self.assertEqual(joined, [])
        self.assertEqual(
            audit[0]["exclusion_reason"], "RUNNER_NAME_CORROBORATION_CONFLICT"
        )

    def test_missing_month_is_explicit_and_not_substituted(self):
        sportsbet = [
            dict(
                _sportsbet_row(1, "Alpha", winner=True),
                race_id="Race 1 - AP_K - 2026-08-01",
                race_date="2026-08-01",
                jump_at="2026-08-01T19:04:00+10:00",
            )
        ]

        joined, audit = join_surfaces(sportsbet, [], {"2026-06", "2026-07"})

        self.assertEqual(joined, [])
        self.assertEqual(
            audit[0]["exclusion_reason"], "BETFAIR_MONTHLY_FILE_UNAVAILABLE"
        )

    def test_scheduled_off_and_bsp_diagnostics_remain_separate(self):
        joined = [
            {
                "race_id": "R1",
                "sportsbet_normalized_probability": 0.6,
                "betfair_scheduled_off_back_price": 2.0,
                "betfair_bsp": 2.5,
            },
            {
                "race_id": "R1",
                "sportsbet_normalized_probability": 0.4,
                "betfair_scheduled_off_back_price": 4.0,
                "betfair_bsp": None,
            },
        ]

        scheduled = summarize_price_surface(
            joined, "betfair_scheduled_off_back_price", "scheduled"
        )
        bsp = summarize_price_surface(joined, "betfair_bsp", "bsp")

        self.assertEqual(scheduled["complete_price_races"], 1)
        self.assertEqual(scheduled["mean_overround"], 0.75)
        self.assertEqual(bsp["complete_price_races"], 0)
        self.assertEqual(bsp["missing_price_runner_rows"], 1)

    def test_average_ranks_uses_midrank_for_ties(self):
        self.assertEqual(average_ranks([0.5, 0.5, 0.2]), [1.5, 1.5, 3.0])

    def test_nonfinite_published_price_is_preserved_as_unusable(self):
        self.assertEqual(
            parse_optional_price("inf", "WIN_BSP"),
            (None, "NONFINITE_LITERAL"),
        )
        self.assertEqual(
            parse_optional_price("", "WIN_BSP"), (None, "MISSING_BLANK")
        )

    def test_sportsbet_matrix_requires_valid_normalized_market(self):
        valid_rows = [
            _sportsbet_row(1, "Alpha", winner=True),
            _sportsbet_row(2, "Bravo"),
        ]
        invalid_cases = [
            (
                "odds",
                [
                    dict(valid_rows[0], canonical_sportsbet_win_odds=1.0),
                    valid_rows[1],
                ],
                "sportsbet_odds_invalid",
            ),
            (
                "probability",
                [
                    dict(valid_rows[0], market_implied_probability=0.0),
                    valid_rows[1],
                ],
                "sportsbet_probability_invalid",
            ),
            (
                "sum",
                [
                    dict(valid_rows[0], market_implied_probability=0.6),
                    dict(valid_rows[1], market_implied_probability=0.3),
                ],
                "sportsbet_probability_sum_invalid",
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.jsonl"
            for case, rows, error_pattern in invalid_cases:
                with self.subTest(case=case):
                    path.write_text(
                        "".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8",
                    )
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with mock.patch(
                        "scripts.audit_betfair_historical_surface."
                        "EXPECTED_SPORTSBET_SHA256",
                        digest,
                    ):
                        with self.assertRaisesRegex(ValueError, error_pattern):
                            load_sportsbet_matrix(path)

    def test_report_verdict_is_derived_from_coverage_and_conflicts(self):
        manifest = {
            "sources": [{"retrieved_at_utc": "2026-08-17T00:00:00Z"}],
            "unavailable_sources": [],
        }
        common = {
            "artifact_root": Path("unused"),
            "manifest": manifest,
            "betfair_meta": {},
        }
        joined = [_joined_row(1), _joined_row(2)]

        ready = build_report(
            **common,
            sportsbet_meta={"races": 1, "runner_rows": 2},
            joined=joined,
            audits=[_matched_audit()],
        )
        self.assertEqual(ready["verdict"], "READY")
        self.assertEqual(
            ready["terminal_state"], "BETFAIR_HISTORICAL_SURFACE_READY"
        )

        excluded = {
            "race_id": "Race 2 - AP_K - 2026-06-11",
            "status": "EXCLUDED",
            "exclusion_reason": "SCHEDULED_TIME_MISMATCH",
            "sportsbet_runner_count": 1,
        }
        partial = build_report(
            **common,
            sportsbet_meta={"races": 2, "runner_rows": 3},
            joined=joined,
            audits=[_matched_audit(), excluded],
        )
        self.assertEqual(partial["verdict"], "PARTIAL")
        self.assertEqual(
            partial["terminal_state"], "BETFAIR_HISTORICAL_SURFACE_PARTIAL"
        )
        self.assertTrue(
            any("SCHEDULED_TIME_MISMATCH" in reason for reason in partial["verdict_reasons"])
        )

        conflict = dict(
            excluded,
            exclusion_reason="RUNNER_NAME_CORROBORATION_CONFLICT",
        )
        not_ready = build_report(
            **common,
            sportsbet_meta={"races": 2, "runner_rows": 3},
            joined=joined,
            audits=[_matched_audit(), conflict],
        )
        self.assertEqual(not_ready["verdict"], "NOT_READY")
        self.assertEqual(
            not_ready["terminal_state"], "BETFAIR_HISTORICAL_SURFACE_NOT_READY"
        )


if __name__ == "__main__":
    unittest.main()
