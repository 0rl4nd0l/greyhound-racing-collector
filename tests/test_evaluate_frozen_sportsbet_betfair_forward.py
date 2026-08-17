import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from datetime import date
from pathlib import Path
from unittest import mock

from scripts import evaluate_frozen_sportsbet_betfair_forward as forward


class TestFrozenSportsbetBetfairForward(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.august = self.root / "ANZ_Greyhounds_2026_08.csv"
        self.september = self.root / "ANZ_Greyhounds_2026_09.csv"
        self.sportsbet = self.root / "sportsbet.jsonl"
        self.sportsbet_receipt = self.root / "sportsbet_receipt.json"
        self.betfair_receipt = self.root / "betfair_receipt.json"
        self.results = self.root / "results.jsonl"
        self.frozen_hashes = {
            "frozen_consensus_rule.json": "a" * 64,
            "protocol.json": "b" * 64,
            "future_eligibility_protocol.json": "c" * 64,
            "scorer": "d" * 64,
        }
        self._write_betfair_files()
        self._write_sportsbet()
        self._write_sportsbet_receipt()
        self._write_betfair_receipt()
        self._write_results()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path, value):
        path.write_bytes(forward.canonical_json_bytes(value))

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_bytes(b"".join(forward.canonical_json_bytes(row) for row in rows))

    def _write_betfair_files(
        self,
        *,
        august_date="2026-08-18",
        market_id="1.100",
        first_selection_id="selection-a",
        scheduled_clock="12:00:00.000",
    ):
        header = list(forward.BETFAIR_REQUIRED_COLUMNS) + [
            "WIN_RESULT",
            "WIN_BSP",
            "ACTUAL_OFF_TIME",
        ]
        with self.august.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(
                [
                    august_date,
                    scheduled_clock,
                    "Angle Park",
                    1,
                    market_id,
                    first_selection_id,
                    1,
                    "Alpha",
                    2.0,
                    "WINNER",
                    1.8,
                    "12:00:01.000",
                ]
            )
            writer.writerow(
                [
                    august_date,
                    scheduled_clock,
                    "Angle Park",
                    1,
                    market_id,
                    "selection-b",
                    2,
                    "Beta",
                    3.0,
                    "LOSER",
                    3.2,
                    "12:00:01.000",
                ]
            )
        with self.september.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(header)

    def _sportsbet_rows(self):
        base = {
            "schema_version": forward.SPORTSBET_ROW_SCHEMA,
            "race_date": "2026-08-18",
            "sportsbet_venue": "AP K",
            "race_number": 1,
            "scheduled_race_time_raw": "12:00:00.000",
            "sportsbet_source_sha256": "e" * 64,
        }
        return [
            {
                **base,
                "box_number": 1,
                "runner_name": "Alpha",
                "sportsbet_normalized_probability": 0.6,
                "sportsbet_source_row_id": "sportsbet-1",
            },
            {
                **base,
                "box_number": 2,
                "runner_name": "Beta",
                "sportsbet_normalized_probability": 0.4,
                "sportsbet_source_row_id": "sportsbet-2",
            },
        ]

    def _write_sportsbet(self, rows=None):
        self._write_jsonl(self.sportsbet, self._sportsbet_rows() if rows is None else rows)

    def _write_sportsbet_receipt(self):
        self._write_json(
            self.sportsbet_receipt,
            {
                "schema_version": "sportsbet_forward_completeness_receipt_v1",
                "sportsbet_predictor_sha256": self._sha256(self.sportsbet),
                "start_date_inclusive": "2026-08-18",
                "end_date_inclusive": "2026-09-30",
                "declared_complete_without_results": True,
                "labels_inspected": False,
                "results_inspected": False,
            },
        )

    def _betfair_source(self, path):
        return {
            "filename": path.name,
            "source_url": f"https://promo.betfair.com/betfairsp/prices/{path.name}",
            "byte_size": path.stat().st_size,
            "sha256": self._sha256(path),
        }

    def _write_betfair_receipt(self):
        self._write_json(
            self.betfair_receipt,
            {
                "schema_version": forward.BETFAIR_SOURCE_RECEIPT_SCHEMA,
                "terminal_state": "BETFAIR_FORWARD_SOURCES_FROZEN_LABEL_BLIND",
                "window": {
                    "start_date_inclusive": "2026-08-18",
                    "end_date_inclusive": "2026-09-30",
                },
                "declared_complete_without_results": True,
                "labels_inspected": False,
                "results_inspected": False,
                "sources": [
                    self._betfair_source(self.august),
                    self._betfair_source(self.september),
                ],
            },
        )

    def _write_results(self, rows=None):
        if rows is None:
            rows = [
                {
                    "schema_version": forward.RESULT_ROW_SCHEMA,
                    "race_date": "2026-08-18",
                    "sportsbet_venue": "AP K",
                    "race_number": 1,
                    "scheduled_race_time_raw": "12:00:00.000",
                    "winner_box": 1,
                    "approved_result_source_sha256": "f" * 64,
                    "approved_result_source_row_id": "result-1",
                }
            ]
        self._write_jsonl(self.results, rows)

    def _source_contract(self):
        return forward.verify_betfair_source_receipt(
            self.betfair_receipt,
            [self.august, self.september],
        )[0]

    def _seal(self, name="population"):
        sources = self._source_contract()
        sportsbet = forward.load_sportsbet(self.sportsbet)
        betfair = forward.load_betfair([self.august, self.september], sources)
        predictor_rows, audit_rows = forward.seal_population(sportsbet, betfair)
        output = self.root / name
        forward.write_seal(
            output,
            predictor_rows,
            audit_rows,
            self.sportsbet,
            [self.august, self.september],
            self.sportsbet_receipt,
            self.betfair_receipt,
            sources,
            self.frozen_hashes,
        )
        return output

    def _approval_receipt(self, population, name="approval.json"):
        path = self.root / name
        self._write_json(
            path,
            {
                "schema_version": forward.POPULATION_APPROVAL_RECEIPT_SCHEMA,
                "terminal_state": "POPULATION_EXTERNALLY_APPROVED_FOR_ONE_SHOT_SCORE",
                "external_approval": True,
                "population_manifest_sha256": self._sha256(
                    population / "population_manifest.json"
                ),
                "population_review_used_results": False,
                "approved_by": "synthetic-reviewer",
                "approved_at_utc": "2026-10-01T00:00:00Z",
            },
        )
        return path

    def test_direct_cli_help_works_outside_repo(self):
        script = Path(forward.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("seal-population", result.stdout)
        self.assertIn("score", result.stdout)

    def test_clock_accepts_only_seconds_or_dot_zero_zero_zero(self):
        self.assertEqual(forward.normalized_clock("12:34:56", "clock"), "12:34:56")
        self.assertEqual(forward.normalized_clock("12:34:56.000", "clock"), "12:34:56")
        for value in ("12:34:56.001", "12:34:56.999", "24:00:00"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(forward.ForwardContractError, "invalid clock"):
                    forward.normalized_clock(value, "clock")

    def test_betfair_source_month_must_match_every_row(self):
        self._write_betfair_files(august_date="2026-09-01")
        self._write_betfair_receipt()
        sources = self._source_contract()

        with self.assertRaisesRegex(forward.ForwardContractError, "source file month mismatch"):
            forward.load_betfair([self.august, self.september], sources)

    def test_betfair_source_receipt_binds_bytes_and_hash(self):
        self.august.write_bytes(self.august.read_bytes() + b"\n")

        with self.assertRaisesRegex(forward.ForwardContractError, "source receipt drift"):
            self._source_contract()

    def test_empty_native_ids_fail_closed(self):
        cases = [
            ({"market_id": ""}, "WIN_MARKET_ID"),
            ({"first_selection_id": ""}, "SELECTION_ID"),
        ]
        for arguments, message in cases:
            with self.subTest(message=message):
                self._write_betfair_files(**arguments)
                self._write_betfair_receipt()
                sources = self._source_contract()
                with self.assertRaisesRegex(forward.ForwardContractError, message):
                    forward.load_betfair([self.august, self.september], sources)

    def test_forbidden_betfair_fields_are_not_projected(self):
        source = self._source_contract()[self.august.name]
        runners = forward._project_betfair_csv(self.august, source)
        projected = asdict(runners[0])

        self.assertNotIn("WIN_RESULT", projected)
        self.assertNotIn("WIN_BSP", projected)
        self.assertNotIn("ACTUAL_OFF_TIME", projected)

    def test_population_approval_receipt_rejects_manifest_drift(self):
        population = self._seal()
        approval = self._approval_receipt(population)
        forward.verify_population_approval_receipt(
            population,
            approval,
            self.frozen_hashes,
        )
        manifest = population / "population_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")

        with self.assertRaisesRegex(forward.ForwardContractError, "approval receipt"):
            forward.verify_population_approval_receipt(
                population,
                approval,
                self.frozen_hashes,
            )

    def test_approved_results_must_exactly_equal_sealed_races(self):
        population = self._seal()
        approval = self._approval_receipt(population)
        manifest, _ = forward.verify_population_approval_receipt(
            population,
            approval,
            self.frozen_hashes,
        )
        rows = json.loads(self.results.read_text(encoding="utf-8"))
        rows["race_number"] = 2
        self._write_results([rows])

        with self.assertRaisesRegex(forward.ForwardContractError, "does not exactly equal"):
            forward.load_sealed_races(population, self.results, manifest)

    def test_score_is_forbidden_through_window_end(self):
        for value in (date(2026, 8, 17), date(2026, 9, 30)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(forward.ForwardContractError, "forbidden until after"):
                    forward.enforce_score_date(value)
        forward.enforce_score_date(date(2026, 10, 1))

    def test_score_consumed_marker_is_exclusive_and_output_is_fixed(self):
        population = self._seal()
        approval = self._approval_receipt(population)
        _, provenance = forward.verify_population_approval_receipt(
            population,
            approval,
            self.frozen_hashes,
        )
        output, marker = forward.consume_score_once(population, provenance)

        self.assertEqual(
            output,
            self.root
            / f"sportsbet_betfair_forward_{provenance['population_manifest_sha256']}.evaluation.json",
        )
        self.assertTrue(marker.exists())
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        self.assertIs(marker_payload["approved_results_opened_before_marker"], False)
        with self.assertRaisesRegex(forward.ForwardContractError, "already been consumed"):
            forward.consume_score_once(population, provenance)

    def test_sealing_and_scoring_replay_are_byte_identical(self):
        first = self._seal("population-one")
        second = self._seal("population-two")
        first_members = {path.name: path.read_bytes() for path in first.iterdir()}
        second_members = {path.name: path.read_bytes() for path in second.iterdir()}
        self.assertEqual(first_members, second_members)

        approval = self._approval_receipt(first)
        first_manifest, first_provenance = forward.verify_population_approval_receipt(
            first,
            approval,
            self.frozen_hashes,
        )
        second_manifest, second_provenance = forward.verify_population_approval_receipt(
            second,
            approval,
            self.frozen_hashes,
        )
        first_races, first_results = forward.load_sealed_races(
            first,
            self.results,
            first_manifest,
        )
        second_races, second_results = forward.load_sealed_races(
            second,
            self.results,
            second_manifest,
        )
        with mock.patch.object(
            forward.frozen,
            "evaluate",
            wraps=forward.frozen.evaluate,
        ) as evaluate:
            first_report = forward.score_forward(
                first_races,
                {**first_provenance, **first_results},
            )
        second_report = forward.score_forward(
            second_races,
            {**second_provenance, **second_results},
        )

        self.assertEqual(
            forward.canonical_json_bytes(first_report),
            forward.canonical_json_bytes(second_report),
        )
        self.assertEqual(
            [call.args[1] for call in evaluate.call_args_list],
            ["sportsbet", "consensus"],
        )
        self.assertEqual(set(first_report["metrics"]), {"sportsbet", "consensus"})
        self.assertNotIn("betfair_only", json.dumps(first_report, sort_keys=True))
        self.assertIn("paired_deltas_consensus_minus_sportsbet", first_report)
        self.assertNotIn("paired_deltas_alternative_minus_sportsbet", first_report)
        self.assertIs(forward.forward_report_schema()["additionalProperties"], False)


if __name__ == "__main__":
    unittest.main()
