import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from unittest import mock

from scripts import evaluate_frozen_sportsbet_betfair_forward as forward


class TestFrozenSportsbetBetfairForward(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.august = self.root / "ANZ_Greyhounds_2026_08.csv"
        self.september = self.root / "ANZ_Greyhounds_2026_09.csv"
        self.sportsbet = self.root / forward.EXPECTED_SPORTSBET_PREDICTOR_FILENAME
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
        august_date="2026-08-20",
        market_id="1.100",
        first_selection_id="selection-a",
        scheduled_clock="12:00:00.000",
        include_outcomes=False,
    ):
        header = list(forward.BETFAIR_REQUIRED_COLUMNS)
        if include_outcomes:
            header += ["WIN_RESULT", "WIN_BSP", "ACTUAL_OFF_TIME"]
        with self.august.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            first = [
                    august_date,
                    scheduled_clock,
                    "Angle Park",
                    1,
                    market_id,
                    first_selection_id,
                    1,
                    "Alpha",
                    2.0,
            ]
            second = [
                    august_date,
                    scheduled_clock,
                    "Angle Park",
                    1,
                    market_id,
                    "selection-b",
                    2,
                    "Beta",
                    3.0,
            ]
            if include_outcomes:
                first += [
                    "WINNER",
                    1.8,
                    "12:00:01.000",
                ]
                second += [
                    "LOSER",
                    3.2,
                    "12:00:01.000",
                ]
            writer.writerow(first)
            writer.writerow(second)
        with self.september.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(header)

    def _sportsbet_rows(self):
        base = {
            "schema_version": forward.SPORTSBET_ROW_SCHEMA,
            "race_date": "2026-08-20",
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
                "sportsbet_predictor_filename": self.sportsbet.name,
                "sportsbet_predictor_sha256": self._sha256(self.sportsbet),
                "start_date_inclusive": "2026-08-20",
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
                    "start_date_inclusive": "2026-08-20",
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
                    "race_date": "2026-08-20",
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
        sportsbet = forward.load_sportsbet(self.sportsbet, self.sportsbet_receipt)
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

    def _authorize_and_load(self, population, approval):
        with mock.patch.object(forward, "datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 1)
            return forward.authorize_and_load_sealed_races_for_score(
                population,
                self.results,
                approval,
                self.frozen_hashes,
            )

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

    def test_replacement_artifacts_bind_the_exact_predecessor_candidate(self):
        repo_root = Path(forward.__file__).resolve().parents[1]
        predecessor = (
            repo_root
            / "artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only"
        )
        replacement = (
            repo_root
            / "artifacts/sportsbet_betfair_pristine_forward_confirmation_20260818"
        )

        self.assertEqual(
            (replacement / "frozen_consensus_rule.json").read_bytes(),
            (predecessor / "frozen_consensus_rule.json").read_bytes(),
        )
        self.assertEqual(
            forward.sha256_file(predecessor / "SHA256SUMS"),
            "af1fa6e4e3586248f903a3863bd71d7393eb382db7a473dbd99056f06d079a03",
        )
        self.assertEqual(
            forward.sha256_file(replacement / "frozen_consensus_rule.json"),
            forward.EXPECTED_RULE_SHA256,
        )
        self.assertEqual(
            forward.sha256_file(replacement / "protocol.json"),
            forward.EXPECTED_PROTOCOL_SHA256,
        )
        replacement_manifest = json.loads(
            (replacement / "cohort_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            replacement_manifest["bindings"]["replacement_evaluator_sha256"],
            forward.sha256_file(Path(forward.__file__)),
        )
        self.assertEqual(
            replacement_manifest["bindings"]["predecessor_future_eligibility_sha256"],
            forward.EXPECTED_PREDECESSOR_ELIGIBILITY_SHA256,
        )
        self.assertEqual(
            forward.verify_frozen_artifacts(replacement),
            {
                "frozen_consensus_rule.json": forward.EXPECTED_RULE_SHA256,
                "protocol.json": forward.EXPECTED_PROTOCOL_SHA256,
                "predecessor_future_eligibility_protocol.json": (
                    forward.EXPECTED_PREDECESSOR_ELIGIBILITY_SHA256
                ),
                "scorer": forward.EXPECTED_SCORER_SHA256,
            },
        )

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

    def test_result_bearing_betfair_projection_fails_before_rows_are_parsed(self):
        self._write_betfair_files(include_outcomes=True)
        self._write_betfair_receipt()
        original_sha256_file = forward.sha256_file

        def deny_full_outcome_file_read(path):
            if path == self.august:
                raise AssertionError("outcome-bearing Betfair rows hashed")
            return original_sha256_file(path)

        with mock.patch.object(forward, "sha256_file", side_effect=deny_full_outcome_file_read):
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "result-bearing Betfair columns are quarantined",
            ):
                self._source_contract()

    def test_predictor_only_betfair_projection_has_no_outcome_members(self):
        source = self._source_contract()[self.august.name]
        projected = asdict(forward._project_betfair_csv(self.august, source)[0])

        self.assertNotIn("WIN_RESULT", projected)
        self.assertNotIn("WIN_BSP", projected)
        self.assertNotIn("ACTUAL_OFF_TIME", projected)

    def test_wrong_path_predictor_parse_fails_before_result_open(self):
        original_open = Path.open

        def deny_result_open(path, *args, **kwargs):
            if path == self.results:
                raise AssertionError("result opened")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=deny_result_open):
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "not the frozen input name",
            ):
                forward.load_sportsbet(self.results, self.sportsbet_receipt)

    def test_no_import_level_result_read_bypass_is_exposed(self):
        for name in (
            "load_sealed_races",
            "authorize_outcome_read_for_score",
            "OutcomeReadAuthorization",
            "_ISSUE_OUTCOME_AUTHORIZATION",
            "strict_jsonl",
            "_strict_jsonl_unchecked",
            "_parse_jsonl_handle",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(forward, name))

    def test_authorization_fails_before_end_without_consuming_or_opening_results(self):
        population = self._seal()
        approval = self._approval_receipt(population)

        original_open = Path.open

        def deny_result_open(path, *args, **kwargs):
            if path == self.results:
                raise AssertionError("result opened")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=deny_result_open):
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "forbidden until after",
            ):
                with mock.patch.object(forward, "datetime") as clock:
                    clock.now.return_value = datetime(2026, 9, 30)
                    forward.authorize_and_load_sealed_races_for_score(
                        population,
                        self.results,
                        approval,
                        self.frozen_hashes,
                    )
        self.assertEqual(list(self.root.glob("*.score_consumed.json")), [])

    def test_authorized_result_read_is_single_use(self):
        population = self._seal()
        approval = self._approval_receipt(population)
        self._authorize_and_load(population, approval)

        original_open = Path.open

        def deny_result_open(path, *args, **kwargs):
            if path == self.results:
                raise AssertionError("result reopened")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=deny_result_open):
            with self.assertRaisesRegex(
                forward.ForwardContractError,
                "already been consumed",
            ):
                with mock.patch.object(forward, "datetime") as clock:
                    clock.now.return_value = datetime(2026, 10, 1)
                    forward.authorize_and_load_sealed_races_for_score(
                        population,
                        self.results,
                        approval,
                        self.frozen_hashes,
                    )

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
        rows = json.loads(self.results.read_text(encoding="utf-8"))
        rows["race_number"] = 2
        self._write_results([rows])

        with self.assertRaisesRegex(forward.ForwardContractError, "does not exactly equal"):
            self._authorize_and_load(population, approval)

    def test_score_is_forbidden_through_window_end(self):
        for value in (date(2026, 8, 19), date(2026, 8, 20), date(2026, 9, 30)):
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
        (
            first_races,
            first_results,
            _,
            first_provenance,
            _,
            _,
        ) = self._authorize_and_load(
            first,
            approval,
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
            first_races,
            {**first_provenance, **first_results},
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
