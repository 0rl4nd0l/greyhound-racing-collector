import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_cumulative_runner_matrix_challenger import run_evaluation


class CumulativeRunnerMatrixChallengerTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_runner_matrix(self, path: Path) -> Path:
        rows = [
            {
                "candidate_key": "stage2_uncalibrated_market_blend_70",
                "market_candidate_key": "market_only_implied",
                "race_id": "Race 1 - TEST - 2026-07-01",
                "source_report": "unified_a/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2026-07-01",
                "dog_name": "Winner One",
                "box_number": 1,
                "is_winner": "True",
                "finish_position": 1,
                "odds_decimal": 1.5,
                "market_probability": 0.7,
                "candidate_probability": 0.55,
                "primary_shadow_probability_norm": 0.2,
                "stage2_shadow_probability_norm": 0.2,
                "stage2_shadow_uncalibrated_probability_norm": 0.2,
            },
            {
                "candidate_key": "stage2_uncalibrated_market_blend_70",
                "market_candidate_key": "market_only_implied",
                "race_id": "Race 1 - TEST - 2026-07-01",
                "source_report": "unified_a/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2026-07-01",
                "dog_name": "Runner One",
                "box_number": 2,
                "is_winner": "False",
                "finish_position": 2,
                "odds_decimal": 3.0,
                "market_probability": 0.3,
                "candidate_probability": 0.45,
                "primary_shadow_probability_norm": 0.8,
                "stage2_shadow_probability_norm": 0.8,
                "stage2_shadow_uncalibrated_probability_norm": 0.8,
            },
            {
                "candidate_key": "stage2_uncalibrated_market_blend_70",
                "market_candidate_key": "market_only_implied",
                "race_id": "Race 2 - TEST - 2026-07-01",
                "source_report": "unified_b/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 2,
                "race_date": "2026-07-01",
                "dog_name": "Winner Two",
                "box_number": 3,
                "is_winner": "True",
                "finish_position": 1,
                "odds_decimal": 1.8,
                "market_probability": 0.65,
                "candidate_probability": 0.515,
                "primary_shadow_probability_norm": 0.2,
                "stage2_shadow_probability_norm": 0.2,
                "stage2_shadow_uncalibrated_probability_norm": 0.2,
            },
            {
                "candidate_key": "stage2_uncalibrated_market_blend_70",
                "market_candidate_key": "market_only_implied",
                "race_id": "Race 2 - TEST - 2026-07-01",
                "source_report": "unified_b/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 2,
                "race_date": "2026-07-01",
                "dog_name": "Runner Two",
                "box_number": 4,
                "is_winner": "False",
                "finish_position": 2,
                "odds_decimal": 4.0,
                "market_probability": 0.35,
                "candidate_probability": 0.485,
                "primary_shadow_probability_norm": 0.8,
                "stage2_shadow_probability_norm": 0.8,
                "stage2_shadow_uncalibrated_probability_norm": 0.8,
            },
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_old_result(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["race_id", "market_winner_rank"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "race_id": "Race 99 - OLD - 2026-05-31",
                    "market_winner_rank": 1,
                }
            )
        return path

    def _build_input_packet(self, root: Path) -> tuple[Path, Path]:
        matrix = self._write_runner_matrix(root / "market_residual_runner_matrix.csv")
        rolling = self._write_json(
            root / "rolling_model_comparison_report.json",
            {
                "minimum_races_for_review": 2,
                "candidate_metrics_by_key": {
                    "market_only_implied": {},
                    "primary_shadow": {},
                    "stage2_shadow": {},
                    "stage2_shadow_uncalibrated": {},
                    "stage2_uncalibrated_market_blend_70": {},
                },
            },
        )
        packet = self._write_json(
            root / "CUMULATIVE_RUNNER_MATRIX_CHALLENGER_PACKET.json",
            {
                "schema_version": "cumulative_runner_matrix_challenger_packet_v1",
                "status": "READY_FOR_REPORT_ONLY_CHALLENGER",
                "input_surface": "current_cumulative_rolling_runner_matrix",
                "paths": {
                    "rolling_report": str(rolling),
                    "runner_matrix": str(matrix),
                },
                "counts": {
                    "runner_matrix_race_count": 2,
                    "runner_matrix_rows": 4,
                    "complete_valid_odds_races": 2,
                    "official_result_joined_races": 2,
                },
                "readiness": {
                    "complete_market_comparable_status": "READY",
                    "race_count_match": True,
                    "runner_row_count_match": True,
                },
            },
        )
        return packet, matrix

    def test_report_only_evaluation_blocks_when_market_safety_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet, _ = self._build_input_packet(root)
            old_result = self._write_old_result(root / "old.tsv")

            report = run_evaluation(
                adapter_packet_path=packet,
                output_dir=root / "report",
                old_result_per_race_path=old_result,
                now=datetime(2026, 7, 2, tzinfo=timezone.utc),
            )

            self.assertEqual(report["final_status"], "BLOCKED_KEEP_BASELINE")
            self.assertEqual(report["state"], "DONE_WITH_RISK")
            self.assertFalse(report["market_safety_rank_first_gate"]["promotion_ready"])
            self.assertIn(
                "top1_not_above_market",
                report["market_safety_rank_first_gate"]["blockers"],
            )
            self.assertEqual(
                report["old_recovered_input_guard"]["status"],
                "OLD_INPUT_DIAGNOSTIC",
            )
            self.assertTrue(
                Path(report["output_paths"]["candidate_comparison"]).exists()
            )

    def test_evaluation_rejects_non_current_adapter_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet, _ = self._build_input_packet(root)
            payload = json.loads(packet.read_text(encoding="utf-8"))
            payload["input_surface"] = "old_recovered_history_form_input"
            packet.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "adapter_packet_not_current"):
                run_evaluation(
                    adapter_packet_path=packet,
                    output_dir=root / "report",
                    now=datetime(2026, 7, 2, tzinfo=timezone.utc),
                )


if __name__ == "__main__":
    unittest.main()
