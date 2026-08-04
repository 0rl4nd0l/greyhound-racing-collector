from __future__ import annotations

import importlib.util
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "classify_forecasting_changes",
    ROOT / "scripts/ci/classify_forecasting_changes.py",
)
assert SPEC and SPEC.loader
CLASSIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLASSIFIER
SPEC.loader.exec_module(CLASSIFIER)


class ForecastingChangeClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = CLASSIFIER.load_rules()
        cls.representative = {
            "ci_contract": ".github/workflows/forecasting-tests.yml",
            "manual_prediction": "scripts/predict_market_form_residual.py",
            "official_results": "scripts/ingest_results_for_date.py",
            "forward_corpus": "race_collection/scheduled_forward_corpus.py",
            "operator_ui": "src/operator_ui/foundation.py",
            "forecasting_core": "race_collection/forecasting.py",
        }

    def classify(self, *changes: tuple[str, str]) -> dict:
        return CLASSIFIER.classify_changes(
            [
                CLASSIFIER.Change(status=status, paths=(path,))
                for status, path in changes
            ],
            self.rules,
        )

    def test_rules_define_exact_required_tiers(self):
        self.assertEqual(set(self.rules), set(CLASSIFIER.TIERS))
        self.assertEqual(
            set(CLASSIFIER.TIERS),
            {
                "ci_contract",
                "manual_prediction",
                "official_results",
                "forward_corpus",
                "operator_ui",
                "forecasting_core",
                "full_forecasting",
                "non_forecasting",
            },
        )

    def test_each_path_family_selects_smallest_trusted_tier(self):
        fixtures = {
            "ci_contract": (
                ".github/forecasting-paths.ini",
                ".github/workflows/backend-tests.yml",
                "scripts/ci/run_forecasting_ci_contract.py",
                "tests/ci/test_forecasting_change_classifier.py",
                "docs/forecasting_ci_tiers.md",
            ),
            "manual_prediction": (
                "configs/prediction/market-only.json",
                "scripts/predict_market_form_residual.py",
                "src/predictor/market_form_residual.py",
                "tests/test_predict_market_form_residual.py",
            ),
            "official_results": (
                "scripts/autonomous_official_result_capture.py",
                "scripts/collect_expert_form_official_result_labels_report_only.py",
                "scripts/ingest_results_for_date.py",
                "tests/test_results_ingest_official_first.py",
            ),
            "forward_corpus": (
                "race_collection/forward_sealed_corpus.py",
                "race_collection/scheduled_forward_corpus.py",
                "scripts/collect_forward_sealed_corpus.py",
                "scripts/observe_forward_official_results.py",
                "tests/race_collection/test_forward_official_result_observer.py",
                "tests/race_collection/test_forward_sealed_corpus.py",
                "tests/race_collection/test_phase7_source_admission.py",
                "tests/race_collection/test_scheduled_forward_corpus.py",
            ),
            "operator_ui": (
                "src/operator_ui/foundation.py",
                "src/operator_ui/job_store.py",
                "static/css/operator-ui.css",
                "static/js/operator-ui-connected.js",
                "templates/operator_ui_connected.jinja",
                "tests/operator_ui/test_foundation.py",
            ),
            "forecasting_core": (
                "race_collection/forecast_service.py",
                "race_collection/forecasting.py",
                "tests/race_collection/test_phase3_forecasting.py",
                "tests/race_collection/test_phase4_model_serving.py",
            ),
            "full_forecasting": (
                "race_collection/domain.py",
                "race_collection/features.py",
                "race_collection/identity.py",
                "race_collection/source_admission.py",
                "race_collection/training.py",
                "race_collection/synchronous_manual_capture.py",
                "configs/prediction/schemas/manual.schema.json",
                "requirements/all.in",
                "scripts/predict_race_now.py",
                "src/operator_ui/bootstrap.py",
                "tests/race_collection/test_phase5_ordered_finish_training.py",
                "tests/operator_ui/test_deployment_generator.py",
                "tests/test_predict_race_now.py",
                "utils/csv_metadata.py",
            ),
            "non_forecasting": (
                "README.md",
                "docs/race_evidence_inventory.md",
                "reports/agent_jobs/example/README.md",
                "static/css/base.css",
                "templates/index.html",
            ),
        }
        for expected, paths in fixtures.items():
            for path in paths:
                with self.subTest(expected=expected, path=path):
                    self.assertEqual(self.classify(("M", path))["tier"], expected)

    def test_ci_only_routing_change_never_selects_complete_suite(self):
        result = self.classify(
            ("M", ".github/forecasting-paths.ini"),
            ("M", ".github/workflows/forecasting-tests.yml"),
            ("M", "scripts/ci/classify_forecasting_changes.py"),
            ("M", "tests/ci/test_forecasting_change_classifier.py"),
        )
        self.assertEqual(result["tier"], "ci_contract")
        self.assertEqual(result["reason"], "single_trusted_tier")

    def test_same_focused_family_and_docs_are_compatible(self):
        for tier, path in self.representative.items():
            with self.subTest(tier=tier):
                result = self.classify(("M", path), ("M", "README.md"))
                self.assertEqual(result["tier"], tier)

    def test_every_cross_tier_focused_combination_escalates(self):
        for left, right in itertools.combinations(self.representative, 2):
            with self.subTest(left=left, right=right):
                result = self.classify(
                    ("M", self.representative[left]),
                    ("M", self.representative[right]),
                )
                self.assertEqual(result["tier"], "full_forecasting")
                self.assertEqual(
                    result["reason"], "incompatible_mixed_tiers_default_to_full"
                )

    def test_shared_core_mixed_with_focused_escalates(self):
        result = self.classify(
            ("M", "scripts/predict_market_form_residual.py"),
            ("M", "race_collection/source_admission.py"),
        )
        self.assertEqual(result["tier"], "full_forecasting")
        self.assertEqual(result["reason"], "shared_or_high_risk_path_requires_full")

    def test_unknown_path_escalates(self):
        result = self.classify(("A", "new_subsystem/adapter.py"))
        self.assertEqual(result["tier"], "full_forecasting")
        self.assertEqual(result["reason"], "unknown_path_defaults_to_full")

    def test_destructive_change_in_every_tier_escalates(self):
        paths = list(self.representative.values()) + ["README.md"]
        for status in ("D", "R100", "C100", "T"):
            for path in paths:
                with self.subTest(status=status, path=path):
                    paths_for_change = (path, path + ".moved") if status[0] in "RC" else (path,)
                    result = CLASSIFIER.classify_changes(
                        [CLASSIFIER.Change(status=status, paths=paths_for_change)],
                        self.rules,
                    )
                    self.assertEqual(result["tier"], "full_forecasting")
                    self.assertEqual(
                        result["reason"], "destructive_change_defaults_to_full"
                    )

    def test_empty_unknown_status_and_unsafe_path_escalate(self):
        self.assertEqual(
            CLASSIFIER.classify_changes([], self.rules)["tier"], "full_forecasting"
        )
        for change in (
            CLASSIFIER.Change(status="?", paths=("README.md",)),
            CLASSIFIER.Change(status="X", paths=("README.md",)),
            CLASSIFIER.Change(status="M", paths=("../outside.py",)),
            CLASSIFIER.Change(status="M", paths=("/absolute.py",)),
        ):
            with self.subTest(change=change):
                self.assertEqual(
                    CLASSIFIER.classify_changes([change], self.rules)["tier"],
                    "full_forecasting",
                )

    def test_name_status_parser_keeps_both_rename_and_copy_paths(self):
        changes = CLASSIFIER.parse_name_status(
            b"R100\0scripts/predict_market_form_residual.py\0archive/manual_predictor.py\0"
            b"C100\0README.md\0docs/copied.md\0"
            b"D\0race_collection/source_admission.py\0"
        )
        self.assertEqual(
            changes,
            [
                CLASSIFIER.Change(
                    status="R100",
                    paths=(
                        "scripts/predict_market_form_residual.py",
                        "archive/manual_predictor.py",
                    ),
                ),
                CLASSIFIER.Change(
                    status="C100", paths=("README.md", "docs/copied.md")
                ),
                CLASSIFIER.Change(
                    status="D", paths=("race_collection/source_admission.py",)
                ),
            ],
        )

    def test_invalid_rules_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rules = Path(temp_dir) / "rules.ini"
            rules.write_text(
                "[metadata]\nschema_version = forecasting-change-rules-v1\n",
                encoding="utf-8",
            )
            with self.assertRaises(CLASSIFIER.ClassificationError):
                CLASSIFIER.load_rules(rules)

    def test_workflow_preserves_stable_gate_and_full_escape_hatches(self):
        workflow = (ROOT / ".github/workflows/forecasting-tests.yml").read_text(
            encoding="utf-8"
        )
        for expected in (
            "name: tests-race-collection",
            "schedule:",
            "workflow_dispatch:",
            "ci:full-forecasting",
            "--force-full",
            "forecasting-ci-attestation-v2",
            '"tier": tier',
            '"${selected_command[@]}"',
            'forecasting-command.txt',
            '--with PyYAML python scripts/ci/run_forecasting_ci_contract.py',
            '"commit": subprocess.check_output',
            '"tree": subprocess.check_output',
            '"log_sha256": hashlib.sha256',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, workflow)
        for tier in CLASSIFIER.TIERS:
            self.assertIn(tier, workflow)

    def test_ci_contract_runner_has_only_named_fast_smokes(self):
        runner = (
            ROOT / "scripts/ci/run_forecasting_ci_contract.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"tests/race_collection"', runner)
        self.assertIn("test_scores_exact_packet_deterministically", runner)
        self.assertIn("test_parse_sportsbet_result_text", runner)
        self.assertIn("test_fixture_scheduled_capture_admits_once", runner)
        self.assertIn("test_valid_envelope_is_deterministic", runner)
        self.assertIn("test_prediction_rejected_before_close", runner)

    def test_full_runner_contains_complete_and_focused_suites(self):
        runner = (ROOT / "scripts/ci/run_full_forecasting.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"scripts/ci/run_forecasting_ci_contract.py"', runner)
        self.assertIn('"tests/race_collection"', runner)
        self.assertIn('"tests/test_predict_market_form_residual.py"', runner)
        self.assertIn('"tests/test_predict_race_now.py"', runner)
        self.assertIn('"tests/test_results_ingest_official_first.py"', runner)
        self.assertIn('"tests/operator_ui/test_foundation.py"', runner)
        for direct_full_test in (
            '"tests/operator_ui/test_api.py"',
            '"tests/operator_ui/test_bootstrap.py"',
            '"tests/operator_ui/test_deployment_generator.py"',
            '"tests/operator_ui/test_live_adapters.py"',
        ):
            self.assertIn(direct_full_test, runner)


if __name__ == "__main__":
    unittest.main()
