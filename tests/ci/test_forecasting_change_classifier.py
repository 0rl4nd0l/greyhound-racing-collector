from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import json
import os
import subprocess
import sys
import tempfile
import textwrap
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
                "configs/prediction/manual-independent-capture-v1/config.schema.json",
                "configs/prediction/market-only.json",
                "docs/manual_independent_capture_v1.md",
                "scripts/predict_market_form_residual.py",
                "src/predictor/manual_independent_capture.py",
                "src/predictor/manual_independent_capture_sealer.py",
                "src/predictor/market_form_residual.py",
                "tests/test_manual_independent_capture.py",
                "tests/test_manual_independent_capture_sealer.py",
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
                    result = self.classify(("M", path))
                    self.assertEqual(result["tier"], expected)
                    self.assertIs(
                        result["ci_contract_changed"], expected == "ci_contract"
                    )

    def test_ci_only_routing_change_never_selects_complete_suite(self):
        result = self.classify(
            ("M", ".github/forecasting-paths.ini"),
            ("M", ".github/workflows/forecasting-tests.yml"),
            ("M", "scripts/ci/classify_forecasting_changes.py"),
            ("M", "tests/ci/test_forecasting_change_classifier.py"),
        )
        self.assertEqual(result["tier"], "ci_contract")
        self.assertEqual(result["reason"], "single_trusted_tier")
        self.assertIs(result["ci_contract_changed"], True)

    def test_ci_contract_is_transparent_with_each_single_product_tier(self):
        for tier, path in self.representative.items():
            if tier == "ci_contract":
                continue
            with self.subTest(tier=tier):
                result = self.classify(
                    ("M", ".github/forecasting-paths.ini"),
                    ("M", path),
                )
                self.assertEqual(result["tier"], tier)
                self.assertEqual(
                    result["reason"], "single_product_tier_with_ci_contract"
                )
                self.assertIs(result["ci_contract_changed"], True)

    def test_ghu_050_path_mix_selects_manual_prediction_with_ci_contract(self):
        result = self.classify(
            ("M", ".github/forecasting-paths.ini"),
            ("M", ".github/workflows/forecasting-tests.yml"),
            ("A", "configs/prediction/manual-independent-capture-v1/config.schema.json"),
            ("A", "configs/prediction/manual-independent-capture-v1/example-config.json"),
            (
                "A",
                "configs/prediction/manual-independent-capture-v1/terminal-artifact.schema.json",
            ),
            ("M", "docs/forecasting_ci_tiers.md"),
            ("A", "docs/manual_independent_capture_v1.md"),
            ("M", "scripts/ci/run_full_forecasting.py"),
            ("A", "src/predictor/manual_independent_capture.py"),
            ("M", "tests/ci/test_forecasting_change_classifier.py"),
            ("A", "tests/test_manual_independent_capture.py"),
        )
        self.assertEqual(result["tier"], "manual_prediction")
        self.assertIs(result["ci_contract_changed"], True)

    def test_ghu_051_path_mix_selects_manual_prediction_with_ci_contract(self):
        result = self.classify(
            ("M", ".github/forecasting-paths.ini"),
            ("M", ".github/workflows/forecasting-tests.yml"),
            ("M", "docs/forecasting_ci_tiers.md"),
            ("M", "docs/manual_independent_capture_v1.md"),
            ("M", "scripts/ci/run_full_forecasting.py"),
            ("A", "src/predictor/manual_independent_capture_executor.py"),
            ("M", "tests/ci/test_forecasting_change_classifier.py"),
            ("A", "tests/fixtures/manual_independent_capture_child.py"),
            ("A", "tests/test_manual_independent_capture_executor.py"),
        )
        self.assertEqual(result["tier"], "manual_prediction")
        self.assertIs(result["ci_contract_changed"], True)

    def test_ghu_052_path_mix_selects_manual_prediction_with_ci_contract(self):
        result = self.classify(
            ("M", ".github/forecasting-paths.ini"),
            ("M", ".github/workflows/forecasting-tests.yml"),
            (
                "A",
                "configs/prediction/manual-independent-capture-v1/evidence-bundle.schema.json",
            ),
            (
                "A",
                "configs/prediction/manual-independent-capture-v1/evidence-manifest.schema.json",
            ),
            ("M", "docs/forecasting_ci_tiers.md"),
            ("M", "docs/manual_independent_capture_v1.md"),
            ("M", "scripts/ci/run_full_forecasting.py"),
            ("M", "src/predictor/manual_independent_capture_executor.py"),
            ("A", "src/predictor/manual_independent_capture_sealer.py"),
            ("M", "tests/ci/test_forecasting_change_classifier.py"),
            ("M", "tests/fixtures/manual_independent_capture_child.py"),
            ("M", "tests/test_manual_independent_capture_executor.py"),
            ("A", "tests/test_manual_independent_capture_sealer.py"),
        )
        self.assertEqual(result["tier"], "manual_prediction")
        self.assertIs(result["ci_contract_changed"], True)

    def test_ghu_054_path_mix_selects_manual_prediction_with_ci_contract(self):
        result = self.classify(
            ("M", ".github/forecasting-paths.ini"),
            ("M", ".github/workflows/forecasting-tests.yml"),
            (
                "A",
                "configs/prediction/manual-independent-capture-v1/manual-research-adapter-response.schema.json",
            ),
            ("A", "docs/manual_research_prediction_cli.md"),
            ("A", "src/predictor/manual_research_scoring.py"),
            ("A", "src/predictor/manual_research_cli.py"),
            ("M", "tests/ci/test_forecasting_change_classifier.py"),
            ("A", "tests/test_manual_research_scoring.py"),
            ("A", "tests/test_manual_research_cli.py"),
        )
        self.assertEqual(result["tier"], "manual_prediction")
        self.assertEqual(result["reason"], "single_product_tier_with_ci_contract")
        self.assertIs(result["ci_contract_changed"], True)

    def test_ghu_058_exact_changed_path_set_selects_manual_prediction(self):
        changes = CLASSIFIER.git_changes(
            "5e9a370477a905a67bdcb26c9b9315ef0050b362", "HEAD"
        )
        changed_paths = {
            path for change in changes for path in change.paths
        }
        self.assertEqual(
            changed_paths,
            {
                ".github/forecasting-paths.ini",
                ".github/workflows/forecasting-tests.yml",
                "configs/prediction/manual-readiness-v1/scoring-readiness.schema.json",
                "docs/manual_prediction_scoring_readiness.md",
                "race_collection/manual_scoring_readiness.py",
                "race_collection/synchronous_manual_capture.py",
                "scripts/ci/classify_forecasting_changes.py",
                "tests/ci/test_forecasting_change_classifier.py",
                "tests/race_collection/test_manual_scoring_readiness.py",
            },
        )
        result = CLASSIFIER.classify_changes(changes, self.rules)
        self.assertEqual(result["tier"], "manual_prediction")
        self.assertEqual(result["reason"], "single_product_tier_with_ci_contract")
        self.assertIs(result["ci_contract_changed"], True)
        self.assertNotEqual(result["tier"], "full_forecasting")
        shared = next(
            item
            for item in result["paths"]
            if item["path"] == "race_collection/synchronous_manual_capture.py"
        )
        self.assertEqual(shared["tier"], "manual_prediction")
        self.assertEqual(shared["matched_tiers"], ["manual_prediction"])

    def test_shared_capture_path_without_exact_allowlist_remains_full(self):
        result = self.classify(
            ("M", "race_collection/synchronous_manual_capture.py")
        )
        self.assertEqual(result["tier"], "full_forecasting")
        self.assertEqual(result["reason"], "shared_or_high_risk_path_requires_full")

    def test_future_manual_path_registration_stays_focused(self):
        future_path = "src/predictor/manual_isolated_executor.py"
        future_rules = dict(self.rules)
        future_rules["manual_prediction"] = (
            *future_rules["manual_prediction"],
            future_path,
        )
        result = CLASSIFIER.classify_changes(
            [
                CLASSIFIER.Change(
                    status="M", paths=(".github/forecasting-paths.ini",)
                ),
                CLASSIFIER.Change(status="A", paths=(future_path,)),
            ],
            future_rules,
        )
        self.assertEqual(result["tier"], "manual_prediction")
        self.assertIs(result["ci_contract_changed"], True)

    def test_same_focused_family_and_docs_are_compatible(self):
        for tier, path in self.representative.items():
            with self.subTest(tier=tier):
                result = self.classify(("M", path), ("M", "README.md"))
                self.assertEqual(result["tier"], tier)

    def test_every_cross_tier_focused_combination_escalates(self):
        product_tiers = {
            tier: path
            for tier, path in self.representative.items()
            if tier != "ci_contract"
        }
        for left, right in itertools.combinations(product_tiers, 2):
            with self.subTest(left=left, right=right):
                result = self.classify(
                    ("M", product_tiers[left]),
                    ("M", product_tiers[right]),
                )
                self.assertEqual(result["tier"], "full_forecasting")
                self.assertEqual(
                    result["reason"], "incompatible_mixed_tiers_default_to_full"
                )

    def test_ci_contract_does_not_hide_two_product_tiers(self):
        result = self.classify(
            ("M", ".github/workflows/forecasting-tests.yml"),
            ("M", "scripts/predict_market_form_residual.py"),
            ("M", "scripts/ingest_results_for_date.py"),
        )
        self.assertEqual(result["tier"], "full_forecasting")
        self.assertEqual(result["reason"], "incompatible_mixed_tiers_default_to_full")
        self.assertIs(result["ci_contract_changed"], True)

    def test_one_path_matching_two_product_rules_escalates(self):
        overlapping_rules = dict(self.rules)
        path = "src/predictor/manual_independent_capture.py"
        overlapping_rules["official_results"] = (
            *overlapping_rules["official_results"],
            path,
        )
        result = CLASSIFIER.classify_changes(
            [CLASSIFIER.Change(status="M", paths=(path,))],
            overlapping_rules,
        )
        self.assertEqual(result["tier"], "full_forecasting")
        self.assertEqual(result["reason"], "incompatible_mixed_tiers_default_to_full")
        self.assertIs(result["ci_contract_changed"], False)

    def test_shared_core_mixed_with_focused_escalates(self):
        result = self.classify(
            ("M", "scripts/predict_market_form_residual.py"),
            ("M", "race_collection/source_admission.py"),
        )
        self.assertEqual(result["tier"], "full_forecasting")
        self.assertEqual(result["reason"], "shared_or_high_risk_path_requires_full")

    def test_full_path_escalates_even_with_ci_contract_and_one_product_tier(self):
        result = self.classify(
            ("M", ".github/workflows/forecasting-tests.yml"),
            ("M", "scripts/predict_market_form_residual.py"),
            ("M", "race_collection/source_admission.py"),
        )
        self.assertEqual(result["tier"], "full_forecasting")
        self.assertEqual(result["reason"], "shared_or_high_risk_path_requires_full")
        self.assertIs(result["ci_contract_changed"], True)

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
            CLASSIFIER.Change(status="M100", paths=("README.md",)),
            CLASSIFIER.Change(status="R101", paths=("README.md", "docs/moved.md")),
            CLASSIFIER.Change(status="R0000", paths=("README.md", "docs/moved.md")),
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
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = CLASSIFIER.main(
                    ["--base", "HEAD", "--head", "HEAD", "--rules", str(rules)]
                )
            self.assertEqual(exit_code, 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["tier"], "full_forecasting")
            self.assertEqual(result["reason"], "classifier_error_defaults_to_full")
            self.assertIs(result["ci_contract_changed"], False)

    def test_github_outputs_include_exact_ci_contract_boolean(self):
        result = self.classify(
            ("M", ".github/forecasting-paths.ini"),
            ("M", "scripts/predict_market_form_residual.py"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            CLASSIFIER.write_github_output(output, result)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "tier=manual_prediction\n"
                "reason=single_product_tier_with_ci_contract\n"
                "ci_contract_changed=true\n",
            )

    def test_force_full_writes_exact_safe_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = CLASSIFIER.main(
                    ["--force-full", "--github-output", str(output)]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["tier"], "full_forecasting")
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "tier=full_forecasting\n"
                "reason=explicit_full_validation\n"
                "ci_contract_changed=false\n",
            )

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
            "forecasting-ci-attestation-v3",
            '"tier": tier',
            "ci_contract_changed: ${{ steps.classify.outputs.ci_contract_changed }}",
            'CI_CONTRACT_CHANGED: ${{ needs.classify.outputs.ci_contract_changed }}',
            'run_command ci_contract "${ci_contract_command[@]}"',
            'run_command "$FORECASTING_TIER" "${selected_command[@]}"',
            'if [[ "$CI_CONTRACT_CHANGED" == "true"',
            '"ci_contract_changed": ci_contract_changed',
            '"commands": commands',
            '"exit_code": exit_code',
            '"log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()',
            "forecasting-command-manifest.tsv",
            'forecasting-command.txt',
            '--with PyYAML python scripts/ci/run_forecasting_ci_contract.py',
            "commit = subprocess.check_output",
            "tree = subprocess.check_output",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, workflow)
        for tier in CLASSIFIER.TIERS:
            self.assertIn(tier, workflow)
        self.assertEqual(workflow.count("scripts/ci/run_full_forecasting.py"), 1)
        self.assertLess(
            workflow.index('run_command ci_contract "${ci_contract_command[@]}"'),
            workflow.index(
                'run_command "$FORECASTING_TIER" "${selected_command[@]}"'
            ),
        )

    def test_mixed_command_attestation_binds_both_logs_to_exact_head_and_tree(self):
        workflow = (ROOT / ".github/workflows/forecasting-tests.yml").read_text(
            encoding="utf-8"
        )
        marker = "          python - <<'PY'\n"
        start = workflow.index(marker) + len(marker)
        source = textwrap.dedent(workflow[start : workflow.index("\n          PY", start)])
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id = "123"
            fake_bin = Path(temp_dir) / "bin"
            fake_bin.mkdir()
            fake_uv = fake_bin / "uv"
            fake_uv.write_text("#!/bin/sh\nprintf 'uv-test 0.0\\n'\n", encoding="utf-8")
            fake_uv.chmod(0o755)
            evidence_dir = Path(temp_dir) / f"forecasting-acceptance-{run_id}"
            evidence_dir.mkdir()
            (evidence_dir / "forecasting-command-manifest.tsv").write_text(
                "ci_contract\t0\tci_contract-command.txt\tci_contract.log\n"
                "manual_prediction\t0\tmanual_prediction-command.txt\tmanual_prediction.log\n",
                encoding="utf-8",
            )
            (evidence_dir / "ci_contract-command.txt").write_text(
                "ci-contract-command\n", encoding="utf-8"
            )
            (evidence_dir / "manual_prediction-command.txt").write_text(
                "manual-command\n", encoding="utf-8"
            )
            (evidence_dir / "ci_contract.log").write_text(
                "ci contract passed\n", encoding="utf-8"
            )
            (evidence_dir / "manual_prediction.log").write_text(
                "manual suite passed\n", encoding="utf-8"
            )
            (evidence_dir / "forecasting-command.txt").write_text(
                "ci-contract-command\nmanual-command\n", encoding="utf-8"
            )
            (evidence_dir / "forecasting-suite.log").write_text(
                "ci contract passed\nmanual suite passed\n", encoding="utf-8"
            )
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
            ).strip()
            env = {
                **os.environ,
                "CI_CONTRACT_CHANGED": "true",
                "CLASSIFICATION_REASON": "single_product_tier_with_ci_contract",
                "EXPECTED_HEAD": head,
                "FORECASTING_TIER": "manual_prediction",
                "GITHUB_RUN_ID": run_id,
                "GITHUB_SHA": head,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "RUNNER_TEMP": temp_dir,
                "SUITE_OUTCOME": "success",
            }
            subprocess.run([sys.executable, "-c", source], cwd=ROOT, env=env, check=True)
            attestation = json.loads(
                (evidence_dir / "forecasting-ci-attestation.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(attestation["schema_version"], "forecasting-ci-attestation-v3")
        self.assertIs(attestation["ci_contract_changed"], True)
        self.assertEqual(attestation["uv"], "uv-test 0.0")
        self.assertEqual(
            [command["name"] for command in attestation["commands"]],
            ["ci_contract", "manual_prediction"],
        )
        for command in attestation["commands"]:
            self.assertEqual(command["commit"], head)
            self.assertEqual(command["tree"], tree)
            self.assertEqual(len(command["log_sha256"]), 64)

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
        self.assertIn('"tests/test_manual_independent_capture_sealer.py"', runner)
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
