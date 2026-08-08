from __future__ import annotations

import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "classify_backend_changes",
    ROOT / "scripts/ci/classify_backend_changes.py",
)
assert SPEC and SPEC.loader
CLASSIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLASSIFIER
SPEC.loader.exec_module(CLASSIFIER)


class BackendChangeClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = CLASSIFIER.load_rules()

    def classify(self, *paths: str) -> dict:
        return CLASSIFIER.classify_changes(
            [CLASSIFIER.Change(status="M", paths=(path,)) for path in paths],
            self.rules,
        )

    def test_rules_are_explicit_and_nonempty(self):
        self.assertEqual(
            set(self.rules), {"backend_excluded", "ui_only", "ui_backend"}
        )
        self.assertTrue(self.rules["backend_excluded"])
        self.assertTrue(self.rules["ui_only"])
        self.assertTrue(self.rules["ui_backend"])

    def test_ghu058_style_manual_prediction_paths_skip_backend_and_ui(self):
        result = self.classify(
            ".github/forecasting-paths.ini",
            ".github/workflows/forecasting-tests.yml",
            "configs/prediction/manual-readiness-v1/scoring-readiness.schema.json",
            "docs/manual_prediction_scoring_readiness.md",
            "race_collection/manual_scoring_readiness.py",
            "race_collection/synchronous_manual_capture.py",
            "scripts/ci/classify_forecasting_changes.py",
            "scripts/ci/run_forecasting_ci_contract.py",
            "tests/ci/test_forecasting_change_classifier.py",
            "tests/race_collection/test_manual_scoring_readiness.py",
        )
        self.assertEqual(
            (result["backend_required"], result["ui_e2e_required"]),
            (False, False),
        )
        self.assertTrue(result["trusted"])

    def test_docs_only_skips_expensive_jobs(self):
        result = self.classify("docs/forecasting_ci_tiers.md", "README.md")
        self.assertEqual(
            (result["backend_required"], result["ui_e2e_required"]),
            (False, False),
        )

    def test_manual_prediction_only_paths_skip_backend(self):
        result = self.classify(
            "race_collection/manual_prediction_collector_request.py",
            "scripts/predict_race_now.py",
            "src/predictor/on_demand.py",
            "tests/test_predict_race_now.py",
            "tests/test_prediction_bundle_sealed.py",
        )
        self.assertEqual(
            (result["backend_required"], result["ui_e2e_required"]),
            (False, False),
        )

    def test_backend_risk_paths_run_backend_without_unrelated_ui(self):
        for path in (
            "app.py",
            "alembic/versions/123_add_column.py",
            "migrations/add_db_meta_table.py",
            "requirements/requirements.lock",
            "tests/test_flask_api.py",
            "tests/test_database_integrity.py",
            "src/shared_backend_helper.py",
        ):
            with self.subTest(path=path):
                result = self.classify(path)
                self.assertTrue(result["backend_required"])
                self.assertEqual(result["ui_e2e_required"], path == "app.py")

    def test_ui_only_paths_run_only_ui_e2e(self):
        for path in (
            "static/css/operator-ui.css",
            "templates/operator_ui_connected.jinja",
            "cypress/e2e/test-helper-routes.cy.js",
            "tests/playwright/nav-dropdowns-hidden.spec.ts",
            "package-lock.json",
        ):
            with self.subTest(path=path):
                result = self.classify(path)
                self.assertEqual(
                    (result["backend_required"], result["ui_e2e_required"]),
                    (False, True),
                )

    def test_mixed_ui_and_backend_paths_run_both(self):
        result = self.classify("static/app.js", "app.py")
        self.assertEqual(
            (result["backend_required"], result["ui_e2e_required"]),
            (True, True),
        )

    def test_unknown_source_path_fails_closed_to_backend(self):
        result = self.classify("src/new_shared_runtime_module.py")
        self.assertTrue(result["backend_required"])
        self.assertFalse(result["ui_e2e_required"])
        self.assertEqual(result["reason"], "backend_risk_path_or_destructive_change")

    def test_workflow_change_runs_backend_and_ui_checks(self):
        result = self.classify(".github/workflows/backend-tests.yml")
        self.assertEqual(
            (result["backend_required"], result["ui_e2e_required"]),
            (True, True),
        )

    def test_destructive_changes_fail_closed(self):
        result = CLASSIFIER.classify_changes(
            [CLASSIFIER.Change(status="D", paths=("docs/old.md",))], self.rules
        )
        self.assertTrue(result["backend_required"])

    def test_malformed_changes_fail_closed(self):
        result = CLASSIFIER.classify_changes(
            [CLASSIFIER.Change(status="", paths=())], self.rules
        )
        self.assertFalse(result["trusted"])
        self.assertTrue(result["backend_required"])

    def test_all_pairs_of_known_safe_families_remain_safe(self):
        safe_paths = (
            "docs/example.md",
            "scripts/ci/example.py",
            "tests/ci/example.py",
            "scripts/predict_market_form_residual.py",
        )
        for left, right in itertools.combinations(safe_paths, 2):
            with self.subTest(left=left, right=right):
                result = self.classify(left, right)
                self.assertFalse(result["backend_required"])


if __name__ == "__main__":
    unittest.main()
