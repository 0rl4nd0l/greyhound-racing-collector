from __future__ import annotations

import importlib.util
import sys
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

    def test_fixture_tiers(self):
        fixtures = [
            (
                "manual-only",
                [
                    ("M", "scripts/predict_race_now.py"),
                    ("M", "configs/prediction/manual-default.json"),
                    ("M", "docs/on_demand_race_prediction.md"),
                    ("M", "tests/test_predict_race_now.py"),
                ],
                "manual_prediction",
            ),
            (
                "core-only",
                [("M", "race_collection/forecasting.py")],
                "full_forecasting",
            ),
            (
                "mixed",
                [
                    ("M", "scripts/predict_race_now.py"),
                    ("M", "race_collection/source_admission.py"),
                ],
                "full_forecasting",
            ),
            (
                "workflow-only",
                [("M", ".github/workflows/forecasting-tests.yml")],
                "full_forecasting",
            ),
            (
                "forecasting-contract-doc",
                [("M", "docs/FORECASTING_PUBLICATION_VALIDATION.md")],
                "full_forecasting",
            ),
            (
                "docs-only",
                [("M", "docs/race_evidence_inventory.md")],
                "non_forecasting",
            ),
            (
                "rename",
                [
                    (
                        "R100",
                        "scripts/predict_race_now.py",
                        "archive/manual_predictor.py",
                    )
                ],
                "full_forecasting",
            ),
            (
                "delete",
                [("D", "race_collection/model_bundle.py")],
                "full_forecasting",
            ),
            (
                "unknown",
                [("A", "new_subsystem/adapter.py")],
                "full_forecasting",
            ),
            (
                "generated-dependency",
                [("M", "requirements.lock")],
                "full_forecasting",
            ),
            (
                "shared-import",
                [("M", "utils/csv_metadata.py")],
                "full_forecasting",
            ),
        ]
        for name, changes, expected in fixtures:
            with self.subTest(name=name):
                values = [
                    CLASSIFIER.Change(status=change[0], paths=tuple(change[1:]))
                    for change in changes
                ]
                self.assertEqual(
                    CLASSIFIER.classify_changes(values, self.rules)["tier"],
                    expected,
                )

    def test_overlap_selects_broader_tier(self):
        result = CLASSIFIER.classify_changes(
            [
                CLASSIFIER.Change(
                    status="M", paths=("docs/on_demand_race_prediction.md",)
                )
            ],
            self.rules,
        )
        self.assertEqual(result["tier"], "manual_prediction")
        self.assertEqual(
            result["paths"][0]["matched_tiers"],
            ["non_forecasting", "manual_prediction"],
        )

    def test_empty_or_unknown_status_defaults_to_full(self):
        self.assertEqual(
            CLASSIFIER.classify_changes([], self.rules)["tier"], "full_forecasting"
        )
        for status in ("?", "X"):
            with self.subTest(status=status):
                self.assertEqual(
                    CLASSIFIER.classify_changes(
                        [CLASSIFIER.Change(status=status, paths=("README.md",))],
                        self.rules,
                    )["tier"],
                    "full_forecasting",
                )

    def test_name_status_parser_keeps_both_rename_paths(self):
        changes = CLASSIFIER.parse_name_status(
            b"R100\0scripts/predict_race_now.py\0archive/manual_predictor.py\0"
            b"D\0race_collection/source_admission.py\0"
        )
        self.assertEqual(
            changes,
            [
                CLASSIFIER.Change(
                    status="R100",
                    paths=(
                        "scripts/predict_race_now.py",
                        "archive/manual_predictor.py",
                    ),
                ),
                CLASSIFIER.Change(
                    status="D", paths=("race_collection/source_admission.py",)
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
