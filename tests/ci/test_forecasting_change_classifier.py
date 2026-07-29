from __future__ import annotations

import html
import importlib.util
import re
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


def changes(*items: tuple[str, ...]):
    return [CLASSIFIER.Change(status=item[0], paths=tuple(item[1:])) for item in items]


class ForecastingChangeClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = CLASSIFIER.load_rules()

    def assert_selection(
        self,
        expected_tier: str,
        expected_suite: str,
        *items: tuple[str, ...],
    ):
        result = CLASSIFIER.classify_changes(changes(*items), self.rules)
        self.assertEqual(result["tier"], expected_tier)
        self.assertEqual(result["suite"], expected_suite)
        return result

    def test_representative_deterministic_matrix(self):
        fixtures = [
            ("ordinary-docs", "non_forecasting", "", ("M", "docs/guide.md")),
            (
                "on-demand-docs",
                "non_forecasting",
                "",
                ("M", "docs/on_demand_race_prediction.md"),
            ),
            ("frontend", "non_forecasting", "", ("M", "frontend/src/App.tsx")),
            ("ui", "non_forecasting", "", ("M", "ui/prediction/card.ts")),
            ("static", "non_forecasting", "", ("M", "static/site.css")),
            (
                "manual-source",
                "focused_forecasting",
                "manual_prediction",
                ("M", "scripts/predict_race_now.py"),
            ),
            (
                "official-results",
                "focused_forecasting",
                "official_results",
                ("M", "scripts/ingest_results_for_date.py"),
            ),
            (
                "inventory",
                "focused_forecasting",
                "race_collection_inventory",
                ("M", "race_collection/inventory.py"),
            ),
            (
                "two-focused-subsystems",
                "full_forecasting",
                "full_forecasting",
                ("M", "scripts/predict_race_now.py"),
                ("M", "scripts/ingest_results_for_date.py"),
            ),
            (
                "shared-domain",
                "full_forecasting",
                "full_forecasting",
                ("M", "race_collection/domain.py"),
            ),
            (
                "shared-artifacts",
                "full_forecasting",
                "full_forecasting",
                ("M", "race_collection/artifacts.py"),
            ),
            (
                "collector-protocol",
                "full_forecasting",
                "full_forecasting",
                ("M", "race_collection/manual_prediction_collector_request.py"),
            ),
            (
                "migration-schema",
                "full_forecasting",
                "full_forecasting",
                ("M", "migrations/001_forecasting.sql"),
            ),
            (
                "dependency",
                "full_forecasting",
                "full_forecasting",
                ("M", "requirements.lock"),
            ),
            (
                "rules-control",
                "full_forecasting",
                "full_forecasting",
                ("M", ".github/forecasting-paths.ini"),
            ),
            (
                "classifier-control",
                "full_forecasting",
                "full_forecasting",
                ("M", "scripts/ci/classify_forecasting_changes.py"),
            ),
            (
                "workflow-control",
                "full_forecasting",
                "full_forecasting",
                ("M", ".github/workflows/forecasting-tests.yml"),
            ),
            (
                "unknown-source",
                "full_forecasting",
                "full_forecasting",
                ("A", "new_subsystem/adapter.py"),
            ),
            (
                "literal-backslash",
                "full_forecasting",
                "full_forecasting",
                ("A", r"docs\forecasting.md"),
            ),
        ]
        for name, tier, suite, *items in fixtures:
            with self.subTest(name=name):
                result = self.assert_selection(tier, suite, *items)
                self.assertEqual(
                    [path["path"] for path in result["paths"]],
                    [path for item in items for path in item[1:]],
                )

    def test_every_docs_only_path_is_non_forecasting(self):
        focused_patterns = [
            pattern
            for group in self.rules
            if group.tier == "focused_forecasting"
            for pattern in group.patterns
        ]
        self.assertFalse(
            any(
                pattern == "docs" or pattern.startswith("docs/")
                for pattern in focused_patterns
            )
        )
        for path in (
            "docs/on_demand_race_prediction.md",
            "docs/manual_market_form_residual_prediction.md",
            "docs/manual_live_market_form_residual_prediction.md",
            "docs/FORECASTING_PUBLICATION_VALIDATION.md",
            "docs/forecasting_validation_logs/current.md",
            "docs/nested/prediction.md",
        ):
            with self.subTest(path=path):
                self.assert_selection("non_forecasting", "", ("M", path))

    def test_destructive_and_ambiguous_changes_force_full(self):
        for item in (
            ("R100", "scripts/predict_race_now.py", "archive/predict.py"),
            ("C100", "scripts/predict_race_now.py", "scripts/copy.py"),
            ("D", "scripts/predict_race_now.py"),
            ("T", "docs/guide.md"),
        ):
            with self.subTest(status=item[0]):
                result = self.assert_selection(
                    "full_forecasting", "full_forecasting", item
                )
                self.assertEqual(result["reason"], "unsafe_git_status_forces_full")

    def test_malformed_status_and_change_set_force_full(self):
        for values in (
            [],
            changes(("X", "docs/guide.md")),
            changes(("", "README.md")),
            changes(("M100", "scripts/predict_race_now.py")),
            changes(("R101", "docs/old.md", "docs/new.md")),
        ):
            with self.subTest(values=values):
                result = CLASSIFIER.classify_changes(values, self.rules)
                self.assertEqual(result["tier"], "full_forecasting")
                self.assertEqual(result["suite"], "full_forecasting")

    def test_rule_classification_is_distinct_from_escalated_selection(self):
        result = self.assert_selection(
            "full_forecasting",
            "full_forecasting",
            ("M", "scripts/predict_race_now.py"),
            ("M", "scripts/ingest_results_for_date.py"),
        )
        self.assertEqual(
            [item["rule_tier"] for item in result["paths"]],
            ["focused_forecasting", "focused_forecasting"],
        )
        self.assertEqual(
            [item["rule_suite"] for item in result["paths"]],
            ["manual_prediction", "official_results"],
        )
        self.assertEqual(result["reason"], "multiple_focused_subsystems_force_full")

    def test_force_full_selection(self):
        result = CLASSIFIER.force_full_result()
        self.assertEqual(result["tier"], "full_forecasting")
        self.assertEqual(result["suite"], "full_forecasting")
        self.assertEqual(result["reason"], "manual_dispatch_forces_full")

    def test_name_status_parser_keeps_exact_posix_filenames(self):
        parsed = CLASSIFIER.parse_name_status(
            b"M\0docs\\forecasting.md\0"
            b"R100\0scripts/predict_race_now.py\0archive/predict.py\0"
        )
        self.assertEqual(parsed[0].paths, (r"docs\forecasting.md",))
        self.assertEqual(len(parsed[1].paths), 2)
        with self.assertRaises(CLASSIFIER.ClassificationError):
            CLASSIFIER.parse_name_status(b"R100\0only-one-path\0")
        with self.assertRaises(CLASSIFIER.ClassificationError):
            CLASSIFIER.parse_name_status(b"M\0invalid-\xff\0")

    def test_configuration_is_fail_closed(self):
        original = Path(CLASSIFIER.DEFAULT_RULES).read_text(encoding="utf-8")
        malformed = {
            "missing-section": original.replace("[non_forecasting]", "[wrong]"),
            "unknown-suite": original.replace(
                "suite = manual_prediction", "suite = injected"
            ),
            "substituted-command": original.replace(
                "tests/test_predict_race_now.py", "tests/injected.py", 1
            ),
            "unknown-option": original.replace(
                "[metadata]", "[metadata]\nextra = value"
            ),
            "unsafe-pattern": original.replace("    docs/**", "    ../docs/**", 1),
        }
        for name, contents in malformed.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rules.ini"
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(CLASSIFIER.ClassificationError):
                    CLASSIFIER.load_rules(path)

    def test_summary_renders_every_dynamic_value_as_inert_exact_code(self):
        dynamic_values = [
            "M![status](https://example.invalid/status)",
            "odd\\name|`![image](https://example.invalid/image)<img src=x>\r\n.md",
            "tier [link](https://example.invalid/tier)",
            "suite <script>alert(1)</script>",
            "selected_tier ![image](https://example.invalid/selected)",
            "selected_suite [link](https://example.invalid/suite)",
            "command `unsafe` | <b>html</b>",
            "reason\\with\r\nnewlines",
        ]
        result = {
            "paths": [
                {
                    "status": dynamic_values[0],
                    "path": dynamic_values[1],
                    "rule_tier": dynamic_values[2],
                    "rule_suite": dynamic_values[3],
                }
            ],
            "tier": dynamic_values[4],
            "suite": dynamic_values[5],
            "command": dynamic_values[6],
            "reason": dynamic_values[7],
        }
        summary = CLASSIFIER.github_summary(result)
        encoded_values = re.findall(r"<code>((?:&#[0-9]+;)*)</code>", summary)
        self.assertEqual(
            [html.unescape(value) for value in encoded_values],
            dynamic_values,
        )
        for active_syntax in (
            "![image](",
            "[link](",
            "<img",
            "<script>",
            "`unsafe`",
            "reason\\with",
        ):
            self.assertNotIn(active_syntax, summary)
        self.assertIn("Rule classification", summary)
        self.assertIn("Effective trusted execution selection", summary)

    def test_github_outputs_include_only_trusted_selection(self):
        result = self.assert_selection(
            "focused_forecasting",
            "race_collection_inventory",
            ("M", "race_collection/inventory.py"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output"
            CLASSIFIER.write_github_output(path, result)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [
                    "tier=focused_forecasting",
                    "suite=race_collection_inventory",
                    f"reason={result['reason']}",
                ],
            )


if __name__ == "__main__":
    unittest.main()
