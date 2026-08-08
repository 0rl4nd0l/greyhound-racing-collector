"""Validate Forecasting CI routing without running a complete product suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"

SMOKE_NODES = (
    "tests/test_predict_market_form_residual.py::test_scores_exact_packet_deterministically",
    "tests/test_results_ingest_official_first.py::test_parse_sportsbet_result_text_extracts_top_four_boxes",
    "tests/race_collection/test_scheduled_forward_corpus.py::test_fixture_scheduled_capture_admits_once_and_exact_replay_is_byte_stable",
    "tests/operator_ui/test_foundation.py::test_valid_envelope_is_deterministic_finite_immutable_and_serializable",
    "tests/race_collection/test_phase3_forecasting.py::test_prediction_rejected_before_close_and_binds_exact_seal_and_pin",
)
FORECASTING_WORKFLOW = WORKFLOWS / "forecasting-tests.yml"


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _validate_focused_checkout_history(parsed: object) -> None:
    if not isinstance(parsed, dict):
        raise SystemExit("forecasting workflow root must be a mapping")
    jobs = parsed.get("jobs")
    if not isinstance(jobs, dict):
        raise SystemExit("forecasting workflow jobs must be a mapping")
    gate = jobs.get("forecasting-gate")
    if not isinstance(gate, dict):
        raise SystemExit("forecasting-gate job is missing")
    steps = gate.get("steps")
    if not isinstance(steps, list):
        raise SystemExit("forecasting-gate steps must be a list")
    checkout_steps = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("name") == "Check out exact validation head"
    ]
    if len(checkout_steps) != 1:
        raise SystemExit(
            "forecasting-gate must contain exactly one exact-head checkout step"
        )
    checkout_with = checkout_steps[0].get("with")
    if not isinstance(checkout_with, dict):
        raise SystemExit("forecasting-gate exact-head checkout is missing with")
    if checkout_with.get("fetch-depth") != "0":
        raise SystemExit(
            "forecasting-gate exact-head checkout must fetch base history"
        )


def _validate_workflow_yaml() -> None:
    workflow_files = sorted(WORKFLOWS.glob("*.yml")) + sorted(
        WORKFLOWS.glob("*.yaml")
    )
    if not workflow_files:
        raise SystemExit("no workflow YAML files found")
    for workflow in workflow_files:
        try:
            parsed = yaml.load(workflow.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        except yaml.YAMLError as exc:
            raise SystemExit(f"invalid workflow YAML: {workflow}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit(f"workflow root must be a mapping: {workflow}")
        if workflow == FORECASTING_WORKFLOW:
            _validate_focused_checkout_history(parsed)
    print(f"validated {len(workflow_files)} workflow YAML files", flush=True)


def main() -> int:
    _run([sys.executable, "tests/ci/test_forecasting_change_classifier.py"])
    _validate_workflow_yaml()
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--noconftest",
            *SMOKE_NODES,
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
