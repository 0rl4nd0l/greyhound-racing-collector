"""Run the complete Forecasting validation, including every focused tier."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMMANDS = (
    (
        sys.executable,
        "scripts/ci/run_forecasting_ci_contract.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--noconftest",
        "tests/race_collection",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--noconftest",
        "tests/test_manual_independent_capture.py",
        "tests/test_manual_independent_capture_executor.py",
        "tests/test_manual_independent_capture_sealer.py",
        "tests/test_manual_research_scoring.py",
        "tests/test_manual_research_cli.py",
        "tests/test_market_form_residual.py",
        "tests/test_market_form_residual_portability.py",
        "tests/test_predict_market_form_residual.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--noconftest",
        "tests/test_predict_race_now.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_results_ingest_official_first.py",
        "tests/test_autonomous_official_result_capture.py",
        "tests/test_expert_form_official_result_labels_packet.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/operator_ui/test_foundation.py",
        "tests/operator_ui/test_job_store.py",
        "tests/operator_ui/test_prediction_worker.py",
        "tests/operator_ui/test_security.py",
        "tests/operator_ui/test_r3_api.py",
        "tests/operator_ui/test_r3_e2e_safety.py",
        "tests/operator_ui/test_connected_ui.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/operator_ui/test_api.py",
        "tests/operator_ui/test_bootstrap.py",
        "tests/operator_ui/test_deployment_generator.py",
        "tests/operator_ui/test_live_adapters.py",
    ),
)


def main() -> int:
    for command in COMMANDS:
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
