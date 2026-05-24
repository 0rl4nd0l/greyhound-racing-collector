#!/usr/bin/env python3
"""
Automation Controller for the Greyhound Racing ML Pipeline
=========================================================

This script orchestrates the entire ML pipeline, including:
1.  **Data Integrity Checks**: Validates and cleans the database.
2.  **Model Retraining**: Retrains the ML models if data has significantly changed.
3.  **Prediction Generation**: Generates predictions for upcoming races.
4.  **Reporting**: Creates reports on pipeline status and performance.

Author: AI Assistant
Date: 2025-01-28
"""

import json
import logging
import os

# (existing imports from data_integrity_system.py)
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


class AutomationController:
    """Manages the automated execution of the greyhound racing pipeline."""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.setup_logging()
        self.reports_dir = Path("reports")
        self.models_dir = Path("models")
        self.reports_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    def setup_logging(self):
        """Sets up a centralized logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/automation_controller.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def run_command(self, command: list) -> Tuple[bool, str, str]:
        """Executes a shell command and returns status, stdout, and stderr."""
        self.logger.info(f"Executing command: {' '.join(command)}")
        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            return True, process.stdout, process.stderr
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            self.logger.error(f"Stderr: {e.stderr}")
            return False, e.stdout, e.stderr

    def run_data_integrity_checks(self) -> Optional[Dict]:
        """Runs the data integrity system and returns the report."""
        self.logger.info("--- Starting Data Integrity Checks ---")
        # Reuse the DataIntegrityManager for this
        from data_integrity_system import DataIntegrityManager

        try:
            with DataIntegrityManager(self.db_path) as integrity_manager:
                report = integrity_manager.run_comprehensive_integrity_check()
                report_path = (
                    self.reports_dir
                    / f"integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)

                self.logger.info(f"Data integrity report saved to {report_path}")

                if report.get("issues_found"):
                    self.logger.warning(
                        "Data integrity issues found. Aborting further steps."
                    )
                    # In a real-world scenario, you might send an alert here
                    return None

                self.logger.info("Data integrity checks passed successfully.")
                return report
        except Exception as e:
            self.logger.error(f"An error occurred during data integrity checks: {e}")
            return None

    def check_for_new_data(self) -> bool:
        """Checks if there is new data since the last model training."""
        self.logger.info("--- Checking for New Data ---")
        model_path = self.models_dir / "advanced_ensemble_model.joblib"
        if not model_path.exists():
            self.logger.info("No existing model found. Retraining is required.")
            return True

        last_training_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        self.logger.info(f"Last model training time: {last_training_time}")

        # Check for new races since the last training
        conn = sqlite3.connect(self.db_path)
        try:
            # Use extraction_timestamp from race_metadata
            query = "SELECT COUNT(*) FROM race_metadata WHERE extraction_timestamp > ?"
            new_races_count = conn.execute(query, (last_training_time,)).fetchone()[0]
            self.logger.info(f"Found {new_races_count} new races since last training.")
            return new_races_count > 0
        finally:
            conn.close()

    def run_model_retraining(self):
        """Initiates the model retraining process."""
        self.logger.info("--- Starting Model Retraining ---")

        success, stdout, stderr = self.run_command(
            ["python", "advanced_ensemble_ml_system.py", "--train"]
        )

        if success:
            self.logger.info("Model retraining completed successfully.")
        else:
            self.logger.error("Model retraining failed.")
            self.logger.error(f"Stderr: {stderr}")

    def run_prediction_generation(self):
        """Generates predictions for upcoming races using the live prediction system."""
        self.logger.info("--- Starting Live Prediction Generation ---")

        try:
            from live_prediction_system import LivePredictionSystem

            live_system = LivePredictionSystem(self.db_path)
            live_system.run(max_races=3)  # Process up to 3 races
            self.logger.info("Live prediction generation completed successfully.")
        except Exception as e:
            self.logger.error(f"Live prediction generation failed: {e}")
            # Fallback to the original system
            self.logger.info("Falling back to original prediction system...")
            success, stdout, stderr = self.run_command(
                ["python", "prediction_orchestrator.py"]
            )

            if success:
                self.logger.info(
                    "Fallback prediction generation completed successfully."
                )
                self.logger.info(f"Output:\n{stdout}")
            else:
                self.logger.error("Fallback prediction generation also failed.")
                self.logger.error(f"Stderr: {stderr}")

    def start(self):
        """Main entry point to start the automation pipeline."""
        self.logger.info("===== Automation Pipeline Started =====")

        # 1. Data Integrity
        integrity_report = self.run_data_integrity_checks()
        if not integrity_report:
            self.logger.error("Pipeline halted due to data integrity issues.")
            return

        # 2. Check for new data and decide on retraining
        if self.check_for_new_data():
            self.run_model_retraining()
        else:
            self.logger.info(
                "No significant new data found. Skipping model retraining."
            )

        # 3. Generate predictions
        self.run_prediction_generation()

        self.logger.info("===== Automation Pipeline Finished =====")


if __name__ == "__main__":
    controller = AutomationController()
    controller.start()
