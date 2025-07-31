#!/usr/bin/env python3
"""
Greyhound Racing Data Pipeline Orchestrator
===========================================

This script orchestrates the entire data collection, processing, and analysis pipeline
to automate the workflow and ensure proper data flow between components.

Author: AI Assistant
Date: July 26, 2025
"""

import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline_orchestrator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete data pipeline workflow"""

    def __init__(self):
        self.base_dir = Path(".")
        self.unprocessed_dir = self.base_dir / "unprocessed"
        self.processed_dir = self.base_dir / "processed"
        self.upcoming_dir = self.base_dir / "upcoming_races"
        self.predictions_dir = self.base_dir / "predictions"
        self.database_path = "greyhound_racing_data.db"

        # Create directories
        for directory in [
            self.unprocessed_dir,
            self.processed_dir,
            self.upcoming_dir,
            self.predictions_dir,
        ]:
            directory.mkdir(exist_ok=True)

        # Pipeline statistics
        self.stats = {
            "start_time": datetime.now(),
            "races_collected": 0,
            "races_processed": 0,
            "predictions_generated": 0,
            "errors": [],
        }

        logger.info("🚀 Pipeline Orchestrator initialized")

    def get_database_stats(self) -> Dict[str, int]:
        """Get current database statistics"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            total_dogs = cursor.fetchone()[0]

            conn.close()

            return {"total_races": total_races, "total_dogs": total_dogs}
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"total_races": 0, "total_dogs": 0}

    def collect_historical_races(self, days_back: int = 7) -> bool:
        """Collect historical race data for the past N days"""
        logger.info(f"📊 Starting historical race collection ({days_back} days back)")

        try:
            # Run form guide scraper for historical data
            result = subprocess.run(
                [sys.executable, "form_guide_csv_scraper.py"],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                logger.info("✅ Historical race collection completed")
                self.stats["races_collected"] += self._count_new_files()
                return True
            else:
                error_msg = f"Historical collection failed: {result.stderr[:200]}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
                return False

        except subprocess.TimeoutExpired:
            error_msg = "Historical collection timed out (10 minutes)"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False
        except Exception as e:
            error_msg = f"Historical collection error: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False

    def collect_upcoming_races(self) -> bool:
        """Collect upcoming race information"""
        logger.info("🏁 Starting upcoming race collection")

        try:
            from upcoming_race_browser import UpcomingRaceBrowser

            browser = UpcomingRaceBrowser()

            # Get races for next 2 days
            races = browser.get_upcoming_races(days_ahead=2)

            logger.info(f"✅ Found {len(races)} upcoming races")

            # Save race information for later processing
            races_file = (
                self.upcoming_dir
                / f"upcoming_races_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(races_file, "w") as f:
                json.dump(races, f, indent=2)

            return True

        except Exception as e:
            error_msg = f"Upcoming race collection error: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False

    def _count_new_files(self) -> int:
        """Count new CSV files in unprocessed directory"""
        try:
            return len([f for f in self.unprocessed_dir.glob("*.csv")])
        except:
            return 0

    def process_new_data(self) -> bool:
        """Process any new CSV files in unprocessed directory"""
        logger.info("⚙️ Starting data processing")

        # Check for new files
        csv_files = list(self.unprocessed_dir.glob("*.csv"))
        if not csv_files:
            logger.info("ℹ️ No new files to process")
            return True

        logger.info(f"📈 Processing {len(csv_files)} new files")

        try:
            # Get pre-processing database stats
            pre_stats = self.get_database_stats()

            # Run enhanced comprehensive processor
            result = subprocess.run(
                [sys.executable, "enhanced_comprehensive_processor.py"],
                capture_output=True,
                text=True,
                timeout=1800,
            )  # 30 minute timeout

            if result.returncode == 0:
                # Get post-processing database stats
                post_stats = self.get_database_stats()

                races_added = post_stats["total_races"] - pre_stats["total_races"]
                dogs_added = post_stats["total_dogs"] - pre_stats["total_dogs"]

                logger.info(f"✅ Data processing completed")
                logger.info(f"   📊 Added {races_added} races, {dogs_added} dogs")

                self.stats["races_processed"] += races_added
                return True
            else:
                error_msg = f"Data processing failed: {result.stderr[:200]}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
                return False

        except subprocess.TimeoutExpired:
            error_msg = "Data processing timed out (30 minutes)"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False
        except Exception as e:
            error_msg = f"Data processing error: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False

    def generate_predictions(self) -> bool:
        """Generate predictions for upcoming races"""
        logger.info("🎯 Starting prediction generation")

        try:
            # Check if we have upcoming race files
            upcoming_files = list(self.upcoming_dir.glob("*.csv"))
            if not upcoming_files:
                logger.info("ℹ️ No upcoming race files for predictions")
                return True

            # Use unified predictor if available
            if os.path.exists("unified_predictor.py"):
                logger.info("🧠 Using unified predictor system")

                from unified_predictor import UnifiedPredictor

                predictor = UnifiedPredictor()

                successful_predictions = 0
                for race_file in upcoming_files:
                    try:
                        result = predictor.predict_race_file(str(race_file))
                        if result.get("success"):
                            successful_predictions += 1
                            logger.info(f"✅ Predicted: {race_file.name}")
                        else:
                            logger.warning(
                                f"⚠️ Prediction failed for {race_file.name}: {result.get('error')}"
                            )
                    except Exception as e:
                        logger.error(f"❌ Error predicting {race_file.name}: {e}")

                self.stats["predictions_generated"] = successful_predictions
                logger.info(
                    f"🏁 Generated {successful_predictions}/{len(upcoming_files)} predictions"
                )
                return successful_predictions > 0

            else:
                logger.warning("⚠️ No prediction system available")
                return False

        except Exception as e:
            error_msg = f"Prediction generation error: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False

    def run_data_validation(self) -> bool:
        """Run data validation checks"""
        logger.info("🔍 Running data validation")

        try:
            db_stats = self.get_database_stats()

            # Basic validation checks
            issues = []

            if db_stats["total_races"] < 5:
                issues.append(
                    f"Low race count: {db_stats['total_races']} (minimum 5 recommended)"
                )

            if db_stats["total_dogs"] < db_stats["total_races"] * 3:
                issues.append(
                    f"Low dog-to-race ratio: {db_stats['total_dogs']}/{db_stats['total_races']}"
                )

            # Check for orphaned records
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT COUNT(*) FROM dog_race_data d
                    WHERE NOT EXISTS (
                        SELECT 1 FROM race_metadata r WHERE r.race_id = d.race_id
                    )
                """
                )
                orphaned_dogs = cursor.fetchone()[0]

                if orphaned_dogs > 0:
                    issues.append(f"Found {orphaned_dogs} orphaned dog records")

                conn.close()

            except Exception as e:
                issues.append(f"Database validation error: {e}")

            if issues:
                for issue in issues:
                    logger.warning(f"⚠️ Validation issue: {issue}")
                return False
            else:
                logger.info("✅ Data validation passed")
                return True

        except Exception as e:
            error_msg = f"Data validation error: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False

    def run_full_pipeline(self, collect_historical: bool = True) -> Dict[str, Any]:
        """Run the complete data pipeline"""
        logger.info("🚀 Starting full pipeline execution")

        # Step 1: Collect historical data (if requested)
        if collect_historical:
            logger.info("📈 Step 1: Collecting historical race data")
            historical_success = self.collect_historical_races()
        else:
            logger.info("⏭️ Step 1: Skipping historical collection")
            historical_success = True

        # Step 2: Collect upcoming races
        logger.info("🏁 Step 2: Collecting upcoming races")
        upcoming_success = self.collect_upcoming_races()

        # Step 3: Process new data
        logger.info("⚙️ Step 3: Processing new data")
        processing_success = self.process_new_data()

        # Step 4: Generate predictions
        logger.info("🎯 Step 4: Generating predictions")
        prediction_success = self.generate_predictions()

        # Step 5: Validate data
        logger.info("🔍 Step 5: Validating data quality")
        validation_success = self.run_data_validation()

        # Compile results
        self.stats["end_time"] = datetime.now()
        self.stats["duration"] = (
            self.stats["end_time"] - self.stats["start_time"]
        ).total_seconds()

        final_db_stats = self.get_database_stats()

        results = {
            "success": all(
                [
                    historical_success,
                    upcoming_success,
                    processing_success,
                    prediction_success,
                    validation_success,
                ]
            ),
            "steps": {
                "historical_collection": historical_success,
                "upcoming_collection": upcoming_success,
                "data_processing": processing_success,
                "prediction_generation": prediction_success,
                "data_validation": validation_success,
            },
            "statistics": self.stats,
            "database_stats": final_db_stats,
            "timestamp": datetime.now().isoformat(),
        }

        # Log final results
        if results["success"]:
            logger.info("🎉 Pipeline execution completed successfully!")
        else:
            logger.error("❌ Pipeline execution completed with errors")

        logger.info(
            f"📊 Final stats: {final_db_stats['total_races']} races, {final_db_stats['total_dogs']} dogs"
        )
        logger.info(f"⏱️ Duration: {self.stats['duration']:.1f} seconds")

        return results

    def run_quick_update(self) -> Dict[str, Any]:
        """Run a quick update without historical collection"""
        logger.info("⚡ Starting quick pipeline update")
        return self.run_full_pipeline(collect_historical=False)


def main():
    """Main entry point for pipeline orchestration"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Greyhound Racing Data Pipeline Orchestrator"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "historical", "process", "predict"],
        default="full",
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--days-back", type=int, default=7, help="Days back for historical collection"
    )

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator()

    try:
        if args.mode == "full":
            results = orchestrator.run_full_pipeline()
        elif args.mode == "quick":
            results = orchestrator.run_quick_update()
        elif args.mode == "historical":
            success = orchestrator.collect_historical_races(args.days_back)
            results = {"success": success, "mode": "historical_only"}
        elif args.mode == "process":
            success = orchestrator.process_new_data()
            results = {"success": success, "mode": "process_only"}
        elif args.mode == "predict":
            success = orchestrator.generate_predictions()
            results = {"success": success, "mode": "predict_only"}

        # Save results
        results_file = (
            f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"📄 Results saved to: {results_file}")

        # Exit with appropriate code
        sys.exit(0 if results.get("success", False) else 1)

    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
