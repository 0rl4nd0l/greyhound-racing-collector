#!/usr/bin/env python3
"""
ML Accuracy Test with Synthetic Data
====================================

This script tests the ML System V4 prediction accuracy using synthetic race data
when historical data with outcomes is not available. It focuses on testing:
• Model pipeline integrity and functionality
• Prediction consistency and calibration
• Feature engineering pipeline
• Model confidence metrics

"""

import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# Import our ML System V4
from ml_system_v4 import MLSystemV4

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_accuracy_test_synthetic.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class SyntheticMLTester:
    """Test ML System V4 using synthetic race data"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.ml_system = MLSystemV4(db_path)
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "test_type": "synthetic_data",
            "database_path": db_path,
            "total_races_tested": 0,
            "total_predictions": 0,
            "pipeline_tests": {},
            "consistency_tests": {},
            "calibration_tests": {},
            "performance_tests": {},
        }

        logger.info(f"🔧 Initializing Synthetic ML Tester")
        logger.info(f"   Database: {self.db_path}")

    def generate_synthetic_race(self, race_id: str, num_dogs: int = 8) -> pd.DataFrame:
        """Generate synthetic race data for testing"""

        # Common venue/track data
        venues = [
            "ALBION_PARK",
            "ANGLE_PARK",
            "BALLARAT",
            "BENDIGO",
            "BULLI",
            "CANNINGTON",
        ]
        grades = ["5", "4", "3", "M", "FFA"]
        distances = [515, 461, 395, 520, 600, 715]

        np.random.seed(hash(race_id) % 2**32)  # Deterministic but varied per race

        race_data = []
        for box_num in range(1, num_dogs + 1):
            dog_data = {
                "race_id": race_id,
                "dog_name": f"TEST_DOG_{box_num}",
                "dog_clean_name": f"TEST_DOG_{box_num}",
                "box_number": box_num,
                "weight": np.random.normal(32.0, 3.0),  # kg
                "trainer": f"TRAINER_{np.random.randint(1, 20)}",
                "odds": np.random.exponential(4.0) + 1.0,  # Starting price
                "venue": np.random.choice(venues),
                "race_date": (
                    datetime.now() - timedelta(days=np.random.randint(30, 90))
                ).strftime("%Y-%m-%d"),
                "distance": np.random.choice(distances),
                "grade": np.random.choice(grades),
                "track_condition": np.random.choice(["Good", "Slow", "Heavy"]),
                "weather": np.random.choice(["Fine", "Showery", "Overcast"]),
                "field_size": num_dogs,
                "race_time": "14:30",
            }
            race_data.append(dog_data)

        return pd.DataFrame(race_data)

    def test_prediction_pipeline(self, num_test_races: int = 50) -> Dict[str, Any]:
        """Test the prediction pipeline with synthetic data"""
        logger.info(
            f"🧪 Testing prediction pipeline with {num_test_races} synthetic races"
        )

        pipeline_results = {
            "successful_predictions": 0,
            "failed_predictions": 0,
            "prediction_times": [],
            "probability_sums": [],
            "prediction_consistency": {},
            "field_size_performance": {},
            "error_types": {},
        }

        failed_races = []

        for race_num in range(num_test_races):
            race_id = f"SYNTHETIC_RACE_{race_num:03d}"
            field_size = np.random.randint(4, 11)  # 4-10 dogs

            try:
                start_time = time.time()

                # Generate synthetic race
                race_data = self.generate_synthetic_race(race_id, field_size)

                # Make prediction
                result = self.ml_system.predict_race(race_data, race_id)

                prediction_time = time.time() - start_time
                pipeline_results["prediction_times"].append(prediction_time)

                if result.get("success", False):
                    pipeline_results["successful_predictions"] += 1

                    predictions = result.get("predictions", [])
                    if predictions:
                        # Check probability sum
                        prob_sum = sum(p.get("win_probability", 0) for p in predictions)
                        pipeline_results["probability_sums"].append(prob_sum)

                        # Track field size performance
                        if field_size not in pipeline_results["field_size_performance"]:
                            pipeline_results["field_size_performance"][field_size] = {
                                "count": 0,
                                "success_rate": 0,
                                "avg_prediction_time": 0,
                            }

                        pipeline_results["field_size_performance"][field_size][
                            "count"
                        ] += 1
                        pipeline_results["field_size_performance"][field_size][
                            "avg_prediction_time"
                        ] = (
                            pipeline_results["field_size_performance"][field_size][
                                "avg_prediction_time"
                            ]
                            * (
                                pipeline_results["field_size_performance"][field_size][
                                    "count"
                                ]
                                - 1
                            )
                            + prediction_time
                        ) / pipeline_results[
                            "field_size_performance"
                        ][
                            field_size
                        ][
                            "count"
                        ]

                        logger.debug(
                            f"✅ Race {race_id}: {len(predictions)} predictions, sum={prob_sum:.3f}"
                        )
                    else:
                        logger.warning(f"No predictions returned for {race_id}")
                        failed_races.append((race_id, "no_predictions"))
                else:
                    pipeline_results["failed_predictions"] += 1
                    error_msg = result.get("error", "Unknown error")
                    error_type = result.get("fallback_reason", "unknown")

                    if error_type not in pipeline_results["error_types"]:
                        pipeline_results["error_types"][error_type] = 0
                    pipeline_results["error_types"][error_type] += 1

                    failed_races.append((race_id, error_msg))
                    logger.warning(f"Failed to predict {race_id}: {error_msg}")

            except Exception as e:
                pipeline_results["failed_predictions"] += 1
                failed_races.append((race_id, str(e)))
                logger.error(f"Exception in race {race_id}: {e}")

        # Calculate success rates
        total_tests = (
            pipeline_results["successful_predictions"]
            + pipeline_results["failed_predictions"]
        )
        pipeline_results["success_rate"] = (
            pipeline_results["successful_predictions"] / total_tests
            if total_tests > 0
            else 0
        )

        # Calculate field size success rates
        for field_size, stats in pipeline_results["field_size_performance"].items():
            field_total = (
                sum(1 for rid, _ in failed_races if "RACE_" in rid) + stats["count"]
            )
            stats["success_rate"] = (
                stats["count"] / field_total if field_total > 0 else 0
            )

        # Statistics on prediction times and probability sums
        if pipeline_results["prediction_times"]:
            pipeline_results["avg_prediction_time"] = np.mean(
                pipeline_results["prediction_times"]
            )
            pipeline_results["max_prediction_time"] = np.max(
                pipeline_results["prediction_times"]
            )

        if pipeline_results["probability_sums"]:
            pipeline_results["avg_probability_sum"] = np.mean(
                pipeline_results["probability_sums"]
            )
            pipeline_results["probability_sum_std"] = np.std(
                pipeline_results["probability_sums"]
            )
            pipeline_results["well_normalized_races"] = sum(
                1 for ps in pipeline_results["probability_sums"] if 0.95 <= ps <= 1.05
            )

        pipeline_results["failed_races"] = failed_races

        logger.info(f"📊 Pipeline Test Results:")
        logger.info(
            f"   Success rate: {pipeline_results['success_rate']:.3f} ({pipeline_results['successful_predictions']}/{total_tests})"
        )
        logger.info(
            f"   Avg prediction time: {pipeline_results.get('avg_prediction_time', 0):.3f}s"
        )
        logger.info(
            f"   Well-normalized races: {pipeline_results.get('well_normalized_races', 0)}/{len(pipeline_results['probability_sums'])}"
        )

        return pipeline_results

    def test_prediction_consistency(self, num_tests: int = 10) -> Dict[str, Any]:
        """Test prediction consistency - same race should give same predictions"""
        logger.info(
            f"🔄 Testing prediction consistency with {num_tests} repeated predictions"
        )

        consistency_results = {
            "tests_run": 0,
            "consistent_predictions": 0,
            "max_probability_deviation": 0,
            "avg_probability_deviation": 0,
            "ranking_consistency": 0,
        }

        # Generate one test race
        test_race_id = "CONSISTENCY_TEST_RACE"
        race_data = self.generate_synthetic_race(test_race_id, 6)

        predictions_list = []
        successful_tests = 0

        for test_num in range(num_tests):
            try:
                result = self.ml_system.predict_race(
                    race_data.copy(), f"{test_race_id}_{test_num}"
                )

                if result.get("success", False):
                    predictions = result.get("predictions", [])
                    if predictions:
                        # Extract probabilities in consistent order
                        prob_dict = {
                            p["dog_name"]: p["win_probability"] for p in predictions
                        }
                        predictions_list.append(prob_dict)
                        successful_tests += 1

            except Exception as e:
                logger.warning(f"Consistency test {test_num} failed: {e}")

        if len(predictions_list) >= 2:
            # Analyze consistency
            dog_names = list(predictions_list[0].keys())

            max_deviations = []
            avg_deviations = []
            ranking_matches = []

            for i in range(1, len(predictions_list)):
                deviations = []

                for dog in dog_names:
                    if dog in predictions_list[i]:
                        deviation = abs(
                            predictions_list[0][dog] - predictions_list[i][dog]
                        )
                        deviations.append(deviation)

                if deviations:
                    max_deviations.append(max(deviations))
                    avg_deviations.append(np.mean(deviations))

                # Check ranking consistency
                rank_0 = sorted(
                    predictions_list[0].items(), key=lambda x: x[1], reverse=True
                )
                rank_i = sorted(
                    predictions_list[i].items(), key=lambda x: x[1], reverse=True
                )

                rank_match = sum(
                    1
                    for j in range(min(len(rank_0), len(rank_i)))
                    if rank_0[j][0] == rank_i[j][0]
                )
                ranking_matches.append(rank_match / len(dog_names))

            consistency_results.update(
                {
                    "tests_run": len(predictions_list),
                    "consistent_predictions": successful_tests,
                    "max_probability_deviation": (
                        float(np.mean(max_deviations)) if max_deviations else 0
                    ),
                    "avg_probability_deviation": (
                        float(np.mean(avg_deviations)) if avg_deviations else 0
                    ),
                    "ranking_consistency": (
                        float(np.mean(ranking_matches)) if ranking_matches else 0
                    ),
                }
            )

        logger.info(f"🔄 Consistency Results:")
        logger.info(
            f"   Successful tests: {consistency_results['consistent_predictions']}/{consistency_results['tests_run']}"
        )
        logger.info(
            f"   Max probability deviation: {consistency_results['max_probability_deviation']:.6f}"
        )
        logger.info(
            f"   Ranking consistency: {consistency_results['ranking_consistency']:.3f}"
        )

        return consistency_results

    def test_model_calibration(self) -> Dict[str, Any]:
        """Test model calibration properties"""
        logger.info("📏 Testing model calibration properties")

        calibration_results = {
            "probability_range_tests": {},
            "field_size_normalization": {},
            "extreme_case_handling": {},
            "model_info_validation": {},
        }

        # Test 1: Probability ranges
        test_cases = [("small_field", 4), ("medium_field", 6), ("large_field", 10)]

        for case_name, field_size in test_cases:
            race_id = f"CALIBRATION_TEST_{case_name.upper()}"
            race_data = self.generate_synthetic_race(race_id, field_size)

            try:
                result = self.ml_system.predict_race(race_data, race_id)

                if result.get("success", False):
                    predictions = result.get("predictions", [])

                    if predictions:
                        probs = [p["win_probability"] for p in predictions]

                        calibration_results["probability_range_tests"][case_name] = {
                            "field_size": field_size,
                            "min_probability": float(min(probs)),
                            "max_probability": float(max(probs)),
                            "probability_sum": float(sum(probs)),
                            "probability_std": float(np.std(probs)),
                            "expected_baseline": 1.0 / field_size,
                            "baseline_deviation": float(
                                abs(np.mean(probs) - (1.0 / field_size))
                            ),
                        }

            except Exception as e:
                calibration_results["probability_range_tests"][case_name] = {
                    "error": str(e)
                }

        # Test 2: Model info validation
        if hasattr(self.ml_system, "model_info") and self.ml_system.model_info:
            model_info = self.ml_system.model_info
            calibration_results["model_info_validation"] = {
                "model_type": model_info.get("model_type", "unknown"),
                "has_calibration": "calibration_method" in model_info,
                "calibration_method": model_info.get("calibration_method", "none"),
                "n_features": model_info.get("n_features", 0),
                "test_auc": model_info.get("test_auc", 0),
                "model_age_days": None,
            }

            # Calculate model age if trained_at is available
            if "trained_at" in model_info:
                try:
                    trained_at = datetime.fromisoformat(model_info["trained_at"])
                    age = (datetime.now() - trained_at).days
                    calibration_results["model_info_validation"]["model_age_days"] = age
                except Exception:
                    pass

        logger.info(f"📏 Calibration test completed for {len(test_cases)} field sizes")

        return calibration_results

    def run_comprehensive_synthetic_test(self) -> Dict[str, Any]:
        """Run all synthetic tests"""
        logger.info("🚀 Starting comprehensive synthetic ML test")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Test 1: Pipeline functionality
            logger.info("📋 Running pipeline tests...")
            self.results["pipeline_tests"] = self.test_prediction_pipeline(50)

            # Test 2: Prediction consistency
            logger.info("🔄 Running consistency tests...")
            self.results["consistency_tests"] = self.test_prediction_consistency(10)

            # Test 3: Model calibration
            logger.info("📏 Running calibration tests...")
            self.results["calibration_tests"] = self.test_model_calibration()

            # Summary statistics
            test_duration = time.time() - start_time
            self.results["test_duration_seconds"] = test_duration
            self.results["success"] = True

            # Calculate aggregate metrics
            pipeline = self.results["pipeline_tests"]
            consistency = self.results["consistency_tests"]

            self.results["total_races_tested"] = pipeline.get(
                "successful_predictions", 0
            )
            self.results["total_predictions"] = sum(
                fs["count"]
                for fs in pipeline.get("field_size_performance", {}).values()
            )

            logger.info("✅ Comprehensive synthetic test completed")
            logger.info(f"   Duration: {test_duration:.1f} seconds")
            logger.info(
                f"   Pipeline success rate: {pipeline.get('success_rate', 0):.3f}"
            )
            logger.info(
                f"   Consistency score: {consistency.get('ranking_consistency', 0):.3f}"
            )

            return self.results

        except Exception as e:
            logger.error(f"❌ Comprehensive synthetic test failed: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")

            self.results["success"] = False
            self.results["error"] = str(e)
            self.results["test_duration_seconds"] = time.time() - start_time

            return self.results

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report"""
        if not self.results.get("success", False):
            return f"❌ Test Failed: {self.results.get('error', 'Unknown error')}"

        pipeline = self.results.get("pipeline_tests", {})
        consistency = self.results.get("consistency_tests", {})
        calibration = self.results.get("calibration_tests", {})

        report = f"""
ML System V4 - Synthetic Accuracy Test Report
============================================
Test Date: {self.results['test_timestamp']}
Test Type: Synthetic Data Testing
Duration: {self.results.get('test_duration_seconds', 0):.1f} seconds

🧪 Pipeline Functionality Tests:
• Success Rate: {pipeline.get('success_rate', 0):.3f} ({pipeline.get('successful_predictions', 0)}/{pipeline.get('successful_predictions', 0) + pipeline.get('failed_predictions', 0)})
• Average Prediction Time: {pipeline.get('avg_prediction_time', 0):.3f} seconds
• Well-Normalized Predictions: {pipeline.get('well_normalized_races', 0)}/{len(pipeline.get('probability_sums', []))}
• Average Probability Sum: {pipeline.get('avg_probability_sum', 0):.4f}

🔄 Consistency Tests:
• Prediction Consistency: {consistency.get('consistent_predictions', 0)}/{consistency.get('tests_run', 0)}
• Max Probability Deviation: {consistency.get('max_probability_deviation', 0):.6f}
• Average Probability Deviation: {consistency.get('avg_probability_deviation', 0):.6f}
• Ranking Consistency: {consistency.get('ranking_consistency', 0):.3f}

📏 Model Calibration:
"""

        # Add probability range results
        prob_tests = calibration.get("probability_range_tests", {})
        for test_name, results in prob_tests.items():
            if "error" not in results:
                report += f"• {test_name.replace('_', ' ').title()}: "
                report += f"Sum={results.get('probability_sum', 0):.3f}, "
                report += f"Range=[{results.get('min_probability', 0):.3f}-{results.get('max_probability', 0):.3f}]\n"

        # Add model info
        model_info = calibration.get("model_info_validation", {})
        if model_info:
            report += f"""
🤖 Model Information:
• Model Type: {model_info.get('model_type', 'Unknown')}
• Calibration: {model_info.get('calibration_method', 'None')}
• Features: {model_info.get('n_features', 0)}
• Test AUC: {model_info.get('test_auc', 0):.4f}
• Model Age: {model_info.get('model_age_days', 'Unknown')} days
"""

        # Assessment
        success_rate = pipeline.get("success_rate", 0)
        consistency_score = consistency.get("ranking_consistency", 0)
        avg_prob_sum = pipeline.get("avg_probability_sum", 0)

        report += f"""
⭐ Overall Assessment:
"""

        if (
            success_rate >= 0.95
            and consistency_score >= 0.95
            and 0.98 <= avg_prob_sum <= 1.02
        ):
            assessment = "✅ EXCELLENT - Pipeline is working perfectly"
        elif (
            success_rate >= 0.90
            and consistency_score >= 0.90
            and 0.95 <= avg_prob_sum <= 1.05
        ):
            assessment = "🟢 GOOD - Pipeline is functioning well"
        elif success_rate >= 0.80 and consistency_score >= 0.80:
            assessment = "🟡 FAIR - Pipeline has some issues to address"
        else:
            assessment = "🔴 POOR - Pipeline needs significant fixes"

        report += f"• {assessment}\n"
        report += f"• Pipeline is ready for production use: {'Yes' if success_rate >= 0.95 else 'No'}\n"

        # Error analysis if present
        error_types = pipeline.get("error_types", {})
        if error_types:
            report += f"\n🚨 Error Analysis:\n"
            for error_type, count in error_types.items():
                report += f"• {error_type}: {count} occurrences\n"

        return report

    def save_results(self, filepath: str = None) -> str:
        """Save test results to JSON file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ml_synthetic_test_results_{timestamp}.json"

        try:
            with open(filepath, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            logger.info(f"💾 Results saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""


def main():
    """Main execution function"""
    logger.info("🎬 Starting Synthetic ML Accuracy Test V4")

    tester = SyntheticMLTester()
    results = tester.run_comprehensive_synthetic_test()

    # Save results
    results_file = tester.save_results()

    # Generate and display summary
    summary = tester.generate_summary_report()
    print("\n" + summary)

    # Write summary to file
    summary_file = results_file.replace(".json", "_summary.txt")
    try:
        with open(summary_file, "w") as f:
            f.write(summary)
        logger.info(f"📄 Summary report saved to: {summary_file}")
    except Exception as e:
        logger.warning(f"Could not save summary report: {e}")

    return results


if __name__ == "__main__":
    results = main()

    if results.get("success", False):
        logger.info("🎉 Synthetic ML accuracy test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Synthetic ML accuracy test failed!")
        sys.exit(1)
