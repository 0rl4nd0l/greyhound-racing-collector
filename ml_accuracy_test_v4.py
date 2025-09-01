#!/usr/bin/env python3
"""
ML Accuracy Test V4 - Comprehensive Performance Evaluation
==========================================================

This script provides comprehensive testing of the ML System V4 prediction accuracy:
• Load real historical race data with winners
• Test model predictions vs actual outcomes
• Calculate calibration, AUC, win rate accuracy
• Analyze prediction confidence and expected value
• Generate detailed performance report

"""

import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# Import our ML System V4
from ml_system_v4 import MLSystemV4

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_accuracy_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class MLAccuracyTester:
    """Comprehensive accuracy testing for ML System V4"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.ml_system = MLSystemV4(db_path)
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "database_path": db_path,
            "total_races_tested": 0,
            "total_predictions": 0,
            "metrics": {},
            "calibration_analysis": {},
            "confidence_analysis": {},
            "field_size_analysis": {},
            "venue_analysis": {},
            "prediction_examples": [],
        }

        logger.info(f"🔧 Initializing ML Accuracy Tester")
        logger.info(f"   Database: {self.db_path}")

    def load_test_data(
        self, limit_races: int = 200, days_back: int = 90
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load recent historical race data for testing"""
        logger.info(
            f"📊 Loading test data: {limit_races} races from last {days_back} days"
        )

        try:
            conn = sqlite3.connect(self.db_path)

            # Get recent races with complete data
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )

            query = """
            SELECT DISTINCT
                dp.race_id,
                rm.venue,
                rm.race_date,
                rm.distance,
                rm.grade,
                rm.track_condition,
                rm.weather,
                rm.field_size,
                COUNT(dp.dog_name) as actual_field_size
            FROM dog_performances dp
            JOIN race_metadata rm ON dp.race_id = rm.race_id
            WHERE dp.finish_position IS NOT NULL 
                AND rm.race_date >= ?
                AND dp.finish_position BETWEEN 1 AND 12
            GROUP BY dp.race_id
            HAVING actual_field_size >= 4 AND actual_field_size <= 11
            ORDER BY rm.race_date DESC
            LIMIT ?
            """

            race_info = pd.read_sql_query(
                query, conn, params=(cutoff_date, limit_races)
            )

            if race_info.empty:
                logger.error("No suitable test races found")
                return pd.DataFrame(), pd.DataFrame()

            logger.info(f"✅ Found {len(race_info)} suitable races for testing")

            # Get detailed performance data for these races
            race_ids = tuple(race_info["race_id"].tolist())
            placeholders = ",".join("?" * len(race_ids))

            detailed_query = f"""
            SELECT 
                dp.race_id,
                dp.dog_name,
                dp.box_number,
                dp.finish_position,
                dp.race_time,
                dp.weight,
                dp.trainer,
                dp.odds,
                rm.venue,
                rm.race_date,
                rm.distance,
                rm.grade,
                COALESCE(rm.track_condition, 'Good') as track_condition,
                COALESCE(rm.weather, 'Fine') as weather,
                rm.field_size,
                CASE WHEN dp.finish_position = 1 THEN 1 ELSE 0 END as won
            FROM dog_performances dp
            JOIN race_metadata rm ON dp.race_id = rm.race_id
            WHERE dp.race_id IN ({placeholders})
                AND dp.finish_position IS NOT NULL
                AND dp.finish_position BETWEEN 1 AND 12
            ORDER BY dp.race_id, dp.box_number
            """

            detailed_data = pd.read_sql_query(detailed_query, conn, params=race_ids)
            conn.close()

            logger.info(
                f"✅ Loaded {len(detailed_data)} dog performances across {len(race_info)} races"
            )
            logger.info(
                f"   Date range: {detailed_data['race_date'].min()} to {detailed_data['race_date'].max()}"
            )
            logger.info(
                f"   Winners: {detailed_data['won'].sum()}, Total dogs: {len(detailed_data)}"
            )

            return race_info, detailed_data

        except Exception as e:
            logger.error(f"❌ Failed to load test data: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return pd.DataFrame(), pd.DataFrame()

    def test_model_predictions(
        self, race_info: pd.DataFrame, detailed_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Test model predictions against actual outcomes"""
        logger.info("🎯 Testing model predictions against actual outcomes")

        predictions_list = []
        actuals_list = []
        race_results = []
        failed_races = []

        for _, race_row in race_info.iterrows():
            race_id = race_row["race_id"]

            try:
                # Get race data for this race
                race_data = detailed_data[detailed_data["race_id"] == race_id].copy()

                if len(race_data) < 4:  # Skip races with too few dogs
                    continue

                # Prepare race data for prediction (remove outcome data)
                prediction_data = race_data.drop(
                    ["finish_position", "won"], axis=1, errors="ignore"
                ).copy()

                # Make predictions
                logger.debug(
                    f"Predicting race {race_id} with {len(prediction_data)} dogs"
                )

                result = self.ml_system.predict_race(prediction_data, race_id)

                if not result.get("success", False):
                    logger.warning(
                        f"Failed to predict race {race_id}: {result.get('error', 'Unknown error')}"
                    )
                    failed_races.append(race_id)
                    continue

                predictions = result.get("predictions", [])
                if not predictions:
                    logger.warning(f"No predictions returned for race {race_id}")
                    failed_races.append(race_id)
                    continue

                # Match predictions with actual outcomes
                race_predictions = []
                race_actuals = []

                for pred in predictions:
                    dog_name = pred.get("dog_name", pred.get("dog_clean_name", ""))

                    # Find actual outcome for this dog
                    actual_row = race_data[
                        race_data["dog_name"].str.upper() == str(dog_name).upper()
                    ]

                    if len(actual_row) == 1:
                        win_prob = pred.get(
                            "win_probability", pred.get("win_prob_norm", 0)
                        )
                        actual_won = int(actual_row.iloc[0]["won"])

                        predictions_list.append(float(win_prob))
                        actuals_list.append(actual_won)

                        race_predictions.append(
                            {
                                "dog_name": dog_name,
                                "predicted_prob": float(win_prob),
                                "predicted_rank": pred.get("predicted_rank", 0),
                                "confidence": pred.get("confidence", 0),
                                "actual_won": actual_won,
                                "actual_position": int(
                                    actual_row.iloc[0]["finish_position"]
                                ),
                            }
                        )

                if race_predictions:
                    race_results.append(
                        {
                            "race_id": race_id,
                            "venue": race_row["venue"],
                            "race_date": race_row["race_date"],
                            "field_size": len(race_predictions),
                            "predictions": race_predictions,
                            "winner_predicted_correctly": any(
                                p["predicted_rank"] == 1 and p["actual_won"] == 1
                                for p in race_predictions
                            ),
                            "winner_in_top_3": any(
                                p["predicted_rank"] <= 3 and p["actual_won"] == 1
                                for p in race_predictions
                            ),
                        }
                    )

                logger.debug(
                    f"✅ Processed race {race_id}: {len(race_predictions)} predictions"
                )

            except Exception as e:
                logger.error(f"Error processing race {race_id}: {e}")
                failed_races.append(race_id)
                continue

        if not predictions_list:
            logger.error("No valid predictions to analyze")
            return {}

        # Convert to numpy arrays for analysis
        y_pred = np.array(predictions_list)
        y_true = np.array(actuals_list)

        logger.info(
            f"✅ Collected {len(predictions_list)} predictions from {len(race_results)} races"
        )
        logger.info(f"   Failed races: {len(failed_races)}")

        return {
            "predictions": y_pred,
            "actuals": y_true,
            "race_results": race_results,
            "failed_races": failed_races,
        }

    def calculate_metrics(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive prediction metrics"""
        logger.info("📈 Calculating prediction metrics")

        try:
            # Basic metrics
            auc = roc_auc_score(y_true, y_pred)
            brier_score = brier_score_loss(y_true, y_pred)
            log_loss_score = log_loss(y_true, y_pred, eps=1e-15)

            # Threshold-based metrics
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_true, y_pred_binary)

            # Win rate accuracy (how often we pick the actual winner)
            # This requires race-level analysis

            # Calibration analysis
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
            ece = np.mean(np.abs(prob_true - prob_pred))  # Expected Calibration Error

            # Distribution analysis
            pred_mean = float(np.mean(y_pred))
            pred_std = float(np.std(y_pred))
            actual_win_rate = float(np.mean(y_true))

            metrics = {
                "auc": float(auc),
                "brier_score": float(brier_score),
                "log_loss": float(log_loss_score),
                "accuracy_threshold_05": float(accuracy),
                "expected_calibration_error": float(ece),
                "prediction_mean": pred_mean,
                "prediction_std": pred_std,
                "actual_win_rate": actual_win_rate,
                "calibration_slope": (
                    float(np.corrcoef(prob_pred, prob_true)[0, 1])
                    if len(prob_pred) > 1
                    else 0.0
                ),
            }

            logger.info(f"📊 Core Metrics:")
            logger.info(f"   AUC: {auc:.4f}")
            logger.info(f"   Brier Score: {brier_score:.4f}")
            logger.info(f"   Log Loss: {log_loss_score:.4f}")
            logger.info(f"   Calibration Error: {ece:.4f}")
            logger.info(f"   Actual win rate: {actual_win_rate:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def analyze_race_level_performance(
        self, race_results: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze performance at the race level"""
        logger.info("🏁 Analyzing race-level performance")

        total_races = len(race_results)
        winner_predicted_correctly = sum(
            1 for r in race_results if r["winner_predicted_correctly"]
        )
        winner_in_top_3 = sum(1 for r in race_results if r["winner_in_top_3"])

        # Field size analysis
        field_size_stats = defaultdict(list)
        for race in race_results:
            field_size = race["field_size"]
            field_size_stats[field_size].append(
                {
                    "winner_correct": race["winner_predicted_correctly"],
                    "winner_top3": race["winner_in_top_3"],
                }
            )

        field_analysis = {}
        for size, results in field_size_stats.items():
            if results:
                field_analysis[f"field_size_{size}"] = {
                    "races": len(results),
                    "winner_accuracy": sum(r["winner_correct"] for r in results)
                    / len(results),
                    "top3_accuracy": sum(r["winner_top3"] for r in results)
                    / len(results),
                }

        # Venue analysis
        venue_stats = defaultdict(list)
        for race in race_results:
            venue = race.get("venue", "Unknown")
            venue_stats[venue].append(
                {
                    "winner_correct": race["winner_predicted_correctly"],
                    "winner_top3": race["winner_in_top_3"],
                }
            )

        venue_analysis = {}
        for venue, results in venue_stats.items():
            if len(results) >= 5:  # Only analyze venues with sufficient data
                venue_analysis[venue] = {
                    "races": len(results),
                    "winner_accuracy": sum(r["winner_correct"] for r in results)
                    / len(results),
                    "top3_accuracy": sum(r["winner_top3"] for r in results)
                    / len(results),
                }

        race_analysis = {
            "total_races": total_races,
            "winner_predicted_exactly": winner_predicted_correctly,
            "winner_accuracy": (
                winner_predicted_correctly / total_races if total_races > 0 else 0
            ),
            "winner_in_top3": winner_in_top_3,
            "top3_accuracy": winner_in_top_3 / total_races if total_races > 0 else 0,
            "field_size_breakdown": field_analysis,
            "venue_breakdown": venue_analysis,
        }

        logger.info(f"🏆 Race-level Performance:")
        logger.info(
            f"   Winner accuracy: {race_analysis['winner_accuracy']:.3f} ({winner_predicted_correctly}/{total_races})"
        )
        logger.info(
            f"   Top-3 accuracy: {race_analysis['top3_accuracy']:.3f} ({winner_in_top_3}/{total_races})"
        )

        return race_analysis

    def analyze_confidence_calibration(
        self, race_results: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze confidence calibration"""
        logger.info("🎯 Analyzing confidence calibration")

        confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        bin_analysis = {}

        all_confidences = []
        all_accuracies = []

        for race in race_results:
            for pred in race["predictions"]:
                confidence = pred["confidence"]
                accuracy = 1.0 if pred["actual_won"] else 0.0

                all_confidences.append(confidence)
                all_accuracies.append(accuracy)

        # Bin analysis
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i + 1]

            mask = (np.array(all_confidences) >= low) & (
                np.array(all_confidences) < high
            )
            if i == len(confidence_bins) - 2:  # Last bin includes upper bound
                mask = (np.array(all_confidences) >= low) & (
                    np.array(all_confidences) <= high
                )

            if mask.sum() > 0:
                bin_accuracies = np.array(all_accuracies)[mask]
                bin_analysis[f"confidence_{low}_{high}"] = {
                    "count": int(mask.sum()),
                    "mean_confidence": float(np.mean(np.array(all_confidences)[mask])),
                    "actual_accuracy": float(np.mean(bin_accuracies)),
                    "calibration_gap": float(
                        np.mean(bin_accuracies)
                        - np.mean(np.array(all_confidences)[mask])
                    ),
                }

        confidence_analysis = {
            "bins": bin_analysis,
            "overall_confidence_mean": float(np.mean(all_confidences)),
            "overall_accuracy_mean": float(np.mean(all_accuracies)),
            "confidence_accuracy_correlation": float(
                np.corrcoef(all_confidences, all_accuracies)[0, 1]
            ),
        }

        logger.info(f"📊 Confidence Analysis:")
        logger.info(
            f"   Mean confidence: {confidence_analysis['overall_confidence_mean']:.3f}"
        )
        logger.info(
            f"   Mean accuracy: {confidence_analysis['overall_accuracy_mean']:.3f}"
        )
        logger.info(
            f"   Confidence-accuracy correlation: {confidence_analysis['confidence_accuracy_correlation']:.3f}"
        )

        return confidence_analysis

    def run_comprehensive_test(self, limit_races: int = 200) -> Dict[str, Any]:
        """Run comprehensive accuracy test"""
        logger.info("🚀 Starting comprehensive ML accuracy test")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Load test data
            race_info, detailed_data = self.load_test_data(limit_races)

            if race_info.empty or detailed_data.empty:
                logger.error("No test data available")
                return {"success": False, "error": "No test data available"}

            self.results["total_races_available"] = len(race_info)
            self.results["total_performances"] = len(detailed_data)

            # Test predictions
            prediction_results = self.test_model_predictions(race_info, detailed_data)

            if not prediction_results:
                logger.error("No prediction results available")
                return {"success": False, "error": "Prediction testing failed"}

            y_pred = prediction_results["predictions"]
            y_true = prediction_results["actuals"]
            race_results = prediction_results["race_results"]

            self.results["total_races_tested"] = len(race_results)
            self.results["total_predictions"] = len(y_pred)
            self.results["failed_races"] = prediction_results["failed_races"]

            # Calculate metrics
            self.results["metrics"] = self.calculate_metrics(y_pred, y_true)

            # Race-level analysis
            self.results["race_level_analysis"] = self.analyze_race_level_performance(
                race_results
            )

            # Confidence calibration analysis
            self.results["confidence_analysis"] = self.analyze_confidence_calibration(
                race_results
            )

            # Store examples for review
            self.results["prediction_examples"] = race_results[
                :5
            ]  # First 5 races as examples

            # Performance summary
            test_duration = time.time() - start_time
            self.results["test_duration_seconds"] = test_duration
            self.results["success"] = True

            logger.info("✅ Comprehensive accuracy test completed")
            logger.info(f"   Duration: {test_duration:.1f} seconds")
            logger.info(f"   Races tested: {len(race_results)}")
            logger.info(f"   Predictions analyzed: {len(y_pred)}")

            return self.results

        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")

            self.results["success"] = False
            self.results["error"] = str(e)
            self.results["test_duration_seconds"] = time.time() - start_time

            return self.results

    def save_results(self, filepath: str = None) -> str:
        """Save test results to JSON file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ml_accuracy_test_results_{timestamp}.json"

        try:
            with open(filepath, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            logger.info(f"💾 Results saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report"""
        if not self.results.get("success", False):
            return f"❌ Test Failed: {self.results.get('error', 'Unknown error')}"

        metrics = self.results.get("metrics", {})
        race_analysis = self.results.get("race_level_analysis", {})
        confidence_analysis = self.results.get("confidence_analysis", {})

        report = f"""
ML System V4 - Accuracy Test Report
===================================
Test Date: {self.results['test_timestamp']}
Duration: {self.results.get('test_duration_seconds', 0):.1f} seconds

📊 Test Coverage:
• Races Available: {self.results.get('total_races_available', 0)}
• Races Successfully Tested: {self.results.get('total_races_tested', 0)}
• Total Predictions: {self.results.get('total_predictions', 0)}
• Failed Races: {len(self.results.get('failed_races', []))}

🎯 Core Performance Metrics:
• AUC (Area Under Curve): {metrics.get('auc', 0):.4f}
• Brier Score (lower is better): {metrics.get('brier_score', 0):.4f}
• Log Loss (lower is better): {metrics.get('log_loss', 0):.4f}
• Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.4f}

🏆 Race-Level Accuracy:
• Winner Predicted Exactly: {race_analysis.get('winner_accuracy', 0):.3f} ({race_analysis.get('winner_predicted_exactly', 0)}/{race_analysis.get('total_races', 0)})
• Winner in Top 3 Predictions: {race_analysis.get('top3_accuracy', 0):.3f} ({race_analysis.get('winner_in_top3', 0)}/{race_analysis.get('total_races', 0)})

📈 Calibration Quality:
• Mean Predicted Probability: {metrics.get('prediction_mean', 0):.4f}
• Actual Win Rate: {metrics.get('actual_win_rate', 0):.4f}
• Confidence-Accuracy Correlation: {confidence_analysis.get('confidence_accuracy_correlation', 0):.3f}

🔍 Field Size Performance:
"""

        field_breakdown = race_analysis.get("field_size_breakdown", {})
        for size_key, stats in sorted(field_breakdown.items()):
            field_size = size_key.replace("field_size_", "")
            report += f"• {field_size} dogs: Winner accuracy {stats['winner_accuracy']:.3f} ({stats['races']} races)\n"

        report += f"""
⭐ Overall Assessment:
"""

        # Assessment logic
        auc = metrics.get("auc", 0)
        winner_acc = race_analysis.get("winner_accuracy", 0)
        ece = metrics.get("expected_calibration_error", 0)

        if auc >= 0.65 and winner_acc >= 0.15 and ece <= 0.1:
            assessment = "✅ EXCELLENT - Model shows strong predictive power"
        elif auc >= 0.60 and winner_acc >= 0.12 and ece <= 0.15:
            assessment = "🟢 GOOD - Model performs well above baseline"
        elif auc >= 0.55 and winner_acc >= 0.08:
            assessment = "🟡 FAIR - Model shows some predictive ability"
        else:
            assessment = "🔴 POOR - Model needs improvement"

        report += f"• {assessment}\n"
        report += f"• Baseline expectation (random): ~{1/8:.3f} winner accuracy for 8-dog races\n"

        return report


def main():
    """Main execution function"""
    logger.info("🎬 Starting ML Accuracy Test V4")

    # Check if we're in test mode (skip heavy operations)
    if os.getenv("TESTING", "0").lower() in ("1", "true", "yes"):
        logger.info("🧪 Running in test mode - using lightweight operations")
        limit_races = 10
    else:
        limit_races = 200

    tester = MLAccuracyTester()
    results = tester.run_comprehensive_test(limit_races=limit_races)

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
        logger.info("🎉 ML accuracy test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ ML accuracy test failed!")
        sys.exit(1)
