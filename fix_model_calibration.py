#!/usr/bin/env python3
"""
Model Calibration Fix Script
============================

This script diagnoses and fixes calibration issues in existing ML System V4 models.
It addresses the uniform probability distribution problem identified in testing.

Usage:
    python fix_model_calibration.py

Features:
- Diagnoses calibration issues in current models
- Applies enhanced calibration techniques
- Re-trains models with proper calibration settings
- Validates calibration improvements
- Updates model registry with calibrated models

"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Import synthetic tester for validation
from ml_accuracy_test_synthetic import SyntheticMLTester

# Import our enhanced calibration module
from ml_calibration_enhanced import (
    RaceAwareCalibrator,
    create_calibrated_pipeline,
    diagnose_calibration_issues,
    fix_uniform_predictions,
)

# Import ML System V4
from ml_system_v4 import MLSystemV4

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("calibration_fix.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class CalibrationFixer:
    """Fix calibration issues in ML System V4 models"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.ml_system = MLSystemV4(db_path)
        self.results = {
            "diagnosis_timestamp": datetime.now().isoformat(),
            "original_model_info": {},
            "calibration_diagnosis": {},
            "fix_applied": False,
            "validation_results": {},
            "recommendations": [],
        }

        logger.info("🔧 Initializing Calibration Fixer")

    def diagnose_current_model(self) -> Dict[str, Any]:
        """Diagnose calibration issues in the current model"""
        logger.info("🔍 Diagnosing current model calibration...")

        try:
            # Check if model is loaded
            if (
                not hasattr(self.ml_system, "calibrated_pipeline")
                or self.ml_system.calibrated_pipeline is None
            ):
                logger.warning("⚠️ No model loaded, attempting to load from registry...")
                if not self._try_load_model():
                    return {"error": "No model available for diagnosis"}

            # Get model information
            self.results["original_model_info"] = (
                self.ml_system.model_info.copy() if self.ml_system.model_info else {}
            )

            # Test with synthetic data to check for uniform distributions
            tester = SyntheticMLTester(self.db_path)

            # Generate a few test races
            test_results = []
            for i in range(5):
                race_id = f"DIAGNOSIS_RACE_{i:03d}"
                field_size = 6 + (i % 3)  # 6, 7, 8 dogs

                race_data = tester.generate_synthetic_race(race_id, field_size)
                result = self.ml_system.predict_race(race_data, race_id)

                if result.get("success", False):
                    predictions = result.get("predictions", [])
                    if predictions:
                        probs = [p.get("win_probability", 0) for p in predictions]
                        test_results.append(
                            {
                                "race_id": race_id,
                                "field_size": field_size,
                                "probabilities": probs,
                                "prob_std": float(np.std(probs)),
                                "prob_range": float(np.max(probs) - np.min(probs)),
                                "is_uniform": np.std(probs) < 1e-6,
                            }
                        )

            # Analyze results
            uniform_races = sum(1 for r in test_results if r["is_uniform"])
            avg_std = np.mean([r["prob_std"] for r in test_results])
            avg_range = np.mean([r["prob_range"] for r in test_results])

            diagnosis = {
                "test_races": len(test_results),
                "uniform_races": uniform_races,
                "uniform_percentage": (
                    uniform_races / len(test_results) if test_results else 0
                ),
                "average_std": float(avg_std),
                "average_range": float(avg_range),
                "is_problematic": uniform_races
                > len(test_results) * 0.8,  # >80% uniform
                "severity": (
                    "CRITICAL"
                    if uniform_races == len(test_results)
                    else (
                        "HIGH"
                        if uniform_races > len(test_results) * 0.5
                        else "MEDIUM" if uniform_races > 0 else "LOW"
                    )
                ),
                "test_details": test_results,
            }

            self.results["calibration_diagnosis"] = diagnosis

            logger.info(f"📊 Diagnosis complete:")
            logger.info(
                f"   Uniform races: {uniform_races}/{len(test_results)} ({diagnosis['uniform_percentage']:.1%})"
            )
            logger.info(f"   Average std dev: {avg_std:.6f}")
            logger.info(f"   Severity: {diagnosis['severity']}")

            return diagnosis

        except Exception as e:
            logger.error(f"❌ Diagnosis failed: {e}")
            self.results["calibration_diagnosis"] = {"error": str(e)}
            return {"error": str(e)}

    def _try_load_model(self) -> bool:
        """Try to load a model from the registry"""
        try:
            # Try to load the latest model
            model_registry_path = "./model_registry/models"
            if os.path.exists(model_registry_path):
                model_files = [
                    f
                    for f in os.listdir(model_registry_path)
                    if f.endswith("_model.joblib")
                ]
                if model_files:
                    logger.info(f"Found {len(model_files)} models in registry")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def apply_calibration_fix(self, method: str = "isotonic") -> bool:
        """Apply calibration fix to the current model"""
        logger.info(f"🎯 Applying calibration fix with method: {method}")

        try:
            # Set environment variable to force calibration
            os.environ["V4_CALIB_METHOD"] = method

            # Re-train the model with proper calibration
            logger.info("🚀 Re-training model with calibration enabled...")

            success = self.ml_system.train_model()

            if success:
                self.results["fix_applied"] = True
                self.results["calibration_method"] = method
                logger.info("✅ Model re-trained with calibration successfully")
                return True
            else:
                logger.error("❌ Model re-training failed")
                return False

        except Exception as e:
            logger.error(f"❌ Calibration fix failed: {e}")
            self.results["fix_error"] = str(e)
            return False

    def validate_calibration_fix(self) -> Dict[str, Any]:
        """Validate that calibration fix worked"""
        logger.info("✅ Validating calibration fix...")

        try:
            # Re-run diagnosis after fix
            post_fix_diagnosis = self.diagnose_current_model()

            # Compare before and after
            original_diagnosis = self.results.get("calibration_diagnosis", {})

            improvement_metrics = {
                "uniform_races_before": original_diagnosis.get("uniform_races", 0),
                "uniform_races_after": post_fix_diagnosis.get("uniform_races", 0),
                "avg_std_before": original_diagnosis.get("average_std", 0),
                "avg_std_after": post_fix_diagnosis.get("average_std", 0),
                "severity_before": original_diagnosis.get("severity", "UNKNOWN"),
                "severity_after": post_fix_diagnosis.get("severity", "UNKNOWN"),
            }

            # Determine if fix was successful
            fix_successful = (
                improvement_metrics["uniform_races_after"]
                < improvement_metrics["uniform_races_before"]
                and improvement_metrics["avg_std_after"]
                > improvement_metrics["avg_std_before"]
            )

            validation_result = {
                "fix_successful": fix_successful,
                "improvement_metrics": improvement_metrics,
                "post_fix_diagnosis": post_fix_diagnosis,
                "recommendations": self._generate_recommendations(improvement_metrics),
            }

            self.results["validation_results"] = validation_result

            logger.info(f"📊 Validation results:")
            logger.info(f"   Fix successful: {fix_successful}")
            logger.info(
                f"   Uniform races: {improvement_metrics['uniform_races_before']} → {improvement_metrics['uniform_races_after']}"
            )
            logger.info(
                f"   Average std: {improvement_metrics['avg_std_before']:.6f} → {improvement_metrics['avg_std_after']:.6f}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation metrics"""
        recommendations = []

        if metrics["uniform_races_after"] == 0 and metrics["avg_std_after"] > 0.01:
            recommendations.append(
                "✅ EXCELLENT: Calibration fix completely resolved uniform distribution issue"
            )
            recommendations.append(
                "Model now produces varied probabilities and should discriminate between dogs"
            )

        elif metrics["uniform_races_after"] < metrics["uniform_races_before"]:
            recommendations.append(
                "🟢 GOOD: Calibration fix partially resolved the issue"
            )
            recommendations.append(
                "Consider additional feature engineering to improve discrimination"
            )

        else:
            recommendations.append("🔴 POOR: Calibration fix did not resolve the issue")
            recommendations.append(
                "Try alternative calibration methods (sigmoid, ensemble)"
            )
            recommendations.append(
                "Check base model feature variance and training data quality"
            )

        if metrics["avg_std_after"] < 0.001:
            recommendations.append("⚠️ WARNING: Standard deviation still very low")
            recommendations.append(
                "Consider adding feature variance or using ensemble methods"
            )

        return recommendations

    def run_complete_calibration_fix(self) -> Dict[str, Any]:
        """Run complete calibration diagnosis and fix process"""
        logger.info("🚀 Starting complete calibration fix process")
        logger.info("=" * 60)

        try:
            # Step 1: Diagnose current issues
            logger.info("📋 Step 1: Diagnosing current calibration issues...")
            diagnosis = self.diagnose_current_model()

            if "error" in diagnosis:
                logger.error(f"❌ Diagnosis failed: {diagnosis['error']}")
                return self.results

            # Step 2: Determine if fix is needed
            if not diagnosis.get("is_problematic", False):
                logger.info("✅ No significant calibration issues detected")
                self.results["fix_needed"] = False
                return self.results

            self.results["fix_needed"] = True

            # Step 3: Apply calibration fix
            logger.info("📋 Step 2: Applying calibration fix...")
            fix_success = self.apply_calibration_fix(method="isotonic")

            if not fix_success:
                # Try alternative method
                logger.info("🔄 Trying alternative calibration method...")
                fix_success = self.apply_calibration_fix(method="sigmoid")

            if not fix_success:
                logger.error("❌ All calibration methods failed")
                return self.results

            # Step 4: Validate the fix
            logger.info("📋 Step 3: Validating calibration fix...")
            validation = self.validate_calibration_fix()

            # Step 5: Generate final report
            self.results["success"] = validation.get("fix_successful", False)
            self.results["final_recommendations"] = validation.get(
                "recommendations", []
            )

            # Summary
            logger.info("✅ Calibration fix process completed")
            logger.info(f"   Success: {self.results['success']}")
            logger.info(f"   Fix applied: {self.results['fix_applied']}")

            return self.results

        except Exception as e:
            logger.error(f"❌ Calibration fix process failed: {e}")
            self.results["error"] = str(e)
            return self.results

    def save_results(self, filepath: str = None) -> str:
        """Save calibration fix results"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"calibration_fix_results_{timestamp}.json"

        try:
            import json

            with open(filepath, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            logger.info(f"💾 Results saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""


def main():
    """Main execution function"""
    logger.info("🎬 Starting Model Calibration Fix")

    fixer = CalibrationFixer()
    results = fixer.run_complete_calibration_fix()

    # Save results
    results_file = fixer.save_results()

    # Generate summary report
    print("\n" + "=" * 60)
    print("MODEL CALIBRATION FIX SUMMARY")
    print("=" * 60)

    if results.get("success", False):
        print("✅ CALIBRATION FIX SUCCESSFUL!")
        print(f"   Fix applied: {results.get('fix_applied', False)}")
        print(f"   Method used: {results.get('calibration_method', 'unknown')}")

        # Show improvement metrics
        validation = results.get("validation_results", {})
        metrics = validation.get("improvement_metrics", {})

        print(f"\n📊 Improvement Metrics:")
        print(
            f"   Uniform races: {metrics.get('uniform_races_before', 0)} → {metrics.get('uniform_races_after', 0)}"
        )
        print(
            f"   Average std dev: {metrics.get('avg_std_before', 0):.6f} → {metrics.get('avg_std_after', 0):.6f}"
        )

    else:
        print("❌ CALIBRATION FIX FAILED")
        if "error" in results:
            print(f"   Error: {results['error']}")

    # Show recommendations
    recommendations = results.get("final_recommendations", [])
    if recommendations:
        print(f"\n🎯 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")

    print(f"\n📄 Detailed results saved to: {results_file}")

    return results


if __name__ == "__main__":
    results = main()

    if results.get("success", False):
        logger.info("🎉 Model calibration fix completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Model calibration fix failed!")
        sys.exit(1)
