#!/usr/bin/env python3
"""
Comprehensive Sanity Checks Demo & Integration
=============================================

This script demonstrates the complete workflow:
1. Validate predictions
2. Fix inconsistencies automatically 
3. Integrate with existing prediction pipeline
4. Process real prediction files
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List

from sanity_checks import SanityChecks

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PredictionProcessor:
    """
    Complete prediction processing pipeline with sanity checks and auto-fixes.
    """

    def __init__(self):
        self.sanity_checker = SanityChecks()
        self.processed_count = 0
        self.fixed_count = 0

    def process_predictions(
        self, predictions: List[Dict], auto_fix: bool = True
    ) -> Dict:
        """
        Process predictions with validation and optional auto-fixing.

        Args:
            predictions: List of prediction dictionaries
            auto_fix: Whether to automatically fix issues

        Returns:
            Dictionary with original, validation results, and fixed predictions
        """
        result = {
            "original_predictions": copy.deepcopy(predictions),
            "validation_results": {},
            "fixed_predictions": None,
            "fixes_applied": False,
            "processing_summary": {},
        }

        # Step 1: Validate original predictions
        logger.info(f"🔍 Validating {len(predictions)} predictions...")
        validation_results = self.sanity_checker.validate_predictions(predictions)
        result["validation_results"] = validation_results

        # Log validation results
        if validation_results["flags"]:
            logger.warning(f"⚠️  Found {len(validation_results['flags'])} issues:")
            for flag in validation_results["flags"]:
                logger.warning(f"   - {flag}")
        else:
            logger.info("✅ All validations passed!")

        # Step 2: Apply fixes if needed and requested
        if validation_results["flags"] and auto_fix:
            logger.info("🔧 Applying automatic fixes...")
            fixed_predictions = self.sanity_checker.fix_predictions(predictions)
            result["fixed_predictions"] = fixed_predictions
            result["fixes_applied"] = True
            self.fixed_count += 1

            # Validate fixed predictions
            logger.info("🔍 Re-validating fixed predictions...")
            fixed_validation = self.sanity_checker.validate_predictions(
                fixed_predictions
            )
            result["fixed_validation_results"] = fixed_validation

            if fixed_validation["flags"]:
                logger.warning(
                    f"⚠️  {len(fixed_validation['flags'])} issues remain after fixes"
                )
            else:
                logger.info("✅ All issues resolved!")

        self.processed_count += 1

        # Step 3: Generate processing summary
        result["processing_summary"] = {
            "total_predictions": len(predictions),
            "issues_found": len(validation_results["flags"]),
            "fixes_applied": result["fixes_applied"],
            "issues_remaining": len(
                result.get("fixed_validation_results", {}).get("flags", [])
            ),
            "passed_checks": len(validation_results["passed_checks"]),
            "failed_checks": len(validation_results["failed_checks"]),
        }

        return result

    def process_prediction_file(self, file_path: str, save_fixed: bool = True) -> Dict:
        """
        Process a prediction JSON file.

        Args:
            file_path: Path to prediction JSON file
            save_fixed: Whether to save fixed predictions to a new file

        Returns:
            Processing results dictionary
        """
        try:
            logger.info(f"📄 Processing file: {Path(file_path).name}")

            # Load predictions from file
            with open(file_path, "r") as f:
                data = json.load(f)

            predictions = data.get("predictions", [])
            if not predictions:
                logger.error(f"No predictions found in {file_path}")
                return {"error": "No predictions found"}

            # Process predictions
            result = self.process_predictions(predictions)

            # Save fixed predictions if requested and fixes were applied
            if save_fixed and result["fixes_applied"]:
                fixed_file_path = str(Path(file_path).with_suffix(".fixed.json"))

                # Create new data structure with fixed predictions
                fixed_data = copy.deepcopy(data)
                fixed_data["predictions"] = result["fixed_predictions"]
                fixed_data["sanity_check_applied"] = True
                fixed_data["original_issues"] = result["validation_results"]["flags"]
                fixed_data["fixes_summary"] = result["processing_summary"]

                with open(fixed_file_path, "w") as f:
                    json.dump(fixed_data, f, indent=2)

                logger.info(
                    f"💾 Fixed predictions saved to: {Path(fixed_file_path).name}"
                )
                result["fixed_file_path"] = fixed_file_path

            return result

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found"}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            return {"error": "Invalid JSON"}
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {"error": str(e)}

    def batch_process_directory(
        self, directory_path: str, max_files: int = 10
    ) -> List[Dict]:
        """
        Process multiple prediction files in a directory.

        Args:
            directory_path: Path to directory containing prediction files
            max_files: Maximum number of files to process

        Returns:
            List of processing results for each file
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []

        json_files = list(directory.glob("*.json"))[:max_files]
        logger.info(
            f"🗂️  Found {len(json_files)} JSON files, processing {len(json_files)}..."
        )

        results = []
        for file_path in json_files:
            result = self.process_prediction_file(str(file_path))
            results.append(
                {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "result": result,
                }
            )

        return results

    def generate_report(self, results: List[Dict]) -> str:
        """
        Generate a summary report of batch processing results.

        Args:
            results: List of processing results

        Returns:
            Formatted report string
        """
        total_files = len(results)
        files_with_issues = 0
        files_fixed = 0
        total_issues = 0

        for result_item in results:
            result = result_item["result"]
            if "processing_summary" in result:
                summary = result["processing_summary"]
                if summary["issues_found"] > 0:
                    files_with_issues += 1
                    total_issues += summary["issues_found"]
                if summary["fixes_applied"]:
                    files_fixed += 1

        report = f"""
📊 BATCH PROCESSING REPORT
========================
Total Files Processed: {total_files}
Files with Issues: {files_with_issues}
Files Fixed: {files_fixed}
Total Issues Found: {total_issues}
Success Rate: {((total_files - files_with_issues) / total_files * 100):.1f}%

Processed by: {self.__class__.__name__}
Total Batches: {self.processed_count}
Total Fixes Applied: {self.fixed_count}
"""
        return report


def demonstrate_all_scenarios():
    """
    Demonstrate all possible scenarios with detailed examples.
    """
    processor = PredictionProcessor()

    print("🚀 COMPREHENSIVE SANITY CHECKS DEMONSTRATION")
    print("=" * 60)

    # Scenario 1: Perfect predictions (should pass all checks)
    print("\n📋 SCENARIO 1: Perfect Predictions")
    perfect_predictions = [
        {
            "dog_name": "Champion",
            "win_probability": 0.4,
            "place_probability": 0.7,
            "predicted_rank": 1,
        },
        {
            "dog_name": "Runner-up",
            "win_probability": 0.3,
            "place_probability": 0.6,
            "predicted_rank": 2,
        },
        {
            "dog_name": "Third Place",
            "win_probability": 0.2,
            "place_probability": 0.5,
            "predicted_rank": 3,
        },
        {
            "dog_name": "Fourth",
            "win_probability": 0.1,
            "place_probability": 0.4,
            "predicted_rank": 4,
        },
    ]

    result = processor.process_predictions(perfect_predictions)
    print(f"   Result: {result['processing_summary']}")

    # Scenario 2: Multiple issues (should trigger all fixes)
    print("\n📋 SCENARIO 2: Multiple Issues")
    problematic_predictions = [
        {
            "dog_name": "Bad Prob",
            "win_probability": 1.5,
            "place_probability": -0.1,
            "predicted_rank": 2,
        },
        {
            "dog_name": "Duplicate Rank",
            "win_probability": 0.8,
            "place_probability": 0.9,
            "predicted_rank": 1,
        },
        {
            "dog_name": "Wrong Rank",
            "win_probability": 0.9,
            "place_probability": 0.8,
            "predicted_rank": 1,
        },
        {
            "dog_name": "NaN Issue",
            "win_probability": float("nan"),
            "place_probability": 0.5,
            "predicted_rank": 3,
        },
    ]

    result = processor.process_predictions(problematic_predictions)
    print(f"   Issues Found: {result['processing_summary']['issues_found']}")
    print(f"   Fixes Applied: {result['processing_summary']['fixes_applied']}")
    print(f"   Issues Remaining: {result['processing_summary']['issues_remaining']}")

    # Scenario 3: Process real prediction files if available
    print("\n📋 SCENARIO 3: Real Prediction Files")
    predictions_dir = Path("predictions")
    if predictions_dir.exists():
        results = processor.batch_process_directory("predictions", max_files=3)
        print(processor.generate_report(results))

        # Show detailed results for first file
        if results:
            first_result = results[0]
            print(f"\n   Detailed example from {first_result['file_name']}:")
            if "processing_summary" in first_result["result"]:
                summary = first_result["result"]["processing_summary"]
                for key, value in summary.items():
                    print(f"      {key}: {value}")
    else:
        print("   No predictions directory found")

    print(f"\n🎯 DEMONSTRATION COMPLETE!")
    print(f"   Files Processed: {processor.processed_count}")
    print(f"   Fixes Applied: {processor.fixed_count}")


if __name__ == "__main__":
    demonstrate_all_scenarios()
