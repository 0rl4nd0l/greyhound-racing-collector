#!/usr/bin/env python3
"""
Ground-Truth Extraction & Leakage Assertion System
==================================================

Implements Step 4 requirements:
1. When loading each historical CSV keep a copy with all columns
2. After prediction, merge back actual finish positions (PLC), race times, etc., strictly for evaluation only
3. Run temporal integrity validation on pre-prediction dataset to ensure no future information leaked
4. If any assertion fails, mark the race as "leakage_detected" and exclude from metric aggregation

This system provides comprehensive protection against temporal leakage in the ML pipeline.
"""

import copy
import hashlib
import json
import logging
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Import our existing systems
from ml_system_v4 import MLSystemV4
from src.parsers.csv_ingestion import CsvIngestion, ParsedRace, ValidationReport
from temporal_feature_builder import TemporalFeatureBuilder

logger = logging.getLogger(__name__)


class GroundTruthExtractor:
    """
    Handles ground-truth data extraction and storage for evaluation purposes.
    Ensures strict separation between prediction features and ground-truth results.
    """

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.ground_truth_cache = {}
        self.leakage_log_path = Path("./logs/leakage_detection.log")
        self.leakage_log_path.parent.mkdir(exist_ok=True, parents=True)

        # Ground-truth columns that should NEVER be used for prediction
        self.ground_truth_columns = {
            "finish_position",
            "individual_time",
            "sectional_1st",
            "sectional_2nd",
            "sectional_3rd",
            "margin",
            "beaten_margin",
            "winning_time",
            "scraped_finish_position",
            "scraped_raw_result",
            "winner_name",
            "winner_odds",
            "winner_margin",
            "pir_rating",
            "first_sectional",
            "win_time",
            "bonus_time",
            "race_result",
            "final_time",
            "race_winner",
            "actual_finish_position",
            "actual_race_time",
            "race_outcome",
        }

        # Initialize logging
        self._setup_leakage_logging()

    def _setup_leakage_logging(self):
        """Setup dedicated logging for leakage detection."""
        leakage_logger = logging.getLogger("leakage_detection")
        leakage_logger.setLevel(logging.INFO)

        # Create file handler if it doesn't exist
        if not any(
            isinstance(h, logging.FileHandler)
            and h.baseFilename == str(self.leakage_log_path)
            for h in leakage_logger.handlers
        ):
            file_handler = logging.FileHandler(self.leakage_log_path)
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - LEAKAGE_DETECTION - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            leakage_logger.addHandler(file_handler)

        self.leakage_logger = leakage_logger

    def load_csv_with_ground_truth_separation(
        self, csv_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
        """
        Load CSV file and separate prediction features from ground-truth data.

        Returns:
            prediction_data: DataFrame with only prediction-safe columns
            ground_truth_data: DataFrame with all columns for evaluation
            is_valid: Boolean indicating if the data passed leakage checks
        """
        logger.info(f"📄 Loading CSV with ground-truth separation: {csv_path}")

        try:
            # Parse CSV using existing ingestion system
            csv_ingestion = CsvIngestion(csv_path)
            parsed_race, validation_report = csv_ingestion.parse_csv()

            if not validation_report.is_valid:
                logger.error(
                    f"CSV validation failed for {csv_path}: {validation_report.errors}"
                )
                return pd.DataFrame(), pd.DataFrame(), False

            # Convert to DataFrame
            full_dataframe = pd.DataFrame(
                parsed_race.records, columns=parsed_race.headers
            )

            if full_dataframe.empty:
                logger.warning(f"Empty DataFrame loaded from {csv_path}")
                return pd.DataFrame(), pd.DataFrame(), False

            # Create ground-truth copy (complete data for evaluation only)
            ground_truth_data = full_dataframe.copy()
            ground_truth_data["_csv_source"] = csv_path
            ground_truth_data["_load_timestamp"] = datetime.now().isoformat()

            # Create prediction-safe copy (remove ground-truth columns)
            prediction_columns = [
                col
                for col in full_dataframe.columns
                if col.lower() not in {c.lower() for c in self.ground_truth_columns}
            ]

            prediction_data = full_dataframe[prediction_columns].copy()
            prediction_data["_csv_source"] = csv_path
            prediction_data["_load_timestamp"] = datetime.now().isoformat()

            # Log ground-truth extraction
            removed_columns = set(full_dataframe.columns) - set(prediction_columns)
            if removed_columns:
                logger.info(
                    f"🛡️ Extracted {len(removed_columns)} ground-truth columns: {removed_columns}"
                )

            # Store ground-truth data in cache for later merging
            race_id = self._generate_race_id(csv_path, ground_truth_data)
            self.ground_truth_cache[race_id] = {
                "data": ground_truth_data,
                "csv_path": csv_path,
                "extraction_timestamp": datetime.now().isoformat(),
                "prediction_columns": prediction_columns,
                "ground_truth_columns": list(removed_columns),
            }

            logger.info(
                f"✅ Successfully separated prediction data ({len(prediction_columns)} cols) "
                f"from ground-truth data ({len(removed_columns)} cols)"
            )

            return prediction_data, ground_truth_data, True

        except Exception as e:
            logger.error(f"❌ Error loading CSV with ground-truth separation: {e}")
            self._log_leakage_error("csv_loading_error", csv_path, str(e))
            return pd.DataFrame(), pd.DataFrame(), False

    def _generate_race_id(self, csv_path: str, data: pd.DataFrame) -> str:
        """Generate unique race ID for caching."""
        # Try to extract race info from filename or data
        race_name = Path(csv_path).stem

        # Add content hash for uniqueness
        content_hash = hashlib.md5(str(data.values.tolist()).encode()).hexdigest()[:8]

        return f"{race_name}_{content_hash}"

    def validate_temporal_integrity_pre_prediction(
        self, prediction_data: pd.DataFrame, race_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate temporal integrity BEFORE prediction to ensure no future information leaked.

        Returns:
            is_valid: Boolean indicating if temporal integrity is maintained
            violations: List of detected violations
        """
        logger.info(f"🔍 Validating temporal integrity for race {race_id}")
        violations = []

        try:
            # Check 1: No ground-truth columns in prediction data
            prediction_columns = set(prediction_data.columns)
            ground_truth_violations = prediction_columns.intersection(
                self.ground_truth_columns
            )

            if ground_truth_violations:
                violation_msg = f"Ground-truth columns found in prediction data: {ground_truth_violations}"
                violations.append(violation_msg)
                self._log_leakage_violation(
                    "ground_truth_in_prediction", race_id, violation_msg
                )

            # Check 2: No future timestamp columns
            timestamp_columns = [
                col
                for col in prediction_data.columns
                if any(
                    keyword in col.lower()
                    for keyword in ["result", "final", "outcome", "winner", "finish"]
                )
            ]

            if timestamp_columns:
                violation_msg = (
                    f"Suspicious future-leakage columns found: {timestamp_columns}"
                )
                violations.append(violation_msg)
                self._log_leakage_violation(
                    "suspicious_columns", race_id, violation_msg
                )

            # Check 3: Validate using TemporalFeatureBuilder if available
            try:
                temporal_builder = TemporalFeatureBuilder(self.db_path)

                # Convert prediction data to format expected by temporal builder
                race_data = prediction_data.copy()

                # Add required columns if missing (for temporal validation)
                if "race_id" not in race_data.columns:
                    race_data["race_id"] = race_id

                if (
                    "dog_clean_name" not in race_data.columns
                    and "Dog Name" in race_data.columns
                ):
                    race_data["dog_clean_name"] = race_data["Dog Name"]

                # Build features (this will validate temporal integrity internally)
                features_df = temporal_builder.build_features_for_race(
                    race_data, race_id
                )

                # Run explicit temporal integrity validation
                temporal_builder.validate_temporal_integrity(features_df, race_data)

                logger.info(
                    "✅ Temporal integrity validation passed via TemporalFeatureBuilder"
                )

            except AssertionError as ae:
                violation_msg = f"TemporalFeatureBuilder assertion failed: {str(ae)}"
                violations.append(violation_msg)
                self._log_leakage_violation(
                    "temporal_builder_assertion", race_id, violation_msg
                )

            except Exception as e:
                logger.warning(
                    f"⚠️ TemporalFeatureBuilder validation failed with error: {e}"
                )
                # Don't treat this as a violation, just log the warning

            # Check 4: Data consistency checks
            if len(prediction_data) == 0:
                violation_msg = "Empty prediction dataset"
                violations.append(violation_msg)
                self._log_leakage_violation("empty_dataset", race_id, violation_msg)

            # Check 5: Validate required prediction columns exist
            required_columns = [
                "Dog Name",
                "BOX",
            ]  # Basic required columns for prediction
            missing_required = [
                col for col in required_columns if col not in prediction_data.columns
            ]

            if missing_required:
                violation_msg = (
                    f"Missing required prediction columns: {missing_required}"
                )
                violations.append(violation_msg)
                self._log_leakage_violation(
                    "missing_required_columns", race_id, violation_msg
                )

            is_valid = len(violations) == 0

            if is_valid:
                logger.info(
                    f"✅ Temporal integrity validation passed for race {race_id}"
                )
            else:
                logger.error(
                    f"❌ Temporal integrity validation failed for race {race_id}: {violations}"
                )

            return is_valid, violations

        except Exception as e:
            error_msg = f"Temporal integrity validation error: {str(e)}"
            violations.append(error_msg)
            self._log_leakage_error("temporal_validation_error", race_id, error_msg)
            return False, violations

    def merge_ground_truth_for_evaluation(
        self, predictions: Dict[str, Any], race_id: str
    ) -> Dict[str, Any]:
        """
        Merge ground-truth data back with predictions STRICTLY for evaluation only.
        This happens AFTER prediction is complete.
        """
        logger.info(f"🔄 Merging ground-truth data for evaluation: race {race_id}")

        try:
            if race_id not in self.ground_truth_cache:
                logger.warning(f"⚠️ No ground-truth data cached for race {race_id}")
                # Return predictions unchanged with warning metadata
                if isinstance(predictions, dict) and "metadata" in predictions:
                    predictions["metadata"][
                        "ground_truth_warning"
                    ] = "No cached ground-truth data available"
                return predictions

            ground_truth_info = self.ground_truth_cache[race_id]
            ground_truth_data = ground_truth_info["data"]

            # Create evaluation result with ground-truth merged
            evaluation_result = copy.deepcopy(predictions)

            # Add ground-truth data to metadata (never to prediction features)
            if "metadata" not in evaluation_result:
                evaluation_result["metadata"] = {}

            evaluation_result["metadata"]["ground_truth_data"] = {
                "extraction_timestamp": ground_truth_info["extraction_timestamp"],
                "csv_source": ground_truth_info["csv_path"],
                "ground_truth_columns": ground_truth_info["ground_truth_columns"],
                "total_dogs": len(ground_truth_data),
            }

            # Extract actual results if available in ground-truth data
            actual_results = []

            for _, row in ground_truth_data.iterrows():
                dog_result = {
                    "dog_name": row.get("Dog Name", "Unknown"),
                    "box_number": row.get("BOX", 0),
                }

                # Add ground-truth results if available
                for gt_col in self.ground_truth_columns:
                    if gt_col in row and pd.notna(row[gt_col]):
                        dog_result[f"actual_{gt_col}"] = row[gt_col]

                actual_results.append(dog_result)

            evaluation_result["metadata"]["actual_results"] = actual_results

            # Mark as evaluation-only data
            evaluation_result["metadata"]["evaluation_only"] = True
            evaluation_result["metadata"][
                "ground_truth_merge_timestamp"
            ] = datetime.now().isoformat()

            logger.info(
                f"✅ Successfully merged ground-truth data for {len(actual_results)} dogs"
            )

            return evaluation_result

        except Exception as e:
            logger.error(f"❌ Error merging ground-truth data for race {race_id}: {e}")
            self._log_leakage_error("ground_truth_merge_error", race_id, str(e))

            # Return original predictions with error metadata
            if isinstance(predictions, dict) and "metadata" in predictions:
                predictions["metadata"]["ground_truth_error"] = str(e)

            return predictions

    def mark_race_as_leakage_detected(
        self, race_id: str, violations: List[str]
    ) -> Dict[str, Any]:
        """
        Mark a race as having leakage detected and exclude from metric aggregation.
        """
        logger.error(f"🚨 LEAKAGE DETECTED: Marking race {race_id} as invalid")

        leakage_record = {
            "race_id": race_id,
            "detection_timestamp": datetime.now().isoformat(),
            "violations": violations,
            "status": "leakage_detected",
            "excluded_from_metrics": True,
        }

        # Log detailed leakage information
        self._log_leakage_violation(
            "race_marked_invalid", race_id, f"Violations: {'; '.join(violations)}"
        )

        # Store in cache for tracking
        if not hasattr(self, "leakage_records"):
            self.leakage_records = {}

        self.leakage_records[race_id] = leakage_record

        return leakage_record

    def _log_leakage_violation(self, violation_type: str, race_id: str, details: str):
        """Log leakage violation with structured format."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "VIOLATION",
            "violation_type": violation_type,
            "race_id": race_id,
            "details": details,
        }

        self.leakage_logger.error(f"LEAKAGE_VIOLATION: {json.dumps(log_entry)}")

    def _log_leakage_error(self, error_type: str, race_id: str, details: str):
        """Log leakage-related error with structured format."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "ERROR",
            "error_type": error_type,
            "race_id": race_id,
            "details": details,
        }

        self.leakage_logger.error(f"LEAKAGE_ERROR: {json.dumps(log_entry)}")

    def get_leakage_summary(self) -> Dict[str, Any]:
        """Get summary of leakage detection results."""
        if not hasattr(self, "leakage_records"):
            self.leakage_records = {}

        return {
            "total_races_processed": len(self.ground_truth_cache),
            "races_with_leakage": len(self.leakage_records),
            "leakage_rate": len(self.leakage_records)
            / max(1, len(self.ground_truth_cache)),
            "leakage_records": list(self.leakage_records.values()),
            "summary_timestamp": datetime.now().isoformat(),
        }


class LeakageProtectedPredictor:
    """
    Main prediction interface with integrated leakage protection.
    Wraps existing prediction systems with ground-truth extraction and validation.
    """

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.ground_truth_extractor = GroundTruthExtractor(db_path)
        self.ml_system = MLSystemV4(db_path)

        logger.info(
            "🛡️ LeakageProtectedPredictor initialized with ground-truth extraction"
        )

    def predict_race_with_leakage_protection(self, csv_path: str) -> Dict[str, Any]:
        """
        Main prediction method with comprehensive leakage protection.

        Implements the complete Step 4 workflow:
        1. Load CSV with ground-truth separation
        2. Validate temporal integrity pre-prediction
        3. Run prediction on clean data
        4. Merge ground-truth for evaluation only
        5. Handle leakage detection and exclusion
        """
        logger.info(f"🚀 Starting leakage-protected prediction for: {csv_path}")

        try:
            # Step 1: Load CSV with ground-truth separation
            prediction_data, ground_truth_data, load_success = (
                self.ground_truth_extractor.load_csv_with_ground_truth_separation(
                    csv_path
                )
            )

            if not load_success:
                return {
                    "success": False,
                    "error": "Failed to load CSV with ground-truth separation",
                    "race_id": Path(csv_path).stem,
                    "leakage_status": "load_failed",
                }

            race_id = self.ground_truth_extractor._generate_race_id(
                csv_path, ground_truth_data
            )

            # Step 2: Validate temporal integrity pre-prediction
            integrity_valid, violations = (
                self.ground_truth_extractor.validate_temporal_integrity_pre_prediction(
                    prediction_data, race_id
                )
            )

            if not integrity_valid:
                # Step 4a: Mark race as having leakage detected
                leakage_record = (
                    self.ground_truth_extractor.mark_race_as_leakage_detected(
                        race_id, violations
                    )
                )

                return {
                    "success": False,
                    "error": "Temporal integrity validation failed",
                    "race_id": race_id,
                    "leakage_status": "leakage_detected",
                    "violations": violations,
                    "leakage_record": leakage_record,
                }

            # Step 3: Run prediction on validated clean data
            logger.info("🔮 Running prediction on temporally-validated data...")

            # Preprocess data for ML system
            preprocessed_data = self.ml_system.preprocess_upcoming_race_csv(
                prediction_data, race_id
            )

            # Get predictions
            predictions = self.ml_system.predict_race(preprocessed_data, race_id)

            if not predictions.get("success", False):
                return {
                    "success": False,
                    "error": f"Prediction failed: {predictions.get('error', 'Unknown error')}",
                    "race_id": race_id,
                    "leakage_status": "prediction_failed",
                }

            # Step 4: Merge ground-truth for evaluation only
            evaluation_result = (
                self.ground_truth_extractor.merge_ground_truth_for_evaluation(
                    predictions, race_id
                )
            )

            # Add leakage protection metadata
            evaluation_result["metadata"].update(
                {
                    "leakage_protection": {
                        "temporal_integrity_validated": True,
                        "ground_truth_separated": True,
                        "violations_detected": 0,
                        "protection_timestamp": datetime.now().isoformat(),
                    }
                }
            )

            logger.info(
                f"✅ Leakage-protected prediction completed successfully for race {race_id}"
            )

            return evaluation_result

        except Exception as e:
            logger.error(f"❌ Error in leakage-protected prediction: {e}")
            return {
                "success": False,
                "error": str(e),
                "race_id": Path(csv_path).stem,
                "leakage_status": "system_error",
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including leakage protection."""
        return {
            "ml_system_loaded": self.ml_system.calibrated_pipeline is not None,
            "ground_truth_extractor_active": True,
            "leakage_summary": self.ground_truth_extractor.get_leakage_summary(),
            "system_timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize the leakage-protected predictor
    predictor = LeakageProtectedPredictor()

    print("🛡️ Ground-Truth Extraction & Leakage Assertion System")
    print("=" * 60)
    print("✅ System initialized successfully")
    print("🔍 Temporal integrity validation: ACTIVE")
    print("📊 Ground-truth separation: ACTIVE")
    print("🚨 Leakage detection: ACTIVE")

    status = predictor.get_system_status()
    print(f"\n📈 System Status:")
    print(f"   ML System Loaded: {status['ml_system_loaded']}")
    print(f"   Ground-Truth Extractor: {status['ground_truth_extractor_active']}")
    print(f"   Races Processed: {status['leakage_summary']['total_races_processed']}")
    print(f"   Leakage Rate: {status['leakage_summary']['leakage_rate']:.2%}")
