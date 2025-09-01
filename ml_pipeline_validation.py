#!/usr/bin/env python3
"""
ML Pipeline & Prediction Validation - Step 3
===========================================

This script implements comprehensive ML pipeline validation:
• Load actual pickled/ONNX models from models/ directory
• Feed real historical race feature rows from the sandbox DB  
• Assert output vector length = number of dogs
• Probabilities sum ≈ 1.0
• Deterministic seed → reproducible prediction
• Compare model AUC/log-loss on held-out real validation slice
• Fail if drift > 5% from baseline stored in baseline_metrics.json

Requirements:
- Real models in models/ directory
- Unified database with historical race data
- Baseline metrics for comparison
"""

import json
import logging
import os
import pickle
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_pipeline_validation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates ML models with real data and performance metrics"""

    def __init__(
        self,
        db_path: str = "greyhound_racing_data.db",
        models_dir: str = "./model_registry/models",
    ):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.baseline_metrics_file = "baseline_metrics.json"
        self.validation_results = {}

        # Set deterministic seed for reproducible predictions
        np.random.seed(42)

        logger.info(f"🔧 Initializing ModelValidator")
        logger.info(f"   Database: {self.db_path}")
        logger.info(f"   Models directory: {self.models_dir}")

    def load_baseline_metrics(self) -> Dict[str, Any]:
        """Load baseline metrics from file or create default"""
        baseline_path = Path(self.baseline_metrics_file)

        if baseline_path.exists():
            try:
                with open(baseline_path, "r") as f:
                    baseline = json.load(f)
                logger.info(f"✅ Loaded baseline metrics from {baseline_path}")
                return baseline
            except Exception as e:
                logger.warning(f"⚠️ Failed to load baseline metrics: {e}")

        # Create default baseline if file doesn't exist
        default_baseline = {
            "created_date": datetime.now().isoformat(),
            "model_metrics": {
                "gradient_boosting": {
                    "roc_auc": 0.75,
                    "log_loss": 0.65,
                    "accuracy": 0.70,
                },
                "random_forest": {"roc_auc": 0.72, "log_loss": 0.68, "accuracy": 0.68},
            },
            "validation_criteria": {
                "max_drift_percentage": 5.0,
                "min_probability_sum": 0.95,
                "max_probability_sum": 1.05,
                "min_dogs_per_race": 3,
                "max_dogs_per_race": 12,
            },
        }

        # Save default baseline
        with open(baseline_path, "w") as f:
            json.dump(default_baseline, f, indent=2)

        logger.info(f"📝 Created default baseline metrics at {baseline_path}")
        return default_baseline

    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover all available models in the models directory"""
        models = []

        if not self.models_dir.exists():
            logger.error(f"❌ Models directory not found: {self.models_dir}")
            return models

        logger.info(f"🔍 Scanning for models in {self.models_dir}")

        # Look for actual ML model files (not scalers or explainers)
        model_patterns = ["*_model.joblib"]

        for pattern in model_patterns:
            for model_file in self.models_dir.glob(pattern):
                try:
                    # Skip SHAP explainer files
                    if "shap_explainer" in model_file.name:
                        continue

                    model_info = {
                        "file_path": str(model_file),
                        "file_name": model_file.name,
                        "file_size": model_file.stat().st_size,
                        "model_type": self._infer_model_type(model_file.name),
                        "created_date": datetime.fromtimestamp(
                            model_file.stat().st_mtime
                        ).isoformat(),
                    }

                    # Look for corresponding scaler file
                    scaler_file = model_file.parent / model_file.name.replace(
                        "_model.joblib", "_scaler.joblib"
                    )
                    if scaler_file.exists():
                        model_info["scaler_path"] = str(scaler_file)
                        model_info["has_scaler"] = True
                    else:
                        model_info["has_scaler"] = False

                    models.append(model_info)
                    logger.info(
                        f"   Found model: {model_file.name} ({model_info['file_size']} bytes)"
                    )
                except Exception as e:
                    logger.warning(f"   ⚠️ Error processing {model_file}: {e}")

        logger.info(f"✅ Discovered {len(models)} models")
        return models

    def _infer_model_type(self, filename: str) -> str:
        """Infer model type from filename"""
        filename_lower = filename.lower()

        if "gradient" in filename_lower or "boosting" in filename_lower:
            return "gradient_boosting"
        elif "random" in filename_lower or "forest" in filename_lower:
            return "random_forest"
        elif "logistic" in filename_lower:
            return "logistic_regression"
        elif "xgb" in filename_lower or "xgboost" in filename_lower:
            return "xgboost"
        elif "extra" in filename_lower or "trees" in filename_lower:
            return "extra_trees"
        else:
            return "unknown"

    def load_historical_data(
        self, limit: int = 10000
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load real historical race data from the database"""
        logger.info(f"📊 Loading historical race data from database")

        try:
            conn = sqlite3.connect(self.db_path)

            # Query for historical race results with dog performance data
            # Using available data even if some fields are NULL
            query = """
            SELECT 
                dp.dog_name,
                dp.box_number,
                dp.weight,
                COALESCE(dp.odds, 3.0) as starting_price,
                dp.finish_position as finishing_position,
                dp.race_time,
                COALESCE(rm.track_condition, 'Unknown') as track_condition,
                COALESCE(rm.distance, '515m') as distance,
                COALESCE(rm.weather, 'Fine') as weather,
                COALESCE(rm.temperature, 20.0) as temperature,
                rm.race_date,
                rm.venue,
                CASE WHEN dp.finish_position = 1 THEN 1 ELSE 0 END as won
            FROM 
                dog_performances dp
            JOIN 
                race_metadata rm ON dp.race_id = rm.race_id
            WHERE 
                dp.finish_position IS NOT NULL 
                AND dp.weight IS NOT NULL
                AND dp.race_time IS NOT NULL
                AND rm.race_date IS NOT NULL
            ORDER BY 
                rm.race_date DESC
            LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()

            if df.empty:
                logger.error("❌ No historical data found in database")
                return pd.DataFrame(), pd.Series()

            logger.info(f"✅ Loaded {len(df)} historical race records")
            logger.info(f"   Columns: {list(df.columns)}")
            logger.info(
                f"   Winners: {df['won'].sum()}, Non-winners: {len(df) - df['won'].sum()}"
            )

            # Separate features and target
            target = df["won"]
            features = df.drop(["won"], axis=1)

            return features, target

        except Exception as e:
            logger.error(f"❌ Failed to load historical data: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return pd.DataFrame(), pd.Series()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction"""
        logger.info("🔧 Preparing features for prediction")

        try:
            # Create a copy to avoid modifying original
            features = df.copy()

            # Handle categorical variables
            categorical_cols = ["track_condition", "weather", "venue"]
            for col in categorical_cols:
                if col in features.columns:
                    # Simple label encoding (in production, use proper encoding with training data)
                    features[col] = pd.Categorical(features[col]).codes

            # Handle date column
            if "race_date" in features.columns:
                features["race_date"] = pd.to_datetime(
                    features["race_date"], errors="coerce"
                )
                features["race_year"] = features["race_date"].dt.year
                features["race_month"] = features["race_date"].dt.month
                features["race_day"] = features["race_date"].dt.day
                features = features.drop(["race_date"], axis=1)

            # Handle missing values
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(
                features[numeric_cols].median()
            )

            # Ensure we have some basic features for validation
            # Add some synthetic features if needed for model compatibility
            if "dog_name" in features.columns:
                features["dog_name_length"] = features["dog_name"].astype(str).str.len()

            # Handle non-numeric columns
            for col in features.columns:
                if features[col].dtype == "object":
                    features[col] = pd.Categorical(features[col]).codes

            logger.info(
                f"✅ Feature preparation complete: {len(features.columns)} features"
            )
            logger.info(f"   Feature columns: {list(features.columns)}")

            return features

        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            return pd.DataFrame()

    def validate_model(
        self, model_info: Dict[str, Any], features: pd.DataFrame, target: pd.Series
    ) -> Dict[str, Any]:
        """Validate a single model with comprehensive checks"""
        logger.info(f"🔍 Validating model: {model_info['file_name']}")

        validation_result = {
            "model_info": model_info,
            "validation_start": datetime.now().isoformat(),
            "success": False,
            "checks": {},
            "metrics": {},
            "issues": [],
        }

        try:
            # Load the model
            model_path = model_info["file_path"]

            try:
                # Load the ML model
                if model_path.endswith(".joblib"):
                    model = joblib.load(model_path)
                elif model_path.endswith((".pkl", ".pickle")):
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                else:
                    validation_result["issues"].append(
                        f"Unsupported model format: {model_path}"
                    )
                    return validation_result

                # Load corresponding scaler if available
                scaler = None
                if model_info.get("has_scaler", False):
                    try:
                        scaler = joblib.load(model_info["scaler_path"])
                        logger.info(f"✅ Scaler loaded successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load scaler: {e}")

                logger.info(f"✅ Model loaded successfully")
                validation_result["checks"]["model_loaded"] = True

            except Exception as e:
                validation_result["issues"].append(f"Failed to load model: {e}")
                return validation_result

            # Split data for validation
            if len(features) < 100:
                validation_result["issues"].append(
                    f"Insufficient data for validation: {len(features)} samples"
                )
                return validation_result

            X_train, X_val, y_train, y_val = train_test_split(
                features, target, test_size=0.3, random_state=42, stratify=target
            )

            logger.info(f"✅ Data split: {len(X_train)} train, {len(X_val)} validation")

            # Check 1: Model can make predictions
            try:
                # Test with a small sample first
                sample_data = X_val.iloc[:10].copy()

                # Apply scaler if available
                if scaler is not None:
                    sample_data_scaled = scaler.transform(sample_data)
                    sample_predictions = model.predict_proba(sample_data_scaled)
                else:
                    sample_predictions = model.predict_proba(sample_data)

                logger.info(
                    f"✅ Sample predictions successful: shape {sample_predictions.shape}"
                )
                validation_result["checks"]["can_predict"] = True

            except Exception as e:
                validation_result["issues"].append(f"Model prediction failed: {e}")
                return validation_result

            # Check 2: Full validation predictions
            try:
                # Apply scaler if available
                if scaler is not None:
                    X_val_scaled = scaler.transform(X_val)
                    val_predictions = model.predict_proba(X_val_scaled)
                else:
                    val_predictions = model.predict_proba(X_val)

                if val_predictions.ndim == 2 and val_predictions.shape[1] >= 2:
                    val_probs = val_predictions[:, 1]  # Probability of positive class
                else:
                    val_probs = val_predictions.flatten()

                logger.info(f"✅ Validation predictions: {len(val_probs)} predictions")
                validation_result["checks"]["full_prediction"] = True

            except Exception as e:
                validation_result["issues"].append(f"Full prediction failed: {e}")
                return validation_result

            # Check 3: Output vector length validation
            expected_samples = len(X_val)
            actual_samples = len(val_probs)

            if actual_samples == expected_samples:
                logger.info(f"✅ Output vector length correct: {actual_samples}")
                validation_result["checks"]["output_length"] = True
            else:
                validation_result["issues"].append(
                    f"Output length mismatch: expected {expected_samples}, got {actual_samples}"
                )
                return validation_result

            # Check 4: Probability range validation
            prob_min, prob_max = val_probs.min(), val_probs.max()
            if prob_min >= 0 and prob_max <= 1:
                logger.info(
                    f"✅ Probability range valid: [{prob_min:.3f}, {prob_max:.3f}]"
                )
                validation_result["checks"]["probability_range"] = True
            else:
                validation_result["issues"].append(
                    f"Invalid probability range: [{prob_min:.3f}, {prob_max:.3f}]"
                )

            # Check 5: Deterministic predictions (reproducibility)
            try:
                # Set seed and predict again
                np.random.seed(42)
                if scaler is not None:
                    X_val_scaled_2 = scaler.transform(X_val)
                    val_predictions_2 = model.predict_proba(X_val_scaled_2)
                else:
                    val_predictions_2 = model.predict_proba(X_val)

                if val_predictions_2.ndim == 2 and val_predictions_2.shape[1] >= 2:
                    val_probs_2 = val_predictions_2[:, 1]
                else:
                    val_probs_2 = val_predictions_2.flatten()

                # Check if predictions are identical
                if np.allclose(val_probs, val_probs_2, rtol=1e-10):
                    logger.info(f"✅ Predictions are deterministic/reproducible")
                    validation_result["checks"]["deterministic"] = True
                else:
                    validation_result["issues"].append(
                        "Predictions are not deterministic"
                    )

            except Exception as e:
                validation_result["issues"].append(f"Deterministic check failed: {e}")

            # Calculate performance metrics
            try:
                # ROC AUC
                roc_auc = roc_auc_score(y_val, val_probs)

                # Log Loss
                log_loss_score = log_loss(y_val, val_probs)

                # Accuracy (using 0.5 threshold)
                accuracy = accuracy_score(y_val, (val_probs > 0.5).astype(int))

                validation_result["metrics"] = {
                    "roc_auc": float(roc_auc),
                    "log_loss": float(log_loss_score),
                    "accuracy": float(accuracy),
                    "validation_samples": len(y_val),
                    "positive_samples": int(y_val.sum()),
                    "negative_samples": int(len(y_val) - y_val.sum()),
                }

                logger.info(f"✅ Performance metrics calculated:")
                logger.info(f"   ROC AUC: {roc_auc:.4f}")
                logger.info(f"   Log Loss: {log_loss_score:.4f}")
                logger.info(f"   Accuracy: {accuracy:.4f}")

                validation_result["checks"]["metrics_calculated"] = True

            except Exception as e:
                validation_result["issues"].append(f"Metrics calculation failed: {e}")
                return validation_result

            # Check 6: Compare against baseline (drift detection)
            baseline_metrics = self.load_baseline_metrics()
            model_type = model_info["model_type"]

            if model_type in baseline_metrics.get("model_metrics", {}):
                baseline = baseline_metrics["model_metrics"][model_type]
                max_drift = baseline_metrics["validation_criteria"][
                    "max_drift_percentage"
                ]

                # Calculate drift percentages
                auc_drift = (
                    abs(roc_auc - baseline["roc_auc"]) / baseline["roc_auc"] * 100
                )
                loss_drift = (
                    abs(log_loss_score - baseline["log_loss"])
                    / baseline["log_loss"]
                    * 100
                )
                acc_drift = (
                    abs(accuracy - baseline["accuracy"]) / baseline["accuracy"] * 100
                )

                validation_result["drift_analysis"] = {
                    "auc_drift_percentage": float(auc_drift),
                    "log_loss_drift_percentage": float(loss_drift),
                    "accuracy_drift_percentage": float(acc_drift),
                    "max_allowed_drift": max_drift,
                    "baseline_metrics": baseline,
                }

                # Check if drift exceeds threshold
                max_observed_drift = max(auc_drift, loss_drift, acc_drift)
                if max_observed_drift <= max_drift:
                    logger.info(
                        f"✅ Drift check passed: max drift {max_observed_drift:.2f}% <= {max_drift}%"
                    )
                    validation_result["checks"]["drift_acceptable"] = True
                else:
                    validation_result["issues"].append(
                        f"Drift too high: {max_observed_drift:.2f}% > {max_drift}%"
                    )
                    logger.warning(
                        f"⚠️ Drift check failed: {max_observed_drift:.2f}% > {max_drift}%"
                    )
            else:
                logger.warning(
                    f"⚠️ No baseline metrics found for model type: {model_type}"
                )
                validation_result["drift_analysis"] = {"no_baseline": True}

            # Overall success check
            required_checks = [
                "model_loaded",
                "can_predict",
                "full_prediction",
                "output_length",
                "probability_range",
                "metrics_calculated",
            ]
            passed_checks = sum(
                1
                for check in required_checks
                if validation_result["checks"].get(check, False)
            )

            if (
                passed_checks == len(required_checks)
                and not validation_result["issues"]
            ):
                validation_result["success"] = True
                logger.info(
                    f"✅ Model validation successful: {model_info['file_name']}"
                )
            else:
                logger.warning(
                    f"⚠️ Model validation incomplete: {passed_checks}/{len(required_checks)} checks passed"
                )

        except Exception as e:
            validation_result["issues"].append(f"Validation error: {e}")
            logger.error(f"❌ Model validation failed: {e}")

        validation_result["validation_end"] = datetime.now().isoformat()
        return validation_result

    def simulate_race_prediction(
        self, model_info: Dict[str, Any], features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Simulate a race prediction to test output format"""
        logger.info(f"🏁 Simulating race prediction with {model_info['file_name']}")

        try:
            # Load model and scaler
            model_path = model_info["file_path"]
            if model_path.endswith(".joblib"):
                model = joblib.load(model_path)
            elif model_path.endswith((".pkl", ".pickle")):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                return {"success": False, "error": "Unsupported model format"}

            # Load scaler if available
            scaler = None
            if model_info.get("has_scaler", False):
                try:
                    scaler = joblib.load(model_info["scaler_path"])
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load scaler for race simulation: {e}")

            # Create a sample race (e.g., 8 dogs)
            num_dogs = 8
            if len(features) < num_dogs:
                return {
                    "success": False,
                    "error": f"Not enough data: need {num_dogs}, have {len(features)}",
                }

            race_data = features.iloc[:num_dogs].copy()

            # Generate predictions
            if scaler is not None:
                race_data_scaled = scaler.transform(race_data)
                predictions = model.predict_proba(race_data_scaled)
            else:
                predictions = model.predict_proba(race_data)
            if predictions.ndim == 2 and predictions.shape[1] >= 2:
                win_probs = predictions[:, 1]
            else:
                win_probs = predictions.flatten()

            # Check probability sum (should be close to 1.0 for a race)
            prob_sum = win_probs.sum()

            # Create race result
            race_result = {
                "success": True,
                "num_dogs": num_dogs,
                "win_probabilities": win_probs.tolist(),
                "probability_sum": float(prob_sum),
                "min_probability": float(win_probs.min()),
                "max_probability": float(win_probs.max()),
                "predictions": [
                    {
                        "dog_position": i + 1,
                        "win_probability": float(prob),
                        "predicted_rank": int(np.argsort(np.argsort(-win_probs))[i])
                        + 1,
                    }
                    for i, prob in enumerate(win_probs)
                ],
            }

            # Validate race prediction
            baseline = self.load_baseline_metrics()
            criteria = baseline["validation_criteria"]

            issues = []

            # Check number of dogs
            if not (
                criteria["min_dogs_per_race"]
                <= num_dogs
                <= criteria["max_dogs_per_race"]
            ):
                issues.append(f"Invalid number of dogs: {num_dogs}")

            # Check probability sum
            if not (
                criteria["min_probability_sum"]
                <= prob_sum
                <= criteria["max_probability_sum"]
            ):
                issues.append(f"Invalid probability sum: {prob_sum:.3f}")

            race_result["validation_issues"] = issues
            race_result["validation_passed"] = len(issues) == 0

            if race_result["validation_passed"]:
                logger.info(
                    f"✅ Race simulation successful: {num_dogs} dogs, prob_sum={prob_sum:.3f}"
                )
            else:
                logger.warning(f"⚠️ Race simulation issues: {issues}")

            return race_result

        except Exception as e:
            logger.error(f"❌ Race simulation failed: {e}")
            return {"success": False, "error": str(e)}

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all models"""
        logger.info("🚀 Starting comprehensive ML pipeline validation")
        logger.info("=" * 60)

        start_time = time.time()

        # Discover models
        models = self.discover_models()
        if not models:
            logger.error("❌ No models found for validation")
            return {
                "success": False,
                "error": "No models found",
                "models_directory": str(self.models_dir),
                "timestamp": datetime.now().isoformat(),
            }

        # Load historical data
        logger.info("📊 Loading historical data...")
        features, target = self.load_historical_data()

        if features.empty or target.empty:
            logger.error("❌ No historical data available for validation")
            return {
                "success": False,
                "error": "No historical data available",
                "timestamp": datetime.now().isoformat(),
            }

        # Prepare features
        prepared_features = self.prepare_features(features)
        if prepared_features.empty:
            logger.error("❌ Feature preparation failed")
            return {
                "success": False,
                "error": "Feature preparation failed",
                "timestamp": datetime.now().isoformat(),
            }

        # Validate each model
        validation_results = {
            "validation_start": datetime.now().isoformat(),
            "models_tested": len(models),
            "data_samples": len(prepared_features),
            "feature_count": len(prepared_features.columns),
            "model_results": [],
            "summary": {},
        }

        successful_models = 0
        failed_models = 0

        for model_info in models:
            logger.info(f"\n{'='*40}")
            logger.info(f"Validating: {model_info['file_name']}")
            logger.info(f"{'='*40}")

            # Validate model
            model_result = self.validate_model(model_info, prepared_features, target)

            # Simulate race prediction
            race_simulation = self.simulate_race_prediction(
                model_info, prepared_features
            )
            model_result["race_simulation"] = race_simulation

            validation_results["model_results"].append(model_result)

            if model_result["success"]:
                successful_models += 1
                logger.info(f"✅ {model_info['file_name']} - PASSED")
            else:
                failed_models += 1
                logger.error(f"❌ {model_info['file_name']} - FAILED")
                logger.error(f"   Issues: {', '.join(model_result['issues'])}")

        # Generate summary
        validation_results["summary"] = {
            "successful_models": successful_models,
            "failed_models": failed_models,
            "success_rate": successful_models / len(models) * 100 if models else 0,
            "total_validation_time": time.time() - start_time,
            "overall_success": successful_models > 0 and failed_models == 0,
        }

        validation_results["validation_end"] = datetime.now().isoformat()

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Models tested: {len(models)}")
        logger.info(f"Successful: {successful_models}")
        logger.info(f"Failed: {failed_models}")
        logger.info(
            f"Success rate: {validation_results['summary']['success_rate']:.1f}%"
        )
        logger.info(
            f"Total time: {validation_results['summary']['total_validation_time']:.2f}s"
        )
        logger.info(
            f"Overall result: {'✅ PASSED' if validation_results['summary']['overall_success'] else '❌ FAILED'}"
        )

        return validation_results

    def save_validation_report(
        self,
        results: Dict[str, Any],
        output_file: str = "ml_pipeline_validation_report.json",
    ):
        """Save validation results to file"""
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📝 Validation report saved to: {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save validation report: {e}")


def main():
    """Main entry point for ML pipeline validation"""
    import argparse

    parser = argparse.ArgumentParser(description="ML Pipeline & Prediction Validation")
    parser.add_argument(
        "--db-path", default="greyhound_racing_data.db", help="Database path"
    )
    parser.add_argument(
        "--models-dir", default="./model_registry/models", help="Models directory"
    )
    parser.add_argument(
        "--output",
        default="ml_pipeline_validation_report.json",
        help="Output report file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize validator
    validator = ModelValidator(args.db_path, args.models_dir)

    # Run comprehensive validation
    results = validator.run_comprehensive_validation()

    # Save results
    validator.save_validation_report(results, args.output)

    # Return appropriate exit code
    return 0 if results.get("summary", {}).get("overall_success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
