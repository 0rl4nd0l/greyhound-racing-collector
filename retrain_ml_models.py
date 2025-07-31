#!/usr/bin/env python3
"""
Comprehensive ML Model Retraining
==================================

Extract training data from embedded historical race data and retrain ML models
for improved prediction accuracy and differentiation.
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from advanced_ml_system_v2 import AdvancedMLSystemV2
from enhanced_pipeline_v2 import EnhancedPipelineV2


def extract_training_data():
    """Extract training data from all race files"""
    print("🔍 Extracting training data from race files...")

    pipeline = EnhancedPipelineV2()
    training_samples = []
    processed_files = 0

    # Process all race files
    race_files = [f for f in os.listdir("upcoming_races") if f.endswith(".csv")]

    for race_file in race_files[:20]:  # Process first 20 files to get substantial data
        try:
            race_file_path = os.path.join("upcoming_races", race_file)
            race_df = pipeline._load_race_file(race_file_path)
            dogs = pipeline._extract_participating_dogs(race_df, race_file_path)

            for dog in dogs:
                historical_data = dog.get("historical_data", [])

                # Each historical race can be a training sample
                for i, race_record in enumerate(historical_data):
                    # Skip the first record (current race)
                    if i == 0:
                        continue

                    # Create synthetic dog_info for historical race
                    historical_dog_info = {
                        "name": dog["name"],
                        "box": race_record.get("BOX", 1),
                        "historical_data": historical_data[i:],  # Use remaining history
                    }

                    # Extract features for this historical point
                    features = pipeline._extract_features_from_historical_data(
                        historical_dog_info, race_file_path
                    )

                    # Create target from finish position (1 = win, 0 = loss)
                    finish_pos = race_record.get("PLC")
                    if finish_pos and str(finish_pos).replace(".0", "").isdigit():
                        target = 1 if int(float(finish_pos)) == 1 else 0
                        features["target"] = target
                        features["dog_name"] = dog["name"]
                        features["race_date"] = race_record.get("DATE", "")
                        features["race_file"] = race_file

                        training_samples.append(features)

            processed_files += 1
            if processed_files % 5 == 0:
                print(
                    f"   Processed {processed_files} files, {len(training_samples)} samples so far..."
                )

        except Exception as e:
            print(f"   ⚠️ Error processing {race_file}: {e}")

    print(
        f"✅ Extracted {len(training_samples)} training samples from {processed_files} files"
    )
    return training_samples


def prepare_training_data(training_samples):
    """Prepare training data for ML models"""
    print("📊 Preparing training data...")

    df = pd.DataFrame(training_samples)

    # Remove non-feature columns
    feature_columns = [
        col
        for col in df.columns
        if col not in ["target", "dog_name", "race_date", "race_file"]
    ]

    X = df[feature_columns]
    y = df["target"]

    # Handle missing values
    X = X.fillna(0.0)

    print(f"   Features: {len(feature_columns)}")
    print(f"   Samples: {len(X)}")
    print(f"   Win rate: {y.mean():.3f}")

    # Show feature distribution
    print("\n   Feature Statistics:")
    key_features = [
        "weighted_recent_form",
        "venue_win_rate",
        "speed_trend",
        "recent_momentum",
    ]
    for feature in key_features:
        if feature in X.columns:
            print(
                f"     {feature}: mean={X[feature].mean():.3f}, std={X[feature].std():.3f}"
            )

    return X, y, feature_columns


def retrain_models(X, y, feature_columns):
    """Retrain ML models with new data"""
    print("\n🤖 Retraining ML models...")

    # Initialize ML system
    ml_system = AdvancedMLSystemV2()

    # Convert to enhanced features format expected by ML system
    enhanced_features_list = []
    for i in range(len(X)):
        features_dict = X.iloc[i].to_dict()
        enhanced_features_list.append({"features": features_dict, "target": y.iloc[i]})

    # Prepare training data
    X_train, y_train, feature_cols, df_train = ml_system.prepare_training_data(
        enhanced_features_list
    )

    if X_train is None:
        print("❌ Failed to prepare training data")
        return None

    # Train models
    results = ml_system.train_advanced_models(X_train, y_train, feature_cols)

    # Save the best performing model to the registry
    best_model_name = max(results.keys(), key=lambda k: results[k]["cv_auc_mean"])
    best_model = results[best_model_name]

    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split

        from model_registry import get_model_registry

        # Get the model registry
        registry = get_model_registry()

        # Calculate additional metrics for the registry
        X_test_split, X_val_split, y_test_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Make predictions for metrics calculation
        if best_model_name in ["neural_network", "logistic_regression_optimized"]:
            X_scaled = ml_system.scaler.fit_transform(X_test_split)
            y_pred = best_model["model"].predict(X_scaled)
        else:
            y_pred = best_model["model"].predict(X_test_split)

        # Calculate comprehensive metrics
        f1 = f1_score(y_test_split, y_pred, average="binary", zero_division=0)
        precision = precision_score(
            y_test_split, y_pred, average="binary", zero_division=0
        )
        recall = recall_score(y_test_split, y_pred, average="binary", zero_division=0)

        # Prepare performance metrics
        performance_metrics = {
            "accuracy": best_model["cv_accuracy_mean"],
            "auc": best_model["cv_auc_mean"],
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }

        # Prepare training info
        training_info = {
            "training_samples": len(X_train),
            "test_samples": len(X_test_split),
            "training_duration": 0.0,  # Could be tracked if needed
            "validation_method": "time_series_cv",
            "cv_scores": best_model.get("cv_accuracy_scores", []),
            "is_ensemble": False,
            "ensemble_components": [],
            "data_quality_score": 0.8,  # Estimated based on feature completeness
            "inference_time_ms": 5.0,  # Estimated
        }

        # Register the model in the registry
        model_id = registry.register_model(
            model_obj=best_model["model"],
            scaler_obj=ml_system.scaler,
            model_name=best_model_name,
            model_type="retrained_model",
            performance_metrics=performance_metrics,
            training_info=training_info,
            feature_names=feature_cols,
            hyperparameters=ml_system.model_configs.get(best_model_name, {}).get(
                "params", {}
            ),
            notes=f"Retrained model from {len(X_train)} historical samples",
        )

        print(f"✅ Model registered in registry: {model_id}")
        print(
            f"   📊 Performance: Acc={performance_metrics['accuracy']:.3f}, AUC={performance_metrics['auc']:.3f}, F1={performance_metrics['f1_score']:.3f}"
        )

        # Also save legacy format for backward compatibility
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filepath = (
            f"comprehensive_trained_models/retrained_model_{timestamp}.joblib"
        )

        # Ensure directory exists
        os.makedirs("comprehensive_trained_models", exist_ok=True)

        model_data = {
            "model": best_model["model"],
            "model_name": best_model_name,
            "accuracy": best_model["cv_accuracy_mean"],
            "auc": best_model["cv_auc_mean"],
            "feature_columns": feature_cols,
            "scaler": ml_system.scaler,
            "timestamp": timestamp,
            "training_samples": len(X_train),
            "registry_id": model_id,
        }

        joblib.dump(model_data, model_filepath)
        print(f"📁 Legacy model also saved to: {model_filepath}")

        return ml_system, model_filepath

    except ImportError:
        print("⚠️ Model registry not available, using legacy save only...")
        # Fallback to legacy save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filepath = (
            f"comprehensive_trained_models/retrained_model_{timestamp}.joblib"
        )

        # Ensure directory exists
        os.makedirs("comprehensive_trained_models", exist_ok=True)

        model_data = {
            "model": best_model["model"],
            "model_name": best_model_name,
            "accuracy": best_model["cv_accuracy_mean"],
            "auc": best_model["cv_auc_mean"],
            "feature_columns": feature_cols,
            "scaler": ml_system.scaler,
            "timestamp": timestamp,
            "training_samples": len(X_train),
        }

        joblib.dump(model_data, model_filepath)
        print(f"✅ Best model ({best_model_name}) saved to {model_filepath}")
        print(
            f"   Accuracy: {best_model['cv_accuracy_mean']:.3f} ± {best_model['cv_accuracy_std']:.3f}"
        )
        print(
            f"   AUC: {best_model['cv_auc_mean']:.3f} ± {best_model['cv_auc_std']:.3f}"
        )

        return ml_system, model_filepath

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        print("   Falling back to legacy save...")
        # Fallback to legacy save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filepath = (
            f"comprehensive_trained_models/retrained_model_{timestamp}.joblib"
        )

        # Ensure directory exists
        os.makedirs("comprehensive_trained_models", exist_ok=True)

        model_data = {
            "model": best_model["model"],
            "model_name": best_model_name,
            "accuracy": best_model["cv_accuracy_mean"],
            "auc": best_model["cv_auc_mean"],
            "feature_columns": feature_cols,
            "scaler": ml_system.scaler,
            "timestamp": timestamp,
            "training_samples": len(X_train),
        }

        joblib.dump(model_data, model_filepath)
        print(f"✅ Best model ({best_model_name}) saved to {model_filepath}")
        print(
            f"   Accuracy: {best_model['cv_accuracy_mean']:.3f} ± {best_model['cv_accuracy_std']:.3f}"
        )
        print(
            f"   AUC: {best_model['cv_auc_mean']:.3f} ± {best_model['cv_auc_std']:.3f}"
        )

        return ml_system, model_filepath


def test_retrained_model():
    """Test the retrained model on sample data"""
    print("\n🧪 Testing retrained model...")

    pipeline = EnhancedPipelineV2()
    result = pipeline.predict_race_file(
        "upcoming_races/Race 3 - DUBBO - 2025-07-26.csv"
    )

    if result["success"]:
        scores = [pred["prediction_score"] for pred in result["predictions"]]
        print(f"   Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"   Score variance: {np.var(scores):.6f}")

        print("\n   Top 3 predictions:")
        for i, pred in enumerate(result["predictions"][:3]):
            print(f"     {i+1}. {pred['dog_name']:15}: {pred['prediction_score']:.3f}")

        # Check if we have good differentiation now
        if np.var(scores) > 0.001:
            print("✅ Model shows good score differentiation!")
        else:
            print("⚠️ Model still shows limited score differentiation")
    else:
        print(f"❌ Test failed: {result.get('error', 'Unknown error')}")


def main():
    """Main retraining process"""
    print("🚀 Comprehensive ML Model Retraining")
    print("=" * 50)

    # Step 1: Extract training data
    training_samples = extract_training_data()

    if len(training_samples) < 50:
        print(f"❌ Insufficient training data: {len(training_samples)} samples")
        print("   Need at least 50 samples for reliable training")
        return

    # Step 2: Prepare data
    X, y, feature_columns = prepare_training_data(training_samples)

    # Step 3: Retrain models
    ml_system, model_filepath = retrain_models(X, y, feature_columns)

    if ml_system is None:
        print("❌ Model retraining failed")
        return

    # Step 4: Test retrained model
    test_retrained_model()

    print("\n🎉 Retraining process completed!")
    print(f"   New model saved: {model_filepath}")


if __name__ == "__main__":
    main()
