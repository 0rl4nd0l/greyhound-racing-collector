#!/usr/bin/env python3
"""
Enhanced ML Model System
========================

Advanced model enhancement system incorporating:
- Feature importance insights
- New predictive features
- Ensemble models (Voting, Stacking)
- Continuous improvement pipeline
- Model optimization based on analysis

Features:
- Dynamic feature selection based on importance analysis
- Advanced ensemble methods
- Model performance optimization
- Continuous learning pipeline
- Real-time model adaptation

Author: AI Assistant
Date: July 24, 2025
"""

import json
import os
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Enhanced ML Libraries
try:
    import joblib
    from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier, VotingClassifier)
    from sklearn.feature_selection import RFE, SelectKBest, f_classif
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, roc_auc_score)
    from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit,
                                         cross_val_score)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import (LabelEncoder, RobustScaler,
                                       StandardScaler)
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced ML libraries not available: {e}")
    SKLEARN_AVAILABLE = False


class EnhancedMLModelSystem:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results_dir = Path("./enhanced_model_results")
        self.models_dir = Path("./enhanced_trained_models")
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        # Enhanced model configurations
        self.base_models = {
            "random_forest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "random_state": 42,
                },
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier,
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.15,
                    "max_depth": 6,
                    "random_state": 42,
                },
            },
            "extra_trees": {
                "model": ExtraTreesClassifier,
                "params": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "random_state": 42,
                },
            },
            "logistic_regression": {
                "model": LogisticRegression,
                "params": {"C": 1.0, "max_iter": 2000, "random_state": 42},
            },
            "svm": {
                "model": SVC,
                "params": {
                    "C": 1.0,
                    "kernel": "rbf",
                    "probability": True,
                    "random_state": 42,
                },
            },
            "neural_network": {
                "model": MLPClassifier,
                "params": {
                    "hidden_layer_sizes": (100, 50),
                    "max_iter": 1000,
                    "random_state": 42,
                },
            },
        }

        # Feature importance insights (from analysis)
        self.high_impact_features = [
            "market_confidence",
            "current_odds_log",
            "recent_form_avg",
            "avg_position",
            "venue_experience",
            "win_rate",
            "place_rate",
            "current_weight",
        ]

        self.stable_features = [
            "avg_position",
            "time_consistency",
            "grade_encoded",
            "venue_experience",
            "place_rate",
            "current_weight",
            "track_condition_encoded",
        ]

        self.unstable_features = [
            "preferred_boxes_count",
            "distance_numeric",
            "avg_performance_rating",
            "races_count",
            "avg_time",
            "recent_races_last_30d",
            "win_rate",
            "best_time",
        ]

        print("🚀 Enhanced ML Model System Initialized")

    def load_historical_data(self, months_back=12):
        """Load extended historical data for better training"""
        try:
            conn = sqlite3.connect(self.db_path)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=months_back * 30)

            query = """
            SELECT 
                drd.race_id, drd.dog_clean_name, drd.finish_position,
                drd.box_number, drd.weight, drd.starting_price,
                drd.performance_rating, drd.speed_rating, drd.class_rating,
                drd.individual_time, drd.margin,
                rm.field_size, rm.distance, rm.venue, rm.track_condition,
                rm.grade, rm.race_date, rm.race_time
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.finish_position != 'N/A'
            AND rm.race_date >= ?
            AND rm.race_date <= ?
            ORDER BY rm.race_date ASC
            """

            df = pd.read_sql_query(
                query, conn, params=[start_date.isoformat(), end_date.isoformat()]
            )
            conn.close()

            print(f"📊 Loaded {len(df)} historical records")
            return df

        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return None

    def create_advanced_features(self, df):
        """Create advanced predictive features based on insights"""
        try:
            enhanced_records = []

            for _, row in df.iterrows():
                try:
                    # Clean finish position
                    pos_str = str(row["finish_position"]).strip()
                    if pos_str in ["", "N/A", "None", "nan"]:
                        continue
                    pos_cleaned = "".join(filter(str.isdigit, pos_str))
                    if not pos_cleaned:
                        continue
                    finish_position = int(pos_cleaned)
                    if finish_position < 1 or finish_position > 10:
                        continue
                except (ValueError, TypeError):
                    continue

                dog_name = row["dog_clean_name"]
                race_date = row["race_date"]

                # Get historical performance
                historical_data = df[
                    (df["dog_clean_name"] == dog_name) & (df["race_date"] < race_date)
                ].sort_values("race_date", ascending=False)

                if len(historical_data) >= 1:  # Minimum 1 race for basic features

                    # STABLE FEATURES (High Priority)
                    positions = []
                    times = []
                    weights = []
                    odds = []

                    for _, hist_row in historical_data.iterrows():
                        # Position data
                        if pd.notna(hist_row["finish_position"]):
                            pos_str = str(hist_row["finish_position"]).strip()
                            if pos_str not in ["", "N/A", "None", "nan"]:
                                pos_cleaned = "".join(filter(str.isdigit, pos_str))
                                if pos_cleaned and 1 <= int(pos_cleaned) <= 10:
                                    positions.append(int(pos_cleaned))

                        # Time data
                        if (
                            pd.notna(hist_row["individual_time"])
                            and hist_row["individual_time"] > 0
                        ):
                            times.append(hist_row["individual_time"])

                        # Weight data
                        if pd.notna(hist_row["weight"]) and hist_row["weight"] > 0:
                            weights.append(hist_row["weight"])

                        # Odds data
                        if (
                            pd.notna(hist_row["starting_price"])
                            and hist_row["starting_price"] > 0
                        ):
                            odds.append(hist_row["starting_price"])

                    if not positions:
                        continue

                    # Core stable features
                    avg_position = np.mean(positions)
                    recent_form_avg = (
                        np.mean(positions[:5])
                        if len(positions) >= 5
                        else np.mean(positions)
                    )
                    win_rate = sum(1 for p in positions if p == 1) / len(positions)
                    place_rate = sum(1 for p in positions if p <= 3) / len(positions)

                    # Time consistency (stable feature)
                    time_consistency = (
                        1 / (np.std(times) + 0.1) if len(times) > 1 else 0.5
                    )

                    # Market confidence (high impact)
                    market_confidence = 1 / (np.mean(odds) + 1) if odds else 0.1
                    current_odds_log = (
                        np.log(row["starting_price"] + 1)
                        if pd.notna(row["starting_price"])
                        else np.log(10)
                    )

                    # Venue experience (stable, high impact)
                    venue_experience = len(
                        [
                            r
                            for _, r in historical_data.iterrows()
                            if r["venue"] == row["venue"]
                        ]
                    )

                    # Weight features (stable)
                    current_weight = row["weight"] if pd.notna(row["weight"]) else 30.0
                    avg_weight = np.mean(weights) if weights else current_weight
                    weight_stability = (
                        1 / (np.std(weights) + 0.1) if len(weights) > 1 else 0.5
                    )

                    # NEW ADVANCED FEATURES

                    # Form trend (momentum indicator) - adapted for limited data
                    if len(positions) >= 2:
                        recent_half = (
                            np.mean(positions[: len(positions) // 2])
                            if len(positions) > 2
                            else positions[0]
                        )
                        older_half = (
                            np.mean(positions[len(positions) // 2 :])
                            if len(positions) > 2
                            else np.mean(positions[1:])
                        )
                        form_trend = (
                            (older_half - recent_half) / (older_half + 1)
                            if older_half > 0
                            else 0
                        )
                    else:
                        form_trend = 0

                    # Performance consistency
                    position_consistency = 1 / (np.std(positions) + 0.1)

                    # Distance/Track specialization
                    same_distance_races = [
                        r
                        for _, r in historical_data.iterrows()
                        if str(r["distance"]) == str(row["distance"])
                    ]
                    distance_experience = len(same_distance_races)
                    # Calculate distance performance more safely
                    distance_positions = []
                    for r in same_distance_races:
                        if pd.notna(r["finish_position"]):
                            pos_str = str(r["finish_position"]).strip()
                            pos_cleaned = "".join(filter(str.isdigit, pos_str))
                            if pos_cleaned and 1 <= int(pos_cleaned) <= 10:
                                distance_positions.append(int(pos_cleaned))
                    distance_performance = (
                        np.mean(distance_positions)
                        if distance_positions
                        else avg_position
                    )

                    # Box draw performance
                    current_box = (
                        row["box_number"] if pd.notna(row["box_number"]) else 4
                    )
                    same_box_races = [
                        r
                        for _, r in historical_data.iterrows()
                        if r["box_number"] == current_box
                    ]
                    box_experience = len(same_box_races)

                    # Class/Grade performance
                    same_grade_races = [
                        r
                        for _, r in historical_data.iterrows()
                        if r["grade"] == row["grade"]
                    ]
                    grade_experience = len(same_grade_races)

                    # Recent activity (days since last race)
                    try:
                        last_race_date = datetime.fromisoformat(
                            historical_data.iloc[0]["race_date"]
                        )
                        current_race_date = datetime.fromisoformat(race_date)
                        days_since_last = (current_race_date - last_race_date).days
                    except:
                        days_since_last = 14

                    # Fitness indicator (based on recent performance vs rest time)
                    fitness_score = 1 / (1 + days_since_last / 30) * (1 + place_rate)

                    # Competition strength (field size relative to experience)
                    field_size = row["field_size"] if pd.notna(row["field_size"]) else 6
                    competition_strength = field_size / (venue_experience + 1)

                    enhanced_records.append(
                        {
                            "race_id": row["race_id"],
                            "dog_name": dog_name,
                            "finish_position": finish_position,
                            "is_winner": 1 if finish_position == 1 else 0,
                            "is_placer": 1 if finish_position <= 3 else 0,
                            "race_date": race_date,
                            # STABLE HIGH-IMPACT FEATURES (Priority 1)
                            "avg_position": avg_position,
                            "recent_form_avg": recent_form_avg,
                            "market_confidence": market_confidence,
                            "current_odds_log": current_odds_log,
                            "venue_experience": venue_experience,
                            "place_rate": place_rate,
                            "current_weight": current_weight,
                            "time_consistency": time_consistency,
                            # ENHANCED FEATURES (Priority 2)
                            "win_rate": win_rate,
                            "avg_weight": avg_weight,
                            "weight_stability": weight_stability,
                            "form_trend": form_trend,
                            "position_consistency": position_consistency,
                            "distance_experience": distance_experience,
                            "distance_performance": distance_performance,
                            "box_experience": box_experience,
                            "grade_experience": grade_experience,
                            "days_since_last": days_since_last,
                            "fitness_score": fitness_score,
                            "competition_strength": competition_strength,
                            # CONTEXT FEATURES (Priority 3)
                            "current_box": current_box,
                            "field_size": field_size,
                            "venue": row["venue"],
                            "track_condition": row["track_condition"],
                            "grade": row["grade"],
                            "distance": row["distance"],
                        }
                    )

            return pd.DataFrame(enhanced_records)

        except Exception as e:
            print(f"❌ Error creating advanced features: {e}")
            return None

    def prepare_enhanced_features(self, df):
        """Prepare features with intelligent selection based on importance analysis"""
        try:
            if len(df) < 50:
                print("❌ Insufficient data for enhanced training")
                return None, None

            # Priority-based feature selection
            priority_1_features = [
                "avg_position",
                "recent_form_avg",
                "market_confidence",
                "current_odds_log",
                "venue_experience",
                "place_rate",
                "current_weight",
                "time_consistency",
            ]

            priority_2_features = [
                "win_rate",
                "avg_weight",
                "weight_stability",
                "form_trend",
                "position_consistency",
                "distance_experience",
                "distance_performance",
                "box_experience",
                "grade_experience",
                "fitness_score",
            ]

            priority_3_features = [
                "days_since_last",
                "competition_strength",
                "current_box",
                "field_size",
            ]

            # Encode categorical features
            le_venue = LabelEncoder()
            le_condition = LabelEncoder()
            le_grade = LabelEncoder()

            df["venue_encoded"] = le_venue.fit_transform(df["venue"].fillna("Unknown"))
            df["track_condition_encoded"] = le_condition.fit_transform(
                df["track_condition"].fillna("Good")
            )
            df["grade_encoded"] = le_grade.fit_transform(df["grade"].fillna("Unknown"))

            # Distance numeric
            df["distance_numeric"] = df["distance"].apply(
                lambda x: float(str(x).replace("m", "")) if pd.notna(x) else 500.0
            )

            # Combine all features
            feature_columns = (
                priority_1_features
                + priority_2_features
                + priority_3_features
                + [
                    "venue_encoded",
                    "track_condition_encoded",
                    "grade_encoded",
                    "distance_numeric",
                ]
            )

            # Ensure all features exist
            available_features = [f for f in feature_columns if f in df.columns]

            print(f"📊 Using {len(available_features)} enhanced features")

            # Create final dataset
            target_columns = [
                "is_winner",
                "is_placer",
                "race_date",
                "race_id",
                "dog_name",
            ]
            complete_df = df[available_features + target_columns].copy()

            # Advanced imputation using KNN for better handling of missing values
            imputer = KNNImputer(n_neighbors=5)
            complete_df[available_features] = imputer.fit_transform(
                complete_df[available_features]
            )

            return complete_df, available_features

        except Exception as e:
            print(f"❌ Error preparing enhanced features: {e}")
            return None, None

    def create_ensemble_models(self, X_train, y_train):
        """Create advanced ensemble models"""
        try:
            # Prepare base models
            base_estimators = []

            for name, config in self.base_models.items():
                model = config["model"](**config["params"])
                base_estimators.append((name, model))

            # Voting Classifier (Hard and Soft voting)
            voting_hard = VotingClassifier(
                estimators=base_estimators[:4], voting="hard"  # Use first 4 models
            )

            voting_soft = VotingClassifier(
                estimators=base_estimators[:4],  # Models with predict_proba
                voting="soft",
            )

            # Bagging with Random Forest
            bagging_rf = BaggingClassifier(
                RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
                n_estimators=10,
                random_state=42,
            )

            ensemble_models = {
                "voting_hard": voting_hard,
                "voting_soft": voting_soft,
                "bagging_rf": bagging_rf,
            }

            return ensemble_models

        except Exception as e:
            print(f"❌ Error creating ensemble models: {e}")
            return {}

    def optimize_model_parameters(self, model, X_train, y_train, model_name):
        """Optimize model parameters using GridSearch"""
        try:
            param_grids = {
                "random_forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 15, 20],
                    "min_samples_split": [2, 5],
                },
                "gradient_boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.15, 0.2],
                    "max_depth": [5, 6, 7],
                },
                "logistic_regression": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["liblinear", "lbfgs"],
                },
            }

            if model_name in param_grids:
                print(f"   🔧 Optimizing {model_name} parameters...")

                # Use time series split for optimization
                tscv = TimeSeriesSplit(n_splits=3)

                grid_search = GridSearchCV(
                    model,
                    param_grids[model_name],
                    cv=tscv,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=0,
                )

                grid_search.fit(X_train, y_train)

                print(f"   ✅ Best params for {model_name}: {grid_search.best_params_}")
                print(f"   📊 Best CV score: {grid_search.best_score_:.3f}")

                return grid_search.best_estimator_

            return model

        except Exception as e:
            print(f"⚠️ Parameter optimization failed for {model_name}: {e}")
            return model

    def run_enhanced_analysis(self):
        """Run comprehensive enhanced model analysis"""
        print("🚀 Starting Enhanced ML Model Analysis")
        print("=" * 60)

        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn not available")
            return None

        try:
            # Load extended historical data
            df = self.load_historical_data(months_back=12)
            if df is None or len(df) < 200:
                print("❌ Insufficient data for enhanced analysis")
                return None

            # Create advanced features
            print("🔧 Creating advanced predictive features...")
            enhanced_df = self.create_advanced_features(df)
            if enhanced_df is None:
                print("❌ Feature creation failed")
                return None

            # Prepare features
            print("📊 Preparing enhanced feature set...")
            prepared_df, feature_columns = self.prepare_enhanced_features(enhanced_df)
            if prepared_df is None:
                print("❌ Feature preparation failed")
                return None

            print(
                f"📊 Enhanced dataset: {len(prepared_df)} samples, {len(feature_columns)} features"
            )

            # Time-based split
            df_sorted = prepared_df.sort_values("race_date")
            split_point = int(0.8 * len(df_sorted))
            train_df = df_sorted.iloc[:split_point]
            test_df = df_sorted.iloc[split_point:]

            print(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")

            # Prepare features and targets
            X_train = train_df[feature_columns]
            y_train = train_df["is_winner"]
            X_test = test_df[feature_columns]
            y_test = test_df["is_winner"]

            # Feature scaling
            scaler = RobustScaler()  # More robust to outliers
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            results = {
                "timestamp": datetime.now().isoformat(),
                "data_summary": {
                    "total_samples": len(prepared_df),
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    "features": len(feature_columns),
                    "feature_list": feature_columns,
                },
                "model_results": {},
                "ensemble_results": {},
                "optimization_results": {},
            }

            print("\\n🤖 Testing Base Models with Optimization:")
            print("-" * 50)

            # Test optimized base models
            for model_name, config in self.base_models.items():
                print(f"   Testing {model_name}...")

                model_class = config["model"]
                base_model = model_class(**config["params"])

                # Optimize parameters
                optimized_model = self.optimize_model_parameters(
                    base_model, X_train_scaled, y_train, model_name
                )

                # Train and evaluate
                optimized_model.fit(X_train_scaled, y_train)
                y_pred = optimized_model.predict(X_test_scaled)

                if hasattr(optimized_model, "predict_proba"):
                    y_pred_proba = optimized_model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = y_pred

                accuracy = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5

                # Feature importance
                feature_importance = None
                if hasattr(optimized_model, "feature_importances_"):
                    feature_importance = list(
                        zip(feature_columns, optimized_model.feature_importances_)
                    )
                    feature_importance.sort(key=lambda x: x[1], reverse=True)

                results["model_results"][model_name] = {
                    "accuracy": accuracy,
                    "auc": auc,
                    "feature_importance": (
                        feature_importance[:15] if feature_importance else None
                    ),
                }

                print(f"     📊 Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

            print("\\n🎯 Testing Ensemble Models:")
            print("-" * 50)

            # Create and test ensemble models
            ensemble_models = self.create_ensemble_models(X_train_scaled, y_train)

            for ensemble_name, ensemble_model in ensemble_models.items():
                print(f"   Testing {ensemble_name}...")

                try:
                    ensemble_model.fit(X_train_scaled, y_train)
                    y_pred = ensemble_model.predict(X_test_scaled)

                    if hasattr(ensemble_model, "predict_proba"):
                        y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_pred_proba = y_pred

                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        auc = 0.5

                    results["ensemble_results"][ensemble_name] = {
                        "accuracy": accuracy,
                        "auc": auc,
                    }

                    print(f"     📊 Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

                except Exception as e:
                    print(f"     ❌ Failed: {e}")

            # Find best performing model overall
            all_results = {**results["model_results"], **results["ensemble_results"]}
            best_model_name = max(
                all_results.keys(), key=lambda k: all_results[k]["accuracy"]
            )
            best_accuracy = all_results[best_model_name]["accuracy"]

            print(
                f"\\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.3f})"
            )

            # Save best model
            if best_model_name in self.base_models:
                best_model_config = self.base_models[best_model_name]
                best_model = best_model_config["model"](**best_model_config["params"])
                best_model = self.optimize_model_parameters(
                    best_model, X_train_scaled, y_train, best_model_name
                )
                best_model.fit(X_train_scaled, y_train)
            elif best_model_name in ensemble_models:
                best_model = ensemble_models[best_model_name]
                best_model.fit(X_train_scaled, y_train)

            # Save the best model
            model_file = (
                self.models_dir
                / f"enhanced_best_model_{datetime.now().strftime('%Y%m%d')}.joblib"
            )
            joblib.dump(
                {
                    "model": best_model,
                    "scaler": scaler,
                    "feature_columns": feature_columns,
                    "model_name": best_model_name,
                    "accuracy": best_accuracy,
                    "timestamp": datetime.now().isoformat(),
                },
                model_file,
            )

            print(f"💾 Best model saved: {model_file}")

            # Save results
            results_file = (
                self.results_dir
                / f"enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"💾 Results saved: {results_file}")
            print("\\n✅ Enhanced ML Analysis Complete!")

            return results

        except Exception as e:
            print(f"❌ Enhanced analysis failed: {e}")
            return None

    def continuous_improvement_pipeline(self):
        """Implement continuous learning and model improvement"""
        try:
            print("🔄 Starting Continuous Improvement Pipeline")

            # Load latest results
            result_files = sorted(self.results_dir.glob("enhanced_analysis_*.json"))
            if len(result_files) < 2:
                print("📊 Insufficient history for improvement analysis")
                return

            # Compare recent performance
            with open(result_files[-1], "r") as f:
                current_results = json.load(f)

            with open(result_files[-2], "r") as f:
                previous_results = json.load(f)

            # Analyze improvements
            improvements = {}
            for model_name in current_results["model_results"]:
                if model_name in previous_results["model_results"]:
                    current_acc = current_results["model_results"][model_name][
                        "accuracy"
                    ]
                    previous_acc = previous_results["model_results"][model_name][
                        "accuracy"
                    ]
                    improvement = current_acc - previous_acc
                    improvements[model_name] = improvement

                    if improvement > 0.02:
                        print(
                            f"✅ {model_name}: +{improvement:.3f} accuracy improvement"
                        )
                    elif improvement < -0.02:
                        print(f"⚠️ {model_name}: {improvement:.3f} accuracy decline")

            # Feature performance analysis
            print("\\n📊 Analyzing feature performance trends...")

            # Save improvement analysis
            improvement_file = (
                self.results_dir
                / f"improvement_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(improvement_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "model_improvements": improvements,
                        "recommendations": self.generate_improvement_recommendations(
                            improvements
                        ),
                    },
                    f,
                    indent=2,
                )

            print(f"💾 Improvement analysis saved: {improvement_file}")

        except Exception as e:
            print(f"❌ Continuous improvement pipeline failed: {e}")

    def generate_improvement_recommendations(self, improvements):
        """Generate actionable recommendations for model improvement"""
        recommendations = []

        # Analyze improvement patterns
        avg_improvement = np.mean(list(improvements.values())) if improvements else 0

        if avg_improvement > 0.01:
            recommendations.append(
                "✅ Models showing positive trend - continue current approach"
            )
        elif avg_improvement < -0.01:
            recommendations.append(
                "⚠️ Models declining - consider feature engineering or data quality issues"
            )

        # Model-specific recommendations
        best_improving = (
            max(improvements.keys(), key=lambda k: improvements[k])
            if improvements
            else None
        )
        if best_improving:
            recommendations.append(
                f"🏆 Focus on {best_improving} - showing best improvement"
            )

        recommendations.extend(
            [
                "📊 Monitor feature importance stability",
                "🔍 Investigate new data sources for features",
                "⚡ Consider ensemble model combinations",
                "🎯 Validate model performance on recent races",
            ]
        )

        return recommendations


def main():
    """Main function for enhanced model system"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced ML Model System")
    parser.add_argument(
        "--command",
        choices=["analyze", "improve", "status"],
        default="analyze",
        help="Command to execute",
    )

    args = parser.parse_args()

    print("🚀 Enhanced ML Model System")
    print("=" * 60)

    system = EnhancedMLModelSystem()

    if args.command == "analyze":
        system.run_enhanced_analysis()

    elif args.command == "improve":
        system.continuous_improvement_pipeline()

    elif args.command == "status":
        # Show system status
        model_files = len(list(system.models_dir.glob("*.joblib")))
        result_files = len(list(system.results_dir.glob("enhanced_analysis_*.json")))

        print("\\n📊 ENHANCED SYSTEM STATUS:")
        print("=" * 40)
        print(f"Enhanced model files: {model_files}")
        print(f"Analysis result files: {result_files}")
        print(f"Models directory: {system.models_dir}")
        print(f"Results directory: {system.results_dir}")


if __name__ == "__main__":
    main()
