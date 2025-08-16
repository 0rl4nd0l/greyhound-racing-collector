#!/usr/bin/env python3
"""
Advanced System Performance Analyzer
====================================

Comprehensive analysis tool for the greyhound racing prediction system.
Analyzes data quality, model performance, and provides optimization recommendations.
"""

import json
import os
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ML Libraries for analysis
try:
    import joblib
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    from sklearn.model_selection import cross_val_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AdvancedSystemAnalyzer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.analysis_dir = Path("./system_analysis")
        self.analysis_dir.mkdir(exist_ok=True)

        # Performance benchmarks
        self.performance_benchmarks = {
            "accuracy_excellent": 0.85,
            "accuracy_good": 0.75,
            "accuracy_poor": 0.60,
            "data_quality_excellent": 0.95,
            "data_quality_good": 0.85,
            "data_quality_poor": 0.70,
        }

        print("🔍 Advanced System Analyzer Initialized")

    def analyze_data_quality(self):
        """Comprehensive data quality analysis"""
        print("\n📊 ANALYZING DATA QUALITY...")

        try:
            conn = sqlite3.connect(self.db_path)

            # Basic statistics
            race_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM race_metadata", conn
            ).iloc[0]["count"]
            dog_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM dog_race_data", conn
            ).iloc[0]["count"]
            unique_dogs = pd.read_sql_query(
                "SELECT COUNT(DISTINCT dog_clean_name) as count FROM dog_race_data",
                conn,
            ).iloc[0]["count"]
            unique_venues = pd.read_sql_query(
                "SELECT COUNT(DISTINCT venue) as count FROM race_metadata", conn
            ).iloc[0]["count"]

            # Data completeness analysis
            race_data = pd.read_sql_query(
                """
                SELECT 
                    COUNT(*) as total_races,
                    COUNT(winner_name) as races_with_winners,
                    COUNT(track_condition) as races_with_track_condition,
                    COUNT(weather_condition) as races_with_weather,
                    COUNT(temperature) as races_with_temperature
                FROM race_metadata
            """,
                conn,
            )

            dog_data = pd.read_sql_query(
                """
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(finish_position) as entries_with_position,
                    COUNT(individual_time) as entries_with_time,
                    COUNT(starting_price) as entries_with_odds,
                    COUNT(weight) as entries_with_weight
                FROM dog_race_data
                WHERE dog_name IS NOT NULL AND dog_name != ''
            """,
                conn,
            )

            # Calculate completeness percentages
            race_completeness = {
                "winners": race_data.iloc[0]["races_with_winners"]
                / race_data.iloc[0]["total_races"],
                "track_condition": race_data.iloc[0]["races_with_track_condition"]
                / race_data.iloc[0]["total_races"],
                "weather": race_data.iloc[0]["races_with_weather"]
                / race_data.iloc[0]["total_races"],
                "temperature": race_data.iloc[0]["races_with_temperature"]
                / race_data.iloc[0]["total_races"],
            }

            dog_completeness = {
                "finish_position": dog_data.iloc[0]["entries_with_position"]
                / dog_data.iloc[0]["total_entries"],
                "individual_time": dog_data.iloc[0]["entries_with_time"]
                / dog_data.iloc[0]["total_entries"],
                "starting_price": dog_data.iloc[0]["entries_with_odds"]
                / dog_data.iloc[0]["total_entries"],
                "weight": dog_data.iloc[0]["entries_with_weight"]
                / dog_data.iloc[0]["total_entries"],
            }

            # Overall data quality score
            all_completeness = list(race_completeness.values()) + list(
                dog_completeness.values()
            )
            overall_quality = np.mean(all_completeness)

            conn.close()

            quality_analysis = {
                "basic_stats": {
                    "total_races": int(race_count),
                    "total_dog_entries": int(dog_count),
                    "unique_dogs": int(unique_dogs),
                    "unique_venues": int(unique_venues),
                    "avg_entries_per_race": round(dog_count / race_count, 1),
                },
                "data_completeness": {
                    "race_data": race_completeness,
                    "dog_data": dog_completeness,
                    "overall_quality_score": round(overall_quality, 3),
                },
                "quality_assessment": self._assess_quality(overall_quality),
            }

            return quality_analysis

        except Exception as e:
            print(f"❌ Error analyzing data quality: {e}")
            return None

    def analyze_model_performance(self):
        """Analyze current model performance"""
        print("\n🤖 ANALYZING MODEL PERFORMANCE...")

        try:
            models_dir = Path("./comprehensive_trained_models")
            if not models_dir.exists():
                return {"error": "No trained models directory found"}

            # Find latest model
            model_files = list(models_dir.glob("comprehensive_best_model_*.joblib"))
            if not model_files:
                return {"error": "No trained models found"}

            latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)

            # Load model metadata
            model_data = joblib.load(latest_model_file)

            model_analysis = {
                "model_info": {
                    "name": model_data.get("model_name", "Unknown"),
                    "accuracy": model_data.get("accuracy", 0),
                    "features_count": len(model_data.get("feature_columns", [])),
                    "timestamp": model_data.get("timestamp", ""),
                    "file_path": str(latest_model_file),
                },
                "feature_importance": self._analyze_feature_importance(model_data),
                "performance_assessment": self._assess_model_performance(
                    model_data.get("accuracy", 0)
                ),
            }

            return model_analysis

        except Exception as e:
            print(f"❌ Error analyzing model performance: {e}")
            return {"error": str(e)}

    def analyze_prediction_accuracy(self):
        """Analyze historical prediction accuracy"""
        print("\n🎯 ANALYZING PREDICTION ACCURACY...")

        try:
            predictions_dir = Path("./predictions")
            if not predictions_dir.exists():
                return {"error": "No predictions directory found"}

            prediction_files = list(predictions_dir.glob("*.json"))

            if not prediction_files:
                return {"error": "No prediction files found"}

            # Sample recent predictions for analysis
            recent_predictions = sorted(
                prediction_files, key=lambda x: x.stat().st_mtime, reverse=True
            )[:50]

            accuracy_stats = {
                "total_predictions_analyzed": len(recent_predictions),
                "average_confidence": 0,
                "prediction_distribution": {},
                "files_analyzed": len(recent_predictions),
            }

            # Analyze confidence scores from predictions
            confidence_scores = []
            for pred_file in recent_predictions[
                :10
            ]:  # Sample first 10 for detailed analysis
                try:
                    with open(pred_file, "r") as f:
                        pred_data = json.load(f)

                    if "predictions" in pred_data:
                        for pred in pred_data["predictions"]:
                            if "confidence_score" in pred:
                                confidence_scores.append(pred["confidence_score"])
                except:
                    continue

            if confidence_scores:
                accuracy_stats["average_confidence"] = round(
                    np.mean(confidence_scores), 3
                )
                accuracy_stats["confidence_std"] = round(np.std(confidence_scores), 3)

            return accuracy_stats

        except Exception as e:
            print(f"❌ Error analyzing prediction accuracy: {e}")
            return {"error": str(e)}

    def analyze_system_performance(self):
        """Analyze overall system performance with detailed logging"""
        import logging

        logger = logging.getLogger(__name__)

        print("\n⚡ ANALYZING SYSTEM PERFORMANCE...")
        logger.info("🔍 Starting comprehensive system performance analysis")

        start_time = time.time()
        analysis_steps = []

        try:
            # Step 1: Directory and file organization analysis
            step_start = time.time()
            logger.info(
                "📂 Step 1: Analyzing directory structure and file organization"
            )

            directories = {
                "upcoming_races": Path("./upcoming_races"),
                "predictions": Path("./predictions"),
                "comprehensive_model_results": Path("./comprehensive_model_results"),
                "form_guides": Path("./form_guides"),
                "gpt_enhanced_predictions": Path("./gpt_enhanced_predictions"),
                "historical_races": Path("./historical_races"),
                "processed": Path("./processed"),
                "unprocessed": Path("./unprocessed"),
            }

            file_stats = {}
            total_files_processed = 0

            for name, path in directories.items():
                dir_start = time.time()
                logger.info(f"  📁 Analyzing directory: {name} -> {path}")

                if path.exists():
                    files = list(path.glob("*"))
                    csv_files = [f for f in files if f.suffix == ".csv"]
                    json_files = [f for f in files if f.suffix == ".json"]
                    other_files = [
                        f for f in files if f.suffix not in [".csv", ".json"]
                    ]

                    # Analyze file ages
                    recent_files = []
                    old_files = []
                    current_time = time.time()

                    for file in files:
                        file_age_days = (current_time - file.stat().st_mtime) / (
                            24 * 3600
                        )
                        if file_age_days <= 7:
                            recent_files.append(file)
                        elif file_age_days > 30:
                            old_files.append(file)

                    file_stats[name] = {
                        "total_files": len(files),
                        "csv_files": len(csv_files),
                        "json_files": len(json_files),
                        "other_files": len(other_files),
                        "recent_files_7days": len(recent_files),
                        "old_files_30days": len(old_files),
                        "directory_size_mb": sum(
                            f.stat().st_size for f in files if f.is_file()
                        )
                        / (1024 * 1024),
                        "analysis_time_ms": round((time.time() - dir_start) * 1000, 2),
                    }

                    total_files_processed += len(files)
                    logger.info(
                        f"    ✅ {name}: {len(files)} files ({len(csv_files)} CSV, {len(json_files)} JSON, {len(recent_files)} recent)"
                    )

                else:
                    file_stats[name] = {
                        "error": "Directory not found",
                        "analysis_time_ms": round((time.time() - dir_start) * 1000, 2),
                    }
                    logger.warning(f"    ⚠️ {name}: Directory not found at {path}")

            step1_time = time.time() - step_start
            analysis_steps.append(
                {
                    "step": "Directory Analysis",
                    "time_seconds": step1_time,
                    "files_processed": total_files_processed,
                }
            )
            logger.info(
                f"  ✅ Step 1 completed in {step1_time:.2f}s - Processed {total_files_processed} files"
            )

            # Step 2: Database analysis
            step_start = time.time()
            logger.info("💾 Step 2: Analyzing database performance and size")

            db_analysis = {}
            if os.path.exists(self.db_path):
                db_size_bytes = os.path.getsize(self.db_path)
                db_size_mb = db_size_bytes / (1024 * 1024)
                db_analysis = {
                    "size_bytes": db_size_bytes,
                    "size_mb": round(db_size_mb, 2),
                    "size_gb": round(db_size_mb / 1024, 3),
                }

                # Try to get table counts
                try:
                    import sqlite3

                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()

                        # Get table list
                        cursor.execute(
                            "SELECT name FROM sqlite_master WHERE type='table';"
                        )
                        tables = cursor.fetchall()

                        table_stats = {}
                        for (table_name,) in tables:
                            try:
                                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                                count = cursor.fetchone()[0]
                                table_stats[table_name] = count
                                logger.info(
                                    f"    📊 Table {table_name}: {count:,} records"
                                )
                            except Exception as e:
                                table_stats[table_name] = f"Error: {e}"
                                logger.warning(
                                    f"    ⚠️ Could not count {table_name}: {e}"
                                )

                        db_analysis["tables"] = table_stats
                        db_analysis["total_tables"] = len(tables)

                except Exception as e:
                    logger.warning(f"    ⚠️ Could not analyze database tables: {e}")
                    db_analysis["table_analysis_error"] = str(e)

                logger.info(
                    f"  📊 Database: {db_size_mb:.2f} MB with {len(db_analysis.get('tables', {}))} tables"
                )
            else:
                db_analysis = {"error": "Database file not found"}
                logger.error(f"  ❌ Database not found: {self.db_path}")

            step2_time = time.time() - step_start
            analysis_steps.append(
                {"step": "Database Analysis", "time_seconds": step2_time}
            )
            logger.info(f"  ✅ Step 2 completed in {step2_time:.2f}s")

            # Step 3: System health assessment
            step_start = time.time()
            logger.info("🏥 Step 3: Assessing overall system health")

            system_health = self._assess_system_health_detailed(file_stats)

            step3_time = time.time() - step_start
            analysis_steps.append(
                {"step": "Health Assessment", "time_seconds": step3_time}
            )
            logger.info(
                f"  ✅ Step 3 completed in {step3_time:.2f}s - Health: {system_health['overall_status']}"
            )

            # Step 4: Performance metrics calculation
            step_start = time.time()
            logger.info("📈 Step 4: Calculating performance metrics")

            total_analysis_time = time.time() - start_time
            performance_metrics = {
                "analysis_duration_seconds": round(total_analysis_time, 2),
                "files_per_second": (
                    round(total_files_processed / total_analysis_time, 2)
                    if total_analysis_time > 0
                    else 0
                ),
                "analysis_steps": analysis_steps,
                "timestamp": datetime.now().isoformat(),
            }

            step4_time = time.time() - step_start
            analysis_steps.append(
                {"step": "Metrics Calculation", "time_seconds": step4_time}
            )
            logger.info(f"  ✅ Step 4 completed in {step4_time:.2f}s")

            # Compile final analysis
            performance_analysis = {
                "file_organization": file_stats,
                "database_analysis": db_analysis,
                "system_health": system_health,
                "performance_metrics": performance_metrics,
                "analysis_summary": {
                    "total_directories_analyzed": len(directories),
                    "total_files_found": total_files_processed,
                    "directories_healthy": sum(
                        1 for stats in file_stats.values() if "error" not in stats
                    ),
                    "analysis_success": True,
                },
            }

            logger.info(
                f"🎉 System performance analysis completed successfully in {total_analysis_time:.2f}s"
            )
            logger.info(
                f"📊 Summary: {len(directories)} directories, {total_files_processed} files, Health: {system_health['overall_status']}"
            )

            return performance_analysis

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Error analyzing system performance: {e}"
            logger.error(f"❌ {error_msg} (after {total_time:.2f}s)")
            logger.exception("Full exception details:")

            return {
                "error": error_msg,
                "analysis_duration_seconds": round(total_time, 2),
                "analysis_steps": analysis_steps,
                "timestamp": datetime.now().isoformat(),
            }

    def generate_recommendations(self, analysis_results):
        """Generate optimization recommendations"""
        print("\n💡 GENERATING RECOMMENDATIONS...")

        recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "advanced_enhancements": [],
        }

        # Data Quality Recommendations
        if "data_quality" in analysis_results:
            quality_score = analysis_results["data_quality"]["data_completeness"][
                "overall_quality_score"
            ]

            if quality_score < self.performance_benchmarks["data_quality_poor"]:
                recommendations["high_priority"].append(
                    {
                        "category": "Data Quality",
                        "issue": f"Low data completeness ({quality_score:.1%})",
                        "recommendation": "Implement data validation pipeline and fix missing data issues",
                        "impact": "High - Poor data quality directly affects prediction accuracy",
                    }
                )
            elif quality_score < self.performance_benchmarks["data_quality_good"]:
                recommendations["medium_priority"].append(
                    {
                        "category": "Data Quality",
                        "issue": f"Moderate data completeness ({quality_score:.1%})",
                        "recommendation": "Enhance data collection processes for better completeness",
                        "impact": "Medium - Improved data quality will enhance predictions",
                    }
                )

        # Model Performance Recommendations
        if (
            "model_performance" in analysis_results
            and "model_info" in analysis_results["model_performance"]
        ):
            accuracy = analysis_results["model_performance"]["model_info"]["accuracy"]

            if accuracy < self.performance_benchmarks["accuracy_poor"]:
                recommendations["high_priority"].append(
                    {
                        "category": "Model Performance",
                        "issue": f"Low model accuracy ({accuracy:.1%})",
                        "recommendation": "Retrain models with improved feature engineering and hyperparameter tuning",
                        "impact": "Critical - Low accuracy makes predictions unreliable",
                    }
                )
            elif accuracy < self.performance_benchmarks["accuracy_good"]:
                recommendations["medium_priority"].append(
                    {
                        "category": "Model Performance",
                        "issue": f"Moderate model accuracy ({accuracy:.1%})",
                        "recommendation": "Implement ensemble methods and advanced feature selection",
                        "impact": "Medium - Better accuracy will improve betting performance",
                    }
                )

        # Advanced Enhancement Recommendations
        recommendations["advanced_enhancements"].extend(
            [
                {
                    "category": "Deep Learning",
                    "recommendation": "Implement LSTM networks for sequence modeling of dog performance",
                    "impact": "High - Better temporal pattern recognition",
                    "complexity": "High",
                },
                {
                    "category": "Feature Engineering",
                    "recommendation": "Add momentum indicators and rolling performance metrics",
                    "impact": "Medium - Capture performance trends",
                    "complexity": "Medium",
                },
                {
                    "category": "Real-time Processing",
                    "recommendation": "Implement streaming prediction pipeline for live races",
                    "impact": "High - Enable live betting strategies",
                    "complexity": "High",
                },
                {
                    "category": "Automated Retraining",
                    "recommendation": "Implement continuous learning with model drift detection",
                    "impact": "Medium - Maintain model performance over time",
                    "complexity": "Medium",
                },
            ]
        )

        return recommendations

    def _assess_quality(self, score):
        """Assess data quality level"""
        if score >= self.performance_benchmarks["data_quality_excellent"]:
            return "Excellent"
        elif score >= self.performance_benchmarks["data_quality_good"]:
            return "Good"
        elif score >= self.performance_benchmarks["data_quality_poor"]:
            return "Fair"
        else:
            return "Poor"

    def _assess_model_performance(self, accuracy):
        """Assess model performance level"""
        if accuracy >= self.performance_benchmarks["accuracy_excellent"]:
            return "Excellent"
        elif accuracy >= self.performance_benchmarks["accuracy_good"]:
            return "Good"
        elif accuracy >= self.performance_benchmarks["accuracy_poor"]:
            return "Fair"
        else:
            return "Poor"

    def _assess_system_health(self, file_stats):
        """Assess overall system health"""
        health_score = 0
        total_checks = 0

        # Check if key directories have files
        key_dirs = ["upcoming_races", "predictions"]
        for dir_name in key_dirs:
            total_checks += 1
            if dir_name in file_stats and "total_files" in file_stats[dir_name]:
                if file_stats[dir_name]["total_files"] > 0:
                    health_score += 1

        # Check predictions directory has recent files
        predictions_path = Path("./predictions")
        if predictions_path.exists():
            recent_files = [
                f
                for f in predictions_path.glob("*.json")
                if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days < 7
            ]
            total_checks += 1
            if len(recent_files) > 0:
                health_score += 1

        health_percentage = health_score / total_checks if total_checks > 0 else 0

        if health_percentage >= 0.8:
            return "Healthy"
        elif health_percentage >= 0.6:
            return "Fair"
        else:
            return "Needs Attention"

    def _analyze_feature_importance(self, model_data):
        """Analyze feature importance from model"""
        try:
            model = model_data.get("model")
            feature_columns = model_data.get("feature_columns", [])

            if hasattr(model, "feature_importances_") and feature_columns:
                importances = model.feature_importances_
                feature_importance = list(zip(feature_columns, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)

                return {
                    "top_10_features": feature_importance[:10],
                    "total_features": len(feature_columns),
                    "importance_distribution": {
                        "high_importance": len([f for f in importances if f > 0.05]),
                        "medium_importance": len(
                            [f for f in importances if 0.01 < f <= 0.05]
                        ),
                        "low_importance": len([f for f in importances if f <= 0.01]),
                    },
                }
            else:
                return {"error": "Feature importance not available"}
        except Exception as e:
            return {"error": str(e)}

    def run_complete_analysis(self):
        """Run complete system analysis"""
        print("🚀 STARTING COMPREHENSIVE SYSTEM ANALYSIS")
        print("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "analysis_version": "1.0",
            "data_quality": self.analyze_data_quality(),
            "model_performance": self.analyze_model_performance(),
            "prediction_accuracy": self.analyze_prediction_accuracy(),
            "system_performance": self.analyze_system_performance(),
        }

        # Generate recommendations
        results["recommendations"] = self.generate_recommendations(results)

        # Save results
        output_file = (
            self.analysis_dir
            / f"system_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print summary
        self._print_analysis_summary(results)

        print(f"\n📄 Full analysis saved to: {output_file}")
        return results

    def _print_analysis_summary(self, results):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)

        # Data Quality Summary
        if results["data_quality"]:
            dq = results["data_quality"]
            print(f"\n📈 DATA QUALITY:")
            print(f"  • Total Races: {dq['basic_stats']['total_races']:,}")
            print(f"  • Unique Dogs: {dq['basic_stats']['unique_dogs']:,}")
            print(
                f"  • Overall Quality: {dq['data_completeness']['overall_quality_score']:.1%} ({dq['quality_assessment']})"
            )

        # Model Performance Summary
        if (
            results["model_performance"]
            and "model_info" in results["model_performance"]
        ):
            mp = results["model_performance"]["model_info"]
            print(f"\n🤖 MODEL PERFORMANCE:")
            print(f"  • Model: {mp['name']}")
            print(f"  • Accuracy: {mp['accuracy']:.1%}")
            print(f"  • Features: {mp['features_count']}")

        # System Health Summary
        if results["system_performance"]:
            sp = results["system_performance"]
            print(f"\n⚡ SYSTEM HEALTH:")
            print(f"  • Database Size: {sp['database_size_mb']:.1f} MB")
            print(f"  • System Status: {sp['system_health']}")

        # Recommendations Summary
        if results["recommendations"]:
            recs = results["recommendations"]
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"  • High Priority: {len(recs['high_priority'])} items")
            print(f"  • Medium Priority: {len(recs['medium_priority'])} items")
            print(
                f"  • Advanced Enhancements: {len(recs['advanced_enhancements'])} items"
            )


if __name__ == "__main__":
    analyzer = AdvancedSystemAnalyzer()
    results = analyzer.run_complete_analysis()
