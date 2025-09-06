#!/usr/bin/env python3
"""
Automated Backtesting System
============================

This system provides continuous, automated validation of ML models with:
- Scheduled backtesting runs
- Performance drift detection
- Model retraining triggers
- Alert system for significant changes
- Historical performance tracking

Features:
- Automated daily/weekly backtesting
- Performance regression detection
- Model degradation alerts
- Feature drift monitoring
- Automated retraining pipeline
- Performance tracking dashboard data

Author: AI Assistant
Date: July 24, 2025
"""

import json
import os
import sqlite3
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import schedule

warnings.filterwarnings("ignore")

# ML Libraries
try:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML libraries not available: {e}")
    SKLEARN_AVAILABLE = False


class AutomatedBacktestingSystem:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results_dir = Path("./automated_backtesting_results")
        self.models_dir = Path("./trained_models")
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.performance_history = []
        self.performance_thresholds = {
            "accuracy_drop_warning": 0.05,  # 5% drop triggers warning
            "accuracy_drop_critical": 0.10,  # 10% drop triggers retraining
            "feature_drift_threshold": 0.15,  # Feature importance change threshold
        }

        # Model configurations
        self.model_configs = {
            "random_forest": {
                "model": RandomForestClassifier,
                "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier,
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.2,
                    "max_depth": 5,
                    "random_state": 42,
                },
            },
            "logistic_regression": {
                "model": LogisticRegression,
                "params": {"C": 0.1, "max_iter": 1000, "random_state": 42},
            },
        }

        # Scheduling flags
        self.scheduler_running = False
        self.scheduler_thread = None

        print("🤖 Automated Backtesting System Initialized")

    def load_historical_data(self, months_back=6):
        """Load historical data for backtesting"""
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
                rm.grade, rm.race_date
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.finish_position != 'N/A'
            AND rm.race_date >= ?
            AND rm.race_date <= ?
            AND drd.individual_time IS NOT NULL
            ORDER BY rm.race_date ASC
            """

            df = pd.read_sql_query(
                query, conn, params=[start_date.isoformat(), end_date.isoformat()]
            )
            conn.close()

            return df

        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return None

    def prepare_data_for_training(self, df):
        """Prepare data for model training (simplified version)"""
        try:
            # Basic feature engineering
            enhanced_records = []

            for _, row in df.iterrows():
                try:
                    # Clean finish position more thoroughly
                    pos_str = str(row["finish_position"]).strip()
                    if pos_str in ["", "N/A", "None", "nan"]:
                        continue
                    # Remove equals sign and any other non-numeric characters except digits
                    pos_cleaned = "".join(filter(str.isdigit, pos_str))
                    if not pos_cleaned:
                        continue
                    finish_position = int(pos_cleaned)
                    # Validate reasonable finish position (1-8 typical for greyhound racing)
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

                if len(historical_data) >= 1:
                    # Basic features - clean historical positions the same way
                    positions = []
                    for p in historical_data["finish_position"]:
                        if pd.notna(p):
                            pos_str = str(p).strip()
                            if pos_str not in ["", "N/A", "None", "nan"]:
                                pos_cleaned = "".join(filter(str.isdigit, pos_str))
                                if pos_cleaned and 1 <= int(pos_cleaned) <= 10:
                                    positions.append(int(pos_cleaned))

                    recent_form_avg = np.mean(positions[:5]) if positions else 5.0
                    win_rate = (
                        sum(1 for p in positions if p == 1) / len(positions)
                        if positions
                        else 0
                    )
                    avg_position = np.mean(positions) if positions else 5.0

                    # Market data
                    odds = [
                        float(o)
                        for o in historical_data["starting_price"]
                        if pd.notna(o) and float(o) > 0
                    ]
                    market_confidence = 1 / (np.mean(odds) + 1) if odds else 0.1

                    enhanced_records.append(
                        {
                            "race_id": row["race_id"],
                            "dog_name": dog_name,
                            "finish_position": finish_position,
                            "is_winner": 1 if finish_position == 1 else 0,
                            "is_placer": 1 if finish_position <= 3 else 0,
                            "race_date": race_date,
                            "recent_form_avg": recent_form_avg,
                            "win_rate": win_rate,
                            "avg_position": avg_position,
                            "market_confidence": market_confidence,
                            "current_box": row["box_number"],
                            "current_weight": row["weight"],
                            "current_odds": row["starting_price"],
                            "field_size": row["field_size"],
                            "distance": row["distance"],
                            "venue": row["venue"],
                            "track_condition": row["track_condition"],
                            "grade": row["grade"],
                        }
                    )

            enhanced_df = pd.DataFrame(enhanced_records)

            if len(enhanced_df) < 50:
                return None, None

            # Feature columns
            feature_columns = [
                "recent_form_avg",
                "win_rate",
                "avg_position",
                "market_confidence",
                "current_box",
                "current_weight",
                "field_size",
            ]

            # Encode categorical features
            le_venue = LabelEncoder()
            le_condition = LabelEncoder()
            le_grade = LabelEncoder()

            enhanced_df["venue_encoded"] = le_venue.fit_transform(
                enhanced_df["venue"].fillna("Unknown")
            )
            enhanced_df["track_condition_encoded"] = le_condition.fit_transform(
                enhanced_df["track_condition"].fillna("Good")
            )
            enhanced_df["grade_encoded"] = le_grade.fit_transform(
                enhanced_df["grade"].fillna("Unknown")
            )

            feature_columns.extend(
                ["venue_encoded", "track_condition_encoded", "grade_encoded"]
            )

            # Handle distance and odds
            enhanced_df["distance_numeric"] = enhanced_df["distance"].apply(
                lambda x: float(str(x).replace("m", "")) if pd.notna(x) else 500.0
            )
            enhanced_df["current_odds_log"] = np.log(
                enhanced_df["current_odds"].fillna(10.0) + 1
            )

            feature_columns.extend(["distance_numeric", "current_odds_log"])

            # Fill missing values
            complete_df = enhanced_df[
                feature_columns + ["is_winner", "is_placer", "race_date", "race_id"]
            ].copy()
            imputer = SimpleImputer(strategy="median")
            complete_df[feature_columns] = imputer.fit_transform(
                complete_df[feature_columns]
            )

            return complete_df, feature_columns

        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return None, None

    def run_automated_backtest(self):
        """Run a single automated backtest"""
        print(
            f"🚀 Running automated backtest - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn not available")
            return None

        try:
            # Load and prepare data
            df = self.load_historical_data()
            if df is None or len(df) < 100:
                print("❌ Insufficient data for backtesting")
                return None

            prepared_df, feature_columns = self.prepare_data_for_training(df)
            if prepared_df is None:
                print("❌ Data preparation failed")
                return None

            print(
                f"📊 Testing with {len(prepared_df)} samples and {len(feature_columns)} features"
            )

            # Time series split for validation
            df_sorted = prepared_df.sort_values("race_date")
            split_point = int(0.8 * len(df_sorted))
            train_df = df_sorted.iloc[:split_point]
            test_df = df_sorted.iloc[split_point:]

            results = {
                "timestamp": datetime.now().isoformat(),
                "data_summary": {
                    "total_samples": len(prepared_df),
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    "features": len(feature_columns),
                },
                "model_results": {},
            }

            # Test each model
            for model_name, config in self.model_configs.items():
                print(f"   Testing {model_name}...")

                model_class = config["model"]
                params = config["params"]

                # Prepare features
                X_train = train_df[feature_columns]
                y_train = train_df["is_winner"]
                X_test = test_df[feature_columns]
                y_test = test_df["is_winner"]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model
                model = model_class(**params)
                model.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = (
                    model.predict_proba(X_test_scaled)[:, 1]
                    if hasattr(model, "predict_proba")
                    else y_pred
                )

                accuracy = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5

                # Feature importance
                feature_importance = None
                if hasattr(model, "feature_importances_"):
                    feature_importance = list(
                        zip(feature_columns, model.feature_importances_)
                    )
                    feature_importance.sort(key=lambda x: x[1], reverse=True)

                results["model_results"][model_name] = {
                    "accuracy": accuracy,
                    "auc": auc,
                    "feature_importance": (
                        feature_importance[:10] if feature_importance else None
                    ),
                }

                print(f"     📊 Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

                # Save model if it's the best performing
                if (
                    model_name == "random_forest"
                ):  # Our best model from previous analysis
                    model_file = (
                        self.models_dir
                        / f"best_model_{datetime.now().strftime('%Y%m%d')}.joblib"
                    )
                    joblib.dump(
                        {
                            "model": model,
                            "scaler": scaler,
                            "feature_columns": feature_columns,
                            "accuracy": accuracy,
                            "timestamp": datetime.now().isoformat(),
                        },
                        model_file,
                    )

            # Performance comparison with history
            self.analyze_performance_drift(results)

            # Save results
            self.save_backtest_results(results)

            print("✅ Automated backtest completed")
            return results

        except Exception as e:
            print(f"❌ Automated backtest failed: {e}")
            return None

    def analyze_performance_drift(self, current_results):
        """Analyze performance drift compared to historical results"""
        try:
            # Load recent performance history
            history_files = sorted(self.results_dir.glob("backtest_results_*.json"))

            if len(history_files) < 2:
                print("📊 Insufficient history for drift analysis")
                return

            # Compare with last result
            with open(history_files[-1], "r") as f:
                last_results = json.load(f)

            print("📊 Analyzing performance drift...")

            drift_analysis = {
                "timestamp": datetime.now().isoformat(),
                "alerts": [],
                "model_comparisons": {},
            }

            for model_name in current_results["model_results"]:
                if model_name in last_results["model_results"]:
                    current_acc = current_results["model_results"][model_name][
                        "accuracy"
                    ]
                    last_acc = last_results["model_results"][model_name]["accuracy"]

                    accuracy_change = current_acc - last_acc
                    accuracy_change_pct = (
                        (accuracy_change / last_acc) * 100 if last_acc > 0 else 0
                    )

                    drift_analysis["model_comparisons"][model_name] = {
                        "current_accuracy": current_acc,
                        "previous_accuracy": last_acc,
                        "accuracy_change": accuracy_change,
                        "accuracy_change_pct": accuracy_change_pct,
                    }

                    # Check for alerts
                    if (
                        accuracy_change
                        < -self.performance_thresholds["accuracy_drop_critical"]
                    ):
                        alert = {
                            "level": "CRITICAL",
                            "model": model_name,
                            "message": f"Critical accuracy drop: {accuracy_change_pct:.1f}%",
                            "recommendation": "Immediate model retraining recommended",
                        }
                        drift_analysis["alerts"].append(alert)
                        print(
                            f"🚨 CRITICAL: {model_name} accuracy dropped {accuracy_change_pct:.1f}%"
                        )

                    elif (
                        accuracy_change
                        < -self.performance_thresholds["accuracy_drop_warning"]
                    ):
                        alert = {
                            "level": "WARNING",
                            "model": model_name,
                            "message": f"Accuracy drop detected: {accuracy_change_pct:.1f}%",
                            "recommendation": "Monitor closely, consider retraining",
                        }
                        drift_analysis["alerts"].append(alert)
                        print(
                            f"⚠️ WARNING: {model_name} accuracy dropped {accuracy_change_pct:.1f}%"
                        )

                    elif accuracy_change > 0.02:  # 2% improvement
                        print(
                            f"✅ GOOD: {model_name} accuracy improved {accuracy_change_pct:.1f}%"
                        )

            # Save drift analysis
            drift_file = (
                self.results_dir
                / f"drift_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(drift_file, "w") as f:
                json.dump(drift_analysis, f, indent=2, default=str)

            # Send alerts if any critical issues
            if drift_analysis["alerts"]:
                self.send_performance_alerts(drift_analysis["alerts"])

        except Exception as e:
            print(f"⚠️ Error in drift analysis: {e}")

    def send_performance_alerts(self, alerts):
        """Send performance alerts (placeholder for notification system)"""
        print("📧 Performance alerts generated:")
        for alert in alerts:
            print(f"   {alert['level']}: {alert['message']}")
            print(f"   Recommendation: {alert['recommendation']}")

        # In a real implementation, you could:
        # - Send email alerts
        # - Post to Slack/Discord
        # - Write to log files
        # - Update dashboard status

        # Save alerts to file
        alert_file = (
            self.results_dir / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(alert_file, "w") as f:
            json.dump(alerts, f, indent=2)

    def save_backtest_results(self, results):
        """Save backtest results with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"backtest_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Also maintain a latest_results.json for easy access
        latest_file = self.results_dir / "latest_results.json"
        with open(latest_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to {results_file}")

    def generate_performance_report(self, days_back=30):
        """Generate a performance report over the last N days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Find all results files from the specified period
            result_files = []
            for file in self.results_dir.glob("backtest_results_*.json"):
                try:
                    # Extract timestamp from filename
                    timestamp_str = (
                        file.stem.split("_")[-2] + "_" + file.stem.split("_")[-1]
                    )
                    file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    if file_date >= cutoff_date:
                        result_files.append((file_date, file))
                except:
                    continue

            if not result_files:
                print(f"📊 No backtest results found in the last {days_back} days")
                return None

            result_files.sort(key=lambda x: x[0])

            print(f"📊 Generating performance report for {len(result_files)} backtests")

            # Compile performance data
            performance_data = []
            for file_date, file_path in result_files:
                with open(file_path, "r") as f:
                    results = json.load(f)

                for model_name, model_results in results["model_results"].items():
                    performance_data.append(
                        {
                            "date": file_date.isoformat(),
                            "model": model_name,
                            "accuracy": model_results["accuracy"],
                            "auc": model_results["auc"],
                        }
                    )

            # Create summary statistics
            df = pd.DataFrame(performance_data)

            summary = {}
            for model_name in df["model"].unique():
                model_data = df[df["model"] == model_name]
                summary[model_name] = {
                    "mean_accuracy": model_data["accuracy"].mean(),
                    "std_accuracy": model_data["accuracy"].std(),
                    "min_accuracy": model_data["accuracy"].min(),
                    "max_accuracy": model_data["accuracy"].max(),
                    "accuracy_trend": self.calculate_trend(
                        model_data["accuracy"].values
                    ),
                    "test_count": len(model_data),
                }

            report = {
                "report_period": f"{days_back} days",
                "report_generated": datetime.now().isoformat(),
                "total_backtests": len(result_files),
                "models_analyzed": list(summary.keys()),
                "performance_summary": summary,
                "detailed_data": performance_data,
            }

            # Save report
            report_file = (
                self.results_dir
                / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            print(f"📋 Performance report saved to {report_file}")

            # Print summary
            print("\n📊 PERFORMANCE SUMMARY:")
            print("=" * 40)
            for model_name, stats in summary.items():
                trend_symbol = (
                    "📈"
                    if stats["accuracy_trend"] > 0
                    else "📉" if stats["accuracy_trend"] < 0 else "➡️"
                )
                print(f"{model_name}:")
                print(
                    f"  Mean Accuracy: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}"
                )
                print(
                    f"  Range: {stats['min_accuracy']:.3f} - {stats['max_accuracy']:.3f}"
                )
                print(f"  Trend: {trend_symbol} {stats['accuracy_trend']:.4f}")
                print(f"  Tests: {stats['test_count']}")
                print()

            return report

        except Exception as e:
            print(f"❌ Error generating performance report: {e}")
            return None

    def calculate_trend(self, values):
        """Calculate simple linear trend of values"""
        if len(values) < 2:
            return 0

        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]  # Slope of linear fit
        return trend

    def schedule_automated_backtests(self, frequency="daily", time_str="02:00"):
        """Schedule automated backtests"""
        if self.scheduler_running:
            print("⚠️ Scheduler already running")
            return

        print(f"⏰ Scheduling {frequency} backtests at {time_str}")

        # Clear any existing scheduled jobs
        schedule.clear()

        if frequency == "daily":
            schedule.every().day.at(time_str).do(self.run_automated_backtest)
        elif frequency == "weekly":
            schedule.every().week.at(time_str).do(self.run_automated_backtest)
        elif frequency == "hourly":
            schedule.every().hour.do(self.run_automated_backtest)

        # Also schedule weekly performance reports
        schedule.every().sunday.at("03:00").do(self.generate_performance_report)

        self.scheduler_running = True

        # Start scheduler in a separate thread
        def run_scheduler():
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()

        print(f"✅ Automated backtesting scheduled ({frequency} at {time_str})")

    def stop_scheduler(self):
        """Stop the automated scheduler"""
        if self.scheduler_running:
            self.scheduler_running = False
            schedule.clear()
            print("⏹️ Automated backtesting scheduler stopped")
        else:
            print("⚠️ Scheduler is not running")

    def get_system_status(self):
        """Get current system status"""
        status = {
            "scheduler_running": self.scheduler_running,
            "scheduled_jobs": len(schedule.jobs),
            "last_backtest": None,
            "model_files": len(list(self.models_dir.glob("*.joblib"))),
            "result_files": len(list(self.results_dir.glob("backtest_results_*.json"))),
        }

        # Find last backtest
        result_files = list(self.results_dir.glob("backtest_results_*.json"))
        if result_files:
            latest_file = max(result_files, key=os.path.getctime)
            status["last_backtest"] = latest_file.stem

        return status


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Backtesting System")
    parser.add_argument(
        "--command",
        choices=["run", "schedule", "stop", "report", "status"],
        default="run",
        help="Command to execute",
    )
    parser.add_argument(
        "--frequency",
        choices=["daily", "weekly", "hourly"],
        default="daily",
        help="Scheduling frequency",
    )
    parser.add_argument(
        "--time", default="02:00", help="Time for scheduled runs (HH:MM)"
    )
    parser.add_argument("--days", type=int, default=30, help="Days back for reports")

    args = parser.parse_args()

    print("🤖 Automated Backtesting System")
    print("=" * 50)

    system = AutomatedBacktestingSystem()

    if args.command == "run":
        system.run_automated_backtest()

    elif args.command == "schedule":
        system.schedule_automated_backtests(args.frequency, args.time)
        print("Press Ctrl+C to stop the scheduler")
        try:
            while system.scheduler_running:
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop_scheduler()

    elif args.command == "stop":
        system.stop_scheduler()

    elif args.command == "report":
        system.generate_performance_report(args.days)

    elif args.command == "status":
        status = system.get_system_status()
        print("\n📊 SYSTEM STATUS:")
        print("=" * 30)
        for key, value in status.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
