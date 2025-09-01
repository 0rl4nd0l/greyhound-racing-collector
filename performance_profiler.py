#!/usr/bin/env python3
"""
Performance Profiling & Concurrency Tuning Script
================================================

This script implements comprehensive performance profiling using cProfile and py-spy
to identify backend bottlenecks and measure latency improvements.

Author: AI Assistant
Date: August 2, 2025
"""

import cProfile
import json
import os
import pstats
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import psutil


class PerformanceProfiler:
    def __init__(self):
        self.baseline_metrics = {}
        self.tuned_metrics = {}
        self.profile_results = {}

    @contextmanager
    def profile_context(self, profile_name):
        """Context manager for profiling code blocks"""
        profiler = cProfile.Profile()
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        profiler.enable()
        try:
            yield profiler
        finally:
            profiler.disable()
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Save profile data
            profile_filename = f"profiles/{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
            os.makedirs("profiles", exist_ok=True)
            profiler.dump_stats(profile_filename)

            # Generate stats
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")

            # Store metrics
            self.profile_results[profile_name] = {
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "profile_file": profile_filename,
                "timestamp": datetime.now().isoformat(),
            }

            print(f"✅ Profile '{profile_name}' completed:")
            print(f"   ⏱️  Execution time: {end_time - start_time:.4f}s")
            print(f"   💾 Memory usage: {end_memory - start_memory:.2f}MB")
            print(f"   📁 Profile saved: {profile_filename}")

    def profile_database_operations(self):
        """Profile database operations for bottlenecks"""
        with self.profile_context("database_operations"):
            conn = sqlite3.connect("greyhound_racing_data.db")
            cursor = conn.cursor()

            # Test various database operations
            operations = [
                "SELECT COUNT(*) FROM race_metadata",
                "SELECT COUNT(*) FROM dogs",
                "SELECT COUNT(*) FROM dog_race_data",
                """SELECT rm.venue, COUNT(*) as race_count 
                   FROM race_metadata rm 
                   GROUP BY rm.venue 
                   ORDER BY race_count DESC 
                   LIMIT 10""",
                """SELECT d.dog_name, d.total_wins, d.total_races 
                   FROM dogs d 
                   WHERE d.total_races > 10 
                   ORDER BY d.total_wins DESC 
                   LIMIT 20""",
                """SELECT drd.dog_name, rm.venue, drd.finish_position 
                   FROM dog_race_data drd 
                   JOIN race_metadata rm ON drd.race_id = rm.race_id 
                   WHERE drd.finish_position <= 3 
                   LIMIT 100""",
            ]

            for query in operations:
                start_time = time.time()
                cursor.execute(query)
                results = cursor.fetchall()
                end_time = time.time()
                print(
                    f"   📊 Query executed in {end_time - start_time:.4f}s, returned {len(results)} rows"
                )

            conn.close()

    def profile_prediction_pipeline(self):
        """Profile the prediction pipeline for bottlenecks"""
        with self.profile_context("prediction_pipeline"):
            try:
                # Import and test prediction components
                from prediction_pipeline_v3 import PredictionPipelineV3

                pipeline = PredictionPipelineV3()

                # Test with a sample race file if available
                upcoming_dir = "./upcoming_races"
                if os.path.exists(upcoming_dir):
                    csv_files = [
                        f for f in os.listdir(upcoming_dir) if f.endswith(".csv")
                    ]
                    if csv_files:
                        test_file = os.path.join(upcoming_dir, csv_files[0])
                        print(f"   🎯 Testing prediction with: {csv_files[0]}")
                        result = pipeline.predict_race_file(
                            test_file, enhancement_level="basic"
                        )
                        print(
                            f"   ✅ Prediction result: {result.get('success', False)}"
                        )
                    else:
                        print("   ⚠️  No CSV files found for prediction testing")
                else:
                    print("   ⚠️  No upcoming races directory found")

            except ImportError as e:
                print(f"   ⚠️  Prediction pipeline not available: {e}")

    def profile_flask_endpoints(self):
        """Profile Flask endpoint performance"""
        with self.profile_context("flask_endpoints"):
            try:
                from app import app

                with app.test_client() as client:
                    endpoints_to_test = [
                        "/api/health",
                        "/api/races?page=1&per_page=5",
                        "/api/dogs/search?q=test&limit=5",
                        "/api/upcoming_races_csv?page=1&per_page=5",
                    ]

                    for endpoint in endpoints_to_test:
                        start_time = time.time()
                        response = client.get(endpoint)
                        end_time = time.time()
                        print(
                            f"   🌐 {endpoint}: {response.status_code} in {end_time - start_time:.4f}s"
                        )

            except Exception as e:
                print(f"   ⚠️  Flask endpoint profiling failed: {e}")

    def measure_baseline_latency(self):
        """Measure baseline latency before optimizations"""
        print("📏 Measuring baseline latency...")

        baseline_tests = {
            "database_connection": self._measure_db_connection_time,
            "simple_query": self._measure_simple_query_time,
            "complex_query": self._measure_complex_query_time,
            "memory_usage": self._measure_memory_usage,
        }

        for test_name, test_func in baseline_tests.items():
            try:
                result = test_func()
                self.baseline_metrics[test_name] = result
                print(f"   ✅ {test_name}: {result}")
            except Exception as e:
                print(f"   ❌ {test_name} failed: {e}")
                self.baseline_metrics[test_name] = None

    def measure_tuned_latency(self):
        """Measure latency after optimizations"""
        print("📈 Measuring tuned latency...")

        tuned_tests = {
            "database_connection": self._measure_db_connection_time,
            "simple_query": self._measure_simple_query_time,
            "complex_query": self._measure_complex_query_time,
            "memory_usage": self._measure_memory_usage,
        }

        for test_name, test_func in tuned_tests.items():
            try:
                result = test_func()
                self.tuned_metrics[test_name] = result
                print(f"   ✅ {test_name}: {result}")
            except Exception as e:
                print(f"   ❌ {test_name} failed: {e}")
                self.tuned_metrics[test_name] = None

    def _measure_db_connection_time(self):
        """Measure database connection establishment time"""
        start_time = time.time()
        conn = sqlite3.connect("greyhound_racing_data.db")
        conn.close()
        end_time = time.time()
        return f"{(end_time - start_time) * 1000:.2f}ms"

    def _measure_simple_query_time(self):
        """Measure simple query execution time"""
        conn = sqlite3.connect("greyhound_racing_data.db")
        cursor = conn.cursor()

        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        result = cursor.fetchone()
        end_time = time.time()

        conn.close()
        return f"{(end_time - start_time) * 1000:.2f}ms"

    def _measure_complex_query_time(self):
        """Measure complex query execution time"""
        conn = sqlite3.connect("greyhound_racing_data.db")
        cursor = conn.cursor()

        complex_query = """
        SELECT 
            rm.venue,
            COUNT(rm.race_id) as total_races,
            AVG(CAST(drd.finish_position AS FLOAT)) as avg_position,
            COUNT(CASE WHEN drd.finish_position = 1 THEN 1 END) as wins
        FROM race_metadata rm
        JOIN dog_race_data drd ON rm.race_id = drd.race_id
        WHERE rm.race_date >= date('now', '-30 days')
        GROUP BY rm.venue
        ORDER BY total_races DESC
        LIMIT 10
        """

        start_time = time.time()
        cursor.execute(complex_query)
        results = cursor.fetchall()
        end_time = time.time()

        conn.close()
        return f"{(end_time - start_time) * 1000:.2f}ms ({len(results)} rows)"

    def _measure_memory_usage(self):
        """Measure current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.2f}MB"

    def run_py_spy_analysis(self, duration=30):
        """Run py-spy analysis on the running Flask application"""
        print(f"🔍 Running py-spy analysis for {duration} seconds...")

        try:
            # Find Flask process
            flask_pid = None
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if proc.info["cmdline"] and any(
                        "app.py" in cmd or "gunicorn" in cmd
                        for cmd in proc.info["cmdline"]
                    ):
                        flask_pid = proc.info["pid"]
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if flask_pid:
                output_file = f"profiles/py_spy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
                cmd = [
                    "py-spy",
                    "record",
                    "--pid",
                    str(flask_pid),
                    "--duration",
                    str(duration),
                    "--output",
                    output_file,
                    "--format",
                    "svg",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ py-spy report saved: {output_file}")
                else:
                    print(f"   ❌ py-spy failed: {result.stderr}")
            else:
                print("   ⚠️  No Flask process found for py-spy analysis")

        except Exception as e:
            print(f"   ❌ py-spy analysis failed: {e}")

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report_file = (
            f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "baseline_metrics": self.baseline_metrics,
            "tuned_metrics": self.tuned_metrics,
            "profile_results": self.profile_results,
            "improvements": self._calculate_improvements(),
        }

        os.makedirs("reports", exist_ok=True)
        report_path = os.path.join("reports", report_file)

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"📊 Performance report saved: {report_path}")
        self._print_summary_report(report_data)

    def _calculate_improvements(self):
        """Calculate performance improvements"""
        improvements = {}

        for metric in self.baseline_metrics:
            if (
                metric in self.tuned_metrics
                and self.baseline_metrics[metric]
                and self.tuned_metrics[metric]
            ):
                try:
                    # Extract numeric values from metric strings
                    baseline_val = float(self.baseline_metrics[metric].split()[0])
                    tuned_val = float(self.tuned_metrics[metric].split()[0])

                    if baseline_val > 0:
                        improvement_pct = (
                            (baseline_val - tuned_val) / baseline_val
                        ) * 100
                        improvements[metric] = f"{improvement_pct:.2f}%"
                    else:
                        improvements[metric] = "N/A"
                except (ValueError, IndexError):
                    improvements[metric] = "N/A"
            else:
                improvements[metric] = "N/A"

        return improvements

    def _print_summary_report(self, report_data):
        """Print a summary of the performance report"""
        print("\n" + "=" * 60)
        print("🎯 PERFORMANCE TUNING SUMMARY")
        print("=" * 60)

        print("\n📏 BASELINE METRICS:")
        for metric, value in report_data["baseline_metrics"].items():
            print(f"   {metric}: {value}")

        print("\n📈 TUNED METRICS:")
        for metric, value in report_data["tuned_metrics"].items():
            print(f"   {metric}: {value}")

        print("\n🚀 IMPROVEMENTS:")
        for metric, improvement in report_data["improvements"].items():
            print(f"   {metric}: {improvement}")

        print("\n📊 PROFILE RESULTS:")
        for profile_name, results in report_data["profile_results"].items():
            print(f"   {profile_name}:")
            print(f"     ⏱️  Execution time: {results['execution_time']:.4f}s")
            print(f"     💾 Memory usage: {results['memory_usage']:.2f}MB")
            print(f"     📁 Profile file: {results['profile_file']}")


def main():
    """Main execution function"""
    profiler = PerformanceProfiler()

    print("🚀 Starting Performance Profiling & Concurrency Tuning")
    print("=" * 60)

    # Step 1: Measure baseline performance
    profiler.measure_baseline_latency()

    # Step 2: Profile different components
    profiler.profile_database_operations()
    profiler.profile_prediction_pipeline()
    profiler.profile_flask_endpoints()

    # Step 3: Run py-spy analysis (optional, requires running Flask app)
    run_py_spy = (
        input("\n🔍 Run py-spy analysis? (requires running Flask app) [y/N]: ").lower()
        == "y"
    )
    if run_py_spy:
        profiler.run_py_spy_analysis(duration=30)

    # Step 4: Measure performance after any optimizations
    print("\n📈 Now implement database connection pooling and lazy loading...")
    input(
        "Press Enter after implementing optimizations to measure tuned performance..."
    )

    profiler.measure_tuned_latency()

    # Step 5: Generate comprehensive report
    profiler.generate_performance_report()

    print("\n✅ Performance profiling completed!")
    print("📁 Check the 'profiles' and 'reports' directories for detailed results.")


if __name__ == "__main__":
    main()
