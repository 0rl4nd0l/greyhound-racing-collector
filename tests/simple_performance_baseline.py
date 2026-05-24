#!/usr/bin/env python3

"""
Simple Performance Baseline Capture

This script captures basic performance metrics for the dashboard:
- First-byte timing
- DOMContentLoaded simulation
- Full load simulation
- Network requests count
- Total transferred data
- Time-to-Interactive estimation

Stores results in perf_reports/baseline/ for comparison.
"""

import json
import os
import subprocess
import time
from datetime import datetime

import requests

DASHBOARD_URL = "http://localhost:5002"
REPORTS_DIR = "./perf_reports/baseline/"
NUM_TESTS = 5  # Run multiple tests for average


class SimplePerformanceProfiler:
    def __init__(self):
        self.metrics = {}
        self.timestamp = datetime.now().isoformat().replace(":", "-")

    def ensure_reports_dir(self):
        """Ensure reports directory exists"""
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def test_basic_response(self):
        """Test basic HTTP response metrics using curl"""
        print("🚀 Testing basic HTTP response metrics...")

        curl_command = [
            "curl",
            "-w",
            "@-",  # Read format from stdin
            "-o",
            "/dev/null",
            "-s",
            DASHBOARD_URL,
        ]

        # Curl format string for detailed timing
        format_string = """
{
    "time_namelookup": %{time_namelookup},
    "time_connect": %{time_connect},
    "time_appconnect": %{time_appconnect},
    "time_pretransfer": %{time_pretransfer},
    "time_redirect": %{time_redirect},
    "time_starttransfer": %{time_starttransfer},
    "time_total": %{time_total},
    "speed_download": %{speed_download},
    "speed_upload": %{speed_upload},
    "size_download": %{size_download},
    "size_upload": %{size_upload},
    "size_header": %{size_header},
    "size_request": %{size_request},
    "num_connects": %{num_connects},
    "num_redirects": %{num_redirects},
    "http_code": %{http_code}
}
"""

        results = []
        for i in range(NUM_TESTS):
            print(f"  📊 Running test {i+1}/{NUM_TESTS}...")

            process = subprocess.Popen(
                curl_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = process.communicate(input=format_string)

            if process.returncode == 0:
                try:
                    result = json.loads(stdout.strip())
                    results.append(result)
                except json.JSONDecodeError as e:
                    print(f"  ❌ Failed to parse curl output: {e}")
            else:
                print(f"  ❌ Curl failed: {stderr}")

        return results

    def test_api_endpoints(self):
        """Test key API endpoints for response times"""
        print("🌐 Testing API endpoint performance...")

        endpoints = [
            "/",
            "/api/stats",
            "/api/system_status",
            "/api/file_stats",
            "/api/predict_stream",
            "/api/upcoming_races_csv?page=1&per_page=10",
        ]

        endpoint_results = {}

        for endpoint in endpoints:
            url = f"{DASHBOARD_URL}{endpoint}"
            times = []

            print(f"  📡 Testing {endpoint}...")

            for i in range(3):  # 3 tests per endpoint
                try:
                    start_time = time.time()
                    response = requests.get(url, timeout=10)
                    end_time = time.time()

                    duration = (end_time - start_time) * 1000  # Convert to ms
                    times.append(
                        {
                            "duration_ms": duration,
                            "status_code": response.status_code,
                            "content_length": len(response.content),
                            "headers": dict(response.headers),
                        }
                    )

                except requests.RequestException as e:
                    print(f"    ❌ Request failed: {e}")
                    times.append(
                        {"duration_ms": None, "status_code": None, "error": str(e)}
                    )

            endpoint_results[endpoint] = times

        return endpoint_results

    def analyze_page_resources(self):
        """Analyze page resources by parsing HTML"""
        print("📄 Analyzing page resources...")

        try:
            response = requests.get(DASHBOARD_URL, timeout=10)
            html_content = response.text

            # Count different resource types
            resources = {
                "css_files": html_content.count(".css"),
                "js_files": html_content.count(".js"),
                "img_tags": html_content.count("<img"),
                "link_tags": html_content.count("<link"),
                "script_tags": html_content.count("<script"),
                "api_calls": html_content.count("/api/"),
                "total_html_size": len(html_content),
                "estimated_dom_nodes": html_content.count("<")
                + html_content.count("</"),
            }

            return resources

        except requests.RequestException as e:
            print(f"  ❌ Failed to analyze resources: {e}")
            return {}

    def estimate_performance_metrics(
        self, curl_results, api_results, resource_analysis
    ):
        """Estimate key performance metrics from collected data"""
        print("📊 Calculating performance metrics...")

        if not curl_results:
            print("  ❌ No curl results to analyze")
            return {}

        # Calculate averages from curl results
        avg_metrics = {}
        for key in curl_results[0].keys():
            if isinstance(curl_results[0][key], (int, float)):
                values = [
                    r[key] for r in curl_results if isinstance(r.get(key), (int, float))
                ]
                if values:
                    avg_metrics[key] = sum(values) / len(values)

        # Convert to milliseconds and calculate derived metrics
        metrics = {
            # Basic timing (in ms)
            "dns_lookup_time": avg_metrics.get("time_namelookup", 0) * 1000,
            "tcp_connection_time": (
                avg_metrics.get("time_connect", 0)
                - avg_metrics.get("time_namelookup", 0)
            )
            * 1000,
            "ssl_handshake_time": (
                avg_metrics.get("time_appconnect", 0)
                - avg_metrics.get("time_connect", 0)
            )
            * 1000,
            "first_byte_time": avg_metrics.get("time_starttransfer", 0) * 1000,
            "total_time": avg_metrics.get("time_total", 0) * 1000,
            # Network metrics
            "download_speed": avg_metrics.get("speed_download", 0),
            "total_size": avg_metrics.get("size_download", 0),
            "header_size": avg_metrics.get("size_header", 0),
            "num_redirects": avg_metrics.get("num_redirects", 0),
            "http_status": avg_metrics.get("http_code", 0),
            # Estimated metrics (simplified calculations)
            "estimated_dom_content_loaded": avg_metrics.get("time_starttransfer", 0)
            * 1000
            + 500,  # TTFB + 500ms
            "estimated_time_to_interactive": avg_metrics.get("time_total", 0) * 1000
            + 1000,  # Total + 1s
            # Resource analysis
            "resource_analysis": resource_analysis,
            # API performance
            "api_performance": self.calculate_api_averages(api_results),
        }

        return metrics

    def calculate_api_averages(self, api_results):
        """Calculate average response times for API endpoints"""
        api_summary = {}

        for endpoint, results in api_results.items():
            durations = [
                r["duration_ms"] for r in results if r.get("duration_ms") is not None
            ]
            if durations:
                api_summary[endpoint] = {
                    "avg_response_time": sum(durations) / len(durations),
                    "min_response_time": min(durations),
                    "max_response_time": max(durations),
                    "successful_requests": len(durations),
                    "total_requests": len(results),
                }

        return api_summary

    def generate_lighthouse_style_report(self, metrics):
        """Generate a Lighthouse-style performance report"""
        print("🏆 Generating Lighthouse-style report...")

        def score_metric(value, good_threshold, poor_threshold):
            """Score a metric on a 0-100 scale"""
            if value <= good_threshold:
                return 100
            elif value >= poor_threshold:
                return 0
            else:
                return int(
                    100 * (poor_threshold - value) / (poor_threshold - good_threshold)
                )

        # Score the metrics
        scored_metrics = {
            "first-contentful-paint": {
                "value": metrics.get("first_byte_time", 0),
                "score": score_metric(metrics.get("first_byte_time", 0), 1000, 2000),
                "unit": "ms",
                "description": "Time to First Byte (approximating FCP)",
            },
            "time-to-interactive": {
                "value": metrics.get("estimated_time_to_interactive", 0),
                "score": score_metric(
                    metrics.get("estimated_time_to_interactive", 0), 3000, 6000
                ),
                "unit": "ms",
                "description": "Estimated Time to Interactive",
            },
            "first-byte": {
                "value": metrics.get("first_byte_time", 0),
                "score": score_metric(metrics.get("first_byte_time", 0), 200, 500),
                "unit": "ms",
                "description": "Server response time",
            },
            "total-load-time": {
                "value": metrics.get("total_time", 0),
                "score": score_metric(metrics.get("total_time", 0), 2000, 4000),
                "unit": "ms",
                "description": "Complete page load time",
            },
        }

        # Calculate overall performance score
        scores = [m["score"] for m in scored_metrics.values()]
        overall_score = sum(scores) / len(scores) if scores else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "url": DASHBOARD_URL,
            "overall_score": int(overall_score),
            "metrics": scored_metrics,
            "network": {
                "total_size": metrics.get("total_size", 0),
                "download_speed": metrics.get("download_speed", 0),
                "estimated_requests": metrics.get("resource_analysis", {}).get(
                    "css_files", 0
                )
                + metrics.get("resource_analysis", {}).get("js_files", 0)
                + metrics.get("resource_analysis", {}).get("img_tags", 0)
                + 10,  # Base requests
            },
            "raw_metrics": metrics,
        }

    def save_reports(self, lighthouse_report, raw_metrics):
        """Save performance reports to files"""
        print("💾 Saving performance reports...")

        try:
            # Save Lighthouse-style report
            lighthouse_file = os.path.join(
                REPORTS_DIR, f"baseline-lighthouse-{self.timestamp}.json"
            )
            with open(lighthouse_file, "w") as f:
                json.dump(lighthouse_report, f, indent=2)

            # Save raw metrics
            metrics_file = os.path.join(
                REPORTS_DIR, f"baseline-metrics-{self.timestamp}.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(raw_metrics, f, indent=2)

            # Generate markdown summary
            summary = self.generate_summary_report(lighthouse_report)
            summary_file = os.path.join(
                REPORTS_DIR, f"baseline-summary-{self.timestamp}.md"
            )
            with open(summary_file, "w") as f:
                f.write(summary)

            print(f"  ✅ Reports saved to {REPORTS_DIR}")
            return True

        except Exception as e:
            print(f"  ❌ Error saving reports: {e}")
            return False

    def generate_summary_report(self, lighthouse_report):
        """Generate a human-readable summary report"""
        metrics = lighthouse_report["metrics"]
        network = lighthouse_report["network"]

        summary = f"""# Performance Baseline Report

**Generated:** {lighthouse_report['timestamp']}
**URL:** {lighthouse_report['url']}
**Overall Performance Score:** {lighthouse_report['overall_score']}/100

## Core Performance Metrics

| Metric | Value | Score | Status |
|--------|-------|-------|--------|
| First Byte | {metrics['first-byte']['value']:.2f}ms | {metrics['first-byte']['score']}/100 | {'✅ Good' if metrics['first-byte']['score'] >= 75 else '⚠️ Needs Improvement' if metrics['first-byte']['score'] >= 50 else '❌ Poor'} |
| Time to Interactive | {metrics['time-to-interactive']['value']:.2f}ms | {metrics['time-to-interactive']['score']}/100 | {'✅ Good' if metrics['time-to-interactive']['score'] >= 75 else '⚠️ Needs Improvement' if metrics['time-to-interactive']['score'] >= 50 else '❌ Poor'} |
| Total Load Time | {metrics['total-load-time']['value']:.2f}ms | {metrics['total-load-time']['score']}/100 | {'✅ Good' if metrics['total-load-time']['score'] >= 75 else '⚠️ Needs Improvement' if metrics['total-load-time']['score'] >= 50 else '❌ Poor'} |

## Network Performance

- **Total Size Downloaded:** {network['total_size']} bytes ({network['total_size']/1024:.2f} KB)
- **Download Speed:** {network['download_speed']:.0f} bytes/sec ({network['download_speed']/1024:.0f} KB/s)
- **Estimated Requests:** {network['estimated_requests']}

## API Performance

"""

        # Add API performance details
        api_perf = lighthouse_report["raw_metrics"].get("api_performance", {})
        if api_perf:
            summary += "| Endpoint | Avg Response Time | Status |\n|----------|------------------|--------|\n"
            for endpoint, stats in api_perf.items():
                status = (
                    "✅ Good"
                    if stats["avg_response_time"] < 200
                    else (
                        "⚠️ Slow" if stats["avg_response_time"] < 500 else "❌ Very Slow"
                    )
                )
                summary += (
                    f"| {endpoint} | {stats['avg_response_time']:.2f}ms | {status} |\n"
                )

        summary += """

## Recommendations

"""

        # Generate recommendations
        recommendations = []
        if metrics["first-byte"]["score"] < 75:
            recommendations.append(
                "- **Improve Server Response Time:** Consider caching, database optimization, or CDN"
            )
        if metrics["time-to-interactive"]["score"] < 75:
            recommendations.append(
                "- **Reduce JavaScript Execution Time:** Optimize or defer non-critical scripts"
            )
        if network["total_size"] > 1024 * 1024:  # > 1MB
            recommendations.append(
                "- **Reduce Page Size:** Compress images, minify CSS/JS, remove unused resources"
            )
        if network["estimated_requests"] > 50:
            recommendations.append(
                "- **Reduce HTTP Requests:** Combine files, use sprites, implement lazy loading"
            )

        if recommendations:
            summary += "\n".join(recommendations)
        else:
            summary += "✅ Performance looks good! Continue monitoring for regressions."

        return summary

    def display_results(self, lighthouse_report):
        """Display results in console"""
        print("\n📊 Performance Baseline Results:")
        print("=" * 40)

        metrics = lighthouse_report["metrics"]
        network = lighthouse_report["network"]

        print(f"🏆 Overall Performance Score: {lighthouse_report['overall_score']}/100")
        print(f"🚀 First Byte: {metrics['first-byte']['value']:.2f}ms")
        print(
            f"🎯 Time to Interactive: {metrics['time-to-interactive']['value']:.2f}ms"
        )
        print(f"✅ Total Load Time: {metrics['total-load-time']['value']:.2f}ms")
        print(f"📦 Total Size: {network['total_size']/1024:.2f} KB")
        print(f"🌐 Download Speed: {network['download_speed']/1024:.0f} KB/s")
        print(f"📊 Estimated Requests: {network['estimated_requests']}")

        # Display API performance
        api_perf = lighthouse_report["raw_metrics"].get("api_performance", {})
        if api_perf:
            print("\n📡 API Performance:")
            for endpoint, stats in api_perf.items():
                print(f"  {endpoint}: {stats['avg_response_time']:.2f}ms avg")

    def run_profiling(self):
        """Run complete performance profiling"""
        print("🔍 Starting Simple Performance Baseline Capture...")
        print(f"📍 Target URL: {DASHBOARD_URL}")

        self.ensure_reports_dir()

        # Run all tests
        curl_results = self.test_basic_response()
        api_results = self.test_api_endpoints()
        resource_analysis = self.analyze_page_resources()

        # Calculate metrics
        metrics = self.estimate_performance_metrics(
            curl_results, api_results, resource_analysis
        )

        # Generate reports
        lighthouse_report = self.generate_lighthouse_style_report(metrics)

        # Save and display results
        self.save_reports(lighthouse_report, metrics)
        self.display_results(lighthouse_report)

        print("\n🎯 Performance baseline capture completed!")
        print(f"📁 Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    profiler = SimplePerformanceProfiler()
    profiler.run_profiling()
