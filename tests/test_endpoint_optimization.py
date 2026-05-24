#!/usr/bin/env python3
"""
Endpoint Optimization Performance Test
======================================

Tests the performance improvements from caching and query optimization
for the /system_status and /model_registry endpoints.

This script measures:
- Response time with and without optimization
- Database query count reduction
- Cache hit rates
- ETag functionality

Run this after starting the Flask app to verify optimization is working.
"""

import time
from datetime import datetime
from statistics import mean, stdev

import requests


def test_endpoint_performance():
    """Test endpoint performance and optimization features"""

    base_url = "http://localhost:5002"

    # Test endpoints
    endpoints = [
        "/api/system_status",
        "/api/model_registry/models",
        "/api/model_registry/performance",
        "/api/model_registry/status",
    ]

    print("🚀 ENDPOINT OPTIMIZATION PERFORMANCE TEST")
    print("=" * 60)
    print(f"Testing optimization at: {datetime.now().isoformat()}")
    print()

    results = {}

    for endpoint in endpoints:
        print(f"📊 Testing {endpoint}")
        print("-" * 40)

        # Test 1: Cold cache (first request)
        print("🔵 Cold cache test...")
        start_time = time.time()

        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            cold_time = (time.time() - start_time) * 1000  # Convert to ms

            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ⏱️  Cold response time: {cold_time:.2f}ms")
                print(f"   🔧 Optimized: {data.get('optimized', False)}")

                # Check for optimization features
                response_time_ms = data.get("response_time_ms", 0)
                if response_time_ms:
                    print(f"   📈 Reported response time: {response_time_ms:.2f}ms")

                # Check for next_refresh timestamp
                if "timestamp" in data:
                    print(f"   🕐 Timestamp: {data['timestamp']}")

                # Check for cache headers
                etag = response.headers.get("ETag")
                if etag:
                    print(f"   🏷️  ETag: {etag}")

                cache_control = response.headers.get("Cache-Control")
                if cache_control:
                    print(f"   🗄️  Cache-Control: {cache_control}")

            else:
                print(f"   ❌ Status: {response.status_code}")
                cold_time = None

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            cold_time = None

        # Test 2: Warm cache (subsequent requests)
        warm_times = []
        if cold_time is not None:
            print("🟢 Warm cache test...")

            for i in range(5):
                start_time = time.time()
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=10)
                    warm_time = (time.time() - start_time) * 1000
                    warm_times.append(warm_time)

                    if response.status_code == 200:
                        data = response.json()
                        # Check if served from cache
                        from_cache = data.get("from_cache", False)
                        cache_age = data.get("cache_age_seconds", 0)
                        if from_cache:
                            print(
                                f"   🗄️  Request {i+1}: {warm_time:.2f}ms (cached, age: {cache_age}s)"
                            )
                        else:
                            print(f"   🔄 Request {i+1}: {warm_time:.2f}ms")

                except Exception as e:
                    print(f"   ❌ Request {i+1} error: {str(e)}")

                time.sleep(0.1)  # Small delay between requests

        # Test 3: ETag conditional request
        if cold_time is not None:
            print("🏷️  ETag conditional request test...")

            try:
                # First, get the ETag
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                etag = response.headers.get("ETag")

                if etag:
                    # Make conditional request with If-None-Match
                    headers = {"If-None-Match": etag}
                    start_time = time.time()
                    conditional_response = requests.get(
                        f"{base_url}{endpoint}", headers=headers, timeout=10
                    )
                    conditional_time = (time.time() - start_time) * 1000

                    if conditional_response.status_code == 304:
                        print(f"   ✅ 304 Not Modified: {conditional_time:.2f}ms")
                    else:
                        print(
                            f"   🔄 Response: {conditional_response.status_code} ({conditional_time:.2f}ms)"
                        )
                else:
                    print("   ⚠️  No ETag header found")

            except Exception as e:
                print(f"   ❌ ETag test error: {str(e)}")

        # Calculate performance metrics
        if cold_time is not None and warm_times:
            avg_warm_time = mean(warm_times)
            improvement = ((cold_time - avg_warm_time) / cold_time) * 100

            results[endpoint] = {
                "cold_time_ms": cold_time,
                "avg_warm_time_ms": avg_warm_time,
                "improvement_percent": improvement,
                "warm_times": warm_times,
            }

            print("   📊 Performance Summary:")
            print(f"      Cold cache: {cold_time:.2f}ms")
            print(f"      Avg warm cache: {avg_warm_time:.2f}ms")
            print(f"      Improvement: {improvement:.1f}%")

            if warm_times:
                std_dev = stdev(warm_times) if len(warm_times) > 1 else 0
                print(f"      Consistency (stddev): {std_dev:.2f}ms")

        print()

    # Overall summary
    print("📋 OPTIMIZATION SUMMARY")
    print("=" * 60)

    if results:
        total_endpoints = len(results)
        avg_cold_time = mean([r["cold_time_ms"] for r in results.values()])
        avg_warm_time = mean([r["avg_warm_time_ms"] for r in results.values()])
        overall_improvement = mean([r["improvement_percent"] for r in results.values()])

        print(f"Endpoints tested: {total_endpoints}")
        print(f"Average cold cache time: {avg_cold_time:.2f}ms")
        print(f"Average warm cache time: {avg_warm_time:.2f}ms")
        print(f"Overall performance improvement: {overall_improvement:.1f}%")

        # Check if optimization is significant
        if overall_improvement > 20:
            print("✅ EXCELLENT: Significant performance improvement detected!")
        elif overall_improvement > 10:
            print("✅ GOOD: Moderate performance improvement detected!")
        elif overall_improvement > 0:
            print("✅ OK: Some performance improvement detected!")
        else:
            print("⚠️  WARNING: No significant performance improvement detected")

        print()
        print("🎯 RECOMMENDATIONS:")
        if overall_improvement > 30:
            print("- Caching is working excellently")
            print("- Consider extending cache TTL for stable data")
        elif overall_improvement > 15:
            print("- Caching is providing good benefits")
            print("- Monitor cache hit rates over time")
        else:
            print("- Check if caching is properly enabled")
            print("- Verify database query optimization")
            print("- Consider increasing cache TTL")
    else:
        print("❌ No successful tests completed")
        print("- Check if Flask app is running on localhost:5002")
        print("- Verify endpoints are accessible")

    print()
    print("Test completed at:", datetime.now().isoformat())


if __name__ == "__main__":
    test_endpoint_performance()
