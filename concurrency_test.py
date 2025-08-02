#!/usr/bin/env python3
"""
Concurrency Test Script for Greyhound Racing Collector Flask App
Tests multiple endpoints with varying load levels to validate performance improvements.
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from typing import List, Dict, Any
import json

class ConcurrencyTester:
    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {}
        
    def test_endpoint_sync(self, endpoint: str, num_requests: int = 10) -> Dict[str, Any]:
        """Test endpoint with synchronous requests"""
        print(f"\n🧪 Testing {endpoint} with {num_requests} synchronous requests...")
        
        response_times = []
        status_codes = []
        errors = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                response_start = time.time()
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=30)
                response_time = (time.time() - response_start) * 1000  # Convert to ms
                
                response_times.append(response_time)
                status_codes.append(response.status_code)
                
                if response.status_code != 200:
                    errors += 1
                    print(f"❌ Request {i+1}: {response.status_code} - {response.text[:100]}")
                else:
                    print(f"✅ Request {i+1}: {response_time:.2f}ms")
                    
            except Exception as e:
                errors += 1
                print(f"❌ Request {i+1}: Error - {str(e)}")
                
        total_time = time.time() - start_time
        
        if response_times:
            return {
                'endpoint': endpoint,
                'total_requests': num_requests,
                'successful_requests': num_requests - errors,
                'errors': errors,
                'total_time': total_time,
                'avg_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'median_response_time': statistics.median(response_times),
                'requests_per_second': num_requests / total_time if total_time > 0 else 0,
                'status_codes': status_codes
            }
        else:
            return {
                'endpoint': endpoint,
                'total_requests': num_requests,
                'successful_requests': 0,
                'errors': errors,
                'total_time': total_time,
                'requests_per_second': 0
            }
    
    def test_endpoint_concurrent(self, endpoint: str, num_requests: int = 20, max_workers: int = 10) -> Dict[str, Any]:
        """Test endpoint with concurrent requests using ThreadPoolExecutor"""
        print(f"\n🚀 Testing {endpoint} with {num_requests} concurrent requests ({max_workers} workers)...")
        
        response_times = []
        status_codes = []
        errors = 0
        
        def make_request():
            try:
                response_start = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
                response_time = (time.time() - response_start) * 1000
                return response_time, response.status_code, None
            except Exception as e:
                return None, None, str(e)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            
            for i, future in enumerate(as_completed(futures), 1):
                response_time, status_code, error = future.result()
                
                if error:
                    errors += 1
                    print(f"❌ Request {i}: Error - {error}")
                elif status_code != 200:
                    errors += 1
                    status_codes.append(status_code)
                    print(f"❌ Request {i}: {status_code}")
                else:
                    response_times.append(response_time)
                    status_codes.append(status_code)
                    print(f"✅ Request {i}: {response_time:.2f}ms")
        
        total_time = time.time() - start_time
        
        if response_times:
            return {
                'endpoint': endpoint,
                'test_type': 'concurrent',
                'total_requests': num_requests,
                'successful_requests': num_requests - errors,
                'errors': errors,
                'total_time': total_time,
                'avg_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'median_response_time': statistics.median(response_times),
                'requests_per_second': num_requests / total_time if total_time > 0 else 0,
                'max_workers': max_workers
            }
        else:
            return {
                'endpoint': endpoint,
                'test_type': 'concurrent',
                'total_requests': num_requests,
                'successful_requests': 0,
                'errors': errors,
                'total_time': total_time,
                'requests_per_second': 0,
                'max_workers': max_workers
            }
    
    def run_comprehensive_test(self):
        """Run comprehensive tests on all available endpoints"""
        print("🎯 Starting Comprehensive Concurrency Tests")
        print("=" * 50)
        
        # Test endpoints in order of complexity
        test_cases = [
            # Basic health check
            ('/api/health', 15, 5),
            # Stats endpoint  
            ('/api/stats', 10, 5),
            # More complex endpoints if available
            ('/api/dogs', 8, 4),
            ('/api/races', 8, 4),
        ]
        
        all_results = []
        
        for endpoint, num_requests, max_workers in test_cases:
            try:
                # Test synchronous first
                sync_result = self.test_endpoint_sync(endpoint, num_requests // 2)
                all_results.append(sync_result)
                
                # Wait a bit between tests
                time.sleep(2)
                
                # Test concurrent
                concurrent_result = self.test_endpoint_concurrent(endpoint, num_requests, max_workers)
                all_results.append(concurrent_result)
                
                # Wait between endpoint tests
                time.sleep(3)
                
            except KeyboardInterrupt:
                print("\n⚠️ Test interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error testing {endpoint}: {str(e)}")
                continue
        
        # Generate summary report
        self.generate_report(all_results)
        return all_results
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """Generate a comprehensive performance report"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TESTING SUMMARY REPORT")
        print("=" * 60)
        
        for result in results:
            if 'successful_requests' in result and result['successful_requests'] > 0:
                print(f"\n🎯 Endpoint: {result['endpoint']}")
                test_type = result.get('test_type', 'synchronous')
                print(f"   Test Type: {test_type.title()}")
                print(f"   Successful Requests: {result['successful_requests']}/{result['total_requests']}")
                print(f"   Error Rate: {(result['errors']/result['total_requests']*100):.1f}%")
                print(f"   Requests/Second: {result['requests_per_second']:.2f}")
                
                if 'avg_response_time' in result:
                    print(f"   Avg Response Time: {result['avg_response_time']:.2f}ms")
                    print(f"   Min Response Time: {result['min_response_time']:.2f}ms")
                    print(f"   Max Response Time: {result['max_response_time']:.2f}ms")
                    print(f"   Median Response Time: {result['median_response_time']:.2f}ms")
                
                if test_type == 'concurrent':
                    print(f"   Max Workers: {result.get('max_workers', 'N/A')}")
        
        # Calculate overall metrics
        successful_tests = [r for r in results if r.get('successful_requests', 0) > 0]
        if successful_tests:
            total_requests = sum(r['total_requests'] for r in successful_tests)
            total_successful = sum(r['successful_requests'] for r in successful_tests)
            total_errors = sum(r['errors'] for r in successful_tests)
            
            print(f"\n📈 OVERALL METRICS:")
            print(f"   Total Requests Sent: {total_requests}")
            print(f"   Total Successful: {total_successful}")
            print(f"   Total Errors: {total_errors}")
            print(f"   Overall Success Rate: {(total_successful/total_requests*100):.1f}%")
            
            # Average response times for concurrent tests
            concurrent_results = [r for r in successful_tests if r.get('test_type') == 'concurrent' and 'avg_response_time' in r]
            if concurrent_results:
                avg_concurrent_time = statistics.mean([r['avg_response_time'] for r in concurrent_results])
                print(f"   Avg Concurrent Response Time: {avg_concurrent_time:.2f}ms")

if __name__ == "__main__":
    print("🚀 Greyhound Racing Collector - Concurrency Performance Test")
    print("Testing Flask app with optimized DB connection pool and lazy loading...")
    
    tester = ConcurrencyTester()
    
    # Wait for server to be ready
    print("\n⏳ Waiting for server to be ready...")
    time.sleep(3)
    
    try:
        results = tester.run_comprehensive_test()
        
        # Save results to file
        with open('concurrency_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to concurrency_test_results.json")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test suite interrupted by user")
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
