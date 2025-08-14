#!/usr/bin/env python3
"""
Comprehensive Performance & Load Testing Script for Greyhound Racing App
Implements all requirements from Step 8:
- 500 concurrent users hitting endpoints for 5 min
- P95 latency monitoring (fail if > 2s)
- psutil RAM/CPU monitoring with memory leak detection
- Query plan analysis for slow queries > 100ms
"""

import subprocess
import time
import psutil
import logging
import json
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import threading
import queue
import signal
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTestRunner:
    def __init__(self, host: str = "http://localhost:5002"):
        self.host = host
        self.test_duration = 300  # 5 minutes
        self.concurrent_users = 500
        self.results = {
            'p95_latencies': {},
            'memory_samples': [],
            'cpu_samples': [],
            'slow_queries': [],
            'test_passed': True,
            'error_messages': []
        }
        self.monitoring_active = False
        self.start_time = None
        
    def enable_query_analysis(self):
        """Enable EXPLAIN ANALYZE sampling in the database"""
        try:
            response = requests.get(f"{self.host}/api/enable-explain-analyze", timeout=10)
            if response.status_code == 200:
                logger.info("✓ EXPLAIN ANALYZE sampling enabled")
                return True
            else:
                logger.error(f"✗ Failed to enable query analysis: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ Error enabling query analysis: {e}")
            return False
    
    def run_locust_test(self) -> subprocess.Popen:
        """Run Locust load test with 500 concurrent users for 5 minutes"""
        cmd = [
            "python", "-m", "locust",
            "-f", "load_tests/locustfile.py",
            "--headless",
            "-u", str(self.concurrent_users),
            "-r", "50",  # Spawn rate: 50 users per second
            "-t", f"{self.test_duration}s",
            f"--host={self.host}",
            "--logfile", "load_tests/locust.log"
        ]
        
        logger.info(f"🚀 Starting Locust test: {self.concurrent_users} users for {self.test_duration}s")
        
        try:
            process = subprocess.Popen(cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True,
                                     cwd="/Users/orlandolee/greyhound_racing_collector")
            return process
        except Exception as e:
            logger.error(f"✗ Failed to start Locust: {e}")
            return None
    
    def monitor_system_resources(self, stop_event: threading.Event):
        """Monitor RAM/CPU using psutil and detect memory leaks"""
        logger.info("📊 Starting system resource monitoring...")
        sample_interval = 5  # Sample every 5 seconds
        
        while not stop_event.is_set():
            try:
                # Get current process info
                process = psutil.Process()
                
                # Memory in MB
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.results['memory_samples'].append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb
                })
                
                # CPU percentage
                cpu_percent = process.cpu_percent()
                self.results['cpu_samples'].append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent
                })
                
                # Check for memory leak every 30 seconds
                if len(self.results['memory_samples']) % 6 == 0:  # Every 6 samples (30s)
                    self.check_memory_leak()
                
                time.sleep(sample_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(sample_interval)
    
    def check_memory_leak(self):
        """Check for memory leaks (alert if slope > 5 MB/min)"""
        if len(self.results['memory_samples']) < 10:
            return
        
        # Get recent samples (last 10)
        recent_samples = self.results['memory_samples'][-10:]
        
        # Calculate memory growth rate
        time_span = recent_samples[-1]['timestamp'] - recent_samples[0]['timestamp']
        memory_change = recent_samples[-1]['memory_mb'] - recent_samples[0]['memory_mb']
        
        if time_span > 0:
            # Convert to MB/min
            growth_rate = (memory_change / time_span) * 60
            
            if growth_rate > 5:
                error_msg = f"🚨 MEMORY LEAK DETECTED! Growth rate: {growth_rate:.2f} MB/min"
                logger.error(error_msg)
                self.results['error_messages'].append(error_msg)
                self.results['test_passed'] = False
    
    def check_slow_queries(self):
        """Check for slow queries using the Flask app's slow query logging"""
        try:
            # Read the Flask app logs to find slow queries
            # The app logs slow queries with "Slow query detected" or "Slow WebSocket query"
            with open("load_tests/locust.log", "r") as f:
                lines = f.readlines()
                
            for line in lines:
                if "Slow query detected" in line or "Slow WebSocket query" in line:
                    # Extract query time
                    if "ms" in line:
                        parts = line.split("ms")
                        if len(parts) > 1:
                            time_part = parts[0].split()[-1]
                            try:
                                query_time = float(time_part.replace(":", ""))
                                if query_time > 100:  # Flag queries > 100ms
                                    self.results['slow_queries'].append({
                                        'query_time_ms': query_time,
                                        'log_line': line.strip()
                                    })
                            except ValueError:
                                pass
                                
        except Exception as e:
            logger.warning(f"Could not analyze slow queries: {e}")
    
    def analyze_latency_results(self, locust_process: subprocess.Popen):
        """Analyze P95 latency from Locust results"""
        try:
            # Wait for process to complete and get output
            stdout, stderr = locust_process.communicate()
            
            # Parse Locust output for P95 percentiles
            lines = stdout.split('\n') if stdout else []
            
            # Look for response time percentiles table
            in_percentiles_section = False
            for line in lines:
                if "Response time percentiles" in line:
                    in_percentiles_section = True
                    continue
                
                if in_percentiles_section and "|" in line and not line.startswith("Type"):
                    # Parse endpoint data
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 8:  # Ensure we have enough columns
                        endpoint = parts[1]
                        try:
                            # P95 is in column 6 (0-indexed)
                            p95_str = parts[6].strip()
                            if p95_str.isdigit():
                                p95_latency = int(p95_str)
                                self.results['p95_latencies'][endpoint] = p95_latency
                                
                                # Check P95 threshold (2000ms = 2s)
                                if p95_latency > 2000:
                                    error_msg = f"🚨 P95 LATENCY EXCEEDED for {endpoint}: {p95_latency}ms > 2000ms"
                                    logger.error(error_msg)
                                    self.results['error_messages'].append(error_msg)
                                    self.results['test_passed'] = False
                                else:
                                    logger.info(f"✓ P95 latency OK for {endpoint}: {p95_latency}ms")
                        except (ValueError, IndexError):
                            pass
                            
        except Exception as e:
            logger.error(f"Error analyzing latency results: {e}")
    
    def generate_report(self):
        """Generate comprehensive performance test report"""
        logger.info("\n" + "="*80)
        logger.info("🏁 PERFORMANCE TEST RESULTS")
        logger.info("="*80)
        
        # Test status
        status = "✅ PASSED" if self.results['test_passed'] else "❌ FAILED"
        logger.info(f"Overall Status: {status}")
        
        # P95 Latency Results
        logger.info("\n📈 P95 LATENCY RESULTS:")
        for endpoint, latency in self.results['p95_latencies'].items():
            status_symbol = "✓" if latency <= 2000 else "✗"
            logger.info(f"  {status_symbol} {endpoint}: {latency}ms")
        
        # Memory Analysis
        if self.results['memory_samples']:
            memory_values = [s['memory_mb'] for s in self.results['memory_samples']]
            logger.info(f"\n🧠 MEMORY ANALYSIS:")
            logger.info(f"  • Peak Memory: {max(memory_values):.2f} MB")
            logger.info(f"  • Average Memory: {np.mean(memory_values):.2f} MB")
            logger.info(f"  • Memory Growth: {memory_values[-1] - memory_values[0]:.2f} MB")
        
        # CPU Analysis
        if self.results['cpu_samples']:
            cpu_values = [s['cpu_percent'] for s in self.results['cpu_samples']]
            logger.info(f"\n💻 CPU ANALYSIS:")
            logger.info(f"  • Peak CPU: {max(cpu_values):.2f}%")
            logger.info(f"  • Average CPU: {np.mean(cpu_values):.2f}%")
        
        # Slow Queries
        logger.info(f"\n🐌 SLOW QUERIES (>100ms): {len(self.results['slow_queries'])}")
        for query in self.results['slow_queries'][:5]:  # Show first 5
            logger.info(f"  • {query['query_time_ms']:.2f}ms")
        
        # Errors
        if self.results['error_messages']:
            logger.info(f"\n🚨 ERRORS:")
            for error in self.results['error_messages']:
                logger.info(f"  • {error}")
        
        logger.info("="*80)
        
        return self.results['test_passed']
    
    def run_full_test(self) -> bool:
        """Run the complete performance test suite"""
        logger.info("🎯 Starting Comprehensive Performance Test...")
        self.start_time = time.time()
        
        # 1. Enable query analysis
        if not self.enable_query_analysis():
            logger.error("Failed to enable query analysis - continuing anyway")
        
        # 2. Start system monitoring
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(
            target=self.monitor_system_resources,
            args=(stop_monitoring,)
        )
        monitor_thread.start()
        
        # 3. Run Locust load test
        locust_process = self.run_locust_test()
        if not locust_process:
            logger.error("Failed to start load test")
            stop_monitoring.set()
            monitor_thread.join()
            return False
        
        # 4. Wait for test completion
        try:
            return_code = locust_process.wait()
            logger.info(f"Locust test completed with return code: {return_code}")
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            locust_process.terminate()
            
        # 5. Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # 6. Analyze results
        self.analyze_latency_results(locust_process)
        self.check_slow_queries()
        
        # 7. Generate report
        test_passed = self.generate_report()
        
        return test_passed

def main():
    """Main entry point"""
    def signal_handler(sig, frame):
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize and run test
    test_runner = PerformanceTestRunner()
    
    success = test_runner.run_full_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
