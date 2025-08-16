#!/usr/bin/env python3
"""
Prometheus Metrics Exporter
============================

Extended Prometheus exporter for the Greyhound Racing system.
Includes thread count, CPU usage, and Guardian service metrics
to catch performance regressions.

Author: AI Assistant
Date: August 4, 2025
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import psutil
from prometheus_client import (
    CollectorRegistry, Gauge, Counter, Histogram, Info, 
    generate_latest, CONTENT_TYPE_LATEST, start_http_server
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class SystemMetricsCollector:
    """Collects system-level metrics for Prometheus"""
    
    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        self.process = psutil.Process()
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent', 
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage', 
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Process metrics
        self.process_cpu_usage = Gauge(
            'process_cpu_usage_percent',
            'Process CPU usage percentage',
            registry=self.registry
        )
        
        self.process_memory_usage = Gauge(
            'process_memory_usage_bytes',
            'Process memory usage in bytes',
            registry=self.registry
        )
        
        self.process_thread_count = Gauge(
            'process_thread_count',
            'Number of threads in the current process',
            registry=self.registry
        )
        
        self.process_file_descriptors = Gauge(
            'process_file_descriptors_count',
            'Number of open file descriptors',
            registry=self.registry
        )
        
        # Thread-specific metrics
        self.thread_count_by_name = Gauge(
            'threads_by_name',
            'Number of threads by thread name pattern',
            ['thread_name'],
            registry=self.registry
        )
        
        # System load metrics
        self.load_average_1m = Gauge(
            'system_load_average_1m',
            '1-minute load average',
            registry=self.registry
        )
        
        self.load_average_5m = Gauge(
            'system_load_average_5m',
            '5-minute load average',
            registry=self.registry
        )
        
        self.load_average_15m = Gauge(
            'system_load_average_15m',
            '15-minute load average',
            registry=self.registry
        )
    
    def collect_metrics(self):
        """Collect all system metrics"""
        try:
            # System metrics
            self.cpu_usage.set(psutil.cpu_percent(interval=0.1))
            
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            disk = psutil.disk_usage('.')
            self.disk_usage.set(disk.percent)
            
            # Process metrics
            self.process_cpu_usage.set(self.process.cpu_percent())
            self.process_memory_usage.set(self.process.memory_info().rss)
            self.process_thread_count.set(self.process.num_threads())
            
            try:
                self.process_file_descriptors.set(self.process.num_fds())
            except AttributeError:
                # Windows doesn't have num_fds
                pass
            
            # Collect thread information
            self._collect_thread_metrics()
            
            # System load (Unix only)
            try:
                load_avg = os.getloadavg()
                self.load_average_1m.set(load_avg[0])
                self.load_average_5m.set(load_avg[1])
                self.load_average_15m.set(load_avg[2])
            except (AttributeError, OSError):
                # Windows doesn't have getloadavg
                pass
                
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    def _collect_thread_metrics(self):
        """Collect detailed thread metrics"""
        try:
            # Get all threads
            thread_counts = {}
            
            for thread in threading.enumerate():
                thread_name = thread.name
                
                # Categorize threads by name patterns
                if 'Guardian' in thread_name:
                    category = 'guardian'
                elif 'ML' in thread_name or 'model' in thread_name.lower():
                    category = 'ml_processing'
                elif 'Flask' in thread_name or 'werkzeug' in thread_name.lower():
                    category = 'web_server'
                elif 'Thread' in thread_name:
                    category = 'worker_threads'
                elif 'Main' in thread_name:
                    category = 'main'
                else:
                    category = 'other'
                
                thread_counts[category] = thread_counts.get(category, 0) + 1
            
            # Update metrics
            for category, count in thread_counts.items():
                self.thread_count_by_name.labels(thread_name=category).set(count)
                
        except Exception as e:
            print(f"Error collecting thread metrics: {e}")

class ApplicationMetricsCollector:
    """Collects application-specific metrics"""
    
    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        
        # Guardian service metrics
        self.guardian_scan_duration = Histogram(
            'guardian_scan_duration_seconds',
            'Time taken for Guardian scans',
            registry=self.registry
        )
        
        self.guardian_files_scanned = Counter(
            'guardian_files_scanned_total',
            'Total number of files scanned by Guardian',
            registry=self.registry
        )
        
        self.guardian_files_quarantined = Counter(
            'guardian_files_quarantined_total',
            'Total number of files quarantined by Guardian',
            registry=self.registry
        )
        
        self.guardian_issues_found = Counter(
            'guardian_issues_found_total',
            'Total number of issues found by Guardian',
            ['issue_type'],
            registry=self.registry
        )
        
        # ML/Prediction metrics
        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'Time taken for ML predictions',
            registry=self.registry
        )
        
        self.prediction_accuracy = Gauge(
            'ml_prediction_accuracy_percent',
            'Current prediction accuracy percentage',
            ['prediction_type'],
            registry=self.registry
        )
        
        self.predictions_generated = Counter(
            'ml_predictions_generated_total',
            'Total number of predictions generated',
            registry=self.registry
        )
        
        # Database metrics
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Time taken for database queries',
            ['query_type'],
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'database_connections_active',
            'Number of active database connections',
            registry=self.registry
        )
        
        # File processing metrics
        self.files_processed = Counter(
            'files_processed_total',
            'Total number of files processed',
            ['file_type', 'status'],
            registry=self.registry
        )
        
        self.file_processing_duration = Histogram(
            'file_processing_duration_seconds',
            'Time taken to process files',
            ['file_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'greyhound_racing_app_info',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': '1.0.0',
            'component': 'greyhound_racing_collector',
            'start_time': datetime.now().isoformat()
        })
    
    def record_guardian_scan(self, duration: float, files_scanned: int, 
                           files_quarantined: int, issues: List[str]):
        """Record Guardian scan metrics"""
        self.guardian_scan_duration.observe(duration)
        self.guardian_files_scanned.inc(files_scanned)
        self.guardian_files_quarantined.inc(files_quarantined)
        
        # Count issues by type
        issue_types = {}
        for issue in issues:
            # Categorize issues
            if 'test file' in issue.lower():
                issue_type = 'test_file'
            elif 'corruption' in issue.lower() or 'null bytes' in issue.lower():
                issue_type = 'corruption'
            elif 'html content' in issue.lower():
                issue_type = 'html_contamination'
            elif 'size' in issue.lower():
                issue_type = 'size_anomaly'
            else:
                issue_type = 'other'
            
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        for issue_type, count in issue_types.items():
            self.guardian_issues_found.labels(issue_type=issue_type).inc(count)
    
    def record_prediction(self, duration: float, accuracy: float = None, 
                         prediction_type: str = 'win'):
        """Record prediction metrics"""
        self.prediction_latency.observe(duration)
        self.predictions_generated.inc()
        
        if accuracy is not None:
            self.prediction_accuracy.labels(prediction_type=prediction_type).set(accuracy)
    
    def record_database_query(self, duration: float, query_type: str = 'select'):
        """Record database query metrics"""
        self.database_query_duration.labels(query_type=query_type).observe(duration)
    
    def record_file_processing(self, duration: float, file_type: str, status: str):
        """Record file processing metrics"""
        self.file_processing_duration.labels(file_type=file_type).observe(duration)
        self.files_processed.labels(file_type=file_type, status=status).inc()

class PrometheusExporter:
    """Main Prometheus metrics exporter"""
    
    def __init__(self, port: int = 8000, update_interval: int = 15):
        self.port = port
        self.update_interval = update_interval
        self.registry = CollectorRegistry()
        
        # Initialize collectors
        self.system_collector = SystemMetricsCollector(self.registry)
        self.app_collector = ApplicationMetricsCollector(self.registry)
        
        # Control variables
        self.running = False
        self.update_thread = None
        
        print(f"Prometheus exporter initialized on port {port}")
        print(f"Metrics will be updated every {update_interval} seconds")
    
    def start(self):
        """Start the Prometheus HTTP server and metrics collection"""
        if self.running:
            print("Exporter is already running")
            return
        
        try:
            # Start HTTP server
            start_http_server(self.port, registry=self.registry)
            print(f"Prometheus metrics server started on port {self.port}")
            print(f"Metrics available at: http://localhost:{self.port}/metrics")
            
            # Start metrics collection thread
            self.running = True
            self.update_thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
            self.update_thread.start()
            
            print("Metrics collection started")
            
        except Exception as e:
            print(f"Failed to start Prometheus exporter: {e}")
            raise
    
    def stop(self):
        """Stop the metrics collection"""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        print("Prometheus exporter stopped")
    
    def _update_metrics_loop(self):
        """Main metrics update loop"""
        while self.running:
            try:
                self.system_collector.collect_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in metrics update loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def record_guardian_activity(self, scan_duration: float, files_scanned: int,
                               files_quarantined: int, issues: List[str]):
        """Record Guardian service activity"""
        self.app_collector.record_guardian_scan(
            scan_duration, files_scanned, files_quarantined, issues
        )
    
    def record_ml_activity(self, prediction_duration: float, accuracy: float = None):
        """Record ML prediction activity"""
        self.app_collector.record_prediction(prediction_duration, accuracy)

# Global exporter instance
_prometheus_exporter = None

def get_prometheus_exporter() -> PrometheusExporter:
    """Get or create the global Prometheus exporter instance"""
    global _prometheus_exporter
    if _prometheus_exporter is None:
        port = int(os.getenv('PROMETHEUS_PORT', '8000'))
        interval = int(os.getenv('METRICS_UPDATE_INTERVAL', '15'))
        _prometheus_exporter = PrometheusExporter(port=port, update_interval=interval)
    return _prometheus_exporter

def start_prometheus_exporter():
    """Start the global Prometheus exporter"""
    exporter = get_prometheus_exporter()
    exporter.start()
    return exporter

def stop_prometheus_exporter():
    """Stop the global Prometheus exporter"""
    global _prometheus_exporter
    if _prometheus_exporter:
        _prometheus_exporter.stop()

def main():
    """Main entry point for standalone execution"""
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description='Prometheus Metrics Exporter')
    parser.add_argument('--port', type=int, default=8000, help='HTTP server port')
    parser.add_argument('--interval', type=int, default=15, help='Metrics update interval')
    parser.add_argument('--test', action='store_true', help='Run a quick test')
    
    args = parser.parse_args()
    
    if args.test:
        # Run a quick test
        exporter = PrometheusExporter(port=args.port, update_interval=1)
        exporter.start()
        
        print("Running test for 30 seconds...")
        time.sleep(30)
        
        print("\nSample metrics:")
        print(exporter.get_metrics()[:1000] + "...")
        
        exporter.stop()
        return
    
    # Start the exporter
    exporter = PrometheusExporter(port=args.port, update_interval=args.interval)
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        exporter.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    exporter.start()
    
    print("\nPrometheus Exporter Running")
    print("=" * 30)
    print(f"Port: {args.port}")
    print(f"Update interval: {args.interval} seconds")
    print(f"Metrics URL: http://localhost:{args.port}/metrics")
    print("Press Ctrl+C to stop")
    print()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        exporter.stop()

if __name__ == "__main__":
    main()
