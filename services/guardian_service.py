#!/usr/bin/env python3
"""
Guardian Service
===============

Provides periodic scanning and cleanup of problematic files.
Integrates with the main application to prevent corruption and test pollution.

Author: AI Assistant
Date: August 4, 2025
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_integrity_guardian import FileIntegrityGuardian

# Import Prometheus exporter if available
try:
    from monitoring.prometheus_exporter import get_prometheus_exporter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class GuardianService:
    """Background service for continuous file integrity monitoring"""
    
    def __init__(self, scan_interval_minutes: int = 240):
        self.guardian = FileIntegrityGuardian()
        self.scan_interval = scan_interval_minutes * 60  # Convert to seconds
        self.is_running = False
        self.last_scan_time = None
        self.scan_thread = None
        
        # Reduced monitored directories (long-term fix)
        self.monitored_directories = [
            './upcoming_races',
            './processed'
        ]
        
        # File age filtering (only process files newer than this many hours)
        self.max_file_age_hours = 24
        
        # Statistics
        self.stats = {
            'total_scans': 0,
            'files_quarantined': 0,
            'test_files_removed': 0,
            'last_issues_found': 0
        }
        
        # Prometheus metrics integration
        self.prometheus_exporter = None
        if PROMETHEUS_AVAILABLE:
            try:
                self.prometheus_exporter = get_prometheus_exporter()
                print(f"📊 Prometheus metrics integration enabled")
            except Exception as e:
                print(f"⚠️ Failed to initialize Prometheus metrics: {e}")
        
        print(f"🛡️ Guardian Service initialized")
        print(f"⏱️ Scan interval: {scan_interval_minutes} minutes")
        print(f"📁 Monitoring directories: {', '.join(self.monitored_directories)}")
    
    def start(self):
        """Start the guardian service"""
        if self.is_running:
            print("⚠️ Guardian service is already running")
            return
        
        self.is_running = True
        self.scan_thread = threading.Thread(target=self._run_service, daemon=True)
        self.scan_thread.start()
        
        print("🚀 Guardian Service started")
    
    def stop(self):
        """Stop the guardian service"""
        self.is_running = False
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=5)
        
        print("🛑 Guardian Service stopped")
    
    def _run_service(self):
        """Main service loop"""
        while self.is_running:
            try:
                self._perform_scan()
                time.sleep(self.scan_interval)
            except Exception as e:
                print(f"❌ Guardian service error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _perform_scan(self):
        """Perform a comprehensive scan of monitored directories"""
        scan_start_time = datetime.now()
        print(f"🔍 Guardian scan starting at {scan_start_time.strftime('%H:%M:%S')}")
        
        scan_results = []
        issues_found = 0
        files_quarantined = 0
        total_directory_size = 0
        total_files_scanned = 0
        
        for directory in self.monitored_directories:
            if not os.path.exists(directory):
                continue
                
            # Calculate directory size for performance tuning
            dir_start_time = datetime.now()
            dir_size = 0
            dir_file_count = 0
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        dir_size += os.path.getsize(file_path)
                        dir_file_count += 1
                    except OSError:
                        pass  # Skip files we can't access
            
            total_directory_size += dir_size
            total_files_scanned += dir_file_count
                
            print(f"  📂 Scanning: {directory} ({dir_file_count} files, {dir_size/1024/1024:.1f}MB)")
            results = self.guardian.scan_directory(directory, max_age_hours=self.max_file_age_hours)
            scan_results.extend(results)
            
            dir_duration = (datetime.now() - dir_start_time).total_seconds()
            print(f"    ⏱️ Directory scan completed in {dir_duration:.2f}s")
            
            # Count issues and quarantined files
            for result in results:
                if result.issues:
                    issues_found += len(result.issues)
                if result.should_quarantine:
                    files_quarantined += 1
        
        # Clean up test files
        test_files_removed = self.guardian.cleanup_test_files(self.monitored_directories, max_age_hours=self.max_file_age_hours)
        
        # Calculate total scan duration
        scan_duration = (datetime.now() - scan_start_time).total_seconds()
        
        # Update statistics
        self.stats['total_scans'] += 1
        self.stats['files_quarantined'] += files_quarantined
        self.stats['test_files_removed'] += test_files_removed
        self.stats['last_issues_found'] = issues_found
        self.last_scan_time = datetime.now()
        
        # Record Prometheus metrics
        if self.prometheus_exporter and PROMETHEUS_AVAILABLE:
            try:
                # Collect all issues from scan results
                all_issues = []
                for result in scan_results:
                    all_issues.extend(result.issues)
                
                self.prometheus_exporter.record_guardian_activity(
                    scan_duration, total_files_scanned, files_quarantined, all_issues
                )
            except Exception as e:
                print(f"⚠️ Failed to record Prometheus metrics: {e}")
        
        # Log performance metrics for future tuning
        print(f"📊 Scan Performance Metrics:")
        print(f"   Total duration: {scan_duration:.2f}s")
        print(f"   Files scanned: {total_files_scanned}")
        print(f"   Total data: {total_directory_size/1024/1024:.1f}MB")
        print(f"   Scan rate: {total_files_scanned/scan_duration:.1f} files/sec")
        print(f"   Data rate: {total_directory_size/1024/1024/scan_duration:.1f} MB/sec")
        
        # Generate report
        if issues_found > 0 or files_quarantined > 0 or test_files_removed > 0:
            print(f"⚠️ Guardian scan completed:")
            print(f"   Issues found: {issues_found}")
            print(f"   Files quarantined: {files_quarantined}")
            print(f"   Test files removed: {test_files_removed}")
            
            self._save_scan_report(scan_results)
        else:
            print(f"✅ Guardian scan completed - no issues found")
    
    def _save_scan_report(self, results: List):
        """Save scan report to logs"""
        try:
            report = self.guardian.generate_integrity_report(results)
            report['service_stats'] = self.stats.copy()
            
            os.makedirs('./logs/guardian_reports', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f'./logs/guardian_reports/guardian_scan_{timestamp}.json'
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"📝 Scan report saved: {report_path}")
            
        except Exception as e:
            print(f"❌ Failed to save scan report: {e}")
    
    def get_status(self) -> Dict:
        """Get current service status"""
        return {
            'is_running': self.is_running,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'scan_interval_minutes': self.scan_interval // 60,
            'monitored_directories': self.monitored_directories,
            'stats': self.stats.copy()
        }
    
    def force_scan(self) -> Dict:
        """Force an immediate scan and return results"""
        if not self.is_running:
            print("⚠️ Guardian service is not running, performing one-time scan")
            
        self._perform_scan()
        return self.get_status()

# Global service instance
_guardian_service = None

def get_guardian_service() -> GuardianService:
    """Get or create the global guardian service instance"""
    global _guardian_service
    if os.getenv('GUARDIAN_DISABLE', 'false').lower() == 'true':
        print("🛡️ Guardian Service auto-start disabled by environment variable")
        return None
    if _guardian_service is None:
        _guardian_service = GuardianService()
    return _guardian_service

def start_guardian_service():
    """Start the guardian service"""
    service = get_guardian_service()
    if service is None:
        return None
    service.start()
    return service

def stop_guardian_service():
    """Stop the guardian service"""
    global _guardian_service
    if _guardian_service:
        _guardian_service.stop()

if __name__ == "__main__":
    # CLI usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Guardian Service')
    parser.add_argument('--start', action='store_true', help='Start the service')
    parser.add_argument('--scan-now', action='store_true', help='Perform immediate scan')
    parser.add_argument('--interval', type=int, default=240, help='Scan interval in minutes')
    
    args = parser.parse_args()
    
    if args.start:
        service = GuardianService(scan_interval_minutes=args.interval)
        service.start()
        
        try:
            print("Guardian service running. Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping guardian service...")
            service.stop()
    
    elif args.scan_now:
        service = GuardianService()
        status = service.force_scan()
        print(f"\n📊 Scan completed:")
        print(f"   Total scans: {status['stats']['total_scans']}")
        print(f"   Files quarantined: {status['stats']['files_quarantined']}")
        print(f"   Test files removed: {status['stats']['test_files_removed']}")
    
    else:
        print("🛡️ Guardian Service")
        print("Use --help for available options")
