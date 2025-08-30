#!/usr/bin/env python3
"""
TGR System Status Checker
=========================

Quick status check utility for the deployed TGR system.
"""

import sqlite3
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

def check_processes():
    """Check if TGR processes are running."""
    
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout
        
        tgr_processes = []
        for line in processes.split('\n'):
            if 'tgr' in line.lower() and 'python' in line:
                tgr_processes.append(line.strip())
        
        return tgr_processes
    except Exception:
        return []

def check_database_health():
    """Check database connectivity and recent activity."""
    
    try:
        conn = sqlite3.connect("greyhound_racing_data.db")
        cursor = conn.cursor()
        
        # Check total records
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        total_races = cursor.fetchone()[0]
        
        # Check TGR enrichment activity
        cursor.execute("SELECT COUNT(*) FROM tgr_enrichment_jobs WHERE created_at >= datetime('now', '-24 hours')")
        recent_jobs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tgr_enrichment_jobs WHERE status = 'completed' AND completed_at >= datetime('now', '-24 hours')")
        completed_jobs = cursor.fetchone()[0]
        
        # Check performance data
        cursor.execute("SELECT COUNT(*) FROM tgr_dog_performance_summary")
        performance_summaries = cursor.fetchone()[0]
        
        # Check cache entries
        cursor.execute("SELECT COUNT(*) FROM tgr_enhanced_feature_cache WHERE expires_at > datetime('now')")
        active_cache = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'database_accessible': True,
            'total_races': total_races,
            'recent_jobs_24h': recent_jobs,
            'completed_jobs_24h': completed_jobs,
            'performance_summaries': performance_summaries,
            'active_cache_entries': active_cache,
            'success_rate': (completed_jobs / recent_jobs * 100) if recent_jobs > 0 else 0
        }
        
    except Exception as e:
        return {
            'database_accessible': False,
            'error': str(e)
        }

def check_log_files():
    """Check recent log activity."""
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return {'logs_available': False}
    
    log_files = list(logs_dir.glob("*.log"))
    if not log_files:
        return {'logs_available': False}
    
    # Get most recent log file
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    
    # Check file size and recent activity
    file_size = latest_log.stat().st_size
    last_modified = datetime.fromtimestamp(latest_log.stat().st_mtime)
    
    # Read last few lines
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-5:] if len(lines) >= 5 else lines
    except Exception:
        recent_lines = []
    
    return {
        'logs_available': True,
        'latest_log': str(latest_log),
        'file_size_kb': file_size // 1024,
        'last_modified': last_modified.isoformat(),
        'recent_entries': len(recent_lines),
        'is_active': (datetime.now() - last_modified) < timedelta(minutes=5)
    }

def check_config_files():
    """Check if configuration files exist."""
    
    config_files = [
        'production_config.py',
        'tgr_monitoring_dashboard.py',
        'tgr_enrichment_service.py',
        'tgr_service_scheduler.py'
    ]
    
    status = {}
    for config_file in config_files:
        status[config_file] = Path(config_file).exists()
    
    return status

def print_status_report():
    """Generate and print comprehensive status report."""
    
    print("🎯 TGR System Status Report")
    print("=" * 50)
    print(f"📅 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check processes
    print("🔍 Process Status:")
    processes = check_processes()
    if processes:
        print(f"   ✅ Found {len(processes)} TGR process(es)")
        for proc in processes[:3]:  # Show first 3 processes
            print(f"   📊 {proc}")
    else:
        print("   ❌ No TGR processes found")
    print()
    
    # Check database
    print("🗄️ Database Status:")
    db_status = check_database_health()
    if db_status.get('database_accessible'):
        print("   ✅ Database accessible")
        print(f"   📊 Total races: {db_status['total_races']:,}")
        print(f"   ⚙️ Recent jobs (24h): {db_status['recent_jobs_24h']}")
        print(f"   ✅ Completed jobs (24h): {db_status['completed_jobs_24h']}")
        print(f"   📈 Success rate: {db_status['success_rate']:.1f}%")
        print(f"   🐕 Performance summaries: {db_status['performance_summaries']}")
        print(f"   🔄 Active cache entries: {db_status['active_cache_entries']}")
    else:
        print(f"   ❌ Database error: {db_status.get('error', 'Unknown')}")
    print()
    
    # Check logs
    print("📋 Log Status:")
    log_status = check_log_files()
    if log_status.get('logs_available'):
        print("   ✅ Log files available")
        print(f"   📄 Latest: {log_status['latest_log']}")
        print(f"   📏 Size: {log_status['file_size_kb']} KB")
        print(f"   ⏱️ Last modified: {log_status['last_modified']}")
        print(f"   🔄 Active: {'Yes' if log_status['is_active'] else 'No'}")
    else:
        print("   ❌ No log files found")
    print()
    
    # Check configuration
    print("⚙️ Configuration Status:")
    config_status = check_config_files()
    for config_file, exists in config_status.items():
        status_icon = "✅" if exists else "❌"
        print(f"   {status_icon} {config_file}")
    print()
    
    # Overall health assessment
    overall_health = "HEALTHY"
    issues = []
    
    if not processes:
        overall_health = "DOWN"
        issues.append("No processes running")
    
    if not db_status.get('database_accessible'):
        overall_health = "CRITICAL"
        issues.append("Database not accessible")
    
    if db_status.get('success_rate', 0) < 50:
        overall_health = "DEGRADED"
        issues.append("Low job success rate")
    
    if not log_status.get('is_active', False):
        if overall_health == "HEALTHY":
            overall_health = "WARNING"
        issues.append("Logs not active")
    
    # Print overall status
    health_icons = {
        'HEALTHY': '✅',
        'WARNING': '⚠️',
        'DEGRADED': '🔶',
        'CRITICAL': '🚨',
        'DOWN': '❌'
    }
    
    print(f"🎯 Overall Status: {health_icons.get(overall_health, '❓')} {overall_health}")
    
    if issues:
        print("⚠️ Issues detected:")
        for issue in issues:
            print(f"   • {issue}")
    
    print("\n" + "=" * 50)
    
    # Management suggestions
    if overall_health != "HEALTHY":
        print("🔧 Suggested Actions:")
        if "No processes running" in issues:
            print("   • Run: python3 deploy_tgr_system.py")
        if "Database not accessible" in issues:
            print("   • Check database file permissions")
        if "Low job success rate" in issues:
            print("   • Check logs for error patterns")
        if "Logs not active" in issues:
            print("   • Restart system if needed")
    else:
        print("✅ System is running normally")
        print("📋 View live logs: tail -f logs/tgr_system_*.log")

def main():
    """Main status check function."""
    print_status_report()

if __name__ == "__main__":
    main()
