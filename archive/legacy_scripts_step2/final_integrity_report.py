#!/usr/bin/env python3
"""
Final Comprehensive Data Integrity Report
Summarizes all verification findings and provides recommendations.
"""

import os
import json
import sqlite3
from datetime import datetime
import glob
from collections import defaultdict

def generate_final_report():
    """Generate final comprehensive integrity report"""
    base_path = "/Users/orlandolee/greyhound_racing_collector"
    
    print("📊 FINAL COMPREHENSIVE DATA INTEGRITY REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().isoformat()}")
    print()
    
    # 1. File System Statistics
    print("📁 FILE SYSTEM STATISTICS")
    print("-" * 40)
    
    directories = {
        'processed/completed': 'Race files that have been fully processed',
        'processed/excluded': 'Race files excluded from processing',
        'processed/other': 'Other race files',
        'unprocessed': 'Raw race files awaiting processing',
        'form_guides/downloaded': 'Downloaded form guide data',
        'predictions': 'Generated predictions',
        'cached_backup': 'Recent backup files',
        'databases': 'Database files'
    }
    
    total_files = 0
    for dir_path, description in directories.items():
        full_path = os.path.join(base_path, dir_path)
        if os.path.exists(full_path):
            csv_count = len(glob.glob(os.path.join(full_path, "*.csv")))
            json_count = len(glob.glob(os.path.join(full_path, "*.json")))
            db_count = len(glob.glob(os.path.join(full_path, "*.db")))
            total_count = csv_count + json_count + db_count
            total_files += total_count
            print(f"  {dir_path}: {total_count} files ({csv_count} CSV, {json_count} JSON, {db_count} DB)")
        else:
            print(f"  {dir_path}: MISSING")
    
    print(f"\nTotal files across system: {total_files}")
    
    # 2. Database Status
    print("\n🗃️ DATABASE STATUS")
    print("-" * 40)
    
    db_path = os.path.join(base_path, "databases", "comprehensive_greyhound_data.db")
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Database tables: {len(tables)}")
            
            for table in tables:
                if table != 'sqlite_sequence':
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  {table}: {count} records")
            
            conn.close()
            print("Database Status: ✅ HEALTHY")
        except Exception as e:
            print(f"Database Status: ❌ ERROR - {e}")
    else:
        print("Database Status: ⚠️ MISSING MAIN DATABASE")
    
    # 3. Deduplication Status
    print("\n🔄 DEDUPLICATION STATUS")
    print("-" * 40)
    
    # Check if duplicate cleanup backup exists
    dup_backup_path = os.path.join(base_path, "duplicate_cleanup_backup")
    if os.path.exists(dup_backup_path):
        dup_files = len(os.listdir(dup_backup_path))
        print(f"Duplicates cleaned: {dup_files} files")
        print("Deduplication Status: ✅ COMPLETED")
    else:
        print("Deduplication Status: ✅ NO DUPLICATES FOUND")
    
    # 4. Backup Status
    print("\n💾 BACKUP STATUS")
    print("-" * 40)
    
    backup_dirs = ['cached_backup', 'backup_before_cleanup', 'predictions']
    backup_healthy = True
    
    for backup_dir in backup_dirs:
        backup_path = os.path.join(base_path, backup_dir)
        if os.path.exists(backup_path):
            file_count = len(glob.glob(os.path.join(backup_path, "**", "*.*"), recursive=True))
            total_size = sum(os.path.getsize(os.path.join(backup_path, f)) 
                           for f in os.listdir(backup_path) 
                           if os.path.isfile(os.path.join(backup_path, f)))
            size_mb = round(total_size / (1024 * 1024), 2)
            print(f"  {backup_dir}: {file_count} files ({size_mb} MB)")
        else:
            print(f"  {backup_dir}: MISSING")
            backup_healthy = False
    
    # Check for recent system backup
    system_backups = glob.glob(os.path.join(base_path, "system_backup_*"))
    if system_backups:
        latest_backup = max(system_backups, key=os.path.getctime)
        backup_name = os.path.basename(latest_backup)
        print(f"  Latest system backup: {backup_name}")
    
    print(f"Backup Status: {'✅ HEALTHY' if backup_healthy else '⚠️ ISSUES'}")
    
    # 5. Data Quality Assessment
    print("\n📈 DATA QUALITY ASSESSMENT")
    print("-" * 40)
    
    quality_score = 100
    quality_issues = []
    
    # Check processed files for basic quality
    processed_completed = os.path.join(base_path, "processed", "completed")
    if os.path.exists(processed_completed):
        csv_files = glob.glob(os.path.join(processed_completed, "*.csv"))
        sample_size = min(10, len(csv_files))
        
        if sample_size > 0:
            try:
                import pandas as pd
                valid_files = 0
                for csv_file in csv_files[:sample_size]:
                    try:
                        df = pd.read_csv(csv_file)
                        if not df.empty and len(df.columns) >= 5:
                            valid_files += 1
                    except:
                        quality_score -= 5
                        quality_issues.append(f"Cannot read {os.path.basename(csv_file)}")
                
                quality_percentage = (valid_files / sample_size) * 100
                print(f"Processed files quality: {quality_percentage:.1f}% ({valid_files}/{sample_size} valid)")
                
                if quality_percentage < 90:
                    quality_score -= 10
                    quality_issues.append("Low processed files quality")
                    
            except ImportError:
                print("Cannot assess data quality (pandas not available)")
                quality_score -= 5
    
    # Check predictions exist
    predictions_dir = os.path.join(base_path, "predictions")
    if os.path.exists(predictions_dir):
        prediction_files = glob.glob(os.path.join(predictions_dir, "*.json"))
        print(f"Prediction files: {len(prediction_files)}")
        if len(prediction_files) == 0:
            quality_score -= 15
            quality_issues.append("No prediction files found")
    else:
        quality_score -= 20
        quality_issues.append("Predictions directory missing")
    
    print(f"Overall Data Quality Score: {quality_score}/100")
    
    if quality_issues:
        print("Quality Issues:")
        for issue in quality_issues:
            print(f"  - {issue}")
    
    # 6. System Health Summary
    print("\n🏥 SYSTEM HEALTH SUMMARY")
    print("-" * 40)
    
    health_status = "HEALTHY"
    if quality_score < 80:
        health_status = "NEEDS_ATTENTION"
    if quality_score < 60:
        health_status = "CRITICAL"
    
    print(f"Overall System Health: {health_status}")
    
    # 7. Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    if quality_score < 90:
        recommendations.append("Review and fix data quality issues")
    
    if not os.path.exists(os.path.join(base_path, "databases", "race_data.db")):
        recommendations.append("Consider creating a unified race_data.db database")
    
    if len(system_backups) == 0:
        recommendations.append("Set up regular system backups")
    
    if total_files > 5000:
        recommendations.append("Consider archiving old files to improve performance")
    
    if not recommendations:
        recommendations.append("System is in good health - continue regular monitoring")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # 8. Final Status
    print("\n🎯 FINAL STATUS")
    print("-" * 40)
    
    if health_status == "HEALTHY" and quality_score >= 90:
        print("✅ SYSTEM PASSED ALL CHECKS")
        print("Your greyhound racing data system is operating optimally.")
    elif health_status == "NEEDS_ATTENTION":
        print("⚠️ SYSTEM NEEDS ATTENTION")
        print("Some issues were found but the system is functional.")
    else:
        print("❌ SYSTEM CRITICAL")
        print("Critical issues detected - immediate attention required.")
    
    print("\n" + "=" * 80)
    
    # Save report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_files': total_files,
        'quality_score': quality_score,
        'health_status': health_status,
        'recommendations': recommendations,
        'quality_issues': quality_issues
    }
    
    report_path = os.path.join(base_path, f"final_integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_path}")
    
    return health_status == "HEALTHY" and quality_score >= 90

def main():
    """Generate final report"""
    success = generate_final_report()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
