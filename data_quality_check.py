#!/usr/bin/env python3
"""
Data Quality Check and Cleanup
==============================
This script performs comprehensive data quality checks and cleanup.
"""

import pandas as pd
import os
import re
from pathlib import Path
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_csv_integrity(file_path):
    """Check if CSV file is readable and has valid structure"""
    try:
        df = pd.read_csv(file_path)
        return {
            'readable': True,
            'rows': len(df),
            'columns': len(df.columns),
            'empty': df.empty,
            'has_data': len(df) > 0 and len(df.columns) > 0,
            'error': None
        }
    except Exception as e:
        return {
            'readable': False,
            'rows': 0,
            'columns': 0,
            'empty': True,
            'has_data': False,
            'error': str(e)
        }

def analyze_directory_structure():
    """Analyze the current directory structure"""
    logger.info("Analyzing directory structure...")
    
    structure = {}
    base_path = Path('.')
    
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            csv_files = list(item.glob('*.csv'))
            json_files = list(item.glob('*.json'))
            
            structure[item.name] = {
                'csv_count': len(csv_files),
                'json_count': len(json_files),
                'total_files': len(csv_files) + len(json_files)
            }
    
    return structure

def validate_race_data_files():
    """Validate race data files specifically"""
    logger.info("Validating race data files...")
    
    race_files = []
    for csv_file in Path('.').rglob('*.csv'):
        if 'backup' not in str(csv_file) and 'cleanup_archive' not in str(csv_file):
            if csv_file.name.startswith('Race_'):
                race_files.append(csv_file)
    
    validation_results = {}
    
    for file_path in race_files:
        result = check_csv_integrity(file_path)
        result['file_path'] = str(file_path)
        result['file_name'] = file_path.name
        
        # Check if filename follows expected pattern
        if re.match(r'Race_\d{2}_[A-Z_]+_\d{4}-\d{2}-\d{2}\.csv$', file_path.name):
            result['name_compliant'] = True
        else:
            result['name_compliant'] = False
            
        validation_results[str(file_path)] = result
    
    return validation_results

def clean_problematic_analysis_files():
    """Clean up analysis files with problematic suffixes"""
    logger.info("Cleaning problematic analysis files...")
    
    cleaned = 0
    for file_path in Path('.').rglob('Analysis_*.csv'):
        if 'backup' in str(file_path) or 'cleanup_archive' in str(file_path):
            continue
            
        # Check for problematic suffixes
        if re.search(r'_\d{2}_\d{2}', file_path.name):
            # Try to extract clean name
            match = re.match(r'(Analysis_\w+_[A-Z_]+_\d{4}-\d{2}-\d{2}_\d{6}).*\.csv', file_path.name)
            if match:
                clean_base = match.group(1)
                clean_name = f"{clean_base}.csv"
                clean_path = file_path.parent / clean_name
                
                if clean_path.exists() and clean_path != file_path:
                    # Remove duplicate
                    file_path.unlink()
                    logger.info(f"Removed duplicate: {file_path.name}")
                    cleaned += 1
                elif clean_path != file_path:
                    # Rename to clean name
                    file_path.rename(clean_path)
                    logger.info(f"Cleaned: {file_path.name} -> {clean_name}")
                    cleaned += 1
    
    return cleaned

def verify_data_consistency():
    """Verify data consistency across directories"""
    logger.info("Verifying data consistency...")
    
    issues = []
    
    # Check for duplicate files across directories
    all_files = {}
    for csv_file in Path('.').rglob('*.csv'):
        if 'backup' not in str(csv_file) and 'cleanup_archive' not in str(csv_file):
            if csv_file.name in all_files:
                issues.append({
                    'type': 'duplicate_name',
                    'file': csv_file.name,
                    'locations': [all_files[csv_file.name], str(csv_file)]
                })
            else:
                all_files[csv_file.name] = str(csv_file)
    
    # Check for files with invalid dates
    for file_name in all_files.keys():
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_name)
        if date_match:
            try:
                datetime.strptime(date_match.group(1), '%Y-%m-%d')
            except ValueError:
                issues.append({
                    'type': 'invalid_date',
                    'file': file_name,
                    'date': date_match.group(1)
                })
    
    return issues

def generate_quality_report():
    """Generate comprehensive quality report"""
    logger.info("Generating quality report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'directory_structure': analyze_directory_structure(),
        'race_file_validation': validate_race_data_files(),
        'consistency_issues': verify_data_consistency()
    }
    
    # Summary statistics
    race_validation = report['race_file_validation']
    total_race_files = len(race_validation)
    valid_files = sum(1 for r in race_validation.values() if r['readable'] and r['has_data'])
    compliant_names = sum(1 for r in race_validation.values() if r['name_compliant'])
    
    report['summary'] = {
        'total_race_files': total_race_files,
        'valid_files': valid_files,
        'compliant_names': compliant_names,
        'validity_rate': (valid_files / total_race_files * 100) if total_race_files > 0 else 100,
        'compliance_rate': (compliant_names / total_race_files * 100) if total_race_files > 0 else 100
    }
    
    return report

def main():
    logger.info("Starting comprehensive data quality check...")
    
    # Clean problematic files first
    cleaned_analysis = clean_problematic_analysis_files()
    logger.info(f"Cleaned {cleaned_analysis} analysis files")
    
    # Generate quality report
    report = generate_quality_report()
    
    # Save report
    report_file = f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA QUALITY REPORT SUMMARY")
    print("="*60)
    print(f"Total race files: {report['summary']['total_race_files']}")
    print(f"Valid files: {report['summary']['valid_files']}")
    print(f"Compliant names: {report['summary']['compliant_names']}")
    print(f"Validity rate: {report['summary']['validity_rate']:.1f}%")
    print(f"Compliance rate: {report['summary']['compliance_rate']:.1f}%")
    
    if report['consistency_issues']:
        print(f"\nConsistency issues found: {len(report['consistency_issues'])}")
        for issue in report['consistency_issues'][:5]:
            print(f"  - {issue['type']}: {issue['file']}")
    
    print(f"\nDetailed report saved to: {report_file}")
    print("="*60)
    
    logger.info("Data quality check completed")

if __name__ == "__main__":
    main()
