#!/usr/bin/env python3
"""
File Naming Validator
====================
This script validates that files comply with the established naming standards.
"""

import re
from pathlib import Path
import json

def load_standards():
    """Load naming standards from documentation"""
    standards_path = Path(__file__).parent / "naming_standards.json"
    if standards_path.exists():
        with open(standards_path, 'r') as f:
            return json.load(f)
    return {}

def validate_filename(filename, standards):
    """Validate a single filename against standards"""
    compliant_patterns = [
        r'Race_\d{2}_[A-Z_]+_\d{4}-\d{2}-\d{2}\.csv$',
        r'Race_\d{2}_UNKNOWN_\d{4}-\d{2}-\d{2}\.csv$',
        r'FormGuide_[A-Z_]+_\d{4}-\d{2}-\d{2}_\d{2}\.csv$',
        r'Analysis_\w+_[A-Z_]+_\d{4}-\d{2}-\d{2}_\d{6}\.(json|csv)$',
        r'Upcoming_[A-Z_]+_\d{4}-\d{2}-\d{2}_\d{2}\.csv$',
        r'Upcoming_UNKNOWN_\d{4}-\d{2}-\d{2}_\d{2}\.csv$'
    ]
    
    # Check for problematic suffixes that indicate naming conflicts
    if re.search(r'_\d{2}_\d{2}', filename) or re.search(r'_\d{2}_\d{2}_\d{2}', filename):
        return False, "Contains problematic naming conflict suffixes"
    
    for pattern in compliant_patterns:
        if re.match(pattern, filename):
            return True, "Compliant"
    
    return False, "Non-compliant filename"

def validate_directory(directory_path):
    """Validate all files in a directory"""
    base_path = Path(directory_path)
    standards = load_standards()
    
    results = {
        'compliant': [],
        'non_compliant': [],
        'total_files': 0
    }
    
    exclude_dirs = ['backup_before_cleanup', 'quarantine', 'file_naming_standards', 'cleanup_archive']
    
    for file_path in base_path.rglob("*.csv"):
        if any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
            continue
            
        results['total_files'] += 1
        is_compliant, message = validate_filename(file_path.name, standards)
        
        if is_compliant:
            results['compliant'].append(str(file_path))
        else:
            results['non_compliant'].append({
                'file': str(file_path),
                'reason': message
            })
    
    return results

if __name__ == "__main__":
    import sys
    
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    results = validate_directory(directory)
    
    print(f"\nFile Naming Validation Results")
    print("=" * 40)
    print(f"Total files checked: {results['total_files']}")
    print(f"Compliant files: {len(results['compliant'])}")
    print(f"Non-compliant files: {len(results['non_compliant'])}")
    
    if results['non_compliant']:
        print("\nNon-compliant files:")
        for item in results['non_compliant'][:10]:  # Show first 10
            print(f"  - {item['file']}: {item['reason']}")
        
        if len(results['non_compliant']) > 10:
            print(f"  ... and {len(results['non_compliant']) - 10} more")
    
    compliance_rate = len(results['compliant']) / results['total_files'] * 100 if results['total_files'] > 0 else 100
    print(f"\nCompliance rate: {compliance_rate:.1f}%")
