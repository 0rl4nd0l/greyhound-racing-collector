#!/usr/bin/env python3
"""
Fix NaN values in prediction JSON files by replacing them with null
"""

import os
import re
import json
import glob
from pathlib import Path

def fix_nan_in_file(file_path):
    """Fix NaN values in a single JSON file"""
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file contains NaN values
        if 'NaN' not in content:
            return False
        
        print(f"🔧 Fixing NaN values in: {file_path}")
        
        # Create backup
        backup_path = f"{file_path}.backup_{os.path.basename(__file__)}_{os.getpid()}"
        print(f"📋 Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Replace NaN with null (JSON standard)
        # Use regex to find standalone NaN values (not inside strings)
        fixed_content = re.sub(r':\s*NaN\b', ': null', content)
        
        # Validate that the fixed content is valid JSON
        try:
            json.loads(fixed_content)
            print("✅ JSON validation passed")
        except json.JSONDecodeError as e:
            print(f"❌ JSON validation failed: {e}")
            return False
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ Fixed NaN values in: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix NaN values in all prediction JSON files"""
    predictions_dir = "./predictions"
    
    if not os.path.exists(predictions_dir):
        print(f"❌ Predictions directory not found: {predictions_dir}")
        return
    
    # Find all JSON files (excluding backups)
    json_files = glob.glob(os.path.join(predictions_dir, "*.json"))
    json_files = [f for f in json_files if not f.endswith('.backup')]
    
    if not json_files:
        print("❌ No prediction JSON files found")
        return
    
    print(f"🔍 Found {len(json_files)} prediction files to check")
    
    fixed_count = 0
    for json_file in json_files:
        if fix_nan_in_file(json_file):
            fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files checked: {len(json_files)}")
    print(f"   Files fixed: {fixed_count}")
    print(f"   Files with no issues: {len(json_files) - fixed_count}")

if __name__ == "__main__":
    main()
