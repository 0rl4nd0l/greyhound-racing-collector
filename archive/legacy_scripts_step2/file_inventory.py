#!/usr/bin/env python3
"""
File Inventory Tool
==================
Provides exact, consistent file counts and detailed inventory.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

class FileInventory:
    def __init__(self):
        self.base_path = Path('.')
        self.exclude_dirs = [
            'backup_before_cleanup', 
            'cleanup_archive', 
            'quarantine', 
            '__pycache__', 
            '.git', 
            'venv', 
            'ml_env'
        ]
        
    def scan_files(self):
        """Scan all data files and return comprehensive inventory"""
        inventory = {
            'csv_files': [],
            'json_files': [],
            'summary': {},
            'categories': {},
            'directories': {}
        }
        
        for file_path in self.base_path.rglob('*'):
            if file_path.is_file() and not self._is_excluded(file_path):
                if file_path.suffix.lower() in ['.csv', '.json']:
                    file_info = self._get_file_info(file_path)
                    
                    if file_path.suffix.lower() == '.csv':
                        inventory['csv_files'].append(file_info)
                    else:
                        inventory['json_files'].append(file_info)
        
        # Generate summary statistics
        inventory['summary'] = self._generate_summary(inventory)
        inventory['categories'] = self._categorize_files(inventory)
        inventory['directories'] = self._directory_breakdown(inventory)
        
        return inventory
    
    def _is_excluded(self, file_path):
        """Check if file should be excluded"""
        path_str = str(file_path)
        return any(exclude_dir in path_str for exclude_dir in self.exclude_dirs)
    
    def _get_file_info(self, file_path):
        """Extract detailed information about a file"""
        try:
            stat = file_path.stat()
            return {
                'name': file_path.name,
                'path': str(file_path),
                'directory': str(file_path.parent),
                'type': file_path.suffix[1:].upper(),
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 3),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'category': self._categorize_file(file_path)
            }
        except Exception as e:
            return None
    
    def _categorize_file(self, file_path):
        """Categorize file by purpose"""
        name = file_path.name.lower()
        parent = file_path.parent.name.lower()
        
        if name.startswith('race_'):
            return 'Race Data'
        elif name.startswith('analysis_'):
            return 'ML Analysis'
        elif 'prediction' in parent or 'prediction' in name:
            return 'Predictions'
        elif 'enhanced' in parent or 'expert' in parent:
            return 'Enhanced Data'
        elif 'backtest' in parent or 'backtest' in name:
            return 'Backtesting'
        elif 'model' in parent or 'model' in name:
            return 'Models'
        elif 'form' in parent or 'form' in name:
            return 'Form Guides'
        elif name.startswith('upcoming_'):
            return 'Upcoming Races'
        else:
            return 'Other'
    
    def _generate_summary(self, inventory):
        """Generate summary statistics"""
        csv_count = len(inventory['csv_files'])
        json_count = len(inventory['json_files'])
        total_count = csv_count + json_count
        
        total_size = sum(f['size_mb'] for f in inventory['csv_files'] + inventory['json_files'])
        
        return {
            'total_files': total_count,
            'csv_files': csv_count,
            'json_files': json_count,
            'total_size_mb': round(total_size, 2),
            'scan_timestamp': datetime.now().isoformat()
        }
    
    def _categorize_files(self, inventory):
        """Breakdown by category"""
        categories = {}
        all_files = inventory['csv_files'] + inventory['json_files']
        
        for file_info in all_files:
            category = file_info['category']
            if category not in categories:
                categories[category] = {
                    'count': 0,
                    'csv_count': 0,
                    'json_count': 0,
                    'total_size_mb': 0
                }
            
            categories[category]['count'] += 1
            categories[category]['total_size_mb'] += file_info['size_mb']
            
            if file_info['type'] == 'CSV':
                categories[category]['csv_count'] += 1
            else:
                categories[category]['json_count'] += 1
        
        # Sort by count
        return dict(sorted(categories.items(), key=lambda x: x[1]['count'], reverse=True))
    
    def _directory_breakdown(self, inventory):
        """Breakdown by directory"""
        directories = {}
        all_files = inventory['csv_files'] + inventory['json_files']
        
        for file_info in all_files:
            directory = file_info['directory']
            if directory not in directories:
                directories[directory] = {
                    'count': 0,
                    'csv_count': 0,
                    'json_count': 0,
                    'total_size_mb': 0
                }
            
            directories[directory]['count'] += 1
            directories[directory]['total_size_mb'] += file_info['size_mb']
            
            if file_info['type'] == 'CSV':
                directories[directory]['csv_count'] += 1
            else:
                directories[directory]['json_count'] += 1
        
        # Sort by count
        return dict(sorted(directories.items(), key=lambda x: x[1]['count'], reverse=True))
    
    def print_summary(self, inventory):
        """Print comprehensive summary"""
        print("=" * 60)
        print("GREYHOUND RACING DATA INVENTORY")
        print("=" * 60)
        
        summary = inventory['summary']
        print(f"Scan Date: {summary['scan_timestamp']}")
        print(f"Total Files: {summary['total_files']:,}")
        print(f"CSV Files: {summary['csv_files']:,}")
        print(f"JSON Files: {summary['json_files']:,}")
        print(f"Total Size: {summary['total_size_mb']:.1f} MB")
        
        print("\n" + "=" * 60)
        print("BREAKDOWN BY CATEGORY")
        print("=" * 60)
        
        for category, stats in inventory['categories'].items():
            print(f"{category:.<20} {stats['count']:>6,} files ({stats['csv_count']:>4} CSV, {stats['json_count']:>4} JSON) - {stats['total_size_mb']:>6.1f} MB")
        
        print("\n" + "=" * 60)
        print("TOP 10 DIRECTORIES BY FILE COUNT")
        print("=" * 60)
        
        for i, (directory, stats) in enumerate(list(inventory['directories'].items())[:10]):
            dir_name = directory.replace('./', '') or '(root)'
            print(f"{i+1:>2}. {dir_name:.<35} {stats['count']:>6,} files - {stats['total_size_mb']:>6.1f} MB")
        
        print("\n" + "=" * 60)
        print("RACE DATA BREAKDOWN")
        print("=" * 60)
        
        race_categories = ['Race Data', 'Upcoming Races', 'Form Guides']
        race_total = 0
        
        for category in race_categories:
            if category in inventory['categories']:
                count = inventory['categories'][category]['count']
                race_total += count
                print(f"{category:.<20} {count:>6,} files")
        
        print(f"{'Total Race Files':.<20} {race_total:>6,} files")
        
        print("\n" + "=" * 60)
        print("ML & ANALYSIS BREAKDOWN")
        print("=" * 60)
        
        ml_categories = ['ML Analysis', 'Predictions', 'Enhanced Data', 'Backtesting', 'Models']
        ml_total = 0
        
        for category in ml_categories:
            if category in inventory['categories']:
                count = inventory['categories'][category]['count']
                ml_total += count
                print(f"{category:.<20} {count:>6,} files")
        
        print(f"{'Total ML Files':.<20} {ml_total:>6,} files")
    
    def export_detailed_report(self, inventory, filename=None):
        """Export detailed report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"file_inventory_detailed_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(inventory, f, indent=2, default=str)
        
        print(f"\nDetailed report exported to: {filename}")
        return filename

def main():
    print("Scanning files...")
    
    inventory_tool = FileInventory()
    inventory = inventory_tool.scan_files()
    
    # Print summary
    inventory_tool.print_summary(inventory)
    
    # Export detailed report
    inventory_tool.export_detailed_report(inventory)
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMMANDS")
    print("=" * 60)
    print("To verify these counts manually:")
    print("CSV files: find . -name '*.csv' -not -path './backup_before_cleanup/*' -not -path './cleanup_archive/*' | wc -l")
    print("JSON files: find . -name '*.json' -not -path './backup_before_cleanup/*' -not -path './cleanup_archive/*' | wc -l")
    print("Race files: find . -name 'Race_*.csv' -not -path './backup_before_cleanup/*' -not -path './cleanup_archive/*' | wc -l")

if __name__ == "__main__":
    main()
