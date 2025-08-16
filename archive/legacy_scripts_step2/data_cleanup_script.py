#!/usr/bin/env python3
"""
Data Cleanup Script
==================
This script removes duplicate files, fixes corrupted data, and optimizes data storage
based on the findings from the data integrity check.
"""

import os
import json
import shutil
import sqlite3
from pathlib import Path
import logging
import pandas as pd
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleanup:
    def __init__(self, base_path, report_path):
        self.base_path = Path(base_path)
        self.report_path = Path(report_path)
        self.backup_dir = self.base_path / "backup_before_cleanup"
        self.cleaned_stats = {
            'files_removed': 0,
            'files_fixed': 0,
            'storage_freed': 0,
            'backup_created': False
        }
        
        # Load the integrity report
        with open(self.report_path, 'r') as f:
            self.report = json.load(f)
    
    def create_backup(self):
        """Create backup of important directories before cleanup"""
        logger.info("Creating backup before cleanup...")
        
        important_dirs = [
            'organized_csvs',
            'processed',
            'enhanced_analysis',
            'race_data.db'
        ]
        
        if not self.backup_dir.exists():
            self.backup_dir.mkdir()
        
        for dir_name in important_dirs:
            source = self.base_path / dir_name
            if source.exists():
                if source.is_file():
                    shutil.copy2(source, self.backup_dir / dir_name)
                else:
                    shutil.copytree(source, self.backup_dir / dir_name, dirs_exist_ok=True)
                logger.info(f"Backed up {dir_name}")
        
        self.cleaned_stats['backup_created'] = True
    
    def remove_duplicate_files(self):
        """Remove duplicate files identified in the report"""
        logger.info("Removing duplicate files...")
        
        duplicate_files = self.report.get('duplicate_files', [])
        
        for file_path in duplicate_files:
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                self.cleaned_stats['files_removed'] += 1
                self.cleaned_stats['storage_freed'] += file_size
                logger.debug(f"Removed duplicate: {file_path}")
            except FileNotFoundError:
                logger.warning(f"File already removed: {file_path}")
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
    
    def remove_empty_files(self):
        """Remove empty files"""
        logger.info("Removing empty files...")
        
        empty_files = self.report.get('empty_files', [])
        
        for file_path in empty_files:
            try:
                os.remove(file_path)
                self.cleaned_stats['files_removed'] += 1
                logger.debug(f"Removed empty file: {file_path}")
            except FileNotFoundError:
                logger.warning(f"File already removed: {file_path}")
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
    
    def fix_corrupted_json(self, file_path):
        """Attempt to fix corrupted JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Common JSON fixes
            # Fix trailing commas
            content = content.replace(',}', '}').replace(',]', ']')
            
            # Fix unquoted keys (simple cases)
            import re
            content = re.sub(r'(\w+):', r'"\1":', content)
            
            # Try to parse the fixed content
            json.loads(content)
            
            # If successful, write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Fixed corrupted JSON: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Could not fix JSON {file_path}: {e}")
            return False
    
    def handle_corrupted_files(self):
        """Handle corrupted files by fixing or removing them"""
        logger.info("Handling corrupted files...")
        
        corrupted_files = self.report.get('corrupted_files', [])
        
        for corrupt_info in corrupted_files:
            file_path = corrupt_info['file']
            file_type = corrupt_info['type']
            error = corrupt_info['error']
            
            try:
                if file_type == 'json' and 'JSON decode error' in error:
                    if self.fix_corrupted_json(file_path):
                        self.cleaned_stats['files_fixed'] += 1
                    else:
                        # If can't fix, move to quarantine
                        quarantine_dir = self.base_path / "quarantine"
                        quarantine_dir.mkdir(exist_ok=True)
                        shutil.move(file_path, quarantine_dir / Path(file_path).name)
                        logger.info(f"Moved corrupted file to quarantine: {file_path}")
                
                elif 'Empty' in error:
                    # Remove empty files
                    os.remove(file_path)
                    self.cleaned_stats['files_removed'] += 1
                    logger.info(f"Removed empty corrupted file: {file_path}")
                
            except Exception as e:
                logger.error(f"Error handling corrupted file {file_path}: {e}")
    
    def clean_systematic_duplicates(self):
        """Clean up systematic duplicates (numbered files)"""
        logger.info("Cleaning systematic duplicates...")
        
        systematic_duplicates = self.report.get('systematic_duplicates', {})
        
        for base_name, file_list in systematic_duplicates.items():
            if len(file_list) <= 1:
                continue
            
            # Keep the file without suffix or the lowest numbered one
            files_by_suffix = {}
            for file_path in file_list:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    # Extract suffix number or use 0 for files without suffix
                    name = file_path_obj.stem
                    if '_' in name and name.split('_')[-1].isdigit():
                        suffix_num = int(name.split('_')[-1])
                    else:
                        suffix_num = 0
                    files_by_suffix[suffix_num] = file_path
            
            # Keep the one with the lowest suffix (0 = no suffix)
            if files_by_suffix:
                keep_file = files_by_suffix[min(files_by_suffix.keys())]
                
                for suffix_num, file_path in files_by_suffix.items():
                    if file_path != keep_file:
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            self.cleaned_stats['files_removed'] += 1
                            self.cleaned_stats['storage_freed'] += file_size
                            logger.debug(f"Removed systematic duplicate: {file_path}")
                        except Exception as e:
                            logger.error(f"Error removing {file_path}: {e}")
    
    def optimize_directory_structure(self):
        """Optimize directory structure by consolidating scattered files"""
        logger.info("Optimizing directory structure...")
        
        # Move all processed CSV files to a single organized location
        processed_dir = self.base_path / "consolidated_data"
        processed_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (processed_dir / "race_data").mkdir(exist_ok=True)
        (processed_dir / "form_guides").mkdir(exist_ok=True)
        (processed_dir / "enhanced_analysis").mkdir(exist_ok=True)
        
        # Move files from various scattered locations
        source_dirs = [
            self.base_path / "organized_csvs" / "race_data",
            self.base_path / "processed" / "form_guides",
            self.base_path / "enhanced_analysis"
        ]
        
        for source_dir in source_dirs:
            if source_dir.exists():
                for file_path in source_dir.rglob("*.csv"):
                    try:
                        # Determine destination based on content/naming
                        if "form_guide" in file_path.name.lower():
                            dest = processed_dir / "form_guides" / file_path.name
                        elif "enhanced" in file_path.name.lower():
                            dest = processed_dir / "enhanced_analysis" / file_path.name
                        else:
                            dest = processed_dir / "race_data" / file_path.name
                        
                        # Avoid overwriting existing files
                        counter = 1
                        original_dest = dest
                        while dest.exists():
                            dest = original_dest.parent / f"{original_dest.stem}_{counter}{original_dest.suffix}"
                            counter += 1
                        
                        shutil.move(str(file_path), str(dest))
                        logger.debug(f"Moved {file_path} to {dest}")
                        
                    except Exception as e:
                        logger.error(f"Error moving {file_path}: {e}")
    
    def cleanup_empty_directories(self):
        """Remove empty directories after cleanup"""
        logger.info("Cleaning up empty directories...")
        
        for root, dirs, files in os.walk(self.base_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Directory is empty
                        os.rmdir(dir_path)
                        logger.debug(f"Removed empty directory: {dir_path}")
                except OSError:
                    pass  # Directory not empty or other error
    
    def generate_cleanup_report(self):
        """Generate a report of cleanup actions taken"""
        logger.info("Generating cleanup report...")
        
        cleanup_report = {
            'cleanup_timestamp': pd.Timestamp.now().isoformat(),
            'actions_taken': {
                'files_removed': self.cleaned_stats['files_removed'],
                'files_fixed': self.cleaned_stats['files_fixed'],
                'storage_freed_mb': round(self.cleaned_stats['storage_freed'] / (1024*1024), 2),
                'backup_created': self.cleaned_stats['backup_created']
            },
            'original_integrity_issues': {
                'duplicate_files': len(self.report.get('duplicate_files', [])),
                'corrupted_files': len(self.report.get('corrupted_files', [])),
                'empty_files': len(self.report.get('empty_files', [])),
                'systematic_duplicates': len(self.report.get('systematic_duplicates', {}))
            },
            'recommendations': [
                "Monitor for new duplicate files during data collection",
                "Implement file naming conventions to prevent systematic duplicates",
                "Add validation to data collection processes",
                "Consider implementing automated cleanup procedures",
                "Regular integrity checks should be performed"
            ]
        }
        
        report_path = self.base_path / "cleanup_report.json"
        with open(report_path, 'w') as f:
            json.dump(cleanup_report, f, indent=2)
        
        logger.info(f"Cleanup report saved to {report_path}")
        return cleanup_report
    
    def run_full_cleanup(self, create_backup=True):
        """Run the complete cleanup process"""
        logger.info("Starting comprehensive data cleanup...")
        
        if create_backup:
            self.create_backup()
        
        # Cleanup operations in order of safety
        self.remove_empty_files()
        self.handle_corrupted_files()
        self.remove_duplicate_files()
        self.clean_systematic_duplicates()
        self.optimize_directory_structure()
        self.cleanup_empty_directories()
        
        cleanup_report = self.generate_cleanup_report()
        
        logger.info("Data cleanup completed successfully!")
        return cleanup_report

def print_cleanup_summary(cleanup_report):
    """Print a summary of cleanup actions"""
    print("\n" + "="*60)
    print("DATA CLEANUP SUMMARY")
    print("="*60)
    
    actions = cleanup_report['actions_taken']
    print(f"Files removed: {actions['files_removed']:,}")
    print(f"Files fixed: {actions['files_fixed']:,}")
    print(f"Storage freed: {actions['storage_freed_mb']:.2f} MB")
    print(f"Backup created: {'Yes' if actions['backup_created'] else 'No'}")
    
    original = cleanup_report['original_integrity_issues']
    print(f"\nOriginal Issues Addressed:")
    print(f"  - Duplicate files: {original['duplicate_files']:,}")
    print(f"  - Corrupted files: {original['corrupted_files']:,}")
    print(f"  - Empty files: {original['empty_files']:,}")
    print(f"  - Systematic duplicates: {original['systematic_duplicates']:,}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(cleanup_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    base_path = "/Users/orlandolee/greyhound_racing_collector"
    report_path = f"{base_path}/data_integrity_report.json"
    
    try:
        cleaner = DataCleanup(base_path, report_path)
        cleanup_report = cleaner.run_full_cleanup(create_backup=True)
        print_cleanup_summary(cleanup_report)
        
        print(f"\nCleanup completed! Check {base_path}/cleanup_report.json for details.")
        print(f"Backup created in {base_path}/backup_before_cleanup/")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
