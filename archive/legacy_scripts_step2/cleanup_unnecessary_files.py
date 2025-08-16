#!/usr/bin/env python3
"""
File Cleanup Script for Greyhound Racing Project
===============================================

This script identifies and helps clean up unnecessary files while preserving
the active system components.

IMPORTANT: Review the file lists before running cleanup operations!
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

class ProjectCleaner:
    def __init__(self):
        self.base_dir = Path('.')
        self.cleanup_log = []
        
        # Files and directories that should be preserved (ACTIVE)
        self.active_files = {
            # Primary prediction scripts
            'weather_enhanced_predictor.py',
            'upcoming_race_predictor.py', 
            'comprehensive_enhanced_ml_system.py',
            
            # Web application
            'app.py',
            
            # Automation
            'automation_scheduler.py',
            'automation_control.sh',
            'com.greyhound.automation.plist',
            
            # Data processing & analysis
            'enhanced_race_analyzer.py',
            'sportsbet_odds_integrator.py',
            'venue_mapping_fix.py',
            'enhanced_data_integration.py',
            
            # Utilities
            'logger.py',
            'run.py',
            'initialize_database.py',
            'race_file_manager.py',
            
            # Data collection
            'form_guide_csv_scraper.py',
            'upcoming_race_browser.py',
            'weather_api_service.py',
            'weather_service_open_meteo.py',
            
            # Configuration
            'requirements.txt',
            'README.md',
            'ACTIVE_SCRIPTS_GUIDE.md',
            'AUTOMATION_GUIDE.md',
            
            # Main database
            'greyhound_racing_data.db',
        }
        
        # Directories that should be preserved
        self.active_directories = {
            'templates',
            'static', 
            'upcoming_races',
            'predictions',
            'comprehensive_model_results',
            'comprehensive_trained_models',
            'logs',
            'processed_races',
            'backups',
            'docs'
        }
        
        # File patterns that are candidates for cleanup
        self.cleanup_candidates = {
            'log_files': ['*.log', 'app_*.log', 'flask_*.log', 'server.log'],
            'debug_files': ['Race_01_UNKNOWN_*.json', 'debug_*.txt', '*.html'],
            'empty_databases': ['*.db'],  # Will check if actually empty
            'temp_files': ['*.tmp', '*.temp', '*~'],
            'old_scripts': ['*_old.py', '*_backup.py', '*_test.py']
        }
        
        # Directories that are candidates for cleanup
        self.cleanup_directories = {
            'archive',
            'archive_unused_scripts', 
            'cleanup_archive',
            'outdated_scripts',
            'ml_env',
            'venv',
            'backup_before_cleanup',
            'quarantine',
            'consolidated_data',
            'data',  # if mostly empty
            'integrated_form_data',
            'feature_analysis_results',
            'ml_backtesting_results',
            'automated_backtesting_results',
            'file_naming_standards',
        }

    def analyze_project(self):
        """Analyze the project structure and identify cleanup candidates."""
        print("🔍 Analyzing project structure...")
        
        analysis = {
            'active_files': [],
            'cleanup_candidates': {
                'log_files': [],
                'debug_files': [],
                'empty_databases': [],
                'temp_files': [],
                'old_scripts': [],
                'unknown_files': []
            },
            'cleanup_directories': [],
            'large_files': [],
            'empty_directories': []
        }
        
        # Analyze files
        for file_path in self.base_dir.glob('*'):
            if file_path.is_file():
                filename = file_path.name
                file_size = file_path.stat().st_size
                
                if filename in self.active_files:
                    analysis['active_files'].append({
                        'name': filename,
                        'size': self._format_size(file_size),
                        'status': '✅ ACTIVE'
                    })
                elif self._is_log_file(filename):
                    analysis['cleanup_candidates']['log_files'].append({
                        'name': filename,
                        'size': self._format_size(file_size),
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
                    })
                elif self._is_debug_file(filename):
                    analysis['cleanup_candidates']['debug_files'].append({
                        'name': filename,
                        'size': self._format_size(file_size),
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
                    })
                elif filename.endswith('.db') and filename not in self.active_files:
                    # Check if database is empty or nearly empty
                    if file_size < 100000:  # Less than 100KB
                        analysis['cleanup_candidates']['empty_databases'].append({
                            'name': filename,
                            'size': self._format_size(file_size),
                            'status': 'Likely empty/unused'
                        })
                elif file_size > 50 * 1024 * 1024:  # Files larger than 50MB
                    analysis['large_files'].append({
                        'name': filename,
                        'size': self._format_size(file_size),
                        'status': 'Large file - review needed'
                    })
                else:
                    # Unknown files that might need review
                    if not filename.startswith('.') and filename not in self.active_files:
                        analysis['cleanup_candidates']['unknown_files'].append({
                            'name': filename,
                            'size': self._format_size(file_size),
                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
                        })
        
        # Analyze directories
        for dir_path in self.base_dir.glob('*'):
            if dir_path.is_dir() and not dir_path.name.startswith('.'):
                if dir_path.name in self.cleanup_directories:
                    dir_size = self._get_directory_size(dir_path)
                    analysis['cleanup_directories'].append({
                        'name': dir_path.name,
                        'size': self._format_size(dir_size),
                        'status': '📁 Cleanup candidate'
                    })
                elif self._is_empty_directory(dir_path):
                    analysis['empty_directories'].append({
                        'name': dir_path.name,
                        'status': 'Empty directory'
                    })
        
        return analysis

    def _is_log_file(self, filename):
        """Check if file is a log file."""
        log_patterns = ['.log', 'app_', 'flask_', 'server.log']
        return any(pattern in filename.lower() for pattern in log_patterns)

    def _is_debug_file(self, filename):
        """Check if file is a debug/test file."""
        debug_patterns = ['Race_01_UNKNOWN', 'debug_', '.html', 'race_page_content']
        return any(pattern in filename for pattern in debug_patterns)

    def _get_directory_size(self, directory):
        """Get total size of directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, PermissionError):
            pass
        return total_size

    def _is_empty_directory(self, directory):
        """Check if directory is empty or contains only hidden files."""
        try:
            contents = list(directory.iterdir())
            return len(contents) == 0 or all(item.name.startswith('.') for item in contents)
        except (OSError, PermissionError):
            return False

    def _format_size(self, size_bytes):
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def generate_cleanup_report(self):
        """Generate a detailed cleanup report."""
        analysis = self.analyze_project()
        
        report = f"""
# PROJECT CLEANUP ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 SUMMARY

### Active Files (Preserved): {len(analysis['active_files'])}
### Cleanup Candidates: {sum(len(category) for category in analysis['cleanup_candidates'].values())}
### Cleanup Directories: {len(analysis['cleanup_directories'])}

---

## ✅ ACTIVE FILES (Will be preserved)
"""
        
        for file_info in analysis['active_files']:
            report += f"- {file_info['name']} ({file_info['size']}) {file_info['status']}\n"
        
        report += "\n---\n\n## 🗑️ CLEANUP CANDIDATES\n\n"
        
        if analysis['cleanup_candidates']['log_files']:
            report += "### Log Files:\n"
            for file_info in analysis['cleanup_candidates']['log_files']:
                report += f"- {file_info['name']} ({file_info['size']}) - Modified: {file_info['modified']}\n"
            report += "\n"
        
        if analysis['cleanup_candidates']['debug_files']:
            report += "### Debug/Test Files:\n"
            for file_info in analysis['cleanup_candidates']['debug_files']:
                report += f"- {file_info['name']} ({file_info['size']}) - Modified: {file_info['modified']}\n"
            report += "\n"
        
        if analysis['cleanup_candidates']['empty_databases']:
            report += "### Empty/Unused Databases:\n"
            for file_info in analysis['cleanup_candidates']['empty_databases']:
                report += f"- {file_info['name']} ({file_info['size']}) - {file_info['status']}\n"
            report += "\n"
        
        if analysis['cleanup_candidates']['unknown_files']:
            report += "### Unknown Files (Need Review):\n"
            for file_info in analysis['cleanup_candidates']['unknown_files']:
                report += f"- {file_info['name']} ({file_info['size']}) - Modified: {file_info['modified']}\n"
            report += "\n"
        
        if analysis['cleanup_directories']:
            report += "### Cleanup Directories:\n"
            for dir_info in analysis['cleanup_directories']:
                report += f"- {dir_info['name']} ({dir_info['size']}) {dir_info['status']}\n"
            report += "\n"
        
        if analysis['large_files']:
            report += "### Large Files (Review Needed):\n"
            for file_info in analysis['large_files']:
                report += f"- {file_info['name']} ({file_info['size']}) - {file_info['status']}\n"
            report += "\n"
        
        if analysis['empty_directories']:
            report += "### Empty Directories:\n"
            for dir_info in analysis['empty_directories']:
                report += f"- {dir_info['name']} - {dir_info['status']}\n"
            report += "\n"
        
        report += """
---

## 🚨 CLEANUP RECOMMENDATIONS

### SAFE TO DELETE:
1. **Log files** - Old logs can be safely removed
2. **Debug files** - Race_01_UNKNOWN_*.json and similar test files
3. **Empty databases** - Unused .db files under 100KB
4. **Archive directories** - Already archived content
5. **Virtual environments** - ml_env/, venv/ (can be recreated)

### REVIEW BEFORE DELETING:
1. **Large files** - Check if they contain important data
2. **Unknown files** - Verify they're not needed
3. **HTML files** - Might be test data or cached content

### KEEP (ACTIVE SYSTEM):
1. **All files listed in Active Files section**
2. **Main database** - greyhound_racing_data.db
3. **Prediction results** - predictions/ directory
4. **Web app assets** - templates/, static/
5. **Configuration files** - requirements.txt, *.md files

---

## 🔧 SUGGESTED CLEANUP COMMANDS

```bash
# Review this report first, then run individual cleanup commands:

# Remove log files (SAFE)
rm -f *.log app_*.log flask_*.log server.log

# Remove debug files (SAFE)
rm -f Race_01_UNKNOWN_*.json debug_*.txt *.html

# Remove empty databases (REVIEW FIRST)
# rm -f enhanced_greyhound_racing.db greyhound_data.db greyhound_racing.db

# Remove archive directories (SAFE if already archived)
# rm -rf archive/ archive_unused_scripts/ cleanup_archive/

# Remove virtual environments (SAFE - can recreate)
# rm -rf ml_env/ venv/
```

---

**IMPORTANT**: Always backup important data before running cleanup operations!
"""
        
        return report

    def save_report(self, filename="CLEANUP_ANALYSIS_REPORT.md"):
        """Save the cleanup report to a file."""
        report = self.generate_cleanup_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"📋 Cleanup report saved to: {filename}")
        return filename

if __name__ == "__main__":
    print("🧹 PROJECT CLEANUP ANALYZER")
    print("=" * 40)
    
    cleaner = ProjectCleaner()
    
    # Generate and save cleanup report
    report_file = cleaner.save_report()
    
    print(f"\n✅ Analysis complete!")
    print(f"📋 Review the report: {report_file}")
    print("\n🚨 IMPORTANT: Review the report carefully before deleting any files!")
    print("\n💡 The report contains specific cleanup commands you can run safely.")
