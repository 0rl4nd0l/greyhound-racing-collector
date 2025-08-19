#!/usr/bin/env python3
"""
File Naming Standards Implementation
===================================
This script implements and enforces consistent file naming standards
to prevent future duplicate file issues.
"""

import os
import re
import shutil
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileNamingStandardizer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.standards = {
            'race_data': {
                'pattern': 'Race_{race_num}_{venue}_{date}.csv',
                'example': 'Race_01_AP_K_2025-07-26.csv',
                'description': 'Race data files with standardized format'
            },
            'form_guides': {
                'pattern': 'FormGuide_{venue}_{date}_{race_num}.csv',
                'example': 'FormGuide_BAL_2025-07-26_01.csv',
                'description': 'Form guide files with venue and date'
            },
            'enhanced_analysis': {
                'pattern': 'Analysis_{type}_{venue}_{date}_{timestamp}.json',
                'example': 'Analysis_ML_AP_K_2025-07-26_143022.json',
                'description': 'Enhanced analysis with type and timestamp'
            },
            'upcoming_races': {
                'pattern': 'Upcoming_{venue}_{date}_{race_num}.csv',
                'example': 'Upcoming_GEE_2025-07-27_05.csv',
                'description': 'Upcoming race predictions'
            }
        }
        self.rename_stats = {
            'files_renamed': 0,
            'files_skipped': 0,
            'errors': []
        }
        
        # Create standards directory
        self.standards_dir = self.base_path / "file_naming_standards"
        self.standards_dir.mkdir(exist_ok=True)
    
    def create_naming_documentation(self):
        """Create comprehensive documentation for naming standards"""
        logger.info("Creating file naming standards documentation...")
        
        docs = {
            'title': 'Greyhound Racing Data File Naming Standards',
            'version': '1.0',
            'created_date': datetime.now().isoformat(),
            'overview': 'Standardized naming conventions to prevent duplicates and improve organization',
            'standards': self.standards,
            'rules': {
                'general': [
                    'Use underscores (_) as separators, not spaces or hyphens',
                    'Use YYYY-MM-DD format for dates',
                    'Use uppercase for venue codes (AP_K, BAL, GEE, etc.)',
                    'Use zero-padded numbers for race numbers (01, 02, etc.)',
                    'Include timestamp for files that might be generated multiple times',
                    'No special characters except underscores and hyphens in dates'
                ],
                'venues': [
                    'Use standard venue codes: AP_K, APWE, BAL, BEN, CANN, CASO',
                    'DAPT, GEE, GOSF, GRDN, HEA, HOR, MAND, MOUNT, MURR',
                    'NOR, QOT, RICH, SAL, SAN, TRA, WAR, W_PK'
                ],
                'dates': [
                    'Always use YYYY-MM-DD format',
                    'Future dates for upcoming races, past dates for historical data'
                ],
                'file_types': [
                    'Use .csv for tabular race data',
                    'Use .json for analysis results and metadata',
                    'Use .txt for logs and configuration files'
                ]
            },
            'examples': {
                'good': [
                    'Race_01_AP_K_2025-07-26.csv',
                    'FormGuide_BAL_2025-07-26_01.csv',
                    'Analysis_ML_GEE_2025-07-26_143022.json',
                    'Upcoming_RICH_2025-07-27_08.csv'
                ],
                'bad': [
                    'Race 1 - AP_K - 26 July 2025.csv',
                    'race1-apk-26-7-25.csv',
                    'Race_1_AP_K_26_July_2025_1.csv',
                    'form guide bal 26-07-2025.csv'
                ]
            },
            'validation_script': 'file_naming_validator.py',
            'enforcement': 'Automated validation runs on data collection'
        }
        
        # Save documentation
        docs_path = self.standards_dir / "naming_standards.json"
        with open(docs_path, 'w') as f:
            json.dump(docs, f, indent=2)
        
        # Create markdown version for easy reading
        md_content = self.create_markdown_documentation(docs)
        md_path = self.standards_dir / "NAMING_STANDARDS.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Documentation created at {docs_path} and {md_path}")
    
    def create_markdown_documentation(self, docs):
        """Create markdown version of documentation"""
        md = f"""# {docs['title']}

**Version:** {docs['version']}  
**Created:** {docs['created_date'][:10]}

## Overview
{docs['overview']}

## File Naming Standards

"""
        
        for category, info in docs['standards'].items():
            md += f"### {category.replace('_', ' ').title()}\n"
            md += f"- **Pattern:** `{info['pattern']}`\n"
            md += f"- **Example:** `{info['example']}`\n"
            md += f"- **Description:** {info['description']}\n\n"
        
        md += "## Naming Rules\n\n"
        
        for rule_category, rules in docs['rules'].items():
            md += f"### {rule_category.replace('_', ' ').title()}\n"
            for rule in rules:
                md += f"- {rule}\n"
            md += "\n"
        
        md += "## Examples\n\n"
        md += "### ✅ Good Examples\n"
        for example in docs['examples']['good']:
            md += f"- `{example}`\n"
        
        md += "\n### ❌ Bad Examples\n"
        for example in docs['examples']['bad']:
            md += f"- `{example}`\n"
        
        md += f"""
## Validation
Run `python {docs['validation_script']}` to validate existing files against these standards.

## Enforcement
{docs['enforcement']}
"""
        
        return md
    
    def extract_file_components(self, filename):
        """Extract components from filename using various patterns"""
        components = {
            'race_number': None,
            'venue': None,
            'date': None,
            'file_type': None,
            'original_name': filename
        }
        
        # Extract race number
        race_patterns = [r'Race\s*(\d+)', r'Race_(\d+)', r'race[\s_-]*(\d+)']
        for pattern in race_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                components['race_number'] = int(match.group(1))
                break
        
        # Extract venue
        venue_patterns = [
            'AP_K', 'APWE', 'BAL', 'BEN', 'CANN', 'CASO', 'DAPT', 'GEE',
            'GOSF', 'GRDN', 'HEA', 'HOR', 'LADBROKES', 'MAND', 'MOUNT',
            'MURR', 'NOR', 'QOT', 'RICH', 'SAL', 'SAN', 'TRA', 'WAR', 'W_PK'
        ]
        
        filename_upper = filename.upper()
        for venue in venue_patterns:
            if venue in filename_upper:
                components['venue'] = venue
                break
        
        # Extract date
        date_patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
            r'(\d{2})/(\d{2})/(\d{4})',  # DD/MM/YYYY
            r'(\d{4})(\d{2})(\d{2})'     # YYYYMMDD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    if 'January' in pattern:  # Month name format
                        day, month_name, year = match.groups()
                        month_names = {
                            'january': 1, 'february': 2, 'march': 3, 'april': 4,
                            'may': 5, 'june': 6, 'july': 7, 'august': 8,
                            'september': 9, 'october': 10, 'november': 11, 'december': 12
                        }
                        month = month_names.get(month_name.lower(), 1)
                        components['date'] = f"{year}-{month:02d}-{int(day):02d}"
                    elif '-' in pattern or len(match.groups()) == 3:
                        if 'YYYY' in pattern and pattern.index('YYYY') == 1:  # YYYY-MM-DD
                            year, month, day = match.groups()
                            components['date'] = f"{year}-{month}-{day}"
                        elif len(match.groups()[0]) == 4:  # YYYYMMDD
                            year, month, day = match.groups()
                            components['date'] = f"{year}-{month}-{day}"
                        else:  # DD/MM/YYYY
                            day, month, year = match.groups()
                            components['date'] = f"{year}-{month:0>2}-{day:0>2}"
                    break
                except:
                    continue
        
        # Determine file type based on path and content
        if 'form_guide' in filename.lower() or 'formguide' in filename.lower():
            components['file_type'] = 'form_guides'
        elif 'upcoming' in filename.lower():
            components['file_type'] = 'upcoming_races'
        elif 'analysis' in filename.lower() or 'enhanced' in filename.lower():
            components['file_type'] = 'enhanced_analysis'
        else:
            components['file_type'] = 'race_data'
        
        return components
    
    def generate_standard_name(self, components, file_extension):
        """Generate standardized filename from components"""
        if not components['venue']:
            components['venue'] = 'UNKNOWN'
        
        if not components['date']:
            components['date'] = datetime.now().strftime('%Y-%m-%d')
        
        race_num = f"{components['race_number']:02d}" if components['race_number'] else "01"
        
        file_type = components['file_type']
        pattern = self.standards[file_type]['pattern']
        
        # Replace placeholders in pattern
        if file_type == 'race_data':
            new_name = f"Race_{race_num}_{components['venue']}_{components['date']}{file_extension}"
        elif file_type == 'form_guides':
            new_name = f"FormGuide_{components['venue']}_{components['date']}_{race_num}{file_extension}"
        elif file_type == 'enhanced_analysis':
            timestamp = datetime.now().strftime('%H%M%S')
            new_name = f"Analysis_ML_{components['venue']}_{components['date']}_{timestamp}{file_extension}"
        elif file_type == 'upcoming_races':
            new_name = f"Upcoming_{components['venue']}_{components['date']}_{race_num}{file_extension}"
        else:
            # Fallback to race_data pattern
            new_name = f"Race_{race_num}_{components['venue']}_{components['date']}{file_extension}"
        
        return new_name
    
    def rename_file_if_needed(self, file_path):
        """Rename file to comply with standards if needed"""
        try:
            file_obj = Path(file_path)
            current_name = file_obj.name
            
            # Skip already standardized files
            if self.is_filename_compliant(current_name):
                self.rename_stats['files_skipped'] += 1
                return file_path
            
            # Extract components
            components = self.extract_file_components(current_name)
            
            # Generate new name
            new_name = self.generate_standard_name(components, file_obj.suffix)
            
            # Avoid name conflicts
            new_path = file_obj.parent / new_name
            counter = 1
            while new_path.exists() and new_path != file_obj:
                name_parts = new_name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_{counter:02d}.{name_parts[1]}"
                else:
                    new_name = f"{new_name}_{counter:02d}"
                new_path = file_obj.parent / new_name
                counter += 1
            
            # Rename file
            if new_path != file_obj:
                file_obj.rename(new_path)
                logger.info(f"Renamed: {current_name} → {new_name}")
                self.rename_stats['files_renamed'] += 1
                return str(new_path)
            else:
                self.rename_stats['files_skipped'] += 1
                return file_path
                
        except Exception as e:
            error_msg = f"Error renaming {file_path}: {e}"
            logger.error(error_msg)
            self.rename_stats['errors'].append(error_msg)
            return file_path
    
    def is_filename_compliant(self, filename):
        """Check if filename already complies with standards"""
        # Check for common compliant patterns
        compliant_patterns = [
            r'Race_\d{2}_[A-Z_]+_\d{4}-\d{2}-\d{2}\.csv',
            r'FormGuide_[A-Z_]+_\d{4}-\d{2}-\d{2}_\d{2}\.csv',
            r'Analysis_\w+_[A-Z_]+_\d{4}-\d{2}-\d{2}_\d{6}\.(json|csv)',
            r'Upcoming_[A-Z_]+_\d{4}-\d{2}-\d{2}_\d{2}\.csv'
        ]
        
        for pattern in compliant_patterns:
            if re.match(pattern, filename):
                return True
        return False
    
    def standardize_all_files(self):
        """Apply naming standards to all relevant files"""
        logger.info("Applying naming standards to all files...")
        
        # Find files that need standardization
        file_patterns = ['**/*.csv', '**/*.json']
        exclude_dirs = ['backup_before_cleanup', 'quarantine', 'file_naming_standards']
        
        for pattern in file_patterns:
            for file_path in self.base_path.glob(pattern):
                # Skip excluded directories
                if any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
                    continue
                
                self.rename_file_if_needed(file_path)
        
        logger.info(f"Standardization complete. Renamed {self.rename_stats['files_renamed']} files")
    
    def create_validation_script(self):
        """Create a validation script to check compliance"""
        validator_content = '''#!/usr/bin/env python3
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
        r'Race_\\d{2}_[A-Z_]+_\\d{4}-\\d{2}-\\d{2}\\.csv',
        r'FormGuide_[A-Z_]+_\\d{4}-\\d{2}-\\d{2}_\\d{2}\\.csv',
        r'Analysis_\\w+_[A-Z_]+_\\d{4}-\\d{2}-\\d{2}_\\d{6}\\.(json|csv)',
        r'Upcoming_[A-Z_]+_\\d{4}-\\d{2}-\\d{2}_\\d{2}\\.csv'
    ]
    
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
    
    exclude_dirs = ['backup_before_cleanup', 'quarantine', 'file_naming_standards']
    
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
    
    print(f"\\nFile Naming Validation Results")
    print("=" * 40)
    print(f"Total files checked: {results['total_files']}")
    print(f"Compliant files: {len(results['compliant'])}")
    print(f"Non-compliant files: {len(results['non_compliant'])}")
    
    if results['non_compliant']:
        print("\\nNon-compliant files:")
        for item in results['non_compliant'][:10]:  # Show first 10
            print(f"  - {item['file']}: {item['reason']}")
        
        if len(results['non_compliant']) > 10:
            print(f"  ... and {len(results['non_compliant']) - 10} more")
    
    compliance_rate = len(results['compliant']) / results['total_files'] * 100 if results['total_files'] > 0 else 100
    print(f"\\nCompliance rate: {compliance_rate:.1f}%")
'''
        
        validator_path = self.standards_dir / "file_naming_validator.py"
        with open(validator_path, 'w') as f:
            f.write(validator_content)
        
        # Make it executable
        validator_path.chmod(0o755)
        
        logger.info(f"Validation script created at {validator_path}")
    
    def create_enforcement_hook(self):
        """Create a git pre-commit hook to enforce naming standards"""
        hook_content = '''#!/bin/sh
# Pre-commit hook to enforce file naming standards

echo "Checking file naming standards..."

# Run the validator
python file_naming_standards/file_naming_validator.py .

# Check if there are any non-compliant files being committed
non_compliant=$(python file_naming_standards/file_naming_validator.py . | grep "Non-compliant files:" | cut -d: -f2 | tr -d ' ')

if [ "$non_compliant" != "0" ]; then
    echo "❌ Commit rejected: Files do not comply with naming standards"
    echo "Run 'python file_naming_standards.py' to fix naming issues"
    exit 1
fi

echo "✅ All files comply with naming standards"
exit 0
'''
        
        git_hooks_dir = self.base_path / ".git" / "hooks"
        if git_hooks_dir.exists():
            hook_path = git_hooks_dir / "pre-commit"
            with open(hook_path, 'w') as f:
                f.write(hook_content)
            hook_path.chmod(0o755)
            logger.info(f"Git pre-commit hook created at {hook_path}")
        else:
            logger.info("No git repository found, skipping pre-commit hook creation")
    
    def generate_standards_report(self):
        """Generate a report on naming standards implementation"""
        report = {
            'implementation_timestamp': datetime.now().isoformat(),
            'standards_version': '1.0',
            'rename_statistics': self.rename_stats,
            'standards_implemented': list(self.standards.keys()),
            'documentation_created': [
                'naming_standards.json',
                'NAMING_STANDARDS.md',
                'file_naming_validator.py'
            ],
            'enforcement_enabled': True
        }
        
        report_path = self.base_path / "naming_standards_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def implement_all_standards(self):
        """Implement all naming standards and enforcement"""
        logger.info("Implementing comprehensive file naming standards...")
        
        self.create_naming_documentation()
        self.create_validation_script()
        self.create_enforcement_hook()
        self.standardize_all_files()
        
        report = self.generate_standards_report()
        
        logger.info("File naming standards implementation completed!")
        return report

def print_standards_summary(report):
    """Print implementation summary"""
    print("\n" + "="*60)
    print("FILE NAMING STANDARDS IMPLEMENTATION")
    print("="*60)
    
    stats = report['rename_statistics']
    print(f"Files renamed: {stats['files_renamed']:,}")
    print(f"Files already compliant: {stats['files_skipped']:,}")
    print(f"Errors encountered: {len(stats['errors'])}")
    
    print(f"\nStandards implemented:")
    for standard in report['standards_implemented']:
        print(f"  - {standard.replace('_', ' ').title()}")
    
    print(f"\nDocumentation created:")
    for doc in report['documentation_created']:
        print(f"  - {doc}")
    
    print(f"\nEnforcement enabled: {'Yes' if report['enforcement_enabled'] else 'No'}")
    
    if stats['errors']:
        print(f"\nErrors (first 3):")
        for error in stats['errors'][:3]:
            print(f"  - {error}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    base_path = "/Users/orlandolee/greyhound_racing_collector"
    standardizer = FileNamingStandardizer(base_path)
    
    try:
        report = standardizer.implement_all_standards()
        print_standards_summary(report)
        
    except Exception as e:
        logger.error(f"Standards implementation failed: {e}")
        import traceback
        traceback.print_exc()
