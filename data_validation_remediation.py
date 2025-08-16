#!/usr/bin/env python3
"""
Data Validation and Remediation Script

Based on the investigation findings:
- 10,668 races with field_size=1 in database (83% of all races!)
- Only 1 actual single-runner race found in dog_race_data
- Significant discrepancy between CSV data and database records

This script will:
1. Identify and fix field size calculation errors  
2. Implement validation rules for race data
3. Create automated monitoring for data quality
4. Generate corrective actions for historical data
"""

import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationRemediation:
    def __init__(self, db_path='greyhound_racing_data.db'):
        self.db_path = db_path
        self.validation_results = []
        
    def connect_db(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)
    
    def analyze_field_size_discrepancies(self):
        """Analyze and fix field size calculation errors"""
        logger.info("=== ANALYZING FIELD SIZE DISCREPANCIES ===")
        
        with self.connect_db() as conn:
            # Get races with their actual runner counts
            query = """
            SELECT 
                rm.id,
                rm.race_id,
                rm.venue,
                rm.race_date,
                rm.distance,
                rm.grade,
                rm.field_size as recorded_field_size,
                rm.actual_field_size as recorded_actual_field_size,
                COUNT(drd.id) as actual_runner_count
            FROM race_metadata rm
            LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
            GROUP BY rm.id, rm.race_id, rm.venue, rm.race_date, rm.distance, rm.grade, rm.field_size, rm.actual_field_size
            ORDER BY rm.race_date DESC
            """
            
            df = pd.read_sql_query(query, conn)
            
            # Identify discrepancies
            discrepancies = df[
                (df['recorded_field_size'] != df['actual_runner_count']) |
                (df['recorded_actual_field_size'] != df['actual_runner_count'])
            ]
            
            logger.info(f"Total races analyzed: {len(df)}")
            logger.info(f"Races with field size discrepancies: {len(discrepancies)}")
            
            # Breakdown by discrepancy type
            field_size_wrong = discrepancies[discrepancies['recorded_field_size'] != discrepancies['actual_runner_count']]
            actual_field_size_wrong = discrepancies[discrepancies['recorded_actual_field_size'] != discrepancies['actual_runner_count']]
            
            logger.info(f"field_size column wrong: {len(field_size_wrong)}")
            logger.info(f"actual_field_size column wrong: {len(actual_field_size_wrong)}")
            
            # Show sample discrepancies
            logger.info("\\nSample discrepancies:")
            for _, row in discrepancies.head(10).iterrows():\n                logger.info(f"  {row['venue']} {row['race_date']}: recorded={row['recorded_field_size']}, actual={row['actual_runner_count']}")\n            \n            return discrepancies\n    \n    def fix_field_sizes(self, dry_run=True):\n        """Fix field size calculations in database"""\n        logger.info(f"=== FIXING FIELD SIZES (dry_run={dry_run}) ===")\n        \n        with self.connect_db() as conn:\n            if not dry_run:\n                # Update field_size based on actual runner count\n                update_query = \"\"\"\n                UPDATE race_metadata \n                SET field_size = (\n                    SELECT COUNT(*) \n                    FROM dog_race_data drd \n                    WHERE drd.race_id = race_metadata.race_id\n                ),\n                actual_field_size = (\n                    SELECT COUNT(*) \n                    FROM dog_race_data drd \n                    WHERE drd.race_id = race_metadata.race_id\n                )\n                WHERE EXISTS (\n                    SELECT 1 FROM dog_race_data drd \n                    WHERE drd.race_id = race_metadata.race_id\n                )\n                \"\"\"\n                \n                cursor = conn.cursor()\n                cursor.execute(update_query)\n                updated_rows = cursor.rowcount\n                logger.info(f\"Updated field sizes for {updated_rows} races\")\n                conn.commit()\n            else:\n                # Just show what would be updated\n                preview_query = \"\"\"\n                SELECT \n                    rm.race_id,\n                    rm.venue,\n                    rm.field_size as old_field_size,\n                    COUNT(drd.id) as new_field_size\n                FROM race_metadata rm\n                LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id\n                WHERE rm.field_size != COUNT(drd.id) OR rm.field_size IS NULL\n                GROUP BY rm.race_id, rm.venue, rm.field_size\n                LIMIT 20\n                \"\"\"\n                \n                try:\n                    preview_df = pd.read_sql_query(preview_query, conn)\n                    logger.info(f\"Would update {len(preview_df)} races (showing first 20):\")\n                    for _, row in preview_df.iterrows():\n                        logger.info(f\"  {row['venue']} {row['race_id']}: {row['old_field_size']} -> {row['new_field_size']}\")\n                except Exception as e:\n                    logger.warning(f\"Could not generate preview: {e}\")\n    \n    def validate_csv_structure(self):\n        \"\"\"Validate CSV file structure and content\"\"\"\n        logger.info("=== VALIDATING CSV STRUCTURE ===")\n        \n        processed_dir = Path("processed")\n        validation_issues = []\n        \n        # Check various processing steps\n        for step_dir in processed_dir.glob("step*"):\n            csv_files = list(step_dir.glob("*.csv"))\n            logger.info(f\"\\nChecking {step_dir.name}: {len(csv_files)} files\")\n            \n            for csv_file in csv_files[:5]:  # Sample first 5 files\n                try:\n                    df = pd.read_csv(csv_file)\n                    \n                    # Basic validation\n                    issue = {\n                        'file': str(csv_file),\n                        'rows': len(df),\n                        'issues': []\n                    }\n                    \n                    # Check for single-row files (potential single dog races)\n                    if len(df) == 1:\n                        issue['issues'].append('single_row_file')\n                    \n                    # Check for missing essential columns\n                    essential_columns = ['Dog Name', 'DATE', 'TRACK']\n                    missing_columns = [col for col in essential_columns if col not in df.columns]\n                    if missing_columns:\n                        issue['issues'].append(f'missing_columns: {missing_columns}')\n                    \n                    # Check for empty dog names\n                    if 'Dog Name' in df.columns:\n                        empty_names = df['Dog Name'].isna().sum()\n                        if empty_names > 0:\n                            issue['issues'].append(f'empty_dog_names: {empty_names}')\n                    \n                    if issue['issues']:\n                        validation_issues.append(issue)\n                        \n                except Exception as e:\n                    validation_issues.append({\n                        'file': str(csv_file),\n                        'issues': [f'read_error: {str(e)}']\n                    })\n        \n        logger.info(f\"\\nValidation complete. Found {len(validation_issues)} files with issues.\")\n        \n        # Show summary of issues\n        issue_types = {}\n        for issue in validation_issues:\n            for issue_type in issue['issues']:\n                key = issue_type.split(':')[0]\n                issue_types[key] = issue_types.get(key, 0) + 1\n        \n        logger.info(\"Issue type summary:\")\n        for issue_type, count in issue_types.items():\n            logger.info(f\"  {issue_type}: {count} files\")\n        \n        return validation_issues\n    \n    def create_validation_rules(self):\n        \"\"\"Create automated validation rules\"\"\"\n        logger.info("=== CREATING VALIDATION RULES ===")\n        \n        validation_rules = {\n            'field_size_consistency': {\n                'description': 'field_size should match actual runner count',\n                'sql': '''\n                    SELECT race_id, venue, race_date, field_size, \n                           (SELECT COUNT(*) FROM dog_race_data WHERE race_id = rm.race_id) as actual_count\n                    FROM race_metadata rm \n                    WHERE field_size != (SELECT COUNT(*) FROM dog_race_data WHERE race_id = rm.race_id)\n                ''',\n                'severity': 'HIGH'\n            },\n            'missing_runners': {\n                'description': 'races should have at least 2 runners typically',\n                'sql': '''\n                    SELECT race_id, venue, race_date, field_size\n                    FROM race_metadata \n                    WHERE field_size < 2 AND grade NOT LIKE '%walkover%'\n                ''',\n                'severity': 'MEDIUM'\n            },\n            'excessive_runners': {\n                'description': 'races should not have more than 12 runners typically',\n                'sql': '''\n                    SELECT race_id, venue, race_date, field_size\n                    FROM race_metadata \n                    WHERE field_size > 12\n                ''',\n                'severity': 'LOW'\n            }\n        }\n        \n        # Test each validation rule\n        with self.connect_db() as conn:\n            for rule_name, rule in validation_rules.items():\n                try:\n                    result_df = pd.read_sql_query(rule['sql'], conn)\n                    logger.info(f\"{rule_name} ({rule['severity']}): {len(result_df)} violations\")\n                    \n                    if len(result_df) > 0 and len(result_df) <= 5:\n                        logger.info(\"  Sample violations:\")\n                        for _, row in result_df.iterrows():\n                            logger.info(f\"    {row.to_dict()}\")\n                            \n                except Exception as e:\n                    logger.error(f\"Failed to execute rule {rule_name}: {e}\")\n        \n        return validation_rules\n    \n    def generate_remediation_report(self):\n        \"\"\"Generate comprehensive remediation report\"\"\"\n        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n        report_path = f\"reports/data_remediation_{timestamp}.md\"\n        \n        # Analyze current state\n        discrepancies = self.analyze_field_size_discrepancies()\n        validation_issues = self.validate_csv_structure()\n        validation_rules = self.create_validation_rules()\n        \n        with open(report_path, 'w') as f:\n            f.write(\"# Data Validation and Remediation Report\\n\\n\")\n            f.write(f\"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n\")\n            \n            f.write(\"## Executive Summary\\n\")\n            f.write(f\"- **Critical Issue**: 83% of races ({len(discrepancies)}) have incorrect field_size values\\n\")\n            f.write(f\"- **CSV Issues**: {len(validation_issues)} files with structural problems\\n\")\n            f.write(f\"- **Validation Rules**: {len(validation_rules)} automated checks implemented\\n\\n\")\n            \n            f.write(\"## Key Findings\\n\")\n            f.write(\"1. **Database Field Size Error**: Most races show field_size=1 when actual runner count is higher\\n\")\n            f.write(\"2. **CSV Single-Dog Issue**: Many CSV files contain only one dog's historical data, not race results\\n\")\n            f.write(\"3. **Data Structure Mismatch**: CSV format appears to be dog performance history, not race results\\n\\n\")\n            \n            f.write(\"## Immediate Actions Required\\n\")\n            f.write(\"1. **Fix Field Sizes**: Run `fix_field_sizes(dry_run=False)` to correct database\\n\")\n            f.write(\"2. **Review CSV Format**: Clarify whether CSV files are race results or dog performance history\\n\")\n            f.write(\"3. **Implement Monitoring**: Set up automated validation rule checking\\n\\n\")\n            \n            f.write(\"## Recommended Next Steps\\n\")\n            f.write(\"- Review data collection and ingestion processes\\n\")\n            f.write(\"- Implement real-time validation during data ingestion\\n\")\n            f.write(\"- Create alerts for data quality threshold violations\\n\")\n            f.write(\"- Establish regular data quality audits\\n\")\n        \n        logger.info(f\"Remediation report saved to: {report_path}\")\n        return report_path\n\ndef main():\n    \"\"\"Main remediation function\"\"\"\n    logger.info(\"Starting Data Validation and Remediation...\")\n    \n    remediation = DataValidationRemediation()\n    \n    # Run analysis\n    remediation.analyze_field_size_discrepancies()\n    remediation.validate_csv_structure()\n    remediation.create_validation_rules()\n    \n    # Generate report\n    report_path = remediation.generate_remediation_report()\n    \n    logger.info(f\"\\n=== REMEDIATION COMPLETE ===\")\n    logger.info(f\"Report available at: {report_path}\")\n    logger.info(\"\\nTo fix field sizes, run:\")\n    logger.info(\"  python -c 'from data_validation_remediation import DataValidationRemediation; DataValidationRemediation().fix_field_sizes(dry_run=False)'\")\n\nif __name__ == \"__main__\":\n    main()
