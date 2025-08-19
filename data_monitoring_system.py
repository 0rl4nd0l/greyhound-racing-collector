#!/usr/bin/env python3
"""
Comprehensive Data Monitoring and Validation System

This script implements:
1. Automated data quality checks
2. Alert system for data issues
3. Regular monitoring schedules
4. Comprehensive reporting
"""

import pandas as pd
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMonitoringSystem:
    def __init__(self, db_path='greyhound_racing_data.db'):
        self.db_path = db_path
        self.validation_rules = self._define_validation_rules()
        self.alert_thresholds = {
            'critical': 50,  # Alert if more than 50 critical issues
            'warning': 100,  # Alert if more than 100 warning issues
            'info': 500      # Log if more than 500 info issues
        }
        
    def connect_db(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)
    
    def _define_validation_rules(self):
        """Define comprehensive validation rules"""
        return {
            'field_size_consistency': {
                'description': 'Field size should match actual runner count',
                'sql': '''
                    SELECT race_id, venue, race_date, field_size, 
                           (SELECT COUNT(*) FROM dog_race_data WHERE race_id = rm.race_id) as actual_count
                    FROM race_metadata rm 
                    WHERE field_size != (SELECT COUNT(*) FROM dog_race_data WHERE race_id = rm.race_id)
                ''',
                'severity': 'CRITICAL',
                'threshold': 10
            },
            'zero_runners': {
                'description': 'Races should not have zero runners',
                'sql': '''
                    SELECT race_id, venue, race_date, field_size
                    FROM race_metadata 
                    WHERE field_size = 0
                ''',
                'severity': 'CRITICAL',
                'threshold': 1
            },
            'single_runner_races': {
                'description': 'Single runner races should be rare',
                'sql': '''
                    SELECT race_id, venue, race_date, field_size, grade
                    FROM race_metadata 
                    WHERE field_size = 1 AND grade NOT LIKE '%walkover%' AND grade NOT LIKE '%scratch%'
                ''',
                'severity': 'WARNING',
                'threshold': 50
            },
            'excessive_runners': {
                'description': 'Races should not have more than 12 runners typically',
                'sql': '''
                    SELECT race_id, venue, race_date, field_size
                    FROM race_metadata 
                    WHERE field_size > 12
                ''',
                'severity': 'INFO',
                'threshold': 100
            },
            'missing_race_metadata': {
                'description': 'Dog race data should have corresponding race metadata',
                'sql': '''
                    SELECT DISTINCT drd.race_id
                    FROM dog_race_data drd
                    LEFT JOIN race_metadata rm ON drd.race_id = rm.race_id
                    WHERE rm.race_id IS NULL
                ''',
                'severity': 'CRITICAL',
                'threshold': 1
            },
            'duplicate_race_ids': {
                'description': 'Race IDs should be unique in race_metadata',
                'sql': '''
                    SELECT race_id, COUNT(*) as count
                    FROM race_metadata
                    GROUP BY race_id
                    HAVING COUNT(*) > 1
                ''',
                'severity': 'CRITICAL',
                'threshold': 1
            },
            'future_race_dates': {
                'description': 'Race dates should not be too far in the future',
                'sql': '''
                    SELECT race_id, venue, race_date
                    FROM race_metadata 
                    WHERE race_date > date('now', '+7 days')
                ''',
                'severity': 'WARNING',
                'threshold': 20
            },
            'missing_essential_data': {
                'description': 'Essential fields should not be null',
                'sql': '''
                    SELECT race_id, venue, race_date
                    FROM race_metadata
                    WHERE venue IS NULL OR race_date IS NULL OR distance IS NULL
                ''',
                'severity': 'CRITICAL',
                'threshold': 1
            }
        }
    
    def run_validation_rule(self, rule_name, rule_config):
        """Run a single validation rule"""
        try:
            with self.connect_db() as conn:
                result_df = pd.read_sql_query(rule_config['sql'], conn)
                
                violation_count = len(result_df)
                severity = rule_config['severity']
                threshold = rule_config['threshold']
                
                # Determine if this exceeds threshold
                exceeds_threshold = violation_count > threshold
                
                result = {
                    'rule_name': rule_name,
                    'description': rule_config['description'],
                    'severity': severity,
                    'violation_count': violation_count,
                    'threshold': threshold,
                    'exceeds_threshold': exceeds_threshold,
                    'sample_violations': result_df.head(5).to_dict('records') if not result_df.empty else []
                }
                
                # Log results
                if exceeds_threshold:
                    logger.warning(f"{rule_name} ({severity}): {violation_count} violations (threshold: {threshold})")
                else:
                    logger.info(f"{rule_name} ({severity}): {violation_count} violations (threshold: {threshold})")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to execute rule {rule_name}: {e}")
            return {
                'rule_name': rule_name,
                'error': str(e),
                'severity': rule_config['severity']
            }
    
    def run_all_validations(self):
        """Run all validation rules"""
        logger.info("=== RUNNING ALL VALIDATION RULES ===")
        
        results = []
        critical_issues = 0
        warning_issues = 0
        info_issues = 0
        
        for rule_name, rule_config in self.validation_rules.items():
            result = self.run_validation_rule(rule_name, rule_config)
            results.append(result)
            
            # Count issues by severity
            if 'violation_count' in result:
                if result['severity'] == 'CRITICAL' and result['exceeds_threshold']:
                    critical_issues += result['violation_count']
                elif result['severity'] == 'WARNING' and result['exceeds_threshold']:
                    warning_issues += result['violation_count']
                elif result['severity'] == 'INFO' and result['exceeds_threshold']:
                    info_issues += result['violation_count']
        
        # Generate alerts based on thresholds
        alerts = []
        if critical_issues > self.alert_thresholds['critical']:
            alerts.append(f"CRITICAL ALERT: {critical_issues} critical issues detected")
        if warning_issues > self.alert_thresholds['warning']:
            alerts.append(f"WARNING ALERT: {warning_issues} warning issues detected")
        if info_issues > self.alert_thresholds['info']:
            alerts.append(f"INFO ALERT: {info_issues} info issues detected")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'critical_issues': critical_issues,
                'warning_issues': warning_issues,
                'info_issues': info_issues
            },
            'alerts': alerts
        }
    
    def create_monitoring_report(self, validation_results):
        """Create comprehensive monitoring report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/data_monitoring_{timestamp}.md"
        
        os.makedirs("reports", exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Data Monitoring Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n")
            summary = validation_results['summary']
            f.write(f"- **Critical Issues**: {summary['critical_issues']}\n")
            f.write(f"- **Warning Issues**: {summary['warning_issues']}\n")
            f.write(f"- **Info Issues**: {summary['info_issues']}\n\n")
            
            # Alerts
            if validation_results['alerts']:
                f.write("## 🚨 Alerts\n")
                for alert in validation_results['alerts']:
                    f.write(f"- {alert}\n")
                f.write("\n")
            
            # Detailed Results
            f.write("## Validation Results\n\n")
            
            for result in validation_results['results']:
                if 'error' in result:
                    f.write(f"### ❌ {result['rule_name']} (ERROR)\n")
                    f.write(f"Error: {result['error']}\n\n")
                    continue
                
                # Status icon based on severity and threshold
                if result['exceeds_threshold']:
                    if result['severity'] == 'CRITICAL':
                        icon = "🔴"
                    elif result['severity'] == 'WARNING':
                        icon = "🟡"
                    else:
                        icon = "🔵"
                else:
                    icon = "✅"
                
                f.write(f"### {icon} {result['rule_name']} ({result['severity']})\n")
                f.write(f"**Description**: {result['description']}\n")
                f.write(f"**Violations**: {result['violation_count']} (threshold: {result['threshold']})\n")
                
                if result['sample_violations']:
                    f.write("**Sample Violations**:\n")
                    for violation in result['sample_violations'][:3]:
                        f.write(f"- {violation}\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n")
            f.write("1. **Address Critical Issues**: Fix all critical issues immediately\n")
            f.write("2. **Monitor Trends**: Track issue counts over time\n")
            f.write("3. **Implement Fixes**: Create automated fixes for common issues\n")
            f.write("4. **Regular Monitoring**: Run this monitoring system daily\n\n")
        
        logger.info(f"Monitoring report saved to: {report_path}")
        return report_path
    
    def save_validation_results(self, validation_results):
        """Save validation results as JSON for trend analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"reports/validation_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results saved to: {results_path}")
        return results_path
    
    def check_csv_data_quality(self):
        """Check CSV data quality issues"""
        logger.info("=== CHECKING CSV DATA QUALITY ===")
        
        processed_dir = Path("processed")
        issues = []
        
        # Check for single-dog CSV files (the original issue)
        single_dog_files = 0
        total_files = 0
        
        for step_dir in processed_dir.glob("step*"):
            csv_files = list(step_dir.glob("*.csv"))
            for csv_file in csv_files:
                total_files += 1
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) == 1:
                        single_dog_files += 1
                except Exception as e:
                    issues.append(f"Could not read {csv_file}: {e}")
        
        logger.info(f"CSV Quality Check: {single_dog_files}/{total_files} files have single dog entries")
        
        return {
            'total_files': total_files,
            'single_dog_files': single_dog_files,
            'single_dog_percentage': (single_dog_files / total_files * 100) if total_files > 0 else 0,
            'read_errors': len(issues)
        }

def main():
    """Main monitoring function"""
    logger.info("Starting Comprehensive Data Monitoring System...")
    
    monitor = DataMonitoringSystem()
    
    # Run all validations
    validation_results = monitor.run_all_validations()
    
    # Check CSV quality
    csv_quality = monitor.check_csv_data_quality()
    validation_results['csv_quality'] = csv_quality
    
    # Create reports
    report_path = monitor.create_monitoring_report(validation_results)
    results_path = monitor.save_validation_results(validation_results)
    
    logger.info(f"\n=== MONITORING COMPLETE ===")
    logger.info(f"Report: {report_path}")
    logger.info(f"Results: {results_path}")
    
    # Print summary
    summary = validation_results['summary']
    logger.info(f"\nSUMMARY:")
    logger.info(f"  Critical Issues: {summary['critical_issues']}")
    logger.info(f"  Warning Issues: {summary['warning_issues']}")
    logger.info(f"  Info Issues: {summary['info_issues']}")
    logger.info(f"  CSV Single-Dog Files: {csv_quality['single_dog_files']}/{csv_quality['total_files']} ({csv_quality['single_dog_percentage']:.1f}%)")
    
    # Show alerts
    if validation_results['alerts']:
        logger.warning("ALERTS:")
        for alert in validation_results['alerts']:
            logger.warning(f"  {alert}")

if __name__ == "__main__":
    main()
