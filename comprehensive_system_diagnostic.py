#!/usr/bin/env python3
"""
Comprehensive System Diagnostic
===============================

This script performs a complete diagnostic of the greyhound prediction system
to identify all issues that might prevent it from working correctly.

It checks:
1. Database schema integrity
2. Dependencies and imports
3. File structure
4. Model availability
5. Prediction pipeline functionality
6. API endpoints
"""

import os
import sys
import sqlite3
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

class SystemDiagnostic:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []
        self.db_path = "greyhound_racing_data.db"
        
    def log_issue(self, message, category="ERROR"):
        """Log an issue with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {category}: {message}"
        
        if category == "ERROR":
            self.issues.append(full_message)
            print(f"❌ {full_message}")
        elif category == "WARNING":
            self.warnings.append(full_message)
            print(f"⚠️  {full_message}")
        else:
            self.successes.append(full_message)
            print(f"✅ {full_message}")
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print("\n🔍 Checking Dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'flask', 'sqlite3',
            'joblib', 'flask_cors', 'pathlib', 'datetime'
        ]
        
        optional_packages = [
            ('xgboost', 'XGBoost for advanced ML models'),
            ('openai', 'OpenAI GPT integration'),
            ('requests', 'HTTP requests for scraping')
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.log_issue(f"Required package '{package}' is available", "SUCCESS")
            except ImportError as e:
                self.log_issue(f"Required package '{package}' is missing: {e}")
        
        for package, description in optional_packages:
            try:
                __import__(package)
                self.log_issue(f"Optional package '{package}' is available", "SUCCESS")
            except ImportError:
                self.log_issue(f"Optional package '{package}' is missing ({description})", "WARNING")
    
    def check_database_schema(self):
        """Check database schema integrity"""
        print("\n🔍 Checking Database Schema...")
        
        if not os.path.exists(self.db_path):
            self.log_issue(f"Database file {self.db_path} does not exist")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for essential tables
            essential_tables = [
                'race_metadata',
                'dog_race_data', 
                'dogs',
                'trainers'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            for table in essential_tables:
                if table in existing_tables:
                    self.log_issue(f"Essential table '{table}' exists", "SUCCESS")
                    
                    # Check table structure
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    self.log_issue(f"Table '{table}' has {len(columns)} columns", "SUCCESS")
                else:
                    self.log_issue(f"Essential table '{table}' is missing")
            
            # Check data availability
            for table in existing_tables:
                if table.startswith('sqlite_'):
                    continue
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        self.log_issue(f"Table '{table}' contains {count} records", "SUCCESS")
                    else:
                        self.log_issue(f"Table '{table}' is empty", "WARNING")
                except Exception as e:
                    self.log_issue(f"Error checking table '{table}': {e}")
            
            conn.close()
            
        except Exception as e:
            self.log_issue(f"Database connection error: {e}")
    
    def check_file_structure(self):
        """Check file and directory structure"""
        print("\n🔍 Checking File Structure...")
        
        essential_files = [
            'app.py',
            'ml_system_v3.py',
            'prediction_pipeline_v3.py',
            'logger.py',
            'traditional_analysis.py'
        ]
        
        essential_dirs = [
            'upcoming_races',
            'predictions',
            'templates',
            'static'
        ]
        
        for file in essential_files:
            if os.path.exists(file):
                self.log_issue(f"Essential file '{file}' exists", "SUCCESS")
            else:
                self.log_issue(f"Essential file '{file}' is missing")
        
        for directory in essential_dirs:
            if os.path.exists(directory):
                file_count = len(os.listdir(directory)) if os.path.isdir(directory) else 0
                self.log_issue(f"Directory '{directory}' exists with {file_count} items", "SUCCESS")
            else:
                self.log_issue(f"Directory '{directory}' is missing")
    
    def check_imports(self):
        """Check if key modules can be imported"""
        print("\n🔍 Checking Module Imports...")
        
        modules_to_test = [
            ('logger', 'Logging system'),
            ('traditional_analysis', 'Traditional analysis system'),
            ('model_registry', 'Model registry system'),
            ('utils.file_naming', 'File naming utilities')
        ]
        
        for module, description in modules_to_test:
            try:
                __import__(module)
                self.log_issue(f"Module '{module}' imports successfully", "SUCCESS")
            except ImportError as e:
                self.log_issue(f"Module '{module}' import failed ({description}): {e}")
            except Exception as e:
                self.log_issue(f"Module '{module}' has runtime error: {e}", "WARNING")
    
    def check_ml_system(self):
        """Check ML system availability"""
        print("\n🔍 Checking ML System...")
        
        try:
            # Try to import without initializing (avoid XGBoost issues)
            import ml_system_v3
            self.log_issue("ML System V3 module imports successfully", "SUCCESS")
            
            # Check if XGBoost is the issue
            try:
                import xgboost
                self.log_issue("XGBoost is available", "SUCCESS")
            except ImportError as e:
                self.log_issue(f"XGBoost import failed: {e}")
                self.log_issue("This is likely the main issue preventing system startup", "ERROR")
                self.log_issue("Run: brew install libomp", "WARNING")
            
        except ImportError as e:
            self.log_issue(f"ML System V3 import failed: {e}")
        except Exception as e:
            self.log_issue(f"ML System V3 has runtime error: {e}", "WARNING")
    
    def check_prediction_pipeline(self):
        """Check prediction pipeline"""
        print("\n🔍 Checking Prediction Pipeline...")
        
        try:
            # Try to import prediction pipeline
            import prediction_pipeline_v3
            self.log_issue("Prediction Pipeline V3 imports successfully", "SUCCESS")
            
        except ImportError as e:
            self.log_issue(f"Prediction Pipeline V3 import failed: {e}")
        except Exception as e:
            self.log_issue(f"Prediction Pipeline V3 has runtime error: {e}", "WARNING")
    
    def check_flask_app(self):
        """Check if Flask app can be imported"""
        print("\n🔍 Checking Flask Application...")
        
        try:
            # Try to import app module (without running it)
            import app
            self.log_issue("Flask app module imports successfully", "SUCCESS")
            
            # Check if app has the required routes
            if hasattr(app, 'app'):
                routes = []
                for rule in app.app.url_map.iter_rules():
                    routes.append(str(rule))
                
                self.log_issue(f"Flask app has {len(routes)} routes defined", "SUCCESS")
                
                # Check for essential routes
                essential_routes = [
                    '/api/race_files_status',
                    '/api/predict_single_race',
                    '/ml_dashboard'
                ]
                
                for route in essential_routes:
                    if any(route in r for r in routes):
                        self.log_issue(f"Essential route '{route}' is defined", "SUCCESS")
                    else:
                        self.log_issue(f"Essential route '{route}' is missing")
            
        except ImportError as e:
            self.log_issue(f"Flask app import failed: {e}")
        except Exception as e:
            self.log_issue(f"Flask app has runtime error: {e}", "WARNING")
    
    def check_race_files(self):
        """Check upcoming race files"""
        print("\n🔍 Checking Race Files...")
        
        upcoming_dir = "upcoming_races"
        if not os.path.exists(upcoming_dir):
            self.log_issue(f"Upcoming races directory '{upcoming_dir}' does not exist")
            return
        
        csv_files = [f for f in os.listdir(upcoming_dir) if f.endswith('.csv')]
        
        if csv_files:
            self.log_issue(f"Found {len(csv_files)} CSV files in upcoming_races", "SUCCESS")
            
            # Check a sample file
            sample_file = os.path.join(upcoming_dir, csv_files[0])
            try:
                with open(sample_file, 'r') as f:
                    content = f.read()
                    if len(content) > 100:
                        self.log_issue(f"Sample file '{csv_files[0]}' has content ({len(content)} chars)", "SUCCESS")
                    else:
                        self.log_issue(f"Sample file '{csv_files[0]}' seems too small", "WARNING")
            except Exception as e:
                self.log_issue(f"Error reading sample file: {e}")
        else:
            self.log_issue("No CSV files found in upcoming_races directory", "WARNING")
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE SYSTEM DIAGNOSTIC REPORT")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"  ✅ Successes: {len(self.successes)}")
        print(f"  ⚠️  Warnings: {len(self.warnings)}")
        print(f"  ❌ Errors: {len(self.issues)}")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES TO FIX:")
            for issue in self.issues:
                print(f"  {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.successes:
            print(f"\n✅ WORKING COMPONENTS:")
            for success in self.successes[-5:]:  # Show last 5 successes
                print(f"  {success}")
            if len(self.successes) > 5:
                print(f"  ... and {len(self.successes) - 5} more")
        
        # Priority recommendations
        print(f"\n🚀 PRIORITY ACTIONS:")
        if any("XGBoost" in issue for issue in self.issues):
            print("  1. HIGHEST: Fix XGBoost dependency - run 'brew install libomp'")
        if any("database" in issue.lower() for issue in self.issues):
            print("  2. HIGH: Fix database schema issues")
        if any("import" in issue.lower() for issue in self.issues):
            print("  3. MEDIUM: Fix import/module issues")
        
        print(f"\n📋 Next Steps:")
        print("  1. Fix XGBoost/OpenMP issue first")
        print("  2. Test basic import of ml_system_v3")
        print("  3. Test prediction pipeline import")
        print("  4. Test Flask app startup")
        print("  5. Test single race prediction")
        
        return len(self.issues) == 0

def run_diagnostic():
    """Run the complete system diagnostic"""
    print("🚀 Starting Comprehensive System Diagnostic...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Timestamp: {datetime.now()}")
    
    diagnostic = SystemDiagnostic()
    
    # Run all checks
    diagnostic.check_dependencies()
    diagnostic.check_database_schema()
    diagnostic.check_file_structure()
    diagnostic.check_imports()
    diagnostic.check_ml_system()
    diagnostic.check_prediction_pipeline()
    diagnostic.check_flask_app()
    diagnostic.check_race_files()
    
    # Generate final report
    success = diagnostic.generate_report()
    
    if success:
        print(f"\n🎉 System appears to be healthy!")
        return True
    else:
        print(f"\n🔧 System needs repair - fix critical issues first")
        return False

if __name__ == "__main__":
    run_diagnostic()
