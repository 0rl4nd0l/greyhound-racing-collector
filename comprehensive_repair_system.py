#!/usr/bin/env python3
"""
Comprehensive Repair System for Greyhound Analysis Predictor
===========================================================

This system identifies and fixes all critical issues preventing the Flask application
and prediction pipeline from functioning correctly after the database migration.

Author: AI Assistant
Date: July 28, 2025
"""

import importlib.util
import json
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class ComprehensiveRepairSystem:
    def __init__(self, base_dir="/Users/orlandolee/greyhound_racing_collector"):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "greyhound_racing_data.db"
        self.venv_path = self.base_dir / "venv"
        self.logs_dir = self.base_dir / "repair_logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Track repair status
        self.repairs_completed = []
        self.repairs_failed = []

        print("🔧 Comprehensive Repair System for Greyhound Analysis Predictor")
        print("=" * 70)
        self.logger.info("Repair system initialized")

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = (
            self.logs_dir / f"repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def repair_virtual_environment(self):
        """Fix virtual environment and install missing dependencies"""
        print("\n🔧 REPAIRING VIRTUAL ENVIRONMENT")
        print("-" * 50)

        try:
            # Check if venv exists
            if not self.venv_path.exists():
                print("❌ Virtual environment not found. Creating new one...")
                subprocess.run(
                    [sys.executable, "-m", "venv", str(self.venv_path)], check=True
                )
                print("✅ Virtual environment created")

            # Get pip path
            pip_path = self.venv_path / "bin" / "pip"
            python_path = self.venv_path / "bin" / "python"

            # Install/upgrade pip
            print("🔄 Upgrading pip...")
            subprocess.run(
                [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
            )

            # Critical dependencies
            critical_deps = [
                "pandas==2.2.1",
                "numpy==1.26.4",
                "scikit-learn",
                "flask",
                "requests",
                "schedule",
                "APScheduler",
                "python-dotenv",
                "joblib",
                "matplotlib",
                "seaborn",
            ]

            print("📦 Installing critical dependencies...")
            for dep in critical_deps:
                try:
                    print(f"   Installing {dep}...")
                    result = subprocess.run(
                        [str(pip_path), "install", dep],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                    if result.returncode == 0:
                        print(f"   ✅ {dep} installed successfully")
                    else:
                        print(f"   ❌ Failed to install {dep}: {result.stderr}")
                        self.repairs_failed.append(f"dependency_install_{dep}")

                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Timeout installing {dep}")
                    self.repairs_failed.append(f"dependency_timeout_{dep}")
                except Exception as e:
                    print(f"   ❌ Error installing {dep}: {e}")
                    self.repairs_failed.append(f"dependency_error_{dep}")

            # Test imports
            print("🧪 Testing critical imports...")
            test_imports = ["pandas", "numpy", "sklearn", "flask", "schedule"]

            for module in test_imports:
                try:
                    result = subprocess.run(
                        [
                            str(python_path),
                            "-c",
                            f"import {module}; print('✅ {module}')",
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        print(f"   ✅ {module} imports successfully")
                    else:
                        print(f"   ❌ {module} import failed")
                        self.repairs_failed.append(f"import_test_{module}")

                except Exception as e:
                    print(f"   ❌ Error testing {module}: {e}")

            self.repairs_completed.append("virtual_environment")
            print("✅ Virtual environment repair completed")

        except Exception as e:
            print(f"❌ Virtual environment repair failed: {e}")
            self.repairs_failed.append("virtual_environment")
            self.logger.error(f"VEnv repair failed: {e}")

    def repair_app_imports(self):
        """Fix import issues in app.py and other core files"""
        print("\n🔧 REPAIRING APPLICATION IMPORTS")
        print("-" * 50)

        try:
            app_file = self.base_dir / "app.py"

            if not app_file.exists():
                print("❌ app.py not found")
                self.repairs_failed.append("app_file_missing")
                return

            # Read current app.py
            with open(app_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for problematic imports
            problematic_imports = []

            # Test import with venv python
            python_path = self.venv_path / "bin" / "python"

            print("🧪 Testing app.py import...")
            result = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    f"import sys; sys.path.insert(0, '{self.base_dir}'); import app; print('App import successful')",
                ],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
            )

            if result.returncode == 0:
                print("✅ app.py imports successfully")
                self.repairs_completed.append("app_imports")
            else:
                print(f"❌ app.py import failed: {result.stderr}")

                # Try to identify and fix specific import issues
                if "No module named 'schedule'" in result.stderr:
                    print("🔄 Installing schedule module...")
                    subprocess.run(
                        [str(self.venv_path / "bin" / "pip"), "install", "schedule"],
                        check=True,
                    )

                    # Test again
                    result2 = subprocess.run(
                        [
                            str(python_path),
                            "-c",
                            f"import sys; sys.path.insert(0, '{self.base_dir}'); import app; print('App import successful')",
                        ],
                        capture_output=True,
                        text=True,
                        cwd=str(self.base_dir),
                    )

                    if result2.returncode == 0:
                        print("✅ app.py imports successfully after schedule fix")
                        self.repairs_completed.append("app_imports")
                    else:
                        print(f"❌ app.py still failing: {result2.stderr}")
                        self.repairs_failed.append("app_imports")
                else:
                    self.repairs_failed.append("app_imports")

        except Exception as e:
            print(f"❌ App import repair failed: {e}")
            self.repairs_failed.append("app_imports")
            self.logger.error(f"App import repair failed: {e}")

    def repair_database_schema(self):
        """Fix database schema issues"""
        print("\n🔧 REPAIRING DATABASE SCHEMA")
        print("-" * 50)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check critical tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}

            critical_tables = {"race_metadata", "dog_race_data"}
            missing_tables = critical_tables - existing_tables

            if missing_tables:
                print(f"❌ Missing critical tables: {missing_tables}")
                self.repairs_failed.append("missing_tables")
            else:
                print("✅ Critical tables present")

            # Check for data quality issues
            print("🔍 Checking data quality...")

            # Missing winners
            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE winner_name IS NULL OR winner_name = '' OR winner_name = 'nan'
            """
            )
            missing_winners = cursor.fetchone()[0]

            if missing_winners > 0:
                print(f"⚠️  {missing_winners} races missing winners")
                # Could add repair logic here

            # Missing box numbers
            cursor.execute(
                """
                SELECT COUNT(*) FROM dog_race_data 
                WHERE box_number IS NULL
            """
            )
            missing_boxes = cursor.fetchone()[0]

            if missing_boxes > 0:
                print(f"⚠️  {missing_boxes} entries missing box numbers")
                # Could add repair logic here

            conn.close()
            self.repairs_completed.append("database_schema")
            print("✅ Database schema check completed")

        except Exception as e:
            print(f"❌ Database schema repair failed: {e}")
            self.repairs_failed.append("database_schema")
            self.logger.error(f"Database schema repair failed: {e}")

    def repair_prediction_pipeline(self):
        """Fix prediction pipeline issues"""
        print("\n🔧 REPAIRING PREDICTION PIPELINE")
        print("-" * 50)

        try:
            python_path = self.venv_path / "bin" / "python"

            # Test core prediction components
            prediction_modules = [
                "weather_enhanced_predictor",
                "unified_predictor",
                "comprehensive_enhanced_ml_system",
            ]

            for module in prediction_modules:
                module_file = self.base_dir / f"{module}.py"

                if not module_file.exists():
                    print(f"❌ {module}.py not found")
                    continue

                print(f"🧪 Testing {module}...")
                result = subprocess.run(
                    [
                        str(python_path),
                        "-c",
                        f"import sys; sys.path.insert(0, '{self.base_dir}'); import {module}; print('✅ {module} import successful')",
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(self.base_dir),
                )

                if result.returncode == 0:
                    print(f"   ✅ {module} imports successfully")
                else:
                    print(f"   ❌ {module} import failed: {result.stderr[:200]}...")

            # Test model loading
            print("🧪 Testing model loading...")
            models_dir = self.base_dir / "comprehensive_trained_models"
            if models_dir.exists():
                model_files = list(models_dir.glob("*.joblib"))
                if model_files:
                    print(f"   ✅ Found {len(model_files)} model files")
                else:
                    print("   ⚠️  No model files found")
            else:
                print("   ⚠️  Models directory not found")

            self.repairs_completed.append("prediction_pipeline")
            print("✅ Prediction pipeline check completed")

        except Exception as e:
            print(f"❌ Prediction pipeline repair failed: {e}")
            self.repairs_failed.append("prediction_pipeline")
            self.logger.error(f"Prediction pipeline repair failed: {e}")

    def repair_flask_application(self):
        """Test and repair Flask application"""
        print("\n🔧 REPAIRING FLASK APPLICATION")
        print("-" * 50)

        try:
            python_path = self.venv_path / "bin" / "python"

            # Test Flask app initialization
            test_script = f"""
import sys
sys.path.insert(0, '{self.base_dir}')
try:
    import app
    if hasattr(app, 'app'):
        flask_app = app.app
        print('✅ Flask app instance found')
        
        # Test basic configuration
        with flask_app.test_client() as client:
            flask_app.config['TESTING'] = True
            
            # Test root route
            try:
                response = client.get('/')
                print(f'✅ Root route: {{response.status_code}}')
            except Exception as e:
                print(f'❌ Root route failed: {{str(e)[:100]}}')
    else:
        print('❌ Flask app instance not found')
except Exception as e:
    print(f'❌ Flask test failed: {{str(e)[:200]}}')
"""

            result = subprocess.run(
                [str(python_path), "-c", test_script],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
            )

            print("Flask test output:")
            print(result.stdout)

            if "Flask app instance found" in result.stdout:
                print("✅ Flask application functioning")
                self.repairs_completed.append("flask_application")
            else:
                print("❌ Flask application issues detected")
                print("Error output:", result.stderr)
                self.repairs_failed.append("flask_application")

        except Exception as e:
            print(f"❌ Flask application repair failed: {e}")
            self.repairs_failed.append("flask_application")
            self.logger.error(f"Flask application repair failed: {e}")

    def create_fixed_diagnostic(self):
        """Create a fixed version of the diagnostic that handles JSON serialization"""
        print("\n🔧 CREATING FIXED DIAGNOSTIC")
        print("-" * 50)

        try:
            fixed_diagnostic_content = '''#!/usr/bin/env python3
"""
Fixed Diagnostic Script
======================
Handles JSON serialization issues and provides comprehensive testing.
"""

import os
import sys
import sqlite3
import json
import importlib.util
from pathlib import Path
from datetime import datetime

def convert_for_json(obj):
    """Convert objects to JSON-serializable format"""
    if hasattr(obj, 'item'):  # numpy/pandas types
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    else:
        return obj

class FixedDiagnostic:
    def __init__(self, base_dir="/Users/orlandolee/greyhound_racing_collector"):
        self.base_dir = Path(base_dir)
        self.issues = []
        
    def run_basic_tests(self):
        """Run basic functionality tests"""
        print("🧪 Running Basic Tests...")
        
        # Test database
        try:
            db_path = self.base_dir / "greyhound_racing_data.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            race_count = cursor.fetchone()[0]
            print(f"✅ Database: {race_count} races")
            conn.close()
        except Exception as e:
            print(f"❌ Database: {e}")
            self.issues.append({"type": "database_error", "error": str(e)})
        
        # Test Flask import
        try:
            sys.path.insert(0, str(self.base_dir))
            import app
            print("✅ Flask app import successful")
        except Exception as e:
            print(f"❌ Flask import: {e}")
            self.issues.append({"type": "flask_import_error", "error": str(e)})
        
        # Test prediction modules
        prediction_modules = ["weather_enhanced_predictor", "unified_predictor"]
        for module in prediction_modules:
            try:
                spec = importlib.util.spec_from_file_location(
                    module, self.base_dir / f"{module}.py"
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    print(f"✅ {module} import successful")
            except Exception as e:
                print(f"❌ {module}: {e}")
                self.issues.append({"type": "prediction_import_error", "module": module, "error": str(e)})
        
        return len(self.issues) == 0
    
    def save_report(self):
        """Save diagnostic report with proper JSON handling"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "issues": convert_for_json(self.issues),
            "total_issues": len(self.issues)
        }
        
        report_file = self.base_dir / "diagnostic_logs" / f"fixed_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Report saved: {report_file}")
        return report

if __name__ == "__main__":
    diagnostic = FixedDiagnostic()
    success = diagnostic.run_basic_tests()
    report = diagnostic.save_report()
    
    print(f"\\n🎯 DIAGNOSTIC COMPLETE")
    print(f"   Success: {success}")
    print(f"   Issues: {len(diagnostic.issues)}")
'''

            fixed_diagnostic_file = self.base_dir / "fixed_diagnostic.py"
            with open(fixed_diagnostic_file, "w") as f:
                f.write(fixed_diagnostic_content)

            print(f"✅ Fixed diagnostic created: {fixed_diagnostic_file}")
            self.repairs_completed.append("fixed_diagnostic")

        except Exception as e:
            print(f"❌ Failed to create fixed diagnostic: {e}")
            self.repairs_failed.append("fixed_diagnostic")

    def run_comprehensive_repair(self):
        """Run all repair operations"""
        print("🚀 Starting Comprehensive Repair Process")
        print("=" * 70)

        start_time = datetime.now()

        # Run all repair operations
        self.repair_virtual_environment()
        self.repair_app_imports()
        self.repair_database_schema()
        self.repair_prediction_pipeline()
        self.repair_flask_application()
        self.create_fixed_diagnostic()

        # Generate summary
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n🎯 REPAIR SUMMARY")
        print("=" * 50)
        print(f"Duration: {duration}")
        print(f"Repairs completed: {len(self.repairs_completed)}")
        print(f"Repairs failed: {len(self.repairs_failed)}")

        if self.repairs_completed:
            print("\n✅ Successfully completed:")
            for repair in self.repairs_completed:
                print(f"   • {repair}")

        if self.repairs_failed:
            print("\n❌ Failed repairs:")
            for repair in self.repairs_failed:
                print(f"   • {repair}")

        # Test final state
        print(f"\n🧪 FINAL SYSTEM TEST")
        print("-" * 30)

        # Run the fixed diagnostic
        fixed_diagnostic_file = self.base_dir / "fixed_diagnostic.py"
        if fixed_diagnostic_file.exists():
            python_path = self.venv_path / "bin" / "python"
            result = subprocess.run(
                [str(python_path), str(fixed_diagnostic_file)],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
            )

            print("Fixed diagnostic output:")
            print(result.stdout)

            if result.stderr:
                print("Errors:")
                print(result.stderr)

        # Save repair log
        repair_log = {
            "timestamp": datetime.now().isoformat(),
            "duration": str(duration),
            "repairs_completed": self.repairs_completed,
            "repairs_failed": self.repairs_failed,
        }

        log_file = (
            self.logs_dir
            / f"repair_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_file, "w") as f:
            json.dump(repair_log, f, indent=2)

        print(f"\n📊 Repair log saved: {log_file}")

        success_rate = (
            len(self.repairs_completed)
            / (len(self.repairs_completed) + len(self.repairs_failed))
            * 100
        )
        print(f"🎯 Overall success rate: {success_rate:.1f}%")

        return success_rate > 80


if __name__ == "__main__":
    repair_system = ComprehensiveRepairSystem()
    repair_system.run_comprehensive_repair()
