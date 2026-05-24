#!/usr/bin/env python3
"""
Unit-level Health Check System for Core ML Modules
==================================================

This script performs health checks for each specified module:
1. Import without side-effects
2. Run available built-in self-tests or lightweight dummy calls  
3. Record success/failure plus exceptions to logs/ml_system_diagnostic.jsonl
"""

import importlib
import inspect
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class ModuleHealthChecker:
    """Handles health checks for ML system modules"""

    def __init__(self):
        self.log_file = Path("logs/ml_system_diagnostic.jsonl")
        self.modules_to_check = [
            "ml_system_v4",
            "prediction_pipeline_v3",
            "comprehensive_prediction_pipeline",
            "unified_predictor",
            "probability_calibrator",
            "ensemble_roi_weighter",
        ]

    def log_result(self, module_name: str, status: str, details: Dict[str, Any]):
        """Log health check result to JSONL file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module_name,
            "status": status,
            "details": details,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def safe_import(self, module_name: str) -> tuple:
        """Safely import a module and return (module, success, error_info)"""
        try:
            module = importlib.import_module(module_name)
            return module, True, None
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
            return None, False, error_info

    def find_test_methods(self, module) -> List[str]:
        """Find potential test/self-check methods in the module"""
        test_methods = []

        # Look for common test method patterns
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and (
                name.startswith("test_")
                or name.startswith("self_test")
                or name.startswith("validate")
                or name.startswith("check_")
                or name == "run_tests"
                or name == "diagnose"
            ):
                test_methods.append(name)

        return test_methods

    def find_classes_with_tests(self, module) -> List[tuple]:
        """Find classes that might have test methods"""
        classes_with_tests = []

        for name in dir(module):
            obj = getattr(module, name)
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                # Look for test methods in the class
                test_methods = []
                for method_name in dir(obj):
                    method = getattr(obj, method_name)
                    if callable(method) and (
                        method_name.startswith("test_")
                        or method_name.startswith("self_test")
                        or method_name.startswith("validate")
                        or method_name == "run_diagnostics"
                    ):
                        test_methods.append(method_name)

                if test_methods:
                    classes_with_tests.append((name, obj, test_methods))

        return classes_with_tests

    def run_lightweight_tests(self, module) -> Dict[str, Any]:
        """Run lightweight tests/dummy calls on the module"""
        results = {
            "module_functions_tested": [],
            "class_tests_run": [],
            "test_results": {},
            "errors": [],
        }

        # Test module-level functions
        test_methods = self.find_test_methods(module)
        for method_name in test_methods:
            try:
                method = getattr(module, method_name)
                # Try calling with no arguments first
                try:
                    result = method()
                    results["module_functions_tested"].append(method_name)
                    results["test_results"][method_name] = {
                        "success": True,
                        "result": str(result)[:200],
                    }
                except TypeError:
                    # Try with empty arguments if it needs parameters
                    try:
                        result = method([])  # Try with empty list
                        results["module_functions_tested"].append(method_name)
                        results["test_results"][method_name] = {
                            "success": True,
                            "result": str(result)[:200],
                        }
                    except:
                        results["test_results"][method_name] = {
                            "success": False,
                            "error": "Parameter mismatch",
                        }
            except Exception as e:
                results["errors"].append(f"{method_name}: {str(e)[:100]}")

        # Test classes with test methods
        classes_with_tests = self.find_classes_with_tests(module)
        for class_name, cls, test_methods in classes_with_tests:
            try:
                # Try to instantiate the class
                try:
                    instance = cls()
                except TypeError:
                    # Skip if constructor requires parameters
                    continue

                for test_method in test_methods:
                    try:
                        method = getattr(instance, test_method)
                        result = method()
                        results["class_tests_run"].append(f"{class_name}.{test_method}")
                        results["test_results"][f"{class_name}.{test_method}"] = {
                            "success": True,
                            "result": str(result)[:200],
                        }
                    except Exception as e:
                        results["errors"].append(
                            f"{class_name}.{test_method}: {str(e)[:100]}"
                        )

            except Exception as e:
                results["errors"].append(f"{class_name} instantiation: {str(e)[:100]}")

        return results

    def check_module_attributes(self, module) -> Dict[str, Any]:
        """Check module attributes and basic structure"""
        attributes = {
            "has_classes": False,
            "has_functions": False,
            "main_classes": [],
            "main_functions": [],
            "module_docstring": getattr(module, "__doc__", None),
            "module_file": getattr(module, "__file__", None),
        }

        for name in dir(module):
            if not name.startswith("_"):
                obj = getattr(module, name)
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    attributes["has_classes"] = True
                    attributes["main_classes"].append(name)
                elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    attributes["has_functions"] = True
                    attributes["main_functions"].append(name)

        return attributes

    def check_single_module(self, module_name: str):
        """Perform complete health check on a single module"""
        print(f"\n=== Checking {module_name} ===")

        # Step 1: Import without side-effects
        module, import_success, import_error = self.safe_import(module_name)

        if not import_success:
            print(f"❌ Import failed: {import_error['error_message']}")
            self.log_result(
                module_name, "IMPORT_FAILED", {"step": "import", "error": import_error}
            )
            return

        print(f"✅ Import successful")

        # Step 2: Check module structure
        module_attrs = self.check_module_attributes(module)
        print(
            f"📋 Module structure: {len(module_attrs['main_classes'])} classes, {len(module_attrs['main_functions'])} functions"
        )

        # Step 3: Run lightweight tests
        test_results = self.run_lightweight_tests(module)

        total_tests = len(test_results["module_functions_tested"]) + len(
            test_results["class_tests_run"]
        )
        errors_count = len(test_results["errors"])

        if total_tests > 0:
            print(f"🧪 Tests run: {total_tests}, Errors: {errors_count}")
        else:
            print(f"⚠️  No built-in tests found")

        # Log comprehensive results
        status = (
            "SUCCESS"
            if import_success and errors_count == 0
            else "PARTIAL_SUCCESS" if import_success else "FAILED"
        )

        self.log_result(
            module_name,
            status,
            {
                "step": "complete_check",
                "import_success": import_success,
                "module_attributes": module_attrs,
                "test_results": test_results,
                "summary": {
                    "total_tests_attempted": total_tests,
                    "errors_encountered": errors_count,
                    "has_test_methods": total_tests > 0,
                },
            },
        )

        print(f"📝 Results logged to {self.log_file}")

    def run_all_checks(self):
        """Run health checks on all specified modules"""
        print("🏥 Starting Unit-level Health Checks for Core ML Modules")
        print("=" * 60)

        # Clear previous log entries for this run
        timestamp = datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "event": "health_check_session_start",
                        "modules_to_check": self.modules_to_check,
                    }
                )
                + "\n"
            )

        success_count = 0
        for module_name in self.modules_to_check:
            try:
                self.check_single_module(module_name)
                success_count += 1
            except Exception as e:
                print(f"❌ Critical error checking {module_name}: {e}")
                self.log_result(
                    module_name,
                    "CRITICAL_ERROR",
                    {
                        "step": "health_check",
                        "error": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    },
                )

        # Summary
        print("\n" + "=" * 60)
        print(
            f"🏁 Health Check Complete: {success_count}/{len(self.modules_to_check)} modules processed successfully"
        )

        with open(self.log_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "health_check_session_complete",
                        "summary": {
                            "total_modules": len(self.modules_to_check),
                            "processed_successfully": success_count,
                            "log_file": str(self.log_file),
                        },
                    }
                )
                + "\n"
            )


def main():
    """Main entry point"""
    checker = ModuleHealthChecker()
    checker.run_all_checks()


if __name__ == "__main__":
    main()
