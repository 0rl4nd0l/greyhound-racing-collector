#!/usr/bin/env python3
"""
Step 7 Comprehensive QA & Regression Tests
==========================================

Runs the complete test suite for Step 7 requirements:
- Sample datasets (normal, header errors, duplicate files, continuation rows)
- Automated tests confirming correct predictions generated once
- Re-runs skip unchanged files unless --force
- Debug mode surfaces counts/errors

Usage:
    python run_batch_tests.py                    # Run all Step 7 tests
    python run_batch_tests.py --smoke-test       # Run manual smoke test
    python run_batch_tests.py --debug            # Run with debug mode enabled
    python run_batch_tests.py --force            # Force re-process all files

Author: AI Assistant
Date: January 2025
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def print_banner():
    """Print test runner banner"""
    print("=" * 70)
    print("🧪 BATCH PREDICTION PIPELINE TEST RUNNER")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")

    required_modules = ["pytest", "pandas", "psutil"]
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - MISSING")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n⚠️  Missing required modules: {', '.join(missing_modules)}")
        print("   Install with: pip install " + " ".join(missing_modules))
        return False

    print("✅ All dependencies available")
    print()
    return True


def check_test_files():
    """Check if test files exist"""
    print("📁 Checking test files...")

    test_files = [
        "test_batch_prediction_edge_cases.py",
        "batch_prediction_pipeline.py",
        "cli_batch_predictor.py",
    ]

    missing_files = []

    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"  ✅ {test_file}")
        else:
            print(f"  ❌ {test_file} - MISSING")
            missing_files.append(test_file)

    if missing_files:
        print(f"\n⚠️  Missing required files: {', '.join(missing_files)}")
        return False

    print("✅ All test files available")
    print()
    return True


def run_tests(test_args, verbose=False):
    """Run pytest with specified arguments"""
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.extend(["-v", "--tb=short"])
    else:
        cmd.extend(["-q"])

    cmd.extend(test_args)

    print(f"🚀 Running: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏸️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Batch Prediction Pipeline Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--edge-cases", action="store_true", help="Run only edge case tests"
    )
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick tests only (skip slow tests)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run with verbose output"
    )
    parser.add_argument(
        "--no-banner", action="store_true", help="Skip banner and dependency checks"
    )
    parser.add_argument("--pattern", "-k", help="Run tests matching pattern")
    parser.add_argument("--marker", "-m", help="Run tests with specific marker")

    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

        # Check dependencies and files
        if not check_dependencies():
            sys.exit(1)

        if not check_test_files():
            sys.exit(1)

    # Build test arguments
    test_args = []

    if args.edge_cases:
        test_args.extend(["-k", "TestBatchPredictionEdgeCases"])
        print("🎯 Running edge case tests only")

    elif args.integration:
        test_args.extend(["-k", "TestBatchPipelineIntegration"])
        print("🔗 Running integration tests only")

    elif args.quick:
        test_args.extend(["-m", "not slow"])
        print("⚡ Running quick tests only")

    else:
        test_args.append("test_batch_prediction_edge_cases.py")
        print("🧪 Running all batch prediction tests")

    if args.pattern:
        test_args.extend(["-k", args.pattern])
        print(f"🔍 Using pattern filter: {args.pattern}")

    if args.marker:
        test_args.extend(["-m", args.marker])
        print(f"🏷️  Using marker filter: {args.marker}")

    print()

    # Run the tests
    success = run_tests(test_args, verbose=args.verbose)

    print()
    print("-" * 50)

    if success:
        print("✅ All tests passed!")
        print("🎉 Batch prediction pipeline is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Check the output above for details")

    print("=" * 70)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏸️  Test runner interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        sys.exit(1)
