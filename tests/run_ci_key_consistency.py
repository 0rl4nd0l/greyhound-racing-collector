#!/usr/bin/env python3
"""
CI-Optimized Key Consistency Test Runner
========================================

Lightweight test runner specifically designed for CI environments.
Focuses on fast execution while ensuring comprehensive key consistency checking.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run key consistency tests optimized for CI"""
    print("🔧 CI Key Consistency Tests - Step 6 Implementation")
    print("=" * 50)

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Run fast key consistency tests only
    test_cmd = [
        "python",
        "-m",
        "pytest",
        "tests/test_key_consistency.py",
        "-m",
        "key_consistency and not slow",  # Exclude slow tests
        "-v",
        "--tb=short",
        "--maxfail=3",  # Stop after 3 failures
        "--durations=0",  # Report slowest tests
        "--disable-warnings",  # Reduce noise in CI
        "-x",  # Stop on first failure for faster feedback
    ]

    print("Running command:", " ".join(test_cmd))
    print("-" * 50)

    try:
        result = subprocess.run(test_cmd, timeout=1800)  # 30 minute total timeout

        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("🎉 All key consistency tests passed!")
            print("✅ No KeyErrors detected - prediction layers are consistent")
            print("✅ CI pipeline can proceed safely")
            return 0
        else:
            print("\n" + "=" * 50)
            print("❌ Key consistency tests failed!")
            print("🚨 CRITICAL: Key handling regression detected!")
            print("🛑 CI pipeline blocked - fix key consistency issues before merge")
            return 1

    except subprocess.TimeoutExpired:
        print("\n" + "=" * 50)
        print("⏰ Key consistency tests timed out!")
        print("🚨 Tests took longer than 30 minutes - possible performance regression")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
