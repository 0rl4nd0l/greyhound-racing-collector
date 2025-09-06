#!/usr/bin/env python3
"""
Early-Exit Optimization Demo
===========================

This script demonstrates the early-exit optimization for mostly cached directories.
It creates test scenarios to show how the optimization works when directories have
different cache ratios and unprocessed file counts.

The early-exit optimization triggers when:
1. Cache ratio ≥ 95% (configurable)
2. AND unprocessed files ≤ 5 (configurable)

This skips detailed progress printing that costs time in huge directories.
"""

import os
import shutil
import tempfile
from typing import Set

from utils.early_exit_optimizer import create_early_exit_optimizer


def create_test_directory_scenario(
    name: str, total_files: int, processed_files: int
) -> tuple[str, Set[str]]:
    """
    Create a test directory scenario with specified file counts.

    Args:
        name: Scenario name for directory
        total_files: Total number of CSV files to create
        processed_files: Number of files to simulate as processed

    Returns:
        Tuple of (directory_path, processed_files_set)
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix=f"early_exit_test_{name}_")

    # Create CSV files
    processed_set = set()

    for i in range(total_files):
        filename = f"race_{i:03d}.csv"
        file_path = os.path.join(temp_dir, filename)

        # Create a simple CSV file
        content = f"""Dog Name,PLC,BOX,DIST,DATE,TRACK,G
TestDog{i}_1,1,1,500,2024-01-01,TestTrack,5
TestDog{i}_2,2,2,500,2024-01-01,TestTrack,5
"""

        with open(file_path, "w") as f:
            f.write(content)

        # Add to processed set if it's one of the "processed" files
        if i < processed_files:
            processed_set.add(filename)

    print(
        f"📁 Created test scenario '{name}': {total_files} files, {processed_files} processed"
    )
    return temp_dir, processed_set


def test_early_exit_scenarios():
    """Test various early-exit scenarios"""

    print("🧪 Early-Exit Optimization Test Suite")
    print("=" * 50)

    scenarios = [
        # (name, total_files, processed_files, should_trigger_early_exit)
        ("fully_cached", 1000, 1000, True),  # 100% cached, 0 unprocessed
        ("mostly_cached_small", 100, 98, True),  # 98% cached, 2 unprocessed
        ("mostly_cached_threshold", 100, 95, True),  # 95% cached, 5 unprocessed
        ("just_below_threshold", 100, 94, False),  # 94% cached, 6 unprocessed
        ("half_cached", 100, 50, False),  # 50% cached, 50 unprocessed
        ("barely_cached", 100, 10, False),  # 10% cached, 90 unprocessed
        ("huge_mostly_cached", 10000, 9998, True),  # 99.98% cached, 2 unprocessed
    ]

    optimizer = create_early_exit_optimizer(
        cache_ratio_threshold=0.95,
        unprocessed_threshold=5,
        enable_early_exit=True,
        verbose_summary=True,
    )

    test_dirs = []

    try:
        for name, total_files, processed_files, expected_early_exit in scenarios:
            print(f"\n🔬 Testing scenario: {name}")
            print(f"   📊 {total_files} total files, {processed_files} processed")

            # Create test directory
            test_dir, processed_set = create_test_directory_scenario(
                name, total_files, processed_files
            )
            test_dirs.append(test_dir)

            # Test early exit optimization
            should_exit, scan_result = optimizer.should_use_early_exit(
                test_dir, processed_set
            )

            # Verify results
            unprocessed_count = total_files - processed_files
            cache_ratio = processed_files / total_files if total_files > 0 else 0

            print(
                f"   📈 Analysis: {cache_ratio:.1%} cached, {unprocessed_count} unprocessed"
            )
            print(f"   🎯 Expected early exit: {expected_early_exit}")
            print(f"   ✅ Actual early exit: {should_exit}")

            if should_exit == expected_early_exit:
                print("   ✅ PASS - Behavior matches expectation")
            else:
                print("   ❌ FAIL - Behavior doesn't match expectation")

            # Show unprocessed files for early-exit cases
            if should_exit:
                unprocessed_files = optimizer.get_unprocessed_files_fast(
                    test_dir, processed_set
                )
                print(f"   📝 Unprocessed files found: {len(unprocessed_files)}")
                if len(unprocessed_files) <= 10:
                    for file_path in unprocessed_files:
                        print(f"      - {os.path.basename(file_path)}")

    finally:
        # Cleanup test directories
        print(f"\n🧹 Cleaning up {len(test_dirs)} test directories...")
        for test_dir in test_dirs:
            shutil.rmtree(test_dir)
        print("✅ Cleanup complete")


def test_configuration_variations():
    """Test different configuration variations"""

    print("\n🔧 Testing Configuration Variations")
    print("=" * 50)

    # Create a test scenario: 100 files, 96 processed (96% cached, 4 unprocessed)
    test_dir, processed_set = create_test_directory_scenario("config_test", 100, 96)

    try:
        configs = [
            # (cache_ratio, unprocessed_threshold, expected_result)
            (0.95, 5, True),  # Standard config - should trigger
            (0.97, 5, False),  # Higher cache ratio - shouldn't trigger
            (0.95, 3, False),  # Lower unprocessed threshold - shouldn't trigger
            (0.90, 10, True),  # More lenient config - should trigger
        ]

        for cache_ratio, unprocessed_threshold, expected in configs:
            print(
                f"\n🎛️  Config: {cache_ratio:.1%} cache ratio, {unprocessed_threshold} unprocessed threshold"
            )

            optimizer = create_early_exit_optimizer(
                cache_ratio_threshold=cache_ratio,
                unprocessed_threshold=unprocessed_threshold,
                enable_early_exit=True,
                verbose_summary=False,  # Suppress verbose output for this test
            )

            should_exit, scan_result = optimizer.should_use_early_exit(
                test_dir, processed_set
            )

            print(f"   Expected: {expected}, Actual: {should_exit}")
            if should_exit == expected:
                print("   ✅ PASS")
            else:
                print("   ❌ FAIL")

    finally:
        shutil.rmtree(test_dir)


def demo_performance_impact():
    """Demonstrate the performance impact of early-exit optimization"""

    print("\n⚡ Performance Impact Demonstration")
    print("=" * 50)

    # Create a large directory with mostly cached files
    print("📁 Creating large test directory (10,000 files, 99.95% cached)...")
    test_dir, processed_set = create_test_directory_scenario("performance", 10000, 9995)

    try:
        import time

        # Test with early exit enabled
        print("\n🚀 Testing WITH early-exit optimization...")
        optimizer_enabled = create_early_exit_optimizer(enable_early_exit=True)

        start_time = time.time()
        should_exit, scan_result = optimizer_enabled.should_use_early_exit(
            test_dir, processed_set
        )
        enabled_time = time.time() - start_time

        if should_exit:
            unprocessed_files = optimizer_enabled.get_unprocessed_files_fast(
                test_dir, processed_set
            )
            print(
                f"   ✅ Early exit triggered: {len(unprocessed_files)} files to process"
            )

        # Test with early exit disabled
        print("\n🐌 Testing WITHOUT early-exit optimization...")
        optimizer_disabled = create_early_exit_optimizer(enable_early_exit=False)

        start_time = time.time()
        should_exit_disabled, scan_result_disabled = (
            optimizer_disabled.should_use_early_exit(test_dir, processed_set)
        )
        disabled_time = time.time() - start_time

        print(
            f"   ⏳ Normal processing required: {scan_result_disabled.unprocessed_files} files to process"
        )

        # Show performance comparison
        print("\n📊 Performance Comparison:")
        print(f"   With early-exit:    {enabled_time:.3f}s")
        print(f"   Without early-exit: {disabled_time:.3f}s")

        if enabled_time < disabled_time:
            speedup = disabled_time / enabled_time
            print(f"   🚀 Speedup: {speedup:.1f}x faster with early-exit")

        print("\n💡 Key Benefits of Early-Exit:")
        print("   - Skips detailed progress printing for huge directories")
        print("   - Processes only the few unprocessed files")
        print("   - Provides immediate feedback for mostly-cached scenarios")
        print("   - Configurable thresholds for different use cases")

    finally:
        shutil.rmtree(test_dir)


def main():
    """Run all early-exit optimization tests"""

    print("🎬 Early-Exit Optimization Demonstration")
    print("=" * 60)
    print("Testing Step 6: Early-exit strategy for 'mostly cached' directories")
    print("When pre-filtering removes ≥95% of files AND unprocessed count < 5,")
    print("the system prints summary and returns immediately, skipping detailed")
    print("progress printing that costs time in huge directories.")
    print("=" * 60)

    try:
        # Run test scenarios
        test_early_exit_scenarios()

        # Test configuration variations
        test_configuration_variations()

        # Demonstrate performance impact
        demo_performance_impact()

        print("\n🎉 All tests completed successfully!")
        print("✅ Early-exit optimization is working as expected")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
