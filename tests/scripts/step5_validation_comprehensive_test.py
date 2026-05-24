#!/usr/bin/env python3
"""
Step 5 Comprehensive Validation Test
===================================

This script tests both successful and failing validation cases to demonstrate
that the probability normalization and formatting validation works correctly.

Author: AI Assistant
Date: December 2024
"""

import logging
import sys

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_valid_predictions():
    """Test case with properly normalized predictions."""

    print("=== TEST 1: Valid Predictions (Should Pass) ===")

    predictions = pd.DataFrame(
        {
            "dog_clean_name": ["DOG_A", "DOG_B", "DOG_C"],
            "win_probability": [0.5, 0.3, 0.2],  # Sum = 1.0
        }
    )

    try:
        # Validation assertions from task
        prob_sum = predictions["win_probability"].sum()
        assert abs(prob_sum - 1) < 1e-3, "Probabilities not normalized"
        assert all(
            col in predictions.columns for col in ["dog_clean_name", "win_probability"]
        )

        print(f"✓ PASSED: Probability sum = {prob_sum:.6f}")
        print("✓ PASSED: Required columns present")

        # Log first three rows for manual inspection
        print("\nFirst three rows for manual inspection:")
        for i in range(min(3, len(predictions))):
            row = predictions.iloc[i]
            logger.info(
                f"Row {i+1}: {row['dog_clean_name']} -> {row['win_probability']:.4f}"
            )

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_unnormalized_predictions():
    """Test case with unnormalized predictions (should fail)."""

    print("\n=== TEST 2: Unnormalized Predictions (Should Fail) ===")

    predictions = pd.DataFrame(
        {
            "dog_clean_name": ["DOG_A", "DOG_B", "DOG_C"],
            "win_probability": [0.6, 0.4, 0.3],  # Sum = 1.3 (not normalized!)
        }
    )

    try:
        # Validation assertions from task
        prob_sum = predictions["win_probability"].sum()
        assert abs(prob_sum - 1) < 1e-3, "Probabilities not normalized"
        assert all(
            col in predictions.columns for col in ["dog_clean_name", "win_probability"]
        )

        print("✗ UNEXPECTED: Test should have failed but didn't!")
        return False

    except AssertionError as e:
        print(f"✓ EXPECTED FAILURE: {e}")
        print(
            f"✓ Correctly detected non-normalized probabilities (sum = {prob_sum:.6f})"
        )
        return True
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        return False


def test_missing_columns():
    """Test case with missing required columns (should fail)."""

    print("\n=== TEST 3: Missing Required Columns (Should Fail) ===")

    predictions = pd.DataFrame(
        {
            "dog_name": ["DOG_A", "DOG_B", "DOG_C"],  # Wrong column name!
            "probability": [0.5, 0.3, 0.2],  # Wrong column name!
        }
    )

    try:
        # Validation assertions from task
        prob_sum = predictions[
            "win_probability"
        ].sum()  # This will fail - column doesn't exist
        assert abs(prob_sum - 1) < 1e-3, "Probabilities not normalized"
        assert all(
            col in predictions.columns for col in ["dog_clean_name", "win_probability"]
        )

        print("✗ UNEXPECTED: Test should have failed but didn't!")
        return False

    except KeyError as e:
        print(f"✓ EXPECTED FAILURE: Missing column - {e}")
        return True
    except AssertionError as e:
        print(f"✓ EXPECTED FAILURE: {e}")
        return True
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        return False


def test_edge_case_normalization():
    """Test edge case with normalization error beyond tolerance."""

    print("\n=== TEST 4: Edge Case - Normalization Error Beyond Tolerance ===")

    # Create probabilities that sum to 1.002 (beyond 1e-3 tolerance)
    predictions = pd.DataFrame(
        {
            "dog_clean_name": ["DOG_A", "DOG_B", "DOG_C"],
            "win_probability": [
                0.502,
                0.302,
                0.202,
            ],  # Sum = 1.006 (error = 0.006 > 0.001)
        }
    )

    try:
        # Validation assertions from task
        prob_sum = predictions["win_probability"].sum()
        assert abs(prob_sum - 1) < 1e-3, "Probabilities not normalized"
        assert all(
            col in predictions.columns for col in ["dog_clean_name", "win_probability"]
        )

        print(f"✗ UNEXPECTED: Test should have failed but didn't! Sum = {prob_sum:.6f}")
        return False

    except AssertionError as e:
        print(f"✓ EXPECTED FAILURE: {e}")
        print(
            f"✓ Correctly detected normalization error beyond tolerance (sum = {prob_sum:.6f})"
        )
        return True
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        return False


def test_within_tolerance():
    """Test case with normalization error within tolerance."""

    print("\n=== TEST 5: Within Tolerance - Small Normalization Error ===")

    # Create probabilities that sum to 0.9995 (within tolerance)
    predictions = pd.DataFrame(
        {
            "dog_clean_name": ["DOG_A", "DOG_B", "DOG_C"],
            "win_probability": [0.4998, 0.3000, 0.1997],  # Sum = 0.9995
        }
    )

    try:
        # Validation assertions from task
        prob_sum = predictions["win_probability"].sum()
        assert abs(prob_sum - 1) < 1e-3, "Probabilities not normalized"
        assert all(
            col in predictions.columns for col in ["dog_clean_name", "win_probability"]
        )

        print(
            f"✓ PASSED: Within tolerance (sum = {prob_sum:.6f}, error = {abs(prob_sum-1):.6f})"
        )
        print("✓ PASSED: Required columns present")

        # Log first three rows for manual inspection
        print("\nFirst three rows for manual inspection:")
        for i in range(min(3, len(predictions))):
            row = predictions.iloc[i]
            logger.info(
                f"Row {i+1}: {row['dog_clean_name']} -> {row['win_probability']:.4f}"
            )

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run comprehensive validation tests."""

    print("=== Step 5: Comprehensive Validation Test Suite ===\n")

    test_results = []

    # Run all test cases
    test_results.append(("Valid Predictions", test_valid_predictions()))
    test_results.append(("Unnormalized Predictions", test_unnormalized_predictions()))
    test_results.append(("Missing Columns", test_missing_columns()))
    test_results.append(("Edge Case Normalization", test_edge_case_normalization()))
    test_results.append(("Within Tolerance", test_within_tolerance()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All validation tests completed successfully!")
        print("✓ Probability normalization validation works correctly")
        print("✓ Column validation works correctly")
        print("✓ Tolerance checking works correctly")
    else:
        print("✗ Some tests failed - validation logic needs review")
        sys.exit(1)


if __name__ == "__main__":
    main()
