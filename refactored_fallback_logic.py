#!/usr/bin/env python3
"""
Example of refactored Flask fallback logic for DOG_NAME_KEY

This demonstrates the refactoring task:
- Simplify multi-key try/except block to use DOG_NAME_KEY
- Remove dead/legacy key checks 
- Add explicit logger warning if key is missing
"""

import logging

from constants import DOG_NAME_KEY

logger = logging.getLogger(__name__)


def refactored_dog_name_extraction(prediction_data):
    """
    Refactored version: Single key access using DOG_NAME_KEY with proper error handling
    """
    try:
        # Simple, direct access using the standardized key
        dog_name = prediction_data[DOG_NAME_KEY]
        return dog_name
    except KeyError:
        # Explicit logger warning for missing key
        logger.warning(
            f"Missing {DOG_NAME_KEY} in prediction data. Available keys: {list(prediction_data.keys())}"
        )
        return None


def original_multi_key_fallback(prediction_data):
    """
    Original problematic pattern (BEFORE refactoring):
    Multi-key try/except blocks that mask bugs and create maintenance issues
    """
    # This is the problematic pattern that needs to be refactored
    try:
        # Legacy key attempt 1
        dog_name = prediction_data["dog_name"]
    except KeyError:
        try:
            # Legacy key attempt 2
            dog_name = prediction_data["Dog_Name"]
        except KeyError:
            try:
                # Legacy key attempt 3
                dog_name = prediction_data["name"]
            except KeyError:
                try:
                    # Legacy key attempt 4
                    dog_name = prediction_data["DOG_NAME"]
                except KeyError:
                    # Silent failure - masks real issues
                    dog_name = None

    return dog_name


def demonstrate_refactoring():
    """Demonstrate the improvement"""

    # Test data examples
    test_cases = [
        {"dog_name": "FAST HOUND"},  # Standard case
        {"Dog_Name": "SPEEDY PUP"},  # Legacy case
        {"name": "QUICK DOG"},  # Another legacy case
        {"other_field": "value"},  # Missing key case
    ]

    print("=== REFACTORING DEMONSTRATION ===\n")

    for i, test_data in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_data}")

        # Original approach (problematic)
        original_result = original_multi_key_fallback(test_data)
        print(f"  Original result: {original_result}")

        # Refactored approach (improved)
        refactored_result = refactored_dog_name_extraction(test_data)
        print(f"  Refactored result: {refactored_result}")

        print()


if __name__ == "__main__":
    # Configure logging to show warnings
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    demonstrate_refactoring()

    print("\n=== REFACTORING BENEFITS ===")
    print("✅ Single key access using standardized DOG_NAME_KEY")
    print("✅ Explicit error logging for missing keys")
    print("✅ No masking of real data structure issues")
    print("✅ Easier to debug and maintain")
    print("✅ Forces upstream code to use correct key format")
