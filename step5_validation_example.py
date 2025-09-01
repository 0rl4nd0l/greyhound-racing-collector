#!/usr/bin/env python3
"""
Step 5 Validation Example - Exact Task Requirements
===================================================

This script demonstrates the exact validation code mentioned in the task:

After predictions:  
```python
prob_sum = predictions['win_probability'].sum()  
assert abs(prob_sum - 1) < 1e-3, "Probabilities not normalized"  
assert all(col in predictions.columns for col in ["dog_clean_name","win_probability"])
```  
Logs first three rows for manual inspection.

Author: AI Assistant  
Date: December 2024
"""

import logging

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_predictions():
    """Create sample predictions data for demonstration."""

    # Sample predictions data with proper format
    predictions = pd.DataFrame(
        {
            "dog_clean_name": [
                "THUNDER STRIKE",
                "LIGHTNING BOLT",
                "FAST EDDIE",
                "RACING RUBY",
                "STEADY SAM",
            ],
            "win_probability": [
                0.35,  # 35%
                0.25,  # 25%
                0.20,  # 20%
                0.12,  # 12%
                0.08,  # 8%
            ],
            "box_number": [1, 2, 3, 4, 5],
            "predicted_rank": [1, 2, 3, 4, 5],
        }
    )

    return predictions


def main():
    """Main function demonstrating the exact validation requirements."""

    print("=== Step 5 Validation Example ===\n")

    # Create sample predictions (in practice, this would be loaded from your prediction system)
    predictions = create_sample_predictions()

    print("Sample Predictions Data:")
    print(predictions.to_string(index=False))
    print()

    # =================================================================
    # EXACT CODE FROM TASK REQUIREMENTS:
    # =================================================================

    print("Running validation assertions...")

    # Validate probability normalization
    prob_sum = predictions["win_probability"].sum()
    assert abs(prob_sum - 1) < 1e-3, "Probabilities not normalized"
    print(f"✓ Probability normalization check PASSED: sum = {prob_sum:.6f}")

    # Validate required columns exist
    assert all(
        col in predictions.columns for col in ["dog_clean_name", "win_probability"]
    )
    print("✓ Required columns check PASSED: dog_clean_name, win_probability")

    # Log first three rows for manual inspection
    print("\nFirst three rows for manual inspection:")
    print("=" * 50)

    for i in range(min(3, len(predictions))):
        row = predictions.iloc[i]
        logger.info(
            f"Row {i+1}: {row['dog_clean_name']} -> Win Probability: {row['win_probability']:.4f} ({row['win_probability']*100:.2f}%)"
        )

    print("\n✓ All validation requirements completed successfully!")
    print("✓ Probabilities sum to 1.0 within tolerance (±0.001)")
    print("✓ Required columns present")
    print("✓ First three rows logged for manual inspection")


if __name__ == "__main__":
    main()
