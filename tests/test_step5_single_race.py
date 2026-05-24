#!/usr/bin/env python3
"""
Test Step 5: Single Race Probability Conversion Demo
===================================================

Demonstrates how Step 5 converts strength scores to win probabilities
for a simulated single race with realistic dog performance data.

Author: AI Assistant
Date: December 2024
"""

import logging

import numpy as np
import pandas as pd

from step5_probability_converter import ProbabilityConverter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_single_race_demo_data():
    """Create realistic single race data for demonstration."""

    # Simulate a race with 6 dogs with realistic strength scores
    race_data = {
        "dog_name": [
            "Thunder Strike",  # Strong favorite
            "Lightning Bolt",  # Second favorite
            "Fast Eddie",  # Mid-tier contender
            "Racing Ruby",  # Mid-tier contender
            "Steady Sam",  # Outsider but decent
            "Lucky Charm",  # Longshot
        ],
        "normalized_strength_score": [
            85.5,  # Strong favorite
            72.3,  # Second favorite
            58.7,  # Mid-tier
            51.2,  # Mid-tier
            35.8,  # Outsider
            18.4,  # Longshot
        ],
    }

    return pd.DataFrame(race_data)


def main():
    """Demonstrate Step 5 probability conversion for a single race."""
    print("=== Step 5: Single Race Probability Conversion Demo ===\n")

    # Create demo race data
    race_df = create_single_race_demo_data()

    print("Race Participants (with strength scores):")
    print("=" * 50)
    for i, (_, dog) in enumerate(race_df.iterrows()):
        print(
            f"{i+1}. {dog['dog_name']:<15} Score: {dog['normalized_strength_score']:6.2f}"
        )

    # Initialize probability converter
    converter = ProbabilityConverter(temperature=1.0)

    # Test different temperature values to show effect
    temperatures = [0.5, 1.0, 2.0, 5.0]

    print("\nEffect of Temperature Parameter (τ) on Probabilities:")
    print("=" * 80)
    print(f"{'Dog Name':<15} {'Score':<6} ", end="")
    for temp in temperatures:
        print(f"τ={temp:<3} ", end="")
    print()
    print("-" * 80)

    for i, (_, dog) in enumerate(race_df.iterrows()):
        print(f"{dog['dog_name']:<15} {dog['normalized_strength_score']:6.2f} ", end="")

        for temp in temperatures:
            # Calculate probability with this temperature
            scores = race_df["normalized_strength_score"].values
            raw_probs = converter.apply_softmax(scores, temperature=temp)
            smoothed_probs = converter.apply_bayesian_smoothing(raw_probs)
            final_probs = converter.enforce_minimum_probability(smoothed_probs)

            prob_pct = final_probs[i] * 100
            print(f"{prob_pct:5.1f}% ", end="")
        print()

    # Show probability sum verification for each temperature
    print("\nProbability Sum Verification:")
    print("-" * 40)
    for temp in temperatures:
        scores = race_df["normalized_strength_score"].values
        raw_probs = converter.apply_softmax(scores, temperature=temp)
        smoothed_probs = converter.apply_bayesian_smoothing(raw_probs)
        final_probs = converter.enforce_minimum_probability(smoothed_probs)
        prob_sum = np.sum(final_probs)
        print(
            f"τ = {temp}: Sum = {prob_sum:.6f} {'✓' if abs(prob_sum - 1.0) < 1e-10 else '✗'}"
        )

    # Demonstrate full conversion process with optimal temperature
    print("\nFull Conversion Process (with temperature calibration):")
    print("=" * 60)

    # Convert to probabilities (will calibrate temperature automatically)
    probabilities_df = converter.convert_to_probabilities(
        scores_df=race_df, calibrate_temperature=True
    )

    print("\nFinal Race Probabilities:")
    print("=" * 50)
    print(
        f"{'Rank':<4} {'Dog Name':<15} {'Probability':<12} {'Percentage':<10} {'Odds':<10}"
    )
    print("-" * 50)

    for i, (_, row) in enumerate(probabilities_df.iterrows()):
        # Calculate implied odds
        prob = row["final_probability"]
        if prob > 0:
            odds = 1.0 / prob
            odds_str = f"{odds:.1f}:1"
        else:
            odds_str = "∞:1"

        print(
            f"{i+1:<4} {row['dog_name']:<15} {prob:.4f}       {prob*100:5.2f}%     {odds_str:<10}"
        )

    # Verify requirements
    print("\nRequirement Verification:")
    print("-" * 30)

    total_prob = probabilities_df["final_probability"].sum()
    min_prob = probabilities_df["final_probability"].min()

    print(f"✓ Softmax applied with temperature τ = {converter.optimal_temperature:.3f}")
    print(f"✓ Bayesian smoothing applied with α = {converter.bayesian_alpha}")
    print(
        f"✓ Minimum probability floor: {converter.min_probability:.3f} ({converter.min_probability*100:.1f}%)"
    )
    print(
        f"✓ Sum of probabilities: {total_prob:.6f} {'(exactly 1.0)' if abs(total_prob - 1.0) < 1e-10 else '(ERROR!)'}"
    )
    print(f"✓ All probabilities > 0: {min_prob:.4f} > 0 {'✓' if min_prob > 0 else '✗'}")
    print(f"✓ No dog has 0% probability: {'✓' if min_prob > 0 else '✗'}")

    # Show betting market comparison
    print("\nBetting Market Interpretation:")
    print("-" * 35)

    for i, (_, row) in enumerate(probabilities_df.iterrows()):
        prob = row["final_probability"]

        if prob > 0.25:
            category = "Strong Favorite"
        elif prob > 0.15:
            category = "Favorite"
        elif prob > 0.10:
            category = "Contender"
        elif prob > 0.05:
            category = "Outsider"
        else:
            category = "Longshot"

        print(f"{row['dog_name']:<15} {prob*100:5.2f}% - {category}")

    print("\n=== Single Race Probability Conversion Complete ===")


if __name__ == "__main__":
    main()
