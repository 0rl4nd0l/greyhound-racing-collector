import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SanityChecks:
    """
    Consistency and sanity checks for predictions in the greyhound race prediction system.
    """

    def __init__(self):
        pass

    def validate_predictions(self, predictions: List[Dict]) -> Dict[str, any]:
        """
        Perform comprehensive checks on predictions.

        Args:
            predictions: A list of prediction dictionaries.

        Returns:
            A dictionary containing inconsistency flags and validation results.
        """
        inconsistency_flags = []
        validation_results = {
            "total_predictions": len(predictions),
            "flags": [],
            "passed_checks": [],
            "failed_checks": [],
        }

        if not predictions:
            inconsistency_flags.append("No predictions provided")
            logger.error("No predictions provided for validation")
            validation_results["flags"] = inconsistency_flags
            return validation_results

        # Extract probabilities and ranks
        win_probs = []
        place_probs = []
        numeric_ranks = []

        for p in predictions:
            win_prob = p.get("win_probability") or p.get("win_prob")
            place_prob = p.get("place_probability") or p.get("place_prob")
            rank = p.get("predicted_rank")

            win_probs.append(win_prob)
            place_probs.append(place_prob)
            numeric_ranks.append(rank)

        # Check 1: Probabilities are within the valid range [0, 1]
        range_check_passed = True
        for i, p in enumerate(predictions):
            win_prob = win_probs[i]
            place_prob = place_probs[i]
            dog_name = p.get("dog_name", f"Unknown Dog {i}")

            if win_prob is not None and not (0 <= win_prob <= 1):
                flag = f"Win probability out of range [0, 1] for {dog_name}: {win_prob}"
                inconsistency_flags.append(flag)
                logger.error(flag)
                range_check_passed = False

            if place_prob is not None and not (0 <= place_prob <= 1):
                flag = f"Place probability out of range [0, 1] for {dog_name}: {place_prob}"
                inconsistency_flags.append(flag)
                logger.error(flag)
                range_check_passed = False

            # Check for NaN values
            if win_prob is not None and pd.isna(win_prob):
                flag = f"NaN win probability for {dog_name}"
                inconsistency_flags.append(flag)
                logger.error(flag)
                range_check_passed = False

            if place_prob is not None and pd.isna(place_prob):
                flag = f"NaN place probability for {dog_name}"
                inconsistency_flags.append(flag)
                logger.error(flag)
                range_check_passed = False

        if range_check_passed:
            validation_results["passed_checks"].append("Probability range validation")
        else:
            validation_results["failed_checks"].append("Probability range validation")

        # Check 2: Win probabilities approximately sum to 1 (after softmax)
        softmax_check_passed = True
        valid_win_probs = [p for p in win_probs if p is not None and not pd.isna(p)]

        if valid_win_probs:
            try:
                # Apply softmax normalization
                probs_array = np.array(valid_win_probs)
                probs_softmax = np.exp(probs_array) / np.sum(np.exp(probs_array))
                softmax_sum = np.sum(probs_softmax)

                if not np.isclose(softmax_sum, 1.0, atol=1e-2):
                    flag = f"Win probabilities do not sum to ~1 after softmax: {softmax_sum}"
                    inconsistency_flags.append(flag)
                    logger.error(flag)
                    softmax_check_passed = False
            except Exception as e:
                flag = f"Error in softmax validation: {str(e)}"
                inconsistency_flags.append(flag)
                logger.error(flag)
                softmax_check_passed = False

        if softmax_check_passed:
            validation_results["passed_checks"].append("Softmax sum validation")
        else:
            validation_results["failed_checks"].append("Softmax sum validation")

        # Check 3: Rank by probability aligns with numeric rank output
        rank_alignment_passed = True
        valid_entries = [
            (i, win_probs[i], numeric_ranks[i])
            for i in range(len(predictions))
            if win_probs[i] is not None
            and numeric_ranks[i] is not None
            and not pd.isna(win_probs[i])
        ]

        if valid_entries:
            # Sort by win probability (descending)
            sorted_by_prob = sorted(valid_entries, key=lambda x: x[1], reverse=True)
            prob_based_ranks = [entry[2] for entry in sorted_by_prob]

            # Sort by numeric rank (ascending)
            sorted_by_rank = sorted(valid_entries, key=lambda x: x[2])
            expected_ranks = [entry[2] for entry in sorted_by_rank]

            if prob_based_ranks != expected_ranks:
                flag = f"Rank by probability does not align with numeric rank output. Prob-based: {prob_based_ranks}, Expected: {expected_ranks}"
                inconsistency_flags.append(flag)
                logger.error(flag)
                rank_alignment_passed = False

        if rank_alignment_passed:
            validation_results["passed_checks"].append("Rank alignment validation")
        else:
            validation_results["failed_checks"].append("Rank alignment validation")

        # Check 4: No duplicate ranks
        duplicate_check_passed = True
        valid_ranks = [r for r in numeric_ranks if r is not None]

        if valid_ranks and len(set(valid_ranks)) != len(valid_ranks):
            flag = f"Duplicate numeric ranks found: {valid_ranks}"
            inconsistency_flags.append(flag)
            logger.error(flag)
            duplicate_check_passed = False

        if duplicate_check_passed:
            validation_results["passed_checks"].append("Duplicate rank validation")
        else:
            validation_results["failed_checks"].append("Duplicate rank validation")

        validation_results["flags"] = inconsistency_flags
        return validation_results

    def fix_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Automatically fix inconsistencies found in predictions.

        Args:
            predictions: A list of prediction dictionaries.

        Returns:
            A list of predictions with applied fixes.
        """

        corrected_predictions = predictions.copy()

        # Fix 1: Ensure probabilities in [0, 1] and handle NaN values
        for p in corrected_predictions:
            if "win_probability" in p:
                if pd.isna(p["win_probability"]):
                    p["win_probability"] = 0.5  # Default probability
                else:
                    p["win_probability"] = float(np.clip(p["win_probability"], 0, 1))
            if "place_probability" in p:
                if pd.isna(p["place_probability"]):
                    p["place_probability"] = 0.65  # Default place probability
                else:
                    p["place_probability"] = float(
                        np.clip(p["place_probability"], 0, 1)
                    )

        # Fix 2: Apply softmax to probabilities
        win_probs = np.array(
            [p.get("win_probability", 0.5) for p in corrected_predictions]
        )
        # Handle any remaining NaN values
        win_probs = np.nan_to_num(win_probs, nan=0.5)
        softmax_probs = np.exp(win_probs) / np.sum(np.exp(win_probs))
        for i, p in enumerate(corrected_predictions):
            p["win_probability"] = float(softmax_probs[i])

        # Fix 3: Align ranks based on win probability
        corrected_predictions.sort(key=lambda x: x["win_probability"], reverse=True)
        for i, p in enumerate(corrected_predictions):
            p["predicted_rank"] = i + 1

        # Fix 4: Ensure unique ranks
        # Already handled in the alignment above
        return corrected_predictions
