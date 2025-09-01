#!/usr/bin/env python3
"""
Step 5: Convert Strength Scores to Win Probabilities
===================================================

Apply softmax: Pᵢ = exp(Sᵢ/τ) / Σ exp(Sⱼ/τ), with temperature τ tuned from historical race outcomes for calibration.
Add Bayesian smoothing so outsiders never have 0%.
Ensure ΣPᵢ = 1 (100%).

Author: AI Assistant
Date: December 2024
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ProbabilityConverter:
    """
    Convert strength scores to calibrated win probabilities using softmax with temperature scaling.

    Includes Bayesian smoothing to ensure no dog has 0% probability and proper normalization
    to ensure probabilities sum to 100%.
    """

    def __init__(self, strength_scores_file: str = None, temperature: float = 1.0):
        """
        Initialize the probability converter.

        Args:
            strength_scores_file: Path to CSV file with strength scores from Step 4
            temperature: Initial temperature parameter for softmax scaling
        """
        self.strength_scores_file = strength_scores_file
        self.temperature = temperature
        self.optimal_temperature = temperature
        self.bayesian_alpha = 1.0  # Smoothing parameter
        self.min_probability = 0.001  # Minimum probability floor (0.1%)

        logger.info(f"ProbabilityConverter initialized with temperature={temperature}")

    def load_strength_scores(self) -> pd.DataFrame:
        """Load strength scores from Step 4 output."""
        if self.strength_scores_file is None:
            # Look for most recent strength scores file
            files = [
                f for f in os.listdir(".") if f.startswith("step4_strength_scores_")
            ]
            if not files:
                raise FileNotFoundError(
                    "No strength scores file found. Run Step 4 first."
                )

            # Get most recent file
            self.strength_scores_file = max(files, key=os.path.getctime)
            logger.info(
                f"Using most recent strength scores file: {self.strength_scores_file}"
            )

        try:
            scores_df = pd.read_csv(self.strength_scores_file)
            logger.info(f"Loaded strength scores for {len(scores_df)} dogs")
            return scores_df
        except Exception as e:
            logger.error(f"Error loading strength scores file: {e}")
            raise

    def apply_softmax(
        self, scores: np.ndarray, temperature: float = None
    ) -> np.ndarray:
        """
        Apply softmax transformation with temperature scaling.

        Args:
            scores: Array of strength scores
            temperature: Temperature parameter (tau). Higher = more uniform, lower = more peaked

        Returns:
            Array of probabilities that sum to 1.0
        """
        if temperature is None:
            temperature = self.temperature

        # Apply temperature scaling
        scaled_scores = scores / temperature

        # Numerical stability: subtract max to prevent overflow
        scaled_scores = scaled_scores - np.max(scaled_scores)

        # Apply exponential
        exp_scores = np.exp(scaled_scores)

        # Normalize to get probabilities
        probabilities = exp_scores / np.sum(exp_scores)

        return probabilities

    def apply_bayesian_smoothing(
        self, probabilities: np.ndarray, alpha: float = None
    ) -> np.ndarray:
        """
        Apply Bayesian smoothing to ensure no probability is exactly 0.

        Args:
            probabilities: Array of raw probabilities
            alpha: Smoothing parameter (higher = more smoothing)

        Returns:
            Smoothed probabilities that sum to 1.0
        """
        if alpha is None:
            alpha = self.bayesian_alpha

        # Add alpha to each probability (uniform prior)
        smoothed = probabilities + alpha

        # Renormalize to ensure sum = 1
        smoothed = smoothed / np.sum(smoothed)

        return smoothed

    def enforce_minimum_probability(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Enforce minimum probability floor for outsiders.

        Args:
            probabilities: Array of probabilities

        Returns:
            Adjusted probabilities with minimum floor applied
        """
        # Apply minimum probability floor
        adjusted = np.maximum(probabilities, self.min_probability)

        # Renormalize to ensure sum = 1
        adjusted = adjusted / np.sum(adjusted)

        return adjusted

    def create_synthetic_race_outcomes(
        self, scores_df: pd.DataFrame, n_races: int = 1000
    ) -> List[Dict]:
        """
        Create synthetic race outcomes for temperature calibration.

        This simulates historical race data based on strength scores to tune temperature.
        In practice, this would use actual historical race results.

        Args:
            scores_df: DataFrame with strength scores
            n_races: Number of synthetic races to generate

        Returns:
            List of race outcome dictionaries
        """
        logger.info(f"Generating {n_races} synthetic races for temperature calibration")

        races = []
        np.random.seed(42)  # For reproducibility

        for race_id in range(n_races):
            # Randomly select dogs for each race (limited by available population)
            max_race_size = min(8, len(scores_df))
            min_race_size = min(4, len(scores_df))
            race_size = np.random.randint(min_race_size, max_race_size + 1)
            selected_dogs = scores_df.sample(
                n=race_size, replace=False if race_size <= len(scores_df) else True
            )

            # Convert scores to win probabilities using softmax (with noise)
            scores = selected_dogs["normalized_strength_score"].values
            noise = np.random.normal(0, 2, len(scores))  # Add some randomness
            noisy_scores = scores + noise

            # Apply softmax with temperature = 2.0 (for generating synthetic data)
            probabilities = self.apply_softmax(noisy_scores, temperature=2.0)

            # Select winner based on probabilities
            winner_idx = np.random.choice(len(probabilities), p=probabilities)

            race_data = {
                "race_id": race_id,
                "dogs": selected_dogs["dog_name"].tolist(),
                "scores": scores.tolist(),
                "winner": selected_dogs.iloc[winner_idx]["dog_name"],
                "winner_score": scores[winner_idx],
            }
            races.append(race_data)

        return races

    def calibrate_temperature(
        self, scores_df: pd.DataFrame, use_synthetic: bool = True
    ) -> float:
        """
        Calibrate temperature parameter using historical race outcomes.

        Args:
            scores_df: DataFrame with strength scores
            use_synthetic: Whether to use synthetic data (True) or load real historical data

        Returns:
            Optimal temperature parameter
        """
        logger.info("Starting temperature calibration...")

        if use_synthetic:
            # Generate synthetic race data for calibration
            races = self.create_synthetic_race_outcomes(scores_df, n_races=500)

            def temperature_loss(temperature):
                """Calculate log-loss for given temperature on synthetic races."""
                total_log_loss = 0
                valid_races = 0

                for race in races:
                    if len(race["dogs"]) < 2:
                        continue

                    # Get scores for dogs in this race
                    race_scores = np.array(race["scores"])

                    # Calculate probabilities with this temperature
                    probabilities = self.apply_softmax(race_scores, temperature)
                    probabilities = self.apply_bayesian_smoothing(probabilities)
                    probabilities = self.enforce_minimum_probability(probabilities)

                    # Create binary outcome vector (1 for winner, 0 for others)
                    winner_name = race["winner"]
                    outcome = np.array(
                        [1 if dog == winner_name else 0 for dog in race["dogs"]]
                    )

                    # Calculate log loss for this race
                    # Clip probabilities to avoid log(0)
                    prob_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
                    race_log_loss = -np.sum(outcome * np.log(prob_clipped))

                    total_log_loss += race_log_loss
                    valid_races += 1

                return total_log_loss / valid_races if valid_races > 0 else float("inf")

            # Optimize temperature using golden section search
            result = minimize_scalar(
                temperature_loss, bounds=(0.1, 10.0), method="bounded"
            )
            optimal_temp = result.x

            logger.info(
                f"Temperature calibration complete: optimal τ = {optimal_temp:.3f}"
            )
            logger.info(f"Final log-loss: {result.fun:.4f}")

        else:
            # TODO: Load real historical race data for calibration
            logger.warning(
                "Real historical data calibration not implemented yet. Using default temperature."
            )
            optimal_temp = 2.0

        self.optimal_temperature = optimal_temp
        return optimal_temp

    def convert_to_probabilities(
        self, scores_df: pd.DataFrame = None, calibrate_temperature: bool = True
    ) -> pd.DataFrame:
        """
        Convert strength scores to calibrated win probabilities.

        Args:
            scores_df: DataFrame with strength scores (if None, loads from file)
            calibrate_temperature: Whether to calibrate temperature parameter

        Returns:
            DataFrame with dogs and their win probabilities
        """
        # Load strength scores if not provided
        if scores_df is None:
            scores_df = self.load_strength_scores()

        # Calibrate temperature if requested
        if calibrate_temperature:
            self.calibrate_temperature(scores_df)

        # Extract strength scores
        scores = scores_df["normalized_strength_score"].values

        # Apply softmax with optimal temperature
        logger.info(
            f"Applying softmax with temperature τ = {self.optimal_temperature:.3f}"
        )
        raw_probabilities = self.apply_softmax(scores, self.optimal_temperature)

        # Apply Bayesian smoothing
        logger.info(f"Applying Bayesian smoothing with α = {self.bayesian_alpha}")
        smoothed_probabilities = self.apply_bayesian_smoothing(raw_probabilities)

        # Enforce minimum probability floor
        final_probabilities = self.enforce_minimum_probability(smoothed_probabilities)

        # Verify normalization
        prob_sum = np.sum(final_probabilities)
        logger.info(f"Probability sum: {prob_sum:.6f} (should be 1.000000)")

        if abs(prob_sum - 1.0) > 1e-10:
            logger.warning(f"Probabilities don't sum to 1.0! Sum = {prob_sum}")
            # Force normalization
            final_probabilities = final_probabilities / prob_sum

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "dog_name": scores_df["dog_name"],
                "strength_score": scores,
                "raw_probability": raw_probabilities,
                "smoothed_probability": smoothed_probabilities,
                "final_probability": final_probabilities,
                "win_percentage": final_probabilities * 100,
            }
        )

        # Sort by probability (descending)
        results_df = results_df.sort_values("final_probability", ascending=False)
        results_df["probability_rank"] = range(1, len(results_df) + 1)

        logger.info(f"Converted {len(results_df)} strength scores to win probabilities")
        logger.info(
            f"Probability range: {final_probabilities.min():.4f} - {final_probabilities.max():.4f}"
        )
        logger.info(
            f"Percentage range: {final_probabilities.min()*100:.2f}% - {final_probabilities.max()*100:.2f}%"
        )

        return results_df

    def analyze_probability_distribution(self, probabilities_df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of win probabilities.

        Args:
            probabilities_df: DataFrame with win probabilities

        Returns:
            Dictionary with distribution statistics
        """
        probs = probabilities_df["final_probability"].values

        analysis = {
            "total_dogs": len(probs),
            "probability_sum": np.sum(probs),
            "mean_probability": np.mean(probs),
            "std_probability": np.std(probs),
            "min_probability": np.min(probs),
            "max_probability": np.max(probs),
            "entropy": -np.sum(probs * np.log(probs + 1e-15)),  # Shannon entropy
            "gini_coefficient": self._calculate_gini(probs),
            "favorites_count": np.sum(probs > 0.1),  # Dogs with >10% chance
            "longshots_count": np.sum(probs < 0.01),  # Dogs with <1% chance
        }

        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            analysis[f"p{p}_probability"] = np.percentile(probs, p)

        return analysis

    def _calculate_gini(self, probabilities: np.ndarray) -> float:
        """Calculate Gini coefficient for probability distribution."""
        sorted_probs = np.sort(probabilities)
        n = len(sorted_probs)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_probs)) / (n * np.sum(sorted_probs)) - (
            n + 1
        ) / n

    def simulate_race_outcomes(
        self, probabilities_df: pd.DataFrame, n_simulations: int = 10000
    ) -> Dict:
        """
        Simulate race outcomes to validate probability calibration.

        Args:
            probabilities_df: DataFrame with win probabilities
            n_simulations: Number of race simulations to run

        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running {n_simulations} race simulations for validation")

        probabilities = probabilities_df["final_probability"].values
        dog_names = probabilities_df["dog_name"].values

        # Run simulations
        winners = []
        np.random.seed(42)

        for _ in range(n_simulations):
            winner_idx = np.random.choice(len(probabilities), p=probabilities)
            winners.append(dog_names[winner_idx])

        # Calculate empirical win rates
        winner_counts = pd.Series(winners).value_counts()
        empirical_rates = winner_counts / n_simulations

        # Compare with predicted probabilities
        comparison_data = []
        for dog_name in dog_names:
            predicted_prob = probabilities_df[probabilities_df["dog_name"] == dog_name][
                "final_probability"
            ].iloc[0]
            empirical_rate = empirical_rates.get(dog_name, 0.0)

            comparison_data.append(
                {
                    "dog_name": dog_name,
                    "predicted_probability": predicted_prob,
                    "empirical_rate": empirical_rate,
                    "difference": abs(predicted_prob - empirical_rate),
                    "relative_error": (
                        abs(predicted_prob - empirical_rate) / predicted_prob
                        if predicted_prob > 0
                        else 0
                    ),
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Calculate calibration metrics
        mean_absolute_error = comparison_df["difference"].mean()
        mean_relative_error = comparison_df["relative_error"].mean()

        results = {
            "n_simulations": n_simulations,
            "mean_absolute_error": mean_absolute_error,
            "mean_relative_error": mean_relative_error,
            "max_difference": comparison_df["difference"].max(),
            "comparison_data": comparison_df,
        }

        logger.info(f"Simulation validation complete:")
        logger.info(f"  Mean Absolute Error: {mean_absolute_error:.4f}")
        logger.info(f"  Mean Relative Error: {mean_relative_error:.2%}")

        return results

    def plot_probability_distribution(
        self, probabilities_df: pd.DataFrame, save_path: str = None
    ):
        """
        Create visualization of probability distribution.

        Args:
            probabilities_df: DataFrame with win probabilities
            save_path: Path to save plot (if None, displays plot)
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            probs = probabilities_df["final_probability"].values

            # 1. Histogram of probabilities
            ax1.hist(probs, bins=50, alpha=0.7, color="blue", edgecolor="black")
            ax1.set_xlabel("Win Probability")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Distribution of Win Probabilities")
            ax1.grid(True, alpha=0.3)

            # 2. Top 20 dogs
            top_20 = probabilities_df.head(20)
            ax2.barh(
                range(len(top_20)), top_20["win_percentage"], color="green", alpha=0.7
            )
            ax2.set_yticks(range(len(top_20)))
            ax2.set_yticklabels(top_20["dog_name"], fontsize=8)
            ax2.set_xlabel("Win Percentage (%)")
            ax2.set_title("Top 20 Dogs by Win Probability")
            ax2.grid(True, alpha=0.3)

            # 3. Probability vs Rank
            ax3.plot(
                probabilities_df["probability_rank"],
                probabilities_df["win_percentage"],
                "o-",
                markersize=3,
                alpha=0.6,
            )
            ax3.set_xlabel("Probability Rank")
            ax3.set_ylabel("Win Percentage (%)")
            ax3.set_title("Win Percentage vs Rank")
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale("log")

            # 4. Cumulative distribution
            sorted_probs = np.sort(probs)[::-1]  # Descending order
            cumsum = np.cumsum(sorted_probs)
            ax4.plot(range(1, len(cumsum) + 1), cumsum * 100, "r-", linewidth=2)
            ax4.set_xlabel("Number of Top Dogs")
            ax4.set_ylabel("Cumulative Probability (%)")
            ax4.set_title("Cumulative Probability Distribution")
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=50, color="black", linestyle="--", alpha=0.5, label="50%")
            ax4.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Probability distribution plot saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.warning(f"Could not create plot: {e}")

    def save_parameters(self, filepath: str = None) -> str:
        """Save calibrated parameters to disk."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"step5_probability_parameters_{timestamp}.pkl"

        parameters = {
            "optimal_temperature": self.optimal_temperature,
            "bayesian_alpha": self.bayesian_alpha,
            "min_probability": self.min_probability,
            "strength_scores_file": self.strength_scores_file,
        }

        joblib.dump(parameters, filepath)
        logger.info(f"Probability conversion parameters saved to {filepath}")
        return filepath

    def load_parameters(self, filepath: str):
        """Load calibrated parameters from disk."""
        parameters = joblib.load(filepath)

        self.optimal_temperature = parameters.get("optimal_temperature", 1.0)
        self.bayesian_alpha = parameters.get("bayesian_alpha", 1.0)
        self.min_probability = parameters.get("min_probability", 0.001)
        self.strength_scores_file = parameters.get("strength_scores_file")

        logger.info(f"Probability conversion parameters loaded from {filepath}")


def main():
    """Main function to demonstrate probability conversion."""
    print("=== Greyhound Probability Converter (Step 5) ===\n")

    try:
        # Initialize converter
        converter = ProbabilityConverter()

        # Convert strength scores to probabilities
        print("Converting strength scores to win probabilities...")
        probabilities_df = converter.convert_to_probabilities(
            calibrate_temperature=True
        )

        # Display top 20 dogs
        print("\nTop 20 Dogs by Win Probability:")
        print("=" * 70)
        for i, (_, row) in enumerate(probabilities_df.head(20).iterrows()):
            print(
                f"{i+1:2d}. {row['dog_name']:<20} "
                f"Prob: {row['final_probability']:.4f} "
                f"({row['win_percentage']:5.2f}%) "
                f"Score: {row['strength_score']:6.2f}"
            )

        # Analyze probability distribution
        analysis = converter.analyze_probability_distribution(probabilities_df)

        print(f"\nProbability Distribution Analysis:")
        print("=" * 40)
        print(f"Total Dogs: {analysis['total_dogs']}")
        print(f"Probability Sum: {analysis['probability_sum']:.6f}")
        print(
            f"Mean Probability: {analysis['mean_probability']:.4f} ({analysis['mean_probability']*100:.2f}%)"
        )
        print(
            f"Min Probability: {analysis['min_probability']:.4f} ({analysis['min_probability']*100:.2f}%)"
        )
        print(
            f"Max Probability: {analysis['max_probability']:.4f} ({analysis['max_probability']*100:.2f}%)"
        )
        print(f"Shannon Entropy: {analysis['entropy']:.3f}")
        print(f"Gini Coefficient: {analysis['gini_coefficient']:.3f}")
        print(f"Favorites (>10%): {analysis['favorites_count']}")
        print(f"Longshots (<1%): {analysis['longshots_count']}")

        # Simulate races for validation
        print("\nValidating probabilities with race simulations...")
        simulation_results = converter.simulate_race_outcomes(
            probabilities_df, n_simulations=10000
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save probabilities
        output_file = f"step5_win_probabilities_{timestamp}.csv"
        probabilities_df.to_csv(output_file, index=False)
        print(f"\nWin probabilities saved to: {output_file}")

        # Save parameters
        params_file = converter.save_parameters()
        print(f"Calibration parameters saved to: {params_file}")

        # Create and save plot
        plot_file = f"step5_probability_distribution_{timestamp}.png"
        converter.plot_probability_distribution(probabilities_df, save_path=plot_file)

        # Final verification
        print(f"\nFinal Verification:")
        print(f"Temperature (τ): {converter.optimal_temperature:.3f}")
        print(f"Bayesian Smoothing (α): {converter.bayesian_alpha}")
        print(
            f"Minimum Probability Floor: {converter.min_probability:.3f} ({converter.min_probability*100:.1f}%)"
        )
        print(
            f"Sum of all probabilities: {probabilities_df['final_probability'].sum():.6f}"
        )

        if abs(probabilities_df["final_probability"].sum() - 1.0) < 1e-10:
            print("✓ Probabilities correctly sum to 1.0 (100%)")
        else:
            print("✗ ERROR: Probabilities do not sum to 1.0!")

        if probabilities_df["final_probability"].min() > 0:
            print("✓ All probabilities are positive (no 0% probabilities)")
        else:
            print("✗ ERROR: Some probabilities are 0!")

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in main(): {e}")

    print("\n=== Probability Conversion Complete ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
