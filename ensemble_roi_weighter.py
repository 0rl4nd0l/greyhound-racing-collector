#!/usr/bin/env python3
"""
Ensemble ROI Weighter
====================

This utility computes ROI-optimized weights for ensemble models by:
1. Loading historical race predictions and realized outcomes from the database
2. Running constrained optimization to maximize historical ROI subject to ∑w=1, w≥0
3. Returning optimized weights for use in VotingClassifier

The optimization maximizes the expected profitability of the ensemble by finding
the optimal linear combination of base model predictions.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class EnsembleROIWeighter:
    """
    Compute ROI-optimized weights for ensemble models based on historical performance.
    """

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.model_names = []

    def load_historical_data(
        self, limit_records: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load historical predictions and realized outcomes from the database.

        Returns:
            X: DataFrame with predictions from different sources (columns) for each race/dog (rows)
            y: Series indicating whether each prediction was correct (1) or not (0)
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Query to get predictions with their outcomes
            query = """
            SELECT 
                p.race_id,
                p.dog_clean_name,
                p.prediction_source,
                p.predicted_probability,
                CASE 
                    WHEN rm.winner_name = p.dog_clean_name THEN 1 
                    ELSE 0 
                END as won
            FROM predictions p
            JOIN race_metadata rm ON p.race_id = rm.race_id
            WHERE p.predicted_probability IS NOT NULL 
              AND rm.winner_name IS NOT NULL
              AND p.predicted_probability > 0
            ORDER BY p.timestamp DESC
            """

            if limit_records:
                query += f" LIMIT {limit_records}"

            logger.info(f"Loading historical data from {self.db_path}...")
            data = pd.read_sql_query(query, conn)
            conn.close()

            if data.empty:
                logger.info("No historical prediction data found in database")
                logger.info(
                    "This is expected for new systems without prediction history"
                )
                logger.info("ROI optimization will be enabled once predictions are stored")
                raise ValueError(
                    "No historical prediction data - equal weights fallback"
                )

            logger.info(f"Loaded {len(data)} prediction records")

            # Create a unique identifier for each prediction (race + dog)
            data["prediction_id"] = data["race_id"] + "_" + data["dog_clean_name"]

            # Pivot to get predictions from different sources as columns
            X = data.pivot(
                index="prediction_id",
                columns="prediction_source",
                values="predicted_probability",
            )

            # Get the outcomes
            outcomes = data.set_index("prediction_id")["won"]
            y = outcomes.groupby("prediction_id").first()  # Remove duplicates if any

            # Align X and y indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            # Fill NaN values with 0.5 (neutral probability) for missing predictions
            X = X.fillna(0.5)

            # Store model names for later use
            self.model_names = list(X.columns)

            logger.info(f"Final dataset shape: {X.shape}, Models: {self.model_names}")
            logger.info(f"Win rate in dataset: {y.mean():.3f}")

            return X, y

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise

    def compute_weights(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Compute ensemble weights to maximize ROI using constrained optimization.

        The objective function maximizes expected profit by finding optimal weights
        that best predict winners with higher probabilities.
        """
        num_models = X.shape[1]

        if num_models == 0:
            raise ValueError("No models to optimize weights for")

        def negative_roi(weights: np.ndarray) -> float:
            """Objective function: negative ROI to minimize (maximize ROI)"""
            # Weighted ensemble predictions
            ensemble_predictions = (X * weights).sum(axis=1)

            # ROI calculation: sum of (probability * outcome) - cost
            # For binary outcomes, this approximates profit from betting
            roi = np.mean(y * ensemble_predictions - (1 - y) * ensemble_predictions)

            # Return negative to minimize (scipy minimizes by default)
            return -roi

        # Constraints: weights sum to 1 and are non-negative
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # sum(weights) = 1
        ]

        # Bounds: each weight >= 0 (non-negative)
        bounds = [(0.0, 1.0) for _ in range(num_models)]

        # Initial guess: equal weights
        initial_weights = np.ones(num_models) / num_models

        logger.info(f"Optimizing weights for {num_models} models...")
        logger.info(
            f"Initial ROI with equal weights: {-negative_roi(initial_weights):.4f}"
        )

        # Run optimization with multiple methods for robustness
        methods = ["SLSQP", "trust-constr"]
        best_result = None
        best_roi = float("-inf")

        for method in methods:
            try:
                result = minimize(
                    negative_roi,
                    initial_weights,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000, "ftol": 1e-8},
                )

                if result.success:
                    roi_value = -result.fun
                    if roi_value > best_roi:
                        best_roi = roi_value
                        best_result = result

                logger.info(
                    f"Method {method}: {'Success' if result.success else 'Failed'}, ROI: {-result.fun:.4f}"
                )

            except Exception as e:
                logger.warning(f"Optimization method {method} failed: {e}")
                continue

        if best_result is None or not best_result.success:
            logger.warning("All optimization methods failed, using equal weights")
            return initial_weights

        optimized_weights = best_result.x

        # Normalize weights to ensure they sum to exactly 1.0
        optimized_weights = optimized_weights / np.sum(optimized_weights)

        logger.info(f"✅ Optimization successful!")
        logger.info(f"   Final ROI: {best_roi:.4f}")
        logger.info(
            f"   Optimized weights: {dict(zip(self.model_names, optimized_weights))}"
        )

        return optimized_weights

    def save_weights(
        self, weights: np.ndarray, model_names: List[str], output_path: str
    ):
        """
        Save weights to a JSON file with model names.
        """
        # Create models directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        weights_dict = {
            "weights": {
                name: float(weight) for name, weight in zip(model_names, weights)
            },
            "metadata": {
                "total_models": len(model_names),
                "weights_sum": float(np.sum(weights)),
                "optimization_timestamp": datetime.now().isoformat(),
                "model_names": model_names,
            },
        }

        with open(output_path, "w") as f:
            json.dump(weights_dict, f, indent=4)

        logger.info(f"💾 Weights saved to {output_path}")

    def load_weights(self, weights_path: str) -> Optional[Dict[str, float]]:
        """
        Load previously saved weights from JSON file.
        """
        try:
            if not Path(weights_path).exists():
                return None

            with open(weights_path, "r") as f:
                weights_data = json.load(f)

            if "weights" in weights_data:
                return weights_data["weights"]
            else:
                # Legacy format
                return weights_data

        except Exception as e:
            logger.error(f"Error loading weights from {weights_path}: {e}")
            return None

    def compute_and_save_weights(
        self,
        output_path: str = "models/ensemble_weights.json",
        limit_records: Optional[int] = 10000,
    ) -> Dict[str, float]:
        """
        Complete workflow: load data, compute weights, and save them.
        """
        try:
            # Load historical data
            X, y = self.load_historical_data(limit_records=limit_records)

            # Compute optimal weights
            weights = self.compute_weights(X, y)

            # Save weights
            self.save_weights(weights, self.model_names, output_path)

            # Return as dictionary
            return {
                name: float(weight) for name, weight in zip(self.model_names, weights)
            }

        except Exception as e:
            logger.error(f"Error in compute_and_save_weights: {e}")
            raise


# Convenience function
def optimize_ensemble_weights(
    db_path: str = "greyhound_racing_data.db",
    output_path: str = "models/ensemble_weights.json",
    limit_records: Optional[int] = 10000,
) -> Dict[str, float]:
    """
    Standalone function to optimize ensemble weights.
    """
    weighter = EnsembleROIWeighter(db_path)
    return weighter.compute_and_save_weights(output_path, limit_records)
