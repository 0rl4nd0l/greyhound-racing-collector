#!/usr/bin/env python3
"""
Dog Performance Features Engineering - Step 2
=============================================

For every dog calculate comprehensive performance metrics:
- Mean, median, and best race time (overall and at Ballarat/distance)
- Place percentage (wins, top-3)
- Average beaten margin and average margin when winning
- Early speed proxy (first-split times or sectional rank)
- Recent-form trend (linear regression slope of times over last 5 runs)

Store in a features vector ready for modelling.

Author: AI Assistant
Date: December 2024
"""

import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DogPerformanceFeatureEngineer:
    """
    Comprehensive performance feature engineering for individual dogs.

    Calculates advanced metrics needed for machine learning models
    including time-based performance, positional statistics, and trend analysis.
    """

    def __init__(self, historical_data_path: Optional[str] = None):
        """
        Initialize the feature engineering system.

        Args:
            historical_data_path: Path to directory containing dog historical data
        """
        self.historical_data_path = historical_data_path or "dog_records"
        self.ballarat_codes = ["BAL", "BALLARAT", "Ballarat"]

        # Initialize feature cache for performance
        self._feature_cache = {}

        logger.info("DogPerformanceFeatureEngineer initialized")

    def extract_all_dog_features(self, dog_names: List[str] = None) -> pd.DataFrame:
        """
        Extract comprehensive performance features for all dogs or specified dogs.

        Args:
            dog_names: Optional list of specific dog names to process

        Returns:
            DataFrame with comprehensive features for each dog
        """
        if dog_names is None:
            dog_names = self._get_all_dog_names()

        all_features = []

        for dog_name in dog_names:
            try:
                logger.info(f"Processing features for {dog_name}")
                features = self.extract_dog_features(dog_name)
                features["dog_name"] = dog_name
                all_features.append(features)

            except Exception as e:
                logger.error(f"Error processing {dog_name}: {e}")
                continue

        if not all_features:
            logger.warning("No features extracted for any dogs")
            return pd.DataFrame()

        features_df = pd.DataFrame(all_features)

        # Set dog_name as index and reorder columns
        features_df.set_index("dog_name", inplace=True)

        logger.info(
            f"Extracted features for {len(features_df)} dogs with {len(features_df.columns)} features each"
        )
        return features_df

    def extract_dog_features(self, dog_name: str) -> Dict[str, Any]:
        """
        Extract comprehensive performance features for a single dog.

        Args:
            dog_name: Name of the dog

        Returns:
            Dictionary of computed features
        """
        # Check cache first
        if dog_name in self._feature_cache:
            return self._feature_cache[dog_name].copy()

        # Load historical data
        historical_data = self._load_dog_history(dog_name)

        if historical_data.empty:
            logger.warning(f"No historical data found for {dog_name}")
            return self._get_default_features()

        # Calculate comprehensive features
        features = {}

        # Time-based performance features
        features.update(self._calculate_time_features(historical_data))

        # Positional performance features
        features.update(self._calculate_positional_features(historical_data))

        # Margin analysis features
        features.update(self._calculate_margin_features(historical_data))

        # Early speed features
        features.update(self._calculate_early_speed_features(historical_data))

        # Recent form trend features
        features.update(self._calculate_form_trend_features(historical_data))

        # Venue-specific features (Ballarat focus)
        features.update(self._calculate_venue_specific_features(historical_data))

        # Distance-specific features
        features.update(self._calculate_distance_specific_features(historical_data))

        # Consistency and reliability features
        features.update(self._calculate_consistency_features(historical_data))

        # Race context features
        features.update(self._calculate_race_context_features(historical_data))

        # Cache the results
        self._feature_cache[dog_name] = features.copy()

        return features

    def _calculate_time_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive time-based performance features."""
        features = {}

        # Ensure race_time is numeric
        data["race_time_clean"] = pd.to_numeric(data["race_time"], errors="coerce")
        time_data = data["race_time_clean"].dropna()

        if len(time_data) == 0:
            return {
                "mean_race_time": 30.0,
                "median_race_time": 30.0,
                "best_race_time": 29.0,
                "worst_race_time": 35.0,
                "time_std": 2.0,
                "time_variance": 4.0,
                "time_range": 6.0,
                "time_improvement_slope": 0.0,
                "time_consistency_score": 0.5,
            }

        # Basic time statistics
        features["mean_race_time"] = float(time_data.mean())
        features["median_race_time"] = float(time_data.median())
        features["best_race_time"] = float(time_data.min())
        features["worst_race_time"] = float(time_data.max())
        features["time_std"] = float(time_data.std()) if len(time_data) > 1 else 0.0
        features["time_variance"] = (
            float(time_data.var()) if len(time_data) > 1 else 0.0
        )
        features["time_range"] = (
            features["worst_race_time"] - features["best_race_time"]
        )

        # Time improvement trend (linear regression slope)
        if len(time_data) >= 3:
            x = np.arange(len(time_data))
            try:
                slope, _, _, _, _ = stats.linregress(x, time_data.values)
                features["time_improvement_slope"] = float(slope)
            except:
                features["time_improvement_slope"] = 0.0
        else:
            features["time_improvement_slope"] = 0.0

        # Time consistency score (inverse of coefficient of variation)
        if features["mean_race_time"] > 0 and features["time_std"] > 0:
            cv = features["time_std"] / features["mean_race_time"]
            features["time_consistency_score"] = float(1.0 / (1.0 + cv))
        else:
            features["time_consistency_score"] = 1.0

        return features

    def _calculate_positional_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate position-based performance features."""
        features = {}

        # Ensure place is numeric
        data["place_clean"] = pd.to_numeric(data["place"], errors="coerce")
        position_data = data["place_clean"].dropna()

        if len(position_data) == 0:
            return {
                "mean_position": 4.0,
                "median_position": 4.0,
                "best_position": 1.0,
                "worst_position": 8.0,
                "win_rate": 0.1,
                "place_rate_top3": 0.3,
                "place_rate_top2": 0.2,
                "position_consistency": 0.5,
                "avg_beaten_lengths": 5.0,
            }

        # Basic positional statistics
        features["mean_position"] = float(position_data.mean())
        features["median_position"] = float(position_data.median())
        features["best_position"] = float(position_data.min())
        features["worst_position"] = float(position_data.max())

        # Place percentages
        total_races = len(position_data)
        features["win_rate"] = float((position_data == 1).sum() / total_races)
        features["place_rate_top3"] = float((position_data <= 3).sum() / total_races)
        features["place_rate_top2"] = float((position_data <= 2).sum() / total_races)

        # Position consistency (inverse of standard deviation)
        pos_std = position_data.std() if len(position_data) > 1 else 0.0
        features["position_consistency"] = float(1.0 / (1.0 + pos_std))

        # Average beaten lengths (estimate from margins if available)
        if "margin" in data.columns:
            margin_data = pd.to_numeric(data["margin"], errors="coerce").dropna()
            if len(margin_data) > 0:
                features["avg_beaten_lengths"] = float(margin_data.mean())
            else:
                features["avg_beaten_lengths"] = 5.0
        else:
            # Estimate based on average position
            features["avg_beaten_lengths"] = float(
                max(0, (features["mean_position"] - 1) * 2.5)
            )

        return features

    def _calculate_margin_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate margin-based performance features."""
        features = {}

        # Process margin data
        if "margin" in data.columns:
            data["margin_clean"] = pd.to_numeric(data["margin"], errors="coerce")
            margin_data = data["margin_clean"].dropna()

            # Separate winning and losing margins
            place_data = pd.to_numeric(data["place"], errors="coerce")
            winning_races = data[place_data == 1]
            losing_races = data[place_data > 1]

            # Average beaten margin (when not winning)
            if len(losing_races) > 0:
                losing_margins = pd.to_numeric(
                    losing_races["margin"], errors="coerce"
                ).dropna()
                if len(losing_margins) > 0:
                    features["avg_beaten_margin"] = float(losing_margins.mean())
                else:
                    features["avg_beaten_margin"] = 5.0
            else:
                features["avg_beaten_margin"] = 0.0

            # Average margin when winning
            if len(winning_races) > 0:
                winning_margins = pd.to_numeric(
                    winning_races["margin"], errors="coerce"
                ).dropna()
                if len(winning_margins) > 0:
                    features["avg_winning_margin"] = float(winning_margins.abs().mean())
                else:
                    features["avg_winning_margin"] = 2.0
            else:
                features["avg_winning_margin"] = 0.0

            # Overall margin statistics
            if len(margin_data) > 0:
                features["margin_std"] = (
                    float(margin_data.std()) if len(margin_data) > 1 else 0.0
                )
                features["best_winning_margin"] = (
                    float(margin_data.min()) if len(margin_data) > 0 else 0.0
                )
                features["worst_losing_margin"] = (
                    float(margin_data.max()) if len(margin_data) > 0 else 10.0
                )
            else:
                features["margin_std"] = 3.0
                features["best_winning_margin"] = 1.0
                features["worst_losing_margin"] = 10.0
        else:
            # Default values when margin data is not available
            features["avg_beaten_margin"] = 5.0
            features["avg_winning_margin"] = 2.0
            features["margin_std"] = 3.0
            features["best_winning_margin"] = 1.0
            features["worst_losing_margin"] = 10.0

        return features

    def _calculate_early_speed_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate early speed proxy features from first-split times."""
        features = {}

        # Process first section times
        if "first_section_time" in data.columns:
            data["first_section_clean"] = pd.to_numeric(
                data["first_section_time"], errors="coerce"
            )
            first_section_data = data["first_section_clean"].dropna()

            if len(first_section_data) > 0:
                features["mean_first_section"] = float(first_section_data.mean())
                features["best_first_section"] = float(first_section_data.min())
                features["first_section_consistency"] = (
                    float(1.0 / (1.0 + first_section_data.std()))
                    if len(first_section_data) > 1
                    else 1.0
                )

                # Early speed rank approximation (lower time = better rank)
                # Normalize to typical sectional times (4.5-5.5 seconds)
                avg_section = features["mean_first_section"]
                if 4.0 <= avg_section <= 6.0:
                    # Scale to 1-8 rank (1 = fastest)
                    features["early_speed_rank"] = float(
                        1 + 7 * (avg_section - 4.0) / 2.0
                    )
                else:
                    features["early_speed_rank"] = 4.0
            else:
                features["mean_first_section"] = 4.8
                features["best_first_section"] = 4.6
                features["first_section_consistency"] = 0.7
                features["early_speed_rank"] = 4.0
        else:
            # Default early speed features
            features["mean_first_section"] = 4.8
            features["best_first_section"] = 4.6
            features["first_section_consistency"] = 0.7
            features["early_speed_rank"] = 4.0

        # Early speed score (combination of section time and consistency)
        section_score = max(
            0, 1 - (features["mean_first_section"] - 4.5) / 1.0
        )  # Normalize around 4.5s
        consistency_score = features["first_section_consistency"]
        features["early_speed_score"] = float(
            0.7 * section_score + 0.3 * consistency_score
        )

        return features

    def _calculate_form_trend_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate recent form trend using linear regression on last 5 runs."""
        features = {}

        # Sort by race date to get chronological order
        if "race_date" in data.columns:
            data_sorted = data.sort_values("race_date", ascending=False)
        else:
            # Use the order as-is if no date column
            data_sorted = data.copy()

        # Get last 5 runs for trend analysis
        last_5_runs = data_sorted.head(5)

        if len(last_5_runs) >= 3:
            # Position trend
            positions = pd.to_numeric(last_5_runs["place"], errors="coerce").dropna()
            if len(positions) >= 3:
                x = np.arange(len(positions))
                try:
                    slope, _, r_value, _, _ = stats.linregress(x, positions.values)
                    features["recent_position_trend"] = float(
                        -slope
                    )  # Negative slope = improving
                    features["recent_form_r_squared"] = float(r_value**2)
                except:
                    features["recent_position_trend"] = 0.0
                    features["recent_form_r_squared"] = 0.0
            else:
                features["recent_position_trend"] = 0.0
                features["recent_form_r_squared"] = 0.0

            # Time trend (if available)
            times = pd.to_numeric(last_5_runs["race_time"], errors="coerce").dropna()
            if len(times) >= 3:
                x = np.arange(len(times))
                try:
                    slope, _, _, _, _ = stats.linregress(x, times.values)
                    features["recent_time_trend"] = float(
                        -slope
                    )  # Negative slope = improving times
                except:
                    features["recent_time_trend"] = 0.0
            else:
                features["recent_time_trend"] = 0.0

            # Recent form quality score
            recent_positions = positions.values if len(positions) > 0 else [4, 4, 4]
            features["recent_form_avg"] = float(np.mean(recent_positions))
            features["recent_form_best"] = float(np.min(recent_positions))
            features["recent_form_worst"] = float(np.max(recent_positions))

        else:
            # Default values for insufficient data
            features["recent_position_trend"] = 0.0
            features["recent_time_trend"] = 0.0
            features["recent_form_avg"] = 4.0
            features["recent_form_best"] = 2.0
            features["recent_form_worst"] = 6.0
            features["recent_form_r_squared"] = 0.0

        return features

    def _calculate_venue_specific_features(
        self, data: pd.DataFrame, target_venue: str = None
    ) -> Dict[str, Any]:
        """Calculate venue-specific performance features with contextual weighting."""
        features = {}

        # Apply contextual weighting based on target venue
        if target_venue:
            # Check if target venue is Ballarat
            ballarat_codes = ["BAL", "BALLARAT", "Ballarat"]
            is_ballarat = target_venue.upper() in [
                code.upper() for code in ballarat_codes
            ]

            if is_ballarat:
                # Filter for Ballarat races with higher emphasis
                if "track_code" in data.columns:
                    ballarat_data = data[
                        data["track_code"]
                        .str.upper()
                        .isin([code.upper() for code in ballarat_codes])
                    ]
                elif "venue" in data.columns:
                    ballarat_data = data[
                        data["venue"]
                        .str.upper()
                        .isin([code.upper() for code in ballarat_codes])
                    ]
                else:
                    ballarat_data = pd.DataFrame()
            else:
                # Filter for same venue races
                if "venue" in data.columns:
                    ballarat_data = data[data["venue"] == target_venue]
                elif "track_code" in data.columns:
                    ballarat_data = data[data["track_code"] == target_venue]
                else:
                    ballarat_data = pd.DataFrame()
        else:
            # Default: Filter for Ballarat races
            if "track_code" in data.columns:
                ballarat_data = data[data["track_code"].isin(self.ballarat_codes)]
            else:
                # If no track code, assume no Ballarat data
                ballarat_data = pd.DataFrame()

        if len(ballarat_data) > 0:
            # Ballarat-specific time features
            ballarat_times = pd.to_numeric(
                ballarat_data["race_time"], errors="coerce"
            ).dropna()
            if len(ballarat_times) > 0:
                features["ballarat_mean_time"] = float(ballarat_times.mean())
                features["ballarat_best_time"] = float(ballarat_times.min())
                features["ballarat_time_consistency"] = (
                    float(1.0 / (1.0 + ballarat_times.std()))
                    if len(ballarat_times) > 1
                    else 1.0
                )
            else:
                features["ballarat_mean_time"] = 30.0
                features["ballarat_best_time"] = 29.0
                features["ballarat_time_consistency"] = 0.5

            # Ballarat positional features
            ballarat_positions = pd.to_numeric(
                ballarat_data["place"], errors="coerce"
            ).dropna()
            if len(ballarat_positions) > 0:
                features["ballarat_mean_position"] = float(ballarat_positions.mean())
                features["ballarat_win_rate"] = float(
                    (ballarat_positions == 1).sum() / len(ballarat_positions)
                )
                features["ballarat_place_rate"] = float(
                    (ballarat_positions <= 3).sum() / len(ballarat_positions)
                )
            else:
                features["ballarat_mean_position"] = 4.0
                features["ballarat_win_rate"] = 0.1
                features["ballarat_place_rate"] = 0.3

            features["ballarat_experience"] = len(ballarat_data)
        else:
            # No Ballarat experience
            features["ballarat_mean_time"] = 30.0
            features["ballarat_best_time"] = 29.0
            features["ballarat_time_consistency"] = 0.5
            features["ballarat_mean_position"] = 4.0
            features["ballarat_win_rate"] = 0.05  # Lower than average for no experience
            features["ballarat_place_rate"] = 0.2
            features["ballarat_experience"] = 0

        return features

    def _calculate_distance_specific_features(
        self, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate distance-specific performance features."""
        features = {}

        if "distance_m" in data.columns:
            # Group by distance and calculate performance
            distance_groups = data.groupby("distance_m")

            distance_performance = {}
            for distance, group in distance_groups:
                if len(group) >= 2:  # Need at least 2 races for meaningful stats
                    times = pd.to_numeric(group["race_time"], errors="coerce").dropna()
                    positions = pd.to_numeric(group["place"], errors="coerce").dropna()

                    if len(times) > 0 and len(positions) > 0:
                        distance_performance[distance] = {
                            "mean_time": times.mean(),
                            "best_time": times.min(),
                            "mean_position": positions.mean(),
                            "win_rate": (positions == 1).sum() / len(positions),
                            "races": len(group),
                        }

            # Find best distance performance
            if distance_performance:
                best_distance = min(
                    distance_performance.keys(),
                    key=lambda d: distance_performance[d]["mean_position"],
                )

                features["best_distance"] = float(best_distance)
                features["best_distance_win_rate"] = float(
                    distance_performance[best_distance]["win_rate"]
                )
                features["best_distance_mean_time"] = float(
                    distance_performance[best_distance]["mean_time"]
                )
                features["distance_specialization"] = float(len(distance_performance))

                # Most common distance
                most_common_distance = max(
                    distance_performance.keys(),
                    key=lambda d: distance_performance[d]["races"],
                )
                features["preferred_distance"] = float(most_common_distance)
            else:
                features["best_distance"] = 400.0
                features["best_distance_win_rate"] = 0.1
                features["best_distance_mean_time"] = 30.0
                features["distance_specialization"] = 1.0
                features["preferred_distance"] = 400.0
        else:
            # Default distance features
            features["best_distance"] = 400.0
            features["best_distance_win_rate"] = 0.1
            features["best_distance_mean_time"] = 30.0
            features["distance_specialization"] = 1.0
            features["preferred_distance"] = 400.0

        return features

    def _calculate_consistency_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate consistency and reliability features."""
        features = {}

        # Overall consistency metrics
        positions = pd.to_numeric(data["place"], errors="coerce").dropna()
        times = pd.to_numeric(data["race_time"], errors="coerce").dropna()

        if len(positions) > 1:
            # Position consistency
            pos_cv = positions.std() / positions.mean() if positions.mean() > 0 else 1.0
            features["position_coefficient_variation"] = float(pos_cv)
            features["position_reliability"] = float(1.0 / (1.0 + pos_cv))

            # Finish rate (percentage of races finished)
            # Assume all races in data were finished
            features["finish_rate"] = 1.0
        else:
            features["position_coefficient_variation"] = 0.5
            features["position_reliability"] = 0.5
            features["finish_rate"] = 1.0

        if len(times) > 1:
            # Time consistency
            time_cv = times.std() / times.mean() if times.mean() > 0 else 0.1
            features["time_coefficient_variation"] = float(time_cv)
            features["time_reliability"] = float(1.0 / (1.0 + time_cv))
        else:
            features["time_coefficient_variation"] = 0.05
            features["time_reliability"] = 0.8

        # Performance predictability score
        features["performance_predictability"] = float(
            0.5 * features["position_reliability"] + 0.5 * features["time_reliability"]
        )

        return features

    def _calculate_race_context_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate race context and competitive level features."""
        features = {}

        # Grade/class analysis
        if "grade" in data.columns:
            grade_counts = data["grade"].value_counts()
            features["most_common_grade"] = (
                str(grade_counts.index[0]) if len(grade_counts) > 0 else "M"
            )
            features["grade_diversity"] = len(grade_counts)
        else:
            features["most_common_grade"] = "M"
            features["grade_diversity"] = 1

        # Starting price analysis (market confidence)
        if "starting_price" in data.columns:
            prices = pd.to_numeric(data["starting_price"], errors="coerce").dropna()
            if len(prices) > 0:
                features["avg_starting_price"] = float(prices.mean())
                features["best_starting_price"] = float(prices.min())
                features["market_support"] = float(
                    (prices <= 5.0).sum() / len(prices)
                )  # Percentage of races as favorite
            else:
                features["avg_starting_price"] = 10.0
                features["best_starting_price"] = 8.0
                features["market_support"] = 0.1
        else:
            features["avg_starting_price"] = 10.0
            features["best_starting_price"] = 8.0
            features["market_support"] = 0.1

        # Race frequency and activity level
        if "race_date" in data.columns:
            try:
                dates = pd.to_datetime(data["race_date"], errors="coerce").dropna()
                if len(dates) > 1:
                    date_diffs = dates.diff().dt.days.dropna()
                    features["avg_days_between_races"] = float(date_diffs.mean())
                    features["racing_frequency"] = float(
                        365.0 / max(features["avg_days_between_races"], 1)
                    )
                else:
                    features["avg_days_between_races"] = 14.0
                    features["racing_frequency"] = 26.0
            except:
                features["avg_days_between_races"] = 14.0
                features["racing_frequency"] = 26.0
        else:
            features["avg_days_between_races"] = 14.0
            features["racing_frequency"] = 26.0

        # Total career statistics
        features["total_races"] = len(data)
        features["career_earnings_proxy"] = float(
            features["total_races"] * features.get("win_rate", 0.1) * 1000
        )  # Estimate

        return features

    def _load_dog_history(self, dog_name: str) -> pd.DataFrame:
        """Load historical racing data for a specific dog."""
        file_patterns = [
            f"{dog_name}_historical_runs.csv",
            f"{dog_name.replace(' ', '_')}_historical_runs.csv",
            f"{dog_name.replace(' ', '')}_historical_runs.csv",
        ]

        for pattern in file_patterns:
            file_path = os.path.join(self.historical_data_path, pattern)
            if os.path.exists(file_path):
                try:
                    # Try different separators
                    for sep in ["|", ",", "\t"]:
                        try:
                            data = pd.read_csv(file_path, sep=sep)
                            if len(data.columns) > 5:  # Reasonable number of columns
                                logger.debug(f"Loaded {len(data)} races for {dog_name}")
                                return data
                        except:
                            continue
                except Exception as e:
                    logger.error(f"Error loading data for {dog_name}: {e}")
                    continue

        logger.warning(f"No historical data file found for {dog_name}")
        return pd.DataFrame()

    def _get_all_dog_names(self) -> List[str]:
        """Get list of all dogs with historical data files."""
        if not os.path.exists(self.historical_data_path):
            logger.error(
                f"Historical data path does not exist: {self.historical_data_path}"
            )
            return []

        dog_names = []
        for file in os.listdir(self.historical_data_path):
            if file.endswith("_historical_runs.csv"):
                dog_name = file.replace("_historical_runs.csv", "").replace("_", " ")
                dog_names.append(dog_name)

        logger.info(f"Found {len(dog_names)} dogs with historical data")
        return dog_names

    def _get_default_features(self) -> Dict[str, Any]:
        """Return default feature values for dogs with no data."""
        return {
            # Time features
            "mean_race_time": 30.0,
            "median_race_time": 30.0,
            "best_race_time": 29.0,
            "worst_race_time": 32.0,
            "time_std": 1.5,
            "time_variance": 2.25,
            "time_range": 3.0,
            "time_improvement_slope": 0.0,
            "time_consistency_score": 0.5,
            # Position features
            "mean_position": 4.0,
            "median_position": 4.0,
            "best_position": 2.0,
            "worst_position": 6.0,
            "win_rate": 0.1,
            "place_rate_top3": 0.3,
            "place_rate_top2": 0.2,
            "position_consistency": 0.5,
            "avg_beaten_lengths": 5.0,
            # Margin features
            "avg_beaten_margin": 5.0,
            "avg_winning_margin": 2.0,
            "margin_std": 2.5,
            "best_winning_margin": 1.0,
            "worst_losing_margin": 8.0,
            # Early speed features
            "mean_first_section": 4.8,
            "best_first_section": 4.6,
            "first_section_consistency": 0.7,
            "early_speed_rank": 4.0,
            "early_speed_score": 0.5,
            # Form trend features
            "recent_position_trend": 0.0,
            "recent_time_trend": 0.0,
            "recent_form_avg": 4.0,
            "recent_form_best": 2.0,
            "recent_form_worst": 6.0,
            "recent_form_r_squared": 0.0,
            # Venue features (Ballarat)
            "ballarat_mean_time": 30.0,
            "ballarat_best_time": 29.0,
            "ballarat_time_consistency": 0.5,
            "ballarat_mean_position": 4.0,
            "ballarat_win_rate": 0.05,
            "ballarat_place_rate": 0.2,
            "ballarat_experience": 0,
            # Distance features
            "best_distance": 400.0,
            "best_distance_win_rate": 0.1,
            "best_distance_mean_time": 30.0,
            "distance_specialization": 1.0,
            "preferred_distance": 400.0,
            # Consistency features
            "position_coefficient_variation": 0.5,
            "position_reliability": 0.5,
            "time_coefficient_variation": 0.05,
            "time_reliability": 0.8,
            "performance_predictability": 0.65,
            "finish_rate": 1.0,
            # Race context features
            "most_common_grade": "M",
            "grade_diversity": 1,
            "avg_starting_price": 10.0,
            "best_starting_price": 8.0,
            "market_support": 0.1,
            "avg_days_between_races": 14.0,
            "racing_frequency": 26.0,
            "total_races": 10,
            "career_earnings_proxy": 1000.0,
        }

    def save_features_to_file(
        self, features_df: pd.DataFrame, filename: str = None
    ) -> str:
        """
        Save extracted features to CSV file.

        Args:
            features_df: DataFrame containing features
            filename: Optional filename, auto-generated if None

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dog_performance_features_{timestamp}.csv"

        features_df.to_csv(filename)
        logger.info(f"Features saved to {filename}")
        return filename


def main():
    """Main function to demonstrate feature extraction."""
    # Initialize feature engineer
    feature_engineer = DogPerformanceFeatureEngineer()

    # Extract features for all dogs
    print("Extracting performance features for all dogs...")
    features_df = feature_engineer.extract_all_dog_features()

    if not features_df.empty:
        print(
            f"\nExtracted {len(features_df)} dog profiles with {len(features_df.columns)} features each"
        )
        print("\nFeature summary:")
        print(features_df.describe())

        # Save to file
        output_file = feature_engineer.save_features_to_file(features_df)
        print(f"\nFeatures saved to: {output_file}")

        # Display sample features for first few dogs
        print(f"\nSample features for first 3 dogs:")
        for dog_name in features_df.index[:3]:
            print(f"\n{dog_name}:")
            print(
                f"  Mean Race Time: {features_df.loc[dog_name, 'mean_race_time']:.2f}s"
            )
            print(
                f"  Best Race Time: {features_df.loc[dog_name, 'best_race_time']:.2f}s"
            )
            print(f"  Win Rate: {features_df.loc[dog_name, 'win_rate']:.3f}")
            print(
                f"  Place Rate (Top 3): {features_df.loc[dog_name, 'place_rate_top3']:.3f}"
            )
            print(
                f"  Recent Form Trend: {features_df.loc[dog_name, 'recent_position_trend']:.3f}"
            )
            print(
                f"  Early Speed Score: {features_df.loc[dog_name, 'early_speed_score']:.3f}"
            )
            print(
                f"  Ballarat Experience: {features_df.loc[dog_name, 'ballarat_experience']} races"
            )
    else:
        print("No features extracted. Check data availability.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
