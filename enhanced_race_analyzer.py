#!/usr/bin/env python3
"""
Enhanced Race Analyzer
=====================

This script provides advanced analysis capabilities that address:
1. Dog frequency bias normalization
2. Temporal analysis and trends
3. Performance consistency metrics
4. Advanced feature engineering

Author: AI Assistant
Date: July 11, 2025
"""

import sqlite3
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class EnhancedRaceAnalyzer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.data = None
        self.normalized_data = None

    def load_data(self):
        """Load and prepare race data"""
        print("📊 Loading race data...")

        query = """
        SELECT 
            rm.race_id,
            rm.race_date,
            rm.venue,
            rm.race_number,
            rm.field_size,
            rm.grade,
            rm.distance,
            rm.track_condition,
            drd.dog_clean_name as dog_name,
            drd.box_number,
            drd.finish_position,
            drd.weight,
            drd.starting_price,
            drd.individual_time,
            drd.margin,
            drd.sectional_1st,
            drd.trainer_name
        FROM race_metadata rm
        JOIN dog_race_data drd ON rm.race_id = drd.race_id
        WHERE drd.dog_clean_name != ''
        AND rm.race_date IS NOT NULL
        AND drd.finish_position IS NOT NULL
        AND drd.finish_position != 'N/A'
        AND drd.finish_position != ''
        ORDER BY rm.race_date DESC, rm.race_id, drd.box_number
        """

        self.data = pd.read_sql_query(query, self.conn)
        # Handle different date formats
        self.data["race_date"] = pd.to_datetime(
            self.data["race_date"], format="mixed", dayfirst=False
        )

        # Remove any remaining N/A finish positions and convert to numeric
        self.data = self.data[
            self.data["finish_position"].notna()
            & (self.data["finish_position"] != "N/A")
        ]
        self.data["finish_position"] = pd.to_numeric(
            self.data["finish_position"], errors="coerce"
        )
        self.data = self.data[
            self.data["finish_position"].notna()
        ]  # Remove any that couldn't be converted

        print(f"✅ Loaded {len(self.data)} race entries")
        return self.data

    def engineer_features(self):
        """Create advanced features for analysis"""
        print("🔧 Engineering features...")

        # Dog frequency metrics
        dog_stats = (
            self.data.groupby("dog_name")
            .agg(
                {
                    "race_date": ["count", "min", "max"],
                    "finish_position": ["mean", "std", "median"],
                    "weight": "mean",
                    "starting_price": "mean",
                }
            )
            .reset_index()
        )

        dog_stats.columns = [
            "dog_name",
            "race_count",
            "first_race",
            "last_race",
            "avg_position",
            "position_std",
            "median_position",
            "avg_weight",
            "avg_starting_price",
        ]

        # Calculate career span
        dog_stats["career_span_days"] = (
            dog_stats["last_race"] - dog_stats["first_race"]
        ).dt.days
        dog_stats["races_per_month"] = dog_stats["race_count"] / (
            dog_stats["career_span_days"] / 30.44
        )
        dog_stats["races_per_month"] = dog_stats["races_per_month"].fillna(
            dog_stats["race_count"]
        )

        # Add recency metrics
        latest_date = self.data["race_date"].max()
        dog_stats["days_since_last_race"] = (
            latest_date - dog_stats["last_race"]
        ).dt.days

        # Merge back with main data
        self.data = self.data.merge(dog_stats, on="dog_name", how="left")

        # Track performance features
        self.data["race_sequence"] = self.data.groupby("dog_name").cumcount() + 1
        self.data["is_frequent_racer"] = self.data["race_count"] > 2

        print(f"✅ Added {len(dog_stats.columns)} dog-level features")

    def add_race_condition_features(self):
        """Add features based on race conditions (grade, distance, track condition)"""
        print("🏁 Adding race condition features...")

        # Extract numeric distance from distance string (e.g., '400m' -> 400)
        self.data["distance_numeric"] = pd.to_numeric(
            self.data["distance"].str.replace("m", "").str.replace("M", ""),
            errors="coerce",
        ).fillna(0)

        # Create distance categories
        self.data["distance_category"] = pd.cut(
            self.data["distance_numeric"],
            bins=[0, 350, 450, 550, 1000],
            labels=["Short", "Medium", "Long", "Extra_Long"],
        )

        # Normalize grade to handle various formats
        self.data["grade_normalized"] = self.data["grade"].str.upper().fillna("UNKNOWN")

        # Extract numeric grade where possible
        self.data["grade_numeric"] = pd.to_numeric(
            self.data["grade"].str.extract(r"(\d+)")[0], errors="coerce"
        ).fillna(
            999
        )  # High number for non-numeric grades

        # Track condition normalized
        track_mapping = {
            "FAST": 1,
            "GOOD": 2,
            "SLOW": 3,
            "HEAVY": 4,
            "Fast": 1,
            "Good": 2,
            "slow": 3,
            "Heavy": 4,
        }
        self.data["track_condition_numeric"] = (
            self.data["track_condition"].map(track_mapping).fillna(2)
        )  # Default to 'Good'

        # Create venue performance features using label encoding
        from sklearn.preprocessing import LabelEncoder

        venue_encoder = LabelEncoder()
        self.data["venue_encoded"] = venue_encoder.fit_transform(self.data["venue"])

        # Store venue encoder for later use
        self.venue_encoder = venue_encoder

        # Calculate venue-specific average performance
        venue_performance_avg = self.data.groupby("venue")["performance_score"].mean()
        self.data["venue_avg_performance"] = self.data["venue"].map(
            venue_performance_avg
        )

        # Calculate dog performance by race conditions
        for dog_name in self.data["dog_name"].unique():
            dog_mask = self.data["dog_name"] == dog_name
            dog_data = self.data[dog_mask]

            # Performance by distance category
            distance_performance = dog_data.groupby("distance_category")[
                "performance_score"
            ].mean()
            for dist_cat in ["Short", "Medium", "Long", "Extra_Long"]:
                self.data.loc[dog_mask, f"dog_performance_{dist_cat.lower()}"] = (
                    distance_performance.get(dist_cat, 0.5)
                )

            # Performance by grade
            grade_performance = dog_data.groupby("grade_normalized")[
                "performance_score"
            ].mean()
            self.data.loc[dog_mask, "dog_avg_grade_performance"] = dog_data[
                "performance_score"
            ].mean()

            # Performance by track condition
            track_performance = dog_data.groupby("track_condition_numeric")[
                "performance_score"
            ].mean()
            self.data.loc[dog_mask, "dog_avg_track_performance"] = dog_data[
                "performance_score"
            ].mean()

            # Performance by venue (dog-specific venue performance)
            venue_performance = dog_data.groupby("venue")["performance_score"].mean()
            self.data.loc[dog_mask, "dog_venue_avg_performance"] = dog_data[
                "performance_score"
            ].mean()

        # Add weather condition features
        self.add_weather_condition_features()

        print("✅ Race condition features added")

    def add_weather_condition_features(self):
        """Add weather-based features to the analysis"""
        print("🌤️ Adding weather condition features...")

        try:
            # Load weather data from race metadata
            query = """
            SELECT 
                race_id,
                weather_condition,
                temperature,
                humidity,
                wind_speed,
                wind_direction,
                pressure,
                weather_adjustment_factor
            FROM race_metadata 
            WHERE weather_condition IS NOT NULL
            """

            weather_data = pd.read_sql_query(query, self.conn)

            if len(weather_data) > 0:
                print(f"📊 Found weather data for {len(weather_data)} races")

                # Merge weather data with main data
                self.data = self.data.merge(weather_data, on="race_id", how="left")

                # Weather condition encoding
                weather_encoding = {
                    "clear": 1,
                    "partly_cloudy": 2,
                    "cloudy": 3,
                    "overcast": 4,
                    "rain": 5,
                    "heavy_rain": 6,
                    "fog": 7,
                }
                self.data["weather_condition_encoded"] = (
                    self.data["weather_condition"].map(weather_encoding).fillna(0)
                )

                # Temperature features
                self.data["temperature"] = pd.to_numeric(
                    self.data["temperature"], errors="coerce"
                ).fillna(20)
                self.data["temp_deviation_from_optimal"] = abs(
                    self.data["temperature"] - 20
                )  # 20°C as optimal
                self.data["is_extreme_temperature"] = (
                    (self.data["temperature"] < 10) | (self.data["temperature"] > 30)
                ).astype(int)

                # Weather adjustment factor
                self.data["weather_adjustment_factor"] = pd.to_numeric(
                    self.data["weather_adjustment_factor"], errors="coerce"
                ).fillna(1.0)

                # Additional weather features
                self.data["humidity"] = pd.to_numeric(
                    self.data["humidity"], errors="coerce"
                ).fillna(50)
                self.data["wind_speed"] = pd.to_numeric(
                    self.data["wind_speed"], errors="coerce"
                ).fillna(0)
                self.data["pressure"] = pd.to_numeric(
                    self.data["pressure"], errors="coerce"
                ).fillna(1013)

                # Rain indicator
                self.data["is_rain"] = (
                    self.data["weather_condition"]
                    .str.contains("rain", case=False, na=False)
                    .astype(int)
                )

                # Calculate dog performance by weather conditions
                for dog_name in self.data["dog_name"].unique():
                    dog_mask = self.data["dog_name"] == dog_name
                    dog_data = self.data[dog_mask]

                    # Performance in different weather conditions
                    weather_performance = dog_data.groupby("weather_condition")[
                        "performance_score"
                    ].mean()

                    # Rain performance
                    rain_races = dog_data[dog_data["is_rain"] == 1]
                    if len(rain_races) > 0:
                        self.data.loc[dog_mask, "rain_performance_avg"] = rain_races[
                            "performance_score"
                        ].mean()
                        self.data.loc[dog_mask, "rain_experience"] = len(rain_races)
                    else:
                        self.data.loc[dog_mask, "rain_performance_avg"] = dog_data[
                            "performance_score"
                        ].mean()
                        self.data.loc[dog_mask, "rain_experience"] = 0

                    # Temperature performance analysis
                    temp_groups = pd.cut(
                        dog_data["temperature"],
                        bins=[0, 15, 25, 50],
                        labels=["Cold", "Optimal", "Hot"],
                    )
                    temp_performance = dog_data.groupby(temp_groups)[
                        "performance_score"
                    ].mean()

                    for temp_cat in ["Cold", "Optimal", "Hot"]:
                        self.data.loc[
                            dog_mask, f"temp_performance_{temp_cat.lower()}"
                        ] = temp_performance.get(
                            temp_cat, dog_data["performance_score"].mean()
                        )

                print("✅ Weather condition features added")
            else:
                print("⚠️ No weather data found in database")
                # Add default weather features
                self.data["weather_condition_encoded"] = 0
                self.data["temperature"] = 20
                self.data["temp_deviation_from_optimal"] = 0
                self.data["is_extreme_temperature"] = 0
                self.data["weather_adjustment_factor"] = 1.0
                self.data["humidity"] = 50
                self.data["wind_speed"] = 0
                self.data["pressure"] = 1013
                self.data["is_rain"] = 0
                self.data["rain_performance_avg"] = self.data.get(
                    "performance_score", 0.5
                )
                self.data["rain_experience"] = 0

        except Exception as e:
            print(f"⚠️ Error adding weather features: {e}")

    def normalize_performance(self):
        """Normalize performance metrics to account for frequency bias"""
        print("⚖️  Normalizing performance metrics...")

        # Validate and fix field_size issues
        print(
            f"📊 Field size stats before validation: min={self.data['field_size'].min()}, max={self.data['field_size'].max()}, mean={self.data['field_size'].mean():.2f}"
        )

        # Fix unrealistic field sizes - use actual maximum finish position per race as field size
        race_actual_field_sizes = (
            self.data.groupby("race_id")["finish_position"].max().reset_index()
        )
        race_actual_field_sizes.columns = ["race_id", "actual_field_size"]

        # Merge back to get actual field sizes
        self.data = self.data.merge(race_actual_field_sizes, on="race_id", how="left")

        # Create corrected field size - use the maximum of field_size and actual field size, with minimum of 3
        self.data["corrected_field_size"] = self.data.apply(
            lambda row: max(
                3, row["field_size"], row.get("actual_field_size", row["field_size"])
            ),
            axis=1,
        )

        print(
            f"📊 Field size stats after validation: min={self.data['corrected_field_size'].min()}, max={self.data['corrected_field_size'].max()}, mean={self.data['corrected_field_size'].mean():.2f}"
        )

        # Weight inversely proportional to race frequency
        max_races = self.data["race_count"].max()
        self.data["frequency_weight"] = np.sqrt(max_races / self.data["race_count"])

        # Create normalized performance score using corrected field size
        # Lower finish position is better, so invert it
        self.data["performance_score"] = (
            self.data["corrected_field_size"] - self.data["finish_position"] + 1
        ) / self.data["corrected_field_size"]
        self.data["weighted_performance"] = (
            self.data["performance_score"] * self.data["frequency_weight"]
        )

        # Consistency score (lower std is better) - handle NaN and infinity
        position_std_safe = self.data["position_std"].fillna(0)
        # Cap extremely high std values to prevent infinity
        position_std_safe = np.clip(position_std_safe, 0, 10)
        self.data["consistency_score"] = 1 / (1 + position_std_safe)

        # Recent form (last 3 races)
        self.data = self.data.sort_values(["dog_name", "race_date"])
        self.data["recent_form"] = self.data.groupby("dog_name")[
            "performance_score"
        ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

        # Form trend (improvement/decline over last 5 races)
        self.data["form_trend"] = (
            self.data.groupby("dog_name")["performance_score"]
            .transform(
                lambda x: x.rolling(window=5, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0
                )
            )
            .fillna(0)
        )

        # Handle any remaining NaN or infinity values
        numeric_cols = [
            "frequency_weight",
            "performance_score",
            "weighted_performance",
            "consistency_score",
            "recent_form",
        ]
        for col in numeric_cols:
            self.data[col] = self.data[col].replace([np.inf, -np.inf], np.nan)
            self.data[col] = self.data[col].fillna(self.data[col].median())

        print("✅ Performance normalization complete")

    def temporal_analysis(self):
        """Analyze temporal trends and patterns"""
        print("📅 Performing temporal analysis...")

        # Monthly performance trends
        self.data["year_month"] = self.data["race_date"].dt.to_period("M")
        # Ensure numeric conversion for aggregation
        self.data["performance_score"] = pd.to_numeric(
            self.data["performance_score"], errors="coerce"
        )
        self.data["finish_position"] = pd.to_numeric(
            self.data["finish_position"], errors="coerce"
        )
        self.data["individual_time"] = pd.to_numeric(
            self.data["individual_time"], errors="coerce"
        )

        monthly_stats = (
            self.data.groupby("year_month")
            .agg({"performance_score": ["mean", "std", "count"], "race_id": "nunique"})
            .reset_index()
        )

        monthly_stats.columns = [
            "year_month",
            "avg_performance",
            "performance_std",
            "total_entries",
            "unique_races",
        ]

        # Venue performance
        venue_stats = (
            self.data.groupby("venue")
            .agg(
                {
                    "performance_score": ["mean", "std", "count"],
                    "finish_position": "mean",
                    "field_size": "mean",
                }
            )
            .reset_index()
        )

        venue_stats.columns = [
            "venue",
            "avg_performance",
            "performance_std",
            "race_count",
            "avg_finish_pos",
            "avg_field_size",
        ]

        return monthly_stats, venue_stats

    def analyze_race_conditions(self):
        """Analyze performance by race conditions"""
        print("🏁 Analyzing race condition impacts...")

        # Distance analysis
        distance_stats = (
            self.data.groupby("distance_category")
            .agg(
                {
                    "performance_score": ["mean", "std", "count"],
                    "finish_position": "mean",
                    "individual_time": "mean",
                }
            )
            .reset_index()
        )
        distance_stats.columns = [
            "distance_category",
            "avg_performance",
            "performance_std",
            "race_count",
            "avg_finish_pos",
            "avg_time",
        ]

        # Grade analysis
        grade_stats = (
            self.data.groupby("grade_normalized")
            .agg(
                {
                    "performance_score": ["mean", "std", "count"],
                    "finish_position": "mean",
                    "field_size": "mean",
                }
            )
            .reset_index()
        )
        grade_stats.columns = [
            "grade",
            "avg_performance",
            "performance_std",
            "race_count",
            "avg_finish_pos",
            "avg_field_size",
        ]

        # Track condition analysis
        track_stats = (
            self.data[self.data["track_condition"].notna()]
            .groupby("track_condition")
            .agg(
                {
                    "performance_score": ["mean", "std", "count"],
                    "finish_position": "mean",
                    "individual_time": "mean",
                }
            )
            .reset_index()
        )
        track_stats.columns = [
            "track_condition",
            "avg_performance",
            "performance_std",
            "race_count",
            "avg_finish_pos",
            "avg_time",
        ]

        # Venue analysis
        venue_performance_stats = (
            self.data.groupby("venue")
            .agg(
                {
                    "performance_score": ["mean", "std", "count"],
                    "finish_position": "mean",
                    "individual_time": "mean",
                    "field_size": "mean",
                }
            )
            .reset_index()
        )
        venue_performance_stats.columns = [
            "venue",
            "avg_performance",
            "performance_std",
            "race_count",
            "avg_finish_pos",
            "avg_time",
            "avg_field_size",
        ]

        # Distance vs Performance correlation
        distance_correlation = self.data["distance_numeric"].corr(
            self.data["performance_score"]
        )

        # Weather analysis
        weather_stats = None
        weather_performance_impact = None
        if (
            "weather_condition" in self.data.columns
            and self.data["weather_condition"].notna().any()
        ):
            weather_stats = (
                self.data.groupby("weather_condition")
                .agg(
                    {
                        "performance_score": ["mean", "std", "count"],
                        "finish_position": "mean",
                        "individual_time": "mean",
                        "weather_adjustment_factor": "mean",
                    }
                )
                .reset_index()
            )
            weather_stats.columns = [
                "weather_condition",
                "avg_performance",
                "performance_std",
                "race_count",
                "avg_finish_pos",
                "avg_time",
                "avg_adjustment_factor",
            ]

            # Weather performance impact analysis
            weather_performance_impact = {
                "clear_vs_rain": {
                    "clear_avg": self.data[self.data["weather_condition"] == "clear"][
                        "performance_score"
                    ].mean(),
                    "rain_avg": self.data[self.data["is_rain"] == 1][
                        "performance_score"
                    ].mean(),
                    "difference": self.data[self.data["weather_condition"] == "clear"][
                        "performance_score"
                    ].mean()
                    - self.data[self.data["is_rain"] == 1]["performance_score"].mean(),
                },
                "temperature_correlation": self.data["temperature"].corr(
                    self.data["performance_score"]
                ),
                "humidity_correlation": self.data["humidity"].corr(
                    self.data["performance_score"]
                ),
                "wind_correlation": self.data["wind_speed"].corr(
                    self.data["performance_score"]
                ),
            }

        return {
            "distance_stats": distance_stats,
            "grade_stats": grade_stats.sort_values("race_count", ascending=False).head(
                20
            ),
            "track_stats": track_stats,
            "venue_stats": venue_performance_stats.sort_values(
                "race_count", ascending=False
            ),
            "distance_correlation": distance_correlation,
            "weather_stats": weather_stats,
            "weather_performance_impact": weather_performance_impact,
        }

    def identify_top_performers(self, min_races=3):
        """Identify top performers using normalized metrics"""
        print(f"🏆 Identifying top performers (min {min_races} races)...")

        # Filter dogs with minimum race count
        frequent_dogs = self.data[self.data["race_count"] >= min_races].copy()

        # Calculate comprehensive performance metrics
        dog_rankings = (
            frequent_dogs.groupby("dog_name")
            .agg(
                {
                    "weighted_performance": "mean",
                    "performance_score": "mean",
                    "consistency_score": "mean",
                    "recent_form": "last",
                    "race_count": "first",
                    "avg_position": "first",
                    "career_span_days": "first",
                    "days_since_last_race": "first",
                }
            )
            .reset_index()
        )

        # Create composite score
        scaler = StandardScaler()
        score_features = ["weighted_performance", "consistency_score", "recent_form"]
        dog_rankings[score_features] = scaler.fit_transform(
            dog_rankings[score_features]
        )

        dog_rankings["composite_score"] = (
            dog_rankings["weighted_performance"] * 0.4
            + dog_rankings["consistency_score"] * 0.3
            + dog_rankings["recent_form"] * 0.3
        )

        return dog_rankings.sort_values("composite_score", ascending=False)

    def predict_performance(self, upcoming_races_df=None):
        """Build ML model to predict race performance"""
        print("🤖 Building performance prediction model...")

        # Prepare features for ML (including race condition, venue, and weather features)
        feature_cols = [
            "box_number",
            "field_size",
            "race_count",
            "avg_position",
            "position_std",
            "avg_weight",
            "races_per_month",
            "race_sequence",
            "consistency_score",
            "recent_form",
            "distance_numeric",
            "grade_numeric",
            "track_condition_numeric",
            "venue_encoded",
            "venue_avg_performance",
            "dog_performance_short",
            "dog_performance_medium",
            "dog_performance_long",
            "dog_avg_grade_performance",
            "dog_avg_track_performance",
            "dog_venue_avg_performance",
        ]

        # Add weather features if available
        weather_features = [
            "weather_condition_encoded",
            "temperature",
            "temp_deviation_from_optimal",
            "is_extreme_temperature",
            "weather_adjustment_factor",
            "humidity",
            "wind_speed",
            "pressure",
            "is_rain",
            "rain_performance_avg",
            "rain_experience",
        ]

        # Only add weather features that exist in the data
        available_weather_features = [
            f for f in weather_features if f in self.data.columns
        ]
        feature_cols.extend(available_weather_features)

        if available_weather_features:
            print(
                f"🌤️ Including {len(available_weather_features)} weather features in ML model"
            )

        # Filter complete cases and handle infinity values
        ml_data = self.data[feature_cols + ["performance_score"]].copy()

        # Replace infinity values with NaN
        ml_data = ml_data.replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN values
        ml_data = ml_data.dropna()

        # Additional safety check: clip extreme values
        for col in feature_cols:
            if ml_data[col].dtype in ["float64", "float32"]:
                q99 = ml_data[col].quantile(0.99)
                q01 = ml_data[col].quantile(0.01)
                ml_data[col] = np.clip(ml_data[col], q01, q99)

        X = ml_data[feature_cols]
        y = ml_data["performance_score"]

        print(f"📊 ML dataset: {len(ml_data)} valid samples")

        if len(ml_data) < 100:
            print("⚠️  Warning: Limited data for ML model")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Model trained - RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return model, feature_importance, {"rmse": rmse, "r2": r2}

    def generate_insights(self):
        """Generate comprehensive insights"""
        print("💡 Generating insights...")

        insights = {
            "data_summary": {
                "total_races": self.data["race_id"].nunique(),
                "total_dogs": self.data["dog_name"].nunique(),
                "total_entries": len(self.data),
                "date_range": f"{self.data['race_date'].min().date()} to {self.data['race_date'].max().date()}",
            },
            "frequency_analysis": {
                "single_race_dogs": len(
                    self.data[self.data["race_count"] == 1]["dog_name"].unique()
                ),
                "frequent_racers": len(
                    self.data[self.data["race_count"] > 3]["dog_name"].unique()
                ),
                "avg_races_per_dog": self.data["race_count"].mean(),
                "max_races_per_dog": self.data["race_count"].max(),
            },
            "performance_metrics": {
                "avg_performance_score": self.data["performance_score"].mean(),
                "performance_std": self.data["performance_score"].std(),
                "consistency_leader": self.data.loc[
                    self.data["consistency_score"].idxmax(), "dog_name"
                ],
            },
        }

        return insights

    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting comprehensive analysis...")

        # Load and prepare data
        self.load_data()
        self.engineer_features()
        self.normalize_performance()
        self.add_race_condition_features()

        # Perform analyses
        monthly_stats, venue_stats = self.temporal_analysis()
        race_condition_analysis = self.analyze_race_conditions()
        top_performers = self.identify_top_performers()
        model, feature_importance, model_metrics = self.predict_performance()
        insights = self.generate_insights()

        # Display results
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE ANALYSIS RESULTS")
        print("=" * 60)

        print(f"\n📈 DATA SUMMARY:")
        for key, value in insights["data_summary"].items():
            print(f"  {key}: {value}")

        print(f"\n🏁 RACE CONDITION ANALYSIS:")
        print(
            f"  Distance correlation with performance: {race_condition_analysis['distance_correlation']:.3f}"
        )
        print("\nDistance Categories:")
        print(race_condition_analysis["distance_stats"])
        print("\nTop Grades by Frequency:")
        print(race_condition_analysis["grade_stats"].head(10))
        print("\nTrack Conditions:")
        print(race_condition_analysis["track_stats"])
        print("\nVenue Performance (Top 10):")
        print(race_condition_analysis["venue_stats"].head(10))

        # Weather analysis results
        if race_condition_analysis.get("weather_stats") is not None:
            print(f"\n🌤️ WEATHER CONDITION ANALYSIS:")
            print(race_condition_analysis["weather_stats"])

            if race_condition_analysis.get("weather_performance_impact"):
                impact = race_condition_analysis["weather_performance_impact"]
                print(f"\n🌦️ Weather Performance Impact:")
                clear_vs_rain = impact.get("clear_vs_rain", {})
                if clear_vs_rain:
                    print(
                        f"  Clear weather avg: {clear_vs_rain.get('clear_avg', 0):.3f}"
                    )
                    print(f"  Rain avg: {clear_vs_rain.get('rain_avg', 0):.3f}")
                    print(
                        f"  Performance difference: {clear_vs_rain.get('difference', 0):.3f}"
                    )

                print(
                    f"  Temperature correlation: {impact.get('temperature_correlation', 0):.3f}"
                )
                print(
                    f"  Humidity correlation: {impact.get('humidity_correlation', 0):.3f}"
                )
                print(f"  Wind correlation: {impact.get('wind_correlation', 0):.3f}")
        else:
            print(f"\n🌤️ No weather data available for analysis")

        print(f"\n🔢 FREQUENCY ANALYSIS:")
        for key, value in insights["frequency_analysis"].items():
            print(f"  {key}: {value}")

        print(f"\n🏆 TOP 10 PERFORMERS (Normalized):")
        for i, row in top_performers.head(10).iterrows():
            print(
                f"  {row['dog_name']}: Score {row['composite_score']:.3f} "
                f"({row['race_count']} races, Avg pos: {row['avg_position']:.1f})"
            )

        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"  RMSE: {model_metrics['rmse']:.4f}")
        print(f"  R²: {model_metrics['r2']:.4f}")

        print(f"\n🔍 TOP FEATURE IMPORTANCE:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        return {
            "top_performers": top_performers,
            "monthly_stats": monthly_stats,
            "venue_stats": venue_stats,
            "race_condition_analysis": race_condition_analysis,
            "model": model,
            "feature_importance": feature_importance,
            "insights": insights,
        }

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main execution function"""
    analyzer = EnhancedRaceAnalyzer()
    results = analyzer.run_comprehensive_analysis()

    print("\n✅ Analysis complete!")
    print("📄 Results saved to analyzer object")


if __name__ == "__main__":
    main()
