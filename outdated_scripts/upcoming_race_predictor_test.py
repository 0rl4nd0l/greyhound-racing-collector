#!/usr/bin/env python3
"""
Upcoming Race Predictor - Test Version
=====================================

This script analyzes upcoming races (without results) and makes predictions
based on historical data and form analysis.

Author: AI Assistant
Date: July 24, 2025
"""

import json
import os
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Advanced ML and analysis libraries
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier, VotingClassifier)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, log_loss
    from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                         train_test_split)
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import enhanced feature engineering
try:
    from enhanced_feature_engineering import EnhancedFeatureEngineer

    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    print("⚠️ Enhanced feature engineering not available")

# Import enhanced race processor
try:
    from enhanced_race_processor_fixed import EnhancedRaceProcessor

    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError:
    ENHANCED_PROCESSOR_AVAILABLE = False
    print("⚠️ Enhanced race processor not available")


class UpcomingRacePredictor:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.upcoming_dir = Path("./upcoming_races")
        self.predictions_dir = Path("./predictions")

        # Create predictions directory
        self.predictions_dir.mkdir(exist_ok=True)

        # Advanced features
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.ml_models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}

        # Enhanced analysis flags
        self.use_ml_predictions = SKLEARN_AVAILABLE
        self.use_fuzzy_matching = False
        self.use_weather_data = False
        self.use_enhanced_features = ENHANCED_FEATURES_AVAILABLE
        self.use_enhanced_processor = ENHANCED_PROCESSOR_AVAILABLE

        # Initialize enhanced feature engineering
        if self.use_enhanced_features:
            self.feature_engineer = EnhancedFeatureEngineer(db_path)
        else:
            self.feature_engineer = None

        # Initialize enhanced race processor
        if self.use_enhanced_processor:
            self.race_processor = EnhancedRaceProcessor(db_path)
        else:
            self.race_processor = None

        print(f"🎯 Advanced Predictor Initialized")
        print(f"✅ ML Available: {SKLEARN_AVAILABLE}")
        print(f"✅ Enhanced Features: {ENHANCED_FEATURES_AVAILABLE}")
        print(f"✅ Enhanced Processor: {ENHANCED_PROCESSOR_AVAILABLE}")

    def extract_form_data_from_csv(self, dog_name, df):
        """Extract form data for a dog directly from the CSV file"""
        try:
            # Find all rows for this dog (including blank dog name rows that follow)
            dog_rows = []
            found_dog = False

            for idx, row in df.iterrows():
                if pd.notna(row.get("Dog Name", "")) and row["Dog Name"].strip():
                    current_dog = row["Dog Name"].strip()
                    # Extract dog name from numbered format (e.g., "1. Steely Mac" -> "Steely Mac")
                    clean_current = current_dog
                    if ". " in current_dog:
                        clean_current = current_dog.split(". ", 1)[1].strip()

                    found_dog = clean_current.upper() == dog_name.upper()

                if found_dog:
                    # Check if this row has race data
                    if pd.notna(row.get("PLC")) and pd.notna(row.get("TIME")):
                        try:
                            form_entry = {
                                "position": (
                                    int(row["PLC"]) if pd.notna(row["PLC"]) else None
                                ),
                                "time": (
                                    float(row["TIME"])
                                    if pd.notna(row["TIME"])
                                    else None
                                ),
                                "distance": (
                                    int(row["DIST"]) if pd.notna(row["DIST"]) else None
                                ),
                                "weight": (
                                    float(row["WGT"]) if pd.notna(row["WGT"]) else None
                                ),
                                "box": (
                                    int(row["BOX"]) if pd.notna(row["BOX"]) else None
                                ),
                                "margin": (
                                    float(row["MGN"])
                                    if pd.notna(row["MGN"])
                                    and str(row["MGN"])
                                    .replace(".", "")
                                    .replace("-", "")
                                    .isdigit()
                                    else 0
                                ),
                                "date": row["DATE"] if pd.notna(row["DATE"]) else None,
                                "track": (
                                    row["TRACK"] if pd.notna(row["TRACK"]) else None
                                ),
                                "grade": row["G"] if pd.notna(row["G"]) else None,
                                "sectional_1": (
                                    float(row["1 SEC"])
                                    if pd.notna(row["1 SEC"])
                                    and str(row["1 SEC"]).replace(".", "").isdigit()
                                    else None
                                ),
                                "win_time": (
                                    float(row["WIN"]) if pd.notna(row["WIN"]) else None
                                ),
                                "bonus_time": (
                                    float(row["BON"]) if pd.notna(row["BON"]) else None
                                ),
                                "sex": row["Sex"] if pd.notna(row["Sex"]) else None,
                                "starting_price": (
                                    float(row["SP"]) if pd.notna(row["SP"]) else None
                                ),
                                "winner_runner_up": (
                                    row["W/2G"] if pd.notna(row["W/2G"]) else None
                                ),
                                "position_in_race": (
                                    row["PIR"] if pd.notna(row["PIR"]) else None
                                ),
                            }
                            dog_rows.append(form_entry)
                            print(
                                f"    📊 Extracted form entry: PLC={form_entry['position']}, SP={form_entry['starting_price']}, PIR={form_entry['position_in_race']}, Sex={form_entry['sex']}"
                            )
                        except (ValueError, TypeError) as e:
                            print(f"    ⚠️ Error parsing row: {e}")
                            continue

                # Stop when we hit the next numbered dog
                elif (
                    found_dog
                    and pd.notna(row.get("Dog Name", ""))
                    and row["Dog Name"].strip()
                    and any(char.isdigit() for char in row["Dog Name"][:3])
                ):
                    break

            print(f"  📋 Extracted {len(dog_rows)} form entries for {dog_name}")
            return dog_rows

        except Exception as e:
            print(f"Error extracting form data for {dog_name}: {e}")
            return []

    def analyze_form_data(self, form_data, dog_name):
        """Analyze extracted form data and calculate performance metrics"""

        # Utility functions
        def calculate_time_behind(entry):
            return (
                entry["time"] - entry["win_time"]
                if entry["time"] and entry["win_time"]
                else None
            )

        def calculate_relative_performance(entry):
            win_time = entry["win_time"] if entry["win_time"] else entry["time"]
            return (entry["time"] / win_time) * 100 if win_time else None

        if not form_data:
            return None

        try:
            # Extract relevant metrics
            positions = [
                entry["position"]
                for entry in form_data
                if entry["position"] is not None
            ]
            times = [entry["time"] for entry in form_data if entry["time"] is not None]
            weights = [
                entry["weight"] for entry in form_data if entry["weight"] is not None
            ]
            margins = [
                entry["margin"] for entry in form_data if entry["margin"] is not None
            ]
            sectionals = [
                entry["sectional_1"]
                for entry in form_data
                if entry["sectional_1"] is not None
            ]
            distances = [
                entry["distance"]
                for entry in form_data
                if entry["distance"] is not None
            ]

            # New metrics from unused data
            starting_prices = [
                entry["starting_price"]
                for entry in form_data
                if entry["starting_price"] is not None
            ]
            winners_beaten = [
                entry["winner_runner_up"]
                for entry in form_data
                if entry["winner_runner_up"] is not None
            ]
            pir_data = [
                entry["position_in_race"]
                for entry in form_data
                if entry["position_in_race"] is not None
            ]
            sexes = [entry["sex"] for entry in form_data if entry["sex"] is not None]
            grades = [
                entry["grade"] for entry in form_data if entry["grade"] is not None
            ]

            print(
                f"    📊 Data extracted - Positions: {len(positions)}, SPs: {len(starting_prices)}, PIR: {len(pir_data)}, Sex: {len(sexes)}"
            )

            if not positions or not times:
                return None

            # Calculate comprehensive statistics including new metrics
            stats = {
                "races_count": len(form_data),
                "avg_position": np.mean(positions) if positions else 0,
                "median_position": np.median(positions) if positions else 0,
                "win_rate": (
                    sum(1 for p in positions if p == 1) / len(positions)
                    if positions
                    else 0
                ),
                "place_rate": (
                    sum(1 for p in positions if p <= 3) / len(positions)
                    if positions
                    else 0
                ),
                "top_half_rate": (
                    sum(1 for p in positions if p <= 4) / len(positions)
                    if positions
                    else 0
                ),
                "avg_time": np.mean(times) if times else 0,
                "best_time": min(times) if times else 0,
                "time_consistency": 1 / (np.std(times) + 0.1) if len(times) > 1 else 1,
                "position_consistency": (
                    1 / (np.std(positions) + 1) if len(positions) > 1 else 1
                ),
                "recent_form": positions[:5] if positions else [],
                "form_trend": (
                    self.calculate_form_trend(positions[:10])
                    if len(positions) >= 3
                    else 0
                ),
                "avg_weight": np.mean(weights) if weights else 30.0,
                "avg_margin": np.mean([abs(m) for m in margins]) if margins else 0,
                "avg_sectional": np.mean(sectionals) if sectionals else 0,
                "distance_preference": (
                    max(set(distances), key=distances.count) if distances else None
                ),
                # Speed ratings based on times and distances
                "speed_index": self.calculate_speed_index(form_data),
                "class_assessment": self.assess_class_from_form(form_data),
                "track_versatility": len(
                    set(entry["track"] for entry in form_data if entry["track"])
                ),
                "recent_activity": self.calculate_recent_activity_from_form(form_data),
                "trainer_stability": 0.5,  # Default for CSV form data
                "class_progression": 0.0,  # Default for CSV form data
                # Default ratings derived from performance
                "avg_performance_rating": self.derive_performance_rating(
                    positions, times
                ),
                "avg_speed_rating": self.derive_speed_rating(times, distances),
                "avg_class_rating": self.derive_class_rating(form_data),
                # NEW METRICS FROM UNUSED DATA
                # Market Intelligence (SP - Starting Price)
                "avg_starting_price": (
                    np.mean(starting_prices) if starting_prices else 10.0
                ),
                "price_consistency": (
                    1 / (np.std(starting_prices) + 0.1)
                    if len(starting_prices) > 1
                    else 1
                ),
                "market_confidence": self.calculate_market_confidence(
                    starting_prices, positions
                ),
                "value_indicator": self.calculate_value_indicator(
                    starting_prices, positions
                ),
                "market_position_trend": self.analyze_market_trends(starting_prices),
                # Competition Quality (W/2G - Winner/Runner-up)
                "competition_quality": self.analyze_competition_quality(winners_beaten),
                "quality_opponents_beaten": self.count_quality_opponents(
                    winners_beaten, positions
                ),
                "field_strength_faced": self.assess_field_strength(winners_beaten),
                # Position in Race Analysis (PIR)
                "sectional_analysis": self.analyze_sectional_positions(pir_data),
                "running_style": self.determine_running_style(pir_data),
                "tactical_ability": self.assess_tactical_ability(pir_data, positions),
                # Gender Analysis (Sex)
                "gender": sexes[0] if sexes else "Unknown",
                "gender_performance_modifier": self.calculate_gender_modifier(
                    sexes[0] if sexes else None, distances, positions
                ),
                # Enhanced Grade Analysis
                "grade_distribution": self.analyze_grade_distribution(grades),
                "class_level_assessment": self.assess_class_levels(grades, positions),
                "grade_progression_detailed": self.track_grade_progression(
                    grades, form_data
                ),
                # Advanced Time Analysis
                "time_behind_winner_avg": np.mean(
                    [
                        calculate_time_behind(entry)
                        for entry in form_data
                        if calculate_time_behind(entry) is not None
                    ]
                ),
                "relative_performance_avg": np.mean(
                    [
                        calculate_relative_performance(entry)
                        for entry in form_data
                        if calculate_relative_performance(entry) is not None
                    ]
                ),
                "closing_sectional_strength": self.analyze_closing_speed(form_data),
                # Raw data for ML features
                "raw_results": form_data,
            }

            print(
                f"    ✅ Analysis complete - Win rate: {stats['win_rate']:.1%}, Avg SP: {stats['avg_starting_price']:.1f}, Running style: {stats['running_style']}"
            )
            return stats

        except Exception as e:
            print(f"Error analyzing form data for {dog_name}: {e}")
            return None

    # New methods for analyzing unused data fields
    def calculate_market_confidence(self, starting_prices, positions):
        """Calculate market confidence from starting prices and positions"""
        if not starting_prices:
            return 0
        try:
            # Confidence is higher for stable markets with many favorites performing well
            volatility = np.std(starting_prices)
            favored_positions = sum(
                1 for sp, pos in zip(starting_prices, positions) if sp < 5 and pos < 3
            )

            # Improved formula: normalize volatility and emphasize successful favorites
            # Base confidence from successful favorites
            favorite_success_rate = (
                favored_positions / len(starting_prices) if starting_prices else 0
            )

            # Volatility penalty (normalize by average price to make it scale-independent)
            avg_price = np.mean(starting_prices)
            normalized_volatility = volatility / (
                avg_price + 1
            )  # +1 to avoid division by zero
            volatility_penalty = min(0.5, normalized_volatility)  # Cap penalty at 0.5

            confidence = max(
                0, min(1, favorite_success_rate + 0.3 - volatility_penalty)
            )

            return confidence
        except Exception as e:
            print(f"    ⚠️ Market confidence error: {e}")
            return 0

    def analyze_competition_quality(self, winners_beaten):
        """Analyze the quality of competition based on winners beaten"""
        if not winners_beaten:
            return 0
        return len(set(winners_beaten))

    def analyze_sectional_positions(self, pir_data):
        """Analyze sectional positions based on position-in-race data"""
        try:
            if not pir_data:
                return [0, 0, 0, 0]

            # Convert PIR data to string format for analysis
            pir_strings = []
            for p in pir_data:
                pir_str = str(p).zfill(4)  # Pad with zeros to ensure 4 digits
                if len(pir_str) >= 3:  # Need at least 3 positions
                    pir_strings.append(pir_str)

            if not pir_strings:
                return [0, 0, 0, 0]

            # Calculate average positions at each call
            avg_positions = []
            for i in range(4):
                positions_at_call = []
                for pir_str in pir_strings:
                    if i < len(pir_str):
                        try:
                            pos = int(pir_str[i])
                            if pos > 0:  # Valid position
                                positions_at_call.append(pos)
                        except ValueError:
                            continue

                if positions_at_call:
                    avg_positions.append(np.mean(positions_at_call))
                else:
                    avg_positions.append(0)

            return avg_positions
        except Exception as e:
            print(f"    ⚠️ PIR analysis error: {e}")
            return [0, 0, 0, 0]

    def determine_running_style(self, pir_data):
        """Determine running style based on average early and late positions"""
        try:
            if not pir_data:
                return "Unknown"

            early_positions = []
            late_positions = []

            for p in pir_data:
                pir_str = str(p).zfill(4)  # Pad with zeros
                if len(pir_str) >= 3:
                    try:
                        early_pos = int(pir_str[0])  # First call
                        late_pos = int(pir_str[2])  # Third call
                        if early_pos > 0 and late_pos > 0:
                            early_positions.append(early_pos)
                            late_positions.append(late_pos)
                    except (ValueError, IndexError):
                        continue

            if not early_positions or not late_positions:
                return "Unknown"

            avg_early = np.mean(early_positions)
            avg_late = np.mean(late_positions)

            print(
                f"    🏃 PIR Analysis: Early avg={avg_early:.1f}, Late avg={avg_late:.1f}"
            )

            if abs(avg_early - avg_late) < 0.5:
                return "Consistent"
            elif avg_late < avg_early:
                return "Finisher"
            else:
                return "Fader"
        except Exception as e:
            print(f"    ⚠️ Running style analysis error: {e}")
            return "Unknown"

    def calculate_gender_modifier(self, gender, distances, positions):
        """Calculate a gender performance modifier"""
        if not gender or not positions:
            return 1.0
        try:
            if gender.lower() == "b":  # Bitch
                favored_distance = (
                    max(set(distances), key=distances.count) if distances else None
                )
                if favored_distance:
                    favored_rate = sum(
                        1
                        for d, p in zip(distances, positions)
                        if d == favored_distance and p < 3
                    ) / len(distances)
                    return 1.05 if favored_rate > 0.3 else 1.0
            return 1.0
        except:
            return 1.0

    def calculate_value_indicator(self, starting_prices, positions):
        """Calculate a value indicator based on starting prices and performance"""
        try:
            underpriced_success_rate = sum(
                1 for sp, pos in zip(starting_prices, positions) if sp > 5 and pos < 3
            ) / len(starting_prices)
            return max(0, min(1, underpriced_success_rate))
        except:
            return 0

    def analyze_market_trends(self, starting_prices):
        """Analyze market trends from starting prices"""
        try:
            if len(starting_prices) > 5:
                trend = np.polyfit(range(len(starting_prices)), starting_prices, 1)[0]
                return trend
            return 0
        except:
            return 0

    def count_quality_opponents(self, winners_beaten, positions):
        """Count quality opponents beaten"""
        try:
            return sum(
                1
                for w in winners_beaten
                if w and positions[winners_beaten.index(w)] < 3
            )
        except:
            return 0

    def assess_field_strength(self, winners_beaten):
        """Assess the field strength"""
        try:
            return len(set(winners_beaten)) * 0.1
        except:
            return 0

    def assess_tactical_ability(self, pir_data, positions):
        """Assess tactical ability based on PIR and final positions"""
        try:
            improvements = sum(int(p[-1]) < int(p[0]) for p in pir_data if len(p) > 3)
            return improvements / len(positions) if positions else 0
        except:
            return 0

    def analyze_grade_distribution(self, grades):
        """Analyze grade distribution"""
        try:
            if not grades:
                return {}
            return {g: grades.count(g) / len(grades) for g in set(grades)}
        except:
            return {}

    def assess_class_levels(self, grades, positions):
        """Assess class levels relative to positions"""
        try:
            return sum(p < 3 for p in positions) / len(positions)
        except:
            return 0

    def track_grade_progression(self, grades, form_data):
        """Track grade progression over form data"""
        try:
            return len(set(grades)) / len(form_data) if grades and form_data else 0
        except:
            return 0

    def analyze_closing_speed(self, form_data):
        """Analyze closing speed from sectional times"""
        try:
            sectional_times = [
                entry["sectional_1"] for entry in form_data if entry["sectional_1"]
            ]
            overall_times = [entry["time"] for entry in form_data if entry["time"]]
            closing_speeds = [
                o - s for s, o in zip(sectional_times, overall_times) if o and s
            ]
            return np.mean(closing_speeds) if closing_speeds else 0
        except:
            return 0

    # Supporting methods
    def calculate_speed_index(self, form_data):
        """Calculate speed index based on times and distances"""
        try:
            if not form_data:
                return 50.0

            speed_ratings = []
            for entry in form_data:
                if entry.get("time") and entry.get("distance"):
                    speed = entry["distance"] / entry["time"]
                    speed_ratings.append(speed)

            if speed_ratings:
                avg_speed = np.mean(speed_ratings)
                return min(100, max(0, (avg_speed - 15) * 5))

            return 50.0
        except:
            return 50.0

    def assess_class_from_form(self, form_data):
        """Assess class level from form data"""
        try:
            if not form_data:
                return 50.0

            grades = [entry.get("grade", "") for entry in form_data]
            grade_scores = []

            for grade in grades:
                if not grade:
                    continue

                grade_str = str(grade).upper()

                if "5" in grade_str:
                    grade_scores.append(90)
                elif "4" in grade_str:
                    grade_scores.append(70)
                elif "3" in grade_str:
                    grade_scores.append(50)
                elif "2" in grade_str:
                    grade_scores.append(40)
                elif "1" in grade_str:
                    grade_scores.append(35)
                elif grade_str in ["MAIDEN", "MDN", "M"]:
                    grade_scores.append(25)
                else:
                    grade_scores.append(50)

            return np.mean(grade_scores) if grade_scores else 50.0
        except:
            return 50.0

    def calculate_recent_activity_from_form(self, form_data):
        """Calculate recent activity from form data dates"""
        try:
            if not form_data:
                return {
                    "days_since_last_race": 365,
                    "recent_frequency": 0,
                    "activity_score": 0,
                }

            dates = []
            for entry in form_data:
                if entry.get("date"):
                    try:
                        date_obj = pd.to_datetime(entry["date"])
                        dates.append(date_obj)
                    except:
                        continue

            if not dates:
                return {
                    "days_since_last_race": 365,
                    "recent_frequency": 0,
                    "activity_score": 0,
                }

            dates.sort(reverse=True)
            now = pd.Timestamp.now()

            days_since_last = (now - dates[0]).days if dates else 365
            six_months_ago = now - pd.Timedelta(days=180)
            recent_races = sum(1 for date in dates if date >= six_months_ago)
            frequency = recent_races / 6.0
            activity_score = max(0, min(1, (60 - days_since_last) / 60)) * min(
                1, frequency / 2
            )

            return {
                "days_since_last_race": days_since_last,
                "recent_frequency": frequency,
                "activity_score": activity_score,
            }
        except:
            return {
                "days_since_last_race": 30,
                "recent_frequency": 1,
                "activity_score": 0.5,
            }

    def derive_performance_rating(self, positions, times):
        """Derive performance rating from positions and times"""
        try:
            if not positions:
                return 50.0

            avg_position = np.mean(positions)
            position_rating = max(0, min(100, (8 - avg_position) * 12.5))

            if len(times) > 1:
                time_std = np.std(times)
                consistency_bonus = max(0, 10 - time_std)
                position_rating += consistency_bonus

            return min(100, position_rating)
        except:
            return 50.0

    def derive_speed_rating(self, times, distances):
        """Derive speed rating from times and distances"""
        try:
            if not times or not distances:
                return 50.0

            speed_ratings = []
            for i, time in enumerate(times):
                if i < len(distances) and distances[i] and time:
                    speed = distances[i] / time
                    normalized = (speed - 15) * 4
                    speed_ratings.append(max(0, min(100, normalized)))

            return np.mean(speed_ratings) if speed_ratings else 50.0
        except:
            return 50.0

    def derive_class_rating(self, form_data):
        """Derive class rating from form data"""
        try:
            return self.assess_class_from_form(form_data)
        except:
            return 50.0

    def calculate_form_trend(self, positions):
        """Calculate form trend (positive = improving, negative = declining)"""
        if len(positions) < 3:
            return 0
        try:
            x = np.arange(len(positions))
            y = np.array(positions)
            slope = np.polyfit(x, y, 1)[0]
            return -slope  # Negative slope (improving positions) = positive trend
        except:
            return 0

    def get_comprehensive_dog_performance(self, dog_name, df=None):
        """Get comprehensive historical performance data with advanced metrics"""
        # First try to get data from CSV form if available
        if df is not None:
            form_data = self.extract_form_data_from_csv(dog_name, df)
            if form_data:
                return self.analyze_form_data(form_data, dog_name)

        # Fallback to database query
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT 
                    drd.finish_position,
                    drd.individual_time,
                    drd.margin,
                    drd.starting_price,
                    drd.performance_rating,
                    drd.speed_rating,
                    drd.class_rating,
                    drd.win_probability,
                    drd.place_probability,
                    drd.weight,
                    drd.box_number,
                    drd.trainer_name,
                    drd.recent_form,
                    drd.best_time,
                    rm.venue,
                    rm.race_date,
                    rm.distance,
                    rm.track_condition,
                    rm.weather,
                    rm.temperature,
                    rm.field_size,
                    rm.grade
                FROM dog_race_data drd
                JOIN race_metadata rm ON drd.race_id = rm.race_id
                WHERE drd.dog_clean_name = ? 
                AND drd.finish_position IS NOT NULL 
                AND drd.finish_position != ''
                ORDER BY rm.race_date DESC, drd.extraction_timestamp DESC
                LIMIT 20
            """,
                (dog_name.upper(),),
            )

            results = cursor.fetchall()

            if not results:
                return None

            # Enhanced statistical calculations
            positions = [int(r[0]) for r in results if r[0] and str(r[0]).isdigit()]
            times = [
                float(r[1])
                for r in results
                if r[1] and str(r[1]).replace(".", "").isdigit()
            ]
            odds = [
                float(r[3])
                for r in results
                if r[3] and str(r[3]).replace(".", "").replace("-", "").isdigit()
            ]
            weights = [
                float(r[9])
                for r in results
                if r[9] and str(r[9]).replace(".", "").isdigit()
            ]
            venues = [r[14] for r in results if r[14]]
            distances = [r[16] for r in results if r[16]]

            # Advanced statistics
            stats = {
                "races_count": len(results),
                "avg_position": np.mean(positions) if positions else 0,
                "median_position": np.median(positions) if positions else 0,
                "win_rate": (
                    sum(1 for p in positions if p == 1) / len(positions)
                    if positions
                    else 0
                ),
                "place_rate": (
                    sum(1 for p in positions if p <= 3) / len(positions)
                    if positions
                    else 0
                ),
                "top_half_rate": (
                    sum(1 for p in positions if p <= 4) / len(positions)
                    if positions
                    else 0
                ),
                "avg_time": np.mean(times) if times else 0,
                "best_time": min(times) if times else 0,
                "time_consistency": 1 / (np.std(times) + 0.1) if len(times) > 1 else 1,
                "position_consistency": (
                    1 / (np.std(positions) + 1) if len(positions) > 1 else 1
                ),
                "recent_form": positions[:5] if positions else [],
                "form_trend": (
                    self.calculate_form_trend(positions[:10])
                    if len(positions) >= 3
                    else 0
                ),
                "avg_odds": np.mean(odds) if odds else 10.0,
                "avg_weight": np.mean(weights) if weights else 30.0,
                "venue_diversity": len(set(venues)) if venues else 0,
                "distance_preference": (
                    max(set(distances), key=distances.count) if distances else None
                ),
                # Original ratings (enhanced)
                "avg_performance_rating": (
                    np.mean([r[4] for r in results if r[4]])
                    if any(r[4] for r in results)
                    else 50
                ),
                "avg_speed_rating": (
                    np.mean([r[5] for r in results if r[5]])
                    if any(r[5] for r in results)
                    else 50
                ),
                "avg_class_rating": (
                    np.mean([r[6] for r in results if r[6]])
                    if any(r[6] for r in results)
                    else 50
                ),
                # Raw data for ML features
                "raw_results": results,
            }

            return stats

        except Exception as e:
            print(f"Error getting historical data for {dog_name}: {e}")
            return None
        finally:
            conn.close()

    def calculate_enhanced_prediction_score(self, dog_stats, race_context=None):
        """Calculate enhanced prediction score using advanced features"""
        if not dog_stats:
            return 0.1

        score = 0

        # Primary performance metrics (50% total)
        score += dog_stats["win_rate"] * 0.20
        score += dog_stats["place_rate"] * 0.15
        score += dog_stats["top_half_rate"] * 0.10

        # Position quality (15%)
        if dog_stats["avg_position"] > 0:
            position_score = (8 - dog_stats["avg_position"]) / 8
            score += position_score * 0.15

        # Consistency and form (20% total)
        score += dog_stats["position_consistency"] * 0.08
        score += dog_stats["time_consistency"] * 0.07

        form_trend_normalized = max(0, min(1, (dog_stats["form_trend"] + 2) / 4))
        score += form_trend_normalized * 0.05

        # Recent activity and fitness (10% total)
        activity = dog_stats.get("recent_activity", {})
        if isinstance(activity, dict):
            score += activity.get("activity_score", 0.5) * 0.10

        # Class and experience (5% total)
        score += min(1.0, dog_stats["races_count"] / 20) * 0.03

        return min(score, 1.0)

    def get_enhanced_bet_recommendation(self, score, dog_stats):
        """Get enhanced betting recommendation"""
        confidence = self.calculate_confidence_level(dog_stats)

        if score >= 0.7 and confidence >= 0.8:
            return "Strong Win"
        elif score >= 0.5 and confidence >= 0.6:
            return "Win/Place"
        elif score >= 0.3 and confidence >= 0.4:
            return "Place Only"
        elif score >= 0.2:
            return "Each-Way"
        else:
            return "Avoid"

    def calculate_confidence_level(self, dog_stats):
        """Calculate confidence level based on data quality"""
        if not dog_stats:
            return 0.1

        confidence = 0.5

        races_count = dog_stats.get("races_count", 0)
        confidence += min(0.3, races_count / 20)

        activity = dog_stats.get("recent_activity", {})
        if isinstance(activity, dict):
            confidence += activity.get("activity_score", 0) * 0.2

        return min(confidence, 1.0)

    def identify_key_factors(self, dog_stats, race_context):
        """Identify key factors influencing the prediction"""
        factors = []

        if not dog_stats:
            factors.append("No historical data available")
            return factors

        if dog_stats.get("win_rate", 0) > 0.3:
            factors.append(f"Strong win rate ({dog_stats['win_rate']:.1%})")

        if dog_stats.get("form_trend", 0) > 0.5:
            factors.append("Improving form")
        elif dog_stats.get("form_trend", 0) < -0.5:
            factors.append("Declining form")

        if dog_stats.get("position_consistency", 0) > 0.8:
            factors.append("Highly consistent")

        activity = dog_stats.get("recent_activity", {})
        if isinstance(activity, dict):
            days_since = activity.get("days_since_last_race", 30)
            if days_since < 14:
                factors.append("Recently active")
            elif days_since > 60:
                factors.append("Long layoff")

        races_count = dog_stats.get("races_count", 0)
        if races_count < 3:
            factors.append("Limited experience")
        elif races_count > 15:
            factors.append("Highly experienced")

        return factors

    def clean_dog_name(self, name):
        """Clean dog name for database lookup"""
        if not name:
            return ""

        import re

        cleaned = re.sub(r'^["\d\.\s]+', "", str(name))
        cleaned = re.sub(r'["\s]+$', "", cleaned)
        return cleaned.strip().upper()

    def enhanced_clean_dog_name(self, name):
        """Clean dog name using exact matching only"""
        return self.clean_dog_name(name)

    def get_race_context(self, race_info, df):
        """Get current race context for predictions"""
        context = {
            "venue": race_info.get("venue", "Unknown"),
            "race_date": race_info.get(
                "race_date", datetime.now().strftime("%Y-%m-%d")
            ),
            "field_size": len(df),
        }

        if not df.empty:
            first_row = df.iloc[0]

            if "DIST" in df.columns:
                dist_value = first_row.get("DIST")
                if pd.notna(dist_value) and dist_value:
                    try:
                        context["distance"] = str(int(float(dist_value)))
                        print(f"   📏 Extracted race distance: {context['distance']}m")
                    except (ValueError, TypeError):
                        context["distance"] = "500"
                        print(f"   ⚠️ Could not parse distance, using 500m default")
                else:
                    context["distance"] = "500"
            else:
                context["distance"] = "500"

            if "G" in df.columns:
                grade_value = first_row.get("G")
                if pd.notna(grade_value) and grade_value:
                    grade_str = str(grade_value).strip()
                    if grade_str.upper() in ["MAIDEN", "MDN", "M", "MAID"]:
                        context["grade"] = "Maiden"
                    else:
                        context["grade"] = grade_str
                    print(f"   🏁 Extracted race grade: {context['grade']}")

            context["track_condition"] = "Good"  # Default

        return context

    def predict_race(self, race_file_path):
        """Predict outcomes for a single race using enhanced analysis"""
        try:
            df = pd.read_csv(race_file_path)

            filename = os.path.basename(race_file_path)
            race_info = self.extract_race_info(filename)

            race_context = self.get_race_context(race_info, df)

            predictions = []
            print(f"   🎯 Analyzing {len(df)} dogs with enhanced metrics...")

            for _, row in df.iterrows():
                raw_dog_name = str(row.get("Dog Name", row.get("DOG", ""))).strip()
                if (
                    not raw_dog_name
                    or raw_dog_name.lower() == "nan"
                    or raw_dog_name == '""'
                ):
                    continue

                import re

                box_number_pattern = r"^(\d+)\."
                match_box = re.match(box_number_pattern, raw_dog_name)
                box_number = match_box.group(1) if match_box else ""

                dog_name = re.sub(box_number_pattern, "", raw_dog_name).strip()
                clean_name = self.enhanced_clean_dog_name(dog_name)

                print(f"  📊 Analyzing: {dog_name}")
                dog_stats = self.get_comprehensive_dog_performance(clean_name, df)

                prediction_score = self.calculate_enhanced_prediction_score(
                    dog_stats, race_context
                )

                if not box_number:
                    box_number = row.get("BOX", "")
                trainer = row.get("TRAINER", "")
                weight = row.get("WEIGHT", "")

                prediction = {
                    "dog_name": dog_name,
                    "clean_name": clean_name,
                    "box_number": box_number,
                    "trainer": trainer,
                    "weight": weight,
                    "prediction_score": prediction_score,
                    "historical_stats": dog_stats,
                    "recommended_bet": self.get_enhanced_bet_recommendation(
                        prediction_score, dog_stats
                    ),
                    "confidence_level": self.calculate_confidence_level(dog_stats),
                    "key_factors": self.identify_key_factors(dog_stats, race_context),
                }

                predictions.append(prediction)
                print(
                    f"     📊 {dog_name}: {prediction_score:.1%} confidence - {prediction['recommended_bet']}"
                )

            predictions.sort(key=lambda x: x["prediction_score"], reverse=True)

            for i, pred in enumerate(predictions):
                pred["predicted_rank"] = i + 1

            race_summary = {
                "total_dogs": len(predictions),
                "dogs_with_data": len(
                    [p for p in predictions if p.get("historical_stats")]
                ),
                "average_confidence": np.mean(
                    [p.get("confidence_level", 0) for p in predictions]
                ),
            }

            return {
                "race_info": race_info,
                "race_context": race_context,
                "predictions": predictions,
                "top_pick": predictions[0] if predictions else None,
                "race_summary": race_summary,
                "prediction_timestamp": datetime.now().isoformat(),
                "analysis_version": "2.0_enhanced_test",
            }

        except Exception as e:
            print(f"Error predicting race {race_file_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def extract_race_info(self, filename):
        """Extract race information from filename"""
        import re

        race_pattern = r"Race\s+(\d+)\s+-\s+([A-Z_]+)\s+-\s+([\d\-]+)"
        match = re.search(race_pattern, filename, re.IGNORECASE)

        if match:
            race_number, venue, date_str = match.groups()
            return {
                "race_number": int(race_number),
                "venue": venue,
                "date_str": date_str,
                "filename": filename,
            }

        return {"filename": filename}


def test_analysis_pipeline(csv_file_path):
    """Test function for the complete analysis pipeline"""
    print(f"🧪 TESTING ANALYSIS PIPELINE")
    print(f"=" * 50)
    print(f"📁 Test file: {csv_file_path}")

    predictor = UpcomingRacePredictor()
    result = predictor.predict_race(csv_file_path)

    if result and "predictions" in result:
        print(f"\n✅ PIPELINE TEST SUCCESSFUL")
        print(f"📊 Race Summary:")
        print(f"   - Total dogs analyzed: {result['race_summary']['total_dogs']}")
        print(
            f"   - Dogs with historical data: {result['race_summary']['dogs_with_data']}"
        )
        print(
            f"   - Average confidence: {result['race_summary']['average_confidence']:.1%}"
        )

        print(f"\n🏆 Top 3 Predictions:")
        for i, pred in enumerate(result["predictions"][:3]):
            stats = pred.get("historical_stats", {})
            enhanced_data = ""
            if stats:
                enhanced_data = f" | SP Avg: {stats.get('avg_starting_price', 0):.1f} | Style: {stats.get('running_style', 'Unknown')} | Gender: {stats.get('gender', 'Unknown')}"

            print(f"   {i+1}. {pred['dog_name']} (Box {pred['box_number']})")
            print(
                f"      Score: {pred['prediction_score']:.1%} | {pred['recommended_bet']} | Confidence: {pred['confidence_level']:.1%}"
            )
            print(f"      Key factors: {', '.join(pred.get('key_factors', ['None']))}")
            print(f"      Enhanced data{enhanced_data}")

        print(f"\n🎯 ENHANCED DATA VERIFICATION:")
        for pred in result["predictions"][:3]:
            stats = pred.get("historical_stats", {})
            if stats:
                print(f"\n   {pred['dog_name']}:")
                print(
                    f"   - Market confidence: {stats.get('market_confidence', 0):.3f}"
                )
                print(
                    f"   - Competition quality: {stats.get('competition_quality', 0)}"
                )
                print(f"   - Sectional analysis: {stats.get('sectional_analysis', [])}")
                print(f"   - Running style: {stats.get('running_style', 'Unknown')}")
                print(
                    f"   - Gender modifier: {stats.get('gender_performance_modifier', 1.0):.3f}"
                )
                print(f"   - Grade distribution: {stats.get('grade_distribution', {})}")
                print(f"   - Value indicator: {stats.get('value_indicator', 0):.3f}")
                print(f"   - Market trend: {stats.get('market_position_trend', 0):.3f}")

        return True
    else:
        print(f"❌ PIPELINE TEST FAILED")
        return False


if __name__ == "__main__":
    # Test with sample file
    test_file = "/Users/orlandolee/greyhound_racing_collector/upcoming_races/Race 1 - DAPT - 2025-07-24.csv"
    test_analysis_pipeline(test_file)
