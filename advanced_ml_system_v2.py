#!/usr/bin/env python3


# Wrapper class for unified predictor compatibility
class AdvancedMLSystemV2:
    """Advanced ML system v2 - wrapper for compatibility"""

    def __init__(self):
        pass

    def train_model(self, data):
        # Placeholder implementation
        print("Training with data")


"""
Comprehensive Enhanced ML Model System with Traditional Analysis Integration
=========================================================================

This system integrates:
1. Feature importance insights from analysis
2. Rich form guide CSV historical data 
3. Database race results
4. Traditional race analysis for feature enrichment
5. New predictive features based on insights
6. Ensemble models (Voting, Stacking)
7. Continuous improvement pipeline

Features:
- Comprehensive historical data from CSV form guides
- Advanced feature engineering from race history
- Traditional handicapping analysis integration
- Dynamic feature selection based on importance analysis
- Multiple ensemble methods
- Model performance optimization
- Continuous learning pipeline

Author: AI Assistant
Date: July 25, 2025
"""

import json
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")

# Import traditional analysis system
try:
    from traditional_analysis import (TraditionalRaceAnalyzer,
                                      get_traditional_ml_features)

    TRADITIONAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Traditional analysis system not available: {e}")
    TRADITIONAL_ANALYSIS_AVAILABLE = False

# Import enhanced data integration
try:
    from enhanced_data_integration import EnhancedDataIntegrator

    ENHANCED_DATA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced data integration not available: {e}")
    ENHANCED_DATA_AVAILABLE = False

# Enhanced ML Libraries
try:
    import joblib
    from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier, VotingClassifier)
    from sklearn.feature_selection import RFE, SelectKBest, f_classif
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit,
                                         cross_val_score)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import (LabelEncoder, RobustScaler,
                                       StandardScaler)
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced ML libraries not available: {e}")
    SKLEARN_AVAILABLE = False


class ComprehensiveEnhancedMLSystem:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results_dir = Path("./comprehensive_model_results")
        self.models_dir = Path("./comprehensive_trained_models")
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        # Form guide directories - use both sources for comprehensive historical data
        self.form_guides_dir = Path("./form_guides")
        self.downloaded_forms_dir = self.form_guides_dir / "downloaded"
        self.unprocessed_forms_dir = Path(
            "./unprocessed"
        )  # Primary source with 1,228 files

        # Enhanced model configurations with class balancing
        self.base_models = {
            "balanced_random_forest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "min_samples_split": 3,
                    "class_weight": "balanced",
                    "random_state": 42,
                },
            },
            "calibrated_random_forest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "min_samples_split": 3,
                    "class_weight": {0: 0.6, 1: 1.4},
                    "random_state": 42,
                },
            },
            "natural_random_forest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "min_samples_split": 3,
                    "class_weight": None,
                    "random_state": 42,
                },
            },
            "balanced_gradient_boosting": {
                "model": GradientBoostingClassifier,
                "params": {
                    "n_estimators": 300,
                    "learning_rate": 0.1,
                    "max_depth": 8,
                    "random_state": 42,
                },
            },
            "balanced_extra_trees": {
                "model": ExtraTreesClassifier,
                "params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "min_samples_split": 3,
                    "class_weight": "balanced",
                    "random_state": 42,
                },
            },
            "balanced_logistic": {
                "model": LogisticRegression,
                "params": {
                    "C": 2.0,
                    "max_iter": 3000,
                    "class_weight": "balanced",
                    "random_state": 42,
                },
            },
            "balanced_svm": {
                "model": SVC,
                "params": {
                    "C": 2.0,
                    "kernel": "rbf",
                    "probability": True,
                    "class_weight": "balanced",
                    "random_state": 42,
                },
            },
            "neural_network": {
                "model": MLPClassifier,
                "params": {
                    "hidden_layer_sizes": (150, 100, 50),
                    "max_iter": 2000,
                    "random_state": 42,
                },
            },
        }

        # Add imbalanced-learn models if available
        try:
            from imblearn.ensemble import (BalancedBaggingClassifier,
                                           BalancedRandomForestClassifier)

            self.imbalanced_models = {
                "balanced_rf_classifier": {
                    "model": BalancedRandomForestClassifier,
                    "params": {
                        "n_estimators": 300,
                        "max_depth": 20,
                        "sampling_strategy": "auto",
                        "random_state": 42,
                    },
                },
                "balanced_bagging": {
                    "model": BalancedBaggingClassifier,
                    "params": {
                        "n_estimators": 100,
                        "sampling_strategy": "auto",
                        "random_state": 42,
                    },
                },
            }
        except ImportError:
            self.imbalanced_models = {}

        # Feature importance insights (from previous analysis)
        self.high_impact_features = ["recent_form_avg", "avg_position"]

        self.stable_features = [
            "box_versatility",
            "current_weight",
            "position_consistency",
            "distance_experience",
            "track_condition_encoded",
            "form_trend",
            "avg_performance_rating",
            "distance_numeric",
        ]

        print("🚀 Comprehensive Enhanced ML Model System Initialized")

    def load_form_guide_data(self):
        """Load and parse all form guide CSV files with proper blank row handling"""
        try:
            print("📊 Loading form guide data from CSV files...")

            form_data = {}

            # Load from both sources for comprehensive historical data
            csv_files = []

            # Primary source: unprocessed directory (1,228 files with rich historical data)
            if self.unprocessed_forms_dir.exists():
                unprocessed_files = list(self.unprocessed_forms_dir.glob("*.csv"))
                csv_files.extend(unprocessed_files)
                print(
                    f"📁 Found {len(unprocessed_files)} files in unprocessed directory"
                )

            # Secondary source: downloaded directory (backup/additional data)
            if self.downloaded_forms_dir.exists():
                downloaded_files = list(self.downloaded_forms_dir.glob("*.csv"))
                csv_files.extend(downloaded_files)
                print(f"📁 Found {len(downloaded_files)} files in downloaded directory")

            print(f"📁 Total: {len(csv_files)} form guide files to process")

            processed_files = 0
            skipped_files = 0

            for csv_file in csv_files:
                try:
                    # First, check if file is actually a CSV (not HTML)
                    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                        first_line = f.readline().strip()
                        if "DOCTYPE html" in first_line or "<html>" in first_line:
                            print(f"⚠️ Skipping HTML file: {csv_file.name}")
                            skipped_files += 1
                            continue
                        if not first_line.startswith("Dog Name"):
                            print(
                                f"⚠️ Skipping non-standard CSV: {csv_file.name} (starts with: {first_line[:50]}...)"
                            )
                            skipped_files += 1
                            continue

                    # Parse race info from filename
                    filename = csv_file.stem
                    parts = filename.split(" - ")
                    if len(parts) >= 3:
                        race_num = parts[0].replace("Race ", "")
                        venue = parts[1]
                        date_str = parts[2]

                        # Read the CSV with error handling
                        df = pd.read_csv(
                            csv_file,
                            on_bad_lines="skip",  # Skip problematic lines
                            encoding="utf-8",
                        )

                        processed_files += 1

                        # Process dogs properly handling blank rows
                        current_dog_name = None

                        for idx, row in df.iterrows():
                            dog_name_raw = str(row["Dog Name"]).strip()

                            # Check if this is a new dog or continuation of previous
                            if (
                                dog_name_raw != '""'
                                and dog_name_raw != ""
                                and dog_name_raw != "nan"
                            ):
                                # New dog - clean the name
                                current_dog_name = dog_name_raw
                                # Remove box number prefix (e.g., "1. Mel Monelli" -> "Mel Monelli")
                                if ". " in current_dog_name:
                                    current_dog_name = current_dog_name.split(". ", 1)[
                                        1
                                    ]

                                # Initialize dog record if not exists
                                if current_dog_name not in form_data:
                                    form_data[current_dog_name] = []

                            # Skip if we don't have a current dog (shouldn't happen with proper format)
                            if current_dog_name is None:
                                continue

                            # Parse this row as historical race data for current dog
                            historical_race = {
                                "sex": str(row.get("Sex", "")).strip(),
                                "place": str(row.get("PLC", "")).strip(),
                                "box": str(row.get("BOX", "")).strip(),
                                "weight": str(row.get("WGT", "")).strip(),
                                "distance": str(row.get("DIST", "")).strip(),
                                "date": str(row.get("DATE", "")).strip(),
                                "track": str(row.get("TRACK", "")).strip(),
                                "grade": str(row.get("G", "")).strip(),
                                "time": str(row.get("TIME", "")).strip(),
                                "win_time": str(row.get("WIN", "")).strip(),
                                "bonus": str(row.get("BON", "")).strip(),
                                "first_sectional": str(row.get("1 SEC", "")).strip(),
                                "margin": str(row.get("MGN", "")).strip(),
                                "runner_up": str(row.get("W/2G", "")).strip(),
                                "pir": str(row.get("PIR", "")).strip(),
                                "starting_price": str(row.get("SP", "")).strip(),
                                "source_race": f"{race_num}_{venue}_{date_str}",
                                "source_file": str(csv_file),
                            }

                            # Only add if we have meaningful data (at least place and date)
                            if historical_race["place"] and historical_race["date"]:
                                # Check for duplicates before adding
                                race_key = (
                                    historical_race["date"],
                                    historical_race["track"],
                                    historical_race["place"],
                                    historical_race["box"],
                                    historical_race["weight"],
                                    historical_race["distance"],
                                )

                                # Check if this exact race already exists for this dog
                                is_duplicate = False
                                for existing_race in form_data[current_dog_name]:
                                    existing_key = (
                                        existing_race["date"],
                                        existing_race["track"],
                                        existing_race["place"],
                                        existing_race["box"],
                                        existing_race["weight"],
                                        existing_race["distance"],
                                    )
                                    if race_key == existing_key:
                                        is_duplicate = True
                                        break

                                if not is_duplicate:
                                    form_data[current_dog_name].append(historical_race)

                except Exception as e:
                    print(f"⚠️ Error processing {csv_file}: {e}")
                    skipped_files += 1
                    continue

            print(
                f"✅ Processed {processed_files} files, skipped {skipped_files} problematic files"
            )
            print(f"✅ Loaded form data for {len(form_data)} dogs")

            # Print summary statistics
            total_races = sum(len(races) for races in form_data.values())
            avg_races = total_races / len(form_data) if form_data else 0
            print(f"📈 Total historical races: {total_races}")
            print(f"📊 Average races per dog: {avg_races:.1f}")

            return form_data

        except Exception as e:
            print(f"❌ Error loading form guide data: {e}")
            return {}

    def parse_historical_race_data(self, race_data_str):
        """Parse individual historical race data string"""
        try:
            # Split by comma to get race details
            parts = race_data_str.split(",")

            race_info = {}

            # Basic parsing - this would need to be refined based on exact format
            if len(parts) >= 10:
                try:
                    race_info["position"] = (
                        int(parts[2]) if parts[2].isdigit() else None
                    )
                    race_info["box"] = int(parts[3]) if parts[3].isdigit() else None
                    race_info["weight"] = (
                        float(parts[4]) if parts[4].replace(".", "").isdigit() else None
                    )
                    race_info["distance"] = (
                        int(parts[5]) if parts[5].isdigit() else None
                    )
                    race_info["date"] = parts[6] if len(parts[6]) > 5 else None
                    race_info["track"] = parts[7] if parts[7] else None
                    race_info["grade"] = parts[8] if parts[8] else None
                    race_info["time"] = (
                        float(parts[9]) if parts[9].replace(".", "").isdigit() else None
                    )
                    race_info["starting_price"] = (
                        float(parts[-1])
                        if parts[-1].replace(".", "").isdigit()
                        else None
                    )
                except:
                    pass

            return race_info

        except Exception as e:
            return {}

    def load_race_results_data(self):
        """Load actual race results from database"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT 
                drd.race_id, drd.dog_clean_name, drd.finish_position,
                drd.box_number, drd.weight, drd.starting_price,
                drd.performance_rating, drd.speed_rating, drd.class_rating,
                drd.individual_time, drd.margin,
                rm.field_size, rm.distance, rm.venue, rm.track_condition,
                rm.grade, rm.race_date, rm.race_time,
                rm.weather_condition, rm.temperature, rm.humidity, 
                rm.wind_speed, rm.wind_direction, rm.pressure,
                rm.weather_adjustment_factor
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.finish_position != 'N/A'
            ORDER BY rm.race_date ASC
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            print(f"📊 Loaded {len(df)} race results from database")
            return df

        except Exception as e:
            print(f"❌ Error loading race results: {e}")
            return None

    def create_comprehensive_features(self, race_results_df, form_data):
        """Create comprehensive features using both database and form guide data"""
        try:
            print("🔧 Creating comprehensive predictive features...")

            enhanced_records = []
            processed_count = 0

            for _, row in race_results_df.iterrows():
                try:
                    # Clean finish position
                    pos_str = str(row["finish_position"]).strip()
                    if pos_str in ["", "N/A", "None", "nan"]:
                        continue
                    pos_cleaned = "".join(filter(str.isdigit, pos_str))
                    if not pos_cleaned:
                        continue
                    finish_position = int(pos_cleaned)
                    if finish_position < 1 or finish_position > 10:
                        continue
                except (ValueError, TypeError):
                    continue

                dog_name = row["dog_clean_name"]
                race_date = row["race_date"]

                # Get form guide historical data for this dog
                historical_form_data = form_data.get(dog_name, [])

                # Get database historical data
                db_historical_data = race_results_df[
                    (race_results_df["dog_clean_name"] == dog_name)
                    & (race_results_df["race_date"] < race_date)
                ].sort_values("race_date", ascending=False)

                # Combine both data sources for comprehensive analysis
                # Reduce requirements - accept dogs with any historical data
                if len(db_historical_data) >= 1 or len(historical_form_data) >= 1:

                    # DATABASE-BASED FEATURES
                    db_positions = []
                    db_times = []
                    db_weights = []
                    db_odds = []

                    for _, hist_row in db_historical_data.iterrows():
                        # Position data
                        if pd.notna(hist_row["finish_position"]):
                            pos_str = str(hist_row["finish_position"]).strip()
                            if pos_str not in ["", "N/A", "None", "nan"]:
                                pos_cleaned = "".join(filter(str.isdigit, pos_str))
                                if pos_cleaned and 1 <= int(pos_cleaned) <= 10:
                                    db_positions.append(int(pos_cleaned))

                        # Time data
                        try:
                            if (
                                pd.notna(hist_row["individual_time"])
                                and float(hist_row["individual_time"]) > 0
                            ):
                                db_times.append(float(hist_row["individual_time"]))
                        except (ValueError, TypeError):
                            pass

                        # Weight data
                        try:
                            if (
                                pd.notna(hist_row["weight"])
                                and float(hist_row["weight"]) > 0
                            ):
                                db_weights.append(float(hist_row["weight"]))
                        except (ValueError, TypeError):
                            pass

                        # Odds data
                        try:
                            if (
                                pd.notna(hist_row["starting_price"])
                                and float(hist_row["starting_price"]) > 0
                            ):
                                db_odds.append(float(hist_row["starting_price"]))
                        except (ValueError, TypeError):
                            pass

                    # FORM GUIDE-BASED FEATURES (Enhanced with properly parsed data)
                    form_positions = []
                    form_times = []
                    form_weights = []
                    form_odds = []
                    form_distances = []
                    form_tracks = []

                    # Use properly parsed form guide data
                    for form_entry in historical_form_data[
                        :20
                    ]:  # Use up to 20 historical races
                        try:
                            # Parse position
                            place_str = form_entry.get("place", "").strip()
                            if place_str and place_str.isdigit():
                                position = int(place_str)
                                if 1 <= position <= 10:
                                    form_positions.append(position)

                            # Parse time
                            time_str = form_entry.get("time", "").strip()
                            if time_str:
                                try:
                                    time_val = float(time_str)
                                    if (
                                        15.0 <= time_val <= 60.0
                                    ):  # Reasonable time range
                                        form_times.append(time_val)
                                except (ValueError, TypeError):
                                    pass

                            # Parse weight
                            weight_str = form_entry.get("weight", "").strip()
                            if weight_str:
                                try:
                                    weight_val = float(weight_str)
                                    if (
                                        20.0 <= weight_val <= 40.0
                                    ):  # Reasonable weight range
                                        form_weights.append(weight_val)
                                except (ValueError, TypeError):
                                    pass

                            # Parse starting price (odds)
                            sp_str = form_entry.get("starting_price", "").strip()
                            if sp_str:
                                try:
                                    sp_val = float(sp_str)
                                    if 1.0 <= sp_val <= 1000.0:  # Reasonable odds range
                                        form_odds.append(sp_val)
                                except (ValueError, TypeError):
                                    pass

                            # Parse distance
                            dist_str = form_entry.get("distance", "").strip()
                            if dist_str:
                                try:
                                    dist_val = int(dist_str)
                                    if (
                                        300 <= dist_val <= 800
                                    ):  # Reasonable distance range
                                        form_distances.append(dist_val)
                                except (ValueError, TypeError):
                                    pass

                            # Parse track
                            track_str = form_entry.get("track", "").strip()
                            if track_str and track_str != "":
                                form_tracks.append(track_str)

                        except Exception as e:
                            continue

                    # Combine all position data
                    all_positions = db_positions + form_positions
                    all_times = db_times + form_times
                    all_weights = db_weights + form_weights
                    all_odds = db_odds + form_odds

                    if not all_positions:
                        continue

                    # CORE STABLE FEATURES (Priority 1 - from analysis)
                    avg_position = np.mean(all_positions)
                    recent_form_avg = (
                        np.mean(all_positions[:8])
                        if len(all_positions) >= 8
                        else np.mean(all_positions)
                    )
                    win_rate = sum(1 for p in all_positions if p == 1) / len(
                        all_positions
                    )
                    place_rate = sum(1 for p in all_positions if p <= 3) / len(
                        all_positions
                    )

                    # Time consistency (stable feature)
                    time_consistency = (
                        1 / (np.std(all_times) + 0.1) if len(all_times) > 1 else 0.5
                    )

                    # Market confidence (high impact)
                    market_confidence = 1 / (np.mean(all_odds) + 1) if all_odds else 0.1
                    try:
                        current_odds_log = (
                            np.log(float(row["starting_price"]) + 1)
                            if pd.notna(row["starting_price"])
                            else np.log(10)
                        )
                    except (ValueError, TypeError):
                        current_odds_log = np.log(10)

                    # ENHANCED FEATURES USING COMPREHENSIVE DATA

                    # Long-term form analysis (using extended history)
                    if len(all_positions) >= 10:
                        recent_10 = np.mean(all_positions[:10])
                        older_10 = (
                            np.mean(all_positions[10:20])
                            if len(all_positions) >= 20
                            else np.mean(all_positions[10:])
                        )
                        long_term_form_trend = (
                            (older_10 - recent_10) / (older_10 + 1)
                            if older_10 > 0
                            else 0
                        )
                    else:
                        long_term_form_trend = 0

                    # Performance consistency over extended period
                    position_consistency = 1 / (np.std(all_positions) + 0.1)

                    # Speed analysis
                    if all_times:
                        avg_time = np.mean(all_times)
                        best_time = min(all_times)
                        time_improvement_trend = (
                            -np.polyfit(range(len(all_times)), all_times, 1)[0]
                            if len(all_times) > 2
                            else 0
                        )
                    else:
                        avg_time = 30.0
                        best_time = 28.0
                        time_improvement_trend = 0

                    # Weight analysis
                    if all_weights:
                        avg_weight = np.mean(all_weights)
                        weight_consistency = 1 / (np.std(all_weights) + 0.1)
                        try:
                            current_weight = (
                                float(row["weight"])
                                if pd.notna(row["weight"])
                                else avg_weight
                            )
                            weight_vs_avg = current_weight - avg_weight
                        except (ValueError, TypeError):
                            current_weight = avg_weight
                            weight_vs_avg = 0
                    else:
                        avg_weight = 30.0
                        weight_consistency = 0.5
                        try:
                            current_weight = (
                                float(row["weight"])
                                if pd.notna(row["weight"])
                                else 30.0
                            )
                        except (ValueError, TypeError):
                            current_weight = 30.0
                        weight_vs_avg = 0

                    # Distance specialization
                    try:
                        current_distance = (
                            float(str(row["distance"]).replace("m", ""))
                            if pd.notna(row["distance"])
                            else 500.0
                        )
                    except (ValueError, TypeError):
                        current_distance = 500.0
                    if form_distances:
                        distance_experience_count = sum(
                            1 for d in form_distances if abs(d - current_distance) <= 50
                        )
                        distance_specialization = (
                            distance_experience_count / len(form_distances)
                            if form_distances
                            else 0
                        )
                    else:
                        distance_specialization = 0.1

                    # Track/Venue experience
                    current_venue = row["venue"]
                    venue_experience = len(
                        [
                            r
                            for _, r in db_historical_data.iterrows()
                            if r["venue"] == current_venue
                        ]
                    )
                    venue_experience += sum(
                        1
                        for track in form_tracks
                        if track and current_venue in str(track)
                    )

                    # Grade/Class analysis
                    current_grade = row["grade"]
                    grade_experience = len(
                        [
                            r
                            for _, r in db_historical_data.iterrows()
                            if r["grade"] == current_grade
                        ]
                    )

                    # Recent activity pattern
                    try:
                        if len(db_historical_data) > 0:
                            last_race_date = datetime.fromisoformat(
                                db_historical_data.iloc[0]["race_date"]
                            )
                            current_race_date = datetime.fromisoformat(race_date)
                            days_since_last = (current_race_date - last_race_date).days
                        else:
                            days_since_last = 14
                    except:
                        days_since_last = 14

                    # Fitness/form indicator
                    fitness_score = (
                        (1 / (1 + days_since_last / 30))
                        * (1 + place_rate)
                        * (1 + win_rate)
                    )

                    # Competition strength
                    try:
                        field_size = (
                            int(row["field_size"]) if pd.notna(row["field_size"]) else 6
                        )
                    except (ValueError, TypeError):
                        field_size = 6
                    competition_strength = field_size / (venue_experience + 1)

                    # Box draw performance
                    try:
                        current_box = (
                            int(row["box_number"]) if pd.notna(row["box_number"]) else 4
                        )
                    except (ValueError, TypeError):
                        current_box = 4
                    box_win_rate = 0
                    if len(db_historical_data) > 0:
                        same_box_races = db_historical_data[
                            db_historical_data["box_number"] == current_box
                        ]
                        if len(same_box_races) > 0:
                            same_box_positions = [
                                int("".join(filter(str.isdigit, str(p))))
                                for p in same_box_races["finish_position"]
                                if pd.notna(p) and "".join(filter(str.isdigit, str(p)))
                            ]
                            box_win_rate = (
                                sum(1 for p in same_box_positions if p == 1)
                                / len(same_box_positions)
                                if same_box_positions
                                else 0
                            )

                    # TRADITIONAL ANALYSIS FEATURES (Priority 1 - Expert Handicapping)
                    traditional_features = {}
                    if TRADITIONAL_ANALYSIS_AVAILABLE:
                        try:
                            # Create race context for traditional analysis
                            race_context = {
                                "venue": current_venue,
                                "distance": f"{current_distance}m",
                                "grade": current_grade,
                                "track_condition": row.get("track_condition", "Good"),
                            }

                            # Get traditional analysis features for this dog
                            traditional_features = get_traditional_ml_features(
                                dog_name, race_context, self.db_path
                            )

                            if processed_count % 50 == 0:  # Log occasionally
                                print(
                                    f"     Traditional analysis for {dog_name}: {traditional_features.get('traditional_overall_score', 0):.3f}"
                                )

                        except Exception as e:
                            # Don't let traditional analysis errors stop the training
                            if processed_count % 100 == 0:  # Log errors occasionally
                                print(
                                    f"     ⚠️ Traditional analysis failed for {dog_name}: {e}"
                                )
                            traditional_features = {}

                    # WEATHER FEATURES (Priority 2 - Environmental Impact)
                    weather_condition = (
                        str(row.get("weather_condition", "")).strip().lower()
                    )
                    temperature = (
                        float(row.get("temperature", 15.0))
                        if pd.notna(row.get("temperature"))
                        else 15.0
                    )
                    humidity = (
                        float(row.get("humidity", 60.0))
                        if pd.notna(row.get("humidity"))
                        else 60.0
                    )
                    wind_speed = (
                        float(row.get("wind_speed", 10.0))
                        if pd.notna(row.get("wind_speed"))
                        else 10.0
                    )
                    pressure = (
                        float(row.get("pressure", 1013.0))
                        if pd.notna(row.get("pressure"))
                        else 1013.0
                    )
                    weather_adjustment_factor = (
                        float(row.get("weather_adjustment_factor", 1.0))
                        if pd.notna(row.get("weather_adjustment_factor"))
                        else 1.0
                    )

                    # Weather condition encoding
                    weather_clear = (
                        1
                        if "clear" in weather_condition or "sunny" in weather_condition
                        else 0
                    )
                    weather_cloudy = (
                        1
                        if "cloud" in weather_condition
                        or "overcast" in weather_condition
                        else 0
                    )
                    weather_rain = (
                        1
                        if "rain" in weather_condition or "shower" in weather_condition
                        else 0
                    )
                    weather_fog = (
                        1
                        if "fog" in weather_condition or "mist" in weather_condition
                        else 0
                    )

                    # Temperature categories (optimal range 15-25°C)
                    temp_cold = 1 if temperature < 10 else 0
                    temp_cool = 1 if 10 <= temperature < 15 else 0
                    temp_optimal = 1 if 15 <= temperature <= 25 else 0
                    temp_warm = 1 if 25 < temperature <= 30 else 0
                    temp_hot = 1 if temperature > 30 else 0

                    # Wind impact categories
                    wind_calm = 1 if wind_speed < 5 else 0
                    wind_light = 1 if 5 <= wind_speed < 15 else 0
                    wind_moderate = 1 if 15 <= wind_speed < 25 else 0
                    wind_strong = 1 if wind_speed >= 25 else 0

                    # Humidity categories
                    humidity_low = 1 if humidity < 40 else 0
                    humidity_normal = 1 if 40 <= humidity <= 70 else 0
                    humidity_high = 1 if humidity > 70 else 0

                    # Historical weather performance for this dog
                    weather_experience_count = 0
                    weather_performance = 0
                    if len(db_historical_data) > 0:
                        similar_weather_races = db_historical_data[
                            (
                                db_historical_data["weather_condition"]
                                .str.lower()
                                .str.contains(weather_condition, na=False)
                            )
                            | (
                                db_historical_data["weather_condition"].isnull()
                                & (weather_condition == "")
                            )
                        ]
                        if len(similar_weather_races) > 0:
                            weather_experience_count = len(similar_weather_races)
                            weather_positions = [
                                int("".join(filter(str.isdigit, str(p))))
                                for p in similar_weather_races["finish_position"]
                                if pd.notna(p) and "".join(filter(str.isdigit, str(p)))
                            ]
                            if weather_positions:
                                weather_performance = np.mean(weather_positions)

                    # Create comprehensive record
                    enhanced_records.append(
                        {
                            "race_id": row["race_id"],
                            "dog_name": dog_name,
                            "finish_position": finish_position,
                            "is_winner": 1 if finish_position == 1 else 0,
                            "is_placer": 1 if finish_position <= 3 else 0,
                            "race_date": race_date,
                            # STABLE HIGH-IMPACT FEATURES (Priority 1)
                            "avg_position": avg_position,
                            "recent_form_avg": recent_form_avg,
                            "market_confidence": market_confidence,
                            "current_odds_log": current_odds_log,
                            "venue_experience": venue_experience,
                            "place_rate": place_rate,
                            "current_weight": current_weight,
                            "time_consistency": time_consistency,
                            # ENHANCED COMPREHENSIVE FEATURES (Priority 2)
                            "win_rate": win_rate,
                            "long_term_form_trend": long_term_form_trend,
                            "position_consistency": position_consistency,
                            "avg_time": avg_time,
                            "best_time": best_time,
                            "time_improvement_trend": time_improvement_trend,
                            "avg_weight": avg_weight,
                            "weight_consistency": weight_consistency,
                            "weight_vs_avg": weight_vs_avg,
                            "distance_specialization": distance_specialization,
                            "grade_experience": grade_experience,
                            "days_since_last": days_since_last,
                            "fitness_score": fitness_score,
                            "competition_strength": competition_strength,
                            "box_win_rate": box_win_rate,
                            # WEATHER FEATURES (Priority 2 - Environmental Impact)
                            "temperature": temperature,
                            "humidity": humidity,
                            "wind_speed": wind_speed,
                            "pressure": pressure,
                            "weather_adjustment_factor": weather_adjustment_factor,
                            "weather_clear": weather_clear,
                            "weather_cloudy": weather_cloudy,
                            "weather_rain": weather_rain,
                            "weather_fog": weather_fog,
                            "temp_cold": temp_cold,
                            "temp_cool": temp_cool,
                            "temp_optimal": temp_optimal,
                            "temp_warm": temp_warm,
                            "temp_hot": temp_hot,
                            "wind_calm": wind_calm,
                            "wind_light": wind_light,
                            "wind_moderate": wind_moderate,
                            "wind_strong": wind_strong,
                            "humidity_low": humidity_low,
                            "humidity_normal": humidity_normal,
                            "humidity_high": humidity_high,
                            "weather_experience_count": weather_experience_count,
                            "weather_performance": weather_performance,
                            # CONTEXT FEATURES (Priority 3)
                            "current_box": current_box,
                            "field_size": field_size,
                            "venue": row["venue"],
                            "track_condition": row["track_condition"],
                            "grade": row["grade"],
                            "distance": row["distance"],
                            "historical_races_count": len(all_positions),
                            # TRADITIONAL ANALYSIS FEATURES (Priority 1 - Expert Handicapping)
                            **traditional_features,  # Merge all traditional analysis features
                        }
                    )

                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"   Processed {processed_count} records...")

            print(
                f"✅ Created comprehensive features for {len(enhanced_records)} records"
            )
            return pd.DataFrame(enhanced_records)

        except Exception as e:
            print(f"❌ Error creating comprehensive features: {e}")
            return None

    def prepare_comprehensive_features(self, df):
        """Prepare features with intelligent selection based on importance analysis"""
        try:
            if len(df) < 30:
                print(
                    f"❌ Insufficient data for comprehensive training (need at least 30, got {len(df)})"
                )
                return None, None
            elif len(df) < 100:
                print(
                    f"⚠️ Limited training data ({len(df)} samples) - results may be less reliable"
                )

            print(f"📊 Preparing comprehensive feature set from {len(df)} records...")

            # Priority-based feature selection (based on importance analysis)
            priority_1_features = [  # Stable, high-impact features
                "avg_position",
                "recent_form_avg",
                "market_confidence",
                "current_odds_log",
                "venue_experience",
                "place_rate",
                "current_weight",
                "time_consistency",
                # Traditional analysis core features (Priority 1 - Expert Knowledge)
                "traditional_overall_score",
                "traditional_performance_score",
                "traditional_form_score",
                "traditional_consistency_score",
                "traditional_confidence_level",
            ]

            priority_2_features = [  # Enhanced comprehensive features
                "win_rate",
                "long_term_form_trend",
                "position_consistency",
                "avg_time",
                "best_time",
                "time_improvement_trend",
                "avg_weight",
                "weight_consistency",
                "weight_vs_avg",
                "distance_specialization",
                "grade_experience",
                "fitness_score",
                # Traditional analysis extended features
                "traditional_class_score",
                "traditional_fitness_score",
                "traditional_experience_score",
                "traditional_trainer_score",
                "traditional_track_condition_score",
                "traditional_distance_score",
                "traditional_key_factors_count",
                "traditional_risk_factors_count",
                # Weather features
                "temperature",
                "humidity",
                "wind_speed",
                "pressure",
                "weather_adjustment_factor",
                "weather_clear",
                "weather_cloudy",
                "weather_rain",
                "weather_fog",
                "temp_cold",
                "temp_cool",
                "temp_optimal",
                "temp_warm",
                "temp_hot",
                "wind_calm",
                "wind_light",
                "wind_moderate",
                "wind_strong",
                "humidity_low",
                "humidity_normal",
                "humidity_high",
                "weather_experience_count",
                "weather_performance",
            ]

            priority_3_features = [  # Context and situational features
                "days_since_last",
                "competition_strength",
                "box_win_rate",
                "current_box",
                "field_size",
                "historical_races_count",
            ]

            # Encode categorical features
            le_venue = LabelEncoder()
            le_condition = LabelEncoder()
            le_grade = LabelEncoder()

            df["venue_encoded"] = le_venue.fit_transform(df["venue"].fillna("Unknown"))
            df["track_condition_encoded"] = le_condition.fit_transform(
                df["track_condition"].fillna("Good")
            )
            df["grade_encoded"] = le_grade.fit_transform(df["grade"].fillna("Unknown"))

            # Distance numeric
            def safe_distance_convert(x):
                try:
                    if pd.notna(x):
                        x_str = str(x)
                        if "b'" in x_str or "\\x" in x_str:  # Binary data detected
                            return 500.0
                        return float(x_str.replace("m", ""))
                    return 500.0
                except (ValueError, TypeError):
                    return 500.0

            df["distance_numeric"] = df["distance"].apply(safe_distance_convert)

            # Combine all features
            feature_columns = (
                priority_1_features
                + priority_2_features
                + priority_3_features
                + [
                    "venue_encoded",
                    "track_condition_encoded",
                    "grade_encoded",
                    "distance_numeric",
                ]
            )

            # Ensure all features exist
            available_features = [f for f in feature_columns if f in df.columns]

            print(f"📊 Using {len(available_features)} comprehensive features")

            # Create final dataset
            target_columns = [
                "is_winner",
                "is_placer",
                "race_date",
                "race_id",
                "dog_name",
            ]
            complete_df = df[available_features + target_columns].copy()

            # Advanced imputation using KNN for better handling of missing values
            imputer = KNNImputer(n_neighbors=5)
            complete_df[available_features] = imputer.fit_transform(
                complete_df[available_features]
            )

            return complete_df, available_features

        except Exception as e:
            print(f"❌ Error preparing comprehensive features: {e}")
            return None, None

    def train_models(self):
        """Train and validate ML models with comprehensive feature engineering"""
        print("\n🤖 COMPREHENSIVE ML MODEL TRAINING")
        print("═" * 70)
        print(f"   🚀 Training Mode: Enhanced Feature Engineering")
        print(f"   🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 70)

        overall_start_time = time.time()

        # Step 1: Load comprehensive data
        print("\n📊 STEP 1: Loading Comprehensive Data")
        print("-" * 50)
        step_start = time.time()
        df = self.load_comprehensive_data()
        if df is None or len(df) < 100:
            print("❌ Insufficient training data")
            return
        step_time = time.time() - step_start
        print(f"   ✅ Step 1 completed in {step_time:.1f}s")

        # Step 2: Enhanced feature engineering
        print("\n🔧 STEP 2: Advanced Feature Engineering")
        print("-" * 50)
        step_start = time.time()
        enhanced_df = self.create_advanced_features(df)
        step_time = time.time() - step_start
        print(f"   📊 Created {len(enhanced_df):,} enhanced samples")
        print(f"   ✅ Step 2 completed in {step_time:.1f}s")

        # Step 3: Feature preparation and splitting
        print("\n⚙️  STEP 3: Feature Preparation & Data Splitting")
        print("-" * 50)
        step_start = time.time()
        X, y_win, y_place, feature_names = self.prepare_features(enhanced_df)

        # Time-based split
        split_date = enhanced_df["race_date"].quantile(0.8)
        train_mask = enhanced_df["race_date"] <= split_date

        X_train, X_test = X[train_mask], X[~train_mask]
        y_win_train, y_win_test = y_win[train_mask], y_win[~train_mask]
        y_place_train, y_place_test = y_place[train_mask], y_place[~train_mask]

        step_time = time.time() - step_start
        print(f"   📈 Training samples: {len(X_train):,}")
        print(f"   🧪 Test samples: {len(X_test):,}")
        print(f"   🔢 Features: {len(feature_names)}")
        print(f"   📅 Split date: {split_date}")
        print(f"   ✅ Step 3 completed in {step_time:.1f}s")

        # Step 4: Train ensemble models
        print("\n🎯 STEP 4: Ensemble Model Training")
        print("-" * 50)
        step_start = time.time()

        # Win prediction ensemble
        print("   🏆 Training WIN prediction ensemble...")
        win_models = self.train_ensemble(X_train, y_win_train, "win")
        win_accuracy = self.evaluate_ensemble(win_models, X_test, y_win_test, "win")

        # Place prediction ensemble
        print("   🥉 Training PLACE prediction ensemble...")
        place_models = self.train_ensemble(X_train, y_place_train, "place")
        place_accuracy = self.evaluate_ensemble(
            place_models, X_test, y_place_test, "place"
        )

        step_time = time.time() - step_start
        print(f"   ✅ Step 4 completed in {step_time:.1f}s")

        # Step 5: Save models and results
        print("\n💾 STEP 5: Saving Models & Results")
        print("-" * 50)
        step_start = time.time()

        # Save models and feature importance
        self.save_models(
            {
                "win_models": win_models,
                "place_models": place_models,
                "feature_names": feature_names,
                "scaler": self.scaler,
                "training_stats": {
                    "win_accuracy": win_accuracy,
                    "place_accuracy": place_accuracy,
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features_count": len(feature_names),
                },
            }
        )

        step_time = time.time() - step_start
        print(f"   ✅ Step 5 completed in {step_time:.1f}s")

        # Final summary
        total_time = time.time() - overall_start_time
        print(f"\n🎉 TRAINING COMPLETE!")
        print("═" * 70)

        # Performance summary with color coding
        win_status = (
            "🟢 EXCELLENT"
            if win_accuracy > 0.65
            else "🟡 GOOD" if win_accuracy > 0.55 else "🔴 NEEDS WORK"
        )
        place_status = (
            "🟢 EXCELLENT"
            if place_accuracy > 0.70
            else "🟡 GOOD" if place_accuracy > 0.60 else "🔴 NEEDS WORK"
        )

        print(f"   🏆 Win Accuracy: {win_accuracy:.3f} | {win_status}")
        print(f"   🥉 Place Accuracy: {place_accuracy:.3f} | {place_status}")
        print(f"   📊 Training Samples: {len(X_train):,}")
        print(f"   🧪 Test Samples: {len(X_test):,}")
        print(f"   🔢 Features Used: {len(feature_names)}")
        print(f"   ⏱️  Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"   🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 70)

    def run_comprehensive_analysis(self):
        """Run comprehensive enhanced model analysis with form guide integration"""
        print("🚀 Starting Comprehensive Enhanced ML Model Analysis")
        print("=" * 70)

        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn not available")
            return None

        try:
            # Load form guide data
            form_data = self.load_form_guide_data()

            # Load race results
            race_results_df = self.load_race_results_data()
            if race_results_df is None or len(race_results_df) < 100:
                print("❌ Insufficient race results data")
                return None

            # Create comprehensive features
            enhanced_df = self.create_comprehensive_features(race_results_df, form_data)
            if enhanced_df is None:
                print("❌ Comprehensive feature creation failed")
                return None

            # Prepare features
            print("📊 Preparing comprehensive feature set...")
            prepared_df, feature_columns = self.prepare_comprehensive_features(
                enhanced_df
            )
            if prepared_df is None:
                print("❌ Comprehensive feature preparation failed")
                return None

            print(
                f"📊 Comprehensive dataset: {len(prepared_df)} samples, {len(feature_columns)} features"
            )

            # Time-based split
            df_sorted = prepared_df.sort_values("race_date")
            split_point = int(0.8 * len(df_sorted))
            train_df = df_sorted.iloc[:split_point]
            test_df = df_sorted.iloc[split_point:]

            print(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")

            # Prepare features and targets
            X_train = train_df[feature_columns]
            y_train = train_df["is_winner"]
            X_test = test_df[feature_columns]
            y_test = test_df["is_winner"]

            # Feature scaling with robust scaler
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            results = {
                "timestamp": datetime.now().isoformat(),
                "data_summary": {
                    "total_samples": len(prepared_df),
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    "features": len(feature_columns),
                    "feature_list": feature_columns,
                    "form_data_dogs": len(form_data),
                    "avg_historical_races": (
                        np.mean([len(races) for races in form_data.values()])
                        if form_data
                        else 0
                    ),
                },
                "model_results": {},
                "ensemble_results": {},
                "optimization_results": {},
            }

            print("\\n🤖 Testing Enhanced Base Models:")
            print("-" * 60)

            # Test enhanced base models
            for model_name, config in self.base_models.items():
                print(f"   Testing {model_name}...")

                model_class = config["model"]
                model = model_class(**config["params"])

                # Train and evaluate
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = y_pred

                accuracy = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5

                # Feature importance
                feature_importance = None
                if hasattr(model, "feature_importances_"):
                    feature_importance = list(
                        zip(feature_columns, model.feature_importances_)
                    )
                    feature_importance.sort(key=lambda x: x[1], reverse=True)

                results["model_results"][model_name] = {
                    "accuracy": accuracy,
                    "auc": auc,
                    "feature_importance": (
                        feature_importance[:20] if feature_importance else None
                    ),
                }

                print(f"     📊 Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

            print("\\n🎯 Testing Advanced Ensemble Models:")
            print("-" * 60)

            # Create and test advanced ensemble models
            ensemble_models = self.create_advanced_ensemble_models(
                X_train_scaled, y_train
            )

            for ensemble_name, ensemble_model in ensemble_models.items():
                print(f"   Testing {ensemble_name}...")

                try:
                    ensemble_model.fit(X_train_scaled, y_train)
                    y_pred = ensemble_model.predict(X_test_scaled)

                    if hasattr(ensemble_model, "predict_proba"):
                        y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_pred_proba = y_pred

                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        auc = 0.5

                    results["ensemble_results"][ensemble_name] = {
                        "accuracy": accuracy,
                        "auc": auc,
                    }

                    print(f"     📊 Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

                except Exception as e:
                    print(f"     ❌ Failed: {e}")

            # Find best performing model overall
            all_results = {**results["model_results"], **results["ensemble_results"]}
            best_model_name = max(
                all_results.keys(), key=lambda k: all_results[k]["accuracy"]
            )
            best_accuracy = all_results[best_model_name]["accuracy"]

            print(
                f"\\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.3f})"
            )

            # Save best model
            if best_model_name in self.base_models:
                best_model_config = self.base_models[best_model_name]
                best_model = best_model_config["model"](**best_model_config["params"])
                best_model.fit(X_train_scaled, y_train)
            elif best_model_name in ensemble_models:
                best_model = ensemble_models[best_model_name]
                best_model.fit(X_train_scaled, y_train)
            else:
                # Fallback to the best performing base model
                best_model_config = self.base_models[best_model_name]
                best_model = best_model_config["model"](**best_model_config["params"])
                best_model.fit(X_train_scaled, y_train)

            # COMPREHENSIVE VALIDATION: Test Traditional Analysis + Combined Approach on Past Races
            print("\\n🔍 COMPREHENSIVE VALIDATION ON PAST RACES:")
            print("=" * 70)

            validation_results = self._run_comprehensive_validation(
                test_df, feature_columns, best_model, scaler
            )
            results["validation_results"] = validation_results

            # Register the best model with the model registry
            try:
                from model_registry import get_model_registry

                registry = get_model_registry()

                # Prepare performance metrics
                performance_metrics = {
                    "accuracy": best_accuracy,
                    "auc": all_results[best_model_name].get("auc", 0.5),
                    "f1_score": 0.0,  # Calculate if available
                    "precision": 0.0,  # Calculate if available
                    "recall": 0.0,  # Calculate if available
                }

                # Prepare training info
                training_info = {
                    "training_samples": results["data_summary"]["train_samples"],
                    "test_samples": results["data_summary"]["test_samples"],
                    "training_duration": 0.0,  # Add timing if available
                    "validation_method": "time_series_split",
                    "is_ensemble": "ensemble" in best_model_name.lower(),
                    "ensemble_components": (
                        [best_model_name]
                        if "ensemble" not in best_model_name.lower()
                        else ["RandomForest", "GradientBoosting", "LogisticRegression"]
                    ),
                    "data_quality_score": 0.8,  # Estimate based on comprehensive data
                    "inference_time_ms": 10.0,  # Estimate
                }

                # Register the model
                model_id = registry.register_model(
                    model_obj=best_model,
                    scaler_obj=scaler,
                    model_name="comprehensive_enhanced",
                    model_type=best_model_name,
                    performance_metrics=performance_metrics,
                    training_info=training_info,
                    feature_names=feature_columns,
                    hyperparameters={},
                    notes=f"Comprehensive enhanced ML model trained on {results['data_summary']['total_samples']} samples",
                )

                print(f"📝 Model registered with ID: {model_id}")

                # Also save the traditional format for backward compatibility
                model_file = (
                    self.models_dir
                    / f"comprehensive_best_model_{datetime.now().strftime('%Y%m%d')}.joblib"
                )
                joblib.dump(
                    {
                        "model": best_model,
                        "scaler": scaler,
                        "feature_columns": feature_columns,
                        "model_name": best_model_name,
                        "accuracy": best_accuracy,
                        "timestamp": datetime.now().isoformat(),
                        "data_summary": results["data_summary"],
                        "registry_model_id": model_id,
                    },
                    model_file,
                )

                print(f"💾 Best comprehensive model saved: {model_file}")

            except Exception as e:
                print(f"⚠️ Could not register model with registry: {e}")
                # Fallback to traditional saving
                model_file = (
                    self.models_dir
                    / f"comprehensive_best_model_{datetime.now().strftime('%Y%m%d')}.joblib"
                )
                joblib.dump(
                    {
                        "model": best_model,
                        "scaler": scaler,
                        "feature_columns": feature_columns,
                        "model_name": best_model_name,
                        "accuracy": best_accuracy,
                        "timestamp": datetime.now().isoformat(),
                        "data_summary": results["data_summary"],
                    },
                    model_file,
                )

                print(f"💾 Best comprehensive model saved: {model_file}")

            # Save results
            results_file = (
                self.results_dir
                / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"💾 Results saved: {results_file}")

            # AUTO-UPDATE: Run automated feature importance updater
            print("\n🔄 Running automated feature importance update...")
            try:
                from automated_feature_importance_updater import \
                    AutomatedFeatureImportanceUpdater

                updater = AutomatedFeatureImportanceUpdater()
                update_success = updater.run_automated_update()
                if update_success:
                    print(
                        "✅ Prediction system automatically updated with latest insights!"
                    )
                else:
                    print("⚠️ Automated update completed with warnings - check logs")
            except Exception as e:
                print(f"⚠️ Could not run automated update: {e}")

            print("\\n✅ Comprehensive Enhanced ML Analysis Complete!")

            return results

        except Exception as e:
            print(f"❌ Comprehensive analysis failed: {e}")
            return None

    def auto_optimize_model_parameters(self, prepared_df, feature_columns):
        """Automatically test different model configurations and select the best one based on historical validation"""
        try:
            print("🔬 AUTO-OPTIMIZATION: Testing multiple model configurations...")
            print("=" * 70)

            # Time-based split for validation
            df_sorted = prepared_df.sort_values("race_date")
            split_point = int(0.7 * len(df_sorted))  # 70% train, 30% validation
            train_df = df_sorted.iloc[:split_point]
            val_df = df_sorted.iloc[split_point:]

            X_train = train_df[feature_columns]
            y_train = train_df["is_winner"]
            X_val = val_df[feature_columns]
            y_val = val_df["is_winner"]

            # Feature scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            print(
                f"📊 Training: {len(train_df)} races, Validation: {len(val_df)} races"
            )
            print(f"📊 Win rate in validation: {y_val.mean():.1%}")

            # Test multiple class weight configurations
            class_weight_configs = {
                "natural": None,
                "balanced": "balanced",
                "light_balanced": {0: 0.8, 1: 1.2},
                "moderate_balanced": {0: 0.6, 1: 1.4},
                "custom_balanced": {0: 0.7, 1: 1.3},
                "heavy_balanced": {0: 0.4, 1: 1.6},
            }

            # Test different model types with various configurations
            model_configs = {}

            # Random Forest variations
            for weight_name, weight_config in class_weight_configs.items():
                model_configs[f"rf_{weight_name}"] = {
                    "model": RandomForestClassifier,
                    "params": {
                        "n_estimators": 300,
                        "max_depth": 20,
                        "min_samples_split": 3,
                        "class_weight": weight_config,
                        "random_state": 42,
                    },
                    "type": "RandomForest",
                    "weight_config": weight_name,
                }

            # Extra Trees variations
            for weight_name, weight_config in list(class_weight_configs.items())[
                :3
            ]:  # Test top 3
                model_configs[f"et_{weight_name}"] = {
                    "model": ExtraTreesClassifier,
                    "params": {
                        "n_estimators": 300,
                        "max_depth": 20,
                        "min_samples_split": 3,
                        "class_weight": weight_config,
                        "random_state": 42,
                    },
                    "type": "ExtraTrees",
                    "weight_config": weight_name,
                }

            # Gradient Boosting (no class weights, but test different learning rates)
            for lr in [0.05, 0.1, 0.15]:
                model_configs[f"gb_lr{lr}"] = {
                    "model": GradientBoostingClassifier,
                    "params": {
                        "n_estimators": 200,
                        "learning_rate": lr,
                        "max_depth": 6,
                        "random_state": 42,
                    },
                    "type": "GradientBoosting",
                    "learning_rate": lr,
                }

            validation_results = []

            print(
                f"\n🧪 Testing {len(model_configs)} different model configurations..."
            )
            print("-" * 70)

            for config_name, config in model_configs.items():
                try:
                    print(f"   Testing {config_name}...", end=" ")

                    # Train model
                    model = config["model"](**config["params"])
                    model.fit(X_train_scaled, y_train)

                    # Predict on validation set
                    y_pred = model.predict(X_val_scaled)

                    if hasattr(model, "predict_proba"):
                        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                    else:
                        y_pred_proba = y_pred.astype(float)

                    # Calculate metrics
                    accuracy = accuracy_score(y_val, y_pred)
                    try:
                        auc = roc_auc_score(y_val, y_pred_proba)
                    except:
                        auc = 0.5

                    # Calculate precision and recall for winners
                    precision = precision_score(y_val, y_pred, zero_division=0)
                    recall = recall_score(y_val, y_pred, zero_division=0)
                    f1 = f1_score(y_val, y_pred, zero_division=0)

                    # Test probability calibration quality
                    prob_range = y_pred_proba.max() - y_pred_proba.min()
                    prob_mean = y_pred_proba.mean()
                    prob_std = y_pred_proba.std()

                    # Realistic probability check (should be in reasonable range)
                    realistic_probs = np.sum(
                        (y_pred_proba >= 0.05) & (y_pred_proba <= 0.8)
                    ) / len(y_pred_proba)

                    result = {
                        "config_name": config_name,
                        "model_type": config["type"],
                        "accuracy": accuracy,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "prob_range": prob_range,
                        "prob_mean": prob_mean,
                        "prob_std": prob_std,
                        "realistic_prob_ratio": realistic_probs,
                        "config": config,
                        "model": model,
                    }

                    validation_results.append(result)

                    print(
                        f"Acc: {accuracy:.3f}, AUC: {auc:.3f}, F1: {f1:.3f}, Prob Range: {prob_range:.3f}"
                    )

                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            if not validation_results:
                print("❌ No models successfully trained")
                return None, None, None

            # Calculate composite score for model selection
            print("\n📊 CALCULATING COMPOSITE SCORES:")
            print("-" * 50)

            for result in validation_results:
                # Composite score weights different aspects
                composite_score = (
                    result["accuracy"] * 0.3  # Overall accuracy
                    + result["auc"] * 0.25  # Ranking ability
                    + result["f1_score"] * 0.2  # Balance of precision/recall
                    + result["realistic_prob_ratio"] * 0.15  # Probability realism
                    + min(result["prob_range"] / 0.5, 1.0)
                    * 0.1  # Good probability spread
                )
                result["composite_score"] = composite_score

            # Sort by composite score
            validation_results.sort(key=lambda x: x["composite_score"], reverse=True)

            print("\n🏆 TOP 5 MODEL CONFIGURATIONS:")
            print("-" * 70)
            print(
                f"{'Rank':<4} {'Model':<20} {'Accuracy':<8} {'AUC':<6} {'F1':<6} {'ProbRange':<10} {'Composite':<10}"
            )
            print("-" * 70)

            for i, result in enumerate(validation_results[:5]):
                print(
                    f"{i+1:<4} {result['config_name']:<20} {result['accuracy']:<8.3f} {result['auc']:<6.3f} {result['f1_score']:<6.3f} {result['prob_range']:<10.3f} {result['composite_score']:<10.3f}"
                )

            # Select best model
            best_result = validation_results[0]
            best_model = best_result["model"]

            print(f"\n✅ SELECTED BEST MODEL: {best_result['config_name']}")
            print(f"   📊 Composite Score: {best_result['composite_score']:.3f}")
            print(f"   📊 Accuracy: {best_result['accuracy']:.3f}")
            print(f"   📊 AUC: {best_result['auc']:.3f}")
            print(f"   📊 F1 Score: {best_result['f1_score']:.3f}")
            print(f"   📊 Probability Range: {best_result['prob_range']:.3f}")
            print(
                f"   📊 Realistic Probabilities: {best_result['realistic_prob_ratio']:.1%}"
            )

            return best_model, scaler, validation_results

        except Exception as e:
            print(f"❌ Auto-optimization failed: {e}")
            return None, None, None

    def validate_model_on_historical_races(
        self, model, scaler, feature_columns, test_df
    ):
        """Validate model performance by predicting past races and comparing against actual results"""
        try:
            print("\n🎯 HISTORICAL RACE VALIDATION:")
            print("=" * 50)

            # Prepare test data
            X_test = test_df[feature_columns]
            y_test = test_df["is_winner"]
            X_test_scaled = scaler.transform(X_test)

            # Get predictions
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict(X_test_scaled).astype(float)

            # Test different prediction thresholds
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            threshold_results = []

            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                threshold_results.append(
                    {
                        "threshold": threshold,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    }
                )

            print("\n📊 THRESHOLD OPTIMIZATION:")
            print(
                f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
            )
            print("-" * 55)

            for result in threshold_results:
                print(
                    f"{result['threshold']:<10.1f} {result['accuracy']:<10.3f} {result['precision']:<10.3f} {result['recall']:<10.3f} {result['f1_score']:<10.3f}"
                )

            # Find optimal threshold
            best_threshold_result = max(threshold_results, key=lambda x: x["f1_score"])
            optimal_threshold = best_threshold_result["threshold"]

            print(
                f"\n✅ OPTIMAL THRESHOLD: {optimal_threshold} (F1-Score: {best_threshold_result['f1_score']:.3f})"
            )

            # Detailed race-by-race analysis
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

            # Analyze prediction quality by probability ranges
            prob_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            bin_analysis = []

            for bin_min, bin_max in prob_bins:
                mask = (y_pred_proba >= bin_min) & (y_pred_proba < bin_max)
                if np.sum(mask) > 0:
                    bin_accuracy = accuracy_score(y_test[mask], y_pred_optimal[mask])
                    bin_count = np.sum(mask)
                    actual_win_rate = y_test[mask].mean()
                    predicted_win_rate = y_pred_proba[mask].mean()

                    bin_analysis.append(
                        {
                            "prob_range": f"{bin_min:.1f}-{bin_max:.1f}",
                            "count": bin_count,
                            "accuracy": bin_accuracy,
                            "actual_win_rate": actual_win_rate,
                            "predicted_win_rate": predicted_win_rate,
                            "calibration_error": abs(
                                actual_win_rate - predicted_win_rate
                            ),
                        }
                    )

            print("\n📊 PROBABILITY CALIBRATION ANALYSIS:")
            print(
                f"{'Prob Range':<12} {'Count':<8} {'Accuracy':<10} {'Actual WR':<12} {'Pred WR':<12} {'Cal Error':<12}"
            )
            print("-" * 75)

            for result in bin_analysis:
                print(
                    f"{result['prob_range']:<12} {result['count']:<8} {result['accuracy']:<10.3f} {result['actual_win_rate']:<12.3f} {result['predicted_win_rate']:<12.3f} {result['calibration_error']:<12.3f}"
                )

            # Overall calibration quality
            avg_calibration_error = np.mean(
                [r["calibration_error"] for r in bin_analysis]
            )
            print(f"\n📊 AVERAGE CALIBRATION ERROR: {avg_calibration_error:.3f}")

            # Top prediction analysis
            top_predictions_mask = y_pred_proba >= np.percentile(
                y_pred_proba, 90
            )  # Top 10% predictions
            if np.sum(top_predictions_mask) > 0:
                top_prediction_accuracy = y_test[top_predictions_mask].mean()
                print(
                    f"📊 TOP 10% PREDICTIONS WIN RATE: {top_prediction_accuracy:.1%} (Expected: ~{y_pred_proba[top_predictions_mask].mean():.1%})"
                )

            validation_summary = {
                "optimal_threshold": optimal_threshold,
                "optimal_f1_score": best_threshold_result["f1_score"],
                "optimal_accuracy": best_threshold_result["accuracy"],
                "optimal_precision": best_threshold_result["precision"],
                "optimal_recall": best_threshold_result["recall"],
                "avg_calibration_error": avg_calibration_error,
                "threshold_results": threshold_results,
                "bin_analysis": bin_analysis,
                "total_races_tested": len(test_df),
                "actual_win_rate": y_test.mean(),
                "predicted_win_rate": y_pred_proba.mean(),
            }

            return validation_summary

        except Exception as e:
            print(f"❌ Historical validation failed: {e}")
            return {}

    def create_advanced_ensemble_models(self, X_train, y_train):
        """Create advanced ensemble models for testing"""
        try:
            ensemble_models = {}

            # Voting Classifier with top performers
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=15, class_weight="balanced", random_state=42
            )
            et = ExtraTreesClassifier(
                n_estimators=200, max_depth=15, class_weight="balanced", random_state=42
            )
            gb = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
            )

            voting_clf = VotingClassifier(
                estimators=[("rf", rf), ("et", et), ("gb", gb)], voting="soft"
            )
            ensemble_models["voting_ensemble"] = voting_clf

            # Bagging ensemble
            bagging_clf = BaggingClassifier(
                estimator=ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=42,
                ),
                n_estimators=10,
                random_state=42,
            )
            ensemble_models["bagging_ensemble"] = bagging_clf

            # Add imbalanced-learn ensembles if available
            if self.imbalanced_models:
                for name, config in self.imbalanced_models.items():
                    ensemble_models[f"ensemble_{name}"] = config["model"](
                        **config["params"]
                    )

            return ensemble_models

        except Exception as e:
            print(f"⚠️ Error creating ensemble models: {e}")
            return {}

    def _run_comprehensive_validation(
        self, test_df, feature_columns, best_model, scaler
    ):
        """Comprehensive validation testing ML vs Traditional vs Combined approaches on past races"""
        try:
            validation_results = {}

            print("\n1️⃣ TESTING PURE ML PREDICTIONS:")
            print("-" * 50)

            # Test pure ML predictions
            X_test = test_df[feature_columns]
            y_test = test_df["is_winner"]
            X_test_scaled = scaler.transform(X_test)

            # ML predictions
            ml_predictions = best_model.predict(X_test_scaled)
            if hasattr(best_model, "predict_proba"):
                ml_probabilities = best_model.predict_proba(X_test_scaled)[:, 1]
            else:
                ml_probabilities = ml_predictions.astype(float)

            ml_accuracy = accuracy_score(y_test, ml_predictions)
            try:
                ml_auc = roc_auc_score(y_test, ml_probabilities)
            except:
                ml_auc = 0.5

            print(f"   📊 Pure ML Accuracy: {ml_accuracy:.3f}")
            print(f"   📊 Pure ML AUC: {ml_auc:.3f}")

            validation_results["pure_ml"] = {
                "accuracy": ml_accuracy,
                "auc": ml_auc,
                "predictions_count": len(ml_predictions),
            }

            print("\n2️⃣ TESTING PURE TRADITIONAL ANALYSIS:")
            print("-" * 50)

            # Test pure traditional analysis (if available)
            traditional_predictions = []
            traditional_scores = []

            if (
                TRADITIONAL_ANALYSIS_AVAILABLE
                and "traditional_overall_score" in test_df.columns
            ):
                traditional_scores = (
                    test_df["traditional_overall_score"].fillna(0.5).tolist()
                )

                # Convert traditional scores to binary predictions (threshold = 0.6)
                traditional_threshold = 0.6
                traditional_predictions = [
                    1 if score >= traditional_threshold else 0
                    for score in traditional_scores
                ]

                if len(traditional_predictions) > 0:
                    traditional_accuracy = accuracy_score(
                        y_test, traditional_predictions
                    )
                    try:
                        traditional_auc = roc_auc_score(y_test, traditional_scores)
                    except:
                        traditional_auc = 0.5

                    print(
                        f"   📊 Pure Traditional Accuracy: {traditional_accuracy:.3f}"
                    )
                    print(f"   📊 Pure Traditional AUC: {traditional_auc:.3f}")
                    print(
                        f"   📊 Average Traditional Score: {np.mean(traditional_scores):.3f}"
                    )

                    validation_results["pure_traditional"] = {
                        "accuracy": traditional_accuracy,
                        "auc": traditional_auc,
                        "avg_score": np.mean(traditional_scores),
                        "threshold_used": traditional_threshold,
                        "predictions_count": len(traditional_predictions),
                    }
                else:
                    print("   ⚠️ No valid traditional predictions generated")
                    validation_results["pure_traditional"] = {
                        "error": "No valid predictions"
                    }
            else:
                print("   ⚠️ Traditional analysis features not available in test data")
                validation_results["pure_traditional"] = {
                    "error": "Traditional features missing"
                }

            print("\n3️⃣ TESTING COMBINED APPROACH (ML + Traditional):")
            print("-" * 50)

            # Test combined approach (weighted average)
            if traditional_scores and len(traditional_scores) == len(ml_probabilities):
                # Combined scores (70% ML, 30% Traditional - as used in predictor)
                ml_weight = 0.7
                traditional_weight = 0.3

                combined_scores = [
                    (ml_weight * ml_prob) + (traditional_weight * trad_score)
                    for ml_prob, trad_score in zip(ml_probabilities, traditional_scores)
                ]

                # Convert to binary predictions
                combined_threshold = 0.5
                combined_predictions = [
                    1 if score >= combined_threshold else 0 for score in combined_scores
                ]

                combined_accuracy = accuracy_score(y_test, combined_predictions)
                try:
                    combined_auc = roc_auc_score(y_test, combined_scores)
                except:
                    combined_auc = 0.5

                print(f"   📊 Combined Accuracy: {combined_accuracy:.3f}")
                print(f"   📊 Combined AUC: {combined_auc:.3f}")
                print(
                    f"   📊 ML Weight: {ml_weight}, Traditional Weight: {traditional_weight}"
                )
                print(f"   📊 Average Combined Score: {np.mean(combined_scores):.3f}")

                validation_results["combined"] = {
                    "accuracy": combined_accuracy,
                    "auc": combined_auc,
                    "ml_weight": ml_weight,
                    "traditional_weight": traditional_weight,
                    "avg_score": np.mean(combined_scores),
                    "threshold_used": combined_threshold,
                    "predictions_count": len(combined_predictions),
                }

                # Performance comparison
                print("\n4️⃣ PERFORMANCE COMPARISON:")
                print("-" * 50)

                improvement_vs_ml = combined_accuracy - ml_accuracy
                improvement_vs_traditional = combined_accuracy - validation_results.get(
                    "pure_traditional", {}
                ).get("accuracy", 0)

                print(f"   📈 Combined vs Pure ML: {improvement_vs_ml:+.3f} accuracy")
                if (
                    "pure_traditional" in validation_results
                    and "accuracy" in validation_results["pure_traditional"]
                ):
                    print(
                        f"   📈 Combined vs Pure Traditional: {improvement_vs_traditional:+.3f} accuracy"
                    )

                validation_results["performance_comparison"] = {
                    "combined_vs_ml_improvement": improvement_vs_ml,
                    "combined_vs_traditional_improvement": improvement_vs_traditional,
                    "best_approach": (
                        "combined" if combined_accuracy >= ml_accuracy else "ml"
                    ),
                }

            else:
                print(
                    "   ⚠️ Cannot test combined approach - traditional scores not available"
                )
                validation_results["combined"] = {
                    "error": "Traditional scores not available for combination"
                }

            print("\n5️⃣ TRADITIONAL FEATURE IMPORTANCE ANALYSIS:")
            print("-" * 50)

            # Analyze correlation between traditional and ML features
            if TRADITIONAL_ANALYSIS_AVAILABLE:
                traditional_feature_columns = [
                    col for col in feature_columns if col.startswith("traditional_")
                ]
                print(
                    f"   📊 Traditional features in model: {len(traditional_feature_columns)}"
                )

                if traditional_feature_columns:
                    # Calculate correlations with outcome
                    feature_correlations = []
                    for col in traditional_feature_columns:
                        if col in test_df.columns:
                            corr = test_df[col].corr(test_df["is_winner"])
                            if not pd.isna(corr):
                                feature_correlations.append((col, abs(corr)))

                    # Sort by correlation strength
                    feature_correlations.sort(key=lambda x: x[1], reverse=True)

                    print("   📊 Top traditional feature correlations with winning:")
                    for i, (feature, corr) in enumerate(feature_correlations[:10]):
                        print(f"      {i+1}. {feature}: {corr:.3f}")

                    validation_results["traditional_feature_analysis"] = {
                        "traditional_features_count": len(traditional_feature_columns),
                        "top_correlations": feature_correlations[:10],
                    }
                else:
                    print("   ⚠️ No traditional features found in model")

            print("\n6️⃣ DIVERGENCE ANALYSIS:")
            print("-" * 50)

            # Analyze cases where ML and Traditional strongly diverge
            if traditional_scores and len(traditional_scores) == len(ml_probabilities):
                divergence_threshold = 0.3
                divergent_cases = []

                for i, (ml_prob, trad_score, actual) in enumerate(
                    zip(ml_probabilities, traditional_scores, y_test)
                ):
                    divergence = abs(ml_prob - trad_score)
                    if divergence >= divergence_threshold:
                        divergent_cases.append(
                            {
                                "index": i,
                                "ml_score": ml_prob,
                                "traditional_score": trad_score,
                                "actual_winner": actual,
                                "divergence": divergence,
                                "dog_name": test_df.iloc[i]["dog_name"],
                                "race_id": test_df.iloc[i]["race_id"],
                            }
                        )

                print(
                    f"   📊 Found {len(divergent_cases)} cases with divergence >= {divergence_threshold}"
                )

                if divergent_cases:
                    # Analyze which approach was more accurate in divergent cases
                    ml_correct_divergent = 0
                    traditional_correct_divergent = 0

                    for case in divergent_cases:
                        ml_prediction = 1 if case["ml_score"] >= 0.5 else 0
                        traditional_prediction = (
                            1 if case["traditional_score"] >= 0.6 else 0
                        )

                        if ml_prediction == case["actual_winner"]:
                            ml_correct_divergent += 1
                        if traditional_prediction == case["actual_winner"]:
                            traditional_correct_divergent += 1

                    ml_divergent_accuracy = ml_correct_divergent / len(divergent_cases)
                    traditional_divergent_accuracy = (
                        traditional_correct_divergent / len(divergent_cases)
                    )

                    print(
                        f"   📊 ML accuracy in divergent cases: {ml_divergent_accuracy:.3f}"
                    )
                    print(
                        f"   📊 Traditional accuracy in divergent cases: {traditional_divergent_accuracy:.3f}"
                    )

                    # Show some examples
                    print(f"   📊 Example divergent cases (first 3):")
                    for i, case in enumerate(divergent_cases[:3]):
                        print(
                            f"      {i+1}. {case['dog_name']} - ML: {case['ml_score']:.3f}, Traditional: {case['traditional_score']:.3f}, Actual: {case['actual_winner']}"
                        )

                    validation_results["divergence_analysis"] = {
                        "divergent_cases_count": len(divergent_cases),
                        "divergence_threshold": divergence_threshold,
                        "ml_accuracy_in_divergent": ml_divergent_accuracy,
                        "traditional_accuracy_in_divergent": traditional_divergent_accuracy,
                        "example_cases": divergent_cases[:5],  # Store first 5 examples
                    }

            print("\n✅ COMPREHENSIVE VALIDATION COMPLETE")
            print("=" * 70)

            return validation_results

        except Exception as e:
            print(f"❌ Error in comprehensive validation: {e}")
            return {"error": str(e)}

    def predict_race(self, race_data):
        """Make predictions for a complete race"""
        if not self.models_loaded:
            print("❌ Models not loaded. Please train models first or load from file.")
            return None

        print(f"\n🔮 RACE PREDICTION SYSTEM")
        print("═" * 50)
        print(f"   🏁 Dogs to analyze: {len(race_data)}")
        print(f"   🕐 Prediction time: {datetime.now().strftime('%H:%M:%S')}")
        print("═" * 50)

        predictions = []
        prediction_start = time.time()

        print(f"   📊 Processing individual dog predictions...")

        for i, (_, dog_data) in enumerate(race_data.iterrows(), 1):
            if (
                i % 3 == 1 or len(race_data) <= 8
            ):  # Show progress for every 3rd dog or all if ≤8 dogs
                print(
                    f"      [{i}/{len(race_data)}] {dog_data.get('dog_name', f'Dog {i}')}..."
                )

            try:
                # Example feature extraction and prediction code
                # Replace with actual feature extraction
                features = self.extract_features(dog_data)
                scaled_features = self.scaler.transform([features])

                prediction = self.model.predict(scaled_features)
                win_prob = prediction[0]  # Or logic to extract probability
                predictions.append(
                    {
                        "dog_name": dog_data["dog_clean_name"],
                        "win_probability": win_prob,
                    }
                )
            except Exception as e:
                print(
                    f"      ⚠️ Error predicting {dog_data.get('dog_name', f'Dog {i}')}: {e}"
                )
                continue

        prediction_time = time.time() - prediction_start

        if not predictions:
            print("❌ No valid predictions generated")
            return None

        predictions_df = pd.DataFrame(predictions)

        # Sort by win probability
        predictions_df = predictions_df.sort_values("win_probability", ascending=False)

        print(f"\n   ✅ Predictions completed in {prediction_time:.1f}s")
        print(f"\n🏆 RACE PREDICTIONS RANKED:")
        print("═" * 80)
        print(
            f"{'Rank':<4} {'Dog Name':<20} {'Win Prob':<10} {'Place Prob':<11} {'Confidence':<12} {'Status':<8}"
        )
        print("─" * 80)

        for i, (_, pred) in enumerate(predictions_df.iterrows()):
            rank = i + 1

            # Confidence levels and colors
            if pred.get("confidence_score", 0.5) > 0.7:
                confidence_level = "HIGH"
                status = "🟢 STRONG"
            elif pred.get("confidence_score", 0.5) > 0.5:
                confidence_level = "MEDIUM"
                status = "🟡 FAIR"
            else:
                confidence_level = "LOW"
                status = "🔴 WEAK"

            print(
                f"{rank:<4} {pred['dog_name']:<20} {pred['win_probability']:.3f}      {pred['place_probability']:.3f}       {confidence_level:<12} {status}"
            )

        # Summary statistics
        avg_win_prob = predictions_df["win_probability"].mean()
        top_dog_advantage = (
            predictions_df.iloc[0]["win_probability"]
            - predictions_df.iloc[1]["win_probability"]
            if len(predictions_df) > 1
            else 0
        )

        print("═" * 80)
        print(f"📈 RACE ANALYSIS SUMMARY:")
        print(
            f"   🎯 Favorite: {predictions_df.iloc[0]['dog_name']} ({predictions_df.iloc[0]['win_probability']:.3f})"
        )
        print(f"   📊 Average Win Probability: {avg_win_prob:.3f}")
        print(f"   ⚡ Top Dog Advantage: {top_dog_advantage:.3f}")
        print(
            f"   🏁 Field Competitiveness: {'TIGHT' if top_dog_advantage < 0.1 else 'MODERATE' if top_dog_advantage < 0.2 else 'CLEAR FAVORITE'}"
        )

        return predictions_df

    def _extract_participating_dogs_from_race(self, race_df):
        """Extract participating dogs from race CSV"""
        try:
            dogs = []
            current_dog_name = None

            for idx, row in race_df.iterrows():
                dog_name_raw = str(row.get("Dog Name", "")).strip()

                # Check if this is a new dog or continuation of previous
                if dog_name_raw not in ['""', "", "nan"] and dog_name_raw != "nan":
                    # New dog - clean the name
                    current_dog_name = dog_name_raw
                    if ". " in current_dog_name:
                        parts = current_dog_name.split(". ", 1)
                        if len(parts) == 2:
                            try:
                                box_number = int(parts[0])
                                current_dog_name = parts[1]
                            except (ValueError, TypeError):
                                box_number = len(dogs) + 1
                        else:
                            box_number = len(dogs) + 1
                    else:
                        box_number = len(dogs) + 1

                    dogs.append(
                        {
                            "name": current_dog_name,
                            "box": box_number,
                            "raw_name": dog_name_raw,
                        }
                    )

            return dogs

        except Exception as e:
            print(f"⚠️ Error extracting participating dogs: {e}")
            return []

    def _extract_race_info_from_path(self, race_file_path):
        """Extract race information from file path"""
        try:
            filename = os.path.basename(race_file_path)
            parts = filename.replace(".csv", "").split(" - ")

            if len(parts) >= 3:
                race_number = parts[0].replace("Race ", "")
                venue = parts[1]
                date_str = parts[2]

                return {
                    "filename": filename,
                    "race_number": race_number,
                    "venue": venue,
                    "race_date": date_str,
                    "filepath": race_file_path,
                }
            else:
                return {
                    "filename": filename,
                    "race_number": "Unknown",
                    "venue": "Unknown",
                    "race_date": "Unknown",
                    "filepath": race_file_path,
                }

        except Exception as e:
            print(f"⚠️ Error extracting race info: {e}")
            return {
                "filename": os.path.basename(race_file_path),
                "race_number": "Unknown",
                "venue": "Unknown",
                "race_date": "Unknown",
                "filepath": race_file_path,
            }

    def _create_dog_features_for_prediction(
        self, dog_info, dog_historical, dog_form_data, race_info, feature_columns
    ):
        """Create feature vector for a dog for prediction purposes"""
        try:
            # Initialize feature dict with defaults
            features = {}

            # Process database historical data
            if len(dog_historical) > 0:
                positions = []
                times = []
                weights = []
                odds = []

                for _, hist_row in dog_historical.iterrows():
                    # Position data
                    if pd.notna(hist_row["finish_position"]):
                        pos_str = str(hist_row["finish_position"]).strip()
                        if pos_str not in ["", "N/A", "None", "nan"]:
                            pos_cleaned = "".join(filter(str.isdigit, pos_str))
                            if pos_cleaned and 1 <= int(pos_cleaned) <= 10:
                                positions.append(int(pos_cleaned))

                    # Time data
                    try:
                        if (
                            pd.notna(hist_row["individual_time"])
                            and float(hist_row["individual_time"]) > 0
                        ):
                            times.append(float(hist_row["individual_time"]))
                    except (ValueError, TypeError):
                        pass

                    # Weight data
                    try:
                        if (
                            pd.notna(hist_row["weight"])
                            and float(hist_row["weight"]) > 0
                        ):
                            weights.append(float(hist_row["weight"]))
                    except (ValueError, TypeError):
                        pass

                    # Odds data
                    try:
                        if (
                            pd.notna(hist_row["starting_price"])
                            and float(hist_row["starting_price"]) > 0
                        ):
                            odds.append(float(hist_row["starting_price"]))
                    except (ValueError, TypeError):
                        pass
            else:
                positions = []
                times = []
                weights = []
                odds = []

            # Process form guide data
            form_positions = []
            form_times = []
            form_weights = []
            form_odds = []

            for form_entry in dog_form_data[:20]:  # Use up to 20 historical races
                try:
                    # Parse position
                    place_str = form_entry.get("place", "").strip()
                    if place_str and place_str.isdigit():
                        position = int(place_str)
                        if 1 <= position <= 10:
                            form_positions.append(position)

                    # Parse time
                    time_str = form_entry.get("time", "").strip()
                    if time_str:
                        try:
                            time_val = float(time_str)
                            if 15.0 <= time_val <= 60.0:
                                form_times.append(time_val)
                        except (ValueError, TypeError):
                            pass

                    # Parse weight
                    weight_str = form_entry.get("weight", "").strip()
                    if weight_str:
                        try:
                            weight_val = float(weight_str)
                            if 20.0 <= weight_val <= 40.0:
                                form_weights.append(weight_val)
                        except (ValueError, TypeError):
                            pass

                    # Parse starting price
                    sp_str = form_entry.get("starting_price", "").strip()
                    if sp_str:
                        try:
                            sp_val = float(sp_str)
                            if 1.0 <= sp_val <= 1000.0:
                                form_odds.append(sp_val)
                        except (ValueError, TypeError):
                            pass

                except Exception:
                    continue

            # Combine all data
            all_positions = positions + form_positions
            all_times = times + form_times
            all_weights = weights + form_weights
            all_odds = odds + form_odds

            if not all_positions:
                return None

            # Calculate core features (matching the trained model)
            features["avg_position"] = np.mean(all_positions)
            features["recent_form_avg"] = (
                np.mean(all_positions[:8])
                if len(all_positions) >= 8
                else np.mean(all_positions)
            )
            features["market_confidence"] = (
                1 / (np.mean(all_odds) + 1) if all_odds else 0.1
            )
            features["current_odds_log"] = np.log(10)  # Default odds
            features["venue_experience"] = len([p for p in positions])  # Simplified
            features["place_rate"] = sum(1 for p in all_positions if p <= 3) / len(
                all_positions
            )
            features["current_weight"] = np.mean(all_weights) if all_weights else 30.0
            features["time_consistency"] = (
                1 / (np.std(all_times) + 0.1) if len(all_times) > 1 else 0.5
            )
            features["win_rate"] = sum(1 for p in all_positions if p == 1) / len(
                all_positions
            )

            # Additional comprehensive features
            features["long_term_form_trend"] = 0  # Simplified
            features["position_consistency"] = 1 / (np.std(all_positions) + 0.1)
            features["avg_time"] = np.mean(all_times) if all_times else 30.0
            features["best_time"] = min(all_times) if all_times else 28.0
            features["time_improvement_trend"] = 0  # Simplified
            features["avg_weight"] = np.mean(all_weights) if all_weights else 30.0
            features["weight_consistency"] = (
                1 / (np.std(all_weights) + 0.1) if len(all_weights) > 1 else 0.5
            )
            features["weight_vs_avg"] = 0  # Simplified
            features["distance_specialization"] = 0.1  # Simplified
            features["grade_experience"] = 5  # Default
            features["days_since_last"] = 14  # Default
            features["fitness_score"] = features["place_rate"] * features["win_rate"]
            features["competition_strength"] = 0.5  # Default
            features["box_win_rate"] = 0.1  # Default
            features["current_box"] = dog_info.get("box", 4)
            features["field_size"] = 6  # Default
            features["historical_races_count"] = len(all_positions)

            # Encoded features (defaults)
            features["venue_encoded"] = 0
            features["track_condition_encoded"] = 0
            features["grade_encoded"] = 0
            features["distance_numeric"] = 500.0

            # Ensure all required features are present
            feature_vector = []
            for feature_name in feature_columns:
                feature_vector.append(features.get(feature_name, 0.0))

            return np.array(feature_vector).reshape(1, -1)

        except Exception as e:
            print(f"⚠️ Error creating features for {dog_info['name']}: {e}")
            return None


def main():
    """Main function for comprehensive enhanced model system"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Enhanced ML Model System"
    )
    parser.add_argument(
        "--command",
        choices=["analyze", "status"],
        default="analyze",
        help="Command to execute",
    )

    args = parser.parse_args()

    print("🚀 Comprehensive Enhanced ML Model System")
    print("=" * 70)

    system = ComprehensiveEnhancedMLSystem()

    if args.command == "analyze":
        system.run_comprehensive_analysis()

    elif args.command == "status":
        # Show system status
        model_files = len(list(system.models_dir.glob("*.joblib")))
        result_files = len(
            list(system.results_dir.glob("comprehensive_analysis_*.json"))
        )
        form_files = (
            len(list(system.downloaded_forms_dir.glob("*.csv")))
            if system.downloaded_forms_dir.exists()
            else 0
        )

        print("\\n📊 COMPREHENSIVE SYSTEM STATUS:")
        print("=" * 50)
        print(f"Model files: {model_files}")
        print(f"Analysis result files: {result_files}")
        print(f"Form guide CSV files: {form_files}")
        print(f"Models directory: {system.models_dir}")
        print(f"Results directory: {system.results_dir}")


if __name__ == "__main__":
    main()
