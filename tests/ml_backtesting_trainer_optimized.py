#!/usr/bin/env python3
"""
ML Backtesting and Training System
==================================

This script trains and validates ML models on past races with known outcomes,
allowing us to optimize feature weightings and find winning correlations.

Features:
- Backtest predictions against actual race results
- Optimize model parameters using cross-validation
- Analyze feature importance and correlations
- Generate performance reports with accuracy metrics
- Test different model configurations

Author: AI Assistant
Date: July 24, 2025
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import sys
import time
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    import optuna
    from imblearn.over_sampling import SMOTENC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, f1_score
    import mlflow
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, log_loss, roc_auc_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML libraries not available: {e}")
    SKLEARN_AVAILABLE = False

class MLBacktestingTrainer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results_dir = Path('./ml_backtesting_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Model configurations to test
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'max_iter': [1000, 2000, 3000],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        self.correlation_analysis = {}
        
        print("🎯 ML Backtesting Trainer Initialized")
        
    def load_historical_race_data(self, months_back=12):
        """Load historical race data with outcomes for training/validation"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months_back * 30)
            
            print(f"📊 Loading historical data from {start_date.date()} to {end_date.date()}")
            
            # Enhanced query with more features
            query = """
            SELECT 
                drd.race_id,
                drd.dog_clean_name,
                drd.finish_position,
                drd.box_number,
                drd.weight,
                drd.starting_price,
                drd.performance_rating,
                drd.speed_rating,
                drd.class_rating,
                drd.individual_time,
                drd.margin,
                rm.field_size,
                rm.distance,
                rm.venue,
                rm.track_condition,
                rm.weather,
                rm.temperature,
                rm.grade,
                rm.race_date,
                rm.race_time,
                -- Winner information for context
                (SELECT COUNT(*) FROM dog_race_data WHERE race_id = drd.race_id) as total_runners,
                (SELECT MIN(individual_time) FROM dog_race_data WHERE race_id = drd.race_id AND individual_time IS NOT NULL) as winning_time
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.finish_position != 'N/A'
            AND rm.race_date >= ?
            AND rm.race_date <= ?
            AND drd.individual_time IS NOT NULL
            ORDER BY rm.race_date ASC, drd.race_id, drd.finish_position
            """
            
            df = pd.read_sql_query(query, conn, params=[start_date.isoformat(), end_date.isoformat()])
            conn.close()
            
            print(f"✅ Loaded {len(df)} race records covering {df['race_id'].nunique()} races")
            print(f"   📈 Dogs: {df['dog_clean_name'].nunique()}")
            print(f"   🏟️ Venues: {df['venue'].nunique()}")
            print(f"   📅 Date range: {df['race_date'].min()} to {df['race_date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return None
    
    def create_enhanced_features(self, df):
        """Create enhanced features for each dog based on historical performance"""
        print("\n🔧 Creating enhanced features from historical data...")
        print("═" * 70)
        
        enhanced_records = []
        total_records = len(df)
        start_time = time.time()
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                elapsed = time.time() - start_time
                progress = idx / total_records * 100
                records_per_sec = idx / elapsed if elapsed > 0 else 0
                eta = (total_records - idx) / records_per_sec if records_per_sec > 0 else 0
                
                # Create progress bar
                bar_length = 30
                filled_length = int(bar_length * idx // total_records)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"\r   [{bar}] {progress:6.1f}% | {idx:,}/{total_records:,} | {records_per_sec:.1f} rec/s | ETA: {eta:.0f}s", end="")
                sys.stdout.flush()
            
            dog_name = row['dog_clean_name']
            race_date = row['race_date']
            race_id = row['race_id']
            
            # Get historical performance up to this race date (but not including this race)
            historical_data = df[
                (df['dog_clean_name'] == dog_name) & 
                (df['race_date'] < race_date)
            ].sort_values('race_date', ascending=False)
            
            if len(historical_data) >= 1:  # Need at least some history
                # Calculate enhanced features
                enhanced_features = self.calculate_dog_features(historical_data, row)
                # Handle finish position parsing
                try:
                    finish_position = int(str(row['finish_position']).replace('=', ''))
                except (ValueError, TypeError):
                    continue  # Skip records with invalid finish positions
                
                enhanced_features.update({
                    'race_id': race_id,
                    'dog_name': dog_name,
                    'finish_position': finish_position,
                    'is_winner': 1 if finish_position == 1 else 0,
                    'is_placer': 1 if finish_position <= 3 else 0,
                    'is_top_half': 1 if finish_position <= row.get('total_runners', 8) // 2 else 0,
                    'race_date': race_date,
                    # Current race context
                    'current_box': row['box_number'],
                    'current_weight': row['weight'],
                    'current_odds': row['starting_price'],
                    'field_size': row['field_size'],
                    'distance': row['distance'],
                    'venue': row['venue'],
                    'track_condition': row['track_condition'],
                    'grade': row['grade']
                })
                
                enhanced_records.append(enhanced_features)
        
        enhanced_df = pd.DataFrame(enhanced_records)
        total_time = time.time() - start_time
        print(f"\n\n✅ Feature creation completed!")
        print(f"   📊 Records processed: {total_records:,}")
        print(f"   📈 Enhanced records created: {len(enhanced_df):,}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   🚀 Processing rate: {total_records/total_time:.1f} records/second")
        
        return enhanced_df
    
    def calculate_dog_features(self, historical_data, current_race):
        """Calculate comprehensive features for a dog based on historical performance"""
        try:
            # Basic historical stats - Clean data before processing
            def clean_numeric_value(value, default=0):
                """Clean numeric values that may have = signs or other characters"""
                if pd.isna(value):
                    return default
                try:
                    # Convert to string and clean common issues
                    clean_val = str(value).strip()
                    # Remove = signs and other non-numeric characters except . and -
                    clean_val = ''.join(c for c in clean_val if c.isdigit() or c in '.-')
                    if clean_val == '' or clean_val == '.' or clean_val == '-':
                        return default
                    return float(clean_val)
                except (ValueError, TypeError):
                    return default
            
            positions = [int(clean_numeric_value(p, 5)) for p in historical_data['finish_position'] if clean_numeric_value(p) != 0]
            times = [clean_numeric_value(t, 30.0) for t in historical_data['individual_time'] if clean_numeric_value(t) > 0]
            margins = [clean_numeric_value(m, 0.0) for m in historical_data['margin'] if clean_numeric_value(m) != 0]
            weights = [clean_numeric_value(w, 30.0) for w in historical_data['weight'] if clean_numeric_value(w) > 0]
            odds = [clean_numeric_value(o, 10.0) for o in historical_data['starting_price'] if clean_numeric_value(o) > 0]
            boxes = [int(clean_numeric_value(b, 1)) for b in historical_data['box_number'] if clean_numeric_value(b) > 0]
            
            # Recent form (last 5 races)
            recent_positions = positions[:5]
            recent_form_avg = np.mean(recent_positions) if recent_positions else 5.0
            
            # Form trend (improving = negative slope, declining = positive slope)
            if len(positions) >= 3:
                x = np.arange(len(positions))
                form_trend = -np.polyfit(x, positions, 1)[0] if len(positions) > 1 else 0
            else:
                form_trend = 0
            
            # Performance metrics
            win_rate = sum(1 for p in positions if p == 1) / len(positions) if positions else 0
            place_rate = sum(1 for p in positions if p <= 3) / len(positions) if positions else 0
            avg_position = np.mean(positions) if positions else 5.0
            position_consistency = 1 / (np.std(positions) + 1) if len(positions) > 1 else 0.5
            
            # Time analysis
            avg_time = np.mean(times) if times else 30.0
            best_time = min(times) if times else 30.0
            time_consistency = 1 / (np.std(times) + 0.1) if len(times) > 1 else 0.5
            
            # Weight analysis
            avg_weight = np.mean(weights) if weights else 30.0
            weight_trend = np.polyfit(range(len(weights)), weights, 1)[0] if len(weights) > 1 else 0
            
            # Market confidence (inverse of average odds)
            market_confidence = 1 / (np.mean(odds) + 1) if odds else 0.1
            
            # Box draw analysis
            box_versatility = len(set(boxes)) if boxes else 1
            preferred_boxes = [b for b in range(1, 9) if boxes.count(b) >= 2]
            
            # Distance analysis
            distances = []
            for d in historical_data['distance']:
                if pd.notna(d):
                    try:
                        distances.append(float(str(d).replace('m', '')))
                    except ValueError:
                        pass  # Skip invalid distances
            try:
                current_distance = float(current_race.get('distance', '500').replace('m', ''))
            except ValueError:
                current_distance = 500  # Default if parsing fails
            distance_experience = sum(1 for d in distances if abs(d - current_distance) <= 50) / len(distances) if distances else 0
            
            # Venue analysis
            venues = list(historical_data['venue'])
            current_venue = current_race.get('venue', '')
            venue_experience = venues.count(current_venue) / len(venues) if venues else 0
            
            # Class analysis
            performance_ratings = [float(r) for r in historical_data['performance_rating'] if pd.notna(r)]
            speed_ratings = [float(r) for r in historical_data['speed_rating'] if pd.notna(r)]
            class_ratings = [float(r) for r in historical_data['class_rating'] if pd.notna(r)]
            
            # Recent activity
            if len(historical_data) > 0:
                last_race_date = pd.to_datetime(historical_data.iloc[0]['race_date'])
                current_race_date = pd.to_datetime(current_race['race_date'])
                days_since_last = (current_race_date - last_race_date).days
            else:
                days_since_last = 365
            
            return {
                'races_count': len(positions),
                'recent_form_avg': recent_form_avg,
                'form_trend': form_trend,
                'win_rate': win_rate,
                'place_rate': place_rate,
                'avg_position': avg_position,
                'position_consistency': position_consistency,
                'avg_time': avg_time,
                'best_time': best_time,
                'time_consistency': time_consistency,
                'avg_weight': avg_weight,
                'weight_trend': weight_trend,
                'market_confidence': market_confidence,
                'box_versatility': box_versatility,
                'preferred_boxes_count': len(preferred_boxes),
                'distance_experience': distance_experience,
                'venue_experience': venue_experience,
                'avg_performance_rating': np.mean(performance_ratings) if performance_ratings else 50.0,
                'avg_speed_rating': np.mean(speed_ratings) if speed_ratings else 50.0,
                'avg_class_rating': np.mean(class_ratings) if class_ratings else 50.0,
                'days_since_last_race': days_since_last,
                'recent_races_last_30d': sum(1 for _, race in historical_data.iterrows() 
                                           if (pd.to_datetime(current_race['race_date']) - pd.to_datetime(race['race_date'])).days <= 30)
            }
            
        except Exception as e:
            print(f"Warning: Feature calculation error: {e}")
            # Return default features
            return {
                'races_count': 0, 'recent_form_avg': 5.0, 'form_trend': 0, 'win_rate': 0,
                'place_rate': 0, 'avg_position': 5.0, 'position_consistency': 0.5,
                'avg_time': 30.0, 'best_time': 30.0, 'time_consistency': 0.5,
                'avg_weight': 30.0, 'weight_trend': 0, 'market_confidence': 0.1,
                'box_versatility': 1, 'preferred_boxes_count': 0,
                'distance_experience': 0, 'venue_experience': 0,
                'avg_performance_rating': 50.0, 'avg_speed_rating': 50.0, 'avg_class_rating': 50.0,
                'days_since_last_race': 365, 'recent_races_last_30d': 0
            }
    
    def prepare_ml_dataset(self, enhanced_df):
        """Prepare the dataset for ML training with proper encoding"""
        print("🔧 Preparing ML dataset...")
        
        # Feature columns for ML
        feature_columns = [
            'races_count', 'recent_form_avg', 'form_trend', 'win_rate', 'place_rate',
            'avg_position', 'position_consistency', 'avg_time', 'best_time', 'time_consistency',
            'avg_weight', 'weight_trend', 'market_confidence', 'box_versatility',
            'preferred_boxes_count', 'distance_experience', 'venue_experience',
            'avg_performance_rating', 'avg_speed_rating', 'avg_class_rating',
            'days_since_last_race', 'recent_races_last_30d',
            'current_box', 'current_weight', 'field_size', 'distance'
        ]
        
        # Add encoded categorical features
        le_venue = LabelEncoder()
        le_condition = LabelEncoder()
        le_grade = LabelEncoder()
        
        enhanced_df['venue_encoded'] = le_venue.fit_transform(enhanced_df['venue'].fillna('Unknown'))
        enhanced_df['track_condition_encoded'] = le_condition.fit_transform(enhanced_df['track_condition'].fillna('Good'))
        enhanced_df['grade_encoded'] = le_grade.fit_transform(enhanced_df['grade'].fillna('Unknown'))
        
        feature_columns.extend(['venue_encoded', 'track_condition_encoded', 'grade_encoded'])
        
        # Handle odds (log transform to handle extreme values)
        enhanced_df['current_odds_log'] = np.log(enhanced_df['current_odds'].fillna(10.0) + 1)
        feature_columns.append('current_odds_log')
        
        # Handle distance field (convert strings like '525m' to numeric)
        if 'distance' in enhanced_df.columns:
            enhanced_df['distance_numeric'] = enhanced_df['distance'].apply(
                lambda x: float(str(x).replace('m', '')) if pd.notna(x) else 500.0
            )
            # Replace 'distance' with 'distance_numeric' in feature columns
            if 'distance' in feature_columns:
                feature_columns = [col if col != 'distance' else 'distance_numeric' for col in feature_columns]
        
        # Filter to complete cases
        complete_df = enhanced_df[feature_columns + ['is_winner', 'is_placer', 'race_date', 'race_id', 'dog_name']].copy()
        
        # Fill missing values
        imputer = SimpleImputer(strategy='median')
        complete_df[feature_columns] = imputer.fit_transform(complete_df[feature_columns])
        
        print(f"✅ Prepared dataset with {len(complete_df)} samples and {len(feature_columns)} features")
        
        return complete_df, feature_columns
    
    def time_series_split_validation(self, df, feature_columns, target_column='is_winner', n_splits=5):
        """Perform time-series cross-validation to avoid look-ahead bias"""
        print(f"\n🔍 Time-Series Cross-Validation: {target_column.upper()} Prediction")
        print("═" * 70)
        print(f"   📊 Dataset size: {len(df):,} samples")
        print(f"   🔢 Features: {len(feature_columns)}")
        print(f"   📈 CV splits: {n_splits}")
        print(f"   🎯 Target: {target_column}")
        print()
        
        # Sort by date to ensure proper time series split
        df_sorted = df.sort_values('race_date')
        
        X = df_sorted[feature_columns]
        y = df_sorted[target_column]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {}
        
        model_count = len(self.model_configs)
        for i, (model_name, config) in enumerate(self.model_configs.items(), 1):
            print(f"   [{i}/{model_count}] Testing {model_name.replace('_', ' ').title()}...", end=" ")
            
            model_class = config['model']
            scores = []
            feature_importances = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model with default params for quick validation
                if model_name == 'logistic_regression':
                    model = model_class(random_state=42, max_iter=1000)
                else:
                    model = model_class(random_state=42)
                
                model.fit(X_train_scaled, y_train)
                
                # Predict and score
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                scores.append(accuracy)
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
            
            # Average results
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            results[model_name] = {
                'avg_accuracy': avg_score,
                'std_accuracy': std_score,
                'cv_scores': scores
            }
            
            if feature_importances:
                avg_importance = np.mean(feature_importances, axis=0)
                self.feature_importance[f"{model_name}_{target_column}"] = list(zip(feature_columns, avg_importance))
            
            # Show result with color coding
            if avg_score >= 0.65:
                status = "🟢 EXCELLENT"
            elif avg_score >= 0.55:
                status = "🟡 GOOD"
            else:
                status = "🔴 NEEDS WORK"
            
            print(f"{status} | Accuracy: {avg_score:.3f} ± {std_score:.3f}")
        
        return results
    
    def optimize_best_model(self, df, feature_columns, target_column='is_winner'):
        """Optimize the best performing model using grid search"""
        print(f"\n🎯 Model Optimization: {target_column.upper()} Prediction")
        print("═" * 70)
        
        # Sort by date
        df_sorted = df.sort_values('race_date')
        
        # Use 80% for training, 20% for final test (respecting time order)
        split_point = int(0.8 * len(df_sorted))
        train_df = df_sorted.iloc[:split_point]
        test_df = df_sorted.iloc[split_point:]
        
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        # Apply SMOTENC
        categorical_indices = [i for i, col in enumerate(feature_columns) if df[col].dtype == 'object']
        smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
        X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

        # Scale features
        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        best_model = None
        best_score = 0
        best_params = None
        best_model_name = None
        
        total_models = len(self.model_configs)
        print(f"   🔧 Optimizing {total_models} model types with grid search...")
        print()
        
        for i, (model_name, config) in enumerate(self.model_configs.items(), 1):
            model_display = model_name.replace('_', ' ').title()
            print(f"   [{i}/{total_models}] {model_display}...")
            
            model_class = config['model']
            param_grid = config['params']
            
            # Simplified param grid for faster execution
            if model_name == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, None],
                    'min_samples_split': [2, 5]
                }
            elif model_name == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
            elif model_name == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'max_iter': [1000]
                }
            
            # Grid search with time series CV
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                model_class(random_state=42),
                param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )

            # Define optuna objective
            def objective(trial):
                model_class = config['model']
                params = {key: trial.suggest_categorical(key, values) for key, values in config['params'].items()}
                model = model_class(**params, random_state=42)
                model = CalibratedClassifierCV(model, method='sigmoid')
                score = cross_val_score(model, X_resampled_scaled, y_resampled, cv=tscv, scoring='f1').mean()
                return score

            # Optuna study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=30)

            # Retrain with best params
            best_params = study.best_params
            best_model = model_class(**best_params, random_state=42)
            best_model = CalibratedClassifierCV(best_model, method='sigmoid')
            best_model.fit(X_resampled_scaled, y_resampled)

            # Evaluate on test set
            y_pred = best_model.predict(X_test_scaled)
            test_score = f1_score(y_test, y_pred, average='binary')

            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric('f1_score', test_score)
            mlflow.log_artifact('model', best_model)

            # Color code the results
            if test_score >= 0.65:
                result_color = "🟢"
            elif test_score >= 0.55:
                result_color = "🟡"
            else:
                result_color = "🔴"
            
            print(f"      {result_color} Test F1 Score: {test_score:.3f}")
            print(f"      🔧 Best params: {best_params}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = best_model
                best_params = best_params
                best_model_name = model_name
        
        print(f"\n🏆 OPTIMIZATION COMPLETE!")
        print(f"   🥇 Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"   📊 Test Accuracy: {best_score:.3f}")
        print(f"   🎯 Parameters: {best_params}")
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'test_accuracy': best_score,
            'params': best_params,
            'scaler': scaler,
            'feature_columns': feature_columns
        }
    
    def analyze_predictions_vs_results(self, best_model_info, test_df):
        """Analyze model predictions against actual results"""
        print(f"\n📊 Prediction Analysis")
        print("═" * 70)
        print(f"   🔍 Analyzing {len(test_df):,} test predictions...")
        
        model = best_model_info['model']
        scaler = best_model_info['scaler']
        feature_columns = best_model_info['feature_columns']
        
        X_test = test_df[feature_columns]
        X_test_scaled = scaler.transform(X_test)
        
        # Get predictions and probabilities
        predictions = model.predict(X_test_scaled)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test_scaled)[:, 1]  # Probability of winning
        else:
            probabilities = predictions.astype(float)
        
        # Add to test dataframe
        test_df = test_df.copy()
        test_df['predicted_winner'] = predictions
        test_df['win_probability'] = probabilities
        
        # Group by race to analyze race-level predictions
        race_analysis = []
        
        for race_id in test_df['race_id'].unique():
            race_data = test_df[test_df['race_id'] == race_id].copy()
            race_data = race_data.sort_values('win_probability', ascending=False)
            
            actual_winner = race_data[race_data['is_winner'] == 1]
            predicted_winner = race_data.iloc[0]  # Highest probability
            
            if len(actual_winner) > 0:
                actual_winner = actual_winner.iloc[0]
                
                race_analysis.append({
                    'race_id': race_id,
                    'predicted_winner': predicted_winner['dog_name'],
                    'predicted_prob': predicted_winner['win_probability'],
                    'actual_winner': actual_winner['dog_name'],
                    'prediction_correct': predicted_winner['dog_name'] == actual_winner['dog_name'],
                    'actual_winner_predicted_rank': list(race_data['dog_name']).index(actual_winner['dog_name']) + 1,
                    'field_size': len(race_data)
                })
        
        race_df = pd.DataFrame(race_analysis)
        
        # Calculate metrics
        race_accuracy = race_df['prediction_correct'].mean()
        avg_prob_correct = race_df[race_df['prediction_correct']]['predicted_prob'].mean()
        avg_prob_incorrect = race_df[~race_df['prediction_correct']]['predicted_prob'].mean()
        
        # Detailed analysis output
        print(f"\n   📈 PREDICTION PERFORMANCE:")
        print(f"      🎯 Race Accuracy: {race_accuracy:.1%}")
        print(f"      ✅ Correct Pred Confidence: {avg_prob_correct:.3f}")
        print(f"      ❌ Incorrect Pred Confidence: {avg_prob_incorrect:.3f}")
        print(f"      📊 Total Races Analyzed: {len(race_df):,}")
        print(f"      🏁 Avg Field Size: {race_df['field_size'].mean():.1f}")
        
        return {
            'race_accuracy': race_accuracy,
            'race_analysis': race_df,
            'individual_predictions': test_df
        }
    
    def generate_correlation_analysis(self, df, feature_columns):
        """Generate correlation analysis between features and winning"""
        print(f"\n🔗 Feature Correlation Analysis")
        print("═" * 70)
        
        # Calculate correlations with winning
        correlations = []
        for feature in feature_columns:
            corr = df[feature].corr(df['is_winner'])
            correlations.append({
                'feature': feature,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
        
        correlations_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        
        print("\n   🏆 TOP 10 FEATURES CORRELATED WITH WINNING:")
        print("   " + "-" * 50)
        for i, (_, row) in enumerate(correlations_df.head(10).iterrows(), 1):
            corr_str = f"{row['correlation']:+.3f}"
            if abs(row['correlation']) >= 0.3:
                icon = "🔥"
            elif abs(row['correlation']) >= 0.2:
                icon = "⭐"
            else:
                icon = "📊"
            print(f"   {i:2d}. {icon} {row['feature']:<25} | {corr_str}")
        
        return correlations_df
    
    def save_results(self, results):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results_file = self.results_dir / f"ml_backtesting_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            json_results = self.convert_numpy_types(results)
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_file}")
        
        # Save feature importance
        if self.feature_importance:
            importance_file = self.results_dir / f"feature_importance_{timestamp}.json"
            with open(importance_file, 'w') as f:
                json_importance = self.convert_numpy_types(self.feature_importance)
                json.dump(json_importance, f, indent=2, default=str)
            
            print(f"💾 Feature importance saved to {importance_file}")
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def run_comprehensive_backtest(self, months_back=12):
        """Run comprehensive backtesting analysis"""
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn not available. Cannot run ML backtesting.")
            return None
        
        print("🚀 COMPREHENSIVE ML BACKTESTING SYSTEM")
        print("═" * 70)
        print(f"   📅 Analysis Period: {months_back} months")
        print(f"   🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 70)
        
        overall_start_time = time.time()
        
        # Step 1: Load historical data
        print("\n📊 STEP 1: Loading Historical Data")
        print("-" * 50)
        step_start = time.time()
        historical_df = self.load_historical_race_data(months_back)
        if historical_df is None or len(historical_df) < 100:
            print("❌ Insufficient historical data for backtesting")
            return None
        step_time = time.time() - step_start
        print(f"   ✅ Step 1 completed in {step_time:.1f}s")
        
        # Step 2: Create enhanced features
        print("\n🔧 STEP 2: Feature Engineering")
        print("-" * 50)
        step_start = time.time()
        enhanced_df = self.create_enhanced_features(historical_df)
        if len(enhanced_df) < 50:
            print("❌ Insufficient enhanced data for backtesting")
            return None
        step_time = time.time() - step_start
        print(f"   ✅ Step 2 completed in {step_time:.1f}s")
        
        # Step 3: Prepare ML dataset
        print("\n⚙️  STEP 3: ML Dataset Preparation")
        print("-" * 50)
        step_start = time.time()
        ml_df, feature_columns = self.prepare_ml_dataset(enhanced_df)
        step_time = time.time() - step_start
        print(f"   ✅ Step 3 completed in {step_time:.1f}s")
        
        # Step 4: Cross-validation
        print("\n🔍 STEP 4: Cross-Validation")
        print("-" * 50)
        step_start = time.time()
        win_validation = self.time_series_split_validation(ml_df, feature_columns, 'is_winner')
        place_validation = self.time_series_split_validation(ml_df, feature_columns, 'is_placer')
        step_time = time.time() - step_start
        print(f"   ✅ Step 4 completed in {step_time:.1f}s")
        
        # Step 5: Model optimization
        print("\n🎯 STEP 5: Model Optimization")
        print("-" * 50)
        step_start = time.time()
        best_win_model = self.optimize_best_model(ml_df, feature_columns, 'is_winner')
        best_place_model = self.optimize_best_model(ml_df, feature_columns, 'is_placer')
        step_time = time.time() - step_start
        print(f"   ✅ Step 5 completed in {step_time:.1f}s")
        
        # Step 6: Final analysis
        print("\n📈 STEP 6: Performance Analysis")
        print("-" * 50)
        step_start = time.time()
        
        # Final test split for analysis
        split_point = int(0.8 * len(ml_df.sort_values('race_date')))
        test_df = ml_df.sort_values('race_date').iloc[split_point:]
        
        # Analyze predictions vs results
        win_analysis = self.analyze_predictions_vs_results(best_win_model, test_df)
        
        # Correlation analysis
        correlations = self.generate_correlation_analysis(ml_df, feature_columns)
        step_time = time.time() - step_start
        print(f"   ✅ Step 6 completed in {step_time:.1f}s")
        
        # Compile results
        results = {
            'backtest_summary': {
                'total_races': historical_df['race_id'].nunique(),
                'total_records': len(historical_df),
                'enhanced_records': len(enhanced_df),
                'ml_records': len(ml_df),
                'test_period_races': test_df['race_id'].nunique(),
                'date_range': {
                    'start': historical_df['race_date'].min(),
                    'end': historical_df['race_date'].max()
                }
            },
            'model_validation': {
                'win_prediction': win_validation,
                'place_prediction': place_validation
            },
            'best_models': {
                'win_model': {
                    'name': best_win_model['model_name'],
                    'accuracy': best_win_model['test_accuracy'],
                    'params': best_win_model['params']
                },
                'place_model': {
                    'name': best_place_model['model_name'],
                    'accuracy': best_place_model['test_accuracy'],
                    'params': best_place_model['params']
                }
            },
            'prediction_analysis': win_analysis,
            'feature_correlations': correlations.to_dict('records'),
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self.save_results(results)
        
        # Step 7: System updates
        print("\n🔄 STEP 7: System Updates")
        print("-" * 50)
        step_start = time.time()
        
        # AUTO-UPDATE: Run automated feature importance updater
        print("   🔄 Running automated feature importance update...")
        try:
            from automated_feature_importance_updater import AutomatedFeatureImportanceUpdater
            updater = AutomatedFeatureImportanceUpdater()
            update_success = updater.run_automated_update()
            if update_success:
                print("   ✅ Prediction system automatically updated!")
            else:
                print("   ⚠️ Automated update completed with warnings")
        except Exception as e:
            print(f"   ⚠️ Could not run automated update: {e}")
        
        step_time = time.time() - step_start
        print(f"   ✅ Step 7 completed in {step_time:.1f}s")
        
        # Final summary
        total_time = time.time() - overall_start_time
        print(f"\n🎉 BACKTESTING COMPLETE!")
        print("═" * 70)
        print(f"   ⏱️  Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"   🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 70)
        
        return results

def main():
    """Main execution function"""
    print("🎯 ML Backtesting and Training System")
    print("=" * 50)
    
    trainer = MLBacktestingTrainer()
    
    # Run comprehensive backtest
    results = trainer.run_comprehensive_backtest(months_back=6)  # Last 6 months
    
    if results:
        print("\n📊 SUMMARY RESULTS:")
        print("=" * 30)
        
        # Print key metrics
        backtest = results['backtest_summary']
        print(f"📈 Analyzed {backtest['total_races']} races")
        print(f"🏆 Win prediction accuracy: {results['best_models']['win_model']['accuracy']:.3f}")
        print(f"🥉 Place prediction accuracy: {results['best_models']['place_model']['accuracy']:.3f}")
        print(f"🎯 Race-level accuracy: {results['prediction_analysis']['race_accuracy']:.3f}")
        
        print(f"\n🏆 Best Models:")
        print(f"   Win: {results['best_models']['win_model']['name']}")
        print(f"   Place: {results['best_models']['place_model']['name']}")
        
        print(f"\n🔗 Top 5 Winning Correlations:")
        for i, corr in enumerate(results['feature_correlations'][:5]):
            print(f"   {i+1}. {corr['feature']}: {corr['correlation']:.3f}")

if __name__ == "__main__":
    main()
