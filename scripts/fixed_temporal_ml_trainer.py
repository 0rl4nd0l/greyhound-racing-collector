#!/usr/bin/env python3
"""
Fixed Temporal ML Trainer for Greyhound Racing
==============================================

This version properly handles temporal data to prevent leakage:
- Historical features only use data from BEFORE each race
- Proper time-based train/test splits
- No future information leakage

Author: AI Assistant  
"""

import json
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")

class FixedTemporalMLTrainer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results_dir = Path("./ml_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.label_encoders = {}
        
        print("🎯 Fixed Temporal ML Trainer Initialized")
        print("   ✅ Proper temporal leakage protection enabled")

    def load_real_data(self, months_back=6):
        """Load real historical race data with proper date filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Use a shorter time window to ensure we have enough data for temporal features
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months_back * 30)
            
            print(f"📊 Loading temporal data from {start_date.date()} to {end_date.date()}")
            
            query = """
            SELECT 
                drd.race_id,
                drd.dog_clean_name,
                drd.finish_position,
                drd.box_number,
                rm.field_size,
                rm.distance,
                rm.venue,
                rm.race_date,
                rm.grade,
                rm.track_condition
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.finish_position != 'N/A'
            AND rm.race_date >= ?
            AND rm.race_date <= ?
            ORDER BY rm.race_date ASC, drd.race_id
            """
            
            df = pd.read_sql_query(
                query, conn, params=[start_date.isoformat(), end_date.isoformat()]
            )
            conn.close()
            
            # Clean and convert data types
            df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
            df['box_number'] = pd.to_numeric(df['box_number'], errors='coerce').fillna(1)
            df['field_size'] = pd.to_numeric(df['field_size'], errors='coerce').fillna(8)
            df['distance_numeric'] = pd.to_numeric(df['distance'], errors='coerce').fillna(500)
            df['race_date'] = pd.to_datetime(df['race_date'])
            
            # Remove invalid positions
            df = df.dropna(subset=['finish_position'])
            df = df[df['finish_position'] > 0]
            
            print(f"✅ Loaded {len(df):,} clean race records")
            print(f"   📈 Races: {df['race_id'].nunique():,}")
            print(f"   🐕 Dogs: {df['dog_clean_name'].nunique():,}")
            print(f"   📅 Date range: {df['race_date'].min().date()} to {df['race_date'].max().date()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def create_temporal_features(self, df):
        """Create features with proper temporal filtering - NO LEAKAGE"""
        print("\\n🔧 Creating temporal features (no leakage)...")
        
        # Sort by date to ensure proper temporal order
        df = df.sort_values(['race_date', 'race_id']).reset_index(drop=True)
        
        # Create target variables
        df['is_winner'] = (df['finish_position'] == 1).astype(int)
        df['is_placer'] = (df['finish_position'] <= 3).astype(int)
        
        enhanced_records = []
        total_records = len(df)
        
        print(f"   📊 Processing {total_records:,} records with temporal protection...")
        
        for idx, current_race in df.iterrows():
            if idx % 1000 == 0:
                progress = idx / total_records * 100
                print(f"\\r   Processing: {progress:.1f}% ({idx:,}/{total_records:,})", end="")
                sys.stdout.flush()
            
            dog_name = current_race['dog_clean_name']
            current_date = current_race['race_date']
            
            # Get ONLY historical data (races before current date) - NO LEAKAGE
            historical_data = df[
                (df['dog_clean_name'] == dog_name) & 
                (df['race_date'] < current_date)
            ].copy()
            
            # Create features based on available historical data
            features = self.calculate_historical_features(historical_data, current_race)
            
            # Add current race info
            features.update({
                'race_id': current_race['race_id'],
                'dog_name': dog_name,
                'race_date': current_date,
                'finish_position': current_race['finish_position'],
                'is_winner': current_race['is_winner'],
                'is_placer': current_race['is_placer'],
                'box_number': current_race['box_number'],
                'field_size': current_race['field_size'],
                'distance_numeric': current_race['distance_numeric'],
                'venue': current_race['venue'],
                'grade': current_race['grade']
            })
            
            enhanced_records.append(features)
        
        enhanced_df = pd.DataFrame(enhanced_records)
        print(f"\\n✅ Temporal feature engineering complete: {len(enhanced_df):,} records")
        
        return enhanced_df

    def calculate_historical_features(self, historical_data, current_race):
        """Calculate features using ONLY historical data (no leakage)"""
        
        # If no historical data, return defaults
        if len(historical_data) == 0:
            return {
                'races_count': 0,
                'avg_position': 5.0,
                'win_rate': 0.0,
                'place_rate': 0.0,
                'position_std': 2.0,
                'recent_form': 5.0,
                'venue_experience': 0,
                'venue_win_rate': 0.0,
                'distance_experience': 0,
                'days_since_last_race': 999,
                'inside_box_rate': 0.33,
                'form_trend': 0.0
            }
        
        # Basic historical stats
        positions = historical_data['finish_position'].tolist()
        races_count = len(positions)
        avg_position = np.mean(positions)
        position_std = np.std(positions) if len(positions) > 1 else 2.0
        win_rate = (historical_data['finish_position'] == 1).mean()
        place_rate = (historical_data['finish_position'] <= 3).mean()
        
        # Recent form (last 3-5 races)
        recent_positions = positions[-3:] if len(positions) >= 3 else positions
        recent_form = np.mean(recent_positions) if recent_positions else 5.0
        
        # Form trend (improving vs declining)
        if len(positions) >= 3:
            # Simple trend: compare first half vs second half of recent races
            mid_point = len(positions) // 2
            early_avg = np.mean(positions[:mid_point]) if mid_point > 0 else 5.0
            late_avg = np.mean(positions[mid_point:])
            form_trend = early_avg - late_avg  # Positive = improving (positions getting lower/better)
        else:
            form_trend = 0.0
        
        # Venue-specific performance
        current_venue = current_race['venue']
        venue_races = historical_data[historical_data['venue'] == current_venue]
        venue_experience = len(venue_races)
        venue_win_rate = (venue_races['finish_position'] == 1).mean() if len(venue_races) > 0 else 0.0
        
        # Distance experience
        current_distance = current_race['distance_numeric']
        distance_races = historical_data[
            abs(historical_data['distance_numeric'] - current_distance) <= 50
        ]
        distance_experience = len(distance_races)
        
        # Time since last race
        if len(historical_data) > 0:
            last_race_date = historical_data['race_date'].max()
            days_since_last_race = (current_race['race_date'] - last_race_date).days
        else:
            days_since_last_race = 999
        
        # Box preference
        box_data = historical_data['box_number'].dropna()
        inside_box_rate = (box_data <= 3).mean() if len(box_data) > 0 else 0.33
        
        return {
            'races_count': races_count,
            'avg_position': float(avg_position),
            'win_rate': float(win_rate),
            'place_rate': float(place_rate),
            'position_std': float(position_std),
            'recent_form': float(recent_form),
            'venue_experience': venue_experience,
            'venue_win_rate': float(venue_win_rate),
            'distance_experience': distance_experience,
            'days_since_last_race': days_since_last_race,
            'inside_box_rate': float(inside_box_rate),
            'form_trend': float(form_trend)
        }

    def prepare_training_data(self, df):
        """Prepare features with proper temporal ordering"""
        print("\\n🎯 Preparing temporal training data...")
        
        # Only include dogs with some historical data
        df = df[df['races_count'] >= 1].copy()
        
        print(f"   📊 Records with historical data: {len(df):,}")
        
        # Define feature columns
        feature_columns = [
            'box_number',
            'field_size', 
            'distance_numeric',
            'races_count',
            'avg_position',
            'win_rate',
            'place_rate',
            'position_std',
            'recent_form',
            'venue_experience',
            'venue_win_rate',
            'distance_experience',
            'days_since_last_race',
            'inside_box_rate',
            'form_trend'
        ]
        
        # Encode categorical features
        categorical_features = ['venue', 'grade']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                feature_columns.append(f'{feature}_encoded')
                self.label_encoders[feature] = le
        
        # Create feature matrix
        X = df[feature_columns].fillna(0)
        y_winner = df['is_winner']
        y_placer = df['is_placer']
        
        # Add temporal information for proper splitting
        df['race_date_ordinal'] = df['race_date'].map(pd.Timestamp.toordinal)
        
        print(f"   📊 Feature matrix: {X.shape}")
        print(f"   🎯 Winner target: {y_winner.sum():,}/{len(y_winner):,} ({y_winner.mean():.1%})")
        print(f"   🥉 Placer target: {y_placer.sum():,}/{len(y_placer):,} ({y_placer.mean():.1%})")
        
        return X, y_winner, y_placer, feature_columns, df['race_date_ordinal']

    def train_temporal_models(self, X, y_winner, y_placer, feature_columns, dates):
        """Train models with proper temporal train/test split"""
        print("\\n🚀 Training models with temporal split (no leakage)...")
        
        # Temporal split: use earlier races for training, later for testing
        split_date = np.percentile(dates, 70)  # 70% for training, 30% for testing
        
        train_mask = dates <= split_date
        test_mask = dates > split_date
        
        X_train = X[train_mask]
        X_test = X[test_mask] 
        y_win_train = y_winner[train_mask]
        y_win_test = y_winner[test_mask]
        y_place_train = y_placer[train_mask]
        y_place_test = y_placer[test_mask]
        
        print(f"   📊 Training set: {X_train.shape[0]:,} races (earlier dates)")
        print(f"   🧪 Test set: {X_test.shape[0]:,} races (later dates)")
        print(f"   📅 Split date: {pd.Timestamp.fromordinal(int(split_date)).date()}")
        
        results = {}
        
        # Train Winner Prediction Model
        print("\\n   🏆 Training Winner Prediction Model...")
        winner_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        start_time = time.time()
        winner_model.fit(X_train, y_win_train)
        train_time = time.time() - start_time
        
        # Evaluate with proper temporal testing
        y_win_pred = winner_model.predict(X_test)
        y_win_pred_proba = winner_model.predict_proba(X_test)[:, 1]
        
        win_accuracy = accuracy_score(y_win_test, y_win_pred)
        win_auc = roc_auc_score(y_win_test, y_win_pred_proba)
        
        results['winner_model'] = {
            'model': winner_model,
            'accuracy': win_accuracy,
            'auc': win_auc,
            'training_time': train_time
        }
        
        print(f"      ✅ Accuracy: {win_accuracy:.3f}")
        print(f"      ✅ AUC: {win_auc:.3f}")
        print(f"      ⏱️  Training time: {train_time:.2f}s")
        
        # Train Placer Prediction Model
        print("\\n   🥉 Training Placer (Top 3) Prediction Model...")
        placer_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        start_time = time.time()
        placer_model.fit(X_train, y_place_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_place_pred = placer_model.predict(X_test)
        y_place_pred_proba = placer_model.predict_proba(X_test)[:, 1]
        
        place_accuracy = accuracy_score(y_place_test, y_place_pred)
        place_auc = roc_auc_score(y_place_test, y_place_pred_proba)
        
        results['placer_model'] = {
            'model': placer_model,
            'accuracy': place_accuracy,
            'auc': place_auc,
            'training_time': train_time
        }
        
        print(f"      ✅ Accuracy: {place_accuracy:.3f}")
        print(f"      ✅ AUC: {place_auc:.3f}")
        print(f"      ⏱️  Training time: {train_time:.2f}s")
        
        # Feature Importance Analysis
        print("\\n   📊 Feature Importance Analysis...")
        winner_importance = dict(zip(feature_columns, winner_model.feature_importances_))
        placer_importance = dict(zip(feature_columns, placer_model.feature_importances_))
        
        print("      🏆 Top Winner Prediction Features:")
        for feature, importance in sorted(winner_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"        {feature}: {importance:.3f}")
            
        print("      🥉 Top Placer Prediction Features:")
        for feature, importance in sorted(placer_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"        {feature}: {importance:.3f}")
        
        results['feature_importance'] = {
            'winner': winner_importance,
            'placer': placer_importance
        }
        
        return results

    def save_fixed_models(self, results, feature_columns):
        """Save the properly trained models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        joblib.dump(results['winner_model']['model'], f"fixed_winner_model_{timestamp}.pkl")
        joblib.dump(results['placer_model']['model'], f"fixed_placer_model_{timestamp}.pkl")
        joblib.dump(self.label_encoders, f"fixed_encoders_{timestamp}.pkl")
        
        model_metrics = {
            'timestamp': timestamp,
            'model_type': 'temporal_fixed_models',
            'leakage_protected': True,
            'winner_model': {
                'accuracy': float(results['winner_model']['accuracy']),
                'auc': float(results['winner_model']['auc']),
                'training_time': float(results['winner_model']['training_time'])
            },
            'placer_model': {
                'accuracy': float(results['placer_model']['accuracy']),
                'auc': float(results['placer_model']['auc']),
                'training_time': float(results['placer_model']['training_time'])
            },
            'feature_columns': feature_columns,
            'feature_importance': {
                'winner': {k: float(v) for k, v in results['feature_importance']['winner'].items()},
                'placer': {k: float(v) for k, v in results['feature_importance']['placer'].items()}
            }
        }
        
        with open(f"fixed_model_results_{timestamp}.json", "w") as f:
            json.dump(model_metrics, f, indent=2)
        
        print(f"\\n💾 Fixed models saved:")
        print(f"   🏆 Winner model: fixed_winner_model_{timestamp}.pkl")
        print(f"   🥉 Placer model: fixed_placer_model_{timestamp}.pkl")
        print(f"   📊 Results: fixed_model_results_{timestamp}.json")
        
        return model_metrics

    def run_fixed_training(self):
        """Run the complete fixed training pipeline"""
        print("🎯 Starting Fixed Temporal ML Training Pipeline")
        print("   ✅ No data leakage - proper temporal protection")
        print("="*60)
        
        # Load data
        df = self.load_real_data()
        if df is None or len(df) == 0:
            print("❌ No data available for training")
            return None
        
        # Create temporal features (no leakage)
        df = self.create_temporal_features(df)
        
        # Prepare training data
        X, y_winner, y_placer, feature_columns, dates = self.prepare_training_data(df)
        
        # Train models with temporal split
        results = self.train_temporal_models(X, y_winner, y_placer, feature_columns, dates)
        
        # Save results
        metrics = self.save_fixed_models(results, feature_columns)
        
        print("\\n🎉 Fixed Temporal ML Training Complete!")
        print(f"   🏆 Winner Model AUC: {metrics['winner_model']['auc']:.3f}")
        print(f"   🥉 Placer Model AUC: {metrics['placer_model']['auc']:.3f}")
        print("   ✅ No data leakage - results are realistic!")
        
        return metrics

def main():
    """Main function"""
    print("🚀 Fixed Temporal ML Training for Greyhound Racing")
    print("   🔒 Proper leakage protection enabled")
    print("   📊 Using only historical data for each prediction")
    print()
    
    trainer = FixedTemporalMLTrainer()
    results = trainer.run_fixed_training()
    
    if results:
        print("\\n✅ SUCCESS: Realistic ML models trained without data leakage!")
        print("   These models should show realistic performance metrics.")
    else:
        print("\\n❌ Training failed - check data availability")

if __name__ == "__main__":
    main()
