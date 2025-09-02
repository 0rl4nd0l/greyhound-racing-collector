#!/usr/bin/env python3
"""
Real Data ML Trainer for Greyhound Racing
=========================================

This script trains ML models using the ACTUAL data from your database,
working with the columns that are actually available instead of requiring
synthetic data.

Available features:
- race_id, dog_clean_name, finish_position, box_number
- field_size, distance, venue, race_date, grade, track_condition

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")

class RealDataMLTrainer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results_dir = Path("./ml_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Label encoders for categorical features
        self.label_encoders = {}
        
        print("🎯 Real Data ML Trainer Initialized")
        print("   Using actual database schema with available columns")

    def load_real_data(self, months_back=12):
        """Load real historical race data from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months_back * 30)
            
            print(f"📊 Loading real data from {start_date.date()} to {end_date.date()}")
            
            # Query using AVAILABLE columns only
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
            ORDER BY rm.race_date ASC
            """
            
            df = pd.read_sql_query(
                query, conn, params=[start_date.isoformat(), end_date.isoformat()]
            )
            conn.close()
            
            print(f"✅ Loaded {len(df):,} real race records")
            print(f"   📈 Races: {df['race_id'].nunique():,}")
            print(f"   🐕 Dogs: {df['dog_clean_name'].nunique():,}")
            print(f"   🏟️ Venues: {df['venue'].nunique():,}")
            print(f"   📅 Date range: {df['race_date'].min()} to {df['race_date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            return None

    def create_real_features(self, df):
        """Create ML features from real available data"""
        print("\n🔧 Creating features from real data...")
        
        # Clean finish positions
        df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
        df = df.dropna(subset=['finish_position'])
        
        # Create target variables
        df['is_winner'] = (df['finish_position'] == 1).astype(int)
        df['is_placer'] = (df['finish_position'] <= 3).astype(int)
        
        # Feature engineering with available data
        print("   📊 Engineering features...")
        
        # Distance features
        df['distance_numeric'] = pd.to_numeric(df['distance'], errors='coerce').fillna(500)
        df['distance_category'] = pd.cut(df['distance_numeric'], 
                                        bins=[0, 350, 500, 700, 1000], 
                                        labels=['sprint', 'short', 'medium', 'long'])
        
        # Box number features  
        df['box_number'] = pd.to_numeric(df['box_number'], errors='coerce').fillna(1)
        df['inside_box'] = (df['box_number'] <= 3).astype(int)
        df['outside_box'] = (df['box_number'] >= 6).astype(int)
        
        # Field size features
        df['field_size'] = pd.to_numeric(df['field_size'], errors='coerce').fillna(8)
        df['small_field'] = (df['field_size'] <= 6).astype(int)
        df['large_field'] = (df['field_size'] >= 10).astype(int)
        
        # Historical performance by dog
        print("   📈 Calculating historical performance...")
        dog_stats = df.groupby('dog_clean_name').agg({
            'finish_position': ['mean', 'std', 'count'],
            'is_winner': 'mean',
            'is_placer': 'mean'
        }).round(3)
        
        # Flatten column names
        dog_stats.columns = ['_'.join(col) for col in dog_stats.columns]
        dog_stats = dog_stats.reset_index()
        
        # Merge back to main dataframe
        df = df.merge(dog_stats, on='dog_clean_name', how='left', suffixes=('', '_hist'))
        
        # Venue performance
        venue_stats = df.groupby(['dog_clean_name', 'venue']).agg({
            'is_winner': 'mean',
            'is_placer': 'mean',
            'finish_position': 'mean'
        }).round(3)
        venue_stats.columns = [f'venue_{col}' for col in venue_stats.columns]
        venue_stats = venue_stats.reset_index()
        
        df = df.merge(venue_stats, on=['dog_clean_name', 'venue'], how='left', suffixes=('', '_venue'))
        
        # Distance performance
        distance_stats = df.groupby(['dog_clean_name', 'distance_category']).agg({
            'is_winner': 'mean',
            'is_placer': 'mean'
        }).round(3)
        distance_stats.columns = [f'distance_{col}' for col in distance_stats.columns]
        distance_stats = distance_stats.reset_index()
        
        df = df.merge(distance_stats, on=['dog_clean_name', 'distance_category'], how='left', suffixes=('', '_dist'))
        
        # Fill NaN values with defaults
        df = df.fillna({
            'finish_position_std': 2.0,
            'venue_is_winner': 0.1,
            'venue_is_placer': 0.3,
            'venue_finish_position': 5.0,
            'distance_is_winner': 0.1,
            'distance_is_placer': 0.3
        })
        
        print(f"✅ Feature engineering complete: {len(df):,} records with enhanced features")
        return df

    def prepare_training_data(self, df):
        """Prepare features and targets for ML training"""
        print("\n🎯 Preparing training data...")
        
        # Define feature columns (using only what we have)
        feature_columns = [
            'box_number',
            'distance_numeric', 
            'field_size',
            'inside_box',
            'outside_box',
            'small_field',
            'large_field',
            'finish_position_mean',
            'finish_position_std', 
            'finish_position_count',
            'is_winner_mean',
            'is_placer_mean',
            'venue_is_winner',
            'venue_is_placer', 
            'venue_finish_position',
            'distance_is_winner',
            'distance_is_placer'
        ]
        
        # Encode categorical features
        categorical_features = ['venue', 'grade', 'distance_category']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                feature_columns.append(f'{feature}_encoded')
                self.label_encoders[feature] = le
        
        # Prepare feature matrix and targets
        X = df[feature_columns].fillna(0)
        y_winner = df['is_winner']
        y_placer = df['is_placer']
        
        print(f"   📊 Feature matrix: {X.shape}")
        print(f"   🎯 Winner target: {y_winner.sum():,}/{len(y_winner):,} ({y_winner.mean():.1%})")
        print(f"   🥉 Placer target: {y_placer.sum():,}/{len(y_placer):,} ({y_placer.mean():.1%})")
        
        return X, y_winner, y_placer, feature_columns

    def train_models(self, X, y_winner, y_placer, feature_columns):
        """Train ML models on real data"""
        print("\n🚀 Training ML models on real data...")
        
        results = {}
        
        # Split data
        X_train, X_test, y_win_train, y_win_test = train_test_split(
            X, y_winner, test_size=0.2, random_state=42, stratify=y_winner
        )
        _, _, y_place_train, y_place_test = train_test_split(
            X, y_placer, test_size=0.2, random_state=42, stratify=y_placer
        )
        
        print(f"   📊 Training set: {X_train.shape[0]:,} races")
        print(f"   🧪 Test set: {X_test.shape[0]:,} races")
        
        # Train Winner Prediction Model
        print("\n   🏆 Training Winner Prediction Model...")
        winner_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10, 
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        start_time = time.time()
        winner_model.fit(X_train, y_win_train)
        train_time = time.time() - start_time
        
        # Evaluate
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
        print("\n   🥉 Training Placer (Top 3) Prediction Model...")
        placer_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5, 
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
        print("\n   📊 Feature Importance Analysis...")
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

    def save_models_and_results(self, results, feature_columns):
        """Save trained models and results"""
        print("\n💾 Saving models and results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        joblib.dump(results['winner_model']['model'], f"real_winner_model_{timestamp}.pkl")
        joblib.dump(results['placer_model']['model'], f"real_placer_model_{timestamp}.pkl")
        joblib.dump(self.label_encoders, f"label_encoders_{timestamp}.pkl")
        
        # Save results
        model_metrics = {
            'timestamp': timestamp,
            'model_type': 'real_data_models',
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
        
        with open(f"real_model_results_{timestamp}.json", "w") as f:
            json.dump(model_metrics, f, indent=2)
        
        print(f"✅ Models saved:")
        print(f"   🏆 Winner model: real_winner_model_{timestamp}.pkl")
        print(f"   🥉 Placer model: real_placer_model_{timestamp}.pkl")
        print(f"   📊 Results: real_model_results_{timestamp}.json")
        
        return model_metrics

    def run_full_training(self):
        """Run the complete real data ML training pipeline"""
        print("🎯 Starting Real Data ML Training Pipeline")
        print("="*60)
        
        # Load real data
        df = self.load_real_data()
        if df is None or len(df) == 0:
            print("❌ No data available for training")
            return None
        
        # Create features
        df = self.create_real_features(df)
        
        # Prepare training data  
        X, y_winner, y_placer, feature_columns = self.prepare_training_data(df)
        
        # Train models
        results = self.train_models(X, y_winner, y_placer, feature_columns)
        
        # Save results
        metrics = self.save_models_and_results(results, feature_columns)
        
        print("\n🎉 Real Data ML Training Complete!")
        print(f"   🏆 Winner Model AUC: {metrics['winner_model']['auc']:.3f}")
        print(f"   🥉 Placer Model AUC: {metrics['placer_model']['auc']:.3f}")
        print(f"   📊 Training Records: {len(df):,}")
        
        return metrics

def main():
    """Main function to run real data ML training"""
    print("🚀 Real Data ML Training for Greyhound Racing")
    print("   Training with ACTUAL database records, not synthetic data!")
    print()
    
    trainer = RealDataMLTrainer()
    results = trainer.run_full_training()
    
    if results:
        print("\n✅ SUCCESS: Real ML models trained and saved!")
        print("   These models use your actual race data and are ready for predictions.")
    else:
        print("\n❌ Training failed - check data availability")

if __name__ == "__main__":
    main()
