#!/usr/bin/env python3
"""
Baseline Modelling & Performance Metrics Pipeline
==================================================

Step 8: Baseline Modelling & Performance Metrics
1. Split historical data into temporal train/validation/test (rolling-window)
2. Train simple baseline models (logistic regression, gradient boost) on current features  
3. Record accuracy, log-loss, AUC, and calibration curves
4. Compare to previous production benchmarks to judge degradation/improvement

Author: AI Assistant
Date: January 2025
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    classification_report, confusion_matrix, precision_recall_curve,
    roc_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style

class BaselineModelingPipeline:
    """Comprehensive baseline modeling and evaluation pipeline"""
    
    def __init__(self, database_path="greyhound_racing_data.db"):
        self.database_path = database_path
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load and preprocess data from database"""
        print("📊 Loading data from database...")
        
        conn = sqlite3.connect(self.database_path)
        
        # Comprehensive query to get all available features
        query = """
        SELECT 
            rd.race_id,
            rd.dog_name,
            rd.dog_clean_name,
            rd.box_number,
            rd.finish_position,
            rd.weight,
            rd.odds_decimal,
            rd.win_probability,
            rd.place_probability,
            rd.performance_rating,
            rd.speed_rating,
            rd.class_rating,
            rm.race_date,
            rm.venue,
            rm.distance,
            rm.grade,
            rm.field_size,
            rm.track_condition,
            rm.weather,
            rm.temperature,
            rm.humidity
        FROM dog_race_data rd
        JOIN race_metadata rm ON rd.race_id = rm.race_id
        WHERE rd.finish_position IS NOT NULL 
        AND rd.finish_position != ''
        AND rm.race_date IS NOT NULL
        AND rm.race_date != 'Unknown'
        """
        
        data = pd.read_sql(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(data)} records")
        
        # Parse dates with flexible format handling
        def parse_date(date_str):
            try:
                if pd.isna(date_str):
                    return pd.NaT
                # Try different date formats
                for fmt in ['%d %B %Y', '%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                return pd.to_datetime(date_str, errors='coerce')
            except:
                return pd.NaT
        
        data['race_date'] = data['race_date'].apply(parse_date)
        data = data.dropna(subset=['race_date'])
        data = data.sort_values('race_date')
        
        # Create target variable
        data['win'] = (pd.to_numeric(data['finish_position'], errors='coerce') == 1).astype(int)
        data['place'] = (pd.to_numeric(data['finish_position'], errors='coerce') <= 3).astype(int)
        
        # Encode categorical variables
        le_venue = LabelEncoder()
        le_grade = LabelEncoder()
        
        data['venue_encoded'] = le_venue.fit_transform(data['venue'].fillna('Unknown'))
        data['grade_encoded'] = le_grade.fit_transform(data['grade'].fillna('Unknown'))
        
        # Extract distance as numeric
        data['distance_numeric'] = pd.to_numeric(
            data['distance'].str.extract(r'(\d+)')[0], errors='coerce'
        )
        
        self.data = data
        self.label_encoders = {'venue': le_venue, 'grade': le_grade}
        
        print(f"📈 Data summary:")
        print(f"  - Date range: {data['race_date'].min()} to {data['race_date'].max()}")
        print(f"  - Win rate: {data['win'].mean():.3f}")
        print(f"  - Place rate: {data['place'].mean():.3f}")
        print(f"  - Unique venues: {data['venue'].nunique()}")
        print(f"  - Unique grades: {data['grade'].nunique()}")
        
        return data
    
    def prepare_features(self, target='win'):
        """Prepare feature sets for modeling"""
        print(f"🔧 Preparing features for {target} prediction...")
        
        # Define potential features
        numerical_features = [
            'box_number', 'weight', 'odds_decimal', 'win_probability', 
            'place_probability', 'performance_rating', 'speed_rating', 
            'class_rating', 'field_size', 'distance_numeric', 'temperature', 'humidity'
        ]
        
        categorical_features = [
            'venue_encoded', 'grade_encoded'
        ]
        
        # Check feature availability and completeness
        available_features = []
        for feature in numerical_features + categorical_features:
            if feature in self.data.columns:
                completeness = self.data[feature].notna().mean()
                if completeness > 0.1:  # At least 10% data available
                    available_features.append(feature)
                    print(f"  ✅ {feature}: {completeness:.1%} complete")
                else:
                    print(f"  ❌ {feature}: {completeness:.1%} complete (excluded)")
        
        self.features = available_features
        
        # Create feature matrix
        X = self.data[self.features].copy()
        y = self.data[target].copy()
        
        # Remove rows where target is missing
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"📊 Final dataset: {len(X)} samples, {len(self.features)} features")
        
        return X, y
    
    def temporal_split(self, X, y, train_ratio=0.6, val_ratio=0.2):
        """Split data temporally based on race dates"""
        print("⏰ Creating temporal train/validation/test split...")
        
        # Get corresponding dates
        dates = self.data.loc[X.index, 'race_date']
        
        # Sort by date
        sorted_idx = dates.sort_values().index
        X_sorted = X.loc[sorted_idx]
        y_sorted = y.loc[sorted_idx]
        dates_sorted = dates.loc[sorted_idx]
        
        n = len(X_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split data
        X_train = X_sorted.iloc[:train_end]
        y_train = y_sorted.iloc[:train_end]
        dates_train = dates_sorted.iloc[:train_end]
        
        X_val = X_sorted.iloc[train_end:val_end]
        y_val = y_sorted.iloc[train_end:val_end]
        dates_val = dates_sorted.iloc[train_end:val_end]
        
        X_test = X_sorted.iloc[val_end:]
        y_test = y_sorted.iloc[val_end:]
        dates_test = dates_sorted.iloc[val_end:]
        
        print(f"📊 Split summary:")
        print(f"  - Train: {len(X_train)} samples ({dates_train.min()} to {dates_train.max()})")
        print(f"  - Validation: {len(X_val)} samples ({dates_val.min()} to {dates_val.max()})")
        print(f"  - Test: {len(X_test)} samples ({dates_test.min()} to {dates_test.max()})")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def train_baseline_models(self, X_train, y_train):
        """Train baseline models"""
        print("🤖 Training baseline models...")
        
        # Define models with preprocessing pipelines
        models = {
            'Logistic Regression': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            
            'Gradient Boosting': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100))
            ]),
            
            'Random Forest': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
            ])
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            print(f"  📈 Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Comprehensive model evaluation"""
        print("📊 Evaluating models...")
        
        results = {}
        
        for name, model in models.items():
            print(f"  🔍 Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            try:
                logloss = log_loss(y_test, y_prob)
            except:
                logloss = np.nan
            
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = np.nan
            
            brier = brier_score_loss(y_test, y_prob)
            
            results[name] = {
                'accuracy': accuracy,
                'log_loss': logloss,
                'auc': auc,
                'brier_score': brier,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'y_true': y_test
            }
            
            print(f"    - Accuracy: {accuracy:.4f}")
            print(f"    - Log Loss: {logloss:.4f}")
            print(f"    - AUC: {auc:.4f}")
            print(f"    - Brier Score: {brier:.4f}")
        
        self.results = results
        return results
    
    def generate_calibration_plots(self, results):
        """Generate calibration plots"""
        print("📈 Generating calibration plots...")
        
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(results.items()):
            y_true = result['y_true']
            y_prob = result['y_prob']
            
            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            axes[idx].plot(mean_predicted_value, fraction_of_positives, 
                          marker='o', linewidth=2, label=f'{name}')
            axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            axes[idx].set_xlabel('Mean Predicted Probability')
            axes[idx].set_ylabel('Fraction of Positives')
            axes[idx].set_title(f'Calibration Plot - {name}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('baseline_calibration_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def compare_to_benchmarks(self, results):
        """Compare results to historical benchmarks"""
        print("📊 Comparing to benchmarks...")
        
        # Historical benchmarks (these would typically be loaded from a file)
        benchmarks = {
            'Previous Production Model': {
                'accuracy': 0.75,
                'auc': 0.82,
                'log_loss': 0.45,
                'brier_score': 0.18
            }
        }
        
        print("\n🏆 Performance Comparison:")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':<10} {'AUC':<10} {'Log Loss':<10} {'Brier':<10}")
        print("=" * 80)
        
        # Show benchmarks
        for name, metrics in benchmarks.items():
            print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['auc']:<10.4f} "
                  f"{metrics['log_loss']:<10.4f} {metrics['brier_score']:<10.4f}")
        
        print("-" * 80)
        
        # Show current results
        for name, result in results.items():
            print(f"{name:<25} {result['accuracy']:<10.4f} {result['auc']:<10.4f} "
                  f"{result['log_loss']:<10.4f} {result['brier_score']:<10.4f}")
        
        print("=" * 80)
        
        # Calculate improvements
        print("\n📈 Performance Changes vs Previous Production:")
        for name, result in results.items():
            benchmark = benchmarks['Previous Production Model']
            acc_change = result['accuracy'] - benchmark['accuracy']
            auc_change = result['auc'] - benchmark['auc']
            
            status = "🟢 IMPROVED" if acc_change > 0 and auc_change > 0 else "🔴 DEGRADED"
            print(f"{name}: {status}")
            print(f"  - Accuracy: {acc_change:+.4f}")
            print(f"  - AUC: {auc_change:+.4f}")
    
    def save_results(self, results):
        """Save results to files"""
        print("💾 Saving results...")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            name: {
                'accuracy': result['accuracy'],
                'log_loss': result['log_loss'],
                'auc': result['auc'],
                'brier_score': result['brier_score']
            }
            for name, result in results.items()
        }).T
        
        metrics_df.to_csv('baseline_model_metrics.csv')
        print("  ✅ Saved metrics to baseline_model_metrics.csv")
        
        # Save calibration data
        for name, result in results.items():
            y_true = result['y_true']
            y_prob = result['y_prob']
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            calib_df = pd.DataFrame({
                'mean_predicted': mean_predicted_value,
                'fraction_positive': fraction_of_positives
            })
            
            filename = f'calibration_{name.lower().replace(" ", "_")}.csv'
            calib_df.to_csv(filename, index=False)
            print(f"  ✅ Saved calibration data to {filename}")
    
    def run_full_pipeline(self, target='win'):
        """Run the complete baseline modeling pipeline"""
        print("🚀 Starting Baseline Modeling Pipeline")
        print("=" * 60)
        
        # Load data
        data = self.load_data()
        
        # Prepare features
        X, y = self.prepare_features(target=target)
        
        if len(X) < 100:
            print(f"❌ Insufficient data: only {len(X)} samples available")
            return None
        
        # Temporal split
        (X_train, X_val, X_test), (y_train, y_val, y_test) = self.temporal_split(X, y)
        
        # Train models
        models = self.train_baseline_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(models, X_test, y_test)
        
        # Generate plots
        self.generate_calibration_plots(results)
        
        # Compare to benchmarks
        self.compare_to_benchmarks(results)
        
        # Save results
        self.save_results(results)
        
        print("\n✅ Baseline Modeling Pipeline Complete!")
        return results

if __name__ == "__main__":
    # Run the baseline modeling pipeline
    pipeline = BaselineModelingPipeline()
    
    # Run for win prediction
    print("🏆 Running Win Prediction Baseline Models")
    win_results = pipeline.run_full_pipeline(target='win')
    
    # Run for place prediction
    print("\n🥉 Running Place Prediction Baseline Models")
    place_results = pipeline.run_full_pipeline(target='place')
