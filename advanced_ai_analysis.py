#!/usr/bin/env python3
"""
Advanced AI Analysis for Greyhound Racing Data
==============================================

This script performs advanced AI-powered analysis on processed greyhound racing data,
including machine learning predictions, pattern recognition, and statistical analysis.

Features:
- Winner prediction using Random Forest and other ML models
- Performance trend analysis
- Venue-specific insights
- Trainer and dog performance analytics
- Race competitiveness scoring
- Advanced statistical modeling

Author: AI Assistant
Date: July 11, 2025
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Statistical analysis
try:
    import scipy.stats as stats
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class AdvancedAIAnalyzer:
    """Advanced AI analyzer for greyhound racing data"""
    
    def __init__(self, db_path="./databases/comprehensive_greyhound_data.db"):
        self.db_path = db_path
        self.results_dir = "./advanced_results"
        self.models_dir = "./ai_models"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        print("🤖 Advanced AI Analyzer Initialized")
        print(f"✅ Scikit-learn Available: {SKLEARN_AVAILABLE}")
        print(f"✅ SciPy Available: {SCIPY_AVAILABLE}")
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for analysis"""
        print("📊 Loading data from database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load race and dog data with joins
        query = """
        SELECT 
            r.race_id,
            r.venue,
            r.race_number,
            r.race_date,
            r.grade,
            r.distance,
            r.track_condition,
            r.weather,
            r.temperature,
            r.humidity,
            r.wind_speed,
            r.field_size,
            d.dog_name,
            d.dog_clean_name,
            d.box_number,
            d.finish_position,
            d.weight,
            d.starting_price,
            d.individual_time,
            d.margin,
            d.sectional_1st,
            d.performance_rating,
            d.speed_rating,
            d.class_rating,
            d.win_probability,
            d.place_probability,
            d.trainer_name
        FROM race_metadata r
        LEFT JOIN dog_race_data d ON r.race_id = d.race_id
        WHERE d.dog_name IS NOT NULL
        ORDER BY r.race_date DESC, r.race_number, d.box_number
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("⚠️ No data found in database")
            return df
        
        print(f"✅ Loaded {len(df)} records from database")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning"""
        print("🔧 Preparing features for ML...")
        
        # Create target variable (winner = 1, others = 0)
        df['is_winner'] = (df['finish_position'] == '1').astype(int)
        
        # Convert categorical variables
        le_venue = LabelEncoder()
        le_grade = LabelEncoder()
        le_track_condition = LabelEncoder()
        le_weather = LabelEncoder()
        
        df['venue_encoded'] = le_venue.fit_transform(df['venue'].fillna('Unknown'))
        df['grade_encoded'] = le_grade.fit_transform(df['grade'].fillna('Unknown'))
        df['track_condition_encoded'] = le_track_condition.fit_transform(df['track_condition'].fillna('Good'))
        df['weather_encoded'] = le_weather.fit_transform(df['weather'].fillna('Clear'))
        
        # Store encoders for future use
        self.encoders = {
            'venue': le_venue,
            'grade': le_grade,
            'track_condition': le_track_condition,
            'weather': le_weather
        }
        
        # Create additional features
        df['distance_numeric'] = pd.to_numeric(df['distance'], errors='coerce')
        df['box_number_group'] = df['box_number'].apply(lambda x: 'inside' if x <= 4 else 'outside')
        df['weight_normalized'] = df['weight'] / df['weight'].mean()
        df['odds_inverse'] = 1 / (df['starting_price'] + 0.001)  # Avoid division by zero
        
        # Implement race splits as features to improve model accuracy
        df['avg_split'] = df['individual_time'] / df['distance_numeric'].replace(0, 1)
        
        # Time-based features
        df['race_date'] = pd.to_datetime(df['race_date'])
        df['month'] = df['race_date'].dt.month
        df['day_of_week'] = df['race_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Fill missing values
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'field_size', 'weight', 
                       'starting_price', 'performance_rating', 'speed_rating', 'class_rating',
                       'win_probability', 'place_probability', 'distance_numeric', 'avg_split']
        
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        print(f"✅ Features prepared. Dataset shape: {df.shape}")
        return df

        # Ensure all necessary columns are filled.
        df.fillna(0, inplace=True)

        print(f"Dataset shape post preparation: {df.shape}")
        return df
    
    def train_winner_prediction_model(self, df: pd.DataFrame) -> dict:
        """Train machine learning models to predict race winners"""
        print("🎯 Training winner prediction models...")
        
        if not SKLEARN_AVAILABLE:
            print("⚠️ Scikit-learn not available, skipping ML training")
            return {}
        
        # Select features for training
        feature_cols = [
            'venue_encoded', 'grade_encoded', 'track_condition_encoded', 'weather_encoded',
            'box_number', 'weight_normalized', 'odds_inverse', 'distance_numeric',
            'performance_rating', 'speed_rating', 'class_rating', 'win_probability',
            'temperature', 'humidity', 'wind_speed', 'field_size', 'month', 'day_of_week',
            'is_weekend', 'avg_split'
        ]

        # Filter data with complete features
        df_ml = df[feature_cols + ['is_winner']].dropna()
        
        if len(df_ml) < 100:
            print("⚠️ Insufficient data for ML training")
            return {}

        X = df_ml[feature_cols]
        y = df_ml['is_winner']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['winner_prediction'] = scaler

        # Train multiple models with ensemble
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        results = {}

        for name, model in models.items():
            print(f"🔄 Training {name}...")

            # Use scaled data for logistic regression, original for tree-based models
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Evaluate model
            accuracy = accuracy_score(y_test, y_pred)

            # Cross-validation
            if name == 'Logistic Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': self.get_feature_importance(model, feature_cols)
            }

            print(f"✅ {name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Store best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.models['winner_prediction'] = results[best_model_name]['model']

        print(f"🏆 Best model: {best_model_name}")
        return results


        # Implement ensemble predictions
        all_predictions = []
        for model_name, model_info in results.items():
            model = model_info['model']
            if model_name == 'Logistic Regression':
                all_predictions.append(model.predict_proba(X_test_scaled)[:, 1])
            else:
                all_predictions.append(model.predict_proba(X_test)[:, 1])

        # Average predictions
        ensemble_prediction = np.mean(all_predictions, axis=0)
        ensemble_accuracy = accuracy_score(y_test, (ensemble_prediction > 0.5).astype(int))
        print(f"✨ Ensemble model accuracy: {ensemble_accuracy:.3f}")

        # Save ensemble model as best
        self.models['winner_prediction'] = lambda x: np.mean([model.predict(x) for model, _ in models.values()], axis=0)
        return results


        
    
    def get_feature_importance(self, model, feature_cols):
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return {}
        
        importance_dict = dict(zip(feature_cols, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_venue_performance(self, df: pd.DataFrame) -> dict:
        """Analyze performance patterns by venue"""
        print("🏟️ Analyzing venue performance patterns...")
        
        venue_analysis = {}
        
        for venue in df['venue'].unique():
            venue_data = df[df['venue'] == venue]
            
            analysis = {
                'total_races': len(venue_data['race_id'].unique()),
                'total_entries': len(venue_data),
                'avg_field_size': venue_data['field_size'].mean(),
                'avg_winning_odds': venue_data[venue_data['finish_position'] == '1']['starting_price'].mean(),
                'track_bias': self.calculate_track_bias(venue_data),
                'weather_impact': self.analyze_weather_impact(venue_data),
                'distance_distribution': venue_data['distance'].value_counts().to_dict()
            }
            
            venue_analysis[venue] = analysis
        
        return venue_analysis
    
    def calculate_track_bias(self, venue_data: pd.DataFrame) -> dict:
        """Calculate track bias for inside vs outside boxes"""
        inside_boxes = venue_data[venue_data['box_number'] <= 4]
        outside_boxes = venue_data[venue_data['box_number'] > 4]
        
        inside_win_rate = len(inside_boxes[inside_boxes['finish_position'] == '1']) / len(inside_boxes) if len(inside_boxes) > 0 else 0
        outside_win_rate = len(outside_boxes[outside_boxes['finish_position'] == '1']) / len(outside_boxes) if len(outside_boxes) > 0 else 0
        
        return {
            'inside_win_rate': inside_win_rate,
            'outside_win_rate': outside_win_rate,
            'bias_factor': inside_win_rate - outside_win_rate
        }
    
    def analyze_weather_impact(self, venue_data: pd.DataFrame) -> dict:
        """Analyze impact of weather on race outcomes"""
        weather_impact = {}
        
        for weather in venue_data['weather'].dropna().unique():
            weather_data = venue_data[venue_data['weather'] == weather]
            
            if len(weather_data) > 5:  # Minimum threshold
                avg_time = pd.to_numeric(weather_data['individual_time'], errors='coerce').mean()
                weather_impact[weather] = {
                    'races': len(weather_data),
                    'avg_time': avg_time,
                    'avg_winning_odds': weather_data[weather_data['finish_position'] == '1']['starting_price'].mean()
                }
        
        return weather_impact
    
    def analyze_trainer_performance(self, df: pd.DataFrame) -> dict:
        """Analyze trainer performance statistics"""
        print("👨‍💼 Analyzing trainer performance...")
        
        trainer_stats = {}
        
        for trainer in df['trainer_name'].dropna().unique():
            trainer_data = df[df['trainer_name'] == trainer]
            
            if len(trainer_data) >= 10:  # Minimum threshold
                wins = len(trainer_data[trainer_data['finish_position'] == '1'])
                places = len(trainer_data[trainer_data['finish_position'].isin(['1', '2', '3'])])
                
                trainer_stats[trainer] = {
                    'total_starts': len(trainer_data),
                    'wins': wins,
                    'places': places,
                    'win_rate': wins / len(trainer_data),
                    'place_rate': places / len(trainer_data),
                    'avg_odds': trainer_data['starting_price'].mean(),
                    'avg_performance_rating': trainer_data['performance_rating'].mean()
                }
        
        # Sort by win rate
        trainer_stats = dict(sorted(trainer_stats.items(), 
                                  key=lambda x: x[1]['win_rate'], reverse=True))
        
        return trainer_stats
    
    def generate_race_predictions(self, df: pd.DataFrame) -> dict:
        """Generate predictions for recent races"""
        print("🔮 Generating race predictions...")
        
        if 'winner_prediction' not in self.models:
            print("⚠️ No trained model available for predictions")
            return {}
        
        # Get recent races (last 7 days)
        recent_date = datetime.now() - timedelta(days=7)
        recent_races = df[df['race_date'] >= recent_date]
        
        if recent_races.empty:
            print("⚠️ No recent races found for prediction")
            return {}
        
        predictions = {}
        
        for race_id in recent_races['race_id'].unique():
            race_data = recent_races[recent_races['race_id'] == race_id]
            
            # Prepare features
            feature_cols = [
                'venue_encoded', 'grade_encoded', 'track_condition_encoded', 'weather_encoded',
                'box_number', 'weight_normalized', 'odds_inverse', 'distance_numeric',
                'performance_rating', 'speed_rating', 'class_rating', 'win_probability',
                'temperature', 'humidity', 'wind_speed', 'field_size', 'month', 'day_of_week',
                'is_weekend'
            ]
            
            X = race_data[feature_cols].fillna(race_data[feature_cols].median())
            
            # Make predictions
            model = self.models['winner_prediction']
            win_probabilities = model.predict_proba(X)[:, 1]
            
            # Create prediction summary
            race_predictions = []
            for idx, (_, dog) in enumerate(race_data.iterrows()):
                race_predictions.append({
                    'dog_name': dog['dog_clean_name'],
                    'box_number': dog['box_number'],
                    'predicted_win_probability': win_probabilities[idx],
                    'actual_odds': dog['starting_price'],
                    'actual_result': dog['finish_position']
                })
            
            # Sort by predicted probability
            race_predictions.sort(key=lambda x: x['predicted_win_probability'], reverse=True)
            
            predictions[race_id] = {
                'race_info': {
                    'venue': race_data.iloc[0]['venue'],
                    'race_date': race_data.iloc[0]['race_date'],
                    'race_number': race_data.iloc[0]['race_number']
                },
                'predictions': race_predictions
            }
        
        return predictions
    
    def generate_comprehensive_report(self, df: pd.DataFrame, ml_results: dict, 
                                    venue_analysis: dict, trainer_stats: dict, 
                                    predictions: dict) -> str:
        """Generate comprehensive analysis report"""
        print("📋 Generating comprehensive analysis report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.results_dir, f"ai_analysis_report_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write(f"""# Advanced AI Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Records Analyzed**: {len(df):,}
- **Unique Races**: {df['race_id'].nunique():,}
- **Venues Covered**: {df['venue'].nunique()}
- **Date Range**: {df['race_date'].min()} to {df['race_date'].max()}

## Machine Learning Model Performance
""")
            
            if ml_results:
                f.write("### Model Comparison\n")
                for model_name, results in ml_results.items():
                    f.write(f"- **{model_name}**: {results['accuracy']:.3f} accuracy (CV: {results['cv_mean']:.3f} ± {results['cv_std']:.3f})\n")
                
                f.write("\n### Feature Importance (Top 10)\n")
                best_model = max(ml_results.keys(), key=lambda x: ml_results[x]['accuracy'])
                importance = ml_results[best_model]['feature_importance']
                for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                    f.write(f"{i+1}. **{feature}**: {imp:.3f}\n")
            
            f.write(f"\n## Venue Analysis\n")
            for venue, analysis in venue_analysis.items():
                f.write(f"### {venue}\n")
                f.write(f"- **Total Races**: {analysis['total_races']}\n")
                f.write(f"- **Average Field Size**: {analysis['avg_field_size']:.1f}\n")
                f.write(f"- **Average Winning Odds**: ${analysis['avg_winning_odds']:.2f}\n")
                f.write(f"- **Inside Box Win Rate**: {analysis['track_bias']['inside_win_rate']:.2%}\n")
                f.write(f"- **Outside Box Win Rate**: {analysis['track_bias']['outside_win_rate']:.2%}\n")
                f.write(f"- **Track Bias**: {analysis['track_bias']['bias_factor']:.3f}\n\n")
            
            f.write(f"\n## Top Trainers (by Win Rate)\n")
            for i, (trainer, stats) in enumerate(list(trainer_stats.items())[:10]):
                f.write(f"{i+1}. **{trainer}** - {stats['win_rate']:.2%} ({stats['wins']}/{stats['total_starts']})\n")
            
            if predictions:
                f.write(f"\n## Recent Race Predictions\n")
                for race_id, pred in list(predictions.items())[:5]:
                    f.write(f"### {pred['race_info']['venue']} Race {pred['race_info']['race_number']}\n")
                    f.write(f"Date: {pred['race_info']['race_date']}\n")
                    f.write("**Top 3 Predictions:**\n")
                    for i, dog in enumerate(pred['predictions'][:3]):
                        f.write(f"{i+1}. **{dog['dog_name']}** (Box {dog['box_number']}) - {dog['predicted_win_probability']:.2%}\n")
                    f.write("\n")
            
            f.write(f"\n## Statistical Insights\n")
            f.write(f"- **Most Successful Box**: {df[df['finish_position'] == '1']['box_number'].mode().iloc[0] if len(df[df['finish_position'] == '1']) > 0 else 'N/A'}\n")
            f.write(f"- **Average Winning Odds**: ${df[df['finish_position'] == '1']['starting_price'].mean():.2f}\n")
            f.write(f"- **Longest Shot Winner**: ${df[df['finish_position'] == '1']['starting_price'].max():.2f}\n")
            f.write(f"- **Shortest Odds Winner**: ${df[df['finish_position'] == '1']['starting_price'].min():.2f}\n")
            
            f.write(f"\n---\n*Report generated by Advanced AI Analyzer*")
        
        return report_path
    
    def save_analysis_to_database(self, ml_results: dict, venue_analysis: dict, 
                                trainer_stats: dict, predictions: dict):
        """Save analysis results to database"""
        print("💾 Saving analysis results to database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create analysis results table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_type TEXT,
                analysis_data TEXT,
                timestamp DATETIME,
                model_accuracy REAL,
                records_analyzed INTEGER
            )
        ''')
        
        # Save ML results
        if ml_results:
            best_model = max(ml_results.keys(), key=lambda x: ml_results[x]['accuracy'])
            cursor.execute('''
                INSERT INTO analysis_results (analysis_type, analysis_data, timestamp, model_accuracy, records_analyzed)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                'machine_learning',
                json.dumps(ml_results, default=str),
                datetime.now(),
                ml_results[best_model]['accuracy'],
                0  # Will be updated separately
            ))
        
        # Save venue analysis
        cursor.execute('''
            INSERT INTO analysis_results (analysis_type, analysis_data, timestamp, model_accuracy, records_analyzed)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            'venue_analysis',
            json.dumps(venue_analysis, default=str),
            datetime.now(),
            None,
            0
        ))
        
        # Save trainer stats
        cursor.execute('''
            INSERT INTO analysis_results (analysis_type, analysis_data, timestamp, model_accuracy, records_analyzed)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            'trainer_analysis',
            json.dumps(trainer_stats, default=str),
            datetime.now(),
            None,
            0
        ))
        
        conn.commit()
        conn.close()
        
        print("✅ Analysis results saved to database")
    
    def run_comprehensive_analysis(self):
        """Run complete AI analysis pipeline"""
        print("🚀 Starting comprehensive AI analysis...")
        
        # Load data
        df = self.load_data()
        if df.empty:
            print("❌ No data available for analysis")
            return
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Train ML models
        ml_results = self.train_winner_prediction_model(df)
        
        # Analyze venues
        venue_analysis = self.analyze_venue_performance(df)
        
        # Analyze trainers
        trainer_stats = self.analyze_trainer_performance(df)
        
        # Generate predictions
        predictions = self.generate_race_predictions(df)
        
        # Generate report
        report_path = self.generate_comprehensive_report(df, ml_results, venue_analysis, trainer_stats, predictions)
        
        # Save to database
        self.save_analysis_to_database(ml_results, venue_analysis, trainer_stats, predictions)
        
        print(f"✅ Comprehensive analysis complete!")
        print(f"📋 Report saved to: {report_path}")
        
        return {
            'report_path': report_path,
            'ml_results': ml_results,
            'venue_analysis': venue_analysis,
            'trainer_stats': trainer_stats,
            'predictions': predictions
        }


def main():
    """Main function to run advanced AI analysis"""
    print("🤖 ADVANCED AI ANALYSIS FOR GREYHOUND RACING")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = AdvancedAIAnalyzer()
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        if results:
            print("\n🎉 ANALYSIS COMPLETE!")
            print(f"📊 Report: {os.path.basename(results['report_path'])}")
            print(f"🤖 ML Models: {len(results['ml_results'])} trained")
            print(f"🏟️ Venues: {len(results['venue_analysis'])} analyzed")
            print(f"👨‍💼 Trainers: {len(results['trainer_stats'])} analyzed")
            print(f"🔮 Predictions: {len(results['predictions'])} races")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
