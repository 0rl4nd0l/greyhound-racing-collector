#!/usr/bin/env python3
"""
Confidence-Weighted Ultimate Greyhound Racing Analysis
=====================================================

This system integrates data confidence scores with the ultimate analysis
to provide more reliable insights by weighting results based on data quality.

Features:
- Confidence-weighted statistics
- Reliability indicators for all results
- Filtered analysis by confidence level
- Uncertainty quantification
- Robust analytical framework
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
import warnings
from data_confidence_system import DataConfidenceScorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

class ConfidenceWeightedAnalysis:
    """Ultimate analysis system with confidence weighting"""
    
    def __init__(self, db_path="comprehensive_greyhound_data.db", min_confidence_grade='C'):
        self.db_path = db_path
        self.min_confidence_grade = min_confidence_grade
        self.confidence_scorer = DataConfidenceScorer(db_path)
        self.weighted_data = None
        self.confidence_stats = None
        
    def load_confidence_weighted_data(self):
        """Load data with confidence scores and weights"""
        try:
            print("🎯 Loading confidence-weighted data...")
            
            # Get confidence-weighted dataset
            self.weighted_data = self.confidence_scorer.create_weighted_dataset(
                min_grade=self.min_confidence_grade
            )
            
            if self.weighted_data is None:
                print("❌ Failed to load confidence-weighted data")
                return False
            
            print(f"✅ Loaded {len(self.weighted_data):,} confidence-weighted records")
            print(f"   Average confidence weight: {self.weighted_data['confidence_weight'].mean():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading confidence-weighted data: {e}")
            return False
    
    def analyze_confidence_distribution(self):
        """Analyze the distribution of confidence scores"""
        if self.weighted_data is None:
            return None
        
        try:
            confidence_dist = self.weighted_data['confidence_grade'].value_counts()
            
            analysis = {
                'grade_distribution': confidence_dist.to_dict(),
                'confidence_statistics': {
                    'mean_confidence': float(self.weighted_data['confidence_score'].mean()),
                    'median_confidence': float(self.weighted_data['confidence_score'].median()),
                    'std_confidence': float(self.weighted_data['confidence_score'].std()),
                    'min_confidence': float(self.weighted_data['confidence_score'].min()),
                    'max_confidence': float(self.weighted_data['confidence_score'].max())
                },
                'data_quality_metrics': {
                    'total_records': len(self.weighted_data),
                    'high_confidence_records': len(self.weighted_data[self.weighted_data['confidence_score'] >= 70]),
                    'low_confidence_records': len(self.weighted_data[self.weighted_data['confidence_score'] < 50]),
                    'avg_confidence_weight': float(self.weighted_data['confidence_weight'].mean())
                }
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing confidence distribution: {e}")
            return None
    
    def weighted_track_bias_analysis(self):
        """Track bias analysis with confidence weighting"""
        if self.weighted_data is None:
            return None
        
        try:
            # Calculate weighted win rates by box and venue
            bias_data = self.weighted_data.groupby(['venue', 'box_number']).apply(
                lambda x: pd.Series({
                    'weighted_win_rate': (
                        ((x['finish_position'] == '1st') * x['confidence_weight']).sum() / 
                        x['confidence_weight'].sum()
                    ),
                    'unweighted_win_rate': (x['finish_position'] == '1st').mean(),
                    'total_weight': x['confidence_weight'].sum(),
                    'sample_size': len(x),
                    'avg_confidence': x['confidence_score'].mean()
                })
            ).reset_index()
            
            # Filter for statistical significance
            significant_bias = bias_data[
                (bias_data['total_weight'] >= 5) & 
                (bias_data['sample_size'] >= 10)
            ]
            
            # Sort by weighted win rate
            top_biases = significant_bias.nlargest(10, 'weighted_win_rate')
            
            return {
                'confidence_weighted_biases': top_biases.to_dict('records'),
                'bias_reliability': {
                    'total_combinations': len(bias_data),
                    'statistically_significant': len(significant_bias),
                    'avg_confidence_of_significant': float(significant_bias['avg_confidence'].mean()) if len(significant_bias) > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"⚠️  Error in weighted track bias analysis: {e}")
            return None
    
    def weighted_trainer_performance(self):
        """Trainer performance analysis with confidence weighting"""
        if self.weighted_data is None:
            return None
        
        try:
            # Calculate weighted trainer statistics
            trainer_stats = self.weighted_data.groupby('trainer_name').apply(
                lambda x: pd.Series({
                    'weighted_win_rate': (
                        ((x['finish_position'] == '1st') * x['confidence_weight']).sum() / 
                        x['confidence_weight'].sum()
                    ),
                    'unweighted_win_rate': (x['finish_position'] == '1st').mean(),
                    'total_weight': x['confidence_weight'].sum(),
                    'sample_size': len(x),
                    'avg_confidence': x['confidence_score'].mean(),
                    'confidence_range': x['confidence_score'].max() - x['confidence_score'].min()
                })
            ).reset_index()
            
            # Filter for trainers with sufficient data
            qualified_trainers = trainer_stats[
                (trainer_stats['total_weight'] >= 3) & 
                (trainer_stats['sample_size'] >= 5)
            ]
            
            # Sort by weighted win rate
            top_trainers = qualified_trainers.nlargest(15, 'weighted_win_rate')
            
            return {
                'confidence_weighted_trainers': top_trainers.to_dict('records'),
                'trainer_reliability': {
                    'total_trainers': len(trainer_stats),
                    'qualified_trainers': len(qualified_trainers),
                    'avg_confidence_of_qualified': float(qualified_trainers['avg_confidence'].mean()) if len(qualified_trainers) > 0 else 0,
                    'most_reliable_trainer': qualified_trainers.loc[qualified_trainers['avg_confidence'].idxmax(), 'trainer_name'] if len(qualified_trainers) > 0 else None
                }
            }
            
        except Exception as e:
            print(f"⚠️  Error in weighted trainer analysis: {e}")
            return None
    
    def weighted_market_efficiency(self):
        """Market efficiency analysis with confidence weighting"""
        if self.weighted_data is None:
            return None
        
        try:
            # Filter for records with odds
            odds_data = self.weighted_data[self.weighted_data['odds_decimal'].notna()].copy()
            
            if odds_data.empty:
                return {'note': 'No odds data available for market efficiency analysis'}
            
            # Create odds brackets
            odds_data['odds_bracket'] = pd.cut(
                odds_data['odds_decimal'],
                bins=[0, 2, 4, 8, 15, float('inf')],
                labels=['Favorite', 'Second_Favorite', 'Mid_Price', 'Longshot', 'Very_Long']
            )
            
            # Calculate weighted performance by bracket
            bracket_performance = odds_data.groupby('odds_bracket', observed=True).apply(
                lambda x: pd.Series({
                    'weighted_win_rate': (
                        ((x['finish_position'] == '1st') * x['confidence_weight']).sum() / 
                        x['confidence_weight'].sum()
                    ),
                    'unweighted_win_rate': (x['finish_position'] == '1st').mean(),
                    'implied_probability': (1 / x['odds_decimal'] * x['confidence_weight']).sum() / x['confidence_weight'].sum(),
                    'total_weight': x['confidence_weight'].sum(),
                    'sample_size': len(x),
                    'avg_confidence': x['confidence_score'].mean()
                })
            ).reset_index()
            
            # Calculate value ratios
            bracket_performance['weighted_value_ratio'] = (
                bracket_performance['weighted_win_rate'] / bracket_performance['implied_probability']
            )
            bracket_performance['unweighted_value_ratio'] = (
                bracket_performance['unweighted_win_rate'] / bracket_performance['implied_probability']
            )
            
            return {
                'confidence_weighted_market_efficiency': bracket_performance.to_dict('records'),
                'market_reliability': {
                    'total_odds_records': len(odds_data),
                    'avg_confidence_with_odds': float(odds_data['confidence_score'].mean()),
                    'weighted_correlation': float(np.corrcoef(
                        odds_data['implied_probability'] * odds_data['confidence_weight'],
                        (odds_data['finish_position'] == '1st').astype(int) * odds_data['confidence_weight']
                    )[0, 1]) if len(odds_data) > 1 else 0
                }
            }
            
        except Exception as e:
            print(f"⚠️  Error in weighted market efficiency analysis: {e}")
            return None
    
    def build_confidence_weighted_model(self):
        """Build predictive model with confidence weighting"""
        if self.weighted_data is None:
            return None
        
        try:
            # Prepare features for modeling
            model_data = self.weighted_data.copy()
            
            # Select features
            numeric_features = ['box_number', 'confidence_score', 'confidence_weight']
            categorical_features = ['venue', 'trainer_name', 'confidence_grade']
            
            # Filter features that exist
            existing_numeric = [f for f in numeric_features if f in model_data.columns]
            existing_categorical = [f for f in categorical_features if f in model_data.columns]
            
            # Prepare data
            feature_data = model_data[existing_numeric + existing_categorical + ['finish_position']].copy()
            feature_data = feature_data.dropna(subset=['finish_position'])
            
            if len(feature_data) < 100:
                return {'note': 'Insufficient data for confidence-weighted modeling'}
            
            # Create target variable
            feature_data['is_winner'] = (feature_data['finish_position'] == '1st').astype(int)
            
            # Encode categorical variables
            for col in existing_categorical:
                if col in feature_data.columns:
                    le = LabelEncoder()
                    feature_data[f'{col}_encoded'] = le.fit_transform(feature_data[col].fillna('unknown'))
            
            # Final features
            encoded_features = [f'{col}_encoded' for col in existing_categorical if f'{col}_encoded' in feature_data.columns]
            final_features = existing_numeric + encoded_features
            
            # Prepare X and y
            X = feature_data[final_features].fillna(0)
            y = feature_data['is_winner']
            weights = feature_data['confidence_weight']
            
            # Split data
            X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
                X, y, weights, test_size=0.3, random_state=42
            )
            
            # Train confidence-weighted model
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train, sample_weight=weights_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate weighted accuracy
            weighted_accuracy = accuracy_score(y_test, y_pred, sample_weight=weights_test)
            unweighted_accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'model_performance': {
                    'weighted_accuracy': float(weighted_accuracy),
                    'unweighted_accuracy': float(unweighted_accuracy),
                    'improvement': float(weighted_accuracy - unweighted_accuracy),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'avg_training_weight': float(weights_train.mean()),
                    'avg_test_weight': float(weights_test.mean())
                },
                'feature_importance': [
                    {'feature': name, 'importance': float(imp)} 
                    for name, imp in zip(final_features, model.feature_importances_)
                ]
            }
            
        except Exception as e:
            print(f"⚠️  Error in confidence-weighted modeling: {e}")
            return None
    
    def generate_confidence_weighted_insights(self):
        """Generate comprehensive confidence-weighted insights"""
        print("🎯 GENERATING CONFIDENCE-WEIGHTED ANALYSIS")
        print("=" * 60)
        
        # Load confidence-weighted data
        if not self.load_confidence_weighted_data():
            return None
        
        # Generate all analyses
        insights = {
            'analysis_metadata': {
                'min_confidence_grade': self.min_confidence_grade,
                'total_records': len(self.weighted_data),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'confidence_distribution': self.analyze_confidence_distribution(),
            'weighted_track_bias': self.weighted_track_bias_analysis(),
            'weighted_trainer_performance': self.weighted_trainer_performance(),
            'weighted_market_efficiency': self.weighted_market_efficiency(),
            'confidence_weighted_model': self.build_confidence_weighted_model()
        }
        
        # Save insights
        with open('confidence_weighted_insights.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print("✅ Confidence-weighted insights generated!")
        return insights
    
    def print_confidence_weighted_summary(self, insights):
        """Print summary of confidence-weighted analysis"""
        print("\n📊 CONFIDENCE-WEIGHTED ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Analysis metadata
        metadata = insights['analysis_metadata']
        print(f"🎯 Analysis Parameters:")
        print(f"   Minimum Confidence Grade: {metadata['min_confidence_grade']}")
        print(f"   Records Analyzed: {metadata['total_records']:,}")
        
        # Confidence distribution
        if 'confidence_distribution' in insights:
            dist = insights['confidence_distribution']
            print(f"\n📈 Data Quality Metrics:")
            print(f"   Mean Confidence: {dist['confidence_statistics']['mean_confidence']:.1f}%")
            print(f"   Average Weight: {dist['data_quality_metrics']['avg_confidence_weight']:.3f}")
            print(f"   High Confidence Records: {dist['data_quality_metrics']['high_confidence_records']:,}")
        
        # Track bias
        if 'weighted_track_bias' in insights:
            bias = insights['weighted_track_bias']
            if 'confidence_weighted_biases' in bias:
                print(f"\n🏁 Confidence-Weighted Track Bias:")
                print(f"   Significant Combinations: {bias['bias_reliability']['statistically_significant']}")
                if bias['confidence_weighted_biases']:
                    top_bias = bias['confidence_weighted_biases'][0]
                    print(f"   Top Bias: Box {top_bias['box_number']} at {top_bias['venue']}")
                    print(f"             Weighted: {top_bias['weighted_win_rate']:.1%} | Unweighted: {top_bias['unweighted_win_rate']:.1%}")
        
        # Trainer performance
        if 'weighted_trainer_performance' in insights:
            trainers = insights['weighted_trainer_performance']
            if 'confidence_weighted_trainers' in trainers:
                print(f"\n👨‍💼 Confidence-Weighted Trainer Performance:")
                print(f"   Qualified Trainers: {trainers['trainer_reliability']['qualified_trainers']}")
                if trainers['confidence_weighted_trainers']:
                    top_trainer = trainers['confidence_weighted_trainers'][0]
                    print(f"   Top Trainer: {top_trainer['trainer_name']}")
                    print(f"               Weighted: {top_trainer['weighted_win_rate']:.1%} | Unweighted: {top_trainer['unweighted_win_rate']:.1%}")
        
        # Market efficiency
        if 'weighted_market_efficiency' in insights:
            market = insights['weighted_market_efficiency']
            if 'confidence_weighted_market_efficiency' in market:
                print(f"\n💰 Confidence-Weighted Market Efficiency:")
                print(f"   Records with Odds: {market['market_reliability']['total_odds_records']:,}")
                print(f"   Weighted Correlation: {market['market_reliability']['weighted_correlation']:.3f}")
        
        # Model performance
        if 'confidence_weighted_model' in insights:
            model = insights['confidence_weighted_model']
            if 'model_performance' in model:
                perf = model['model_performance']
                print(f"\n🤖 Confidence-Weighted Model:")
                print(f"   Weighted Accuracy: {perf['weighted_accuracy']:.2%}")
                print(f"   Unweighted Accuracy: {perf['unweighted_accuracy']:.2%}")
                print(f"   Improvement: {perf['improvement']:.2%}")
                print(f"   Training Samples: {perf['training_samples']:,}")

def main():
    """Main execution"""
    # Test different confidence levels
    confidence_levels = ['C', 'B', 'A']
    
    for level in confidence_levels:
        print(f"\n🎯 TESTING CONFIDENCE LEVEL: {level}")
        print("=" * 50)
        
        analyzer = ConfidenceWeightedAnalysis(min_confidence_grade=level)
        
        try:
            insights = analyzer.generate_confidence_weighted_insights()
            
            if insights:
                analyzer.print_confidence_weighted_summary(insights)
                
                # Save level-specific report
                with open(f'confidence_weighted_insights_{level}.json', 'w') as f:
                    json.dump(insights, f, indent=2, default=str)
                
                print(f"\n✅ Analysis complete for confidence level {level}")
            else:
                print(f"❌ No insights generated for level {level}")
                
        except Exception as e:
            print(f"❌ Error in confidence level {level}: {e}")
    
    print(f"\n🎯 CONFIDENCE-WEIGHTED ANALYSIS COMPLETE")
    print(f"📊 Multiple confidence levels tested and analyzed")

if __name__ == "__main__":
    main()
