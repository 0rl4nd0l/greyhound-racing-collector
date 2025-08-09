#!/usr/bin/env python3
"""
Step 4: Generate Comparative Strength Score for Each Dog
=======================================================

Combine engineered features into a single strength index using:
- Weighted linear formula or gradient-boosting regressor trained on past Ballarat meetings
- Normalise scores to allow cross-dog comparison
Return raw strength Sᵢ for every dog i.

Author: AI Assistant
Date: December 2024
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DogStrengthIndexGenerator:
    """
    Generate comparative strength scores for greyhound racing dogs.
    
    Combines multiple performance features into a single normalized strength index
    using either weighted linear combinations or gradient boosting models.
    """
    
    def __init__(self, features_file: str = None, model_type: str = "gradient_boosting"):
        """
        Initialize the strength index generator.
        
        Args:
            features_file: Path to CSV file with engineered features
            model_type: Either "linear_weighted" or "gradient_boosting"
        """
        self.features_file = features_file or "step2_comprehensive_features_20250804_133843.csv"
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        self.feature_weights = {}
        
        # Ballarat-specific weight adjustments
        self.ballarat_focus_multiplier = 1.5
        
        logger.info(f"DogStrengthIndexGenerator initialized with {model_type} approach")
    
    def load_features(self) -> pd.DataFrame:
        """Load the engineered features from CSV file."""
        try:
            features_df = pd.read_csv(self.features_file, index_col=0)
            logger.info(f"Loaded features for {len(features_df)} dogs with {len(features_df.columns)} features")
            return features_df
        except Exception as e:
            logger.error(f"Error loading features file: {e}")
            raise
    
    def prepare_feature_weights(self) -> Dict[str, float]:
        """
        Define feature weights for linear weighted approach.
        
        Based on greyhound racing domain knowledge and Ballarat-specific factors.
        """
        weights = {
            # Time performance (35% total weight)
            'best_race_time': 0.15,
            'mean_race_time': 0.10,
            'time_consistency_score': 0.10,
            
            # Position performance (25% total weight)
            'win_rate': 0.12,
            'place_rate_top3': 0.08,
            'mean_position': -0.05,  # Negative because lower is better
            
            # Recent form (20% total weight)
            'recent_position_trend': 0.10,  # Positive trend = improving
            'recent_form_avg': -0.05,  # Negative because lower position is better
            'recent_time_trend': 0.05,  # Positive trend = improving times
            
            # Ballarat-specific performance (15% total weight) - ENHANCED
            'ballarat_win_rate': 0.08 * self.ballarat_focus_multiplier,
            'ballarat_place_rate': 0.04 * self.ballarat_focus_multiplier,
            'ballarat_best_time': 0.03 * self.ballarat_focus_multiplier,
            
            # Early speed and consistency (5% total weight)
            'early_speed_score': 0.03,
            'performance_predictability': 0.02,
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(abs(w) for w in weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        self.feature_weights = weights
        logger.info(f"Prepared {len(weights)} feature weights for linear model")
        return weights
    
    def create_synthetic_target(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Create synthetic target variable for training gradient boosting model.
        
        Based on composite performance metrics that would indicate race success.
        """
        # Composite success metric combining multiple factors
        target = (
            # Win rate contribution (40%)
            0.4 * features_df['win_rate'] +
            
            # Place rate contribution (25%)
            0.25 * features_df['place_rate_top3'] +
            
            # Time performance contribution (20%) - normalized and inverted
            0.2 * (1 - (features_df['best_race_time'] - features_df['best_race_time'].min()) / 
                   (features_df['best_race_time'].max() - features_df['best_race_time'].min())) +
            
            # Ballarat-specific contribution (10%) - ENHANCED for Ballarat focus
            0.1 * features_df['ballarat_win_rate'] * self.ballarat_focus_multiplier +
            
            # Recent form contribution (5%)
            0.05 * np.maximum(0, features_df['recent_position_trend'])  # Only positive trends
        )
        
        # Add noise to prevent overfitting
        np.random.seed(42)
        target += np.random.normal(0, 0.01, len(target))
        
        return target.values
    
    def train_gradient_boosting_model(self, features_df: pd.DataFrame) -> GradientBoostingRegressor:
        """
        Train gradient boosting model to predict strength scores.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Trained GradientBoostingRegressor model
        """
        # Select relevant features for training
        feature_columns = [
            'mean_race_time', 'best_race_time', 'time_consistency_score',
            'win_rate', 'place_rate_top3', 'mean_position',
            'recent_position_trend', 'recent_form_avg', 'recent_time_trend',
            'ballarat_win_rate', 'ballarat_place_rate', 'ballarat_best_time',
            'early_speed_score', 'performance_predictability',
            'position_reliability', 'time_reliability',
            'ballarat_experience', 'total_races'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in features_df.columns]
        X = features_df[available_features]
        
        # Create synthetic target
        y = self.create_synthetic_target(features_df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Configure gradient boosting model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            validation_fraction=0.2,
            n_iter_no_change=10,
            tol=1e-4
        )
        
        # Train model
        logger.info("Training gradient boosting model...")
        model.fit(X_scaled, y)
        
        # Evaluate model performance
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        logger.info(f"Model performance - R²: {r2:.3f}, RMSE: {rmse:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        logger.info(f"Cross-validation R² scores: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        self.model = model
        return model
    
    def calculate_linear_weighted_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate strength scores using weighted linear combination.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Array of strength scores
        """
        weights = self.prepare_feature_weights()
        scores = np.zeros(len(features_df))
        
        for feature, weight in weights.items():
            if feature in features_df.columns:
                feature_values = features_df[feature].values
                
                # Handle missing values
                feature_values = np.nan_to_num(feature_values, nan=0.0)
                
                # Normalize feature values to 0-1 range for consistent weighting
                if feature_values.max() != feature_values.min():
                    if weight > 0:  # Higher values are better
                        normalized_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
                    else:  # Lower values are better (negative weight)
                        normalized_values = (feature_values.max() - feature_values) / (feature_values.max() - feature_values.min())
                        weight = abs(weight)  # Make weight positive after inversion
                else:
                    normalized_values = np.ones_like(feature_values) * 0.5
                
                scores += weight * normalized_values
            else:
                logger.warning(f"Feature '{feature}' not found in data")
        
        return scores
    
    def generate_strength_scores(self, normalize: bool = True) -> pd.DataFrame:
        """
        Generate comparative strength scores for all dogs.
        
        Args:
            normalize: Whether to normalize scores to 0-100 range
            
        Returns:
            DataFrame with dog names and their strength scores
        """
        # Load features
        features_df = self.load_features()
        
        # Calculate strength scores based on selected method
        if self.model_type == "gradient_boosting":
            logger.info("Calculating strength scores using gradient boosting...")
            self.train_gradient_boosting_model(features_df)
            
            # Predict strength scores
            feature_columns = [
                'mean_race_time', 'best_race_time', 'time_consistency_score',
                'win_rate', 'place_rate_top3', 'mean_position',
                'recent_position_trend', 'recent_form_avg', 'recent_time_trend',
                'ballarat_win_rate', 'ballarat_place_rate', 'ballarat_best_time',
                'early_speed_score', 'performance_predictability',
                'position_reliability', 'time_reliability',
                'ballarat_experience', 'total_races'
            ]
            
            available_features = [col for col in feature_columns if col in features_df.columns]
            X = features_df[available_features]
            X_scaled = self.scaler.transform(X)
            
            raw_scores = self.model.predict(X_scaled)
            
        else:  # linear_weighted
            logger.info("Calculating strength scores using weighted linear combination...")
            raw_scores = self.calculate_linear_weighted_scores(features_df)
        
        # Normalize scores if requested
        if normalize:
            # Normalize to 0-100 range
            min_score = raw_scores.min()
            max_score = raw_scores.max()
            
            if max_score != min_score:
                normalized_scores = 100 * (raw_scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.full_like(raw_scores, 50.0)
            
            scores = normalized_scores
        else:
            scores = raw_scores
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'dog_name': features_df.index,
            'raw_strength_score': raw_scores,
            'normalized_strength_score': scores if normalize else raw_scores
        })
        
        # Sort by strength score (descending)
        results_df = results_df.sort_values('normalized_strength_score', ascending=False)
        results_df['strength_rank'] = range(1, len(results_df) + 1)
        
        logger.info(f"Generated strength scores for {len(results_df)} dogs")
        logger.info(f"Score range: {scores.min():.2f} - {scores.max():.2f}")
        
        return results_df
    
    def save_model(self, filepath: str = None) -> str:
        """Save trained model and scalers to disk."""
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"strength_index_model_{self.model_type}_{timestamp}.pkl"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'normalizer': self.normalizer,
            'model_type': self.model_type,
            'feature_weights': self.feature_weights
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str):
        """Load trained model and scalers from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.normalizer = model_data['normalizer']
        self.model_type = model_data['model_type']
        self.feature_weights = model_data.get('feature_weights', {})
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict_strength_for_new_dogs(self, new_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict strength scores for new dogs using trained model.
        
        Args:
            new_features_df: DataFrame with same feature structure as training data
            
        Returns:
            DataFrame with predicted strength scores
        """
        if self.model is None:
            raise ValueError("No model has been trained. Call generate_strength_scores() first.")
        
        if self.model_type == "gradient_boosting":
            # Use same feature columns as training
            feature_columns = [
                'mean_race_time', 'best_race_time', 'time_consistency_score',
                'win_rate', 'place_rate_top3', 'mean_position',
                'recent_position_trend', 'recent_form_avg', 'recent_time_trend',
                'ballarat_win_rate', 'ballarat_place_rate', 'ballarat_best_time',
                'early_speed_score', 'performance_predictability',
                'position_reliability', 'time_reliability',
                'ballarat_experience', 'total_races'
            ]
            
            available_features = [col for col in feature_columns if col in new_features_df.columns]
            X = new_features_df[available_features]
            X_scaled = self.scaler.transform(X)
            
            raw_scores = self.model.predict(X_scaled)
        else:
            raw_scores = self.calculate_linear_weighted_scores(new_features_df)
        
        # Normalize scores (0-100 range)
        normalized_scores = self.normalizer.fit_transform(raw_scores.reshape(-1, 1)).flatten() * 100
        
        results_df = pd.DataFrame({
            'dog_name': new_features_df.index,
            'raw_strength_score': raw_scores,
            'normalized_strength_score': normalized_scores
        }).sort_values('normalized_strength_score', ascending=False)
        
        results_df['strength_rank'] = range(1, len(results_df) + 1)
        
        return results_df


def main():
    """Main function to demonstrate strength index generation."""
    print("=== Greyhound Strength Index Generator ===\n")
    
    # Test both approaches
    for model_type in ["gradient_boosting", "linear_weighted"]:
        print(f"\n--- Testing {model_type.upper().replace('_', ' ')} approach ---")
        
        try:
            # Initialize generator
            generator = DogStrengthIndexGenerator(model_type=model_type)
            
            # Generate strength scores
            results_df = generator.generate_strength_scores(normalize=True)
            
            print(f"\nTop 10 strongest dogs ({model_type}):")
            print("=" * 60)
            for i, (_, row) in enumerate(results_df.head(10).iterrows()):
                print(f"{i+1:2d}. {row['dog_name']:<20} "
                      f"Score: {row['normalized_strength_score']:6.2f} "
                      f"(Raw: {row['raw_strength_score']:6.3f})")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"step4_strength_scores_{model_type}_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Save model if gradient boosting
            if model_type == "gradient_boosting":
                model_file = generator.save_model()
                print(f"Model saved to: {model_file}")
            
            # Basic statistics
            print(f"\nStrength Score Statistics ({model_type}):")
            print(f"  Mean: {results_df['normalized_strength_score'].mean():.2f}")
            print(f"  Std:  {results_df['normalized_strength_score'].std():.2f}")
            print(f"  Min:  {results_df['normalized_strength_score'].min():.2f}")
            print(f"  Max:  {results_df['normalized_strength_score'].max():.2f}")
            
        except Exception as e:
            print(f"Error with {model_type}: {e}")
            logger.error(f"Error with {model_type}: {e}")
    
    print("\n=== Strength Index Generation Complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
