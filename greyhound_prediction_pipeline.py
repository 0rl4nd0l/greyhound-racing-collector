#!/usr/bin/env python3
"""
Greyhound Racing Prediction Pipeline
===================================

A comprehensive machine learning pipeline for predicting greyhound race outcomes.
This script provides functions for:
- CSV parsing and data ingestion
- Feature engineering with temporal leakage protection  
- Model training and prediction scoring
- Probability calibration and ranking output

Usage:
    python greyhound_prediction_pipeline.py parse --input data.csv --output processed/
    python greyhound_prediction_pipeline.py predict --race race_data.csv --output predictions.csv
    python greyhound_prediction_pipeline.py full-pipeline --data-dir ./data --output-dir ./results

Author: AI Assistant
Date: 2025
"""

import argparse
import csv
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Local imports
try:
    from ml_system_v4 import MLSystemV4
    from prediction_pipeline_v4 import PredictionPipelineV4
    from enhanced_feature_engineering_v2 import EnhancedFeatureEngineer
    from step6_calibrate_validate import ProbabilityCalibrator
    from step7_ranked_output import RankedOutputGenerator
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure all required modules are in the Python path")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GreyhoundPredictionPipeline:
    """
    Main pipeline class that orchestrates the complete prediction workflow.
    """
    
    def __init__(self, db_path: str = "greyhound_racing_data.db", model_dir: str = "models/"):
        """
        Initialize the pipeline with database and model paths.
        
        Args:
            db_path: Path to the SQLite database
            model_dir: Directory containing trained models
        """
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline components
        self.ml_system = MLSystemV4(db_path=db_path)
        self.prediction_pipeline = PredictionPipelineV4(db_path=db_path)
        self.feature_engineer = EnhancedFeatureEngineer(db_path=db_path)
        self.calibrator = ProbabilityCalibrator()
        self.output_generator = RankedOutputGenerator()
        
        logger.info(f"Pipeline initialized with database: {db_path}")
    
    def parse_csv(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Parse CSV race data and extract structured information.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save parsed data
            
        Returns:
            Dictionary with parsing results and statistics
        """
        logger.info(f"Parsing CSV: {input_path}")
        
        try:
            # Read CSV data
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Basic validation
            required_columns = ['dog_name', 'race_date', 'venue']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract race information
            races = []
            for _, row in df.iterrows():
                race_info = {
                    'race_id': row.get('race_id', f"{row['venue']}_{row['race_date']}"),
                    'date': row['race_date'],
                    'venue': row['venue'],
                    'distance': row.get('distance', '500m'),
                    'track_condition': row.get('track_condition', 'Good'),
                    'dogs': []
                }
                
                # Add dog information
                dog_info = {
                    'name': row['dog_name'],
                    'box': row.get('box', 1),
                    'trainer': row.get('trainer', 'Unknown'),
                    'weight': row.get('weight', 30.0),
                    'odds': row.get('odds', 5.0)
                }
                race_info['dogs'].append(dog_info)
                races.append(race_info)
            
            # Save parsed data
            output_file = output_path / 'parsed_races.json'
            with open(output_file, 'w') as f:
                json.dump(races, f, indent=2)
            
            # Generate summary statistics
            stats = {
                'total_rows': len(df),
                'unique_races': len(set(f"{row['venue']}_{row['race_date']}" for _, row in df.iterrows())),
                'unique_dogs': df['dog_name'].nunique(),
                'date_range': [df['race_date'].min(), df['race_date'].max()],
                'venues': df['venue'].unique().tolist(),
                'output_file': str(output_file)
            }
            
            logger.info(f"Parsing complete. Stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise
    
    def feature_engineer(self, race_data: Dict, historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Engineer features for race prediction.
        
        Args:
            race_data: Dictionary containing race information
            historical_data: Optional historical performance data
            
        Returns:
            Dictionary of engineered features for each dog
        """
        logger.info("Engineering features for race prediction")
        
        try:
            dog_features = {}
            
            for dog in race_data.get('dogs', []):
                dog_name = dog['name']
                
                # Get historical stats (mock data if not available)
                if historical_data is not None:
                    dog_stats = self._extract_dog_stats(dog_name, historical_data)
                else:
                    dog_stats = self._get_default_dog_stats(dog)
                
                # Engineer features using the enhanced feature engineer
                features = self.feature_engineer.create_advanced_features(
                    dog_stats, 
                    race_context=race_data
                )
                
                # Add race-specific features
                features.update({
                    'box_position': dog.get('box', 1),
                    'weight': dog.get('weight', 30.0),
                    'trainer_hash': hash(dog.get('trainer', 'Unknown')) % 1000,
                    'odds': dog.get('odds', 5.0)
                })
                
                dog_features[dog_name] = features
            
            logger.info(f"Generated features for {len(dog_features)} dogs")
            return dog_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def score(self, features_dict: Dict[str, Dict], model_path: Optional[str] = None) -> Dict[str, float]:
        """
        Score dogs using trained ML model.
        
        Args:
            features_dict: Dictionary of features for each dog
            model_path: Optional path to specific model file
            
        Returns:
            Dictionary of raw prediction scores for each dog
        """
        logger.info("Scoring dogs with ML model")
        
        try:
            # Use ML system for scoring
            if model_path and os.path.exists(model_path):
                # Load specific model if provided
                self.ml_system.load_model(model_path)
            
            scores = {}
            
            # Convert features to format expected by ML system
            for dog_name, features in features_dict.items():
                try:
                    # Create feature vector
                    feature_vector = np.array(list(features.values())).reshape(1, -1)
                    
                    # Get prediction score
                    score = self.ml_system.predict_single_dog(feature_vector)
                    scores[dog_name] = float(score)
                    
                except Exception as e:
                    logger.warning(f"Error scoring {dog_name}: {e}")
                    scores[dog_name] = 0.5  # Default score
            
            logger.info(f"Scored {len(scores)} dogs")
            return scores
            
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            raise
    
    def probabilities(self, scores: Dict[str, float], calibrate: bool = True) -> Dict[str, float]:
        """
        Convert raw scores to win probabilities.
        
        Args:
            scores: Dictionary of raw prediction scores
            calibrate: Whether to apply probability calibration
            
        Returns:
            Dictionary of win probabilities for each dog
        """
        logger.info("Converting scores to probabilities")
        
        try:
            # Convert scores to probabilities using softmax
            score_values = np.array(list(scores.values()))
            
            # Apply temperature scaling for better calibration
            temperature = 1.5
            scaled_scores = score_values / temperature
            
            # Softmax transformation
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Create probability dictionary
            prob_dict = {}
            for i, dog_name in enumerate(scores.keys()):
                prob_dict[dog_name] = float(probabilities[i])
            
            # Apply calibration if requested
            if calibrate:
                try:
                    prob_dict = self.calibrator.calibrate_probabilities(prob_dict)
                except Exception as e:
                    logger.warning(f"Calibration failed: {e}, using uncalibrated probabilities")
            
            # Ensure probabilities sum to 1
            total_prob = sum(prob_dict.values())
            if total_prob > 0:
                prob_dict = {k: v/total_prob for k, v in prob_dict.items()}
            
            logger.info(f"Generated probabilities for {len(prob_dict)} dogs")
            return prob_dict
            
        except Exception as e:
            logger.error(f"Error converting to probabilities: {e}")
            raise
    
    def output(self, probabilities: Dict[str, float], race_data: Dict, output_path: str) -> str:
        """
        Generate ranked output with predictions.
        
        Args:
            probabilities: Dictionary of win probabilities
            race_data: Original race data
            output_path: Path to save output file
            
        Returns:
            Path to generated output file
        """
        logger.info("Generating ranked output")
        
        try:
            # Sort dogs by probability (highest first)
            ranked_dogs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            # Create output data
            output_data = []
            
            for rank, (dog_name, prob) in enumerate(ranked_dogs, 1):
                # Find dog details from race data
                dog_details = next(
                    (dog for dog in race_data.get('dogs', []) if dog['name'] == dog_name),
                    {'box': 0, 'trainer': 'Unknown', 'weight': 30.0, 'odds': 5.0}
                )
                
                output_row = {
                    'rank': rank,
                    'dog_name': dog_name,
                    'win_probability': round(prob, 4),
                    'win_percentage': round(prob * 100, 2),
                    'box': dog_details.get('box', 0),
                    'trainer': dog_details.get('trainer', 'Unknown'),
                    'weight': dog_details.get('weight', 30.0),
                    'market_odds': dog_details.get('odds', 5.0),
                    'implied_odds': round(1/prob if prob > 0 else 999, 2),
                    'value_ratio': round((dog_details.get('odds', 5.0) * prob), 3)
                }
                output_data.append(output_row)
            
            # Add race metadata
            race_info = {
                'race_id': race_data.get('race_id', 'unknown'),
                'date': race_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'venue': race_data.get('venue', 'Unknown'),
                'distance': race_data.get('distance', '500m'),
                'track_condition': race_data.get('track_condition', 'Good'),
                'field_size': len(ranked_dogs),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Save as CSV
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', newline='') as csvfile:
                if output_data:
                    fieldnames = output_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(output_data)
            
            # Save metadata as JSON
            metadata_path = output_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(race_info, f, indent=2)
            
            logger.info(f"Output saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating output: {e}")
            raise
    
    def run_full_pipeline(self, input_data: str, output_dir: str, **kwargs) -> Dict[str, Any]:
        """
        Run the complete prediction pipeline.
        
        Args:
            input_data: Path to input CSV or race data JSON
            output_dir: Directory to save all outputs
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with pipeline results and file paths
        """
        logger.info(f"Running full pipeline: {input_data} -> {output_dir}")
        
        try:
            results = {}
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Parse input data
            if input_data.endswith('.csv'):
                parse_results = self.parse_csv(input_data, str(output_path / 'parsed'))
                results['parsing'] = parse_results
                
                # Load parsed race data
                with open(parse_results['output_file'], 'r') as f:
                    race_data_list = json.load(f)
                
            elif input_data.endswith('.json'):
                with open(input_data, 'r') as f:
                    race_data_list = json.load(f)
                    if not isinstance(race_data_list, list):
                        race_data_list = [race_data_list]
            else:
                raise ValueError("Input must be CSV or JSON file")
            
            # Process each race
            race_results = []
            
            for i, race_data in enumerate(race_data_list):
                logger.info(f"Processing race {i+1}/{len(race_data_list)}")
                
                try:
                    # Step 2: Feature engineering
                    features = self.feature_engineer(race_data)
                    
                    # Step 3: Scoring
                    scores = self.score(features)
                    
                    # Step 4: Convert to probabilities
                    probabilities = self.probabilities(scores, calibrate=kwargs.get('calibrate', True))
                    
                    # Step 5: Generate output
                    race_output_path = output_path / f"race_{i+1}_predictions.csv"
                    output_file = self.output(probabilities, race_data, str(race_output_path))
                    
                    race_result = {
                        'race_id': race_data.get('race_id', f'race_{i+1}'),
                        'output_file': output_file,
                        'top_pick': max(probabilities.items(), key=lambda x: x[1]),
                        'field_size': len(probabilities)
                    }
                    race_results.append(race_result)
                    
                except Exception as e:
                    logger.error(f"Error processing race {i+1}: {e}")
                    race_results.append({
                        'race_id': race_data.get('race_id', f'race_{i+1}'),
                        'error': str(e)
                    })
            
            results['races'] = race_results
            results['total_races'] = len(race_data_list)
            results['successful_races'] = len([r for r in race_results if 'error' not in r])
            
            # Save summary results
            summary_path = output_path / 'pipeline_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Pipeline complete. Results: {results['successful_races']}/{results['total_races']} races processed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _extract_dog_stats(self, dog_name: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract historical statistics for a dog."""
        dog_data = historical_data[historical_data['dog_name'] == dog_name]
        
        if len(dog_data) == 0:
            return self._get_default_dog_stats({'name': dog_name})
        
        # Calculate statistics
        stats = {
            'races_count': len(dog_data),
            'win_rate': (dog_data['finish_position'] == 1).mean(),
            'place_rate': (dog_data['finish_position'] <= 3).mean(),
            'avg_position': dog_data['finish_position'].mean(),
            'recent_form': dog_data['finish_position'].tail(5).tolist(),
            'best_time': dog_data['race_time'].min() if 'race_time' in dog_data.columns else 30.0,
            'avg_speed_rating': dog_data.get('speed_rating', pd.Series([50])).mean()
        }
        
        return stats
    
    def _get_default_dog_stats(self, dog_info: Dict) -> Dict[str, Any]:
        """Generate default statistics for dogs with no historical data."""
        return {
            'races_count': 0,
            'win_rate': 0.1,
            'place_rate': 0.3,
            'avg_position': 4.0,
            'recent_form': [4, 4, 4],
            'best_time': 30.0,
            'avg_speed_rating': 50.0,
            'weight': dog_info.get('weight', 30.0),
            'trainer': dog_info.get('trainer', 'Unknown')
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Greyhound Racing Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Parse CSV data
    python greyhound_prediction_pipeline.py parse --input races.csv --output ./parsed/
    
    # Run predictions on single race
    python greyhound_prediction_pipeline.py predict --race race.json --output predictions.csv
    
    # Run full pipeline
    python greyhound_prediction_pipeline.py full-pipeline --input races.csv --output ./results/
    
    # Train new model (if training data available)
    python greyhound_prediction_pipeline.py train --data training_data.csv --output models/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse CSV race data')
    parse_parser.add_argument('--input', required=True, help='Input CSV file')
    parse_parser.add_argument('--output', required=True, help='Output directory')
    
    # Predict command  
    predict_parser = subparsers.add_parser('predict', help='Predict race outcomes')
    predict_parser.add_argument('--race', required=True, help='Race data file (JSON)')
    predict_parser.add_argument('--output', required=True, help='Output CSV file')
    predict_parser.add_argument('--no-calibrate', action='store_true', help='Skip probability calibration')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full-pipeline', help='Run complete pipeline')
    full_parser.add_argument('--input', required=True, help='Input data file (CSV or JSON)')
    full_parser.add_argument('--output', required=True, help='Output directory')
    full_parser.add_argument('--no-calibrate', action='store_true', help='Skip probability calibration')
    full_parser.add_argument('--db-path', default='greyhound_racing_data.db', help='Database path')
    
    # Train command (placeholder)
    train_parser = subparsers.add_parser('train', help='Train new prediction model')
    train_parser.add_argument('--data', required=True, help='Training data CSV')
    train_parser.add_argument('--output', required=True, help='Output model directory')
    train_parser.add_argument('--db-path', default='greyhound_racing_data.db', help='Database path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize pipeline
        pipeline = GreyhoundPredictionPipeline(
            db_path=getattr(args, 'db_path', 'greyhound_racing_data.db')
        )
        
        if args.command == 'parse':
            results = pipeline.parse_csv(args.input, args.output)
            print(f"Parsing complete: {results}")
            
        elif args.command == 'predict':
            # Load race data
            with open(args.race, 'r') as f:
                race_data = json.load(f)
            
            # Run prediction steps
            features = pipeline.feature_engineer(race_data)
            scores = pipeline.score(features)
            probabilities = pipeline.probabilities(scores, calibrate=not args.no_calibrate)
            output_file = pipeline.output(probabilities, race_data, args.output)
            
            print(f"Predictions saved to: {output_file}")
            
        elif args.command == 'full-pipeline':
            results = pipeline.run_full_pipeline(
                args.input, 
                args.output,
                calibrate=not args.no_calibrate
            )
            print(f"Pipeline complete: {results['successful_races']}/{results['total_races']} races processed")
            
        elif args.command == 'train':
            print("Training functionality not yet implemented")
            print("Please use the ML system training scripts directly")
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
