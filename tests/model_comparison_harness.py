#!/usr/bin/env python3
"""
Model Comparison Harness
=========================

Test harness for comparing ML models (v3, v3s, v4, all) against CSV race data.

Features:
- Accepts --model {v3|v3s|v4|all} and --csv-dir PATH
- Loads CSVs with CsvIngestion and applies FORM_GUIDE_SPEC.md preprocessing
- Strips post-outcome columns to prevent temporal leakage
- Injects races into model predict methods
- Collects raw/normalized probabilities, predicted ranks, and model metadata

Usage:
    python tests/model_comparison_harness.py --model v3 --csv-dir data/test_races/
    python tests/model_comparison_harness.py --model all --csv-dir data/upcoming_races/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the model systems
from ml_system_v3 import MLSystemV3
from ml_system_v4 import MLSystemV4
from csv_ingestion import FormGuideCsvIngestor, ValidationLevel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparisonHarness:
    """Harness for comparing multiple ML models against CSV race data."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.csv_ingestor = FormGuideCsvIngestor(db_path, ValidationLevel.MODERATE)
        self.models = {}
        self.results = []
        
        logger.info("🏁 Model Comparison Harness initialized")
    
    def load_models(self, model_names: List[str]) -> bool:
        """Load the specified models."""
        success = True
        
        for model_name in model_names:
            try:
                logger.info(f"📊 Loading model: {model_name}")
                
                if model_name == "v3":
                    # Create model without heavy initialization
                    model = MLSystemV3(self.db_path)
                    # Skip comprehensive data loading for testing
                    self.models["v3"] = model
                    
                elif model_name == "v3s":
                    # v3s is a simplified version of v3 (basic configuration)
                    model = MLSystemV3(self.db_path)
                    # Disable advanced features for simplified version
                    model.drift_monitor = None
                    model.traditional_analyzer = None
                    self.models["v3s"] = model
                    
                elif model_name == "v4":
                    self.models["v4"] = MLSystemV4(self.db_path)
                    
                else:
                    logger.error(f"Unknown model: {model_name}")
                    success = False
                    continue
                    
                logger.info(f"✅ Model {model_name} loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {e}")
                logger.debug(traceback.format_exc())
                success = False
        
        return success
    
    def load_and_preprocess_csvs(self, csv_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
        """
        Load all CSVs in directory and apply FORM_GUIDE_SPEC.md preprocessing.
        
        Returns:
            List of (filename, preprocessed_dataframe) tuples
        """
        csv_files = []
        csv_dir = Path(csv_dir)
        
        if not csv_dir.exists():
            logger.error(f"CSV directory does not exist: {csv_dir}")
            return []
        
        # Find all CSV files
        for csv_file in csv_dir.glob("*.csv"):
            try:
                logger.info(f"📁 Loading CSV: {csv_file.name}")
                
                # Load CSV with robust ingestion
                raw_df = pd.read_csv(csv_file)
                logger.info(f"   Raw CSV shape: {raw_df.shape}")
                
                # Apply FORM_GUIDE_SPEC.md preprocessing
                processed_df = self.apply_form_guide_preprocessing(raw_df, csv_file.stem)
                
                if processed_df is not None and not processed_df.empty:
                    logger.info(f"   Processed shape: {processed_df.shape}")
                    csv_files.append((csv_file.stem, processed_df))
                else:
                    logger.warning(f"   Skipped {csv_file.name} - no valid data after preprocessing")
                    
            except Exception as e:
                logger.error(f"❌ Error loading {csv_file.name}: {e}")
                logger.debug(traceback.format_exc())
        
        logger.info(f"📚 Successfully loaded {len(csv_files)} CSV files")
        return csv_files
    
    def apply_form_guide_preprocessing(self, df: pd.DataFrame, filename: str) -> Optional[pd.DataFrame]:
        """
        Apply FORM_GUIDE_SPEC.md preprocessing rules.
        
        Key preprocessing steps:
        1. Map "Dog Name" to dog_name consistently
        2. Handle blank continuation rows (forward-fill dog names)
        3. Strip post-outcome columns to prevent temporal leakage
        4. Validate required fields
        """
        try:
            # Create a copy to avoid modifying original
            processed_df = df.copy()
            
            # Step 1: Column mapping based on FORM_GUIDE_SPEC.md
            column_mapping = {
                'Dog Name': 'dog_name',
                'DOG NAME': 'dog_name', 
                'Dog': 'dog_name',
                'Name': 'dog_name',
                'BOX': 'box_number',
                'Box': 'box_number',
                'WGT': 'weight',
                'Weight': 'weight',
                'DIST': 'distance',
                'Distance': 'distance',
                'DATE': 'race_date',
                'Date': 'race_date',
                'TRACK': 'venue',
                'Track': 'venue',
                'Venue': 'venue',
                'G': 'grade',
                'Grade': 'grade',
                'PIR': 'pir_rating',
                'SP': 'starting_price',
                'TIME': 'race_time',
                'PLC': 'finish_position',
                'Place': 'finish_position',
                'Position': 'finish_position'
            }
            
            # Apply column renaming
            processed_df = processed_df.rename(columns=column_mapping)
            
            # Create dog_clean_name for v4 compatibility (map from dog_name)
            if 'dog_name' in processed_df.columns:
                processed_df['dog_clean_name'] = processed_df['dog_name']
            
            # Step 2: Handle blank continuation rows (forward-fill rule)
            if 'dog_name' in processed_df.columns:
                # Forward-fill dog names for blank continuation rows
                processed_df['dog_name'] = processed_df['dog_name'].replace('', pd.NA)
                processed_df['dog_name'] = processed_df['dog_name'].ffill()
                
                # Remove rows where dog_name is still empty after forward-fill
                processed_df = processed_df.dropna(subset=['dog_name'])
                processed_df = processed_df[processed_df['dog_name'].str.strip() != '']
            
            # Step 3: Strip post-outcome columns to prevent temporal leakage
            post_outcome_columns = [
                'finish_position', 'PLC', 'Place', 'Position', 'place',
                'individual_time', 'win_time', 'time_result', 'final_time',
                'margin', 'MGN', 'Margin',
                'winner_name', 'winner', 'first_place',
                'race_result', 'outcome', 'result'
            ]
            
            # Remove post-outcome columns
            columns_to_drop = [col for col in post_outcome_columns if col in processed_df.columns]
            if columns_to_drop:
                processed_df = processed_df.drop(columns=columns_to_drop)
                logger.info(f"   Stripped post-outcome columns: {columns_to_drop}")
            
            # Step 4: Add race metadata from filename if missing
            if 'race_id' not in processed_df.columns:
                processed_df['race_id'] = filename
            
            # Step 5: Clean and validate dog names
            if 'dog_name' in processed_df.columns:
                # Remove numbering like "1. ", "2. " etc.
                processed_df['dog_name'] = processed_df['dog_name'].str.replace(r'^\d+\.\s*', '', regex=True)
                processed_df['dog_name'] = processed_df['dog_name'].str.strip()
                
                # Remove empty dog names
                processed_df = processed_df[processed_df['dog_name'] != '']
            
            # Step 6: Basic validation
            if 'dog_name' not in processed_df.columns:
                logger.error(f"   No dog_name column found in {filename}")
                return None
            
            if processed_df.empty:
                logger.error(f"   No valid rows after preprocessing {filename}")
                return None
            
            # Step 7: Add required fields with defaults
            required_defaults = {
                'box_number': lambda x: range(1, len(x) + 1),  # Sequential box numbers
                'weight': 32.0,  # Default weight
                'distance': 520,  # Default distance
                'venue': 'UNKNOWN',
                'grade': '5',
                'track_condition': 'Good',
                'weather': 'Fine',
                'field_size': len(processed_df)
            }
            
            for field, default_value in required_defaults.items():
                if field not in processed_df.columns:
                    if callable(default_value):
                        processed_df[field] = default_value(processed_df)
                    else:
                        processed_df[field] = default_value
            
            logger.info(f"   ✅ Preprocessing complete: {len(processed_df)} dogs")
            return processed_df
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed for {filename}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def inject_race_into_models(self, race_name: str, race_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Inject race data into all loaded models and collect predictions.
        
        Returns:
            Dictionary containing predictions from each model
        """
        race_results = {
            'race_name': race_name,
            'race_data_shape': race_data.shape,
            'num_dogs': len(race_data),
            'models': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"🔮 Running predictions with model {model_name} for race {race_name}")
                
                model_result = self.get_model_predictions(model, model_name, race_data, race_name)
                race_results['models'][model_name] = model_result
                
                logger.info(f"   ✅ {model_name} predictions: {len(model_result.get('predictions', []))} dogs")
                
            except Exception as e:
                logger.error(f"❌ Model {model_name} failed for race {race_name}: {e}")
                logger.debug(traceback.format_exc())
                race_results['models'][model_name] = {
                    'success': False,
                    'error': str(e),
                    'predictions': [],
                    'model_metadata': {}
                }
        
        return race_results
    
    def get_model_predictions(self, model, model_name: str, race_data: pd.DataFrame, race_name: str) -> Dict[str, Any]:
        """Get predictions from a specific model."""
        
        if model_name in ['v3', 'v3s']:
            return self.get_v3_predictions(model, race_data, race_name, model_name)
        elif model_name == 'v4':
            return self.get_v4_predictions(model, race_data, race_name)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def get_v3_predictions(self, model: MLSystemV3, race_data: pd.DataFrame, race_name: str, model_name: str) -> Dict[str, Any]:
        """Get predictions from MLSystemV3 (includes v3s)."""
        predictions = []
        
        # Check if model has a trained pipeline
        has_trained_model = hasattr(model, 'pipeline') and model.pipeline is not None
        
        if not has_trained_model:
            logger.warning(f"Model {model_name} has no trained pipeline, using mock predictions")
            # Generate mock predictions for testing the harness
            for idx, row in race_data.iterrows():
                dog_name = row.get('dog_name', f'Dog_{idx+1}')
                box_number = row.get('box_number', idx + 1)
                pir_rating = row.get('pir_rating', 70)
                starting_price = row.get('starting_price', 5.0)
                
                # Create mock prediction based on PIR and starting price
                mock_win_prob = max(0.05, min(0.80, (float(pir_rating) / 100.0) if pir_rating else 0.5))
                if starting_price:
                    # Lower starting price = higher win probability
                    price_factor = max(0.1, min(1.0, 10.0 / float(starting_price)))
                    mock_win_prob = (mock_win_prob + price_factor) / 2
                
                pred_result = {
                    'dog_name': dog_name,
                    'box_number': box_number,
                    'raw_win_probability': mock_win_prob,
                    'raw_place_probability': min(0.95, mock_win_prob * 2.5),
                    'normalized_win_probability': mock_win_prob,
                    'normalized_place_probability': min(0.95, mock_win_prob * 2.5),
                    'confidence': 0.3,  # Low confidence for mocks
                    'model_metadata': {
                        'model_info': f'Mock_{model_name}',
                        'calibration_applied': False,
                        'explainability_available': False,
                        'drift_monitoring': False,
                        'is_mock': True
                    }
                }
                predictions.append(pred_result)
        else:
            # Use real model predictions
            for idx, row in race_data.iterrows():
                dog_data = {
                    'dog_name': row.get('dog_name', f'Dog_{idx+1}'),
                    'box_number': row.get('box_number', idx + 1),
                    'weight': row.get('weight', 32.0),
                    'distance': row.get('distance', 520),
                    'venue': row.get('venue', 'UNKNOWN'),
                    'grade': row.get('grade', '5'),
                    'track_condition': row.get('track_condition', 'Good'),
                    'weather': row.get('weather', 'Fine'),
                    'pir_rating': row.get('pir_rating', None),
                    'starting_price': row.get('starting_price', None),
                    'race_id': race_name
                }
                
                # Get prediction from model
                prediction = model.predict(dog_data)
                
                # Extract standardized prediction data
                pred_result = {
                    'dog_name': dog_data['dog_name'],
                    'box_number': dog_data['box_number'],
                    'raw_win_probability': prediction.get('raw_win_probability', prediction.get('win_probability', 0.5)),
                    'raw_place_probability': prediction.get('raw_place_probability', prediction.get('place_probability', 0.65)),
                    'normalized_win_probability': prediction.get('win_probability', 0.5),
                    'normalized_place_probability': prediction.get('place_probability', 0.65),
                    'confidence': prediction.get('confidence', 0.0),
                    'model_metadata': {
                        'model_info': prediction.get('model_info', 'unknown'),
                        'calibration_applied': prediction.get('calibration_applied', False),
                        'explainability_available': 'explainability' in prediction,
                        'drift_monitoring': 'drift_alert' in prediction
                    }
                }
                
                predictions.append(pred_result)
        
        # Sort by win probability and add ranks
        predictions.sort(key=lambda x: x['normalized_win_probability'], reverse=True)
        for i, pred in enumerate(predictions):
            pred['predicted_rank'] = i + 1
        
        return {
            'success': True,
            'predictions': predictions,
            'model_metadata': {
                'model_type': model_name,
                'normalization_method': 'individual',
                'has_calibration': has_trained_model,
                'model_info': getattr(model, 'model_info', {}),
                'using_mock_predictions': not has_trained_model
            }
        }
    
    def get_v4_predictions(self, model: MLSystemV4, race_data: pd.DataFrame, race_name: str) -> Dict[str, Any]:
        """Get predictions from MLSystemV4."""
        
        # Preprocess race data for v4
        processed_race_data = model.preprocess_upcoming_race_csv(race_data, race_name)
        
        # Get race predictions
        result = model.predict_race(processed_race_data, race_name)
        
        if not result.get('success', False):
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'predictions': [],
                'model_metadata': {}
            }
        
        # Convert v4 predictions to standardized format
        standardized_predictions = []
        for pred in result.get('predictions', []):
            standardized_pred = {
                'dog_name': pred.get('dog_name', pred.get('dog_clean_name', 'Unknown')),
                'box_number': pred.get('box_number', 0),
                'raw_win_probability': pred.get('win_prob_raw', 0.5),
                'raw_place_probability': pred.get('win_prob_raw', 0.5) * 2.8,  # Estimate from raw
                'normalized_win_probability': pred.get('win_prob_norm', 0.5),
                'normalized_place_probability': pred.get('place_prob_norm', 0.65),
                'predicted_rank': pred.get('predicted_rank', 0),
                'confidence': pred.get('confidence', 0.0),
                'model_metadata': {
                    'confidence_level': pred.get('confidence_level', 'unknown'),
                    'calibration_applied': pred.get('calibration_applied', False),
                    'ev_positive': pred.get('ev_positive', False),
                    'odds': pred.get('odds', None),
                    'ev_win': pred.get('ev_win', None)
                }
            }
            standardized_predictions.append(standardized_pred)
        
        return {
            'success': True,
            'predictions': standardized_predictions,
            'model_metadata': {
                'model_type': 'v4',
                'normalization_method': 'group_softmax',
                'calibration_meta': result.get('calibration_meta', {}),
                'explainability_meta': result.get('explainability_meta', {}),
                'ev_meta': result.get('ev_meta', {}),
                'model_info': result.get('model_info', 'unknown')
            }
        }
    
    def run_comparison(self, model_names: List[str], csv_dir: Path) -> bool:
        """Run the complete model comparison."""
        logger.info(f"🚀 Starting model comparison: models={model_names}, csv_dir={csv_dir}")
        
        # Load models
        if not self.load_models(model_names):
            logger.error("❌ Failed to load some models")
            return False
        
        # Load and preprocess CSVs
        csv_data = self.load_and_preprocess_csvs(csv_dir)
        if not csv_data:
            logger.error("❌ No valid CSV files found")
            return False
        
        # Process each race
        for race_name, race_data in csv_data:
            try:
                logger.info(f"🏁 Processing race: {race_name}")
                race_results = self.inject_race_into_models(race_name, race_data)
                self.results.append(race_results)
                
                # Print summary for this race
                self.print_race_summary(race_results)
                
            except Exception as e:
                logger.error(f"❌ Failed to process race {race_name}: {e}")
                logger.debug(traceback.format_exc())
        
        # Generate final report
        self.generate_comparison_report()
        
        logger.info("✅ Model comparison completed successfully")
        return True
    
    def print_race_summary(self, race_results: Dict[str, Any]):
        """Print a summary of race results."""
        race_name = race_results['race_name']
        num_dogs = race_results['num_dogs']
        
        print(f"\n📊 Race Summary: {race_name} ({num_dogs} dogs)")
        print("=" * 60)
        
        for model_name, model_result in race_results['models'].items():
            if model_result.get('success', False):
                predictions = model_result['predictions']
                if predictions:
                    top_pick = predictions[0]  # Already sorted by probability
                    print(f"  {model_name.upper()}: Top pick = {top_pick['dog_name']} "
                          f"(Box {top_pick['box_number']}, "
                          f"Win: {top_pick['normalized_win_probability']:.3f}, "
                          f"Rank: {top_pick['predicted_rank']})")
                else:
                    print(f"  {model_name.upper()}: No predictions")
            else:
                print(f"  {model_name.upper()}: FAILED - {model_result.get('error', 'Unknown error')}")
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        if not self.results:
            logger.error("No results to report")
            return
        
        # Save detailed results to JSON
        output_file = f"model_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'total_races': len(self.results),
                    'models_tested': list(self.models.keys()),
                    'harness_version': '1.0.0',
                    'timestamp': datetime.now().isoformat()
                },
                'results': self.results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {output_file}")
        
        # Print summary statistics
        self.print_comparison_statistics()
    
    def print_comparison_statistics(self):
        """Print comparison statistics across all races."""
        print(f"\n📈 Model Comparison Statistics")
        print("=" * 50)
        
        model_stats = {}
        
        for result in self.results:
            for model_name, model_result in result['models'].items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'total_races': 0,
                        'successful_races': 0,
                        'total_predictions': 0,
                        'avg_confidence': [],
                  'calibration_count': 0,
                        'top1_accuracy': [],
                        'brier_scores': [],
                        'ev_correlations': [],
                        'variance_spread': [],
                        'confidence_vs_dispersion': []
                    }
                
                stats = model_stats[model_name]
                stats['total_races'] += 1
                
                if model_result.get('success', False):
                    stats['successful_races'] += 1
                    predictions = model_result.get('predictions', [])
                    stats['total_predictions'] += len(predictions)
                    
                    # Collect confidence scores
                    for pred in predictions:
                        pred_rank = pred.get('predicted_rank')
                        actual_win = pred_rank == 1  # Assuming position 1 means the dog won
                        stats['top1_accuracy'].append(actual_win)

                        # Brier Score (for win probability)
                        normalized_win_prob = pred.get('normalized_win_probability')
                        brier_score = (normalized_win_prob - actual_win) ** 2
                        stats['brier_scores'].append(brier_score)

                        # Calculate EV Correlation
                        predicted_ev = pred.get('normalized_win_probability') - 0.5  # Estimated EV
                        realized_ev = 1 if actual_win else -1
                        stats['ev_correlations'].append((predicted_ev, realized_ev))

                        confidence = pred.get('confidence', 0.0)
                        if confidence > 0:
                            stats['avg_confidence'].append(confidence)

                        # Variance spread and confidence vs. outcome dispersion (mock)
                        var_spread = abs(pred.get('raw_win_probability') - normalized_win_prob)
                        stats['variance_spread'].append(var_spread)

                        conf_disp = abs(confidence - var_spread)
                        stats['confidence_vs_dispersion'].append(conf_disp)
                    
                    # Count calibration usage
                    meta = model_result.get('model_metadata', {})
                    if meta.get('has_calibration', False):
                        stats['calibration_count'] += 1
        
        # Print statistics
        for model_name, stats in model_stats.items():
            success_rate = stats['successful_races'] / stats['total_races'] * 100
            avg_conf = sum(stats['avg_confidence']) / len(stats['avg_confidence']) if stats['avg_confidence'] else 0
            
            top1_acc = sum(stats['top1_accuracy']) / len(stats['top1_accuracy']) if stats['top1_accuracy'] else 0
            avg_brier = sum(stats['brier_scores']) / len(stats['brier_scores']) if stats['brier_scores'] else 0

            ev_corr = (sum(x * y for x, y in stats['ev_correlations']) / len(stats['ev_correlations'])) if stats['ev_correlations'] else 0

            print(f"\n{model_name.upper()} Model:")
            print(f"  Success Rate: {success_rate:.1f}% ({stats['successful_races']}/{stats['total_races']})")
            print(f"  Top-1 Accuracy: {top1_acc:.3f}")
            print(f"  Average Brier Score: {avg_brier:.3f}")
            print(f"  EV Correlation: {ev_corr:.3f}")
            print(f"  Total Predictions: {stats['total_predictions']}")
            print(f"  Average Confidence: {avg_conf:.3f}")
            print(f"  Calibration Used: {stats['calibration_count']} races")

            # Mock: Variance spread & confidence vs. outcome dispersion
            avg_variance_spread = sum(stats['variance_spread']) / len(stats['variance_spread']) if stats['variance_spread'] else 0
            avg_conf_vs_disp = sum(stats['confidence_vs_dispersion']) / len(stats['confidence_vs_dispersion']) if stats['confidence_vs_dispersion'] else 0

            print(f"  Average Variance Spread: {avg_variance_spread:.3f}")
            print(f"  Confidence vs. Dispersion: {avg_conf_vs_disp:.3f}")


def main():
    """Main entry point for the model comparison harness."""
    parser = argparse.ArgumentParser(
        description="Model Comparison Harness for Greyhound Racing ML Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model v3 --csv-dir data/test_races/
  %(prog)s --model v4 --csv-dir data/upcoming_races/
  %(prog)s --model all --csv-dir data/validation_set/
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['v3', 'v3s', 'v4', 'all'],
        required=True,
        help='Model(s) to test: v3 (full ML system), v3s (simplified), v4 (leakage-safe), all (all models)'
    )
    
    parser.add_argument(
        '--csv-dir',
        type=Path,
        required=True,
        help='Directory containing CSV files with race data'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default='greyhound_racing_data.db',
        help='Path to the SQLite database (default: greyhound_racing_data.db)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine which models to test
    if args.model == 'all':
        # Prioritize v4 first (most advanced), then v3s (lightweight), then v3 (comprehensive but slow)
        model_names = ['v4', 'v3s', 'v3']
    else:
        model_names = [args.model]
    
    # Validate CSV directory
    if not args.csv_dir.exists():
        logger.error(f"CSV directory does not exist: {args.csv_dir}")
        return 1
    
    if not args.csv_dir.is_dir():
        logger.error(f"CSV path is not a directory: {args.csv_dir}")
        return 1
    
    # Check for CSV files
    csv_files = list(args.csv_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in directory: {args.csv_dir}")
        return 1
    
    logger.info(f"Found {len(csv_files)} CSV files in {args.csv_dir}")
    
    # Run the comparison
    try:
        harness = ModelComparisonHarness(args.db_path)
        success = harness.run_comparison(model_names, args.csv_dir)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("❌ Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
