#!/usr/bin/env python3
"""
Upcoming Race Predictor Wrapper
==============================
Simple wrapper script that uses PredictionPipelineV4 to predict all CSV files
in the upcoming races directory and outputs results in the format expected by the API.
"""

import os
import sys
import glob
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to predict all upcoming race CSV files."""
    try:
        # Import the prediction pipeline
        from prediction_pipeline_v4 import PredictionPipelineV4
        
        # Get upcoming races directory from environment or use default
        upcoming_dir = os.getenv('UPCOMING_RACES_DIR', './upcoming_races_temp')
        
        if not os.path.exists(upcoming_dir):
            print(f"❌ Upcoming races directory not found: {upcoming_dir}")
            return 1
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(upcoming_dir, "*.csv"))
        csv_files = [f for f in csv_files if not f.endswith('README.csv')]
        
        if not csv_files:
            print(f"❌ No CSV files found in {upcoming_dir}")
            return 1
        
        print(f"🔍 Found {len(csv_files)} CSV files to process")
        
        # Initialize prediction pipeline with correct database path
        db_path = os.path.join(os.getcwd(), 'greyhound_racing_data.db')
        print(f"Using database: {db_path}")
        
        # Ensure GREYHOUND_DB_PATH is set to override any auto-detection
        os.environ['GREYHOUND_DB_PATH'] = db_path
        
        pipeline = PredictionPipelineV4(db_path)
        
        successful_predictions = 0
        failed_predictions = 0
        
        # Process each CSV file
        for csv_file in csv_files:
            try:
                filename = os.path.basename(csv_file)
                print(f"🎯 Predicting: {filename}")
                
                # Run prediction
                result = pipeline.predict_race_file(csv_file)
                
                if result and result.get('success'):
                    # Print results in expected format
                    predictions = result.get('predictions', [])
                    if predictions and len(predictions) >= 3:
                        top_3 = predictions[:3]
                        top_3_str = ', '.join([f"{p.get('dog_name', 'Unknown')} ({p.get('win_probability', 0):.2f})" for p in top_3])
                        print(f"🏆 Top 3 picks: {top_3_str}")
                        successful_predictions += 1
                    else:
                        print(f"⚠️ {filename}: Prediction succeeded but insufficient results")
                        failed_predictions += 1
                        
                    # Save prediction result to predictions directory
                    predictions_dir = './predictions'
                    os.makedirs(predictions_dir, exist_ok=True)
                    
                    prediction_filename = filename.replace('.csv', '_prediction.json')
                    prediction_path = os.path.join(predictions_dir, prediction_filename)
                    
                    with open(prediction_path, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                    print(f"❌ {filename}: Prediction failed - {error_msg}")
                    failed_predictions += 1
                    
            except Exception as e:
                print(f"❌ {filename}: Exception during prediction - {str(e)}")
                failed_predictions += 1
                logger.exception(f"Error processing {csv_file}")
        
        # Summary
        print(f"\n📊 Prediction Summary:")
        print(f"✅ Successful: {successful_predictions}")
        print(f"❌ Failed: {failed_predictions}")
        print(f"📁 Total files processed: {len(csv_files)}")
        
        return 0 if successful_predictions > 0 else 1
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        logger.exception("Fatal error in main()")
        return 1

if __name__ == "__main__":
    sys.exit(main())
