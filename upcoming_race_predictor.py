#!/usr/bin/env python3
"""
Upcoming Race Predictor
========================

Predicts upcoming races using the comprehensive ML system and weather enhancement.
Can be called with a specific CSV file or will process all files in upcoming_races directory.

Usage:
    python upcoming_race_predictor.py                    # Process all upcoming races
    python upcoming_race_predictor.py path/to/race.csv  # Process specific race file
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

def predict_race_file(race_file_path):
    """Predict a specific race file using the unified prediction system"""
    print(f"🎯 Predicting: {os.path.basename(race_file_path)}")
    
    if not os.path.exists(race_file_path):
        print(f"❌ Race file not found: {race_file_path}")
        return False
    
    # Use the unified predictor system (PRIMARY SYSTEM)
    try:
        print("🚀 Using Unified Prediction System...")
        from unified_predictor import UnifiedPredictor
        
        # Initialize unified predictor
        predictor = UnifiedPredictor()
        
        # Make prediction with full level (includes GPT enhancement)
        results = predictor.predict_race_file(race_file_path, enhancement_level='full')
        
        if results and results.get('success'):
            method_used = results.get('prediction_method', 'unknown')
            prediction_time = results.get('prediction_time_seconds', 0)
            
            print(f"✅ Unified prediction completed successfully")
            print(f"📊 Method: {method_used}")
            print(f"⏱️  Time: {prediction_time:.2f}s")
            
            if results.get('enhanced_with_gpt'):
                print("🤖 Enhanced with GPT analysis")
            
            display_prediction_results(results)
            
            # Save prediction results in the expected format
            save_prediction_results(race_file_path, results)
            return True
        else:
            print(f"⚠️ Unified prediction failed: {results.get('error', 'Unknown error')}")
            print("🔄 Falling back to basic prediction...")
            
    except Exception as e:
        print(f"⚠️ Unified predictor failed: {e}")
        print("🔄 Falling back to basic prediction...")
    
    # Basic fallback prediction
    print("🔄 Using basic prediction fallback...")
    return basic_prediction_fallback(race_file_path)

def basic_prediction_fallback(race_file_path):
    """Basic prediction fallback when ML systems are not available"""
    try:
        import pandas as pd
        
        # Read the race file
        df = pd.read_csv(race_file_path)
        
        if df.empty:
            print("❌ Race file is empty")
            return False
        
        print(f"📊 Found {len(df)} dogs in race")
        
        # Basic prediction based on available data
        predictions = []
        for idx, row in df.iterrows():
            dog_name = str(row.get('dog_name', f'Dog_{idx+1}'))
            box_number = row.get('box', idx + 1)
            
            # Simple scoring based on available metrics
            score = 0.5  # Base score
            
            # Adjust based on box number (inside boxes often have slight advantage)
            if box_number in [1, 2, 3, 4]:
                score += 0.1
            
            # Add some randomness for basic prediction
            import random
            score += random.uniform(-0.2, 0.2)
            score = max(0.1, min(0.9, score))  # Clamp between 0.1 and 0.9
            
            predictions.append({
                'dog_name': dog_name,
                'box_number': box_number,
                'prediction_score': score,
                'confidence_level': 'LOW',
                'reasoning': 'Basic fallback prediction - no ML model available'
            })
        
        # Sort by prediction score
        predictions.sort(key=lambda x: x['prediction_score'], reverse=True)
        
        # Save basic prediction results
        race_id = os.path.basename(race_file_path).replace('.csv', '')
        prediction_data = {
            'race_info': {
                'filename': os.path.basename(race_file_path),
                'race_id': race_id
            },
            'predictions': predictions,
            'prediction_method': 'basic_fallback',
            'prediction_timestamp': datetime.now().isoformat(),
            'top_pick': predictions[0] if predictions else None
        }
        
        # Save to predictions directory
        predictions_dir = Path('./predictions')
        predictions_dir.mkdir(exist_ok=True)
        
        prediction_file = predictions_dir / f'prediction_{race_id}.json'
        with open(prediction_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        print("✅ Basic prediction completed")
        display_prediction_results({'predictions': predictions})
        
        return True
        
    except Exception as e:
        print(f"❌ Basic prediction fallback failed: {e}")
        return False

def display_prediction_results(results):
    """Display prediction results in a formatted way"""
    predictions = results.get('predictions', [])
    
    if not predictions:
        print("⚠️ No predictions available")
        return
    
    print("\n🏆 Top 3 picks:")
    for i, pred in enumerate(predictions[:3], 1):
        dog_name = pred.get('dog_name', 'Unknown')
        score = pred.get('prediction_score', 0)
        box_num = pred.get('box_number', 'N/A')
        confidence = pred.get('confidence_level', 'UNKNOWN')
        
        print(f"  {i}. {dog_name} (Box {box_num}) - Score: {score:.3f} - Confidence: {confidence}")
    
    print(f"\n📊 Total dogs analyzed: {len(predictions)}")
    
    # Show score distribution for quality verification
    if len(predictions) > 1:
        scores = [p.get('prediction_score', 0) for p in predictions]
        score_range = max(scores) - min(scores)
        print(f"📈 Score range: {score_range:.3f} (Min: {min(scores):.3f}, Max: {max(scores):.3f})")

def save_prediction_results(race_file_path, results):
    """Save prediction results to the predictions directory with duplicate prevention"""
    try:
        # Extract race info for filename
        race_id = os.path.basename(race_file_path).replace('.csv', '')
        
        # Prepare data in the format expected by the Flask app
        prediction_data = {
            'race_info': results.get('race_info', {
                'filename': os.path.basename(race_file_path),
                'race_id': race_id
            }),
            'predictions': results.get('predictions', []),
            'prediction_method': results.get('prediction_method', 'unified_predictor'),
            'prediction_timestamp': results.get('timestamp', datetime.now().isoformat()),
            'prediction_time_seconds': results.get('prediction_time_seconds', 0),
            'unified_predictor_version': results.get('unified_predictor_version', '1.0.0'),
            'top_pick': results.get('predictions', [{}])[0] if results.get('predictions') else None,
            'enhanced_with_gpt': results.get('enhanced_with_gpt', False)
        }
        
        # Save to predictions directory
        predictions_dir = Path('./predictions')
        predictions_dir.mkdir(exist_ok=True)
        
        # Create standardized filename without timestamp to prevent duplicates
        filename = f'unified_prediction_{race_id}.json'
        filepath = predictions_dir / filename
        
        # Check if file already exists and is recent (within last hour)
        if filepath.exists():
            file_age = time.time() - filepath.stat().st_mtime
            if file_age < 3600:  # Less than 1 hour old
                print(f"⚠️ Recent prediction already exists: {filename} (age: {file_age/60:.1f} min)")
                print(f"💡 Skipping save to prevent duplicate")
                return
            else:
                # Create backup of old prediction
                backup_filename = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = predictions_dir / backup_filename
                filepath.rename(backup_path)
                print(f"📁 Backed up old prediction to: {backup_filename}")
        
        # Save new prediction
        with open(filepath, 'w') as f:
            json.dump(prediction_data, f, indent=2, default=str)
        
        print(f"💾 Prediction results saved to: {filepath}")
        
    except Exception as e:
        print(f"⚠️ Error saving prediction results: {e}")

def main():
    """Main entry point"""  
    if len(sys.argv) > 1:
        # Predict specific file
        race_file_path = sys.argv[1]
        success = predict_race_file(race_file_path)
        sys.exit(0 if success else 1)
    
    # Process all upcoming races
    upcoming_dir = Path('./upcoming_races')
    if not upcoming_dir.exists():
        print("❌ No upcoming_races directory found")
        sys.exit(1)
    
    race_files = list(upcoming_dir.glob('*.csv'))
    if not race_files:
        print("ℹ️ No race files found in upcoming_races directory")
        sys.exit(0)
    
    print(f"🎯 Found {len(race_files)} race files to predict")
    
    successful_predictions = 0
    for race_file in race_files:
        if race_file.name == 'README.md':
            continue
            
        print(f"\n{'='*50}")
        if predict_race_file(str(race_file)):
            successful_predictions += 1
    
    print(f"\n🏁 Prediction completed: {successful_predictions}/{len(race_files)} successful")
    
    if successful_predictions == 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
