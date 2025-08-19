#!/usr/bin/env python3
"""
Simple Ballarat Prediction Runner using Pipeline V4
"""

import os
import sys
import json
from datetime import datetime
from prediction_pipeline_v4 import PredictionPipelineV4
import pandas as pd

def run_ballarat_prediction():
    """Run prediction for Ballarat race using v4 pipeline"""
    
    # Find the Ballarat race file
    ballarat_file = "/Users/orlandolee/greyhound_racing_collector/upcoming_races/Race 10 - BAL - 2025-08-04.csv"
    
    # Fix CSV structure if required
    cleaned_file = fix_csv_structure(ballarat_file)
    
    if not os.path.exists(cleaned_file):
        print(f"❌ Ballarat race file not found: {cleaned_file}")
        return
    
    print(f"🏁 Running Ballarat prediction using Pipeline V4")
    print(f"📁 File: {cleaned_file}")
    print()
    
    try:
        # Initialize v4 pipeline
        print("🚀 Initializing Prediction Pipeline V4...")
        pipeline_v4 = PredictionPipelineV4()
        print("✅ Pipeline V4 initialized successfully")
        print()
        
        # Run prediction
        print("🎯 Running prediction...")
        result = pipeline_v4.predict_race_file(cleaned_file)
        
        if result.get('success', False):
            print("✅ Prediction successful!")
            print()
            
            # Debug: Print all prediction data
            predictions = result.get('predictions', [])
            if predictions:
                print("🔍 DEBUG: Raw prediction data:")
                for i, pred in enumerate(predictions[:3]):  # Show first 3 dogs
                    print(f"  Dog {i+1}: {pred.get('dog_name', 'Unknown')}")
                    print(f"    Raw prob: {pred.get('win_prob_raw', 0.0):.6f}")
                    print(f"    Norm prob: {pred.get('win_prob_norm', 0.0):.6f}")
                    print(f"    Confidence: {pred.get('confidence', 'Unknown')}")
                print()
            
            # Display results
            print("🏆 BALLARAT RACE PREDICTIONS")
            print("=" * 50)
            
            predictions = result.get('predictions', [])
            if predictions:
                print(f"Race: {result.get('race_id', 'Unknown')}")
                print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
                print(f"System: ML System V4")
                print()
                
                # Sort by win probability (using the correct key from ML System v4)
                sorted_predictions = sorted(predictions, key=lambda x: x.get('win_prob_norm', 0), reverse=True)
                
                print("Rank | Box | Dog Name                  | Win Prob | Confidence")
                print("-" * 65)
                
                for i, pred in enumerate(sorted_predictions, 1):
                    dog_name = pred.get('dog_name', 'Unknown')[:25].ljust(25)
                    box_num = pred.get('box_number', '?')
                    win_prob = pred.get('win_prob_norm', 0.0) * 100  # Use the correct key
                    confidence = pred.get('confidence', 'Unknown')
                    
                    print(f"{i:4} | {box_num:3} | {dog_name} | {win_prob:7.1f}% | {confidence:.3f}")
                
                print()
                
                # Show system info
                if result.get('model_info'):
                    print(f"Model: {result['model_info']}")
                
                if result.get('fallback_used'):
                    print(f"⚠️  Fallback system used: {result.get('fallback_reason', 'Unknown')}")
                
            else:
                print("⚠️  No predictions generated")
                
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error running prediction: {e}")
        import traceback
        traceback.print_exc()

def fix_csv_structure(file_path):
    """Fix any structural issues in the CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Replace empty string dog names with proper forward fill
        df['Dog Name'] = df['Dog Name'].replace('""', pd.NA)
        df['Dog Name'].fillna(method='ffill', inplace=True)
        
        # Filter to only keep rows with dog names that start with numbers (actual race entries)
        # This extracts just the dogs running in today's race, not their historical data
        race_dogs = df[df['Dog Name'].str.match(r'^\d+\.')].copy()
        
        # Keep only the first entry for each unique dog (most recent race entry)
        race_dogs_unique = race_dogs.drop_duplicates(subset=['Dog Name'], keep='first')
        
        print(f"📊 Found {len(race_dogs_unique)} unique dogs in today's race:")
        for i, row in race_dogs_unique.iterrows():
            print(f"  {row['Dog Name']}")
        
        # Write back the race-only data
        race_dogs_unique.to_csv(file_path + ".race_only", index=False)
        print(f"✅ Created race-only file: {file_path}.race_only")
        
        return file_path + ".race_only"
        
    except Exception as e:
        print(f"❌ Error in fixing CSV: {e}")
        return file_path

if __name__ == "__main__":
    run_ballarat_prediction()
