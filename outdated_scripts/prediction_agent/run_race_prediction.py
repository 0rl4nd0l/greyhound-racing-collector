#!/usr/bin/env python3
"""
Custom script to run predictions on the specific race file
"""

import pandas as pd
import numpy as np
import json
import os
from prediction_assistant import get_weight_bracket, get_weight_bracket_performance, calculate_metrics, clean_dog_name

def clean_race_data(df):
    """
    Clean and prepare the race data for prediction
    """
    # Filter out rows with empty dog names (continuation rows)
    df = df[df['Dog Name'].str.contains(r'^\d+\.', na=False)]
    
    # Clean dog names - remove the number prefix
    df['Dog Name'] = df['Dog Name'].str.replace(r'^\d+\.\s*', '', regex=True)
    
    # Convert data types
    df['WGT'] = pd.to_numeric(df['WGT'], errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['DIST'] = pd.to_numeric(df['DIST'], errors='coerce')
    df['1 SEC'] = pd.to_numeric(df['1 SEC'], errors='coerce')
    df['BOX'] = pd.to_numeric(df['BOX'], errors='coerce')
    df['SP'] = pd.to_numeric(df['SP'], errors='coerce')
    df['PLC'] = pd.to_numeric(df['PLC'], errors='coerce')
    
    return df

def run_race_prediction(race_file_path):
    """
    Run prediction on the race file
    """
    # Load Ultra Insights JSON
    with open("../ultra_insights.json", 'r') as f:
        ultra_insights = json.load(f)
    
    # Load race data
    race_data = pd.read_csv(race_file_path)
    race_data = clean_race_data(race_data)
    
    print(f"Found {len(race_data)} greyhounds in the race:")
    for _, row in race_data.iterrows():
        print(f"  {row['Dog Name']} (Box {row['BOX']}, Weight {row['WGT']}kg)")
    print()
    
    # Calculate predictions for each greyhound
    predictions = []
    
    for _, dog_row in race_data.iterrows():
        # For this prediction, we'll use the current race data as the "recent form"
        # In a real scenario, you'd have historical data for each dog
        dog_data = pd.DataFrame([dog_row])
        
        # Calculate basic metrics from available data
        weight = dog_row['WGT'] if pd.notna(dog_row['WGT']) else 30.0
        time = dog_row['TIME'] if pd.notna(dog_row['TIME']) else 17.0
        dist = dog_row['DIST'] if pd.notna(dog_row['DIST']) else 300
        sectional = dog_row['1 SEC'] if pd.notna(dog_row['1 SEC']) else 6.0
        box = dog_row['BOX'] if pd.notna(dog_row['BOX']) else 1
        
        # Calculate metrics
        metrics = {
            'relative_time': time / dist if dist > 0 else 0.057,
            'early_speed': dist / sectional if sectional > 0 else 17.0,
            'relative_weight': weight / 31.0,
            'individual_time_numeric': time,
            'avg_speed': dist / time if time > 0 else 17.0,
            'box_number': box,
            'sectional_1_numeric': sectional,
            'sp_mean': dog_row['SP'] if pd.notna(dog_row['SP']) else 10.0
        }
        
        # Score based on ultra_insights feature importance
        score = 0
        for fi in ultra_insights['ultra_predictive_model']['feature_importance']:
            feature = fi['feature']
            if feature in metrics and not np.isnan(metrics[feature]):
                # Apply feature importance
                score += fi['importance'] * metrics[feature]
        
        # Adjust for weight bracket performance
        weight_bracket = get_weight_bracket(weight)
        weight_performance = get_weight_bracket_performance(ultra_insights, weight_bracket)
        score += weight_performance * 0.15
        
        # Speed bonus
        if metrics['early_speed'] > ultra_insights['speed_analysis']['early_speed_analysis']['winners_avg_early_speed']:
            score += 0.1
        
        # Box position factor (lower box numbers often have slight advantage)
        if box <= 3:
            score += 0.05
        
        predictions.append({
            "greyhound_name": dog_row['Dog Name'],
            "box": int(box),
            "weight": weight,
            "weight_bracket": weight_bracket,
            "early_speed": metrics['early_speed'],
            "relative_time": metrics['relative_time'],
            "confidence": score,
            "sp": dog_row['SP'] if pd.notna(dog_row['SP']) else 'N/A'
        })
    
    # Normalize confidence scores
    total_score = sum(p['confidence'] for p in predictions)
    for p in predictions:
        p['confidence'] = p['confidence'] / total_score if total_score > 0 else 1 / len(predictions)
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

if __name__ == "__main__":
    race_file = "form_guides/Race 1 - HEA - 11 July 2025.csv"
    
    if not os.path.exists(race_file):
        print(f"Error: {race_file} not found")
        exit(1)
    
    if not os.path.exists("../ultra_insights.json"):
        print("Error: ultra_insights.json not found in parent directory")
        exit(1)
    
    predictions = run_race_prediction(race_file)
    
    print("🏁 RACE PREDICTIONS - Race 1 HEA - 11 July 2025")
    print("=" * 60)
    print()
    
    for i, pred in enumerate(predictions, 1):
        confidence_pct = pred['confidence'] * 100
        print(f"{i}. {pred['greyhound_name']} (Box {pred['box']})")
        print(f"   Confidence: {confidence_pct:.1f}%")
        print(f"   Weight: {pred['weight']}kg ({pred['weight_bracket']})")
        print(f"   Early Speed: {pred['early_speed']:.1f} m/s")
        print(f"   Starting Price: ${pred['sp']}")
        print()
    
    print("Key Insights:")
    print(f"• Top pick: {predictions[0]['greyhound_name']} with {predictions[0]['confidence']*100:.1f}% confidence")
    print(f"• Fastest early speed: {max(p['early_speed'] for p in predictions):.1f} m/s")
    print(f"• Weight range: {min(p['weight'] for p in predictions):.1f}kg - {max(p['weight'] for p in predictions):.1f}kg")
