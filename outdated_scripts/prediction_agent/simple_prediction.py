#!/usr/bin/env python3
"""
Simple Greyhound Race Prediction Script
Uses available data to predict race outcomes
"""

import pandas as pd
import numpy as np
import json
import os

def clean_race_data(df):
    """Clean and prepare the race data for prediction"""
    # Filter out rows with empty dog names (continuation rows)
    df = df[df['Dog Name'].str.contains(r'^\d+\.', na=False)].copy()
    
    # Clean dog names - remove the number prefix
    df['Dog Name'] = df['Dog Name'].str.replace(r'^\d+\.\s*', '', regex=True)
    
    # Convert data types
    numeric_cols = ['WGT', 'TIME', 'DIST', '1 SEC', 'BOX', 'SP', 'PLC']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def get_venue_bias(insights_data, venue):
    """Get venue bias information"""
    if 'track_bias_analysis' in insights_data and 'venue_bias_summary' in insights_data['track_bias_analysis']:
        for bias_info in insights_data['track_bias_analysis']['venue_bias_summary']:
            if bias_info['venue'] == venue:
                return bias_info
    return {"best_box": 3, "avg_bias": 0.0, "positions_analyzed": 0}

def calculate_performance_metrics(dog_row):
    """Calculate performance metrics for a greyhound"""
    
    # Extract basic values with defaults
    weight = dog_row['WGT'] if pd.notna(dog_row['WGT']) else 30.0
    time = dog_row['TIME'] if pd.notna(dog_row['TIME']) else 17.0
    dist = dog_row['DIST'] if pd.notna(dog_row['DIST']) else 300
    sectional = dog_row['1 SEC'] if pd.notna(dog_row['1 SEC']) else 6.0
    box = dog_row['BOX'] if pd.notna(dog_row['BOX']) else 1
    sp = dog_row['SP'] if pd.notna(dog_row['SP']) else 10.0
    
    # Calculate performance metrics
    metrics = {
        'weight': weight,
        'time': time,
        'distance': dist,
        'sectional_time': sectional,
        'box_number': box,
        'starting_price': sp,
        'early_speed': dist / sectional if sectional > 0 else 17.0,
        'avg_speed': dist / time if time > 0 else 17.0,
        'relative_time': time / dist if dist > 0 else 0.057,
        'relative_weight': weight / 30.0,
        'track': dog_row['TRACK'] if pd.notna(dog_row['TRACK']) else 'HEA'
    }
    
    return metrics

def calculate_prediction_score(metrics, insights_data):
    """Calculate prediction score based on available data"""
    
    # Base scoring based on performance metrics
    score = 0
    
    # Early speed factor (higher is better)
    early_speed_factor = metrics['early_speed'] / 20.0  # Normalize around 20 m/s
    score += early_speed_factor * 0.30
    
    # Overall speed factor
    avg_speed_factor = metrics['avg_speed'] / 18.0  # Normalize around 18 m/s
    score += avg_speed_factor * 0.25
    
    # Weight factor (lighter is often better)
    weight_factor = 1.0 / metrics['relative_weight']
    score += weight_factor * 0.20
    
    # Time factor (faster is better)
    time_factor = 1.0 / metrics['relative_time']
    score += time_factor * 0.15
    
    # Market factor (shorter odds suggest better chance)
    market_factor = 1.0 / (metrics['starting_price'] / 5.0)  # Normalize around $5
    score += market_factor * 0.10
    
    # Venue bias adjustment
    venue_bias = get_venue_bias(insights_data, metrics['track'])
    box_bonus = 0.1 if metrics['box_number'] == venue_bias['best_box'] else 0
    score += box_bonus
    
    return score

def run_race_prediction(race_file_path):
    """Run race prediction analysis"""
    
    # Load insights data if available
    insights_data = {}
    if os.path.exists("../ultimate_insights.json"):
        with open("../ultimate_insights.json", 'r') as f:
            insights_data = json.load(f)
    
    # Load race data
    race_data = pd.read_csv(race_file_path)
    race_data = clean_race_data(race_data)
    
    print("🏁 SIMPLE RACE PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"Race: {race_file_path}")
    print(f"Analyzing {len(race_data)} greyhounds")
    
    # Get the main track from the race data
    main_track = race_data['TRACK'].iloc[0] if len(race_data) > 0 else 'HEA'
    track_bias = get_venue_bias(insights_data, main_track)
    print(f"\n🏁 {main_track} TRACK INFO:")
    print(f"   • Best Box: {track_bias['best_box']}")
    print(f"   • Positions Analyzed: {track_bias['positions_analyzed']}")
    print(f"   • Track Bias: {track_bias['avg_bias']:.3f}")
    print()
    
    # Calculate predictions
    predictions = []
    
    for _, dog_row in race_data.iterrows():
        # Calculate performance metrics
        metrics = calculate_performance_metrics(dog_row)
        
        # Calculate prediction score
        score = calculate_prediction_score(metrics, insights_data)
        
        predictions.append({
            "greyhound_name": dog_row['Dog Name'],
            "box": int(metrics['box_number']),
            "weight": metrics['weight'],
            "early_speed": metrics['early_speed'],
            "avg_speed": metrics['avg_speed'],
            "sectional_time": metrics['sectional_time'],
            "starting_price": metrics['starting_price'],
            "track": metrics['track'],
            "score": score,
            "performance_metrics": metrics
        })
    
    # Normalize confidence scores
    total_score = sum(p['score'] for p in predictions)
    for p in predictions:
        p['confidence'] = p['score'] / total_score if total_score > 0 else 1 / len(predictions)
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

def print_prediction_results(predictions):
    """Print detailed prediction results"""
    
    print("📊 RACE PREDICTIONS")
    print("=" * 60)
    
    for i, pred in enumerate(predictions, 1):
        confidence_pct = pred['confidence'] * 100
        print(f"\n{i}. {pred['greyhound_name']} (Box {pred['box']})")
        print(f"   🎯 Confidence: {confidence_pct:.1f}%")
        print(f"   ⚖️  Weight: {pred['weight']:.1f}kg")
        print(f"   🏃 Early Speed: {pred['early_speed']:.1f} m/s")
        print(f"   ⏱️  Average Speed: {pred['avg_speed']:.1f} m/s")
        print(f"   🕒 Sectional Time: {pred['sectional_time']:.2f}s")
        print(f"   💰 Starting Price: ${pred['starting_price']:.1f}")
        print(f"   📈 Score: {pred['score']:.3f}")
    
    print(f"\n🎯 BETTING RECOMMENDATIONS")
    print("=" * 60)
    
    # Top 3 picks
    print("🥇 TOP PICKS:")
    for i, pred in enumerate(predictions[:3], 1):
        value_ratio = pred['confidence'] * 100 / pred['starting_price']
        print(f"  {i}. {pred['greyhound_name']} - {pred['confidence']*100:.1f}% confidence @ ${pred['starting_price']:.1f} (Value: {value_ratio:.2f})")
    
    # Quinella suggestions
    print("\n🎯 QUINELLA SUGGESTIONS:")
    print(f"  1-2: {predictions[0]['greyhound_name']} + {predictions[1]['greyhound_name']}")
    print(f"  1-3: {predictions[0]['greyhound_name']} + {predictions[2]['greyhound_name']}")
    
    # Value bets
    print("\n💰 VALUE OPPORTUNITIES:")
    for pred in predictions:
        value_ratio = pred['confidence'] * 100 / pred['starting_price']
        if value_ratio > 2.0:  # Good value threshold
            print(f"  • {pred['greyhound_name']}: {value_ratio:.2f} value ratio")

if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    if len(sys.argv) > 1:
        race_file = sys.argv[1]
    else:
        race_file = "form_guides/Race 2 - HEA - 11 July 2025.csv"
    
    if not os.path.exists(race_file):
        print(f"Error: {race_file} not found")
        exit(1)
    
    try:
        predictions = run_race_prediction(race_file)
        print_prediction_results(predictions)
        
        print(f"\n✅ PREDICTION COMPLETE")
        print("=" * 60)
        print("This analysis uses:")
        print("• Performance metrics (speed, weight, time)")
        print("• Track bias information")
        print("• Market analysis (starting prices)")
        print("• Statistical scoring system")
        
    except Exception as e:
        print(f"Error running prediction: {e}")
        import traceback
        traceback.print_exc()
