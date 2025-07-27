#!/usr/bin/env python3
"""
Smart Prediction Script with Auto Dataset Integration
Automatically copies uploaded CSV files to unprocessed analysis folder
"""

import pandas as pd
import numpy as np
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client if available
client = None
if os.environ.get("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        print("OpenAI library not available")

def copy_to_unprocessed(csv_file_path):
    """Copy uploaded CSV file to analysis_agent/unprocessed folder"""
    
    # Define paths
    unprocessed_dir = "../analysis_agent/unprocessed"
    
    # Create unprocessed directory if it doesn't exist
    os.makedirs(unprocessed_dir, exist_ok=True)
    
    # Get filename and create timestamped version
    filename = os.path.basename(csv_file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create new filename with timestamp
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{timestamp}{ext}"
    
    # Copy file to unprocessed folder
    destination = os.path.join(unprocessed_dir, new_filename)
    
    try:
        shutil.copy2(csv_file_path, destination)
        print(f"✅ CSV copied to analysis_agent dataset: {destination}")
        return destination
    except Exception as e:
        print(f"❌ Error copying CSV to analysis_agent dataset: {e}")
        return None

def get_venue_bias(ultimate_insights, venue):
    """Get venue bias information"""
    for bias_info in ultimate_insights['track_bias_analysis']['venue_bias_summary']:
        if bias_info['venue'] == venue:
            return bias_info
    return {"best_box": 4, "avg_bias": 0.0, "positions_analyzed": 0}

def get_market_rank_performance(ultimate_insights, market_rank):
    """Get market rank performance"""
    for rank_info in ultimate_insights['market_dynamics']['market_rank_performance']:
        if rank_info['market_rank'] == market_rank:
            return rank_info
    return {"win_rate": 0.15, "avg_odds": 10.0}

def clean_race_data(df):
    """Clean and prepare race data"""
    df = df[df['Dog Name'].str.contains(r'^\d+\.', na=False)].copy()
    df['Dog Name'] = df['Dog Name'].str.replace(r'^\d+\.\s*', '', regex=True)
    
    numeric_cols = ['WGT', 'TIME', 'DIST', '1 SEC', 'BOX', 'SP', 'PLC']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_enhanced_metrics(dog_row, ultimate_insights):
    """Calculate enhanced metrics"""
    
    weight = dog_row['WGT'] if pd.notna(dog_row['WGT']) else 30.0
    time = dog_row['TIME'] if pd.notna(dog_row['TIME']) else 17.0
    dist = dog_row['DIST'] if pd.notna(dog_row['DIST']) else 300
    sectional = dog_row['1 SEC'] if pd.notna(dog_row['1 SEC']) else 6.0
    box = dog_row['BOX'] if pd.notna(dog_row['BOX']) else 1
    sp = dog_row['SP'] if pd.notna(dog_row['SP']) else 10.0
    venue = dog_row['TRACK'] if pd.notna(dog_row['TRACK']) else 'HEA'
    
    metrics = {
        'relative_time': time / dist if dist > 0 else 0.057,
        'early_speed': dist / sectional if sectional > 0 else 17.0,
        'relative_weight': weight / 30.0,
        'individual_time': time,
        'avg_speed': dist / time if time > 0 else 17.0,
        'box_number': box,
        'sectional_1': sectional,
        'starting_price': sp,
        'venue': venue
    }
    
    return metrics

def calculate_prediction_score(metrics, ultimate_insights):
    """Calculate prediction score"""
    
    # Enhanced scoring algorithm
    base_score = (
        metrics['early_speed'] * 0.30 +  # Early speed critical
        (1 / metrics['relative_time']) * 0.25 +  # Faster times
        (1 / metrics['relative_weight']) * 0.15 +  # Weight efficiency
        metrics['avg_speed'] * 0.15 +  # Overall speed
        (1 / metrics['starting_price']) * 0.15  # Market confidence
    )
    
    # Venue bias adjustment
    venue_bias = get_venue_bias(ultimate_insights, metrics['venue'])
    box_bonus = 0.12 if metrics['box_number'] == venue_bias['best_box'] else 0
    
    # Market rank estimation
    estimated_rank = (1 if metrics['starting_price'] <= 3 else
                     2 if metrics['starting_price'] <= 6 else
                     3 if metrics['starting_price'] <= 10 else
                     4 if metrics['starting_price'] <= 15 else 5)
    
    market_performance = get_market_rank_performance(ultimate_insights, estimated_rank)
    market_adjustment = market_performance['win_rate'] * 0.18
    
    final_score = base_score + box_bonus + market_adjustment
    
    factors = {
        'base_score': base_score,
        'box_bonus': box_bonus,
        'market_adjustment': market_adjustment,
        'venue_best_box': venue_bias['best_box'],
        'estimated_market_rank': estimated_rank,
        'market_win_rate': market_performance['win_rate']
    }
    
    return final_score, factors

def generate_ai_insights(predictions, ultimate_insights, race_info):
    """Generate AI-powered insights"""
    
    if client is None:
        return "AI insights unavailable - API key not set"
    
    race_summary = {
        "race_info": race_info,
        "total_runners": len(predictions),
        "venue_bias": get_venue_bias(ultimate_insights, predictions[0]['venue']),
        "top_3_picks": [
            {
                "name": pred["greyhound_name"],
                "confidence": f"{pred['confidence']*100:.1f}%",
                "box": pred["box"],
                "early_speed": f"{pred['early_speed']:.1f} m/s",
                "starting_price": f"${pred['starting_price']:.1f}",
                "market_rank": pred["estimated_rank"]
            }
            for pred in predictions[:3]
        ],
        "dataset_stats": {
            "total_bias_records": ultimate_insights['track_bias_analysis']['total_bias_records'],
            "market_records": ultimate_insights['market_dynamics']['total_market_records']
        }
    }
    
    prompt = f"""
    Analyze this greyhound race using comprehensive track bias and market data:
    
    {json.dumps(race_summary, indent=2)}
    
    Provide:
    1. **TACTICAL RACE ANALYSIS**: How the race will unfold
    2. **VALUE BETTING OPPORTUNITIES**: Best value bets
    3. **RISK ASSESSMENT**: Key risks and variables
    4. **BETTING STRATEGY**: Specific recommendations
    
    Focus on actionable insights for profitable betting.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert greyhound racing analyst. Provide specific, actionable betting advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI Analysis error: {str(e)}"

def run_smart_prediction(csv_file_path):
    """Run smart prediction with auto dataset integration"""
    
    # First, copy CSV to unprocessed folder
    dataset_copy = copy_to_unprocessed(csv_file_path)
    
    # Load ultimate insights
    if not os.path.exists("../ultimate_insights.json"):
        print("❌ Error: ultimate_insights.json not found")
        return None
    
    with open("../ultimate_insights.json", 'r') as f:
        ultimate_insights = json.load(f)
    
    # Load and process race data
    race_data = pd.read_csv(csv_file_path)
    race_data = clean_race_data(race_data)
    
    # Extract race info from filename
    race_info = os.path.basename(csv_file_path).replace('.csv', '')
    
    print(f"🏁 SMART PREDICTION ANALYSIS - {race_info}")
    print("=" * 80)
    print(f"📊 Dataset Integration: CSV copied to analysis_agent/unprocessed folder")
    print(f"📈 Analysis Records: {ultimate_insights['track_bias_analysis']['total_bias_records']}")
    print(f"🎯 Market Records: {ultimate_insights['market_dynamics']['total_market_records']}")
    
    # Get venue info
    venue = race_data['TRACK'].iloc[0] if len(race_data) > 0 else 'Unknown'
    venue_bias = get_venue_bias(ultimate_insights, venue)
    
    print(f"\n🏁 {venue} TRACK INSIGHTS:")
    print(f"   • Best Box: {venue_bias['best_box']}")
    print(f"   • Positions Analyzed: {venue_bias['positions_analyzed']}")
    print(f"   • Average Bias: {venue_bias['avg_bias']:.3f}")
    print()
    
    # Calculate predictions
    predictions = []
    
    for _, dog_row in race_data.iterrows():
        metrics = calculate_enhanced_metrics(dog_row, ultimate_insights)
        final_score, factors = calculate_prediction_score(metrics, ultimate_insights)
        
        predictions.append({
            "greyhound_name": dog_row['Dog Name'],
            "box": int(metrics['box_number']),
            "weight": metrics['relative_weight'] * 30.0,
            "early_speed": metrics['early_speed'],
            "avg_speed": metrics['avg_speed'],
            "starting_price": metrics['starting_price'],
            "venue": metrics['venue'],
            "final_score": final_score,
            "factors": factors,
            "estimated_rank": factors['estimated_market_rank'],
            "market_win_rate": factors['market_win_rate']
        })
    
    # Normalize confidence scores
    total_score = sum(p['final_score'] for p in predictions)
    for p in predictions:
        p['confidence'] = p['final_score'] / total_score if total_score > 0 else 1 / len(predictions)
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Generate AI insights
    if client:
        print("🧠 GENERATING AI INSIGHTS...")
        print("=" * 80)
        ai_insights = generate_ai_insights(predictions, ultimate_insights, race_info)
        print(ai_insights)
        print()
    
    # Print predictions
    print("🔥 SMART PREDICTIONS")
    print("=" * 80)
    
    for i, pred in enumerate(predictions, 1):
        confidence_pct = pred['confidence'] * 100
        print(f"\n{i}. {pred['greyhound_name']} (Box {pred['box']})")
        print(f"   🎯 Confidence: {confidence_pct:.1f}%")
        print(f"   🏃 Early Speed: {pred['early_speed']:.1f} m/s")
        print(f"   💰 Starting Price: ${pred['starting_price']:.1f}")
        print(f"   📈 Market Rank: {pred['estimated_rank']} (Win Rate: {pred['market_win_rate']:.1%})")
        print(f"   📊 Final Score: {pred['final_score']:.4f}")
    
    # Key insights
    print(f"\n🔥 KEY INSIGHTS")
    print("=" * 80)
    
    top_pick = predictions[0]
    market_leader = min(predictions, key=lambda x: x['starting_price'])
    
    print(f"• TOP PICK: {top_pick['greyhound_name']} ({top_pick['confidence']*100:.1f}% confidence)")
    print(f"• MARKET LEADER: {market_leader['greyhound_name']} (${market_leader['starting_price']:.1f})")
    print(f"• TRACK BIAS: Box {venue_bias['best_box']} is statistically best")
    
    # Best value bets
    print(f"\n💡 BETTING RECOMMENDATIONS")
    print("=" * 80)
    
    print("🎯 VALUE BETS:")
    for i, pred in enumerate(predictions[:3], 1):
        value_ratio = pred['confidence'] * 100 / pred['starting_price']
        print(f"  {i}. {pred['greyhound_name']} - {pred['confidence']*100:.1f}% confidence @ ${pred['starting_price']:.1f} (Value: {value_ratio:.2f})")
    
    print(f"🏆 COMBINATIONS:")
    print(f"  • WIN: {predictions[0]['greyhound_name']}")
    print(f"  • PLACE: {predictions[1]['greyhound_name']}")
    print(f"  • QUINELLA: {predictions[0]['greyhound_name']} + {predictions[1]['greyhound_name']}")
    
    print(f"\n✅ ANALYSIS COMPLETE - CSV ADDED TO ANALYSIS_AGENT DATASET")
    print("=" * 80)
    return predictions

def main():
    """Main function to handle command line arguments"""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 smart_prediction.py <csv_file_path>")
        print("Available files in form_guides:")
        for file in os.listdir("form_guides"):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    csv_file = sys.argv[1]
    
    # If relative path, check in form_guides
    if not os.path.isabs(csv_file) and not os.path.exists(csv_file):
        csv_file = os.path.join("form_guides", csv_file)
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found")
        return
    
    run_smart_prediction(csv_file)

if __name__ == "__main__":
    main()
