#!/usr/bin/env python3
"""
Ultimate Prediction Script using ultimate_insights.json
Race 2 - HEA - 11 July 2025 Analysis
"""

import pandas as pd
import numpy as np
import json
import os
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

def get_venue_bias(ultimate_insights, venue):
    """Get venue bias information from ultimate_insights"""
    for bias_info in ultimate_insights['track_bias_analysis']['venue_bias_summary']:
        if bias_info['venue'] == venue:
            return bias_info
    return {"best_box": 4, "avg_bias": 0.0, "positions_analyzed": 0}

def get_market_rank_performance(ultimate_insights, market_rank):
    """Get market rank performance from ultimate_insights"""
    for rank_info in ultimate_insights['market_dynamics']['market_rank_performance']:
        if rank_info['market_rank'] == market_rank:
            return rank_info
    return {"win_rate": 0.15, "avg_odds": 10.0}

def clean_race_data(df):
    """Clean and prepare the race data"""
    # Filter out continuation rows
    df = df[df['Dog Name'].str.contains(r'^\d+\.', na=False)].copy()
    
    # Clean dog names
    df['Dog Name'] = df['Dog Name'].str.replace(r'^\d+\.\s*', '', regex=True)
    
    # Convert numeric columns
    numeric_cols = ['WGT', 'TIME', 'DIST', '1 SEC', 'BOX', 'SP', 'PLC']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_ultimate_metrics(dog_row, ultimate_insights):
    """Calculate advanced metrics using ultimate_insights data"""
    
    # Basic values
    weight = dog_row['WGT'] if pd.notna(dog_row['WGT']) else 30.0
    time = dog_row['TIME'] if pd.notna(dog_row['TIME']) else 17.0
    dist = dog_row['DIST'] if pd.notna(dog_row['DIST']) else 300
    sectional = dog_row['1 SEC'] if pd.notna(dog_row['1 SEC']) else 6.0
    box = dog_row['BOX'] if pd.notna(dog_row['BOX']) else 1
    sp = dog_row['SP'] if pd.notna(dog_row['SP']) else 10.0
    venue = dog_row['TRACK'] if pd.notna(dog_row['TRACK']) else 'HEA'
    
    # Calculate core metrics
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

def calculate_ultimate_score(metrics, ultimate_insights):
    """Calculate prediction score using ultimate_insights"""
    
    score = 0
    factors = {}
    
    # Base performance scoring
    base_score = (
        metrics['early_speed'] * 0.25 +  # Early speed is critical
        (1 / metrics['relative_time']) * 0.20 +  # Faster times are better
        (1 / metrics['relative_weight']) * 0.15 +  # Lighter is often better
        metrics['avg_speed'] * 0.15 +  # Overall speed
        (1 / metrics['starting_price']) * 0.10  # Market confidence
    )
    
    # Venue bias adjustment
    venue_bias = get_venue_bias(ultimate_insights, metrics['venue'])
    box_bonus = 0.1 if metrics['box_number'] == venue_bias['best_box'] else 0
    
    # Market rank estimation (based on starting price)
    estimated_rank = 1 if metrics['starting_price'] <= 3 else \
                    2 if metrics['starting_price'] <= 6 else \
                    3 if metrics['starting_price'] <= 10 else \
                    4 if metrics['starting_price'] <= 15 else 5
    
    market_performance = get_market_rank_performance(ultimate_insights, estimated_rank)
    market_adjustment = market_performance['win_rate'] * 0.15
    
    # Final score calculation
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

def run_ultimate_prediction(race_file_path):
    """Run comprehensive ultimate prediction"""
    
    # Load ultimate insights
    with open("../ultimate_insights.json", 'r') as f:
        ultimate_insights = json.load(f)
    
    # Load race data
    race_data = pd.read_csv(race_file_path)
    race_data = clean_race_data(race_data)
    
    print("🔥 ULTIMATE GREYHOUND RACING ANALYSIS - Race 2 HEA - 11 July 2025")
    print("=" * 80)
    print(f"Analyzing {len(race_data)} greyhounds using ultimate_insights.json")
    print(f"Total Analysis Records: {ultimate_insights['track_bias_analysis']['total_bias_records']}")
    print(f"Market Records: {ultimate_insights['market_dynamics']['total_market_records']}")
    print()
    
    # HEA venue specific info
    hea_bias = get_venue_bias(ultimate_insights, 'HEA')
    print(f"🏁 HEA TRACK INSIGHTS:")
    print(f"   • Best Box: {hea_bias['best_box']}")
    print(f"   • Positions Analyzed: {hea_bias['positions_analyzed']}")
    print(f"   • Average Bias: {hea_bias['avg_bias']:.3f}")
    print()
    
    # Calculate predictions
    predictions = []
    
    for _, dog_row in race_data.iterrows():
        # Calculate metrics
        metrics = calculate_ultimate_metrics(dog_row, ultimate_insights)
        
        # Calculate prediction score
        final_score, factors = calculate_ultimate_score(metrics, ultimate_insights)
        
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
    
    return predictions, ultimate_insights

def generate_ultimate_ai_insights(predictions, ultimate_insights):
    """Generate AI insights using ultimate_insights context"""
    
    if client is None:
        return "AI insights unavailable - OpenAI API key not set"
    
    # Prepare enhanced data for AI
    race_summary = {
        "race_info": "Race 2 HEA - 11 July 2025",
        "total_runners": len(predictions),
        "hea_track_bias": get_venue_bias(ultimate_insights, 'HEA'),
        "top_3_picks": [
            {
                "name": pred["greyhound_name"],
                "confidence": f"{pred['confidence']*100:.1f}%",
                "box": pred["box"],
                "weight": f"{pred['weight']:.1f}kg",
                "early_speed": f"{pred['early_speed']:.1f} m/s",
                "starting_price": f"${pred['starting_price']:.1f}",
                "market_rank": pred["estimated_rank"],
                "market_win_rate": f"{pred['market_win_rate']:.1%}"
            }
            for pred in predictions[:3]
        ],
        "market_analysis": {
            "favorites_win_rate": "54.1% (rank 1)",
            "second_favorites_win_rate": "14.3% (rank 2)",
            "long_shots_performance": "7.1% (rank 5+)"
        },
        "track_insights": {
            "venue": "HEA",
            "best_box": get_venue_bias(ultimate_insights, 'HEA')['best_box'],
            "total_bias_records": ultimate_insights['track_bias_analysis']['total_bias_records']
        }
    }
    
    prompt = f"""
    As an expert greyhound racing analyst, provide advanced insights for Race 2 at HEA using comprehensive track bias and market analysis:

    RACE DATA:
    {json.dumps(race_summary, indent=2)}

    Based on the ultimate insights analysis, provide:
    1. **TRACK BIAS ANALYSIS**: How the HEA track bias affects this race
    2. **MARKET EFFICIENCY**: Value opportunities based on market rank performance
    3. **BOX DRAW IMPACT**: Specific advantages/disadvantages for each runner
    4. **BETTING STRATEGY**: Tactical recommendations based on track bias and market data
    5. **RISK FACTORS**: Key variables that could affect the predicted outcomes

    Focus on actionable insights that leverage the track bias data and market performance statistics.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional greyhound racing analyst specializing in track bias and market analysis. Use the provided data to give specific, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI Analysis error: {str(e)}"

def print_ultimate_analysis(predictions, ultimate_insights):
    """Print comprehensive ultimate analysis"""
    
    print("📊 ULTIMATE PREDICTIONS")
    print("=" * 80)
    
    for i, pred in enumerate(predictions, 1):
        confidence_pct = pred['confidence'] * 100
        print(f"\n{i}. {pred['greyhound_name']} (Box {pred['box']})")
        print(f"   🎯 Confidence: {confidence_pct:.1f}%")
        print(f"   ⚖️  Weight: {pred['weight']:.1f}kg")
        print(f"   🏃 Early Speed: {pred['early_speed']:.1f} m/s")
        print(f"   ⏱️  Average Speed: {pred['avg_speed']:.1f} m/s")
        print(f"   💰 Starting Price: ${pred['starting_price']:.1f}")
        print(f"   📈 Market Rank: {pred['estimated_rank']} (Win Rate: {pred['market_win_rate']:.1%})")
        print(f"   📋 Score Breakdown:")
        print(f"      • Base Score: {pred['factors']['base_score']:.4f}")
        print(f"      • Box Bonus: +{pred['factors']['box_bonus']:.4f}")
        print(f"      • Market Adjustment: +{pred['factors']['market_adjustment']:.4f}")
        print(f"      • Final Score: {pred['final_score']:.4f}")
    
    # Key insights
    print(f"\n🔥 ULTIMATE INSIGHTS")
    print("=" * 80)
    
    top_pick = predictions[0]
    hea_bias = get_venue_bias(ultimate_insights, 'HEA')
    
    print(f"• TOP PICK: {top_pick['greyhound_name']} ({top_pick['confidence']*100:.1f}% confidence)")
    print(f"• HEA TRACK BIAS: Box {hea_bias['best_box']} is statistically best")
    print(f"• MARKET LEADER: {min(predictions, key=lambda x: x['starting_price'])['greyhound_name']} (${min(p['starting_price'] for p in predictions):.1f})")
    print(f"• VALUE OPPORTUNITY: Look for dogs with high confidence but higher odds")
    
    # Box analysis
    boxes_in_race = [p['box'] for p in predictions]
    best_box_runners = [p for p in predictions if p['box'] == hea_bias['best_box']]
    
    if best_box_runners:
        print(f"• BEST BOX RUNNERS: {', '.join([p['greyhound_name'] for p in best_box_runners])}")
    
    print(f"• MARKET EFFICIENCY: Favorites win {54.1:.1f}% of races, 2nd favorites win {14.3:.1f}%")

def run_complete_ultimate_analysis():
    """Run complete ultimate analysis with AI insights"""
    
    race_file = "form_guides/Race 2 - HEA - 11 July 2025.csv"
    
    if not os.path.exists(race_file):
        print(f"Error: {race_file} not found")
        return
    
    if not os.path.exists("../ultimate_insights.json"):
        print("Error: ultimate_insights.json not found")
        return
    
    try:
        # Run ultimate prediction
        predictions, ultimate_insights = run_ultimate_prediction(race_file)
        
        # Generate AI insights if available
        if client:
            print("🧠 GENERATING ULTIMATE AI INSIGHTS...")
            print("=" * 80)
            ai_insights = generate_ultimate_ai_insights(predictions, ultimate_insights)
            print(ai_insights)
            print()
        
        # Print detailed analysis
        print_ultimate_analysis(predictions, ultimate_insights)
        
        # Betting recommendations
        print(f"\n💡 ULTIMATE BETTING RECOMMENDATIONS")
        print("=" * 80)
        
        # Value bets based on confidence vs odds
        print("🎯 VALUE BETS:")
        for i, pred in enumerate(predictions[:5], 1):
            value_ratio = pred['confidence'] * 100 / pred['starting_price']
            print(f"  {i}. {pred['greyhound_name']} - {pred['confidence']*100:.1f}% confidence @ ${pred['starting_price']:.1f} (Value: {value_ratio:.2f})")
        
        # Best combinations
        print("\n🏆 RECOMMENDED COMBINATIONS:")
        print(f"  • WIN: {predictions[0]['greyhound_name']} (top confidence)")
        print(f"  • PLACE: {predictions[1]['greyhound_name']} (value pick)")
        print(f"  • QUINELLA: {predictions[0]['greyhound_name']} + {predictions[1]['greyhound_name']}")
        
        # Track bias specific recommendations
        hea_bias = get_venue_bias(ultimate_insights, 'HEA')
        best_box_runners = [p for p in predictions if p['box'] == hea_bias['best_box']]
        if best_box_runners:
            print(f"  • TRACK BIAS SPECIAL: {best_box_runners[0]['greyhound_name']} (Box {hea_bias['best_box']} advantage)")
        
        print("\n✅ ULTIMATE ANALYSIS COMPLETE")
        print("=" * 80)
        print("This analysis incorporates:")
        print("• Track bias analysis from 5,130 records")
        print("• Market performance data from 504 market records")
        print("• HEA-specific track insights")
        print("• AI-powered strategic recommendations")
        
    except Exception as e:
        print(f"Error in ultimate analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_complete_ultimate_analysis()
