#!/usr/bin/env python3
"""
Real Race Prediction Test
========================

Test the prediction system on a real greyhound race.
"""

import pandas as pd
from prediction_pipeline_v4 import PredictionPipelineV4

def test_real_race():
    """Test prediction on real race data"""
    print("🏁 REAL RACE PREDICTION TEST")
    print("=" * 60)
    
    # Test multiple race files
    race_files = [
        "/Users/orlandolee/greyhound_racing_collector/archive/corrupt_or_legacy_race_files/20250730162231_Race 1 - BEN - 02 July 2025.csv",
        "/Users/orlandolee/greyhound_racing_collector/archive/corrupt_or_legacy_race_files/20250730162231_Race 1 - RICH - 04 July 2025.csv"
    ]
    
    success_count = 0
    
    for race_file in race_files:
        success = test_single_race(race_file)
        if success:
            success_count += 1
        print("\n" + "-" * 60)
    
    return success_count > 0

def test_single_race(race_file):
    """Test prediction on a single race file"""
    race_name = race_file.split('/')[-1].replace('.csv', '').replace('20250730162231_', '')
    print(f"\n📄 Testing: {race_name}")
    
    try:
        # Load the race data to show what we're predicting
        race_data = pd.read_csv(race_file)
        # Get race info from filename
        venue = race_data['TRACK'].iloc[0] if 'TRACK' in race_data.columns else 'Unknown'
        date = race_data['DATE'].iloc[0] if 'DATE' in race_data.columns else 'Unknown'
        distance = race_data['DIST'].iloc[0] if 'DIST' in race_data.columns else 'Unknown'
        
        print(f"\n📊 Race Details:")
        print(f"   🐕 Dogs in race: {len(race_data)}")
        print(f"   📍 Venue: {venue}")
        print(f"   📅 Date: {date}")
        print(f"   🏃 Distance: {distance}m")
        
        print(f"\n🐕 Dogs in this race:")
        for i, row in race_data.iterrows():
            dog_name = row['Dog Name']
            # Handle NaN or empty dog names
            if pd.isna(dog_name) or dog_name == '':
                continue
            
            # Extract clean dog name
            if '.' in str(dog_name):
                dog_name = str(dog_name).split('.')[1].strip()
            else:
                dog_name = str(dog_name).strip()
            
            # Skip empty names after processing
            if not dog_name or dog_name == '""':
                continue
                
            box = row['BOX']
            weight = row['WGT']
            print(f"   Box {box}: {dog_name} ({weight}kg)")
        
        # Initialize prediction pipeline
        pipeline = PredictionPipelineV4()
        
        print(f"\n🤖 Running prediction...")
        result = pipeline.predict_race_file(race_file)
        
        if result.get('success'):
            predictions = result.get('predictions', [])
            print(f"\n🎯 PREDICTION RESULTS")
            print("-" * 40)
            
            for i, pred in enumerate(predictions, 1):
                dog_name = pred.get('dog_clean_name', 'Unknown')
                win_prob = pred.get('win_prob_norm', 0)
                confidence = pred.get('confidence_level', 'Unknown')
                box_number = pred.get('box_number', 'Unknown')
                
                # Add medal emoji for top 3
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                
                print(f"{medal} {dog_name} (Box {box_number})")
                print(f"   📊 Win Probability: {win_prob:.1%}")
                print(f"   🎲 Confidence: {confidence}")
                print()
            
            # Summary stats
            top_3_prob = sum(p.get('win_prob_norm', 0) for p in predictions[:3])
            print(f"📈 Top 3 combined probability: {top_3_prob:.1%}")
            
            return True
            
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ Prediction failed: {error}")
            
            # Show detailed error if available
            if 'details' in result:
                print(f"   Details: {result['details']}")
            
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_race()
    print("\n" + "=" * 60)
    if success:
        print("🎉 REAL RACE PREDICTION TEST SUCCESSFUL!")
    else:
        print("⚠️ Real race prediction test failed")
    exit(0 if success else 1)
