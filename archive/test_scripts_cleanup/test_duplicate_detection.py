#!/usr/bin/env python3
"""
Test script to detect duplicate race data in CSV loading
"""

import sys
import os
sys.path.append(os.getcwd())

from comprehensive_enhanced_ml_system import ComprehensiveEnhancedMLSystem
from collections import defaultdict

def test_duplicate_detection():
    """Test for duplicate race entries in the loaded form data"""
    
    print("🔍 Testing for duplicate race data...")
    
    # Initialize the system
    system = ComprehensiveEnhancedMLSystem()
    
    # Load form guide data
    form_data = system.load_form_guide_data()
    
    print(f"\n📊 Analyzing {len(form_data)} dogs for duplicates...")
    
    total_duplicates = 0
    dogs_with_duplicates = 0
    
    for dog_name, races in form_data.items():
        # Create a set to track unique races
        unique_races = set()
        duplicates_for_dog = 0
        
        for race in races:
            # Create a unique identifier for each race
            race_id = (
                race['date'],
                race['track'], 
                race['place'],
                race['box'],
                race['weight'],
                race['distance']
            )
            
            if race_id in unique_races:
                duplicates_for_dog += 1
                total_duplicates += 1
            else:
                unique_races.add(race_id)
        
        if duplicates_for_dog > 0:
            dogs_with_duplicates += 1
            print(f"⚠️  {dog_name}: {duplicates_for_dog} duplicate races (total: {len(races)})")
            
            # Show detailed duplicates for first few dogs
            if dogs_with_duplicates <= 3:
                print(f"    Detailed analysis for {dog_name}:")
                race_counts = defaultdict(int)
                for race in races:
                    race_key = f"{race['date']} at {race['track']} - Place {race['place']}"
                    race_counts[race_key] += 1
                
                for race_key, count in race_counts.items():
                    if count > 1:
                        print(f"      🔄 {race_key} appears {count} times")
    
    print(f"\n📈 Duplicate Detection Summary:")
    print(f"Total duplicate races found: {total_duplicates}")
    print(f"Dogs with duplicates: {dogs_with_duplicates}/{len(form_data)}")
    
    if total_duplicates > 0:
        print(f"⚠️  Duplicate rate: {100*total_duplicates/sum(len(races) for races in form_data.values()):.2f}%")
        
        # Analyze the source of duplicates
        print(f"\n🔍 Analyzing duplicate sources...")
        
        # Check if duplicates come from overlapping CSV files
        race_sources = defaultdict(list)
        
        for dog_name, races in form_data.items():
            for race in races:
                race_key = (
                    dog_name,
                    race['date'],
                    race['track'],
                    race['place']
                )
                race_sources[race_key].append(race.get('source_file', 'unknown'))
        
        duplicate_sources = {k: v for k, v in race_sources.items() if len(v) > 1}
        
        print(f"Races appearing in multiple source files: {len(duplicate_sources)}")
        
        # Show some examples
        for i, (race_key, sources) in enumerate(list(duplicate_sources.items())[:5]):
            dog_name, date, track, place = race_key
            print(f"  {i+1}. {dog_name} - {date} at {track} (Place {place})")
            for source in sources:
                print(f"     Source: {source}")
    
    else:
        print("✅ No duplicates found!")
    
    return total_duplicates > 0

if __name__ == "__main__":
    has_duplicates = test_duplicate_detection()
    if has_duplicates:
        print("\n🛠️  Need to implement deduplication logic!")
    else:
        print("\n✅ Data is clean - no duplicates detected!")
