#!/usr/bin/env python3
"""
Comprehensive test to verify all historical races are being loaded from all CSV files
"""

import sys
import os
sys.path.append(os.getcwd())

from comprehensive_enhanced_ml_system import ComprehensiveEnhancedMLSystem
import random

def test_comprehensive_loading():
    """Test comprehensive CSV loading to verify all historical races"""
    
    print("🧪 Testing comprehensive CSV loading...")
    
    # Initialize the system
    system = ComprehensiveEnhancedMLSystem()
    
    # Load form guide data
    form_data = system.load_form_guide_data()
    
    print(f"\n📊 Overall Summary:")
    print(f"Total dogs with form data: {len(form_data)}")
    
    # Calculate statistics
    total_races = sum(len(races) for races in form_data.values())
    avg_races = total_races / len(form_data) if form_data else 0
    max_races = max(len(races) for races in form_data.values()) if form_data else 0
    min_races = min(len(races) for races in form_data.values()) if form_data else 0
    
    print(f"Total historical races loaded: {total_races}")
    print(f"Average races per dog: {avg_races:.1f}")
    print(f"Maximum races for a dog: {max_races}")
    print(f"Minimum races for a dog: {min_races}")
    
    # Sample some dogs to verify their data
    sample_dogs = random.sample(list(form_data.keys()), min(10, len(form_data)))
    
    print(f"\n🔍 Sample of {len(sample_dogs)} dogs and their race counts:")
    for dog_name in sample_dogs:
        races = form_data[dog_name]
        print(f"  {dog_name}: {len(races)} races")
        
        # Show first 3 races for this dog
        if races:
            print(f"    Recent races:")
            for i, race in enumerate(races[:3]):
                print(f"      {i+1}. {race['place']} place on {race['date']} at {race['track']}")
    
    # Find dogs with most and least races
    if form_data:
        most_races_dog = max(form_data.keys(), key=lambda dog: len(form_data[dog]))
        least_races_dog = min(form_data.keys(), key=lambda dog: len(form_data[dog]))
        
        print(f"\n🏆 Dog with most races: {most_races_dog} ({len(form_data[most_races_dog])} races)")
        print(f"🥉 Dog with least races: {least_races_dog} ({len(form_data[least_races_dog])} races)")
        
        # Show all races for the dog with most races
        print(f"\n📈 All races for {most_races_dog}:")
        for i, race in enumerate(form_data[most_races_dog]):
            print(f"  {i+1:2d}. Place {race['place']:2s} | {race['date']} | {race['track']:4s} | Grade {race['grade']} | Weight {race['weight']}")
    
    # Verify data quality
    races_with_dates = sum(1 for races in form_data.values() for race in races if race['date'])
    races_with_places = sum(1 for races in form_data.values() for race in races if race['place'])
    races_with_tracks = sum(1 for races in form_data.values() for race in races if race['track'])
    
    print(f"\n✅ Data Quality Check:")
    print(f"Races with dates: {races_with_dates}/{total_races} ({100*races_with_dates/total_races:.1f}%)")
    print(f"Races with places: {races_with_places}/{total_races} ({100*races_with_places/total_races:.1f}%)")
    print(f"Races with tracks: {races_with_tracks}/{total_races} ({100*races_with_tracks/total_races:.1f}%)")
    
    return form_data

if __name__ == "__main__":
    test_comprehensive_loading()
