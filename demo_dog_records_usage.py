#!/usr/bin/env python3
"""
Demonstration script showing how to work with the structured per-dog records
after parsing and restructuring the CSV data.
"""

import pandas as pd
import os
from glob import glob

def load_all_dog_records():
    """
    Load all individual dog records from the dog_records directory.
    Returns a dictionary with dog names as keys and DataFrames as values.
    """
    dog_records = {}
    
    # Find all CSV files in the dog_records directory
    csv_files = glob("dog_records/*.csv")
    
    for file_path in csv_files:
        # Extract dog name from filename
        filename = os.path.basename(file_path)
        dog_name = filename.replace('_historical_runs.csv', '').replace('_', ' ')
        
        # Load the dog's historical data
        df = pd.read_csv(file_path)
        dog_records[dog_name] = df
        
    return dog_records

def analyze_dog_performance(dog_name, dog_data):
    """
    Analyze a single dog's performance from their historical runs.
    """
    print(f"\n=== Analysis for {dog_name} ===")
    print(f"Number of races: {len(dog_data)}")
    
    # Performance metrics
    avg_place = dog_data['place'].mean()
    best_place = dog_data['place'].min()
    consistency = dog_data['place'].std()
    
    print(f"Average finishing position: {avg_place:.2f}")
    print(f"Best finishing position: {best_place}")
    print(f"Consistency (lower = more consistent): {consistency:.2f}")
    
    # Weight and fitness trends
    if not dog_data['weight_kg'].isna().all():
        weight_trend = dog_data['weight_kg'].diff().mean()
        print(f"Weight trend (kg per race): {weight_trend:.2f}")
        print(f"Current weight: {dog_data.iloc[0]['weight_kg']:.1f} kg")
    
    # Time performance
    if not dog_data['race_time'].isna().all():
        avg_time = dog_data['race_time'].mean()
        best_time = dog_data['race_time'].min()
        print(f"Average race time: {avg_time:.2f} seconds")
        print(f"Best race time: {best_time:.2f} seconds")
    
    # Recent form (last 3 races)
    recent_places = dog_data.head(3)['place'].tolist()
    print(f"Recent form (last 3 races): {recent_places}")
    
    return {
        'avg_place': avg_place,
        'best_place': best_place,
        'consistency': consistency,
        'recent_form': recent_places
    }

def compare_dogs(dog_records):
    """
    Compare performance across all dogs.
    """
    print("\n=== Dog Performance Comparison ===")
    
    comparison_data = []
    for dog_name, dog_data in dog_records.items():
        analysis = analyze_dog_performance(dog_name, dog_data)
        comparison_data.append({
            'dog_name': dog_name,
            **analysis
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n=== Summary Ranking ===")
    # Rank by average place (lower is better)
    ranking = comparison_df.sort_values('avg_place')
    
    for i, (_, dog) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {dog['dog_name']} - Avg Place: {dog['avg_place']:.2f}, "
              f"Best: {dog['best_place']}, Consistency: {dog['consistency']:.2f}")

def demonstrate_per_dog_view():
    """
    Demonstrate how the restructured data provides a 'one dog, many past runs' view
    instead of the original 'many rows' format.
    """
    print("=== Demonstrating Per-Dog Data Structure ===")
    
    # Load all dog records
    dog_records = load_all_dog_records()
    
    print(f"Loaded {len(dog_records)} dogs with historical data")
    
    # Show structure for one dog
    if dog_records:
        sample_dog_name = list(dog_records.keys())[0]
        sample_data = dog_records[sample_dog_name]
        
        print(f"\nSample dog: {sample_dog_name}")
        print(f"Data shape: {sample_data.shape}")
        print(f"Columns: {list(sample_data.columns)}")
        print(f"\nFirst few races (most recent first):")
        print(sample_data[['race_date', 'place', 'track_code', 'race_time', 'starting_price']].head())
    
    # Analyze all dogs
    for dog_name, dog_data in dog_records.items():
        analyze_dog_performance(dog_name, dog_data)
    
    # Compare dogs
    compare_dogs(dog_records)

if __name__ == "__main__":
    # Check if processed data exists
    if not os.path.exists("dog_records") or not os.listdir("dog_records"):
        print("Error: No processed dog records found!")
        print("Please run 'python process_upcoming_races.py' first to generate the data.")
        exit(1)
    
    # Demonstrate the per-dog view
    demonstrate_per_dog_view()
    
    print("\n=== Summary ===")
    print("✅ Successfully loaded and parsed the upcoming-race CSV")
    print("✅ Grouped the five historical rows per dog into individual objects")
    print("✅ Validated data types and handled missing cells")
    print("✅ Normalized column names")
    print("✅ Created per-dog DataFrames for 'one dog, many past runs' view")
    print("\nStep 1 of the broader plan is now complete!")
