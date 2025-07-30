#!/usr/bin/env python3
"""
Fix Missing Probabilities and Ratings
=====================================

This script updates existing database entries to calculate missing
win probabilities, performance ratings, and other metrics.
"""

import sqlite3
import pandas as pd
from datetime import datetime

def calculate_probabilities_from_sp(starting_price):
    """Calculate win and place probabilities from starting price"""
    try:
        if starting_price and starting_price > 0:
            win_probability = 1 / starting_price
            place_probability = min(win_probability * 3, 1.0)
            return {
                'win_probability': min(win_probability, 1.0),
                'place_probability': place_probability
            }
    except:
        pass
    return {'win_probability': 0.1, 'place_probability': 0.3}

def calculate_performance_rating_from_time(time_str):
    """Calculate performance rating from time"""
    try:
        if time_str and str(time_str).replace('.', '').isdigit():
            time_val = float(time_str)
            # Simple rating: lower time = higher rating
            return max(0, 100 - (time_val - 20) * 5)
    except:
        pass
    return 50.0

def calculate_class_rating_from_grade(grade_str):
    """Calculate class rating from grade"""
    try:
        grade = str(grade_str).upper()
        if 'MAIDEN' in grade:
            return 30.0
        elif 'GRADE 7' in grade or 'G7' in grade:
            return 40.0
        elif 'GRADE 6' in grade or 'G6' in grade:
            return 50.0
        elif 'GRADE 5' in grade or 'G5' in grade:
            return 60.0
        elif 'GRADE 4' in grade or 'G4' in grade:
            return 70.0
        elif 'GRADE 3' in grade or 'G3' in grade:
            return 80.0
        elif 'GRADE 2' in grade or 'G2' in grade:
            return 90.0
        elif 'GRADE 1' in grade or 'G1' in grade:
            return 100.0
    except:
        pass
    return 50.0

def main():
    print("🔧 FIXING MISSING PROBABILITIES AND RATINGS")
    print("=" * 50)
    
    conn = sqlite3.connect('greyhound_racing_data.db')
    
    try:
        # Get all dog data with missing probabilities/ratings
        query = '''
        SELECT 
            d.id,
            d.race_id,
            d.dog_clean_name,
            d.starting_price,
            d.individual_time,
            d.win_probability,
            d.place_probability,
            d.performance_rating,
            d.speed_rating,
            d.class_rating,
            r.grade
        FROM dog_race_data d
        JOIN race_metadata r ON d.race_id = r.race_id
        WHERE d.race_id IN (
            SELECT race_id FROM race_metadata 
            WHERE race_date >= date('now', '-30 days')
        )
        AND (
            d.win_probability IS NULL 
            OR d.place_probability IS NULL 
            OR d.performance_rating IS NULL
            OR d.class_rating IS NULL
        )
        '''
        
        df = pd.read_sql_query(query, conn)
        print(f"📊 Found {len(df)} records needing updates")
        
        if len(df) == 0:
            print("✅ No records need updating")
            return
        
        updates_made = 0
        
        for idx, row in df.iterrows():
            # Calculate missing values
            starting_price = row['starting_price']
            individual_time = row['individual_time']
            grade = row['grade']
            
            # Calculate probabilities
            probabilities = calculate_probabilities_from_sp(starting_price)
            win_prob = probabilities['win_probability']
            place_prob = probabilities['place_probability']
            
            # Calculate ratings
            perf_rating = calculate_performance_rating_from_time(individual_time)
            class_rating = calculate_class_rating_from_grade(grade)
            speed_rating = 50.0  # Default for now
            
            # Update the database
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE dog_race_data 
                SET 
                    win_probability = ?,
                    place_probability = ?,
                    performance_rating = ?,
                    speed_rating = ?,
                    class_rating = ?
                WHERE id = ?
            ''', (
                win_prob,
                place_prob, 
                perf_rating,
                speed_rating,
                class_rating,
                row['id']
            ))
            
            updates_made += 1
            
            if updates_made % 100 == 0:
                print(f"   📈 Updated {updates_made}/{len(df)} records...")
        
        conn.commit()
        print(f"✅ Successfully updated {updates_made} records")
        
        # Verify the updates
        print("\n🔍 VERIFYING UPDATES...")
        verification_query = '''
        SELECT 
            COUNT(*) as total_dogs,
            COUNT(win_probability) as non_null_win_prob,
            AVG(win_probability) as avg_win_prob,
            MIN(win_probability) as min_win_prob,
            MAX(win_probability) as max_win_prob,
            COUNT(performance_rating) as non_null_perf_rating,
            AVG(performance_rating) as avg_perf_rating
        FROM dog_race_data 
        WHERE race_id IN (SELECT race_id FROM race_metadata WHERE race_date >= date('now', '-30 days'))
        '''
        
        verification = pd.read_sql_query(verification_query, conn)
        print("Updated statistics:")
        print(f"  Total dogs: {verification.iloc[0]['total_dogs']}")
        print(f"  Dogs with win probability: {verification.iloc[0]['non_null_win_prob']}")
        print(f"  Average win probability: {verification.iloc[0]['avg_win_prob']:.3f}")
        print(f"  Win probability range: {verification.iloc[0]['min_win_prob']:.3f} - {verification.iloc[0]['max_win_prob']:.3f}")
        print(f"  Dogs with performance rating: {verification.iloc[0]['non_null_perf_rating']}")
        print(f"  Average performance rating: {verification.iloc[0]['avg_perf_rating']:.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    print("\n🎉 Fix completed!")

if __name__ == "__main__":
    main()
