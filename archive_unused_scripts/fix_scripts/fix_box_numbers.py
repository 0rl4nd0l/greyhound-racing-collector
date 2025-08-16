#!/usr/bin/env python3
"""
Fix Box Numbers in Past Predictions
==================================

This script corrects the box number assignments in existing prediction files
by re-parsing the original race files and updating the predictions with correct data.
"""

import json
import os
import re
from pathlib import Path

import pandas as pd


def extract_correct_box_numbers(race_file_path):
    """Extract correct box numbers from race CSV file"""
    try:
        df = pd.read_csv(race_file_path)
        box_mappings = {}
        
        for _, row in df.iterrows():
            raw_dog_name = str(row.get('Dog Name', row.get('DOG', ''))).strip()
            if not raw_dog_name or raw_dog_name.lower() == 'nan' or raw_dog_name == '""':
                continue
            
            # Extract box number from format like "3. WHIRLER"
            box_number_pattern = r'^(\d+)\.'
            match_box = re.match(box_number_pattern, raw_dog_name)
            if match_box:
                box_number = int(match_box.group(1))
                # Extract clean dog name
                dog_name = re.sub(box_number_pattern, '', raw_dog_name).strip()
                box_mappings[dog_name.upper()] = box_number
        
        return box_mappings
    except Exception as e:
        print(f"Error reading {race_file_path}: {e}")
        return {}

def find_race_file(race_info):
    """Find the corresponding race file for prediction"""
    filename = race_info.get('filename', '')
    
    # Check upcoming_races directory first
    upcoming_path = f"upcoming_races/{filename}"
    if os.path.exists(upcoming_path):
        return upcoming_path
    
    # Check unprocessed directory
    unprocessed_path = f"unprocessed/{filename}"
    if os.path.exists(unprocessed_path):
        return unprocessed_path
    
    # Check processed directory
    processed_path = f"processed/{filename}"
    if os.path.exists(processed_path):
        return processed_path
    
    return None

def fix_prediction_file(prediction_file_path):
    """Fix box numbers in a single prediction file"""
    try:
        print(f"Processing: {prediction_file_path}")
        
        # Load prediction file
        with open(prediction_file_path, 'r') as f:
            prediction_data = json.load(f)
        
        # Find corresponding race file
        race_info = prediction_data.get('race_info', {})
        race_file_path = find_race_file(race_info)
        
        if not race_file_path:
            print(f"  ⚠️ Race file not found for {race_info.get('filename', 'unknown')}")
            return False
        
        # Get correct box mappings
        box_mappings = extract_correct_box_numbers(race_file_path)
        if not box_mappings:
            print(f"  ⚠️ No box mappings found in {race_file_path}")
            return False
        
        # Track corrections
        corrections_made = 0
        
        # Fix predictions
        predictions = prediction_data.get('predictions', [])
        for prediction in predictions:
            dog_name = prediction.get('dog_name', '').upper()
            clean_name = prediction.get('clean_name', '').upper()
            current_box = prediction.get('box_number')
            
            # Try both dog_name and clean_name
            correct_box = None
            if dog_name in box_mappings:
                correct_box = box_mappings[dog_name]
            elif clean_name in box_mappings:
                correct_box = box_mappings[clean_name]
            
            if correct_box and str(current_box) != str(correct_box):
                print(f"  🔧 {dog_name}: Box {current_box} → {correct_box}")
                prediction['box_number'] = correct_box
                corrections_made += 1
        
        # Fix top_pick if it exists
        top_pick = prediction_data.get('top_pick')
        if top_pick:
            dog_name = top_pick.get('dog_name', '').upper()
            clean_name = top_pick.get('clean_name', '').upper()
            current_box = top_pick.get('box_number')
            
            correct_box = None
            if dog_name in box_mappings:
                correct_box = box_mappings[dog_name]
            elif clean_name in box_mappings:
                correct_box = box_mappings[clean_name]
            
            if correct_box and str(current_box) != str(correct_box):
                print(f"  🔧 Top Pick {dog_name}: Box {current_box} → {correct_box}")
                top_pick['box_number'] = correct_box
                corrections_made += 1
        
        # Save corrected file if changes were made
        if corrections_made > 0:
            with open(prediction_file_path, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            print(f"  ✅ Fixed {corrections_made} box numbers")
            return True
        else:
            print(f"  ✅ No corrections needed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {prediction_file_path}: {e}")
        return False

def main():
    """Main function to fix all prediction files"""
    predictions_dir = "predictions"
    
    if not os.path.exists(predictions_dir):
        print("❌ Predictions directory not found")
        return
    
    # Get all prediction JSON files
    prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.json') and 'prediction_' in f]
    
    if not prediction_files:
        print("❌ No prediction files found")
        return
    
    print(f"🚀 Found {len(prediction_files)} prediction files to check")
    
    total_fixed = 0
    files_with_corrections = 0
    
    for filename in sorted(prediction_files):
        file_path = os.path.join(predictions_dir, filename)
        if fix_prediction_file(file_path):
            files_with_corrections += 1
        total_fixed += 1
    
    print(f"\n📊 Summary:")
    print(f"   📁 Files processed: {total_fixed}")
    print(f"   🔧 Files corrected: {files_with_corrections}")
    print(f"   ✅ Box number corrections completed!")

if __name__ == "__main__":
    main()
