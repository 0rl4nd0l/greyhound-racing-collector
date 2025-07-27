#!/usr/bin/env python3

import json
import os

def test_prediction_parsing():
    """Test parsing of prediction files to verify confidence level extraction"""
    
    # Test with the TAREE prediction file we examined
    prediction_file = "./predictions/prediction_Race 1 - TAREE - 2025-07-26.json"
    
    if not os.path.exists(prediction_file):
        print(f"❌ Prediction file not found: {prediction_file}")
        return
    
    try:
        with open(prediction_file, 'r') as f:
            data = json.load(f)
        
        print("🔍 TESTING PREDICTION FILE PARSING")
        print("=" * 50)
        
        # Extract race info
        race_info = data.get('race_info', {})
        predictions_list = data.get('predictions', [])
        prediction_methods = data.get('prediction_methods_used', [])
        
        print(f"📋 Race Info:")
        print(f"   Venue: {race_info.get('venue', 'Unknown')}")
        print(f"   Date: {race_info.get('date', 'Unknown')}")
        print(f"   Distance: {race_info.get('distance', 'Unknown')}")
        print(f"   Grade: {race_info.get('grade', 'Unknown')}")
        print()
        
        print(f"🧠 Prediction Methods: {prediction_methods}")
        ml_used = any(method in ['ml_system', 'enhanced_data', 'weather_enhanced'] for method in prediction_methods) if prediction_methods else False
        print(f"   ML Used: {ml_used}")
        print()
        
        print(f"🐕 Total Dogs: {len(predictions_list)}")
        print()
        
        if predictions_list:
            # Check top pick (first prediction)
            first_pred = predictions_list[0]
            
            print("🏆 TOP PICK ANALYSIS:")
            print(f"   Dog Name: {first_pred.get('dog_name', 'Unknown')}")
            print(f"   Box Number: {first_pred.get('box_number', 'N/A')}")
            print(f"   Final Score: {first_pred.get('final_score', 0)}")
            print(f"   Confidence Level: {first_pred.get('confidence_level', 'MISSING')}")
            print(f"   Prediction Scores: {first_pred.get('prediction_scores', {})}")
            print()
            
            # Create the data structure as the API would
            top_pick_data = {
                'dog_name': first_pred.get('dog_name', 'Unknown'),
                'clean_name': first_pred.get('dog_name', 'Unknown'),
                'box_number': first_pred.get('box_number', 'N/A'),
                'prediction_score': float(first_pred.get('final_score', 0)),
                'confidence_level': first_pred.get('confidence_level', 'MEDIUM')
            }
            
            print("📤 API OUTPUT FOR TOP PICK:")
            print(json.dumps(top_pick_data, indent=2))
            print()
            
            # Check all predictions confidence levels
            print("📊 ALL PREDICTIONS CONFIDENCE LEVELS:")
            for i, pred in enumerate(predictions_list, 1):
                dog_name = pred.get('dog_name', 'Unknown')
                confidence = pred.get('confidence_level', 'MISSING')
                score = pred.get('final_score', 0)
                print(f"   {i:2d}. {dog_name:20s} | Confidence: {confidence:10s} | Score: {score:.3f}")
        
        else:
            print("❌ No predictions found in file")
            
    except Exception as e:
        print(f"❌ Error parsing prediction file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_parsing()
