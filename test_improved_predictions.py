#!/usr/bin/env python3
"""
Quick Test: Improved Predictions
================================

Tests the newly deployed model to verify it produces more varied predictions.
"""

import pickle
import pandas as pd
import sqlite3
import numpy as np

def test_improved_predictions():
    print("🧪 Testing improved model predictions...")
    print("=" * 50)
    
    # Load the deployed model
    try:
        with open("models/greyhound_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        print(f"✅ Loaded model with {model_data['training_records']:,} training records")
        print(f"📊 Test accuracy: {model_data['test_accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load real race data for testing
    try:
        db_path = "greyhound_racing_data_staging.db"
        
        with sqlite3.connect(db_path) as conn:
            # Get races with 8 dogs for fair comparison
            query = '''
            SELECT 
                h.race_id,
                h.dog_clean_name,
                h.box_number,
                h.weight,
                h.starting_price,
                h.venue,
                h.race_number,
                h.finish_position,
                r.distance,
                r.grade,
                r.field_size
            FROM csv_dog_history_staging h
            LEFT JOIN csv_race_metadata_staging r ON h.race_id = r.race_id
            WHERE h.box_number IS NOT NULL
            AND h.weight IS NOT NULL
            AND h.venue IS NOT NULL
            AND r.field_size = 8  -- Get 8-dog races
            ORDER BY h.race_date DESC
            LIMIT 24  -- 3 races worth
            '''
            
            df = pd.read_sql_query(query, conn)
        
        print(f"📊 Testing with {len(df)} dogs from recent races")
        
        # Group by race_id to process complete races
        races = df.groupby('race_id')
        
        all_variations = []
        race_count = 0
        
        for race_id, race_data in races:
            if len(race_data) == 8:  # Only process complete 8-dog races
                race_count += 1
                print(f"\n🏁 Race {race_count}: {race_id[:8]}...")
                
                # Create features for this race
                features = pd.DataFrame()
                
                # Use model encoders
                le_dog = model_data['dog_encoder']
                le_venue = model_data['venue_encoder']
                le_grade = model_data['grade_encoder']
                
                # Safe encoding function
                def safe_encode(encoder, values, default_value="unknown"):
                    encoded = []
                    for val in values.fillna(default_value):
                        try:
                            encoded.append(encoder.transform([val])[0])
                        except ValueError:
                            encoded.append(0)  # Fallback
                    return encoded
                
                # Build all features
                features['dog_encoded'] = safe_encode(le_dog, race_data['dog_clean_name'])
                features['venue_encoded'] = safe_encode(le_venue, race_data['venue'])
                features['grade_encoded'] = safe_encode(le_grade, race_data['grade'])
                features['box_number'] = race_data['box_number'].fillna(4)
                features['race_number'] = race_data['race_number'].fillna(5)
                features['weight'] = race_data['weight'].fillna(30.5)
                features['weight_heavy'] = (features['weight'] > 32.0).astype(int)
                features['weight_light'] = (features['weight'] < 29.0).astype(int)
                features['distance'] = pd.to_numeric(race_data['distance'].str.replace('m', ''), errors='coerce').fillna(500)
                features['distance_long'] = (features['distance'] > 600).astype(int)
                features['distance_short'] = (features['distance'] < 400).astype(int)
                features['box_inside'] = (features['box_number'] <= 3).astype(int)
                features['box_outside'] = (features['box_number'] >= 6).astype(int)
                features['box_wide'] = (features['box_number'] == 8).astype(int)
                features['field_size'] = race_data['field_size'].fillna(8)
                features['small_field'] = (features['field_size'] <= 6).astype(int)
                features['large_field'] = (features['field_size'] >= 8).astype(int)
                features['has_odds'] = race_data['starting_price'].notna().astype(int)
                features['starting_price'] = race_data['starting_price'].fillna(8.0)
                features['favorite'] = (features['starting_price'] <= 3.0).astype(int)
                features['longshot'] = (features['starting_price'] >= 10.0).astype(int)
                features['box_weight_interaction'] = features['box_number'] * features['weight'] / 100
                features['venue_distance_interaction'] = features['venue_encoded'] * features['distance'] / 1000
                
                # Make predictions
                model = model_data['model']
                predictions = model.predict_proba(features)[:, 1]
                
                # Calculate variation metrics
                pred_std = predictions.std()
                pred_range = predictions.max() - predictions.min()
                all_variations.append(pred_std)
                
                # Show predictions sorted by probability (highest first)
                results = []
                for i, (_, row) in enumerate(race_data.iterrows()):
                    results.append({
                        'box': row['box_number'],
                        'dog': row['dog_clean_name'][:15],
                        'weight': row['weight'],
                        'odds': row['starting_price'],
                        'prediction': predictions[i],
                        'actual_position': row['finish_position']
                    })
                
                # Sort by prediction (best first)
                results.sort(key=lambda x: x['prediction'], reverse=True)
                
                for i, r in enumerate(results):
                    actual = f"(finished {r['actual_position']})" if pd.notna(r['actual_position']) else ""
                    odds_str = f"${r['odds']:.1f}" if pd.notna(r['odds']) else "N/A"
                    print(f"   {i+1}. Box {r['box']:2d} {r['dog']:<15} {r['prediction']:5.1%} {actual:15} odds:{odds_str}")
                
                print(f"   📈 Range: {pred_range:.1%}, Std: {pred_std:.3f}")
                
                if race_count >= 3:  # Limit to 3 races for readability
                    break
        
        # Overall statistics
        if all_variations:
            avg_variation = np.mean(all_variations)
            print(f"\n📊 Overall Results:")
            print(f"   • Races tested: {len(all_variations)}")
            print(f"   • Average std deviation: {avg_variation:.3f}")
            print(f"   • Min variation: {min(all_variations):.3f}")
            print(f"   • Max variation: {max(all_variations):.3f}")
            
            if avg_variation > 0.08:
                print("   ✅ EXCELLENT: High prediction variation!")
            elif avg_variation > 0.05:
                print("   ✅ GOOD: Adequate prediction variation.")
            elif avg_variation > 0.02:
                print("   ⚠️  FAIR: Some prediction variation.")
            else:
                print("   ❌ POOR: Low prediction variation.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_predictions()
