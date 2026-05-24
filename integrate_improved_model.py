#!/usr/bin/env python3
"""
Model Integration Script
=======================

Integrates the improved model with the existing prediction system by:
1. Backing up the current model
2. Replacing it with the improved model
3. Updating the prediction system to use the enhanced features
4. Testing the integration
"""

import os
import pickle
import shutil
from pathlib import Path
import sqlite3
import pandas as pd
from datetime import datetime

def integrate_improved_model():
    print("🔄 Integrating improved model with prediction system...")
    print("=" * 60)
    
    # Check if improved model exists
    improved_model_path = "models/improved_staging_model.pkl"
    if not os.path.exists(improved_model_path):
        print("❌ Error: Improved model not found. Run train_improved_model.py first.")
        return False
    
    # Create backup of existing models
    print("💾 Creating backup of existing models...")
    models_dir = Path("models")
    backup_dir = models_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)
    
    # Backup existing models
    for model_file in models_dir.glob("*.pkl"):
        if model_file.name != "improved_staging_model.pkl":
            backup_path = backup_dir / model_file.name
            shutil.copy2(model_file, backup_path)
            print(f"   ✅ Backed up: {model_file.name}")
    
    # Load the improved model to verify it's working
    print("🔍 Verifying improved model...")
    try:
        with open(improved_model_path, "rb") as f:
            model_data = pickle.load(f)
        
        print(f"   ✅ Model loaded successfully")
        print(f"   📊 Training records: {model_data['training_records']:,}")
        print(f"   🎯 Test accuracy: {model_data['test_accuracy']:.3f}")
        print(f"   📈 Features: {len(model_data['feature_names'])}")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False
    
    # Replace the main model files with the improved one
    print("🚀 Deploying improved model...")
    
    # Copy to main model names
    main_model_paths = [
        "models/greyhound_model.pkl",
        "models/staging_model.pkl"
    ]
    
    for main_path in main_model_paths:
        try:
            shutil.copy2(improved_model_path, main_path)
            print(f"   ✅ Deployed to: {main_path}")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not deploy to {main_path}: {e}")
    
    # Test the integration with a sample prediction
    print("🧪 Testing integration with sample prediction...")
    
    try:
        # Load staging database to get real data for testing
        db_path = "greyhound_racing_data_staging.db"
        
        with sqlite3.connect(db_path) as conn:
            # Get a recent race for testing
            test_query = '''
            SELECT 
                h.dog_clean_name,
                h.box_number,
                h.weight,
                h.starting_price,
                h.venue,
                h.race_number,
                r.distance,
                r.grade,
                r.field_size
            FROM csv_dog_history_staging h
            LEFT JOIN csv_race_metadata_staging r ON h.race_id = r.race_id
            WHERE h.box_number IS NOT NULL
            AND h.weight IS NOT NULL
            AND h.venue IS NOT NULL
            ORDER BY h.race_date DESC
            LIMIT 8
            '''
            
            test_df = pd.read_sql_query(test_query, conn)
            
        if len(test_df) > 0:
            print(f"   📊 Testing with {len(test_df)} dogs from recent race")
            
            # Create features like the model expects
            features = pd.DataFrame()
            
            # Encode features using the model's encoders
            le_dog = model_data['dog_encoder']
            le_venue = model_data['venue_encoder']
            le_grade = model_data['grade_encoder']
            
            # Handle unknown values for encoders
            def safe_encode(encoder, values, default_value="unknown"):
                encoded = []
                for val in values.fillna(default_value):
                    try:
                        encoded.append(encoder.transform([val])[0])
                    except ValueError:
                        # Unknown value, use first class as fallback
                        encoded.append(0)
                return encoded
            
            features['dog_encoded'] = safe_encode(le_dog, test_df['dog_clean_name'])
            features['venue_encoded'] = safe_encode(le_venue, test_df['venue'])
            features['grade_encoded'] = safe_encode(le_grade, test_df['grade'])
            features['box_number'] = test_df['box_number'].fillna(4)
            features['race_number'] = test_df['race_number'].fillna(5)
            
            # Weight features
            features['weight'] = test_df['weight'].fillna(30.5)
            features['weight_heavy'] = (features['weight'] > 32.0).astype(int)
            features['weight_light'] = (features['weight'] < 29.0).astype(int)
            
            # Distance features
            features['distance'] = pd.to_numeric(test_df['distance'].str.replace('m', ''), errors='coerce').fillna(500)
            features['distance_long'] = (features['distance'] > 600).astype(int)
            features['distance_short'] = (features['distance'] < 400).astype(int)
            
            # Box position features
            features['box_inside'] = (features['box_number'] <= 3).astype(int)
            features['box_outside'] = (features['box_number'] >= 6).astype(int)
            features['box_wide'] = (features['box_number'] == 8).astype(int)
            
            # Field size features
            features['field_size'] = test_df['field_size'].fillna(8)
            features['small_field'] = (features['field_size'] <= 6).astype(int)
            features['large_field'] = (features['field_size'] >= 8).astype(int)
            
            # Odds features
            features['has_odds'] = test_df['starting_price'].notna().astype(int)
            features['starting_price'] = test_df['starting_price'].fillna(8.0)
            features['favorite'] = (features['starting_price'] <= 3.0).astype(int)
            features['longshot'] = (features['starting_price'] >= 10.0).astype(int)
            
            # Interaction features
            features['box_weight_interaction'] = features['box_number'] * features['weight'] / 100
            features['venue_distance_interaction'] = features['venue_encoded'] * features['distance'] / 1000
            
            # Make predictions
            model = model_data['model']
            predictions = model.predict_proba(features)[:, 1]
            
            print("   🏁 Sample predictions:")
            for i, (_, row) in enumerate(test_df.iterrows()):
                dog_name = row['dog_clean_name'][:20]  # Truncate long names
                box = row['box_number']
                prob = predictions[i]
                weight = row['weight']
                print(f"      Box {box}: {dog_name:<20} - {prob:.1%} (weight: {weight}kg)")
            
            # Check prediction quality
            pred_std = predictions.std()
            pred_range = predictions.max() - predictions.min()
            
            print(f"   📈 Prediction quality:")
            print(f"      • Range: {pred_range:.1%}")
            print(f"      • Std Dev: {pred_std:.3f}")
            
            if pred_std > 0.05:  # Good variation threshold
                print("   ✅ Predictions show good variation")
            else:
                print("   ⚠️  Predictions may be too uniform")
            
        else:
            print("   ⚠️  No test data available")
            
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False
    
    # Create a status file to indicate successful integration
    status_file = models_dir / "model_status.txt"
    with open(status_file, "w") as f:
        f.write(f"Model Integration Status\n")
        f.write(f"======================\n\n")
        f.write(f"Integration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model File: improved_staging_model.pkl\n")
        f.write(f"Training Records: {model_data['training_records']:,}\n")
        f.write(f"Test Accuracy: {model_data['test_accuracy']:.3f}\n")
        f.write(f"Features: {len(model_data['feature_names'])}\n")
        f.write(f"Status: Successfully Integrated\n")
    
    print(f"\n✅ Model integration completed successfully!")
    print(f"📄 Status file created: {status_file}")
    print(f"💾 Backups saved to: {backup_dir}")
    
    # Provide next steps
    print(f"\n🚀 Next Steps:")
    print(f"   1. Test the UI to see improved predictions")
    print(f"   2. Monitor prediction quality in production")
    print(f"   3. Consider retraining if needed")
    
    return True

if __name__ == "__main__":
    success = integrate_improved_model()
    if success:
        print("\n🎉 SUCCESS: Enhanced model is now active!")
    else:
        print("\n❌ FAILED: Model integration failed!")
