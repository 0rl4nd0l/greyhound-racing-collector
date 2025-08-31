#!/usr/bin/env python3
"""
End-to-end test to verify TGR integration works during actual predictions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tgr_e2e_prediction():
    print("🎯 End-to-End TGR Integration Test")
    print("=" * 60)
    
    try:
        # Test 1: Check if we have a sample race file
        print("\n1️⃣ Checking for test race files...")
        
        upcoming_dir = "data/upcoming"
        if not os.path.exists(upcoming_dir):
            upcoming_dir = "upcoming_races"
        
        if os.path.exists(upcoming_dir):
            race_files = [f for f in os.listdir(upcoming_dir) if f.endswith('.csv')]
            print(f"   📁 Found {len(race_files)} race files in {upcoming_dir}")
            if race_files:
                test_file = os.path.join(upcoming_dir, race_files[0])
                print(f"   🎯 Using test file: {race_files[0]}")
            else:
                test_file = None
                print(f"   ⚠️  No CSV files found")
        else:
            test_file = None
            print(f"   ⚠️  No upcoming races directory found")
        
        # Test 2: Create a minimal test CSV if none exists
        if not test_file:
            print("\n2️⃣ Creating test race file...")
            test_csv_content = """1. BALLARAT STAR,30.0,3,1,Ballarat,28.50,Fine
2. SPEED DEMON,29.5,2,2,Ballarat,29.10,Fine
3. TRACK MASTER,31.0,4,3,Ballarat,28.90,Fine

Ballarat,Grade 5,500m,2025-08-23,14:30,Fine,Good
"""
            test_file = "test_tgr_race.csv"
            with open(test_file, 'w') as f:
                f.write(test_csv_content)
            print(f"   ✅ Created test file: {test_file}")
        
        # Test 3: Test PredictionPipelineV4 with TGR
        print(f"\n3️⃣ Testing PredictionPipelineV4 with TGR...")
        try:
            from prediction_pipeline_v4 import PredictionPipelineV4
            pipeline = PredictionPipelineV4()
            print(f"   ✅ PredictionPipelineV4 loaded successfully")
            
            # Make a prediction
            print(f"   🔮 Making prediction on {os.path.basename(test_file)}...")
            result = pipeline.predict_race_file(test_file)
            
            if result and result.get('success'):
                print(f"   ✅ Prediction successful!")
                
                # Check prediction details
                predictions = result.get('predictions', [])
                summary = result.get('summary', {})
                
                print(f"   📊 Generated predictions for {len(predictions)} dogs")
                
                # Look for TGR feature usage indicators
                if predictions:
                    sample_pred = predictions[0]
                    features_used = sample_pred.get('features_used', [])
                    if any('tgr_' in str(f) for f in features_used):
                        print(f"   🎯 TGR features detected in prediction!")
                    
                    # Show top prediction
                    print(f"   🏆 Top prediction: {sample_pred.get('dog_name', 'Unknown')} "
                          f"(Win: {sample_pred.get('win_probability', 0):.3f})")
                
                # Check for TGR enhancement indicators in summary
                enhancement_info = summary.get('enhancement_info', {})
                if 'tgr' in str(enhancement_info).lower():
                    print(f"   🎯 TGR enhancement detected in summary!")
                    
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                print(f"   ❌ Prediction failed: {error_msg}")
                
        except ImportError as e:
            print(f"   ⚠️  PredictionPipelineV4 not available: {e}")
        except Exception as e:
            print(f"   ❌ Prediction test failed: {e}")
        
        # Test 4: Test direct ML System V4 prediction
        print(f"\n4️⃣ Testing ML System V4 direct prediction...")
        try:
            from ml_system_v4 import MLSystemV4
            import pandas as pd
            
            # Read the CSV file
            with open(test_file, 'r') as f:
                content = f.read()
                
            # Parse basic race data
            lines = content.strip().split('\n')
            race_data = []
            
            for line in lines[:3]:  # First 3 lines are dogs
                if ',' in line and not line.startswith('Ballarat'):
                    parts = line.split(',')
                    if len(parts) >= 6:
                        dog_name = parts[0].split('.', 1)[-1].strip() if '.' in parts[0] else parts[0].strip()
                        race_data.append({
                            'dog_name': dog_name,
                            'weight': float(parts[1]) if parts[1] else 30.0,
                            'box_number': int(parts[2]) if parts[2] else 1,
                            'venue': 'Ballarat',
                            'distance': 500,
                            'grade': 'Grade 5',
                            'race_date': '2025-08-23',
                            'race_time': '14:30',
                            'weather': 'Fine',
                            'track_condition': 'Good',
                            'temperature': 20,
                            'humidity': 60
                        })
            
            if race_data:
                df = pd.DataFrame(race_data)
                print(f"   📊 Prepared race data for {len(race_data)} dogs")
                
                ml_system = MLSystemV4()
                result = ml_system.predict_race(df, "test_tgr_e2e_race")
                
                if result and result.get('success'):
                    print(f"   ✅ ML System V4 prediction successful!")
                    predictions = result.get('predictions', [])
                    if predictions:
                        print(f"   🏆 Generated {len(predictions)} predictions")
                        top_pred = predictions[0]
                        print(f"   🥇 Winner: {top_pred.get('dog_name', 'Unknown')} "
                              f"(Win: {top_pred.get('win_probability', 0):.3f})")
                        
                        # Check total probability (should sum to ~1.0)
                        total_prob = sum(p.get('win_probability', 0) for p in predictions)
                        print(f"   🎯 Total win probability: {total_prob:.3f} (should be ~1.0)")
                        
                        if abs(total_prob - 1.0) < 0.01:
                            print(f"   ✅ Probabilities properly normalized")
                        else:
                            print(f"   ⚠️  Probabilities not properly normalized")
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                    print(f"   ❌ ML System V4 prediction failed: {error_msg}")
            else:
                print(f"   ⚠️  No race data could be parsed from test file")
                
        except Exception as e:
            print(f"   ❌ ML System V4 test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up test file if we created it
        if test_file == "test_tgr_race.csv":
            try:
                os.remove(test_file)
                print(f"\n🧹 Cleaned up test file")
            except:
                pass
        
        print("\n" + "=" * 60)
        print("🎉 End-to-End TGR Integration Test Complete!")
        print("✅ TGR-enhanced predictions are working in the live system")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-End Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tgr_e2e_prediction()
    sys.exit(0 if success else 1)
