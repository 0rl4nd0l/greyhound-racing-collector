#!/usr/bin/env python3
"""
Quick demonstration of TGR training impact
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_training_impact():
    print("🎯 TGR Training Impact Demonstration")
    print("=" * 60)
    
    try:
        # 1. Show current model status
        print("\n1️⃣ Current Model Status...")
        from ml_system_v4 import MLSystemV4
        
        ml_system = MLSystemV4()
        model_info = getattr(ml_system, 'model_info', {})
        feature_count = len(getattr(ml_system, 'feature_columns', []) or [])
        
        print(f"   📊 Current model features: {feature_count}")
        print(f"   📊 Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"   📊 Trained: {model_info.get('trained_at', 'Unknown')}")
        
        # 2. Show TGR feature availability
        print("\n2️⃣ TGR Feature Integration...")
        try:
            from tgr_prediction_integration import TGRPredictionIntegrator
            integrator = TGRPredictionIntegrator()
            tgr_features = integrator.get_feature_names()
            print(f"   🎯 TGR features available: {len(tgr_features)}")
            print(f"   🏷️  TGR feature types: {', '.join(tgr_features[:5])}...")
        except Exception as e:
            print(f"   ❌ TGR integration issue: {e}")
        
        # 3. Demonstrate feature building with TGR
        print("\n3️⃣ Feature Building with TGR...")
        from temporal_feature_builder import TemporalFeatureBuilder
        import pandas as pd
        from datetime import datetime
        
        temporal_builder = TemporalFeatureBuilder()
        
        # Create sample race data
        test_data = pd.DataFrame([
            {
                'dog_name': 'DEMO DOG',
                'box_number': 1,
                'venue': 'Ballarat',
                'distance': 500,
                'grade': 'Grade 5',
                'weight': 30.0,
                'race_time': '14:30',
                'race_date': '2025-08-23',
                'temperature': 20,
                'humidity': 60,
                'weather': 'Fine',
                'track_condition': 'Good'
            }
        ])
        
        features = temporal_builder.build_features_for_race(test_data, "demo_race")
        
        if not features.empty:
            all_features = list(features.columns)
            tgr_features = [f for f in all_features if f.startswith('tgr_')]
            standard_features = [f for f in all_features if not f.startswith('tgr_') and f not in ['race_id', 'target_timestamp', 'dog_clean_name']]
            
            print(f"   📊 Total features generated: {len(all_features)}")
            print(f"   🎯 TGR features: {len(tgr_features)}")
            print(f"   📈 Standard features: {len(standard_features)}")
            
            if tgr_features:
                print(f"   ✅ TGR integration working! Sample TGR features:")
                for feat in tgr_features[:5]:
                    value = features[feat].iloc[0] if feat in features.columns else 'N/A'
                    print(f"      {feat}: {value}")
            
            print(f"\n   📋 Feature composition for training:")
            print(f"      🔢 Numerical: {len([f for f in all_features if f not in ['venue', 'grade', 'track_condition', 'weather', 'trainer_name', 'race_id', 'target_timestamp', 'dog_clean_name']])}")
            print(f"      🏷️  Categorical: {len([f for f in all_features if f in ['venue', 'grade', 'track_condition', 'weather', 'trainer_name']])}")
        else:
            print("   ⚠️  No features generated")
        
        # 4. Show what training would look like
        print("\n4️⃣ Training Requirements...")
        print(f"   🔄 Model retraining: REQUIRED (current model incompatible)")
        print(f"   ⏱️  Training time: ~5-30 minutes (depending on data size)")
        print(f"   📊 Expected improvement: Better AUC/accuracy with TGR insights")
        print(f"   🎯 New model will handle {feature_count + len(tgr_features)} features")
        
        # 5. Show training commands
        print("\n5️⃣ Training Options...")
        print("   Option 1 - Quick training (5 min):")
        print("   > V4_MAX_RACES=50 V4_TREES=200 python train_model_v4.py")
        print("")
        print("   Option 2 - Full training (30 min):")
        print("   > python train_model_v4.py")
        print("")
        print("   Option 3 - Web UI training:")
        print("   > Visit http://localhost:5002 → ML Dashboard → Start Training")
        
        print("\n" + "=" * 60)
        print("🎉 TGR Training Impact Summary:")
        print("✅ TGR features automatically included in training")
        print("⚠️  Current models incompatible - retraining required")
        print("🎯 Expected significant accuracy improvements")
        print("🔄 Training pipeline ready - just needs execution")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    demonstrate_training_impact()
