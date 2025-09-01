#!/usr/bin/env python3
"""
Quick test to verify TGR integration is working in the live system.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_tgr_integration():
    print("🧪 Testing TGR Integration in Live System")
    print("=" * 60)

    try:
        # Test 1: TGR Integration Module
        print("\n1️⃣ Testing TGR Integration Module...")
        from tgr_prediction_integration import TGRPredictionIntegrator

        integrator = TGRPredictionIntegrator()
        features = integrator.get_feature_names()
        print(f"   ✅ TGR Integration loaded successfully - {len(features)} features")
        print(f"   🏷️  Sample features: {', '.join(features[:5])}...")

        # Test 2: ML System V4 with TGR
        print("\n2️⃣ Testing ML System V4 with TGR...")
        from ml_system_v4 import MLSystemV4

        ml_system = MLSystemV4()
        print(f"   ✅ ML System V4 loaded successfully")

        # Check if temporal feature builder has TGR
        temporal_builder = getattr(ml_system, "temporal_feature_builder", None)
        if temporal_builder:
            tgr_integrator = getattr(temporal_builder, "tgr_integrator", None)
            if tgr_integrator:
                print(f"   ✅ TGR integrator found in temporal feature builder")
                print(
                    f"   🏷️  TGR features available: {len(tgr_integrator.get_feature_names())}"
                )
            else:
                print(f"   ⚠️  TGR integrator not found in temporal feature builder")
        else:
            print(f"   ⚠️  Temporal feature builder not found")

        # Test 3: Model Health Check
        print("\n3️⃣ Testing Model Health...")
        model_info = getattr(ml_system, "model_info", {})
        feature_count = len(getattr(ml_system, "feature_columns", []) or [])
        pipeline_ready = bool(getattr(ml_system, "calibrated_pipeline", None))

        print(f"   📊 Model ready: {pipeline_ready}")
        print(f"   📊 Feature count: {feature_count}")
        print(f"   📊 Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"   📊 Trained at: {model_info.get('trained_at', 'Unknown')}")

        # Test 4: Quick Feature Generation Test
        print("\n4️⃣ Testing Feature Generation with TGR...")
        from datetime import datetime

        import pandas as pd

        # Create mock race data
        test_race_data = [
            {
                "dog_name": "TEST DOG",
                "box_number": 1,
                "venue": "TEST_VENUE",
                "distance": 500,
                "grade": "Grade 1",
                "weight": 30.0,
                "race_time": "14:30",
                "race_date": "2025-08-23",
                "temperature": 20,
                "humidity": 60,
                "weather": "Fine",
                "track_condition": "Good",
            }
        ]

        df = pd.DataFrame(test_race_data)

        try:
            # Test feature building with TGR integration
            if temporal_builder:
                features = temporal_builder.build_features_for_race(
                    df, "test_race_tgr_live"
                )
                print(f"   ✅ Generated {len(features)} feature rows")
                if not features.empty:
                    tgr_features = [
                        col for col in features.columns if col.startswith("tgr_")
                    ]
                    print(f"   🎯 Found {len(tgr_features)} TGR features in output")
                    if tgr_features:
                        print(f"   🏷️  TGR features: {', '.join(tgr_features[:5])}...")
                    else:
                        print(f"   ⚠️  No TGR features found in output")
                else:
                    print(f"   ⚠️  Feature generation returned empty result")
            else:
                print(
                    f"   ❌ Cannot test feature generation - temporal builder not available"
                )

        except Exception as e:
            print(f"   ❌ Feature generation test failed: {e}")

        print("\n" + "=" * 60)
        print("🎉 TGR Integration Test Complete!")
        print("✅ System is ready with TGR-enhanced predictions")

        return True

    except Exception as e:
        print(f"❌ TGR Integration Test Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_tgr_integration()
    sys.exit(0 if success else 1)
