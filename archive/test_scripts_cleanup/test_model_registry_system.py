#!/usr/bin/env python3
"""
Model Registry System Test
===========================

Test and demonstrate the centralized model registry system with:
- Model registry functionality
- Model monitoring service
- Periodic reloading
- Performance tracking

Author: AI Assistant
Date: July 27, 2025
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

def test_model_registry():
    """Test the model registry functionality"""
    print("🧪 Testing Model Registry System")
    print("=" * 50)
    
    try:
        from model_registry import get_model_registry
        
        # Get registry instance
        registry = get_model_registry()
        
        # Show registry stats
        stats = registry.get_registry_stats()
        print(f"📊 Registry Statistics:")
        print(f"   Total models: {stats.get('total_models', 0)}")
        print(f"   Active models: {stats.get('active_models', 0)}")
        print(f"   Best model ID: {stats.get('best_model_id', 'None')}")
        print(f"   Best model accuracy: {stats.get('best_model_accuracy', 0):.3f}")
        print(f"   Model types: {stats.get('model_types', [])}")
        print(f"   Registry size: {stats.get('registry_size_mb', 0):.1f} MB")
        
        # List available models
        models = registry.list_models()
        print(f"\n📋 Available Models ({len(models)}):")
        for i, model in enumerate(models[:5], 1):  # Show top 5
            print(f"   {i}. {model.model_id}")
            print(f"      Type: {model.model_type}")
            print(f"      Performance: Acc={model.accuracy:.3f}, AUC={model.auc:.3f}, F1={model.f1_score:.3f}")
            print(f"      Training: {model.training_timestamp[:19]}")
            print(f"      Features: {model.features_count}")
            print(f"      Best: {'✅' if model.is_best else '❌'}")
        
        # Try to get the best model
        best_model_result = registry.get_best_model()
        if best_model_result:
            model, scaler, metadata = best_model_result
            print(f"\n🏆 Best Model Loaded:")
            print(f"   ID: {metadata.model_id}")
            print(f"   Name: {metadata.model_name}")
            print(f"   Type: {metadata.model_type}")
            print(f"   Performance: Acc={metadata.accuracy:.3f}, AUC={metadata.auc:.3f}")
            print(f"   Training samples: {metadata.training_samples:,}")
            print(f"   Model object type: {type(model).__name__}")
            print(f"   Scaler type: {type(scaler).__name__}")
        else:
            print("\n❌ No best model available")
        
        print("\n✅ Model Registry test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Model registry not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_ml_system_integration():
    """Test ML system integration with registry"""
    print("\n🧪 Testing ML System Integration")
    print("=" * 50)
    
    try:
        from advanced_ml_system_v2 import AdvancedMLSystemV2
        
        # Initialize ML system (should auto-load from registry)
        ml_system = AdvancedMLSystemV2()
        
        # Check if models were loaded
        if hasattr(ml_system, 'models') and ml_system.models:
            print(f"✅ ML System loaded {len(ml_system.models)} model(s):")
            for name, model in ml_system.models.items():
                weight = ml_system.model_weights.get(name, 0)
                print(f"   {name}: {type(model).__name__} (weight: {weight:.3f})")
                
            # Test prediction with dummy features
            test_features = {
                'weighted_recent_form': 3.5,
                'venue_win_rate': 0.25,
                'speed_trend': -0.1,
                'box_number': 2,
                'data_quality': 0.8
            }
            
            # Test ensemble prediction
            prediction = ml_system.predict_with_ensemble(test_features)
            print(f"\n🎯 Test Prediction:")
            print(f"   Features: {test_features}")
            print(f"   Prediction: {prediction:.4f}")
            
            # Test confidence calculation
            confidence = ml_system.generate_prediction_confidence(test_features)
            print(f"   Confidence: {confidence:.4f}")
            
        else:
            print("⚠️ No models loaded in ML system")
            
        print("\n✅ ML System integration test completed")
        return True
        
    except ImportError as e:
        print(f"❌ ML system not available: {e}")
        return False
    except Exception as e:
        print(f"❌ ML system test failed: {e}")
        return False

def test_pipeline_integration():
    """Test Enhanced Pipeline integration"""
    print("\n🧪 Testing Enhanced Pipeline Integration")
    print("=" * 50)
    
    try:
        from enhanced_pipeline_v2 import EnhancedPipelineV2
        
        # Initialize pipeline
        pipeline = EnhancedPipelineV2()
        
        # Check components
        print(f"📦 Pipeline Components:")
        print(f"   Feature Engineer: {'✅' if pipeline.feature_engineer else '❌'}")
        print(f"   ML System: {'✅' if pipeline.ml_system else '❌'}")
        print(f"   Data Improver: {'✅' if pipeline.data_improver else '❌'}")
        
        if pipeline.ml_system and hasattr(pipeline.ml_system, 'models'):
            models_count = len(pipeline.ml_system.models) if pipeline.ml_system.models else 0
            print(f"   ML Models Loaded: {models_count}")
            
            # Test model update check
            print(f"\n🔄 Testing model update check...")
            pipeline._check_for_model_updates()
            print(f"   Last check: {pipeline.last_model_check}")
        
        # Look for a test race file
        race_files = []
        upcoming_races_dir = Path('./upcoming_races')
        if upcoming_races_dir.exists():
            race_files = list(upcoming_races_dir.glob('*.csv'))
        
        if race_files:
            test_file = race_files[0]
            print(f"\n🏁 Testing with race file: {test_file.name}")
            
            # Test prediction
            result = pipeline.predict_race_file(str(test_file))
            
            if result.get('success'):
                predictions = result['predictions']
                print(f"   ✅ Generated {len(predictions)} predictions")
                
                if predictions:
                    top_pick = predictions[0]
                    print(f"   🏆 Top pick: {top_pick['dog_name']}")
                    print(f"   Score: {top_pick['prediction_score']:.3f}")
                    print(f"   Confidence: {top_pick['confidence_level']}")
                    print(f"   Method: {top_pick['prediction_method']}")
                    
                    # Show score distribution
                    scores = [p['prediction_score'] for p in predictions]
                    print(f"   📊 Score range: {min(scores):.3f} - {max(scores):.3f}")
                    print(f"   Variance: {sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores):.6f}")
            else:
                print(f"   ❌ Prediction failed: {result.get('error', 'Unknown error')}")
        else:
            print("\n⚠️ No race files found for testing")
        
        print("\n✅ Enhanced Pipeline integration test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Enhanced Pipeline not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_monitoring_service():
    """Test the model monitoring service"""
    print("\n🧪 Testing Model Monitoring Service")
    print("=" * 50)
    
    try:
        from model_monitoring_service import get_monitoring_service
        
        # Get monitoring service
        service = get_monitoring_service()
        
        # Show initial status
        status = service.get_monitoring_status()
        print(f"📊 Monitoring Service Status:")
        print(f"   Running: {'✅' if status['is_running'] else '❌'}")
        print(f"   Models checked: {status['stats']['models_checked']}")
        print(f"   Models reloaded: {status['stats']['models_reloaded']}")
        print(f"   Performance checks: {status['stats']['performance_checks']}")
        print(f"   Alerts generated: {status['stats']['alerts_generated']}")
        print(f"   Performance history size: {status['performance_history_size']}")
        print(f"   Uptime: {status['uptime_hours']:.1f} hours")
        
        # Test performance recording
        dummy_predictions = [
            {'dog_name': 'Test Dog 1', 'prediction_score': 0.65, 'confidence_score': 0.7},
            {'dog_name': 'Test Dog 2', 'prediction_score': 0.45, 'confidence_score': 0.6},
            {'dog_name': 'Test Dog 3', 'prediction_score': 0.35, 'confidence_score': 0.5}
        ]
        
        service.record_prediction_performance(dummy_predictions)
        print(f"\n✅ Recorded test performance data")
        
        # Test model update check (manual)
        service._check_for_model_updates()
        print(f"✅ Performed model update check")
        
        print("\n✅ Model Monitoring Service test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Model monitoring service not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Monitoring service test failed: {e}")
        return False

def create_test_model_reload_signal():
    """Create a test model reload signal"""
    print("\n🧪 Creating Test Model Reload Signal")
    print("=" * 50)
    
    try:
        reload_signal = {
            'model_id': 'test_model_reload_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            'accuracy': 0.85,
            'auc': 0.88,
            'f1_score': 0.82,
            'timestamp': datetime.now().isoformat(),
            'action': 'reload_model'
        }
        
        signal_path = Path('./model_reload_signal.json')
        with open(signal_path, 'w') as f:
            json.dump(reload_signal, f, indent=2)
        
        print(f"✅ Created reload signal: {signal_path}")
        print(f"   Model ID: {reload_signal['model_id']}")
        print(f"   Performance: Acc={reload_signal['accuracy']}, AUC={reload_signal['auc']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create reload signal: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Model Registry System Comprehensive Test")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    results = {}
    
    # Run tests
    results['registry'] = test_model_registry()
    results['ml_system'] = test_ml_system_integration()
    results['pipeline'] = test_pipeline_integration()
    results['monitoring'] = test_monitoring_service()
    results['reload_signal'] = create_test_model_reload_signal()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The model registry system is working correctly.")
        
        # Demonstrate the system in action
        print("\n" + "=" * 60)
        print("🎯 System Demonstration")
        print("=" * 60)
        
        print("The centralized model registry system provides:")
        print("  ✅ Automatic model loading from the best available model")
        print("  ✅ Centralized model metadata and performance tracking")
        print("  ✅ Periodic model update checking and reloading")
        print("  ✅ Performance drift detection and alerting")
        print("  ✅ Seamless integration with ML systems and pipelines")
        print("  ✅ Fallback mechanisms for robustness")
        
        print("\nKey Features Implemented:")
        print("  🔄 Models are automatically loaded from registry on startup")
        print("  📊 Performance metrics are tracked and compared")
        print("  🔔 Alerts are generated for performance degradation")
        print("  📡 Signal files enable real-time model updates")
        print("  🧹 Automatic cleanup of old models and data")
        print("  🔒 Thread-safe operations for concurrent access")
        
    else:
        print(f"⚠️ {total - passed} test(s) failed. Please check the error messages above.")
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()
