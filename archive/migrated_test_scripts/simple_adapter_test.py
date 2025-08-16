#!/usr/bin/env python3
"""
Simple Direct Test for Prediction Adapters
==========================================

Tests the adapter classes directly without triggering the full system.
"""

import logging
import tempfile
import os

# Configure basic logging
logging.basicConfig(level=logging.ERROR)  # Suppress INFO logs

def create_test_csv():
    """Create a simple test CSV file with proper filename format."""
    content = """Dog Name,BOX,WGT,SP,TRAINER,G,DIST,PIR
1. FAST RUNNER,1,30.5,2.50,J SMITH,G5,500,85
2. QUICK STAR,2,31.0,3.20,M JONES,G5,500,82
3. SPEEDY DOG,3,29.8,4.10,S BROWN,G5,500,78
4. RAPID HOUND,4,30.2,5.50,P WILSON,G5,500,75"""
    
    # Create properly named test file for V4 processing
    test_filename = "Race 1 - TEST - 04 August 2025.csv"
    with open(test_filename, 'w') as f:
        f.write(content)
    
    return test_filename

def test_adapter_initialization():
    """Test basic adapter initialization."""
    print("🧪 Testing Adapter Initialization")
    print("-" * 40)
    
    try:
        from prediction_adapters import V3Adapter, V3SAdapter, V4Adapter
        
        # Test V3Adapter
        try:
            v3 = V3Adapter()
            print("✅ V3Adapter: Initialization OK")
        except Exception as e:
            print(f"❌ V3Adapter: Initialization failed - {e}")
        
        # Test V3SAdapter  
        try:
            v3s = V3SAdapter()
            print("✅ V3SAdapter: Initialization OK")
        except Exception as e:
            print(f"❌ V3SAdapter: Initialization failed - {e}")
        
        # Test V4Adapter
        try:
            v4 = V4Adapter()
            print("✅ V4Adapter: Initialization OK")
        except Exception as e:
            print(f"❌ V4Adapter: Initialization failed - {e}")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")

def test_standardized_result():
    """Test the StandardizedResult helper class."""
    print("\n🧪 Testing StandardizedResult Helper")
    print("-" * 40)
    
    try:
        from prediction_adapters import StandardizedResult
        
        # Test create_result
        test_predictions = [
            {"dog": "TEST DOG", "raw_prob": 0.3, "win_prob_norm": 0.0}
        ]
        
        result = StandardizedResult.create_result(
            race_id="test_race",
            predictions=test_predictions,
            metadata={"adapter": "test", "method": "test"}
        )
        
        # Check structure
        required_keys = ["race_id", "predictions", "metadata"]
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
        else:
            print("✅ StandardizedResult.create_result: Structure OK")
        
        # Test normalization
        test_predictions = [
            {"dog": "DOG1", "raw_prob": 0.2, "win_prob_norm": 0.0},
            {"dog": "DOG2", "raw_prob": 0.4, "win_prob_norm": 0.0},
            {"dog": "DOG3", "raw_prob": 0.6, "win_prob_norm": 0.0}
        ]
        
        normalized = StandardizedResult.normalize_probabilities(test_predictions)
        norm_sum = sum(pred["win_prob_norm"] for pred in normalized)
        
        if 0.95 <= norm_sum <= 1.05:
            print("✅ StandardizedResult.normalize_probabilities: Normalization OK")
        else:
            print(f"❌ Normalization failed: sum = {norm_sum:.3f}")
            
    except Exception as e:
        print(f"❌ StandardizedResult test failed: {e}")

def test_simple_prediction():
    """Test a simple prediction with minimal setup."""
    print("\n🧪 Testing Simple Prediction")
    print("-" * 40)
    
    # Create test CSV
    test_csv = create_test_csv()
    print(f"📁 Created test CSV: {test_csv}")
    
    try:
        from prediction_adapters import V4Adapter
        
        # Test V4Adapter (most likely to work standalone)
        v4 = V4Adapter()
        
        # Make prediction
        result = v4.predict_race(test_csv)
        
        # Check result
        if result and "metadata" in result:
            success = result["metadata"].get("success", False)
            if success:
                print("✅ V4Adapter prediction: SUCCESS")
                print(f"   Race ID: {result.get('race_id', 'Unknown')}")
                print(f"   Predictions: {len(result.get('predictions', []))}")
            else:
                error = result["metadata"].get("error", "Unknown error")
                print(f"❌ V4Adapter prediction failed: {error}")
        else:
            print("❌ V4Adapter returned invalid result structure")
            
    except Exception as e:
        print(f"❌ Simple prediction test failed: {e}")
    
    finally:
        # Cleanup
        try:
            os.unlink(test_csv)
            print(f"🗑️ Cleaned up test file")
        except:
            pass

def main():
    """Run all tests."""
    print("🚀 Simple Prediction Adapter Tests")
    print("=" * 50)
    
    test_adapter_initialization()
    test_standardized_result()
    test_simple_prediction()
    
    print("\n✅ Simple tests completed!")

if __name__ == "__main__":
    main()
