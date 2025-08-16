#!/usr/bin/env python3
"""
Example Usage of ConfigLoader in Other Modules
==============================================

This demonstrates how to use the config_loader module in your applications.
"""

from config_loader import get_config_loader, get_config, ConfigLoader

def main():
    print("🧪 Testing Config Loader Usage Patterns")
    print("=" * 50)
    
    # Method 1: Using convenience function
    print("\n📋 Method 1: Using get_config() convenience function")
    config = get_config()
    print(f"Min confidence from convenience function: {config['min_confidence']}")
    
    # Method 2: Using global config loader instance
    print("\n🔧 Method 2: Using get_config_loader() for global instance")
    loader = get_config_loader()
    print(f"Max calibration error: {loader.get_max_calibration_error()}")
    print(f"Form guide required fields: {loader.get_form_guide_required_fields()}")
    
    # Method 3: Creating a custom instance
    print("\n⚙️ Method 3: Creating custom instance with specific settings")
    custom_loader = ConfigLoader(
        config_file='config/defaults.yaml',
        parse_cli_args=False,  # Don't parse CLI args for this instance
        cli_args=['--drift-window-days', '60']  # Use specific test args
    )
    print(f"Custom drift window: {custom_loader.get_drift_window_days()}")
    
    # Method 4: Direct parameter access
    print("\n🎯 Method 4: Using specific getter methods")
    print(f"Imbalance ratio warning: {loader.get_imbalance_ratio_warn()}")
    print(f"Generic get method: {loader.get('min_confidence', 'not found')}")
    
    # Method 5: Integration example for a prediction system
    print("\n🤖 Method 5: Example integration in prediction system")
    def make_prediction_with_config():
        config_loader = get_config_loader()
        
        # Use configuration in your prediction logic
        min_conf = config_loader.get_min_confidence()
        max_cal_error = config_loader.get_max_calibration_error()
        
        print(f"Making prediction with min_confidence={min_conf}, max_calibration_error={max_cal_error}")
        
        # Mock prediction logic
        mock_confidence = 0.75
        mock_calibration_error = 0.08
        
        if mock_confidence >= min_conf and mock_calibration_error <= max_cal_error:
            return {"prediction": "Winner", "confidence": mock_confidence}
        else:
            return {"prediction": "Skip", "reason": "Below confidence threshold"}
    
    result = make_prediction_with_config()
    print(f"Prediction result: {result}")
    
    print("\n✅ Config loader usage demonstration completed!")

if __name__ == "__main__":
    main()
