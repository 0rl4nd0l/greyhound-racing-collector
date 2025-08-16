#!/usr/bin/env python3
"""
Integration Example: Adding Sanity Checks to Existing Pipeline
==============================================================

This script shows exactly how to integrate sanity checks into your 
existing prediction pipeline files with minimal code changes.
"""

import logging
from pathlib import Path
from prediction_sanity_integration import apply_sanity_checks_to_response, with_sanity_checks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_ml_system_v3_integration():
    """
    Show how to integrate with ml_system_v3.py
    """
    print("🔧 ML System V3 Integration")
    print("=" * 40)
    
    integration_steps = """
1. In ml_system_v3.py, add this import at the top:
   from prediction_sanity_integration import apply_sanity_checks_to_response

2. Modify the predict() method around line 378:

   # Original code:
   result = {
       "win_probability": float(calibrated_probs['calibrated_win_prob']),
       "place_probability": float(calibrated_probs['calibrated_place_prob']),
       "raw_win_probability": float(raw_win_prob),
       "raw_place_probability": float(raw_place_prob),
       "confidence": float(confidence),
       "model_info": self.model_info.get("model_type", "unknown"),
       # ... other fields
   }
   return result

   # Modified code:
   result = {
       "win_probability": float(calibrated_probs['calibrated_win_prob']),
       "place_probability": float(calibrated_probs['calibrated_place_prob']),
       "raw_win_probability": float(raw_win_prob),
       "raw_place_probability": float(raw_place_prob),
       "confidence": float(confidence),
       "model_info": self.model_info.get("model_type", "unknown"),
       # ... other fields
   }
   
   # Apply sanity checks if enabled
   if hasattr(self, 'enable_sanity_checks') and self.enable_sanity_checks:
       response = {'predictions': [result]}
       response = apply_sanity_checks_to_response(response)
       if response.get('sanity_check', {}).get('fixes_applied'):
           logger.info("Sanity check fixes applied to ML prediction")
       result = response['predictions'][0]
   
   return result

3. Enable sanity checks in the constructor (__init__):
   self.enable_sanity_checks = True  # Add this line
"""
    
    print(integration_steps)

def show_prediction_pipeline_v3_integration():
    """
    Show how to integrate with prediction_pipeline_v3.py
    """
    print("\n🔧 Prediction Pipeline V3 Integration")
    print("=" * 40)
    
    integration_steps = """
1. In prediction_pipeline_v3.py, add this import at the top:
   from prediction_sanity_integration import with_sanity_checks

2. Add the decorator to the predict_race_file method around line 149:

   # Original method signature:
   def predict_race_file(self, race_file_path: str, enhancement_level="full") -> dict:

   # Modified method signature:
   @with_sanity_checks(auto_fix=True)
   def predict_race_file(self, race_file_path: str, enhancement_level="full") -> dict:

   That's it! The decorator will automatically:
   - Validate all predictions in the response
   - Apply fixes if issues are found
   - Add sanity check results to the response
   - Log any issues or fixes applied

3. The response will now include a 'sanity_check' section with:
   {
     "success": True,
     "predictions": [...],  # Fixed predictions if needed
     "sanity_check": {
       "validation_performed": True,
       "issues_found": 0,
       "fixes_applied": False,
       "passed_checks": ["Probability range validation", ...],
       "flags": []
     }
   }
"""
    
    print(integration_steps)

def show_app_py_integration():
    """
    Show how to integrate with app.py Flask routes
    """
    print("\n🔧 Flask App.py Integration")
    print("=" * 40)
    
    integration_steps = """
1. In app.py, add this import at the top:
   from prediction_sanity_integration import apply_sanity_checks_to_response

2. Find your prediction routes (e.g., /predict, /predict_race) and modify them:

   # Original route:
   @app.route('/predict_race', methods=['POST'])
   def predict_race():
       # ... existing prediction logic ...
       
       response = {
           "success": True,
           "predictions": predictions,
           "race_info": race_info
       }
       return jsonify(response)

   # Modified route:
   @app.route('/predict_race', methods=['POST'])
   def predict_race():
       # ... existing prediction logic ...
       
       response = {
           "success": True,
           "predictions": predictions,
           "race_info": race_info
       }
       
       # Apply sanity checks before returning
       response = apply_sanity_checks_to_response(response)
       
       return jsonify(response)

3. The API response will now include sanity check information for debugging.
"""
    
    print(integration_steps)

def create_patch_files():
    """
    Create actual patch files that can be applied to existing files
    """
    print("\n📄 Creating Integration Patch Files")
    print("=" * 40)
    
    # ML System V3 patch
    ml_patch = '''--- ml_system_v3.py.orig
+++ ml_system_v3.py
@@ -1,4 +1,5 @@
 # ... existing imports ...
+from prediction_sanity_integration import apply_sanity_checks_to_response
 
 class MLSystemV3:
     def __init__(self, db_path="greyhound_racing_data.db"):
@@ -10,6 +11,7 @@
         self.challenger_model_info = {}
         
         # ... existing initialization code ...
+        self.enable_sanity_checks = True  # Enable sanity checks
         
         self._try_load_latest_model()
 
@@ -380,6 +382,15 @@
             }
             
+            # Apply sanity checks if enabled
+            if hasattr(self, 'enable_sanity_checks') and self.enable_sanity_checks:
+                response = {'predictions': [result]}
+                response = apply_sanity_checks_to_response(response)
+                if response.get('sanity_check', {}).get('fixes_applied'):
+                    logger.info("Sanity check fixes applied to ML prediction")
+                result = response['predictions'][0]
+            
             return result
'''
    
    # Pipeline V3 patch
    pipeline_patch = '''--- prediction_pipeline_v3.py.orig
+++ prediction_pipeline_v3.py
@@ -1,4 +1,5 @@
 # ... existing imports ...
+from prediction_sanity_integration import with_sanity_checks
 
 class PredictionPipelineV3:
     # ... existing code ...
@@ -148,6 +149,7 @@
         print(f"  Overall: {available_systems}/5 systems available")
 
+    @with_sanity_checks(auto_fix=True)
     def predict_race_file(self, race_file_path: str, enhancement_level="full") -> dict:
         """Main prediction method with intelligent fallback and enhancement levels."""
'''
    
    # Save patch files
    patches_dir = Path("integration_patches")
    patches_dir.mkdir(exist_ok=True)
    
    with open(patches_dir / "ml_system_v3.patch", "w") as f:
        f.write(ml_patch)
    
    with open(patches_dir / "prediction_pipeline_v3.patch", "w") as f:
        f.write(pipeline_patch)
    
    print(f"✅ Patch files created in {patches_dir}/")
    print("   - ml_system_v3.patch")
    print("   - prediction_pipeline_v3.patch")

def main():
    """
    Main integration guide
    """
    print("🚀 SANITY CHECKS INTEGRATION GUIDE")
    print("=" * 50)
    
    print("""
This guide shows you exactly how to add sanity checks to your existing 
prediction pipeline with minimal code changes.

The integration provides:
✅ Automatic probability validation
✅ Range checking (0 ≤ prob ≤ 1)
✅ Softmax normalization
✅ Rank alignment fixes
✅ Duplicate rank detection
✅ NaN value handling
✅ Detailed logging
✅ Zero downtime deployment
""")
    
    show_ml_system_v3_integration()
    show_prediction_pipeline_v3_integration()
    show_app_py_integration()
    create_patch_files()
    
    print(f"""
🎯 NEXT STEPS:
1. Choose your integration method:
   - Decorator (@with_sanity_checks) - Easiest
   - Function call (apply_sanity_checks_to_response) - More control
   - Middleware (SanityCheckMiddleware) - For complex pipelines

2. Test with a single prediction first
3. Enable logging to monitor fixes applied
4. Deploy gradually to production

📁 All integration files are ready:
   - sanity_checks.py (core validation)
   - prediction_sanity_integration.py (integration helpers)
   - integration_patches/ (patch files)
   
🔍 For testing, run:
   python test_sanity_checks.py
   python comprehensive_sanity_demo.py
""")

if __name__ == "__main__":
    main()
