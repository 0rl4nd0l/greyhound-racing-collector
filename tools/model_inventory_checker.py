#!/usr/bin/env python3
"""
Model Inventory and Integrity Checker
=====================================

This script discovers all ML models in the repository, attempts to load them,
and generates a comprehensive inventory report with metadata.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import traceback
import warnings
warnings.filterwarnings("ignore")

import joblib
import pickle
import numpy as np
import pandas as pd

def calculate_sha256(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception:
        return ""

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        return 0.0

def discover_model_files() -> List[str]:
    """Discover model artifact files under known model directories only.

    This avoids scanning caches and non-model pickles.
    """
    model_extensions = [".joblib", ".pkl", ".h5", ".pt", ".pth"]
    model_files = []
    
    # Restrict to dedicated model directories
    search_paths = [
        "model_registry/models",
        "ml_models_v4",
        "ml_models",
        "comprehensive_trained_models",
        "advanced_models"
    ]
    
    excluded_dirs = {".git", ".cache", ".pytest_cache", "tests", "test-results", "artifacts", "node_modules"}
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                # prune excluded dirs
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                for file in files:
                    if any(file.endswith(ext) for ext in model_extensions):
                        full_path = os.path.join(root, file)
                        model_files.append(full_path)
    
    return sorted(list(set(model_files)))

def attempt_model_load(file_path: str) -> Dict[str, Any]:
    """Attempt to load a model and extract metadata."""
    result = {
        "path": file_path,
        "load_success": False,
        "load_error": None,
        "model_type": "unknown",
        "sklearn_version": None,
        "feature_count": 0,
        "feature_names": [],
        "has_predict_method": False,
        "has_predict_proba_method": False,
        "is_pipeline": False,
        "pipeline_steps": [],
        "model_attributes": []
    }
    
    try:
        # Attempt joblib load first
        if file_path.endswith(('.joblib', '.pkl')):
            try:
                model = joblib.load(file_path)
                result["load_success"] = True
                result["loader"] = "joblib"
            except Exception as e:
                # Try pickle as fallback
                try:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                    result["load_success"] = True
                    result["loader"] = "pickle"
                except Exception as e2:
                    result["load_error"] = str(e)
                    return result
        else:
            result["load_error"] = "Unsupported file format"
            return result
        
        # Extract model information
        result["model_type"] = type(model).__name__
        
        # Check for sklearn attributes
        if hasattr(model, "__version__"):
            result["sklearn_version"] = getattr(model, "__version__")
        
        # Check for feature information
        if hasattr(model, "feature_names_in_"):
            result["feature_names"] = list(model.feature_names_in_)
            result["feature_count"] = len(model.feature_names_in_)
        elif hasattr(model, "feature_names"):
            result["feature_names"] = list(model.feature_names)
            result["feature_count"] = len(model.feature_names)
        elif hasattr(model, "n_features_"):
            result["feature_count"] = model.n_features_
        elif hasattr(model, "n_features_in_"):
            result["feature_count"] = model.n_features_in_
        
        # Check for methods
        result["has_predict_method"] = hasattr(model, "predict")
        result["has_predict_proba_method"] = hasattr(model, "predict_proba")
        
        # Check if it's a pipeline
        if hasattr(model, "steps"):
            result["is_pipeline"] = True
            result["pipeline_steps"] = [step[0] for step in model.steps]
        
        # Get all attributes
        result["model_attributes"] = [attr for attr in dir(model) 
                                    if not attr.startswith('_') and not callable(getattr(model, attr))]
        
        # Try to get a sample prediction to test functionality
        if result["has_predict_method"] and result["feature_count"] > 0:
            try:
                # Create dummy data
                if result["feature_names"]:
                    dummy_data = pd.DataFrame([[0.5] * len(result["feature_names"])], 
                                            columns=result["feature_names"])
                else:
                    dummy_data = np.array([[0.5] * result["feature_count"]])
                
                pred = model.predict(dummy_data)
                result["sample_prediction"] = str(pred)
                result["prediction_test"] = "success"
                
                if result["has_predict_proba_method"]:
                    proba = model.predict_proba(dummy_data)
                    result["sample_proba"] = str(proba)
                    
            except Exception as e:
                result["prediction_test"] = f"failed: {str(e)}"
        
    except Exception as e:
        result["load_error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    return result

def load_registry_metadata() -> Dict[str, Any]:
    """Load model registry metadata if available."""
    registry_path = "model_registry/model_index.json"
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load registry metadata: {e}")
    return {}

def generate_inventory_report():
    """Generate comprehensive model inventory report."""
    print("🔍 Discovering model files...")
    model_files = discover_model_files()
    print(f"Found {len(model_files)} candidate artifact files")
    
    # Load registry metadata
    registry_metadata = load_registry_metadata()
    
    inventory = {
        "scan_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files_found": len(model_files),
        "models": []
    }
    
    print("🧪 Testing model loading...")
    successful_loads = 0
    true_models = 0
    
    for i, file_path in enumerate(model_files):
        print(f"  [{i+1}/{len(model_files)}] Testing {file_path}...")
        
        # Basic file info
        file_info = {
            "file_hash": calculate_sha256(file_path),
            "file_size_mb": round(get_file_size_mb(file_path), 3),
            "modification_time": time.ctime(os.path.getmtime(file_path))
        }
        
        # Attempt to load model
        model_info = attempt_model_load(file_path)
        
        # Merge file info with model info
        model_info.update(file_info)
        
        # Check if model is in registry
        model_name = os.path.basename(file_path).replace('.joblib', '').replace('.pkl', '')
        model_info["in_registry"] = any(
            model_name in key or file_path in metadata.get("model_file_path", "")
            for key, metadata in registry_metadata.items()
        )
        
        # Add registry metadata if found
        for key, metadata in registry_metadata.items():
            if model_name in key or file_path in metadata.get("model_file_path", ""):
                model_info["registry_metadata"] = metadata
                break
        
        if model_info["load_success"]:
            successful_loads += 1
            # Only report true model objects (have predict or predict_proba)
            if model_info.get("has_predict_method") or model_info.get("has_predict_proba_method"):
                true_models += 1
                print(f"    ✅ Model: {model_info['model_type']} with {model_info['feature_count']} features")
                inventory["models"].append(model_info)
            else:
                print("    ↪️  Skipped non-model artifact (no predict methods)")
        else:
            print(f"    ❌ Failed: {model_info.get('load_error', 'Unknown error')}")
    
    inventory["successful_loads"] = successful_loads
    inventory["true_models"] = true_models
    inventory["load_success_rate"] = successful_loads / len(model_files) if model_files else 0
    
    # Save inventory report
    output_path = "artifacts/model_inventory.json"
    with open(output_path, 'w') as f:
        json.dump(inventory, f, indent=2, default=str)
    
    print(f"\n📊 Inventory Summary:")
    print(f"   Total files: {len(model_files)}")
    print(f"   Successful loads: {successful_loads}")
    print(f"   Load success rate: {inventory['load_success_rate']:.1%}")
    print(f"   Report saved: {output_path}")
    
    # Identify best candidates among true models
    working_models = [m for m in inventory["models"] if m.get("load_success")]
    pipeline_models = [m for m in working_models if m.get("is_pipeline")]
    registry_models = [m for m in working_models if m.get("in_registry")]
    
    print(f"\n🎯 Model Categories:")
    print(f"   True models discovered: {len(working_models)}")
    print(f"   Pipeline models: {len(pipeline_models)}")
    print(f"   Registry models: {len(registry_models)}")
    
    if working_models:
        print(f"\n✨ Top Model Candidates:")
        # Sort by registry presence, pipeline status, and feature count
        candidates = sorted(working_models, 
                          key=lambda x: (x["in_registry"], x["is_pipeline"], x["feature_count"]), 
                          reverse=True)
        
        for i, model in enumerate(candidates[:5]):
            status_indicators = []
            if model["in_registry"]: status_indicators.append("📋 Registry")
            if model["is_pipeline"]: status_indicators.append("🔧 Pipeline")
            if model["has_predict_proba_method"]: status_indicators.append("📊 Proba")
            
            print(f"   {i+1}. {model['path']}")
            print(f"      Type: {model['model_type']}, Features: {model['feature_count']}")
            print(f"      Status: {' | '.join(status_indicators) if status_indicators else 'Basic Model'}")
    
    return inventory

if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir("/Users/test/Desktop/greyhound_racing_collector")
    
    # Generate inventory
    inventory = generate_inventory_report()
