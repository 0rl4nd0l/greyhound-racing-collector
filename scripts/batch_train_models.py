#!/usr/bin/env python3
"""
Batch ML Training Script for Multiple Models

Trains multiple model types using the optimized pipeline, compares performance,
and selects the best performing model for deployment.

Usage:
    python scripts/batch_train_models.py
    python scripts/batch_train_models.py --models extratrees lgbm gb
    python scripts/batch_train_models.py --optimize --compare
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_optimized_v4 import OptimizedTrainingPipeline


def run_batch_training(model_types: List[str], optimize_hyperparams: bool = False,
                      cv_folds: int = 3) -> Dict[str, Any]:
    """Run batch training for multiple model types."""
    
    print("🚀 Starting batch training pipeline")
    print(f"📋 Models to train: {', '.join(model_types)}")
    
    # Initialize pipeline
    analytics_db = os.getenv("ANALYTICS_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or "greyhound_racing_data.db"
    staging_db = os.getenv("STAGING_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or "greyhound_racing_data_stage.db"
    
    pipeline = OptimizedTrainingPipeline(analytics_db, staging_db)
    
    results = {}
    best_model = None
    best_auc = 0.0
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"🏃‍♂️ Training {model_type.upper()} model...")
        print(f"{'='*60}")
        
        try:
            result = pipeline.train_model(
                model_type=model_type,
                optimize_hyperparams=optimize_hyperparams,
                cv_folds=cv_folds
            )
            
            results[model_type] = result
            
            if result["success"]:
                auc = result["metrics"]["auc"]
                if auc > best_auc:
                    best_auc = auc
                    best_model = {
                        "type": model_type,
                        "model_id": result["model_id"],
                        "auc": auc
                    }
                    
                print(f"✅ {model_type} completed successfully!")
                print(f"   📊 AUC: {auc:.3f}")
                print(f"   🎯 Accuracy: {result['metrics']['accuracy']:.3f}")
                if 'top1_accuracy' in result['metrics']:
                    print(f"   🏆 Top-1 Race Accuracy: {result['metrics']['top1_accuracy']:.3f}")
            else:
                print(f"❌ {model_type} failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {model_type} failed with exception: {e}")
            results[model_type] = {"success": False, "error": str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 BATCH TRAINING SUMMARY")
    print(f"{'='*60}")
    
    successful_models = [k for k, v in results.items() if v.get("success", False)]
    failed_models = [k for k, v in results.items() if not v.get("success", False)]
    
    print(f"✅ Successful: {len(successful_models)}/{len(model_types)}")
    print(f"❌ Failed: {len(failed_models)}/{len(model_types)}")
    
    if successful_models:
        print("\n🏆 Model Performance Ranking:")
        model_aucs = [(k, v["metrics"]["auc"]) for k, v in results.items() if v.get("success")]
        model_aucs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_type, auc) in enumerate(model_aucs, 1):
            marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"  {marker} {i}. {model_type}: {auc:.3f} AUC")
    
    if best_model:
        print(f"\n🎯 Best Model: {best_model['type']} (ID: {best_model['model_id']}, AUC: {best_model['auc']:.3f})")
    
    if failed_models:
        print(f"\n❌ Failed Models: {', '.join(failed_models)}")
    
    return {
        "summary": {
            "total_models": len(model_types),
            "successful": len(successful_models),
            "failed": len(failed_models),
            "best_model": best_model
        },
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Batch ML Training Pipeline")
    parser.add_argument("--models", nargs="+", 
                       choices=["extratrees", "lgbm", "hgb", "gb", "ensemble"],
                       default=["extratrees", "gb"],
                       help="Model types to train")
    parser.add_argument("--optimize", action="store_true",
                       help="Enable hyperparameter optimization")
    parser.add_argument("--cv-folds", type=int, default=3,
                       help="Cross-validation folds")
    parser.add_argument("--compare", action="store_true",
                       help="Include comparison with existing models")
    
    args = parser.parse_args()
    
    # Run batch training
    batch_results = run_batch_training(
        model_types=args.models,
        optimize_hyperparams=args.optimize,
        cv_folds=args.cv_folds
    )
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_training_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    print(f"\n📁 Full results saved to: {results_file}")
    
    return 0 if batch_results["summary"]["successful"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
