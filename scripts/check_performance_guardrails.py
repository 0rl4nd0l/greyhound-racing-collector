#!/usr/bin/env python3
"""
Check performance guardrails for the trained model.
"""
import json
import sys

def main():
    # Load metrics
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
    
    auc = metrics["auc"]
    latency = metrics["prediction_latency"]
    
    print(f"Checking performance guardrails:")
    print(f"AUC: {auc:.4f}")
    print(f"Latency: {latency:.4f}s")
    
    # Define thresholds
    min_auc = 0.7  # Minimum acceptable AUC
    max_latency = 1.0  # Maximum acceptable latency (1 second)
    
    # Check guardrails
    failed = False
    
    if auc < min_auc:
        print(f"❌ FAILED: AUC {auc:.4f} is below minimum threshold {min_auc}")
        failed = True
    else:
        print(f"✅ PASSED: AUC {auc:.4f} meets minimum threshold {min_auc}")
    
    if latency > max_latency:
        print(f"❌ FAILED: Prediction latency {latency:.4f}s exceeds maximum threshold {max_latency}s")
        failed = True
    else:
        print(f"✅ PASSED: Prediction latency {latency:.4f}s is within acceptable range")
    
    if failed:
        print("Pipeline failed due to performance guardrails")
        sys.exit(1)
    else:
        print("All performance guardrails passed!")

if __name__ == "__main__":
    main()
