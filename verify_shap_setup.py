#!/usr/bin/env python3
"""
SHAP Setup Verification
======================

Simple script to verify SHAP explainability has been properly set up.
"""

import os
import sys
from pathlib import Path


def verify_shap_installation():
    """Verify SHAP is installed and working."""
    print("🔍 Verifying SHAP Installation...")

    try:
        import shap

        print(f"✅ SHAP installed successfully (version: {shap.__version__})")
        return True
    except ImportError as e:
        print(f"❌ SHAP not installed: {e}")
        print("   Install with: pip install shap")
        return False


def verify_shap_module():
    """Verify our SHAP explainer module exists and imports."""
    print("\n🔍 Verifying SHAP Explainer Module...")

    try:
        from shap_explainer import SHAPExplainer, get_shap_values

        print("✅ SHAP explainer module imports successfully")
        return True
    except ImportError as e:
        print(f"❌ SHAP explainer module import failed: {e}")
        return False


def verify_models_directory():
    """Verify models directory exists and has expected structure."""
    print("\n🔍 Verifying Models Directory Structure...")

    models_dir = Path("models")
    if not models_dir.exists():
        print(f"❌ Models directory does not exist at: {models_dir.absolute()}")
        return False

    print(f"✅ Models directory exists at: {models_dir.absolute()}")

    # Check for cached explainers
    explainer_files = list(models_dir.glob("shap_explainer_*.joblib"))
    if explainer_files:
        print(f"✅ Found {len(explainer_files)} cached SHAP explainers:")
        for file in explainer_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name} ({size_mb:.1f} MB)")
    else:
        print("⚠️  No cached SHAP explainers found (will be created on first use)")

    return True


def verify_model_registry():
    """Verify model registry has models available."""
    print("\n🔍 Verifying Model Registry...")

    registry_dir = Path("model_registry/models")
    if not registry_dir.exists():
        print(
            f"❌ Model registry directory does not exist at: {registry_dir.absolute()}"
        )
        return False

    model_files = list(registry_dir.glob("*.joblib"))
    if not model_files:
        print(f"⚠️  No model files found in registry at: {registry_dir.absolute()}")
        print("   Train some models first using the ML system")
        return False

    print(f"✅ Found {len(model_files)} model files in registry:")
    for file in model_files[:5]:  # Show first 5
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.1f} MB)")

    if len(model_files) > 5:
        print(f"   ... and {len(model_files) - 5} more")

    return True


def verify_integration():
    """Verify integration with ML system and prediction pipeline."""
    print("\n🔍 Verifying Integration...")

    # Check ML System V3 integration
    try:
        from ml_system_v3 import SHAP_EXPLAINABILITY_AVAILABLE

        if SHAP_EXPLAINABILITY_AVAILABLE:
            print("✅ ML System V3 has SHAP explainability enabled")
        else:
            print("⚠️  ML System V3 does not have SHAP explainability enabled")
    except ImportError:
        print("❌ Could not import ML System V3")
        return False

    # Check Prediction Pipeline V3 integration
    try:
        with open("prediction_pipeline_v3.py", "r") as f:
            content = f.read()
            if "shap_explainer" in content and "explainability" in content:
                print("✅ Prediction Pipeline V3 has SHAP integration")
            else:
                print("⚠️  Prediction Pipeline V3 may not have SHAP integration")
    except FileNotFoundError:
        print("❌ Could not find prediction_pipeline_v3.py")
        return False

    return True


def verify_test_files():
    """Verify test files are in place."""
    print("\n🔍 Verifying Test Files...")

    test_file = Path("tests/test_shap_integration.py")
    if test_file.exists():
        print(f"✅ Test file exists at: {test_file}")
    else:
        print(f"⚠️  Test file not found at: {test_file}")

    doc_file = Path("docs/SHAP_EXPLAINABILITY.md")
    if doc_file.exists():
        print(f"✅ Documentation exists at: {doc_file}")
    else:
        print(f"⚠️  Documentation not found at: {doc_file}")

    return True


def main():
    """Run all verification checks."""
    print("🚀 SHAP Explainability Setup Verification")
    print("=" * 50)

    results = []

    # Run all verification checks
    results.append(verify_shap_installation())
    results.append(verify_shap_module())
    results.append(verify_models_directory())
    results.append(verify_model_registry())
    results.append(verify_integration())
    results.append(verify_test_files())

    # Summary
    print("\n" + "=" * 50)
    print("📊 Verification Summary:")
    print(f"   Checks run: {len(results)}")
    print(f"   Passed: {sum(results)}")
    print(f"   Failed/Warnings: {len(results) - sum(results)}")

    if all(results):
        print("\n🎉 All verifications passed! SHAP explainability is properly set up.")
        print("\n💡 Next steps:")
        print("   1. Train ML models if none exist in model registry")
        print("   2. Run tests: python tests/test_shap_integration.py")
        print("   3. Make predictions to see explainability in action")
        return 0
    else:
        print("\n⚠️  Some verifications failed. Check the output above for details.")
        print("\n💡 Common fixes:")
        print("   - Install SHAP: pip install shap")
        print("   - Train ML models using the ML system")
        print("   - Ensure all required files are in place")
        return 1


if __name__ == "__main__":
    sys.exit(main())
