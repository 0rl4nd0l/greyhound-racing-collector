#!/usr/bin/env python3
"""
Test script to verify successful imports of ML libraries
"""
import traceback


def test_import(library_name, import_statement):
    """Test importing a library and log any errors"""
    print(f"\n{'='*50}")
    print(f"Testing import: {library_name}")
    print(f"{'='*50}")

    try:
        exec(import_statement)
        print(f"✅ SUCCESS: {library_name} imported successfully")

        # Additional version check if possible
        if library_name == "openai":
            import openai

            print(f"   Version: {openai.__version__}")
        elif library_name == "xgboost":
            import xgboost as xgb

            print(f"   Version: {xgb.__version__}")
        elif library_name == "shap":
            import shap

            print(f"   Version: {shap.__version__}")
        elif library_name == "evidently":
            import evidently

            print(
                f"   Version: {evidently.__version__ if hasattr(evidently, '__version__') else 'Version info not accessible'}"
            )

    except ImportError as e:
        print(f"❌ IMPORT ERROR: {library_name}")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
    except Exception as e:
        print(f"⚠️  OTHER ERROR: {library_name}")
        print(f"   Error: {str(e)}")
        traceback.print_exc()


def main():
    """Run import tests for all ML libraries"""
    print("ML Libraries Import Test")
    print("=" * 60)

    # Test libraries specified in the task
    libraries_to_test = [
        ("openai", "import openai"),
        ("xgboost", "import xgboost as xgb"),
        ("shap", "import shap"),
        ("evidently", "import evidently"),
    ]

    # Additional tests for GPU/binary compilation issues
    print("\n" + "=" * 60)
    print("ADDITIONAL TESTS FOR BINARY/GPU COMPILATION")
    print("=" * 60)

    # Test XGBoost GPU support (common issue)
    try:
        import xgboost as xgb

        print("\n🔍 XGBoost GPU Support Test:")
        # Try to check if GPU is available (won't fail if not available)
        try:
            # This will show available devices without failing
            print(f"   XGBoost build info available: {hasattr(xgb, 'build_info')}")
            if hasattr(xgb, "build_info"):
                print(f"   Build info: {xgb.build_info()}")
        except Exception as e:
            print(f"   GPU check: {str(e)} (This is often normal if no GPU)")
    except Exception as e:
        print(f"❌ XGBoost GPU test failed: {str(e)}")

    # Test SHAP with simple functionality
    try:
        import shap

        print("\n🔍 SHAP Basic Functionality Test:")
        print(f"   SHAP explainers available: {hasattr(shap, 'Explainer')}")
        print(f"   SHAP plots available: {hasattr(shap, 'plots')}")
    except Exception as e:
        print(f"❌ SHAP functionality test failed: {str(e)}")

    # Run all import tests
    for lib_name, import_stmt in libraries_to_test:
        test_import(lib_name, import_stmt)

    print(f"\n{'='*60}")
    print("Import test completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
