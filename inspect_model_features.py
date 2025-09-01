#!/usr/bin/env python3
"""
Inspect a serialized sklearn pipeline/model to extract the exact feature names it expects
and save a contract JSON with an ordered feature list.

Usage:
  python inspect_model_features.py --model-path ./model_registry/models/<artifact>.joblib \
    [--output ./docs/model_contracts/<contract_name>.json]

Defaults:
  - If --output is omitted, a file will be written under docs/model_contracts using the
    model artifact stem, e.g., V4_GradientBoosting_..._model.json -> V4_GradientBoosting_..._model.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import joblib
    import pandas as pd

    print("✅ Required libraries imported successfully")
except ImportError as e:
    print(f"❌ Failed to import required libraries: {e}")
    sys.exit(1)


def _infer_default_output_path(model_path: str) -> Path:
    stem = Path(model_path).name
    # write contracts next to docs/model_contracts
    out_dir = Path("./docs/model_contracts")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use the artifact filename (without extension) as contract name + .json
    return out_dir / f"{Path(stem).with_suffix('').name}.json"


def inspect_model(model_path: str, output_path: Path | None = None):
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None

    if output_path is None:
        output_path = _infer_default_output_path(model_path)

    print(f"📁 Loading model from: {model_path}")

    try:
        # Load the model
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded model of type: {type(model)}")

        # Try different ways to extract feature names
        feature_names = None
        feature_count = None

        # Method 1: If it's a pipeline, check if it has feature names
        if hasattr(model, "named_steps"):
            print(
                f"🔍 Model is a pipeline with steps: {list(model.named_steps.keys())}"
            )

            # Check for preprocessor
            if "preprocessor" in model.named_steps:
                preprocessor = model.named_steps["preprocessor"]
                if hasattr(preprocessor, "get_feature_names_out"):
                    try:
                        # Some preprocessors require fitted input to produce names; we only log the method presence here
                        print(
                            "ℹ️ Preprocessor provides get_feature_names_out; names require fitted context"
                        )
                    except Exception:
                        pass

            # Check for feature_names attributes on steps
            for step_name, step in model.named_steps.items():
                if hasattr(step, "feature_names_"):
                    feature_names = step.feature_names_
                    print(
                        f"✅ Found feature_names_ in step '{step_name}': {len(feature_names)} features"
                    )
                    break
                elif hasattr(step, "feature_names_in_"):
                    feature_names = step.feature_names_in_
                    print(
                        f"✅ Found feature_names_in_ in step '{step_name}': {len(feature_names)} features"
                    )
                    break
                elif hasattr(step, "n_features_in_"):
                    feature_count = step.n_features_in_
                    print(f"📊 Step '{step_name}' expects {feature_count} features")

        # Method 2: Check direct attributes on the model
        if feature_names is None:
            if hasattr(model, "feature_names_"):
                feature_names = model.feature_names_
                print(
                    f"✅ Found feature_names_ on model: {len(feature_names)} features"
                )
            elif hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
                print(
                    f"✅ Found feature_names_in_ on model: {len(feature_names)} features"
                )
            elif hasattr(model, "n_features_in_"):
                feature_count = model.n_features_in_
                print(f"📊 Model expects {feature_count} features")

        # If we have feature names, save them
        if feature_names is not None:
            feature_list = (
                feature_names.tolist()
                if hasattr(feature_names, "tolist")
                else list(feature_names)
            )

            contract = {
                "model_name": Path(model_path).stem,
                "model_path": model_path,
                "feature_count": len(feature_list),
                "features": feature_list,
                "extraction_method": "direct_inspection",
                "extracted_at": pd.Timestamp.now().isoformat(),
            }

            with open(output_path, "w") as f:
                json.dump(contract, f, indent=2)

            print(f"💾 Saved feature contract to: {output_path}")
            print(f"📊 Total features required: {len(feature_list)}")

            # Print first 10 features
            print("\n🔍 First 10 features:")
            for i, feature in enumerate(feature_list[:10]):
                print(f"  {i+1:2d}. {feature}")

            if len(feature_list) > 10:
                print(f"  ... and {len(feature_list) - 10} more")

            return feature_list

        # If no feature names found but we have a count
        if feature_count is not None:
            print(
                f"⚠️ Model expects {feature_count} features, but feature names not found"
            )
            print(
                "💡 You'll need to inspect the training code or data to get the exact feature names"
            )
            return None

        print("❌ Could not extract feature information from the model")
        print("🔍 Model attributes:")
        attrs = [attr for attr in dir(model) if not attr.startswith("_")]
        for attr in attrs[:20]:  # Show first 20 attributes
            try:
                value = getattr(model, attr)
                if not callable(value):
                    print(f"  - {attr}: {type(value)}")
            except Exception:
                print(f"  - {attr}: <could not access>")
        return None

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract feature contract from a serialized sklearn model/pipeline"
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to model artifact (.joblib/.pkl)"
    )
    parser.add_argument(
        "--output", required=False, help="Optional output path for contract JSON"
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    features = inspect_model(args.model_path, output_path)
    if features:
        print(f"\n✅ Successfully extracted {len(features)} feature names")
    else:
        print("\n❌ Could not extract feature names automatically")
        sys.exit(1)
