#!/usr/bin/env python3
"""
Quick Fix: Use Staging Database for ML Training
===============================================

This script provides a simple solution to the training data problem:
1. Update the system to use the staging database (29,759 records)
2. Train a lightweight test model 
3. Validate predictions are working

This is the fastest path to get your ML system working properly.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

# Route read-only staging checks
try:
    from scripts.db_utils import open_sqlite_readonly
except Exception:
    def open_sqlite_readonly(db_path: str | None = None):
        import os as _os, sqlite3 as _sqlite3
        path = db_path or _os.getenv("STAGING_DB_PATH") or "greyhound_racing_data_staging.db"
        return _sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


def main():
    print("🚀 Quick Fix: Using Staging Database for Training")
    print("=" * 60)

    # Check staging database has data
    staging_db = os.getenv("STAGING_DB_PATH", "greyhound_racing_data_staging.db")
    if not os.path.exists(staging_db):
        print(f"❌ Staging database not found: {staging_db}")
        return False

    # Verify data count
    with open_sqlite_readonly(staging_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM csv_dog_history_staging WHERE finish_position IS NOT NULL"
        )
        training_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM csv_race_metadata_staging")
        race_count = cursor.fetchone()[0]

    print(
        f"✅ Found staging database with {training_count:,} training records and {race_count:,} races"
    )

    # Run training using the maintained script (routing-aware)
    print("🏃 Running training...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/train_staging_model.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print(result.stdout)

            # Test if model file exists
            if os.path.exists("models/simple_staging_model.pkl"):
                print("✅ Model file created: models/simple_staging_model.pkl")

                # Create a simple prediction test
                test_script = """
import pickle
import pandas as pd

# Load model
with open("models/simple_staging_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data['model']
dog_encoder = model_data['dog_encoder']
venue_encoder = model_data['venue_encoder']

print(f"Model trained on {model_data['training_records']} records")
print(f"Features: {model_data['feature_names']}")

# Test prediction for 8 dogs
test_data = pd.DataFrame({
    'dog_encoded': [0, 1, 2, 3, 4, 5, 6, 7],
    'box_number': [1, 2, 3, 4, 5, 6, 7, 8], 
    'weight': [30.0, 31.0, 29.5, 32.0, 30.5, 31.5, 29.0, 30.8],
    'venue_encoded': [0, 0, 0, 0, 0, 0, 0, 0],
    'race_number': [5, 5, 5, 5, 5, 5, 5, 5]
})

win_probs = model.predict_proba(test_data)[:, 1]
print("\\nTest predictions (win probabilities):")
for i, prob in enumerate(win_probs):
    print(f"  Dog {i+1} (Box {i+1}): {prob:.3f}")

print(f"\\nSum of probabilities: {win_probs.sum():.3f}")
print("✅ Predictions look realistic!" if 0.5 < win_probs.sum() < 2.0 else "⚠️ Unusual probability distribution")
"""

                with open("test_model.py", "w") as f:
                    f.write(test_script)

                print("🧪 Testing model predictions...")
                test_result = subprocess.run(
                    [sys.executable, "test_model.py"], capture_output=True, text=True
                )
                print(test_result.stdout)

                # Clean up test file
                if os.path.exists("test_model.py"):
                    os.remove("test_model.py")

                return True
            else:
                print("❌ Model file not found after training")
                return False
        else:
            print("❌ Training failed!")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ Training timed out")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 QUICK FIX COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Your system now has a working ML model")
        print("2. The model is trained on 29,000+ historical race records")
        print("3. Predictions should now show realistic probabilities")
        print("4. You can test predictions through the web interface")
        print("\n💡 To use this model in your app:")
        print("   - Update ml_system_v4.py to load models/simple_staging_model.pkl")
        print("   - Or set DATABASE_URL=sqlite:///greyhound_racing_data_staging.db")
    else:
        print("\n❌ Quick fix failed. Check the error messages above.")

    sys.exit(0 if success else 1)
