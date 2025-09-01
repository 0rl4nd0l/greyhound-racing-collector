#!/usr/bin/env python3
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def create_simple_model():
    print("🤖 Training simple model with staging data...")

    # Connect to staging database
    db_path = "greyhound_racing_data_staging.db"

    with sqlite3.connect(db_path) as conn:
        # Get training data
        query = """
        SELECT 
            h.dog_clean_name,
            h.box_number,
            h.weight,
            h.finish_position,
            r.venue,
            r.race_number
        FROM csv_dog_history_staging h
        JOIN csv_race_metadata_staging r ON h.race_id = r.race_id
        WHERE h.finish_position IS NOT NULL
        AND h.finish_position > 0
        AND h.box_number IS NOT NULL
        AND h.weight IS NOT NULL
        LIMIT 10000
        """

        import pandas as pd

        df = pd.read_sql_query(query, conn)

    if len(df) < 100:
        print(f"❌ Insufficient data: {len(df)} records")
        return False

    print(f"📊 Training with {len(df):,} records")

    # Prepare features
    le_dog = LabelEncoder()
    le_venue = LabelEncoder()

    X = pd.DataFrame(
        {
            "dog_encoded": le_dog.fit_transform(df["dog_clean_name"].fillna("unknown")),
            "box_number": df["box_number"].fillna(4),
            "weight": df["weight"].fillna(30.0),
            "venue_encoded": le_venue.fit_transform(df["venue"].fillna("unknown")),
            "race_number": df["race_number"].fillna(5),
        }
    )

    # Target: win (1) vs not win (0)
    y = (df["finish_position"] == 1).astype(int)

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Save model and encoders
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_data = {
        "model": model,
        "dog_encoder": le_dog,
        "venue_encoder": le_venue,
        "training_records": len(df),
        "feature_names": list(X.columns),
    }

    with open("models/simple_staging_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("✅ Model trained and saved!")

    # Test predictions
    sample_probs = model.predict_proba(X[:8])[:, 1]  # Probability of winning
    print(f"📈 Sample win probabilities: {sample_probs}")

    return True


if __name__ == "__main__":
    create_simple_model()
