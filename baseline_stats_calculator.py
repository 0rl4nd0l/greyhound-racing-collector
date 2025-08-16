import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_statistics(database_path):
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT weight, race_time FROM dog_performances", conn)
    conn.close()
    
    monitored_features = ["weight", "race_time"]
    stats = {}
    for feature in monitored_features:
        if feature in df.columns:
            stats[feature] = {
                "mean": df[feature].mean(),
                "std": df[feature].std(),
                "quantiles": df[feature].quantile([0.25, 0.5, 0.75]).tolist()
            }
    return stats

if __name__ == "__main__":
    # Define paths
    db_path = "databases/race_data.db"
    git_sha = "157602b"
    date = datetime.now().strftime('%Y%m%d')
    versioned_dir = f"baseline_stats/{git_sha}_{date}"
    os.makedirs(versioned_dir, exist_ok=True)
    
    # Calculate statistics
    statistics = calculate_statistics(db_path)
    
    # Save statistics
    stats_file_path = os.path.join(versioned_dir, "stats.json")
    with open(stats_file_path, "w") as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Statistics saved to {stats_file_path}")

