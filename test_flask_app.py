#!/usr/bin/env python3
"""
Minimal Flask app for testing - runs on port 5002
"""

from flask import Flask, jsonify, request
import json
import os
import glob
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Cache for upcoming races API
_upcoming_races_cache = {
    "data": None,
    "timestamp": None,
    "expires_in_minutes": 5,  # Cache for 5 minutes
}

def load_upcoming_races(refresh=False):
    """Helper function to load upcoming races from CSV and JSON files."""
    global _upcoming_races_cache
    now = datetime.now()

    # Check if we should use cache
    if (not refresh and _upcoming_races_cache["data"] is not None and 
        _upcoming_races_cache["timestamp"] is not None and 
        (now - _upcoming_races_cache["timestamp"]).total_seconds() < 
        (_upcoming_races_cache["expires_in_minutes"] * 60)):
        print("Using cached race list")  # This is the log message we need to check
        return _upcoming_races_cache["data"]

    print("Loading fresh race data from files")  # Log when loading fresh data
    upcoming_races_dir = "./upcoming_races"
    races = []

    # Process all files in the upcoming_races directory
    if os.path.exists(upcoming_races_dir):
        for filename in os.listdir(upcoming_races_dir):
            if filename.endswith(".csv") or filename.endswith(".json"):
                file_path = os.path.join(upcoming_races_dir, filename)
                try:
                    if filename.endswith(".csv"):
                        df = pd.read_csv(file_path)
                    else:
                        with open(file_path, "r") as f:
                            df = pd.json_normalize(json.load(f))

                    # Process each row in the file
                    for _, row in df.iterrows():
                        # Build race metadata without post-outcome fields
                        race_metadata = {
                            "name": row.get("Race Name", row.get("name", "Unknown Race")),
                            "venue": row.get("Venue", row.get("venue", "Unknown Venue")),
                            "date": row.get("Date", row.get("date", "Unknown Date")),
                            "distance": row.get("Distance", row.get("distance", "Unknown Distance")),
                            "grade": row.get("Grade", row.get("grade", "Unknown Grade")),
                            "number": row.get("Race Number", row.get("number", row.get("race_number", 0))),
                            "filename": filename,  # Include source filename
                        }
                        # Explicitly exclude post-outcome fields
                        # No winner_name, winner_margin, winner_odds, etc.
                        races.append(race_metadata)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue

    # Update cache
    _upcoming_races_cache["data"] = races
    _upcoming_races_cache["timestamp"] = now

    return races

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "Greyhound Racing Test Flask Server",
        "port": 5002
    })

@app.route('/api/upcoming_races')
def upcoming_races():
    """API endpoint to get upcoming races from files"""
    races = []
    upcoming_dir = "./upcoming_races"
    
    # Get CSV files
    csv_files = glob.glob(os.path.join(upcoming_dir, "*.csv"))
    for csv_file in csv_files:
        races.append({
            "file": os.path.basename(csv_file),
            "type": "csv",
            "path": csv_file
        })
    
    # Get JSON files  
    json_files = glob.glob(os.path.join(upcoming_dir, "*.json"))
    for json_file in json_files:
        races.append({
            "file": os.path.basename(json_file),
            "type": "json", 
            "path": json_file
        })
    
    return jsonify({
        "success": True,
        "upcoming_races": races,
        "total_files": len(races),
        "csv_count": len(csv_files),
        "json_count": len(json_files)
    })

@app.route('/api/upcoming_races_csv')
def api_upcoming_races_csv():
    """API endpoint to fetch upcoming races from CSV and JSON files with caching"""
    try:
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        races = load_upcoming_races(refresh=refresh)

        # Prepare response data
        response_data = {
            "success": True,
            "races": races,
            "count": len(races),
            "timestamp": datetime.now().isoformat(),
            "from_cache": not refresh and _upcoming_races_cache["data"] is not None,
            "cache_expires_in_minutes": _upcoming_races_cache["expires_in_minutes"],
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "success": False, 
            "error": f"Error fetching upcoming races: {str(e)}"
        }), 500

@app.route('/api/clear_cache')
def clear_cache():
    """API endpoint to clear in-memory cache"""
    global _upcoming_races_cache
    # Clear the cache
    _upcoming_races_cache["data"] = None
    _upcoming_races_cache["timestamp"] = None
    return jsonify({
        "success": True,
        "message": "Cache cleared",
        "refresh": True
    })

@app.route('/api/database_status')
def database_status():
    """Check database status"""
    db_file = "greyhound_racing_data.db"
    
    if os.path.exists(db_file):
        stat_info = os.stat(db_file)
        return jsonify({
            "success": True,
            "database_exists": True,
            "database_size": stat_info.st_size,
            "last_modified": stat_info.st_mtime
        })
    else:
        return jsonify({
            "success": False,
            "database_exists": False,
            "message": "Database not found"
        })

if __name__ == "__main__":
    print("🚀 Starting test Flask server on port 5002...")
    app.run(debug=False, host="localhost", port=5002)
