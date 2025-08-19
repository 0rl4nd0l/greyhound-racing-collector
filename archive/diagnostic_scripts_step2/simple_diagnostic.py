#!/usr/bin/env python3
import sys
import sqlite3
from pathlib import Path

def main():
    base_dir = Path("/Users/orlandolee/greyhound_racing_collector")
    db_path = base_dir / "greyhound_racing_data.db"
    
    print("🧪 Simple System Health Check")
    print("=" * 40)
    
    # Test database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        race_count = cursor.fetchone()[0]
        print(f"✅ Database: {race_count} races")
        conn.close()
    except Exception as e:
        print(f"❌ Database: {e}")
    
    # Test Flask import
    try:
        sys.path.insert(0, str(base_dir))
        import app
        print("✅ Flask app import successful")
        
        if hasattr(app, "app"):
            print("✅ Flask app instance found")
        else:
            print("❌ Flask app instance not found")
            
    except Exception as e:
        print(f"❌ Flask import: {e}")
    
    # Test prediction import
    try:
        from weather_enhanced_predictor import WeatherEnhancedPredictor
        print("✅ Weather predictor import successful")
    except Exception as e:
        print(f"❌ Weather predictor: {e}")
    
    print("\n🎯 Health check complete")

if __name__ == "__main__":
    main()
