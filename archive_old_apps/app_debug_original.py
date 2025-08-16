#!/usr/bin/env python3
"""
Greyhound Racing Dashboard - Debug Version
=========================================

Simplified Flask app to isolate import/initialization issues.
"""

import json
import os
import sqlite3
from datetime import datetime

from flask import Flask, jsonify

# Basic Flask app setup
app = Flask(__name__)
app.secret_key = 'greyhound_racing_secret_key_2025'

# Configuration
DATABASE_PATH = 'greyhound_racing_data.db'

# Global test prediction status tracking
test_prediction_status = {
    'running': False,
    'progress': 0,
    'current_step': '',
    'log': [],
    'start_time': None,
    'completed': False,
    'results': None,
    'error': None
}

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

@app.route('/')
def index():
    """Simple home page"""
    return jsonify({
        'message': 'Greyhound Racing Dashboard - Debug Mode',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test_prediction_status')
def api_test_prediction_status():
    """API endpoint to get test prediction status"""
    global test_prediction_status
    
    return jsonify({
        'success': True,
        'status': test_prediction_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ping')
def api_ping():
    """Simple ping endpoint"""
    return jsonify({
        'success': True,
        'message': 'pong',
        'timestamp': datetime.now().isoformat()
    })

# Initialize database manager - this might be the issue
print("🔧 Initializing database manager...")
db_manager = DatabaseManager(DATABASE_PATH)
print("✅ Database manager initialized")

if __name__ == '__main__':
    print("🚀 Starting debug Flask app on port 5002...")
    app.run(host='localhost', port=5002, debug=False, use_reloader=False)
