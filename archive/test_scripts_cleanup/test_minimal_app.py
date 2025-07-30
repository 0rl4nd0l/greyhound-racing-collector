#!/usr/bin/env python3
"""
Minimal Flask test app to isolate the hang issue
"""

from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

# Test prediction status tracking
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

if __name__ == '__main__':
    print("🚀 Starting minimal Flask test app...")
    app.run(host='0.0.0.0', port=5003, debug=True)
