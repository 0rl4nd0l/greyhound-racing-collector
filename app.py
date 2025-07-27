#!/usr/bin/env python3
"""
Greyhound Racing Dashboard - Main Flask Application
====================================================

This is the main Flask web application that provides the dashboard interface
for the greyhound racing prediction system.

Author: Orlando Lee
Date: July 27, 2025
"""

import os
import sys
import json
import sqlite3
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'greyhound_racing_secret_key_2025'

# Configuration constants
DATABASE_PATH = 'greyhound_racing_data.db'
UNPROCESSED_DIR = './unprocessed'
PROCESSED_DIR = './processed'
UPCOMING_RACES_DIR = './upcoming_races'
PREDICTIONS_DIR = './predictions'

# Upload configuration
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UNPROCESSED_DIR

# Global status tracking for async operations
operation_status = {
    'running': False,
    'progress': 0,
    'current_step': '',
    'log': [],
    'start_time': None,
    'completed': False,
    'results': None,
    'error': None
}

import shutil
from werkzeug.utils import secure_filename

class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_recent_races(self, limit=10):
        """Get recent race data"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Try to get races from the database
            cursor.execute("""
                SELECT race_date, venue, race_number, COUNT(*) as runners
                FROM races 
                ORDER BY race_date DESC 
                LIMIT ?
            """, (limit,))
            
            races = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'date': race[0],
                    'venue': race[1],
                    'race_number': race[2],
                    'runners': race[3]
                }
                for race in races
            ]
        except Exception as e:
            print(f"Error getting recent races: {e}")
            return []
    
    def get_stats(self):
        """Get database statistics"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            stats = {}
            
            # Get total races
            cursor.execute("SELECT COUNT(*) FROM races")
            stats['total_races'] = cursor.fetchone()[0]
            
            # Get total runners
            cursor.execute("SELECT COUNT(*) FROM runners")
            stats['total_runners'] = cursor.fetchone()[0]
            
            # Get recent predictions count
            cursor.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE created_at > date('now', '-7 days')
            """)
            stats['recent_predictions'] = cursor.fetchone()[0]
            
            conn.close()
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'total_races': 0,
                'total_runners': 0,
                'recent_predictions': 0
            }

# Initialize database manager
db_manager = DatabaseManager(DATABASE_PATH)

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get basic statistics
        stats = db_manager.get_stats()
        recent_races = db_manager.get_recent_races(5)
        
        # Check for unprocessed files
        unprocessed_count = 0
        if os.path.exists(UNPROCESSED_DIR):
            unprocessed_count = len([f for f in os.listdir(UNPROCESSED_DIR) if f.endswith('.csv')])
        
        # Check for upcoming races
        upcoming_count = 0
        if os.path.exists(UPCOMING_RACES_DIR):
            upcoming_count = len([f for f in os.listdir(UPCOMING_RACES_DIR) if f.endswith('.csv')])
        
        return render_template('dashboard.html',
                             stats=stats,
                             recent_races=recent_races,
                             unprocessed_count=unprocessed_count,
                             upcoming_count=upcoming_count,
                             last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                             
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/status')
def api_status():
    """API endpoint to get system status"""
    global operation_status
    
    return jsonify({
        'success': True,
        'status': operation_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/collect', methods=['POST'])
def api_collect():
    """API endpoint to start data collection"""
    global operation_status
    
    if operation_status['running']:
        return jsonify({
            'success': False,
            'message': 'Collection already running'
        })
    
    # Start collection in background thread
    thread = threading.Thread(target=run_data_collection)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Data collection started'
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint to start data analysis"""
    global operation_status
    
    if operation_status['running']:
        return jsonify({
            'success': False,
            'message': 'Analysis already running'
        })
    
    # Start analysis in background thread
    thread = threading.Thread(target=run_data_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Data analysis started'
    })

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Upload form guide CSV for processing"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create unprocessed directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(file_path)
            flash(f'File "{filename}" uploaded successfully and ready for processing!', 'success')
            return redirect(url_for('scraping_status'))
        else:
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint to start predictions"""
    global operation_status
    
    if operation_status['running']:
        return jsonify({
            'success': False,
            'message': 'Prediction already running'
        })
    
    # Start prediction in background thread
    thread = threading.Thread(target=run_predictions)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Predictions started'
    })

@app.route('/api/recent_predictions')
def api_recent_predictions():
    """API endpoint to get recent predictions"""
    try:
        predictions = []
        
        if os.path.exists(PREDICTIONS_DIR):
            # Get recent prediction files
            pred_files = list(Path(PREDICTIONS_DIR).glob('*.json'))
            pred_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for pred_file in pred_files[:10]:  # Get last 10
                try:
                    with open(pred_file, 'r') as f:
                        pred_data = json.load(f)
                        predictions.append({
                            'filename': pred_file.name,
                            'venue': pred_data.get('venue', 'Unknown'),
                            'race_number': pred_data.get('race_number', 'Unknown'),
                            'top_pick': pred_data.get('predictions', [{}])[0].get('dog_name', 'Unknown') if pred_data.get('predictions') else 'No prediction',
                            'confidence': pred_data.get('predictions', [{}])[0].get('confidence', 0) if pred_data.get('predictions') else 0,
                            'created': datetime.fromtimestamp(pred_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                        })
                except Exception as e:
                    print(f"Error reading prediction file {pred_file}: {e}")
                    continue
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/races')
def races():
    """Races listing page"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    all_races = db_manager.get_recent_races(limit=per_page * page)
    
    return render_template('races.html', races=all_races, page=page)

@app.route('/scraping')
def scraping_status():
    """Scraping status and controls"""
    # Get basic file statistics
    unprocessed_count = 0
    processed_count = 0
    
    if os.path.exists(UNPROCESSED_DIR):
        unprocessed_count = len([f for f in os.listdir(UNPROCESSED_DIR) if f.endswith('.csv')])
    
    if os.path.exists(PROCESSED_DIR):
        processed_count = len([f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv')])
    
    # Get recent unprocessed files
    unprocessed_files = []
    if os.path.exists(UNPROCESSED_DIR):
        files = [f for f in os.listdir(UNPROCESSED_DIR) if f.endswith('.csv')]
        for filename in sorted(files, reverse=True)[:10]:
            file_path = os.path.join(UNPROCESSED_DIR, filename)
            file_stat = os.stat(file_path)
            unprocessed_files.append({
                'filename': filename,
                'size': file_stat.st_size,
                'modified': datetime.fromtimestamp(file_stat.st_mtime)
            })
    
    file_stats = {
        'unprocessed_files': unprocessed_count,
        'processed_files': processed_count
    }
    
    return render_template('scraping_status.html',
                         file_stats=file_stats,
                         unprocessed_files=unprocessed_files)

@app.route('/api/processing_status')
def api_processing_status():
    """API endpoint for processing status"""
    global operation_status
    
    status = operation_status.copy()
    status['last_update'] = datetime.now().isoformat()
    
    return jsonify(status)

def run_data_collection():
    """Run data collection process in background"""
    global operation_status
    
    operation_status.update({
        'running': True,
        'progress': 0,
        'current_step': 'Starting data collection...',
        'log': [],
        'start_time': datetime.now(),
        'completed': False,
        'results': None,
        'error': None
    })
    
    try:
        # Call the main run.py script for collection
        result = subprocess.run([sys.executable, 'run.py', 'collect'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            operation_status.update({
                'running': False,
                'progress': 100,
                'current_step': 'Collection completed',
                'completed': True,
                'results': {'message': 'Data collection completed successfully'}
            })
        else:
            operation_status.update({
                'running': False,
                'error': f'Collection failed: {result.stderr}',
                'completed': True
            })
            
    except Exception as e:
        operation_status.update({
            'running': False,
            'error': str(e),
            'completed': True
        })

def run_data_analysis():
    """Run data analysis process in background"""
    global operation_status
    
    operation_status.update({
        'running': True,
        'progress': 0,
        'current_step': 'Starting data analysis...',
        'log': [],
        'start_time': datetime.now(),
        'completed': False,
        'results': None,
        'error': None
    })
    
    try:
        # Call the main run.py script for analysis
        result = subprocess.run([sys.executable, 'run.py', 'analyze'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            operation_status.update({
                'running': False,
                'progress': 100,
                'current_step': 'Analysis completed',
                'completed': True,
                'results': {'message': 'Data analysis completed successfully'}
            })
        else:
            operation_status.update({
                'running': False,
                'error': f'Analysis failed: {result.stderr}',
                'completed': True
            })
            
    except Exception as e:
        operation_status.update({
            'running': False,
            'error': str(e),
            'completed': True
        })

# Additional function for processing files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv'}

def run_predictions():
    """Run predictions process in background"""
    global operation_status
    
    operation_status.update({
        'running': True,
        'progress': 0,
        'current_step': 'Starting predictions...',
        'log': [],
        'start_time': datetime.now(),
        'completed': False,
        'results': None,
        'error': None
    })
    
    try:
        # Call the main run.py script for predictions
        result = subprocess.run([sys.executable, 'run.py', 'predict'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            operation_status.update({
                'running': False,
                'progress': 100,
                'current_step': 'Predictions completed',
                'completed': True,
                'results': {'message': 'Predictions completed successfully'}
            })
        else:
            operation_status.update({
                'running': False,
                'error': f'Predictions failed: {result.stderr}',
                'completed': True
            })
            
    except Exception as e:
        operation_status.update({
            'running': False,
            'error': str(e),
            'completed': True
        })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    print("🚀 Starting Greyhound Racing Dashboard...")
    print(f"📊 Database: {DATABASE_PATH}")
    print(f"📁 Unprocessed: {UNPROCESSED_DIR}")
    print(f"🎯 Predictions: {PREDICTIONS_DIR}")
    print()
    
    # Create necessary directories
    os.makedirs(UNPROCESSED_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(UPCOMING_RACES_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    # Start the Flask app
    app.run(host='localhost', port=5001, debug=False)
