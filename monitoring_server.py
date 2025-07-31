#!/usr/bin/env python3
"""
Real-Time Monitoring Web Server
===============================

Flask web server providing REST API endpoints for the real-time monitoring dashboard.
Serves monitoring data, system health, and prediction metrics.

Author: AI Assistant
Date: July 27, 2025
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from prometheus_client import generate_latest, CollectorRegistry, Gauge
import os
import json
from pathlib import Path
from datetime import datetime
from monitoring_api import get_monitoring_api
from logger import logger

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

# Get monitoring API instance
monitoring = get_monitoring_api()

@app.route('/')
def dashboard():
    """Serve the main dashboard HTML file"""
    try:
        return send_from_directory('.', 'monitoring_dashboard.html')
    except Exception as e:
        logger.log_error(f"Error serving dashboard: {str(e)}", context={'component': 'web_server'})
        return jsonify({'error': 'Dashboard not available'}), 500

@app.route('/metrics')
def get_metrics_prometheus():
    """Expose Prometheus metrics"""
    try:
        registry = CollectorRegistry()
        scrape_duration = Gauge('scrape_duration_seconds', 'Scrape duration by job', registry=registry)
        model_latency = Gauge('model_latency_seconds', 'Model inference time', registry=registry)
        queue_length = Gauge('queue_length', 'Job queue length', registry=registry)

        # Simulate metric values
        scrape_duration.set(monitoring._calculate_live_metrics().get('response_time', 0))
        # Example: Set fixed values for other metrics
        model_latency.set(1.5)
        queue_length.set(3)

        return generate_latest(registry), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        logger.log_error(f"Prometheus metrics generation failed: {str(e)}", context={'component': 'web_api'})
        return "Internal Server Error", 500

@app.route('/api/health')
def get_health():
    """Get system health status"""
    try:
        health_data = monitoring.get_system_health()
        return jsonify(health_data)
    except Exception as e:
        logger.log_error(f"Health endpoint error: {str(e)}", context={'component': 'web_api'})
        return jsonify({
            'success': False,
            'error': str(e),
            'system_status': {
                'status': 'error',
                'message': 'Health check failed'
            }
        }), 500

@app.route('/api/metrics')
def get_metrics():
    """Get performance metrics"""
    try:
        metrics_data = monitoring.get_performance_metrics()
        return jsonify(metrics_data)
    except Exception as e:
        logger.log_error(f"Metrics endpoint error: {str(e)}", context={'component': 'web_api'})
        return jsonify({
            'success': False,
            'error': str(e),
            'system_status': {
                'status': 'error',
                'message': 'Metrics calculation failed'
            }
        }), 500

@app.route('/api/predictions')
def get_predictions():
    """Get recent predictions"""
    try:
        limit = request.args.get('limit', 10, type=int)
        predictions_data = monitoring.get_recent_predictions(limit=limit)
        return jsonify(predictions_data)
    except Exception as e:
        logger.log_error(f"Predictions endpoint error: {str(e)}", context={'component': 'web_api'})
        return jsonify({
            'success': False,
            'error': str(e),
            'predictions': []
        }), 500

@app.route('/api/trends')
def get_trends():
    """Get accuracy trends"""
    try:
        days = request.args.get('days', 7, type=int)
        trends_data = monitoring.get_accuracy_trends(days=days)
        return jsonify(trends_data)
    except Exception as e:
        logger.log_error(f"Trends endpoint error: {str(e)}", context={'component': 'web_api'})
        return jsonify({
            'success': False,
            'error': str(e),
            'trends': []
        }), 500

@app.route('/api/alerts')
def get_alerts():
    """Get system alerts"""
    try:
        alerts_data = monitoring.get_system_alerts()
        return jsonify(alerts_data)
    except Exception as e:
        logger.log_error(f"Alerts endpoint error: {str(e)}", context={'component': 'web_api'})
        return jsonify({
            'success': False,
            'error': str(e),
            'alerts': []
        }), 500

@app.route('/api/status')
def get_status():
    """Get combined status information"""
    try:
        # Get all status information in one call
        health = monitoring.get_system_health()
        metrics = monitoring.get_performance_metrics()
        alerts = monitoring.get_system_alerts()
        
        combined_status = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'system': {
                'status': health.get('system_status', {}).get('status', 'unknown'),
                'message': health.get('system_status', {}).get('message', 'Status unknown'),
                'components': health.get('components', {}),
                'resources': health.get('resources', {})
            },
            'predictions': {
                'win_accuracy': metrics.get('metrics', {}).get('win_accuracy', 0),
                'place_accuracy': metrics.get('metrics', {}).get('place_accuracy', 0),
                'predictions_today': metrics.get('metrics', {}).get('predictions_today', 0),
                'response_time': metrics.get('metrics', {}).get('response_time', 0)
            },
            'alerts': {
                'count': alerts.get('alert_count', 0),
                'items': alerts.get('alerts', [])
            }
        }
        
        return jsonify(combined_status)
        
    except Exception as e:
        logger.log_error(f"Status endpoint error: {str(e)}", context={'component': 'web_api'})
        return jsonify({
            'success': False,
            'error': str(e),
            'system': {
                'status': 'error',
                'message': 'Status check failed'
            }
        }), 500

@app.route('/api/info')
def get_info():
    """Get API information"""
    return jsonify({
        'name': 'Greyhound Racing Monitoring API',
        'version': '1.0.0',
        'description': 'Real-time monitoring API for greyhound racing prediction system',
        'endpoints': {
            '/api/health': 'System health metrics',
            '/api/metrics': 'Performance metrics',
            '/api/predictions': 'Recent predictions',
            '/api/trends': 'Accuracy trends',
            '/api/alerts': 'System alerts',
            '/api/status': 'Combined status information'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested API endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.log_error(f"Internal server error: {str(error)}", context={'component': 'web_server'})
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

@app.before_request
def log_request():
    """Log incoming requests"""
    if request.endpoint and not request.endpoint.startswith('static'):
        logger.log_system(
            f"API Request: {request.method} {request.path}",
            "INFO",
            "WEB_API"
        )

@app.after_request
def log_response(response):
    """Log outgoing responses"""
    if request.endpoint and not request.endpoint.startswith('static'):
        logger.log_system(
            f"API Response: {request.method} {request.path} - {response.status_code}",
            "INFO",
            "WEB_API"
        )
    return response

def run_server(host='localhost', port=5000, debug=False):
    """Run the monitoring web server"""
    logger.log_system(
        f"Starting monitoring web server on http://{host}:{port}",
        "INFO",
        "WEB_SERVER"
    )
    
    # Check if dashboard HTML file exists
    dashboard_file = Path('./monitoring_dashboard.html')
    if not dashboard_file.exists():
        logger.log_error(
            "Dashboard HTML file not found at ./monitoring_dashboard.html",
            context={'component': 'web_server'}
        )
        print("⚠️  Warning: Dashboard HTML file not found!")
        print("   Make sure 'monitoring_dashboard.html' exists in the current directory")
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except Exception as e:
        logger.log_error(f"Failed to start web server: {str(e)}", context={'component': 'web_server'})
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Greyhound Racing Monitoring Web Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🏁 Greyhound Racing Monitoring Server")
    print("=" * 40)
    print(f"🌐 Server URL: http://{args.host}:{args.port}")
    print(f"📊 Dashboard: http://{args.host}:{args.port}/")
    print(f"🔧 API Info: http://{args.host}:{args.port}/api/info")
    print("=" * 40)
    print("📝 Available API Endpoints:")
    print(f"   • GET /api/health    - System health")
    print(f"   • GET /api/metrics   - Performance metrics")
    print(f"   • GET /api/predictions - Recent predictions")
    print(f"   • GET /api/trends    - Accuracy trends")
    print(f"   • GET /api/alerts    - System alerts")
    print(f"   • GET /api/status    - Combined status")
    print("=" * 40)
    print("Press Ctrl+C to stop the server")
    print()
    
    run_server(host=args.host, port=args.port, debug=args.debug)
