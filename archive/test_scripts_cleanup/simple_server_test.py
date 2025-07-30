#!/usr/bin/env python3
"""
Simple HTTP server test to isolate network issues
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime

class TestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/ping':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'success': True,
                'message': 'pong from simple server',
                'timestamp': datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/api/test_prediction_status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'success': True,
                'status': {
                    'running': False,
                    'progress': 0,
                    'log': []
                },
                'timestamp': datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[{datetime.now().isoformat()}] {format % args}")

if __name__ == '__main__':
    server = HTTPServer(('localhost', 5002), TestHandler)
    print(f"🚀 Simple HTTP server starting on port 5002...")
    server.serve_forever()
