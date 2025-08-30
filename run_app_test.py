#!/usr/bin/env python3
"""
Simple script to run the Flask app for testing
"""
import os
import sys

# Set environment variables
os.environ["FLASK_ENV"] = "development"
os.environ["PORT"] = "5002"

# Import and run the app
if __name__ == "__main__":
    from app import app
    app.run(host="127.0.0.1", port=5002, debug=False)
