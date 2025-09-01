"""
Config package
- Exposes Flask app configuration via get_config and config classes
- Hosts OpenAI config as a submodule (config.openai_config)

This file mirrors the public API from the top-level config.py so imports like:
    from config import get_config
continue to work even though there's now a config package for openai_config.
"""

import os
from pathlib import Path


# Reproduce the configuration classes and helper from top-level config.py
class Config:
    """Base configuration class"""

    # Flask settings
    SECRET_KEY = os.environ.get("SECRET_KEY") or "greyhound_racing_secret_key_2025"

    # Database settings
    # Prefer GREYHOUND_DB_PATH when provided; fall back to DATABASE_PATH for backward compatibility
    DATABASE_PATH = (
        os.environ.get("GREYHOUND_DB_PATH")
        or os.environ.get("DATABASE_PATH")
        or "greyhound_racing_data.db"
    )

    # Feature flags and modes (defaults aligned with live UI usage; CI/tests override via env)
    ENABLE_RESULTS_SCRAPERS = str(
        os.environ.get("ENABLE_RESULTS_SCRAPERS", "1")
    ).lower() not in ("0", "false", "no", "off")
    ENABLE_LIVE_SCRAPING = str(
        os.environ.get("ENABLE_LIVE_SCRAPING", "1")
    ).lower() not in ("0", "false", "no", "off")
    PREDICTION_IMPORT_MODE = os.environ.get("PREDICTION_IMPORT_MODE", "historical")

    # Directory paths
    BASE_DIR = Path(__file__).resolve().parents[1]
    UNPROCESSED_DIR = str(BASE_DIR / "unprocessed")
    PROCESSED_DIR = str(BASE_DIR / "processed")
    HISTORICAL_DIR = str(BASE_DIR / "historical_races")
    UPCOMING_DIR = os.environ.get("UPCOMING_RACES_DIR") or str(
        BASE_DIR / "upcoming_races"
    )

    # Upload settings
    UPLOAD_FOLDER = UPCOMING_DIR
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {"csv"}

    # Cache settings
    SEND_FILE_MAX_AGE_DEFAULT = 300  # 5 minutes for development

    # Flask-Compress base settings
    COMPRESS_MIMETYPES = [
        "text/html",
        "text/css",
        "text/xml",
        "text/plain",
        "application/json",
        "application/javascript",
        "application/xml+rss",
        "application/atom+xml",
        "image/svg+xml",
    ]
    COMPRESS_LEVEL = 6  # Default compression level
    COMPRESS_MIN_SIZE = 500  # Don't compress responses smaller than 500 bytes
    # Prefer gzip across all environments; disable Brotli by default for test compatibility
    COMPRESS_ALGORITHM = "gzip"
    COMPRESS_BR = False


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True

    # Flask-Compress settings for development
    COMPRESS_LEVEL = 4  # Faster compression for development
    COMPRESS_MIN_SIZE = 1000  # Higher threshold for development


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False

    # Enhanced caching for production
    SEND_FILE_MAX_AGE_DEFAULT = 60 * 60 * 24 * 365  # 1 year for static files

    # Flask-Compress production settings
    COMPRESS_LEVEL = 6  # Good balance of compression ratio and speed
    COMPRESS_MIN_SIZE = 500  # Compress responses >= 500 bytes
    COMPRESS_CACHE_KEY = None  # Use default cache key
    COMPRESS_CACHE_BACKEND = None  # Disable caching to avoid callable issues
    COMPRESS_REGISTER = True  # Auto-register compression
    COMPRESS_ALGORITHM = "gzip"  # Use gzip compression

    # Additional production optimizations
    COMPRESS_BR = False  # Disable Brotli for compatibility
    COMPRESS_BR_LEVEL = 4  # Brotli level if enabled
    COMPRESS_BR_MODE = 0  # Generic mode
    COMPRESS_BR_WINDOW = 22  # Window size
    COMPRESS_BR_BLOCK = 0  # Block size


class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True
    WTF_CSRF_ENABLED = False

    # Disable compression for testing
    COMPRESS_LEVEL = 0
    COMPRESS_MIN_SIZE = 999999  # Effectively disable compression


# Configuration mapping
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config(config_name=None):
    """Get configuration class based on environment or name"""
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "default")

    return config.get(config_name, DevelopmentConfig)
