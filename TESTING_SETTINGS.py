import os

class TestingConfig:
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "postgresql://test_user:test_password@localhost:5433/greyhound_test")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REDIS_URL = "redis://localhost:6380/0"
    UPLOAD_FOLDER = "/workspace/tests/uploads"

# Assign the TestingConfig for use
config = TestingConfig
