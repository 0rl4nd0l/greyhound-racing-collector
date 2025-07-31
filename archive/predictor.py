import os
from threading import Lock

class Predictor:
    """Base Predictor class for strategy pattern."""

    def __init__(self):
        self.lock = Lock()

    def predict(self, race_file_path):
        raise NotImplementedError("Predict method must be implemented by subclasses")

