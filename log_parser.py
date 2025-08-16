import os
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PredictionRecord:
    win_prob: float
    variance: float
    extraction_time: datetime
    model_version: str

class LogParser:
    def __init__(self, log_directory: str):
        self.log_directory = Path(log_directory)

    def parse_logs(self) -> List[PredictionRecord]:
        records = []

        for file_path in self.log_directory.glob('*.jsonl'):
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        entry = json.loads(line)
                        record = self.extract_fields(entry)
                        if record is not None:
                            records.append(record)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error parsing line: {line}. Error: {e}")
                    except Exception as e:
                        print(f"Unexpected error: {e}")
        return records

    def extract_fields(self, entry: Dict[str, Any]) -> PredictionRecord:
        try:
            win_prob = entry['details']['win_prob']
            variance = entry['details'].get('variance', 0.0)
            extraction_time_str = entry['timestamp']
            model_version = entry['details']['model_version']

            # Attempt to parse timestamp
            try:
                extraction_time = datetime.fromisoformat(extraction_time_str)
            except ValueError:
                print(f"Timestamp parsing error for: {extraction_time_str}")
                return None

            return PredictionRecord(win_prob, variance, extraction_time, model_version)

        except KeyError as e:
            print(f"Missing expected field: {e}")
        except Exception as e:
            print(f"Unexpected error during extraction: {e}")

        return None

if __name__ == '__main__':
    parser = LogParser(log_directory='logs/prediction')
    records = parser.parse_logs()
    for record in records:
        # Here you would typically process or store records
        print(record)
