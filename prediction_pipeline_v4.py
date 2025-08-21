"""
Prediction Pipeline V4 - Advanced Integrated System
==================================================

An advanced prediction pipeline based on ML System V4, leveraging all available model improvements
and EV calculations for enhanced predictions.
"""

import logging
import os
import pandas as pd
from datetime import datetime

from ml_system_v4 import MLSystemV4
from src.parsers.csv_ingestion import CsvIngestion

logger = logging.getLogger(__name__)

class PredictionPipelineV4:
    def __init__(self, db_path="greyhound_racing_data.db"):
        # Resolve database path intelligently
        resolved_db = os.getenv('GREYHOUND_DB_PATH') or db_path
        candidates = [
            resolved_db,
            os.path.join('.', resolved_db) if not os.path.isabs(resolved_db) else resolved_db,
            os.path.join('.', 'greyhound_racing_data.db'),
            os.path.join('.', 'databases', 'comprehensive_greyhound_data.db'),
            os.path.join('.', 'databases', 'greyhound_racing_data.db'),
        ]
        chosen = None
        for cand in candidates:
            try:
                if cand and os.path.isfile(cand):
                    chosen = cand
                    break
            except Exception:
                continue
        self.db_path = chosen or resolved_db
        if not os.path.isfile(self.db_path):
            logger.warning(f"⚠️ Database not found at {self.db_path}. Historical features may be empty. Set GREYHOUND_DB_PATH to fix.")
        else:
            logger.info(f"🗄️ Using database: {self.db_path}")

        self.ml_system_v4 = MLSystemV4(self.db_path)
        logger.info("🚀 Prediction Pipeline V4 - Advanced System Initialized")

    def predict_race_file(self, race_file_path: str) -> dict:
        """Main prediction method using ML System V4."""
        logger.info(
            f"🚀 Starting prediction for: {os.path.basename(race_file_path)} using ML System V4"
        )

        # Pre-prediction module sanity check
        # IMPORTANT: This call ensures no historical-data collectors, scrapers, or heavy frameworks
        # are imported during prediction. Module guard enforces prediction_only import policy.
        # Keep this import local and NEVER at module top-level to avoid false positives in tests.
        try:
            from utils import module_guard
            module_guard.pre_prediction_sanity_check(
                context='PredictionPipelineV4.predict_race_file',
                extra_info={'race_file_path': os.path.basename(race_file_path)}
            )
        except Exception as e:
            logger.error(f"🛑 Module guard blocked prediction: {e}")
            # Provide clear, actionable error response
            guidance = []
            if hasattr(e, 'resolution'):
                guidance = getattr(e, 'resolution', [])
            return {
                'success': False,
                'error': str(e),
                'race_id': os.path.basename(race_file_path).replace('.csv', ''),
                'fallback_reason': 'Disallowed module(s) loaded – see guidance',
                'resolution': guidance,
            }

        try:
            # Use CSV ingestion to read race data (prediction-only safe component)
            # NOTE: CsvIngestion reads a single race CSV (race data) and does not load
            # historical results. Keep ingestion imports narrow to avoid pulling in
            # broader scraping frameworks at prediction time.
            ingestion = CsvIngestion(race_file_path)
            parsed_race, validation_report = ingestion.parse_csv()
            
            if not validation_report.is_valid:
                errors = '\n'.join(validation_report.errors)
                logger.error(f"🛑 Validation failed for {race_file_path} with errors: {errors}")
                return {'success': False, 'error': errors, 'race_id': race_file_path}
            
            # Convert to DataFrame
            race_data = pd.DataFrame(parsed_race.records, columns=parsed_race.headers)
            race_id = os.path.basename(race_file_path).replace('.csv', '')  # Use full filename without extension as race_id

            # Map CSV columns to expected ML System V4 format
            race_data = self._map_csv_to_v4_format(race_data, race_file_path)

            # Perform prediction with V4 system
            result = self.ml_system_v4.predict_race(race_data, race_id)

            if result.get('success'):
                logger.info(f"✅ Prediction successful for {race_id}")
            else:
                logger.warning(f"❌ Prediction failed for {race_id}: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Error processing file {race_file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'race_id': race_file_path
            }

    def _map_csv_to_v4_format(self, race_data: pd.DataFrame, race_file_path: str) -> pd.DataFrame:
        """Map CSV columns to the expected ML System V4 format with proper data type handling."""
        
        # Extract race information from filename and first row
        filename = os.path.basename(race_file_path)
        
        # Parse race date from filename (e.g., "Race 1 - GOUL - 01 August 2025.csv")
        parts = filename.replace('.csv', '').split(' - ')
        if len(parts) >= 3:
            date_part = parts[2]  # "01 August 2025"
            try:
                # Convert to YYYY-MM-DD format
                date_obj = datetime.strptime(date_part, "%d %B %Y")
                race_date = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                # Fallback to current date
                race_date = datetime.now().strftime("%Y-%m-%d")
            
            venue = parts[1] if len(parts) > 1 else "Unknown"
        else:
            race_date = datetime.now().strftime("%Y-%m-%d")
            venue = "Unknown"
        
        # Create mapped DataFrame with required columns
        mapped_data = []

        # Normalizers for consistent joins
        def _normalize_dog_name(name: str) -> str:
            if name is None:
                return ''
            s = str(name)
            # Normalize common unicode punctuation and remove quotes/apostrophes/backticks
            for a, b in [("\u201c", ''), ("\u201d", ''), ("\u2018", ''), ("\u2019", ''), ("\u2013", '-'), ("\u2014", '-')]:
                s = s.replace(a, b)
            s = s.replace('"','').replace("'", '').replace('`','')
            return s.strip()
        
        for _, row in race_data.iterrows():
            dog_name = _normalize_dog_name(row.get('Dog Name', ''))
            if not dog_name or dog_name.startswith('""') or dog_name.lower() in ['nan', 'none', '']:
                continue  # Skip empty rows
            
            # Extract box number from dog name if it starts with a number
            try:
                if '.' in dog_name and dog_name.split('.')[0].isdigit():
                    box_number = int(dog_name.split('.')[0])
                    clean_dog_name = dog_name.split('.', 1)[1].strip() if '.' in dog_name else dog_name
                else:
                    # Try to get box number from BOX column
                    box_val = row.get('BOX', 1)
                    box_number = int(pd.to_numeric(box_val, errors='coerce')) if pd.notna(box_val) else 1
                    clean_dog_name = dog_name
            except (ValueError, TypeError):
                box_number = 1
                clean_dog_name = dog_name

            # Final normalization to match DB dog_clean_name conventions
            clean_dog_name = ' '.join(clean_dog_name.split())  # collapse whitespace
            clean_dog_name = clean_dog_name.upper()
            
            # Safely convert numeric values with proper error handling
            def safe_float_convert(value, default=0.0):
                """Safely convert value to float with fallback."""
                try:
                    if pd.isna(value) or value == '' or value is None:
                        return default
                    return float(pd.to_numeric(value, errors='coerce'))
                except (ValueError, TypeError):
                    return default
            
            # Map the row to V4 expected format with safe conversions
            mapped_row = {
                'race_id': filename.replace('.csv', ''),
                'dog_clean_name': clean_dog_name,
                'box_number': int(box_number),
                'weight': safe_float_convert(row.get('WGT'), 30.0),
                'starting_price': safe_float_convert(row.get('SP'), 3.0),
                'trainer_name': str(row.get('TRAINER', 'Unknown')),
                'venue': str(venue).upper().replace(' ', '_'),
                'grade': str(row.get('G', 'G5')).upper(),
                'track_condition': 'Good',  # Default value
                'weather': 'Fine',  # Default value
                'temperature': 20.0,  # Default value
                'humidity': 60.0,  # Default value
                'wind_speed': 10.0,  # Default value
                'field_size': len(race_data),
                'race_date': race_date,
                'race_time': '14:30',  # Default race time
                # Add additional fields that ML System V4 might expect
                'distance': safe_float_convert(row.get('DIST'), 500.0),
                'margin': None,  # Upcoming race - no margin yet
                'individual_time': None,  # Upcoming race - no time yet
                'finish_position': None,  # Upcoming race - no finish position
                'performance_rating': safe_float_convert(row.get('PERF'), 0.0),
                'speed_rating': safe_float_convert(row.get('SPEED'), 0.0),
                'class_rating': safe_float_convert(row.get('CLASS'), 0.0)
            }
            
            mapped_data.append(mapped_row)
        
        if not mapped_data:
            logger.warning(f"No valid dog data found in {race_file_path}")
            return pd.DataFrame()
        
        # Create DataFrame and ensure proper data types
        result_df = pd.DataFrame(mapped_data)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['box_number', 'weight', 'starting_price', 'temperature', 
                          'humidity', 'wind_speed', 'field_size', 'distance', 
                          'performance_rating', 'speed_rating', 'class_rating']
        
        for col in numeric_columns:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0)
        
        logger.info(f"📋 Mapped {len(result_df)} dogs for ML System V4 prediction")
        return result_df
