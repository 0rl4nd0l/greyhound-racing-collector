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

    def predict_race_file(self, race_file_path: str, tgr_enabled: bool | None = None) -> dict:
        """Main prediction method using ML System V4.
        
        Args:
            race_file_path: Path to upcoming race CSV
            tgr_enabled: Optional runtime toggle to include TGR features (DB-only) in predictions
        """
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

            # Apply runtime TGR toggle if provided
            try:
                if tgr_enabled is not None and hasattr(self.ml_system_v4, 'set_tgr_enabled'):
                    self.ml_system_v4.set_tgr_enabled(bool(tgr_enabled))
            except Exception:
                pass

            # Perform prediction with V4 system
            result = self.ml_system_v4.predict_race(race_data, race_id)

            # Enrich metadata and race context for UI/consumers
            try:
                if isinstance(result, dict) and result.get('success'):
                    # Predictor/methods/version defaults
                    result.setdefault('predictor_used', 'PredictionPipelineV4')
                    if not result.get('prediction_methods_used'):
                        result['prediction_methods_used'] = ['ml_system']
                    result.setdefault('analysis_version', 'ML System V4')

                    # Normalize prediction item keys for frontend compatibility
                    try:
                        preds = result.get('predictions') or result.get('enhanced_predictions') or []
                        if isinstance(preds, list):
                            for p in preds:
                                if isinstance(p, dict):
                                    if 'dog_name' not in p and 'dog_clean_name' in p:
                                        p['dog_name'] = p.get('dog_clean_name')
                                    if 'name' not in p and 'dog_clean_name' in p:
                                        p['name'] = p.get('dog_clean_name')
                                    # Normalize probability keys for UI consumers
                                    if p.get('win_prob') is None and p.get('win_probability') is not None:
                                        try:
                                            wp = float(p.get('win_probability'))
                                            p['win_prob'] = max(0.0, min(1.0, wp))
                                        except Exception:
                                            pass
                                    if p.get('win_probability') is None and p.get('win_prob_norm') is not None:
                                        try:
                                            wp2 = float(p.get('win_prob_norm'))
                                            p['win_probability'] = max(0.0, min(1.0, wp2))
                                        except Exception:
                                            pass
                    except Exception:
                        pass

                    # Inject CSV-derived historical stats from the enriched input DataFrame so UI can fallback gracefully
                    try:
                        preds = result.get('predictions') or result.get('enhanced_predictions') or []
                        if isinstance(preds, list) and len(preds) > 0:
                            # Build a normalization helper and a lookup map from race_data
                            def _norm(s: str) -> str:
                                try:
                                    import re
                                    return re.sub(r"[^A-Za-z0-9]", "", (s or '').upper())
                                except Exception:
                                    return (s or '').upper().replace(' ', '')
                            try:
                                # race_data in this scope is the already-enriched DataFrame
                                csv_cols = [
                                    'csv_historical_races','csv_avg_finish_position','csv_best_finish_position',
                                    'csv_recent_form','csv_win_rate','csv_place_rate','csv_avg_time','csv_best_time'
                                ]
                                lookup = {}
                                if isinstance(race_data, pd.DataFrame) and 'dog_clean_name' in race_data.columns:
                                    for _, row_df in race_data.iterrows():
                                        key = _norm(str(row_df.get('dog_clean_name')))
                                        if not key:
                                            continue
                                        entry = {}
                                        for c in csv_cols:
                                            if c in race_data.columns and pd.notna(row_df.get(c)):
                                                entry[c] = row_df.get(c)
                                        if entry:
                                            lookup[key] = entry
                                # Merge into predictions
                                for p in preds:
                                    if not isinstance(p, dict):
                                        continue
                                    dn = p.get('dog_clean_name') or p.get('dog_name') or p.get('name')
                                    key = _norm(str(dn))
                                    if key in lookup:
                                        for k, v in lookup[key].items():
                                            # Do not overwrite if already present
                                            if p.get(k) is None:
                                                p[k] = v
                                    # Ensure presence of csv_historical_races key for downstream UI logic
                                    if 'csv_historical_races' not in p:
                                        p['csv_historical_races'] = 0
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Race context
                    try:
                        cols = set(race_data.columns)
                        rc_venue = str(race_data['venue'].iloc[0]) if 'venue' in cols and len(race_data) > 0 else None
                        rc_date = str(race_data['race_date'].iloc[0]) if 'race_date' in cols and len(race_data) > 0 else None
                        rc_distance = None
                        if 'distance' in cols and len(race_data) > 0:
                            _dist = pd.to_numeric(race_data['distance'], errors='coerce').dropna()
                            if len(_dist) > 0:
                                try:
                                    rc_distance = int(_dist.mode().iloc[0]) if not _dist.mode().empty else int(_dist.iloc[0])
                                except Exception:
                                    rc_distance = float(_dist.mode().iloc[0]) if not _dist.mode().empty else float(_dist.iloc[0])
                        rc_grade = None
                        if 'grade' in cols and len(race_data) > 0:
                            rc_grade = str(race_data['grade'].iloc[0])
                        total_dogs = int(race_data['dog_clean_name'].nunique()) if 'dog_clean_name' in cols else int(len(race_data))

                        result.setdefault('race_context', {
                            'filename': os.path.basename(race_file_path),
                            'venue': rc_venue,
                            'race_date': rc_date,
                            'distance': f"{int(rc_distance)}m" if rc_distance is not None else None,
                            'grade': rc_grade,
                            'total_dogs': total_dogs,
                        })
                    except Exception:
                        # Soft-fail race_context enrichment
                        pass
            except Exception:
                pass

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
        """Map CSV columns to the expected ML System V4 format with proper data type handling.
        
        IMPORTANT: This CSV format contains race participants followed by their historical data.
        - Participants have actual dog names (e.g., "2. Austrian Rose")
        - Historical data rows have empty dog names (shown as '""' in CSV)
        - We extract both participants AND their embedded historical data for enrichment.
        """
        
        # Extract race information from filename and first row
        filename = os.path.basename(race_file_path)
        
        # Parse race date from filename (e.g., "Race 1 - GOUL - 01 August 2025.csv" or ISO "Race 7 - MURR - 2025-08-24.csv")
        parts = filename.replace('.csv', '').split(' - ')
        race_date = datetime.now().strftime("%Y-%m-%d")
        venue = "Unknown"
        if len(parts) >= 3:
            date_part = parts[2]
            # Try multiple common formats
            for fmt in ("%d %B %Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
                try:
                    race_date = datetime.strptime(date_part, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            venue = parts[1] if len(parts) > 1 else "Unknown"
        
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
        
        def _is_participant_row(dog_name: str) -> bool:
            """Determine if this row represents a race participant (vs historical data).
            
            Participants have actual dog names, often with box number prefixes.
            Historical data rows have empty names or just quotes.
            """
            if not dog_name or dog_name.lower() in ['nan', 'none', '']:
                return False
            
            # Empty string after normalization (historical data)
            if dog_name == '""' or dog_name == '':
                return False
            
            # If it starts with a number and dot, it's likely a participant (e.g., "2. Austrian Rose")
            if '.' in dog_name and dog_name.split('.')[0].strip().isdigit():
                return True
            
            # If it has substantial text content, treat as participant
            clean_name = dog_name.replace('"', '').strip()
            if len(clean_name) > 2 and not clean_name.isdigit():
                return True
            
            return False
        
        participant_count = 0
        for _, row in race_data.iterrows():
            dog_name = _normalize_dog_name(row.get('Dog Name', ''))
            
            # Only process rows that represent actual race participants
            if not _is_participant_row(dog_name):
                logger.debug(f"Skipping historical data row with dog_name='{dog_name}'")
                continue  # Skip historical data rows
            
            participant_count += 1
            logger.debug(f"Processing participant row: '{dog_name}'")
            
            # Extract box number from dog name if it starts with a number
            try:
                if '.' in dog_name and dog_name.split('.')[0].strip().isdigit():
                    box_number = int(dog_name.split('.')[0].strip())
                    clean_dog_name = dog_name.split('.', 1)[1].strip() if '.' in dog_name else dog_name
                else:
                    # Try to get box number from BOX column
                    box_val = row.get('BOX', participant_count)
                    box_number = int(pd.to_numeric(box_val, errors='coerce')) if pd.notna(box_val) else participant_count
                    clean_dog_name = dog_name
            except (ValueError, TypeError):
                box_number = participant_count
                clean_dog_name = dog_name

            # Final normalization to match DB dog_clean_name conventions
            clean_dog_name = ' '.join(clean_dog_name.split())  # collapse whitespace
            # Keep proper case (Title Case) to match database format - don't convert to uppercase
            clean_dog_name = clean_dog_name.title()
            
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
'venue': str(venue).upper().replace(' ', '_').replace('/', '_'),
                'grade': str(row.get('G', 'G5')).upper(),
                'track_condition': 'Good',  # Default value
                'weather': 'Fine',  # Default value
                'temperature': 20.0,  # Default value
                'humidity': 60.0,  # Default value
                'wind_speed': 10.0,  # Default value
                'field_size': participant_count,  # Use actual participant count, not total rows
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
        
        # Extract embedded historical data for each dog and attach as supplementary info
        result_df = self._enrich_with_csv_historical_data(result_df, race_data)
        
        logger.info(f"📋 Mapped {len(result_df)} dogs for ML System V4 prediction")
        return result_df
    
    def _enrich_with_csv_historical_data(self, participants_df: pd.DataFrame, raw_csv_data: pd.DataFrame) -> pd.DataFrame:
        """Extract embedded historical data from CSV and attach to participant records.
        
        This supplements the database historical lookup with recent form data directly from the CSV.
        """
        logger.info("🔍 Extracting embedded historical data from CSV...")
        
        # Parse embedded historical data structure
        csv_historical_data = {}
        current_dog = None
        
        def _normalize_dog_name(name: str) -> str:
            if name is None:
                return ''
            s = str(name).replace('"', '').strip()
            return s
        
        for _, row in raw_csv_data.iterrows():
            dog_name = _normalize_dog_name(row.get('Dog Name', ''))
            
            if dog_name and '.' in dog_name and dog_name.split('.')[0].strip().isdigit():
                # This is a participant row
                clean_name = dog_name.split('.', 1)[1].strip().title()
                current_dog = clean_name
                csv_historical_data[current_dog] = []
            elif current_dog and not dog_name:
                # This is historical data for the current dog
                try:
                    historical_race = {
                        'date': row.get('DATE', ''),
                        'track': row.get('TRACK', ''),
                        'finish_position': int(row.get('PLC', 0)) if str(row.get('PLC', '')).isdigit() else None,
                        'time': float(row.get('TIME', 0)) if row.get('TIME') and str(row.get('TIME')).replace('.', '').isdigit() else None,
                        'distance': int(row.get('DIST', 0)) if str(row.get('DIST', '')).isdigit() else None,
                        'margin': float(row.get('MGN', 0)) if row.get('MGN') and str(row.get('MGN')).replace('.', '').isdigit() else None,
                        'weight': float(row.get('WGT', 0)) if row.get('WGT') and str(row.get('WGT')).replace('.', '').isdigit() else None
                    }
                    
                    # Only add if we have essential data
                    if historical_race['finish_position'] is not None:
                        csv_historical_data[current_dog].append(historical_race)
                        
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping malformed historical row for {current_dog}: {e}")
                    continue
        
        # Calculate CSV-based historical features for each participant
        enriched_participants = []
        
        for _, participant in participants_df.iterrows():
            participant_dict = participant.to_dict()
            dog_name = participant_dict['dog_clean_name']
            
            if dog_name in csv_historical_data and csv_historical_data[dog_name]:
                history = csv_historical_data[dog_name]
                
                # Calculate basic statistics from CSV historical data
                positions = [h['finish_position'] for h in history if h['finish_position'] is not None]
                times = [h['time'] for h in history if h['time'] is not None]
                
                if positions:
                    participant_dict['csv_historical_races'] = len(positions)
                    participant_dict['csv_avg_finish_position'] = sum(positions) / len(positions)
                    participant_dict['csv_best_finish_position'] = min(positions)
                    participant_dict['csv_recent_form'] = positions[0] if positions else None  # Most recent finish
                    participant_dict['csv_win_rate'] = len([p for p in positions if p == 1]) / len(positions)
                    participant_dict['csv_place_rate'] = len([p for p in positions if p <= 3]) / len(positions)
                    
                    if times:
                        participant_dict['csv_avg_time'] = sum(times) / len(times)
                        participant_dict['csv_best_time'] = min(times)
                    
                    logger.debug(f"{dog_name}: Found {len(positions)} CSV races, avg finish: {participant_dict['csv_avg_finish_position']:.1f}")
                else:
                    logger.debug(f"{dog_name}: No valid historical position data in CSV")
            else:
                # No CSV historical data for this dog
                participant_dict['csv_historical_races'] = 0
                logger.debug(f"{dog_name}: No CSV historical data found")
            
            enriched_participants.append(participant_dict)
        
        enriched_df = pd.DataFrame(enriched_participants)
        
        # Log summary of CSV enrichment
        dogs_with_csv_history = len([p for p in enriched_participants if p.get('csv_historical_races', 0) > 0])
        logger.info(f"📊 CSV Historical Enrichment: {dogs_with_csv_history}/{len(enriched_participants)} dogs have embedded historical data")
        
        return enriched_df
