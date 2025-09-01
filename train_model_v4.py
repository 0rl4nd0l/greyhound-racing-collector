#!/usr/bin/env python3
"""
MLSystemV4 Training Script with Conditional Retraining
======================================================

This script provides conditional model retraining functionality for MLSystemV4:
- Checks for data schema changes
- Monitors data volume thresholds
- Detects model performance degradation flags
- Retrains models when conditions are met
- Saves artifacts with proper versioning and manifests

Features:
- Conditional retraining based on multiple triggers
- Automatic artifact management with timestamps
- Git commit hash tracking
- Dataset checksum verification
- Comprehensive manifest generation
- Integration with monitoring systems

Author: AI Assistant
Date: August 3, 2025
"""

import hashlib
import json
import logging
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_monitoring import DataMonitor
from drift_monitor import DriftMonitor
from ml_system_v4 import MLSystemV4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train_model_v4.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class ConditionalRetrainingManager:
    """Manages conditional retraining of MLSystemV4 based on various triggers."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        # Resolve DB path to an absolute, stable location to avoid accidental empty DB creation
        env_db = os.getenv('GREYHOUND_DB_PATH') or os.getenv('DATABASE_PATH')
        # Prefer explicit env var, else use provided db_path arg, else fall back to repo-root default
        candidate = env_db if (env_db and str(env_db).strip()) else db_path
        if candidate and str(candidate).strip():
            resolved_db = str(Path(candidate).expanduser().resolve())
        else:
            # Default to repo-root greyhound_racing_data.db
            resolved_db = str((Path(__file__).resolve().parent / 'greyhound_racing_data.db').resolve())
        self.db_path = resolved_db
        self.ml_system = MLSystemV4(self.db_path)
        self.data_monitor = DataMonitor(self.db_path)
        
        # Retraining thresholds and conditions
        self.thresholds = {
            "data_volume_increase_pct": 20.0,  # 20% increase in data volume
            "schema_change_threshold": 0.1,    # 10% change in schema
            "performance_drop_threshold": 0.05, # 5% performance drop
            "drift_threshold": 0.25,           # PSI > 0.25 indicates high drift
            "days_since_last_training": 7      # Retrain every 7 days minimum
        }
        
        # Current state tracking
        self.current_state = {
            "last_training_timestamp": None,
            "last_data_volume": 0,
            "last_schema_hash": None,
            "last_performance_metrics": {},
            "last_dataset_checksum": None
        }
        
        self._load_current_state()
        
        logger.info("🎯 Conditional Retraining Manager initialized")
        logger.info(f"   Data volume threshold: {self.thresholds['data_volume_increase_pct']}%")
        logger.info(f"   Performance drop threshold: {self.thresholds['performance_drop_threshold']}")
        logger.info(f"   Drift threshold: {self.thresholds['drift_threshold']}")
    
    def _load_current_state(self):
        """Load current state from previous training runs."""
        state_file = Path("ml_models_v4/training_state.json")
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    stored_state = json.load(f)
                    self.current_state.update(stored_state)
                    logger.info(f"📥 Loaded training state from {state_file}")
            except Exception as e:
                logger.warning(f"Could not load training state: {e}")
    
    def _save_current_state(self):
        """Save current state for future training runs."""
        state_file = Path("ml_models_v4/training_state.json")
        state_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(state_file, 'w') as f:
                json.dump(self.current_state, f, indent=2, default=str)
                logger.info(f"💾 Saved training state to {state_file}")
        except Exception as e:
            logger.error(f"Could not save training state: {e}")
    
    def check_data_schema_changes(self) -> Tuple[bool, Dict]:
        """Check for significant data schema changes."""
        logger.info("🔍 Checking for data schema changes...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current schema information
            schema_info = {}
            tables = ['race_metadata', 'dog_race_data', 'enhanced_expert_data']
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                schema_info[table] = {
                    'column_count': len(columns),
                    'columns': [col[1] for col in columns],  # column names
                    'column_types': [col[2] for col in columns]  # column types
                }
            
            conn.close()
            
            # Calculate schema hash
            schema_str = json.dumps(schema_info, sort_keys=True)
            current_schema_hash = hashlib.md5(schema_str.encode()).hexdigest()
            
            # Compare with previous schema
            schema_changed = False
            change_details = {}
            
            if self.current_state['last_schema_hash']:
                if current_schema_hash != self.current_state['last_schema_hash']:
                    schema_changed = True
                    change_details = {
                        'previous_hash': self.current_state['last_schema_hash'],
                        'current_hash': current_schema_hash,
                        'schema_info': schema_info
                    }
                    logger.info("📊 Schema change detected!")
                else:
                    logger.info("✅ No schema changes detected")
            else:
                logger.info("📋 First time schema check - recording baseline")
            
            # Update current state
            self.current_state['last_schema_hash'] = current_schema_hash
            
            return schema_changed, change_details
            
        except Exception as e:
            logger.error(f"Error checking schema changes: {e}")
            return False, {}
    
    def check_data_volume_threshold(self) -> Tuple[bool, Dict]:
        """Check if data volume has crossed the threshold."""
        logger.info("📊 Checking data volume threshold...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current data volumes
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            race_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            dog_race_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM enhanced_expert_data")
            expert_data_count = cursor.fetchone()[0]
            
            conn.close()
            
            current_volume = race_count + dog_race_count + expert_data_count
            
            # Calculate volume change
            volume_changed = False
            change_details = {}
            
            if self.current_state['last_data_volume'] > 0:
                volume_increase_pct = (
                    (current_volume - self.current_state['last_data_volume']) / 
                    self.current_state['last_data_volume'] * 100
                )
                
                if volume_increase_pct >= self.thresholds['data_volume_increase_pct']:
                    volume_changed = True
                    change_details = {
                        'previous_volume': self.current_state['last_data_volume'],
                        'current_volume': current_volume,
                        'increase_pct': round(volume_increase_pct, 2),
                        'threshold_pct': self.thresholds['data_volume_increase_pct']
                    }
                    logger.info(f"📈 Data volume increase: {volume_increase_pct:.1f}% (threshold: {self.thresholds['data_volume_increase_pct']}%)")
                else:
                    logger.info(f"📊 Data volume increase: {volume_increase_pct:.1f}% (below threshold)")
            else:
                logger.info("📋 First time volume check - recording baseline")
            
            # Update current state
            self.current_state['last_data_volume'] = current_volume
            
            return volume_changed, change_details
            
        except Exception as e:
            logger.error(f"Error checking data volume: {e}")
            return False, {}
    
    def check_model_performance_flags(self) -> Tuple[bool, Dict]:
        """Check for model performance degradation flags."""
        logger.info("🎯 Checking model performance flags...")
        
        try:
            # Check for performance alerts from monitoring service
            alerts_file = Path("model_alerts.json")
            
            if not alerts_file.exists():
                logger.info("📋 No performance alerts file found")
                return False, {}
            
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
            
            # Look for recent performance degradation alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_performance_alerts = []
            
            for alert in alerts:
                if alert.get('type') in ['performance_warning', 'performance_critical']:
                    alert_time = datetime.fromisoformat(alert['timestamp'])
                    if alert_time > cutoff_time:
                        recent_performance_alerts.append(alert)
            
            performance_flag_raised = len(recent_performance_alerts) > 0
            
            if performance_flag_raised:
                logger.info(f"🚨 {len(recent_performance_alerts)} recent performance alerts found")
            else:
                logger.info("✅ No recent performance degradation detected")
            
            return performance_flag_raised, {
                'recent_alerts': recent_performance_alerts,
                'alert_count': len(recent_performance_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error checking performance flags: {e}")
            return False, {}
    
    def check_data_drift(self) -> Tuple[bool, Dict]:
        """Check for significant data drift."""
        logger.info("🌊 Checking for data drift...")
        
        try:
            # Load reference data from the last training
            reference_data_file = Path("ml_models_v4/reference_training_data.parquet")
            
            if not reference_data_file.exists():
                logger.info("📋 No reference data found - will create after training")
                return False, {}
            
            reference_data = pd.read_parquet(reference_data_file)
            
            # Get recent data for comparison
            conn = sqlite3.connect(self.db_path)
            recent_cutoff = datetime.now() - timedelta(days=7)
            
            query = """
            SELECT d.*, r.venue, r.grade, r.distance, r.track_condition, r.weather,
                   r.temperature, r.humidity, r.wind_speed, r.field_size,
                   r.race_date, r.race_time, r.winner_name, r.winner_odds,
                   e.pir_rating, e.first_sectional, e.win_time, e.bonus_time
            FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            LEFT JOIN enhanced_expert_data e ON d.race_id = e.race_id 
                AND d.dog_clean_name = e.dog_clean_name
            WHERE r.race_date >= ?
            ORDER BY r.race_date DESC
            LIMIT 1000
            """
            
            recent_data = pd.read_sql_query(query, conn, params=[recent_cutoff.strftime('%Y-%m-%d')])
            conn.close()
            
            if recent_data.empty:
                logger.info("📊 No recent data available for drift comparison")
                return False, {}
            
            # Initialize drift monitor and check for drift
            drift_monitor = DriftMonitor(reference_data)
            drift_results = drift_monitor.check_for_drift(recent_data)
            
            # Determine if drift is significant enough to trigger retraining
            high_drift_features = drift_results.get('summary', {}).get('high_drift_features', [])
            drift_detected = len(high_drift_features) > 0
            
            logger.info(f"🌊 Drift analysis complete: {len(high_drift_features)} high-drift features")
            
            return drift_detected, drift_results
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return False, {}
    
    def check_time_threshold(self) -> Tuple[bool, Dict]:
        """Check if enough time has passed since last training."""
        logger.info("⏰ Checking time threshold...")
        
        time_threshold_crossed = False
        time_details = {}
        
        if self.current_state['last_training_timestamp']:
            last_training = datetime.fromisoformat(self.current_state['last_training_timestamp'])
            days_since_training = (datetime.now() - last_training).days
            
            if days_since_training >= self.thresholds['days_since_last_training']:
                time_threshold_crossed = True
                time_details = {
                    'last_training': self.current_state['last_training_timestamp'],
                    'days_since_training': days_since_training,
                    'threshold_days': self.thresholds['days_since_last_training']
                }
                logger.info(f"⏰ Time threshold crossed: {days_since_training} days since last training")
            else:
                logger.info(f"⏰ {days_since_training} days since last training (threshold: {self.thresholds['days_since_last_training']})")
        else:
            logger.info("📋 First time training - no previous timestamp")
            time_threshold_crossed = True  # First training
            time_details = {'first_training': True}
        
        return time_threshold_crossed, time_details
    
    def calculate_dataset_checksum(self) -> str:
        """Calculate checksum of current dataset."""
        logger.info("🔐 Calculating dataset checksum...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get data from key tables and create a hash
            tables_data = []
            
            # Race metadata
            df_races = pd.read_sql_query("SELECT * FROM race_metadata ORDER BY race_id", conn)
            tables_data.append(df_races.to_string())
            
            # Dog race data
            df_dogs = pd.read_sql_query("SELECT * FROM dog_race_data ORDER BY race_id, box_number", conn)
            tables_data.append(df_dogs.to_string())
            
            # Enhanced expert data
            df_expert = pd.read_sql_query("SELECT * FROM enhanced_expert_data ORDER BY race_id, dog_clean_name", conn)
            tables_data.append(df_expert.to_string())
            
            conn.close()
            
            # Create combined hash
            combined_data = "|||".join(tables_data)
            checksum = hashlib.sha256(combined_data.encode()).hexdigest()
            
            logger.info(f"🔐 Dataset checksum: {checksum[:16]}...")
            return checksum
            
        except Exception as e:
            logger.error(f"Error calculating dataset checksum: {e}")
            return "error_calculating_checksum"
    
    def get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
                logger.info(f"📝 Git commit hash: {commit_hash[:10]}...")
                return commit_hash
            else:
                logger.warning("Could not get git commit hash")
                return "unknown_commit"
        except Exception as e:
            logger.warning(f"Error getting git commit hash: {e}")
            return "unknown_commit"
    
    def create_training_manifest(self, retraining_triggers: List[str], 
                               model_metrics: Dict, dataset_checksum: str) -> Dict:
        """Create comprehensive training manifest."""
        timestamp = datetime.now()
        git_hash = self.get_git_commit_hash()
        
        manifest = {
            "training_metadata": {
                "timestamp": timestamp.isoformat(),
                "date_time": timestamp.strftime("%Y%m%d_%H%M%S"),
                "model_version": "v4",
                "git_commit_hash": git_hash,
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "training_triggers": retraining_triggers
            },
            "dataset_info": {
                "database_path": self.db_path,
                "dataset_checksum": dataset_checksum,
                "data_volume": self.current_state['last_data_volume'],
                "schema_hash": self.current_state['last_schema_hash']
            },
            "model_hyperparameters": {
                "model_type": "ExtraTreesClassifier_Calibrated",
                "n_estimators": 500,
                "min_samples_leaf": 3,
                "max_depth": 15,
                "calibration_method": "isotonic",
                "temporal_leakage_protection": True
            },
            "performance_metrics": model_metrics,
            "thresholds_used": self.thresholds.copy(),
            "artifacts": {
                "model_file": f"ml_model_v4_{timestamp.strftime('%Y%m%d_%H%M%S')}.joblib",
                "manifest_file": f"training_manifest_{timestamp.strftime('%Y%m%d_%H%M%S')}.json",
                "reference_data_file": "reference_training_data.parquet"
            }
        }
        
        return manifest
    
    def save_training_artifacts(self, manifest: Dict, training_data: pd.DataFrame):
        """Save training artifacts with proper organization."""
        timestamp = manifest["training_metadata"]["date_time"]
        artifacts_dir = Path(f"ml_models_v4/{timestamp}")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving training artifacts to {artifacts_dir}")
        
        try:
            # Save manifest
            manifest_file = artifacts_dir / f"training_manifest_{timestamp}.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            # Save reference training data for future drift detection
            reference_file = Path("ml_models_v4/reference_training_data.parquet")
            training_data.to_parquet(reference_file)
            
            # Also save a timestamped copy
            timestamped_reference = artifacts_dir / "reference_training_data.parquet"
            training_data.to_parquet(timestamped_reference)
            
            # Create artifact summary
            summary_file = artifacts_dir / "artifact_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"MLSystemV4 Training Artifacts - {timestamp}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Training Date: {manifest['training_metadata']['timestamp']}\n")
                f.write(f"Git Commit: {manifest['training_metadata']['git_commit_hash']}\n")
                f.write(f"Dataset Checksum: {manifest['dataset_info']['dataset_checksum']}\n")
                f.write(f"Triggers: {', '.join(manifest['training_metadata']['training_triggers'])}\n")
                f.write(f"Test Accuracy: {manifest['performance_metrics'].get('test_accuracy', 'N/A')}\n")
                f.write(f"Test AUC: {manifest['performance_metrics'].get('test_auc', 'N/A')}\n")
                f.write("\nArtifacts:\n")
                for artifact_type, filename in manifest["artifacts"].items():
                    f.write(f"  - {artifact_type}: {filename}\n")
            
            logger.info(f"✅ Training artifacts saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving training artifacts: {e}")
    
    def should_retrain_model(self) -> Tuple[bool, List[str], Dict]:
        """Determine if model should be retrained based on all conditions."""
        logger.info("🎯 Evaluating retraining conditions...")
        
        retraining_triggers = []
        condition_details = {}
        
        # Check all conditions
        conditions = [
            ("data_schema_change", self.check_data_schema_changes),
            ("data_volume_threshold", self.check_data_volume_threshold),
            ("model_performance_flags", self.check_model_performance_flags),
            ("data_drift", self.check_data_drift),
            ("time_threshold", self.check_time_threshold)
        ]
        
        for condition_name, check_function in conditions:
            try:
                condition_met, details = check_function()
                condition_details[condition_name] = {
                    'condition_met': condition_met,
                    'details': details
                }
                
                if condition_met:
                    retraining_triggers.append(condition_name)
                    logger.info(f"✅ Trigger: {condition_name}")
                else:
                    logger.info(f"❌ No trigger: {condition_name}")
                    
            except Exception as e:
                logger.error(f"Error checking condition {condition_name}: {e}")
                condition_details[condition_name] = {
                    'condition_met': False,
                    'error': str(e)
                }
        
        should_retrain = len(retraining_triggers) > 0
        
        logger.info(f"🎯 Retraining decision: {'YES' if should_retrain else 'NO'}")
        if should_retrain:
            logger.info(f"🚀 Triggers: {', '.join(retraining_triggers)}")
        
        return should_retrain, retraining_triggers, condition_details
    
    def execute_retraining(self) -> bool:
        """Execute the actual model retraining process."""
        logger.info("🚀 Starting model retraining process...")
        
        try:
            # Prepare training data and save reference for drift detection
            train_data, test_data = self.ml_system.prepare_time_ordered_data()
            
            if train_data.empty:
                logger.error("❌ No training data available")
                return False
            
            # Calculate dataset checksum
            dataset_checksum = self.calculate_dataset_checksum()
            
            # Train the model
            training_success = self.ml_system.train_model()
            
            if not training_success:
                logger.error("❌ Model training failed")
                return False
            
            # Create training manifest
            retraining_triggers, _, _ = self.should_retrain_model()
            manifest = self.create_training_manifest(
                retraining_triggers,
                self.ml_system.model_info,
                dataset_checksum
            )
            
            # Save artifacts
            self.save_training_artifacts(manifest, train_data)
            
            # Update current state
            self.current_state.update({
                'last_training_timestamp': datetime.now().isoformat(),
                'last_performance_metrics': self.ml_system.model_info,
                'last_dataset_checksum': dataset_checksum
            })
            
            self._save_current_state()
            
            logger.info("✅ Model retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during retraining: {e}")
            return False


def main():
    """Main function for conditional model retraining."""
    print("🎯 MLSystemV4 Conditional Retraining")
    print("=" * 50)
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Initialize retraining manager
        manager = ConditionalRetrainingManager()
        
        # Check if retraining is needed
        should_retrain, triggers, details = manager.should_retrain_model()
        
        print(f"\n📊 Retraining Evaluation Results:")
        print(f"   Should retrain: {'✅ YES' if should_retrain else '❌ NO'}")
        
        if should_retrain:
            print(f"   Triggers: {', '.join(triggers)}")
            
            # Execute retraining
            print("\n🚀 Starting retraining process...")
            success = manager.execute_retraining()
            
            if success:
                print("✅ Retraining completed successfully!")
                return 0
            else:
                print("❌ Retraining failed!")
                return 1
        else:
            print("✅ No retraining needed at this time")
            
            # Log details for monitoring
            logger.info("📋 Condition evaluation summary:")
            for condition, info in details.items():
                logger.info(f"   {condition}: {'✅' if info['condition_met'] else '❌'}")
            
            return 0
            
    except Exception as e:
        print(f"❌ Error in retraining process: {e}")
        logger.error(f"Critical error in main retraining process: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
