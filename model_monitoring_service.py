#!/usr/bin/env python3
"""
Model Monitoring Service
========================

This service provides continuous monitoring and automatic reloading of ML models:
- Periodic checks for new models in the registry
- Performance drift detection
- Automatic model reloading when better models are available
- Health checks and alert system
- Model performance tracking

Author: AI Assistant
Date: July 27, 2025
"""

import os
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMonitoringService:
    def __init__(self, check_interval_minutes=30):
        self.check_interval_minutes = check_interval_minutes
        self.is_running = False
        self.monitor_thread = None
        self.performance_history = []
        self.current_model_info = None
        self.alerts = []
        
        # Performance thresholds
        self.thresholds = {
            'accuracy_drop_warning': 0.05,  # 5% drop triggers warning
            'accuracy_drop_critical': 0.10,  # 10% drop triggers reload
            'performance_window_hours': 24,  # Performance tracking window
            'min_predictions_for_analysis': 50  # Minimum predictions needed
        }
        
        # Monitoring statistics
        self.stats = {
            'models_checked': 0,
            'models_reloaded': 0,
            'performance_checks': 0,
            'alerts_generated': 0,
            'last_check_time': None,
            'last_reload_time': None,
            'service_start_time': datetime.now()
        }
        
        logger.info("🔍 Model Monitoring Service initialized")
        logger.info(f"   Check interval: {check_interval_minutes} minutes")
    
    def start_monitoring(self):
        """Start the model monitoring service"""
        if self.is_running:
            logger.warning("Monitoring service is already running")
            return
        
        self.is_running = True
        
        # Schedule periodic checks
        schedule.every(self.check_interval_minutes).minutes.do(self._check_for_model_updates)
        schedule.every(1).hours.do(self._analyze_performance_drift)
        schedule.every(6).hours.do(self._cleanup_old_data)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("✅ Model monitoring service started")
        logger.info(f"   🔄 Checking for updates every {self.check_interval_minutes} minutes")
        logger.info("   📊 Performance analysis every hour")
        logger.info("   🧹 Data cleanup every 6 hours")
    
    def stop_monitoring(self):
        """Stop the model monitoring service"""
        if not self.is_running:
            return
        
        self.is_running = False
        schedule.clear()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("🔴 Model monitoring service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("🚀 Starting monitoring loop...")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute for scheduled tasks
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_for_model_updates(self):
        """Check for new or better models in the registry"""
        try:
            from model_registry import get_model_registry
            
            registry = get_model_registry()
            self.stats['models_checked'] += 1
            self.stats['last_check_time'] = datetime.now()
            
            # Get current best model
            best_model_result = registry.get_best_model()
            if not best_model_result:
                logger.warning("No models found in registry")
                return
            
            model, scaler, metadata = best_model_result
            
            # Check if this is a new model
            if (self.current_model_info is None or 
                self.current_model_info.get('model_id') != metadata.model_id):
                
                logger.info(f"🔄 New best model detected: {metadata.model_id}")
                logger.info(f"   📊 Performance: Acc={metadata.accuracy:.3f}, AUC={metadata.auc:.3f}")
                logger.info(f"   🕐 Training: {metadata.training_timestamp[:19]}")
                
                # Trigger model reload in applications
                self._trigger_model_reload(metadata)
                
                self.current_model_info = {
                    'model_id': metadata.model_id,
                    'accuracy': metadata.accuracy,
                    'auc': metadata.auc,
                    'f1_score': metadata.f1_score,
                    'timestamp': metadata.training_timestamp,
                    'reload_time': datetime.now().isoformat()
                }
                
                self.stats['models_reloaded'] += 1
                self.stats['last_reload_time'] = datetime.now()
                
                # Generate alert
                self._generate_alert('model_updated', {
                    'model_id': metadata.model_id,
                    'accuracy': metadata.accuracy,
                    'auc': metadata.auc
                })
            
            logger.debug(f"✓ Model check completed (current: {metadata.model_id})")
            
        except ImportError:
            logger.warning("Model registry not available")
        except Exception as e:
            logger.error(f"Error checking for model updates: {e}")
    
    def _trigger_model_reload(self, metadata):
        """Trigger model reload in running applications"""
        try:
            # Create reload signal file
            reload_signal_path = Path('./model_reload_signal.json')
            reload_info = {
                'model_id': metadata.model_id,
                'accuracy': metadata.accuracy,
                'auc': metadata.auc,
                'f1_score': metadata.f1_score,
                'timestamp': datetime.now().isoformat(),
                'action': 'reload_model'
            }
            
            with open(reload_signal_path, 'w') as f:
                json.dump(reload_info, f, indent=2)
            
            logger.info(f"📡 Model reload signal created: {reload_signal_path}")
            
            # Also update a model status file for applications to check
            status_path = Path('./current_model_status.json')
            status_info = {
                'current_model_id': metadata.model_id,
                'model_name': metadata.model_name,
                'model_type': metadata.model_type,
                'accuracy': metadata.accuracy,
                'auc': metadata.auc,
                'f1_score': metadata.f1_score,
                'training_timestamp': metadata.training_timestamp,
                'last_updated': datetime.now().isoformat(),
                'feature_count': len(metadata.feature_names),
                'training_samples': metadata.training_samples
            }
            
            with open(status_path, 'w') as f:
                json.dump(status_info, f, indent=2)
            
            logger.info(f"📋 Model status updated: {status_path}")
            
        except Exception as e:
            logger.error(f"Error triggering model reload: {e}")
    
    def _analyze_performance_drift(self):
        """Analyze performance drift and detect degradation"""
        try:
            self.stats['performance_checks'] += 1
            
            if len(self.performance_history) < self.thresholds['min_predictions_for_analysis']:
                logger.debug("Insufficient prediction history for performance analysis")
                return
            
            # Get recent performance data
            cutoff_time = datetime.now() - timedelta(hours=self.thresholds['performance_window_hours'])
            recent_performance = [
                entry for entry in self.performance_history 
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
            
            if not recent_performance:
                logger.debug("No recent performance data available")
                return
            
            # Calculate performance metrics
            accuracies = [entry['accuracy'] for entry in recent_performance if 'accuracy' in entry]
            predictions_count = sum(entry.get('predictions_count', 0) for entry in recent_performance)
            
            if not accuracies:
                logger.debug("No accuracy data in recent performance history")
                return
            
            current_accuracy = sum(accuracies) / len(accuracies)
            
            # Compare with baseline (stored in current model info)
            if self.current_model_info and 'accuracy' in self.current_model_info:
                baseline_accuracy = self.current_model_info['accuracy']
                accuracy_drop = baseline_accuracy - current_accuracy
                
                logger.info(f"📊 Performance Analysis:")
                logger.info(f"   Baseline accuracy: {baseline_accuracy:.3f}")
                logger.info(f"   Current accuracy: {current_accuracy:.3f}")
                logger.info(f"   Performance drop: {accuracy_drop:.3f}")
                logger.info(f"   Predictions analyzed: {predictions_count}")
                
                # Check thresholds
                if accuracy_drop > self.thresholds['accuracy_drop_critical']:
                    self._generate_alert('performance_critical', {
                        'baseline_accuracy': baseline_accuracy,
                        'current_accuracy': current_accuracy,
                        'accuracy_drop': accuracy_drop,
                        'predictions_count': predictions_count
                    })
                    logger.warning(f"🚨 CRITICAL: Performance drop of {accuracy_drop:.3f}")
                    
                elif accuracy_drop > self.thresholds['accuracy_drop_warning']:
                    self._generate_alert('performance_warning', {
                        'baseline_accuracy': baseline_accuracy,
                        'current_accuracy': current_accuracy,
                        'accuracy_drop': accuracy_drop,
                        'predictions_count': predictions_count
                    })
                    logger.warning(f"⚠️ WARNING: Performance drop of {accuracy_drop:.3f}")
                
                else:
                    logger.info("✅ Model performance within acceptable range")
            
        except Exception as e:
            logger.error(f"Error analyzing performance drift: {e}")
    
    def _generate_alert(self, alert_type: str, data: Dict[str, Any]):
        """Generate and store alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'data': data,
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        self.stats['alerts_generated'] += 1
        
        # Keep only last 100 alerts
        self.alerts = self.alerts[-100:]
        
        # Save alert to file
        alerts_file = Path('./model_alerts.json')
        try:
            with open(alerts_file, 'w') as f:
                json.dump(self.alerts, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
        
        logger.info(f"🚨 Alert generated: {alert_type} ({alert['severity']})")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level"""
        severity_map = {
            'model_updated': 'info',
            'performance_warning': 'warning',
            'performance_critical': 'critical',
            'model_load_error': 'error'
        }
        return severity_map.get(alert_type, 'info')
    
    def _cleanup_old_data(self):
        """Clean up old performance history and alerts"""
        try:
            # Clean up old performance history (keep last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            original_count = len(self.performance_history)
            
            self.performance_history = [
                entry for entry in self.performance_history 
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
            
            cleaned_count = original_count - len(self.performance_history)
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} old performance records")
            
            # Clean up old alerts (keep last 30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            original_alert_count = len(self.alerts)
            
            self.alerts = [
                alert for alert in self.alerts 
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time
            ]
            
            cleaned_alert_count = original_alert_count - len(self.alerts)
            if cleaned_alert_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_alert_count} old alerts")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def record_prediction_performance(self, predictions: List[Dict], actual_results: List[int] = None):
        """Record prediction performance for drift analysis"""
        try:
            timestamp = datetime.now().isoformat()
            
            performance_entry = {
                'timestamp': timestamp,
                'predictions_count': len(predictions),
                'average_confidence': sum(p.get('confidence_score', 0.5) for p in predictions) / len(predictions),
                'score_variance': self._calculate_score_variance(predictions)
            }
            
            # Calculate accuracy if actual results provided
            if actual_results and len(actual_results) == len(predictions):
                correct = sum(
                    1 for pred, actual in zip(predictions, actual_results)
                    if (pred.get('prediction_score', 0.5) > 0.5) == (actual == 1)
                )
                performance_entry['accuracy'] = correct / len(predictions)
            
            self.performance_history.append(performance_entry)
            
            # Keep only last 1000 entries
            self.performance_history = self.performance_history[-1000:]
            
            logger.debug(f"📈 Recorded performance: {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error recording prediction performance: {e}")
    
    def _calculate_score_variance(self, predictions: List[Dict]) -> float:
        """Calculate variance in prediction scores"""
        try:
            scores = [p.get('prediction_score', 0.5) for p in predictions]
            if len(scores) < 2:
                return 0.0
            
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / (len(scores) - 1)
            return variance
        except:
            return 0.0
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'is_running': self.is_running,
            'stats': self.stats.copy(),
            'current_model': self.current_model_info,
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'performance_history_size': len(self.performance_history),
            'thresholds': self.thresholds.copy(),
            'uptime_hours': (datetime.now() - self.stats['service_start_time']).total_seconds() / 3600
        }


# Global monitoring service instance
_monitoring_service = None
_service_lock = threading.Lock()

def get_monitoring_service() -> ModelMonitoringService:
    """Get the global monitoring service instance (singleton)"""
    global _monitoring_service
    if _monitoring_service is None:
        with _service_lock:
            if _monitoring_service is None:
                _monitoring_service = ModelMonitoringService()
    return _monitoring_service


if __name__ == "__main__":
    # Example usage
    service = get_monitoring_service()
    
    print("🚀 Starting Model Monitoring Service")
    print("=" * 50)
    
    try:
        service.start_monitoring()
        
        # Keep service running
        while True:
            time.sleep(10)
            status = service.get_monitoring_status()
            
            if status['stats']['last_check_time']:
                last_check = datetime.fromisoformat(status['stats']['last_check_time'])
                minutes_since_check = (datetime.now() - last_check).total_seconds() / 60
                print(f"⏰ Last check: {minutes_since_check:.1f} minutes ago")
            
            if status['recent_alerts']:
                print(f"🚨 Recent alerts: {len(status['recent_alerts'])}")
    
    except KeyboardInterrupt:
        print("\n🔴 Shutting down monitoring service...")
        service.stop_monitoring()
        print("✅ Service stopped")
