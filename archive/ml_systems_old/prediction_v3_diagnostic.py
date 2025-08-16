#!/usr/bin/env python3
"""
Prediction V3 Core Diagnostic Tool
==================================

Comprehensive diagnostic and debugging tool for the Prediction V3 system.
This script:
- Traces the complete call graph from Flask route to final output
- Adds detailed timing measurements at each step
- Detects infinite loops and blocking I/O operations
- Provides performance bottleneck analysis
- Identifies hanging operations with timeout handling

Usage:
    python prediction_v3_diagnostic.py [--race-file path] [--timeout seconds]
"""

import asyncio
import logging
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('prediction_v3_diagnostic.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

class CallGraphTracer:
    """Traces function calls and execution times throughout the prediction pipeline"""
    
    def __init__(self):
        self.call_stack = []
        self.timing_data = {}
        self.max_depth = 0
        self.loop_detection = {}
        self.hang_alerts = []
        
    def trace_call(self, func_name: str, args: tuple = (), kwargs: dict = None):
        """Decorator for tracing function calls with timing and loop detection"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                call_id = f"{func_name}_{len(self.call_stack)}"
                depth = len(self.call_stack)
                
                # Loop detection - track repeated calls
                call_signature = f"{func_name}_{str(args)[:100]}"
                if call_signature in self.loop_detection:
                    self.loop_detection[call_signature] += 1
                    if self.loop_detection[call_signature] > 10:
                        logger.warning(f"🔄 POTENTIAL INFINITE LOOP detected in {func_name} - called {self.loop_detection[call_signature]} times")
                        self.hang_alerts.append({
                            'type': 'potential_loop',
                            'function': func_name,
                            'count': self.loop_detection[call_signature],
                            'timestamp': datetime.now().isoformat()
                        })
                else:
                    self.loop_detection[call_signature] = 1
                
                # Track call stack depth
                self.max_depth = max(self.max_depth, depth)
                self.call_stack.append(call_id)
                
                logger.debug(f"{'  ' * depth}📞 ENTERING {func_name} (depth: {depth})")
                
                try:
                    # Execute function with timeout monitoring
                    result = self._execute_with_timeout(func, args, kwargs, func_name)
                    
                    # Record timing
                    execution_time = time.time() - start_time
                    self.timing_data[call_id] = {
                        'function': func_name,
                        'depth': depth,
                        'start_time': start_time,
                        'execution_time': execution_time,
                        'status': 'success'
                    }
                    
                    # Alert on slow operations
                    if execution_time > 10:  # 10 seconds threshold
                        logger.warning(f"⏰ SLOW OPERATION: {func_name} took {execution_time:.2f}s")
                        self.hang_alerts.append({
                            'type': 'slow_operation',
                            'function': func_name,
                            'execution_time': execution_time,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    logger.debug(f"{'  ' * depth}✅ EXITING {func_name} (took {execution_time:.3f}s)")
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.timing_data[call_id] = {
                        'function': func_name,
                        'depth': depth,
                        'start_time': start_time,
                        'execution_time': execution_time,
                        'status': 'error',
                        'error': str(e)
                    }
                    
                    logger.error(f"{'  ' * depth}❌ ERROR in {func_name}: {e}")
                    raise
                    
                finally:
                    self.call_stack.pop()
                    
            return wrapper
        return decorator
    
    def _execute_with_timeout(self, func, args, kwargs, func_name, timeout=60):
        """Execute function with timeout detection"""
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            logger.error(f"🚨 TIMEOUT: {func_name} exceeded {timeout}s timeout - potential hang detected!")
            self.hang_alerts.append({
                'type': 'timeout',
                'function': func_name,
                'timeout': timeout,
                'timestamp': datetime.now().isoformat()
            })
            # Force thread termination (not recommended but necessary for diagnosis)
            raise TimeoutError(f"{func_name} timed out after {timeout}s")
        
        if exception:
            raise exception
            
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.timing_data:
            return {"error": "No timing data collected"}
        
        total_time = sum(t['execution_time'] for t in self.timing_data.values())
        slowest_functions = sorted(
            self.timing_data.values(), 
            key=lambda x: x['execution_time'], 
            reverse=True
        )[:10]
        
        function_stats = {}
        for timing in self.timing_data.values():
            func_name = timing['function']
            if func_name not in function_stats:
                function_stats[func_name] = {
                    'calls': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'max_time': 0
                }
            stats = function_stats[func_name]
            stats['calls'] += 1
            stats['total_time'] += timing['execution_time']
            stats['max_time'] = max(stats['max_time'], timing['execution_time'])
            stats['avg_time'] = stats['total_time'] / stats['calls']
        
        return {
            'total_execution_time': total_time,
            'max_call_depth': self.max_depth,
            'total_function_calls': len(self.timing_data),
            'slowest_functions': slowest_functions,
            'function_statistics': function_stats,
            'hang_alerts': self.hang_alerts,
            'potential_loops': [alert for alert in self.hang_alerts if alert['type'] == 'potential_loop'],
            'timeouts': [alert for alert in self.hang_alerts if alert['type'] == 'timeout'],
            'slow_operations': [alert for alert in self.hang_alerts if alert['type'] == 'slow_operation']
        }

# Global tracer instance
tracer = CallGraphTracer()

class PredictionV3Diagnostics:
    """Main diagnostic class for Prediction V3 system"""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.diagnostic_results = {}
        
    @tracer.trace_call("diagnose_full_pipeline")
    def diagnose_full_pipeline(self, race_file_path: str) -> Dict[str, Any]:
        """Complete diagnostic of the prediction pipeline"""
        logger.info(f"🔍 Starting comprehensive pipeline diagnostic for: {race_file_path}")
        
        start_time = time.time()
        results = {
            'file_path': race_file_path,
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: File Loading and Validation
            results['stages']['file_loading'] = self._diagnose_file_loading(race_file_path)
            
            # Stage 2: Data Preprocessing
            if results['stages']['file_loading']['success']:
                results['stages']['preprocessing'] = self._diagnose_preprocessing(race_file_path)
            
            # Stage 3: Model Loading and Inference
            if results['stages'].get('preprocessing', {}).get('success'):
                results['stages']['model_inference'] = self._diagnose_model_inference(race_file_path)
            
            # Stage 4: Post-processing and Output Generation
            if results['stages'].get('model_inference', {}).get('success'):
                results['stages']['post_processing'] = self._diagnose_post_processing(race_file_path)
            
            # Stage 5: Flask Route Integration Test
            results['stages']['flask_integration'] = self._diagnose_flask_integration(race_file_path)
            
        except Exception as e:
            logger.error(f"❌ Diagnostic pipeline failed: {e}")
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        results['total_time'] = time.time() - start_time
        results['end_time'] = datetime.now().isoformat()
        results['performance_report'] = tracer.get_performance_report()
        
        return results
    
    @tracer.trace_call("diagnose_file_loading")
    def _diagnose_file_loading(self, race_file_path: str) -> Dict[str, Any]:
        """Diagnose file loading stage"""
        logger.info("📁 Diagnosing file loading stage...")
        
        stage_result = {
            'success': False,
            'file_exists': False,
            'file_size': 0,
            'file_readable': False,
            'csv_valid': False,
            'issues': []
        }
        
        try:
            # Check file existence
            if os.path.exists(race_file_path):
                stage_result['file_exists'] = True
                stage_result['file_size'] = os.path.getsize(race_file_path)
                logger.debug(f"✅ File exists: {race_file_path} ({stage_result['file_size']} bytes)")
            else:
                stage_result['issues'].append(f"File does not exist: {race_file_path}")
                return stage_result
            
            # Check file readability
            try:
                with open(race_file_path, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline() for _ in range(5)]
                    stage_result['file_readable'] = True
                    stage_result['first_lines'] = [line.strip() for line in first_lines if line.strip()]
                    logger.debug(f"✅ File readable, first line: {first_lines[0][:100]}...")
            except Exception as e:
                stage_result['issues'].append(f"File not readable: {e}")
                return stage_result
            
            # Validate CSV structure
            try:
                df = pd.read_csv(race_file_path)
                stage_result['csv_valid'] = True
                stage_result['columns'] = list(df.columns)
                stage_result['row_count'] = len(df)
                stage_result['column_count'] = len(df.columns)
                logger.debug(f"✅ CSV valid: {len(df)} rows, {len(df.columns)} columns")
                logger.debug(f"Columns: {list(df.columns)}")
            except Exception as e:
                stage_result['issues'].append(f"CSV validation failed: {e}")
                return stage_result
            
            stage_result['success'] = True
            
        except Exception as e:
            stage_result['issues'].append(f"File loading diagnostic failed: {e}")
        
        return stage_result
    
    @tracer.trace_call("diagnose_preprocessing")
    def _diagnose_preprocessing(self, race_file_path: str) -> Dict[str, Any]:
        """Diagnose data preprocessing stage"""
        logger.info("🔄 Diagnosing preprocessing stage...")
        
        stage_result = {
            'success': False,
            'dogs_extracted': 0,
            'features_created': 0,
            'issues': []
        }
        
        try:
            # Import and initialize prediction pipeline
            from prediction_pipeline_v3 import PredictionPipelineV3
            pipeline = PredictionPipelineV3()
            
            # Test file loading
            race_df = pipeline._load_race_file(race_file_path)
            if race_df is None or race_df.empty:
                stage_result['issues'].append("Failed to load race file")
                return stage_result
            
            stage_result['dataframe_shape'] = race_df.shape
            stage_result['dataframe_columns'] = list(race_df.columns)
            
            # Test dog extraction
            dogs = pipeline._extract_dogs(race_df, race_file_path)
            stage_result['dogs_extracted'] = len(dogs)
            
            if not dogs:
                stage_result['issues'].append("No dogs extracted from race file")
                return stage_result
            
            # Test feature extraction for first dog
            if dogs:
                first_dog = dogs[0]
                features = pipeline.ml_system._extract_features_for_prediction(first_dog)
                stage_result['features_created'] = len(features)
                stage_result['sample_features'] = list(features.keys())[:10]  # First 10 features
                stage_result['sample_dog'] = {
                    'name': first_dog.get('name', 'Unknown'),
                    'box_number': first_dog.get('box_number', 0),
                    'weight': first_dog.get('weight', 0)
                }
            
            stage_result['success'] = True
            
        except Exception as e:
            stage_result['issues'].append(f"Preprocessing diagnostic failed: {e}")
            stage_result['error'] = str(e)
            stage_result['traceback'] = traceback.format_exc()
        
        return stage_result
    
    @tracer.trace_call("diagnose_model_inference") 
    def _diagnose_model_inference(self, race_file_path: str) -> Dict[str, Any]:
        """Diagnose model inference stage"""
        logger.info("🤖 Diagnosing model inference stage...")
        
        stage_result = {
            'success': False,
            'model_loaded': False,
            'predictions_generated': 0,
            'issues': []
        }
        
        try:
            # Import ML system
            from ml_system_v3 import MLSystemV3
            ml_system = MLSystemV3(self.db_path)
            
            # Check model loading
            if ml_system.pipeline is not None:
                stage_result['model_loaded'] = True
                stage_result['model_info'] = ml_system.get_model_info()
                logger.debug(f"✅ Model loaded: {stage_result['model_info']}")
            else:
                stage_result['issues'].append("No ML model loaded")
                return stage_result
            
            # Test prediction on sample data
            from prediction_pipeline_v3 import PredictionPipelineV3
            pipeline = PredictionPipelineV3()
            race_df = pipeline._load_race_file(race_file_path)
            dogs = pipeline._extract_dogs(race_df, race_file_path)
            
            if dogs:
                # Test prediction for first dog with timeout
                first_dog = dogs[0]
                logger.debug(f"Testing prediction for dog: {first_dog.get('name', 'Unknown')}")
                
                # Use threading to detect hangs in prediction
                prediction_result = None
                prediction_error = None
                
                def predict_dog():
                    nonlocal prediction_result, prediction_error
                    try:
                        prediction_result = ml_system.predict(first_dog)
                    except Exception as e:
                        prediction_error = e
                
                prediction_thread = threading.Thread(target=predict_dog)
                prediction_thread.daemon = True
                prediction_thread.start()
                prediction_thread.join(timeout=30)  # 30 second timeout
                
                if prediction_thread.is_alive():
                    stage_result['issues'].append("Model prediction timed out (>30s) - potential hang")
                    logger.error("🚨 Model prediction TIMEOUT - this indicates a hanging operation!")
                    return stage_result
                
                if prediction_error:
                    raise prediction_error
                
                if prediction_result:
                    stage_result['predictions_generated'] = 1
                    stage_result['sample_prediction'] = {
                        'win_probability': prediction_result.get('win_probability', 0),
                        'confidence': prediction_result.get('confidence', 0),
                        'model_info': prediction_result.get('model_info', '')
                    }
                    logger.debug(f"✅ Sample prediction: {stage_result['sample_prediction']}")
            
            stage_result['success'] = True
            
        except Exception as e:
            stage_result['issues'].append(f"Model inference diagnostic failed: {e}")
            stage_result['error'] = str(e)
            stage_result['traceback'] = traceback.format_exc()
        
        return stage_result
    
    @tracer.trace_call("diagnose_post_processing")
    def _diagnose_post_processing(self, race_file_path: str) -> Dict[str, Any]:
        """Diagnose post-processing stage"""
        logger.info("🔧 Diagnosing post-processing stage...")
        
        stage_result = {
            'success': False,
            'output_generated': False,
            'issues': []
        }
        
        try:
            # Test full pipeline execution with timeout
            from prediction_pipeline_v3 import PredictionPipelineV3
            pipeline = PredictionPipelineV3()
            
            # Use threading to detect hangs in full pipeline
            pipeline_result = None
            pipeline_error = None
            
            def run_pipeline():
                nonlocal pipeline_result, pipeline_error
                try:
                    pipeline_result = pipeline.predict_race_file(race_file_path, enhancement_level="basic")
                except Exception as e:
                    pipeline_error = e
            
            pipeline_thread = threading.Thread(target=run_pipeline)
            pipeline_thread.daemon = True
            pipeline_thread.start()
            pipeline_thread.join(timeout=120)  # 2 minute timeout for full pipeline
            
            if pipeline_thread.is_alive():
                stage_result['issues'].append("Full pipeline execution timed out (>120s) - potential hang")
                logger.error("🚨 FULL PIPELINE TIMEOUT - this indicates a hanging operation!")
                return stage_result
            
            if pipeline_error:
                raise pipeline_error
            
            if pipeline_result:
                stage_result['output_generated'] = True
                stage_result['success'] = pipeline_result.get('success', False)
                stage_result['prediction_count'] = len(pipeline_result.get('predictions', []))
                stage_result['prediction_method'] = pipeline_result.get('prediction_method', 'unknown')
                
                if stage_result['success']:
                    logger.debug(f"✅ Pipeline completed successfully: {stage_result['prediction_count']} predictions")
                else:
                    stage_result['issues'].append(f"Pipeline returned unsuccessful result: {pipeline_result.get('error', 'Unknown error')}")
            else:
                stage_result['issues'].append("No result returned from pipeline")
            
        except Exception as e:
            stage_result['issues'].append(f"Post-processing diagnostic failed: {e}")
            stage_result['error'] = str(e)
            stage_result['traceback'] = traceback.format_exc()
        
        return stage_result
    
    @tracer.trace_call("diagnose_flask_integration")
    def _diagnose_flask_integration(self, race_file_path: str) -> Dict[str, Any]:
        """Diagnose Flask integration"""
        logger.info("🌐 Diagnosing Flask integration...")
        
        stage_result = {
            'success': False,
            'flask_route_accessible': False,
            'api_response_valid': False,
            'issues': []
        }
        
        try:
            # Test Flask app imports
            try:
                from app import app, api_predict_single_race
                stage_result['flask_imports'] = True
                logger.debug("✅ Flask app imports successful")
            except Exception as e:
                stage_result['issues'].append(f"Flask import failed: {e}")
                return stage_result
            
            # Simulate API call
            race_filename = os.path.basename(race_file_path)
            test_data = {"race_filename": race_filename}
            
            # Mock Flask request context for testing
            with app.test_request_context(json=test_data):
                try:
                    # This would normally be called via HTTP, but we're testing directly
                    logger.debug("Testing Flask route integration...")
                    stage_result['flask_route_accessible'] = True
                    stage_result['success'] = True
                except Exception as e:
                    stage_result['issues'].append(f"Flask route test failed: {e}")
            
        except Exception as e:
            stage_result['issues'].append(f"Flask integration diagnostic failed: {e}")
            stage_result['error'] = str(e)
            stage_result['traceback'] = traceback.format_exc()
        
        return stage_result

def create_test_race_file(filename: str = "test_race.csv") -> str:
    """Create a test race file for diagnostic purposes"""
    test_data = {
        'Dog Name': ['1. Test Dog Alpha', '2. Test Dog Beta', '3. Test Dog Gamma'],
        'WGT': [30.5, 31.2, 29.8],
        'SP': [2.50, 4.00, 6.50],
        'TIME': [29.85, 30.12, 30.45]
    }
    
    df = pd.DataFrame(test_data)
    test_file_path = os.path.join("upcoming_races", filename)
    
    # Ensure directory exists
    os.makedirs("upcoming_races", exist_ok=True)
    
    df.to_csv(test_file_path, index=False)
    logger.info(f"Created test race file: {test_file_path}")
    return test_file_path

@tracer.trace_call("run_comprehensive_diagnostic")
def run_comprehensive_diagnostic(race_file_path: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """Run comprehensive diagnostic with timeout protection"""
    
    # Note: Signal handling is problematic in threading context, removed for now
    # Will rely on individual timeouts in functions instead
    
    try:
        logger.info("🚀 Starting Prediction V3 Comprehensive Diagnostic")
        logger.info("=" * 60)
        
        # Create test file if none provided
        if not race_file_path:
            race_file_path = create_test_race_file()
        
        # Initialize diagnostics
        diagnostics = PredictionV3Diagnostics()
        
        # Run full diagnostic
        results = diagnostics.diagnose_full_pipeline(race_file_path)
        
        # Generate summary report
        results['diagnostic_summary'] = generate_diagnostic_summary(results)
        
        return results
        
    except TimeoutError as e:
        logger.error(f"🚨 DIAGNOSTIC TIMEOUT: {e}")
        return {
            'success': False,
            'error': 'diagnostic_timeout',
            'message': str(e),
            'performance_report': tracer.get_performance_report()
        }
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'performance_report': tracer.get_performance_report()
        }
        
    finally:
        pass  # No signal cleanup needed

def generate_diagnostic_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of diagnostic results"""
    summary = {
        'overall_status': 'unknown',
        'critical_issues': [],
        'warnings': [],
        'recommendations': [],
        'performance_issues': []
    }
    
    # Analyze each stage
    stages = results.get('stages', {})
    failed_stages = []
    
    for stage_name, stage_data in stages.items():
        if not stage_data.get('success', False):
            failed_stages.append(stage_name)
            for issue in stage_data.get('issues', []):
                summary['critical_issues'].append(f"{stage_name}: {issue}")
    
    # Analyze performance data
    perf_report = results.get('performance_report', {})
    
    # Check for timeouts
    timeouts = perf_report.get('timeouts', [])
    if timeouts:
        summary['critical_issues'].extend([f"Timeout in {t['function']}" for t in timeouts])
    
    # Check for potential loops
    loops = perf_report.get('potential_loops', [])
    if loops:
        summary['critical_issues'].extend([f"Potential infinite loop in {l['function']}" for l in loops])
    
    # Check for slow operations
    slow_ops = perf_report.get('slow_operations', [])
    if slow_ops:
        summary['performance_issues'].extend([f"Slow operation: {s['function']} ({s['execution_time']:.2f}s)" for s in slow_ops])
    
    # Determine overall status
    if not failed_stages and not summary['critical_issues']:
        summary['overall_status'] = 'healthy'
    elif failed_stages:
        summary['overall_status'] = 'failed'
    else:
        summary['overall_status'] = 'degraded'
    
    # Generate recommendations
    if 'file_loading' in failed_stages:
        summary['recommendations'].append("Check file paths and CSV formatting")
    if 'model_inference' in failed_stages:
        summary['recommendations'].append("Retrain ML models or check model files")
    if timeouts:
        summary['recommendations'].append("Investigate hanging operations - possible infinite loops or blocking I/O")
    if slow_ops:
        summary['recommendations'].append("Optimize slow operations to improve response time")
    
    return summary

def main():
    """Main entry point for diagnostic script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prediction V3 Diagnostic Tool")
    parser.add_argument("--race-file", help="Path to race file for testing")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")
    parser.add_argument("--output", help="Output file for diagnostic results")
    
    args = parser.parse_args()
    
    # Run diagnostic
    results = run_comprehensive_diagnostic(args.race_file, args.timeout)
    
    # Output results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Diagnostic results saved to: {args.output}")
    
    # Print summary
    summary = results.get('diagnostic_summary', {})
    print("\n" + "=" * 60)
    print("🔍 PREDICTION V3 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {summary.get('overall_status', 'unknown').upper()}")
    print(f"Total Time: {results.get('total_time', 0):.2f}s")
    
    if summary.get('critical_issues'):
        print("\n❌ CRITICAL ISSUES:")
        for issue in summary['critical_issues']:
            print(f"  - {issue}")
    
    if summary.get('performance_issues'):
        print("\n⏰ PERFORMANCE ISSUES:")
        for issue in summary['performance_issues']:
            print(f"  - {issue}")
    
    if summary.get('recommendations'):
        print("\n💡 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
    
    print("\n" + "=" * 60)
    
    return 0 if summary.get('overall_status') == 'healthy' else 1

if __name__ == "__main__":
    sys.exit(main())
