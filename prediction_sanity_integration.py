#!/usr/bin/env python3
"""
Prediction Sanity Integration Module
====================================

This module provides easy integration of sanity checks and auto-fixes 
into existing prediction pipelines. It can be imported and used with 
minimal code changes.
"""

import logging
from typing import Dict, List, Optional, Tuple
from sanity_checks import SanityChecks

logger = logging.getLogger(__name__)

def apply_sanity_checks_to_response(response: Dict, auto_fix: bool = True) -> Dict:
    """
    Apply sanity checks to a prediction response and optionally fix issues.
    
    Args:
        response: Prediction response dictionary containing 'predictions' key
        auto_fix: Whether to automatically fix detected issues
        
    Returns:
        Updated response with sanity check results and fixes if applied
    """
    if 'predictions' not in response:
        logger.warning("No 'predictions' key found in response")
        return response
    
    checker = SanityChecks()
    predictions = response['predictions']
    
    # Validate predictions
    validation_results = checker.validate_predictions(predictions)
    
    # Add validation results to response
    response['sanity_check'] = {
        'validation_performed': True,
        'issues_found': len(validation_results['flags']),
        'passed_checks': validation_results['passed_checks'],
        'failed_checks': validation_results['failed_checks'],
        'flags': validation_results['flags']
    }
    
    # Apply fixes if issues found and auto_fix enabled
    if validation_results['flags'] and auto_fix:
        logger.info(f"Applying automatic fixes for {len(validation_results['flags'])} issues")
        fixed_predictions = checker.fix_predictions(predictions)
        
        # Re-validate fixed predictions
        fixed_validation = checker.validate_predictions(fixed_predictions)
        
        # Update response with fixed predictions
        response['predictions'] = fixed_predictions
        response['sanity_check'].update({
            'fixes_applied': True,
            'original_predictions_backup': predictions,
            'issues_remaining': len(fixed_validation['flags']),
            'fix_success': len(fixed_validation['flags']) == 0
        })
        
        logger.info(f"✅ Fixes applied. Issues remaining: {len(fixed_validation['flags'])}")
    else:
        response['sanity_check']['fixes_applied'] = False
    
    return response

def validate_and_fix_predictions(predictions: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Standalone function to validate and fix predictions.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Tuple of (fixed_predictions, validation_info)
    """
    checker = SanityChecks()
    
    # Validate original predictions
    validation_results = checker.validate_predictions(predictions)
    
    validation_info = {
        'original_issues': len(validation_results['flags']),
        'flags': validation_results['flags'],
        'fixes_applied': False
    }
    
    # Fix if issues found
    if validation_results['flags']:
        fixed_predictions = checker.fix_predictions(predictions)
        
        # Re-validate
        fixed_validation = checker.validate_predictions(fixed_predictions)
        
        validation_info.update({
            'fixes_applied': True,
            'issues_remaining': len(fixed_validation['flags']),
            'fix_success': len(fixed_validation['flags']) == 0
        })
        
        return fixed_predictions, validation_info
    
    return predictions, validation_info

# Decorator for automatic sanity checking
def with_sanity_checks(auto_fix: bool = True):
    """
    Decorator to automatically apply sanity checks to prediction functions.
    
    Args:
        auto_fix: Whether to automatically fix detected issues
        
    Returns:
        Decorated function with sanity checks applied
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Call original function
            result = func(*args, **kwargs)
            
            # Apply sanity checks if result contains predictions
            if isinstance(result, dict) and 'predictions' in result:
                result = apply_sanity_checks_to_response(result, auto_fix=auto_fix)
            
            return result
        return wrapper
    return decorator

class SanityCheckMiddleware:
    """
    Middleware class for pipeline integration.
    """
    
    def __init__(self, auto_fix: bool = True, log_issues: bool = True):
        self.auto_fix = auto_fix
        self.log_issues = log_issues
        self.checker = SanityChecks()
        self.stats = {
            'predictions_processed': 0,
            'issues_found': 0,
            'fixes_applied': 0
        }
    
    def process(self, predictions: List[Dict]) -> List[Dict]:
        """
        Process predictions through sanity checks.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Processed (and potentially fixed) predictions
        """
        self.stats['predictions_processed'] += len(predictions)
        
        # Validate
        validation_results = self.checker.validate_predictions(predictions)
        
        if validation_results['flags']:
            self.stats['issues_found'] += len(validation_results['flags'])
            
            if self.log_issues:
                logger.warning(f"Found {len(validation_results['flags'])} sanity check issues")
                for flag in validation_results['flags']:
                    logger.warning(f"  - {flag}")
            
            # Fix if enabled
            if self.auto_fix:
                fixed_predictions = self.checker.fix_predictions(predictions)
                self.stats['fixes_applied'] += 1
                
                if self.log_issues:
                    logger.info("Applied automatic fixes")
                
                return fixed_predictions
        
        return predictions
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()

# Example integration patterns
def integrate_with_ml_system_v3():
    """
    Example of how to integrate with MLSystemV3.
    """
    example_code = '''
    # In ml_system_v3.py, modify the predict method:
    
    from prediction_sanity_integration import apply_sanity_checks_to_response
    
    def predict(self, dog_features):
        # ... existing prediction logic ...
        
        result = {
            "win_probability": float(calibrated_probs['calibrated_win_prob']),
            "place_probability": float(calibrated_probs['calibrated_place_prob']),
            # ... other fields ...
        }
        
        # Apply sanity checks before returning
        if hasattr(self, 'apply_sanity_checks') and self.apply_sanity_checks:
            result = apply_sanity_checks_to_response({'predictions': [result]})
            if result.get('sanity_check', {}).get('fixes_applied'):
                logger.info("Sanity check fixes applied to prediction")
            return result['predictions'][0]
        
        return result
    '''
    return example_code

def integrate_with_prediction_pipeline_v3():
    """
    Example of how to integrate with PredictionPipelineV3.
    """
    example_code = '''
    # In prediction_pipeline_v3.py, modify the predict_race_file method:
    
    from prediction_sanity_integration import with_sanity_checks
    
    @with_sanity_checks(auto_fix=True)
    def predict_race_file(self, race_file_path: str, enhancement_level="full") -> dict:
        # ... existing prediction logic ...
        
        return {
            "success": True,
            "predictions": predictions,
            "prediction_method": method_used,
            # ... other fields ...
        }
        
    # The decorator will automatically apply sanity checks to the returned predictions
    '''
    return example_code

if __name__ == "__main__":
    # Example usage
    print("🔧 Prediction Sanity Integration Examples")
    print("=" * 50)
    
    # Example 1: Direct function usage
    test_predictions = [
        {'dog_name': 'Test Dog 1', 'win_probability': 1.2, 'predicted_rank': 1},
        {'dog_name': 'Test Dog 2', 'win_probability': 0.3, 'predicted_rank': 2}
    ]
    
    fixed_predictions, info = validate_and_fix_predictions(test_predictions)
    print(f"Direct usage - Issues found: {info['original_issues']}, Fixed: {info['fixes_applied']}")
    
    # Example 2: Response processing
    test_response = {
        'success': True,
        'predictions': test_predictions,
        'method': 'test'
    }
    
    processed_response = apply_sanity_checks_to_response(test_response)
    print(f"Response processing - Issues: {processed_response['sanity_check']['issues_found']}")
    
    # Example 3: Middleware usage
    middleware = SanityCheckMiddleware(auto_fix=True)
    processed_predictions = middleware.process(test_predictions)
    print(f"Middleware stats: {middleware.get_stats()}")
    
    print("\n📖 See function docstrings for integration examples")
