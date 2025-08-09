#!/usr/bin/env python3
"""
Advisory Message Generator
=========================

Step 8: Advisory Message Generator

`advisory.py` combines issues from validator & analyzer, formats human-readable messages + JSON for downstream ML.  
Message types: WARNING, CRITICAL, INFO.  
Support OpenAI summarisation when API key verified; fallback to template strings.

Author: AI Assistant
Date: August 4, 2025
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
from enum import Enum
from validator import validate_output
from qa_analyzer import QAAnalyzer
from openai_connectivity_verifier import OpenAIConnectivityVerifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Advisory message types"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class AdvisoryGenerator:
    """
    Generates advisory messages based on validation and analysis results.
    Supports OpenAI summarization when API key is verified; fallback to template strings.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        # Initialize the QA Analyzer
        self.qa_analyzer = QAAnalyzer()
        # Verify OpenAI API connectivity
        self.openai_verifier = OpenAIConnectivityVerifier()
        self.openai_client = None
        self.openai_available = False
        
        # Try to initialize OpenAI
        if self.openai_verifier.load_api_key():
            if self.openai_verifier.initialize_client():
                self.openai_client = self.openai_verifier.client
                self.openai_available = not self.openai_verifier.is_mock
                
        logger.info(f"Advisory Generator initialized - OpenAI available: {self.openai_available}")

    def generate_advisory(self, file_path: Optional[str] = None, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Combines validation and analysis results into an advisory report.
        
        Args:
            file_path: Path to JSON file to validate and analyze
            data: Dictionary data to analyze (alternative to file_path)
            
        Returns:
            Dictionary containing advisory report with messages and JSON output
        """
        start_time = datetime.now()
        
        try:
            # Validate the output
            validation_result = None
            if file_path:
                validation_result = validate_output(file_path)
                # Load data from file for analysis
                with open(file_path, 'r') as file:
                    data = json.load(file)
            elif data is None:
                return self._create_error_response("No file path or data provided", start_time)
            
            # If we have data, analyze it
            analysis_results = None
            if data:
                analysis_results = self.qa_analyzer.comprehensive_qa_analysis(data)
            
            # Generate advisory messages
            messages = self._generate_messages(validation_result, analysis_results)
            
            # Create human-readable summary
            human_readable = self._create_human_readable_summary(messages, validation_result, analysis_results)
            
            # Generate JSON for downstream ML
            ml_json = self._create_ml_json(messages, validation_result, analysis_results)
            
            advisory_report = {
                'timestamp': start_time.isoformat(),
                'success': True,
                'file_path': file_path,
                'validation_result': validation_result,
                'analysis_results': analysis_results,
                'messages': messages,
                'human_readable_summary': human_readable,
                'ml_json': ml_json,
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'openai_used': self.openai_available
            }
            
            logger.info(f"Advisory generated successfully - {len(messages)} messages, "
                       f"processing time: {advisory_report['processing_time_ms']:.1f}ms")
            
            return advisory_report
            
        except Exception as e:
            logger.error(f"Error generating advisory: {e}")
            return self._create_error_response(str(e), start_time)

    def _generate_messages(self, validation_result: Optional[Dict], analysis_results: Optional[Dict]) -> List[Dict[str, Any]]:
        """Generate structured advisory messages based on validation and analysis results"""
        messages = []
        
        # Validation messages
        if validation_result:
            if not validation_result.get('valid', False):
                for error in validation_result.get('errors', []):
                    messages.append({
                        'type': MessageType.CRITICAL.value,
                        'category': 'validation',
                        'title': 'Validation Error',
                        'message': error,
                        'timestamp': datetime.now().isoformat()
                    })
            
            for warning in validation_result.get('warnings', []):
                messages.append({
                    'type': MessageType.WARNING.value,
                    'category': 'validation',
                    'title': 'Validation Warning',
                    'message': warning,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Analysis messages
        if analysis_results:
            quality_score = analysis_results.get('overall_quality_score', 100)
            issue_categories = analysis_results.get('issue_categories', [])
            
            # Overall quality assessment
            if quality_score >= 90:
                messages.append({
                    'type': MessageType.INFO.value,
                    'category': 'quality_assessment',
                    'title': 'High Quality Predictions',
                    'message': f"Prediction quality score: {quality_score}/100 - Excellent",
                    'timestamp': datetime.now().isoformat()
                })
            elif quality_score >= 70:
                messages.append({
                    'type': MessageType.WARNING.value,
                    'category': 'quality_assessment',
                    'title': 'Moderate Quality Predictions',
                    'message': f"Prediction quality score: {quality_score}/100 - Some issues detected",
                    'timestamp': datetime.now().isoformat()
                })
            else:
                messages.append({
                    'type': MessageType.CRITICAL.value,
                    'category': 'quality_assessment',
                    'title': 'Low Quality Predictions',
                    'message': f"Prediction quality score: {quality_score}/100 - Significant issues detected",
                    'timestamp': datetime.now().isoformat()
                })
            
            # Specific issue messages
            individual_analyses = analysis_results.get('individual_analyses', {})
            
            # Low confidence/variance issues
            if 'confidence_variance' in individual_analyses:
                cv_analysis = individual_analyses['confidence_variance']
                if cv_analysis.get('issues_detected', False):
                    low_conf_count = cv_analysis.get('low_confidence_count', 0)
                    messages.append({
                        'type': MessageType.WARNING.value,
                        'category': 'confidence_analysis',
                        'title': 'Low Confidence Predictions',
                        'message': f"{low_conf_count} predictions below confidence threshold",
                        'timestamp': datetime.now().isoformat(),
                        'details': cv_analysis.get('flagged_predictions', [])
                    })
            
            # Class imbalance issues
            if 'class_imbalance' in individual_analyses:
                ci_analysis = individual_analyses['class_imbalance']
                if ci_analysis.get('issues_detected', False):
                    entropy = ci_analysis.get('normalized_entropy', 0)
                    messages.append({
                        'type': MessageType.WARNING.value,
                        'category': 'class_imbalance',
                        'title': 'Class Imbalance Detected',
                        'message': f"Low probability distribution entropy: {entropy:.3f}",
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Calibration drift issues
            if 'calibration_drift' in individual_analyses:
                cd_analysis = individual_analyses['calibration_drift']
                if cd_analysis.get('issues_detected', False):
                    messages.append({
                        'type': MessageType.CRITICAL.value,
                        'category': 'calibration',
                        'title': 'Calibration Drift Detected',
                        'message': "Model calibration may have drifted - consider retraining",
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Leakage and date drift issues
            if 'leakage_date_drift' in individual_analyses:
                ld_analysis = individual_analyses['leakage_date_drift']
                if ld_analysis.get('issues_detected', False):
                    errors = ld_analysis.get('errors', [])
                    for error in errors:
                        messages.append({
                            'type': MessageType.CRITICAL.value,
                            'category': 'data_leakage',
                            'title': 'Data Leakage or Temporal Issue',
                            'message': error,
                            'timestamp': datetime.now().isoformat()
                        })
        
        return messages

    def _create_human_readable_summary(self, messages: List[Dict], validation_result: Optional[Dict], analysis_results: Optional[Dict]) -> str:
        """Create human-readable summary of advisory messages"""
        
        if self.openai_available:
            return self._get_openai_summary(messages, validation_result, analysis_results)
        else:
            return self._get_template_summary(messages, validation_result, analysis_results)

    def _get_openai_summary(self, messages: List[Dict], validation_result: Optional[Dict], analysis_results: Optional[Dict]) -> str:
        """Uses OpenAI to generate human-readable summary"""
        try:
            # Create context for OpenAI
            context = {
                'total_messages': len(messages),
                'message_types': {},
                'categories': {},
                'validation_status': validation_result.get('valid', True) if validation_result else True,
                'quality_score': analysis_results.get('overall_quality_score', 100) if analysis_results else 100
            }
            
            # Count message types and categories
            for msg in messages:
                msg_type = msg['type']
                category = msg['category']
                context['message_types'][msg_type] = context['message_types'].get(msg_type, 0) + 1
                context['categories'][category] = context['categories'].get(category, 0) + 1
            
            prompt = f"""
Generate a professional, concise summary of prediction quality analysis results.

Context:
- Total issues found: {len(messages)}
- Message types: {context['message_types']}
- Categories affected: {list(context['categories'].keys())}
- Validation passed: {context['validation_status']}
- Quality score: {context['quality_score']}/100

Key Issues:
{json.dumps([{'type': m['type'], 'title': m['title'], 'message': m['message']} for m in messages[:5]], indent=2)}

Provide a 2-3 sentence executive summary focusing on the most critical issues and overall assessment.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a data quality analyst providing executive summaries of prediction quality assessments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"OpenAI summary failed, falling back to template: {e}")
            return self._get_template_summary(messages, validation_result, analysis_results)

    def _get_template_summary(self, messages: List[Dict], validation_result: Optional[Dict], analysis_results: Optional[Dict]) -> str:
        """Generate template-based summary as fallback"""
        
        # Count message types
        critical_count = len([m for m in messages if m['type'] == MessageType.CRITICAL.value])
        warning_count = len([m for m in messages if m['type'] == MessageType.WARNING.value])
        info_count = len([m for m in messages if m['type'] == MessageType.INFO.value])
        
        # Get quality score
        quality_score = 100
        if analysis_results:
            quality_score = analysis_results.get('overall_quality_score', 100)
        
        # Create summary
        if critical_count > 0:
            summary = f"CRITICAL: {critical_count} critical issues detected. "
        elif warning_count > 0:
            summary = f"WARNING: {warning_count} warnings identified. "
        else:
            summary = "INFO: All quality checks passed. "
        
        summary += f"Overall quality score: {quality_score}/100. "
        
        if critical_count > 0:
            summary += "Immediate attention required for critical issues."
        elif warning_count > 0:
            summary += "Review recommended for identified warnings."
        else:
            summary += "Predictions meet quality standards."
        
        return summary

    def _create_ml_json(self, messages: List[Dict], validation_result: Optional[Dict], analysis_results: Optional[Dict]) -> Dict[str, Any]:
        """Create structured JSON output for downstream ML processing"""
        
        return {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_messages': len(messages),
                'critical_count': len([m for m in messages if m['type'] == MessageType.CRITICAL.value]),
                'warning_count': len([m for m in messages if m['type'] == MessageType.WARNING.value]),
                'info_count': len([m for m in messages if m['type'] == MessageType.INFO.value]),
                'validation_passed': validation_result.get('valid', True) if validation_result else True,
                'quality_score': analysis_results.get('overall_quality_score', 100) if analysis_results else 100
            },
            'messages': messages,
            'raw_validation': validation_result,
            'raw_analysis': analysis_results,
            'feature_flags': {
                'has_validation_errors': validation_result and not validation_result.get('valid', True),
                'has_quality_issues': analysis_results and analysis_results.get('total_issues_detected', 0) > 0,
                'low_quality_score': (analysis_results.get('overall_quality_score', 100) if analysis_results else 100) < 70
            }
        }

    def _create_error_response(self, error_message: str, start_time: datetime) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'timestamp': start_time.isoformat(),
            'success': False,
            'error': error_message,
            'messages': [{
                'type': MessageType.CRITICAL.value,
                'category': 'system_error',
                'title': 'Advisory Generation Error',
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            }],
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advisory Message Generator')
    parser.add_argument('--file', help='Path to JSON file to analyze')
    parser.add_argument('--test', action='store_true', help='Run with test data')
    
    args = parser.parse_args()
    
    advisory_generator = AdvisoryGenerator()
    
    if args.test:
        # Generate test data
        test_data = {
            'race_id': 'test_race_advisory',
            'race_date': '2025-08-04',
            'race_time': '14:30',
            'extraction_time': '2025-08-04T12:00:00',
            'predictions': [
                {'dog_name': 'Test Dog 1', 'box_number': 1, 'win_prob': 0.15},  # Low confidence
                {'dog_name': 'Test Dog 2', 'box_number': 2, 'win_prob': 0.85},  # High confidence
                {'dog_name': 'Test Dog 3', 'box_number': 3, 'win_prob': 0.0}   # Zero confidence
            ]
        }
        
        print("🧪 Testing Advisory Generator with test data...")
        result = advisory_generator.generate_advisory(data=test_data)
        
    elif args.file:
        print(f"📊 Analyzing file: {args.file}")
        result = advisory_generator.generate_advisory(file_path=args.file)
        
    else:
        print("Please provide --file path or use --test for test data")
        exit(1)
    
    # Display results
    print(f"\n✅ Advisory Generation Complete!")
    print(f"Success: {result['success']}")
    print(f"Messages: {len(result.get('messages', []))}")
    print(f"Processing time: {result.get('processing_time_ms', 0):.1f}ms")
    print(f"OpenAI used: {result.get('openai_used', False)}")
    
    if result.get('human_readable_summary'):
        print(f"\n📋 Summary:")
        print(result['human_readable_summary'])
    
    # Show message breakdown
    if result.get('messages'):
        print(f"\n📝 Messages:")
        for msg in result['messages']:
            print(f"  [{msg['type']}] {msg['title']}: {msg['message']}")
    
    print(f"\n💾 Full result saved to advisory_result.json")
    with open('advisory_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

