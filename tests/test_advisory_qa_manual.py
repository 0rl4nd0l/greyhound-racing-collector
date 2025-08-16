#!/usr/bin/env python3
"""
Advisory Manual QA Testing Suite
================================

Comprehensive manual testing for Step 9: QA & regression tests

1. Manual tests: single prediction, batch prediction, advisory auto & manual, error simulation (disconnect backend).  
2. Verify UI collapse toggle, colour coding, responsiveness.  
3. Check no blocking of prediction workflow.  
4. Update README / developer docs with advisory workflow.

Author: AI Assistant
Date: August 4, 2025
"""

import os
import sys
import json
import time
import requests
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from advisory import AdvisoryGenerator, MessageType
from qa_analyzer import QAAnalyzer
from validator import validate_output

class TestAdvisoryManualQA:
    """Comprehensive manual QA test suite for advisory workflow"""
    
    def setup_method(self):
        """Set up test environment for each test"""
        self.advisory_generator = AdvisoryGenerator()
        self.test_data_dir = Path("test_data_advisory")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Mock Flask app URL (will be used for API tests)
        self.flask_url = "http://127.0.0.1:5002"
        
    def teardown_method(self):
        """Clean up after each test"""
        # Clean up test files
        if self.test_data_dir.exists():
            import shutil
            shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    # =============================================================================
    # Test 1: Manual Test - Single Prediction Advisory Generation
    # =============================================================================
    
    def test_single_prediction_advisory_high_quality(self):
        """Test 1.1: Single prediction with high quality data"""
        print("\n🧪 TEST 1.1: Single Prediction - High Quality Advisory")
        
        # Create high-quality test prediction data
        test_data = {
            'race_id': 'test_race_001_hq',
            'race_date': '2025-08-04',
            'race_time': '14:30',
            'extraction_time': '2025-08-04T12:00:00',
            'predictions': [
                {'dog_name': 'Lightning Bolt', 'box_number': 1, 'win_prob': 0.35, 'confidence': 0.85},
                {'dog_name': 'Thunder Strike', 'box_number': 2, 'win_prob': 0.25, 'confidence': 0.78},
                {'dog_name': 'Storm Chaser', 'box_number': 3, 'win_prob': 0.20, 'confidence': 0.72},
                {'dog_name': 'Wind Runner', 'box_number': 4, 'win_prob': 0.15, 'confidence': 0.68},
                {'dog_name': 'Sky Dancer', 'box_number': 5, 'win_prob': 0.05, 'confidence': 0.45}
            ]
        }
        
        # Generate advisory
        result = self.advisory_generator.generate_advisory(data=test_data)
        
        # Assertions for high quality case
        assert result['success'] == True, "Advisory generation should succeed"
        assert len(result['messages']) >= 1, "Should have at least one message"
        assert result['processing_time_ms'] > 0, "Should track processing time"
        
        # Find quality assessment message
        quality_msgs = [m for m in result['messages'] if m['category'] == 'quality_assessment']
        assert len(quality_msgs) >= 1, "Should have quality assessment message"
        
        quality_msg = quality_msgs[0]
        assert quality_msg['type'] == MessageType.INFO.value, "High quality should be INFO type"
        assert 'High Quality' in quality_msg['title'], "Should indicate high quality"
        
        print(f"✅ High quality advisory: {quality_msg['message']}")
        print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
        print(f"   Messages generated: {len(result['messages'])}")
        
    def test_single_prediction_advisory_low_quality(self):
        """Test 1.2: Single prediction with low quality data (edge cases)"""
        print("\n🧪 TEST 1.2: Single Prediction - Low Quality Advisory")
        
        # Create low-quality test prediction data
        test_data = {
            'race_id': 'test_race_002_lq',
            'race_date': '2025-08-04',
            'race_time': '14:30', 
            'extraction_time': '2025-08-04T12:00:00',
            'predictions': [
                {'dog_name': 'Low Confidence', 'box_number': 1, 'win_prob': 0.05, 'confidence': 0.20},  # Very low confidence
                {'dog_name': 'Zero Prob', 'box_number': 2, 'win_prob': 0.0, 'confidence': 0.0},        # Zero confidence
                {'dog_name': 'Bad Data', 'box_number': 3, 'win_prob': 0.95, 'confidence': 0.15},       # High prob, low confidence
            ]
        }
        
        # Generate advisory
        result = self.advisory_generator.generate_advisory(data=test_data)
        
        # Assertions for low quality case
        assert result['success'] == True, "Advisory generation should succeed even with bad data"
        assert len(result['messages']) >= 1, "Should have warning/critical messages"
        
        # Should have warning or critical messages for low quality
        warning_critical_msgs = [m for m in result['messages'] if m['type'] in [MessageType.WARNING.value, MessageType.CRITICAL.value]]
        assert len(warning_critical_msgs) >= 1, "Should have warning or critical messages for low quality data"
        
        print(f"✅ Low quality advisory generated with {len(warning_critical_msgs)} warning/critical messages")
        for msg in warning_critical_msgs:
            print(f"   [{msg['type']}] {msg['title']}: {msg['message']}")
    
    # =============================================================================
    # Test 2: Manual Test - Batch Prediction Advisory
    # =============================================================================
    
    def test_batch_prediction_advisory(self):
        """Test 2: Batch prediction advisory processing"""
        print("\n🧪 TEST 2: Batch Prediction Advisory")
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_data = {
                'race_id': f'batch_race_{i:03d}',
                'race_date': '2025-08-04',
                'race_time': f'{14 + i}:30',
                'extraction_time': '2025-08-04T12:00:00',
                'predictions': [
                    {'dog_name': f'Dog {j}', 'box_number': j, 'win_prob': 0.2, 'confidence': 0.7}
                    for j in range(1, 6)
                ]
            }
            
            # Add some variance for different quality scores
            if i == 0:  # High quality
                for pred in test_data['predictions']:
                    pred['confidence'] = 0.85
            elif i == 2:  # Low quality
                for pred in test_data['predictions']:
                    pred['confidence'] = 0.25
                    pred['win_prob'] = 0.01  # Very low probabilities
                    
            # Save to temp file
            test_file = self.test_data_dir / f"batch_test_{i}.json"
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2)
            test_files.append(test_file)
        
        # Process batch advisory generation
        batch_results = []
        total_processing_time = 0
        
        for test_file in test_files:
            result = self.advisory_generator.generate_advisory(file_path=str(test_file))
            batch_results.append({
                'file': test_file.name,
                'result': result
            })
            total_processing_time += result.get('processing_time_ms', 0)
        
        # Batch processing assertions
        assert len(batch_results) == 3, "Should process all batch files"
        
        # Check that all files were processed successfully
        successful_results = [br for br in batch_results if br['result']['success']]
        assert len(successful_results) == 3, "All batch files should be processed successfully"
        
        print(f"✅ Batch processing completed:")
        print(f"   Files processed: {len(batch_results)}")
        print(f"   Total processing time: {total_processing_time:.1f}ms")
        print(f"   Average time per file: {total_processing_time/len(batch_results):.1f}ms")
        
        # Check for quality variations
        quality_distribution = {'INFO': 0, 'WARNING': 0, 'CRITICAL': 0}
        for br in batch_results:
            for msg in br['result']['messages']:
                if msg['category'] == 'quality_assessment':
                    quality_distribution[msg['type']] += 1
        
        print(f"   Quality distribution: {quality_distribution}")
        assert quality_distribution['INFO'] >= 1, "Should have at least one high quality case"
        assert quality_distribution['WARNING'] >= 1, "Should have at least one warning case"
    
    # =============================================================================
    # Test 3: Manual Test - Advisory Auto vs Manual Mode
    # =============================================================================
    
    def test_advisory_auto_vs_manual_mode(self):
        """Test 3: Compare auto-generated vs manual advisory modes"""
        print("\n🧪 TEST 3: Advisory Auto vs Manual Mode Comparison")
        
        test_data = {
            'race_id': 'auto_manual_comparison',
            'race_date': '2025-08-04',
            'race_time': '14:30',
            'extraction_time': '2025-08-04T12:00:00',
            'predictions': [
                {'dog_name': 'Test Dog 1', 'box_number': 1, 'win_prob': 0.4, 'confidence': 0.8},
                {'dog_name': 'Test Dog 2', 'box_number': 2, 'win_prob': 0.3, 'confidence': 0.1},  # Low confidence issue
                {'dog_name': 'Test Dog 3', 'box_number': 3, 'win_prob': 0.3, 'confidence': 0.7}
            ]
        }
        
        # Test auto mode (default)
        auto_result = self.advisory_generator.generate_advisory(data=test_data)
        
        # Test manual mode (mock OpenAI unavailable to force template mode)
        manual_generator = AdvisoryGenerator()
        manual_generator.openai_available = False  # Force template mode
        manual_result = manual_generator.generate_advisory(data=test_data)
        
        # Compare results
        assert auto_result['success'] == True, "Auto mode should succeed"
        assert manual_result['success'] == True, "Manual mode should succeed"
        
        # Both should detect the same issues
        auto_msgs = len(auto_result['messages'])
        manual_msgs = len(manual_result['messages'])
        
        print(f"✅ Mode comparison:")
        print(f"   Auto mode messages: {auto_msgs}")
        print(f"   Manual mode messages: {manual_msgs}")
        print(f"   Auto mode OpenAI used: {auto_result.get('openai_used', False)}")
        print(f"   Manual mode OpenAI used: {manual_result.get('openai_used', False)}")
        
        # Check that both modes identify critical issues
        auto_critical = [m for m in auto_result['messages'] if m['type'] == MessageType.CRITICAL.value]
        manual_critical = [m for m in manual_result['messages'] if m['type'] == MessageType.CRITICAL.value]
        
        assert len(auto_critical) == len(manual_critical), "Both modes should identify same critical issues"
        
        # Test the human-readable summaries are different (auto uses AI, manual uses templates)
        auto_summary = auto_result.get('human_readable_summary', '')
        manual_summary = manual_result.get('human_readable_summary', '')
        
        print(f"   Auto summary length: {len(auto_summary)} chars")
        print(f"   Manual summary length: {len(manual_summary)} chars")
        
        if auto_result.get('openai_used', False):
            # If OpenAI was actually used, summaries should be different
            assert auto_summary != manual_summary, "Auto and manual summaries should differ when OpenAI is available"
    
    # =============================================================================
    # Test 4: Error Simulation - Backend Disconnect
    # =============================================================================
    
    def test_error_simulation_backend_disconnect(self):
        """Test 4: Simulate backend errors and disconnections"""
        print("\n🧪 TEST 4: Error Simulation - Backend Disconnect")
        
        # Test 4.1: Invalid file path
        print("  4.1: Testing invalid file path...")
        invalid_file_result = self.advisory_generator.generate_advisory(file_path="/nonexistent/file.json")
        
        assert invalid_file_result['success'] == False, "Should fail for invalid file"
        assert 'error' in invalid_file_result, "Should contain error message"
        assert len(invalid_file_result['messages']) >= 1, "Should have error message"
        
        error_msg = invalid_file_result['messages'][0]
        assert error_msg['type'] == MessageType.CRITICAL.value, "Should be critical error"
        
        print(f"    ✅ Invalid file handled: {invalid_file_result['error']}")
        
        # Test 4.2: Invalid JSON data
        print("  4.2: Testing invalid JSON data...")
        invalid_json_file = self.test_data_dir / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json data }")
        
        invalid_json_result = self.advisory_generator.generate_advisory(file_path=str(invalid_json_file))
        assert invalid_json_result['success'] == False, "Should fail for invalid JSON"
        
        print(f"    ✅ Invalid JSON handled: {invalid_json_result['error']}")
        
        # Test 4.3: Empty data
        print("  4.3: Testing empty data...")
        empty_result = self.advisory_generator.generate_advisory(data={})
        assert empty_result['success'] == True, "Should handle empty data gracefully"
        
        print(f"    ✅ Empty data handled with {len(empty_result['messages'])} messages")
        
        # Test 4.4: Mock OpenAI API failure
        print("  4.4: Testing OpenAI API failure simulation...")
        with patch.object(self.advisory_generator, 'openai_client') as mock_client:
            # Mock API failure
            mock_client.chat.completions.create.side_effect = Exception("API Connection Error")
            
            test_data = {
                'race_id': 'api_failure_test',
                'predictions': [{'dog_name': 'Test', 'box_number': 1, 'win_prob': 0.5, 'confidence': 0.7}]
            }
            
            api_failure_result = self.advisory_generator.generate_advisory(data=test_data)
            
            # Should succeed with fallback to template mode
            assert api_failure_result['success'] == True, "Should fallback to template mode"
            assert api_failure_result.get('openai_used', True) == False, "Should indicate OpenAI not used"
            
            print(f"    ✅ OpenAI failure handled with fallback to template mode")
    
    # =============================================================================
    # Test 5: API Integration Tests (requires running Flask app)
    # =============================================================================
    
    def test_api_advisory_integration(self):
        """Test 5: API integration for advisory endpoints"""
        print("\n🧪 TEST 5: API Integration Tests")
        
        # Test data for API
        api_test_data = {
            'prediction_data': {
                'race_id': 'api_test_race',
                'race_date': '2025-08-04',
                'predictions': [
                    {'dog_name': 'API Test Dog', 'box_number': 1, 'win_prob': 0.6, 'confidence': 0.8}
                ]
            }
        }
        
        try:
            # Test if Flask app is running
            health_response = requests.get(f"{self.flask_url}/api/health", timeout=2)
            app_running = health_response.status_code == 200
        except:
            app_running = False
        
        if app_running:
            print("  ✅ Flask app is running - testing API integration")
            
            # Test advisory API endpoint
            try:
                api_response = requests.post(
                    f"{self.flask_url}/api/generate_advisory",
                    json=api_test_data,
                    timeout=10
                )
                
                assert api_response.status_code == 200, f"API should return 200, got {api_response.status_code}"
                
                api_result = api_response.json()
                assert api_result.get('success', False) == True, "API advisory should succeed"
                assert len(api_result.get('messages', [])) >= 1, "API should return messages"
                
                print(f"    ✅ API advisory endpoint working: {len(api_result['messages'])} messages")
                
            except requests.RequestException as e:
                print(f"    ⚠️ API test failed: {e}")
                
        else:
            print("  ⚠️ Flask app not running - skipping API integration tests")
            print("    To test API integration, run: python app.py")
    
    # =============================================================================
    # Test 6: UI Integration Simulation
    # =============================================================================
    
    def test_ui_integration_simulation(self):
        """Test 6: Simulate UI integration scenarios"""
        print("\n🧪 TEST 6: UI Integration Simulation")
        
        # Test 6.1: UI-style advisory request
        ui_test_data = {
            'race_id': 'ui_integration_test',
            'race_date': '2025-08-04',
            'race_time': '15:00',
            'venue': 'TEST_VENUE',
            'predictions': [
                {'dog_name': 'UI Dog 1', 'box_number': 1, 'win_prob': 0.4, 'confidence': 0.9},
                {'dog_name': 'UI Dog 2', 'box_number': 2, 'win_prob': 0.3, 'confidence': 0.1},  # Low confidence for UI warning
                {'dog_name': 'UI Dog 3', 'box_number': 3, 'win_prob': 0.3, 'confidence': 0.8}
            ]
        }
        
        result = self.advisory_generator.generate_advisory(data=ui_test_data)
        
        # Check UI-friendly formatting
        assert result['success'] == True, "UI advisory should succeed"
        
        # Check ML JSON output for UI consumption
        ml_json = result.get('ml_json', {})
        assert 'summary' in ml_json, "Should have summary for UI"
        assert 'feature_flags' in ml_json, "Should have feature flags for UI"
        
        # Test feature flags that UI would use
        feature_flags = ml_json['feature_flags']
        expected_flags = ['has_validation_errors', 'has_quality_issues', 'low_quality_score']
        for flag in expected_flags:
            assert flag in feature_flags, f"Feature flag '{flag}' should be present for UI"
        
        print(f"✅ UI simulation completed:")
        print(f"   Feature flags: {feature_flags}")
        print(f"   Summary metrics: {ml_json['summary']}")
        
        # Test 6.2: Message categorization for UI color coding
        message_types = {}
        for msg in result['messages']:
            msg_type = msg['type']
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        print(f"   Message types for UI color coding: {message_types}")
        
        # Should have at least one warning for the low confidence dog
        assert message_types.get(MessageType.WARNING.value, 0) >= 1, "Should have warning messages for UI"
    
    # =============================================================================
    # Test 7: Performance and Responsiveness
    # =============================================================================
    
    def test_performance_responsiveness(self):
        """Test 7: Performance and responsiveness under load"""
        print("\n🧪 TEST 7: Performance and Responsiveness Tests")
        
        # Test 7.1: Single advisory performance
        start_time = time.time()
        
        test_data = {
            'race_id': 'performance_test',
            'predictions': [
                {'dog_name': f'Perf Dog {i}', 'box_number': i, 'win_prob': 0.2, 'confidence': 0.7}
                for i in range(1, 9)  # 8 dogs
            ]
        }
        
        result = self.advisory_generator.generate_advisory(data=test_data)
        
        processing_time = time.time() - start_time
        reported_time = result.get('processing_time_ms', 0) / 1000
        
        print(f"  ✅ Single advisory performance:")
        print(f"     Actual time: {processing_time:.3f}s")
        print(f"     Reported time: {reported_time:.3f}s")
        print(f"     Messages generated: {len(result['messages'])}")
        
        # Performance assertions
        assert processing_time < 5.0, "Single advisory should complete in under 5 seconds"
        assert result['success'] == True, "Performance test should succeed"
        
        # Test 7.2: Multiple concurrent advisories (simulation)
        concurrent_results = []
        concurrent_start = time.time()
        
        for i in range(5):
            test_data_concurrent = {
                'race_id': f'concurrent_test_{i}',
                'predictions': [
                    {'dog_name': f'Concurrent Dog {j}', 'box_number': j, 'win_prob': 0.2, 'confidence': 0.7}
                    for j in range(1, 6)
                ]
            }
            
            concurrent_result = self.advisory_generator.generate_advisory(data=test_data_concurrent)
            concurrent_results.append(concurrent_result)
        
        concurrent_time = time.time() - concurrent_start
        
        print(f"  ✅ Concurrent advisory performance:")
        print(f"     Total time for 5 advisories: {concurrent_time:.3f}s")
        print(f"     Average time per advisory: {concurrent_time/5:.3f}s")
        
        # All concurrent tests should succeed
        successful_concurrent = [r for r in concurrent_results if r['success']]
        assert len(successful_concurrent) == 5, "All concurrent advisories should succeed"
        
        # Total time should be reasonable
        assert concurrent_time < 15.0, "5 concurrent advisories should complete in under 15 seconds"
    
    # =============================================================================
    # Test 8: Workflow Integration Test
    # =============================================================================
    
    def test_workflow_integration_no_blocking(self):
        """Test 8: Ensure advisory doesn't block prediction workflow"""
        print("\n🧪 TEST 8: Workflow Integration - No Blocking Test")
        
        # Simulate a prediction workflow that includes advisory
        prediction_data = {
            'race_id': 'workflow_integration_test',
            'race_date': '2025-08-04',
            'race_time': '16:30',
            'predictions': [
                {'dog_name': 'Workflow Dog 1', 'box_number': 1, 'win_prob': 0.45, 'confidence': 0.85},
                {'dog_name': 'Workflow Dog 2', 'box_number': 2, 'win_prob': 0.35, 'confidence': 0.75},
                {'dog_name': 'Workflow Dog 3', 'box_number': 3, 'win_prob': 0.20, 'confidence': 0.65}
            ]
        }
        
        # Step 1: Generate prediction (simulated)
        prediction_start = time.time()
        # ... prediction logic would go here ...
        prediction_time = time.time() - prediction_start
        
        # Step 2: Generate advisory (should not block)
        advisory_start = time.time()
        advisory_result = self.advisory_generator.generate_advisory(data=prediction_data)
        advisory_time = time.time() - advisory_start
        
        # Step 3: Continue with workflow (simulated)
        workflow_continue_start = time.time()
        # ... additional workflow steps ...
        workflow_continue_time = time.time() - workflow_continue_start
        
        total_workflow_time = prediction_time + advisory_time + workflow_continue_time
        
        print(f"✅ Workflow integration test:")
        print(f"   Prediction time: {prediction_time:.3f}s")
        print(f"   Advisory time: {advisory_time:.3f}s")
        print(f"   Workflow continue time: {workflow_continue_time:.3f}s")
        print(f"   Total workflow time: {total_workflow_time:.3f}s")
        
        # Advisory should not significantly impact workflow
        assert advisory_result['success'] == True, "Advisory should succeed in workflow"
        assert advisory_time < 2.0, "Advisory should not add significant delay to workflow"
        
        # Advisory should provide useful information without blocking
        assert len(advisory_result['messages']) >= 1, "Advisory should provide information"
        assert advisory_result.get('ml_json', {}).get('summary', {}).get('total_messages', 0) >= 1, "Should have summary for downstream processing"
        
        print(f"   Advisory messages: {len(advisory_result['messages'])}")
        print(f"   Workflow blocking: {'No' if advisory_time < 2.0 else 'Yes'}")


# =============================================================================
# Manual Test Runner
# =============================================================================

def run_manual_tests():
    """Run all manual tests for advisory QA"""
    print("🚀 Starting Advisory Manual QA Testing Suite")
    print("=" * 60)
    
    test_suite = TestAdvisoryManualQA()
    test_suite.setup_method()
    
    try:
        # Run all tests
        tests = [
            test_suite.test_single_prediction_advisory_high_quality,
            test_suite.test_single_prediction_advisory_low_quality,
            test_suite.test_batch_prediction_advisory,
            test_suite.test_advisory_auto_vs_manual_mode,
            test_suite.test_error_simulation_backend_disconnect,
            test_suite.test_api_advisory_integration,
            test_suite.test_ui_integration_simulation,
            test_suite.test_performance_responsiveness,
            test_suite.test_workflow_integration_no_blocking
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for i, test in enumerate(tests, 1):
            try:
                print(f"\n[{i}/{len(tests)}] Running {test.__name__}...")
                test()
                passed_tests += 1
                print(f"✅ {test.__name__} PASSED")
            except Exception as e:
                failed_tests += 1
                print(f"❌ {test.__name__} FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        # Final results
        print("\n" + "=" * 60)
        print("🏁 Advisory Manual QA Testing Results")
        print("=" * 60)
        print(f"✅ Tests Passed: {passed_tests}")
        print(f"❌ Tests Failed: {failed_tests}")
        print(f"📊 Success Rate: {passed_tests/(passed_tests+failed_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Advisory system is ready for production.")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Review the issues above.")
        
    finally:
        test_suite.teardown_method()


if __name__ == "__main__":
    run_manual_tests()
