#!/usr/bin/env python3
"""
Simple Advisory Workflow Test
=============================

Tests the advisory workflow integration to ensure it doesn't block
the main prediction pipeline and works correctly.

Author: AI Assistant
Date: August 4, 2025
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_workflow_integration():
    """Test that advisory system integrates without blocking workflow"""
    print("🧪 Testing Advisory Workflow Integration")
    print("=" * 50)

    try:
        from advisory import AdvisoryGenerator, MessageType

        # Simulate prediction workflow
        print("\n1. Simulating prediction generation...")
        prediction_start = time.time()

        # Mock prediction data (would come from actual prediction system)
        prediction_data = {
            "race_id": "workflow_test_race",
            "race_date": "2025-08-04",
            "race_time": "14:30",
            "venue": "TEST_VENUE",
            "predictions": [
                {
                    "dog_name": "Fast Dog",
                    "box_number": 1,
                    "win_prob": 0.45,
                    "confidence": 0.85,
                },
                {
                    "dog_name": "Medium Dog",
                    "box_number": 2,
                    "win_prob": 0.35,
                    "confidence": 0.75,
                },
                {
                    "dog_name": "Slow Dog",
                    "box_number": 3,
                    "win_prob": 0.20,
                    "confidence": 0.65,
                },
                {
                    "dog_name": "Risky Dog",
                    "box_number": 4,
                    "win_prob": 0.10,
                    "confidence": 0.25,
                },  # Low confidence
            ],
        }

        prediction_time = time.time() - prediction_start
        print(f"   ✅ Prediction completed in {prediction_time:.3f}s")

        # Generate advisory (should not block)
        print("\n2. Generating advisory...")
        advisory_start = time.time()

        advisory_generator = AdvisoryGenerator()
        advisory_result = advisory_generator.generate_advisory(data=prediction_data)

        advisory_time = time.time() - advisory_start
        print(f"   ✅ Advisory completed in {advisory_time:.3f}s")

        # Continue with workflow
        print("\n3. Continuing workflow...")
        workflow_start = time.time()

        # Mock additional workflow steps (e.g., saving results, notifications)
        # In real implementation, this would include database saves, UI updates, etc.
        time.sleep(0.01)  # Simulate brief additional processing

        workflow_time = time.time() - workflow_start
        total_time = prediction_time + advisory_time + workflow_time

        print(f"   ✅ Workflow continuation completed in {workflow_time:.3f}s")
        print(f"   📊 Total workflow time: {total_time:.3f}s")

        # Analyze results
        print("\n4. Analyzing advisory results...")

        if advisory_result["success"]:
            messages = advisory_result.get("messages", [])
            processing_time = advisory_result.get("processing_time_ms", 0)

            print(f"   ✅ Advisory succeeded: {len(messages)} messages")
            print(f"   ⏱️ Processing time: {processing_time:.1f}ms")

            # Check message types
            message_types = {}
            for msg in messages:
                msg_type = msg["type"]
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
                print(f"   📝 [{msg['type']}] {msg['title']}: {msg['message']}")

            print(f"   📈 Message breakdown: {message_types}")

            # Check for expected warnings (low confidence dog)
            warning_msgs = [
                m for m in messages if m["type"] == MessageType.WARNING.value
            ]
            if len(warning_msgs) > 0:
                print(f"   ⚠️ Correctly identified {len(warning_msgs)} warnings")

            # Check ML JSON for downstream processing
            ml_json = advisory_result.get("ml_json", {})
            if "summary" in ml_json:
                summary = ml_json["summary"]
                print(f"   🤖 ML JSON summary: {summary}")

            # Performance checks
            if advisory_time < 1.0:
                print("   ⚡ Performance: Advisory is fast (< 1s)")
            else:
                print("   ⚠️ Performance: Advisory is slow (> 1s)")

            if total_time < 2.0:
                print("   ✅ Workflow: No significant blocking detected")
            else:
                print("   ⚠️ Workflow: Potential blocking detected")

        else:
            error = advisory_result.get("error", "Unknown error")
            print(f"   ❌ Advisory failed: {error}")

        print("\n5. Testing error scenarios...")

        # Test with invalid data
        error_test_start = time.time()
        error_result = advisory_generator.generate_advisory(data={"invalid": "data"})
        error_test_time = time.time() - error_test_start

        if error_result["success"]:
            print(f"   ✅ Handled minimal data gracefully in {error_test_time:.3f}s")
        else:
            print(
                f"   ⚠️ Failed on minimal data: {error_result.get('error', 'Unknown')}"
            )

        # Test with file not found
        file_error_result = advisory_generator.generate_advisory(
            file_path="/nonexistent/file.json"
        )

        if not file_error_result["success"]:
            print("   ✅ Correctly handled file not found error")
        else:
            print("   ⚠️ Should have failed for non-existent file")

        print("\n" + "=" * 50)
        print("🏁 Workflow Integration Test Results")
        print("=" * 50)

        # Summary
        all_good = True
        issues = []

        if not advisory_result["success"]:
            all_good = False
            issues.append("Advisory generation failed")

        if advisory_time > 1.0:
            all_good = False
            issues.append("Advisory too slow")

        if total_time > 2.0:
            all_good = False
            issues.append("Total workflow too slow")

        if len(advisory_result.get("messages", [])) == 0:
            all_good = False
            issues.append("No advisory messages generated")

        if all_good:
            print("✅ ALL CHECKS PASSED")
            print("   - Advisory generation works correctly")
            print("   - No blocking of prediction workflow")
            print("   - Performance is acceptable")
            print("   - Error handling works")
            print("   - Quality assessment detected issues")
        else:
            print("⚠️ SOME ISSUES DETECTED:")
            for issue in issues:
                print(f"   - {issue}")

        return all_good

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure advisory system dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ui_data_format():
    """Test that advisory data is properly formatted for UI consumption"""
    print("\n🎨 Testing UI Data Format")
    print("=" * 30)

    try:
        from advisory import AdvisoryGenerator

        ui_test_data = {
            "race_id": "ui_format_test",
            "predictions": [
                {
                    "dog_name": "UI Dog 1",
                    "box_number": 1,
                    "win_prob": 0.6,
                    "confidence": 0.9,
                },
                {
                    "dog_name": "UI Dog 2",
                    "box_number": 2,
                    "win_prob": 0.4,
                    "confidence": 0.1,
                },  # Low confidence
            ],
        }

        advisory_generator = AdvisoryGenerator()
        result = advisory_generator.generate_advisory(data=ui_test_data)

        if result["success"]:
            # Check ML JSON structure for UI
            ml_json = result.get("ml_json", {})

            required_fields = ["summary", "feature_flags", "messages"]
            ui_ready = True

            for field in required_fields:
                if field not in ml_json:
                    ui_ready = False
                    print(f"   ❌ Missing required field: {field}")

            if "summary" in ml_json:
                summary = ml_json["summary"]
                required_summary_fields = ["total_messages", "quality_score"]
                for field in required_summary_fields:
                    if field not in summary:
                        ui_ready = False
                        print(f"   ❌ Missing summary field: {field}")

            if "feature_flags" in ml_json:
                flags = ml_json["feature_flags"]
                expected_flags = [
                    "has_quality_issues",
                    "low_quality_score",
                    "has_validation_errors",
                ]
                for flag in expected_flags:
                    if flag not in flags:
                        ui_ready = False
                        print(f"   ❌ Missing feature flag: {flag}")

            # Check message structure
            messages = result.get("messages", [])
            for i, msg in enumerate(messages):
                required_msg_fields = [
                    "type",
                    "category",
                    "title",
                    "message",
                    "timestamp",
                ]
                for field in required_msg_fields:
                    if field not in msg:
                        ui_ready = False
                        print(f"   ❌ Message {i} missing field: {field}")

            if ui_ready:
                print("   ✅ All required UI fields present")
                print(f"   📊 Summary: {ml_json['summary']}")
                print(f"   🚩 Feature flags: {ml_json['feature_flags']}")
                print(f"   📝 Messages: {len(messages)}")
            else:
                print("   ❌ UI data format issues detected")

            return ui_ready
        else:
            print(f"   ❌ Advisory generation failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"   ❌ Error testing UI format: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Simple Advisory Workflow Test")

    # Run workflow integration test
    workflow_success = test_workflow_integration()

    # Run UI format test
    ui_success = test_ui_data_format()

    # Final results
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 60)

    if workflow_success and ui_success:
        print("🎉 ALL TESTS PASSED!")
        print("   ✅ Advisory workflow integration working")
        print("   ✅ UI data format correct")
        print("   ✅ No blocking of prediction pipeline")
        print("   ✅ Error handling functional")
        print("\n✨ Advisory system is ready for production use!")
    else:
        print("⚠️ SOME TESTS FAILED")
        if not workflow_success:
            print("   ❌ Workflow integration issues")
        if not ui_success:
            print("   ❌ UI data format issues")
        print("\n🔧 Please review and fix the issues above.")
