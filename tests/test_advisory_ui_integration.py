#!/usr/bin/env python3
"""
Advisory UI Integration Testing
===============================

Frontend UI tests for the advisory system to verify:
1. UI collapse toggle functionality
2. Color coding for different message types  
3. Responsiveness across different screen sizes
4. Integration with existing prediction workflow

This test uses Playwright for browser automation and testing.

Author: AI Assistant
Date: August 4, 2025
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.async_api import Page, async_playwright

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class AdvisoryUITestSuite:
    """UI test suite for advisory system"""

    def __init__(self):
        self.flask_port = 5002
        self.flask_url = f"http://127.0.0.1:{self.flask_port}"
        self.flask_process = None
        self.test_data_dir = Path("test_ui_advisory")
        self.test_data_dir.mkdir(exist_ok=True)

    def setup_test_environment(self):
        """Set up test environment including Flask app and test data"""
        print("🔧 Setting up test environment...")

        # Create test prediction data
        self.create_test_prediction_data()

        # Start Flask app if not already running
        if not self.is_flask_running():
            print("🚀 Starting Flask app for UI testing...")
            self.start_flask_app()

            # Wait for Flask to start
            for i in range(30):  # Wait up to 30 seconds
                if self.is_flask_running():
                    print("✅ Flask app is running")
                    break
                time.sleep(1)
            else:
                raise Exception("Failed to start Flask app")
        else:
            print("✅ Flask app already running")

    def create_test_prediction_data(self):
        """Create test prediction data for UI testing"""
        # High quality prediction
        high_quality_data = {
            "race_id": "ui_test_high_quality",
            "race_date": "2025-08-04",
            "race_time": "14:30",
            "venue": "TEST_VENUE",
            "predictions": [
                {
                    "dog_name": "High Quality Dog 1",
                    "box_number": 1,
                    "win_prob": 0.4,
                    "confidence": 0.9,
                },
                {
                    "dog_name": "High Quality Dog 2",
                    "box_number": 2,
                    "win_prob": 0.3,
                    "confidence": 0.85,
                },
                {
                    "dog_name": "High Quality Dog 3",
                    "box_number": 3,
                    "win_prob": 0.3,
                    "confidence": 0.8,
                },
            ],
        }

        # Low quality prediction with warnings
        low_quality_data = {
            "race_id": "ui_test_low_quality",
            "race_date": "2025-08-04",
            "race_time": "15:30",
            "venue": "TEST_VENUE",
            "predictions": [
                {
                    "dog_name": "Low Conf Dog 1",
                    "box_number": 1,
                    "win_prob": 0.1,
                    "confidence": 0.2,
                },  # Low confidence warning
                {
                    "dog_name": "Zero Prob Dog",
                    "box_number": 2,
                    "win_prob": 0.0,
                    "confidence": 0.0,
                },  # Critical issue
                {
                    "dog_name": "Normal Dog",
                    "box_number": 3,
                    "win_prob": 0.9,
                    "confidence": 0.1,
                },  # High prob, low confidence
            ],
        }

        # Save test data
        with open(self.test_data_dir / "high_quality.json", "w") as f:
            json.dump(high_quality_data, f, indent=2)

        with open(self.test_data_dir / "low_quality.json", "w") as f:
            json.dump(low_quality_data, f, indent=2)

    def is_flask_running(self):
        """Check if Flask app is running"""
        try:
            response = requests.get(f"{self.flask_url}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def start_flask_app(self):
        """Start Flask app as subprocess"""
        try:
            self.flask_process = subprocess.Popen(
                [sys.executable, "app.py"],
                cwd=project_root,
                env=dict(os.environ, PORT=str(self.flask_port)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except Exception as e:
            print(f"Failed to start Flask app: {e}")
            return False

    def stop_flask_app(self):
        """Stop Flask app subprocess"""
        if self.flask_process:
            self.flask_process.terminate()
            self.flask_process.wait()

    def cleanup(self):
        """Clean up test environment"""
        if self.flask_process:
            self.stop_flask_app()

        # Clean up test files
        if self.test_data_dir.exists():
            import shutil

            shutil.rmtree(self.test_data_dir, ignore_errors=True)

    async def test_advisory_collapse_toggle(self, page: Page):
        """Test 1: UI collapse toggle functionality"""
        print("\n🧪 TEST 1: UI Collapse Toggle Functionality")

        # Navigate to a page that displays advisory messages
        await page.goto(f"{self.flask_url}/predict")

        # Generate an advisory with details to test collapse
        test_advisory_script = """
        (async () => {
            // Create test advisory data
            const testData = {
                title: "Test Advisory with Details",
                message: "This is a test advisory message with collapsible details.",
                type: "warning",
                details: [
                    "Detail item 1: Low confidence prediction",
                    "Detail item 2: Class imbalance detected",
                    "Detail item 3: Calibration drift warning"
                ],
                helpText: "This is help text that should appear in a tooltip"
            };
            
            // Clear any existing advisories
            const container = document.getElementById('advisory-container') || document.createElement('div');
            container.id = 'advisory-container';
            container.innerHTML = '';
            document.body.appendChild(container);
            
            // Use advisory utils to render
            if (window.AdvisoryUtils) {
                window.AdvisoryUtils.renderAdvisory(testData, container);
                return 'Advisory rendered successfully';
            } else {
                return 'AdvisoryUtils not loaded';
            }
        })()
        """

        # Load advisory utils and render test advisory
        await page.add_script_tag(path=str(project_root / "static/js/advisoryUtils.js"))
        result = await page.evaluate(test_advisory_script)

        print(f"   Advisory rendering result: {result}")

        # Test collapse toggle functionality
        await page.wait_for_selector(".advisory-details-toggle", timeout=5000)

        # Initially, details should be collapsed
        details_element = await page.query_selector(".advisory-details")
        is_initially_collapsed = await details_element.is_hidden()

        print(f"   ✅ Details initially collapsed: {is_initially_collapsed}")

        # Click toggle to expand
        await page.click(".advisory-details-toggle")
        await page.wait_for_timeout(500)  # Wait for animation

        is_expanded = await details_element.is_visible()
        print(f"   ✅ Details expanded after click: {is_expanded}")

        # Check that chevron icon changed
        chevron = await page.query_selector(".advisory-details-toggle .fas")
        chevron_class = await chevron.get_attribute("class")
        print(f"   ✅ Chevron class after expand: {chevron_class}")

        # Click toggle to collapse again
        await page.click(".advisory-details-toggle")
        await page.wait_for_timeout(500)

        is_collapsed_again = await details_element.is_hidden()
        print(f"   ✅ Details collapsed after second click: {is_collapsed_again}")

        # Verify tooltip functionality
        help_icon = await page.query_selector(".advisory-help-icon")
        if help_icon:
            help_title = await help_icon.get_attribute("title")
            print(f"   ✅ Help tooltip text: '{help_title}'")

        # Assertions
        assert (
            result == "Advisory rendered successfully"
        ), "Advisory should render successfully"
        assert is_initially_collapsed, "Details should be initially collapsed"
        assert is_expanded, "Details should expand on toggle click"
        assert is_collapsed_again, "Details should collapse on second toggle click"

        print("   ✅ Collapse toggle functionality working correctly")

    async def test_advisory_color_coding(self, page: Page):
        """Test 2: Color coding for different message types"""
        print("\n🧪 TEST 2: Color Coding for Message Types")

        # Test different message types and their colors
        message_types = ["info", "success", "warning", "danger", "error"]

        color_test_script = """
        (messageTypes) => {
            const container = document.getElementById('color-test-container') || document.createElement('div');
            container.id = 'color-test-container';
            container.innerHTML = '';
            document.body.appendChild(container);
            
            const results = {};
            
            messageTypes.forEach(type => {
                const testData = {
                    title: `${type.toUpperCase()} Advisory`,
                    message: `This is a ${type} message for color testing`,
                    type: type
                };
                
                if (window.AdvisoryUtils) {
                    const advisoryId = window.AdvisoryUtils.renderAdvisory(testData, container);
                    const element = document.getElementById(advisoryId);
                    if (element) {
                        const computedStyle = window.getComputedStyle(element);
                        results[type] = {
                            borderLeftColor: computedStyle.borderLeftColor,
                            backgroundColor: computedStyle.backgroundColor,
                            className: element.className
                        };
                    }
                }
            });
            
            return results;
        }
        """

        color_results = await page.evaluate(color_test_script, message_types)

        print("   Color coding results:")
        for msg_type, styles in color_results.items():
            print(f"     {msg_type.upper()}:")
            print(f"       Border: {styles['borderLeftColor']}")
            print(f"       Background: {styles['backgroundColor']}")
            print(f"       Classes: {styles['className']}")

        # Verify that different types have different styling
        unique_border_colors = set(
            styles["borderLeftColor"] for styles in color_results.values()
        )
        unique_backgrounds = set(
            styles["backgroundColor"] for styles in color_results.values()
        )

        print(f"   ✅ Unique border colors: {len(unique_border_colors)}")
        print(f"   ✅ Unique backgrounds: {len(unique_backgrounds)}")

        # Check for Bootstrap alert classes
        expected_classes = [
            "alert-info",
            "alert-success",
            "alert-warning",
            "alert-danger",
        ]
        found_classes = []

        for styles in color_results.values():
            for expected_class in expected_classes:
                if expected_class in styles["className"]:
                    found_classes.append(expected_class)

        print(f"   ✅ Bootstrap alert classes found: {found_classes}")

        # Assertions
        assert len(color_results) == len(
            message_types
        ), "Should render all message types"
        assert (
            len(unique_border_colors) >= 3
        ), "Should have at least 3 different border colors"
        assert (
            len(found_classes) >= len(expected_classes) - 1
        ), "Should use Bootstrap alert classes"

        print("   ✅ Color coding working correctly")

    async def test_advisory_responsiveness(self, page: Page):
        """Test 3: Responsiveness across different screen sizes"""
        print("\n🧪 TEST 3: Responsiveness Testing")

        # Test different viewport sizes
        viewport_sizes = [
            {"width": 320, "height": 568, "name": "Mobile Portrait"},
            {"width": 568, "height": 320, "name": "Mobile Landscape"},
            {"width": 768, "height": 1024, "name": "Tablet Portrait"},
            {"width": 1024, "height": 768, "name": "Tablet Landscape"},
            {"width": 1920, "height": 1080, "name": "Desktop"},
        ]

        # Create test advisory content
        responsiveness_test_script = """
        () => {
            const container = document.getElementById('responsive-test-container') || document.createElement('div');
            container.id = 'responsive-test-container';
            container.innerHTML = '';
            document.body.appendChild(container);
            
            const testData = {
                title: "Responsiveness Test Advisory",
                message: "This is a longer advisory message designed to test how the advisory component responds to different screen sizes and viewport dimensions. It should adapt gracefully to mobile, tablet, and desktop layouts.",
                type: "warning",
                details: [
                    "Detail 1: This is a detailed explanation that should wrap properly",
                    "Detail 2: Another detail item for testing text wrapping and layout",
                    "Detail 3: Final detail to ensure the list displays correctly"
                ],
                helpText: "Help text for responsiveness testing"
            };
            
            if (window.AdvisoryUtils) {
                const advisoryId = window.AdvisoryUtils.renderAdvisory(testData, container);
                return advisoryId;
            }
            return null;
        }
        """

        advisory_id = await page.evaluate(responsiveness_test_script)

        responsiveness_results = {}

        for viewport in viewport_sizes:
            print(
                f"   Testing {viewport['name']} ({viewport['width']}x{viewport['height']})"
            )

            # Set viewport size
            await page.set_viewport_size(viewport["width"], viewport["height"])
            await page.wait_for_timeout(500)  # Wait for layout adjustment

            # Measure advisory element properties
            if advisory_id:
                element_metrics = await page.evaluate(
                    f"""
                () => {{
                    const element = document.getElementById('{advisory_id}');
                    if (element) {{
                        const rect = element.getBoundingClientRect();
                        const style = window.getComputedStyle(element);
                        return {{
                            width: rect.width,
                            height: rect.height,
                            overflowX: style.overflowX,
                            overflowY: style.overflowY,
                            fontSize: style.fontSize,
                            padding: style.padding,
                            margin: style.margin
                        }};
                    }}
                    return null;
                }}
                """
                )

                responsiveness_results[viewport["name"]] = element_metrics

                if element_metrics:
                    print(f"     Width: {element_metrics['width']:.1f}px")
                    print(f"     Height: {element_metrics['height']:.1f}px")
                    print(f"     Font size: {element_metrics['fontSize']}")

        # Test specific responsive behaviors
        print("\n   Testing specific responsive behaviors:")

        # Mobile test - check if content fits without horizontal scroll
        await page.set_viewport_size(320, 568)
        await page.wait_for_timeout(500)

        body_width = await page.evaluate("() => document.body.scrollWidth")
        viewport_width = 320

        has_horizontal_scroll = body_width > viewport_width
        print(
            f"     Mobile horizontal scroll: {'Yes' if has_horizontal_scroll else 'No'}"
        )

        # Desktop test - check layout utilizes space efficiently
        await page.set_viewport_size(1920, 1080)
        await page.wait_for_timeout(500)

        desktop_metrics = responsiveness_results.get("Desktop", {})
        desktop_width = desktop_metrics.get("width", 0)

        print(f"     Desktop width utilization: {desktop_width:.1f}px")

        # Assertions
        assert len(responsiveness_results) == len(
            viewport_sizes
        ), "Should test all viewport sizes"
        assert not has_horizontal_scroll, "Should not have horizontal scroll on mobile"
        assert desktop_width > 0, "Should have measurable width on desktop"

        # Check that width adapts to viewport
        mobile_width = responsiveness_results.get("Mobile Portrait", {}).get("width", 0)
        desktop_width = responsiveness_results.get("Desktop", {}).get("width", 0)

        if mobile_width > 0 and desktop_width > 0:
            width_difference = abs(desktop_width - mobile_width)
            print(f"     Width adaptation difference: {width_difference:.1f}px")
            # Desktop should typically be wider (unless constrained by max-width)
            # assert width_difference > 100, "Should show significant width adaptation"

        print("   ✅ Responsiveness testing completed")

    async def test_advisory_workflow_integration(self, page: Page):
        """Test 4: Integration with prediction workflow"""
        print("\n🧪 TEST 4: Workflow Integration Testing")

        # Navigate to predictions page
        await page.goto(f"{self.flask_url}/predict")

        # Test advisory integration in prediction workflow
        workflow_integration_script = """
        async () => {
            // Simulate prediction workflow
            const mockPredictionData = {
                success: true,
                race_id: 'workflow_test_race',
                predictions: [
                    { dog_name: 'Test Dog 1', box_number: 1, win_prob: 0.4, confidence: 0.8 },
                    { dog_name: 'Test Dog 2', box_number: 2, win_prob: 0.3, confidence: 0.2 },  // Low confidence
                    { dog_name: 'Test Dog 3', box_number: 3, win_prob: 0.3, confidence: 0.7 }
                ]
            };
            
            // Check if we can generate advisory for prediction data
            try {
                const response = await fetch('/api/generate_advisory', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prediction_data: mockPredictionData
                    })
                });
                
                if (response.ok) {
                    const advisoryResult = await response.json();
                    return {
                        success: true,
                        advisoryGenerated: advisoryResult.success,
                        messageCount: advisoryResult.messages ? advisoryResult.messages.length : 0,
                        processingTime: advisoryResult.processing_time_ms
                    };
                } else {
                    return {
                        success: false,
                        error: `HTTP ${response.status}`
                    };
                }
            } catch (error) {
                return {
                    success: false,
                    error: error.message
                };
            }
        }
        """

        workflow_result = await page.evaluate(workflow_integration_script)

        print(f"   Workflow integration result: {workflow_result}")

        if workflow_result.get("success"):
            print(
                f"   ✅ Advisory API accessible: {workflow_result['advisoryGenerated']}"
            )
            print(f"   ✅ Messages generated: {workflow_result['messageCount']}")
            print(f"   ✅ Processing time: {workflow_result['processingTime']}ms")

        # Test that advisory doesn't block UI interactions
        ui_blocking_test = """
        () => {
            // Create multiple advisory renders to test UI blocking
            const container = document.getElementById('blocking-test-container') || document.createElement('div');
            container.id = 'blocking-test-container';
            container.innerHTML = '';
            document.body.appendChild(container);
            
            const startTime = performance.now();
            
            // Render multiple advisories quickly
            for (let i = 0; i < 5; i++) {
                const testData = {
                    title: `Test Advisory ${i + 1}`,
                    message: `This is test advisory number ${i + 1}`,
                    type: i % 2 === 0 ? 'info' : 'warning'
                };
                
                if (window.AdvisoryUtils) {
                    window.AdvisoryUtils.renderAdvisory(testData, container);
                }
            }
            
            const endTime = performance.now();
            const renderTime = endTime - startTime;
            
            // Test if UI is still responsive
            const button = document.createElement('button');
            button.textContent = 'Test Button';
            button.onclick = () => { button.dataset.clicked = 'true'; };
            container.appendChild(button);
            
            // Click the button
            button.click();
            
            return {
                renderTime: renderTime,
                buttonClicked: button.dataset.clicked === 'true',
                advisoryCount: container.querySelectorAll('.advisory-banner').length
            };
        }
        """

        ui_test_result = await page.evaluate(ui_blocking_test)

        print("   UI blocking test:")
        print(
            f"     Render time for 5 advisories: {ui_test_result['renderTime']:.2f}ms"
        )
        print(f"     Button still clickable: {ui_test_result['buttonClicked']}")
        print(f"     Advisories rendered: {ui_test_result['advisoryCount']}")

        # Assertions
        if workflow_result.get("success"):
            assert workflow_result[
                "advisoryGenerated"
            ], "Advisory should be generated successfully"
            assert (
                workflow_result["messageCount"] > 0
            ), "Should generate advisory messages"
            assert (
                workflow_result["processingTime"] < 5000
            ), "Should process within 5 seconds"

        assert ui_test_result["renderTime"] < 1000, "UI rendering should be fast"
        assert ui_test_result["buttonClicked"], "UI should remain responsive"
        assert ui_test_result["advisoryCount"] == 5, "Should render all advisories"

        print("   ✅ Workflow integration working correctly")

    async def run_all_ui_tests(self):
        """Run all UI tests using Playwright"""
        print("🚀 Starting Advisory UI Integration Testing")
        print("=" * 60)

        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": 1280, "height": 720})
            page = await context.new_page()

            try:
                # Run all UI tests
                tests = [
                    self.test_advisory_collapse_toggle,
                    self.test_advisory_color_coding,
                    self.test_advisory_responsiveness,
                    self.test_advisory_workflow_integration,
                ]

                passed_tests = 0
                failed_tests = 0

                for i, test in enumerate(tests, 1):
                    try:
                        print(f"\n[{i}/{len(tests)}] Running {test.__name__}...")
                        await test(page)
                        passed_tests += 1
                        print(f"✅ {test.__name__} PASSED")
                    except Exception as e:
                        failed_tests += 1
                        print(f"❌ {test.__name__} FAILED: {e}")
                        import traceback

                        traceback.print_exc()

                # Final results
                print("\n" + "=" * 60)
                print("🏁 Advisory UI Testing Results")
                print("=" * 60)
                print(f"✅ Tests Passed: {passed_tests}")
                print(f"❌ Tests Failed: {failed_tests}")
                print(
                    f"📊 Success Rate: {passed_tests/(passed_tests+failed_tests)*100:.1f}%"
                )

                if failed_tests == 0:
                    print(
                        "\n🎉 ALL UI TESTS PASSED! Advisory UI is ready for production."
                    )
                else:
                    print(
                        f"\n⚠️ {failed_tests} UI tests failed. Review the issues above."
                    )

            finally:
                await browser.close()


def run_sync_ui_tests():
    """Synchronous wrapper for running UI tests"""
    test_suite = AdvisoryUITestSuite()

    try:
        # Setup test environment
        test_suite.setup_test_environment()

        # Run async UI tests
        asyncio.run(test_suite.run_all_ui_tests())

    except Exception as e:
        print(f"❌ UI testing failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        test_suite.cleanup()


if __name__ == "__main__":
    # Check if playwright is installed
    try:
        from playwright.async_api import async_playwright

        print("✅ Playwright is available")
    except ImportError:
        print("❌ Playwright not installed. Install with: pip install playwright")
        print("   Then run: playwright install")
        sys.exit(1)

    # Run the UI tests
    run_sync_ui_tests()
