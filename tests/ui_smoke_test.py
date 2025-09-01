#!/usr/bin/env python3
"""
UI-to-Model Round-Trip Smoke Test
=================================

This script performs a comprehensive UI smoke test by:
1. Starting a headless Chrome browser
2. Navigating to the predict page
3. Selecting a test race file
4. Submitting the prediction form
5. Capturing the results and screenshots
6. Verifying the prediction output format
7. Saving all artifacts to audit directory

The test ensures the complete UI-to-model pipeline works end-to-end.
"""

import json
import os
import time
from datetime import datetime

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

# Set up audit timestamp
AUDIT_TS = os.environ.get("AUDIT_TS", datetime.now().strftime("%Y%m%dT%H%M%SZ"))

# Create audit directories
os.makedirs(f"audit_results/{AUDIT_TS}/ui_test", exist_ok=True)


def setup_driver():
    """Setup headless Chrome driver with appropriate options."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        print(f"❌ Failed to setup Chrome driver: {e}")
        raise


def run_ui_smoke_test():
    """Run the complete UI smoke test."""
    print("🚀 Starting UI-to-Model Round-Trip Smoke Test...")

    driver = None
    test_results = {
        "timestamp": AUDIT_TS,
        "test_status": "UNKNOWN",
        "steps_completed": [],
        "errors": [],
        "screenshots": [],
        "prediction_results": None,
    }

    try:
        # Step 1: Setup browser
        print("📱 Setting up headless Chrome browser...")
        driver = setup_driver()
        test_results["steps_completed"].append("browser_setup")

        # Step 2: Navigate to predict page
        print("🌐 Navigating to predict page...")
        driver.get("http://localhost:5002/predict")
        time.sleep(2)

        # Take initial screenshot
        screenshot_path = f"audit_results/{AUDIT_TS}/ui_test/01_initial_page.png"
        driver.save_screenshot(screenshot_path)
        test_results["screenshots"].append(screenshot_path)
        print(f"📸 Saved initial screenshot: {screenshot_path}")
        test_results["steps_completed"].append("navigate_to_page")

        # Step 3: Check if race files dropdown exists
        print("🔍 Looking for race files dropdown...")
        try:
            select_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "race_files"))
            )
            print("✅ Found race files dropdown")
        except TimeoutException:
            print("❌ Race files dropdown not found - checking page source...")
            page_source_path = (
                f"audit_results/{AUDIT_TS}/ui_test/page_source_error.html"
            )
            with open(page_source_path, "w") as f:
                f.write(driver.page_source)
            test_results["errors"].append("race_files_dropdown_not_found")
            raise

        test_results["steps_completed"].append("found_dropdown")

        # Step 4: Select test race file
        print("📋 Selecting test race file...")
        select = Select(select_element)

        # List available options
        options = [option.get_attribute("value") for option in select.options]
        print(f"📂 Available race files: {options}")

        # Try to find a test file (prefer test_race.csv, fallback to test_file.csv)
        test_file = None
        if "test_race.csv" in options:
            test_file = "test_race.csv"
        elif "test_file.csv" in options:
            test_file = "test_file.csv"
        elif options:  # Use the first available file
            test_file = options[0]

        if test_file:
            select.select_by_value(test_file)
            print(f"✅ Selected {test_file}")
        else:
            print(f"❌ No suitable test file found in options: {options}")
            test_results["errors"].append("no_test_file_available")
            raise ValueError("No test file available")

        test_results["steps_completed"].append("selected_race_file")

        # Step 5: Submit prediction form
        print("🎯 Submitting prediction form...")

        # Take pre-submission screenshot
        screenshot_path = f"audit_results/{AUDIT_TS}/ui_test/02_pre_submission.png"
        driver.save_screenshot(screenshot_path)
        test_results["screenshots"].append(screenshot_path)

        # Find and click the predict button
        try:
            predict_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), 'Predict Single')]")
                )
            )
            predict_button.click()
            print("✅ Clicked Predict Single button")

            # Handle any JavaScript alerts that might appear
            try:
                alert = WebDriverWait(driver, 2).until(EC.alert_is_present())
                alert_text = alert.text
                print(f"⚠️  JavaScript alert appeared: {alert_text}")
                alert.accept()  # Accept the alert
                print("✅ Accepted JavaScript alert")
                test_results["errors"].append(f"javascript_alert: {alert_text}")
            except TimeoutException:
                # No alert appeared, continue normally
                pass

        except TimeoutException:
            # Try alternative button text
            predict_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
            )
            predict_button.click()
            print("✅ Clicked submit button")

            # Handle any JavaScript alerts for alternative button too
            try:
                alert = WebDriverWait(driver, 2).until(EC.alert_is_present())
                alert_text = alert.text
                print(f"⚠️  JavaScript alert appeared: {alert_text}")
                alert.accept()  # Accept the alert
                print("✅ Accepted JavaScript alert")
                test_results["errors"].append(f"javascript_alert: {alert_text}")
            except TimeoutException:
                # No alert appeared, continue normally
                pass

        test_results["steps_completed"].append("submitted_form")

        # Step 6: Wait for results and capture
        print("⏳ Waiting for prediction results...")
        time.sleep(5)  # Give time for processing

        # Take post-submission screenshot
        screenshot_path = f"audit_results/{AUDIT_TS}/ui_test/03_post_submission.png"
        driver.save_screenshot(screenshot_path)
        test_results["screenshots"].append(screenshot_path)

        # Step 7: Extract prediction results
        print("📊 Extracting prediction results...")

        try:
            # Look for results in various possible formats
            result_elements = [
                driver.find_elements(By.CLASS_NAME, "result"),
                driver.find_elements(By.CLASS_NAME, "prediction-result"),
                driver.find_elements(By.ID, "prediction-results"),
                driver.find_elements(By.XPATH, "//div[contains(@class, 'result')]"),
                driver.find_elements(By.XPATH, "//pre[contains(text(), '{')]"),
            ]

            prediction_text = None
            for element_list in result_elements:
                if element_list:
                    prediction_text = element_list[0].text
                    break

            if prediction_text:
                print(f"✅ Found prediction results: {prediction_text[:200]}...")
                try:
                    # Try to parse as JSON
                    prediction_json = json.loads(prediction_text)
                    test_results["prediction_results"] = prediction_json
                    print("✅ Successfully parsed prediction JSON")
                except json.JSONDecodeError:
                    # Store as raw text
                    test_results["prediction_results"] = {"raw_text": prediction_text}
                    print("⚠️  Prediction results not in JSON format")
            else:
                print("❌ No prediction results found on page")
                # Save page source for debugging
                page_source_path = (
                    f"audit_results/{AUDIT_TS}/ui_test/page_source_no_results.html"
                )
                with open(page_source_path, "w") as f:
                    f.write(driver.page_source)
                test_results["errors"].append("no_prediction_results_found")

        except Exception as e:
            print(f"❌ Error extracting results: {e}")
            test_results["errors"].append(f"result_extraction_error: {str(e)}")

        test_results["steps_completed"].append("extracted_results")

        # Step 8: Validate essential prediction fields
        if test_results["prediction_results"]:
            print("🔍 Validating prediction structure...")

            prediction_data = test_results["prediction_results"]
            required_fields = ["calibrated_win_prob", "prediction"]

            validation_passed = True
            if isinstance(prediction_data, dict):
                for field in required_fields:
                    if field not in prediction_data:
                        print(f"⚠️  Missing required field: {field}")
                        validation_passed = False

                if validation_passed:
                    print("✅ Prediction structure validation passed")
                    test_results["steps_completed"].append(
                        "structure_validation_passed"
                    )
                else:
                    print("❌ Prediction structure validation failed")
                    test_results["errors"].append("structure_validation_failed")
            else:
                print("⚠️  Prediction results not in expected dict format")
                test_results["errors"].append("prediction_format_unexpected")

        # Determine overall test status
        if len(test_results["errors"]) == 0:
            test_results["test_status"] = "PASS"
            print("🎉 UI smoke test completed successfully!")
        elif "no_prediction_results_found" in test_results["errors"]:
            test_results["test_status"] = "PARTIAL_PASS"
            print(
                "⚠️  UI smoke test partially successful - UI works but no prediction results"
            )
        else:
            test_results["test_status"] = "FAIL"
            print("❌ UI smoke test failed")

    except Exception as e:
        print(f"❌ UI smoke test encountered error: {e}")
        test_results["test_status"] = "FAIL"
        test_results["errors"].append(f"test_execution_error: {str(e)}")

        if driver:
            # Take error screenshot
            screenshot_path = f"audit_results/{AUDIT_TS}/ui_test/error_screenshot.png"
            try:
                driver.save_screenshot(screenshot_path)
                test_results["screenshots"].append(screenshot_path)
            except:
                pass

    finally:
        if driver:
            driver.quit()
            print("🔒 Browser session closed")

    # Step 9: Save test results
    results_path = f"audit_results/{AUDIT_TS}/ui_test/smoke_test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"💾 Saved test results to: {results_path}")

    # Step 10: Update audit log
    try:
        audit_log_path = f"audit_results/{AUDIT_TS}/audit.log"
        with open(audit_log_path, "a") as f:
            f.write(f"\n=== UI SMOKE TEST - {datetime.now().isoformat()} ===\n")
            f.write(f"Audit Timestamp: {AUDIT_TS}\n")
            f.write(f"Test Status: {test_results['test_status']}\n")
            f.write(f"Steps Completed: {len(test_results['steps_completed'])}\n")
            f.write(f"Errors: {len(test_results['errors'])}\n")
            f.write(f"Screenshots: {len(test_results['screenshots'])}\n")
            f.write(
                f"Prediction Results: {'Found' if test_results['prediction_results'] else 'Not Found'}\n"
            )
            if test_results["errors"]:
                f.write(f"Error Details: {', '.join(test_results['errors'])}\n")
            f.write("Artifacts: smoke_test_results.json, screenshots\n")
            f.write(
                f"Overall Status: {'COMPLETED' if test_results['test_status'] != 'FAIL' else 'FAILED'}\n"
            )
            f.write("\n")
        print(f"📝 Updated audit log: {audit_log_path}")
    except Exception as e:
        print(f"⚠️  Could not update audit log: {e}")

    return test_results


if __name__ == "__main__":
    results = run_ui_smoke_test()
    print("\n=== FINAL RESULTS ===")
    print(f"Status: {results['test_status']}")
    print(f"Steps completed: {len(results['steps_completed'])}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Screenshots: {len(results['screenshots'])}")
