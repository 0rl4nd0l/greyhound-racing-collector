import os
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

# Get audit timestamp from environment
audit_ts = os.environ.get('AUDIT_TS', '20250803T104852Z')

# Configure Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--window-size=1920,1080')
chrome_options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

# Create a new Chrome session
service = Service('/usr/local/bin/chromedriver')
browser = webdriver.Chrome(service=service, options=chrome_options)

try:
    print("Opening Flask app at http://localhost:5002/predict")
    browser.get('http://localhost:5002/predict')
    
    # Take initial screenshot
    browser.save_screenshot(f'audit_results/{audit_ts}/ui/ui_initial.png')
    print("Initial screenshot saved")
    
    # Wait for the page to load and find the multi-select dropdown
    wait = WebDriverWait(browser, 15)
    race_select = wait.until(EC.presence_of_element_located((By.ID, 'race_files')))
    print("Race dropdown found")
    
    # Select the test race
    select = Select(race_select)
    select.select_by_value('test_race.csv')
    print("Selected test_race.csv")
    
    # Find and click the Predict Single button
    predict_button = browser.find_element(By.XPATH, "//button[@value='single']")
    predict_button.click()
    print("Clicked Predict Single button")
    
    # Wait for response (could be redirected to results page)
    time.sleep(5)
    
    # Take screenshot after prediction
    browser.save_screenshot(f'audit_results/{audit_ts}/ui/ui_roundtrip.png')
    print("Final screenshot saved")
    
    # Try to get the response data
    page_source = browser.page_source
    
    # Create a basic response JSON file
    response_data = {
        "test_completed": True,
        "page_title": browser.title,
        "current_url": browser.current_url,
        "status": "completed"
    }
    
    # Try to find result elements in the page
    try:
        result_element = browser.find_element(By.CLASS_NAME, 'result')
        json_text = result_element.text.strip()
        try:
            json_data = json.loads(json_text)
            response_data['prediction_result'] = json_data
            if 'calibrated_win_prob' in json_data:
                print("✓ Found calibrated_win_prob in response!")
        except json.JSONDecodeError:
            response_data['raw_response'] = json_text
            print(f"Raw response text: {json_text[:200]}...")
    except:
        print("No result element found, capturing page source instead")
        response_data['page_source_snippet'] = page_source[:1000]
    
    # Save response data
    with open(f'audit_results/{audit_ts}/ui/ui_response.json', 'w') as f:
        json.dump(response_data, f, indent=4)
    
    print("Test completed successfully!")
    print(f"Response saved to audit_results/{audit_ts}/ui/ui_response.json")
    print(f"Screenshots saved to audit_results/{audit_ts}/ui/")
    
except Exception as e:
    print(f"An error occurred: {e}")
    # Take error screenshot
    try:
        browser.save_screenshot(f'audit_results/{audit_ts}/ui/ui_error.png')
    except:
        pass
    raise
finally:
    # Gracefully terminate
    browser.quit()
    print("Browser closed")

